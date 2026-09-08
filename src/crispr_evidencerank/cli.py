"""Command-line interface for validation, featurization, and benchmarking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
from datetime import UTC, date, datetime
from importlib import metadata as importlib_metadata
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import get_args

import pandas as pd

from . import __version__
from .baselines import (
    adapt_direction_aligned_drugz_baseline,
    adapt_repo_native_mageck_baseline,
    evaluate_fixed_score_baselines,
)
from .clinical_context import ClinicalContextResult, summarize_clinical_context
from .clinicaltrials_gov import (
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_CANDIDATE_ROWS,
    DEFAULT_MAX_DERIVED_BYTES,
    DEFAULT_MAX_ELAPSED_SECONDS,
    DEFAULT_MAX_PAGE_BYTES,
    DEFAULT_MAX_PAGES,
    DEFAULT_MAX_STUDIES,
    DEFAULT_MAX_TOTAL_BYTES,
    DEFAULT_PAGE_SIZE,
    DEFAULT_TIMEOUT_SECONDS,
    fetch_clinicaltrials_gov_snapshot,
    verify_clinicaltrials_gov_snapshot,
)
from .contracts import (
    CONTRACTS,
    ClinicalTrialEvidenceRecord,
    DataAssetRecord,
    ImmuneScreenEvidenceRecord,
    PerturbationModality,
    validate_records,
)
from .curation import (
    write_completed_dual_review_bundle,
    write_curation_batch,
    write_dual_review_bundle,
)
from .evaluation import (
    paired_cluster_bootstrap_difference,
    reliability_table,
    study_cluster_bootstrap_intervals,
    within_query_feature_invariance,
)
from .event_adjudication import (
    write_event_review_comparison_bundle,
    write_event_review_packet,
    write_final_adjudication_bundle,
)
from .features import featurize_count_table, featurize_experimental_design
from .immuno_context import ImmunoContextResult, summarize_immuno_context
from .intake import SUPPORTED_POLICY_VERSIONS, triage_orcs_index
from .io import read_table
from .model_artifact import MANIFEST_FILENAME, freeze_model_bundle, verify_model_bundle
from .modeling import (
    DEFAULT_FEATURE_COLUMNS,
    FEATURE_PROFILES,
    grouped_oof_predictions,
)
from .orcs import parse_orcs_index, parse_orcs_screen_scores
from .orcs_prepare import prepare_orcs_release
from .orcs_release import load_orcs_release_spec
from .screen_report import ScreenReportResult, rank_screen
from .workflow_cli import add_workflow_parsers


def _write_frame(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() in {".tsv", ".txt"}:
        frame.to_csv(path, sep="\t", index=False)
    else:
        raise ValueError("output must be CSV or TSV")


def _contract_string_fields(contract: type) -> dict[str, str]:
    return {
        name: "string"
        for name, field in contract.model_fields.items()
        if field.annotation is str or str in get_args(field.annotation)
    }


def _read_contract_table(path: str | Path, contract: type) -> pd.DataFrame:
    return read_table(path, dtype=_contract_string_fields(contract))


def _read_table_snapshot(
    path: str | Path,
    *,
    dtype: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = Path(path)
    content = path.read_bytes()
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(BytesIO(content), dtype=dtype)
    elif path.suffix.lower() in {".tsv", ".txt"}:
        frame = pd.read_csv(BytesIO(content), sep="\t", dtype=dtype)
    else:
        raise ValueError(f"unsupported table extension: {path.suffix}")
    return frame, {
        "path": str(path),
        "filename": path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_size": len(content),
    }


def _assert_input_snapshots_unchanged(
    snapshots: dict[str, dict[str, object]],
) -> None:
    changed = sorted(
        role
        for role, snapshot in snapshots.items()
        if _file_sha256(str(snapshot["path"])) != snapshot["sha256"]
    )
    if changed:
        raise ValueError(f"input files changed during the run: {changed}")


def command_validate(args: argparse.Namespace) -> int:
    model = CONTRACTS[args.contract]
    frame = _read_contract_table(args.table, model)
    valid, errors = validate_records(frame, args.contract)
    report = {
        "contract": args.contract,
        "records": len(frame),
        "valid_records": len(valid),
        "errors": errors.to_dict(orient="records"),
    }
    print(json.dumps(report, indent=2))
    return 1 if not errors.empty else 0


def command_featurize(args: argparse.Namespace) -> int:
    counts = read_table(args.counts)
    samples = read_table(args.samples)
    features = featurize_count_table(
        counts,
        samples,
        positive_lfc_means=args.positive_lfc_means,
        pseudocount=args.pseudocount,
        low_count_threshold=args.low_count_threshold,
        normalization_method=args.normalization_method,
        direction_deadband=args.direction_deadband,
    )
    _write_frame(features, args.output)
    print(json.dumps({"output": str(args.output), "rows": len(features)}, indent=2))
    return 0


def command_featurize_design(args: argparse.Namespace) -> int:
    features = featurize_experimental_design(
        read_table(args.screens),
        read_table(args.screen_designs),
        read_table(args.contrasts),
        read_table(args.samples),
    )
    _write_frame(features, args.output)
    print(json.dumps({"output": str(args.output), "rows": len(features)}, indent=2))
    return 0


def _write_immuno_context_bundle(
    result: ImmunoContextResult,
    output_dir: str | Path,
    *,
    input_paths: tuple[str | Path, ...],
) -> None:
    output_dir = Path(output_dir)
    output_paths = {
        output_dir / "immune_context.tsv",
        output_dir / "immune_context_exclusions.tsv",
        output_dir / "immune_context_used_evidence.tsv",
        output_dir / "rank_list_audit.tsv",
        output_dir / "summary.json",
    }
    resolved_inputs = {Path(path).resolve() for path in input_paths}
    if any(path.resolve() in resolved_inputs for path in output_paths):
        raise ValueError("immune-context outputs cannot overwrite an input file")
    if output_dir.exists():
        raise FileExistsError(f"immune-context output directory exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        _write_frame(result.summary, staging / "immune_context.tsv")
        _write_frame(
            result.exclusions,
            staging / "immune_context_exclusions.tsv",
        )
        _write_frame(
            result.used_evidence,
            staging / "immune_context_used_evidence.tsv",
        )
        _write_frame(result.rank_list_audit, staging / "rank_list_audit.tsv")
        result.metadata["outputs"] = {
            name: {
                "filename": filename,
                "sha256": _file_sha256(staging / filename),
            }
            for name, filename in (
                ("summary", "immune_context.tsv"),
                ("exclusions", "immune_context_exclusions.tsv"),
                ("used_evidence", "immune_context_used_evidence.tsv"),
                ("rank_list_audit", "rank_list_audit.tsv"),
            )
        }
        (staging / "summary.json").write_text(
            json.dumps(result.metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def command_summarize_immuno_context(args: argparse.Namespace) -> int:
    evidence, evidence_snapshot = _read_table_snapshot(
        args.evidence,
        dtype=_contract_string_fields(ImmuneScreenEvidenceRecord),
    )
    candidate_string_columns = {
        args.gene_column: "string",
        "screen_id": "string",
        "contrast_id": "string",
        "phenotype_direction": "string",
        "analysis_tail": "string",
        "screen_signal_percentile_scope": "string",
        "ranking_type": "string",
    }
    candidates, candidates_snapshot = _read_table_snapshot(
        args.candidates,
        dtype=candidate_string_columns,
    )
    result = summarize_immuno_context(
        evidence,
        candidates,
        cutoff_date=args.cutoff_date,
        target_modality=args.target_modality,
        candidate_gene_column=args.gene_column,
        excluded_source_families=args.exclude_source_family,
        excluded_raw_data_families=args.exclude_raw_data_family,
        recurrence_stratum_id=args.recurrence_stratum_id,
        dual_action_group_id=args.dual_action_group_id,
        dual_action_group_version=args.dual_action_group_version,
        max_source_fdr=args.max_source_fdr,
        target_absence_attested=args.target_not_in_compendium,
    )
    result.metadata.update(
        {
            "package_version": __version__,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "inputs": {
                "evidence": evidence_snapshot,
                "candidates": candidates_snapshot,
            },
        }
    )
    _assert_input_snapshots_unchanged(result.metadata["inputs"])
    _write_immuno_context_bundle(
        result,
        args.output_dir,
        input_paths=(args.evidence, args.candidates),
    )
    print(json.dumps(result.metadata, indent=2, sort_keys=True))
    return 0


def _write_clinical_context_bundle(
    result: ClinicalContextResult,
    output_dir: str | Path,
    *,
    input_paths: tuple[str | Path, ...],
    input_snapshots: dict[str, dict[str, object]],
) -> None:
    output_dir = Path(output_dir)
    output_names = (
        "clinical_context.tsv",
        "clinical_context_studies.tsv",
        "clinical_context_exclusions.tsv",
        "clinical_context_used_assets.tsv",
        "summary.json",
    )
    resolved_inputs = {Path(path).resolve() for path in input_paths}
    if any((output_dir / name).resolve() in resolved_inputs for name in output_names):
        raise ValueError("clinical-context outputs cannot overwrite an input file")
    if output_dir.exists():
        raise FileExistsError(f"clinical-context output directory exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        _write_frame(result.summary, staging / "clinical_context.tsv")
        _write_frame(result.studies, staging / "clinical_context_studies.tsv")
        _write_frame(
            result.exclusions,
            staging / "clinical_context_exclusions.tsv",
        )
        _write_frame(
            result.used_assets,
            staging / "clinical_context_used_assets.tsv",
        )
        result.metadata["outputs"] = {
            name: {
                "filename": filename,
                "sha256": _file_sha256(staging / filename),
            }
            for name, filename in (
                ("summary", "clinical_context.tsv"),
                ("studies", "clinical_context_studies.tsv"),
                ("exclusions", "clinical_context_exclusions.tsv"),
                ("used_assets", "clinical_context_used_assets.tsv"),
            )
        }
        (staging / "summary.json").write_text(
            json.dumps(result.metadata, allow_nan=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _assert_input_snapshots_unchanged(input_snapshots)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _clinical_software_provenance() -> dict[str, object]:
    metadata_bytes = (
        resources.files("crispr_evidencerank")
        .joinpath("_build_metadata.json")
        .read_bytes()
    )
    build_metadata = json.loads(metadata_bytes)
    required_hashes = (
        "dependency_lock_sha256",
        "clinical_schema_sha256",
        "clinicaltrials_gov_curation_candidate_schema_sha256",
        "clinicaltrials_gov_snapshot_manifest_schema_sha256",
        "clinicaltrials_gov_study_inventory_schema_sha256",
    )
    for name in required_hashes:
        value = build_metadata.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"invalid packaged build metadata field: {name}")

    runtime_dependencies = {}
    for distribution in ("numpy", "pandas", "pydantic", "PyYAML", "scikit-learn"):
        runtime_dependencies[distribution] = importlib_metadata.version(distribution)
    provenance: dict[str, object] = {
        "package_version": __version__,
        "python_version": platform.python_version(),
        "clinical_contract": "clinical_trial_evidence",
        "build_metadata_format_version": build_metadata.get("format_version"),
        "build_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "dependency_lock_sha256": build_metadata["dependency_lock_sha256"],
        "clinical_schema_sha256": build_metadata["clinical_schema_sha256"],
        "clinicaltrials_gov_curation_candidate_schema_sha256": build_metadata[
            "clinicaltrials_gov_curation_candidate_schema_sha256"
        ],
        "clinicaltrials_gov_snapshot_manifest_schema_sha256": build_metadata[
            "clinicaltrials_gov_snapshot_manifest_schema_sha256"
        ],
        "clinicaltrials_gov_study_inventory_schema_sha256": build_metadata[
            "clinicaltrials_gov_study_inventory_schema_sha256"
        ],
        "runtime_dependencies": runtime_dependencies,
    }
    provenance["build_revision"] = os.environ.get("CRISPR_EVIDENCERANK_BUILD_REVISION")
    return provenance


def _benchmark_software_provenance() -> dict[str, object]:
    packaged = _clinical_software_provenance()
    return {
        "package_version": packaged["package_version"],
        "python_version": packaged["python_version"],
        "runtime_dependencies": packaged["runtime_dependencies"],
        "build_metadata_sha256": packaged["build_metadata_sha256"],
        "dependency_lock_sha256": packaged["dependency_lock_sha256"],
        "build_revision": packaged["build_revision"],
    }


def _write_benchmark_artifacts(
    predictions: pd.DataFrame,
    report: dict[str, object],
    *,
    prediction_path: Path,
    metrics_path: Path | None,
    input_snapshot: dict[str, object],
) -> str:
    """Stage and no-clobber publish benchmark artifacts after input recheck."""

    targets = [prediction_path] + ([metrics_path] if metrics_path is not None else [])
    staged: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{prediction_path.stem}.staging-",
            suffix=prediction_path.suffix,
            dir=prediction_path.parent,
        )
        os.close(descriptor)
        staged_predictions = Path(temporary_name)
        _write_frame(predictions, staged_predictions)
        staged.append((staged_predictions, prediction_path))

        report["inputs"] = {
            "training_table": {
                "filename": input_snapshot["filename"],
                "sha256": input_snapshot["sha256"],
                "byte_size": input_snapshot["byte_size"],
            }
        }
        report["software_provenance"] = _benchmark_software_provenance()
        report["outputs"] = {
            "predictions": {
                "filename": prediction_path.name,
                "sha256": _file_sha256(staged_predictions),
                "byte_size": staged_predictions.stat().st_size,
                "row_count": len(predictions),
            },
            "metrics": (
                {"filename": metrics_path.name, "self_hash_embedded": False}
                if metrics_path is not None
                else {"destination": "stdout", "self_hash_embedded": False}
            ),
        }
        encoded = json.dumps(
            _replace_nonfinite_with_none(report),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        if metrics_path is not None:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{metrics_path.stem}.staging-",
                suffix=metrics_path.suffix,
                dir=metrics_path.parent,
            )
            os.close(descriptor)
            staged_metrics = Path(temporary_name)
            staged_metrics.write_text(encoded + "\n", encoding="utf-8")
            staged.append((staged_metrics, metrics_path))

        _assert_input_snapshots_unchanged({"training_table": input_snapshot})
        for source, target in staged:
            os.link(source, target)
            published.append(target)
        return encoded
    except Exception:
        for target in published:
            target.unlink(missing_ok=True)
        raise
    finally:
        for source, _ in staged:
            source.unlink(missing_ok=True)


def command_summarize_clinical_context(args: argparse.Namespace) -> int:
    evidence, evidence_snapshot = _read_table_snapshot(
        args.evidence,
        dtype=_contract_string_fields(ClinicalTrialEvidenceRecord),
    )
    assets, assets_snapshot = _read_table_snapshot(
        args.assets,
        dtype=_contract_string_fields(DataAssetRecord),
    )
    result = summarize_clinical_context(
        evidence,
        assets,
        treatment_concept_id=args.treatment_concept_id,
        treatment_mapping_source=args.treatment_mapping_source,
        treatment_mapping_version=args.treatment_mapping_version,
        cancer_concept_id=args.cancer_concept_id,
        cancer_mapping_source=args.cancer_mapping_source,
        cancer_mapping_version=args.cancer_mapping_version,
        cutoff_date=args.cutoff_date,
        excluded_source_families=args.exclude_source_family,
    )
    result.metadata.update(
        {
            "bundle_type": "clinical_context",
            "bundle_schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "software": _clinical_software_provenance(),
            "inputs": {
                "evidence": evidence_snapshot,
                "assets": assets_snapshot,
            },
        }
    )
    _assert_input_snapshots_unchanged(result.metadata["inputs"])
    _write_clinical_context_bundle(
        result,
        args.output_dir,
        input_paths=(args.evidence, args.assets),
        input_snapshots=result.metadata["inputs"],
    )
    print(json.dumps(result.metadata, indent=2, sort_keys=True))
    return 0


def command_fetch_clinicaltrials_gov(args: argparse.Namespace) -> int:
    manifest = fetch_clinicaltrials_gov_snapshot(
        condition_query=args.condition_query,
        intervention_query=args.intervention_query,
        output_dir=args.output_dir,
        page_size=args.page_size,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
        backoff_seconds=args.backoff_seconds,
        max_pages=args.max_pages,
        max_studies=args.max_studies,
        max_candidate_rows=args.max_candidate_rows,
        max_page_bytes=args.max_page_bytes,
        max_total_bytes=args.max_total_bytes,
        max_derived_bytes=args.max_derived_bytes,
        max_elapsed_seconds=args.max_elapsed_seconds,
    )
    report = {
        "snapshot_id": manifest["snapshot_id"],
        "output_dir": str(args.output_dir),
        "api_version": manifest["source"]["api_version"],
        "data_timestamp": manifest["source"]["data_timestamp"],
        "page_count": manifest["page_count"],
        "study_count": manifest["observed_unique_study_count"],
        "complete": manifest["complete"],
        "clinical_context_eligible": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def command_verify_clinicaltrials_gov(args: argparse.Namespace) -> int:
    report = verify_clinicaltrials_gov_snapshot(args.snapshot_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_screen_report_bundle(
    result: ScreenReportResult,
    manifest: dict[str, object],
    output_dir: str | Path,
    *,
    input_paths: tuple[str | Path, ...],
) -> None:
    output_dir = Path(output_dir)
    output_names = (
        "ranked_candidates.tsv",
        "qc_summary.json",
        "run_manifest.json",
        "report.md",
    )
    resolved_inputs = {Path(path).resolve() for path in input_paths}
    if any((output_dir / name).resolve() in resolved_inputs for name in output_names):
        raise ValueError("screen-report outputs cannot overwrite an input file")
    if output_dir.exists():
        raise FileExistsError(f"screen-report output directory exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        _write_frame(result.ranked_candidates, staging / "ranked_candidates.tsv")
        (staging / "qc_summary.json").write_text(
            json.dumps(result.qc_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (staging / "report.md").write_text(
            result.report_markdown,
            encoding="utf-8",
        )
        manifest["outputs"] = {
            name: {
                "filename": filename,
                "sha256": _file_sha256(staging / filename),
            }
            for name, filename in (
                ("ranked_candidates", "ranked_candidates.tsv"),
                ("qc_summary", "qc_summary.json"),
                ("report", "report.md"),
            )
        }
        (staging / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def command_rank_screen(args: argparse.Namespace) -> int:
    snapshots: dict[str, dict[str, object]] = {}
    mageck = None
    counts = None
    samples = None
    for role, path in (
        ("mageck_summary", args.mageck_summary),
        ("counts", args.counts),
        ("samples", args.samples),
    ):
        if path is None:
            continue
        dtype = {
            "mageck_summary": {
                "id": "string",
                "gene": "string",
                "Gene": "string",
                "gene_symbol": "string",
            },
            "counts": {
                "sgrna_id": "string",
                "gene_symbol": "string",
            },
            "samples": {
                "sample_id": "string",
                "screen_id": "string",
                "contrast_id": "string",
                "condition_role": "string",
            },
        }[role]
        frame, snapshot = _read_table_snapshot(path, dtype=dtype)
        snapshots[role] = snapshot
        if role == "mageck_summary":
            mageck = frame
        elif role == "counts":
            counts = frame
        else:
            samples = frame
    screen_id = args.screen_id.strip() if args.screen_id is not None else None
    contrast_id = args.contrast_id.strip() if args.contrast_id is not None else None
    result = rank_screen(
        mageck_summary=mageck,
        counts=counts,
        samples=samples,
        screen_id=screen_id,
        contrast_id=contrast_id,
        positive_tail_means=args.positive_tail_means,
        positive_lfc_means=args.positive_lfc_means,
        pseudocount=args.pseudocount,
        low_count_threshold=args.low_count_threshold,
        normalization_method=args.normalization_method,
        direction_deadband=args.direction_deadband,
        fdr_threshold=args.fdr_threshold,
    )
    input_paths = tuple(
        path
        for path in (args.mageck_summary, args.counts, args.samples)
        if path is not None
    )
    manifest: dict[str, object] = {
        "report_type": "screen_signal_baseline",
        "package_version": __version__,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "inputs": snapshots,
        "parameters": {
            "screen_id": screen_id,
            "contrast_id": contrast_id,
            "resolved_screen_ids": result.qc_summary["screens"],
            "resolved_contrast_ids": result.qc_summary["contrasts"],
            "positive_tail_means": args.positive_tail_means,
            "positive_lfc_means": args.positive_lfc_means,
            "pseudocount": args.pseudocount,
            "low_count_threshold": args.low_count_threshold,
            "normalization_method": args.normalization_method,
            "direction_deadband": args.direction_deadband,
            "fdr_threshold": args.fdr_threshold,
        },
        "interpretation_boundary": (
            "screen-signal rank; no validation probability and no therapeutic "
            "recommendation"
        ),
    }
    _assert_input_snapshots_unchanged(snapshots)
    _write_screen_report_bundle(
        result,
        manifest,
        args.output_dir,
        input_paths=input_paths,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                **result.qc_summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _write_orcs_result(
    output_dir: str | Path,
    frames: dict[str, pd.DataFrame],
    *,
    header_map: dict[str, str],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        _write_frame(frame, output_dir / f"{name}.tsv")
    (output_dir / "header_map.json").write_text(
        json.dumps(header_map, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def command_ingest_orcs_index(args: argparse.Namespace) -> int:
    parsed = parse_orcs_index(
        args.table,
        release=args.release,
        retrieved_date=args.retrieved_date,
        available_date=args.available_date,
        organism_scope=args.organism_scope,
    )
    frames = {
        "raw_index": parsed.raw_index,
        "normalized_index": parsed.normalized_index,
        "studies": parsed.studies,
        "screens": parsed.screens,
        "screen_designs": parsed.screen_designs,
        "contrasts": parsed.contrasts,
        "external_screen_maps": parsed.external_screen_maps,
    }
    _write_orcs_result(
        args.output_dir,
        frames,
        header_map=parsed.header_map,
    )
    print(
        json.dumps(
            {
                "release": parsed.release,
                "output_dir": str(args.output_dir),
                "screens": len(parsed.screens),
                "studies": len(parsed.studies),
            },
            indent=2,
        )
    )
    return 0


def command_triage_orcs_index(args: argparse.Namespace) -> int:
    result = triage_orcs_index(
        args.table,
        release=args.release,
        retrieved_date=args.retrieved_date,
        available_date=args.available_date,
        organism_scope=args.organism_scope,
        policy_version=args.policy_version,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(result.screen_intake, output_dir / "screen_intake.tsv")
    _write_frame(
        result.eligibility_checks,
        output_dir / "eligibility_checks.tsv",
    )
    _write_frame(result.curation_queue, output_dir / "curation_queue.tsv")
    candidate_text = "".join(
        f"{screen_id}\n" for screen_id in result.candidate_screen_ids
    )
    (output_dir / "candidate_screen_ids.txt").write_text(
        candidate_text,
        encoding="utf-8",
    )
    (output_dir / "triage_summary.json").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    return 0


def command_prepare_orcs_release(args: argparse.Namespace) -> int:
    spec = load_orcs_release_spec(args.release_registry, release=args.release)
    prepared = prepare_orcs_release(
        spec,
        args.output_dir,
        retrieved_date=args.retrieved_date,
        policy_version=args.policy_version,
        archive_path=args.archive,
        cache_dir=args.cache_dir,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(prepared.summary, indent=2, sort_keys=True))
    return 0


def command_select_curation_batch(args: argparse.Namespace) -> int:
    manifest = write_curation_batch(
        args.queue,
        args.output_dir,
        batch_id=args.batch_id,
        selected_date=args.selected_date,
        start_rank=args.start_rank,
        batch_size=args.batch_size,
        require_unique_source_families=not args.allow_repeated_source_families,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def command_compare_curation_reviews(args: argparse.Namespace) -> int:
    manifest = write_dual_review_bundle(
        args.primary_reviews,
        args.secondary_reviews,
        args.selection,
        args.output_dir,
        assessed_date=args.assessed_date,
        allow_partial_secondary=args.allow_partial_secondary,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def command_complete_curation_reviews(args: argparse.Namespace) -> int:
    manifest = write_completed_dual_review_bundle(
        args.primary_reviews,
        args.completion_reviews,
        args.progress_reviews,
        args.progress_manifest,
        args.selection,
        args.partial_secondary_reviews,
        args.partial_comparison,
        args.partial_manifest,
        args.output_dir,
        assessed_date=args.assessed_date,
        expected_checkpoint_manifest_sha256=(args.expected_checkpoint_manifest_sha256),
        expected_progress_manifest_sha256=(args.expected_progress_manifest_sha256),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def command_prepare_event_review_packet(args: argparse.Namespace) -> int:
    manifest = write_event_review_packet(
        args.evidence_cards,
        args.actor_registry,
        args.rubric,
        args.output_dir,
        randomization_seed=args.randomization_seed,
        access_isolation_attested=args.access_isolation_attested,
        neutral_card_audit_attested=args.neutral_card_audit_attested,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def command_compare_event_reviews(args: argparse.Namespace) -> int:
    manifest = write_event_review_comparison_bundle(
        args.packet_dir,
        args.primary_reviews,
        args.secondary_reviews,
        args.actor_registry,
        args.output_dir,
        compared_at_utc=args.compared_at_utc,
        expected_packet_manifest_sha256=(args.expected_packet_manifest_sha256),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def command_finalize_event_adjudication(args: argparse.Namespace) -> int:
    manifest = write_final_adjudication_bundle(
        args.comparison_bundle_dir,
        args.adjudications,
        args.output_dir,
        released_at_utc=args.released_at_utc,
        expected_comparison_manifest_sha256=(args.expected_comparison_manifest_sha256),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def command_ingest_orcs_screen(args: argparse.Namespace) -> int:
    metadata = read_table(args.index_metadata) if args.index_metadata else None
    parsed = parse_orcs_screen_scores(
        args.table,
        release=args.release,
        index_metadata=metadata,
        contrast_id=args.contrast_id,
        source_file=Path(args.table).name,
    )
    frames = {
        "raw_scores": parsed.raw_scores,
        "normalized_scores": parsed.normalized_scores,
        "gene_scores": parsed.gene_scores,
        "issues": parsed.issues,
    }
    _write_orcs_result(
        args.output_dir,
        frames,
        header_map=parsed.header_map,
    )
    print(
        json.dumps(
            {
                "release": parsed.release,
                "output_dir": str(args.output_dir),
                "gene_score_rows": len(parsed.gene_scores),
                "issues": len(parsed.issues),
            },
            indent=2,
        )
    )
    return 1 if not parsed.issues.empty else 0


def command_benchmark(args: argparse.Namespace) -> int:
    table_path = Path(args.table).resolve()
    prediction_path = Path(args.output).resolve()
    metrics_path = Path(args.metrics_output).resolve() if args.metrics_output else None
    output_paths = [prediction_path]
    if metrics_path is not None:
        output_paths.append(metrics_path)
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("--output and --metrics-output must be different paths")
    if table_path in output_paths:
        raise ValueError("benchmark outputs cannot overwrite --table")
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"benchmark output paths already exist: {existing}")
    if prediction_path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
        raise ValueError("--output must be CSV or TSV")
    if metrics_path is not None and metrics_path.suffix.lower() != ".json":
        raise ValueError("--metrics-output must use a .json extension")

    frame, table_snapshot = _read_table_snapshot(args.table)
    if args.features and args.feature_profile:
        raise ValueError("--features and --feature-profile are mutually exclusive")
    if args.features:
        feature_columns = [
            value.strip() for value in args.features.split(",") if value.strip()
        ]
    elif args.feature_profile:
        feature_columns = FEATURE_PROFILES[args.feature_profile]
    else:
        feature_columns = DEFAULT_FEATURE_COLUMNS
    predictions, metrics = grouped_oof_predictions(
        frame,
        feature_columns=feature_columns,
        group_column=args.group_column,
        metric_study_column=args.metric_study_column,
        screen_column=args.screen_column,
        n_splits=args.folds,
        model_kind=args.model_kind,
        random_state=args.random_state,
        propensity_clip=(
            args.propensity_clip_lower,
            args.propensity_clip_upper,
        ),
    )
    query_columns = [args.screen_column]
    for column in ("contrast_id", "phenotype_direction"):
        if column in predictions and column not in query_columns:
            query_columns.append(column)

    seed_values = _parse_integer_list(args.seed_sweep, argument="--seed-sweep")
    if args.random_state not in seed_values:
        seed_values.insert(0, args.random_state)
    seed_metric_names = args.seed_metric or ["study_macro_normalized_average_precision"]
    unknown_seed_metrics = sorted(set(seed_metric_names) - set(metrics))
    if unknown_seed_metrics:
        raise ValueError(f"unknown --seed-metric: {unknown_seed_metrics}")
    seed_metric_values: dict[str, list[float]] = {
        metric_name: [] for metric_name in seed_metric_names
    }
    seed_runs: list[dict[str, object]] = []
    for seed in seed_values:
        if seed == args.random_state:
            seed_metrics = metrics
        else:
            try:
                _, seed_metrics = grouped_oof_predictions(
                    frame,
                    feature_columns=feature_columns,
                    group_column=args.group_column,
                    metric_study_column=args.metric_study_column,
                    screen_column=args.screen_column,
                    n_splits=args.folds,
                    model_kind=args.model_kind,
                    random_state=seed,
                    propensity_clip=(
                        args.propensity_clip_lower,
                        args.propensity_clip_upper,
                    ),
                )
            except ValueError as exc:
                seed_runs.append(
                    {
                        "seed": seed,
                        "status": "unavailable",
                        "reason": str(exc),
                    }
                )
                continue
        run_values: dict[str, float | None] = {}
        for metric_name in seed_metric_names:
            value = float(seed_metrics[metric_name])
            if pd.notna(value):
                seed_metric_values[metric_name].append(value)
                run_values[metric_name] = value
            else:
                run_values[metric_name] = None
        seed_runs.append({"seed": seed, "status": "evaluated", "metrics": run_values})

    seed_sensitivity = {}
    for metric_name, values in seed_metric_values.items():
        series = pd.Series(values, dtype=float)
        seed_sensitivity[metric_name] = {
            "effective_seeds": int(series.notna().sum()),
            "mean": float(series.mean()) if not series.empty else None,
            "sample_sd": (float(series.std(ddof=1)) if len(series) > 1 else None),
            "minimum": float(series.min()) if not series.empty else None,
            "maximum": float(series.max()) if not series.empty else None,
            "range": (float(series.max() - series.min()) if not series.empty else None),
        }

    if args.random_baseline_seed_count < 1:
        raise ValueError("--random-baseline-seed-count must be at least 1")
    baseline_predictions = predictions
    mageck_attestation = {
        "positive_tail_means": args.mageck_positive_tail_means,
        "mageck_version": args.mageck_version,
        "source_locator": args.mageck_source_locator,
        "source_sha256": args.mageck_source_sha256,
    }
    if any(value is not None for value in mageck_attestation.values()):
        missing = [name for name, value in mageck_attestation.items() if value is None]
        if missing:
            raise ValueError(
                f"MAGeCK baseline attestation requires all fields; missing {missing}"
            )
        baseline_predictions = adapt_repo_native_mageck_baseline(
            baseline_predictions,
            **mageck_attestation,
        )
    drugz_attestation = {
        "positive_normz_means": args.drugz_positive_normz_means,
        "drugz_version": args.drugz_version,
        "source_locator": args.drugz_source_locator,
        "source_sha256": args.drugz_source_sha256,
    }
    if any(value is not None for value in drugz_attestation.values()):
        missing = [name for name, value in drugz_attestation.items() if value is None]
        if missing:
            raise ValueError(
                f"drugZ baseline attestation requires all fields; missing {missing}"
            )
        baseline_predictions = adapt_direction_aligned_drugz_baseline(
            baseline_predictions,
            **drugz_attestation,
        )
    baselines = evaluate_fixed_score_baselines(
        baseline_predictions,
        screen_column=args.screen_column,
        target_column="target",
        study_column=args.metric_study_column,
        random_seeds=tuple(range(args.random_baseline_seed_count)),
        candidate_score_column="reproducibility_score",
        cluster_column=args.bootstrap_cluster_column,
        paired_metric=args.paired_baseline_metric,
        paired_bootstrap_repetitions=args.bootstrap_repetitions,
        paired_bootstrap_confidence=args.bootstrap_confidence,
        paired_bootstrap_min_effective_fraction=(args.bootstrap_min_effective_fraction),
        paired_bootstrap_random_state=args.bootstrap_random_state,
    )
    if args.model_kind == "logistic":
        baselines["regularized_logistic"] = {
            "status": "evaluated_as_candidate_model",
            "metrics": metrics,
        }

    else:
        logistic_predictions, logistic_metrics = grouped_oof_predictions(
            frame,
            feature_columns=feature_columns,
            group_column=args.group_column,
            metric_study_column=args.metric_study_column,
            screen_column=args.screen_column,
            n_splits=args.folds,
            model_kind="logistic",
            random_state=args.random_state,
            propensity_clip=(
                args.propensity_clip_lower,
                args.propensity_clip_upper,
            ),
        )
        baselines["regularized_logistic"] = {
            "status": "evaluated",
            "metrics": logistic_metrics,
        }
        if args.bootstrap_repetitions > 0:
            comparison = predictions.copy()
            comparison["_regularized_logistic_score"] = logistic_predictions[
                "reproducibility_score"
            ].to_numpy()
            baselines["regularized_logistic"]["paired_candidate_difference"] = (
                paired_cluster_bootstrap_difference(
                    comparison,
                    cluster_column=args.bootstrap_cluster_column,
                    metric_study_column=args.metric_study_column,
                    screen_column=args.screen_column,
                    target_column="target",
                    candidate_score_column="reproducibility_score",
                    comparator_score_column="_regularized_logistic_score",
                    metric_name=args.paired_baseline_metric,
                    query_columns=tuple(query_columns),
                    n_bootstrap=args.bootstrap_repetitions,
                    confidence=args.bootstrap_confidence,
                    min_effective_fraction=args.bootstrap_min_effective_fraction,
                    random_state=args.bootstrap_random_state,
                )
            )
    feature_invariance = within_query_feature_invariance(
        predictions,
        feature_columns=feature_columns,
        query_columns=query_columns,
    )
    conditional_calibration = reliability_table(
        predictions,
        target_column="target",
        score_column="reproducibility_score",
        n_bins=args.calibration_bins,
    )

    bootstrap = None
    if args.bootstrap_repetitions > 0:
        if args.bootstrap_cluster_column not in predictions:
            raise ValueError(
                "bootstrap cluster column is missing from predictions: "
                f"{args.bootstrap_cluster_column}"
            )
        bootstrap = study_cluster_bootstrap_intervals(
            predictions,
            study_column=args.bootstrap_cluster_column,
            metric_study_column=args.metric_study_column,
            screen_column=args.screen_column,
            target_column="target",
            score_column="reproducibility_score",
            n_bootstrap=args.bootstrap_repetitions,
            confidence=args.bootstrap_confidence,
            min_effective_fraction=args.bootstrap_min_effective_fraction,
            random_state=args.bootstrap_random_state,
        )

    report = {
        "report_type": "grouped_oof_benchmark",
        "package_version": __version__,
        "parameters": {
            "feature_columns": list(feature_columns),
            "group_column": args.group_column,
            "metric_study_column": args.metric_study_column,
            "screen_column": args.screen_column,
            "folds": args.folds,
            "model_kind": args.model_kind,
            "random_state": args.random_state,
            "propensity_clip": [
                args.propensity_clip_lower,
                args.propensity_clip_upper,
            ],
            "seed_sweep": seed_values,
            "seed_metrics": seed_metric_names,
            "bootstrap_cluster_column": args.bootstrap_cluster_column,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_confidence": args.bootstrap_confidence,
            "bootstrap_min_effective_fraction": (args.bootstrap_min_effective_fraction),
            "paired_baseline_metric": args.paired_baseline_metric,
            "mageck_attestation_supplied": all(
                value is not None for value in mageck_attestation.values()
            ),
            "drugz_attestation_supplied": all(
                value is not None for value in drugz_attestation.values()
            ),
        },
        "point_estimates": metrics,
        "baselines": baselines,
        "within_query_feature_invariance": feature_invariance,
        "conditional_calibration_bins": conditional_calibration,
        "seed_sensitivity": seed_sensitivity,
        "seed_runs": seed_runs,
        "cluster_bootstrap_intervals": bootstrap,
        "interpretation_boundaries": {
            "relative_score_not_validated_probability": True,
            "calibration_is_conditional_on_adjudicated_rows": True,
            "documented_success_metrics_count_missing_outcomes_as_not_documented": (
                True
            ),
            "bootstrap_conditions_on_fitted_oof_scores": True,
            "seed_sensitivity_is_not_a_confidence_interval": True,
        },
    }
    encoded = _write_benchmark_artifacts(
        predictions,
        report,
        prediction_path=prediction_path,
        metrics_path=metrics_path,
        input_snapshot=table_snapshot,
    )
    print(encoded)
    return 0


def command_freeze_model(args: argparse.Namespace) -> int:
    frame, snapshot = _read_table_snapshot(args.table)
    if args.features and args.feature_profile:
        raise ValueError("--features and --feature-profile are mutually exclusive")
    if args.features:
        feature_columns = [
            value.strip() for value in args.features.split(",") if value.strip()
        ]
    elif args.feature_profile:
        feature_columns = FEATURE_PROFILES[args.feature_profile]
    else:
        feature_columns = DEFAULT_FEATURE_COLUMNS
    manifest = freeze_model_bundle(
        frame,
        args.output_dir,
        feature_columns=list(feature_columns),
        training_table_sha256=snapshot["sha256"],
        training_table_name=snapshot["filename"],
        training_table_path=snapshot["path"],
        feature_registry_path=args.feature_registry,
        environment_lock_path=args.environment_lock,
        code_revision=args.code_revision,
        created_at_utc=args.created_at_utc,
        current_checkout_path=args.current_checkout_path,
        study_column=args.study_column,
        source_family_column=args.source_family_column,
        raw_family_column=args.raw_family_column,
        model_kind=args.model_kind,
        logistic_c=args.logistic_c,
        random_state=args.random_state,
        propensity_clip=(
            args.propensity_clip_lower,
            args.propensity_clip_upper,
        ),
    )
    _assert_input_snapshots_unchanged({"training_table": snapshot})
    report = {
        "bundle_dir": str(args.output_dir),
        "manifest_sha256": _file_sha256(Path(args.output_dir) / MANIFEST_FILENAME),
        "manifest": manifest,
    }
    print(
        json.dumps(
            _replace_nonfinite_with_none(report),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_verify_frozen_model(args: argparse.Namespace) -> int:
    manifest_path = Path(args.bundle_dir) / MANIFEST_FILENAME
    manifest = verify_model_bundle(
        args.bundle_dir,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    report = {
        "manifest_sha256": _file_sha256(manifest_path),
        "external_identity_anchor_checked": bool(args.expected_manifest_sha256),
        "manifest": manifest,
    }
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


def _parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an ISO-8601 datetime") from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("datetime must include a timezone")
    return parsed


def _parse_integer_list(value: str, *, argument: str) -> list[int]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"{argument} must contain at least one integer")
    try:
        parsed = [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"{argument} must be a comma-separated integer list") from exc
    return list(dict.fromkeys(parsed))


def _replace_nonfinite_with_none(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _replace_nonfinite_with_none(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_nonfinite_with_none(item) for item in value]
    if isinstance(value, tuple):
        return [_replace_nonfinite_with_none(item) for item in value]
    if isinstance(value, float) and not pd.notna(value):
        return None
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crispr-evidencerank")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a registry table")
    validate.add_argument("--table", required=True)
    validate.add_argument("--contract", choices=sorted(CONTRACTS), required=True)
    validate.set_defaults(func=command_validate)

    featurize = subparsers.add_parser(
        "featurize-counts", help="aggregate guide counts to gene features"
    )
    featurize.add_argument("--counts", required=True)
    featurize.add_argument("--samples", required=True)
    featurize.add_argument("--output", required=True)
    featurize.add_argument(
        "--positive-lfc-means",
        choices=["resistance", "sensitization"],
        required=True,
        help="explicit biological meaning of a positive treatment-minus-control LFC",
    )
    featurize.add_argument("--pseudocount", type=float, default=1.0)
    featurize.add_argument("--low-count-threshold", type=float, default=30.0)
    featurize.add_argument(
        "--normalization-method",
        choices=["median_ratio", "cpm"],
        default="median_ratio",
    )
    featurize.add_argument("--direction-deadband", type=float, default=0.1)
    featurize.set_defaults(func=command_featurize)

    design = subparsers.add_parser(
        "featurize-design",
        help="derive contrast-level prevalidation design features",
    )
    design.add_argument("--screens", required=True)
    design.add_argument("--screen-designs", required=True)
    design.add_argument("--contrasts", required=True)
    design.add_argument("--samples", required=True)
    design.add_argument("--output", required=True)
    design.set_defaults(func=command_featurize_design)

    orcs_index = subparsers.add_parser(
        "ingest-orcs-index",
        help="normalize a release-pinned BioGRID ORCS screen index",
    )
    orcs_index.add_argument("--table", required=True)
    orcs_index.add_argument("--release", required=True)
    orcs_index.add_argument("--retrieved-date", required=True)
    orcs_index.add_argument("--available-date")
    orcs_index.add_argument("--organism-scope")
    orcs_index.add_argument("--output-dir", required=True)
    orcs_index.set_defaults(func=command_ingest_orcs_index)

    orcs_triage = subparsers.add_parser(
        "triage-orcs-index",
        help="triage ORCS screens as exclude or metadata-only candidates",
    )
    orcs_triage.add_argument("--table", required=True)
    orcs_triage.add_argument("--release", required=True)
    orcs_triage.add_argument("--retrieved-date", required=True)
    orcs_triage.add_argument("--available-date")
    orcs_triage.add_argument("--organism-scope")
    orcs_triage.add_argument(
        "--policy-version",
        type=int,
        choices=sorted(SUPPORTED_POLICY_VERSIONS),
        default=2,
    )
    orcs_triage.add_argument("--output-dir", required=True)
    orcs_triage.set_defaults(func=command_triage_orcs_index)

    orcs_prepare = subparsers.add_parser(
        "prepare-orcs-release",
        help="verify and atomically prepare one pinned BioGRID ORCS release",
    )
    orcs_prepare.add_argument(
        "--release-registry",
        default="config/orcs_releases.yaml",
    )
    orcs_prepare.add_argument("--release", default="2.0.18")
    orcs_prepare.add_argument(
        "--retrieved-date",
        type=date.fromisoformat,
        required=True,
    )
    orcs_prepare.add_argument("--archive")
    orcs_prepare.add_argument("--cache-dir", default="data/external/orcs")
    orcs_prepare.add_argument(
        "--policy-version",
        type=int,
        choices=sorted(SUPPORTED_POLICY_VERSIONS),
        default=2,
    )
    orcs_prepare.add_argument("--timeout-seconds", type=float, default=60.0)
    orcs_prepare.add_argument("--output-dir", required=True)
    orcs_prepare.set_defaults(func=command_prepare_orcs_release)

    orcs_screen = subparsers.add_parser(
        "ingest-orcs-screen",
        help="normalize BioGRID ORCS per-screen author gene scores",
    )
    orcs_screen.add_argument("--table", required=True)
    orcs_screen.add_argument("--release", required=True)
    orcs_screen.add_argument("--index-metadata")
    orcs_screen.add_argument("--contrast-id")
    orcs_screen.add_argument("--output-dir", required=True)
    orcs_screen.set_defaults(func=command_ingest_orcs_screen)

    curation_batch = subparsers.add_parser(
        "select-curation-batch",
        help="freeze a contiguous outcome-blind full-text curation batch",
    )
    curation_batch.add_argument("--queue", required=True)
    curation_batch.add_argument("--output-dir", required=True)
    curation_batch.add_argument("--batch-id", required=True)
    curation_batch.add_argument(
        "--selected-date", type=date.fromisoformat, required=True
    )
    curation_batch.add_argument("--start-rank", type=int, default=1)
    curation_batch.add_argument("--batch-size", type=int, default=10)
    curation_batch.add_argument(
        "--allow-repeated-source-families",
        action="store_true",
    )
    curation_batch.set_defaults(func=command_select_curation_batch)

    review_comparison = subparsers.add_parser(
        "compare-curation-reviews",
        help="compare two review sets without releasing benchmark labels",
    )
    review_comparison.add_argument("--primary-reviews", required=True)
    review_comparison.add_argument("--secondary-reviews", required=True)
    review_comparison.add_argument("--selection", required=True)
    review_comparison.add_argument("--output-dir", required=True)
    review_comparison.add_argument(
        "--assessed-date", type=date.fromisoformat, required=True
    )
    review_comparison.add_argument(
        "--allow-partial-secondary",
        action="store_true",
        help="explicitly permit a checksum-bound incomplete secondary review",
    )
    review_comparison.set_defaults(func=command_compare_curation_reviews)

    review_completion = subparsers.add_parser(
        "complete-curation-reviews",
        help=(
            "complete checksum-pinned review checkpoints atomically for "
            "cooperating CLI writers"
        ),
    )
    review_completion.add_argument("--primary-reviews", required=True)
    review_completion.add_argument("--completion-reviews", required=True)
    review_completion.add_argument("--progress-reviews", required=True)
    review_completion.add_argument("--progress-manifest", required=True)
    review_completion.add_argument("--selection", required=True)
    review_completion.add_argument("--partial-secondary-reviews", required=True)
    review_completion.add_argument("--partial-comparison", required=True)
    review_completion.add_argument("--partial-manifest", required=True)
    review_completion.add_argument(
        "--expected-checkpoint-manifest-sha256", required=True
    )
    review_completion.add_argument("--expected-progress-manifest-sha256", required=True)
    review_completion.add_argument("--output-dir", required=True)
    review_completion.add_argument(
        "--assessed-date", type=date.fromisoformat, required=True
    )
    review_completion.set_defaults(func=command_complete_curation_reviews)

    event_packet = subparsers.add_parser(
        "prepare-event-review-packet",
        help=(
            "freeze a randomized content-filtered packet in an access-isolated "
            "review workspace"
        ),
    )
    event_packet.add_argument("--evidence-cards", required=True)
    event_packet.add_argument("--actor-registry", required=True)
    event_packet.add_argument("--rubric", required=True)
    event_packet.add_argument("--output-dir", required=True)
    event_packet.add_argument(
        "--randomization-seed",
        required=True,
        help="secret seed; only its SHA-256 is written to the packet manifest",
    )
    event_packet.add_argument(
        "--access-isolation-attested",
        action="store_true",
        required=True,
        help=(
            "attest that reviewers cannot access joinable prior-review, "
            "outcome-aware, or model-output artifacts"
        ),
    )
    event_packet.add_argument(
        "--neutral-card-audit-attested",
        action="store_true",
        required=True,
        help=(
            "attest that every distributable card received a manual neutrality "
            "audit; automated leakage checks alone are insufficient"
        ),
    )
    event_packet.set_defaults(func=command_prepare_event_review_packet)

    event_comparison = subparsers.add_parser(
        "compare-event-reviews",
        help="freeze two complete human review streams and agreement statistics",
    )
    event_comparison.add_argument("--packet-dir", required=True)
    event_comparison.add_argument("--primary-reviews", required=True)
    event_comparison.add_argument("--secondary-reviews", required=True)
    event_comparison.add_argument("--actor-registry", required=True)
    event_comparison.add_argument("--output-dir", required=True)
    event_comparison.add_argument(
        "--compared-at-utc",
        type=datetime.fromisoformat,
        required=True,
        help="UTC ISO-8601 timestamp, for example 2026-09-04T12:00:00+00:00",
    )
    event_comparison.add_argument(
        "--expected-packet-manifest-sha256",
        required=True,
    )
    event_comparison.set_defaults(func=command_compare_event_reviews)

    event_adjudication = subparsers.add_parser(
        "finalize-event-adjudication",
        help="publish named human adjudications and derived validation events",
    )
    event_adjudication.add_argument("--comparison-bundle-dir", required=True)
    event_adjudication.add_argument("--adjudications", required=True)
    event_adjudication.add_argument("--output-dir", required=True)
    event_adjudication.add_argument(
        "--released-at-utc",
        type=datetime.fromisoformat,
        required=True,
        help="UTC ISO-8601 timestamp, for example 2026-09-04T14:00:00+00:00",
    )
    event_adjudication.add_argument(
        "--expected-comparison-manifest-sha256",
        required=True,
    )
    event_adjudication.set_defaults(func=command_finalize_event_adjudication)

    rank = subparsers.add_parser(
        "rank-screen",
        help="create a self-contained screen-signal report for a new screen",
    )
    rank.add_argument("--mageck-summary")
    rank.add_argument("--counts")
    rank.add_argument("--samples")
    rank.add_argument("--screen-id")
    rank.add_argument("--contrast-id")
    rank.add_argument(
        "--positive-tail-means",
        choices=["resistance", "sensitization"],
    )
    rank.add_argument(
        "--positive-lfc-means",
        choices=["resistance", "sensitization"],
    )
    rank.add_argument("--pseudocount", type=float, default=1.0)
    rank.add_argument("--low-count-threshold", type=float, default=30.0)
    rank.add_argument(
        "--normalization-method",
        choices=["median_ratio", "cpm"],
        default="median_ratio",
    )
    rank.add_argument("--direction-deadband", type=float, default=0.1)
    rank.add_argument("--fdr-threshold", type=float, default=0.05)
    rank.add_argument("--output-dir", required=True)
    rank.set_defaults(func=command_rank_screen)

    immuno_context = subparsers.add_parser(
        "summarize-immuno-context",
        help="report source-family-aware immune context without changing ranking",
    )
    immuno_context.add_argument("--evidence", required=True)
    immuno_context.add_argument("--candidates", required=True)
    immuno_context.add_argument("--gene-column", default="gene_symbol")
    immuno_context.add_argument(
        "--cutoff-date",
        type=date.fromisoformat,
        required=True,
    )
    immuno_context.add_argument(
        "--target-modality",
        choices=[value.value for value in PerturbationModality],
        required=True,
    )
    immuno_context.add_argument(
        "--exclude-source-family",
        action="append",
        default=[],
    )
    immuno_context.add_argument(
        "--exclude-raw-data-family",
        action="append",
        default=[],
    )
    immuno_context.add_argument("--recurrence-stratum-id")
    immuno_context.add_argument("--dual-action-group-id")
    immuno_context.add_argument("--dual-action-group-version")
    immuno_context.add_argument("--max-source-fdr", type=float, default=0.05)
    immuno_context.add_argument(
        "--target-not-in-compendium",
        action="store_true",
        help=(
            "attest that the target screen and sibling raw/source families are "
            "absent from the evidence compendium"
        ),
    )
    immuno_context.add_argument("--output-dir", required=True)
    immuno_context.set_defaults(func=command_summarize_immuno_context)

    clinical_context = subparsers.add_parser(
        "summarize-clinical-context",
        help=(
            "report a frozen exact treatment-by-cancer interventional-trial "
            "landscape without changing gene ranking"
        ),
    )
    clinical_context.add_argument("--evidence", required=True)
    clinical_context.add_argument("--assets", required=True)
    clinical_context.add_argument("--treatment-concept-id", required=True)
    clinical_context.add_argument("--treatment-mapping-source", required=True)
    clinical_context.add_argument("--treatment-mapping-version", required=True)
    clinical_context.add_argument("--cancer-concept-id", required=True)
    clinical_context.add_argument("--cancer-mapping-source", required=True)
    clinical_context.add_argument("--cancer-mapping-version", required=True)
    clinical_context.add_argument(
        "--cutoff-date",
        type=date.fromisoformat,
        required=True,
    )
    clinical_context.add_argument(
        "--exclude-source-family",
        action="append",
        default=[],
    )
    clinical_context.add_argument("--output-dir", required=True)
    clinical_context.set_defaults(func=command_summarize_clinical_context)

    clinicaltrials_fetch = subparsers.add_parser(
        "fetch-clinicaltrials-gov",
        help=(
            "atomically freeze and verify a complete ClinicalTrials.gov v2 "
            "treatment/cancer search without creating clinical evidence"
        ),
    )
    clinicaltrials_fetch.add_argument("--condition-query", required=True)
    clinicaltrials_fetch.add_argument("--intervention-query", required=True)
    clinicaltrials_fetch.add_argument(
        "--page-size", type=int, default=DEFAULT_PAGE_SIZE
    )
    clinicaltrials_fetch.add_argument(
        "--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS
    )
    clinicaltrials_fetch.add_argument(
        "--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS
    )
    clinicaltrials_fetch.add_argument(
        "--backoff-seconds", type=float, default=DEFAULT_BACKOFF_SECONDS
    )
    clinicaltrials_fetch.add_argument(
        "--max-pages", type=int, default=DEFAULT_MAX_PAGES
    )
    clinicaltrials_fetch.add_argument(
        "--max-studies", type=int, default=DEFAULT_MAX_STUDIES
    )
    clinicaltrials_fetch.add_argument(
        "--max-candidate-rows", type=int, default=DEFAULT_MAX_CANDIDATE_ROWS
    )
    clinicaltrials_fetch.add_argument(
        "--max-page-bytes", type=int, default=DEFAULT_MAX_PAGE_BYTES
    )
    clinicaltrials_fetch.add_argument(
        "--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES
    )
    clinicaltrials_fetch.add_argument(
        "--max-derived-bytes", type=int, default=DEFAULT_MAX_DERIVED_BYTES
    )
    clinicaltrials_fetch.add_argument(
        "--max-elapsed-seconds",
        type=float,
        default=DEFAULT_MAX_ELAPSED_SECONDS,
    )
    clinicaltrials_fetch.add_argument("--output-dir", required=True)
    clinicaltrials_fetch.set_defaults(func=command_fetch_clinicaltrials_gov)

    clinicaltrials_verify = subparsers.add_parser(
        "verify-clinicaltrials-gov",
        help=("offline verification of a ClinicalTrials.gov raw pagination bundle"),
    )
    clinicaltrials_verify.add_argument("--snapshot-dir", required=True)
    clinicaltrials_verify.set_defaults(func=command_verify_clinicaltrials_gov)

    benchmark = subparsers.add_parser(
        "benchmark", help="run grouped out-of-fold baseline evaluation"
    )
    benchmark.add_argument("--table", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--features")
    benchmark.add_argument(
        "--feature-profile",
        choices=sorted(FEATURE_PROFILES),
    )
    benchmark.add_argument("--group-column", default="study_id")
    benchmark.add_argument(
        "--metric-study-column",
        default="study_id",
        help="study identifier used for study-macro metrics, separate from folds",
    )
    benchmark.add_argument("--screen-column", default="screen_id")
    benchmark.add_argument("--folds", type=int, default=5)
    benchmark.add_argument(
        "--model-kind",
        choices=["logistic", "hist_gradient_boosting"],
        default="logistic",
    )
    benchmark.add_argument("--random-state", type=int, default=17)
    benchmark.add_argument("--calibration-bins", type=int, default=10)
    benchmark.add_argument("--propensity-clip-lower", type=float, default=0.1)
    benchmark.add_argument("--propensity-clip-upper", type=float, default=0.95)
    benchmark.add_argument(
        "--seed-sweep",
        default="17,29,43,59,71",
        help="comma-separated split/model seeds for sensitivity diagnostics",
    )
    benchmark.add_argument(
        "--seed-metric",
        action="append",
        default=None,
        help=(
            "metric for seed sensitivity; repeat to request multiple metrics "
            "(default: study_macro_normalized_average_precision)"
        ),
    )
    benchmark.add_argument("--bootstrap-repetitions", type=int, default=200)
    benchmark.add_argument("--random-baseline-seed-count", type=int, default=100)
    benchmark.add_argument(
        "--paired-baseline-metric",
        default="study_macro_normalized_average_precision",
    )
    benchmark.add_argument(
        "--mageck-positive-tail-means",
        choices=["resistance", "sensitization"],
    )
    benchmark.add_argument("--mageck-version")
    benchmark.add_argument("--mageck-source-locator")
    benchmark.add_argument("--mageck-source-sha256")
    benchmark.add_argument(
        "--drugz-positive-normz-means",
        choices=["resistance", "sensitization"],
    )
    benchmark.add_argument("--drugz-version")
    benchmark.add_argument("--drugz-source-locator")
    benchmark.add_argument("--drugz-source-sha256")
    benchmark.add_argument("--bootstrap-random-state", type=int, default=17)
    benchmark.add_argument("--bootstrap-confidence", type=float, default=0.95)
    benchmark.add_argument(
        "--bootstrap-min-effective-fraction",
        type=float,
        default=0.8,
    )
    benchmark.add_argument(
        "--bootstrap-cluster-column",
        default="_outer_fold_group",
    )
    benchmark.add_argument("--metrics-output")
    benchmark.set_defaults(func=command_benchmark)

    freeze_model = subparsers.add_parser(
        "freeze-model",
        help="fit and atomically freeze a checksum-bound prospective model bundle",
    )
    freeze_model.add_argument("--table", required=True)
    freeze_model.add_argument("--output-dir", required=True)
    freeze_model.add_argument("--features")
    freeze_model.add_argument(
        "--feature-profile",
        choices=sorted(FEATURE_PROFILES),
    )
    freeze_model.add_argument("--feature-registry", default="config/features.yaml")
    freeze_model.add_argument("--environment-lock", default="uv.lock")
    freeze_model.add_argument("--code-revision", required=True)
    freeze_model.add_argument(
        "--current-checkout-path",
        help=(
            "optional Git checkout to attest: HEAD must equal --code-revision "
            "and tracked/untracked status must be clean"
        ),
    )
    freeze_model.add_argument(
        "--created-at-utc",
        type=_parse_datetime,
        required=True,
    )
    freeze_model.add_argument("--study-column", default="study_id")
    freeze_model.add_argument("--source-family-column", default="source_family_id")
    freeze_model.add_argument("--raw-family-column", default="raw_data_family_id")
    freeze_model.add_argument(
        "--model-kind",
        choices=["logistic", "hist_gradient_boosting"],
        default="logistic",
    )
    freeze_model.add_argument("--random-state", type=int, default=17)
    freeze_model.add_argument("--logistic-c", type=float, default=1.0)
    freeze_model.add_argument("--propensity-clip-lower", type=float, default=0.1)
    freeze_model.add_argument("--propensity-clip-upper", type=float, default=0.95)
    freeze_model.set_defaults(func=command_freeze_model)

    verify_frozen_model = subparsers.add_parser(
        "verify-frozen-model",
        help="verify a frozen model bundle without deserializing it",
    )
    verify_frozen_model.add_argument("--bundle-dir", required=True)
    verify_frozen_model.add_argument(
        "--expected-manifest-sha256",
        help=(
            "externally recorded manifest SHA-256; without it verification checks "
            "internal integrity but not bundle identity"
        ),
    )
    verify_frozen_model.set_defaults(func=command_verify_frozen_model)
    add_workflow_parsers(subparsers)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))
