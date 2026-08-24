"""Command-line interface for validation, featurization, and benchmarking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, date, datetime
from io import BytesIO
from pathlib import Path
from typing import get_args

import pandas as pd

from . import __version__
from .contracts import (
    CONTRACTS,
    ImmuneScreenEvidenceRecord,
    PerturbationModality,
    validate_records,
)
from .curation import (
    write_completed_dual_review_bundle,
    write_curation_batch,
    write_dual_review_bundle,
)
from .features import featurize_count_table, featurize_experimental_design
from .immuno_context import ImmunoContextResult, summarize_immuno_context
from .intake import SUPPORTED_POLICY_VERSIONS, triage_orcs_index
from .io import read_table
from .modeling import (
    DEFAULT_FEATURE_COLUMNS,
    FEATURE_PROFILES,
    grouped_oof_predictions,
)
from .orcs import parse_orcs_index, parse_orcs_screen_scores
from .orcs_prepare import prepare_orcs_release
from .orcs_release import load_orcs_release_spec
from .screen_report import ScreenReportResult, rank_screen


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
) -> tuple[pd.DataFrame, dict[str, str]]:
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
    }


def _assert_input_snapshots_unchanged(
    snapshots: dict[str, dict[str, str]],
) -> None:
    changed = sorted(
        role
        for role, snapshot in snapshots.items()
        if _file_sha256(snapshot["path"]) != snapshot["sha256"]
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
    snapshots: dict[str, dict[str, str]] = {}
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
    frame = read_table(args.table)
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
        screen_column=args.screen_column,
        n_splits=args.folds,
        model_kind=args.model_kind,
    )
    _write_frame(predictions, args.output)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


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
    benchmark.add_argument("--screen-column", default="screen_id")
    benchmark.add_argument("--folds", type=int, default=5)
    benchmark.add_argument(
        "--model-kind",
        choices=["logistic", "hist_gradient_boosting"],
        default="logistic",
    )
    benchmark.set_defaults(func=command_benchmark)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))
