"""Command-line interface for validation, featurization, and benchmarking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import UTC, date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import get_args

import numpy as np
import pandas as pd

from . import __version__
from .adjudication import (
    finalize_adjudication,
    hash_validation_events,
    prepare_adjudication_packet,
)
from .contracts import (
    CONTRACTS,
    BiomarkerAxisObservationStatus,
    BiomarkerFeatureType,
    ImmuneScreenEvidenceRecord,
    InterventionModality,
    MolecularMeasurementTimepoint,
    PatientMolecularEvidenceRecord,
    PerturbationModality,
    PerturbedCompartment,
    PreclinicalEvidenceRecord,
    RegimenComponentRelation,
    ScreenEndpointCategory,
    TreatmentDiseaseContextRecord,
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
from .screen_report import (
    RANK_SCREEN_CANDIDATE_SCHEMA,
    RANK_SCREEN_CANDIDATE_SCHEMA_VERSION,
    RANK_SCREEN_INTERPRETATION_BOUNDARY,
    RANK_SCREEN_MANIFEST_SCHEMA,
    RANK_SCREEN_MANIFEST_SCHEMA_VERSION,
    RANK_SCREEN_METHOD_VERSION,
    ScreenReportResult,
    rank_screen,
    render_screen_report_markdown,
    summarize_screen_qc,
)
from .translation_context import (
    ClinicalTrialsSnapshot,
    TranslationContextResult,
    build_translation_context_report,
    clinical_trials_snapshot_from_document,
    fetch_clinical_trials_concept_v2,
)

_RANK_SCREEN_COMMON_COLUMNS = frozenset(
    {
        "gene_symbol",
        "screen_id",
        "contrast_id",
        "phenotype_direction",
        "analysis_tail",
        "screen_signal_rank",
        "screen_signal_rank_source",
        "screen_signal_percentile",
        "screen_signal_percentile_scope",
        "ranking_type",
    }
)
_RANK_SCREEN_MAGECK_REQUIRED_COLUMNS = frozenset({"method", "cnv_corrected"})
_RANK_SCREEN_MAGECK_OPTIONAL_COLUMNS = frozenset(
    {
        "mageck_score",
        "mageck_lfc",
        "mageck_p_value",
        "mageck_fdr",
        "mageck_rank",
        "mageck_input_sgrna_n",
        "mageck_good_sgrna_n",
    }
)
_RANK_SCREEN_COUNT_COLUMNS = frozenset(
    {
        "gene_symbol",
        "guide_n",
        "median_guide_lfc",
        "mean_guide_lfc",
        "mean_control_count",
        "low_count_fraction",
        "zero_fraction_control",
        "zero_fraction_treatment",
        "positive_guide_fraction",
        "negative_guide_fraction",
        "neutral_guide_fraction",
        "guide_lfc_mad",
        "guide_lfc_iqr",
        "top2_abs_lfc_mean",
        "leave_one_guide_out_median_sd",
        "strongest_guide_dominance",
        "guide_direction_agreement",
        "absolute_median_guide_lfc",
        "absolute_mean_guide_lfc",
        "signal_direction",
        "is_sensitization_signal",
        "is_neutral_signal",
        "within_screen_effect_percentile",
        "replicate_correlation",
        "control_replicate_correlation",
        "treatment_replicate_correlation",
        "replicate_effect_sd",
        "median_library_size",
        "normalization_method",
        "screen_id",
        "contrast_id",
        "control_sample_n",
        "treatment_sample_n",
        "native_lfc_direction",
        "phenotype_direction",
        "analysis_tail",
        "screen_signal_rank",
        "screen_signal_percentile",
        "screen_signal_rank_source",
        "screen_signal_percentile_scope",
        "ranking_type",
    }
)
_RANK_SCREEN_COMBINED_COUNT_COLUMNS = frozenset(
    {
        column
        for column in _RANK_SCREEN_COUNT_COLUMNS
        if column not in {"gene_symbol", "screen_id", "contrast_id"}
        and column not in _RANK_SCREEN_COMMON_COLUMNS
    }
    | {
        "guide_phenotype_direction",
        "guide_analysis_tail",
        "guide_screen_signal_rank",
        "guide_screen_signal_percentile",
        "guide_screen_signal_rank_source",
        "guide_screen_signal_percentile_scope",
        "guide_ranking_type",
        "mageck_guide_direction_agreement",
    }
)


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
        _assert_input_snapshots_unchanged(result.metadata["inputs"])
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


def _is_missing_scalar(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _read_json_snapshot(
    path: str | Path,
    *,
    retrieved_at_utc: str | None = None,
) -> tuple[ClinicalTrialsSnapshot, dict[str, str]]:
    path = Path(path)
    content = path.read_bytes()
    try:
        document = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON input: {path}") from exc
    snapshot = clinical_trials_snapshot_from_document(
        document,
        retrieved_at_utc=retrieved_at_utc,
    )
    return snapshot, {
        "path": str(path),
        "filename": path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _read_rank_screen_manifest_snapshot(
    path: str | Path,
    *,
    candidate_path: str | Path,
    candidates: pd.DataFrame,
    candidate_sha256: str,
    screen_id: str | None,
    contrast_id: str | None,
) -> tuple[dict[str, object], dict[str, dict[str, str]]]:
    path = Path(path)
    content = path.read_bytes()
    try:
        document = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 rank-screen manifest: {path}") from exc
    if not isinstance(document, dict) or document.get("report_type") != (
        "screen_signal_baseline"
    ):
        raise ValueError("candidate manifest must be a rank-screen baseline manifest")
    expected_top_level = {
        "report_type",
        "mode",
        "manifest_schema",
        "manifest_schema_version",
        "rank_screen_method_version",
        "candidate_table_schema",
        "candidate_table_schema_version",
        "package_version",
        "created_at_utc",
        "inputs",
        "parameters",
        "interpretation_boundary",
        "outputs",
    }
    if set(document) != expected_top_level:
        raise ValueError("rank-screen manifest has an unsupported top-level schema")
    expected_markers = {
        "manifest_schema": RANK_SCREEN_MANIFEST_SCHEMA,
        "manifest_schema_version": RANK_SCREEN_MANIFEST_SCHEMA_VERSION,
        "rank_screen_method_version": RANK_SCREEN_METHOD_VERSION,
        "candidate_table_schema": RANK_SCREEN_CANDIDATE_SCHEMA,
        "candidate_table_schema_version": RANK_SCREEN_CANDIDATE_SCHEMA_VERSION,
        "interpretation_boundary": RANK_SCREEN_INTERPRETATION_BOUNDARY,
    }
    for field, expected in expected_markers.items():
        if document.get(field) != expected:
            raise ValueError(f"rank-screen manifest has unsupported {field}")
    if (
        not isinstance(document.get("package_version"), str)
        or not str(document["package_version"]).strip()
    ):
        raise ValueError("rank-screen manifest requires a package_version")
    created_at = document.get("created_at_utc")
    if not isinstance(created_at, str):
        raise ValueError("rank-screen manifest requires created_at_utc")
    try:
        parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("rank-screen manifest created_at_utc is invalid") from exc
    if parsed_created_at.tzinfo is None or parsed_created_at.utcoffset() != timedelta(
        0
    ):
        raise ValueError("rank-screen manifest created_at_utc must use UTC")

    def require_sha256(value: object, *, field: str) -> str:
        if (
            not isinstance(value, str)
            or not all(character in "0123456789abcdef" for character in value)
            or len(value) != 64
        ):
            raise ValueError(f"{field} must be a lowercase SHA-256")
        return value

    inputs = document.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("rank-screen manifest requires inputs")
    input_roles = set(inputs)
    mode_by_inputs = {
        frozenset({"mageck_summary"}): "mageck",
        frozenset({"counts", "samples"}): "counts",
        frozenset({"mageck_summary", "counts", "samples"}): "mageck_plus_counts",
    }
    mode = mode_by_inputs.get(frozenset(input_roles))
    if mode is None:
        raise ValueError("rank-screen manifest has an invalid input-role set")
    if document.get("mode") != mode:
        raise ValueError("rank-screen manifest mode disagrees with its input roles")
    for role, snapshot in inputs.items():
        if not isinstance(snapshot, dict) or set(snapshot) != {
            "path",
            "filename",
            "sha256",
        }:
            raise ValueError(f"rank-screen manifest input {role} is malformed")
        if not all(
            isinstance(snapshot[field], str) and snapshot[field].strip()
            for field in ("path", "filename")
        ):
            raise ValueError(f"rank-screen manifest input {role} lacks a filename")
        require_sha256(snapshot["sha256"], field=f"inputs.{role}.sha256")

    outputs = document.get("outputs")
    expected_output_files = {
        "ranked_candidates": "ranked_candidates.tsv",
        "qc_summary": "qc_summary.json",
        "report": "report.md",
    }
    if not isinstance(outputs, dict) or set(outputs) != set(expected_output_files):
        raise ValueError("rank-screen manifest outputs are incomplete or unsupported")
    for role, filename in expected_output_files.items():
        output = outputs.get(role)
        expected_output_fields = {"filename", "sha256"}
        if role == "ranked_candidates":
            expected_output_fields |= {
                "schema",
                "schema_version",
                "media_type",
                "row_count",
                "columns",
            }
        if not isinstance(output, dict) or set(output) != expected_output_fields:
            raise ValueError(f"rank-screen manifest output {role} is malformed")
        if output.get("filename") != filename:
            raise ValueError(f"rank-screen manifest output {role} filename is invalid")
        require_sha256(output.get("sha256"), field=f"outputs.{role}.sha256")
    ranked_output = (
        outputs.get("ranked_candidates") if isinstance(outputs, dict) else None
    )
    declared_sha256 = (
        ranked_output.get("sha256") if isinstance(ranked_output, dict) else None
    )
    if declared_sha256 != candidate_sha256:
        raise ValueError("candidate file SHA-256 disagrees with rank-screen manifest")
    if Path(candidate_path).name != expected_output_files["ranked_candidates"]:
        raise ValueError("rank-screen candidates must retain their canonical filename")
    ranked_row_count = ranked_output.get("row_count")
    ranked_columns = ranked_output.get("columns")
    if (
        ranked_output.get("schema") != RANK_SCREEN_CANDIDATE_SCHEMA
        or ranked_output.get("schema_version") != RANK_SCREEN_CANDIDATE_SCHEMA_VERSION
        or ranked_output.get("media_type") != "text/tab-separated-values"
        or isinstance(ranked_row_count, bool)
        or not isinstance(ranked_row_count, int)
        or ranked_row_count != len(candidates)
        or not isinstance(ranked_columns, list)
        or not all(isinstance(column, str) for column in ranked_columns)
        or ranked_columns != candidates.columns.tolist()
    ):
        raise ValueError("rank-screen candidate output metadata disagrees with table")

    parameters = document.get("parameters")
    expected_parameters = {
        "screen_id",
        "contrast_id",
        "resolved_screen_ids",
        "resolved_contrast_ids",
        "positive_tail_means",
        "positive_lfc_means",
        "pseudocount",
        "low_count_threshold",
        "normalization_method",
        "direction_deadband",
        "fdr_threshold",
    }
    if not isinstance(parameters, dict) or set(parameters) != expected_parameters:
        raise ValueError("rank-screen manifest parameters have an unsupported schema")
    resolved_screen_ids = parameters.get("resolved_screen_ids")
    resolved_contrast_ids = parameters.get("resolved_contrast_ids")
    for values, label in (
        (resolved_screen_ids, "resolved_screen_ids"),
        (resolved_contrast_ids, "resolved_contrast_ids"),
    ):
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) and value.strip() for value in values)
            or values != sorted(set(values))
        ):
            raise ValueError(f"rank-screen manifest {label} is invalid")
    if screen_id is not None and resolved_screen_ids != [screen_id]:
        raise ValueError("rank-screen manifest screen_id conflicts with context")
    if contrast_id is not None and resolved_contrast_ids != [contrast_id]:
        raise ValueError("rank-screen manifest contrast_id conflicts with context")
    for declared, resolved, label in (
        (parameters.get("screen_id"), resolved_screen_ids, "screen_id"),
        (parameters.get("contrast_id"), resolved_contrast_ids, "contrast_id"),
    ):
        if declared is not None and (
            not isinstance(declared, str) or resolved != [declared]
        ):
            raise ValueError(f"rank-screen manifest declared {label} is inconsistent")
    for field, lower_bound, upper_bound, lower_inclusive in (
        ("pseudocount", 0.0, None, False),
        ("low_count_threshold", 0.0, None, True),
        ("direction_deadband", 0.0, None, True),
        ("fdr_threshold", 0.0, 1.0, True),
    ):
        value = parameters.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"rank-screen manifest {field} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"rank-screen manifest {field} must be finite")
        if (numeric < lower_bound) or (not lower_inclusive and numeric == lower_bound):
            raise ValueError(f"rank-screen manifest {field} is out of range")
        if upper_bound is not None and numeric > upper_bound:
            raise ValueError(f"rank-screen manifest {field} is out of range")
    if parameters.get("normalization_method") not in {"median_ratio", "cpm"}:
        raise ValueError("rank-screen manifest normalization_method is invalid")
    allowed_directions = {"resistance", "sensitization"}
    positive_tail = parameters.get("positive_tail_means")
    positive_lfc = parameters.get("positive_lfc_means")
    if mode in {"mageck", "mageck_plus_counts"} and positive_tail not in (
        allowed_directions
    ):
        raise ValueError("rank-screen manifest lacks positive_tail_means")
    if mode == "counts" and positive_tail is not None:
        raise ValueError("count-only rank-screen manifest cannot declare a MAGeCK tail")
    if mode in {"counts", "mageck_plus_counts"} and positive_lfc not in (
        allowed_directions
    ):
        raise ValueError("rank-screen manifest lacks positive_lfc_means")
    if mode == "mageck" and positive_lfc is not None:
        raise ValueError(
            "MAGeCK-only rank-screen manifest cannot declare LFC direction"
        )
    if mode == "mageck_plus_counts" and positive_lfc != positive_tail:
        raise ValueError("combined rank-screen direction declarations conflict")

    required_candidate_columns = set(_RANK_SCREEN_COMMON_COLUMNS)
    if not required_candidate_columns <= set(candidates):
        missing = sorted(required_candidate_columns - set(candidates))
        raise ValueError(f"rank-screen candidate table is missing columns: {missing}")
    observed_candidate_columns = set(candidates)
    mageck_allowed_columns = (
        _RANK_SCREEN_COMMON_COLUMNS
        | _RANK_SCREEN_MAGECK_REQUIRED_COLUMNS
        | _RANK_SCREEN_MAGECK_OPTIONAL_COLUMNS
    )
    if mode == "counts":
        required_mode_columns = _RANK_SCREEN_COUNT_COLUMNS
        allowed_mode_columns = _RANK_SCREEN_COUNT_COLUMNS
    elif mode == "mageck":
        required_mode_columns = (
            _RANK_SCREEN_COMMON_COLUMNS | _RANK_SCREEN_MAGECK_REQUIRED_COLUMNS
        )
        allowed_mode_columns = mageck_allowed_columns
    else:
        required_mode_columns = (
            _RANK_SCREEN_COMMON_COLUMNS
            | _RANK_SCREEN_MAGECK_REQUIRED_COLUMNS
            | _RANK_SCREEN_COMBINED_COUNT_COLUMNS
        )
        allowed_mode_columns = (
            mageck_allowed_columns | _RANK_SCREEN_COMBINED_COUNT_COLUMNS
        )
    missing_mode_columns = sorted(required_mode_columns - observed_candidate_columns)
    unsupported_mode_columns = sorted(observed_candidate_columns - allowed_mode_columns)
    if missing_mode_columns or unsupported_mode_columns:
        raise ValueError(
            "rank-screen candidate columns disagree with the versioned mode schema: "
            f"missing={missing_mode_columns}, unsupported={unsupported_mode_columns}"
        )

    def validated_numeric_column(column: str) -> pd.Series:
        raw = candidates[column]
        if raw.map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError(
                f"rank-screen candidate {column} cannot contain boolean values"
            )
        numeric = pd.to_numeric(raw, errors="coerce")
        observed = ~raw.map(_is_missing_scalar)
        if (
            numeric.loc[observed].isna().any()
            or not numeric.loc[observed].map(math.isfinite).all()
        ):
            raise ValueError(f"rank-screen candidate {column} is not finite numeric")
        return numeric

    if mode in {"mageck", "mageck_plus_counts"}:
        if not candidates["method"].astype(str).eq("MAGeCK-RRA").all():
            raise ValueError("rank-screen MAGeCK method is invalid")
        if (
            candidates["cnv_corrected"]
            .map(lambda value: not isinstance(value, (bool, np.bool_)))
            .any()
            or candidates["cnv_corrected"].map(bool).any()
        ):
            raise ValueError("rank-screen MAGeCK cnv_corrected must be literal false")
        mageck_numeric = {
            column: validated_numeric_column(column)
            for column in _RANK_SCREEN_MAGECK_OPTIONAL_COLUMNS
            if column in candidates
        }
        for column in ("mageck_score", "mageck_p_value", "mageck_fdr"):
            if column in mageck_numeric:
                observed = mageck_numeric[column].dropna()
                if not observed.between(0.0, 1.0).all():
                    raise ValueError(f"rank-screen candidate {column} is out of range")
        for column, minimum in (
            ("mageck_rank", 1),
            ("mageck_input_sgrna_n", 0),
            ("mageck_good_sgrna_n", 0),
        ):
            if column in mageck_numeric:
                observed = mageck_numeric[column].dropna()
                if (observed < minimum).any() or not observed.map(
                    lambda value: float(value).is_integer()
                ).all():
                    raise ValueError(
                        f"rank-screen candidate {column} must contain integers"
                    )
        if {"mageck_input_sgrna_n", "mageck_good_sgrna_n"} <= set(mageck_numeric):
            input_n = mageck_numeric["mageck_input_sgrna_n"]
            good_n = mageck_numeric["mageck_good_sgrna_n"]
            if (good_n > input_n).fillna(False).any():
                raise ValueError(
                    "rank-screen candidate good sgRNA count exceeds input count"
                )

    if mode in {"counts", "mageck_plus_counts"}:
        count_numeric_columns = {
            "guide_n",
            "median_guide_lfc",
            "mean_guide_lfc",
            "mean_control_count",
            "low_count_fraction",
            "zero_fraction_control",
            "zero_fraction_treatment",
            "positive_guide_fraction",
            "negative_guide_fraction",
            "neutral_guide_fraction",
            "guide_lfc_mad",
            "guide_lfc_iqr",
            "top2_abs_lfc_mean",
            "leave_one_guide_out_median_sd",
            "strongest_guide_dominance",
            "guide_direction_agreement",
            "absolute_median_guide_lfc",
            "absolute_mean_guide_lfc",
            "is_sensitization_signal",
            "is_neutral_signal",
            "within_screen_effect_percentile",
            "replicate_correlation",
            "control_replicate_correlation",
            "treatment_replicate_correlation",
            "replicate_effect_sd",
            "median_library_size",
            "control_sample_n",
            "treatment_sample_n",
        }
        count_numeric = {
            column: validated_numeric_column(column)
            for column in count_numeric_columns
            if column in candidates
        }
        nullable_count_metrics = {
            "leave_one_guide_out_median_sd",
            "replicate_correlation",
            "control_replicate_correlation",
            "treatment_replicate_correlation",
            "replicate_effect_sd",
        }
        required_count_metrics = count_numeric_columns - nullable_count_metrics
        observed_count_rows = pd.DataFrame(
            {
                column: ~candidates[column].map(_is_missing_scalar)
                for column in required_count_metrics
            }
        ).any(axis=1)
        if mode == "counts" and not observed_count_rows.all():
            raise ValueError("count-mode rank-screen rows require complete count data")
        missing_required_count = pd.DataFrame(
            {
                column: candidates[column].map(_is_missing_scalar)
                for column in required_count_metrics
            }
        ).any(axis=1)
        if (observed_count_rows & missing_required_count).any():
            raise ValueError(
                "rank-screen observed guide rows require all intrinsic count metrics"
            )
        required_count_text = {
            "normalization_method",
            "native_lfc_direction",
            "signal_direction",
        }
        if (
            pd.DataFrame(
                {
                    column: candidates[column].map(_is_missing_scalar)
                    for column in required_count_text
                }
            )
            .loc[observed_count_rows]
            .any(axis=1)
            .any()
        ):
            raise ValueError(
                "rank-screen observed guide rows require count direction metadata"
            )
        if mode == "mageck_plus_counts":
            required_combined_text = {
                "guide_phenotype_direction",
                "guide_analysis_tail",
                "guide_screen_signal_rank_source",
                "guide_screen_signal_percentile_scope",
                "guide_ranking_type",
                "mageck_guide_direction_agreement",
            }
            if (
                pd.DataFrame(
                    {
                        column: candidates[column].map(_is_missing_scalar)
                        for column in required_combined_text
                    }
                )
                .loc[observed_count_rows]
                .any(axis=1)
                .any()
            ):
                raise ValueError(
                    "rank-screen observed combined guide rows require guide metadata"
                )
            no_guide_rows = ~observed_count_rows
            if (
                no_guide_rows.any()
                and pd.DataFrame(
                    {
                        column: ~candidates[column].map(_is_missing_scalar)
                        for column in _RANK_SCREEN_COMBINED_COUNT_COLUMNS
                    }
                )
                .loc[no_guide_rows]
                .any(axis=1)
                .any()
            ):
                raise ValueError(
                    "rank-screen missing guide rows cannot carry guide-derived fields"
                )
        for column in (
            "guide_n",
            "control_sample_n",
            "treatment_sample_n",
        ):
            observed = count_numeric[column].dropna()
            if (observed < 1).any() or not observed.map(
                lambda value: float(value).is_integer()
            ).all():
                raise ValueError(
                    f"rank-screen candidate {column} must contain positive integers"
                )
        for column in (
            "low_count_fraction",
            "zero_fraction_control",
            "zero_fraction_treatment",
            "positive_guide_fraction",
            "negative_guide_fraction",
            "neutral_guide_fraction",
            "strongest_guide_dominance",
            "guide_direction_agreement",
            "is_sensitization_signal",
            "is_neutral_signal",
            "within_screen_effect_percentile",
        ):
            if not count_numeric[column].dropna().between(0.0, 1.0).all():
                raise ValueError(f"rank-screen candidate {column} is out of range")
        for column in (
            "mean_control_count",
            "guide_lfc_mad",
            "guide_lfc_iqr",
            "top2_abs_lfc_mean",
            "leave_one_guide_out_median_sd",
            "absolute_median_guide_lfc",
            "absolute_mean_guide_lfc",
            "replicate_effect_sd",
            "median_library_size",
        ):
            if (count_numeric[column].dropna() < 0).any():
                raise ValueError(f"rank-screen candidate {column} cannot be negative")
        for column in (
            "replicate_correlation",
            "control_replicate_correlation",
            "treatment_replicate_correlation",
        ):
            if not count_numeric[column].dropna().between(-1.0, 1.0).all():
                raise ValueError(f"rank-screen candidate {column} is out of range")
        direction_fraction_sum = sum(
            count_numeric[column]
            for column in (
                "positive_guide_fraction",
                "negative_guide_fraction",
                "neutral_guide_fraction",
            )
        )
        if not np.allclose(
            direction_fraction_sum.loc[observed_count_rows],
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("rank-screen guide direction fractions must sum to one")
        discrete_fraction_denominators = {
            "low_count_fraction": count_numeric["guide_n"],
            "positive_guide_fraction": count_numeric["guide_n"],
            "negative_guide_fraction": count_numeric["guide_n"],
            "neutral_guide_fraction": count_numeric["guide_n"],
            "zero_fraction_control": (
                count_numeric["guide_n"] * count_numeric["control_sample_n"]
            ),
            "zero_fraction_treatment": (
                count_numeric["guide_n"] * count_numeric["treatment_sample_n"]
            ),
        }
        for column, denominator in discrete_fraction_denominators.items():
            implied_count = count_numeric[column] * denominator
            if not np.allclose(
                implied_count.loc[observed_count_rows],
                np.rint(implied_count.loc[observed_count_rows]),
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    f"rank-screen candidate {column} conflicts with its denominator"
                )
        leave_one_missing = candidates["leave_one_guide_out_median_sd"].map(
            _is_missing_scalar
        )
        expected_leave_one_missing = count_numeric["guide_n"].lt(3)
        if (
            not leave_one_missing.loc[observed_count_rows]
            .eq(expected_leave_one_missing.loc[observed_count_rows])
            .all()
        ):
            raise ValueError(
                "rank-screen leave-one-guide-out missingness conflicts with guide_n"
            )
        replicate_effect_missing = candidates["replicate_effect_sd"].map(
            _is_missing_scalar
        )
        expected_replicate_effect_missing = count_numeric["treatment_sample_n"].lt(2)
        if (
            not replicate_effect_missing.loc[observed_count_rows]
            .eq(expected_replicate_effect_missing.loc[observed_count_rows])
            .all()
        ):
            raise ValueError(
                "rank-screen replicate-effect missingness conflicts with sample count"
            )
        expected_agreement = pd.concat(
            [
                count_numeric["positive_guide_fraction"],
                count_numeric["negative_guide_fraction"],
            ],
            axis=1,
        ).max(axis=1)
        if not np.allclose(
            count_numeric["guide_direction_agreement"].loc[observed_count_rows],
            expected_agreement.loc[observed_count_rows],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("rank-screen guide direction agreement is inconsistent")
        if not np.allclose(
            count_numeric["absolute_mean_guide_lfc"].loc[observed_count_rows],
            count_numeric["mean_guide_lfc"].loc[observed_count_rows].abs(),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("rank-screen absolute mean guide LFC is inconsistent")
        for _, indices in (
            candidates.loc[observed_count_rows]
            .groupby(["screen_id", "contrast_id"], sort=False)
            .groups.items()
        ):
            gene_effects = pd.DataFrame(
                {
                    "gene_symbol": candidates.loc[indices, "gene_symbol"].astype(str),
                    "effect": count_numeric["median_guide_lfc"].loc[indices],
                    "reported": count_numeric["within_screen_effect_percentile"].loc[
                        indices
                    ],
                }
            )
            if (
                gene_effects.groupby("gene_symbol", sort=False)[["effect", "reported"]]
                .nunique(dropna=False)
                .gt(1)
                .any()
                .any()
            ):
                raise ValueError(
                    "rank-screen guide features vary across duplicated MAGeCK tails"
                )
            unique_gene_effects = gene_effects.drop_duplicates("gene_symbol")
            expected_effect_percentile = (
                unique_gene_effects["effect"].abs().rank(method="average", pct=True)
            )
            if not np.allclose(
                unique_gene_effects["reported"],
                expected_effect_percentile,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "rank-screen within-screen effect percentile is inconsistent"
                )
            for column in (
                "control_sample_n",
                "treatment_sample_n",
                "replicate_correlation",
                "control_replicate_correlation",
                "treatment_replicate_correlation",
                "median_library_size",
            ):
                if count_numeric[column].loc[indices].drop_duplicates().size > 1:
                    raise ValueError(
                        f"rank-screen candidate {column} varies within a contrast"
                    )
        if mode == "mageck_plus_counts":
            agreement = candidates["mageck_guide_direction_agreement"]
            observed_agreement = ~agreement.map(_is_missing_scalar)
            if (
                agreement.loc[observed_agreement]
                .map(lambda value: not isinstance(value, (bool, np.bool_)))
                .any()
            ):
                raise ValueError(
                    "rank-screen MAGeCK/guide agreement must be literal boolean"
                )
    if candidates.empty:
        raise ValueError("rank-screen candidate table cannot be empty")
    if not candidates["ranking_type"].astype(str).eq("screen_signal_baseline").all():
        raise ValueError("rank-screen candidate ranking_type is invalid")
    candidate_screens = sorted(candidates["screen_id"].astype(str).unique().tolist())
    candidate_contrasts = sorted(
        candidates["contrast_id"].astype(str).unique().tolist()
    )
    if candidate_screens != resolved_screen_ids or candidate_contrasts != (
        resolved_contrast_ids
    ):
        raise ValueError("rank-screen candidate identifiers disagree with manifest")
    candidate_directions = candidates["phenotype_direction"].astype(str)
    if not candidate_directions.isin({"resistance", "sensitization", "neutral"}).all():
        raise ValueError("rank-screen candidate phenotype_direction is invalid")
    rank_values = candidates["screen_signal_rank"]
    if rank_values.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError("rank-screen candidate ranks cannot be boolean")
    numeric_ranks = pd.to_numeric(rank_values, errors="coerce")
    raw_ranks_missing = rank_values.map(_is_missing_scalar)
    directional = candidate_directions.isin({"resistance", "sensitization"})
    directional_ranks = numeric_ranks.loc[directional]
    if (
        directional_ranks.isna().any()
        or not directional_ranks.map(math.isfinite).all()
        or (directional_ranks < 1).any()
        or not directional_ranks.map(lambda value: float(value).is_integer()).all()
    ):
        raise ValueError(
            "rank-screen directional ranks must be finite positive integers"
        )
    if not raw_ranks_missing.loc[~directional].all():
        raise ValueError("rank-screen neutral candidates must be unranked")
    raw_percentiles = candidates["screen_signal_percentile"]
    percentiles = pd.to_numeric(raw_percentiles, errors="coerce")
    raw_percentiles_missing = raw_percentiles.map(_is_missing_scalar)
    directional_percentiles = percentiles.loc[directional]
    if (
        directional_percentiles.isna().any()
        or not directional_percentiles.map(math.isfinite).all()
        or not directional_percentiles.between(0.0, 1.0).all()
        or not raw_percentiles_missing.loc[~directional].all()
    ):
        raise ValueError("rank-screen candidate percentiles are invalid")
    neutral = ~directional
    if neutral.any() and not (
        candidates.loc[neutral, "screen_signal_rank_source"]
        .astype(str)
        .eq("neutral_not_ranked")
        .all()
        and candidates.loc[neutral, "screen_signal_percentile_scope"]
        .astype(str)
        .eq("neutral_not_ranked")
        .all()
    ):
        raise ValueError("rank-screen neutral ranking metadata is invalid")
    candidate_keys = [
        "screen_id",
        "contrast_id",
        "gene_symbol",
        "phenotype_direction",
    ]
    if candidates.duplicated(candidate_keys).any():
        raise ValueError("rank-screen candidate keys are duplicated")
    expected_order = candidates.sort_values(
        [
            "screen_id",
            "contrast_id",
            "phenotype_direction",
            "screen_signal_rank",
            "gene_symbol",
        ],
        kind="stable",
        na_position="last",
    ).index.tolist()
    if expected_order != candidates.index.tolist():
        raise ValueError("rank-screen candidate rows are not in canonical order")

    positive_direction = positive_tail if mode != "counts" else positive_lfc
    negative_direction = (
        "sensitization" if positive_direction == "resistance" else "resistance"
    )
    if mode in {"mageck", "mageck_plus_counts"}:
        expected_tail_direction = {
            "mageck_pos": positive_direction,
            "mageck_neg": negative_direction,
        }
        allowed_rank_sources = {
            "mageck_native_rank",
            "derived_from_mageck_fdr",
            "derived_from_mageck_p_value",
            "derived_from_mageck_score",
        }
        expected_scope = "observed_rows_within_tail"
    else:
        expected_tail_direction = {
            "count_positive_lfc": positive_direction,
            "count_negative_lfc": negative_direction,
            "count_neutral": "neutral",
        }
        allowed_rank_sources = {"absolute_median_guide_lfc", "neutral_not_ranked"}
        expected_scope = "observed_rows_within_screen_contrast_direction"
    observed_tails = candidates["analysis_tail"].astype(str)
    if not observed_tails.isin(expected_tail_direction).all():
        raise ValueError("rank-screen candidate analysis_tail is invalid for its mode")
    mapped_directions = observed_tails.map(expected_tail_direction)
    if not mapped_directions.eq(candidate_directions).all():
        raise ValueError("rank-screen tail and phenotype direction disagree")
    observed_sources = candidates["screen_signal_rank_source"].astype(str)
    if not observed_sources.isin(allowed_rank_sources).all():
        raise ValueError("rank-screen rank source is invalid for its mode")

    if mode in {"counts", "mageck_plus_counts"}:
        raw_guide_lfc = candidates["median_guide_lfc"]
        guide_lfc = pd.to_numeric(raw_guide_lfc, errors="coerce")
        guide_observed = guide_lfc.notna()
        if (
            ~guide_observed & ~raw_guide_lfc.map(_is_missing_scalar)
        ).any() or not guide_lfc.loc[guide_observed].map(math.isfinite).all():
            raise ValueError("rank-screen median_guide_lfc is invalid")
        deadband = float(parameters["direction_deadband"])
        expected_native = pd.Series(pd.NA, index=candidates.index, dtype="string")
        expected_native.loc[guide_observed & guide_lfc.gt(deadband)] = "positive"
        expected_native.loc[guide_observed & guide_lfc.lt(-deadband)] = "negative"
        expected_native.loc[guide_observed & guide_lfc.between(-deadband, deadband)] = (
            "neutral"
        )
        expected_guide_direction = expected_native.map(
            {
                "positive": positive_lfc,
                "negative": (
                    "sensitization" if positive_lfc == "resistance" else "resistance"
                ),
                "neutral": "neutral",
            }
        )
        expected_guide_tail = expected_native.map(
            {
                "positive": "count_positive_lfc",
                "negative": "count_negative_lfc",
                "neutral": "count_neutral",
            }
        )
        prefix = "" if mode == "counts" else "guide_"
        for column, expected in (
            ("native_lfc_direction", expected_native),
            (f"{prefix}phenotype_direction", expected_guide_direction),
            (f"{prefix}analysis_tail", expected_guide_tail),
        ):
            observed = candidates[column].astype("string")
            if not observed.loc[guide_observed].eq(expected.loc[guide_observed]).all():
                raise ValueError(
                    "rank-screen guide LFC, tail, and phenotype direction disagree"
                )
            if not observed.loc[~guide_observed].isna().all():
                raise ValueError("rank-screen missing guide rows carry direction data")
        absolute_effect = pd.to_numeric(
            candidates["absolute_median_guide_lfc"], errors="coerce"
        )
        if not all(
            math.isclose(float(observed), abs(float(expected)), abs_tol=1e-12)
            for observed, expected in zip(
                absolute_effect.loc[guide_observed],
                guide_lfc.loc[guide_observed],
                strict=True,
            )
        ):
            raise ValueError("rank-screen absolute guide LFC is inconsistent")
        signal_direction = candidates["signal_direction"].astype("string")
        if (
            not signal_direction.loc[guide_observed]
            .eq(expected_guide_direction.loc[guide_observed])
            .all()
        ):
            raise ValueError("rank-screen guide signal_direction is inconsistent")
        expected_sensitization = expected_guide_direction.eq("sensitization").astype(
            float
        )
        expected_neutral = expected_guide_direction.eq("neutral").astype(float)
        for column, expected in (
            ("is_sensitization_signal", expected_sensitization),
            ("is_neutral_signal", expected_neutral),
        ):
            observed = pd.to_numeric(candidates[column], errors="coerce")
            if not observed.loc[guide_observed].eq(expected.loc[guide_observed]).all():
                raise ValueError(f"rank-screen {column} is inconsistent")
        if (
            not candidates["normalization_method"]
            .astype(str)
            .eq(parameters["normalization_method"])
            .loc[guide_observed]
            .all()
        ):
            raise ValueError("rank-screen normalization method is inconsistent")

        if mode == "mageck_plus_counts":
            expected_agreement = expected_guide_direction.eq(candidate_directions)
            observed_agreement = (
                candidates["mageck_guide_direction_agreement"]
                .astype("string")
                .str.casefold()
            )
            if (
                not observed_agreement.loc[guide_observed]
                .eq(expected_agreement.loc[guide_observed].astype(str).str.casefold())
                .all()
            ):
                raise ValueError("rank-screen MAGeCK/guide concordance is inconsistent")
            if not observed_agreement.loc[~guide_observed].isna().all():
                raise ValueError("rank-screen missing guide rows claim concordance")

            guide_rank_values = candidates["guide_screen_signal_rank"]
            guide_ranks = pd.to_numeric(guide_rank_values, errors="coerce")
            guide_percentile_values = candidates["guide_screen_signal_percentile"]
            guide_percentiles = pd.to_numeric(guide_percentile_values, errors="coerce")
            guide_directional = expected_guide_direction.isin(
                {"resistance", "sensitization"}
            )
            if (
                guide_ranks.loc[guide_directional].isna().any()
                or not guide_ranks.loc[guide_directional].map(math.isfinite).all()
                or not guide_ranks.loc[guide_directional]
                .map(lambda value: value >= 1 and float(value).is_integer())
                .all()
                or not guide_rank_values.loc[~guide_directional]
                .map(_is_missing_scalar)
                .all()
                or not guide_percentile_values.loc[~guide_directional]
                .map(_is_missing_scalar)
                .all()
            ):
                raise ValueError("rank-screen combined guide ranks are invalid")
            if (
                not candidates.loc[guide_directional, "guide_screen_signal_rank_source"]
                .astype(str)
                .eq("absolute_median_guide_lfc")
                .all()
                or not candidates.loc[
                    ~guide_directional, "guide_screen_signal_rank_source"
                ]
                .astype(str)
                .eq("neutral_not_ranked")
                .all()
            ):
                raise ValueError("rank-screen combined guide rank source is invalid")
            if (
                not candidates.loc[guide_observed, "guide_ranking_type"]
                .astype(str)
                .eq("screen_signal_baseline")
                .all()
            ):
                raise ValueError("rank-screen combined guide ranking_type is invalid")
            guide_scopes = candidates["guide_screen_signal_percentile_scope"].astype(
                "string"
            )
            if (
                not guide_scopes.loc[guide_directional]
                .eq("observed_rows_within_screen_contrast_direction")
                .all()
                or not guide_scopes.loc[guide_observed & ~guide_directional]
                .eq("neutral_not_ranked")
                .all()
            ):
                raise ValueError(
                    "rank-screen combined guide percentile scope is invalid"
                )
            for _, indices in (
                candidates.loc[guide_directional]
                .groupby(
                    ["screen_id", "contrast_id", "guide_analysis_tail"], sort=False
                )
                .groups.items()
            ):
                expected_guide_ranks = (
                    guide_lfc.loc[indices].abs().rank(method="min", ascending=False)
                )
                expected_guide_percentiles = 1.0 - (expected_guide_ranks - 1.0) / max(
                    len(indices) - 1, 1
                )
                if not all(
                    math.isclose(float(observed), float(expected), abs_tol=1e-12)
                    for observed, expected in zip(
                        guide_ranks.loc[indices], expected_guide_ranks, strict=True
                    )
                ) or not all(
                    math.isclose(float(observed), float(expected), abs_tol=1e-12)
                    for observed, expected in zip(
                        guide_percentiles.loc[indices],
                        expected_guide_percentiles,
                        strict=True,
                    )
                ):
                    raise ValueError("rank-screen combined guide ranking is invalid")

    observed_scopes = candidates["screen_signal_percentile_scope"].astype(str)
    if not observed_scopes.loc[directional].eq(expected_scope).all():
        raise ValueError("rank-screen percentile scope is invalid for its mode")
    for _, indices in (
        candidates.loc[directional]
        .groupby(["screen_id", "contrast_id", "analysis_tail"], sort=False)
        .groups.items()
    ):
        ranks = numeric_ranks.loc[indices]
        expected_percentiles = 1.0 - (ranks.rank(method="min") - 1.0) / max(
            len(ranks) - 1, 1
        )
        if not all(
            math.isclose(
                float(observed),
                float(expected),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for observed, expected in zip(
                percentiles.loc[indices], expected_percentiles, strict=True
            )
        ):
            raise ValueError("rank-screen candidate percentile formula is inconsistent")

        sources = observed_sources.loc[indices]
        if sources.nunique() != 1:
            raise ValueError("rank-screen rank source must be uniform within a tail")
        source = sources.iloc[0]
        metric_by_source = {
            "mageck_native_rank": ("mageck_rank", True, False),
            "derived_from_mageck_fdr": ("mageck_fdr", True, True),
            "derived_from_mageck_p_value": ("mageck_p_value", True, True),
            "derived_from_mageck_score": ("mageck_score", True, True),
            "absolute_median_guide_lfc": ("median_guide_lfc", False, True),
        }
        metric_name, ascending, derive_rank = metric_by_source[source]
        if metric_name not in candidates:
            raise ValueError(
                f"rank-screen rank source requires candidate column {metric_name}"
            )
        metric = pd.to_numeric(candidates.loc[indices, metric_name], errors="coerce")
        if metric.isna().any() or not metric.map(math.isfinite).all():
            raise ValueError("rank-screen rank-source metric is incomplete or invalid")
        if source == "absolute_median_guide_lfc":
            metric = metric.abs()
        if mode in {"mageck", "mageck_plus_counts"}:
            priority = (
                ("mageck_rank", "mageck_native_rank"),
                ("mageck_fdr", "derived_from_mageck_fdr"),
                ("mageck_p_value", "derived_from_mageck_p_value"),
                ("mageck_score", "derived_from_mageck_score"),
            )
            expected_source = next(
                (
                    candidate_source
                    for candidate_metric, candidate_source in priority
                    if candidate_metric in candidates
                    and pd.to_numeric(
                        candidates.loc[indices, candidate_metric], errors="coerce"
                    )
                    .notna()
                    .all()
                ),
                None,
            )
            if source != expected_source:
                raise ValueError(
                    "rank-screen MAGeCK rank source violates producer priority"
                )
        expected_ranks = (
            metric.rank(method="min", ascending=ascending) if derive_rank else metric
        )
        if not all(
            math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12)
            for observed, expected in zip(ranks, expected_ranks, strict=True)
        ):
            raise ValueError("rank-screen rank disagrees with its declared source")

    qc_path = path.parent / expected_output_files["qc_summary"]
    report_path = path.parent / expected_output_files["report"]
    auxiliary_snapshots: dict[str, dict[str, str]] = {}
    auxiliary_contents: dict[str, bytes] = {}
    for role, auxiliary_path in (
        ("candidate_manifest_qc_summary", qc_path),
        ("candidate_manifest_report", report_path),
    ):
        auxiliary_content = auxiliary_path.read_bytes()
        output_role = "qc_summary" if role.endswith("qc_summary") else "report"
        if (
            hashlib.sha256(auxiliary_content).hexdigest()
            != outputs[output_role]["sha256"]
        ):
            raise ValueError(
                f"rank-screen {output_role} checksum disagrees with manifest"
            )
        auxiliary_snapshots[role] = {
            "path": str(auxiliary_path),
            "filename": auxiliary_path.name,
            "sha256": hashlib.sha256(auxiliary_content).hexdigest(),
        }
        auxiliary_contents[output_role] = auxiliary_content
    try:
        qc = json.loads(auxiliary_contents["qc_summary"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("rank-screen qc_summary.json is invalid") from exc
    if not isinstance(qc, dict) or qc.get("report_type") != ("screen_signal_baseline"):
        raise ValueError("rank-screen QC report_type is invalid")
    if qc.get("mode") != mode:
        raise ValueError("rank-screen QC mode disagrees with manifest inputs")
    qc_gene_rows = qc.get("gene_rows")
    qc_unique_genes = qc.get("unique_genes")
    if (
        isinstance(qc_gene_rows, bool)
        or not isinstance(qc_gene_rows, int)
        or isinstance(qc_unique_genes, bool)
        or not isinstance(qc_unique_genes, int)
        or qc_gene_rows != len(candidates)
        or qc_unique_genes != int(candidates["gene_symbol"].astype(str).nunique())
    ):
        raise ValueError("rank-screen QC candidate counts disagree with table")
    if qc.get("screens") != resolved_screen_ids or qc.get("contrasts") != (
        resolved_contrast_ids
    ):
        raise ValueError("rank-screen QC identifiers disagree with manifest")
    if qc.get("fdr_threshold") != parameters.get("fdr_threshold"):
        raise ValueError("rank-screen QC FDR threshold disagrees with manifest")
    expected_qc = summarize_screen_qc(
        candidates,
        mode=mode,
        fdr_threshold=float(parameters["fdr_threshold"]),
    )
    if qc != expected_qc:
        raise ValueError("rank-screen QC semantics disagree with candidate table")
    try:
        report_text = auxiliary_contents["report"].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("rank-screen report.md is not valid UTF-8") from exc
    if report_text != render_screen_report_markdown(expected_qc, candidates):
        raise ValueError("rank-screen report semantics disagree with candidate table")
    snapshots = {
        "candidate_manifest": {
            "path": str(path),
            "filename": path.name,
            "sha256": hashlib.sha256(content).hexdigest(),
        },
        **auxiliary_snapshots,
    }
    return document, snapshots


def _write_translation_context_bundle(
    result: TranslationContextResult,
    output_dir: str | Path,
    *,
    input_paths: tuple[str | Path, ...],
) -> None:
    output_dir = Path(output_dir)
    output_names = (
        "context.json",
        "clinical_trials.tsv",
        "preclinical_used_evidence.tsv",
        "preclinical_exclusions.tsv",
        "patient_used_evidence.tsv",
        "patient_exclusions.tsv",
        "candidate_translation_context.tsv",
        "missingness.tsv",
        "clinicaltrials_snapshot.json",
        "report.md",
        "summary.json",
    )
    resolved_inputs = {Path(path).resolve() for path in input_paths}
    if any((output_dir / name).resolve() in resolved_inputs for name in output_names):
        raise ValueError("translation-context outputs cannot overwrite an input file")
    if output_dir.exists():
        raise FileExistsError(
            f"translation-context output directory exists: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        (staging / "context.json").write_text(
            json.dumps(
                result.metadata["context"],
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        for frame, filename in (
            (result.clinical_trials, "clinical_trials.tsv"),
            (result.preclinical_used_evidence, "preclinical_used_evidence.tsv"),
            (result.preclinical_exclusions, "preclinical_exclusions.tsv"),
            (result.patient_used_evidence, "patient_used_evidence.tsv"),
            (result.patient_exclusions, "patient_exclusions.tsv"),
            (result.candidate_context, "candidate_translation_context.tsv"),
            (result.missingness, "missingness.tsv"),
        ):
            _write_frame(frame, staging / filename)
        (staging / "clinicaltrials_snapshot.json").write_text(
            json.dumps(
                result.clinicaltrials_snapshot,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        (staging / "report.md").write_text(
            result.report_markdown,
            encoding="utf-8",
        )
        result.metadata["outputs"] = {
            name: {
                "filename": filename,
                "sha256": _file_sha256(staging / filename),
            }
            for name, filename in (
                ("context", "context.json"),
                ("clinical_trials", "clinical_trials.tsv"),
                ("preclinical_used", "preclinical_used_evidence.tsv"),
                ("preclinical_exclusions", "preclinical_exclusions.tsv"),
                ("patient_used", "patient_used_evidence.tsv"),
                ("patient_exclusions", "patient_exclusions.tsv"),
                ("candidate_context", "candidate_translation_context.tsv"),
                ("missingness", "missingness.tsv"),
                ("clinicaltrials_snapshot", "clinicaltrials_snapshot.json"),
                ("report", "report.md"),
            )
        }
        (staging / "summary.json").write_text(
            json.dumps(result.metadata, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        _assert_input_snapshots_unchanged(result.metadata["inputs"])
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def command_summarize_translation_context(args: argparse.Namespace) -> int:
    snapshots: dict[str, dict[str, str]] = {}
    input_paths: list[str | Path] = []
    candidate_manifest_document: dict[str, object] | None = None
    if args.clinicaltrials_json:
        clinical_snapshot, file_snapshot = _read_json_snapshot(
            args.clinicaltrials_json,
            retrieved_at_utc=args.retrieved_at_utc,
        )
        snapshots["clinicaltrials_json"] = file_snapshot
        input_paths.append(args.clinicaltrials_json)
    else:
        if args.retrieved_at_utc is not None:
            raise ValueError(
                "--retrieved-at-utc is only valid with --clinicaltrials-json"
            )
        clinical_snapshot = fetch_clinical_trials_concept_v2(
            args.treatment,
            args.cancer_type,
            treatment_entity_aliases=args.treatment_entity_alias,
            treatment_class_terms=args.treatment_class_term,
            cancer_entity_aliases=args.cancer_entity_alias,
            cancer_ancestor_terms=args.cancer_ancestor_term,
            disease_subtype=args.disease_subtype,
            subtype_entity_aliases=args.subtype_entity_alias,
            page_size=args.api_page_size,
            max_studies_per_query=args.api_max_studies,
            timeout_seconds=args.api_timeout_seconds,
        )

    optional_frames: dict[str, pd.DataFrame | None] = {
        "candidates": None,
        "preclinical_evidence": None,
        "patient_evidence": None,
    }
    for role, path, contract in (
        ("preclinical_evidence", args.preclinical_evidence, PreclinicalEvidenceRecord),
        ("patient_evidence", args.patient_evidence, PatientMolecularEvidenceRecord),
    ):
        if path is None:
            continue
        frame, snapshot = _read_table_snapshot(
            path,
            dtype=_contract_string_fields(contract),
        )
        optional_frames[role] = frame
        snapshots[role] = snapshot
        input_paths.append(path)
    if args.candidates:
        candidates, snapshot = _read_table_snapshot(
            args.candidates,
            dtype={
                "gene_symbol": "string",
                "screen_id": "string",
                "contrast_id": "string",
                "phenotype_direction": "string",
                "analysis_tail": "string",
                "ranking_type": "string",
            },
        )
        optional_frames["candidates"] = candidates
        snapshots["candidates"] = snapshot
        input_paths.append(args.candidates)
        ranking_columns = {"ranking_type", "screen_signal_rank"}
        present_ranking_columns = ranking_columns & set(candidates.columns)
        if present_ranking_columns and present_ranking_columns != ranking_columns:
            raise ValueError(
                "ranked candidates require both ranking_type and screen_signal_rank"
            )
        claims_screen_ranking = ranking_columns <= set(candidates.columns)
        if claims_screen_ranking and not args.candidate_manifest:
            raise ValueError(
                "ranked candidates require --candidate-manifest from rank-screen"
            )
        if args.candidate_manifest and not claims_screen_ranking:
            raise ValueError(
                "--candidate-manifest cannot bind structurally unranked candidates"
            )
        if args.candidate_manifest:
            candidate_manifest_document, manifest_snapshot = (
                _read_rank_screen_manifest_snapshot(
                    args.candidate_manifest,
                    candidate_path=args.candidates,
                    candidates=candidates,
                    candidate_sha256=snapshot["sha256"],
                    screen_id=args.screen_id,
                    contrast_id=args.contrast_id,
                )
            )
            snapshots.update(manifest_snapshot)
            input_paths.extend(
                [
                    args.candidate_manifest,
                    Path(args.candidate_manifest).parent / "qc_summary.json",
                    Path(args.candidate_manifest).parent / "report.md",
                ]
            )
    elif args.candidate_manifest:
        raise ValueError("--candidate-manifest requires --candidates")

    context = TreatmentDiseaseContextRecord.model_validate(
        {
            "context_id": args.context_id,
            "screen_id": args.screen_id,
            "contrast_id": args.contrast_id,
            "treatment_name": args.treatment,
            "treatment_id": args.treatment_id,
            "treatment_ontology_name": args.treatment_ontology_name,
            "treatment_ontology_version": args.treatment_ontology_version,
            "treatment_modality": args.treatment_modality,
            "regimen_name": args.regimen_name,
            "regimen_active_exposure_ids_json": (
                json.dumps(args.regimen_active_exposure_id)
                if args.regimen_active_exposure_id
                else None
            ),
            "regimen_component_relation": args.regimen_component_relation,
            "regimen_active_exposures_verified": (
                args.regimen_active_exposures_verified
            ),
            "regimen_active_exposure_identifier_source": (
                args.regimen_active_exposure_identifier_source
            ),
            "regimen_active_exposure_identifier_version": (
                args.regimen_active_exposure_identifier_version
            ),
            "cancer_type": args.cancer_type,
            "cancer_id": args.cancer_id,
            "disease_subtype": args.disease_subtype,
            "disease_subtype_id": args.disease_subtype_id,
            "disease_subtype_parent_id": args.disease_subtype_parent_id,
            "disease_subtype_parent_binding_verified": (
                args.disease_subtype_parent_binding_verified
            ),
            "disease_ontology_name": args.disease_ontology_name,
            "disease_ontology_version": args.disease_ontology_version,
            "stage": args.stage,
            "biomarker_context": args.biomarker_context,
            "biomarker_feature_type": args.biomarker_feature_type,
            "biomarker_state": args.biomarker_state,
            "biomarker_specimen_type": args.biomarker_specimen_type,
            "biomarker_measurement_timepoint": (args.biomarker_measurement_timepoint),
            "biomarker_axes_informative_verified": (
                args.biomarker_axes_informative_verified
            ),
            "biomarker_axes_observation_status": (
                args.biomarker_axes_observation_status
            ),
            "line_of_therapy": args.line_of_therapy,
            "screen_perturbation_modality": args.screen_perturbation_modality,
            "perturbed_compartment": args.perturbed_compartment,
            "screen_endpoint_category": args.screen_endpoint_category,
            "context_date": args.context_date,
        }
    )
    result = build_translation_context_report(
        context,
        clinical_snapshot,
        candidates=optional_frames["candidates"],
        preclinical_evidence=optional_frames["preclinical_evidence"],
        patient_evidence=optional_frames["patient_evidence"],
        evidence_cutoff_date=args.evidence_cutoff_date,
        treatment_entity_aliases=args.treatment_entity_alias,
        treatment_class_terms=args.treatment_class_term,
        cancer_entity_aliases=args.cancer_entity_alias,
        cancer_ancestor_terms=args.cancer_ancestor_term,
        subtype_entity_aliases=args.subtype_entity_alias,
        biomarker_aliases=args.biomarker_alias,
        target_source_family_id=args.target_source_family_id,
        target_raw_data_family_id=args.target_raw_data_family_id,
        target_absence_attested=args.target_not_in_evidence_catalog,
    )
    result.metadata.update(
        {
            "package_version": __version__,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "inputs": snapshots,
        }
    )
    result.metadata["candidate_input"].update(
        {
            "rank_screen_manifest_bound": candidate_manifest_document is not None,
            "ranking_claim": (
                "rank_screen_v1_contract_validated_checksum_bound"
                if candidate_manifest_document is not None
                else result.metadata["candidate_input"]["ranking_claim"]
            ),
            "rank_screen_manifest_report_type": (
                None
                if candidate_manifest_document is None
                else candidate_manifest_document["report_type"]
            ),
        }
    )
    _assert_input_snapshots_unchanged(snapshots)
    _write_translation_context_bundle(
        result,
        args.output_dir,
        input_paths=tuple(input_paths),
    )
    print(json.dumps(result.metadata, indent=2, sort_keys=True))
    return 0


def _write_screen_report_bundle(
    result: ScreenReportResult,
    manifest: dict[str, object],
    output_dir: str | Path,
    *,
    input_paths: tuple[str | Path, ...],
    input_snapshots: dict[str, dict[str, str]],
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
            json.dumps(result.qc_summary, indent=2, sort_keys=True, allow_nan=False),
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
        manifest["outputs"]["ranked_candidates"].update(
            {
                "schema": RANK_SCREEN_CANDIDATE_SCHEMA,
                "schema_version": RANK_SCREEN_CANDIDATE_SCHEMA_VERSION,
                "media_type": "text/tab-separated-values",
                "row_count": len(result.ranked_candidates),
                "columns": result.ranked_candidates.columns.tolist(),
            }
        )
        (staging / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        _assert_input_snapshots_unchanged(input_snapshots)
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
        "mode": result.qc_summary["mode"],
        "manifest_schema": RANK_SCREEN_MANIFEST_SCHEMA,
        "manifest_schema_version": RANK_SCREEN_MANIFEST_SCHEMA_VERSION,
        "rank_screen_method_version": RANK_SCREEN_METHOD_VERSION,
        "candidate_table_schema": RANK_SCREEN_CANDIDATE_SCHEMA,
        "candidate_table_schema_version": RANK_SCREEN_CANDIDATE_SCHEMA_VERSION,
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
        "interpretation_boundary": RANK_SCREEN_INTERPRETATION_BOUNDARY,
    }
    _assert_input_snapshots_unchanged(snapshots)
    _write_screen_report_bundle(
        result,
        manifest,
        args.output_dir,
        input_paths=input_paths,
        input_snapshots=snapshots,
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


def command_prepare_validation_adjudication(args: argparse.Namespace) -> int:
    manifest = prepare_adjudication_packet(
        args.completed_review_manifest,
        args.output_dir,
        expected_completed_review_manifest_sha256=(
            args.expected_completed_review_manifest_sha256
        ),
        expected_comparison_sha256=args.expected_comparison_sha256,
        packet_id=args.packet_id,
        prepared_date=args.prepared_date,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def command_finalize_validation_adjudication(args: argparse.Namespace) -> int:
    manifest = finalize_adjudication(
        args.packet_manifest,
        args.decisions,
        args.validation_events,
        args.output_dir,
        expected_packet_manifest_sha256=args.expected_packet_manifest_sha256,
        expected_decisions_sha256=args.expected_decisions_sha256,
        expected_validation_events_sha256=(args.expected_validation_events_sha256),
        adjudicated_date=args.adjudicated_date,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def command_hash_validation_events(args: argparse.Namespace) -> int:
    hashes, metadata = hash_validation_events(args.validation_events)
    _write_frame(hashes, args.output)
    metadata["output"] = {
        "filename": Path(args.output).name,
        "sha256": _file_sha256(args.output),
    }
    print(json.dumps(metadata, indent=2, sort_keys=True))
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
    if not args.development_synthetic_labels_only:
        raise ValueError(
            "real-label benchmark execution is disabled until a checksum-verified "
            "released-compendium manifest is implemented; use "
            "--development-synthetic-labels-only solely for synthetic tests"
        )
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
    predictions.insert(0, "evaluation_scope", "development_only")
    predictions.insert(1, "scientific_use_prohibited", True)
    predictions.insert(2, "label_provenance", "synthetic_unverified")
    _write_frame(predictions, args.output)
    print(
        json.dumps(
            {
                "evaluation_scope": "development_only",
                "scientific_use_prohibited": True,
                "label_provenance": "synthetic_unverified",
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )
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

    adjudication_preparation = subparsers.add_parser(
        "prepare-validation-adjudication",
        help=(
            "prepare an unsigned human adjudication packet from a "
            "checksum-pinned completed review bundle"
        ),
    )
    adjudication_preparation.add_argument("--completed-review-manifest", required=True)
    adjudication_preparation.add_argument(
        "--expected-completed-review-manifest-sha256", required=True
    )
    adjudication_preparation.add_argument("--expected-comparison-sha256", required=True)
    adjudication_preparation.add_argument("--packet-id", required=True)
    adjudication_preparation.add_argument(
        "--prepared-date", type=date.fromisoformat, required=True
    )
    adjudication_preparation.add_argument("--output-dir", required=True)
    adjudication_preparation.set_defaults(func=command_prepare_validation_adjudication)

    adjudication_finalization = subparsers.add_parser(
        "finalize-validation-adjudication",
        help=(
            "atomically release checksum-pinned human adjudication decisions "
            "and validation events"
        ),
    )
    adjudication_finalization.add_argument("--packet-manifest", required=True)
    adjudication_finalization.add_argument(
        "--expected-packet-manifest-sha256", required=True
    )
    adjudication_finalization.add_argument("--decisions", required=True)
    adjudication_finalization.add_argument("--expected-decisions-sha256", required=True)
    adjudication_finalization.add_argument("--validation-events", required=True)
    adjudication_finalization.add_argument(
        "--expected-validation-events-sha256", required=True
    )
    adjudication_finalization.add_argument(
        "--adjudicated-date", type=date.fromisoformat, required=True
    )
    adjudication_finalization.add_argument("--output-dir", required=True)
    adjudication_finalization.set_defaults(
        func=command_finalize_validation_adjudication
    )

    event_hashing = subparsers.add_parser(
        "hash-validation-events",
        help=(
            "validate validation events and emit canonical row hashes without "
            "assigning labels"
        ),
    )
    event_hashing.add_argument("--validation-events", required=True)
    event_hashing.add_argument("--output", required=True)
    event_hashing.set_defaults(func=command_hash_validation_events)

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

    translation = subparsers.add_parser(
        "summarize-translation-context",
        help=(
            "retrieve treatment/disease trial context and summarize separately "
            "curated patient/preclinical evidence without reranking genes"
        ),
    )
    translation.add_argument("--context-id", required=True)
    translation.add_argument("--screen-id")
    translation.add_argument("--contrast-id")
    translation.add_argument("--treatment", required=True)
    translation.add_argument("--treatment-id")
    translation.add_argument("--treatment-ontology-name")
    translation.add_argument("--treatment-ontology-version")
    translation.add_argument(
        "--treatment-modality",
        choices=[value.value for value in InterventionModality],
        required=True,
    )
    translation.add_argument("--regimen-name")
    translation.add_argument(
        "--regimen-active-exposure-id",
        action="append",
        default=[],
        help="canonical active-ingredient CURIE; repeat for fixed combinations",
    )
    translation.add_argument(
        "--regimen-component-relation",
        choices=[value.value for value in RegimenComponentRelation],
    )
    translation.add_argument(
        "--regimen-active-exposures-verified",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="curator verification of canonical active regimen components",
    )
    translation.add_argument("--regimen-active-exposure-identifier-source")
    translation.add_argument("--regimen-active-exposure-identifier-version")
    translation.add_argument(
        "--treatment-entity-alias",
        action="append",
        default=[],
        help="same-entity alias only; use --treatment-class-term for class terms",
    )
    translation.add_argument(
        "--treatment-class-term",
        action="append",
        default=[],
        help="broader treatment-class discovery term; never an exact entity match",
    )
    translation.add_argument("--cancer-type", required=True)
    translation.add_argument("--cancer-id")
    translation.add_argument(
        "--cancer-entity-alias",
        action="append",
        default=[],
        help="same-disease alias only; use --cancer-ancestor-term for ancestors",
    )
    translation.add_argument(
        "--cancer-ancestor-term",
        action="append",
        default=[],
        help="broader disease ancestor for discovery; never an exact disease match",
    )
    translation.add_argument("--disease-subtype")
    translation.add_argument("--disease-subtype-id")
    translation.add_argument("--disease-subtype-parent-id")
    translation.add_argument(
        "--disease-subtype-parent-binding-verified",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="curator attestation that subtype parent ID equals the cancer ID",
    )
    translation.add_argument("--disease-ontology-name")
    translation.add_argument("--disease-ontology-version")
    translation.add_argument(
        "--subtype-entity-alias",
        action="append",
        default=[],
        help="same-subtype alias or abbreviation",
    )
    translation.add_argument("--stage")
    translation.add_argument("--biomarker-context")
    translation.add_argument(
        "--biomarker-feature-type",
        choices=[value.value for value in BiomarkerFeatureType],
    )
    translation.add_argument("--biomarker-state")
    translation.add_argument("--biomarker-specimen-type")
    translation.add_argument(
        "--biomarker-measurement-timepoint",
        choices=[value.value for value in MolecularMeasurementTimepoint],
    )
    translation.add_argument(
        "--biomarker-axes-informative-verified",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "curator attestation that the biomarker term and every typed axis are "
            "informative; required for strict typed matching"
        ),
    )
    translation.add_argument(
        "--biomarker-axes-observation-status",
        choices=[value.value for value in BiomarkerAxisObservationStatus],
        help="controlled observed/missingness status for the typed biomarker bundle",
    )
    translation.add_argument("--biomarker-alias", action="append", default=[])
    translation.add_argument("--line-of-therapy")
    translation.add_argument(
        "--screen-perturbation-modality",
        choices=[value.value for value in PerturbationModality],
        required=True,
    )
    translation.add_argument(
        "--perturbed-compartment",
        choices=[value.value for value in PerturbedCompartment],
        required=True,
    )
    translation.add_argument(
        "--screen-endpoint-category",
        choices=[value.value for value in ScreenEndpointCategory],
        required=True,
    )
    translation.add_argument("--context-date", type=date.fromisoformat, required=True)
    translation.add_argument(
        "--evidence-cutoff-date", type=date.fromisoformat, required=True
    )
    translation.add_argument(
        "--clinicaltrials-json",
        help=(
            "frozen API response/snapshot for reproducible offline execution; "
            "the live v2 API is queried when omitted"
        ),
    )
    translation.add_argument(
        "--retrieved-at-utc",
        help=(
            "required provenance timestamp for a raw frozen API page that lacks "
            "one; cannot restamp a wrapped snapshot or a live retrieval"
        ),
    )
    translation.add_argument("--api-page-size", type=int, default=100)
    translation.add_argument(
        "--api-max-studies",
        type=int,
        default=500,
        help="maximum studies per declared treatment/condition query pair",
    )
    translation.add_argument("--api-timeout-seconds", type=float, default=30.0)
    translation.add_argument("--preclinical-evidence")
    translation.add_argument("--patient-evidence")
    translation.add_argument("--candidates")
    translation.add_argument(
        "--candidate-manifest",
        help=(
            "rank-screen run_manifest.json required when candidates claim a "
            "screen_signal_baseline ranking"
        ),
    )
    translation.add_argument("--target-source-family-id")
    translation.add_argument("--target-raw-data-family-id")
    translation.add_argument(
        "--target-not-in-evidence-catalog",
        action="store_true",
        help=(
            "attest that the target screen and sibling source/raw families are "
            "absent from the curated patient and preclinical evidence inputs"
        ),
    )
    translation.add_argument("--output-dir", required=True)
    translation.set_defaults(func=command_summarize_translation_context)

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
        "--development-synthetic-labels-only",
        action="store_true",
        help=(
            "explicitly mark this run as development-only synthetic evaluation; "
            "real released-label benchmarking remains disabled"
        ),
    )
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
