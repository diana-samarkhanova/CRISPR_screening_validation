"""Apply a trusted frozen KO model to an outcome-free candidate roster."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import tempfile
from datetime import UTC, datetime
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from .clinicaltrials_gov import _publish_directory_noreplace
from .model_artifact import MANIFEST_FILENAME, load_model_bundle
from .modeling import ALLOWED_SUCCESS_FEATURES, _prepare_direction_features
from .prediction_baseline import verify_prediction_baseline

PREDICTION_MANIFEST_FILENAME = "prediction_manifest.json"
IDENTITY_COLUMNS = (
    "study_id",
    "screen_id",
    "contrast_id",
    "gene_symbol",
    "phenotype_direction",
    "perturbation_modality",
    "source_family_id",
    "raw_data_family_id",
)
QUERY_COLUMNS = ("study_id", "screen_id", "contrast_id", "phenotype_direction")
OPTIONAL_METADATA = ("candidate_role", "prior_outcome_exposure", "screen_signal_rank")
RAW_FEATURE_COLUMNS = {
    "median_guide_lfc",
    "mean_guide_lfc",
    "signal_direction",
    "positive_lfc_means",
}


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _file_record(path: Path, *, row_count: int | None = None) -> dict:
    content = path.read_bytes()
    record = {
        "filename": path.name,
        "sha256": _hash(content),
        "byte_size": len(content),
    }
    if row_count is not None:
        record["row_count"] = row_count
    return record


def _read_candidates(content: bytes, suffix: str) -> pd.DataFrame:
    if suffix not in {".csv", ".tsv", ".txt"}:
        raise ValueError("candidate table must be CSV or TSV")
    sep = "," if suffix == ".csv" else "\t"
    header = next(csv.reader(StringIO(content.decode("utf-8-sig")), delimiter=sep), [])
    if not header or len(header) != len(set(header)):
        raise ValueError("candidate table has an empty or duplicate column header")
    # Preserve opaque identifiers, including leading zeros and the literal 'NA'.
    return pd.read_csv(BytesIO(content), sep=sep, dtype=str, keep_default_na=False)


def _validate_roster(frame: pd.DataFrame) -> None:
    missing = set(IDENTITY_COLUMNS) - set(frame)
    if missing or frame.empty:
        raise ValueError(
            f"nonempty candidate roster requires identity columns: {sorted(missing)}"
        )
    allowed = set(IDENTITY_COLUMNS) | set(OPTIONAL_METADATA) | RAW_FEATURE_COLUMNS
    unknown = set(frame) - allowed - ALLOWED_SUCCESS_FEATURES
    if unknown:
        raise ValueError(
            f"candidate table contains unreviewed or outcome columns: {sorted(unknown)}"
        )
    for column in IDENTITY_COLUMNS:
        values = frame[column]
        if values.isna().any() or values.str.strip().eq("").any():
            raise ValueError(f"missing candidate identifier: {column}")
        if not values.eq(values.str.strip()).all():
            raise ValueError(
                f"candidate identifiers must be canonical without whitespace: {column}"
            )
    if not frame.perturbation_modality.eq("CRISPR_KO").all():
        raise ValueError("frozen v1 model inference accepts only CRISPR_KO")
    if not frame.phenotype_direction.isin(["resistance", "sensitization"]).all():
        raise ValueError(
            "candidate phenotype_direction must be resistance or sensitization"
        )
    if {"median_guide_lfc", "mean_guide_lfc"} & set(frame):
        if (
            "positive_lfc_means" not in frame
            or not frame.positive_lfc_means.isin(["resistance", "sensitization"]).all()
        ):
            raise ValueError(
                "raw guide LFC features require explicit positive_lfc_means"
            )
    if frame.duplicated(
        ["screen_id", "contrast_id", "gene_symbol", "phenotype_direction"]
    ).any():
        raise ValueError("duplicate gene-screen-contrast-direction candidates")
    for _, group in frame.groupby("screen_id", sort=False):
        for column in ("study_id", "source_family_id", "raw_data_family_id"):
            if group[column].nunique() != 1:
                raise ValueError(f"a screen has conflicting {column} identities")
    enums = {
        "prior_outcome_exposure": {"not_seen", "seen", "unknown"},
        "candidate_role": {"candidate", "positive_control", "negative_control"},
    }
    for column, choices in enums.items():
        if column in frame and not frame[column].isin(choices).all():
            raise ValueError(f"invalid {column} declaration")
    if "screen_signal_rank" in frame:
        ranks = pd.to_numeric(frame.screen_signal_rank, errors="raise")
        if not np.isfinite(ranks).all() or (ranks <= 0).any():
            raise ValueError("screen_signal_rank must be finite and positive")


def _training_overlap(frame: pd.DataFrame, manifest: dict) -> dict:
    provenance = manifest.get("training_provenance", {})
    identities = provenance.get("identities", {})
    assessments = {}
    for role in ("study_id", "source_family_id", "raw_data_family_id"):
        entry = identities.get(role, {})
        known = set(entry.get("identity_sha256", []))
        hashes = {
            _hash((role + "\0" + value.strip()).encode()) for value in frame[role]
        }
        assessments[role] = {
            "training_inventory_status": entry.get("status", "not_attested"),
            "overlap_count": len(hashes & known),
            "candidate_identity_count": len(hashes),
        }
    return {
        "identities": assessments,
        "complete_inventory": all(
            x["training_inventory_status"] == "complete" for x in assessments.values()
        ),
        "any_overlap": any(x["overlap_count"] for x in assessments.values()),
        "limitation": (
            "Checks declared identities only; cannot detect undisclosed reuse "
            "or prior outcome exposure."
        ),
    }


def predict_frozen_model(
    candidate_table_path: str | Path,
    model_bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_model_manifest_sha256: str,
    trusted_local_bundle: bool = False,
    created_at_utc: datetime,
    code_revision: str,
    allow_training_overlap: bool = False,
    max_missing_feature_fraction: float = 1.0,
    baseline_report_dir: str | Path | None = None,
    expected_baseline_manifest_sha256: str | None = None,
) -> dict:
    """Freeze relative scores; no model fitting, outcome calls or calibration."""
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"prediction output directory exists: {output_dir}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_model_manifest_sha256):
        raise ValueError("expected_model_manifest_sha256 must be a lowercase SHA-256")
    if not re.fullmatch(r"[0-9a-f]{40}", code_revision):
        raise ValueError("code_revision must be a full lowercase Git SHA")
    if created_at_utc.tzinfo is None or created_at_utc.utcoffset() is None:
        raise ValueError("created_at_utc must be timezone-aware")
    if (
        not np.isfinite(max_missing_feature_fraction)
        or not 0 <= max_missing_feature_fraction <= 1
    ):
        raise ValueError("max_missing_feature_fraction must be between zero and one")
    table_path = Path(candidate_table_path)
    content = table_path.read_bytes()
    frame = _read_candidates(content, table_path.suffix.lower())
    _validate_roster(frame)
    if (baseline_report_dir is None) != (expected_baseline_manifest_sha256 is None):
        raise ValueError(
            "baseline report and manifest SHA-256 must be supplied together"
        )
    baseline_linkage = {"status": "not_attested"}
    baseline_snapshots = {}
    if baseline_report_dir is not None:
        baseline_linkage, baseline_snapshots = verify_prediction_baseline(
            frame, baseline_report_dir, expected_baseline_manifest_sha256
        )
    manifest_path = Path(model_bundle_dir) / MANIFEST_FILENAME
    manifest_bytes = manifest_path.read_bytes()
    if _hash(manifest_bytes) != expected_model_manifest_sha256:
        raise ValueError("frozen model manifest identity hash mismatch")
    model, model_manifest = load_model_bundle(
        model_bundle_dir,
        trusted_local_bundle=trusted_local_bundle,
        expected_manifest_sha256=expected_model_manifest_sha256,
    )
    overlap = _training_overlap(frame, model_manifest)
    if not allow_training_overlap and (
        overlap["any_overlap"] or not overlap["complete_inventory"]
    ):
        raise ValueError(
            "independent inference requires complete training identity inventory "
            "and no training overlap"
        )

    numeric = frame.copy()
    for column in (
        ALLOWED_SUCCESS_FEATURES | {"median_guide_lfc", "mean_guide_lfc"}
    ) & set(numeric):
        numeric[column] = pd.to_numeric(
            numeric[column].replace("", np.nan), errors="raise"
        )
        if np.isinf(numeric[column].to_numpy(dtype=float)).any():
            raise ValueError(f"infinite model feature: {column}")
    numeric = _prepare_direction_features(numeric)
    missing_columns = set(model.feature_columns) - set(numeric)
    if missing_columns:
        raise ValueError(f"missing model feature columns: {sorted(missing_columns)}")
    x = numeric[model.feature_columns]
    preprocessor = model.success_model.named_steps["preprocess"]
    unsupported = set(getattr(preprocessor, "unsupported_columns_", []))
    supported_columns = [
        column for i, column in enumerate(model.feature_columns) if i not in unsupported
    ]
    if not supported_columns or x[supported_columns].isna().all(axis=1).any():
        raise ValueError("candidate has no observed feature supported in training")
    missing_fraction = x.isna().mean(axis=1)
    if (
        missing_fraction.eq(1) | missing_fraction.gt(max_missing_feature_fraction)
    ).any():
        raise ValueError(
            "candidate exceeds missing-feature threshold or has all features missing"
        )
    scores = model.predict(numeric)
    if (
        not np.isfinite(scores.to_numpy()).all()
        or not scores.ge(0).all().all()
        or not scores.le(1).all().all()
    ):
        raise ValueError("model produced invalid scores")
    preserved = [*IDENTITY_COLUMNS, *(c for c in OPTIONAL_METADATA if c in frame)]
    predictions = pd.concat([frame[preserved], scores], axis=1)
    predictions["model_rank"] = (
        predictions.groupby(list(QUERY_COLUMNS))["reproducibility_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    predictions["feature_missing_fraction"] = missing_fraction
    predictions = predictions.sort_values(
        [*QUERY_COLUMNS, "model_rank", "gene_symbol"], kind="stable"
    ).reset_index(drop=True)
    diagnostics = pd.DataFrame(
        {
            "feature": model.feature_columns,
            "missing_row_count": x.isna().sum().to_numpy(),
            "missing_fraction": x.isna().mean().to_numpy(),
            "unsupported_in_training": [
                i in unsupported for i in range(len(model.feature_columns))
            ],
        }
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        predictions.to_csv(staging / "predictions.tsv", sep="\t", index=False)
        diagnostics.to_csv(staging / "feature_diagnostics.tsv", sep="\t", index=False)
        (staging / "model_manifest.json").write_bytes(manifest_bytes)
        manifest = {
            "schema_version": 1,
            "bundle_type": "frozen_model_predictions",
            "created_at_utc": created_at_utc.astimezone(UTC).isoformat(),
            "code_revision": code_revision,
            "model_code_revision": model_manifest["code_revision"],
            "inputs": {
                "candidate_table": {
                    "filename": table_path.name,
                    "sha256": _hash(content),
                    "byte_size": len(content),
                },
                "model_manifest": _file_record(staging / "model_manifest.json"),
            },
            "outputs": {
                "predictions": _file_record(
                    staging / "predictions.tsv", row_count=len(predictions)
                ),
                "feature_diagnostics": _file_record(
                    staging / "feature_diagnostics.tsv", row_count=len(diagnostics)
                ),
            },
            "training_overlap": overlap,
            "baseline_linkage": baseline_linkage,
            "inference_mode": "exploratory_overlap_allowed"
            if allow_training_overlap
            else "declared_families_disjoint",
            "max_missing_feature_fraction": max_missing_feature_fraction,
            "score_semantics": {
                "primary": "reproducibility_score",
                "calibrated_probability": False,
                "selection_propensity_is_validation_probability": False,
                "rank_ties": "minimum_rank_then_gene_symbol_for_display",
            },
            "scientific_boundary": {
                "no_model_refitting": True,
                "no_wet_lab_outcome_in_input": True,
                "external_timestamp_not_created": True,
                "human_prior_exposure_not_verified": True,
                "does_not_establish_model_efficacy": True,
            },
        }
        (staging / PREDICTION_MANIFEST_FILENAME).write_bytes(_json_bytes(manifest))
        if (
            table_path.read_bytes() != content
            or manifest_path.read_bytes() != manifest_bytes
            or any(
                path.read_bytes() != value for path, value in baseline_snapshots.items()
            )
        ):
            raise ValueError("prediction input changed while creating the bundle")
        _publish_directory_noreplace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest
