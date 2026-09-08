"""Freeze paired candidate allocation before prospective wet-lab outcomes exist.

The design evaluates the complete union of two top-k lists and samples the
remaining eligible genes without replacement. It does not analyze outcomes,
approve a statistical analysis plan, or establish an independent timestamp.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .clinicaltrials_gov import _publish_directory_noreplace
from .model_artifact import (
    _validate_training_identity_inventory,
    training_identity_sha256,
)

MANIFEST_FILENAME = "prospective_selection_manifest.json"
ALLOCATION_FILENAME = "allocation.tsv"
ASSAY_ROSTER_FILENAME = "assay_roster.tsv"
SUMMARY_FILENAME = "selection_summary.json"
QUERY_COLUMNS = (
    "screen_id",
    "contrast_id",
    "perturbation_modality",
    "phenotype_direction",
)
DECLARATION_COLUMNS = ("prior_outcome_exposure", "candidate_role")
OPTIONAL_METADATA_COLUMNS = (
    "study_id",
    "source_family_id",
    "raw_data_family_id",
    "cell_line_id",
    "treatment_id",
    "cancer_type",
    "selection_propensity",
    "model_rank",
    "feature_missing_fraction",
)
BASELINE_COLUMNS = {
    "screen_signal_rank": False,
    "mageck_rank": False,
    "mageck_fdr": False,
    "drugz_direction_aligned_normz": True,
}


@dataclass(frozen=True)
class ProspectiveSelectionResult:
    allocation: pd.DataFrame
    summary: dict[str, object]
    assay_roster: pd.DataFrame


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _check_hash(value: str, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _read_pinned(path: Path, expected: str, name: str) -> bytes:
    _check_hash(expected, name)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} source must be a regular non-symlink file")
    content = path.read_bytes()
    if _digest(content) != expected:
        raise ValueError(f"{name} hash mismatch")
    return content


def _load_json(content: bytes, name: str) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains nonfinite JSON: {value}")

    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} has duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        result = json.loads(
            content,
            parse_constant=reject_constant,
            object_pairs_hook=unique_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must contain valid JSON") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return result


def _read_roster(content: bytes, suffix: str) -> pd.DataFrame:
    if suffix not in {".tsv", ".txt", ".csv"}:
        raise ValueError("roster must be CSV or TSV")
    # Read identifiers literally, including genes named NA; validate scores below.
    return pd.read_csv(
        io.BytesIO(content),
        sep="," if suffix == ".csv" else "\t",
        dtype=str,
        keep_default_na=False,
    )


def build_prospective_selection(
    roster: pd.DataFrame,
    *,
    top_k: int,
    random_comparison_n: int,
    random_seed: int,
    model_score_column: str = "reproducibility_score",
    baseline_score_column: str = "screen_signal_rank",
    baseline_higher_is_better: bool = False,
) -> ProspectiveSelectionResult:
    """Allocate one query without examining validation labels or outcomes.

    Exposure and control-role fields are upstream user declarations, not
    verified human attestations. Previously seen/unknown outcomes are excluded
    from the prospective comparison. Controls are retained for assay execution
    and never counted as evidence of model performance.
    """

    for name, value, minimum in (
        ("top_k", top_k, 1),
        ("random_comparison_n", random_comparison_n, 0),
        ("random_seed", random_seed, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if random_seed >= 2**64:
        raise ValueError("random_seed must be less than 2**64")
    if model_score_column != "reproducibility_score":
        raise ValueError("model_score_column must be reproducibility_score")
    if baseline_score_column not in BASELINE_COLUMNS:
        raise ValueError("baseline_score_column is not a registered comparator")
    if (
        not isinstance(baseline_higher_is_better, bool)
        or baseline_higher_is_better != BASELINE_COLUMNS[baseline_score_column]
    ):
        raise ValueError("baseline direction conflicts with registered score semantics")
    required = {
        "gene_symbol",
        *QUERY_COLUMNS,
        *DECLARATION_COLUMNS,
        model_score_column,
        baseline_score_column,
    }
    if not roster.columns.is_unique:
        raise ValueError("roster columns must be unique")
    missing = sorted(required - set(roster.columns))
    if missing:
        raise ValueError(f"prospective roster is missing required columns: {missing}")
    unexpected = sorted(
        set(roster.columns)
        - required
        - set(OPTIONAL_METADATA_COLUMNS)
        - set(BASELINE_COLUMNS)
    )
    if unexpected:
        raise ValueError(
            "prospective roster contains unapproved columns; outcomes are forbidden: "
            f"{unexpected}"
        )
    if roster.empty:
        raise ValueError("prospective roster cannot be empty")
    frame = roster.copy().reset_index(drop=True)
    for column in ("gene_symbol", *QUERY_COLUMNS, *DECLARATION_COLUMNS):
        if frame[column].isna().any():
            raise ValueError(f"{column} cannot be missing")
        values = frame[column].astype(str)
        if values.str.strip().eq("").any() or values.ne(values.str.strip()).any():
            raise ValueError(f"{column} cannot be empty or have surrounding whitespace")
        if values.str.contains(r"[\x00-\x1f\x7f]", regex=True).any():
            raise ValueError(f"{column} cannot contain control characters")
        frame[column] = values
    if frame["gene_symbol"].duplicated().any():
        raise ValueError("gene_symbol must be unique within the query")
    for column in QUERY_COLUMNS:
        if frame[column].nunique() != 1:
            raise ValueError(f"one prospective packet must contain one {column}")
    if frame["perturbation_modality"].iloc[0] not in {
        "CRISPR_KO",
        "CRISPRa",
        "CRISPRi",
    }:
        raise ValueError(
            "prospective perturbation_modality must be CRISPR_KO/CRISPRa/CRISPRi"
        )
    if frame["phenotype_direction"].iloc[0] not in {"resistance", "sensitization"}:
        raise ValueError("phenotype_direction must be resistance or sensitization")
    if not frame["prior_outcome_exposure"].isin({"not_seen", "seen", "unknown"}).all():
        raise ValueError("prior_outcome_exposure must be not_seen, seen, or unknown")
    if (
        not frame["candidate_role"]
        .isin({"candidate", "positive_control", "negative_control"})
        .all()
    ):
        raise ValueError(
            "candidate_role must be candidate, positive_control, or negative_control"
        )
    for column in (model_score_column, baseline_score_column):
        if frame[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError(f"{column} cannot contain booleans")
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must be complete and finite on the same roster")
        frame[column] = values.astype(float)
    if not frame[model_score_column].between(0, 1).all():
        raise ValueError("reproducibility_score must lie in [0,1]")
    if baseline_score_column in {"screen_signal_rank", "mageck_rank"}:
        values = frame[baseline_score_column]
        if (values < 1).any() or values.ne(np.floor(values)).any():
            raise ValueError("baseline ranks must be positive integers")
    if baseline_score_column == "mageck_fdr":
        if not frame[baseline_score_column].between(0, 1).all():
            raise ValueError("mageck_fdr must lie in [0,1]")

    frame = frame.sort_values("gene_symbol", kind="stable").reset_index(drop=True)
    eligible = frame["candidate_role"].eq("candidate") & frame[
        "prior_outcome_exposure"
    ].eq("not_seen")
    candidates = frame.loc[eligible]
    if len(candidates) < top_k:
        raise ValueError("top_k exceeds the number of eligible unseen candidates")
    model_order = candidates.sort_values(
        [model_score_column, "gene_symbol"], ascending=[False, True], kind="stable"
    ).index
    baseline_order = candidates.sort_values(
        [baseline_score_column, "gene_symbol"],
        ascending=[not baseline_higher_is_better, True],
        kind="stable",
    ).index
    model_top = set(model_order[:top_k])
    baseline_top = set(baseline_order[:top_k])
    union = model_top | baseline_top
    remainder = [index for index in candidates.index if index not in union]
    if random_comparison_n > len(remainder):
        raise ValueError("random_comparison_n exceeds the remaining eligible pool")
    sampling_keys = {
        index: _digest(
            (
                f"prospective-selection-v1\0{random_seed}\0"
                f"{frame.at[index, 'gene_symbol']}"
            ).encode()
        )
        for index in remainder
    }
    random_order = sorted(
        remainder,
        key=lambda index: (sampling_keys[index], frame.at[index, "gene_symbol"]),
    )
    random_set = set(random_order[:random_comparison_n])
    probability = random_comparison_n / len(remainder) if remainder else None
    model_ranks = {index: rank for rank, index in enumerate(model_order, start=1)}
    baseline_ranks = {index: rank for rank, index in enumerate(baseline_order, start=1)}
    records = []
    for index, row in frame.iterrows():
        role = row["candidate_role"]
        exposure = row["prior_outcome_exposure"]
        is_eligible = bool(eligible.loc[index])
        if role != "candidate":
            stratum, reason = "assay_control", "control_not_comparison_candidate"
        elif exposure != "not_seen":
            stratum, reason = "excluded", f"prior_outcome_{exposure}"
        elif index in model_top & baseline_top:
            stratum, reason = "both_top_k", ""
        elif index in model_top:
            stratum, reason = "model_only_top_k", ""
        elif index in baseline_top:
            stratum, reason = "baseline_only_top_k", ""
        else:
            stratum, reason = "remainder", ""
        comparison_selected = index in union or index in random_set
        records.append(
            {
                "gene_symbol": row["gene_symbol"],
                **{column: row[column] for column in QUERY_COLUMNS},
                "candidate_role": role,
                "prior_outcome_exposure": exposure,
                "model_score": row[model_score_column],
                "baseline_score": row[baseline_score_column],
                "eligible_for_comparison": is_eligible,
                "exclusion_reason": reason,
                "model_rank": model_ranks.get(index),
                "baseline_rank": baseline_ranks.get(index),
                "in_model_top_k": index in model_top,
                "in_baseline_top_k": index in baseline_top,
                "selection_stratum": stratum,
                "random_comparison_selected": index in random_set,
                "selected_for_comparison": comparison_selected,
                "selected_for_wet_lab": comparison_selected or role != "candidate",
                "inclusion_probability": (
                    (1.0 if index in union else probability) if is_eligible else None
                ),
                "randomization_key": sampling_keys.get(index, ""),
            }
        )
    allocation = pd.DataFrame.from_records(records)
    query_key = "\0".join(frame[column].iloc[0] for column in QUERY_COLUMNS)
    allocation["assay_id"] = [
        "ASSAY-"
        + _digest(f"prospective-assay-v1\0{random_seed}\0{query_key}\0{gene}".encode())[
            :20
        ]
        if selected
        else ""
        for gene, selected in zip(
            allocation["gene_symbol"], allocation["selected_for_wet_lab"], strict=True
        )
    ]
    assay_roster = (
        allocation.loc[
            allocation["selected_for_wet_lab"],
            ["assay_id", "gene_symbol", *QUERY_COLUMNS, "candidate_role"],
        ]
        .sort_values("assay_id", kind="stable")
        .reset_index(drop=True)
    )
    if assay_roster["assay_id"].duplicated().any():
        raise ValueError("opaque assay identifier collision")
    for column in ("model_rank", "baseline_rank"):
        allocation[column] = allocation[column].astype("Int64")
    summary: dict[str, object] = {
        "schema_version": 1,
        "design": "complete_paired_top_k_union_plus_random_remainder",
        "query": {column: frame[column].iloc[0] for column in QUERY_COLUMNS},
        "parameters": {
            "top_k": top_k,
            "random_comparison_n": random_comparison_n,
            "random_seed": random_seed,
            "model_score_column": model_score_column,
            "baseline_score_column": baseline_score_column,
            "baseline_higher_is_better": baseline_higher_is_better,
        },
        "counts": {
            "input_roster": len(frame),
            "eligible_comparison_candidates": len(candidates),
            "model_top_k": len(model_top),
            "baseline_top_k": len(baseline_top),
            "top_k_overlap": len(model_top & baseline_top),
            "top_k_union": len(union),
            "remainder_pool": len(remainder),
            "random_comparison_selected": len(random_set),
            "selected_for_comparison": len(union | random_set),
            "assay_controls": int(frame["candidate_role"].ne("candidate").sum()),
            "excluded_candidates": int(
                (~eligible & frame["candidate_role"].eq("candidate")).sum()
            ),
            "selected_for_wet_lab": int(allocation["selected_for_wet_lab"].sum()),
        },
        "sampling": {
            "method": "seeded_sha256_order_without_replacement",
            "tie_break": "gene_symbol_ascending_unicode_codepoint",
            "seed_must_be_prespecified_before_inspecting_selection": True,
            "remainder_inclusion_probability": probability,
            "all_eligible_have_positive_inclusion_probability": (
                not remainder or random_comparison_n > 0
            ),
            "sampling_probabilities_are_design_values_not_response_probabilities": True,
        },
        "interpretation": {
            "top_k_union_is_fully_allocated": True,
            "wet_lab_outcomes_present": False,
            "untested_candidates_are_not_negative_labels": True,
            "exposure_and_control_roles_are_unverified_upstream_declarations": True,
            "model_scores_are_not_proven_clinical_or_validation_probabilities": True,
            "controls_excluded_from_model_comparison": True,
            "full_roster_claim_requires_weighted_analysis_and_outcomes": True,
            "within_query_comparison_does_not_establish_generalization": True,
            "assay_failure_missingness_and_power_require_completed_analysis_plan": True,
            "external_timestamp_and_human_approval_not_created": True,
            "assay_roster_hides_model_scores_and_allocation_strata": True,
            "assay_order_is_not_plate_block_randomization": True,
        },
    }
    return ProspectiveSelectionResult(allocation, summary, assay_roster)


def _validate_linkage(
    prediction_content: bytes,
    model_content: bytes,
    *,
    roster_sha256: str,
    model_sha256: str,
    roster: pd.DataFrame,
    created_at_utc: str,
    baseline_score_column: str,
    baseline_higher_is_better: bool,
) -> None:
    prediction = _load_json(prediction_content, "prediction manifest")
    model = _load_json(model_content, "model manifest")
    for name, manifest in (("prediction", prediction), ("model", model)):
        if (
            type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1
        ):
            raise ValueError(f"unsupported {name} manifest schema version")
    if prediction.get("bundle_type") != "frozen_model_predictions":
        raise ValueError("prediction manifest must describe frozen_model_predictions")
    overlap = prediction.get("training_overlap")
    if (
        prediction.get("inference_mode") != "declared_families_disjoint"
        or not isinstance(overlap, dict)
        or overlap.get("complete_inventory") is not True
        or overlap.get("any_overlap") is not False
    ):
        raise ValueError(
            "prospective allocation requires complete declared disjoint training "
            "families; exploratory predictions are ineligible"
        )
    if model.get("bundle_type") != "frozen_reproducibility_model":
        raise ValueError("model manifest must describe frozen_reproducibility_model")
    if model.get("status") != "frozen_not_yet_prospectively_evaluated":
        raise ValueError("model manifest has unsupported frozen model status")
    training_table = model.get("training_table")
    row_count = (
        training_table.get("row_count") if isinstance(training_table, dict) else None
    )
    if type(row_count) is not int or row_count <= 0:
        raise ValueError("model training_table must declare a positive row_count")
    _validate_training_identity_inventory(model.get("training_provenance"), row_count)
    for role, inventory in model["training_provenance"]["identities"].items():
        if inventory["status"] != "complete" or role not in roster:
            raise ValueError(
                "prospective allocation requires complete family identities"
            )
        hashes = {training_identity_sha256(role, value) for value in roster[role]}
        if hashes & set(inventory["identity_sha256"]):
            raise ValueError("roster has training overlap despite disjoint declaration")
    dates = []
    for label, value in (
        ("model", model.get("created_at_utc")),
        ("prediction", prediction.get("created_at_utc")),
        ("selection", created_at_utc),
    ):
        try:
            parsed = datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{label} created_at_utc must be a timestamp") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"{label} created_at_utc must include a timezone")
        dates.append(parsed)
    if not dates[0] <= dates[1] <= dates[2]:
        raise ValueError("timestamps must satisfy model <= prediction <= selection")
    baseline = prediction.get("baseline_linkage")
    if (
        not isinstance(baseline, dict)
        or baseline.get("status") != "verified_report_linkage"
        or baseline.get("baseline_score_column") != baseline_score_column
        or baseline.get("higher_is_better") is not baseline_higher_is_better
    ):
        raise ValueError("selected baseline must match verified frozen report linkage")
    try:
        if prediction["outputs"]["predictions"]["sha256"] != roster_sha256:
            raise ValueError("roster must exactly match frozen prediction output bytes")
        if prediction["inputs"]["model_manifest"]["sha256"] != model_sha256:
            raise ValueError("prediction and model manifest identities do not match")
        if prediction["score_semantics"]["primary"] != "reproducibility_score":
            raise ValueError("prediction primary score must be reproducibility_score")
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "prediction manifest is missing required provenance linkage"
        ) from exc


def _allocation_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, sep="\t", lineterminator="\n").encode("utf-8")


def freeze_prospective_selection(
    roster_path: str | Path,
    output_dir: str | Path,
    *,
    expected_roster_sha256: str,
    model_manifest_path: str | Path,
    expected_model_manifest_sha256: str,
    prediction_manifest_path: str | Path,
    expected_prediction_manifest_sha256: str,
    sap_path: str | Path,
    expected_sap_sha256: str,
    code_revision: str,
    created_at_utc: datetime,
    top_k: int,
    random_comparison_n: int,
    random_seed: int,
    model_score_column: str = "reproducibility_score",
    baseline_score_column: str = "screen_signal_rank",
    baseline_higher_is_better: bool = False,
) -> dict[str, object]:
    """Publish an outcome-free allocation bundle with pinned upstream artifacts."""
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"prospective selection output already exists: {output}")
    if not isinstance(code_revision, str) or not re.fullmatch(
        r"[0-9a-f]{40}", code_revision
    ):
        raise ValueError("code_revision must be a full lowercase Git SHA")
    if not isinstance(created_at_utc, datetime) or created_at_utc.tzinfo is None:
        raise ValueError("created_at_utc must include a timezone")
    if created_at_utc.utcoffset() is None:
        raise ValueError("created_at_utc must include a timezone")
    roster_path = Path(roster_path)
    suffix = roster_path.suffix.lower()
    sources = {
        "roster": (roster_path, expected_roster_sha256, f"source_roster{suffix}"),
        "model_manifest": (
            Path(model_manifest_path),
            expected_model_manifest_sha256,
            "model_manifest.json",
        ),
        "prediction_manifest": (
            Path(prediction_manifest_path),
            expected_prediction_manifest_sha256,
            "prediction_manifest.json",
        ),
        "analysis_plan": (Path(sap_path), expected_sap_sha256, "analysis_plan.source"),
    }
    content = {
        role: _read_pinned(path, expected, role)
        for role, (path, expected, _) in sources.items()
    }
    if not content["analysis_plan"].strip():
        raise ValueError("analysis plan source cannot be empty")
    _validate_linkage(
        content["prediction_manifest"],
        content["model_manifest"],
        roster_sha256=expected_roster_sha256,
        model_sha256=expected_model_manifest_sha256,
        roster=_read_roster(content["roster"], suffix),
        created_at_utc=created_at_utc.astimezone(UTC).isoformat(),
        baseline_score_column=baseline_score_column,
        baseline_higher_is_better=baseline_higher_is_better,
    )
    result = build_prospective_selection(
        _read_roster(content["roster"], suffix),
        top_k=top_k,
        random_comparison_n=random_comparison_n,
        random_seed=random_seed,
        model_score_column=model_score_column,
        baseline_score_column=baseline_score_column,
        baseline_higher_is_better=baseline_higher_is_better,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        files = {}
        for role, (_, expected, filename) in sources.items():
            (staging / filename).write_bytes(content[role])
            files[role] = {
                "filename": filename,
                "sha256": expected,
                "byte_size": len(content[role]),
            }
        for role, filename, body in (
            ("allocation", ALLOCATION_FILENAME, _allocation_bytes(result.allocation)),
            (
                "assay_roster",
                ASSAY_ROSTER_FILENAME,
                _allocation_bytes(result.assay_roster),
            ),
            ("summary", SUMMARY_FILENAME, _json_bytes(result.summary)),
        ):
            (staging / filename).write_bytes(body)
            files[role] = {
                "filename": filename,
                "sha256": _digest(body),
                "byte_size": len(body),
            }
        manifest: dict[str, object] = {
            "schema_version": 1,
            "bundle_type": "frozen_prospective_selection",
            "status": "allocated_awaiting_wet_lab_outcomes",
            "package_version": __version__,
            "code_revision": code_revision,
            "code_revision_is_unverified_declaration": True,
            "created_at_utc": created_at_utc.astimezone(UTC).isoformat(),
            "parameters": result.summary["parameters"],
            "files": files,
            "scientific_boundary": result.summary["interpretation"],
            "analysis_plan_approval_or_completeness_not_verified": True,
            "upstream_manifest_identity_verified_but_model_not_deserialized": True,
            "external_manifest_hash_anchor_required_for_identity": True,
        }
        (staging / MANIFEST_FILENAME).write_bytes(_json_bytes(manifest))
        verify_prospective_selection(staging)
        for role, (path, expected, _) in sources.items():
            _read_pinned(path, expected, role)
        _publish_directory_noreplace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def verify_prospective_selection(
    bundle_dir: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Check all hashes and reproduce allocation without reading any outcomes."""
    bundle = Path(bundle_dir)
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("prospective bundle must be a regular non-symlink directory")
    path = bundle / MANIFEST_FILENAME
    if path.is_symlink() or not path.is_file():
        raise ValueError("prospective manifest must be a regular non-symlink file")
    manifest_content = path.read_bytes()
    if expected_manifest_sha256 is not None:
        _check_hash(expected_manifest_sha256, "expected_manifest_sha256")
        if _digest(manifest_content) != expected_manifest_sha256:
            raise ValueError("prospective manifest identity hash mismatch")
    manifest = _load_json(manifest_content, "prospective manifest")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
        or manifest.get("bundle_type") != "frozen_prospective_selection"
    ):
        raise ValueError("unsupported prospective selection manifest")
    if manifest.get("status") != "allocated_awaiting_wet_lab_outcomes":
        raise ValueError("unsupported prospective selection status")
    for field in (
        "code_revision_is_unverified_declaration",
        "analysis_plan_approval_or_completeness_not_verified",
        "upstream_manifest_identity_verified_but_model_not_deserialized",
        "external_manifest_hash_anchor_required_for_identity",
    ):
        if manifest.get(field) is not True:
            raise ValueError(f"prospective manifest boundary must be true: {field}")
    revision = manifest.get("code_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("prospective code_revision must be a full lowercase Git SHA")
    files = manifest.get("files")
    expected_roles = {
        "roster",
        "model_manifest",
        "prediction_manifest",
        "analysis_plan",
        "allocation",
        "assay_roster",
        "summary",
    }
    if not isinstance(files, dict) or set(files) != expected_roles:
        raise ValueError("prospective manifest contains invalid file roles")
    fixed_names = {
        "model_manifest": "model_manifest.json",
        "prediction_manifest": "prediction_manifest.json",
        "analysis_plan": "analysis_plan.source",
        "allocation": ALLOCATION_FILENAME,
        "assay_roster": ASSAY_ROSTER_FILENAME,
        "summary": SUMMARY_FILENAME,
    }
    content = {}
    names = {MANIFEST_FILENAME}
    for role, record in files.items():
        if not isinstance(record, dict):
            raise ValueError("prospective file record must be an object")
        filename = record.get("filename")
        allowed = (
            {"source_roster.csv", "source_roster.tsv", "source_roster.txt"}
            if role == "roster"
            else {fixed_names[role]}
        )
        if filename not in allowed:
            raise ValueError(
                f"unsafe or invalid prospective bundle filename for {role}"
            )
        names.add(filename)
        body = _read_pinned(bundle / filename, record.get("sha256"), role)
        if isinstance(record.get("byte_size"), bool) or record.get("byte_size") != len(
            body
        ):
            raise ValueError(f"{role} byte_size mismatch")
        content[role] = body
    if {entry.name for entry in bundle.iterdir()} != names:
        raise ValueError("prospective bundle contains unexpected files")
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != {
        "top_k",
        "random_comparison_n",
        "random_seed",
        "model_score_column",
        "baseline_score_column",
        "baseline_higher_is_better",
    }:
        raise ValueError("prospective selection parameters are invalid")
    _validate_linkage(
        content["prediction_manifest"],
        content["model_manifest"],
        roster_sha256=files["roster"]["sha256"],
        model_sha256=files["model_manifest"]["sha256"],
        roster=_read_roster(
            content["roster"], Path(files["roster"]["filename"]).suffix
        ),
        created_at_utc=manifest.get("created_at_utc"),
        baseline_score_column=parameters["baseline_score_column"],
        baseline_higher_is_better=parameters["baseline_higher_is_better"],
    )
    rebuilt = build_prospective_selection(
        _read_roster(content["roster"], Path(files["roster"]["filename"]).suffix),
        **parameters,
    )
    if _allocation_bytes(rebuilt.allocation) != content["allocation"]:
        raise ValueError("allocation does not reproduce from the pinned roster")
    if _allocation_bytes(rebuilt.assay_roster) != content["assay_roster"]:
        raise ValueError("assay roster does not reproduce from the pinned roster")
    if _json_bytes(rebuilt.summary) != content["summary"]:
        raise ValueError("selection summary does not reproduce from the pinned roster")
    if manifest.get("scientific_boundary") != rebuilt.summary["interpretation"]:
        raise ValueError("prospective scientific boundaries do not match the design")
    return manifest
