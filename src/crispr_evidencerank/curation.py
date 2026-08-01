"""Outcome-blind selection of fixed full-text curation batches."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

from .contracts import (
    ReviewComparisonRecord,
    ReviewComparisonStatus,
    ReviewEvidenceLevel,
    validate_records,
)
from .io import read_table

FORBIDDEN_SELECTION_COLUMNS = frozenset(
    {
        "author_hit",
        "effect",
        "fdr",
        "hit",
        "label_code",
        "number_of_hits",
        "p_value",
        "score",
        "testing_status",
        "validation_outcome",
        "validation_status",
    }
)

CURATION_QUEUE_STRING_COLUMNS = {
    "queue_id": "string",
    "source_version": "string",
    "bucket": "string",
    "screen_id": "string",
    "external_screen_id": "string",
    "source_id": "string",
    "source_type": "string",
    "source_family_id": "string",
    "author": "string",
    "screen_name": "string",
    "experimental_setup": "string",
    "condition_name": "string",
    "cell_line": "string",
    "reason_codes": "string",
}

CURATION_QUEUE_INTEGER_COLUMNS = (
    "queue_rank",
    "source_round",
    "policy_version",
    "scope_unknown_count",
    "metadata_completeness_count",
)

REVIEW_LINKAGE_FIELDS = (
    "queue_id",
    "queue_rank",
    "screen_id",
    "external_screen_id",
    "source_id",
    "source_family_id",
)

REVIEW_GENE_FIELDS = {
    "candidate_v3_genes": ReviewEvidenceLevel.CANDIDATE_V3,
    "candidate_v2_genes": ReviewEvidenceLevel.CANDIDATE_V2,
    "candidate_v1_genes": ReviewEvidenceLevel.CANDIDATE_V1,
    "nonqualifying_validation_genes": ReviewEvidenceLevel.NONQUALIFYING,
}

CRITICAL_REVIEW_FIELDS = (
    "doi",
    "paper_url",
    "full_text_url",
    "supplement_url",
    "full_text_reviewed",
    "supplement_review",
    "scope_outcome",
    "design_review",
    "sample_map_review",
    "screen_model",
    "library_design",
    "treatment_contrast",
    "screen_replication",
    "analysis_method",
    "quantitative_asset_family_id",
    "raw_data_family_id",
    "quantitative_data_status",
    "quantitative_asset_locator",
    "data_accession",
    "rights_outcome",
    "rights_basis",
    "validation_status",
    "validation_source_locator",
    "disposition",
    "blocker_codes",
)

REVIEW_COMPARISON_COLUMNS = tuple(ReviewComparisonRecord.model_fields)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _read_table_text_cells(path: Path) -> pd.DataFrame:
    """Read textual cells without treating literal values such as ``NA`` as null."""

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype="string", keep_default_na=False)
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(
            path,
            sep="\t",
            dtype="string",
            keep_default_na=False,
        )
    raise ValueError(f"unsupported table extension: {path.suffix}")


def select_curation_batch(
    queue: pd.DataFrame,
    *,
    start_rank: int,
    batch_size: int,
    require_unique_source_families: bool = True,
) -> pd.DataFrame:
    """Select a contiguous queue slice without inspecting outcome columns."""

    if start_rank < 1:
        raise ValueError("start_rank must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    normalized_columns = {str(column).strip().lower() for column in queue.columns}
    forbidden = sorted(normalized_columns & FORBIDDEN_SELECTION_COLUMNS)
    if forbidden:
        raise ValueError(f"outcome-bearing columns are forbidden: {forbidden}")

    required = {
        "queue_id",
        "queue_rank",
        "screen_id",
        "source_family_id",
        "source_version",
        "policy_version",
    }
    missing = sorted(required - set(queue.columns))
    if missing:
        raise ValueError(f"curation queue is missing columns: {missing}")

    valid_queue, queue_errors = validate_records(queue, "curation_queue")
    if not queue_errors.empty:
        details = queue_errors.head(5).to_dict(orient="records")
        raise ValueError(f"curation queue failed contract validation: {details}")

    ranked = valid_queue.copy()
    ranked["queue_rank"] = pd.to_numeric(ranked["queue_rank"], errors="raise")
    if ranked["queue_rank"].duplicated().any():
        raise ValueError("queue_rank must be unique")
    if ranked["queue_id"].duplicated().any():
        raise ValueError("queue_id must be unique")

    stop_rank = start_rank + batch_size - 1
    selected = ranked.loc[ranked["queue_rank"].between(start_rank, stop_rank)].copy()
    selected = selected.sort_values("queue_rank", kind="stable").reset_index(drop=True)
    expected_ranks = list(range(start_rank, stop_rank + 1))
    if selected["queue_rank"].astype(int).tolist() != expected_ranks:
        raise ValueError("requested queue ranks are missing or non-contiguous")

    families = selected["source_family_id"].astype("string").str.strip()
    if families.isna().any() or families.eq("").any():
        raise ValueError("selected rows require source_family_id")
    if require_unique_source_families and families.duplicated().any():
        raise ValueError("selected rows must come from distinct source families")
    return selected


def write_curation_batch(
    queue_path: str | Path,
    output_dir: str | Path,
    *,
    batch_id: str,
    selected_date: date,
    start_rank: int,
    batch_size: int,
    require_unique_source_families: bool = True,
) -> dict[str, object]:
    """Write the frozen selection and a portable checksum manifest."""

    queue_path = Path(queue_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"curation batch already exists: {output_dir}")
    queue = read_table(queue_path, dtype=CURATION_QUEUE_STRING_COLUMNS)
    selected = select_curation_batch(
        queue,
        start_rank=start_rank,
        batch_size=batch_size,
        require_unique_source_families=require_unique_source_families,
    )
    selected["queue_rank"] = selected["queue_rank"].astype(int)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.staging-",
    ) as staging_name:
        staging_dir = Path(staging_name)
        selection_path = staging_dir / "selection.tsv"
        selected.to_csv(selection_path, sep="\t", index=False, lineterminator="\n")
        manifest: dict[str, object] = {
            "schema_version": 1,
            "batch_id": batch_id,
            "selected_date": selected_date.isoformat(),
            "selection_policy": "contiguous_queue_rank_outcome_blind_v1",
            "start_rank": start_rank,
            "end_rank": start_rank + batch_size - 1,
            "record_count": len(selected),
            "require_unique_source_families": require_unique_source_families,
            "source_queue": {
                "filename": queue_path.name,
                "sha256": _sha256(queue_path),
                "record_count": len(queue),
            },
            "selection": {
                "filename": selection_path.name,
                "sha256": _sha256(selection_path),
            },
            "forbidden_selection_columns": sorted(FORBIDDEN_SELECTION_COLUMNS),
        }
        manifest_path = staging_dir / "selection_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging_dir.replace(output_dir)
    return manifest


def build_review_manifest(
    reviews_path: str | Path,
    selection_path: str | Path,
) -> dict[str, object]:
    """Validate a downstream review table and derive its portable manifest."""

    reviews_path = Path(reviews_path)
    selection_path = Path(selection_path)
    reviews = read_table(reviews_path, dtype="string")
    selection = read_table(selection_path, dtype="string")
    valid, errors = validate_records(reviews, "full_text_review")
    if not errors.empty:
        details = errors.head(5).to_dict(orient="records")
        raise ValueError(f"full-text reviews failed contract validation: {details}")

    missing_selection = sorted(set(REVIEW_LINKAGE_FIELDS) - set(selection.columns))
    if missing_selection:
        raise ValueError(f"selection is missing linkage fields: {missing_selection}")
    if len(valid) != len(selection):
        raise ValueError("selection and review record counts differ")
    for field_name in REVIEW_LINKAGE_FIELDS:
        reviewed_values = valid[field_name].astype(str).tolist()
        selected_values = selection[field_name].astype(str).tolist()
        if reviewed_values != selected_values:
            raise ValueError(f"selection and reviews differ for {field_name}")

    batch_ids = valid["batch_id"].drop_duplicates().tolist()
    assessed_dates = valid["assessed_date"].drop_duplicates().tolist()
    if len(batch_ids) != 1 or len(assessed_dates) != 1:
        raise ValueError("one review manifest requires one batch_id and assessed_date")
    curator_count = int(valid["curator"].nunique())
    curation_status = (
        "single_curator_requires_independent_adjudication"
        if curator_count == 1
        else "multiple_curators_require_independent_adjudication"
    )

    def counts(column: str) -> dict[str, int]:
        observed = valid[column].value_counts().sort_index()
        return {str(key): int(value) for key, value in observed.items()}

    return {
        "schema_version": 1,
        "batch_id": str(batch_ids[0]),
        "assessed_date": str(assessed_dates[0]),
        "curation_status": curation_status,
        "curator_count": curator_count,
        "review_contract": "full_text_review",
        "record_count": len(valid),
        "full_text_reviewed_count": int(valid["full_text_reviewed"].sum()),
        "supplement_review_counts": counts("supplement_review"),
        "quantitative_data_status_counts": counts("quantitative_data_status"),
        "quantitative_asset_family_count": int(
            valid["quantitative_asset_family_id"].nunique()
        ),
        "raw_data_family_resolved_count": int(valid["raw_data_family_id"].nunique()),
        "validation_status_counts": counts("validation_status"),
        "disposition_counts": counts("disposition"),
        "benchmark_ready_count": 0,
        "selection": {
            "filename": selection_path.name,
            "sha256": _sha256(selection_path),
        },
        "reviews": {
            "filename": reviews_path.name,
            "sha256": _sha256(reviews_path),
        },
    }


def _validate_review_against_selection(
    reviews: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    role: str,
    allow_subset: bool = False,
) -> pd.DataFrame:
    normalized_selection = selection.copy()
    for column in CURATION_QUEUE_INTEGER_COLUMNS:
        if column in normalized_selection:
            normalized_selection[column] = pd.to_numeric(
                normalized_selection[column], errors="coerce"
            )
    if "full_gene_score_set_available" in normalized_selection:
        availability = normalized_selection["full_gene_score_set_available"]
        normalized_selection["full_gene_score_set_available"] = availability.map(
            lambda value: (
                {
                    "true": True,
                    "false": False,
                }.get(str(value).strip().lower(), value)
                if not pd.isna(value)
                else None
            )
        )
    valid_selection, selection_errors = validate_records(
        normalized_selection, "curation_queue"
    )
    if not selection_errors.empty:
        details = selection_errors.head(5).to_dict(orient="records")
        raise ValueError(f"selection failed contract validation: {details}")
    for identifier in ("queue_id", "queue_rank", "screen_id"):
        if valid_selection[identifier].duplicated().any():
            raise ValueError(f"selection {identifier} values must be unique")

    valid, errors = validate_records(reviews, "full_text_review")
    if not errors.empty:
        details = errors.head(5).to_dict(orient="records")
        raise ValueError(f"{role} reviews failed contract validation: {details}")
    if not allow_subset and len(valid) != len(valid_selection):
        raise ValueError(f"{role} review and selection record counts differ")
    if allow_subset and len(valid) > len(valid_selection):
        raise ValueError(f"{role} review has more rows than the selection")
    if valid["review_id"].duplicated().any():
        raise ValueError(f"{role} review_id values must be unique")
    if valid["screen_id"].duplicated().any():
        raise ValueError(f"{role} reviews require one row per selected screen")
    if not valid["full_text_reviewed"].all():
        raise ValueError(f"{role} reviews require full_text_reviewed=true")
    selection_by_queue = valid_selection.set_index("queue_id", drop=False)
    if allow_subset:
        unknown_queue_ids = sorted(
            set(valid["queue_id"].astype(str))
            - set(valid_selection["queue_id"].astype(str))
        )
        if unknown_queue_ids:
            raise ValueError(
                f"{role} reviews contain unselected queue IDs: {unknown_queue_ids}"
            )
    for field_name in REVIEW_LINKAGE_FIELDS:
        if field_name not in valid_selection:
            raise ValueError(f"selection is missing linkage field: {field_name}")
        reviewed_values = valid[field_name].astype(str).tolist()
        if allow_subset:
            selected_values = [
                str(selection_by_queue.loc[str(queue_id), field_name])
                for queue_id in valid["queue_id"].astype(str)
            ]
        else:
            selected_values = valid_selection[field_name].astype(str).tolist()
        if reviewed_values != selected_values:
            raise ValueError(f"{role} reviews and selection differ for {field_name}")
    return valid


def _explode_review_gene_evidence(
    reviews: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, object]]:
    evidence: dict[tuple[str, str], dict[str, object]] = {}
    for _, row in reviews.iterrows():
        for field_name, level in REVIEW_GENE_FIELDS.items():
            values = row.get(field_name)
            if values is None or pd.isna(values) or not str(values).strip():
                continue
            for gene_symbol in str(values).split("|"):
                key = (str(row["screen_id"]), gene_symbol)
                if key in evidence:
                    raise ValueError(
                        f"one review cannot assign multiple evidence levels to {key}"
                    )
                evidence[key] = {
                    "level": level,
                    "review_id": str(row["review_id"]),
                    "source_locator": str(row["validation_source_locator"]),
                    "batch_id": str(row["batch_id"]),
                    "queue_id": str(row["queue_id"]),
                    "queue_rank": int(row["queue_rank"]),
                    "external_screen_id": str(row["external_screen_id"]),
                }
    return evidence


def compare_full_text_reviews(
    primary_reviews: pd.DataFrame,
    secondary_reviews: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    assessed_date: date,
    allow_partial_secondary: bool = False,
) -> pd.DataFrame:
    """Compare two review sets without turning agreement into a final label."""

    comparison_date = pd.Timestamp(assessed_date).date()
    primary = _validate_review_against_selection(
        primary_reviews,
        selection,
        role="primary",
    )
    secondary = _validate_review_against_selection(
        secondary_reviews,
        selection,
        role="secondary",
        allow_subset=allow_partial_secondary,
    )
    primary_batches = set(primary["batch_id"].astype(str))
    secondary_batches = set(secondary["batch_id"].astype(str))
    if len(primary_batches) != 1 or len(secondary_batches) != 1:
        raise ValueError("each review set must contain exactly one batch_id")
    if primary_batches != secondary_batches:
        raise ValueError("primary and secondary reviews must use the same batch_id")
    if set(primary["review_id"].astype(str)) & set(secondary["review_id"].astype(str)):
        raise ValueError("primary and secondary review IDs must be disjoint")
    primary_curators = set(primary["curator"].astype(str))
    secondary_curators = set(secondary["curator"].astype(str))
    if primary_curators & secondary_curators:
        raise ValueError("primary and secondary curator identities must be disjoint")
    review_dates = pd.to_datetime(
        pd.concat([primary["assessed_date"], secondary["assessed_date"]]),
        errors="raise",
    ).dt.date
    if any(review_date > comparison_date for review_date in review_dates):
        raise ValueError("comparison assessed_date cannot precede a source review")

    primary_evidence = _explode_review_gene_evidence(primary)
    secondary_evidence = _explode_review_gene_evidence(secondary)
    secondary_screen_ids = set(secondary["screen_id"].astype(str))
    primary_evidence = {
        key: value
        for key, value in primary_evidence.items()
        if key[0] in secondary_screen_ids
    }
    keys = set(primary_evidence) | set(secondary_evidence)
    if not keys:
        return pd.DataFrame(columns=REVIEW_COMPARISON_COLUMNS)

    absent = ReviewEvidenceLevel.NOT_ANNOTATED
    records: list[dict[str, object]] = []
    for screen_id, gene_symbol in keys:
        primary_item = primary_evidence.get((screen_id, gene_symbol))
        secondary_item = secondary_evidence.get((screen_id, gene_symbol))
        anchor = primary_item or secondary_item
        if anchor is None:  # pragma: no cover - guarded by the union above
            raise RuntimeError("missing review comparison anchor")
        primary_level = primary_item["level"] if primary_item else absent
        secondary_level = secondary_item["level"] if secondary_item else absent
        if primary_level == secondary_level:
            status = ReviewComparisonStatus.PROVISIONAL_AGREEMENT
        elif absent in {primary_level, secondary_level}:
            status = ReviewComparisonStatus.SINGLE_CURATOR_ONLY
        else:
            status = ReviewComparisonStatus.LABEL_DISAGREEMENT
        primary_row = primary.loc[primary["screen_id"].astype(str).eq(screen_id)].iloc[
            0
        ]
        secondary_row = secondary.loc[
            secondary["screen_id"].astype(str).eq(screen_id)
        ].iloc[0]
        comparison_identity = json.dumps(
            [
                anchor["batch_id"],
                screen_id,
                gene_symbol,
                str(primary_row["review_id"]),
                str(secondary_row["review_id"]),
                comparison_date.isoformat(),
            ],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        comparison_digest = hashlib.sha256(comparison_identity).hexdigest()
        records.append(
            {
                "comparison_id": (
                    f"{anchor['batch_id']}:review-comparison:sha256:{comparison_digest}"
                ),
                "batch_id": anchor["batch_id"],
                "queue_id": anchor["queue_id"],
                "queue_rank": anchor["queue_rank"],
                "screen_id": screen_id,
                "external_screen_id": anchor["external_screen_id"],
                "gene_symbol": gene_symbol,
                "primary_review_id": str(primary_row["review_id"]),
                "secondary_review_id": str(secondary_row["review_id"]),
                "primary_evidence_level": primary_level.value,
                "secondary_evidence_level": secondary_level.value,
                "comparison_status": status.value,
                "primary_source_locator": str(primary_row["validation_source_locator"]),
                "secondary_source_locator": str(
                    secondary_row["validation_source_locator"]
                ),
                "human_adjudication_required": True,
                "assessed_date": comparison_date,
                "notes": (
                    "Comparison only; human adjudication is required and no "
                    "benchmark label is released."
                ),
            }
        )
    comparison = pd.DataFrame.from_records(records).sort_values(
        ["queue_rank", "gene_symbol"], kind="stable"
    )
    comparison = comparison.reset_index(drop=True)
    valid, errors = validate_records(comparison, "review_comparison")
    if not errors.empty:
        details = errors.head(5).to_dict(orient="records")
        raise ValueError(f"review comparison failed contract validation: {details}")
    return valid


def build_dual_review_manifest(
    primary_reviews_path: str | Path,
    secondary_reviews_path: str | Path,
    selection_path: str | Path,
    comparison_path: str | Path,
    *,
    assessed_date: date | None = None,
) -> dict[str, object]:
    """Validate a dual-review comparison and derive a checksum manifest."""

    primary_reviews_path = Path(primary_reviews_path)
    secondary_reviews_path = Path(secondary_reviews_path)
    selection_path = Path(selection_path)
    comparison_path = Path(comparison_path)
    primary = read_table(primary_reviews_path, dtype="string")
    secondary = read_table(secondary_reviews_path, dtype="string")
    selection = read_table(selection_path, dtype="string")
    observed = read_table(comparison_path, dtype="string")
    if observed.empty:
        if observed.columns.tolist() != list(REVIEW_COMPARISON_COLUMNS):
            raise ValueError("empty review comparison columns are not canonical")
        if assessed_date is None:
            raise ValueError(
                "assessed_date is required for an empty gene-level comparison"
            )
        comparison_date = pd.Timestamp(assessed_date).date()
        valid_comparison = observed.copy()
    else:
        valid_comparison, errors = validate_records(observed, "review_comparison")
        if not errors.empty:
            details = errors.head(5).to_dict(orient="records")
            raise ValueError(f"review comparison failed contract validation: {details}")
        dates = valid_comparison["assessed_date"].drop_duplicates().tolist()
        if len(dates) != 1:
            raise ValueError("one review comparison requires one assessed_date")
        comparison_date = pd.Timestamp(dates[0]).date()
        if assessed_date is not None and pd.Timestamp(assessed_date).date() != (
            comparison_date
        ):
            raise ValueError("provided assessed_date differs from the comparison")
    expected = compare_full_text_reviews(
        primary,
        secondary,
        selection,
        assessed_date=comparison_date,
        allow_partial_secondary=len(secondary) < len(selection),
    )

    comparable_columns = expected.columns.tolist()
    if observed.columns.tolist() != comparable_columns:
        raise ValueError("review comparison columns are not canonical")
    expected_text = expected.fillna("").astype(str)
    observed_text = valid_comparison.fillna("").astype(str)
    if not observed_text.equals(expected_text):
        raise ValueError("review comparison is not the deterministic derivation")

    primary_valid = _validate_review_against_selection(
        primary,
        selection,
        role="primary",
    )
    secondary_valid = _validate_review_against_selection(
        secondary,
        selection,
        role="secondary",
        allow_subset=len(secondary) < len(selection),
    )
    primary_subset = primary_valid.set_index("screen_id").loc[
        secondary_valid["screen_id"]
    ]
    secondary_aligned = secondary_valid.set_index("screen_id")
    field_disagreements = {
        field_name: int(
            primary_subset[field_name]
            .fillna("")
            .astype(str)
            .ne(secondary_aligned[field_name].fillna("").astype(str))
            .sum()
        )
        for field_name in CRITICAL_REVIEW_FIELDS
    }

    def counts(column: str) -> dict[str, int]:
        observed_counts = valid_comparison[column].value_counts().sort_index()
        return {str(key): int(value) for key, value in observed_counts.items()}

    return {
        "schema_version": 1,
        "batch_id": str(primary_valid["batch_id"].iloc[0]),
        "assessed_date": comparison_date.isoformat(),
        "status": (
            "dual_review_complete_requires_human_adjudication"
            if len(secondary_valid) == len(selection)
            else "partial_dual_review_requires_completion_and_human_adjudication"
        ),
        "benchmark_ready_count": 0,
        "selected_screen_count": len(selection),
        "second_reviewed_screen_count": len(secondary_valid),
        "pending_second_review_screen_count": len(selection) - len(secondary_valid),
        "compared_gene_count": len(valid_comparison),
        "comparison_status_counts": counts("comparison_status"),
        "critical_field_disagreement_counts": field_disagreements,
        "primary_curators": sorted(set(primary_valid["curator"].astype(str))),
        "secondary_curators": sorted(set(secondary_valid["curator"].astype(str))),
        "selection": {
            "filename": selection_path.name,
            "sha256": _sha256(selection_path),
        },
        "primary_reviews": {
            "filename": primary_reviews_path.name,
            "sha256": _sha256(primary_reviews_path),
        },
        "secondary_reviews": {
            "filename": secondary_reviews_path.name,
            "sha256": _sha256(secondary_reviews_path),
        },
        "comparison": {
            "filename": comparison_path.name,
            "sha256": _sha256(comparison_path),
        },
    }


def _require_expected_sha256(value: str, *, field_name: str) -> str:
    expected = value.strip()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return expected


def _derive_secondary_review_progress_manifest(
    progress_reviews_path: Path,
    primary_reviews_path: Path,
    selection_path: Path,
    frozen_secondary_reviews_path: Path,
    frozen_comparison_path: Path,
    predecessor_manifest_path: Path,
    *,
    assessed_date: date,
    expected_predecessor_manifest_sha256: str,
) -> dict[str, object]:
    """Derive progress from private immutable snapshots."""

    canonical_names = {
        "progress": "reviews_curator_2_completion_progress.tsv",
        "primary": "reviews.tsv",
        "selection": "selection.tsv",
        "frozen": "reviews_curator_2_partial.tsv",
        "comparison": "review_comparison_partial.tsv",
        "predecessor": "dual_review_manifest_partial.json",
    }
    observed_names = {
        "progress": progress_reviews_path.name,
        "primary": primary_reviews_path.name,
        "selection": selection_path.name,
        "frozen": frozen_secondary_reviews_path.name,
        "comparison": frozen_comparison_path.name,
        "predecessor": predecessor_manifest_path.name,
    }
    for component, expected_name in canonical_names.items():
        if observed_names[component] != expected_name:
            raise ValueError(
                f"secondary review progress requires canonical {component} "
                f"filename {expected_name!r}"
            )

    expected_predecessor_sha = _require_expected_sha256(
        expected_predecessor_manifest_sha256,
        field_name="expected_predecessor_manifest_sha256",
    )
    input_sha = {
        "progress": _sha256(progress_reviews_path),
        "primary": _sha256(primary_reviews_path),
        "selection": _sha256(selection_path),
        "frozen": _sha256(frozen_secondary_reviews_path),
        "comparison": _sha256(frozen_comparison_path),
        "predecessor": _sha256(predecessor_manifest_path),
    }
    if input_sha["predecessor"] != expected_predecessor_sha:
        raise ValueError(
            "predecessor checkpoint manifest SHA-256 does not match expected"
        )
    try:
        predecessor = json.loads(predecessor_manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "predecessor checkpoint manifest is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(predecessor, dict):
        raise ValueError("predecessor checkpoint manifest must be a JSON object")
    try:
        predecessor_timestamp = pd.Timestamp(predecessor["assessed_date"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "predecessor checkpoint requires a valid assessed_date"
        ) from exc
    if pd.isna(predecessor_timestamp):
        raise ValueError("predecessor checkpoint requires a valid assessed_date")
    predecessor_date = predecessor_timestamp.date()
    derived_predecessor = build_dual_review_manifest(
        primary_reviews_path,
        frozen_secondary_reviews_path,
        selection_path,
        frozen_comparison_path,
        assessed_date=predecessor_date,
    )
    if predecessor != derived_predecessor:
        raise ValueError(
            "predecessor checkpoint manifest is not the exact deterministic derivation"
        )
    if derived_predecessor["status"] != (
        "partial_dual_review_requires_completion_and_human_adjudication"
    ):
        raise ValueError("predecessor checkpoint is not an incomplete dual review")

    primary = read_table(primary_reviews_path, dtype="string")
    selection = read_table(selection_path, dtype="string")
    frozen = read_table(frozen_secondary_reviews_path, dtype="string")
    progress = read_table(progress_reviews_path, dtype="string")
    primary_valid = _validate_review_against_selection(
        primary,
        selection,
        role="primary",
    )
    frozen_valid = _validate_review_against_selection(
        frozen,
        selection,
        role="frozen secondary",
        allow_subset=True,
    )
    progress_valid = _validate_review_against_selection(
        progress,
        selection,
        role="completion progress",
        allow_subset=True,
    )
    if progress_valid.empty:
        raise ValueError("secondary review progress requires at least one row")
    forbidden_release_columns = {"benchmark_ready", "label_code"}
    if forbidden_release_columns & set(progress.columns):
        raise ValueError("secondary review progress contains released label columns")
    frozen_ids = set(frozen_valid["queue_id"].astype(str))
    progress_ids = set(progress_valid["queue_id"].astype(str))
    overlap = sorted(frozen_ids & progress_ids)
    if overlap:
        raise ValueError(f"secondary review progress overlaps frozen rows: {overlap}")
    if set(frozen_valid["review_id"].astype(str)) & set(
        progress_valid["review_id"].astype(str)
    ):
        raise ValueError("secondary review progress reuses a frozen review_id")
    frozen_batches = set(frozen_valid["batch_id"].astype(str))
    progress_batches = set(progress_valid["batch_id"].astype(str))
    if len(frozen_batches) != 1 or progress_batches != frozen_batches:
        raise ValueError(
            "frozen and progress reviews must use the same single batch_id"
        )
    frozen_batch = next(iter(frozen_batches))
    frozen_curators = sorted(set(frozen_valid["curator"].astype(str)))
    if derived_predecessor["secondary_curators"] != frozen_curators:
        raise ValueError(
            "predecessor secondary_curators do not match frozen secondary rows"
        )
    predecessor_primary_curators = set(primary_valid["curator"].astype(str))
    progress_curators = set(progress_valid["curator"].astype(str))
    primary_overlap = sorted(progress_curators & predecessor_primary_curators)
    if primary_overlap:
        raise ValueError(
            "completion progress curator identities overlap primary curators: "
            f"{primary_overlap}"
        )
    primary_review_ids = set(primary_valid["review_id"].astype(str))
    progress_review_ids = set(progress_valid["review_id"].astype(str))
    reused_primary_ids = sorted(primary_review_ids & progress_review_ids)
    if reused_primary_ids:
        raise ValueError(
            f"completion progress reuses primary review IDs: {reused_primary_ids}"
        )

    selection_ids = set(selection["queue_id"].astype(str))
    covered_ids = frozen_ids | progress_ids
    pending_ids = selection_ids - covered_ids
    if not pending_ids:
        raise ValueError(
            "a complete second review must use the completed dual-review bundle"
        )
    manifest_date = pd.Timestamp(assessed_date).date()
    if predecessor_date > manifest_date:
        raise ValueError(
            "progress assessed_date cannot precede the predecessor checkpoint"
        )
    review_dates = pd.to_datetime(
        pd.concat([frozen_valid["assessed_date"], progress_valid["assessed_date"]]),
        errors="raise",
    ).dt.date
    if any(review_date > manifest_date for review_date in review_dates):
        raise ValueError("progress assessed_date cannot precede a source review")

    rank_by_queue = {
        str(row["queue_id"]): int(row["queue_rank"]) for _, row in selection.iterrows()
    }

    def ranks(queue_ids: set[str]) -> list[int]:
        return sorted(rank_by_queue[queue_id] for queue_id in queue_ids)

    return {
        "schema_version": 1,
        "batch_id": frozen_batch,
        "assessed_date": manifest_date.isoformat(),
        "status": (
            "secondary_review_completion_progress_requires_remaining_review_"
            "and_human_adjudication"
        ),
        "human_adjudication_required": True,
        "adjudicated_gene_count": 0,
        "released_label_count": 0,
        "benchmark_ready_count": 0,
        "selected_screen_count": len(selection),
        "frozen_second_review_screen_count": len(frozen_valid),
        "completion_progress_screen_count": len(progress_valid),
        "second_reviewed_screen_count": len(covered_ids),
        "pending_second_review_screen_count": len(pending_ids),
        "pending_queue_ranks": ranks(pending_ids),
        "selection": {
            "filename": selection_path.name,
            "sha256": input_sha["selection"],
        },
        "predecessor_checkpoint": {
            "manifest": {
                "filename": predecessor_manifest_path.name,
                "sha256": input_sha["predecessor"],
            },
            "secondary_reviews": {
                "filename": frozen_secondary_reviews_path.name,
                "sha256": input_sha["frozen"],
            },
            "queue_ranks": ranks(frozen_ids),
        },
        "completion_progress": {
            "filename": progress_reviews_path.name,
            "sha256": input_sha["progress"],
            "queue_ranks": ranks(progress_ids),
        },
    }


def build_secondary_review_progress_manifest(
    progress_reviews_path: str | Path,
    primary_reviews_path: str | Path,
    selection_path: str | Path,
    frozen_secondary_reviews_path: str | Path,
    frozen_comparison_path: str | Path,
    predecessor_manifest_path: str | Path,
    *,
    assessed_date: date,
    expected_predecessor_manifest_sha256: str,
) -> dict[str, object]:
    """Record an authenticated incomplete addendum from one-read snapshots.

    This manifest is intentionally not a dual-review comparison.  It preserves
    independently completed rows while at least one selected screen is still
    awaiting a second review, and it cannot release labels or readiness.
    """

    input_paths = {
        "progress": Path(progress_reviews_path),
        "primary": Path(primary_reviews_path),
        "selection": Path(selection_path),
        "frozen": Path(frozen_secondary_reviews_path),
        "comparison": Path(frozen_comparison_path),
        "predecessor": Path(predecessor_manifest_path),
    }
    input_content = {name: path.read_bytes() for name, path in input_paths.items()}
    with tempfile.TemporaryDirectory(prefix=".secondary-review-progress-") as temp_name:
        snapshots_dir = Path(temp_name)
        snapshot_paths: dict[str, Path] = {}
        for name, content in input_content.items():
            snapshot_path = snapshots_dir / name / input_paths[name].name
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_bytes(content)
            snapshot_paths[name] = snapshot_path
        return _derive_secondary_review_progress_manifest(
            snapshot_paths["progress"],
            snapshot_paths["primary"],
            snapshot_paths["selection"],
            snapshot_paths["frozen"],
            snapshot_paths["comparison"],
            snapshot_paths["predecessor"],
            assessed_date=assessed_date,
            expected_predecessor_manifest_sha256=(expected_predecessor_manifest_sha256),
        )


def write_dual_review_bundle(
    primary_reviews_path: str | Path,
    secondary_reviews_path: str | Path,
    selection_path: str | Path,
    output_dir: str | Path,
    *,
    assessed_date: date,
    allow_partial_secondary: bool = False,
) -> dict[str, object]:
    """Atomically write a deterministic comparison and checksum manifest."""

    primary_reviews_path = Path(primary_reviews_path)
    secondary_reviews_path = Path(secondary_reviews_path)
    selection_path = Path(selection_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"dual-review bundle already exists: {output_dir}")
    primary = read_table(primary_reviews_path, dtype="string")
    secondary = read_table(secondary_reviews_path, dtype="string")
    selection = read_table(selection_path, dtype="string")
    is_partial = len(secondary) < len(selection)
    if is_partial and not allow_partial_secondary:
        raise ValueError(
            "partial secondary review requires allow_partial_secondary=True"
        )
    comparison = compare_full_text_reviews(
        primary,
        secondary,
        selection,
        assessed_date=assessed_date,
        allow_partial_secondary=is_partial,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.staging-",
    ) as staging_name:
        staging_dir = Path(staging_name)
        comparison_path = staging_dir / "review_comparison.tsv"
        comparison.to_csv(
            comparison_path,
            sep="\t",
            index=False,
            lineterminator="\n",
        )
        manifest = build_dual_review_manifest(
            primary_reviews_path,
            secondary_reviews_path,
            selection_path,
            comparison_path,
            assessed_date=assessed_date,
        )
        (staging_dir / "dual_review_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging_dir.replace(output_dir)
    return manifest


def write_completed_dual_review_bundle(
    primary_reviews_path: str | Path,
    completion_reviews_path: str | Path,
    progress_reviews_path: str | Path,
    progress_manifest_path: str | Path,
    selection_path: str | Path,
    partial_secondary_reviews_path: str | Path,
    partial_comparison_path: str | Path,
    partial_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    assessed_date: date,
    expected_checkpoint_manifest_sha256: str,
    expected_progress_manifest_sha256: str,
) -> dict[str, object]:
    """Complete a frozen partial dual-review checkpoint as one atomic bundle.

    All input paths are read once after acquiring a sibling publisher lock;
    validation and publication then use only private snapshots.  The final
    directory rename is atomic for cooperating CLI writers.  Out-of-band
    writers are detected by a second destination check immediately before the
    rename, but they are outside the lock's cooperation guarantee.
    """

    input_paths = {
        "primary_reviews": Path(primary_reviews_path),
        "completion_reviews": Path(completion_reviews_path),
        "progress_reviews": Path(progress_reviews_path),
        "progress_manifest": Path(progress_manifest_path),
        "selection": Path(selection_path),
        "partial_secondary_reviews": Path(partial_secondary_reviews_path),
        "partial_comparison": Path(partial_comparison_path),
        "partial_manifest": Path(partial_manifest_path),
    }
    if input_paths["completion_reviews"].suffix.lower() != ".tsv":
        raise ValueError("completion reviews must use the .tsv format")
    if input_paths["progress_reviews"].suffix.lower() != ".tsv":
        raise ValueError("progress reviews must use the .tsv format")
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.publish.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"completed dual-review publisher lock already exists: {lock_path}"
        ) from exc

    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        if _path_lexists(output_dir):
            raise FileExistsError(
                f"completed dual-review bundle already exists: {output_dir}"
            )

        # These are the only reads of caller-owned input paths.  Every later
        # operation uses files reconstructed from this immutable byte snapshot.
        input_content = {name: path.read_bytes() for name, path in input_paths.items()}

        with tempfile.TemporaryDirectory(
            dir=output_dir.parent,
            prefix=f".{output_dir.name}.work-",
        ) as work_name:
            work_dir = Path(work_name)
            snapshots_dir = work_dir / "snapshots"
            snapshot_paths: dict[str, Path] = {}
            for name, content in input_content.items():
                snapshot_path = snapshots_dir / name / input_paths[name].name
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                snapshot_path.write_bytes(content)
                snapshot_paths[name] = snapshot_path

            expected_checkpoint_sha = _require_expected_sha256(
                expected_checkpoint_manifest_sha256,
                field_name="expected_checkpoint_manifest_sha256",
            )
            expected_progress_sha = _require_expected_sha256(
                expected_progress_manifest_sha256,
                field_name="expected_progress_manifest_sha256",
            )
            input_sha = {name: _sha256(path) for name, path in snapshot_paths.items()}
            if input_sha["partial_manifest"] != expected_checkpoint_sha:
                raise ValueError(
                    "partial checkpoint manifest SHA-256 does not match expected"
                )
            if input_sha["progress_manifest"] != expected_progress_sha:
                raise ValueError(
                    "progress checkpoint manifest SHA-256 does not match expected"
                )

            # Parse only after authenticating the exact checkpoint manifest bytes.
            try:
                observed_partial_manifest = json.loads(
                    snapshot_paths["partial_manifest"].read_text(encoding="utf-8")
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "partial checkpoint manifest is not valid UTF-8 JSON"
                ) from exc
            if not isinstance(observed_partial_manifest, dict):
                raise ValueError("partial checkpoint manifest must be a JSON object")
            try:
                predecessor_timestamp = pd.Timestamp(
                    observed_partial_manifest["assessed_date"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "partial checkpoint manifest requires a valid assessed_date"
                ) from exc
            if pd.isna(predecessor_timestamp):
                raise ValueError(
                    "partial checkpoint manifest requires a valid assessed_date"
                )
            predecessor_date = predecessor_timestamp.date()
            completion_date = pd.Timestamp(assessed_date).date()
            if completion_date < predecessor_date:
                raise ValueError(
                    "completed assessed_date cannot precede the partial checkpoint"
                )

            derived_partial_manifest = build_dual_review_manifest(
                snapshot_paths["primary_reviews"],
                snapshot_paths["partial_secondary_reviews"],
                snapshot_paths["selection"],
                snapshot_paths["partial_comparison"],
                assessed_date=predecessor_date,
            )
            if observed_partial_manifest != derived_partial_manifest:
                raise ValueError(
                    "partial checkpoint manifest is not the exact deterministic "
                    "derivation"
                )
            canonical_predecessor_filenames = {
                "selection": "selection.tsv",
                "primary_reviews": "reviews.tsv",
                "secondary_reviews": "reviews_curator_2_partial.tsv",
                "comparison": "review_comparison_partial.tsv",
            }
            for component, expected_filename in canonical_predecessor_filenames.items():
                component_manifest = derived_partial_manifest.get(component)
                observed_filename = (
                    component_manifest.get("filename")
                    if isinstance(component_manifest, dict)
                    else None
                )
                if observed_filename != expected_filename:
                    raise ValueError(
                        "partial checkpoint must use canonical predecessor filename "
                        f"{expected_filename!r} for {component}"
                    )
            if derived_partial_manifest["status"] != (
                "partial_dual_review_requires_completion_and_human_adjudication"
            ):
                raise ValueError(
                    "predecessor checkpoint is not an incomplete dual review"
                )

            try:
                observed_progress_manifest = json.loads(
                    snapshot_paths["progress_manifest"].read_text(encoding="utf-8")
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "progress checkpoint manifest is not valid UTF-8 JSON"
                ) from exc
            if not isinstance(observed_progress_manifest, dict):
                raise ValueError("progress checkpoint manifest must be a JSON object")
            try:
                progress_timestamp = pd.Timestamp(
                    observed_progress_manifest["assessed_date"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "progress checkpoint manifest requires a valid assessed_date"
                ) from exc
            if pd.isna(progress_timestamp):
                raise ValueError(
                    "progress checkpoint manifest requires a valid assessed_date"
                )
            progress_date = progress_timestamp.date()
            if completion_date < progress_date:
                raise ValueError(
                    "completed assessed_date cannot precede the progress checkpoint"
                )
            derived_progress_manifest = _derive_secondary_review_progress_manifest(
                snapshot_paths["progress_reviews"],
                snapshot_paths["primary_reviews"],
                snapshot_paths["selection"],
                snapshot_paths["partial_secondary_reviews"],
                snapshot_paths["partial_comparison"],
                snapshot_paths["partial_manifest"],
                assessed_date=progress_date,
                expected_predecessor_manifest_sha256=expected_checkpoint_sha,
            )
            if observed_progress_manifest != derived_progress_manifest:
                raise ValueError(
                    "progress checkpoint manifest is not the exact deterministic "
                    "derivation"
                )

            primary = read_table(snapshot_paths["primary_reviews"], dtype="string")
            completion = read_table(
                snapshot_paths["completion_reviews"], dtype="string"
            )
            progress = read_table(snapshot_paths["progress_reviews"], dtype="string")
            selection = read_table(snapshot_paths["selection"], dtype="string")
            partial_secondary = read_table(
                snapshot_paths["partial_secondary_reviews"], dtype="string"
            )
            partial_secondary_raw = _read_table_text_cells(
                snapshot_paths["partial_secondary_reviews"]
            )
            completion_raw = _read_table_text_cells(
                snapshot_paths["completion_reviews"]
            )
            progress_raw = _read_table_text_cells(snapshot_paths["progress_reviews"])
            primary_valid = _validate_review_against_selection(
                primary,
                selection,
                role="primary",
            )
            partial_valid = _validate_review_against_selection(
                partial_secondary,
                selection,
                role="partial secondary",
                allow_subset=True,
            )
            completion_valid = _validate_review_against_selection(
                completion,
                selection,
                role="completion secondary",
                allow_subset=True,
            )
            progress_valid = _validate_review_against_selection(
                progress,
                selection,
                role="completion progress",
                allow_subset=True,
            )

            for role, reviews in (
                ("partial secondary", partial_valid),
                ("completion secondary", completion_valid),
                ("completion progress", progress_valid),
            ):
                if reviews["queue_id"].duplicated().any():
                    raise ValueError(f"{role} queue_id values must be unique")

            selection_ids = set(selection["queue_id"].astype(str))
            partial_ids = set(partial_valid["queue_id"].astype(str))
            completion_ids = set(completion_valid["queue_id"].astype(str))
            overlap = sorted(partial_ids & completion_ids)
            if overlap:
                raise ValueError(
                    f"completion reviews overlap frozen partial rows: {overlap}"
                )
            expected_completion_ids = selection_ids - partial_ids
            missing = sorted(expected_completion_ids - completion_ids)
            unexpected = sorted(completion_ids - expected_completion_ids)
            if missing or unexpected:
                raise ValueError(
                    "completion reviews are not the exact selection complement: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            progress_ids = set(progress_valid["queue_id"].astype(str))
            missing_progress = sorted(progress_ids - completion_ids)
            if missing_progress:
                raise ValueError(
                    f"completion reviews omit frozen progress rows: {missing_progress}"
                )
            if set(progress_raw.columns) != set(completion_raw.columns):
                raise ValueError(
                    "progress and completion review columns must match exactly"
                )
            progress_aligned = progress_raw.loc[:, completion_raw.columns]
            completion_by_queue = completion_raw.set_index("queue_id", drop=False)
            completion_progress_rows = completion_by_queue.loc[
                progress_aligned["queue_id"].astype(str), completion_raw.columns
            ].reset_index(drop=True)
            if (
                not completion_progress_rows.fillna("")
                .astype(str)
                .equals(progress_aligned.fillna("").astype(str).reset_index(drop=True))
            ):
                raise ValueError(
                    "completion reviews changed authenticated progress cells"
                )

            if set(partial_secondary_raw.columns) != set(completion_raw.columns):
                raise ValueError(
                    "partial and completion review columns must match exactly"
                )
            raw_columns = partial_secondary_raw.columns.tolist()
            completion_aligned = completion_raw.loc[:, raw_columns]
            combined_secondary = pd.concat(
                [partial_secondary_raw, completion_aligned], ignore_index=True
            )
            selection_order = {
                str(queue_id): index
                for index, queue_id in enumerate(selection["queue_id"].astype(str))
            }
            combined_secondary["_selection_order"] = combined_secondary["queue_id"].map(
                selection_order
            )
            combined_secondary = (
                combined_secondary.sort_values("_selection_order", kind="stable")
                .drop(columns="_selection_order")
                .reset_index(drop=True)
            )
            if len(combined_secondary) != len(selection):
                raise ValueError(
                    "completed secondary review count differs from the selection"
                )

            combined_valid = pd.concat(
                [partial_valid, completion_valid], ignore_index=True
            )
            combined_valid["_selection_order"] = combined_valid["queue_id"].map(
                selection_order
            )
            combined_valid = (
                combined_valid.sort_values("_selection_order", kind="stable")
                .drop(columns="_selection_order")
                .reset_index(drop=True)
            )

            comparison = compare_full_text_reviews(
                primary_valid,
                combined_valid,
                selection,
                assessed_date=completion_date,
            )
            forbidden_release_columns = {"benchmark_ready", "label_code"}
            if forbidden_release_columns & set(combined_secondary.columns):
                raise ValueError("completed reviews contain released label columns")
            if forbidden_release_columns & set(comparison.columns):
                raise ValueError("completed comparison contains released label columns")

            partial_ranks = sorted(partial_valid["queue_rank"].astype(int).tolist())
            completion_ranks = sorted(
                completion_valid["queue_rank"].astype(int).tolist()
            )
            staging_dir = work_dir / "publish"
            staging_dir.mkdir()
            preserved_files = {
                "primary_reviews": "reviews.tsv",
                "selection": "selection.tsv",
                "partial_secondary_reviews": "reviews_curator_2_partial.tsv",
                "partial_comparison": "review_comparison_partial.tsv",
                "partial_manifest": "dual_review_manifest_partial.json",
                "progress_reviews": "reviews_curator_2_completion_progress.tsv",
                "progress_manifest": "secondary_review_progress_manifest.json",
                "completion_reviews": "reviews_curator_2_completion.tsv",
            }
            for key, filename in preserved_files.items():
                shutil.copyfile(snapshot_paths[key], staging_dir / filename)

            full_secondary_path = staging_dir / "reviews_curator_2.tsv"
            full_comparison_path = staging_dir / "review_comparison.tsv"
            full_manifest_path = staging_dir / "dual_review_manifest.json"
            combined_secondary.to_csv(
                full_secondary_path,
                sep="\t",
                index=False,
                lineterminator="\n",
            )
            comparison.to_csv(
                full_comparison_path,
                sep="\t",
                index=False,
                lineterminator="\n",
            )

            # The authenticated raw partial and completion cells, including
            # surrounding whitespace, must survive the full-table write exactly.
            written_secondary = _read_table_text_cells(full_secondary_path)
            written_by_queue = written_secondary.set_index("queue_id", drop=False)
            for role, source in (
                ("partial", partial_secondary_raw),
                ("progress", progress_aligned),
                ("completion", completion_aligned),
            ):
                written_subset = written_by_queue.loc[
                    source["queue_id"].astype(str), raw_columns
                ].reset_index(drop=True)
                source_text = source.loc[:, raw_columns].fillna("").astype(str)
                written_text = written_subset.fillna("").astype(str)
                if not written_text.equals(source_text.reset_index(drop=True)):
                    raise RuntimeError(
                        f"completed reviews did not preserve {role} raw cells"
                    )

            manifest = build_dual_review_manifest(
                staging_dir / "reviews.tsv",
                full_secondary_path,
                staging_dir / "selection.tsv",
                full_comparison_path,
                assessed_date=completion_date,
            )
            if manifest["status"] != (
                "dual_review_complete_requires_human_adjudication"
            ):
                raise RuntimeError(
                    "completed bundle did not reach the complete review state"
                )
            manifest.update(
                {
                    "schema_version": 2,
                    "human_adjudication_required": True,
                    "adjudication_status": "pending_human_adjudication",
                    "adjudicated_gene_count": 0,
                    "released_label_count": 0,
                    "benchmark_ready_count": 0,
                    "predecessor_checkpoint": {
                        "manifest": {
                            "filename": preserved_files["partial_manifest"],
                            "sha256": input_sha["partial_manifest"],
                        },
                        "secondary_reviews": {
                            "filename": preserved_files["partial_secondary_reviews"],
                            "sha256": input_sha["partial_secondary_reviews"],
                        },
                        "comparison": {
                            "filename": preserved_files["partial_comparison"],
                            "sha256": input_sha["partial_comparison"],
                        },
                        "second_reviewed_screen_count": len(partial_valid),
                        "queue_ranks": partial_ranks,
                    },
                    "progress_checkpoint": {
                        "manifest": {
                            "filename": preserved_files["progress_manifest"],
                            "sha256": input_sha["progress_manifest"],
                        },
                        "secondary_reviews": {
                            "filename": preserved_files["progress_reviews"],
                            "sha256": input_sha["progress_reviews"],
                        },
                        "second_reviewed_screen_count": len(progress_valid),
                        "queue_ranks": sorted(
                            progress_valid["queue_rank"].astype(int).tolist()
                        ),
                    },
                    "completion_lineage": {
                        "input": {
                            "filename": preserved_files["completion_reviews"],
                            "sha256": input_sha["completion_reviews"],
                        },
                        "completed_screen_count": len(completion_valid),
                        "queue_ranks": completion_ranks,
                    },
                }
            )
            full_manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            if _path_lexists(output_dir):
                raise FileExistsError(
                    "completed dual-review destination appeared while staging: "
                    f"{output_dir}"
                )
            staging_dir.rename(output_dir)
        return manifest
    finally:
        os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def build_run_accession_map_manifest(
    run_map_path: str | Path,
    source_inventory_path: str | Path,
    contrast_scope_path: str | Path,
) -> dict[str, object]:
    """Validate an accession map and summarize the reconstructable contrast."""

    run_map_path = Path(run_map_path)
    source_inventory_path = Path(source_inventory_path)
    contrast_scope_path = Path(contrast_scope_path)
    run_map = read_table(run_map_path, dtype="string")
    source_inventory = read_table(source_inventory_path, dtype="string")
    contrast_scope = read_table(contrast_scope_path, dtype="string")
    valid, errors = validate_records(run_map, "run_accession_map")
    if not errors.empty:
        details = errors.head(5).to_dict(orient="records")
        raise ValueError(f"run accession map failed contract validation: {details}")
    valid_inventory, inventory_errors = validate_records(
        source_inventory,
        "run_accession_inventory",
    )
    if not inventory_errors.empty:
        details = inventory_errors.head(5).to_dict(orient="records")
        raise ValueError(f"source run inventory failed contract validation: {details}")
    valid_scope, scope_errors = validate_records(
        contrast_scope,
        "run_contrast_scope",
    )
    if not scope_errors.empty:
        details = scope_errors.head(5).to_dict(orient="records")
        raise ValueError(f"contrast scope failed contract validation: {details}")
    study_accessions = valid["study_accession"].drop_duplicates().tolist()
    bioproject_accessions = valid["bioproject_accession"].drop_duplicates().tolist()
    if len(study_accessions) != 1 or len(bioproject_accessions) != 1:
        raise ValueError(
            "one run map manifest requires one study and BioProject accession"
        )
    inventory_studies = valid_inventory["study_accession"].drop_duplicates().tolist()
    inventory_bioprojects = (
        valid_inventory["bioproject_accession"].drop_duplicates().tolist()
    )
    if (
        inventory_studies != study_accessions
        or inventory_bioprojects != bioproject_accessions
    ):
        raise ValueError("run map and source inventory identify different accessions")
    mapped_runs = set(valid["run_accession"].astype(str))
    inventory_runs = set(valid_inventory["run_accession"].astype(str))
    if mapped_runs != inventory_runs:
        missing = sorted(inventory_runs - mapped_runs)
        unexpected = sorted(mapped_runs - inventory_runs)
        raise ValueError(
            "run accession map must exactly match the pinned source inventory; "
            f"missing={missing}, unexpected={unexpected}"
        )
    inventory_fields = tuple(
        field_name
        for field_name in valid_inventory.columns
        if field_name != "run_accession"
    )
    mapped_by_run = valid.set_index("run_accession").loc[
        valid_inventory["run_accession"].astype(str)
    ]
    inventory_by_run = valid_inventory.set_index("run_accession")
    mismatched_inventory_fields = [
        field_name
        for field_name in inventory_fields
        if mapped_by_run[field_name]
        .fillna("")
        .astype(str)
        .ne(inventory_by_run[field_name].fillna("").astype(str))
        .any()
    ]
    if mismatched_inventory_fields:
        raise ValueError(
            "run accession map differs from the pinned source inventory for fields: "
            f"{mismatched_inventory_fields}"
        )
    scope_studies = valid_scope["study_accession"].drop_duplicates().tolist()
    if scope_studies != study_accessions:
        raise ValueError("run map and contrast scope identify different studies")
    inventory_groups = set(valid_inventory["repository_screen_group"].astype(str))
    scope_groups = set(valid_scope["repository_screen_group"].astype(str))
    if inventory_groups != scope_groups:
        missing = sorted(inventory_groups - scope_groups)
        unexpected = sorted(scope_groups - inventory_groups)
        raise ValueError(
            "contrast scope must exactly cover repository screen groups; "
            f"missing={missing}, unexpected={unexpected}"
        )
    scope_fields = (
        "inclusion_status",
        "screen_id",
        "contrast_id",
        "condition_role",
        "treatment_name",
        "treatment_mapping_evidence",
    )
    expected_scope = (
        valid_inventory[["run_accession", "repository_screen_group"]]
        .merge(
            valid_scope[["repository_screen_group", *scope_fields]],
            on="repository_screen_group",
            how="left",
            validate="many_to_one",
        )
        .set_index("run_accession")
    )
    mismatched_scope_fields = [
        field_name
        for field_name in scope_fields
        if mapped_by_run[field_name]
        .fillna("")
        .astype(str)
        .ne(expected_scope[field_name].fillna("").astype(str))
        .any()
    ]
    if mismatched_scope_fields:
        raise ValueError(
            "run accession map differs from the curated contrast scope for fields: "
            f"{mismatched_scope_fields}"
        )
    if valid["map_id"].duplicated().any():
        raise ValueError("run accession map_id values must be unique")
    included = valid.loc[valid["inclusion_status"].eq("included_drug_contrast")].copy()
    if included.empty:
        raise ValueError("run accession map contains no included drug-contrast runs")
    screen_ids = included["screen_id"].drop_duplicates().tolist()
    contrast_ids = included["contrast_id"].drop_duplicates().tolist()
    if len(screen_ids) != 1 or len(contrast_ids) != 1:
        raise ValueError("one run map manifest requires one screen and contrast")

    expected_cells = {
        ("control", "dividing"),
        ("control", "nondividing"),
        ("treatment", "dividing"),
        ("treatment", "nondividing"),
    }
    incomplete_donors: list[str] = []
    for donor_id, donor_runs in included.groupby("donor_id", sort=True):
        observed_cell_rows = list(
            donor_runs[["condition_role", "phenotype_bin"]].itertuples(
                index=False, name=None
            )
        )
        observed_cells = set(observed_cell_rows)
        if observed_cells != expected_cells or len(observed_cell_rows) != len(
            expected_cells
        ):
            incomplete_donors.append(str(donor_id))
    if incomplete_donors:
        raise ValueError(
            "included donors lack the complete condition-by-bin design: "
            f"{incomplete_donors}"
        )

    identity_fields = (
        "sample_accession",
        "secondary_sample_accession",
        "source_sample_id",
    )
    identity_metadata_fields = [
        *identity_fields,
        "donor_id",
        "inclusion_status",
        "screen_id",
        "contrast_id",
        "condition_role",
        "treatment_name",
        "phenotype_bin",
    ]
    inconsistent_identifiers: list[str] = []
    for identifier in identity_fields:
        for identifier_value, identifier_runs in valid.groupby(identifier, sort=True):
            if len(identifier_runs[identity_metadata_fields].drop_duplicates()) != 1:
                inconsistent_identifiers.append(f"{identifier}={identifier_value}")
    if inconsistent_identifiers:
        raise ValueError(
            "sample identifiers map to contradictory design metadata: "
            f"{sorted(inconsistent_identifiers)}"
        )

    treatment_names_by_role: dict[str, str] = {}
    for condition_role, role_runs in included.groupby("condition_role", sort=True):
        treatment_names = role_runs["treatment_name"].drop_duplicates().tolist()
        if len(treatment_names) != 1:
            raise ValueError("each condition role requires exactly one treatment name")
        treatment_names_by_role[str(condition_role)] = str(treatment_names[0])
    if len(set(treatment_names_by_role.values())) != len(treatment_names_by_role):
        raise ValueError("control and treatment names must be distinct")

    mapping_evidence = set(included["treatment_mapping_evidence"].astype(str))
    status = (
        "conditional_article_supported_condition_mapping"
        if "article_supported" in mapping_evidence
        else "repository_explicit_condition_mapping"
    )

    def counts(column: str, frame: pd.DataFrame = valid) -> dict[str, int]:
        observed = frame[column].value_counts().sort_index()
        return {str(key): int(value) for key, value in observed.items()}

    return {
        "schema_version": 1,
        "study_accession": str(study_accessions[0]),
        "bioproject_accession": str(bioproject_accessions[0]),
        "screen_id": str(screen_ids[0]),
        "contrast_id": str(contrast_ids[0]),
        "status": status,
        "benchmark_ready_count": 0,
        "raw_reads_ingested": False,
        "record_count": len(valid),
        "included_run_count": len(included),
        "excluded_other_screen_run_count": int(
            valid["inclusion_status"].eq("excluded_other_screen").sum()
        ),
        "included_donor_count": int(included["donor_id"].nunique()),
        "inclusion_status_counts": counts("inclusion_status"),
        "mapping_evidence_counts": counts("treatment_mapping_evidence", included),
        "included_treatment_names_by_role": treatment_names_by_role,
        "included_condition_role_counts": counts("condition_role", included),
        "included_phenotype_bin_counts": counts("phenotype_bin", included),
        "design_estimand": (
            "donor-blocked treatment-by-phenotype-bin interaction; not a "
            "single treatment-versus-control bin comparison"
        ),
        "run_map": {
            "filename": run_map_path.name,
            "sha256": _sha256(run_map_path),
        },
        "source_inventory": {
            "filename": source_inventory_path.name,
            "sha256": _sha256(source_inventory_path),
            "record_count": len(valid_inventory),
            "repository_screen_group_counts": counts(
                "repository_screen_group",
                valid_inventory,
            ),
        },
        "contrast_scope": {
            "filename": contrast_scope_path.name,
            "sha256": _sha256(contrast_scope_path),
            "rule_count": len(valid_scope),
            "included_repository_screen_groups": sorted(
                valid_scope.loc[
                    valid_scope["inclusion_status"].eq("included_drug_contrast"),
                    "repository_screen_group",
                ]
                .astype(str)
                .tolist()
            ),
            "mapping_evidence_counts": counts(
                "treatment_mapping_evidence",
                valid_scope,
            ),
        },
    }
