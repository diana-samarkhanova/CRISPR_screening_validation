"""Outcome-blind selection of fixed full-text curation batches."""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

from .contracts import validate_records
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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

    linkage_fields = (
        "queue_id",
        "queue_rank",
        "screen_id",
        "external_screen_id",
        "source_family_id",
    )
    missing_selection = sorted(set(linkage_fields) - set(selection.columns))
    if missing_selection:
        raise ValueError(f"selection is missing linkage fields: {missing_selection}")
    if len(valid) != len(selection):
        raise ValueError("selection and review record counts differ")
    for field_name in linkage_fields:
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
