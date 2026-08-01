from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from crispr_evidencerank.contracts import FullTextReviewRecord, validate_records
from crispr_evidencerank.curation import (
    FORBIDDEN_SELECTION_COLUMNS,
    build_review_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests" / "orcs_2.0.18"
BATCH_DIR = MANIFEST_DIR / "curation_batches" / "batch_001"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_tsv(name: str) -> pd.DataFrame:
    return pd.read_csv(BATCH_DIR / name, sep="\t", dtype=str)


def _base_review() -> dict[str, object]:
    raw = _read_tsv("reviews.tsv").iloc[0].to_dict()
    clean = {key: None if pd.isna(value) else value for key, value in raw.items()}
    return FullTextReviewRecord.model_validate(clean).model_dump(mode="json")


def test_batch_selection_is_the_frozen_outcome_blind_queue_prefix():
    queue = pd.read_csv(MANIFEST_DIR / "curation_queue.tsv", sep="\t", dtype=str)
    selection = _read_tsv("selection.tsv")
    manifest = json.loads(
        (BATCH_DIR / "selection_manifest.json").read_text(encoding="utf-8")
    )

    pd.testing.assert_frame_equal(selection, queue.head(10).reset_index(drop=True))
    assert selection["queue_rank"].astype(int).tolist() == list(range(1, 11))
    assert selection["source_family_id"].nunique() == 10
    assert set(selection.columns).isdisjoint(FORBIDDEN_SELECTION_COLUMNS)
    assert _sha256(BATCH_DIR / "selection.tsv") == manifest["selection"]["sha256"]
    assert (
        _sha256(MANIFEST_DIR / "curation_queue.tsv")
        == manifest["source_queue"]["sha256"]
    )


def test_batch_reviews_validate_and_match_the_frozen_selection():
    selection = _read_tsv("selection.tsv")
    reviews = _read_tsv("reviews.tsv")
    valid, errors = validate_records(reviews, "full_text_review")

    assert errors.empty, errors.to_dict(orient="records")
    assert len(valid) == 10
    for key in ("queue_id", "queue_rank", "screen_id", "external_screen_id"):
        assert reviews[key].tolist() == selection[key].tolist()
    assert (
        reviews["source_family_id"].tolist() == selection["source_family_id"].tolist()
    )
    assert reviews["disposition"].eq("metadata_only").all()
    assert reviews["full_text_reviewed"].eq("True").all()
    assert "benchmark_ready" not in reviews.columns
    assert "benchmark_ready" not in FullTextReviewRecord.model_fields


def test_review_manifest_matches_files_and_conservative_status_counts():
    reviews = _read_tsv("reviews.tsv")
    manifest = json.loads(
        (BATCH_DIR / "review_manifest.json").read_text(encoding="utf-8")
    )
    derived = build_review_manifest(
        BATCH_DIR / "reviews.tsv",
        BATCH_DIR / "selection.tsv",
    )

    assert manifest == derived
    assert _sha256(BATCH_DIR / "reviews.tsv") == manifest["reviews"]["sha256"]
    assert _sha256(BATCH_DIR / "selection.tsv") == manifest["selection"]["sha256"]
    assert len(reviews) == manifest["record_count"] == 10
    assert (
        reviews["disposition"].value_counts().to_dict()
        == manifest["disposition_counts"]
    )
    assert (
        reviews["quantitative_data_status"].value_counts().to_dict()
        == manifest["quantitative_data_status_counts"]
    )
    assert (
        reviews["validation_status"].value_counts().to_dict()
        == manifest["validation_status_counts"]
    )
    assert manifest["benchmark_ready_count"] == 0
    assert manifest["full_text_reviewed_count"] == int(
        reviews["full_text_reviewed"].eq("True").sum()
    )
    assert manifest["supplement_review_counts"] == {
        "complete": 9,
        "partial": 1,
    }
    assert manifest["quantitative_asset_family_count"] == 10
    assert manifest["raw_data_family_resolved_count"] == 1
    assert manifest["curation_status"] == (
        "single_curator_requires_independent_adjudication"
    )
    assert manifest["curator_count"] == 1


def test_outcome_review_does_not_overstate_raw_data_or_validation_readiness():
    reviews = _read_tsv("reviews.tsv")
    raw = reviews.loc[reviews["quantitative_data_status"].eq("raw_reads_public")]
    scores_only = reviews.loc[
        reviews["quantitative_data_status"].eq("author_scores_public")
    ]

    assert raw["external_screen_id"].tolist() == ["1110"]
    assert raw["data_accession"].tolist() == ["SRP158611"]
    assert not reviews["quantitative_data_status"].eq("raw_counts_public").any()
    assert (
        reviews["blocker_codes"]
        .str.contains("labels.adjudicated_validation_event", regex=False)
        .all()
    )
    assert (
        reviews["blocker_codes"]
        .str.contains("data.count_level_signal", regex=False)
        .all()
    )
    assert reviews["rights_outcome"].eq("unknown").all()
    assert raw["raw_data_family_id"].notna().all()
    assert (
        not raw["blocker_codes"]
        .str.contains("provenance.raw_data_family", regex=False)
        .any()
    )
    assert scores_only["raw_data_family_id"].isna().all()
    assert (
        scores_only["blocker_codes"]
        .str.contains("provenance.raw_data_family", regex=False)
        .all()
    )


def test_full_text_review_contract_rejects_misgraded_candidates_and_raw_data():
    review = _read_tsv("reviews.tsv").iloc[0].to_dict()
    review["validation_status"] = "candidate_v3"
    review["candidate_v3_genes"] = None
    _, candidate_errors = validate_records(pd.DataFrame([review]), "full_text_review")
    assert "same level" in candidate_errors.iloc[0]["error"]

    review = _read_tsv("reviews.tsv").iloc[0].to_dict()
    review["quantitative_data_status"] = "raw_reads_public"
    review["data_accession"] = None
    _, raw_errors = validate_records(pd.DataFrame([review]), "full_text_review")
    assert "data_accession" in raw_errors.iloc[0]["error"]


@pytest.mark.parametrize(
    ("updates", "expected_valid"),
    [
        ({}, True),
        (
            {
                "quantitative_data_status": "raw_reads_public",
                "data_accession": None,
            },
            False,
        ),
        (
            {
                "validation_status": "candidate_v3",
                "candidate_v3_genes": None,
            },
            False,
        ),
        ({"candidate_v3_genes": "GENE1"}, False),
        ({"candidate_v2_genes": "CDK6 |TOP2A"}, False),
        ({"disposition": "exclude", "scope_outcome": "pass"}, False),
        ({"disposition": "metadata_only", "scope_outcome": "fail"}, False),
    ],
)
def test_full_text_review_json_schema_matches_runtime_invariants(
    updates: dict[str, object],
    expected_valid: bool,
):
    record = _base_review() | updates
    schema_valid = Draft202012Validator(
        FullTextReviewRecord.model_json_schema()
    ).is_valid(record)

    try:
        FullTextReviewRecord.model_validate(record)
    except ValidationError:
        runtime_valid = False
    else:
        runtime_valid = True

    assert schema_valid is expected_valid
    assert runtime_valid is expected_valid
