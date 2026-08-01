from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from crispr_evidencerank.cli import build_parser
from crispr_evidencerank.contracts import ReviewComparisonRecord
from crispr_evidencerank.curation import (
    build_dual_review_manifest,
    compare_full_text_reviews,
    write_dual_review_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = (
    ROOT / "data" / "manifests" / "orcs_2.0.18" / "curation_batches" / "batch_001"
)
PARTIAL_SECONDARY = BATCH_DIR / "reviews_curator_2_partial.tsv"
PARTIAL_COMPARISON = BATCH_DIR / "review_comparison_partial.tsv"
PARTIAL_MANIFEST = BATCH_DIR / "dual_review_manifest_partial.json"


def _reviews() -> pd.DataFrame:
    return pd.read_csv(BATCH_DIR / "reviews.tsv", sep="\t", dtype=str)


def _selection() -> pd.DataFrame:
    return pd.read_csv(BATCH_DIR / "selection.tsv", sep="\t", dtype=str)


def _secondary_reviews() -> pd.DataFrame:
    secondary = _reviews().copy()
    secondary["review_id"] = secondary["review_id"].str.replace(
        ":v1", ":v2", regex=False
    )
    secondary["curator"] = "Independent reviewer B"
    return secondary


def test_exact_second_review_agreement_stays_provisional():
    comparison = compare_full_text_reviews(
        _reviews(),
        _secondary_reviews(),
        _selection(),
        assessed_date=date(2026, 8, 1),
    )

    assert len(comparison) == 18
    assert comparison["comparison_status"].eq("provisional_agreement").all()
    assert comparison["human_adjudication_required"].all()
    assert "label_code" not in comparison
    assert "benchmark_ready" not in comparison
    assert "label_code" not in ReviewComparisonRecord.model_fields
    assert "benchmark_ready" not in ReviewComparisonRecord.model_fields


def test_comparison_preserves_label_disagreement_and_one_sided_annotation():
    secondary = _secondary_reviews()
    secondary.loc[0, "candidate_v2_genes"] = "CDK6"
    secondary.loc[0, "candidate_v1_genes"] = "TOP2A"
    secondary.loc[0, "validation_status"] = "candidate_v2"
    secondary.loc[1, "candidate_v1_genes"] = "COQ10B|COQ9|DEPDC5"

    comparison = compare_full_text_reviews(
        _reviews(),
        secondary,
        _selection(),
        assessed_date=date(2026, 8, 1),
    ).set_index(["queue_rank", "gene_symbol"])

    assert comparison.loc[(1, "CDK6"), "comparison_status"] == ("label_disagreement")
    assert comparison.loc[(2, "NPRL2"), "comparison_status"] == ("single_curator_only")
    assert comparison.loc[(2, "NPRL2"), "secondary_evidence_level"] == ("not_annotated")


def test_comparison_rejects_reused_curator_identity():
    secondary = _secondary_reviews()
    secondary["curator"] = _reviews()["curator"]
    with pytest.raises(ValueError, match="curator identities must be disjoint"):
        compare_full_text_reviews(
            _reviews(),
            secondary,
            _selection(),
            assessed_date=date(2026, 8, 1),
        )


def test_comparison_rejects_incomplete_review_and_mismatched_batch():
    secondary = _secondary_reviews()
    secondary.loc[0, "full_text_reviewed"] = "False"
    with pytest.raises(ValueError, match="full_text_reviewed=true"):
        compare_full_text_reviews(
            _reviews(),
            secondary,
            _selection(),
            assessed_date=date(2026, 8, 1),
        )

    secondary = _secondary_reviews()
    secondary["batch_id"] = "different-batch"
    with pytest.raises(ValueError, match="same batch_id"):
        compare_full_text_reviews(
            _reviews(),
            secondary,
            _selection(),
            assessed_date=date(2026, 8, 1),
        )


def test_comparison_rejects_cross_file_review_id_collision_and_early_date():
    secondary = _secondary_reviews()
    secondary.loc[0, "review_id"] = _reviews().loc[1, "review_id"]
    with pytest.raises(ValueError, match="review IDs must be disjoint"):
        compare_full_text_reviews(
            _reviews(),
            secondary,
            _selection(),
            assessed_date=date(2026, 8, 1),
        )

    with pytest.raises(ValueError, match="cannot precede"):
        compare_full_text_reviews(
            _reviews(),
            _secondary_reviews(),
            _selection(),
            assessed_date=date(2026, 7, 31),
        )


def test_comparison_rejects_invalid_selection_contract():
    invalid_selection = _selection().assign(outcome_label="V2")
    with pytest.raises(ValueError, match="selection failed contract validation"):
        compare_full_text_reviews(
            _reviews(),
            _secondary_reviews(),
            invalid_selection,
            assessed_date=date(2026, 8, 1),
        )


def test_comparison_rejects_unrelated_source_id():
    secondary = _secondary_reviews()
    secondary.loc[0, "source_id"] = "UNRELATED-SOURCE"
    with pytest.raises(ValueError, match="differ for source_id"):
        compare_full_text_reviews(
            _reviews(),
            secondary,
            _selection(),
            assessed_date=date(2026, 8, 1),
        )


def test_comparison_identity_tracks_review_pair_and_assessment_date():
    baseline = compare_full_text_reviews(
        _reviews(),
        _secondary_reviews(),
        _selection(),
        assessed_date=date(2026, 8, 1),
    ).set_index(["screen_id", "gene_symbol"])

    revised_secondary = _secondary_reviews()
    revised_secondary["review_id"] = revised_secondary["review_id"].str.replace(
        ":v2", ":v3", regex=False
    )
    revised = compare_full_text_reviews(
        _reviews(),
        revised_secondary,
        _selection(),
        assessed_date=date(2026, 8, 1),
    ).set_index(["screen_id", "gene_symbol"])
    later = compare_full_text_reviews(
        _reviews(),
        _secondary_reviews(),
        _selection(),
        assessed_date=date(2026, 8, 2),
    ).set_index(["screen_id", "gene_symbol"])

    assert baseline["comparison_id"].ne(revised["comparison_id"]).all()
    assert baseline["comparison_id"].ne(later["comparison_id"]).all()


def test_dual_review_manifest_is_checksum_bound_and_deterministic(tmp_path):
    primary = _reviews()
    secondary = _secondary_reviews()
    selection = _selection()
    comparison = compare_full_text_reviews(
        primary,
        secondary,
        selection,
        assessed_date=date(2026, 8, 1),
    )
    primary_path = tmp_path / "primary.tsv"
    secondary_path = tmp_path / "secondary.tsv"
    selection_path = tmp_path / "selection.tsv"
    comparison_path = tmp_path / "comparison.tsv"
    primary.to_csv(primary_path, sep="\t", index=False, lineterminator="\n")
    secondary.to_csv(secondary_path, sep="\t", index=False, lineterminator="\n")
    selection.to_csv(selection_path, sep="\t", index=False, lineterminator="\n")
    comparison.to_csv(comparison_path, sep="\t", index=False, lineterminator="\n")

    manifest = build_dual_review_manifest(
        primary_path,
        secondary_path,
        selection_path,
        comparison_path,
    )

    assert manifest["status"] == ("dual_review_complete_requires_human_adjudication")
    assert manifest["benchmark_ready_count"] == 0
    assert manifest["compared_gene_count"] == 18
    assert manifest["second_reviewed_screen_count"] == 10
    assert manifest["pending_second_review_screen_count"] == 0
    assert manifest["comparison_status_counts"] == {"provisional_agreement": 18}
    assert set(manifest["primary_reviews"]["sha256"]) <= set("0123456789abcdef")

    comparison.loc[0, "gene_symbol"] = "TAMPERED"
    comparison.to_csv(comparison_path, sep="\t", index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="not the deterministic derivation"):
        build_dual_review_manifest(
            primary_path,
            secondary_path,
            selection_path,
            comparison_path,
        )


def test_dual_review_manifest_records_interpreted_design_disagreement(tmp_path):
    primary = _reviews()
    secondary = _secondary_reviews()
    secondary.loc[0, "treatment_contrast"] = "DIFFERENT DRUG/CONTROL CONTRAST"
    comparison = compare_full_text_reviews(
        primary,
        secondary,
        _selection(),
        assessed_date=date(2026, 8, 1),
    )
    primary_path = tmp_path / "primary.tsv"
    secondary_path = tmp_path / "secondary.tsv"
    selection_path = tmp_path / "selection.tsv"
    comparison_path = tmp_path / "comparison.tsv"
    primary.to_csv(primary_path, sep="\t", index=False, lineterminator="\n")
    secondary.to_csv(secondary_path, sep="\t", index=False, lineterminator="\n")
    _selection().to_csv(selection_path, sep="\t", index=False, lineterminator="\n")
    comparison.to_csv(comparison_path, sep="\t", index=False, lineterminator="\n")

    manifest = build_dual_review_manifest(
        primary_path,
        secondary_path,
        selection_path,
        comparison_path,
    )

    assert manifest["critical_field_disagreement_counts"]["treatment_contrast"] == 1


def test_dual_review_bundle_is_atomic_and_refuses_overwrite(tmp_path):
    primary_path = tmp_path / "primary.tsv"
    secondary_path = tmp_path / "secondary.tsv"
    selection_path = tmp_path / "selection.tsv"
    _reviews().to_csv(primary_path, sep="\t", index=False, lineterminator="\n")
    _secondary_reviews().to_csv(
        secondary_path, sep="\t", index=False, lineterminator="\n"
    )
    _selection().to_csv(selection_path, sep="\t", index=False, lineterminator="\n")
    output_dir = tmp_path / "dual_review"

    manifest = write_dual_review_bundle(
        primary_path,
        secondary_path,
        selection_path,
        output_dir,
        assessed_date=date(2026, 8, 1),
    )

    assert manifest["compared_gene_count"] == 18
    assert (output_dir / "review_comparison.tsv").is_file()
    assert (output_dir / "dual_review_manifest.json").is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        write_dual_review_bundle(
            primary_path,
            secondary_path,
            selection_path,
            output_dir,
            assessed_date=date(2026, 8, 1),
        )


def test_dual_review_bundle_supports_zero_gene_evidence(tmp_path):
    primary = _reviews()
    secondary = _secondary_reviews()
    gene_columns = [
        "candidate_v3_genes",
        "candidate_v2_genes",
        "candidate_v1_genes",
        "nonqualifying_validation_genes",
    ]
    for reviews in (primary, secondary):
        reviews[gene_columns] = None
        reviews["validation_status"] = "none_reported"

    primary_path = tmp_path / "primary.tsv"
    secondary_path = tmp_path / "secondary.tsv"
    selection_path = tmp_path / "selection.tsv"
    primary.to_csv(primary_path, sep="\t", index=False, lineterminator="\n")
    secondary.to_csv(secondary_path, sep="\t", index=False, lineterminator="\n")
    _selection().to_csv(selection_path, sep="\t", index=False, lineterminator="\n")
    output_dir = tmp_path / "zero-gene-comparison"

    manifest = write_dual_review_bundle(
        primary_path,
        secondary_path,
        selection_path,
        output_dir,
        assessed_date=date(2026, 8, 1),
    )

    comparison = pd.read_csv(output_dir / "review_comparison.tsv", sep="\t")
    assert comparison.empty
    assert manifest["compared_gene_count"] == 0
    assert manifest["comparison_status_counts"] == {}
    assert manifest["second_reviewed_screen_count"] == 10
    assert manifest["benchmark_ready_count"] == 0


def test_partial_second_review_is_explicit_and_compares_only_completed_screens(
    tmp_path,
):
    primary = _reviews()
    secondary = _secondary_reviews().tail(5).reset_index(drop=True)
    selection = _selection()
    comparison = compare_full_text_reviews(
        primary,
        secondary,
        selection,
        assessed_date=date(2026, 8, 1),
        allow_partial_secondary=True,
    )
    assert comparison["queue_rank"].min() == 6
    assert comparison["queue_rank"].max() == 10
    assert set(comparison["queue_rank"]) == {6, 7, 8, 9, 10}

    primary_path = tmp_path / "primary.tsv"
    secondary_path = tmp_path / "secondary.tsv"
    selection_path = tmp_path / "selection.tsv"
    comparison_path = tmp_path / "comparison.tsv"
    primary.to_csv(primary_path, sep="\t", index=False, lineterminator="\n")
    secondary.to_csv(secondary_path, sep="\t", index=False, lineterminator="\n")
    selection.to_csv(selection_path, sep="\t", index=False, lineterminator="\n")
    comparison.to_csv(comparison_path, sep="\t", index=False, lineterminator="\n")
    manifest = build_dual_review_manifest(
        primary_path,
        secondary_path,
        selection_path,
        comparison_path,
    )
    assert manifest["status"] == (
        "partial_dual_review_requires_completion_and_human_adjudication"
    )
    assert manifest["second_reviewed_screen_count"] == 5
    assert manifest["pending_second_review_screen_count"] == 5
    assert manifest["benchmark_ready_count"] == 0


def test_checked_in_partial_review_bundle_is_deterministic_and_not_a_label():
    observed_manifest = json.loads(PARTIAL_MANIFEST.read_text(encoding="utf-8"))
    derived_manifest = build_dual_review_manifest(
        BATCH_DIR / "reviews.tsv",
        PARTIAL_SECONDARY,
        BATCH_DIR / "selection.tsv",
        PARTIAL_COMPARISON,
    )
    secondary = pd.read_csv(PARTIAL_SECONDARY, sep="\t", dtype=str)
    comparison = pd.read_csv(PARTIAL_COMPARISON, sep="\t", dtype=str)

    assert observed_manifest == derived_manifest
    assert secondary["queue_rank"].astype(int).tolist() == [6, 7, 8, 9, 10]
    assert observed_manifest["pending_second_review_screen_count"] == 5
    assert observed_manifest["benchmark_ready_count"] == 0
    disagreements = observed_manifest["critical_field_disagreement_counts"]
    assert disagreements["raw_data_family_id"] == 1
    assert disagreements["supplement_review"] == 1
    assert disagreements["blocker_codes"] == 1
    assert disagreements["screen_replication"] == 2
    assert disagreements["quantitative_asset_locator"] == 4
    assert disagreements["validation_source_locator"] == 4
    assert comparison["comparison_status"].eq("provisional_agreement").all()
    assert comparison["human_adjudication_required"].eq("True").all()
    assert "label_code" not in comparison
    assert "benchmark_ready" not in comparison


def test_compare_curation_reviews_cli_writes_partial_bundle(tmp_path, capsys):
    output_dir = tmp_path / "comparison"
    args = build_parser().parse_args(
        [
            "compare-curation-reviews",
            "--primary-reviews",
            str(BATCH_DIR / "reviews.tsv"),
            "--secondary-reviews",
            str(PARTIAL_SECONDARY),
            "--selection",
            str(BATCH_DIR / "selection.tsv"),
            "--output-dir",
            str(output_dir),
            "--assessed-date",
            "2026-08-01",
            "--allow-partial-secondary",
        ]
    )

    assert args.func(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["second_reviewed_screen_count"] == 5
    assert report["pending_second_review_screen_count"] == 5
    assert report["benchmark_ready_count"] == 0
    assert (output_dir / "review_comparison.tsv").is_file()
    assert (output_dir / "dual_review_manifest.json").is_file()


def test_compare_curation_reviews_cli_requires_explicit_partial_opt_in(tmp_path):
    args = build_parser().parse_args(
        [
            "compare-curation-reviews",
            "--primary-reviews",
            str(BATCH_DIR / "reviews.tsv"),
            "--secondary-reviews",
            str(PARTIAL_SECONDARY),
            "--selection",
            str(BATCH_DIR / "selection.tsv"),
            "--output-dir",
            str(tmp_path / "comparison"),
            "--assessed-date",
            "2026-08-01",
        ]
    )

    with pytest.raises(ValueError, match="allow_partial_secondary=True"):
        args.func(args)


@pytest.mark.parametrize(
    ("updates", "expected_valid"),
    [
        ({}, True),
        ({"human_adjudication_required": False}, False),
        ({"comparison_status": "label_disagreement"}, False),
        ({"secondary_evidence_level": "candidate_v2"}, False),
    ],
)
def test_review_comparison_schema_and_runtime_contract(
    updates: dict[str, object],
    expected_valid: bool,
):
    record = {
        "comparison_id": "C1",
        "batch_id": "B1",
        "queue_id": "Q1",
        "queue_rank": 1,
        "screen_id": "S1",
        "external_screen_id": "1",
        "gene_symbol": "GENE1",
        "primary_review_id": "R1",
        "secondary_review_id": "R2",
        "primary_evidence_level": "candidate_v1",
        "secondary_evidence_level": "candidate_v1",
        "comparison_status": "provisional_agreement",
        "primary_source_locator": "Figure 1",
        "secondary_source_locator": "Figure 1",
        "human_adjudication_required": True,
        "assessed_date": "2026-08-01",
        "notes": None,
    } | updates
    schema_valid = Draft202012Validator(
        ReviewComparisonRecord.model_json_schema()
    ).is_valid(record)
    try:
        ReviewComparisonRecord.model_validate(record)
    except ValidationError:
        runtime_valid = False
    else:
        runtime_valid = True

    if updates in (
        {"comparison_status": "label_disagreement"},
        {"secondary_evidence_level": "candidate_v2"},
    ):
        assert schema_valid is True
        semantic_rules = ReviewComparisonRecord.model_json_schema()["x-semantic-rules"]
        assert any("derived" in rule for rule in semantic_rules)
    else:
        assert schema_valid is expected_valid
    assert runtime_valid is expected_valid
