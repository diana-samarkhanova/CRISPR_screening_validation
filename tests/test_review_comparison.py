from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

import crispr_evidencerank.curation as curation_module
from crispr_evidencerank.cli import build_parser
from crispr_evidencerank.contracts import ReviewComparisonRecord
from crispr_evidencerank.curation import (
    build_dual_review_manifest,
    build_secondary_review_progress_manifest,
    compare_full_text_reviews,
    write_completed_dual_review_bundle,
    write_dual_review_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = (
    ROOT / "data" / "manifests" / "orcs_2.0.18" / "curation_batches" / "batch_001"
)
PARTIAL_SECONDARY = BATCH_DIR / "reviews_curator_2_partial.tsv"
PARTIAL_COMPARISON = BATCH_DIR / "review_comparison_partial.tsv"
PARTIAL_MANIFEST = BATCH_DIR / "dual_review_manifest_partial.json"
COMPLETION_PROGRESS = BATCH_DIR / "reviews_curator_2_completion_progress.tsv"
COMPLETION_PROGRESS_MANIFEST = BATCH_DIR / "secondary_review_progress_manifest.json"
COMPLETION_REVIEWS = BATCH_DIR / "reviews_curator_2_completion.tsv"
FULL_SECONDARY = BATCH_DIR / "reviews_curator_2.tsv"
FULL_COMPARISON = BATCH_DIR / "review_comparison.tsv"
FULL_MANIFEST = BATCH_DIR / "dual_review_manifest.json"


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


def _write_partial_checkpoint(
    tmp_path: Path, *, zero_gene_evidence: bool = False
) -> dict[str, Path | str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    primary_path = tmp_path / "reviews.tsv"
    selection_path = tmp_path / "selection.tsv"
    partial_secondary_path = tmp_path / "reviews_curator_2_partial.tsv"
    partial_comparison_path = tmp_path / "review_comparison_partial.tsv"
    partial_manifest_path = tmp_path / "dual_review_manifest_partial.json"
    completion_path = tmp_path / "reviews_curator_2_completion.tsv"
    progress_path = tmp_path / "reviews_curator_2_completion_progress.tsv"
    progress_manifest_path = tmp_path / "secondary_review_progress_manifest.json"

    primary = _reviews()
    selection = _selection()
    secondary = _secondary_reviews()
    if zero_gene_evidence:
        gene_columns = [
            "candidate_v3_genes",
            "candidate_v2_genes",
            "candidate_v1_genes",
            "nonqualifying_validation_genes",
        ]
        for reviews in (primary, secondary):
            reviews[gene_columns] = None
            reviews["validation_status"] = "none_reported"
    partial_secondary = secondary.tail(5).reset_index(drop=True)
    completion = secondary.head(5).reset_index(drop=True)
    partial_comparison = compare_full_text_reviews(
        primary,
        partial_secondary,
        selection,
        assessed_date=date(2026, 8, 1),
        allow_partial_secondary=True,
    )
    primary.to_csv(primary_path, sep="\t", index=False, lineterminator="\n")
    selection.to_csv(selection_path, sep="\t", index=False, lineterminator="\n")
    partial_secondary.to_csv(
        partial_secondary_path, sep="\t", index=False, lineterminator="\n"
    )
    partial_comparison.to_csv(
        partial_comparison_path, sep="\t", index=False, lineterminator="\n"
    )
    completion.to_csv(completion_path, sep="\t", index=False, lineterminator="\n")
    completion.loc[completion["queue_rank"].ne("2")].to_csv(
        progress_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    partial_manifest = build_dual_review_manifest(
        primary_path,
        partial_secondary_path,
        selection_path,
        partial_comparison_path,
        assessed_date=date(2026, 8, 1),
    )
    partial_manifest_path.write_text(
        json.dumps(partial_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint_sha = hashlib.sha256(partial_manifest_path.read_bytes()).hexdigest()
    progress_manifest = build_secondary_review_progress_manifest(
        progress_path,
        primary_path,
        selection_path,
        partial_secondary_path,
        partial_comparison_path,
        partial_manifest_path,
        assessed_date=date(2026, 8, 1),
        expected_predecessor_manifest_sha256=checkpoint_sha,
    )
    progress_manifest_path.write_text(
        json.dumps(progress_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    progress_sha = hashlib.sha256(progress_manifest_path.read_bytes()).hexdigest()
    return {
        "primary": primary_path,
        "selection": selection_path,
        "partial_secondary": partial_secondary_path,
        "partial_comparison": partial_comparison_path,
        "partial_manifest": partial_manifest_path,
        "completion": completion_path,
        "progress": progress_path,
        "progress_manifest": progress_manifest_path,
        "checkpoint_sha": checkpoint_sha,
        "progress_sha": progress_sha,
    }


def _refresh_progress_manifest(checkpoint: dict[str, Path | str]) -> None:
    progress_manifest = _build_progress_manifest(checkpoint)
    progress_manifest_path = Path(checkpoint["progress_manifest"])
    progress_manifest_path.write_text(
        json.dumps(progress_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint["progress_sha"] = hashlib.sha256(
        progress_manifest_path.read_bytes()
    ).hexdigest()


def _build_progress_manifest(
    checkpoint: dict[str, Path | str],
    *,
    progress_path: Path | None = None,
) -> dict[str, object]:
    return build_secondary_review_progress_manifest(
        progress_path or checkpoint["progress"],
        checkpoint["primary"],
        checkpoint["selection"],
        checkpoint["partial_secondary"],
        checkpoint["partial_comparison"],
        checkpoint["partial_manifest"],
        assessed_date=date(2026, 8, 1),
        expected_predecessor_manifest_sha256=str(checkpoint["checkpoint_sha"]),
    )


def _complete_checkpoint(
    checkpoint: dict[str, Path | str],
    output_dir: Path,
    *,
    assessed_date: date = date(2026, 8, 1),
) -> dict[str, object]:
    return write_completed_dual_review_bundle(
        checkpoint["primary"],
        checkpoint["completion"],
        checkpoint["progress"],
        checkpoint["progress_manifest"],
        checkpoint["selection"],
        checkpoint["partial_secondary"],
        checkpoint["partial_comparison"],
        checkpoint["partial_manifest"],
        output_dir,
        assessed_date=assessed_date,
        expected_checkpoint_manifest_sha256=str(checkpoint["checkpoint_sha"]),
        expected_progress_manifest_sha256=str(checkpoint["progress_sha"]),
    )


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


def test_completed_dual_review_bundle_preserves_checkpoint_and_releases_no_labels(
    tmp_path,
):
    checkpoint = _write_partial_checkpoint(tmp_path)
    output_dir = tmp_path / "completed"
    checkpoint_bytes = {
        "reviews_curator_2_partial.tsv": Path(
            checkpoint["partial_secondary"]
        ).read_bytes(),
        "review_comparison_partial.tsv": Path(
            checkpoint["partial_comparison"]
        ).read_bytes(),
        "dual_review_manifest_partial.json": Path(
            checkpoint["partial_manifest"]
        ).read_bytes(),
        "reviews_curator_2_completion_progress.tsv": Path(
            checkpoint["progress"]
        ).read_bytes(),
        "secondary_review_progress_manifest.json": Path(
            checkpoint["progress_manifest"]
        ).read_bytes(),
    }

    manifest = _complete_checkpoint(checkpoint, output_dir)

    assert {path.name for path in output_dir.iterdir()} == {
        "reviews.tsv",
        "selection.tsv",
        "reviews_curator_2_partial.tsv",
        "review_comparison_partial.tsv",
        "dual_review_manifest_partial.json",
        "reviews_curator_2_completion_progress.tsv",
        "secondary_review_progress_manifest.json",
        "reviews_curator_2_completion.tsv",
        "reviews_curator_2.tsv",
        "review_comparison.tsv",
        "dual_review_manifest.json",
    }
    for filename, expected in checkpoint_bytes.items():
        assert (output_dir / filename).read_bytes() == expected
    full_secondary = pd.read_csv(
        output_dir / "reviews_curator_2.tsv", sep="\t", dtype=str
    )
    full_comparison = pd.read_csv(
        output_dir / "review_comparison.tsv", sep="\t", dtype=str
    )
    preserved_partial = pd.read_csv(
        output_dir / "reviews_curator_2_partial.tsv", sep="\t", dtype=str
    )
    assert full_secondary["queue_rank"].astype(int).tolist() == list(range(1, 11))
    assert preserved_partial["queue_rank"].astype(int).tolist() == [6, 7, 8, 9, 10]
    full_partial_rows = full_secondary.loc[
        full_secondary["queue_id"].isin(preserved_partial["queue_id"]),
        preserved_partial.columns,
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(full_partial_rows, preserved_partial)
    assert {"label_code", "benchmark_ready"}.isdisjoint(full_secondary.columns)
    assert {"label_code", "benchmark_ready"}.isdisjoint(full_comparison.columns)
    assert manifest["status"] == ("dual_review_complete_requires_human_adjudication")
    assert manifest["human_adjudication_required"] is True
    assert manifest["adjudication_status"] == "pending_human_adjudication"
    assert manifest["adjudicated_gene_count"] == 0
    assert manifest["released_label_count"] == 0
    assert manifest["benchmark_ready_count"] == 0
    assert manifest["predecessor_checkpoint"]["queue_ranks"] == [6, 7, 8, 9, 10]
    assert manifest["progress_checkpoint"]["queue_ranks"] == [1, 3, 4, 5]
    assert (
        manifest["progress_checkpoint"]["manifest"]["sha256"]
        == (checkpoint["progress_sha"])
    )
    assert manifest["completion_lineage"]["queue_ranks"] == [1, 2, 3, 4, 5]
    assert (
        manifest["predecessor_checkpoint"]["manifest"]["sha256"]
        == (checkpoint["checkpoint_sha"])
    )
    assert (
        manifest["completion_lineage"]["input"]["sha256"]
        == hashlib.sha256(Path(checkpoint["completion"]).read_bytes()).hexdigest()
    )
    assert manifest["completion_lineage"]["input"]["filename"] == (
        "reviews_curator_2_completion.tsv"
    )
    assert manifest["primary_reviews"]["filename"] == "reviews.tsv"
    assert manifest["selection"]["filename"] == "selection.tsv"


def test_completed_bundle_accepts_exact_checked_in_progress_checkpoint(tmp_path):
    progress = pd.read_csv(
        COMPLETION_PROGRESS,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    primary = pd.read_csv(
        BATCH_DIR / "reviews.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    rank_two = primary.loc[primary["queue_rank"].eq("2")].copy()
    rank_two["review_id"] = rank_two["review_id"].str.replace(":v1", ":v2", regex=False)
    rank_two["curator"] = (
        "Independent Codex literature reviewer; per-study outcome mapping withheld"
    )
    completion = pd.concat([progress, rank_two], ignore_index=True)
    completion["_rank"] = completion["queue_rank"].astype(int)
    completion = completion.sort_values("_rank", kind="stable").drop(columns="_rank")
    completion_path = tmp_path / "reviews_curator_2_completion.tsv"
    completion.to_csv(completion_path, sep="\t", index=False, lineterminator="\n")
    output_dir = tmp_path / "completed"

    manifest = write_completed_dual_review_bundle(
        BATCH_DIR / "reviews.tsv",
        completion_path,
        COMPLETION_PROGRESS,
        COMPLETION_PROGRESS_MANIFEST,
        BATCH_DIR / "selection.tsv",
        PARTIAL_SECONDARY,
        PARTIAL_COMPARISON,
        PARTIAL_MANIFEST,
        output_dir,
        assessed_date=date(2026, 8, 1),
        expected_checkpoint_manifest_sha256=hashlib.sha256(
            PARTIAL_MANIFEST.read_bytes()
        ).hexdigest(),
        expected_progress_manifest_sha256=hashlib.sha256(
            COMPLETION_PROGRESS_MANIFEST.read_bytes()
        ).hexdigest(),
    )

    assert manifest["progress_checkpoint"]["queue_ranks"] == [1, 3, 4, 5]
    assert (output_dir / COMPLETION_PROGRESS.name).read_bytes() == (
        COMPLETION_PROGRESS.read_bytes()
    )
    full_secondary = pd.read_csv(
        output_dir / "reviews_curator_2.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    ).set_index("queue_id", drop=False)
    for _, progress_row in progress.iterrows():
        assert full_secondary.loc[progress_row["queue_id"]].to_dict() == (
            progress_row.to_dict()
        )


def test_completed_bundle_is_self_contained_and_byte_reproducible(tmp_path):
    checkpoint = _write_partial_checkpoint(tmp_path / "source")
    first_output = tmp_path / "first"
    _complete_checkpoint(checkpoint, first_output)
    replay_checkpoint: dict[str, Path | str] = {
        "primary": first_output / "reviews.tsv",
        "selection": first_output / "selection.tsv",
        "partial_secondary": first_output / "reviews_curator_2_partial.tsv",
        "partial_comparison": first_output / "review_comparison_partial.tsv",
        "partial_manifest": first_output / "dual_review_manifest_partial.json",
        "progress": first_output / "reviews_curator_2_completion_progress.tsv",
        "progress_manifest": first_output / "secondary_review_progress_manifest.json",
        "completion": first_output / "reviews_curator_2_completion.tsv",
        "checkpoint_sha": hashlib.sha256(
            (first_output / "dual_review_manifest_partial.json").read_bytes()
        ).hexdigest(),
        "progress_sha": hashlib.sha256(
            (first_output / "secondary_review_progress_manifest.json").read_bytes()
        ).hexdigest(),
    }
    second_output = tmp_path / "second"

    _complete_checkpoint(replay_checkpoint, second_output)

    assert {path.name for path in first_output.iterdir()} == {
        path.name for path in second_output.iterdir()
    }
    for first_path in first_output.iterdir():
        assert (second_output / first_path.name).read_bytes() == first_path.read_bytes()


def test_completed_bundle_supports_empty_partial_gene_comparison_and_date_guard(
    tmp_path,
):
    checkpoint = _write_partial_checkpoint(tmp_path / "source", zero_gene_evidence=True)
    output_dir = tmp_path / "completed"

    manifest = _complete_checkpoint(checkpoint, output_dir)

    comparison = pd.read_csv(output_dir / "review_comparison.tsv", sep="\t")
    assert comparison.empty
    assert manifest["compared_gene_count"] == 0
    assert manifest["status"] == "dual_review_complete_requires_human_adjudication"

    earlier_output = tmp_path / "earlier"
    with pytest.raises(ValueError, match="cannot precede the partial checkpoint"):
        _complete_checkpoint(
            checkpoint,
            earlier_output,
            assessed_date=date(2026, 7, 31),
        )
    assert not earlier_output.exists()


def test_completed_bundle_preserves_authenticated_whitespace_in_raw_rows(tmp_path):
    checkpoint = _write_partial_checkpoint(tmp_path / "source")
    partial_path = Path(checkpoint["partial_secondary"])
    partial = pd.read_csv(partial_path, sep="\t", dtype=str)
    partial.loc[0, "notes"] = "  authenticated checkpoint whitespace  "
    partial.loc[1, "notes"] = "NA"
    partial.to_csv(partial_path, sep="\t", index=False, lineterminator="\n")
    manifest = build_dual_review_manifest(
        checkpoint["primary"],
        partial_path,
        checkpoint["selection"],
        checkpoint["partial_comparison"],
        assessed_date=date(2026, 8, 1),
    )
    manifest_path = Path(checkpoint["partial_manifest"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint["checkpoint_sha"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    _refresh_progress_manifest(checkpoint)

    output_dir = tmp_path / "completed"
    _complete_checkpoint(checkpoint, output_dir)

    completed = pd.read_csv(
        output_dir / "reviews_curator_2.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    ).set_index("queue_id")
    queue_id = str(partial.loc[0, "queue_id"])
    assert completed.loc[queue_id, "notes"] == (
        "  authenticated checkpoint whitespace  "
    )
    second_queue_id = str(partial.loc[1, "queue_id"])
    assert completed.loc[second_queue_id, "notes"] == "NA"


def test_completed_bundle_manifest_is_independent_of_completion_source_basename(
    tmp_path,
):
    first_checkpoint = _write_partial_checkpoint(tmp_path / "first-source")
    second_checkpoint = _write_partial_checkpoint(tmp_path / "second-source")
    renamed_completion = (
        Path(second_checkpoint["completion"]).parent / "arbitrary-caller-name.tsv"
    )
    Path(second_checkpoint["completion"]).rename(renamed_completion)
    second_checkpoint["completion"] = renamed_completion
    first_output = tmp_path / "first-output"
    second_output = tmp_path / "second-output"

    first_manifest = _complete_checkpoint(first_checkpoint, first_output)
    second_manifest = _complete_checkpoint(second_checkpoint, second_output)

    assert first_manifest == second_manifest
    assert first_manifest["completion_lineage"]["input"]["filename"] == (
        "reviews_curator_2_completion.tsv"
    )
    for filename in (
        "reviews_curator_2_completion.tsv",
        "reviews_curator_2.tsv",
        "review_comparison.tsv",
        "dual_review_manifest.json",
    ):
        assert (first_output / filename).read_bytes() == (
            second_output / filename
        ).read_bytes()


@pytest.mark.parametrize(
    ("checkpoint_key", "renamed_filename"),
    [
        ("primary", "arbitrary-primary.tsv"),
        ("selection", "arbitrary-selection.tsv"),
        ("partial_secondary", "arbitrary-partial-secondary.tsv"),
        ("partial_comparison", "arbitrary-partial-comparison.tsv"),
    ],
)
def test_completed_bundle_rejects_noncanonical_predecessor_filenames(
    tmp_path, checkpoint_key, renamed_filename
):
    checkpoint = _write_partial_checkpoint(tmp_path / "source")
    original_path = Path(checkpoint[checkpoint_key])
    renamed_path = original_path.with_name(renamed_filename)
    original_path.rename(renamed_path)
    checkpoint[checkpoint_key] = renamed_path
    manifest = build_dual_review_manifest(
        checkpoint["primary"],
        checkpoint["partial_secondary"],
        checkpoint["selection"],
        checkpoint["partial_comparison"],
        assessed_date=date(2026, 8, 1),
    )
    manifest_path = Path(checkpoint["partial_manifest"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint["checkpoint_sha"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    with pytest.raises(ValueError, match="canonical predecessor filename"):
        _complete_checkpoint(checkpoint, tmp_path / "completed")
    assert not (tmp_path / "completed").exists()


def test_completed_bundle_rejects_non_tsv_completion_input(tmp_path):
    checkpoint = _write_partial_checkpoint(tmp_path / "source")
    completion_path = Path(checkpoint["completion"])
    completion = pd.read_csv(completion_path, sep="\t", dtype=str)
    csv_path = completion_path.with_suffix(".csv")
    completion.to_csv(csv_path, index=False, lineterminator="\n")
    checkpoint["completion"] = csv_path

    with pytest.raises(ValueError, match=r"completion reviews must use the \.tsv"):
        _complete_checkpoint(checkpoint, tmp_path / "completed")
    assert not (tmp_path / "completed").exists()


def test_secondary_review_progress_preserves_pending_rank_without_comparison(
    tmp_path,
):
    checkpoint = _write_partial_checkpoint(tmp_path / "source")
    completion = pd.read_csv(checkpoint["completion"], sep="\t", dtype=str)
    progress = completion.loc[completion["queue_rank"].ne("2")].reset_index(drop=True)
    progress_path = (
        Path(checkpoint["completion"]).parent
        / "reviews_curator_2_completion_progress.tsv"
    )
    progress.to_csv(progress_path, sep="\t", index=False, lineterminator="\n")

    manifest = _build_progress_manifest(checkpoint)

    assert manifest["status"] == (
        "secondary_review_completion_progress_requires_remaining_review_"
        "and_human_adjudication"
    )
    assert manifest["human_adjudication_required"] is True
    assert manifest["second_reviewed_screen_count"] == 9
    assert manifest["completion_progress_screen_count"] == 4
    assert manifest["pending_second_review_screen_count"] == 1
    assert manifest["pending_queue_ranks"] == [2]
    assert manifest["completion_progress"]["queue_ranks"] == [1, 3, 4, 5]
    assert manifest["released_label_count"] == 0
    assert manifest["benchmark_ready_count"] == 0
    assert "comparison" not in manifest


def test_checked_in_secondary_review_progress_is_exact_and_checksum_bound():
    observed = json.loads(COMPLETION_PROGRESS_MANIFEST.read_text(encoding="utf-8"))
    expected = build_secondary_review_progress_manifest(
        COMPLETION_PROGRESS,
        BATCH_DIR / "reviews.tsv",
        BATCH_DIR / "selection.tsv",
        PARTIAL_SECONDARY,
        PARTIAL_COMPARISON,
        PARTIAL_MANIFEST,
        assessed_date=date(2026, 8, 1),
        expected_predecessor_manifest_sha256=hashlib.sha256(
            PARTIAL_MANIFEST.read_bytes()
        ).hexdigest(),
    )

    assert observed == expected
    assert observed["completion_progress"]["queue_ranks"] == [1, 3, 4, 5]
    assert observed["pending_queue_ranks"] == [2]
    assert observed["second_reviewed_screen_count"] == 9
    assert observed["adjudicated_gene_count"] == 0
    assert observed["released_label_count"] == 0
    assert observed["benchmark_ready_count"] == 0


def test_checked_in_completed_dual_review_bundle_is_exact_and_unreleased(tmp_path):
    output_dir = tmp_path / "completed"
    manifest = write_completed_dual_review_bundle(
        BATCH_DIR / "reviews.tsv",
        COMPLETION_REVIEWS,
        COMPLETION_PROGRESS,
        COMPLETION_PROGRESS_MANIFEST,
        BATCH_DIR / "selection.tsv",
        PARTIAL_SECONDARY,
        PARTIAL_COMPARISON,
        PARTIAL_MANIFEST,
        output_dir,
        assessed_date=date(2026, 8, 2),
        expected_checkpoint_manifest_sha256=hashlib.sha256(
            PARTIAL_MANIFEST.read_bytes()
        ).hexdigest(),
        expected_progress_manifest_sha256=hashlib.sha256(
            COMPLETION_PROGRESS_MANIFEST.read_bytes()
        ).hexdigest(),
    )

    bundle_filenames = {
        "dual_review_manifest.json",
        "dual_review_manifest_partial.json",
        "review_comparison.tsv",
        "review_comparison_partial.tsv",
        "reviews.tsv",
        "reviews_curator_2.tsv",
        "reviews_curator_2_completion.tsv",
        "reviews_curator_2_completion_progress.tsv",
        "reviews_curator_2_partial.tsv",
        "secondary_review_progress_manifest.json",
        "selection.tsv",
    }
    assert {path.name for path in output_dir.iterdir()} == bundle_filenames
    for filename in bundle_filenames:
        assert (output_dir / filename).read_bytes() == (
            BATCH_DIR / filename
        ).read_bytes()

    assert manifest == json.loads(FULL_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == "dual_review_complete_requires_human_adjudication"
    assert manifest["second_reviewed_screen_count"] == 10
    assert manifest["pending_second_review_screen_count"] == 0
    assert manifest["compared_gene_count"] == 20
    assert manifest["comparison_status_counts"] == {
        "label_disagreement": 1,
        "provisional_agreement": 17,
        "single_curator_only": 2,
    }
    assert manifest["adjudication_status"] == "pending_human_adjudication"
    assert manifest["adjudicated_gene_count"] == 0
    assert manifest["released_label_count"] == 0
    assert manifest["benchmark_ready_count"] == 0

    completion = pd.read_csv(COMPLETION_REVIEWS, sep="\t", dtype=str)
    assert completion["queue_rank"].astype(int).tolist() == [1, 2, 3, 4, 5]
    rank_two = pd.read_csv(FULL_COMPARISON, sep="\t", dtype=str).loc[
        lambda frame: frame["queue_rank"].eq("2")
    ]
    assert rank_two["comparison_status"].value_counts().to_dict() == {
        "provisional_agreement": 4,
        "single_curator_only": 1,
    }
    assert rank_two["human_adjudication_required"].eq("True").all()


def test_secondary_review_progress_rejects_overlap_and_complete_set(tmp_path):
    overlap_checkpoint = _write_partial_checkpoint(tmp_path / "overlap")
    overlap = pd.read_csv(
        overlap_checkpoint["partial_secondary"], sep="\t", dtype=str
    ).iloc[[0]]
    overlap_path = (
        Path(overlap_checkpoint["completion"]).parent
        / "reviews_curator_2_completion_progress.tsv"
    )
    overlap.to_csv(overlap_path, sep="\t", index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="overlaps frozen rows"):
        _build_progress_manifest(overlap_checkpoint)

    complete_checkpoint = _write_partial_checkpoint(tmp_path / "complete")
    complete_path = (
        Path(complete_checkpoint["completion"]).parent
        / "reviews_curator_2_completion_progress.tsv"
    )
    Path(complete_checkpoint["completion"]).replace(complete_path)
    with pytest.raises(ValueError, match="complete second review must use"):
        _build_progress_manifest(
            complete_checkpoint,
            progress_path=complete_path,
        )


def test_secondary_review_progress_is_bound_to_frozen_checkpoint(tmp_path):
    checkpoint = _write_partial_checkpoint(tmp_path / "source")
    completion = pd.read_csv(checkpoint["completion"], sep="\t", dtype=str).iloc[[0]]
    progress_path = (
        Path(checkpoint["completion"]).parent
        / "reviews_curator_2_completion_progress.tsv"
    )
    completion.to_csv(progress_path, sep="\t", index=False, lineterminator="\n")
    frozen_path = Path(checkpoint["partial_secondary"])
    frozen_path.write_bytes(frozen_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="exact deterministic derivation"):
        _build_progress_manifest(checkpoint)


def test_secondary_review_progress_rejects_primary_curator_identity(tmp_path):
    checkpoint = _write_partial_checkpoint(tmp_path / "source")
    progress = pd.read_csv(checkpoint["completion"], sep="\t", dtype=str).iloc[[0]]
    progress["curator"] = _reviews()["curator"].iloc[0]
    progress_path = (
        Path(checkpoint["completion"]).parent
        / "reviews_curator_2_completion_progress.tsv"
    )
    progress.to_csv(progress_path, sep="\t", index=False, lineterminator="\n")

    with pytest.raises(ValueError, match="overlap primary curators"):
        _build_progress_manifest(checkpoint)


def test_secondary_review_progress_rejects_forged_predecessor_fields(tmp_path):
    checkpoint = _write_partial_checkpoint(tmp_path / "source")
    manifest_path = Path(checkpoint["partial_manifest"])
    forged = json.loads(manifest_path.read_text(encoding="utf-8"))
    forged["primary_reviews"]["sha256"] = "0" * 64
    forged["comparison"]["sha256"] = "f" * 64
    forged["compared_gene_count"] = 999_999
    manifest_path.write_text(
        json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint["checkpoint_sha"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    with pytest.raises(ValueError, match="exact deterministic derivation"):
        _build_progress_manifest(checkpoint)


def test_secondary_review_progress_rejects_primary_review_id_reuse(tmp_path):
    checkpoint = _write_partial_checkpoint(tmp_path / "source")
    progress_path = Path(checkpoint["progress"])
    progress = pd.read_csv(progress_path, sep="\t", dtype=str)
    primary = pd.read_csv(checkpoint["primary"], sep="\t", dtype=str).set_index(
        "queue_id"
    )
    first_queue_id = progress.loc[0, "queue_id"]
    progress.loc[0, "review_id"] = primary.loc[first_queue_id, "review_id"]
    progress.to_csv(progress_path, sep="\t", index=False, lineterminator="\n")

    with pytest.raises(ValueError, match="reuses primary review IDs"):
        _build_progress_manifest(checkpoint)


def test_secondary_review_progress_rejects_empty_predecessor_date(tmp_path):
    checkpoint = _write_partial_checkpoint(tmp_path / "source")
    manifest_path = Path(checkpoint["partial_manifest"])
    malformed = json.loads(manifest_path.read_text(encoding="utf-8"))
    malformed["assessed_date"] = ""
    manifest_path.write_text(
        json.dumps(malformed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint["checkpoint_sha"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    with pytest.raises(ValueError, match="requires a valid assessed_date"):
        _build_progress_manifest(checkpoint)


def test_secondary_review_progress_uses_one_read_snapshots(tmp_path, monkeypatch):
    checkpoint = _write_partial_checkpoint(tmp_path / "source")
    progress_path = Path(checkpoint["progress"])
    predecessor_path = Path(checkpoint["partial_manifest"])
    expected_progress = progress_path.read_bytes()
    expected_predecessor = predecessor_path.read_bytes()
    input_paths = {
        Path(checkpoint[key])
        for key in (
            "progress",
            "primary",
            "selection",
            "partial_secondary",
            "partial_comparison",
            "partial_manifest",
        )
    }
    read_counts = {path: 0 for path in input_paths}
    original_read_bytes = Path.read_bytes

    def mutate_after_snapshot(path):
        content = original_read_bytes(path)
        if path in read_counts:
            read_counts[path] += 1
        if path == progress_path and read_counts[path] == 1:
            path.write_bytes(b"progress changed after snapshot\n")
        if path == predecessor_path and read_counts[path] == 1:
            path.write_bytes(b"manifest changed after snapshot\n")
        return content

    monkeypatch.setattr(Path, "read_bytes", mutate_after_snapshot)

    manifest = _build_progress_manifest(checkpoint)

    assert set(read_counts.values()) == {1}
    assert (
        manifest["completion_progress"]["sha256"]
        == hashlib.sha256(expected_progress).hexdigest()
    )
    assert manifest["predecessor_checkpoint"]["manifest"]["sha256"] == (
        hashlib.sha256(expected_predecessor).hexdigest()
    )


def test_completed_bundle_requires_expected_checkpoint_byte_sha(tmp_path):
    checkpoint = _write_partial_checkpoint(tmp_path)
    checkpoint["checkpoint_sha"] = "0" * 64

    with pytest.raises(ValueError, match="manifest SHA-256 does not match"):
        _complete_checkpoint(checkpoint, tmp_path / "completed")
    assert not (tmp_path / "completed").exists()


def test_completed_bundle_requires_expected_progress_manifest_sha(tmp_path):
    checkpoint = _write_partial_checkpoint(tmp_path)
    checkpoint["progress_sha"] = "0" * 64

    with pytest.raises(ValueError, match="progress checkpoint manifest SHA-256"):
        _complete_checkpoint(checkpoint, tmp_path / "completed")
    assert not (tmp_path / "completed").exists()


def test_completed_bundle_rejects_changed_authenticated_progress_row(tmp_path):
    checkpoint = _write_partial_checkpoint(tmp_path)
    completion_path = Path(checkpoint["completion"])
    completion = pd.read_csv(
        completion_path, sep="\t", dtype=str, keep_default_na=False
    )
    progress_ids = set(
        pd.read_csv(checkpoint["progress"], sep="\t", dtype=str)["queue_id"]
    )
    changed_index = completion.index[completion["queue_id"].isin(progress_ids)][0]
    completion.loc[changed_index, "notes"] = "changed after progress checkpoint"
    completion.to_csv(completion_path, sep="\t", index=False, lineterminator="\n")

    with pytest.raises(ValueError, match="changed authenticated progress cells"):
        _complete_checkpoint(checkpoint, tmp_path / "completed")
    assert not (tmp_path / "completed").exists()


def test_completed_bundle_rejects_tampered_progress_manifest(tmp_path):
    checkpoint = _write_partial_checkpoint(tmp_path)
    manifest_path = Path(checkpoint["progress_manifest"])
    observed = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed["pending_second_review_screen_count"] = 2
    manifest_path.write_text(
        json.dumps(observed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint["progress_sha"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="progress checkpoint manifest is not"):
        _complete_checkpoint(checkpoint, tmp_path / "completed")
    assert not (tmp_path / "completed").exists()


def test_completed_bundle_rejects_valid_json_tampered_checkpoint_manifest(tmp_path):
    checkpoint = _write_partial_checkpoint(tmp_path)
    manifest_path = Path(checkpoint["partial_manifest"])
    observed = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed["pending_second_review_screen_count"] = 4
    manifest_path.write_text(
        json.dumps(observed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint["checkpoint_sha"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    with pytest.raises(ValueError, match="exact deterministic derivation"):
        _complete_checkpoint(checkpoint, tmp_path / "completed")
    assert not (tmp_path / "completed").exists()


def test_completed_bundle_rejects_tampered_partial_comparison(tmp_path):
    checkpoint = _write_partial_checkpoint(tmp_path)
    comparison_path = Path(checkpoint["partial_comparison"])
    comparison = pd.read_csv(comparison_path, sep="\t", dtype=str)
    comparison.loc[0, "notes"] = "TAMPERED"
    comparison.to_csv(comparison_path, sep="\t", index=False, lineterminator="\n")

    with pytest.raises(ValueError, match="not the deterministic derivation"):
        _complete_checkpoint(checkpoint, tmp_path / "completed")
    assert not (tmp_path / "completed").exists()


def test_completed_bundle_rejects_partial_review_row_mutation(tmp_path):
    checkpoint = _write_partial_checkpoint(tmp_path)
    partial_path = Path(checkpoint["partial_secondary"])
    partial = pd.read_csv(partial_path, sep="\t", dtype=str)
    partial.loc[0, "notes"] = "checkpoint row changed"
    partial.to_csv(partial_path, sep="\t", index=False, lineterminator="\n")

    with pytest.raises(ValueError, match="exact deterministic derivation"):
        _complete_checkpoint(checkpoint, tmp_path / "completed")
    assert not (tmp_path / "completed").exists()


def test_completed_bundle_rejects_overlap_with_frozen_partial_rows(tmp_path):
    checkpoint = _write_partial_checkpoint(tmp_path)
    completion_path = Path(checkpoint["completion"])
    completion = pd.read_csv(completion_path, sep="\t", dtype=str)
    partial = pd.read_csv(checkpoint["partial_secondary"], sep="\t", dtype=str)
    completion.iloc[-1] = partial.iloc[0]
    completion.to_csv(completion_path, sep="\t", index=False, lineterminator="\n")

    with pytest.raises(ValueError, match="overlap frozen partial rows"):
        _complete_checkpoint(checkpoint, tmp_path / "completed")


def test_completed_bundle_rejects_missing_and_unselected_completion_rows(tmp_path):
    missing_checkpoint = _write_partial_checkpoint(tmp_path / "missing")
    missing_path = Path(missing_checkpoint["completion"])
    missing = pd.read_csv(missing_path, sep="\t", dtype=str).iloc[:-1]
    missing.to_csv(missing_path, sep="\t", index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="exact selection complement"):
        _complete_checkpoint(missing_checkpoint, tmp_path / "missing-output")

    unexpected_checkpoint = _write_partial_checkpoint(tmp_path / "unexpected")
    unexpected_path = Path(unexpected_checkpoint["completion"])
    unexpected = pd.read_csv(unexpected_path, sep="\t", dtype=str)
    extra = unexpected.iloc[0].copy()
    extra["review_id"] = "unexpected-review-id"
    extra["queue_id"] = "unexpected-queue-id"
    extra["screen_id"] = "unexpected-screen-id"
    unexpected = pd.concat([unexpected, extra.to_frame().T], ignore_index=True)
    unexpected.to_csv(unexpected_path, sep="\t", index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="unselected queue IDs"):
        _complete_checkpoint(unexpected_checkpoint, tmp_path / "unexpected-output")


def test_completed_bundle_rejects_duplicate_completion_row(tmp_path):
    checkpoint = _write_partial_checkpoint(tmp_path)
    completion_path = Path(checkpoint["completion"])
    completion = pd.read_csv(completion_path, sep="\t", dtype=str)
    completion = pd.concat([completion, completion.iloc[[0]]], ignore_index=True)
    completion.to_csv(completion_path, sep="\t", index=False, lineterminator="\n")

    with pytest.raises(ValueError, match="completion secondary reviews failed"):
        _complete_checkpoint(checkpoint, tmp_path / "completed")


@pytest.mark.parametrize("mutated_input", ["selection", "primary"])
def test_completed_bundle_rejects_changed_checkpoint_inputs(tmp_path, mutated_input):
    checkpoint = _write_partial_checkpoint(tmp_path)
    path = Path(checkpoint[mutated_input])
    frame = pd.read_csv(path, sep="\t", dtype=str)
    frame.loc[0, "source_id"] = "CHANGED-SOURCE"
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")

    with pytest.raises(ValueError):
        _complete_checkpoint(checkpoint, tmp_path / "completed")
    assert not (tmp_path / "completed").exists()


def test_completed_bundle_failure_during_staging_is_atomic(tmp_path, monkeypatch):
    checkpoint = _write_partial_checkpoint(tmp_path)
    output_dir = tmp_path / "completed"
    checkpoint_bytes = {
        key: Path(checkpoint[key]).read_bytes()
        for key in ("partial_secondary", "partial_comparison", "partial_manifest")
    }
    original_builder = curation_module.build_dual_review_manifest
    calls = 0

    def fail_final_manifest(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected final-manifest failure")
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(
        curation_module, "build_dual_review_manifest", fail_final_manifest
    )
    with pytest.raises(RuntimeError, match="injected final-manifest failure"):
        _complete_checkpoint(checkpoint, output_dir)

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".completed.work-*"))
    assert not (tmp_path / ".completed.publish.lock").exists()
    for key, expected in checkpoint_bytes.items():
        assert Path(checkpoint[key]).read_bytes() == expected


def test_completed_bundle_exclusive_lock_rejects_recursive_writer(
    tmp_path, monkeypatch
):
    checkpoint = _write_partial_checkpoint(tmp_path / "source")
    output_dir = tmp_path / "completed"
    original_read_bytes = Path.read_bytes
    nested_errors: list[Exception] = []
    triggered = False

    def invoke_nested_writer(path):
        nonlocal triggered
        if not triggered and path == checkpoint["primary"]:
            triggered = True
            try:
                _complete_checkpoint(checkpoint, output_dir)
            except Exception as exc:  # the assertion below verifies the exact failure
                nested_errors.append(exc)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", invoke_nested_writer)

    _complete_checkpoint(checkpoint, output_dir)

    assert len(nested_errors) == 1
    assert isinstance(nested_errors[0], FileExistsError)
    assert "publisher lock already exists" in str(nested_errors[0])
    assert not (tmp_path / ".completed.publish.lock").exists()


def test_completed_bundle_does_not_replace_destination_injected_before_rename(
    tmp_path, monkeypatch
):
    checkpoint = _write_partial_checkpoint(tmp_path / "source")
    output_dir = tmp_path / "completed"
    calls = 0
    original_lexists = curation_module._path_lexists

    def inject_destination(path):
        nonlocal calls
        if path == output_dir:
            calls += 1
            if calls == 2:
                output_dir.mkdir()
                (output_dir / "sentinel.txt").write_text(
                    "out-of-band destination", encoding="utf-8"
                )
                return True
        return original_lexists(path)

    monkeypatch.setattr(curation_module, "_path_lexists", inject_destination)

    with pytest.raises(FileExistsError, match="destination appeared while staging"):
        _complete_checkpoint(checkpoint, output_dir)

    assert (output_dir / "sentinel.txt").read_text(encoding="utf-8") == (
        "out-of-band destination"
    )
    assert {path.name for path in output_dir.iterdir()} == {"sentinel.txt"}
    assert not (tmp_path / ".completed.publish.lock").exists()


def test_completed_bundle_uses_one_read_snapshots_after_input_mutation(
    tmp_path, monkeypatch
):
    checkpoint = _write_partial_checkpoint(tmp_path / "source")
    output_dir = tmp_path / "completed"
    selection_path = Path(checkpoint["selection"])
    completion_path = Path(checkpoint["completion"])
    expected_selection = selection_path.read_bytes()
    expected_completion = completion_path.read_bytes()
    input_paths = {
        Path(checkpoint[key])
        for key in (
            "primary",
            "selection",
            "partial_secondary",
            "partial_comparison",
            "partial_manifest",
            "progress",
            "progress_manifest",
            "completion",
        )
    }
    read_counts = {path: 0 for path in input_paths}
    original_read_bytes = Path.read_bytes

    def mutate_after_snapshot(path):
        content = original_read_bytes(path)
        if path in read_counts:
            read_counts[path] += 1
        if path == selection_path and read_counts[path] == 1:
            path.write_bytes(b"selection changed after snapshot\n")
        if path == completion_path and read_counts[path] == 1:
            path.write_bytes(b"completion changed after snapshot\n")
        return content

    monkeypatch.setattr(Path, "read_bytes", mutate_after_snapshot)

    _complete_checkpoint(checkpoint, output_dir)

    assert set(read_counts.values()) == {1}
    assert selection_path.read_bytes() != expected_selection
    assert completion_path.read_bytes() != expected_completion
    assert (output_dir / "selection.tsv").read_bytes() == expected_selection
    assert (
        output_dir / "reviews_curator_2_completion.tsv"
    ).read_bytes() == expected_completion


def test_complete_curation_reviews_cli_writes_atomic_full_bundle(tmp_path, capsys):
    checkpoint = _write_partial_checkpoint(tmp_path)
    output_dir = tmp_path / "completed"
    args = build_parser().parse_args(
        [
            "complete-curation-reviews",
            "--primary-reviews",
            str(checkpoint["primary"]),
            "--completion-reviews",
            str(checkpoint["completion"]),
            "--progress-reviews",
            str(checkpoint["progress"]),
            "--progress-manifest",
            str(checkpoint["progress_manifest"]),
            "--selection",
            str(checkpoint["selection"]),
            "--partial-secondary-reviews",
            str(checkpoint["partial_secondary"]),
            "--partial-comparison",
            str(checkpoint["partial_comparison"]),
            "--partial-manifest",
            str(checkpoint["partial_manifest"]),
            "--expected-checkpoint-manifest-sha256",
            str(checkpoint["checkpoint_sha"]),
            "--expected-progress-manifest-sha256",
            str(checkpoint["progress_sha"]),
            "--output-dir",
            str(output_dir),
            "--assessed-date",
            "2026-08-01",
        ]
    )

    assert args.func(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "dual_review_complete_requires_human_adjudication"
    assert report["released_label_count"] == 0
    assert (output_dir / "reviews_curator_2.tsv").is_file()
    assert (output_dir / "review_comparison.tsv").is_file()
    assert (output_dir / "dual_review_manifest.json").is_file()


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
