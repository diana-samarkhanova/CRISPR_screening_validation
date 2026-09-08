from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

import crispr_evidencerank.cli as cli_module
from crispr_evidencerank.cli import build_parser
from crispr_evidencerank.screen_report import rank_screen


def _mageck_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["GENE1", "GENE2", "GENE3"],
            "pos|score": [0.001, 0.02, 0.5],
            "pos|fdr": [0.01, 0.05, 0.8],
            "pos|rank": [1, 2, 3],
            "pos|lfc": [2.0, 1.0, 0.0],
            "neg|score": [0.4, 0.03, 0.002],
            "neg|fdr": [0.7, 0.08, 0.02],
            "neg|rank": [3, 2, 1],
            "neg|lfc": [0.0, -1.0, -2.0],
        }
    )


def _counts_and_samples() -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.DataFrame(
        {
            "sgrna_id": ["g1", "g2", "g3", "g4", "g5", "g6"],
            "gene_symbol": ["GENE1", "GENE1", "GENE2", "GENE2", "GENE3", "GENE3"],
            "c1": [100, 120, 100, 120, 100, 120],
            "c2": [110, 115, 110, 115, 110, 115],
            "t1": [500, 450, 25, 30, 100, 120],
            "t2": [480, 430, 30, 35, 110, 115],
        }
    )
    samples = pd.DataFrame(
        {
            "sample_id": ["c1", "c2", "t1", "t2"],
            "screen_id": ["S1"] * 4,
            "contrast_id": ["olaparib_vs_vehicle"] * 4,
            "condition_role": ["control", "control", "treatment", "treatment"],
            "replicate": [1, 2, 1, 2],
        }
    )
    return counts, samples


def test_mageck_report_maps_tails_without_claiming_validation() -> None:
    result = rank_screen(
        mageck_summary=_mageck_summary(),
        screen_id="S1",
        contrast_id="olaparib_vs_vehicle",
        positive_tail_means="resistance",
    )
    ranked = result.ranked_candidates
    assert set(ranked["phenotype_direction"]) == {"resistance", "sensitization"}
    assert (
        ranked.loc[ranked["analysis_tail"].eq("mageck_pos"), "phenotype_direction"]
        .eq("resistance")
        .all()
    )
    assert (
        ranked.loc[ranked["analysis_tail"].eq("mageck_neg"), "phenotype_direction"]
        .eq("sensitization")
        .all()
    )
    assert ranked["ranking_type"].eq("screen_signal_baseline").all()
    assert not any("validation" in column.lower() for column in ranked.columns)
    assert "**not** a calibrated\nvalidation probability" in result.report_markdown
    assert "| resistance | 1" in result.report_markdown
    assert "| sensitization | 1" in result.report_markdown


def test_mageck_report_preserves_available_guide_counts() -> None:
    summary = _mageck_summary()
    summary["num"] = [4, 5, 6]
    summary["pos|goodsgrna"] = [3, 4, 5]
    summary["neg|goodsgrna"] = [2, 3, 4]
    result = rank_screen(
        mageck_summary=summary,
        screen_id="S1",
        contrast_id="C1",
        positive_tail_means="resistance",
    )
    ranked = result.ranked_candidates
    assert set(ranked["mageck_input_sgrna_n"]) == {4.0, 5.0, 6.0}
    assert set(ranked["mageck_good_sgrna_n"]) == {2.0, 3.0, 4.0, 5.0}


def test_mageck_rejects_impossible_good_guide_count() -> None:
    summary = _mageck_summary()
    summary["num"] = [2, 2, 2]
    summary["pos|goodsgrna"] = [3, 2, 2]
    summary["neg|goodsgrna"] = [2, 2, 2]
    with pytest.raises(ValueError, match="cannot exceed"):
        rank_screen(
            mageck_summary=summary,
            screen_id="S1",
            contrast_id="C1",
            positive_tail_means="resistance",
        )


def test_count_report_ranks_positive_and_negative_lfc_separately() -> None:
    counts, samples = _counts_and_samples()
    result = rank_screen(
        counts=counts,
        samples=samples,
        positive_lfc_means="resistance",
    )
    ranked = result.ranked_candidates.set_index("gene_symbol")
    assert ranked.loc["GENE1", "phenotype_direction"] == "resistance"
    assert ranked.loc["GENE2", "phenotype_direction"] == "sensitization"
    assert ranked.loc["GENE3", "phenotype_direction"] == "neutral"
    assert pd.isna(ranked.loc["GENE3", "screen_signal_rank"])
    assert (
        ranked.loc[["GENE1", "GENE2"], "screen_signal_percentile_scope"]
        .eq("observed_rows_within_screen_contrast_direction")
        .all()
    )
    assert ranked.loc["GENE3", "screen_signal_percentile_scope"] == "neutral_not_ranked"
    assert result.qc_summary["mode"] == "counts"
    assert (
        "no MAGeCK FDR: ranking is count-effect-only" in (result.qc_summary["warnings"])
    )


def test_count_mode_rejects_conflicting_declared_and_sample_sheet_ids() -> None:
    counts, samples = _counts_and_samples()
    with pytest.raises(ValueError, match="does not match sample sheet"):
        rank_screen(
            counts=counts,
            samples=samples,
            screen_id="WRONG",
            contrast_id="olaparib_vs_vehicle",
            positive_lfc_means="resistance",
        )
    with pytest.raises(ValueError, match="does not match sample sheet"):
        rank_screen(
            counts=counts,
            samples=samples,
            screen_id="S1",
            contrast_id="WRONG",
            positive_lfc_means="resistance",
        )


def test_count_report_rejects_unknown_condition_role() -> None:
    counts, samples = _counts_and_samples()
    samples.loc[0, "condition_role"] = "typo"
    with pytest.raises(ValueError, match="must be control or treatment"):
        rank_screen(
            counts=counts,
            samples=samples,
            positive_lfc_means="resistance",
        )


def test_count_report_normalizes_identifier_whitespace_for_qc() -> None:
    counts, samples = _counts_and_samples()
    counts = counts.rename(columns={"c1": " c1 ", "t1": " t1 "})
    counts["sgrna_id"] = " " + counts["sgrna_id"] + " "
    counts["gene_symbol"] = " " + counts["gene_symbol"] + " "
    samples["sample_id"] = " " + samples["sample_id"] + " "
    samples["screen_id"] = " " + samples["screen_id"] + " "
    samples["contrast_id"] = " " + samples["contrast_id"] + " "
    samples["condition_role"] = " " + samples["condition_role"] + " "
    result = rank_screen(
        counts=counts,
        samples=samples,
        positive_lfc_means="resistance",
    )
    assert set(result.ranked_candidates["gene_symbol"]) == {
        "GENE1",
        "GENE2",
        "GENE3",
    }
    assert result.qc_summary["replicate_sample_counts"] == [
        {
            "screen_id": "S1",
            "contrast_id": "olaparib_vs_vehicle",
            "control_sample_n": 2,
            "treatment_sample_n": 2,
        }
    ]


def test_mageck_rejects_gene_collision_after_whitespace_normalization() -> None:
    summary = _mageck_summary()
    summary.loc[1, "id"] = " GENE1 "
    with pytest.raises(ValueError, match="duplicate gene rows"):
        rank_screen(
            mageck_summary=summary,
            screen_id="S1",
            contrast_id="C1",
            positive_tail_means="resistance",
        )


def test_positive_lfc_semantics_can_be_explicitly_reversed() -> None:
    counts, samples = _counts_and_samples()
    result = rank_screen(
        counts=counts,
        samples=samples,
        positive_lfc_means="sensitization",
    )
    ranked = result.ranked_candidates.set_index("gene_symbol")
    assert ranked.loc["GENE1", "phenotype_direction"] == "sensitization"
    assert ranked.loc["GENE2", "phenotype_direction"] == "resistance"


def test_single_replicate_arms_emit_explicit_qc_warning() -> None:
    counts, samples = _counts_and_samples()
    samples = samples.loc[samples["sample_id"].isin(["c1", "t1"])].copy()
    counts = counts.drop(columns=["c2", "t2"])
    result = rank_screen(
        counts=counts,
        samples=samples,
        positive_lfc_means="resistance",
    )
    assert result.qc_summary["replicate_sample_counts"] == [
        {
            "screen_id": "S1",
            "contrast_id": "olaparib_vs_vehicle",
            "control_sample_n": 1,
            "treatment_sample_n": 1,
        }
    ]
    assert any(
        "fewer than 2 samples" in warning for warning in result.qc_summary["warnings"]
    )


def test_combined_report_keeps_mageck_rank_and_adds_guide_qc() -> None:
    counts, samples = _counts_and_samples()
    result = rank_screen(
        mageck_summary=_mageck_summary(),
        counts=counts,
        samples=samples,
        screen_id="S1",
        contrast_id="olaparib_vs_vehicle",
        positive_tail_means="resistance",
        positive_lfc_means="resistance",
    )
    ranked = result.ranked_candidates
    assert result.qc_summary["mode"] == "mageck_plus_counts"
    assert "mageck_rank" in ranked
    assert "guide_n" in ranked
    assert "mageck_guide_direction_agreement" in ranked
    assert ranked["screen_signal_rank_source"].eq("mageck_native_rank").all()


def test_combined_report_marks_missing_guide_qc_as_unavailable() -> None:
    counts, samples = _counts_and_samples()
    counts = counts.loc[counts["gene_symbol"].eq("GENE1")].copy()
    result = rank_screen(
        mageck_summary=_mageck_summary(),
        counts=counts,
        samples=samples,
        screen_id="S1",
        contrast_id="olaparib_vs_vehicle",
        positive_tail_means="resistance",
        positive_lfc_means="resistance",
    )
    missing = result.ranked_candidates.loc[
        result.ranked_candidates["gene_symbol"].isin(["GENE2", "GENE3"]),
        "mageck_guide_direction_agreement",
    ]
    assert missing.isna().all()
    assert result.qc_summary["guide_qc_gene_coverage_n"] == 1
    assert result.qc_summary["guide_qc_gene_coverage_fraction"] == pytest.approx(1 / 3)
    assert any(
        "guide-level QC is missing" in value for value in result.qc_summary["warnings"]
    )


def test_combined_report_rejects_conflicting_direction_semantics() -> None:
    counts, samples = _counts_and_samples()
    with pytest.raises(ValueError, match="same phenotype direction"):
        rank_screen(
            mageck_summary=_mageck_summary(),
            counts=counts,
            samples=samples,
            screen_id="S1",
            contrast_id="olaparib_vs_vehicle",
            positive_tail_means="resistance",
            positive_lfc_means="sensitization",
        )


def test_report_requires_explicit_direction_semantics() -> None:
    with pytest.raises(ValueError, match="positive_tail_means"):
        rank_screen(
            mageck_summary=_mageck_summary(),
            screen_id="S1",
            contrast_id="C1",
        )
    counts, samples = _counts_and_samples()
    with pytest.raises(ValueError, match="positive_lfc_means"):
        rank_screen(counts=counts, samples=samples)


def test_cli_writes_self_contained_bundle_with_checksums(tmp_path) -> None:
    mageck_path = tmp_path / "mageck.tsv"
    output_dir = tmp_path / "report"
    _mageck_summary().to_csv(mageck_path, sep="\t", index=False)
    args = build_parser().parse_args(
        [
            "rank-screen",
            "--mageck-summary",
            str(mageck_path),
            "--screen-id",
            "S1",
            "--contrast-id",
            "olaparib_vs_vehicle",
            "--positive-tail-means",
            "resistance",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert args.func(args) == 0
    assert {path.name for path in output_dir.iterdir()} == {
        "ranked_candidates.tsv",
        "qc_summary.json",
        "run_manifest.json",
        "report.md",
    }
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert len(manifest["inputs"]["mageck_summary"]["sha256"]) == 64
    assert manifest["inputs"]["mageck_summary"]["filename"] == "mageck.tsv"
    assert manifest["report_type"] == "screen_signal_baseline"
    for output in manifest["outputs"].values():
        assert (
            hashlib.sha256((output_dir / output["filename"]).read_bytes()).hexdigest()
            == output["sha256"]
        )


def test_mageck_nonfinite_values_are_rejected() -> None:
    bad = _mageck_summary()
    bad.loc[0, "pos|fdr"] = float("inf")
    with pytest.raises(ValueError, match="finite numeric"):
        rank_screen(
            mageck_summary=bad,
            screen_id="S1",
            contrast_id="C1",
            positive_tail_means="resistance",
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("pos|rank", 0, "positive integers"),
        ("pos|rank", 1.5, "positive integers"),
        ("pos|p-value", -0.1, "p-values"),
        ("pos|p-value", 1.1, "p-values"),
        ("pos|score", -0.1, "RRA scores"),
        ("pos|score", 1.1, "RRA scores"),
    ],
)
def test_mageck_rank_and_pvalue_bounds_are_enforced(
    column: str,
    value: float,
    message: str,
) -> None:
    bad = _mageck_summary()
    if column not in bad:
        bad[column] = [0.01, 0.02, 0.03]
    else:
        bad[column] = bad[column].astype(float)
    bad.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        rank_screen(
            mageck_summary=bad,
            screen_id="S1",
            contrast_id="C1",
            positive_tail_means="resistance",
        )


def test_mageck_missing_gene_identifier_is_rejected() -> None:
    bad = _mageck_summary()
    bad.loc[0, "id"] = None
    with pytest.raises(ValueError, match="cannot be missing"):
        rank_screen(
            mageck_summary=bad,
            screen_id="S1",
            contrast_id="C1",
            positive_tail_means="resistance",
        )


def test_subset_native_ranks_produce_bounded_local_percentiles() -> None:
    subset = _mageck_summary().iloc[:2].copy()
    subset["pos|rank"] = [100, 101]
    subset["neg|rank"] = [101, 100]
    result = rank_screen(
        mageck_summary=subset,
        screen_id="S1",
        contrast_id="C1",
        positive_tail_means="resistance",
    )
    percentiles = result.ranked_candidates["screen_signal_percentile"]
    assert percentiles.between(0, 1).all()
    assert (
        result.ranked_candidates["screen_signal_percentile_scope"]
        .eq("observed_rows_within_tail")
        .all()
    )
    assert any(
        "percentile is local" in warning for warning in result.qc_summary["warnings"]
    )


def test_partial_native_rank_uses_a_complete_fallback_metric() -> None:
    summary = _mageck_summary()
    summary["pos|rank"] = [1.0, None, None]
    summary["pos|fdr"] = [0.01, None, None]
    summary["pos|p-value"] = [0.01, 0.02, 0.03]
    result = rank_screen(
        mageck_summary=summary,
        screen_id="S1",
        contrast_id="C1",
        positive_tail_means="resistance",
    )
    positive = result.ranked_candidates.loc[
        result.ranked_candidates["analysis_tail"].eq("mageck_pos")
    ]
    assert positive["screen_signal_rank"].notna().all()
    assert positive["screen_signal_rank_source"].eq("derived_from_mageck_p_value").all()


def test_existing_screen_bundle_is_preserved(tmp_path) -> None:
    mageck_path = tmp_path / "mageck.tsv"
    output_dir = tmp_path / "report"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    _mageck_summary().to_csv(mageck_path, sep="\t", index=False)
    args = build_parser().parse_args(
        [
            "rank-screen",
            "--mageck-summary",
            str(mageck_path),
            "--screen-id",
            "S1",
            "--contrast-id",
            "C1",
            "--positive-tail-means",
            "resistance",
            "--output-dir",
            str(output_dir),
        ]
    )
    with pytest.raises(FileExistsError, match="output directory exists"):
        args.func(args)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_input_mutation_aborts_before_bundle_publish(tmp_path, monkeypatch) -> None:
    mageck_path = tmp_path / "mageck.tsv"
    output_dir = tmp_path / "report"
    _mageck_summary().to_csv(mageck_path, sep="\t", index=False)
    original_rank_screen = cli_module.rank_screen

    def mutate_after_compute(**kwargs):
        result = original_rank_screen(**kwargs)
        mageck_path.write_bytes(mageck_path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(cli_module, "rank_screen", mutate_after_compute)
    args = build_parser().parse_args(
        [
            "rank-screen",
            "--mageck-summary",
            str(mageck_path),
            "--screen-id",
            "S1",
            "--contrast-id",
            "C1",
            "--positive-tail-means",
            "resistance",
            "--output-dir",
            str(output_dir),
        ]
    )
    with pytest.raises(ValueError, match="changed during the run"):
        args.func(args)
    assert not output_dir.exists()


def test_cli_preserves_leading_zero_sample_identifiers(tmp_path) -> None:
    counts_path = tmp_path / "counts.tsv"
    samples_path = tmp_path / "samples.tsv"
    output_dir = tmp_path / "report"
    pd.DataFrame(
        {
            "sgrna_id": ["0001", "0002"],
            "gene_symbol": ["GENE1", "GENE1"],
            "001": [100, 120],
            "002": [300, 360],
        }
    ).to_csv(counts_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "sample_id": ["001", "002"],
            "screen_id": ["0007", "0007"],
            "contrast_id": ["0009", "0009"],
            "condition_role": ["control", "treatment"],
            "replicate": [1, 1],
        }
    ).to_csv(samples_path, sep="\t", index=False)
    args = build_parser().parse_args(
        [
            "rank-screen",
            "--counts",
            str(counts_path),
            "--samples",
            str(samples_path),
            "--positive-lfc-means",
            "resistance",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert args.func(args) == 0
    ranked = pd.read_csv(output_dir / "ranked_candidates.tsv", sep="\t", dtype=str)
    assert ranked["screen_id"].unique().tolist() == ["0007"]
    assert ranked["contrast_id"].unique().tolist() == ["0009"]
