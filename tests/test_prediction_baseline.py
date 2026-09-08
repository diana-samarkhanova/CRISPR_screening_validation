import hashlib
import json

import pandas as pd
import pytest

from crispr_evidencerank.prediction_baseline import verify_prediction_baseline


def _write_report(directory, ranked, **parameters):
    directory.mkdir(exist_ok=True)
    path = directory / "ranked_candidates.tsv"
    ranked.to_csv(path, sep="\t", index=False)
    manifest = {
        "report_type": "screen_signal_baseline",
        "parameters": {
            "positive_tail_means": "resistance",
            "positive_lfc_means": None,
            **parameters,
        },
        "outputs": {
            "ranked_candidates": {
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        },
    }
    return _write_manifest(directory, manifest)


def _write_manifest(directory, manifest):
    content = json.dumps(manifest).encode()
    (directory / "run_manifest.json").write_bytes(content)
    return hashlib.sha256(content).hexdigest()


@pytest.fixture
def baseline(tmp_path):
    ranked = pd.DataFrame(
        {
            "screen_id": ["001"] * 4,
            "contrast_id": ["001"] * 4,
            "gene_symbol": ["NA", "B", "NA", "B"],
            "phenotype_direction": ["resistance"] * 2 + ["sensitization"] * 2,
            "screen_signal_rank": [1, 2, 2, 1],
            "screen_signal_rank_source": ["mageck_native_rank"] * 4,
            "analysis_tail": ["mageck_pos"] * 2 + ["mageck_neg"] * 2,
            "ranking_type": ["screen_signal_baseline"] * 4,
        }
    )
    report = tmp_path / "report"
    digest = _write_report(report, ranked)
    candidates = ranked.drop(columns=["analysis_tail", "ranking_type"]).copy()
    return ranked, candidates, report, digest


def test_baseline_linkage_preserves_identity_rank_source_and_snapshots(baseline):
    _, candidates, report, digest = baseline
    provenance, snapshots = verify_prediction_baseline(candidates, report, digest)
    assert provenance["status"] == "verified_report_linkage"
    assert provenance["methods"] == ["mageck_native_rank"]
    assert provenance["positive_mappings"] == {"positive_tail_means": "resistance"}
    assert not provenance["candidate_is_report_subset"]
    assert provenance["query_methods"][0]["screen_id"] == "001"
    assert len(snapshots) == 2
    for path, content in snapshots.items():
        assert path.read_bytes() == content


def test_candidate_subset_is_recorded_without_reranking(baseline):
    _, candidates, report, digest = baseline
    provenance, _ = verify_prediction_baseline(candidates.iloc[[1]], report, digest)
    assert provenance["candidate_is_report_subset"]
    assert provenance["candidate_row_count"] == 1
    assert provenance["directional_report_row_count"] == 4
    assert provenance["query_methods"] == [
        {
            "screen_id": "001",
            "contrast_id": "001",
            "phenotype_direction": "resistance",
            "screen_signal_rank_source": "mageck_native_rank",
        }
    ]
    candidates.loc[1, "screen_signal_rank"] = 1
    with pytest.raises(ValueError, match="differs from"):
        verify_prediction_baseline(candidates.iloc[[1]], report, digest)


@pytest.mark.parametrize(
    "column", ["screen_id", "contrast_id", "gene_symbol", "phenotype_direction"]
)
def test_baseline_linkage_uses_complete_candidate_identity(baseline, column):
    _, candidates, report, digest = baseline
    candidates = candidates.iloc[[0]].copy()
    candidates[column] = (
        "sensitization" if column == "phenotype_direction" else "different"
    )
    with pytest.raises(ValueError, match="(?:not found|differs from)"):
        verify_prediction_baseline(candidates, report, digest)


def test_swapped_phenotype_tail_mapping_is_rejected_even_with_valid_hash(baseline):
    ranked, candidates, report, _ = baseline
    digest = _write_report(report, ranked, positive_tail_means="sensitization")
    with pytest.raises(ValueError, match="contradicts declared"):
        verify_prediction_baseline(candidates, report, digest)


def test_mapping_must_be_explicit_for_method_actually_used(baseline):
    ranked, candidates, report, _ = baseline
    digest = _write_report(report, ranked, positive_tail_means=None)
    with pytest.raises(ValueError, match="explicit positive_tail_means"):
        verify_prediction_baseline(candidates, report, digest)


def test_count_lfc_mapping_supports_neutral_unranked_report_rows(baseline):
    ranked, _, report, _ = baseline
    ranked["analysis_tail"] = ["count_positive_lfc"] * 2 + ["count_negative_lfc"] * 2
    ranked["screen_signal_rank_source"] = "absolute_median_guide_lfc"
    neutral = ranked.iloc[[0]].copy()
    neutral["gene_symbol"] = "NEUTRAL"
    neutral["phenotype_direction"] = "neutral"
    neutral["analysis_tail"] = "count_neutral"
    neutral["screen_signal_rank_source"] = "neutral_not_ranked"
    neutral["screen_signal_rank"] = float("nan")
    report_frame = pd.concat([ranked, neutral], ignore_index=True)
    digest = _write_report(
        report, report_frame, positive_tail_means=None, positive_lfc_means="resistance"
    )
    provenance, _ = verify_prediction_baseline(ranked, report, digest)
    assert provenance["report_row_count"] == 5
    assert provenance["directional_report_row_count"] == 4
    assert provenance["methods"] == ["absolute_median_guide_lfc"]
    assert provenance["positive_mappings"] == {"positive_lfc_means": "resistance"}


def test_baseline_rejects_duplicate_report_or_candidate_keys(baseline):
    ranked, candidates, report, digest = baseline
    with pytest.raises(ValueError, match="duplicate candidate join keys"):
        verify_prediction_baseline(
            pd.concat([candidates, candidates.iloc[[0]]]), report, digest
        )
    digest = _write_report(report, pd.concat([ranked, ranked.iloc[[0]]]))
    with pytest.raises(ValueError, match="duplicate candidate join keys"):
        verify_prediction_baseline(candidates, report, digest)


def test_baseline_rejects_mixed_methods_in_one_query(baseline):
    ranked, candidates, report, _ = baseline
    ranked.loc[0, "screen_signal_rank_source"] = "derived_from_mageck_fdr"
    digest = _write_report(report, ranked)
    with pytest.raises(ValueError, match="mixes rank methods"):
        verify_prediction_baseline(candidates, report, digest)


def test_separate_queries_may_have_different_preserved_rank_sources(baseline):
    ranked, _, report, _ = baseline
    ranked.loc[2:, "screen_signal_rank_source"] = "derived_from_mageck_fdr"
    digest = _write_report(report, ranked)
    provenance, _ = verify_prediction_baseline(ranked, report, digest)
    assert provenance["methods"] == ["derived_from_mageck_fdr", "mageck_native_rank"]


@pytest.mark.parametrize("bad", [0, 1.5, float("inf"), True])
def test_candidate_rank_requires_finite_positive_integer(baseline, bad):
    _, candidates, report, digest = baseline
    candidates["screen_signal_rank"] = bad
    with pytest.raises(ValueError, match="ranks"):
        verify_prediction_baseline(candidates, report, digest)


def test_baseline_rejects_unpinned_or_changed_output(baseline):
    _, candidates, report, digest = baseline
    with pytest.raises(ValueError, match="manifest identity hash mismatch"):
        verify_prediction_baseline(candidates, report, "f" * 64)
    (report / "ranked_candidates.tsv").write_text("changed\n")
    with pytest.raises(ValueError, match="ranked output hash mismatch"):
        verify_prediction_baseline(candidates, report, digest)


def test_fixed_ranked_output_filename_prevents_path_redirection(baseline):
    _, candidates, report, _ = baseline
    manifest = json.loads((report / "run_manifest.json").read_text())
    manifest["outputs"]["ranked_candidates"]["filename"] = "../other.tsv"
    digest = _write_manifest(report, manifest)
    with pytest.raises(ValueError, match="fixed filename"):
        verify_prediction_baseline(candidates, report, digest)


def test_rank_source_cannot_be_relabelled_by_candidate_table(baseline):
    _, candidates, report, digest = baseline
    candidates["screen_signal_rank_source"] = "derived_from_mageck_fdr"
    with pytest.raises(ValueError, match="rank source differs"):
        verify_prediction_baseline(candidates, report, digest)
