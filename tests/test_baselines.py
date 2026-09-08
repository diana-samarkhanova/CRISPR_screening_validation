import numpy as np
import pandas as pd
import pytest

import crispr_evidencerank.baselines as baselines_module
from crispr_evidencerank.baselines import (
    adapt_direction_aligned_drugz_baseline,
    adapt_repo_native_mageck_baseline,
    evaluate_fixed_score_baselines,
)


def baseline_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "study_id": ["S1"] * 6 + ["S2"] * 6,
            "screen_id": ["SC1"] * 6 + ["SC2"] * 6,
            "contrast_id": ["C1"] * 6 + ["C2"] * 6,
            "phenotype_direction": ["resistance"] * 12,
            "target": [1, 0, 1, 0, 0, 0] * 2,
            "absolute_median_guide_lfc": [0.9, 0.2, 0.8, 0.3, 0.1, 0.0] * 2,
        }
    )


def test_baseline_runner_evaluates_safe_scores_and_reason_codes_missing_inputs():
    result = evaluate_fixed_score_baselines(
        baseline_frame(),
        screen_column="screen_id",
        target_column="target",
        study_column="study_id",
        random_seeds=(1, 2, 3, 4, 5),
        ks=(2,),
    )
    assert result["random"]["status"] == "evaluated"
    assert result["random"]["random_seed_count"] == 5
    assert result["absolute_effect"]["status"] == "evaluated"
    assert (
        result["absolute_effect"]["metrics"]["study_macro_normalized_average_precision"]
        == 1
    )
    assert result["mageck_rank"]["status"] == "unavailable_missing_column"
    assert result["drugz_normz"]["required_column"] == ("drugz_direction_aligned_normz")
    assert result["cross_screen_recurrence"]["status"] == "not_implemented"


def test_baseline_runner_rejects_partial_score_coverage():
    frame = baseline_frame()
    frame["mageck_direction_rank"] = range(1, len(frame) + 1)
    frame.loc[0, "mageck_direction_rank"] = None
    result = evaluate_fixed_score_baselines(
        frame,
        screen_column="screen_id",
        target_column="target",
        study_column="study_id",
        random_seeds=(1,),
        ks=(2,),
    )
    assert result["mageck_rank"]["status"] == ("unavailable_incomplete_score_coverage")
    assert result["mageck_rank"]["score_coverage"] < 1


@pytest.mark.parametrize(
    ("column", "invalid_value", "message"),
    [
        ("absolute_median_guide_lfc", -0.1, "non-negative"),
        ("mageck_direction_rank", 0, "positive integers"),
        ("mageck_direction_rank", 1.5, "positive integers"),
        ("mageck_direction_fdr", -0.01, r"\[0, 1\]"),
        ("mageck_direction_fdr", 1.01, r"\[0, 1\]"),
        ("drugz_direction_aligned_normz", np.inf, "finite"),
        ("drugz_direction_aligned_normz", "not-a-number", "non-numeric"),
    ],
)
def test_fixed_baselines_fail_closed_on_invalid_values(
    column: str,
    invalid_value: object,
    message: str,
):
    frame = baseline_frame()
    if column != "absolute_median_guide_lfc":
        frame[column] = 1.0
    if isinstance(invalid_value, str):
        frame[column] = frame[column].astype(object)
    frame.loc[0, column] = invalid_value

    with pytest.raises(ValueError, match=message):
        evaluate_fixed_score_baselines(
            frame,
            screen_column="screen_id",
            target_column="target",
            study_column="study_id",
            random_seeds=(1,),
            ks=(2,),
        )


def test_mageck_adapter_requires_and_preserves_direction_and_provenance():
    native = pd.DataFrame(
        {
            "gene_symbol": ["A", "B", "A", "B"],
            "analysis_tail": [
                "mageck_pos",
                "mageck_pos",
                "mageck_neg",
                "mageck_neg",
            ],
            "phenotype_direction": [
                "resistance",
                "resistance",
                "sensitization",
                "sensitization",
            ],
            "mageck_rank": [1, 2, 2, 1],
            "mageck_fdr": [0.01, 0.2, 0.2, 0.01],
        }
    )

    adapted = adapt_repo_native_mageck_baseline(
        native,
        positive_tail_means="resistance",
        mageck_version="0.5.9.5",
        source_locator="frozen/input/gene_summary.txt",
        source_sha256="a" * 64,
    )

    assert adapted["mageck_direction_rank"].equals(native["mageck_rank"])
    assert adapted["mageck_direction_fdr"].equals(native["mageck_fdr"])
    assert adapted["mageck_positive_tail_means_attested"].eq("resistance").all()
    assert adapted["mageck_source_method"].eq("MAGeCK-RRA").all()
    assert adapted["mageck_source_version"].eq("0.5.9.5").all()
    assert adapted["mageck_source_sha256"].eq("a" * 64).all()


def test_mageck_adapter_rejects_direction_mismatch_instead_of_guessing():
    native = pd.DataFrame(
        {
            "analysis_tail": ["mageck_pos"],
            "phenotype_direction": ["sensitization"],
            "mageck_rank": [1],
        }
    )
    with pytest.raises(ValueError, match="contradicts the explicit tail attestation"):
        adapt_repo_native_mageck_baseline(
            native,
            positive_tail_means="resistance",
            mageck_version="0.5.9.5",
            source_locator="frozen/input/gene_summary.txt",
            source_sha256="a" * 64,
        )


def test_mageck_adapter_rejects_missing_direction_attestation():
    native = pd.DataFrame(
        {
            "analysis_tail": ["mageck_pos"],
            "phenotype_direction": [None],
            "mageck_rank": [1],
        }
    )
    with pytest.raises(ValueError, match="contradicts the explicit tail attestation"):
        adapt_repo_native_mageck_baseline(
            native,
            positive_tail_means="resistance",
            mageck_version="0.5.9.5",
            source_locator="frozen/input/gene_summary.txt",
            source_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"positive_tail_means": "unknown"}, "explicitly attest"),
        ({"mageck_version": ""}, "mageck_version"),
        ({"source_locator": "  "}, "source_locator"),
        ({"source_sha256": "A" * 64}, "lowercase SHA-256"),
        ({"source_sha256": "a" * 63}, "lowercase SHA-256"),
    ],
)
def test_mageck_adapter_requires_explicit_attestation_metadata(
    overrides: dict[str, str],
    message: str,
):
    arguments = {
        "positive_tail_means": "resistance",
        "mageck_version": "0.5.9.5",
        "source_locator": "frozen/input/gene_summary.txt",
        "source_sha256": "a" * 64,
        **overrides,
    }
    native = pd.DataFrame(
        {
            "analysis_tail": ["mageck_pos"],
            "phenotype_direction": ["resistance"],
            "mageck_rank": [1],
        }
    )
    with pytest.raises(ValueError, match=message):
        adapt_repo_native_mageck_baseline(native, **arguments)


def test_mageck_and_drugz_baselines_use_declared_positive_orientation():
    frame = baseline_frame()
    per_query_rank = [1, 4, 2, 3, 5, 6]
    per_query_fdr = [0.01, 0.4, 0.02, 0.3, 0.5, 0.6]
    per_query_normz = [4.0, 0.0, 3.0, 1.0, -1.0, -2.0]
    frame["analysis_tail"] = "mageck_pos"
    frame["mageck_rank"] = per_query_rank * 2
    frame["mageck_fdr"] = per_query_fdr * 2
    frame["drugz_normz"] = per_query_normz * 2
    frame = adapt_repo_native_mageck_baseline(
        frame,
        positive_tail_means="resistance",
        mageck_version="0.5.9.5",
        source_locator="frozen/input/gene_summary.txt",
        source_sha256="a" * 64,
    )
    frame = adapt_direction_aligned_drugz_baseline(
        frame,
        positive_normz_means="resistance",
        drugz_version="1.1.0",
        source_locator="frozen/input/drugz.tsv",
        source_sha256="b" * 64,
    )

    result = evaluate_fixed_score_baselines(
        frame,
        screen_column="screen_id",
        target_column="target",
        study_column="study_id",
        random_seeds=(1,),
        ks=(2,),
    )

    for name in ("mageck_rank", "mageck_fdr", "drugz_normz"):
        assert result[name]["status"] == "evaluated"
        assert result[name]["metrics"]["study_macro_normalized_average_precision"] == 1


def test_direction_specific_baseline_columns_cannot_bypass_attestation():
    frame = baseline_frame()
    frame["mageck_direction_rank"] = list(range(1, 7)) * 2
    with pytest.raises(ValueError, match="direction/provenance attestation"):
        evaluate_fixed_score_baselines(
            frame,
            screen_column="screen_id",
            target_column="target",
            study_column="study_id",
            random_seeds=(1,),
            ks=(2,),
        )


def test_drugz_adapter_reverses_native_sign_for_opposite_candidate_direction():
    frame = pd.DataFrame(
        {
            "phenotype_direction": ["resistance", "sensitization"],
            "drugz_normz": [2.0, 2.0],
        }
    )
    adapted = adapt_direction_aligned_drugz_baseline(
        frame,
        positive_normz_means="resistance",
        drugz_version="1.1.0",
        source_locator="frozen/input/drugz.tsv",
        source_sha256="b" * 64,
    )
    assert adapted["drugz_direction_aligned_normz"].tolist() == [2.0, -2.0]


def test_paired_bootstrap_minimum_effective_fraction_is_forwarded(monkeypatch):
    captured: dict[str, object] = {}

    def fake_paired_bootstrap(frame, **kwargs):
        del frame
        captured.update(kwargs)
        return {"interval_available": 0.0}

    monkeypatch.setattr(
        baselines_module,
        "paired_cluster_bootstrap_difference",
        fake_paired_bootstrap,
    )
    frame = baseline_frame()
    frame["candidate_score"] = frame["absolute_median_guide_lfc"]
    frame["cluster_id"] = frame["study_id"]

    evaluate_fixed_score_baselines(
        frame,
        screen_column="screen_id",
        target_column="target",
        study_column="study_id",
        random_seeds=(1,),
        ks=(2,),
        candidate_score_column="candidate_score",
        cluster_column="cluster_id",
        paired_bootstrap_repetitions=10,
        paired_bootstrap_min_effective_fraction=0.9,
    )

    assert captured["min_effective_fraction"] == 0.9


def _attested_multi_source_frame():
    frame = baseline_frame()
    sources = []
    for index, (_, query) in enumerate(frame.groupby("screen_id")):
        query = query.copy()
        query["analysis_tail"] = "mageck_pos"
        query["mageck_rank"] = [1, 4, 2, 3, 5, 6]
        query["drugz_normz"] = [4.0, 0.0, 3.0, 1.0, -1.0, -2.0]
        if index:
            query["analysis_tail"] = "mageck_neg"
            query["drugz_normz"] *= -1
        meaning = "sensitization" if index else "resistance"
        query = adapt_repo_native_mageck_baseline(
            query,
            positive_tail_means=meaning,
            mageck_version=f"0.5.9.{index + 4}",
            source_locator=f"frozen/study{index}/mageck.tsv",
            source_sha256=str(index) * 64,
        )
        sources.append(
            adapt_direction_aligned_drugz_baseline(
                query,
                positive_normz_means=meaning,
                drugz_version="1.1.0",
                source_locator=f"frozen/study{index}/drugz.tsv",
                source_sha256=str(index + 2) * 64,
            )
        )
    return pd.concat(sources)


def test_benchmark_allows_independent_sources_and_tail_semantics_across_queries():
    frame = _attested_multi_source_frame()
    result = evaluate_fixed_score_baselines(
        frame,
        screen_column="screen_id",
        target_column="target",
        study_column="study_id",
        random_seeds=(1,),
        ks=(2,),
    )
    for name in ("mageck_rank", "drugz_normz"):
        assert result[name]["status"] == "evaluated"
        assert result[name]["metrics"]["study_macro_normalized_average_precision"] == 1


@pytest.mark.parametrize("column", ["mageck_source_sha256", "drugz_source_locator"])
def test_benchmark_still_rejects_mixed_provenance_within_one_query(column):
    frame = _attested_multi_source_frame()
    frame.loc[0, column] = "f" * 64
    with pytest.raises(ValueError, match="one complete value"):
        evaluate_fixed_score_baselines(
            frame,
            screen_column="screen_id",
            target_column="target",
            study_column="study_id",
            random_seeds=(1,),
            ks=(2,),
        )


@pytest.mark.parametrize("seeds", [(1, 1), (True,), (-1,), (1.5,)])
def test_random_baseline_requires_distinct_nonnegative_integer_seeds(seeds):
    with pytest.raises(ValueError, match="random_seeds"):
        evaluate_fixed_score_baselines(
            baseline_frame(),
            screen_column="screen_id",
            target_column="target",
            study_column="study_id",
            random_seeds=seeds,
        )


@pytest.mark.parametrize(
    "column", ["_baseline_query_id", "_fixed_baseline_score", "_random_score_1"]
)
def test_baselines_cannot_overwrite_caller_data_with_temporary_columns(column):
    frame = baseline_frame()
    frame[column] = frame.target
    with pytest.raises(ValueError, match="reserved internal columns"):
        evaluate_fixed_score_baselines(
            frame,
            screen_column="screen_id",
            target_column="target",
            study_column="study_id",
            random_seeds=(1,),
        )
    assert frame[column].equals(frame.target)


def test_mageck_adapter_rejects_invalid_native_score_before_attestation():
    frame = baseline_frame()
    frame["analysis_tail"] = "mageck_pos"
    frame["mageck_fdr"] = 2.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        adapt_repo_native_mageck_baseline(
            frame,
            positive_tail_means="resistance",
            mageck_version="0.5.9.5",
            source_locator="frozen/mageck.tsv",
            source_sha256="a" * 64,
        )


def test_drugz_adapter_does_not_convert_boolean_metadata_into_scores():
    frame = baseline_frame()
    frame["drugz_normz"] = True
    with pytest.raises(ValueError, match="not boolean"):
        adapt_direction_aligned_drugz_baseline(
            frame,
            positive_normz_means="resistance",
            drugz_version="1.1.0",
            source_locator="frozen/drugz.tsv",
            source_sha256="a" * 64,
        )
