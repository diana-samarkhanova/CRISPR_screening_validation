import numpy as np
import pandas as pd
import pytest

from crispr_evidencerank.evaluation import (
    TOP_K_TIE_POLICY,
    grouped_ranking_metrics,
    paired_cluster_bootstrap_difference,
    precision_at_k,
    reliability_table,
    study_cluster_bootstrap_intervals,
    within_query_feature_invariance,
)


def sparse_bootstrap_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "study_id": ["S1", "S1", "S2", "S2", "S3", "S3"],
            "screen_id": ["SC1", "SC1", "SC2", "SC2", "SC3", "SC3"],
            "contrast_id": ["C1", "C1", "C2", "C2", "C3", "C3"],
            "phenotype_direction": ["resistance"] * 6,
            "target": [0, 0, 1, 1, 0, 1],
            "score": [0.1, 0.2, 0.8, 0.9, 0.3, 0.7],
        }
    )


def test_bootstrap_reports_metric_specific_effective_draws():
    intervals = study_cluster_bootstrap_intervals(
        sparse_bootstrap_frame(),
        study_column="study_id",
        screen_column="screen_id",
        target_column="target",
        score_column="score",
        n_bootstrap=100,
        min_effective_fraction=0.9,
        random_state=17,
    )
    macro = intervals["query_macro_average_precision"]
    assert macro["effective_draws"] < 90
    assert macro["interval_available"] == 0
    assert "median" not in macro
    assert intervals["_diagnostics"]["minimum_effective_draws"] == 90


def test_bootstrap_rejects_globally_single_class_target():
    frame = sparse_bootstrap_frame()
    frame["target"] = 0
    with pytest.raises(ValueError, match="binary target classes"):
        study_cluster_bootstrap_intervals(
            frame,
            study_column="study_id",
            screen_column="screen_id",
            target_column="target",
            score_column="score",
            n_bootstrap=10,
        )


def test_top_k_metrics_have_resolving_power_and_report_saturation():
    truth = np.array([1, 0, 1, 0, 0, 0], dtype=float)
    perfect = np.array([0.9, 0.2, 0.8, 0.3, 0.1, 0.0])
    middle = np.array([0.9, 0.8, 0.4, 0.3, 0.2, 0.1])
    inverted = np.array([0.0, 0.8, 0.1, 0.7, 0.9, 0.6])

    assert precision_at_k(truth, perfect, 2) > precision_at_k(truth, middle, 2)
    assert precision_at_k(truth, middle, 2) > precision_at_k(truth, inverted, 2)
    assert np.isnan(precision_at_k(truth, perfect, len(truth)))

    frame = pd.DataFrame(
        {
            "study_id": ["S1"] * len(truth),
            "query": ["Q1"] * len(truth),
            "target": truth,
            "score": perfect,
        }
    )
    metrics = grouped_ranking_metrics(
        frame,
        group_column="query",
        target_column="target",
        score_column="score",
        study_column="study_id",
        ks=(2, 6),
    )
    assert metrics["adjudicated_precision_at_2"] == 1
    assert np.isnan(metrics["adjudicated_precision_at_6"])
    assert metrics["n_groups_adjudicated_precision_at_6_defined"] == 0
    assert metrics["adjudicated_ndcg_at_6"] == pytest.approx(1)
    assert metrics["n_groups_adjudicated_ndcg_at_6_unsaturated"] == 0


def test_top_k_cutoff_ties_use_fractional_permutation_invariant_policy():
    truth = np.array([1, 1, 0, 0], dtype=float)
    score = np.array([0.9, 0.8, 0.8, 0.1], dtype=float)
    permutation = np.array([0, 2, 1, 3])

    assert TOP_K_TIE_POLICY == "fractional_expected_relevance_at_cutoff"
    assert precision_at_k(truth, score, 2) == pytest.approx(0.75)
    assert precision_at_k(truth[permutation], score[permutation], 2) == pytest.approx(
        0.75
    )

    frame = pd.DataFrame(
        {
            "study_id": ["S1"] * 4,
            "query": ["Q1"] * 4,
            "target": truth,
            "score": score,
        }
    )
    original = grouped_ranking_metrics(
        frame,
        group_column="query",
        target_column="target",
        score_column="score",
        study_column="study_id",
        ks=(2,),
    )
    permuted = grouped_ranking_metrics(
        frame.iloc[permutation].reset_index(drop=True),
        group_column="query",
        target_column="target",
        score_column="score",
        study_column="study_id",
        ks=(2,),
    )
    for name in (
        "adjudicated_precision_at_2",
        "documented_success_capture_at_2",
        "documented_success_recall_at_2",
        "documented_success_ndcg_at_2",
    ):
        assert original[name] == pytest.approx(permuted[name])
    assert original["documented_success_capture_at_2"] == pytest.approx(0.75)
    assert original["documented_success_recall_at_2"] == pytest.approx(0.75)


def test_single_class_queries_are_counted_but_excluded_from_ranking_macros():
    frame = pd.DataFrame(
        {
            "study_id": ["mixed"] * 3 + ["positive", "negative"],
            "query": ["Q-mixed"] * 3 + ["Q-positive", "Q-negative"],
            "target": [1, 0, 0, 1, 0],
            "score": [0.9, 0.2, 0.1, 0.01, 0.99],
        }
    )
    metrics = grouped_ranking_metrics(
        frame,
        group_column="query",
        target_column="target",
        score_column="score",
        study_column="study_id",
        ks=(1,),
    )

    assert metrics["study_macro_average_precision"] == pytest.approx(1)
    assert metrics["study_macro_adjudicated_precision_at_1"] == pytest.approx(1)
    assert metrics["study_macro_adjudicated_ndcg_at_1"] == pytest.approx(1)
    assert metrics["n_groups_with_adjudicated_labels"] == 3
    assert metrics["n_groups_with_both_classes"] == 1
    assert metrics["n_groups_adjudicated_single_class"] == 2
    assert metrics["n_groups_adjudicated_all_positive"] == 1
    assert metrics["n_groups_adjudicated_all_negative"] == 1
    assert metrics["fraction_groups_with_both_classes_of_adjudicated"] == pytest.approx(
        1 / 3
    )
    assert metrics["n_groups_adjudicated_precision_at_1_defined"] == 1
    assert metrics["n_groups_adjudicated_ndcg_at_1_defined"] == 1
    assert metrics["n_groups_documented_success_with_both_classes"] == 1
    assert metrics["n_groups_documented_success_single_class"] == 2


def test_metrics_report_prevalence_normalized_ap_and_honest_documented_names():
    frame = pd.DataFrame(
        {
            "study_id": ["S1"] * 6,
            "query": ["Q1"] * 6,
            "target": [1, 0, 1, 0, np.nan, np.nan],
            "score": [0.9, 0.2, 0.8, 0.1, 0.4, 0.3],
        }
    )
    metrics = grouped_ranking_metrics(
        frame,
        group_column="query",
        target_column="target",
        score_column="score",
        study_column="study_id",
        ks=(2,),
    )
    assert metrics["positive_prevalence"] == 0.5
    assert metrics["normalized_average_precision"] == 1
    assert metrics["documented_success_capture_at_2"] == 1
    assert metrics["documented_success_lift_at_2"] == 3
    assert "observed_success_yield_at_2" not in metrics
    assert np.isfinite(metrics["calibration_slope"])
    assert np.isfinite(metrics["calibration_intercept"])


def test_paired_cluster_bootstrap_uses_shared_draws_for_score_difference():
    frame = pd.DataFrame(
        {
            "family": ["F1"] * 6 + ["F2"] * 6 + ["F3"] * 6,
            "screen_id": ["S1"] * 6 + ["S2"] * 6 + ["S3"] * 6,
            "target": [1, 0, 1, 0, 0, 0] * 3,
            "candidate": [0.9, 0.2, 0.8, 0.3, 0.1, 0.0] * 3,
            "comparator": [0.0, 0.8, 0.1, 0.7, 0.9, 0.6] * 3,
        }
    )
    interval = paired_cluster_bootstrap_difference(
        frame,
        cluster_column="family",
        screen_column="screen_id",
        target_column="target",
        candidate_score_column="candidate",
        comparator_score_column="comparator",
        metric_name="study_macro_normalized_average_precision",
        n_bootstrap=20,
        min_effective_fraction=0.8,
        random_state=17,
    )
    assert interval["point_difference"] > 0
    assert interval["interval_available"] == 1
    assert interval["lower"] > 0
    assert interval["uncertainty_boundary"].startswith("conditions_on")
    assert interval["candidate_score_coverage"] == 1
    assert interval["comparator_score_coverage"] == 1
    assert interval["common_score_coverage"] == 1
    assert interval["common_score_rows"] == len(frame)


def test_paired_bootstrap_keeps_resampling_cluster_and_study_macro_separate():
    frame = pd.DataFrame(
        {
            "family": ["F1"] * 8 + ["F2"] * 4,
            "study_id": ["S1"] * 4 + ["S2"] * 4 + ["S3"] * 4,
            "screen_id": ["Q1"] * 4 + ["Q2"] * 4 + ["Q3"] * 4,
            "target": [1, 0, 1, 0] * 3,
            "candidate": [0.9, 0.2, 0.8, 0.1] * 3,
            "comparator": [0.1, 0.8, 0.2, 0.9] * 3,
        }
    )
    expected_candidate = grouped_ranking_metrics(
        frame,
        group_column="screen_id",
        target_column="target",
        score_column="candidate",
        study_column="study_id",
    )["study_macro_normalized_average_precision"]
    interval = paired_cluster_bootstrap_difference(
        frame,
        cluster_column="family",
        metric_study_column="study_id",
        screen_column="screen_id",
        target_column="target",
        candidate_score_column="candidate",
        comparator_score_column="comparator",
        metric_name="study_macro_normalized_average_precision",
        n_bootstrap=10,
        min_effective_fraction=0.5,
    )
    assert interval["candidate_point"] == pytest.approx(expected_candidate)


def test_within_query_feature_invariance_distinguishes_main_effects():
    frame = pd.DataFrame(
        {
            "screen_id": ["S1", "S1", "S2", "S2"],
            "contrast_id": ["C1", "C1", "C2", "C2"],
            "design": [2, 2, 3, 3],
            "gene_signal": [0.1, 0.9, 0.2, 0.8],
        }
    )
    report = within_query_feature_invariance(
        frame,
        feature_columns=["design", "gene_signal"],
        query_columns=["screen_id", "contrast_id"],
    )
    assert report["query_count"] == 2
    assert report["invariant_in_all_queries"] == ["design"]
    assert report["varying_in_at_least_one_query"] == ["gene_signal"]
    assert report["features"]["design"]["invariant_query_fraction"] == 1


def test_reliability_table_uses_only_adjudicated_rows_and_exact_denominators():
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, np.nan],
            "score": [0.1, 0.9, 0.2, 0.8, 0.99],
        }
    )
    bins = reliability_table(
        frame,
        target_column="target",
        score_column="score",
        n_bins=2,
    )
    assert len(bins) == 2
    assert sum(record["n"] for record in bins) == 4
    assert bins[0]["event_fraction"] == 0
    assert bins[1]["event_fraction"] == 1


def test_reliability_bins_preserve_ties_and_are_permutation_invariant():
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "score": [0.5] * 6,
        }
    )
    original = reliability_table(
        frame,
        target_column="target",
        score_column="score",
        n_bins=5,
    )
    permuted = reliability_table(
        frame.sample(frac=1, random_state=23).reset_index(drop=True),
        target_column="target",
        score_column="score",
        n_bins=5,
    )

    assert original == permuted
    assert original == [
        {
            "bin": 0.0,
            "n": 6.0,
            "mean_score": 0.5,
            "event_fraction": 0.5,
            "score_minimum": 0.5,
            "score_maximum": 0.5,
        }
    ]


@pytest.mark.parametrize(
    ("score_column", "bad_value"),
    [("candidate", np.nan), ("comparator", np.inf)],
)
def test_paired_bootstrap_fails_closed_on_nonfinite_common_roster(
    score_column,
    bad_value,
):
    frame = pd.DataFrame(
        {
            "family": ["F1"] * 4 + ["F2"] * 4,
            "study_id": ["S1"] * 4 + ["S2"] * 4,
            "screen_id": ["Q1"] * 4 + ["Q2"] * 4,
            "target": [1, 0, 1, 0] * 2,
            "candidate": [0.9, 0.2, 0.8, 0.1] * 2,
            "comparator": [0.8, 0.3, 0.7, 0.2] * 2,
        }
    )
    frame.loc[0, score_column] = bad_value
    with pytest.raises(ValueError, match="complete common roster"):
        paired_cluster_bootstrap_difference(
            frame,
            cluster_column="family",
            metric_study_column="study_id",
            screen_column="screen_id",
            target_column="target",
            candidate_score_column="candidate",
            comparator_score_column="comparator",
            metric_name="study_macro_normalized_average_precision",
            query_columns=("screen_id",),
            n_bootstrap=5,
        )


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [("family", None), ("study_id", "  "), ("screen_id", np.nan)],
)
def test_bootstraps_fail_closed_on_missing_cluster_study_or_query_ids(
    column,
    bad_value,
):
    frame = pd.DataFrame(
        {
            "family": ["F1"] * 4 + ["F2"] * 4,
            "study_id": ["S1"] * 4 + ["S2"] * 4,
            "screen_id": ["Q1"] * 4 + ["Q2"] * 4,
            "target": [1, 0, 1, 0] * 2,
            "candidate": [0.9, 0.2, 0.8, 0.1] * 2,
            "comparator": [0.8, 0.3, 0.7, 0.2] * 2,
        }
    )
    frame.loc[0, column] = bad_value

    with pytest.raises(ValueError, match="identifiers contain missing or blank"):
        study_cluster_bootstrap_intervals(
            frame,
            study_column="family",
            metric_study_column="study_id",
            screen_column="screen_id",
            target_column="target",
            score_column="candidate",
            query_columns=("screen_id",),
            n_bootstrap=5,
        )
    with pytest.raises(ValueError, match="identifiers contain missing or blank"):
        paired_cluster_bootstrap_difference(
            frame,
            cluster_column="family",
            metric_study_column="study_id",
            screen_column="screen_id",
            target_column="target",
            candidate_score_column="candidate",
            comparator_score_column="comparator",
            metric_name="study_macro_normalized_average_precision",
            query_columns=("screen_id",),
            n_bootstrap=5,
        )
