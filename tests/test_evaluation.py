import pandas as pd
import pytest

from crispr_evidencerank.evaluation import (
    study_cluster_bootstrap_intervals,
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
