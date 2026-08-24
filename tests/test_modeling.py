from pathlib import Path

import pandas as pd
import pytest

from crispr_evidencerank.evaluation import _composite_query_keys
from crispr_evidencerank.modeling import (
    ReproducibilityModel,
    WeightedMedianStandardizer,
    _cross_fitted_propensity,
    _study_family_groups,
    grouped_oof_predictions,
    validate_success_feature_columns,
)

ROOT = Path(__file__).resolve().parents[1]


def test_grouped_oof_predictions_cover_every_row():
    path = ROOT / "examples" / "synthetic" / "labeled_gene_features.csv"
    frame = pd.read_csv(path)
    predictions, metrics = grouped_oof_predictions(frame, n_splits=4)
    assert predictions["reproducibility_score"].notna().all()
    assert predictions["fold"].nunique() == 4
    assert metrics["n_groups"] == 16
    assert 0 <= metrics["average_precision"] <= 1
    assert 0 <= metrics["macro_average_precision"] <= 1


def test_two_study_benchmark_disables_non_cross_fitted_ipw():
    path = ROOT / "examples" / "synthetic" / "labeled_gene_features.csv"
    frame = pd.read_csv(path)
    keep = (
        frame[["study_id", "raw_data_family_id"]]
        .drop_duplicates()
        .drop_duplicates("raw_data_family_id")["study_id"]
        .iloc[:2]
    )
    _, metrics = grouped_oof_predictions(
        frame.loc[frame["study_id"].isin(keep)],
        n_splits=2,
    )
    assert metrics["ipw_available"] == 0


def test_weighted_preprocessor_equalizes_study_mass():
    values = pd.DataFrame({"feature": [1.0] * 100 + [100.0]})
    weights = pd.Series([0.01] * 100 + [1.0])
    preprocessor = WeightedMedianStandardizer().fit(
        values,
        sample_weight=weights,
    )
    assert abs(preprocessor.mean_[0] - 50.5) < 1e-9


def test_cross_fitted_propensity_rejects_single_class_inner_fold():
    features = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    target = pd.Series([0.0, 0.0, 1.0, 1.0])
    groups = pd.Series(["S1", "S1", "S2", "S2"])
    with pytest.raises(
        ValueError,
        match="both testing-status classes in every independent inner",
    ):
        _cross_fitted_propensity(features, target, groups)


def test_v1_model_rejects_unknown_or_discordant_candidate_direction():
    path = ROOT / "examples" / "synthetic" / "labeled_gene_features.csv"
    frame = pd.read_csv(path)
    frame.loc[frame.index[0], "phenotype_direction"] = "unknown"
    with pytest.raises(ValueError, match="accepts only resistance"):
        grouped_oof_predictions(frame, n_splits=4)


def test_composite_query_keys_are_collision_safe():
    frame = pd.DataFrame(
        {
            "screen_id": ["A||B", "A"],
            "contrast_id": ["C", "B||C"],
            "phenotype_direction": ["resistance", "resistance"],
        }
    )
    keys = _composite_query_keys(
        frame,
        ["screen_id", "contrast_id", "phenotype_direction"],
    )
    assert keys.nunique() == 2


def test_missing_indicator_is_stable_for_transform_only_missingness():
    train = pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [3.0, 4.0]})
    preprocessor = WeightedMedianStandardizer().fit(train)
    transformed = preprocessor.transform(
        pd.DataFrame({"feature_a": [float("nan")], "feature_b": [3.5]})
    )
    assert transformed.shape == (1, 4)
    assert transformed[0, 2] == 1.0
    assert transformed[0, 3] == 0.0


@pytest.mark.parametrize(
    "forbidden",
    ["author_hit", "phenotype_reproduced", "validation_outcome"],
)
def test_success_model_rejects_validation_leakage_features(forbidden):
    with pytest.raises(ValueError, match="leakage fields"):
        validate_success_feature_columns(["guide_n", forbidden])


def test_direct_model_fit_rejects_report_only_features() -> None:
    model = ReproducibilityModel(["report_only_immuno_dual_action_class"])
    with pytest.raises(ValueError, match="leakage fields"):
        model.fit(pd.DataFrame())


def test_shared_raw_data_family_stays_in_one_oof_fold():
    path = ROOT / "examples" / "synthetic" / "labeled_gene_features.csv"
    frame = pd.read_csv(path)
    shared_studies = frame["study_id"].drop_duplicates().iloc[:2]
    frame["raw_data_family_id"] = pd.NA
    frame.loc[
        frame["study_id"].isin(shared_studies),
        "raw_data_family_id",
    ] = "SHARED_RAW_DATA"

    predictions, _ = grouped_oof_predictions(frame, n_splits=4)
    shared_rows = predictions["study_id"].isin(shared_studies)
    assert predictions.loc[shared_rows, "fold"].nunique() == 1


def test_shared_source_family_stays_in_one_oof_fold():
    path = ROOT / "examples" / "synthetic" / "labeled_gene_features.csv"
    frame = pd.read_csv(path)
    shared_studies = frame["study_id"].drop_duplicates().iloc[:2]
    frame["source_family_id"] = pd.NA
    frame["raw_data_family_id"] = pd.NA
    frame.loc[
        frame["study_id"].isin(shared_studies),
        "source_family_id",
    ] = "SHARED_SOURCE"

    predictions, _ = grouped_oof_predictions(frame, n_splits=4)
    shared_rows = predictions["study_id"].isin(shared_studies)
    assert predictions.loc[shared_rows, "fold"].nunique() == 1


def test_source_and_raw_family_links_are_transitive():
    frame = pd.DataFrame(
        {
            "study_id": ["S1", "S2", "S2", "S3"],
            "source_family_id": ["SOURCE", "SOURCE", None, None],
            "raw_data_family_id": [None, None, "RAW", "RAW"],
        }
    )
    groups = _study_family_groups(
        frame,
        study_column="study_id",
        family_columns=("source_family_id", "raw_data_family_id"),
    )
    assert groups.nunique() == 1
