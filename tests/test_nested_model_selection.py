import json

import numpy as np
import pandas as pd
import pytest

from crispr_evidencerank.labels import model_target
from crispr_evidencerank.modeling import (
    ReproducibilityModel,
    WeightedMedianStandardizer,
    _prepare_direction_features,
    clone_model,
)
from crispr_evidencerank.nested_model_selection import (
    ModelCandidate,
    nested_grouped_oof_predictions,
    select_final_candidate,
)


@pytest.fixture
def frame():
    """Algorithm fixture only: fabricated labels do not represent biology."""
    rows = []
    labels = ["V2", "V3", "F0", "D", "U", "T"]
    for study in range(6):
        for direction in ("resistance", "sensitization"):
            for gene, label in enumerate(labels):
                rows.append(
                    {
                        "study_id": f"study-{study}",
                        "screen_id": f"screen-{study}",
                        "contrast_id": f"contrast-{study}",
                        "source_family_id": f"source-{study}",
                        "raw_data_family_id": f"raw-{study}",
                        "gene_symbol": f"GENE-{gene}",
                        "phenotype_direction": direction,
                        "perturbation_modality": "CRISPR_KO",
                        "label_code": label,
                        "testing_status": "not_tested" if label == "U" else "tested",
                        "guide_n": float(gene + study / 10 + 1),
                        "guide_direction_agreement": (gene + study % 2) / 7,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def candidates():
    return (
        ModelCandidate("regularized", ("guide_n",), logistic_c=0.1),
        ModelCandidate("less_regularized", ("guide_n",), logistic_c=10.0),
    )


def test_nested_predictions_cover_rows_and_preserve_unjudged_labels(frame, candidates):
    result = nested_grouped_oof_predictions(
        frame, candidates=candidates, outer_splits=3, inner_splits=2
    )
    out = result.predictions
    assert out["reproducibility_score"].between(0, 1).all()
    assert out["fold"].nunique() == 3
    pd.testing.assert_series_equal(out["label_code"], frame["label_code"])
    assert out.loc[out["label_code"].isin(["U", "T"]), "target"].isna().all()
    assert result.metrics["n_labeled"] == 48
    assert result.metrics["n_groups"] == 12
    assert result.metrics["direction_sensitization_n_adjudicated"] == 24
    assert result.metrics["direction_resistance_ranking_available"] == 1
    assert result.diagnostics["calibration_applied"] is False
    # Includes estimator hyperparameters and undefined metric values safely.
    json.dumps(result.diagnostics, allow_nan=False)


def test_transitive_source_raw_families_stay_together_at_both_levels(frame, candidates):
    frame.loc[frame.study_id.isin(["study-0", "study-1"]), "source_family_id"] = (
        "shared"
    )
    frame.loc[frame.study_id.isin(["study-1", "study-2"]), "raw_data_family_id"] = (
        "shared"
    )
    result = nested_grouped_oof_predictions(
        frame, candidates=candidates, outer_splits=3, inner_splits=2
    )
    connected = {"study-0", "study-1", "study-2"}
    out = result.predictions
    assert out.loc[out.study_id.isin(connected), "fold"].nunique() == 1
    assert result.metrics["n_independent_families"] == 4
    for outer in result.diagnostics["folds"]:
        assert not set(outer["train_families"]) & set(outer["test_families"])
        for inner in outer["inner_folds"]:
            assert not set(inner["train_families"]) & set(inner["test_families"])
            assert not set(inner["train_families"]) & set(outer["test_families"])
            train_connected = connected & set(inner["train_studies"])
            test_connected = connected & set(inner["test_studies"])
            assert not (train_connected and test_connected)


def test_outer_outcome_changes_cannot_select_candidate_or_affect_predictions(
    frame, candidates
):
    initial = nested_grouped_oof_predictions(
        frame, candidates=candidates, outer_splits=3, inner_splits=2
    )
    held_out = initial.predictions["fold"].eq(1)
    changed = frame.copy()
    changed.loc[held_out, "label_code"] = changed.loc[held_out, "label_code"].replace(
        {"V2": "F0", "V3": "D", "F0": "V2", "D": "V3"}
    )
    repeated = nested_grouped_oof_predictions(
        changed, candidates=candidates, outer_splits=3, inner_splits=2
    )
    assert initial.diagnostics["folds"][0] == repeated.diagnostics["folds"][0]
    pd.testing.assert_series_equal(
        initial.predictions.loc[held_out, "reproducibility_score"],
        repeated.predictions.loc[held_out, "reproducibility_score"],
    )


def test_outer_feature_changes_cannot_influence_training_imputation_or_selection(
    frame, candidates
):
    initial = nested_grouped_oof_predictions(
        frame, candidates=candidates, outer_splits=3, inner_splits=2
    )
    held_out = initial.predictions["fold"].eq(1)
    changed = frame.copy()
    changed.loc[held_out, "guide_n"] = 1e9
    repeated = nested_grouped_oof_predictions(
        changed, candidates=candidates, outer_splits=3, inner_splits=2
    )
    assert initial.diagnostics["folds"][0] == repeated.diagnostics["folds"][0]


def test_each_inner_fit_imputes_only_its_training_labeled_rows(frame, candidates):
    result = nested_grouped_oof_predictions(
        frame, candidates=candidates, outer_splits=3, inner_splits=2
    )
    for outer in result.diagnostics["folds"]:
        for candidate in outer["candidates"]:
            for fit, partition in zip(
                candidate["fits"], outer["inner_folds"], strict=True
            ):
                training = frame.loc[frame.study_id.isin(partition["train_studies"])]
                labeled = training.label_code.map(model_target).notna()
                # This fixture gives every independent family equal labeled N.
                expected = WeightedMedianStandardizer().fit(
                    training.loc[labeled, ["guide_n"]]
                )
                assert fit["training_medians"] == expected.medians_.tolist()


def test_exact_tie_uses_declared_candidate_order(frame):
    first = ModelCandidate("first", ("guide_n",))
    second = ModelCandidate("second", ("guide_n",))
    result = nested_grouped_oof_predictions(
        frame, candidates=[first, second], outer_splits=3, inner_splits=2
    )
    assert set(result.predictions.selected_candidate_id) == {"first"}


def test_nested_selection_rejects_single_class_inner_train_before_fitting(
    frame, candidates, monkeypatch
):
    # Positive outcomes exist only in one family. Outer training can contain
    # both classes while an inner training fold still cannot fit a classifier.
    frame.loc[
        ~frame.study_id.eq("study-0") & frame.label_code.isin(["V2", "V3"]),
        "label_code",
    ] = "F0"
    calls = []
    monkeypatch.setattr(
        ReproducibilityModel, "fit", lambda *args, **kwargs: calls.append(1)
    )
    with pytest.raises(
        ValueError, match="both explicit positive and negative training"
    ):
        nested_grouped_oof_predictions(
            frame, candidates=candidates, outer_splits=3, inner_splits=2
        )
    assert calls == []


def test_single_class_queries_are_not_a_valid_selection_objective(frame, candidates):
    # All families have both outcomes overall, but each directional query has
    # one class: global AP must not replace unidentifiable within-query ranking.
    judged = frame.label_code.isin(["V2", "V3", "F0", "D"])
    frame.loc[judged & frame.phenotype_direction.eq("resistance"), "label_code"] = "V2"
    frame.loc[judged & frame.phenotype_direction.eq("sensitization"), "label_code"] = (
        "F0"
    )
    with pytest.raises(ValueError, match="selection metric .* is undefined"):
        nested_grouped_oof_predictions(
            frame, candidates=candidates, outer_splits=3, inner_splits=2
        )


def test_oversized_top_k_fails_without_changing_primary_endpoint(frame, candidates):
    with pytest.raises(ValueError, match="requested cutoff covers every"):
        nested_grouped_oof_predictions(
            frame,
            candidates=candidates,
            outer_splits=3,
            inner_splits=2,
            selection_metric="study_macro_strict_adjudicated_ndcg_at_10",
        )


def test_strict_top_k_selection_available_at_informative_cutoff(frame, candidates):
    result = nested_grouped_oof_predictions(
        frame,
        candidates=candidates,
        outer_splits=3,
        inner_splits=2,
        selection_metric="study_macro_strict_adjudicated_ndcg_at_2",
        ks=(2,),
    )
    assert np.isfinite(result.metrics["study_macro_strict_adjudicated_ndcg_at_2"])


def test_missing_family_mapping_rejected(frame, candidates):
    frame.loc[0, "raw_data_family_id"] = pd.NA
    with pytest.raises(ValueError, match="missing or blank"):
        nested_grouped_oof_predictions(frame, candidates=candidates)


def test_at_least_three_independent_families_required(frame, candidates):
    frame["source_family_id"] = "one-dependent-family"
    with pytest.raises(ValueError, match="at least three independent families"):
        nested_grouped_oof_predictions(frame, candidates=candidates)


@pytest.mark.parametrize("modality", ["CRISPRa", "CRISPRi", pd.NA])
def test_v1_does_not_mix_modalities(frame, candidates, modality):
    frame.loc[0, "perturbation_modality"] = modality
    with pytest.raises(ValueError, match="only CRISPR_KO modality"):
        nested_grouped_oof_predictions(frame, candidates=candidates)


def test_duplicate_dataframe_index_cannot_duplicate_or_misalign_predictions(
    frame, candidates
):
    expected = nested_grouped_oof_predictions(
        frame, candidates=candidates, outer_splits=3, inner_splits=2
    )
    frame.index = pd.Index(["duplicate"] * len(frame))
    result = nested_grouped_oof_predictions(
        frame, candidates=candidates, outer_splits=3, inner_splits=2
    )
    assert result.predictions.index.equals(frame.index)
    np.testing.assert_array_equal(
        expected.predictions.reproducibility_score,
        result.predictions.reproducibility_score,
    )


def test_misaligned_selection_groups_fail_closed(frame):
    model = ReproducibilityModel(["guide_n"])
    with pytest.raises(ValueError, match="must match the frame index"):
        model.fit(frame, selection_groups=frame.study_id.iloc[::-1])


@pytest.mark.parametrize("columns", [(), ("guide_n", "guide_n"), ("author_hit",)])
def test_candidate_rejects_empty_duplicated_or_leaking_features(columns):
    with pytest.raises(ValueError):
        ModelCandidate("bad", columns)


@pytest.mark.parametrize(
    "value", [0, -1, float("nan"), float("inf"), True, "1.0", None]
)
def test_invalid_logistic_regularization_rejected(value):
    with pytest.raises(ValueError, match="logistic_c"):
        ModelCandidate("bad", ("guide_n",), logistic_c=value)


def test_logistic_regularization_is_preserved_and_applied(frame):
    model = ReproducibilityModel(["guide_n"], logistic_c=0.125).fit(frame)
    assert model.success_model.named_steps["model"].C == 0.125
    assert clone_model(model).logistic_c == 0.125
    assert clone_model(model).success_model is None
    assert model.selection_model.named_steps["model"].C == 1.0


def test_preprocessor_rejects_feature_order_drift():
    frame = pd.DataFrame(
        {"guide_n": [1.0, 2.0], "guide_direction_agreement": [0.1, 0.8]}
    )
    transformer = WeightedMedianStandardizer().fit(frame)
    with pytest.raises(ValueError, match="feature names or order differ"):
        transformer.transform(frame[frame.columns[::-1]])


def test_boosting_candidate_uses_fixed_hyperparameters(frame):
    candidate = ModelCandidate("boosting", ("guide_n",), "hist_gradient_boosting")
    result = nested_grouped_oof_predictions(
        frame, candidates=[candidate], outer_splits=3, inner_splits=2
    )
    assert result.predictions.reproducibility_score.notna().all()
    for fold in result.diagnostics["folds"]:
        params = fold["outer_training_fit"]["success_hyperparameters"]
        assert params["max_leaf_nodes"] == 15
        assert params["learning_rate"] == 0.05
        assert params["early_stopping"] is False


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"outer_splits": 1}, "outer_splits"),
        ({"inner_splits": True}, "inner_splits"),
        ({"random_state": -1}, "random_state"),
        ({"ks": (0,)}, "ks"),
        ({"ks": (1, 1)}, "ks"),
        ({"selection_metric": "global_average_precision"}, "within-query"),
        ({"propensity_clip": (0.95, 0.1)}, "propensity_clip"),
        ({"raw_family_column": "source_family_id"}, "must be distinct"),
    ],
)
def test_invalid_nested_configuration_rejected(frame, candidates, kwargs, message):
    with pytest.raises(ValueError, match=message):
        nested_grouped_oof_predictions(frame, candidates=candidates, **kwargs)


def test_duplicate_candidate_identifiers_rejected(frame, candidates):
    with pytest.raises(ValueError, match="candidate_id values must be unique"):
        nested_grouped_oof_predictions(frame, candidates=[candidates[0], candidates[0]])


def test_final_selection_uses_development_inner_scores_and_has_no_outer_estimate(
    frame, candidates
):
    result = select_final_candidate(
        frame,
        candidates=candidates,
        inner_splits=3,
        selection_metric="study_macro_normalized_average_precision",
    )
    scores = [item["selection_value"] for item in result.diagnostics["candidates"]]
    expected = candidates[int(np.argmax(scores))]
    assert result.selected_candidate == expected
    assert result.diagnostics["selected_candidate_id"] == expected.candidate_id
    assert (
        result.diagnostics["selection_score_is_unbiased_performance_estimate"] is False
    )
    assert result.diagnostics["n_development_rows"] == len(frame)
    assert result.diagnostics["actual_inner_splits"] == 3
    json.dumps(result.diagnostics, allow_nan=False)


def test_final_selection_rejects_insufficient_families(frame, candidates):
    frame["source_family_id"] = "all-related"
    with pytest.raises(ValueError, match="at least two independent families"):
        select_final_candidate(frame, candidates=candidates)


def test_final_selection_preserves_trainability_guard(frame, candidates):
    frame.loc[frame.label_code.isin(["F0", "D"]), "label_code"] = "V2"
    with pytest.raises(
        ValueError, match="both explicit positive and negative training"
    ):
        select_final_candidate(frame, candidates=candidates)


def test_prediction_rejects_activation_of_ko_model(frame):
    model = ReproducibilityModel(["guide_n"]).fit(frame)
    frame["perturbation_modality"] = "CRISPRa"
    with pytest.raises(ValueError, match="only CRISPR_KO modality"):
        model.predict(frame)


@pytest.mark.parametrize("direction", ["unknown", "RESISTANCE", "resistance "])
def test_prediction_rejects_unknown_candidate_direction(frame, direction):
    model = ReproducibilityModel(["guide_n"]).fit(frame)
    frame["phenotype_direction"] = direction
    with pytest.raises(ValueError, match="accepts only resistance"):
        model.predict(frame)


def test_direction_features_respect_each_rows_native_lfc_convention():
    source = pd.DataFrame(
        {
            "phenotype_direction": [
                "resistance",
                "resistance",
                "sensitization",
                "sensitization",
            ],
            "signal_direction": [
                "resistance",
                "resistance",
                "sensitization",
                "sensitization",
            ],
            "median_guide_lfc": [2.0, -2.0, -3.0, 3.0],
            "mean_guide_lfc": [1.5, -1.5, -2.5, 2.5],
            "positive_lfc_means": [
                "resistance",
                "sensitization",
                "resistance",
                "sensitization",
            ],
        }
    )
    prepared = _prepare_direction_features(source)
    assert prepared.direction_aligned_median_guide_lfc.tolist() == [2.0, 2.0, 3.0, 3.0]
    assert prepared.direction_aligned_mean_guide_lfc.tolist() == [1.5, 1.5, 2.5, 2.5]
    assert prepared.direction_signal_agreement.eq(1.0).all()
    # Preserve the source measurements; only the derived features are aligned.
    pd.testing.assert_series_equal(prepared.median_guide_lfc, source.median_guide_lfc)


def test_model_predictions_invariant_to_reversed_native_sign_and_declaration(frame):
    frame["median_guide_lfc"] = frame.guide_n - 3.0
    frame["mean_guide_lfc"] = frame.guide_n - 3.5
    frame["signal_direction"] = np.where(
        frame.median_guide_lfc >= 0, "resistance", "sensitization"
    )
    frame["positive_lfc_means"] = "resistance"
    reversed_source = frame.copy()
    reversed_source[["median_guide_lfc", "mean_guide_lfc"]] *= -1
    reversed_source["positive_lfc_means"] = "sensitization"
    features = [
        "direction_aligned_median_guide_lfc",
        "direction_aligned_mean_guide_lfc",
        "direction_signal_agreement",
    ]
    model = ReproducibilityModel(features).fit(frame, selection_groups=frame.study_id)
    reversed_model = ReproducibilityModel(features).fit(
        reversed_source, selection_groups=reversed_source.study_id
    )
    pd.testing.assert_frame_equal(model.predict(frame), model.predict(reversed_source))
    pd.testing.assert_frame_equal(
        model.predict(frame), reversed_model.predict(reversed_source)
    )


@pytest.mark.parametrize("value", [pd.NA, None, "", "unknown", "neutral", "Resistance"])
@pytest.mark.parametrize("raw_features_present", [True, False])
def test_invalid_or_missing_native_sign_declaration_fails_closed(
    value, raw_features_present
):
    frame = pd.DataFrame({"positive_lfc_means": ["resistance", value]})
    if raw_features_present:
        frame["phenotype_direction"] = "resistance"
        frame["signal_direction"] = "resistance"
        frame["median_guide_lfc"] = 1.0
        frame["mean_guide_lfc"] = 1.0
    with pytest.raises(
        ValueError, match="positive_lfc_means must contain only complete"
    ):
        _prepare_direction_features(frame)


def test_legacy_absent_native_sign_matches_explicit_positive_resistance():
    frame = pd.DataFrame(
        {
            "phenotype_direction": ["resistance", "sensitization"],
            "signal_direction": ["resistance", "sensitization"],
            "median_guide_lfc": [2.0, -2.0],
            "mean_guide_lfc": [1.5, -1.5],
        }
    )
    explicit = frame.assign(positive_lfc_means="resistance")
    columns = [
        "direction_aligned_median_guide_lfc",
        "direction_aligned_mean_guide_lfc",
        "direction_signal_agreement",
    ]
    pd.testing.assert_frame_equal(
        _prepare_direction_features(frame)[columns],
        _prepare_direction_features(explicit)[columns],
    )
