"""Selection-aware baseline and grouped out-of-fold evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from .contracts import (
    CandidateRecord,
    ClinicalTrialContextRecord,
    EvidenceRecord,
    ImmuneScreenEvidenceRecord,
    PatientMolecularEvidenceRecord,
    PreclinicalEvidenceRecord,
    TreatmentDiseaseContextRecord,
    validate_records,
)
from .evaluation import _composite_query_keys, grouped_ranking_metrics
from .labels import model_target

SCREEN_ONLY_FEATURE_COLUMNS = [
    "guide_n",
    "absolute_median_guide_lfc",
    "absolute_mean_guide_lfc",
    "is_sensitization_signal",
    "is_neutral_signal",
    "direction_aligned_median_guide_lfc",
    "direction_aligned_mean_guide_lfc",
    "direction_signal_agreement",
    "guide_lfc_mad",
    "guide_lfc_iqr",
    "guide_direction_agreement",
    "top2_abs_lfc_mean",
    "leave_one_guide_out_median_sd",
    "strongest_guide_dominance",
    "mean_control_count",
    "low_count_fraction",
    "zero_fraction_control",
    "zero_fraction_treatment",
    "within_screen_effect_percentile",
    "replicate_correlation",
    "control_replicate_correlation",
    "treatment_replicate_correlation",
    "replicate_effect_sd",
    "median_library_size",
]

DESIGN_FEATURE_COLUMNS = [
    "library_size_guides",
    "target_gene_count",
    "guides_per_gene_median",
    "nontargeting_guide_count",
    "library_moi",
    "effector_moi",
    "coverage_transduction",
    "coverage_selection",
    "coverage_harvest",
    "minimum_declared_coverage",
    "infection_replicate_count",
    "observed_infection_replicate_count",
    "observed_biological_replicate_count",
    "observed_technical_replicate_count",
    "observed_timepoint_count",
    "independent_infection_replicates_declared",
    "paired_replicate_fraction",
    "vehicle_control_present",
    "baseline_control_present",
    "same_infection_split",
    "matched_control",
    "cnv_amplification_risk_assessed",
    "control_sample_n",
    "treatment_sample_n",
    "baseline_sample_n",
    "antibiotic_selection_days",
    "editing_maturation_days",
    "plasmid_reads_per_guide",
    "plasmid_zero_guide_fraction",
    "plasmid_skew_ratio",
    "screen_reads_per_guide",
    "gdna_fraction_amplified",
    "pcr_cycle_count",
    "exposure_days",
    "recovery_days",
    "endpoint_timepoint_days",
    "treatment_dose_reported",
    "design_metadata_completeness",
    "screen_scale_genome_wide",
    "selection_strategy_positive_selection",
    "selection_strategy_negative_selection",
    "selection_strategy_bidirectional_selection",
    "selection_strategy_competitive_growth",
    "selection_strategy_marker_sort",
    "control_type_vehicle",
    "control_type_untreated",
    "control_type_baseline_t0",
    "control_type_later_timepoint",
    "control_type_matched_nontargeting",
    "control_type_unknown",
]

FEATURE_PROFILES = {
    "screen_only": SCREEN_ONLY_FEATURE_COLUMNS,
    "screen_plus_design": [
        *SCREEN_ONLY_FEATURE_COLUMNS,
        *DESIGN_FEATURE_COLUMNS,
    ],
}

DEFAULT_FEATURE_COLUMNS = FEATURE_PROFILES["screen_only"]

FORBIDDEN_SUCCESS_FEATURES = {
    "label_code",
    "target",
    "testing_status",
    "author_hit",
    "phenotype_reproduced",
    "opposite_direction_reproduced",
    "perturbation_confirmed",
    "rescue_performed",
    "causal_reversal_performed",
    "validation_outcome",
    "validation_label",
}

REPORT_ONLY_CONTRACT_FEATURES = {
    field_name
    for contract in (
        EvidenceRecord,
        ImmuneScreenEvidenceRecord,
        TreatmentDiseaseContextRecord,
        ClinicalTrialContextRecord,
        PreclinicalEvidenceRecord,
        PatientMolecularEvidenceRecord,
    )
    for field_name in contract.model_fields
}


def validate_success_feature_columns(feature_columns: list[str]) -> None:
    forbidden_tokens = (
        "report_only",
        "validation",
        "phenotype_reproduced",
        "opposite_direction",
        "rescue",
        "causal_reversal",
        "perturbation_confirmed",
        "author_hit",
        "testing_status",
        "label_code",
        "clinical_trial",
        "patient_outcome",
        "patient_molecular",
        "preclinical",
        "translation_context",
        "organoid",
        "pdx",
        "in_vivo",
    )
    forbidden = sorted(
        {
            column
            for column in feature_columns
            if column in FORBIDDEN_SUCCESS_FEATURES
            or column in REPORT_ONLY_CONTRACT_FEATURES
            or any(token in column.lower() for token in forbidden_tokens)
        }
    )
    if forbidden:
        raise ValueError(
            "success-model features contain label, selection, or validation "
            f"leakage fields: {forbidden}"
        )


def _study_family_groups(
    frame: pd.DataFrame,
    *,
    study_column: str,
    family_columns: tuple[str, ...],
) -> pd.Series:
    """Keep transitively connected source/raw families in one benchmark fold."""

    studies = frame[study_column].astype(str)
    parents: dict[str, str] = {}

    def find(value: str) -> str:
        parents.setdefault(value, value)
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    columns = [study_column, *family_columns]
    for values in frame[columns].itertuples(index=False, name=None):
        study_id, *family_ids = values
        study_id = str(study_id)
        study_node = f"study:{study_id}"
        find(study_node)
        for family_column, family_id in zip(family_columns, family_ids, strict=True):
            if pd.notna(family_id) and str(family_id).strip():
                family_node = f"{family_column}:{str(family_id).strip()}"
                union(study_node, family_node)
    return studies.map(lambda value: find(f"study:{value}"))


class WeightedMedianStandardizer(BaseEstimator, TransformerMixin):
    """Weighted median imputation plus weighted scaling and missing indicators."""

    def fit(
        self,
        x: pd.DataFrame | np.ndarray,
        y: object = None,
        *,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> WeightedMedianStandardizer:
        values = np.asarray(x, dtype=float)
        if values.ndim != 2:
            raise ValueError("preprocessor input must be two-dimensional")
        weights = (
            np.ones(values.shape[0], dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        if (
            len(weights) != values.shape[0]
            or not np.isfinite(weights).all()
            or (weights <= 0).any()
        ):
            raise ValueError("sample_weight must be finite, positive, and row-aligned")

        medians = []
        supported = []
        for column_index in range(values.shape[1]):
            column = values[:, column_index]
            observed = np.isfinite(column)
            supported.append(bool(observed.any()))
            if not observed.any():
                medians.append(0.0)
                continue
            ordered = np.argsort(column[observed])
            ordered_values = column[observed][ordered]
            ordered_weights = weights[observed][ordered]
            cutoff = 0.5 * ordered_weights.sum()
            median_index = int(
                np.searchsorted(
                    np.cumsum(ordered_weights),
                    cutoff,
                    side="left",
                )
            )
            medians.append(float(ordered_values[median_index]))
        self.medians_ = np.asarray(medians, dtype=float)
        self.supported_columns_ = np.asarray(supported, dtype=bool)
        self.unsupported_columns_ = np.flatnonzero(~self.supported_columns_)
        # Stable masks are created for every prespecified feature so that
        # test-only missingness is never silently median-imputed without an
        # indicator.
        self.indicator_columns_ = np.arange(values.shape[1], dtype=int)
        augmented = self._impute_and_augment(values)
        self.mean_ = np.average(augmented, axis=0, weights=weights)
        variance = np.average(
            np.square(augmented - self.mean_),
            axis=0,
            weights=weights,
        )
        self.scale_ = np.sqrt(variance)
        self.scale_[self.scale_ == 0] = 1.0
        self.n_features_in_ = values.shape[1]
        return self

    def _impute_and_augment(self, values: np.ndarray) -> np.ndarray:
        missing = ~np.isfinite(values)
        filled = values.copy()
        if missing.any():
            row_indices, column_indices = np.where(missing)
            filled[row_indices, column_indices] = self.medians_[column_indices]
        if len(self.indicator_columns_):
            indicators = missing[:, self.indicator_columns_].astype(float)
            return np.concatenate([filled, indicators], axis=1)
        return filled

    def transform(
        self,
        x: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("preprocessor input has incompatible shape")
        augmented = self._impute_and_augment(values)
        return (augmented - self.mean_) / self.scale_


def _prepare_direction_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    required = {
        "phenotype_direction",
        "signal_direction",
        "median_guide_lfc",
        "mean_guide_lfc",
    }
    if required <= set(result.columns):
        expected_sign = result["phenotype_direction"].map(
            {
                "resistance": 1.0,
                "sensitization": -1.0,
                "neutral": 0.0,
                "discordant": np.nan,
                "unknown": np.nan,
            }
        )
        result["direction_aligned_median_guide_lfc"] = (
            result["median_guide_lfc"] * expected_sign
        )
        result["direction_aligned_mean_guide_lfc"] = (
            result["mean_guide_lfc"] * expected_sign
        )
        result["direction_signal_agreement"] = (
            (
                result["phenotype_direction"].astype(str)
                == result["signal_direction"].astype(str)
            )
            .astype(float)
            .where(expected_sign.notna())
        )
    return result


def _preflight_candidate_frame(
    frame: pd.DataFrame,
    *,
    label_column: str,
    tested_column: str,
) -> None:
    """Enforce candidate-label semantics before any fitting or scoring."""

    required = set(CandidateRecord.model_fields) - {
        "label_code",
        "testing_status",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"model frame is missing candidate fields: {sorted(missing)}")
    if label_column not in frame or tested_column not in frame:
        raise ValueError(f"model frame requires {label_column!r} and {tested_column!r}")
    allowed_directions = {"resistance", "sensitization"}
    observed_directions = set(
        frame["phenotype_direction"].astype(str).str.strip().str.lower()
    )
    invalid_directions = sorted(observed_directions - allowed_directions)
    if invalid_directions:
        raise ValueError(
            "v1 reproducibility model accepts only resistance or "
            f"sensitization candidate directions; found {invalid_directions}"
        )
    candidate = frame[list(required)].copy()
    candidate["label_code"] = frame[label_column]
    candidate["testing_status"] = frame[tested_column]
    valid, errors = validate_records(candidate, CandidateRecord)
    if not errors.empty or len(valid) != len(candidate):
        preview = errors.head(5).to_dict(orient="records")
        raise ValueError(f"invalid candidate model frame: {preview}")

    if frame.groupby("screen_id", dropna=False)["study_id"].nunique().gt(1).any():
        raise ValueError("each screen_id must be nested within one study_id")


def _logistic_pipeline(*, class_weight: str | None = None) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", WeightedMedianStandardizer()),
            (
                "model",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight=class_weight,
                    random_state=17,
                ),
            ),
        ]
    )


def _selection_pipeline() -> Pipeline:
    """Unweighted classifier whose probabilities retain the observed class prior."""

    return _logistic_pipeline(class_weight=None)


def _success_pipeline(model_kind: str) -> Pipeline:
    if model_kind == "logistic":
        return _logistic_pipeline(class_weight=None)
    if model_kind == "hist_gradient_boosting":
        return Pipeline(
            steps=[
                ("preprocess", WeightedMedianStandardizer()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=250,
                        max_leaf_nodes=15,
                        min_samples_leaf=10,
                        l2_regularization=1.0,
                        random_state=17,
                    ),
                ),
            ]
        )
    raise ValueError(f"unknown model_kind: {model_kind}")


def _cross_fitted_propensity(
    x: pd.DataFrame,
    selection_target: pd.Series,
    groups: pd.Series | None,
) -> np.ndarray:
    """Estimate propensities while excluding unknown testing statuses."""

    known = selection_target.notna()
    known_positions = np.flatnonzero(known.to_numpy())
    x_known = x.iloc[known_positions]
    y_known = selection_target.iloc[known_positions].astype(int)
    if y_known.nunique() < 2:
        return np.ones(len(x), dtype=float)
    if groups is None:
        raise ValueError("IPW requires independent selection_groups for cross-fitting")
    known_groups = groups.iloc[known_positions]
    if known_groups.nunique() < 2:
        raise ValueError(
            "IPW requires at least two independent groups with known testing status"
        )

    splitter = GroupKFold(n_splits=min(5, known_groups.nunique()))
    splits = list(splitter.split(x_known, groups=known_groups))
    if any(y_known.iloc[train_idx].nunique() < 2 for train_idx, _ in splits):
        raise ValueError(
            "IPW requires both testing-status classes in every independent "
            "inner training fold"
        )
    propensity = np.full(len(x), np.nan, dtype=float)
    for train_idx, test_idx in splits:
        y_train = y_known.iloc[train_idx]
        original_test_positions = known_positions[test_idx]
        model = _selection_pipeline().fit(x_known.iloc[train_idx], y_train)
        propensity[original_test_positions] = model.predict_proba(
            x_known.iloc[test_idx]
        )[:, 1]
    unknown_positions = np.flatnonzero(~known.to_numpy())
    if len(unknown_positions):
        final_model = _selection_pipeline().fit(x_known, y_known)
        propensity[unknown_positions] = final_model.predict_proba(
            x.iloc[unknown_positions]
        )[:, 1]
    if np.isnan(propensity).any():
        raise RuntimeError("cross-fitted selection propensity contains missing values")
    return propensity


def _testing_status_target(status: pd.Series) -> pd.Series:
    normalized = status.astype("string").str.strip().str.lower()
    allowed = {"tested", "not_tested", "unknown"}
    invalid = sorted(set(normalized.dropna()) - allowed)
    if invalid:
        raise ValueError(f"invalid testing_status values: {invalid}")
    return normalized.map({"tested": 1.0, "not_tested": 0.0, "unknown": np.nan})


@dataclass
class ReproducibilityModel:
    """Two-stage transparent baseline.

    The selection model estimates which genes were tested. The success model is
    fit only on explicit positive/negative validation events. Optional inverse
    propensity weights provide a sensitivity analysis, not a guarantee that
    author-selection bias has been eliminated.
    """

    feature_columns: list[str]
    selection_model: Pipeline | None = None
    success_model: Pipeline | None = None
    propensity_clip: tuple[float, float] = (0.1, 0.95)
    model_kind: str = "logistic"
    fit_diagnostics: dict[str, float] | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        label_column: str = "label_code",
        tested_column: str = "testing_status",
        selection_weighting: bool = False,
        selection_groups: pd.Series | None = None,
    ) -> ReproducibilityModel:
        validate_success_feature_columns(self.feature_columns)
        frame = _prepare_direction_features(frame)
        missing = set(self.feature_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"missing model features: {sorted(missing)}")
        _preflight_candidate_frame(
            frame,
            label_column=label_column,
            tested_column=tested_column,
        )
        if label_column not in frame or tested_column not in frame:
            raise ValueError("label_code and testing_status columns are required")

        x = frame[self.feature_columns]
        selection_target = _testing_status_target(frame[tested_column])
        selection_known = selection_target.notna()
        known_target = selection_target.loc[selection_known].astype(int)
        if known_target.nunique() == 2:
            self.selection_model = _selection_pipeline().fit(
                x.loc[selection_known],
                known_target,
            )
            if selection_weighting:
                propensity = _cross_fitted_propensity(
                    x,
                    selection_target,
                    selection_groups,
                )
            else:
                propensity = self.selection_model.predict_proba(x)[:, 1]
        else:
            if selection_weighting:
                raise ValueError(
                    "IPW requires both tested and explicitly not-tested "
                    "candidates in the training data"
                )
            self.selection_model = None
            propensity = np.ones(len(frame), dtype=float)

        targets = frame[label_column].map(model_target)
        labeled = targets.notna() & selection_target.eq(1.0)
        y = targets.loc[labeled].astype(int)
        if y.nunique() < 2:
            raise ValueError(
                "success model requires explicit positive and negative labels"
            )

        weights = np.ones(len(y), dtype=float)
        study_balancing_applied = False
        if selection_groups is not None:
            labeled_groups = selection_groups.loc[labeled].astype(str)
            group_counts = labeled_groups.value_counts()
            weights *= labeled_groups.map(
                lambda group: 1.0 / group_counts[group]
            ).to_numpy(dtype=float)
            study_balancing_applied = True
        lower, upper = self.propensity_clip
        labeled_propensity = propensity[labeled.to_numpy()]
        clipped = np.clip(labeled_propensity, lower, upper)
        if selection_weighting:
            weights *= 1.0 / clipped
        weights = weights / np.mean(weights)

        diagnostic_weights = 1.0 / clipped
        effective_sample_size = float(
            diagnostic_weights.sum() ** 2 / np.square(diagnostic_weights).sum()
        )
        self.fit_diagnostics = {
            "n_labeled": float(labeled.sum()),
            "n_selection_status_known": float(selection_known.sum()),
            "propensity_min": float(labeled_propensity.min()),
            "propensity_median": float(np.median(labeled_propensity)),
            "propensity_max": float(labeled_propensity.max()),
            "clip_lower_fraction": float(np.mean(labeled_propensity < lower)),
            "clip_upper_fraction": float(np.mean(labeled_propensity > upper)),
            "effective_sample_size": effective_sample_size,
            "study_balancing_applied": float(study_balancing_applied),
            "outcome_observation_weighting_applied": 0.0,
        }

        self.success_model = _success_pipeline(self.model_kind)
        self.success_model.fit(
            x.loc[labeled],
            y,
            preprocess__sample_weight=weights,
            model__sample_weight=weights,
        )
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.success_model is None:
            raise RuntimeError("model has not been fitted")
        frame = _prepare_direction_features(frame)
        x = frame[self.feature_columns]
        score = self.success_model.predict_proba(x)[:, 1]
        if self.selection_model is None:
            propensity = np.ones(len(frame), dtype=float)
        else:
            propensity = self.selection_model.predict_proba(x)[:, 1]
        return pd.DataFrame(
            {
                "reproducibility_score": score,
                "selection_propensity": propensity,
                "selection_propensity_known_status": propensity,
            },
            index=frame.index,
        )


def grouped_oof_predictions(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    group_column: str = "study_id",
    screen_column: str = "screen_id",
    label_column: str = "label_code",
    tested_column: str = "testing_status",
    n_splits: int = 5,
    model_kind: str = "logistic",
    source_family_column: str = "source_family_id",
    raw_family_column: str = "raw_data_family_id",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Generate study-grouped OOF predictions and ranking metrics."""

    frame = _prepare_direction_features(frame)
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    validate_success_feature_columns(list(feature_columns))
    _preflight_candidate_frame(
        frame,
        label_column=label_column,
        tested_column=tested_column,
    )
    family_columns = tuple(
        column
        for column in (source_family_column, raw_family_column)
        if column in frame and frame[column].notna().any()
    )
    if group_column == "study_id" and family_columns:
        groups = _study_family_groups(
            frame,
            study_column=group_column,
            family_columns=family_columns,
        )
    else:
        groups = frame[group_column]
    unique_groups = groups.nunique()
    if unique_groups < 2:
        raise ValueError("at least two independent studies are required")
    actual_splits = min(n_splits, unique_groups)
    if actual_splits < 2:
        raise ValueError("at least two folds are required")
    targets = frame[label_column].map(model_target)
    selection_target = _testing_status_target(frame[tested_column])
    tested = selection_target.eq(1.0)
    stratification_target = targets.fillna(-1).astype(int)
    splitter = StratifiedGroupKFold(
        n_splits=actual_splits,
        shuffle=True,
        random_state=17,
    )
    splits = list(
        splitter.split(
            frame,
            y=stratification_target,
            groups=groups,
        )
    )
    infeasible = []
    for fold, (train_idx, _) in enumerate(splits, start=1):
        train_target = targets.iloc[train_idx]
        train_labeled = train_target.notna() & tested.iloc[train_idx]
        if train_target.loc[train_labeled].nunique() < 2:
            infeasible.append(fold)
    if infeasible:
        raise ValueError(
            "grouped split is not trainable because explicit positive and "
            f"negative labels are not both present in training fold(s) {infeasible}; "
            "reduce folds or add independently failed/discordant validation events"
        )

    predictions = pd.Series(np.nan, index=frame.index, dtype=float)
    predictions_ipw = pd.Series(np.nan, index=frame.index, dtype=float)
    propensities = pd.Series(np.nan, index=frame.index, dtype=float)
    fold_ids = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    fold_groups = groups.astype(str)
    base = ReproducibilityModel(list(feature_columns), model_kind=model_kind)
    fold_diagnostics: list[dict[str, float]] = []
    ipw_available = True

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        unweighted_model = clone_model(base)
        unweighted_model.fit(
            frame.iloc[train_idx],
            label_column=label_column,
            tested_column=tested_column,
            selection_weighting=False,
            selection_groups=groups.iloc[train_idx],
        )
        predicted = unweighted_model.predict(frame.iloc[test_idx])
        predictions.iloc[test_idx] = predicted["reproducibility_score"].to_numpy()
        propensities.iloc[test_idx] = predicted["selection_propensity"].to_numpy()
        fold_ids.iloc[test_idx] = fold
        try:
            ipw_model = clone_model(base)
            ipw_model.fit(
                frame.iloc[train_idx],
                label_column=label_column,
                tested_column=tested_column,
                selection_weighting=True,
                selection_groups=groups.iloc[train_idx],
            )
        except ValueError as exc:
            if not str(exc).startswith("IPW requires"):
                raise
            ipw_available = False
        else:
            predicted_ipw = ipw_model.predict(frame.iloc[test_idx])
            predictions_ipw.iloc[test_idx] = predicted_ipw[
                "reproducibility_score"
            ].to_numpy()
            if ipw_model.fit_diagnostics is not None:
                fold_diagnostics.append(ipw_model.fit_diagnostics)

    if not ipw_available or predictions_ipw.isna().any():
        predictions_ipw[:] = np.nan
        fold_diagnostics.clear()

    output = frame.copy()
    output["target"] = targets
    output["reproducibility_score"] = predictions
    output["reproducibility_score_unweighted"] = predictions
    output["reproducibility_score_ipw"] = predictions_ipw
    output["selection_propensity"] = propensities
    output["selection_propensity_known_status"] = propensities
    output["fold"] = fold_ids
    output["_outer_fold_group"] = fold_groups
    query_columns = [screen_column]
    for column in ("contrast_id", "phenotype_direction"):
        if column in output.columns and column not in query_columns:
            query_columns.append(column)
    output["_ranking_query_id"] = _composite_query_keys(
        output,
        query_columns,
    )
    metrics = grouped_ranking_metrics(
        output,
        group_column="_ranking_query_id",
        target_column="target",
        score_column="reproducibility_score",
        study_column=group_column,
    )
    metrics["ipw_available"] = float(ipw_available and predictions_ipw.notna().all())
    metrics["ipw_scope_selection_only"] = 1.0
    if metrics["ipw_available"]:
        ipw_metrics = grouped_ranking_metrics(
            output,
            group_column="_ranking_query_id",
            target_column="target",
            score_column="reproducibility_score_ipw",
            study_column=group_column,
        )
        metrics.update({f"ipw_{name}": value for name, value in ipw_metrics.items()})
    if metrics["ipw_available"] and fold_diagnostics:
        for name in fold_diagnostics[0]:
            metrics[f"ipw_mean_{name}"] = float(
                np.mean([diagnostic[name] for diagnostic in fold_diagnostics])
            )
    return output, metrics


def clone_model(model: ReproducibilityModel) -> ReproducibilityModel:
    """Create an unfitted copy without relying on sklearn estimator semantics."""

    return ReproducibilityModel(
        feature_columns=list(model.feature_columns),
        propensity_clip=model.propensity_clip,
        model_kind=model.model_kind,
    )
