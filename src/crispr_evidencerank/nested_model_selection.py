"""Nested selection of prespecified models using independent source/raw families.

Outer held-out outcomes never select a model, feature profile, or regularization
strength. The returned scores evaluate the *selection procedure*: different outer
folds may select different candidates. They are relative retrospective scores,
not prospectively calibrated probabilities or a final deployment model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .evaluation import (
    _composite_query_keys,
    _require_complete_identifiers,
    grouped_ranking_metrics,
)
from .labels import model_target
from .modeling import (
    DEFAULT_FEATURE_COLUMNS,
    ReproducibilityModel,
    _preflight_candidate_frame,
    _prepare_direction_features,
    _study_family_groups,
    validate_success_feature_columns,
)


@dataclass(frozen=True)
class ModelCandidate:
    """One configuration, to be declared before inspecting outer outcomes.

    Candidate order breaks exact inner-score ties. Boosting uses the fixed
    parameters in ``ReproducibilityModel``; ``logistic_c`` tunes only the success
    logistic model, never the separate testing-selection model.
    """

    candidate_id: str
    feature_columns: tuple[str, ...] = tuple(DEFAULT_FEATURE_COLUMNS)
    model_kind: str = "logistic"
    logistic_c: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a nonempty string")
        if self.candidate_id != self.candidate_id.strip():
            raise ValueError("candidate_id must not contain surrounding whitespace")
        # Canonical immutable columns prevent caller mutation during selection.
        object.__setattr__(self, "feature_columns", tuple(self.feature_columns))
        validate_success_feature_columns(list(self.feature_columns))
        if self.model_kind not in {"logistic", "hist_gradient_boosting"}:
            raise ValueError(f"unknown model_kind: {self.model_kind}")
        if (
            isinstance(self.logistic_c, bool)
            or not isinstance(self.logistic_c, Real)
            or not np.isfinite(self.logistic_c)
            or self.logistic_c <= 0
        ):
            raise ValueError("logistic_c must be finite and strictly positive")
        if self.model_kind != "logistic" and self.logistic_c != 1.0:
            raise ValueError("logistic_c can only be changed for logistic models")
        object.__setattr__(self, "logistic_c", float(self.logistic_c))

    def make_model(
        self,
        *,
        random_state: int,
        propensity_clip: tuple[float, float],
    ) -> ReproducibilityModel:
        return ReproducibilityModel(
            feature_columns=list(self.feature_columns),
            model_kind=self.model_kind,
            logistic_c=self.logistic_c,
            random_state=random_state,
            propensity_clip=propensity_clip,
        )


@dataclass
class NestedModelSelectionResult:
    predictions: pd.DataFrame
    metrics: dict[str, float]
    diagnostics: dict[str, Any]


@dataclass
class FinalCandidateSelectionResult:
    selected_candidate: ModelCandidate
    diagnostics: dict[str, Any]


def _folds(
    frame: pd.DataFrame,
    groups: pd.Series,
    n_splits: int,
    *,
    context: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    count = min(n_splits, int(groups.nunique()))
    if count < 2:
        raise ValueError(f"{context} requires at least two independent families")
    # Deliberately outcome-independent: changing an outer test label cannot
    # change which training families enter a particular inner fold.
    return list(GroupKFold(n_splits=count).split(frame, groups=groups))


def _require_trainable(frame: pd.DataFrame, *, context: str) -> None:
    targets = frame["label_code"].map(model_target)
    tested = frame["testing_status"].eq("tested")
    if targets.loc[targets.notna() & tested].nunique() != 2:
        raise ValueError(
            f"{context} requires both explicit positive and negative training labels; "
            "add independent adjudicated families or prespecify another split"
        )


def _metric_frame(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    result["target"] = result["label_code"].map(model_target)
    result["reproducibility_score"] = scores
    result["_ranking_query_id"] = _composite_query_keys(
        result, ["screen_id", "contrast_id", "phenotype_direction"]
    )
    return result


def _metrics(frame: pd.DataFrame, *, ks: tuple[int, ...]) -> dict[str, float]:
    return grouped_ranking_metrics(
        frame,
        group_column="_ranking_query_id",
        target_column="target",
        score_column="reproducibility_score",
        study_column="study_id",
        ks=ks,
    )


def _finite_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    """Use JSON null, never nonstandard NaN, in the auditable diagnostics."""

    return {
        key: float(value) if np.isfinite(value) else None
        for key, value in metrics.items()
    }


def _partition_manifest(
    frame: pd.DataFrame,
    groups: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, Any]:
    train_groups = set(groups.iloc[train_idx].astype(str))
    test_groups = set(groups.iloc[test_idx].astype(str))
    if train_groups & test_groups:
        raise RuntimeError("nested split contains overlapping independent families")
    return {
        "train_families": sorted(train_groups),
        "test_families": sorted(test_groups),
        "train_studies": sorted(set(frame.iloc[train_idx]["study_id"].astype(str))),
        "test_studies": sorted(set(frame.iloc[test_idx]["study_id"].astype(str))),
        "n_train_rows": len(train_idx),
        "n_test_rows": len(test_idx),
    }


def _fit_manifest(model: ReproducibilityModel) -> dict[str, Any]:
    if model.success_model is None:
        raise RuntimeError("expected a fitted model")
    preprocess = model.success_model.named_steps["preprocess"]
    # This exposes training-derived imputation/support without exporting a
    # pickled estimator, and allows an audit of test-only feature missingness.
    return {
        "diagnostics": model.fit_diagnostics,
        "feature_columns": list(model.feature_columns),
        "training_medians": preprocess.medians_.tolist(),
        "unsupported_features": [
            model.feature_columns[index] for index in preprocess.unsupported_columns_
        ],
        "success_hyperparameters": model.success_model.named_steps["model"].get_params(
            deep=False
        ),
    }


def _validated_inputs(
    frame: pd.DataFrame,
    *,
    candidates: tuple[ModelCandidate, ...] | list[ModelCandidate],
    inner_splits: int,
    selection_metric: str,
    ks: tuple[int, ...],
    random_state: int,
    propensity_clip: tuple[float, float],
    source_family_column: str,
    raw_family_column: str,
) -> tuple[pd.DataFrame, pd.Series, tuple[ModelCandidate, ...]]:
    if (
        isinstance(inner_splits, bool)
        or not isinstance(inner_splits, int)
        or inner_splits < 2
    ):
        raise ValueError("inner_splits must be an integer of at least two")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise ValueError("random_state must be an integer")
    if not 0 <= random_state <= 2**32 - 1:
        raise ValueError("random_state must be between 0 and 2**32 - 1")
    if (
        not ks
        or len(set(ks)) != len(ks)
        or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in ks)
    ):
        raise ValueError("ks must contain distinct positive integers")
    ks = tuple(ks)
    allowed_metrics = {
        "study_macro_average_precision",
        "study_macro_normalized_average_precision",
        *(f"study_macro_adjudicated_precision_at_{k}" for k in ks),
        *(f"study_macro_strict_adjudicated_ndcg_at_{k}" for k in ks),
    }
    if selection_metric not in allowed_metrics:
        raise ValueError(
            "selection_metric must be a prespecified within-query ranking metric: "
            f"{sorted(allowed_metrics)}"
        )
    candidates = tuple(candidates)
    if not candidates or any(
        not isinstance(item, ModelCandidate) for item in candidates
    ):
        raise ValueError("candidates must contain at least one ModelCandidate")
    candidate_ids = [item.candidate_id for item in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_id values must be unique")
    lower, upper = propensity_clip
    if not 0 < lower < upper <= 1:
        raise ValueError("propensity_clip must satisfy 0 < lower < upper <= 1")
    if source_family_column == raw_family_column:
        raise ValueError("source and raw family columns must be distinct")
    _preflight_candidate_frame(
        frame, label_column="label_code", tested_column="testing_status"
    )
    _require_complete_identifiers(
        frame,
        [
            "study_id",
            "screen_id",
            "contrast_id",
            source_family_column,
            raw_family_column,
        ],
        context="nested source/raw-family benchmark",
    )
    # Duplicate caller indices are supported by working positionally; no score
    # assignment or propensity weighting aligns against a nonunique input index.
    working = _prepare_direction_features(frame.reset_index(drop=True))
    requested_features = {
        column for item in candidates for column in item.feature_columns
    }
    missing_features = requested_features - set(working)
    if missing_features:
        raise ValueError(f"missing model features: {sorted(missing_features)}")
    groups = _study_family_groups(
        working,
        study_column="study_id",
        family_columns=(source_family_column, raw_family_column),
    )
    return working, groups, candidates


def _select_by_inner_oof(
    train: pd.DataFrame,
    train_groups: pd.Series,
    inner: list[tuple[np.ndarray, np.ndarray]],
    *,
    candidates: tuple[ModelCandidate, ...],
    selection_metric: str,
    ks: tuple[int, ...],
    random_state: int,
    propensity_clip: tuple[float, float],
    context: str,
) -> tuple[int, list[dict[str, Any]]]:
    candidate_scores = []
    diagnostics = []
    for candidate in candidates:
        scores = np.full(len(train), np.nan)
        fit_diagnostics = []
        for inner_fold, (inner_train, inner_test) in enumerate(inner, start=1):
            model = candidate.make_model(
                random_state=random_state, propensity_clip=propensity_clip
            ).fit(
                train.iloc[inner_train],
                selection_groups=train_groups.iloc[inner_train],
                selection_weighting=False,
            )
            scores[inner_test] = model.predict(train.iloc[inner_test])[
                "reproducibility_score"
            ].to_numpy()
            fit_diagnostics.append({"inner_fold": inner_fold, **_fit_manifest(model)})
        if not np.isfinite(scores).all():
            raise RuntimeError("inner out-of-fold predictions are incomplete")
        inner_metrics = _metrics(_metric_frame(train, scores), ks=ks)
        value = inner_metrics[selection_metric]
        if not np.isfinite(value):
            raise ValueError(
                f"{context}: selection metric {selection_metric!r} "
                "is undefined on inner OOF queries; insufficient two-class "
                "queries or requested cutoff covers every adjudicated candidate"
            )
        candidate_scores.append(value)
        diagnostics.append(
            {
                "candidate_id": candidate.candidate_id,
                "selection_value": value,
                "inner_oof_metrics": _finite_metrics(inner_metrics),
                "fits": fit_diagnostics,
            }
        )
    return int(np.argmax(candidate_scores)), diagnostics


def nested_grouped_oof_predictions(
    frame: pd.DataFrame,
    *,
    candidates: tuple[ModelCandidate, ...] | list[ModelCandidate],
    outer_splits: int = 5,
    inner_splits: int = 3,
    selection_metric: str = "study_macro_average_precision",
    ks: tuple[int, ...] = (5, 10, 20),
    random_state: int = 17,
    propensity_clip: tuple[float, float] = (0.1, 0.95),
    source_family_column: str = "source_family_id",
    raw_family_column: str = "raw_data_family_id",
) -> NestedModelSelectionResult:
    """Evaluate model/feature/C selection without reusing outer test outcomes.

    A complete, upstream-reviewed family map is required. Study/source/raw
    connectivity is computed once from identifiers, with transitive related
    studies indivisible in every outer and inner fold. Each inner prediction is
    generated by an independently fitted pipeline. Candidate selection maximizes
    the chosen metric on the *pooled inner out-of-fold rows*, macro-averaged by
    study using the existing query definitions; it is not a mean of unequal
    fold-level averages. Single-class queries remain in coverage diagnostics but
    provide no ranking information. An undefined selection metric aborts.

    The primary fit balances independent-family mass, matching the existing
    grouped benchmark. IPW and calibration are not selected here. Family IDs
    cannot certify upstream deduplication, and frozen Python configurations do
    not constitute a human preregistration. This API consumes the v1 KO candidate
    contract; if modality metadata is present it must declare CRISPR_KO.
    """

    if (
        isinstance(outer_splits, bool)
        or not isinstance(outer_splits, int)
        or outer_splits < 2
    ):
        raise ValueError("outer_splits must be an integer of at least two")
    working, groups, candidates = _validated_inputs(
        frame,
        candidates=candidates,
        inner_splits=inner_splits,
        selection_metric=selection_metric,
        ks=ks,
        random_state=random_state,
        propensity_clip=propensity_clip,
        source_family_column=source_family_column,
        raw_family_column=raw_family_column,
    )
    original_index = frame.index.copy()
    if groups.nunique() < 3:
        raise ValueError(
            "nested selection requires at least three independent families"
        )
    outer = _folds(working, groups, outer_splits, context="outer selection")

    # Preflight the entire declared design before any estimator fits. We neither
    # repair folds using held-out outcomes nor skip failed folds/candidates.
    inner_by_outer = []
    for outer_fold, (train_idx, _) in enumerate(outer, start=1):
        train = working.iloc[train_idx]
        train_groups = groups.iloc[train_idx]
        _require_trainable(train, context=f"outer fold {outer_fold}")
        inner = _folds(
            train,
            train_groups,
            inner_splits,
            context=f"outer fold {outer_fold} inner CV",
        )
        for inner_fold, (inner_train, _) in enumerate(inner, start=1):
            _require_trainable(
                train.iloc[inner_train],
                context=f"outer fold {outer_fold}, inner fold {inner_fold}",
            )
        inner_by_outer.append(inner)

    output_scores = np.full(len(working), np.nan)
    output_propensity = np.full(len(working), np.nan)
    output_folds = np.zeros(len(working), dtype=int)
    output_candidate = np.full(len(working), "", dtype=object)
    fold_diagnostics = []
    for outer_fold, ((train_idx, test_idx), inner) in enumerate(
        zip(outer, inner_by_outer, strict=True), start=1
    ):
        train = working.iloc[train_idx]
        train_groups = groups.iloc[train_idx]
        outer_diagnostic = {
            "outer_fold": outer_fold,
            **_partition_manifest(working, groups, train_idx, test_idx),
            "inner_folds": [
                {
                    "inner_fold": inner_fold,
                    **_partition_manifest(train, train_groups, inner_train, inner_test),
                }
                for inner_fold, (inner_train, inner_test) in enumerate(inner, start=1)
            ],
            "candidates": [],
        }
        winner_index, inner_diagnostics = _select_by_inner_oof(
            train,
            train_groups,
            inner,
            candidates=candidates,
            selection_metric=selection_metric,
            ks=ks,
            random_state=random_state,
            propensity_clip=propensity_clip,
            context=f"outer fold {outer_fold}",
        )
        outer_diagnostic["candidates"] = inner_diagnostics
        winner = candidates[winner_index]
        final_model = winner.make_model(
            random_state=random_state, propensity_clip=propensity_clip
        ).fit(train, selection_groups=train_groups, selection_weighting=False)
        predicted = final_model.predict(working.iloc[test_idx])
        output_scores[test_idx] = predicted["reproducibility_score"].to_numpy()
        output_propensity[test_idx] = predicted["selection_propensity"].to_numpy()
        output_folds[test_idx] = outer_fold
        output_candidate[test_idx] = winner.candidate_id
        outer_diagnostic["selected_candidate_id"] = winner.candidate_id
        outer_diagnostic["selected_inner_score"] = inner_diagnostics[winner_index][
            "selection_value"
        ]
        outer_diagnostic["outer_training_fit"] = _fit_manifest(final_model)
        fold_diagnostics.append(outer_diagnostic)
    if not np.isfinite(output_scores).all() or (output_folds == 0).any():
        raise RuntimeError("nested outer out-of-fold predictions are incomplete")

    output = _metric_frame(working, output_scores)
    output["selection_propensity"] = output_propensity
    output["fold"] = output_folds
    output["selected_candidate_id"] = output_candidate
    output["_outer_fold_group"] = groups.to_numpy()
    metrics = _metrics(output, ks=ks)
    metrics["nested_model_selection"] = 1.0
    metrics["n_outer_folds"] = float(len(outer))
    metrics["n_independent_families"] = float(groups.nunique())
    metrics["n_prespecified_candidates"] = float(len(candidates))
    for direction, subset in output.groupby("phenotype_direction", sort=True):
        target = subset["target"].dropna()
        prefix = f"direction_{direction}_"
        metrics[f"{prefix}n_adjudicated"] = float(len(target))
        metrics[f"{prefix}n_target_classes"] = float(target.nunique())
        available = 0.0
        if len(target):
            direction_metrics = _metrics(subset, ks=ks)
            metrics.update(
                {f"{prefix}{key}": value for key, value in direction_metrics.items()}
            )
            available = float(direction_metrics["n_groups_with_both_classes"] > 0)
        metrics[f"{prefix}ranking_available"] = available
    diagnostics = {
        "procedure": "nested_source_raw_family_model_selection",
        "selection_metric": selection_metric,
        "selection_aggregation": "pooled_inner_oof_query_then_study_macro",
        "tie_policy": "first_candidate_in_prespecified_order",
        "splitter": "GroupKFold_outcome_independent",
        "requested_outer_splits": outer_splits,
        "requested_inner_splits": inner_splits,
        "actual_outer_splits": len(outer),
        "random_state": random_state,
        "random_state_scope": "estimators_only; deterministic_group_splits",
        "ks": list(ks),
        "propensity_clip": list(propensity_clip),
        "selection_weighting_applied": False,
        "family_balancing_applied": True,
        "calibration_applied": False,
        "score_interpretation": "relative_retrospective_reproducibility_score",
        "modality_metadata_verified": "perturbation_modality" in working,
        "family_columns": [source_family_column, raw_family_column],
        "candidates": [asdict(candidate) for candidate in candidates],
        "folds": fold_diagnostics,
    }
    output = output.drop(columns=["_ranking_query_id"])
    output.index = original_index
    return NestedModelSelectionResult(output, metrics, diagnostics)


def select_final_candidate(
    frame: pd.DataFrame,
    *,
    candidates: tuple[ModelCandidate, ...] | list[ModelCandidate],
    inner_splits: int = 3,
    selection_metric: str = "study_macro_average_precision",
    ks: tuple[int, ...] = (5, 10, 20),
    random_state: int = 17,
    propensity_clip: tuple[float, float] = (0.1, 0.95),
    source_family_column: str = "source_family_id",
    raw_family_column: str = "raw_data_family_id",
) -> FinalCandidateSelectionResult:
    """Choose a deployment configuration from development-only grouped OOF.

    Call only after freezing the procedure evaluated by nested CV. ``frame``
    must exclude the prospective/reserved case and all related raw/source
    families. This function cannot verify an unseen holdout manifest; the caller
    must enforce that boundary. The inner score is for choosing a configuration,
    not an unbiased performance estimate. Fit the selected candidate on this
    same development frame before serializing; never choose by outer fold
    performance or by the most frequently selected outer candidate.
    """

    working, groups, candidates = _validated_inputs(
        frame,
        candidates=candidates,
        inner_splits=inner_splits,
        selection_metric=selection_metric,
        ks=ks,
        random_state=random_state,
        propensity_clip=propensity_clip,
        source_family_column=source_family_column,
        raw_family_column=raw_family_column,
    )
    inner = _folds(working, groups, inner_splits, context="final candidate selection")
    for inner_fold, (train_idx, _) in enumerate(inner, start=1):
        _require_trainable(
            working.iloc[train_idx], context=f"final selection inner fold {inner_fold}"
        )
    winner_index, candidate_diagnostics = _select_by_inner_oof(
        working,
        groups,
        inner,
        candidates=candidates,
        selection_metric=selection_metric,
        ks=ks,
        random_state=random_state,
        propensity_clip=propensity_clip,
        context="final candidate selection",
    )
    winner = candidates[winner_index]
    diagnostics = {
        "procedure": "development_only_final_candidate_selection",
        "selected_candidate_id": winner.candidate_id,
        "selected_candidate": asdict(winner),
        "selection_metric": selection_metric,
        "selected_inner_score": candidate_diagnostics[winner_index]["selection_value"],
        "selection_aggregation": "pooled_inner_oof_query_then_study_macro",
        "selection_score_is_unbiased_performance_estimate": False,
        "external_holdout_exclusion_must_be_verified_by_caller": True,
        "tie_policy": "first_candidate_in_prespecified_order",
        "splitter": "GroupKFold_outcome_independent",
        "requested_inner_splits": inner_splits,
        "actual_inner_splits": len(inner),
        "random_state": random_state,
        "random_state_scope": "estimators_only; deterministic_group_splits",
        "ks": list(ks),
        "propensity_clip": list(propensity_clip),
        "family_balancing_applied": True,
        "selection_weighting_applied": False,
        "calibration_applied": False,
        "score_interpretation": "relative_retrospective_reproducibility_score",
        "family_columns": [source_family_column, raw_family_column],
        "n_development_rows": len(working),
        "development_studies": sorted(set(working.study_id.astype(str))),
        "development_families": sorted(set(groups.astype(str))),
        "candidate_configurations": [asdict(candidate) for candidate in candidates],
        "candidates": candidate_diagnostics,
        "inner_folds": [
            {
                "inner_fold": inner_fold,
                **_partition_manifest(working, groups, train_idx, test_idx),
            }
            for inner_fold, (train_idx, test_idx) in enumerate(inner, start=1)
        ],
    }
    return FinalCandidateSelectionResult(winner, diagnostics)
