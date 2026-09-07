"""Ranking and calibration metrics for grouped screen evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, ndcg_score

TOP_K_TIE_POLICY = "fractional_expected_relevance_at_cutoff"


def _require_complete_identifiers(
    frame: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    *,
    context: str,
) -> None:
    """Fail closed when an identifier needed to define an estimand is absent."""

    missing_columns = set(columns) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{context} columns are missing: {sorted(missing_columns)}")
    incomplete = []
    for column in dict.fromkeys(columns):
        values = frame[column]
        missing = values.isna()
        blank = values.astype("string").str.strip().eq("").fillna(True)
        missing = missing | blank
        if missing.any():
            incomplete.append(column)
    if incomplete:
        raise ValueError(
            f"{context} identifiers contain missing or blank values: "
            f"{sorted(incomplete)}"
        )


def _composite_query_keys(
    frame: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    *,
    namespace: object | None = None,
) -> pd.Series:
    """Build collision-safe tuple keys for ranking queries."""

    _require_complete_identifiers(frame, columns, context="query")
    row_keys = list(frame[list(columns)].itertuples(index=False, name=None))
    if namespace is not None:
        row_keys = [(namespace, key) for key in row_keys]
    return pd.Series(row_keys, index=frame.index, dtype=object)


def _expected_relevance_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int,
) -> float:
    """Expected relevance under equal inclusion within a cutoff-score tie.

    Every candidate strictly above the cutoff is selected. If the cutoff
    crosses a score tie, each member of that tie receives the same fractional
    inclusion probability. This avoids outcome-aware or row-order tie breaks.
    """

    truth = np.asarray(y_true, dtype=float)
    score = np.asarray(y_score, dtype=float)
    if truth.ndim != 1 or score.ndim != 1 or len(truth) != len(score):
        raise ValueError("y_true and y_score must be aligned one-dimensional arrays")
    if not 1 <= k <= len(truth):
        raise ValueError("k must be between 1 and the candidate count")
    if not np.isfinite(truth).all() or not np.isfinite(score).all():
        raise ValueError("top-k inputs must be finite")
    cutoff = float(np.partition(score, len(score) - k)[len(score) - k])
    above = score > cutoff
    tied = score == cutoff
    remaining = int(k - above.sum())
    tied_count = int(tied.sum())
    if remaining < 0 or remaining > tied_count or tied_count == 0:
        raise RuntimeError("invalid top-k cutoff partition")
    return float(truth[above].sum() + (remaining / tied_count) * truth[tied].sum())


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Return tie-invariant precision at a genuinely truncated cutoff.

    When ``k`` covers the complete candidate list, precision is just the class
    prevalence and contains no ranking information.  Returning ``NaN`` makes
    that loss of resolving power explicit instead of silently reporting a
    score-independent value. A tie crossing the cutoff uses equal fractional
    inclusion for every tied candidate (``TOP_K_TIE_POLICY``).
    """

    if k < 1:
        raise ValueError("k must be a positive integer")
    if len(y_true) == 0 or k >= len(y_true):
        return float("nan")
    return float(_expected_relevance_at_k(y_true, y_score, k) / k)


def _normalized_average_precision(average_precision: float, prevalence: float) -> float:
    """Scale excess AP by the available improvement above prevalence.

    This is a descriptive normalization, not a claim that AP has become fully
    prevalence invariant.  It is undefined for an all-positive target.
    """

    if not np.isfinite(average_precision) or not 0 <= prevalence < 1:
        return float("nan")
    return float((average_precision - prevalence) / (1.0 - prevalence))


def _calibration_slope_intercept(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[float, float]:
    """Estimate logistic recalibration diagnostics on probability-like scores."""

    if len(y_true) < 2 or np.unique(y_true).size < 2:
        return float("nan"), float("nan")
    if not np.isfinite(y_score).all() or (y_score < 0).any() or (y_score > 1).any():
        return float("nan"), float("nan")
    clipped = np.clip(y_score, 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    if np.ptp(logits) <= 1e-12:
        return float("nan"), float("nan")
    recalibration = LogisticRegression(
        C=1e12,
        solver="lbfgs",
        max_iter=2_000,
    ).fit(logits, y_true.astype(int))
    return (
        float(recalibration.coef_[0, 0]),
        float(recalibration.intercept_[0]),
    )


def _nested_macro_mean(
    records: list[tuple[str, float]],
) -> tuple[float, float]:
    clean = [(study, value) for study, value in records if np.isfinite(value)]
    if not clean:
        return float("nan"), float("nan")
    query_macro = float(np.mean([value for _, value in clean]))
    study_values: dict[str, list[float]] = {}
    for study, value in clean:
        study_values.setdefault(study, []).append(value)
    study_macro = float(np.mean([np.mean(values) for values in study_values.values()]))
    return query_macro, study_macro


def _study_for_group(
    group: pd.DataFrame,
    study_column: str | None,
) -> str:
    if study_column is None:
        return "__all_studies__"
    values = group[study_column].dropna().astype(str).unique()
    if len(values) != 1:
        raise ValueError("each ranking query must be nested within exactly one study")
    return str(values[0])


def within_query_feature_invariance(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | tuple[str, ...],
    query_columns: list[str] | tuple[str, ...],
) -> dict[str, object]:
    """Report which features can vary inside the declared ranking query."""

    missing = (set(feature_columns) | set(query_columns)) - set(frame)
    if missing:
        raise ValueError(f"feature invariance columns are missing: {sorted(missing)}")
    if not feature_columns:
        raise ValueError("feature_columns must not be empty")
    working = frame.copy()
    working["_invariance_query"] = _composite_query_keys(
        working,
        list(query_columns),
    )
    grouped = list(working.groupby("_invariance_query", sort=False))
    if not grouped:
        raise ValueError("feature invariance requires at least one ranking query")
    feature_report: dict[str, dict[str, float | bool]] = {}
    invariant_in_all = []
    varying_in_any = []
    for feature in feature_columns:
        invariant_count = sum(
            group[feature].nunique(dropna=False) <= 1 for _, group in grouped
        )
        invariant_fraction = invariant_count / len(grouped)
        invariant = invariant_count == len(grouped)
        feature_report[feature] = {
            "invariant_query_count": float(invariant_count),
            "query_count": float(len(grouped)),
            "invariant_query_fraction": float(invariant_fraction),
            "invariant_in_all_queries": invariant,
        }
        (invariant_in_all if invariant else varying_in_any).append(feature)
    return {
        "query_columns": list(query_columns),
        "query_count": len(grouped),
        "invariant_in_all_queries": sorted(invariant_in_all),
        "varying_in_at_least_one_query": sorted(varying_in_any),
        "features": feature_report,
        "interpretation": (
            "additive main effects that are invariant within every query cannot "
            "directly order candidates inside that query"
        ),
    }


def reliability_table(
    frame: pd.DataFrame,
    *,
    target_column: str,
    score_column: str,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Create quantile reliability bins on rows with adjudicated outcomes."""

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    missing = {target_column, score_column} - set(frame)
    if missing:
        raise ValueError(f"reliability columns are missing: {sorted(missing)}")
    clean = frame.dropna(subset=[target_column, score_column]).copy()
    if clean.empty:
        return []
    clean[target_column] = pd.to_numeric(clean[target_column], errors="raise")
    clean[score_column] = pd.to_numeric(clean[score_column], errors="raise")
    if not clean[target_column].isin([0, 1]).all():
        raise ValueError("reliability target must be binary")
    if not clean[score_column].between(0, 1).all():
        raise ValueError("reliability score must be in [0, 1]")
    # Quantile the score itself, never a row-order rank. Duplicate boundaries
    # deliberately reduce the bin count so equal predictions remain together.
    unique_scores = int(clean[score_column].nunique())
    if unique_scores == 1:
        clean["_calibration_bin"] = 0
    else:
        resolved_bins = min(n_bins, unique_scores)
        clean["_calibration_bin"] = pd.qcut(
            clean[score_column],
            q=resolved_bins,
            labels=False,
            duplicates="drop",
        )
    records = []
    for bin_index, group in clean.groupby("_calibration_bin", sort=True):
        records.append(
            {
                "bin": float(bin_index),
                "n": float(len(group)),
                "mean_score": float(group[score_column].mean()),
                "event_fraction": float(group[target_column].mean()),
                "score_minimum": float(group[score_column].min()),
                "score_maximum": float(group[score_column].max()),
            }
        )
    return records


def grouped_ranking_metrics(
    frame: pd.DataFrame,
    *,
    group_column: str,
    target_column: str,
    score_column: str,
    study_column: str | None = None,
    ks: tuple[int, ...] = (5, 10, 20),
    probability_score: bool = True,
) -> dict[str, float]:
    """Compute query-level metrics, macro-averaged within study then across studies."""

    identifier_columns = [group_column]
    if study_column is not None:
        identifier_columns.append(study_column)
    _require_complete_identifiers(
        frame,
        identifier_columns,
        context="ranking",
    )
    candidate = frame.dropna(subset=[score_column]).copy()
    clean = candidate.dropna(subset=[target_column]).copy()
    if clean.empty:
        raise ValueError("no labeled predictions are available")

    target = clean[target_column].to_numpy(dtype=float)
    score = clean[score_column].to_numpy(dtype=float)
    if not np.isfinite(target).all() or not np.isin(target, [0, 1]).all():
        raise ValueError("ranking target must contain only binary values 0 or 1")
    if not np.isfinite(score).all():
        raise ValueError("ranking scores must be finite where present")
    global_average_precision = float(average_precision_score(target, score))
    global_prevalence = float(np.mean(target))
    metrics: dict[str, float] = {
        "average_precision": global_average_precision,
        "global_average_precision": global_average_precision,
        "positive_prevalence": global_prevalence,
        "average_precision_above_prevalence": (
            global_average_precision - global_prevalence
        ),
        "normalized_average_precision": _normalized_average_precision(
            global_average_precision,
            global_prevalence,
        ),
        "n_labeled": float(len(clean)),
        "n_groups": float(clean[group_column].nunique()),
        "n_studies": float(
            candidate[study_column].nunique() if study_column is not None else 1
        ),
    }
    probability_like = bool(
        probability_score
        and np.isfinite(score).all()
        and (score >= 0).all()
        and (score <= 1).all()
    )
    if probability_like:
        metrics["uncalibrated_brier_score"] = float(brier_score_loss(target, score))
        slope, intercept = _calibration_slope_intercept(target, score)
        metrics["calibration_slope"] = slope
        metrics["calibration_intercept"] = intercept
    else:
        metrics["uncalibrated_brier_score"] = float("nan")
        metrics["calibration_slope"] = float("nan")
        metrics["calibration_intercept"] = float("nan")
    grouped = list(clean.groupby(group_column, sort=False))
    candidate_grouped = list(candidate.groupby(group_column, sort=False))
    group_average_precisions: list[tuple[str, float]] = []
    group_prevalences: list[tuple[str, float]] = []
    group_ap_above_prevalence: list[tuple[str, float]] = []
    group_normalized_ap: list[tuple[str, float]] = []
    all_positive_groups = 0
    all_negative_groups = 0
    for _, group in grouped:
        truth = group[target_column].to_numpy(dtype=float)
        study = _study_for_group(group, study_column)
        prevalence = float(np.mean(truth))
        group_prevalences.append((study, prevalence))
        if truth.min() == truth.max():
            if truth[0] == 1:
                all_positive_groups += 1
            else:
                all_negative_groups += 1
            continue
        average_precision = float(
            average_precision_score(
                truth,
                group[score_column].to_numpy(dtype=float),
            )
        )
        group_average_precisions.append((study, average_precision))
        group_ap_above_prevalence.append((study, average_precision - prevalence))
        normalized_ap = _normalized_average_precision(average_precision, prevalence)
        if np.isfinite(normalized_ap):
            group_normalized_ap.append((study, normalized_ap))
    query_ap, study_ap = _nested_macro_mean(group_average_precisions)
    query_prevalence, study_prevalence = _nested_macro_mean(group_prevalences)
    query_excess_ap, study_excess_ap = _nested_macro_mean(group_ap_above_prevalence)
    query_normalized_ap, study_normalized_ap = _nested_macro_mean(group_normalized_ap)
    metrics["query_macro_average_precision"] = query_ap
    metrics["study_macro_average_precision"] = study_ap
    metrics["macro_average_precision"] = study_ap
    metrics["query_macro_positive_prevalence"] = query_prevalence
    metrics["study_macro_positive_prevalence"] = study_prevalence
    metrics["query_macro_average_precision_above_prevalence"] = query_excess_ap
    metrics["study_macro_average_precision_above_prevalence"] = study_excess_ap
    metrics["query_macro_normalized_average_precision"] = query_normalized_ap
    metrics["study_macro_normalized_average_precision"] = study_normalized_ap
    metrics["n_groups_with_both_classes"] = float(len(group_average_precisions))
    metrics["n_candidate_groups"] = float(len(candidate_grouped))
    metrics["n_groups_with_adjudicated_labels"] = float(len(grouped))
    metrics["n_groups_without_adjudicated_labels"] = float(
        len(candidate_grouped) - len(grouped)
    )
    metrics["n_groups_adjudicated_single_class"] = float(
        all_positive_groups + all_negative_groups
    )
    metrics["n_groups_adjudicated_all_positive"] = float(all_positive_groups)
    metrics["n_groups_adjudicated_all_negative"] = float(all_negative_groups)
    informative_studies = {study for study, _ in group_average_precisions}
    adjudicated_studies = {
        _study_for_group(group, study_column) for _, group in grouped
    }
    metrics["n_studies_with_adjudicated_labels"] = float(len(adjudicated_studies))
    metrics["n_studies_with_both_class_queries"] = float(len(informative_studies))
    metrics["n_studies_without_both_class_queries"] = float(
        len(adjudicated_studies - informative_studies)
    )
    metrics["fraction_groups_with_both_classes_of_adjudicated"] = float(
        len(group_average_precisions) / len(grouped)
    )

    documented_informative_groups = 0
    documented_all_positive_groups = 0
    documented_all_negative_groups = 0
    for _, group in candidate_grouped:
        documented_truth = group[target_column].fillna(0).to_numpy(dtype=float)
        if documented_truth.min() == documented_truth.max():
            if documented_truth[0] == 1:
                documented_all_positive_groups += 1
            else:
                documented_all_negative_groups += 1
        else:
            documented_informative_groups += 1
    metrics["n_groups_documented_success_with_both_classes"] = float(
        documented_informative_groups
    )
    metrics["n_groups_documented_success_single_class"] = float(
        documented_all_positive_groups + documented_all_negative_groups
    )
    metrics["n_groups_documented_success_all_positive"] = float(
        documented_all_positive_groups
    )
    metrics["n_groups_documented_success_all_negative"] = float(
        documented_all_negative_groups
    )
    for k in ks:
        precisions: list[tuple[str, float]] = []
        ndcgs: list[tuple[str, float]] = []
        strict_ndcgs: list[tuple[str, float]] = []
        precision_lifts: list[tuple[str, float]] = []
        for _, group in grouped:
            truth = group[target_column].to_numpy(dtype=float)
            score = group[score_column].to_numpy(dtype=float)
            study = _study_for_group(group, study_column)
            # Single-class queries have no within-query ranking resolution.
            # They remain visible in the denominators but do not contribute to
            # inferential ranking macros.
            if truth.min() == truth.max():
                continue
            precision = precision_at_k(truth, score, k)
            if np.isfinite(precision):
                precisions.append((study, precision))
                prevalence = float(np.mean(truth))
                if prevalence > 0:
                    precision_lifts.append((study, precision / prevalence))
            ndcg = float(
                ndcg_score(
                    truth.reshape(1, -1),
                    score.reshape(1, -1),
                    k=min(k, len(truth)),
                    ignore_ties=False,
                )
            )
            ndcgs.append((study, ndcg))
            if k < len(truth):
                strict_ndcgs.append((study, ndcg))
        query_precision, study_precision = _nested_macro_mean(precisions)
        query_ndcg, study_ndcg = _nested_macro_mean(ndcgs)
        query_strict_ndcg, study_strict_ndcg = _nested_macro_mean(strict_ndcgs)
        query_precision_lift, study_precision_lift = _nested_macro_mean(precision_lifts)
        metrics[f"query_macro_adjudicated_precision_at_{k}"] = query_precision
        metrics[f"study_macro_adjudicated_precision_at_{k}"] = study_precision
        metrics[f"precision_at_{k}"] = study_precision
        metrics[f"adjudicated_precision_at_{k}"] = metrics[f"precision_at_{k}"]
        metrics[f"query_macro_adjudicated_precision_lift_at_{k}"] = query_precision_lift
        metrics[f"study_macro_adjudicated_precision_lift_at_{k}"] = study_precision_lift
        metrics[f"adjudicated_precision_lift_at_{k}"] = study_precision_lift
        metrics[f"query_macro_adjudicated_ndcg_at_{k}"] = query_ndcg
        metrics[f"study_macro_adjudicated_ndcg_at_{k}"] = study_ndcg
        metrics[f"ndcg_at_{k}"] = study_ndcg
        metrics[f"adjudicated_ndcg_at_{k}"] = metrics[f"ndcg_at_{k}"]
        metrics[f"query_macro_strict_adjudicated_ndcg_at_{k}"] = query_strict_ndcg
        metrics[f"study_macro_strict_adjudicated_ndcg_at_{k}"] = study_strict_ndcg
        metrics[f"strict_adjudicated_ndcg_at_{k}"] = study_strict_ndcg
        metrics[f"n_groups_adjudicated_precision_at_{k}_defined"] = float(
            len(precisions)
        )
        metrics[f"fraction_groups_adjudicated_precision_at_{k}_defined"] = float(
            len(precisions) / len(grouped)
        )
        metrics[f"n_studies_adjudicated_precision_at_{k}_defined"] = float(
            len({study for study, _ in precisions})
        )
        metrics[f"n_groups_adjudicated_ndcg_at_{k}_defined"] = float(len(ndcgs))
        metrics[f"n_studies_adjudicated_ndcg_at_{k}_defined"] = float(
            len({study for study, _ in ndcgs})
        )
        metrics[f"n_groups_adjudicated_ndcg_at_{k}_unsaturated"] = float(
            len(strict_ndcgs)
        )
        metrics[f"fraction_groups_adjudicated_ndcg_at_{k}_unsaturated"] = float(
            len(strict_ndcgs) / len(grouped)
        )

        documented_capture: list[tuple[str, float]] = []
        documented_lift: list[tuple[str, float]] = []
        documented_recalls: list[tuple[str, float]] = []
        documented_ndcgs: list[tuple[str, float]] = []
        strict_documented_ndcgs: list[tuple[str, float]] = []
        for _, group in candidate_grouped:
            study = _study_for_group(group, study_column)
            # This is deliberately named "documented" rather than "observed
            # success": missing outcomes are counted as not documented, not as
            # biological failures.
            documented_truth = group[target_column].fillna(0).to_numpy(dtype=float)
            score = group[score_column].to_numpy(dtype=float)
            if documented_truth.min() == documented_truth.max():
                continue
            top_n = min(k, len(documented_truth))
            selected_relevance = _expected_relevance_at_k(
                documented_truth,
                score,
                top_n,
            )
            documented_positive_n = float(documented_truth.sum())
            if k < len(documented_truth):
                capture = float(selected_relevance / top_n)
                documented_capture.append((study, capture))
                prevalence = documented_positive_n / len(documented_truth)
                if prevalence > 0:
                    documented_lift.append((study, capture / prevalence))
            if documented_positive_n > 0:
                if k < len(documented_truth):
                    documented_recalls.append(
                        (
                            study,
                            float(selected_relevance / documented_positive_n),
                        )
                    )
                documented_ndcg = float(
                    ndcg_score(
                        documented_truth.reshape(1, -1),
                        score.reshape(1, -1),
                        k=top_n,
                        ignore_ties=False,
                    )
                )
                documented_ndcgs.append((study, documented_ndcg))
                if k < len(documented_truth):
                    strict_documented_ndcgs.append((study, documented_ndcg))
        query_capture, study_capture = _nested_macro_mean(documented_capture)
        query_lift, study_lift = _nested_macro_mean(documented_lift)
        query_recall, study_recall = _nested_macro_mean(documented_recalls)
        query_documented_ndcg, study_documented_ndcg = _nested_macro_mean(
            documented_ndcgs
        )
        query_strict_documented_ndcg, study_strict_documented_ndcg = _nested_macro_mean(
            strict_documented_ndcgs
        )
        metrics[f"query_macro_documented_success_capture_at_{k}"] = query_capture
        metrics[f"study_macro_documented_success_capture_at_{k}"] = study_capture
        metrics[f"documented_success_capture_at_{k}"] = study_capture
        metrics[f"query_macro_documented_success_lift_at_{k}"] = query_lift
        metrics[f"study_macro_documented_success_lift_at_{k}"] = study_lift
        metrics[f"documented_success_lift_at_{k}"] = study_lift
        metrics[f"query_macro_documented_success_recall_at_{k}"] = query_recall
        metrics[f"study_macro_documented_success_recall_at_{k}"] = study_recall
        metrics[f"documented_success_recall_at_{k}"] = study_recall
        metrics[f"query_macro_documented_success_ndcg_at_{k}"] = query_documented_ndcg
        metrics[f"study_macro_documented_success_ndcg_at_{k}"] = study_documented_ndcg
        metrics[f"documented_success_ndcg_at_{k}"] = study_documented_ndcg
        metrics[f"query_macro_strict_documented_success_ndcg_at_{k}"] = (
            query_strict_documented_ndcg
        )
        metrics[f"study_macro_strict_documented_success_ndcg_at_{k}"] = (
            study_strict_documented_ndcg
        )
        metrics[f"n_groups_documented_success_capture_at_{k}_defined"] = float(
            len(documented_capture)
        )
        metrics[f"n_studies_documented_success_capture_at_{k}_defined"] = float(
            len({study for study, _ in documented_capture})
        )
        metrics[f"fraction_groups_documented_success_capture_at_{k}_defined"] = float(
            len(documented_capture) / len(candidate_grouped)
        )
        metrics[f"n_groups_documented_success_ndcg_at_{k}_defined"] = float(
            len(documented_ndcgs)
        )
        metrics[f"n_studies_documented_success_ndcg_at_{k}_defined"] = float(
            len({study for study, _ in documented_ndcgs})
        )
        metrics[f"n_groups_documented_success_ndcg_at_{k}_unsaturated"] = float(
            len(strict_documented_ndcgs)
        )
    return metrics


def study_cluster_bootstrap_intervals(
    frame: pd.DataFrame,
    *,
    study_column: str,
    metric_study_column: str | None = None,
    screen_column: str,
    target_column: str,
    score_column: str,
    query_columns: tuple[str, ...] | None = None,
    n_bootstrap: int = 1_000,
    confidence: float = 0.95,
    min_effective_fraction: float = 0.8,
    random_state: int = 17,
) -> dict[str, dict[str, float]]:
    """Cluster-bootstrap intervals with a separate study-macro denominator."""

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    if not 0 < min_effective_fraction <= 1:
        raise ValueError("min_effective_fraction must be in the interval (0, 1]")
    metric_study_column = metric_study_column or study_column
    required_columns = {
        study_column,
        metric_study_column,
        target_column,
        score_column,
    }
    missing_columns = required_columns - set(frame)
    if missing_columns:
        raise ValueError(
            f"cluster bootstrap columns are missing: {sorted(missing_columns)}"
        )
    resolved_query_columns = list(query_columns or (screen_column,))
    if query_columns is None:
        for column in ("contrast_id", "phenotype_direction"):
            if column in frame.columns and column not in resolved_query_columns:
                resolved_query_columns.append(column)
    _require_complete_identifiers(
        frame,
        [study_column, metric_study_column, *resolved_query_columns],
        context="cluster bootstrap",
    )
    working = frame.copy()
    numeric_target = pd.to_numeric(working[target_column], errors="coerce")
    invalid_target = working[target_column].notna() & (
        numeric_target.isna() | ~numeric_target.isin([0, 1])
    )
    if invalid_target.any():
        raise ValueError("cluster bootstrap target must be binary where present")
    numeric_score = pd.to_numeric(working[score_column], errors="coerce")
    invalid_score = working[score_column].notna() & (
        numeric_score.isna() | ~np.isfinite(numeric_score)
    )
    if invalid_score.any():
        raise ValueError("cluster bootstrap score must be finite where present")
    working[target_column] = numeric_target
    working[score_column] = numeric_score
    working["_bootstrap_query_base"] = _composite_query_keys(
        working,
        resolved_query_columns,
    )
    if (
        working.groupby("_bootstrap_query_base", sort=False)[metric_study_column]
        .nunique()
        .gt(1)
        .any()
    ):
        raise ValueError("each bootstrap ranking query must be nested within one study")
    if (
        working.groupby(metric_study_column, sort=False)[study_column]
        .nunique()
        .gt(1)
        .any()
    ):
        raise ValueError(
            "each metric study must be nested within one bootstrap cluster"
        )
    studies = working[study_column].unique()
    if len(studies) < 2:
        raise ValueError("at least two studies are required for cluster bootstrap")
    eligible = working.dropna(
        subset=[
            target_column,
            score_column,
            *resolved_query_columns,
        ]
    )
    target_values = set(
        pd.to_numeric(eligible[target_column], errors="coerce").dropna().unique()
    )
    if target_values != {0, 1}:
        raise ValueError(
            "cluster bootstrap requires globally observed binary target classes {0, 1}"
        )

    rng = np.random.default_rng(random_state)
    draws: dict[str, list[float]] = {}
    valid_draws = 0
    skipped_draws = 0
    attempts = 0
    max_attempts = max(n_bootstrap * 20, n_bootstrap + 100)
    while valid_draws < n_bootstrap and attempts < max_attempts:
        attempts += 1
        sampled = rng.choice(studies, size=len(studies), replace=True)
        blocks = []
        for draw_index, study in enumerate(sampled):
            block = working.loc[working[study_column] == study].copy()
            block["_bootstrap_query"] = _composite_query_keys(
                block,
                resolved_query_columns,
                namespace=("bootstrap_draw", draw_index),
            )
            block["_bootstrap_study"] = str(study) + f"__draw{draw_index}"
            block["_bootstrap_metric_study"] = _composite_query_keys(
                block,
                [metric_study_column],
                namespace=("bootstrap_metric_study", draw_index),
            )
            blocks.append(block)
        bootstrap_frame = pd.concat(blocks, ignore_index=True)
        draw_targets = bootstrap_frame.dropna(
            subset=[
                target_column,
                score_column,
                "_bootstrap_query",
            ]
        )[target_column]
        if set(pd.to_numeric(draw_targets, errors="coerce").dropna().unique()) != {
            0,
            1,
        }:
            skipped_draws += 1
            continue
        try:
            metrics = grouped_ranking_metrics(
                bootstrap_frame,
                group_column="_bootstrap_query",
                target_column=target_column,
                score_column=score_column,
                study_column="_bootstrap_metric_study",
            )
        except ValueError:
            skipped_draws += 1
            continue
        valid_draws += 1
        for name, value in metrics.items():
            if name.startswith("n_"):
                continue
            draws.setdefault(name, [])
            if np.isnan(value):
                continue
            draws.setdefault(name, []).append(value)
    if valid_draws == 0:
        raise ValueError("no bootstrap draw contained adjudicated predictions")

    alpha = (1.0 - confidence) / 2.0
    intervals: dict[str, dict[str, float]] = {}
    minimum_effective_draws = max(
        2,
        int(np.ceil(n_bootstrap * min_effective_fraction)),
    )
    for name, values in draws.items():
        effective_draws = len(values)
        if effective_draws < minimum_effective_draws:
            intervals[name] = {
                "effective_draws": float(effective_draws),
                "interval_available": 0.0,
            }
            continue
        array = np.asarray(values, dtype=float)
        intervals[name] = {
            "effective_draws": float(effective_draws),
            "interval_available": 1.0,
            "median": float(np.median(array)),
            "lower": float(np.quantile(array, alpha)),
            "upper": float(np.quantile(array, 1.0 - alpha)),
        }
    intervals["_diagnostics"] = {
        "requested_draws": float(n_bootstrap),
        "valid_draws": float(valid_draws),
        "skipped_draws": float(skipped_draws),
        "attempts": float(attempts),
        "minimum_effective_draws": float(minimum_effective_draws),
    }
    return intervals


def paired_cluster_bootstrap_difference(
    frame: pd.DataFrame,
    *,
    cluster_column: str,
    metric_study_column: str | None = None,
    screen_column: str,
    target_column: str,
    candidate_score_column: str,
    comparator_score_column: str,
    metric_name: str,
    query_columns: tuple[str, ...] | None = None,
    n_bootstrap: int = 1_000,
    confidence: float = 0.95,
    min_effective_fraction: float = 0.8,
    random_state: int = 17,
) -> dict[str, float | str | None]:
    """Paired cluster-bootstrap interval for candidate-minus-comparator delta.

    Both scores are evaluated on the same resampled source-family roster.  The
    interval conditions on the supplied scores; it does not refit either model.
    """

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    if not 0 < min_effective_fraction <= 1:
        raise ValueError("min_effective_fraction must be in the interval (0, 1]")
    metric_study_column = metric_study_column or cluster_column
    resolved_query_columns = list(query_columns or (screen_column,))
    if query_columns is None:
        for column in ("contrast_id", "phenotype_direction"):
            if column in frame and column not in resolved_query_columns:
                resolved_query_columns.append(column)
    required = {
        cluster_column,
        metric_study_column,
        target_column,
        candidate_score_column,
        comparator_score_column,
        *resolved_query_columns,
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"paired bootstrap columns are missing: {sorted(missing)}")
    if frame.empty:
        raise ValueError("paired bootstrap requires a nonempty candidate roster")
    _require_complete_identifiers(
        frame,
        [cluster_column, metric_study_column, *resolved_query_columns],
        context="paired bootstrap",
    )

    working = frame.copy()
    numeric_target = pd.to_numeric(working[target_column], errors="coerce")
    invalid_target = working[target_column].notna() & (
        numeric_target.isna() | ~numeric_target.isin([0, 1])
    )
    if invalid_target.any():
        raise ValueError("paired bootstrap target must be binary where present")
    working[target_column] = numeric_target

    candidate_scores = pd.to_numeric(
        working[candidate_score_column],
        errors="coerce",
    )
    comparator_scores = pd.to_numeric(
        working[comparator_score_column],
        errors="coerce",
    )
    candidate_finite = candidate_scores.notna() & np.isfinite(candidate_scores)
    comparator_finite = comparator_scores.notna() & np.isfinite(comparator_scores)
    common_finite = candidate_finite & comparator_finite
    candidate_coverage = float(candidate_finite.mean())
    comparator_coverage = float(comparator_finite.mean())
    common_coverage = float(common_finite.mean())
    if not common_finite.all():
        raise ValueError(
            "paired bootstrap requires finite candidate and comparator scores on "
            "the complete common roster; "
            f"candidate_coverage={candidate_coverage:.6f}, "
            f"comparator_coverage={comparator_coverage:.6f}, "
            f"common_coverage={common_coverage:.6f}"
        )
    working[candidate_score_column] = candidate_scores
    working[comparator_score_column] = comparator_scores
    working["_paired_query_base"] = _composite_query_keys(
        working,
        resolved_query_columns,
    )
    if (
        working.groupby("_paired_query_base", sort=False)[metric_study_column]
        .nunique()
        .gt(1)
        .any()
    ):
        raise ValueError("each paired ranking query must be nested within one study")
    if (
        working.groupby(metric_study_column, sort=False)[cluster_column]
        .nunique()
        .gt(1)
        .any()
    ):
        raise ValueError(
            "each metric study must be nested within one bootstrap cluster"
        )
    clusters = working[cluster_column].unique()
    if len(clusters) < 2:
        raise ValueError("at least two clusters are required for paired bootstrap")

    point_frame = working.copy()
    point_frame["_paired_query"] = point_frame["_paired_query_base"]
    candidate_point = grouped_ranking_metrics(
        point_frame,
        group_column="_paired_query",
        target_column=target_column,
        score_column=candidate_score_column,
        study_column=metric_study_column,
    ).get(metric_name)
    comparator_point = grouped_ranking_metrics(
        point_frame,
        group_column="_paired_query",
        target_column=target_column,
        score_column=comparator_score_column,
        study_column=metric_study_column,
        probability_score=False,
    ).get(metric_name)
    if candidate_point is None or comparator_point is None:
        raise ValueError(f"unknown paired bootstrap metric: {metric_name}")
    if not np.isfinite(candidate_point) or not np.isfinite(comparator_point):
        raise ValueError(
            f"paired bootstrap metric is undefined on the common roster: {metric_name}"
        )
    point_difference = float(candidate_point - comparator_point)

    rng = np.random.default_rng(random_state)
    differences: list[float] = []
    skipped_draws = 0
    attempts = 0
    max_attempts = max(n_bootstrap * 20, n_bootstrap + 100)
    while len(differences) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        blocks = []
        for draw_index, cluster in enumerate(sampled):
            block = working.loc[working[cluster_column] == cluster].copy()
            block["_paired_query"] = _composite_query_keys(
                block,
                resolved_query_columns,
                namespace=("paired_bootstrap_draw", draw_index),
            )
            block["_paired_cluster"] = f"{cluster}__draw{draw_index}"
            block["_paired_metric_study"] = _composite_query_keys(
                block,
                [metric_study_column],
                namespace=("paired_metric_study", draw_index),
            )
            blocks.append(block)
        bootstrap_frame = pd.concat(blocks, ignore_index=True)
        draw_targets = pd.to_numeric(
            bootstrap_frame[target_column], errors="coerce"
        ).dropna()
        if set(draw_targets.unique()) != {0, 1}:
            skipped_draws += 1
            continue
        candidate_metrics = grouped_ranking_metrics(
            bootstrap_frame,
            group_column="_paired_query",
            target_column=target_column,
            score_column=candidate_score_column,
            study_column="_paired_metric_study",
        )
        comparator_metrics = grouped_ranking_metrics(
            bootstrap_frame,
            group_column="_paired_query",
            target_column=target_column,
            score_column=comparator_score_column,
            study_column="_paired_metric_study",
            probability_score=False,
        )
        candidate_value = candidate_metrics.get(metric_name, float("nan"))
        comparator_value = comparator_metrics.get(metric_name, float("nan"))
        if not np.isfinite(candidate_value) or not np.isfinite(comparator_value):
            skipped_draws += 1
            continue
        differences.append(float(candidate_value - comparator_value))

    minimum_effective_draws = max(
        2,
        int(np.ceil(n_bootstrap * min_effective_fraction)),
    )
    result: dict[str, float | str | None] = {
        "metric": metric_name,
        "estimand": "candidate_minus_comparator",
        "point_difference": point_difference,
        "candidate_point": float(candidate_point),
        "comparator_point": float(comparator_point),
        "candidate_score_coverage": candidate_coverage,
        "comparator_score_coverage": comparator_coverage,
        "common_score_coverage": common_coverage,
        "common_score_rows": float(common_finite.sum()),
        "effective_draws": float(len(differences)),
        "requested_draws": float(n_bootstrap),
        "skipped_draws": float(skipped_draws),
        "attempts": float(attempts),
        "interval_available": float(len(differences) >= minimum_effective_draws),
        "uncertainty_boundary": "conditions_on_supplied_scores_without_refitting",
    }
    if len(differences) >= minimum_effective_draws:
        alpha = (1.0 - confidence) / 2.0
        values = np.asarray(differences, dtype=float)
        result.update(
            {
                "bootstrap_median_difference": float(np.median(values)),
                "lower": float(np.quantile(values, alpha)),
                "upper": float(np.quantile(values, 1.0 - alpha)),
            }
        )
    return result
