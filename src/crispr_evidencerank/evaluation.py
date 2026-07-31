"""Ranking and calibration metrics for grouped screen evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, ndcg_score


def _composite_query_keys(
    frame: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    *,
    namespace: object | None = None,
) -> pd.Series:
    """Build collision-safe tuple keys for ranking queries."""

    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"query columns are missing: {sorted(missing)}")
    row_keys = list(frame[list(columns)].itertuples(index=False, name=None))
    if namespace is not None:
        row_keys = [(namespace, key) for key in row_keys]
    return pd.Series(row_keys, index=frame.index, dtype=object)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return float("nan")
    selected = np.argsort(-y_score)[: min(k, len(y_true))]
    return float(np.mean(y_true[selected]))


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


def grouped_ranking_metrics(
    frame: pd.DataFrame,
    *,
    group_column: str,
    target_column: str,
    score_column: str,
    study_column: str | None = None,
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    """Compute query-level metrics, macro-averaged within study then across studies."""

    candidate = frame.dropna(subset=[score_column, group_column]).copy()
    clean = candidate.dropna(subset=[target_column]).copy()
    if clean.empty:
        raise ValueError("no labeled predictions are available")

    global_average_precision = float(
        average_precision_score(clean[target_column], clean[score_column])
    )
    metrics: dict[str, float] = {
        "average_precision": global_average_precision,
        "global_average_precision": global_average_precision,
        "uncalibrated_brier_score": float(
            brier_score_loss(clean[target_column], clean[score_column])
        ),
        "n_labeled": float(len(clean)),
        "n_groups": float(clean[group_column].nunique()),
        "n_studies": float(
            candidate[study_column].nunique() if study_column is not None else 1
        ),
    }
    grouped = list(clean.groupby(group_column, sort=False))
    candidate_grouped = list(candidate.groupby(group_column, sort=False))
    group_average_precisions: list[tuple[str, float]] = []
    for _, group in grouped:
        truth = group[target_column].to_numpy(dtype=float)
        if truth.min() == truth.max():
            continue
        group_average_precisions.append(
            (
                _study_for_group(group, study_column),
                average_precision_score(
                    truth,
                    group[score_column].to_numpy(dtype=float),
                ),
            )
        )
    query_ap, study_ap = _nested_macro_mean(group_average_precisions)
    metrics["query_macro_average_precision"] = query_ap
    metrics["study_macro_average_precision"] = study_ap
    metrics["macro_average_precision"] = study_ap
    metrics["n_groups_with_both_classes"] = float(len(group_average_precisions))
    for k in ks:
        precisions: list[tuple[str, float]] = []
        ndcgs: list[tuple[str, float]] = []
        for _, group in grouped:
            truth = group[target_column].to_numpy(dtype=float)
            score = group[score_column].to_numpy(dtype=float)
            study = _study_for_group(group, study_column)
            precisions.append((study, precision_at_k(truth, score, k)))
            if truth.sum() > 0:
                ndcgs.append(
                    (
                        study,
                        ndcg_score(
                            truth.reshape(1, -1),
                            score.reshape(1, -1),
                            k=min(k, len(truth)),
                        ),
                    )
                )
        query_precision, study_precision = _nested_macro_mean(precisions)
        query_ndcg, study_ndcg = _nested_macro_mean(ndcgs)
        metrics[f"query_macro_adjudicated_precision_at_{k}"] = query_precision
        metrics[f"study_macro_adjudicated_precision_at_{k}"] = study_precision
        metrics[f"precision_at_{k}"] = study_precision
        metrics[f"adjudicated_precision_at_{k}"] = metrics[f"precision_at_{k}"]
        metrics[f"query_macro_adjudicated_ndcg_at_{k}"] = query_ndcg
        metrics[f"study_macro_adjudicated_ndcg_at_{k}"] = study_ndcg
        metrics[f"ndcg_at_{k}"] = study_ndcg
        metrics[f"adjudicated_ndcg_at_{k}"] = metrics[f"ndcg_at_{k}"]

        observed_yields: list[tuple[str, float]] = []
        observed_recalls: list[tuple[str, float]] = []
        observed_ndcgs: list[tuple[str, float]] = []
        for _, group in candidate_grouped:
            study = _study_for_group(group, study_column)
            observed_truth = group[target_column].fillna(0).to_numpy(dtype=float)
            score = group[score_column].to_numpy(dtype=float)
            top_n = min(k, len(observed_truth))
            selected = np.argsort(-score)[:top_n]
            observed_yields.append(
                (
                    study,
                    float(observed_truth[selected].sum() / top_n),
                )
            )
            if observed_truth.sum() > 0:
                observed_recalls.append(
                    (
                        study,
                        float(observed_truth[selected].sum() / observed_truth.sum()),
                    )
                )
                observed_ndcgs.append(
                    (
                        study,
                        ndcg_score(
                            observed_truth.reshape(1, -1),
                            score.reshape(1, -1),
                            k=top_n,
                        ),
                    )
                )
        query_yield, study_yield = _nested_macro_mean(observed_yields)
        query_recall, study_recall = _nested_macro_mean(observed_recalls)
        query_observed_ndcg, study_observed_ndcg = _nested_macro_mean(observed_ndcgs)
        metrics[f"query_macro_observed_success_yield_at_{k}"] = query_yield
        metrics[f"study_macro_observed_success_yield_at_{k}"] = study_yield
        metrics[f"observed_success_yield_at_{k}"] = study_yield
        metrics[f"query_macro_observed_recall_at_{k}"] = query_recall
        metrics[f"study_macro_observed_recall_at_{k}"] = study_recall
        metrics[f"observed_recall_at_{k}"] = study_recall
        metrics[f"query_macro_observed_ndcg_at_{k}"] = query_observed_ndcg
        metrics[f"study_macro_observed_ndcg_at_{k}"] = study_observed_ndcg
        metrics[f"observed_ndcg_at_{k}"] = study_observed_ndcg
    return metrics


def study_cluster_bootstrap_intervals(
    frame: pd.DataFrame,
    *,
    study_column: str,
    screen_column: str,
    target_column: str,
    score_column: str,
    query_columns: tuple[str, ...] | None = None,
    n_bootstrap: int = 1_000,
    confidence: float = 0.95,
    min_effective_fraction: float = 0.8,
    random_state: int = 17,
) -> dict[str, dict[str, float]]:
    """Study-cluster bootstrap intervals for grouped ranking metrics."""

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    if not 0 < min_effective_fraction <= 1:
        raise ValueError("min_effective_fraction must be in the interval (0, 1]")
    studies = frame[study_column].dropna().unique()
    if len(studies) < 2:
        raise ValueError("at least two studies are required for cluster bootstrap")
    resolved_query_columns = list(query_columns or (screen_column,))
    if query_columns is None:
        for column in ("contrast_id", "phenotype_direction"):
            if column in frame.columns and column not in resolved_query_columns:
                resolved_query_columns.append(column)
    missing_query_columns = set(resolved_query_columns) - set(frame.columns)
    if missing_query_columns:
        raise ValueError(
            f"bootstrap query columns are missing: {sorted(missing_query_columns)}"
        )
    eligible = frame.dropna(
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
            block = frame.loc[frame[study_column] == study].copy()
            block["_bootstrap_query"] = _composite_query_keys(
                block,
                resolved_query_columns,
                namespace=("bootstrap_draw", draw_index),
            )
            block["_bootstrap_study"] = str(study) + f"__draw{draw_index}"
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
                study_column="_bootstrap_study",
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
