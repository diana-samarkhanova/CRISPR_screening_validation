"""Leakage-conscious fixed-score baselines for grouped ranking benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluation import (
    _composite_query_keys,
    grouped_ranking_metrics,
    paired_cluster_bootstrap_difference,
)


@dataclass(frozen=True)
class FixedBaselineSpec:
    """One baseline whose score is supplied independently of validation labels."""

    name: str
    column: str
    higher_is_better: bool
    interpretation: str


FIXED_BASELINES = (
    FixedBaselineSpec(
        name="absolute_effect",
        column="absolute_median_guide_lfc",
        higher_is_better=True,
        interpretation="absolute count-derived median guide LFC",
    ),
    FixedBaselineSpec(
        name="mageck_rank",
        column="mageck_direction_rank",
        higher_is_better=False,
        interpretation="declared phenotype-tail MAGeCK rank",
    ),
    FixedBaselineSpec(
        name="mageck_fdr",
        column="mageck_direction_fdr",
        higher_is_better=False,
        interpretation="declared phenotype-tail MAGeCK FDR",
    ),
    FixedBaselineSpec(
        name="drugz_normz",
        column="drugz_direction_aligned_normz",
        higher_is_better=True,
        interpretation="drugZ normZ aligned to the candidate phenotype direction",
    ),
)


def adapt_repo_native_mageck_baseline(
    frame: pd.DataFrame,
    *,
    positive_tail_means: str,
    mageck_version: str,
    source_locator: str,
    source_sha256: str,
) -> pd.DataFrame:
    """Attest repo-native MAGeCK tails before exposing baseline columns.

    ``rank-screen`` preserves native positive and negative MAGeCK tails.  This
    adapter requires the caller to attest their biological mapping and to pin
    the upstream summary by version, locator, and checksum.  It never guesses a
    phenotype direction from a score sign or column name.
    """

    if positive_tail_means not in {"resistance", "sensitization"}:
        raise ValueError(
            "positive_tail_means must explicitly attest 'resistance' or 'sensitization'"
        )
    for name, value in (
        ("mageck_version", mageck_version),
        ("source_locator", source_locator),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be non-empty MAGeCK provenance metadata")
    if not (
        isinstance(source_sha256, str)
        and len(source_sha256) == 64
        and source_sha256 == source_sha256.lower()
        and all(character in "0123456789abcdef" for character in source_sha256)
    ):
        raise ValueError("source_sha256 must be a lowercase SHA-256 checksum")

    required = {"analysis_tail", "phenotype_direction"}
    missing = required - set(frame)
    if missing:
        raise ValueError(
            f"repo-native MAGeCK frame is missing columns: {sorted(missing)}"
        )
    score_columns = [
        column for column in ("mageck_rank", "mageck_fdr") if column in frame
    ]
    if not score_columns:
        raise ValueError("repo-native MAGeCK frame requires mageck_rank or mageck_fdr")

    tails = frame["analysis_tail"].astype("string")
    allowed_tails = {"mageck_pos", "mageck_neg"}
    invalid_tails = sorted(set(tails.dropna()) - allowed_tails)
    if tails.isna().any() or invalid_tails:
        raise ValueError(
            "repo-native MAGeCK analysis_tail must contain only mageck_pos or "
            f"mageck_neg; found {invalid_tails}"
        )
    negative_tail_means = (
        "sensitization" if positive_tail_means == "resistance" else "resistance"
    )
    expected_direction = tails.map(
        {
            "mageck_pos": positive_tail_means,
            "mageck_neg": negative_tail_means,
        }
    )
    observed_direction = (
        frame["phenotype_direction"].astype("string").str.strip().str.lower()
    )
    mismatched = (
        observed_direction.isna()
        | expected_direction.isna()
        | observed_direction.ne(expected_direction).fillna(True)
    )
    if mismatched.any():
        rows = frame.index[mismatched].tolist()[:5]
        raise ValueError(
            "repo-native MAGeCK phenotype_direction contradicts the explicit "
            f"tail attestation at row indices {rows}"
        )

    result = frame.copy()
    if "mageck_rank" in result:
        result["mageck_direction_rank"] = result["mageck_rank"]
    if "mageck_fdr" in result:
        result["mageck_direction_fdr"] = result["mageck_fdr"]
    for spec in FIXED_BASELINES:
        if spec.name.startswith("mageck_") and spec.column in result:
            _validated_baseline_numeric(result, spec)
    result["mageck_positive_tail_means_attested"] = positive_tail_means
    result["mageck_source_method"] = "MAGeCK-RRA"
    result["mageck_source_version"] = mageck_version.strip()
    result["mageck_source_locator"] = source_locator.strip()
    result["mageck_source_sha256"] = source_sha256
    return result


def adapt_direction_aligned_drugz_baseline(
    frame: pd.DataFrame,
    *,
    positive_normz_means: str,
    drugz_version: str,
    source_locator: str,
    source_sha256: str,
) -> pd.DataFrame:
    """Create a direction-aligned drugZ score only from explicit semantics."""

    if positive_normz_means not in {"resistance", "sensitization"}:
        raise ValueError(
            "positive_normz_means must explicitly attest 'resistance' or "
            "'sensitization'"
        )
    for name, value in (
        ("drugz_version", drugz_version),
        ("source_locator", source_locator),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be non-empty drugZ provenance metadata")
    if not (
        isinstance(source_sha256, str)
        and len(source_sha256) == 64
        and source_sha256 == source_sha256.lower()
        and all(character in "0123456789abcdef" for character in source_sha256)
    ):
        raise ValueError("source_sha256 must be a lowercase SHA-256 checksum")
    required = {"phenotype_direction", "drugz_normz"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"drugZ frame is missing columns: {sorted(missing)}")
    directions = frame["phenotype_direction"].astype("string").str.strip().str.lower()
    invalid = sorted(set(directions.dropna()) - {"resistance", "sensitization"})
    if directions.isna().any() or invalid:
        raise ValueError(
            "drugZ phenotype_direction must be resistance or sensitization; "
            f"found {invalid}"
        )
    if (
        frame["drugz_normz"]
        .map(lambda value: isinstance(value, (bool, np.bool_)))
        .any()
    ):
        raise ValueError("drugz_normz must contain numeric, not boolean values")
    normz = pd.to_numeric(frame["drugz_normz"], errors="coerce")
    if normz.isna().any() or not np.isfinite(normz.to_numpy(dtype=float)).all():
        raise ValueError("drugz_normz must contain complete finite numeric values")
    multiplier = directions.map(
        {
            positive_normz_means: 1.0,
            (
                "sensitization"
                if positive_normz_means == "resistance"
                else "resistance"
            ): -1.0,
        }
    )
    result = frame.copy()
    result["drugz_direction_aligned_normz"] = normz * multiplier
    result["drugz_positive_normz_means_attested"] = positive_normz_means
    result["drugz_source_version"] = drugz_version.strip()
    result["drugz_source_locator"] = source_locator.strip()
    result["drugz_source_sha256"] = source_sha256
    return result


def _validate_direction_attestation(
    frame: pd.DataFrame,
    spec: FixedBaselineSpec,
) -> None:
    # Provenance is constant within one ranking query, but an external benchmark
    # naturally combines different upstream summaries and software versions.
    if "_baseline_query_id" in frame and frame["_baseline_query_id"].nunique() > 1:
        for _, query in frame.groupby("_baseline_query_id", sort=False):
            _validate_direction_attestation(query, spec)
        return
    if spec.name.startswith("mageck_"):
        required = {
            "analysis_tail",
            "phenotype_direction",
            "mageck_positive_tail_means_attested",
            "mageck_source_method",
            "mageck_source_version",
            "mageck_source_locator",
            "mageck_source_sha256",
        }
        missing = required - set(frame)
        if missing:
            raise ValueError(
                "MAGeCK baseline lacks direction/provenance attestation: "
                f"{sorted(missing)}"
            )
        for column in (
            "mageck_positive_tail_means_attested",
            "mageck_source_method",
            "mageck_source_version",
            "mageck_source_locator",
            "mageck_source_sha256",
        ):
            values = frame[column].astype("string").str.strip()
            if values.isna().any() or values.eq("").any() or values.nunique() != 1:
                raise ValueError(
                    f"MAGeCK baseline requires one complete value for {column}"
                )
        meanings = set(frame["mageck_positive_tail_means_attested"].dropna())
        if len(meanings) != 1:
            raise ValueError("MAGeCK baseline requires one attested tail meaning")
        meaning = str(next(iter(meanings)))
        adapted = adapt_repo_native_mageck_baseline(
            frame,
            positive_tail_means=meaning,
            mageck_version=str(frame["mageck_source_version"].iloc[0]),
            source_locator=str(frame["mageck_source_locator"].iloc[0]),
            source_sha256=str(frame["mageck_source_sha256"].iloc[0]),
        )
        if not adapted[spec.column].equals(frame[spec.column]):
            raise ValueError("MAGeCK direction baseline differs from its native score")
        if not frame["mageck_source_method"].eq("MAGeCK-RRA").all():
            raise ValueError("MAGeCK source method attestation is inconsistent")
    elif spec.name == "drugz_normz":
        required = {
            "phenotype_direction",
            "drugz_normz",
            "drugz_positive_normz_means_attested",
            "drugz_source_version",
            "drugz_source_locator",
            "drugz_source_sha256",
        }
        missing = required - set(frame)
        if missing:
            raise ValueError(
                "drugZ baseline lacks direction/provenance attestation: "
                f"{sorted(missing)}"
            )
        for column in (
            "drugz_positive_normz_means_attested",
            "drugz_source_version",
            "drugz_source_locator",
            "drugz_source_sha256",
        ):
            values = frame[column].astype("string").str.strip()
            if values.isna().any() or values.eq("").any() or values.nunique() != 1:
                raise ValueError(
                    f"drugZ baseline requires one complete value for {column}"
                )
        meanings = set(frame["drugz_positive_normz_means_attested"].dropna())
        if len(meanings) != 1:
            raise ValueError("drugZ baseline requires one attested normZ meaning")
        meaning = str(next(iter(meanings)))
        adapted = adapt_direction_aligned_drugz_baseline(
            frame,
            positive_normz_means=meaning,
            drugz_version=str(frame["drugz_source_version"].iloc[0]),
            source_locator=str(frame["drugz_source_locator"].iloc[0]),
            source_sha256=str(frame["drugz_source_sha256"].iloc[0]),
        )
        if not np.allclose(
            adapted[spec.column].to_numpy(dtype=float),
            pd.to_numeric(frame[spec.column], errors="coerce").to_numpy(dtype=float),
            rtol=0,
            atol=0,
            equal_nan=False,
        ):
            raise ValueError("drugZ aligned baseline contradicts its attestation")


def _validated_baseline_numeric(
    frame: pd.DataFrame,
    spec: FixedBaselineSpec,
) -> pd.Series:
    source = frame[spec.column]
    if source.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError(f"{spec.name} baseline values must be numeric, not boolean")
    numeric = pd.to_numeric(source, errors="coerce")
    nonnumeric = source.notna() & numeric.isna()
    if nonnumeric.any():
        rows = frame.index[nonnumeric].tolist()[:5]
        raise ValueError(
            f"{spec.name} baseline contains non-numeric values at row indices {rows}"
        )
    observed = numeric.dropna().to_numpy(dtype=float)
    if not np.isfinite(observed).all():
        raise ValueError(f"{spec.name} baseline values must be finite")
    if spec.name == "absolute_effect" and (observed < 0).any():
        raise ValueError("absolute_effect baseline values must be non-negative")
    if spec.name == "mageck_rank" and (
        (observed < 1).any() or not np.equal(observed, np.floor(observed)).all()
    ):
        raise ValueError("mageck_rank baseline values must be positive integers")
    if spec.name == "mageck_fdr" and ((observed < 0).any() or (observed > 1).any()):
        raise ValueError("mageck_fdr baseline values must lie in [0, 1]")
    return numeric


def _metric_mean(records: list[dict[str, float]]) -> dict[str, float]:
    names = sorted(set.intersection(*(set(record) for record in records)))
    result: dict[str, float] = {}
    for name in names:
        values = np.asarray([record[name] for record in records], dtype=float)
        finite = values[np.isfinite(values)]
        result[name] = float(np.mean(finite)) if len(finite) else float("nan")
    return result


def _seed_summary(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(finite):
        return {
            "effective_seeds": 0,
            "mean": None,
            "sample_sd": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "effective_seeds": int(len(finite)),
        "mean": float(np.mean(finite)),
        "sample_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def evaluate_fixed_score_baselines(
    frame: pd.DataFrame,
    *,
    screen_column: str,
    target_column: str,
    study_column: str,
    random_seeds: tuple[int, ...] = tuple(range(100)),
    query_columns: tuple[str, ...] | None = None,
    ks: tuple[int, ...] = (5, 10, 20),
    candidate_score_column: str | None = None,
    cluster_column: str | None = None,
    paired_metric: str = "study_macro_normalized_average_precision",
    paired_bootstrap_repetitions: int = 0,
    paired_bootstrap_confidence: float = 0.95,
    paired_bootstrap_min_effective_fraction: float = 0.8,
    paired_bootstrap_random_state: int = 17,
) -> dict[str, dict[str, object]]:
    """Evaluate available baselines on exactly the model prediction roster.

    Domain-specific inputs use explicit direction-aligned column names.  The
    function refuses partial score coverage so a favorable subset cannot be
    compared with the model's full roster by accident.
    """

    if not random_seeds:
        raise ValueError("random_seeds must contain at least one seed")
    if any(
        not isinstance(seed, (int, np.integer))
        or isinstance(seed, (bool, np.bool_))
        or seed < 0
        for seed in random_seeds
    ):
        raise ValueError("random_seeds must be nonnegative integers")
    if len(set(random_seeds)) != len(random_seeds):
        raise ValueError("random_seeds must not contain duplicates")
    if (
        not isinstance(paired_bootstrap_repetitions, (int, np.integer))
        or isinstance(paired_bootstrap_repetitions, (bool, np.bool_))
        or paired_bootstrap_repetitions < 0
    ):
        raise ValueError("paired_bootstrap_repetitions must be a nonnegative integer")
    if not 0 < paired_bootstrap_min_effective_fraction <= 1:
        raise ValueError(
            "paired_bootstrap_min_effective_fraction must be in the interval (0, 1]"
        )
    paired_requested = paired_bootstrap_repetitions > 0
    if paired_requested and (candidate_score_column is None or cluster_column is None):
        raise ValueError(
            "paired baseline inference requires candidate_score_column and "
            "cluster_column"
        )
    resolved_query_columns = list(query_columns or (screen_column,))
    if query_columns is None:
        for column in ("contrast_id", "phenotype_direction"):
            if column in frame and column not in resolved_query_columns:
                resolved_query_columns.append(column)
    working = frame.copy()
    reserved = sorted(
        column
        for column in working.columns
        if isinstance(column, str)
        and (
            column in {"_baseline_query_id", "_fixed_baseline_score"}
            or column.startswith("_random_score_")
        )
    )
    if reserved:
        raise ValueError(
            f"baseline frame contains reserved internal columns: {reserved}"
        )
    working["_baseline_query_id"] = _composite_query_keys(
        working,
        resolved_query_columns,
    )
    required = {
        target_column,
        study_column,
        "_baseline_query_id",
    }
    missing = required - set(working)
    if missing:
        raise ValueError(f"baseline frame is missing columns: {sorted(missing)}")

    results: dict[str, dict[str, object]] = {}
    random_metric_records: list[dict[str, float]] = []
    random_metric_values: dict[str, list[float]] = {}
    for seed in random_seeds:
        score_column = f"_random_score_{seed}"
        working[score_column] = np.random.default_rng(seed).random(len(working))
        metrics = grouped_ranking_metrics(
            working,
            group_column="_baseline_query_id",
            target_column=target_column,
            score_column=score_column,
            study_column=study_column,
            ks=ks,
            probability_score=False,
        )
        random_metric_records.append(metrics)
        for name, value in metrics.items():
            random_metric_values.setdefault(name, []).append(value)
        working.drop(columns=[score_column], inplace=True)
    random_metrics = _metric_mean(random_metric_records)
    results["random"] = {
        "status": "evaluated",
        "random_seed_count": len(random_seeds),
        "random_seeds": list(random_seeds),
        "metrics": random_metrics,
        "seed_sensitivity": {
            name: _seed_summary(values)
            for name, values in random_metric_values.items()
            if name
            in {
                "study_macro_normalized_average_precision",
                "study_macro_documented_success_lift_at_10",
                "study_macro_adjudicated_precision_lift_at_10",
            }
        },
    }

    for spec in FIXED_BASELINES:
        if spec.column not in working:
            results[spec.name] = {
                "status": "unavailable_missing_column",
                "required_column": spec.column,
                "interpretation": spec.interpretation,
            }
            continue
        numeric = _validated_baseline_numeric(working, spec)
        coverage = float(numeric.notna().mean())
        if coverage < 1.0:
            results[spec.name] = {
                "status": "unavailable_incomplete_score_coverage",
                "required_column": spec.column,
                "score_coverage": coverage,
                "interpretation": spec.interpretation,
            }
            continue
        _validate_direction_attestation(working, spec)
        score_column = "_fixed_baseline_score"
        working[score_column] = numeric if spec.higher_is_better else -numeric
        metrics = grouped_ranking_metrics(
            working,
            group_column="_baseline_query_id",
            target_column=target_column,
            score_column=score_column,
            study_column=study_column,
            ks=ks,
            probability_score=False,
        )
        working.drop(columns=[score_column], inplace=True)
        baseline_result: dict[str, object] = {
            "status": "evaluated",
            "source_column": spec.column,
            "score_coverage": coverage,
            "higher_is_better": spec.higher_is_better,
            "interpretation": spec.interpretation,
            "metrics": metrics,
        }
        if paired_requested:
            baseline_result["paired_candidate_difference"] = (
                paired_cluster_bootstrap_difference(
                    working.assign(
                        **{score_column: numeric if spec.higher_is_better else -numeric}
                    ),
                    cluster_column=str(cluster_column),
                    metric_study_column=study_column,
                    screen_column=screen_column,
                    target_column=target_column,
                    candidate_score_column=str(candidate_score_column),
                    comparator_score_column=score_column,
                    metric_name=paired_metric,
                    query_columns=tuple(resolved_query_columns),
                    n_bootstrap=paired_bootstrap_repetitions,
                    confidence=paired_bootstrap_confidence,
                    min_effective_fraction=paired_bootstrap_min_effective_fraction,
                    random_state=paired_bootstrap_random_state,
                )
            )
        results[spec.name] = baseline_result

    results["cross_screen_recurrence"] = {
        "status": "not_implemented",
        "reason": (
            "requires outer-training-family-only recomputation from a frozen "
            "screen universe; a precomputed full-corpus recurrence column is "
            "prohibited because it would leak held-out families"
        ),
    }
    return results


__all__ = [
    "FIXED_BASELINES",
    "FixedBaselineSpec",
    "adapt_direction_aligned_drugz_baseline",
    "adapt_repo_native_mageck_baseline",
    "evaluate_fixed_score_baselines",
]
