"""Self-contained screen-signal reports for new MAGeCK or count inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import featurize_count_table
from .io import normalize_count_inputs, normalize_mageck_gene_summary


@dataclass(frozen=True)
class ScreenReportResult:
    """Deterministic scientific outputs before file-level provenance is added."""

    ranked_candidates: pd.DataFrame
    qc_summary: dict[str, object]
    report_markdown: str


def _opposite_direction(direction: str) -> str:
    if direction == "resistance":
        return "sensitization"
    if direction == "sensitization":
        return "resistance"
    raise ValueError("direction must be resistance or sensitization")


def _require_finite_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & (~np.isfinite(values))
        if invalid.any():
            raise ValueError(f"{column} must contain finite numeric values")


def _rank_mageck(
    mageck_summary: pd.DataFrame,
    *,
    screen_id: str,
    contrast_id: str,
    positive_tail_means: str,
) -> pd.DataFrame:
    gene_column = next(
        (
            column
            for column in ("id", "gene", "Gene", "gene_symbol")
            if column in mageck_summary
        ),
        None,
    )
    if gene_column is None:
        raise ValueError("could not identify the gene column in MAGeCK summary")
    raw_genes = mageck_summary[gene_column]
    if raw_genes.isna().any() or raw_genes.astype(str).str.strip().eq("").any():
        raise ValueError("MAGeCK gene identifiers cannot be missing or empty")
    normalized = normalize_mageck_gene_summary(
        mageck_summary,
        screen_id=screen_id,
        contrast_id=contrast_id,
    )
    if normalized.duplicated(["gene_symbol", "analysis_tail"]).any():
        raise ValueError("MAGeCK summary has duplicate gene rows within a tail")
    _require_finite_numeric(
        normalized,
        [
            "score",
            "effect",
            "p_value",
            "fdr",
            "rank",
            "input_sgrna_n",
            "good_sgrna_n",
        ],
    )
    for column, label in (
        ("fdr", "FDR"),
        ("p_value", "p-values"),
        ("score", "RRA scores"),
    ):
        if column in normalized:
            observed = pd.to_numeric(normalized[column], errors="coerce").dropna()
            if ((observed < 0) | (observed > 1)).any():
                raise ValueError(f"MAGeCK {label} must lie in [0, 1]")
    if "rank" in normalized:
        observed_rank = pd.to_numeric(normalized["rank"], errors="coerce").dropna()
        if (observed_rank < 1).any() or not np.allclose(
            observed_rank,
            np.rint(observed_rank),
            rtol=0,
            atol=1e-9,
        ):
            raise ValueError("MAGeCK ranks must be positive integers")
    for column in ("input_sgrna_n", "good_sgrna_n"):
        if column not in normalized:
            continue
        observed = pd.to_numeric(normalized[column], errors="coerce").dropna()
        if (observed < 0).any() or not np.allclose(
            observed,
            np.rint(observed),
            rtol=0,
            atol=1e-9,
        ):
            raise ValueError(f"MAGeCK {column} must contain non-negative integers")
    if {"input_sgrna_n", "good_sgrna_n"} <= set(normalized):
        input_n = pd.to_numeric(normalized["input_sgrna_n"], errors="coerce")
        good_n = pd.to_numeric(normalized["good_sgrna_n"], errors="coerce")
        if (good_n > input_n).fillna(False).any():
            raise ValueError("MAGeCK good_sgrna_n cannot exceed input_sgrna_n")

    positive_direction = positive_tail_means
    negative_direction = _opposite_direction(positive_tail_means)
    normalized["phenotype_direction"] = normalized["analysis_tail"].map(
        {
            "mageck_pos": positive_direction,
            "mageck_neg": negative_direction,
        }
    )
    rename = {
        "score": "mageck_score",
        "effect": "mageck_lfc",
        "p_value": "mageck_p_value",
        "fdr": "mageck_fdr",
        "rank": "mageck_rank",
        "input_sgrna_n": "mageck_input_sgrna_n",
        "good_sgrna_n": "mageck_good_sgrna_n",
    }
    normalized = normalized.rename(columns=rename)
    rank_sources: list[str] = []
    ranks: list[pd.Series] = []
    for _, tail in normalized.groupby("analysis_tail", sort=False):
        if "mageck_rank" in tail and tail["mageck_rank"].notna().all():
            rank = pd.to_numeric(tail["mageck_rank"], errors="raise")
            source = "mageck_native_rank"
        else:
            metric = next(
                (
                    column
                    for column in (
                        "mageck_fdr",
                        "mageck_p_value",
                        "mageck_score",
                    )
                    if column in tail and tail[column].notna().all()
                ),
                None,
            )
            if metric is None:
                raise ValueError(
                    "MAGeCK summary requires one complete rank, FDR, p-value, "
                    "or score column per tail"
                )
            rank = pd.to_numeric(tail[metric], errors="raise").rank(
                method="min",
                ascending=True,
            )
            source = f"derived_from_{metric}"
        ranks.append(rank)
        rank_sources.extend([source] * len(tail))
    normalized["screen_signal_rank"] = pd.concat(ranks).sort_index()
    normalized["screen_signal_rank_source"] = pd.Series(
        rank_sources,
        index=normalized.index,
    )
    normalized["screen_signal_percentile"] = normalized.groupby(
        "analysis_tail", sort=False
    )["screen_signal_rank"].transform(
        lambda values: (
            1.0
            - (values.rank(method="min", ascending=True) - 1.0)
            / max(len(values) - 1, 1)
        )
    )
    normalized["screen_signal_percentile_scope"] = "observed_rows_within_tail"
    normalized["ranking_type"] = "screen_signal_baseline"
    return normalized.drop(columns=["direction"])


def _rank_count_features(
    counts: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    declared_screen_id: str | None,
    declared_contrast_id: str | None,
    positive_lfc_means: str,
    pseudocount: float,
    low_count_threshold: float,
    normalization_method: str,
    direction_deadband: float,
) -> pd.DataFrame:
    counts, samples = normalize_count_inputs(counts, samples)
    if "condition_role" in samples:
        observed_roles = set(samples["condition_role"].astype(str))
        invalid_roles = sorted(observed_roles - {"control", "treatment"})
        if invalid_roles:
            raise ValueError(
                "count report condition_role must be control or treatment: "
                f"{invalid_roles}"
            )
    for column, declared in (
        ("screen_id", declared_screen_id),
        ("contrast_id", declared_contrast_id),
    ):
        if declared is None or column not in samples:
            continue
        observed = sorted(set(samples[column].astype(str)))
        if observed != [declared]:
            raise ValueError(
                f"declared {column} {declared!r} does not match sample sheet "
                f"values {observed!r}"
            )
    features = featurize_count_table(
        counts,
        samples,
        pseudocount=pseudocount,
        low_count_threshold=low_count_threshold,
        normalization_method=normalization_method,
        direction_deadband=direction_deadband,
    )
    sample_counts = (
        samples.groupby(
            ["screen_id", "contrast_id", "condition_role"],
            sort=False,
        )
        .size()
        .unstack("condition_role", fill_value=0)
        .reset_index()
        .rename(
            columns={
                "control": "control_sample_n",
                "treatment": "treatment_sample_n",
            }
        )
    )
    for column in ("control_sample_n", "treatment_sample_n"):
        if column not in sample_counts:
            sample_counts[column] = 0
    features = features.merge(
        sample_counts[
            [
                "screen_id",
                "contrast_id",
                "control_sample_n",
                "treatment_sample_n",
            ]
        ],
        on=["screen_id", "contrast_id"],
        how="left",
        validate="many_to_one",
    )
    features["native_lfc_direction"] = np.select(
        [
            features["median_guide_lfc"] > direction_deadband,
            features["median_guide_lfc"] < -direction_deadband,
        ],
        ["positive", "negative"],
        default="neutral",
    )
    negative_lfc_means = _opposite_direction(positive_lfc_means)
    features["phenotype_direction"] = features["native_lfc_direction"].map(
        {
            "positive": positive_lfc_means,
            "negative": negative_lfc_means,
            "neutral": "neutral",
        }
    )
    features["signal_direction"] = features["phenotype_direction"]
    features["is_sensitization_signal"] = (
        features["phenotype_direction"].eq("sensitization").astype(float)
    )
    features["analysis_tail"] = features["native_lfc_direction"].map(
        {
            "positive": "count_positive_lfc",
            "negative": "count_negative_lfc",
            "neutral": "count_neutral",
        }
    )
    features["screen_signal_rank"] = np.nan
    features["screen_signal_percentile"] = np.nan
    directional = features["phenotype_direction"].isin(["resistance", "sensitization"])
    for _, indices in (
        features.loc[directional]
        .groupby(
            ["screen_id", "contrast_id", "phenotype_direction"],
            sort=False,
        )
        .groups.items()
    ):
        effect = features.loc[indices, "median_guide_lfc"].abs()
        rank = effect.rank(method="min", ascending=False)
        features.loc[indices, "screen_signal_rank"] = rank
        features.loc[indices, "screen_signal_percentile"] = 1.0 - (rank - 1.0) / max(
            len(rank) - 1, 1
        )
    features["screen_signal_rank_source"] = np.where(
        directional,
        "absolute_median_guide_lfc",
        "neutral_not_ranked",
    )
    features["screen_signal_percentile_scope"] = np.where(
        directional,
        "observed_rows_within_screen_contrast_direction",
        "neutral_not_ranked",
    )
    features["ranking_type"] = "screen_signal_baseline"
    return features


def _merge_mageck_and_counts(
    mageck: pd.DataFrame,
    count_features: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["screen_id", "contrast_id", "gene_symbol"]
    relevant = count_features.merge(
        mageck[keys].drop_duplicates(),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if relevant.empty:
        raise ValueError("MAGeCK and count inputs have no shared screen contrast")
    count_columns = [column for column in relevant.columns if column not in keys]
    count_columns = [
        column if column not in mageck.columns else f"guide_{column}"
        for column in count_columns
    ]
    count_rename = dict(
        zip(
            [column for column in relevant.columns if column not in keys],
            count_columns,
            strict=True,
        )
    )
    relevant = relevant.rename(columns=count_rename)
    merged = mageck.merge(relevant, on=keys, how="left", validate="many_to_one")
    guide_direction_column = (
        "guide_phenotype_direction"
        if "guide_phenotype_direction" in merged
        else "phenotype_direction"
    )
    if guide_direction_column != "phenotype_direction":
        agreement = pd.Series(pd.NA, index=merged.index, dtype="boolean")
        observed = merged[guide_direction_column].notna()
        agreement.loc[observed] = merged.loc[observed, guide_direction_column].eq(
            merged.loc[observed, "phenotype_direction"]
        )
        merged["mageck_guide_direction_agreement"] = agreement
    return merged


def _qc_summary(
    ranked: pd.DataFrame,
    *,
    mode: str,
    fdr_threshold: float,
) -> dict[str, object]:
    warnings: list[str] = []
    guide_qc_gene_coverage_n: int | None = None
    guide_qc_gene_coverage_fraction: float | None = None
    if "guide_n" in ranked:
        gene_guide_qc = ranked[["gene_symbol", "guide_n"]].drop_duplicates(
            "gene_symbol"
        )
        observed_guide_qc = pd.to_numeric(
            gene_guide_qc["guide_n"], errors="coerce"
        ).dropna()
        guide_qc_gene_coverage_n = len(observed_guide_qc)
        guide_qc_gene_coverage_fraction = guide_qc_gene_coverage_n / max(
            len(gene_guide_qc), 1
        )
        if guide_qc_gene_coverage_fraction < 1.0:
            warnings.append(
                "guide-level QC is missing for "
                f"{1.0 - guide_qc_gene_coverage_fraction:.1%} of MAGeCK genes"
            )
        if not observed_guide_qc.empty:
            few_guides = observed_guide_qc.lt(3).mean()
            if few_guides > 0:
                warnings.append(
                    f"{few_guides:.1%} of genes with guide QC have fewer than 3 guides"
                )
    replicate_correlations: dict[str, float | None] = {}
    for column in (
        "replicate_correlation",
        "control_replicate_correlation",
        "treatment_replicate_correlation",
    ):
        candidate_columns = [column, f"guide_{column}"]
        observed_column = next(
            (value for value in candidate_columns if value in ranked),
            None,
        )
        if observed_column is not None:
            values = pd.to_numeric(ranked[observed_column], errors="coerce").dropna()
            value = float(values.median()) if not values.empty else None
            replicate_correlations[column] = value
            if value is not None and value < 0.7:
                warnings.append(f"{column} is below 0.70")
    if "mageck_fdr" not in ranked:
        warnings.append("no MAGeCK FDR: ranking is count-effect-only")
        significant_rows = None
    else:
        fdr = pd.to_numeric(ranked["mageck_fdr"], errors="coerce")
        significant_rows = int(fdr.le(fdr_threshold).sum())
    if "mageck_rank" in ranked:
        incomplete_native_roster = False
        for _, group in ranked.groupby("analysis_tail", sort=False):
            ranks = pd.to_numeric(group["mageck_rank"], errors="coerce")
            if ranks.isna().any() or sorted(ranks.astype(int).tolist()) != list(
                range(1, len(group) + 1)
            ):
                incomplete_native_roster = True
                break
        if incomplete_native_roster:
            warnings.append(
                "native MAGeCK rank roster is incomplete; percentile is local to "
                "observed rows within each tail"
            )
    replicate_sample_counts: list[dict[str, object]] = []
    if {"control_sample_n", "treatment_sample_n"} <= set(ranked):
        replicate_design = ranked[
            [
                "screen_id",
                "contrast_id",
                "control_sample_n",
                "treatment_sample_n",
            ]
        ].drop_duplicates()
        replicate_sample_counts = replicate_design.to_dict(orient="records")
        if (
            replicate_design[["control_sample_n", "treatment_sample_n"]]
            .lt(2)
            .any(axis=None)
        ):
            warnings.append(
                "replicate QC unavailable for an arm with fewer than 2 samples"
            )
    return {
        "report_type": "screen_signal_baseline",
        "mode": mode,
        "gene_rows": len(ranked),
        "unique_genes": int(ranked["gene_symbol"].nunique()),
        "screens": sorted(ranked["screen_id"].astype(str).unique().tolist()),
        "contrasts": sorted(ranked["contrast_id"].astype(str).unique().tolist()),
        "fdr_threshold": fdr_threshold,
        "rows_at_or_below_fdr_threshold": significant_rows,
        "replicate_correlations": replicate_correlations,
        "replicate_sample_counts": replicate_sample_counts,
        "guide_qc_gene_coverage_n": guide_qc_gene_coverage_n,
        "guide_qc_gene_coverage_fraction": guide_qc_gene_coverage_fraction,
        "warnings": sorted(set(warnings)),
        "interpretation": (
            "screen-signal priority only; not a validation probability or "
            "therapeutic recommendation"
        ),
    }


def _markdown_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).replace("|", "\\|")


def _markdown_top_table(ranked: pd.DataFrame, *, top_n: int = 10) -> str:
    effect_column = next(
        (column for column in ("mageck_lfc", "median_guide_lfc") if column in ranked),
        None,
    )
    lines = [
        "| Direction | Rank | Gene | Effect | MAGeCK FDR |",
        "|---|---:|---|---:|---:|",
    ]
    directional = ranked.loc[
        ranked["phenotype_direction"].isin(["resistance", "sensitization"])
    ]
    for direction in ("resistance", "sensitization"):
        top = (
            directional.loc[directional["phenotype_direction"].eq(direction)]
            .sort_values(["screen_signal_rank", "gene_symbol"], kind="stable")
            .head(top_n)
        )
        for row in top.to_dict(orient="records"):
            lines.append(
                "| "
                + " | ".join(
                    (
                        direction,
                        _markdown_value(row.get("screen_signal_rank")),
                        _markdown_value(row.get("gene_symbol")),
                        _markdown_value(row.get(effect_column))
                        if effect_column
                        else "—",
                        _markdown_value(row.get("mageck_fdr")),
                    )
                )
                + " |"
            )
    return "\n".join(lines)


def _markdown_report(qc: dict[str, object], ranked: pd.DataFrame) -> str:
    warnings = qc["warnings"]
    warning_lines = (
        "\n".join(f"- {warning}" for warning in warnings)
        if warnings
        else "- No prespecified warnings were triggered."
    )
    top_table = _markdown_top_table(ranked)
    percentile_note = (
        "Percentiles are local to the observed rows within each analysis tail."
        if "screen_signal_percentile_scope" in ranked
        else "Percentiles are computed within each declared count-effect direction."
    )
    return f"""# CRISPR screen-signal report

This bundle ranks the observed screen signal. It is **not** a calibrated
validation probability and does not claim that a gene is experimentally
validated or therapeutically favorable.

## Run summary

- Input mode: `{qc["mode"]}`
- Gene rows: {qc["gene_rows"]}
- Unique genes: {qc["unique_genes"]}
- Screens: {", ".join(qc["screens"])}
- Contrasts: {", ".join(qc["contrasts"])}

## QC warnings

{warning_lines}

## Top observed signals by direction

{top_table}

{percentile_note}

## Interpretation boundary

MAGeCK tails and count log-fold changes retain the direction mapping declared
for this run. Orthogonal validation evidence, immune context, therapeutic
priority, and novelty are separate downstream analyses.
"""


def rank_screen(
    *,
    mageck_summary: pd.DataFrame | None = None,
    counts: pd.DataFrame | None = None,
    samples: pd.DataFrame | None = None,
    screen_id: str | None = None,
    contrast_id: str | None = None,
    positive_tail_means: str | None = None,
    positive_lfc_means: str | None = None,
    pseudocount: float = 1.0,
    low_count_threshold: float = 30.0,
    normalization_method: str = "median_ratio",
    direction_deadband: float = 0.1,
    fdr_threshold: float = 0.05,
) -> ScreenReportResult:
    """Create a screen-signal report from MAGeCK, counts, or both."""

    if mageck_summary is None and counts is None:
        raise ValueError("provide MAGeCK summary, counts, or both")
    if (counts is None) != (samples is None):
        raise ValueError("counts and sample sheet must be provided together")
    if (
        mageck_summary is not None
        and counts is not None
        and positive_tail_means is not None
        and positive_lfc_means is not None
        and positive_tail_means != positive_lfc_means
    ):
        raise ValueError(
            "combined mode requires positive_tail_means and "
            "positive_lfc_means to describe the same phenotype direction"
        )
    if not 0.0 <= fdr_threshold <= 1.0:
        raise ValueError("fdr_threshold must lie in [0, 1]")

    if screen_id is not None:
        screen_id = str(screen_id).strip()
        if not screen_id:
            raise ValueError("screen_id cannot be empty")
    if contrast_id is not None:
        contrast_id = str(contrast_id).strip()
        if not contrast_id:
            raise ValueError("contrast_id cannot be empty")

    mageck_ranked: pd.DataFrame | None = None
    if mageck_summary is not None:
        if not all((screen_id, contrast_id, positive_tail_means)):
            raise ValueError(
                "MAGeCK mode requires screen_id, contrast_id, and positive_tail_means"
            )
        mageck_ranked = _rank_mageck(
            mageck_summary,
            screen_id=str(screen_id),
            contrast_id=str(contrast_id),
            positive_tail_means=str(positive_tail_means),
        )

    count_ranked: pd.DataFrame | None = None
    if counts is not None and samples is not None:
        if positive_lfc_means is None:
            raise ValueError("count mode requires positive_lfc_means")
        count_ranked = _rank_count_features(
            counts,
            samples,
            declared_screen_id=screen_id,
            declared_contrast_id=contrast_id,
            positive_lfc_means=positive_lfc_means,
            pseudocount=pseudocount,
            low_count_threshold=low_count_threshold,
            normalization_method=normalization_method,
            direction_deadband=direction_deadband,
        )

    if mageck_ranked is not None and count_ranked is not None:
        ranked = _merge_mageck_and_counts(mageck_ranked, count_ranked)
        mode = "mageck_plus_counts"
    elif mageck_ranked is not None:
        ranked = mageck_ranked
        mode = "mageck"
    else:
        assert count_ranked is not None
        ranked = count_ranked
        mode = "counts"

    ranked = ranked.sort_values(
        [
            "screen_id",
            "contrast_id",
            "phenotype_direction",
            "screen_signal_rank",
            "gene_symbol",
        ],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    qc = _qc_summary(ranked, mode=mode, fdr_threshold=fdr_threshold)
    return ScreenReportResult(
        ranked_candidates=ranked,
        qc_summary=qc,
        report_markdown=_markdown_report(qc, ranked),
    )
