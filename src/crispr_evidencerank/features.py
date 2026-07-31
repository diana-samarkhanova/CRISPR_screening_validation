"""Guide-level QC and gene-level feature extraction."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .io import normalize_count_inputs, validate_count_table


def _median_pairwise_spearman(values: pd.DataFrame) -> float:
    if values.shape[1] < 2:
        return float("nan")
    correlations = values.corr(method="spearman")
    pairs = [
        correlations.loc[left, right]
        for left, right in combinations(correlations.columns, 2)
    ]
    return float(np.nanmedian(pairs)) if pairs else float("nan")


def _median_absolute_deviation(series: pd.Series) -> float:
    median = series.median()
    return float((series - median).abs().median())


def _top2_abs_mean(series: pd.Series) -> float:
    return float(series.abs().nlargest(min(2, len(series))).mean())


def _leave_one_out_median_sd(series: pd.Series) -> float:
    values = series.to_numpy(dtype=float)
    if len(values) < 3:
        return float("nan")
    medians = [np.median(np.delete(values, index)) for index in range(len(values))]
    return float(np.std(medians, ddof=1))


def _strongest_guide_dominance(series: pd.Series) -> float:
    absolute = series.abs()
    total = float(absolute.sum())
    return float(absolute.max() / total) if total > 0 else 0.0


def featurize_count_table(
    counts: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    pseudocount: float = 1.0,
    low_count_threshold: float = 30.0,
    normalization_method: str = "median_ratio",
    direction_deadband: float = 0.1,
) -> pd.DataFrame:
    """Aggregate a wide sgRNA count table to one row per gene and contrast.

    Raw counts use robust median-ratio normalization by default. CPM remains
    available as a declared sensitivity analysis. Guide log-fold change is the
    mean normalized treatment signal minus the mean normalized control signal.
    """

    if not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("pseudocount must be finite and greater than zero")
    if not np.isfinite(low_count_threshold) or low_count_threshold < 0:
        raise ValueError("low_count_threshold must be finite and non-negative")
    if not np.isfinite(direction_deadband) or direction_deadband < 0:
        raise ValueError("direction_deadband must be finite and non-negative")
    if normalization_method not in {"median_ratio", "cpm"}:
        raise ValueError("normalization_method must be 'median_ratio' or 'cpm'")
    counts, samples = normalize_count_inputs(counts, samples)
    sample_ids, _ = validate_count_table(counts, samples)
    required_sample_columns = {
        "sample_id",
        "screen_id",
        "contrast_id",
        "condition_role",
        "replicate",
    }
    missing = required_sample_columns - set(samples.columns)
    if missing:
        raise ValueError(f"sample sheet is missing columns: {sorted(missing)}")
    for column in ("screen_id", "contrast_id", "condition_role"):
        if samples[column].isna().any():
            raise ValueError(f"{column} cannot be missing")
        samples[column] = samples[column].astype(str).str.strip()
        if samples[column].eq("").any():
            raise ValueError(f"{column} cannot be empty")
    replicate_numeric = pd.to_numeric(samples["replicate"], errors="coerce")
    if (
        replicate_numeric.isna().any()
        or not np.isfinite(replicate_numeric).all()
        or (replicate_numeric < 1).any()
        or not np.equal(replicate_numeric, np.floor(replicate_numeric)).all()
    ):
        raise ValueError("replicate must contain positive integers")
    samples["replicate"] = replicate_numeric.astype(int)
    screen_ids_in_call = samples["screen_id"].dropna().astype(str).unique()
    if len(screen_ids_in_call) != 1:
        raise ValueError(
            "one featurization call must contain exactly one screen_id; "
            "split multi-screen count matrices before feature extraction"
        )

    numeric = counts[sample_ids].astype(float)
    library_sizes = numeric.sum(axis=0)
    if (library_sizes <= 0).any():
        bad = library_sizes.index[library_sizes <= 0].tolist()
        raise ValueError(f"samples with zero total library size: {bad}")
    if normalization_method == "cpm":
        normalized = numeric.div(library_sizes, axis=1) * 1_000_000
    else:
        positive = numeric.gt(0)
        informative = positive.sum(axis=1).ge(2)
        if not informative.any():
            raise ValueError(
                "median-ratio normalization requires guides with positive "
                "counts in at least two samples"
            )
        positive_counts = numeric.loc[informative].where(positive.loc[informative])
        geometric_means = np.exp(np.log(positive_counts).mean(axis=1))
        ratios = positive_counts.div(geometric_means, axis=0)
        size_factors = ratios.median(axis=0, skipna=True)
        if not np.isfinite(size_factors).all() or (size_factors <= 0).any():
            raise ValueError("median-ratio normalization produced invalid size factors")
        size_factors = size_factors / np.exp(np.log(size_factors).mean())
        normalized = numeric.div(size_factors, axis=1)
    log_normalized = np.log2(normalized + pseudocount)

    outputs: list[pd.DataFrame] = []
    for contrast_id, design in samples.groupby("contrast_id", sort=False):
        screen_ids = design["screen_id"].dropna().unique()
        if len(screen_ids) != 1:
            raise ValueError(
                f"contrast {contrast_id!r} must map to exactly one screen_id"
            )
        roles = design.groupby("condition_role")["sample_id"].apply(list).to_dict()
        controls = roles.get("control", [])
        treatments = roles.get("treatment", [])
        if not controls or not treatments:
            raise ValueError(
                f"contrast {contrast_id!r} requires control and treatment samples"
            )

        guide_lfc = log_normalized[treatments].mean(axis=1) - log_normalized[
            controls
        ].mean(axis=1)
        guide_frame = counts[list(COUNT_COLUMNS)].copy()
        guide_frame["guide_lfc"] = guide_lfc
        guide_frame["mean_control_count"] = numeric[controls].mean(axis=1)
        guide_frame["low_count"] = (
            guide_frame["mean_control_count"] < low_count_threshold
        ).astype(float)
        guide_frame["zero_control"] = (numeric[controls] == 0).mean(axis=1)
        guide_frame["zero_treatment"] = (numeric[treatments] == 0).mean(axis=1)
        guide_frame["positive_lfc"] = (guide_lfc > direction_deadband).astype(float)
        guide_frame["negative_lfc"] = (guide_lfc < -direction_deadband).astype(float)
        guide_frame["neutral_lfc"] = (guide_lfc.abs() <= direction_deadband).astype(
            float
        )

        grouped = guide_frame.groupby("gene_symbol", sort=False)
        gene = grouped.agg(
            guide_n=("sgrna_id", "size"),
            median_guide_lfc=("guide_lfc", "median"),
            mean_guide_lfc=("guide_lfc", "mean"),
            mean_control_count=("mean_control_count", "mean"),
            low_count_fraction=("low_count", "mean"),
            zero_fraction_control=("zero_control", "mean"),
            zero_fraction_treatment=("zero_treatment", "mean"),
            positive_guide_fraction=("positive_lfc", "mean"),
            negative_guide_fraction=("negative_lfc", "mean"),
            neutral_guide_fraction=("neutral_lfc", "mean"),
        )
        gene["guide_lfc_mad"] = grouped["guide_lfc"].apply(_median_absolute_deviation)
        gene["guide_lfc_iqr"] = grouped["guide_lfc"].apply(
            lambda values: float(values.quantile(0.75) - values.quantile(0.25))
        )
        gene["top2_abs_lfc_mean"] = grouped["guide_lfc"].apply(_top2_abs_mean)
        gene["leave_one_guide_out_median_sd"] = grouped["guide_lfc"].apply(
            _leave_one_out_median_sd
        )
        gene["strongest_guide_dominance"] = grouped["guide_lfc"].apply(
            _strongest_guide_dominance
        )
        gene["guide_direction_agreement"] = gene[
            ["positive_guide_fraction", "negative_guide_fraction"]
        ].max(axis=1)
        gene["absolute_median_guide_lfc"] = gene["median_guide_lfc"].abs()
        gene["absolute_mean_guide_lfc"] = gene["mean_guide_lfc"].abs()
        gene["signal_direction"] = np.select(
            [
                gene["median_guide_lfc"].abs() <= direction_deadband,
                gene["median_guide_lfc"] > direction_deadband,
            ],
            ["neutral", "resistance"],
            default="sensitization",
        )
        gene["is_sensitization_signal"] = (
            gene["signal_direction"] == "sensitization"
        ).astype(float)
        gene["is_neutral_signal"] = (gene["signal_direction"] == "neutral").astype(
            float
        )
        gene["within_screen_effect_percentile"] = (
            gene["median_guide_lfc"].abs().rank(method="average", pct=True)
        )

        control_corr = _median_pairwise_spearman(log_normalized[controls])
        treatment_corr = _median_pairwise_spearman(log_normalized[treatments])
        replicate_values = np.array([control_corr, treatment_corr], dtype=float)
        gene["replicate_correlation"] = (
            float(np.nanmedian(replicate_values))
            if not np.isnan(replicate_values).all()
            else float("nan")
        )
        gene["control_replicate_correlation"] = control_corr
        gene["treatment_replicate_correlation"] = treatment_corr

        treatment_gene_effects = []
        for treatment_sample in treatments:
            per_guide = log_normalized[treatment_sample] - log_normalized[
                controls
            ].mean(axis=1)
            per_guide_frame = pd.DataFrame(
                {
                    "gene_symbol": counts["gene_symbol"].to_numpy(),
                    "effect": per_guide.to_numpy(),
                }
            )
            treatment_gene_effects.append(
                per_guide_frame.groupby("gene_symbol", sort=False)["effect"].median()
            )
        replicate_gene_effects = pd.concat(treatment_gene_effects, axis=1)
        gene["replicate_effect_sd"] = replicate_gene_effects.std(
            axis=1, ddof=1
        ).reindex(gene.index)
        gene["median_library_size"] = float(
            library_sizes[controls + treatments].median()
        )
        gene["normalization_method"] = normalization_method
        gene["screen_id"] = str(screen_ids[0])
        gene["contrast_id"] = str(contrast_id)
        outputs.append(gene.reset_index())

    return pd.concat(outputs, ignore_index=True)


COUNT_COLUMNS = ("sgrna_id", "gene_symbol")


DESIGN_NUMERIC_FIELDS = (
    "library_size_guides",
    "target_gene_count",
    "guides_per_gene_median",
    "nontargeting_guide_count",
    "library_moi",
    "effector_moi",
    "coverage_transduction",
    "coverage_selection",
    "coverage_harvest",
    "infection_replicate_count",
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
)


def _nullable_bool(value: object) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return 1.0
        if normalized in {"false", "no", "0"}:
            return 0.0
        raise ValueError(f"could not parse nullable boolean value: {value!r}")
    return float(bool(value))


def _paired_replicate_fraction(sample_rows: pd.DataFrame) -> float:
    if "pair_id" not in sample_rows:
        return float("nan")
    pair_ids = sample_rows["pair_id"].dropna().astype(str).str.strip()
    pair_ids = pair_ids[pair_ids.ne("")]
    if pair_ids.empty:
        return float("nan")
    paired = 0
    total = 0
    for pair_id, pair_rows in sample_rows.loc[
        sample_rows["pair_id"].astype("string").str.strip().isin(pair_ids)
    ].groupby("pair_id"):
        del pair_id
        roles = set(pair_rows["condition_role"].astype(str))
        total += 1
        paired += int({"control", "treatment"} <= roles)
    return float(paired / total) if total else float("nan")


def featurize_experimental_design(
    screens: pd.DataFrame,
    screen_designs: pd.DataFrame,
    contrasts: pd.DataFrame,
    samples: pd.DataFrame,
) -> pd.DataFrame:
    """Create pre-validation design features for each screen contrast.

    Missing reporting remains missing and receives an explicit indicator.
    Author hit flags, validation outcomes, and post-validation evidence are
    intentionally outside this function.
    """

    required = {
        "screens": (screens, {"screen_id"}),
        "screen_designs": (screen_designs, {"screen_id"}),
        "contrasts": (
            contrasts,
            {
                "screen_id",
                "contrast_id",
                "control_type",
                "intended_direction",
            },
        ),
        "samples": (
            samples,
            {
                "sample_id",
                "screen_id",
                "contrast_id",
                "condition_role",
                "replicate",
            },
        ),
    }
    for name, (frame, columns) in required.items():
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing design columns: {sorted(missing)}")
    for name, frame in (
        ("screens", screens),
        ("screen_designs", screen_designs),
    ):
        if frame["screen_id"].duplicated().any():
            raise ValueError(f"{name} must be unique on screen_id")
    if contrasts.duplicated(["screen_id", "contrast_id"]).any():
        raise ValueError("contrasts must be unique on screen_id and contrast_id")

    merged = contrasts.merge(
        screens,
        on="screen_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_screen"),
    ).merge(
        screen_designs,
        on="screen_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_design"),
    )
    if "study_id" not in merged or merged["study_id"].isna().any():
        raise ValueError("every contrast must map to a known screen")

    records: list[dict[str, object]] = []
    for _, contrast in merged.iterrows():
        screen_id = str(contrast["screen_id"])
        contrast_id = str(contrast["contrast_id"])
        sample_rows = samples.loc[
            (samples["screen_id"].astype(str) == screen_id)
            & (samples["contrast_id"].astype(str) == contrast_id)
        ].copy()
        roles = sample_rows["condition_role"].astype(str)
        record: dict[str, object] = {
            "screen_id": screen_id,
            "contrast_id": contrast_id,
            "study_id": contrast.get("study_id"),
            "source_family_id": contrast.get("source_family_id"),
            "raw_data_family_id": contrast.get("raw_data_family_id"),
            "control_sample_n": int((roles == "control").sum()),
            "treatment_sample_n": int((roles == "treatment").sum()),
            "baseline_sample_n": int((roles == "baseline").sum()),
            "paired_replicate_fraction": _paired_replicate_fraction(sample_rows),
            "vehicle_control_present": _nullable_bool(
                contrast.get("vehicle_control_present")
            ),
            "baseline_control_present": _nullable_bool(
                contrast.get("baseline_control_present")
            ),
            "same_infection_split": _nullable_bool(
                contrast.get("same_infection_split")
            ),
            "matched_control": _nullable_bool(contrast.get("matched_control")),
            "cnv_amplification_risk_assessed": _nullable_bool(
                contrast.get("cnv_amplification_risk_assessed")
            ),
            "treatment_dose_reported": float(pd.notna(contrast.get("treatment_dose"))),
        }
        infection_ids = (
            sample_rows.get("infection_replicate_id", pd.Series(dtype=object))
            .dropna()
            .astype(str)
            .str.strip()
        )
        infection_ids = infection_ids[infection_ids.ne("")]
        record["observed_infection_replicate_count"] = (
            float(infection_ids.nunique()) if not infection_ids.empty else float("nan")
        )
        biological_ids = (
            sample_rows.get("biological_replicate_id", pd.Series(dtype=object))
            .dropna()
            .astype(str)
            .str.strip()
        )
        biological_ids = biological_ids[biological_ids.ne("")]
        record["observed_biological_replicate_count"] = (
            float(biological_ids.nunique())
            if not biological_ids.empty
            else float("nan")
        )
        technical_ids = (
            sample_rows.get("technical_replicate_id", pd.Series(dtype=object))
            .dropna()
            .astype(str)
            .str.strip()
        )
        technical_ids = technical_ids[technical_ids.ne("")]
        record["observed_technical_replicate_count"] = (
            float(technical_ids.nunique()) if not technical_ids.empty else float("nan")
        )
        timepoints = pd.to_numeric(
            sample_rows.get("timepoint_days", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        record["observed_timepoint_count"] = (
            float(timepoints.nunique()) if not timepoints.empty else float("nan")
        )

        for field in DESIGN_NUMERIC_FIELDS:
            value = contrast.get(field)
            record[field] = (
                float(value) if value is not None and pd.notna(value) else np.nan
            )
        coverages = [
            record[field]
            for field in (
                "coverage_transduction",
                "coverage_selection",
                "coverage_harvest",
            )
            if pd.notna(record[field])
        ]
        record["minimum_declared_coverage"] = (
            float(min(coverages)) if coverages else float("nan")
        )

        control_type = str(contrast.get("control_type", "unknown"))
        for value in (
            "vehicle",
            "untreated",
            "baseline_t0",
            "later_timepoint",
            "sorted_population",
            "matched_nontargeting",
            "other",
            "unknown",
        ):
            record[f"control_type_{value}"] = float(control_type == value)
        screen_scale = str(contrast.get("screen_scale", "unknown"))
        record["screen_scale_genome_wide"] = float(screen_scale == "genome_wide")
        selection = str(contrast.get("selection_strategy", "unknown"))
        for value in (
            "positive_selection",
            "negative_selection",
            "bidirectional_selection",
            "competitive_growth",
            "marker_sort",
            "other",
            "unknown",
        ):
            record[f"selection_strategy_{value}"] = float(selection == value)
        replicate_unit = str(contrast.get("replicate_unit", "unknown"))
        record["independent_infection_replicates_declared"] = float(
            replicate_unit == "independent_infection"
        )

        completeness_fields = [
            "library_moi",
            "coverage_transduction",
            "coverage_selection",
            "coverage_harvest",
            "infection_replicate_count",
            "guides_per_gene_median",
            "nontargeting_guide_count",
            "exposure_days",
            "control_type",
            "same_infection_split",
        ]
        completeness = [
            contrast.get(field) is not None
            and not pd.isna(contrast.get(field))
            and str(contrast.get(field)).strip() not in {"", "unknown"}
            for field in completeness_fields
        ]
        record["design_metadata_completeness"] = float(np.mean(completeness))
        records.append(record)

    output = pd.DataFrame.from_records(records)
    nullable_features = [
        *DESIGN_NUMERIC_FIELDS,
        "minimum_declared_coverage",
        "paired_replicate_fraction",
        "vehicle_control_present",
        "baseline_control_present",
        "same_infection_split",
        "matched_control",
        "cnv_amplification_risk_assessed",
        "observed_infection_replicate_count",
        "observed_biological_replicate_count",
        "observed_technical_replicate_count",
        "observed_timepoint_count",
    ]
    for field in nullable_features:
        output[f"{field}_missing"] = output[field].isna().astype(float)
    return output


def merge_artifact_context(
    features: pd.DataFrame,
    context: pd.DataFrame,
    *,
    keys: tuple[str, ...] = ("gene_symbol", "cell_line"),
) -> pd.DataFrame:
    """Join versioned context features and derive explicit artifact indicators."""

    missing = set(keys) - set(features.columns)
    if missing:
        raise ValueError(f"features are missing join keys: {sorted(missing)}")
    missing = set(keys) - set(context.columns)
    if missing:
        raise ValueError(f"context is missing join keys: {sorted(missing)}")

    if context.duplicated(list(keys)).any():
        raise ValueError(f"context must be unique on keys {keys}")
    result = features.merge(context, on=list(keys), how="left", validate="many_to_one")
    if "copy_number" in result:
        result["absolute_cnv_deviation"] = (result["copy_number"] - 2.0).abs()
        result["high_amplification_flag"] = result["copy_number"].ge(4.0)
    if "expression_log2_tpm" in result:
        result["low_expression_flag"] = result["expression_log2_tpm"].lt(1.0)
    if {"raw_effect", "corrected_effect"} <= set(result.columns):
        result["raw_corrected_effect_delta"] = (
            result["raw_effect"] - result["corrected_effect"]
        )
        result["cnv_correction_available"] = result["corrected_effect"].notna()
    if {"raw_rank", "corrected_rank"} <= set(result.columns):
        result["raw_corrected_rank_delta"] = (
            result["raw_rank"] - result["corrected_rank"]
        )
    return result
