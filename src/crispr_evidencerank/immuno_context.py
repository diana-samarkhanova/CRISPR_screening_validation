"""Fail-closed immune-context summaries for post-ranking interpretation.

The module is intentionally report-only. It never changes MAGeCK statistics,
candidate validation labels, or the validation-success model. Published display
signs are not interpreted; only native directions plus an explicit endpoint
polarity are used.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from math import comb, isfinite

import pandas as pd

from .contracts import (
    AntitumorDirection,
    DirectionMappingStatus,
    EndpointPolarity,
    ImmuneScreenEvidenceRecord,
    NativeEffectDirection,
    OrthologyMappingStatus,
    PerturbationModality,
    RankListCompleteness,
    validate_records,
)


@dataclass(frozen=True)
class ImmunoContextResult:
    """Auditable outputs from one immune-context query."""

    summary: pd.DataFrame
    exclusions: pd.DataFrame
    used_evidence: pd.DataFrame
    rank_list_audit: pd.DataFrame
    metadata: dict[str, object]


_TUMOR_DUAL_CONSEQUENCES = {
    "tumor_immune_escape",
    "tumor_immune_sensitization",
}
_IMMUNE_DUAL_CONSEQUENCES = {
    "immune_effector_gain",
    "immune_effector_loss",
    "immune_fitness_gain",
    "immune_fitness_loss",
}
_RRA_COMPATIBILITY_COLUMNS = (
    "perturbation_modality",
    "source_organism",
    "perturbed_compartment",
    "experimental_setting",
    "screen_category",
    "cell_model",
    "immune_cell_type",
    "cancer_type",
    "treatment",
    "comparator",
    "contrast_definition",
    "phenotype_endpoint",
    "timepoint",
    "assay_consequence",
    "analysis_tail",
    "endpoint_polarity",
    "source_effect_semantics",
    "raw_effect_sign_semantics",
    "source_score_type",
    "rank_metric_type",
    "rank_ordering",
    "rank_tie_policy",
    "input_data_level",
    "recurrence_stratum_id",
)
_LANE_CONTEXT_COLUMNS = (
    "recurrence_stratum_id",
    "perturbed_compartment",
    "experimental_setting",
    "screen_category",
    "cell_model",
    "immune_cell_type",
    "cancer_type",
    "treatment",
    "comparator",
    "contrast_definition",
    "phenotype_endpoint",
    "timepoint",
    "analysis_tail",
)
_CANDIDATE_PASSTHROUGH_COLUMNS = (
    "screen_id",
    "contrast_id",
    "method",
    "phenotype_direction",
    "analysis_tail",
    "cnv_corrected",
    "mageck_score",
    "mageck_p_value",
    "mageck_rank",
    "screen_signal_rank",
    "screen_signal_rank_source",
    "screen_signal_percentile",
    "screen_signal_percentile_scope",
    "mageck_fdr",
    "mageck_lfc",
    "mageck_input_sgrna_n",
    "mageck_good_sgrna_n",
    "native_lfc_direction",
    "signal_direction",
    "guide_n",
    "median_guide_lfc",
    "mean_guide_lfc",
    "mean_control_count",
    "guide_direction_agreement",
    "low_count_fraction",
    "zero_fraction_control",
    "zero_fraction_treatment",
    "positive_guide_fraction",
    "negative_guide_fraction",
    "neutral_guide_fraction",
    "guide_lfc_mad",
    "guide_lfc_iqr",
    "top2_abs_lfc_mean",
    "leave_one_guide_out_median_sd",
    "strongest_guide_dominance",
    "absolute_median_guide_lfc",
    "absolute_mean_guide_lfc",
    "is_sensitization_signal",
    "is_neutral_signal",
    "within_screen_effect_percentile",
    "control_sample_n",
    "treatment_sample_n",
    "replicate_correlation",
    "control_replicate_correlation",
    "treatment_replicate_correlation",
    "replicate_effect_sd",
    "median_library_size",
    "normalization_method",
    "mageck_guide_direction_agreement",
    "guide_method",
    "guide_phenotype_direction",
    "guide_analysis_tail",
    "guide_cnv_corrected",
    "guide_screen_signal_rank",
    "guide_screen_signal_rank_source",
    "guide_screen_signal_percentile",
    "guide_screen_signal_percentile_scope",
    "guide_ranking_type",
    "ranking_type",
)


def map_antitumor_direction(
    native_direction: NativeEffectDirection | str,
    endpoint_polarity: EndpointPolarity | str,
    mapping_status: DirectionMappingStatus | str,
) -> AntitumorDirection:
    """Map a native observation only when endpoint semantics are explicit.

    Perturbation modality is deliberately absent from this mapping. In
    particular, CRISPRa is never inverted to mimic knockout.
    """

    native_direction = NativeEffectDirection(native_direction)
    endpoint_polarity = EndpointPolarity(endpoint_polarity)
    mapping_status = DirectionMappingStatus(mapping_status)
    if (
        native_direction == NativeEffectDirection.UNKNOWN
        or endpoint_polarity == EndpointPolarity.UNKNOWN
        or mapping_status != DirectionMappingStatus.EXACT
    ):
        return AntitumorDirection.UNKNOWN
    if native_direction == NativeEffectDirection.NEUTRAL:
        return AntitumorDirection.NEUTRAL
    favorable = (
        native_direction == NativeEffectDirection.ENRICHED
        and endpoint_polarity == EndpointPolarity.ENRICHMENT_IS_FAVORABLE
    ) or (
        native_direction == NativeEffectDirection.DEPLETED
        and endpoint_polarity == EndpointPolarity.DEPLETION_IS_FAVORABLE
    )
    return AntitumorDirection.FAVORABLE if favorable else AntitumorDirection.UNFAVORABLE


def order_statistic_rra_pvalue(normalized_ranks: Iterable[float]) -> float:
    """Return a small dependency-free order-statistic RRA baseline.

    Values are normalized ranks in ``(0, 1]`` after conservative provenance
    collapse. The result measures rank recurrence, not effect size, causality,
    or probability of validation.
    """

    ranks = sorted(float(value) for value in normalized_ranks)
    if not ranks:
        raise ValueError("RRA requires at least one normalized rank")
    if any(value <= 0.0 or value > 1.0 for value in ranks):
        raise ValueError("normalized ranks must lie in (0, 1]")
    n = len(ranks)
    order_pvalues: list[float] = []
    for order, rank in enumerate(ranks, start=1):
        beta_cdf = sum(
            comb(n, count) * rank**count * (1.0 - rank) ** (n - count)
            for count in range(order, n + 1)
        )
        order_pvalues.append(beta_cdf)
    return min(1.0, n * min(order_pvalues))


def _validated_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    valid, errors = validate_records(frame, ImmuneScreenEvidenceRecord)
    if not errors.empty:
        details = "; ".join(
            f"row {row.row_number}: {row.error}"
            for row in errors.itertuples(index=False)
        )
        raise ValueError(f"immune evidence validation failed: {details}")
    for column in (
        "available_date",
        "transformation_available_date",
        "source_snapshot_date",
        "retrieved_date",
    ):
        valid[column] = pd.to_datetime(valid[column], errors="raise").dt.date
    _validate_provenance_identity(valid)
    return valid


def _candidate_query(candidates: pd.DataFrame, gene_column: str) -> pd.DataFrame:
    if gene_column not in candidates:
        raise ValueError(f"candidate table is missing gene column: {gene_column}")
    if candidates.empty:
        raise ValueError("candidate table contains no records")
    raw = candidates[gene_column]
    if raw.isna().any() or raw.astype(str).str.strip().eq("").any():
        raise ValueError("candidate gene symbols cannot be missing or empty")
    passthrough = [
        column for column in _CANDIDATE_PASSTHROUGH_COLUMNS if column in candidates
    ]
    query = candidates[[gene_column, *passthrough]].copy()
    query = query.rename(columns={gene_column: "gene_symbol"})
    query["gene_symbol"] = query["gene_symbol"].astype(str).str.strip().str.upper()
    for column in (
        "screen_id",
        "contrast_id",
        "method",
        "analysis_tail",
        "screen_signal_rank_source",
        "screen_signal_percentile_scope",
        "guide_analysis_tail",
        "guide_screen_signal_rank_source",
        "guide_screen_signal_percentile_scope",
        "normalization_method",
    ):
        if column not in query:
            continue
        if query[column].isna().any():
            raise ValueError(f"candidate {column} cannot be missing or empty")
        query[column] = query[column].astype(str).str.strip()
        if query[column].eq("").any():
            raise ValueError(f"candidate {column} cannot be missing or empty")
    if "phenotype_direction" in query:
        allowed_directions = {"resistance", "sensitization", "neutral"}
        invalid_directions = sorted(
            set(query["phenotype_direction"].dropna().astype(str)) - allowed_directions
        )
        if invalid_directions or query["phenotype_direction"].isna().any():
            raise ValueError(
                "candidate phenotype_direction must be resistance, "
                f"sensitization, or neutral: {invalid_directions}"
            )
    if "native_lfc_direction" in query:
        allowed_native_directions = {"positive", "negative", "neutral"}
        invalid_native_directions = sorted(
            set(query["native_lfc_direction"].dropna().astype(str))
            - allowed_native_directions
        )
        if invalid_native_directions or query["native_lfc_direction"].isna().any():
            raise ValueError(
                "candidate native_lfc_direction must be positive, negative, "
                f"or neutral: {invalid_native_directions}"
            )
    if "signal_direction" in query:
        allowed_signal_directions = {"resistance", "sensitization", "neutral"}
        invalid_signal_directions = sorted(
            set(query["signal_direction"].dropna().astype(str))
            - allowed_signal_directions
        )
        if invalid_signal_directions or query["signal_direction"].isna().any():
            raise ValueError(
                "candidate signal_direction must be resistance, sensitization, "
                f"or neutral: {invalid_signal_directions}"
            )
    if "guide_phenotype_direction" in query:
        allowed_guide_directions = {"resistance", "sensitization", "neutral"}
        invalid_guide_directions = sorted(
            set(query["guide_phenotype_direction"].dropna().astype(str))
            - allowed_guide_directions
        )
        if invalid_guide_directions or query["guide_phenotype_direction"].isna().any():
            raise ValueError(
                "candidate guide_phenotype_direction must be resistance, "
                f"sensitization, or neutral: {invalid_guide_directions}"
            )
        if (
            "signal_direction" in query
            and not query["signal_direction"]
            .eq(query["guide_phenotype_direction"])
            .all()
        ):
            raise ValueError(
                "candidate signal_direction must match guide_phenotype_direction"
            )
    numeric_bounds = {
        "screen_signal_rank": (1.0, None),
        "screen_signal_percentile": (0.0, 1.0),
        "guide_screen_signal_rank": (1.0, None),
        "guide_screen_signal_percentile": (0.0, 1.0),
        "mageck_score": (0.0, 1.0),
        "mageck_p_value": (0.0, 1.0),
        "mageck_fdr": (0.0, 1.0),
        "mageck_lfc": (None, None),
        "mageck_rank": (1.0, None),
        "mageck_input_sgrna_n": (0.0, None),
        "mageck_good_sgrna_n": (0.0, None),
        "guide_n": (0.0, None),
        "median_guide_lfc": (None, None),
        "mean_guide_lfc": (None, None),
        "mean_control_count": (0.0, None),
        "guide_direction_agreement": (0.0, 1.0),
        "low_count_fraction": (0.0, 1.0),
        "zero_fraction_control": (0.0, 1.0),
        "zero_fraction_treatment": (0.0, 1.0),
        "positive_guide_fraction": (0.0, 1.0),
        "negative_guide_fraction": (0.0, 1.0),
        "neutral_guide_fraction": (0.0, 1.0),
        "guide_lfc_mad": (0.0, None),
        "guide_lfc_iqr": (0.0, None),
        "top2_abs_lfc_mean": (0.0, None),
        "leave_one_guide_out_median_sd": (0.0, None),
        "strongest_guide_dominance": (0.0, 1.0),
        "absolute_median_guide_lfc": (0.0, None),
        "absolute_mean_guide_lfc": (0.0, None),
        "is_sensitization_signal": (0.0, 1.0),
        "is_neutral_signal": (0.0, 1.0),
        "within_screen_effect_percentile": (0.0, 1.0),
        "control_sample_n": (1.0, None),
        "treatment_sample_n": (1.0, None),
        "replicate_correlation": (-1.0, 1.0),
        "control_replicate_correlation": (-1.0, 1.0),
        "treatment_replicate_correlation": (-1.0, 1.0),
        "replicate_effect_sd": (0.0, None),
        "median_library_size": (0.0, None),
    }
    for column, (lower, upper) in numeric_bounds.items():
        if column not in query:
            continue
        observed = query[column].dropna()
        numeric = pd.to_numeric(observed, errors="coerce")
        if numeric.isna().any() or not numeric.map(isfinite).all():
            raise ValueError(f"candidate {column} must be finite numeric")
        if lower is not None and (numeric < lower).any():
            raise ValueError(f"candidate {column} must be at least {lower:g}")
        if upper is not None and (numeric > upper).any():
            raise ValueError(f"candidate {column} must be at most {upper:g}")
    if (
        "ranking_type" in query
        and not query["ranking_type"].eq("screen_signal_baseline").all()
    ):
        raise ValueError("candidate ranking_type must be screen_signal_baseline")
    if (
        "guide_ranking_type" in query
        and not query["guide_ranking_type"].eq("screen_signal_baseline").all()
    ):
        raise ValueError("candidate guide_ranking_type must be screen_signal_baseline")
    if (
        "normalization_method" in query
        and not query["normalization_method"].isin({"median_ratio", "cpm"}).all()
    ):
        raise ValueError("candidate normalization_method must be median_ratio or cpm")
    identity_columns = [
        column
        for column in (
            "gene_symbol",
            "screen_id",
            "contrast_id",
            "phenotype_direction",
            "analysis_tail",
        )
        if column in query
    ]
    duplicated = query.duplicated(identity_columns, keep=False)
    for _, group in query.loc[duplicated].groupby(
        identity_columns,
        sort=False,
        dropna=False,
    ):
        if any(group[column].nunique(dropna=False) > 1 for column in query.columns):
            raise ValueError("duplicate candidate identity has conflicting values")
    query = query.drop_duplicates(identity_columns, keep="first")
    sort_columns = [
        column
        for column in (
            "screen_id",
            "contrast_id",
            "phenotype_direction",
            "screen_signal_rank",
            "gene_symbol",
            "analysis_tail",
        )
        if column in query
    ]
    return query.sort_values(
        sort_columns, kind="stable", na_position="last"
    ).reset_index(drop=True)


def _primary_axis_relation(
    phenotype_direction: object,
    dual_action_class: object,
) -> str:
    # The candidate table is intentionally unfiltered. A row can represent a
    # non-significant MAGeCK tail, so joining it to immune evidence must not
    # manufacture a compound "screen signal + immune benefit" claim.
    del phenotype_direction, dual_action_class
    return "axes_reported_separately"


def _human_gene_symbol(row: pd.Series) -> str | None:
    mapping = OrthologyMappingStatus(row["orthology_mapping_status"])
    if mapping in {
        OrthologyMappingStatus.AMBIGUOUS,
        OrthologyMappingStatus.UNMAPPED,
    }:
        return None
    organism = str(row["source_organism"]).casefold()
    if organism in {"human", "homo sapiens", "9606"}:
        return str(row["mapped_human_gene_symbol"] or row["gene_symbol"]).upper()
    mapped = row["mapped_human_gene_symbol"]
    return str(mapped).upper() if mapped else None


def _validate_provenance_identity(frame: pd.DataFrame) -> None:
    """Reject identifiers that claim multiple independent provenance identities."""

    checks = (
        (
            ["source_name", "external_study_id"],
            ["source_family_id"],
            "external study",
        ),
        (
            ["source_name", "external_screen_id"],
            [
                "external_study_id",
                "source_family_id",
                "raw_data_family_id",
            ],
            "external screen",
        ),
        (
            ["source_name", "external_comparison_id"],
            [
                "external_study_id",
                "external_screen_id",
                "source_family_id",
                "raw_data_family_id",
            ],
            "external comparison",
        ),
    )
    for key_columns, value_columns, label in checks:
        for key, group in frame.groupby(key_columns, sort=True, dropna=False):
            if len(group[value_columns].drop_duplicates()) > 1:
                raise ValueError(
                    f"{label} {key!r} maps to multiple provenance identities"
                )

    ranked = frame.loc[frame["rank_list_id"].notna()]
    rank_identity_columns = [
        "source_name",
        "source_version",
        "external_study_id",
        "external_screen_id",
        "external_comparison_id",
        "source_family_id",
        "raw_data_family_id",
    ]
    for rank_list_id, group in ranked.groupby("rank_list_id", sort=True):
        if len(group[rank_identity_columns].drop_duplicates()) > 1:
            raise ValueError(
                f"rank list {rank_list_id!r} maps to multiple provenance identities"
            )


def _provenance_components(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(index=frame.index, dtype="string")
    parents: dict[str, str] = {}

    def find(value: str) -> str:
        parents.setdefault(value, value)
        if parents[value] != value:
            parents[value] = find(parents[value])
        return parents[value]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for row in frame.itertuples(index=False):
        union(f"source:{row.source_family_id}", f"raw:{row.raw_data_family_id}")
    return frame.apply(
        lambda row: find(f"source:{row['source_family_id']}"),
        axis=1,
    )


def _family_vote(values: pd.Series) -> str:
    observed = {str(value) for value in values}
    if not observed:
        return AntitumorDirection.UNKNOWN.value
    if len(observed) > 1:
        return AntitumorDirection.DISCORDANT.value
    return next(iter(observed))


def _directional_support_reason(row: pd.Series, *, max_source_fdr: float) -> str:
    if row["direction_mapping_status"] != DirectionMappingStatus.EXACT.value:
        return "direction_mapping_not_exact"
    if row["assay_consequence"] in {"ambiguous", "neutral"}:
        return "assay_consequence_not_directional"
    if row["native_effect_direction"] not in {"enriched", "depleted"}:
        return "native_direction_not_directional"
    if pd.notna(row["raw_effect"]) and float(row["raw_effect"]) == 0.0:
        return "zero_raw_effect"
    if pd.notna(row["raw_effect"]) and row["raw_effect_sign_semantics"] == (
        "unsigned_or_not_applicable"
    ):
        return "raw_effect_sign_semantics_not_directional"
    if pd.isna(row["source_fdr"]):
        return "source_fdr_missing"
    if float(row["source_fdr"]) > max_source_fdr:
        return "source_fdr_above_threshold"
    return "eligible"


def _lane_summary(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "family_n": 0,
            "favorable_family_n": 0,
            "unfavorable_family_n": 0,
            "unresolved_family_n": 0,
            "favorable_fraction": None,
            "unfavorable_fraction": None,
            "discordant_family_n": 0,
            "mixed_context": False,
            "votes": [],
            "components": [],
        }
    votes_by_component = frame.groupby("provenance_component", sort=True)[
        "support_direction"
    ].agg(_family_vote)
    votes = votes_by_component.tolist()
    context_n = len(frame[list(_LANE_CONTEXT_COLUMNS)].drop_duplicates())
    mixed_context = context_n > 1
    denominator = len(votes)
    favorable_n = votes.count(AntitumorDirection.FAVORABLE.value)
    unfavorable_n = votes.count(AntitumorDirection.UNFAVORABLE.value)
    unresolved_n = sum(
        value
        in {
            AntitumorDirection.UNKNOWN.value,
            AntitumorDirection.NEUTRAL.value,
            AntitumorDirection.DISCORDANT.value,
        }
        for value in votes
    )
    return {
        "family_n": len(votes),
        "favorable_family_n": favorable_n,
        "unfavorable_family_n": unfavorable_n,
        "unresolved_family_n": unresolved_n,
        "favorable_fraction": (
            favorable_n / denominator if denominator and not mixed_context else None
        ),
        "unfavorable_fraction": (
            unfavorable_n / denominator if denominator and not mixed_context else None
        ),
        "discordant_family_n": votes.count(AntitumorDirection.DISCORDANT.value),
        "mixed_context": mixed_context,
        "votes": votes,
        "components": votes_by_component.index.astype(str).tolist(),
    }


def _dual_action_class(
    tumor: dict[str, object],
    immune: dict[str, object],
    *,
    requested: bool,
    incompatible_evidence: bool,
) -> tuple[str, str]:
    if not requested:
        return "not_assessed", "not_assessed"
    if incompatible_evidence:
        return "context_dependent", "incompatible_dual_action_evidence"
    tumor_votes = set(tumor["votes"])
    immune_votes = set(immune["votes"])
    if not tumor_votes or not immune_votes:
        return "insufficient_evidence", "insufficient_evidence"
    if tumor["mixed_context"] or immune["mixed_context"]:
        return "context_dependent", "mixed_context_strata"
    conflict_values = {
        AntitumorDirection.DISCORDANT.value,
        AntitumorDirection.NEUTRAL.value,
        AntitumorDirection.UNKNOWN.value,
    }
    if tumor_votes & conflict_values or immune_votes & conflict_values:
        return "context_dependent", "conflicting_or_unresolved"
    if len(tumor_votes) > 1 or len(immune_votes) > 1:
        return "context_dependent", "cross_family_conflict"
    tumor_vote = next(iter(tumor_votes))
    immune_vote = next(iter(immune_votes))
    independent_components = set(tumor["components"]) | set(immune["components"])
    if len(independent_components) < 2:
        return "insufficient_evidence", "single_provenance_component"
    if tumor_vote == immune_vote == AntitumorDirection.FAVORABLE.value:
        recurrent = tumor["family_n"] >= 2 and immune["family_n"] >= 2
        return (
            "dual_benefit_candidate",
            "recurrent" if recurrent else "preliminary",
        )
    if (
        tumor_vote == AntitumorDirection.FAVORABLE.value
        and immune_vote == AntitumorDirection.UNFAVORABLE.value
    ):
        return "immune_liability", "preliminary"
    if (
        tumor_vote == AntitumorDirection.UNFAVORABLE.value
        and immune_vote == AntitumorDirection.FAVORABLE.value
    ):
        return "tumor_liability", "preliminary"
    return "context_dependent", "concordant_unfavorable_or_other"


def _canonical_rank_roster_sha256(group: pd.DataFrame) -> str:
    ordered = group.sort_values("source_rank", kind="stable")
    roster = "".join(
        f"{row.gene_symbol}\t{int(row.source_rank)}\n"
        for row in ordered.itertuples(index=False)
    )
    return hashlib.sha256(roster.encode("utf-8")).hexdigest()


def _append_audit_reason(
    audit: pd.DataFrame,
    mask: pd.Series,
    reason: str,
) -> None:
    audit.loc[mask, "reason"] = audit.loc[mask, "reason"].map(
        lambda current: f"{current};{reason}" if pd.notna(current) else reason
    )
    audit.loc[mask, "status"] = "ineligible"


def _rank_list_audit(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rank_list_id",
        "declared_full",
        "status",
        "reason",
        "observed_row_n",
        "observed_gene_n",
        "observed_canonical_human_gene_n",
        "declared_gene_n",
        "source_family_id",
        "raw_data_family_id",
        "provenance_component",
        "analysis_tail",
        "perturbed_compartment",
        "experimental_setting",
        "screen_category",
        "assay_consequence",
        "source_effect_semantics",
        "canonical_roster_sha256",
    ]
    with_ids = frame.loc[frame["rank_list_id"].notna()].copy()
    records: list[dict[str, object]] = []
    for rank_list_id, group in with_ids.groupby("rank_list_id", sort=True):
        declared_full = (
            group["rank_list_completeness"]
            .eq(RankListCompleteness.FULL_RANKED_LIST.value)
            .all()
        )
        metadata_columns = (
            "gene_universe_size",
            "external_screen_id",
            "external_comparison_id",
            "source_family_id",
            "raw_data_family_id",
            "recurrence_stratum_id",
            "rank_list_sha256",
            *_RRA_COMPATIBILITY_COLUMNS,
        )
        metadata_consistent = all(
            group[column].nunique(dropna=False) == 1 for column in metadata_columns
        )
        declared_values = group["gene_universe_size"].dropna().astype(int).unique()
        declared_gene_n = int(declared_values[0]) if len(declared_values) == 1 else None
        ranks = pd.to_numeric(group["source_rank"], errors="coerce")
        exact_roster = False
        if declared_gene_n is not None and not ranks.isna().any():
            rank_values = sorted(ranks.astype(int).tolist())
            exact_roster = rank_values == list(range(1, declared_gene_n + 1))
        observed_gene_n = group["gene_symbol"].astype(str).nunique()
        canonical_human_symbols = group.apply(_human_gene_symbol, axis=1)
        observed_canonical_human_gene_n = canonical_human_symbols.dropna().nunique()
        reasons: list[str] = []
        if not declared_full:
            reasons.append("not_declared_full")
        if not metadata_consistent:
            reasons.append("inconsistent_rank_list_metadata")
        if declared_gene_n is None or len(group) != declared_gene_n:
            reasons.append("declared_observed_count_mismatch")
        if observed_gene_n != len(group):
            reasons.append("duplicate_gene")
        if canonical_human_symbols.isna().any():
            reasons.append("non_unique_or_unmapped_orthology_in_rank_roster")
        if observed_canonical_human_gene_n != len(group):
            reasons.append("duplicate_canonical_human_gene_in_rank_roster")
        if not exact_roster:
            reasons.append("rank_roster_incomplete_or_duplicated")
        canonical_checksum = (
            _canonical_rank_roster_sha256(group)
            if exact_roster and observed_gene_n == len(group)
            else None
        )
        declared_checksums = group["rank_list_sha256"].dropna().astype(str).unique()
        if (
            canonical_checksum is None
            or len(declared_checksums) != 1
            or declared_checksums[0] != canonical_checksum
        ):
            reasons.append("canonical_roster_checksum_mismatch")
        if not group["source_effect_semantics"].eq("native").all():
            reasons.append("non_native_rank_semantics")
        status = "verified_full_list" if not reasons else "ineligible"
        records.append(
            {
                "rank_list_id": rank_list_id,
                "declared_full": declared_full,
                "status": status,
                "reason": ";".join(reasons) if reasons else None,
                "observed_row_n": len(group),
                "observed_gene_n": observed_gene_n,
                "observed_canonical_human_gene_n": (observed_canonical_human_gene_n),
                "declared_gene_n": declared_gene_n,
                "source_family_id": (
                    group["source_family_id"].iloc[0]
                    if group["source_family_id"].nunique() == 1
                    else None
                ),
                "raw_data_family_id": (
                    group["raw_data_family_id"].iloc[0]
                    if group["raw_data_family_id"].nunique() == 1
                    else None
                ),
                "provenance_component": None,
                "analysis_tail": (
                    group["analysis_tail"].iloc[0]
                    if group["analysis_tail"].nunique() == 1
                    else None
                ),
                "perturbed_compartment": (
                    group["perturbed_compartment"].iloc[0]
                    if group["perturbed_compartment"].nunique() == 1
                    else None
                ),
                "experimental_setting": (
                    group["experimental_setting"].iloc[0]
                    if group["experimental_setting"].nunique() == 1
                    else None
                ),
                "screen_category": (
                    group["screen_category"].iloc[0]
                    if group["screen_category"].nunique() == 1
                    else None
                ),
                "assay_consequence": (
                    group["assay_consequence"].iloc[0]
                    if group["assay_consequence"].nunique() == 1
                    else None
                ),
                "source_effect_semantics": (
                    group["source_effect_semantics"].iloc[0]
                    if group["source_effect_semantics"].nunique() == 1
                    else None
                ),
                "canonical_roster_sha256": canonical_checksum,
            }
        )
    audit = pd.DataFrame(records, columns=columns)
    if audit.empty:
        return audit

    failed_declared_full = audit["declared_full"] & audit["status"].ne(
        "verified_full_list"
    )
    if failed_declared_full.any():
        _append_audit_reason(
            audit,
            audit["status"].eq("verified_full_list"),
            "selected_stratum_contains_failed_declared_full_list",
        )

    initially_verified = audit["status"].eq("verified_full_list")
    verified_ids = set(audit.loc[initially_verified, "rank_list_id"].astype(str))
    verified_rows = with_ids.loc[
        with_ids["rank_list_id"].astype(str).isin(verified_ids)
    ]
    incompatible = [
        column
        for column in _RRA_COMPATIBILITY_COLUMNS
        if verified_rows[column].nunique(dropna=False) > 1
    ]
    if incompatible:
        _append_audit_reason(
            audit,
            initially_verified,
            "incompatible_rank_list_stratum:" + ",".join(incompatible),
        )

    representatives = (
        with_ids.sort_values(["rank_list_id", "source_rank"], kind="stable")
        .drop_duplicates("rank_list_id")
        .copy()
    )
    if not representatives.empty:
        representatives["provenance_component"] = _provenance_components(
            representatives
        )
        component_by_list = representatives.set_index("rank_list_id")[
            "provenance_component"
        ]
        audit["provenance_component"] = audit["rank_list_id"].map(component_by_list)

    still_verified = audit["status"].eq("verified_full_list")
    repeated_components = set(
        audit.loc[still_verified]
        .groupby("provenance_component", dropna=False)["rank_list_id"]
        .nunique()
        .loc[lambda counts: counts > 1]
        .index
    )
    if repeated_components:
        _append_audit_reason(
            audit,
            still_verified,
            "selected_stratum_contains_multiple_lists_per_provenance_component",
        )
    return audit


def _rra_by_gene(
    frame: pd.DataFrame,
    rank_list_audit: pd.DataFrame,
    candidate_symbols: list[str],
) -> dict[str, dict[str, object]]:
    result = {
        gene: {
            "eligible": False,
            "pvalue": None,
            "independent_family_n": 0,
            "verified_list_n": 0,
            "reason": "no_verified_full_rank_lists",
        }
        for gene in candidate_symbols
    }
    verified = set(
        rank_list_audit.loc[
            rank_list_audit["status"].eq("verified_full_list"), "rank_list_id"
        ].astype(str)
    )
    if not verified:
        reasons = sorted(
            {str(value) for value in rank_list_audit["reason"].dropna() if str(value)}
        )
        if reasons:
            for gene in candidate_symbols:
                result[gene]["reason"] = "rank_list_audit_failed:" + "|".join(reasons)
        return result
    ranked = frame.loc[frame["rank_list_id"].astype(str).isin(verified)].copy()
    ranked["human_gene_symbol"] = ranked.apply(_human_gene_symbol, axis=1)
    ranked = ranked.loc[ranked["human_gene_symbol"].notna()].copy()
    ambiguous_human_mapping = ranked.duplicated(
        ["rank_list_id", "human_gene_symbol"],
        keep=False,
    )
    ranked = ranked.loc[~ambiguous_human_mapping].copy()
    ranked["normalized_rank"] = pd.to_numeric(
        ranked["source_rank"], errors="raise"
    ) / pd.to_numeric(ranked["gene_universe_size"], errors="raise")
    ranked["provenance_component"] = _provenance_components(ranked)
    for gene in candidate_symbols:
        gene_rows = ranked.loc[ranked["human_gene_symbol"].eq(gene)]
        result[gene]["verified_list_n"] = len(verified)
        if gene_rows.empty:
            result[gene]["reason"] = "gene_absent_from_verified_lists"
            continue
        observed_lists = set(gene_rows["rank_list_id"].astype(str))
        if observed_lists != verified:
            result[gene]["reason"] = (
                "gene_missing_or_ambiguous_in_one_or_more_verified_lists"
            )
            continue
        family_ranks = (
            gene_rows.groupby("provenance_component", sort=True)["normalized_rank"]
            .median()
            .tolist()
        )
        result[gene]["independent_family_n"] = len(family_ranks)
        if len(family_ranks) < 2:
            result[gene]["reason"] = "fewer_than_two_independent_families"
            continue
        result[gene] = {
            "eligible": True,
            "pvalue": order_statistic_rra_pvalue(family_ranks),
            "independent_family_n": len(family_ranks),
            "verified_list_n": len(verified),
            "reason": "computed",
        }
    return result


def summarize_immuno_context(
    evidence: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    cutoff_date: date,
    target_modality: PerturbationModality | str,
    candidate_gene_column: str = "gene_symbol",
    excluded_source_families: Iterable[str] = (),
    excluded_raw_data_families: Iterable[str] = (),
    recurrence_stratum_id: str | None = None,
    dual_action_group_id: str | None = None,
    dual_action_group_version: str | None = None,
    max_source_fdr: float = 0.05,
    target_absence_attested: bool = False,
) -> ImmunoContextResult:
    """Summarize immune evidence without modifying the primary ranking."""

    target_modality = PerturbationModality(target_modality)
    if not 0.0 <= max_source_fdr <= 1.0:
        raise ValueError("max_source_fdr must lie in [0, 1]")
    if (dual_action_group_id is None) != (dual_action_group_version is None):
        raise ValueError(
            "dual_action_group_id and dual_action_group_version must be requested "
            "together"
        )
    if dual_action_group_id is not None and dual_action_group_version is not None:
        dual_action_group_id = str(dual_action_group_id).strip()
        dual_action_group_version = str(dual_action_group_version).strip()
        if not dual_action_group_id or not dual_action_group_version:
            raise ValueError("dual-action group identity cannot be blank")
    if recurrence_stratum_id is not None:
        recurrence_stratum_id = str(recurrence_stratum_id).strip()
        if not recurrence_stratum_id:
            raise ValueError("recurrence_stratum_id cannot be blank")
    candidate_query = _candidate_query(candidates, candidate_gene_column)
    candidate_symbols = candidate_query["gene_symbol"].drop_duplicates().tolist()
    valid = _validated_evidence(evidence)
    source_exclusions = {str(value) for value in excluded_source_families}
    raw_exclusions = {str(value) for value in excluded_raw_data_families}
    unknown_source_exclusions = source_exclusions - set(
        valid["source_family_id"].astype(str)
    )
    unknown_raw_exclusions = raw_exclusions - set(
        valid["raw_data_family_id"].astype(str)
    )
    if unknown_source_exclusions or unknown_raw_exclusions:
        raise ValueError(
            "declared self-family exclusions are absent from the evidence table: "
            f"source={sorted(unknown_source_exclusions)}, "
            f"raw={sorted(unknown_raw_exclusions)}"
        )

    reason = pd.Series(pd.NA, index=valid.index, dtype="string")
    effective_available = valid[
        [
            "available_date",
            "transformation_available_date",
            "source_snapshot_date",
        ]
    ].max(axis=1)
    reason.loc[effective_available > cutoff_date] = "post_cutoff"
    reason.loc[reason.isna() & valid["source_family_id"].isin(source_exclusions)] = (
        "excluded_source_family"
    )
    reason.loc[reason.isna() & valid["raw_data_family_id"].isin(raw_exclusions)] = (
        "excluded_raw_data_family"
    )
    reason.loc[
        reason.isna() & valid["perturbation_modality"].ne(target_modality.value)
    ] = "modality_mismatch"

    rank_base_mask = reason.isna().copy()
    human_symbols = valid.apply(_human_gene_symbol, axis=1)
    reason.loc[reason.isna() & human_symbols.isna()] = "non_unique_orthology"
    exclusions = valid.loc[reason.notna()].copy()
    exclusions.insert(0, "exclusion_reason", reason.loc[reason.notna()].values)
    exclusions = exclusions.drop(columns=["used_for_label"])
    eligible = valid.loc[reason.isna()].copy()
    eligible["human_gene_symbol"] = human_symbols.loc[eligible.index]
    if eligible.empty:
        eligible["antitumor_direction"] = pd.Series(dtype="string")
    else:
        eligible["antitumor_direction"] = eligible.apply(
            lambda row: (
                map_antitumor_direction(
                    row["native_effect_direction"],
                    row["endpoint_polarity"],
                    row["direction_mapping_status"],
                ).value
            ),
            axis=1,
        )
    eligible["provenance_component"] = _provenance_components(eligible)
    if eligible.empty:
        eligible["directional_support_reason"] = pd.Series(dtype="string")
    else:
        eligible["directional_support_reason"] = eligible.apply(
            lambda row: _directional_support_reason(
                row,
                max_source_fdr=max_source_fdr,
            ),
            axis=1,
        )
    eligible["directional_support_eligible"] = eligible[
        "directional_support_reason"
    ].eq("eligible")
    eligible["support_direction"] = eligible["antitumor_direction"].where(
        eligible["directional_support_eligible"],
        AntitumorDirection.UNKNOWN.value,
    )

    rank_input = valid.loc[
        rank_base_mask
        & (
            valid["recurrence_stratum_id"].eq(recurrence_stratum_id)
            if recurrence_stratum_id is not None
            else False
        )
    ].copy()
    rank_list_audit = _rank_list_audit(rank_input)
    self_exclusion_applied = bool(
        source_exclusions or raw_exclusions or target_absence_attested
    )
    if recurrence_stratum_id is None:
        rra = {
            gene: {
                "eligible": False,
                "pvalue": None,
                "independent_family_n": 0,
                "verified_list_n": 0,
                "reason": "recurrence_stratum_not_requested",
            }
            for gene in candidate_symbols
        }
    elif not self_exclusion_applied:
        rra = {
            gene: {
                "eligible": False,
                "pvalue": None,
                "independent_family_n": 0,
                "verified_list_n": int(
                    rank_list_audit["status"].eq("verified_full_list").sum()
                ),
                "reason": "self_exclusion_unverified",
            }
            for gene in candidate_symbols
        }
    else:
        rra = _rra_by_gene(rank_input, rank_list_audit, candidate_symbols)

    records: list[dict[str, object]] = []
    for gene in candidate_symbols:
        gene_rows = eligible.loc[eligible["human_gene_symbol"].eq(gene)].copy()
        tumor = _lane_summary(
            gene_rows.loc[gene_rows["perturbed_compartment"].eq("tumor_cell")]
        )
        immune = _lane_summary(
            gene_rows.loc[gene_rows["perturbed_compartment"].eq("immune_cell")]
        )
        in_vivo = _lane_summary(
            gene_rows.loc[gene_rows["experimental_setting"].eq("in_vivo")]
        )
        tumor_in_vivo = _lane_summary(
            gene_rows.loc[
                gene_rows["experimental_setting"].eq("in_vivo")
                & gene_rows["perturbed_compartment"].eq("tumor_cell")
            ]
        )
        immune_in_vivo = _lane_summary(
            gene_rows.loc[
                gene_rows["experimental_setting"].eq("in_vivo")
                & gene_rows["perturbed_compartment"].eq("immune_cell")
            ]
        )

        dual_rows = gene_rows
        if dual_action_group_id is not None:
            dual_rows = dual_rows.loc[
                dual_rows["dual_action_group_id"].eq(dual_action_group_id)
                & dual_rows["dual_action_group_version"].eq(dual_action_group_version)
            ]
        tumor_mask = dual_rows["perturbed_compartment"].eq("tumor_cell")
        immune_mask = dual_rows["perturbed_compartment"].eq("immune_cell")
        compatible_tumor = tumor_mask & dual_rows["assay_consequence"].isin(
            _TUMOR_DUAL_CONSEQUENCES
        )
        compatible_immune = immune_mask & dual_rows["assay_consequence"].isin(
            _IMMUNE_DUAL_CONSEQUENCES
        )
        incompatible_dual = bool(
            dual_action_group_id is not None
            and (~(compatible_tumor | compatible_immune)).any()
        )
        dual_tumor = _lane_summary(dual_rows.loc[compatible_tumor])
        dual_immune = _lane_summary(dual_rows.loc[compatible_immune])
        dual_class, dual_confidence = _dual_action_class(
            dual_tumor,
            dual_immune,
            requested=dual_action_group_id is not None,
            incompatible_evidence=incompatible_dual,
        )
        unresolved_fraction = (
            gene_rows["support_direction"].eq(AntitumorDirection.UNKNOWN.value).mean()
            if not gene_rows.empty
            else None
        )
        rra_gene = rra[gene]
        records.append(
            {
                "gene_symbol": gene,
                "report_only_immuno_context_available": not gene_rows.empty,
                "report_only_immuno_record_n": len(gene_rows),
                "report_only_immuno_raw_family_n": gene_rows[
                    "raw_data_family_id"
                ].nunique(),
                "report_only_immuno_source_family_n": gene_rows[
                    "source_family_id"
                ].nunique(),
                "report_only_immuno_independent_family_n": gene_rows[
                    "provenance_component"
                ].nunique(),
                "report_only_immuno_directional_support_record_n": int(
                    gene_rows["directional_support_eligible"].sum()
                ),
                "report_only_immuno_tumor_family_n": tumor["family_n"],
                "report_only_immuno_tumor_favorable_family_n": tumor[
                    "favorable_family_n"
                ],
                "report_only_immuno_tumor_unfavorable_family_n": tumor[
                    "unfavorable_family_n"
                ],
                "report_only_immuno_tumor_unresolved_family_n": tumor[
                    "unresolved_family_n"
                ],
                "report_only_immuno_tumor_favorable_fraction": tumor[
                    "favorable_fraction"
                ],
                "report_only_immuno_tumor_unfavorable_fraction": tumor[
                    "unfavorable_fraction"
                ],
                "report_only_immuno_tumor_discordant_family_n": tumor[
                    "discordant_family_n"
                ],
                "report_only_immuno_tumor_mixed_context": tumor["mixed_context"],
                "report_only_immuno_immune_family_n": immune["family_n"],
                "report_only_immuno_immune_favorable_family_n": immune[
                    "favorable_family_n"
                ],
                "report_only_immuno_immune_unfavorable_family_n": immune[
                    "unfavorable_family_n"
                ],
                "report_only_immuno_immune_unresolved_family_n": immune[
                    "unresolved_family_n"
                ],
                "report_only_immuno_immune_favorable_fraction": immune[
                    "favorable_fraction"
                ],
                "report_only_immuno_immune_unfavorable_fraction": immune[
                    "unfavorable_fraction"
                ],
                "report_only_immuno_immune_discordant_family_n": immune[
                    "discordant_family_n"
                ],
                "report_only_immuno_immune_mixed_context": immune["mixed_context"],
                "report_only_immuno_in_vivo_family_n": in_vivo["family_n"],
                "report_only_immuno_in_vivo_favorable_family_n": in_vivo[
                    "favorable_family_n"
                ],
                "report_only_immuno_in_vivo_unfavorable_family_n": in_vivo[
                    "unfavorable_family_n"
                ],
                "report_only_immuno_in_vivo_unresolved_family_n": in_vivo[
                    "unresolved_family_n"
                ],
                "report_only_immuno_in_vivo_favorable_fraction": in_vivo[
                    "favorable_fraction"
                ],
                "report_only_immuno_in_vivo_mixed_context": in_vivo["mixed_context"],
                "report_only_immuno_tumor_in_vivo_family_n": tumor_in_vivo["family_n"],
                "report_only_immuno_tumor_in_vivo_favorable_family_n": (
                    tumor_in_vivo["favorable_family_n"]
                ),
                "report_only_immuno_tumor_in_vivo_unfavorable_family_n": (
                    tumor_in_vivo["unfavorable_family_n"]
                ),
                "report_only_immuno_tumor_in_vivo_unresolved_family_n": (
                    tumor_in_vivo["unresolved_family_n"]
                ),
                "report_only_immuno_tumor_in_vivo_favorable_fraction": (
                    tumor_in_vivo["favorable_fraction"]
                ),
                "report_only_immuno_tumor_in_vivo_mixed_context": tumor_in_vivo[
                    "mixed_context"
                ],
                "report_only_immuno_immune_in_vivo_family_n": immune_in_vivo[
                    "family_n"
                ],
                "report_only_immuno_immune_in_vivo_favorable_family_n": (
                    immune_in_vivo["favorable_family_n"]
                ),
                "report_only_immuno_immune_in_vivo_unfavorable_family_n": (
                    immune_in_vivo["unfavorable_family_n"]
                ),
                "report_only_immuno_immune_in_vivo_unresolved_family_n": (
                    immune_in_vivo["unresolved_family_n"]
                ),
                "report_only_immuno_immune_in_vivo_favorable_fraction": (
                    immune_in_vivo["favorable_fraction"]
                ),
                "report_only_immuno_immune_in_vivo_mixed_context": immune_in_vivo[
                    "mixed_context"
                ],
                "report_only_immuno_direction_unresolved_fraction": (
                    unresolved_fraction
                ),
                "report_only_immuno_dual_action_class": dual_class,
                "report_only_immuno_dual_action_confidence": dual_confidence,
                "report_only_immuno_rra_eligible": rra_gene["eligible"],
                "report_only_immuno_rra_pvalue": rra_gene["pvalue"],
                "report_only_immuno_rra_independent_family_n": rra_gene[
                    "independent_family_n"
                ],
                "report_only_immuno_rra_verified_list_n": rra_gene["verified_list_n"],
                "report_only_immuno_rra_reason": rra_gene["reason"],
                "report_only_immuno_source_snapshot_date": (
                    max(gene_rows["source_snapshot_date"]).isoformat()
                    if not gene_rows.empty
                    else None
                ),
                "report_only_immuno_target_modality": target_modality.value,
            }
        )

    gene_summary = pd.DataFrame.from_records(records).sort_values(
        "gene_symbol", kind="stable"
    )
    summary = candidate_query.merge(
        gene_summary,
        on="gene_symbol",
        how="left",
        validate="many_to_one",
    )
    if "phenotype_direction" in summary:
        summary["report_only_immuno_primary_axis_relation"] = summary.apply(
            lambda row: _primary_axis_relation(
                row["phenotype_direction"],
                row["report_only_immuno_dual_action_class"],
            ),
            axis=1,
        )
    exclusions = exclusions.sort_values("evidence_id", kind="stable").reset_index(
        drop=True
    )
    metadata: dict[str, object] = {
        "method": "report_only_immune_context",
        "cutoff_date": cutoff_date.isoformat(),
        "target_modality": target_modality.value,
        "candidate_gene_column": candidate_gene_column,
        "recurrence_stratum_id": recurrence_stratum_id,
        "dual_action_group_id": dual_action_group_id,
        "dual_action_group_version": dual_action_group_version,
        "max_source_fdr": max_source_fdr,
        "target_absence_attested": target_absence_attested,
        "input_evidence_records": len(valid),
        "eligible_evidence_records": len(eligible),
        "candidate_genes": len(candidate_symbols),
        "candidate_query_rows": len(candidate_query),
        "exclusion_reason_counts": {
            str(key): int(value)
            for key, value in exclusions["exclusion_reason"]
            .value_counts()
            .sort_index()
            .items()
        }
        if not exclusions.empty
        else {},
        "excluded_source_families": sorted(source_exclusions),
        "excluded_raw_data_families": sorted(raw_exclusions),
        "self_exclusion_status": (
            "user_attested_target_absent"
            if target_absence_attested
            else "applied"
            if source_exclusions or raw_exclusions
            else "unverified_report_only"
        ),
        "rra_method": "order_statistic_baseline_v1_unadjusted",
        "rra_interpretation": (
            "order-statistic rank recurrence; not effect size, validation, or "
            "probability"
        ),
    }
    return ImmunoContextResult(
        summary=summary.reset_index(drop=True),
        exclusions=exclusions,
        used_evidence=eligible.drop(columns=["used_for_label"])
        .sort_values("evidence_id", kind="stable")
        .reset_index(drop=True),
        rank_list_audit=rank_list_audit,
        metadata=metadata,
    )
