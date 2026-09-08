from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import date
from pathlib import Path

import jsonschema
import pandas as pd
import pytest
from pydantic import ValidationError

from crispr_evidencerank.cli import build_parser
from crispr_evidencerank.contracts import ImmuneScreenEvidenceRecord
from crispr_evidencerank.immuno_context import (
    map_antitumor_direction,
    order_statistic_rra_pvalue,
    summarize_immuno_context,
)
from crispr_evidencerank.modeling import validate_success_feature_columns
from crispr_evidencerank.screen_report import rank_screen


def _record(evidence_id: str, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "evidence_id": evidence_id,
        "source_name": "synthetic ICRAFT-inspired fixture",
        "source_version": "fixture-1",
        "source_snapshot_date": "2025-01-01",
        "external_study_id": f"study-{evidence_id}",
        "external_screen_id": f"screen-{evidence_id}",
        "external_comparison_id": f"comparison-{evidence_id}",
        "source_family_id": f"source-{evidence_id}",
        "raw_data_family_id": f"raw-{evidence_id}",
        "gene_symbol": "GENE1",
        "source_organism": "human",
        "mapped_human_gene_symbol": None,
        "orthology_mapping_status": "not_needed",
        "orthology_source": None,
        "orthology_version": None,
        "perturbation_modality": "CRISPR_KO",
        "perturbed_compartment": "tumor_cell",
        "experimental_setting": "in_vitro",
        "screen_category": "coculture_immune_killing",
        "cell_model": "MDA-MB-468",
        "immune_cell_type": None,
        "cancer_type": "triple-negative breast cancer",
        "treatment": "T cells",
        "comparator": "tumor cells alone",
        "contrast_definition": "T-cell coculture versus tumor-only control",
        "phenotype_endpoint": "tumor-cell abundance",
        "assay_consequence": "tumor_immune_sensitization",
        "timepoint": "day 7",
        "recurrence_stratum_id": "ko_tumor_immune_killing_negative_tail",
        "dual_action_group_id": "ko_antitumor_function",
        "dual_action_group_version": "synthetic-v1",
        "native_effect_direction": "depleted",
        "endpoint_polarity": "depletion_is_favorable",
        "direction_mapping_status": "exact",
        "direction_mapping_rule": "native_depletion_is_favorable_v1",
        "raw_effect": -1.5,
        "raw_effect_type": "MAGeCK LFC",
        "raw_effect_sign_semantics": "positive_is_enrichment",
        "source_score": None,
        "source_score_type": None,
        "source_fdr": 0.01,
        "source_rank": None,
        "rank_list_id": None,
        "rank_list_sha256": None,
        "gene_universe_size": None,
        "analysis_tail": None,
        "rank_metric_type": None,
        "rank_ordering": None,
        "rank_tie_policy": None,
        "rank_list_completeness": "top_hits_only",
        "source_effect_semantics": "native",
        "published_sign_inverted": False,
        "input_data_level": "raw_counts",
        "source_url": "https://example.org/evidence",
        "source_locator": "fixture row",
        "available_date": "2025-01-01",
        "transformation_available_date": "2025-01-01",
        "retrieved_date": "2025-02-01",
        "transformation_id": None,
        "used_for_label": False,
        "notes": None,
    }
    record.update(overrides)
    if "raw_effect" not in overrides and record["raw_effect"] is not None:
        magnitude = abs(float(record["raw_effect"]))
        record["raw_effect"] = {
            "enriched": magnitude,
            "depleted": -magnitude,
            "neutral": 0.0,
            "unknown": None,
        }[str(record["native_effect_direction"])]
        if record["raw_effect"] is None:
            record["raw_effect_type"] = None
            record["raw_effect_sign_semantics"] = None
    if record["perturbed_compartment"] == "immune_cell":
        defaults = {
            "treatment": "high-activity CD8 gate",
            "comparator": "low-activity CD8 gate",
            "contrast_definition": "high- versus low-activity sorted CD8 cells",
            "phenotype_endpoint": "effector activity",
            "recurrence_stratum_id": "ko_immune_effector_high_gate_positive_tail",
        }
        for field, value in defaults.items():
            if field not in overrides:
                record[field] = value
    if "direction_mapping_rule" not in overrides:
        record["direction_mapping_rule"] = {
            "enrichment_is_favorable": "native_enrichment_is_favorable_v1",
            "depletion_is_favorable": "native_depletion_is_favorable_v1",
            "unknown": None,
        }[str(record["endpoint_polarity"])]
    return record


def _full_rank_list(
    list_id: str,
    source_family: str,
    raw_family: str,
    gene1_rank: int,
) -> list[dict[str, object]]:
    ranks = {"GENE1": gene1_rank}
    remaining = [rank for rank in (1, 2, 3) if rank != gene1_rank]
    ranks.update({"GENE2": remaining[0], "GENE3": remaining[1]})
    rows: list[dict[str, object]] = []
    for gene, rank in ranks.items():
        rows.append(
            _record(
                f"{list_id}-{gene}",
                source_family_id=source_family,
                raw_data_family_id=raw_family,
                gene_symbol=gene,
                external_study_id=source_family,
                external_screen_id=list_id,
                external_comparison_id=list_id,
                source_rank=rank,
                rank_list_id=list_id,
                rank_list_sha256=("a" if list_id == "list-1" else "b") * 64,
                gene_universe_size=3,
                analysis_tail="negative",
                rank_metric_type="synthetic_native_tail_rank",
                rank_ordering="ascending",
                rank_tie_policy="ordinal_no_ties",
                rank_list_completeness="full_ranked_list",
            )
        )
    roster = "".join(
        f"{row['gene_symbol']}\t{int(row['source_rank'])}\n"
        for row in sorted(rows, key=lambda row: int(row["source_rank"]))
    )
    checksum = hashlib.sha256(roster.encode("utf-8")).hexdigest()
    for row in rows:
        row["rank_list_sha256"] = checksum
    return rows


def test_contract_preserves_native_crispra_effect() -> None:
    record = _record(
        "crispra",
        perturbation_modality="CRISPRa",
        native_effect_direction="enriched",
        endpoint_polarity="enrichment_is_favorable",
        raw_effect=2.0,
        source_score=-2.0,
        source_score_type="ICRAFT display LFC",
        source_effect_semantics="icraft_ko_equivalent_display",
        published_sign_inverted=True,
        transformation_id="icraft_crispra_display_sign_inversion_v1",
    )
    parsed = ImmuneScreenEvidenceRecord.model_validate(record)
    assert parsed.raw_effect == 2.0
    assert parsed.source_score == -2.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"source_score": -1.0}, "negative native LFC"),
        ({"source_score_type": "RRA score"}, "paired LFC"),
        ({"native_effect_direction": "depleted"}, "raw effect sign conflicts"),
        ({"transformation_id": "unregistered-v1"}, "registered transformation"),
    ],
)
def test_contract_rejects_invalid_crispra_display_inversion(
    overrides: dict[str, object],
    message: str,
) -> None:
    record = _record(
        "crispra-invalid",
        perturbation_modality="CRISPRa",
        native_effect_direction="enriched",
        endpoint_polarity="enrichment_is_favorable",
        raw_effect=2.0,
        source_score=-2.0,
        source_score_type="ICRAFT display LFC",
        source_effect_semantics="icraft_ko_equivalent_display",
        published_sign_inverted=True,
        transformation_id="icraft_crispra_display_sign_inversion_v1",
    )
    record.update(overrides)
    with pytest.raises(ValidationError, match=message):
        ImmuneScreenEvidenceRecord.model_validate(record)


def test_contract_rejects_undeclared_display_inversion() -> None:
    record = _record("bad", published_sign_inverted=True)
    with pytest.raises(ValidationError, match="native source semantics"):
        ImmuneScreenEvidenceRecord.model_validate(record)


def test_contract_requires_versioned_one_to_one_orthology() -> None:
    record = _record(
        "mouse",
        source_organism="mouse",
        orthology_mapping_status="one_to_one",
        mapped_human_gene_symbol="GENE1",
    )
    with pytest.raises(ValidationError, match="versioned orthology source"):
        ImmuneScreenEvidenceRecord.model_validate(record)


def test_contract_full_list_requires_checksum_and_roster_fields() -> None:
    record = _record("full", rank_list_completeness="full_ranked_list")
    with pytest.raises(ValidationError, match="full_ranked_list rows require"):
        ImmuneScreenEvidenceRecord.model_validate(record)


@pytest.mark.parametrize(
    "overrides",
    [
        {"rank_list_completeness": "full_ranked_list"},
        {"raw_effect_sign_semantics": None},
        {"raw_effect_type": None},
        {"raw_effect": None},
        {"source_score": 0.2, "source_score_type": None},
        {"source_score": None, "source_score_type": "RRA score"},
        {"dual_action_group_version": None},
        {"dual_action_group_id": None},
        {
            "raw_effect": None,
            "raw_effect_type": None,
            "raw_effect_sign_semantics": None,
            "source_score": None,
            "source_score_type": None,
            "source_fdr": None,
            "source_rank": None,
        },
    ],
)
def test_immune_schema_matches_paired_runtime_invariants(
    overrides: dict[str, object],
) -> None:
    record = _record("schema-pair")
    record.update(overrides)
    schema = json.loads(
        (
            Path(__file__).parents[1] / "schemas" / "immune_screen_evidence.schema.json"
        ).read_text(encoding="utf-8")
    )
    with pytest.raises(ValidationError):
        ImmuneScreenEvidenceRecord.model_validate(record)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(record, schema)


@pytest.mark.parametrize(
    "field",
    ["rank_list_id", "analysis_tail", "rank_metric_type", "rank_tie_policy"],
)
def test_full_rank_list_rejects_blank_semantics(field: str) -> None:
    record = _full_rank_list("list-blank", "SF1", "RF1", 1)[0]
    record[field] = " "
    with pytest.raises(ValidationError, match=field):
        ImmuneScreenEvidenceRecord.model_validate(record)


def test_contract_requires_row_locator_and_typed_numeric_metrics() -> None:
    missing_locator = _record("locator")
    missing_locator.pop("source_locator")
    with pytest.raises(ValidationError, match="source_locator"):
        ImmuneScreenEvidenceRecord.model_validate(missing_locator)

    missing_effect_type = _record("effect-type", raw_effect_type=None)
    with pytest.raises(ValidationError, match="must be supplied together"):
        ImmuneScreenEvidenceRecord.model_validate(missing_effect_type)

    missing_sign_semantics = _record(
        "effect-sign-semantics",
        raw_effect_sign_semantics=None,
    )
    with pytest.raises(ValidationError, match="raw_effect_sign_semantics"):
        ImmuneScreenEvidenceRecord.model_validate(missing_sign_semantics)

    orphan_score_type = _record("score-type", source_score_type="RRA score")
    with pytest.raises(ValidationError, match="must be supplied together"):
        ImmuneScreenEvidenceRecord.model_validate(orphan_score_type)


def test_contract_requires_versioned_dual_action_group() -> None:
    with pytest.raises(ValidationError, match="must be supplied together"):
        ImmuneScreenEvidenceRecord.model_validate(
            _record("unversioned-group", dual_action_group_version=None)
        )


def test_dual_action_rejects_blank_group_identity() -> None:
    with pytest.raises(ValueError, match="cannot be blank"):
        summarize_immuno_context(
            pd.DataFrame([_record("one")]),
            pd.DataFrame({"gene_symbol": ["GENE1"]}),
            cutoff_date=date(2025, 12, 31),
            target_modality="CRISPR_KO",
            dual_action_group_id=" ",
            dual_action_group_version=" ",
        )


def test_direction_mapping_uses_endpoint_semantics_not_modality() -> None:
    assert (
        map_antitumor_direction("depleted", "depletion_is_favorable", "exact")
        == "favorable"
    )
    assert (
        map_antitumor_direction("depleted", "enrichment_is_favorable", "exact")
        == "unfavorable"
    )
    assert map_antitumor_direction("enriched", "unknown", "unresolved") == "unknown"
    assert (
        map_antitumor_direction(
            "depleted",
            "depletion_is_favorable",
            "conditional",
        )
        == "unknown"
    )


def test_human_gene_cannot_be_renamed_through_orthology() -> None:
    with pytest.raises(ValidationError, match="cannot be renamed"):
        ImmuneScreenEvidenceRecord.model_validate(
            _record(
                "renamed",
                gene_symbol="TP53",
                mapped_human_gene_symbol="BRCA1",
            )
        )


def test_ambiguous_assay_consequence_must_remain_unresolved() -> None:
    with pytest.raises(ValidationError, match="must remain unresolved"):
        ImmuneScreenEvidenceRecord.model_validate(
            _record("ambiguous", assay_consequence="ambiguous")
        )


def test_contract_rejects_category_consequence_mismatch() -> None:
    with pytest.raises(ValidationError, match="incompatible with the screen category"):
        ImmuneScreenEvidenceRecord.model_validate(
            _record("marker-masquerading-as-dual", screen_category="marker_expression")
        )


def test_contract_rejects_native_lfc_direction_mismatch() -> None:
    with pytest.raises(ValidationError, match="raw effect sign conflicts"):
        ImmuneScreenEvidenceRecord.model_validate(
            _record(
                "bad-lfc-direction",
                raw_effect=1.5,
                raw_effect_type="log2 fold-change",
            )
        )


def test_unsigned_raw_effect_cannot_create_directional_support() -> None:
    record = _record(
        "unsigned",
        raw_effect=1.5,
        raw_effect_type="unsigned source magnitude",
        raw_effect_sign_semantics="unsigned_or_not_applicable",
    )
    result = summarize_immuno_context(
        pd.DataFrame([record]),
        pd.DataFrame({"gene_symbol": ["GENE1"]}),
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
    )
    assert (
        result.summary.iloc[0]["report_only_immuno_directional_support_record_n"] == 0
    )


def test_summary_deduplicates_families_and_calls_dual_benefit() -> None:
    tumor = _record("tumor", source_family_id="SF1", raw_data_family_id="RF1")
    duplicate = deepcopy(tumor)
    duplicate["evidence_id"] = "tumor-reanalysis"
    duplicate["external_comparison_id"] = "comparison-reanalysis"
    immune = _record(
        "immune",
        source_family_id="SF2",
        raw_data_family_id="RF2",
        perturbed_compartment="immune_cell",
        immune_cell_type="CD8 T cell",
        screen_category="immune_cell_function",
        native_effect_direction="enriched",
        endpoint_polarity="enrichment_is_favorable",
        assay_consequence="immune_effector_gain",
    )
    result = summarize_immuno_context(
        pd.DataFrame([tumor, duplicate, immune]),
        pd.DataFrame({"gene_symbol": ["GENE1"]}),
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
        dual_action_group_id="ko_antitumor_function",
        dual_action_group_version="synthetic-v1",
    )
    row = result.summary.iloc[0]
    assert row["report_only_immuno_record_n"] == 3
    assert row["report_only_immuno_independent_family_n"] == 2
    assert row["report_only_immuno_tumor_family_n"] == 1
    assert row["report_only_immuno_immune_family_n"] == 1
    assert row["report_only_immuno_dual_action_class"] == ("dual_benefit_candidate")
    assert row["report_only_immuno_dual_action_confidence"] == "preliminary"


def test_dual_action_requires_significant_directional_support() -> None:
    tumor = _record("tumor", source_fdr=1.0)
    immune = _record(
        "immune",
        source_family_id="SF2",
        raw_data_family_id="RF2",
        perturbed_compartment="immune_cell",
        screen_category="immune_cell_function",
        native_effect_direction="enriched",
        endpoint_polarity="enrichment_is_favorable",
        assay_consequence="immune_effector_gain",
        source_fdr=1.0,
    )
    result = summarize_immuno_context(
        pd.DataFrame([tumor, immune]),
        pd.DataFrame({"gene_symbol": ["GENE1"]}),
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
        dual_action_group_id="ko_antitumor_function",
        dual_action_group_version="synthetic-v1",
    )
    row = result.summary.iloc[0]
    assert row["report_only_immuno_directional_support_record_n"] == 0
    assert row["report_only_immuno_dual_action_class"] == "context_dependent"


def test_dual_action_liability_requires_independent_provenance() -> None:
    tumor = _record("tumor", source_family_id="SF1", raw_data_family_id="RF1")
    immune = _record(
        "immune",
        source_family_id="SF1",
        raw_data_family_id="RF1",
        perturbed_compartment="immune_cell",
        screen_category="immune_cell_function",
        native_effect_direction="depleted",
        endpoint_polarity="enrichment_is_favorable",
        assay_consequence="immune_effector_loss",
    )
    result = summarize_immuno_context(
        pd.DataFrame([tumor, immune]),
        pd.DataFrame({"gene_symbol": ["GENE1"]}),
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
        dual_action_group_id="ko_antitumor_function",
        dual_action_group_version="synthetic-v1",
    )
    row = result.summary.iloc[0]
    assert row["report_only_immuno_dual_action_class"] == "insufficient_evidence"
    assert row["report_only_immuno_dual_action_confidence"] == (
        "single_provenance_component"
    )


def test_summary_applies_modality_cutoff_self_and_orthology_gates() -> None:
    crispra = _record("crispra", perturbation_modality="CRISPRa")
    future = _record(
        "future",
        available_date="2026-01-01",
        transformation_available_date="2026-01-01",
        source_snapshot_date="2026-01-01",
        retrieved_date="2026-02-01",
    )
    self_row = _record(
        "self",
        source_family_id="SF-target",
        raw_data_family_id="RF-target",
    )
    ambiguous = _record(
        "ambiguous",
        source_organism="mouse",
        orthology_mapping_status="ambiguous",
        mapped_human_gene_symbol=None,
    )
    result = summarize_immuno_context(
        pd.DataFrame([crispra, future, self_row, ambiguous]),
        pd.DataFrame({"gene_symbol": ["GENE1"]}),
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
        excluded_raw_data_families=["RF-target"],
    )
    reasons = set(result.exclusions["exclusion_reason"])
    assert reasons == {
        "modality_mismatch",
        "post_cutoff",
        "excluded_raw_data_family",
        "non_unique_orthology",
    }
    assert not bool(result.summary.iloc[0]["report_only_immuno_context_available"])


def test_conflicting_rows_within_family_are_not_cherry_picked() -> None:
    favorable = _record("favorable", source_family_id="SF1", raw_data_family_id="RF1")
    unfavorable = _record(
        "unfavorable",
        source_family_id="SF1",
        raw_data_family_id="RF1",
        native_effect_direction="enriched",
        assay_consequence="tumor_immune_escape",
    )
    immune = _record(
        "immune",
        source_family_id="SF2",
        raw_data_family_id="RF2",
        perturbed_compartment="immune_cell",
        native_effect_direction="enriched",
        endpoint_polarity="enrichment_is_favorable",
        assay_consequence="immune_effector_gain",
    )
    result = summarize_immuno_context(
        pd.DataFrame([favorable, unfavorable, immune]),
        pd.DataFrame({"gene_symbol": ["GENE1"]}),
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
        dual_action_group_id="ko_antitumor_function",
        dual_action_group_version="synthetic-v1",
    )
    row = result.summary.iloc[0]
    assert row["report_only_immuno_tumor_discordant_family_n"] == 1
    assert row["report_only_immuno_dual_action_class"] == "context_dependent"


def test_rra_requires_verified_complete_lists_and_two_families() -> None:
    rows = [
        *_full_rank_list("list-1", "SF1", "RF1", 1),
        *_full_rank_list("list-2", "SF2", "RF2", 2),
    ]
    result = summarize_immuno_context(
        pd.DataFrame(rows),
        pd.DataFrame({"gene_symbol": ["GENE1"]}),
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
        recurrence_stratum_id="ko_tumor_immune_killing_negative_tail",
        target_absence_attested=True,
    )
    row = result.summary.iloc[0]
    assert set(result.rank_list_audit["status"]) == {"verified_full_list"}
    assert row["report_only_immuno_rra_eligible"]
    assert row["report_only_immuno_rra_independent_family_n"] == 2
    assert 0.0 <= row["report_only_immuno_rra_pvalue"] <= 1.0


def test_rra_abstains_when_self_exclusion_is_unverified() -> None:
    rows = [
        *_full_rank_list("list-1", "SF1", "RF1", 1),
        *_full_rank_list("list-2", "SF2", "RF2", 2),
    ]
    result = summarize_immuno_context(
        pd.DataFrame(rows),
        pd.DataFrame({"gene_symbol": ["GENE1"]}),
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
        recurrence_stratum_id="ko_tumor_immune_killing_negative_tail",
    )
    row = result.summary.iloc[0]
    assert not row["report_only_immuno_rra_eligible"]
    assert row["report_only_immuno_rra_reason"] == "self_exclusion_unverified"


def test_unknown_self_family_exclusion_is_rejected() -> None:
    with pytest.raises(ValueError, match="absent from the evidence table"):
        summarize_immuno_context(
            pd.DataFrame([_record("one")]),
            pd.DataFrame({"gene_symbol": ["GENE1"]}),
            cutoff_date=date(2025, 12, 31),
            target_modality="CRISPR_KO",
            excluded_raw_data_families=["TYPO_NOT_PRESENT"],
        )


def test_rra_rejects_mixed_analysis_tails() -> None:
    first = _full_rank_list("list-1", "SF1", "RF1", 1)
    second = _full_rank_list("list-2", "SF2", "RF2", 2)
    for row in second:
        row["analysis_tail"] = "positive"
    result = summarize_immuno_context(
        pd.DataFrame([*first, *second]),
        pd.DataFrame({"gene_symbol": ["GENE1"]}),
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
        recurrence_stratum_id="ko_tumor_immune_killing_negative_tail",
        target_absence_attested=True,
    )
    row = result.summary.iloc[0]
    assert not row["report_only_immuno_rra_eligible"]
    assert "incompatible_rank_list_stratum" in row["report_only_immuno_rra_reason"]


def test_rra_rejects_reused_stratum_id_for_different_contrast() -> None:
    first = _full_rank_list("list-1", "SF1", "RF1", 1)
    second = _full_rank_list("list-2", "SF2", "RF2", 2)
    for row in second:
        row.update(
            {
                "treatment": "anti-PD1",
                "comparator": "isotype",
                "contrast_definition": "anti-PD1 versus isotype",
                "phenotype_endpoint": "whole-tumor abundance",
                "timepoint": "day 28",
                "cell_model": "MC38",
                "cancer_type": "colon cancer",
            }
        )
    result = summarize_immuno_context(
        pd.DataFrame([*first, *second]),
        pd.DataFrame({"gene_symbol": ["GENE1"]}),
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
        recurrence_stratum_id="ko_tumor_immune_killing_negative_tail",
        target_absence_attested=True,
    )
    row = result.summary.iloc[0]
    assert not row["report_only_immuno_rra_eligible"]
    assert "treatment" in row["report_only_immuno_rra_reason"]


def test_rra_requires_candidate_in_every_verified_list() -> None:
    first = _full_rank_list("list-1", "SF1", "RF1", 1)
    second = _full_rank_list("list-2", "SF2", "RF2", 2)
    third = _full_rank_list("list-3", "SF3", "RF3", 3)
    for row in third:
        if row["gene_symbol"] == "GENE1":
            row["gene_symbol"] = "GENE4"
    roster = "".join(
        f"{row['gene_symbol']}\t{int(row['source_rank'])}\n"
        for row in sorted(third, key=lambda row: int(row["source_rank"]))
    )
    checksum = hashlib.sha256(roster.encode("utf-8")).hexdigest()
    for row in third:
        row["rank_list_sha256"] = checksum
    result = summarize_immuno_context(
        pd.DataFrame([*first, *second, *third]),
        pd.DataFrame({"gene_symbol": ["GENE1"]}),
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
        recurrence_stratum_id="ko_tumor_immune_killing_negative_tail",
        target_absence_attested=True,
    )
    row = result.summary.iloc[0]
    assert not row["report_only_immuno_rra_eligible"]
    assert row["report_only_immuno_rra_verified_list_n"] == 3
    assert "missing_or_ambiguous" in row["report_only_immuno_rra_reason"]


def test_rank_list_checksum_is_verified_from_canonical_roster() -> None:
    rows = _full_rank_list("list-1", "SF1", "RF1", 1)
    rows[0]["rank_list_sha256"] = "0" * 64
    result = summarize_immuno_context(
        pd.DataFrame(rows),
        pd.DataFrame({"gene_symbol": ["GENE1"]}),
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
        recurrence_stratum_id="ko_tumor_immune_killing_negative_tail",
        target_absence_attested=True,
    )
    assert result.rank_list_audit.iloc[0]["status"] == "ineligible"
    assert "checksum" in result.rank_list_audit.iloc[0]["reason"]


def test_one_failed_declared_full_list_blocks_the_selected_stratum() -> None:
    first = _full_rank_list("list-1", "SF1", "RF1", 1)
    second = _full_rank_list("list-2", "SF2", "RF2", 2)
    failed = _full_rank_list("list-3", "SF3", "RF3", 3)
    for row in failed:
        row["rank_list_sha256"] = "0" * 64
    result = summarize_immuno_context(
        pd.DataFrame([*first, *second, *failed]),
        pd.DataFrame({"gene_symbol": ["GENE1"]}),
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
        recurrence_stratum_id="ko_tumor_immune_killing_negative_tail",
        target_absence_attested=True,
    )
    row = result.summary.iloc[0]
    assert not row["report_only_immuno_rra_eligible"]
    assert "failed_declared_full_list" in row["report_only_immuno_rra_reason"]


def test_claimed_full_list_with_missing_gene_abstains() -> None:
    rows = _full_rank_list("list-1", "SF1", "RF1", 1)[:2]
    result = summarize_immuno_context(
        pd.DataFrame(rows),
        pd.DataFrame({"gene_symbol": ["GENE1"]}),
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
        recurrence_stratum_id="ko_tumor_immune_killing_negative_tail",
        target_absence_attested=True,
    )
    row = result.summary.iloc[0]
    assert result.rank_list_audit.iloc[0]["status"] == "ineligible"
    assert not row["report_only_immuno_rra_eligible"]
    assert row["report_only_immuno_rra_pvalue"] is None


def test_rra_rejects_duplicate_canonical_human_gene_roster() -> None:
    rows = [
        *_full_rank_list("list-1", "SF1", "RF1", 1),
        *_full_rank_list("list-2", "SF2", "RF2", 2),
    ]
    for list_id in ("list-1", "list-2"):
        list_rows = [row for row in rows if row["rank_list_id"] == list_id]
        for row in list_rows:
            if row["gene_symbol"] == "GENE2":
                row["gene_symbol"] = "gene1"
        roster = "".join(
            f"{row['gene_symbol']}\t{int(row['source_rank'])}\n"
            for row in sorted(list_rows, key=lambda row: int(row["source_rank"]))
        )
        checksum = hashlib.sha256(roster.encode("utf-8")).hexdigest()
        for row in list_rows:
            row["rank_list_sha256"] = checksum
    result = summarize_immuno_context(
        pd.DataFrame(rows),
        pd.DataFrame({"gene_symbol": ["GENE3"]}),
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
        recurrence_stratum_id="ko_tumor_immune_killing_negative_tail",
        target_absence_attested=True,
    )
    assert result.rank_list_audit["status"].eq("ineligible").all()
    assert (
        result.rank_list_audit["reason"]
        .str.contains("duplicate_canonical_human_gene_in_rank_roster")
        .all()
    )
    assert not result.summary.iloc[0]["report_only_immuno_rra_eligible"]


def test_order_statistic_reference_vector() -> None:
    assert order_statistic_rra_pvalue([0.1, 0.2]) == pytest.approx(0.08)


def test_rra_method_is_explicitly_named_as_unadjusted_baseline() -> None:
    result = summarize_immuno_context(
        pd.DataFrame([_record("one")]),
        pd.DataFrame({"gene_symbol": ["GENE1"]}),
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
    )
    assert result.metadata["rra_method"] == ("order_statistic_baseline_v1_unadjusted")


def test_report_frames_contain_no_validation_or_label_fields() -> None:
    result = summarize_immuno_context(
        pd.DataFrame([_record("one")]),
        pd.DataFrame({"gene_symbol": ["GENE1"]}),
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
    )
    forbidden = ("label", "validation", "testing", "benchmark", "outcome")
    for frame in (
        result.summary,
        result.exclusions,
        result.used_evidence,
        result.rank_list_audit,
    ):
        assert not any(
            token in column.lower() for column in frame.columns for token in forbidden
        )


def test_report_only_columns_are_blocked_from_success_model() -> None:
    with pytest.raises(ValueError, match="leakage fields"):
        validate_success_feature_columns(
            ["guide_n", "report_only_immuno_dual_action_class"]
        )


def test_cli_exposes_immune_context_command() -> None:
    args = build_parser().parse_args(
        [
            "summarize-immuno-context",
            "--evidence",
            "evidence.tsv",
            "--candidates",
            "candidates.tsv",
            "--cutoff-date",
            "2025-12-31",
            "--target-modality",
            "CRISPR_KO",
            "--output-dir",
            "out",
        ]
    )
    assert args.command == "summarize-immuno-context"


def test_cli_preserves_numeric_looking_string_metadata(tmp_path) -> None:
    evidence_path = tmp_path / "evidence.tsv"
    candidates_path = tmp_path / "candidates.tsv"
    output_dir = tmp_path / "out"
    pd.DataFrame([_record("one", source_version="1")]).to_csv(
        evidence_path,
        sep="\t",
        index=False,
    )
    pd.DataFrame(
        {
            "gene_symbol": ["GENE1"],
            "screen_id": ["0007"],
            "contrast_id": ["0009"],
            "phenotype_direction": ["resistance"],
            "analysis_tail": ["mageck_pos"],
            "screen_signal_rank": [1],
            "ranking_type": ["screen_signal_baseline"],
        }
    ).to_csv(
        candidates_path,
        sep="\t",
        index=False,
    )
    args = build_parser().parse_args(
        [
            "summarize-immuno-context",
            "--evidence",
            str(evidence_path),
            "--candidates",
            str(candidates_path),
            "--cutoff-date",
            "2025-12-31",
            "--target-modality",
            "CRISPR_KO",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert args.func(args) == 0
    assert (output_dir / "immune_context.tsv").is_file()
    assert (output_dir / "immune_context_used_evidence.tsv").is_file()
    summary = pd.read_csv(
        output_dir / "immune_context.tsv",
        sep="\t",
        dtype={"screen_id": "string", "contrast_id": "string"},
    )
    assert summary["screen_id"].tolist() == ["0007"]
    assert summary["contrast_id"].tolist() == ["0009"]
    raw_manifest = json.loads((output_dir / "summary.json").read_text())
    for output in raw_manifest["outputs"].values():
        assert (
            hashlib.sha256((output_dir / output["filename"]).read_bytes()).hexdigest()
            == output["sha256"]
        )
    manifest = pd.read_json(output_dir / "summary.json", typ="series")
    assert len(manifest["inputs"]["evidence"]["sha256"]) == 64


def test_existing_immune_bundle_is_preserved(tmp_path) -> None:
    evidence_path = tmp_path / "evidence.tsv"
    candidates_path = tmp_path / "candidates.tsv"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    pd.DataFrame([_record("one")]).to_csv(evidence_path, sep="\t", index=False)
    pd.DataFrame({"gene_symbol": ["GENE1"]}).to_csv(
        candidates_path,
        sep="\t",
        index=False,
    )
    args = build_parser().parse_args(
        [
            "summarize-immuno-context",
            "--evidence",
            str(evidence_path),
            "--candidates",
            str(candidates_path),
            "--cutoff-date",
            "2025-12-31",
            "--target-modality",
            "CRISPR_KO",
            "--output-dir",
            str(output_dir),
        ]
    )
    with pytest.raises(FileExistsError, match="output directory exists"):
        args.func(args)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_unknown_candidate_is_retained_as_missing_context() -> None:
    result = summarize_immuno_context(
        pd.DataFrame([_record("one")]),
        pd.DataFrame({"gene_symbol": ["UNKNOWN1"]}),
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
    )
    row = result.summary.iloc[0]
    assert row["gene_symbol"] == "UNKNOWN1"
    assert not bool(row["report_only_immuno_context_available"])
    assert row["report_only_immuno_dual_action_class"] == "not_assessed"


def test_in_vivo_evidence_is_reported_by_perturbed_compartment() -> None:
    tumor = _record(
        "tumor-in-vivo",
        experimental_setting="in_vivo",
        cell_model="xenograft tumor cells",
    )
    immune = _record(
        "immune-in-vivo",
        source_family_id="SF2",
        raw_data_family_id="RF2",
        perturbed_compartment="immune_cell",
        experimental_setting="in_vivo",
        screen_category="immune_cell_function",
        cell_model="adoptive T cells",
        immune_cell_type="CD8 T cell",
        native_effect_direction="enriched",
        endpoint_polarity="enrichment_is_favorable",
        assay_consequence="immune_effector_gain",
    )
    result = summarize_immuno_context(
        pd.DataFrame([tumor, immune]),
        pd.DataFrame({"gene_symbol": ["GENE1"]}),
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
    )
    row = result.summary.iloc[0]
    assert bool(row["report_only_immuno_in_vivo_mixed_context"])
    assert row["report_only_immuno_tumor_in_vivo_family_n"] == 1
    assert row["report_only_immuno_immune_in_vivo_family_n"] == 1
    assert row["report_only_immuno_tumor_in_vivo_favorable_fraction"] == 1.0
    assert row["report_only_immuno_immune_in_vivo_favorable_fraction"] == 1.0


def test_duplicate_candidate_rows_are_collapsed_to_gene_query() -> None:
    result = summarize_immuno_context(
        pd.DataFrame([_record("one")]),
        pd.DataFrame({"gene_symbol": ["GENE1", "GENE1"]}),
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
    )
    assert result.summary["gene_symbol"].tolist() == ["GENE1"]


def test_rank_screen_candidate_axes_are_preserved_in_immune_output() -> None:
    mageck = pd.DataFrame(
        {
            "id": ["GENE1", "GENE2"],
            "pos|score": [0.01, 0.2],
            "pos|fdr": [0.02, 0.3],
            "pos|rank": [1, 2],
            "pos|lfc": [1.5, 0.2],
            "neg|score": [0.2, 0.01],
            "neg|fdr": [0.3, 0.02],
            "neg|rank": [2, 1],
            "neg|lfc": [-0.2, -1.5],
        }
    )
    ranked = rank_screen(
        mageck_summary=mageck,
        screen_id="S1",
        contrast_id="olaparib_vs_vehicle",
        positive_tail_means="resistance",
    ).ranked_candidates
    immune = _record(
        "immune",
        source_family_id="SF2",
        raw_data_family_id="RF2",
        perturbed_compartment="immune_cell",
        screen_category="immune_cell_function",
        native_effect_direction="enriched",
        endpoint_polarity="enrichment_is_favorable",
        assay_consequence="immune_effector_gain",
    )
    result = summarize_immuno_context(
        pd.DataFrame([_record("tumor"), immune]),
        ranked,
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
        dual_action_group_id="ko_antitumor_function",
        dual_action_group_version="synthetic-v1",
    )
    assert len(result.summary) == 4
    identity = [
        "screen_id",
        "contrast_id",
        "gene_symbol",
        "phenotype_direction",
        "analysis_tail",
    ]
    expected = ranked.sort_values(identity).reset_index(drop=True)
    actual = (
        result.summary[list(ranked.columns)]
        .sort_values(identity)
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    gene1 = result.summary.loc[result.summary["gene_symbol"].eq("GENE1")]
    assert set(gene1["phenotype_direction"]) == {"resistance", "sensitization"}
    relation = gene1.set_index("phenotype_direction")[
        "report_only_immuno_primary_axis_relation"
    ]
    assert relation.eq("axes_reported_separately").all()


def test_combined_rank_screen_output_round_trips_into_immune_context() -> None:
    mageck = pd.DataFrame(
        {
            "id": ["GENE1", "GENE2"],
            "pos|score": [0.01, 0.2],
            "pos|fdr": [0.02, 0.3],
            "pos|rank": [1, 2],
            "pos|lfc": [1.5, 0.2],
            "neg|score": [0.2, 0.01],
            "neg|fdr": [0.3, 0.02],
            "neg|rank": [2, 1],
            "neg|lfc": [-0.2, -1.5],
        }
    )
    counts = pd.DataFrame(
        {
            "sgrna_id": ["g1", "g2", "g3", "g4"],
            "gene_symbol": ["GENE1", "GENE1", "GENE2", "GENE2"],
            "c1": [100, 120, 100, 120],
            "c2": [110, 115, 110, 115],
            "t1": [400, 360, 25, 30],
            "t2": [380, 350, 30, 35],
        }
    )
    samples = pd.DataFrame(
        {
            "sample_id": ["c1", "c2", "t1", "t2"],
            "screen_id": ["S1"] * 4,
            "contrast_id": ["C1"] * 4,
            "condition_role": ["control", "control", "treatment", "treatment"],
            "replicate": [1, 2, 1, 2],
        }
    )
    ranked = rank_screen(
        mageck_summary=mageck,
        counts=counts,
        samples=samples,
        screen_id="S1",
        contrast_id="C1",
        positive_tail_means="resistance",
        positive_lfc_means="resistance",
    ).ranked_candidates
    result = summarize_immuno_context(
        pd.DataFrame([_record("tumor")]),
        ranked,
        cutoff_date=date(2025, 12, 31),
        target_modality="CRISPR_KO",
    )
    identity = [
        "screen_id",
        "contrast_id",
        "gene_symbol",
        "phenotype_direction",
        "analysis_tail",
    ]
    expected = ranked.sort_values(identity).reset_index(drop=True)
    actual = (
        result.summary[list(ranked.columns)]
        .sort_values(identity)
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    gene1 = result.summary.loc[result.summary["gene_symbol"].eq("GENE1")]
    assert (
        gene1["report_only_immuno_primary_axis_relation"]
        .eq("axes_reported_separately")
        .all()
    )


def test_external_screen_cannot_claim_two_provenance_identities() -> None:
    first = _record("one", external_screen_id="SCREEN")
    second = _record(
        "two",
        external_screen_id="SCREEN",
        source_family_id="SF2",
        raw_data_family_id="RF2",
    )
    with pytest.raises(ValueError, match="multiple provenance identities"):
        summarize_immuno_context(
            pd.DataFrame([first, second]),
            pd.DataFrame({"gene_symbol": ["GENE1"]}),
            cutoff_date=date(2025, 12, 31),
            target_modality="CRISPR_KO",
        )


def test_external_screen_cannot_claim_two_studies() -> None:
    first = _record("one", external_screen_id="SCREEN")
    second = _record(
        "two",
        external_screen_id="SCREEN",
        external_study_id="OTHER-STUDY",
        source_family_id=first["source_family_id"],
        raw_data_family_id=first["raw_data_family_id"],
    )
    with pytest.raises(ValueError, match="multiple provenance identities"):
        summarize_immuno_context(
            pd.DataFrame([first, second]),
            pd.DataFrame({"gene_symbol": ["GENE1"]}),
            cutoff_date=date(2025, 12, 31),
            target_modality="CRISPR_KO",
        )


def test_external_study_cannot_claim_two_source_families() -> None:
    first = _record("one", external_study_id="STUDY")
    second = _record(
        "two",
        external_study_id="STUDY",
        source_family_id="OTHER-SOURCE-FAMILY",
        raw_data_family_id="OTHER-RAW-FAMILY",
    )
    with pytest.raises(ValueError, match="external study.*multiple provenance"):
        summarize_immuno_context(
            pd.DataFrame([first, second]),
            pd.DataFrame({"gene_symbol": ["GENE1"]}),
            cutoff_date=date(2025, 12, 31),
            target_modality="CRISPR_KO",
        )


def test_external_study_identity_is_stable_across_source_versions() -> None:
    first = _record(
        "one",
        source_version="v1",
        external_study_id="STUDY",
    )
    second = _record(
        "two",
        source_version="v2",
        external_study_id="STUDY",
        source_family_id="OTHER-SOURCE-FAMILY",
        raw_data_family_id="OTHER-RAW-FAMILY",
    )
    with pytest.raises(ValueError, match="external study.*multiple provenance"):
        summarize_immuno_context(
            pd.DataFrame([first, second]),
            pd.DataFrame({"gene_symbol": ["GENE1"]}),
            cutoff_date=date(2025, 12, 31),
            target_modality="CRISPR_KO",
        )


def test_candidate_screen_axes_are_validated_before_join() -> None:
    candidates = pd.DataFrame(
        {
            "gene_symbol": ["GENE1"],
            "screen_id": ["S1"],
            "contrast_id": ["C1"],
            "phenotype_direction": ["resistance"],
            "analysis_tail": ["mageck_pos"],
            "screen_signal_percentile": [1.5],
            "ranking_type": ["screen_signal_baseline"],
        }
    )
    with pytest.raises(ValueError, match="at most 1"):
        summarize_immuno_context(
            pd.DataFrame([_record("one")]),
            candidates,
            cutoff_date=date(2025, 12, 31),
            target_modality="CRISPR_KO",
        )
