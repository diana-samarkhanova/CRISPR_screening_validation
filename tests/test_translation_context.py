import hashlib
import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import crispr_evidencerank.cli as cli_module
import crispr_evidencerank.translation_context as translation_context_module
from crispr_evidencerank.cli import build_parser
from crispr_evidencerank.contracts import (
    ClinicalTrialContextRecord,
    PatientMolecularEvidenceRecord,
    PreclinicalEvidenceRecord,
    TreatmentDiseaseContextRecord,
)
from crispr_evidencerank.modeling import validate_success_feature_columns
from crispr_evidencerank.translation_context import (
    ClinicalTrialsSnapshot,
    TranslationContextError,
    build_translation_context_report,
    clinical_trials_snapshot_from_document,
    fetch_clinical_trials_concept_v2,
    fetch_clinical_trials_v2,
    normalize_clinical_trials,
)


def _context(**updates):
    record = {
        "context_id": "CTX-OLAPARIB-TNBC",
        "screen_id": "0007",
        "contrast_id": "0009",
        "treatment_name": "olaparib",
        "treatment_id": "NCIT:C71721",
        "treatment_ontology_name": "NCIt",
        "treatment_ontology_version": "26.07d",
        "treatment_modality": "small_molecule",
        "regimen_name": "olaparib monotherapy",
        "regimen_active_exposure_ids_json": '["NCIT:C71721"]',
        "regimen_component_relation": "fixed_all_of",
        "regimen_active_exposures_verified": True,
        "regimen_active_exposure_identifier_source": "NCIt",
        "regimen_active_exposure_identifier_version": "26.07d",
        "cancer_type": "breast cancer",
        "cancer_id": "NCIT:C4872",
        "disease_subtype": "triple-negative breast cancer",
        "disease_subtype_id": "NCIT:C71732",
        "disease_subtype_parent_id": "NCIT:C4872",
        "disease_subtype_parent_binding_verified": True,
        "disease_ontology_name": "NCIt",
        "disease_ontology_version": "26.07d",
        "stage": "advanced",
        "biomarker_context": "BRCA1",
        "biomarker_feature_type": "genomic_mutation",
        "biomarker_state": "pathogenic_or_loss",
        "biomarker_specimen_type": "tumor",
        "biomarker_measurement_timepoint": "pretreatment",
        "biomarker_axes_informative_verified": True,
        "biomarker_axes_observation_status": "observed",
        "line_of_therapy": "second_line",
        "screen_perturbation_modality": "CRISPR_KO",
        "perturbed_compartment": "tumor_cell",
        "screen_endpoint_category": "drug_response_viability",
        "context_date": "2026-08-28",
    }
    record.update(updates)
    if record["regimen_name"] is None:
        record["regimen_active_exposure_ids_json"] = None
        record["regimen_component_relation"] = None
        record["regimen_active_exposures_verified"] = None
        record["regimen_active_exposure_identifier_source"] = None
        record["regimen_active_exposure_identifier_version"] = None
    if record["biomarker_context"] is None:
        record["biomarker_axes_observation_status"] = None
    if record["disease_subtype"] is None:
        record["disease_subtype_id"] = None
        record["disease_subtype_parent_id"] = None
        record["disease_subtype_parent_binding_verified"] = None
    if record["disease_ontology_name"] is None:
        record["cancer_id"] = None
        if (
            record["disease_subtype_id"] is None
            and record["disease_subtype"] is not None
        ):
            record["disease_subtype_parent_id"] = None
            record["disease_subtype_parent_binding_verified"] = False
    return TreatmentDiseaseContextRecord.model_validate(record)


def _trial(
    nct_id,
    *,
    interventions,
    conditions,
    keywords=None,
    has_results=False,
):
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": nct_id,
                "briefTitle": f"Study {nct_id}",
                "officialTitle": f"Official {nct_id}",
            },
            "statusModule": {
                "overallStatus": "COMPLETED",
                "startDateStruct": {"date": "2021-01"},
                "completionDateStruct": {"date": "2024-01"},
                "studyFirstPostDateStruct": {"date": "2020-12-01"},
                "lastUpdatePostDateStruct": {"date": "2025-01-01"},
            },
            "designModule": {
                "studyType": "INTERVENTIONAL",
                "phases": ["PHASE2"],
                "enrollmentInfo": {"count": 42, "type": "ACTUAL"},
            },
            "conditionsModule": {
                "conditions": conditions,
                "keywords": keywords or [],
            },
            "armsInterventionsModule": {"interventions": interventions},
            "outcomesModule": {
                "primaryOutcomes": [{"measure": "Objective response rate"}]
            },
            "referencesModule": {
                "references": [{"pmid": "12345678", "type": "RESULT"}]
            },
        },
        "hasResults": has_results,
    }


def _snapshot_document(studies, *, complete=True):
    return {
        "source": "ClinicalTrials.gov",
        "source_api_major": "v2",
        "api_version": "2.0.5",
        "data_timestamp": "2026-08-28T09:00:06Z",
        "version_stable": True,
        "retrieved_at_utc": "2026-08-28T12:00:00+00:00",
        "request_urls": [
            "https://clinicaltrials.gov/api/v2/studies?query.intr=olaparib"
        ],
        "total_count": len(studies),
        "complete": complete,
        "pages": [{"studies": studies, "totalCount": len(studies)}],
    }


def _trial_set():
    return [
        _trial(
            "NCT00000001",
            interventions=[
                {"type": "DRUG", "name": "Olaparib", "otherNames": ["AZD2281"]},
                {"type": "BIOLOGICAL", "name": "Durvalumab"},
            ],
            conditions=["Triple-Negative Breast Cancer", "Breast Cancer"],
            keywords=["BRCA1"],
        ),
        _trial(
            "NCT00000002",
            interventions=[{"type": "DRUG", "name": "Lynparza", "otherNames": []}],
            conditions=["Breast Cancer"],
            has_results=True,
        ),
        _trial(
            "NCT00000003",
            interventions=[{"type": "DRUG", "name": "Talazoparib"}],
            conditions=["Triple Negative Breast Cancer"],
        ),
    ]


def _preclinical_record(evidence_id="P1", **updates):
    record = {
        "evidence_id": evidence_id,
        "source_study_id": "PMID:1",
        "source_family_id": "SOURCE-1",
        "raw_data_family_id": "RAW-1",
        "evidence_scope": "gene_specific",
        "claim_type": "direct_perturbational_interaction",
        "gene_symbol": "LAMTOR2",
        "gene_id": "SYN:LAMTOR2",
        "gene_identifier_source": "synthetic",
        "gene_identifier_version": "v1",
        "gene_identity_curator_verified": True,
        "perturbation_modality": "CRISPR_KO",
        "perturbed_compartment": "tumor_cell",
        "endpoint_category": "drug_response_viability",
        "phenotype_direction": "resistance",
        "treatment_name": "olaparib",
        "treatment_id": "NCIT:C71721",
        "treatment_ontology_name": "NCIt",
        "treatment_ontology_version": "26.07d",
        "regimen_name": "olaparib monotherapy",
        "regimen_active_exposure_ids_json": '["NCIT:C71721"]',
        "regimen_component_relation": "fixed_all_of",
        "regimen_active_exposures_verified": True,
        "regimen_active_exposure_identifier_source": "NCIt",
        "regimen_active_exposure_identifier_version": "26.07d",
        "comparator": "vehicle",
        "comparator_exposure_type": "vehicle",
        "comparator_active_exposure_ids_json": "[]",
        "comparator_regimen_component_relation": "none",
        "cancer_type": "breast cancer",
        "cancer_id": "NCIT:C4872",
        "disease_subtype": "triple-negative breast cancer",
        "disease_subtype_id": "NCIT:C71732",
        "disease_subtype_parent_id": "NCIT:C4872",
        "disease_subtype_parent_binding_verified": True,
        "disease_ontology_name": "NCIt",
        "disease_ontology_version": "26.07d",
        "biomarker_context": "BRCA1",
        "biomarker_feature_type": "genomic_mutation",
        "biomarker_state": "pathogenic_or_loss",
        "biomarker_specimen_type": "tumor",
        "biomarker_measurement_timepoint": "pretreatment",
        "biomarker_axes_informative_verified": True,
        "biomarker_axes_observation_status": "observed",
        "model_type": "cell_line_2d",
        "model_name": "MDA-MB-468",
        "organism": "human",
        "endpoint": "clonogenic survival",
        "outcome_text": "KO increased survival during olaparib treatment",
        "vehicle_or_baseline_control_present": True,
        "genotype_by_treatment_tested": True,
        "direction_rule_id": "ko_treatment_interaction_v1",
        "direction_rule_version": "1.0.0",
        "direction_inference_status": "direction_supported",
        "direction_inference_curator_verified": True,
        "native_effect": 0.7,
        "native_effect_type": "interaction_log2_fc",
        "native_reference_group": "NTC vehicle",
        "effect_numeric": 0.7,
        "effect_type": "interaction_log2_fc",
        "p_value": 0.01,
        "sample_n": 3,
        "source_name": "Primary paper",
        "source_version": "v1",
        "source_url": "https://example.org/preclinical",
        "source_locator": "Figure 2",
        "available_date": "2025-01-01",
        "retrieved_date": "2026-08-28",
        "used_for_label": False,
    }
    record.update(updates)
    if record["regimen_name"] is None:
        record["regimen_active_exposure_ids_json"] = None
        record["regimen_component_relation"] = None
        record["regimen_active_exposures_verified"] = None
        record["regimen_active_exposure_identifier_source"] = None
        record["regimen_active_exposure_identifier_version"] = None
    if record["biomarker_context"] is None:
        record["biomarker_axes_observation_status"] = None
    if record["disease_subtype"] is None:
        record["disease_subtype_id"] = None
        record["disease_subtype_parent_id"] = None
        record["disease_subtype_parent_binding_verified"] = None
    if record["disease_ontology_name"] is None:
        record["cancer_id"] = None
        if (
            record["disease_subtype_id"] is None
            and record["disease_subtype"] is not None
        ):
            record["disease_subtype_parent_id"] = None
            record["disease_subtype_parent_binding_verified"] = False
    if record["gene_symbol"] is None:
        record["gene_id"] = None
        record["gene_identifier_source"] = None
        record["gene_identifier_version"] = None
        record["gene_identity_curator_verified"] = None
    return record


def _patient_record(evidence_id="C1", **updates):
    record = {
        "evidence_id": evidence_id,
        "source_study_id": "PMID:2",
        "cohort_id": "COHORT-1",
        "source_family_id": "PATIENT-SOURCE-1",
        "raw_data_family_id": "PATIENT-RAW-1",
        "gene_symbol": "NME6",
        "predictor_gene_symbol": "NME6",
        "gene_id": "SYN:NME6",
        "gene_identifier_source": "synthetic",
        "gene_identifier_version": "v1",
        "predictor_feature_type": "rna_expression",
        "predictor_state": "high",
        "predictor_specimen_type": "tumor",
        "predictor_identity_curator_verified": True,
        "treatment_name": "olaparib",
        "treatment_id": "NCIT:C71721",
        "treatment_ontology_name": "NCIt",
        "treatment_ontology_version": "26.07d",
        "regimen_name": "olaparib monotherapy",
        "comparator_regimen_name": "placebo",
        "treatment_assignment_id": "ARM-OLAPARIB",
        "comparator_assignment_id": "ARM-PHYSICIAN-CHOICE",
        "treatment_active_exposure_ids_json": '["NCIT:C71721"]',
        "comparator_active_exposure_ids_json": "[]",
        "treatment_regimen_component_relation": "fixed_all_of",
        "comparator_regimen_component_relation": "none",
        "comparator_exposure_type": "placebo",
        "active_exposure_identifier_source": "NCIt",
        "active_exposure_identifier_version": "26.07d",
        "active_exposure_ids_curator_verified": True,
        "cancer_type": "breast cancer",
        "cancer_id": "NCIT:C4872",
        "disease_subtype": "triple-negative breast cancer",
        "disease_subtype_id": "NCIT:C71732",
        "disease_subtype_parent_id": "NCIT:C4872",
        "disease_subtype_parent_binding_verified": True,
        "disease_ontology_name": "NCIt",
        "disease_ontology_version": "26.07d",
        "stage": "advanced",
        "line_of_therapy": "second_line",
        "biomarker_context": "BRCA1",
        "biomarker_feature_type": "genomic_mutation",
        "biomarker_state": "pathogenic_or_loss",
        "biomarker_specimen_type": "tumor",
        "biomarker_measurement_timepoint": "pretreatment",
        "biomarker_axes_informative_verified": True,
        "biomarker_axes_observation_status": "observed",
        "measurement_type": "pretreatment RNA-seq",
        "measurement_platform": "Illumina",
        "measurement_timepoint": "pretreatment",
        "clinical_outcome": "PFS",
        "association_interpretation": "predictive_interaction",
        "comparator_arm_present": True,
        "treatment_predictor_interaction_tested": True,
        "interaction_inference_status": "supported",
        "interaction_effect_scale": "hazard_ratio",
        "interaction_inference_rule_id": "interaction_support_v1",
        "interaction_inference_rule_version": "1.0.0",
        "interaction_significance_threshold": 0.05,
        "interaction_p_value_role": "departure_from_null",
        "interaction_inference_curator_verified": True,
        "treatment_exposure_verified": True,
        "treatment_comparator_exposures_distinct_verified": True,
        "paired_baseline_present": None,
        "longitudinal_change_tested": None,
        "analysis_model": "Cox model with treatment-by-expression interaction",
        "patient_n": 120,
        "treatment_arm_patient_n": 60,
        "comparator_arm_patient_n": 60,
        "interaction_analysis_evaluable_patient_n": 120,
        "interaction_model_parameter_n": 4,
        "interaction_outcome_event_n": 60,
        "interaction_model_estimable_verified": True,
        "biomarker_variation_in_each_arm_verified": True,
        "effect_numeric": 0.72,
        "effect_type": "hazard_ratio",
        "confidence_interval_lower": 0.55,
        "confidence_interval_upper": 0.94,
        "p_value": 0.02,
        "outcome_text": "Pretreatment expression interacted with assigned treatment",
        "source_name": "Clinical cohort",
        "source_version": "v1",
        "source_url": "https://example.org/cohort",
        "source_locator": "Table 3",
        "available_date": "2025-06-01",
        "retrieved_date": "2026-08-28",
        "used_for_label": False,
    }
    record.update(updates)
    if record["biomarker_context"] is None:
        record["biomarker_axes_observation_status"] = None
    if record["disease_subtype"] is None:
        record["disease_subtype_id"] = None
        record["disease_subtype_parent_id"] = None
        record["disease_subtype_parent_binding_verified"] = None
    if record["disease_ontology_name"] is None:
        record["cancer_id"] = None
        if (
            record["disease_subtype_id"] is None
            and record["disease_subtype"] is not None
        ):
            record["disease_subtype_parent_id"] = None
            record["disease_subtype_parent_binding_verified"] = False
    if not record["comparator_arm_present"]:
        for field_name in (
            "comparator_regimen_name",
            "comparator_assignment_id",
            "comparator_active_exposure_ids_json",
            "comparator_regimen_component_relation",
            "comparator_exposure_type",
        ):
            record[field_name] = None
    if not record["treatment_predictor_interaction_tested"]:
        record["interaction_inference_status"] = "not_tested"
        for field_name in (
            "interaction_effect_scale",
            "interaction_inference_rule_id",
            "interaction_inference_rule_version",
            "interaction_significance_threshold",
            "interaction_p_value_role",
            "interaction_equivalence_lower",
            "interaction_equivalence_upper",
            "interaction_inference_curator_verified",
            "treatment_arm_patient_n",
            "comparator_arm_patient_n",
            "interaction_analysis_evaluable_patient_n",
            "interaction_model_parameter_n",
            "interaction_outcome_event_n",
            "interaction_model_estimable_verified",
            "biomarker_variation_in_each_arm_verified",
        ):
            record[field_name] = None
    return record


def test_context_requires_screen_and_contrast_together():
    with pytest.raises(ValueError, match="screen_id and contrast_id"):
        _context(contrast_id=None)


def test_context_ontology_identifiers_are_versioned():
    with pytest.raises(ValueError, match="treatment ontology"):
        _context(treatment_ontology_version=None)
    with pytest.raises(ValueError, match="disease ontology"):
        _context(
            disease_ontology_name=None,
            disease_subtype_id=None,
            disease_subtype_parent_id=None,
            disease_subtype_parent_binding_verified=False,
        )
    with pytest.raises(ValueError, match="biomarker term"):
        _context(biomarker_feature_type=None)


def test_public_report_boundaries_revalidate_existing_context_models():
    mutated = _context()
    mutated.treatment_ontology_version = None
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    with pytest.raises(ValueError, match="treatment ontology"):
        normalize_clinical_trials(snapshot, mutated)
    with pytest.raises(ValueError, match="treatment ontology"):
        build_translation_context_report(
            mutated,
            snapshot,
            evidence_cutoff_date=date(2026, 8, 28),
        )


def test_public_trial_normalizer_rejects_forged_snapshot_dataclass():
    forged = ClinicalTrialsSnapshot(
        document={},
        studies=[_trial_set()[0]],
        request_urls=[],
        total_count=1,
        complete=True,
        retrieved_at_utc=datetime(2026, 8, 28, 12, tzinfo=UTC),
        source_mode="live_api",
        api_version="2.0.5",
        data_timestamp=None,
        version_stable=True,
    )
    with pytest.raises(TranslationContextError):
        normalize_clinical_trials(forged, _context())


def test_public_trial_normalizer_rejects_nested_snapshot_mutation():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    snapshot.studies[0]["hasResults"] = True
    with pytest.raises(
        TranslationContextError,
        match="metadata or studies changed after validation",
    ):
        normalize_clinical_trials(snapshot, _context())


def test_subtype_parent_binding_is_explicit_and_fail_closed():
    with pytest.raises(ValueError, match="parent ID equal to cancer_id"):
        _context(disease_subtype_parent_id="NCIT:C99999")
    with pytest.raises(ValueError, match="parent ID equal to cancer_id"):
        PatientMolecularEvidenceRecord.model_validate(
            _patient_record(disease_subtype_parent_id="NCIT:C99999")
        )

    candidates = pd.DataFrame(
        {
            "gene_symbol": ["LAMTOR2"],
            "screen_id": ["0007"],
            "contrast_id": ["0009"],
            "phenotype_direction": ["resistance"],
        }
    )
    result = build_translation_context_report(
        _context(disease_subtype_parent_binding_verified=False),
        clinical_trials_snapshot_from_document(_snapshot_document(_trial_set())),
        candidates=candidates,
        preclinical_evidence=pd.DataFrame([_preclinical_record()]),
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=True,
    )
    assert (
        result.candidate_context.iloc[0][
            "report_only_preclinical_exact_context_family_n"
        ]
        == 0
    )
    assert (
        result.preclinical_used_evidence.iloc[0]["report_only_subtype_match"]
        == "parent_binding_unverified"
    )


def test_versioned_ontology_identifiers_reject_placeholders_and_self_parenting():
    with pytest.raises(ValueError, match="non-placeholder|CURIE"):
        _context(
            cancer_id="unknown_value",
            disease_subtype_id="NCIT:C71732",
            disease_subtype_parent_id="unknown_value",
            disease_subtype_parent_binding_verified=True,
        )
    with pytest.raises(ValueError, match="subtype ID must differ"):
        _context(
            disease_subtype_id="NCIT:C4872",
            disease_subtype_parent_id="NCIT:C4872",
        )
    with pytest.raises(ValueError, match="CURIE prefix"):
        _context(
            cancer_id="MONDO:0000001",
            disease_subtype_parent_id="MONDO:0000001",
        )

    record = _context().model_dump(mode="json")
    record.update(
        {
            "cancer_id": None,
            "disease_subtype_id": None,
            "disease_subtype_parent_id": "NCIT:C4872",
            "disease_subtype_parent_binding_verified": False,
            "disease_ontology_name": None,
            "disease_ontology_version": None,
        }
    )
    with pytest.raises(ValueError, match="disease ontology"):
        TreatmentDiseaseContextRecord.model_validate(record)


@pytest.mark.parametrize("placeholder", ["not_reported", "unknown_value", "no_data"])
def test_snake_case_missingness_placeholders_fail_closed(placeholder):
    with pytest.raises(ValueError, match="informative|placeholder"):
        _context(biomarker_context=placeholder)
    with pytest.raises(ValueError, match="informative|placeholder"):
        PatientMolecularEvidenceRecord.model_validate(
            _patient_record(predictor_state=placeholder)
        )


def test_patient_predictor_identity_and_measurement_are_gene_bound():
    with pytest.raises(ValueError, match="predictor_gene_symbol"):
        PatientMolecularEvidenceRecord.model_validate(
            _patient_record(gene_symbol="FAKEGENE")
        )
    with pytest.raises(ValueError, match="feature type conflicts"):
        PatientMolecularEvidenceRecord.model_validate(
            _patient_record(predictor_feature_type="genomic_mutation")
        )

    for ambiguous_measurement in (
        "protein expression by IHC",
        "DNA expression panel",
        "journal assay",
    ):
        with pytest.raises(ValueError, match="feature type conflicts"):
            PatientMolecularEvidenceRecord.model_validate(
                _patient_record(
                    measurement_type=ambiguous_measurement,
                    measurement_platform=None,
                )
            )


@pytest.mark.parametrize(
    ("feature_type", "measurement_type", "measurement_platform"),
    [
        ("genomic_mutation", "DNA expression by RNA-seq", "Illumina RNA-seq"),
        ("genomic_mutation", "DNA copy number profiling", "SNP array"),
        ("copy_number", "RNA copy number expression", "RNA-seq"),
        ("copy_number", "copy number and fusion panel", "targeted panel"),
        ("fusion", "copy number fusion panel", "SNP array"),
        ("fusion", "protein fusion expression", "IHC"),
    ],
)
def test_patient_predictor_measurement_rejects_mutually_exclusive_semantics(
    feature_type, measurement_type, measurement_platform
):
    with pytest.raises(ValueError, match="feature type conflicts"):
        PatientMolecularEvidenceRecord.model_validate(
            _patient_record(
                predictor_feature_type=feature_type,
                measurement_type=measurement_type,
                measurement_platform=measurement_platform,
            )
        )


@pytest.mark.parametrize(
    ("feature_type", "measurement_type", "measurement_platform"),
    [
        ("genomic_mutation", "DNA mutation sequencing", "targeted DNA panel"),
        ("copy_number", "DNA copy number profiling", "SNP array"),
        ("fusion", "RNA-seq fusion detection", "Illumina RNA-seq"),
    ],
)
def test_patient_predictor_measurement_accepts_feature_specific_assays(
    feature_type, measurement_type, measurement_platform
):
    PatientMolecularEvidenceRecord.model_validate(
        _patient_record(
            predictor_feature_type=feature_type,
            measurement_type=measurement_type,
            measurement_platform=measurement_platform,
        )
    )


def test_preclinical_gene_identity_is_versioned_and_informative():
    with pytest.raises(ValueError, match="informative"):
        PreclinicalEvidenceRecord.model_validate(
            _preclinical_record(gene_symbol="not_reported")
        )
    with pytest.raises(ValueError, match="curator-attested versioned gene identity"):
        PreclinicalEvidenceRecord.model_validate(_preclinical_record(gene_id=None))


@pytest.mark.parametrize(
    "field",
    [
        "effect_numeric",
        "confidence_interval_lower",
        "confidence_interval_upper",
        "p_value",
        "patient_n",
    ],
)
def test_patient_scientific_numeric_fields_reject_boolean(field):
    with pytest.raises(ValueError, match="cannot be boolean"):
        PatientMolecularEvidenceRecord.model_validate(_patient_record(**{field: True}))


@pytest.mark.parametrize(
    "field",
    [
        "regimen_active_exposures_verified",
        "disease_subtype_parent_binding_verified",
        "biomarker_axes_informative_verified",
    ],
)
def test_context_attestations_reject_boolean_coercion(field):
    record = _context().model_dump(mode="json")
    record[field] = 1
    with pytest.raises(ValueError, match="literal booleans"):
        TreatmentDiseaseContextRecord.model_validate(record)


@pytest.mark.parametrize(
    "field",
    [
        "regimen_active_exposures_verified",
        "gene_identity_curator_verified",
        "disease_subtype_parent_binding_verified",
        "biomarker_axes_informative_verified",
        "vehicle_or_baseline_control_present",
        "genotype_by_treatment_tested",
        "direction_inference_curator_verified",
        "neutrality_or_discordance_rule_prespecified",
        "used_for_label",
    ],
)
def test_preclinical_attestations_reject_boolean_coercion(field):
    with pytest.raises(ValueError, match="literal booleans"):
        PreclinicalEvidenceRecord.model_validate(_preclinical_record(**{field: "true"}))


@pytest.mark.parametrize(
    "field",
    [
        "predictor_identity_curator_verified",
        "active_exposure_ids_curator_verified",
        "disease_subtype_parent_binding_verified",
        "biomarker_axes_informative_verified",
        "comparator_arm_present",
        "treatment_predictor_interaction_tested",
        "interaction_inference_curator_verified",
        "treatment_exposure_verified",
        "treatment_comparator_exposures_distinct_verified",
        "paired_baseline_present",
        "longitudinal_change_tested",
        "progression_documented",
        "interaction_model_estimable_verified",
        "biomarker_variation_in_each_arm_verified",
        "used_for_label",
    ],
)
def test_patient_attestations_reject_boolean_coercion(field):
    with pytest.raises(ValueError, match="literal booleans"):
        PatientMolecularEvidenceRecord.model_validate(_patient_record(**{field: 1}))


def test_normalized_clinical_trial_flags_reject_boolean_coercion():
    record = (
        normalize_clinical_trials(
            clinical_trials_snapshot_from_document(_snapshot_document(_trial_set())),
            _context(),
        )
        .iloc[0]
        .to_dict()
    )
    record["has_results"] = "false"
    with pytest.raises(ValueError, match="literal booleans"):
        ClinicalTrialContextRecord.model_validate(record)


def _normalized_clinical_trial_record():
    return (
        normalize_clinical_trials(
            clinical_trials_snapshot_from_document(_snapshot_document(_trial_set())),
            _context(),
        )
        .iloc[0]
        .to_dict()
    )


@pytest.mark.parametrize(
    ("source_url", "source_api_version"),
    [
        ("https://evil.example/study/NCT00000001", "2.0.5"),
        ("https://clinicaltrials.gov/study/NCT99999999", "2.0.5"),
        ("https://clinicaltrials.gov/study/NCT00000001?spoof=true", "2.0.5"),
        ("https://clinicaltrials.gov/study/NCT00000001", "totally-not-v2"),
    ],
)
def test_clinical_trial_contract_binds_official_url_and_api_version(
    source_url, source_api_version
):
    record = _normalized_clinical_trial_record()
    record.update(source_url=source_url, source_api_version=source_api_version)
    with pytest.raises(ValueError):
        ClinicalTrialContextRecord.model_validate(record)


@pytest.mark.parametrize("source_api_version", ["2", "2.0.5-synthetic", "unverified"])
def test_clinical_trial_contract_accepts_explicit_v2_or_unverified_version(
    source_api_version,
):
    record = _normalized_clinical_trial_record()
    record["source_api_version"] = source_api_version
    ClinicalTrialContextRecord.model_validate(record)


@pytest.mark.parametrize(
    ("match_field", "terms_field", "match_value", "terms_json"),
    [
        (
            "intervention_match",
            "intervention_match_terms_json",
            "exact_canonical",
            "[]",
        ),
        (
            "intervention_match",
            "intervention_match_terms_json",
            "no_structured_match",
            '["Olaparib"]',
        ),
        (
            "disease_match",
            "disease_match_terms_json",
            "explicit_subtype_term",
            "[]",
        ),
        (
            "disease_match",
            "disease_match_terms_json",
            "no_structured_match",
            '["Triple-Negative Breast Cancer"]',
        ),
        (
            "biomarker_match",
            "biomarker_match_terms_json",
            "explicit_structured_term",
            "[]",
        ),
        (
            "biomarker_match",
            "biomarker_match_terms_json",
            "not_reported_in_structured_terms",
            '["BRCA1"]',
        ),
    ],
)
def test_clinical_trial_match_status_requires_consistent_term_list(
    match_field, terms_field, match_value, terms_json
):
    record = _normalized_clinical_trial_record()
    record.update({match_field: match_value, terms_field: terms_json})
    with pytest.raises(ValueError, match="must agree"):
        ClinicalTrialContextRecord.model_validate(record)


@pytest.mark.parametrize("terms_json", ["[null]", '[""]', "[1]"])
def test_clinical_trial_match_terms_must_be_meaningful_strings(terms_json):
    record = _normalized_clinical_trial_record()
    record["intervention_match_terms_json"] = terms_json
    with pytest.raises(ValueError, match="meaningful non-empty string terms"):
        ClinicalTrialContextRecord.model_validate(record)


def test_verified_regimen_labels_cannot_hide_additional_agents():
    with pytest.raises(ValueError, match="canonical monotherapy"):
        _context(regimen_name="olaparib + durvalumab combination")
    with pytest.raises(ValueError, match="canonical monotherapy"):
        PreclinicalEvidenceRecord.model_validate(
            _preclinical_record(regimen_name="olaparib + durvalumab combination")
        )


def test_verified_patient_exposures_are_bound_outside_formal_interactions():
    with pytest.raises(ValueError, match="contain treatment_id"):
        PatientMolecularEvidenceRecord.model_validate(
            _patient_record(
                association_interpretation="treated_cohort_association",
                comparator_arm_present=False,
                treatment_predictor_interaction_tested=False,
                treatment_active_exposure_ids_json='["NCIT:C99999"]',
            )
        )


def test_preclinical_direction_requires_informative_supported_inference():
    with pytest.raises(ValueError, match="stable non-placeholder"):
        PreclinicalEvidenceRecord.model_validate(
            _preclinical_record(direction_rule_id="unknown")
        )
    with pytest.raises(ValueError, match="non-zero effect"):
        PreclinicalEvidenceRecord.model_validate(
            _preclinical_record(native_effect=0.0, effect_numeric=0.0, p_value=1.0)
        )
    with pytest.raises(ValueError, match="matching curator-verified"):
        PreclinicalEvidenceRecord.model_validate(
            _preclinical_record(direction_inference_status="unsupported")
        )
    for field in ("native_effect_type", "native_reference_group", "effect_type"):
        with pytest.raises(ValueError, match="informative observed"):
            PreclinicalEvidenceRecord.model_validate(
                _preclinical_record(**{field: "not_reported"})
            )


def test_synthetic_active_exposure_ontology_cannot_create_exact_regimen_match():
    context = _context(
        treatment_name="synthetic drug",
        treatment_id="SYN:001",
        treatment_ontology_name="synthetic",
        treatment_ontology_version="v1",
        regimen_name="synthetic drug monotherapy",
        regimen_active_exposure_ids_json='["SYN:001"]',
        regimen_active_exposure_identifier_source="synthetic",
        regimen_active_exposure_identifier_version="v1",
    )
    preclinical = _preclinical_record(
        treatment_name="synthetic drug",
        treatment_id="SYN:001",
        treatment_ontology_name="synthetic",
        treatment_ontology_version="v1",
        regimen_name="synthetic drug monotherapy",
        regimen_active_exposure_ids_json='["SYN:001"]',
        regimen_active_exposure_identifier_source="synthetic",
        regimen_active_exposure_identifier_version="v1",
    )
    result = build_translation_context_report(
        context,
        clinical_trials_snapshot_from_document(_snapshot_document(_trial_set())),
        candidates=pd.DataFrame(
            {
                "gene_symbol": ["LAMTOR2"],
                "screen_id": ["0007"],
                "contrast_id": ["0009"],
                "phenotype_direction": ["resistance"],
            }
        ),
        preclinical_evidence=pd.DataFrame([preclinical]),
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=True,
    )
    row = result.candidate_context.iloc[0]
    assert row["report_only_preclinical_exact_regimen_family_n"] == 0
    assert row["report_only_preclinical_compatible_nonexact_context_family_n"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("biomarker_context", "not available"),
        ("biomarker_feature_type", "other"),
        ("biomarker_state", "undetermined"),
        ("biomarker_specimen_type", "not assessed"),
        ("biomarker_measurement_timepoint", "unknown"),
    ],
)
def test_biomarker_informative_attestation_fails_closed(field, value):
    with pytest.raises(ValueError, match="marked informative"):
        _context(**{field: value})
    with pytest.raises(ValueError, match="marked informative"):
        PatientMolecularEvidenceRecord.model_validate(_patient_record(**{field: value}))


def test_biomarker_bundle_requires_explicit_informative_attestation():
    with pytest.raises(ValueError, match="explicit.*informative"):
        _context(biomarker_axes_informative_verified=None)


def test_patient_predictive_claim_requires_comparator_and_interaction():
    with pytest.raises(ValueError, match="comparator arm|comparator metadata"):
        PatientMolecularEvidenceRecord.model_validate(
            _patient_record(comparator_arm_present=False)
        )
    with pytest.raises(ValueError, match="interaction test"):
        PatientMolecularEvidenceRecord.model_validate(
            _patient_record(treatment_predictor_interaction_tested=False)
        )
    with pytest.raises(ValueError, match="measurement_timepoint|pretreatment"):
        PatientMolecularEvidenceRecord.model_validate(
            _patient_record(measurement_timepoint="on_treatment")
        )
    with pytest.raises(ValueError, match="effect_type|numerical interaction effect"):
        PatientMolecularEvidenceRecord.model_validate(
            _patient_record(
                effect_numeric=None,
                effect_type=None,
                confidence_interval_lower=None,
                confidence_interval_upper=None,
                p_value=None,
            )
        )
    with pytest.raises(ValueError, match="distinct treatment and comparator"):
        PatientMolecularEvidenceRecord.model_validate(
            _patient_record(comparator_regimen_name=" OLAPARIB---MONOTHERAPY ")
        )
    with pytest.raises(ValueError, match="canonical monotherapy"):
        PatientMolecularEvidenceRecord.model_validate(
            _patient_record(
                regimen_name="olaparib + durvalumab",
                comparator_regimen_name="durvalumab and olaparib",
            )
        )
    with pytest.raises(ValueError, match="distinct non-empty source assignment"):
        PatientMolecularEvidenceRecord.model_validate(
            _patient_record(comparator_assignment_id=" ARM---OLAPARIB ")
        )
    with pytest.raises(ValueError, match="exposures are distinct"):
        PatientMolecularEvidenceRecord.model_validate(
            _patient_record(
                treatment_comparator_exposures_distinct_verified=False,
            )
        )


def test_patient_comparator_flag_and_metadata_must_agree():
    record = _patient_record()
    record["comparator_arm_present"] = False
    with pytest.raises(ValueError, match="comparator metadata"):
        PatientMolecularEvidenceRecord.model_validate(record)


@pytest.mark.parametrize(
    "updates",
    [
        {
            "p_value": 0.2,
            "confidence_interval_lower": 0.55,
            "confidence_interval_upper": 0.94,
        },
        {
            "p_value": 0.01,
            "effect_numeric": 1.0,
            "confidence_interval_lower": 0.8,
            "confidence_interval_upper": 1.2,
        },
        {"interaction_p_value_role": "equivalence_to_null"},
        {
            "interaction_equivalence_lower": 0.8,
            "interaction_equivalence_upper": 1.25,
        },
    ],
)
def test_supported_interaction_requires_consistent_departure_evidence(updates):
    with pytest.raises(ValueError, match="supported|departure|equivalence bounds"):
        PatientMolecularEvidenceRecord.model_validate(_patient_record(**updates))


def _formal_null_patient(**updates):
    values = {
        "association_interpretation": "interaction_tested_null",
        "interaction_inference_status": "null",
        "interaction_p_value_role": "equivalence_to_null",
        "interaction_equivalence_lower": 0.8,
        "interaction_equivalence_upper": 1.25,
        "effect_numeric": 1.0,
        "confidence_interval_lower": 0.95,
        "confidence_interval_upper": 1.05,
        "p_value": 0.01,
    }
    values.update(updates)
    return _patient_record(**values)


def test_formal_null_interaction_requires_equivalence_evidence():
    parsed = PatientMolecularEvidenceRecord.model_validate(_formal_null_patient())
    assert parsed.interaction_inference_status == "null"
    for updates in (
        {"confidence_interval_lower": 0.5, "confidence_interval_upper": 1.5},
        {"p_value": 0.2},
        {"interaction_p_value_role": "departure_from_null"},
        {"interaction_equivalence_lower": -0.8},
    ):
        with pytest.raises(ValueError, match="formal null|equivalence|positive"):
            PatientMolecularEvidenceRecord.model_validate(
                _formal_null_patient(**updates)
            )


@pytest.mark.parametrize(
    ("status", "interpretation", "p_value", "ci", "accepted"),
    [
        ("inconclusive", "interaction_tested_inconclusive", 0.01, (0.8, 1.2), True),
        ("inconclusive", "interaction_tested_inconclusive", 0.2, (0.7, 0.9), True),
        ("inconclusive", "interaction_tested_inconclusive", 0.01, (0.7, 0.9), False),
        ("inconclusive", "interaction_tested_inconclusive", 0.2, (0.8, 1.2), False),
        ("unsupported", "interaction_tested_unsupported", 0.2, (0.8, 1.2), True),
        ("unsupported", "interaction_tested_unsupported", 0.01, (0.8, 1.2), False),
        ("unsupported", "interaction_tested_unsupported", 0.2, (0.7, 0.9), False),
    ],
)
def test_nonconfirmatory_interaction_statuses_follow_prespecified_rule(
    status, interpretation, p_value, ci, accepted
):
    record = _patient_record(
        association_interpretation=interpretation,
        interaction_inference_status=status,
        effect_numeric=1.0 if ci[0] < 1.0 < ci[1] else 0.8,
        confidence_interval_lower=ci[0],
        confidence_interval_upper=ci[1],
        p_value=p_value,
    )
    if accepted:
        assert PatientMolecularEvidenceRecord.model_validate(record)
    else:
        with pytest.raises(ValueError, match="inconclusive|unsupported"):
            PatientMolecularEvidenceRecord.model_validate(record)


@pytest.mark.parametrize(
    ("status", "interpretation"),
    [
        ("inconclusive", "interaction_tested_inconclusive"),
        ("unsupported", "interaction_tested_unsupported"),
    ],
)
def test_nonnull_interactions_reject_equivalence_semantics(status, interpretation):
    base = {
        "association_interpretation": interpretation,
        "interaction_inference_status": status,
        "effect_numeric": 1.0,
        "confidence_interval_lower": 0.8,
        "confidence_interval_upper": 1.2,
        "p_value": 0.2 if status == "unsupported" else 0.01,
    }
    with pytest.raises(ValueError, match="departure"):
        PatientMolecularEvidenceRecord.model_validate(
            _patient_record(
                **base,
                interaction_p_value_role="equivalence_to_null",
            )
        )
    with pytest.raises(ValueError, match="equivalence bounds"):
        PatientMolecularEvidenceRecord.model_validate(
            _patient_record(
                **base,
                interaction_equivalence_lower=0.8,
                interaction_equivalence_upper=1.25,
            )
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"interaction_analysis_evaluable_patient_n": 119},
        {"patient_n": 100},
        {"interaction_model_parameter_n": 120},
        {"interaction_model_estimable_verified": False},
        {"biomarker_variation_in_each_arm_verified": False},
        {"interaction_outcome_event_n": 4},
        {"interaction_outcome_event_n": 121},
        {
            "interaction_effect_scale": "odds_ratio",
            "effect_type": "odds_ratio",
            "interaction_outcome_event_n": 118,
        },
    ],
)
def test_interaction_counts_and_estimability_fail_closed(updates):
    with pytest.raises(ValueError, match="counts|evaluable|estimab|event|variation"):
        PatientMolecularEvidenceRecord.model_validate(_patient_record(**updates))


def test_untested_interaction_rejects_retained_interaction_counts():
    for field, value in (
        ("treatment_arm_patient_n", 60),
        ("comparator_arm_patient_n", 60),
        ("interaction_analysis_evaluable_patient_n", 120),
        ("interaction_model_parameter_n", 4),
        ("interaction_outcome_event_n", 60),
        ("interaction_model_estimable_verified", True),
        ("biomarker_variation_in_each_arm_verified", True),
    ):
        record = _patient_record(
            association_interpretation="treated_cohort_association",
            comparator_arm_present=False,
            treatment_predictor_interaction_tested=False,
        )
        record[field] = value
        with pytest.raises(ValueError, match="untested interactions"):
            PatientMolecularEvidenceRecord.model_validate(record)


def test_nonformal_patient_exposure_bundles_enforce_relation_and_provenance():
    treated = {
        "association_interpretation": "treated_cohort_association",
        "comparator_arm_present": False,
        "treatment_predictor_interaction_tested": False,
    }
    with pytest.raises(ValueError, match="relation none"):
        PatientMolecularEvidenceRecord.model_validate(
            _patient_record(
                **treated,
                treatment_regimen_component_relation="none",
            )
        )
    with pytest.raises(ValueError, match="must use relation none"):
        PatientMolecularEvidenceRecord.model_validate(
            _patient_record(
                **{**treated, "comparator_arm_present": True},
                comparator_regimen_component_relation="fixed_all_of",
            )
        )
    with pytest.raises(ValueError, match="identifier source|ontology"):
        PatientMolecularEvidenceRecord.model_validate(
            _patient_record(
                **{**treated, "comparator_arm_present": True},
                comparator_regimen_name="active comparator",
                comparator_active_exposure_ids_json='["DRUGBANK:DB00001"]',
                comparator_regimen_component_relation="fixed_all_of",
                comparator_exposure_type="active_therapeutic",
            )
        )
    with pytest.raises(ValueError, match="JSON array"):
        PatientMolecularEvidenceRecord.model_validate(
            _patient_record(
                **treated,
                treatment_active_exposure_ids_json="not-json",
            )
        )


def test_treated_cohort_association_is_not_predictive():
    record = _patient_record(
        association_interpretation="treated_cohort_association",
        comparator_arm_present=False,
        treatment_predictor_interaction_tested=False,
    )
    parsed = PatientMolecularEvidenceRecord.model_validate(record)
    assert parsed.association_interpretation == "treated_cohort_association"


def test_prognostic_predictor_cannot_be_measured_after_treatment():
    with pytest.raises(ValueError, match="prognostic_only requires"):
        PatientMolecularEvidenceRecord.model_validate(
            _patient_record(
                association_interpretation="prognostic_only",
                comparator_arm_present=False,
                treatment_predictor_interaction_tested=False,
                treatment_exposure_verified=False,
                measurement_timepoint="post_treatment",
            )
        )


def test_unverified_patient_treatment_exposure_cannot_be_exact_context():
    record = _patient_record(
        association_interpretation="prognostic_only",
        comparator_arm_present=False,
        treatment_predictor_interaction_tested=False,
        treatment_exposure_verified=False,
    )
    result = build_translation_context_report(
        _context(),
        clinical_trials_snapshot_from_document(_snapshot_document(_trial_set())),
        candidates=pd.DataFrame(
            {
                "gene_symbol": ["NME6"],
                "screen_id": ["0007"],
                "contrast_id": ["0009"],
                "phenotype_direction": ["resistance"],
            }
        ),
        patient_evidence=pd.DataFrame([record]),
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=True,
    )
    row = result.candidate_context.iloc[0]
    assert row["report_only_patient_exact_context_family_n"] == 0
    assert row["report_only_patient_compatible_nonexact_context_family_n"] == 1


def test_directional_preclinical_claim_requires_interaction_controls():
    with pytest.raises(ValueError, match="vehicle or baseline"):
        PreclinicalEvidenceRecord.model_validate(
            _preclinical_record(vehicle_or_baseline_control_present=False)
        )


def test_preclinical_scope_cannot_overstate_gene_specific_claims():
    with pytest.raises(ValueError, match="gene-specific claim"):
        PreclinicalEvidenceRecord.model_validate(
            _preclinical_record(
                evidence_scope="treatment_context",
                claim_type="natural_biomarker_association",
                gene_symbol=None,
                perturbation_modality=None,
                phenotype_direction="unknown",
            )
        )
    with pytest.raises(ValueError, match="treatment_context scope"):
        PreclinicalEvidenceRecord.model_validate(
            _preclinical_record(
                claim_type="treatment_activity_only",
                perturbation_modality=None,
                phenotype_direction="unknown",
            )
        )
    with pytest.raises(ValueError, match="genotype-by-treatment"):
        PreclinicalEvidenceRecord.model_validate(
            _preclinical_record(genotype_by_treatment_tested=False)
        )
    with pytest.raises(ValueError, match="non-perturbational"):
        PreclinicalEvidenceRecord.model_validate(
            _preclinical_record(
                claim_type="natural_biomarker_association",
                perturbation_modality=None,
                direction_rule_id=None,
                direction_rule_version=None,
                direction_inference_status=None,
                direction_inference_curator_verified=None,
            )
        )


def test_clinical_trial_normalization_separates_match_axes():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    normalized = normalize_clinical_trials(
        snapshot,
        _context(),
        treatment_entity_aliases=["Lynparza", "AZD2281"],
        subtype_entity_aliases=["TNBC"],
    ).set_index("nct_id")
    combination = normalized.loc["NCT00000001"]
    alias = normalized.loc["NCT00000002"]
    other_parpi = normalized.loc["NCT00000003"]
    assert combination["intervention_match"] == "exact_canonical"
    assert combination["disease_match"] == "explicit_subtype_term"
    assert combination["biomarker_match"] == "explicit_structured_term"
    assert combination["regimen_relation"] == "additional_active_agent_listed"
    assert alias["intervention_match"] == "explicit_alias"
    assert alias["disease_match"] == "cancer_type_term_only"
    assert alias["regimen_relation"] == "no_additional_active_agent_listed"
    assert other_parpi["intervention_match"] == "no_structured_match"
    assert not bool(other_parpi["registry_supports_patient_level_omics"])
    assert not bool(other_parpi["used_for_gene_ranking"])

    invalid_boolean = _trial_set()[0]
    invalid_boolean["hasResults"] = "false"
    invalid_snapshot = clinical_trials_snapshot_from_document(
        _snapshot_document([invalid_boolean])
    )
    with pytest.raises(TranslationContextError, match="hasResults must be a boolean"):
        normalize_clinical_trials(invalid_snapshot, _context())


def test_placebo_mention_and_embedded_combination_do_not_look_like_monotherapy():
    studies = [
        _trial(
            "NCT00000005",
            interventions=[{"type": "DRUG", "name": "Placebo matching olaparib"}],
            conditions=["Triple-Negative Breast Cancer"],
        ),
        _trial(
            "NCT00000006",
            interventions=[
                {
                    "type": "COMBINATION_PRODUCT",
                    "name": "olaparib + durvalumab",
                }
            ],
            conditions=["Triple-Negative Breast Cancer"],
        ),
    ]
    normalized = normalize_clinical_trials(
        clinical_trials_snapshot_from_document(_snapshot_document(studies)),
        _context(),
    ).set_index("nct_id")
    assert normalized.loc["NCT00000005", "intervention_match"] == (
        "no_structured_match"
    )
    assert normalized.loc["NCT00000006", "regimen_relation"] == (
        "additional_active_agent_listed"
    )


def test_declared_class_and_ancestor_terms_never_become_exact_entities():
    study = _trial(
        "NCT00000008",
        interventions=[{"type": "DRUG", "name": "PARP inhibitor"}],
        conditions=["Solid Tumor"],
    )
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document([study]))
    normalized = normalize_clinical_trials(
        snapshot,
        _context(
            biomarker_context=None,
            biomarker_feature_type=None,
            biomarker_state=None,
            biomarker_specimen_type=None,
            biomarker_measurement_timepoint=None,
            biomarker_axes_informative_verified=None,
        ),
        treatment_class_terms=["PARP inhibitor"],
        cancer_ancestor_terms=["solid tumor"],
    )
    assert normalized.iloc[0]["intervention_match"] == "declared_class_term"
    assert normalized.iloc[0]["disease_match"] == "declared_ancestor_term"
    result = build_translation_context_report(
        _context(
            biomarker_context=None,
            biomarker_feature_type=None,
            biomarker_state=None,
            biomarker_specimen_type=None,
            biomarker_measurement_timepoint=None,
            biomarker_axes_informative_verified=None,
        ),
        snapshot,
        evidence_cutoff_date=date(2026, 8, 28),
        treatment_class_terms=["PARP inhibitor"],
        cancer_ancestor_terms=["solid tumor"],
    )
    clinical_status = result.missingness.set_index("lane").loc[
        "clinical_trial_registry", "status"
    ]
    assert clinical_status == "frozen_query_context_unverified"


def test_subtype_keyword_cannot_override_conflicting_cancer_type():
    study = _trial(
        "NCT00000007",
        interventions=[{"type": "DRUG", "name": "olaparib"}],
        conditions=["Ovarian Cancer"],
        keywords=["TNBC"],
    )
    normalized = normalize_clinical_trials(
        clinical_trials_snapshot_from_document(_snapshot_document([study])),
        _context(),
        subtype_entity_aliases=["TNBC"],
    )
    assert normalized.iloc[0]["disease_match"] == "no_structured_match"


def test_disease_exactness_does_not_conflate_small_cell_with_non_small_cell():
    context = _context(
        cancer_type="small cell lung cancer",
        disease_subtype=None,
        disease_subtype_id=None,
        disease_ontology_name=None,
        disease_ontology_version=None,
        regimen_name=None,
        biomarker_context=None,
        biomarker_feature_type=None,
        biomarker_state=None,
        biomarker_specimen_type=None,
        biomarker_measurement_timepoint=None,
        biomarker_axes_informative_verified=None,
    )
    study = _trial(
        "NCT00000009",
        interventions=[{"type": "DRUG", "name": "olaparib"}],
        conditions=["Non-Small Cell Lung Cancer"],
    )

    def opener(request, timeout):
        if request.full_url.endswith("/version"):
            return _Response(
                {
                    "apiVersion": "2.0.5",
                    "dataTimestamp": "2026-08-28T09:00:06Z",
                }
            )
        return _Response({"studies": [study], "totalCount": 1})

    snapshot = fetch_clinical_trials_concept_v2(
        "olaparib",
        "small cell lung cancer",
        retrieved_at_utc=datetime(2026, 8, 28, 12, tzinfo=UTC),
        opener=opener,
    )
    result = build_translation_context_report(
        context,
        snapshot,
        evidence_cutoff_date=date(2026, 8, 28),
    )
    assert result.clinical_trials.iloc[0]["disease_match"] == "no_structured_match"
    assert (
        result.metadata["clinicaltrials"]["strict_structured_registry_candidate_count"]
        == 0
    )

    contradictory_context = _context(
        cancer_type="small cell lung cancer",
        disease_subtype="non-small cell lung cancer",
        disease_subtype_id=None,
        disease_ontology_name=None,
        disease_ontology_version=None,
        regimen_name=None,
        biomarker_context=None,
        biomarker_feature_type=None,
        biomarker_state=None,
        biomarker_specimen_type=None,
        biomarker_measurement_timepoint=None,
        biomarker_axes_informative_verified=None,
    )
    contradictory_snapshot = fetch_clinical_trials_concept_v2(
        "olaparib",
        "small cell lung cancer",
        disease_subtype="non-small cell lung cancer",
        retrieved_at_utc=datetime(2026, 8, 28, 12, tzinfo=UTC),
        opener=opener,
    )
    contradictory_result = build_translation_context_report(
        contradictory_context,
        contradictory_snapshot,
        evidence_cutoff_date=date(2026, 8, 28),
    )
    assert (
        contradictory_result.metadata["clinicaltrials"][
            "strict_structured_registry_candidate_count"
        ]
        == 0
    )


def test_signed_subtype_and_parent_cancer_are_required_for_registry_exactness():
    context = _context(
        disease_subtype="HER2+",
        disease_subtype_id=None,
        disease_ontology_name=None,
        disease_ontology_version=None,
        regimen_name=None,
        stage=None,
        biomarker_context=None,
        biomarker_feature_type=None,
        biomarker_state=None,
        biomarker_specimen_type=None,
        biomarker_measurement_timepoint=None,
        biomarker_axes_informative_verified=None,
    )
    studies = [
        _trial(
            "NCT00000010",
            interventions=[{"type": "DRUG", "name": "olaparib"}],
            conditions=["HER2"],
        ),
        _trial(
            "NCT00000011",
            interventions=[{"type": "DRUG", "name": "olaparib"}],
            conditions=["HER2+"],
        ),
        _trial(
            "NCT00000012",
            interventions=[{"type": "DRUG", "name": "olaparib"}],
            conditions=["HER2+", "Breast Cancer"],
        ),
    ]
    normalized = normalize_clinical_trials(
        clinical_trials_snapshot_from_document(_snapshot_document(studies)),
        context,
    ).set_index("nct_id")
    assert normalized.loc["NCT00000010", "disease_match"] == "no_structured_match"
    assert normalized.loc["NCT00000011", "disease_match"] == "no_structured_match"
    assert normalized.loc["NCT00000012", "disease_match"] == ("explicit_subtype_term")


@pytest.mark.parametrize(
    "signed_subtype",
    ["HER2 + breast cancer", "HER2 - breast cancer"],
)
def test_spaced_signed_subtype_does_not_match_unsigned_registry_term(signed_subtype):
    context = _context(
        disease_subtype=signed_subtype,
        disease_subtype_id=None,
        disease_ontology_name=None,
        disease_ontology_version=None,
        regimen_name=None,
        stage=None,
        biomarker_context=None,
        biomarker_feature_type=None,
        biomarker_state=None,
        biomarker_specimen_type=None,
        biomarker_measurement_timepoint=None,
        biomarker_axes_informative_verified=None,
    )
    study = _trial(
        "NCT00000015",
        interventions=[{"type": "DRUG", "name": "olaparib"}],
        conditions=["HER2 breast cancer", "Breast Cancer"],
    )
    normalized = normalize_clinical_trials(
        clinical_trials_snapshot_from_document(_snapshot_document([study])),
        context,
    )
    assert normalized.iloc[0]["disease_match"] == "cancer_type_term_only"


def test_duplicate_nct_records_fail_closed():
    duplicated = [_trial_set()[0], _trial_set()[0]]
    with pytest.raises(TranslationContextError, match="duplicate NCT"):
        clinical_trials_snapshot_from_document(_snapshot_document(duplicated))


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _freeze_translation_context_wall_clock(monkeypatch):
    original_utc_datetime = translation_context_module._utc_datetime
    frozen_instant = datetime(2026, 8, 28, 12, tzinfo=UTC)

    def frozen_utc_datetime(value):
        if value is None:
            return frozen_instant
        return original_utc_datetime(value)

    monkeypatch.setattr(
        translation_context_module,
        "_utc_datetime",
        frozen_utc_datetime,
    )


def _concept_snapshot_for_integrity_tests():
    def opener(request, timeout):
        if request.full_url.endswith("/version"):
            return _Response(
                {
                    "apiVersion": "2.0.5",
                    "dataTimestamp": "2026-08-28T09:00:06Z",
                }
            )
        if "query.intr=Lynparza" in request.full_url:
            return _Response({"studies": [_trial_set()[1]], "totalCount": 1})
        return _Response({"studies": [_trial_set()[0]], "totalCount": 1})

    return fetch_clinical_trials_concept_v2(
        "olaparib",
        "breast cancer",
        treatment_entity_aliases=["Lynparza"],
        retrieved_at_utc=datetime(2026, 8, 28, 12, tzinfo=UTC),
        opener=opener,
    )


def _canonical_json_sha256(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_live_adapter_tracks_pagination_and_truncation():
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        if request.full_url.endswith("/version"):
            return _Response(
                {
                    "apiVersion": "2.0.5",
                    "dataTimestamp": "2026-08-28T09:00:06Z",
                }
            )
        if "pageToken=" in request.full_url:
            return _Response({"studies": [_trial_set()[1]], "totalCount": 2})
        return _Response(
            {
                "studies": [_trial_set()[0]],
                "totalCount": 2,
                "nextPageToken": "TOKEN",
            }
        )

    complete = fetch_clinical_trials_v2(
        "olaparib",
        "breast cancer",
        page_size=1,
        max_studies=2,
        retrieved_at_utc=datetime(2026, 8, 28, 12, tzinfo=UTC),
        opener=opener,
    )
    assert complete.complete
    assert len(complete.studies) == 2
    assert len(complete.document["page_canonical_sha256"]) == 2
    assert len([url for url, _ in calls if "/studies?" in url]) == 2
    replayed = clinical_trials_snapshot_from_document(complete.document)
    expected_frozen = json.loads(json.dumps(complete.document))
    expected_frozen["version_stable"] = False
    assert replayed.document == expected_frozen
    assert not replayed.version_stable
    truncated = fetch_clinical_trials_v2(
        "olaparib",
        "breast cancer",
        page_size=1,
        max_studies=1,
        retrieved_at_utc=datetime(2026, 8, 28, 12, tzinfo=UTC),
        opener=opener,
    )
    assert not truncated.complete
    assert len(truncated.studies) == 1


@pytest.mark.parametrize(
    "version_payload",
    [
        {"apiVersion": "99.0", "dataTimestamp": "2026-08-28T09:00:06Z"},
        {"apiVersion": "2.0.5", "dataTimestamp": "not-a-date"},
        {
            "apiVersion": "2.0.5",
            "dataTimestamp": "2026-08-28T09:00:06+03:00",
        },
    ],
)
def test_live_adapter_rejects_version_metadata_it_cannot_replay(version_payload):
    def opener(request, timeout):
        if request.full_url.endswith("/version"):
            return _Response(version_payload)
        raise AssertionError("study endpoint must not run after invalid version")

    with pytest.raises(TranslationContextError):
        fetch_clinical_trials_v2("olaparib", "breast cancer", opener=opener)


@pytest.mark.parametrize(
    ("page", "max_studies", "message"),
    [
        (
            {"studies": [_trial_set()[0]], "totalCount": "1"},
            1,
            "non-negative integer",
        ),
        (
            {
                "studies": [_trial_set()[0], _trial_set()[0]],
                "totalCount": 2,
            },
            2,
            "duplicate NCT",
        ),
        (
            {
                "studies": [_trial_set()[0]],
                "totalCount": 1,
                "nextPageToken": "MORE",
            },
            1,
            "truncated pagination is inconsistent",
        ),
    ],
)
def test_live_adapter_returns_only_frozen_replayable_snapshots(
    page, max_studies, message
):
    def opener(request, timeout):
        if request.full_url.endswith("/version"):
            return _Response(
                {
                    "apiVersion": "2.0.5",
                    "dataTimestamp": "2026-08-28T09:00:06Z",
                }
            )
        return _Response(page)

    with pytest.raises(TranslationContextError, match=message):
        fetch_clinical_trials_v2(
            "olaparib",
            "breast cancer",
            max_studies=max_studies,
            retrieved_at_utc=datetime(2026, 8, 28, 12, tzinfo=UTC),
            opener=opener,
        )


def test_live_concept_adapter_queries_declared_aliases_and_deduplicates():
    def opener(request, timeout):
        if request.full_url.endswith("/version"):
            return _Response(
                {
                    "apiVersion": "2.0.5",
                    "dataTimestamp": "2026-08-28T09:00:06Z",
                }
            )
        if "query.intr=Lynparza" in request.full_url:
            return _Response({"studies": [_trial_set()[1]], "totalCount": 1})
        return _Response({"studies": [_trial_set()[0]], "totalCount": 1})

    snapshot = fetch_clinical_trials_concept_v2(
        "olaparib",
        "breast cancer",
        treatment_entity_aliases=["Lynparza"],
        retrieved_at_utc=datetime(2026, 8, 28, 12, tzinfo=UTC),
        opener=opener,
    )
    assert snapshot.complete
    assert len(snapshot.studies) == 2
    assert len(snapshot.document["declared_query_set"]) == 2
    assert {
        query["treatment_query_lane"]
        for query in snapshot.document["declared_query_set"]
    } == {"treatment_canonical", "treatment_entity_alias"}
    assert {
        query["condition_query_lane"]
        for query in snapshot.document["declared_query_set"]
    } == {"cancer_canonical"}
    assert snapshot.document["declared_query_semantics_version"] == (
        "typed_query_lanes_v1"
    )
    assert snapshot.document["ontology_concept_recall_complete"] is False
    tampered = json.loads(json.dumps(snapshot.document))
    tampered["declared_query_set"][0]["pages"][0]["studies"][0]["hasResults"] = True
    with pytest.raises(TranslationContextError, match="declared query page checksum"):
        clinical_trials_snapshot_from_document(tampered)
    missing_lane_version = json.loads(json.dumps(snapshot.document))
    missing_lane_version.pop("declared_query_semantics_version")
    with pytest.raises(TranslationContextError, match="require.*semantics_version"):
        clinical_trials_snapshot_from_document(missing_lane_version)
    invalid_lane = json.loads(json.dumps(snapshot.document))
    invalid_lane["declared_query_set"][0]["treatment_query_lane"] = "drug_class"
    with pytest.raises(TranslationContextError, match="query_lane is invalid"):
        clinical_trials_snapshot_from_document(invalid_lane)


def test_live_concept_adapter_queries_subtype_class_and_ancestor_lanes():
    def opener(request, timeout):
        if request.full_url.endswith("/version"):
            return _Response(
                {
                    "apiVersion": "2.0.5",
                    "dataTimestamp": "2026-08-28T09:00:06Z",
                }
            )
        return _Response({"studies": [_trial_set()[0]], "totalCount": 1})

    snapshot = fetch_clinical_trials_concept_v2(
        "olaparib",
        "breast cancer",
        treatment_class_terms=["PARP inhibitor"],
        disease_subtype="triple-negative breast cancer",
        cancer_ancestor_terms=["solid tumor"],
        retrieved_at_utc=datetime(2026, 8, 28, 12, tzinfo=UTC),
        opener=opener,
    )
    assert len(snapshot.studies) == 1
    assert len(snapshot.document["declared_query_set"]) == 6
    assert {
        query["treatment_query_lane"]
        for query in snapshot.document["declared_query_set"]
    } == {"treatment_canonical", "treatment_class_term"}
    assert {
        query["condition_query_lane"]
        for query in snapshot.document["declared_query_set"]
    } == {
        "cancer_canonical",
        "disease_subtype_canonical",
        "cancer_ancestor_term",
    }


def test_frozen_declared_query_page_size_matches_stored_page():
    def opener(request, timeout):
        if request.full_url.endswith("/version"):
            return _Response(
                {
                    "apiVersion": "2.0.5",
                    "dataTimestamp": "2026-08-28T09:00:06Z",
                }
            )
        return _Response({"studies": _trial_set()[:2], "totalCount": 2})

    document = fetch_clinical_trials_concept_v2(
        "olaparib",
        "breast cancer",
        retrieved_at_utc=datetime(2026, 8, 28, 12, tzinfo=UTC),
        opener=opener,
    ).document
    narrowed = json.loads(json.dumps(document))
    narrowed_url = narrowed["request_urls"][0].replace("pageSize=100", "pageSize=1")
    narrowed["request_urls"][0] = narrowed_url
    narrowed["declared_query_set"][0]["request_urls"][0] = narrowed_url
    with pytest.raises(TranslationContextError, match="more studies than requested"):
        clinical_trials_snapshot_from_document(narrowed)


def test_live_concept_adapter_rejects_ambiguous_query_lane_declarations():
    with pytest.raises(TranslationContextError, match="multiple query lanes"):
        fetch_clinical_trials_concept_v2(
            "olaparib",
            "breast cancer",
            treatment_entity_aliases=["OLAPARIB"],
        )
    with pytest.raises(TranslationContextError, match="require a canonical"):
        fetch_clinical_trials_concept_v2(
            "olaparib",
            "breast cancer",
            subtype_entity_aliases=["TNBC"],
        )


def test_typed_query_binding_does_not_upgrade_an_injected_transport():
    def opener(request, timeout):
        if request.full_url.endswith("/version"):
            return _Response(
                {
                    "apiVersion": "2.0.5",
                    "dataTimestamp": "2026-08-28T09:00:06Z",
                }
            )
        return _Response({"studies": [_trial_set()[0]], "totalCount": 1})

    snapshot = fetch_clinical_trials_concept_v2(
        "olaparib",
        "breast cancer",
        treatment_entity_aliases=["Lynparza"],
        disease_subtype="triple-negative breast cancer",
        retrieved_at_utc=datetime(2026, 8, 28, 12, tzinfo=UTC),
        opener=opener,
    )
    context = _context(
        regimen_name=None,
        stage=None,
        line_of_therapy=None,
        biomarker_context=None,
        biomarker_feature_type=None,
        biomarker_state=None,
        biomarker_specimen_type=None,
        biomarker_measurement_timepoint=None,
        biomarker_axes_informative_verified=None,
    )
    result = build_translation_context_report(
        context,
        snapshot,
        evidence_cutoff_date=date(2026, 8, 28),
        treatment_entity_aliases=["Lynparza"],
    )
    assert (
        result.metadata["clinicaltrials"]["declared_query_context_binding"]
        == "verified_typed_query_cross_product"
    )
    assert (
        result.missingness.set_index("lane").loc["clinical_trial_registry", "status"]
        == "injected_source_provenance_unverified"
    )

    mismatched_context = context.model_copy(
        update={
            "treatment_name": "durvalumab",
            "treatment_id": None,
            "treatment_ontology_name": None,
            "treatment_ontology_version": None,
        }
    )
    with pytest.raises(TranslationContextError, match="does not match the requested"):
        build_translation_context_report(
            mismatched_context,
            snapshot,
            evidence_cutoff_date=date(2026, 8, 28),
        )


def test_stock_live_transport_can_support_a_strict_registry_count(monkeypatch):
    def opener(request, timeout):
        if request.full_url.endswith("/version"):
            return _Response(
                {
                    "apiVersion": "2.0.5",
                    "dataTimestamp": "2025-08-28T09:00:06Z",
                }
            )
        return _Response({"studies": [_trial_set()[0]], "totalCount": 1})

    monkeypatch.setattr(translation_context_module, "urlopen", opener)
    _freeze_translation_context_wall_clock(monkeypatch)
    snapshot = fetch_clinical_trials_concept_v2(
        "olaparib",
        "breast cancer",
        disease_subtype="triple-negative breast cancer",
    )
    context = _context(
        regimen_name=None,
        stage=None,
        line_of_therapy=None,
        biomarker_context=None,
        biomarker_feature_type=None,
        biomarker_state=None,
        biomarker_specimen_type=None,
        biomarker_measurement_timepoint=None,
        biomarker_axes_informative_verified=None,
    )
    result = build_translation_context_report(
        context,
        snapshot,
        evidence_cutoff_date=date(2026, 8, 28),
    )
    assert snapshot.source_mode == "live_api_declared_query_set"
    assert snapshot.version_stable is True
    assert (
        result.metadata["clinicaltrials"]["strict_structured_registry_candidate_count"]
        == 1
    )
    assert (
        result.missingness.set_index("lane").loc["clinical_trial_registry", "status"]
        == "strict_structured_registry_candidates_present"
    )

    name_only_context = TreatmentDiseaseContextRecord.model_validate(
        {
            **context.model_dump(mode="json"),
            "treatment_id": None,
            "treatment_ontology_name": None,
            "treatment_ontology_version": None,
            "cancer_id": None,
            "disease_subtype_id": None,
            "disease_subtype_parent_id": None,
            "disease_subtype_parent_binding_verified": False,
            "disease_ontology_name": None,
            "disease_ontology_version": None,
        }
    )
    name_only_result = build_translation_context_report(
        name_only_context,
        snapshot,
        evidence_cutoff_date=date(2026, 8, 28),
    )
    assert (
        name_only_result.metadata["clinicaltrials"][
            "strict_structured_registry_candidate_count"
        ]
        == 0
    )


def test_strict_registry_requires_verified_requested_subtype_parent(monkeypatch):
    def opener(request, timeout):
        if request.full_url.endswith("/version"):
            return _Response(
                {
                    "apiVersion": "2.0.5",
                    "dataTimestamp": "2025-08-28T09:00:06Z",
                }
            )
        return _Response({"studies": [_trial_set()[0]], "totalCount": 1})

    monkeypatch.setattr(translation_context_module, "urlopen", opener)
    _freeze_translation_context_wall_clock(monkeypatch)
    snapshot = fetch_clinical_trials_concept_v2(
        "olaparib",
        "breast cancer",
        disease_subtype="triple-negative breast cancer",
    )
    context = _context(
        disease_subtype_parent_binding_verified=False,
        regimen_name=None,
        stage=None,
        line_of_therapy=None,
        biomarker_context=None,
        biomarker_feature_type=None,
        biomarker_state=None,
        biomarker_specimen_type=None,
        biomarker_measurement_timepoint=None,
        biomarker_axes_informative_verified=None,
    )
    result = build_translation_context_report(
        context,
        snapshot,
        evidence_cutoff_date=date(2026, 8, 28),
    )
    assert (
        result.metadata["clinicaltrials"]["strict_structured_registry_candidate_count"]
        == 0
    )
    assert (
        result.missingness.set_index("lane").loc["clinical_trial_registry", "status"]
        == "broader_or_unresolved_registry_matches_only"
    )


def test_typed_query_binding_preserves_signed_subtype_identity():
    study = _trial(
        "NCT00000013",
        interventions=[{"type": "DRUG", "name": "olaparib"}],
        conditions=["HER2", "Breast Cancer"],
    )

    def opener(request, timeout):
        if request.full_url.endswith("/version"):
            return _Response(
                {
                    "apiVersion": "2.0.5",
                    "dataTimestamp": "2026-08-28T09:00:06Z",
                }
            )
        return _Response({"studies": [study], "totalCount": 1})

    snapshot = fetch_clinical_trials_concept_v2(
        "olaparib",
        "breast cancer",
        disease_subtype="HER2",
        retrieved_at_utc=datetime(2026, 8, 28, 12, tzinfo=UTC),
        opener=opener,
    )
    signed_context = _context(
        disease_subtype="HER2+",
        disease_subtype_id=None,
        disease_ontology_name=None,
        disease_ontology_version=None,
        regimen_name=None,
        stage=None,
        biomarker_context=None,
        biomarker_feature_type=None,
        biomarker_state=None,
        biomarker_specimen_type=None,
        biomarker_measurement_timepoint=None,
        biomarker_axes_informative_verified=None,
    )
    with pytest.raises(TranslationContextError, match="does not match the requested"):
        build_translation_context_report(
            signed_context,
            snapshot,
            evidence_cutoff_date=date(2026, 8, 28),
        )


@pytest.mark.parametrize(
    "context_update",
    [
        {"regimen_name": "olaparib monotherapy"},
        {"stage": "advanced"},
        {"line_of_therapy": "second line"},
    ],
)
def test_registry_strictness_abstains_on_unparsed_clinical_axes(
    context_update, monkeypatch
):
    def opener(request, timeout):
        if request.full_url.endswith("/version"):
            return _Response(
                {
                    "apiVersion": "2.0.5",
                    "dataTimestamp": "2025-08-28T09:00:06Z",
                }
            )
        return _Response({"studies": [_trial_set()[0]], "totalCount": 1})

    monkeypatch.setattr(translation_context_module, "urlopen", opener)
    _freeze_translation_context_wall_clock(monkeypatch)
    snapshot = fetch_clinical_trials_concept_v2(
        "olaparib",
        "breast cancer",
        disease_subtype="triple-negative breast cancer",
    )
    context = _context(
        **{
            "regimen_name": None,
            "stage": None,
            "line_of_therapy": None,
            "biomarker_context": None,
            "biomarker_feature_type": None,
            "biomarker_state": None,
            "biomarker_specimen_type": None,
            "biomarker_measurement_timepoint": None,
            "biomarker_axes_informative_verified": None,
            **context_update,
        }
    )
    result = build_translation_context_report(
        context,
        snapshot,
        evidence_cutoff_date=date(2026, 8, 28),
    )
    assert (
        result.metadata["clinicaltrials"]["strict_structured_registry_candidate_count"]
        == 0
    )
    assert (
        result.missingness.set_index("lane").loc["clinical_trial_registry", "status"]
        == "broader_or_unresolved_registry_matches_only"
    )


def test_registry_component_mentions_never_create_strict_treatment_match():
    study = _trial(
        "NCT00000014",
        interventions=[{"type": "DRUG", "name": "Non-olaparib therapy"}],
        conditions=["Breast Cancer"],
    )

    def opener(request, timeout):
        if request.full_url.endswith("/version"):
            return _Response(
                {
                    "apiVersion": "2.0.5",
                    "dataTimestamp": "2026-08-28T09:00:06Z",
                }
            )
        return _Response({"studies": [study], "totalCount": 1})

    snapshot = fetch_clinical_trials_concept_v2(
        "olaparib",
        "breast cancer",
        retrieved_at_utc=datetime(2026, 8, 28, 12, tzinfo=UTC),
        opener=opener,
    )
    context = _context(
        disease_subtype=None,
        disease_subtype_id=None,
        disease_ontology_name=None,
        disease_ontology_version=None,
        regimen_name=None,
        stage=None,
        biomarker_context=None,
        biomarker_feature_type=None,
        biomarker_state=None,
        biomarker_specimen_type=None,
        biomarker_measurement_timepoint=None,
        biomarker_axes_informative_verified=None,
    )
    result = build_translation_context_report(
        context,
        snapshot,
        evidence_cutoff_date=date(2026, 8, 28),
    )
    assert result.clinical_trials.iloc[0]["intervention_match"] == (
        "explicit_component"
    )
    assert (
        result.metadata["clinicaltrials"]["strict_structured_registry_candidate_count"]
        == 0
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source", "Fake Registry", "source must be exactly"),
        ("source_api_major", "v99", "source_api_major must be v2"),
        ("api_version", 2, "version string"),
        ("api_version", "99.0", "major version 2"),
        ("data_timestamp", "not-a-date", "valid UTC timestamp"),
        ("data_timestamp", "2026-08-28T09:00:06+03:00", "UTC offset"),
        (
            "ontology_concept_recall_complete",
            True,
            "cannot be true for text queries",
        ),
    ],
)
def test_frozen_wrapper_rejects_spoofed_source_metadata(field, value, message):
    document = _snapshot_document(_trial_set())
    document[field] = value
    with pytest.raises(TranslationContextError, match=message):
        clinical_trials_snapshot_from_document(document)


def test_frozen_wrapper_accepts_official_utc_timestamp_spelling():
    document = _snapshot_document(_trial_set())
    document["data_timestamp"] = "2026-08-28T09:00:06 UTC"
    snapshot = clinical_trials_snapshot_from_document(document)
    assert snapshot.data_timestamp == "2026-08-28T09:00:06 UTC"


def test_declared_query_manifest_is_recomputed_from_frozen_pages():
    base = _concept_snapshot_for_integrity_tests().document

    wrong_reported_total = json.loads(json.dumps(base))
    wrong_reported_total["declared_query_set"][0]["reported_total_count"] = 999
    with pytest.raises(TranslationContextError, match="reported_total_count"):
        clinical_trials_snapshot_from_document(wrong_reported_total)

    wrong_retrieved_count = json.loads(json.dumps(base))
    wrong_retrieved_count["declared_query_set"][0]["retrieved_record_count"] = 0
    with pytest.raises(TranslationContextError, match="retrieved_record_count"):
        clinical_trials_snapshot_from_document(wrong_retrieved_count)

    wrong_query_complete = json.loads(json.dumps(base))
    wrong_query_complete["declared_query_set"][0]["complete"] = False
    with pytest.raises(TranslationContextError, match="completeness disagrees"):
        clinical_trials_snapshot_from_document(wrong_query_complete)

    wrong_query_url = json.loads(json.dumps(base))
    wrong_query_url["declared_query_set"][0]["request_urls"][0] = wrong_query_url[
        "declared_query_set"
    ][1]["request_urls"][0]
    with pytest.raises(TranslationContextError, match="query.intr disagrees"):
        clinical_trials_snapshot_from_document(wrong_query_url)

    for narrowing_parameter in (
        "filter.overallStatus=RECRUITING",
        "query.id=NCT00000001",
    ):
        narrowed = json.loads(json.dumps(base))
        narrowed_url = (
            narrowed["declared_query_set"][0]["request_urls"][0]
            + f"&{narrowing_parameter}"
        )
        narrowed["declared_query_set"][0]["request_urls"][0] = narrowed_url
        narrowed["request_urls"][0] = narrowed_url
        with pytest.raises(TranslationContextError, match="narrowing query"):
            clinical_trials_snapshot_from_document(narrowed)

    wrong_page_token = json.loads(json.dumps(base))
    wrong_page_token["declared_query_set"][0]["request_urls"][0] += "&pageToken=FORGED"
    with pytest.raises(TranslationContextError, match="no pageToken|pageToken chain"):
        clinical_trials_snapshot_from_document(wrong_page_token)

    wrong_top_urls = json.loads(json.dumps(base))
    wrong_top_urls["request_urls"] = list(reversed(wrong_top_urls["request_urls"]))
    with pytest.raises(TranslationContextError, match="top-level request_urls"):
        clinical_trials_snapshot_from_document(wrong_top_urls)

    wrong_top_complete = json.loads(json.dumps(base))
    wrong_top_complete["complete"] = False
    with pytest.raises(TranslationContextError, match="top-level completeness"):
        clinical_trials_snapshot_from_document(wrong_top_complete)


def test_declared_query_top_level_studies_are_exact_nct_payload_union():
    document = json.loads(json.dumps(_concept_snapshot_for_integrity_tests().document))
    document["pages"][0]["studies"][0]["hasResults"] = True
    document["page_canonical_sha256"] = [_canonical_json_sha256(document["pages"][0])]
    with pytest.raises(TranslationContextError, match="declared-query NCT union"):
        clinical_trials_snapshot_from_document(document)

    wrong_total = json.loads(
        json.dumps(_concept_snapshot_for_integrity_tests().document)
    )
    wrong_total["pages"][0]["studies"] = wrong_total["pages"][0]["studies"][:1]
    wrong_total["pages"][0]["totalCount"] = 1
    wrong_total["total_count"] = 1
    wrong_total["page_canonical_sha256"] = [
        _canonical_json_sha256(wrong_total["pages"][0])
    ]
    with pytest.raises(TranslationContextError, match="top-level total_count"):
        clinical_trials_snapshot_from_document(wrong_total)


def test_incomplete_declared_query_snapshot_remains_explicitly_truncated():
    def opener(request, timeout):
        if request.full_url.endswith("/version"):
            return _Response(
                {
                    "apiVersion": "2.0.5",
                    "dataTimestamp": "2026-08-28T09:00:06Z",
                }
            )
        return _Response(
            {
                "studies": [_trial_set()[0]],
                "totalCount": 2,
                "nextPageToken": "MORE",
            }
        )

    live = fetch_clinical_trials_concept_v2(
        "olaparib",
        "breast cancer",
        max_studies_per_query=1,
        retrieved_at_utc=datetime(2026, 8, 28, 12, tzinfo=UTC),
        opener=opener,
    )
    frozen = clinical_trials_snapshot_from_document(live.document)
    assert frozen.complete is False
    assert frozen.total_count is None


def test_snapshot_integrity_fields_fail_closed():
    count_mismatch = _snapshot_document(_trial_set())
    count_mismatch["total_count"] = 99
    with pytest.raises(TranslationContextError, match="disagrees"):
        clinical_trials_snapshot_from_document(count_mismatch)

    wrong_boolean = _snapshot_document(_trial_set())
    wrong_boolean["complete"] = "false"
    with pytest.raises(TranslationContextError, match="must be a boolean"):
        clinical_trials_snapshot_from_document(wrong_boolean)

    canonical = clinical_trials_snapshot_from_document(
        _snapshot_document(_trial_set())
    ).document
    canonical["pages"][0]["studies"][0]["hasResults"] = True
    with pytest.raises(TranslationContextError, match="checksum mismatch"):
        clinical_trials_snapshot_from_document(canonical)

    wrapped = _snapshot_document(_trial_set())
    with pytest.raises(TranslationContextError, match="cannot override"):
        clinical_trials_snapshot_from_document(
            wrapped,
            retrieved_at_utc="2026-08-27T12:00:00+00:00",
        )


def test_report_revalidates_snapshot_after_nested_mutation():
    snapshot = _concept_snapshot_for_integrity_tests()
    snapshot.studies[0]["protocolSection"]["armsInterventionsModule"]["interventions"][
        0
    ]["name"] = "tampered olaparib claim"
    context = _context(
        disease_subtype=None,
        disease_subtype_id=None,
        disease_ontology_name=None,
        disease_ontology_version=None,
        regimen_name=None,
        stage=None,
        biomarker_context=None,
        biomarker_feature_type=None,
        biomarker_state=None,
        biomarker_specimen_type=None,
        biomarker_measurement_timepoint=None,
        biomarker_axes_informative_verified=None,
    )
    with pytest.raises(
        TranslationContextError,
        match="checksum mismatch|changed after validation",
    ):
        build_translation_context_report(
            context,
            snapshot,
            evidence_cutoff_date=date(2026, 8, 28),
            treatment_entity_aliases=["Lynparza"],
        )


def test_frozen_snapshot_cannot_forge_a_live_acquisition_capability():
    frozen = clinical_trials_snapshot_from_document(
        _concept_snapshot_for_integrity_tests().document
    )
    forged_document = json.loads(json.dumps(frozen.document))
    forged_document["version_stable"] = True
    forged = replace(
        frozen,
        document=forged_document,
        source_mode="live_api_declared_query_set",
        version_stable=True,
        _live_acquisition_witness=object(),
    )
    with pytest.raises(TranslationContextError, match="in-process version audit"):
        build_translation_context_report(
            _context(),
            forged,
            evidence_cutoff_date=date(2026, 8, 28),
        )


def test_live_acquisition_capability_is_bound_to_document_digest(monkeypatch):
    def opener(request, timeout):
        if request.full_url.endswith("/version"):
            return _Response(
                {
                    "apiVersion": "2.0.5",
                    "dataTimestamp": "2025-08-28T09:00:06Z",
                }
            )
        return _Response({"studies": [_trial_set()[0]], "totalCount": 1})

    monkeypatch.setattr(translation_context_module, "urlopen", opener)
    snapshot = fetch_clinical_trials_concept_v2("olaparib", "breast cancer")
    changed_retrieval = datetime(2026, 8, 28, 23, 59, tzinfo=UTC)
    changed_document = json.loads(json.dumps(snapshot.document))
    changed_document["retrieved_at_utc"] = changed_retrieval.isoformat()
    mutated = replace(
        snapshot,
        document=changed_document,
        retrieved_at_utc=changed_retrieval,
    )
    with pytest.raises(TranslationContextError, match="in-process version audit"):
        build_translation_context_report(
            _context(disease_subtype=None),
            mutated,
            evidence_cutoff_date=date(2026, 8, 28),
        )


def test_typed_query_set_requires_one_version_audit_per_query():
    document = json.loads(json.dumps(_concept_snapshot_for_integrity_tests().document))
    assert len(document["declared_query_set"]) > 1
    document["version_audit_set"] = document["version_audit_set"][:-1]
    with pytest.raises(TranslationContextError, match="one version audit per query"):
        clinical_trials_snapshot_from_document(document)


def test_incomplete_live_and_serialized_replay_never_emit_strict_count(monkeypatch):
    def opener(request, timeout):
        if request.full_url.endswith("/version"):
            return _Response(
                {
                    "apiVersion": "2.0.5",
                    "dataTimestamp": "2025-08-28T09:00:06Z",
                }
            )
        return _Response(
            {
                "studies": [_trial_set()[0]],
                "totalCount": 2,
                "nextPageToken": "MORE",
            }
        )

    monkeypatch.setattr(translation_context_module, "urlopen", opener)
    _freeze_translation_context_wall_clock(monkeypatch)
    live = fetch_clinical_trials_concept_v2(
        "olaparib",
        "breast cancer",
        max_studies_per_query=1,
    )
    context = _context(
        disease_subtype=None,
        regimen_name=None,
        stage=None,
        line_of_therapy=None,
        biomarker_context=None,
        biomarker_feature_type=None,
        biomarker_state=None,
        biomarker_specimen_type=None,
        biomarker_measurement_timepoint=None,
        biomarker_axes_informative_verified=None,
    )
    live_result = build_translation_context_report(
        context,
        live,
        evidence_cutoff_date=date(2026, 8, 28),
    )
    assert (
        live_result.metadata["clinicaltrials"][
            "strict_structured_registry_candidate_count"
        ]
        == 0
    )
    assert (
        live_result.missingness.set_index("lane").loc[
            "clinical_trial_registry", "status"
        ]
        == "truncated_current_snapshot"
    )
    assert "registry-resolvable requested axes: **0**" in live_result.report_markdown

    replay = clinical_trials_snapshot_from_document(live.document)
    replay_result = build_translation_context_report(
        context,
        replay,
        evidence_cutoff_date=date(2026, 8, 28),
    )
    assert (
        replay_result.metadata["clinicaltrials"][
            "strict_structured_registry_candidate_count"
        ]
        == 0
    )
    assert (
        replay_result.missingness.set_index("lane").loc[
            "clinical_trial_registry", "status"
        ]
        == "frozen_source_provenance_unverified"
    )


def test_live_adapter_rejects_empty_continuation_page():
    def opener(request, timeout):
        if request.full_url.endswith("/version"):
            return _Response(
                {
                    "apiVersion": "2.0.5",
                    "dataTimestamp": "2026-08-28T09:00:06Z",
                }
            )
        return _Response({"studies": [], "totalCount": 1, "nextPageToken": "NEXT"})

    with pytest.raises(TranslationContextError, match="empty page"):
        fetch_clinical_trials_v2(
            "olaparib",
            "breast cancer",
            retrieved_at_utc=datetime(2026, 8, 28, 12, tzinfo=UTC),
            opener=opener,
        )


def test_adapter_failure_does_not_become_no_trials():
    def opener(*args, **kwargs):
        raise OSError("offline")

    with pytest.raises(TranslationContextError, match="absence of trials cannot"):
        fetch_clinical_trials_v2("olaparib", "breast cancer", opener=opener)


def test_live_adapter_rejects_a_mixed_source_snapshot():
    version_calls = 0

    def opener(request, timeout):
        nonlocal version_calls
        if request.full_url.endswith("/version"):
            version_calls += 1
            return _Response(
                {
                    "apiVersion": "2.0.5",
                    "dataTimestamp": (
                        "2026-08-28T09:00:06Z"
                        if version_calls == 1
                        else "2026-08-28T10:00:06Z"
                    ),
                }
            )
        return _Response({"studies": [_trial_set()[0]], "totalCount": 1})

    with pytest.raises(TranslationContextError, match="dataTimestamp changed"):
        fetch_clinical_trials_v2(
            "olaparib",
            "breast cancer",
            retrieved_at_utc=datetime(2026, 8, 28, 12, tzinfo=UTC),
            opener=opener,
        )


def test_report_preserves_candidate_order_and_deduplicates_families():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    candidates = pd.DataFrame(
        {
            "gene_symbol": ["NME6", "LAMTOR2", "NO_EVIDENCE"],
            "screen_id": ["0007"] * 3,
            "contrast_id": ["0009"] * 3,
            "phenotype_direction": ["resistance"] * 3,
            "screen_signal_rank": [1, 2, 3],
            "ranking_type": ["screen_signal_baseline"] * 3,
        }
    )
    preclinical = pd.DataFrame(
        [
            _preclinical_record(),
            _preclinical_record(
                "P2",
                source_study_id="PMID:1-reanalysis",
                source_family_id="SOURCE-2",
                raw_data_family_id="RAW-1",
                model_type="pdx",
                model_name="PDX-1",
            ),
            _preclinical_record(
                "P3",
                source_family_id="SOURCE-FUTURE",
                raw_data_family_id="RAW-FUTURE",
                available_date="2027-01-01",
                retrieved_date="2027-01-02",
            ),
        ]
    )
    patient = pd.DataFrame([_patient_record()])
    result = build_translation_context_report(
        _context(),
        snapshot,
        candidates=candidates,
        preclinical_evidence=preclinical,
        patient_evidence=patient,
        evidence_cutoff_date=date(2026, 8, 28),
        treatment_entity_aliases=["Lynparza"],
        subtype_entity_aliases=["TNBC"],
        target_absence_attested=True,
    )
    summary = result.candidate_context.set_index("gene_symbol")
    assert (
        result.candidate_context["gene_symbol"].tolist()
        == candidates["gene_symbol"].tolist()
    )
    assert result.candidate_context["screen_signal_rank"].tolist() == [1, 2, 3]
    assert summary.loc["LAMTOR2", "report_only_preclinical_record_n"] == 2
    assert summary.loc["LAMTOR2", "report_only_preclinical_family_n"] == 1
    assert (
        summary.loc[
            "LAMTOR2",
            "report_only_preclinical_in_vitro_exact_context_family_n",
        ]
        == 1
    )
    assert (
        summary.loc[
            "LAMTOR2",
            "report_only_preclinical_in_vivo_exact_context_family_n",
        ]
        == 1
    )
    assert (
        summary.loc["NME6", "report_only_patient_predictive_exact_context_family_n"]
        == 1
    )
    assert summary.loc["NO_EVIDENCE", "report_only_patient_status"] == (
        "insufficient_matched_patient_data"
    )
    assert result.preclinical_exclusions["reason"].tolist() == ["post_cutoff"]
    assert result.metadata["clinicaltrials"]["historical_feature_eligible"] is False
    assert result.metadata["candidate_input"]["ranking_claim"] == (
        "screen_signal_baseline_structure_validated_manifest_unbound"
    )
    assert (
        result.metadata["clinicaltrials"]["strict_structured_registry_candidate_count"]
        == 0
    )
    assert (
        result.missingness.set_index("lane").loc["clinical_trial_registry", "status"]
        == "frozen_query_context_unverified"
    )


@pytest.mark.parametrize("rank", [1.5, float("inf"), True, 0, -1])
def test_manifest_unbound_candidate_ranks_are_finite_positive_integers(rank):
    candidates = pd.DataFrame(
        {
            "gene_symbol": ["NME6"],
            "screen_id": ["0007"],
            "contrast_id": ["0009"],
            "phenotype_direction": ["resistance"],
            "screen_signal_rank": [rank],
            "ranking_type": ["screen_signal_baseline"],
        }
    )
    with pytest.raises(TranslationContextError, match="finite positive integer"):
        build_translation_context_report(
            _context(),
            clinical_trials_snapshot_from_document(_snapshot_document(_trial_set())),
            candidates=candidates,
            evidence_cutoff_date=date(2026, 8, 28),
            target_absence_attested=True,
        )


@pytest.mark.parametrize("rank", [1, "garbage"])
def test_neutral_manifest_unbound_candidate_cannot_claim_a_rank(rank):
    candidates = pd.DataFrame(
        {
            "gene_symbol": ["NME6"],
            "screen_id": ["0007"],
            "contrast_id": ["0009"],
            "phenotype_direction": ["neutral"],
            "screen_signal_rank": [rank],
            "ranking_type": ["screen_signal_baseline"],
        }
    )
    with pytest.raises(TranslationContextError, match="must be unranked"):
        build_translation_context_report(
            _context(),
            clinical_trials_snapshot_from_document(_snapshot_document(_trial_set())),
            candidates=candidates,
            evidence_cutoff_date=date(2026, 8, 28),
            target_absence_attested=True,
        )


def test_patient_interaction_statuses_remain_separate_and_family_counted():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    candidates = pd.DataFrame(
        {
            "gene_symbol": ["NME6"],
            "screen_id": ["0007"],
            "contrast_id": ["0009"],
            "phenotype_direction": ["resistance"],
        }
    )

    def provenance(tag):
        code = {
            "SUPPORTED": "1001",
            "NULL": "1002",
            "INCONCLUSIVE": "1003",
            "UNSUPPORTED": "1004",
        }[tag]
        return {
            "evidence_id": f"EVIDENCE-{tag}",
            "source_study_id": f"PMID:{code}",
            "cohort_id": f"COHORT-{code}",
            "source_family_id": f"SOURCE-{code}",
            "raw_data_family_id": f"RAW-{code}",
        }

    supported = _patient_record(**provenance("SUPPORTED"))
    formal_null = _formal_null_patient(**provenance("NULL"))
    inconclusive = _patient_record(
        **provenance("INCONCLUSIVE"),
        association_interpretation="interaction_tested_inconclusive",
        interaction_inference_status="inconclusive",
        effect_numeric=1.0,
        confidence_interval_lower=0.8,
        confidence_interval_upper=1.2,
        p_value=0.01,
    )
    unsupported = _patient_record(
        **provenance("UNSUPPORTED"),
        association_interpretation="interaction_tested_unsupported",
        interaction_inference_status="unsupported",
        effect_numeric=1.0,
        confidence_interval_lower=0.8,
        confidence_interval_upper=1.2,
        p_value=0.2,
    )
    result = build_translation_context_report(
        _context(),
        snapshot,
        candidates=candidates,
        patient_evidence=pd.DataFrame(
            [supported, formal_null, inconclusive, unsupported]
        ),
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=True,
    )
    row = result.candidate_context.iloc[0]
    assert row["report_only_patient_predictive_exact_context_family_n"] == 1
    assert row["report_only_patient_interaction_null_exact_context_family_n"] == 1
    assert (
        row["report_only_patient_interaction_inconclusive_exact_context_family_n"] == 1
    )
    assert (
        row["report_only_patient_interaction_unsupported_exact_context_family_n"] == 1
    )
    assert row["report_only_patient_status"] == (
        "supported_and_nonconfirmatory_interactions_present"
    )

    nonconfirmatory = build_translation_context_report(
        _context(),
        snapshot,
        candidates=candidates,
        patient_evidence=pd.DataFrame([formal_null, inconclusive, unsupported]),
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=True,
    )
    assert (
        nonconfirmatory.candidate_context.iloc[0]["report_only_patient_status"]
        == "multiple_nonconfirmatory_interaction_results_present"
    )


def test_conflicting_subtype_cannot_create_exact_predictive_status():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    candidates = pd.DataFrame(
        {
            "gene_symbol": ["NME6"],
            "screen_id": ["0007"],
            "contrast_id": ["0009"],
            "phenotype_direction": ["resistance"],
        }
    )
    broader_patient = pd.DataFrame(
        [
            _patient_record(
                disease_subtype="ER-positive breast cancer",
                disease_subtype_id="NCIT:C53555",
            )
        ]
    )
    result = build_translation_context_report(
        _context(),
        snapshot,
        candidates=candidates,
        patient_evidence=broader_patient,
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=True,
    )
    row = result.candidate_context.iloc[0]
    assert row["report_only_patient_predictive_exact_context_family_n"] == 0
    assert row["report_only_patient_predictive_conflicting_context_family_n"] == 1
    assert row["report_only_patient_status"] == "conflicting_patient_context_only"

    biomarker_mismatch = pd.DataFrame(
        [
            _patient_record(
                biomarker_feature_type="rna_expression",
                biomarker_state="high",
            )
        ]
    )
    typed_result = build_translation_context_report(
        _context(),
        snapshot,
        candidates=candidates,
        patient_evidence=biomarker_mismatch,
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=True,
    )
    typed_row = typed_result.candidate_context.iloc[0]
    assert typed_row["report_only_patient_exact_context_family_n"] == 0
    assert typed_row["report_only_patient_predictive_conflicting_context_family_n"] == 1


def test_unspecified_subtype_is_compatible_broader_not_conflicting():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    candidates = pd.DataFrame(
        {
            "gene_symbol": ["NME6"],
            "screen_id": ["0007"],
            "contrast_id": ["0009"],
            "phenotype_direction": ["resistance"],
        }
    )
    patient = pd.DataFrame(
        [
            _patient_record(
                disease_subtype=None,
                disease_subtype_id=None,
                disease_ontology_name=None,
                disease_ontology_version=None,
            )
        ]
    )
    result = build_translation_context_report(
        _context(),
        snapshot,
        candidates=candidates,
        patient_evidence=patient,
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=True,
    )
    row = result.candidate_context.iloc[0]
    assert (
        row["report_only_patient_predictive_compatible_nonexact_context_family_n"] == 1
    )
    assert row["report_only_patient_predictive_conflicting_context_family_n"] == 0
    assert row["report_only_patient_status"] == (
        "compatible_nonexact_patient_context_only"
    )


@pytest.mark.parametrize(
    ("context_state", "evidence_state"),
    [
        ("+", "-"),
        ("HER2+", "HER2-"),
        ("positive (+)", "positive (-)"),
    ],
)
def test_signed_biomarker_state_cannot_collapse_to_exact(context_state, evidence_state):
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    candidates = pd.DataFrame(
        {
            "gene_symbol": ["NME6"],
            "screen_id": ["0007"],
            "contrast_id": ["0009"],
            "phenotype_direction": ["resistance"],
        }
    )
    result = build_translation_context_report(
        _context(biomarker_state=context_state),
        snapshot,
        candidates=candidates,
        patient_evidence=pd.DataFrame(
            [_patient_record(biomarker_state=evidence_state)]
        ),
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=True,
    )
    row = result.candidate_context.iloc[0]
    assert row["report_only_patient_predictive_exact_context_family_n"] == 0
    assert row["report_only_patient_predictive_conflicting_context_family_n"] == 1


@pytest.mark.parametrize(
    ("context_specimen", "evidence_specimen"),
    [
        ("CD3+ sorted cells", "CD3- sorted cells"),
        ("EpCAM+ tumor cells", "EpCAM- tumor cells"),
    ],
)
def test_signed_biomarker_specimen_cannot_collapse_to_exact(
    context_specimen, evidence_specimen
):
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    candidates = pd.DataFrame(
        {
            "gene_symbol": ["NME6"],
            "screen_id": ["0007"],
            "contrast_id": ["0009"],
            "phenotype_direction": ["resistance"],
        }
    )
    result = build_translation_context_report(
        _context(biomarker_specimen_type=context_specimen),
        snapshot,
        candidates=candidates,
        patient_evidence=pd.DataFrame(
            [_patient_record(biomarker_specimen_type=evidence_specimen)]
        ),
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=True,
    )
    row = result.candidate_context.iloc[0]
    assert row["report_only_patient_predictive_exact_context_family_n"] == 0
    assert row["report_only_patient_predictive_conflicting_context_family_n"] == 1


def test_uninformative_biomarker_attestation_cannot_create_exact_context():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    candidates = pd.DataFrame(
        {
            "gene_symbol": ["NME6"],
            "screen_id": ["0007"],
            "contrast_id": ["0009"],
            "phenotype_direction": ["resistance"],
        }
    )
    context = _context(
        biomarker_state="not available",
        biomarker_axes_informative_verified=False,
    )
    patient = pd.DataFrame(
        [
            _patient_record(
                biomarker_state="not available",
                biomarker_axes_informative_verified=False,
            )
        ]
    )
    result = build_translation_context_report(
        context,
        snapshot,
        candidates=candidates,
        patient_evidence=patient,
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=True,
    )
    row = result.candidate_context.iloc[0]
    assert row["report_only_patient_predictive_exact_context_family_n"] == 0
    assert (
        row["report_only_patient_predictive_compatible_nonexact_context_family_n"] == 1
    )


def test_signed_subtype_is_preserved_for_curated_evidence_matching():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    context = _context(
        disease_subtype="HER2+",
        disease_subtype_id=None,
        disease_ontology_name=None,
        disease_ontology_version=None,
        regimen_name=None,
        stage=None,
        biomarker_context=None,
        biomarker_feature_type=None,
        biomarker_state=None,
        biomarker_specimen_type=None,
        biomarker_measurement_timepoint=None,
        biomarker_axes_informative_verified=None,
    )
    candidates = pd.DataFrame(
        {
            "gene_symbol": ["LAMTOR2", "NME6"],
            "screen_id": ["0007", "0007"],
            "contrast_id": ["0009", "0009"],
            "phenotype_direction": ["resistance", "resistance"],
        }
    )
    common_updates = {
        "disease_subtype": "HER2",
        "disease_subtype_id": None,
        "disease_ontology_name": None,
        "disease_ontology_version": None,
        "biomarker_context": None,
        "biomarker_feature_type": None,
        "biomarker_state": None,
        "biomarker_specimen_type": None,
        "biomarker_measurement_timepoint": None,
        "biomarker_axes_informative_verified": None,
    }
    result = build_translation_context_report(
        context,
        snapshot,
        candidates=candidates,
        preclinical_evidence=pd.DataFrame(
            [_preclinical_record(regimen_name=None, **common_updates)]
        ),
        patient_evidence=pd.DataFrame([_patient_record(stage=None, **common_updates)]),
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=True,
    )
    summary = result.candidate_context.set_index("gene_symbol")
    assert (
        summary.loc["LAMTOR2", "report_only_preclinical_conflicting_context_family_n"]
        == 1
    )
    assert (
        summary.loc[
            "NME6", "report_only_patient_predictive_conflicting_context_family_n"
        ]
        == 1
    )


def test_mixed_compatible_and_conflicting_evidence_is_not_labeled_only():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    candidates = pd.DataFrame(
        {
            "gene_symbol": ["LAMTOR2", "NME6"],
            "screen_id": ["0007", "0007"],
            "contrast_id": ["0009", "0009"],
            "phenotype_direction": ["resistance", "resistance"],
        }
    )
    compatible = {
        "disease_subtype": None,
        "disease_subtype_id": None,
        "disease_ontology_name": None,
        "disease_ontology_version": None,
    }
    conflicting = {"disease_subtype": "ER-positive breast cancer"}
    result = build_translation_context_report(
        _context(),
        snapshot,
        candidates=candidates,
        preclinical_evidence=pd.DataFrame(
            [
                _preclinical_record("P-COMPATIBLE", **compatible),
                _preclinical_record(
                    "P-CONFLICT",
                    source_family_id="SOURCE-CONFLICT",
                    raw_data_family_id="RAW-CONFLICT",
                    **conflicting,
                ),
            ]
        ),
        patient_evidence=pd.DataFrame(
            [
                _patient_record("C-COMPATIBLE", **compatible),
                _patient_record(
                    "C-CONFLICT",
                    cohort_id="COHORT-CONFLICT",
                    source_family_id="PATIENT-SOURCE-CONFLICT",
                    raw_data_family_id="PATIENT-RAW-CONFLICT",
                    **conflicting,
                ),
            ]
        ),
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=True,
    )
    summary = result.candidate_context.set_index("gene_symbol")
    assert summary.loc["LAMTOR2", "report_only_preclinical_status"] == (
        "compatible_and_conflicting_context_present"
    )
    assert summary.loc["NME6", "report_only_patient_status"] == (
        "compatible_and_conflicting_patient_context_present"
    )


def test_explicit_unverified_extra_regimen_components_are_conflicting():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    result = build_translation_context_report(
        _context(),
        snapshot,
        candidates=pd.DataFrame(
            {
                "gene_symbol": ["LAMTOR2"],
                "screen_id": ["0007"],
                "contrast_id": ["0009"],
                "phenotype_direction": ["resistance"],
            }
        ),
        preclinical_evidence=pd.DataFrame(
            [
                _preclinical_record(
                    regimen_name="olaparib plus durvalumab",
                    regimen_active_exposure_ids_json=('["NCIT:C71721","NCIT:C12345"]'),
                    regimen_active_exposures_verified=False,
                )
            ]
        ),
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=True,
    )
    evidence_row = result.preclinical_used_evidence.iloc[0]
    assert evidence_row["report_only_regimen_match"] == "additional_active_components"
    summary_row = result.candidate_context.iloc[0]
    assert summary_row["report_only_preclinical_conflicting_context_family_n"] == 1
    assert summary_row["report_only_preclinical_status"] == "conflicting_context_only"


def test_explicit_subtype_parent_id_mismatch_is_conflicting():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    result = build_translation_context_report(
        _context(),
        snapshot,
        candidates=pd.DataFrame(
            {
                "gene_symbol": ["LAMTOR2"],
                "screen_id": ["0007"],
                "contrast_id": ["0009"],
                "phenotype_direction": ["resistance"],
            }
        ),
        preclinical_evidence=pd.DataFrame(
            [
                _preclinical_record(
                    disease_subtype_parent_id="NCIT:C99999",
                    disease_subtype_parent_binding_verified=False,
                )
            ]
        ),
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=True,
    )
    evidence_row = result.preclinical_used_evidence.iloc[0]
    assert evidence_row["report_only_subtype_match"] == "parent_id_conflict"
    summary_row = result.candidate_context.iloc[0]
    assert summary_row["report_only_preclinical_conflicting_context_family_n"] == 1
    assert summary_row["report_only_preclinical_status"] == "conflicting_context_only"


def test_conflicting_versioned_gene_identities_abort_curated_table():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    evidence = pd.DataFrame(
        [
            _preclinical_record("P-GENE-1"),
            _preclinical_record(
                "P-GENE-2",
                source_study_id="PMID:OTHER",
                source_family_id="SOURCE-OTHER",
                raw_data_family_id="RAW-OTHER",
                gene_id="SYN:DIFFERENT",
            ),
        ]
    )
    with pytest.raises(
        TranslationContextError,
        match="conflicting versioned gene identities",
    ):
        build_translation_context_report(
            _context(),
            snapshot,
            preclinical_evidence=evidence,
            evidence_cutoff_date=date(2026, 8, 28),
            target_absence_attested=True,
        )


def test_conflicting_treatment_identifier_is_excluded():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    result = build_translation_context_report(
        _context(),
        snapshot,
        patient_evidence=pd.DataFrame(
            [
                _patient_record(
                    treatment_id="NCIT:WRONG",
                    treatment_active_exposure_ids_json='["NCIT:WRONG"]',
                    treatment_ontology_name="NCIt",
                    treatment_ontology_version="26.07d",
                )
            ]
        ),
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=True,
    )
    assert result.patient_used_evidence.empty
    assert result.patient_exclusions["reason"].tolist() == ["treatment_mismatch"]


def test_direction_concordance_requires_matching_perturbation_modality():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    candidates = pd.DataFrame(
        {
            "gene_symbol": ["LAMTOR2"],
            "screen_id": ["0007"],
            "contrast_id": ["0009"],
            "phenotype_direction": ["resistance"],
        }
    )
    overexpression = pd.DataFrame(
        [_preclinical_record(perturbation_modality="overexpression")]
    )
    result = build_translation_context_report(
        _context(),
        snapshot,
        candidates=candidates,
        preclinical_evidence=overexpression,
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=True,
    )
    row = result.candidate_context.iloc[0]
    assert (
        row["report_only_preclinical_direction_concordant_exact_context_family_n"] == 0
    )
    assert (
        row["report_only_preclinical_direction_incomparable_exact_context_family_n"]
        == 1
    )


def test_direction_concordance_requires_matching_compartment_and_endpoint():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    candidates = pd.DataFrame(
        {
            "gene_symbol": ["LAMTOR2"],
            "screen_id": ["0007"],
            "contrast_id": ["0009"],
            "phenotype_direction": ["resistance"],
        }
    )
    immune_killing = pd.DataFrame(
        [
            _preclinical_record(
                perturbed_compartment="immune_cell",
                endpoint_category="immune_killing",
                model_type="immune_coculture",
            )
        ]
    )
    result = build_translation_context_report(
        _context(),
        snapshot,
        candidates=candidates,
        preclinical_evidence=immune_killing,
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=True,
    )
    row = result.candidate_context.iloc[0]
    assert row["report_only_preclinical_exact_context_family_n"] == 0
    assert row["report_only_preclinical_conflicting_context_family_n"] == 1
    assert (
        row["report_only_preclinical_direction_concordant_exact_context_family_n"] == 0
    )


@pytest.mark.parametrize(
    ("context_updates", "evidence_updates", "match_column"),
    [
        (
            {"perturbed_compartment": "tumor_cell"},
            {"perturbed_compartment": "unknown"},
            "report_only_perturbed_compartment_match",
        ),
        (
            {"perturbed_compartment": "unknown"},
            {"perturbed_compartment": "tumor_cell"},
            "report_only_perturbed_compartment_match",
        ),
        (
            {"perturbed_compartment": "unknown"},
            {"perturbed_compartment": "unknown"},
            "report_only_perturbed_compartment_match",
        ),
        (
            {"screen_endpoint_category": "drug_response_viability"},
            {"endpoint_category": "unknown"},
            "report_only_endpoint_category_match",
        ),
        (
            {"screen_endpoint_category": "unknown"},
            {"endpoint_category": "drug_response_viability"},
            "report_only_endpoint_category_match",
        ),
        (
            {"screen_endpoint_category": "unknown"},
            {"endpoint_category": "unknown"},
            "report_only_endpoint_category_match",
        ),
    ],
)
def test_unknown_preclinical_design_axis_is_unresolved_not_exact_or_conflicting(
    context_updates, evidence_updates, match_column
):
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    candidates = pd.DataFrame(
        {
            "gene_symbol": ["LAMTOR2"],
            "screen_id": ["0007"],
            "contrast_id": ["0009"],
            "phenotype_direction": ["resistance"],
        }
    )
    result = build_translation_context_report(
        _context(**context_updates),
        snapshot,
        candidates=candidates,
        preclinical_evidence=pd.DataFrame([_preclinical_record(**evidence_updates)]),
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=True,
    )
    assert result.preclinical_used_evidence.iloc[0][match_column] == "unresolved"
    row = result.candidate_context.iloc[0]
    assert row["report_only_preclinical_exact_context_family_n"] == 0
    assert row["report_only_preclinical_conflicting_context_family_n"] == 0
    assert row["report_only_preclinical_compatible_nonexact_context_family_n"] == 1
    assert row["report_only_preclinical_status"] == "compatible_nonexact_context_only"


@pytest.mark.parametrize(
    ("candidate_direction", "evidence_direction"),
    [
        ("unknown", "unknown"),
        ("resistance", "unknown"),
        ("unknown", "resistance"),
    ],
)
def test_unknown_direction_is_unresolved_not_concordant_or_discordant(
    candidate_direction, evidence_direction
):
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    candidates = pd.DataFrame(
        {
            "gene_symbol": ["LAMTOR2"],
            "screen_id": ["0007"],
            "contrast_id": ["0009"],
            "phenotype_direction": [candidate_direction],
        }
    )
    evidence_updates = {}
    if evidence_direction == "unknown":
        evidence_updates = {
            "phenotype_direction": "unknown",
            "direction_rule_id": None,
            "direction_rule_version": None,
            "direction_inference_status": "not_assessed",
            "direction_inference_curator_verified": False,
            "native_effect": None,
            "native_effect_type": None,
            "native_reference_group": None,
            "effect_numeric": None,
            "effect_type": None,
            "p_value": None,
            "sample_n": None,
        }
    result = build_translation_context_report(
        _context(),
        snapshot,
        candidates=candidates,
        preclinical_evidence=pd.DataFrame([_preclinical_record(**evidence_updates)]),
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=True,
    )
    row = result.candidate_context.iloc[0]
    assert (
        row["report_only_preclinical_direction_concordant_exact_context_family_n"] == 0
    )
    assert (
        row["report_only_preclinical_direction_discordant_exact_context_family_n"] == 0
    )
    assert (
        row["report_only_preclinical_direction_unresolved_exact_context_family_n"] == 1
    )


def test_candidate_context_binding_and_temporal_cutoff_fail_closed():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    wrong_candidates = pd.DataFrame(
        {
            "gene_symbol": ["LAMTOR2"],
            "screen_id": ["WRONG"],
            "contrast_id": ["0009"],
            "phenotype_direction": ["resistance"],
        }
    )
    with pytest.raises(TranslationContextError, match="screen_id conflicts"):
        build_translation_context_report(
            _context(),
            snapshot,
            candidates=wrong_candidates,
            evidence_cutoff_date=date(2026, 8, 28),
        )
    spoofed = wrong_candidates.assign(
        screen_id="0007",
        report_only_patient_status="predictive_interaction_evidence_present",
    )
    with pytest.raises(TranslationContextError, match="reserved report columns"):
        build_translation_context_report(
            _context(),
            snapshot,
            candidates=spoofed,
            evidence_cutoff_date=date(2026, 8, 28),
        )
    with pytest.raises(TranslationContextError, match="later than context_date"):
        build_translation_context_report(
            _context(),
            snapshot,
            evidence_cutoff_date=date(2026, 8, 29),
        )


def test_target_family_exclusion_is_transitive():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    candidates = pd.DataFrame(
        {
            "gene_symbol": ["LAMTOR2"],
            "screen_id": ["0007"],
            "contrast_id": ["0009"],
            "phenotype_direction": ["resistance"],
        }
    )
    evidence = pd.DataFrame(
        [
            _preclinical_record(
                "TARGET", source_family_id="TARGET-SOURCE", raw_data_family_id="RAW-X"
            ),
            _preclinical_record(
                "REANALYSIS",
                source_family_id="OTHER-SOURCE",
                raw_data_family_id="RAW-X",
            ),
        ]
    )
    result = build_translation_context_report(
        _context(),
        snapshot,
        candidates=candidates,
        preclinical_evidence=evidence,
        evidence_cutoff_date=date(2026, 8, 28),
        target_source_family_id="TARGET-SOURCE",
    )
    assert result.preclinical_used_evidence.empty
    assert result.preclinical_exclusions["reason"].tolist() == [
        "target_source_family",
        "target_family_component",
    ]


def test_excluded_bridge_does_not_split_one_provenance_family():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    candidates = pd.DataFrame(
        {
            "gene_symbol": ["LAMTOR2"],
            "screen_id": ["0007"],
            "contrast_id": ["0009"],
            "phenotype_direction": ["resistance"],
        }
    )
    evidence = pd.DataFrame(
        [
            _preclinical_record(
                "A",
                source_study_id="PMID:A",
                source_family_id="S1",
                raw_data_family_id="R1",
            ),
            _preclinical_record(
                "BRIDGE",
                source_study_id="PMID:X",
                source_family_id="S1",
                raw_data_family_id="R2",
                treatment_name="talazoparib",
                treatment_id="NCIT:C957",
                regimen_name="talazoparib monotherapy",
                regimen_active_exposure_ids_json='["NCIT:C957"]',
            ),
            _preclinical_record(
                "B",
                source_study_id="PMID:B",
                source_family_id="S2",
                raw_data_family_id="R2",
            ),
        ]
    )
    result = build_translation_context_report(
        _context(),
        snapshot,
        candidates=candidates,
        preclinical_evidence=evidence,
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=True,
    )
    assert result.preclinical_exclusions["reason"].tolist() == ["treatment_mismatch"]
    assert result.candidate_context.iloc[0]["report_only_preclinical_family_n"] == 1


def test_explicit_subtype_and_typed_biomarker_conflicts_take_precedence():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    subtype_conflict = _preclinical_record(
        disease_subtype_id="NCIT:C99999",
        disease_subtype_parent_binding_verified=False,
    )
    biomarker_conflict = _patient_record(
        biomarker_context="BRCA",
        biomarker_feature_type="rna_expression",
        biomarker_state="high",
        biomarker_specimen_type="tumor",
        biomarker_measurement_timepoint="pretreatment",
    )
    result = build_translation_context_report(
        _context(),
        snapshot,
        preclinical_evidence=pd.DataFrame([subtype_conflict]),
        patient_evidence=pd.DataFrame([biomarker_conflict]),
        evidence_cutoff_date=date(2026, 8, 28),
        biomarker_aliases=["BRCA"],
        target_absence_attested=True,
    )
    assert (
        result.preclinical_used_evidence.iloc[0]["report_only_subtype_match"]
        == "id_conflict"
    )
    assert (
        result.patient_used_evidence.iloc[0]["report_only_biomarker_match"]
        == "typed_axis_conflict"
    )


def test_unverified_self_exclusion_is_explicit():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    result = build_translation_context_report(
        _context(),
        snapshot,
        candidates=pd.DataFrame(
            {
                "gene_symbol": ["LAMTOR2"],
                "screen_id": ["0007"],
                "contrast_id": ["0009"],
                "phenotype_direction": ["resistance"],
            }
        ),
        preclinical_evidence=pd.DataFrame([_preclinical_record()]),
        evidence_cutoff_date=date(2026, 8, 28),
    )
    assert result.candidate_context.iloc[0]["report_only_preclinical_status"] == (
        "independence_unverified"
    )


@pytest.mark.parametrize("value", ["false", 0, 1])
def test_target_absence_attestation_rejects_boolean_coercion(value):
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    with pytest.raises(
        TranslationContextError,
        match="target_absence_attested must be a literal boolean",
    ):
        build_translation_context_report(
            _context(),
            snapshot,
            preclinical_evidence=pd.DataFrame([_preclinical_record()]),
            evidence_cutoff_date=date(2026, 8, 28),
            target_absence_attested=value,
        )


@pytest.mark.parametrize("value", [False, True, np.bool_(False), np.bool_(True)])
def test_target_absence_attestation_accepts_literal_booleans(value):
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    result = build_translation_context_report(
        _context(),
        snapshot,
        preclinical_evidence=pd.DataFrame([_preclinical_record()]),
        evidence_cutoff_date=date(2026, 8, 28),
        target_absence_attested=value,
    )
    expected = bool(value)
    assert result.metadata["independence"]["target_absence_attested"] is expected
    assert result.metadata["independence"]["preclinical_verified"] is expected


def test_target_source_family_is_excluded_from_prior_support():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    result = build_translation_context_report(
        _context(),
        snapshot,
        candidates=pd.DataFrame(
            {
                "gene_symbol": ["LAMTOR2"],
                "screen_id": ["0007"],
                "contrast_id": ["0009"],
                "phenotype_direction": ["resistance"],
            }
        ),
        preclinical_evidence=pd.DataFrame([_preclinical_record()]),
        evidence_cutoff_date=date(2026, 8, 28),
        target_source_family_id="SOURCE-1",
    )
    assert result.preclinical_used_evidence.empty
    assert result.preclinical_exclusions["reason"].tolist() == ["target_source_family"]
    assert result.candidate_context.iloc[0]["report_only_preclinical_family_n"] == 0


def test_post_cutoff_target_family_cannot_verify_prior_evidence_independence():
    snapshot = clinical_trials_snapshot_from_document(_snapshot_document(_trial_set()))
    candidates = pd.DataFrame(
        {
            "gene_symbol": ["LAMTOR2"],
            "screen_id": ["0007"],
            "contrast_id": ["0009"],
            "phenotype_direction": ["resistance"],
        }
    )
    evidence = pd.DataFrame(
        [
            _preclinical_record("PRIOR"),
            _preclinical_record(
                "FUTURE-TARGET",
                source_study_id="PMID:FUTURE",
                source_family_id="TARGET-SOURCE",
                raw_data_family_id="TARGET-RAW",
                gene_symbol="OTHER",
                gene_id="SYN:OTHER",
                available_date="2026-08-29",
                retrieved_date="2026-08-29",
            ),
        ]
    )
    result = build_translation_context_report(
        _context(),
        snapshot,
        candidates=candidates,
        preclinical_evidence=evidence,
        evidence_cutoff_date=date(2026, 8, 28),
        target_source_family_id="TARGET-SOURCE",
    )
    assert result.preclinical_exclusions[["evidence_id", "reason"]].to_dict(
        "records"
    ) == [{"evidence_id": "FUTURE-TARGET", "reason": "post_cutoff"}]
    assert result.metadata["independence"]["preclinical_verified"] is False
    row = result.candidate_context.iloc[0]
    assert row["report_only_preclinical_family_n"] == 1
    assert row["report_only_preclinical_status"] == "independence_unverified"


def test_car_t_context_reports_core_model_not_applicable():
    context = _context(
        treatment_name="CEA-directed CAR-T",
        treatment_id=None,
        treatment_ontology_name=None,
        treatment_ontology_version=None,
        treatment_modality="cell_therapy",
        regimen_name=None,
        cancer_type="colorectal cancer",
        cancer_id=None,
        disease_subtype=None,
        disease_subtype_id=None,
        disease_ontology_name=None,
        disease_ontology_version=None,
        biomarker_context="CEA",
        biomarker_feature_type="protein_expression",
        biomarker_state="positive",
        biomarker_specimen_type="tumor",
        biomarker_measurement_timepoint="pretreatment",
        screen_endpoint_category="immune_killing",
    )
    snapshot = clinical_trials_snapshot_from_document(
        _snapshot_document(
            [
                _trial(
                    "NCT00000004",
                    interventions=[
                        {"type": "BIOLOGICAL", "name": "CEA-directed CAR-T"}
                    ],
                    conditions=["Colorectal Cancer"],
                    keywords=["CEA"],
                )
            ]
        )
    )
    result = build_translation_context_report(
        context,
        snapshot,
        evidence_cutoff_date=date(2026, 8, 28),
    )
    applicability = result.metadata["core_model_applicability"]
    assert applicability["status"] == "not_applicable"
    assert "intervention_not_small_molecule" in applicability["reasons"]
    assert "endpoint_not_drug_response_viability" in applicability["reasons"]


def test_translation_columns_are_blocked_from_success_model():
    for column in (
        "report_only_patient_predictive_exact_context_family_n",
        "clinical_trial_count",
        "preclinical_pdx_support",
        "patient_molecular_outcome",
    ):
        with pytest.raises(ValueError, match="leakage fields"):
            validate_success_feature_columns(["guide_n", column])
    for contract in (PatientMolecularEvidenceRecord, PreclinicalEvidenceRecord):
        for column in contract.model_fields:
            with pytest.raises(ValueError, match="leakage fields"):
                validate_success_feature_columns(["guide_n", column])


def test_translation_schemas_publish_runtime_semantic_rules():
    patient_schema = PatientMolecularEvidenceRecord.model_json_schema()
    preclinical_schema = PreclinicalEvidenceRecord.model_json_schema()
    assert any(
        "controlled effect scale" in rule for rule in patient_schema["x-semantic-rules"]
    )
    assert any(
        "genotype-by-treatment" in rule
        for rule in preclinical_schema["x-semantic-rules"]
    )


@pytest.mark.parametrize(
    ("contract", "record_factory"),
    [
        (PreclinicalEvidenceRecord, _preclinical_record),
        (PatientMolecularEvidenceRecord, _patient_record),
    ],
)
def test_translation_evidence_requires_raw_data_family(contract, record_factory):
    record = record_factory()
    record.pop("raw_data_family_id")
    with pytest.raises(ValueError, match="raw_data_family_id"):
        contract.model_validate(record)


def _write_test_rank_screen_bundle(tmp_path):
    mageck_path = tmp_path / "mageck.tsv"
    output_dir = tmp_path / "rank-screen"
    pd.DataFrame(
        {
            "id": ["LAMTOR2", "NME6"],
            "num": [4, 4],
            "pos|score": [0.01, 0.02],
            "pos|p-value": [0.01, 0.02],
            "pos|fdr": [0.02, 0.03],
            "pos|rank": [1, 2],
            "pos|goodsgrna": [3, 3],
            "pos|lfc": [0.8, 0.6],
            "neg|score": [0.4, 0.5],
            "neg|p-value": [0.4, 0.5],
            "neg|fdr": [0.5, 0.6],
            "neg|rank": [1, 2],
            "neg|goodsgrna": [2, 2],
            "neg|lfc": [-0.2, -0.1],
        }
    ).to_csv(mageck_path, sep="\t", index=False)
    args = build_parser().parse_args(
        [
            "rank-screen",
            "--mageck-summary",
            str(mageck_path),
            "--screen-id",
            "0007",
            "--contrast-id",
            "0009",
            "--positive-tail-means",
            "resistance",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert args.func(args) == 0
    return output_dir / "ranked_candidates.tsv", output_dir / "run_manifest.json"


def _write_test_count_rank_screen_bundle(tmp_path, *, combined):
    counts_path = tmp_path / "counts.tsv"
    samples_path = tmp_path / "samples.tsv"
    output_dir = tmp_path / (
        "combined-rank-screen" if combined else "count-rank-screen"
    )
    pd.DataFrame(
        {
            "sgrna_id": ["g1", "g2", "g3", "g4"],
            "gene_symbol": ["LAMTOR2", "LAMTOR2", "NME6", "NME6"],
            "c1": [100, 120, 100, 120],
            "c2": [110, 115, 110, 115],
            "t1": [500, 450, 25, 30],
            "t2": [480, 430, 30, 35],
        }
    ).to_csv(counts_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "sample_id": ["c1", "c2", "t1", "t2"],
            "screen_id": ["0007"] * 4,
            "contrast_id": ["0009"] * 4,
            "condition_role": ["control", "control", "treatment", "treatment"],
            "replicate": [1, 2, 1, 2],
        }
    ).to_csv(samples_path, sep="\t", index=False)
    command = [
        "rank-screen",
        "--counts",
        str(counts_path),
        "--samples",
        str(samples_path),
        "--screen-id",
        "0007",
        "--contrast-id",
        "0009",
        "--positive-lfc-means",
        "resistance",
        "--output-dir",
        str(output_dir),
    ]
    if combined:
        mageck_source = tmp_path / "mageck-source"
        mageck_source.mkdir()
        mageck_path, _ = _write_test_rank_screen_bundle(mageck_source)
        source_manifest = json.loads(
            (mageck_path.parent / "run_manifest.json").read_text(encoding="utf-8")
        )
        original_mageck_path = Path(source_manifest["inputs"]["mageck_summary"]["path"])
        command.extend(
            [
                "--mageck-summary",
                str(original_mageck_path),
                "--positive-tail-means",
                "resistance",
            ]
        )
    args = build_parser().parse_args(command)
    assert args.func(args) == 0
    return output_dir / "ranked_candidates.tsv", output_dir / "run_manifest.json"


def _read_bound_rank_bundle(candidates_path, manifest_path):
    candidates = pd.read_csv(
        candidates_path,
        sep="\t",
        dtype={"screen_id": "string", "contrast_id": "string"},
    )
    return cli_module._read_rank_screen_manifest_snapshot(
        manifest_path,
        candidate_path=candidates_path,
        candidates=candidates,
        candidate_sha256=hashlib.sha256(candidates_path.read_bytes()).hexdigest(),
        screen_id="0007",
        contrast_id="0009",
    )


def _rewrite_bound_candidates(candidates_path, manifest_path, candidates):
    candidates.to_csv(candidates_path, sep="\t", index=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ranked_output = manifest["outputs"]["ranked_candidates"]
    ranked_output["sha256"] = hashlib.sha256(candidates_path.read_bytes()).hexdigest()
    ranked_output["row_count"] = len(candidates)
    ranked_output["columns"] = candidates.columns.tolist()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("mode", ["mageck", "counts", "mageck_plus_counts"])
def test_every_rank_screen_mode_round_trips_through_manifest_validator(tmp_path, mode):
    if mode == "mageck":
        candidates_path, manifest_path = _write_test_rank_screen_bundle(tmp_path)
    else:
        candidates_path, manifest_path = _write_test_count_rank_screen_bundle(
            tmp_path,
            combined=mode == "mageck_plus_counts",
        )
    document, _ = _read_bound_rank_bundle(candidates_path, manifest_path)
    assert document["mode"] == mode


def test_bound_mageck_rank_source_must_follow_producer_priority(tmp_path):
    candidates_path, manifest_path = _write_test_rank_screen_bundle(tmp_path)
    candidates = pd.read_csv(
        candidates_path,
        sep="\t",
        dtype={"screen_id": "string", "contrast_id": "string"},
    )
    candidates["screen_signal_rank_source"] = "derived_from_mageck_fdr"
    _rewrite_bound_candidates(candidates_path, manifest_path, candidates)
    with pytest.raises(ValueError, match="producer priority"):
        _read_bound_rank_bundle(candidates_path, manifest_path)


def test_bound_rank_screen_rechecks_native_scientific_domains(tmp_path):
    candidates_path, manifest_path = _write_test_rank_screen_bundle(tmp_path)
    candidates = pd.read_csv(
        candidates_path,
        sep="\t",
        dtype={"screen_id": "string", "contrast_id": "string"},
    )
    candidates.loc[0, "mageck_fdr"] = -5.0
    _rewrite_bound_candidates(candidates_path, manifest_path, candidates)
    with pytest.raises(ValueError, match="mageck_fdr is out of range"):
        _read_bound_rank_bundle(candidates_path, manifest_path)

    count_dir = tmp_path / "count-domain"
    count_dir.mkdir()
    candidates_path, manifest_path = _write_test_count_rank_screen_bundle(
        count_dir, combined=False
    )
    candidates = pd.read_csv(
        candidates_path,
        sep="\t",
        dtype={"screen_id": "string", "contrast_id": "string"},
    )
    candidates.loc[0, "low_count_fraction"] = 1.5
    _rewrite_bound_candidates(candidates_path, manifest_path, candidates)
    with pytest.raises(ValueError, match="low_count_fraction is out of range"):
        _read_bound_rank_bundle(candidates_path, manifest_path)

    missing_dir = tmp_path / "count-missing"
    missing_dir.mkdir()
    candidates_path, manifest_path = _write_test_count_rank_screen_bundle(
        missing_dir, combined=False
    )
    candidates = pd.read_csv(
        candidates_path,
        sep="\t",
        dtype={"screen_id": "string", "contrast_id": "string"},
    )
    candidates.loc[0, "guide_n"] = None
    _rewrite_bound_candidates(candidates_path, manifest_path, candidates)
    with pytest.raises(ValueError, match="intrinsic count metrics"):
        _read_bound_rank_bundle(candidates_path, manifest_path)

    agreement_dir = tmp_path / "count-agreement"
    agreement_dir.mkdir()
    candidates_path, manifest_path = _write_test_count_rank_screen_bundle(
        agreement_dir, combined=False
    )
    candidates = pd.read_csv(
        candidates_path,
        sep="\t",
        dtype={"screen_id": "string", "contrast_id": "string"},
    )
    candidates.loc[0, "guide_direction_agreement"] = 0.0
    _rewrite_bound_candidates(candidates_path, manifest_path, candidates)
    with pytest.raises(ValueError, match="direction agreement is inconsistent"):
        _read_bound_rank_bundle(candidates_path, manifest_path)

    discrete_dir = tmp_path / "count-discrete"
    discrete_dir.mkdir()
    candidates_path, manifest_path = _write_test_count_rank_screen_bundle(
        discrete_dir, combined=False
    )
    candidates = pd.read_csv(
        candidates_path,
        sep="\t",
        dtype={"screen_id": "string", "contrast_id": "string"},
    )
    candidates.loc[0, "positive_guide_fraction"] = 0.2
    candidates.loc[0, "negative_guide_fraction"] = 0.8
    candidates.loc[0, "neutral_guide_fraction"] = 0.0
    candidates.loc[0, "guide_direction_agreement"] = 0.8
    _rewrite_bound_candidates(candidates_path, manifest_path, candidates)
    with pytest.raises(ValueError, match="conflicts with its denominator"):
        _read_bound_rank_bundle(candidates_path, manifest_path)


def test_bound_count_direction_is_recomputed_from_lfc_and_deadband(tmp_path):
    candidates_path, manifest_path = _write_test_count_rank_screen_bundle(
        tmp_path, combined=False
    )
    candidates = pd.read_csv(
        candidates_path,
        sep="\t",
        dtype={"screen_id": "string", "contrast_id": "string"},
    )
    candidates.loc[0, "median_guide_lfc"] *= -1
    _rewrite_bound_candidates(candidates_path, manifest_path, candidates)
    with pytest.raises(ValueError, match="guide LFC, tail, and phenotype"):
        _read_bound_rank_bundle(candidates_path, manifest_path)


def test_bound_combined_concordance_is_recomputed(tmp_path):
    candidates_path, manifest_path = _write_test_count_rank_screen_bundle(
        tmp_path, combined=True
    )
    candidates = pd.read_csv(
        candidates_path,
        sep="\t",
        dtype={"screen_id": "string", "contrast_id": "string"},
    )
    observed = candidates.loc[0, "mageck_guide_direction_agreement"]
    candidates.loc[0, "mageck_guide_direction_agreement"] = (
        str(observed).casefold() != "true"
    )
    _rewrite_bound_candidates(candidates_path, manifest_path, candidates)
    with pytest.raises(ValueError, match="concordance"):
        _read_bound_rank_bundle(candidates_path, manifest_path)

    scope_dir = tmp_path / "combined-scope"
    scope_dir.mkdir()
    candidates_path, manifest_path = _write_test_count_rank_screen_bundle(
        scope_dir, combined=True
    )
    candidates = pd.read_csv(
        candidates_path,
        sep="\t",
        dtype={"screen_id": "string", "contrast_id": "string"},
    )
    candidates["guide_screen_signal_percentile_scope"] = "forged_scope"
    _rewrite_bound_candidates(candidates_path, manifest_path, candidates)
    with pytest.raises(ValueError, match="guide percentile scope is invalid"):
        _read_bound_rank_bundle(candidates_path, manifest_path)


def test_bound_rank_screen_rejects_dropped_native_signal_columns(tmp_path):
    candidates_path, manifest_path = _write_test_rank_screen_bundle(tmp_path)
    candidates = pd.read_csv(
        candidates_path,
        sep="\t",
        dtype={"screen_id": "string", "contrast_id": "string"},
    )
    candidates = candidates.drop(
        columns=[column for column in candidates if column.startswith("mageck_")]
    )
    _rewrite_bound_candidates(candidates_path, manifest_path, candidates)
    with pytest.raises(ValueError, match="requires candidate column"):
        _read_bound_rank_bundle(candidates_path, manifest_path)


def test_bound_rank_screen_recomputes_qc_and_report_semantics(tmp_path):
    candidates_path, manifest_path = _write_test_rank_screen_bundle(tmp_path)
    qc_path = manifest_path.parent / "qc_summary.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    qc["warnings"].append("forged warning")
    qc_path.write_text(json.dumps(qc), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["qc_summary"]["sha256"] = hashlib.sha256(
        qc_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="QC semantics"):
        _read_bound_rank_bundle(candidates_path, manifest_path)

    report_forgery_dir = tmp_path / "report-forgery"
    report_forgery_dir.mkdir()
    candidates_path, manifest_path = _write_test_rank_screen_bundle(report_forgery_dir)
    report_path = manifest_path.parent / "report.md"
    report_path.write_text(
        report_path.read_text(encoding="utf-8") + "\nforged claim\n",
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["report"]["sha256"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="report semantics"):
        _read_bound_rank_bundle(candidates_path, manifest_path)


def test_bound_rank_screen_recomputes_rank_from_declared_source(tmp_path):
    candidates_path, manifest_path = _write_test_rank_screen_bundle(tmp_path)
    candidates = pd.read_csv(
        candidates_path,
        sep="\t",
        dtype={"screen_id": "string", "contrast_id": "string"},
    )
    candidates.loc[0, "mageck_rank"] = 2
    candidates.to_csv(candidates_path, sep="\t", index=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["ranked_candidates"]["sha256"] = hashlib.sha256(
        candidates_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="rank disagrees with its declared source"):
        _read_bound_rank_bundle(candidates_path, manifest_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("created_at_utc", "2026-08-28T13:00:00+01:00", "must use UTC"),
        ("row_count", True, "metadata disagrees"),
    ],
)
def test_bound_rank_screen_manifest_rejects_ambiguous_scalar_types(
    tmp_path, field, value, message
):
    candidates_path, manifest_path = _write_test_rank_screen_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field == "created_at_utc":
        manifest[field] = value
    else:
        manifest["outputs"]["ranked_candidates"][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _read_bound_rank_bundle(candidates_path, manifest_path)


def test_cli_offline_bundle_is_atomic_and_checksum_bound(tmp_path, monkeypatch):
    trials_path = tmp_path / "clinicaltrials.json"
    mageck_path = tmp_path / "mageck.tsv"
    rank_output_dir = tmp_path / "rank-screen"
    candidates_path = rank_output_dir / "ranked_candidates.tsv"
    candidate_manifest_path = rank_output_dir / "run_manifest.json"
    output_dir = tmp_path / "translation"
    trials_path.write_text(
        json.dumps(_snapshot_document(_trial_set())), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "id": ["LAMTOR2"],
            "num": [4],
            "pos|score": [0.01],
            "pos|p-value": [0.01],
            "pos|fdr": [0.02],
            "pos|rank": [1],
            "pos|goodsgrna": [3],
            "pos|lfc": [0.8],
            "neg|score": [0.4],
            "neg|p-value": [0.4],
            "neg|fdr": [0.5],
            "neg|rank": [1],
            "neg|goodsgrna": [2],
            "neg|lfc": [-0.2],
        }
    ).to_csv(mageck_path, sep="\t", index=False)
    rank_args = build_parser().parse_args(
        [
            "rank-screen",
            "--mageck-summary",
            str(mageck_path),
            "--screen-id",
            "0007",
            "--contrast-id",
            "0009",
            "--positive-tail-means",
            "resistance",
            "--output-dir",
            str(rank_output_dir),
        ]
    )
    assert rank_args.func(rank_args) == 0
    args = build_parser().parse_args(
        [
            "summarize-translation-context",
            "--context-id",
            "CTX",
            "--screen-id",
            "0007",
            "--contrast-id",
            "0009",
            "--treatment",
            "olaparib",
            "--treatment-modality",
            "small_molecule",
            "--treatment-entity-alias",
            "Lynparza",
            "--cancer-type",
            "breast cancer",
            "--disease-subtype",
            "triple-negative breast cancer",
            "--no-disease-subtype-parent-binding-verified",
            "--subtype-entity-alias",
            "TNBC",
            "--screen-perturbation-modality",
            "CRISPR_KO",
            "--perturbed-compartment",
            "tumor_cell",
            "--screen-endpoint-category",
            "drug_response_viability",
            "--context-date",
            "2026-08-28",
            "--evidence-cutoff-date",
            "2026-08-28",
            "--clinicaltrials-json",
            str(trials_path),
            "--candidates",
            str(candidates_path),
            "--candidate-manifest",
            str(candidate_manifest_path),
            "--target-not-in-evidence-catalog",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert args.func(args) == 0
    manifest = json.loads((output_dir / "summary.json").read_text())
    assert manifest["candidate_input"]["rank_screen_manifest_bound"] is True
    assert manifest["candidate_input"]["ranking_claim"] == (
        "rank_screen_v1_contract_validated_checksum_bound"
    )
    for output in manifest["outputs"].values():
        path = output_dir / output["filename"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == output["sha256"]
    candidates = pd.read_csv(
        output_dir / "candidate_translation_context.tsv",
        sep="\t",
        dtype={"screen_id": "string", "contrast_id": "string"},
    )
    assert candidates["screen_id"].tolist() == ["0007", "0007"]
    assert candidates["contrast_id"].tolist() == ["0009", "0009"]
    with pytest.raises(FileExistsError):
        args.func(args)

    args.output_dir = str(tmp_path / "missing-manifest")
    args.candidate_manifest = None
    with pytest.raises(ValueError, match="ranked candidates require"):
        args.func(args)

    minimal_manifest_path = rank_output_dir / "minimal_manifest.json"
    minimal_manifest_path.write_text(
        json.dumps(
            {
                "report_type": "screen_signal_baseline",
                "parameters": {
                    "resolved_screen_ids": ["0007"],
                    "resolved_contrast_ids": ["0009"],
                },
                "outputs": {
                    "ranked_candidates": {
                        "sha256": hashlib.sha256(
                            candidates_path.read_bytes()
                        ).hexdigest()
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    args.output_dir = str(tmp_path / "minimal-manifest")
    args.candidate_manifest = str(minimal_manifest_path)
    with pytest.raises(ValueError, match="unsupported top-level schema"):
        args.func(args)

    unranked_path = tmp_path / "unranked_candidates.tsv"
    pd.DataFrame(
        {
            "gene_symbol": ["LAMTOR2"],
            "screen_id": ["0007"],
            "contrast_id": ["0009"],
            "phenotype_direction": ["resistance"],
        }
    ).to_csv(unranked_path, sep="\t", index=False)
    unranked_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    unranked_manifest["outputs"]["ranked_candidates"]["sha256"] = hashlib.sha256(
        unranked_path.read_bytes()
    ).hexdigest()
    unranked_manifest_path = tmp_path / "unranked_manifest.json"
    unranked_manifest_path.write_text(json.dumps(unranked_manifest), encoding="utf-8")
    args.candidates = str(unranked_path)
    args.candidate_manifest = str(unranked_manifest_path)
    args.output_dir = str(tmp_path / "unranked-manifest")
    with pytest.raises(ValueError, match="cannot bind structurally unranked"):
        args.func(args)

    partial_rank_path = tmp_path / "partial_rank_candidates.tsv"
    pd.DataFrame(
        {
            "gene_symbol": ["LAMTOR2"],
            "screen_id": ["0007"],
            "contrast_id": ["0009"],
            "phenotype_direction": ["resistance"],
            "screen_signal_rank": [1],
        }
    ).to_csv(partial_rank_path, sep="\t", index=False)
    args.candidates = str(partial_rank_path)
    args.candidate_manifest = None
    args.output_dir = str(tmp_path / "partial-ranking")
    with pytest.raises(ValueError, match="both ranking_type and screen_signal_rank"):
        args.func(args)

    args.candidates = str(candidates_path)

    bad_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    bad_manifest["outputs"]["ranked_candidates"]["sha256"] = "0" * 64
    bad_manifest_path = tmp_path / "bad_manifest.json"
    bad_manifest_path.write_text(json.dumps(bad_manifest), encoding="utf-8")
    args.output_dir = str(tmp_path / "bad-manifest")
    args.candidate_manifest = str(bad_manifest_path)
    with pytest.raises(ValueError, match="SHA-256 disagrees"):
        args.func(args)

    args.output_dir = str(tmp_path / "translation-mutated")
    args.candidate_manifest = str(candidate_manifest_path)
    original_write_frame = cli_module._write_frame
    mutated = False

    def mutating_write_frame(frame, path):
        nonlocal mutated
        original_write_frame(frame, path)
        if not mutated:
            candidates_path.write_bytes(candidates_path.read_bytes() + b"\n")
            mutated = True

    monkeypatch.setattr(cli_module, "_write_frame", mutating_write_frame)
    with pytest.raises(ValueError, match="input files changed during the run"):
        args.func(args)
    assert not (tmp_path / "translation-mutated").exists()
