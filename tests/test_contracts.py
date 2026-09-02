import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from crispr_evidencerank.contracts import (
    CONTRACTS,
    DOCUMENT_CONTRACTS,
    INTAKE_POLICY_V2_BENCHMARK_RULE_IDS,
    INTAKE_POLICY_V2_SCOPE_RULE_IDS,
    CandidateRecord,
    ContrastRecord,
    DataAssetRecord,
    EvidenceRecord,
    SampleRecord,
    ScreenRecord,
    ValidationEventRecord,
    validate_records,
    validate_registry_integrity,
)

ROOT = Path(__file__).resolve().parents[1]


def _assert_exported_schema_matches_runtime(name, model):
    schema_path = ROOT / "schemas" / f"{name}.schema.json"
    exported_text = schema_path.read_text(encoding="utf-8")
    expected_text = json.dumps(model.model_json_schema(), indent=2)
    assert exported_text == expected_text
    assert json.loads(exported_text) == model.model_json_schema()


def base_event():
    return {
        "event_id": "E1",
        "study_id": "S1",
        "screen_id": "SC1",
        "contrast_id": "C1",
        "gene_symbol": "GENE1",
        "drug_name": "DRUG",
        "cell_line": "CELL",
        "perturbation_modality": "CRISPR_KO",
        "phenotype_direction": "resistance",
        "label_code": "V2",
        "testing_status": "tested",
        "perturbation_confirmed": True,
        "independent_reagent_count": 2,
        "orthogonal_perturbation": False,
        "appropriate_control": True,
        "assay_adequate": True,
        "phenotype_reproduced": True,
        "opposite_direction_reproduced": False,
        "rescue_performed": False,
        "causal_reversal_performed": False,
        "effect_size": 1.2,
        "effect_metric": "log2_fc",
        "p_value": 0.01,
        "source_url": "https://example.org/paper",
        "source_locator": "Figure 2",
        "curator": "Diana",
        "adjudication_status": "single_curator",
        "notes": None,
    }


def test_exported_contract_schemas_match_runtime_models():
    for name, model in CONTRACTS.items():
        _assert_exported_schema_matches_runtime(name, model)


def test_exported_document_schemas_match_runtime_models():
    assert "clinicaltrials_gov_snapshot_manifest" in DOCUMENT_CONTRACTS
    for name, model in DOCUMENT_CONTRACTS.items():
        _assert_exported_schema_matches_runtime(name, model)


def test_snapshot_manifest_schema_hash_matches_packaged_build_metadata():
    schema_path = ROOT / "schemas/clinicaltrials_gov_snapshot_manifest.schema.json"
    build_metadata = json.loads(
        (ROOT / "src/crispr_evidencerank/_build_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        build_metadata["clinicaltrials_gov_snapshot_manifest_schema_sha256"]
        == hashlib.sha256(schema_path.read_bytes()).hexdigest()
    )


def test_valid_v2_event():
    event = ValidationEventRecord.model_validate(base_event())
    assert event.label_code == "V2"


def test_v3_requires_rescue():
    record = base_event()
    record["label_code"] = "V3"
    valid, errors = validate_records(pd.DataFrame([record]), ValidationEventRecord)
    assert valid.empty
    assert not errors.empty


def test_v2_requires_appropriate_control():
    record = base_event()
    record["appropriate_control"] = False
    valid, errors = validate_records(pd.DataFrame([record]), ValidationEventRecord)
    assert valid.empty
    assert not errors.empty


def test_validation_event_rejects_directional_contradiction():
    record = base_event()
    record["opposite_direction_reproduced"] = True
    valid, errors = validate_records(pd.DataFrame([record]), ValidationEventRecord)
    assert valid.empty
    assert not errors.empty


def test_f0_rejects_opposite_direction_and_u_rejects_outcome_evidence():
    failed = base_event()
    failed["label_code"] = "F0"
    failed["phenotype_reproduced"] = False
    failed["opposite_direction_reproduced"] = True
    valid, errors = validate_records(pd.DataFrame([failed]), ValidationEventRecord)
    assert valid.empty
    assert not errors.empty

    unknown = base_event()
    unknown["label_code"] = "U"
    unknown["testing_status"] = "unknown"
    valid, errors = validate_records(pd.DataFrame([unknown]), ValidationEventRecord)
    assert valid.empty
    assert not errors.empty


def test_candidate_contract_and_empty_table_preflight():
    candidate = {
        "study_id": "S1",
        "screen_id": "SC1",
        "contrast_id": "C1",
        "gene_symbol": "GENE1",
        "phenotype_direction": "resistance",
        "label_code": "U",
        "testing_status": "unknown",
    }
    valid, errors = validate_records(pd.DataFrame([candidate]), CandidateRecord)
    assert len(valid) == 1
    assert errors.empty

    valid, errors = validate_records(pd.DataFrame(columns=candidate), CandidateRecord)
    assert valid.empty
    assert not errors.empty


def test_duplicate_rows_are_removed_and_source_rows_are_preserved():
    duplicate = pd.DataFrame([base_event(), base_event()])
    valid, errors = validate_records(duplicate, ValidationEventRecord)
    assert valid.empty
    assert errors["row_number"].tolist() == [2, 3]


def test_validation_event_schema_contains_runtime_label_invariants():
    schema = ValidationEventRecord.model_json_schema()
    assert any(
        branch.get("not", {})
        .get("properties", {})
        .get("phenotype_reproduced", {})
        .get("const")
        is True
        for branch in schema["allOf"]
    )
    assert any(
        branch.get("then", {})
        .get("properties", {})
        .get("testing_status", {})
        .get("const")
        == "tested"
        for branch in schema["allOf"]
    )


def test_registry_checks_orphan_events_and_evidence_chronology():
    studies = pd.DataFrame([{"study_id": "S1"}])
    screens = pd.DataFrame([{"screen_id": "SC1", "study_id": "S1"}])
    orphan = pd.DataFrame([{"study_id": "MISSING", "screen_id": None}])
    evidence = pd.DataFrame(
        [
            {
                "available_date": "2025-02-01",
                "retrieved_date": "2025-01-01",
            }
        ]
    )
    errors = validate_registry_integrity(
        studies=studies,
        screens=screens,
        validation_events=orphan,
        evidence=evidence,
    )
    assert errors["error"].str.contains("unknown study_id").any()
    assert errors["error"].str.contains("retrieved_date cannot precede").any()


def test_evidence_row_schema_and_runtime_share_structural_scope():
    record = {
        "evidence_id": "EV1",
        "gene_symbol": "GENE1",
        "evidence_type": "expression",
        "context_type": "cell_line",
        "context_value": "CELL",
        "value_numeric": 1.0,
        "value_text": None,
        "source_name": "SOURCE",
        "source_version": "v1",
        "source_url": "https://example.org/data",
        "source_license": "CC0",
        "available_date": "2025-02-01",
        "retrieved_date": "2025-01-01",
        "transformation_id": None,
        "notes": None,
    }
    parsed = EvidenceRecord.model_validate(record)
    assert str(parsed.retrieved_date) == "2025-01-01"
    assert "x-semantic-rules" in EvidenceRecord.model_json_schema()


def test_data_asset_manifest_matches_contract_and_validates():
    manifest = pd.read_csv(
        ROOT / "data" / "manifests" / "data_assets.tsv",
        sep="\t",
    )
    assert list(manifest.columns) == list(DataAssetRecord.model_fields)
    valid, errors = validate_records(manifest, DataAssetRecord)
    assert len(valid) == 4
    assert errors.empty
    checksummed = valid.loc[valid["sha256"].notna()]
    assert set(checksummed["checksum_provenance"]) == {
        "locally_computed_not_publisher_provided",
        "locally_computed_from_retrieved_publisher_supplement",
        "project_computed_from_verified_archive",
    }
    by_id = valid.set_index("asset_id")
    sra = by_id.loc["srp158611:public-raw-reads"]
    findlay = by_id.loc["pmid30154076:supplementary-dataset-ev3"]
    assert pd.isna(sra["sha256"])
    assert sra["checksum_provenance"] == "not_retrieved_no_local_checksum"
    assert sra["curator_status"] == "accession_verified_not_retrieved"
    assert not bool(sra["redistribution_raw"])
    assert findlay["sha256"] == (
        "0e6bf30164160dee495eb11e4bbbb7aebce9768121603c7f52ea3192265ed0b1"
    )
    assert findlay["byte_size"] == 6_574_038
    assert bool(findlay["redistribution_raw"])


def test_data_asset_rejects_incomplete_checksum_or_invalid_dates():
    record = {
        "asset_id": "A1",
        "source_name": "SOURCE",
        "source_version": "v1",
        "asset_role": "archive",
        "source_url": "https://example.org/archive.tar.gz",
        "available_date": "2026-02-01",
        "retrieved_date": "2026-01-01",
        "sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="retrieved_date cannot precede"):
        DataAssetRecord.model_validate(record)

    record["available_date"] = "2025-02-01"
    with pytest.raises(ValueError, match="sha256 and byte_size"):
        DataAssetRecord.model_validate(record)


@pytest.mark.parametrize(
    ("model", "record"),
    [
        (
            ScreenRecord,
            {
                "screen_id": "SC1",
                "study_id": "S1",
                "perturbation_modality": "CRISPR_KO",
                "screen_design": "drug_response",
                "cell_line": "CELL",
                "treatment_dose": 1.0,
            },
        ),
        (
            ContrastRecord,
            {
                "screen_id": "SC1",
                "contrast_id": "C1",
                "contrast_name": "drug_vs_control",
                "treatment_name": "DRUG",
                "control_type": "untreated",
                "phenotype_endpoint": "viability",
                "intended_direction": "resistance",
                "treatment_dose": 1.0,
            },
        ),
        (
            SampleRecord,
            {
                "sample_id": "SM1",
                "screen_id": "SC1",
                "contrast_id": "C1",
                "condition_role": "treatment",
                "replicate": 1,
                "treatment_dose": 1.0,
            },
        ),
    ],
)
def test_treatment_dose_requires_unit(model, record):
    with pytest.raises(ValueError, match="treatment_unit is required"):
        model.model_validate(record)


def test_vehicle_contrast_requires_named_comparator():
    with pytest.raises(ValueError, match="comparator_name is required"):
        ContrastRecord.model_validate(
            {
                "screen_id": "SC1",
                "contrast_id": "C1",
                "contrast_name": "drug_vs_vehicle",
                "treatment_name": "DRUG",
                "control_type": "vehicle",
                "phenotype_endpoint": "viability",
                "intended_direction": "resistance",
            }
        )


def test_sample_timepoint_requires_reference():
    with pytest.raises(ValueError, match="timepoint_reference is required"):
        SampleRecord.model_validate(
            {
                "sample_id": "SM1",
                "screen_id": "SC1",
                "contrast_id": "C1",
                "condition_role": "treatment",
                "replicate": 1,
                "timepoint_days": 14,
            }
        )


def test_registry_rejects_mismatched_sample_contrast_foreign_key():
    errors = validate_registry_integrity(
        studies=pd.DataFrame([{"study_id": "S1"}]),
        screens=pd.DataFrame(
            [
                {"screen_id": "SC1", "study_id": "S1"},
                {"screen_id": "SC2", "study_id": "S1"},
            ]
        ),
        contrasts=pd.DataFrame([{"screen_id": "SC1", "contrast_id": "C1"}]),
        samples=pd.DataFrame(
            [
                {
                    "sample_id": "SM1",
                    "screen_id": "SC2",
                    "contrast_id": "C1",
                }
            ]
        ),
    )
    assert (
        errors["error"]
        .str.contains("unknown or mismatched screen_id/contrast_id")
        .any()
    )


def _intake_row(
    intake_id,
    screen_id,
    *,
    stage="curated",
    status="metadata_only",
    candidate=True,
    benchmark_ready=False,
):
    return {
        "intake_id": intake_id,
        "screen_id": screen_id,
        "policy_version": 2,
        "assessment_stage": stage,
        "status": status,
        "candidate_for_full_curation": candidate,
        "benchmark_ready": benchmark_ready,
    }


def _eligibility_check(
    check_id,
    intake_id,
    screen_id,
    *,
    stage="curated",
    rule_id="scope.organism_human",
    outcome="pass",
    required_for_scope=True,
    required_for_benchmark=True,
):
    return {
        "check_id": check_id,
        "intake_id": intake_id,
        "screen_id": screen_id,
        "assessment_stage": stage,
        "rule_id": rule_id,
        "outcome": outcome,
        "required_for_scope": required_for_scope,
        "required_for_benchmark": required_for_benchmark,
    }


def _passing_policy_checks(intake_id, screen_id):
    return [
        _eligibility_check(
            f"{intake_id}:{rule_id}",
            intake_id,
            screen_id,
            rule_id=rule_id,
            required_for_scope=(rule_id in INTAKE_POLICY_V2_SCOPE_RULE_IDS),
            required_for_benchmark=True,
        )
        for rule_id in sorted(INTAKE_POLICY_V2_BENCHMARK_RULE_IDS)
    ]


def _ready_registry_facts(*, include_validation=True, include_rights=True):
    asset = {
        "asset_id": "A1",
        "screen_id": "SC1",
        "study_id": "S1",
        "source_family_id": "SF1",
        "raw_data_family_id": "RF1",
        "asset_role": "gene_score_table",
        "curator_status": "approved_for_benchmark",
        "redistribution_derived": False,
    }
    if include_rights:
        asset["license_terms_url"] = "https://example.org/terms"
    validation_event = base_event()
    validation_event["adjudication_status"] = "consensus_adjudicated"
    facts = {
        "studies": pd.DataFrame([{"study_id": "S1"}]),
        "screens": pd.DataFrame(
            [
                {
                    "screen_id": "SC1",
                    "study_id": "S1",
                    "source_family_id": "SF1",
                    "raw_data_family_id": "RF1",
                }
            ]
        ),
        "contrasts": pd.DataFrame(
            [
                {
                    "screen_id": "SC1",
                    "contrast_id": "C1",
                    "treatment_name": "DRUG",
                    "control_type": "vehicle",
                }
            ]
        ),
        "samples": pd.DataFrame(
            [
                {
                    "sample_id": "T1",
                    "screen_id": "SC1",
                    "contrast_id": "C1",
                    "condition_role": "treatment",
                },
                {
                    "sample_id": "C1",
                    "screen_id": "SC1",
                    "contrast_id": "C1",
                    "condition_role": "control",
                },
            ]
        ),
        "gene_scores": pd.DataFrame(
            [
                {
                    "screen_id": "SC1",
                    "contrast_id": "C1",
                    "gene_symbol": "GENE1",
                    "score": 1.0,
                    "author_hit": True,
                }
            ]
        ),
        "data_assets": pd.DataFrame([asset]),
        "validation_events": (
            pd.DataFrame([validation_event]) if include_validation else None
        ),
    }
    return facts


def test_registry_rejects_orphan_and_mismatched_eligibility_checks():
    errors = validate_registry_integrity(
        studies=pd.DataFrame([{"study_id": "S1"}]),
        screens=pd.DataFrame(
            [
                {"screen_id": "SC1", "study_id": "S1"},
                {"screen_id": "SC2", "study_id": "S1"},
            ]
        ),
        screen_intake=pd.DataFrame([_intake_row("I1", "SC1")]),
        eligibility_checks=pd.DataFrame(
            [
                _eligibility_check("EC1", "MISSING", "SC1"),
                _eligibility_check("EC2", "I1", "SC2"),
                _eligibility_check("EC3", "I1", "SC1", stage="index"),
            ]
        ),
    )
    check_errors = errors.loc[errors["table"] == "eligibility_checks", "error"]
    assert check_errors.str.contains("unknown intake_id: MISSING", regex=False).any()
    assert check_errors.str.contains(
        "screen_id does not match linked screen_intake row", regex=False
    ).any()
    assert check_errors.str.contains(
        "assessment_stage does not match linked screen_intake row", regex=False
    ).any()


def test_registry_rejects_unsupported_intake_field_combinations():
    errors = validate_registry_integrity(
        studies=pd.DataFrame([{"study_id": "S1"}]),
        screens=pd.DataFrame(
            [
                {"screen_id": "SC1", "study_id": "S1"},
                {"screen_id": "SC2", "study_id": "S1"},
                {"screen_id": "SC3", "study_id": "S1"},
            ]
        ),
        screen_intake=pd.DataFrame(
            [
                _intake_row(
                    "I1",
                    "SC1",
                    stage="index",
                    status="benchmark_ready",
                    benchmark_ready=True,
                ),
                _intake_row(
                    "I2",
                    "SC2",
                    status="exclude",
                    candidate=True,
                ),
                _intake_row(
                    "I3",
                    "SC3",
                    status="metadata_only",
                    benchmark_ready=True,
                ),
            ]
        ),
    )
    messages = errors.loc[errors["table"] == "screen_intake", "error"]
    assert messages.str.contains(
        "index-stage intake cannot establish benchmark readiness", regex=False
    ).any()
    assert messages.str.contains(
        "excluded screens cannot be full-curation candidates", regex=False
    ).any()
    assert messages.str.contains(
        "benchmark_ready must agree with benchmark_ready status", regex=False
    ).any()


def test_registry_status_requires_supporting_eligibility_outcomes():
    errors = validate_registry_integrity(
        studies=pd.DataFrame([{"study_id": "S1"}]),
        screens=pd.DataFrame(
            [
                {"screen_id": "SC1", "study_id": "S1"},
                {"screen_id": "SC2", "study_id": "S1"},
                {"screen_id": "SC3", "study_id": "S1"},
            ]
        ),
        screen_intake=pd.DataFrame(
            [
                _intake_row(
                    "I1",
                    "SC1",
                    status="benchmark_ready",
                    candidate=False,
                    benchmark_ready=True,
                ),
                _intake_row("I2", "SC2"),
                _intake_row(
                    "I3",
                    "SC3",
                    status="exclude",
                    candidate=False,
                ),
            ]
        ),
        eligibility_checks=pd.DataFrame(
            [
                _eligibility_check(
                    "EC1",
                    "I1",
                    "SC1",
                    outcome="unknown",
                    required_for_scope=False,
                ),
                _eligibility_check(
                    "EC2",
                    "I2",
                    "SC2",
                    outcome="fail",
                ),
                _eligibility_check(
                    "EC3",
                    "I3",
                    "SC3",
                    outcome="pass",
                ),
            ]
        ),
    )
    messages = errors.loc[errors["table"] == "screen_intake", "error"]
    assert messages.str.contains(
        "benchmark_ready status is unsupported because", regex=False
    ).any()
    assert messages.str.contains(
        "status metadata_only is unsupported because", regex=False
    ).any()
    assert messages.str.contains(
        "exclude status is unsupported without a failed", regex=False
    ).any()


def test_registry_rejects_all_pass_ready_assertion_without_linked_facts():
    errors = validate_registry_integrity(
        studies=pd.DataFrame([{"study_id": "S1"}]),
        screens=pd.DataFrame([{"screen_id": "SC1", "study_id": "S1"}]),
        screen_intake=pd.DataFrame(
            [
                _intake_row(
                    "I1",
                    "SC1",
                    status="benchmark_ready",
                    candidate=True,
                    benchmark_ready=True,
                )
            ]
        ),
        eligibility_checks=pd.DataFrame(_passing_policy_checks("I1", "SC1")),
    )
    message = "\n".join(errors["error"])
    assert "linked registry facts do not establish" in message
    assert "curation.comparator_and_sample_map" in message
    assert "data.count_level_signal" in message
    assert "provenance.source_family" in message
    assert "provenance.raw_data_family" in message
    assert "rights.source_and_raw_data" in message
    assert "labels.adjudicated_validation_event" in message


def test_registry_accepts_ready_status_only_with_linked_fact_evidence():
    errors = validate_registry_integrity(
        **_ready_registry_facts(),
        screen_intake=pd.DataFrame(
            [
                _intake_row(
                    "I1",
                    "SC1",
                    status="benchmark_ready",
                    candidate=True,
                    benchmark_ready=True,
                )
            ]
        ),
        eligibility_checks=pd.DataFrame(_passing_policy_checks("I1", "SC1")),
    )
    assert errors.empty


def test_registry_does_not_treat_orcs_author_hit_as_validation_evidence():
    errors = validate_registry_integrity(
        **_ready_registry_facts(include_validation=False),
        screen_intake=pd.DataFrame(
            [
                _intake_row(
                    "I1",
                    "SC1",
                    status="benchmark_ready",
                    candidate=True,
                    benchmark_ready=True,
                )
            ]
        ),
        eligibility_checks=pd.DataFrame(_passing_policy_checks("I1", "SC1")),
    )
    message = "\n".join(errors["error"])
    assert "labels.adjudicated_validation_event" in message


def test_registry_requires_reviewed_validation_event_adjudication():
    facts = _ready_registry_facts()
    facts["validation_events"].loc[0, "adjudication_status"] = "single_curator"
    errors = validate_registry_integrity(
        **facts,
        screen_intake=pd.DataFrame(
            [
                _intake_row(
                    "I1",
                    "SC1",
                    status="benchmark_ready",
                    candidate=True,
                    benchmark_ready=True,
                )
            ]
        ),
        eligibility_checks=pd.DataFrame(_passing_policy_checks("I1", "SC1")),
    )
    message = "\n".join(errors["error"])
    assert "labels.adjudicated_validation_event" in message


def test_registry_requires_signal_and_validation_on_same_contrast_and_gene():
    facts = _ready_registry_facts()
    facts["contrasts"] = pd.concat(
        [
            facts["contrasts"],
            pd.DataFrame(
                [
                    {
                        "screen_id": "SC1",
                        "contrast_id": "C2",
                        "treatment_name": "DRUG",
                        "control_type": "vehicle",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    facts["samples"] = pd.concat(
        [
            facts["samples"],
            pd.DataFrame(
                [
                    {
                        "sample_id": "T2",
                        "screen_id": "SC1",
                        "contrast_id": "C2",
                        "condition_role": "treatment",
                    },
                    {
                        "sample_id": "CTRL2",
                        "screen_id": "SC1",
                        "contrast_id": "C2",
                        "condition_role": "control",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    facts["validation_events"].loc[0, "contrast_id"] = "C2"
    errors = validate_registry_integrity(
        **facts,
        screen_intake=pd.DataFrame(
            [
                _intake_row(
                    "I1",
                    "SC1",
                    status="benchmark_ready",
                    candidate=True,
                    benchmark_ready=True,
                )
            ]
        ),
        eligibility_checks=pd.DataFrame(_passing_policy_checks("I1", "SC1")),
    )
    assert "labels.adjudicated_validation_event" in "\n".join(errors["error"])

    facts = _ready_registry_facts()
    facts["validation_events"].loc[0, "gene_symbol"] = "OTHER_GENE"
    errors = validate_registry_integrity(
        **facts,
        screen_intake=pd.DataFrame(
            [
                _intake_row(
                    "I1",
                    "SC1",
                    status="benchmark_ready",
                    candidate=True,
                    benchmark_ready=True,
                )
            ]
        ),
        eligibility_checks=pd.DataFrame(_passing_policy_checks("I1", "SC1")),
    )
    assert "labels.adjudicated_validation_event" in "\n".join(errors["error"])

    facts = _ready_registry_facts()
    facts["gene_scores"].loc[0, "direction"] = "sensitization"
    errors = validate_registry_integrity(
        **facts,
        screen_intake=pd.DataFrame(
            [
                _intake_row(
                    "I1",
                    "SC1",
                    status="benchmark_ready",
                    candidate=True,
                    benchmark_ready=True,
                )
            ]
        ),
        eligibility_checks=pd.DataFrame(_passing_policy_checks("I1", "SC1")),
    )
    assert "labels.adjudicated_validation_event" in "\n".join(errors["error"])


def test_registry_does_not_treat_author_hit_as_quantitative_screen_signal():
    facts = _ready_registry_facts()
    facts["gene_scores"] = facts["gene_scores"].drop(columns="score")
    errors = validate_registry_integrity(
        **facts,
        screen_intake=pd.DataFrame(
            [
                _intake_row(
                    "I1",
                    "SC1",
                    status="benchmark_ready",
                    candidate=True,
                    benchmark_ready=True,
                )
            ]
        ),
        eligibility_checks=pd.DataFrame(_passing_policy_checks("I1", "SC1")),
    )
    message = "\n".join(errors["error"])
    assert "data.count_level_signal" in message


def test_registry_requires_documented_rights_for_ready_signal_asset():
    errors = validate_registry_integrity(
        **_ready_registry_facts(include_rights=False),
        screen_intake=pd.DataFrame(
            [
                _intake_row(
                    "I1",
                    "SC1",
                    status="benchmark_ready",
                    candidate=True,
                    benchmark_ready=True,
                )
            ]
        ),
        eligibility_checks=pd.DataFrame(_passing_policy_checks("I1", "SC1")),
    )
    message = "\n".join(errors["error"])
    assert "rights.source_and_raw_data" in message


def test_registry_rejects_unapproved_status_substring_for_ready_asset():
    facts = _ready_registry_facts()
    facts["data_assets"].loc[0, "curator_status"] = "unapproved"
    errors = validate_registry_integrity(
        **facts,
        screen_intake=pd.DataFrame(
            [
                _intake_row(
                    "I1",
                    "SC1",
                    status="benchmark_ready",
                    candidate=True,
                    benchmark_ready=True,
                )
            ]
        ),
        eligibility_checks=pd.DataFrame(_passing_policy_checks("I1", "SC1")),
    )
    message = "\n".join(errors["error"])
    assert "data.count_level_signal" in message
    assert "rights.source_and_raw_data" in message


def test_validation_event_study_can_be_an_independent_validation_source():
    event = base_event()
    event["study_id"] = "S2"
    errors = validate_registry_integrity(
        studies=pd.DataFrame([{"study_id": "S1"}, {"study_id": "S2"}]),
        screens=pd.DataFrame([{"screen_id": "SC1", "study_id": "S1"}]),
        contrasts=pd.DataFrame([{"screen_id": "SC1", "contrast_id": "C1"}]),
        validation_events=pd.DataFrame([event]),
    )
    assert errors.empty


def test_registry_rejects_ready_status_with_one_required_rule_missing():
    checks = [
        check
        for check in _passing_policy_checks("I1", "SC1")
        if check["rule_id"] != "provenance.raw_data_family"
    ]
    errors = validate_registry_integrity(
        studies=pd.DataFrame([{"study_id": "S1"}]),
        screens=pd.DataFrame([{"screen_id": "SC1", "study_id": "S1"}]),
        screen_intake=pd.DataFrame(
            [
                _intake_row(
                    "I1",
                    "SC1",
                    status="benchmark_ready",
                    candidate=False,
                    benchmark_ready=True,
                )
            ]
        ),
        eligibility_checks=pd.DataFrame(checks),
    )
    assert (
        errors["error"]
        .str.contains("required policy rules are missing", regex=False)
        .any()
    )
