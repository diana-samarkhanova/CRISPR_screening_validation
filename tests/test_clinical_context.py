from __future__ import annotations

import hashlib
import json
from datetime import date

import pandas as pd
import pytest

import crispr_evidencerank.cli as cli_module
from crispr_evidencerank.cli import build_parser
from crispr_evidencerank.clinical_context import summarize_clinical_context
from crispr_evidencerank.modeling import validate_success_feature_columns

ASSET_SHA = "a" * 64


def _asset(asset_id: str = "ctg-export", **updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "asset_id": asset_id,
        "source_name": "ClinicalTrials.gov",
        "source_version": "snapshot-2026-08-01",
        "asset_role": "clinical_trials_query_export",
        "source_url": "https://clinicaltrials.gov/api/v2/studies",
        "available_date": "2026-08-01",
        "retrieved_date": "2026-08-02",
        "sha256": ASSET_SHA,
        "byte_size": 100,
    }
    record.update(updates)
    return record


def _record(evidence_id: str = "EV1", **updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "evidence_id": evidence_id,
        "source_asset_id": "ctg-export",
        "source_asset_sha256": ASSET_SHA,
        "source_name": "ClinicalTrials.gov",
        "source_version": "snapshot-2026-08-01",
        "source_api_version": "2.0.5",
        "source_snapshot_date": "2026-08-01",
        "source_study_id": "NCT00000001",
        "source_family_id": "TRIAL:NCT00000001",
        "source_record_version": "2026-07-20",
        "record_last_update_date": "2026-07-20",
        "study_type": "interventional",
        "source_overall_status": "COMPLETED",
        "status_category": "completed",
        "source_phase": "PHASE2",
        "phase_category": "phase_2",
        "source_treatment_text": "Olaparib",
        "treatment_concept_id": "NCIT:C71721",
        "treatment_preferred_name": "Olaparib",
        "treatment_mapping_source": "NCIt",
        "treatment_mapping_version": "26.07d",
        "treatment_mapping_relation": "exact",
        "treatment_mapping_review_status": "curator_reviewed",
        "treatment_mapping_review_id": "MAP-REVIEW-TREATMENT-1",
        "treatment_mapping_review_date": "2026-08-01",
        "source_condition_text": "Triple Negative Breast Cancer",
        "cancer_concept_id": "NCIT:C71732",
        "cancer_preferred_name": "Triple-Negative Breast Carcinoma",
        "cancer_mapping_source": "NCIt",
        "cancer_mapping_version": "26.07d",
        "cancer_mapping_relation": "exact",
        "cancer_mapping_review_status": "curator_reviewed",
        "cancer_mapping_review_id": "MAP-REVIEW-CANCER-1",
        "cancer_mapping_review_date": "2026-08-01",
        "intervention_role": "experimental",
        "regimen_context": "monotherapy",
        "source_subtype_definition": "ER <1%, PR <1%, HER2-negative",
        "results_posted": False,
        "results_first_posted_date": None,
        "source_url": "https://clinicaltrials.gov/study/NCT00000001",
        "source_locator": "protocolSection",
        "available_date": "2020-01-01",
        "transformation_available_date": "2026-08-01",
        "retrieved_date": "2026-08-02",
        "used_for_label": False,
        "notes": None,
    }
    record.update(updates)
    return record


def _summarize(
    evidence: list[dict[str, object]],
    assets: list[dict[str, object]] | None = None,
    **kwargs: object,
):
    return summarize_clinical_context(
        pd.DataFrame(evidence),
        pd.DataFrame(assets or [_asset()]),
        treatment_concept_id=str(kwargs.pop("treatment_concept_id", "NCIT:C71721")),
        treatment_mapping_source=str(kwargs.pop("treatment_mapping_source", "NCIt")),
        treatment_mapping_version=str(
            kwargs.pop("treatment_mapping_version", "26.07d")
        ),
        cancer_concept_id=str(kwargs.pop("cancer_concept_id", "NCIT:C71732")),
        cancer_mapping_source=str(kwargs.pop("cancer_mapping_source", "NCIt")),
        cancer_mapping_version=str(kwargs.pop("cancer_mapping_version", "26.07d")),
        cutoff_date=kwargs.pop("cutoff_date", date(2026, 8, 1)),
        **kwargs,
    )


def test_exact_query_is_report_only_and_checksum_bound() -> None:
    result = _summarize([_record()])

    row = result.summary.iloc[0]
    assert bool(row["report_only_clinical_context_available"])
    assert row["report_only_clinical_observed_record_n"] == 1
    assert row["report_only_clinical_observed_source_family_n"] == 1
    assert row["report_only_clinical_status_completed_family_n"] == 1
    assert row["report_only_clinical_phase_category_phase_2_family_n"] == 1
    assert row["report_only_clinical_regimen_monotherapy_family_n"] == 1
    assert len(result.studies) == 1
    assert result.exclusions.empty
    assert result.used_assets["asset_id"].tolist() == ["ctg-export"]
    assert result.used_assets["sha256"].tolist() == [ASSET_SHA]
    assert "used_for_label" not in result.studies
    non_query_columns = set(result.summary) - {
        "treatment_concept_id",
        "treatment_mapping_source",
        "treatment_mapping_version",
        "cancer_concept_id",
        "cancer_mapping_source",
        "cancer_mapping_version",
    }
    assert all(
        column.startswith("report_only_clinical_") for column in non_query_columns
    )
    assert result.metadata["matching_policy"] == (
        "exact_treatment_and_cancer_concept_ids_and_mapping_releases"
    )
    assert "not efficacy" in result.metadata["interpretation_boundary"]


@pytest.mark.parametrize(
    ("evidence_updates", "asset_updates", "message"),
    [
        ({"source_asset_id": "missing"}, {}, "unknown source assets"),
        (
            {"source_asset_sha256": "b" * 64},
            {},
            "source_asset_sha256 does not match",
        ),
        ({}, {"sha256": None, "byte_size": None}, "require a checksum"),
        ({}, {"source_name": "Other registry"}, "source_name does not match"),
        ({}, {"source_version": "other-version"}, "source_version does not match"),
    ],
)
def test_source_asset_binding_fails_closed(
    evidence_updates: dict[str, object],
    asset_updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _summarize(
            [_record(**evidence_updates)],
            [_asset(**asset_updates)],
        )


def test_entire_asset_registry_is_validated_not_only_referenced_rows() -> None:
    invalid_unreferenced = _asset(
        "unreferenced",
        sha256="b" * 64,
        byte_size=None,
    )
    with pytest.raises(ValueError, match="data asset validation failed"):
        _summarize([_record()], [_asset(), invalid_unreferenced])


def test_cutoff_is_inclusive_and_post_cutoff_snapshot_is_excluded() -> None:
    boundary = _record(
        source_snapshot_date="2026-08-01",
        transformation_available_date="2026-08-01",
    )
    boundary_result = _summarize([boundary], cutoff_date=date(2026, 8, 1))
    assert len(boundary_result.studies) == 1

    later = _record(
        source_snapshot_date="2026-08-02",
        transformation_available_date="2026-08-02",
        retrieved_date="2026-08-03",
    )
    later_result = _summarize([later], cutoff_date=date(2026, 8, 1))
    assert later_result.studies.empty
    assert later_result.exclusions["exclusion_reason"].tolist() == ["post_cutoff"]


def test_source_asset_availability_participates_in_cutoff() -> None:
    result = _summarize(
        [
            _record(
                source_snapshot_date="2026-08-01",
                transformation_available_date="2026-08-01",
                retrieved_date="2026-08-03",
            )
        ],
        [
            _asset(
                available_date="2026-08-02",
                retrieved_date="2026-08-03",
            )
        ],
        cutoff_date=date(2026, 8, 1),
    )
    assert result.exclusions["exclusion_reason"].tolist() == ["post_cutoff"]


def test_exact_concept_ids_not_names_control_matching() -> None:
    other_treatment = _record(
        "EV-T",
        source_study_id="NCT00000002",
        source_family_id="TRIAL:NCT00000002",
        source_treatment_text="Olaparib",
        treatment_preferred_name="Olaparib",
        treatment_concept_id="NCIT:OTHER",
    )
    other_cancer = _record(
        "EV-C",
        source_study_id="NCT00000003",
        source_family_id="TRIAL:NCT00000003",
        source_condition_text="Triple Negative Breast Cancer",
        cancer_preferred_name="Triple-Negative Breast Carcinoma",
        cancer_concept_id="NCIT:OTHER-CANCER",
    )
    observational = _record(
        "EV-O",
        source_study_id="NCT00000004",
        source_family_id="TRIAL:NCT00000004",
        study_type="observational",
        source_phase="N/A",
        phase_category="not_applicable",
    )
    result = _summarize([other_treatment, other_cancer, observational])

    reasons = result.exclusions.set_index("evidence_id")["exclusion_reason"]
    assert reasons.to_dict() == {
        "EV-C": "cancer_mismatch",
        "EV-O": "study_type_mismatch",
        "EV-T": "treatment_mismatch",
    }
    assert not bool(result.summary.iloc[0]["report_only_clinical_context_available"])
    assert (
        result.summary.iloc[0]["report_only_clinical_absence_interpretation"]
        == "not_observed_in_supplied_snapshot_not_proof_of_absence"
    )


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        (
            {"treatment_mapping_relation": "related"},
            "treatment_mapping_not_exact",
        ),
        (
            {
                "treatment_mapping_review_status": "automated_unreviewed",
                "treatment_mapping_review_id": None,
                "treatment_mapping_review_date": None,
            },
            "treatment_mapping_not_reviewed",
        ),
        (
            {"cancer_mapping_relation": "broader_than_source"},
            "cancer_mapping_not_exact",
        ),
        (
            {
                "cancer_mapping_review_status": "source_asserted",
                "cancer_mapping_review_id": None,
                "cancer_mapping_review_date": None,
            },
            "cancer_mapping_not_reviewed",
        ),
        (
            {"intervention_role": "active_comparator"},
            "intervention_role_mismatch",
        ),
    ],
)
def test_unreviewed_or_indirect_mappings_and_nonexperimental_roles_are_excluded(
    updates: dict[str, object],
    reason: str,
) -> None:
    result = _summarize([_record(**updates)])
    assert result.studies.empty
    assert result.exclusions["exclusion_reason"].tolist() == [reason]


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        (
            {"treatment_mapping_source": "OtherOntology"},
            "treatment_mapping_source_mismatch",
        ),
        (
            {"treatment_mapping_version": "99.future"},
            "treatment_mapping_version_mismatch",
        ),
        (
            {"cancer_mapping_source": "OtherOntology"},
            "cancer_mapping_source_mismatch",
        ),
        (
            {"cancer_mapping_version": "99.future"},
            "cancer_mapping_version_mismatch",
        ),
    ],
)
def test_query_pins_mapping_sources_and_versions(
    updates: dict[str, object],
    reason: str,
) -> None:
    result = _summarize([_record(**updates)])
    assert result.studies.empty
    assert result.exclusions["exclusion_reason"].tolist() == [reason]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"treatment_mapping_review_id": None},
            "curator-reviewed treatment mapping requires review ID and date",
        ),
        (
            {"cancer_mapping_review_date": None},
            "curator-reviewed cancer mapping requires review ID and date",
        ),
        (
            {"treatment_mapping_review_date": "2026-08-02"},
            "treatment mapping review date cannot follow transformation",
        ),
        (
            {"cancer_mapping_review_status": "source_asserted"},
            "cancer mapping review ID/date require curator_reviewed status",
        ),
    ],
)
def test_mapping_review_attestation_is_contract_enforced(
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _summarize([_record(**updates)])


def test_source_family_exclusion_is_explicit_and_unknown_family_is_rejected() -> None:
    result = _summarize(
        [_record()],
        excluded_source_families=["TRIAL:NCT00000001"],
    )
    assert result.exclusions["exclusion_reason"].tolist() == ["excluded_source_family"]

    with pytest.raises(ValueError, match="absent from clinical evidence"):
        _summarize([_record()], excluded_source_families=["UNKNOWN"])


def test_duplicate_normalized_trial_query_is_rejected() -> None:
    duplicate = _record("EV2")
    with pytest.raises(ValueError, match="duplicate normalized"):
        _summarize([_record(), duplicate])


def test_source_study_cannot_claim_two_source_families() -> None:
    second = _record(
        "EV2",
        cancer_concept_id="NCIT:SECOND-CANCER",
        cancer_preferred_name="Second cancer",
        source_family_id="TRIAL:OTHER",
    )
    with pytest.raises(ValueError, match="multiple source families"):
        _summarize([_record(), second])


def test_cross_source_family_conflicts_abstain_instead_of_double_counting() -> None:
    second_sha = "b" * 64
    second = _record(
        "EV2",
        source_asset_id="other-export",
        source_asset_sha256=second_sha,
        source_name="Other registry",
        source_version="snapshot-2026-08-01",
        source_study_id="OTHER-1",
        source_family_id="TRIAL:NCT00000001",
        source_overall_status="RECRUITING",
        status_category="recruiting",
    )
    other_asset = _asset(
        "other-export",
        source_name="Other registry",
        sha256=second_sha,
    )
    with pytest.raises(ValueError, match="conflicting summary fields"):
        _summarize([_record(), second], [_asset(), other_asset])


def test_results_availability_is_counted_without_becoming_efficacy() -> None:
    posted = _record(
        "EV2",
        source_study_id="NCT00000002",
        source_family_id="TRIAL:NCT00000002",
        results_posted=True,
        results_first_posted_date="2026-07-25",
    )
    result = _summarize([_record(), posted])
    row = result.summary.iloc[0]
    assert row["report_only_clinical_results_posted_family_n"] == 1
    assert row["report_only_clinical_results_not_posted_family_n"] == 1
    forbidden = ("effect", "response", "survival", "benefit", "validation", "score")
    assert not any(
        token in column.lower()
        for column in result.summary.columns
        for token in forbidden
    )
    assert row["report_only_clinical_interpretation"] == (
        "registry_presence_and_results_availability_only_not_efficacy"
    )


def test_output_order_is_deterministic() -> None:
    second = _record(
        "EV2",
        source_study_id="NCT00000002",
        source_family_id="TRIAL:NCT00000002",
    )
    first_order = _summarize([second, _record()])
    second_order = _summarize([_record(), second])
    pd.testing.assert_frame_equal(first_order.summary, second_order.summary)
    pd.testing.assert_frame_equal(first_order.studies, second_order.studies)
    pd.testing.assert_frame_equal(first_order.used_assets, second_order.used_assets)


def test_report_frames_contain_no_validation_or_label_fields() -> None:
    result = _summarize([_record()])
    forbidden = ("label", "validation", "testing", "benchmark", "outcome")
    for frame in (
        result.summary,
        result.studies,
        result.exclusions,
        result.used_assets,
    ):
        assert not any(
            token in column.lower() for column in frame.columns for token in forbidden
        )


def test_report_only_clinical_columns_are_blocked_from_success_model() -> None:
    with pytest.raises(ValueError, match="leakage fields"):
        validate_success_feature_columns(
            ["guide_n", "report_only_clinical_phase_category_phase_3_family_n"]
        )


def _clinical_cli_args(
    evidence_path,
    assets_path,
    output_dir,
):
    return build_parser().parse_args(
        [
            "summarize-clinical-context",
            "--evidence",
            str(evidence_path),
            "--assets",
            str(assets_path),
            "--treatment-concept-id",
            "NCIT:C71721",
            "--treatment-mapping-source",
            "NCIt",
            "--treatment-mapping-version",
            "26.07d",
            "--cancer-concept-id",
            "NCIT:C71732",
            "--cancer-mapping-source",
            "NCIt",
            "--cancer-mapping-version",
            "26.07d",
            "--cutoff-date",
            "2026-08-01",
            "--output-dir",
            str(output_dir),
        ]
    )


def test_cli_exposes_clinical_context_command() -> None:
    args = _clinical_cli_args("evidence.tsv", "assets.tsv", "out")
    assert args.command == "summarize-clinical-context"
    assert args.cutoff_date == date(2026, 8, 1)


def test_cli_writes_checksum_bound_atomic_bundle(tmp_path) -> None:
    evidence_path = tmp_path / "evidence.tsv"
    assets_path = tmp_path / "assets.tsv"
    output_dir = tmp_path / "out"
    pd.DataFrame([_record(source_version="1")]).to_csv(
        evidence_path,
        sep="\t",
        index=False,
    )
    pd.DataFrame([_asset(source_version="1")]).to_csv(
        assets_path,
        sep="\t",
        index=False,
    )

    args = _clinical_cli_args(evidence_path, assets_path, output_dir)
    assert args.func(args) == 0

    expected_files = {
        "clinical_context.tsv",
        "clinical_context_studies.tsv",
        "clinical_context_exclusions.tsv",
        "clinical_context_used_assets.tsv",
        "summary.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_files
    summary = pd.read_csv(output_dir / "clinical_context.tsv", sep="\t", dtype=str)
    assert summary["treatment_concept_id"].tolist() == ["NCIT:C71721"]
    assert summary["cancer_concept_id"].tolist() == ["NCIT:C71732"]

    manifest = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert manifest["bundle_type"] == "clinical_context"
    assert manifest["bundle_schema_version"] == 1
    assert set(manifest["outputs"]) == {
        "summary",
        "studies",
        "exclusions",
        "used_assets",
    }
    assert len({output["filename"] for output in manifest["outputs"].values()}) == 4
    for output in manifest["outputs"].values():
        output_path = output_dir / output["filename"]
        assert hashlib.sha256(output_path.read_bytes()).hexdigest() == output["sha256"]
        assert len(output["sha256"]) == 64
        assert output["sha256"] == output["sha256"].lower()
    assert len(manifest["inputs"]["evidence"]["sha256"]) == 64
    assert len(manifest["inputs"]["assets"]["sha256"]) == 64
    assert len(manifest["software"]["clinical_schema_sha256"]) == 64


def test_existing_clinical_bundle_is_preserved(tmp_path) -> None:
    evidence_path = tmp_path / "evidence.tsv"
    assets_path = tmp_path / "assets.tsv"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    pd.DataFrame([_record()]).to_csv(evidence_path, sep="\t", index=False)
    pd.DataFrame([_asset()]).to_csv(assets_path, sep="\t", index=False)

    args = _clinical_cli_args(evidence_path, assets_path, output_dir)
    with pytest.raises(FileExistsError, match="output directory exists"):
        args.func(args)
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("input_role", ["evidence", "assets"])
def test_clinical_input_mutation_after_compute_aborts_before_bundle_publish(
    tmp_path,
    monkeypatch,
    input_role: str,
) -> None:
    evidence_path = tmp_path / "evidence.tsv"
    assets_path = tmp_path / "assets.tsv"
    output_dir = tmp_path / "out"
    pd.DataFrame([_record()]).to_csv(evidence_path, sep="\t", index=False)
    pd.DataFrame([_asset()]).to_csv(assets_path, sep="\t", index=False)
    original_summarize = cli_module.summarize_clinical_context
    mutated_path = evidence_path if input_role == "evidence" else assets_path

    def mutate_after_compute(*args, **kwargs):
        result = original_summarize(*args, **kwargs)
        mutated_path.write_bytes(mutated_path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(
        cli_module,
        "summarize_clinical_context",
        mutate_after_compute,
    )
    args = _clinical_cli_args(evidence_path, assets_path, output_dir)
    with pytest.raises(ValueError, match="changed during the run"):
        args.func(args)
    assert not output_dir.exists()


@pytest.mark.parametrize("input_role", ["evidence", "assets"])
def test_clinical_input_mutation_during_staged_write_aborts_publish(
    tmp_path,
    monkeypatch,
    input_role: str,
) -> None:
    evidence_path = tmp_path / "evidence.tsv"
    assets_path = tmp_path / "assets.tsv"
    output_dir = tmp_path / "out"
    pd.DataFrame([_record()]).to_csv(evidence_path, sep="\t", index=False)
    pd.DataFrame([_asset()]).to_csv(assets_path, sep="\t", index=False)
    mutated_path = evidence_path if input_role == "evidence" else assets_path
    original_write_frame = cli_module._write_frame
    writes = 0

    def mutate_during_write(*args, **kwargs):
        nonlocal writes
        original_write_frame(*args, **kwargs)
        writes += 1
        if writes == 1:
            mutated_path.write_bytes(mutated_path.read_bytes() + b"\n")

    monkeypatch.setattr(cli_module, "_write_frame", mutate_during_write)
    args = _clinical_cli_args(evidence_path, assets_path, output_dir)
    with pytest.raises(ValueError, match="changed during the run"):
        args.func(args)
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".out.staging-*"))


def test_clinical_staged_write_failure_leaves_no_partial_bundle(
    tmp_path,
    monkeypatch,
) -> None:
    evidence_path = tmp_path / "evidence.tsv"
    assets_path = tmp_path / "assets.tsv"
    output_dir = tmp_path / "out"
    pd.DataFrame([_record()]).to_csv(evidence_path, sep="\t", index=False)
    pd.DataFrame([_asset()]).to_csv(assets_path, sep="\t", index=False)
    original_write_frame = cli_module._write_frame
    writes = 0

    def fail_during_write(*args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 3:
            raise OSError("synthetic staged write failure")
        original_write_frame(*args, **kwargs)

    monkeypatch.setattr(cli_module, "_write_frame", fail_during_write)
    args = _clinical_cli_args(evidence_path, assets_path, output_dir)
    with pytest.raises(OSError, match="synthetic staged write failure"):
        args.func(args)
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".out.staging-*"))


def test_clinical_output_aliasing_an_input_is_rejected(tmp_path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    evidence_path = output_dir / "clinical_context.tsv"
    assets_path = tmp_path / "assets.tsv"
    original_bytes = pd.DataFrame([_record()]).to_csv(sep="\t", index=False).encode()
    evidence_path.write_bytes(original_bytes)
    pd.DataFrame([_asset()]).to_csv(assets_path, sep="\t", index=False)

    args = _clinical_cli_args(evidence_path, assets_path, output_dir)
    with pytest.raises(ValueError, match="cannot overwrite an input file"):
        args.func(args)
    assert evidence_path.read_bytes() == original_bytes
