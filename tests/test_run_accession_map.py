from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from crispr_evidencerank.contracts import RunAccessionMapRecord
from crispr_evidencerank.curation import build_run_accession_map_manifest

ROOT = Path(__file__).resolve().parents[1]
RUN_MAP = (
    ROOT
    / "data"
    / "manifests"
    / "orcs_2.0.18"
    / "curation_batches"
    / "batch_001"
    / "srp158611_run_map.tsv"
)
RUN_MANIFEST = RUN_MAP.with_name("srp158611_run_map_manifest.json")
RUN_INVENTORY = RUN_MAP.with_name("srp158611_run_inventory.tsv")
RUN_SCOPE = RUN_MAP.with_name("srp158611_contrast_scope.tsv")


def _base_run() -> dict[str, object]:
    return {
        "map_id": "SRP158611:SRR7741079",
        "bioproject_accession": "PRJNA487352",
        "study_accession": "SRP158611",
        "run_accession": "SRR7741079",
        "experiment_accession": "SRX4597426",
        "sample_accession": "SAMN09881960",
        "secondary_sample_accession": "SRS3703510",
        "source_sample_id": "S21",
        "repository_sample_alias": "CGS_GW_Stim1X_D5_Div",
        "repository_screen_group": "CGS-21680",
        "library_strategy": "AMPLICON",
        "library_source": "GENOMIC",
        "library_selection": "PCR",
        "library_layout": "SINGLE",
        "instrument_model": "Illumina HiSeq 4000",
        "inclusion_status": "included_drug_contrast",
        "screen_id": "orcs:2.0.18:screen:1110",
        "contrast_id": "orcs:2.0.18:screen:1110:contrast:cgs-21680",
        "condition_role": "treatment",
        "treatment_name": "CGS-21680",
        "donor_id": "D5",
        "phenotype_bin": "dividing",
        "treatment_mapping_evidence": "repository_explicit",
        "repository_metadata_url": "https://www.ebi.ac.uk/ena/browser/view/SRR7741079",
        "article_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6689405/",
        "source_locator": "SRA alias and Figure 6/Table S6",
        "retrieved_date": "2026-08-01",
        "notes": None,
    }


def _inventory() -> pd.DataFrame:
    return pd.read_csv(RUN_INVENTORY, sep="\t", dtype=str)


@pytest.mark.parametrize(
    ("updates", "expected_valid"),
    [
        ({}, True),
        ({"screen_id": None}, False),
        ({"condition_role": "baseline"}, False),
        ({"treatment_mapping_evidence": "not_applicable"}, False),
        (
            {
                "inclusion_status": "excluded_other_screen",
                "screen_id": None,
                "contrast_id": None,
                "condition_role": None,
                "treatment_name": None,
                "treatment_mapping_evidence": "not_applicable",
            },
            True,
        ),
        (
            {
                "inclusion_status": "excluded_other_screen",
                "treatment_mapping_evidence": "not_applicable",
            },
            False,
        ),
    ],
)
def test_run_accession_map_schema_matches_runtime(
    updates: dict[str, object],
    expected_valid: bool,
):
    record = _base_run() | updates
    schema_valid = Draft202012Validator(
        RunAccessionMapRecord.model_json_schema()
    ).is_valid(record)
    try:
        RunAccessionMapRecord.model_validate(record)
    except ValidationError:
        runtime_valid = False
    else:
        runtime_valid = True

    assert schema_valid is expected_valid
    assert runtime_valid is expected_valid


@pytest.mark.parametrize(
    "field_name",
    ["screen_id", "contrast_id", "condition_role", "treatment_name"],
)
def test_included_run_schema_and_runtime_reject_omitted_linkage(field_name):
    record = _base_run()
    del record[field_name]
    assert not Draft202012Validator(RunAccessionMapRecord.model_json_schema()).is_valid(
        record
    )
    with pytest.raises(ValidationError):
        RunAccessionMapRecord.model_validate(record)


def test_vehicle_mapping_can_be_article_supported_without_overclaiming_repository():
    record = _base_run() | {
        "run_accession": "SRR7741090",
        "experiment_accession": "SRX4597415",
        "sample_accession": "SAMN09881956",
        "secondary_sample_accession": "SRS3703499",
        "source_sample_id": "S17",
        "repository_sample_alias": "SecondaryGW_Stim1X_D5_Div",
        "repository_screen_group": "SecondaryGW",
        "condition_role": "control",
        "treatment_name": "vehicle",
        "treatment_mapping_evidence": "article_supported",
    }

    parsed = RunAccessionMapRecord.model_validate(record)
    assert parsed.treatment_mapping_evidence == "article_supported"
    assert parsed.condition_role == "control"


def test_checked_in_srp158611_map_is_complete_but_not_benchmark_ready(tmp_path):
    manifest = build_run_accession_map_manifest(RUN_MAP, RUN_INVENTORY, RUN_SCOPE)
    observed_manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))

    assert observed_manifest == manifest
    assert manifest["record_count"] == 24
    assert manifest["included_run_count"] == 8
    assert manifest["excluded_other_screen_run_count"] == 16
    assert manifest["included_donor_count"] == 2
    assert manifest["mapping_evidence_counts"] == {
        "article_supported": 4,
        "repository_explicit": 4,
    }
    assert manifest["benchmark_ready_count"] == 0
    assert manifest["raw_reads_ingested"] is False
    assert manifest["source_inventory"]["record_count"] == 24
    assert manifest["source_inventory"]["repository_screen_group_counts"] == {
        "CGS-21680": 4,
        "Pilot CellSurf": 4,
        "PrimaryGW": 12,
        "SecondaryGW": 4,
    }
    assert manifest["contrast_scope"]["included_repository_screen_groups"] == [
        "CGS-21680",
        "SecondaryGW",
    ]

    tampered = pd.read_csv(RUN_MAP, sep="\t", dtype=str)
    tampered = tampered.loc[~tampered["run_accession"].eq("SRR7741082")]
    tampered_path = tmp_path / "tampered.tsv"
    tampered.to_csv(tampered_path, sep="\t", index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="exactly match the pinned source inventory"):
        build_run_accession_map_manifest(tampered_path, RUN_INVENTORY, RUN_SCOPE)


@pytest.mark.parametrize(
    "subset",
    ["drop_excluded", "drop_d6", "d5_included_only"],
)
def test_run_map_rejects_false_complete_inventory_subsets(tmp_path, subset):
    run_map = pd.read_csv(RUN_MAP, sep="\t", dtype=str)
    if subset == "drop_excluded":
        run_map = run_map.loc[run_map["inclusion_status"].eq("included_drug_contrast")]
    elif subset == "drop_d6":
        run_map = run_map.loc[
            run_map["inclusion_status"].eq("excluded_other_screen")
            | run_map["donor_id"].eq("D5")
        ]
    else:
        run_map = run_map.loc[
            run_map["inclusion_status"].eq("included_drug_contrast")
            & run_map["donor_id"].eq("D5")
        ]
    subset_path = tmp_path / f"{subset}.tsv"
    run_map.to_csv(subset_path, sep="\t", index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="exactly match the pinned source inventory"):
        build_run_accession_map_manifest(subset_path, RUN_INVENTORY, RUN_SCOPE)


def test_run_map_rejects_curated_scope_misclassification(tmp_path):
    run_map = pd.read_csv(RUN_MAP, sep="\t", dtype=str)
    d6_included = run_map["inclusion_status"].eq("included_drug_contrast") & run_map[
        "donor_id"
    ].eq("D6")
    run_map.loc[d6_included, "inclusion_status"] = "excluded_other_screen"
    run_map.loc[
        d6_included,
        ["screen_id", "contrast_id", "condition_role", "treatment_name"],
    ] = None
    run_map.loc[d6_included, "treatment_mapping_evidence"] = "not_applicable"
    path = tmp_path / "misclassified.tsv"
    run_map.to_csv(path, sep="\t", index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="curated contrast scope"):
        build_run_accession_map_manifest(path, RUN_INVENTORY, RUN_SCOPE)


def test_run_map_rejects_mixed_accession_families(tmp_path):
    mixed = pd.read_csv(RUN_MAP, sep="\t", dtype=str)
    mixed.loc[0, "study_accession"] = "SRP000001"
    mixed_path = tmp_path / "mixed.tsv"
    mixed.to_csv(mixed_path, sep="\t", index=False, lineterminator="\n")

    with pytest.raises(ValueError, match="one study and BioProject"):
        build_run_accession_map_manifest(mixed_path, RUN_INVENTORY, RUN_SCOPE)


@pytest.mark.parametrize(
    ("field_name", "tampered_value"),
    [
        ("repository_sample_alias", "CGS-21680, 1X, D5, Div"),
        ("library_layout", "PAIRED"),
    ],
)
def test_run_map_rejects_tampered_repository_metadata(
    tmp_path, field_name, tampered_value
):
    tampered = pd.read_csv(RUN_MAP, sep="\t", dtype=str)
    tampered.loc[0, field_name] = tampered_value
    tampered_path = tmp_path / f"tampered-{field_name}.tsv"
    tampered.to_csv(tampered_path, sep="\t", index=False, lineterminator="\n")

    with pytest.raises(ValueError, match="differs from the pinned source inventory"):
        build_run_accession_map_manifest(tampered_path, RUN_INVENTORY, RUN_SCOPE)


def test_run_map_rejects_duplicate_design_cell(tmp_path):
    duplicated = pd.read_csv(RUN_MAP, sep="\t", dtype=str)
    extra = duplicated.loc[duplicated["run_accession"].eq("SRR7741079")].copy()
    extra["map_id"] = "SRP158611:SRR9999999"
    extra["run_accession"] = "SRR9999999"
    extra["experiment_accession"] = "SRX9999999"
    extra["sample_accession"] = "SAMN99999999"
    extra["secondary_sample_accession"] = "SRS9999999"
    duplicated = pd.concat([duplicated, extra], ignore_index=True)
    duplicated_path = tmp_path / "duplicated.tsv"
    duplicated.to_csv(
        duplicated_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    inventory = _inventory()
    inventory_extra = inventory.loc[inventory["run_accession"].eq("SRR7741079")].copy()
    inventory_extra["run_accession"] = "SRR9999999"
    inventory_extra["experiment_accession"] = "SRX9999999"
    inventory_extra["sample_accession"] = "SAMN99999999"
    inventory_extra["secondary_sample_accession"] = "SRS9999999"
    inventory = pd.concat([inventory, inventory_extra], ignore_index=True)
    inventory_path = tmp_path / "duplicated-inventory.tsv"
    inventory.to_csv(inventory_path, sep="\t", index=False, lineterminator="\n")

    with pytest.raises(ValueError, match="complete condition-by-bin design"):
        build_run_accession_map_manifest(duplicated_path, inventory_path, RUN_SCOPE)


def test_run_map_rejects_duplicate_map_id(tmp_path):
    duplicated = pd.read_csv(RUN_MAP, sep="\t", dtype=str)
    duplicated.loc[1, "map_id"] = duplicated.loc[0, "map_id"]
    duplicated_path = tmp_path / "duplicate-map-id.tsv"
    duplicated.to_csv(duplicated_path, sep="\t", index=False, lineterminator="\n")

    with pytest.raises(ValueError, match="map_id values must be unique"):
        build_run_accession_map_manifest(duplicated_path, RUN_INVENTORY, RUN_SCOPE)


def test_run_map_rejects_inconsistent_role_names_and_sample_identity(tmp_path):
    inconsistent_names = pd.read_csv(RUN_MAP, sep="\t", dtype=str)
    treatment_rows = inconsistent_names["condition_role"].eq("treatment")
    inconsistent_names.loc[treatment_rows.idxmax(), "treatment_name"] = "OTHER"
    names_path = tmp_path / "inconsistent-names.tsv"
    inconsistent_names.to_csv(names_path, sep="\t", index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="curated contrast scope"):
        build_run_accession_map_manifest(names_path, RUN_INVENTORY, RUN_SCOPE)

    inconsistent_samples = pd.read_csv(RUN_MAP, sep="\t", dtype=str)
    included = inconsistent_samples["inclusion_status"].eq("included_drug_contrast")
    for identifier in (
        "sample_accession",
        "secondary_sample_accession",
        "source_sample_id",
    ):
        inconsistent_samples.loc[included, identifier] = inconsistent_samples.loc[
            included, identifier
        ].iloc[0]
    samples_path = tmp_path / "inconsistent-samples.tsv"
    inconsistent_samples.to_csv(
        samples_path, sep="\t", index=False, lineterminator="\n"
    )
    inventory = _inventory()
    included_runs = set(inconsistent_samples.loc[included, "run_accession"].astype(str))
    for identifier in (
        "sample_accession",
        "secondary_sample_accession",
        "source_sample_id",
    ):
        inventory.loc[inventory["run_accession"].isin(included_runs), identifier] = (
            inconsistent_samples.loc[included, identifier].iloc[0]
        )
    inventory_path = tmp_path / "inconsistent-samples-inventory.tsv"
    inventory.to_csv(inventory_path, sep="\t", index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="contradictory design metadata"):
        build_run_accession_map_manifest(samples_path, inventory_path, RUN_SCOPE)
