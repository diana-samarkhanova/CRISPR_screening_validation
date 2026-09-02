from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

import crispr_evidencerank.clinicaltrials_gov as ctgov_module
from crispr_evidencerank.clinicaltrials_gov import (
    STUDY_FIELDS,
    VERSION_ENDPOINT,
    ClinicalTrialsGovIntakeError,
    HttpResponse,
    fetch_clinicaltrials_gov_snapshot,
    request_url_query,
    verify_clinicaltrials_gov_snapshot,
)
from crispr_evidencerank.contracts import ClinicalTrialsGovStudyInventoryRecord

_OMIT = object()
_NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def _json_bytes(value: object, *, indent: int | None = None) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _version_bytes(
    *,
    api_version: str = "2.0.5",
    data_timestamp: str = "2026-08-31T09:00:04Z",
    padding: str | None = None,
    synthetic_fixture: bool = False,
) -> bytes:
    if synthetic_fixture and "synthetic" not in api_version.casefold():
        api_version = f"{api_version}-synthetic"
    value: dict[str, object] = {
        "apiVersion": api_version,
        "dataTimestamp": data_timestamp,
    }
    if synthetic_fixture:
        value["syntheticFixture"] = True
    if padding is not None:
        value["syntheticPadding"] = padding
    return _json_bytes(value, indent=2)


def _study(
    nct_id: str,
    *,
    intervention: str | None = "Olaparib",
    condition: str | None = "Triple-Negative Breast Cancer",
    title: str | None = None,
    has_results: bool | None = False,
    results_first_post_date: str | None = None,
) -> dict[str, object]:
    protocol: dict[str, object] = {
        "identificationModule": {
            "nctId": nct_id,
            "briefTitle": title or f"Synthetic study {nct_id}",
            "officialTitle": f"Synthetic protocol for {nct_id}",
        },
        "statusModule": {
            "overallStatus": "RECRUITING",
            "studyFirstPostDateStruct": {"date": "2025-01-02", "type": "ACTUAL"},
            "lastUpdatePostDateStruct": {"date": "2026-08", "type": "ACTUAL"},
        },
        "designModule": {
            "studyType": "INTERVENTIONAL",
            "phases": ["PHASE2"],
            "designInfo": {
                "allocation": "RANDOMIZED",
                "interventionModel": "PARALLEL",
                "primaryPurpose": "TREATMENT",
            },
            "enrollmentInfo": {"count": 42, "type": "ESTIMATED"},
        },
        "descriptionModule": {
            "briefSummary": "A wholly synthetic offline test record.",
            "detailedDescription": "No person or real trial is represented.",
        },
        "eligibilityModule": {
            "eligibilityCriteria": "Synthetic inclusion criteria",
            "healthyVolunteers": False,
            "sex": "ALL",
            "minimumAge": "18 Years",
            "maximumAge": "80 Years",
        },
    }
    if condition is not None:
        protocol["conditionsModule"] = {
            "conditions": [condition],
            "keywords": ["synthetic", "BRCA"],
        }
    if intervention is not None:
        protocol["armsInterventionsModule"] = {
            "interventions": [
                {
                    "type": "DRUG",
                    "name": intervention,
                    "description": "Synthetic intervention description",
                    "armGroupLabels": ["Experimental arm"],
                    "otherNames": ["AZD2281"],
                }
            ],
            "armGroups": [
                {
                    "label": "Experimental arm",
                    "type": "EXPERIMENTAL",
                    "description": "Synthetic arm",
                    "interventionNames": [f"DRUG: {intervention}"],
                }
            ],
        }
    status_module = protocol["statusModule"]
    assert isinstance(status_module, dict)
    if results_first_post_date is not None:
        status_module["resultsFirstPostDateStruct"] = {
            "date": results_first_post_date,
            "type": "ACTUAL",
        }
    study: dict[str, object] = {
        "protocolSection": protocol,
        "derivedSection": {"miscInfoModule": {"versionHolder": "2026-08-31"}},
    }
    if has_results is not None:
        study["hasResults"] = has_results
    return study


def _study_with_cartesian_candidates(
    nct_id: str,
    *,
    intervention_count: int,
    condition_count: int,
) -> dict[str, object]:
    study = _study(nct_id)
    protocol = study["protocolSection"]
    assert isinstance(protocol, dict)
    conditions_module = protocol["conditionsModule"]
    arms_module = protocol["armsInterventionsModule"]
    assert isinstance(conditions_module, dict)
    assert isinstance(arms_module, dict)
    conditions_module["conditions"] = [
        f"Synthetic cancer {index}" for index in range(condition_count)
    ]
    arms_module["interventions"] = [
        {
            "type": "DRUG",
            "name": f"Synthetic drug {index}",
            "description": f"Synthetic intervention {index}",
            "armGroupLabels": [],
            "otherNames": [],
        }
        for index in range(intervention_count)
    ]
    arms_module["armGroups"] = []
    return study


def _page_bytes(
    studies: Sequence[dict[str, object]],
    *,
    total_count: object = _OMIT,
    next_page_token: object = _OMIT,
    indent: int | None = 2,
    synthetic_fixture: bool = False,
) -> bytes:
    value: dict[str, object] = {"studies": list(studies)}
    if synthetic_fixture:
        value["syntheticFixture"] = True
    if total_count is not _OMIT:
        value["totalCount"] = total_count
    if next_page_token is not _OMIT:
        value["nextPageToken"] = next_page_token
    return _json_bytes(value, indent=indent)


class _ScriptedRequester:
    def __init__(
        self,
        responses: Sequence[bytes],
        *,
        final_url: Callable[[str, int], str] | None = None,
        header_updates: Mapping[str, str] | None = None,
    ) -> None:
        self._responses = list(responses)
        self._final_url = final_url
        self._header_updates = dict(header_updates or {})
        self.urls: list[str] = []
        self.arguments: list[dict[str, object]] = []

    def __call__(
        self,
        url: str,
        *,
        timeout_seconds: float,
        max_response_bytes: int,
        max_attempts: int,
        backoff_seconds: float,
    ) -> HttpResponse:
        index = len(self.urls)
        if index >= len(self._responses):
            raise AssertionError(f"unexpected request {index + 1}: {url}")
        body = self._responses[index]
        self.urls.append(url)
        self.arguments.append(
            {
                "timeout_seconds": timeout_seconds,
                "max_response_bytes": max_response_bytes,
                "max_attempts": max_attempts,
                "backoff_seconds": backoff_seconds,
            }
        )
        headers = {
            "content-type": "application/json; charset=utf-8",
            "content-length": str(len(body)),
            "content-encoding": "identity",
            **self._header_updates,
        }
        return HttpResponse(
            body=body,
            status_code=200,
            final_url=(
                self._final_url(url, index) if self._final_url is not None else url
            ),
            headers=headers,
            attempt_count=1,
        )


def _two_page_requester() -> tuple[_ScriptedRequester, bytes, bytes, bytes]:
    version = _version_bytes()
    page_one = _page_bytes(
        [_study("NCT00000001")],
        total_count=2,
        next_page_token="opaque token/+==",
    )
    page_two = _page_bytes(
        [
            _study(
                "NCT00000002",
                intervention="Carboplatin",
                condition="Breast Cancer",
            )
        ]
    )
    return (
        _ScriptedRequester([version, page_one, page_two, version]),
        version,
        page_one,
        page_two,
    )


def _fetch(
    output_dir: Path,
    requester: _ScriptedRequester,
    **updates: object,
) -> dict[str, object]:
    options: dict[str, object] = {
        "condition_query": "Triple Negative Breast Cancer",
        "intervention_query": "olaparib",
        "output_dir": output_dir,
        "page_size": 1,
        "requester": requester,
        "clock": lambda: _NOW,
        "monotonic": lambda: 0.0,
    }
    options.update(updates)
    return fetch_clinicaltrials_gov_snapshot(**options)  # type: ignore[arg-type]


def _canonical_manifest_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_manifest_for_changed_output(snapshot: Path, filename: str) -> None:
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = snapshot / filename
    for output in manifest["outputs"]:
        if output["filename"] == filename:
            output["sha256"] = _sha256(changed)
            output["byte_size"] = changed.stat().st_size
            break
    else:  # pragma: no cover - helper misuse protection
        raise AssertionError(f"output not found in manifest: {filename}")
    content_index = [
        {
            "filename": output["filename"],
            "sha256": output["sha256"],
            "byte_size": output["byte_size"],
        }
        for output in manifest["outputs"]
    ]
    manifest["bundle_content_sha256"] = hashlib.sha256(
        _canonical_manifest_bytes(content_index)
    ).hexdigest()
    manifest_path.write_bytes(_canonical_manifest_bytes(manifest))


def _mutate_manifest(
    snapshot: Path,
    mutation: Callable[[dict[str, object]], None],
) -> None:
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    mutation(manifest)
    manifest_path.write_bytes(_canonical_manifest_bytes(manifest))


def _inventory_contract_record(
    *,
    source_has_results: bool | None,
    results_first_post_date: str | None,
) -> dict[str, object]:
    return {
        "snapshot_id": "ctgov-contract-test",
        "source_study_id": "NCT00000001",
        "source_family_id": "ctgov:NCT00000001",
        "source_asset_id": "ctgov-contract-test:page:000001",
        "source_asset_sha256": "a" * 64,
        "source_page_index": 1,
        "source_study_index": 0,
        "source_version_holder": "2026-08-31",
        "record_last_update_date": "2026-08",
        "study_first_post_date": "2025-01-02",
        "results_first_post_date": results_first_post_date,
        "source_study_type": "INTERVENTIONAL",
        "source_overall_status": "RECRUITING",
        "source_brief_title": "Synthetic contract record",
        "source_official_title": None,
        "source_phases_json": "[]",
        "source_conditions_json": "[]",
        "source_keywords_json": "[]",
        "source_design_json": "{}",
        "source_interventions_json": "[]",
        "source_arm_groups_json": "[]",
        "population_scope_locator_candidates_json": "[]",
        "population_scope_text_sha256": "b" * 64,
        "source_has_results": source_has_results,
        "source_url": "https://clinicaltrials.gov/study/NCT00000001",
        "retrieved_at_utc": "2026-09-01T12:00:00Z",
        "tsv_formula_escape_applied": False,
        "normalization_status": "raw_registry_inventory_only",
        "used_for_label": False,
    }


def test_fetch_preserves_exact_pages_and_builds_a_verifiable_fail_closed_bundle(
    tmp_path: Path,
) -> None:
    requester, version, page_one, page_two = _two_page_requester()
    output = tmp_path / "snapshot"

    manifest = _fetch(output, requester)

    assert output.is_dir()
    assert (output / "version_start.json").read_bytes() == version
    assert (output / "version_end.json").read_bytes() == version
    assert (output / "pages/page_000001.json").read_bytes() == page_one
    assert (output / "pages/page_000002.json").read_bytes() == page_two
    assert manifest["page_count"] == 2
    assert manifest["total_count"] == 2
    assert manifest["observed_unique_study_count"] == 2
    source = manifest["source"]
    assert isinstance(source, dict)
    assert source["name"] == "ClinicalTrials.gov API-shaped injected transport"
    assert source["synthetic_fixture"] is False
    assert source["transport_mode"] == "injected"
    assert source["clock_mode"] == "injected"
    assert source["elapsed_clock_mode"] == "injected"
    assert source["mutable_registry"] is None

    # The API currently returns totalCount on the first page only.  The opaque
    # nextPageToken must be passed back byte-for-byte after URL encoding.
    assert len(requester.urls) == 4
    assert requester.urls[0] == VERSION_ENDPOINT
    first_query = request_url_query(requester.urls[1])
    second_query = request_url_query(requester.urls[2])
    assert first_query["countTotal"] == ["true"]
    assert first_query["format"] == ["json"]
    assert first_query["markupFormat"] == ["markdown"]
    assert first_query["pageSize"] == ["1"]
    assert first_query["fields"] == [",".join(STUDY_FIELDS)]
    assert "pageToken" not in first_query
    assert second_query["pageToken"] == ["opaque token/+=="]
    assert requester.urls[3] == VERSION_ENDPOINT

    inventory = pd.read_csv(output / "study_inventory.tsv", sep="\t")
    queue = pd.read_csv(output / "curation_queue.tsv", sep="\t")
    assets = pd.read_csv(output / "data_assets.tsv", sep="\t")
    assert set(assets["source_name"]) == {
        "ClinicalTrials.gov API-shaped injected transport"
    }
    assert assets["download_method"].str.contains("injected response transport").all()
    assert inventory["source_study_id"].tolist() == [
        "NCT00000001",
        "NCT00000002",
    ]
    assert len(queue) == 2
    assert queue["candidate_id"].is_unique
    assert set(queue["snapshot_id"]) == {manifest["snapshot_id"]}
    assert not queue["eligible_for_clinical_context"].any()
    assert not queue["used_for_label"].any()
    assert queue["co_mention_only"].all()
    assert set(queue["treatment_mapping_review_status"]) == {"not_performed"}
    assert set(queue["cancer_mapping_review_status"]) == {"not_performed"}
    assert set(queue["treatment_cancer_linkage_status"]) == {"not_performed"}
    assert set(queue["intervention_role"]) == {"unknown"}
    assert set(queue["normalization_status"]) == {"requires_curator_review"}
    assert set(queue["exclusion_reason"]) == {"unreviewed_study_level_co_mention"}
    assert queue["treatment_concept_id"].isna().all()
    assert queue["cancer_concept_id"].isna().all()
    assert json.loads(queue.iloc[0]["source_linked_arm_groups_json"]) == [
        {
            "description": "Synthetic arm",
            "interventionNames": ["DRUG: Olaparib"],
            "label": "Experimental arm",
            "type": "EXPERIMENTAL",
        }
    ]

    page_assets = assets.loc[assets["asset_role"] == "registry_api_page"]
    expected_page_hashes = {
        hashlib.sha256(page_one).hexdigest(),
        hashlib.sha256(page_two).hexdigest(),
    }
    assert set(page_assets["sha256"]) == expected_page_hashes
    assert set(inventory["source_asset_sha256"]) == expected_page_hashes
    assert set(queue["source_asset_sha256"]) == expected_page_hashes

    verification = verify_clinicaltrials_gov_snapshot(output)
    assert verification == {
        "snapshot_id": manifest["snapshot_id"],
        "api_version": "2.0.5",
        "data_timestamp": "2026-08-31T09:00:04Z",
        "page_count": 2,
        "study_count": 2,
        "complete": True,
        "integrity_scope": "internal_bundle_consistency_not_publisher_authenticity",
    }


def test_empty_intermediate_page_with_a_token_is_accepted(tmp_path: Path) -> None:
    version = _version_bytes()
    requester = _ScriptedRequester(
        [
            version,
            _page_bytes([], total_count=1, next_page_token="first"),
            _page_bytes([], next_page_token="second"),
            _page_bytes([_study("NCT00000001")]),
            version,
        ]
    )

    manifest = _fetch(tmp_path / "snapshot", requester)

    assert manifest["page_count"] == 3
    assert manifest["total_count"] == 1
    assert verify_clinicaltrials_gov_snapshot(tmp_path / "snapshot")["study_count"] == 1


def test_zero_result_snapshot_is_complete_and_verifiable(tmp_path: Path) -> None:
    version = _version_bytes()
    requester = _ScriptedRequester([version, _page_bytes([], total_count=0), version])

    manifest = _fetch(tmp_path / "snapshot", requester)

    assert manifest["total_count"] == 0
    assert pd.read_csv(tmp_path / "snapshot/study_inventory.tsv", sep="\t").empty
    assert pd.read_csv(tmp_path / "snapshot/curation_queue.tsv", sep="\t").empty
    assert verify_clinicaltrials_gov_snapshot(tmp_path / "snapshot")["study_count"] == 0


@pytest.mark.parametrize(
    ("has_results", "results_first_post_date"),
    [
        (True, None),
        (False, "2026-08-15"),
    ],
)
def test_inventory_contract_does_not_infer_one_results_field_from_the_other(
    has_results: bool,
    results_first_post_date: str | None,
) -> None:
    record = ClinicalTrialsGovStudyInventoryRecord.model_validate(
        _inventory_contract_record(
            source_has_results=has_results,
            results_first_post_date=results_first_post_date,
        )
    )

    assert record.source_has_results is has_results
    assert record.results_first_post_date == results_first_post_date


@pytest.mark.parametrize(
    ("has_results", "results_first_post_date"),
    [
        (True, None),
        (False, "2026-08-15"),
    ],
)
def test_has_results_and_results_first_post_date_are_independent_source_fields(
    tmp_path: Path,
    has_results: bool,
    results_first_post_date: str | None,
) -> None:
    version = _version_bytes()
    page = _page_bytes(
        [
            _study(
                "NCT00000001",
                has_results=has_results,
                results_first_post_date=results_first_post_date,
            )
        ],
        total_count=1,
    )
    output = tmp_path / f"has-results-{has_results}"

    _fetch(output, _ScriptedRequester([version, page, version]))

    inventory = pd.read_csv(
        output / "study_inventory.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    row = inventory.iloc[0]
    assert row["source_has_results"] == str(has_results)
    assert row["results_first_post_date"] == (results_first_post_date or "")
    assert verify_clinicaltrials_gov_snapshot(output)["study_count"] == 1


def test_candidate_identifiers_are_namespaced_by_snapshot(tmp_path: Path) -> None:
    version = _version_bytes()
    page = _page_bytes([_study("NCT00000001")], total_count=1)
    first_requester = _ScriptedRequester([version, page, version])
    second_requester = _ScriptedRequester([version, page, version])
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = _fetch(first, first_requester)
    second_manifest = _fetch(
        second,
        second_requester,
        condition_query="Breast Cancer",
    )
    first_ids = set(pd.read_csv(first / "curation_queue.tsv", sep="\t")["candidate_id"])
    second_ids = set(
        pd.read_csv(second / "curation_queue.tsv", sep="\t")["candidate_id"]
    )

    assert first_manifest["snapshot_id"] != second_manifest["snapshot_id"]
    assert first_ids.isdisjoint(second_ids)
    assert all(str(first_manifest["snapshot_id"]) in value for value in first_ids)
    assert all(str(second_manifest["snapshot_id"]) in value for value in second_ids)


def test_same_raw_query_and_version_produce_same_snapshot_identity(
    tmp_path: Path,
) -> None:
    first_requester, _, _, _ = _two_page_requester()
    second_requester, _, _, _ = _two_page_requester()

    first = _fetch(tmp_path / "first", first_requester)
    second = _fetch(tmp_path / "second", second_requester)

    assert first["snapshot_id"] == second["snapshot_id"]


def test_synthetic_fixture_mode_is_explicit_and_uses_project_provenance(
    tmp_path: Path,
) -> None:
    version = _version_bytes(synthetic_fixture=True)
    page = _page_bytes(
        [_study("NCT90000001", title="Synthetic olaparib TNBC fixture")],
        total_count=1,
        synthetic_fixture=True,
    )
    requester = _ScriptedRequester(
        [
            version,
            page,
            version,
        ]
    )
    output = tmp_path / "synthetic"

    manifest = _fetch(output, requester, synthetic_fixture=True)

    assert manifest["source"]["name"] == "ClinicalTrials.gov synthetic fixture"
    assert manifest["source"]["synthetic_fixture"] is True
    assert manifest["source"]["transport_mode"] == "injected"
    assert manifest["source"]["clock_mode"] == "injected"
    assert manifest["source"]["elapsed_clock_mode"] == "injected"
    assert manifest["source"]["mutable_registry"] is False
    assert manifest["scientific_boundary"]["synthetic_fixture"] is True
    snapshot_id = str(manifest["snapshot_id"])
    source_family_id = f"crispr-evidencerank:synthetic-ctgov:{snapshot_id}:NCT90000001"
    raw_family_id = f"crispr-evidencerank:synthetic-ctgov:raw-snapshot:{snapshot_id}"
    reference_base = (
        "https://example.invalid/crispr-evidencerank/"
        f"synthetic-clinicaltrials-gov/{snapshot_id}"
    )
    inventory = pd.read_csv(output / "study_inventory.tsv", sep="\t")
    queue = pd.read_csv(output / "curation_queue.tsv", sep="\t")
    assets = pd.read_csv(output / "data_assets.tsv", sep="\t")
    assert inventory["source_study_id"].tolist() == ["NCT90000001"]
    assert set(inventory["source_family_id"]) == {source_family_id}
    assert set(queue["source_family_id"]) == {source_family_id}
    assert queue["candidate_id"].str.startswith(f"{snapshot_id}:synthetic-ctgov:").all()
    assert set(inventory["source_url"]) == {f"{reference_base}/studies/NCT90000001"}
    page_asset_id = f"crispr-evidencerank:synthetic-ctgov:{snapshot_id}:page:000001"
    assert set(inventory["source_asset_id"]) == {page_asset_id}
    assert set(queue["source_asset_id"]) == {page_asset_id}
    assert set(inventory["source_asset_sha256"]) == {hashlib.sha256(page).hexdigest()}
    assert set(queue["source_asset_sha256"]) == {hashlib.sha256(page).hexdigest()}
    assert set(assets["source_name"]) == {"ClinicalTrials.gov synthetic fixture"}
    assert (
        assets["asset_id"]
        .str.startswith(f"crispr-evidencerank:synthetic-ctgov:{snapshot_id}:")
        .all()
    )
    assert set(assets["raw_data_family_id"]) == {raw_family_id}
    expected_asset_paths = {
        "version-start": output / "version_start.json",
        "page-000001": output / "pages/page_000001.json",
        "version-end": output / "version_end.json",
    }
    assert set(assets["source_url"]) == {
        f"{reference_base}/raw-assets/{suffix}" for suffix in expected_asset_paths
    }
    for suffix, path in expected_asset_paths.items():
        row = assets.loc[
            assets["source_url"] == f"{reference_base}/raw-assets/{suffix}"
        ].iloc[0]
        assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert int(row["byte_size"]) == path.stat().st_size
    assert set(assets["checksum_provenance"]) == {
        "project_computed_from_exact_injected_synthetic_response_bytes"
    }
    assert assets["redistribution_raw"].all()
    assert assets["redistribution_derived"].all()
    assert set(assets["license_spdx"]) == {"Apache-2.0"}
    assert assets["notes"].str.startswith("SYNTHETIC FIXTURE:").all()
    assert assets["download_method"].str.contains("no network").all()
    assert verify_clinicaltrials_gov_snapshot(output)["study_count"] == 1


def test_synthetic_fixture_mode_requires_an_injected_requester(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="injected offline requester"):
        fetch_clinicaltrials_gov_snapshot(
            condition_query="synthetic cancer",
            intervention_query="synthetic treatment",
            output_dir=tmp_path / "snapshot",
            synthetic_fixture=True,
        )


def test_response_final_url_must_match_the_exact_requested_query(
    tmp_path: Path,
) -> None:
    version = _version_bytes()
    page = _page_bytes([], total_count=0)
    requester = _ScriptedRequester(
        [version, page, version],
        final_url=lambda url, index: f"{url}&injected=true" if index == 1 else url,
    )

    with pytest.raises(
        ClinicalTrialsGovIntakeError,
        match="response URL differs from the exact requested URL",
    ):
        _fetch(tmp_path / "snapshot", requester)

    assert not (tmp_path / "snapshot").exists()


@pytest.mark.parametrize(
    ("header_updates", "message"),
    [
        ({"content-type": "text/html"}, "not application/json"),
        ({"content-type": "application/jsonevil"}, "not application/json"),
        ({"content-encoding": "gzip"}, "Content-Encoding must be identity"),
        ({"content-length": "1"}, "differs from Content-Length"),
    ],
)
def test_response_metadata_is_fail_closed(
    tmp_path: Path,
    header_updates: dict[str, str],
    message: str,
) -> None:
    version = _version_bytes()
    requester = _ScriptedRequester(
        [version],
        header_updates=header_updates,
    )

    with pytest.raises(ClinicalTrialsGovIntakeError, match=message):
        _fetch(tmp_path / "snapshot", requester)


def test_repeated_pagination_token_is_rejected(tmp_path: Path) -> None:
    version = _version_bytes()
    requester = _ScriptedRequester(
        [
            version,
            _page_bytes(
                [_study("NCT00000001")],
                total_count=3,
                next_page_token="loop",
            ),
            _page_bytes([_study("NCT00000002")], next_page_token="loop"),
        ]
    )

    with pytest.raises(ClinicalTrialsGovIntakeError, match="token repeated"):
        _fetch(tmp_path / "snapshot", requester)


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        (
            lambda version: [
                version,
                _page_bytes(
                    [_study("NCT00000001")],
                    total_count=2,
                    next_page_token="next",
                ),
                _page_bytes([_study("NCT00000001")]),
            ],
            "repeat across pages",
        ),
        (
            lambda version: [
                version,
                _page_bytes(
                    [_study("NCT00000001")],
                    total_count=2,
                    next_page_token="next",
                ),
                _page_bytes([_study("NCT00000002")], total_count=3),
            ],
            "totalCount changed",
        ),
        (
            lambda version: [
                version,
                _page_bytes([_study("NCT00000001")], total_count=2),
                version,
            ],
            "count does not equal totalCount",
        ),
    ],
)
def test_duplicate_identifiers_and_count_drift_are_rejected(
    tmp_path: Path,
    responses: Callable[[bytes], list[bytes]],
    message: str,
) -> None:
    version = _version_bytes()
    requester = _ScriptedRequester(responses(version))

    with pytest.raises(ClinicalTrialsGovIntakeError, match=message):
        _fetch(tmp_path / "snapshot", requester)


def test_version_envelope_drift_is_rejected(tmp_path: Path) -> None:
    requester = _ScriptedRequester(
        [
            _version_bytes(),
            _page_bytes([], total_count=0),
            _version_bytes(data_timestamp="2026-09-01T09:00:04Z"),
        ]
    )

    with pytest.raises(
        ClinicalTrialsGovIntakeError,
        match="version/data timestamp changed",
    ):
        _fetch(tmp_path / "snapshot", requester)


@pytest.mark.parametrize(
    ("invalid_page", "message"),
    [
        (b"{\n", "not valid JSON"),
        (b"\xff", "not UTF-8 JSON"),
        (
            b'{"studies":[],"studies":[],"totalCount":0}',
            "duplicate JSON object key: studies",
        ),
        (
            b'{"studies":[],"totalCount":NaN}',
            "non-finite JSON constant is forbidden: NaN",
        ),
    ],
)
def test_malformed_json_utf8_duplicate_keys_and_nan_are_rejected(
    tmp_path: Path,
    invalid_page: bytes,
    message: str,
) -> None:
    requester = _ScriptedRequester([_version_bytes(), invalid_page])

    with pytest.raises(ClinicalTrialsGovIntakeError, match=message):
        _fetch(tmp_path / "snapshot", requester)


def test_page_limit_stops_before_an_unbounded_followup_request(tmp_path: Path) -> None:
    version = _version_bytes()
    requester = _ScriptedRequester(
        [
            version,
            _page_bytes(
                [_study("NCT00000001")],
                total_count=2,
                next_page_token="next",
            ),
        ]
    )

    with pytest.raises(ClinicalTrialsGovIntakeError, match="before max_pages"):
        _fetch(tmp_path / "snapshot", requester, max_pages=1)

    assert len(requester.urls) == 2


@pytest.mark.parametrize(
    ("page", "max_studies", "message"),
    [
        (
            _page_bytes(
                [_study("NCT00000001")],
                total_count=2,
                next_page_token="next",
            ),
            1,
            "totalCount exceeds",
        ),
        (
            _page_bytes(
                [_study("NCT00000001"), _study("NCT00000002")],
                total_count=1,
            ),
            1,
            "observed study count exceeds",
        ),
    ],
)
def test_declared_and_observed_study_caps_are_enforced(
    tmp_path: Path,
    page: bytes,
    max_studies: int,
    message: str,
) -> None:
    requester = _ScriptedRequester([_version_bytes(), page])

    with pytest.raises(ClinicalTrialsGovIntakeError, match=message):
        _fetch(tmp_path / "snapshot", requester, max_studies=max_studies)


def test_candidate_cartesian_product_is_preflighted_before_row_materialization(
    tmp_path: Path,
) -> None:
    version = _version_bytes()
    page = _page_bytes(
        [
            _study_with_cartesian_candidates(
                "NCT00000001",
                intervention_count=3,
                condition_count=4,
            )
        ],
        total_count=1,
    )
    requester = _ScriptedRequester([version, page])

    with pytest.raises(
        ClinicalTrialsGovIntakeError,
        match="candidate product exceeds the configured limit",
    ):
        _fetch(tmp_path / "snapshot", requester, max_candidate_rows=11)

    assert not (tmp_path / "snapshot").exists()
    assert list(tmp_path.glob(".snapshot.staging-*")) == []
    assert not (tmp_path / ".snapshot.lock").exists()


def test_candidate_row_cap_is_cumulative_across_pages(tmp_path: Path) -> None:
    version = _version_bytes()
    requester = _ScriptedRequester(
        [
            version,
            _page_bytes(
                [_study("NCT00000001")],
                total_count=2,
                next_page_token="second",
            ),
            _page_bytes([_study("NCT00000002")]),
        ]
    )

    with pytest.raises(
        ClinicalTrialsGovIntakeError,
        match="candidate product exceeds the configured limit",
    ):
        _fetch(tmp_path / "snapshot", requester, max_candidate_rows=1)

    assert len(requester.urls) == 3
    assert not (tmp_path / "snapshot").exists()


def test_derived_byte_limit_accepts_exact_boundary_and_rejects_one_less(
    tmp_path: Path,
) -> None:
    version = _version_bytes()
    page = _page_bytes([], total_count=0)
    baseline = _fetch(
        tmp_path / "baseline",
        _ScriptedRequester([version, page, version]),
    )
    retrieval = baseline["retrieval"]
    assert isinstance(retrieval, dict)
    exact_limit = retrieval["total_derived_bytes"]
    assert isinstance(exact_limit, int) and exact_limit > 1

    exact = _fetch(
        tmp_path / "exact",
        _ScriptedRequester([version, page, version]),
        max_derived_bytes=exact_limit,
    )
    exact_retrieval = exact["retrieval"]
    assert isinstance(exact_retrieval, dict)
    assert exact_retrieval["total_derived_bytes"] == exact_limit

    with pytest.raises(
        ClinicalTrialsGovIntakeError,
        match="derived snapshot tables exceed the configured byte limit",
    ):
        _fetch(
            tmp_path / "too-small",
            _ScriptedRequester([version, page, version]),
            max_derived_bytes=exact_limit - 1,
        )

    assert not (tmp_path / "too-small").exists()


def test_per_response_byte_cap_is_enforced(tmp_path: Path) -> None:
    page = _page_bytes([_study("NCT00000001")], total_count=1)
    assert len(_version_bytes()) < 256 < len(page)
    requester = _ScriptedRequester([_version_bytes(), page])

    with pytest.raises(ClinicalTrialsGovIntakeError, match="byte limit"):
        _fetch(
            tmp_path / "snapshot",
            requester,
            max_page_bytes=256,
            max_total_bytes=256,
        )


def test_default_requester_checks_deadline_after_a_blocking_eof_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _version_bytes()

    class FakeResponse:
        status = 200
        headers = {
            "content-type": "application/json",
            "content-length": str(len(body)),
        }

        def __init__(self) -> None:
            self.read_count = 0

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def getcode(self) -> int:
            return self.status

        def geturl(self) -> str:
            return VERSION_ENDPOINT

        def read1(self, _: int) -> bytes:
            self.read_count += 1
            return body if self.read_count == 1 else b""

        def read(self, size: int) -> bytes:
            return self.read1(size)

    class FakeOpener:
        def open(self, *_: object, **__: object) -> FakeResponse:
            return FakeResponse()

    monotonic_values = iter([0.0, 0.0, 0.0, 0.0, 0.0, 5.0])
    monkeypatch.setattr(ctgov_module, "build_opener", lambda *_: FakeOpener())
    monkeypatch.setattr(
        ctgov_module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    with pytest.raises(ClinicalTrialsGovIntakeError, match="time budget"):
        ctgov_module._default_json_requester(
            VERSION_ENDPOINT,
            timeout_seconds=1.0,
            max_response_bytes=1024,
            max_attempts=1,
            backoff_seconds=0.0,
        )


def test_cumulative_raw_byte_cap_includes_both_version_envelopes(
    tmp_path: Path,
) -> None:
    version = _version_bytes(padding="x" * 80)
    page = _page_bytes([], total_count=0)
    assert len(version) < 256 and len(page) < 256
    assert len(version) + len(page) < 256 < len(version) * 2 + len(page)
    requester = _ScriptedRequester([version, page, version])

    with pytest.raises(
        ClinicalTrialsGovIntakeError,
        match=r"configured (?:total )?byte limit",
    ):
        _fetch(
            tmp_path / "snapshot",
            requester,
            max_page_bytes=256,
            max_total_bytes=256,
        )


def test_elapsed_time_cap_is_checked_between_requests(tmp_path: Path) -> None:
    times = iter([0.0, 0.0, 2.0])
    requester = _ScriptedRequester([_version_bytes()])

    with pytest.raises(ClinicalTrialsGovIntakeError, match="max_elapsed_seconds"):
        _fetch(
            tmp_path / "snapshot",
            requester,
            max_elapsed_seconds=1.0,
            monotonic=lambda: next(times),
        )

    assert requester.urls == [VERSION_ENDPOINT]


def test_verifier_applies_declared_page_cap_to_version_envelopes(
    tmp_path: Path,
) -> None:
    version = _version_bytes()
    page = _page_bytes([], total_count=0)
    assert len(version) > len(page)
    output = tmp_path / "version-cap"
    _fetch(output, _ScriptedRequester([version, page, version]))

    def mutate(manifest: dict[str, object]) -> None:
        retrieval = manifest["retrieval"]
        assert isinstance(retrieval, dict)
        retrieval["max_page_bytes"] = len(page)

    _mutate_manifest(output, mutate)

    with pytest.raises(
        ClinicalTrialsGovIntakeError,
        match="version response exceeds declared max_page_bytes",
    ):
        verify_clinicaltrials_gov_snapshot(output)


def test_failed_retrieval_removes_staging_and_never_publishes_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "snapshot"
    requester = _ScriptedRequester([_version_bytes(), b"{"])

    with pytest.raises(ClinicalTrialsGovIntakeError):
        _fetch(output, requester)

    assert not output.exists()
    assert list(tmp_path.glob(".snapshot.staging-*")) == []


def test_preexisting_exclusive_publication_lock_blocks_before_network(
    tmp_path: Path,
) -> None:
    output = tmp_path / "snapshot"
    requester, _, _, _ = _two_page_requester()

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    parent_descriptor = os.open(tmp_path, flags)
    fcntl.flock(parent_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(FileExistsError, match="publication is already locked"):
            _fetch(output, requester)
    finally:
        fcntl.flock(parent_descriptor, fcntl.LOCK_UN)
        os.close(parent_descriptor)

    assert requester.urls == []
    assert not output.exists()
    assert list(tmp_path.glob(".snapshot.staging-*")) == []


def test_active_fetch_lock_excludes_a_reentrant_fetch(tmp_path: Path) -> None:
    output = tmp_path / "snapshot"
    outer_requester, _, _, _ = _two_page_requester()
    nested_requester, _, _, _ = _two_page_requester()
    nested_attempted = False

    def reentrant_requester(
        url: str,
        *,
        timeout_seconds: float,
        max_response_bytes: int,
        max_attempts: int,
        backoff_seconds: float,
    ) -> HttpResponse:
        nonlocal nested_attempted
        if not nested_attempted:
            nested_attempted = True
            with pytest.raises(
                FileExistsError,
                match="publication is already locked",
            ):
                _fetch(output, nested_requester)
        return outer_requester(
            url,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
        )

    _fetch(output, reentrant_requester)  # type: ignore[arg-type]

    assert nested_attempted
    assert nested_requester.urls == []
    assert output.is_dir()
    assert not (tmp_path / ".snapshot.lock").exists()


def test_raced_destination_is_preserved_by_atomic_no_replace_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "snapshot"
    requester, _, _, _ = _two_page_requester()
    original_publish = ctgov_module._publish_directory_noreplace

    def publish_after_race(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "raced-owner.txt").write_text(
            "must survive\n",
            encoding="utf-8",
        )
        original_publish(source, destination)

    monkeypatch.setattr(
        ctgov_module,
        "_publish_directory_noreplace",
        publish_after_race,
    )

    with pytest.raises(FileExistsError, match="snapshot output exists"):
        _fetch(output, requester)

    assert (output / "raced-owner.txt").read_text(encoding="utf-8") == (
        "must survive\n"
    )
    assert sorted(path.name for path in output.iterdir()) == ["raced-owner.txt"]
    assert list(tmp_path.glob(".snapshot.staging-*")) == []
    assert not (tmp_path / ".snapshot.lock").exists()


def test_post_publication_verification_rejects_a_last_moment_staging_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "snapshot"
    requester, _, _, _ = _two_page_requester()
    original_publish = ctgov_module._publish_directory_noreplace

    def mutate_then_publish(source: Path, destination: Path) -> None:
        inventory = source / "study_inventory.tsv"
        inventory.write_bytes(inventory.read_bytes() + b" ")
        original_publish(source, destination)

    monkeypatch.setattr(
        ctgov_module,
        "_publish_directory_noreplace",
        mutate_then_publish,
    )

    with pytest.raises(ClinicalTrialsGovIntakeError, match="checksum/size mismatch"):
        _fetch(output, requester)

    assert output.is_dir()
    with pytest.raises(ClinicalTrialsGovIntakeError, match="checksum/size mismatch"):
        verify_clinicaltrials_gov_snapshot(output)


def test_verifier_rejects_raw_file_tampering(tmp_path: Path) -> None:
    requester, _, _, _ = _two_page_requester()
    output = tmp_path / "snapshot"
    _fetch(output, requester)
    page = output / "pages/page_000001.json"
    page.write_bytes(page.read_bytes() + b" ")

    with pytest.raises(ClinicalTrialsGovIntakeError, match="checksum/size mismatch"):
        verify_clinicaltrials_gov_snapshot(output)


def test_verifier_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    requester, _, _, _ = _two_page_requester()
    output = tmp_path / "snapshot"
    _fetch(output, requester)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"][0]["filename"] = "../version_start.json"
    manifest_path.write_bytes(_canonical_manifest_bytes(manifest))

    with pytest.raises(ClinicalTrialsGovIntakeError, match="unsafe bundle filename"):
        verify_clinicaltrials_gov_snapshot(output)


def test_verifier_rejects_a_symlinked_output(tmp_path: Path) -> None:
    requester, _, _, _ = _two_page_requester()
    output = tmp_path / "snapshot"
    _fetch(output, requester)
    page = output / "pages/page_000001.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(page.read_bytes())
    page.unlink()
    page.symlink_to(outside)

    with pytest.raises(ClinicalTrialsGovIntakeError, match="contain a symlink"):
        verify_clinicaltrials_gov_snapshot(output)


def test_verifier_rejects_fail_closed_queue_tampering_even_when_rehashed(
    tmp_path: Path,
) -> None:
    requester, _, _, _ = _two_page_requester()
    output = tmp_path / "snapshot"
    _fetch(output, requester)
    queue_path = output / "curation_queue.tsv"
    queue = pd.read_csv(queue_path, sep="\t", dtype=str, keep_default_na=False)
    queue.loc[0, "eligible_for_clinical_context"] = "True"
    queue.to_csv(queue_path, sep="\t", index=False, lineterminator="\n")
    _rewrite_manifest_for_changed_output(output, "curation_queue.tsv")

    with pytest.raises(ClinicalTrialsGovIntakeError, match="curation queue"):
        verify_clinicaltrials_gov_snapshot(output)


def test_verifier_rederives_queue_from_raw_pages_instead_of_trusting_rehashed_tsv(
    tmp_path: Path,
) -> None:
    requester, _, _, _ = _two_page_requester()
    output = tmp_path / "snapshot"
    _fetch(output, requester)
    queue_path = output / "curation_queue.tsv"
    queue = pd.read_csv(queue_path, sep="\t", dtype=str, keep_default_na=False)
    queue.loc[0, "source_condition_text"] = "Broad Breast Cancer"
    queue.to_csv(queue_path, sep="\t", index=False, lineterminator="\n")
    _rewrite_manifest_for_changed_output(output, "curation_queue.tsv")

    with pytest.raises(ClinicalTrialsGovIntakeError):
        verify_clinicaltrials_gov_snapshot(output)


def test_verifier_rejects_a_tampered_response_url_in_manifest(tmp_path: Path) -> None:
    requester, _, _, _ = _two_page_requester()
    output = tmp_path / "snapshot"
    _fetch(output, requester)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original = manifest["pages"][0]["response_url"]
    separator = "&" if "?" in original else "?"
    manifest["pages"][0]["response_url"] = f"{original}{separator}injected=true"
    manifest_path.write_bytes(_canonical_manifest_bytes(manifest))

    with pytest.raises(ClinicalTrialsGovIntakeError):
        verify_clinicaltrials_gov_snapshot(output)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("query_sha256", "query digest is inconsistent"),
        ("page_sha256", "page checksum/size differs"),
        ("page_byte_size", "page checksum/size differs"),
        ("source_endpoint", "source field is not canonical"),
        ("output_role", "output roles are not canonical"),
        ("page_count", "page_count"),
        ("attempt_count", "attempts exceed the declared retry limit"),
        ("total_derived_bytes", "total derived byte count is inconsistent"),
        ("total_raw_response_bytes", "total raw byte count is inconsistent"),
    ],
)
def test_verifier_rejects_semantically_tampered_manifest_fields(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    requester, _, _, _ = _two_page_requester()
    output = tmp_path / case
    _fetch(output, requester)

    def mutate(manifest: dict[str, object]) -> None:
        pages = manifest["pages"]
        outputs = manifest["outputs"]
        source = manifest["source"]
        retrieval = manifest["retrieval"]
        assert isinstance(pages, list) and isinstance(pages[0], dict)
        assert isinstance(outputs, list) and isinstance(outputs[0], dict)
        assert isinstance(source, dict)
        assert isinstance(retrieval, dict)
        if case == "query_sha256":
            manifest["query_sha256"] = "0" * 64
        elif case == "page_sha256":
            pages[0]["sha256"] = "0" * 64
        elif case == "page_byte_size":
            pages[0]["byte_size"] = int(pages[0]["byte_size"]) + 1
        elif case == "source_endpoint":
            source["studies_endpoint"] = "https://clinicaltrials.gov/api/v2/not-studies"
        elif case == "output_role":
            outputs[0]["role"] = "noncanonical_version_role"
        elif case == "page_count":
            manifest["page_count"] = int(manifest["page_count"]) + 1
        elif case == "attempt_count":
            pages[0]["attempt_count"] = int(retrieval["max_attempts"]) + 1
        elif case == "total_derived_bytes":
            retrieval["total_derived_bytes"] = int(retrieval["total_derived_bytes"]) + 1
        elif case == "total_raw_response_bytes":
            retrieval["total_raw_response_bytes"] = (
                int(retrieval["total_raw_response_bytes"]) + 1
            )
        else:  # pragma: no cover - parametrization misuse protection
            raise AssertionError(case)

    _mutate_manifest(output, mutate)

    with pytest.raises(ClinicalTrialsGovIntakeError, match=message):
        verify_clinicaltrials_gov_snapshot(output)


@pytest.mark.parametrize(
    "case",
    [
        "complete_integer_one",
        "schema_version_boolean_true",
        "count_total_integer_one",
        "mutable_registry_integer_one",
        "source_synthetic_marker_integer_zero",
        "boundary_synthetic_marker_integer_zero",
        "query_match_integer_zero",
        "queue_eligibility_integer_zero",
        "gene_ranking_integer_zero",
        "validation_label_integer_zero",
        "registry_efficacy_integer_zero",
        "results_endpoint_integer_zero",
    ],
)
def test_manifest_boolean_attestations_reject_json_integer_lookalikes(
    tmp_path: Path,
    case: str,
) -> None:
    requester, _, _, _ = _two_page_requester()
    output = tmp_path / case
    _fetch(output, requester)

    def mutate(manifest: dict[str, object]) -> None:
        source = manifest["source"]
        query = manifest["query"]
        boundary = manifest["scientific_boundary"]
        assert isinstance(source, dict)
        assert isinstance(query, dict)
        assert isinstance(boundary, dict)
        if case == "complete_integer_one":
            manifest["complete"] = 1
        elif case == "schema_version_boolean_true":
            manifest["bundle_schema_version"] = True
        elif case == "count_total_integer_one":
            query["count_total"] = 1
        elif case == "mutable_registry_integer_one":
            source["mutable_registry"] = 1
        elif case == "source_synthetic_marker_integer_zero":
            source["synthetic_fixture"] = 0
        elif case == "boundary_synthetic_marker_integer_zero":
            boundary["synthetic_fixture"] = 0
        elif case == "query_match_integer_zero":
            boundary["query_match_is_exact_concept_mapping"] = 0
        elif case == "queue_eligibility_integer_zero":
            boundary["curation_queue_eligible_for_clinical_context"] = 0
        elif case == "gene_ranking_integer_zero":
            boundary["used_for_gene_ranking"] = 0
        elif case == "validation_label_integer_zero":
            boundary["used_for_validation_label"] = 0
        elif case == "registry_efficacy_integer_zero":
            boundary["registry_presence_is_efficacy"] = 0
        elif case == "results_endpoint_integer_zero":
            boundary["results_posted_is_endpoint_met"] = 0
        else:  # pragma: no cover - parametrization misuse protection
            raise AssertionError(case)

    _mutate_manifest(output, mutate)

    with pytest.raises(
        ClinicalTrialsGovIntakeError,
        match="manifest contract validation failed",
    ):
        verify_clinicaltrials_gov_snapshot(output)


def test_verifier_binds_manifest_to_the_recorded_package_version(
    tmp_path: Path,
) -> None:
    requester, _, _, _ = _two_page_requester()
    output = tmp_path / "package-version"
    _fetch(output, requester)

    def mutate(manifest: dict[str, object]) -> None:
        software = manifest["software"]
        assert isinstance(software, dict)
        software["package_version"] = "fabricated-build"

    _mutate_manifest(output, mutate)

    with pytest.raises(ClinicalTrialsGovIntakeError, match="recorded package version"):
        verify_clinicaltrials_gov_snapshot(output)


def test_manifest_wall_clock_duration_cannot_exceed_elapsed_limit(
    tmp_path: Path,
) -> None:
    requester, _, _, _ = _two_page_requester()
    output = tmp_path / "elapsed-duration"
    _fetch(output, requester)

    def mutate(manifest: dict[str, object]) -> None:
        retrieval = manifest["retrieval"]
        assert isinstance(retrieval, dict)
        retrieval["completed_at_utc"] = "2026-09-01T14:00:00Z"
        retrieval["max_elapsed_seconds"] = 1.0

    _mutate_manifest(output, mutate)

    with pytest.raises(
        ClinicalTrialsGovIntakeError,
        match="manifest contract validation failed",
    ):
        verify_clinicaltrials_gov_snapshot(output)


def test_query_parser_preserves_opaque_tokens_and_repeated_parameters() -> None:
    url = "https://clinicaltrials.gov/api/v2/studies?a=1&a=2&pageToken=x%2B%2F%3D"

    assert request_url_query(url) == {
        "a": ["1", "2"],
        "pageToken": ["x+/="],
    }
    assert parse_qs(urlsplit(url).query) == request_url_query(url)


def test_field_projection_is_locked_for_later_conservative_curation() -> None:
    assert STUDY_FIELDS == (
        "NCTId",
        "StudyType",
        "Phase",
        "Condition",
        "Keyword",
        "InterventionName",
        "InterventionType",
        "InterventionOtherName",
        "InterventionDescription",
        "InterventionArmGroupLabel",
        "ArmGroupLabel",
        "ArmGroupType",
        "ArmGroupDescription",
        "ArmGroupInterventionName",
        "OverallStatus",
        "WhyStopped",
        "StudyFirstPostDate",
        "StudyFirstPostDateType",
        "ResultsFirstPostDate",
        "ResultsFirstPostDateType",
        "LastUpdatePostDate",
        "LastUpdatePostDateType",
        "VersionHolder",
        "HasResults",
        "BriefTitle",
        "OfficialTitle",
        "BriefSummary",
        "DetailedDescription",
        "EligibilityCriteria",
        "Sex",
        "MinimumAge",
        "MaximumAge",
        "HealthyVolunteers",
        "DesignAllocation",
        "DesignInterventionModel",
        "DesignPrimaryPurpose",
        "EnrollmentCount",
        "EnrollmentType",
    )
