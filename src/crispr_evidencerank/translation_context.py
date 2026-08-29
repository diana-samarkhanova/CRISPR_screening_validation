"""Report-only treatment, clinical-trial, and curated evidence context.

This module deliberately does not compute a gene-prioritization score. A
ClinicalTrials.gov record describes a treatment/disease landscape, while
gene-level patient and preclinical claims require their own curated contracts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
import weakref
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .contracts import (
    ClinicalTrialContextRecord,
    InteractionInferenceStatus,
    InterventionModality,
    PatientAssociationInterpretation,
    PatientMolecularEvidenceRecord,
    PerturbationModality,
    PerturbedCompartment,
    PhenotypeDirection,
    PreclinicalClaimType,
    PreclinicalDirectionInferenceStatus,
    PreclinicalEvidenceRecord,
    PreclinicalModelType,
    ScreenEndpointCategory,
    TreatmentDiseaseContextRecord,
    TrialBiomarkerMatch,
    TrialDiseaseMatch,
    TrialInterventionMatch,
    TrialRegimenRelation,
    validate_records,
)

CLINICALTRIALS_API_ENDPOINT = "https://clinicaltrials.gov/api/v2/studies"
CLINICALTRIALS_VERSION_ENDPOINT = "https://clinicaltrials.gov/api/v2/version"
CLINICALTRIALS_API_MAJOR = "v2"
CLINICALTRIALS_FIELDS = ",".join(
    (
        "IdentificationModule",
        "StatusModule",
        "ConditionsModule",
        "DesignModule",
        "ArmsInterventionsModule",
        "OutcomesModule",
        "ReferencesModule",
        "IPDSharingStatementModule",
        "ResultsSection",
        "HasResults",
    )
)
TRANSLATION_CONTEXT_METHOD_VERSION = "translation_context_v1"


class TranslationContextError(ValueError):
    """Raised when a source snapshot cannot support a reliable report."""


def _canonical_page_sha256(page: dict[str, Any]) -> str:
    """Hash one parsed page using an explicit canonical JSON representation."""

    payload = json.dumps(
        page,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_deep_copy(value: Any) -> Any:
    """Detach nested mutable source objects at every snapshot trust boundary."""

    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TranslationContextError(
            "snapshot must contain finite JSON values"
        ) from exc


def _validated_version_audit(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TranslationContextError(f"{field} must be an object")
    if value.get("endpoint") != CLINICALTRIALS_VERSION_ENDPOINT:
        raise TranslationContextError(
            f"{field}.endpoint must be the official ClinicalTrials.gov "
            "/version endpoint"
        )
    before = value.get("before")
    after = value.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise TranslationContextError(f"{field} requires before/after payloads")
    before_version = _api_version_fields(before)
    after_version = _api_version_fields(after)
    if before_version != after_version:
        raise TranslationContextError(f"{field} before/after versions are not stable")
    before_hash = _canonical_page_sha256(before)
    after_hash = _canonical_page_sha256(after)
    if (
        value.get("before_canonical_sha256") != before_hash
        or value.get("after_canonical_sha256") != after_hash
    ):
        raise TranslationContextError(f"{field} version payload checksum mismatch")
    return {
        "endpoint": CLINICALTRIALS_VERSION_ENDPOINT,
        "before": _json_deep_copy(before),
        "after": _json_deep_copy(after),
        "before_canonical_sha256": before_hash,
        "after_canonical_sha256": after_hash,
    }


def _live_acquisition_capabilities():
    class Witness:
        __slots__ = ("__weakref__",)

    registry: weakref.WeakKeyDictionary[object, tuple[str, str]] = (
        weakref.WeakKeyDictionary()
    )

    def issue(*, source_mode: str, document_sha256: str) -> object:
        witness = Witness()
        registry[witness] = (source_mode, document_sha256)
        return witness

    def validates(
        witness: object | None, *, source_mode: str, document_sha256: str
    ) -> bool:
        if witness is None:
            return False
        try:
            registered = registry.get(witness)
        except TypeError:
            return False
        return registered == (source_mode, document_sha256)

    return issue, validates


_issue_live_acquisition_witness, _valid_live_acquisition_witness = (
    _live_acquisition_capabilities()
)


@dataclass(frozen=True)
class ClinicalTrialsSnapshot:
    document: dict[str, Any]
    studies: list[dict[str, Any]]
    request_urls: list[str]
    total_count: int | None
    complete: bool
    retrieved_at_utc: datetime
    source_mode: str
    api_version: str
    data_timestamp: str | None
    version_stable: bool
    _live_acquisition_witness: object | None = None


@dataclass
class TranslationContextResult:
    clinical_trials: pd.DataFrame
    preclinical_used_evidence: pd.DataFrame
    preclinical_exclusions: pd.DataFrame
    patient_used_evidence: pd.DataFrame
    patient_exclusions: pd.DataFrame
    candidate_context: pd.DataFrame
    missingness: pd.DataFrame
    report_markdown: str
    metadata: dict[str, Any]
    clinicaltrials_snapshot: dict[str, Any]


def _utc_datetime(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    parsed = (
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        if isinstance(value, str)
        else value
    )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TranslationContextError("retrieved_at_utc must include a UTC offset")
    if parsed.utcoffset().total_seconds() != 0:
        raise TranslationContextError("retrieved_at_utc must use the UTC offset")
    return parsed.astimezone(UTC)


def _source_utc_timestamp(value: Any) -> str:
    """Validate a source timestamp while preserving its reported spelling."""

    if not isinstance(value, str) or not value.strip():
        raise TranslationContextError(
            "ClinicalTrials.gov data_timestamp must be a non-empty UTC timestamp"
        )
    reported = value.strip()
    normalized = f"{reported[:-4]}+00:00" if reported.endswith(" UTC") else reported
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TranslationContextError(
            "ClinicalTrials.gov data_timestamp must be a valid UTC timestamp"
        ) from exc
    offset = parsed.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise TranslationContextError(
            "ClinicalTrials.gov data_timestamp must use the UTC offset"
        )
    return reported


def _source_timestamp_datetime(value: Any) -> datetime:
    reported = _source_utc_timestamp(value)
    normalized = f"{reported[:-4]}+00:00" if reported.endswith(" UTC") else reported
    return datetime.fromisoformat(normalized.replace("Z", "+00:00")).astimezone(UTC)


def _strict_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TranslationContextError(f"{field} must be a non-negative integer")
    return value


def _study_nct_id(study: Any, field: str) -> str:
    if not isinstance(study, dict):
        raise TranslationContextError(f"{field} studies must be JSON objects")
    protocol = study.get("protocolSection")
    if not isinstance(protocol, dict):
        raise TranslationContextError(f"{field} study requires protocolSection")
    identification = protocol.get("identificationModule")
    if not isinstance(identification, dict):
        raise TranslationContextError(f"{field} study requires identificationModule")
    nct_id = str(identification.get("nctId", "")).strip()
    if not re.fullmatch(r"NCT[0-9]{8}", nct_id):
        raise TranslationContextError(f"{field} study requires a valid NCT ID")
    return nct_id


def _validate_official_studies_url(
    url: Any,
    *,
    field: str,
    treatment_query: str | None = None,
    condition_query: str | None = None,
    expected_page_token: str | None = None,
    observed_study_count: int | None = None,
) -> str:
    if not isinstance(url, str) or not url.strip():
        raise TranslationContextError(f"{field} must contain non-empty URLs")
    value = url.strip()
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise TranslationContextError(f"{field} contains an invalid URL") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != "clinicaltrials.gov"
        or port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/api/v2/studies"
        or parsed.fragment
    ):
        raise TranslationContextError(
            f"{field} must use the official ClinicalTrials.gov v2 studies endpoint"
        )
    parameters = parse_qs(parsed.query, keep_blank_values=True)
    if treatment_query is not None or condition_query is not None:
        allowed_parameters = {
            "format",
            "markupFormat",
            "fields",
            "query.cond",
            "query.intr",
            "pageSize",
            "countTotal",
            "pageToken",
        }
        unexpected = sorted(set(parameters) - allowed_parameters)
        if unexpected:
            raise TranslationContextError(
                f"{field} contains unsupported or narrowing query parameters: "
                f"{unexpected}"
            )
        if any(len(values) != 1 for values in parameters.values()):
            raise TranslationContextError(
                f"{field} query parameters must occur exactly once"
            )
        expected_values = {
            "format": "json",
            "markupFormat": "markdown",
            "fields": CLINICALTRIALS_FIELDS,
            "query.intr": treatment_query,
            "query.cond": condition_query,
        }
        for name, expected in expected_values.items():
            if expected is None or parameters.get(name) != [expected]:
                raise TranslationContextError(
                    f"{field} {name} disagrees with the declared complete query"
                )
        raw_page_size = parameters.get("pageSize")
        if raw_page_size is None:
            raise TranslationContextError(f"{field} requires pageSize")
        try:
            page_size = int(raw_page_size[0])
        except (TypeError, ValueError) as exc:
            raise TranslationContextError(
                f"{field} pageSize must be an integer"
            ) from exc
        if not 1 <= page_size <= 1000 or str(page_size) != raw_page_size[0]:
            raise TranslationContextError(
                f"{field} pageSize must be a canonical integer from 1 to 1000"
            )
        if observed_study_count is not None and observed_study_count > page_size:
            raise TranslationContextError(
                f"{field} source page contains more studies than requested pageSize"
            )
        if expected_page_token is None:
            if parameters.get("countTotal") != ["true"] or "pageToken" in parameters:
                raise TranslationContextError(
                    f"{field} first page requires countTotal=true and no pageToken"
                )
        elif parameters.get("pageToken") != [expected_page_token] or (
            "countTotal" in parameters
        ):
            raise TranslationContextError(
                f"{field} continuation page token disagrees with source pages"
            )
    return value


def _validate_query_pages(
    pages: Any,
    *,
    field: str,
) -> tuple[list[dict[str, Any]], int, bool]:
    """Recompute studies, registry total, and completeness from raw pages."""

    if not isinstance(pages, list) or not pages:
        raise TranslationContextError(f"{field} must be a non-empty page list")
    studies: list[dict[str, Any]] = []
    reported_total: int | None = None
    seen_tokens: set[str] = set()
    seen_nct_ids: set[str] = set()
    for page_index, page in enumerate(pages):
        page_field = f"{field}[{page_index}]"
        if not isinstance(page, dict) or not isinstance(page.get("studies"), list):
            raise TranslationContextError(f"{page_field} requires studies[]")
        page_studies = page["studies"]
        for study in page_studies:
            nct_id = _study_nct_id(study, page_field)
            if nct_id in seen_nct_ids:
                raise TranslationContextError(f"{field} contains duplicate NCT records")
            seen_nct_ids.add(nct_id)
        studies.extend(page_studies)
        if page_index == 0:
            if "totalCount" not in page:
                raise TranslationContextError(f"{page_field} requires totalCount")
            reported_total = _strict_nonnegative_int(
                page["totalCount"], f"{page_field}.totalCount"
            )
        elif "totalCount" in page:
            page_total = _strict_nonnegative_int(
                page["totalCount"], f"{page_field}.totalCount"
            )
            if page_total != reported_total:
                raise TranslationContextError(
                    f"{field} totalCount changed between pages"
                )
        token = page.get("nextPageToken")
        if token is not None and (not isinstance(token, str) or not token.strip()):
            raise TranslationContextError(
                f"{page_field}.nextPageToken must be a non-empty string"
            )
        if token:
            if token in seen_tokens:
                raise TranslationContextError(f"{field} repeats a page token")
            seen_tokens.add(token)
            if not page_studies:
                raise TranslationContextError(f"{field} has an empty continuation page")
        if page_index < len(pages) - 1 and not token:
            raise TranslationContextError(
                f"{field} has pages after pagination was complete"
            )
    assert reported_total is not None
    complete = pages[-1].get("nextPageToken") is None
    if complete and len(studies) != reported_total:
        raise TranslationContextError(
            f"{field} complete pagination disagrees with totalCount"
        )
    if not complete and len(studies) >= reported_total:
        raise TranslationContextError(
            f"{field} truncated pagination is inconsistent with totalCount"
        )
    return studies, reported_total, complete


def _response_json(response: Any, *, require_studies: bool) -> dict[str, Any]:
    payload = response.read()
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TranslationContextError(
            "ClinicalTrials.gov returned invalid UTF-8 JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise TranslationContextError("ClinicalTrials.gov response must be an object")
    if require_studies and (
        "studies" not in decoded or not isinstance(decoded["studies"], list)
    ):
        raise TranslationContextError(
            "ClinicalTrials.gov response requires a studies array"
        )
    return decoded


def _fetch_json_object(
    url: str,
    *,
    timeout_seconds: float,
    opener: Any,
    require_studies: bool,
) -> dict[str, Any]:
    request = Request(
        url,
        headers={"User-Agent": "crispr-evidencerank/translation-context"},
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            return _response_json(response, require_studies=require_studies)
    except TranslationContextError:
        raise
    except Exception as exc:
        raise TranslationContextError(
            "ClinicalTrials.gov retrieval failed; absence of trials cannot be inferred"
        ) from exc


def _api_version_fields(payload: dict[str, Any]) -> tuple[str, str]:
    api_version = str(payload.get("apiVersion", "")).strip()
    if not api_version or payload.get("dataTimestamp") is None:
        raise TranslationContextError(
            "ClinicalTrials.gov /version omitted apiVersion or dataTimestamp"
        )
    if not re.match(r"^2(?:\.|$)", api_version):
        raise TranslationContextError(
            "ClinicalTrials.gov /version apiVersion must have major version 2"
        )
    data_timestamp = _source_utc_timestamp(payload["dataTimestamp"])
    return api_version, data_timestamp


def fetch_clinical_trials_v2(
    treatment: str,
    condition: str,
    *,
    page_size: int = 100,
    max_studies: int = 500,
    timeout_seconds: float = 30.0,
    retrieved_at_utc: datetime | None = None,
    opener: Any | None = None,
) -> ClinicalTrialsSnapshot:
    """Retrieve a bounded, pagination-audited current API snapshot."""

    if not treatment.strip() or not condition.strip():
        raise TranslationContextError("treatment and condition cannot be empty")
    if not 1 <= page_size <= 1000:
        raise TranslationContextError("page_size must be between 1 and 1000")
    if max_studies < 1:
        raise TranslationContextError("max_studies must be positive")
    if timeout_seconds <= 0:
        raise TranslationContextError("timeout_seconds must be positive")
    acquisition_verified = opener is None and retrieved_at_utc is None
    active_opener = urlopen if opener is None else opener

    studies: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    request_urls: list[str] = []
    next_page_token: str | None = None
    total_count: int | None = None
    seen_page_tokens: set[str] = set()
    version_before = _fetch_json_object(
        CLINICALTRIALS_VERSION_ENDPOINT,
        timeout_seconds=timeout_seconds,
        opener=active_opener,
        require_studies=False,
    )
    api_version_before, data_timestamp_before = _api_version_fields(version_before)

    while len(studies) < max_studies:
        parameters: dict[str, str | int] = {
            "format": "json",
            "markupFormat": "markdown",
            "fields": CLINICALTRIALS_FIELDS,
            "query.cond": condition.strip(),
            "query.intr": treatment.strip(),
            "pageSize": min(page_size, max_studies - len(studies)),
        }
        if next_page_token:
            parameters["pageToken"] = next_page_token
        else:
            parameters["countTotal"] = "true"
        url = f"{CLINICALTRIALS_API_ENDPOINT}?{urlencode(parameters)}"
        request_urls.append(url)
        page = _fetch_json_object(
            url,
            timeout_seconds=timeout_seconds,
            opener=active_opener,
            require_studies=True,
        )
        page_studies = page.get("studies", [])
        requested_page_size = int(parameters["pageSize"])
        if len(page_studies) > requested_page_size:
            raise TranslationContextError(
                "ClinicalTrials.gov returned more studies than requested"
            )
        pages.append(page)
        studies.extend(page_studies)
        if total_count is None:
            if "totalCount" not in page:
                raise TranslationContextError(
                    "the first ClinicalTrials.gov page requires integer totalCount"
                )
            total_count = _strict_nonnegative_int(page["totalCount"], "totalCount")
        elif page.get("totalCount") is not None:
            page_total_count = _strict_nonnegative_int(page["totalCount"], "totalCount")
            if page_total_count != total_count:
                raise TranslationContextError(
                    "ClinicalTrials.gov totalCount changed during pagination"
                )
        token = page.get("nextPageToken")
        if token is not None and (not isinstance(token, str) or not token.strip()):
            raise TranslationContextError("nextPageToken must be a non-empty string")
        if token and token in seen_page_tokens:
            raise TranslationContextError("ClinicalTrials.gov repeated a page token")
        if token:
            seen_page_tokens.add(token)
        if token and not page_studies:
            raise TranslationContextError(
                "ClinicalTrials.gov returned an empty page with a continuation token"
            )
        next_page_token = token
        if not next_page_token:
            break

    complete = next_page_token is None
    if complete and total_count is not None and len(studies) != total_count:
        raise TranslationContextError(
            "complete ClinicalTrials.gov pagination disagrees with totalCount"
        )
    version_after = _fetch_json_object(
        CLINICALTRIALS_VERSION_ENDPOINT,
        timeout_seconds=timeout_seconds,
        opener=active_opener,
        require_studies=False,
    )
    api_version_after, data_timestamp_after = _api_version_fields(version_after)
    if api_version_before != api_version_after:
        raise TranslationContextError(
            "ClinicalTrials.gov API version changed during retrieval"
        )
    if data_timestamp_before != data_timestamp_after:
        raise TranslationContextError(
            "ClinicalTrials.gov dataTimestamp changed during retrieval; mixed "
            "snapshots are not published"
        )
    retrieved = _utc_datetime(retrieved_at_utc)
    if _source_timestamp_datetime(data_timestamp_before) > retrieved:
        raise TranslationContextError(
            "ClinicalTrials.gov data_timestamp cannot be later than retrieved_at_utc"
        )
    version_audit = {
        "endpoint": CLINICALTRIALS_VERSION_ENDPOINT,
        "before": version_before,
        "after": version_after,
        "before_canonical_sha256": _canonical_page_sha256(version_before),
        "after_canonical_sha256": _canonical_page_sha256(version_after),
    }
    document = {
        "source": "ClinicalTrials.gov",
        "source_api_major": CLINICALTRIALS_API_MAJOR,
        "api_version": api_version_before,
        "data_timestamp": data_timestamp_before,
        "version_stable": acquisition_verified,
        "version_audit": version_audit,
        "retrieved_at_utc": retrieved.isoformat(),
        "request_urls": request_urls,
        "total_count": total_count,
        "complete": complete,
        "ontology_concept_recall_complete": False,
        "page_canonical_sha256": [_canonical_page_sha256(page) for page in pages],
        "pages": pages,
    }
    replayed = clinical_trials_snapshot_from_document(document)
    expected_replayed_document = _json_deep_copy(document)
    expected_replayed_document["version_stable"] = False
    if replayed.document != expected_replayed_document:
        raise TranslationContextError(
            "live ClinicalTrials.gov snapshot failed frozen-replay invariance"
        )
    source_mode = (
        "live_api"
        if acquisition_verified
        else ("injected_transport" if opener is not None else "injected_clock")
    )
    live_document = _json_deep_copy(replayed.document)
    live_document["version_stable"] = acquisition_verified
    witness = (
        _issue_live_acquisition_witness(
            source_mode=source_mode,
            document_sha256=_canonical_page_sha256(live_document),
        )
        if acquisition_verified
        else None
    )
    return ClinicalTrialsSnapshot(
        document=live_document,
        studies=replayed.studies,
        request_urls=replayed.request_urls,
        total_count=replayed.total_count,
        complete=replayed.complete,
        retrieved_at_utc=replayed.retrieved_at_utc,
        source_mode=source_mode,
        api_version=api_version_before,
        data_timestamp=data_timestamp_before,
        version_stable=acquisition_verified,
        _live_acquisition_witness=witness,
    )


def fetch_clinical_trials_concept_v2(
    treatment: str,
    condition: str,
    *,
    treatment_entity_aliases: list[str] | tuple[str, ...] = (),
    treatment_class_terms: list[str] | tuple[str, ...] = (),
    cancer_entity_aliases: list[str] | tuple[str, ...] = (),
    cancer_ancestor_terms: list[str] | tuple[str, ...] = (),
    disease_subtype: str | None = None,
    subtype_entity_aliases: list[str] | tuple[str, ...] = (),
    page_size: int = 100,
    max_studies_per_query: int = 500,
    timeout_seconds: float = 30.0,
    retrieved_at_utc: datetime | None = None,
    opener: Any | None = None,
) -> ClinicalTrialsSnapshot:
    """Retrieve typed treatment/condition query lanes and deduplicate NCTs."""

    if subtype_entity_aliases and not disease_subtype:
        raise TranslationContextError(
            "subtype entity aliases require a canonical disease_subtype"
        )

    treatment_queries = _typed_query_terms(
        [
            ("treatment_canonical", [treatment]),
            ("treatment_entity_alias", list(treatment_entity_aliases)),
            ("treatment_class_term", list(treatment_class_terms)),
        ]
    )
    condition_queries = _typed_query_terms(
        [
            ("cancer_canonical", [condition]),
            ("cancer_entity_alias", list(cancer_entity_aliases)),
            ("disease_subtype_canonical", [disease_subtype]),
            ("subtype_entity_alias", list(subtype_entity_aliases)),
            ("cancer_ancestor_term", list(cancer_ancestor_terms)),
        ]
    )
    query_pairs = [
        (treatment_query, treatment_lane, condition_query, condition_lane)
        for treatment_query, treatment_lane in treatment_queries
        for condition_query, condition_lane in condition_queries
    ]
    if not query_pairs:
        raise TranslationContextError("at least one concept query is required")
    if len(query_pairs) > 25:
        raise TranslationContextError(
            "declared typed-query cross-product exceeds the 25-query safety limit"
        )
    snapshots = [
        fetch_clinical_trials_v2(
            treatment_query,
            condition_query,
            page_size=page_size,
            max_studies=max_studies_per_query,
            timeout_seconds=timeout_seconds,
            retrieved_at_utc=retrieved_at_utc,
            opener=opener,
        )
        for treatment_query, _, condition_query, _ in query_pairs
    ]
    api_versions = {snapshot.api_version for snapshot in snapshots}
    data_timestamps = {snapshot.data_timestamp for snapshot in snapshots}
    if len(api_versions) != 1 or len(data_timestamps) != 1:
        raise TranslationContextError(
            "ClinicalTrials.gov version changed across declared concept queries"
        )
    retrieved = max(snapshot.retrieved_at_utc for snapshot in snapshots)
    acquisition_verified = opener is None and retrieved_at_utc is None
    if acquisition_verified and not all(
        snapshot.source_mode == "live_api" and snapshot.version_stable
        for snapshot in snapshots
    ):
        raise TranslationContextError(
            "verified concept acquisition contains an unverified child query"
        )

    studies_by_nct: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        for study in snapshot.studies:
            protocol = _mapping(study.get("protocolSection"), "protocolSection")
            identification = _mapping(
                protocol.get("identificationModule"), "identificationModule"
            )
            nct_id = str(identification.get("nctId", "")).strip()
            if not re.fullmatch(r"NCT[0-9]{8}", nct_id):
                raise TranslationContextError(
                    "concept query returned a study without a valid NCT ID"
                )
            existing = studies_by_nct.get(nct_id)
            if existing is not None and _canonical_page_sha256(existing) != (
                _canonical_page_sha256(study)
            ):
                raise TranslationContextError(
                    f"conflicting payloads for {nct_id} across concept queries"
                )
            studies_by_nct.setdefault(nct_id, study)
    studies = list(studies_by_nct.values())
    merged_page = {"studies": studies, "totalCount": len(studies)}
    request_urls = [url for snapshot in snapshots for url in snapshot.request_urls]
    query_audit = []
    for (
        treatment_query,
        treatment_query_lane,
        condition_query,
        condition_query_lane,
    ), snapshot in zip(query_pairs, snapshots, strict=True):
        query_audit.append(
            {
                "treatment_query": treatment_query,
                "treatment_query_lane": treatment_query_lane,
                "condition_query": condition_query,
                "condition_query_lane": condition_query_lane,
                "reported_total_count": snapshot.total_count,
                "retrieved_record_count": len(snapshot.studies),
                "complete": snapshot.complete,
                "request_urls": snapshot.request_urls,
                "page_canonical_sha256": snapshot.document["page_canonical_sha256"],
                "pages": snapshot.document["pages"],
            }
        )
    complete = all(snapshot.complete for snapshot in snapshots)
    document = {
        "source": "ClinicalTrials.gov",
        "source_api_major": CLINICALTRIALS_API_MAJOR,
        "api_version": snapshots[0].api_version,
        "data_timestamp": snapshots[0].data_timestamp,
        "version_stable": acquisition_verified,
        "version_audit_set": [
            snapshot.document["version_audit"] for snapshot in snapshots
        ],
        "retrieved_at_utc": retrieved.isoformat(),
        "request_urls": request_urls,
        "total_count": len(studies) if complete else None,
        "complete": complete,
        "declared_query_set": query_audit,
        "declared_query_semantics_version": "typed_query_lanes_v1",
        "ontology_concept_recall_complete": False,
        "page_canonical_sha256": [_canonical_page_sha256(merged_page)],
        "pages": [merged_page],
    }
    replayed = clinical_trials_snapshot_from_document(document)
    expected_replayed_document = _json_deep_copy(document)
    expected_replayed_document["version_stable"] = False
    if replayed.document != expected_replayed_document:
        raise TranslationContextError(
            "live ClinicalTrials.gov concept snapshot failed frozen-replay invariance"
        )
    source_mode = (
        "live_api_declared_query_set"
        if acquisition_verified
        else (
            "injected_transport_declared_query_set"
            if opener is not None
            else "injected_clock_declared_query_set"
        )
    )
    live_document = _json_deep_copy(replayed.document)
    live_document["version_stable"] = acquisition_verified
    witness = (
        _issue_live_acquisition_witness(
            source_mode=source_mode,
            document_sha256=_canonical_page_sha256(live_document),
        )
        if acquisition_verified
        else None
    )
    return ClinicalTrialsSnapshot(
        document=live_document,
        studies=replayed.studies,
        request_urls=replayed.request_urls,
        total_count=replayed.total_count,
        complete=replayed.complete,
        retrieved_at_utc=replayed.retrieved_at_utc,
        source_mode=source_mode,
        api_version=snapshots[0].api_version,
        data_timestamp=snapshots[0].data_timestamp,
        version_stable=acquisition_verified,
        _live_acquisition_witness=witness,
    )


def clinical_trials_snapshot_from_document(
    document: dict[str, Any],
    *,
    retrieved_at_utc: datetime | str | None = None,
) -> ClinicalTrialsSnapshot:
    """Load either a raw API page or this package's multi-page snapshot."""

    if not isinstance(document, dict):
        raise TranslationContextError("clinical-trial snapshot must be a JSON object")
    document = _json_deep_copy(document)
    wrapped = "pages" in document
    validated_version_audit: dict[str, Any] | None = None
    validated_version_audit_set: list[dict[str, Any]] | None = None
    if wrapped:
        if document.get("source") != "ClinicalTrials.gov":
            raise TranslationContextError(
                "frozen wrapper source must be exactly ClinicalTrials.gov"
            )
        if document.get("source_api_major") != CLINICALTRIALS_API_MAJOR:
            raise TranslationContextError("frozen wrapper source_api_major must be v2")
        raw_api_version = document.get("api_version")
        if not isinstance(raw_api_version, str):
            raise TranslationContextError(
                "frozen wrapper api_version must be a version string"
            )
        api_version = raw_api_version.strip()
        if not re.match(r"^2(?:\.|$)", api_version):
            raise TranslationContextError(
                "frozen wrapper api_version must have major version 2"
            )
        data_timestamp = _source_utc_timestamp(document.get("data_timestamp"))
        if "version_audit" in document and "version_audit_set" in document:
            raise TranslationContextError(
                "snapshot cannot contain both version_audit and version_audit_set"
            )
        if "version_audit" in document:
            validated_version_audit = _validated_version_audit(
                document["version_audit"], field="version_audit"
            )
            if _api_version_fields(validated_version_audit["before"]) != (
                api_version,
                data_timestamp,
            ):
                raise TranslationContextError(
                    "version_audit disagrees with top-level API version"
                )
        if "version_audit_set" in document:
            raw_audits = document["version_audit_set"]
            if not isinstance(raw_audits, list) or not raw_audits:
                raise TranslationContextError(
                    "version_audit_set must be a non-empty list"
                )
            validated_version_audit_set = [
                _validated_version_audit(value, field=f"version_audit_set[{index}]")
                for index, value in enumerate(raw_audits)
            ]
            if any(
                _api_version_fields(audit["before"]) != (api_version, data_timestamp)
                for audit in validated_version_audit_set
            ):
                raise TranslationContextError(
                    "version_audit_set disagrees with top-level API version"
                )
        concept_recall = document.get("ontology_concept_recall_complete", False)
        if not isinstance(concept_recall, bool):
            raise TranslationContextError(
                "ontology_concept_recall_complete must be a boolean"
            )
        if concept_recall:
            raise TranslationContextError(
                "ontology_concept_recall_complete cannot be true for text queries"
            )
        pages = document["pages"]
        if not isinstance(pages, list) or not pages:
            raise TranslationContextError("snapshot pages must be a non-empty list")
        studies: list[dict[str, Any]] = []
        for page in pages:
            if (
                not isinstance(page, dict)
                or "studies" not in page
                or not isinstance(page["studies"], list)
            ):
                raise TranslationContextError("every snapshot page requires studies[]")
            studies.extend(page.get("studies", []))
        request_urls = document.get("request_urls", [])
        if not isinstance(request_urls, list) or not all(
            isinstance(value, str) for value in request_urls
        ):
            raise TranslationContextError("request_urls must be a JSON string list")
        request_urls = [
            _validate_official_studies_url(value, field="request_urls")
            for value in request_urls
        ]
        raw_total = document.get("total_count")
        raw_complete = document.get("complete", False)
        if not isinstance(raw_complete, bool):
            raise TranslationContextError("snapshot complete must be a boolean")
        complete = raw_complete
        if complete and raw_total is None:
            raise TranslationContextError(
                "a complete frozen snapshot requires total_count"
            )
        document_retrieved_at = document.get("retrieved_at_utc")
        if retrieved_at_utc is not None and document_retrieved_at is not None:
            if _utc_datetime(retrieved_at_utc) != _utc_datetime(document_retrieved_at):
                raise TranslationContextError(
                    "retrieved_at_utc cannot override a wrapped snapshot timestamp"
                )
        retrieved_value = document_retrieved_at or retrieved_at_utc
        raw_version_stable = document.get("version_stable", False)
        if not isinstance(raw_version_stable, bool):
            raise TranslationContextError("snapshot version_stable must be a boolean")
    else:
        if "studies" not in document or not isinstance(document["studies"], list):
            raise TranslationContextError("raw API snapshot requires studies[]")
        pages = [document]
        studies = list(document.get("studies", []))
        request_urls = []
        raw_total = document.get("totalCount")
        next_page_token = document.get("nextPageToken")
        if next_page_token is not None and (
            not isinstance(next_page_token, str) or not next_page_token.strip()
        ):
            raise TranslationContextError(
                "raw snapshot nextPageToken must be a non-empty string"
            )
        complete = next_page_token is None and raw_total is not None
        retrieved_value = retrieved_at_utc
        api_version = "unverified"
        data_timestamp = None
    if retrieved_value is None:
        raise TranslationContextError(
            "a frozen ClinicalTrials.gov JSON input requires retrieved_at_utc"
        )
    total_count = (
        None
        if raw_total is None
        else _strict_nonnegative_int(raw_total, "snapshot total count")
    )
    retrieved = _utc_datetime(retrieved_value)
    if data_timestamp is not None and _source_timestamp_datetime(data_timestamp) > (
        retrieved
    ):
        raise TranslationContextError(
            "ClinicalTrials.gov data_timestamp cannot be later than retrieved_at_utc"
        )
    if complete and total_count is not None and len(studies) != total_count:
        raise TranslationContextError(
            "complete frozen pagination disagrees with total count"
        )
    page_hashes = [_canonical_page_sha256(page) for page in pages]
    declared_page_hashes = document.get("page_canonical_sha256")
    if declared_page_hashes is not None:
        if not isinstance(declared_page_hashes, list) or not all(
            isinstance(value, str) for value in declared_page_hashes
        ):
            raise TranslationContextError(
                "page_canonical_sha256 must be a JSON string list"
            )
        if declared_page_hashes != page_hashes:
            raise TranslationContextError(
                "frozen ClinicalTrials.gov page checksum mismatch"
            )
    canonical = {
        "source": "ClinicalTrials.gov",
        "source_api_major": CLINICALTRIALS_API_MAJOR,
        "api_version": api_version,
        "data_timestamp": data_timestamp,
        # Serialized input is self-attested at load time. Live callers restore
        # this to true only while carrying the non-serializable witness.
        "version_stable": False,
        "retrieved_at_utc": retrieved.isoformat(),
        "request_urls": request_urls,
        "total_count": total_count,
        "complete": complete,
        "page_canonical_sha256": page_hashes,
        "pages": pages,
    }
    if wrapped:
        canonical["ontology_concept_recall_complete"] = False
        if validated_version_audit is not None:
            canonical["version_audit"] = validated_version_audit
        if validated_version_audit_set is not None:
            canonical["version_audit_set"] = validated_version_audit_set
    if "declared_query_set" in document:
        declared_query_set = document["declared_query_set"]
        if not isinstance(declared_query_set, list) or not declared_query_set:
            raise TranslationContextError(
                "declared_query_set must be a non-empty JSON list"
            )
        query_semantics_version = document.get("declared_query_semantics_version")
        if query_semantics_version not in {None, "typed_query_lanes_v1"}:
            raise TranslationContextError(
                "unsupported declared_query_semantics_version"
            )
        has_typed_lane_fields = any(
            isinstance(query, dict)
            and ("treatment_query_lane" in query or "condition_query_lane" in query)
            for query in declared_query_set
        )
        if has_typed_lane_fields and query_semantics_version is None:
            raise TranslationContextError(
                "typed query lane fields require declared_query_semantics_version"
            )
        if query_semantics_version == "typed_query_lanes_v1" and (
            validated_version_audit_set is None
            or len(validated_version_audit_set) != len(declared_query_set)
        ):
            raise TranslationContextError(
                "typed declared_query_set requires exactly one version audit per query"
            )
        treatment_query_lanes = {
            "treatment_canonical",
            "treatment_entity_alias",
            "treatment_class_term",
        }
        condition_query_lanes = {
            "cancer_canonical",
            "cancer_entity_alias",
            "disease_subtype_canonical",
            "subtype_entity_alias",
            "cancer_ancestor_term",
        }
        validated_queries: list[dict[str, Any]] = []
        flattened_urls: list[str] = []
        union_by_nct: dict[str, dict[str, Any]] = {}
        seen_query_pairs: set[tuple[str, str]] = set()
        typed_treatment_terms: dict[str, str] = {}
        typed_condition_terms: dict[str, str] = {}
        for query_index, query in enumerate(declared_query_set):
            if not isinstance(query, dict):
                raise TranslationContextError(
                    "each declared_query_set entry must be an object"
                )
            query_field = f"declared_query_set[{query_index}]"
            treatment_query = query.get("treatment_query")
            condition_query = query.get("condition_query")
            if not isinstance(treatment_query, str) or not treatment_query.strip():
                raise TranslationContextError(
                    f"{query_field}.treatment_query must be non-empty"
                )
            if not isinstance(condition_query, str) or not condition_query.strip():
                raise TranslationContextError(
                    f"{query_field}.condition_query must be non-empty"
                )
            treatment_query = treatment_query.strip()
            condition_query = condition_query.strip()
            treatment_query_lane = query.get("treatment_query_lane")
            condition_query_lane = query.get("condition_query_lane")
            if query_semantics_version == "typed_query_lanes_v1":
                if treatment_query_lane not in treatment_query_lanes:
                    raise TranslationContextError(
                        f"{query_field}.treatment_query_lane is invalid"
                    )
                if condition_query_lane not in condition_query_lanes:
                    raise TranslationContextError(
                        f"{query_field}.condition_query_lane is invalid"
                    )
                for term, lane, observed in (
                    (
                        treatment_query,
                        treatment_query_lane,
                        typed_treatment_terms,
                    ),
                    (condition_query, condition_query_lane, typed_condition_terms),
                ):
                    normalized_term = _normalize_entity_identity(term)
                    previous_lane = observed.get(normalized_term)
                    if previous_lane is not None and previous_lane != lane:
                        raise TranslationContextError(
                            "the same normalized term cannot occur in multiple "
                            "declared query lanes"
                        )
                    observed.setdefault(normalized_term, lane)
            query_pair = (
                _normalize_entity_identity(treatment_query),
                _normalize_entity_identity(condition_query),
            )
            if query_pair in seen_query_pairs:
                raise TranslationContextError(
                    "declared_query_set contains a duplicate query pair"
                )
            seen_query_pairs.add(query_pair)
            query_pages = query.get("pages")
            query_hashes = query.get("page_canonical_sha256")
            if not isinstance(query_hashes, list) or not all(
                isinstance(value, str) for value in query_hashes
            ):
                raise TranslationContextError(
                    f"{query_field}.page_canonical_sha256 must be a string list"
                )
            query_studies, query_total, query_complete = _validate_query_pages(
                query_pages,
                field=f"{query_field}.pages",
            )
            observed_hashes = [_canonical_page_sha256(page) for page in query_pages]
            if observed_hashes != query_hashes:
                raise TranslationContextError(
                    f"declared query page checksum mismatch at index {query_index}"
                )
            declared_total = _strict_nonnegative_int(
                query.get("reported_total_count"),
                f"{query_field}.reported_total_count",
            )
            if declared_total != query_total:
                raise TranslationContextError(
                    f"{query_field} reported_total_count disagrees with pages"
                )
            declared_retrieved = _strict_nonnegative_int(
                query.get("retrieved_record_count"),
                f"{query_field}.retrieved_record_count",
            )
            if declared_retrieved != len(query_studies):
                raise TranslationContextError(
                    f"{query_field} retrieved_record_count disagrees with pages"
                )
            declared_complete = query.get("complete")
            if not isinstance(declared_complete, bool):
                raise TranslationContextError(
                    f"{query_field}.complete must be a boolean"
                )
            if declared_complete != query_complete:
                raise TranslationContextError(
                    f"{query_field} completeness disagrees with pages"
                )
            query_urls = query.get("request_urls")
            if not isinstance(query_urls, list) or len(query_urls) != len(query_pages):
                raise TranslationContextError(
                    f"{query_field}.request_urls must have one URL per page"
                )
            validated_urls = []
            for page_index, value in enumerate(query_urls):
                expected_page_token = (
                    None
                    if page_index == 0
                    else str(query_pages[page_index - 1]["nextPageToken"])
                )
                validated_urls.append(
                    _validate_official_studies_url(
                        value,
                        field=f"{query_field}.request_urls",
                        treatment_query=treatment_query,
                        condition_query=condition_query,
                        expected_page_token=expected_page_token,
                        observed_study_count=len(query_pages[page_index]["studies"]),
                    )
                )
            for page_index, url in enumerate(validated_urls):
                parameters = parse_qs(urlsplit(url).query, keep_blank_values=True)
                observed_page_token = parameters.get("pageToken")
                expected_page_token = (
                    None
                    if page_index == 0
                    else [query_pages[page_index - 1]["nextPageToken"]]
                )
                if observed_page_token != expected_page_token:
                    raise TranslationContextError(
                        f"{query_field}.request_urls pageToken chain disagrees "
                        "with pages"
                    )
            flattened_urls.extend(validated_urls)
            query_seen_nct: set[str] = set()
            for study in query_studies:
                nct_id = _study_nct_id(study, query_field)
                if nct_id in query_seen_nct:
                    raise TranslationContextError(
                        f"{query_field} contains duplicate NCT records"
                    )
                query_seen_nct.add(nct_id)
                existing = union_by_nct.get(nct_id)
                if existing is not None and _canonical_page_sha256(existing) != (
                    _canonical_page_sha256(study)
                ):
                    raise TranslationContextError(
                        f"conflicting payloads for {nct_id} across declared queries"
                    )
                union_by_nct.setdefault(nct_id, study)
            validated_queries.append(
                {
                    "treatment_query": treatment_query,
                    "condition_query": condition_query,
                    **(
                        {
                            "treatment_query_lane": treatment_query_lane,
                            "condition_query_lane": condition_query_lane,
                        }
                        if query_semantics_version == "typed_query_lanes_v1"
                        else {}
                    ),
                    "reported_total_count": query_total,
                    "retrieved_record_count": len(query_studies),
                    "complete": query_complete,
                    "request_urls": validated_urls,
                    "page_canonical_sha256": observed_hashes,
                    "pages": query_pages,
                }
            )
        if request_urls != flattened_urls:
            raise TranslationContextError(
                "top-level request_urls disagree with declared_query_set"
            )
        derived_complete = all(query["complete"] for query in validated_queries)
        if complete != derived_complete:
            raise TranslationContextError(
                "top-level completeness disagrees with declared_query_set"
            )
        if derived_complete:
            if total_count != len(union_by_nct):
                raise TranslationContextError(
                    "top-level total_count disagrees with declared NCT union"
                )
        elif total_count is not None:
            raise TranslationContextError(
                "an incomplete declared query set requires null top-level total_count"
            )
        top_by_nct: dict[str, dict[str, Any]] = {}
        for study in studies:
            nct_id = _study_nct_id(study, "top-level pages")
            if nct_id in top_by_nct:
                raise TranslationContextError(
                    "top-level pages contain duplicate NCT records"
                )
            top_by_nct[nct_id] = study
        if set(top_by_nct) != set(union_by_nct) or any(
            _canonical_page_sha256(top_by_nct[nct_id])
            != _canonical_page_sha256(union_by_nct[nct_id])
            for nct_id in union_by_nct
        ):
            raise TranslationContextError(
                "top-level studies disagree with the declared-query NCT union"
            )
        for page_index, page in enumerate(pages):
            if "totalCount" in page and _strict_nonnegative_int(
                page["totalCount"], f"pages[{page_index}].totalCount"
            ) != len(union_by_nct):
                raise TranslationContextError(
                    "top-level page totalCount disagrees with declared NCT union"
                )
        canonical["declared_query_set"] = validated_queries
        if query_semantics_version is not None:
            canonical["declared_query_semantics_version"] = query_semantics_version
    elif wrapped:
        page_studies, page_total, page_complete = _validate_query_pages(
            pages,
            field="pages",
        )
        if page_studies != studies:
            raise TranslationContextError("top-level page reconstruction failed")
        if total_count != page_total:
            raise TranslationContextError(
                "top-level total_count disagrees with page totalCount"
            )
        if complete != page_complete:
            raise TranslationContextError("top-level completeness disagrees with pages")
        if len(request_urls) != len(pages):
            raise TranslationContextError(
                "request_urls must have one URL per frozen page"
            )
    return ClinicalTrialsSnapshot(
        document=canonical,
        studies=_json_deep_copy(studies),
        request_urls=list(request_urls),
        total_count=total_count,
        complete=complete,
        retrieved_at_utc=retrieved,
        source_mode="frozen_json",
        api_version=api_version,
        data_timestamp=data_timestamp,
        # A serialized wrapper is self-attested even when its stored version
        # audit is internally consistent. Only the live retrieval path upgrades
        # this flag after both official /version calls complete in-process.
        version_stable=False,
    )


def _normalize_term(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _normalize_entity_identity(value: str) -> str:
    """Normalize entity labels while retaining signed biomarker state."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    for symbol, token in (
        ("!=", " statenotequal "),
        ("≥", " stategreaterorequal "),
        ("≤", " statelessorequal "),
        ("≠", " statenotequal "),
        ("±", " stateplusminus "),
        ("≈", " stateapproximatelyequal "),
        ("~", " stateapproximately "),
        (">", " stategreater "),
        ("<", " stateless "),
        ("=", " stateequal "),
        ("↑", " stateup "),
        ("↓", " statedown "),
    ):
        normalized = normalized.replace(symbol, token)
    normalized = normalized.replace("+", " stateplus ")
    normalized = re.sub(
        r"(?<!\w)[\-\N{MINUS SIGN}\N{EN DASH}\N{EM DASH}]\s*(?=\d)"
        r"|(?<!\w)[\-\N{MINUS SIGN}\N{EN DASH}\N{EM DASH}](?!\w)"
        r"|(?<=\w)[\-\N{MINUS SIGN}\N{EN DASH}\N{EM DASH}](?=$|[^\w])",
        " stateminus ",
        normalized,
    )
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _ordered_entity_terms(
    canonical: str,
    aliases: list[str] | tuple[str, ...],
) -> list[str]:
    normalized: list[str] = []
    for value in (canonical, *aliases):
        term = _normalize_entity_identity(value)
        if term and term not in normalized:
            normalized.append(term)
    if not normalized:
        raise TranslationContextError("a canonical entity-matching term is required")
    return normalized


def _declared_entity_terms(values: list[str] | tuple[str, ...]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        term = _normalize_entity_identity(value)
        if term and term not in normalized:
            normalized.append(term)
    return normalized


def _typed_query_terms(
    declarations: list[tuple[str, list[str | None]]],
) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    seen: dict[str, str] = {}
    for lane, raw_values in declarations:
        for raw in raw_values:
            if raw is None:
                continue
            if not isinstance(raw, str):
                raise TranslationContextError(
                    f"declared query term in {lane} must be a string"
                )
            value = raw.strip()
            key = _normalize_entity_identity(value)
            if not key:
                raise TranslationContextError(
                    f"declared query term in {lane} cannot be empty"
                )
            previous_lane = seen.get(key)
            if previous_lane is not None and previous_lane != lane:
                raise TranslationContextError(
                    "the same normalized term cannot be declared in multiple "
                    f"query lanes ({previous_lane}, {lane})"
                )
            if previous_lane is None:
                values.append((value, lane))
                seen[key] = lane
    return values


def _typed_query_context_binding(
    snapshot: ClinicalTrialsSnapshot,
    context: TreatmentDiseaseContextRecord,
    *,
    treatment_entity_aliases: list[str] | tuple[str, ...],
    treatment_class_terms: list[str] | tuple[str, ...],
    cancer_entity_aliases: list[str] | tuple[str, ...],
    cancer_ancestor_terms: list[str] | tuple[str, ...],
    subtype_entity_aliases: list[str] | tuple[str, ...],
) -> str:
    if snapshot.document.get("declared_query_semantics_version") != (
        "typed_query_lanes_v1"
    ):
        return "unverified_legacy_or_raw_snapshot"
    treatment_queries = _typed_query_terms(
        [
            ("treatment_canonical", [context.treatment_name]),
            ("treatment_entity_alias", list(treatment_entity_aliases)),
            ("treatment_class_term", list(treatment_class_terms)),
        ]
    )
    condition_queries = _typed_query_terms(
        [
            ("cancer_canonical", [context.cancer_type]),
            ("cancer_entity_alias", list(cancer_entity_aliases)),
            ("disease_subtype_canonical", [context.disease_subtype]),
            ("subtype_entity_alias", list(subtype_entity_aliases)),
            ("cancer_ancestor_term", list(cancer_ancestor_terms)),
        ]
    )
    expected = {
        (
            _normalize_entity_identity(treatment_query),
            treatment_lane,
            _normalize_entity_identity(condition_query),
            condition_lane,
        )
        for treatment_query, treatment_lane in treatment_queries
        for condition_query, condition_lane in condition_queries
    }
    declared_query_set = snapshot.document.get("declared_query_set")
    if not isinstance(declared_query_set, list):
        raise TranslationContextError(
            "typed ClinicalTrials snapshot requires declared_query_set"
        )
    observed = {
        (
            _normalize_entity_identity(str(query["treatment_query"])),
            str(query["treatment_query_lane"]),
            _normalize_entity_identity(str(query["condition_query"])),
            str(query["condition_query_lane"]),
        )
        for query in declared_query_set
    }
    if observed != expected:
        raise TranslationContextError(
            "ClinicalTrials declared query set does not match the requested "
            "treatment/disease context and discovery terms"
        )
    return "verified_typed_query_cross_product"


def _contains_term(candidate: str, term: str) -> bool:
    return candidate == term or f" {term} " in f" {candidate} "


def _matched_original_terms(values: list[str], terms: list[str]) -> list[str]:
    matched = []
    for value in values:
        normalized = _normalize_entity_identity(value)
        if any(_contains_term(normalized, term) for term in terms):
            matched.append(value)
    return sorted(set(matched), key=str.casefold)


def _matched_exact_terms(values: list[str], terms: list[str]) -> list[str]:
    matched = [value for value in values if _normalize_entity_identity(value) in terms]
    return sorted(set(matched), key=str.casefold)


def _json_list(values: list[Any]) -> str:
    unique: list[Any] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return json.dumps(unique, ensure_ascii=False, separators=(",", ":"))


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TranslationContextError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TranslationContextError(f"{field} must be a list")
    return value


def _normalize_one_trial(
    study: dict[str, Any],
    *,
    context: TreatmentDiseaseContextRecord,
    treatment_terms: list[str],
    treatment_class_terms: list[str],
    subtype_terms: list[str],
    cancer_terms: list[str],
    cancer_ancestor_terms: list[str],
    biomarker_terms: list[str],
    retrieved_at_utc: datetime,
    api_version: str,
) -> dict[str, Any]:
    protocol = _mapping(study.get("protocolSection"), "protocolSection")
    identification = _mapping(
        protocol.get("identificationModule"), "identificationModule"
    )
    status = _mapping(protocol.get("statusModule"), "statusModule")
    design = _mapping(protocol.get("designModule"), "designModule")
    conditions_module = _mapping(protocol.get("conditionsModule"), "conditionsModule")
    arms_module = _mapping(
        protocol.get("armsInterventionsModule"), "armsInterventionsModule"
    )
    outcomes_module = _mapping(protocol.get("outcomesModule"), "outcomesModule")
    references_module = _mapping(protocol.get("referencesModule"), "referencesModule")

    nct_id = str(identification.get("nctId", "")).strip()
    brief_title = str(identification.get("briefTitle", "")).strip()
    interventions = _sequence(arms_module.get("interventions"), "interventions")
    intervention_names: list[str] = []
    intervention_types: list[str] = []
    searchable_interventions: list[str] = []
    active_match_flags: list[bool] = []
    active_combination_flags: list[bool] = []
    for index, raw in enumerate(interventions):
        item = _mapping(raw, f"interventions[{index}]")
        name = str(item.get("name", "")).strip()
        item_type = str(item.get("type", "UNKNOWN")).strip() or "UNKNOWN"
        if name:
            intervention_names.append(name)
            other_names = _sequence(item.get("otherNames"), "otherNames")
            item_search_terms = [name, *(str(value) for value in other_names)]
            is_active = item_type.upper() in {
                "DRUG",
                "BIOLOGICAL",
                "GENETIC",
                "COMBINATION_PRODUCT",
                "RADIATION",
            } and not re.search(r"\b(placebo|sham)\b", name, re.IGNORECASE)
            if is_active:
                searchable_interventions.extend(item_search_terms)
                active_match_flags.append(
                    any(
                        any(
                            _contains_term(_normalize_term(value), term)
                            for term in treatment_terms
                        )
                        for value in item_search_terms
                    )
                )
                active_combination_flags.append(
                    item_type.upper() == "COMBINATION_PRODUCT"
                    or bool(
                        re.search(
                            r"(?:\+|/|\bplus\b|\bwith\b|\band\b)",
                            name,
                            re.IGNORECASE,
                        )
                    )
                )
        intervention_types.append(item_type)

    canonical_treatment = treatment_terms[0]
    normalized_interventions = [
        _normalize_entity_identity(value) for value in searchable_interventions
    ]
    exact_canonical = [
        original
        for original, normalized in zip(
            searchable_interventions,
            normalized_interventions,
            strict=True,
        )
        if normalized == canonical_treatment
    ]
    exact_alias = [
        original
        for original, normalized in zip(
            searchable_interventions,
            normalized_interventions,
            strict=True,
        )
        if normalized in treatment_terms[1:]
    ]
    component_matches = _matched_original_terms(
        searchable_interventions, treatment_terms
    )
    class_matches = _matched_original_terms(
        searchable_interventions, treatment_class_terms
    )
    if exact_canonical:
        intervention_match = TrialInterventionMatch.EXACT_CANONICAL
        intervention_match_terms = exact_canonical
    elif exact_alias:
        intervention_match = TrialInterventionMatch.EXPLICIT_ALIAS
        intervention_match_terms = exact_alias
    elif component_matches:
        intervention_match = TrialInterventionMatch.EXPLICIT_COMPONENT
        intervention_match_terms = component_matches
    elif class_matches:
        intervention_match = TrialInterventionMatch.DECLARED_CLASS_TERM
        intervention_match_terms = class_matches
    else:
        intervention_match = TrialInterventionMatch.NO_STRUCTURED_MATCH
        intervention_match_terms = []

    conditions = [
        str(value).strip()
        for value in _sequence(conditions_module.get("conditions"), "conditions")
        if str(value).strip()
    ]
    keywords = [
        str(value).strip()
        for value in _sequence(conditions_module.get("keywords"), "keywords")
        if str(value).strip()
    ]
    disease_fields = conditions
    canonical_subtype_matches = _matched_exact_terms(disease_fields, subtype_terms[:1])
    alias_subtype_matches = _matched_exact_terms(disease_fields, subtype_terms[1:])
    canonical_cancer_matches = _matched_exact_terms(disease_fields, cancer_terms[:1])
    alias_cancer_matches = _matched_exact_terms(disease_fields, cancer_terms[1:])
    ancestor_matches = _matched_exact_terms(disease_fields, cancer_ancestor_terms)
    # A subtype phrase is not proof of its parent cancer: substring inference
    # conflates entities such as small-cell and non-small-cell lung cancer.
    # Until an ontology relation is validated, require a separate exact parent
    # condition in the structured registry field.
    subtype_parent_bound = bool(canonical_cancer_matches)
    if subtype_terms and canonical_subtype_matches and subtype_parent_bound:
        disease_match = TrialDiseaseMatch.EXPLICIT_SUBTYPE_TERM
        disease_match_terms = canonical_subtype_matches
    elif subtype_terms and alias_subtype_matches and subtype_parent_bound:
        disease_match = TrialDiseaseMatch.EXPLICIT_SUBTYPE_ALIAS
        disease_match_terms = alias_subtype_matches
    elif canonical_cancer_matches:
        disease_match = TrialDiseaseMatch.CANCER_TYPE_TERM_ONLY
        disease_match_terms = canonical_cancer_matches
    elif alias_cancer_matches:
        disease_match = TrialDiseaseMatch.CANCER_ENTITY_ALIAS
        disease_match_terms = alias_cancer_matches
    elif ancestor_matches:
        disease_match = TrialDiseaseMatch.DECLARED_ANCESTOR_TERM
        disease_match_terms = ancestor_matches
    else:
        disease_match = TrialDiseaseMatch.NO_STRUCTURED_MATCH
        disease_match_terms = []

    biomarker_matches = _matched_original_terms(
        [*conditions, *keywords], biomarker_terms
    )
    if not biomarker_terms:
        biomarker_match = TrialBiomarkerMatch.NOT_REQUESTED
    elif biomarker_matches:
        biomarker_match = TrialBiomarkerMatch.EXPLICIT_STRUCTURED_TERM
    else:
        biomarker_match = TrialBiomarkerMatch.NOT_REPORTED_IN_STRUCTURED_TERMS

    if not intervention_names:
        regimen_relation = TrialRegimenRelation.UNRESOLVED
    elif not any(active_match_flags):
        regimen_relation = TrialRegimenRelation.UNRESOLVED
    elif any(not matched for matched in active_match_flags) or any(
        matched and combination
        for matched, combination in zip(
            active_match_flags, active_combination_flags, strict=True
        )
    ):
        regimen_relation = TrialRegimenRelation.ADDITIONAL_ACTIVE_AGENT_LISTED
    else:
        regimen_relation = TrialRegimenRelation.NO_ADDITIONAL_ACTIVE_AGENT_LISTED

    primary_outcomes = []
    for index, raw in enumerate(
        _sequence(outcomes_module.get("primaryOutcomes"), "primaryOutcomes")
    ):
        item = _mapping(raw, f"primaryOutcomes[{index}]")
        measure = str(item.get("measure", "")).strip()
        if measure:
            primary_outcomes.append(measure)
    publications = []
    for index, raw in enumerate(
        _sequence(references_module.get("references"), "references")
    ):
        item = _mapping(raw, f"references[{index}]")
        pmid = str(item.get("pmid", "")).strip()
        citation = str(item.get("citation", "")).strip()
        if pmid:
            publications.append(f"PMID:{pmid}")
        elif citation:
            publications.append(citation)

    enrollment = _mapping(design.get("enrollmentInfo"), "enrollmentInfo")
    has_results = study.get("hasResults", False)
    if not isinstance(has_results, bool):
        raise TranslationContextError("hasResults must be a boolean")
    record = {
        "context_id": context.context_id,
        "nct_id": nct_id,
        "brief_title": brief_title,
        "official_title": identification.get("officialTitle"),
        "study_type": str(design.get("studyType", "UNKNOWN")),
        "overall_status": str(status.get("overallStatus", "UNKNOWN")),
        "phases_json": _json_list(
            [str(value) for value in _sequence(design.get("phases"), "phases")]
        ),
        "conditions_json": _json_list(conditions),
        "interventions_json": _json_list(intervention_names),
        "intervention_types_json": _json_list(intervention_types),
        "primary_outcomes_json": _json_list(primary_outcomes),
        "linked_publications_json": _json_list(publications),
        "enrollment_count": enrollment.get("count"),
        "enrollment_type": enrollment.get("type"),
        "start_date": _mapping(status.get("startDateStruct"), "startDateStruct").get(
            "date"
        ),
        "completion_date": _mapping(
            status.get("completionDateStruct"), "completionDateStruct"
        ).get("date"),
        "source_first_post_date": _mapping(
            status.get("studyFirstPostDateStruct"), "studyFirstPostDateStruct"
        ).get("date"),
        "source_last_update_date": _mapping(
            status.get("lastUpdatePostDateStruct"), "lastUpdatePostDateStruct"
        ).get("date"),
        "has_results": has_results,
        "intervention_match": intervention_match,
        "disease_match": disease_match,
        "biomarker_match": biomarker_match,
        "intervention_match_terms_json": _json_list(intervention_match_terms),
        "disease_match_terms_json": _json_list(disease_match_terms),
        "biomarker_match_terms_json": _json_list(biomarker_matches),
        "regimen_relation": regimen_relation,
        "source_url": f"https://clinicaltrials.gov/study/{nct_id}",
        "source_api_version": api_version,
        "retrieved_at_utc": retrieved_at_utc,
    }
    return ClinicalTrialContextRecord.model_validate(record).model_dump(mode="json")


def _revalidate_treatment_disease_context(
    context: TreatmentDiseaseContextRecord | dict[str, Any],
) -> TreatmentDiseaseContextRecord:
    payload = (
        context.model_dump(mode="python")
        if isinstance(context, TreatmentDiseaseContextRecord)
        else context
    )
    return TreatmentDiseaseContextRecord.model_validate(payload)


def normalize_clinical_trials(
    snapshot: ClinicalTrialsSnapshot,
    context: TreatmentDiseaseContextRecord | dict[str, Any],
    *,
    treatment_entity_aliases: list[str] | tuple[str, ...] = (),
    treatment_class_terms: list[str] | tuple[str, ...] = (),
    cancer_entity_aliases: list[str] | tuple[str, ...] = (),
    cancer_ancestor_terms: list[str] | tuple[str, ...] = (),
    subtype_entity_aliases: list[str] | tuple[str, ...] = (),
    biomarker_aliases: list[str] | tuple[str, ...] = (),
) -> pd.DataFrame:
    snapshot = _revalidate_snapshot_for_report(snapshot)
    context = _revalidate_treatment_disease_context(context)
    if subtype_entity_aliases and not context.disease_subtype:
        raise TranslationContextError(
            "subtype entity aliases require a canonical disease_subtype"
        )
    treatment_terms = _ordered_entity_terms(
        context.treatment_name, treatment_entity_aliases
    )
    normalized_treatment_class_terms = _declared_entity_terms(treatment_class_terms)
    cancer_terms = _ordered_entity_terms(context.cancer_type, cancer_entity_aliases)
    normalized_cancer_ancestor_terms = _declared_entity_terms(cancer_ancestor_terms)
    subtype_terms = (
        _ordered_entity_terms(context.disease_subtype, subtype_entity_aliases)
        if context.disease_subtype
        else []
    )
    biomarker_terms = (
        _ordered_entity_terms(context.biomarker_context, biomarker_aliases)
        if context.biomarker_context
        else []
    )
    records = []
    seen: set[str] = set()
    for raw in snapshot.studies:
        if not isinstance(raw, dict):
            raise TranslationContextError("every study must be a JSON object")
        record = _normalize_one_trial(
            raw,
            context=context,
            treatment_terms=treatment_terms,
            treatment_class_terms=normalized_treatment_class_terms,
            subtype_terms=subtype_terms,
            cancer_terms=cancer_terms,
            cancer_ancestor_terms=normalized_cancer_ancestor_terms,
            biomarker_terms=biomarker_terms,
            retrieved_at_utc=snapshot.retrieved_at_utc,
            api_version=snapshot.api_version,
        )
        nct_id = record["nct_id"]
        if nct_id in seen:
            raise TranslationContextError(f"duplicate NCT record in snapshot: {nct_id}")
        seen.add(nct_id)
        records.append(record)
    columns = list(ClinicalTrialContextRecord.model_fields)
    return pd.DataFrame.from_records(records, columns=columns)


def _validated_frame(
    frame: pd.DataFrame | None,
    contract: type[PreclinicalEvidenceRecord] | type[PatientMolecularEvidenceRecord],
) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(columns=list(contract.model_fields))
    valid, errors = validate_records(frame, contract)
    if not errors.empty:
        first = errors.iloc[0]
        raise TranslationContextError(
            f"invalid {contract.__name__} row {first['row_number']}: {first['error']}"
        )
    return valid


def _validate_gene_identity_consistency(*frames: pd.DataFrame) -> None:
    """Reject contradictory curator mappings within one identifier release."""

    observed: dict[tuple[str, str, str], str] = {}
    for frame in frames:
        required = {
            "gene_symbol",
            "gene_id",
            "gene_identifier_source",
            "gene_identifier_version",
        }
        if frame.empty or not required.issubset(frame.columns):
            continue
        for _, row in frame.iterrows():
            values = [row[field] for field in required]
            if any(value is None or pd.isna(value) for value in values):
                continue
            key = (
                _normalize_entity_identity(str(row["gene_symbol"])),
                _normalize_term(str(row["gene_identifier_source"])),
                str(row["gene_identifier_version"]).strip(),
            )
            gene_id = _normalize_term(str(row["gene_id"]))
            previous = observed.setdefault(key, gene_id)
            if previous != gene_id:
                raise TranslationContextError(
                    "conflicting versioned gene identities for one gene symbol"
                )


def _canonical_component_set(value: Any) -> set[str]:
    """Parse already contract-validated exposure IDs for relation-aware matching."""

    if value is None or pd.isna(value):
        return set()
    parsed = json.loads(str(value))
    return {
        " ".join(unicodedata.normalize("NFKC", item).casefold().split())
        for item in parsed
    }


def _structured_context_match(
    row: pd.Series,
    context: TreatmentDiseaseContextRecord,
    *,
    treatment_terms: list[str],
    cancer_terms: list[str],
    subtype_terms: list[str],
    biomarker_terms: list[str],
) -> dict[str, str]:
    treatment_value = _normalize_entity_identity(str(row["treatment_name"]))
    if treatment_value == treatment_terms[0]:
        treatment_name_match = "exact_canonical"
    elif treatment_value in treatment_terms[1:]:
        treatment_name_match = "explicit_alias_unverified"
    else:
        treatment_name_match = "none"
    treatment_id = row.get("treatment_id")
    treatment_id = (
        None
        if treatment_id is None or pd.isna(treatment_id)
        else str(treatment_id).strip()
    )
    if treatment_name_match == "none":
        treatment_match = "none"
    elif context.treatment_id and treatment_id is None:
        treatment_match = "name_only_unverified"
    elif context.treatment_id and treatment_id != context.treatment_id:
        treatment_match = "id_conflict"
    elif context.treatment_id:
        row_ontology_name = str(row.get("treatment_ontology_name", "")).strip()
        row_ontology_version = str(row.get("treatment_ontology_version", "")).strip()
        if (
            _normalize_term(row_ontology_name)
            != _normalize_term(context.treatment_ontology_name or "")
            or row_ontology_version != context.treatment_ontology_version
        ):
            treatment_match = "exact_id_different_ontology_version"
        else:
            treatment_match = "exact_id"
    else:
        treatment_match = treatment_name_match

    cancer_value = _normalize_entity_identity(str(row["cancer_type"]))
    cancer_name_match = (
        "exact_canonical"
        if cancer_value == cancer_terms[0]
        else "explicit_alias_unverified"
        if cancer_value in cancer_terms[1:]
        else "none"
    )
    cancer_id = row.get("cancer_id")
    cancer_id = (
        None if cancer_id is None or pd.isna(cancer_id) else str(cancer_id).strip()
    )
    if cancer_name_match == "none":
        cancer_match = "none"
    elif context.cancer_id and cancer_id is None:
        cancer_match = "name_only_unverified"
    elif context.cancer_id and cancer_id != context.cancer_id:
        cancer_match = "id_conflict"
    elif context.cancer_id:
        row_ontology_name = str(row.get("disease_ontology_name", "")).strip()
        row_ontology_version = str(row.get("disease_ontology_version", "")).strip()
        if (
            _normalize_term(row_ontology_name)
            != _normalize_term(context.disease_ontology_name or "")
            or row_ontology_version != context.disease_ontology_version
        ):
            cancer_match = "exact_id_different_ontology_version"
        else:
            cancer_match = "exact_id"
    else:
        cancer_match = cancer_name_match
    subtype_value = row.get("disease_subtype")
    if not subtype_terms:
        subtype_match = (
            "context_axis_unresolved"
            if subtype_value is None or pd.isna(subtype_value)
            else "evidence_narrower_than_context"
        )
    elif subtype_value is None or pd.isna(subtype_value):
        subtype_match = "unspecified"
    elif _normalize_entity_identity(str(subtype_value)) in subtype_terms:
        subtype_name_match = (
            "exact_canonical"
            if _normalize_entity_identity(str(subtype_value)) == subtype_terms[0]
            else "explicit_alias_unverified"
        )
        subtype_id = row.get("disease_subtype_id")
        subtype_id = (
            None
            if subtype_id is None or pd.isna(subtype_id)
            else str(subtype_id).strip()
        )
        observed_parent_verified = row.get("disease_subtype_parent_binding_verified")
        parent_binding_verified = (
            context.disease_subtype_parent_binding_verified is True
            and observed_parent_verified is not None
            and not pd.isna(observed_parent_verified)
            and bool(observed_parent_verified)
            and str(context.disease_subtype_parent_id) == str(context.cancer_id)
            and str(row.get("disease_subtype_parent_id")) == str(row.get("cancer_id"))
            and str(row.get("disease_subtype_parent_id"))
            == str(context.disease_subtype_parent_id)
        )
        observed_parent_id = row.get("disease_subtype_parent_id")
        observed_parent_id = (
            None
            if observed_parent_id is None or pd.isna(observed_parent_id)
            else str(observed_parent_id).strip()
        )
        if (
            context.disease_subtype_id
            and subtype_id is not None
            and subtype_id != context.disease_subtype_id
        ):
            subtype_match = "id_conflict"
        elif (
            context.disease_subtype_parent_id
            and observed_parent_id is not None
            and observed_parent_id != context.disease_subtype_parent_id
        ):
            subtype_match = "parent_id_conflict"
        elif not parent_binding_verified:
            subtype_match = "parent_binding_unverified"
        elif context.disease_subtype_id and subtype_id is None:
            subtype_match = "name_only_unverified"
        elif context.disease_subtype_id:
            row_ontology_name = str(row.get("disease_ontology_name", "")).strip()
            row_ontology_version = str(row.get("disease_ontology_version", "")).strip()
            if (
                _normalize_term(row_ontology_name)
                != _normalize_term(context.disease_ontology_name or "")
                or row_ontology_version != context.disease_ontology_version
            ):
                subtype_match = "exact_id_different_ontology_version"
            else:
                subtype_match = "exact_id"
        else:
            subtype_match = subtype_name_match
    else:
        subtype_match = "conflict_or_unmapped"

    biomarker_value = row.get("biomarker_context")
    if not biomarker_terms:
        biomarker_match = (
            "context_axis_unresolved"
            if biomarker_value is None or pd.isna(biomarker_value)
            else "evidence_narrower_than_context"
        )
    elif biomarker_value is None or pd.isna(biomarker_value):
        biomarker_match = "unspecified"
    elif _normalize_entity_identity(str(biomarker_value)) in biomarker_terms:
        biomarker_name_is_canonical = (
            _normalize_entity_identity(str(biomarker_value)) == biomarker_terms[0]
        )
        typed_axes = (
            ("biomarker_feature_type", context.biomarker_feature_type),
            ("biomarker_state", context.biomarker_state),
            ("biomarker_specimen_type", context.biomarker_specimen_type),
            (
                "biomarker_measurement_timepoint",
                context.biomarker_measurement_timepoint,
            ),
        )
        typed_matches = []
        for field_name, expected in typed_axes:
            observed = row.get(field_name)
            if hasattr(expected, "value"):
                expected = expected.value
            if hasattr(observed, "value"):
                observed = observed.value
            normalizer = (
                _normalize_entity_identity
                if field_name in {"biomarker_state", "biomarker_specimen_type"}
                else _normalize_term
            )
            typed_matches.append(
                observed is not None
                and not pd.isna(observed)
                and normalizer(str(observed)) == normalizer(str(expected))
            )
        observed_attestation = row.get("biomarker_axes_informative_verified")
        observed_status = row.get("biomarker_axes_observation_status")
        axes_attested = (
            context.biomarker_axes_informative_verified is True
            and str(context.biomarker_axes_observation_status) == "observed"
            and observed_attestation is not None
            and not pd.isna(observed_attestation)
            and bool(observed_attestation)
            and observed_status is not None
            and not pd.isna(observed_status)
            and str(observed_status) == "observed"
        )
        if not axes_attested:
            biomarker_match = "typed_axis_unresolved"
        elif not all(typed_matches):
            biomarker_match = "typed_axis_conflict"
        elif not biomarker_name_is_canonical:
            biomarker_match = "alias_unverified"
        else:
            biomarker_match = "exact_typed"
    else:
        biomarker_match = "conflict_or_unmapped"

    regimen_value = row.get("regimen_name")
    if not context.regimen_name:
        regimen_match = (
            "context_axis_unresolved"
            if regimen_value is None or pd.isna(regimen_value)
            else "evidence_narrower_than_context"
        )
    elif regimen_value is None or pd.isna(regimen_value):
        regimen_match = "unspecified"
    else:
        context_verified = context.regimen_active_exposures_verified is True
        patient_regimen_fields = "treatment_active_exposure_ids_json" in row.index
        observed_verified = row.get(
            "active_exposure_ids_curator_verified"
            if patient_regimen_fields
            else "regimen_active_exposures_verified"
        )
        observed_verified = (
            observed_verified is not None
            and not pd.isna(observed_verified)
            and bool(observed_verified)
        )
        observed_source = row.get(
            "active_exposure_identifier_source"
            if patient_regimen_fields
            else "regimen_active_exposure_identifier_source"
        )
        observed_version = row.get(
            "active_exposure_identifier_version"
            if patient_regimen_fields
            else "regimen_active_exposure_identifier_version"
        )
        context_source = context.regimen_active_exposure_identifier_source
        source_is_synthetic = (
            _normalize_term(str(context_source or "")) == "synthetic"
            or _normalize_term(str(observed_source or "")) == "synthetic"
        )
        provenance_matches = (
            observed_source is not None
            and not pd.isna(observed_source)
            and _normalize_term(str(observed_source))
            == _normalize_term(str(context_source))
            and observed_version is not None
            and not pd.isna(observed_version)
            and str(observed_version)
            == str(context.regimen_active_exposure_identifier_version)
        )
        expected_components = _canonical_component_set(
            context.regimen_active_exposure_ids_json
        )
        observed_components = _canonical_component_set(
            row.get(
                "treatment_active_exposure_ids_json"
                if patient_regimen_fields
                else "regimen_active_exposure_ids_json"
            )
        )
        if expected_components < observed_components:
            regimen_match = "additional_active_components"
        elif observed_components < expected_components:
            regimen_match = "missing_active_components"
        elif observed_components != expected_components:
            regimen_match = "different_active_components"
        elif (
            not context_verified
            or not observed_verified
            or not provenance_matches
            or source_is_synthetic
        ):
            regimen_match = "unverified_active_components"
        else:
            context_relation = str(context.regimen_component_relation)
            observed_relation = str(
                row.get(
                    "treatment_regimen_component_relation"
                    if patient_regimen_fields
                    else "regimen_component_relation"
                )
            )
            if context_relation != observed_relation:
                regimen_match = "relation_conflict"
            else:
                regimen_match = "exact_active_components"

    if (
        regimen_match == "exact_active_components"
        and "treatment_exposure_verified" in row.index
    ):
        exposure_verified = row.get("treatment_exposure_verified")
        if (
            exposure_verified is None
            or pd.isna(exposure_verified)
            or not bool(exposure_verified)
        ):
            regimen_match = "treatment_exposure_unverified"

    stage_value = row.get("stage")
    if not context.stage:
        stage_match = (
            "context_axis_unresolved"
            if stage_value is None or pd.isna(stage_value)
            else "evidence_narrower_than_context"
        )
    elif stage_value is None or pd.isna(stage_value):
        stage_match = "unspecified"
    elif _normalize_entity_identity(str(stage_value)) == _normalize_entity_identity(
        context.stage
    ):
        stage_match = "exact"
    else:
        stage_match = "different_or_unmapped"

    line_value = row.get("line_of_therapy")
    if not context.line_of_therapy:
        line_match = (
            "context_axis_unresolved"
            if line_value is None or pd.isna(line_value)
            else "evidence_narrower_than_context"
        )
    elif line_value is None or pd.isna(line_value):
        line_match = "unspecified"
    elif _normalize_entity_identity(str(line_value)) == _normalize_entity_identity(
        context.line_of_therapy
    ):
        line_match = "exact"
    else:
        line_match = "different_or_unmapped"

    if "perturbed_compartment" not in row.index:
        compartment_match = "not_applicable"
    elif (
        str(row["perturbed_compartment"]) == PerturbedCompartment.UNKNOWN.value
        or context.perturbed_compartment == PerturbedCompartment.UNKNOWN
    ):
        compartment_match = "unresolved"
    elif str(row["perturbed_compartment"]) == context.perturbed_compartment.value:
        compartment_match = "exact"
    else:
        compartment_match = "different"

    if "endpoint_category" not in row.index:
        endpoint_match = "not_applicable"
    elif (
        str(row["endpoint_category"]) == ScreenEndpointCategory.UNKNOWN.value
        or context.screen_endpoint_category == ScreenEndpointCategory.UNKNOWN
    ):
        endpoint_match = "unresolved"
    elif str(row["endpoint_category"]) == context.screen_endpoint_category.value:
        endpoint_match = "exact"
    else:
        endpoint_match = "different"
    return {
        "report_only_treatment_match": treatment_match,
        "report_only_cancer_match": cancer_match,
        "report_only_subtype_match": subtype_match,
        "report_only_biomarker_match": biomarker_match,
        "report_only_regimen_match": regimen_match,
        "report_only_stage_match": stage_match,
        "report_only_line_of_therapy_match": line_match,
        "report_only_perturbed_compartment_match": compartment_match,
        "report_only_endpoint_category_match": endpoint_match,
    }


def _family_components(frame: pd.DataFrame) -> pd.Series:
    """Collapse transitive source/raw-family links without double counting."""

    if frame.empty:
        return pd.Series(dtype="string", index=frame.index)
    parents: dict[str, str] = {}

    def find(value: str) -> str:
        parents.setdefault(value, value)
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    row_nodes: dict[Any, str] = {}
    for index, row in frame.iterrows():
        source_node = f"source:{row['source_family_id']}"
        row_nodes[index] = source_node
        find(source_node)
        study_node = f"study:{row['source_study_id']}"
        union(source_node, study_node)
        cohort_id = row.get("cohort_id")
        if cohort_id is not None and not pd.isna(cohort_id) and str(cohort_id).strip():
            union(study_node, f"cohort:{row['source_study_id']}:{cohort_id}")
        raw_family = row.get("raw_data_family_id")
        if (
            raw_family is not None
            and not pd.isna(raw_family)
            and str(raw_family).strip()
        ):
            union(source_node, f"raw:{raw_family}")
    return pd.Series(
        {index: find(node) for index, node in row_nodes.items()}, dtype="string"
    )


def _target_component_membership(
    frame: pd.DataFrame,
    *,
    target_source_family_id: str | None,
    target_raw_data_family_id: str | None,
) -> tuple[pd.Series, bool]:
    """Resolve target overlap transitively across source and raw-data families."""

    if frame.empty:
        return pd.Series(dtype=bool, index=frame.index), False
    components = _family_components(frame)
    source_matches = (
        frame["source_family_id"].astype(str).eq(target_source_family_id)
        if target_source_family_id
        else pd.Series(False, index=frame.index)
    )
    raw_values = frame["raw_data_family_id"]
    raw_matches = (
        raw_values.notna() & raw_values.astype(str).eq(target_raw_data_family_id)
        if target_raw_data_family_id
        else pd.Series(False, index=frame.index)
    )
    source_roots = set(components.loc[source_matches].astype(str))
    raw_roots = set(components.loc[raw_matches].astype(str))
    if source_roots and raw_roots and source_roots.isdisjoint(raw_roots):
        raise TranslationContextError(
            "target source and raw-data family IDs resolve to different components"
        )
    target_roots = source_roots | raw_roots
    membership = components.astype(str).isin(target_roots)
    supplied_identifiers_found = (
        (not target_source_family_id or bool(source_roots))
        and (not target_raw_data_family_id or bool(raw_roots))
        and bool(target_source_family_id or target_raw_data_family_id)
    )
    return membership, supplied_identifiers_found


def _screen_evidence(
    frame: pd.DataFrame,
    *,
    context: TreatmentDiseaseContextRecord,
    cutoff_date: date,
    treatment_terms: list[str],
    cancer_terms: list[str],
    subtype_terms: list[str],
    biomarker_terms: list[str],
    target_source_family_id: str | None,
    target_raw_data_family_id: str | None,
    target_absence_attested: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    used_records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    all_components = _family_components(frame)
    target_membership, _ = _target_component_membership(
        frame,
        target_source_family_id=target_source_family_id,
        target_raw_data_family_id=target_raw_data_family_id,
    )
    available_dates = frame["available_date"].map(
        lambda value: date.fromisoformat(value) if isinstance(value, str) else value
    )
    cutoff_eligible = frame.loc[available_dates.map(lambda value: value <= cutoff_date)]
    # Use the full graph for conservative bridge/self-family exclusion, but do
    # not let a future evidence row positively verify independence at cutoff.
    _, supplied_identifiers_found = _target_component_membership(
        cutoff_eligible,
        target_source_family_id=target_source_family_id,
        target_raw_data_family_id=target_raw_data_family_id,
    )
    if target_absence_attested and bool(target_membership.any()):
        raise TranslationContextError(
            "target absence attestation conflicts with evidence-family overlap"
        )
    independence_verified = target_absence_attested or supplied_identifiers_found
    for index, row in frame.iterrows():
        record = row.to_dict()
        match = _structured_context_match(
            row,
            context,
            treatment_terms=treatment_terms,
            cancer_terms=cancer_terms,
            subtype_terms=subtype_terms,
            biomarker_terms=biomarker_terms,
        )
        raw_family = row.get("raw_data_family_id")
        target_raw_family_matches = bool(
            target_raw_data_family_id
            and raw_family is not None
            and not pd.isna(raw_family)
            and str(raw_family) == target_raw_data_family_id
        )
        available_date = row["available_date"]
        if isinstance(available_date, str):
            available_date = date.fromisoformat(available_date)
        reason = None
        if available_date > cutoff_date:
            reason = "post_cutoff"
        elif target_source_family_id and (
            str(row["source_family_id"]) == target_source_family_id
        ):
            reason = "target_source_family"
        elif target_raw_family_matches:
            reason = "target_raw_data_family"
        elif bool(target_membership.get(index, False)):
            reason = "target_family_component"
        elif match["report_only_treatment_match"] in {"none", "id_conflict"}:
            reason = "treatment_mismatch"
        elif match["report_only_cancer_match"] in {"none", "id_conflict"}:
            reason = "cancer_mismatch"
        if reason:
            exclusions.append(
                {
                    "evidence_id": row["evidence_id"],
                    "reason": reason,
                    **match,
                }
            )
            continue
        used_records.append(
            {
                **record,
                **match,
                "report_only_independence_unit_id": all_components.loc[index],
            }
        )
    match_columns = [
        "report_only_treatment_match",
        "report_only_cancer_match",
        "report_only_subtype_match",
        "report_only_biomarker_match",
        "report_only_regimen_match",
        "report_only_stage_match",
        "report_only_line_of_therapy_match",
        "report_only_perturbed_compartment_match",
        "report_only_endpoint_category_match",
    ]
    used = pd.DataFrame.from_records(
        used_records,
        columns=[
            *frame.columns,
            *[column for column in match_columns if column not in frame.columns],
            "report_only_independence_unit_id",
        ],
    )
    if not used.empty:
        used["report_only_independence_verified"] = independence_verified
    excluded = pd.DataFrame.from_records(
        exclusions,
        columns=[
            "evidence_id",
            "reason",
            "report_only_treatment_match",
            "report_only_cancer_match",
            "report_only_subtype_match",
            "report_only_biomarker_match",
            "report_only_regimen_match",
            "report_only_stage_match",
            "report_only_line_of_therapy_match",
            "report_only_perturbed_compartment_match",
            "report_only_endpoint_category_match",
        ],
    )
    return used, excluded, independence_verified


def _preclinical_model_lane(model: str) -> str:
    if model in {
        PreclinicalModelType.CELL_LINE_2D,
        PreclinicalModelType.CELL_LINE_3D,
        PreclinicalModelType.IMMUNE_COCULTURE,
    }:
        return "in_vitro"
    if model in {
        PreclinicalModelType.ORGANOID,
        PreclinicalModelType.PDX_DERIVED_ORGANOID,
        PreclinicalModelType.EX_VIVO_TISSUE,
    }:
        return "organoid_ex_vivo"
    if model in {
        PreclinicalModelType.CELL_LINE_XENOGRAFT,
        PreclinicalModelType.PDX,
        PreclinicalModelType.SYNGENEIC,
        PreclinicalModelType.GENETICALLY_ENGINEERED_MODEL,
        PreclinicalModelType.HUMANIZED_MOUSE,
        PreclinicalModelType.OTHER_IN_VIVO,
    }:
        return "in_vivo"
    return "other"


def _family_count(frame: pd.DataFrame, mask: pd.Series | None = None) -> int:
    if frame.empty:
        return 0
    selected = frame if mask is None else frame.loc[mask]
    if selected.empty:
        return 0
    return int(selected["report_only_independence_unit_id"].nunique())


def _exact_context_mask(
    frame: pd.DataFrame,
    *,
    include_clinical_setting: bool,
    include_preclinical_design: bool,
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool, index=frame.index)
    exact = (
        frame["report_only_treatment_match"].eq("exact_id")
        & frame["report_only_cancer_match"].eq("exact_id")
        & frame["report_only_subtype_match"].isin(["exact_id", "exact_canonical"])
        & frame["report_only_biomarker_match"].eq("exact_typed")
        & frame["report_only_regimen_match"].eq("exact_active_components")
    )
    if include_clinical_setting:
        exact &= frame["report_only_stage_match"].eq("exact")
        exact &= frame["report_only_line_of_therapy_match"].eq("exact")
    if include_preclinical_design:
        exact &= frame["report_only_perturbed_compartment_match"].eq("exact")
        exact &= frame["report_only_endpoint_category_match"].eq("exact")
    return exact


def _is_missing_scalar(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _conflicting_context_mask(
    frame: pd.DataFrame,
    *,
    include_clinical_setting: bool,
    include_preclinical_design: bool,
) -> pd.Series:
    """Identify explicit contradictions, keeping missing/unverified rows separate."""

    if frame.empty:
        return pd.Series(dtype=bool, index=frame.index)
    conflicting = (
        frame["report_only_subtype_match"].isin(["conflict_or_unmapped", "id_conflict"])
        | frame["report_only_subtype_match"].eq("parent_id_conflict")
        | frame["report_only_biomarker_match"].isin(
            ["conflict_or_unmapped", "typed_axis_conflict"]
        )
        | frame["report_only_regimen_match"].isin(
            [
                "additional_active_components",
                "missing_active_components",
                "different_active_components",
                "relation_conflict",
            ]
        )
    )
    if include_clinical_setting:
        conflicting |= frame["report_only_stage_match"].eq(
            "different_or_unmapped"
        ) | frame["report_only_line_of_therapy_match"].eq("different_or_unmapped")
    if include_preclinical_design:
        conflicting |= frame["report_only_perturbed_compartment_match"].eq(
            "different"
        ) | frame["report_only_endpoint_category_match"].eq("different")
    return conflicting


def _candidate_context_summary(
    candidates: pd.DataFrame | None,
    preclinical: pd.DataFrame,
    patient: pd.DataFrame,
    *,
    context: TreatmentDiseaseContextRecord,
    preclinical_independence_verified: bool,
    patient_independence_verified: bool,
) -> pd.DataFrame:
    report_columns = [
        "report_only_preclinical_record_n",
        "report_only_preclinical_family_n",
        "report_only_preclinical_exact_context_family_n",
        "report_only_preclinical_compatible_nonexact_context_family_n",
        "report_only_preclinical_conflicting_context_family_n",
        "report_only_preclinical_exact_subtype_family_n",
        "report_only_preclinical_unspecified_subtype_family_n",
        "report_only_preclinical_exact_regimen_family_n",
        "report_only_preclinical_unspecified_regimen_family_n",
        "report_only_preclinical_in_vitro_exact_context_family_n",
        "report_only_preclinical_organoid_ex_vivo_exact_context_family_n",
        "report_only_preclinical_in_vivo_exact_context_family_n",
        "report_only_preclinical_direct_interaction_exact_context_family_n",
        "report_only_preclinical_perturbation_comparable_exact_context_family_n",
        "report_only_preclinical_direction_concordant_exact_context_family_n",
        "report_only_preclinical_direction_discordant_exact_context_family_n",
        "report_only_preclinical_direction_unresolved_exact_context_family_n",
        "report_only_preclinical_direction_incomparable_exact_context_family_n",
        "report_only_preclinical_status",
        "report_only_patient_record_n",
        "report_only_patient_family_n",
        "report_only_patient_exact_context_family_n",
        "report_only_patient_compatible_nonexact_context_family_n",
        "report_only_patient_conflicting_context_family_n",
        "report_only_patient_predictive_exact_context_family_n",
        "report_only_patient_predictive_compatible_nonexact_context_family_n",
        "report_only_patient_predictive_conflicting_context_family_n",
        "report_only_patient_interaction_null_exact_context_family_n",
        "report_only_patient_interaction_null_compatible_nonexact_context_family_n",
        "report_only_patient_interaction_null_conflicting_context_family_n",
        "report_only_patient_interaction_inconclusive_exact_context_family_n",
        "report_only_patient_interaction_inconclusive_compatible_nonexact_"
        "context_family_n",
        "report_only_patient_interaction_inconclusive_conflicting_context_family_n",
        "report_only_patient_interaction_unsupported_exact_context_family_n",
        "report_only_patient_interaction_unsupported_compatible_nonexact_"
        "context_family_n",
        "report_only_patient_interaction_unsupported_conflicting_context_family_n",
        "report_only_patient_treated_association_exact_context_family_n",
        "report_only_patient_prognostic_exact_context_family_n",
        "report_only_patient_pharmacodynamic_exact_context_family_n",
        "report_only_patient_acquired_resistance_exact_context_family_n",
        "report_only_patient_on_treatment_association_exact_context_family_n",
        "report_only_patient_post_progression_association_exact_context_family_n",
        "report_only_patient_status",
    ]
    if candidates is None:
        return pd.DataFrame(columns=["gene_symbol", *report_columns])
    if candidates.empty:
        raise TranslationContextError("candidate table cannot be empty")
    required = {"gene_symbol", "phenotype_direction"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise TranslationContextError(f"candidate table is missing columns: {missing}")
    reserved = sorted(
        column for column in candidates.columns if column.startswith("report_only_")
    )
    if reserved:
        raise TranslationContextError(
            f"candidate input contains reserved report columns: {reserved}"
        )
    gene_values = candidates["gene_symbol"]
    if gene_values.isna().any() or gene_values.astype(str).str.strip().eq("").any():
        raise TranslationContextError("candidate gene_symbol values cannot be empty")
    normalized_genes = gene_values.astype(str).str.strip()
    try:
        normalized_directions = candidates["phenotype_direction"].map(
            lambda value: PhenotypeDirection(str(value)).value
        )
    except ValueError as exc:
        raise TranslationContextError(
            "candidate phenotype_direction contains an unsupported value"
        ) from exc
    has_rank = "screen_signal_rank" in candidates.columns
    has_ranking_type = "ranking_type" in candidates.columns
    if has_rank != has_ranking_type:
        raise TranslationContextError(
            "candidate screen_signal_rank and ranking_type must be supplied together"
        )
    if has_ranking_type:
        ranking_types = candidates["ranking_type"].astype(str)
        if not ranking_types.eq("screen_signal_baseline").all():
            raise TranslationContextError(
                "candidate ranking_type must be screen_signal_baseline"
            )
        numeric_rank = pd.to_numeric(candidates["screen_signal_rank"], errors="coerce")
        directional = normalized_directions.isin(
            [
                PhenotypeDirection.RESISTANCE.value,
                PhenotypeDirection.SENSITIZATION.value,
            ]
        )
        boolean_rank = candidates["screen_signal_rank"].map(
            lambda value: isinstance(value, (bool, np.bool_))
        )
        raw_rank_missing = candidates["screen_signal_rank"].map(_is_missing_scalar)
        directional_rank = numeric_rank.loc[directional]
        if (
            boolean_rank.any()
            or directional_rank.isna().any()
            or not directional_rank.map(math.isfinite).all()
            or (directional_rank < 1).any()
            or not directional_rank.map(lambda value: float(value).is_integer()).all()
            or not raw_rank_missing.loc[~directional].all()
        ):
            raise TranslationContextError(
                "directional candidate screen_signal_rank must be a finite positive "
                "integer and non-directional candidates must be unranked"
            )
        rank_audit = pd.DataFrame(
            {
                "direction": normalized_directions,
                "rank": numeric_rank,
            }
        )
        for _, group in rank_audit.loc[directional].groupby("direction", sort=False):
            if not group["rank"].is_monotonic_increasing:
                raise TranslationContextError(
                    "candidate input order conflicts with screen_signal_rank"
                )
    identifier_columns = {"screen_id", "contrast_id"} & set(candidates.columns)
    if identifier_columns and identifier_columns != {"screen_id", "contrast_id"}:
        raise TranslationContextError(
            "candidate screen_id and contrast_id must be supplied together"
        )
    if context.screen_id and identifier_columns != {"screen_id", "contrast_id"}:
        raise TranslationContextError(
            "screen-bound context requires candidate screen_id and contrast_id"
        )
    if context.screen_id is None and identifier_columns:
        raise TranslationContextError(
            "candidate screen identifiers require a screen-bound context"
        )
    if context.screen_id:
        if not candidates["screen_id"].astype(str).eq(context.screen_id).all():
            raise TranslationContextError("candidate screen_id conflicts with context")
        if not candidates["contrast_id"].astype(str).eq(context.contrast_id).all():
            raise TranslationContextError(
                "candidate contrast_id conflicts with context"
            )
    candidate_key = pd.DataFrame(
        {
            "gene_symbol": normalized_genes,
            "phenotype_direction": normalized_directions,
        },
        index=candidates.index,
    )
    if identifier_columns:
        candidate_key["screen_id"] = candidates["screen_id"].astype(str)
        candidate_key["contrast_id"] = candidates["contrast_id"].astype(str)
    if candidate_key.duplicated().any():
        raise TranslationContextError(
            "candidate table contains duplicate candidate keys"
        )
    result_rows = []
    for position, (_, candidate) in enumerate(candidates.iterrows()):
        gene = normalized_genes.iloc[position]
        direction = normalized_directions.iloc[position]
        gene_preclinical = (
            preclinical.loc[
                preclinical.get("gene_symbol", pd.Series(dtype=object)).astype(str)
                == gene
            ]
            if not preclinical.empty and "gene_symbol" in preclinical
            else pd.DataFrame()
        )
        if not gene_preclinical.empty:
            gene_preclinical = gene_preclinical.copy()
            gene_preclinical["report_only_model_lane"] = gene_preclinical[
                "model_type"
            ].map(_preclinical_model_lane)
            evidence_direction = gene_preclinical["phenotype_direction"].astype(str)
            comparable_perturbation = (
                gene_preclinical["claim_type"]
                .astype(str)
                .eq(PreclinicalClaimType.DIRECT_PERTURBATIONAL_INTERACTION)
                & gene_preclinical["perturbation_modality"]
                .astype(str)
                .eq(context.screen_perturbation_modality.value)
                & gene_preclinical["report_only_perturbed_compartment_match"].eq(
                    "exact"
                )
                & gene_preclinical["report_only_endpoint_category_match"].eq("exact")
            )
            informative_direction = evidence_direction.isin(
                [
                    PhenotypeDirection.RESISTANCE.value,
                    PhenotypeDirection.SENSITIZATION.value,
                ]
            )
            supported_direction = gene_preclinical["direction_inference_status"].astype(
                str
            ).eq(
                PreclinicalDirectionInferenceStatus.DIRECTION_SUPPORTED
            ) & gene_preclinical["direction_inference_curator_verified"].eq(True)
            candidate_direction_is_informative = direction in {
                PhenotypeDirection.RESISTANCE.value,
                PhenotypeDirection.SENSITIZATION.value,
            }
            direction_comparable = (
                comparable_perturbation
                & informative_direction
                & supported_direction
                & candidate_direction_is_informative
            )
            concordant = direction_comparable & (evidence_direction == direction)
            opposite = {
                PhenotypeDirection.RESISTANCE.value: (
                    PhenotypeDirection.SENSITIZATION.value
                ),
                PhenotypeDirection.SENSITIZATION.value: (
                    PhenotypeDirection.RESISTANCE.value
                ),
            }.get(direction)
            discordant = (
                direction_comparable & (evidence_direction == opposite)
                if opposite
                else pd.Series(False, index=gene_preclinical.index)
            )
            unresolved = comparable_perturbation & ~(concordant | discordant)
            incomparable = ~comparable_perturbation
        else:
            comparable_perturbation = pd.Series(dtype=bool)
            concordant = discordant = unresolved = incomparable = pd.Series(dtype=bool)
        exact_subtype = (
            gene_preclinical["report_only_subtype_match"].isin(
                ["exact_id", "exact_canonical"]
            )
            if not gene_preclinical.empty
            else pd.Series(dtype=bool)
        )
        unspecified_subtype = (
            gene_preclinical["report_only_subtype_match"].eq("unspecified")
            if not gene_preclinical.empty
            else pd.Series(dtype=bool)
        )
        exact_preclinical_context = _exact_context_mask(
            gene_preclinical,
            include_clinical_setting=False,
            include_preclinical_design=True,
        )
        conflicting_preclinical_context = _conflicting_context_mask(
            gene_preclinical,
            include_clinical_setting=False,
            include_preclinical_design=True,
        )
        compatible_nonexact_preclinical_context = ~(
            exact_preclinical_context | conflicting_preclinical_context
        )
        exact_regimen = (
            gene_preclinical["report_only_regimen_match"].eq("exact_active_components")
            if not gene_preclinical.empty
            else pd.Series(dtype=bool)
        )
        unspecified_regimen = (
            gene_preclinical["report_only_regimen_match"].eq("unspecified")
            if not gene_preclinical.empty
            else pd.Series(dtype=bool)
        )
        if gene_preclinical.empty:
            preclinical_status = "no_match_in_provided_curated_evidence"
        elif not preclinical_independence_verified:
            preclinical_status = "independence_unverified"
        elif _family_count(gene_preclinical, exact_preclinical_context):
            preclinical_status = "exact_context_curated_evidence_present"
        elif _family_count(
            gene_preclinical, compatible_nonexact_preclinical_context
        ) and _family_count(gene_preclinical, conflicting_preclinical_context):
            preclinical_status = "compatible_and_conflicting_context_present"
        elif _family_count(gene_preclinical, compatible_nonexact_preclinical_context):
            preclinical_status = "compatible_nonexact_context_only"
        elif _family_count(gene_preclinical, conflicting_preclinical_context):
            preclinical_status = "conflicting_context_only"
        else:
            preclinical_status = "descriptive_or_unresolved_context"

        gene_patient = (
            patient.loc[patient["gene_symbol"].astype(str) == gene]
            if not patient.empty and "gene_symbol" in patient
            else pd.DataFrame()
        )
        interpretation = (
            gene_patient["association_interpretation"].astype(str)
            if not gene_patient.empty
            else pd.Series(dtype="string")
        )
        exact_patient_context = _exact_context_mask(
            gene_patient,
            include_clinical_setting=True,
            include_preclinical_design=False,
        )
        conflicting_patient_context = _conflicting_context_mask(
            gene_patient,
            include_clinical_setting=True,
            include_preclinical_design=False,
        )
        compatible_nonexact_patient_context = ~(
            exact_patient_context | conflicting_patient_context
        )
        inference_status = (
            gene_patient["interaction_inference_status"].astype(str)
            if not gene_patient.empty
            else pd.Series(dtype="string")
        )
        predictive_mask = interpretation.eq(
            PatientAssociationInterpretation.PREDICTIVE_INTERACTION
        ) & inference_status.eq(InteractionInferenceStatus.SUPPORTED)
        null_interaction_mask = interpretation.eq(
            PatientAssociationInterpretation.INTERACTION_TESTED_NULL
        ) & inference_status.eq(InteractionInferenceStatus.NULL)
        inconclusive_interaction_mask = interpretation.eq(
            PatientAssociationInterpretation.INTERACTION_TESTED_INCONCLUSIVE
        ) & inference_status.eq(InteractionInferenceStatus.INCONCLUSIVE)
        unsupported_interaction_mask = interpretation.eq(
            PatientAssociationInterpretation.INTERACTION_TESTED_UNSUPPORTED
        ) & inference_status.eq(InteractionInferenceStatus.UNSUPPORTED)
        predictive_exact_n = _family_count(
            gene_patient,
            predictive_mask & exact_patient_context,
        )
        predictive_compatible_nonexact_n = _family_count(
            gene_patient,
            predictive_mask & compatible_nonexact_patient_context,
        )
        predictive_conflicting_n = _family_count(
            gene_patient,
            predictive_mask & conflicting_patient_context,
        )
        interaction_null_exact_n = _family_count(
            gene_patient,
            null_interaction_mask & exact_patient_context,
        )
        interaction_null_compatible_nonexact_n = _family_count(
            gene_patient,
            null_interaction_mask & compatible_nonexact_patient_context,
        )
        interaction_null_conflicting_n = _family_count(
            gene_patient,
            null_interaction_mask & conflicting_patient_context,
        )
        interaction_inconclusive_exact_n = _family_count(
            gene_patient,
            inconclusive_interaction_mask & exact_patient_context,
        )
        interaction_inconclusive_compatible_nonexact_n = _family_count(
            gene_patient,
            inconclusive_interaction_mask & compatible_nonexact_patient_context,
        )
        interaction_inconclusive_conflicting_n = _family_count(
            gene_patient,
            inconclusive_interaction_mask & conflicting_patient_context,
        )
        interaction_unsupported_exact_n = _family_count(
            gene_patient,
            unsupported_interaction_mask & exact_patient_context,
        )
        interaction_unsupported_compatible_nonexact_n = _family_count(
            gene_patient,
            unsupported_interaction_mask & compatible_nonexact_patient_context,
        )
        interaction_unsupported_conflicting_n = _family_count(
            gene_patient,
            unsupported_interaction_mask & conflicting_patient_context,
        )
        treated_n = _family_count(
            gene_patient,
            interpretation.eq(
                PatientAssociationInterpretation.TREATED_COHORT_ASSOCIATION
            )
            & exact_patient_context,
        )
        prognostic_n = _family_count(
            gene_patient,
            interpretation.eq(PatientAssociationInterpretation.PROGNOSTIC_ONLY)
            & exact_patient_context,
        )
        pharmacodynamic_n = _family_count(
            gene_patient,
            interpretation.eq(PatientAssociationInterpretation.PHARMACODYNAMIC)
            & exact_patient_context,
        )
        acquired_n = _family_count(
            gene_patient,
            interpretation.eq(PatientAssociationInterpretation.ACQUIRED_RESISTANCE)
            & exact_patient_context,
        )
        on_treatment_n = _family_count(
            gene_patient,
            interpretation.eq(PatientAssociationInterpretation.ON_TREATMENT_ASSOCIATION)
            & exact_patient_context,
        )
        post_progression_n = _family_count(
            gene_patient,
            interpretation.eq(
                PatientAssociationInterpretation.POST_PROGRESSION_ASSOCIATION
            )
            & exact_patient_context,
        )
        if gene_patient.empty:
            patient_status = "insufficient_matched_patient_data"
        elif not patient_independence_verified:
            patient_status = "independence_unverified"
        elif predictive_exact_n and (
            interaction_null_exact_n
            or interaction_inconclusive_exact_n
            or interaction_unsupported_exact_n
        ):
            patient_status = "supported_and_nonconfirmatory_interactions_present"
        elif predictive_exact_n:
            patient_status = "predictive_interaction_evidence_present"
        elif (
            sum(
                bool(value)
                for value in (
                    interaction_null_exact_n,
                    interaction_inconclusive_exact_n,
                    interaction_unsupported_exact_n,
                )
            )
            > 1
        ):
            patient_status = "multiple_nonconfirmatory_interaction_results_present"
        elif interaction_null_exact_n:
            patient_status = "interaction_tested_null_present"
        elif interaction_inconclusive_exact_n:
            patient_status = "interaction_tested_inconclusive_present"
        elif interaction_unsupported_exact_n:
            patient_status = "interaction_tested_unsupported_present"
        elif treated_n:
            patient_status = "treated_cohort_association_present"
        elif (
            prognostic_n
            or pharmacodynamic_n
            or acquired_n
            or on_treatment_n
            or post_progression_n
        ):
            patient_status = "nonpredictive_exact_patient_context_present"
        elif _family_count(
            gene_patient, compatible_nonexact_patient_context
        ) and _family_count(gene_patient, conflicting_patient_context):
            patient_status = "compatible_and_conflicting_patient_context_present"
        elif _family_count(gene_patient, compatible_nonexact_patient_context):
            patient_status = "compatible_nonexact_patient_context_only"
        elif _family_count(gene_patient, conflicting_patient_context):
            patient_status = "conflicting_patient_context_only"
        else:
            patient_status = "descriptive_or_unresolved_patient_context"

        result_rows.append(
            {
                **candidate.to_dict(),
                "gene_symbol": gene,
                "phenotype_direction": direction,
                "report_only_preclinical_record_n": len(gene_preclinical),
                "report_only_preclinical_family_n": _family_count(gene_preclinical),
                "report_only_preclinical_exact_context_family_n": _family_count(
                    gene_preclinical, exact_preclinical_context
                ),
                (
                    "report_only_preclinical_compatible_nonexact_context_family_n"
                ): _family_count(
                    gene_preclinical, compatible_nonexact_preclinical_context
                ),
                "report_only_preclinical_conflicting_context_family_n": _family_count(
                    gene_preclinical, conflicting_preclinical_context
                ),
                "report_only_preclinical_exact_subtype_family_n": _family_count(
                    gene_preclinical, exact_subtype
                ),
                "report_only_preclinical_unspecified_subtype_family_n": _family_count(
                    gene_preclinical, unspecified_subtype
                ),
                "report_only_preclinical_exact_regimen_family_n": _family_count(
                    gene_preclinical, exact_regimen
                ),
                "report_only_preclinical_unspecified_regimen_family_n": _family_count(
                    gene_preclinical, unspecified_regimen
                ),
                (
                    "report_only_preclinical_in_vitro_exact_context_family_n"
                ): _family_count(
                    gene_preclinical,
                    exact_preclinical_context
                    & gene_preclinical.get(
                        "report_only_model_lane", pd.Series(dtype="string")
                    ).eq("in_vitro"),
                ),
                (
                    "report_only_preclinical_organoid_ex_vivo_exact_context_family_n"
                ): _family_count(
                    gene_preclinical,
                    exact_preclinical_context
                    & gene_preclinical.get(
                        "report_only_model_lane", pd.Series(dtype="string")
                    ).eq("organoid_ex_vivo"),
                ),
                "report_only_preclinical_in_vivo_exact_context_family_n": _family_count(
                    gene_preclinical,
                    exact_preclinical_context
                    & gene_preclinical.get(
                        "report_only_model_lane", pd.Series(dtype="string")
                    ).eq("in_vivo"),
                ),
                (
                    "report_only_preclinical_direct_interaction_exact_context_family_n"
                ): _family_count(
                    gene_preclinical,
                    exact_preclinical_context
                    & (
                        gene_preclinical.get("claim_type", pd.Series(dtype="string"))
                        .astype(str)
                        .eq(PreclinicalClaimType.DIRECT_PERTURBATIONAL_INTERACTION)
                    ),
                ),
                (
                    "report_only_preclinical_perturbation_comparable_"
                    "exact_context_family_n"
                ): _family_count(
                    gene_preclinical,
                    exact_preclinical_context & comparable_perturbation,
                ),
                (
                    "report_only_preclinical_direction_concordant_"
                    "exact_context_family_n"
                ): _family_count(
                    gene_preclinical, concordant & exact_preclinical_context
                ),
                (
                    "report_only_preclinical_direction_discordant_"
                    "exact_context_family_n"
                ): _family_count(
                    gene_preclinical, discordant & exact_preclinical_context
                ),
                (
                    "report_only_preclinical_direction_unresolved_"
                    "exact_context_family_n"
                ): _family_count(
                    gene_preclinical, unresolved & exact_preclinical_context
                ),
                (
                    "report_only_preclinical_direction_incomparable_"
                    "exact_context_family_n"
                ): _family_count(
                    gene_preclinical, incomparable & exact_preclinical_context
                ),
                "report_only_preclinical_status": preclinical_status,
                "report_only_patient_record_n": len(gene_patient),
                "report_only_patient_family_n": _family_count(gene_patient),
                "report_only_patient_exact_context_family_n": _family_count(
                    gene_patient, exact_patient_context
                ),
                (
                    "report_only_patient_compatible_nonexact_context_family_n"
                ): _family_count(gene_patient, compatible_nonexact_patient_context),
                "report_only_patient_conflicting_context_family_n": _family_count(
                    gene_patient, conflicting_patient_context
                ),
                (
                    "report_only_patient_predictive_exact_context_family_n"
                ): predictive_exact_n,
                (
                    "report_only_patient_predictive_compatible_nonexact_context_family_n"
                ): predictive_compatible_nonexact_n,
                (
                    "report_only_patient_predictive_conflicting_context_family_n"
                ): predictive_conflicting_n,
                (
                    "report_only_patient_interaction_null_exact_context_family_n"
                ): interaction_null_exact_n,
                (
                    "report_only_patient_interaction_null_compatible_nonexact_context_family_n"
                ): interaction_null_compatible_nonexact_n,
                (
                    "report_only_patient_interaction_null_conflicting_context_family_n"
                ): interaction_null_conflicting_n,
                (
                    "report_only_patient_interaction_inconclusive_exact_context_family_n"
                ): interaction_inconclusive_exact_n,
                (
                    "report_only_patient_interaction_inconclusive_"
                    "compatible_nonexact_context_family_n"
                ): interaction_inconclusive_compatible_nonexact_n,
                (
                    "report_only_patient_interaction_inconclusive_"
                    "conflicting_context_family_n"
                ): interaction_inconclusive_conflicting_n,
                (
                    "report_only_patient_interaction_unsupported_exact_context_family_n"
                ): interaction_unsupported_exact_n,
                (
                    "report_only_patient_interaction_unsupported_"
                    "compatible_nonexact_context_family_n"
                ): interaction_unsupported_compatible_nonexact_n,
                (
                    "report_only_patient_interaction_unsupported_"
                    "conflicting_context_family_n"
                ): interaction_unsupported_conflicting_n,
                (
                    "report_only_patient_treated_association_exact_context_family_n"
                ): treated_n,
                ("report_only_patient_prognostic_exact_context_family_n"): prognostic_n,
                (
                    "report_only_patient_pharmacodynamic_exact_context_family_n"
                ): pharmacodynamic_n,
                (
                    "report_only_patient_acquired_resistance_exact_context_family_n"
                ): acquired_n,
                (
                    "report_only_patient_on_treatment_association_"
                    "exact_context_family_n"
                ): on_treatment_n,
                (
                    "report_only_patient_post_progression_association_"
                    "exact_context_family_n"
                ): post_progression_n,
                "report_only_patient_status": patient_status,
            }
        )
    return pd.DataFrame.from_records(result_rows)


def _core_model_applicability(
    context: TreatmentDiseaseContextRecord,
) -> tuple[str, list[str]]:
    reasons = []
    if context.treatment_modality != InterventionModality.SMALL_MOLECULE:
        reasons.append("intervention_not_small_molecule")
    if context.screen_perturbation_modality != PerturbationModality.CRISPR_KO:
        reasons.append("screen_not_crispr_ko")
    if context.perturbed_compartment != PerturbedCompartment.TUMOR_CELL:
        reasons.append("perturbed_compartment_not_tumor_cell")
    if context.screen_endpoint_category != (
        ScreenEndpointCategory.DRUG_RESPONSE_VIABILITY
    ):
        reasons.append("endpoint_not_drug_response_viability")
    return ("in_scope", []) if not reasons else ("not_applicable", reasons)


def _lane_status(
    frame: pd.DataFrame | None,
    used: pd.DataFrame,
    *,
    independence_verified: bool,
) -> str:
    if frame is None:
        return "not_provided"
    if used.empty:
        return "no_match_in_provided_curated_table"
    if not independence_verified:
        return "independence_unverified"
    return "matched_curated_evidence_present"


def _clinical_trial_match_masks(
    clinical_trials: pd.DataFrame,
    context: TreatmentDiseaseContextRecord,
) -> tuple[pd.Series, pd.Series]:
    if clinical_trials.empty:
        empty = pd.Series(dtype=bool, index=clinical_trials.index)
        return empty, empty
    retrieved_candidate = clinical_trials["intervention_match"].ne(
        TrialInterventionMatch.NO_STRUCTURED_MATCH
    ) & clinical_trials["disease_match"].ne(TrialDiseaseMatch.NO_STRUCTURED_MATCH)
    strict_structured_candidate = retrieved_candidate & clinical_trials[
        "intervention_match"
    ].isin(
        [
            TrialInterventionMatch.EXACT_CANONICAL,
        ]
    )
    # A matching display name is discovery context, not ontology-resolved
    # identity. Strict registry candidates require versioned canonical IDs in
    # the requested context even though the registry itself exposes text.
    if not context.treatment_id or not context.cancer_id:
        strict_structured_candidate &= False
    if context.disease_subtype:
        strict_structured_candidate &= clinical_trials["disease_match"].eq(
            TrialDiseaseMatch.EXPLICIT_SUBTYPE_TERM
        )
        strict_structured_candidate &= (
            context.disease_subtype_parent_binding_verified is True
        )
    else:
        strict_structured_candidate &= clinical_trials["disease_match"].eq(
            TrialDiseaseMatch.CANCER_TYPE_TERM_ONLY
        )
    if context.biomarker_context:
        # Registry terms do not encode the typed feature/state/specimen/timepoint
        # required by the curated evidence contracts.
        strict_structured_candidate &= False
    if context.regimen_name:
        # The registry adapter currently sees study-level intervention lists, not
        # verified arm-level coassignment to the requested regimen.
        strict_structured_candidate &= False
    if context.stage or context.line_of_therapy:
        # Stage and line-of-therapy are not parsed from structured arm assignment
        # or eligibility data by this adapter, so they cannot support strictness.
        strict_structured_candidate &= False
    return retrieved_candidate, strict_structured_candidate


def _build_report_markdown(
    context: TreatmentDiseaseContextRecord,
    clinical_trials: pd.DataFrame,
    missingness: pd.DataFrame,
    *,
    snapshot: ClinicalTrialsSnapshot,
    query_context_binding: str,
    core_status: str,
    core_reasons: list[str],
) -> str:
    intervention_matches = (
        int(
            clinical_trials["intervention_match"]
            .ne(TrialInterventionMatch.NO_STRUCTURED_MATCH)
            .sum()
        )
        if not clinical_trials.empty
        else 0
    )
    subtype_matches = (
        int(
            clinical_trials["disease_match"]
            .eq(TrialDiseaseMatch.EXPLICIT_SUBTYPE_TERM)
            .sum()
        )
        if not clinical_trials.empty
        else 0
    )
    results_n = (
        int(clinical_trials["has_results"].sum()) if not clinical_trials.empty else 0
    )
    _, strict_structured_mask = _clinical_trial_match_masks(clinical_trials, context)
    unbound_strict_structured_n = int(strict_structured_mask.sum())
    source_verified = snapshot.source_mode in {
        "live_api",
        "live_api_declared_query_set",
    }
    strict_structured_n = (
        unbound_strict_structured_n
        if (
            query_context_binding == "verified_typed_query_cross_product"
            and source_verified
            and snapshot.version_stable
            and snapshot.complete
        )
        else 0
    )
    lines = [
        "# Translation-context report",
        "",
        f"- Treatment: `{context.treatment_name}`",
        f"- Cancer: `{context.cancer_type}`",
        f"- Subtype: `{context.disease_subtype or 'not specified'}`",
        f"- Current ClinicalTrials.gov records retrieved: **{len(clinical_trials)}**",
        f"- Records with a structured intervention match: **{intervention_matches}**",
        f"- Records with an explicit structured subtype term: **{subtype_matches}**",
        f"- Records with posted aggregate results: **{results_n}**",
        (
            "- Records satisfying all registry-resolvable requested axes: "
            f"**{strict_structured_n}**"
        ),
        f"- Declared query-set pagination complete: **{snapshot.complete}**",
        f"- Declared query/context binding: **{query_context_binding}**",
        "- Exhaustive ontology-concept recall guaranteed: **False**",
        f"- Core reproducibility model applicability: **{core_status}**",
    ]
    if core_reasons:
        lines.append(f"- Applicability reasons: `{', '.join(core_reasons)}`")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "Trial phase, status, enrollment, and trial counts describe the "
                "treatment/disease landscape. They do not rank genes and are not "
                "validation labels. ClinicalTrials.gov does not establish "
                "patient-level RNA-seq plus treatment outcome for a gene merely "
                "because a trial record exists."
            ),
            "",
            (
                "A `treated_cohort_association` is not called predictive. Predictive "
                "evidence requires a pretreatment measurement, a comparator arm, "
                "and a formal treatment-by-gene-predictor interaction test. "
                "Current API "
                "records are mutable current snapshots and are not eligible as "
                "historical model features without a version reconstructed at the "
                "cutoff."
            ),
            "",
            (
                "Only a verified typed query/context binding establishes that every "
                "declared canonical, entity-alias, class, subtype, and ancestor lane "
                "was paginated for this requested context. It still does not prove "
                "exhaustive recall for undisclosed synonyms or unstructured "
                "eligibility text."
            ),
            "",
            "## Evidence-lane status",
            "",
            "| Lane | Status | Records |",
            "|---|---|---:|",
        ]
    )
    for _, row in missingness.iterrows():
        lines.append(f"| {row['lane']} | {row['status']} | {row['record_n']} |")
    lines.append("")
    return "\n".join(lines)


def _revalidate_snapshot_for_report(
    snapshot: ClinicalTrialsSnapshot,
) -> ClinicalTrialsSnapshot:
    """Bind report construction to a replay-valid immutable snapshot view."""

    if not isinstance(snapshot, ClinicalTrialsSnapshot):
        raise TranslationContextError("snapshot must be a ClinicalTrialsSnapshot")
    allowed_source_modes = {
        "frozen_json",
        "live_api",
        "live_api_declared_query_set",
        "injected_transport",
        "injected_transport_declared_query_set",
        "injected_clock",
        "injected_clock_declared_query_set",
    }
    if snapshot.source_mode not in allowed_source_modes:
        raise TranslationContextError("ClinicalTrialsSnapshot source_mode is invalid")
    replayed = clinical_trials_snapshot_from_document(snapshot.document)
    comparable_snapshot_document = _json_deep_copy(snapshot.document)
    comparable_snapshot_document["version_stable"] = False
    metadata_matches = (
        comparable_snapshot_document == replayed.document
        and snapshot.studies == replayed.studies
        and snapshot.request_urls == replayed.request_urls
        and snapshot.total_count == replayed.total_count
        and snapshot.complete == replayed.complete
        and snapshot.retrieved_at_utc == replayed.retrieved_at_utc
        and snapshot.api_version == replayed.api_version
        and snapshot.data_timestamp == replayed.data_timestamp
    )
    if not metadata_matches:
        raise TranslationContextError(
            "ClinicalTrialsSnapshot metadata or studies changed after validation"
        )
    live_source = snapshot.source_mode in {
        "live_api",
        "live_api_declared_query_set",
    }
    required_audit_field = (
        "version_audit_set"
        if snapshot.source_mode == "live_api_declared_query_set"
        else "version_audit"
    )
    if live_source and (
        snapshot.version_stable is not True
        or required_audit_field not in replayed.document
        or not _valid_live_acquisition_witness(
            snapshot._live_acquisition_witness,
            source_mode=snapshot.source_mode,
            document_sha256=_canonical_page_sha256(snapshot.document),
        )
    ):
        raise TranslationContextError(
            "live ClinicalTrialsSnapshot lacks its in-process version audit"
        )
    if not live_source and snapshot.version_stable is not False:
        raise TranslationContextError(
            "frozen ClinicalTrialsSnapshot cannot self-attest version stability"
        )
    return ClinicalTrialsSnapshot(
        document=(
            _json_deep_copy(snapshot.document) if live_source else replayed.document
        ),
        studies=replayed.studies,
        request_urls=replayed.request_urls,
        total_count=replayed.total_count,
        complete=replayed.complete,
        retrieved_at_utc=replayed.retrieved_at_utc,
        source_mode=snapshot.source_mode,
        api_version=replayed.api_version,
        data_timestamp=replayed.data_timestamp,
        version_stable=live_source,
        _live_acquisition_witness=(
            snapshot._live_acquisition_witness if live_source else None
        ),
    )


def build_translation_context_report(
    context: TreatmentDiseaseContextRecord | dict[str, Any],
    snapshot: ClinicalTrialsSnapshot,
    *,
    candidates: pd.DataFrame | None = None,
    preclinical_evidence: pd.DataFrame | None = None,
    patient_evidence: pd.DataFrame | None = None,
    evidence_cutoff_date: date,
    treatment_entity_aliases: list[str] | tuple[str, ...] = (),
    treatment_class_terms: list[str] | tuple[str, ...] = (),
    cancer_entity_aliases: list[str] | tuple[str, ...] = (),
    cancer_ancestor_terms: list[str] | tuple[str, ...] = (),
    subtype_entity_aliases: list[str] | tuple[str, ...] = (),
    biomarker_aliases: list[str] | tuple[str, ...] = (),
    target_source_family_id: str | None = None,
    target_raw_data_family_id: str | None = None,
    target_absence_attested: bool = False,
) -> TranslationContextResult:
    """Build independent report lanes without changing candidate order."""

    if isinstance(target_absence_attested, np.bool_):
        target_absence_attested = bool(target_absence_attested)
    elif type(target_absence_attested) is not bool:
        raise TranslationContextError(
            "target_absence_attested must be a literal boolean"
        )
    snapshot = _revalidate_snapshot_for_report(snapshot)
    parsed_context = _revalidate_treatment_disease_context(context)
    if evidence_cutoff_date > parsed_context.context_date:
        raise TranslationContextError(
            "evidence_cutoff_date cannot be later than context_date"
        )
    if snapshot.retrieved_at_utc.date() > parsed_context.context_date:
        raise TranslationContextError(
            "ClinicalTrials.gov snapshot retrieval cannot postdate context_date"
        )
    query_context_binding = _typed_query_context_binding(
        snapshot,
        parsed_context,
        treatment_entity_aliases=treatment_entity_aliases,
        treatment_class_terms=treatment_class_terms,
        cancer_entity_aliases=cancer_entity_aliases,
        cancer_ancestor_terms=cancer_ancestor_terms,
        subtype_entity_aliases=subtype_entity_aliases,
    )
    clinical_trials = normalize_clinical_trials(
        snapshot,
        parsed_context,
        treatment_entity_aliases=treatment_entity_aliases,
        treatment_class_terms=treatment_class_terms,
        cancer_entity_aliases=cancer_entity_aliases,
        cancer_ancestor_terms=cancer_ancestor_terms,
        subtype_entity_aliases=subtype_entity_aliases,
        biomarker_aliases=biomarker_aliases,
    )
    preclinical = _validated_frame(preclinical_evidence, PreclinicalEvidenceRecord)
    patient = _validated_frame(patient_evidence, PatientMolecularEvidenceRecord)
    _validate_gene_identity_consistency(preclinical, patient)
    treatment_terms = _ordered_entity_terms(
        parsed_context.treatment_name, treatment_entity_aliases
    )
    normalized_treatment_class_terms = _declared_entity_terms(treatment_class_terms)
    cancer_terms = _ordered_entity_terms(
        parsed_context.cancer_type, cancer_entity_aliases
    )
    normalized_cancer_ancestor_terms = _declared_entity_terms(cancer_ancestor_terms)
    subtype_terms = (
        _ordered_entity_terms(parsed_context.disease_subtype, subtype_entity_aliases)
        if parsed_context.disease_subtype
        else []
    )
    biomarker_terms = (
        _ordered_entity_terms(parsed_context.biomarker_context, biomarker_aliases)
        if parsed_context.biomarker_context
        else []
    )
    preclinical_used, preclinical_exclusions, preclinical_independence = (
        _screen_evidence(
            preclinical,
            context=parsed_context,
            cutoff_date=evidence_cutoff_date,
            treatment_terms=treatment_terms,
            cancer_terms=cancer_terms,
            subtype_terms=subtype_terms,
            biomarker_terms=biomarker_terms,
            target_source_family_id=target_source_family_id,
            target_raw_data_family_id=target_raw_data_family_id,
            target_absence_attested=target_absence_attested,
        )
    )
    patient_used, patient_exclusions, patient_independence = _screen_evidence(
        patient,
        context=parsed_context,
        cutoff_date=evidence_cutoff_date,
        treatment_terms=treatment_terms,
        cancer_terms=cancer_terms,
        subtype_terms=subtype_terms,
        biomarker_terms=biomarker_terms,
        target_source_family_id=target_source_family_id,
        target_raw_data_family_id=target_raw_data_family_id,
        target_absence_attested=target_absence_attested,
    )
    candidate_context = _candidate_context_summary(
        candidates,
        preclinical_used,
        patient_used,
        context=parsed_context,
        preclinical_independence_verified=preclinical_independence,
        patient_independence_verified=patient_independence,
    )
    structured_trial_match, strict_structured_trial_match = _clinical_trial_match_masks(
        clinical_trials, parsed_context
    )
    source_verified = snapshot.source_mode in {
        "live_api",
        "live_api_declared_query_set",
    }
    if query_context_binding != "verified_typed_query_cross_product":
        clinical_status = "frozen_query_context_unverified"
    elif not source_verified:
        clinical_status = (
            "frozen_source_provenance_unverified"
            if snapshot.source_mode == "frozen_json"
            else "injected_source_provenance_unverified"
        )
    elif not snapshot.version_stable:
        clinical_status = "current_snapshot_version_unverified"
    elif not snapshot.complete:
        clinical_status = "truncated_current_snapshot"
    elif int(strict_structured_trial_match.sum()) > 0:
        clinical_status = "strict_structured_registry_candidates_present"
    elif int(structured_trial_match.sum()) > 0:
        clinical_status = "broader_or_unresolved_registry_matches_only"
    else:
        clinical_status = "no_structured_match_in_complete_declared_query_set"
    core_status, core_reasons = _core_model_applicability(parsed_context)
    missingness = pd.DataFrame.from_records(
        [
            {
                "lane": "clinical_trial_registry",
                "status": clinical_status,
                "record_n": len(clinical_trials),
                "interpretation": "treatment_disease_context_only",
            },
            {
                "lane": "matched_patient_molecular_outcome",
                "status": _lane_status(
                    patient_evidence,
                    patient_used,
                    independence_verified=patient_independence,
                ),
                "record_n": len(patient_used),
                "interpretation": "gene_level_curated_claims",
            },
            {
                "lane": "preclinical",
                "status": _lane_status(
                    preclinical_evidence,
                    preclinical_used,
                    independence_verified=preclinical_independence,
                ),
                "record_n": len(preclinical_used),
                "interpretation": "separate_model_lanes_no_hierarchy",
            },
            {
                "lane": "tumor_context",
                "status": "not_implemented_in_this_adapter",
                "record_n": 0,
                "interpretation": "must_not_be_called_treatment_response",
            },
            {
                "lane": "core_reproducibility_model",
                "status": core_status,
                "record_n": 0,
                "interpretation": ";".join(core_reasons) or "v1_scope_match",
            },
        ]
    )
    metadata = {
        "report_type": "translation_context_report_only",
        "method_version": TRANSLATION_CONTEXT_METHOD_VERSION,
        "context": parsed_context.model_dump(mode="json"),
        "evidence_cutoff_date": evidence_cutoff_date.isoformat(),
        "matching_terms": {
            "treatment": treatment_terms,
            "treatment_class_discovery": normalized_treatment_class_terms,
            "cancer": cancer_terms,
            "cancer_ancestor_discovery": normalized_cancer_ancestor_terms,
            "subtype": subtype_terms,
            "biomarker": biomarker_terms,
        },
        "clinicaltrials": {
            "source_api_major": CLINICALTRIALS_API_MAJOR,
            "source_api_version": snapshot.api_version,
            "data_timestamp": snapshot.data_timestamp,
            "version_stable": snapshot.version_stable,
            "source_mode": snapshot.source_mode,
            "retrieved_at_utc": snapshot.retrieved_at_utc.isoformat(),
            "request_urls": snapshot.request_urls,
            "reported_total_count": snapshot.total_count,
            "retrieved_unique_nct_count": len(clinical_trials),
            "structured_treatment_disease_candidate_count": int(
                structured_trial_match.sum()
            ),
            "strict_structured_registry_candidate_count": int(
                strict_structured_trial_match.sum()
                if (
                    query_context_binding == "verified_typed_query_cross_product"
                    and source_verified
                    and snapshot.version_stable
                    and snapshot.complete
                )
                else 0
            ),
            "unbound_registry_resolvable_candidate_count": int(
                strict_structured_trial_match.sum()
            ),
            "pagination_complete": snapshot.complete,
            "page_canonical_sha256": snapshot.document.get("page_canonical_sha256", []),
            "declared_query_set": snapshot.document.get("declared_query_set", []),
            "declared_query_context_binding": query_context_binding,
            "ontology_concept_recall_complete": snapshot.document.get(
                "ontology_concept_recall_complete", False
            ),
            "historical_feature_eligible": False,
            "patient_level_omics_inferred": False,
        },
        "independence": {
            "target_source_family_id": target_source_family_id,
            "target_raw_data_family_id": target_raw_data_family_id,
            "target_absence_attested": target_absence_attested,
            "preclinical_verified": preclinical_independence,
            "patient_verified": patient_independence,
        },
        "core_model_applicability": {
            "status": core_status,
            "reasons": core_reasons,
        },
        "candidate_input": {
            "provided": candidates is not None,
            "row_count": 0 if candidates is None else len(candidates),
            "order_semantics": "input_order_preserved_no_reranking",
            "ranking_claim": (
                "screen_signal_baseline_structure_validated_manifest_unbound"
                if candidates is not None and "ranking_type" in candidates.columns
                else "unranked_or_unverified_input_order"
            ),
        },
        "interpretation_boundary": (
            "report only; no validation label, no reproducibility feature, no "
            "candidate reranking, and no therapeutic recommendation"
        ),
    }
    report = _build_report_markdown(
        parsed_context,
        clinical_trials,
        missingness,
        snapshot=snapshot,
        query_context_binding=query_context_binding,
        core_status=core_status,
        core_reasons=core_reasons,
    )
    return TranslationContextResult(
        clinical_trials=clinical_trials,
        preclinical_used_evidence=preclinical_used,
        preclinical_exclusions=preclinical_exclusions,
        patient_used_evidence=patient_used,
        patient_exclusions=patient_exclusions,
        candidate_context=candidate_context,
        missingness=missingness,
        report_markdown=report,
        metadata=metadata,
        clinicaltrials_snapshot=snapshot.document,
    )
