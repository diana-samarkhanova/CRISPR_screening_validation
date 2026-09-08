"""Reproducible, fail-closed ClinicalTrials.gov API v2 snapshot intake.

The live search is a retrieval step only.  Query matches are emitted as an
unreviewed curation queue and can never become normalized clinical evidence,
gene-level support, a validation label, or a model feature automatically.
"""

from __future__ import annotations

import csv
import ctypes
import errno
import fcntl
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import pandas as pd

from . import __version__
from .contracts import (
    ClinicalTrialsGovCurationCandidateRecord,
    ClinicalTrialsGovSnapshotManifest,
    ClinicalTrialsGovStudyInventoryRecord,
    DataAssetRecord,
    validate_records,
)

API_ORIGIN = "https://clinicaltrials.gov"
API_BASE_URL = f"{API_ORIGIN}/api/v2"
STUDIES_ENDPOINT = f"{API_BASE_URL}/studies"
VERSION_ENDPOINT = f"{API_BASE_URL}/version"
API_DOCUMENTATION_URL = "https://clinicaltrials.gov/data-api/about-api"
TERMS_URL = "https://clinicaltrials.gov/about-site/terms-conditions"
DATA_TIMESTAMP_INTERPRETATION = (
    "raw value stored verbatim; interpreted as UTC only because the official API "
    "contract defines dataTimestamp as UTC"
)
SNAPSHOT_ISOLATION_CLAIM = (
    "not_asserted; stable version endpoint values are an envelope check, not proof "
    "of transactional isolation"
)
INTEGRITY_SCOPE = (
    "internal checksums and provenance; not publisher authenticity, historical "
    "reconstruction, or redistribution authorization"
)

DEFAULT_PAGE_SIZE = 1000
MAX_PAGE_SIZE = 1000
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF_SECONDS = 1.0
DEFAULT_MAX_PAGES = 10_000
HARD_MAX_PAGES = 10_000
DEFAULT_MAX_STUDIES = 1_000_000
DEFAULT_MAX_CANDIDATE_ROWS = 100_000
DEFAULT_MAX_PAGE_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_DERIVED_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_ELAPSED_SECONDS = 3600.0
HARD_MAX_PAGE_BYTES = 100 * 1024 * 1024
HARD_MAX_TOTAL_BYTES = 10 * 1024 * 1024 * 1024
HARD_MAX_STUDIES = 2_000_000
HARD_MAX_CANDIDATE_ROWS = 250_000
HARD_MAX_DERIVED_BYTES = 512 * 1024 * 1024
HARD_MAX_MANIFEST_BYTES = 20 * 1024 * 1024
MAX_QUERY_LENGTH = 512
MAX_PAGE_TOKEN_LENGTH = 4096
SYNTHETIC_REFERENCE_BASE = (
    "https://example.invalid/crispr-evidencerank/synthetic-clinicaltrials-gov"
)
STUDY_FIELDS = (
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
STUDY_FIELDS_PARAMETER = ",".join(STUDY_FIELDS)

_RETRIABLE_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_CAPTURED_HEADERS = (
    "content-type",
    "content-length",
    "content-encoding",
    "date",
    "etag",
    "last-modified",
    "retry-after",
)
_FIXED_DERIVED_FILENAMES = (
    "study_inventory.tsv",
    "curation_queue.tsv",
    "data_assets.tsv",
)


class ClinicalTrialsGovIntakeError(RuntimeError):
    """Raised when a source response cannot support an auditable snapshot."""


@dataclass(frozen=True)
class HttpResponse:
    """Exact HTTP response bytes plus the small provenance envelope we retain."""

    body: bytes
    status_code: int
    final_url: str
    headers: Mapping[str, str]
    attempt_count: int = 1


class JsonRequester(Protocol):
    def __call__(
        self,
        url: str,
        *,
        timeout_seconds: float,
        max_response_bytes: int,
        max_attempts: int,
        backoff_seconds: float,
    ) -> HttpResponse: ...


@dataclass(frozen=True)
class _PageCapture:
    page_index: int
    filename: str
    request_url: str
    requested_page_token: str | None
    next_page_token: str | None
    sha256: str
    byte_size: int
    study_ids: tuple[str, ...]
    total_count: int | None
    response_url: str
    response_headers: dict[str, str]
    attempt_count: int
    retrieved_at_utc: str


def _page_manifest_entry(page: _PageCapture) -> dict[str, object]:
    study_ids = list(page.study_ids)
    return {
        "page_index": page.page_index,
        "filename": page.filename,
        "request_url": page.request_url,
        "response_url": page.response_url,
        "requested_page_token": page.requested_page_token,
        "next_page_token": page.next_page_token,
        "sha256": page.sha256,
        "byte_size": page.byte_size,
        "study_count": len(study_ids),
        "study_ids": study_ids,
        "study_ids_sha256": _sha256_bytes(_canonical_json_bytes(study_ids)),
        "total_count": page.total_count,
        "response_headers": page.response_headers,
        "attempt_count": page.attempt_count,
        "retrieved_at_utc": page.retrieved_at_utc,
    }


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
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


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ClinicalTrialsGovIntakeError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ClinicalTrialsGovIntakeError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _load_json_object(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ClinicalTrialsGovIntakeError(f"{label} is not UTF-8 JSON") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except ClinicalTrialsGovIntakeError:
        raise
    except (TypeError, ValueError, RecursionError) as exc:
        raise ClinicalTrialsGovIntakeError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ClinicalTrialsGovIntakeError(f"{label} must be a JSON object")
    return value


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_iso(value: datetime, *, label: str) -> str:
    offset = value.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f"{label} must include the UTC timezone")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _normalize_query_text(value: str, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    if len(normalized) > MAX_QUERY_LENGTH:
        raise ValueError(f"{label} exceeds {MAX_QUERY_LENGTH} characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError(f"{label} contains control characters")
    return normalized


def _validate_limits(
    *,
    page_size: int,
    timeout_seconds: float,
    max_attempts: int,
    backoff_seconds: float,
    max_pages: int,
    max_studies: int,
    max_candidate_rows: int,
    max_page_bytes: int,
    max_total_bytes: int,
    max_derived_bytes: int,
    max_elapsed_seconds: float,
) -> None:
    if isinstance(page_size, bool) or not 1 <= page_size <= MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be in 1..{MAX_PAGE_SIZE}")
    if isinstance(timeout_seconds, bool) or not 0 < timeout_seconds <= 300:
        raise ValueError("timeout_seconds must be in (0, 300]")
    if isinstance(max_attempts, bool) or not 1 <= max_attempts <= 10:
        raise ValueError("max_attempts must be in 1..10")
    if isinstance(backoff_seconds, bool) or not 0 <= backoff_seconds <= 60:
        raise ValueError("backoff_seconds must be in [0, 60]")
    if isinstance(max_pages, bool) or not 1 <= max_pages <= HARD_MAX_PAGES:
        raise ValueError(f"max_pages must be in 1..{HARD_MAX_PAGES}")
    if isinstance(max_studies, bool) or not 0 <= max_studies <= HARD_MAX_STUDIES:
        raise ValueError(f"max_studies must be in 0..{HARD_MAX_STUDIES}")
    if (
        isinstance(max_candidate_rows, bool)
        or not 0 <= max_candidate_rows <= HARD_MAX_CANDIDATE_ROWS
    ):
        raise ValueError(f"max_candidate_rows must be in 0..{HARD_MAX_CANDIDATE_ROWS}")
    if isinstance(max_page_bytes, bool) or not (
        1 <= max_page_bytes <= HARD_MAX_PAGE_BYTES
    ):
        raise ValueError(f"max_page_bytes must be in 1..{HARD_MAX_PAGE_BYTES}")
    if isinstance(max_total_bytes, bool) or not (
        max_page_bytes <= max_total_bytes <= HARD_MAX_TOTAL_BYTES
    ):
        raise ValueError(
            "max_total_bytes must be at least max_page_bytes and no more than "
            f"{HARD_MAX_TOTAL_BYTES}"
        )
    if isinstance(max_derived_bytes, bool) or not (
        1 <= max_derived_bytes <= HARD_MAX_DERIVED_BYTES
    ):
        raise ValueError(f"max_derived_bytes must be in 1..{HARD_MAX_DERIVED_BYTES}")
    if isinstance(max_elapsed_seconds, bool) or not 0 < max_elapsed_seconds <= 86_400:
        raise ValueError("max_elapsed_seconds must be in (0, 86400]")


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            return None
        seconds = (retry_at - datetime.now(UTC)).total_seconds()
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


def _default_json_requester(
    url: str,
    *,
    timeout_seconds: float,
    max_response_bytes: int,
    max_attempts: int,
    backoff_seconds: float,
) -> HttpResponse:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Cache-Control": "no-cache",
            "User-Agent": (
                f"crispr-evidencerank/{__version__} ClinicalTrials.gov-intake"
            ),
        },
        method="GET",
    )

    class _RejectRedirects(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise ClinicalTrialsGovIntakeError(
                "ClinicalTrials.gov redirects are not accepted"
            )

    opener = build_opener(ProxyHandler({}), _RejectRedirects())
    last_error: Exception | None = None
    request_deadline = time.monotonic() + timeout_seconds
    for attempt in range(1, max_attempts + 1):
        remaining_seconds = request_deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise ClinicalTrialsGovIntakeError(
                "ClinicalTrials.gov request exceeded its time budget"
            ) from last_error
        try:
            with opener.open(request, timeout=remaining_seconds) as response:
                status_code = int(getattr(response, "status", response.getcode()))
                headers = {
                    key.lower(): value for key, value in response.headers.items()
                }
                content_length = headers.get("content-length")
                if content_length is not None:
                    try:
                        declared_size = int(content_length)
                    except ValueError as exc:
                        raise ClinicalTrialsGovIntakeError(
                            "response Content-Length is not an integer"
                        ) from exc
                    if declared_size < 0 or declared_size > max_response_bytes:
                        raise ClinicalTrialsGovIntakeError(
                            "response exceeds the configured byte limit"
                        )
                body_parts: list[bytes] = []
                observed_bytes = 0
                read_once = getattr(response, "read1", response.read)
                while True:
                    if time.monotonic() >= request_deadline:
                        raise ClinicalTrialsGovIntakeError(
                            "ClinicalTrials.gov response exceeded its time budget"
                        )
                    chunk = read_once(
                        min(64 * 1024, max_response_bytes + 1 - observed_bytes)
                    )
                    if time.monotonic() >= request_deadline:
                        raise ClinicalTrialsGovIntakeError(
                            "ClinicalTrials.gov response exceeded its time budget"
                        )
                    if not chunk:
                        break
                    body_parts.append(chunk)
                    observed_bytes += len(chunk)
                    if observed_bytes > max_response_bytes:
                        raise ClinicalTrialsGovIntakeError(
                            "response exceeds the configured byte limit"
                        )
                body = b"".join(body_parts)
                if content_length is not None and declared_size != len(body):
                    raise ClinicalTrialsGovIntakeError(
                        "response byte count differs from Content-Length"
                    )
                return HttpResponse(
                    body=body,
                    status_code=status_code,
                    final_url=response.geturl(),
                    headers=headers,
                    attempt_count=attempt,
                )
        except HTTPError as exc:
            last_error = exc
            if exc.code not in _RETRIABLE_HTTP_STATUS or attempt == max_attempts:
                raise ClinicalTrialsGovIntakeError(
                    f"ClinicalTrials.gov returned HTTP {exc.code}"
                ) from exc
            headers = {key.lower(): value for key, value in exc.headers.items()}
            delay = _retry_after_seconds(headers)
        except (TimeoutError, URLError) as exc:
            last_error = exc
            if attempt == max_attempts:
                raise ClinicalTrialsGovIntakeError(
                    "ClinicalTrials.gov request failed after retries"
                ) from exc
            delay = None
        if delay is None:
            delay = min(60.0, backoff_seconds * (2 ** (attempt - 1)))
        remaining_seconds = request_deadline - time.monotonic()
        if delay > remaining_seconds:
            raise ClinicalTrialsGovIntakeError(
                "ClinicalTrials.gov retry delay exceeds the request time budget"
            ) from last_error
        if delay:
            time.sleep(delay)
    raise ClinicalTrialsGovIntakeError(
        "ClinicalTrials.gov request failed"
    ) from last_error


def _captured_headers(headers: Mapping[str, str]) -> dict[str, str]:
    lowered = {str(key).lower(): str(value) for key, value in headers.items()}
    return {key: lowered[key] for key in _CAPTURED_HEADERS if key in lowered}


def _is_json_media_type(value: str) -> bool:
    return value.split(";", 1)[0].strip().lower() == "application/json"


def _validate_response(
    response: HttpResponse,
    *,
    expected_url: str,
    expected_path: str,
    max_response_bytes: int,
) -> None:
    if response.status_code != 200:
        raise ClinicalTrialsGovIntakeError(
            f"ClinicalTrials.gov returned HTTP {response.status_code}"
        )
    if not 1 <= response.attempt_count <= 10:
        raise ClinicalTrialsGovIntakeError("invalid HTTP attempt count")
    if len(response.body) > max_response_bytes:
        raise ClinicalTrialsGovIntakeError("response exceeds the configured byte limit")
    parsed_url = urlsplit(response.final_url)
    if (
        parsed_url.scheme != "https"
        or parsed_url.hostname != "clinicaltrials.gov"
        or parsed_url.port not in {None, 443}
        or parsed_url.path != expected_path
        or parsed_url.fragment
    ):
        raise ClinicalTrialsGovIntakeError(
            "response redirected outside the pinned ClinicalTrials.gov endpoint"
        )
    if response.final_url != expected_url:
        raise ClinicalTrialsGovIntakeError(
            "response URL differs from the exact requested URL"
        )
    content_type = _captured_headers(response.headers).get("content-type", "")
    if not _is_json_media_type(content_type):
        raise ClinicalTrialsGovIntakeError(
            "ClinicalTrials.gov response is not application/json"
        )
    content_encoding = _captured_headers(response.headers).get(
        "content-encoding", "identity"
    )
    if content_encoding.lower() != "identity":
        raise ClinicalTrialsGovIntakeError(
            "ClinicalTrials.gov response Content-Encoding must be identity"
        )
    declared_length = _captured_headers(response.headers).get("content-length")
    if declared_length is not None:
        try:
            declared_size = int(declared_length)
        except ValueError as exc:
            raise ClinicalTrialsGovIntakeError(
                "response Content-Length is not an integer"
            ) from exc
        if declared_size != len(response.body):
            raise ClinicalTrialsGovIntakeError(
                "response byte count differs from Content-Length"
            )


def _request_json(
    requester: JsonRequester,
    url: str,
    *,
    expected_path: str,
    timeout_seconds: float,
    max_response_bytes: int,
    max_attempts: int,
    backoff_seconds: float,
) -> tuple[HttpResponse, dict[str, Any]]:
    response = requester(
        url,
        timeout_seconds=timeout_seconds,
        max_response_bytes=max_response_bytes,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
    )
    if response.attempt_count > max_attempts:
        raise ClinicalTrialsGovIntakeError(
            "HTTP attempt count exceeds the configured retry limit"
        )
    _validate_response(
        response,
        expected_url=url,
        expected_path=expected_path,
        max_response_bytes=max_response_bytes,
    )
    return response, _load_json_object(response.body, label=expected_path)


def _parse_version(value: dict[str, Any]) -> tuple[str, str, date]:
    api_version = value.get("apiVersion")
    data_timestamp = value.get("dataTimestamp")
    if not isinstance(api_version, str) or not api_version.strip():
        raise ClinicalTrialsGovIntakeError("version response lacks apiVersion")
    if not isinstance(data_timestamp, str) or not data_timestamp.strip():
        raise ClinicalTrialsGovIntakeError("version response lacks dataTimestamp")
    if any(ord(character) < 32 or ord(character) == 127 for character in api_version):
        raise ClinicalTrialsGovIntakeError("apiVersion contains control characters")
    try:
        if "T" not in data_timestamp:
            raise ValueError("timestamp lacks time component")
        parsed_timestamp = datetime.fromisoformat(data_timestamp.replace("Z", "+00:00"))
        timestamp_date = parsed_timestamp.date()
    except ValueError as exc:
        raise ClinicalTrialsGovIntakeError(
            "version dataTimestamp is not ISO 8601"
        ) from exc
    timestamp_offset = parsed_timestamp.utcoffset()
    if timestamp_offset is not None and timestamp_offset.total_seconds() != 0:
        raise ClinicalTrialsGovIntakeError(
            "version dataTimestamp must use UTC when an offset is present"
        )
    return api_version, data_timestamp, timestamp_date


def _require_explicit_synthetic_version(value: Mapping[str, Any]) -> None:
    api_version = value.get("apiVersion")
    if value.get("syntheticFixture") is not True or not (
        isinstance(api_version, str) and "synthetic" in api_version.lower()
    ):
        raise ClinicalTrialsGovIntakeError(
            "synthetic fixture version response lacks its explicit marker"
        )


def _require_explicit_synthetic_page(value: Mapping[str, Any]) -> None:
    if value.get("syntheticFixture") is not True:
        raise ClinicalTrialsGovIntakeError(
            "synthetic fixture studies page lacks its explicit marker"
        )
    studies = value.get("studies")
    if not isinstance(studies, list):
        raise ClinicalTrialsGovIntakeError("synthetic fixture page lacks studies")
    for study_index, study in enumerate(studies):
        study_object = _required_mapping(
            study,
            label=f"synthetic study {study_index}",
        )
        protocol = _required_mapping(
            study_object.get("protocolSection"),
            label=f"synthetic study {study_index}.protocolSection",
        )
        identification = _required_mapping(
            protocol.get("identificationModule"),
            label=f"synthetic study {study_index}.identificationModule",
        )
        nct_id = identification.get("nctId")
        titles = (
            identification.get("briefTitle"),
            identification.get("officialTitle"),
        )
        if not (
            isinstance(nct_id, str)
            and nct_id.startswith("NCT9")
            and len(nct_id) == 11
            and nct_id[3:].isdigit()
            and any(
                isinstance(title, str) and "synthetic" in title.lower()
                for title in titles
            )
        ):
            raise ClinicalTrialsGovIntakeError(
                "synthetic fixture study lacks a reserved ID and synthetic title"
            )


def _studies_url(
    *,
    condition_query: str,
    intervention_query: str,
    page_size: int,
    page_token: str | None,
) -> str:
    parameters = [
        ("format", "json"),
        ("markupFormat", "markdown"),
        ("fields", STUDY_FIELDS_PARAMETER),
        ("query.cond", condition_query),
        ("query.intr", intervention_query),
        ("pageSize", str(page_size)),
        ("countTotal", "true"),
    ]
    if page_token is not None:
        parameters.append(("pageToken", page_token))
    return f"{STUDIES_ENDPOINT}?{urlencode(parameters)}"


def _required_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ClinicalTrialsGovIntakeError(f"{label} must be a JSON object")
    return value


def _optional_mapping(
    parent: Mapping[str, Any], key: str, *, label: str
) -> dict[str, Any]:
    value = parent.get(key)
    if value is None:
        return {}
    return _required_mapping(value, label=label)


def _source_string(value: object, *, label: str, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ClinicalTrialsGovIntakeError(f"{label} must be a non-empty string")
    if "\x00" in value:
        raise ClinicalTrialsGovIntakeError(f"{label} contains a NUL character")
    return value.strip()


def _source_string_list(
    value: object,
    *,
    label: str,
    missing_is_empty: bool = True,
) -> list[str]:
    if value is None and missing_is_empty:
        return []
    if not isinstance(value, list):
        raise ClinicalTrialsGovIntakeError(f"{label} must be a JSON list")
    return [
        _source_string(item, label=f"{label}[{index}]", required=True) or ""
        for index, item in enumerate(value)
    ]


def _source_date(value: object, *, label: str) -> str | None:
    text = _source_string(value, label=label)
    if text is None:
        return None
    parts = text.split("-")
    if len(parts) not in {1, 2, 3} or any(not part.isdigit() for part in parts):
        raise ClinicalTrialsGovIntakeError(f"{label} must be an ISO partial date")
    if len(parts[0]) != 4:
        raise ClinicalTrialsGovIntakeError(f"{label} must start with a four-digit year")
    year = int(parts[0])
    if not 1 <= year <= 9999:
        raise ClinicalTrialsGovIntakeError(f"{label} year is out of range")
    if len(parts) >= 2:
        if len(parts[1]) != 2 or not 1 <= int(parts[1]) <= 12:
            raise ClinicalTrialsGovIntakeError(f"{label} month is out of range")
    if len(parts) == 3:
        if len(parts[2]) != 2:
            raise ClinicalTrialsGovIntakeError(f"{label} day must use two digits")
        try:
            date(year, int(parts[1]), int(parts[2]))
        except ValueError as exc:
            raise ClinicalTrialsGovIntakeError(f"{label} day is out of range") from exc
    return text


def _canonical_json_text(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _spreadsheet_safe_text(value: str | None) -> tuple[str | None, bool]:
    if value is not None and value.startswith(("=", "+", "-", "@")):
        return f"'{value}", True
    return value, False


def _extract_study(
    study: object,
    *,
    page_index: int,
    study_index: int,
    page_sha256: str,
    retrieved_at_utc: str,
    condition_query: str,
    intervention_query: str,
    max_candidate_rows: int,
    max_derived_work_bytes: int,
) -> tuple[str, dict[str, object], list[dict[str, object]], int]:
    study_object = _required_mapping(
        study,
        label=f"page {page_index} study {study_index}",
    )
    protocol = _required_mapping(
        study_object.get("protocolSection"),
        label=f"page {page_index} study {study_index}.protocolSection",
    )
    identification = _required_mapping(
        protocol.get("identificationModule"),
        label=f"page {page_index} study {study_index}.identificationModule",
    )
    nct_id = _source_string(
        identification.get("nctId"),
        label=f"page {page_index} study {study_index}.nctId",
        required=True,
    )
    assert nct_id is not None
    if len(nct_id) != 11 or not nct_id.startswith("NCT") or not nct_id[3:].isdigit():
        raise ClinicalTrialsGovIntakeError(f"invalid ClinicalTrials.gov ID: {nct_id}")

    status_module = _optional_mapping(
        protocol,
        "statusModule",
        label=f"{nct_id}.statusModule",
    )
    design_module = _optional_mapping(
        protocol,
        "designModule",
        label=f"{nct_id}.designModule",
    )
    conditions_module = _optional_mapping(
        protocol,
        "conditionsModule",
        label=f"{nct_id}.conditionsModule",
    )
    description_module = _optional_mapping(
        protocol,
        "descriptionModule",
        label=f"{nct_id}.descriptionModule",
    )
    eligibility_module = _optional_mapping(
        protocol,
        "eligibilityModule",
        label=f"{nct_id}.eligibilityModule",
    )
    arms_module = _optional_mapping(
        protocol,
        "armsInterventionsModule",
        label=f"{nct_id}.armsInterventionsModule",
    )
    derived_section = _optional_mapping(
        study_object,
        "derivedSection",
        label=f"{nct_id}.derivedSection",
    )
    misc_module = _optional_mapping(
        derived_section,
        "miscInfoModule",
        label=f"{nct_id}.miscInfoModule",
    )

    phases = _source_string_list(
        design_module.get("phases"),
        label=f"{nct_id}.phases",
    )
    conditions = _source_string_list(
        conditions_module.get("conditions"),
        label=f"{nct_id}.conditions",
    )
    keywords = _source_string_list(
        conditions_module.get("keywords"),
        label=f"{nct_id}.keywords",
    )
    design_info = _optional_mapping(
        design_module,
        "designInfo",
        label=f"{nct_id}.designInfo",
    )
    enrollment_info = _optional_mapping(
        design_module,
        "enrollmentInfo",
        label=f"{nct_id}.enrollmentInfo",
    )
    enrollment_count = enrollment_info.get("count")
    if enrollment_count is not None and (
        isinstance(enrollment_count, bool) or not isinstance(enrollment_count, int)
    ):
        raise ClinicalTrialsGovIntakeError(
            f"{nct_id}.enrollmentInfo.count must be an integer"
        )
    design_summary = {
        "allocation": _source_string(
            design_info.get("allocation"),
            label=f"{nct_id}.designInfo.allocation",
        ),
        "interventionModel": _source_string(
            design_info.get("interventionModel"),
            label=f"{nct_id}.designInfo.interventionModel",
        ),
        "primaryPurpose": _source_string(
            design_info.get("primaryPurpose"),
            label=f"{nct_id}.designInfo.primaryPurpose",
        ),
        "enrollmentCount": enrollment_count,
        "enrollmentType": _source_string(
            enrollment_info.get("type"),
            label=f"{nct_id}.enrollmentInfo.type",
        ),
    }
    interventions_value = arms_module.get("interventions")
    if interventions_value is None:
        interventions_value = []
    if not isinstance(interventions_value, list):
        raise ClinicalTrialsGovIntakeError(f"{nct_id}.interventions must be a list")
    interventions: list[dict[str, object]] = []
    for intervention_index, intervention_value in enumerate(interventions_value):
        intervention = _required_mapping(
            intervention_value,
            label=f"{nct_id}.interventions[{intervention_index}]",
        )
        name = _source_string(
            intervention.get("name"),
            label=f"{nct_id}.interventions[{intervention_index}].name",
            required=True,
        )
        intervention_type = _source_string(
            intervention.get("type"),
            label=f"{nct_id}.interventions[{intervention_index}].type",
        )
        arm_labels = _source_string_list(
            intervention.get("armGroupLabels"),
            label=f"{nct_id}.interventions[{intervention_index}].armGroupLabels",
        )
        other_names = _source_string_list(
            intervention.get("otherNames"),
            label=f"{nct_id}.interventions[{intervention_index}].otherNames",
        )
        interventions.append(
            {
                "name": name,
                "type": intervention_type,
                "description": _source_string(
                    intervention.get("description"),
                    label=(f"{nct_id}.interventions[{intervention_index}].description"),
                ),
                "armGroupLabels": arm_labels,
                "otherNames": other_names,
            }
        )

    arm_groups_value = arms_module.get("armGroups")
    if arm_groups_value is None:
        arm_groups_value = []
    if not isinstance(arm_groups_value, list):
        raise ClinicalTrialsGovIntakeError(f"{nct_id}.armGroups must be a list")
    arm_groups: list[dict[str, object]] = []
    arm_groups_by_label: dict[str, tuple[int, dict[str, object]]] = {}
    for arm_index, arm_value in enumerate(arm_groups_value):
        arm = _required_mapping(
            arm_value,
            label=f"{nct_id}.armGroups[{arm_index}]",
        )
        arm_label = _source_string(
            arm.get("label"),
            label=f"{nct_id}.armGroups[{arm_index}].label",
            required=True,
        )
        assert arm_label is not None
        if arm_label in arm_groups_by_label:
            raise ClinicalTrialsGovIntakeError(
                f"{nct_id} contains duplicate arm-group label: {arm_label}"
            )
        normalized_arm = {
            "label": arm_label,
            "type": _source_string(
                arm.get("type"),
                label=f"{nct_id}.armGroups[{arm_index}].type",
            ),
            "description": _source_string(
                arm.get("description"),
                label=f"{nct_id}.armGroups[{arm_index}].description",
            ),
            "interventionNames": _source_string_list(
                arm.get("interventionNames"),
                label=f"{nct_id}.armGroups[{arm_index}].interventionNames",
            ),
        }
        arm_groups.append(normalized_arm)
        arm_groups_by_label[arm_label] = (arm_index, normalized_arm)

    last_update_struct = _optional_mapping(
        status_module,
        "lastUpdatePostDateStruct",
        label=f"{nct_id}.lastUpdatePostDateStruct",
    )
    first_post_struct = _optional_mapping(
        status_module,
        "studyFirstPostDateStruct",
        label=f"{nct_id}.studyFirstPostDateStruct",
    )
    results_first_post_struct = _optional_mapping(
        status_module,
        "resultsFirstPostDateStruct",
        label=f"{nct_id}.resultsFirstPostDateStruct",
    )
    last_update_date = _source_date(
        last_update_struct.get("date"),
        label=f"{nct_id}.lastUpdatePostDateStruct.date",
    )
    first_post_date = _source_date(
        first_post_struct.get("date"),
        label=f"{nct_id}.studyFirstPostDateStruct.date",
    )
    results_first_post_date = _source_date(
        results_first_post_struct.get("date"),
        label=f"{nct_id}.resultsFirstPostDateStruct.date",
    )
    version_holder = _source_string(
        misc_module.get("versionHolder"),
        label=f"{nct_id}.versionHolder",
    )
    if version_holder is not None:
        _source_date(version_holder, label=f"{nct_id}.versionHolder")

    has_results = study_object.get("hasResults")
    if has_results is not None and not isinstance(has_results, bool):
        raise ClinicalTrialsGovIntakeError(f"{nct_id}.hasResults must be boolean")

    page_asset_placeholder = f"__PAGE_ASSET_{page_index:06d}__"
    source_study_type, escaped_study_type = _spreadsheet_safe_text(
        _source_string(
            design_module.get("studyType"),
            label=f"{nct_id}.studyType",
        )
    )
    source_overall_status, escaped_overall_status = _spreadsheet_safe_text(
        _source_string(
            status_module.get("overallStatus"),
            label=f"{nct_id}.overallStatus",
        )
    )
    brief_title = _source_string(
        identification.get("briefTitle"),
        label=f"{nct_id}.briefTitle",
    )
    official_title = _source_string(
        identification.get("officialTitle"),
        label=f"{nct_id}.officialTitle",
    )
    safe_brief_title, escaped_brief_title = _spreadsheet_safe_text(brief_title)
    safe_official_title, escaped_official_title = _spreadsheet_safe_text(official_title)
    population_scope_values: dict[str, object] = {}
    for path_suffix, source_value in (
        ("identificationModule.briefTitle", brief_title),
        ("identificationModule.officialTitle", official_title),
        (
            "descriptionModule.briefSummary",
            _source_string(
                description_module.get("briefSummary"),
                label=f"{nct_id}.briefSummary",
            ),
        ),
        (
            "descriptionModule.detailedDescription",
            _source_string(
                description_module.get("detailedDescription"),
                label=f"{nct_id}.detailedDescription",
            ),
        ),
        (
            "eligibilityModule.eligibilityCriteria",
            _source_string(
                eligibility_module.get("eligibilityCriteria"),
                label=f"{nct_id}.eligibilityCriteria",
            ),
        ),
        (
            "eligibilityModule.sex",
            _source_string(
                eligibility_module.get("sex"),
                label=f"{nct_id}.sex",
            ),
        ),
        (
            "eligibilityModule.minimumAge",
            _source_string(
                eligibility_module.get("minimumAge"),
                label=f"{nct_id}.minimumAge",
            ),
        ),
        (
            "eligibilityModule.maximumAge",
            _source_string(
                eligibility_module.get("maximumAge"),
                label=f"{nct_id}.maximumAge",
            ),
        ),
    ):
        if source_value is not None:
            population_scope_values[
                f"studies[{study_index}].protocolSection.{path_suffix}"
            ] = source_value
    healthy_volunteers = eligibility_module.get("healthyVolunteers")
    if healthy_volunteers is not None:
        if not isinstance(healthy_volunteers, bool):
            raise ClinicalTrialsGovIntakeError(
                f"{nct_id}.healthyVolunteers must be boolean"
            )
        population_scope_values[
            f"studies[{study_index}].protocolSection."
            "eligibilityModule.healthyVolunteers"
        ] = healthy_volunteers
    for condition_index, condition_text in enumerate(conditions):
        population_scope_values[
            f"studies[{study_index}].protocolSection.conditionsModule."
            f"conditions[{condition_index}]"
        ] = condition_text
    for keyword_index, keyword in enumerate(keywords):
        population_scope_values[
            f"studies[{study_index}].protocolSection.conditionsModule."
            f"keywords[{keyword_index}]"
        ] = keyword
    for design_key, design_value in design_summary.items():
        if design_value is not None:
            if design_key.startswith("enrollment"):
                locator = (
                    f"studies[{study_index}].protocolSection.designModule."
                    f"enrollmentInfo.{design_key.removeprefix('enrollment').lower()}"
                )
            else:
                locator = (
                    f"studies[{study_index}].protocolSection.designModule."
                    f"designInfo.{design_key}"
                )
            population_scope_values[locator] = design_value
    why_stopped = _source_string(
        status_module.get("whyStopped"),
        label=f"{nct_id}.whyStopped",
    )
    if why_stopped is not None:
        population_scope_values[
            f"studies[{study_index}].protocolSection.statusModule.whyStopped"
        ] = why_stopped
    for intervention_index, intervention in enumerate(interventions):
        if intervention["description"] is not None:
            population_scope_values[
                f"studies[{study_index}].protocolSection.armsInterventionsModule."
                f"interventions[{intervention_index}].description"
            ] = intervention["description"]
    for arm_index, arm in enumerate(arm_groups):
        if arm["description"] is not None:
            population_scope_values[
                f"studies[{study_index}].protocolSection.armsInterventionsModule."
                f"armGroups[{arm_index}].description"
            ] = arm["description"]
    population_scope_locators = list(population_scope_values)
    population_scope_sha256 = _sha256_bytes(
        _canonical_json_bytes(population_scope_values)
    )
    inventory = {
        "snapshot_id": "__SNAPSHOT_ID__",
        "source_study_id": nct_id,
        "source_family_id": f"ctgov:{nct_id}",
        "source_asset_id": page_asset_placeholder,
        "source_asset_sha256": page_sha256,
        "source_page_index": page_index,
        "source_study_index": study_index,
        "source_version_holder": version_holder,
        "record_last_update_date": last_update_date,
        "study_first_post_date": first_post_date,
        "results_first_post_date": results_first_post_date,
        "source_study_type": source_study_type,
        "source_overall_status": source_overall_status,
        "source_brief_title": safe_brief_title,
        "source_official_title": safe_official_title,
        "source_phases_json": _canonical_json_text(phases),
        "source_conditions_json": _canonical_json_text(conditions),
        "source_keywords_json": _canonical_json_text(keywords),
        "source_design_json": _canonical_json_text(design_summary),
        "source_interventions_json": _canonical_json_text(interventions),
        "source_arm_groups_json": _canonical_json_text(arm_groups),
        "population_scope_locator_candidates_json": _canonical_json_text(
            population_scope_locators
        ),
        "population_scope_text_sha256": population_scope_sha256,
        "source_has_results": has_results,
        "source_url": f"{API_ORIGIN}/study/{nct_id}",
        "retrieved_at_utc": retrieved_at_utc,
        "tsv_formula_escape_applied": (
            escaped_study_type
            or escaped_overall_status
            or escaped_brief_title
            or escaped_official_title
        ),
        "normalization_status": "raw_registry_inventory_only",
        "used_for_label": False,
    }
    derived_work_bytes = len(_canonical_json_bytes(inventory))
    if derived_work_bytes > max_derived_work_bytes:
        raise ClinicalTrialsGovIntakeError(
            "derived row materialization exceeds the configured byte limit"
        )

    candidate_rows: list[dict[str, object]] = []
    intervention_items: list[tuple[int | None, dict[str, object] | None]] = (
        list(enumerate(interventions)) if interventions else [(None, None)]
    )
    condition_items: list[tuple[int | None, str | None]] = (
        list(enumerate(conditions)) if conditions else [(None, None)]
    )
    candidate_count = len(intervention_items) * len(condition_items)
    if candidate_count > max_candidate_rows:
        raise ClinicalTrialsGovIntakeError(
            "intervention/condition candidate product exceeds the configured limit"
        )
    for intervention_index, intervention in intervention_items:
        for condition_index, condition_text in condition_items:
            if intervention is not None and condition_text is not None:
                normalization_status = "requires_curator_review"
                exclusion_reason = "unreviewed_study_level_co_mention"
            elif intervention is not None:
                normalization_status = "missing_source_condition"
                exclusion_reason = normalization_status
            elif condition_text is not None:
                normalization_status = "missing_source_intervention"
                exclusion_reason = normalization_status
            else:
                normalization_status = "missing_source_intervention_and_condition"
                exclusion_reason = normalization_status
            intervention_token = (
                str(intervention_index) if intervention_index is not None else "missing"
            )
            condition_token = (
                str(condition_index) if condition_index is not None else "missing"
            )
            treatment_text = intervention["name"] if intervention is not None else None
            intervention_type = (
                intervention["type"] if intervention is not None else None
            )
            safe_treatment_text, escaped_treatment = _spreadsheet_safe_text(
                treatment_text if isinstance(treatment_text, str) else None
            )
            safe_intervention_type, escaped_intervention_type = _spreadsheet_safe_text(
                intervention_type if isinstance(intervention_type, str) else None
            )
            safe_condition_text, escaped_condition = _spreadsheet_safe_text(
                condition_text
            )
            safe_intervention_query, escaped_intervention_query = (
                _spreadsheet_safe_text(intervention_query)
            )
            safe_condition_query, escaped_condition_query = _spreadsheet_safe_text(
                condition_query
            )
            linked_arms = []
            linked_arm_locators = []
            unmatched_arm_labels = []
            if intervention is not None:
                for arm_label in intervention["armGroupLabels"]:
                    linked_arm_entry = arm_groups_by_label.get(str(arm_label))
                    if linked_arm_entry is None:
                        unmatched_arm_labels.append(arm_label)
                    else:
                        linked_arm_index, linked_arm = linked_arm_entry
                        linked_arms.append(linked_arm)
                        linked_arm_locators.append(
                            f"studies[{study_index}].protocolSection."
                            "armsInterventionsModule."
                            f"armGroups[{linked_arm_index}]"
                        )
            candidate_row = {
                "candidate_id": (
                    f"__SNAPSHOT_ID__:ctgov:{nct_id}:"
                    f"intervention:{intervention_token}:"
                    f"condition:{condition_token}"
                ),
                "snapshot_id": "__SNAPSHOT_ID__",
                "source_study_id": nct_id,
                "source_family_id": f"ctgov:{nct_id}",
                "source_asset_id": page_asset_placeholder,
                "source_asset_sha256": page_sha256,
                "source_page_index": page_index,
                "source_study_index": study_index,
                "source_intervention_index": intervention_index,
                "source_condition_index": condition_index,
                "source_treatment_locator": (
                    f"studies[{study_index}].protocolSection."
                    "armsInterventionsModule."
                    f"interventions[{intervention_index}]"
                    if intervention_index is not None
                    else None
                ),
                "source_condition_locator": (
                    f"studies[{study_index}].protocolSection."
                    "conditionsModule."
                    f"conditions[{condition_index}]"
                    if condition_index is not None
                    else None
                ),
                "source_treatment_text": safe_treatment_text,
                "source_intervention_type": safe_intervention_type,
                "source_arm_group_labels_json": _canonical_json_text(
                    intervention["armGroupLabels"] if intervention is not None else []
                ),
                "source_other_names_json": _canonical_json_text(
                    intervention["otherNames"] if intervention is not None else []
                ),
                "source_linked_arm_groups_json": _canonical_json_text(linked_arms),
                "source_unmatched_arm_group_labels_json": _canonical_json_text(
                    unmatched_arm_labels
                ),
                "source_linked_arm_locators_json": _canonical_json_text(
                    linked_arm_locators
                ),
                "population_scope_locator_candidates_json": (
                    _canonical_json_text(population_scope_locators)
                ),
                "population_scope_text_sha256": population_scope_sha256,
                "source_condition_text": safe_condition_text,
                "query_intervention_text": safe_intervention_query,
                "query_condition_text": safe_condition_query,
                "treatment_concept_id": None,
                "treatment_mapping_relation": "unknown",
                "treatment_mapping_review_status": "not_performed",
                "cancer_concept_id": None,
                "cancer_mapping_relation": "unknown",
                "cancer_mapping_review_status": "not_performed",
                "intervention_role": "unknown",
                "regimen_context": "unknown",
                "population_scope_review_status": "not_performed",
                "treatment_cancer_linkage_status": "not_performed",
                "treatment_cancer_linkage_locator": None,
                "co_mention_only": True,
                "blocker_codes_json": _canonical_json_text(
                    [
                        "retrieval_hit_only",
                        "treatment_mapping_not_performed",
                        "cancer_mapping_not_performed",
                        "treatment_cancer_linkage_not_performed",
                        "intervention_role_not_performed",
                        "regimen_context_not_performed",
                        "population_scope_not_performed",
                        *(
                            ["source_intervention_missing"]
                            if intervention is None
                            else []
                        ),
                        *(
                            ["source_condition_missing"]
                            if condition_text is None
                            else []
                        ),
                    ]
                ),
                "normalization_status": normalization_status,
                "exclusion_reason": exclusion_reason,
                "tsv_formula_escape_applied": any(
                    (
                        escaped_treatment,
                        escaped_intervention_type,
                        escaped_condition,
                        escaped_intervention_query,
                        escaped_condition_query,
                    )
                ),
                "eligible_for_clinical_context": False,
                "used_for_label": False,
            }
            candidate_work_bytes = len(_canonical_json_bytes(candidate_row))
            if derived_work_bytes + candidate_work_bytes > max_derived_work_bytes:
                raise ClinicalTrialsGovIntakeError(
                    "derived row materialization exceeds the configured byte limit"
                )
            derived_work_bytes += candidate_work_bytes
            candidate_rows.append(candidate_row)
    return nct_id, inventory, candidate_rows, derived_work_bytes


def _parse_studies_page(
    value: dict[str, Any],
    *,
    page_index: int,
    page_sha256: str,
    retrieved_at_utc: str,
    condition_query: str,
    intervention_query: str,
    max_candidate_rows: int,
    max_derived_work_bytes: int,
) -> tuple[
    list[str],
    list[dict[str, object]],
    list[dict[str, object]],
    str | None,
    int | None,
    int,
]:
    studies = value.get("studies")
    if not isinstance(studies, list):
        raise ClinicalTrialsGovIntakeError("studies response lacks a studies list")
    total_count = value.get("totalCount")
    if page_index == 1:
        if isinstance(total_count, bool) or not isinstance(total_count, int):
            raise ClinicalTrialsGovIntakeError(
                "first countTotal=true response lacks an integer totalCount"
            )
    elif total_count is not None and (
        isinstance(total_count, bool) or not isinstance(total_count, int)
    ):
        raise ClinicalTrialsGovIntakeError(
            "later-page totalCount must be an integer when present"
        )
    if total_count is not None and total_count < 0:
        raise ClinicalTrialsGovIntakeError("totalCount cannot be negative")
    next_page_token = value.get("nextPageToken")
    if next_page_token is not None:
        if (
            not isinstance(next_page_token, str)
            or not next_page_token
            or len(next_page_token) > MAX_PAGE_TOKEN_LENGTH
        ):
            raise ClinicalTrialsGovIntakeError(
                "nextPageToken must be a non-empty string when present"
            )
        if any(
            ord(character) < 32 or ord(character) == 127
            for character in next_page_token
        ):
            raise ClinicalTrialsGovIntakeError(
                "nextPageToken contains control characters"
            )

    study_ids: list[str] = []
    inventory_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    derived_work_bytes = 0
    for study_index, study in enumerate(studies):
        nct_id, inventory, candidates, study_work_bytes = _extract_study(
            study,
            page_index=page_index,
            study_index=study_index,
            page_sha256=page_sha256,
            retrieved_at_utc=retrieved_at_utc,
            condition_query=condition_query,
            intervention_query=intervention_query,
            max_candidate_rows=max_candidate_rows - len(candidate_rows),
            max_derived_work_bytes=(max_derived_work_bytes - derived_work_bytes),
        )
        study_ids.append(nct_id)
        inventory_rows.append(inventory)
        candidate_rows.extend(candidates)
        derived_work_bytes += study_work_bytes
    if len(study_ids) != len(set(study_ids)):
        raise ClinicalTrialsGovIntakeError(
            f"page {page_index} contains duplicate NCT identifiers"
        )
    return (
        study_ids,
        inventory_rows,
        candidate_rows,
        next_page_token,
        total_count,
        derived_work_bytes,
    )


def _frame_tsv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(str(column) for column in frame.columns)
    for row in frame.itertuples(index=False, name=None):
        serialized: list[str] = []
        for value in row:
            if bool(pd.isna(value)):
                serialized.append("")
            elif isinstance(value, bool):
                serialized.append("True" if value else "False")
            elif isinstance(value, float) and value.is_integer():
                serialized.append(str(int(value)))
            else:
                serialized.append(str(value))
        writer.writerow(serialized)
    return buffer.getvalue().encode("utf-8")


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.write_bytes(_frame_tsv_bytes(frame))


def _bind_snapshot_rows(
    inventory_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    *,
    snapshot_id: str,
    page_asset_ids: Mapping[int, str],
    synthetic_fixture: bool,
) -> None:
    """Bind raw-derived rows to one immutable snapshot and page asset roster."""

    for row in inventory_rows:
        page_index = int(row["source_page_index"])
        try:
            source_asset_id = page_asset_ids[page_index]
        except KeyError as exc:
            raise ClinicalTrialsGovIntakeError(
                f"inventory row references unknown page {page_index}"
            ) from exc
        row["snapshot_id"] = snapshot_id
        row["source_asset_id"] = source_asset_id
        if synthetic_fixture:
            nct_id = str(row["source_study_id"])
            row["source_family_id"] = (
                f"crispr-evidencerank:synthetic-ctgov:{snapshot_id}:{nct_id}"
            )
            row["source_url"] = (
                f"{SYNTHETIC_REFERENCE_BASE}/{snapshot_id}/studies/{nct_id}"
            )
    for row in candidate_rows:
        page_index = int(row["source_page_index"])
        try:
            source_asset_id = page_asset_ids[page_index]
        except KeyError as exc:
            raise ClinicalTrialsGovIntakeError(
                f"curation row references unknown page {page_index}"
            ) from exc
        candidate_id = str(row["candidate_id"])
        if not candidate_id.startswith("__SNAPSHOT_ID__:"):
            raise ClinicalTrialsGovIntakeError(
                "internal candidate identifier lacks its snapshot placeholder"
            )
        bound_candidate_id = candidate_id.replace("__SNAPSHOT_ID__", snapshot_id, 1)
        if synthetic_fixture:
            bound_candidate_id = bound_candidate_id.replace(
                f"{snapshot_id}:ctgov:",
                f"{snapshot_id}:synthetic-ctgov:",
                1,
            )
        row["candidate_id"] = bound_candidate_id
        row["snapshot_id"] = snapshot_id
        row["source_asset_id"] = source_asset_id
        if synthetic_fixture:
            nct_id = str(row["source_study_id"])
            row["source_family_id"] = (
                f"crispr-evidencerank:synthetic-ctgov:{snapshot_id}:{nct_id}"
            )


def _validated_frame(
    rows: list[dict[str, object]],
    model: type[ClinicalTrialsGovStudyInventoryRecord]
    | type[ClinicalTrialsGovCurationCandidateRecord]
    | type[DataAssetRecord],
    *,
    label: str,
) -> pd.DataFrame:
    columns = list(model.model_fields)
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows, columns=columns)
    valid, errors = validate_records(frame, model)
    if not errors.empty:
        details = "; ".join(
            f"row {row.row_number}: {row.error}"
            for row in errors.itertuples(index=False)
        )
        raise ClinicalTrialsGovIntakeError(f"{label} validation failed: {details}")
    return valid.loc[:, columns]


def _asset_row(
    *,
    asset_id: str,
    source_version: str,
    asset_role: str,
    accession: str,
    source_url: str,
    available_date: date,
    retrieved_at_utc: str,
    sha256: str,
    byte_size: int,
    raw_data_family_id: str,
    notes: str,
    code_commit: str | None,
    synthetic_fixture: bool = False,
    transport_mode: str = "live_https",
) -> dict[str, object]:
    if synthetic_fixture:
        source_name = "ClinicalTrials.gov synthetic fixture"
        license_spdx = "Apache-2.0"
        license_terms_url = "https://www.apache.org/licenses/LICENSE-2.0"
        rights_holder = "CRISPR EvidenceRank project; wholly synthetic test content"
        redistribution_raw = True
        redistribution_derived = True
        download_method = "injected deterministic synthetic transport; no network"
        checksum_provenance = (
            "project_computed_from_exact_injected_synthetic_response_bytes"
        )
        curator_status = "project_generated_synthetic_fixture"
    elif transport_mode == "injected":
        source_name = "ClinicalTrials.gov API-shaped injected transport"
        license_spdx = None
        license_terms_url = TERMS_URL
        rights_holder = "Not attested; response supplied by an injected transport"
        redistribution_raw = False
        redistribution_derived = False
        download_method = "injected response transport; no live HTTPS attestation"
        checksum_provenance = "project_computed_from_exact_injected_response_bytes"
        curator_status = "machine_supplied_unreviewed"
    else:
        source_name = "ClinicalTrials.gov"
        license_spdx = None
        license_terms_url = TERMS_URL
        rights_holder = (
            "National Library of Medicine; study content supplied by registry "
            "sponsors or investigators"
        )
        redistribution_raw = False
        redistribution_derived = False
        download_method = "ClinicalTrials.gov Data API v2 HTTPS GET"
        checksum_provenance = "project_computed_from_exact_http_response_bytes"
        curator_status = "machine_retrieved_unreviewed"
    return {
        "asset_id": asset_id,
        "source_name": source_name,
        "source_version": source_version,
        "asset_role": asset_role,
        "accession": accession,
        "source_url": source_url,
        "available_date": available_date.isoformat(),
        "retrieved_date": retrieved_at_utc[:10],
        "retrieved_at_utc": retrieved_at_utc,
        "sha256": sha256,
        "byte_size": byte_size,
        "checksum_provenance": checksum_provenance,
        "license_spdx": license_spdx,
        "license_terms_url": license_terms_url,
        "rights_holder": rights_holder,
        "redistribution_raw": redistribution_raw,
        "redistribution_derived": redistribution_derived,
        "study_id": None,
        "screen_id": None,
        "source_family_id": None,
        "raw_data_family_id": raw_data_family_id,
        "download_method": download_method,
        "transformation_entrypoint": "fetch-clinicaltrials-gov",
        "code_commit": code_commit,
        "curator_status": curator_status,
        "notes": notes,
    }


def _asset_source_url(
    live_url: str,
    *,
    synthetic_fixture: bool,
    snapshot_id: str,
    asset_suffix: str,
) -> str:
    if not synthetic_fixture:
        return live_url
    return f"{SYNTHETIC_REFERENCE_BASE}/{snapshot_id}/raw-assets/{asset_suffix}"


def _asset_note(
    text: str,
    *,
    synthetic_fixture: bool,
    transport_mode: str,
) -> str:
    if synthetic_fixture:
        text = text.replace(
            " Redistribution is not asserted by this project.", ""
        ).replace(" Redistribution is not asserted.", "")
        return (
            "SYNTHETIC FIXTURE: project-generated content; no real registry study "
            f"or person is represented. {text}"
        )
    if transport_mode == "injected":
        return (
            "INJECTED TRANSPORT: response origin and retrieval time are not attested "
            f"as live ClinicalTrials.gov HTTPS. {text}"
        )
    return text


def _output_entry(path: Path, *, role: str, filename: str) -> dict[str, object]:
    return {
        "role": role,
        "filename": filename,
        "sha256": _file_sha256(path),
        "byte_size": path.stat().st_size,
    }


def _validate_manifest_document(value: dict[str, Any]) -> None:
    try:
        ClinicalTrialsGovSnapshotManifest.model_validate_json(
            _canonical_json_bytes(value),
            strict=True,
        )
    except Exception as exc:
        raise ClinicalTrialsGovIntakeError(
            f"snapshot manifest contract validation failed: {exc}"
        ) from exc


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ClinicalTrialsGovIntakeError(
                f"snapshot staging tree cannot contain a symlink: {path.name}"
            )
        mode = path.stat(follow_symlinks=False).st_mode
        if stat.S_ISREG(mode):
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ClinicalTrialsGovIntakeError(
                        f"snapshot entry is not a regular file: {path.name}"
                    )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        elif stat.S_ISDIR(mode):
            directories.append(path)
        else:
            raise ClinicalTrialsGovIntakeError(
                f"snapshot staging tree contains a special file: {path.name}"
            )
    for directory in sorted(
        directories, key=lambda item: len(item.parts), reverse=True
    ):
        _fsync_directory(directory)


def _acquire_publication_lock(output_parent: Path) -> int:
    """Advisory-lock the output parent without creating an unlinkable lock path."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(output_parent, flags)
    except OSError as exc:
        raise ClinicalTrialsGovIntakeError(
            f"cannot open snapshot output parent for locking: {output_parent}"
        ) from exc
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise FileExistsError(
            "ClinicalTrials.gov snapshot publication is already locked in "
            f"{output_parent}"
        ) from exc
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ClinicalTrialsGovIntakeError(
                "snapshot publication lock target is not a directory"
            )
    except Exception:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        raise
    return descriptor


def _release_publication_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _publish_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing a raced destination."""

    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise ClinicalTrialsGovIntakeError(
            "atomic no-replace directory publication is unavailable"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            f"ClinicalTrials.gov snapshot output exists: {destination}"
        )
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        raise ClinicalTrialsGovIntakeError(
            "atomic no-replace directory publication is unavailable"
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _safe_bundle_path(root: Path, filename: str) -> Path:
    if not isinstance(filename, str) or not filename:
        raise ClinicalTrialsGovIntakeError("bundle filename must be non-empty")
    pure = PurePosixPath(filename)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ClinicalTrialsGovIntakeError(f"unsafe bundle filename: {filename}")
    if "\\" in filename:
        raise ClinicalTrialsGovIntakeError(f"unsafe bundle filename: {filename}")
    path = root.joinpath(*pure.parts)
    current = root
    for index, part in enumerate(pure.parts):
        current = current / part
        try:
            current_mode = current.stat(follow_symlinks=False).st_mode
        except FileNotFoundError as exc:
            raise ClinicalTrialsGovIntakeError(
                f"bundle file escapes or is missing: {filename}"
            ) from exc
        if stat.S_ISLNK(current_mode):
            raise ClinicalTrialsGovIntakeError(
                f"bundle path cannot contain a symlink: {filename}"
            )
        expected_directory = index < len(pure.parts) - 1
        if expected_directory and not stat.S_ISDIR(current_mode):
            raise ClinicalTrialsGovIntakeError(
                f"bundle parent is not a directory: {filename}"
            )
        if not expected_directory and not stat.S_ISREG(current_mode):
            raise ClinicalTrialsGovIntakeError(
                f"bundle entry is not a regular file: {filename}"
            )
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise ClinicalTrialsGovIntakeError(
            f"bundle file escapes or is missing: {filename}"
        ) from exc
    return path


def verify_clinicaltrials_gov_snapshot(snapshot_dir: str | Path) -> dict[str, object]:
    """Rebuild and verify a frozen bundle without network access."""

    root = Path(snapshot_dir)
    if root.is_symlink() or not root.is_dir():
        raise ClinicalTrialsGovIntakeError(
            "snapshot directory must be a real directory"
        )
    root_mode = root.stat(follow_symlinks=False).st_mode
    if not stat.S_ISDIR(root_mode):
        raise ClinicalTrialsGovIntakeError("snapshot path is not a directory")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ClinicalTrialsGovIntakeError("snapshot manifest.json is missing")
    if manifest_path.stat().st_size > 20 * 1024 * 1024:
        raise ClinicalTrialsGovIntakeError("snapshot manifest is unexpectedly large")
    manifest_bytes = manifest_path.read_bytes()
    manifest = _load_json_object(manifest_bytes, label="manifest.json")
    _validate_manifest_document(manifest)

    outputs = manifest["outputs"]
    pages = manifest["pages"]
    source = manifest["source"]
    query = manifest["query"]
    retrieval = manifest["retrieval"]
    version_envelope = manifest["version_envelope"]
    software = manifest["software"]
    assert isinstance(outputs, list)
    assert isinstance(pages, list)
    assert isinstance(source, dict)
    assert isinstance(query, dict)
    assert isinstance(retrieval, dict)
    assert isinstance(version_envelope, dict)
    assert isinstance(software, dict)
    if len(pages) > HARD_MAX_PAGES:
        raise ClinicalTrialsGovIntakeError("snapshot page count exceeds verifier limit")
    if manifest["page_count"] != len(pages):
        raise ClinicalTrialsGovIntakeError("manifest page_count is inconsistent")

    expected_filenames = [
        "version_start.json",
        *(f"pages/page_{index:06d}.json" for index in range(1, len(pages) + 1)),
        "version_end.json",
        "study_inventory.tsv",
        "curation_queue.tsv",
        "data_assets.tsv",
    ]
    expected_roles = [
        "api_version_start",
        *("studies_page" for _ in pages),
        "api_version_end",
        "deterministic_source_projection_inventory",
        "unreviewed_mapping_queue",
        "raw_response_asset_manifest",
    ]
    for entry in outputs:
        assert isinstance(entry, dict)
        filename = entry.get("filename")
        if not isinstance(filename, str):
            raise ClinicalTrialsGovIntakeError(
                "manifest output filename must be a string"
            )
        _safe_bundle_path(root, filename)
    if [entry.get("filename") for entry in outputs] != expected_filenames:
        raise ClinicalTrialsGovIntakeError(
            "manifest output roster/order is not canonical"
        )
    if [entry.get("role") for entry in outputs] != expected_roles:
        raise ClinicalTrialsGovIntakeError("manifest output roles are not canonical")

    output_by_name: dict[str, dict[str, object]] = {}
    observed_derived_bytes = 0
    for entry in outputs:
        assert isinstance(entry, dict)
        filename = entry["filename"]
        sha256 = entry["sha256"]
        byte_size = entry["byte_size"]
        assert isinstance(filename, str)
        assert isinstance(sha256, str)
        assert isinstance(byte_size, int)
        path = _safe_bundle_path(root, filename)
        observed_size = path.stat().st_size
        if filename in _FIXED_DERIVED_FILENAMES:
            observed_derived_bytes += observed_size
            if observed_derived_bytes > retrieval["max_derived_bytes"]:
                raise ClinicalTrialsGovIntakeError(
                    "derived snapshot tables exceed the declared byte limit"
                )
        elif filename in {"version_start.json", "version_end.json"}:
            if observed_size > 1024 * 1024:
                raise ClinicalTrialsGovIntakeError(
                    "version response exceeds verifier byte limit"
                )
            if observed_size > retrieval["max_page_bytes"]:
                raise ClinicalTrialsGovIntakeError(
                    "version response exceeds declared max_page_bytes"
                )
        elif observed_size > HARD_MAX_PAGE_BYTES:
            raise ClinicalTrialsGovIntakeError("raw page exceeds verifier byte limit")
        if observed_size != byte_size or _file_sha256(path) != sha256:
            raise ClinicalTrialsGovIntakeError(
                f"snapshot output checksum/size mismatch: {filename}"
            )
        output_by_name[filename] = entry

    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        mode = path.stat(follow_symlinks=False).st_mode
        if stat.S_ISLNK(mode):
            raise ClinicalTrialsGovIntakeError(
                f"snapshot directory contains a symlink: {relative}"
            )
        if stat.S_ISREG(mode):
            if relative != "manifest.json":
                actual_files.add(relative)
        elif stat.S_ISDIR(mode):
            actual_directories.add(relative)
        else:
            raise ClinicalTrialsGovIntakeError(
                f"snapshot directory contains a special file: {relative}"
            )
    if actual_files != set(expected_filenames) or actual_directories != {"pages"}:
        raise ClinicalTrialsGovIntakeError(
            "snapshot directory contains unmanifested files or directories"
        )

    version_start_path = _safe_bundle_path(root, "version_start.json")
    version_end_path = _safe_bundle_path(root, "version_end.json")
    version_start_bytes = version_start_path.read_bytes()
    version_end_bytes = version_end_path.read_bytes()
    if len(version_start_bytes) > 1024 * 1024 or len(version_end_bytes) > 1024 * 1024:
        raise ClinicalTrialsGovIntakeError(
            "version response exceeds verifier byte limit"
        )
    if (
        len(version_start_bytes) > retrieval["max_page_bytes"]
        or len(version_end_bytes) > retrieval["max_page_bytes"]
    ):
        raise ClinicalTrialsGovIntakeError(
            "version response exceeds declared max_page_bytes"
        )
    version_start_value = _load_json_object(
        version_start_bytes, label="version_start.json"
    )
    version_end_value = _load_json_object(version_end_bytes, label="version_end.json")
    parsed_start = _parse_version(version_start_value)
    parsed_end = _parse_version(version_end_value)
    if parsed_start[:2] != parsed_end[:2]:
        raise ClinicalTrialsGovIntakeError(
            "API version/data timestamp changed across the snapshot"
        )
    synthetic_fixture = source.get("synthetic_fixture")
    if not isinstance(synthetic_fixture, bool):
        raise ClinicalTrialsGovIntakeError(
            "manifest source synthetic_fixture flag is invalid"
        )
    if synthetic_fixture:
        _require_explicit_synthetic_version(version_start_value)
        _require_explicit_synthetic_version(version_end_value)
    transport_mode = source["transport_mode"]
    clock_mode = source["clock_mode"]
    elapsed_clock_mode = source["elapsed_clock_mode"]
    assert isinstance(transport_mode, str)
    assert isinstance(clock_mode, str)
    assert isinstance(elapsed_clock_mode, str)
    if synthetic_fixture:
        expected_source_name = "ClinicalTrials.gov synthetic fixture"
        expected_mutable_registry: bool | None = False
        expected_transport_mode = "injected"
        expected_clock_mode = "injected"
        expected_elapsed_clock_mode = "injected"
    elif transport_mode == "injected":
        expected_source_name = "ClinicalTrials.gov API-shaped injected transport"
        expected_mutable_registry = None
        expected_transport_mode = "injected"
        expected_clock_mode = clock_mode
        expected_elapsed_clock_mode = elapsed_clock_mode
    else:
        expected_source_name = "ClinicalTrials.gov"
        expected_mutable_registry = True
        expected_transport_mode = "live_https"
        expected_clock_mode = "system_utc"
        expected_elapsed_clock_mode = "system_monotonic"
    expected_source_values = {
        "name": expected_source_name,
        "synthetic_fixture": synthetic_fixture,
        "transport_mode": expected_transport_mode,
        "clock_mode": expected_clock_mode,
        "elapsed_clock_mode": expected_elapsed_clock_mode,
        "api_base_url": API_BASE_URL,
        "studies_endpoint": STUDIES_ENDPOINT,
        "version_endpoint": VERSION_ENDPOINT,
        "api_documentation_url": API_DOCUMENTATION_URL,
        "terms_url": TERMS_URL,
        "api_version": parsed_start[0],
        "data_timestamp": parsed_start[1],
        "data_timestamp_timezone_interpretation": DATA_TIMESTAMP_INTERPRETATION,
        "mutable_registry": expected_mutable_registry,
    }
    for key, expected_value in expected_source_values.items():
        if source.get(key) != expected_value:
            raise ClinicalTrialsGovIntakeError(
                f"manifest source field is not canonical: {key}"
            )

    condition_query = _normalize_query_text(
        query.get("condition"), label="manifest condition query"
    )
    intervention_query = _normalize_query_text(
        query.get("intervention"), label="manifest intervention query"
    )
    page_size = query["page_size"]
    assert isinstance(page_size, int)
    expected_query = {
        "condition": condition_query,
        "intervention": intervention_query,
        "page_size": page_size,
        "count_total": True,
        "format": "json",
        "markup_format": "markdown",
        "fields": list(STUDY_FIELDS),
    }
    if query != expected_query:
        raise ClinicalTrialsGovIntakeError("manifest query is not canonical")
    if manifest["query_sha256"] != _sha256_bytes(_canonical_json_bytes(expected_query)):
        raise ClinicalTrialsGovIntakeError("manifest query digest is inconsistent")

    required_completeness_basis = [
        "pagination_terminated_without_nextPageToken",
        "page_tokens_did_not_repeat",
        "NCT_identifiers_are_unique_across_pages",
        "observed_unique_study_count_equals_totalCount",
        "API_version_and_dataTimestamp_stable_before_and_after",
    ]
    if manifest["pagination_completeness_basis"] != required_completeness_basis:
        raise ClinicalTrialsGovIntakeError(
            "manifest pagination completeness basis is not canonical"
        )
    scientific_boundary = manifest["scientific_boundary"]
    assert isinstance(scientific_boundary, dict)
    expected_scientific_boundary = {
        "query_match_is_exact_concept_mapping": False,
        "curation_queue_review_status": "not_performed",
        "intervention_condition_rows_are": (
            "study_level_co_mentions_not_arm_or_cohort_linkage"
        ),
        "treatment_cancer_linkage_review_status": "not_performed",
        "curation_queue_eligible_for_clinical_context": False,
        "used_for_gene_ranking": False,
        "used_for_validation_label": False,
        "registry_presence_is_efficacy": False,
        "results_posted_is_endpoint_met": False,
        "synthetic_fixture": synthetic_fixture,
    }
    if scientific_boundary != expected_scientific_boundary:
        raise ClinicalTrialsGovIntakeError(
            "manifest scientific boundary is not canonical"
        )
    if version_envelope["stable_before_and_after_pagination"] is not True:
        raise ClinicalTrialsGovIntakeError("version envelope is not stable")
    if version_envelope["snapshot_isolation_claim"] != SNAPSHOT_ISOLATION_CLAIM:
        raise ClinicalTrialsGovIntakeError(
            "manifest snapshot-isolation boundary is not canonical"
        )
    if manifest["integrity_scope"] != INTEGRITY_SCOPE:
        raise ClinicalTrialsGovIntakeError(
            "manifest integrity boundary is not canonical"
        )
    if software.get("package") != "crispr-evidencerank":
        raise ClinicalTrialsGovIntakeError("manifest software package is not canonical")
    if software.get("package_version") != __version__:
        raise ClinicalTrialsGovIntakeError(
            "snapshot must be verified with its recorded package version"
        )
    for field_name in (
        "version_start_retrieved_at_utc",
        "version_end_retrieved_at_utc",
    ):
        if not isinstance(version_envelope[field_name], str):
            raise ClinicalTrialsGovIntakeError(
                f"manifest {field_name} must be an ISO timestamp string"
            )
    for field_name in ("started_at_utc", "completed_at_utc"):
        if not isinstance(retrieval[field_name], str):
            raise ClinicalTrialsGovIntakeError(
                f"manifest {field_name} must be an ISO timestamp string"
            )
    if len(pages) > retrieval["max_pages"]:
        raise ClinicalTrialsGovIntakeError("page roster exceeds declared max_pages")

    all_ids: list[str] = []
    seen_study_ids: set[str] = set()
    expected_inventory_rows: list[dict[str, object]] = []
    expected_candidate_rows: list[dict[str, object]] = []
    expected_token: str | None = None
    seen_tokens: set[str] = set()
    observed_total: int | None = None
    page_sha256_values: list[str] = []
    total_raw_bytes = len(version_start_bytes) + len(version_end_bytes)
    derived_work_bytes = 0
    for expected_index, page_entry in enumerate(pages, start=1):
        assert isinstance(page_entry, dict)
        if page_entry["page_index"] != expected_index:
            raise ClinicalTrialsGovIntakeError(
                "manifest page indices are not contiguous"
            )
        filename = f"pages/page_{expected_index:06d}.json"
        if page_entry["filename"] != filename:
            raise ClinicalTrialsGovIntakeError(
                "manifest page filename is not canonical"
            )
        if page_entry["requested_page_token"] != expected_token:
            raise ClinicalTrialsGovIntakeError(
                "manifest pagination token chain is broken"
            )
        request_url = _studies_url(
            condition_query=condition_query,
            intervention_query=intervention_query,
            page_size=page_size,
            page_token=expected_token,
        )
        if page_entry["request_url"] != request_url:
            raise ClinicalTrialsGovIntakeError(
                "manifest page request URL is not canonical"
            )
        if page_entry["response_url"] != request_url:
            raise ClinicalTrialsGovIntakeError(
                "manifest page response URL is not canonical"
            )
        if page_entry["attempt_count"] > retrieval["max_attempts"]:
            raise ClinicalTrialsGovIntakeError(
                "manifest page attempts exceed the declared retry limit"
            )
        if not isinstance(page_entry["retrieved_at_utc"], str):
            raise ClinicalTrialsGovIntakeError(
                "manifest page retrieval time must be an ISO timestamp string"
            )
        page_path = _safe_bundle_path(root, filename)
        page_bytes = page_path.read_bytes()
        page_size_bytes = len(page_bytes)
        page_sha256 = _sha256_bytes(page_bytes)
        if page_size_bytes > HARD_MAX_PAGE_BYTES:
            raise ClinicalTrialsGovIntakeError("raw page exceeds verifier byte limit")
        if (
            page_entry["byte_size"] != page_size_bytes
            or page_entry["sha256"] != page_sha256
        ):
            raise ClinicalTrialsGovIntakeError(
                "manifest page checksum/size differs from its raw file"
            )
        if (
            output_by_name[filename]["byte_size"] != page_size_bytes
            or output_by_name[filename]["sha256"] != page_sha256
        ):
            raise ClinicalTrialsGovIntakeError(
                "manifest page entry differs from its output entry"
            )
        headers = page_entry["response_headers"]
        assert isinstance(headers, dict)
        if not set(headers).issubset(_CAPTURED_HEADERS) or any(
            key != key.lower() for key in headers
        ):
            raise ClinicalTrialsGovIntakeError(
                "manifest page response headers are not from the safe allowlist"
            )
        if not _is_json_media_type(str(headers.get("content-type", ""))):
            raise ClinicalTrialsGovIntakeError(
                "manifest page lacks an application/json Content-Type"
            )
        if str(headers.get("content-encoding", "identity")).lower() != "identity":
            raise ClinicalTrialsGovIntakeError(
                "manifest page Content-Encoding is not identity"
            )
        if "content-length" in headers:
            try:
                manifest_content_length = int(headers["content-length"])
            except (TypeError, ValueError) as exc:
                raise ClinicalTrialsGovIntakeError(
                    "manifest page Content-Length is invalid"
                ) from exc
            if manifest_content_length != page_size_bytes:
                raise ClinicalTrialsGovIntakeError(
                    "manifest page Content-Length differs from raw bytes"
                )
        page_value = _load_json_object(page_bytes, label=filename)
        if synthetic_fixture:
            _require_explicit_synthetic_page(page_value)
        (
            page_ids,
            page_inventory,
            page_candidates,
            next_token,
            total_count,
            page_derived_work_bytes,
        ) = _parse_studies_page(
            page_value,
            page_index=expected_index,
            page_sha256=page_sha256,
            retrieved_at_utc=page_entry["retrieved_at_utc"],
            condition_query=condition_query,
            intervention_query=intervention_query,
            max_candidate_rows=(
                retrieval["max_candidate_rows"] - len(expected_candidate_rows)
            ),
            max_derived_work_bytes=(
                retrieval["max_derived_bytes"] - derived_work_bytes
            ),
        )
        derived_work_bytes += page_derived_work_bytes
        if expected_index == 1:
            assert total_count is not None
            observed_total = total_count
        elif total_count is not None and total_count != observed_total:
            raise ClinicalTrialsGovIntakeError("totalCount changed between pages")
        if (
            next_token is not None
            and observed_total is not None
            and len(all_ids) + len(page_ids) >= observed_total
        ):
            raise ClinicalTrialsGovIntakeError(
                "nextPageToken remains after totalCount studies were observed"
            )
        duplicate_ids = sorted(seen_study_ids.intersection(page_ids))
        if duplicate_ids:
            raise ClinicalTrialsGovIntakeError(
                f"NCT identifiers repeat across pages: {duplicate_ids[:5]}"
            )
        if page_entry["study_ids"] != page_ids:
            raise ClinicalTrialsGovIntakeError(
                "manifest study IDs differ from raw page"
            )
        if page_entry["study_count"] != len(page_ids):
            raise ClinicalTrialsGovIntakeError(
                "manifest page study_count is inconsistent"
            )
        if page_entry["next_page_token"] != next_token:
            raise ClinicalTrialsGovIntakeError(
                "manifest nextPageToken differs from raw page"
            )
        if page_entry["total_count"] != total_count:
            raise ClinicalTrialsGovIntakeError(
                "manifest page totalCount differs from raw page"
            )
        if page_entry["study_ids_sha256"] != _sha256_bytes(
            _canonical_json_bytes(page_ids)
        ):
            raise ClinicalTrialsGovIntakeError(
                "manifest page NCT roster hash is inconsistent"
            )
        all_ids.extend(page_ids)
        seen_study_ids.update(page_ids)
        expected_inventory_rows.extend(page_inventory)
        expected_candidate_rows.extend(page_candidates)
        if len(expected_candidate_rows) > retrieval["max_candidate_rows"]:
            raise ClinicalTrialsGovIntakeError(
                "curation candidates exceed the declared row limit"
            )
        page_sha256_values.append(page_sha256)
        total_raw_bytes += page_size_bytes
        if next_token is not None:
            if next_token in seen_tokens:
                raise ClinicalTrialsGovIntakeError("pagination token repeated")
            seen_tokens.add(next_token)
        expected_token = next_token
    if expected_token is not None:
        raise ClinicalTrialsGovIntakeError("final page still contains nextPageToken")
    if observed_total is None or observed_total != len(all_ids):
        raise ClinicalTrialsGovIntakeError(
            "observed unique study count does not equal totalCount"
        )
    if manifest["total_count"] != observed_total or manifest[
        "observed_unique_study_count"
    ] != len(all_ids):
        raise ClinicalTrialsGovIntakeError("manifest study totals are inconsistent")
    if len(all_ids) > retrieval["max_studies"]:
        raise ClinicalTrialsGovIntakeError("study roster exceeds declared max_studies")
    if any(page["byte_size"] > retrieval["max_page_bytes"] for page in pages):
        raise ClinicalTrialsGovIntakeError("raw page exceeds declared max_page_bytes")
    if total_raw_bytes != retrieval["total_raw_response_bytes"]:
        raise ClinicalTrialsGovIntakeError(
            "manifest total raw byte count is inconsistent"
        )
    if total_raw_bytes > retrieval["max_total_bytes"]:
        raise ClinicalTrialsGovIntakeError("snapshot exceeds declared max_total_bytes")

    raw_identity = {
        "api_version": parsed_start[0],
        "data_timestamp": parsed_start[1],
        "synthetic_fixture": synthetic_fixture,
        "transport_mode": transport_mode,
        "clock_mode": clock_mode,
        "elapsed_clock_mode": elapsed_clock_mode,
        "query": expected_query,
        "version_start_sha256": _sha256_bytes(version_start_bytes),
        "page_sha256": page_sha256_values,
        "version_end_sha256": _sha256_bytes(version_end_bytes),
    }
    expected_snapshot_id = (
        f"ctgov-{_sha256_bytes(_canonical_json_bytes(raw_identity))[:20]}"
    )
    if manifest["snapshot_id"] != expected_snapshot_id:
        raise ClinicalTrialsGovIntakeError("snapshot_id does not match raw identity")

    provenance_namespace = (
        "crispr-evidencerank:synthetic-ctgov" if synthetic_fixture else "ctgov"
    )
    page_asset_ids = {
        index: f"{provenance_namespace}:{expected_snapshot_id}:page:{index:06d}"
        for index in range(1, len(pages) + 1)
    }
    _bind_snapshot_rows(
        expected_inventory_rows,
        expected_candidate_rows,
        snapshot_id=expected_snapshot_id,
        page_asset_ids=page_asset_ids,
        synthetic_fixture=synthetic_fixture,
    )
    expected_inventory = _validated_frame(
        expected_inventory_rows,
        ClinicalTrialsGovStudyInventoryRecord,
        label="rederived study inventory",
    )
    expected_queue = _validated_frame(
        expected_candidate_rows,
        ClinicalTrialsGovCurationCandidateRecord,
        label="rederived curation queue",
    )
    inventory_path = _safe_bundle_path(root, "study_inventory.tsv")
    queue_path = _safe_bundle_path(root, "curation_queue.tsv")
    if inventory_path.read_bytes() != _frame_tsv_bytes(expected_inventory):
        raise ClinicalTrialsGovIntakeError(
            "derived study inventory differs from frozen raw pages"
        )
    if queue_path.read_bytes() != _frame_tsv_bytes(expected_queue):
        raise ClinicalTrialsGovIntakeError(
            "derived curation queue differs from frozen raw pages"
        )

    source_version = f"api-{parsed_start[0]};data-{parsed_start[1]}"
    raw_family_id = f"{provenance_namespace}:raw-snapshot:{expected_snapshot_id}"
    build_revision = software.get("build_revision")
    if build_revision is not None and not isinstance(build_revision, str):
        raise ClinicalTrialsGovIntakeError("manifest build revision is invalid")
    expected_asset_rows = [
        _asset_row(
            asset_id=f"{provenance_namespace}:{expected_snapshot_id}:version:start",
            source_version=source_version,
            asset_role="registry_api_version_response",
            accession=expected_snapshot_id,
            source_url=_asset_source_url(
                VERSION_ENDPOINT,
                synthetic_fixture=synthetic_fixture,
                snapshot_id=expected_snapshot_id,
                asset_suffix="version-start",
            ),
            available_date=parsed_start[2],
            retrieved_at_utc=version_envelope["version_start_retrieved_at_utc"],
            sha256=_sha256_bytes(version_start_bytes),
            byte_size=len(version_start_bytes),
            raw_data_family_id=raw_family_id,
            notes=_asset_note(
                (
                    "Pre-pagination version envelope; exact response bytes retained. "
                    "Redistribution is not asserted by this project."
                ),
                synthetic_fixture=synthetic_fixture,
                transport_mode=transport_mode,
            ),
            code_commit=build_revision,
            synthetic_fixture=synthetic_fixture,
            transport_mode=transport_mode,
        )
    ]
    for page_entry in pages:
        page_index = page_entry["page_index"]
        expected_asset_rows.append(
            _asset_row(
                asset_id=page_asset_ids[page_index],
                source_version=source_version,
                asset_role="registry_api_page",
                accession=expected_snapshot_id,
                source_url=_asset_source_url(
                    page_entry["request_url"],
                    synthetic_fixture=synthetic_fixture,
                    snapshot_id=expected_snapshot_id,
                    asset_suffix=f"page-{page_index:06d}",
                ),
                available_date=parsed_start[2],
                retrieved_at_utc=page_entry["retrieved_at_utc"],
                sha256=page_entry["sha256"],
                byte_size=page_entry["byte_size"],
                raw_data_family_id=raw_family_id,
                notes=_asset_note(
                    (
                        "Frozen JSON page using the manifest-pinned scientific "
                        "field projection. Query matching is retrieval only and "
                        "is not a reviewed treatment/cancer mapping."
                    ),
                    synthetic_fixture=synthetic_fixture,
                    transport_mode=transport_mode,
                ),
                code_commit=build_revision,
                synthetic_fixture=synthetic_fixture,
                transport_mode=transport_mode,
            )
        )
    expected_asset_rows.append(
        _asset_row(
            asset_id=f"{provenance_namespace}:{expected_snapshot_id}:version:end",
            source_version=source_version,
            asset_role="registry_api_version_response",
            accession=expected_snapshot_id,
            source_url=_asset_source_url(
                VERSION_ENDPOINT,
                synthetic_fixture=synthetic_fixture,
                snapshot_id=expected_snapshot_id,
                asset_suffix="version-end",
            ),
            available_date=parsed_start[2],
            retrieved_at_utc=version_envelope["version_end_retrieved_at_utc"],
            sha256=_sha256_bytes(version_end_bytes),
            byte_size=len(version_end_bytes),
            raw_data_family_id=raw_family_id,
            notes=_asset_note(
                (
                    "Post-pagination version envelope; values must equal the "
                    "pre-pagination envelope. Redistribution is not asserted."
                ),
                synthetic_fixture=synthetic_fixture,
                transport_mode=transport_mode,
            ),
            code_commit=build_revision,
            synthetic_fixture=synthetic_fixture,
            transport_mode=transport_mode,
        )
    )
    derived_work_bytes += sum(
        len(_canonical_json_bytes(row)) for row in expected_asset_rows
    )
    if derived_work_bytes > retrieval["max_derived_bytes"]:
        raise ClinicalTrialsGovIntakeError(
            "derived row materialization exceeds the declared byte limit"
        )
    expected_assets = _validated_frame(
        expected_asset_rows,
        DataAssetRecord,
        label="rederived data assets",
    )
    assets_path = _safe_bundle_path(root, "data_assets.tsv")
    if assets_path.read_bytes() != _frame_tsv_bytes(expected_assets):
        raise ClinicalTrialsGovIntakeError(
            "derived data assets differ from frozen raw responses"
        )
    rederived_bytes = sum(
        len(content)
        for content in (
            _frame_tsv_bytes(expected_inventory),
            _frame_tsv_bytes(expected_queue),
            _frame_tsv_bytes(expected_assets),
        )
    )
    if rederived_bytes != observed_derived_bytes:
        raise ClinicalTrialsGovIntakeError(
            "rederived table byte count differs from the manifest"
        )
    if rederived_bytes != retrieval["total_derived_bytes"]:
        raise ClinicalTrialsGovIntakeError(
            "manifest total derived byte count is inconsistent"
        )

    content_index = [
        {
            "filename": entry["filename"],
            "sha256": entry["sha256"],
            "byte_size": entry["byte_size"],
        }
        for entry in outputs
    ]
    if manifest["bundle_content_sha256"] != _sha256_bytes(
        _canonical_json_bytes(content_index)
    ):
        raise ClinicalTrialsGovIntakeError("bundle content digest is inconsistent")
    if manifest_path.read_bytes() != manifest_bytes:
        raise ClinicalTrialsGovIntakeError(
            "snapshot manifest changed during verification"
        )
    for entry in outputs:
        filename = entry["filename"]
        path = _safe_bundle_path(root, filename)
        if (
            path.stat().st_size != entry["byte_size"]
            or _file_sha256(path) != entry["sha256"]
        ):
            raise ClinicalTrialsGovIntakeError(
                f"snapshot output changed during verification: {filename}"
            )
    return {
        "snapshot_id": expected_snapshot_id,
        "api_version": parsed_start[0],
        "data_timestamp": parsed_start[1],
        "page_count": len(pages),
        "study_count": len(all_ids),
        "complete": True,
        "integrity_scope": "internal_bundle_consistency_not_publisher_authenticity",
    }


def fetch_clinicaltrials_gov_snapshot(
    *,
    condition_query: str,
    intervention_query: str,
    output_dir: str | Path,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_studies: int = DEFAULT_MAX_STUDIES,
    max_candidate_rows: int = DEFAULT_MAX_CANDIDATE_ROWS,
    max_page_bytes: int = DEFAULT_MAX_PAGE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_derived_bytes: int = DEFAULT_MAX_DERIVED_BYTES,
    max_elapsed_seconds: float = DEFAULT_MAX_ELAPSED_SECONDS,
    synthetic_fixture: bool = False,
    requester: JsonRequester | None = None,
    clock: Callable[[], datetime] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> dict[str, object]:
    """Fetch every API page and atomically publish a checksum-bound snapshot."""

    condition_query = _normalize_query_text(condition_query, label="condition_query")
    intervention_query = _normalize_query_text(
        intervention_query, label="intervention_query"
    )
    _validate_limits(
        page_size=page_size,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
        max_pages=max_pages,
        max_studies=max_studies,
        max_candidate_rows=max_candidate_rows,
        max_page_bytes=max_page_bytes,
        max_total_bytes=max_total_bytes,
        max_derived_bytes=max_derived_bytes,
        max_elapsed_seconds=max_elapsed_seconds,
    )
    requester_injected = requester is not None
    clock_injected = clock is not None
    elapsed_clock_injected = monotonic is not None
    if not isinstance(synthetic_fixture, bool):
        raise ValueError("synthetic_fixture must be boolean")
    if synthetic_fixture and not requester_injected:
        raise ValueError("synthetic_fixture requires an injected offline requester")
    if synthetic_fixture and not (clock_injected and elapsed_clock_injected):
        raise ValueError("synthetic_fixture requires injected deterministic clocks")
    if not requester_injected and (clock_injected or elapsed_clock_injected):
        raise ValueError("live HTTPS retrieval requires system clocks")
    transport_mode = "injected" if requester_injected else "live_https"
    clock_mode = "injected" if clock_injected else "system_utc"
    elapsed_clock_mode = "injected" if elapsed_clock_injected else "system_monotonic"
    requester = requester or _default_json_requester
    clock = clock or _utc_now
    monotonic = monotonic or time.monotonic
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.parent.is_symlink() or not output_dir.parent.is_dir():
        raise ClinicalTrialsGovIntakeError("snapshot output parent cannot be a symlink")
    lock_descriptor = _acquire_publication_lock(output_dir.parent)
    staging: Path | None = None
    try:
        _fsync_directory(output_dir.parent)
        if os.path.lexists(output_dir):
            raise FileExistsError(
                f"ClinicalTrials.gov snapshot output exists: {output_dir}"
            )
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.staging-",
                dir=output_dir.parent,
            )
        )
        (staging / "pages").mkdir()
        monotonic_start = monotonic()

        def assert_elapsed_within_limit() -> None:
            elapsed = monotonic() - monotonic_start
            if elapsed < 0 or elapsed > max_elapsed_seconds:
                raise ClinicalTrialsGovIntakeError(
                    "snapshot retrieval exceeded max_elapsed_seconds"
                )

        def request_timeout_within_limit() -> float:
            elapsed = monotonic() - monotonic_start
            remaining = max_elapsed_seconds - elapsed
            if elapsed < 0 or remaining <= 0:
                raise ClinicalTrialsGovIntakeError(
                    "snapshot retrieval exceeded max_elapsed_seconds"
                )
            return min(timeout_seconds, remaining)

        started_at = _utc_iso(clock(), label="retrieval start time")
        version_start_response, version_start_value = _request_json(
            requester,
            VERSION_ENDPOINT,
            expected_path="/api/v2/version",
            timeout_seconds=request_timeout_within_limit(),
            max_response_bytes=min(max_page_bytes, 1024 * 1024, max_total_bytes),
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
        )
        version_start_retrieved = _utc_iso(
            clock(), label="version-start retrieval time"
        )
        (staging / "version_start.json").write_bytes(version_start_response.body)
        api_version, data_timestamp, source_available_date = _parse_version(
            version_start_value
        )
        if synthetic_fixture:
            _require_explicit_synthetic_version(version_start_value)

        page_captures: list[_PageCapture] = []
        inventory_rows: list[dict[str, object]] = []
        candidate_rows: list[dict[str, object]] = []
        derived_work_bytes = 0
        all_study_ids: list[str] = []
        seen_study_ids: set[str] = set()
        observed_total: int | None = None
        page_token: str | None = None
        seen_tokens: set[str] = set()
        total_raw_bytes = len(version_start_response.body)
        manifest_page_work_bytes = 0
        page_index = 0
        while True:
            page_index += 1
            if page_index > max_pages:
                raise ClinicalTrialsGovIntakeError(
                    "pagination did not terminate before max_pages"
                )
            request_url = _studies_url(
                condition_query=condition_query,
                intervention_query=intervention_query,
                page_size=page_size,
                page_token=page_token,
            )
            remaining_raw_bytes = max_total_bytes - total_raw_bytes
            if remaining_raw_bytes <= 0:
                raise ClinicalTrialsGovIntakeError(
                    "snapshot exhausted max_total_bytes before pagination completed"
                )
            response, value = _request_json(
                requester,
                request_url,
                expected_path="/api/v2/studies",
                timeout_seconds=request_timeout_within_limit(),
                max_response_bytes=min(max_page_bytes, remaining_raw_bytes),
                max_attempts=max_attempts,
                backoff_seconds=backoff_seconds,
            )
            retrieved_at = _utc_iso(clock(), label=f"page {page_index} retrieval time")
            assert_elapsed_within_limit()
            total_raw_bytes += len(response.body)
            if total_raw_bytes > max_total_bytes:
                raise ClinicalTrialsGovIntakeError(
                    "snapshot exceeds the configured total byte limit"
                )
            filename = f"pages/page_{page_index:06d}.json"
            page_path = staging / filename
            page_path.write_bytes(response.body)
            page_sha = _sha256_bytes(response.body)
            if synthetic_fixture:
                _require_explicit_synthetic_page(value)
            (
                page_study_ids,
                page_inventory,
                page_candidates,
                next_page_token,
                total_count,
                page_derived_work_bytes,
            ) = _parse_studies_page(
                value,
                page_index=page_index,
                page_sha256=page_sha,
                retrieved_at_utc=retrieved_at,
                condition_query=condition_query,
                intervention_query=intervention_query,
                max_candidate_rows=max_candidate_rows - len(candidate_rows),
                max_derived_work_bytes=(max_derived_bytes - derived_work_bytes),
            )
            derived_work_bytes += page_derived_work_bytes
            if page_index == 1:
                assert total_count is not None
                observed_total = total_count
                if observed_total > max_studies:
                    raise ClinicalTrialsGovIntakeError(
                        "totalCount exceeds the configured study limit"
                    )
            elif total_count is not None and total_count != observed_total:
                raise ClinicalTrialsGovIntakeError("totalCount changed between pages")
            if (
                next_page_token is not None
                and observed_total is not None
                and len(all_study_ids) + len(page_study_ids) >= observed_total
            ):
                raise ClinicalTrialsGovIntakeError(
                    "nextPageToken remains after totalCount studies were observed"
                )
            duplicate_ids = sorted(seen_study_ids.intersection(page_study_ids))
            if duplicate_ids:
                raise ClinicalTrialsGovIntakeError(
                    f"NCT identifiers repeat across pages: {duplicate_ids[:5]}"
                )
            page_capture = _PageCapture(
                page_index=page_index,
                filename=filename,
                request_url=request_url,
                requested_page_token=page_token,
                next_page_token=next_page_token,
                sha256=page_sha,
                byte_size=len(response.body),
                study_ids=tuple(page_study_ids),
                total_count=total_count,
                response_url=response.final_url,
                response_headers=_captured_headers(response.headers),
                attempt_count=response.attempt_count,
                retrieved_at_utc=retrieved_at,
            )
            page_captures.append(page_capture)
            manifest_page_work_bytes += len(
                _canonical_json_bytes(_page_manifest_entry(page_capture))
            )
            manifest_page_work_bytes += 512
            if manifest_page_work_bytes > HARD_MAX_MANIFEST_BYTES:
                raise ClinicalTrialsGovIntakeError(
                    "snapshot page metadata exceeds the manifest size limit"
                )
            all_study_ids.extend(page_study_ids)
            seen_study_ids.update(page_study_ids)
            if len(all_study_ids) > max_studies:
                raise ClinicalTrialsGovIntakeError(
                    "observed study count exceeds the configured limit"
                )
            inventory_rows.extend(page_inventory)
            candidate_rows.extend(page_candidates)
            if next_page_token is None:
                break
            if next_page_token in seen_tokens:
                raise ClinicalTrialsGovIntakeError("pagination token repeated")
            seen_tokens.add(next_page_token)
            page_token = next_page_token

        remaining_raw_bytes = max_total_bytes - total_raw_bytes
        if remaining_raw_bytes <= 0:
            raise ClinicalTrialsGovIntakeError(
                "snapshot exhausted max_total_bytes before the final version check"
            )
        version_end_response, version_end_value = _request_json(
            requester,
            VERSION_ENDPOINT,
            expected_path="/api/v2/version",
            timeout_seconds=request_timeout_within_limit(),
            max_response_bytes=min(
                max_page_bytes,
                1024 * 1024,
                remaining_raw_bytes,
            ),
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
        )
        version_end_retrieved = _utc_iso(clock(), label="version-end retrieval time")
        assert_elapsed_within_limit()
        total_raw_bytes += len(version_end_response.body)
        if total_raw_bytes > max_total_bytes:
            raise ClinicalTrialsGovIntakeError(
                "snapshot exceeds the configured total byte limit"
            )
        (staging / "version_end.json").write_bytes(version_end_response.body)
        end_api_version, end_data_timestamp, _ = _parse_version(version_end_value)
        if synthetic_fixture:
            _require_explicit_synthetic_version(version_end_value)
        if (api_version, data_timestamp) != (end_api_version, end_data_timestamp):
            raise ClinicalTrialsGovIntakeError(
                "ClinicalTrials.gov API version/data timestamp changed during retrieval"
            )
        if observed_total is None or observed_total != len(all_study_ids):
            raise ClinicalTrialsGovIntakeError(
                "complete pagination count does not equal totalCount"
            )

        query_identity = {
            "condition": condition_query,
            "intervention": intervention_query,
            "page_size": page_size,
            "count_total": True,
            "format": "json",
            "markup_format": "markdown",
            "fields": list(STUDY_FIELDS),
        }
        raw_identity = {
            "api_version": api_version,
            "data_timestamp": data_timestamp,
            "synthetic_fixture": synthetic_fixture,
            "transport_mode": transport_mode,
            "clock_mode": clock_mode,
            "elapsed_clock_mode": elapsed_clock_mode,
            "query": query_identity,
            "version_start_sha256": _sha256_bytes(version_start_response.body),
            "page_sha256": [page.sha256 for page in page_captures],
            "version_end_sha256": _sha256_bytes(version_end_response.body),
        }
        snapshot_id = f"ctgov-{_sha256_bytes(_canonical_json_bytes(raw_identity))[:20]}"
        provenance_namespace = (
            "crispr-evidencerank:synthetic-ctgov" if synthetic_fixture else "ctgov"
        )
        raw_family_id = f"{provenance_namespace}:raw-snapshot:{snapshot_id}"
        source_version = f"api-{api_version};data-{data_timestamp}"
        build_revision = os.environ.get("CRISPR_EVIDENCERANK_BUILD_REVISION")

        page_asset_ids = {
            page.page_index: (
                f"{provenance_namespace}:{snapshot_id}:page:{page.page_index:06d}"
            )
            for page in page_captures
        }
        _bind_snapshot_rows(
            inventory_rows,
            candidate_rows,
            snapshot_id=snapshot_id,
            page_asset_ids=page_asset_ids,
            synthetic_fixture=synthetic_fixture,
        )

        inventory = _validated_frame(
            inventory_rows,
            ClinicalTrialsGovStudyInventoryRecord,
            label="study inventory",
        )
        queue = _validated_frame(
            candidate_rows,
            ClinicalTrialsGovCurationCandidateRecord,
            label="curation queue",
        )
        asset_rows = [
            _asset_row(
                asset_id=f"{provenance_namespace}:{snapshot_id}:version:start",
                source_version=source_version,
                asset_role="registry_api_version_response",
                accession=snapshot_id,
                source_url=_asset_source_url(
                    VERSION_ENDPOINT,
                    synthetic_fixture=synthetic_fixture,
                    snapshot_id=snapshot_id,
                    asset_suffix="version-start",
                ),
                available_date=source_available_date,
                retrieved_at_utc=version_start_retrieved,
                sha256=_sha256_bytes(version_start_response.body),
                byte_size=len(version_start_response.body),
                raw_data_family_id=raw_family_id,
                notes=_asset_note(
                    (
                        "Pre-pagination version envelope; exact response bytes "
                        "retained. Redistribution is not asserted by this project."
                    ),
                    synthetic_fixture=synthetic_fixture,
                    transport_mode=transport_mode,
                ),
                code_commit=build_revision,
                synthetic_fixture=synthetic_fixture,
                transport_mode=transport_mode,
            )
        ]
        for page in page_captures:
            asset_rows.append(
                _asset_row(
                    asset_id=page_asset_ids[page.page_index],
                    source_version=source_version,
                    asset_role="registry_api_page",
                    accession=snapshot_id,
                    source_url=_asset_source_url(
                        page.request_url,
                        synthetic_fixture=synthetic_fixture,
                        snapshot_id=snapshot_id,
                        asset_suffix=f"page-{page.page_index:06d}",
                    ),
                    available_date=source_available_date,
                    retrieved_at_utc=page.retrieved_at_utc,
                    sha256=page.sha256,
                    byte_size=page.byte_size,
                    raw_data_family_id=raw_family_id,
                    notes=_asset_note(
                        (
                            "Frozen JSON page using the manifest-pinned scientific "
                            "field projection. Query matching is retrieval only and "
                            "is not a reviewed treatment/cancer mapping."
                        ),
                        synthetic_fixture=synthetic_fixture,
                        transport_mode=transport_mode,
                    ),
                    code_commit=build_revision,
                    synthetic_fixture=synthetic_fixture,
                    transport_mode=transport_mode,
                )
            )
        asset_rows.append(
            _asset_row(
                asset_id=f"{provenance_namespace}:{snapshot_id}:version:end",
                source_version=source_version,
                asset_role="registry_api_version_response",
                accession=snapshot_id,
                source_url=_asset_source_url(
                    VERSION_ENDPOINT,
                    synthetic_fixture=synthetic_fixture,
                    snapshot_id=snapshot_id,
                    asset_suffix="version-end",
                ),
                available_date=source_available_date,
                retrieved_at_utc=version_end_retrieved,
                sha256=_sha256_bytes(version_end_response.body),
                byte_size=len(version_end_response.body),
                raw_data_family_id=raw_family_id,
                notes=_asset_note(
                    (
                        "Post-pagination version envelope; values must equal the "
                        "pre-pagination envelope. Redistribution is not asserted."
                    ),
                    synthetic_fixture=synthetic_fixture,
                    transport_mode=transport_mode,
                ),
                code_commit=build_revision,
                synthetic_fixture=synthetic_fixture,
                transport_mode=transport_mode,
            )
        )
        derived_work_bytes += sum(len(_canonical_json_bytes(row)) for row in asset_rows)
        if derived_work_bytes > max_derived_bytes:
            raise ClinicalTrialsGovIntakeError(
                "derived row materialization exceeds the configured byte limit"
            )
        assets = _validated_frame(asset_rows, DataAssetRecord, label="data assets")
        derived_content = {
            "study_inventory.tsv": _frame_tsv_bytes(inventory),
            "curation_queue.tsv": _frame_tsv_bytes(queue),
            "data_assets.tsv": _frame_tsv_bytes(assets),
        }
        total_derived_bytes = sum(len(content) for content in derived_content.values())
        if total_derived_bytes > max_derived_bytes:
            raise ClinicalTrialsGovIntakeError(
                "derived snapshot tables exceed the configured byte limit"
            )
        for filename, content in derived_content.items():
            (staging / filename).write_bytes(content)

        assert_elapsed_within_limit()
        completed_at = _utc_iso(clock(), label="retrieval completion time")
        outputs = [
            _output_entry(
                staging / "version_start.json",
                role="api_version_start",
                filename="version_start.json",
            ),
            *[
                _output_entry(
                    staging / page.filename,
                    role="studies_page",
                    filename=page.filename,
                )
                for page in page_captures
            ],
            _output_entry(
                staging / "version_end.json",
                role="api_version_end",
                filename="version_end.json",
            ),
            _output_entry(
                staging / "study_inventory.tsv",
                role="deterministic_source_projection_inventory",
                filename="study_inventory.tsv",
            ),
            _output_entry(
                staging / "curation_queue.tsv",
                role="unreviewed_mapping_queue",
                filename="curation_queue.tsv",
            ),
            _output_entry(
                staging / "data_assets.tsv",
                role="raw_response_asset_manifest",
                filename="data_assets.tsv",
            ),
        ]
        content_index = [
            {
                "filename": output["filename"],
                "sha256": output["sha256"],
                "byte_size": output["byte_size"],
            }
            for output in outputs
        ]
        if synthetic_fixture:
            source_name = "ClinicalTrials.gov synthetic fixture"
            mutable_registry: bool | None = False
        elif transport_mode == "injected":
            source_name = "ClinicalTrials.gov API-shaped injected transport"
            mutable_registry = None
        else:
            source_name = "ClinicalTrials.gov"
            mutable_registry = True
        manifest: dict[str, object] = {
            "bundle_type": "clinicaltrials_gov_api_snapshot",
            "bundle_schema_version": 1,
            "snapshot_id": snapshot_id,
            "complete": True,
            "source": {
                "name": source_name,
                "synthetic_fixture": synthetic_fixture,
                "transport_mode": transport_mode,
                "clock_mode": clock_mode,
                "elapsed_clock_mode": elapsed_clock_mode,
                "api_base_url": API_BASE_URL,
                "studies_endpoint": STUDIES_ENDPOINT,
                "version_endpoint": VERSION_ENDPOINT,
                "api_documentation_url": API_DOCUMENTATION_URL,
                "terms_url": TERMS_URL,
                "api_version": api_version,
                "data_timestamp": data_timestamp,
                "data_timestamp_timezone_interpretation": (
                    DATA_TIMESTAMP_INTERPRETATION
                ),
                "mutable_registry": mutable_registry,
            },
            "query": query_identity,
            "query_sha256": _sha256_bytes(_canonical_json_bytes(query_identity)),
            "retrieval": {
                "started_at_utc": started_at,
                "completed_at_utc": completed_at,
                "timeout_seconds": timeout_seconds,
                "max_attempts": max_attempts,
                "backoff_seconds": backoff_seconds,
                "max_pages": max_pages,
                "max_studies": max_studies,
                "max_candidate_rows": max_candidate_rows,
                "max_page_bytes": max_page_bytes,
                "max_total_bytes": max_total_bytes,
                "max_derived_bytes": max_derived_bytes,
                "max_elapsed_seconds": max_elapsed_seconds,
                "total_raw_response_bytes": total_raw_bytes,
                "total_derived_bytes": total_derived_bytes,
            },
            "version_envelope": {
                "stable_before_and_after_pagination": True,
                "version_start_retrieved_at_utc": version_start_retrieved,
                "version_end_retrieved_at_utc": version_end_retrieved,
                "snapshot_isolation_claim": SNAPSHOT_ISOLATION_CLAIM,
            },
            "pagination_completeness_basis": [
                "pagination_terminated_without_nextPageToken",
                "page_tokens_did_not_repeat",
                "NCT_identifiers_are_unique_across_pages",
                "observed_unique_study_count_equals_totalCount",
                "API_version_and_dataTimestamp_stable_before_and_after",
            ],
            "page_count": len(page_captures),
            "total_count": observed_total,
            "observed_unique_study_count": len(all_study_ids),
            "pages": [_page_manifest_entry(page) for page in page_captures],
            "outputs": outputs,
            "bundle_content_sha256": _sha256_bytes(
                _canonical_json_bytes(content_index)
            ),
            "software": {
                "package": "crispr-evidencerank",
                "package_version": __version__,
                "build_revision": build_revision,
            },
            "scientific_boundary": {
                "query_match_is_exact_concept_mapping": False,
                "curation_queue_review_status": "not_performed",
                "intervention_condition_rows_are": (
                    "study_level_co_mentions_not_arm_or_cohort_linkage"
                ),
                "treatment_cancer_linkage_review_status": "not_performed",
                "curation_queue_eligible_for_clinical_context": False,
                "used_for_gene_ranking": False,
                "used_for_validation_label": False,
                "registry_presence_is_efficacy": False,
                "results_posted_is_endpoint_met": False,
                "synthetic_fixture": synthetic_fixture,
            },
            "integrity_scope": INTEGRITY_SCOPE,
        }
        _validate_manifest_document(manifest)
        manifest_bytes = _canonical_json_bytes(manifest)
        if len(manifest_bytes) > HARD_MAX_MANIFEST_BYTES:
            raise ClinicalTrialsGovIntakeError(
                "snapshot manifest exceeds the verifier size limit"
            )
        (staging / "manifest.json").write_bytes(manifest_bytes)
        assert_elapsed_within_limit()
        verify_clinicaltrials_gov_snapshot(staging)
        assert_elapsed_within_limit()
        _fsync_tree(staging)
        assert_elapsed_within_limit()
        _publish_directory_noreplace(staging, output_dir)
        staging = None
        _fsync_directory(output_dir.parent)
        verify_clinicaltrials_gov_snapshot(output_dir)
        assert_elapsed_within_limit()
        return manifest
    except Exception:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        _release_publication_lock(lock_descriptor)
        _fsync_directory(output_dir.parent)


def request_url_query(url: str) -> dict[str, list[str]]:
    """Return decoded query parameters for tests and external audit tooling."""

    return parse_qs(urlsplit(url).query, keep_blank_values=True)
