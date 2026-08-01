"""Pinned acquisition and safe extraction for BioGRID ORCS releases."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import tarfile
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import yaml

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_SIZE = 1024 * 1024
_SPEC_FIELDS = frozenset(
    {
        "release",
        "source_name",
        "organism",
        "taxonomy_id",
        "archive_url",
        "archive_filename",
        "archive_sha256",
        "archive_byte_size",
        "checksum_provenance",
        "expected_regular_member_count",
        "expected_screen_member_count",
        "expected_total_uncompressed_bytes",
        "expected_max_member_bytes",
        "index_member",
        "index_sha256",
        "index_byte_size",
        "expected_index_data_rows",
        "expected_index_headers",
        "compiled_date",
        "available_date",
        "license_spdx",
        "license_url",
        "allowed_download_hosts",
    }
)


class OrcsAcquisitionError(ValueError):
    """Base exception for invalid release assets or acquisition metadata."""


class ReleaseSpecError(OrcsAcquisitionError):
    """Raised when a pinned release specification is incomplete or unsafe."""


class AssetIntegrityError(OrcsAcquisitionError):
    """Raised when downloaded or local bytes do not match the pinned asset."""


class UnsafeArchiveError(OrcsAcquisitionError):
    """Raised when a tar archive is unsafe or differs from its inventory."""


@dataclass(frozen=True)
class OrcsReleaseSpec:
    """Validated, immutable acquisition parameters for one ORCS release."""

    release: str
    source_name: str
    organism: str
    taxonomy_id: int
    archive_url: str
    archive_filename: str
    archive_sha256: str
    archive_byte_size: int
    checksum_provenance: str
    expected_regular_member_count: int
    expected_screen_member_count: int
    expected_total_uncompressed_bytes: int
    expected_max_member_bytes: int
    index_member: str
    index_sha256: str
    index_byte_size: int
    expected_index_data_rows: int
    expected_index_headers: tuple[str, ...]
    compiled_date: date
    available_date: date
    license_spdx: str
    license_url: str
    allowed_download_hosts: tuple[str, ...]


@dataclass(frozen=True)
class VerifiedArchive:
    """A local archive whose bytes match a pinned release specification."""

    path: Path
    source_url: str
    resolved_url: str
    sha256: str
    byte_size: int
    retrieved_date: date
    checksum_provenance: str
    cache_hit: bool

    def to_manifest(self) -> dict[str, object]:
        """Return JSON-serializable acquisition provenance."""

        record = asdict(self)
        record["path"] = str(self.path)
        record["retrieved_date"] = self.retrieved_date.isoformat()
        return record


@dataclass(frozen=True)
class OrcsArchiveInventory:
    """Observed regular-file inventory of an ORCS tar archive."""

    regular_member_count: int
    screen_member_count: int
    total_uncompressed_bytes: int
    max_member_bytes: int
    screen_ids_sha256: str


@dataclass(frozen=True)
class ExtractedIndex:
    """A safely extracted ORCS screen index and its lineage."""

    path: Path
    member_name: str
    sha256: str
    byte_size: int
    archive_sha256: str
    inventory: OrcsArchiveInventory
    data_rows: int
    screen_ids_sha256: str

    def to_manifest(self) -> dict[str, object]:
        """Return JSON-serializable extraction provenance."""

        return {
            "path": str(self.path),
            "member_name": self.member_name,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "archive_sha256": self.archive_sha256,
            "inventory": asdict(self.inventory),
            "data_rows": self.data_rows,
            "screen_ids_sha256": self.screen_ids_sha256,
        }


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReleaseSpecError(f"{field} must be a non-empty string")
    return value.strip()


def _required_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReleaseSpecError(f"{field} must be a positive integer")
    return value


def _parse_date(value: object, field: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(_required_text(value, field))
    except ValueError as exc:
        raise ReleaseSpecError(f"{field} must be an ISO date") from exc


def _validate_https_url(
    value: object,
    field: str,
    *,
    allowed_hosts: tuple[str, ...] | None = None,
) -> str:
    url = _required_text(value, field)
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ReleaseSpecError(
            f"{field} must be a credential-free HTTPS URL without query or fragment"
        )
    hostname = parsed.hostname.casefold()
    if allowed_hosts is not None and hostname not in allowed_hosts:
        raise ReleaseSpecError(f"{field} host is not allowlisted: {hostname}")
    return url


def _safe_member_name(name: str) -> None:
    if not name or "\x00" in name or "\\" in name:
        raise UnsafeArchiveError(f"unsafe tar member name: {name!r}")
    posix_path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise UnsafeArchiveError(f"unsafe tar member path: {name!r}")


def _validate_spec(release_key: str, raw: Mapping[str, object]) -> OrcsReleaseSpec:
    missing = sorted(_SPEC_FIELDS - raw.keys())
    extra = sorted(raw.keys() - _SPEC_FIELDS)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"unexpected={extra}")
        raise ReleaseSpecError(
            "invalid ORCS release specification fields: " + ", ".join(details)
        )

    release = _required_text(raw["release"], "release")
    if release != release_key:
        raise ReleaseSpecError(
            f"release key {release_key!r} does not match release value {release!r}"
        )
    archive_filename = _required_text(raw["archive_filename"], "archive_filename")
    if (
        "LATEST" in archive_filename.upper()
        or release not in archive_filename
        or PurePosixPath(archive_filename).name != archive_filename
    ):
        raise ReleaseSpecError(
            "archive_filename must be a release-qualified basename, never LATEST"
        )

    hosts_value = raw["allowed_download_hosts"]
    if not isinstance(hosts_value, list) or not hosts_value:
        raise ReleaseSpecError("allowed_download_hosts must be a non-empty list")
    hosts = tuple(
        _required_text(host, "allowed_download_hosts item").casefold()
        for host in hosts_value
    )
    if len(hosts) != len(set(hosts)):
        raise ReleaseSpecError("allowed_download_hosts contains duplicates")

    archive_url = _validate_https_url(
        raw["archive_url"],
        "archive_url",
        allowed_hosts=hosts,
    )
    parsed_archive_url = urlsplit(archive_url)
    expected_archive_segment = f"/Release-Archive/BIOGRID-ORCS-{release}/"
    if (
        not parsed_archive_url.path.endswith(f"/{archive_filename}")
        or expected_archive_segment not in parsed_archive_url.path
        or "LATEST" in parsed_archive_url.path.upper()
    ):
        raise ReleaseSpecError(
            "archive_url must point to the release-qualified Release-Archive file"
        )

    archive_sha256 = _required_text(raw["archive_sha256"], "archive_sha256")
    if _SHA256_RE.fullmatch(archive_sha256) is None:
        raise ReleaseSpecError("archive_sha256 must be 64 lowercase hexadecimal chars")
    index_sha256 = _required_text(raw["index_sha256"], "index_sha256")
    if _SHA256_RE.fullmatch(index_sha256) is None:
        raise ReleaseSpecError("index_sha256 must be 64 lowercase hexadecimal chars")

    index_member = _required_text(raw["index_member"], "index_member")
    try:
        _safe_member_name(index_member)
    except UnsafeArchiveError as exc:
        raise ReleaseSpecError(f"invalid index_member: {index_member!r}") from exc
    if PurePosixPath(index_member).name != index_member or release not in index_member:
        raise ReleaseSpecError(
            "index_member must be a release-qualified archive-root basename"
        )

    headers_value = raw["expected_index_headers"]
    if (
        not isinstance(headers_value, list)
        or not headers_value
        or not all(isinstance(header, str) and header for header in headers_value)
    ):
        raise ReleaseSpecError(
            "expected_index_headers must be a non-empty list of strings"
        )
    expected_index_headers = tuple(headers_value)
    if len(expected_index_headers) != len(set(expected_index_headers)):
        raise ReleaseSpecError("expected_index_headers contains duplicates")
    if expected_index_headers[0] != "#SCREEN_ID":
        raise ReleaseSpecError("expected_index_headers must begin with #SCREEN_ID")

    compiled_date = _parse_date(raw["compiled_date"], "compiled_date")
    available_date = _parse_date(raw["available_date"], "available_date")
    if compiled_date > available_date:
        raise ReleaseSpecError("compiled_date cannot follow available_date")

    checksum_provenance = _required_text(
        raw["checksum_provenance"], "checksum_provenance"
    )
    if checksum_provenance != "locally_computed_not_publisher_provided":
        raise ReleaseSpecError(
            "checksum_provenance must explicitly state that the checksum was "
            "locally computed and not publisher-provided"
        )

    regular_count = _required_positive_int(
        raw["expected_regular_member_count"],
        "expected_regular_member_count",
    )
    screen_count = _required_positive_int(
        raw["expected_screen_member_count"],
        "expected_screen_member_count",
    )
    total_bytes = _required_positive_int(
        raw["expected_total_uncompressed_bytes"],
        "expected_total_uncompressed_bytes",
    )
    max_bytes = _required_positive_int(
        raw["expected_max_member_bytes"],
        "expected_max_member_bytes",
    )
    if regular_count != screen_count + 1:
        raise ReleaseSpecError(
            "expected screen members must leave exactly one regular index member"
        )
    if max_bytes > total_bytes:
        raise ReleaseSpecError(
            "expected_max_member_bytes cannot exceed total uncompressed bytes"
        )

    license_url = _validate_https_url(raw["license_url"], "license_url")
    return OrcsReleaseSpec(
        release=release,
        source_name=_required_text(raw["source_name"], "source_name"),
        organism=_required_text(raw["organism"], "organism"),
        taxonomy_id=_required_positive_int(raw["taxonomy_id"], "taxonomy_id"),
        archive_url=archive_url,
        archive_filename=archive_filename,
        archive_sha256=archive_sha256,
        archive_byte_size=_required_positive_int(
            raw["archive_byte_size"], "archive_byte_size"
        ),
        checksum_provenance=checksum_provenance,
        expected_regular_member_count=regular_count,
        expected_screen_member_count=screen_count,
        expected_total_uncompressed_bytes=total_bytes,
        expected_max_member_bytes=max_bytes,
        index_member=index_member,
        index_sha256=index_sha256,
        index_byte_size=_required_positive_int(
            raw["index_byte_size"], "index_byte_size"
        ),
        expected_index_data_rows=_required_positive_int(
            raw["expected_index_data_rows"], "expected_index_data_rows"
        ),
        expected_index_headers=expected_index_headers,
        compiled_date=compiled_date,
        available_date=available_date,
        license_spdx=_required_text(raw["license_spdx"], "license_spdx"),
        license_url=license_url,
        allowed_download_hosts=hosts,
    )


def load_orcs_release_spec(
    path: str | Path,
    release: str = "2.0.18",
) -> OrcsReleaseSpec:
    """Load and strictly validate one release from the pinned YAML registry."""

    with Path(path).open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "releases",
    }:
        raise ReleaseSpecError(
            "ORCS release registry must contain only schema_version and releases"
        )
    if document["schema_version"] != 1:
        raise ReleaseSpecError("unsupported ORCS release registry schema_version")
    releases = document["releases"]
    if not isinstance(releases, dict) or release not in releases:
        raise ReleaseSpecError(f"ORCS release is not pinned: {release}")
    raw = releases[release]
    if not isinstance(raw, dict):
        raise ReleaseSpecError(f"ORCS release entry must be a mapping: {release}")
    return _validate_spec(release, raw)


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
            byte_size += len(chunk)
    return digest.hexdigest(), byte_size


def verify_orcs_archive(
    path: str | Path,
    spec: OrcsReleaseSpec,
    *,
    retrieved_date: date,
    source_url: str | None = None,
    resolved_url: str | None = None,
    cache_hit: bool = True,
) -> VerifiedArchive:
    """Verify local archive bytes against the pinned size and SHA-256."""

    archive_path = Path(path)
    if not archive_path.is_file():
        raise AssetIntegrityError(f"ORCS archive is not a regular file: {archive_path}")
    sha256, byte_size = _sha256_and_size(archive_path)
    if byte_size != spec.archive_byte_size:
        raise AssetIntegrityError(
            f"ORCS archive size mismatch: expected {spec.archive_byte_size}, "
            f"observed {byte_size}"
        )
    if sha256 != spec.archive_sha256:
        raise AssetIntegrityError(
            f"ORCS archive SHA-256 mismatch: expected {spec.archive_sha256}, "
            f"observed {sha256}"
        )
    requested = source_url or spec.archive_url
    resolved = resolved_url or requested
    _validate_https_url(
        requested,
        "source_url",
        allowed_hosts=spec.allowed_download_hosts,
    )
    _validate_https_url(
        resolved,
        "resolved_url",
        allowed_hosts=spec.allowed_download_hosts,
    )
    return VerifiedArchive(
        path=archive_path.resolve(),
        source_url=requested,
        resolved_url=resolved,
        sha256=sha256,
        byte_size=byte_size,
        retrieved_date=retrieved_date,
        checksum_provenance=spec.checksum_provenance,
        cache_hit=cache_hit,
    )


def _response_url(response: BinaryIO, fallback: str) -> str:
    getter = getattr(response, "geturl", None)
    return str(getter()) if callable(getter) else fallback


def download_orcs_archive(
    spec: OrcsReleaseSpec,
    cache_dir: str | Path,
    *,
    retrieved_date: date,
    opener: Callable[..., BinaryIO] = urlopen,
    timeout_seconds: float = 60.0,
) -> VerifiedArchive:
    """Download a pinned release atomically, or verify and reuse its cache."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    target_dir = Path(cache_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / spec.archive_filename
    if target.exists():
        return verify_orcs_archive(
            target,
            spec,
            retrieved_date=retrieved_date,
            cache_hit=True,
        )

    request = Request(
        spec.archive_url,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "crispr-evidencerank-orcs-acquisition/0.2",
        },
    )
    temporary_path: Path | None = None
    resolved_url = spec.archive_url
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{spec.archive_filename}.",
            suffix=".part",
            dir=target_dir,
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            digest = hashlib.sha256()
            byte_size = 0
            with opener(request, timeout=timeout_seconds) as response:
                resolved_url = _response_url(response, spec.archive_url)
                _validate_https_url(
                    resolved_url,
                    "resolved_url",
                    allowed_hosts=spec.allowed_download_hosts,
                )
                while chunk := response.read(_CHUNK_SIZE):
                    byte_size += len(chunk)
                    if byte_size > spec.archive_byte_size:
                        raise AssetIntegrityError(
                            "download exceeded the pinned ORCS archive byte size"
                        )
                    digest.update(chunk)
                    destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())

        observed_sha256 = digest.hexdigest()
        if byte_size != spec.archive_byte_size:
            raise AssetIntegrityError(
                f"download size mismatch: expected {spec.archive_byte_size}, "
                f"observed {byte_size}"
            )
        if observed_sha256 != spec.archive_sha256:
            raise AssetIntegrityError(
                f"download SHA-256 mismatch: expected {spec.archive_sha256}, "
                f"observed {observed_sha256}"
            )
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return VerifiedArchive(
        path=target.resolve(),
        source_url=spec.archive_url,
        resolved_url=resolved_url,
        sha256=spec.archive_sha256,
        byte_size=spec.archive_byte_size,
        retrieved_date=retrieved_date,
        checksum_provenance=spec.checksum_provenance,
        cache_hit=False,
    )


def _validate_inventory(
    inventory: OrcsArchiveInventory,
    spec: OrcsReleaseSpec,
) -> None:
    expected = OrcsArchiveInventory(
        regular_member_count=spec.expected_regular_member_count,
        screen_member_count=spec.expected_screen_member_count,
        total_uncompressed_bytes=spec.expected_total_uncompressed_bytes,
        max_member_bytes=spec.expected_max_member_bytes,
        screen_ids_sha256=inventory.screen_ids_sha256,
    )
    if inventory != expected:
        raise UnsafeArchiveError(
            f"ORCS archive inventory mismatch: expected {expected}, "
            f"observed {inventory}"
        )


def _screen_ids_sha256(screen_ids: set[int]) -> str:
    payload = "".join(f"{screen_id}\n" for screen_id in sorted(screen_ids))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _validate_extracted_index(
    path: Path,
    spec: OrcsReleaseSpec,
    archive_screen_ids: set[int],
) -> tuple[int, str]:
    observed_sha256, observed_size = _sha256_and_size(path)
    if observed_size != spec.index_byte_size:
        raise AssetIntegrityError(
            f"ORCS index size mismatch: expected {spec.index_byte_size}, "
            f"observed {observed_size}"
        )
    if observed_sha256 != spec.index_sha256:
        raise AssetIntegrityError(
            f"ORCS index SHA-256 mismatch: expected {spec.index_sha256}, "
            f"observed {observed_sha256}"
        )

    index_screen_ids: set[int] = set()
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            headers = next(reader)
            if tuple(headers) != spec.expected_index_headers:
                raise AssetIntegrityError(
                    "ORCS index header differs from the pinned release contract"
                )
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(headers):
                    raise AssetIntegrityError(
                        f"ORCS index row {row_number} has {len(row)} columns; "
                        f"expected {len(headers)}"
                    )
                try:
                    screen_id = int(row[0])
                except ValueError as exc:
                    raise AssetIntegrityError(
                        f"ORCS index row {row_number} has an invalid SCREEN_ID"
                    ) from exc
                if screen_id in index_screen_ids:
                    raise AssetIntegrityError(
                        f"ORCS index contains duplicate SCREEN_ID {screen_id}"
                    )
                index_screen_ids.add(screen_id)
    except (OSError, UnicodeError, StopIteration) as exc:
        raise AssetIntegrityError(f"cannot parse ORCS index: {exc}") from exc

    if len(index_screen_ids) != spec.expected_index_data_rows:
        raise AssetIntegrityError(
            f"ORCS index row count mismatch: expected "
            f"{spec.expected_index_data_rows}, observed {len(index_screen_ids)}"
        )
    if index_screen_ids != archive_screen_ids:
        missing = sorted(archive_screen_ids - index_screen_ids)[:5]
        extra = sorted(index_screen_ids - archive_screen_ids)[:5]
        raise AssetIntegrityError(
            "ORCS index SCREEN_ID values do not match archive screen members: "
            f"missing_from_index={missing}, extra_in_index={extra}"
        )
    return len(index_screen_ids), _screen_ids_sha256(index_screen_ids)


def extract_orcs_index(
    archive: VerifiedArchive,
    spec: OrcsReleaseSpec,
    output_dir: str | Path,
) -> ExtractedIndex:
    """Safely extract only the exact pinned index member in one tar pass."""

    if (
        archive.sha256 != spec.archive_sha256
        or archive.byte_size != spec.archive_byte_size
    ):
        raise AssetIntegrityError("VerifiedArchive does not match the release spec")
    if not archive.path.is_file() or archive.path.stat().st_size != archive.byte_size:
        raise AssetIntegrityError("verified ORCS archive changed before extraction")
    observed_sha256, observed_size = _sha256_and_size(archive.path)
    if (
        observed_sha256 != spec.archive_sha256
        or observed_size != spec.archive_byte_size
    ):
        raise AssetIntegrityError("verified ORCS archive changed before extraction")

    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / spec.index_member
    temporary_path: Path | None = None
    index_sha256: str | None = None
    index_byte_size: int | None = None
    regular_count = 0
    screen_count = 0
    total_bytes = 0
    max_bytes = 0
    seen_names: set[str] = set()
    screen_ids: set[int] = set()
    index_count = 0
    screen_member_pattern = re.compile(
        rf"^BIOGRID-ORCS-SCREEN_(?P<screen_id>[0-9]+)-"
        rf"{re.escape(spec.release)}\.screen\.tab\.txt$"
    )

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{spec.index_member}.",
            suffix=".part",
            dir=destination_dir,
            delete=False,
        ) as index_destination:
            temporary_path = Path(index_destination.name)
            try:
                with tarfile.open(archive.path, mode="r|gz") as tar:
                    for member in tar:
                        _safe_member_name(member.name)
                        if member.name in seen_names:
                            raise UnsafeArchiveError(
                                f"duplicate tar member name: {member.name!r}"
                            )
                        seen_names.add(member.name)
                        if member.issym() or member.islnk():
                            raise UnsafeArchiveError(
                                f"tar links are prohibited: {member.name!r}"
                            )
                        if not member.isfile():
                            raise UnsafeArchiveError(
                                f"non-regular tar member is prohibited: {member.name!r}"
                            )

                        regular_count += 1
                        total_bytes += member.size
                        max_bytes = max(max_bytes, member.size)
                        screen_match = screen_member_pattern.fullmatch(member.name)
                        if screen_match is not None:
                            screen_count += 1
                            screen_id = int(screen_match.group("screen_id"))
                            if screen_id in screen_ids:
                                raise UnsafeArchiveError(
                                    f"duplicate ORCS screen member ID: {screen_id}"
                                )
                            screen_ids.add(screen_id)
                        elif member.name != spec.index_member:
                            raise UnsafeArchiveError(
                                f"unexpected regular tar member: {member.name!r}"
                            )
                        if (
                            regular_count > spec.expected_regular_member_count
                            or screen_count > spec.expected_screen_member_count
                            or total_bytes > spec.expected_total_uncompressed_bytes
                            or member.size > spec.expected_max_member_bytes
                        ):
                            raise UnsafeArchiveError(
                                "ORCS archive exceeds its pinned inventory bounds"
                            )

                        if member.name != spec.index_member:
                            continue
                        index_count += 1
                        if index_count > 1:
                            raise UnsafeArchiveError(
                                "ORCS archive contains duplicate index members"
                            )
                        source = tar.extractfile(member)
                        if source is None:
                            raise UnsafeArchiveError(
                                "ORCS index member cannot be read as a regular file"
                            )
                        digest = hashlib.sha256()
                        extracted_bytes = 0
                        while chunk := source.read(_CHUNK_SIZE):
                            extracted_bytes += len(chunk)
                            if extracted_bytes > member.size:
                                raise UnsafeArchiveError(
                                    "ORCS index extraction exceeded declared size"
                                )
                            digest.update(chunk)
                            index_destination.write(chunk)
                        if extracted_bytes != member.size:
                            raise UnsafeArchiveError(
                                "ORCS index extraction was shorter than declared size"
                            )
                        index_sha256 = digest.hexdigest()
                        index_byte_size = extracted_bytes
            except (tarfile.TarError, OSError) as exc:
                raise UnsafeArchiveError(
                    f"cannot read ORCS tar archive: {exc}"
                ) from exc

            inventory = OrcsArchiveInventory(
                regular_member_count=regular_count,
                screen_member_count=screen_count,
                total_uncompressed_bytes=total_bytes,
                max_member_bytes=max_bytes,
                screen_ids_sha256=_screen_ids_sha256(screen_ids),
            )
            _validate_inventory(inventory, spec)
            if index_count != 1 or index_sha256 is None or index_byte_size is None:
                raise UnsafeArchiveError(
                    f"ORCS archive is missing the pinned index: {spec.index_member}"
                )
            index_destination.flush()
            os.fsync(index_destination.fileno())

        data_rows, screen_ids_sha256 = _validate_extracted_index(
            temporary_path,
            spec,
            screen_ids,
        )
        os.replace(temporary_path, destination_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return ExtractedIndex(
        path=destination_path.resolve(),
        member_name=spec.index_member,
        sha256=index_sha256,
        byte_size=index_byte_size,
        archive_sha256=archive.sha256,
        inventory=inventory,
        data_rows=data_rows,
        screen_ids_sha256=screen_ids_sha256,
    )
