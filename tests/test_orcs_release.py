from __future__ import annotations

import hashlib
import io
import tarfile
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
import yaml

from crispr_evidencerank.orcs_release import (
    AssetIntegrityError,
    ReleaseSpecError,
    UnsafeArchiveError,
    VerifiedArchive,
    download_orcs_archive,
    extract_orcs_index,
    load_orcs_release_spec,
    verify_orcs_archive,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "config" / "orcs_releases.yaml"


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, url: str):
        super().__init__(data)
        self._url = url

    def geturl(self) -> str:
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _archive_bytes(
    members: list[tuple[str, bytes, bytes | None]],
) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content, member_type in members:
            info = tarfile.TarInfo(name)
            info.mtime = 0
            if member_type is None:
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
            else:
                info.type = member_type
                info.size = 0
                if member_type in {tarfile.SYMTYPE, tarfile.LNKTYPE}:
                    info.linkname = "target"
                tar.addfile(info)
    return buffer.getvalue()


def _fixture_spec(
    archive_data: bytes,
    members: list[tuple[str, bytes, bytes | None]],
):
    base = load_orcs_release_spec(SPEC_PATH)
    regular = [
        (name, content) for name, content, member_type in members if member_type is None
    ]
    index = next(content for name, content in regular if name == base.index_member)
    index_lines = index.decode("utf-8").splitlines()
    return replace(
        base,
        archive_sha256=hashlib.sha256(archive_data).hexdigest(),
        archive_byte_size=len(archive_data),
        expected_regular_member_count=len(regular),
        expected_screen_member_count=sum(
            name.endswith(".screen.tab.txt") for name, _content in regular
        ),
        expected_total_uncompressed_bytes=sum(
            len(content) for _name, content in regular
        ),
        expected_max_member_bytes=max(len(content) for _name, content in regular),
        index_sha256=hashlib.sha256(index).hexdigest(),
        index_byte_size=len(index),
        expected_index_data_rows=max(len(index_lines) - 1, 0),
        expected_index_headers=tuple(index_lines[0].split("\t")),
    )


def _verified(path: Path, spec) -> VerifiedArchive:
    return verify_orcs_archive(
        path,
        spec,
        retrieved_date=date(2026, 7, 31),
    )


def test_real_release_spec_is_exactly_pinned():
    spec = load_orcs_release_spec(SPEC_PATH)

    assert spec.release == "2.0.18"
    assert spec.archive_sha256 == (
        "39222a9650eed083edf193debe45eedc4aabc779ca04ea70107b6bd1efd9b8d7"
    )
    assert spec.archive_byte_size == 752653348
    assert spec.expected_regular_member_count == 1953
    assert spec.expected_screen_member_count == 1952
    assert spec.expected_total_uncompressed_bytes == 2883261168
    assert spec.expected_max_member_bytes == 3934397
    assert spec.index_member == ("BIOGRID-ORCS-SCREEN_INDEX-2.0.18.index.tab.txt")
    assert spec.index_sha256 == (
        "6754e87ef758a3525ab7d690b183338f2ee6de72a36dde6e85fe19dee165f02d"
    )
    assert spec.index_byte_size == 1328911
    assert spec.expected_index_data_rows == 1952
    assert len(spec.expected_index_headers) == 38
    assert spec.compiled_date == date(2025, 9, 9)
    assert spec.available_date == date(2025, 10, 7)
    assert spec.checksum_provenance == ("locally_computed_not_publisher_provided")
    assert "LATEST" not in spec.archive_url


def test_release_spec_rejects_latest_and_unpinned_checksum(tmp_path):
    document = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    release = document["releases"]["2.0.18"]
    release["archive_filename"] = release["archive_filename"].replace(
        "2.0.18", "LATEST"
    )
    release["archive_url"] = release["archive_url"].replace(
        "BIOGRID-ORCS-ALL-homo_sapiens-2.0.18.screens.tar.gz",
        "BIOGRID-ORCS-ALL-homo_sapiens-LATEST.screens.tar.gz",
    )
    release["archive_sha256"] = "unknown"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ReleaseSpecError, match="never LATEST"):
        load_orcs_release_spec(path)


def test_release_spec_rejects_unsafe_index_member(tmp_path):
    document = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    document["releases"]["2.0.18"]["index_member"] = "../index.tab.txt"
    path = tmp_path / "bad-index.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ReleaseSpecError, match="invalid index_member"):
        load_orcs_release_spec(path)


def test_verify_archive_rejects_size_and_sha_mismatch(tmp_path):
    path = tmp_path / "archive.tar.gz"
    path.write_bytes(b"correct")
    base = load_orcs_release_spec(SPEC_PATH)
    size_mismatch = replace(
        base,
        archive_sha256=hashlib.sha256(b"correct").hexdigest(),
        archive_byte_size=999,
    )
    with pytest.raises(AssetIntegrityError, match="size mismatch"):
        _verified(path, size_mismatch)

    sha_mismatch = replace(
        base,
        archive_sha256="0" * 64,
        archive_byte_size=len(b"correct"),
    )
    with pytest.raises(AssetIntegrityError, match="SHA-256 mismatch"):
        _verified(path, sha_mismatch)


def test_download_is_atomic_and_reuses_verified_cache(tmp_path):
    data = b"release bytes"
    base = load_orcs_release_spec(SPEC_PATH)
    spec = replace(
        base,
        archive_sha256=hashlib.sha256(data).hexdigest(),
        archive_byte_size=len(data),
    )
    calls = 0

    def opener(_request, *, timeout):
        nonlocal calls
        assert timeout == 12
        calls += 1
        return FakeResponse(
            data,
            "https://biogrid-downloads.nyc3.digitaloceanspaces.com/release.tar.gz",
        )

    first = download_orcs_archive(
        spec,
        tmp_path,
        retrieved_date=date(2026, 7, 31),
        opener=opener,
        timeout_seconds=12,
    )
    second = download_orcs_archive(
        spec,
        tmp_path,
        retrieved_date=date(2026, 8, 1),
        opener=lambda *_args, **_kwargs: pytest.fail("cache should avoid download"),
    )

    assert calls == 1
    assert first.path.read_bytes() == data
    assert not first.cache_hit
    assert second.cache_hit
    assert second.sha256 == spec.archive_sha256
    assert not list(tmp_path.glob("*.part"))


def test_failed_download_leaves_no_archive_or_part_file(tmp_path):
    data = b"wrong bytes"
    base = load_orcs_release_spec(SPEC_PATH)
    spec = replace(
        base,
        archive_sha256="0" * 64,
        archive_byte_size=len(data),
    )

    with pytest.raises(AssetIntegrityError, match="SHA-256 mismatch"):
        download_orcs_archive(
            spec,
            tmp_path,
            retrieved_date=date(2026, 7, 31),
            opener=lambda *_args, **_kwargs: FakeResponse(
                data,
                spec.archive_url,
            ),
        )

    assert not (tmp_path / spec.archive_filename).exists()
    assert not list(tmp_path.glob("*.part"))


def test_download_rejects_redirect_to_unapproved_host(tmp_path):
    data = b"x"
    base = load_orcs_release_spec(SPEC_PATH)
    spec = replace(
        base,
        archive_sha256=hashlib.sha256(data).hexdigest(),
        archive_byte_size=len(data),
    )

    with pytest.raises(ReleaseSpecError, match="not allowlisted"):
        download_orcs_archive(
            spec,
            tmp_path,
            retrieved_date=date(2026, 7, 31),
            opener=lambda *_args, **_kwargs: FakeResponse(
                data,
                "https://example.org/archive.tar.gz",
            ),
        )

    assert not list(tmp_path.iterdir())


def test_extracts_only_exact_index_and_records_inventory(tmp_path):
    index = b"#SCREEN_ID\tSOURCE_ID\n1\t123\n2\t456\n"
    members = [
        ("BIOGRID-ORCS-SCREEN_INDEX-2.0.18.index.tab.txt", index, None),
        ("BIOGRID-ORCS-SCREEN_1-2.0.18.screen.tab.txt", b"one", None),
        ("BIOGRID-ORCS-SCREEN_2-2.0.18.screen.tab.txt", b"two-two", None),
    ]
    data = _archive_bytes(members)
    spec = _fixture_spec(data, members)
    archive_path = tmp_path / spec.archive_filename
    archive_path.write_bytes(data)

    result = extract_orcs_index(
        _verified(archive_path, spec),
        spec,
        tmp_path / "out",
    )

    assert result.path.read_bytes() == index
    assert result.sha256 == hashlib.sha256(index).hexdigest()
    assert result.inventory.regular_member_count == 3
    assert result.inventory.screen_member_count == 2
    assert result.inventory.total_uncompressed_bytes == (
        len(index) + len(b"one") + len(b"two-two")
    )
    assert [path.name for path in (tmp_path / "out").iterdir()] == [spec.index_member]


def test_extraction_rechecks_archive_bytes_after_verification(tmp_path):
    members = [
        (
            "BIOGRID-ORCS-SCREEN_INDEX-2.0.18.index.tab.txt",
            b"index",
            None,
        ),
        ("BIOGRID-ORCS-SCREEN_1-2.0.18.screen.tab.txt", b"screen", None),
    ]
    data = _archive_bytes(members)
    spec = _fixture_spec(data, members)
    archive_path = tmp_path / spec.archive_filename
    archive_path.write_bytes(data)
    verified = _verified(archive_path, spec)
    changed = bytearray(data)
    changed[-1] ^= 1
    archive_path.write_bytes(changed)

    with pytest.raises(AssetIntegrityError, match="changed before extraction"):
        extract_orcs_index(verified, spec, tmp_path / "out")


@pytest.mark.parametrize(
    ("bad_name", "member_type"),
    [
        ("../escape.txt", None),
        ("/absolute.txt", None),
        (r"C:\escape.txt", None),
        ("link", tarfile.SYMTYPE),
        ("hardlink", tarfile.LNKTYPE),
        ("device", tarfile.CHRTYPE),
    ],
)
def test_extraction_rejects_unsafe_names_and_special_members(
    tmp_path,
    bad_name,
    member_type,
):
    members = [
        (
            "BIOGRID-ORCS-SCREEN_INDEX-2.0.18.index.tab.txt",
            b"index",
            None,
        ),
        ("BIOGRID-ORCS-SCREEN_1-2.0.18.screen.tab.txt", b"screen", None),
        (bad_name, b"bad", member_type),
    ]
    data = _archive_bytes(members)
    regular = [(name, content) for name, content, kind in members if kind is None]
    base = load_orcs_release_spec(SPEC_PATH)
    spec = replace(
        base,
        archive_sha256=hashlib.sha256(data).hexdigest(),
        archive_byte_size=len(data),
        expected_regular_member_count=len(regular),
        expected_screen_member_count=1,
        expected_total_uncompressed_bytes=sum(len(content) for _, content in regular),
        expected_max_member_bytes=max(len(content) for _, content in regular),
    )
    archive_path = tmp_path / spec.archive_filename
    archive_path.write_bytes(data)

    with pytest.raises(UnsafeArchiveError):
        extract_orcs_index(
            _verified(archive_path, spec),
            spec,
            tmp_path / "out",
        )

    assert not (tmp_path / "out" / spec.index_member).exists()
    assert not list((tmp_path / "out").glob("*.part"))


def test_extraction_rejects_duplicate_index_and_inventory_mismatch(tmp_path):
    index_name = "BIOGRID-ORCS-SCREEN_INDEX-2.0.18.index.tab.txt"
    duplicate_members = [
        (index_name, b"first", None),
        (index_name, b"second", None),
        ("BIOGRID-ORCS-SCREEN_1-2.0.18.screen.tab.txt", b"screen", None),
    ]
    duplicate_data = _archive_bytes(duplicate_members)
    duplicate_spec = _fixture_spec(duplicate_data, duplicate_members)
    duplicate_path = tmp_path / duplicate_spec.archive_filename
    duplicate_path.write_bytes(duplicate_data)
    with pytest.raises(UnsafeArchiveError, match="duplicate"):
        extract_orcs_index(
            _verified(duplicate_path, duplicate_spec),
            duplicate_spec,
            tmp_path / "duplicate-out",
        )

    valid_members = [
        (index_name, b"index", None),
        ("BIOGRID-ORCS-SCREEN_1-2.0.18.screen.tab.txt", b"screen", None),
    ]
    valid_data = _archive_bytes(valid_members)
    spec = replace(
        _fixture_spec(valid_data, valid_members),
        expected_total_uncompressed_bytes=999,
    )
    path = tmp_path / "inventory.tar.gz"
    path.write_bytes(valid_data)
    with pytest.raises(UnsafeArchiveError, match="inventory mismatch"):
        extract_orcs_index(
            _verified(path, spec),
            spec,
            tmp_path / "inventory-out",
        )
