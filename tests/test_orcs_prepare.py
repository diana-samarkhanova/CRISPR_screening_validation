from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from crispr_evidencerank.cli import build_parser
from crispr_evidencerank.orcs_prepare import prepare_orcs_release
from crispr_evidencerank.orcs_release import load_orcs_release_spec

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "config" / "orcs_releases.yaml"


def _one_screen_release() -> tuple[bytes, bytes]:
    spec = load_orcs_release_spec(SPEC_PATH)
    values = {header: "-" for header in spec.expected_index_headers}
    values.update(
        {
            "#SCREEN_ID": "1",
            "SOURCE_ID": "12345",
            "SOURCE_TYPE": "pubmed",
            "AUTHOR": "Example et al.",
            "SCREEN_NAME": "Example drug screen",
            "SCORES_SIZE": "1",
            "FULL_SIZE": "1",
            "FULL_SIZE_AVAILABLE": "Yes",
            "NUMBER_OF_HITS": "1",
            "THROUGHPUT": "High Throughput",
            "SCREEN_TYPE": "Positive",
            "SCREEN_FORMAT": "Pool",
            "EXPERIMENTAL_SETUP": "Drug Exposure",
            "CONDITION_NAME": "Drug A",
            "LIBRARY_TYPE": "CRISPRn",
            "LIBRARY_METHODOLOGY": "Knockout",
            "ENZYME": "Cas9",
            "CELL_LINE": "A375",
            "SCORE_COL_COUNT": "1",
            "SCORE.1_TYPE": "Score",
            "ORGANISM_ID": "9606",
            "ORGANISM_OFFICIAL": "H. sapiens",
        }
    )
    index = (
        "\t".join(spec.expected_index_headers)
        + "\n"
        + "\t".join(values[header] for header in spec.expected_index_headers)
        + "\n"
    ).encode()

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in (
            (spec.index_member, index),
            (
                "BIOGRID-ORCS-SCREEN_1-2.0.18.screen.tab.txt",
                b"#SCREEN_ID\tIDENTIFIER_ID\n1\t1\n",
            ),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue(), index


def _fixture_spec(archive: bytes, index: bytes):
    base = load_orcs_release_spec(SPEC_PATH)
    screen = b"#SCREEN_ID\tIDENTIFIER_ID\n1\t1\n"
    return replace(
        base,
        archive_sha256=hashlib.sha256(archive).hexdigest(),
        archive_byte_size=len(archive),
        expected_regular_member_count=2,
        expected_screen_member_count=1,
        expected_total_uncompressed_bytes=len(index) + len(screen),
        expected_max_member_bytes=max(len(index), len(screen)),
        index_sha256=hashlib.sha256(index).hexdigest(),
        index_byte_size=len(index),
        expected_index_data_rows=1,
    )


def test_prepare_release_is_atomic_portable_and_not_trainable(tmp_path):
    archive, index = _one_screen_release()
    spec = _fixture_spec(archive, index)
    archive_path = tmp_path / spec.archive_filename
    archive_path.write_bytes(archive)
    output_dir = tmp_path / "prepared"

    prepared = prepare_orcs_release(
        spec,
        output_dir,
        retrieved_date=date(2026, 7, 31),
        archive_path=archive_path,
    )

    assert prepared.output_dir == output_dir.resolve()
    assert prepared.screen_count == 1
    assert prepared.study_count == 1
    assert prepared.candidate_screen_count == 1
    assert not list(tmp_path.glob(".prepared.*"))
    assert {
        "archive_manifest.json",
        "index_manifest.json",
        "release_spec.json",
        "release_summary.json",
        "raw_index.tsv",
        "normalized_index.tsv",
        "studies.tsv",
        "screens.tsv",
        "screen_designs.tsv",
        "contrasts.tsv",
        "external_screen_maps.tsv",
        "screen_intake.tsv",
        "eligibility_checks.tsv",
        "curation_queue.tsv",
        "candidate_screen_ids.txt",
        "triage_summary.json",
        "header_map.json",
        spec.index_member,
    } == {path.name for path in output_dir.iterdir()}

    archive_manifest = json.loads(
        (output_dir / "archive_manifest.json").read_text(encoding="utf-8")
    )
    assert archive_manifest["archive_filename"] == spec.archive_filename
    assert archive_manifest["checksum_provenance"] == (
        "locally_computed_not_publisher_provided"
    )
    assert archive_manifest["upstream_raw_data_rights_established"] is False
    assert not Path(archive_manifest["archive_filename"]).is_absolute()

    summary = json.loads(
        (output_dir / "release_summary.json").read_text(encoding="utf-8")
    )
    assert summary["triage"]["benchmark_ready_count"] == 0
    assert summary["training_readiness"].startswith("not_trainable")
    queue = pd.read_csv(output_dir / "curation_queue.tsv", sep="\t")
    assert queue["external_screen_id"].astype(str).tolist() == ["1"]


def test_prepare_release_refuses_to_mix_existing_output(tmp_path):
    archive, index = _one_screen_release()
    spec = _fixture_spec(archive, index)
    archive_path = tmp_path / spec.archive_filename
    archive_path.write_bytes(archive)
    output_dir = tmp_path / "prepared"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        prepare_orcs_release(
            spec,
            output_dir,
            retrieved_date=date(2026, 7, 31),
            archive_path=archive_path,
        )

    assert marker.read_text(encoding="utf-8") == "keep"


def test_prepare_release_rejects_prepublication_retrieval_date(tmp_path):
    archive, index = _one_screen_release()
    spec = _fixture_spec(archive, index)

    with pytest.raises(ValueError, match="cannot precede"):
        prepare_orcs_release(
            spec,
            tmp_path / "prepared",
            retrieved_date=date(2025, 10, 6),
            archive_path=tmp_path / spec.archive_filename,
        )

    assert not (tmp_path / "prepared").exists()


def test_prepare_release_cli_parses_pinned_inputs(tmp_path):
    args = build_parser().parse_args(
        [
            "prepare-orcs-release",
            "--retrieved-date",
            "2026-07-31",
            "--archive",
            str(tmp_path / "archive.tar.gz"),
            "--output-dir",
            str(tmp_path / "prepared"),
        ]
    )

    assert args.release == "2.0.18"
    assert args.retrieved_date == date(2026, 7, 31)
    assert args.policy_version == 2
