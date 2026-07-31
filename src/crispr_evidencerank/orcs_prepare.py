"""End-to-end preparation of one pinned BioGRID ORCS release."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from .intake import OrcsIntakeResult, triage_orcs_index
from .orcs import OrcsIndexParseResult, parse_orcs_index
from .orcs_release import (
    ExtractedIndex,
    OrcsReleaseSpec,
    VerifiedArchive,
    download_orcs_archive,
    extract_orcs_index,
    verify_orcs_archive,
)


@dataclass(frozen=True)
class PreparedOrcsRelease:
    """Portable summary of an atomically prepared ORCS intake directory."""

    output_dir: Path
    release: str
    screen_count: int
    study_count: int
    candidate_screen_count: int
    summary: dict[str, object]


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False)


def _write_json(value: object, path: Path) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _spec_manifest(spec: OrcsReleaseSpec) -> dict[str, object]:
    manifest = asdict(spec)
    manifest["compiled_date"] = spec.compiled_date.isoformat()
    manifest["available_date"] = spec.available_date.isoformat()
    manifest["allowed_download_hosts"] = list(spec.allowed_download_hosts)
    manifest["expected_index_headers"] = list(spec.expected_index_headers)
    return manifest


def _archive_manifest(
    archive: VerifiedArchive,
    spec: OrcsReleaseSpec,
) -> dict[str, object]:
    return {
        "source_name": spec.source_name,
        "source_version": spec.release,
        "archive_filename": archive.path.name,
        "source_url": archive.source_url,
        "resolved_url": archive.resolved_url,
        "available_date": spec.available_date.isoformat(),
        "retrieved_date": archive.retrieved_date.isoformat(),
        "sha256": archive.sha256,
        "byte_size": archive.byte_size,
        "checksum_provenance": archive.checksum_provenance,
        "cache_hit": archive.cache_hit,
        "license_spdx": spec.license_spdx,
        "license_url": spec.license_url,
        "license_scope": "BioGRID ORCS-distributed files only",
        "upstream_raw_data_rights_established": False,
    }


def _index_manifest(
    index: ExtractedIndex,
    spec: OrcsReleaseSpec,
) -> dict[str, object]:
    return {
        "source_name": spec.source_name,
        "source_version": spec.release,
        "filename": index.path.name,
        "archive_sha256": index.archive_sha256,
        "sha256": index.sha256,
        "byte_size": index.byte_size,
        "data_rows": index.data_rows,
        "screen_ids_sha256": index.screen_ids_sha256,
        "inventory": asdict(index.inventory),
        "available_date": spec.available_date.isoformat(),
        "license_spdx": spec.license_spdx,
        "license_scope": "BioGRID ORCS-distributed files only",
    }


def _write_parsed_index(
    parsed: OrcsIndexParseResult,
    output_dir: Path,
) -> None:
    frames = {
        "raw_index": parsed.raw_index,
        "normalized_index": parsed.normalized_index,
        "studies": parsed.studies,
        "screens": parsed.screens,
        "screen_designs": parsed.screen_designs,
        "contrasts": parsed.contrasts,
        "external_screen_maps": parsed.external_screen_maps,
    }
    for name, frame in frames.items():
        _write_frame(frame, output_dir / f"{name}.tsv")
    _write_json(parsed.header_map, output_dir / "header_map.json")


def _write_triage(result: OrcsIntakeResult, output_dir: Path) -> None:
    _write_frame(result.screen_intake, output_dir / "screen_intake.tsv")
    _write_frame(result.eligibility_checks, output_dir / "eligibility_checks.tsv")
    _write_frame(result.curation_queue, output_dir / "curation_queue.tsv")
    (output_dir / "candidate_screen_ids.txt").write_text(
        "".join(f"{screen_id}\n" for screen_id in result.candidate_screen_ids),
        encoding="utf-8",
    )
    _write_json(result.summary, output_dir / "triage_summary.json")


def _validate_prepared_results(
    parsed: OrcsIndexParseResult,
    triage: OrcsIntakeResult,
    index: ExtractedIndex,
    spec: OrcsReleaseSpec,
) -> None:
    observed_counts = {
        "index_rows": len(parsed.raw_index),
        "parsed_screens": len(parsed.screens),
        "triaged_screens": int(triage.summary["total_screens"]),
        "index_data_rows": index.data_rows,
    }
    mismatched = {
        name: count
        for name, count in observed_counts.items()
        if count != spec.expected_index_data_rows
    }
    if mismatched:
        raise ValueError(
            "prepared ORCS release has inconsistent screen counts: "
            f"expected={spec.expected_index_data_rows}, observed={mismatched}"
        )
    if int(triage.summary["benchmark_ready_count"]) != 0:
        raise ValueError("index-stage ORCS intake cannot be benchmark-ready")
    if len(triage.curation_queue) != int(triage.summary["candidate_screen_count"]):
        raise ValueError("ORCS curation queue and candidate count differ")


def prepare_orcs_release(
    spec: OrcsReleaseSpec,
    output_dir: str | Path,
    *,
    retrieved_date: date,
    policy_version: int = 2,
    archive_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    timeout_seconds: float = 60.0,
) -> PreparedOrcsRelease:
    """Verify/acquire, extract, normalize, triage, and publish atomically.

    The destination must not already exist. This prevents an interrupted or
    accidental rerun from mixing files produced under different release
    contracts.
    """

    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"ORCS preparation output already exists: {target}")
    if target.name in {"", ".", ".."}:
        raise ValueError("output_dir must name a dedicated release directory")
    if archive_path is None and cache_dir is None:
        raise ValueError("cache_dir is required when archive_path is not provided")
    if retrieved_date < spec.available_date:
        raise ValueError("retrieved_date cannot precede the pinned available_date")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        if archive_path is not None:
            archive = verify_orcs_archive(
                archive_path,
                spec,
                retrieved_date=retrieved_date,
            )
        else:
            archive = download_orcs_archive(
                spec,
                cache_dir=cache_dir,
                retrieved_date=retrieved_date,
                timeout_seconds=timeout_seconds,
            )

        index = extract_orcs_index(archive, spec, staging)
        parsed = parse_orcs_index(
            index.path,
            release=spec.release,
            retrieved_date=retrieved_date,
            available_date=spec.available_date,
            organism_scope=spec.organism,
        )
        triage = triage_orcs_index(
            index.path,
            release=spec.release,
            retrieved_date=retrieved_date,
            available_date=spec.available_date,
            organism_scope=spec.organism,
            policy_version=policy_version,
        )
        _validate_prepared_results(parsed, triage, index, spec)

        _write_parsed_index(parsed, staging)
        _write_triage(triage, staging)
        _write_json(_spec_manifest(spec), staging / "release_spec.json")
        _write_json(_archive_manifest(archive, spec), staging / "archive_manifest.json")
        _write_json(_index_manifest(index, spec), staging / "index_manifest.json")
        release_summary = {
            "source_name": spec.source_name,
            "release": spec.release,
            "compiled_date": spec.compiled_date.isoformat(),
            "available_date": spec.available_date.isoformat(),
            "retrieved_date": retrieved_date.isoformat(),
            "screen_count": len(parsed.screens),
            "study_count": len(parsed.studies),
            "archive_sha256": archive.sha256,
            "index_sha256": index.sha256,
            "triage": triage.summary,
            "training_readiness": (
                "not_trainable_index_intake_requires_full_text_curation"
            ),
        }
        _write_json(release_summary, staging / "release_summary.json")

        os.replace(staging, target)
        return PreparedOrcsRelease(
            output_dir=target.resolve(),
            release=spec.release,
            screen_count=len(parsed.screens),
            study_count=len(parsed.studies),
            candidate_screen_count=len(triage.candidate_screen_ids),
            summary=release_summary,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
