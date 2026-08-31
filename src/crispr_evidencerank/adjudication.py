"""Checksum-bound, human-only release of validation-event labels."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, get_args

import pandas as pd

from .contracts import (
    AdjudicationDecisionDisposition,
    AdjudicationDecisionRecord,
    AdjudicationPacketRecord,
    FullTextReviewRecord,
    LabelCode,
    ReviewComparisonRecord,
    ValidationEventRecord,
    validate_records,
)
from .curation import build_dual_review_manifest
from .labels import _resolve_released_validation_events

ADJUDICATION_METHOD_VERSION = "validation_adjudication_v1"
PACKET_MANIFEST_SCHEMA = "crispr-evidencerank.adjudication-packet-manifest"
PACKET_MANIFEST_SCHEMA_VERSION = 1
RELEASE_MANIFEST_SCHEMA = "crispr-evidencerank.adjudication-release-manifest"
RELEASE_MANIFEST_SCHEMA_VERSION = 1

_SOURCE_COMPONENTS = {
    "selection": "selection.tsv",
    "primary_reviews": "reviews.tsv",
    "secondary_reviews": "reviews_curator_2.tsv",
    "comparison": "review_comparison.tsv",
}


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _expected_sha256(value: str, *, field: str) -> str:
    normalized = value.strip()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return normalized


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _canonical_member(base: Path, filename: object, *, expected: str) -> Path:
    if not isinstance(filename, str) or filename != expected:
        raise ValueError(f"completed review manifest requires {expected!r}")
    if Path(filename).name != filename or "\\" in filename:
        raise ValueError("manifest filenames must be canonical basenames")
    member = base / filename
    if member.parent.resolve() != base.resolve():
        raise ValueError("manifest member escapes its bundle directory")
    return member


def _json_object(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _table_from_bytes(
    content: bytes,
    *,
    filename: str,
    contract: type | None = None,
) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".tsv", ".txt", ".csv"}:
        raise ValueError(f"unsupported table extension: {suffix}")
    dtype: dict[str, str] | None = None
    if contract is not None:
        dtype = {
            name: "string"
            for name, field in contract.model_fields.items()
            if field.annotation is str or str in get_args(field.annotation)
        }
    kwargs: dict[str, Any] = {"dtype": dtype}
    if suffix in {".tsv", ".txt"}:
        kwargs["sep"] = "\t"
    return pd.read_csv(BytesIO(content), **kwargs)


def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (None if value is None or bool(pd.isna(value)) else value)
        for key, value in row.items()
    }


def _canonical_model_sha(record: Any) -> str:
    payload = json.dumps(
        record.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_completed_review_snapshot(
    manifest_path: Path,
    manifest_content: bytes,
    *,
    expected_manifest_sha256: str,
    expected_comparison_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, Path],
    dict[str, bytes],
    dict[str, pd.DataFrame],
]:
    observed_manifest_sha = _sha256_bytes(manifest_content)
    if observed_manifest_sha != expected_manifest_sha256:
        raise ValueError("completed review manifest SHA-256 does not match expected")
    manifest = _json_object(manifest_content, label="completed review manifest")
    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise ValueError("completed review manifest schema_version must be 1 or 2")
    if manifest.get("status") != ("dual_review_complete_requires_human_adjudication"):
        raise ValueError("completed review manifest is not ready for adjudication")
    if manifest.get("benchmark_ready_count") != 0:
        raise ValueError("completed review manifest must release no readiness")
    if manifest.get("pending_second_review_screen_count") != 0:
        raise ValueError("all selected screens require a completed second review")
    if manifest.get("selected_screen_count") != manifest.get(
        "second_reviewed_screen_count"
    ):
        raise ValueError("dual review does not cover the frozen screen selection")
    if schema_version == 2:
        required_state = {
            "human_adjudication_required": True,
            "adjudication_status": "pending_human_adjudication",
            "adjudicated_gene_count": 0,
            "released_label_count": 0,
        }
        for key, expected in required_state.items():
            if manifest.get(key) != expected:
                raise ValueError(
                    f"completed review manifest has invalid {key}: "
                    f"{manifest.get(key)!r}"
                )

    base = manifest_path.parent
    source_paths: dict[str, Path] = {}
    source_content: dict[str, bytes] = {}
    for component, expected_filename in _SOURCE_COMPONENTS.items():
        entry = manifest.get(component)
        if not isinstance(entry, dict):
            raise ValueError(f"completed review manifest omits {component}")
        path = _canonical_member(
            base, entry.get("filename"), expected=expected_filename
        )
        expected_sha = _expected_sha256(
            str(entry.get("sha256", "")),
            field=f"{component}.sha256",
        )
        content = path.read_bytes()
        if _sha256_bytes(content) != expected_sha:
            raise ValueError(f"{component} SHA-256 does not match its manifest")
        source_paths[component] = path
        source_content[component] = content
    if manifest["comparison"]["sha256"] != expected_comparison_sha256:
        raise ValueError("comparison SHA-256 does not match expected")

    with tempfile.TemporaryDirectory(prefix=".adjudication-source-snapshot-") as name:
        root = Path(name)
        snapshot_paths: dict[str, Path] = {}
        for component, content in source_content.items():
            path = root / _SOURCE_COMPONENTS[component]
            path.write_bytes(content)
            snapshot_paths[component] = path
        assessed = pd.Timestamp(manifest.get("assessed_date")).date()
        derived = build_dual_review_manifest(
            snapshot_paths["primary_reviews"],
            snapshot_paths["secondary_reviews"],
            snapshot_paths["selection"],
            snapshot_paths["comparison"],
            assessed_date=assessed,
        )
    for key, expected in derived.items():
        if key == "schema_version":
            continue
        if manifest.get(key) != expected:
            raise ValueError(
                f"completed review manifest is not the deterministic derivation: {key}"
            )

    frames = {
        "selection": _table_from_bytes(
            source_content["selection"],
            filename=_SOURCE_COMPONENTS["selection"],
        ),
        "primary_reviews": _table_from_bytes(
            source_content["primary_reviews"],
            filename=_SOURCE_COMPONENTS["primary_reviews"],
            contract=FullTextReviewRecord,
        ),
        "secondary_reviews": _table_from_bytes(
            source_content["secondary_reviews"],
            filename=_SOURCE_COMPONENTS["secondary_reviews"],
            contract=FullTextReviewRecord,
        ),
        "comparison": _table_from_bytes(
            source_content["comparison"],
            filename=_SOURCE_COMPONENTS["comparison"],
            contract=ReviewComparisonRecord,
        ),
    }
    return manifest, source_paths, source_content, frames


def _packet_rows(
    *,
    packet_id: str,
    parent_manifest_sha256: str,
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    primary, primary_errors = validate_records(
        frames["primary_reviews"], FullTextReviewRecord
    )
    secondary, secondary_errors = validate_records(
        frames["secondary_reviews"], FullTextReviewRecord
    )
    comparison, comparison_errors = validate_records(
        frames["comparison"], ReviewComparisonRecord
    )
    errors = pd.concat(
        [primary_errors, secondary_errors, comparison_errors], ignore_index=True
    )
    if not errors.empty:
        raise ValueError(
            "completed review rows failed contract validation: "
            f"{errors.head(5).to_dict(orient='records')}"
        )
    primary_by_id = {str(row["review_id"]): row for _, row in primary.iterrows()}
    secondary_by_id = {str(row["review_id"]): row for _, row in secondary.iterrows()}
    records: list[dict[str, Any]] = []
    for _, comparison_row in comparison.sort_values(
        ["queue_rank", "gene_symbol"], kind="stable"
    ).iterrows():
        comparison_model = ReviewComparisonRecord.model_validate(
            _clean_row(comparison_row.to_dict())
        )
        reviewer_a_row = primary_by_id.get(comparison_model.primary_review_id)
        reviewer_b_row = secondary_by_id.get(comparison_model.secondary_review_id)
        if reviewer_a_row is None or reviewer_b_row is None:
            raise ValueError("comparison references an unknown review ID")
        reviewer_a = FullTextReviewRecord.model_validate(
            _clean_row(reviewer_a_row.to_dict())
        )
        reviewer_b = FullTextReviewRecord.model_validate(
            _clean_row(reviewer_b_row.to_dict())
        )
        immutable_pairs = {
            "batch_id": (reviewer_a.batch_id, reviewer_b.batch_id),
            "screen_id": (reviewer_a.screen_id, reviewer_b.screen_id),
            "source_family_id": (
                reviewer_a.source_family_id,
                reviewer_b.source_family_id,
            ),
            "doi": (reviewer_a.doi, reviewer_b.doi),
            "paper_url": (str(reviewer_a.paper_url), str(reviewer_b.paper_url)),
            "full_text_url": (
                str(reviewer_a.full_text_url),
                str(reviewer_b.full_text_url),
            ),
            "supplement_url": (
                str(reviewer_a.supplement_url),
                str(reviewer_b.supplement_url),
            ),
        }
        disagreements = [
            field for field, pair in immutable_pairs.items() if pair[0] != pair[1]
        ]
        if disagreements:
            raise ValueError(
                f"reviewers disagree on immutable source identity: {disagreements}"
            )
        comparison_sha = _canonical_model_sha(comparison_model)
        item_digest = _sha256_bytes(
            json.dumps(
                [packet_id, comparison_model.comparison_id, comparison_sha],
                separators=(",", ":"),
            ).encode("utf-8")
        )
        records.append(
            {
                "packet_item_id": f"{packet_id}:item:{item_digest}",
                "packet_id": packet_id,
                "comparison_id": comparison_model.comparison_id,
                "comparison_row_sha256": comparison_sha,
                "parent_dual_review_manifest_sha256": parent_manifest_sha256,
                "batch_id": comparison_model.batch_id,
                "queue_id": comparison_model.queue_id,
                "queue_rank": comparison_model.queue_rank,
                "screen_id": comparison_model.screen_id,
                "external_screen_id": comparison_model.external_screen_id,
                "gene_symbol": comparison_model.gene_symbol,
                "source_family_id": reviewer_a.source_family_id,
                "doi": reviewer_a.doi,
                "paper_url": str(reviewer_a.paper_url),
                "full_text_url": str(reviewer_a.full_text_url),
                "supplement_url": str(reviewer_a.supplement_url),
                "reviewer_a_review_id": reviewer_a.review_id,
                "reviewer_a_curator": reviewer_a.curator,
                "reviewer_a_row_sha256": _canonical_model_sha(reviewer_a),
                "reviewer_a_evidence_level": (
                    comparison_model.primary_evidence_level.value
                ),
                "reviewer_a_source_locator": (comparison_model.primary_source_locator),
                "reviewer_a_screen_model": reviewer_a.screen_model,
                "reviewer_a_treatment_contrast": reviewer_a.treatment_contrast,
                "reviewer_a_screen_replication": reviewer_a.screen_replication,
                "reviewer_a_notes": reviewer_a.notes,
                "reviewer_b_review_id": reviewer_b.review_id,
                "reviewer_b_curator": reviewer_b.curator,
                "reviewer_b_row_sha256": _canonical_model_sha(reviewer_b),
                "reviewer_b_evidence_level": (
                    comparison_model.secondary_evidence_level.value
                ),
                "reviewer_b_source_locator": (
                    comparison_model.secondary_source_locator
                ),
                "reviewer_b_screen_model": reviewer_b.screen_model,
                "reviewer_b_treatment_contrast": reviewer_b.treatment_contrast,
                "reviewer_b_screen_replication": reviewer_b.screen_replication,
                "reviewer_b_notes": reviewer_b.notes,
                "comparison_assessed_date": comparison_model.assessed_date,
                "human_adjudication_required": True,
            }
        )
    packet = pd.DataFrame.from_records(
        records,
        columns=list(AdjudicationPacketRecord.model_fields),
    )
    valid, errors = validate_records(packet, AdjudicationPacketRecord)
    if not errors.empty:
        raise ValueError(
            "adjudication packet failed contract validation: "
            f"{errors.head(5).to_dict(orient='records')}"
        )
    return valid


def _decision_template(packet: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, item in packet.iterrows():
        rows.append(
            {
                "decision_id": f"{item['packet_item_id']}:decision:v1",
                "packet_id": item["packet_id"],
                "packet_item_id": item["packet_item_id"],
                "comparison_id": item["comparison_id"],
                "comparison_row_sha256": item["comparison_row_sha256"],
                "reviewer_a_review_id": item["reviewer_a_review_id"],
                "reviewer_a_row_sha256": item["reviewer_a_row_sha256"],
                "reviewer_b_review_id": item["reviewer_b_review_id"],
                "reviewer_b_row_sha256": item["reviewer_b_row_sha256"],
                "parent_dual_review_manifest_sha256": item[
                    "parent_dual_review_manifest_sha256"
                ],
                "packet_manifest_sha256": None,
                "batch_id": item["batch_id"],
                "screen_id": item["screen_id"],
                "gene_symbol": item["gene_symbol"],
                "disposition": None,
                "validation_event_id": None,
                "validation_event_row_sha256": None,
                "followup_roster_status": None,
                "adjudicator_name": None,
                "adjudicator_id": None,
                "adjudicator_affiliation": None,
                "adjudicated_date": None,
                "source_evidence_reviewed_attested": None,
                "independent_human_decision_attested": None,
                "reviewer_identity_independence_attested": None,
                "model_outputs_unseen_attested": None,
                "no_automated_label_assignment_attested": None,
                "conflict_of_interest_declared": None,
                "conflict_of_interest_notes": None,
                "evidence_source_locator": None,
                "decision_rationale": None,
            }
        )
    return pd.DataFrame.from_records(
        rows,
        columns=list(AdjudicationDecisionRecord.model_fields),
    )


def _event_template(packet: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, item in packet.iterrows():
        row = {name: None for name in ValidationEventRecord.model_fields}
        row.update(
            {
                "screen_id": item["screen_id"],
                "gene_symbol": item["gene_symbol"],
                "source_url": item["paper_url"],
                "source_family_id": item["source_family_id"],
                "review_comparison_id": item["comparison_id"],
                "adjudication_packet_id": item["packet_id"],
                "adjudication_method_version": ADJUDICATION_METHOD_VERSION,
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(
        rows,
        columns=list(ValidationEventRecord.model_fields),
    )


def hash_validation_events(
    validation_events_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and hash exact event payloads without assigning any label."""

    path = Path(validation_events_path)
    content = path.read_bytes()
    if content.strip():
        events = _table_from_bytes(
            content,
            filename=path.name,
            contract=ValidationEventRecord,
        )
        if events.empty:
            if list(events.columns) != list(ValidationEventRecord.model_fields):
                raise ValueError(
                    "an empty validation-event table requires the canonical "
                    "contract header"
                )
            valid_events = pd.DataFrame(
                columns=list(ValidationEventRecord.model_fields)
            )
        else:
            valid_events, errors = validate_records(events, ValidationEventRecord)
            if not errors.empty or len(valid_events) != len(events):
                raise ValueError(
                    "validation events failed contract validation: "
                    f"{errors.head(5).to_dict(orient='records')}"
                )
    else:
        valid_events = pd.DataFrame(columns=list(ValidationEventRecord.model_fields))

    records = []
    for _, row in valid_events.sort_values("event_id", kind="stable").iterrows():
        event = ValidationEventRecord.model_validate(_clean_row(row.to_dict()))
        records.append(
            {
                "event_id": event.event_id,
                "validation_event_row_sha256": _canonical_model_sha(event),
            }
        )
    if _sha256_bytes(path.read_bytes()) != _sha256_bytes(content):
        raise ValueError("validation-event input changed during hashing")
    hashes = pd.DataFrame.from_records(
        records,
        columns=["event_id", "validation_event_row_sha256"],
    )
    return hashes, {
        "schema": "crispr-evidencerank.validation-event-hashes",
        "schema_version": 1,
        "method_version": ADJUDICATION_METHOD_VERSION,
        "input": {
            "filename": path.name,
            "sha256": _sha256_bytes(content),
        },
        "record_count": len(hashes),
        "label_assignment_performed": False,
    }


def _packet_readme(packet_id: str, rows: int) -> str:
    return f"""# Human validation adjudication packet

Packet: `{packet_id}`  
Items: {rows}

This bundle is unsigned and releases no validation labels. Reviewer evidence
levels are curator extractions, not final decisions. Do not infer `V2`, `V3`,
`F0`, or `D` from reviewer agreement.

For every row in `adjudication_decisions.template.tsv`, a named human must
inspect the cited source and choose exactly one disposition:

- `release_validation_event`: provide one fully populated event row;
- `no_qualifying_event`: the cited material is not a qualifying validation
  event; this is not `F0` and not an untested negative;
- `defer_unresolved`: evidence is insufficient and no label is released.

The decision template intentionally contains no prefilled disposition or label.
The event template is a worksheet only; delete non-release rows before
finalization. Each release decision must bind the canonical event row SHA-256,
and the finalizer separately pins the complete event-table SHA-256. The human
must attest that model outputs were unseen, no automated label was assigned,
and the adjudicator is independent of both reviewers. Because the frozen review
records lack stable person identifiers, independence remains a human
attestation plus a display-name sanity check, not cryptographic identity proof.
"""


def prepare_adjudication_packet(
    completed_review_manifest: str | Path,
    output_dir: str | Path,
    *,
    expected_completed_review_manifest_sha256: str,
    expected_comparison_sha256: str,
    packet_id: str,
    prepared_date: date,
) -> dict[str, Any]:
    """Atomically create a neutral, unsigned human-decision packet."""

    if not packet_id or any(character.isspace() for character in packet_id):
        raise ValueError("packet_id must be a stable identifier without whitespace")
    manifest_path = Path(completed_review_manifest)
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.publish.lock"
    expected_manifest_sha = _expected_sha256(
        expected_completed_review_manifest_sha256,
        field="expected_completed_review_manifest_sha256",
    )
    expected_comparison_sha = _expected_sha256(
        expected_comparison_sha256,
        field="expected_comparison_sha256",
    )
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"adjudication packet lock exists: {lock_path}") from exc
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        if _path_lexists(output):
            raise FileExistsError(f"adjudication packet already exists: {output}")
        manifest_content = manifest_path.read_bytes()
        manifest, source_paths, source_content, frames = (
            _validate_completed_review_snapshot(
                manifest_path,
                manifest_content,
                expected_manifest_sha256=expected_manifest_sha,
                expected_comparison_sha256=expected_comparison_sha,
            )
        )
        if prepared_date < pd.Timestamp(manifest["assessed_date"]).date():
            raise ValueError("prepared_date cannot precede the dual review")
        packet = _packet_rows(
            packet_id=packet_id,
            parent_manifest_sha256=expected_manifest_sha,
            frames=frames,
        )
        decisions = _decision_template(packet)
        events = _event_template(packet)
        with tempfile.TemporaryDirectory(
            dir=output.parent,
            prefix=f".{output.name}.work-",
        ) as work_name:
            staging = Path(work_name) / "publish"
            staging.mkdir()
            packet_path = staging / "adjudication_packet.tsv"
            decisions_path = staging / "adjudication_decisions.template.tsv"
            events_path = staging / "validation_events.template.tsv"
            readme_path = staging / "README.md"
            _write_tsv(packet, packet_path)
            _write_tsv(decisions, decisions_path)
            _write_tsv(events, events_path)
            readme_path.write_text(
                _packet_readme(packet_id, len(packet)),
                encoding="utf-8",
            )
            packet_manifest = {
                "schema": PACKET_MANIFEST_SCHEMA,
                "schema_version": PACKET_MANIFEST_SCHEMA_VERSION,
                "method_version": ADJUDICATION_METHOD_VERSION,
                "packet_id": packet_id,
                "prepared_date": prepared_date.isoformat(),
                "status": "unsigned_pending_human_adjudication",
                "human_adjudication_required": True,
                "record_count": len(packet),
                "released_label_count": 0,
                "benchmark_ready_count": 0,
                "parent_dual_review_manifest": {
                    "filename": manifest_path.name,
                    "sha256": expected_manifest_sha,
                },
                "source_files": {
                    component: {
                        "filename": _SOURCE_COMPONENTS[component],
                        "sha256": _sha256_bytes(source_content[component]),
                    }
                    for component in sorted(source_content)
                },
                "outputs": {
                    "packet": {
                        "filename": packet_path.name,
                        "sha256": _sha256(packet_path),
                        "record_count": len(packet),
                    },
                    "decision_template": {
                        "filename": decisions_path.name,
                        "sha256": _sha256(decisions_path),
                        "record_count": len(decisions),
                        "decision_outcome_fields_prefilled": False,
                    },
                    "validation_event_template": {
                        "filename": events_path.name,
                        "sha256": _sha256(events_path),
                        "record_count": len(events),
                        "label_fields_prefilled": False,
                    },
                    "readme": {
                        "filename": readme_path.name,
                        "sha256": _sha256(readme_path),
                    },
                },
            }
            _write_json(
                packet_manifest,
                staging / "adjudication_packet_manifest.json",
            )
            changed = [
                str(path)
                for component, path in source_paths.items()
                if _sha256_bytes(path.read_bytes())
                != _sha256_bytes(source_content[component])
            ]
            if _sha256_bytes(manifest_path.read_bytes()) != expected_manifest_sha:
                changed.append(str(manifest_path))
            if changed:
                raise ValueError(f"adjudication inputs changed during run: {changed}")
            if _path_lexists(output):
                raise FileExistsError(
                    f"adjudication packet destination appeared: {output}"
                )
            staging.rename(output)
        return packet_manifest
    finally:
        os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _verify_packet_manifest(
    manifest_path: Path,
    content: bytes,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if _sha256_bytes(content) != expected_sha256:
        raise ValueError("adjudication packet manifest SHA-256 does not match expected")
    manifest = _json_object(content, label="adjudication packet manifest")
    if (
        manifest.get("schema") != PACKET_MANIFEST_SCHEMA
        or manifest.get("schema_version") != PACKET_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("unsupported adjudication packet manifest")
    required = {
        "status": "unsigned_pending_human_adjudication",
        "human_adjudication_required": True,
        "released_label_count": 0,
        "benchmark_ready_count": 0,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"packet manifest has invalid {key}")
    packet_entry = manifest.get("outputs", {}).get("packet")
    if not isinstance(packet_entry, dict):
        raise ValueError("packet manifest omits the packet table")
    packet_path = _canonical_member(
        manifest_path.parent,
        packet_entry.get("filename"),
        expected="adjudication_packet.tsv",
    )
    packet_content = packet_path.read_bytes()
    if _sha256_bytes(packet_content) != packet_entry.get("sha256"):
        raise ValueError("adjudication packet table checksum mismatch")
    packet = _table_from_bytes(
        packet_content,
        filename=packet_path.name,
        contract=AdjudicationPacketRecord,
    )
    valid, errors = validate_records(packet, AdjudicationPacketRecord)
    if not errors.empty or len(valid) != manifest.get("record_count"):
        raise ValueError("adjudication packet rows do not match the manifest")
    return manifest, valid


def finalize_adjudication(
    packet_manifest: str | Path,
    decisions_path: str | Path,
    validation_events_path: str | Path,
    output_dir: str | Path,
    *,
    expected_packet_manifest_sha256: str,
    expected_decisions_sha256: str,
    expected_validation_events_sha256: str,
    adjudicated_date: date,
) -> dict[str, Any]:
    """Release only explicitly attested human decisions as an atomic bundle."""

    packet_manifest_path = Path(packet_manifest)
    decisions_path = Path(decisions_path)
    events_path = Path(validation_events_path)
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.publish.lock"
    expected_packet_sha = _expected_sha256(
        expected_packet_manifest_sha256,
        field="expected_packet_manifest_sha256",
    )
    expected_decision_sha = _expected_sha256(
        expected_decisions_sha256,
        field="expected_decisions_sha256",
    )
    expected_events_sha = _expected_sha256(
        expected_validation_events_sha256,
        field="expected_validation_events_sha256",
    )
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"adjudication release lock exists: {lock_path}") from exc
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        if _path_lexists(output):
            raise FileExistsError(f"adjudication release already exists: {output}")
        input_content = {
            "packet_manifest": packet_manifest_path.read_bytes(),
            "decisions": decisions_path.read_bytes(),
            "events": events_path.read_bytes(),
        }
        packet_meta, packet = _verify_packet_manifest(
            packet_manifest_path,
            input_content["packet_manifest"],
            expected_sha256=expected_packet_sha,
        )
        try:
            packet_prepared_date = pd.Timestamp(packet_meta["prepared_date"]).date()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("packet manifest has an invalid prepared_date") from exc
        if adjudicated_date < packet_prepared_date:
            raise ValueError("adjudication cannot predate its checksum-bound packet")
        if _sha256_bytes(input_content["decisions"]) != expected_decision_sha:
            raise ValueError("adjudication decisions SHA-256 does not match expected")
        if _sha256_bytes(input_content["events"]) != expected_events_sha:
            raise ValueError("validation events SHA-256 does not match expected")
        decisions = _table_from_bytes(
            input_content["decisions"],
            filename=decisions_path.name,
            contract=AdjudicationDecisionRecord,
        )
        valid_decisions, decision_errors = validate_records(
            decisions, AdjudicationDecisionRecord
        )
        if not decision_errors.empty:
            raise ValueError(
                "adjudication decisions failed contract validation: "
                f"{decision_errors.head(5).to_dict(orient='records')}"
            )
        if len(valid_decisions) != len(packet):
            raise ValueError("decisions must cover every packet item exactly once")
        if valid_decisions["packet_item_id"].astype(str).duplicated().any():
            raise ValueError("packet_item_id decisions must be unique")
        packet_by_item = packet.set_index("packet_item_id", drop=False)
        decision_items = set(valid_decisions["packet_item_id"].astype(str))
        if decision_items != set(packet_by_item.index.astype(str)):
            raise ValueError("decisions have missing or unexpected packet items")
        immutable_pairs = {
            "packet_id": "packet_id",
            "comparison_id": "comparison_id",
            "comparison_row_sha256": "comparison_row_sha256",
            "reviewer_a_review_id": "reviewer_a_review_id",
            "reviewer_a_row_sha256": "reviewer_a_row_sha256",
            "reviewer_b_review_id": "reviewer_b_review_id",
            "reviewer_b_row_sha256": "reviewer_b_row_sha256",
            "parent_dual_review_manifest_sha256": (
                "parent_dual_review_manifest_sha256"
            ),
            "batch_id": "batch_id",
            "screen_id": "screen_id",
            "gene_symbol": "gene_symbol",
        }
        for _, decision in valid_decisions.iterrows():
            item = packet_by_item.loc[str(decision["packet_item_id"])]
            changed = [
                decision_field
                for decision_field, packet_field in immutable_pairs.items()
                if str(decision[decision_field]) != str(item[packet_field])
            ]
            if changed:
                raise ValueError(
                    f"adjudication decision changed immutable packet fields: {changed}"
                )
            if str(decision["packet_manifest_sha256"]) != expected_packet_sha:
                raise ValueError("decision is not bound to this packet manifest")
            if pd.Timestamp(decision["adjudicated_date"]).date() != adjudicated_date:
                raise ValueError("decision adjudicated_date differs from the release")
            if adjudicated_date < pd.Timestamp(item["comparison_assessed_date"]).date():
                raise ValueError("adjudication cannot predate its review comparison")
            curator_tokens = {
                " ".join(str(item[field]).casefold().split())
                for field in ("reviewer_a_curator", "reviewer_b_curator")
            }
            if " ".join(str(decision["adjudicator_name"]).casefold().split()) in (
                curator_tokens
            ):
                raise ValueError("adjudicator must be distinct from both reviewers")

        if input_content["events"].strip():
            events = _table_from_bytes(
                input_content["events"],
                filename=events_path.name,
                contract=ValidationEventRecord,
            )
            if events.empty:
                if list(events.columns) != list(ValidationEventRecord.model_fields):
                    raise ValueError(
                        "an empty validation-event table requires the canonical "
                        "contract header"
                    )
                valid_events = pd.DataFrame(
                    columns=list(ValidationEventRecord.model_fields)
                )
            else:
                valid_events, event_errors = validate_records(
                    events, ValidationEventRecord
                )
                if not event_errors.empty:
                    raise ValueError(
                        "validation events failed contract validation: "
                        f"{event_errors.head(5).to_dict(orient='records')}"
                    )
        else:
            valid_events = pd.DataFrame(
                columns=list(ValidationEventRecord.model_fields)
            )
        released = valid_decisions.loc[
            valid_decisions["disposition"]
            .astype(str)
            .eq(AdjudicationDecisionDisposition.RELEASE_VALIDATION_EVENT.value)
        ]
        expected_event_ids = set(released["validation_event_id"].astype(str))
        observed_event_ids = set(
            valid_events.get("event_id", pd.Series(dtype=str)).astype(str)
        )
        if expected_event_ids != observed_event_ids:
            raise ValueError("released decisions and validation events do not match")
        decisions_by_event = {
            str(row["validation_event_id"]): row for _, row in released.iterrows()
        }
        for _, event in valid_events.iterrows():
            decision = decisions_by_event[str(event["event_id"])]
            item = packet_by_item.loc[str(decision["packet_item_id"])]
            event_record = ValidationEventRecord.model_validate(
                _clean_row(event.to_dict())
            )
            if str(decision["validation_event_row_sha256"]) != _canonical_model_sha(
                event_record
            ):
                raise ValueError(
                    "released decision is not bound to the canonical event row"
                )
            if str(event["label_code"]) == LabelCode.U.value:
                raise ValueError("this workflow cannot release U as an event")
            if str(event["adjudication_status"]) != "consensus_adjudicated":
                raise ValueError("released events require consensus_adjudicated status")
            event_links = {
                "screen_id": item["screen_id"],
                "gene_symbol": item["gene_symbol"],
                "source_family_id": item["source_family_id"],
                "review_comparison_id": item["comparison_id"],
                "adjudication_decision_id": decision["decision_id"],
                "adjudication_packet_id": item["packet_id"],
                "adjudication_method_version": ADJUDICATION_METHOD_VERSION,
                "curator": decision["adjudicator_name"],
            }
            mismatches = [
                field
                for field, expected in event_links.items()
                if str(event[field]) != str(expected)
            ]
            if mismatches:
                raise ValueError(
                    f"released event disagrees with its human decision: {mismatches}"
                )
            if str(event["source_url"]) != str(item["paper_url"]):
                raise ValueError("released event source_url differs from packet source")
            if pd.Timestamp(event["evidence_available_date"]).date() > adjudicated_date:
                raise ValueError("event evidence cannot postdate adjudication")
            if str(event["label_code"]) in {"V2", "V3", "F0", "D"} and (
                pd.isna(event["contrast_id"]) or not str(event["contrast_id"]).strip()
            ):
                raise ValueError("primary labels require an explicit contrast_id")

        candidate_labels = _resolve_released_validation_events(
            valid_events,
            valid_decisions,
        )
        nonreleased = valid_decisions.loc[
            ~valid_decisions["disposition"]
            .astype(str)
            .eq(AdjudicationDecisionDisposition.RELEASE_VALIDATION_EVENT.value)
        ].copy()
        with tempfile.TemporaryDirectory(
            dir=output.parent,
            prefix=f".{output.name}.work-",
        ) as work_name:
            staging = Path(work_name) / "publish"
            staging.mkdir()
            raw_decisions_path = staging / "signed_adjudication_decisions.tsv"
            events_output_path = staging / "validation_events.tsv"
            nonreleased_path = staging / "nonreleased_decisions.tsv"
            candidates_path = staging / "candidate_adjudications.tsv"
            raw_decisions_path.write_bytes(input_content["decisions"])
            _write_tsv(valid_events, events_output_path)
            _write_tsv(nonreleased, nonreleased_path)
            _write_tsv(candidate_labels, candidates_path)
            label_counts = {
                str(key): int(value)
                for key, value in valid_events.get("label_code", pd.Series(dtype=str))
                .value_counts()
                .sort_index()
                .items()
            }
            disposition_counts = {
                str(key): int(value)
                for key, value in valid_decisions["disposition"]
                .astype(str)
                .value_counts()
                .sort_index()
                .items()
            }
            decision_record_sha256 = {
                str(row["decision_id"]): _canonical_model_sha(
                    AdjudicationDecisionRecord.model_validate(_clean_row(row.to_dict()))
                )
                for _, row in valid_decisions.sort_values(
                    "decision_id", kind="stable"
                ).iterrows()
            }
            validation_event_record_sha256 = {
                str(row["event_id"]): _canonical_model_sha(
                    ValidationEventRecord.model_validate(_clean_row(row.to_dict()))
                )
                for _, row in valid_events.sort_values(
                    "event_id", kind="stable"
                ).iterrows()
            }
            packet_item_record_sha256 = {
                str(row["packet_item_id"]): _canonical_model_sha(
                    AdjudicationPacketRecord.model_validate(_clean_row(row.to_dict()))
                )
                for _, row in packet.sort_values(
                    "packet_item_id", kind="stable"
                ).iterrows()
            }
            release_manifest = {
                "schema": RELEASE_MANIFEST_SCHEMA,
                "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
                "method_version": ADJUDICATION_METHOD_VERSION,
                "packet_id": packet_meta["packet_id"],
                "adjudicated_date": adjudicated_date.isoformat(),
                "status": "human_adjudication_released_without_readiness_promotion",
                "reviewer_identity_independence": {
                    "human_attestation_required": True,
                    "stable_reviewer_identity_ids_available": False,
                    "display_name_nonidentity_checked": True,
                },
                "parent_packet_manifest": {
                    "filename": packet_manifest_path.name,
                    "sha256": expected_packet_sha,
                },
                "decision_input": {
                    "filename": decisions_path.name,
                    "sha256": expected_decision_sha,
                    "record_count": len(valid_decisions),
                },
                "event_input": {
                    "filename": events_path.name,
                    "sha256": expected_events_sha,
                },
                "decision_count": len(valid_decisions),
                "packet_item_count": len(packet),
                "released_event_count": len(valid_events),
                "released_primary_label_count": int(
                    valid_events.get("label_code", pd.Series(dtype=str))
                    .astype(str)
                    .isin({"V2", "V3", "F0", "D"})
                    .sum()
                ),
                "candidate_adjudication_count": len(candidate_labels),
                "disposition_counts": disposition_counts,
                "label_counts": label_counts,
                "record_sha256": {
                    "decisions": decision_record_sha256,
                    "packet_items": packet_item_record_sha256,
                    "validation_events": validation_event_record_sha256,
                },
                "benchmark_ready_count": 0,
                "outputs": {
                    "decisions": {
                        "filename": raw_decisions_path.name,
                        "sha256": _sha256(raw_decisions_path),
                    },
                    "validation_events": {
                        "filename": events_output_path.name,
                        "sha256": _sha256(events_output_path),
                    },
                    "nonreleased_decisions": {
                        "filename": nonreleased_path.name,
                        "sha256": _sha256(nonreleased_path),
                    },
                    "candidate_adjudications": {
                        "filename": candidates_path.name,
                        "sha256": _sha256(candidates_path),
                    },
                },
            }
            _write_json(
                release_manifest,
                staging / "adjudication_release_manifest.json",
            )
            original_paths = {
                "packet_manifest": packet_manifest_path,
                "decisions": decisions_path,
                "events": events_path,
            }
            changed = [
                role
                for role, path in original_paths.items()
                if _sha256_bytes(path.read_bytes())
                != _sha256_bytes(input_content[role])
            ]
            if changed:
                raise ValueError(f"adjudication inputs changed during run: {changed}")
            if _path_lexists(output):
                raise FileExistsError(
                    f"adjudication release destination appeared: {output}"
                )
            staging.rename(output)
        return release_manifest
    finally:
        os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
