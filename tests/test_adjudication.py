from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from crispr_evidencerank.adjudication import (
    ADJUDICATION_METHOD_VERSION,
    finalize_adjudication,
    hash_validation_events,
    prepare_adjudication_packet,
)
from crispr_evidencerank.contracts import (
    AdjudicationDecisionRecord,
    AdjudicationPacketRecord,
    ValidationEventRecord,
)
from crispr_evidencerank.curation import build_dual_review_manifest

ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = (
    ROOT / "data" / "manifests" / "orcs_2.0.18" / "curation_batches" / "batch_001"
)
REVIEW_DATE = date(2026, 8, 2)
ADJUDICATION_DATE = date(2026, 8, 29)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_record_sha(
    record: ValidationEventRecord | AdjudicationDecisionRecord,
) -> str:
    payload = json.dumps(
        record.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _completed_review_bundle(tmp_path: Path) -> tuple[Path, str, str]:
    bundle = tmp_path / "completed_review"
    bundle.mkdir()
    for filename in (
        "selection.tsv",
        "reviews.tsv",
        "reviews_curator_2.tsv",
        "review_comparison.tsv",
    ):
        (bundle / filename).write_bytes((BATCH_DIR / filename).read_bytes())

    manifest = build_dual_review_manifest(
        bundle / "reviews.tsv",
        bundle / "reviews_curator_2.tsv",
        bundle / "selection.tsv",
        bundle / "review_comparison.tsv",
        assessed_date=REVIEW_DATE,
    )
    manifest_path = bundle / "dual_review_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return (
        manifest_path,
        _sha256(manifest_path),
        _sha256(bundle / "review_comparison.tsv"),
    )


def _prepared_packet(tmp_path: Path) -> tuple[Path, dict[str, object], str]:
    manifest_path, manifest_sha, comparison_sha = _completed_review_bundle(tmp_path)
    output = tmp_path / "adjudication_packet"
    manifest = prepare_adjudication_packet(
        manifest_path,
        output,
        expected_completed_review_manifest_sha256=manifest_sha,
        expected_comparison_sha256=comparison_sha,
        packet_id="orcs-2.0.18-batch-001-adjudication-v1",
        prepared_date=ADJUDICATION_DATE,
    )
    packet_manifest_path = output / "adjudication_packet_manifest.json"
    return packet_manifest_path, manifest, _sha256(packet_manifest_path)


def _signed_decisions(
    packet_manifest_path: Path,
    packet_manifest_sha: str,
    output_path: Path,
    *,
    release_first: bool,
) -> tuple[pd.DataFrame, str | None]:
    packet_dir = packet_manifest_path.parent
    decisions = pd.read_csv(
        packet_dir / "adjudication_decisions.template.tsv",
        sep="\t",
        dtype="string",
    )
    decisions["packet_manifest_sha256"] = packet_manifest_sha
    decisions["disposition"] = "defer_unresolved"
    decisions["validation_event_id"] = pd.NA
    decisions["validation_event_row_sha256"] = pd.NA
    decisions["followup_roster_status"] = "positive_only_or_unclear"
    decisions["adjudicator_name"] = "Independent human adjudicator"
    decisions["adjudicator_id"] = "orcid:0000-0001-2345-6789"
    decisions["adjudicator_affiliation"] = "Independent validation team"
    decisions["adjudicated_date"] = ADJUDICATION_DATE.isoformat()
    decisions["source_evidence_reviewed_attested"] = True
    decisions["independent_human_decision_attested"] = True
    decisions["reviewer_identity_independence_attested"] = True
    decisions["model_outputs_unseen_attested"] = True
    decisions["no_automated_label_assignment_attested"] = True
    decisions["conflict_of_interest_declared"] = False
    decisions["conflict_of_interest_notes"] = pd.NA
    decisions["evidence_source_locator"] = "Human-reviewed source locator"
    decisions["decision_rationale"] = "Evidence remains unresolved after review."

    released_event_id: str | None = None
    if release_first:
        released_event_id = "validation-event:human-release:1"
        decisions.loc[0, "disposition"] = "release_validation_event"
        decisions.loc[0, "validation_event_id"] = released_event_id
        decisions.loc[0, "followup_roster_status"] = "complete_followup_roster"
        decisions.loc[0, "decision_rationale"] = (
            "The cited experiment meets the prespecified V2 criteria."
        )
    decisions.to_csv(output_path, sep="\t", index=False, lineterminator="\n")
    return decisions, released_event_id


def _released_event(
    packet_manifest_path: Path,
    decisions: pd.DataFrame,
    event_id: str,
    output_path: Path,
) -> pd.DataFrame:
    packet = pd.read_csv(
        packet_manifest_path.parent / "adjudication_packet.tsv",
        sep="\t",
        dtype="string",
    )
    item = packet.iloc[0]
    decision = decisions.iloc[0]
    row = {field: None for field in ValidationEventRecord.model_fields}
    row.update(
        {
            "event_id": event_id,
            "study_id": "study:human-validation:1",
            "screen_id": item["screen_id"],
            "contrast_id": "contrast:human-validation:1",
            "gene_symbol": item["gene_symbol"],
            "drug_name": "ETOPOSIDE",
            "cell_line": "KBM-7",
            "perturbation_modality": "CRISPR_KO",
            "phenotype_direction": "resistance",
            "label_code": "V2",
            "testing_status": "tested",
            "perturbation_confirmed": True,
            "independent_reagent_count": 2,
            "orthogonal_perturbation": False,
            "appropriate_control": True,
            "assay_adequate": True,
            "phenotype_reproduced": True,
            "opposite_direction_reproduced": False,
            "rescue_performed": False,
            "causal_reversal_performed": False,
            "effect_size": 1.2,
            "effect_metric": "log2_fold_change",
            "p_value": 0.01,
            "source_url": item["paper_url"],
            "source_locator": "Human-reviewed source locator",
            "source_family_id": item["source_family_id"],
            "evidence_available_date": REVIEW_DATE.isoformat(),
            "review_comparison_id": item["comparison_id"],
            "adjudication_decision_id": decision["decision_id"],
            "adjudication_packet_id": item["packet_id"],
            "adjudication_method_version": ADJUDICATION_METHOD_VERSION,
            "curator": decision["adjudicator_name"],
            "adjudication_status": "consensus_adjudicated",
            "notes": "Synthetic test of an explicit human release.",
        }
    )
    events = pd.DataFrame([row], columns=list(ValidationEventRecord.model_fields))
    events.to_csv(output_path, sep="\t", index=False, lineterminator="\n")
    return events


def test_prepare_packet_is_neutral_and_releases_no_automatic_label(tmp_path: Path):
    packet_manifest_path, manifest, _ = _prepared_packet(tmp_path)
    packet_dir = packet_manifest_path.parent
    packet = pd.read_csv(packet_dir / "adjudication_packet.tsv", sep="\t")
    decisions = pd.read_csv(
        packet_dir / "adjudication_decisions.template.tsv", sep="\t"
    )
    events = pd.read_csv(packet_dir / "validation_events.template.tsv", sep="\t")

    assert len(packet) == 20
    assert list(packet.columns) == list(AdjudicationPacketRecord.model_fields)
    assert "label_code" not in packet.columns
    assert "disposition" not in packet.columns
    assert decisions["disposition"].isna().all()
    assert decisions["validation_event_id"].isna().all()
    assert events["label_code"].isna().all()
    assert events["adjudication_status"].isna().all()
    assert manifest["status"] == "unsigned_pending_human_adjudication"
    assert manifest["released_label_count"] == 0
    assert manifest["benchmark_ready_count"] == 0
    assert (
        manifest["outputs"]["decision_template"]["decision_outcome_fields_prefilled"]
        is False
    )
    assert (
        manifest["outputs"]["validation_event_template"]["label_fields_prefilled"]
        is False
    )


@pytest.mark.parametrize("wrong_binding", ["manifest", "comparison"])
def test_prepare_packet_rejects_an_incorrect_explicit_sha(
    tmp_path: Path, wrong_binding: str
):
    manifest_path, manifest_sha, comparison_sha = _completed_review_bundle(tmp_path)
    if wrong_binding == "manifest":
        manifest_sha = "0" * 64
    else:
        comparison_sha = "0" * 64

    with pytest.raises(ValueError, match="SHA-256 does not match expected"):
        prepare_adjudication_packet(
            manifest_path,
            tmp_path / "packet",
            expected_completed_review_manifest_sha256=manifest_sha,
            expected_comparison_sha256=comparison_sha,
            packet_id="packet:test:wrong-sha",
            prepared_date=ADJUDICATION_DATE,
        )
    assert not (tmp_path / "packet").exists()


def test_prepare_packet_never_overwrites_existing_or_symlink_destinations(
    tmp_path: Path,
):
    manifest_path, manifest_sha, comparison_sha = _completed_review_bundle(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        prepare_adjudication_packet(
            manifest_path,
            existing,
            expected_completed_review_manifest_sha256=manifest_sha,
            expected_comparison_sha256=comparison_sha,
            packet_id="packet:test:existing",
            prepared_date=ADJUDICATION_DATE,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"

    dangling = tmp_path / "dangling-packet"
    dangling.symlink_to(tmp_path / "does-not-exist", target_is_directory=True)
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_adjudication_packet(
            manifest_path,
            dangling,
            expected_completed_review_manifest_sha256=manifest_sha,
            expected_comparison_sha256=comparison_sha,
            packet_id="packet:test:symlink",
            prepared_date=ADJUDICATION_DATE,
        )
    assert dangling.is_symlink()


def test_decision_contract_rejects_a_false_human_attestation(tmp_path: Path):
    packet_manifest_path, _, packet_manifest_sha = _prepared_packet(tmp_path)
    decisions_path = tmp_path / "decisions.tsv"
    decisions, _ = _signed_decisions(
        packet_manifest_path,
        packet_manifest_sha,
        decisions_path,
        release_first=False,
    )
    record = decisions.iloc[0].to_dict()
    record["no_automated_label_assignment_attested"] = False

    with pytest.raises(ValidationError, match="attestations must be true"):
        AdjudicationDecisionRecord.model_validate(record)


@pytest.mark.parametrize(
    "wrong_binding", ["packet_manifest", "decisions", "validation_events"]
)
def test_finalize_rejects_an_incorrect_explicit_sha(tmp_path: Path, wrong_binding: str):
    packet_manifest_path, _, packet_manifest_sha = _prepared_packet(tmp_path)
    decisions_path = tmp_path / "signed-decisions.tsv"
    _signed_decisions(
        packet_manifest_path,
        packet_manifest_sha,
        decisions_path,
        release_first=False,
    )
    decisions_sha = _sha256(decisions_path)
    if wrong_binding == "packet_manifest":
        packet_manifest_sha = "0" * 64
    else:
        decisions_sha = "0" * 64
    events_path = tmp_path / "validation-events.tsv"
    events_path.write_bytes(b"")
    events_sha = _sha256(events_path)
    if wrong_binding == "validation_events":
        events_sha = "0" * 64

    with pytest.raises(ValueError, match="SHA-256 does not match expected"):
        finalize_adjudication(
            packet_manifest_path,
            decisions_path,
            events_path,
            tmp_path / "release",
            expected_packet_manifest_sha256=packet_manifest_sha,
            expected_decisions_sha256=decisions_sha,
            expected_validation_events_sha256=events_sha,
            adjudicated_date=ADJUDICATION_DATE,
        )
    assert not (tmp_path / "release").exists()


def test_finalize_all_deferred_decisions_releases_zero_labels(tmp_path: Path):
    packet_manifest_path, _, packet_manifest_sha = _prepared_packet(tmp_path)
    decisions_path = tmp_path / "signed-decisions.tsv"
    _signed_decisions(
        packet_manifest_path,
        packet_manifest_sha,
        decisions_path,
        release_first=False,
    )
    events_path = tmp_path / "validation-events.tsv"
    events_path.write_bytes(b"")
    release_dir = tmp_path / "release"

    manifest = finalize_adjudication(
        packet_manifest_path,
        decisions_path,
        events_path,
        release_dir,
        expected_packet_manifest_sha256=packet_manifest_sha,
        expected_decisions_sha256=_sha256(decisions_path),
        expected_validation_events_sha256=_sha256(events_path),
        adjudicated_date=ADJUDICATION_DATE,
    )

    assert manifest["decision_count"] == 20
    assert manifest["released_event_count"] == 0
    assert manifest["released_primary_label_count"] == 0
    assert manifest["candidate_adjudication_count"] == 0
    assert manifest["benchmark_ready_count"] == 0
    assert manifest["disposition_counts"] == {"defer_unresolved": 20}
    assert len(manifest["record_sha256"]["decisions"]) == 20
    assert manifest["record_sha256"]["validation_events"] == {}
    candidates = pd.read_csv(release_dir / "candidate_adjudications.tsv", sep="\t")
    assert candidates.empty

    with pytest.raises(FileExistsError, match="already exists"):
        finalize_adjudication(
            packet_manifest_path,
            decisions_path,
            events_path,
            release_dir,
            expected_packet_manifest_sha256=packet_manifest_sha,
            expected_decisions_sha256=_sha256(decisions_path),
            expected_validation_events_sha256=_sha256(events_path),
            adjudicated_date=ADJUDICATION_DATE,
        )


def test_finalize_accepts_canonical_header_only_event_table(tmp_path: Path):
    packet_manifest_path, _, packet_manifest_sha = _prepared_packet(tmp_path)
    decisions_path = tmp_path / "signed-decisions.tsv"
    _signed_decisions(
        packet_manifest_path,
        packet_manifest_sha,
        decisions_path,
        release_first=False,
    )
    events_path = tmp_path / "validation-events.tsv"
    pd.DataFrame(columns=list(ValidationEventRecord.model_fields)).to_csv(
        events_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    manifest = finalize_adjudication(
        packet_manifest_path,
        decisions_path,
        events_path,
        tmp_path / "release",
        expected_packet_manifest_sha256=packet_manifest_sha,
        expected_decisions_sha256=_sha256(decisions_path),
        expected_validation_events_sha256=_sha256(events_path),
        adjudicated_date=ADJUDICATION_DATE,
    )

    assert manifest["released_event_count"] == 0
    assert manifest["record_sha256"]["validation_events"] == {}


def test_finalize_rejects_adjudication_before_checksum_bound_packet(
    tmp_path: Path,
):
    packet_manifest_path, _, packet_manifest_sha = _prepared_packet(tmp_path)
    decisions_path = tmp_path / "signed-decisions.tsv"
    _signed_decisions(
        packet_manifest_path,
        packet_manifest_sha,
        decisions_path,
        release_first=False,
    )
    events_path = tmp_path / "validation-events.tsv"
    events_path.write_bytes(b"")

    with pytest.raises(
        ValueError,
        match="adjudication cannot predate its checksum-bound packet",
    ):
        finalize_adjudication(
            packet_manifest_path,
            decisions_path,
            events_path,
            tmp_path / "release",
            expected_packet_manifest_sha256=packet_manifest_sha,
            expected_decisions_sha256=_sha256(decisions_path),
            expected_validation_events_sha256=_sha256(events_path),
            adjudicated_date=date(2026, 8, 28),
        )
    assert not (tmp_path / "release").exists()


def test_finalize_one_explicit_release_emits_only_that_human_label(tmp_path: Path):
    packet_manifest_path, _, packet_manifest_sha = _prepared_packet(tmp_path)
    decisions_path = tmp_path / "signed-decisions.tsv"
    decisions, event_id = _signed_decisions(
        packet_manifest_path,
        packet_manifest_sha,
        decisions_path,
        release_first=True,
    )
    assert event_id is not None
    events_path = tmp_path / "validation-events.tsv"
    events = _released_event(packet_manifest_path, decisions, event_id, events_path)
    event_record = ValidationEventRecord.model_validate(
        {
            key: (None if pd.isna(value) else value)
            for key, value in events.iloc[0].to_dict().items()
        }
    )
    decisions.loc[0, "validation_event_row_sha256"] = _canonical_record_sha(
        event_record
    )
    decisions.to_csv(decisions_path, sep="\t", index=False, lineterminator="\n")
    release_dir = tmp_path / "release"

    manifest = finalize_adjudication(
        packet_manifest_path,
        decisions_path,
        events_path,
        release_dir,
        expected_packet_manifest_sha256=packet_manifest_sha,
        expected_decisions_sha256=_sha256(decisions_path),
        expected_validation_events_sha256=_sha256(events_path),
        adjudicated_date=ADJUDICATION_DATE,
    )

    assert manifest["released_event_count"] == 1
    assert manifest["released_primary_label_count"] == 1
    assert manifest["candidate_adjudication_count"] == 1
    assert manifest["label_counts"] == {"V2": 1}
    assert manifest["benchmark_ready_count"] == 0
    assert manifest["record_sha256"]["validation_events"] == {
        event_id: _canonical_record_sha(event_record)
    }
    released_decision = AdjudicationDecisionRecord.model_validate(
        {
            key: (None if pd.isna(value) else value)
            for key, value in decisions.iloc[0].to_dict().items()
        }
    )
    assert manifest["record_sha256"]["decisions"][
        released_decision.decision_id
    ] == _canonical_record_sha(released_decision)
    candidate = pd.read_csv(release_dir / "candidate_adjudications.tsv", sep="\t").iloc[
        0
    ]
    assert candidate["label_code"] == "V2"
    assert candidate["validation_event_count"] == 1
    nonreleased = pd.read_csv(release_dir / "nonreleased_decisions.tsv", sep="\t")
    assert len(nonreleased) == 19
    assert nonreleased["disposition"].eq("defer_unresolved").all()


def test_finalize_rejects_event_payload_not_bound_to_human_decision(
    tmp_path: Path,
):
    packet_manifest_path, _, packet_manifest_sha = _prepared_packet(tmp_path)
    decisions_path = tmp_path / "signed-decisions.tsv"
    decisions, event_id = _signed_decisions(
        packet_manifest_path,
        packet_manifest_sha,
        decisions_path,
        release_first=True,
    )
    assert event_id is not None
    events_path = tmp_path / "validation-events.tsv"
    events = _released_event(packet_manifest_path, decisions, event_id, events_path)
    original_record = ValidationEventRecord.model_validate(
        {
            key: (None if pd.isna(value) else value)
            for key, value in events.iloc[0].to_dict().items()
        }
    )
    decisions.loc[0, "validation_event_row_sha256"] = _canonical_record_sha(
        original_record
    )
    decisions.to_csv(decisions_path, sep="\t", index=False, lineterminator="\n")

    events.loc[0, "notes"] = "Payload changed after the human attestation."
    events.to_csv(events_path, sep="\t", index=False, lineterminator="\n")

    with pytest.raises(ValueError, match="not bound to the canonical event row"):
        finalize_adjudication(
            packet_manifest_path,
            decisions_path,
            events_path,
            tmp_path / "release",
            expected_packet_manifest_sha256=packet_manifest_sha,
            expected_decisions_sha256=_sha256(decisions_path),
            expected_validation_events_sha256=_sha256(events_path),
            adjudicated_date=ADJUDICATION_DATE,
        )


def test_hash_validation_events_emits_the_finalizer_canonical_digest(
    tmp_path: Path,
):
    packet_manifest_path, _, packet_manifest_sha = _prepared_packet(tmp_path)
    decisions_path = tmp_path / "decisions.tsv"
    decisions, event_id = _signed_decisions(
        packet_manifest_path,
        packet_manifest_sha,
        decisions_path,
        release_first=True,
    )
    assert event_id is not None
    events_path = tmp_path / "validation-events.tsv"
    events = _released_event(packet_manifest_path, decisions, event_id, events_path)
    expected = _canonical_record_sha(
        ValidationEventRecord.model_validate(
            {
                key: (None if pd.isna(value) else value)
                for key, value in events.iloc[0].to_dict().items()
            }
        )
    )

    hashes, metadata = hash_validation_events(events_path)

    assert hashes.to_dict(orient="records") == [
        {
            "event_id": event_id,
            "validation_event_row_sha256": expected,
        }
    ]
    assert metadata["input"]["sha256"] == _sha256(events_path)
    assert metadata["label_assignment_performed"] is False
