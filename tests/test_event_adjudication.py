from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

import crispr_evidencerank.cli as cli_module
from crispr_evidencerank.cli import build_parser
from crispr_evidencerank.contracts import (
    EventAdjudicationRecord,
    EventReviewRecord,
)
from crispr_evidencerank.event_adjudication import (
    compare_event_reviews,
    finalize_event_adjudication,
    prepare_event_review_packet,
    summarize_event_review_agreement,
    write_event_review_comparison_bundle,
    write_event_review_packet,
    write_final_adjudication_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
RUBRIC_PATH = ROOT / "config" / "event_review_rubric.yaml"
RUBRIC_VERSION = "gold_event_review_v1.0.0"

PREPARED_AT = datetime(2026, 9, 4, 8, 0, tzinfo=UTC)
REVIEWED_A_AT = PREPARED_AT + timedelta(hours=1)
REVIEWED_B_AT = PREPARED_AT + timedelta(hours=2)
COMPARED_AT = PREPARED_AT + timedelta(hours=3)
ADJUDICATED_AT = PREPARED_AT + timedelta(hours=4)
RELEASED_AT = PREPARED_AT + timedelta(hours=5)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _actors() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "actor_id": "packet-preparer",
                "actor_type": "ai_system",
                "display_name": "Packet preparation system",
                "identity_verified": True,
                "affiliation": None,
                "persistent_id": "test:packet-preparer",
                "notes": None,
            },
            {
                "actor_id": "reviewer-a",
                "actor_type": "human",
                "display_name": "Reviewer A",
                "identity_verified": True,
                "affiliation": "Example University",
                "persistent_id": "test:human:a",
                "notes": None,
            },
            {
                "actor_id": "reviewer-b",
                "actor_type": "human",
                "display_name": "Reviewer B",
                "identity_verified": True,
                "affiliation": "Example University",
                "persistent_id": "test:human:b",
                "notes": None,
            },
            {
                "actor_id": "adjudicator-c",
                "actor_type": "human",
                "display_name": "Adjudicator C",
                "identity_verified": True,
                "affiliation": "Independent Institute",
                "persistent_id": "test:human:c",
                "notes": None,
            },
        ]
    )


def _raw_cards(count: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "packet_id": "gold-packet-001",
                "packet_version": 1,
                "packet_event_id": f"opaque-event-{index:03d}",
                "batch_id": "gold-batch-001",
                "source_family_id": f"source-family-{index:03d}",
                "study_id": f"study-{index:03d}",
                "screen_id": f"screen-{index:03d}",
                "contrast_id": f"contrast-{index:03d}",
                "gene_symbol": f"GENE{index}",
                "screen_predicted_direction": "resistance",
                "screen_drug": "olaparib",
                "screen_cell_model": "MDA-MB-468",
                "validation_drug": "olaparib",
                "validation_cell_model": "MDA-MB-468",
                "reported_perturbation_modality": "CRISPR_KO",
                "reported_assay_endpoint": "clonogenic survival",
                "source_url": f"https://example.org/source/{index}",
                "source_locator": f"Figure {index} and Methods",
                "prepared_at_utc": PREPARED_AT,
                "prepared_by_actor_id": "packet-preparer",
            }
            for index in range(1, count + 1)
        ]
    )


def _packet(count: int = 3) -> pd.DataFrame:
    packet, _ = prepare_event_review_packet(
        _raw_cards(count),
        _actors(),
        rubric_version=RUBRIC_VERSION,
        randomization_seed="fixed-test-seed",
    )
    return packet


def _decision_for_label(label: str) -> dict[str, object]:
    decision: dict[str, object] = {
        "event_scope": "qualifying_validation",
        "context_match": "exact",
        "testing_status": "tested",
        "perturbation_confirmed": "yes",
        "independent_reagent_count": 2,
        "orthogonal_perturbation": "no",
        "appropriate_control": "yes",
        "assay_adequate": "yes",
        "outcome_relation": "concordant",
        "null_result_evaluable": "not_applicable",
        "rescue_or_causal_reversal": "no",
        "proposed_label_code": label,
        "confidence": "high",
        "reason_codes": "matched_context|source_verified",
    }
    if label == "V3":
        decision["rescue_or_causal_reversal"] = "yes"
    elif label == "F0":
        decision["outcome_relation"] = "no_effect"
        decision["null_result_evaluable"] = "yes"
    elif label == "D":
        decision["outcome_relation"] = "opposite"
    elif label == "T":
        decision.update(
            {
                "perturbation_confirmed": "no",
                "independent_reagent_count": 0,
                "orthogonal_perturbation": "not_applicable",
                "appropriate_control": "no",
                "assay_adequate": "no",
                "outcome_relation": "not_interpretable",
            }
        )
    elif label == "U":
        decision.update(
            {
                "event_scope": "unclear",
                "context_match": "unclear",
                "testing_status": "unknown",
                "perturbation_confirmed": "not_reported",
                "independent_reagent_count": None,
                "orthogonal_perturbation": "not_reported",
                "appropriate_control": "not_reported",
                "assay_adequate": "not_reported",
                "outcome_relation": "not_reported",
                "null_result_evaluable": "not_reported",
                "rescue_or_causal_reversal": "not_reported",
            }
        )
    return decision


def _reviews(
    packet: pd.DataFrame,
    reviewer_id: str,
    *,
    labels: list[str] | None = None,
    reviewed_at: datetime | None = None,
) -> pd.DataFrame:
    if labels is None:
        labels = ["V2"] * len(packet)
    assert len(labels) == len(packet)
    when = reviewed_at or (
        REVIEWED_A_AT if reviewer_id == "reviewer-a" else REVIEWED_B_AT
    )
    rows: list[dict[str, object]] = []
    for (_, card), label in zip(packet.iterrows(), labels, strict=True):
        rows.append(
            {
                "event_review_id": (
                    f"event-review:{reviewer_id}:{card['packet_event_id']}"
                ),
                "packet_id": card["packet_id"],
                "packet_version": card["packet_version"],
                "packet_event_id": card["packet_event_id"],
                "packet_payload_sha256": card["packet_payload_sha256"],
                "review_session_id": f"session:{reviewer_id}",
                "reviewer_id": reviewer_id,
                "rubric_version": RUBRIC_VERSION,
                "reviewed_at_utc": when,
                "independence_attested": True,
                "peer_review_blinded": True,
                "model_score_blinded": True,
                "prior_label_blinded": True,
                **_decision_for_label(label),
                "source_locator": card["source_locator"],
                "notes": None,
            }
        )
    return pd.DataFrame(rows, columns=list(EventReviewRecord.model_fields))


def _comparison_inputs(
    count: int = 3,
    *,
    labels_a: list[str] | None = None,
    labels_b: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    packet = _packet(count)
    primary = _reviews(packet, "reviewer-a", labels=labels_a)
    secondary = _reviews(packet, "reviewer-b", labels=labels_b)
    comparison = compare_event_reviews(
        primary,
        secondary,
        packet,
        _actors(),
        rubric_version=RUBRIC_VERSION,
        compared_at_utc=COMPARED_AT,
    )
    return packet, primary, secondary, comparison


def _adjudications(
    comparison: pd.DataFrame,
    *,
    method: str = "third_expert",
    final_labels: list[str] | None = None,
) -> pd.DataFrame:
    if final_labels is None:
        final_labels = ["V2"] * len(comparison)
    assert len(final_labels) == len(comparison)
    rows: list[dict[str, object]] = []
    for (_, compared), label in zip(comparison.iterrows(), final_labels, strict=True):
        review_decision = _decision_for_label(label)
        final_decision = {
            "final_event_scope": review_decision["event_scope"],
            "final_context_match": review_decision["context_match"],
            "final_testing_status": review_decision["testing_status"],
            "final_perturbation_confirmed": review_decision["perturbation_confirmed"],
            "final_independent_reagent_count": review_decision[
                "independent_reagent_count"
            ],
            "final_orthogonal_perturbation": review_decision["orthogonal_perturbation"],
            "final_appropriate_control": review_decision["appropriate_control"],
            "final_assay_adequate": review_decision["assay_adequate"],
            "final_outcome_relation": review_decision["outcome_relation"],
            "final_null_result_evaluable": review_decision["null_result_evaluable"],
            "final_rescue_or_causal_reversal": review_decision[
                "rescue_or_causal_reversal"
            ],
            "final_label_code": label,
            "final_confidence": review_decision["confidence"],
        }
        consensus = method == "two_reviewer_consensus"
        rows.append(
            {
                "adjudication_id": (
                    f"adjudication:{compared['packet_event_id']}:{method}"
                ),
                "comparison_id": compared["comparison_id"],
                "packet_id": compared["packet_id"],
                "packet_version": compared["packet_version"],
                "packet_event_id": compared["packet_event_id"],
                "packet_payload_sha256": compared["packet_payload_sha256"],
                "primary_review_id": compared["primary_review_id"],
                "secondary_review_id": compared["secondary_review_id"],
                "primary_reviewer_id": compared["primary_reviewer_id"],
                "secondary_reviewer_id": compared["secondary_reviewer_id"],
                "rubric_version": compared["rubric_version"],
                "resolution_method": method,
                "adjudicator_id": "reviewer-a" if consensus else "adjudicator-c",
                "co_signer_id": "reviewer-b" if consensus else None,
                "adjudicated_at_utc": ADJUDICATED_AT,
                "model_score_blinded": True,
                **final_decision,
                "final_source_locator": "Figure and Methods rechecked",
                "adjudication_reason_codes": ("matched_context|source_rechecked"),
                "notes": None,
            }
        )
    return pd.DataFrame(rows, columns=list(EventAdjudicationRecord.model_fields))


def _write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def _write_packet_stage(tmp_path: Path, *, count: int = 3) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    actors_path = tmp_path / "actors.tsv"
    evidence_path = tmp_path / "neutral_cards.tsv"
    packet_dir = tmp_path / "packet"
    _write_tsv(_actors(), actors_path)
    _write_tsv(_raw_cards(count), evidence_path)
    manifest = write_event_review_packet(
        evidence_path,
        actors_path,
        RUBRIC_PATH,
        packet_dir,
        randomization_seed="fixed-test-seed",
        access_isolation_attested=True,
        neutral_card_audit_attested=True,
    )
    packet = pd.read_csv(packet_dir / "event_evidence_packet.tsv", sep="\t")
    return {
        "actors_path": actors_path,
        "evidence_path": evidence_path,
        "packet_dir": packet_dir,
        "packet": packet,
        "manifest": manifest,
        "manifest_sha": _sha256(packet_dir / "event_packet_manifest.json"),
    }


def _write_comparison_stage(tmp_path: Path, *, count: int = 3) -> dict[str, object]:
    stage = _write_packet_stage(tmp_path, count=count)
    packet = stage["packet"]
    assert isinstance(packet, pd.DataFrame)
    primary = _reviews(packet, "reviewer-a")
    secondary = _reviews(packet, "reviewer-b")
    primary_path = tmp_path / "reviews-a.tsv"
    secondary_path = tmp_path / "reviews-b.tsv"
    comparison_dir = tmp_path / "comparison"
    _write_tsv(primary, primary_path)
    _write_tsv(secondary, secondary_path)
    manifest = write_event_review_comparison_bundle(
        stage["packet_dir"],
        primary_path,
        secondary_path,
        stage["actors_path"],
        comparison_dir,
        compared_at_utc=COMPARED_AT,
        expected_packet_manifest_sha256=str(stage["manifest_sha"]),
    )
    stage.update(
        {
            "primary": primary,
            "secondary": secondary,
            "primary_path": primary_path,
            "secondary_path": secondary_path,
            "comparison_dir": comparison_dir,
            "comparison_manifest": manifest,
            "comparison_manifest_sha": _sha256(
                comparison_dir / "event_review_comparison_manifest.json"
            ),
        }
    )
    return stage


def test_packet_is_deterministic_allowlisted_and_review_template_is_blank():
    evidence = _raw_cards(4)
    first, template = prepare_event_review_packet(
        evidence,
        _actors(),
        rubric_version=RUBRIC_VERSION,
        randomization_seed="reproducible-seed",
    )
    second, _ = prepare_event_review_packet(
        evidence.iloc[::-1].reset_index(drop=True),
        _actors(),
        rubric_version=RUBRIC_VERSION,
        randomization_seed="reproducible-seed",
    )

    pd.testing.assert_frame_equal(first, second)
    assert first["packet_payload_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert {
        "model_score",
        "predicted_probability",
        "label_code",
        "comparison_status",
        "benchmark_ready",
    }.isdisjoint(first.columns)
    assert template["proposed_label_code"].eq("").all()
    assert template["reviewer_id"].eq("").all()
    assert "benchmark_ready" not in template


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("MoDeL-Score", 0.99),
        ("PREDICTED probability", 0.99),
        ("EvidenceRank", 1),
        ("Curator_A_Decision", "V2"),
    ],
)
def test_packet_rejects_casefolded_or_punctuated_leakage_columns(field, value):
    cards = _raw_cards(1).assign(**{field: value})
    with pytest.raises(ValueError, match="non-allowlisted|forbidden"):
        prepare_event_review_packet(
            cards,
            _actors(),
            rubric_version=RUBRIC_VERSION,
            randomization_seed="seed",
        )


def test_packet_rejects_nested_leakage_and_all_nonempty_notes():
    nested = _raw_cards(1)
    nested.loc[0, "source_locator"] = json.dumps({"reported": {"EvIdEnCe_RaNk": 1}})
    with pytest.raises(ValueError, match="forbidden nested key"):
        prepare_event_review_packet(
            nested,
            _actors(),
            rubric_version=RUBRIC_VERSION,
            randomization_seed="seed",
        )

    notes = _raw_cards(1).assign(notes="harmless-looking pipeline note")
    with pytest.raises(ValueError, match="require empty notes"):
        prepare_event_review_packet(
            notes,
            _actors(),
            rubric_version=RUBRIC_VERSION,
            randomization_seed="seed",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "reported_assay_endpoint",
            ["clonogenic survival", {"assigned_label": "V2"}],
        ),
        (
            "reported_assay_endpoint",
            "robust concordant response; grade V3",
        ),
        ("reported_assay_endpoint", "class=F0"),
        ("reported_assay_endpoint", "rated as V2"),
        ("reported_assay_endpoint", "rating: D"),
        ("source_locator", "Reviewer A assigned label V2 after review"),
        (
            "validation_cell_model",
            {"AI": {"probability": 0.98}},
        ),
    ],
)
def test_packet_recursively_rejects_leakage_in_every_card_field(field, value):
    cards = _raw_cards(1)
    cards[field] = cards[field].astype(object)
    cards.at[0, field] = value

    with pytest.raises(ValueError, match="forbidden nested key|leakage"):
        prepare_event_review_packet(
            cards,
            _actors(),
            rubric_version=RUBRIC_VERSION,
            randomization_seed="seed",
        )


@pytest.mark.parametrize(
    ("actor_update", "error"),
    [
        ({"actor_type": "ai_system"}, "verified human"),
        ({"identity_verified": False}, "identity_verified=true"),
        ({"persistent_id": "TEST:HUMAN:A"}, "reuse a persistent identity"),
    ],
)
def test_two_reviews_require_distinct_verified_humans(actor_update, error):
    packet = _packet(2)
    primary = _reviews(packet, "reviewer-a")
    secondary = _reviews(packet, "reviewer-b")
    actors = _actors()
    actors.loc[actors["actor_id"].eq("reviewer-b"), list(actor_update)] = list(
        actor_update.values()
    )

    with pytest.raises(ValueError, match=error):
        compare_event_reviews(
            primary,
            secondary,
            packet,
            actors,
            rubric_version=RUBRIC_VERSION,
            compared_at_utc=COMPARED_AT,
        )


@pytest.mark.parametrize("mutation", ["missing_event", "wrong_payload", "wrong_rubric"])
def test_review_streams_require_complete_same_packet_and_rubric_coverage(mutation):
    packet = _packet(3)
    primary = _reviews(packet, "reviewer-a")
    secondary = _reviews(packet, "reviewer-b")
    if mutation == "missing_event":
        secondary = secondary.iloc[:-1].copy()
    elif mutation == "wrong_payload":
        secondary.loc[0, "packet_payload_sha256"] = "f" * 64
    else:
        secondary.loc[0, "rubric_version"] = "different-rubric"

    with pytest.raises(ValueError, match="cover|wrong card payload|wrong rubric"):
        compare_event_reviews(
            primary,
            secondary,
            packet,
            _actors(),
            rubric_version=RUBRIC_VERSION,
            compared_at_utc=COMPARED_AT,
        )


def test_numeric_reagent_count_representation_does_not_create_disagreement():
    packet = _packet(2)
    primary = _reviews(packet, "reviewer-a", labels=["V2", "V2"])
    secondary = _reviews(packet, "reviewer-b", labels=["V2", "U"])
    assert primary["independent_reagent_count"].dtype.kind in "iu"
    assert secondary["independent_reagent_count"].dtype.kind == "f"

    comparison = compare_event_reviews(
        primary,
        secondary,
        packet,
        _actors(),
        rubric_version=RUBRIC_VERSION,
        compared_at_utc=COMPARED_AT,
    )

    agreed = comparison.loc[
        comparison["packet_event_id"].eq(packet.loc[0, "packet_event_id"])
    ].iloc[0]
    assert agreed["comparison_status"] == "full_agreement"
    assert pd.isna(agreed["disagreement_fields"])


def test_exact_agreement_is_only_pre_adjudication_and_cannot_release_a_label():
    packet, primary, secondary, comparison = _comparison_inputs(2)

    assert comparison["comparison_status"].eq("full_agreement").all()
    assert comparison["human_adjudication_required"].all()
    assert {
        "final_label_code",
        "label_code",
        "adjudication_status",
        "benchmark_ready",
    }.isdisjoint(comparison.columns)

    empty = pd.DataFrame(columns=list(EventAdjudicationRecord.model_fields))
    with pytest.raises(ValueError, match="failed contract validation|exactly cover"):
        finalize_event_adjudication(
            packet,
            comparison,
            primary,
            secondary,
            empty,
            _actors(),
            rubric_version=RUBRIC_VERSION,
            released_at_utc=RELEASED_AT,
        )


def test_third_expert_and_two_reviewer_consensus_have_distinct_identity_rules():
    packet, primary, secondary, comparison = _comparison_inputs(1)

    third_party = _adjudications(comparison)
    _, third_party_events = finalize_event_adjudication(
        packet,
        comparison,
        primary,
        secondary,
        third_party,
        _actors(),
        rubric_version=RUBRIC_VERSION,
        released_at_utc=RELEASED_AT,
    )
    assert third_party_events.loc[0, "curator"] == "adjudicator-c"
    assert third_party_events.loc[0, "adjudication_status"] == "adjudicated"

    not_independent = third_party.copy()
    not_independent.loc[0, "adjudicator_id"] = "reviewer-a"
    with pytest.raises(
        ValueError, match="differ from both reviewers|independent human"
    ):
        finalize_event_adjudication(
            packet,
            comparison,
            primary,
            secondary,
            not_independent,
            _actors(),
            rubric_version=RUBRIC_VERSION,
            released_at_utc=RELEASED_AT,
        )

    consensus = _adjudications(comparison, method="two_reviewer_consensus")
    _, consensus_events = finalize_event_adjudication(
        packet,
        comparison,
        primary,
        secondary,
        consensus,
        _actors(),
        rubric_version=RUBRIC_VERSION,
        released_at_utc=RELEASED_AT,
    )
    assert consensus_events.loc[0, "adjudication_status"] == ("dual_curator_consensus")

    invalid_consensus = consensus.copy()
    invalid_consensus.loc[0, "co_signer_id"] = "adjudicator-c"
    with pytest.raises(ValueError, match="signed by both reviewers"):
        finalize_event_adjudication(
            packet,
            comparison,
            primary,
            secondary,
            invalid_consensus,
            _actors(),
            rubric_version=RUBRIC_VERSION,
            released_at_utc=RELEASED_AT,
        )


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        (
            {
                "final_label_code": "V3",
                "final_rescue_or_causal_reversal": "no",
            },
            "rescue",
        ),
        (
            {
                "final_label_code": "F0",
                "final_outcome_relation": "no_effect",
                "final_null_result_evaluable": "no",
            },
            "evaluable null",
        ),
        (
            {
                "final_label_code": "D",
                "final_outcome_relation": "concordant",
            },
            "opposite",
        ),
        (
            {
                "final_label_code": "F0",
                "final_outcome_relation": "no_effect",
                "final_null_result_evaluable": "yes",
                "final_perturbation_confirmed": "no",
            },
            "confirmed perturbation",
        ),
    ],
)
def test_final_adjudication_reuses_fail_closed_label_rules(updates, error):
    packet, primary, secondary, comparison = _comparison_inputs(1)
    adjudication = _adjudications(comparison)
    for field, value in updates.items():
        adjudication.loc[0, field] = value

    with pytest.raises(ValueError, match=error):
        finalize_event_adjudication(
            packet,
            comparison,
            primary,
            secondary,
            adjudication,
            _actors(),
            rubric_version=RUBRIC_VERSION,
            released_at_utc=RELEASED_AT,
        )


@pytest.mark.parametrize(
    ("validation_drug", "validation_cell_model"),
    [
        ("cisplatin", "MDA-MB-468"),
        ("olaparib", "HeLa"),
    ],
)
def test_primary_labels_reject_mismatched_exact_validation_context(
    validation_drug,
    validation_cell_model,
):
    raw = _raw_cards(1)
    raw.loc[0, "validation_drug"] = validation_drug
    raw.loc[0, "validation_cell_model"] = validation_cell_model
    packet, _ = prepare_event_review_packet(
        raw,
        _actors(),
        rubric_version=RUBRIC_VERSION,
        randomization_seed="context-mismatch",
    )
    primary = _reviews(packet, "reviewer-a")
    secondary = _reviews(packet, "reviewer-b")
    comparison = compare_event_reviews(
        primary,
        secondary,
        packet,
        _actors(),
        rubric_version=RUBRIC_VERSION,
        compared_at_utc=COMPARED_AT,
    )

    with pytest.raises(ValueError, match="exact context requires normalized equality"):
        finalize_event_adjudication(
            packet,
            comparison,
            primary,
            secondary,
            _adjudications(comparison),
            _actors(),
            rubric_version=RUBRIC_VERSION,
            released_at_utc=RELEASED_AT,
        )


def test_primary_exact_context_comparison_is_casefolded_and_trimmed():
    raw = _raw_cards(1)
    raw.loc[0, "validation_drug"] = " OLAPARIB "
    raw.loc[0, "validation_cell_model"] = " mda-mb-468 "
    packet, _ = prepare_event_review_packet(
        raw,
        _actors(),
        rubric_version=RUBRIC_VERSION,
        randomization_seed="normalized-context",
    )
    primary = _reviews(packet, "reviewer-a")
    secondary = _reviews(packet, "reviewer-b")
    comparison = compare_event_reviews(
        primary,
        secondary,
        packet,
        _actors(),
        rubric_version=RUBRIC_VERSION,
        compared_at_utc=COMPARED_AT,
    )

    _, events = finalize_event_adjudication(
        packet,
        comparison,
        primary,
        secondary,
        _adjudications(comparison),
        _actors(),
        rubric_version=RUBRIC_VERSION,
        released_at_utc=RELEASED_AT,
    )
    assert events.loc[0, "label_code"] == "V2"


def test_f0_adjudication_maps_evaluable_null_to_validation_event():
    packet, primary, secondary, comparison = _comparison_inputs(
        1,
        labels_a=["F0"],
        labels_b=["F0"],
    )

    _, events = finalize_event_adjudication(
        packet,
        comparison,
        primary,
        secondary,
        _adjudications(comparison, final_labels=["F0"]),
        _actors(),
        rubric_version=RUBRIC_VERSION,
        released_at_utc=RELEASED_AT,
    )

    assert events.loc[0, "label_code"] == "F0"
    assert bool(events.loc[0, "null_result_evaluable"]) is True


def test_finalize_rejects_incomplete_or_mutated_comparison_lineage():
    packet, primary, secondary, comparison = _comparison_inputs(2)

    incomplete_comparison = comparison.iloc[:1].copy()
    incomplete_adjudication = _adjudications(incomplete_comparison)
    with pytest.raises(ValueError, match="cover|derivation|lineage"):
        finalize_event_adjudication(
            packet,
            incomplete_comparison,
            primary,
            secondary,
            incomplete_adjudication,
            _actors(),
            rubric_version=RUBRIC_VERSION,
            released_at_utc=RELEASED_AT,
        )

    mutated_comparison = comparison.copy()
    mutated_comparison.loc[0, "packet_payload_sha256"] = "f" * 64
    mutated_adjudication = _adjudications(mutated_comparison)
    with pytest.raises(ValueError, match="payload|derivation|lineage"):
        finalize_event_adjudication(
            packet,
            mutated_comparison,
            primary,
            secondary,
            mutated_adjudication,
            _actors(),
            rubric_version=RUBRIC_VERSION,
            released_at_utc=RELEASED_AT,
        )


def test_agreement_statistics_are_pre_adjudication_only():
    packet = _packet(3)
    primary = _reviews(packet, "reviewer-a", labels=["V2", "F0", "V1"])
    secondary = _reviews(packet, "reviewer-b", labels=["V2", "D", "V1"])

    agreement = summarize_event_review_agreement(primary, secondary)

    assert agreement["record_count"] == 3
    assert agreement["exact_label_agreement_count"] == 2
    assert agreement["exact_label_agreement_fraction"] == pytest.approx(2 / 3)
    assert agreement["cohen_kappa"] == pytest.approx(4 / 7)
    assert agreement["gwet_ac1"] == pytest.approx(71 / 113)
    assert agreement["label_marginal_prevalence"]["V2"]["primary"] == (
        pytest.approx(1 / 3)
    )
    assert agreement["label_class_specific_agreement"]["V2"]["positive_agreement"] == 1
    assert agreement["testing_status_agreement"]["exact_agreement_fraction"] == 1
    assert agreement["testing_status_agreement"]["cohen_kappa"] is None
    assert agreement["testing_status_agreement"]["gwet_ac1"] == 1
    assert agreement["confusion_matrix_axes"] == {
        "rows": "primary_reviews.proposed_label_code",
        "columns": "secondary_reviews.proposed_label_code",
    }
    assert agreement["confusion_matrix"]["F0"]["D"] == 1
    assert agreement["calculation_boundary"] == "pre_adjudication_reviews_only"
    assert "final_label" not in json.dumps(agreement).casefold()


def test_empty_agreement_has_the_same_stable_schema_without_recursion():
    packet = _packet(1)
    primary = _reviews(packet, "reviewer-a").iloc[0:0].copy()
    secondary = _reviews(packet, "reviewer-b").iloc[0:0].copy()

    agreement = summarize_event_review_agreement(primary, secondary)

    assert agreement["record_count"] == 0
    assert agreement["exact_label_agreement_fraction"] is None
    assert agreement["label_marginal_prevalence"]["V3"]["pooled"] is None
    assert (
        agreement["label_class_specific_agreement"]["F0"][
            "positive_agreement_denominator"
        ]
        == 0
    )
    assert agreement["testing_status_agreement"]["record_count"] == 0
    assert (
        agreement["testing_status_agreement"]["confusion_matrix"]["tested"]["tested"]
        == 0
    )
    assert agreement["field_agreement"]["testing_status"]["agreement_fraction"] is None
    assert agreement["calculation_boundary"] == "pre_adjudication_reviews_only"


@pytest.mark.parametrize("duplicate_role", ["primary", "secondary"])
def test_agreement_statistics_reject_duplicate_composite_keys(duplicate_role):
    packet = _packet(2)
    primary = _reviews(packet, "reviewer-a")
    secondary = _reviews(packet, "reviewer-b")
    if duplicate_role == "primary":
        primary = pd.concat([primary, primary.iloc[[0]]], ignore_index=True)
    else:
        secondary = pd.concat([secondary, secondary.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate composite review keys"):
        summarize_event_review_agreement(primary, secondary)


def test_packet_writer_is_reproducible_refuses_overwrite_and_fails_atomically(tmp_path):
    actors_path = tmp_path / "actors.tsv"
    evidence_path = tmp_path / "cards.tsv"
    _write_tsv(_actors(), actors_path)
    _write_tsv(_raw_cards(3), evidence_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    unattested = tmp_path / "unattested"
    with pytest.raises(ValueError, match="access_isolation_attested"):
        write_event_review_packet(
            evidence_path,
            actors_path,
            RUBRIC_PATH,
            unattested,
            randomization_seed="same-seed",
            access_isolation_attested=False,
            neutral_card_audit_attested=True,
        )
    assert not unattested.exists()

    unaudited = tmp_path / "unaudited"
    with pytest.raises(ValueError, match="neutral_card_audit_attested"):
        write_event_review_packet(
            evidence_path,
            actors_path,
            RUBRIC_PATH,
            unaudited,
            randomization_seed="same-seed",
            access_isolation_attested=True,
            neutral_card_audit_attested=False,
        )
    assert not unaudited.exists()

    first_manifest = write_event_review_packet(
        evidence_path,
        actors_path,
        RUBRIC_PATH,
        first,
        randomization_seed="same-seed",
        access_isolation_attested=True,
        neutral_card_audit_attested=True,
    )
    second_manifest = write_event_review_packet(
        evidence_path,
        actors_path,
        RUBRIC_PATH,
        second,
        randomization_seed="same-seed",
        access_isolation_attested=True,
        neutral_card_audit_attested=True,
    )

    assert first_manifest == second_manifest
    assert first_manifest["status"] == "event_review_packet_prepared"
    assert first_manifest["access_isolation"] == {
        "attested": True,
        "evidence": "caller_attestation_only",
        "scope": (
            "reviewer workspace is isolated from joinable prior-review, "
            "outcome-aware, and model-output artifacts"
        ),
    }
    assert first_manifest["neutral_card_audit"] == {
        "attested": True,
        "evidence": "caller_attestation_only",
        "method": "manual_review_of_every_distributable_card",
        "limitation": ("automated leakage patterns do not prove cognitive neutrality"),
    }
    assert first_manifest["scientific_boundary"]["prior_review_artifacts_not_ingested"]
    assert first_manifest["scientific_boundary"][
        "external_source_and_declaration_bundle_required"
    ]
    assert "roster_selection" not in json.dumps(first_manifest)
    assert "signed" not in json.dumps(first_manifest).casefold()
    assert {path.name for path in first.iterdir()} == {
        path.name for path in second.iterdir()
    }
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        write_event_review_packet(
            evidence_path,
            actors_path,
            RUBRIC_PATH,
            first,
            randomization_seed="same-seed",
            access_isolation_attested=True,
            neutral_card_audit_attested=True,
        )

    bad_evidence = tmp_path / "bad-cards.tsv"
    _write_tsv(_raw_cards(1).assign(notes="must remain private"), bad_evidence)
    failed_output = tmp_path / "failed-output"
    with pytest.raises(ValueError, match="require empty notes"):
        write_event_review_packet(
            bad_evidence,
            actors_path,
            RUBRIC_PATH,
            failed_output,
            randomization_seed="same-seed",
            access_isolation_attested=True,
            neutral_card_audit_attested=True,
        )
    assert not failed_output.exists()
    assert not (tmp_path / ".failed-output.publish.lock").exists()
    assert not list(tmp_path.glob(".failed-output.work-*"))


def test_bundle_lineage_rejects_mutation_and_never_emits_benchmark_ready(tmp_path):
    stage = _write_comparison_stage(tmp_path / "valid", count=2)
    comparison_dir = stage["comparison_dir"]
    assert isinstance(comparison_dir, Path)
    comparison = pd.read_csv(comparison_dir / "event_review_comparison.tsv", sep="\t")
    adjudications_path = tmp_path / "adjudications.tsv"
    _write_tsv(_adjudications(comparison), adjudications_path)
    final_dir = tmp_path / "final"

    final_manifest = write_final_adjudication_bundle(
        comparison_dir,
        adjudications_path,
        final_dir,
        released_at_utc=RELEASED_AT,
        expected_comparison_manifest_sha256=str(stage["comparison_manifest_sha"]),
    )

    events = pd.read_csv(final_dir / "validation_events.tsv", sep="\t")
    assert "benchmark_ready" not in events
    assert "benchmark_ready" not in EventAdjudicationRecord.model_fields
    assert "benchmark_ready" not in json.dumps(final_manifest)
    assert final_manifest["scientific_boundary"] == {
        "released_records_are_validation_events_only": True,
        "screen_readiness_requires_separate_registry_gates": True,
        "no_automatic_benchmark_promotion": True,
    }
    with pytest.raises(FileExistsError, match="already exists"):
        write_final_adjudication_bundle(
            comparison_dir,
            adjudications_path,
            final_dir,
            released_at_utc=RELEASED_AT,
            expected_comparison_manifest_sha256=str(stage["comparison_manifest_sha"]),
        )

    tampered_root = tmp_path / "tampered"
    tampered_stage = _write_comparison_stage(tampered_root, count=2)
    tampered_comparison_dir = tampered_stage["comparison_dir"]
    assert isinstance(tampered_comparison_dir, Path)
    review_path = tampered_comparison_dir / "event_reviews_a.tsv"
    review_path.write_bytes(review_path.read_bytes() + b"\n")
    tampered_comparison = pd.read_csv(
        tampered_comparison_dir / "event_review_comparison.tsv", sep="\t"
    )
    tampered_adjudications = tmp_path / "tampered-adjudications.tsv"
    _write_tsv(_adjudications(tampered_comparison), tampered_adjudications)
    rejected_output = tmp_path / "rejected-final"
    with pytest.raises(ValueError, match="checksum mismatch"):
        write_final_adjudication_bundle(
            tampered_comparison_dir,
            tampered_adjudications,
            rejected_output,
            released_at_utc=RELEASED_AT,
            expected_comparison_manifest_sha256=str(
                tampered_stage["comparison_manifest_sha"]
            ),
        )
    assert not rejected_output.exists()


def test_packet_manifest_hash_is_required_before_review_comparison(tmp_path):
    stage = _write_packet_stage(tmp_path, count=2)
    packet = stage["packet"]
    assert isinstance(packet, pd.DataFrame)
    primary_path = tmp_path / "a.tsv"
    secondary_path = tmp_path / "b.tsv"
    _write_tsv(_reviews(packet, "reviewer-a"), primary_path)
    _write_tsv(_reviews(packet, "reviewer-b"), secondary_path)
    output_dir = tmp_path / "comparison"

    with pytest.raises(ValueError, match="manifest SHA-256 does not match"):
        write_event_review_comparison_bundle(
            stage["packet_dir"],
            primary_path,
            secondary_path,
            stage["actors_path"],
            output_dir,
            compared_at_utc=COMPARED_AT,
            expected_packet_manifest_sha256="0" * 64,
        )
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("argv", "expected_command"),
    [
        (
            [
                "prepare-event-review-packet",
                "--evidence-cards",
                "cards.tsv",
                "--actor-registry",
                "actors.tsv",
                "--rubric",
                "rubric.yaml",
                "--output-dir",
                "packet",
                "--randomization-seed",
                "secret-seed",
                "--access-isolation-attested",
                "--neutral-card-audit-attested",
            ],
            "prepare-event-review-packet",
        ),
        (
            [
                "compare-event-reviews",
                "--packet-dir",
                "packet",
                "--primary-reviews",
                "a.tsv",
                "--secondary-reviews",
                "b.tsv",
                "--actor-registry",
                "actors.tsv",
                "--output-dir",
                "comparison",
                "--compared-at-utc",
                "2026-09-04T11:00:00+00:00",
                "--expected-packet-manifest-sha256",
                "0" * 64,
            ],
            "compare-event-reviews",
        ),
        (
            [
                "finalize-event-adjudication",
                "--comparison-bundle-dir",
                "comparison",
                "--adjudications",
                "adjudications.tsv",
                "--output-dir",
                "final",
                "--released-at-utc",
                "2026-09-04T13:00:00+00:00",
                "--expected-comparison-manifest-sha256",
                "f" * 64,
            ],
            "finalize-event-adjudication",
        ),
    ],
)
def test_cli_exposes_gold_event_workflow(argv, expected_command):
    args = build_parser().parse_args(argv)

    assert args.command == expected_command
    if expected_command == "prepare-event-review-packet":
        assert args.access_isolation_attested is True
        assert args.neutral_card_audit_attested is True


def test_prepare_event_packet_cli_passes_access_isolation_attestation(
    monkeypatch,
    capsys,
):
    captured: dict[str, object] = {}

    def fake_writer(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"status": "event_review_packet_prepared"}

    monkeypatch.setattr(cli_module, "write_event_review_packet", fake_writer)
    args = build_parser().parse_args(
        [
            "prepare-event-review-packet",
            "--evidence-cards",
            "cards.tsv",
            "--actor-registry",
            "actors.tsv",
            "--rubric",
            "rubric.yaml",
            "--output-dir",
            "packet",
            "--randomization-seed",
            "secret-seed",
            "--access-isolation-attested",
            "--neutral-card-audit-attested",
        ]
    )

    assert args.func(args) == 0
    assert captured["kwargs"] == {
        "randomization_seed": "secret-seed",
        "access_isolation_attested": True,
        "neutral_card_audit_attested": True,
    }
    assert json.loads(capsys.readouterr().out)["status"] == (
        "event_review_packet_prepared"
    )
