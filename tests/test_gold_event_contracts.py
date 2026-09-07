from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pytest
from pydantic import ValidationError

from crispr_evidencerank.contracts import (
    AdjudicationResolutionMethod,
    CriterionAssessment,
    CurationActorRecord,
    CurationActorType,
    EventAdjudicationRecord,
    EventEvidenceCardRecord,
    EventReviewComparisonRecord,
    EventReviewComparisonStatus,
    EventReviewRecord,
    ReviewConfidence,
    ValidationContextMatch,
    ValidationEventScope,
    ValidationOutcomeRelation,
    validate_records,
)

UTC_TIME = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
SHA256 = "a" * 64


def _actor(**updates: object) -> dict[str, object]:
    return {
        "actor_id": "human-reviewer-a",
        "actor_type": "human",
        "display_name": "Reviewer A",
        "identity_verified": True,
        "affiliation": None,
        "persistent_id": None,
        "notes": None,
    } | updates


def _card(**updates: object) -> dict[str, object]:
    return {
        "packet_id": "gold-batch-001",
        "packet_version": 1,
        "packet_event_id": "gold-batch-001:event:001",
        "packet_payload_sha256": SHA256,
        "batch_id": "batch-001",
        "source_family_id": "source-family-001",
        "study_id": "study-001",
        "screen_id": "screen-001",
        "contrast_id": "contrast-001",
        "gene_symbol": "GENE1",
        "screen_predicted_direction": "resistance",
        "screen_drug": "DRUG1",
        "screen_cell_model": "CELL1",
        "validation_drug": "DRUG1",
        "validation_cell_model": "CELL1",
        "reported_perturbation_modality": "CRISPR_KO",
        "reported_assay_endpoint": "clonogenic survival",
        "source_url": "https://example.org/article",
        "source_locator": "Figure 2A-C",
        "prepared_at_utc": UTC_TIME,
        "prepared_by_actor_id": "ai-preparer",
        "notes": None,
    } | updates


def _review(label: str = "V2", **updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "event_review_id": "review-a-001",
        "packet_id": "gold-batch-001",
        "packet_version": 1,
        "packet_event_id": "gold-batch-001:event:001",
        "packet_payload_sha256": SHA256,
        "review_session_id": "session-a",
        "reviewer_id": "human-reviewer-a",
        "rubric_version": "gold-v1",
        "reviewed_at_utc": UTC_TIME,
        "independence_attested": True,
        "peer_review_blinded": True,
        "model_score_blinded": True,
        "prior_label_blinded": True,
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
        "source_locator": "Figure 2A-C",
        "reason_codes": "adequate_assay|matched_context",
        "notes": None,
    }
    if label == "V3":
        record["rescue_or_causal_reversal"] = "yes"
    elif label == "F0":
        record["outcome_relation"] = "no_effect"
        record["null_result_evaluable"] = "yes"
    elif label == "D":
        record["outcome_relation"] = "opposite"
    elif label == "T":
        record.update(
            {
                "perturbation_confirmed": "no",
                "independent_reagent_count": None,
                "orthogonal_perturbation": "not_applicable",
                "appropriate_control": "no",
                "assay_adequate": "no",
                "outcome_relation": "not_interpretable",
            }
        )
    elif label == "U":
        record.update(
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
                "rescue_or_causal_reversal": "not_reported",
            }
        )
    return record | updates


def _comparison(**updates: object) -> dict[str, object]:
    return {
        "comparison_id": "comparison-001",
        "packet_id": "gold-batch-001",
        "packet_version": 1,
        "packet_event_id": "gold-batch-001:event:001",
        "packet_payload_sha256": SHA256,
        "rubric_version": "gold-v1",
        "primary_review_id": "review-a-001",
        "secondary_review_id": "review-b-001",
        "primary_reviewer_id": "human-reviewer-a",
        "secondary_reviewer_id": "human-reviewer-b",
        "primary_label_code": "V2",
        "secondary_label_code": "V2",
        "label_agreement": True,
        "disagreement_fields": None,
        "comparison_status": "full_agreement",
        "human_adjudication_required": True,
        "compared_at_utc": UTC_TIME,
        "notes": None,
    } | updates


def _adjudication(**updates: object) -> dict[str, object]:
    return {
        "adjudication_id": "adjudication-001",
        "comparison_id": "comparison-001",
        "packet_id": "gold-batch-001",
        "packet_version": 1,
        "packet_event_id": "gold-batch-001:event:001",
        "packet_payload_sha256": SHA256,
        "primary_review_id": "review-a-001",
        "secondary_review_id": "review-b-001",
        "primary_reviewer_id": "human-reviewer-a",
        "secondary_reviewer_id": "human-reviewer-b",
        "rubric_version": "gold-v1",
        "resolution_method": "third_expert",
        "adjudicator_id": "human-adjudicator-c",
        "co_signer_id": None,
        "adjudicated_at_utc": UTC_TIME,
        "model_score_blinded": True,
        "final_event_scope": "qualifying_validation",
        "final_context_match": "exact",
        "final_testing_status": "tested",
        "final_perturbation_confirmed": "yes",
        "final_independent_reagent_count": 2,
        "final_orthogonal_perturbation": "no",
        "final_appropriate_control": "yes",
        "final_assay_adequate": "yes",
        "final_outcome_relation": "concordant",
        "final_null_result_evaluable": "not_applicable",
        "final_rescue_or_causal_reversal": "no",
        "final_label_code": "V2",
        "final_confidence": "high",
        "final_source_locator": "Figure 2A-C",
        "adjudication_reason_codes": "matched_context|reviewed_source",
        "notes": None,
    } | updates


def test_gold_event_enum_values_are_frozen():
    assert {item.value for item in CriterionAssessment} == {
        "yes",
        "no",
        "unclear",
        "not_reported",
        "not_applicable",
    }
    assert {item.value for item in ValidationEventScope} == {
        "qualifying_validation",
        "external_support",
        "not_validation",
        "unclear",
    }
    assert {item.value for item in ValidationContextMatch} == {
        "exact",
        "close",
        "external",
        "unclear",
    }
    assert {item.value for item in ValidationOutcomeRelation} == {
        "concordant",
        "opposite",
        "no_effect",
        "mixed",
        "not_interpretable",
        "not_reported",
    }
    assert {item.value for item in ReviewConfidence} == {
        "high",
        "moderate",
        "low",
    }
    assert {item.value for item in EventReviewComparisonStatus} == {
        "full_agreement",
        "label_agreement_fact_disagreement",
        "label_disagreement",
    }
    assert {item.value for item in AdjudicationResolutionMethod} == {
        "third_expert",
        "two_reviewer_consensus",
        "agreement_confirmed",
        "insufficient_evidence",
    }
    assert {item.value for item in CurationActorType} == {
        "human",
        "ai_system",
        "software",
        "human_ai_team",
    }


def test_valid_rows_for_all_five_gold_event_contracts():
    assert CurationActorRecord.model_validate(_actor()).actor_type == "human"
    assert EventEvidenceCardRecord.model_validate(_card()).packet_event_id.endswith(
        ":001"
    )
    assert EventReviewRecord.model_validate(_review()).proposed_label_code == "V2"
    assert (
        EventReviewComparisonRecord.model_validate(_comparison()).comparison_status
        == "full_agreement"
    )
    assert (
        EventAdjudicationRecord.model_validate(_adjudication()).final_label_code == "V2"
    )


def test_evidence_card_rejects_a_coordinator_or_ai_summary():
    with pytest.raises(ValidationError, match="neutral_evidence_summary"):
        EventEvidenceCardRecord.model_validate(
            _card(neutral_evidence_summary="This looks like a strong validation.")
        )


@pytest.mark.parametrize(
    ("model", "record"),
    [
        (CurationActorRecord, _actor(identity_verified="true")),
        (EventReviewRecord, _review(independence_attested="true")),
        (EventReviewRecord, _review(peer_review_blinded=1)),
        (EventReviewRecord, _review(model_score_blinded="True")),
        (EventReviewRecord, _review(prior_label_blinded="false")),
        (EventReviewComparisonRecord, _comparison(label_agreement="true")),
        (
            EventReviewComparisonRecord,
            _comparison(human_adjudication_required=1),
        ),
        (EventAdjudicationRecord, _adjudication(model_score_blinded="true")),
    ],
)
def test_gold_signatures_use_strict_booleans(model, record):
    with pytest.raises(ValidationError):
        model.model_validate(record)


@pytest.mark.parametrize(
    ("model", "record"),
    [
        (
            EventEvidenceCardRecord,
            _card(prepared_at_utc=UTC_TIME.replace(tzinfo=None)),
        ),
        (EventReviewRecord, _review(reviewed_at_utc=UTC_TIME.replace(tzinfo=None))),
        (
            EventReviewComparisonRecord,
            _comparison(compared_at_utc=UTC_TIME.replace(tzinfo=None)),
        ),
        (
            EventAdjudicationRecord,
            _adjudication(adjudicated_at_utc=UTC_TIME.replace(tzinfo=None)),
        ),
        (
            EventReviewRecord,
            _review(reviewed_at_utc=UTC_TIME.astimezone(timezone(timedelta(hours=5)))),
        ),
    ],
)
def test_gold_event_timestamps_must_be_explicit_utc(model, record):
    with pytest.raises(ValidationError, match="UTC timezone"):
        model.model_validate(record)


@pytest.mark.parametrize(
    "updates",
    [
        {"reason_codes": "z_reason|a_reason"},
        {"reason_codes": "same_reason|same_reason"},
        {"reason_codes": ""},
    ],
)
def test_review_reason_codes_are_nonempty_sorted_and_unique(updates):
    with pytest.raises(ValidationError):
        EventReviewRecord.model_validate(_review(**updates))


def test_adjudication_reason_codes_are_sorted_and_unique():
    with pytest.raises(ValidationError):
        EventAdjudicationRecord.model_validate(
            _adjudication(adjudication_reason_codes="z_reason|a_reason")
        )


def test_evidence_card_composite_identity_allows_multiple_events_per_packet():
    cards = pd.DataFrame(
        [
            _card(packet_event_id="gold-batch-001:event:001"),
            _card(packet_event_id="gold-batch-001:event:002", gene_symbol="GENE2"),
        ]
    )
    valid, errors = validate_records(cards, EventEvidenceCardRecord)
    assert len(valid) == 2
    assert errors.empty

    duplicated, duplicate_errors = validate_records(
        pd.DataFrame([_card(), _card(source_locator="Figure 3")]),
        EventEvidenceCardRecord,
    )
    assert duplicated.empty
    assert duplicate_errors["error"].str.contains("duplicate primary key").all()


@pytest.mark.parametrize(
    "model,record",
    [
        (EventEvidenceCardRecord, _card()),
        (EventReviewRecord, _review()),
        (EventReviewComparisonRecord, _comparison()),
        (EventAdjudicationRecord, _adjudication()),
    ],
)
def test_packet_event_id_is_required_across_event_contracts(model, record):
    record.pop("packet_event_id")
    with pytest.raises(ValidationError):
        model.model_validate(record)


@pytest.mark.parametrize("label", ["V2", "V3", "F0", "D", "T", "U"])
def test_valid_rubric_examples(label):
    assert EventReviewRecord.model_validate(_review(label)).proposed_label_code == label


@pytest.mark.parametrize(
    "label,updates,error",
    [
        ("V2", {"context_match": "close"}, "context_match=exact"),
        ("V2", {"event_scope": "external_support"}, "qualifying_validation"),
        (
            "V2",
            {"independent_reagent_count": 1},
            "at least two independent reagents",
        ),
        ("V2", {"appropriate_control": "unclear"}, "appropriate control"),
        ("V2", {"assay_adequate": "no"}, "adequate assay"),
        ("V2", {"outcome_relation": "no_effect"}, "concordant outcome"),
        ("V3", {"rescue_or_causal_reversal": "no"}, "rescue"),
        ("F0", {"outcome_relation": "concordant"}, "no_effect"),
        ("F0", {"null_result_evaluable": "no"}, "evaluable null"),
        ("D", {"outcome_relation": "concordant"}, "opposite"),
        ("T", {"outcome_relation": "mixed"}, "not_interpretable"),
        ("U", {"testing_status": "tested"}, "not_tested or unknown"),
    ],
)
def test_rubric_rejects_inconsistent_labels(label, updates, error):
    with pytest.raises(ValidationError, match=error):
        EventReviewRecord.model_validate(_review(label, **updates))


def test_orthogonal_perturbation_can_satisfy_primary_reagent_rule():
    review = EventReviewRecord.model_validate(
        _review(independent_reagent_count=1, orthogonal_perturbation="yes")
    )
    assert review.proposed_label_code == "V2"


@pytest.mark.parametrize("label", ["V2", "V3", "F0", "D"])
def test_primary_labels_need_a_primary_reagent_even_with_orthogonal_evidence(label):
    with pytest.raises(ValidationError, match="at least one independent primary"):
        EventReviewRecord.model_validate(
            _review(
                label,
                independent_reagent_count=0,
                orthogonal_perturbation="yes",
            )
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"perturbation_confirmed": "yes"},
        {"independent_reagent_count": 1},
        {"orthogonal_perturbation": "yes"},
        {"appropriate_control": "yes"},
        {"assay_adequate": "yes"},
        {"null_result_evaluable": "yes"},
        {"rescue_or_causal_reversal": "yes"},
        {"outcome_relation": "concordant"},
        {"outcome_relation": "opposite"},
        {"outcome_relation": "no_effect"},
        {"outcome_relation": "mixed"},
    ],
)
def test_u_cannot_carry_affirmative_or_completed_outcome_evidence(updates):
    with pytest.raises(ValidationError, match="affirmative or completed"):
        EventReviewRecord.model_validate(_review("U", **updates))


def test_v1_cannot_undergrade_an_event_meeting_v2_or_v3_criteria():
    with pytest.raises(ValidationError, match="satisfying V2 criteria"):
        EventReviewRecord.model_validate(_review("V1"))
    with pytest.raises(ValidationError, match="satisfying V3 criteria"):
        EventReviewRecord.model_validate(_review("V1", rescue_or_causal_reversal="yes"))


def test_ambiguous_label_cannot_undergrade_a_concordant_v2_event():
    with pytest.raises(ValidationError, match="A cannot undergrade"):
        EventReviewRecord.model_validate(_review("A"))
    ambiguous = EventReviewRecord.model_validate(_review("A", outcome_relation="mixed"))
    assert ambiguous.proposed_label_code == "A"


@pytest.mark.parametrize(
    "outcome_relation,null_result_evaluable,expected_label",
    [
        ("opposite", "not_applicable", "D"),
        ("no_effect", "yes", "F0"),
    ],
)
def test_ambiguous_label_cannot_suppress_a_clear_negative_event(
    outcome_relation, null_result_evaluable, expected_label
):
    with pytest.raises(ValidationError, match=rf"satisfying {expected_label} criteria"):
        EventReviewRecord.model_validate(
            _review(
                "A",
                outcome_relation=outcome_relation,
                null_result_evaluable=null_result_evaluable,
            )
        )


@pytest.mark.parametrize("event_scope", ["qualifying_validation", "external_support"])
def test_u_requires_not_validation_or_unclear_scope(event_scope):
    with pytest.raises(ValidationError, match="not_validation or unclear"):
        EventReviewRecord.model_validate(_review("U", event_scope=event_scope))


def test_close_or_external_context_cannot_create_a_primary_label():
    for context_match in ("close", "external"):
        with pytest.raises(ValidationError, match="context_match=exact"):
            EventReviewRecord.model_validate(
                _review("D", context_match=context_match, outcome_relation="opposite")
            )


def test_v1_requires_concordance_but_accepts_external_support():
    record = _review(
        "V1",
        event_scope="external_support",
        context_match="external",
        independent_reagent_count=1,
    )
    assert EventReviewRecord.model_validate(record).proposed_label_code == "V1"
    with pytest.raises(ValidationError, match="V1 requires a concordant outcome"):
        EventReviewRecord.model_validate(_review("V1", outcome_relation="mixed"))


def test_review_requires_all_four_true_attestations():
    for field in (
        "independence_attested",
        "peer_review_blinded",
        "model_score_blinded",
        "prior_label_blinded",
    ):
        with pytest.raises(ValidationError, match="requires true attestations"):
            EventReviewRecord.model_validate(_review(**{field: False}))


@pytest.mark.parametrize(
    "updates",
    [
        {"secondary_review_id": "review-a-001"},
        {"secondary_reviewer_id": "human-reviewer-a"},
        {"human_adjudication_required": False},
        {"label_agreement": False},
        {"comparison_status": "label_disagreement"},
    ],
)
def test_full_agreement_comparison_rejects_inconsistent_identity_or_state(updates):
    with pytest.raises(ValidationError):
        EventReviewComparisonRecord.model_validate(_comparison(**updates))


def test_comparison_distinguishes_fact_and_label_disagreements():
    fact_disagreement = EventReviewComparisonRecord.model_validate(
        _comparison(
            disagreement_fields="assay_adequate",
            comparison_status="label_agreement_fact_disagreement",
        )
    )
    assert fact_disagreement.label_agreement is True

    label_disagreement = EventReviewComparisonRecord.model_validate(
        _comparison(
            secondary_label_code="V1",
            label_agreement=False,
            disagreement_fields="proposed_label_code",
            comparison_status="label_disagreement",
        )
    )
    assert label_disagreement.label_agreement is False

    with pytest.raises(ValidationError, match="proposed_label_code"):
        EventReviewComparisonRecord.model_validate(
            _comparison(
                secondary_label_code="V1",
                label_agreement=False,
                disagreement_fields="assay_adequate",
                comparison_status="label_disagreement",
            )
        )


def test_third_expert_adjudication_requires_an_independent_human_id():
    assert (
        EventAdjudicationRecord.model_validate(_adjudication()).adjudicator_id
        == "human-adjudicator-c"
    )
    with pytest.raises(ValidationError, match="differ from both reviewers"):
        EventAdjudicationRecord.model_validate(
            _adjudication(adjudicator_id="human-reviewer-a")
        )
    with pytest.raises(ValidationError, match="may use a co-signer"):
        EventAdjudicationRecord.model_validate(
            _adjudication(co_signer_id="human-adjudicator-d")
        )


def test_two_reviewer_consensus_requires_both_reviewers_as_signers():
    consensus = EventAdjudicationRecord.model_validate(
        _adjudication(
            resolution_method="two_reviewer_consensus",
            adjudicator_id="human-reviewer-a",
            co_signer_id="human-reviewer-b",
        )
    )
    assert consensus.co_signer_id == "human-reviewer-b"

    with pytest.raises(ValidationError, match="requires a co-signer"):
        EventAdjudicationRecord.model_validate(
            _adjudication(
                resolution_method="two_reviewer_consensus",
                adjudicator_id="human-reviewer-a",
            )
        )
    with pytest.raises(ValidationError, match="signed by both reviewers"):
        EventAdjudicationRecord.model_validate(
            _adjudication(
                resolution_method="two_reviewer_consensus",
                adjudicator_id="human-reviewer-a",
                co_signer_id="human-adjudicator-c",
            )
        )


@pytest.mark.parametrize(
    "method,updates",
    [
        ("agreement_confirmed", {}),
        (
            "insufficient_evidence",
            {"final_label_code": "A", "final_outcome_relation": "mixed"},
        ),
    ],
)
def test_other_resolution_methods_also_require_independent_signoff(method, updates):
    with pytest.raises(ValidationError, match="differ from both reviewers"):
        EventAdjudicationRecord.model_validate(
            _adjudication(
                resolution_method=method,
                adjudicator_id="human-reviewer-b",
                **updates,
            )
        )


def test_adjudication_reuses_the_strict_label_rubric():
    with pytest.raises(ValidationError, match="evaluable null"):
        EventAdjudicationRecord.model_validate(
            _adjudication(
                final_label_code="F0",
                final_outcome_relation="no_effect",
                final_null_result_evaluable="no",
            )
        )


@pytest.mark.parametrize(
    "label,updates",
    [
        ("V2", {}),
        ("V3", {"final_rescue_or_causal_reversal": "yes"}),
        (
            "F0",
            {
                "final_outcome_relation": "no_effect",
                "final_null_result_evaluable": "yes",
            },
        ),
        ("D", {"final_outcome_relation": "opposite"}),
    ],
)
def test_insufficient_evidence_cannot_release_a_primary_label(label, updates):
    with pytest.raises(ValidationError, match="cannot release V2/V3/F0/D"):
        EventAdjudicationRecord.model_validate(
            _adjudication(
                resolution_method="insufficient_evidence",
                final_label_code=label,
                **updates,
            )
        )
