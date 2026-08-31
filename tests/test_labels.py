import pandas as pd

from crispr_evidencerank.labels import (
    LabelCode,
    _resolve_released_validation_events,
    model_target,
    resolve_event_labels,
)


def test_primary_label_mapping():
    assert model_target("V3") == 1
    assert model_target("V2") == 1
    assert model_target("F0") == 0
    assert model_target("D") == 0
    assert model_target("U") is None
    assert model_target("T") is None


def test_every_label_has_an_explicit_model_role():
    expected = {
        "V3": 1,
        "V2": 1,
        "V1": None,
        "F0": 0,
        "D": 0,
        "A": None,
        "T": None,
        "U": None,
    }
    assert {label: model_target(label) for label in expected} == expected


def test_conflicting_events_become_ambiguous():
    assert resolve_event_labels(["V2", "F0"]) == LabelCode.A
    assert resolve_event_labels(["V2", "V3"]) == LabelCode.V3


def test_empty_or_unlabeled_events_resolve_conservatively():
    assert resolve_event_labels([]) == LabelCode.U
    assert resolve_event_labels(["T", "A"]) == LabelCode.T


def test_single_curator_event_is_not_released_as_candidate_label():
    events = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "screen_id": "SC1",
                "contrast_id": "C1",
                "gene_symbol": "GENE1",
                "phenotype_direction": "resistance",
                "label_code": "V2",
                "adjudication_decision_id": "DEC1",
                "adjudication_status": "single_curator",
            }
        ]
    )
    decisions = pd.DataFrame(
        [
            {
                "decision_id": "DEC1",
                "disposition": "release_validation_event",
                "validation_event_id": "E1",
            }
        ]
    )

    adjudicated = _resolve_released_validation_events(events, decisions)

    assert adjudicated.empty
