from crispr_evidencerank.labels import LabelCode, model_target, resolve_event_labels


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
