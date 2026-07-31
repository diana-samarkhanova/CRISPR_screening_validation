"""Validation-label ontology and model-target rules."""

from __future__ import annotations

from enum import StrEnum

import pandas as pd


class LabelCode(StrEnum):
    """Evidence labels for an individual validation event."""

    V3 = "V3"
    V2 = "V2"
    V1 = "V1"
    F0 = "F0"
    D = "D"
    A = "A"
    T = "T"
    U = "U"


POSITIVE_LABELS = frozenset({LabelCode.V2, LabelCode.V3})
NEGATIVE_LABELS = frozenset({LabelCode.F0, LabelCode.D})
UNLABELED_LABELS = frozenset({LabelCode.V1, LabelCode.A, LabelCode.T, LabelCode.U})


def model_target(label: str | LabelCode) -> int | None:
    """Map a label to the primary binary target.

    Returns ``None`` for supportive, ambiguous, technical-failure, and untested
    records. These records must not silently become negatives.
    """

    code = LabelCode(label)
    if code in POSITIVE_LABELS:
        return 1
    if code in NEGATIVE_LABELS:
        return 0
    return None


def resolve_event_labels(labels: list[str | LabelCode]) -> LabelCode:
    """Resolve repeated events conservatively for a model row.

    Conflicting verified positive and negative events resolve to ``A`` rather
    than letting an arbitrary precedence rule create a training label.
    """

    codes = {LabelCode(label) for label in labels}
    has_positive = bool(codes & POSITIVE_LABELS)
    has_negative = bool(codes & NEGATIVE_LABELS)
    if has_positive and has_negative:
        return LabelCode.A
    if LabelCode.V3 in codes:
        return LabelCode.V3
    if LabelCode.V2 in codes:
        return LabelCode.V2
    if LabelCode.D in codes:
        return LabelCode.D
    if LabelCode.F0 in codes:
        return LabelCode.F0
    if LabelCode.V1 in codes:
        return LabelCode.V1
    if LabelCode.T in codes:
        return LabelCode.T
    if LabelCode.A in codes:
        return LabelCode.A
    return LabelCode.U


CANDIDATE_KEY = (
    "screen_id",
    "contrast_id",
    "gene_symbol",
    "phenotype_direction",
)


def adjudicate_validation_events(events: pd.DataFrame) -> pd.DataFrame:
    """Resolve linked event rows to one conservative candidate-level label."""

    required = {*CANDIDATE_KEY, "label_code"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"validation events are missing columns: {sorted(missing)}")
    linked = events.dropna(subset=list(CANDIDATE_KEY)).copy()
    if linked.empty:
        return pd.DataFrame(
            columns=[*CANDIDATE_KEY, "label_code", "validation_event_count"]
        )
    adjudicated = (
        linked.groupby(list(CANDIDATE_KEY), sort=False, dropna=False)["label_code"]
        .agg(
            label_code=lambda values: resolve_event_labels(values.tolist()).value,
            validation_event_count="size",
        )
        .reset_index()
    )
    return adjudicated
