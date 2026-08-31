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


def _resolve_released_validation_events(
    events: pd.DataFrame,
    adjudication_decisions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Purely resolve prevalidated, release-verified events to candidate labels.

    This private helper is not a release verifier. Callers must first validate
    the full contracts and a checksum-pinned adjudication release manifest.
    """

    required = {
        *CANDIDATE_KEY,
        "event_id",
        "label_code",
        "adjudication_decision_id",
        "adjudication_status",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"validation events are missing columns: {sorted(missing)}")
    if adjudication_decisions is None:
        linked = events.iloc[0:0].copy()
    else:
        decision_required = {
            "decision_id",
            "disposition",
            "validation_event_id",
        }
        decision_missing = decision_required - set(adjudication_decisions.columns)
        if decision_missing:
            raise ValueError(
                "adjudication decisions are missing columns: "
                f"{sorted(decision_missing)}"
            )
        released = adjudication_decisions.loc[
            adjudication_decisions["disposition"]
            .astype(str)
            .eq("release_validation_event")
        ].copy()
        if released["validation_event_id"].isna().any():
            raise ValueError("released decisions require validation_event_id")
        if released["validation_event_id"].astype(str).duplicated().any():
            raise ValueError("one validation event cannot be released twice")
        decision_by_event = {
            str(row["validation_event_id"]): str(row["decision_id"])
            for _, row in released.iterrows()
        }
        linked = events.loc[
            events["event_id"].astype(str).isin(decision_by_event)
        ].copy()
        linked = linked.loc[
            linked.apply(
                lambda row: (
                    str(row["adjudication_decision_id"])
                    == decision_by_event.get(str(row["event_id"]))
                ),
                axis=1,
            )
        ]
        linked = linked.loc[
            linked["adjudication_status"].astype(str).eq("consensus_adjudicated")
        ]
    linked = linked.dropna(subset=list(CANDIDATE_KEY)).copy()
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
