"""Deterministic, auditable eligibility triage for BioGRID ORCS screens."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TextIO

import pandas as pd

from .contracts import (
    INTAKE_POLICY_RULE_IDS,
    AssessmentStage,
    EligibilityCheckRecord,
    EligibilityOutcome,
    IntakeStatus,
    ScreenIntakeRecord,
    StrictRecord,
)
from .orcs import (
    OrcsIndexParseResult,
    orcs_screen_id,
    parse_orcs_index,
)

SUPPORTED_POLICY_VERSIONS = frozenset(INTAKE_POLICY_RULE_IDS)


@dataclass(frozen=True)
class OrcsIntakeResult:
    """ORCS parse result plus screen- and rule-level intake assessments."""

    parsed: OrcsIndexParseResult
    screen_intake: pd.DataFrame
    eligibility_checks: pd.DataFrame
    candidate_screen_ids: tuple[str, ...]
    summary: dict[str, object]


@dataclass(frozen=True)
class IntakeDecision:
    """Status fields derived from auditable eligibility checks."""

    status: IntakeStatus
    candidate_for_full_curation: bool
    benchmark_ready: bool
    reason_codes: str | None


def _text(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text and text != "-" else None


def _normalized(value: object) -> str | None:
    text = _text(value)
    if text is None:
        return None
    return " ".join(text.casefold().replace("_", " ").split())


def _record_frame(
    records: list[dict[str, object]], model: type[StrictRecord]
) -> pd.DataFrame:
    parsed = [
        model.model_validate(record).model_dump(mode="json") for record in records
    ]
    return pd.DataFrame.from_records(parsed, columns=list(model.model_fields))


def derive_intake_decision(
    checks: (pd.DataFrame | Iterable[EligibilityCheckRecord | Mapping[str, object]]),
    *,
    assessment_stage: AssessmentStage | str,
    policy_version: int = 2,
) -> IntakeDecision:
    """Derive intake status from checks instead of accepting a declaration.

    A screen is excluded only by an explicit failure of a scope rule. At the
    curated stage every benchmark-required check must pass before the screen
    can become ``benchmark_ready``. Index metadata can never establish that
    status, even if all checks present in an index happen to pass.
    """

    if policy_version not in INTAKE_POLICY_RULE_IDS:
        supported = ", ".join(
            str(version) for version in sorted(SUPPORTED_POLICY_VERSIONS)
        )
        raise ValueError(
            f"unsupported intake policy_version {policy_version}; "
            f"supported versions: {supported}"
        )
    stage = AssessmentStage(assessment_stage)
    scope_rule_ids, benchmark_rule_ids = INTAKE_POLICY_RULE_IDS[policy_version]
    raw_checks = (
        checks.to_dict(orient="records")
        if isinstance(checks, pd.DataFrame)
        else list(checks)
    )
    validated = [
        (
            check
            if isinstance(check, EligibilityCheckRecord)
            else EligibilityCheckRecord.model_validate(check)
        )
        for check in raw_checks
    ]
    if not validated:
        raise ValueError("at least one eligibility check is required")
    mismatched = {
        check.assessment_stage for check in validated if check.assessment_stage != stage
    }
    if mismatched:
        raise ValueError(
            "eligibility check assessment_stage does not match reducer stage"
        )
    rule_counts = Counter(check.rule_id for check in validated)
    duplicate_rule_ids = sorted(
        rule_id for rule_id, count in rule_counts.items() if count > 1
    )
    if duplicate_rule_ids:
        raise ValueError(
            f"duplicate eligibility checks for policy rule(s): {duplicate_rule_ids}"
        )
    checks_by_rule = {check.rule_id: check for check in validated}

    scope_failures = list(
        dict.fromkeys(
            rule_id
            for rule_id in sorted(scope_rule_ids)
            if (
                rule_id in checks_by_rule
                and checks_by_rule[rule_id].outcome == EligibilityOutcome.FAIL
            )
        )
    )
    if scope_failures:
        return IntakeDecision(
            status=IntakeStatus.EXCLUDE,
            candidate_for_full_curation=False,
            benchmark_ready=False,
            reason_codes="|".join(scope_failures),
        )

    missing_rule_ids = sorted(benchmark_rule_ids - checks_by_rule.keys())
    unresolved = [
        rule_id
        for rule_id in sorted(benchmark_rule_ids)
        if (
            rule_id in checks_by_rule
            and checks_by_rule[rule_id].outcome != EligibilityOutcome.PASS
        )
    ]
    readiness_blockers = [
        *(f"missing:{rule_id}" for rule_id in missing_rule_ids),
        *unresolved,
    ]
    if stage == AssessmentStage.INDEX:
        return IntakeDecision(
            status=IntakeStatus.METADATA_ONLY,
            candidate_for_full_curation=True,
            benchmark_ready=False,
            reason_codes=(
                "|".join(readiness_blockers)
                if readiness_blockers
                else "full_text_curation_required"
            ),
        )
    if readiness_blockers:
        return IntakeDecision(
            status=IntakeStatus.METADATA_ONLY,
            candidate_for_full_curation=True,
            benchmark_ready=False,
            reason_codes="|".join(readiness_blockers),
        )
    return IntakeDecision(
        status=IntakeStatus.BENCHMARK_READY,
        candidate_for_full_curation=True,
        benchmark_ready=True,
        reason_codes=None,
    )


def _organism_check(value: object) -> tuple[EligibilityOutcome, str]:
    normalized = _normalized(value)
    if normalized is None:
        return EligibilityOutcome.UNKNOWN, "organism_unreported"
    if normalized in {"human", "homo sapiens", "9606"}:
        return EligibilityOutcome.PASS, "human_scope_confirmed"
    return EligibilityOutcome.FAIL, "non_human_organism"


def _modality_check(
    library_type: object, methodology: object
) -> tuple[EligibilityOutcome, str, str | None]:
    values = [
        value
        for value in (_normalized(library_type), _normalized(methodology))
        if value
    ]
    combined = " | ".join(values) if values else None
    searchable = " ".join(values)
    if "knockout" in searchable or "crisprn" in searchable or "crispr ko" in searchable:
        return EligibilityOutcome.PASS, "crispr_ko_confirmed", combined
    explicit_other = (
        "activation",
        "inhibition",
        "crispra",
        "crispri",
        "base editing",
        "prime editing",
    )
    if any(term in searchable for term in explicit_other):
        return EligibilityOutcome.FAIL, "non_ko_perturbation", combined
    return EligibilityOutcome.UNKNOWN, "perturbation_unresolved", combined


def _setup_check(value: object) -> tuple[EligibilityOutcome, str]:
    normalized = _normalized(value)
    if normalized is None:
        return EligibilityOutcome.UNKNOWN, "experimental_setup_unreported"
    if normalized == "drug exposure":
        return EligibilityOutcome.PASS, "drug_exposure_confirmed"
    if normalized == "toxin exposure":
        return (
            EligibilityOutcome.UNKNOWN,
            "toxin_or_drug_exposure_requires_curation",
        )
    explicit_non_drug = (
        "infection",
        "pathogen",
        "virus",
        "radiation",
        "hypoxia",
        "nutrient",
        "temperature",
        "genetic interaction",
    )
    if any(term in normalized for term in explicit_non_drug):
        return EligibilityOutcome.FAIL, "non_drug_primary_setup"
    return EligibilityOutcome.UNKNOWN, "drug_exposure_unresolved"


def _format_check(value: object) -> tuple[EligibilityOutcome, str]:
    normalized = _normalized(value)
    if normalized is None:
        return EligibilityOutcome.UNKNOWN, "screen_format_unreported"
    if normalized in {"pool", "pooled"}:
        return EligibilityOutcome.PASS, "pooled_screen_confirmed"
    if normalized in {"array", "arrayed"}:
        return EligibilityOutcome.FAIL, "arrayed_screen_out_of_scope"
    return EligibilityOutcome.UNKNOWN, "screen_format_unresolved"


def triage_orcs_index(
    source: str | Path | TextIO | pd.DataFrame,
    *,
    release: str,
    retrieved_date: date | str,
    organism_scope: str | None = None,
    policy_version: int = 2,
) -> OrcsIntakeResult:
    """Triage a release-pinned ORCS index without overclaiming readiness.

    Index metadata can establish explicit scope exclusions, but cannot prove
    comparator reconstruction, count-level availability, data rights, or
    orthogonal validation. Consequently, this stage emits only ``exclude`` or
    ``metadata_only``.
    """

    if policy_version not in SUPPORTED_POLICY_VERSIONS:
        supported = ", ".join(
            str(version) for version in sorted(SUPPORTED_POLICY_VERSIONS)
        )
        raise ValueError(
            f"unsupported intake policy_version {policy_version}; "
            f"supported versions: {supported}"
        )

    parsed = parse_orcs_index(
        source,
        release=release,
        retrieved_date=retrieved_date,
        organism_scope=organism_scope,
    )
    assessed_date = (
        retrieved_date
        if isinstance(retrieved_date, date)
        else date.fromisoformat(retrieved_date)
    )
    intake_records: list[dict[str, object]] = []
    check_records: list[dict[str, object]] = []
    candidate_external_ids: list[str] = []
    failed_scope_rules: Counter[str] = Counter()

    for row in parsed.normalized_index.to_dict(orient="records"):
        external_screen_id = _text(row.get("screen_id"))
        if external_screen_id is None:
            raise ValueError("normalized ORCS index contains a missing screen_id")
        screen_id = orcs_screen_id(release, external_screen_id)
        intake_id = f"{screen_id}:intake:index:v{policy_version}"
        source_id = _text(row.get("source_id"))
        source_locator = (
            f"BioGRID ORCS {release} screen index; SCREEN_ID={external_screen_id}"
        )
        external_dataset_id = _text(row.get("external_dataset_id"))

        rules: list[
            tuple[
                str,
                EligibilityOutcome,
                object,
                str | None,
                bool,
                bool,
                str,
                str | None,
            ]
        ] = []

        organism_value = _text(row.get("organism_official")) or organism_scope
        organism_outcome, organism_reason = _organism_check(organism_value)
        rules.append(
            (
                "scope.organism_human",
                organism_outcome,
                organism_value,
                _normalized(organism_value),
                True,
                True,
                organism_reason,
                None,
            )
        )

        modality_outcome, modality_reason, modality_value = _modality_check(
            row.get("library_type"), row.get("library_methodology")
        )
        rules.append(
            (
                "scope.perturbation_crispr_ko",
                modality_outcome,
                modality_value,
                modality_value,
                True,
                True,
                modality_reason,
                None,
            )
        )

        setup_value = _text(row.get("experimental_setup"))
        setup_outcome, setup_reason = _setup_check(setup_value)
        rules.append(
            (
                "scope.drug_exposure",
                setup_outcome,
                setup_value,
                _normalized(setup_value),
                True,
                True,
                setup_reason,
                None,
            )
        )

        format_value = _text(row.get("screen_format"))
        format_outcome, format_reason = _format_check(format_value)
        rules.append(
            (
                "scope.pooled_format",
                format_outcome,
                format_value,
                _normalized(format_value),
                True,
                True,
                format_reason,
                None,
            )
        )

        drug_name = _text(row.get("condition_name"))
        drug_identity_confirmed = (
            drug_name is not None and setup_outcome == EligibilityOutcome.PASS
        )
        rules.append(
            (
                "metadata.identifiable_drug",
                (
                    EligibilityOutcome.PASS
                    if drug_identity_confirmed
                    else EligibilityOutcome.UNKNOWN
                ),
                drug_name,
                drug_name if drug_identity_confirmed else None,
                False,
                True,
                (
                    "drug_name_confirmed_in_drug_exposure"
                    if drug_identity_confirmed
                    else (
                        "condition_name_requires_drug_exposure_curation"
                        if drug_name
                        else "drug_name_unreported"
                    )
                ),
                (
                    None
                    if drug_identity_confirmed or drug_name is None
                    else (
                        "CONDITION NAME may name a toxin, virus, medium, or "
                        "other exposure and is not sufficient drug identity."
                    )
                ),
            )
        )

        cell_line = _text(row.get("cell_line"))
        rules.append(
            (
                "metadata.identifiable_cell_context",
                (EligibilityOutcome.PASS if cell_line else EligibilityOutcome.UNKNOWN),
                cell_line,
                cell_line,
                False,
                True,
                ("cell_line_reported" if cell_line else "cell_context_unreported"),
                None,
            )
        )

        full_size_available = _normalized(row.get("full_size_available"))
        if full_size_available in {"yes", "true", "1"}:
            full_size_outcome = EligibilityOutcome.PASS
            full_size_reason = "full_orcs_score_set_available"
        elif full_size_available in {"no", "false", "0"}:
            full_size_outcome = EligibilityOutcome.UNKNOWN
            full_size_reason = "orcs_score_set_incomplete"
        else:
            full_size_outcome = EligibilityOutcome.UNKNOWN
            full_size_reason = "orcs_score_completeness_unreported"

        rules.extend(
            [
                (
                    "metadata.gene_level_mapping",
                    EligibilityOutcome.PASS,
                    "ORCS standardized per-screen gene scores",
                    "orcs_gene_scores",
                    False,
                    True,
                    "gene_level_mapping_available",
                    "ORCS author scores are not validation labels.",
                ),
                (
                    "metadata.full_gene_score_set",
                    full_size_outcome,
                    _text(row.get("full_size_available")),
                    full_size_available,
                    False,
                    False,
                    full_size_reason,
                    (
                        "An incomplete ORCS score set may be rescued by a "
                        "separate authorized count-level source."
                    ),
                ),
                (
                    "curation.comparator_and_sample_map",
                    EligibilityOutcome.UNKNOWN,
                    None,
                    None,
                    False,
                    True,
                    "comparator_sample_map_unresolved",
                    None,
                ),
                (
                    "data.count_level_signal",
                    EligibilityOutcome.UNKNOWN,
                    None,
                    None,
                    False,
                    True,
                    "count_level_data_unresolved",
                    None,
                ),
                (
                    "provenance.source_family",
                    EligibilityOutcome.UNKNOWN,
                    source_id,
                    source_id,
                    False,
                    True,
                    (
                        "source_family_provisional_from_source_id"
                        if source_id
                        else "source_family_unresolved_missing_source_id"
                    ),
                    (
                        (
                            "The ORCS SOURCE ID groups records from the same "
                            "declared source, but publication/preprint and "
                            "repository mirrors still require transitive "
                            "curation."
                        )
                        if source_id
                        else (
                            "SCREEN ID is release-specific and cannot serve "
                            "as a publication/source-family identifier."
                        )
                    ),
                ),
                (
                    "provenance.raw_data_family",
                    EligibilityOutcome.UNKNOWN,
                    None,
                    None,
                    False,
                    True,
                    "raw_data_family_unresolved",
                    None,
                ),
                (
                    "rights.source_and_raw_data",
                    EligibilityOutcome.UNKNOWN,
                    None,
                    None,
                    False,
                    True,
                    "source_raw_rights_unresolved",
                    (
                        "The ORCS download license does not automatically "
                        "establish rights for original count or raw files."
                    ),
                ),
                (
                    "labels.adjudicated_validation_event",
                    EligibilityOutcome.UNKNOWN,
                    None,
                    None,
                    False,
                    True,
                    "validation_events_unreviewed",
                    "ORCS HIT is never an orthogonal-validation label.",
                ),
            ]
        )

        screen_check_records: list[dict[str, object]] = []
        for (
            rule_id,
            outcome,
            observed,
            normalized,
            required_for_scope,
            required_for_benchmark,
            reason_code,
            notes,
        ) in rules:
            check_record = {
                "check_id": f"{intake_id}:check:{rule_id}",
                "intake_id": intake_id,
                "screen_id": screen_id,
                "assessment_stage": AssessmentStage.INDEX,
                "rule_id": rule_id,
                "outcome": outcome,
                "observed_value": _text(observed),
                "normalized_value": _text(normalized),
                "required_for_scope": required_for_scope,
                "required_for_benchmark": required_for_benchmark,
                "reason_code": reason_code,
                "source_locator": source_locator,
                "notes": notes,
            }
            screen_check_records.append(check_record)
            check_records.append(check_record)

        decision = derive_intake_decision(
            screen_check_records,
            assessment_stage=AssessmentStage.INDEX,
            policy_version=policy_version,
        )
        explicit_scope_failures = [
            rule_id
            for (
                rule_id,
                outcome,
                _observed,
                _normalized_value,
                required_for_scope,
                _required_for_benchmark,
                _reason_code,
                _notes,
            ) in rules
            if required_for_scope and outcome == EligibilityOutcome.FAIL
        ]
        failed_scope_rules.update(explicit_scope_failures)
        if decision.candidate_for_full_curation:
            candidate_external_ids.append(external_screen_id)

        intake_records.append(
            {
                "intake_id": intake_id,
                "screen_id": screen_id,
                "external_screen_id": external_screen_id,
                "external_dataset_id": external_dataset_id,
                "source_name": "BioGRID ORCS",
                "source_version": release,
                "policy_version": policy_version,
                "assessment_stage": AssessmentStage.INDEX,
                "status": decision.status,
                "candidate_for_full_curation": (decision.candidate_for_full_curation),
                "benchmark_ready": decision.benchmark_ready,
                "reason_codes": decision.reason_codes,
                "assessed_date": assessed_date,
                "notes": ("Index-stage triage cannot establish benchmark readiness."),
            }
        )

    screen_intake = _record_frame(intake_records, ScreenIntakeRecord)
    eligibility_checks = _record_frame(check_records, EligibilityCheckRecord)
    status_counts = {
        str(status): int(count)
        for status, count in screen_intake["status"].value_counts().items()
    }
    summary: dict[str, object] = {
        "release": release,
        "policy_version": policy_version,
        "assessment_stage": AssessmentStage.INDEX.value,
        "total_screens": int(len(screen_intake)),
        "status_counts": status_counts,
        "candidate_screen_count": int(len(candidate_external_ids)),
        "benchmark_ready_count": int(screen_intake["benchmark_ready"].sum()),
        "failed_scope_rule_counts": dict(sorted(failed_scope_rules.items())),
    }
    return OrcsIntakeResult(
        parsed=parsed,
        screen_intake=screen_intake,
        eligibility_checks=eligibility_checks,
        candidate_screen_ids=tuple(candidate_external_ids),
        summary=summary,
    )
