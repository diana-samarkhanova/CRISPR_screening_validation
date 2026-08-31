"""Fail-closed clinical trial context for treatment-by-cancer queries.

The module is intentionally report-only.  It summarizes the registry records
observed in a checksum-bound evidence snapshot; it does not rank genes, infer
efficacy, create validation labels, or make treatment recommendations.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import pandas as pd

from .contracts import (
    ClinicalInterventionRole,
    ClinicalMappingRelation,
    ClinicalMappingReviewStatus,
    ClinicalPhaseCategory,
    ClinicalRegimenContext,
    ClinicalStatusCategory,
    ClinicalStudyType,
    ClinicalTrialEvidenceRecord,
    DataAssetRecord,
    validate_records,
)


@dataclass(frozen=True)
class ClinicalContextResult:
    """Auditable outputs for one exact treatment-by-cancer query."""

    summary: pd.DataFrame
    studies: pd.DataFrame
    exclusions: pd.DataFrame
    used_assets: pd.DataFrame
    metadata: dict[str, object]


_DATE_COLUMNS = (
    "source_snapshot_date",
    "record_last_update_date",
    "results_first_posted_date",
    "available_date",
    "transformation_available_date",
    "retrieved_date",
)

_AVAILABILITY_DATE_COLUMNS = (
    "source_snapshot_date",
    "record_last_update_date",
    "results_first_posted_date",
    "available_date",
    "transformation_available_date",
)

_FAMILY_SUMMARY_FIELDS = (
    "study_type",
    "status_category",
    "phase_category",
    "intervention_role",
    "regimen_context",
    "results_posted",
)


def _validated_table(
    frame: pd.DataFrame,
    model: type[ClinicalTrialEvidenceRecord] | type[DataAssetRecord],
    *,
    label: str,
) -> pd.DataFrame:
    valid, errors = validate_records(frame, model)
    if not errors.empty:
        details = "; ".join(
            f"row {row.row_number}: {row.error}"
            for row in errors.itertuples(index=False)
        )
        raise ValueError(f"{label} validation failed: {details}")
    return valid


def _normalize_nonempty(value: object, *, label: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{label} cannot be blank")
    return normalized


def _normalize_excluded_families(values: Iterable[str]) -> set[str]:
    normalized = {
        _normalize_nonempty(value, label="excluded source family") for value in values
    }
    return normalized


def _validate_evidence_identity(evidence: pd.DataFrame) -> None:
    duplicate_key = [
        "source_name",
        "source_study_id",
        "treatment_concept_id",
        "cancer_concept_id",
    ]
    duplicated = evidence.duplicated(duplicate_key, keep=False)
    if duplicated.any():
        examples = (
            evidence.loc[duplicated, duplicate_key]
            .drop_duplicates()
            .sort_values(duplicate_key, kind="stable")
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            "clinical evidence contains duplicate normalized "
            "source-study-treatment-cancer records: "
            f"{examples}"
        )

    identity_columns = ["source_name", "source_study_id"]
    for identity, group in evidence.groupby(
        identity_columns,
        sort=True,
        dropna=False,
    ):
        if group["source_family_id"].nunique(dropna=False) != 1:
            raise ValueError(
                f"clinical study {identity!r} maps to multiple source families"
            )


def _validate_and_resolve_assets(
    evidence: pd.DataFrame,
    assets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    valid_assets = _validated_table(assets, DataAssetRecord, label="data asset")
    asset_index = valid_assets.set_index("asset_id")
    if not asset_index.index.is_unique:
        # validate_records already rejects duplicate primary keys.  Keep the
        # explicit assertion here so a future adapter cannot silently change
        # the many-to-one evidence-to-asset relationship.
        raise ValueError("data asset identifiers must be unique")
    referenced_ids = set(evidence["source_asset_id"].astype(str))
    missing_ids = sorted(referenced_ids - set(asset_index.index.astype(str)))
    if missing_ids:
        raise ValueError(
            f"clinical evidence references unknown source assets: {missing_ids}"
        )

    expected_sha = evidence["source_asset_id"].map(asset_index["sha256"])
    missing_sha = sorted(
        set(evidence.loc[expected_sha.isna(), "source_asset_id"].astype(str))
    )
    if missing_sha:
        raise ValueError(
            f"referenced clinical source assets require a checksum: {missing_sha}"
        )
    checksum_mismatch = (
        evidence["source_asset_sha256"].astype(str).ne(expected_sha.astype(str))
    )
    if checksum_mismatch.any():
        bad = sorted(
            evidence.loc[checksum_mismatch, "evidence_id"].astype(str).tolist()
        )
        raise ValueError(
            "clinical evidence source_asset_sha256 does not match DataAssetRecord "
            f"for evidence rows: {bad}"
        )

    for evidence_column, asset_column in (
        ("source_name", "source_name"),
        ("source_version", "source_version"),
    ):
        expected = evidence["source_asset_id"].map(asset_index[asset_column])
        mismatch = evidence[evidence_column].astype(str).ne(expected.astype(str))
        if mismatch.any():
            bad = sorted(evidence.loc[mismatch, "evidence_id"].astype(str).tolist())
            raise ValueError(
                f"clinical evidence {evidence_column} does not match its "
                f"DataAssetRecord for evidence rows: {bad}"
            )

    used_assets = (
        valid_assets.loc[valid_assets["asset_id"].astype(str).isin(referenced_ids)]
        .sort_values("asset_id", kind="stable")
        .reset_index(drop=True)
    )
    asset_available = evidence["source_asset_id"].map(asset_index["available_date"])
    return used_assets, asset_available


def _effective_available_date(
    evidence: pd.DataFrame,
    asset_available: pd.Series,
) -> pd.Series:
    components: dict[str, pd.Series] = {}
    for column in _AVAILABILITY_DATE_COLUMNS:
        components[column] = pd.to_datetime(evidence[column], errors="raise")
    components["source_asset_available_date"] = pd.to_datetime(
        asset_available,
        errors="coerce",
    )
    return pd.DataFrame(components, index=evidence.index).max(axis=1)


def _validate_family_summary_identity(eligible: pd.DataFrame) -> None:
    for family_id, group in eligible.groupby(
        "source_family_id",
        sort=True,
        dropna=False,
    ):
        conflicts = [
            column
            for column in _FAMILY_SUMMARY_FIELDS
            if group[column].nunique(dropna=False) > 1
        ]
        if conflicts:
            raise ValueError(
                "clinical source family has conflicting summary fields: "
                f"source_family_id={family_id!r}, fields={conflicts}"
            )


def _family_summary(eligible: pd.DataFrame) -> pd.DataFrame:
    if eligible.empty:
        return pd.DataFrame(columns=["source_family_id", *_FAMILY_SUMMARY_FIELDS])
    _validate_family_summary_identity(eligible)
    return (
        eligible.sort_values(
            ["source_family_id", "source_name", "source_study_id", "evidence_id"],
            kind="stable",
        )
        .drop_duplicates("source_family_id", keep="first")
        .reset_index(drop=True)
    )


def _summary_frame(
    eligible: pd.DataFrame,
    families: pd.DataFrame,
    *,
    treatment_concept_id: str,
    treatment_mapping_source: str,
    treatment_mapping_version: str,
    cancer_concept_id: str,
    cancer_mapping_source: str,
    cancer_mapping_version: str,
) -> pd.DataFrame:
    record: dict[str, object] = {
        "treatment_concept_id": treatment_concept_id,
        "treatment_mapping_source": treatment_mapping_source,
        "treatment_mapping_version": treatment_mapping_version,
        "cancer_concept_id": cancer_concept_id,
        "cancer_mapping_source": cancer_mapping_source,
        "cancer_mapping_version": cancer_mapping_version,
        "report_only_clinical_context_available": not eligible.empty,
        "report_only_clinical_observed_record_n": len(eligible),
        "report_only_clinical_observed_study_n": (
            len(eligible[["source_name", "source_study_id"]].drop_duplicates())
            if not eligible.empty
            else 0
        ),
        "report_only_clinical_observed_source_family_n": len(families),
        "report_only_clinical_results_posted_family_n": (
            int(families["results_posted"].eq(True).sum())  # noqa: E712
            if not families.empty
            else 0
        ),
        "report_only_clinical_results_not_posted_family_n": (
            int(families["results_posted"].eq(False).sum())  # noqa: E712
            if not families.empty
            else 0
        ),
        "report_only_clinical_source_snapshot_date": (
            max(eligible["source_snapshot_date"]).isoformat()
            if not eligible.empty
            else None
        ),
        "report_only_clinical_interpretation": (
            "registry_presence_and_results_availability_only_not_efficacy"
        ),
        "report_only_clinical_absence_interpretation": (
            "not_observed_in_supplied_snapshot_not_proof_of_absence"
        ),
    }

    for status in ClinicalStatusCategory:
        record[f"report_only_clinical_status_{status.value}_family_n"] = (
            int(families["status_category"].eq(status.value).sum())
            if not families.empty
            else 0
        )
    for phase in ClinicalPhaseCategory:
        record[f"report_only_clinical_phase_category_{phase.value}_family_n"] = (
            int(families["phase_category"].eq(phase.value).sum())
            if not families.empty
            else 0
        )
    for role in ClinicalInterventionRole:
        record[f"report_only_clinical_intervention_role_{role.value}_family_n"] = (
            int(families["intervention_role"].eq(role.value).sum())
            if not families.empty
            else 0
        )
    for regimen in ClinicalRegimenContext:
        record[f"report_only_clinical_regimen_{regimen.value}_family_n"] = (
            int(families["regimen_context"].eq(regimen.value).sum())
            if not families.empty
            else 0
        )
    return pd.DataFrame([record])


def summarize_clinical_context(
    evidence: pd.DataFrame,
    assets: pd.DataFrame,
    *,
    treatment_concept_id: str,
    treatment_mapping_source: str,
    treatment_mapping_version: str,
    cancer_concept_id: str,
    cancer_mapping_source: str,
    cancer_mapping_version: str,
    cutoff_date: date,
    excluded_source_families: Iterable[str] = (),
) -> ClinicalContextResult:
    """Summarize checksum-bound trial registry context without efficacy claims.

    Matching pins exact treatment/cancer identifiers and their mapping sources
    and releases. Preferred names and raw source terms are retained for
    provenance but never act as implicit aliases. A zero-row result means only
    that no eligible row was observed in the supplied snapshot.
    """

    treatment_concept_id = _normalize_nonempty(
        treatment_concept_id,
        label="treatment_concept_id",
    )
    treatment_mapping_source = _normalize_nonempty(
        treatment_mapping_source,
        label="treatment_mapping_source",
    )
    treatment_mapping_version = _normalize_nonempty(
        treatment_mapping_version,
        label="treatment_mapping_version",
    )
    cancer_concept_id = _normalize_nonempty(
        cancer_concept_id,
        label="cancer_concept_id",
    )
    cancer_mapping_source = _normalize_nonempty(
        cancer_mapping_source,
        label="cancer_mapping_source",
    )
    cancer_mapping_version = _normalize_nonempty(
        cancer_mapping_version,
        label="cancer_mapping_version",
    )
    if not isinstance(cutoff_date, date):
        raise TypeError("cutoff_date must be a date")

    valid = _validated_table(
        evidence,
        ClinicalTrialEvidenceRecord,
        label="clinical evidence",
    )
    for column in _DATE_COLUMNS:
        valid[column] = pd.to_datetime(valid[column], errors="raise").dt.date
    _validate_evidence_identity(valid)

    used_assets, asset_available = _validate_and_resolve_assets(valid, assets)
    source_exclusions = _normalize_excluded_families(excluded_source_families)
    unknown_exclusions = source_exclusions - set(valid["source_family_id"].astype(str))
    if unknown_exclusions:
        raise ValueError(
            "declared source-family exclusions are absent from clinical evidence: "
            f"{sorted(unknown_exclusions)}"
        )

    reason = pd.Series(pd.NA, index=valid.index, dtype="string")
    effective_available = _effective_available_date(valid, asset_available)
    reason.loc[effective_available > pd.Timestamp(cutoff_date)] = "post_cutoff"
    reason.loc[reason.isna() & valid["source_family_id"].isin(source_exclusions)] = (
        "excluded_source_family"
    )
    reason.loc[
        reason.isna()
        & valid["treatment_concept_id"].astype(str).ne(treatment_concept_id)
    ] = "treatment_mismatch"
    reason.loc[
        reason.isna() & valid["cancer_concept_id"].astype(str).ne(cancer_concept_id)
    ] = "cancer_mismatch"
    reason.loc[
        reason.isna()
        & valid["treatment_mapping_source"].astype(str).ne(treatment_mapping_source)
    ] = "treatment_mapping_source_mismatch"
    reason.loc[
        reason.isna()
        & valid["treatment_mapping_version"].astype(str).ne(treatment_mapping_version)
    ] = "treatment_mapping_version_mismatch"
    reason.loc[
        reason.isna()
        & valid["cancer_mapping_source"].astype(str).ne(cancer_mapping_source)
    ] = "cancer_mapping_source_mismatch"
    reason.loc[
        reason.isna()
        & valid["cancer_mapping_version"].astype(str).ne(cancer_mapping_version)
    ] = "cancer_mapping_version_mismatch"
    reason.loc[
        reason.isna()
        & valid["treatment_mapping_relation"].ne(ClinicalMappingRelation.EXACT.value)
    ] = "treatment_mapping_not_exact"
    reason.loc[
        reason.isna()
        & valid["treatment_mapping_review_status"].ne(
            ClinicalMappingReviewStatus.CURATOR_REVIEWED.value
        )
    ] = "treatment_mapping_not_reviewed"
    reason.loc[
        reason.isna()
        & valid["cancer_mapping_relation"].ne(ClinicalMappingRelation.EXACT.value)
    ] = "cancer_mapping_not_exact"
    reason.loc[
        reason.isna()
        & valid["cancer_mapping_review_status"].ne(
            ClinicalMappingReviewStatus.CURATOR_REVIEWED.value
        )
    ] = "cancer_mapping_not_reviewed"
    reason.loc[
        reason.isna() & valid["study_type"].ne(ClinicalStudyType.INTERVENTIONAL.value)
    ] = "study_type_mismatch"
    reason.loc[
        reason.isna()
        & valid["intervention_role"].ne(ClinicalInterventionRole.EXPERIMENTAL.value)
    ] = "intervention_role_mismatch"

    exclusions = valid.loc[reason.notna()].copy()
    exclusions.insert(0, "exclusion_reason", reason.loc[reason.notna()].values)
    exclusions = (
        exclusions.drop(columns=["used_for_label"])
        .sort_values("evidence_id", kind="stable")
        .reset_index(drop=True)
    )
    eligible = valid.loc[reason.isna()].copy()
    studies = (
        eligible.drop(columns=["used_for_label"])
        .sort_values(
            ["source_family_id", "source_name", "source_study_id", "evidence_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    families = _family_summary(eligible)
    summary = _summary_frame(
        eligible,
        families,
        treatment_concept_id=treatment_concept_id,
        treatment_mapping_source=treatment_mapping_source,
        treatment_mapping_version=treatment_mapping_version,
        cancer_concept_id=cancer_concept_id,
        cancer_mapping_source=cancer_mapping_source,
        cancer_mapping_version=cancer_mapping_version,
    )
    metadata: dict[str, object] = {
        "method": "report_only_clinical_context",
        "matching_policy": (
            "exact_treatment_and_cancer_concept_ids_and_mapping_releases"
        ),
        "mapping_eligibility_policy": "curator_reviewed_exact_mappings_only",
        "intervention_eligibility_policy": "experimental_role_only",
        "cutoff_date": cutoff_date.isoformat(),
        "treatment_concept_id": treatment_concept_id,
        "treatment_mapping_source": treatment_mapping_source,
        "treatment_mapping_version": treatment_mapping_version,
        "cancer_concept_id": cancer_concept_id,
        "cancer_mapping_source": cancer_mapping_source,
        "cancer_mapping_version": cancer_mapping_version,
        "input_evidence_records": len(valid),
        "eligible_evidence_records": len(eligible),
        "observed_source_families": len(families),
        "input_asset_records": len(assets),
        "referenced_asset_records": len(used_assets),
        "excluded_source_families": sorted(source_exclusions),
        "exclusion_reason_counts": (
            {
                str(key): int(value)
                for key, value in exclusions["exclusion_reason"]
                .value_counts()
                .sort_index()
                .items()
            }
            if not exclusions.empty
            else {}
        ),
        "interpretation_boundary": (
            "registry presence and aggregate results availability only; not "
            "efficacy, validation, predictive-biomarker evidence, or a therapeutic "
            "recommendation"
        ),
        "absence_interpretation": (
            "zero eligible rows means not observed in the supplied checksum-bound "
            "snapshot, not proof that no trial exists"
        ),
    }
    return ClinicalContextResult(
        summary=summary,
        studies=studies,
        exclusions=exclusions,
        used_assets=used_assets,
        metadata=metadata,
    )
