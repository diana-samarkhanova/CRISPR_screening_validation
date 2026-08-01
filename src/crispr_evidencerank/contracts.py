"""Pydantic contracts for the normalized evidence registry."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from .labels import CANDIDATE_KEY, LabelCode, adjudicate_validation_events


class PerturbationModality(StrEnum):
    CRISPR_KO = "CRISPR_KO"
    CRISPRA = "CRISPRa"
    CRISPRI = "CRISPRi"
    RNAI = "RNAi"
    OVEREXPRESSION = "overexpression"
    MUTATION = "mutation"
    INHIBITOR = "inhibitor"
    OTHER = "other"


class PhenotypeDirection(StrEnum):
    RESISTANCE = "resistance"
    SENSITIZATION = "sensitization"
    NEUTRAL = "neutral"
    DISCORDANT = "discordant"
    UNKNOWN = "unknown"


class SampleRole(StrEnum):
    CONTROL = "control"
    TREATMENT = "treatment"
    BASELINE = "baseline"


class TestingStatus(StrEnum):
    TESTED = "tested"
    NOT_TESTED = "not_tested"
    UNKNOWN = "unknown"


class SourceRole(StrEnum):
    ORIGINAL_SCREEN = "original_screen"
    PROTOCOL = "protocol"
    REANALYSIS = "reanalysis"
    VALIDATION_ONLY = "validation_only"
    REVIEW = "review"
    OTHER = "other"


class ScreenScale(StrEnum):
    GENOME_WIDE = "genome_wide"
    TARGETED = "targeted"
    SATURATION = "saturation"
    OTHER = "other"
    UNKNOWN = "unknown"


class SelectionStrategy(StrEnum):
    POSITIVE = "positive_selection"
    NEGATIVE = "negative_selection"
    BIDIRECTIONAL = "bidirectional_selection"
    COMPETITIVE_GROWTH = "competitive_growth"
    MARKER_SORT = "marker_sort"
    OTHER = "other"
    UNKNOWN = "unknown"


class ControlType(StrEnum):
    VEHICLE = "vehicle"
    UNTREATED = "untreated"
    BASELINE_T0 = "baseline_t0"
    LATER_TIMEPOINT = "later_timepoint"
    SORTED_POPULATION = "sorted_population"
    MATCHED_NONTARGETING = "matched_nontargeting"
    OTHER = "other"
    UNKNOWN = "unknown"


class ExposureSchedule(StrEnum):
    CONTINUOUS = "continuous"
    PULSE = "pulse"
    INTERMITTENT = "intermittent"
    REPEATED_DOSE = "repeated_dose"
    OTHER = "other"
    UNKNOWN = "unknown"


class ReplicateUnit(StrEnum):
    INDEPENDENT_INFECTION = "independent_infection"
    BIOLOGICAL_CULTURE = "biological_culture"
    TECHNICAL_SPLIT = "technical_split"
    UNKNOWN = "unknown"


class IntakeStatus(StrEnum):
    BENCHMARK_READY = "benchmark_ready"
    METADATA_ONLY = "metadata_only"
    EXCLUDE = "exclude"


class CurationQueueBucket(StrEnum):
    CONFIRMED_SCOPE = "confirmed_scope"
    MANUAL_SCOPE_REVIEW = "manual_scope_review"


class AssessmentStage(StrEnum):
    INDEX = "index"
    CURATED = "curated"


class EligibilityOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class ReviewCompleteness(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNRESOLVED = "unresolved"
    NOT_APPLICABLE = "not_applicable"


class QuantitativeDataStatus(StrEnum):
    RAW_READS_PUBLIC = "raw_reads_public"
    RAW_COUNTS_PUBLIC = "raw_counts_public"
    DERIVED_COUNTS_PUBLIC = "derived_counts_public"
    AUTHOR_SCORES_PUBLIC = "author_scores_public"
    AVAILABLE_FROM_AUTHORS = "available_from_authors"
    NOT_FOUND = "not_found"
    UNRESOLVED = "unresolved"


class ValidationReviewStatus(StrEnum):
    CANDIDATE_V3 = "candidate_v3"
    CANDIDATE_V2 = "candidate_v2"
    CANDIDATE_V1 = "candidate_v1"
    NONQUALIFYING_ONLY = "nonqualifying_only"
    NONE_REPORTED = "none_reported"
    UNRESOLVED = "unresolved"


INTAKE_POLICY_V2_SCOPE_RULE_IDS = frozenset(
    {
        "scope.organism_human",
        "scope.perturbation_crispr_ko",
        "scope.drug_exposure",
        "scope.pooled_format",
    }
)

INTAKE_POLICY_V2_BENCHMARK_RULE_IDS = frozenset(
    {
        *INTAKE_POLICY_V2_SCOPE_RULE_IDS,
        "metadata.identifiable_drug",
        "metadata.identifiable_cell_context",
        "metadata.gene_level_mapping",
        "curation.comparator_and_sample_map",
        "data.count_level_signal",
        "provenance.source_family",
        "provenance.raw_data_family",
        "rights.source_and_raw_data",
        "labels.adjudicated_validation_event",
    }
)

INTAKE_POLICY_RULE_IDS: dict[int, tuple[frozenset[str], frozenset[str]]] = {
    2: (
        INTAKE_POLICY_V2_SCOPE_RULE_IDS,
        INTAKE_POLICY_V2_BENCHMARK_RULE_IDS,
    )
}


class StrictRecord(BaseModel):
    """Base model that rejects misspelled or unexpected columns."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )
    primary_key: ClassVar[tuple[str, ...]] = ()


class StudyRecord(StrictRecord):
    primary_key = ("study_id",)

    study_id: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    doi: str | None = None
    publication_date: date | None = None
    organism: str = "human"
    source_url: HttpUrl | None = None
    data_license: str | None = None
    source_role: SourceRole = SourceRole.ORIGINAL_SCREEN
    independent_screen_source: bool = True
    source_id: str | None = None
    source_type: str | None = None
    pubmed_id: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def protocol_is_not_an_independent_screen(self) -> StudyRecord:
        non_screen_roles = {
            SourceRole.PROTOCOL,
            SourceRole.REANALYSIS,
            SourceRole.REVIEW,
        }
        if self.source_role in non_screen_roles and self.independent_screen_source:
            raise ValueError(
                "protocol, reanalysis, and review records cannot be marked "
                "as independent_screen_source"
            )
        return self


class ScreenRecord(StrictRecord):
    primary_key = ("screen_id",)
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "required": ["treatment_dose"],
                        "properties": {"treatment_dose": {"type": "number"}},
                    },
                    "then": {
                        "required": ["treatment_unit"],
                        "properties": {
                            "treatment_unit": {
                                "type": "string",
                                "minLength": 1,
                            }
                        },
                    },
                }
            ]
        },
    )

    screen_id: str = Field(min_length=1)
    study_id: str = Field(min_length=1)
    source_family_id: str | None = None
    raw_data_family_id: str | None = None
    perturbation_modality: PerturbationModality
    screen_design: str = Field(min_length=1)
    cell_line: str = Field(min_length=1)
    depmap_id: str | None = None
    cellosaurus_id: str | None = None
    engineered_genotype: str | None = None
    cancer_type: str | None = None
    library_id: str | None = None
    drug_name: str | None = None
    drug_id: str | None = None
    drug_class: str | None = None
    library_name: str | None = None
    library_version: str | None = None
    treatment_dose: float | None = Field(default=None, ge=0)
    treatment_unit: str | None = None
    duration_days: float | None = Field(default=None, ge=0)
    intended_direction: PhenotypeDirection = PhenotypeDirection.UNKNOWN
    input_mode: str | None = None
    data_accession: str | None = None
    source_url: HttpUrl | None = None
    available_date: date | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def dose_requires_unit(self) -> ScreenRecord:
        if self.treatment_dose is not None and not self.treatment_unit:
            raise ValueError("treatment_unit is required when treatment_dose is set")
        return self


class LibraryRecord(StrictRecord):
    primary_key = ("library_id",)

    library_id: str = Field(min_length=1)
    library_name: str = Field(min_length=1)
    library_version: str | None = None
    perturbation_modality: PerturbationModality
    library_scope: ScreenScale = ScreenScale.UNKNOWN
    target_gene_count: int | None = Field(default=None, ge=1)
    total_guide_count: int | None = Field(default=None, ge=1)
    expected_guides_per_gene: float | None = Field(default=None, gt=0)
    nontargeting_guide_count: int | None = Field(default=None, ge=0)
    vector_architecture: str | None = None
    enzyme: str | None = None
    guide_target_region: str | None = None
    reference_url: HttpUrl | None = None
    source_locator: str | None = None
    notes: str | None = None


class GuideAnnotationRecord(StrictRecord):
    primary_key = ("library_id", "sgrna_id")

    library_id: str = Field(min_length=1)
    sgrna_id: str = Field(min_length=1)
    gene_symbol: str = Field(min_length=1)
    targeting_class: str = Field(min_length=1)
    sequence: str | None = None
    chromosome: str | None = None
    cut_position: int | None = Field(default=None, ge=0)
    multi_target_count: int | None = Field(default=None, ge=0)
    activity_score: float | None = None
    specificity_score: float | None = None
    annotation_source: str | None = None
    annotation_version: str | None = None
    notes: str | None = None


class ScreenDesignRecord(StrictRecord):
    primary_key = ("screen_id",)

    screen_id: str = Field(min_length=1)
    source_role: SourceRole = SourceRole.ORIGINAL_SCREEN
    parent_screen_id: str | None = None
    screen_scale: ScreenScale = ScreenScale.UNKNOWN
    screen_format: str | None = None
    experimental_setup: str | None = None
    selection_strategy: SelectionStrategy = SelectionStrategy.UNKNOWN
    library_type: str | None = None
    library_methodology: str | None = None
    enzyme: str | None = None
    vector_architecture: str | None = None
    effector_components: str | None = None
    guide_target_region: str | None = None
    library_size_guides: int | None = Field(default=None, ge=1)
    target_gene_count: int | None = Field(default=None, ge=1)
    guides_per_gene_median: float | None = Field(default=None, gt=0)
    nontargeting_guide_count: int | None = Field(default=None, ge=0)
    library_moi: float | None = Field(default=None, ge=0)
    effector_moi: float | None = Field(default=None, ge=0)
    coverage_transduction: float | None = Field(default=None, ge=0)
    coverage_selection: float | None = Field(default=None, ge=0)
    coverage_harvest: float | None = Field(default=None, ge=0)
    infection_replicate_count: int | None = Field(default=None, ge=1)
    replicate_unit: ReplicateUnit = ReplicateUnit.UNKNOWN
    antibiotic_selection_days: float | None = Field(default=None, ge=0)
    editing_maturation_days: float | None = Field(default=None, ge=0)
    plasmid_reads_per_guide: float | None = Field(default=None, ge=0)
    plasmid_zero_guide_fraction: float | None = Field(default=None, ge=0, le=1)
    plasmid_skew_ratio: float | None = Field(default=None, ge=0)
    screen_reads_per_guide: float | None = Field(default=None, ge=0)
    gdna_fraction_amplified: float | None = Field(default=None, ge=0, le=1)
    pcr_cycle_count: int | None = Field(default=None, ge=0)
    cnv_amplification_risk_assessed: bool | None = None
    analysis_method: str | None = None
    normalization_method: str | None = None
    source_locator: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def derived_screen_requires_parent(self) -> ScreenDesignRecord:
        if self.source_role == SourceRole.REANALYSIS and not self.parent_screen_id:
            raise ValueError("reanalysis screens require parent_screen_id")
        return self


class ContrastRecord(StrictRecord):
    primary_key = ("screen_id", "contrast_id")

    screen_id: str = Field(min_length=1)
    contrast_id: str = Field(min_length=1)
    contrast_name: str = Field(min_length=1)
    treatment_name: str = Field(min_length=1)
    treatment_id: str | None = None
    drug_class: str | None = None
    treatment_dose: float | None = Field(default=None, ge=0)
    treatment_unit: str | None = None
    dose_basis: str | None = None
    control_type: ControlType
    comparator_name: str | None = None
    exposure_schedule: ExposureSchedule = ExposureSchedule.UNKNOWN
    exposure_days: float | None = Field(default=None, ge=0)
    recovery_days: float | None = Field(default=None, ge=0)
    endpoint_timepoint_days: float | None = Field(default=None, ge=0)
    phenotype_endpoint: str = Field(min_length=1)
    intended_direction: PhenotypeDirection
    vehicle_control_present: bool | None = None
    baseline_control_present: bool | None = None
    same_infection_split: bool | None = None
    matched_control: bool | None = None
    positive_control_description: str | None = None
    negative_control_description: str | None = None
    source_locator: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def contrast_units_and_comparator(self) -> ContrastRecord:
        if self.treatment_dose is not None and not self.treatment_unit:
            raise ValueError("treatment_unit is required when treatment_dose is set")
        if self.control_type == ControlType.VEHICLE and not self.comparator_name:
            raise ValueError("comparator_name is required when control_type is vehicle")
        return self


class ExternalScreenMapRecord(StrictRecord):
    primary_key = ("mapping_id",)

    mapping_id: str = Field(min_length=1)
    screen_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    external_dataset_id: str | None = None
    external_screen_id: str = Field(min_length=1)
    relationship: str = Field(min_length=1)
    replicate_number: int | None = Field(default=None, ge=1)
    source_url: HttpUrl | None = None
    retrieved_date: date
    notes: str | None = None


class ScreenIntakeRecord(StrictRecord):
    """One deterministic screen-level eligibility assessment."""

    primary_key = ("intake_id",)
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "required": ["assessment_stage"],
                        "properties": {
                            "assessment_stage": {"const": "index"},
                        },
                    },
                    "then": {
                        "properties": {
                            "status": {
                                "enum": ["metadata_only", "exclude"],
                            },
                            "benchmark_ready": {"const": False},
                        },
                    },
                },
                {
                    "if": {
                        "required": ["benchmark_ready"],
                        "properties": {
                            "benchmark_ready": {"const": True},
                        },
                    },
                    "then": {
                        "properties": {
                            "status": {"const": "benchmark_ready"},
                            "candidate_for_full_curation": {"const": True},
                        },
                    },
                    "else": {
                        "properties": {
                            "status": {
                                "enum": ["metadata_only", "exclude"],
                            },
                        },
                    },
                },
                {
                    "if": {
                        "required": ["status"],
                        "properties": {
                            "status": {"const": "exclude"},
                        },
                    },
                    "then": {
                        "properties": {
                            "candidate_for_full_curation": {"const": False},
                        },
                    },
                },
            ],
        },
    )

    intake_id: str = Field(min_length=1)
    screen_id: str = Field(min_length=1)
    external_screen_id: str | None = None
    external_dataset_id: str | None = None
    source_name: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    policy_version: Literal[2]
    assessment_stage: AssessmentStage
    status: IntakeStatus
    candidate_for_full_curation: bool
    benchmark_ready: bool
    reason_codes: str | None = None
    assessed_date: date
    notes: str | None = None

    @model_validator(mode="after")
    def status_is_internally_consistent(self) -> ScreenIntakeRecord:
        if self.assessment_stage == AssessmentStage.INDEX and (
            self.status == IntakeStatus.BENCHMARK_READY or self.benchmark_ready
        ):
            raise ValueError("index-only metadata cannot establish benchmark readiness")
        if self.benchmark_ready != (self.status == IntakeStatus.BENCHMARK_READY):
            raise ValueError("benchmark_ready must agree with benchmark_ready status")
        if self.benchmark_ready and not self.candidate_for_full_curation:
            raise ValueError(
                "benchmark-ready screens must remain full-curation candidates"
            )
        if self.status == IntakeStatus.EXCLUDE and self.candidate_for_full_curation:
            raise ValueError("excluded screens cannot be full-curation candidates")
        return self


class CurationQueueRecord(StrictRecord):
    """Deterministic, outcome-blind ordering for full-text screen curation."""

    primary_key = ("queue_id",)

    queue_id: str = Field(min_length=1)
    queue_rank: int = Field(ge=1)
    source_round: int = Field(ge=1)
    source_version: str = Field(min_length=1)
    policy_version: Literal[2]
    bucket: CurationQueueBucket
    screen_id: str = Field(min_length=1)
    external_screen_id: str = Field(min_length=1)
    source_id: str | None = None
    source_type: str | None = None
    source_family_id: str | None = None
    author: str | None = None
    screen_name: str | None = None
    experimental_setup: str | None = None
    condition_name: str | None = None
    cell_line: str | None = None
    scope_unknown_count: int = Field(ge=0)
    metadata_completeness_count: int = Field(ge=0)
    full_gene_score_set_available: bool | None = None
    reason_codes: str | None = None


_FULL_TEXT_REVIEW_JSON_SCHEMA = {
    "allOf": [
        {
            "if": {
                "required": ["quantitative_data_status"],
                "properties": {
                    "quantitative_data_status": {
                        "enum": ["raw_reads_public", "raw_counts_public"]
                    }
                },
            },
            "then": {
                "required": ["data_accession", "raw_data_family_id"],
                "properties": {
                    "data_accession": {"type": "string", "minLength": 1},
                    "raw_data_family_id": {"type": "string", "minLength": 1},
                },
            },
        },
        *[
            {
                "if": {
                    "required": ["validation_status"],
                    "properties": {"validation_status": {"const": status}},
                },
                "then": {
                    "required": [field_name],
                    "properties": {field_name: {"type": "string", "minLength": 1}},
                },
            }
            for status, field_name in (
                ("candidate_v3", "candidate_v3_genes"),
                ("candidate_v2", "candidate_v2_genes"),
                ("candidate_v1", "candidate_v1_genes"),
                ("nonqualifying_only", "nonqualifying_validation_genes"),
            )
        ],
        {
            "if": {
                "required": ["disposition"],
                "properties": {"disposition": {"const": "exclude"}},
            },
            "then": {
                "required": ["scope_outcome"],
                "properties": {"scope_outcome": {"const": "fail"}},
            },
            "else": {"properties": {"scope_outcome": {"enum": ["pass", "unknown"]}}},
        },
        {
            "if": {
                "required": ["validation_status"],
                "properties": {"validation_status": {"const": "candidate_v2"}},
            },
            "then": {"properties": {"candidate_v3_genes": {"type": "null"}}},
        },
        {
            "if": {
                "required": ["validation_status"],
                "properties": {"validation_status": {"const": "candidate_v1"}},
            },
            "then": {
                "properties": {
                    "candidate_v3_genes": {"type": "null"},
                    "candidate_v2_genes": {"type": "null"},
                }
            },
        },
        {
            "if": {
                "required": ["validation_status"],
                "properties": {
                    "validation_status": {
                        "enum": ["nonqualifying_only", "none_reported", "unresolved"]
                    }
                },
            },
            "then": {
                "properties": {
                    "candidate_v3_genes": {"type": "null"},
                    "candidate_v2_genes": {"type": "null"},
                    "candidate_v1_genes": {"type": "null"},
                }
            },
        },
        {
            "if": {
                "required": ["validation_status"],
                "properties": {
                    "validation_status": {"enum": ["none_reported", "unresolved"]}
                },
            },
            "then": {
                "properties": {"nonqualifying_validation_genes": {"type": "null"}}
            },
        },
    ],
    "x-semantic-rules": [
        "Pipe-delimited blocker and gene lists are sorted and unique.",
        "A gene cannot appear at multiple validation evidence levels.",
        "Blocker codes must be policy-v2 rules and agree with raw-family resolution.",
    ],
}


class FullTextReviewRecord(StrictRecord):
    """Outcome-aware review kept downstream of the frozen curation queue.

    This record deliberately has no ``benchmark_ready`` field. Readiness is
    established only by ``ScreenIntakeRecord`` plus linked registry facts and
    ``validate_registry_integrity``.
    """

    primary_key = ("review_id",)
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
        json_schema_extra=_FULL_TEXT_REVIEW_JSON_SCHEMA,
    )

    review_id: str = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    queue_id: str = Field(min_length=1)
    queue_rank: int = Field(ge=1)
    screen_id: str = Field(min_length=1)
    external_screen_id: str = Field(min_length=1)
    source_family_id: str = Field(min_length=1)
    quantitative_asset_family_id: str = Field(min_length=1)
    raw_data_family_id: str | None = None
    source_id: str = Field(min_length=1)
    doi: str = Field(min_length=1)
    paper_url: HttpUrl
    full_text_url: HttpUrl
    supplement_url: HttpUrl
    full_text_reviewed: bool
    supplement_review: ReviewCompleteness
    scope_outcome: EligibilityOutcome
    design_review: ReviewCompleteness
    sample_map_review: ReviewCompleteness
    screen_model: str = Field(min_length=1)
    library_design: str = Field(min_length=1)
    treatment_contrast: str = Field(min_length=1)
    screen_replication: str = Field(min_length=1)
    analysis_method: str = Field(min_length=1)
    quantitative_data_status: QuantitativeDataStatus
    quantitative_asset_locator: str = Field(min_length=1)
    data_accession: str | None = None
    rights_outcome: EligibilityOutcome
    rights_basis: str = Field(min_length=1)
    validation_status: ValidationReviewStatus
    candidate_v3_genes: str | None = Field(default=None, pattern=r"^[^|]+(?:\|[^|]+)*$")
    candidate_v2_genes: str | None = Field(default=None, pattern=r"^[^|]+(?:\|[^|]+)*$")
    candidate_v1_genes: str | None = Field(default=None, pattern=r"^[^|]+(?:\|[^|]+)*$")
    nonqualifying_validation_genes: str | None = Field(
        default=None, pattern=r"^[^|]+(?:\|[^|]+)*$"
    )
    validation_source_locator: str = Field(min_length=1)
    disposition: Literal["metadata_only", "exclude"]
    blocker_codes: str = Field(min_length=1, pattern=r"^[^|]+(?:\|[^|]+)*$")
    assessed_date: date
    curator: str = Field(min_length=1)
    notes: str | None = None

    @model_validator(mode="after")
    def review_is_internally_consistent(self) -> FullTextReviewRecord:
        candidate_fields = {
            ValidationReviewStatus.CANDIDATE_V3: self.candidate_v3_genes,
            ValidationReviewStatus.CANDIDATE_V2: self.candidate_v2_genes,
            ValidationReviewStatus.CANDIDATE_V1: self.candidate_v1_genes,
        }
        candidate_statuses = set(candidate_fields)
        if (
            self.validation_status in candidate_statuses
            and not candidate_fields[self.validation_status]
        ):
            raise ValueError(
                "candidate validation status requires genes at the same level"
            )
        if (
            self.validation_status == ValidationReviewStatus.NONQUALIFYING_ONLY
            and not self.nonqualifying_validation_genes
        ):
            raise ValueError(
                "nonqualifying validation status requires the reviewed genes"
            )
        if (
            self.validation_status
            in {
                ValidationReviewStatus.NONE_REPORTED,
                ValidationReviewStatus.UNRESOLVED,
            }
            and self.nonqualifying_validation_genes
        ):
            raise ValueError(
                "none_reported or unresolved status cannot list nonqualifying genes"
            )
        populated_candidate_statuses = [
            status for status, genes in candidate_fields.items() if genes
        ]
        expected_status = next(
            (
                status
                for status in (
                    ValidationReviewStatus.CANDIDATE_V3,
                    ValidationReviewStatus.CANDIDATE_V2,
                    ValidationReviewStatus.CANDIDATE_V1,
                )
                if status in populated_candidate_statuses
            ),
            None,
        )
        if expected_status is not None and self.validation_status != expected_status:
            raise ValueError(
                "validation_status must equal the highest populated candidate level"
            )
        if (
            self.quantitative_data_status
            in {
                QuantitativeDataStatus.RAW_READS_PUBLIC,
                QuantitativeDataStatus.RAW_COUNTS_PUBLIC,
            }
            and not self.data_accession
        ):
            raise ValueError("public raw reads or counts require a data_accession")
        if (
            self.quantitative_data_status
            in {
                QuantitativeDataStatus.RAW_READS_PUBLIC,
                QuantitativeDataStatus.RAW_COUNTS_PUBLIC,
            }
            and not self.raw_data_family_id
        ):
            raise ValueError("public raw reads or counts require raw_data_family_id")
        if (
            self.disposition == "exclude"
            and self.scope_outcome != EligibilityOutcome.FAIL
        ):
            raise ValueError("exclude disposition requires a failed scope review")
        if (
            self.disposition == "metadata_only"
            and self.scope_outcome == EligibilityOutcome.FAIL
        ):
            raise ValueError("failed scope review requires exclude disposition")

        blockers = self.blocker_codes.split("|")
        if blockers != sorted(set(blockers)) or any(not value for value in blockers):
            raise ValueError("blocker_codes must be unique, sorted, and pipe-delimited")
        unknown_blockers = set(blockers) - INTAKE_POLICY_V2_BENCHMARK_RULE_IDS
        if unknown_blockers:
            unknown_rules = sorted(unknown_blockers)
            raise ValueError(
                f"blocker_codes contain unknown policy rules: {unknown_rules}"
            )
        raw_family_blocker = "provenance.raw_data_family"
        if self.raw_data_family_id and raw_family_blocker in blockers:
            raise ValueError(
                "a resolved raw_data_family_id cannot retain its provenance blocker"
            )
        if not self.raw_data_family_id and raw_family_blocker not in blockers:
            raise ValueError(
                "a missing raw_data_family_id requires its provenance blocker"
            )
        gene_fields = {
            "candidate_v3_genes": self.candidate_v3_genes,
            "candidate_v2_genes": self.candidate_v2_genes,
            "candidate_v1_genes": self.candidate_v1_genes,
            "nonqualifying_validation_genes": self.nonqualifying_validation_genes,
        }
        seen_genes: set[str] = set()
        for field_name, values in gene_fields.items():
            if not values:
                continue
            genes = values.split("|")
            if genes != sorted(set(genes)) or any(not value for value in genes):
                raise ValueError(
                    f"{field_name} must be unique, sorted, and pipe-delimited"
                )
            overlap = seen_genes & set(genes)
            if overlap:
                raise ValueError(
                    "a validation gene cannot appear at multiple evidence levels: "
                    f"{sorted(overlap)}"
                )
            seen_genes.update(genes)
        return self


class EligibilityCheckRecord(StrictRecord):
    """Auditable result for one intake rule applied to one screen."""

    primary_key = ("check_id",)

    check_id: str = Field(min_length=1)
    intake_id: str = Field(min_length=1)
    screen_id: str = Field(min_length=1)
    assessment_stage: AssessmentStage
    rule_id: str = Field(min_length=1)
    outcome: EligibilityOutcome
    observed_value: str | None = None
    normalized_value: str | None = None
    required_for_scope: bool
    required_for_benchmark: bool
    reason_code: str = Field(min_length=1)
    source_locator: str = Field(min_length=1)
    notes: str | None = None


class DesignProvenanceRecord(StrictRecord):
    primary_key = ("provenance_id",)

    provenance_id: str = Field(min_length=1)
    study_id: str | None = None
    screen_id: str | None = None
    contrast_id: str | None = None
    sample_id: str | None = None
    entity_type: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    field_name: str = Field(min_length=1)
    reported_value: str | None = None
    normalized_value: str | None = None
    source_kind: str = Field(min_length=1)
    source_url: HttpUrl
    source_locator: str = Field(min_length=1)
    extraction_method: str = Field(min_length=1)
    confidence: str = Field(min_length=1)
    available_date: date
    retrieved_date: date
    curator: str = Field(min_length=1)
    notes: str | None = None

    @model_validator(mode="after")
    def provenance_value_and_dates(self) -> DesignProvenanceRecord:
        if not self.reported_value and not self.normalized_value:
            raise ValueError(
                "design provenance requires reported_value or normalized_value"
            )
        if self.retrieved_date < self.available_date:
            raise ValueError("retrieved_date cannot precede available_date")
        return self


class SampleRecord(StrictRecord):
    primary_key = ("sample_id",)

    sample_id: str = Field(min_length=1)
    screen_id: str = Field(min_length=1)
    contrast_id: str = Field(min_length=1)
    condition_role: SampleRole
    sample_subrole: str | None = None
    replicate: int = Field(ge=1)
    biological_replicate_id: str | None = None
    infection_replicate_id: str | None = None
    technical_replicate_id: str | None = None
    pair_id: str | None = None
    replicate_unit: ReplicateUnit = ReplicateUnit.UNKNOWN
    timepoint_days: float | None = Field(default=None, ge=0)
    timepoint_reference: str | None = None
    treatment_name: str | None = None
    treatment_dose: float | None = Field(default=None, ge=0)
    treatment_unit: str | None = None
    batch: str | None = None
    library_prep_batch: str | None = None
    sequencing_batch: str | None = None
    cell_count: int | None = Field(default=None, ge=0)
    coverage_per_guide: float | None = Field(default=None, ge=0)
    fastq_accession: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def sample_time_and_dose_units(self) -> SampleRecord:
        if self.timepoint_days is not None and not self.timepoint_reference:
            raise ValueError(
                "timepoint_reference is required when timepoint_days is set"
            )
        if self.treatment_dose is not None and not self.treatment_unit:
            raise ValueError("treatment_unit is required when treatment_dose is set")
        return self


class GeneScoreRecord(StrictRecord):
    primary_key = (
        "screen_id",
        "contrast_id",
        "gene_symbol",
        "method",
        "analysis_tail",
        "direction",
        "cnv_corrected",
    )
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "required": ["cnv_corrected"],
                        "properties": {"cnv_corrected": {"const": True}},
                    },
                    "then": {
                        "required": ["correction_method"],
                        "properties": {
                            "correction_method": {
                                "type": "string",
                                "minLength": 1,
                            }
                        },
                    },
                }
            ],
            "anyOf": [
                {
                    "required": [field],
                    "properties": {field: {"type": "number"}},
                }
                for field in ("effect", "p_value", "fdr", "rank", "score")
            ]
            + [
                {
                    "required": ["author_hit"],
                    "properties": {"author_hit": {"type": "boolean"}},
                }
            ],
        },
    )

    screen_id: str = Field(min_length=1)
    contrast_id: str = Field(min_length=1)
    gene_symbol: str = Field(min_length=1)
    method: str = Field(min_length=1)
    analysis_tail: str | None = None
    direction: PhenotypeDirection
    effect: float | None = None
    p_value: float | None = Field(default=None, ge=0, le=1)
    fdr: float | None = Field(default=None, ge=0, le=1)
    rank: float | None = Field(default=None, ge=1)
    score: float | None = None
    author_hit: bool | None = None
    score_type: str | None = None
    cnv_corrected: bool = False
    correction_method: str | None = None
    method_version: str | None = None
    source_database: str | None = None
    source_screen_id: str | None = None
    source_file: str | None = None

    @model_validator(mode="after")
    def corrected_requires_method(self) -> GeneScoreRecord:
        if self.cnv_corrected and not self.correction_method:
            raise ValueError("correction_method is required when cnv_corrected is true")
        if all(
            value is None
            for value in (
                self.effect,
                self.p_value,
                self.fdr,
                self.rank,
                self.score,
                self.author_hit,
            )
        ):
            raise ValueError("gene score requires a numeric measurement or author_hit")
        return self


_VALIDATION_EVENT_JSON_SCHEMA = {
    "allOf": [
        {
            "not": {
                "required": [
                    "phenotype_reproduced",
                    "opposite_direction_reproduced",
                ],
                "properties": {
                    "phenotype_reproduced": {"const": True},
                    "opposite_direction_reproduced": {"const": True},
                },
            }
        },
        {
            "if": {
                "properties": {
                    "label_code": {"enum": ["V3", "V2", "V1", "F0", "D", "A", "T"]}
                }
            },
            "then": {
                "required": ["testing_status"],
                "properties": {"testing_status": {"const": "tested"}},
            },
        },
        {
            "if": {"properties": {"label_code": {"enum": ["V2", "V3"]}}},
            "then": {
                "required": [
                    "testing_status",
                    "perturbation_confirmed",
                    "phenotype_reproduced",
                    "appropriate_control",
                ],
                "properties": {
                    "testing_status": {"const": "tested"},
                    "perturbation_confirmed": {"const": True},
                    "phenotype_reproduced": {"const": True},
                    "appropriate_control": {"const": True},
                    "opposite_direction_reproduced": {"enum": [False, None]},
                },
                "anyOf": [
                    {
                        "required": ["independent_reagent_count"],
                        "properties": {
                            "independent_reagent_count": {
                                "type": "integer",
                                "minimum": 2,
                            }
                        },
                    },
                    {
                        "required": ["orthogonal_perturbation"],
                        "properties": {"orthogonal_perturbation": {"const": True}},
                    },
                ],
            },
        },
        {
            "if": {"properties": {"label_code": {"const": "V3"}}},
            "then": {
                "anyOf": [
                    {
                        "required": ["rescue_performed"],
                        "properties": {"rescue_performed": {"const": True}},
                    },
                    {
                        "required": ["causal_reversal_performed"],
                        "properties": {"causal_reversal_performed": {"const": True}},
                    },
                ]
            },
        },
        {
            "if": {"properties": {"label_code": {"const": "F0"}}},
            "then": {
                "required": [
                    "testing_status",
                    "perturbation_confirmed",
                    "assay_adequate",
                    "phenotype_reproduced",
                ],
                "properties": {
                    "testing_status": {"const": "tested"},
                    "perturbation_confirmed": {"const": True},
                    "assay_adequate": {"const": True},
                    "phenotype_reproduced": {"const": False},
                    "opposite_direction_reproduced": {"enum": [False, None]},
                },
            },
        },
        {
            "if": {"properties": {"label_code": {"const": "D"}}},
            "then": {
                "required": [
                    "testing_status",
                    "perturbation_confirmed",
                    "phenotype_reproduced",
                    "opposite_direction_reproduced",
                ],
                "properties": {
                    "testing_status": {"const": "tested"},
                    "perturbation_confirmed": {"const": True},
                    "phenotype_reproduced": {"const": False},
                    "opposite_direction_reproduced": {"const": True},
                },
            },
        },
        {
            "if": {"properties": {"label_code": {"const": "U"}}},
            "then": {
                "required": ["testing_status"],
                "properties": {
                    "testing_status": {"enum": ["not_tested", "unknown"]},
                    "perturbation_confirmed": {"enum": [False, None]},
                    "independent_reagent_count": {"enum": [0, None]},
                    "orthogonal_perturbation": {"enum": [False, None]},
                    "phenotype_reproduced": {"enum": [False, None]},
                    "opposite_direction_reproduced": {"enum": [False, None]},
                    "rescue_performed": {"enum": [False, None]},
                    "causal_reversal_performed": {"enum": [False, None]},
                    "effect_size": {"type": "null"},
                    "p_value": {"type": "null"},
                },
            },
        },
        {
            "if": {
                "required": ["validation_treatment_dose"],
                "properties": {"validation_treatment_dose": {"type": "number"}},
            },
            "then": {
                "required": ["validation_treatment_unit"],
                "properties": {
                    "validation_treatment_unit": {
                        "type": "string",
                        "minLength": 1,
                    }
                },
            },
        },
    ]
}


class ValidationEventRecord(StrictRecord):
    primary_key = ("event_id",)
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
        json_schema_extra=_VALIDATION_EVENT_JSON_SCHEMA,
    )

    event_id: str = Field(min_length=1)
    study_id: str = Field(min_length=1)
    screen_id: str | None = None
    contrast_id: str | None = None
    gene_symbol: str = Field(min_length=1)
    drug_name: str = Field(min_length=1)
    cell_line: str = Field(min_length=1)
    perturbation_modality: PerturbationModality
    phenotype_direction: PhenotypeDirection
    label_code: LabelCode
    testing_status: TestingStatus
    perturbation_confirmed: bool | None = None
    independent_reagent_count: int | None = Field(default=None, ge=0)
    orthogonal_perturbation: bool | None = None
    appropriate_control: bool | None = None
    assay_adequate: bool | None = None
    phenotype_reproduced: bool | None = None
    opposite_direction_reproduced: bool | None = None
    rescue_performed: bool | None = None
    causal_reversal_performed: bool | None = None
    effect_size: float | None = None
    effect_metric: str | None = None
    p_value: float | None = Field(default=None, ge=0, le=1)
    validation_treatment_dose: float | None = Field(default=None, ge=0)
    validation_treatment_unit: str | None = None
    validation_exposure_days: float | None = Field(default=None, ge=0)
    validation_assay_endpoint: str | None = None
    validation_control_type: ControlType | None = None
    validation_biological_replicate_count: int | None = Field(default=None, ge=1)
    validation_guide_source: str | None = None
    protein_confirmed: bool | None = None
    second_model_tested: bool | None = None
    source_url: HttpUrl
    source_locator: str = Field(min_length=1)
    curator: str | None = None
    adjudication_status: str = "single_curator"
    notes: str | None = None

    @model_validator(mode="after")
    def label_consistency(self) -> ValidationEventRecord:
        if (
            self.validation_treatment_dose is not None
            and not self.validation_treatment_unit
        ):
            raise ValueError(
                "validation_treatment_unit is required when "
                "validation_treatment_dose is set"
            )
        if (
            self.phenotype_reproduced is True
            and self.opposite_direction_reproduced is True
        ):
            raise ValueError(
                "predicted and opposite-direction phenotypes cannot both "
                "be marked reproduced in one event"
            )
        if self.label_code in {LabelCode.V2, LabelCode.V3}:
            if self.testing_status != TestingStatus.TESTED:
                raise ValueError("V2/V3 require testing_status=tested")
            if self.perturbation_confirmed is not True:
                raise ValueError("V2/V3 require confirmed perturbation")
            if self.phenotype_reproduced is not True:
                raise ValueError("V2/V3 require reproduced phenotype")
            if self.opposite_direction_reproduced is True:
                raise ValueError("V2/V3 cannot reproduce the opposite direction")
            if self.appropriate_control is not True:
                raise ValueError("V2/V3 require an appropriate control")
            reagent_rule_met = (
                self.independent_reagent_count or 0
            ) >= 2 or self.orthogonal_perturbation is True
            if not reagent_rule_met:
                raise ValueError(
                    "V2/V3 require at least two independent reagents or an "
                    "orthogonal perturbation strategy"
                )
        if (
            self.label_code == LabelCode.V3
            and self.rescue_performed is not True
            and self.causal_reversal_performed is not True
        ):
            raise ValueError("V3 requires rescue or equivalent causal reversal")
        if self.label_code == LabelCode.F0:
            if (
                self.testing_status != TestingStatus.TESTED
                or self.perturbation_confirmed is not True
                or self.assay_adequate is not True
                or self.phenotype_reproduced is not False
                or self.opposite_direction_reproduced is True
            ):
                raise ValueError(
                    "F0 requires a tested, confirmed perturbation, adequate assay, "
                    "and non-reproduced phenotype"
                )
        if self.label_code == LabelCode.D:
            if (
                self.testing_status != TestingStatus.TESTED
                or self.perturbation_confirmed is not True
                or self.phenotype_reproduced is not False
                or self.opposite_direction_reproduced is not True
            ):
                raise ValueError(
                    "D requires confirmed testing with the opposite phenotype"
                )
        if (
            self.label_code == LabelCode.U
            and self.testing_status == TestingStatus.TESTED
        ):
            raise ValueError("U cannot be used for a known tested event")
        if self.label_code == LabelCode.U:
            outcome_evidence_present = (
                self.perturbation_confirmed is True
                or (self.independent_reagent_count or 0) > 0
                or self.orthogonal_perturbation is True
                or self.phenotype_reproduced is True
                or self.opposite_direction_reproduced is True
                or self.rescue_performed is True
                or self.causal_reversal_performed is True
                or self.effect_size is not None
                or self.p_value is not None
            )
            if outcome_evidence_present:
                raise ValueError(
                    "U cannot carry evidence of a completed validation outcome"
                )
        if (
            self.label_code != LabelCode.U
            and self.testing_status != TestingStatus.TESTED
        ):
            raise ValueError("V3/V2/V1/F0/D/A/T require testing_status=tested")
        return self


class CandidateRecord(StrictRecord):
    primary_key = (
        "screen_id",
        "contrast_id",
        "gene_symbol",
        "phenotype_direction",
    )
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
        json_schema_extra={
            "allOf": [
                {
                    "if": {"properties": {"label_code": {"const": "U"}}},
                    "then": {
                        "properties": {
                            "testing_status": {"enum": ["not_tested", "unknown"]}
                        }
                    },
                },
                {
                    "if": {
                        "properties": {
                            "label_code": {
                                "enum": [
                                    "V3",
                                    "V2",
                                    "V1",
                                    "F0",
                                    "D",
                                    "A",
                                    "T",
                                ]
                            }
                        }
                    },
                    "then": {"properties": {"testing_status": {"const": "tested"}}},
                },
            ]
        },
    )

    study_id: str = Field(min_length=1)
    screen_id: str = Field(min_length=1)
    contrast_id: str = Field(min_length=1)
    gene_symbol: str = Field(min_length=1)
    phenotype_direction: PhenotypeDirection
    label_code: LabelCode
    testing_status: TestingStatus

    @model_validator(mode="after")
    def testing_status_consistency(self) -> CandidateRecord:
        if (
            self.label_code == LabelCode.U
            and self.testing_status == TestingStatus.TESTED
        ):
            raise ValueError("U candidates cannot have testing_status=tested")
        if (
            self.label_code != LabelCode.U
            and self.testing_status != TestingStatus.TESTED
        ):
            raise ValueError(
                "adjudicated or technical labels require testing_status=tested"
            )
        return self


class EvidenceRecord(StrictRecord):
    primary_key = ("evidence_id",)
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
        json_schema_extra={
            "anyOf": [
                {
                    "required": ["value_numeric"],
                    "properties": {"value_numeric": {"type": "number"}},
                },
                {
                    "required": ["value_text"],
                    "properties": {
                        "value_text": {
                            "type": "string",
                            "minLength": 1,
                        }
                    },
                },
            ],
            "x-semantic-rules": [
                "retrieved_date must be on or after available_date; enforced "
                "by validate_registry_integrity"
            ],
        },
    )

    evidence_id: str = Field(min_length=1)
    source_study_id: str | None = None
    source_screen_id: str | None = None
    raw_data_family_id: str | None = None
    gene_symbol: str = Field(min_length=1)
    evidence_type: str = Field(min_length=1)
    context_type: str = Field(min_length=1)
    context_value: str = Field(min_length=1)
    value_numeric: float | None = None
    value_text: str | None = None
    source_name: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_url: HttpUrl
    source_license: str | None = None
    available_date: date
    retrieved_date: date
    transformation_id: str | None = None
    source_locator: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def evidence_value(self) -> EvidenceRecord:
        if self.value_numeric is None and not self.value_text:
            raise ValueError("evidence requires value_numeric or non-empty value_text")
        return self


class DataAssetRecord(StrictRecord):
    """Versioned pointer to raw or derived data without redistributing it."""

    primary_key = ("asset_id",)
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "required": ["sha256"],
                        "properties": {"sha256": {"type": "string"}},
                    },
                    "then": {
                        "required": ["byte_size"],
                        "properties": {"byte_size": {"type": "integer"}},
                    },
                },
                {
                    "if": {
                        "required": ["byte_size"],
                        "properties": {"byte_size": {"type": "integer"}},
                    },
                    "then": {
                        "required": ["sha256"],
                        "properties": {"sha256": {"type": "string"}},
                    },
                },
                {
                    "if": {
                        "anyOf": [
                            {
                                "required": ["redistribution_raw"],
                                "properties": {"redistribution_raw": {"const": True}},
                            },
                            {
                                "required": ["redistribution_derived"],
                                "properties": {
                                    "redistribution_derived": {"const": True}
                                },
                            },
                        ]
                    },
                    "then": {
                        "anyOf": [
                            {
                                "required": ["license_spdx"],
                                "properties": {
                                    "license_spdx": {
                                        "type": "string",
                                        "minLength": 1,
                                    }
                                },
                            },
                            {
                                "required": ["license_terms_url"],
                                "properties": {"license_terms_url": {"type": "string"}},
                            },
                        ]
                    },
                },
                {
                    "if": {
                        "required": ["retrieved_at_utc"],
                        "properties": {"retrieved_at_utc": {"type": "string"}},
                    },
                    "then": {
                        "properties": {
                            "retrieved_at_utc": {"pattern": "(?:Z|[+-]00:00)$"}
                        }
                    },
                },
            ],
            "x-semantic-rules": [
                "retrieved_date must be on or after available_date; enforced "
                "at runtime because JSON Schema cannot compare fields",
                "retrieved_at_utc must carry a zero UTC offset",
            ],
        },
    )

    asset_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    asset_role: str = Field(min_length=1)
    accession: str | None = None
    source_url: HttpUrl
    available_date: date | None = None
    retrieved_date: date
    retrieved_at_utc: datetime | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    byte_size: int | None = Field(default=None, ge=0)
    checksum_provenance: str | None = None
    license_spdx: str | None = None
    license_terms_url: HttpUrl | None = None
    rights_holder: str | None = None
    redistribution_raw: bool | None = None
    redistribution_derived: bool | None = None
    study_id: str | None = None
    screen_id: str | None = None
    source_family_id: str | None = None
    raw_data_family_id: str | None = None
    download_method: str | None = None
    transformation_entrypoint: str | None = None
    code_commit: str | None = None
    curator_status: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def provenance_is_internally_consistent(self) -> DataAssetRecord:
        if self.available_date and self.retrieved_date < self.available_date:
            raise ValueError("retrieved_date cannot precede available_date")
        if self.retrieved_at_utc is not None:
            offset = self.retrieved_at_utc.utcoffset()
            if offset is None or offset.total_seconds() != 0:
                raise ValueError("retrieved_at_utc must include the UTC timezone")
        if (self.sha256 is None) != (self.byte_size is None):
            raise ValueError("sha256 and byte_size must be reported together")
        if (
            self.redistribution_raw is True or self.redistribution_derived is True
        ) and not (self.license_spdx or self.license_terms_url):
            raise ValueError(
                "permitted redistribution requires a license or terms basis"
            )
        return self


CONTRACTS: dict[str, type[StrictRecord]] = {
    "study": StudyRecord,
    "screen": ScreenRecord,
    "library": LibraryRecord,
    "guide_annotation": GuideAnnotationRecord,
    "screen_design": ScreenDesignRecord,
    "contrast": ContrastRecord,
    "sample": SampleRecord,
    "gene_score": GeneScoreRecord,
    "validation_event": ValidationEventRecord,
    "candidate": CandidateRecord,
    "evidence": EvidenceRecord,
    "external_screen_map": ExternalScreenMapRecord,
    "screen_intake": ScreenIntakeRecord,
    "curation_queue": CurationQueueRecord,
    "full_text_review": FullTextReviewRecord,
    "eligibility_check": EligibilityCheckRecord,
    "design_provenance": DesignProvenanceRecord,
    "data_asset": DataAssetRecord,
}


def validate_records(
    frame: pd.DataFrame, contract: str | type[StrictRecord]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate a dataframe without stopping at the first bad row.

    Returns a normalized dataframe and a row-level error dataframe.
    """

    model = CONTRACTS[contract] if isinstance(contract, str) else contract
    missing_columns = [
        name
        for name, field in model.model_fields.items()
        if field.is_required() and name not in frame.columns
    ]
    if missing_columns:
        return pd.DataFrame(), pd.DataFrame(
            [
                {
                    "row_number": 1,
                    "error": (
                        f"table is missing required columns: {sorted(missing_columns)}"
                    ),
                }
            ]
        )
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame(
            [{"row_number": 1, "error": "table contains no records"}]
        )

    normalized: list[dict[str, Any]] = []
    source_rows: list[int] = []
    errors: list[dict[str, Any]] = []
    clean_frame = frame.astype(object).where(pd.notna(frame), None)
    for row_number, record in enumerate(clean_frame.to_dict(orient="records"), start=2):
        try:
            parsed = model.model_validate(record)
            normalized.append(parsed.model_dump(mode="json"))
            source_rows.append(row_number)
        except Exception as exc:  # pydantic supplies detailed context in text
            errors.append({"row_number": row_number, "error": str(exc)})

    valid = pd.DataFrame(normalized)
    if not valid.empty and model.primary_key:
        duplicated = valid.duplicated(list(model.primary_key), keep=False)
        duplicate_indices = valid.index[duplicated].tolist()
        for idx in duplicate_indices:
            errors.append(
                {
                    "row_number": source_rows[int(idx)],
                    "error": f"duplicate primary key: {model.primary_key}",
                }
            )
        if duplicate_indices:
            valid = valid.loc[~duplicated].reset_index(drop=True)
    return valid, pd.DataFrame(errors)


_PRIMARY_VALIDATION_LABELS = frozenset(
    {LabelCode.V2, LabelCode.V3, LabelCode.F0, LabelCode.D}
)
_BENCHMARK_APPROVED_CURATOR_STATUSES = frozenset(
    {
        "approved",
        "verified",
        "curated",
        "approved_for_benchmark",
        "verified_for_benchmark",
        "curated_for_benchmark",
    }
)
_BENCHMARK_ADJUDICATION_STATUSES = frozenset(
    {
        "adjudicated",
        "consensus",
        "consensus_adjudicated",
        "double_curated",
        "dual_curator_consensus",
        "approved_for_benchmark",
        "verified_for_benchmark",
    }
)


def _fact_text(value: object) -> str | None:
    """Return a stripped scalar value while treating pandas missing values as absent."""

    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return None
    text = str(value).strip()
    return text or None


def _fact_token(value: object) -> str | None:
    text = _fact_text(value)
    if text is None:
        return None
    return "_".join(text.lower().replace("-", " ").split())


def _screen_records(
    table: pd.DataFrame | None, screen_id: object
) -> list[dict[str, Any]]:
    if table is None or "screen_id" not in table.columns:
        return []
    return [
        record
        for record in table.to_dict(orient="records")
        if record.get("screen_id") == screen_id
    ]


def _asset_rights_are_documented(record: dict[str, Any], *, raw: bool) -> bool:
    """Require a reviewed terms basis and an explicit redistribution decision.

    A ``False`` redistribution decision is still useful: it allows local use under
    documented terms without incorrectly claiming that the bytes may be published.
    """

    license_basis = _fact_text(record.get("license_spdx")) or _fact_text(
        record.get("license_terms_url")
    )
    status = _fact_token(record.get("curator_status"))
    review_complete = status in _BENCHMARK_APPROVED_CURATOR_STATUSES
    decision_field = "redistribution_raw" if raw else "redistribution_derived"
    decision = record.get(decision_field)
    decision_recorded = decision is not None and not pd.isna(decision)
    return bool(license_basis and review_complete and decision_recorded)


def _is_count_level_asset(role: object) -> bool:
    token = _fact_token(role) or ""
    return any(
        marker in token
        for marker in (
            "count_level",
            "count_matrix",
            "count_table",
            "guide_counts",
            "sgrna_counts",
            "raw_counts",
            "normalized_counts",
        )
    )


def _is_approved_analysis_asset(record: dict[str, Any]) -> bool:
    role = _fact_token(record.get("asset_role")) or ""
    is_analysis = any(
        marker in role
        for marker in (
            "analysis_result",
            "analysis_output",
            "gene_score",
            "screen_score",
            "author_score",
        )
    )
    status = _fact_token(record.get("curator_status")) or ""
    benchmark_approved = status in _BENCHMARK_APPROVED_CURATOR_STATUSES
    return is_analysis and benchmark_approved


def _benchmark_fact_failures(
    *,
    screen_id: object,
    screens: pd.DataFrame,
    contrasts: pd.DataFrame | None,
    samples: pd.DataFrame | None,
    gene_scores: pd.DataFrame | None,
    validation_events: pd.DataFrame | None,
    data_assets: pd.DataFrame | None,
) -> list[str]:
    """Derive policy-v2 readiness gates from linked registry facts.

    Passing eligibility rows are curator assertions. This check independently
    requires the records that those assertions summarize.
    """

    failures: list[str] = []
    screen_rows = _screen_records(screens, screen_id)
    screen_row = screen_rows[0] if screen_rows else {}
    source_family = _fact_text(screen_row.get("source_family_id"))
    raw_family = _fact_text(screen_row.get("raw_data_family_id"))
    if source_family is None:
        failures.append("provenance.source_family")
    if raw_family is None:
        failures.append("provenance.raw_data_family")

    mapped_contrasts: set[str] = set()
    contrast_rows = _screen_records(contrasts, screen_id)
    sample_rows = _screen_records(samples, screen_id)
    for contrast in contrast_rows:
        contrast_id = _fact_text(contrast.get("contrast_id"))
        control_type = _fact_token(contrast.get("control_type"))
        if (
            contrast_id is None
            or not _fact_text(contrast.get("treatment_name"))
            or control_type in {None, "unknown"}
        ):
            continue
        roles = {
            _fact_token(sample.get("condition_role"))
            for sample in sample_rows
            if sample.get("contrast_id") == contrast.get("contrast_id")
        }
        if "treatment" in roles and roles & {"control", "baseline"}:
            mapped_contrasts.add(contrast_id)
    if not mapped_contrasts:
        failures.append("curation.comparator_and_sample_map")

    numeric_score_pairs = {
        (
            _fact_text(record.get("contrast_id")),
            _fact_text(record.get("gene_symbol")),
        )
        for record in _screen_records(gene_scores, screen_id)
        if any(
            _fact_text(record.get(field)) is not None
            for field in ("effect", "p_value", "fdr", "rank", "score")
        )
    }
    numeric_score_pairs.discard((None, None))
    score_contrasts = {
        contrast_id
        for contrast_id, gene_symbol in numeric_score_pairs
        if contrast_id is not None and gene_symbol is not None
    }
    score_directions: dict[tuple[str | None, str | None], set[str | None]] = {}
    for record in _screen_records(gene_scores, screen_id):
        pair = (
            _fact_text(record.get("contrast_id")),
            _fact_text(record.get("gene_symbol")),
        )
        if pair not in numeric_score_pairs:
            continue
        score_directions.setdefault(pair, set()).add(
            _fact_token(record.get("direction"))
        )
    signal_bearing_contrasts: set[str] = set()
    rights_verified = False
    for asset in _screen_records(data_assets, screen_id):
        families_match = (
            source_family is not None
            and raw_family is not None
            and _fact_text(asset.get("source_family_id")) == source_family
            and _fact_text(asset.get("raw_data_family_id")) == raw_family
        )
        if not families_match:
            continue
        if _is_count_level_asset(asset.get("asset_role")):
            signal_bearing_contrasts.update(mapped_contrasts & score_contrasts)
            if _asset_rights_are_documented(asset, raw=True):
                rights_verified = True
        elif _is_approved_analysis_asset(asset):
            signal_bearing_contrasts.update(mapped_contrasts & score_contrasts)
            if _asset_rights_are_documented(asset, raw=False):
                rights_verified = True
    if not signal_bearing_contrasts:
        failures.append("data.count_level_signal")
    if not rights_verified:
        failures.append("rights.source_and_raw_data")

    primary_event_present = False
    for event in _screen_records(validation_events, screen_id):
        event_contrast = _fact_text(event.get("contrast_id"))
        event_gene = _fact_text(event.get("gene_symbol"))
        if (
            event_contrast not in signal_bearing_contrasts
            or (
                event_contrast,
                event_gene,
            )
            not in numeric_score_pairs
        ):
            continue
        directions = score_directions[(event_contrast, event_gene)]
        event_direction = _fact_token(event.get("phenotype_direction"))
        if (
            directions
            and None not in directions
            and PhenotypeDirection.UNKNOWN.value not in directions
            and event_direction not in directions
        ):
            continue
        clean_event = {
            key: (None if pd.isna(value) else value) for key, value in event.items()
        }
        try:
            parsed = ValidationEventRecord.model_validate(clean_event)
        except (TypeError, ValueError):
            continue
        adjudication_status = _fact_token(parsed.adjudication_status)
        if (
            parsed.label_code in _PRIMARY_VALIDATION_LABELS
            and adjudication_status in _BENCHMARK_ADJUDICATION_STATUSES
        ):
            primary_event_present = True
            break
    if not primary_event_present:
        failures.append("labels.adjudicated_validation_event")
    return failures


def validate_registry_integrity(
    *,
    studies: pd.DataFrame,
    screens: pd.DataFrame,
    libraries: pd.DataFrame | None = None,
    guide_annotations: pd.DataFrame | None = None,
    screen_designs: pd.DataFrame | None = None,
    contrasts: pd.DataFrame | None = None,
    samples: pd.DataFrame | None = None,
    gene_scores: pd.DataFrame | None = None,
    candidates: pd.DataFrame | None = None,
    validation_events: pd.DataFrame | None = None,
    evidence: pd.DataFrame | None = None,
    external_screen_maps: pd.DataFrame | None = None,
    screen_intake: pd.DataFrame | None = None,
    eligibility_checks: pd.DataFrame | None = None,
    design_provenance: pd.DataFrame | None = None,
    data_assets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Check core foreign-key relationships across normalized registry tables."""

    errors: list[dict[str, Any]] = []
    study_ids = set(studies.get("study_id", pd.Series(dtype=str)).dropna())
    screen_ids = set(screens.get("screen_id", pd.Series(dtype=str)).dropna())
    library_ids = (
        set(libraries.get("library_id", pd.Series(dtype=str)).dropna())
        if libraries is not None
        else set()
    )
    contrast_pairs = (
        set(
            zip(
                contrasts.get("screen_id", pd.Series(dtype=str)),
                contrasts.get("contrast_id", pd.Series(dtype=str)),
                strict=False,
            )
        )
        if contrasts is not None
        else set()
    )
    screen_to_study = dict(
        zip(
            screens.get("screen_id", pd.Series(dtype=str)),
            screens.get("study_id", pd.Series(dtype=str)),
            strict=False,
        )
    )

    for row, study_id in enumerate(
        screens.get("study_id", pd.Series(dtype=object)), start=2
    ):
        if study_id not in study_ids:
            errors.append(
                {
                    "table": "screens",
                    "row_number": row,
                    "error": f"unknown study_id: {study_id}",
                }
            )

    if libraries is not None and "library_id" in screens:
        for row, library_id in enumerate(screens["library_id"], start=2):
            if pd.isna(library_id):
                continue
            if library_id not in library_ids:
                errors.append(
                    {
                        "table": "screens",
                        "row_number": row,
                        "error": f"unknown library_id: {library_id}",
                    }
                )

    if guide_annotations is not None:
        for row, library_id in enumerate(
            guide_annotations.get("library_id", pd.Series(dtype=object)),
            start=2,
        ):
            if pd.isna(library_id) or library_id not in library_ids:
                errors.append(
                    {
                        "table": "guide_annotations",
                        "row_number": row,
                        "error": f"unknown library_id: {library_id}",
                    }
                )

    for table_name, table in (
        ("screen_designs", screen_designs),
        ("contrasts", contrasts),
        ("external_screen_maps", external_screen_maps),
        ("screen_intake", screen_intake),
        ("eligibility_checks", eligibility_checks),
    ):
        if table is None:
            continue
        for row, screen_id in enumerate(
            table.get("screen_id", pd.Series(dtype=object)), start=2
        ):
            if pd.isna(screen_id) or screen_id not in screen_ids:
                errors.append(
                    {
                        "table": table_name,
                        "row_number": row,
                        "error": f"unknown screen_id: {screen_id}",
                    }
                )

    intake_lookup: dict[object, dict[str, Any]] = {}
    matched_checks: dict[object, list[dict[str, Any]]] = {}
    if screen_intake is not None:
        for row_number, intake_row in enumerate(
            screen_intake.to_dict(orient="records"), start=2
        ):
            intake_id = intake_row.get("intake_id")
            if pd.notna(intake_id):
                intake_lookup[intake_id] = {
                    **intake_row,
                    "_row_number": row_number,
                }

            stage = intake_row.get("assessment_stage")
            status = intake_row.get("status")
            stage_value = (
                stage.value if isinstance(stage, AssessmentStage) else str(stage)
            )
            status_value = (
                status.value if isinstance(status, IntakeStatus) else str(status)
            )
            benchmark_ready = intake_row.get("benchmark_ready")
            candidate = intake_row.get("candidate_for_full_curation")
            ready_flag = None if pd.isna(benchmark_ready) else bool(benchmark_ready)
            candidate_flag = None if pd.isna(candidate) else bool(candidate)

            if stage_value == AssessmentStage.INDEX.value and (
                status_value == IntakeStatus.BENCHMARK_READY.value or ready_flag
            ):
                errors.append(
                    {
                        "table": "screen_intake",
                        "row_number": row_number,
                        "error": (
                            "index-stage intake cannot establish benchmark readiness"
                        ),
                    }
                )
            if ready_flag is not None and ready_flag != (
                status_value == IntakeStatus.BENCHMARK_READY.value
            ):
                errors.append(
                    {
                        "table": "screen_intake",
                        "row_number": row_number,
                        "error": (
                            "benchmark_ready must agree with benchmark_ready status"
                        ),
                    }
                )
            if (
                status_value == IntakeStatus.BENCHMARK_READY.value
                and candidate_flag is not True
            ):
                errors.append(
                    {
                        "table": "screen_intake",
                        "row_number": row_number,
                        "error": (
                            "benchmark_ready screens must remain "
                            "full-curation candidates"
                        ),
                    }
                )
            if status_value == IntakeStatus.EXCLUDE.value and candidate_flag is True:
                errors.append(
                    {
                        "table": "screen_intake",
                        "row_number": row_number,
                        "error": (
                            "excluded screens cannot be full-curation candidates"
                        ),
                    }
                )

    if eligibility_checks is not None:
        for row_number, check_row in enumerate(
            eligibility_checks.to_dict(orient="records"), start=2
        ):
            intake_id = check_row.get("intake_id")
            intake_row = intake_lookup.get(intake_id)
            if intake_row is None:
                errors.append(
                    {
                        "table": "eligibility_checks",
                        "row_number": row_number,
                        "error": f"unknown intake_id: {intake_id}",
                    }
                )
                continue

            check_screen_id = check_row.get("screen_id")
            check_stage = check_row.get("assessment_stage")
            check_stage_value = (
                check_stage.value
                if isinstance(check_stage, AssessmentStage)
                else str(check_stage)
            )
            intake_stage = intake_row.get("assessment_stage")
            intake_stage_value = (
                intake_stage.value
                if isinstance(intake_stage, AssessmentStage)
                else str(intake_stage)
            )
            matches_intake = True
            if check_screen_id != intake_row.get("screen_id"):
                matches_intake = False
                errors.append(
                    {
                        "table": "eligibility_checks",
                        "row_number": row_number,
                        "error": (
                            "screen_id does not match linked screen_intake row "
                            f"for intake_id {intake_id}"
                        ),
                    }
                )
            if check_stage_value != intake_stage_value:
                matches_intake = False
                errors.append(
                    {
                        "table": "eligibility_checks",
                        "row_number": row_number,
                        "error": (
                            "assessment_stage does not match linked "
                            f"screen_intake row for intake_id {intake_id}"
                        ),
                    }
                )
            if matches_intake:
                matched_checks.setdefault(intake_id, []).append(
                    {**check_row, "_row_number": row_number}
                )

    if screen_intake is not None and eligibility_checks is not None:
        for intake_id, intake_row in intake_lookup.items():
            status = intake_row.get("status")
            status_value = (
                status.value if isinstance(status, IntakeStatus) else str(status)
            )
            linked_checks = matched_checks.get(intake_id, [])
            try:
                policy_version = int(intake_row.get("policy_version"))
            except (TypeError, ValueError):
                policy_version = -1
            policy_rule_ids = INTAKE_POLICY_RULE_IDS.get(policy_version)
            if policy_rule_ids is None:
                errors.append(
                    {
                        "table": "screen_intake",
                        "row_number": intake_row["_row_number"],
                        "error": (
                            "unsupported or missing intake policy_version: "
                            f"{intake_row.get('policy_version')}"
                        ),
                    }
                )
                continue
            scope_rule_ids, benchmark_rule_ids = policy_rule_ids
            checks_by_rule: dict[str, list[dict[str, Any]]] = {}
            for check in linked_checks:
                rule_id = str(check.get("rule_id"))
                checks_by_rule.setdefault(rule_id, []).append(check)
                expected_scope = rule_id in scope_rule_ids
                expected_benchmark = rule_id in benchmark_rule_ids
                scope_flag = check.get("required_for_scope")
                benchmark_flag = check.get("required_for_benchmark")
                observed_scope = False if pd.isna(scope_flag) else bool(scope_flag)
                observed_benchmark = (
                    False if pd.isna(benchmark_flag) else bool(benchmark_flag)
                )
                if (
                    observed_scope != expected_scope
                    or observed_benchmark != expected_benchmark
                ):
                    errors.append(
                        {
                            "table": "eligibility_checks",
                            "row_number": check["_row_number"],
                            "error": (
                                f"rule flags do not match policy v{policy_version} "
                                f"for {rule_id}"
                            ),
                        }
                    )

            duplicate_rule_ids = sorted(
                rule_id for rule_id, checks in checks_by_rule.items() if len(checks) > 1
            )
            if duplicate_rule_ids:
                errors.append(
                    {
                        "table": "eligibility_checks",
                        "row_number": min(
                            check["_row_number"]
                            for rule_id in duplicate_rule_ids
                            for check in checks_by_rule[rule_id]
                        ),
                        "error": (
                            "duplicate rule_id values for intake_id "
                            f"{intake_id}: {duplicate_rule_ids}"
                        ),
                    }
                )

            outcomes_by_rule = {
                rule_id: (
                    outcome.value
                    if isinstance(outcome, EligibilityOutcome)
                    else str(outcome)
                )
                for rule_id, checks in checks_by_rule.items()
                if len(checks) == 1
                for outcome in [checks[0].get("outcome")]
            }

            failed_scope_rules = [
                rule_id
                for rule_id in sorted(scope_rule_ids)
                if outcomes_by_rule.get(rule_id) == EligibilityOutcome.FAIL.value
            ]

            if status_value == IntakeStatus.BENCHMARK_READY.value:
                missing_rules = sorted(benchmark_rule_ids - checks_by_rule.keys())
                nonpassing_rules = [
                    rule_id
                    for rule_id in sorted(benchmark_rule_ids)
                    if (
                        rule_id not in missing_rules
                        and outcomes_by_rule.get(rule_id)
                        != EligibilityOutcome.PASS.value
                    )
                ]
                if missing_rules:
                    errors.append(
                        {
                            "table": "screen_intake",
                            "row_number": intake_row["_row_number"],
                            "error": (
                                "benchmark_ready status is unsupported because "
                                "required policy rules are missing: "
                                f"{missing_rules}"
                            ),
                        }
                    )
                if nonpassing_rules:
                    errors.append(
                        {
                            "table": "screen_intake",
                            "row_number": intake_row["_row_number"],
                            "error": (
                                "benchmark_ready status is unsupported because "
                                "required eligibility checks are not all pass: "
                                f"{sorted(nonpassing_rules)}"
                            ),
                        }
                    )
                fact_failures = _benchmark_fact_failures(
                    screen_id=intake_row.get("screen_id"),
                    screens=screens,
                    contrasts=contrasts,
                    samples=samples,
                    gene_scores=gene_scores,
                    validation_events=validation_events,
                    data_assets=data_assets,
                )
                if fact_failures:
                    errors.append(
                        {
                            "table": "screen_intake",
                            "row_number": intake_row["_row_number"],
                            "error": (
                                "benchmark_ready status is unsupported because "
                                "linked registry facts do not establish: "
                                f"{fact_failures}"
                            ),
                        }
                    )

            if failed_scope_rules and status_value != IntakeStatus.EXCLUDE.value:
                errors.append(
                    {
                        "table": "screen_intake",
                        "row_number": intake_row["_row_number"],
                        "error": (
                            f"status {status_value} is unsupported because "
                            "required_for_scope checks failed: "
                            f"{sorted(failed_scope_rules)}"
                        ),
                    }
                )
            if status_value == IntakeStatus.EXCLUDE.value and not failed_scope_rules:
                errors.append(
                    {
                        "table": "screen_intake",
                        "row_number": intake_row["_row_number"],
                        "error": (
                            "exclude status is unsupported without a failed "
                            "required_for_scope check"
                        ),
                    }
                )

    if screen_designs is not None and "parent_screen_id" in screen_designs:
        for row, parent_screen_id in enumerate(
            screen_designs["parent_screen_id"], start=2
        ):
            if pd.isna(parent_screen_id):
                continue
            if parent_screen_id not in screen_ids:
                errors.append(
                    {
                        "table": "screen_designs",
                        "row_number": row,
                        "error": (f"unknown parent_screen_id: {parent_screen_id}"),
                    }
                )

    for table_name, table in (
        ("samples", samples),
        ("gene_scores", gene_scores),
        ("candidates", candidates),
        ("validation_events", validation_events),
    ):
        if table is None:
            continue
        if "study_id" in table:
            for row, study_id in enumerate(table["study_id"], start=2):
                if pd.isna(study_id) or study_id not in study_ids:
                    errors.append(
                        {
                            "table": table_name,
                            "row_number": row,
                            "error": f"unknown study_id: {study_id}",
                        }
                    )
        if "screen_id" not in table:
            continue
        for row, screen_id in enumerate(table["screen_id"], start=2):
            if pd.isna(screen_id):
                continue
            if screen_id not in screen_ids:
                errors.append(
                    {
                        "table": table_name,
                        "row_number": row,
                        "error": f"unknown screen_id: {screen_id}",
                    }
                )
                continue
            linked_study = screen_to_study.get(screen_id)
            if (
                table_name != "validation_events"
                and "study_id" in table
                and table.iloc[row - 2]["study_id"] != linked_study
            ):
                errors.append(
                    {
                        "table": table_name,
                        "row_number": row,
                        "error": (
                            "study_id does not match the study referenced by "
                            f"screen_id {screen_id}"
                        ),
                    }
                )

    if evidence is not None:
        required_dates = {"available_date", "retrieved_date"}
        if required_dates <= set(evidence.columns):
            available = pd.to_datetime(evidence["available_date"], errors="coerce")
            retrieved = pd.to_datetime(evidence["retrieved_date"], errors="coerce")
            invalid = available.notna() & retrieved.notna() & (retrieved < available)
            for row_number, is_invalid in enumerate(invalid, start=2):
                if is_invalid:
                    errors.append(
                        {
                            "table": "evidence",
                            "row_number": row_number,
                            "error": ("retrieved_date cannot precede available_date"),
                        }
                    )

    if candidates is not None and validation_events is not None:
        try:
            adjudicated = adjudicate_validation_events(validation_events)
        except ValueError as exc:
            errors.append(
                {
                    "table": "validation_events",
                    "row_number": 1,
                    "error": str(exc),
                }
            )
        else:
            candidate_lookup = {
                tuple(row[column] for column in CANDIDATE_KEY): row
                for _, row in candidates.iterrows()
            }
            adjudicated_keys = set()
            for _, event_row in adjudicated.iterrows():
                key = tuple(event_row[column] for column in CANDIDATE_KEY)
                adjudicated_keys.add(key)
                candidate_row = candidate_lookup.get(key)
                if candidate_row is None:
                    errors.append(
                        {
                            "table": "validation_events",
                            "row_number": 1,
                            "error": (
                                f"adjudicated event key has no candidate row: {key}"
                            ),
                        }
                    )
                elif str(candidate_row["label_code"]) != str(event_row["label_code"]):
                    errors.append(
                        {
                            "table": "candidates",
                            "row_number": 1,
                            "error": (
                                f"candidate label {candidate_row['label_code']} "
                                "does not match event adjudication "
                                f"{event_row['label_code']} for {key}"
                            ),
                        }
                    )
            for row_number, (_, candidate_row) in enumerate(
                candidates.iterrows(), start=2
            ):
                key = tuple(candidate_row[column] for column in CANDIDATE_KEY)
                if (
                    str(candidate_row["label_code"]) != LabelCode.U.value
                    and key not in adjudicated_keys
                ):
                    errors.append(
                        {
                            "table": "candidates",
                            "row_number": row_number,
                            "error": (
                                f"non-U candidate has no linked validation event: {key}"
                            ),
                        }
                    )

    if contrasts is not None:
        for table_name, table in (
            ("samples", samples),
            ("gene_scores", gene_scores),
            ("candidates", candidates),
            ("validation_events", validation_events),
        ):
            if table is None or not {"screen_id", "contrast_id"} <= set(table.columns):
                continue
            for row_number, row in enumerate(
                table[["screen_id", "contrast_id"]].itertuples(index=False, name=None),
                start=2,
            ):
                if tuple(row) not in contrast_pairs:
                    errors.append(
                        {
                            "table": table_name,
                            "row_number": row_number,
                            "error": (
                                "unknown or mismatched screen_id/contrast_id: "
                                f"{tuple(row)}"
                            ),
                        }
                    )

    if design_provenance is not None:
        for row_number, row in enumerate(
            design_provenance.to_dict(orient="records"), start=2
        ):
            study_id = row.get("study_id")
            screen_id = row.get("screen_id")
            contrast_id = row.get("contrast_id")
            if pd.notna(study_id) and study_id not in study_ids:
                errors.append(
                    {
                        "table": "design_provenance",
                        "row_number": row_number,
                        "error": f"unknown study_id: {study_id}",
                    }
                )
            if pd.notna(screen_id) and screen_id not in screen_ids:
                errors.append(
                    {
                        "table": "design_provenance",
                        "row_number": row_number,
                        "error": f"unknown screen_id: {screen_id}",
                    }
                )
            if (
                pd.notna(screen_id)
                and pd.notna(contrast_id)
                and (screen_id, contrast_id) not in contrast_pairs
            ):
                errors.append(
                    {
                        "table": "design_provenance",
                        "row_number": row_number,
                        "error": (
                            "unknown or mismatched screen_id/contrast_id: "
                            f"{(screen_id, contrast_id)}"
                        ),
                    }
                )
    if data_assets is not None:
        for row_number, row in enumerate(
            data_assets.to_dict(orient="records"), start=2
        ):
            study_id = row.get("study_id")
            screen_id = row.get("screen_id")
            if pd.notna(study_id) and study_id not in study_ids:
                errors.append(
                    {
                        "table": "data_assets",
                        "row_number": row_number,
                        "error": f"unknown study_id: {study_id}",
                    }
                )
            if pd.notna(screen_id) and screen_id not in screen_ids:
                errors.append(
                    {
                        "table": "data_assets",
                        "row_number": row_number,
                        "error": f"unknown screen_id: {screen_id}",
                    }
                )
    return pd.DataFrame(errors)
