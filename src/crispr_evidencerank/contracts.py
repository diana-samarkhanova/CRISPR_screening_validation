"""Pydantic contracts for the normalized evidence registry."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from .labels import CANDIDATE_KEY, LabelCode, _resolve_released_validation_events


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


class PerturbedCompartment(StrEnum):
    """Cell compartment in which the perturbation was introduced."""

    TUMOR_CELL = "tumor_cell"
    IMMUNE_CELL = "immune_cell"
    OTHER = "other"
    UNKNOWN = "unknown"


class ExperimentalSetting(StrEnum):
    IN_VITRO = "in_vitro"
    IN_VIVO = "in_vivo"
    EX_VIVO = "ex_vivo"
    UNKNOWN = "unknown"


class InterventionModality(StrEnum):
    SMALL_MOLECULE = "small_molecule"
    ANTIBODY = "antibody"
    CELL_THERAPY = "cell_therapy"
    GENE_THERAPY = "gene_therapy"
    RADIOTHERAPY = "radiotherapy"
    COMBINATION = "combination"
    OTHER = "other"
    UNKNOWN = "unknown"


class ScreenEndpointCategory(StrEnum):
    DRUG_RESPONSE_VIABILITY = "drug_response_viability"
    IMMUNE_KILLING = "immune_killing"
    IMMUNE_CELL_FITNESS = "immune_cell_fitness"
    MARKER_EXPRESSION = "marker_expression"
    OTHER = "other"
    UNKNOWN = "unknown"


class PreclinicalEvidenceScope(StrEnum):
    TREATMENT_CONTEXT = "treatment_context"
    GENE_SPECIFIC = "gene_specific"


class PreclinicalDirectionInferenceStatus(StrEnum):
    DIRECTION_SUPPORTED = "direction_supported"
    NEUTRAL_SUPPORTED = "neutral_supported"
    INCONCLUSIVE = "inconclusive"
    UNSUPPORTED = "unsupported"
    NOT_ASSESSED = "not_assessed"


class PreclinicalModelType(StrEnum):
    CELL_LINE_2D = "cell_line_2d"
    CELL_LINE_3D = "cell_line_3d"
    IMMUNE_COCULTURE = "immune_coculture"
    ORGANOID = "organoid"
    PDX_DERIVED_ORGANOID = "pdx_derived_organoid"
    EX_VIVO_TISSUE = "ex_vivo_tissue"
    CELL_LINE_XENOGRAFT = "cell_line_xenograft"
    PDX = "pdx"
    SYNGENEIC = "syngeneic"
    GENETICALLY_ENGINEERED_MODEL = "genetically_engineered_model"
    HUMANIZED_MOUSE = "humanized_mouse"
    OTHER_IN_VIVO = "other_in_vivo"
    OTHER = "other"


class PreclinicalClaimType(StrEnum):
    DIRECT_PERTURBATIONAL_INTERACTION = "direct_perturbational_interaction"
    NATURAL_BIOMARKER_ASSOCIATION = "natural_biomarker_association"
    TREATMENT_ACTIVITY_ONLY = "treatment_activity_only"
    MECHANISTIC_ONLY = "mechanistic_only"


class MolecularMeasurementTimepoint(StrEnum):
    PRETREATMENT = "pretreatment"
    ON_TREATMENT = "on_treatment"
    POST_TREATMENT = "post_treatment"
    UNKNOWN = "unknown"


class BiomarkerFeatureType(StrEnum):
    GENOMIC_MUTATION = "genomic_mutation"
    COPY_NUMBER = "copy_number"
    FUSION = "fusion"
    RNA_EXPRESSION = "rna_expression"
    PROTEIN_EXPRESSION = "protein_expression"
    EPIGENETIC = "epigenetic"
    GENOMIC_SIGNATURE = "genomic_signature"
    FUNCTIONAL_STATUS = "functional_status"
    OTHER = "other"


class BiomarkerAxisObservationStatus(StrEnum):
    """Curator-visible availability state for a typed biomarker bundle."""

    OBSERVED = "observed"
    NOT_ASSESSED = "not_assessed"
    NOT_REPORTED = "not_reported"
    INSUFFICIENT_MATERIAL = "insufficient_material"
    UNKNOWN = "unknown"


class ComparatorExposureType(StrEnum):
    """Whether a comparator arm contains an active therapeutic exposure."""

    ACTIVE_THERAPEUTIC = "active_therapeutic"
    PLACEBO = "placebo"
    VEHICLE = "vehicle"
    NO_ACTIVE_THERAPEUTIC = "no_active_therapeutic"
    UNRESOLVED = "unresolved"


class RegimenComponentRelation(StrEnum):
    """How listed canonical active components form a regimen."""

    FIXED_ALL_OF = "fixed_all_of"
    ALTERNATIVE_ONE_OF = "alternative_one_of"
    NONE = "none"
    UNRESOLVED = "unresolved"


class InteractionEffectScale(StrEnum):
    """Controlled interaction-effect scales with a known statistical null."""

    ADDITIVE_COEFFICIENT = "additive_coefficient"
    DIFFERENCE_IN_EFFECT = "difference_in_effect"
    HAZARD_RATIO = "hazard_ratio"
    ODDS_RATIO = "odds_ratio"
    RISK_RATIO = "risk_ratio"
    RATIO_OF_RATIOS = "ratio_of_ratios"


class InteractionInferenceStatus(StrEnum):
    """Curated conclusion of a formal treatment-by-predictor interaction test."""

    NOT_TESTED = "not_tested"
    SUPPORTED = "supported"
    NULL = "null"
    INCONCLUSIVE = "inconclusive"
    UNSUPPORTED = "unsupported"


class InteractionPValueRole(StrEnum):
    """Inferential question answered by a reported interaction p-value."""

    DEPARTURE_FROM_NULL = "departure_from_null"
    EQUIVALENCE_TO_NULL = "equivalence_to_null"


_UNINFORMATIVE_BIOMARKER_TEXT = {
    "missing",
    "n a",
    "na",
    "none",
    "not applicable",
    "not assessed",
    "not available",
    "not collected",
    "not done",
    "not evaluable",
    "not measured",
    "not reported",
    "not tested",
    "null",
    "no data",
    "other",
    "pending",
    "quantity not sufficient",
    "insufficient tissue",
    "tbd",
    "undetermined",
    "unknown",
    "unreported",
    "unspecified",
}

_UNINFORMATIVE_CLAIM_PATTERN = re.compile(
    r"\b(?:missing|none|null|undetermined|unknown|unreported|unspecified|"
    r"unavailable|pending|tbd)\b|\bn\s+a\b|\bna\b|\bno\s+data\b|"
    r"\binsufficient\s+(?:material|sample|tissue)\b|"
    r"\bquantity\s+not\s+sufficient\b|\bnot\s+(?:applicable|assessed|"
    r"available|collected|determined|done|evaluable|measured|reported|"
    r"specified|tested)\b"
)

_ACTIVE_EXPOSURE_CURIE_PATTERN = re.compile(
    r"^(?:CHEBI|DRON|DRUGBANK|NCIT|RXNORM|SYN):[A-Za-z0-9][A-Za-z0-9._-]*$",
    flags=re.IGNORECASE,
)
_EXPOSURE_SOURCE_PREFIX = {
    "chebi": "chebi",
    "dron": "dron",
    "drugbank": "drugbank",
    "ncit": "ncit",
    "rxnorm": "rxnorm",
    "synthetic": "syn",
}

_STABLE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
_ONTOLOGY_CURIE_PATTERN = re.compile(
    r"^[A-Za-z][A-Za-z0-9._-]*:[A-Za-z0-9][A-Za-z0-9._-]*$"
)
_ONTOLOGY_SOURCE_PREFIX = {
    **_EXPOSURE_SOURCE_PREFIX,
    "diseaseontology": "doid",
    "doid": "doid",
    "efo": "efo",
    "hgnc": "hgnc",
    "mondo": "mondo",
    "ncbigene": "ncbigene",
    "nci thesaurus": "ncit",
    "oncotree": "oncotree",
    "syntheticgene": "syngene",
}


def _normalized_claim_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _is_uninformative_claim_text(value: str, *, include_other: bool = False) -> bool:
    normalized = _normalized_claim_text(value)
    return (
        not normalized
        or bool(_UNINFORMATIVE_CLAIM_PATTERN.search(normalized))
        or (include_other and bool(re.search(r"\bother\b", normalized)))
    )


def _canonical_exposure_id_set(
    value: str,
    *,
    field_name: str,
    allow_empty: bool = False,
) -> set[str]:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must encode a JSON array") from exc
    if not isinstance(parsed, list) or (not parsed and not allow_empty):
        qualifier = "a JSON array" if allow_empty else "a non-empty JSON array"
        raise ValueError(f"{field_name} must encode {qualifier}")
    if not all(isinstance(item, str) and item.strip() for item in parsed):
        raise ValueError(f"{field_name} must contain non-empty string IDs")
    if any(_is_uninformative_claim_text(item) for item in parsed):
        raise ValueError(f"{field_name} cannot contain placeholder exposure IDs")
    if any(
        not _ACTIVE_EXPOSURE_CURIE_PATTERN.fullmatch(item.strip()) for item in parsed
    ):
        raise ValueError(
            f"{field_name} must contain controlled CURIE identifiers from the "
            "supported treatment ontologies"
        )
    normalized = {
        " ".join(unicodedata.normalize("NFKC", item).casefold().split())
        for item in parsed
    }
    if len(normalized) != len(parsed):
        raise ValueError(f"{field_name} cannot contain duplicate exposure IDs")
    return normalized


def _validate_required_claim_texts(**values: str | None) -> None:
    """Reject missingness placeholders in identity, provenance, and claim fields."""

    for field_name, value in values.items():
        if value is None or _is_uninformative_claim_text(value):
            raise ValueError(f"{field_name} must be an informative observed value")


def _validate_stable_identifier(value: str, *, field_name: str) -> None:
    if _is_uninformative_claim_text(value) or not _STABLE_IDENTIFIER_PATTERN.fullmatch(
        value.strip()
    ):
        raise ValueError(f"{field_name} must be a stable non-placeholder identifier")


def _validate_versioned_ontology_identifier(
    value: str,
    *,
    ontology_name: str,
    ontology_version: str,
    field_name: str,
) -> None:
    _validate_stable_identifier(value, field_name=field_name)
    _validate_required_claim_texts(
        **{
            f"{field_name}_ontology_name": ontology_name,
            f"{field_name}_ontology_version": ontology_version,
        }
    )
    if not _ONTOLOGY_CURIE_PATTERN.fullmatch(value.strip()):
        raise ValueError(f"{field_name} must be a controlled CURIE identifier")
    observed_prefix = value.split(":", 1)[0].casefold()
    source_key = _normalized_claim_text(ontology_name)
    compact_source_key = source_key.replace(" ", "")
    expected_prefix = _ONTOLOGY_SOURCE_PREFIX.get(
        source_key,
        _ONTOLOGY_SOURCE_PREFIX.get(compact_source_key, compact_source_key),
    )
    if observed_prefix != expected_prefix:
        raise ValueError(
            f"{field_name} CURIE prefix disagrees with its versioned ontology"
        )


def _reject_boolean_numeric_fields(
    value: Any,
    *,
    field_names: set[str],
) -> Any:
    if isinstance(value, dict):
        boolean_fields = sorted(
            field_name
            for field_name in field_names
            if isinstance(value.get(field_name), (bool, np.bool_))
        )
        if boolean_fields:
            raise ValueError(
                "scientific numeric fields cannot be boolean: "
                + ", ".join(boolean_fields)
            )
    return value


def _require_boolean_fields(
    value: Any,
    *,
    field_names: set[str],
) -> Any:
    """Reject Pydantic's permissive coercion for scientific attestations."""

    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    invalid: list[str] = []
    for field_name in sorted(field_names):
        observed = value.get(field_name)
        if observed is None:
            continue
        if type(observed) is bool:
            continue
        if isinstance(observed, np.bool_):
            normalized[field_name] = bool(observed)
            continue
        invalid.append(field_name)
    if invalid:
        raise ValueError(
            "scientific boolean fields require literal booleans: " + ", ".join(invalid)
        )
    return normalized


def _validate_v1_monotherapy_regimen(
    *,
    treatment_name: str,
    regimen_name: str,
    component_relation: RegimenComponentRelation,
    field_prefix: str,
) -> None:
    if component_relation != RegimenComponentRelation.FIXED_ALL_OF:
        raise ValueError(
            f"v1 verified {field_prefix} requires fixed_all_of component relation"
        )
    treatment_tokens = set(re.findall(r"\w+", _normalized_claim_text(treatment_name)))
    regimen_tokens = set(re.findall(r"\w+", _normalized_claim_text(regimen_name)))
    if not treatment_tokens or not (
        treatment_tokens <= regimen_tokens
        and regimen_tokens - treatment_tokens <= {"alone", "monotherapy"}
    ):
        raise ValueError(
            f"v1 verified {field_prefix} label must be canonical monotherapy"
        )


def _predictor_measurement_is_compatible(
    feature_type: BiomarkerFeatureType,
    *,
    measurement_type: str,
    measurement_platform: str | None,
) -> bool:
    observed = _normalized_claim_text(
        " ".join(value for value in (measurement_type, measurement_platform) if value)
    )
    required_terms = {
        BiomarkerFeatureType.GENOMIC_MUTATION: {
            "dna",
            "genomic",
            "mutation",
            "variant",
            "wes",
            "wgs",
        },
        BiomarkerFeatureType.COPY_NUMBER: {"copy number", "cnv", "cna"},
        BiomarkerFeatureType.FUSION: {"fusion", "rearrangement"},
        BiomarkerFeatureType.RNA_EXPRESSION: {
            "rna",
            "rna seq",
            "rnaseq",
            "transcript",
            "transcriptome",
            "gene expression",
        },
        BiomarkerFeatureType.PROTEIN_EXPRESSION: {
            "protein",
            "ihc",
            "immunohistochemistry",
            "cytometry",
        },
        BiomarkerFeatureType.EPIGENETIC: {
            "epigenetic",
            "methylation",
            "chromatin",
        },
        BiomarkerFeatureType.GENOMIC_SIGNATURE: {
            "signature",
            "hrd",
            "genomic scar",
        },
        BiomarkerFeatureType.FUNCTIONAL_STATUS: {
            "functional",
            "activity",
        },
    }
    terms = required_terms.get(feature_type)
    if terms is None:
        return False

    def contains_term(term: str) -> bool:
        pattern = r"(?<!\w)" + r"\s+".join(map(re.escape, term.split())) + r"(?!\w)"
        return bool(re.search(pattern, observed))

    if not any(contains_term(term) for term in terms):
        return False
    conflicting_terms = {
        BiomarkerFeatureType.GENOMIC_MUTATION: {
            "rna",
            "rna seq",
            "rnaseq",
            "transcript",
            "transcriptome",
            "gene expression",
            "protein",
            "ihc",
            "immunohistochemistry",
            "cytometry",
            "copy number",
            "cnv",
            "cna",
            "fusion",
            "rearrangement",
        },
        BiomarkerFeatureType.COPY_NUMBER: {
            "rna",
            "rna seq",
            "rnaseq",
            "transcript",
            "transcriptome",
            "gene expression",
            "protein",
            "ihc",
            "immunohistochemistry",
            "cytometry",
            "mutation",
            "variant",
            "fusion",
            "rearrangement",
        },
        BiomarkerFeatureType.FUSION: {
            "copy number",
            "cnv",
            "cna",
            "gene expression",
            "protein",
            "ihc",
            "immunohistochemistry",
            "cytometry",
            "epigenetic",
            "methylation",
        },
        BiomarkerFeatureType.RNA_EXPRESSION: {
            "protein",
            "ihc",
            "immunohistochemistry",
            "cytometry",
            "dna",
            "genomic mutation",
            "copy number",
            "cnv",
            "cna",
        },
        BiomarkerFeatureType.PROTEIN_EXPRESSION: {
            "rna",
            "transcript",
            "transcriptome",
            "dna",
            "genomic mutation",
            "copy number",
            "cnv",
            "cna",
        },
    }
    return not any(
        contains_term(term) for term in conflicting_terms.get(feature_type, set())
    )


def _validate_exposure_ontology_binding(
    exposures: set[str],
    *,
    source: str,
    version: str,
    field_name: str,
) -> None:
    _validate_required_claim_texts(
        active_exposure_identifier_source=source,
        active_exposure_identifier_version=version,
    )
    source_key = _normalized_claim_text(source).replace(" ", "")
    expected_prefix = _EXPOSURE_SOURCE_PREFIX.get(source_key)
    if expected_prefix is None:
        raise ValueError(
            f"{field_name} uses an unsupported active-exposure ontology source"
        )
    if any(value.split(":", 1)[0] != expected_prefix for value in exposures):
        raise ValueError(
            f"{field_name} CURIE prefixes disagree with the versioned ontology source"
        )


def _interaction_null_value(scale: InteractionEffectScale) -> float:
    if scale in {
        InteractionEffectScale.ADDITIVE_COEFFICIENT,
        InteractionEffectScale.DIFFERENCE_IN_EFFECT,
    }:
        return 0.0
    return 1.0


def _normalize_biomarker_contract_text(
    value: str,
    *,
    preserve_state_sign: bool,
) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    if preserve_state_sign:
        for symbol, token in (
            ("!=", " statenotequal "),
            ("≥", " stategreaterorequal "),
            ("≤", " statelessorequal "),
            ("≠", " statenotequal "),
            ("±", " stateplusminus "),
            ("≈", " stateapproximatelyequal "),
            ("~", " stateapproximately "),
            (">", " stategreater "),
            ("<", " stateless "),
            ("=", " stateequal "),
            ("↑", " stateup "),
            ("↓", " statedown "),
        ):
            normalized = normalized.replace(symbol, token)
        normalized = normalized.replace("+", " stateplus ")
        normalized = re.sub(
            r"(?<!\w)[\-\N{MINUS SIGN}\N{EN DASH}\N{EM DASH}]\s*(?=\d)"
            r"|(?<!\w)[\-\N{MINUS SIGN}\N{EN DASH}\N{EM DASH}](?!\w)"
            r"|(?<=\w)[\-\N{MINUS SIGN}\N{EN DASH}\N{EM DASH}](?=$|[^\w])",
            " stateminus ",
            normalized,
        )
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _validate_biomarker_bundle(
    *,
    biomarker_context: str | None,
    feature_type: BiomarkerFeatureType | None,
    state: str | None,
    specimen_type: str | None,
    measurement_timepoint: MolecularMeasurementTimepoint | None,
    informative_verified: bool | None,
    observation_status: BiomarkerAxisObservationStatus | None,
) -> None:
    """Require an explicit curator decision before typed axes can be exact."""

    fields = (
        biomarker_context,
        feature_type,
        state,
        specimen_type,
        measurement_timepoint,
    )
    any_present = any(value is not None for value in fields)
    if any_present and not all(value is not None for value in fields):
        raise ValueError(
            "biomarker term, feature type, state, specimen, and measurement "
            "timepoint must be supplied together"
        )
    if not any_present:
        if informative_verified is not None or observation_status is not None:
            raise ValueError(
                "biomarker observation status/attestation requires a biomarker bundle"
            )
        return
    if informative_verified is None or observation_status is None:
        raise ValueError(
            "a biomarker bundle requires explicit observation status and "
            "biomarker_axes_informative_verified attestation"
        )
    if observation_status != BiomarkerAxisObservationStatus.OBSERVED:
        if informative_verified is not False:
            raise ValueError(
                "a non-observed biomarker bundle cannot be marked informative"
            )
        return
    if informative_verified is not True:
        return
    normalized_text = {
        _normalize_biomarker_contract_text(
            value,
            preserve_state_sign=preserve_state_sign,
        )
        for value, preserve_state_sign in (
            (biomarker_context, True),
            (state, True),
            (specimen_type, True),
        )
        if value is not None
    }
    if (
        feature_type == BiomarkerFeatureType.OTHER
        or measurement_timepoint == MolecularMeasurementTimepoint.UNKNOWN
        or "" in normalized_text
        or normalized_text & _UNINFORMATIVE_BIOMARKER_TEXT
        or any(
            bool(_UNINFORMATIVE_CLAIM_PATTERN.search(_normalized_claim_text(value)))
            or bool(re.search(r"\bother\b", _normalized_claim_text(value)))
            for value in (biomarker_context, state, specimen_type)
            if value is not None
        )
    ):
        raise ValueError(
            "biomarker axes marked informative cannot contain unknown, other, or "
            "unassessed values"
        )


class PatientAssociationInterpretation(StrEnum):
    PREDICTIVE_INTERACTION = "predictive_interaction"
    INTERACTION_TESTED_NULL = "interaction_tested_null"
    INTERACTION_TESTED_INCONCLUSIVE = "interaction_tested_inconclusive"
    INTERACTION_TESTED_UNSUPPORTED = "interaction_tested_unsupported"
    TREATED_COHORT_ASSOCIATION = "treated_cohort_association"
    PROGNOSTIC_ONLY = "prognostic_only"
    PHARMACODYNAMIC = "pharmacodynamic"
    ACQUIRED_RESISTANCE = "acquired_resistance"
    ON_TREATMENT_ASSOCIATION = "on_treatment_association"
    POST_PROGRESSION_ASSOCIATION = "post_progression_association"
    ELIGIBILITY_ONLY = "eligibility_only"
    DESCRIPTIVE_ONLY = "descriptive_only"
    UNRESOLVED = "unresolved"


class TrialInterventionMatch(StrEnum):
    EXACT_CANONICAL = "exact_canonical"
    EXPLICIT_ALIAS = "explicit_alias"
    EXPLICIT_COMPONENT = "explicit_component"
    DECLARED_CLASS_TERM = "declared_class_term"
    NO_STRUCTURED_MATCH = "no_structured_match"


class TrialDiseaseMatch(StrEnum):
    EXPLICIT_SUBTYPE_TERM = "explicit_subtype_term"
    EXPLICIT_SUBTYPE_ALIAS = "explicit_subtype_alias"
    CANCER_TYPE_TERM_ONLY = "cancer_type_term_only"
    CANCER_ENTITY_ALIAS = "cancer_entity_alias"
    DECLARED_ANCESTOR_TERM = "declared_ancestor_term"
    NO_STRUCTURED_MATCH = "no_structured_match"


class TrialBiomarkerMatch(StrEnum):
    EXPLICIT_STRUCTURED_TERM = "explicit_structured_term"
    NOT_REPORTED_IN_STRUCTURED_TERMS = "not_reported_in_structured_terms"
    NOT_REQUESTED = "not_requested"


class TrialRegimenRelation(StrEnum):
    NO_ADDITIONAL_ACTIVE_AGENT_LISTED = "no_additional_active_agent_listed"
    ADDITIONAL_ACTIVE_AGENT_LISTED = "additional_active_agent_listed"
    UNRESOLVED = "unresolved"


class ImmuneScreenCategory(StrEnum):
    """Phenotype strata adapted from, but not restricted to, ICRAFT."""

    CELL_VIABILITY_PROLIFERATION = "cell_viability_proliferation"
    MARKER_EXPRESSION = "marker_expression"
    COCULTURE_IMMUNE_KILLING = "coculture_immune_killing"
    IMMUNE_CELL_FUNCTION = "immune_cell_function"
    IN_VIVO_SELECTION = "in_vivo_selection"
    OTHER = "other"


class AssayConsequence(StrEnum):
    DRUG_RESISTANCE = "drug_resistance"
    DRUG_SENSITIZATION = "drug_sensitization"
    TUMOR_IMMUNE_ESCAPE = "tumor_immune_escape"
    TUMOR_IMMUNE_SENSITIZATION = "tumor_immune_sensitization"
    IMMUNE_EFFECTOR_GAIN = "immune_effector_gain"
    IMMUNE_EFFECTOR_LOSS = "immune_effector_loss"
    IMMUNE_FITNESS_GAIN = "immune_fitness_gain"
    IMMUNE_FITNESS_LOSS = "immune_fitness_loss"
    MARKER_INCREASED = "marker_increased"
    MARKER_DECREASED = "marker_decreased"
    IN_VIVO_POSITIVE_SELECTION = "in_vivo_positive_selection"
    IN_VIVO_NEGATIVE_SELECTION = "in_vivo_negative_selection"
    NEUTRAL = "neutral"
    AMBIGUOUS = "ambiguous"


ALLOWED_CONSEQUENCES_BY_SCREEN_CATEGORY = {
    ImmuneScreenCategory.CELL_VIABILITY_PROLIFERATION: {
        AssayConsequence.DRUG_RESISTANCE,
        AssayConsequence.DRUG_SENSITIZATION,
        AssayConsequence.IMMUNE_FITNESS_GAIN,
        AssayConsequence.IMMUNE_FITNESS_LOSS,
    },
    ImmuneScreenCategory.MARKER_EXPRESSION: {
        AssayConsequence.MARKER_INCREASED,
        AssayConsequence.MARKER_DECREASED,
    },
    ImmuneScreenCategory.COCULTURE_IMMUNE_KILLING: {
        AssayConsequence.TUMOR_IMMUNE_ESCAPE,
        AssayConsequence.TUMOR_IMMUNE_SENSITIZATION,
        AssayConsequence.IMMUNE_EFFECTOR_GAIN,
        AssayConsequence.IMMUNE_EFFECTOR_LOSS,
    },
    ImmuneScreenCategory.IMMUNE_CELL_FUNCTION: {
        AssayConsequence.IMMUNE_EFFECTOR_GAIN,
        AssayConsequence.IMMUNE_EFFECTOR_LOSS,
        AssayConsequence.IMMUNE_FITNESS_GAIN,
        AssayConsequence.IMMUNE_FITNESS_LOSS,
    },
    ImmuneScreenCategory.IN_VIVO_SELECTION: {
        AssayConsequence.IN_VIVO_POSITIVE_SELECTION,
        AssayConsequence.IN_VIVO_NEGATIVE_SELECTION,
    },
    ImmuneScreenCategory.OTHER: set(),
}


class NativeEffectDirection(StrEnum):
    """Direction in the source's native, non-display-inverted contrast."""

    ENRICHED = "enriched"
    DEPLETED = "depleted"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class EndpointPolarity(StrEnum):
    """Whether enrichment or depletion is favorable for antitumor activity."""

    ENRICHMENT_IS_FAVORABLE = "enrichment_is_favorable"
    DEPLETION_IS_FAVORABLE = "depletion_is_favorable"
    UNKNOWN = "unknown"


class AntitumorDirection(StrEnum):
    FAVORABLE = "favorable"
    UNFAVORABLE = "unfavorable"
    NEUTRAL = "neutral"
    DISCORDANT = "discordant"
    UNKNOWN = "unknown"


class DirectionMappingStatus(StrEnum):
    EXACT = "exact"
    CONDITIONAL = "conditional"
    UNRESOLVED = "unresolved"


EXACT_DIRECTION_RULE_BY_POLARITY = {
    EndpointPolarity.ENRICHMENT_IS_FAVORABLE: ("native_enrichment_is_favorable_v1"),
    EndpointPolarity.DEPLETION_IS_FAVORABLE: ("native_depletion_is_favorable_v1"),
}

ICRAFT_CRISPRA_DISPLAY_TRANSFORMATION_ID = "icraft_crispra_display_sign_inversion_v1"


class SourceEffectSemantics(StrEnum):
    NATIVE = "native"
    ICRAFT_KO_EQUIVALENT_DISPLAY = "icraft_ko_equivalent_display"
    OTHER = "other"


class RawEffectSignSemantics(StrEnum):
    POSITIVE_IS_ENRICHMENT = "positive_is_enrichment"
    POSITIVE_IS_DEPLETION = "positive_is_depletion"
    UNSIGNED_OR_NOT_APPLICABLE = "unsigned_or_not_applicable"


def _is_lfc_metric_label(value: str) -> bool:
    normalized = "".join(
        character for character in value.casefold() if character.isalnum()
    )
    return any(
        token in normalized
        for token in ("lfc", "logfc", "logfoldchange", "log2foldchange")
    )


class OrthologyMappingStatus(StrEnum):
    NOT_NEEDED = "not_needed"
    ONE_TO_ONE = "one_to_one"
    AMBIGUOUS = "ambiguous"
    UNMAPPED = "unmapped"


class RankListCompleteness(StrEnum):
    FULL_RANKED_LIST = "full_ranked_list"
    TOP_HITS_ONLY = "top_hits_only"
    UNKNOWN = "unknown"


class InputDataLevel(StrEnum):
    FASTQ = "fastq"
    RAW_COUNTS = "raw_counts"
    NORMALIZED_COUNTS = "normalized_counts"
    DERIVED_GENE_SCORES = "derived_gene_scores"
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


class ReviewEvidenceLevel(StrEnum):
    """Per-gene evidence level extracted from one full-text review."""

    CANDIDATE_V3 = "candidate_v3"
    CANDIDATE_V2 = "candidate_v2"
    CANDIDATE_V1 = "candidate_v1"
    NONQUALIFYING = "nonqualifying"
    NOT_ANNOTATED = "not_annotated"


class ReviewComparisonStatus(StrEnum):
    """Relationship between two independently produced review records."""

    PROVISIONAL_AGREEMENT = "provisional_agreement"
    LABEL_DISAGREEMENT = "label_disagreement"
    SINGLE_CURATOR_ONLY = "single_curator_only"


class AdjudicationDecisionDisposition(StrEnum):
    """Human disposition for one immutable review-comparison item."""

    RELEASE_VALIDATION_EVENT = "release_validation_event"
    NO_QUALIFYING_EVENT = "no_qualifying_event"
    DEFER_UNRESOLVED = "defer_unresolved"


class FollowupRosterStatus(StrEnum):
    """How completely a source reports the genes selected for follow-up."""

    COMPLETE_FOLLOWUP_ROSTER = "complete_followup_roster"
    PARTIAL_EXPLICIT_TESTED_ROSTER = "partial_explicit_tested_roster"
    POSITIVE_ONLY_OR_UNCLEAR = "positive_only_or_unclear"


class RunInclusionStatus(StrEnum):
    INCLUDED_DRUG_CONTRAST = "included_drug_contrast"
    EXCLUDED_OTHER_SCREEN = "excluded_other_screen"


class MappingEvidence(StrEnum):
    REPOSITORY_EXPLICIT = "repository_explicit"
    ARTICLE_SUPPORTED = "article_supported"
    NOT_APPLICABLE = "not_applicable"


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


class RunAccessionInventoryRecord(StrictRecord):
    """Pinned repository inventory used to prove that a run map is complete."""

    primary_key = ("run_accession",)

    bioproject_accession: str = Field(min_length=1)
    study_accession: str = Field(min_length=1)
    run_accession: str = Field(min_length=1)
    experiment_accession: str = Field(min_length=1)
    sample_accession: str = Field(min_length=1)
    secondary_sample_accession: str = Field(min_length=1)
    source_sample_id: str = Field(min_length=1)
    repository_sample_alias: str = Field(min_length=1)
    repository_screen_group: str = Field(min_length=1)
    library_strategy: str = Field(min_length=1)
    library_source: str = Field(min_length=1)
    library_selection: str = Field(min_length=1)
    library_layout: str = Field(min_length=1)
    instrument_model: str = Field(min_length=1)
    donor_id: str = Field(min_length=1)
    phenotype_bin: Literal["dividing", "nondividing"]
    repository_metadata_url: HttpUrl
    retrieved_date: date


class RunContrastScopeRecord(StrictRecord):
    """Curated rule linking a repository group to one contrast or exclusion."""

    primary_key = ("scope_rule_id",)
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "required": ["inclusion_status"],
                        "properties": {
                            "inclusion_status": {"const": "included_drug_contrast"}
                        },
                    },
                    "then": {
                        "required": [
                            "screen_id",
                            "contrast_id",
                            "condition_role",
                            "treatment_name",
                        ],
                        "properties": {
                            "screen_id": {"type": "string", "minLength": 1},
                            "contrast_id": {"type": "string", "minLength": 1},
                            "condition_role": {"enum": ["control", "treatment"]},
                            "treatment_name": {"type": "string", "minLength": 1},
                            "treatment_mapping_evidence": {
                                "enum": [
                                    "repository_explicit",
                                    "article_supported",
                                ]
                            },
                        },
                    },
                },
                {
                    "if": {
                        "required": ["inclusion_status"],
                        "properties": {
                            "inclusion_status": {"const": "excluded_other_screen"}
                        },
                    },
                    "then": {
                        "properties": {
                            "screen_id": {"type": "null"},
                            "contrast_id": {"type": "null"},
                            "condition_role": {"type": "null"},
                            "treatment_name": {"type": "null"},
                            "treatment_mapping_evidence": {"const": "not_applicable"},
                        }
                    },
                },
            ],
            "x-semantic-rules": [
                "Contrast inclusion is a curated decision, not repository truth.",
                "Article-supported and repository-explicit mappings remain distinct.",
            ],
        },
    )

    scope_rule_id: str = Field(min_length=1)
    study_accession: str = Field(min_length=1)
    repository_screen_group: str = Field(min_length=1)
    inclusion_status: RunInclusionStatus
    screen_id: str | None = None
    contrast_id: str | None = None
    condition_role: Literal["control", "treatment"] | None = None
    treatment_name: str | None = None
    treatment_mapping_evidence: MappingEvidence
    evidence_url: HttpUrl
    source_locator: str = Field(min_length=1)
    assessed_date: date

    @model_validator(mode="after")
    def scope_is_internally_consistent(self) -> RunContrastScopeRecord:
        linked_fields = (
            self.screen_id,
            self.contrast_id,
            self.condition_role,
            self.treatment_name,
        )
        if self.inclusion_status == RunInclusionStatus.INCLUDED_DRUG_CONTRAST:
            if any(value is None for value in linked_fields):
                raise ValueError(
                    "included scope rules require screen, contrast, role, and "
                    "treatment mappings"
                )
            if self.treatment_mapping_evidence == MappingEvidence.NOT_APPLICABLE:
                raise ValueError("included scope rules require mapping evidence")
        else:
            if any(value is not None for value in linked_fields):
                raise ValueError("excluded scope rules cannot link to the contrast")
            if self.treatment_mapping_evidence != MappingEvidence.NOT_APPLICABLE:
                raise ValueError(
                    "excluded scope rules require "
                    "treatment_mapping_evidence=not_applicable"
                )
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
    candidate_v3_genes: str | None = Field(
        default=None, pattern=r"^[^|\s]+(?:\|[^|\s]+)*$"
    )
    candidate_v2_genes: str | None = Field(
        default=None, pattern=r"^[^|\s]+(?:\|[^|\s]+)*$"
    )
    candidate_v1_genes: str | None = Field(
        default=None, pattern=r"^[^|\s]+(?:\|[^|\s]+)*$"
    )
    nonqualifying_validation_genes: str | None = Field(
        default=None, pattern=r"^[^|\s]+(?:\|[^|\s]+)*$"
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


class ReviewComparisonRecord(StrictRecord):
    """Gene-level comparison of two reviews, never a released label.

    The record intentionally has no final label or benchmark-readiness field.
    Even exact agreement remains provisional until a named human adjudicator
    verifies the source evidence and creates a separate validation event.
    """

    primary_key = ("comparison_id",)
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
        json_schema_extra={
            "allOf": [
                {
                    "properties": {
                        "human_adjudication_required": {"const": True},
                    }
                }
            ],
            "x-semantic-rules": [
                "Primary and secondary review identifiers must differ.",
                "comparison_status is derived from the two evidence levels.",
                "Both evidence levels cannot be not_annotated.",
                "This record cannot establish a benchmark label.",
            ],
        },
    )

    comparison_id: str = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    queue_id: str = Field(min_length=1)
    queue_rank: int = Field(ge=1)
    screen_id: str = Field(min_length=1)
    external_screen_id: str = Field(min_length=1)
    gene_symbol: str = Field(min_length=1)
    primary_review_id: str = Field(min_length=1)
    secondary_review_id: str = Field(min_length=1)
    primary_evidence_level: ReviewEvidenceLevel
    secondary_evidence_level: ReviewEvidenceLevel
    comparison_status: ReviewComparisonStatus
    primary_source_locator: str = Field(min_length=1)
    secondary_source_locator: str = Field(min_length=1)
    human_adjudication_required: bool = True
    assessed_date: date
    notes: str | None = None

    @model_validator(mode="after")
    def comparison_is_internally_consistent(self) -> ReviewComparisonRecord:
        if self.human_adjudication_required is not True:
            raise ValueError("review comparisons require human adjudication")
        if self.primary_review_id == self.secondary_review_id:
            raise ValueError("primary and secondary review IDs must differ")
        absent = ReviewEvidenceLevel.NOT_ANNOTATED
        if (
            self.primary_evidence_level == absent
            and self.secondary_evidence_level == absent
        ):
            raise ValueError("both evidence levels cannot be not_annotated")
        if self.primary_evidence_level == self.secondary_evidence_level:
            expected = ReviewComparisonStatus.PROVISIONAL_AGREEMENT
        elif absent in {
            self.primary_evidence_level,
            self.secondary_evidence_level,
        }:
            expected = ReviewComparisonStatus.SINGLE_CURATOR_ONLY
        else:
            expected = ReviewComparisonStatus.LABEL_DISAGREEMENT
        if self.comparison_status != expected:
            raise ValueError("comparison_status does not match the two evidence levels")
        return self


class AdjudicationPacketRecord(StrictRecord):
    """Read-only evidence card for one pending human decision.

    The contract deliberately has no final label or disposition field.  It is
    a checksum-bound prompt, not a released validation outcome.
    """

    primary_key = ("packet_item_id",)

    packet_item_id: str = Field(min_length=1)
    packet_id: str = Field(min_length=1)
    comparison_id: str = Field(min_length=1)
    comparison_row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_dual_review_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    batch_id: str = Field(min_length=1)
    queue_id: str = Field(min_length=1)
    queue_rank: int = Field(ge=1)
    screen_id: str = Field(min_length=1)
    external_screen_id: str = Field(min_length=1)
    gene_symbol: str = Field(min_length=1)
    source_family_id: str = Field(min_length=1)
    doi: str = Field(min_length=1)
    paper_url: HttpUrl
    full_text_url: HttpUrl
    supplement_url: HttpUrl
    reviewer_a_review_id: str = Field(min_length=1)
    reviewer_a_curator: str = Field(min_length=1)
    reviewer_a_row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_a_evidence_level: ReviewEvidenceLevel
    reviewer_a_source_locator: str = Field(min_length=1)
    reviewer_a_screen_model: str = Field(min_length=1)
    reviewer_a_treatment_contrast: str = Field(min_length=1)
    reviewer_a_screen_replication: str = Field(min_length=1)
    reviewer_a_notes: str | None = None
    reviewer_b_review_id: str = Field(min_length=1)
    reviewer_b_curator: str = Field(min_length=1)
    reviewer_b_row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_b_evidence_level: ReviewEvidenceLevel
    reviewer_b_source_locator: str = Field(min_length=1)
    reviewer_b_screen_model: str = Field(min_length=1)
    reviewer_b_treatment_contrast: str = Field(min_length=1)
    reviewer_b_screen_replication: str = Field(min_length=1)
    reviewer_b_notes: str | None = None
    comparison_assessed_date: date
    human_adjudication_required: bool = True

    @model_validator(mode="before")
    @classmethod
    def require_literal_boolean(cls, value: Any) -> Any:
        return _require_boolean_fields(
            value,
            field_names={"human_adjudication_required"},
        )

    @model_validator(mode="after")
    def packet_item_is_read_only_and_linked(self) -> AdjudicationPacketRecord:
        if self.human_adjudication_required is not True:
            raise ValueError("adjudication packet items require a human decision")
        if self.reviewer_a_review_id == self.reviewer_b_review_id:
            raise ValueError("packet reviewer IDs must be distinct")
        if _normalized_claim_text(self.reviewer_a_curator) == _normalized_claim_text(
            self.reviewer_b_curator
        ):
            raise ValueError("packet curator identities must be distinct")
        _validate_stable_identifier(self.packet_id, field_name="packet_id")
        _validate_stable_identifier(
            self.packet_item_id,
            field_name="packet_item_id",
        )
        return self


class AdjudicationDecisionRecord(StrictRecord):
    """Named human decision linked to one immutable packet item."""

    primary_key = ("decision_id",)

    decision_id: str = Field(min_length=1)
    packet_id: str = Field(min_length=1)
    packet_item_id: str = Field(min_length=1)
    comparison_id: str = Field(min_length=1)
    comparison_row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_a_review_id: str = Field(min_length=1)
    reviewer_a_row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_b_review_id: str = Field(min_length=1)
    reviewer_b_row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_dual_review_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    packet_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    batch_id: str = Field(min_length=1)
    screen_id: str = Field(min_length=1)
    gene_symbol: str = Field(min_length=1)
    disposition: AdjudicationDecisionDisposition
    validation_event_id: str | None = None
    validation_event_row_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    followup_roster_status: FollowupRosterStatus
    adjudicator_name: str = Field(min_length=1)
    adjudicator_id: str = Field(min_length=1)
    adjudicator_affiliation: str = Field(min_length=1)
    adjudicated_date: date
    source_evidence_reviewed_attested: bool
    independent_human_decision_attested: bool
    reviewer_identity_independence_attested: bool
    model_outputs_unseen_attested: bool
    no_automated_label_assignment_attested: bool
    conflict_of_interest_declared: bool
    conflict_of_interest_notes: str | None = None
    evidence_source_locator: str = Field(min_length=1)
    decision_rationale: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def require_literal_attestations(cls, value: Any) -> Any:
        return _require_boolean_fields(
            value,
            field_names={
                "source_evidence_reviewed_attested",
                "independent_human_decision_attested",
                "reviewer_identity_independence_attested",
                "model_outputs_unseen_attested",
                "no_automated_label_assignment_attested",
                "conflict_of_interest_declared",
            },
        )

    @model_validator(mode="after")
    def decision_is_explicit_and_attested(self) -> AdjudicationDecisionRecord:
        _validate_stable_identifier(self.decision_id, field_name="decision_id")
        _validate_stable_identifier(self.packet_id, field_name="packet_id")
        _validate_stable_identifier(
            self.packet_item_id,
            field_name="packet_item_id",
        )
        _validate_stable_identifier(
            self.adjudicator_id,
            field_name="adjudicator_id",
        )
        if not all(
            (
                self.source_evidence_reviewed_attested,
                self.independent_human_decision_attested,
                self.reviewer_identity_independence_attested,
                self.model_outputs_unseen_attested,
                self.no_automated_label_assignment_attested,
            )
        ):
            raise ValueError("all human adjudication attestations must be true")
        if self.conflict_of_interest_declared and not self.conflict_of_interest_notes:
            raise ValueError("declared conflicts require conflict_of_interest_notes")
        is_release = (
            self.disposition == AdjudicationDecisionDisposition.RELEASE_VALIDATION_EVENT
        )
        if is_release != bool(self.validation_event_id) or is_release != bool(
            self.validation_event_row_sha256
        ):
            raise ValueError(
                "release_validation_event decisions require both an event ID and "
                "its canonical row SHA-256; non-release decisions require neither"
            )
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


class RunAccessionMapRecord(StrictRecord):
    """Accession-level run map with explicit evidence for condition labels."""

    primary_key = ("run_accession",)
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "required": ["inclusion_status"],
                        "properties": {
                            "inclusion_status": {"const": "included_drug_contrast"}
                        },
                    },
                    "then": {
                        "required": [
                            "screen_id",
                            "contrast_id",
                            "condition_role",
                            "treatment_name",
                        ],
                        "properties": {
                            "screen_id": {"type": "string", "minLength": 1},
                            "contrast_id": {"type": "string", "minLength": 1},
                            "condition_role": {"enum": ["control", "treatment"]},
                            "treatment_name": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "donor_id": {"type": "string", "minLength": 1},
                            "phenotype_bin": {"enum": ["dividing", "nondividing"]},
                            "treatment_mapping_evidence": {
                                "enum": [
                                    "repository_explicit",
                                    "article_supported",
                                ]
                            },
                        },
                    },
                },
                {
                    "if": {
                        "required": ["inclusion_status"],
                        "properties": {
                            "inclusion_status": {"const": "excluded_other_screen"}
                        },
                    },
                    "then": {
                        "properties": {
                            "screen_id": {"type": "null"},
                            "contrast_id": {"type": "null"},
                            "condition_role": {"type": "null"},
                            "treatment_name": {"type": "null"},
                            "treatment_mapping_evidence": {"const": "not_applicable"},
                        }
                    },
                },
            ],
            "x-semantic-rules": [
                "Repository-explicit and article-supported condition mappings "
                "are distinct.",
                "Runs from other screens cannot be attached to the drug contrast.",
            ],
        },
    )

    map_id: str = Field(min_length=1)
    bioproject_accession: str = Field(min_length=1)
    study_accession: str = Field(min_length=1)
    run_accession: str = Field(min_length=1)
    experiment_accession: str = Field(min_length=1)
    sample_accession: str = Field(min_length=1)
    secondary_sample_accession: str = Field(min_length=1)
    source_sample_id: str = Field(min_length=1)
    repository_sample_alias: str = Field(min_length=1)
    repository_screen_group: str = Field(min_length=1)
    library_strategy: str = Field(min_length=1)
    library_source: str = Field(min_length=1)
    library_selection: str = Field(min_length=1)
    library_layout: str = Field(min_length=1)
    instrument_model: str = Field(min_length=1)
    inclusion_status: RunInclusionStatus
    screen_id: str | None = None
    contrast_id: str | None = None
    condition_role: Literal["control", "treatment"] | None = None
    treatment_name: str | None = None
    donor_id: str = Field(min_length=1)
    phenotype_bin: Literal["dividing", "nondividing"]
    treatment_mapping_evidence: MappingEvidence
    repository_metadata_url: HttpUrl
    article_url: HttpUrl
    source_locator: str = Field(min_length=1)
    retrieved_date: date
    notes: str | None = None

    @model_validator(mode="after")
    def run_mapping_is_internally_consistent(self) -> RunAccessionMapRecord:
        linked_fields = (
            self.screen_id,
            self.contrast_id,
            self.condition_role,
            self.treatment_name,
        )
        if self.inclusion_status == RunInclusionStatus.INCLUDED_DRUG_CONTRAST:
            if any(value is None for value in linked_fields):
                raise ValueError(
                    "included drug-contrast runs require screen, contrast, role, "
                    "and treatment mappings"
                )
            if self.treatment_mapping_evidence == MappingEvidence.NOT_APPLICABLE:
                raise ValueError("included drug-contrast runs require mapping evidence")
        else:
            if any(value is not None for value in linked_fields):
                raise ValueError(
                    "runs from other screens cannot be linked to the drug contrast"
                )
            if self.treatment_mapping_evidence != MappingEvidence.NOT_APPLICABLE:
                raise ValueError(
                    "excluded runs require treatment_mapping_evidence=not_applicable"
                )
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
                },
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


BENCHMARK_ADJUDICATION_STATUSES = frozenset(
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
                    "assay_adequate",
                ],
                "properties": {
                    "testing_status": {"const": "tested"},
                    "perturbation_confirmed": {"const": True},
                    "phenotype_reproduced": {"const": True},
                    "appropriate_control": {"const": True},
                    "assay_adequate": {"const": True},
                    "opposite_direction_reproduced": {"enum": [False, None]},
                    "phenotype_direction": {"enum": ["resistance", "sensitization"]},
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
                    "appropriate_control",
                ],
                "properties": {
                    "testing_status": {"const": "tested"},
                    "perturbation_confirmed": {"const": True},
                    "assay_adequate": {"const": True},
                    "phenotype_reproduced": {"const": False},
                    "appropriate_control": {"const": True},
                    "opposite_direction_reproduced": {"enum": [False, None]},
                    "phenotype_direction": {"enum": ["resistance", "sensitization"]},
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
            "if": {"properties": {"label_code": {"const": "D"}}},
            "then": {
                "required": [
                    "testing_status",
                    "perturbation_confirmed",
                    "phenotype_reproduced",
                    "opposite_direction_reproduced",
                    "appropriate_control",
                    "assay_adequate",
                ],
                "properties": {
                    "testing_status": {"const": "tested"},
                    "perturbation_confirmed": {"const": True},
                    "phenotype_reproduced": {"const": False},
                    "opposite_direction_reproduced": {"const": True},
                    "appropriate_control": {"const": True},
                    "assay_adequate": {"const": True},
                    "phenotype_direction": {"enum": ["resistance", "sensitization"]},
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
                "properties": {
                    "adjudication_status": {
                        "enum": sorted(BENCHMARK_ADJUDICATION_STATUSES)
                    }
                }
            },
            "then": {
                "required": [
                    "curator",
                    "source_family_id",
                    "evidence_available_date",
                    "review_comparison_id",
                    "adjudication_decision_id",
                    "adjudication_packet_id",
                    "adjudication_method_version",
                ]
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
    source_family_id: str | None = None
    evidence_available_date: date | None = None
    review_comparison_id: str | None = None
    adjudication_decision_id: str | None = None
    adjudication_packet_id: str | None = None
    adjudication_method_version: str | None = None
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
            if self.assay_adequate is not True:
                raise ValueError("V2/V3 require an adequate assay")
            if self.phenotype_direction not in {
                PhenotypeDirection.RESISTANCE,
                PhenotypeDirection.SENSITIZATION,
            }:
                raise ValueError("V2/V3 require a resolved phenotype direction")
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
                or self.appropriate_control is not True
                or self.phenotype_reproduced is not False
                or self.opposite_direction_reproduced is True
            ):
                raise ValueError(
                    "F0 requires a tested, confirmed perturbation, adequate assay, "
                    "appropriate control, and non-reproduced phenotype"
                )
            if self.phenotype_direction not in {
                PhenotypeDirection.RESISTANCE,
                PhenotypeDirection.SENSITIZATION,
            }:
                raise ValueError("F0 requires a resolved phenotype direction")
            if not (
                (self.independent_reagent_count or 0) >= 2
                or self.orthogonal_perturbation is True
            ):
                raise ValueError(
                    "F0 requires at least two independent reagents or an "
                    "orthogonal perturbation strategy"
                )
        if self.label_code == LabelCode.D:
            if (
                self.testing_status != TestingStatus.TESTED
                or self.perturbation_confirmed is not True
                or self.assay_adequate is not True
                or self.appropriate_control is not True
                or self.phenotype_reproduced is not False
                or self.opposite_direction_reproduced is not True
            ):
                raise ValueError(
                    "D requires confirmed testing, adequate assay and control, "
                    "with the opposite phenotype"
                )
            if self.phenotype_direction not in {
                PhenotypeDirection.RESISTANCE,
                PhenotypeDirection.SENSITIZATION,
            }:
                raise ValueError("D requires a resolved phenotype direction")
            if not (
                (self.independent_reagent_count or 0) >= 2
                or self.orthogonal_perturbation is True
            ):
                raise ValueError(
                    "D requires at least two independent reagents or an "
                    "orthogonal perturbation strategy"
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
        if self.adjudication_status in BENCHMARK_ADJUDICATION_STATUSES:
            required_lineage = {
                "curator": self.curator,
                "source_family_id": self.source_family_id,
                "evidence_available_date": self.evidence_available_date,
                "review_comparison_id": self.review_comparison_id,
                "adjudication_decision_id": self.adjudication_decision_id,
                "adjudication_packet_id": self.adjudication_packet_id,
                "adjudication_method_version": self.adjudication_method_version,
            }
            missing = sorted(
                name
                for name, value in required_lineage.items()
                if value is None or (isinstance(value, str) and not value.strip())
            )
            if missing:
                raise ValueError(
                    f"benchmark adjudication requires checksum-bound lineage: {missing}"
                )
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


class ImmuneScreenEvidenceRecord(StrictRecord):
    """Canonical comparison-by-gene evidence for immune-context reporting.

    This contract deliberately stores the native source direction separately
    from endpoint desirability. It is auxiliary evidence: it cannot create a
    validation label or enter the success model as a current-snapshot feature.
    """

    primary_key = ("evidence_id",)
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
        json_schema_extra={
            "anyOf": [
                {
                    "required": ["raw_effect"],
                    "properties": {"raw_effect": {"type": "number"}},
                },
                {
                    "required": ["source_score"],
                    "properties": {"source_score": {"type": "number"}},
                },
                {
                    "required": ["source_fdr"],
                    "properties": {"source_fdr": {"type": "number"}},
                },
                {
                    "required": ["source_rank"],
                    "properties": {"source_rank": {"type": "integer"}},
                },
            ],
            "allOf": [
                {
                    "if": {
                        "properties": {
                            "rank_list_completeness": {"const": "full_ranked_list"}
                        },
                        "required": ["rank_list_completeness"],
                    },
                    "then": {
                        "required": [
                            "rank_list_id",
                            "rank_list_sha256",
                            "source_rank",
                            "gene_universe_size",
                            "analysis_tail",
                            "rank_metric_type",
                            "rank_ordering",
                            "rank_tie_policy",
                        ],
                        "properties": {
                            "rank_list_id": {"type": "string", "minLength": 1},
                            "rank_list_sha256": {
                                "type": "string",
                                "pattern": "^[0-9a-f]{64}$",
                            },
                            "source_rank": {"type": "integer", "minimum": 1},
                            "gene_universe_size": {
                                "type": "integer",
                                "minimum": 1,
                            },
                            "analysis_tail": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "rank_metric_type": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "rank_ordering": {"enum": ["ascending", "descending"]},
                            "rank_tie_policy": {
                                "type": "string",
                                "minLength": 1,
                            },
                        },
                    },
                },
                {
                    "if": {
                        "required": ["raw_effect"],
                        "properties": {"raw_effect": {"type": "number"}},
                    },
                    "then": {
                        "required": [
                            "raw_effect_type",
                            "raw_effect_sign_semantics",
                        ],
                        "properties": {
                            "raw_effect_type": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "raw_effect_sign_semantics": {
                                "enum": [
                                    "positive_is_enrichment",
                                    "positive_is_depletion",
                                    "unsigned_or_not_applicable",
                                ]
                            },
                        },
                    },
                },
                {
                    "if": {
                        "required": ["raw_effect_type"],
                        "properties": {"raw_effect_type": {"type": "string"}},
                    },
                    "then": {
                        "required": [
                            "raw_effect",
                            "raw_effect_sign_semantics",
                        ],
                        "properties": {
                            "raw_effect": {"type": "number"},
                            "raw_effect_sign_semantics": {
                                "enum": [
                                    "positive_is_enrichment",
                                    "positive_is_depletion",
                                    "unsigned_or_not_applicable",
                                ]
                            },
                        },
                    },
                },
                {
                    "if": {
                        "required": ["raw_effect_sign_semantics"],
                        "properties": {
                            "raw_effect_sign_semantics": {
                                "enum": [
                                    "positive_is_enrichment",
                                    "positive_is_depletion",
                                    "unsigned_or_not_applicable",
                                ]
                            }
                        },
                    },
                    "then": {
                        "required": ["raw_effect", "raw_effect_type"],
                        "properties": {
                            "raw_effect": {"type": "number"},
                            "raw_effect_type": {
                                "type": "string",
                                "minLength": 1,
                            },
                        },
                    },
                },
                {
                    "if": {
                        "required": ["source_score"],
                        "properties": {"source_score": {"type": "number"}},
                    },
                    "then": {
                        "required": ["source_score_type"],
                        "properties": {
                            "source_score_type": {
                                "type": "string",
                                "minLength": 1,
                            }
                        },
                    },
                },
                {
                    "if": {
                        "required": ["source_score_type"],
                        "properties": {"source_score_type": {"type": "string"}},
                    },
                    "then": {
                        "required": ["source_score"],
                        "properties": {"source_score": {"type": "number"}},
                    },
                },
                {
                    "if": {
                        "required": ["dual_action_group_id"],
                        "properties": {"dual_action_group_id": {"type": "string"}},
                    },
                    "then": {
                        "required": ["dual_action_group_version"],
                        "properties": {
                            "dual_action_group_version": {
                                "type": "string",
                                "minLength": 1,
                            }
                        },
                    },
                },
                {
                    "if": {
                        "required": ["dual_action_group_version"],
                        "properties": {"dual_action_group_version": {"type": "string"}},
                    },
                    "then": {
                        "required": ["dual_action_group_id"],
                        "properties": {
                            "dual_action_group_id": {
                                "type": "string",
                                "minLength": 1,
                            }
                        },
                    },
                },
            ],
            "x-semantic-rules": [
                "raw_effect is always native and is never overwritten by an "
                "ICRAFT CRISPRa display inversion",
                "a full_ranked_list row requires checksum-bound rank-list ID, "
                "source rank, universe, tail, metric, ordering, and tie policy",
                "full-list completeness is verified across the complete input "
                "table before RRA; the row declaration alone is insufficient",
                "used_for_label is always false",
                "ambiguous or unmapped orthology remains annotation-only",
                "raw effect sign is interpreted only through the controlled "
                "raw_effect_sign_semantics field",
            ],
        },
    )

    evidence_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_snapshot_date: date
    external_study_id: str = Field(min_length=1)
    external_screen_id: str = Field(min_length=1)
    external_comparison_id: str = Field(min_length=1)
    source_family_id: str = Field(min_length=1)
    raw_data_family_id: str = Field(min_length=1)
    gene_symbol: str = Field(min_length=1)
    source_organism: str = Field(min_length=1)
    mapped_human_gene_symbol: str | None = None
    orthology_mapping_status: OrthologyMappingStatus
    orthology_source: str | None = None
    orthology_version: str | None = None
    perturbation_modality: PerturbationModality
    perturbed_compartment: PerturbedCompartment
    experimental_setting: ExperimentalSetting
    screen_category: ImmuneScreenCategory
    cell_model: str | None = None
    immune_cell_type: str | None = None
    cancer_type: str | None = None
    treatment: str = Field(min_length=1)
    comparator: str = Field(min_length=1)
    contrast_definition: str = Field(min_length=1)
    phenotype_endpoint: str = Field(min_length=1)
    assay_consequence: AssayConsequence
    timepoint: str = Field(min_length=1)
    recurrence_stratum_id: str = Field(min_length=1)
    dual_action_group_id: str | None = Field(default=None, min_length=1)
    dual_action_group_version: str | None = Field(default=None, min_length=1)
    native_effect_direction: NativeEffectDirection
    endpoint_polarity: EndpointPolarity
    direction_mapping_status: DirectionMappingStatus
    direction_mapping_rule: str | None = Field(default=None, min_length=1)
    raw_effect: float | None = None
    raw_effect_type: str | None = Field(default=None, min_length=1)
    raw_effect_sign_semantics: RawEffectSignSemantics | None = None
    source_score: float | None = None
    source_score_type: str | None = Field(default=None, min_length=1)
    source_fdr: float | None = Field(default=None, ge=0.0, le=1.0)
    source_rank: int | None = Field(default=None, ge=1)
    rank_list_id: str | None = Field(default=None, min_length=1)
    rank_list_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "SHA-256 of the canonical UTF-8 roster sorted by source rank as "
            "gene_symbol<TAB>source_rank<LF>"
        ),
    )
    gene_universe_size: int | None = Field(default=None, ge=1)
    analysis_tail: str | None = Field(default=None, min_length=1)
    rank_metric_type: str | None = Field(default=None, min_length=1)
    rank_ordering: Literal["ascending", "descending"] | None = Field(
        default=None,
        description=(
            "Ordering of the source metric used to derive the ordinal rank; "
            "source_rank=1 must always denote the best-ranked gene"
        ),
    )
    rank_tie_policy: str | None = Field(default=None, min_length=1)
    rank_list_completeness: RankListCompleteness
    source_effect_semantics: SourceEffectSemantics = SourceEffectSemantics.NATIVE
    published_sign_inverted: bool = False
    input_data_level: InputDataLevel = InputDataLevel.UNKNOWN
    source_url: HttpUrl
    source_locator: str = Field(min_length=1)
    available_date: date
    transformation_available_date: date
    retrieved_date: date
    transformation_id: str | None = None
    used_for_label: Literal[False] = False
    notes: str | None = None

    @model_validator(mode="after")
    def immune_evidence_is_fail_closed(self) -> ImmuneScreenEvidenceRecord:
        if self.retrieved_date < self.available_date:
            raise ValueError("retrieved_date cannot precede available_date")
        if self.source_snapshot_date > self.retrieved_date:
            raise ValueError("source_snapshot_date cannot follow retrieved_date")
        if self.transformation_available_date > self.retrieved_date:
            raise ValueError(
                "transformation_available_date cannot follow retrieved_date"
            )
        if all(
            value is None
            for value in (
                self.raw_effect,
                self.source_score,
                self.source_fdr,
                self.source_rank,
            )
        ):
            raise ValueError("immune evidence requires an effect, score, FDR, or rank")
        if (self.raw_effect is None) != (self.raw_effect_type is None):
            raise ValueError("raw_effect and raw_effect_type must be supplied together")
        if (self.raw_effect is None) != (self.raw_effect_sign_semantics is None):
            raise ValueError(
                "raw_effect and raw_effect_sign_semantics must be supplied together"
            )
        if (self.source_score is None) != (self.source_score_type is None):
            raise ValueError(
                "source_score and source_score_type must be supplied together"
            )
        if self.raw_effect is not None and self.raw_effect_sign_semantics in {
            RawEffectSignSemantics.POSITIVE_IS_ENRICHMENT,
            RawEffectSignSemantics.POSITIVE_IS_DEPLETION,
        }:
            positive_direction = (
                NativeEffectDirection.ENRICHED
                if self.raw_effect_sign_semantics
                == RawEffectSignSemantics.POSITIVE_IS_ENRICHMENT
                else NativeEffectDirection.DEPLETED
            )
            negative_direction = (
                NativeEffectDirection.DEPLETED
                if positive_direction == NativeEffectDirection.ENRICHED
                else NativeEffectDirection.ENRICHED
            )
            expected_native_direction = (
                positive_direction
                if self.raw_effect > 0
                else negative_direction
                if self.raw_effect < 0
                else NativeEffectDirection.NEUTRAL
            )
            if self.native_effect_direction != expected_native_direction:
                raise ValueError(
                    "raw effect sign conflicts with native_effect_direction under "
                    "the declared sign semantics"
                )
        if (self.dual_action_group_id is None) != (
            self.dual_action_group_version is None
        ):
            raise ValueError(
                "dual_action_group_id and dual_action_group_version must be "
                "supplied together"
            )

        is_human = self.source_organism.casefold() in {
            "human",
            "homo sapiens",
            "9606",
        }
        if is_human and self.orthology_mapping_status != (
            OrthologyMappingStatus.NOT_NEEDED
        ):
            raise ValueError("human evidence must use not_needed orthology")
        if (
            is_human
            and self.mapped_human_gene_symbol
            and self.mapped_human_gene_symbol.casefold() != self.gene_symbol.casefold()
        ):
            raise ValueError(
                "human evidence cannot be renamed through an orthology field"
            )
        if (
            not is_human
            and self.orthology_mapping_status == OrthologyMappingStatus.ONE_TO_ONE
        ):
            if not all(
                (
                    self.mapped_human_gene_symbol,
                    self.orthology_source,
                    self.orthology_version,
                )
            ):
                raise ValueError(
                    "one-to-one non-human mapping requires a human symbol and "
                    "versioned orthology source"
                )
        if not is_human and self.orthology_mapping_status == (
            OrthologyMappingStatus.NOT_NEEDED
        ):
            raise ValueError("non-human evidence requires an orthology decision")
        if (
            self.orthology_mapping_status
            in {
                OrthologyMappingStatus.AMBIGUOUS,
                OrthologyMappingStatus.UNMAPPED,
            }
            and self.mapped_human_gene_symbol
        ):
            raise ValueError("ambiguous/unmapped orthology cannot claim one human gene")

        if self.rank_list_completeness == RankListCompleteness.FULL_RANKED_LIST:
            required = {
                "rank_list_id": self.rank_list_id,
                "rank_list_sha256": self.rank_list_sha256,
                "source_rank": self.source_rank,
                "gene_universe_size": self.gene_universe_size,
                "analysis_tail": self.analysis_tail,
                "rank_metric_type": self.rank_metric_type,
                "rank_ordering": self.rank_ordering,
                "rank_tie_policy": self.rank_tie_policy,
            }
            missing = sorted(name for name, value in required.items() if value is None)
            if missing:
                raise ValueError(
                    "full_ranked_list rows require fields: " + ", ".join(missing)
                )
            if self.source_rank is not None and self.gene_universe_size is not None:
                if self.source_rank > self.gene_universe_size:
                    raise ValueError("source_rank cannot exceed gene_universe_size")

        if self.source_effect_semantics == SourceEffectSemantics.NATIVE:
            if self.published_sign_inverted:
                raise ValueError("native source semantics cannot be sign-inverted")
        elif not self.transformation_id:
            raise ValueError("non-native source semantics require transformation_id")

        if (
            self.source_effect_semantics
            == SourceEffectSemantics.ICRAFT_KO_EQUIVALENT_DISPLAY
        ):
            if self.perturbation_modality != PerturbationModality.CRISPRA:
                raise ValueError(
                    "ICRAFT KO-equivalent display semantics apply only to CRISPRa"
                )
            if not self.published_sign_inverted:
                raise ValueError(
                    "ICRAFT KO-equivalent display semantics require the inversion flag"
                )
            if self.transformation_id != ICRAFT_CRISPRA_DISPLAY_TRANSFORMATION_ID:
                raise ValueError(
                    "ICRAFT CRISPRa display semantics require the registered "
                    f"transformation {ICRAFT_CRISPRA_DISPLAY_TRANSFORMATION_ID!r}"
                )
            if self.raw_effect is None or self.source_score is None:
                raise ValueError(
                    "ICRAFT CRISPRa display semantics require paired native and "
                    "display effects"
                )
            if self.raw_effect_sign_semantics != (
                RawEffectSignSemantics.POSITIVE_IS_ENRICHMENT
            ):
                raise ValueError(
                    "ICRAFT native CRISPRa LFC requires positive_is_enrichment "
                    "sign semantics"
                )
            effect_types = (
                self.raw_effect_type or "",
                self.source_score_type or "",
            )
            if not all(_is_lfc_metric_label(value) for value in effect_types):
                raise ValueError(
                    "ICRAFT CRISPRa display inversion is defined only for paired "
                    "LFC values"
                )
            if abs(self.source_score + self.raw_effect) > 1e-12:
                raise ValueError(
                    "ICRAFT CRISPRa display LFC must be the negative native LFC"
                )
            expected_direction = (
                NativeEffectDirection.ENRICHED
                if self.raw_effect > 0
                else NativeEffectDirection.DEPLETED
                if self.raw_effect < 0
                else NativeEffectDirection.NEUTRAL
            )
            if self.native_effect_direction != expected_direction:
                raise ValueError(
                    "native CRISPRa direction conflicts with the native LFC sign"
                )

        if self.endpoint_polarity == EndpointPolarity.UNKNOWN:
            if self.direction_mapping_status != DirectionMappingStatus.UNRESOLVED:
                raise ValueError("unknown endpoint polarity must remain unresolved")
        elif self.direction_mapping_status == DirectionMappingStatus.UNRESOLVED:
            raise ValueError("known endpoint polarity cannot be marked unresolved")
        if (
            self.direction_mapping_status == DirectionMappingStatus.CONDITIONAL
            and not self.direction_mapping_rule
        ):
            raise ValueError("conditional direction mapping requires an explicit rule")

        if self.direction_mapping_status == DirectionMappingStatus.EXACT:
            expected_rule = EXACT_DIRECTION_RULE_BY_POLARITY.get(self.endpoint_polarity)
            if self.direction_mapping_rule != expected_rule:
                raise ValueError(
                    "exact direction mapping requires the registered rule "
                    f"{expected_rule!r}"
                )

        if (
            self.assay_consequence
            in {
                AssayConsequence.AMBIGUOUS,
                AssayConsequence.NEUTRAL,
            }
            and self.direction_mapping_status != DirectionMappingStatus.UNRESOLVED
        ):
            raise ValueError(
                "ambiguous/neutral assay consequences must remain unresolved"
            )

        immune_consequences = {
            AssayConsequence.IMMUNE_EFFECTOR_GAIN,
            AssayConsequence.IMMUNE_EFFECTOR_LOSS,
            AssayConsequence.IMMUNE_FITNESS_GAIN,
            AssayConsequence.IMMUNE_FITNESS_LOSS,
        }
        tumor_consequences = {
            AssayConsequence.TUMOR_IMMUNE_ESCAPE,
            AssayConsequence.TUMOR_IMMUNE_SENSITIZATION,
        }
        non_directional_consequences = {
            AssayConsequence.AMBIGUOUS,
            AssayConsequence.NEUTRAL,
        }
        allowed_consequences = (
            ALLOWED_CONSEQUENCES_BY_SCREEN_CATEGORY[self.screen_category]
            | non_directional_consequences
        )
        if self.assay_consequence not in allowed_consequences:
            raise ValueError(
                "assay consequence is incompatible with the screen category"
            )
        if (
            self.assay_consequence in immune_consequences
            and self.perturbed_compartment != PerturbedCompartment.IMMUNE_CELL
        ):
            raise ValueError(
                "immune-cell consequences require immune-cell perturbation"
            )
        if (
            self.assay_consequence in tumor_consequences
            and self.perturbed_compartment != PerturbedCompartment.TUMOR_CELL
        ):
            raise ValueError(
                "tumor-immune consequences require tumor-cell perturbation"
            )

        favorable_consequences = {
            AssayConsequence.DRUG_SENSITIZATION,
            AssayConsequence.TUMOR_IMMUNE_SENSITIZATION,
            AssayConsequence.IMMUNE_EFFECTOR_GAIN,
            AssayConsequence.IMMUNE_FITNESS_GAIN,
        }
        unfavorable_consequences = {
            AssayConsequence.DRUG_RESISTANCE,
            AssayConsequence.TUMOR_IMMUNE_ESCAPE,
            AssayConsequence.IMMUNE_EFFECTOR_LOSS,
            AssayConsequence.IMMUNE_FITNESS_LOSS,
        }
        if (
            self.direction_mapping_status == DirectionMappingStatus.EXACT
            and self.native_effect_direction
            in {NativeEffectDirection.ENRICHED, NativeEffectDirection.DEPLETED}
        ):
            mapped_favorable = (
                self.native_effect_direction == NativeEffectDirection.ENRICHED
                and self.endpoint_polarity == EndpointPolarity.ENRICHMENT_IS_FAVORABLE
            ) or (
                self.native_effect_direction == NativeEffectDirection.DEPLETED
                and self.endpoint_polarity == EndpointPolarity.DEPLETION_IS_FAVORABLE
            )
            if self.assay_consequence in favorable_consequences and not (
                mapped_favorable
            ):
                raise ValueError(
                    "assay consequence conflicts with the exact direction mapping"
                )
            if self.assay_consequence in unfavorable_consequences and (
                mapped_favorable
            ):
                raise ValueError(
                    "assay consequence conflicts with the exact direction mapping"
                )
        return self


class TreatmentDiseaseContextRecord(StrictRecord):
    """One explicit treatment/disease question for a translation report."""

    model_config = ConfigDict(
        json_schema_extra={
            "x-semantic-rules": [
                "screen_id and contrast_id must be supplied together",
                "treatment ID, ontology name, and ontology version are all-or-none",
                "disease IDs require ontology name and ontology version",
                "disease_subtype_id requires disease_subtype",
                (
                    "subtypes require explicit parent verification; verified "
                    "bindings require versioned subtype and cancer IDs with equal "
                    "parent and cancer IDs"
                ),
                (
                    "regimen name, canonical active-exposure IDs, component "
                    "relation, verification, identifier source, and identifier "
                    "version are all-or-none"
                ),
                (
                    "v1 verified regimens require a singleton exposure equal to "
                    "treatment_id and provenance matching the treatment ontology "
                    "release"
                ),
                "biomarker term, type, state, specimen, and timepoint are all-or-none",
                "typed biomarker exactness requires observed status and attestation",
                "scientific attestations require literal booleans",
                (
                    "strict registry and exact curated status require versioned "
                    "canonical treatment and cancer IDs"
                ),
            ]
        }
    )
    primary_key = ("context_id",)

    context_id: str = Field(min_length=1)
    screen_id: str | None = Field(default=None, min_length=1)
    contrast_id: str | None = Field(default=None, min_length=1)
    treatment_name: str = Field(min_length=1)
    treatment_id: str | None = Field(default=None, min_length=1)
    treatment_ontology_name: str | None = Field(default=None, min_length=1)
    treatment_ontology_version: str | None = Field(default=None, min_length=1)
    treatment_modality: InterventionModality
    regimen_name: str | None = Field(default=None, min_length=1)
    regimen_active_exposure_ids_json: str | None = Field(default=None, min_length=2)
    regimen_component_relation: RegimenComponentRelation | None = None
    regimen_active_exposures_verified: bool | None = None
    regimen_active_exposure_identifier_source: str | None = Field(
        default=None, min_length=1
    )
    regimen_active_exposure_identifier_version: str | None = Field(
        default=None, min_length=1
    )
    cancer_type: str = Field(min_length=1)
    cancer_id: str | None = Field(default=None, min_length=1)
    disease_subtype: str | None = Field(default=None, min_length=1)
    disease_subtype_id: str | None = Field(default=None, min_length=1)
    disease_subtype_parent_id: str | None = Field(default=None, min_length=1)
    disease_subtype_parent_binding_verified: bool | None = None
    disease_ontology_name: str | None = Field(default=None, min_length=1)
    disease_ontology_version: str | None = Field(default=None, min_length=1)
    stage: str | None = Field(default=None, min_length=1)
    biomarker_context: str | None = Field(default=None, min_length=1)
    biomarker_feature_type: BiomarkerFeatureType | None = None
    biomarker_state: str | None = Field(default=None, min_length=1)
    biomarker_specimen_type: str | None = Field(default=None, min_length=1)
    biomarker_measurement_timepoint: MolecularMeasurementTimepoint | None = None
    biomarker_axes_informative_verified: bool | None = None
    biomarker_axes_observation_status: BiomarkerAxisObservationStatus | None = None
    line_of_therapy: str | None = Field(default=None, min_length=1)
    screen_perturbation_modality: PerturbationModality
    perturbed_compartment: PerturbedCompartment
    screen_endpoint_category: ScreenEndpointCategory
    context_date: date
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def context_attestations_are_literal_booleans(cls, value: Any) -> Any:
        return _require_boolean_fields(
            value,
            field_names={
                "regimen_active_exposures_verified",
                "disease_subtype_parent_binding_verified",
                "biomarker_axes_informative_verified",
            },
        )

    @model_validator(mode="after")
    def context_identifiers_are_paired(self) -> TreatmentDiseaseContextRecord:
        if (self.screen_id is None) != (self.contrast_id is None):
            raise ValueError("screen_id and contrast_id must be supplied together")
        if self.disease_subtype_id and not self.disease_subtype:
            raise ValueError("disease_subtype_id requires disease_subtype")
        if self.disease_subtype:
            if self.disease_subtype_parent_binding_verified is None:
                raise ValueError(
                    "disease subtype requires explicit parent-binding verification"
                )
            if self.disease_subtype_parent_binding_verified is True:
                if not (
                    self.disease_subtype_id
                    and self.cancer_id
                    and self.disease_subtype_parent_id == self.cancer_id
                ):
                    raise ValueError(
                        "verified disease subtype requires versioned IDs and a "
                        "parent ID equal to cancer_id"
                    )
        elif (
            self.disease_subtype_id is not None
            or self.disease_subtype_parent_id is not None
            or self.disease_subtype_parent_binding_verified is not None
        ):
            raise ValueError("disease subtype metadata requires disease_subtype")
        treatment_ontology_fields = (
            self.treatment_id,
            self.treatment_ontology_name,
            self.treatment_ontology_version,
        )
        if any(treatment_ontology_fields) and not all(treatment_ontology_fields):
            raise ValueError(
                "treatment ontology ID, name, and version must be supplied together"
            )
        disease_ids_present = bool(
            self.cancer_id or self.disease_subtype_id or self.disease_subtype_parent_id
        )
        disease_ontology_fields = (
            self.disease_ontology_name,
            self.disease_ontology_version,
        )
        if disease_ids_present and not all(disease_ontology_fields):
            raise ValueError(
                "disease ontology name and version are required with disease IDs"
            )
        if not disease_ids_present and any(disease_ontology_fields):
            raise ValueError("disease ontology metadata requires a disease ID")
        if all(treatment_ontology_fields):
            _validate_versioned_ontology_identifier(
                self.treatment_id,
                ontology_name=self.treatment_ontology_name,
                ontology_version=self.treatment_ontology_version,
                field_name="treatment_id",
            )
        if disease_ids_present:
            for field_name in (
                "cancer_id",
                "disease_subtype_id",
                "disease_subtype_parent_id",
            ):
                value = getattr(self, field_name)
                if value is not None:
                    _validate_versioned_ontology_identifier(
                        value,
                        ontology_name=self.disease_ontology_name,
                        ontology_version=self.disease_ontology_version,
                        field_name=field_name,
                    )
        if (
            self.cancer_id
            and self.disease_subtype_id
            and (self.cancer_id.casefold() == self.disease_subtype_id.casefold())
        ):
            raise ValueError("disease subtype ID must differ from cancer ID")
        _validate_required_claim_texts(
            treatment_name=self.treatment_name,
            cancer_type=self.cancer_type,
        )
        _validate_required_claim_texts(
            **{
                field_name: value
                for field_name, value in (
                    ("disease_subtype", self.disease_subtype),
                    ("stage", self.stage),
                    ("line_of_therapy", self.line_of_therapy),
                )
                if value is not None
            }
        )
        regimen_fields = (
            self.regimen_name,
            self.regimen_active_exposure_ids_json,
            self.regimen_component_relation,
            self.regimen_active_exposures_verified,
            self.regimen_active_exposure_identifier_source,
            self.regimen_active_exposure_identifier_version,
        )
        if any(value is not None for value in regimen_fields) and not all(
            value is not None for value in regimen_fields
        ):
            raise ValueError(
                "regimen name, active-exposure IDs, and verification must be "
                "supplied together"
            )
        if self.regimen_name is not None:
            _validate_required_claim_texts(regimen_name=self.regimen_name)
            exposures = _canonical_exposure_id_set(
                self.regimen_active_exposure_ids_json,
                field_name="regimen_active_exposure_ids_json",
            )
            _validate_exposure_ontology_binding(
                exposures,
                source=self.regimen_active_exposure_identifier_source,
                version=self.regimen_active_exposure_identifier_version,
                field_name="regimen_active_exposure_ids_json",
            )
            if self.treatment_id and self.treatment_id.casefold() not in exposures:
                raise ValueError(
                    "regimen_active_exposure_ids_json must contain treatment_id"
                )
            if self.regimen_active_exposures_verified is True:
                if not self.treatment_id or exposures != {self.treatment_id.casefold()}:
                    raise ValueError(
                        "v1 verified regimens require exactly the canonical "
                        "treatment_id"
                    )
                if (
                    _normalized_claim_text(
                        self.regimen_active_exposure_identifier_source
                    )
                    != _normalized_claim_text(self.treatment_ontology_name)
                    or self.regimen_active_exposure_identifier_version
                    != self.treatment_ontology_version
                ):
                    raise ValueError(
                        "verified regimen exposure provenance must match the "
                        "treatment ontology release"
                    )
                _validate_v1_monotherapy_regimen(
                    treatment_name=self.treatment_name,
                    regimen_name=self.regimen_name,
                    component_relation=self.regimen_component_relation,
                    field_prefix="context regimen",
                )
            if self.regimen_component_relation in {
                RegimenComponentRelation.NONE,
                RegimenComponentRelation.UNRESOLVED,
            }:
                raise ValueError(
                    "a named regimen requires a resolved active-component relation"
                )
        _validate_biomarker_bundle(
            biomarker_context=self.biomarker_context,
            feature_type=self.biomarker_feature_type,
            state=self.biomarker_state,
            specimen_type=self.biomarker_specimen_type,
            measurement_timepoint=self.biomarker_measurement_timepoint,
            informative_verified=self.biomarker_axes_informative_verified,
            observation_status=self.biomarker_axes_observation_status,
        )
        return self


class ClinicalTrialContextRecord(StrictRecord):
    """Normalized current-snapshot ClinicalTrials.gov trial metadata.

    The match fields describe structured-text retrieval only. A registry row
    is treatment-level context and can never become gene-level evidence.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "x-semantic-rules": [
                "retrieved_at_utc must use a zero UTC offset",
                "all *_json fields must encode JSON lists",
                "registry records are current-snapshot treatment context only",
                "registry records cannot be used for gene ranking",
                (
                    "source_url is the official ClinicalTrials.gov study URL "
                    "bound to nct_id"
                ),
                "source_api_version is API v2 or explicitly unverified",
                (
                    "structured match statuses require non-empty meaningful "
                    "string term lists; no-match statuses require empty lists"
                ),
                (
                    "intervention alias, component, and class matches cannot "
                    "establish strict canonical treatment identity"
                ),
                (
                    "disease alias and ancestor matches cannot establish strict "
                    "canonical disease identity"
                ),
                "registry biomarker matches are untyped discovery context only",
                (
                    "strict subtype status additionally requires verified requested "
                    "parent binding and a separate exact parent-cancer condition"
                ),
                "registry flags require literal booleans",
                (
                    "strict status requires versioned canonical treatment and "
                    "cancer IDs in the requested context"
                ),
            ]
        }
    )
    primary_key = ("context_id", "nct_id")

    context_id: str = Field(min_length=1)
    nct_id: str = Field(pattern=r"^NCT[0-9]{8}$")
    brief_title: str = Field(min_length=1)
    official_title: str | None = None
    study_type: str = Field(min_length=1)
    overall_status: str = Field(min_length=1)
    phases_json: str = Field(min_length=2)
    conditions_json: str = Field(min_length=2)
    interventions_json: str = Field(min_length=2)
    intervention_types_json: str = Field(min_length=2)
    primary_outcomes_json: str = Field(min_length=2)
    linked_publications_json: str = Field(min_length=2)
    enrollment_count: int | None = Field(default=None, ge=0)
    enrollment_type: str | None = None
    start_date: str | None = None
    completion_date: str | None = None
    source_first_post_date: str | None = None
    source_last_update_date: str | None = None
    has_results: bool
    intervention_match: TrialInterventionMatch
    disease_match: TrialDiseaseMatch
    biomarker_match: TrialBiomarkerMatch
    intervention_match_terms_json: str = Field(min_length=2)
    disease_match_terms_json: str = Field(min_length=2)
    biomarker_match_terms_json: str = Field(min_length=2)
    regimen_relation: TrialRegimenRelation
    source_name: Literal["ClinicalTrials.gov"] = "ClinicalTrials.gov"
    source_api_major: Literal["v2"] = "v2"
    source_api_version: str = Field(min_length=1)
    source_url: HttpUrl
    retrieved_at_utc: datetime
    registry_supports_patient_level_omics: Literal[False] = False
    temporal_version_status: Literal["current_snapshot_only"] = "current_snapshot_only"
    used_for_gene_ranking: Literal[False] = False
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def registry_flags_are_literal_booleans(cls, value: Any) -> Any:
        value = _require_boolean_fields(
            value,
            field_names={
                "has_results",
                "registry_supports_patient_level_omics",
                "used_for_gene_ranking",
            },
        )
        return _reject_boolean_numeric_fields(value, field_names={"enrollment_count"})

    @model_validator(mode="after")
    def clinical_trial_context_is_fail_closed(self) -> ClinicalTrialContextRecord:
        offset = self.retrieved_at_utc.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("retrieved_at_utc must include the UTC timezone")
        source_url = self.source_url
        if (
            source_url.scheme != "https"
            or source_url.host != "clinicaltrials.gov"
            or source_url.username is not None
            or source_url.password is not None
            or source_url.path != f"/study/{self.nct_id}"
            or source_url.query is not None
            or source_url.fragment is not None
        ):
            raise ValueError(
                "source_url must be the official ClinicalTrials.gov study URL "
                "for nct_id"
            )
        if self.source_api_version != "unverified" and not re.match(
            r"^2(?:\.|$)", self.source_api_version
        ):
            raise ValueError("source_api_version must identify API v2 or be unverified")
        parsed_lists: dict[str, list[Any]] = {}
        for field_name in (
            "phases_json",
            "conditions_json",
            "interventions_json",
            "intervention_types_json",
            "primary_outcomes_json",
            "linked_publications_json",
            "intervention_match_terms_json",
            "disease_match_terms_json",
            "biomarker_match_terms_json",
        ):
            value = getattr(self, field_name)
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field_name} must contain valid JSON") from exc
            if not isinstance(parsed, list):
                raise ValueError(f"{field_name} must encode a JSON list")
            parsed_lists[field_name] = parsed
        for field_name in (
            "intervention_match_terms_json",
            "disease_match_terms_json",
            "biomarker_match_terms_json",
        ):
            if any(
                not isinstance(term, str) or not term.strip()
                for term in parsed_lists[field_name]
            ):
                raise ValueError(
                    f"{field_name} must contain meaningful non-empty string terms"
                )
        match_term_rules = (
            (
                self.intervention_match != TrialInterventionMatch.NO_STRUCTURED_MATCH,
                parsed_lists["intervention_match_terms_json"],
                "intervention_match",
            ),
            (
                self.disease_match != TrialDiseaseMatch.NO_STRUCTURED_MATCH,
                parsed_lists["disease_match_terms_json"],
                "disease_match",
            ),
            (
                self.biomarker_match == TrialBiomarkerMatch.EXPLICIT_STRUCTURED_TERM,
                parsed_lists["biomarker_match_terms_json"],
                "biomarker_match",
            ),
        )
        for match_present, match_terms, field_name in match_term_rules:
            if match_present != bool(match_terms):
                raise ValueError(
                    f"{field_name} must agree with its structured match-term list"
                )
        return self


class PreclinicalEvidenceRecord(StrictRecord):
    """Curated preclinical claim; never an automatic validation label."""

    model_config = ConfigDict(
        json_schema_extra={
            "x-semantic-rules": [
                "retrieved_date cannot precede available_date",
                "source and raw-data family identifiers are required",
                "ontology IDs require ontology name and version",
                (
                    "subtypes require explicit parent verification; verified "
                    "bindings require versioned subtype and cancer IDs with equal "
                    "parent and cancer IDs"
                ),
                (
                    "regimen name, canonical active-exposure IDs, component "
                    "relation, verification, identifier source, and identifier "
                    "version are all-or-none"
                ),
                (
                    "v1 verified regimens require a singleton exposure equal to "
                    "treatment_id and provenance matching the treatment ontology "
                    "release"
                ),
                "biomarker term, type, state, specimen, and timepoint are all-or-none",
                "typed biomarker exactness requires observed status and attestation",
                (
                    "gene-specific claims require a versioned gene ID bound to "
                    "gene_symbol and a curator identity attestation"
                ),
                "scientific attestations require literal booleans",
                (
                    "exact curated status requires versioned canonical treatment "
                    "and cancer IDs"
                ),
                "treatment_context cannot carry a gene perturbation claim",
                "non-direct claims cannot carry perturbation modality or direction",
                "direct claims require baseline and genotype-by-treatment controls",
                "compartment and endpoint category are required matching axes",
                "non-unknown direct directions require a versioned direction rule",
                (
                    "direct direction, neutral, inconclusive, and unsupported "
                    "inference statuses remain separate and require curator "
                    "adjudication"
                ),
                "neutral or discordant directions require a prespecified rule",
                "used_for_label must remain false",
            ]
        }
    )
    primary_key = ("evidence_id",)

    evidence_id: str = Field(min_length=1)
    source_study_id: str = Field(min_length=1)
    source_family_id: str = Field(min_length=1)
    raw_data_family_id: str = Field(min_length=1)
    evidence_scope: PreclinicalEvidenceScope
    claim_type: PreclinicalClaimType
    gene_symbol: str | None = Field(default=None, min_length=1)
    gene_id: str | None = Field(default=None, min_length=1)
    gene_identifier_source: str | None = Field(default=None, min_length=1)
    gene_identifier_version: str | None = Field(default=None, min_length=1)
    gene_identity_curator_verified: bool | None = None
    perturbation_modality: PerturbationModality | None = None
    perturbed_compartment: PerturbedCompartment
    endpoint_category: ScreenEndpointCategory
    phenotype_direction: PhenotypeDirection = PhenotypeDirection.UNKNOWN
    treatment_name: str = Field(min_length=1)
    treatment_id: str | None = Field(default=None, min_length=1)
    treatment_ontology_name: str | None = Field(default=None, min_length=1)
    treatment_ontology_version: str | None = Field(default=None, min_length=1)
    regimen_name: str | None = Field(default=None, min_length=1)
    regimen_active_exposure_ids_json: str | None = Field(default=None, min_length=2)
    regimen_component_relation: RegimenComponentRelation | None = None
    regimen_active_exposures_verified: bool | None = None
    regimen_active_exposure_identifier_source: str | None = Field(
        default=None, min_length=1
    )
    regimen_active_exposure_identifier_version: str | None = Field(
        default=None, min_length=1
    )
    comparator: str = Field(min_length=1)
    comparator_exposure_type: ComparatorExposureType | None = None
    comparator_active_exposure_ids_json: str | None = Field(default=None, min_length=2)
    comparator_regimen_component_relation: RegimenComponentRelation | None = None
    cancer_type: str = Field(min_length=1)
    cancer_id: str | None = Field(default=None, min_length=1)
    disease_subtype: str | None = Field(default=None, min_length=1)
    disease_subtype_id: str | None = Field(default=None, min_length=1)
    disease_subtype_parent_id: str | None = Field(default=None, min_length=1)
    disease_subtype_parent_binding_verified: bool | None = None
    disease_ontology_name: str | None = Field(default=None, min_length=1)
    disease_ontology_version: str | None = Field(default=None, min_length=1)
    biomarker_context: str | None = Field(default=None, min_length=1)
    biomarker_feature_type: BiomarkerFeatureType | None = None
    biomarker_state: str | None = Field(default=None, min_length=1)
    biomarker_specimen_type: str | None = Field(default=None, min_length=1)
    biomarker_measurement_timepoint: MolecularMeasurementTimepoint | None = None
    biomarker_axes_informative_verified: bool | None = None
    biomarker_axes_observation_status: BiomarkerAxisObservationStatus | None = None
    model_type: PreclinicalModelType
    model_name: str = Field(min_length=1)
    organism: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    outcome_text: str = Field(min_length=1)
    vehicle_or_baseline_control_present: bool | None = None
    genotype_by_treatment_tested: bool | None = None
    direction_rule_id: str | None = Field(default=None, min_length=1)
    direction_rule_version: str | None = Field(default=None, min_length=1)
    direction_inference_status: PreclinicalDirectionInferenceStatus | None = None
    direction_inference_curator_verified: bool | None = None
    neutrality_or_discordance_rule_prespecified: bool | None = None
    native_effect: float | None = None
    native_effect_type: str | None = Field(default=None, min_length=1)
    native_reference_group: str | None = Field(default=None, min_length=1)
    effect_numeric: float | None = None
    effect_type: str | None = Field(default=None, min_length=1)
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    sample_n: int | None = Field(default=None, ge=1)
    source_name: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_url: HttpUrl
    source_locator: str = Field(min_length=1)
    available_date: date
    retrieved_date: date
    used_for_label: Literal[False] = False
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def preclinical_numeric_fields_are_not_boolean(cls, value: Any) -> Any:
        value = _require_boolean_fields(
            value,
            field_names={
                "regimen_active_exposures_verified",
                "gene_identity_curator_verified",
                "disease_subtype_parent_binding_verified",
                "biomarker_axes_informative_verified",
                "vehicle_or_baseline_control_present",
                "genotype_by_treatment_tested",
                "direction_inference_curator_verified",
                "neutrality_or_discordance_rule_prespecified",
                "used_for_label",
            },
        )
        return _reject_boolean_numeric_fields(
            value,
            field_names={"native_effect", "effect_numeric", "p_value", "sample_n"},
        )

    @model_validator(mode="after")
    def preclinical_claim_is_internally_consistent(self) -> PreclinicalEvidenceRecord:
        if self.retrieved_date < self.available_date:
            raise ValueError("retrieved_date cannot precede available_date")
        if (self.effect_numeric is None) != (self.effect_type is None):
            raise ValueError("effect_numeric and effect_type must be supplied together")
        for field_name in ("native_effect", "effect_numeric"):
            value = getattr(self, field_name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        native_fields = (
            self.native_effect,
            self.native_effect_type,
            self.native_reference_group,
        )
        if any(value is not None for value in native_fields) and not all(
            value is not None for value in native_fields
        ):
            raise ValueError(
                "native_effect, native_effect_type, and native_reference_group "
                "must be supplied together"
            )
        if self.effect_type is not None:
            _validate_required_claim_texts(effect_type=self.effect_type)
        if all(value is not None for value in native_fields):
            _validate_required_claim_texts(
                native_effect_type=self.native_effect_type,
                native_reference_group=self.native_reference_group,
            )
        treatment_ontology_fields = (
            self.treatment_id,
            self.treatment_ontology_name,
            self.treatment_ontology_version,
        )
        if any(treatment_ontology_fields) and not all(treatment_ontology_fields):
            raise ValueError(
                "treatment ontology ID, name, and version must be supplied together"
            )
        if self.disease_subtype_id and not self.disease_subtype:
            raise ValueError("disease_subtype_id requires disease_subtype")
        if self.disease_subtype:
            if self.disease_subtype_parent_binding_verified is None:
                raise ValueError(
                    "disease subtype requires explicit parent-binding verification"
                )
            if self.disease_subtype_parent_binding_verified is True:
                if not (
                    self.disease_subtype_id
                    and self.cancer_id
                    and self.disease_subtype_parent_id == self.cancer_id
                ):
                    raise ValueError(
                        "verified disease subtype requires versioned IDs and a "
                        "parent ID equal to cancer_id"
                    )
        elif (
            self.disease_subtype_id is not None
            or self.disease_subtype_parent_id is not None
            or self.disease_subtype_parent_binding_verified is not None
        ):
            raise ValueError("disease subtype metadata requires disease_subtype")
        disease_ids_present = bool(
            self.cancer_id or self.disease_subtype_id or self.disease_subtype_parent_id
        )
        disease_ontology_fields = (
            self.disease_ontology_name,
            self.disease_ontology_version,
        )
        if disease_ids_present and not all(disease_ontology_fields):
            raise ValueError(
                "disease ontology name and version are required with disease IDs"
            )
        if not disease_ids_present and any(disease_ontology_fields):
            raise ValueError("disease ontology metadata requires a disease ID")
        if all(treatment_ontology_fields):
            _validate_versioned_ontology_identifier(
                self.treatment_id,
                ontology_name=self.treatment_ontology_name,
                ontology_version=self.treatment_ontology_version,
                field_name="treatment_id",
            )
        if disease_ids_present:
            for field_name in (
                "cancer_id",
                "disease_subtype_id",
                "disease_subtype_parent_id",
            ):
                value = getattr(self, field_name)
                if value is not None:
                    _validate_versioned_ontology_identifier(
                        value,
                        ontology_name=self.disease_ontology_name,
                        ontology_version=self.disease_ontology_version,
                        field_name=field_name,
                    )
        if (
            self.cancer_id
            and self.disease_subtype_id
            and (self.cancer_id.casefold() == self.disease_subtype_id.casefold())
        ):
            raise ValueError("disease subtype ID must differ from cancer ID")
        _validate_required_claim_texts(
            treatment_name=self.treatment_name,
            comparator=self.comparator,
            cancer_type=self.cancer_type,
            model_name=self.model_name,
            organism=self.organism,
            endpoint=self.endpoint,
            outcome_text=self.outcome_text,
            source_name=self.source_name,
            source_version=self.source_version,
            source_locator=self.source_locator,
        )
        if self.disease_subtype is not None:
            _validate_required_claim_texts(disease_subtype=self.disease_subtype)
        for field_name in ("source_study_id", "source_family_id"):
            _validate_stable_identifier(
                getattr(self, field_name), field_name=field_name
            )
        _validate_stable_identifier(
            self.raw_data_family_id, field_name="raw_data_family_id"
        )
        regimen_fields = (
            self.regimen_name,
            self.regimen_active_exposure_ids_json,
            self.regimen_component_relation,
            self.regimen_active_exposures_verified,
            self.regimen_active_exposure_identifier_source,
            self.regimen_active_exposure_identifier_version,
        )
        if any(value is not None for value in regimen_fields) and not all(
            value is not None for value in regimen_fields
        ):
            raise ValueError(
                "regimen name, active-exposure IDs, and verification must be "
                "supplied together"
            )
        if self.regimen_name is not None:
            _validate_required_claim_texts(regimen_name=self.regimen_name)
            exposures = _canonical_exposure_id_set(
                self.regimen_active_exposure_ids_json,
                field_name="regimen_active_exposure_ids_json",
            )
            _validate_exposure_ontology_binding(
                exposures,
                source=self.regimen_active_exposure_identifier_source,
                version=self.regimen_active_exposure_identifier_version,
                field_name="regimen_active_exposure_ids_json",
            )
            if self.treatment_id and self.treatment_id.casefold() not in exposures:
                raise ValueError(
                    "regimen_active_exposure_ids_json must contain treatment_id"
                )
            if self.regimen_active_exposures_verified is True:
                if not self.treatment_id or exposures != {self.treatment_id.casefold()}:
                    raise ValueError(
                        "v1 verified regimens require exactly the canonical "
                        "treatment_id"
                    )
                if (
                    _normalized_claim_text(
                        self.regimen_active_exposure_identifier_source
                    )
                    != _normalized_claim_text(self.treatment_ontology_name)
                    or self.regimen_active_exposure_identifier_version
                    != self.treatment_ontology_version
                ):
                    raise ValueError(
                        "verified regimen exposure provenance must match the "
                        "treatment ontology release"
                    )
                _validate_v1_monotherapy_regimen(
                    treatment_name=self.treatment_name,
                    regimen_name=self.regimen_name,
                    component_relation=self.regimen_component_relation,
                    field_prefix="preclinical regimen",
                )
            if self.regimen_component_relation in {
                RegimenComponentRelation.NONE,
                RegimenComponentRelation.UNRESOLVED,
            }:
                raise ValueError(
                    "a named regimen requires a resolved active-component relation"
                )
        _validate_biomarker_bundle(
            biomarker_context=self.biomarker_context,
            feature_type=self.biomarker_feature_type,
            state=self.biomarker_state,
            specimen_type=self.biomarker_specimen_type,
            measurement_timepoint=self.biomarker_measurement_timepoint,
            informative_verified=self.biomarker_axes_informative_verified,
            observation_status=self.biomarker_axes_observation_status,
        )
        if self.evidence_scope == PreclinicalEvidenceScope.GENE_SPECIFIC:
            if not self.gene_symbol:
                raise ValueError("gene_specific evidence requires gene_symbol")
            _validate_required_claim_texts(gene_symbol=self.gene_symbol)
            if not (
                self.gene_id
                and self.gene_identifier_source
                and self.gene_identifier_version
                and self.gene_identity_curator_verified is True
            ):
                raise ValueError(
                    "gene_specific evidence requires a curator-attested versioned "
                    "gene identity"
                )
            _validate_versioned_ontology_identifier(
                self.gene_id,
                ontology_name=self.gene_identifier_source,
                ontology_version=self.gene_identifier_version,
                field_name="gene_id",
            )
            if self.claim_type == PreclinicalClaimType.TREATMENT_ACTIVITY_ONLY:
                raise ValueError(
                    "treatment_activity_only must use treatment_context scope"
                )
        else:
            if any(
                value is not None
                for value in (
                    self.gene_symbol,
                    self.gene_id,
                    self.gene_identifier_source,
                    self.gene_identifier_version,
                    self.gene_identity_curator_verified,
                    self.perturbation_modality,
                )
            ):
                raise ValueError(
                    "treatment_context evidence cannot claim a gene perturbation"
                )
            if self.phenotype_direction != PhenotypeDirection.UNKNOWN:
                raise ValueError(
                    "treatment_context evidence cannot assign a gene phenotype"
                )
            if self.claim_type not in {
                PreclinicalClaimType.TREATMENT_ACTIVITY_ONLY,
                PreclinicalClaimType.MECHANISTIC_ONLY,
            }:
                raise ValueError(
                    "treatment_context evidence cannot make a gene-specific claim"
                )
        if self.claim_type == PreclinicalClaimType.DIRECT_PERTURBATIONAL_INTERACTION:
            if self.evidence_scope != PreclinicalEvidenceScope.GENE_SPECIFIC:
                raise ValueError("direct perturbational evidence must be gene_specific")
            if self.perturbation_modality is None:
                raise ValueError(
                    "direct perturbational evidence requires perturbation_modality"
                )
            if self.vehicle_or_baseline_control_present is not True:
                raise ValueError(
                    "direct perturbational evidence requires a vehicle or "
                    "baseline-growth control"
                )
            if self.genotype_by_treatment_tested is not True:
                raise ValueError(
                    "direct perturbational evidence requires a "
                    "genotype-by-treatment test"
                )
            comparator_bundle = (
                self.comparator_exposure_type,
                self.comparator_active_exposure_ids_json,
                self.comparator_regimen_component_relation,
            )
            if any(value is None for value in comparator_bundle):
                raise ValueError(
                    "direct perturbational evidence requires a structured comparator"
                )
            comparator_exposures = _canonical_exposure_id_set(
                self.comparator_active_exposure_ids_json,
                field_name="comparator_active_exposure_ids_json",
                allow_empty=True,
            )
            if (
                self.comparator_exposure_type
                not in {
                    ComparatorExposureType.PLACEBO,
                    ComparatorExposureType.VEHICLE,
                    ComparatorExposureType.NO_ACTIVE_THERAPEUTIC,
                }
                or comparator_exposures
                or self.comparator_regimen_component_relation
                != RegimenComponentRelation.NONE
            ):
                raise ValueError(
                    "v1 direct perturbational evidence requires a resolved no-active "
                    "comparator with an empty active-exposure set"
                )
            comparator_label = _normalized_claim_text(self.comparator)
            allowed_comparator_labels = {
                ComparatorExposureType.PLACEBO: {"placebo", "placebo control"},
                ComparatorExposureType.VEHICLE: {"vehicle", "vehicle control"},
                ComparatorExposureType.NO_ACTIVE_THERAPEUTIC: {
                    "no active therapy",
                    "untreated control",
                },
            }
            if (
                comparator_label
                not in allowed_comparator_labels[self.comparator_exposure_type]
            ):
                raise ValueError(
                    "preclinical comparator label conflicts with its controlled type"
                )
            if self.phenotype_direction != PhenotypeDirection.UNKNOWN and not (
                self.direction_rule_id and self.direction_rule_version
            ):
                raise ValueError(
                    "a non-unknown perturbational phenotype requires a versioned "
                    "direction rule"
                )
            if self.phenotype_direction != PhenotypeDirection.UNKNOWN:
                _validate_stable_identifier(
                    self.direction_rule_id,
                    field_name="direction_rule_id",
                )
                _validate_stable_identifier(
                    self.direction_rule_version,
                    field_name="direction_rule_version",
                )
                if self.native_effect is None and self.effect_numeric is None:
                    raise ValueError(
                        "directional perturbational evidence requires a numerical "
                        "native or harmonized effect"
                    )
                if self.phenotype_direction in {
                    PhenotypeDirection.RESISTANCE,
                    PhenotypeDirection.SENSITIZATION,
                } and all(
                    value is None or value == 0
                    for value in (self.native_effect, self.effect_numeric)
                ):
                    raise ValueError(
                        "resistance or sensitization evidence requires a non-zero "
                        "effect under its versioned direction rule"
                    )
                if self.sample_n is None:
                    raise ValueError(
                        "directional perturbational evidence requires sample_n"
                    )
                expected_status = (
                    PreclinicalDirectionInferenceStatus.NEUTRAL_SUPPORTED
                    if self.phenotype_direction == PhenotypeDirection.NEUTRAL
                    else PreclinicalDirectionInferenceStatus.DIRECTION_SUPPORTED
                )
                if (
                    self.direction_inference_status != expected_status
                    or self.direction_inference_curator_verified is not True
                ):
                    raise ValueError(
                        "non-unknown perturbational direction requires the matching "
                        "curator-verified inference status"
                    )
                if self.phenotype_direction in {
                    PhenotypeDirection.RESISTANCE,
                    PhenotypeDirection.SENSITIZATION,
                    PhenotypeDirection.DISCORDANT,
                }:
                    supplied_effects = [
                        value
                        for value in (self.native_effect, self.effect_numeric)
                        if value is not None
                    ]
                    if not supplied_effects or all(
                        math.isclose(value, 0.0, abs_tol=1e-15)
                        for value in supplied_effects
                    ):
                        raise ValueError(
                            "direction-supported perturbational evidence requires a "
                            "non-zero effect"
                        )
            else:
                if self.direction_inference_status not in {
                    PreclinicalDirectionInferenceStatus.INCONCLUSIVE,
                    PreclinicalDirectionInferenceStatus.UNSUPPORTED,
                    PreclinicalDirectionInferenceStatus.NOT_ASSESSED,
                }:
                    raise ValueError(
                        "unknown perturbational direction requires an inconclusive, "
                        "unsupported, or not-assessed inference status"
                    )
                if self.direction_inference_curator_verified is None:
                    raise ValueError(
                        "direct perturbational inference status requires an explicit "
                        "curator verification decision"
                    )
            if (
                self.phenotype_direction
                in {
                    PhenotypeDirection.NEUTRAL,
                    PhenotypeDirection.DISCORDANT,
                }
                and self.neutrality_or_discordance_rule_prespecified is not True
            ):
                raise ValueError(
                    "neutral or discordant perturbational evidence requires a "
                    "prespecified decision rule"
                )
        else:
            if self.perturbation_modality is not None:
                raise ValueError("non-direct claims cannot carry perturbation_modality")
            if (
                self.direction_rule_id is not None
                or self.direction_rule_version is not None
                or self.direction_inference_status is not None
                or self.direction_inference_curator_verified is not None
                or self.neutrality_or_discordance_rule_prespecified is not None
            ):
                raise ValueError(
                    "non-direct claims cannot carry perturbation direction rules"
                )
            if self.phenotype_direction != PhenotypeDirection.UNKNOWN:
                raise ValueError(
                    "non-perturbational claims cannot be translated into a KO "
                    "phenotype direction"
                )
            if any(
                value is not None
                for value in (
                    self.comparator_exposure_type,
                    self.comparator_active_exposure_ids_json,
                    self.comparator_regimen_component_relation,
                )
            ):
                raise ValueError(
                    "non-direct claims cannot carry structured perturbational "
                    "comparator fields"
                )
        return self


class PatientMolecularEvidenceRecord(StrictRecord):
    """Aggregate gene/outcome claim from one treatment-matched patient cohort."""

    model_config = ConfigDict(
        json_schema_extra={
            "x-semantic-rules": [
                "retrieved_date cannot precede available_date",
                "source and raw-data family identifiers are required",
                "ontology IDs require ontology name and version",
                (
                    "subtypes require explicit parent verification; verified "
                    "bindings require versioned subtype and cancer IDs with equal "
                    "parent and cancer IDs"
                ),
                "biomarker term, type, state, specimen, and timepoint are all-or-none",
                "typed biomarker exactness requires observed status and attestation",
                (
                    "cohort-context biomarker fields are distinct from the "
                    "gene-level tested molecular predictor"
                ),
                (
                    "gene_symbol is bound to a versioned predictor gene ID, "
                    "feature type, state, specimen, measurement, and curator "
                    "verification"
                ),
                (
                    "predictor identity verification is a curator attestation, "
                    "not external resolver authentication"
                ),
                "scientific attestations require literal booleans",
                (
                    "exact curated status requires versioned canonical treatment "
                    "and cancer IDs"
                ),
                (
                    "tested interactions require pretreatment measurement, verified "
                    "canonical active-exposure sets and provenance, v1 canonical "
                    "monotherapy versus placebo/no-active control, distinct "
                    "source-native assignment IDs, evaluable per-arm/model counts, "
                    "scale-appropriate event counts, predictor variation, a "
                    "versioned inference rule, and a controlled effect scale"
                ),
                (
                    "association interpretation, tested flag, and supported/null/"
                    "inconclusive/unsupported/not-tested inference status must agree"
                ),
                (
                    "controlled effect scale determines its null value, positivity "
                    "constraints, and event-count requirements"
                ),
                (
                    "supported and unsupported interaction p-values test departure "
                    "from null; formal null requires prespecified equivalence bounds "
                    "and an equivalence-role p-value when supplied"
                ),
                (
                    "inconclusive interaction requires discordant p-value and "
                    "confidence-interval support"
                ),
                (
                    "untested interactions cannot carry formal interaction "
                    "inference fields"
                ),
                (
                    "pharmacodynamic and acquired-resistance claims require paired "
                    "longitudinal testing"
                ),
                "prognostic-only predictors require pretreatment measurement",
                (
                    "unverified patient treatment exposure cannot establish exact "
                    "treatment context"
                ),
                "post-progression claims require documented progression",
                "used_for_label must remain false",
            ]
        }
    )
    primary_key = ("evidence_id",)

    evidence_id: str = Field(min_length=1)
    source_study_id: str = Field(min_length=1)
    cohort_id: str = Field(min_length=1)
    source_family_id: str = Field(min_length=1)
    raw_data_family_id: str = Field(min_length=1)
    gene_symbol: str = Field(min_length=1)
    predictor_gene_symbol: str = Field(min_length=1)
    gene_id: str = Field(min_length=1)
    gene_identifier_source: str = Field(min_length=1)
    gene_identifier_version: str = Field(min_length=1)
    predictor_feature_type: BiomarkerFeatureType
    predictor_state: str = Field(min_length=1)
    predictor_specimen_type: str = Field(min_length=1)
    predictor_identity_curator_verified: Literal[True]
    treatment_name: str = Field(min_length=1)
    treatment_id: str | None = Field(default=None, min_length=1)
    treatment_ontology_name: str | None = Field(default=None, min_length=1)
    treatment_ontology_version: str | None = Field(default=None, min_length=1)
    regimen_name: str | None = Field(default=None, min_length=1)
    comparator_regimen_name: str | None = Field(default=None, min_length=1)
    treatment_assignment_id: str | None = Field(default=None, min_length=1)
    comparator_assignment_id: str | None = Field(default=None, min_length=1)
    treatment_active_exposure_ids_json: str | None = Field(default=None, min_length=2)
    comparator_active_exposure_ids_json: str | None = Field(default=None, min_length=2)
    treatment_regimen_component_relation: RegimenComponentRelation | None = None
    comparator_regimen_component_relation: RegimenComponentRelation | None = None
    comparator_exposure_type: ComparatorExposureType | None = None
    active_exposure_identifier_source: str | None = Field(default=None, min_length=1)
    active_exposure_identifier_version: str | None = Field(default=None, min_length=1)
    active_exposure_ids_curator_verified: bool | None = None
    cancer_type: str = Field(min_length=1)
    cancer_id: str | None = Field(default=None, min_length=1)
    disease_subtype: str | None = Field(default=None, min_length=1)
    disease_subtype_id: str | None = Field(default=None, min_length=1)
    disease_subtype_parent_id: str | None = Field(default=None, min_length=1)
    disease_subtype_parent_binding_verified: bool | None = None
    disease_ontology_name: str | None = Field(default=None, min_length=1)
    disease_ontology_version: str | None = Field(default=None, min_length=1)
    stage: str | None = Field(default=None, min_length=1)
    line_of_therapy: str | None = Field(default=None, min_length=1)
    biomarker_context: str | None = Field(default=None, min_length=1)
    biomarker_feature_type: BiomarkerFeatureType | None = None
    biomarker_state: str | None = Field(default=None, min_length=1)
    biomarker_specimen_type: str | None = Field(default=None, min_length=1)
    biomarker_measurement_timepoint: MolecularMeasurementTimepoint | None = None
    biomarker_axes_informative_verified: bool | None = None
    biomarker_axes_observation_status: BiomarkerAxisObservationStatus | None = None
    measurement_type: str = Field(min_length=1)
    measurement_platform: str | None = Field(default=None, min_length=1)
    measurement_timepoint: MolecularMeasurementTimepoint
    clinical_outcome: str = Field(min_length=1)
    association_interpretation: PatientAssociationInterpretation
    comparator_arm_present: bool
    treatment_predictor_interaction_tested: bool
    interaction_inference_status: InteractionInferenceStatus
    interaction_effect_scale: InteractionEffectScale | None = None
    interaction_inference_rule_id: str | None = Field(default=None, min_length=1)
    interaction_inference_rule_version: str | None = Field(default=None, min_length=1)
    interaction_significance_threshold: float | None = Field(
        default=None, gt=0.0, le=0.10
    )
    interaction_p_value_role: InteractionPValueRole | None = None
    interaction_equivalence_lower: float | None = None
    interaction_equivalence_upper: float | None = None
    interaction_inference_curator_verified: bool | None = None
    treatment_exposure_verified: bool | None = None
    treatment_comparator_exposures_distinct_verified: bool | None = None
    paired_baseline_present: bool | None = None
    longitudinal_change_tested: bool | None = None
    progression_documented: bool | None = None
    analysis_model: str | None = Field(default=None, min_length=1)
    patient_n: int = Field(ge=1)
    treatment_arm_patient_n: int | None = Field(default=None, ge=3)
    comparator_arm_patient_n: int | None = Field(default=None, ge=3)
    interaction_analysis_evaluable_patient_n: int | None = Field(default=None, ge=6)
    interaction_model_parameter_n: int | None = Field(default=None, ge=4)
    interaction_outcome_event_n: int | None = Field(default=None, ge=1)
    interaction_model_estimable_verified: bool | None = None
    biomarker_variation_in_each_arm_verified: bool | None = None
    effect_numeric: float | None = None
    effect_type: str | None = Field(default=None, min_length=1)
    confidence_interval_lower: float | None = None
    confidence_interval_upper: float | None = None
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    outcome_text: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_url: HttpUrl
    source_locator: str = Field(min_length=1)
    available_date: date
    retrieved_date: date
    used_for_label: Literal[False] = False
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def patient_numeric_fields_are_not_boolean(cls, value: Any) -> Any:
        value = _require_boolean_fields(
            value,
            field_names={
                "predictor_identity_curator_verified",
                "active_exposure_ids_curator_verified",
                "disease_subtype_parent_binding_verified",
                "biomarker_axes_informative_verified",
                "comparator_arm_present",
                "treatment_predictor_interaction_tested",
                "interaction_inference_curator_verified",
                "treatment_exposure_verified",
                "treatment_comparator_exposures_distinct_verified",
                "paired_baseline_present",
                "longitudinal_change_tested",
                "progression_documented",
                "interaction_model_estimable_verified",
                "biomarker_variation_in_each_arm_verified",
                "used_for_label",
            },
        )
        return _reject_boolean_numeric_fields(
            value,
            field_names={
                "interaction_significance_threshold",
                "interaction_equivalence_lower",
                "interaction_equivalence_upper",
                "patient_n",
                "treatment_arm_patient_n",
                "comparator_arm_patient_n",
                "interaction_analysis_evaluable_patient_n",
                "interaction_model_parameter_n",
                "interaction_outcome_event_n",
                "effect_numeric",
                "confidence_interval_lower",
                "confidence_interval_upper",
                "p_value",
            },
        )

    @model_validator(mode="after")
    def patient_claim_is_not_overstated(self) -> PatientMolecularEvidenceRecord:
        if self.retrieved_date < self.available_date:
            raise ValueError("retrieved_date cannot precede available_date")
        if (self.effect_numeric is None) != (self.effect_type is None):
            raise ValueError("effect_numeric and effect_type must be supplied together")
        if (self.confidence_interval_lower is None) != (
            self.confidence_interval_upper is None
        ):
            raise ValueError("both confidence-interval bounds are required together")
        if (
            self.confidence_interval_lower is not None
            and self.confidence_interval_upper is not None
            and self.confidence_interval_lower > self.confidence_interval_upper
        ):
            raise ValueError("confidence-interval lower bound exceeds upper bound")
        treatment_ontology_fields = (
            self.treatment_id,
            self.treatment_ontology_name,
            self.treatment_ontology_version,
        )
        if any(treatment_ontology_fields) and not all(treatment_ontology_fields):
            raise ValueError(
                "treatment ontology ID, name, and version must be supplied together"
            )
        if self.disease_subtype_id and not self.disease_subtype:
            raise ValueError("disease_subtype_id requires disease_subtype")
        if self.disease_subtype:
            if self.disease_subtype_parent_binding_verified is None:
                raise ValueError(
                    "disease subtype requires explicit parent-binding verification"
                )
            if self.disease_subtype_parent_binding_verified is True:
                if not (
                    self.disease_subtype_id
                    and self.cancer_id
                    and self.disease_subtype_parent_id == self.cancer_id
                ):
                    raise ValueError(
                        "verified disease subtype requires versioned IDs and a "
                        "parent ID equal to cancer_id"
                    )
        elif (
            self.disease_subtype_id is not None
            or self.disease_subtype_parent_id is not None
            or self.disease_subtype_parent_binding_verified is not None
        ):
            raise ValueError("disease subtype metadata requires disease_subtype")
        disease_ids_present = bool(
            self.cancer_id or self.disease_subtype_id or self.disease_subtype_parent_id
        )
        disease_ontology_fields = (
            self.disease_ontology_name,
            self.disease_ontology_version,
        )
        if disease_ids_present and not all(disease_ontology_fields):
            raise ValueError(
                "disease ontology name and version are required with disease IDs"
            )
        if not disease_ids_present and any(disease_ontology_fields):
            raise ValueError("disease ontology metadata requires a disease ID")
        if all(treatment_ontology_fields):
            _validate_versioned_ontology_identifier(
                self.treatment_id,
                ontology_name=self.treatment_ontology_name,
                ontology_version=self.treatment_ontology_version,
                field_name="treatment_id",
            )
        if disease_ids_present:
            for field_name in (
                "cancer_id",
                "disease_subtype_id",
                "disease_subtype_parent_id",
            ):
                value = getattr(self, field_name)
                if value is not None:
                    _validate_versioned_ontology_identifier(
                        value,
                        ontology_name=self.disease_ontology_name,
                        ontology_version=self.disease_ontology_version,
                        field_name=field_name,
                    )
        if (
            self.cancer_id
            and self.disease_subtype_id
            and (self.cancer_id.casefold() == self.disease_subtype_id.casefold())
        ):
            raise ValueError("disease subtype ID must differ from cancer ID")
        _validate_required_claim_texts(
            gene_symbol=self.gene_symbol,
            predictor_gene_symbol=self.predictor_gene_symbol,
            predictor_state=self.predictor_state,
            predictor_specimen_type=self.predictor_specimen_type,
            treatment_name=self.treatment_name,
            cancer_type=self.cancer_type,
            measurement_type=self.measurement_type,
            clinical_outcome=self.clinical_outcome,
            outcome_text=self.outcome_text,
            source_name=self.source_name,
            source_version=self.source_version,
            source_locator=self.source_locator,
        )
        if _normalized_claim_text(self.gene_symbol) != _normalized_claim_text(
            self.predictor_gene_symbol
        ):
            raise ValueError("predictor_gene_symbol must equal gene_symbol")
        _validate_versioned_ontology_identifier(
            self.gene_id,
            ontology_name=self.gene_identifier_source,
            ontology_version=self.gene_identifier_version,
            field_name="gene_id",
        )
        if self.predictor_feature_type == BiomarkerFeatureType.OTHER:
            raise ValueError("gene-level predictor feature type cannot be other")
        if not _predictor_measurement_is_compatible(
            self.predictor_feature_type,
            measurement_type=self.measurement_type,
            measurement_platform=self.measurement_platform,
        ):
            raise ValueError(
                "predictor feature type conflicts with measurement type/platform"
            )
        _validate_required_claim_texts(
            **{
                field_name: value
                for field_name, value in (
                    ("disease_subtype", self.disease_subtype),
                    ("stage", self.stage),
                    ("line_of_therapy", self.line_of_therapy),
                )
                if value is not None
            }
        )
        for field_name in (
            "source_study_id",
            "cohort_id",
            "source_family_id",
        ):
            _validate_stable_identifier(
                getattr(self, field_name), field_name=field_name
            )
        _validate_stable_identifier(
            self.raw_data_family_id, field_name="raw_data_family_id"
        )
        _validate_biomarker_bundle(
            biomarker_context=self.biomarker_context,
            feature_type=self.biomarker_feature_type,
            state=self.biomarker_state,
            specimen_type=self.biomarker_specimen_type,
            measurement_timepoint=self.biomarker_measurement_timepoint,
            informative_verified=self.biomarker_axes_informative_verified,
            observation_status=self.biomarker_axes_observation_status,
        )
        treatment_bundle = (
            self.regimen_name,
            self.treatment_active_exposure_ids_json,
            self.treatment_regimen_component_relation,
            self.active_exposure_identifier_source,
            self.active_exposure_identifier_version,
            self.active_exposure_ids_curator_verified,
        )
        if any(value is not None for value in treatment_bundle) and not all(
            value is not None for value in treatment_bundle
        ):
            raise ValueError(
                "patient treatment regimen, active IDs, relation, provenance, and "
                "verification must be supplied together"
            )
        comparator_bundle = (
            self.comparator_regimen_name,
            self.comparator_active_exposure_ids_json,
            self.comparator_regimen_component_relation,
            self.comparator_exposure_type,
        )
        if any(value is not None for value in comparator_bundle) and not all(
            value is not None for value in comparator_bundle
        ):
            raise ValueError(
                "patient comparator name, active IDs, relation, and type must be "
                "supplied together"
            )
        comparator_presence_fields = (*comparator_bundle, self.comparator_assignment_id)
        if self.comparator_arm_present:
            if not all(value is not None for value in comparator_presence_fields):
                raise ValueError(
                    "comparator_arm_present requires a complete comparator bundle "
                    "and assignment ID"
                )
        elif any(value is not None for value in comparator_presence_fields):
            raise ValueError(
                "comparator metadata cannot be supplied when comparator_arm_present "
                "is false"
            )
        if all(value is not None for value in treatment_bundle):
            bundled_treatment_exposures = _canonical_exposure_id_set(
                self.treatment_active_exposure_ids_json,
                field_name="treatment_active_exposure_ids_json",
            )
            _validate_exposure_ontology_binding(
                bundled_treatment_exposures,
                source=self.active_exposure_identifier_source,
                version=self.active_exposure_identifier_version,
                field_name="treatment_active_exposure_ids_json",
            )
            if (
                self.treatment_regimen_component_relation
                == RegimenComponentRelation.NONE
            ):
                raise ValueError(
                    "a non-empty treatment active-exposure set cannot use relation none"
                )
            if self.treatment_id is None or self.treatment_id.casefold() not in (
                bundled_treatment_exposures
            ):
                raise ValueError(
                    "verified treatment active-exposure IDs must contain treatment_id"
                )
            if (
                _normalized_claim_text(self.active_exposure_identifier_source)
                != _normalized_claim_text(self.treatment_ontology_name)
                or self.active_exposure_identifier_version
                != self.treatment_ontology_version
            ):
                raise ValueError(
                    "active-exposure identifier provenance must match the canonical "
                    "treatment ontology release"
                )
            if self.active_exposure_ids_curator_verified is True:
                if bundled_treatment_exposures != {self.treatment_id.casefold()}:
                    raise ValueError(
                        "v1 verified patient regimens require exactly treatment_id"
                    )
                _validate_v1_monotherapy_regimen(
                    treatment_name=self.treatment_name,
                    regimen_name=self.regimen_name,
                    component_relation=self.treatment_regimen_component_relation,
                    field_prefix="patient regimen",
                )
        if all(value is not None for value in comparator_bundle):
            bundled_comparator_exposures = _canonical_exposure_id_set(
                self.comparator_active_exposure_ids_json,
                field_name="comparator_active_exposure_ids_json",
                allow_empty=True,
            )
            no_active_comparator_types = {
                ComparatorExposureType.PLACEBO,
                ComparatorExposureType.NO_ACTIVE_THERAPEUTIC,
                ComparatorExposureType.VEHICLE,
            }
            if self.comparator_exposure_type in no_active_comparator_types:
                if bundled_comparator_exposures:
                    raise ValueError(
                        "a placebo/vehicle/no-active comparator must have no active "
                        "exposure IDs"
                    )
                if (
                    self.comparator_regimen_component_relation
                    != RegimenComponentRelation.NONE
                ):
                    raise ValueError(
                        "an empty placebo/vehicle/no-active comparator must use "
                        "relation none"
                    )
            if (
                self.comparator_exposure_type
                == ComparatorExposureType.ACTIVE_THERAPEUTIC
                and not bundled_comparator_exposures
            ):
                raise ValueError("an active comparator requires active exposure IDs")
            if bundled_comparator_exposures:
                if (
                    self.comparator_regimen_component_relation
                    == RegimenComponentRelation.NONE
                ):
                    raise ValueError(
                        "a non-empty comparator active-exposure set cannot use "
                        "relation none"
                    )
                if (
                    self.active_exposure_identifier_source is None
                    or self.active_exposure_identifier_version is None
                ):
                    raise ValueError(
                        "active comparator IDs require identifier provenance"
                    )
                _validate_exposure_ontology_binding(
                    bundled_comparator_exposures,
                    source=self.active_exposure_identifier_source,
                    version=self.active_exposure_identifier_version,
                    field_name="comparator_active_exposure_ids_json",
                )
        tested_interpretation_status = {
            PatientAssociationInterpretation.PREDICTIVE_INTERACTION: (
                InteractionInferenceStatus.SUPPORTED
            ),
            PatientAssociationInterpretation.INTERACTION_TESTED_NULL: (
                InteractionInferenceStatus.NULL
            ),
            PatientAssociationInterpretation.INTERACTION_TESTED_INCONCLUSIVE: (
                InteractionInferenceStatus.INCONCLUSIVE
            ),
            PatientAssociationInterpretation.INTERACTION_TESTED_UNSUPPORTED: (
                InteractionInferenceStatus.UNSUPPORTED
            ),
        }
        expected_inference_status = tested_interpretation_status.get(
            self.association_interpretation
        )
        if expected_inference_status is not None:
            if not self.treatment_predictor_interaction_tested:
                raise ValueError(
                    "an interaction interpretation requires a formal "
                    "treatment-by-predictor interaction test"
                )
            if self.interaction_inference_status != expected_inference_status:
                raise ValueError(
                    "association_interpretation conflicts with "
                    "interaction_inference_status"
                )
            if self.measurement_timepoint != MolecularMeasurementTimepoint.PRETREATMENT:
                raise ValueError(
                    "tested interactions require a pretreatment measurement"
                )
            if not self.comparator_arm_present:
                raise ValueError("tested interactions require a comparator arm")
            if not self.comparator_regimen_name:
                raise ValueError("tested interactions require comparator_regimen_name")
            if not self.regimen_name:
                raise ValueError("tested interactions require regimen_name")
            if not self.treatment_assignment_id or not self.comparator_assignment_id:
                raise ValueError(
                    "tested interactions require treatment and comparator "
                    "assignment IDs"
                )
            if self.treatment_active_exposure_ids_json is None or (
                self.comparator_active_exposure_ids_json is None
            ):
                raise ValueError(
                    "tested interactions require canonical treatment and "
                    "comparator active-exposure ID arrays"
                )
            treatment_exposures = _canonical_exposure_id_set(
                self.treatment_active_exposure_ids_json,
                field_name="treatment_active_exposure_ids_json",
            )
            comparator_exposures = _canonical_exposure_id_set(
                self.comparator_active_exposure_ids_json,
                field_name="comparator_active_exposure_ids_json",
                allow_empty=True,
            )
            if self.comparator_exposure_type is None:
                raise ValueError("tested interactions require comparator_exposure_type")
            if (
                self.comparator_exposure_type
                in {
                    ComparatorExposureType.PLACEBO,
                    ComparatorExposureType.NO_ACTIVE_THERAPEUTIC,
                }
                and comparator_exposures
            ):
                raise ValueError(
                    "a placebo/no-active comparator must use an empty "
                    "active-exposure array"
                )
            if (
                self.comparator_exposure_type
                == ComparatorExposureType.ACTIVE_THERAPEUTIC
                and not comparator_exposures
            ):
                raise ValueError(
                    "an active comparator requires a non-empty active-exposure array"
                )
            if self.comparator_exposure_type in {
                ComparatorExposureType.ACTIVE_THERAPEUTIC,
                ComparatorExposureType.UNRESOLVED,
            }:
                raise ValueError(
                    "active or unresolved comparators require a versioned concept "
                    "registry and are unsupported for formal interaction claims in v1"
                )
            if (
                self.treatment_regimen_component_relation
                != RegimenComponentRelation.FIXED_ALL_OF
                or self.comparator_regimen_component_relation
                != RegimenComponentRelation.NONE
            ):
                raise ValueError(
                    "v1 tested interactions require fixed treatment components and "
                    "a no-active comparator relation"
                )
            if treatment_exposures == comparator_exposures:
                raise ValueError(
                    "tested interactions require distinct canonical active "
                    "therapeutic exposure sets"
                )
            canonical_treatment_id = (
                unicodedata.normalize("NFKC", self.treatment_id).casefold()
                if self.treatment_id
                else None
            )
            if canonical_treatment_id is None:
                raise ValueError(
                    "tested interactions require a versioned canonical treatment_id"
                )
            if canonical_treatment_id and canonical_treatment_id not in (
                treatment_exposures
            ):
                raise ValueError(
                    "treatment_active_exposure_ids_json must contain treatment_id"
                )
            if treatment_exposures != {canonical_treatment_id}:
                raise ValueError(
                    "v1 formal interaction claims require the treatment arm active "
                    "set to equal the requested canonical treatment_id"
                )
            if (
                canonical_treatment_id
                and canonical_treatment_id in comparator_exposures
            ):
                raise ValueError(
                    "the requested treatment cannot be active in the comparator arm"
                )
            if self.active_exposure_ids_curator_verified is not True:
                raise ValueError(
                    "tested interactions require curator-verified active-exposure IDs"
                )
            _validate_required_claim_texts(
                regimen_name=self.regimen_name,
                comparator_regimen_name=self.comparator_regimen_name,
                active_exposure_identifier_source=(
                    self.active_exposure_identifier_source
                ),
                active_exposure_identifier_version=(
                    self.active_exposure_identifier_version
                ),
                analysis_model=self.analysis_model,
                effect_type=self.effect_type,
            )
            _validate_exposure_ontology_binding(
                treatment_exposures,
                source=self.active_exposure_identifier_source,
                version=self.active_exposure_identifier_version,
                field_name="treatment_active_exposure_ids_json",
            )
            if (
                _normalized_claim_text(self.active_exposure_identifier_source)
                != _normalized_claim_text(self.treatment_ontology_name)
                or self.active_exposure_identifier_version
                != self.treatment_ontology_version
            ):
                raise ValueError(
                    "active-exposure identifier provenance must match the canonical "
                    "treatment ontology release"
                )
            _validate_stable_identifier(
                self.treatment_assignment_id,
                field_name="treatment_assignment_id",
            )
            _validate_stable_identifier(
                self.comparator_assignment_id,
                field_name="comparator_assignment_id",
            )
            normalized_regimen = "".join(
                character
                for character in self.regimen_name.casefold()
                if character.isalnum()
            )
            normalized_comparator = "".join(
                character
                for character in self.comparator_regimen_name.casefold()
                if character.isalnum()
            )
            if not normalized_regimen or not normalized_comparator:
                raise ValueError(
                    "predictive_interaction regimen names must contain letters or "
                    "numbers"
                )
            if normalized_regimen == normalized_comparator:
                raise ValueError(
                    "tested interactions require distinct treatment and "
                    "comparator regimens"
                )
            treatment_label_tokens = set(
                re.findall(r"\w+", _normalized_claim_text(self.treatment_name))
            )
            regimen_label_tokens = set(
                re.findall(r"\w+", _normalized_claim_text(self.regimen_name))
            )
            if not treatment_label_tokens or not (
                treatment_label_tokens <= regimen_label_tokens
                and regimen_label_tokens - treatment_label_tokens
                <= {"alone", "monotherapy"}
            ):
                raise ValueError(
                    "v1 formal interaction treatment label must be canonical "
                    "monotherapy"
                )
            comparator_label = _normalized_claim_text(self.comparator_regimen_name)
            allowed_comparator_labels = {
                ComparatorExposureType.PLACEBO: {"placebo", "placebo control"},
                ComparatorExposureType.NO_ACTIVE_THERAPEUTIC: {
                    "no active therapy",
                    "no treatment",
                    "observation",
                },
            }
            if comparator_label not in allowed_comparator_labels.get(
                self.comparator_exposure_type, set()
            ):
                raise ValueError(
                    "v1 placebo/no-active comparator label conflicts with its "
                    "controlled exposure type"
                )
            normalized_assignment = re.sub(
                r"[^\w]+", "", self.treatment_assignment_id.casefold()
            )
            normalized_comparator_assignment = re.sub(
                r"[^\w]+", "", self.comparator_assignment_id.casefold()
            )
            if normalized_assignment == normalized_comparator_assignment:
                raise ValueError(
                    "tested interactions require distinct non-empty source "
                    "assignment IDs"
                )
            if self.treatment_comparator_exposures_distinct_verified is not True:
                raise ValueError(
                    "tested interactions require curator verification that treatment "
                    "and comparator exposures are distinct"
                )
            if (
                self.treatment_arm_patient_n is None
                or self.comparator_arm_patient_n is None
                or self.interaction_analysis_evaluable_patient_n is None
                or self.interaction_model_parameter_n is None
            ):
                raise ValueError(
                    "tested interactions require evaluable per-arm and model counts"
                )
            if (
                self.treatment_arm_patient_n + self.comparator_arm_patient_n
                != self.interaction_analysis_evaluable_patient_n
                or self.interaction_analysis_evaluable_patient_n > self.patient_n
            ):
                raise ValueError(
                    "per-arm counts must sum to the analysis-evaluable cohort and "
                    "cannot exceed patient_n"
                )
            if self.interaction_analysis_evaluable_patient_n <= (
                self.interaction_model_parameter_n
            ):
                raise ValueError(
                    "interaction analysis requires more evaluable patients than "
                    "model parameters"
                )
            if self.interaction_model_estimable_verified is not True:
                raise ValueError(
                    "tested interactions require verified model estimability"
                )
            if self.biomarker_variation_in_each_arm_verified is not True:
                raise ValueError(
                    "tested interactions require verified predictor variation in "
                    "each arm"
                )
            if self.effect_numeric is None or self.effect_type is None:
                raise ValueError(
                    "tested interactions require a numerical interaction effect"
                )
            numeric_values = (
                self.effect_numeric,
                self.confidence_interval_lower,
                self.confidence_interval_upper,
            )
            if any(
                value is not None and not math.isfinite(value)
                for value in numeric_values
            ):
                raise ValueError("interaction effect and interval must be finite")
            if self.confidence_interval_lower is not None and not (
                self.confidence_interval_lower
                <= self.effect_numeric
                <= self.confidence_interval_upper
            ):
                raise ValueError(
                    "interaction confidence interval must contain the point estimate"
                )
            if self.p_value is None and self.confidence_interval_lower is None:
                raise ValueError(
                    "tested interactions require an interaction p-value or "
                    "confidence interval"
                )
            if self.interaction_effect_scale is None:
                raise ValueError("tested interactions require interaction_effect_scale")
            if _normalized_claim_text(self.effect_type) != _normalized_claim_text(
                self.interaction_effect_scale.value
            ):
                raise ValueError(
                    "effect_type must equal the controlled interaction_effect_scale"
                )
            if self.interaction_effect_scale not in {
                InteractionEffectScale.ADDITIVE_COEFFICIENT,
                InteractionEffectScale.DIFFERENCE_IN_EFFECT,
            } and (
                self.effect_numeric <= 0
                or (
                    self.confidence_interval_lower is not None
                    and self.confidence_interval_lower <= 0
                )
            ):
                raise ValueError("ratio-scale interaction values must be positive")
            if self.interaction_effect_scale in {
                InteractionEffectScale.HAZARD_RATIO,
                InteractionEffectScale.ODDS_RATIO,
                InteractionEffectScale.RISK_RATIO,
            }:
                if (
                    self.interaction_outcome_event_n is None
                    or self.interaction_outcome_event_n
                    <= self.interaction_model_parameter_n
                    or self.interaction_outcome_event_n
                    > self.interaction_analysis_evaluable_patient_n
                ):
                    raise ValueError(
                        "event-based interaction scales require an estimable event "
                        "count greater than model parameters"
                    )
            elif (
                self.interaction_outcome_event_n is not None
                and self.interaction_outcome_event_n
                > self.interaction_analysis_evaluable_patient_n
            ):
                raise ValueError(
                    "interaction outcome event count cannot exceed the evaluable cohort"
                )
            if self.interaction_effect_scale in {
                InteractionEffectScale.ODDS_RATIO,
                InteractionEffectScale.RISK_RATIO,
            } and (
                self.interaction_analysis_evaluable_patient_n
                - self.interaction_outcome_event_n
                <= self.interaction_model_parameter_n
            ):
                raise ValueError(
                    "binary interaction scales require estimable event and non-event "
                    "counts"
                )
            if (
                not self.interaction_inference_rule_id
                or not self.interaction_inference_rule_version
                or self.interaction_inference_curator_verified is not True
            ):
                raise ValueError(
                    "tested interactions require a curator-verified versioned "
                    "inference rule"
                )
            _validate_stable_identifier(
                self.interaction_inference_rule_id,
                field_name="interaction_inference_rule_id",
            )
            _validate_stable_identifier(
                self.interaction_inference_rule_version,
                field_name="interaction_inference_rule_version",
            )
            if (self.p_value is None) != (
                self.interaction_significance_threshold is None
            ) or (self.p_value is None) != (self.interaction_p_value_role is None):
                raise ValueError(
                    "p_value, its role, and interaction_significance_threshold must "
                    "be supplied together"
                )
            null_value = _interaction_null_value(self.interaction_effect_scale)
            p_meets_rule = (
                self.p_value <= self.interaction_significance_threshold
                if self.p_value is not None
                else None
            )
            ci_excludes_null = (
                self.confidence_interval_upper < null_value
                or self.confidence_interval_lower > null_value
                if self.confidence_interval_lower is not None
                else None
            )
            if (
                self.interaction_inference_status != InteractionInferenceStatus.NULL
                and (
                    self.interaction_equivalence_lower is not None
                    or self.interaction_equivalence_upper is not None
                )
            ):
                raise ValueError(
                    "equivalence bounds are reserved for formal null interactions"
                )
            if (
                self.interaction_inference_status
                == InteractionInferenceStatus.SUPPORTED
            ):
                if self.interaction_p_value_role not in {
                    None,
                    InteractionPValueRole.DEPARTURE_FROM_NULL,
                }:
                    raise ValueError(
                        "supported interaction p-values must test departure from null"
                    )
                supplied_support = [
                    value
                    for value in (p_meets_rule, ci_excludes_null)
                    if value is not None
                ]
                if self.effect_numeric == null_value or not all(supplied_support):
                    raise ValueError(
                        "supported interaction metrics must exclude the controlled null"
                    )
            elif self.interaction_inference_status == InteractionInferenceStatus.NULL:
                if (
                    self.interaction_equivalence_lower is None
                    or self.interaction_equivalence_upper is None
                    or self.confidence_interval_lower is None
                ):
                    raise ValueError(
                        "formal null interaction requires prespecified equivalence "
                        "bounds and a confidence interval"
                    )
                if not (
                    self.interaction_equivalence_lower
                    < null_value
                    < self.interaction_equivalence_upper
                    and self.interaction_equivalence_lower
                    <= self.confidence_interval_lower
                    <= self.confidence_interval_upper
                    <= self.interaction_equivalence_upper
                ):
                    raise ValueError(
                        "formal null confidence interval must lie within the "
                        "prespecified equivalence bounds"
                    )
                if (
                    self.interaction_effect_scale
                    not in {
                        InteractionEffectScale.ADDITIVE_COEFFICIENT,
                        InteractionEffectScale.DIFFERENCE_IN_EFFECT,
                    }
                    and self.interaction_equivalence_lower <= 0
                ):
                    raise ValueError(
                        "ratio-scale equivalence bounds must be strictly positive"
                    )
                if self.p_value is not None and (
                    self.interaction_p_value_role
                    != InteractionPValueRole.EQUIVALENCE_TO_NULL
                    or p_meets_rule is not True
                ):
                    raise ValueError(
                        "formal null p-value must support the prespecified "
                        "equivalence test"
                    )
            elif (
                self.interaction_inference_status
                == InteractionInferenceStatus.INCONCLUSIVE
            ):
                if (
                    self.p_value is not None
                    and self.interaction_p_value_role
                    != InteractionPValueRole.DEPARTURE_FROM_NULL
                ):
                    raise ValueError(
                        "inconclusive interaction p-values must test departure "
                        "from null"
                    )
                if (
                    p_meets_rule is None
                    or ci_excludes_null is None
                    or p_meets_rule == ci_excludes_null
                ):
                    raise ValueError(
                        "inconclusive interaction requires discordant p-value and "
                        "confidence-interval support"
                    )
            elif (
                self.interaction_inference_status
                == InteractionInferenceStatus.UNSUPPORTED
            ):
                if (
                    self.p_value is not None
                    and self.interaction_p_value_role
                    != InteractionPValueRole.DEPARTURE_FROM_NULL
                ):
                    raise ValueError(
                        "unsupported interaction p-values must test departure from null"
                    )
                supplied_support = [
                    value
                    for value in (p_meets_rule, ci_excludes_null)
                    if value is not None
                ]
                if any(supplied_support):
                    raise ValueError(
                        "unsupported interaction metrics cannot satisfy the "
                        "departure-from-null rule"
                    )
        else:
            if self.treatment_predictor_interaction_tested:
                raise ValueError(
                    "a tested treatment-by-predictor interaction must use an "
                    "interaction-specific interpretation"
                )
            if (
                self.interaction_inference_status
                != InteractionInferenceStatus.NOT_TESTED
            ):
                raise ValueError(
                    "an untested interaction requires inference status not_tested"
                )
            unused_interaction_fields = (
                self.interaction_effect_scale,
                self.interaction_inference_rule_id,
                self.interaction_inference_rule_version,
                self.interaction_significance_threshold,
                self.interaction_p_value_role,
                self.interaction_equivalence_lower,
                self.interaction_equivalence_upper,
                self.interaction_inference_curator_verified,
                self.treatment_arm_patient_n,
                self.comparator_arm_patient_n,
                self.interaction_analysis_evaluable_patient_n,
                self.interaction_model_parameter_n,
                self.interaction_outcome_event_n,
                self.interaction_model_estimable_verified,
                self.biomarker_variation_in_each_arm_verified,
            )
            if any(value is not None for value in unused_interaction_fields):
                raise ValueError(
                    "untested interactions cannot carry interaction inference fields"
                )
        if (
            self.association_interpretation
            == (PatientAssociationInterpretation.TREATED_COHORT_ASSOCIATION)
            and self.measurement_timepoint != MolecularMeasurementTimepoint.PRETREATMENT
        ):
            raise ValueError(
                "treated_cohort_association requires a pretreatment measurement"
            )
        if (
            self.association_interpretation
            == PatientAssociationInterpretation.PROGNOSTIC_ONLY
            and self.measurement_timepoint != MolecularMeasurementTimepoint.PRETREATMENT
        ):
            raise ValueError(
                "prognostic_only requires a pretreatment or baseline predictor"
            )
        treatment_exposure_claims = {
            PatientAssociationInterpretation.PREDICTIVE_INTERACTION,
            PatientAssociationInterpretation.INTERACTION_TESTED_NULL,
            PatientAssociationInterpretation.INTERACTION_TESTED_INCONCLUSIVE,
            PatientAssociationInterpretation.INTERACTION_TESTED_UNSUPPORTED,
            PatientAssociationInterpretation.TREATED_COHORT_ASSOCIATION,
            PatientAssociationInterpretation.PHARMACODYNAMIC,
            PatientAssociationInterpretation.ACQUIRED_RESISTANCE,
            PatientAssociationInterpretation.ON_TREATMENT_ASSOCIATION,
            PatientAssociationInterpretation.POST_PROGRESSION_ASSOCIATION,
        }
        if (
            self.association_interpretation in treatment_exposure_claims
            and self.treatment_exposure_verified is not True
        ):
            raise ValueError(
                "treatment-linked patient evidence requires verified exposure"
            )
        if self.association_interpretation == (
            PatientAssociationInterpretation.PHARMACODYNAMIC
        ) and self.measurement_timepoint not in {
            MolecularMeasurementTimepoint.ON_TREATMENT,
            MolecularMeasurementTimepoint.POST_TREATMENT,
        }:
            raise ValueError(
                "pharmacodynamic evidence requires on-treatment or post-treatment "
                "measurement"
            )
        if self.association_interpretation == (
            PatientAssociationInterpretation.PHARMACODYNAMIC
        ) and not (
            self.paired_baseline_present is True
            and self.longitudinal_change_tested is True
        ):
            raise ValueError(
                "pharmacodynamic evidence requires paired baseline and a "
                "longitudinal change test"
            )
        if self.association_interpretation == (
            PatientAssociationInterpretation.ACQUIRED_RESISTANCE
        ) and self.measurement_timepoint != (
            MolecularMeasurementTimepoint.POST_TREATMENT
        ):
            raise ValueError(
                "acquired_resistance evidence requires a post-treatment measurement"
            )
        if self.association_interpretation == (
            PatientAssociationInterpretation.ACQUIRED_RESISTANCE
        ) and not (
            self.paired_baseline_present is True
            and self.longitudinal_change_tested is True
        ):
            raise ValueError(
                "acquired_resistance evidence requires paired baseline and a "
                "longitudinal change test"
            )
        if (
            self.association_interpretation
            == (PatientAssociationInterpretation.ACQUIRED_RESISTANCE)
            and self.progression_documented is not True
        ):
            raise ValueError("acquired_resistance requires documented progression")
        if (
            self.association_interpretation
            == (PatientAssociationInterpretation.ON_TREATMENT_ASSOCIATION)
            and self.measurement_timepoint != MolecularMeasurementTimepoint.ON_TREATMENT
        ):
            raise ValueError(
                "on_treatment_association requires an on-treatment measurement"
            )
        if self.association_interpretation == (
            PatientAssociationInterpretation.POST_PROGRESSION_ASSOCIATION
        ):
            if (
                self.measurement_timepoint
                != MolecularMeasurementTimepoint.POST_TREATMENT
            ):
                raise ValueError(
                    "post_progression_association requires a post-treatment measurement"
                )
            if self.progression_documented is not True:
                raise ValueError(
                    "post_progression_association requires documented progression"
                )
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
    "immune_screen_evidence": ImmuneScreenEvidenceRecord,
    "treatment_disease_context": TreatmentDiseaseContextRecord,
    "clinical_trial_context": ClinicalTrialContextRecord,
    "preclinical_evidence": PreclinicalEvidenceRecord,
    "patient_molecular_evidence": PatientMolecularEvidenceRecord,
    "external_screen_map": ExternalScreenMapRecord,
    "screen_intake": ScreenIntakeRecord,
    "curation_queue": CurationQueueRecord,
    "full_text_review": FullTextReviewRecord,
    "review_comparison": ReviewComparisonRecord,
    "adjudication_packet": AdjudicationPacketRecord,
    "adjudication_decision": AdjudicationDecisionRecord,
    "run_accession_inventory": RunAccessionInventoryRecord,
    "run_contrast_scope": RunContrastScopeRecord,
    "run_accession_map": RunAccessionMapRecord,
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


def _canonical_record_sha256(record: StrictRecord) -> str:
    payload = json.dumps(
        record.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_record_sha256(
    frame: pd.DataFrame | None,
    model: type[StrictRecord],
    *,
    key_field: str,
    allow_empty: bool,
) -> tuple[pd.DataFrame, dict[str, str]]:
    if frame is None or frame.empty:
        if allow_empty:
            return pd.DataFrame(columns=list(model.model_fields)), {}
        raise ValueError(f"{model.__name__} release table contains no records")
    valid, errors = validate_records(frame, model)
    if not errors.empty or len(valid) != len(frame):
        raise ValueError(
            f"{model.__name__} release rows failed contract validation: "
            f"{errors.head(5).to_dict(orient='records')}"
        )
    hashes: dict[str, str] = {}
    for _, row in valid.iterrows():
        parsed = model.model_validate(
            {
                key: (None if pd.isna(value) else value)
                for key, value in row.to_dict().items()
            }
        )
        key = str(getattr(parsed, key_field))
        if key in hashes:
            raise ValueError(f"duplicate release record key: {key}")
        hashes[key] = _canonical_record_sha256(parsed)
    return valid, hashes


@dataclass(frozen=True)
class _VerifiedAdjudicationRelease:
    manifest_sha256: str
    decisions: pd.DataFrame
    validation_events: pd.DataFrame
    decision_record_sha256: dict[str, str]
    validation_event_record_sha256: dict[str, str]


def _verify_adjudication_release_manifest(
    manifest_path: str | Path,
    expected_sha256: str,
    *,
    adjudication_decisions: pd.DataFrame | None,
    validation_events: pd.DataFrame | None,
) -> _VerifiedAdjudicationRelease:
    expected = expected_sha256.strip()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError(
            "expected_adjudication_release_manifest_sha256 must be a lowercase SHA-256"
        )
    content = Path(manifest_path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(
            "adjudication release manifest SHA-256 does not match expected"
        )
    try:
        manifest = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "adjudication release manifest is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError("adjudication release manifest must be a JSON object")
    required_state = {
        "schema": "crispr-evidencerank.adjudication-release-manifest",
        "schema_version": 1,
        "method_version": "validation_adjudication_v1",
        "status": "human_adjudication_released_without_readiness_promotion",
        "benchmark_ready_count": 0,
    }
    for key, required in required_state.items():
        if manifest.get(key) != required:
            raise ValueError(f"adjudication release manifest has invalid {key}")

    valid_decisions, decision_hashes = _validated_record_sha256(
        adjudication_decisions,
        AdjudicationDecisionRecord,
        key_field="decision_id",
        allow_empty=False,
    )
    valid_events, event_hashes = _validated_record_sha256(
        validation_events,
        ValidationEventRecord,
        key_field="event_id",
        allow_empty=True,
    )
    declared_hashes = manifest.get("record_sha256")
    if not isinstance(declared_hashes, dict):
        raise ValueError("adjudication release manifest omits record_sha256")
    if declared_hashes.get("decisions") != decision_hashes:
        raise ValueError("decision records differ from the pinned adjudication release")
    if declared_hashes.get("validation_events") != event_hashes:
        raise ValueError(
            "validation-event records differ from the pinned adjudication release"
        )
    packet_item_hashes = declared_hashes.get("packet_items")
    if not isinstance(packet_item_hashes, dict) or any(
        not isinstance(item_id, str)
        or not isinstance(row_hash, str)
        or len(row_hash) != 64
        or any(character not in "0123456789abcdef" for character in row_hash)
        for item_id, row_hash in (
            packet_item_hashes.items() if isinstance(packet_item_hashes, dict) else []
        )
    ):
        raise ValueError("adjudication release has invalid packet-item hashes")
    if manifest.get("decision_count") != len(valid_decisions):
        raise ValueError("adjudication release decision_count is inconsistent")
    if manifest.get("packet_item_count") != len(packet_item_hashes):
        raise ValueError("adjudication release packet-item count is inconsistent")
    if manifest.get("released_event_count") != len(valid_events):
        raise ValueError("adjudication release event count is inconsistent")

    decision_packet_items = valid_decisions["packet_item_id"].astype(str)
    if decision_packet_items.duplicated().any() or set(decision_packet_items) != set(
        packet_item_hashes
    ):
        raise ValueError(
            "adjudication decisions do not cover packet items exactly once"
        )

    released = valid_decisions.loc[
        valid_decisions["disposition"]
        .astype(str)
        .eq(AdjudicationDecisionDisposition.RELEASE_VALIDATION_EVENT.value)
    ]
    released_event_ids = set(released["validation_event_id"].astype(str))
    if released["validation_event_id"].astype(str).duplicated().any() or (
        released_event_ids != set(event_hashes)
    ):
        raise ValueError("adjudication release graph is incomplete or inconsistent")
    released_by_event = {
        str(row["validation_event_id"]): row for _, row in released.iterrows()
    }
    parent_packet = manifest.get("parent_packet_manifest")
    if not isinstance(parent_packet, dict) or not isinstance(
        parent_packet.get("sha256"), str
    ):
        raise ValueError("adjudication release omits its parent packet checksum")
    try:
        manifest_adjudicated_date = pd.Timestamp(manifest["adjudicated_date"]).date()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "adjudication release has an invalid adjudicated_date"
        ) from exc
    parsed_events = {
        str(row["event_id"]): ValidationEventRecord.model_validate(
            {
                key: (None if pd.isna(value) else value)
                for key, value in row.to_dict().items()
            }
        )
        for _, row in valid_events.iterrows()
    }
    for _, row in valid_decisions.iterrows():
        decision = AdjudicationDecisionRecord.model_validate(
            {
                key: (None if pd.isna(value) else value)
                for key, value in row.to_dict().items()
            }
        )
        if (
            decision.adjudicated_date != manifest_adjudicated_date
            or decision.packet_manifest_sha256 != parent_packet["sha256"]
            or decision.packet_id != manifest.get("packet_id")
        ):
            raise ValueError(
                "adjudication release mixes decisions from different packets or dates"
            )
    for event_id, event_hash in event_hashes.items():
        decision = AdjudicationDecisionRecord.model_validate(
            {
                key: (None if pd.isna(value) else value)
                for key, value in released_by_event[event_id].to_dict().items()
            }
        )
        event = parsed_events[event_id]
        if decision.validation_event_row_sha256 != event_hash:
            raise ValueError(
                "adjudication decision does not bind its released event row"
            )
        graph_matches = (
            event.adjudication_decision_id == decision.decision_id
            and event.screen_id == decision.screen_id
            and event.gene_symbol == decision.gene_symbol
            and event.adjudication_packet_id == decision.packet_id
            and event.review_comparison_id == decision.comparison_id
            and _normalized_claim_text(event.curator or "")
            == _normalized_claim_text(decision.adjudicator_name)
            and event.adjudication_method_version == "validation_adjudication_v1"
            and event.adjudication_status == "consensus_adjudicated"
            and event.evidence_available_date is not None
            and event.evidence_available_date <= decision.adjudicated_date
            and decision.adjudicated_date == manifest_adjudicated_date
            and decision.packet_manifest_sha256 == parent_packet["sha256"]
            and decision.packet_id == manifest.get("packet_id")
        )
        if not graph_matches:
            raise ValueError(
                "adjudication release contains an inconsistent decision/event graph"
            )
    disposition_counts = {
        str(key): int(value)
        for key, value in valid_decisions["disposition"]
        .astype(str)
        .value_counts()
        .sort_index()
        .items()
    }
    label_counts = {
        str(key): int(value)
        for key, value in valid_events.get("label_code", pd.Series(dtype=str))
        .astype(str)
        .value_counts()
        .sort_index()
        .items()
    }
    if manifest.get("disposition_counts") != disposition_counts:
        raise ValueError("adjudication release disposition counts are inconsistent")
    if manifest.get("label_counts") != label_counts:
        raise ValueError("adjudication release label counts are inconsistent")
    primary_count = int(
        valid_events.get("label_code", pd.Series(dtype=str))
        .astype(str)
        .isin({"V2", "V3", "F0", "D"})
        .sum()
    )
    if manifest.get("released_primary_label_count") != primary_count:
        raise ValueError("adjudication release primary-label count is inconsistent")
    return _VerifiedAdjudicationRelease(
        manifest_sha256=expected,
        decisions=valid_decisions,
        validation_events=valid_events,
        decision_record_sha256=decision_hashes,
        validation_event_record_sha256=event_hashes,
    )


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
_BENCHMARK_ADJUDICATION_STATUSES = BENCHMARK_ADJUDICATION_STATUSES


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
    adjudication_decisions: pd.DataFrame | None,
    verified_adjudication_release: _VerifiedAdjudicationRelease | None,
    data_assets: pd.DataFrame | None,
) -> list[str]:
    """Derive policy-v2 readiness gates from linked registry facts.

    Passing eligibility rows are curator assertions. This check independently
    requires the records that those assertions summarize.
    """

    failures: list[str] = []
    screen_rows = _screen_records(screens, screen_id)
    screen_row = screen_rows[0] if len(screen_rows) == 1 else {}
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

    event_records = _screen_records(validation_events, screen_id)
    parsed_events: list[ValidationEventRecord] = []
    invalid_event_present = False
    for event in event_records:
        clean_event = {
            key: (None if pd.isna(value) else value) for key, value in event.items()
        }
        try:
            parsed_events.append(ValidationEventRecord.model_validate(clean_event))
        except (TypeError, ValueError):
            invalid_event_present = True

    parsed_decisions: list[AdjudicationDecisionRecord] = []
    invalid_decision_present = False
    for decision in _screen_records(adjudication_decisions, screen_id):
        clean_decision = {
            key: (None if pd.isna(value) else value) for key, value in decision.items()
        }
        try:
            parsed_decisions.append(
                AdjudicationDecisionRecord.model_validate(clean_decision)
            )
        except (TypeError, ValueError):
            invalid_decision_present = True
    released_decisions = [
        decision
        for decision in parsed_decisions
        if decision.disposition
        == AdjudicationDecisionDisposition.RELEASE_VALIDATION_EVENT
    ]
    released_event_ids = [
        str(decision.validation_event_id) for decision in released_decisions
    ]
    if len(released_event_ids) != len(set(released_event_ids)):
        invalid_decision_present = True
    parsed_event_ids = [event.event_id for event in parsed_events]
    if len(parsed_event_ids) != len(set(parsed_event_ids)):
        invalid_event_present = True
    if set(released_event_ids) != set(parsed_event_ids):
        invalid_decision_present = True
    decision_by_event = {
        str(decision.validation_event_id): decision for decision in released_decisions
    }
    if verified_adjudication_release is None:
        invalid_decision_present = True

    primary_event_present = False
    for parsed in parsed_events:
        event_contrast = _fact_text(parsed.contrast_id)
        event_gene = _fact_text(parsed.gene_symbol)
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
        event_direction = _fact_token(parsed.phenotype_direction)
        if directions != {event_direction}:
            continue
        contrast_matches = [
            contrast
            for contrast in contrast_rows
            if _fact_text(contrast.get("contrast_id")) == event_contrast
        ]
        if len(contrast_matches) != 1:
            continue
        contrast = contrast_matches[0]
        expected_direction = _fact_token(contrast.get("intended_direction"))
        if expected_direction != event_direction:
            continue
        if _fact_token(parsed.drug_name) != _fact_token(contrast.get("treatment_name")):
            continue
        if _fact_token(parsed.cell_line) != _fact_token(screen_row.get("cell_line")):
            continue
        if _fact_token(parsed.perturbation_modality) != _fact_token(
            screen_row.get("perturbation_modality")
        ):
            continue
        if _fact_text(parsed.study_id) != _fact_text(screen_row.get("study_id")):
            continue
        if _fact_text(parsed.source_family_id) != source_family:
            continue
        decision = decision_by_event.get(parsed.event_id)
        if decision is None:
            continue
        if verified_adjudication_release is None:
            continue
        if (
            decision.decision_id
            not in verified_adjudication_release.decision_record_sha256
            or parsed.event_id
            not in verified_adjudication_release.validation_event_record_sha256
        ):
            continue
        if (
            decision.decision_id != parsed.adjudication_decision_id
            or decision.packet_id != parsed.adjudication_packet_id
            or decision.comparison_id != parsed.review_comparison_id
            or decision.screen_id != parsed.screen_id
            or decision.gene_symbol != parsed.gene_symbol
            or _normalized_claim_text(decision.adjudicator_name)
            != _normalized_claim_text(parsed.curator or "")
        ):
            continue
        event_row_sha256 = hashlib.sha256(
            json.dumps(
                parsed.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if decision.validation_event_row_sha256 != event_row_sha256:
            continue
        if (
            parsed.evidence_available_date is None
            or parsed.evidence_available_date > decision.adjudicated_date
            or parsed.adjudication_method_version != "validation_adjudication_v1"
        ):
            continue
        adjudication_status = _fact_token(parsed.adjudication_status)
        if (
            parsed.label_code in _PRIMARY_VALIDATION_LABELS
            and adjudication_status == "consensus_adjudicated"
        ):
            primary_event_present = True
            break
    if invalid_event_present or invalid_decision_present or not primary_event_present:
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
    adjudication_decisions: pd.DataFrame | None = None,
    adjudication_release_manifest: str | Path | None = None,
    expected_adjudication_release_manifest_sha256: str | None = None,
    evidence: pd.DataFrame | None = None,
    external_screen_maps: pd.DataFrame | None = None,
    screen_intake: pd.DataFrame | None = None,
    eligibility_checks: pd.DataFrame | None = None,
    design_provenance: pd.DataFrame | None = None,
    data_assets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Check core foreign-key relationships across normalized registry tables."""

    errors: list[dict[str, Any]] = []
    verified_adjudication_release: _VerifiedAdjudicationRelease | None = None
    release_arguments_present = (
        adjudication_release_manifest is not None,
        expected_adjudication_release_manifest_sha256 is not None,
    )
    if any(release_arguments_present) and not all(release_arguments_present):
        errors.append(
            {
                "table": "adjudication_release_manifest",
                "row_number": 1,
                "error": (
                    "both adjudication_release_manifest and its expected SHA-256 "
                    "are required"
                ),
            }
        )
    elif all(release_arguments_present):
        try:
            verified_adjudication_release = _verify_adjudication_release_manifest(
                adjudication_release_manifest,
                str(expected_adjudication_release_manifest_sha256),
                adjudication_decisions=adjudication_decisions,
                validation_events=validation_events,
            )
        except (OSError, TypeError, ValueError) as exc:
            errors.append(
                {
                    "table": "adjudication_release_manifest",
                    "row_number": 1,
                    "error": str(exc),
                }
            )

    effective_validation_events = (
        verified_adjudication_release.validation_events
        if verified_adjudication_release is not None
        else validation_events
    )
    effective_adjudication_decisions = (
        verified_adjudication_release.decisions
        if verified_adjudication_release is not None
        else adjudication_decisions
    )
    for table_name, table, model in (
        ("validation_events", validation_events, ValidationEventRecord),
        (
            "adjudication_decisions",
            adjudication_decisions,
            AdjudicationDecisionRecord,
        ),
    ):
        if table is None or table.empty:
            continue
        _, contract_errors = validate_records(table, model)
        for contract_error in contract_errors.to_dict(orient="records"):
            errors.append(
                {
                    "table": table_name,
                    "row_number": contract_error["row_number"],
                    "error": contract_error["error"],
                }
            )
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
        ("adjudication_decisions", adjudication_decisions),
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
                    validation_events=effective_validation_events,
                    adjudication_decisions=effective_adjudication_decisions,
                    verified_adjudication_release=(verified_adjudication_release),
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

    if candidates is not None:
        non_u_candidate_present = bool(
            candidates.get("label_code", pd.Series(dtype=str))
            .astype(str)
            .ne(LabelCode.U.value)
            .any()
        )
        validation_event_rows_present = (
            validation_events is not None and not validation_events.empty
        )
        release_required = non_u_candidate_present or validation_event_rows_present
        if release_required and verified_adjudication_release is None:
            errors.append(
                {
                    "table": "validation_events",
                    "row_number": 1,
                    "error": (
                        "candidate labels require a checksum-verified "
                        "adjudication release manifest"
                    ),
                }
            )
        elif verified_adjudication_release is not None:
            try:
                adjudicated = _resolve_released_validation_events(
                    effective_validation_events,
                    effective_adjudication_decisions,
                )
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
                    elif str(candidate_row["label_code"]) != str(
                        event_row["label_code"]
                    ):
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
                                    "non-U candidate has no linked validation "
                                    f"event: {key}"
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
