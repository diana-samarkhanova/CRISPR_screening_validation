"""Conservative adapters for BioGRID ORCS 2.0.x tabular exports.

BioGRID ORCS preserves heterogeneous, author-reported gene-level scores.  This
module therefore keeps source values in a raw table and performs only
structural normalization.  In particular, it does not infer an experimental
comparator, translate score signs into biological directions, or treat an
author-defined screen hit as an orthogonal validation outcome.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import quote

import pandas as pd

from .contracts import (
    ContrastRecord,
    ControlType,
    ExposureSchedule,
    ExternalScreenMapRecord,
    GeneScoreRecord,
    PerturbationModality,
    PhenotypeDirection,
    ScreenDesignRecord,
    ScreenRecord,
    ScreenScale,
    SelectionStrategy,
    SourceRole,
    StrictRecord,
    StudyRecord,
)

TableSource = str | Path | TextIO | pd.DataFrame
IndexMetadata = pd.DataFrame | pd.Series | Mapping[str, Any]

_HEADER_ALIASES = {
    "dataset_id": "external_dataset_id",
    "methodology": "library_methodology",
    "score_column_count": "score_col_count",
}
_DOSE_RE = re.compile(
    r"^(?P<value>(?:\d+(?:\.\d*)?|\.\d+))"
    r"\s+"
    r"(?P<unit>[A-Za-zµμ%][A-Za-z0-9µμ%/_.^-]*)$"
)
_DURATION_DAYS_RE = re.compile(
    r"^(?P<value>(?:\d+(?:\.\d*)?|\.\d+))\s+days?$",
    flags=re.IGNORECASE,
)
_PLAIN_NUMBER_RE = re.compile(r"^(?:\d+(?:\.\d*)?|\.\d+)$")


@dataclass(frozen=True)
class OrcsIndexParseResult:
    """Normalized registry tables plus the lossless source index."""

    release: str
    header_map: dict[str, str]
    raw_index: pd.DataFrame
    normalized_index: pd.DataFrame
    studies: pd.DataFrame
    screens: pd.DataFrame
    screen_designs: pd.DataFrame
    contrasts: pd.DataFrame
    external_screen_maps: pd.DataFrame


@dataclass(frozen=True)
class OrcsGeneScoreParseResult:
    """Normalized gene-score records plus source rows and parse issues."""

    release: str
    header_map: dict[str, str]
    raw_scores: pd.DataFrame
    normalized_scores: pd.DataFrame
    gene_scores: pd.DataFrame
    issues: pd.DataFrame


@dataclass(frozen=True)
class OrcsModalityAssessment:
    """One conservative interpretation of ORCS library metadata."""

    modality: PerturbationModality
    conflict: bool
    explicit_non_ko: bool
    observed_value: str | None


def normalize_orcs_header(header: object) -> str:
    """Return a stable snake-case name for a dynamic ORCS header."""

    value = str(header).lstrip("\ufeff").strip()
    if value.startswith("#"):
        value = value[1:].strip()
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    if not value:
        raise ValueError("ORCS table contains an empty header")
    return _HEADER_ALIASES.get(value, value)


def _read_table(source: TableSource) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        frame = source.copy(deep=True)
    else:
        frame = pd.read_csv(
            source,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            na_filter=False,
        )
    if frame.empty:
        raise ValueError("ORCS table contains no records")
    return frame


def _normalize_table(
    source: TableSource,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    source_frame = _read_table(source)
    original_headers = [str(column) for column in source_frame.columns]
    normalized_headers = [normalize_orcs_header(column) for column in original_headers]
    duplicated = sorted(
        {
            header
            for header in normalized_headers
            if normalized_headers.count(header) > 1
        }
    )
    if duplicated:
        raise ValueError(f"ORCS headers collide after normalization: {duplicated}")

    raw = source_frame.copy(deep=True)
    raw.columns = normalized_headers
    normalized = raw.copy(deep=True)
    for column in normalized:
        normalized[column] = normalized[column].map(_normalize_scalar)
    return raw, normalized, dict(zip(original_headers, normalized_headers, strict=True))


def _normalize_scalar(value: object) -> object | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"", "-"}:
            return None
        return stripped
    return value


def _text(value: object) -> str | None:
    value = _normalize_scalar(value)
    return None if value is None else str(value)


def _required_text(value: object) -> str:
    return _text(value) or "unknown"


def _release_value(release: str) -> str:
    value = release.strip()
    if not value:
        raise ValueError("ORCS release cannot be empty")
    return value


def _retrieved_date_value(value: date | str) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _optional_date_value(value: date | str | None) -> date | None:
    if value is None:
        return None
    return _retrieved_date_value(value)


def orcs_screen_id(release: str, external_screen_id: str) -> str:
    """Return a release-qualified internal screen identifier."""

    return f"orcs:{_release_value(release)}:screen:{external_screen_id.strip()}"


def orcs_contrast_id(release: str, external_screen_id: str) -> str:
    """Return the conservative, unpaired contrast identifier for a screen."""

    return f"{orcs_screen_id(release, external_screen_id)}:contrast:reported"


def _study_id(release: str, source_type: str, source_id: str) -> str:
    return f"orcs:{release}:study:{source_type.strip().lower()}:{source_id.strip()}"


def _provisional_study_id(release: str, external_screen_id: str) -> str:
    return f"orcs:{release}:study:provisional-screen:{external_screen_id.strip()}"


def _parse_exact_dose(value: object) -> tuple[float | None, str | None]:
    text = _text(value)
    if text is None:
        return None, None
    match = _DOSE_RE.fullmatch(text)
    if match is None:
        return None, None
    number = float(match.group("value"))
    if not math.isfinite(number) or number < 0:
        return None, None
    unit = match.group("unit").replace("μ", "µ")
    return number, unit


def _parse_duration_days(value: object) -> float | None:
    text = _text(value)
    if text is None:
        return None
    match = _DURATION_DAYS_RE.fullmatch(text)
    if match is None:
        return None
    number = float(match.group("value"))
    return number if math.isfinite(number) and number >= 0 else None


def _parse_exact_nonnegative_number(value: object) -> float | None:
    text = _text(value)
    if text is None or _PLAIN_NUMBER_RE.fullmatch(text) is None:
        return None
    number = float(text)
    return number if math.isfinite(number) and number >= 0 else None


def classify_orcs_modality(
    library_type: object,
    library_methodology: object,
) -> OrcsModalityAssessment:
    """Classify ORCS perturbation metadata without resolving contradictions.

    A row containing incompatible modality signals is deliberately returned as
    ``OTHER``. In particular, a stray ``Knockout`` methodology cannot turn a
    CRISPRa/CRISPRi or base-editing screen into a CRISPR-Cas9 KO screen.
    """

    values = [
        value for value in (_text(library_type), _text(library_methodology)) if value
    ]
    observed = " | ".join(values) if values else None
    searchable = " ".join(value.casefold() for value in values)

    has_activation = "crispra" in searchable or "activation" in searchable
    has_inhibition = "crispri" in searchable or "inhibition" in searchable
    has_knockout = (
        "crisprn" in searchable or "crispr ko" in searchable or "knockout" in searchable
    )
    has_explicit_other = any(
        token in searchable
        for token in (
            "base edit",
            "prime edit",
            "rnai",
            "shrna",
            "sirna",
        )
    )

    if has_explicit_other:
        return OrcsModalityAssessment(
            modality=PerturbationModality.OTHER,
            conflict=(has_activation or has_inhibition),
            explicit_non_ko=True,
            observed_value=observed,
        )

    observed_modalities = sum((has_activation, has_inhibition, has_knockout))
    if observed_modalities > 1:
        return OrcsModalityAssessment(
            modality=PerturbationModality.OTHER,
            conflict=True,
            explicit_non_ko=False,
            observed_value=observed,
        )
    if has_activation:
        modality = PerturbationModality.CRISPRA
    elif has_inhibition:
        modality = PerturbationModality.CRISPRI
    elif has_knockout:
        modality = PerturbationModality.CRISPR_KO
    else:
        modality = PerturbationModality.OTHER
    return OrcsModalityAssessment(
        modality=modality,
        conflict=False,
        explicit_non_ko=False,
        observed_value=observed,
    )


def _perturbation_modality(row: Mapping[str, object]) -> PerturbationModality:
    return classify_orcs_modality(
        row.get("library_type"),
        row.get("library_methodology"),
    ).modality


def _selection_strategy(value: object) -> SelectionStrategy:
    normalized = (_text(value) or "").casefold()
    if normalized == "positive selection":
        return SelectionStrategy.POSITIVE
    if normalized == "negative selection":
        return SelectionStrategy.NEGATIVE
    if normalized in {
        "positive and negative selection",
        "positive/negative selection",
        "bidirectional selection",
    }:
        return SelectionStrategy.BIDIRECTIONAL
    if normalized:
        return SelectionStrategy.OTHER
    return SelectionStrategy.UNKNOWN


def _screen_scale(value: object) -> ScreenScale:
    normalized = (_text(value) or "").casefold().replace("-", " ")
    if "genome" in normalized and "wide" in normalized:
        return ScreenScale.GENOME_WIDE
    if "target" in normalized or "focused" in normalized:
        return ScreenScale.TARGETED
    if "saturation" in normalized or "tiling" in normalized:
        return ScreenScale.SATURATION
    if normalized:
        return ScreenScale.OTHER
    return ScreenScale.UNKNOWN


def _record_frame(
    records: list[dict[str, object]],
    model: type[StrictRecord],
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=list(model.model_fields))
    parsed = [
        model.model_validate(record).model_dump(mode="json") for record in records
    ]
    return pd.DataFrame.from_records(parsed)


def _notes(*parts: str | None) -> str | None:
    retained = [part for part in parts if part]
    return " | ".join(retained) if retained else None


def parse_orcs_index(
    source: TableSource,
    *,
    release: str,
    retrieved_date: date | str,
    available_date: date | str | None = None,
    organism_scope: str | None = None,
) -> OrcsIndexParseResult:
    """Parse an ORCS screen index into conservative registry records.

    One contrast is emitted for every ORCS screen, but its comparator, control
    type, exposure schedule, and biological direction remain unknown.  ORCS
    does not encode the experimental arm pairing needed to infer those fields.
    """

    release = _release_value(release)
    retrieved_date = _retrieved_date_value(retrieved_date)
    available_date = _optional_date_value(available_date)
    if available_date is not None and available_date > retrieved_date:
        raise ValueError("ORCS available_date cannot follow retrieved_date")
    organism_scope = _text(organism_scope)
    raw, normalized, header_map = _normalize_table(source)
    if "screen_id" not in normalized:
        raise ValueError("ORCS index is missing SCREEN_ID")

    external_ids = normalized["screen_id"].map(_text)
    if external_ids.isna().any():
        raise ValueError("ORCS index contains a missing SCREEN_ID")
    if external_ids.duplicated().any():
        duplicated = sorted(set(external_ids[external_ids.duplicated(False)]))
        raise ValueError(
            f"ORCS index contains duplicate SCREEN_ID values: {duplicated}"
        )

    studies_by_id: dict[str, dict[str, object]] = {}
    screens: list[dict[str, object]] = []
    designs: list[dict[str, object]] = []
    contrasts: list[dict[str, object]] = []
    mappings: list[dict[str, object]] = []

    for row in normalized.to_dict(orient="records"):
        external_screen_id = _required_text(row.get("screen_id"))
        internal_screen_id = orcs_screen_id(release, external_screen_id)
        source_id = _text(row.get("source_id"))
        source_type = _text(row.get("source_type")) or "unknown"
        if source_id is None:
            study_id = _provisional_study_id(release, external_screen_id)
            source_family_id = None
        else:
            study_id = _study_id(release, source_type, source_id)
            source_family_id = (
                f"orcs:source-family:{source_type.strip().lower()}:{source_id.strip()}"
            )
        external_dataset_id = _text(row.get("external_dataset_id"))
        author = _text(row.get("author"))
        organism = _text(row.get("organism_official")) or organism_scope or "unknown"
        pubmed_id = (
            source_id
            if source_id is not None and source_type.casefold() == "pubmed"
            else None
        )
        source_identity_note = (
            None
            if source_id is not None
            else (
                "SOURCE ID was not reported; the internal study identity is "
                "provisional and no source-family mapping was inferred."
            )
        )
        studies_by_id.setdefault(
            study_id,
            {
                "study_id": study_id,
                "citation": (
                    author
                    or (
                        f"BioGRID ORCS source {source_id}"
                        if source_id is not None
                        else f"BioGRID ORCS screen {external_screen_id}"
                    )
                ),
                "organism": organism,
                # ORCS indexing does not establish source-file licensing.
                # Rights must be curated per study/data asset.
                "data_license": None,
                "source_role": SourceRole.ORIGINAL_SCREEN,
                "independent_screen_source": True,
                "source_id": source_id,
                "source_type": source_type,
                "pubmed_id": pubmed_id,
                "notes": (
                    "Study metadata imported from the BioGRID ORCS "
                    f"{release} screen index."
                    + (
                        f" {source_identity_note}"
                        if source_identity_note is not None
                        else ""
                    )
                ),
            },
        )

        dose, dose_unit = _parse_exact_dose(row.get("condition_dosage"))
        duration_days = _parse_duration_days(row.get("duration"))
        screen_type = _text(row.get("screen_type"))
        experimental_setup = _text(row.get("experimental_setup"))
        condition_name = _text(row.get("condition_name"))
        rationale = _text(row.get("screen_rationale"))
        source_notes = _text(row.get("notes"))
        screens.append(
            {
                "screen_id": internal_screen_id,
                "study_id": study_id,
                "source_family_id": source_family_id,
                # Publication/dataset membership does not prove reuse of the
                # same underlying raw screen.
                "raw_data_family_id": None,
                "perturbation_modality": _perturbation_modality(row),
                "screen_design": screen_type or "unknown",
                "cell_line": _required_text(row.get("cell_line")),
                "drug_name": (
                    condition_name
                    if (experimental_setup or "").casefold() == "drug exposure"
                    else None
                ),
                "library_name": _text(row.get("library")),
                "treatment_dose": dose,
                "treatment_unit": dose_unit,
                "duration_days": duration_days,
                "intended_direction": PhenotypeDirection.UNKNOWN,
                "input_mode": "orcs_gene_scores",
                "source_url": (
                    "https://orcs.thebiogrid.org/Screen/"
                    f"{quote(external_screen_id, safe='')}"
                ),
                "available_date": available_date,
                "notes": _notes(
                    source_notes,
                    f"ORCS screen rationale: {rationale}" if rationale else None,
                ),
            }
        )

        designs.append(
            {
                "screen_id": internal_screen_id,
                "source_role": SourceRole.ORIGINAL_SCREEN,
                "screen_scale": _screen_scale(row.get("throughput")),
                "screen_format": _text(row.get("screen_format")),
                "experimental_setup": experimental_setup,
                "selection_strategy": _selection_strategy(screen_type),
                "library_type": _text(row.get("library_type")),
                "library_methodology": _text(row.get("library_methodology")),
                "enzyme": _text(row.get("enzyme")),
                "library_moi": _parse_exact_nonnegative_number(row.get("moi")),
                "analysis_method": _text(row.get("analysis")),
                "source_locator": (
                    f"BioGRID ORCS {release} screen index; "
                    f"SCREEN_ID={external_screen_id}"
                ),
                "notes": _notes(
                    f"ORCS screen rationale: {rationale}" if rationale else None,
                    (
                        "ORCS significance criteria: "
                        f"{_text(row.get('significance_criteria'))}"
                    )
                    if _text(row.get("significance_criteria"))
                    else None,
                ),
            }
        )

        contrast_id = orcs_contrast_id(release, external_screen_id)
        screen_name = _text(row.get("screen_name"))
        phenotype = _text(row.get("phenotype"))
        contrasts.append(
            {
                "screen_id": internal_screen_id,
                "contrast_id": contrast_id,
                "contrast_name": (screen_name or f"ORCS screen {external_screen_id}"),
                "treatment_name": condition_name or "unknown",
                "treatment_dose": dose,
                "treatment_unit": dose_unit,
                "control_type": ControlType.UNKNOWN,
                "exposure_schedule": ExposureSchedule.UNKNOWN,
                "phenotype_endpoint": phenotype or "unknown",
                "intended_direction": PhenotypeDirection.UNKNOWN,
                "source_locator": (
                    f"BioGRID ORCS {release} screen index; "
                    f"SCREEN_ID={external_screen_id}"
                ),
                "notes": (
                    "Comparator and biological score direction are not "
                    "represented explicitly in the ORCS index and were not "
                    "inferred."
                ),
            }
        )

        mappings.append(
            {
                "mapping_id": (f"orcs:{release}:external-screen:{external_screen_id}"),
                "screen_id": internal_screen_id,
                "source_name": "BioGRID ORCS",
                "source_version": release,
                "external_dataset_id": external_dataset_id,
                "external_screen_id": external_screen_id,
                "relationship": "source_record",
                "source_url": (
                    "https://orcs.thebiogrid.org/Screen/"
                    f"{quote(external_screen_id, safe='')}"
                ),
                "retrieved_date": retrieved_date,
                "notes": (
                    "External identity is the pair "
                    "(source_version, external_screen_id)."
                ),
            }
        )

    return OrcsIndexParseResult(
        release=release,
        header_map=header_map,
        raw_index=raw,
        normalized_index=normalized,
        studies=_record_frame(list(studies_by_id.values()), StudyRecord),
        screens=_record_frame(screens, ScreenRecord),
        screen_designs=_record_frame(designs, ScreenDesignRecord),
        contrasts=_record_frame(contrasts, ContrastRecord),
        external_screen_maps=_record_frame(mappings, ExternalScreenMapRecord),
    )


def _normalized_metadata(
    metadata: IndexMetadata | None,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    if metadata is None:
        return {}, {}
    if isinstance(metadata, pd.DataFrame):
        _, normalized, _ = _normalize_table(metadata)
        if "screen_id" not in normalized:
            raise ValueError("ORCS index metadata dataframe is missing SCREEN_ID")
        lookup: dict[str, dict[str, object]] = {}
        for row in normalized.to_dict(orient="records"):
            screen_id = _text(row.get("screen_id"))
            if screen_id is not None:
                lookup[screen_id] = row
        return lookup, {}

    items = metadata.to_dict() if isinstance(metadata, pd.Series) else metadata
    normalized_mapping = {
        normalize_orcs_header(key): _normalize_scalar(value)
        for key, value in items.items()
    }
    screen_id = _text(normalized_mapping.get("screen_id"))
    if screen_id is None:
        return {}, normalized_mapping
    return {screen_id: normalized_mapping}, {}


def _parse_author_hit(value: object) -> bool | None:
    text = _text(value)
    if text is None:
        return None
    normalized = text.casefold()
    if normalized == "yes":
        return True
    if normalized == "no":
        return False
    if normalized in {"n/a", "na"}:
        return None
    raise ValueError(f"unrecognized ORCS HIT value: {text}")


def _finite_float(value: object) -> float:
    text = _text(value)
    if text is None:
        raise ValueError("score is missing")
    number = float(text)
    if not math.isfinite(number):
        raise ValueError("score must be finite")
    return number


def _issue(
    *,
    row_number: int,
    external_screen_id: str | None,
    gene_symbol: str | None,
    field: str,
    value: object,
    error: str,
) -> dict[str, object]:
    return {
        "row_number": row_number,
        "external_screen_id": external_screen_id,
        "gene_symbol": gene_symbol,
        "field": field,
        "value": value,
        "error": error,
    }


def parse_orcs_screen_scores(
    source: TableSource,
    *,
    release: str,
    index_metadata: IndexMetadata | None = None,
    contrast_id: str | None = None,
    source_file: str | None = None,
) -> OrcsGeneScoreParseResult:
    """Parse one or more ORCS per-screen files into gene-score records.

    Every populated ``Score.N`` column becomes a distinct record.  ``HIT`` is
    copied only to :attr:`GeneScoreRecord.author_hit`; it is not converted into
    a validation event, testing status, or validation label.
    """

    release = _release_value(release)
    if source_file is None and isinstance(source, (str, Path)):
        source_file = Path(source).name
    raw, normalized, header_map = _normalize_table(source)
    required = {"screen_id", "official_symbol"}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise ValueError(f"ORCS screen table is missing required columns: {missing}")

    score_columns = sorted(
        (column for column in normalized if re.fullmatch(r"score_\d+", column)),
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )
    metadata_by_screen, default_metadata = _normalized_metadata(index_metadata)
    records: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []

    for row_number, row in enumerate(normalized.to_dict(orient="records"), start=2):
        external_screen_id = _text(row.get("screen_id"))
        gene_symbol = _text(row.get("official_symbol"))
        if external_screen_id is None:
            issues.append(
                _issue(
                    row_number=row_number,
                    external_screen_id=None,
                    gene_symbol=gene_symbol,
                    field="screen_id",
                    value=row.get("screen_id"),
                    error="missing SCREEN_ID; row was not imported",
                )
            )
            continue
        if gene_symbol is None:
            issues.append(
                _issue(
                    row_number=row_number,
                    external_screen_id=external_screen_id,
                    gene_symbol=None,
                    field="official_symbol",
                    value=row.get("official_symbol"),
                    error=(
                        "missing OFFICIAL_SYMBOL; identifier was not "
                        "silently substituted"
                    ),
                )
            )
            continue

        metadata = metadata_by_screen.get(external_screen_id, default_metadata)
        method = _text(metadata.get("analysis")) or ("ORCS author-reported")
        try:
            author_hit = _parse_author_hit(row.get("hit"))
        except ValueError as exc:
            issues.append(
                _issue(
                    row_number=row_number,
                    external_screen_id=external_screen_id,
                    gene_symbol=gene_symbol,
                    field="hit",
                    value=row.get("hit"),
                    error=str(exc),
                )
            )
            author_hit = None

        internal_screen_id = orcs_screen_id(release, external_screen_id)
        resolved_contrast_id = contrast_id or orcs_contrast_id(
            release, external_screen_id
        )
        score_record_created = False
        for score_column in score_columns:
            raw_score = row.get(score_column)
            if _text(raw_score) is None:
                continue
            try:
                score = _finite_float(raw_score)
            except (TypeError, ValueError) as exc:
                issues.append(
                    _issue(
                        row_number=row_number,
                        external_screen_id=external_screen_id,
                        gene_symbol=gene_symbol,
                        field=score_column,
                        value=raw_score,
                        error=f"{exc}; score was not imported",
                    )
                )
                continue
            suffix = score_column.rsplit("_", 1)[1]
            score_type = _text(metadata.get(f"score_{suffix}_type"))
            records.append(
                {
                    "screen_id": internal_screen_id,
                    "contrast_id": resolved_contrast_id,
                    "gene_symbol": gene_symbol,
                    "method": method,
                    "analysis_tail": f"orcs_score_{suffix}",
                    "direction": PhenotypeDirection.UNKNOWN,
                    "score": score,
                    "author_hit": author_hit,
                    "score_type": score_type,
                    "cnv_corrected": False,
                    "source_database": "BioGRID ORCS",
                    "source_screen_id": external_screen_id,
                    "source_file": source_file,
                }
            )
            score_record_created = True

        if not score_record_created and author_hit is not None:
            records.append(
                {
                    "screen_id": internal_screen_id,
                    "contrast_id": resolved_contrast_id,
                    "gene_symbol": gene_symbol,
                    "method": method,
                    "analysis_tail": "orcs_author_hit",
                    "direction": PhenotypeDirection.UNKNOWN,
                    "author_hit": author_hit,
                    "score_type": "author_hit",
                    "cnv_corrected": False,
                    "source_database": "BioGRID ORCS",
                    "source_screen_id": external_screen_id,
                    "source_file": source_file,
                }
            )
        elif not score_record_created and author_hit is None:
            issues.append(
                _issue(
                    row_number=row_number,
                    external_screen_id=external_screen_id,
                    gene_symbol=gene_symbol,
                    field="score/hit",
                    value=None,
                    error=(
                        "row has neither a finite score nor an author hit "
                        "status; no GeneScoreRecord was created"
                    ),
                )
            )

    return OrcsGeneScoreParseResult(
        release=release,
        header_map=header_map,
        raw_scores=raw,
        normalized_scores=normalized,
        gene_scores=_record_frame(records, GeneScoreRecord),
        issues=pd.DataFrame.from_records(
            issues,
            columns=[
                "row_number",
                "external_screen_id",
                "gene_symbol",
                "field",
                "value",
                "error",
            ],
        ),
    )


def parse_orcs_screen(
    source: TableSource,
    *,
    release: str,
    index_metadata: IndexMetadata | None = None,
    contrast_id: str | None = None,
    source_file: str | None = None,
) -> OrcsGeneScoreParseResult:
    """Alias for :func:`parse_orcs_screen_scores`."""

    return parse_orcs_screen_scores(
        source,
        release=release,
        index_metadata=index_metadata,
        contrast_id=contrast_id,
        source_file=source_file,
    )
