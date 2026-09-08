"""Bind prospective comparator ranks to an explicitly oriented screen report."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

JOIN_COLUMNS = ("screen_id", "contrast_id", "gene_symbol", "phenotype_direction")
QUERY_COLUMNS = ("screen_id", "contrast_id", "phenotype_direction")
MAGECK_METHODS = {
    "mageck_native_rank",
    "derived_from_mageck_fdr",
    "derived_from_mageck_p_value",
    "derived_from_mageck_score",
}
COUNT_METHOD = "absolute_median_guide_lfc"


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_regular(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            f"baseline source must be a regular non-symlink file: {path.name}"
        )
    return path.read_bytes()


def _json_mapping(content: bytes) -> dict:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate baseline manifest key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"baseline manifest contains nonfinite JSON: {value}")

    try:
        result = json.loads(
            content, object_pairs_hook=unique_pairs, parse_constant=reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("baseline manifest must contain valid JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("baseline manifest must be a JSON object")
    return result


def _canonical_keys(frame: pd.DataFrame, name: str) -> None:
    if not frame.columns.is_unique or frame.empty:
        raise ValueError(f"{name} must be nonempty with unique columns")
    missing = set(JOIN_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"{name} is missing join columns: {sorted(missing)}")
    for column in JOIN_COLUMNS:
        if (
            not frame[column]
            .map(
                lambda value: (
                    isinstance(value, str)
                    and bool(value)
                    and value == value.strip()
                    and re.search(r"[\x00-\x1f\x7f]", value) is None
                )
            )
            .all()
        ):
            raise ValueError(f"{name} requires canonical string join keys: {column}")
    if frame.duplicated(list(JOIN_COLUMNS)).any():
        raise ValueError(f"{name} has duplicate candidate join keys")


def _numeric_ranks(values: pd.Series, name: str) -> pd.Series:
    if values.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError(f"{name} ranks cannot be boolean")
    numeric = pd.to_numeric(values, errors="coerce")
    if (
        not np.isfinite(numeric.to_numpy(dtype=float)).all()
        or (numeric < 1).any()
        or numeric.ne(np.floor(numeric)).any()
    ):
        raise ValueError(f"{name} ranks must be finite positive integers")
    return numeric


def verify_prediction_baseline(
    frame: pd.DataFrame,
    baseline_report_dir: str | Path,
    expected_manifest_sha256: str,
) -> tuple[dict, dict[Path, bytes]]:
    """Verify exact candidate rank linkage, permitting an explicitly recorded subset.

    This establishes identity and declared orientation of the supplied report.
    It does not independently reproduce the upstream screen analysis or certify
    that the report's biological direction declaration is experimentally true.
    """
    if not isinstance(expected_manifest_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_manifest_sha256
    ):
        raise ValueError("expected baseline manifest hash must be a lowercase SHA-256")
    directory = Path(baseline_report_dir)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("baseline report must be a regular non-symlink directory")
    manifest_path = directory / "run_manifest.json"
    manifest_content = _read_regular(manifest_path)
    if _hash(manifest_content) != expected_manifest_sha256:
        raise ValueError("baseline manifest identity hash mismatch")
    manifest = _json_mapping(manifest_content)
    if manifest.get("report_type") != "screen_signal_baseline":
        raise ValueError("baseline manifest must describe screen_signal_baseline")
    try:
        record = manifest["outputs"]["ranked_candidates"]
        parameters = manifest["parameters"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "baseline manifest lacks ranked output or direction parameters"
        ) from exc
    if (
        not isinstance(record, dict)
        or record.get("filename") != "ranked_candidates.tsv"
    ):
        raise ValueError(
            "baseline ranked output must use fixed filename ranked_candidates.tsv"
        )
    if not isinstance(parameters, dict):
        raise ValueError("baseline manifest direction parameters must be a mapping")
    expected_output_hash = record.get("sha256")
    if not isinstance(expected_output_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_output_hash
    ):
        raise ValueError("baseline ranked output hash must be a lowercase SHA-256")
    ranked_path = directory / "ranked_candidates.tsv"
    ranked_content = _read_regular(ranked_path)
    if _hash(ranked_content) != expected_output_hash:
        raise ValueError("baseline ranked output hash mismatch")
    if "byte_size" in record and (
        type(record["byte_size"]) is not int
        or record["byte_size"] != len(ranked_content)
    ):
        raise ValueError("baseline ranked output byte_size mismatch")
    try:
        header = next(
            csv.reader(io.StringIO(ranked_content.decode("utf-8-sig")), delimiter="\t"),
            [],
        )
        if not header or len(header) != len(set(header)):
            raise ValueError(
                "baseline ranked output has empty or duplicate column header"
            )
        ranked = pd.read_csv(
            io.BytesIO(ranked_content), sep="\t", dtype=str, keep_default_na=False
        )
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError("baseline ranked output must contain valid UTF-8 TSV") from exc
    _canonical_keys(ranked, "baseline ranked output")
    _canonical_keys(frame, "prediction candidates")
    required = {
        "screen_signal_rank",
        "screen_signal_rank_source",
        "analysis_tail",
        "ranking_type",
    }
    if required - set(ranked):
        raise ValueError(
            "baseline ranked output lacks rank, source, tail, or ranking type"
        )
    if "screen_signal_rank" not in frame:
        raise ValueError(
            "prediction candidates require screen_signal_rank for baseline linkage"
        )
    if not ranked.ranking_type.eq("screen_signal_baseline").all():
        raise ValueError("baseline ranked rows have an inconsistent ranking_type")
    if not ranked.phenotype_direction.isin(
        {"resistance", "sensitization", "neutral"}
    ).all():
        raise ValueError("baseline ranked rows contain an invalid phenotype_direction")
    if not frame.phenotype_direction.isin({"resistance", "sensitization"}).all():
        raise ValueError("prediction baseline requires resistance or sensitization")

    positive_mappings = {}
    for parameter in ("positive_tail_means", "positive_lfc_means"):
        meaning = parameters.get(parameter)
        if meaning is not None:
            if not isinstance(meaning, str) or meaning not in {
                "resistance",
                "sensitization",
            }:
                raise ValueError(
                    f"baseline has an invalid {parameter} direction mapping"
                )
            positive_mappings[parameter] = meaning
    for method, rows in ranked.groupby("screen_signal_rank_source", sort=False):
        if method in MAGECK_METHODS:
            parameter, positive_tail, negative_tail = (
                "positive_tail_means",
                "mageck_pos",
                "mageck_neg",
            )
        elif method in {COUNT_METHOD, "neutral_not_ranked"}:
            parameter, positive_tail, negative_tail = (
                "positive_lfc_means",
                "count_positive_lfc",
                "count_negative_lfc",
            )
        else:
            raise ValueError(f"baseline uses an unregistered rank method: {method}")
        meaning = parameters.get(parameter)
        if meaning not in {"resistance", "sensitization"}:
            raise ValueError(
                f"baseline requires an explicit {parameter} direction mapping"
            )
        positive_mappings[parameter] = meaning
        mapping = {
            positive_tail: meaning,
            negative_tail: "sensitization" if meaning == "resistance" else "resistance",
        }
        if method == "neutral_not_ranked":
            mapping = {"count_neutral": "neutral"}
        expected = rows.analysis_tail.map(mapping)
        if expected.isna().any() or not expected.eq(rows.phenotype_direction).all():
            raise ValueError(
                "baseline phenotype direction contradicts declared native tail mapping"
            )
    if (
        ranked.groupby(list(QUERY_COLUMNS))
        .screen_signal_rank_source.nunique()
        .gt(1)
        .any()
    ):
        raise ValueError("baseline mixes rank methods within one ranking query")
    directional = ranked.loc[ranked.phenotype_direction.ne("neutral")].copy()
    directional["screen_signal_rank"] = _numeric_ranks(
        directional.screen_signal_rank, "baseline"
    )
    candidates = frame[[*JOIN_COLUMNS, "screen_signal_rank"]].copy()
    candidates["screen_signal_rank"] = _numeric_ranks(
        candidates.screen_signal_rank, "candidate"
    )
    matched = candidates.merge(
        directional[[*JOIN_COLUMNS, "screen_signal_rank", "screen_signal_rank_source"]],
        on=list(JOIN_COLUMNS),
        how="left",
        suffixes=("_candidate", "_report"),
        validate="one_to_one",
        indicator=True,
    )
    if not matched["_merge"].eq("both").all():
        raise ValueError(
            "prediction candidates were not found in the pinned baseline report"
        )
    if not matched.screen_signal_rank_candidate.eq(
        matched.screen_signal_rank_report
    ).all():
        raise ValueError(
            "candidate screen_signal_rank differs from the pinned baseline report"
        )
    if "screen_signal_rank_source" in frame:
        source_lookup = directional.set_index(
            list(JOIN_COLUMNS)
        ).screen_signal_rank_source
        expected_sources = source_lookup.reindex(
            pd.MultiIndex.from_frame(frame[list(JOIN_COLUMNS)])
        )
        if frame.screen_signal_rank_source.tolist() != expected_sources.tolist():
            raise ValueError(
                "candidate rank source differs from the pinned baseline report"
            )
    query_methods = (
        matched[[*QUERY_COLUMNS, "screen_signal_rank_source"]]
        .drop_duplicates()
        .sort_values(list(QUERY_COLUMNS))
        .to_dict(orient="records")
    )
    provenance = {
        "status": "verified_report_linkage",
        "baseline_score_column": "screen_signal_rank",
        "higher_is_better": False,
        "manifest_sha256": expected_manifest_sha256,
        "ranked_output_sha256": expected_output_hash,
        "ranked_output_filename": "ranked_candidates.tsv",
        "methods": sorted(matched.screen_signal_rank_source.unique().tolist()),
        "positive_mappings": positive_mappings,
        "query_methods": query_methods,
        "join_columns": list(JOIN_COLUMNS),
        "report_row_count": len(ranked),
        "directional_report_row_count": len(directional),
        "candidate_row_count": len(candidates),
        "candidate_is_report_subset": len(candidates) < len(directional),
        "rank_roster_scope": "supplied_candidates_with_native_report_ranks_preserved",
        "upstream_screen_analysis_recomputed": False,
        "biological_direction_is_upstream_declaration": True,
    }
    return provenance, {manifest_path: manifest_content, ranked_path: ranked_content}
