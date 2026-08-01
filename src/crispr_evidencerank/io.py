"""Input adapters for count tables and common gene-summary formats."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COUNT_ID_COLUMNS = ("sgrna_id", "gene_symbol")


def read_table(
    path: str | Path,
    *,
    dtype: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Read CSV or TSV based on the file suffix."""

    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=dtype)
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", dtype=dtype)
    raise ValueError(f"unsupported table extension: {path.suffix}")


def normalize_count_inputs(
    counts: pd.DataFrame,
    samples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return copies with stable string column/sample identifiers."""

    normalized_counts = counts.copy()
    normalized_samples = samples.copy()
    normalized_column_names = [str(column) for column in normalized_counts.columns]
    if len(set(normalized_column_names)) != len(normalized_column_names):
        raise ValueError(
            "count-table column names are duplicated after string normalization"
        )
    normalized_counts.columns = normalized_column_names
    if "sample_id" in normalized_samples:
        if normalized_samples["sample_id"].isna().any():
            raise ValueError("sample_id cannot be missing")
        normalized_samples["sample_id"] = (
            normalized_samples["sample_id"].astype(str).str.strip()
        )
    return normalized_counts, normalized_samples


def validate_count_table(
    counts: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    allowed_annotation_columns: tuple[str, ...] = (),
) -> tuple[list[str], list[str]]:
    """Validate a wide count matrix against the sample sheet."""

    counts, samples = normalize_count_inputs(counts, samples)
    missing_ids = [column for column in COUNT_ID_COLUMNS if column not in counts]
    if missing_ids:
        raise ValueError(f"count table is missing columns: {missing_ids}")
    if "sample_id" not in samples:
        raise ValueError("sample sheet is missing sample_id")

    sample_ids = samples["sample_id"].tolist()
    if any(not sample for sample in sample_ids):
        raise ValueError("sample_id cannot be empty")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id must be unique in the sample sheet")
    missing_samples = [sample for sample in sample_ids if sample not in counts]
    extra_columns = [
        column
        for column in counts.columns
        if column not in COUNT_ID_COLUMNS
        and column not in sample_ids
        and column not in allowed_annotation_columns
    ]
    if missing_samples:
        raise ValueError(f"samples absent from count table: {missing_samples[:10]}")
    if extra_columns:
        raise ValueError(
            "count table contains columns not declared in the sample sheet "
            f"or annotation whitelist: {extra_columns[:10]}"
        )

    numeric = counts[sample_ids].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = np.argwhere(numeric.isna().to_numpy())[0]
        raise ValueError(
            "non-numeric count at "
            f"row={int(bad[0]) + 2}, sample={sample_ids[int(bad[1])]}"
        )
    if (numeric < 0).any().any():
        raise ValueError("counts must be non-negative")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("counts must be finite")
    if not np.allclose(
        numeric.to_numpy(dtype=float),
        np.rint(numeric.to_numpy(dtype=float)),
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError("raw count input must contain integer-valued read counts")
    for identifier in COUNT_ID_COLUMNS:
        if counts[identifier].isna().any():
            raise ValueError(f"{identifier} cannot be missing")
        if counts[identifier].astype(str).str.strip().eq("").any():
            raise ValueError(f"{identifier} cannot be empty")
    if counts["sgrna_id"].duplicated().any():
        raise ValueError("sgrna_id must be unique in a count table")
    return sample_ids, extra_columns


def normalize_mageck_gene_summary(
    frame: pd.DataFrame, screen_id: str, contrast_id: str
) -> pd.DataFrame:
    """Convert a standard MAGeCK RRA gene summary to long-form scores.

    MAGeCK's positive and negative directions are retained as separate rows.
    They are not automatically translated into biological resistance or
    sensitization because that mapping depends on the experimental contrast.
    """

    gene_column = next(
        (column for column in ("id", "gene", "Gene", "gene_symbol") if column in frame),
        None,
    )
    if gene_column is None:
        raise ValueError("could not identify the gene column in MAGeCK summary")

    aliases = {
        "score": ("score",),
        "effect": ("lfc",),
        "p_value": ("p-value", "pvalue", "p_value"),
        "fdr": ("fdr",),
        "rank": ("rank",),
    }
    records: list[dict[str, object]] = []
    for mageck_direction in ("pos", "neg"):
        found_any = False
        selected: dict[str, str] = {}
        for target, suffixes in aliases.items():
            for suffix in suffixes:
                candidates = (
                    f"{mageck_direction}|{suffix}",
                    f"{mageck_direction}_{suffix}",
                    f"{mageck_direction}.{suffix}",
                )
                column = next((name for name in candidates if name in frame), None)
                if column:
                    selected[target] = column
                    found_any = True
                    break
        if not found_any:
            continue
        for _, row in frame.iterrows():
            record: dict[str, object] = {
                "screen_id": screen_id,
                "contrast_id": contrast_id,
                "gene_symbol": str(row[gene_column]),
                "method": "MAGeCK-RRA",
                "analysis_tail": f"mageck_{mageck_direction}",
                "direction": "unknown",
                "cnv_corrected": False,
            }
            for target, column in selected.items():
                value = row[column]
                record[target] = None if pd.isna(value) else float(value)
            records.append(record)
    if not records:
        raise ValueError("no MAGeCK positive/negative score columns were found")
    return pd.DataFrame.from_records(records)
