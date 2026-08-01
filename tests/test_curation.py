from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from crispr_evidencerank.curation import (
    select_curation_batch,
    write_curation_batch,
)


def _queue() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "queue_id": f"Q{rank}",
                "queue_rank": rank,
                "source_round": 1,
                "screen_id": f"S{rank}",
                "external_screen_id": str(rank),
                "source_family_id": f"F{rank}",
                "source_version": "2.0.18",
                "policy_version": 2,
                "bucket": "confirmed_scope",
                "scope_unknown_count": 0,
                "metadata_completeness_count": 8,
                "full_gene_score_set_available": True,
            }
            for rank in range(1, 5)
        ]
    )


def test_curation_batch_selection_is_input_order_invariant():
    queue = _queue()
    expected = select_curation_batch(queue, start_rank=1, batch_size=3)
    observed = select_curation_batch(
        queue.sample(frac=1, random_state=17),
        start_rank=1,
        batch_size=3,
    )
    pd.testing.assert_frame_equal(expected, observed)


def test_curation_batch_rejects_outcome_columns_and_repeated_families():
    with pytest.raises(ValueError, match="outcome-bearing"):
        select_curation_batch(
            _queue().assign(label_code="V2"),
            start_rank=1,
            batch_size=3,
        )

    repeated = _queue()
    repeated.loc[1, "source_family_id"] = "F1"
    with pytest.raises(ValueError, match="distinct source families"):
        select_curation_batch(repeated, start_rank=1, batch_size=3)

    with pytest.raises(ValueError, match="failed contract validation"):
        select_curation_batch(
            _queue().assign(future_validation="V3"),
            start_rank=1,
            batch_size=3,
        )

    fractional = _queue()
    fractional["queue_rank"] = [1.5, 2.5, 3.0, 4.0]
    with pytest.raises(ValueError, match="failed contract validation"):
        select_curation_batch(fractional, start_rank=1, batch_size=3)


def test_curation_batch_manifest_checksums_frozen_selection(tmp_path):
    queue_path = tmp_path / "queue.tsv"
    _queue().to_csv(queue_path, sep="\t", index=False, lineterminator="\n")
    output_dir = tmp_path / "batch"

    manifest = write_curation_batch(
        queue_path,
        output_dir,
        batch_id="orcs-2.0.18-batch-001",
        selected_date=date(2026, 8, 1),
        start_rank=1,
        batch_size=3,
    )

    committed = json.loads(
        (output_dir / "selection_manifest.json").read_text(encoding="utf-8")
    )
    assert committed == manifest
    assert committed["record_count"] == 3
    assert pd.read_csv(output_dir / "selection.tsv", sep="\t")[
        "queue_rank"
    ].tolist() == [1, 2, 3]


def test_curation_batch_rejects_invalid_queue_without_partial_output(tmp_path):
    queue_path = tmp_path / "queue.tsv"
    _queue().assign(unexpected_outcome="V3").to_csv(
        queue_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    output_dir = tmp_path / "batch"

    with pytest.raises(ValueError, match="failed contract validation"):
        write_curation_batch(
            queue_path,
            output_dir,
            batch_id="orcs-2.0.18-batch-001",
            selected_date=date(2026, 8, 1),
            start_rank=1,
            batch_size=3,
        )

    assert not output_dir.exists()


def test_curation_batch_refuses_to_modify_a_frozen_output(tmp_path):
    queue_path = tmp_path / "queue.tsv"
    _queue().to_csv(queue_path, sep="\t", index=False, lineterminator="\n")
    output_dir = tmp_path / "batch"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        write_curation_batch(
            queue_path,
            output_dir,
            batch_id="orcs-2.0.18-batch-001",
            selected_date=date(2026, 8, 1),
            start_rank=1,
            batch_size=3,
        )
