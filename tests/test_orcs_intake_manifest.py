from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests" / "orcs_2.0.18"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checked_in_orcs_queue_matches_its_portable_manifests():
    derived = json.loads(
        (MANIFEST_DIR / "derived_manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (MANIFEST_DIR / "triage_summary.json").read_text(encoding="utf-8")
    )
    queue_path = MANIFEST_DIR / derived["curation_queue"]["filename"]
    candidate_path = MANIFEST_DIR / derived["candidate_screen_ids"]["filename"]
    queue = pd.read_csv(queue_path, sep="\t", dtype=str)
    candidate_ids = candidate_path.read_text(encoding="utf-8").splitlines()

    assert _sha256(queue_path) == derived["curation_queue"]["sha256"]
    assert _sha256(candidate_path) == derived["candidate_screen_ids"]["sha256"]
    assert len(queue) == derived["curation_queue"]["record_count"] == 1674
    assert len(candidate_ids) == derived["candidate_screen_ids"]["record_count"]
    assert candidate_ids == queue["screen_id"].tolist()
    assert queue["queue_rank"].astype(int).tolist() == list(range(1, 1675))
    assert queue["policy_version"].astype(int).eq(2).all()
    assert queue["source_version"].eq("2.0.18").all()
    assert queue["bucket"].value_counts().to_dict() == {
        "manual_scope_review": 1239,
        "confirmed_scope": 435,
    }
    assert summary["benchmark_ready_count"] == 0


def test_checked_in_orcs_queue_is_source_diverse_and_outcome_blind():
    queue = pd.read_csv(
        MANIFEST_DIR / "curation_queue.tsv",
        sep="\t",
        dtype=str,
    )
    first_batch = queue.head(10)

    assert first_batch["bucket"].eq("confirmed_scope").all()
    assert first_batch["source_round"].astype(int).eq(1).all()
    assert first_batch["source_family_id"].nunique() == 10
    assert {
        "author_hit",
        "number_of_hits",
        "score",
        "label_code",
        "testing_status",
        "validation_outcome",
    }.isdisjoint(queue.columns)
