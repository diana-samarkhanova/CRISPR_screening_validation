from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from crispr_evidencerank.contracts import DataAssetRecord


def base_data_asset() -> dict[str, object]:
    return {
        "asset_id": "A1",
        "source_name": "SOURCE",
        "source_version": "v1",
        "asset_role": "count_table",
        "source_url": "https://example.org/counts.tsv",
        "available_date": "2026-01-01",
        "retrieved_date": "2026-01-02",
    }


@pytest.mark.parametrize(
    ("updates", "expected_valid"),
    [
        ({}, True),
        ({"sha256": "a" * 64}, False),
        ({"byte_size": 100}, False),
        ({"sha256": "a" * 64, "byte_size": 100}, True),
        ({"redistribution_raw": True}, False),
        ({"redistribution_derived": True, "license_spdx": "MIT"}, True),
        ({"retrieved_at_utc": "2026-01-02T10:00:00Z"}, True),
        ({"retrieved_at_utc": "2026-01-02T10:00:00+00:00"}, True),
        ({"retrieved_at_utc": "2026-01-02T10:00:00+06:00"}, False),
    ],
)
def test_data_asset_json_schema_matches_runtime_invariants(
    updates: dict[str, object],
    expected_valid: bool,
):
    record = base_data_asset() | updates
    schema_valid = Draft202012Validator(DataAssetRecord.model_json_schema()).is_valid(
        record
    )

    try:
        DataAssetRecord.model_validate(record)
    except ValidationError:
        runtime_valid = False
    else:
        runtime_valid = True

    assert schema_valid is expected_valid
    assert runtime_valid is expected_valid


def test_data_asset_schema_declares_cross_field_date_rule():
    semantic_rules = DataAssetRecord.model_json_schema()["x-semantic-rules"]
    assert any("cannot compare fields" in rule for rule in semantic_rules)
