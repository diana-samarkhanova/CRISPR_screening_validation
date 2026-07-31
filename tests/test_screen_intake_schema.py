import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from crispr_evidencerank.contracts import ScreenIntakeRecord


def base_screen_intake() -> dict[str, object]:
    return {
        "intake_id": "I1",
        "screen_id": "SC1",
        "source_name": "SOURCE",
        "source_version": "v1",
        "policy_version": 2,
        "assessment_stage": "curated",
        "status": "metadata_only",
        "candidate_for_full_curation": True,
        "benchmark_ready": False,
        "assessed_date": "2026-07-31",
    }


@pytest.mark.parametrize(
    ("updates", "expected_valid"),
    [
        ({}, True),
        ({"assessment_stage": "index"}, True),
        (
            {
                "assessment_stage": "index",
                "status": "exclude",
                "candidate_for_full_curation": False,
            },
            True,
        ),
        (
            {
                "status": "benchmark_ready",
                "benchmark_ready": True,
            },
            True,
        ),
        (
            {
                "status": "exclude",
                "candidate_for_full_curation": False,
            },
            True,
        ),
        (
            {
                "assessment_stage": "index",
                "status": "benchmark_ready",
                "benchmark_ready": True,
            },
            False,
        ),
        ({"benchmark_ready": True}, False),
        (
            {
                "status": "benchmark_ready",
                "benchmark_ready": False,
            },
            False,
        ),
        (
            {
                "status": "exclude",
                "candidate_for_full_curation": True,
            },
            False,
        ),
    ],
)
def test_screen_intake_json_schema_matches_runtime_invariants(
    updates: dict[str, object],
    expected_valid: bool,
):
    record = base_screen_intake() | updates
    schema_valid = Draft202012Validator(
        ScreenIntakeRecord.model_json_schema()
    ).is_valid(record)

    try:
        ScreenIntakeRecord.model_validate(record)
    except ValidationError:
        runtime_valid = False
    else:
        runtime_valid = True

    assert schema_valid is expected_valid
    assert runtime_valid is expected_valid
