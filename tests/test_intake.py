from io import StringIO

import pandas as pd
import pytest

from crispr_evidencerank.cli import build_parser
from crispr_evidencerank.contracts import (
    INTAKE_POLICY_V2_BENCHMARK_RULE_IDS,
    INTAKE_POLICY_V2_SCOPE_RULE_IDS,
    AssessmentStage,
)
from crispr_evidencerank.intake import (
    derive_intake_decision,
    triage_orcs_index,
)

MIXED_INDEX = (
    "#SCREEN ID\tSOURCE ID\tSOURCE TYPE\tSCREEN FORMAT\t"
    "EXPERIMENTAL SETUP\tCONDITION NAME\tLIBRARY TYPE\t"
    "LIBRARY METHODOLOGY\tCELL LINE\tORGANISM OFFICIAL\t"
    "FULL SIZE AVAILABLE\tDATASET ID\n"
    "1\t1001\tpubmed\tPool\tDrug Exposure\tOlaparib\t"
    "CRISPRn\tKnockout\tA375\tHomo sapiens\tYes\tD1\n"
    "2\t1002\tpubmed\tPool\tDrug Exposure\tOlaparib\t"
    "CRISPRa\tActivation\tA375\tHomo sapiens\tYes\tD2\n"
    "3\t1003\tpubmed\tPool\tToxin Exposure\tRicin\t"
    "CRISPRn\tKnockout\tA375\tHomo sapiens\tYes\tD3\n"
    "4\t1004\tpubmed\tArray\tDrug Exposure\tOlaparib\t"
    "CRISPRn\tKnockout\tA375\tHomo sapiens\tYes\tD4\n"
    "5\t1005\tpubmed\t-\t-\t-\t-\t-\t-\tHomo sapiens\tNo\tD5\n"
    "6\t1006\tpubmed\tPool\tDrug Exposure\tOlaparib\t"
    "CRISPRn\tKnockout\tA375\tMus musculus\tYes\tD6\n"
)


def test_orcs_index_triage_is_conservative_and_never_benchmark_ready():
    result = triage_orcs_index(
        StringIO(MIXED_INDEX),
        release="2.0.18",
        retrieved_date="2026-07-31",
    )

    by_external = result.screen_intake.set_index("external_screen_id")
    assert by_external.loc["1", "status"] == "metadata_only"
    assert by_external.loc["2", "status"] == "exclude"
    assert by_external.loc["3", "status"] == "metadata_only"
    assert by_external.loc["4", "status"] == "exclude"
    assert by_external.loc["5", "status"] == "metadata_only"
    assert by_external.loc["6", "status"] == "exclude"
    assert not result.screen_intake["benchmark_ready"].any()
    assert set(result.candidate_screen_ids) == {"1", "3", "5"}
    assert result.summary["status_counts"] == {
        "exclude": 3,
        "metadata_only": 3,
    }

    unknown_checks = result.eligibility_checks.loc[
        result.eligibility_checks["screen_id"].str.endswith(":5")
    ]
    assert (
        unknown_checks.loc[
            unknown_checks["rule_id"] == "scope.perturbation_crispr_ko",
            "outcome",
        ].item()
        == "unknown"
    )
    assert not (
        result.eligibility_checks["rule_id"]
        .astype(str)
        .str.contains("author_hit|validation_label")
        .any()
    )
    completeness = result.eligibility_checks.loc[
        result.eligibility_checks["rule_id"] == "metadata.full_gene_score_set"
    ].set_index("screen_id")
    assert completeness.loc["orcs:2.0.18:screen:1", "outcome"] == "pass"
    assert (
        completeness.loc["orcs:2.0.18:screen:5", "reason_code"]
        == "orcs_score_set_incomplete"
    )
    toxin_checks = result.eligibility_checks.loc[
        result.eligibility_checks["screen_id"].str.endswith(":3")
    ].set_index("rule_id")
    assert toxin_checks.loc["scope.drug_exposure", "outcome"] == "unknown"
    assert toxin_checks.loc["metadata.identifiable_drug", "outcome"] == "unknown"
    assert (
        toxin_checks.loc["metadata.identifiable_drug", "reason_code"]
        == "condition_name_requires_drug_exposure_curation"
    )
    assert pd.isna(toxin_checks.loc["metadata.identifiable_drug", "normalized_value"])
    source_checks = result.eligibility_checks.loc[
        result.eligibility_checks["rule_id"] == "provenance.source_family"
    ].set_index("screen_id")
    assert source_checks.loc["orcs:2.0.18:screen:1", "outcome"] == "unknown"
    assert (
        source_checks.loc["orcs:2.0.18:screen:1", "reason_code"]
        == "source_family_provisional_from_source_id"
    )


def test_declared_human_archive_scope_fills_missing_row_organism():
    source = StringIO(
        "#SCREEN ID\tSOURCE ID\tSOURCE TYPE\tSCREEN FORMAT\t"
        "EXPERIMENTAL SETUP\tCONDITION NAME\tLIBRARY TYPE\t"
        "LIBRARY METHODOLOGY\tCELL LINE\n"
        "7\t1007\tpubmed\tPool\tDrug Exposure\tDrug A\t"
        "CRISPRn\tKnockout\tA375\n"
    )
    result = triage_orcs_index(
        source,
        release="2.0.18",
        retrieved_date="2026-07-31",
        organism_scope="Homo sapiens",
    )
    organism_check = result.eligibility_checks.loc[
        result.eligibility_checks["rule_id"] == "scope.organism_human"
    ].iloc[0]
    assert organism_check["outcome"] == "pass"
    assert result.parsed.studies.iloc[0]["organism"] == "Homo sapiens"


def test_triage_cli_writes_all_audit_outputs(tmp_path):
    source = tmp_path / "orcs_index.tsv"
    source.write_text(MIXED_INDEX, encoding="utf-8")
    output_dir = tmp_path / "triage"
    args = build_parser().parse_args(
        [
            "triage-orcs-index",
            "--table",
            str(source),
            "--release",
            "2.0.18",
            "--retrieved-date",
            "2026-07-31",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert args.func(args) == 0
    assert (output_dir / "screen_intake.tsv").exists()
    assert (output_dir / "eligibility_checks.tsv").exists()
    assert (output_dir / "triage_summary.json").exists()
    candidates = (output_dir / "candidate_screen_ids.txt").read_text(encoding="utf-8")
    assert candidates.splitlines() == ["1", "3", "5"]
    intake = pd.read_csv(output_dir / "screen_intake.tsv", sep="\t")
    assert set(intake["status"]) == {"exclude", "metadata_only"}


def _curated_check(
    rule_id,
    outcome,
    *,
    required_for_scope=False,
    required_for_benchmark=True,
):
    return {
        "check_id": f"curated:check:{rule_id}",
        "intake_id": "curated:intake",
        "screen_id": "screen:1",
        "assessment_stage": "curated",
        "rule_id": rule_id,
        "outcome": outcome,
        "required_for_scope": required_for_scope,
        "required_for_benchmark": required_for_benchmark,
        "reason_code": f"{rule_id}:{outcome}",
        "source_locator": "curated evidence",
    }


def _passing_policy_v2_checks():
    return [
        _curated_check(
            rule_id,
            "pass",
            required_for_scope=(rule_id in INTAKE_POLICY_V2_SCOPE_RULE_IDS),
            required_for_benchmark=True,
        )
        for rule_id in sorted(INTAKE_POLICY_V2_BENCHMARK_RULE_IDS)
    ]


def test_curated_readiness_is_derived_only_when_all_requirements_pass():
    checks = _passing_policy_v2_checks()
    ready = derive_intake_decision(
        checks,
        assessment_stage=AssessmentStage.CURATED,
    )
    assert ready.status.value == "benchmark_ready"
    assert ready.benchmark_ready
    assert ready.reason_codes is None

    checks = [
        (
            _curated_check("data.count_level_signal", "unknown")
            if check["rule_id"] == "data.count_level_signal"
            else check
        )
        for check in checks
    ]
    unresolved = derive_intake_decision(
        checks,
        assessment_stage=AssessmentStage.CURATED,
    )
    assert unresolved.status.value == "metadata_only"
    assert not unresolved.benchmark_ready
    assert unresolved.reason_codes == "data.count_level_signal"


def test_curated_readiness_rejects_an_incomplete_rule_set():
    checks = [
        check
        for check in _passing_policy_v2_checks()
        if check["rule_id"] != "labels.adjudicated_validation_event"
    ]
    decision = derive_intake_decision(
        checks,
        assessment_stage=AssessmentStage.CURATED,
    )
    assert decision.status.value == "metadata_only"
    assert not decision.benchmark_ready
    assert "missing:labels.adjudicated_validation_event" in decision.reason_codes


def test_curated_reducer_excludes_explicit_scope_failure():
    decision = derive_intake_decision(
        [
            _curated_check(
                "scope.organism_human",
                "fail",
                required_for_scope=True,
            ),
            _curated_check("data.count_level_signal", "pass"),
        ],
        assessment_stage="curated",
    )
    assert decision.status.value == "exclude"
    assert not decision.candidate_for_full_curation
    assert not decision.benchmark_ready


def test_missing_source_id_stays_unresolved_in_intake():
    source = StringIO(
        "#SCREEN ID\tSOURCE ID\tSOURCE TYPE\tSCREEN FORMAT\t"
        "EXPERIMENTAL SETUP\tCONDITION NAME\tLIBRARY TYPE\t"
        "LIBRARY METHODOLOGY\tCELL LINE\tORGANISM OFFICIAL\n"
        "7\t-\tpubmed\tPool\tDrug Exposure\tDrug A\t"
        "CRISPRn\tKnockout\tA375\tHomo sapiens\n"
    )
    result = triage_orcs_index(
        source,
        release="2.0.18",
        retrieved_date="2026-07-31",
    )
    source_check = result.eligibility_checks.loc[
        result.eligibility_checks["rule_id"] == "provenance.source_family"
    ].iloc[0]
    assert source_check["outcome"] == "unknown"
    assert source_check["reason_code"] == "source_family_unresolved_missing_source_id"
    assert pd.isna(source_check["observed_value"])
    assert pd.isna(result.parsed.screens.iloc[0]["source_family_id"])


def test_only_supported_intake_policy_version_is_accepted():
    with pytest.raises(ValueError, match="supported versions: 2"):
        triage_orcs_index(
            StringIO(MIXED_INDEX),
            release="2.0.18",
            retrieved_date="2026-07-31",
            policy_version=3,
        )

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "triage-orcs-index",
                "--table",
                "unused.tsv",
                "--release",
                "2.0.18",
                "--retrieved-date",
                "2026-07-31",
                "--policy-version",
                "3",
                "--output-dir",
                "unused",
            ]
        )
