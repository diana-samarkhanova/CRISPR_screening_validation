"""Allocation must be paired, reproducible, outcome-free, and immutable."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import crispr_evidencerank.prospective as prospective
from crispr_evidencerank.prospective import (
    ALLOCATION_FILENAME,
    MANIFEST_FILENAME,
    build_prospective_selection,
    freeze_prospective_selection,
    verify_prospective_selection,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _roster() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene_symbol": list("ABCDEFGH") + ["KNOWN", "UNCERTAIN", "POS", "NEG"],
            "screen_id": "screen1",
            "study_id": "prospective_study",
            "source_family_id": "prospective_source",
            "raw_data_family_id": "prospective_raw",
            "contrast_id": "drug_vs_vehicle",
            "perturbation_modality": "CRISPR_KO",
            "phenotype_direction": "resistance",
            "reproducibility_score": [
                0.9,
                0.8,
                0.7,
                0.6,
                0.5,
                0.4,
                0.3,
                0.2,
                1,
                1,
                1,
                0,
            ],
            "screen_signal_rank": [1, 3, 2, 4, 5, 6, 7, 8, 1, 1, 1, 12],
            "prior_outcome_exposure": ["not_seen"] * 8
            + ["seen", "unknown", "seen", "seen"],
            "candidate_role": ["candidate"] * 10
            + ["positive_control", "negative_control"],
        }
    )


def _select(frame: pd.DataFrame | None = None, **kwargs):
    parameters = {"top_k": 2, "random_comparison_n": 2, "random_seed": 17}
    parameters.update(kwargs)
    return build_prospective_selection(
        _roster() if frame is None else frame, **parameters
    )


def test_union_disagreement_randomization_and_controls():
    result = _select()
    table = result.allocation.set_index("gene_symbol")
    assert table.at["A", "selection_stratum"] == "both_top_k"
    assert table.at["B", "selection_stratum"] == "model_only_top_k"
    assert table.at["C", "selection_stratum"] == "baseline_only_top_k"
    assert table.loc[list("ABC"), "selected_for_comparison"].all()
    assert table.loc[list("ABC"), "inclusion_probability"].eq(1).all()
    assert table.loc[list("DEFGH"), "inclusion_probability"].eq(2 / 5).all()
    assert table["random_comparison_selected"].sum() == 2
    for gene in ("KNOWN", "UNCERTAIN"):
        assert not table.at[gene, "eligible_for_comparison"]
        assert not table.at[gene, "selected_for_wet_lab"]
        assert pd.isna(table.at[gene, "inclusion_probability"])
    for gene in ("POS", "NEG"):
        assert table.at[gene, "selected_for_wet_lab"]
        assert not table.at[gene, "selected_for_comparison"]
        assert pd.isna(table.at[gene, "model_rank"])
    assert result.summary["counts"]["selected_for_comparison"] == 5
    assert result.summary["counts"]["selected_for_wet_lab"] == 7
    assert "label_code" not in table
    assert "testing_status" not in table


def test_row_order_and_ties_cannot_change_allocation():
    frame = _roster()
    frame.loc[:7, "reproducibility_score"] = 0.5
    frame.loc[:7, "screen_signal_rank"] = 1
    original = _select(frame)
    shuffled = _select(frame.sample(frac=1, random_state=55))
    assert_frame_equal(original.allocation, shuffled.allocation)
    assert_frame_equal(original.assay_roster, shuffled.assay_roster)
    assert original.summary == shuffled.summary
    assert original.allocation.loc[
        original.allocation.in_model_top_k, "gene_symbol"
    ].tolist() == ["A", "B"]


def test_assay_roster_hides_scores_exposure_and_allocation_strata():
    result = _select()
    assay = result.assay_roster
    assert len(assay) == 7
    assert assay["assay_id"].is_unique
    assert assay["assay_id"].str.startswith("ASSAY-").all()
    assert set(assay.columns) == {
        "assay_id",
        "gene_symbol",
        "screen_id",
        "contrast_id",
        "perturbation_modality",
        "phenotype_direction",
        "candidate_role",
    }
    assert "KNOWN" not in assay.gene_symbol.tolist()
    assert "UNCERTAIN" not in assay.gene_symbol.tolist()
    assert set(assay["assay_id"]) == set(
        result.allocation.loc[result.allocation.selected_for_wet_lab, "assay_id"]
    )


def test_seed_changes_remainder_but_not_top_lists():
    result1 = _select(random_seed=17).allocation
    result2 = _select(random_seed=18).allocation
    assert_frame_equal(
        result1[["in_model_top_k", "in_baseline_top_k"]],
        result2[["in_model_top_k", "in_baseline_top_k"]],
    )
    assert not result1.randomization_key.equals(result2.randomization_key)
    assert (
        result1.random_comparison_selected.sum()
        == result2.random_comparison_selected.sum()
        == 2
    )


def test_zero_random_has_explicit_zero_sampling_coverage():
    result = _select(random_comparison_n=0)
    assert not result.summary["sampling"][
        "all_eligible_have_positive_inclusion_probability"
    ]
    table = result.allocation
    assert (
        table.loc[table.selection_stratum.eq("remainder"), "inclusion_probability"]
        .eq(0)
        .all()
    )


def test_census_and_complete_union_are_supported():
    assert (
        _select(random_comparison_n=5).summary["counts"]["selected_for_comparison"] == 8
    )
    all_top = _select(top_k=8, random_comparison_n=0)
    assert all_top.summary["sampling"][
        "all_eligible_have_positive_inclusion_probability"
    ]
    assert all_top.summary["sampling"]["remainder_inclusion_probability"] is None


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"top_k": 9}, "exceeds"),
        ({"top_k": 0}, "integer"),
        ({"top_k": True}, "integer"),
        ({"random_comparison_n": 6}, "remaining"),
        ({"random_comparison_n": -1}, "integer"),
        ({"random_seed": -1}, "integer"),
        ({"random_seed": 2**64}, "less than"),
        ({"baseline_higher_is_better": True}, "direction"),
        ({"model_score_column": "label_code"}, "reproducibility_score"),
        ({"baseline_score_column": "validation_rate"}, "registered"),
    ],
)
def test_invalid_design_parameters_fail(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _select(**kwargs)


@pytest.mark.parametrize(
    "column",
    [
        "label_code",
        "testing_status",
        "validation_outcome",
        "gold_score",
        "success",
        "patient_response",
    ],
)
def test_outcome_fields_are_rejected_even_when_unselected(column):
    frame = _roster().assign(**{column: "U"})
    with pytest.raises(ValueError, match="outcomes are forbidden"):
        _select(frame)


@pytest.mark.parametrize(
    "column,value,match",
    [
        ("gene_symbol", "B", "unique"),
        ("gene_symbol", " A ", "whitespace"),
        ("gene_symbol", None, "missing"),
        ("gene_symbol", "A\nB", "control characters"),
        ("screen_id", "screen2", "one screen_id"),
        ("contrast_id", "contrast2", "one contrast_id"),
        ("perturbation_modality", "CRISPRa", "one perturbation_modality"),
        ("phenotype_direction", "sensitization", "one phenotype_direction"),
        ("prior_outcome_exposure", "no", "not_seen"),
        ("candidate_role", "negative", "candidate_role"),
        ("reproducibility_score", float("nan"), "finite"),
        ("reproducibility_score", float("inf"), "finite"),
        ("reproducibility_score", 1.1, "lie in"),
        ("screen_signal_rank", 0, "positive integers"),
        ("screen_signal_rank", 1.5, "positive integers"),
    ],
)
def test_invalid_roster_fails(column, value, match):
    frame = _roster()
    frame[column] = frame[column].astype(object)
    frame.at[0, column] = value
    with pytest.raises(ValueError, match=match):
        _select(frame)


def test_all_modalities_remain_separate():
    for modality in ("CRISPR_KO", "CRISPRa", "CRISPRi"):
        assert (
            _select(_roster().assign(perturbation_modality=modality)).summary["query"][
                "perturbation_modality"
            ]
            == modality
        )
    with pytest.raises(ValueError, match="CRISPR_KO"):
        _select(_roster().assign(perturbation_modality="KO"))


def test_registered_baselines_have_explicit_directions():
    frame = _roster().assign(drugz_direction_aligned_normz=-np.arange(12))
    result = _select(
        frame,
        baseline_score_column="drugz_direction_aligned_normz",
        baseline_higher_is_better=True,
    )
    assert result.summary["counts"]["top_k_overlap"] == 2
    with pytest.raises(ValueError, match="direction"):
        _select(frame, baseline_score_column="drugz_direction_aligned_normz")
    with pytest.raises(ValueError, match="lie in"):
        _select(frame.assign(mageck_fdr=2), baseline_score_column="mageck_fdr")


@pytest.fixture
def freeze_inputs(tmp_path):
    roster = tmp_path / "predictions.tsv"
    _roster().to_csv(roster, sep="\t", index=False)
    model = tmp_path / "upstream_model.json"
    model.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_type": "frozen_reproducibility_model",
                "status": "frozen_not_yet_prospectively_evaluated",
                "created_at_utc": "2026-09-06T00:00:00+00:00",
                "training_table": {"row_count": 1},
                "training_provenance": {
                    "identity_hash_algorithm": "sha256_utf8_role_nul_stripped_value_v1",
                    "identities": {
                        role: {
                            "source_column": role,
                            "status": "complete",
                            "missing_row_count": 0,
                            "identity_sha256": [
                                hashlib.sha256(
                                    f"{role}\0training_only".encode()
                                ).hexdigest()
                            ],
                        }
                        for role in (
                            "study_id",
                            "source_family_id",
                            "raw_data_family_id",
                        )
                    },
                },
            }
        )
    )
    prediction = tmp_path / "upstream_prediction.json"
    prediction.write_text(
        json.dumps(
            {
                "bundle_type": "frozen_model_predictions",
                "schema_version": 1,
                "created_at_utc": "2026-09-07T00:00:00+00:00",
                "inference_mode": "declared_families_disjoint",
                "training_overlap": {"complete_inventory": True, "any_overlap": False},
                "inputs": {"model_manifest": {"sha256": _hash(model)}},
                "outputs": {"predictions": {"sha256": _hash(roster)}},
                "score_semantics": {"primary": "reproducibility_score"},
                "baseline_linkage": {
                    "status": "verified_report_linkage",
                    "baseline_score_column": "screen_signal_rank",
                    "higher_is_better": False,
                },
            }
        )
    )
    sap = tmp_path / "SAP.md"
    sap.write_text("Template only; no approval implied.\n")
    return {
        "roster_path": roster,
        "output_dir": tmp_path / "allocation",
        "expected_roster_sha256": _hash(roster),
        "model_manifest_path": model,
        "expected_model_manifest_sha256": _hash(model),
        "prediction_manifest_path": prediction,
        "expected_prediction_manifest_sha256": _hash(prediction),
        "sap_path": sap,
        "expected_sap_sha256": _hash(sap),
        "code_revision": "a" * 40,
        "created_at_utc": datetime(2026, 9, 8, tzinfo=UTC),
        "top_k": 2,
        "random_comparison_n": 2,
        "random_seed": 17,
    }


def test_freeze_reproducible_and_pinned(freeze_inputs, tmp_path):
    manifest = freeze_prospective_selection(**freeze_inputs)
    output = freeze_inputs["output_dir"]
    assert manifest == verify_prospective_selection(
        output, expected_manifest_sha256=_hash(output / MANIFEST_FILENAME)
    )
    assert manifest["analysis_plan_approval_or_completeness_not_verified"]
    other = tmp_path / "another"
    freeze_prospective_selection(**{**freeze_inputs, "output_dir": other})
    for path in output.iterdir():
        assert path.read_bytes() == (other / path.name).read_bytes()


def test_existing_output_and_dangling_symlink_never_overwritten(
    freeze_inputs, tmp_path
):
    output = freeze_inputs["output_dir"]
    output.mkdir()
    marker = output / "marker"
    marker.write_text("retain")
    with pytest.raises(FileExistsError):
        freeze_prospective_selection(**freeze_inputs)
    assert marker.read_text() == "retain"
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        freeze_prospective_selection(**{**freeze_inputs, "output_dir": link})
    assert link.is_symlink()


def test_raced_empty_directory_is_not_replaced(freeze_inputs, monkeypatch):
    publish = prospective._publish_directory_noreplace

    def race(source, target):
        target.mkdir()
        publish(source, target)

    monkeypatch.setattr(prospective, "_publish_directory_noreplace", race)
    with pytest.raises(FileExistsError):
        freeze_prospective_selection(**freeze_inputs)
    assert list(freeze_inputs["output_dir"].iterdir()) == []
    assert not list(freeze_inputs["output_dir"].parent.glob(".allocation.staging-*"))


def test_changed_source_during_freeze_is_rejected(freeze_inputs, monkeypatch):
    build = prospective.build_prospective_selection

    def change_after_selection(*args, **kwargs):
        result = build(*args, **kwargs)
        freeze_inputs["sap_path"].write_text("Changed after selection\n")
        return result

    monkeypatch.setattr(
        prospective, "build_prospective_selection", change_after_selection
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        freeze_prospective_selection(**freeze_inputs)
    assert not freeze_inputs["output_dir"].exists()


@pytest.mark.parametrize("field", ["roster", "model", "prediction", "sap"])
def test_all_source_hashes_required(freeze_inputs, field):
    key = (
        f"expected_{field}_sha256"
        if field in {"roster", "sap"}
        else f"expected_{field}_manifest_sha256"
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        freeze_prospective_selection(**{**freeze_inputs, key: "0" * 64})


@pytest.mark.parametrize("part", ["roster", "model", "score"])
def test_prediction_linkage_cannot_be_substituted(freeze_inputs, part):
    path = freeze_inputs["prediction_manifest_path"]
    data = json.loads(path.read_text())
    if part == "roster":
        data["outputs"]["predictions"]["sha256"] = "0" * 64
    elif part == "model":
        data["inputs"]["model_manifest"]["sha256"] = "0" * 64
    else:
        data["score_semantics"]["primary"] = "validation_score"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="match|identities|primary score"):
        freeze_prospective_selection(
            **{**freeze_inputs, "expected_prediction_manifest_sha256": _hash(path)}
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "exploratory",
        "unknown_inventory",
        "overlap",
        "missing_baseline",
        "baseline_direction",
        "future_prediction",
        "schema_bool",
    ],
)
def test_unqualified_prediction_manifest_cannot_be_frozen(freeze_inputs, mutation):
    path = freeze_inputs["prediction_manifest_path"]
    data = json.loads(path.read_text())
    if mutation == "exploratory":
        data["inference_mode"] = "exploratory_overlap_allowed"
    elif mutation == "unknown_inventory":
        data["training_overlap"]["complete_inventory"] = False
    elif mutation == "overlap":
        data["training_overlap"]["any_overlap"] = True
    elif mutation == "missing_baseline":
        data.pop("baseline_linkage")
    elif mutation == "baseline_direction":
        data["baseline_linkage"]["higher_is_better"] = True
    elif mutation == "future_prediction":
        data["created_at_utc"] = "2027-01-01T00:00:00+00:00"
    elif mutation == "schema_bool":
        data["schema_version"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        freeze_prospective_selection(
            **{
                **freeze_inputs,
                "expected_prediction_manifest_sha256": _hash(path),
            }
        )
    assert not freeze_inputs["output_dir"].exists()


@pytest.mark.parametrize(
    "mutation", ["overlap", "bare_manifest", "future_model", "invalid_inventory"]
)
def test_model_identity_inventory_and_chronology_are_checked(freeze_inputs, mutation):
    path = freeze_inputs["model_manifest_path"]
    data = json.loads(path.read_text())
    if mutation == "overlap":
        data["training_provenance"]["identities"]["study_id"]["identity_sha256"] = [
            hashlib.sha256(b"study_id\0prospective_study").hexdigest()
        ]
    elif mutation == "bare_manifest":
        data = {"bundle_type": "frozen_reproducibility_model"}
    elif mutation == "future_model":
        data["created_at_utc"] = "2027-01-01T00:00:00+00:00"
    else:
        data["training_provenance"]["identities"]["study_id"]["status"] = "partial"
    path.write_text(json.dumps(data))
    prediction_path = freeze_inputs["prediction_manifest_path"]
    prediction = json.loads(prediction_path.read_text())
    prediction["inputs"]["model_manifest"]["sha256"] = _hash(path)
    prediction_path.write_text(json.dumps(prediction))
    with pytest.raises(ValueError):
        freeze_prospective_selection(
            **{
                **freeze_inputs,
                "expected_model_manifest_sha256": _hash(path),
                "expected_prediction_manifest_sha256": _hash(prediction_path),
            }
        )


def test_tampered_status_cannot_become_a_validated_result(freeze_inputs):
    freeze_prospective_selection(**freeze_inputs)
    path = freeze_inputs["output_dir"] / MANIFEST_FILENAME
    manifest = json.loads(path.read_text())
    manifest["status"] = "prospectively_validated"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="status"):
        verify_prospective_selection(freeze_inputs["output_dir"])


def test_modified_allocation_detected_even_with_updated_internal_hash(freeze_inputs):
    freeze_prospective_selection(**freeze_inputs)
    output = freeze_inputs["output_dir"]
    anchor = _hash(output / MANIFEST_FILENAME)
    allocation = output / ALLOCATION_FILENAME
    data = pd.read_csv(allocation, sep="\t")
    data.loc[0, "selected_for_comparison"] = False
    data.to_csv(allocation, sep="\t", index=False)
    manifest_path = output / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["allocation"].update(
        sha256=_hash(allocation), byte_size=allocation.stat().st_size
    )
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity hash mismatch"):
        verify_prospective_selection(output, expected_manifest_sha256=anchor)
    with pytest.raises(ValueError, match="does not reproduce"):
        verify_prospective_selection(output)


def test_extra_files_path_traversal_and_symlinks_rejected(freeze_inputs, tmp_path):
    freeze_prospective_selection(**freeze_inputs)
    output = freeze_inputs["output_dir"]
    extra = output / "outcomes.tsv"
    extra.write_text("unapproved")
    with pytest.raises(ValueError, match="unexpected"):
        verify_prospective_selection(output)
    extra.unlink()
    target = output / ALLOCATION_FILENAME
    outside = tmp_path / "outside.tsv"
    target.rename(outside)
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="non-symlink"):
        verify_prospective_selection(output)
    target.unlink()
    outside.rename(target)
    path = output / MANIFEST_FILENAME
    manifest = json.loads(path.read_text())
    manifest["files"]["allocation"]["filename"] = "../outside.tsv"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe"):
        verify_prospective_selection(output)


def test_source_snapshot_preserves_gene_named_na(freeze_inputs):
    path = freeze_inputs["roster_path"]
    frame = _roster()
    frame.at[0, "gene_symbol"] = "NA"
    frame.to_csv(path, sep="\t", index=False)
    prediction_path = freeze_inputs["prediction_manifest_path"]
    prediction = json.loads(prediction_path.read_text())
    prediction["outputs"]["predictions"]["sha256"] = _hash(path)
    prediction_path.write_text(json.dumps(prediction))
    freeze_prospective_selection(
        **{
            **freeze_inputs,
            "expected_roster_sha256": _hash(path),
            "expected_prediction_manifest_sha256": _hash(prediction_path),
        }
    )
    allocation = pd.read_csv(
        freeze_inputs["output_dir"] / ALLOCATION_FILENAME,
        sep="\t",
        keep_default_na=False,
    )
    assert "NA" in allocation.gene_symbol.tolist()
