import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from crispr_evidencerank.model_artifact import MANIFEST_FILENAME, freeze_model_bundle
from crispr_evidencerank.modeling import DEFAULT_FEATURE_COLUMNS
from crispr_evidencerank.prediction import predict_frozen_model

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 8, tzinfo=UTC)


@pytest.fixture
def frozen(tmp_path):
    table = ROOT / "examples/synthetic/labeled_gene_features.csv"
    frame = pd.read_csv(table)
    bundle = tmp_path / "model"
    freeze_model_bundle(
        frame,
        bundle,
        feature_columns=list(DEFAULT_FEATURE_COLUMNS),
        training_table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
        training_table_name=table.name,
        training_table_path=table,
        feature_registry_path=ROOT / "config/features.yaml",
        environment_lock_path=ROOT / "uv.lock",
        code_revision="a" * 40,
        created_at_utc=NOW,
    )
    columns = ["gene_symbol", *DEFAULT_FEATURE_COLUMNS]
    # Direction-aligned features are constructed from source-native inputs.
    columns = [c for c in columns if c in frame]
    frame = frame.loc[:5, columns].copy()
    frame["study_id"] = "001"
    frame["screen_id"] = "001"
    frame["contrast_id"] = "001"
    frame["source_family_id"] = "future-family"
    frame["raw_data_family_id"] = "future-raw"
    frame["perturbation_modality"] = "CRISPR_KO"
    frame["phenotype_direction"] = "resistance"
    frame["median_guide_lfc"] = [1, 2, 3, 4, 5, 6]
    frame["mean_guide_lfc"] = frame.median_guide_lfc
    frame["signal_direction"] = "resistance"
    frame["positive_lfc_means"] = "resistance"
    frame["candidate_role"] = "candidate"
    frame["prior_outcome_exposure"] = "not_seen"
    frame["screen_signal_rank"] = [1, 2, 3, 4, 5, 6]
    input_path = tmp_path / "candidates.tsv"
    frame.to_csv(input_path, sep="\t", index=False)
    options = dict(
        candidate_table_path=input_path,
        model_bundle_dir=bundle,
        output_dir=tmp_path / "predictions",
        created_at_utc=NOW,
        expected_model_manifest_sha256=hashlib.sha256(
            (bundle / MANIFEST_FILENAME).read_bytes()
        ).hexdigest(),
        trusted_local_bundle=True,
        code_revision="b" * 40,
    )
    return frame, options


def _replace(frame, options):
    frame.to_csv(options["candidate_table_path"], sep="\t", index=False)


def test_frozen_prediction_preserves_ids_and_links_exact_output_bytes(frozen):
    _, options = frozen
    manifest = predict_frozen_model(**options)
    output = options["output_dir"]
    predictions = pd.read_csv(
        output / "predictions.tsv", sep="\t", dtype={"screen_id": str}
    )
    assert predictions.screen_id.eq("001").all()
    assert predictions.reproducibility_score.between(0, 1).all()
    assert (
        manifest["outputs"]["predictions"]["sha256"]
        == hashlib.sha256((output / "predictions.tsv").read_bytes()).hexdigest()
    )
    assert manifest["score_semantics"]["calibrated_probability"] is False
    assert not manifest["training_overlap"]["any_overlap"]
    assert json.loads((output / "prediction_manifest.json").read_text()) == manifest
    with pytest.raises(FileExistsError):
        predict_frozen_model(**options)


@pytest.mark.parametrize(
    "column",
    ["label_code", "target", "testing_status", "clinical_phase", "reviewer_label"],
)
def test_prediction_rejects_outcome_or_unreviewed_input(frozen, column):
    frame, options = frozen
    frame[column] = "V3"
    _replace(frame, options)
    with pytest.raises(ValueError, match="unreviewed or outcome"):
        predict_frozen_model(**options)
    assert not options["output_dir"].exists()


@pytest.mark.parametrize("modality", ["CRISPRa", "CRISPRi", "RNAi", ""])
def test_prediction_cannot_apply_ko_model_to_activation(frozen, modality):
    frame, options = frozen
    frame["perturbation_modality"] = modality
    _replace(frame, options)
    with pytest.raises(ValueError):
        predict_frozen_model(**options)


def test_training_source_overlap_requires_explicit_exploratory_mode(frozen):
    frame, options = frozen
    training = pd.read_csv(ROOT / "examples/synthetic/labeled_gene_features.csv")
    frame["source_family_id"] = training.source_family_id.iloc[0]
    _replace(frame, options)
    with pytest.raises(ValueError, match="training overlap"):
        predict_frozen_model(**options)
    manifest = predict_frozen_model(**options, allow_training_overlap=True)
    assert manifest["training_overlap"]["any_overlap"]
    assert manifest["inference_mode"] == "exploratory_overlap_allowed"


def test_prediction_refuses_untrusted_pickle_before_loading(frozen):
    _, options = frozen
    options["trusted_local_bundle"] = False
    with pytest.raises(ValueError, match="can execute code"):
        predict_frozen_model(**options)


@pytest.mark.parametrize("bad", [float("inf"), "not_a_number"])
def test_prediction_rejects_corrupted_numeric_values(frozen, bad):
    frame, options = frozen
    frame["guide_n"] = bad
    _replace(frame, options)
    with pytest.raises(ValueError):
        predict_frozen_model(**options)


def test_duplicate_id_or_header_is_rejected(frozen):
    frame, options = frozen
    duplicate = pd.concat([frame, frame.iloc[:1]])
    _replace(duplicate, options)
    with pytest.raises(ValueError, match="duplicate"):
        predict_frozen_model(**options)
    path = options["candidate_table_path"]
    path.write_text("study_id,study_id\n001,001\n")
    options["candidate_table_path"] = path.with_suffix(".csv")
    path.rename(options["candidate_table_path"])
    with pytest.raises(ValueError, match="duplicate"):
        predict_frozen_model(**options)


def test_prediction_does_not_publish_changed_input(frozen, monkeypatch):
    frame, options = frozen
    from crispr_evidencerank.modeling import ReproducibilityModel

    original = ReproducibilityModel.predict

    def tamper(self, data):
        frame["gene_symbol"] = [f"changed_{i}" for i in range(len(frame))]
        _replace(frame, options)
        return original(self, data)

    monkeypatch.setattr(ReproducibilityModel, "predict", tamper)
    with pytest.raises(ValueError, match="input changed"):
        predict_frozen_model(**options)
    assert not options["output_dir"].exists()


def test_raw_lfc_requires_declared_orientation(frozen):
    frame, options = frozen
    _replace(frame.drop(columns="positive_lfc_means"), options)
    with pytest.raises(ValueError, match="positive_lfc_means"):
        predict_frozen_model(**options)


def test_only_features_absent_in_training_cannot_generate_targets(frozen, tmp_path):
    frame, options = frozen
    table = tmp_path / "training.tsv"
    training = pd.read_csv(ROOT / "examples/synthetic/labeled_gene_features.csv")
    training["guide_n"] = float("nan")
    training.to_csv(table, sep="\t", index=False)
    model_dir = tmp_path / "unsupported-model"
    freeze_model_bundle(
        pd.read_csv(table, sep="\t"),
        model_dir,
        feature_columns=["guide_n", "guide_lfc_mad"],
        training_table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
        training_table_name=table.name,
        training_table_path=table,
        feature_registry_path=ROOT / "config/features.yaml",
        environment_lock_path=ROOT / "uv.lock",
        code_revision="a" * 40,
        created_at_utc=NOW,
    )
    frame["guide_lfc_mad"] = float("nan")
    _replace(frame, options)
    options["model_bundle_dir"] = model_dir
    options["expected_model_manifest_sha256"] = hashlib.sha256(
        (model_dir / MANIFEST_FILENAME).read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="no observed feature supported in training"):
        predict_frozen_model(**options)
