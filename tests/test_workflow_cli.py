import json
from pathlib import Path

import pandas as pd
import pytest

from crispr_evidencerank.cli import build_parser
from crispr_evidencerank.workflow_cli import _candidate_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("command", ["nested-benchmark", "select-deployment-model"])
def test_cli_model_selection_pins_grid_and_writes_scientific_boundaries(
    tmp_path, capsys, command
):
    output = tmp_path / "evaluation"
    argv = [
        command,
        "--table",
        str(ROOT / "examples/synthetic/labeled_gene_features.csv"),
        "--model-grid",
        str(ROOT / "examples/synthetic/model_grid.json"),
        "--inner-folds",
        "2",
        "--output-dir",
        str(output),
    ]
    if command == "nested-benchmark":
        argv += ["--outer-folds", "3"]
    args = build_parser().parse_args(argv)
    assert args.func(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["inputs"]["model_grid"]["sha256"]
    assert report["software_provenance"]["dependency_lock_sha256"]
    assert report["scientific_boundary"]["preregistration_not_created"]
    if command == "nested-benchmark":
        predictions = pd.read_csv(output / "nested_predictions.tsv", sep="\t")
        assert len(predictions) == 480
        assert predictions.fold.nunique() == 3
        assert predictions.reproducibility_score.between(0, 1).all()
    else:
        assert report["selected_candidate"]["candidate_id"] in {
            "logistic_c01",
            "logistic_c1",
        }
        assert report["metrics"] is None
        assert not (output / "nested_predictions.tsv").exists()
    with pytest.raises(FileExistsError):
        args.func(args)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"selection_metric":"a", "selection_metric":"b", "candidates":[]}',
        b'{"selection_metric":"a", "candidates":[], "choose_from_test":true}',
        b'{"selection_metric":"a", "candidates":[{"candidate_id":"x",'
        b'"feature_profile":"clinical","model_kind":"logistic"}]}',
    ],
)
def test_grid_rejects_ambiguous_or_unreviewed_specs(payload):
    with pytest.raises(ValueError):
        _candidate_config(payload)


def test_inference_cli_requires_manifest_anchor_and_explicit_trust_default():
    parser = build_parser()
    args = parser.parse_args(
        [
            "predict-frozen-model",
            "--table",
            "features.tsv",
            "--model-bundle-dir",
            "model",
            "--expected-model-manifest-sha256",
            "a" * 64,
            "--created-at-utc",
            "2026-09-08T12:00:00+00:00",
            "--code-revision",
            "b" * 40,
            "--output-dir",
            "prediction",
        ]
    )
    assert args.trusted_local_bundle is False
    assert args.allow_training_overlap is False


def test_freeze_cli_preserves_selected_regularization():
    args = build_parser().parse_args(
        [
            "freeze-model",
            "--table",
            "features.tsv",
            "--output-dir",
            "model",
            "--code-revision",
            "b" * 40,
            "--created-at-utc",
            "2026-09-08T12:00:00+00:00",
            "--logistic-c",
            "0.1",
        ]
    )
    assert args.logistic_c == 0.1
