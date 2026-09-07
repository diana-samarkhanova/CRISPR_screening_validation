import hashlib
import json
from pathlib import Path

import pandas as pd

from crispr_evidencerank.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_cli_writes_strict_report_and_clean_prediction_columns(
    tmp_path, capsys
):
    predictions_path = tmp_path / "predictions.tsv"
    metrics_path = tmp_path / "metrics.json"
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "--table",
            str(ROOT / "examples/synthetic/labeled_gene_features.csv"),
            "--output",
            str(predictions_path),
            "--metrics-output",
            str(metrics_path),
            "--folds",
            "4",
            "--seed-sweep",
            "17,29",
            "--random-baseline-seed-count",
            "3",
            "--bootstrap-repetitions",
            "5",
            "--bootstrap-min-effective-fraction",
            "0.2",
        ]
    )
    assert args.func(args) == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert printed == saved
    assert saved["cluster_bootstrap_intervals"] is not None
    assert saved["baselines"]["absolute_effect"]["status"] == "evaluated"
    assert saved["baselines"]["mageck_rank"]["status"].startswith("unavailable")
    assert saved["seed_sensitivity"]
    assert saved["within_query_feature_invariance"]["query_count"] == 16
    assert (
        saved["inputs"]["training_table"]["sha256"]
        == hashlib.sha256(
            (ROOT / "examples/synthetic/labeled_gene_features.csv").read_bytes()
        ).hexdigest()
    )
    assert (
        saved["outputs"]["predictions"]["sha256"]
        == hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    )
    assert saved["software_provenance"]["dependency_lock_sha256"]
    assert any(
        key.startswith("direction_resistance_") for key in saved["point_estimates"]
    )

    predictions = pd.read_csv(predictions_path, sep="\t")
    assert "reproducibility_score" in predictions
    assert "selection_propensity" in predictions
    assert "reproducibility_score_unweighted" not in predictions
    assert "selection_propensity_known_status" not in predictions
    assert "_ranking_query_id" not in predictions
    ranking_objects = predictions.dtypes.astype(str).eq(
        "object"
    ) & predictions.columns.str.startswith("_ranking")
    assert not any(ranking_objects)


def test_featurize_cli_requires_explicit_positive_lfc_semantics():
    parser = build_parser()
    action = next(
        action
        for action in parser._subparsers._group_actions[0]
        .choices["featurize-counts"]
        ._actions
        if action.dest == "positive_lfc_means"
    )
    assert action.required is True
    assert set(action.choices) == {"resistance", "sensitization"}


def test_benchmark_cli_rejects_output_collisions_before_reading(tmp_path):
    table_path = tmp_path / "input.tsv"
    table_path.write_text("broken\n", encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "--table",
            str(table_path),
            "--output",
            str(table_path),
            "--metrics-output",
            str(tmp_path / "metrics.json"),
        ]
    )
    try:
        args.func(args)
    except ValueError as exc:
        assert "cannot overwrite --table" in str(exc)
    else:
        raise AssertionError("benchmark accepted an input/output collision")
    assert table_path.read_text(encoding="utf-8") == "broken\n"


def test_benchmark_cli_rejects_shared_prediction_and_metrics_path(tmp_path):
    shared_path = tmp_path / "shared.json"
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "--table",
            str(ROOT / "examples/synthetic/labeled_gene_features.csv"),
            "--output",
            str(shared_path),
            "--metrics-output",
            str(shared_path),
        ]
    )
    try:
        args.func(args)
    except ValueError as exc:
        assert "must be different" in str(exc)
    else:
        raise AssertionError("benchmark accepted colliding outputs")
    assert not shared_path.exists()


def test_seed_metric_option_replaces_default_instead_of_appending():
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "--table",
            "input.csv",
            "--output",
            "predictions.tsv",
            "--seed-metric",
            "average_precision",
        ]
    )
    assert args.seed_metric == ["average_precision"]


def test_histogram_boosting_reports_paired_logistic_comparison(tmp_path, capsys):
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "--table",
            str(ROOT / "examples/synthetic/labeled_gene_features.csv"),
            "--output",
            str(tmp_path / "predictions.tsv"),
            "--folds",
            "4",
            "--model-kind",
            "hist_gradient_boosting",
            "--seed-sweep",
            "17",
            "--random-baseline-seed-count",
            "1",
            "--bootstrap-repetitions",
            "3",
            "--bootstrap-min-effective-fraction",
            "0.3",
        ]
    )
    assert args.func(args) == 0
    report = json.loads(capsys.readouterr().out)
    comparison = report["baselines"]["regularized_logistic"]
    assert comparison["status"] == "evaluated"
    assert comparison["paired_candidate_difference"]["effective_draws"] == 3
