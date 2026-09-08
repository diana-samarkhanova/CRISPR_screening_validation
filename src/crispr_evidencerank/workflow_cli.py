"""CLI integration for nested evaluation and prospective prediction bundles."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd

from .clinicaltrials_gov import _publish_directory_noreplace
from .modeling import FEATURE_PROFILES
from .nested_model_selection import (
    ModelCandidate,
    nested_grouped_oof_predictions,
    select_final_candidate,
)
from .prediction import predict_frozen_model
from .prospective import freeze_prospective_selection, verify_prospective_selection


def _finite_json(value):
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _print(value):
    print(json.dumps(_finite_json(value), sort_keys=True, indent=2, allow_nan=False))
    return 0


def _candidate_config(content: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate model-grid key: {key}")
            result[key] = value
        return result

    config = json.loads(content, object_pairs_hook=pairs)
    if not isinstance(config, dict) or set(config) != {
        "selection_metric",
        "candidates",
    }:
        raise ValueError("model grid requires exactly selection_metric and candidates")
    if not isinstance(config["candidates"], list) or not config["candidates"]:
        raise ValueError("model grid requires nonempty candidates")
    candidates = []
    for row in config["candidates"]:
        if not isinstance(row, dict) or set(row) - {
            "candidate_id",
            "model_kind",
            "logistic_c",
            "feature_profile",
        }:
            raise ValueError("model grid contains unrecognized candidate fields")
        if not {"candidate_id", "feature_profile", "model_kind"} <= set(row):
            raise ValueError(
                "candidate requires candidate_id, feature_profile and model_kind"
            )
        if row["feature_profile"] not in FEATURE_PROFILES:
            raise ValueError("model grid contains unreviewed feature profile")
        candidates.append(
            ModelCandidate(
                candidate_id=row["candidate_id"],
                model_kind=row["model_kind"],
                feature_columns=tuple(FEATURE_PROFILES[row["feature_profile"]]),
                logistic_c=row.get("logistic_c", 1.0),
            )
        )
    return config, tuple(candidates)


def command_nested_benchmark(args):
    output = Path(args.output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"nested benchmark output exists: {output}")
    table = Path(args.table)
    grid = Path(args.model_grid)
    content = table.read_bytes()
    grid_content = grid.read_bytes()
    config, candidates = _candidate_config(grid_content)
    if table.suffix.lower() not in {".csv", ".tsv", ".txt"}:
        raise ValueError("--table must be CSV or TSV")
    frame = pd.read_csv(
        BytesIO(content), sep="," if table.suffix.lower() == ".csv" else "\t"
    )
    deployment = getattr(args, "deployment_selection", False)
    options = {
        "candidates": candidates,
        "inner_splits": args.inner_folds,
        "selection_metric": config["selection_metric"],
        "random_state": args.random_state,
    }
    result = (
        select_final_candidate(frame, **options)
        if deployment
        else nested_grouped_oof_predictions(
            frame, outer_splits=args.outer_folds, **options
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        prediction_path = staging / "nested_predictions.tsv"
        if not deployment:
            result.predictions.to_csv(prediction_path, sep="\t", index=False)
        (staging / "model_grid.json").write_bytes(grid_content)
        report = {
            "schema_version": 1,
            "bundle_type": (
                "development_only_model_selection"
                if deployment
                else "nested_model_selection_benchmark"
            ),
            "inputs": {
                "training_table": {
                    "filename": table.name,
                    "sha256": hashlib.sha256(content).hexdigest(),
                },
                "model_grid": {
                    "filename": grid.name,
                    "sha256": hashlib.sha256(grid_content).hexdigest(),
                },
            },
            "outputs": {}
            if deployment
            else {
                "predictions": {
                    "filename": prediction_path.name,
                    "sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
                    "row_count": len(result.predictions),
                }
            },
            "parameters": {
                "outer_folds": None if deployment else args.outer_folds,
                "inner_folds": args.inner_folds,
                "random_state": args.random_state,
            },
            "metrics": None if deployment else result.metrics,
            "selected_candidate": asdict(result.selected_candidate)
            if deployment
            else None,
            "diagnostics": result.diagnostics,
            "scientific_boundary": {
                "evaluates_model_selection_procedure": not deployment,
                "selection_score_is_unbiased_performance_estimate": False,
                "does_not_select_final_model_from_outer_scores": True,
                "real_biological_performance_requires_real_adjudicated_data": True,
                "preregistration_not_created": True,
            },
        }
        from .cli import _benchmark_software_provenance

        report["software_provenance"] = _benchmark_software_provenance()
        encoded = json.dumps(
            _finite_json(report), sort_keys=True, indent=2, allow_nan=False
        )
        (staging / "nested_benchmark.json").write_text(encoded + "\n", encoding="utf-8")
        if table.read_bytes() != content or grid.read_bytes() != grid_content:
            raise ValueError("nested benchmark input changed during evaluation")
        _publish_directory_noreplace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(encoded)
    return 0


def command_predict_frozen(args):
    return _print(
        predict_frozen_model(
            args.table,
            args.model_bundle_dir,
            args.output_dir,
            expected_model_manifest_sha256=args.expected_model_manifest_sha256,
            trusted_local_bundle=args.trusted_local_bundle,
            created_at_utc=args.created_at_utc,
            code_revision=args.code_revision,
            allow_training_overlap=args.allow_training_overlap,
            max_missing_feature_fraction=args.max_missing_feature_fraction,
            baseline_report_dir=args.baseline_report_dir,
            expected_baseline_manifest_sha256=args.expected_baseline_manifest_sha256,
        )
    )


def command_freeze_selection(args):
    return _print(
        freeze_prospective_selection(
            args.roster,
            args.output_dir,
            expected_roster_sha256=args.expected_roster_sha256,
            model_manifest_path=args.model_manifest,
            expected_model_manifest_sha256=args.expected_model_manifest_sha256,
            prediction_manifest_path=args.prediction_manifest,
            expected_prediction_manifest_sha256=args.expected_prediction_manifest_sha256,
            sap_path=args.analysis_plan,
            expected_sap_sha256=args.expected_analysis_plan_sha256,
            code_revision=args.code_revision,
            created_at_utc=args.created_at_utc,
            top_k=args.top_k,
            random_comparison_n=args.random_comparison_n,
            random_seed=args.random_seed,
            baseline_score_column=args.baseline_score_column,
            baseline_higher_is_better=args.baseline_higher_is_better,
        )
    )


def command_verify_selection(args):
    return _print(
        verify_prospective_selection(
            args.bundle_dir,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
    )


def add_workflow_parsers(subparsers):
    nested = subparsers.add_parser(
        "nested-benchmark",
        help="evaluate a model grid using inner selection and outer family folds",
    )
    nested.add_argument("--table", required=True)
    nested.add_argument("--model-grid", required=True)
    nested.add_argument("--output-dir", required=True)
    nested.add_argument("--outer-folds", type=int, default=5)
    nested.add_argument("--inner-folds", type=int, default=3)
    nested.add_argument("--random-state", type=int, default=17)
    nested.set_defaults(func=command_nested_benchmark)

    deployment = subparsers.add_parser(
        "select-deployment-model",
        help="choose the final configuration on development-only grouped inner scores",
    )
    deployment.add_argument("--table", required=True)
    deployment.add_argument("--model-grid", required=True)
    deployment.add_argument("--output-dir", required=True)
    deployment.add_argument("--inner-folds", type=int, default=3)
    deployment.add_argument("--random-state", type=int, default=17)
    deployment.set_defaults(func=command_nested_benchmark, deployment_selection=True)

    predict = subparsers.add_parser(
        "predict-frozen-model",
        help="apply a trusted frozen KO model to an outcome-free feature table",
    )
    predict.add_argument("--table", required=True)
    predict.add_argument("--model-bundle-dir", required=True)
    predict.add_argument("--baseline-report-dir")
    predict.add_argument("--expected-baseline-manifest-sha256")
    predict.add_argument("--expected-model-manifest-sha256", required=True)
    predict.add_argument(
        "--trusted-local-bundle",
        action="store_true",
        help="acknowledge this trusted model's pickle can execute code",
    )
    predict.add_argument(
        "--allow-training-overlap",
        action="store_true",
        help="exploratory scoring only; no independent-evaluation claim",
    )
    predict.add_argument("--max-missing-feature-fraction", type=float, default=1.0)
    predict.add_argument("--created-at-utc", type=datetime.fromisoformat, required=True)
    predict.add_argument("--code-revision", required=True)
    predict.add_argument("--output-dir", required=True)
    predict.set_defaults(func=command_predict_frozen)

    select = subparsers.add_parser(
        "freeze-target-selection",
        help="freeze paired top-k lists and random comparison candidates",
    )
    for name in (
        "roster",
        "expected-roster-sha256",
        "model-manifest",
        "expected-model-manifest-sha256",
        "prediction-manifest",
        "expected-prediction-manifest-sha256",
        "analysis-plan",
        "expected-analysis-plan-sha256",
        "code-revision",
        "output-dir",
    ):
        select.add_argument(f"--{name}", required=True)
    select.add_argument("--created-at-utc", type=datetime.fromisoformat, required=True)
    select.add_argument("--top-k", type=int, required=True)
    select.add_argument("--random-comparison-n", type=int, required=True)
    select.add_argument("--random-seed", type=int, required=True)
    select.add_argument("--baseline-score-column", default="screen_signal_rank")
    select.add_argument("--baseline-higher-is-better", action="store_true")
    select.set_defaults(func=command_freeze_selection)

    verify = subparsers.add_parser(
        "verify-target-selection",
        help="verify and re-derive frozen candidate allocation without loading a model",
    )
    verify.add_argument("--bundle-dir", required=True)
    verify.add_argument("--expected-manifest-sha256")
    verify.set_defaults(func=command_verify_selection)
