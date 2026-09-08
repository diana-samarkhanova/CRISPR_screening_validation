"""Synthetic-only installed CLI chain from model selection to assay allocation.

All generated candidates, scores and plans exercise software, not biology.
Outputs belong outside the repository and are never a deployable trained model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    executable = Path(sys.executable).parent / "crispr-evidencerank"
    revision = (
        os.environ.get("CRISPR_EVIDENCERANK_BUILD_REVISION")
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    )

    def run(command, *arguments):
        result = subprocess.run(
            [str(executable), command, *map(str, arguments)],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        (output / f"{command}.stdout.json").write_text(result.stdout)
        return json.loads(result.stdout)

    table = ROOT / "examples/synthetic/labeled_gene_features.csv"
    grid = ROOT / "examples/synthetic/model_grid.json"
    run(
        "nested-benchmark",
        "--table",
        table,
        "--model-grid",
        grid,
        "--outer-folds",
        3,
        "--inner-folds",
        2,
        "--output-dir",
        output / "nested",
    )
    selected = run(
        "select-deployment-model",
        "--table",
        table,
        "--model-grid",
        grid,
        "--inner-folds",
        2,
        "--output-dir",
        output / "selection",
    )
    config = selected["selected_candidate"]
    model_dir = output / "model"
    run(
        "freeze-model",
        "--table",
        table,
        "--features",
        ",".join(config["feature_columns"]),
        "--model-kind",
        config["model_kind"],
        "--logistic-c",
        config["logistic_c"],
        "--code-revision",
        revision,
        "--created-at-utc",
        "2026-09-08T12:00:00+00:00",
        "--output-dir",
        model_dir,
    )

    source = pd.read_csv(table).iloc[:12].copy()
    keep = [c for c in config["feature_columns"] if c in source]
    keep += [
        "gene_symbol",
        "median_guide_lfc",
        "mean_guide_lfc",
        "signal_direction",
        "positive_lfc_means",
    ]
    candidates = source[list(dict.fromkeys(keep))].copy()
    for column in (
        "study_id",
        "screen_id",
        "contrast_id",
        "source_family_id",
        "raw_data_family_id",
    ):
        candidates[column] = f"synthetic_future_{column}"
    candidates["perturbation_modality"] = "CRISPR_KO"
    candidates["phenotype_direction"] = "resistance"
    candidates["candidate_role"] = "candidate"
    candidates["prior_outcome_exposure"] = "not_seen"
    candidates["screen_signal_rank"] = np.arange(len(candidates), 0, -1)
    candidate_path = output / "synthetic_candidates.tsv"
    candidates.to_csv(candidate_path, sep="\t", index=False)

    summary = pd.DataFrame(
        {
            "id": candidates.gene_symbol,
            "pos|score": 0.1,
            "pos|fdr": 0.2,
            "pos|rank": candidates.screen_signal_rank,
            "pos|lfc": 1.0,
            "neg|score": 0.3,
            "neg|fdr": 0.4,
            "neg|rank": np.arange(1, len(candidates) + 1),
            "neg|lfc": -1.0,
        }
    )
    summary_path = output / "synthetic_mageck.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    baseline = output / "baseline"
    run(
        "rank-screen",
        "--mageck-summary",
        summary_path,
        "--screen-id",
        "synthetic_future_screen_id",
        "--contrast-id",
        "synthetic_future_contrast_id",
        "--positive-tail-means",
        "resistance",
        "--output-dir",
        baseline,
    )
    model_manifest = model_dir / "frozen_model_manifest.json"
    prediction = output / "prediction"
    run(
        "predict-frozen-model",
        "--table",
        candidate_path,
        "--model-bundle-dir",
        model_dir,
        "--trusted-local-bundle",
        "--expected-model-manifest-sha256",
        sha(model_manifest),
        "--baseline-report-dir",
        baseline,
        "--expected-baseline-manifest-sha256",
        sha(baseline / "run_manifest.json"),
        "--code-revision",
        revision,
        "--created-at-utc",
        "2026-09-08T12:01:00+00:00",
        "--output-dir",
        prediction,
    )
    plan = output / "synthetic_plan.md"
    plan.write_text(
        "Synthetic software test only; not an approved or registered study.\n"
    )
    allocation = output / "allocation"
    run(
        "freeze-target-selection",
        "--roster",
        prediction / "predictions.tsv",
        "--expected-roster-sha256",
        sha(prediction / "predictions.tsv"),
        "--model-manifest",
        model_manifest,
        "--expected-model-manifest-sha256",
        sha(model_manifest),
        "--prediction-manifest",
        prediction / "prediction_manifest.json",
        "--expected-prediction-manifest-sha256",
        sha(prediction / "prediction_manifest.json"),
        "--analysis-plan",
        plan,
        "--expected-analysis-plan-sha256",
        sha(plan),
        "--top-k",
        2,
        "--random-comparison-n",
        2,
        "--random-seed",
        17,
        "--code-revision",
        revision,
        "--created-at-utc",
        "2026-09-08T12:02:00+00:00",
        "--output-dir",
        allocation,
    )
    run(
        "verify-target-selection",
        "--bundle-dir",
        allocation,
        "--expected-manifest-sha256",
        sha(allocation / "prospective_selection_manifest.json"),
    )
    assert (allocation / "assay_roster.tsv").is_file()
    print("Synthetic nested -> model -> prediction -> target allocation: passed")


if __name__ == "__main__":
    main()
