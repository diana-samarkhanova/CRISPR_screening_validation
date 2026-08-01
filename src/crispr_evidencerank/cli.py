"""Command-line interface for validation, featurization, and benchmarking."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import get_args

import pandas as pd

from .contracts import CONTRACTS, validate_records
from .curation import write_curation_batch, write_dual_review_bundle
from .features import featurize_count_table, featurize_experimental_design
from .intake import SUPPORTED_POLICY_VERSIONS, triage_orcs_index
from .io import read_table
from .modeling import (
    DEFAULT_FEATURE_COLUMNS,
    FEATURE_PROFILES,
    grouped_oof_predictions,
)
from .orcs import parse_orcs_index, parse_orcs_screen_scores
from .orcs_prepare import prepare_orcs_release
from .orcs_release import load_orcs_release_spec


def _write_frame(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() in {".tsv", ".txt"}:
        frame.to_csv(path, sep="\t", index=False)
    else:
        raise ValueError("output must be CSV or TSV")


def command_validate(args: argparse.Namespace) -> int:
    model = CONTRACTS[args.contract]
    string_fields = {
        name: "string"
        for name, field in model.model_fields.items()
        if field.annotation is str or str in get_args(field.annotation)
    }
    frame = read_table(args.table, dtype=string_fields)
    valid, errors = validate_records(frame, args.contract)
    report = {
        "contract": args.contract,
        "records": len(frame),
        "valid_records": len(valid),
        "errors": errors.to_dict(orient="records"),
    }
    print(json.dumps(report, indent=2))
    return 1 if not errors.empty else 0


def command_featurize(args: argparse.Namespace) -> int:
    counts = read_table(args.counts)
    samples = read_table(args.samples)
    features = featurize_count_table(
        counts,
        samples,
        pseudocount=args.pseudocount,
        low_count_threshold=args.low_count_threshold,
        normalization_method=args.normalization_method,
        direction_deadband=args.direction_deadband,
    )
    _write_frame(features, args.output)
    print(json.dumps({"output": str(args.output), "rows": len(features)}, indent=2))
    return 0


def command_featurize_design(args: argparse.Namespace) -> int:
    features = featurize_experimental_design(
        read_table(args.screens),
        read_table(args.screen_designs),
        read_table(args.contrasts),
        read_table(args.samples),
    )
    _write_frame(features, args.output)
    print(json.dumps({"output": str(args.output), "rows": len(features)}, indent=2))
    return 0


def _write_orcs_result(
    output_dir: str | Path,
    frames: dict[str, pd.DataFrame],
    *,
    header_map: dict[str, str],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        _write_frame(frame, output_dir / f"{name}.tsv")
    (output_dir / "header_map.json").write_text(
        json.dumps(header_map, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def command_ingest_orcs_index(args: argparse.Namespace) -> int:
    parsed = parse_orcs_index(
        args.table,
        release=args.release,
        retrieved_date=args.retrieved_date,
        available_date=args.available_date,
        organism_scope=args.organism_scope,
    )
    frames = {
        "raw_index": parsed.raw_index,
        "normalized_index": parsed.normalized_index,
        "studies": parsed.studies,
        "screens": parsed.screens,
        "screen_designs": parsed.screen_designs,
        "contrasts": parsed.contrasts,
        "external_screen_maps": parsed.external_screen_maps,
    }
    _write_orcs_result(
        args.output_dir,
        frames,
        header_map=parsed.header_map,
    )
    print(
        json.dumps(
            {
                "release": parsed.release,
                "output_dir": str(args.output_dir),
                "screens": len(parsed.screens),
                "studies": len(parsed.studies),
            },
            indent=2,
        )
    )
    return 0


def command_triage_orcs_index(args: argparse.Namespace) -> int:
    result = triage_orcs_index(
        args.table,
        release=args.release,
        retrieved_date=args.retrieved_date,
        available_date=args.available_date,
        organism_scope=args.organism_scope,
        policy_version=args.policy_version,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(result.screen_intake, output_dir / "screen_intake.tsv")
    _write_frame(
        result.eligibility_checks,
        output_dir / "eligibility_checks.tsv",
    )
    _write_frame(result.curation_queue, output_dir / "curation_queue.tsv")
    candidate_text = "".join(
        f"{screen_id}\n" for screen_id in result.candidate_screen_ids
    )
    (output_dir / "candidate_screen_ids.txt").write_text(
        candidate_text,
        encoding="utf-8",
    )
    (output_dir / "triage_summary.json").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    return 0


def command_prepare_orcs_release(args: argparse.Namespace) -> int:
    spec = load_orcs_release_spec(args.release_registry, release=args.release)
    prepared = prepare_orcs_release(
        spec,
        args.output_dir,
        retrieved_date=args.retrieved_date,
        policy_version=args.policy_version,
        archive_path=args.archive,
        cache_dir=args.cache_dir,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(prepared.summary, indent=2, sort_keys=True))
    return 0


def command_select_curation_batch(args: argparse.Namespace) -> int:
    manifest = write_curation_batch(
        args.queue,
        args.output_dir,
        batch_id=args.batch_id,
        selected_date=args.selected_date,
        start_rank=args.start_rank,
        batch_size=args.batch_size,
        require_unique_source_families=not args.allow_repeated_source_families,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def command_compare_curation_reviews(args: argparse.Namespace) -> int:
    manifest = write_dual_review_bundle(
        args.primary_reviews,
        args.secondary_reviews,
        args.selection,
        args.output_dir,
        assessed_date=args.assessed_date,
        allow_partial_secondary=args.allow_partial_secondary,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def command_ingest_orcs_screen(args: argparse.Namespace) -> int:
    metadata = read_table(args.index_metadata) if args.index_metadata else None
    parsed = parse_orcs_screen_scores(
        args.table,
        release=args.release,
        index_metadata=metadata,
        contrast_id=args.contrast_id,
        source_file=Path(args.table).name,
    )
    frames = {
        "raw_scores": parsed.raw_scores,
        "normalized_scores": parsed.normalized_scores,
        "gene_scores": parsed.gene_scores,
        "issues": parsed.issues,
    }
    _write_orcs_result(
        args.output_dir,
        frames,
        header_map=parsed.header_map,
    )
    print(
        json.dumps(
            {
                "release": parsed.release,
                "output_dir": str(args.output_dir),
                "gene_score_rows": len(parsed.gene_scores),
                "issues": len(parsed.issues),
            },
            indent=2,
        )
    )
    return 1 if not parsed.issues.empty else 0


def command_benchmark(args: argparse.Namespace) -> int:
    frame = read_table(args.table)
    if args.features and args.feature_profile:
        raise ValueError("--features and --feature-profile are mutually exclusive")
    if args.features:
        feature_columns = [
            value.strip() for value in args.features.split(",") if value.strip()
        ]
    elif args.feature_profile:
        feature_columns = FEATURE_PROFILES[args.feature_profile]
    else:
        feature_columns = DEFAULT_FEATURE_COLUMNS
    predictions, metrics = grouped_oof_predictions(
        frame,
        feature_columns=feature_columns,
        group_column=args.group_column,
        screen_column=args.screen_column,
        n_splits=args.folds,
        model_kind=args.model_kind,
    )
    _write_frame(predictions, args.output)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crispr-evidencerank")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a registry table")
    validate.add_argument("--table", required=True)
    validate.add_argument("--contract", choices=sorted(CONTRACTS), required=True)
    validate.set_defaults(func=command_validate)

    featurize = subparsers.add_parser(
        "featurize-counts", help="aggregate guide counts to gene features"
    )
    featurize.add_argument("--counts", required=True)
    featurize.add_argument("--samples", required=True)
    featurize.add_argument("--output", required=True)
    featurize.add_argument("--pseudocount", type=float, default=1.0)
    featurize.add_argument("--low-count-threshold", type=float, default=30.0)
    featurize.add_argument(
        "--normalization-method",
        choices=["median_ratio", "cpm"],
        default="median_ratio",
    )
    featurize.add_argument("--direction-deadband", type=float, default=0.1)
    featurize.set_defaults(func=command_featurize)

    design = subparsers.add_parser(
        "featurize-design",
        help="derive contrast-level prevalidation design features",
    )
    design.add_argument("--screens", required=True)
    design.add_argument("--screen-designs", required=True)
    design.add_argument("--contrasts", required=True)
    design.add_argument("--samples", required=True)
    design.add_argument("--output", required=True)
    design.set_defaults(func=command_featurize_design)

    orcs_index = subparsers.add_parser(
        "ingest-orcs-index",
        help="normalize a release-pinned BioGRID ORCS screen index",
    )
    orcs_index.add_argument("--table", required=True)
    orcs_index.add_argument("--release", required=True)
    orcs_index.add_argument("--retrieved-date", required=True)
    orcs_index.add_argument("--available-date")
    orcs_index.add_argument("--organism-scope")
    orcs_index.add_argument("--output-dir", required=True)
    orcs_index.set_defaults(func=command_ingest_orcs_index)

    orcs_triage = subparsers.add_parser(
        "triage-orcs-index",
        help="triage ORCS screens as exclude or metadata-only candidates",
    )
    orcs_triage.add_argument("--table", required=True)
    orcs_triage.add_argument("--release", required=True)
    orcs_triage.add_argument("--retrieved-date", required=True)
    orcs_triage.add_argument("--available-date")
    orcs_triage.add_argument("--organism-scope")
    orcs_triage.add_argument(
        "--policy-version",
        type=int,
        choices=sorted(SUPPORTED_POLICY_VERSIONS),
        default=2,
    )
    orcs_triage.add_argument("--output-dir", required=True)
    orcs_triage.set_defaults(func=command_triage_orcs_index)

    orcs_prepare = subparsers.add_parser(
        "prepare-orcs-release",
        help="verify and atomically prepare one pinned BioGRID ORCS release",
    )
    orcs_prepare.add_argument(
        "--release-registry",
        default="config/orcs_releases.yaml",
    )
    orcs_prepare.add_argument("--release", default="2.0.18")
    orcs_prepare.add_argument(
        "--retrieved-date",
        type=date.fromisoformat,
        required=True,
    )
    orcs_prepare.add_argument("--archive")
    orcs_prepare.add_argument("--cache-dir", default="data/external/orcs")
    orcs_prepare.add_argument(
        "--policy-version",
        type=int,
        choices=sorted(SUPPORTED_POLICY_VERSIONS),
        default=2,
    )
    orcs_prepare.add_argument("--timeout-seconds", type=float, default=60.0)
    orcs_prepare.add_argument("--output-dir", required=True)
    orcs_prepare.set_defaults(func=command_prepare_orcs_release)

    orcs_screen = subparsers.add_parser(
        "ingest-orcs-screen",
        help="normalize BioGRID ORCS per-screen author gene scores",
    )
    orcs_screen.add_argument("--table", required=True)
    orcs_screen.add_argument("--release", required=True)
    orcs_screen.add_argument("--index-metadata")
    orcs_screen.add_argument("--contrast-id")
    orcs_screen.add_argument("--output-dir", required=True)
    orcs_screen.set_defaults(func=command_ingest_orcs_screen)

    curation_batch = subparsers.add_parser(
        "select-curation-batch",
        help="freeze a contiguous outcome-blind full-text curation batch",
    )
    curation_batch.add_argument("--queue", required=True)
    curation_batch.add_argument("--output-dir", required=True)
    curation_batch.add_argument("--batch-id", required=True)
    curation_batch.add_argument(
        "--selected-date", type=date.fromisoformat, required=True
    )
    curation_batch.add_argument("--start-rank", type=int, default=1)
    curation_batch.add_argument("--batch-size", type=int, default=10)
    curation_batch.add_argument(
        "--allow-repeated-source-families",
        action="store_true",
    )
    curation_batch.set_defaults(func=command_select_curation_batch)

    review_comparison = subparsers.add_parser(
        "compare-curation-reviews",
        help="compare two review sets without releasing benchmark labels",
    )
    review_comparison.add_argument("--primary-reviews", required=True)
    review_comparison.add_argument("--secondary-reviews", required=True)
    review_comparison.add_argument("--selection", required=True)
    review_comparison.add_argument("--output-dir", required=True)
    review_comparison.add_argument(
        "--assessed-date", type=date.fromisoformat, required=True
    )
    review_comparison.add_argument(
        "--allow-partial-secondary",
        action="store_true",
        help="explicitly permit a checksum-bound incomplete secondary review",
    )
    review_comparison.set_defaults(func=command_compare_curation_reviews)

    benchmark = subparsers.add_parser(
        "benchmark", help="run grouped out-of-fold baseline evaluation"
    )
    benchmark.add_argument("--table", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--features")
    benchmark.add_argument(
        "--feature-profile",
        choices=sorted(FEATURE_PROFILES),
    )
    benchmark.add_argument("--group-column", default="study_id")
    benchmark.add_argument("--screen-column", default="screen_id")
    benchmark.add_argument("--folds", type=int, default=5)
    benchmark.add_argument(
        "--model-kind",
        choices=["logistic", "hist_gradient_boosting"],
        default="logistic",
    )
    benchmark.set_defaults(func=command_benchmark)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))
