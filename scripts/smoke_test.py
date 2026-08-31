"""Dependency-light verification for environments without pytest."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from crispr_evidencerank.contracts import (
    CONTRACTS,
    ValidationEventRecord,
    validate_records,
    validate_registry_integrity,
)
from crispr_evidencerank.evaluation import study_cluster_bootstrap_intervals
from crispr_evidencerank.features import (
    featurize_count_table,
    featurize_experimental_design,
)
from crispr_evidencerank.labels import LabelCode, model_target, resolve_event_labels
from crispr_evidencerank.modeling import (
    DEFAULT_FEATURE_COLUMNS,
    FEATURE_PROFILES,
    WeightedMedianStandardizer,
    grouped_oof_predictions,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    assert model_target("V3") == 1
    assert model_target("F0") == 0
    assert model_target("U") is None
    assert resolve_event_labels(["V2", "F0"]) == LabelCode.A
    feature_config = yaml.safe_load(
        (ROOT / "config/features.yaml").read_text(encoding="utf-8")
    )
    declared_features = {
        feature
        for group in feature_config["feature_groups"].values()
        for feature in group.get("features", [])
    }
    assert set(DEFAULT_FEATURE_COLUMNS) <= declared_features
    assert set(FEATURE_PROFILES["screen_plus_design"]) <= declared_features
    label_config = yaml.safe_load(
        (ROOT / "config/labels.yaml").read_text(encoding="utf-8")
    )
    assert set(label_config["labels"]) == {label.value for label in LabelCode}

    fixture_contracts = {
        "study": "studies.csv",
        "screen": "screens.csv",
        "library": "libraries.csv",
        "screen_design": "screen_designs.csv",
        "contrast": "contrasts.csv",
        "sample": "sample_sheet.csv",
        "candidate": "candidate_status.csv",
        "validation_event": "validation_events.csv",
    }
    for contract_name, filename in fixture_contracts.items():
        fixture = pd.read_csv(ROOT / "examples/synthetic" / filename)
        fixture_valid, fixture_errors = validate_records(
            fixture,
            contract_name,
        )
        assert len(fixture_valid) == len(fixture)
        assert fixture_errors.empty
    for contract_name, model in CONTRACTS.items():
        committed_schema = json.loads(
            (ROOT / "schemas" / f"{contract_name}.schema.json").read_text(
                encoding="utf-8"
            )
        )
        assert committed_schema == model.model_json_schema()

    events = pd.read_csv(ROOT / "examples/synthetic/validation_events.csv")
    valid, errors = validate_records(events, ValidationEventRecord)
    assert len(valid) == len(events)
    assert errors.empty
    validation_schema = ValidationEventRecord.model_json_schema()
    assert len(validation_schema.get("allOf", [])) == 9
    reagent_schema = validation_schema["allOf"][2]["then"]["anyOf"][0]["properties"][
        "independent_reagent_count"
    ]
    assert reagent_schema["type"] == "integer"
    registry_tables = {
        "studies": pd.read_csv(ROOT / "examples/synthetic/studies.csv"),
        "screens": pd.read_csv(ROOT / "examples/synthetic/screens.csv"),
        "libraries": pd.read_csv(ROOT / "examples/synthetic/libraries.csv"),
        "screen_designs": pd.read_csv(ROOT / "examples/synthetic/screen_designs.csv"),
        "contrasts": pd.read_csv(ROOT / "examples/synthetic/contrasts.csv"),
        "samples": pd.read_csv(ROOT / "examples/synthetic/sample_sheet.csv"),
        "candidates": pd.read_csv(ROOT / "examples/synthetic/candidate_status.csv"),
    }
    integrity_errors = validate_registry_integrity(
        **registry_tables,
        validation_events=events,
    )
    assert integrity_errors.to_dict(orient="records") == [
        {
            "table": "validation_events",
            "row_number": 1,
            "error": (
                "candidate labels require a checksum-verified adjudication "
                "release manifest"
            ),
        }
    ]
    assert validate_registry_integrity(**registry_tables).empty

    samples = pd.read_csv(ROOT / "examples/synthetic/sample_sheet.csv")
    counts = pd.read_csv(ROOT / "examples/synthetic/guide_counts.csv")
    first_screen = samples["screen_id"].iloc[0]
    design = samples.loc[samples["screen_id"] == first_screen]
    sample_ids = design["sample_id"].tolist()
    screen_counts = counts.loc[
        counts["sgrna_id"].str.startswith(first_screen),
        ["sgrna_id", "gene_symbol", *sample_ids],
    ]
    features = featurize_count_table(screen_counts, design)
    assert len(features) == 60
    assert features["guide_n"].eq(4).all()
    design_features = featurize_experimental_design(
        pd.read_csv(ROOT / "examples/synthetic/screens.csv"),
        pd.read_csv(ROOT / "examples/synthetic/screen_designs.csv"),
        pd.read_csv(ROOT / "examples/synthetic/contrasts.csv"),
        samples,
    )
    assert len(design_features) == 8
    assert design_features["paired_replicate_fraction"].eq(1.0).all()
    assert (
        design_features["recovery_days_missing"]
        .eq(design_features["recovery_days"].isna().astype(float))
        .all()
    )
    composition_counts = pd.DataFrame(
        {
            "sgrna_id": ["g1", "g2", "g3"],
            "gene_symbol": ["A", "B", "C"],
            "c1": [100, 100, 100],
            "t1": [10_000, 100, 100],
        }
    )
    composition_samples = pd.DataFrame(
        {
            "sample_id": ["c1", "t1"],
            "screen_id": ["S1", "S1"],
            "contrast_id": ["C1", "C1"],
            "condition_role": ["control", "treatment"],
            "replicate": [1, 1],
        }
    )
    composition_features = featurize_count_table(
        composition_counts,
        composition_samples,
    ).set_index("gene_symbol")
    assert abs(composition_features.loc["B", "median_guide_lfc"]) < 0.05
    assert composition_features.loc["B", "signal_direction"] == "neutral"
    assert composition_features.loc["B", "neutral_guide_fraction"] == 1
    assert composition_features.loc["B", "guide_direction_agreement"] == 0
    zero_heavy_counts = pd.DataFrame(
        {
            "sgrna_id": ["z1", "z2", "z3", "g1", "g2"],
            "gene_symbol": ["Z1", "Z2", "Z3", "A", "B"],
            "c1": [0, 0, 0, 100, 200],
            "t1": [0, 0, 0, 200, 400],
        }
    )
    zero_heavy_features = featurize_count_table(
        zero_heavy_counts,
        composition_samples,
    ).set_index("gene_symbol")
    assert abs(zero_heavy_features.loc["A", "median_guide_lfc"]) < 0.05
    weighted_preprocessor = WeightedMedianStandardizer().fit(
        pd.DataFrame({"feature": [1.0] * 100 + [100.0]}),
        sample_weight=pd.Series([0.01] * 100 + [1.0]),
    )
    assert abs(weighted_preprocessor.mean_[0] - 50.5) < 1e-9

    labeled = pd.read_csv(ROOT / "examples/synthetic/labeled_gene_features.csv")
    predictions, metrics = grouped_oof_predictions(labeled, n_splits=4)
    assert predictions["reproducibility_score"].notna().all()
    assert metrics["n_groups"] == 16
    assert 0 <= metrics["average_precision"] <= 1
    design_predictions, design_metrics = grouped_oof_predictions(
        labeled,
        feature_columns=FEATURE_PROFILES["screen_plus_design"],
        n_splits=4,
    )
    assert design_predictions["reproducibility_score"].notna().all()
    assert 0 <= design_metrics["average_precision"] <= 1
    independent_studies = labeled["study_id"].unique()[[0, 2]]
    two_studies = labeled.loc[labeled["study_id"].isin(independent_studies)].copy()
    two_study_predictions, two_study_metrics = grouped_oof_predictions(
        two_studies,
        n_splits=2,
    )
    assert two_study_metrics["ipw_available"] == 0
    assert two_study_predictions["reproducibility_score_ipw"].isna().all()
    local_contrasts = labeled.copy()
    local_contrasts["contrast_id"] = "drug_vs_control"
    _, local_contrast_metrics = grouped_oof_predictions(
        local_contrasts,
        n_splits=4,
    )
    assert local_contrast_metrics["n_groups"] == 16
    bootstrap_input = predictions.copy()
    intervals = study_cluster_bootstrap_intervals(
        bootstrap_input,
        study_column="study_id",
        screen_column="screen_id",
        target_column="target",
        score_column="reproducibility_score",
        n_bootstrap=10,
    )
    assert intervals["_diagnostics"]["valid_draws"] == 10
    assert intervals["global_average_precision"]["effective_draws"] == 10
    assert intervals["global_average_precision"]["interval_available"] == 1
    print(
        {
            "status": "passed",
            "validation_events": len(valid),
            "gene_feature_rows": len(labeled),
            "oof_average_precision": round(metrics["average_precision"], 4),
            "oof_ndcg_at_10": round(metrics["ndcg_at_10"], 4),
        }
    )


if __name__ == "__main__":
    main()
