import pandas as pd
import pytest

from crispr_evidencerank.features import (
    featurize_count_table,
    featurize_experimental_design,
)


def test_positive_treatment_shift_produces_positive_lfc():
    counts = pd.DataFrame(
        {
            "sgrna_id": ["g1", "g2", "g3", "g4"],
            "gene_symbol": ["A", "A", "B", "B"],
            "c1": [100, 120, 100, 120],
            "c2": [110, 115, 110, 115],
            "t1": [400, 450, 25, 30],
            "t2": [420, 430, 30, 35],
        }
    )
    samples = pd.DataFrame(
        {
            "sample_id": ["c1", "c2", "t1", "t2"],
            "screen_id": ["S1"] * 4,
            "contrast_id": ["S1_drug"] * 4,
            "condition_role": ["control", "control", "treatment", "treatment"],
            "replicate": [1, 2, 1, 2],
        }
    )
    features = featurize_count_table(counts, samples).set_index("gene_symbol")
    assert features.loc["A", "median_guide_lfc"] > 0
    assert features.loc["B", "median_guide_lfc"] < 0
    assert features.loc["A", "guide_direction_agreement"] == 1.0


def test_zero_effect_guides_do_not_look_directionally_concordant():
    counts = pd.DataFrame(
        {
            "sgrna_id": ["g1", "g2"],
            "gene_symbol": ["A", "A"],
            "c1": [100, 100],
            "t1": [100, 100],
        }
    )
    samples = pd.DataFrame(
        {
            "sample_id": ["c1", "t1"],
            "screen_id": ["S1", "S1"],
            "contrast_id": ["S1_drug", "S1_drug"],
            "condition_role": ["control", "treatment"],
            "replicate": [1, 1],
        }
    )
    features = featurize_count_table(counts, samples)
    assert features.loc[0, "guide_direction_agreement"] == 0.0
    assert features.loc[0, "neutral_guide_fraction"] == 1.0


def test_numeric_sample_ids_are_normalized():
    counts = pd.DataFrame(
        {
            "sgrna_id": ["g1", "g2"],
            "gene_symbol": ["A", "A"],
            1: [100, 120],
            2: [200, 240],
        }
    )
    samples = pd.DataFrame(
        {
            "sample_id": [1, 2],
            "screen_id": ["S1", "S1"],
            "contrast_id": ["C1", "C1"],
            "condition_role": ["control", "treatment"],
            "replicate": [1, 1],
        }
    )
    features = featurize_count_table(counts, samples)
    assert len(features) == 1


def test_invalid_count_inputs_are_rejected():
    counts = pd.DataFrame(
        {
            "sgrna_id": ["g1"],
            "gene_symbol": ["A"],
            "c1": [float("inf")],
            "t1": [100],
        }
    )
    samples = pd.DataFrame(
        {
            "sample_id": ["c1", "t1"],
            "screen_id": ["S1", "S1"],
            "contrast_id": ["C1", "C1"],
            "condition_role": ["control", "treatment"],
            "replicate": [1, 1],
        }
    )
    with pytest.raises(ValueError, match="finite"):
        featurize_count_table(counts, samples)
    with pytest.raises(ValueError, match="pseudocount"):
        featurize_count_table(
            counts.assign(c1=100),
            samples,
            pseudocount=0,
        )


def test_median_ratio_normalization_resists_compositional_shift():
    counts = pd.DataFrame(
        {
            "sgrna_id": ["g1", "g2", "g3"],
            "gene_symbol": ["A", "B", "C"],
            "c1": [100, 100, 100],
            "t1": [10_000, 100, 100],
        }
    )
    samples = pd.DataFrame(
        {
            "sample_id": ["c1", "t1"],
            "screen_id": ["S1", "S1"],
            "contrast_id": ["C1", "C1"],
            "condition_role": ["control", "treatment"],
            "replicate": [1, 1],
        }
    )
    features = featurize_count_table(counts, samples).set_index("gene_symbol")
    assert features.loc["A", "median_guide_lfc"] > 5
    assert abs(features.loc["B", "median_guide_lfc"]) < 0.05
    assert abs(features.loc["C", "median_guide_lfc"]) < 0.05
    assert features.loc["B", "signal_direction"] == "neutral"


def test_all_zero_guides_do_not_bias_median_ratio_size_factors():
    counts = pd.DataFrame(
        {
            "sgrna_id": ["z1", "z2", "z3", "g1", "g2"],
            "gene_symbol": ["Z1", "Z2", "Z3", "A", "B"],
            "c1": [0, 0, 0, 100, 200],
            "t1": [0, 0, 0, 200, 400],
        }
    )
    samples = pd.DataFrame(
        {
            "sample_id": ["c1", "t1"],
            "screen_id": ["S1", "S1"],
            "contrast_id": ["C1", "C1"],
            "condition_role": ["control", "treatment"],
            "replicate": [1, 1],
        }
    )
    features = featurize_count_table(counts, samples).set_index("gene_symbol")
    assert abs(features.loc["A", "median_guide_lfc"]) < 0.05
    assert abs(features.loc["B", "median_guide_lfc"]) < 0.05


def test_fractional_counts_and_unclassified_columns_are_rejected():
    samples = pd.DataFrame(
        {
            "sample_id": ["c1", "t1"],
            "screen_id": ["S1", "S1"],
            "contrast_id": ["C1", "C1"],
            "condition_role": ["control", "treatment"],
            "replicate": [1, 1],
        }
    )
    fractional = pd.DataFrame(
        {
            "sgrna_id": ["g1"],
            "gene_symbol": ["A"],
            "c1": [10.5],
            "t1": [11],
        }
    )
    with pytest.raises(ValueError, match="integer-valued"):
        featurize_count_table(fractional, samples)
    extra = fractional.assign(c1=10, annotation="unexpected")
    with pytest.raises(ValueError, match="not declared"):
        featurize_count_table(extra, samples)


def test_design_features_preserve_zero_and_pairing_metadata():
    screens = pd.DataFrame(
        [
            {
                "screen_id": "SC_ZERO",
                "study_id": "S1",
                "source_family_id": "SF1",
                "raw_data_family_id": "F1",
            },
            {
                "screen_id": "SC_MISSING",
                "study_id": "S2",
                "source_family_id": None,
                "raw_data_family_id": None,
            },
        ]
    )
    screen_designs = pd.DataFrame(
        [
            {
                "screen_id": "SC_ZERO",
                "nontargeting_guide_count": 0,
                "infection_replicate_count": 1,
                "replicate_unit": "independent_infection",
                "screen_scale": "genome_wide",
                "selection_strategy": "positive_selection",
            },
            {
                "screen_id": "SC_MISSING",
                "nontargeting_guide_count": None,
                "infection_replicate_count": None,
                "replicate_unit": "unknown",
                "screen_scale": "unknown",
                "selection_strategy": "unknown",
            },
        ]
    )
    contrasts = pd.DataFrame(
        [
            {
                "screen_id": "SC_ZERO",
                "contrast_id": "C1",
                "control_type": "vehicle",
                "intended_direction": "resistance",
                "treatment_dose": 0.0,
                "vehicle_control_present": True,
                "baseline_control_present": False,
                "same_infection_split": True,
                "matched_control": True,
            },
            {
                "screen_id": "SC_MISSING",
                "contrast_id": "C2",
                "control_type": "untreated",
                "intended_direction": "resistance",
                "treatment_dose": None,
                "vehicle_control_present": None,
                "baseline_control_present": None,
                "same_infection_split": None,
                "matched_control": None,
            },
        ]
    )
    samples = pd.DataFrame(
        [
            {
                "sample_id": "C1",
                "screen_id": "SC_ZERO",
                "contrast_id": "C1",
                "condition_role": "control",
                "replicate": 1,
                "pair_id": "P1",
                "infection_replicate_id": "I1",
            },
            {
                "sample_id": "T1",
                "screen_id": "SC_ZERO",
                "contrast_id": "C1",
                "condition_role": "treatment",
                "replicate": 1,
                "pair_id": "P1",
                "infection_replicate_id": "I1",
            },
            {
                "sample_id": "C2",
                "screen_id": "SC_MISSING",
                "contrast_id": "C2",
                "condition_role": "control",
                "replicate": 1,
            },
            {
                "sample_id": "T2",
                "screen_id": "SC_MISSING",
                "contrast_id": "C2",
                "condition_role": "treatment",
                "replicate": 1,
            },
        ]
    )

    features = featurize_experimental_design(
        screens, screen_designs, contrasts, samples
    ).set_index("screen_id")
    reported = features.loc["SC_ZERO"]
    missing = features.loc["SC_MISSING"]

    assert reported["nontargeting_guide_count"] == 0
    assert reported["source_family_id"] == "SF1"
    assert pd.isna(missing["source_family_id"])
    assert reported["nontargeting_guide_count_missing"] == 0
    assert missing["nontargeting_guide_count_missing"] == 1
    assert reported["treatment_dose_reported"] == 1
    assert missing["treatment_dose_reported"] == 0
    assert reported["control_type_vehicle"] == 1
    assert reported["vehicle_control_present"] == 1
    assert reported["same_infection_split"] == 1
    assert reported["matched_control"] == 1
    assert reported["paired_replicate_fraction"] == 1
