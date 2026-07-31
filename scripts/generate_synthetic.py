"""Create deterministic, non-biological data for software verification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crispr_evidencerank.features import (
    featurize_count_table,
    featurize_experimental_design,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "synthetic"
RNG = np.random.default_rng(20260730)


def label_for_gene(gene_index: int, study_index: int) -> str:
    pattern = {
        0: "V3",
        1: "V2",
        2: "V2",
        3: "V2",
        4: "F0",
        6: "V2",
        7: "V2",
        8: "F0",
        9: "D",
        12: "F0",
        13: "F0",
        14: "V1",
        15: "A",
        16: "T",
    }
    label = pattern.get(gene_index, "U")
    if gene_index == 3 and study_index % 3 == 0:
        return "F0"
    return label


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    studies: list[dict[str, object]] = []
    screens: list[dict[str, object]] = []
    libraries: list[dict[str, object]] = []
    screen_designs: list[dict[str, object]] = []
    contrasts: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    count_blocks: list[pd.DataFrame] = []
    candidate_rows: list[dict[str, object]] = []

    gene_symbols = [f"GENE{index:03d}" for index in range(60)]
    libraries.append(
        {
            "library_id": "SYN-LIB-01",
            "library_name": "SyntheticLibrary",
            "library_version": "v1",
            "perturbation_modality": "CRISPR_KO",
            "library_scope": "genome_wide",
            "target_gene_count": len(gene_symbols),
            "total_guide_count": len(gene_symbols) * 4,
            "expected_guides_per_gene": 4.0,
            "nontargeting_guide_count": 20,
            "vector_architecture": "synthetic single-vector",
            "enzyme": "synthetic Cas9",
            "guide_target_region": "coding sequence",
            "reference_url": "https://example.org/synthetic",
            "source_locator": "software fixture",
            "notes": "Synthetic; not a biological library.",
        }
    )
    for study_index in range(8):
        study_id = f"SYN-STUDY-{study_index + 1:02d}"
        screen_id = f"SYN-SCREEN-{study_index + 1:02d}"
        contrast_id = f"{screen_id}-DRUG-vs-CTRL"
        drug_name = f"SYNTHETIC_DRUG_{study_index % 4 + 1}"
        cell_line = f"SYNTHETIC_CELL_{study_index + 1}"

        studies.append(
            {
                "study_id": study_id,
                "citation": f"Synthetic software test study {study_index + 1}",
                "doi": None,
                "publication_date": f"202{study_index % 5 + 1}-01-01",
                "organism": "human",
                "source_url": "https://example.org/synthetic",
                "data_license": "CC0-1.0",
                "source_role": "original_screen",
                "independent_screen_source": True,
                "source_id": f"SYN-SOURCE-{study_index + 1:02d}",
                "source_type": "software_fixture",
                "pubmed_id": None,
                "notes": "Synthetic; not biological evidence.",
            }
        )
        raw_data_family_id = (
            "SYN-FAMILY-SHARED"
            if study_index in {0, 1}
            else f"SYN-FAMILY-{study_index + 1:02d}"
        )
        source_family_id = (
            "SYN-SOURCE-FAMILY-SHARED"
            if study_index in {0, 1}
            else f"SYN-SOURCE-FAMILY-{study_index + 1:02d}"
        )
        screens.append(
            {
                "screen_id": screen_id,
                "study_id": study_id,
                "source_family_id": source_family_id,
                "raw_data_family_id": raw_data_family_id,
                "perturbation_modality": "CRISPR_KO",
                "screen_design": "pooled_competitive_growth",
                "cell_line": cell_line,
                "depmap_id": None,
                "cellosaurus_id": None,
                "engineered_genotype": "synthetic",
                "cancer_type": "synthetic",
                "library_id": "SYN-LIB-01",
                "drug_name": drug_name,
                "drug_id": None,
                "drug_class": "synthetic",
                "library_name": "SyntheticLibrary",
                "library_version": "v1",
                "treatment_dose": 1.0,
                "treatment_unit": "arbitrary_unit",
                "duration_days": 14.0,
                "intended_direction": "resistance",
                "input_mode": "count_table",
                "data_accession": None,
                "source_url": "https://example.org/synthetic",
                "available_date": "2026-07-30",
                "notes": "Synthetic; not biological evidence.",
            }
        )
        screen_designs.append(
            {
                "screen_id": screen_id,
                "source_role": "original_screen",
                "parent_screen_id": None,
                "screen_scale": "genome_wide",
                "screen_format": "pooled",
                "experimental_setup": "synthetic drug exposure",
                "selection_strategy": (
                    "positive_selection"
                    if study_index % 2 == 0
                    else "competitive_growth"
                ),
                "library_type": "lentiviral",
                "library_methodology": "synthetic",
                "enzyme": "synthetic Cas9",
                "vector_architecture": "synthetic single-vector",
                "effector_components": None,
                "guide_target_region": "coding sequence",
                "library_size_guides": len(gene_symbols) * 4,
                "target_gene_count": len(gene_symbols),
                "guides_per_gene_median": 4.0,
                "nontargeting_guide_count": 20,
                "library_moi": 0.25 + 0.02 * (study_index % 3),
                "effector_moi": None,
                "coverage_transduction": 450.0 + 25.0 * study_index,
                "coverage_selection": (
                    None if study_index == 3 else 400.0 + 20.0 * study_index
                ),
                "coverage_harvest": 350.0 + 15.0 * study_index,
                "infection_replicate_count": 2,
                "replicate_unit": "independent_infection",
                "antibiotic_selection_days": 7.0,
                "editing_maturation_days": 7.0,
                "plasmid_reads_per_guide": 200.0,
                "plasmid_zero_guide_fraction": 0.0,
                "plasmid_skew_ratio": 4.0,
                "screen_reads_per_guide": 500.0,
                "gdna_fraction_amplified": 1.0,
                "pcr_cycle_count": 22,
                "cnv_amplification_risk_assessed": study_index % 2 == 0,
                "analysis_method": "synthetic median guide LFC",
                "normalization_method": "median_ratio",
                "source_locator": "software fixture",
                "notes": "Synthetic; design values test missingness and joins.",
            }
        )
        control_type = "vehicle" if study_index % 2 == 0 else "untreated"
        contrasts.append(
            {
                "screen_id": screen_id,
                "contrast_id": contrast_id,
                "contrast_name": f"{drug_name} versus control",
                "treatment_name": drug_name,
                "treatment_id": None,
                "drug_class": "synthetic",
                "treatment_dose": 1.0,
                "treatment_unit": "arbitrary_unit",
                "dose_basis": "nominal",
                "control_type": control_type,
                "comparator_name": (
                    "synthetic vehicle" if control_type == "vehicle" else None
                ),
                "exposure_schedule": "continuous",
                "exposure_days": 14.0,
                "recovery_days": 0.0 if study_index % 3 else None,
                "endpoint_timepoint_days": 14.0,
                "phenotype_endpoint": "relative abundance",
                "intended_direction": "resistance",
                "vehicle_control_present": control_type == "vehicle",
                "baseline_control_present": study_index % 3 == 0,
                "same_infection_split": True,
                "matched_control": True,
                "positive_control_description": None,
                "negative_control_description": "synthetic non-targeting controls",
                "source_locator": "software fixture",
                "notes": "Synthetic; not biological evidence.",
            }
        )

        sample_ids: list[str] = []
        for role in ("control", "treatment"):
            for replicate in (1, 2):
                sample_id = f"{screen_id}_{role}_{replicate}"
                sample_ids.append(sample_id)
                sample_rows.append(
                    {
                        "sample_id": sample_id,
                        "screen_id": screen_id,
                        "contrast_id": contrast_id,
                        "condition_role": role,
                        "replicate": replicate,
                        "biological_replicate_id": f"BIOREP-{replicate}",
                        "infection_replicate_id": f"INFECTION-{replicate}",
                        "technical_replicate_id": None,
                        "pair_id": f"PAIR-{replicate}",
                        "replicate_unit": "independent_infection",
                        "timepoint_days": 14.0,
                        "timepoint_reference": "start_of_drug_exposure",
                        "treatment_name": drug_name if role == "treatment" else None,
                        "treatment_dose": 1.0 if role == "treatment" else None,
                        "treatment_unit": (
                            "arbitrary_unit" if role == "treatment" else None
                        ),
                        "batch": "synthetic_batch_1",
                        "library_prep_batch": "synthetic_prep_1",
                        "sequencing_batch": "synthetic_seq_1",
                        "cell_count": len(gene_symbols) * 4 * 500,
                        "coverage_per_guide": 500.0,
                        "fastq_accession": None,
                        "notes": "Synthetic; not biological evidence.",
                    }
                )

        records: list[dict[str, object]] = []
        for gene_index, gene_symbol in enumerate(gene_symbols):
            if gene_index < 6:
                true_lfc = 1.6
            elif gene_index < 12:
                true_lfc = -1.6
            elif gene_index < 17:
                true_lfc = {
                    12: 0.9,
                    13: -0.9,
                    14: 0.7,
                    15: -0.7,
                    16: 0.5,
                }[gene_index]
            else:
                true_lfc = float(RNG.normal(0, 0.08))
            phenotype_direction = "resistance" if true_lfc >= 0 else "sensitization"
            for guide_index in range(4):
                baseline = float(RNG.uniform(450, 1_200))
                row: dict[str, object] = {
                    "sgrna_id": (f"{screen_id}_{gene_symbol}_sg{guide_index + 1}"),
                    "gene_symbol": gene_symbol,
                }
                for role in ("control", "treatment"):
                    for replicate in (1, 2):
                        sample_id = f"{screen_id}_{role}_{replicate}"
                        role_effect = true_lfc if role == "treatment" else 0.0
                        guide_noise = RNG.normal(0, 0.12)
                        replicate_noise = RNG.lognormal(0, 0.08)
                        expected = baseline * (2 ** (role_effect + guide_noise))
                        row[sample_id] = int(
                            RNG.poisson(max(2.0, expected * replicate_noise))
                        )
                records.append(row)

            label = label_for_gene(gene_index, study_index)
            testing_status = (
                "tested"
                if label != "U"
                else ("not_tested" if gene_index % 4 == 0 else "unknown")
            )
            candidate_rows.append(
                {
                    "study_id": study_id,
                    "screen_id": screen_id,
                    "contrast_id": contrast_id,
                    "gene_symbol": gene_symbol,
                    "phenotype_direction": phenotype_direction,
                    "label_code": label,
                    "testing_status": testing_status,
                }
            )
            if label != "U":
                positive = label in {"V2", "V3"}
                supportive = label == "V1"
                discordant = label == "D"
                technical = label == "T"
                ambiguous = label == "A"
                validation_rows.append(
                    {
                        "event_id": f"{screen_id}-{gene_symbol}-E1",
                        "study_id": study_id,
                        "screen_id": screen_id,
                        "contrast_id": contrast_id,
                        "gene_symbol": gene_symbol,
                        "drug_name": drug_name,
                        "cell_line": cell_line,
                        "perturbation_modality": "CRISPR_KO",
                        "phenotype_direction": phenotype_direction,
                        "label_code": label,
                        "testing_status": "tested",
                        "perturbation_confirmed": not technical,
                        "independent_reagent_count": (
                            1 if supportive else (None if technical else 2)
                        ),
                        "orthogonal_perturbation": False,
                        "appropriate_control": not technical,
                        "assay_adequate": not technical,
                        "phenotype_reproduced": (
                            True
                            if positive or supportive
                            else (None if ambiguous or technical else False)
                        ),
                        "opposite_direction_reproduced": (
                            True if discordant else (None if ambiguous else False)
                        ),
                        "rescue_performed": label == "V3",
                        "causal_reversal_performed": False,
                        "effect_size": (
                            1.0
                            if positive or supportive
                            else (None if ambiguous or technical else 0.0)
                        ),
                        "effect_metric": "synthetic_effect",
                        "p_value": (
                            0.01
                            if positive or supportive or discordant
                            else (None if ambiguous or technical else 0.7)
                        ),
                        "source_url": "https://example.org/synthetic",
                        "source_locator": "synthetic record",
                        "curator": "software_generator",
                        "adjudication_status": "synthetic",
                        "notes": "Synthetic; not biological evidence.",
                    }
                )

        count_blocks.append(pd.DataFrame(records))

    studies_frame = pd.DataFrame(studies)
    screens_frame = pd.DataFrame(screens)
    libraries_frame = pd.DataFrame(libraries)
    screen_designs_frame = pd.DataFrame(screen_designs)
    contrasts_frame = pd.DataFrame(contrasts)
    samples_frame = pd.DataFrame(sample_rows)
    candidates_frame = pd.DataFrame(candidate_rows)
    validation_frame = pd.DataFrame(validation_rows)

    # Count tables are screen-specific in real use. The synthetic example has a
    # single wide matrix with zeros for samples belonging to other screens.
    all_sample_ids = samples_frame["sample_id"].tolist()
    combined_counts = pd.concat(count_blocks, ignore_index=True, sort=False)
    combined_counts[all_sample_ids] = (
        combined_counts[all_sample_ids].fillna(0).astype(int)
    )

    feature_blocks: list[pd.DataFrame] = []
    for screen_id, design in samples_frame.groupby("screen_id", sort=False):
        sample_ids = design["sample_id"].tolist()
        screen_counts = combined_counts.loc[
            combined_counts["sgrna_id"].str.startswith(screen_id),
            ["sgrna_id", "gene_symbol", *sample_ids],
        ].copy()
        feature_blocks.append(featurize_count_table(screen_counts, design))
    features = pd.concat(feature_blocks, ignore_index=True)
    design_features = featurize_experimental_design(
        screens_frame,
        screen_designs_frame,
        contrasts_frame,
        samples_frame,
    )
    features = features.merge(
        design_features,
        on=["screen_id", "contrast_id"],
        how="left",
        validate="many_to_one",
    )
    labeled_features = features.merge(
        candidates_frame,
        on=[
            "study_id",
            "screen_id",
            "contrast_id",
            "gene_symbol",
        ],
        how="left",
        validate="one_to_one",
    )

    studies_frame.to_csv(OUT / "studies.csv", index=False)
    screens_frame.to_csv(OUT / "screens.csv", index=False)
    libraries_frame.to_csv(OUT / "libraries.csv", index=False)
    screen_designs_frame.to_csv(OUT / "screen_designs.csv", index=False)
    contrasts_frame.to_csv(OUT / "contrasts.csv", index=False)
    samples_frame.to_csv(OUT / "sample_sheet.csv", index=False)
    combined_counts.to_csv(OUT / "guide_counts.csv", index=False)
    first_screen_id = screens_frame["screen_id"].iloc[0]
    first_screen_samples = samples_frame.loc[
        samples_frame["screen_id"] == first_screen_id
    ].copy()
    first_screen_sample_ids = first_screen_samples["sample_id"].tolist()
    first_screen_counts = combined_counts.loc[
        combined_counts["sgrna_id"].str.startswith(first_screen_id),
        ["sgrna_id", "gene_symbol", *first_screen_sample_ids],
    ].copy()
    first_screen_samples.to_csv(OUT / "sample_sheet_screen_01.csv", index=False)
    first_screen_counts.to_csv(OUT / "guide_counts_screen_01.csv", index=False)
    candidates_frame.to_csv(OUT / "candidate_status.csv", index=False)
    validation_frame.to_csv(OUT / "validation_events.csv", index=False)
    # Quantize derived floating-point features at serialization time so that
    # harmless BLAS/libm last-bit differences do not change the checked-in
    # software fixture across supported Python runner images.
    labeled_features.to_csv(
        OUT / "labeled_gene_features.csv",
        index=False,
        float_format="%.10g",
    )
    print(
        {
            "studies": len(studies_frame),
            "screens": len(screens_frame),
            "contrasts": len(contrasts_frame),
            "guides": len(combined_counts),
            "gene_rows": len(labeled_features),
            "validation_events": len(validation_frame),
        }
    )


if __name__ == "__main__":
    main()
