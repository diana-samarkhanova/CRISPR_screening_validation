from io import StringIO

import pandas as pd
import pytest

from crispr_evidencerank.orcs import (
    normalize_orcs_header,
    parse_orcs_index,
    parse_orcs_screen_scores,
)

INDEX = (
    "#SCREEN ID\tSOURCE ID\tSOURCE TYPE\tAUTHOR\tSCREEN NAME\t"
    "ANALYSIS\tSCREEN TYPE\tSCREEN FORMAT\tEXPERIMENTAL SETUP\t"
    "DURATION\tCONDITION NAME\tCONDITION DOSAGE\tMOI\tLIBRARY\t"
    "LIBRARY TYPE\tLIBRARY METHODOLOGY\tENZYME\tCELL LINE\t"
    "PHENOTYPE\tSCORE.1 TYPE\tNOTES\tSCREEN RATIONALE\t"
    "DATASET ID\tFUTURE FIELD\n"
    "634\t30051818\tpubmed\tViswanatha R (2018)\t"
    "2-PMID30051818\tMAGeCK-MLE\tPositive and Negative Selection\t"
    "Pool\tDrug Exposure\t30 Days\tTrametinib\t50.0 nM\t~ 0.3\t"
    "Targeted sub-library\tCRISPRn\tKnockout\tCas9\tS2R+\t"
    "response to chemicals\tZ-score\t-\t"
    "Increased/decreased drug resistance\tDS-42\tkept\n"
)


def test_orcs_index_preserves_raw_values_and_is_conservative():
    parsed = parse_orcs_index(
        StringIO(INDEX),
        release="2.0.18",
        retrieved_date="2026-07-30",
    )

    assert parsed.header_map["#SCREEN ID"] == "screen_id"
    assert parsed.header_map["SCORE.1 TYPE"] == "score_1_type"
    assert parsed.raw_index.loc[0, "notes"] == "-"
    assert pd.isna(parsed.normalized_index.loc[0, "notes"])
    assert parsed.raw_index.loc[0, "future_field"] == "kept"

    study = parsed.studies.iloc[0]
    assert study["source_id"] == "30051818"
    assert study["pubmed_id"] == "30051818"
    assert pd.isna(study["data_license"])

    screen = parsed.screens.iloc[0]
    assert screen["screen_id"] == "orcs:2.0.18:screen:634"
    assert screen["source_family_id"] == ("orcs:source-family:pubmed:30051818")
    assert pd.isna(screen["raw_data_family_id"])
    assert screen["perturbation_modality"] == "CRISPR_KO"
    assert screen["treatment_dose"] == 50.0
    assert screen["treatment_unit"] == "nM"
    assert screen["duration_days"] == 30.0
    assert screen["intended_direction"] == "unknown"

    design = parsed.screen_designs.iloc[0]
    assert design["selection_strategy"] == "bidirectional_selection"
    assert pd.isna(design["library_moi"])

    contrast = parsed.contrasts.iloc[0]
    assert contrast["control_type"] == "unknown"
    assert pd.isna(contrast["comparator_name"])
    assert contrast["intended_direction"] == "unknown"

    external = parsed.external_screen_maps.iloc[0]
    assert external["mapping_id"] == ("orcs:2.0.18:external-screen:634")
    assert external["source_version"] == "2.0.18"
    assert external["external_dataset_id"] == "DS-42"
    assert external["external_screen_id"] == "634"
    assert external["screen_id"] == "orcs:2.0.18:screen:634"


def test_orcs_hit_maps_only_to_author_hit_and_scores_stay_heterogeneous():
    source = StringIO(
        "#SCREEN_ID\tIDENTIFIER_ID\tIDENTIFIER_TYPE\t"
        "OFFICIAL_SYMBOL\tSCORE.1\tSCORE.2\tHIT\n"
        "634\t1\tgene\tGENE1\t1.25\t0.01\tYES\n"
        "634\t2\tgene\tGENE2\t-0.5\t0.20\tNO\n"
        "634\t3\tgene\tGENE3\t0\t0.90\tN/A\n"
    )
    metadata = {
        "SCREEN_ID": "634",
        "ANALYSIS": "MAGeCK-MLE",
        "SCORE.1_TYPE": "Z-score",
        "SCORE.2_TYPE": "FDR",
    }
    parsed = parse_orcs_screen_scores(
        source,
        release="2.0.18",
        index_metadata=metadata,
        source_file="screen_634.tab.txt",
    )

    assert parsed.issues.empty
    assert len(parsed.gene_scores) == 6
    assert set(parsed.gene_scores["analysis_tail"]) == {
        "orcs_score_1",
        "orcs_score_2",
    }
    assert set(parsed.gene_scores["score_type"]) == {"Z-score", "FDR"}
    assert set(parsed.gene_scores["direction"]) == {"unknown"}
    assert "label_code" not in parsed.gene_scores
    assert "testing_status" not in parsed.gene_scores

    by_gene = parsed.gene_scores.groupby("gene_symbol")["author_hit"]
    assert set(by_gene.get_group("GENE1")) == {True}
    assert set(by_gene.get_group("GENE2")) == {False}
    assert all(pd.isna(value) for value in by_gene.get_group("GENE3"))
    assert parsed.raw_scores["hit"].tolist() == ["YES", "NO", "N/A"]


def test_shared_external_dataset_does_not_infer_raw_family_or_replicates():
    source = StringIO(
        "#SCREEN ID\tSOURCE ID\tSOURCE TYPE\tDATASET ID\t"
        "LIBRARY TYPE\tLIBRARY METHODOLOGY\n"
        "95\t12345\tpubmed\t15\tCRISPRn\tKnockout\n"
        "96\t12345\tpubmed\t15\tCRISPRn\tKnockout\n"
    )
    parsed = parse_orcs_index(
        source,
        release="2.0.18",
        retrieved_date="2026-07-30",
    )

    assert parsed.external_screen_maps["external_dataset_id"].tolist() == [
        "15",
        "15",
    ]
    assert parsed.external_screen_maps["replicate_number"].isna().all()
    assert parsed.screens["raw_data_family_id"].isna().all()
    assert parsed.screens["source_family_id"].nunique() == 1


def test_missing_source_id_uses_provisional_study_without_source_family():
    source = StringIO(
        "#SCREEN ID\tSOURCE ID\tSOURCE TYPE\tCELL LINE\t"
        "LIBRARY TYPE\tLIBRARY METHODOLOGY\n"
        "97\t-\tpubmed\tA375\tCRISPRn\tKnockout\n"
    )
    parsed = parse_orcs_index(
        source,
        release="2.0.18",
        retrieved_date="2026-07-30",
    )

    study = parsed.studies.iloc[0]
    screen = parsed.screens.iloc[0]
    assert study["study_id"] == ("orcs:2.0.18:study:provisional-screen:97")
    assert pd.isna(study["source_id"])
    assert pd.isna(study["pubmed_id"])
    assert "identity is provisional" in study["notes"]
    assert pd.isna(screen["source_family_id"])


def test_missing_symbol_is_not_silently_replaced_by_entrez_id():
    source = StringIO(
        "#SCREEN_ID\tIDENTIFIER_ID\tOFFICIAL_SYMBOL\tSCORE.1\tHIT\n"
        "634\t1234\t-\t2.0\tYES\n"
    )
    parsed = parse_orcs_screen_scores(
        source,
        release="2.0.18",
    )
    assert parsed.gene_scores.empty
    assert len(parsed.issues) == 1
    assert "not silently substituted" in parsed.issues.iloc[0]["error"]
    assert parsed.raw_scores.loc[0, "official_symbol"] == "-"
    assert pd.isna(parsed.normalized_scores.loc[0, "official_symbol"])


def test_dynamic_header_collisions_are_rejected():
    frame = pd.DataFrame(
        [["1", "1"]],
        columns=["#SCREEN.ID", "SCREEN ID"],
    )
    with pytest.raises(ValueError, match="collide"):
        parse_orcs_index(
            frame,
            release="2.0.18",
            retrieved_date="2026-07-30",
        )


def test_header_alias_tracks_rest_and_bulk_methodology_names():
    assert normalize_orcs_header("METHODOLOGY") == ("library_methodology")
    assert normalize_orcs_header("DATASET ID") == "external_dataset_id"
