import pandas as pd

from crispr_evidencerank.contracts import GeneScoreRecord, validate_records
from crispr_evidencerank.io import normalize_mageck_gene_summary


def test_mageck_adapter_preserves_analysis_tail_and_validates():
    source = pd.DataFrame(
        {
            "id": ["GENE1"],
            "pos|score": [0.01],
            "pos|fdr": [0.02],
            "neg|score": [0.2],
            "neg|fdr": [0.3],
        }
    )
    normalized = normalize_mageck_gene_summary(source, "SC1", "C1")
    valid, errors = validate_records(normalized, GeneScoreRecord)
    assert len(valid) == 2
    assert errors.empty
    assert set(valid["analysis_tail"]) == {"mageck_pos", "mageck_neg"}
