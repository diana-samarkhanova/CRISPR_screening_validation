# Dataset card — development registry

## Intended use

The future dataset supports post-screen ranking of genes from human pooled
CRISPR-Cas9 knockout drug-response screens by relative likelihood of
orthogonal experimental reproduction.

## Current state

Version `0.5.0.dev0` contains schemas, curation rules, an intake manifest,
synthetic fixtures, a screen-signal report, and independent report-only immune
and clinical-context engines. It does not contain a completed biological
training dataset, a pretrained model, or calibrated probabilities. Clinical
registry context is treatment-by-cancer metadata, not gene-level evidence or a
validation label.

Post-checkpoint development includes a separate frozen ClinicalTrials.gov API
snapshot intake. Its search queries are recall-oriented, and its derived rows
are explicitly unreviewed study-level treatment/condition co-mentions. They do
not constitute clinical evidence and cannot enter the clinical summarizer,
features, ranking, validation-success model, or labels.

The first ten ORCS queue records have completed primary and independent second
full-text review. Eight expose only author-derived score tables, one exposes a
public sgRNA count matrix, and one exposes raw amplicon reads. The authenticated
full comparison contains 20 gene-level rows: 17 provisional agreements, one
evidence-level disagreement, and two single-curator observations. All require
human adjudication; no adjudicated or released label has been created. No screen
currently satisfies the count-level, rights, sample-map, and label-adjudication
gates together. The pilot adds no `benchmark_ready` rows.

## Unit of prediction

`gene × screen × contrast × direction`, linked through the normalized
`study → screen → contrast → sample` hierarchy.

## Labels

Primary positive labels are `V2/V3`; primary negative labels are `F0/D`.
`V1/A/T/U` remain auxiliary or missing. A gene not mentioned in a paper's
validation section is not a negative.

## Sources

BioGRID ORCS is used for screen discovery and author-reported screen scores.
Primary articles, supplements, and raw-data repositories are used to verify
design and targeted validation. ORCS hit calls are never treated as
orthogonal-validation labels.

ClinicalTrials.gov is used only through checksum-bound frozen snapshots and a
separate curator-reviewed treatment-by-cancer evidence layer. Registry search
presence, phase, status, and aggregate-results availability do not establish
efficacy or support a CRISPR gene hit.

## Known biases

- authors preferentially validate statistically strong or biologically
  familiar hits;
- failed validations are underreported;
- reporting quality and available raw data differ by study;
- genes, drugs, cell models, and libraries are unevenly represented;
- later literature and database snapshots can create temporal leakage.

The benchmark therefore uses grouped and temporal evaluation, a separate
selection model, explicit missingness, and sensitivity analyses rather than
treating untested genes as negatives.

## Data governance

Git contains manifests and redistributable derived records only. Raw
sequencing files, private workbooks, copied supplements, and the held-out
unpublished case study remain outside Git history. Every external asset
requires a version, source locator, rights record, and transformation
provenance. Retrieved byte assets also require checksum and size; accession-only
pointers are explicitly marked not retrieved and carry no invented checksum.
Real ClinicalTrials.gov API pages and complete snapshots remain outside Git;
only clearly labeled synthetic registry fixtures may be committed. Snapshot
completeness is scoped to the exact recorded query and token traversal, and a
stable before/after version envelope is not transactional isolation. Reuse must
preserve ClinicalTrials.gov attribution, registry processing date, and the
project's dated modifications under the current source terms.

## Release gate

A biological dataset release requires manual adjudication of at least six
independent public studies, source/raw-data-family deduplication, successful
and failed validation events with exact locators, grouped baselines, a
data-rights audit, and institutional approval for any unpublished component.
