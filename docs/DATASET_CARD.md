# Dataset card — development registry

## Intended use

The future dataset supports post-screen ranking of genes from human pooled
CRISPR-Cas9 knockout drug-response screens by relative likelihood of
orthogonal experimental reproduction.

## Current state

Version `0.2.0.dev0` contains schemas, curation rules, an intake manifest, and
synthetic fixtures. It does not contain a completed biological training
dataset, a pretrained model, or calibrated probabilities.

The first ten ORCS queue records have completed a single-curator full-text
pilot review. Nine expose only author-derived score tables and one exposes raw
sequencing reads, but none currently satisfies the count-level, rights,
sample-map, and independent label-adjudication gates together. The pilot adds
no `benchmark_ready` rows.

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
requires a version, checksum, source locator, rights record, and transformation
provenance.

## Release gate

A biological dataset release requires manual adjudication of at least six
independent public studies, source/raw-data-family deduplication, successful
and failed validation events with exact locators, grouped baselines, a
data-rights audit, and institutional approval for any unpublished component.
