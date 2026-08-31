# ORCS 2.0.18 curation batch 001

This batch freezes queue ranks 1-10 before any outcome review. The ten screens
come from ten distinct publication families and were selected without hit,
score, or validation columns.

`selection.tsv` is the immutable outcome-blind selection. `reviews.tsv` is a
separate downstream full-text and supplement review. A partial independent
review of ranks 6-10 is stored in `reviews_curator_2_partial.tsv`; its derived
gene-level comparison and checksum manifest are separate again. Independently
reviewed ranks 1, 3, 4, and 5 are frozen in
`reviews_curator_2_completion_progress.tsv` with an exact-derived progress
manifest. `reviews_curator_2_completion.tsv` adds the independently reviewed
rank 2 without changing the progress rows. The authenticated full second
review, comparison, and manifest are stored in `reviews_curator_2.tsv`,
`review_comparison.tsv`, and `dual_review_manifest.json`. These layers cannot
alter the original selection or release labels.

## Review result

- 10/10 primary full texts were reviewed; nine supplement sets were reviewed
  completely and one remains partial.
- Eight screens expose author-derived gene or guide scores but not a public
  sample-level count matrix.
- The Findlay screen (PMID 30154076) exposes a public 91,320-row sgRNA count
  matrix in Supplementary Dataset EV3. Its six count columns encode two
  replicates per condition, while the methods describe triplicates, so the
  sample map remains unresolved and the workbook stays outside Git.
- The Shifrut study (PMID 30449619) has 24 public amplicon runs at
  `SRP158611`. Eight runs cover the candidate CGS-21680 contrast across two
  donors and dividing/nondividing bins. Four vehicle labels require
  article-supported interpretation. The map is checked against a clean,
  checksum-bound 24-run ENA inventory containing verbatim sample aliases and
  repository library/instrument fields, plus a separate curated contrast-scope
  table; it remains conditional and no FASTQ has been ingested.
- Independent second review is complete for all ten screens. The full
  comparison contains 20 gene-level rows: 17 provisional agreements, one
  evidence-level disagreement, and two single-curator observations. Every row
  still requires human adjudication.
- The full comparison reports disagreements field by field across
  accessions, data families, supplement completeness, sample maps, replication,
  quantitative assets, rights, source locators, and blocker codes. Gene-level
  agreement is not whole-record consensus.
- All ten screens remain `metadata_only`; none is `benchmark_ready`.
- Candidate validation grades are single-curator extraction results, not final
  labels: one screen has a candidate `V3`, four have candidate `V2`, four have
  candidate `V1`, and one has only nonqualifying mechanistic follow-up.

Distinct IDs, curators, and checksum lineages prevent silent stream replacement
but cannot prove that reviewers were cognitively independent. The pilot's
blinding is a documented process assertion and remains subject to named human
adjudication.

Candidate genes and validation summaries live only in review-layer files.
They are outcomes and must never be joined into pre-follow-up model features.
Even exact reviewer agreement is not a released label. ORCS hit calls and
author ranks remain discovery evidence, never validation labels.

## Human adjudication packet

`adjudication/v1/` is an unsigned, checksum-bound packet derived from the exact
canonical bytes of `dual_review_manifest.json` and all four declared review
inputs. It contains 20 immutable evidence cards plus blank decision and event
worksheets. It contains no final disposition, released label, signed human
decision, or benchmark-ready row. A later human release must cover all 20
packet items exactly once and remain a separate checksum-pinned artifact.

Review tables are checksum-bound evidence snapshots. Later accession, rights,
or sample-map audits are stored separately rather than silently rewriting a
curator's earlier observation.

## Remaining blockers

The recurring blockers are public count-level signal, an explicit raw-data
family and rights decision, and completed independent validation-event
adjudication. Two raw-data families are now identified, but neither has passed
the full ingestion, QC, sample-map, and adjudication gate. Screens with
incomplete replicate/sample mapping retain the comparator and sample-map
blocker. Raw archives, FASTQ files, source workbooks, and private screen data
are not committed here.
