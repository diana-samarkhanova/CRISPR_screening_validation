# ORCS 2.0.18 curation batch 001

This batch freezes queue ranks 1-10 before any outcome review. The ten screens
come from ten distinct publication families and were selected without hit,
score, or validation columns.

`selection.tsv` is the immutable outcome-blind selection. `reviews.tsv` is a
separate downstream full-text and supplement review. A partial independent
review of ranks 6-10 is stored in `reviews_curator_2_partial.tsv`; its derived
gene-level comparison and checksum manifest are separate again. These layers
cannot alter the original selection.

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
- Independent second review is complete only for ranks 6-10. Seven compared
  gene assessments agree provisionally; ranks 1-5 and every final human
  adjudication remain pending.
- The same five-screen comparison reports disagreements field by field across
  accessions, data families, supplement completeness, sample maps, replication,
  quantitative assets, rights, source locators, and blocker codes. Gene-level
  agreement is not whole-record consensus.
- All ten screens remain `metadata_only`; none is `benchmark_ready`.
- Candidate validation grades are single-curator extraction results, not final
  labels: one screen has a candidate `V3`, four have candidate `V2`, four have
  candidate `V1`, and one has only nonqualifying mechanistic follow-up.

Candidate genes and validation summaries live only in review-layer files.
They are outcomes and must never be joined into pre-follow-up model features.
Even exact reviewer agreement is not a released label. ORCS hit calls and
author ranks remain discovery evidence, never validation labels.

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
