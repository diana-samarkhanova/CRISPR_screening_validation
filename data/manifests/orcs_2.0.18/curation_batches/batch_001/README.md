# ORCS 2.0.18 curation batch 001

This batch freezes queue ranks 1-10 before any outcome review. The ten screens
come from ten distinct publication families and were selected without hit,
score, or validation columns.

`selection.tsv` is the immutable outcome-blind selection. `reviews.tsv` is a
separate downstream full-text and supplement review. Their SHA-256 checksums
are recorded independently so that later outcome corrections cannot alter the
original selection.

## Review result

- 10/10 primary full texts were reviewed; nine supplement sets were reviewed
  completely and one remains partial.
- Nine screens expose author-derived gene or guide scores but not a public
  sample-level count matrix.
- One screen (PMID 30449619) has public raw sequencing reads at `SRP158611`;
  those reads are not yet ingested or mapped to a verified sample sheet.
- All ten screens remain `metadata_only`; none is `benchmark_ready`.
- Candidate validation grades are single-curator extraction results, not final
  labels: one screen has a candidate `V3`, four have candidate `V2`, four have
  candidate `V1`, and one has only nonqualifying mechanistic follow-up.

Candidate genes and validation summaries live only in `reviews.tsv`. They are
outcomes and must never be joined into pre-follow-up model features. ORCS hit
calls and author ranks remain discovery evidence, never validation labels.

## Remaining blockers

The recurring blockers are public count-level signal, an explicit raw-data
family and rights decision, and independent validation-event adjudication. One
raw-read family is resolved; the other rows identify only quantitative-asset
families and deliberately leave `raw_data_family_id` empty.
Screens with incomplete replicate/sample mapping also retain the comparator
and sample-map blocker. Raw archives, FASTQ files, source workbooks, and private
screen data are not committed here.
