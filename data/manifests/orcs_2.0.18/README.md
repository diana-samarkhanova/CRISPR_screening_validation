# BioGRID ORCS 2.0.18 intake manifest

This directory contains the Git-safe, outcome-blind products of the pinned
human ORCS 2.0.18 index intake. It does not contain the downloaded archive,
per-screen gene-score files, raw counts, FASTQ files, validation labels, or
private study data.

Regenerate the full ignored working output with:

```bash
crispr-evidencerank prepare-orcs-release \
  --release-registry config/orcs_releases.yaml \
  --release 2.0.18 \
  --retrieved-date 2026-07-31 \
  --archive data/external/orcs_2.0.18/BIOGRID-ORCS-ALL-homo_sapiens-2.0.18.screens.tar.gz \
  --output-dir data/processed/orcs_2.0.18/release_intake
```

The checked-in files are:

- `archive_manifest.json`: portable archive provenance and the explicitly
  project-computed SHA-256;
- `index_manifest.json`: index checksum, archive inventory, and cross-checked
  screen-ID digest;
- `release_summary.json` and `triage_summary.json`: observed counts and the
  explicit statement that index intake is not a trainable corpus;
- `curation_queue.tsv`: deterministic source-diverse curation order;
- `candidate_screen_ids.txt`: release-qualified internal screen IDs in queue
  order;
- `derived_manifest.json`: record counts and checksums for the two derived
  queue artifacts.
- `curation_batches/batch_001/`: the frozen outcome-blind ranks 1-10 selection,
  its separate full-text review, a frozen ranks 6-10 dual-review checkpoint,
  a checksum-bound completion-progress addendum for ranks 1, 3, 4, and 5, and
  the authenticated eleven-file full dual-review bundle.

ORCS `HIT` values are deliberately absent from queue prioritization and are
never interpreted as validation outcomes. The ORCS MIT license covers the
ORCS-distributed files only; it does not establish redistribution rights for
authors' upstream sequencing or count-level data.

Batch 001 has primary and independent second-review records for all ten
screens. Its 20 gene-level comparison rows comprise 17 provisional agreements,
one evidence-level disagreement, and two single-curator observations. All ten
screens remain `metadata_only`; formal human event adjudication is required
before any candidate grade can become a label.
