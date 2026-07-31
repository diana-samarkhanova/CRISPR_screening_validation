# Changelog

All notable project changes are recorded here. The project follows semantic
versioning once public releases begin; development checkpoints use a `.devN`
suffix and are not claims of a trained biological model.

## Unreleased — ORCS 2.0.18 intake checkpoint

- Pinned the human-screen intake to BioGRID ORCS `2.0.18`, compiled on
  `2025-09-09` and publicly available on `2025-10-07`, with a separate actual
  retrieval timestamp.
- Recorded the downloaded archive's SHA-256 as a project-computed integrity
  value, not a checksum published by BioGRID.
- Limited ORCS MIT licensing to ORCS-distributed files; linked publisher,
  GEO/SRA, FASTQ, count-matrix, and other upstream assets retain independent
  rights requirements.
- Added an outcome-blind curation queue that excludes ORCS `HIT`, author score
  magnitude, validation outcomes, and future evidence from ordering.
- Added atomic `prepare-orcs-release` acquisition with a pinned archive/index
  checksum contract, bounded tar inventory, safe index-only extraction, and
  exact cross-checking of all index and screen-member IDs.
- Made benchmark readiness independently fact-derived from linked design,
  quantitative signal, rights, family, and reviewed validation records; a
  self-declared check or `single_curator` draft cannot open the gate.
- Added runtime/JSON-Schema parity checks for intake and data-asset invariants,
  plus portable checksums for the checked-in derived queue.
- Recorded the observed human-index intake: 1,952 total screens, 278
  `exclude`, 1,674 `metadata_only`, 435 confirmed-scope candidates, 1,239
  manual-review candidates, and zero `benchmark_ready`.
- Kept the checkpoint explicitly non-trainable: full-text design curation,
  comparator/sample reconstruction, count-level evidence, data-rights review,
  and independent validation-event adjudication remain required.

## 0.2.0.dev0 — 2026-07-30

- Added the normalized `study → screen → contrast → sample` hierarchy.
- Added structured screen-design, control, treatment, replicate, coverage,
  endpoint, and missingness fields.
- Added version-pinned BioGRID ORCS index and per-screen score import.
- Preserved ORCS `HIT` as `author_hit`, never as a validation label.
- Added source/raw-data-family grouping to prevent train/test leakage.
- Corrected ORCS provenance handling: source families are release-independent,
  raw-data families remain unresolved until curated, external dataset IDs are
  retained, and source-data licenses are never inferred from the software
  license.
- Added deterministic ORCS index triage with screen-level intake records,
  rule-level audit records, candidate IDs, and machine-readable summaries.
- Made policy-v2 readiness fully derived: every canonical benchmark rule must
  be present and pass; missing, duplicated, mismatched, or self-declared checks
  cannot produce `benchmark_ready`.
- Kept `Toxin Exposure` and non-drug `CONDITION NAME` values unresolved until
  curation, tracked incomplete ORCS score sets explicitly, and stopped
  fabricating source families when `SOURCE ID` is absent.
- Added JSON-Schema/runtime parity for intake-state invariants and registry
  integrity checks for intake/check foreign keys and readiness support.
- Added feature profiles for `screen_only`, `screen_plus_design`, and
  `context_aware` evaluation.
- Added repository safety checks and continuous integration for private
  GitHub development.

## 0.1.0.dev0 — 2026-07-30

- Added validation labels `V3/V2/V1/F0/D/A/T/U`.
- Added count-table validation, median-ratio normalization, guide-aware gene
  features, and a MAGeCK summary adapter.
- Added selection-aware logistic and histogram-gradient-boosting baselines.
- Added study-grouped evaluation, ranking metrics, synthetic fixtures, and
  automated tests.
