# Repository and release policy

## Stage 1 — private development repository

GitHub development starts with `0.2.0.dev0`. The private repository contains:

- source code, configuration, schemas, tests, and CI;
- documentation and scientific decision records;
- synthetic examples;
- published-study accession manifests and checksums;
- small derived records only when redistribution is explicitly permitted.

The private repository does not contain the held-out unpublished
drug-response screen. Its biological context, anchor genes, counts, and
results remain outside the development corpus and Git history. It is evaluated
only after the KO model, feature pipeline, and candidate-selection rules are
frozen. Non-KO branches remain separate.

Private GitHub visibility is not a substitute for data governance. Third-party
or unpublished material is kept outside Git history even when the repository
is private.

## ORCS 2.0.18 intake checkpoint

The reproducible intake is pinned to the ORCS `2.0.18` human archive, compiled
on `2025-09-09` and publicly available on `2025-10-07`. Git may contain the
release registry, source URL, project-computed SHA-256, retrieval timestamp,
small normalized manifests, audit summaries, and an outcome-blind curation
queue. The 717.79 MB source archive, extracted per-screen files, and other
large downloaded assets remain outside Git history.

The project-computed SHA-256 verifies the bytes retrieved by this pipeline; it
must not be described as a checksum published by BioGRID. The retrieval date
must remain distinct from the compiled and public-availability dates.

MIT terms for ORCS-distributed files do not confer redistribution rights for
linked publisher supplements, GEO/SRA objects, FASTQ files, author count
matrices, or other upstream datasets. Those assets require independent
rights records and remain excluded unless redistribution is affirmatively
verified.

The observed index triage contains 1,952 records: 278 `exclude`, 1,674
`metadata_only`, 435 confirmed-scope curation candidates, 1,239 manual-review
candidates, and zero `benchmark_ready`. Queue order is outcome-blind and must
not use ORCS `HIT`, author scores, validation outcomes, or future evidence.
These records are a curation intake, not a trainable corpus.

## Stage 2 — pilot data checkpoint

Create the next development tag only after at least six public human
CRISPR-KO drug-response studies have:

- verified study, screen, contrast, and sample maps;
- exact treatment and comparator conditions with explicit missingness;
- source/raw-data-family identifiers;
- independently adjudicated validation events with exact source locators;
- baseline MAGeCK/drugZ or effect-size outputs;
- grouped benchmark results for `screen_only` and `screen_plus_design`.

Large source files remain in their original repositories or controlled local
storage. GitHub receives an accession manifest, retrieval script, checksum,
license record, and reproducible transformation instructions.

## Stage 3 — frozen model and private case study

Before the held-out private screen is evaluated:

1. Freeze the eligible study set and outer split assignments.
2. Freeze feature definitions, preprocessing, candidate models, and selection
   rules.
3. Record the code commit and environment lock.
4. Run the private case study once as the prospective benchmark.

Only redacted, authorized derived results may later enter a manuscript branch.

## Stage 4 — public release

Public visibility requires PI/institutional approval, authorship and CRediT
agreement, intellectual-property review, and a row-level data-rights audit.
The public release contains no private counts or copied supplementary tables.
It is tagged, archived in Zenodo, and linked from `CITATION.cff`.

## Version and branch conventions

- `main`: continuously reproducible private development branch.
- short-lived branches: one curation, feature, bug, or documentation change.
- development tags: `v0.2.0-dev`, `v0.3.0-dev`, and later checkpoints.
- first public software release: assigned only after the public-release gate.

Changing a label definition, prediction unit, leakage rule, or data contract is
a scientific change and must be recorded in `CHANGELOG.md`.
