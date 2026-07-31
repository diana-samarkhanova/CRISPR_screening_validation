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
