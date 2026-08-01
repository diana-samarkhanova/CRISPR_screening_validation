# Data provenance policy

Every external datum must be traceable to:

- source name and version;
- stable accession or URL;
- retrieval date;
- license or terms link;
- transformation script and version;
- checksum for downloaded files;
- study, screen, sample, guide, gene, and validation-event identifiers.

The checked-in Git commit and `uv.lock` identify the code and environment that
generated a bundle. The bundle manifest checksum-binds its data inputs and
outputs but does not embed its own Git commit, which would be self-referential.
A standalone export must therefore travel with a release tag or recorded Git
tree identifier in its archival metadata.

## Normalized experimental levels

Provenance is attached to four distinct levels:

- `study`: publication, preprint, or author-defined research source;
- `screen`: shared cell model, CRISPR library/modality, delivery, and pooled
  experiment;
- `contrast`: one treatment-versus-comparator question;
- `sample`: one sequenced biological or technical observation.

The registry preserves both `source_family_id` and `raw_data_family_id`. A
source family links article/preprint versions, repository mirrors, and curated
database records. A raw-data family links every result derived from the same
experimental counts or author-supplied matrix. Alternative scoring methods do
not create independent training data.

## Repository policy

Commit:

- manifests;
- schemas;
- curation rules;
- download instructions;
- checksums;
- small synthetic examples;
- redistributable derived data with documented terms.

Do not commit by default:

- copied article text or figures;
- third-party matrices lacking redistribution permission;
- unpublished internal screen data;
- private annotation workbooks;
- credentials or signed download URLs.

## Source roles

`training_label` is restricted to documented targeted validation events.
DepMap/Project Score, tumor multi-omics, pathway resources, and literature are
features or contextual evidence, not ground-truth validation labels.

[BioGRID ORCS](https://orcs.thebiogrid.org/) is used for screen discovery,
structured metadata, and author-method gene scores. Record its release,
retrieval date, ORCS screen ID, original publication, scoring method, and
source/raw-family mapping. ORCS ranks and hit calls are screen evidence, not
orthogonal-validation labels. Versioned releases are available from the
[official ORCS release archive](https://downloads.thebiogrid.org/BioGRID-ORCS/Release-Archive/).

The current intake pins the human screen archive to ORCS `2.0.18`, compiled on
`2025-09-09` and publicly available on `2025-10-07`. Its retrieval date and UTC
retrieval timestamp are recorded separately; neither may substitute for the
release's availability date in temporal evaluation. The SHA-256 stored in the
manifest is computed by this project from the downloaded bytes. It is a local
integrity value, not a publisher-supplied checksum. For the observed
752,653,348-byte archive, that digest is
`39222a9650eed083edf193debe45eedc4aabc779ca04ea70107b6bd1efd9b8d7`.

The MIT terms recorded for this asset apply only to files distributed by
BioGRID ORCS. They must not be inherited by publisher supplements, SRA/GEO
objects, FASTQ files, author count matrices, or other upstream data linked from
an ORCS record. Each upstream asset requires its own rights holder, terms,
redistribution decision, and provenance row. Retrieved bytes require a local
or publisher checksum; an accession-only pointer must be marked not retrieved
and must not claim a checksum for bytes the project did not acquire.

Primary methods, supplements, raw-repository metadata, and author files verify
historical conditions. A library-design paper, including Joung et al., may
support library identity and design fields, but its original MOI, coverage,
selection, treatment, duration, or control protocol is never copied into a
later screen when that screen did not report those values.

## Temporal provenance

Every evidence row carries `available_date`. For temporal validation, a feature
is visible only if it existed before the test study's cutoff date. Current
database snapshots must not leak later knowledge into historical prediction.

Cross-screen recurrence and other source-derived features are reconstructed
inside each outer training fold after source/raw-family deduplication. The
held-out family and any evidence first available after the test cutoff are
excluded.

## ORCS intake and curation queue

The ORCS 2.0.18 human index contains 1,952 observed screen records. Automated
index triage yields 278 `exclude` and 1,674 `metadata_only` records. The
outcome-blind curation queue contains 435 confirmed-scope candidates and 1,239
manual-scope-review candidates; no index-only record is `benchmark_ready`.

Queue ordering may use only eligibility scope and metadata completeness. It
must not use ORCS `HIT`, author score magnitude, validation labels, or evidence
that became available after the screen. ORCS `HIT` remains an author-reported
screen call and can never supply a `V2`, `V3`, `F0`, or `D` validation event.
The queue is a worklist for full-text and data-rights curation, not a training
dataset.

The first frozen pilot batch covers queue ranks 1-10 from ten distinct source
families. Selection and review are stored separately with independent SHA-256
checksums. Full-text review found eight author-score-only screens, one public
sgRNA count matrix, and one screen with public raw reads (`SRP158611`). The SRA
study contains 24 amplicon runs; eight are conditionally mapped to the relevant
two-donor drug contrast, with vehicle identities supported by the article
rather than explicit repository aliases. No reviewed screen yet has completed
ingestion, QC, rights, sample mapping, and validation adjudication together.
Consequently all ten remain `metadata_only` and the batch contributes zero
benchmark rows.

Candidate `V1/V2/V3` grades in this batch are downstream review states. The
second review now covers all ten screens through a checksum-bound completion
lineage that preserves the frozen ranks 6-10 checkpoint and ranks 1, 3, 4, and
5 progress rows. The full comparison contains 17 provisional agreements, one
evidence-level disagreement, and two single-curator observations across 20
gene-level rows. Every row still requires named human adjudication; comparison
is excluded from selection, design features, readiness derivation, and released
labels.
