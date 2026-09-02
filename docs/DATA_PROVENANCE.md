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

ICRAFT and related immune-screen resources are versioned external context
providers. A canonical immune comparison records its original study, screen,
comparison, source family, raw-data family, modality, perturbed compartment,
native contrast direction, endpoint polarity, source snapshot, evidence
availability date, transformation availability date, and retrieval date. The
ICRAFT CRISPRa sign inversion is source-display metadata; a restored native
effect is stored separately and never overwritten. Every numeric raw effect
declares controlled sign semantics rather than relying on a free-text metric
name.

The mutable ICRAFT portal is not a reproducible snapshot. Import requires a
frozen export checksum, row-level links to original sources, source/raw-family
mapping, and an independent rights decision. The public crawler/parser
software license must not be inherited by collected papers, counts, FASTQ,
clinical cohorts, or portal database content. This repository currently ships
the contract and report engine, not an ICRAFT export.

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

The v0.3 immune-context output is report-only. It applies the cutoff to the
latest of source evidence, provider snapshot, and transformation availability;
supports explicit target source/raw-family exclusion; and blocks all
`report_only_*` columns from the current validation-success model. A declared
full rank list is accepted for RRA only after the complete row roster, exact
`1..N` ranks, declared ranking semantics, and canonical roster SHA-256 are
verified against its gene count. CLI bundles are computed from exact byte
snapshots, re-hash inputs before atomic publication, and refuse to overwrite an
existing output directory.

The v0.4 clinical-trial registry layer is a separate treatment-by-cancer
report. Matching pins exact concept IDs plus their mapping sources and releases;
display names and free text are never implicit join keys. Eligible mappings
must be marked exact, curator-reviewed, and linked to a review event ID/date.
Each normalized row carries `source_asset_id` and the exact asset SHA-256, which
must resolve to a validated `DataAssetRecord`. The report excludes snapshots or
transformations first available after the cutoff and reports every other
exclusion. A mutable current ClinicalTrials.gov record cannot be used to
reconstruct historical status without a historical snapshot. Registry
presence, phase, completion, and `results_posted` remain context rather than
efficacy, gene evidence, or validation labels.

The ClinicalTrials.gov API snapshot intake is upstream of, and isolated from,
that reviewed report. It records the exact recall-oriented condition and
intervention search strings, scientific field projection, response format,
opaque page-token chain, first-page `totalCount`, API version, registry data
timestamp, per-response retrieval time, raw byte size, and SHA-256. Exact bytes
from the version endpoint before and after pagination and from every studies
page are retained. Every derived inventory and curation row binds to the raw
page asset that contained its study.
The inventory is a deterministic projection and normalizes missing, null, and
empty source collections to an empty list. The checksum-pinned raw JSON remains
authoritative for those structural distinctions. Injected transports, injected
clocks, and wholly synthetic fixtures are named explicitly and cannot claim live
HTTPS provenance; synthetic assets use project-only namespaces and URLs.

Pagination completeness means only complete traversal of the exact manifest-
pinned query: the token chain terminated, tokens and NCT identifiers did not
repeat, and the unique observed count equaled `totalCount`. It is not an exact
concept-mapping claim, a synonym-recall claim, or a claim that another query
would return no additional studies. Stable API version and data-timestamp
values before and after retrieval are an integrity envelope, not transactional
snapshot isolation.

The generated curation queue is fail-closed. Its treatment and condition values
are study-level search co-mentions with mapping review, intervention role,
regimen review, population review, and same-arm/same-cohort linkage marked not
performed. It is never `clinical_trial_evidence` and cannot enter the clinical
summarizer, gene ranking, feature tables, validation-success model, or labels.

Real ClinicalTrials.gov raw responses and complete snapshots stay outside Git;
only explicitly synthetic API fixtures may be checked in. The manifest retains
the official terms URL, registry data timestamp, project retrieval times,
checksums, and transformation identity. Any reproduction or redistribution
must follow the current [ClinicalTrials.gov Terms and
Conditions](https://clinicaltrials.gov/about-site/terms-conditions), including
source attribution, clear display of the date the data were processed by
ClinicalTrials.gov, and disclosure of project modifications and their date.
Checksums do not establish publisher authenticity or redistribution rights, and
submitted registry content may carry third-party or international rights.

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
