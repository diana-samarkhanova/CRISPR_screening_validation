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
not create independent training data. Translation-context evidence requires
both identifiers. Publications or reanalyses that reuse a cohort, experiment,
count matrix, or patient-level dataset must share a curator-assigned
`raw_data_family_id`; source-local `cohort_id` values alone do not prove global
identity. Family assignments therefore require an auditable curation decision.

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

ClinicalTrials.gov is a treatment/disease discovery and registry-context
source, not a gene-validation registry. The v2 adapter records the exact query,
all page URLs, canonical parsed-page checksums, API version, `dataTimestamp`,
retrieval UTC, pagination completeness, and normalized unique-NCT output. It
calls `/version` around every role-tagged declared query, aborts if the source
snapshot changes, and deduplicates only identical NCT payloads. Same-entity
treatment aliases are provenance-distinct from broader treatment-class terms;
same-entity cancer aliases are likewise distinct from broader disease-ancestor
terms, and canonical-subtype and same-subtype-alias lanes retain their own
declared roles. Class and ancestor terms can expand discovery but never become
exact entity matches. An `explicit_component` intervention match is likewise
broader and never strict. Signed subtype identity is retained in compact and
spaced labels, so `HER2+`, `HER2`, and `HER2 - breast cancer` remain distinct in
declared-query binding and local matching. Signs are also preserved for typed
biomarker state and specimen values such as `positive (+)`/`positive (-)` and
`CD3+`/`CD3-`.
Every live single-query and concept document is replayed through the frozen
snapshot parser before it is returned and must reproduce the same canonical
document. Duplicate NCT IDs within a query fail closed; identical records found
through different lanes are unioned once, conflicting payloads abort, and the
top-level study set must equal the recomputed declared-query NCT union. Report
construction replays again and rejects nested mutation after initial validation.
Typed concept snapshots require exactly one before/after version audit per
declared query. A stock live retrieval receives a non-serializable capability
bound to the final document digest and actual completion time. Frozen replay,
an injected HTTP transport, or an injected timestamp remains
source-provenance-unverified and cannot emit a strict registry count.
Query-set completion is not an ontology-recall guarantee.
Broad API search matches are adjudicated locally from structured active-
intervention, condition, and keyword fields; search retrieval or a placebo
mention alone is not an exact match.

Strict registry status uses only requested axes that the adapter can resolve
from structured intervention and condition fields. A biomarker term found in a
registry record is an untyped discovery annotation: it does not resolve feature
type, state, specimen, or measurement timepoint and cannot establish exact
typed biomarker context. A subtype can be strict only when its signed identity
and a separate exact structured parent-cancer condition are both present. An
embedded or substring parent name is not accepted without a versioned,
curator-attested parent-ID binding. Requested regimen, stage, or line of therapy remains unresolved
and forces the strict registry count to zero until arm assignment and
eligibility are parsed.

The API exposes the current record rather than a documented historical/as-of
version. Current trial status, planned outcomes, posted aggregate results, and
linked publications are therefore report-only and forbidden as historical
gene-model features. `hasResults` does not establish patient-level molecular
data, and trial presence does not establish efficacy.

Curated patient-molecular and preclinical records require row-level source
locators, availability/retrieval dates, source/raw-data families, treatment,
disease/subtype with versioned parent binding, the all-or-none biomarker
term/feature/state/specimen/timepoint/observation-status tuple, an explicit
`biomarker_axes_informative_verified` curator decision,
endpoint, and claim type. Exact typed biomarker context requires a positive
attestation and `observed` status in both the requested context and curated row;
false or unobserved status remains unresolved. The tuple is cohort context, not
the tested candidate-gene predictor, and one cannot fill the other. Patient
rows separately bind matching gene/predictor symbols to a versioned gene ID,
explicit feature/state/specimen, a compatible measurement
type/platform/timepoint, and a curator identity attestation; this is not
external resolver authentication. Preclinical gene-specific rows likewise
require a versioned, curator-attested gene identity. Preclinical records additionally require
`perturbed_compartment` and `endpoint_category`. Matching rows are filtered by
the evidence cutoff and transitive target-family exclusion, then collapsed
through source/raw-family links. Exact patient status additionally requires
matching ontology identity/version, regimen, typed biomarker, stage, and line
of therapy; a predictive interaction also requires versioned active-exposure
sets/relations/provenance, distinct source-native arm IDs, evaluable counts,
verified estimability and predictor variation, a controlled effect scale, and a
versioned inference rule. Supported, formal-null, inconclusive, and unsupported
interaction outcomes remain separate. Prognostic-only predictors require a
pretreatment/baseline measurement, and unverified treatment exposure prevents
exact patient regimen context. Exact
preclinical status requires matching compartment and endpoint category. A
non-unknown direct direction requires a versioned rule, numeric effect, sample
size, and matching curator-verified inference status; resistance,
sensitization, and discordant calls require nonzero effect. This status records
direction adjudication and is not a calibrated statistical confidence.
Resistance/sensitization/discordant use `direction_supported`, neutral uses
`neutral_supported`, and unknown uses an explicit inconclusive, unsupported, or
not-assessed state.

Compatible non-exact context and explicit conflicts have distinct provenance
and output partitions. Name-only or ontology-version-unresolved identity and
missing or narrower axes remain compatible non-exact context, never wildcard
exact. Explicit
subtype, typed-biomarker, regimen, stage, line, compartment, or endpoint
contradictions are conflicting context and do not contribute to compatible
non-exact counts or statuses. When compatible and conflicting families coexist
without a higher-priority exact or independence status, the mixed
`compatible_and_conflicting_context_present` or
`compatible_and_conflicting_patient_context_present` status is recorded instead
of either `_only` status. An empty curated input or incomplete source search
means no match in the queried material, never a biological negative. Same-study
follow-up validation remains in `validation_event` and cannot be reused as
prior evidence for its own outcome.

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

The v0.4 translation-context bundle applies the same immutable-input and atomic
publication rules, including a second input re-hash immediately before atomic
rename. It stores a canonical parsed ClinicalTrials.gov snapshot with per-page
checksums and explicitly marks it `current_snapshot_only`; live runs cannot be
backdated and wrapped snapshots cannot be restamped. Candidate-derived patient
and preclinical columns use the `report_only_` prefix and preserve the incoming
candidate order and screen axes. They cannot filter or rerank candidates, enter
the success model, or be interpreted as a validation/reproducibility
probability. The success-model field deny-list is derived from all report-only
evidence contracts and supplemented by reserved leakage-token checks.
Typed ClinicalTrials query roles and their complete cross-product are rebound
to the requested context on replay. A mismatch aborts; legacy/raw snapshots
without typed roles are marked query-context-unverified and cannot claim strict
registry matches.
A candidate TSV is bound only to a complete versioned `rank-screen` bundle when
it contains both ranking fields. Validation covers manifest/method versions,
mode and parameters, every output checksum, ordered columns and row count,
identifiers, rank/tail/direction/percentile/neutral semantics, duplicate keys,
and canonical order. It is an unsigned internal-consistency assertion, not
producer authentication.

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

## Human adjudication provenance

`prepare-validation-adjudication` verifies the completed-review manifest and
comparison checksums before creating an unsigned evidence packet. Packet rows
bind the source-family, screen, gene, both review records, their row digests,
and the parent manifest digest. Reviewer agreement remains a comparison state,
not a label.

Exactly one named human decision is required for every immutable packet item.
Its provenance includes adjudicator identity and affiliation, decision date,
source locator, rationale, conflict declaration, and attestations that the
source was reviewed, the decision was independent, model outputs were unseen,
and no automated label assignment occurred. The only dispositions are
`release_validation_event`, `no_qualifying_event`, and `defer_unresolved`.
The latter two release no event; in particular, `no_qualifying_event` cannot be
reinterpreted as `U` or `F0`.

`finalize-validation-adjudication` requires expected SHA-256 values for its
packet manifest, decision table, and validation-event table. Each release
decision also binds the exact canonical validation-event row SHA-256. The
supported `hash-validation-events` command validates the event contract and
emits these hashes without assigning labels. The
command requires exact decision coverage, validates any released event, and
publishes atomically without overwriting an existing destination.
Adjudication lineage alone cannot satisfy corpus readiness: the release
manifest retains `benchmark_ready_count=0` until independent quantitative
data/QC, rights, comparator/sample-map, and source/raw-family checks pass. The
checked-in pilot contains only an unsigned packet and blank worksheets, never
real signed human decisions.

The pinned release manifest is the registry trust root. It records canonical
hashes for every packet item, decision, and event; candidate reconciliation
rejects bare or partial TSVs even when they are internally self-consistent.
Stable reviewer person identifiers are unavailable in the frozen pilot, so
identity independence is human-attested and explicitly not presented as
cryptographic proof.
