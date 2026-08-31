# CRISPR-EvidenceRank

CRISPR-EvidenceRank is a guide-aware, experimental-design-aware, and
context-aware framework for prioritizing genes from human CRISPR-Cas9 knockout
drug-response screens for orthogonal experimental follow-up.

The scientific target is deliberately narrow:

> rank genes within a completed drug-response screen by their relative
> likelihood of reproducing under independent perturbation, while separating
> screen signal, artifact risk, mechanistic support, therapeutic priority, and
> novelty.

The project does **not** treat a language-model answer, literature frequency,
or a single database score as experimental validation.

## Status

Version `0.4.0.dev0` is a reproducible development system, not a released
pretrained model. It combines a one-command report for a new MAGeCK/count
screen, an auxiliary ICRAFT-inspired immune-context method, and a report-only
treatment/disease translation layer with the normalized experimental-design,
provenance, feature-profile, and leakage-control foundation. It includes:

- normalized records for studies, screens, contrasts, samples, gene scores,
  validation events, and external evidence;
- an explicit validation-label ontology (`V3`, `V2`, `V1`, `F0`, `D`, `A`,
  `T`, `U`);
- raw-count validation, robust median-ratio normalization, and guide-to-gene
  feature extraction;
- a self-contained `rank-screen` bundle for MAGeCK summaries, count tables, or
  both, including QC, input checksums, declared direction semantics, and a
  human-readable report;
- a report-only immune-screen evidence contract, tumor/immune dual-action
  classification, provenance-aware recurrence, and verified-full-list RRA;
- a current-snapshot ClinicalTrials.gov v2 adapter plus strictly separate
  curated patient-molecular and preclinical evidence contracts, with typed
  biomarker and preclinical screen-context axes;
- a checksum-bound, human-only validation-adjudication workflow that prepares
  neutral packets and releases only explicit named-curator decisions;
- storage for raw and CNV-corrected scores without silently substituting a
  homemade correction;
- a two-stage baseline that models both author selection for testing and
  validation success among tested genes, with unweighted primary results and
  inverse-propensity weighting reported as a sensitivity analysis;
- source/raw-family- and study-grouped evaluation specifications and ranking
  metrics;
- deterministic synthetic data and automated tests.

Training on a harmonized public compendium and prospective validation are the
next scientific milestones. Until those are complete, `rank-screen` returns a
**screen-signal priority**, not a reproducibility score or calibrated
probability. The existing grouped model code remains a benchmark harness for
properly labeled data and synthetic tests.

The current ORCS intake milestone pins the human screen archive to BioGRID
ORCS `2.0.18`, compiled on `2025-09-09` and publicly available on
`2025-10-07`. The observed index contains 1,952 human screen records. Index-only
triage assigns 278 to `exclude` and 1,674 to `metadata_only`; within the latter,
435 are confirmed-scope candidates and 1,239 require manual scope review.
There are zero `benchmark_ready` screens at this stage. These counts describe
an outcome-blind curation queue, not a harmonized or trainable biological
corpus.

The first full-text pilot freezes queue ranks 1-10 from ten publication
families before outcome review. Eight screens provide author-derived score
tables only, one has a public sgRNA count matrix, and one has public amplicon
reads (`SRP158611`). Independent second review is complete for all ten screens:
the frozen ranks 6-10 checkpoint, its checksum-bound ranks 1, 3, 4, and 5
progress addendum, and an independently reviewed rank 2 were combined through
the authenticated completion workflow. The full comparison contains 20
gene-level rows: 17 provisional agreements, one evidence-level disagreement,
and two single-curator observations. All require human adjudication.
Adjudication-relevant metadata disagreements remain reported field by field
across source, design, quantitative-data, rights, validation, and blocker
metadata.
All ten screens remain `metadata_only` because count ingestion/QC, rights,
sample mapping, and/or adjudication are incomplete. Candidate validation
grades are stored downstream of selection and are not training labels.

Development from v0.2 onward is versioned in a private GitHub repository.
Public visibility is a later release gate, not a prerequisite for using Git
history and continuous integration. See `docs/REPOSITORY_POLICY.md`.

## V1 scope

Included:

- human pooled CRISPR-Cas9 knockout screens;
- drug-treated versus control competitive/viability screens;
- gene-level resistance or sensitization outcomes;
- count-table input as the primary supported mode;
- MAGeCK/drugZ/CRISPRcleanR-derived summaries as optional features;
- cancer cell-line context and study-level grouping.

Deferred to separate model branches:

- CRISPR activation and CRISPR interference;
- single-cell perturbation readouts;
- non-drug phenotypes;
- clinical target prioritization as the primary endpoint.

This separation prevents incompatible perturbation mechanisms and phenotype
definitions from being pooled into one label.

## Experimental-design hierarchy

Every observation is normalized at four distinct levels:

| Level | Meaning |
|---|---|
| `study` | Publication, preprint, or author-defined research source. |
| `screen` | Shared cell model, CRISPR modality/library, delivery, and pooled experiment. |
| `contrast` | One defined treatment-versus-comparator question, including dose, schedule, endpoint, and intended direction. |
| `sample` | One sequenced biological or technical observation assigned to a contrast. |

A gene prediction remains `gene × screen × contrast × direction`. Conditions
that differ by drug, dose, time point, comparator, or cell model must not be
silently collapsed into one contrast.

## Input modes

| Mode | Input | Current status |
|---|---|---|
| Standard | sgRNA count table + guide annotations + sample sheet | Implemented |
| Lite | MAGeCK gene summary + declared tail semantics | Implemented end-to-end report |
| Full | FASTQ + library + sample sheet | Contract defined; workflow pending |

The Standard mode is the primary modeling input because it preserves
guide-level evidence while avoiding mandatory redistribution of large FASTQ
files.

## Feature profiles

The model is evaluated as nested information profiles rather than pretending
that every user supplies equivalent data:

| Profile | Evidence used |
|---|---|
| `screen_only` | Gene scores/count-derived guide signal and technical QC. |
| `screen_plus_design` | `screen_only` plus structured study/screen/contrast/sample metadata. |
| `context_aware` | `screen_plus_design` plus versioned artifact/CNV, cell, drug, pathway, and training-only cross-screen evidence. |
| `selection_model` | A separate model for which genes authors chose to test; its selection-only fields are forbidden in the success model. |

Missing modalities retain explicit availability indicators. Reported zero is
kept distinct from “not reported,” and a reduced profile or abstention is used
when required inputs lie outside training support.

## Screen discovery and metadata sources

[BioGRID ORCS](https://orcs.thebiogrid.org/) is used to discover published
screens, recover structured metadata, and import author-method gene scores,
ranks, and hit calls. Intake is pinned to the
[ORCS 2.0.18 release archive](https://downloads.thebiogrid.org/BioGRID-ORCS/Release-Archive/BIOGRID-ORCS-2.0.18/)
rather than the mutable latest-release path. The compiled date, public
availability date, and actual retrieval date are recorded separately. ORCS is
not an orthogonal-validation registry: an ORCS hit call can be a screen-signal
feature or baseline, but never a `V2/V3/F0/D` label.

Primary papers, supplements, raw repositories, and author-supplied matrices are
used to verify exact treatment and control conditions. Library references such
as Joung et al. can identify the library design, but their example protocol is
not copied into a historical screen. Unreported MOI, coverage, selection,
treatment, duration, or comparator fields remain missing.

## Architecture

1. `ingest`: normalize the study → screen → contrast → sample hierarchy,
   validate identifiers, and record provenance.
2. `screen signal`: apply declared normalization, compute guide agreement,
   count quality, direction alignment, effect summaries, and within-screen
   ranks.
3. `artifact layer`: join copy number, expression, corrected scores, and
   off-target/multi-target flags.
4. `evidence layer`: add recurrence and context features using versioned
   sources.
   The ICRAFT-inspired immune-context layer remains a separate report-only layer;
   it cannot enter the success model from a mutable current snapshot.
5. `translation layer`: retrieve treatment/disease trial context and summarize
   curated patient/preclinical claims without creating labels, reranking genes,
   or estimating a validation probability.
6. `deduplication`: group publications, reanalyses, repositories, and ORCS
   records derived from the same experimental material into a source/raw family.
7. `selection model`: estimate which hits authors chose to test.
8. `reproducibility model`: estimate relative success among genuinely tested
   hits using grouped validation and selection-aware weights.
9. `report`: return separate component scores, uncertainty, missingness, and
   source provenance.

## Quick start

```bash
python -m pip install -e ".[dev]"
python scripts/generate_synthetic.py
crispr-evidencerank validate \
  --table examples/synthetic/sample_sheet.csv \
  --contract sample
crispr-evidencerank featurize-counts \
  --counts examples/synthetic/guide_counts_screen_01.csv \
  --samples examples/synthetic/sample_sheet_screen_01.csv \
  --output data/processed/synthetic_gene_features.csv
crispr-evidencerank featurize-design \
  --screens examples/synthetic/screens.csv \
  --screen-designs examples/synthetic/screen_designs.csv \
  --contrasts examples/synthetic/contrasts.csv \
  --samples examples/synthetic/sample_sheet.csv \
  --output data/processed/synthetic_design_features.csv
crispr-evidencerank rank-screen \
  --counts examples/synthetic/guide_counts_screen_01.csv \
  --samples examples/synthetic/sample_sheet_screen_01.csv \
  --positive-lfc-means resistance \
  --output-dir data/processed/synthetic_screen_report
pytest
```

The synthetic data demonstrate the software contract only and must never be
used as scientific evidence.

`featurize-counts` expects integer raw read counts. A positive-count
median-ratio variant is the default; all-zero and otherwise uninformative rows
are excluded from size-factor estimation. This avoids the compositional
artifacts that total-count CPM can create when a small number of guides expand
strongly. Use `--normalization-method cpm` only as a declared sensitivity
analysis.

For a MAGeCK `gene_summary` file, direction semantics are mandatory rather than
guessed:

```bash
crispr-evidencerank rank-screen \
  --mageck-summary gene_summary.txt \
  --screen-id mda_mb_468_olaparib \
  --contrast-id olaparib_vs_vehicle \
  --positive-tail-means resistance \
  --output-dir results/olaparib_screen
```

The bundle contains `ranked_candidates.tsv`, `qc_summary.json`,
`run_manifest.json`, and `report.md`; the manifest checksum-binds the three
non-manifest outputs. If counts and a sample sheet are supplied
with the MAGeCK file, native MAGeCK rank remains primary and guide agreement,
low-count fractions, zero-count fractions, and replicate correlations are
added as QC evidence. The command preserves all observed positive and negative
tail rows; it does not silently apply an FDR or top-*N* candidate filter.

## Auxiliary immune-context analysis

The ICRAFT-inspired module asks a separate translational question after the
primary screen-signal ranking: does the same perturbation appear favorable or
harmful in tumor and immune cells? It preserves the native CRISPRa sign, keeps
modalities separate, collapses correlated source/raw families, enforces a
temporal cutoff, and abstains from RRA unless complete ranked-list rosters are
verified.

```bash
crispr-evidencerank summarize-immuno-context \
  --evidence immune_screen_evidence.tsv \
  --candidates results/olaparib_screen/ranked_candidates.tsv \
  --cutoff-date 2026-08-24 \
  --target-modality CRISPR_KO \
  --exclude-raw-data-family TARGET_SCREEN_RAW_FAMILY \
  --dual-action-group-id ko_antitumor_function \
  --dual-action-group-version reviewed-2026-08-24 \
  --output-dir results/olaparib_screen/immune_context
```

The repository implements the contract and analysis engine but does not bundle
an ICRAFT database export. A real import requires a frozen checksum-pinned
export, row-level original-source provenance, source/raw-family mapping, and
reuse-rights review. The immune bundle contains `immune_context.tsv`,
`immune_context_exclusions.tsv`, `immune_context_used_evidence.tsv`,
`rank_list_audit.tsv`, and `summary.json`; primary screen axes are retained in
the joined report. See `docs/IMMUNE_CONTEXT_METHOD.md`.

## Report-only translation context

The translation layer automatically retrieves the current ClinicalTrials.gov
v2 landscape for a treatment and cancer, then keeps trial-level, curated
patient-molecular, and curated preclinical evidence in separate lanes. API
search terms are discovery queries. Same-entity treatment aliases are declared
separately from broader treatment-class terms, and same-entity cancer aliases
are declared separately from broader disease-ancestor terms. Subtype aliases
are same-subtype discovery terms, not cancer ancestors. Alias hits are retained
as alias-unverified context and cannot satisfy the strict canonical identity
lane; class and ancestor terms are also non-exact discovery context. A
treatment found only as an `explicit_component` remains non-exact and is never
strict.

```bash
crispr-evidencerank summarize-translation-context \
  --context-id mda_mb_468_olaparib_tnbc \
  --screen-id mda_mb_468_olaparib \
  --contrast-id olaparib_vs_vehicle \
  --treatment olaparib \
  --treatment-id NCIT:C71721 \
  --treatment-ontology-name NCIt \
  --treatment-ontology-version 26.07d \
  --treatment-modality small_molecule \
  --regimen-name "olaparib monotherapy" \
  --regimen-active-exposure-id NCIT:C71721 \
  --regimen-component-relation fixed_all_of \
  --regimen-active-exposures-verified \
  --regimen-active-exposure-identifier-source NCIt \
  --regimen-active-exposure-identifier-version 26.07d \
  --treatment-entity-alias Lynparza \
  --treatment-entity-alias AZD2281 \
  --treatment-class-term "PARP inhibitor" \
  --cancer-type "breast cancer" \
  --cancer-id NCIT:C4872 \
  --cancer-entity-alias "mammary carcinoma" \
  --cancer-ancestor-term "solid tumor" \
  --disease-subtype "triple-negative breast cancer" \
  --disease-subtype-id NCIT:C71732 \
  --disease-subtype-parent-id NCIT:C4872 \
  --disease-subtype-parent-binding-verified \
  --disease-ontology-name NCIt \
  --disease-ontology-version 26.07d \
  --subtype-entity-alias TNBC \
  --biomarker-context "BRCA1/2 alteration" \
  --biomarker-feature-type genomic_mutation \
  --biomarker-state pathogenic_or_loss \
  --biomarker-specimen-type tumor \
  --biomarker-measurement-timepoint pretreatment \
  --biomarker-axes-informative-verified \
  --biomarker-axes-observation-status observed \
  --screen-perturbation-modality CRISPR_KO \
  --perturbed-compartment tumor_cell \
  --screen-endpoint-category drug_response_viability \
  --context-date 2026-08-28 \
  --evidence-cutoff-date 2026-08-28 \
  --candidates results/olaparib_screen/ranked_candidates.tsv \
  --candidate-manifest results/olaparib_screen/run_manifest.json \
  --target-not-in-evidence-catalog \
  --output-dir results/olaparib_screen/translation_context
```

Omit `--clinicaltrials-json` for a live, version-checked API crawl. Supply a
previously frozen JSON snapshot for deterministic offline replay. Optional
`--patient-evidence` and `--preclinical-evidence` tables must satisfy their
strict contracts; the software does not auto-extract literature claims or
convert search hits into evidence.

Candidate rank provenance is structural and checksum-bound, not a digital
signature. `ranking_type` and `screen_signal_rank` must be supplied together. A
ranked TSV requires the complete versioned `rank-screen` bundle: manifest,
canonical candidate filename, QC JSON, and report. The loader checks schema and
method versions, input mode and parameters, every output checksum, ordered
columns and row count, screen/contrast identities, tail/direction semantics,
finite integral ranks, percentile formulas, neutral rows, duplicate keys, and
canonical order. An unsigned user-authored bundle can attest only internal
consistency; it does not prove producer identity.

Live mode paginates the role-tagged treatment, cancer, and subtype query lanes,
including declared entity aliases and broader class/ancestor discovery terms,
deduplicates NCT records, and records the role of every query and all query
URLs. Completeness applies only to that declared term set, not to unknown
synonyms or unstructured eligibility text. Passing an ontology ancestor or
drug-class term therefore expands discovery; it never turns that term into an
exact entity alias. Every live per-query snapshot and the merged concept
snapshot must round-trip identically through the frozen-snapshot validator
before publication. NCT IDs must be unique within each query and in the merged
top-level snapshot; the same NCT may be collapsed across query lanes only when
its canonical payload is identical, and conflicting payloads fail closed. The
report builder replays the snapshot again before normalization, so a nested
mutation after initial validation also fails closed.

On replay, the typed query cross-product must exactly match the requested
canonical terms, entity aliases, class terms, subtype terms, and ancestors.
A mismatched typed snapshot fails; a legacy/raw snapshot without typed query
roles is explicitly `frozen_query_context_unverified` and cannot produce a
strict registry-match count. A serialized replay cannot self-attest the live
version check; an explicitly injected transport or clock is likewise
source-provenance-unverified. Only the stock in-process live path receives the
non-serializable, snapshot-digest-bound capability required for a strict count.
Signed subtype identity is preserved during query
binding and entity matching: terminal `+` and `-` markers are not discarded, so
`HER2+`, `HER2-`, unsigned `HER2`, and spaced forms such as `HER2 + breast
cancer` remain distinct. The same sign-preserving comparison applies to
biomarker state and specimen values such as `positive (+)`/`positive (-)` and
`CD3+`/`CD3-` cells.

Trial phase, status, enrollment, and count describe one treatment/disease
landscape for the entire screen and never reorder genes. A strict registry
status is limited to axes resolvable from structured registry fields. A
biomarker keyword mention cannot establish the required typed biomarker
feature, state, specimen, and measurement timepoint and is never an exact
biomarker match. When a subtype is requested, an exact structured subtype term
supports strictness only alongside a separate exact structured parent-cancer
condition. A parent name embedded in or inferred as a substring of the subtype
label is not accepted without a versioned, curator-attested parent-ID binding.
Requested regimen, stage, and line-of-therapy axes remain unresolved by the
current study-level adapter and force the strict registry candidate count to
zero until a verified arm-assignment and eligibility parser is implemented.

The biomarker term, feature type, state, specimen type, measurement timepoint,
and observation status are an all-or-none typed context and require an explicit
`biomarker_axes_informative_verified` attestation. Exact typed matching requires
`observation_status=observed` and `true` attestation on both the requested
context and the curated evidence row;
`false` preserves the declared bundle but leaves that axis unresolved rather
than exact. The CLI records both fields with
`--biomarker-axes-informative-verified` and
`--biomarker-axes-observation-status observed`.

The cohort-context biomarker tuple above is separate from the candidate gene
whose association is tested. Every patient row binds `gene_symbol` to the same
`predictor_gene_symbol`, a versioned `gene_id` plus identifier source/release,
and an explicit predictor feature, state, specimen, measurement type/platform,
and timepoint. Feature and assay must be compatible (for example RNA expression
with an RNA/expression measurement). `predictor_identity_curator_verified=true`
records a curator's audited identity assertion; it is not authentication by an
external ontology or gene resolver.

A patient association is called predictive only when pretreatment measurement,
canonical active-exposure sets,
component relations and identifier provenance, distinct source-native arm IDs,
evaluable arm/model counts, scale-appropriate event counts, verified
estimability and predictor variation,
a versioned treatment-by-predictor inference rule, and consistently supportive
departure-from-null metrics coexist. Formal null, inconclusive, and unsupported
results remain separate report lanes. Direct preclinical interaction
requires vehicle/baseline and genotype-by-treatment controls; exact preclinical
context also requires matching `perturbed_compartment` and `endpoint_category`,
and directional concordance additionally requires the target screen's
perturbation modality. A non-unknown direct direction must also carry a numeric
effect, reported sample size, curator-verified versioned inference rule, and a
matching direction status. Neutral, inconclusive, unsupported, and unassessed
states remain separate and cannot masquerade as directional support:
resistance/sensitization/discordant require `direction_supported`, neutral
requires `neutral_supported`, and unknown requires an explicit inconclusive,
unsupported, or not-assessed status. Every
gene-specific preclinical row also requires a versioned, curator-attested gene
identity. Prognostic-only patient predictors must be pretreatment/baseline, and
unverified patient treatment exposure cannot establish exact regimen context.

Versioned treatment and cancer IDs are optional for a broad discovery report
but mandatory for strict registry or exact curated-context status. A matching
name without those IDs remains compatible non-exact.

Compatible but non-exact evidence is counted separately from explicit context
conflicts. Alias-only or ontology-version-unresolved identity can remain
compatible non-exact evidence; explicit subtype, biomarker, regimen, stage, line,
compartment, or endpoint contradictions are conflicting evidence. When both
compatible-non-exact and conflicting families are present and no higher-priority
exact-context or independence status applies, the statuses are
`compatible_and_conflicting_context_present` for preclinical evidence and
`compatible_and_conflicting_patient_context_present` for patient evidence,
rather than incorrectly labeling either lane as `*_only`; the separate family
counts retain both partitions regardless of status precedence. Every added
candidate column is `report_only_*`: the layer does not filter or rerank the
input candidates and does not emit a reproducibility or validation probability.
See `docs/TRANSLATION_CONTEXT_METHOD.md`.

For a release-pinned BioGRID ORCS archive:

```bash
crispr-evidencerank prepare-orcs-release \
  --release-registry config/orcs_releases.yaml \
  --release 2.0.18 \
  --retrieved-date 2026-07-31 \
  --cache-dir data/external/orcs_2.0.18 \
  --output-dir data/processed/orcs_2.0.18/release_intake
```

This command downloads or reuses only the pinned archive, verifies its
project-computed SHA-256 and exact tar inventory, cross-checks all index IDs
against the per-screen member IDs, then atomically writes normalization,
provenance, triage, and curation-queue outputs. It refuses to mix a rerun into
an existing output directory.

The lower-level adapters remain available for focused reprocessing:

```bash
crispr-evidencerank ingest-orcs-index \
  --table BIOGRID-ORCS-SCREEN_INDEX-2.0.18.index.tab.txt \
  --release 2.0.18 \
  --available-date 2025-10-07 \
  --retrieved-date 2026-07-31 \
  --organism-scope "Homo sapiens" \
  --output-dir data/processed/orcs_2.0.18/index

crispr-evidencerank triage-orcs-index \
  --table BIOGRID-ORCS-SCREEN_INDEX-2.0.18.index.tab.txt \
  --release 2.0.18 \
  --available-date 2025-10-07 \
  --retrieved-date 2026-07-31 \
  --organism-scope "Homo sapiens" \
  --output-dir data/processed/orcs_2.0.18/triage

crispr-evidencerank ingest-orcs-screen \
  --table BIOGRID-ORCS-SCREEN_95-2.0.18.screen.tab.txt \
  --index-metadata BIOGRID-ORCS-SCREEN_INDEX-2.0.18.index.tab.txt \
  --release 2.0.18 \
  --output-dir data/processed/orcs_2.0.18/screen_95
```

The adapter preserves lossless raw tables alongside normalized registry
records and parsing issues. `HIT` remains `author_hit`; unreported comparator
and score direction remain unknown.

Index triage writes one screen-level status and one auditable row per
eligibility rule. Explicit non-human, non-KO, non-drug, or arrayed records may
be excluded. Missing metadata remains `metadata_only`; index metadata can
never promote a screen directly to `benchmark_ready`.

For ORCS 2.0.18, the observed deterministic triage is:

| Intake result | Screen records |
|---|---:|
| Total | 1,952 |
| `exclude` | 278 |
| `metadata_only` | 1,674 |
| Confirmed-scope queue | 435 |
| Manual-scope-review queue | 1,239 |
| `benchmark_ready` | 0 |

The queue is outcome-blind: ranking uses scope and metadata completeness, not
ORCS `HIT`, author score magnitude, validation outcomes, or later evidence.
Queue rank is therefore a curation priority only. It must not be interpreted as
a biological hit rank or model-training label.

`Toxin Exposure` is not treated as equivalent to `Drug Exposure`, because the
ORCS condition field may describe a toxin, pathogen, medium, or another
non-drug exposure. Such records remain candidates for manual review. A curated
screen becomes `benchmark_ready` only when the complete policy-v2 rule set is
present and passing, including comparator/sample reconstruction, count-level
signal, source and raw-data families, rights, and adjudicated validation
events. A draft `single_curator` event remains useful for curation but cannot
open the benchmark gate until it receives an approved consensus status.

## Human validation adjudication

Reviewer agreement is evidence for a human adjudicator to inspect, not a
validation label. The software never maps a provisional agreement or reviewer
evidence level to `V2`, `V3`, `F0`, or `D`. It first creates a neutral,
checksum-bound packet from the completed dual-review bundle:

```bash
crispr-evidencerank prepare-validation-adjudication \
  --completed-review-manifest dual_review_manifest.json \
  --expected-completed-review-manifest-sha256 \
  "$COMPLETED_REVIEW_MANIFEST_SHA256" \
  --expected-comparison-sha256 "$REVIEW_COMPARISON_SHA256" \
  --packet-id orcs-2.0.18-batch-001-adjudication-v1 \
  --prepared-date 2026-08-29 \
  --output-dir adjudication_packet
```

A named human must inspect the cited evidence and provide exactly one decision
for every packet item. The permitted dispositions are:

- `release_validation_event`: one complete, contract-valid event is supplied;
- `no_qualifying_event`: the cited material does not establish a qualifying
  event, so no label is emitted;
- `defer_unresolved`: the evidence remains unresolved and no label is emitted.

`no_qualifying_event` is neither `U` nor `F0`: absence of a qualifying event
does not establish an untested gene or a successful negative validation
experiment. Every decision must attest that the source was reviewed, the
decision was made independently by a human, model outputs were unseen, and no
label was assigned automatically. Finalization verifies exact packet coverage,
checksums, identities, attestations, and any linked validation event. Each
release decision binds the exact canonical event-row SHA-256 before publishing
an atomic release bundle. Generate those hashes with the supported neutral
helper, then copy the hash for each released `event_id` into the corresponding
decision row:

```bash
crispr-evidencerank hash-validation-events \
  --validation-events completed_validation_events.tsv \
  --output validation_event_hashes.tsv
```

The helper validates the full event contract, reports the exact input-table
SHA-256, and performs no label assignment. Finalize with those pinned bytes:

```bash
crispr-evidencerank finalize-validation-adjudication \
  --packet-manifest adjudication_packet/adjudication_packet_manifest.json \
  --expected-packet-manifest-sha256 "$ADJUDICATION_PACKET_MANIFEST_SHA256" \
  --decisions completed_adjudication_decisions.tsv \
  --expected-decisions-sha256 "$ADJUDICATION_DECISIONS_SHA256" \
  --validation-events completed_validation_events.tsv \
  --expected-validation-events-sha256 "$VALIDATION_EVENTS_SHA256" \
  --adjudicated-date 2026-08-29 \
  --output-dir adjudication_release
```

An adjudication release deliberately reports `benchmark_ready_count=0`.
Benchmark readiness is derived later from independent count-level data/QC,
rights, comparator/sample-map, and provenance gates as well as the released
event. Registry promotion additionally requires the explicitly pinned release
manifest; self-consistent event and decision TSVs are not a trust root. The
release manifest binds every packet item, human decision, and event by canonical
record SHA-256. The frozen pilot reviews lack stable person identifiers, so
reviewer/adjudicator independence is an explicit human attestation plus a
display-name sanity check, not cryptographic identity proof. The repository
bundles only the unsigned packet and blank worksheets; it contains no real
signed human decisions.

Until a released-compendium manifest is implemented, the `benchmark` CLI
default-denies all labeled runs. Its only escape hatch is
`--development-synthetic-labels-only`; every resulting row is watermarked
`development_only`, `synthetic_unverified`, and
`scientific_use_prohibited=true`.

## Training-label policy

- `V3`: causal validation with independent perturbation plus rescue or a
  comparably strong causal reversal.
- `V2`: phenotype reproduced with independent reagent(s), confirmed
  perturbation, and an appropriate control.
- `V1`: supportive but incomplete follow-up.
- `F0`: perturbation succeeded, but the screen phenotype did not reproduce.
- `D`: a verified result in the opposite direction.
- `A`: ambiguous interpretation.
- `T`: technical failure; phenotype cannot be judged.
- `U`: untested or testing status unknown; `testing_status` records
  `not_tested` and `unknown` separately.

Primary supervised labels are `V2/V3` (positive) and `F0/D` (negative).
`V1/A/T/U` remain missing or auxiliary. A gene absent from a paper's validation
section is not a negative.

## Leakage controls

- split by source/raw-data family and study, never by gene row;
- keep every reanalysis of the same accession or author matrix in one fold;
- report leave-one-study-out and leave-one-drug-class-out performance;
- freeze the model before evaluating any held-out private screen;
- recompute cross-screen evidence from outer-training families only and apply a
  source-availability time cutoff;
- keep exact screen-derived features separate from post-publication evidence;
- use ORCS screen IDs as provenance/join keys and ORCS scores as screen evidence,
  not validation outcomes;
- keep ClinicalTrials.gov trial records at treatment level; current mutable
  snapshots are never historical gene-model features;
- require exact treatment/disease/cohort linkage before calling patient data
  treatment-associated, and a formal interaction before calling it predictive;
- keep treatment-activity panels, natural biomarker associations, and direct
  gene perturbation as different preclinical claim types;
- derive the success-model field deny-list from every report-only evidence
  contract, in addition to the reserved leakage-token guard;
- never use the validation paper's prose or follow-up result as an input
  feature for that same label.

## Held-out prospective case study

An unpublished CRISPR-KO drug-response screen is reserved as a prospective
case study. Its cell model, treatment, anchor genes, count tables, and results
are intentionally excluded from this repository. Any non-KO branch will be
evaluated separately after the KO model is frozen.

## Data and copyright

Code is licensed under Apache-2.0. The ORCS-distributed 2.0.18 archive and its
index/score files are covered by the ORCS download terms recorded in the data
manifest. That license does not transfer to publisher supplements, GEO/SRA
assets, author count matrices, FASTQ files, or other upstream source data;
those retain their own terms and require separate rights review. This
repository should contain accession manifests, download scripts, checksums,
derived features that are lawful to redistribute, and exact provenance—not
copied article text, figures, or unrestricted copies of source matrices.

The recorded SHA-256 for the ORCS archive is computed by this project after
download and is not represented as a checksum published by BioGRID. The
retrieval timestamp records when that exact byte stream was obtained; it is not
the release's compiled or availability date.

For ORCS-derived records, preserve the ORCS release, screen ID, original
publication, author scoring method, and raw/source-family mapping. Multiple
methods or repository copies of one experiment do not create independent
training screens.

The local annotation workbook is a private research artifact and is excluded
from Git by default. Before a public release, verify each dataset version,
license, institutional approval, authorship, embargo, and patent position.

## Intended publication benchmark

The minimum credible paper requires:

- a versioned multi-study training compendium;
- explicit successful and failed validation events;
- MAGeCK/drugZ and effect-size baselines;
- study-, drug-, cell-line-, and gene-cold evaluation;
- observed success yield@5/10/20, NDCG, PR-AUC, calibration, and bootstrap
  uncertainty;
- ablation of guide, CNV/artifact, recurrence, context, and literature layers;
- a frozen-model prospective validation study.

See `docs/SCIENTIFIC_PROTOCOL.md` and `docs/MODEL_CARD.md`.

For the adjacent pre-screen task, [AssayBench](https://github.com/Genentech/AssayBench)
benchmarks prediction of a screen gene ranking from an assay description. It is
an important landscape comparator, but it predicts the screen result before
screening rather than the orthogonal reproducibility of an already observed
guide-level hit.
