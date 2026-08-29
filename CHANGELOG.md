# Changelog

All notable project changes are recorded here. The project follows semantic
versioning once public releases begin; development checkpoints use a `.devN`
suffix and are not claims of a trained biological model.

## 0.4.0.dev0 — translation-context engine

- Added `summarize-translation-context`, with live ClinicalTrials.gov v2
  retrieval or deterministic replay from a frozen JSON snapshot.
- Added before/after API-version and `dataTimestamp` checks, bounded audited
  pagination, canonical page checksums, role-tagged treatment, cancer, subtype,
  entity-alias, class, and ancestor discovery queries, exact NCT deduplication,
  SHA-256 input/output manifests, and atomic publication with a final
  input-mutation check. Every live query and merged concept snapshot must replay
  identically through the frozen validator; duplicate NCTs within a lane or the
  merged output and conflicting same-NCT payloads across lanes fail closed. The
  report builder replays again before normalization and detects nested mutation
  after initial validation.
- Bound verified live provenance to a non-serializable in-process capability and
  the final snapshot digest. Serialized replay, injected transports, and
  injected clocks remain reportable but cannot emit a strict registry count;
  typed query sets require one before/after version audit per query.
- Bound typed ClinicalTrials query cross-products to the requested replay
  context; typed mismatches fail and legacy/raw snapshots cannot emit strict
  registry-match counts. Query binding and entity matching preserve terminal
  or spaced subtype `+`/`-` markers, and typed matching preserves signs in
  biomarker state and specimen values.
- Treated API text search as candidate retrieval and assigned intervention,
  disease/subtype, biomarker, and listed-regimen relations separately from
  structured fields.
- Separated same-entity treatment aliases from broader treatment-class terms
  and same-entity cancer aliases from broader disease-ancestor terms; class and
  ancestor discovery matches cannot become exact entity matches. An
  `explicit_component` intervention match remains broader and never strict.
- Replaced the ambiguous translation CLI alias options with
  `--treatment-entity-alias`, `--cancer-entity-alias`, and
  `--subtype-entity-alias`.
- Added strict treatment/disease, clinical-trial, curated preclinical, and
  patient-molecular contracts plus deterministic JSON Schemas.
- Required biomarker term, feature type, state, specimen type, and measurement
  timepoint and observation status together plus an explicit
  `biomarker_axes_informative_verified` attestation. Exact typed matching
  requires `observed` status and true attestation on both the requested context
  and curated row. A registry biomarker mention remains untyped discovery
  context and cannot establish an exact typed biomarker match or strict
  biomarker-constrained registry status.
- Required versioned canonical active-exposure sets, relations and provenance,
  distinct source-native arm IDs, evaluable arm/model counts, scale-appropriate
  event counts, verified estimability and predictor variation, a controlled effect scale, and a
  versioned inference rule before a patient claim can be called predictive.
  Supported, formal-null, inconclusive, and unsupported interactions are
  separate lanes; longitudinal and post-progression claims remain guarded.
- Separated the cohort-context biomarker tuple from the candidate-gene
  predictor. Patient rows bind matching gene/predictor symbols to a versioned
  gene ID, feature/state/specimen, compatible assay and measurement timepoint,
  and explicit curator identity attestation. Ambiguous or conflicting assay
  text fails closed; the attestation is not resolver authentication.
- Required ontology-aware exact-context matching across treatment, disease,
  subtype, typed biomarker, regimen, stage, and line of therapy, plus matching
  perturbed compartment and endpoint category for preclinical claims.
- Required a requested registry subtype to have a separate exact structured
  parent-cancer condition before it can satisfy strict disease matching. A
  parent name embedded in or inferred as a substring of the subtype label is
  not accepted without a versioned, curator-attested parent-ID binding. Requested
  regimen, stage, or line of therapy forces the strict registry count to zero
  until verified arm-assignment and eligibility parsing is implemented.
- Split compatible non-exact evidence from explicit context conflicts in both
  patient and preclinical counts and statuses. Name-only or ontology-version
  uncertainty remains compatible non-exact context; explicit subtype, biomarker,
  regimen, stage, line, compartment, or endpoint contradictions are reported
  as conflicting context. When no higher-priority exact-context or independence
  status applies, mixed lanes emit
  `compatible_and_conflicting_context_present` or
  `compatible_and_conflicting_patient_context_present` rather than an
  inaccurate `*_only` status; separate family counts still retain both
  partitions.
- Required vehicle/baseline and genotype-by-treatment controls for every direct
  perturbational claim, versioned direction rules, and modality-matched
  direction concordance. Gene-specific preclinical rows now require a
  versioned, curator-attested gene identity; non-unknown directions require a
  numeric effect, sample size, and matching curator-verified inference status.
- Required literal booleans for scientific attestations and rejected boolean
  coercion in MAGeCK metrics, raw counts, replicate numbers, and bound bundle
  numeric fields. The manifest consumer now rechecks native MAGeCK/count
  scientific domains rather than trusting self-consistent checksums alone.
- Required versioned treatment/cancer IDs for strict registry and exact curated
  context, baseline timing for prognostic-only predictors, and verified patient
  treatment exposure for exact regimen context.
- Added source/raw-family collapse, temporal cutoff, target-family exclusion,
  model-applicability/OOD reporting, and explicit missingness.
- Preserved candidate order and primary screen axes, emitted only
  `report_only_*` gene-context columns, and expanded the success-model leakage
  guard with a deny-list derived from every report-only evidence contract. The
  layer neither filters nor reranks candidates and emits no validation or
  reproducibility probability.
- Required candidate TSVs to supply both ranking fields and bind the complete
  versioned `rank-screen` bundle. Validation now covers mode, schema/method
  versions, parameters, all output hashes, ordered columns/row count,
  identifiers, tail/direction/rank/percentile/neutral semantics, duplicate keys,
  and canonical order. The unsigned binding attests consistency, not producer
  identity.
- Kept trial counts, phases, statuses, posted aggregate results, patient
  associations, and model tiers out of the validation score and labels.

## 0.3.0.dev0 — screen report and immune-context engine

- Added `rank-screen`, a one-command bundle for MAGeCK gene summaries, raw
  count tables, or both. It writes ranked candidates, QC, input checksums, run
  parameters, and a human-readable report.
- Required explicit semantics for the MAGeCK positive tail and positive count
  LFC. The software does not guess whether a sign means resistance or
  sensitization.
- Kept the new-screen output explicitly named `screen_signal_baseline`; zero
  released real-data labels means it is not a validation probability.
- Added the `immune_screen_evidence` contract and generated JSON Schema with
  exact modality, compartment, setting, phenotype, contrast, native direction,
  orthology, rank-list, provenance, cutoff, and transformation fields.
- Added `summarize-immuno-context` as a report-only post-ranking method with
  tumor, immune, and in-vivo evidence lanes; dual-action/liability categories;
  source/raw-family collapse; temporal and self-family exclusion; and explicit
  missingness.
- Preserved native CRISPRa effects independently of ICRAFT's KO-equivalent
  display inversion, required a registered numeric LFC sign-pair for that
  display transform, added controlled raw-effect sign semantics, and kept KO,
  CRISPRi, and CRISPRa in separate queries.
- Restricted de novo RRA to checksum-identified rank lists whose complete
  `1..N` roster is observed, and required at least two independent provenance
  components. Ineligible lists produce a null p-value and reason-coded
  abstention.
- Blocked every `report_only_*` immune column from the validation-success model
  and prohibited ICRAFT recurrence, dual-action classes, and expression or
  clinical associations from supplying validation labels.
- Made both report bundles input- and output-checksum-bound and atomic, preserved identifier
  strings and all primary screen axes, required versioned dual-action grouping,
  and added explicit replicate/guide-coverage warnings.
- Documented the ICRAFT comparison, adopted methods, scientific boundaries,
  current lack of a redistributable frozen ICRAFT export, and remaining
  limitations.

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
- Completed the checksum-bound second review for all ten pilot screens. The
  authenticated full comparison contains 20 gene-level rows: 17 provisional
  agreements, one evidence-level disagreement, and two single-curator
  observations. Human adjudication remains explicitly pending, and comparison
  cannot release a benchmark label.
- Added `complete-curation-reviews`, which requires the exact remaining review
  complement plus authenticated predecessor and progress checksums, then
  publishes a self-contained eleven-file bundle atomically for cooperating CLI
  writers. Immutable snapshots, canonical filenames, raw-cell preservation,
  and replay tests prevent partial, progress-rewriting, or basename-dependent
  completion.
- Materialized the first self-contained eleven-file completion bundle while
  preserving the frozen ranks 6-10 checkpoint and ranks 1, 3, 4, and 5 progress
  rows byte-for-byte. Rank 2 was independently reviewed as conservative `V1`;
  the resulting bundle retains zero adjudicated genes, released labels, and
  `benchmark_ready` screens.
- Added a complete 24-run accession map for `SRP158611`, with eight runs
  conditionally assigned to the two-donor CGS-21680 contrast and the other 16
  excluded as different screens. A separately checksummed ENA inventory
  prevents truncated maps from appearing complete, while a distinct curated
  scope table preserves article-supported inclusion decisions. Raw reads remain
  external to Git.
- Registered the public Findlay Supplementary Dataset EV3 sgRNA count matrix
  and retained its methods-versus-matrix replicate conflict as an unresolved
  sample-map blocker.
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
