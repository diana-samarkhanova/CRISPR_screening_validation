# Changelog

All notable project changes are recorded here. The project follows semantic
versioning once public releases begin; development checkpoints use a `.devN`
suffix and are not claims of a trained biological model.

## Unreleased — development selection to prospective allocation

- Added nested source/raw-family grouped configuration selection and a separate
  development-only final selector, with complete inner/outer diagnostics.
- Preserved selected logistic regularization; disabled row-level HGB early
  stopping and corrected explicit reverse-native LFC alignment.
- Hardened model loading with verified bytes, runtime/feature/parameter checks,
  training-family hash inventories and no-clobber publication.
- Added outcome-free frozen KO inference with family-overlap and missing-feature
  guards, source-attested baseline rank linkage, and checksummed predictions.
- Added complete paired top-k allocation, randomized remainder candidates,
  declared prior-exposure exclusions, masked assay roster and offline verifier.
- Added installed CLI end-to-end CI coverage using synthetic inputs only.
- Documented wet-lab validation and remaining scientific/publication gaps.

## 0.5.0.dev0 — ClinicalTrials.gov frozen snapshot intake

- Added a separate bounded ClinicalTrials.gov v2 acquisition path that retains
  the exact version response before and after pagination, every raw studies
  page, the exact recall-oriented query and field projection, and the opaque
  page-token lineage.
- Added a checksum-bound manifest, deterministic study projection, data-asset
  registry, fail-closed curation queue, and offline snapshot verifier.
- Scoped completeness to termination of the exact query's token chain, unique
  NCT identifiers, agreement with first-page `totalCount`, and a stable API
  version/data-timestamp envelope; explicitly disclaimed transactional snapshot
  isolation and ontology/synonym recall.
- Kept all search-derived intervention/condition rows as unreviewed study-level
  co-mentions without inferred exact concepts, intervention role, regimen,
  population scope, or same-arm/same-cohort linkage. They cannot enter the
  clinical summarizer, gene ranking, model features, or validation labels.
- Required real API pages and snapshots to remain outside Git while permitting
  clearly labeled synthetic fixtures for offline tests.
- Documented ClinicalTrials.gov attribution, registry processing date, dated
  project modifications, and independent rights review requirements for reuse;
  hashes establish integrity, not redistribution permission or endorsement.

## 0.4.0.dev0 — clinical treatment-by-cancer context

- Added the strict `clinical_trial_evidence` contract for one frozen source
  study × treatment concept × cancer concept row, with controlled study type,
  status, phase, intervention role, and regimen fields.
- Bound every normalized clinical row to a checksum-pinned `DataAssetRecord`
  and enforced source/version/date consistency before reporting.
- Added `summarize-clinical-context`, a mapping-release-pinned,
  curator-reviewed exact concept-ID, temporal-cutoff, source-family-aware report
  of observed experimental-role interventional registry studies.
- Kept clinical output on the treatment × cancer axis rather than repeating it
  per gene; all derived summary fields are `report_only_clinical_*` and cannot
  change a CRISPR rank or create a validation label.
- Distinguished registry presence, study status, phase, regimen, and aggregate
  results availability from efficacy; zero matches means only not observed in
  the supplied snapshot.
- Replaced blacklist-only success-model protection with a reviewed feature
  allowlist so arbitrarily named clinical or post-cutoff fields fail closed.
- Added packaged schema/lock provenance, installed-wheel CI coverage, and a
  second input-hash check immediately before atomic bundle publication.
- Added a synthetic olaparib–TNBC example and documented why TNBC, HER2-negative
  disease, HRD, somatic BRCA, and germline BRCA are not interchangeable.

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

## Unreleased

### Publication-readiness corrections

- Required explicit `positive_lfc_means` semantics for count featurization,
  retained the native LFC sign separately, and removed the low-level assumption
  that every positive treatment-minus-control LFC denotes resistance.
- Added prevalence, correctly normalized AP, informative top-k denominators,
  strict-cutoff NDCG, honest `documented_success_*` diagnostics, conditional
  calibration slope/intercept and reliability bins, and direction-stratified
  benchmark metrics.
- Added repeated-seed diagnostics, source-family cluster intervals in the
  benchmark CLI, a within-query feature-invariance audit, safe fixed-score
  baselines, and paired candidate-minus-baseline cluster-bootstrap intervals.
- Removed duplicate prediction columns and the CSV-unsafe internal tuple query
  identifier from public benchmark outputs; renamed the surviving propensity
  output to `selection_propensity`.
- Added class-specific label agreement and full chance-corrected agreement for
  `testing_status` to the pre-adjudication Gold report.
- Added checksum-bound frozen-model bundles that pin the training table,
  feature registry, environment lock, full Git revision, and serialized model;
  verification does not deserialize untrusted joblib content. Freeze now rejects
  an in-memory frame that differs from the pinned table and reports a manifest
  hash that can be checked against an external identity anchor.
- Kept source/raw-family fold unioning active for custom split groups and
  separated the bootstrap resampling cluster from the study-macro aggregation
  identifier.
- Made benchmark output paths fail closed against input/output collisions and
  delayed no-clobber writes until all statistical validation and input snapshot
  checks have completed; reports now pin input, prediction, build, and lock
  hashes.
- Made top-k summaries permutation-invariant under cutoff ties, excluded
  single-class queries from inferential ranking macros with explicit
  denominators, made calibration bins tie-preserving, and required one complete
  finite roster for paired inference.
- Added direction/provenance-attested MAGeCK and drugZ adapters with strict
  numeric-domain validation; histogram boosting now receives a paired
  candidate-minus-logistic interval rather than a point-only comparison.
- Added reason-coded alternate-seed failures and removed the accidental
  always-appended default `--seed-metric` when callers select another metric.

### Reporting and documentation

- Added a practical DOME reporting crosswalk and an explicitly adapted,
  non-primary TRIPOD+AI applicability check for the non-clinical
  gene-by-screen ranking task.
- Bound the historical synthetic benchmark table to its four-fold command,
  checkpoint commit, and lock-file provenance; fixed its Markdown separator and
  stopped copying inherently stale test totals into the living checkpoint.
- Corrected the documented model state: `context_aware` and dedicated
  selection-only covariates are specifications, whereas the current selection
  endpoint shares the reviewed pre-outcome inputs of the paired success model.

### Gold validation-event curation

- Added strict contracts and checksum-bound atomic stages for frozen event
  cards, verified human actors, two independent human reviews, field-level
  comparison and agreement statistics, named-human adjudication with recorded
  attestations, and derived validation-event records.
- Added `prepare-event-review-packet`, `compare-event-reviews`, and
  `finalize-event-adjudication`. Each downstream stage requires the exact
  predecessor-manifest SHA-256, complete event-roster coverage, and immutable
  packet version/payload linkage.
- Limited the generated four-file core bundle to randomized source-located
  evidence cards, a blank review template, the frozen rubric, and a checksum
  manifest. Its content excludes model/EvidenceRank predictions, prior labels,
  and co-reviewer decisions; the screen-predicted direction remains visible so
  concordant and opposite outcomes can be judged.
- Kept all reviewer agreement provisional. Every event, including exact
  agreement, requires named human adjudication; comparison alone cannot emit a
  final validation label.
- Limited finalization's scientific label output to derived validation-event
  records while preserving its audit lineage and named adjudications. It
  cannot emit or promote `benchmark_ready`; screen readiness still requires
  the separate registry gates.
- Kept the checked-in event-level examples and tests artificial/synthetic.
  Real source PDFs, copied figures, private workbooks, completed review forms,
  and access-controlled curation bundles remain outside Git.
- Made the packet's blinding boundary explicit: joinable source identifiers
  require a coordinator-attested, access-isolated reviewer workspace. Actor
  identity and declaration fields are audit attestations, not cryptographic
  signature verification.
- Required a coordinator's manual neutral-card audit attestation in addition
  to lexical leakage guards; neither mechanism is described as proof of
  cognitive blinding.
- Classified the generated four-file output as the core table bundle, not a
  complete reviewer handoff: frozen sources/source index and the human
  identity, independence, and conflict declaration remain required external
  checksum-bound inputs.

### ORCS 2.0.18 intake checkpoint

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
