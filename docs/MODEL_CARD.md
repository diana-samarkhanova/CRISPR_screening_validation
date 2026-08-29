# Model card — development version

## Model name

CRISPR-EvidenceRank KO reproducibility model.

## Intended use

Prioritize a finite number of genes for orthogonal laboratory validation after
a human CRISPR-Cas9 knockout drug-response screen. The v0.2 input is normalized
as study → screen → contrast → sample, and predictions are made for a
`gene × screen × contrast × direction` instance.

## Out-of-scope use

- clinical decisions;
- patient-level treatment recommendations;
- claiming a gene is validated without experiment;
- combining CRISPRa/CRISPRi with KO without modality-specific retraining;
- ranking genes from a list that lacks screen and context metadata as though it
  had guide-level support.

## Current model

For a new unlabeled screen, v0.4 exposes a deterministic
`screen_signal_baseline` through `rank-screen`. It preserves native MAGeCK rank
or ranks absolute guide-level effects within the explicitly declared phenotype
direction. This report is not produced by the validation-success model and is
not a probability of reproduction.

The repository implements a transparent two-stage baseline:

1. logistic selection model for `tested` versus explicitly `not_tested`;
2. regularized logistic success model among tested genes;
3. inverse-study base weights so one publication cannot dominate by reporting
   more screens or tested genes;
4. optional clipped selection-IPW, disabled when inner grouped cross-fitting is
   impossible or any inner training fold contains a single testing-status
   class;
5. study-grouped out-of-fold evaluation.

This is a benchmark and infrastructure checkpoint, not the final trained model.
The v0.2 benchmark specification additionally groups all article/preprint,
repository, ORCS, and alternative-method records derived from the same
experimental material into one source/raw-data family.

## Feature profiles

- `screen_only`: gene scores/count-derived guide evidence and technical QC;
- `screen_plus_design`: `screen_only` plus structured experimental design;
- `context_aware`: `screen_plus_design` plus versioned artifact/CNV, cell,
  drug, pathway, and training-only cross-screen evidence;
- `selection_model`: separate author-testing endpoint; selection-only fields are
  forbidden in the validation-success model.

Profiles are evaluated separately. Missing inputs retain explicit availability
indicators; reported zero is not treated as missing, and unsupported inputs
route to a reduced profile or abstention.

## Candidate production models

- regularized logistic regression;
- histogram gradient boosting;
- LightGBM/XGBoost, if added as optional dependencies;
- learning-to-rank with screen query groups;
- selection-aware or positive-unlabeled variants.

No model family is accepted without grouped, temporal, and cold-context
evaluation.

## Outputs

- `reproducibility_score`;
- `selection_propensity_known_status`;
- `artifact_risk` component;
- feature missingness;
- within-screen rank;
- uncertainty from study-level resampling;
- separate mechanistic, therapeutic, and novelty evidence components.
- profile name, metadata completeness, and outside-training-support flags.

The auxiliary immune-context command writes only `report_only_*` columns. It
reports tumor/immune recurrence, conflicts, dual-action hypotheses, and
verified-full-list RRA without changing any primary output. These columns are
programmatically forbidden from the current success model. Candidate rows from
`rank-screen` retain their screen, contrast, direction, tail, and signal rank;
the immune axis neither filters nor reorders them.

The translation-context command is also outside the success model. Trial
records are treatment/disease-level and structurally lack a gene-ranking role.
Curated patient and preclinical summaries add only `report_only_*` candidate
columns and preserve input row order, screen rank, contrast, and direction.
Trial counts, phases, status, enrollment, posted aggregate results, treated
cohort associations, and model-system tier never become a composite score or
probability. The command neither filters nor reranks candidates.

Translation-context matching is typed. Biomarker term, feature type, state,
specimen type, and measurement timepoint are required together. Preclinical
exact context additionally requires the perturbation compartment and endpoint
category to match the target screen. Compatible non-exact evidence and explicit
context conflicts have separate family counts and statuses; they are not pooled
into one “broader” bucket.

## Known limitations

- validation events are selectively reported;
- explicit failed validations are rare;
- study protocols and gene libraries are heterogeneous;
- exact study/screen/contrast/sample metadata are incompletely reported in many
  historical sources;
- CNV correction outputs are method-dependent;
- literature and database coverage are biased toward well-studied genes;
- the first compendium may be dominated by a few drugs or cancer types;
- relative scores are not calibrated probabilities until prospective evidence
  supports that interpretation.
- selection-IPW does not yet correct the separate probability that a tested
  outcome becomes adjudicable (`V2/V3/F0/D` rather than `V1/A/T`).
- V1 accepts only resistance and sensitization prediction queries; neutral,
  unknown, and discordant directions are retained as observed evidence states.
- bootstrap intervals are unavailable rather than silently estimated when a
  metric has fewer than the prespecified effective number of study draws.
- BioGRID ORCS preserves heterogeneous author-defined screen scores and hit
  calls; these are not independent-validation outcomes.
- ICRAFT recurrence and RRA are immune-screen context, not effect size,
  orthogonal validation, therapeutic efficacy, or validation probability. No
  frozen ICRAFT export is bundled in v0.3.
- A dual-action class is a hypothesis-generating candidate category. CRISPRa
  display inversion is accepted only as a registered numeric LFC sign-pair and
  is not propagated, modality-mismatched evidence is not combined, versioned
  dual-action groups require review, and ambiguous orthology is annotation-only.
- ClinicalTrials.gov search is broad candidate retrieval and the current API
  record is mutable. Even a complete role-tagged declared query set is not
  exhaustive ontology-level recall. Local structured matches do not establish
  efficacy, patient-level RNA-seq, gene-specific response, or a historical
  feature. Every live snapshot must reproduce exactly through the frozen-input
  parser; duplicate NCT IDs within a query, conflicting cross-query payloads,
  or disagreement with the recomputed top-level NCT union fail closed. Report
  construction replays once more and rejects nested post-validation mutation.
  Serialized replay, injected transports, and injected clocks cannot
  self-attest live provenance and therefore emit a strict registry count of
  zero. The stock live path uses a non-serializable capability bound to the
  final snapshot digest and actual completion time.
- Same-entity treatment aliases are distinct from broader treatment-class
  discovery terms, and same-entity cancer aliases are distinct from disease
  ancestors. Alias, class, ancestor, and `explicit_component` matches remain
  non-exact; none can become strict canonical entity evidence. Signed subtype state is preserved in
  compact and spaced labels, so `HER2+`, `HER2`, and `HER2 - breast cancer` are
  not interchangeable. Signs are also preserved in biomarker state and
  specimen values such as `positive (+)`/`positive (-)` and `CD3+`/`CD3-`.
  A strict subtype match also requires a separate exact
  structured parent-cancer condition. An embedded or substring parent name is
  not accepted without a versioned, curator-attested parent-ID binding. Strict registry
  status uses only requested axes resolvable from structured registry fields. A
  biomarker keyword does not resolve the typed feature/state/specimen/timepoint
  tuple, and requested regimen, stage, or line of therapy remains unresolved
  until arm assignment and eligibility are parsed; each condition forces the
  strict count to zero.
- `treated_cohort_association` is not predictive evidence without a comparator
  and quantified formal treatment-by-predictor interaction. The cohort-context
  biomarker tuple is distinct from the candidate gene predictor and cannot be
  filled by it. Each patient row binds the predictor gene to a versioned ID,
  feature/state/specimen, compatible measurement type/platform/timepoint, and
  curator identity attestation; the attestation is not external resolver
  authentication. A predictive claim requires versioned active-exposure sets
  and relations, distinct source-native arm IDs, evaluable arm/model counts,
  scale-appropriate event counts, verified estimability and predictor
  variation, a controlled effect scale, and a versioned inference rule with
  consistent departure-from-null evidence. Formal null, inconclusive, and
  unsupported results remain separate. Exact patient context additionally requires
  ontology, regimen, signed subtype, the full typed biomarker tuple, stage, and
  line-of-therapy agreement. Typed biomarker exactness requires an explicit
  `biomarker_axes_observation_status=observed` and positive
  `biomarker_axes_informative_verified` attestation in both context and curated
  row; unobserved or uninformative axes remain unresolved.
  Unverified treatment exposure cannot establish exact patient regimen context,
  and a prognostic-only predictor must be measured pretreatment/baseline.
  Pharmacodynamic and acquired-resistance claims require paired longitudinal
  evidence and are not baseline predictors.
- Preclinical model types answer different questions and are not treated as a
  quality ladder. Treatment activity, natural biomarker association, and
  direct perturbational interaction remain distinct claims; CRISPRa,
  overexpression, RNAi, inhibitor, and CRISPR-KO directions are not treated as
  interchangeable. Gene-specific rows additionally require a versioned,
  curator-attested gene identity. Exact-context and directional-concordance
  counts require matching `perturbed_compartment` and `endpoint_category`. A
  non-unknown direct direction additionally requires a versioned rule, numeric
  effect, sample size, and matching curator-verified inference status;
  resistance, sensitization, and discordant calls require nonzero effect.
  Resistance/sensitization/discordant map to `direction_supported`, neutral to
  `neutral_supported`, and unknown to an explicit inconclusive, unsupported, or
  not-assessed state.
  Concordance is a structured direction adjudication, not a statistical
  confidence claim.
- Alias-only or ontology-version-unresolved compatible context is counted
  separately from explicit subtype, biomarker, regimen, stage, line,
  compartment, or endpoint conflicts. Neither category is exact evidence, and
  no translation-context count is a calibrated probability. When compatible
  and conflicting families both occur without a higher-priority exact or
  independence status,
  `compatible_and_conflicting_context_present` or
  `compatible_and_conflicting_patient_context_present` replaces either
  misleading `_only` status.
- A complete versioned `rank-screen` bundle binds only a candidate TSV that
  carries both ranking fields. It verifies mode, schema/method versions,
  parameters, all output hashes, ordered columns/row count, identifiers, rank
  semantics, and canonical order. This is unsigned internal consistency, not
  producer authentication, and it does not admit report-only
  translation columns into the model. The success-model field deny-list is
  derived from every report-only evidence contract, with a reserved token guard
  as a second layer.
- missingness patterns can encode publication era, repository choice, or input
  mode and therefore require profile- and coverage-stratified analyses.

## Source and protocol boundaries

[BioGRID ORCS](https://orcs.thebiogrid.org/) is a screen-discovery,
structured-metadata, and author-gene-score source. ORCS screen IDs are
provenance/join keys, and ORCS ranks or hit calls are screen evidence, never
`V2/V3/F0/D` labels.

Library references such as Joung et al. establish library design and original
methods only. Their protocol values are not used as defaults for unreported MOI,
coverage, selection, treatment, duration, or comparator conditions in later
historical screens.

## Fairness and scientific bias

Bias is assessed across drug classes, tumor lineages, cell-line ancestry,
library generations, publication year, and data-availability strata. Missing
metadata are reported rather than silently imputed as negative evidence.

## Update policy

Each release freezes:

- data-manifest version;
- source licenses and retrieval dates;
- feature definitions;
- label-adjudication rules;
- split assignments;
- source/raw-data-family deduplication assignments;
- feature-profile and missingness-routing definitions;
- model parameters;
- evaluation report.
