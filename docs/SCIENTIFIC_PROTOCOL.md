# Scientific protocol

## Primary estimand

For a gene \(g\) in screen \(s\), estimate a within-screen ranking score for:

\[
P(Y_{gs}=1 \mid \text{the gene was independently tested}, X_{gs})
\]

where \(Y=1\) denotes a `V2` or `V3` validation outcome and \(X\) contains only
features available before the validation result. Because testing is selected by
authors rather than randomized, this is not automatically identifiable as a
population probability. The first release therefore reports a relative score
and selection-bias sensitivity analyses.

## Unit of analysis

- guide-level observations are retained for QC and aggregation;
- the model row is `gene × screen × contrast × direction`;
- samples are nested in contrasts, contrasts in screens, and screens in studies;
- a validation event is a separate record and can be linked to one or more
  screen rows;
- multiple validation events are never collapsed until a prespecified rule is
  applied.

The four metadata levels have non-interchangeable meanings:

| Level | Definition |
|---|---|
| `study` | Publication, preprint, or author-defined research source. |
| `screen` | Shared cell model, CRISPR library/modality, delivery, and pooled experiment. |
| `contrast` | One treatment-versus-comparator question with a defined dose, schedule, endpoint, and direction. |
| `sample` | One sequenced biological or technical observation assigned to a contrast. |

A change in the shared cell model, library, or perturbation experiment creates a
distinct screen. A change in drug, dose, comparator, or biological time point
creates a distinct contrast unless the prespecified analysis explicitly models
it as part of one longitudinal design. Replicates and time points remain
dependent observations, not independent supervised rows.

## Study eligibility

The V1 compendium includes human pooled CRISPR-Cas9 KO drug-response screens
with a treatment comparator, identifiable drug, cell context, gene mapping, and
authorized/public data. CRISPRa, CRISPRi, and non-drug phenotypes are retained
in the registry but excluded from V1 model fitting. V1 ranking rows must declare
either `resistance` or `sensitization`; neutral, unknown, and discordant
directions remain evidence states rather than prediction queries.

## Processing hierarchy

1. Archive original accession and metadata.
2. Link publication, preprint, repository, author matrix, and BioGRID ORCS
   records into a source/raw-data family before splitting.
3. Normalize and validate the study → screen → contrast → sample hierarchy.
4. Validate sample identities, biological/technical replicate roles, treatment,
   comparator, time reference, and experimental contrast.
5. Count or import guide counts.
6. Run study-level QC and positive-count median-ratio normalization, excluding
   all-zero/uninformative guides from size-factor estimation; retain CPM as a
   named sensitivity analysis rather than the default.
7. Compute at least two established gene-ranking baselines when possible.
8. Retain raw and CNV-corrected outputs.
9. Aggregate reproducible guide-level features.
10. Join versioned experimental-design and context features using stable
    identifiers.
11. Link targeted validation events.
12. Freeze feature timestamps and source-family assignments before splitting
    and training.

## Experimental-design metadata

The design-aware profile records, when reported:

- biological and technical replicate counts;
- control type and identity, including vehicle, untreated, baseline/T0, and
  plasmid-library references;
- library identity, version, scope, guide/gene counts, and guides per gene;
- delivery mode, MOI, and cells-per-guide representation at key stages;
- treatment dose, unit, response-normalized dose basis, schedule, exposure, and
  recovery;
- selection direction and assay endpoint;
- cell-line identifiers and engineered genotype.

Missing metadata remain missing with explicit availability indicators. A
reported zero-day recovery or zero technical replicates is not equivalent to an
unreported value. Raw doses are not compared across unrelated compounds without
a common response-normalized scale.

[BioGRID ORCS](https://orcs.thebiogrid.org/) is a discovery,
structured-metadata, and author-gene-score layer. Its screen scores, ranks, and
hit calls may be used as screen evidence or baselines, with release and
retrieval date recorded, but they are not independent-validation labels. Exact
conditions are verified against primary methods, supplements, repositories, and
author-supplied files.

A library paper documents the library's design and original methods; it does not
prove that every later screen used the same protocol. In particular, Joung et
al. protocol values must not be copied as defaults for unreported MOI, coverage,
selection, treatment, duration, or comparator fields in historical screens.

## CNV policy

CRISPR-EvidenceRank does not create an undocumented universal CNV penalty.
Instead it:

- stores the raw score;
- stores one or more corrected scores;
- records the correction method and version;
- measures the raw-to-corrected change;
- retains absolute copy number and expression as explicit features;
- evaluates conclusions with and without correction-derived features.

For a single screen, an established study-level method such as CRISPRcleanR or
a properly configured MAGeCK CNV-aware model can be used. Joint multi-screen
methods are handled as a separate harmonization track. Method choice and
parameters must be declared in provenance.

## Validation-event adjudication

Two curators should independently annotate high-impact labels. Disagreements
are resolved without seeing model predictions. Each event records:

- perturbation modality and reagent;
- whether perturbation was confirmed;
- number of independent reagents;
- matched cell, drug, dose, and exposure context;
- assay and endpoint;
- phenotype direction;
- effect and uncertainty, when reported;
- rescue or causal reversal;
- technical adequacy;
- exact source locator.

Checksums, distinct review IDs, and distinct curator identifiers preserve the
two review streams but cannot prove cognitive independence. Blinding is a
documented process control, not a cryptographic property. A named human
adjudicator must therefore inspect the source evidence and reviewer differences
before any candidate grade becomes a released label.

`F0` requires a successful perturbation and adequate assay. A failed edit or
missing assay is `T`, not a biological negative.

ORCS hit calls, screen ranks, cross-screen recurrence, and absence from a
paper's validation section cannot create `V2/V3/F0/D`. Validation-protocol
fields are used for adjudication and reporting; they enter the success model
only when the intended protocol was specified before prediction.

## Selection bias

Authors preferentially validate highly ranked, familiar, tractable, or
mechanistically attractive genes. The benchmark therefore uses:

1. a selection model for `testing_status=tested` versus `not_tested`, fitted
   only where the testing denominator is explicitly documented;
2. a success model among tested genes;
3. clipped inverse-propensity weights as a sensitivity analysis;
4. unweighted and weighted results side by side;
5. analyses restricted to studies reporting a reasonably complete testing
   denominator.

The propensity model is not claimed to eliminate unmeasured selection bias.
Rows with `testing_status=unknown` are excluded from fitting the selection
model. The initial IPW analysis also assumes that, conditional on the declared
features, adjudicable outcomes among tested genes are representative. Because
`V1/A/T` may violate that assumption, IPW remains a sensitivity analysis until
an outcome-observation model and adequate reporting denominators are available.
IPW is disabled when an outer training fold lacks at least two independent
inner study groups, lacks both tested and explicitly not-tested candidates, or
an inner training fold contains only one testing-status class. It is never
replaced by an in-sample pseudo-propensity.

## Splits

Required:

- leave-one-source/raw-data-family-out;
- leave-one-study-out or grouped K-fold by study;
- leave-one-compound-out;
- leave-one-drug-class-out;
- cell-line-out;
- lineage-out;
- cold-gene split;
- temporal train/test split based on publication and evidence timestamps.

The primary grouping unit is the transitive source/raw-data family: all
publication and preprint versions, repository mirrors, alternative analysis
methods, and ORCS screen records derived from the same experimental material
remain in one outer fold. A different scoring method does not create an
independent screen.

The held-out private drug-response screen remains completely outside training,
model selection, and Git history. The model and threshold must be frozen before
scoring that case study.

## Metrics

Primary:

- observed V2/V3 yield and recall at 5, 10, and 20 when the full candidate
  universe is ranked;
- observed NDCG@5, @10, and @20 over the full candidate universe;
- macro average precision among screens with both explicit positive and
  explicit negative validation outcomes, averaged within study and then across
  studies;
- uplift over MAGeCK rank, drugZ, effect size, and recurrence baselines.

Secondary:

- precision and NDCG restricted to adjudicated genes, clearly labeled as such;
- Brier score and calibration slope/intercept;
- an explicitly named uncalibrated Brier diagnostic during development;
- recall at the experimental validation budget;
- study-clustered bootstrap confidence intervals;
- failure rate among top-ranked candidates.

Bootstrap query identifiers are tuple-valued, so user-supplied delimiters cannot
merge independent queries. Every interval reports its own effective number of
draws. An interval is withheld when fewer than 80% of requested draws provide a
finite value for that metric, and bootstrapping stops immediately when the
eligible target data do not contain both binary classes.

AUROC is descriptive, not a primary metric.
No score is presented as a calibrated probability until nested calibration and
prospective calibration have been completed.

Untested genes are not biological negatives. Full-universe metrics therefore
use the term **observed success yield**, not precision: they measure recovery of
documented V2/V3 events and are a selection-biased lower bound on the true
validation yield.

## Ablations

Report:

- `screen_only`: gene scores/count-derived guide signal and technical QC;
- `screen_plus_design`: `screen_only` plus experimental-design features;
- `context_aware`: `screen_plus_design` plus versioned artifact/CNV,
  cell/drug/pathway context, and training-only cross-screen recurrence;
- missing-modality and input-mode ablations within each supported profile;
- literature/knowledge graph as a separate component;
- therapeutic priority and novelty as separate endpoints.

This prevents a biologically popular gene from appearing reproducible solely
because it has more papers.

## Auxiliary immune-context analysis

Apply immune-context analysis only after the primary screen-signal ranking and
retain it as a separate report. The query unit is
`gene × modality × compartment × phenotype stratum × analysis tail`.

For every comparison, curate the perturbed compartment, exact cell model,
species and orthology, in-vitro/in-vivo setting, treatment and comparator,
phenotype endpoint, timepoint, native effect direction, endpoint polarity,
source/raw-data families, source snapshot, and transformation date. A sign is
interpretable only after the contrast and endpoint polarity are explicit.
ICRAFT's CRISPRa display inversion is never used as a native biological effect.

Quantitative support is restricted to human evidence or versioned one-to-one
orthology. Collapse reanalyses first through raw-data families and then through
source families; report record, raw-family, source-family, and conservative
independent-component counts separately. Exclude the target screen and every
sibling source/raw family before recurrence. For historical evaluation, use
only rows whose source, provider snapshot, and transformation were all
available by the cutoff.

De novo order-statistic RRA is permitted only within one explicitly selected
recurrence stratum. A rank-list declaration must be verified by observing one
unique gene at every rank from 1 through the declared gene-universe size. At
least two independent provenance components are required. Top-hit subsets,
incomplete lists, mixed tails, or unverified rosters return an abstention with
a null p-value.

Dual-action classification requires an explicit `dual_action_group_id` plus a
versioned, reviewed `dual_action_group_version`, the same perturbation modality
in both compartments, functional CRISPR evidence in both tumor and immune
cells, and no unresolved within-family conflict. Use
`dual_benefit_candidate`, not “validated dual-action target.” scRNA-seq,
DepMap/CCLE, TCGA, and ICB-response associations annotate expression, safety,
or clinical context but do not count as functional support or validation.
The software verifies the declared group version and semantic compatibility;
it cannot authenticate the claimed human review, which remains a provenance
and governance requirement.

For olaparib KO screens, keep the drug-response axis independent. KO enrichment
under olaparib denotes candidate loss-induced resistance; favorable immune
evidence does not turn that gene into an inhibitor target. Report the result as
a PARPi–immune trade-off when the two axes disagree.

## Translation-context analysis

Run translation-context analysis after screen-signal ranking and keep it
outside the reproducibility model. The clinical-trial unit is one unique NCT
record for the treatment/disease question; it has no gene-level outcome. Trial
phase, status, enrollment, planned endpoints, posted aggregate results, and
linked publications remain treatment-level context. Translation-context output
is report-only: it preserves candidate membership and order, does not rerank,
and does not estimate a validation or reproducibility probability.

Treat ClinicalTrials.gov `query.intr` and `query.cond` as discovery searches.
Paginate the role-tagged declared query set and deduplicate NCT IDs. Declare
same-entity treatment aliases separately from broader treatment-class terms,
and same-entity cancer aliases separately from broader disease-ancestor terms.
Subtype aliases must denote the same subtype rather than a disease ancestor.
Alias, class, and ancestor matches expand discovery but can never become strict
canonical entity matches. Never interpret completion of that finite term set as exhaustive
ontology-level recall. Preserve signed subtype identity during query binding and
local matching: `HER2+`, `HER2`, `HER2-`, and spaced signed labels remain
different terms. Preserve `+`/`-` in typed biomarker state and specimen values
as well; for example, `CD3+` and `CD3-` cells cannot be exact matches.

Validate every live document by replaying it through the frozen-input
parser and requiring byte-independent canonical-document equality. Reject
duplicate NCT IDs inside any query, conflicting payloads for one NCT across
queries, and any top-level study set that differs from the NCT union recomputed
from declared queries. Deduplicate only identical cross-query payloads. Replay
again when building the report so nested post-validation mutation fails closed.
Require one before/after version audit per typed query. Only the stock live
transport with its actual completion time receives a digest-bound in-process
capability; serialized replay, injected transports, and injected clocks cannot
emit a strict registry count.

Adjudicate structured intervention, disease/subtype, biomarker, and regimen
relations separately. Exact molecule, molecule as one listed component, drug
class, and unrelated retrieval candidates are not interchangeable. Generic
breast cancer is broader than TNBC, while BRCA mutation and HRD are biomarker
contexts rather than synonyms for TNBC. `explicit_component` remains broader
registry context and can never satisfy strict treatment matching.

Compute strict registry status only over requested axes that the adapter can
resolve from structured intervention and condition fields. A registry
biomarker mention is untyped discovery context: it cannot establish exact
agreement on biomarker feature, state, specimen, and measurement timepoint. A
biomarker-constrained context therefore cannot receive strict typed registry
status merely because a keyword appears. A requested subtype is strict only
when both its signed identity and a separate exact structured parent-cancer
condition are present. Do not infer the parent from an embedded name or
substring without a versioned, curator-attested parent-ID binding. Requested
regimen, stage, and line of therapy remain unresolved and force the strict
registry count to zero until arm assignment and eligibility are parsed;
study-level intervention lists do not resolve them.

A gene-specific patient claim requires actual treatment, compatible disease,
gene measurement, specimen timing, outcome, cohort provenance, and an atomic
effect estimate or qualitative claim in the same cohort. The biomarker term,
feature type, state, specimen type, measurement timepoint, and observation
status are an all-or-none tuple with an explicit
`biomarker_axes_informative_verified` curator decision. An exact typed match
requires `observed` status and positive attestation in both the requested
context and curated row; an
uninformative tuple is unresolved-compatible, not exact. This tuple describes
cohort context and is distinct from the candidate gene predictor; one cannot
fill the other. Bind the predictor to matching `gene_symbol` and
`predictor_gene_symbol`, a versioned gene ID, explicit
feature/state/specimen, a compatible measurement type/platform/timepoint, and
a curator identity attestation. The attestation records curation and is not
external resolver authentication. Reserve
`predictive_interaction` for a pretreatment measurement with versioned canonical
active-exposure sets and relations, distinct source-native arm IDs, evaluable
arm/model counts, scale-appropriate event counts, verified estimability and predictor variation, a
controlled effect scale, and a versioned inference rule with consistent
departure-from-null support. Keep formal null, inconclusive, and unsupported
interaction tests as distinct interpretations and counts.
Require exact ontology, subtype, typed cohort biomarker, regimen, stage, and
line agreement for exact-context status. Missing axes on either side are not
wildcards and remain compatible non-exact. Keep treated-only, prognostic,
pharmacodynamic, acquired-resistance, on-treatment, post-progression,
eligibility-only, and descriptive claims distinct. Longitudinal
interpretations require paired baseline testing and do not create validation
labels. Require prognostic-only predictors to be pretreatment/baseline, and do
not assign exact patient regimen context when treatment exposure is unverified.

Curate preclinical evidence as one
`gene × perturbation × perturbed compartment × model × regimen × comparator ×
endpoint category × endpoint × direction` claim. Every direct-perturbation
claim requires vehicle/baseline-growth control and genotype-by-treatment
testing. Gene-specific evidence requires a versioned, curator-attested gene
identity. A non-unknown direction additionally needs a versioned sign rule,
numeric effect, sample size, and matching curator-verified inference status;
resistance, sensitization, and discordant calls require nonzero effect. Unknown
directions must be explicitly inconclusive, unsupported, or not assessed.
Map resistance/sensitization/discordant to `direction_supported` and neutral to
`neutral_supported`.
Exact preclinical context requires `perturbed_compartment` and
`endpoint_category` to match the target screen. Direction concordance is
evaluated only for exact screen context and the same perturbation modality as
the target screen. This concordance is curator-rule direction mapping, not a
new statistical support test. Treatment activity, natural biomarker association, and
mechanistic evidence do not become CRISPR-KO validation. Cell line,
organoid/ex-vivo, and in-vivo lanes remain separate rather than being combined
into a model-quality score.

Partition non-exact curated evidence into compatible non-exact and conflicting
context. Alias-only or ontology-version-unresolved identity may remain
compatible non-exact evidence. An omitted requested axis is not a wildcard;
narrower evidence remains non-exact. Explicit subtype,
biomarker, regimen, stage, line-of-therapy, perturbed-compartment, or
endpoint-category contradictions are conflicting evidence. Report those
family counts and statuses separately; neither partition may be promoted to an
exact-context claim. If both partitions exist for a gene and no higher-priority
exact or independence status applies, report
`compatible_and_conflicting_context_present` or
`compatible_and_conflicting_patient_context_present`, never an `_only` status.

Bind only the complete versioned `rank-screen` bundle to a candidate TSV that
contains both ranking fields. Verify manifest and method schemas, input mode and
parameters, all output checksums, ordered columns/row count, identifiers,
tail/direction/rank/percentile/neutral semantics, duplicate keys, and canonical
order. This unsigned consistency binding does not authenticate a producer and
does not make translation-context
columns ranking features. Derive the success-model field deny-list from all
report-only evidence contracts and retain the reserved leakage-token guard.

Apply an evidence-availability cutoff and collapse transitive source/raw-data
families. Same-study follow-up belongs in the label registry and is excluded as
prior support. If target-family exclusion is not demonstrated, report
`independence_unverified`. If a source search is incomplete or unavailable, do
not convert the failure into evidence absence.

The current V1 model is applicable only to small-molecule, tumor-cell,
CRISPR-KO drug-response/viability screens. A clinical report may still be
produced for CAR-T or immune-killing contexts, but the core model must return
`not_applicable`.

## Prospective experiment

After model freeze, select genes from:

- high model / high baseline;
- high model / moderate baseline;
- low model / high baseline;
- randomized significant-hit control.

Use multiple sgRNAs, confirm perturbation, repeat the drug phenotype, and add
orthogonal knockdown or rescue when feasible. Report every attempted
validation, including technical and biological failures.
