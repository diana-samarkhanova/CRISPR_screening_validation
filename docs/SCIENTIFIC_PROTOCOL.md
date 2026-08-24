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

## Prospective experiment

After model freeze, select genes from:

- high model / high baseline;
- high model / moderate baseline;
- low model / high baseline;
- randomized significant-hit control.

Use multiple sgRNAs, confirm perturbation, repeat the drug phenotype, and add
orthogonal knockdown or rescue when feasible. Report every attempted
validation, including technical and biological failures.
