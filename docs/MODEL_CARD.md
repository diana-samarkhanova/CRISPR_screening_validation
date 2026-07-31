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
