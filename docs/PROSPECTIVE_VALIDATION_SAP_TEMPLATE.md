# Prospective validation: statistical analysis plan template

Status: **template only — not approved or preregistered**

This document must be completed, approved by the principal investigator and
statistical lead, and timestamped before the private-screen candidate outcomes
are examined. Filling this file after seeing outcomes does not create a
prospective study.

## 1. Frozen objects

Record the immutable identifiers below.

| Object | Required identifier | Value |
|---|---|---|
| Eligible public training roster | SHA-256 and DOI/deposition version | TBD |
| Gold validation-event registry | SHA-256 and DOI/deposition version | TBD |
| Inclusion/exclusion policy | version and SHA-256 | TBD |
| Feature registry | version and SHA-256 | TBD |
| Split assignment manifest | SHA-256 | TBD |
| Code | full Git commit SHA | TBD |
| Environment | `uv.lock` SHA-256 and container digest if used | TBD |
| Frozen model bundle | model-manifest and model SHA-256 | TBD |
| External model-manifest anchor | deposited manifest SHA-256 and timestamp | TBD |
| Private-screen input | governed asset ID and SHA-256 | TBD |
| Deposited predictions | deposition ID, timestamp, and SHA-256 | TBD |

Use `crispr-evidencerank freeze-model` only after every upstream object and the
analysis rule below have been approved. When freezing from the repository,
supply `--current-checkout-path` and write the output outside that checkout; the
command rejects a dirty tree or a HEAD that differs from `--code-revision`.
The bundle embeds immutable copies of the feature registry and environment
lock, but does not copy the training table. The command creates an integrity
bundle; it does not create an independent timestamp or institutional approval.
Deposit the reported manifest SHA-256 outside the bundle. Final verification
must supply that value through `verify-frozen-model
--expected-manifest-sha256`; verification without it checks internal
consistency only, not the identity of a manifest that could have been replaced.

## 2. Scientific estimand

- Target population of genes: **TBD**
- Screen, contrast, cell model, treatment, dose, schedule, comparator, and
  endpoint: **TBD**
- Primary phenotype direction: **TBD**
- Primary estimand in words: **TBD**
- Analysis population (intention-to-test, perturbation-confirmed, or both):
  **TBD**
- Handling of technical failures (`T`), missing outcomes, and exclusions:
  **TBD**

Do not treat failed perturbation delivery or an unevaluable assay as `F0`.
Technical failures remain `T`; `F0` requires a confirmed perturbation and an
evaluable null result under the Gold rubric.

## 3. Comparators and candidate sampling

- Preregistered primary comparator: **TBD**
- Secondary comparators: **TBD**
- Candidate budget and overlap expected between lists: **TBD**
- Sampling design: representative random set, model-preferred/baseline-preferred
  discordant strata, or a prespecified hybrid: **TBD**
- Sampling probabilities/weights, if discordant enrichment is used: **TBD**

A discordant set estimates performance in the model–comparator disagreement
region. It does not by itself establish superiority over the full candidate
universe. Include a representative sentinel stratum or use prespecified
sampling weights if the intended claim concerns the full ranking.

## 4. Assay and masking

- Independent reagents or orthogonal perturbation: **TBD**
- Perturbation-confirmation assay and pass threshold: **TBD**
- Biological and technical replicate structure: **TBD**
- Positive, negative, vehicle, untreated, and baseline controls: **TBD**
- Plate/block randomization and batch factors: **TBD**
- Blinding of assay execution, outcome calling, and Gold adjudication: **TBD**
- Attrition and repeat-assay rules: **TBD**

All candidate outcomes, including null, discordant, and technical results, must
enter the frozen review packet. The same `V3/V2/V1/F0/D/A/T/U` rubric used for
historical evidence applies prospectively.

## 5. Power and sample size

The final calculation must match the actual design. A two-independent-
proportions table is insufficient when candidates are paired, selected for
disagreement, share pathways, or are clustered by plate.

Required inputs:

- minimally relevant candidate-minus-comparator effect: **TBD**
- expected success proportions or paired discordance probabilities: **TBD**
- intraplate/pathway correlation assumptions: **TBD**
- technical-failure and attrition rates: **TBD**
- one- or two-sided type-I error and multiplicity rule: **TBD**
- desired power: **TBD**
- simulation code, seed roster, and result hash: **TBD**

Report power across plausible nuisance-parameter scenarios, not only the most
favorable one. The user/experimental team must supply the budget, plate layout,
replicate plan, expected failure rate, and smallest effect worth detecting.

## 6. Analysis

- Primary metric and direction of improvement: **TBD**
- Paired source/plate-aware interval or test: **TBD**
- Multiplicity rule for secondary baselines/endpoints: **TBD**
- Sensitivity populations: **TBD**
- Abstention/infeasibility rules: **TBD**
- Exact software command and output-manifest location: **TBD**

Model and comparator must be evaluated on the same eligible roster. Report the
paired difference and its uncertainty; do not infer a difference from overlap
or non-overlap of two separate confidence intervals.

## 7. Approval and timestamp

| Role | Name / persistent ID | Decision | Date/time UTC | Signature or governed record |
|---|---|---|---|---|
| Principal investigator | TBD | TBD | TBD | TBD |
| Statistical lead | TBD | TBD | TBD | TBD |
| Experimental lead | TBD | TBD | TBD | TBD |
| Data-rights/governance approver | TBD | TBD | TBD | TBD |

External preregistration/deposition service: **TBD**

Immutable registration ID: **TBD**
Public or embargoed access terms: **TBD**
