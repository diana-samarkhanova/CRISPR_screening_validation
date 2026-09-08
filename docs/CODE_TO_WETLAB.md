# From code to prospectively evaluated targets

Status: executable development workflow; no real trained biological model or
new experimentally validated targets are released by this checkpoint.

## Development data and model selection

Training requires independent human-adjudicated events linked to harmonized
screen features. Freeze the development roster and all source/raw-data family
assignments. Exclude the reserved case and its related families. Previously
observed outcomes do not become an untouched test when a file is renamed.

Choose candidate configurations, feature profiles, inner selection metric,
outer evaluation metric, and comparator before inspecting outer results.
`examples/synthetic/model_grid.json` is a software example, not an approved
primary-metric decision. The human decision in `config/benchmark.yaml` remains
pending.

```bash
crispr-evidencerank nested-benchmark \
  --table /secure/public_development.tsv \
  --model-grid /secure/approved_model_grid.json \
  --outer-folds 5 --inner-folds 3 --output-dir /secure/nested_benchmark

crispr-evidencerank select-deployment-model \
  --table /secure/public_development.tsv \
  --model-grid /secure/approved_model_grid.json \
  --inner-folds 3 --output-dir /secure/deployment_selection
```

Nested evaluation uses outcome-independent grouped splits. Every inner fit,
imputation and configuration choice uses outer-training families only. Any
study/source/raw link keeps connected rows together. Inadequate families,
single-class training folds and undefined selection metrics fail explicitly.
The report records all configurations, folds and fitted preprocessing. Requested
folds are capped by available families; actual counts are reported. HGB automatic
row-level early stopping is disabled.

Outer predictions evaluate the selection procedure, which may choose different
models in different folds. `select-deployment-model` uses the same inner rule on
development-only data. Its score is a selection criterion, not an unbiased
performance estimate. Do not choose the best outer model or majority outer
winner. Fit the selected configuration on that same development roster using
`freeze-model`, its exact features and `--logistic-c`. Model bundles preserve
parameter/feature order, dependencies and hashed training-family identities.

## Frozen inference

`predict-frozen-model` requires an external model-manifest SHA-256, explicit
trust in the local pickle, an outcome-free feature table, a timezone-aware
timestamp and a full code revision. It checks runtime compatibility before
loading. Current inference supports CRISPR_KO resistance/sensitization only.

Required identifiers: `study_id`, `screen_id`, `contrast_id`, `gene_symbol`,
`phenotype_direction`, `perturbation_modality`, `source_family_id`,
`raw_data_family_id`. Opaque identifiers retain leading zeros.

Supply the model's numeric features. Raw median/mean guide LFC requires explicit
`positive_lfc_means`, and `signal_direction` for deriving aligned features.
Native signs are mapped using the declared biology. Older internal model API
calls without that column retain the legacy positive=resistance convention;
the new inference CLI requires the declaration for raw LFC. Duplicate rows,
infinite values, all-missing supported features, and outcome/clinical columns
are rejected. Feature missingness is reported and can be capped by a
prespecified `--max-missing-feature-fraction`. This is not a full distribution-
shift detector.

Default inference rejects declared study/source/raw overlap with training and
requires a complete training inventory. Older bundles without this inventory
can be used in explicit exploratory mode only. `--allow-training-overlap`
marks output exploratory and makes it ineligible for prospective allocation.
Identity checks cannot detect undisclosed reuse or a human's prior knowledge.

For wet-lab allocation also supply `candidate_role` (candidate,
positive_control, negative_control), `prior_outcome_exposure` (not_seen, seen,
unknown), and `screen_signal_rank` before predicting. These are upstream
declarations, not proof of blinding. Link the `rank-screen` bundle through
`--baseline-report-dir` and `--expected-baseline-manifest-sha256`. Every rank
must match the pinned report for the same gene/screen/contrast/direction. Native
MAGeCK ranks and count-derived ranks retain their distinct method labels.
Report subsets are allowed and explicitly disclosed.

Output: `predictions.tsv`, feature diagnostics, copied model manifest and
`prediction_manifest.json`. Scores in 0–1 are relative ranking scores, not
validated probabilities of experimental success.

## Freeze allocation before new outcomes

`freeze-target-selection` consumes exact prediction bytes, linked model and
prediction manifests, and the analysis plan; hashes are mandatory. It rechecks
training-family disjointness and requires model <= prediction <= allocation
timestamps. Hashes and timestamps do not create external registration.

One packet contains one screen/contrast/modality/direction. The complete union
of model top-k and baseline top-k receives experiments, with overlap tested
once, plus a seeded sample from the remaining eligible candidates. Maximum
comparison size is `2*k + random_comparison_n`; actual size depends on overlap.
Insufficient candidates/pool capacity fail without shrinking the design.
Exact rank ties use canonical gene order, declared before outcome collection.

Seen/unknown prior outcomes stay outside the prospective comparison. Controls
remain separate for assay QC. Untested genes never become biological negatives.
The random remainder sample plus complete top-k union and inclusion probabilities
supports a full-roster analysis only with the prespecified weighting and
missingness assumptions. Top-k or disagreement-only sampling cannot establish
superiority across the genome.

The analyst receives full allocation and its provenance. `assay_roster.tsv`
contains randomized opaque assay IDs without model scores or selection strata.
Give it through a separate workspace; its row order does not replace biological
batch/plate/block randomization. `verify-target-selection` re-derives allocations
from frozen inputs without loading a model. An external manifest hash is needed
to establish identity rather than internal consistency alone.

## Experimental work and remaining gaps

Use [the Russian wet-lab plan](WET_LAB_VALIDATION_PLAN_RU.md) and
[the analysis-plan template](PROSPECTIVE_VALIDATION_SAP_TEMPLATE.md).

| Executable now | Still requires data or further implementation |
|---|---|
| Nested and final development-only model selection | Real adjudicated training/benchmark roster |
| Frozen-model verification, inference and provenance | Approved primary metric and power calculation |
| Paired target allocation and masked assay roster | Independent human review and new wet-lab outcomes |
| Conditional calibration diagnostics | Nested fitted calibration and external calibration |
| Screen-only and screen-plus-design features | Reviewed CNV/off-target/guide-efficacy/context features |
| Family-held-out evaluation | Compound/class/cell/lineage/library/temporal evaluation |

Also pending: label-permutation controls, external replication, fuller outcome-
missingness/selection-bias analysis, and archival release. A defensible biological
target list requires a defensible real model. Green tests establish software
behavior, not biological accuracy or readiness for a top-journal publication.

## Installed CLI synthetic verification

```bash
python scripts/smoke_prospective.py --output-dir /tmp/prospective-smoke
```

This runs nested evaluation, development selection, model freezing, MAGeCK
reporting, inference, target allocation and offline verification. All candidates
and the plan are synthetic; the generated model is not for biological use.
The same chain runs in both CI Python versions.
