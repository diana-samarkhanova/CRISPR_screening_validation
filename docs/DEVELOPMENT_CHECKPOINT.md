# Development checkpoint — v0.4.0.dev0

Updated: 2026-08-31

The v0.4 checkpoint retains the v0.2 design contract described below and adds
checksum-bound screen, immune-context, and clinical treatment-by-cancer report
engines. These are development/reporting components; the real harmonized
training set, adjudicated benchmark, pretrained model, and calibrated
validation probability still do not exist.

Release verification on Python 3.11 passes 353 tests at 88.24% statement
coverage, repository and formatting guards, the smoke benchmark, deterministic
schema/fixture regeneration, and an installed-wheel end-to-end run. The
synthetic clinical fixture contains five non-patient registry rows and one
checksum-pinned source asset.

## Executable baseline inherited from v0.1

- strict data contracts and JSON Schemas for studies, screens, samples,
  gene-level scores, validation events, and external evidence;
- event labels `V3/V2/V1/F0/D/A/T/U`, with explicit biological failures kept
  separate from technical failures and untested genes;
- wide sgRNA count-table validation and guide-to-gene feature extraction;
- a MAGeCK RRA gene-summary adapter;
- a two-stage, selection-aware reproducibility baseline;
- logistic and histogram-gradient-boosting candidate models;
- study-grouped out-of-fold predictions and within-screen ranking metrics;
- collision-safe tuple query identifiers and sparse-bootstrap diagnostics;
- deterministic synthetic data, 73 executable contract/feature/model tests,
  smoke tests, and a 27-study seed manifest.

## v0.2 experimental-design specification

The v0.2 configuration and documentation add:

- the normalized study → screen → contrast → sample hierarchy;
- explicit source-family and raw-data-family deduplication before splitting;
- `screen_only`, `screen_plus_design`, `context_aware`, and separate
  `selection_model` feature profiles;
- structured treatment, comparator, replicate, library, representation, and
  endpoint metadata with explicit missingness;
- BioGRID ORCS as an official discovery/metadata/author-score layer, not a
  validation-label source;
- deterministic ORCS index triage with auditable rule outcomes and conservative
  `exclude`/`metadata_only` decisions;
- policy-derived curated readiness requiring the complete versioned rule set,
  plus JSON-Schema/runtime parity and cross-table integrity checks;
- a prohibition on copying unreported historical conditions from the Joung or
  other library-reference protocols;
- AssayBench as an adjacent pre-screen gene-ranking comparator rather than a
  post-screen validation benchmark.

These additions define the v0.2 data and benchmark contract. Real-study
harmonization and prospective validation remain the release gate.

## Synthetic verification

The synthetic fixture contains 8 studies, 8 screens, 16 direction-specific
ranking queries, 480 gene-screen rows, 1,920 guides, and 112 validation events.
It is designed only to verify software behavior.

| Candidate model | Global PR-AUC | Study-macro PR-AUC | Observed NDCG@10 | Observed success yield@5 | Uncalibrated Brier |
|---|---:|---:|---:|---:|
| Regularized logistic regression | 0.6951 | 0.7650 | 0.7696 | 0.4375 | 0.2352 |
| Histogram gradient boosting | 0.6436 | 0.7676 | 0.6734 | 0.4750 | 0.2565 |

These values are not biological performance estimates. On this fixture, the
transparent logistic baseline is the checkpoint winner; the boosted model is
not promoted merely because it is more complex.

Selection-IPW is available in the fixture, but 97.3% of labeled training rows
hit the upper propensity clip. This deliberately visible overlap failure means
the IPW result is not treated as confirmatory; the study-balanced, non-IPW
result remains primary.

The final checkpoint audit also verifies that IPW is disabled for single-class
inner folds, non-model directions are rejected before fitting, orphan validation
events fail registry integrity checks, and each bootstrap metric reports its own
effective draw count.

## Scientific boundary

No public biological screen compendium has yet been harmonized into a training
release. The current model output must therefore be described as a relative
reproducibility score, not a calibrated probability of experimental success.

The held-out private drug-response screen is excluded from training and model
selection. Its biological context and results remain outside Git history and
are reserved for evaluation after the public-data model and selection rules
are frozen.

The ORCS pilot now records a completed checksum-bound second review for all ten
queue ranks and an accession-level map for all 24 `SRP158611` amplicon runs.
The authenticated full comparison contains 20 gene-level rows: 17 provisional
agreements, one evidence-level disagreement, and two single-curator
observations. Human adjudication remains pending. A public Findlay
supplementary matrix was also identified; its two-replicate count columns
conflict with triplicate wording in the methods. These discoveries improve
provenance but do not create training rows: `benchmark_ready` remains zero.

## Next gate

The next release gate is a manually adjudicated pilot containing:

1. at least 6 studies with guide counts or raw reads;
2. exact study, screen, contrast, sample, drug, cell-line, comparator, and
   replicate metadata with explicit missingness;
3. event-level successful, failed, discordant, technical-failure, and untested
   annotations with exact source locators;
4. MAGeCK/drugZ and effect-size baselines;
5. source/raw-data-family-held-out evaluation with clustered uncertainty;
6. profile-specific benchmarks for screen-only, screen-plus-design, and
   context-aware
   inputs.
