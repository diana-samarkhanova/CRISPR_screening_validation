# Reporting-standards crosswalk

Updated: 2026-09-07

## Purpose and status language

This file is the working reporting crosswalk for CRISPR-EvidenceRank. It maps
the repository to the Data, Optimization, Model, and Evaluation principles in
the [DOME recommendations](https://doi.org/10.1038/s41592-021-01205-4) and
records what still requires real data, human decisions, or a public release.
It is a project-specific paraphrase, not a replacement for the current journal
form or the publisher's checklist. Authors must refresh it against the target
journal immediately before submission.

Status terms are deliberately strict:

- **present**: auditable code or documentation exists for the development
  system; this does not imply a publication-ready biological result;
- **partial**: some infrastructure exists, but the stated publication evidence
  is incomplete;
- **not implemented**: the requirement is identified but the executable or
  reportable artifact does not yet exist;
- **pending real data**: the item cannot be closed with synthetic fixtures;
- **pending external action**: it requires a human, institutional, experimental,
  or repository action outside the codebase;
- **not applicable**: the reason must be stated rather than leaving the item
  blank.

The baseline reviewed for this crosswalk is commit `f6d0b62`. Every release
candidate must replace that reference with its own immutable commit, data
manifest, environment-lock hash, model hash, split roster, and manuscript
locators.

## DOME operational evidence matrix

The local identifiers below organize this project crosswalk; they are not
presented as official DOME question numbers.

| Area | ID | Reporting evidence required | Current repository evidence | Status | Owner and closure criterion | Manuscript or supplement locator |
|---|---|---|---|---|---|---|
| Data | D-1 | Define the task, prediction unit, eligible screen population, and exclusions. | `SCIENTIFIC_PROTOCOL.md`, `DATASET_CARD.md`, `config/inclusion_exclusion.yaml`, and the study → screen → contrast → sample contracts define a KO drug-response task at `gene × screen × contrast × direction`. | Present for the protocol; pending real data for the study population | Codex can keep contracts and prose synchronized. Curators must freeze and preregister the real inclusion roster before outcome review. | TBD |
| Data | D-2 | Identify every source, version, retrieval date, transformation, family grouping, checksum, and reuse decision. | `data/manifests/data_assets.tsv`, release-specific ORCS manifests, `DATA_PROVENANCE.md`, and `DataAssetRecord` provide the structure. The central manifest currently covers only the development assets entered into it. | Partial | Codex can validate completeness and generate the release table. A human rights review must resolve every unknown and approve each redistributed derivative. | TBD |
| Data | D-3 | Describe label construction, negative-event ascertainment, reviewer independence, agreement, and adjudication. | `GOLD_STANDARD_CURATION.md`, `config/event_review_rubric.yaml`, and checksum-bound event-review contracts implement the workflow. Checked-in event examples are synthetic and no real Gold release exists. | Pending real data | Two independent human curators and a named human adjudicator must complete the frozen real packet; report denominators, disagreements, exclusions, and agreement uncertainty. | TBD |
| Data | D-4 | Explain preprocessing, feature provenance, missingness, and prevention of future-information leakage. | `config/preprocessing.yaml`, `config/features.yaml`, `features.py`, the model allowlist, and model/data cards cover implemented screen/design features. `context_aware` remains a specification. | Partial | Codex can test implemented transformations. Authors must distinguish implemented, planned, and report-only fields and freeze every external-source cutoff. | TBD |
| Data | D-5 | Make train, validation, test, and external evaluation partitions independent and reproducible. | `config/benchmark.yaml` and family-aware grouped splitting keep connected source/raw-data families together for the implemented grouped benchmark. Most declared cold/temporal/library-out analyses are not implemented. | Partial | Codex must implement and export fixed split rosters without family leakage. Curators must complete family identifiers before splitting. | TBD |
| Data | D-6 | State what data and derived labels can be accessed and under which terms. | `REPOSITORY_POLICY.md`, `PUBLIC_RELEASE_CHECKLIST.md`, manifests, and accession-only handling define a fail-closed policy. No public biological benchmark DOI exists. | Pending external action | PI/institutional rights approval and a public, versioned derivative-label release with DOI are required. Source articles and restricted assets must not be republished by assumption. | TBD |
| Optimization | O-1 | Prespecify candidate models, tuning space, selection rule, and the data used for optimization. | Logistic and histogram-gradient-boosting candidates and fixed defaults are implemented. A publication-grade nested selection protocol and frozen tuning space are not. | Partial | Codex must implement nested, family-aware model selection and machine-readable tuning logs; authors must preregister the selection rule. | TBD |
| Optimization | O-2 | Separate model selection from final estimation and report seed sensitivity. | Grouped OOF evaluation and fixed-fold repeated-seed diagnostics are executable. Nested model selection and a real frozen selection result do not exist. | Partial; pending real data | Codex provides the seed manifest/report. Authors must preregister the model-selection rule and never select from the prospective test. | TBD |
| Optimization | O-3 | Address selective observation of tested genes and diagnose overlap rather than assuming weighting removes bias. | The two-stage baseline reports clipped selection-IPW as a sensitivity analysis and exposes overlap diagnostics. Dedicated selection-only covariates and alternative estimators are not implemented. | Partial | Codex can add cutoff-safe selection-only inputs and sensitivity estimators. Authors must report systematic-validation strata and treat failed positivity as a limitation. | TBD |
| Model | M-1 | Specify the algorithm, preprocessing, parameters, weights, software, and feature profile sufficiently for reconstruction. | `modeling.py`, `MODEL_CARD.md`, `pyproject.toml`, and `uv.lock` describe the development baseline. Primary fitting is independent-provenance-group balanced; “unweighted” is not an accurate description. | Present for the development baseline | Codex must export the final parameter/tuning record and keep public column names unambiguous. Authors must cite the immutable release. | TBD |
| Model | M-2 | State how missing and unsupported modalities are handled at fit and prediction time. | Stable missingness indicators and documented routing principles exist for implemented profiles; `context_aware` routing and abstention are not implemented. | Partial | Codex must implement and test any claimed routing/abstention behavior before it appears in Results. | TBD |
| Model | M-3 | Freeze and make the final model artifact independently verifiable. | `freeze-model` creates an atomic integrity bundle that verifies the in-memory training frame against the checksum-bound table and pins model bytes, features, lock file, and full Git revision. `verify-frozen-model` verifies without deserialization and can require an externally recorded manifest hash. No real trained bundle or timestamped prediction deposit exists. | Infrastructure present; pending real data and external action | Codex maintains/tests the bundle contract. Authors must approve the final inputs, externally deposit the reported manifest hash, then deposit the frozen model and predictions before prospective experiments. | TBD |
| Model | M-4 | Make code and environment reproducible from a clean system. | Locked Python environment, CI on Python 3.11/3.12, deterministic generators, repository guards, and wheel installation checks exist. A public tag, archival DOI, and tested container do not. | Partial | Codex can add and test an OCI build. Repository owners must publish the tag/image, record the immutable digest, and mint the archive DOI. | TBD |
| Evaluation | E-1 | Prespecify one primary endpoint and distinguish secondary/exploratory metrics. | The former list of ten primary metrics was removed. `config/benchmark.yaml` now records one candidate metric and an explicit `requires_human_preregistration` status rather than falsely claiming that registration is complete. | Partial; pending external action | PI and statistical lead must choose and preregister the estimand, population, and metric before real-label analysis; Codex then locks/enforces that decision. | TBD |
| Evaluation | E-2 | Compare against relevant simple and domain baselines under identical leakage-safe splits. | One runner now evaluates random (multi-seed), absolute-effect, explicitly direction- and provenance-attested MAGeCK rank/FDR and drugZ when complete, plus the regularized logistic model. Invalid values fail closed; missing/incomplete inputs are reason-coded; unsafe full-corpus recurrence remains blocked. Real MAGeCK/drugZ data and recurrence recomputation are absent. | Partial | Codex maintains shared evaluation and paired family-bootstrap differences. Authors must provide lawful compatible baseline inputs, select the primary comparator, and freeze versions. | TBD |
| Evaluation | E-3 | Report class prevalence, metric denominators, undefined cases, uncertainty, and repeated-seed variability. | The checksum-bound benchmark report emits prevalence, normalized AP, informative/unsaturated and single-class query counts, fractional cutoff-tie handling, direction strata, seed outcomes, source-family intervals, paired common-roster baseline differences, and strict JSON. The resampling cluster and study-macro aggregation identifier are separate. Bootstrap intervals currently condition on fitted OOF scores. | Present for development; pending full-uncertainty design and real data | Codex must add refit-aware uncertainty if required. The analysis plan must define interval, seed, and multiplicity handling. | TBD |
| Evaluation | E-4 | Demonstrate metric resolution and include negative controls. | Executable tests distinguish ideal, intermediate, and inverted top-k order and enforce precision cutoff saturation. Full-pipeline label-permutation controls are not implemented or reported. | Partial | Codex can add full-pipeline permutations; authors must preregister them and report the real-split floor. | TBD |
| Evaluation | E-5 | Report calibration, subgroup/generalization analyses, and error analysis. | Diagnostic slope/intercept, reliability bins, direction-stratified metrics, and within-query feature-invariance auditing are implemented. Drug/lineage/cell/library/literature/missingness generalization and systematic error-case outputs remain open. | Partial; pending real data | Codex can implement remaining analyses. Domain experts must interpret failure modes without turning post hoc strata into confirmatory claims. | TBD |
| Evaluation | E-6 | Reserve an external or prospective test and document that it was untouched during development. | Repository policy reserves a private KO screen and separates non-KO modalities, but no power analysis, deposited prediction hash, or prospective result exists. | Pending external action | Experimental team/statistician must approve power and assay design. Authors must deposit predictions before experiments and disclose dates. | TBD |

## TRIPOD+AI applicability

The [TRIPOD+AI statement](https://doi.org/10.1136/bmj-2023-078378) is designed
for transparent reporting of clinical prediction models. CRISPR-EvidenceRank's
current unit is a gene within an experimental screen; it has no individual
patient predictors, patient outcome, clinical intervention assignment, or
clinical decision rule. Therefore TRIPOD+AI is an **adapted transparency
cross-check**, not the primary reporting standard, and the project must not
claim unqualified “TRIPOD+AI compliance.” DOME is the primary ML-reporting
crosswalk for the planned methods paper.

| TRIPOD+AI topic used as an adapted check | How it maps here | Current status |
|---|---|---|
| Identify model purpose and development/evaluation status in title and abstract. | State post-screen orthogonal-reproducibility ranking, KO-only scope, and whether results are synthetic, retrospective, or prospective. | Pending manuscript |
| Describe source population and flow. | Replace participant flow with source-family → study → screen → eligible contrast → candidate/event flow, with reason-coded exclusions. | Pending real data |
| Define outcomes and predictors without hindsight. | Report the frozen `V3…U` ontology, independent review, availability cutoffs, and feature profiles. | Partial |
| Justify sample size and missing-data handling. | Report families, screens, candidates, events, class prevalence, informative groups, power, and unsupported modalities. | Pending real data |
| Separate development, tuning, internal validation, and external validation. | Use family-aware nested development and a deposited, untouched prospective screen. | Pending implementation and experiment |
| Report performance with uncertainty and calibration where meaningful. | Report the preregistered ranking endpoint, baselines, family-cluster intervals, seed sensitivity, and calibration without calling relative scores probabilities prematurely. | Pending real data |
| Provide the final model and enough detail for use. | Publish code, lock, container digest, trained artifact hash, input contract, and abstention limits. | Pending public release |
| Discuss limitations, fairness, and potential harms. | Cover reporting/selection bias, rare failures, lineage and gene-popularity imbalance, misuse as clinical evidence, and invalid pooling of KO with CRISPRa/CRISPRi. | Partial |
| Clinical participant, treatment-decision, and patient-outcome items. | There are no participants or patient-level predictions in the current task. ClinicalTrials.gov metadata remain a separate report-only treatment-by-cancer layer. | Not applicable; rationale required |

If a future project version predicts a patient-level clinical outcome, that is
a new target and governance regime. It requires a fresh protocol, patient-data
ethics and privacy review, clinical validation, and a complete current
TRIPOD+AI assessment; this crosswalk cannot be carried over automatically.

## Freeze procedure for a manuscript supplement

Before submission:

1. replace every `TBD` locator with a manuscript section, figure, table, or
   supplement reference;
2. change a status to **present** only when the referenced immutable artifact
   exists and the claim has been rerun on the release candidate;
3. record the release commit, `uv.lock` SHA-256, data-manifest SHA-256, split
   roster SHA-256, model SHA-256, container digest, and DOI;
4. obtain curator, statistician, PI, and rights-review sign-off for their owned
   rows;
5. archive this completed crosswalk with the other supplementary methods files.

An empty locator, synthetic-only result, planned configuration entry, or
unchecked release task cannot be reported as a closed publication item.
