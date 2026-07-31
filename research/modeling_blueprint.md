# CRISPR-EvidenceRank: statistical modeling and benchmark blueprint

**Scope:** prospective orthogonal reproducibility of hits from human pooled CRISPR–Cas9 knockout drug-response screens  
**Status:** design specification, 2026-07-30  
**Primary deployment question:** given a completed screen and a limited validation budget, which gene–screen hits should be tested first?

## 1. Executive decision

The system should not initially claim to estimate an unconditional, universally valid “probability that a gene is real.” The scientifically defensible target is narrower:

> Rank gene–screen instances by the probability that the screen-predicted direction will reproduce under a prespecified orthogonal validation protocol in the same drug and cellular context.

Three design choices are non-negotiable:

1. **The supervised unit is a gene within a screen contrast, not an sgRNA.** Guides, replicate samples, time points, and related contrasts contribute evidence and uncertainty features; they are not independent labeled observations.
2. **Untested genes are not negatives.** Published validation is selectively observed because authors choose which hits to test and may preferentially report successes. Explicit adequate failures are the only clean negative labels.
3. **Retrospective and prospective performance are different estimands.** Retrospective performance can support a relative reproducibility score, but a calibrated probability for arbitrary screen hits requires a prospectively selected, standardized validation panel.

The recommended model family for the first useful release is a regularized logistic baseline plus a gradient-boosted tree classifier, with a screen-wise learning-to-rank model as a secondary analysis. The final user-facing output should keep separate:

- `screen_reproducibility_score`;
- `artifact_risk`;
- `context_support`;
- `evidence_coverage`;
- `uncertainty`;
- and, only after prospective calibration, `estimated_validation_probability`.

Novelty, druggability, safety, and therapeutic priority should remain separate decision layers. They are not synonyms for experimental reproducibility.

## 2. Target estimand and data hierarchy

### 2.1 Indexing

Let:

- \(j\) index a publication or study;
- \(q\) index a screen contrast within study \(j\);
- \(g\) index a gene;
- \(k\) index an sgRNA;
- \(r\) index a biological replicate or sample;
- \(v\) index an orthogonal validation event.

A **screen contrast** is a prespecified comparison such as:

`cell line × drug × dose/schedule × duration/time point × treated/control design × screen direction`.

The principal prediction unit is:

\[
i = (j,q,g),
\]

called a **gene–screen instance**. A gene appearing in ten different drug/cell-line screens creates ten biologically distinct instances.

The hierarchy is:

```text
study
└── screen experiment
    ├── contrast (drug × cell line × time point × direction)
    │   ├── gene–screen instance
    │   │   ├── guides
    │   │   └── validation events
    │   └── replicate/sample counts
    └── shared library and experimental metadata
```

### 2.2 Latent target

For each instance define:

\[
Y_i^\star =
\begin{cases}
1, & \text{predicted-direction phenotype reproduces under the target protocol},\\
0, & \text{it does not reproduce or is significantly discordant}.
\end{cases}
\]

The desired deployment quantity is:

\[
p_i = P(Y_i^\star=1 \mid X_i,\ \mathcal{P}),
\]

where \(X_i\) contains only information available before orthogonal validation and \(\mathcal{P}\) is a sufficiently specified validation protocol. Protocol dependence matters: a gene may reproduce with complete knockout in one line but not with partial knockdown, a different drug exposure, or a different endpoint.

The primary v0 estimand must therefore be phrased as:

> relative reproducibility among gene–screen instances with adequate, reported validation in the curated historical domain.

The v1 prospective estimand can be:

> probability of V2-or-better reproduction under the project’s standardized validation protocol among eligible human CRISPR-KO drug-response hits.

### 2.3 Avoiding pseudo-replication

- Do not make each sgRNA a supervised row carrying the gene’s validation label.
- Do not make each replicate pair a separate supervised row.
- Do not randomly split time points, cell lines, or contrasts from the same study between train and test in the main benchmark.
- Do not count multiple validation assays on the same gene–screen instance as independent model outcomes.

Guide- and replicate-level observations should instead generate features such as concordance, dispersion, leave-one-guide-out stability, and model standard errors. Performance confidence intervals must resample at the study level, not at the gene-row level.

Dependence is partly nested and partly crossed: guides are nested in genes and contrasts, while the same gene and drug can recur across independent studies. The primary uncertainty analysis should cluster on study; a sensitivity analysis should additionally use a two-way study/gene resampling scheme or cold-gene evaluation to show that recurrent well-known genes are not creating artificial precision.

## 3. Cohort definition

### 3.1 v0 inclusion criteria

Include studies that meet all of the following:

- human cells;
- pooled CRISPR–Cas9 knockout or loss-of-function nuclease screen;
- in vitro drug-treated arm with a matched untreated/vehicle/control arm;
- survival, proliferation, competitive fitness, or abundance-based drug-response endpoint;
- genome-wide or broad subgenomic library with identifiable guide-to-gene mapping;
- sufficient metadata to reconstruct at least one treated-versus-control contrast;
- gene-level results or guide counts available;
- full paper and supplements available for validation curation.

Record but initially exclude from the primary model:

- CRISPRa and CRISPRi;
- base/prime editing;
- single-cell readout screens;
- in vivo screens;
- combination-treatment screens without an interpretable comparator;
- screens whose primary phenotype is differentiation, infection, trafficking, morphology, or reporter activity rather than drug-conditioned fitness;
- screens with no recoverable universe of tested genes.

These can become later, explicitly separate domains. Mixing them into v0 would make the outcome and artifact structure incoherent.

### 3.2 Candidate universe

For each contrast, preserve:

1. all genes represented in the analyzed library;
2. whether each gene passed minimum coverage/QC;
3. the complete ranking, not only significant hits;
4. the authors’ reported candidate subset;
5. which candidates were explicitly selected, attempted, technically evaluable, and reported.

The model may be deployed on all QC-eligible genes, but benchmark results should be shown both for:

- the entire QC-eligible universe;
- a prespecified “candidate zone,” such as the top \(K\), top percentile, or a false-discovery/effect-size threshold established without seeing validation outcomes.

This distinction prevents a trivial abundance of null genes from dominating evaluation.

## 4. Validation-label ontology

Labels belong to a **validation event**, then are adjudicated into a gene–screen outcome. They must never be inferred from whether a gene appears in the main text.

| Code | Meaning | Minimum evidentiary rule | Primary binary use |
|---|---|---|---|
| `V3` | Causal validation | Meets V2 and includes a successful rescue/complementation, or an equivalently strong causal reversal/epistasis test that links perturbation to the predicted drug phenotype | Positive |
| `V2` | Reproduced | Perturbation is verified; at least two independent reagents or an orthogonal perturbation strategy is used; appropriate controls are present; a quantitative drug-response phenotype reproduces in the predicted direction in independent experiments | Positive |
| `V1` | Supportive | Predicted-direction evidence exists but only one reagent, incomplete perturbation confirmation, a proxy endpoint, insufficient independent replication, or another limitation prevents V2 | Missing from primary binary task; graded relevance in sensitivity analysis |
| `F0` | Adequate failed validation | Perturbation and assay are technically adequate, but the predicted phenotype is absent; preferably the confidence interval excludes a prespecified minimum meaningful effect | Negative |
| `D` | Discordant | Perturbation is adequate and a statistically/biologically supported effect occurs in the direction opposite to the screen prediction | Negative, retained as a distinct error type |
| `A` | Ambiguous | Mixed reagents, mixed contexts, contradictory assays, insufficient reporting, or unresolved interpretation | Missing |
| `T` | Technical failure | Editing/knockdown, viability window, assay, controls, or another technical component failed, so no biological conclusion is possible | Missing; not negative |
| `U` | Untested/unreported | No adequate validation outcome is reported | Unlabeled; never negative |

### 4.1 Important adjudication rules

- **Non-significance is not automatically `F0`.** If the study is underpowered or reports no effect size/interval, use `A` unless the assay clearly had capacity to detect a meaningful effect.
- **Perturbation failure is `T`, not `F0`.**
- **A different cell line or a different drug is not automatically a target label.** Add `context_match = exact / close / external`. The primary outcome uses exact context, or a narrowly prespecified close-context rule. Other contexts become external evidence features.
- **A pharmacologic inhibitor is not equivalent to gene knockout** unless specificity and causal interpretation are strong enough for a predefined orthogonal-validation rule.
- **Clinical association, mutation, expression correlation, or pathway membership is not a validation label.** It can be an external feature.
- **Direction must be normalized.** Store screen-predicted resistance/sensitization and validation direction separately before deriving concordance.
- **The historical validation protocol is not automatically a predictor.** It is observed after candidate selection and may not be known when a user asks for ranking. Use it for outcome definition and stratified analysis unless the deployment interface explicitly asks the user to choose a protocol before prediction.

### 4.2 Multiple validation events

Keep every event in a `ValidationEvent` table, but create one adjudicated outcome per gene–screen instance:

- `concordant_success`: one or more V2/V3 events and no adequate discordant event;
- `concordant_failure`: one or more F0 events and no V2/V3 event;
- `directional_discordance`: at least one D event;
- `mixed_or_ambiguous`: adequate events disagree in a way not resolved by protocol/context;
- `not_evaluable`: only V1/A/T/U.

The primary binary training set contains `concordant_success` versus `concordant_failure` plus `directional_discordance`. A stricter sensitivity analysis uses V3/V2 versus F0 only and excludes D; another treats D as a separate multinomial class.

Do not use the number of validation experiments, number of figures, or eventual validation grade as a predictor. These are post-outcome variables.

### 4.3 Curation reliability

Two curators should independently label a stratified subset and all difficult cases while blinded to model scores. Record:

- raw agreement;
- weighted agreement for V3/V2/V1/F0/D/A/T/U;
- disagreements and adjudication reason;
- curator confidence;
- exact source location: DOI, figure/table/supplement, page or panel.

A benchmark release should include the written rubric and a frozen adjudication log.

## 5. Selective validation, publication bias, and positive–unlabeled data

### 5.1 The missing-outcome mechanism

Define:

- \(S_i=1\): authors state or document that the gene was selected/attempted for validation;
- \(R_i=1\): an adequate biological outcome is observable and adjudicable;
- \(Y_i\): observed V2/V3 versus F0/D outcome when \(R_i=1\).

Often the literature only reveals \(R_i\), while an unreported attempt is indistinguishable from no attempt. Consequently:

\[
P(R_i=1 \mid X_i,Y_i^\star)
\]

is unlikely to be constant. Authors tend to select high-ranked, plausible, tractable, or novel hits, and failed experiments may be less likely to be reported. This is verification/selection bias, potentially missing not at random.

### 5.2 Why naïve PU learning is insufficient

Classic positive–unlabeled learning under the selected-completely-at-random assumption treats labeled positives as a random subset of all positives. That is implausible here. Elkan and Noto formalized the SCAR setting; Bekker and Davis developed a weaker selected-at-random (SAR) formulation in which labeling propensity can depend on observed attributes. In this project:

- V2/V3 are observed positives;
- F0/D are observed negatives;
- U contains unknown positives and negatives;
- selection depends strongly on original screen rank and biological priors.

Therefore this is closer to **positive–negative–unlabeled data with covariate-dependent verification** than to ordinary PU data. PNU and SAR-PU methods are useful sensitivity analyses, not automatic solutions.

### 5.3 Recommended two-process analysis

Fit and report two distinct models:

1. **Selection/reporting model**

   \[
   \hat e_i=P(R_i=1\mid X_i^{\mathrm{pre}})
   \]

   using every QC-eligible gene in each screen and only features plausibly available to authors before validation.

2. **Reproducibility model**

   \[
   \hat m_i=P(Y_i=1\mid X_i,R_i=1)
   \]

   using adequate observed outcomes.

The first model reveals which features drive historical choice/reporting. The second addresses validation among observed attempts. Comparing their feature attributions is essential: if the “reproducibility” model is effectively a copy of the selection model, it has not learned the desired biology.

### 5.4 Propensity-adjusted sensitivity analysis

If selection is plausibly conditionally ignorable given recorded pre-validation covariates and there is overlap:

- estimate \(\hat e_i\) by cross-fitting within the outer training data;
- use stabilized inverse-probability-of-observation weights for observed validation outcomes;
- truncate extreme weights using a rule fixed in the analysis plan;
- report effective sample size and propensity overlap;
- restrict claims to the overlap population when near-zero propensities occur.

Unweighted, propensity-weighted, and outcome-regression results should all be reported. Doubly robust/AIPW estimates may be used for **aggregate** performance or yield estimates under ignorable missingness, but they do not make an individual untested gene’s outcome identifiable when unmeasured selection and publication bias remain. Bang and Robins give the classical missing-data framework for doubly robust estimators.

### 5.5 PU/PNU sensitivity analyses

Use the following only after the supervised observed-outcome benchmark:

- SAR-PU or SAR-PNU using the selection propensity;
- non-negative PU risk as an alternative when explicit failures are too sparse;
- PNU objectives combining known positives, known failures, and U;
- sensitivity to assumed positive class prevalence in U;
- sensitivity to unreported-failure rates and differential reporting by original rank.

The output should be a rank-stability analysis, not a single unquestioned “corrected” number.

Before using a selection-correction method on the real benchmark, run a semi-synthetic check: take the adjudicated outcomes, hide labels under prespecified SCAR, rank-dependent SAR, and outcome-dependent reporting scenarios, then quantify which methods recover the original ranking and where they fail. This tests implementation and sensitivity; it does not prove that the real missingness mechanism is identified.

### 5.6 What retrospective data cannot solve

No statistical method can fully recover failures that were attempted but never mentioned, or distinguish biological failure from unpublished technical failure, without assumptions or new data. The definitive mitigation is prospective:

- freeze the model;
- select a stratified/random panel across model-score and MAGeCK-rank bins;
- validate all selected instances with one protocol;
- report all outcomes, including T and F0;
- blind outcome adjudication to model score.

Only this panel should anchor the final probability calibration claim.

## 6. Feature specification

Every feature must have a provenance record, version, availability mode, and timestamp. Features are computed per gene–screen instance unless explicitly screen-level.

### 6.1 Group A — primary screen signal

From harmonized MAGeCK/drugZ analysis and guide counts:

- MAGeCK RRA score, rank, \(P\), FDR, and direction;
- MAGeCK-MLE beta and standard error when design supports it;
- drugZ normalized Z-score and rank;
- median/mean sgRNA log fold change;
- robust effect magnitude: median, trimmed mean, MAD/IQR;
- within-screen signed rank percentile;
- raw versus corrected rank and effect;
- effect consistency across replicate-pair contrasts;
- effect consistency across available time points, stored without treating time points as independent;
- effect relative to control-arm dropout;
- treatment-specific interaction versus general fitness effect.

Use both absolute values and direction-aware values. Within-screen percentiles aid cross-study harmonization, while raw effect sizes retain magnitude.

[MAGeCK](https://doi.org/10.1186/s13059-014-0554-4) and [drugZ](https://doi.org/10.1186/s13073-019-0665-3) should be both feature generators and benchmark baselines, not ground truth.

### 6.2 Group B — guide-level reliability

- number of designed and observed guides;
- fraction of guides with adequate baseline counts;
- fraction of guides with the predicted sign;
- pairwise guide concordance;
- guide-effect dispersion;
- top-guide domination;
- leave-one-guide-out change in gene score/rank;
- leave-best-guide-out change;
- multi-target or ambiguous mapping flags;
- predicted guide activity/specificity summaries when available;
- fraction of guides targeting exons shared by major transcripts;
- guide genomic clustering and number of distinct cut sites.

These features encode the difference between a coherent multi-guide signal and a result driven by one reagent.

### 6.3 Group C — replicate and screen quality

- number of biological replicates;
- replicate correlation and rank correlation;
- replicate-specific guide-count dispersion;
- baseline library coverage;
- read-depth distribution, zero-count fraction, and Gini/inequality summary;
- sample bottleneck indicators;
- control separation using prespecified essential/nonessential reference sets;
- treatment severity and global guide depletion/enrichment;
- duration, dose, replicate design, and endpoint;
- library identity and guides-per-gene;
- batch/platform variables;
- availability of plasmid-DNA or T0 reference.

Screen-level features repeat across genes but must not cause one large screen to dominate. Training weights and evaluation should be study/screen balanced.

### 6.4 Group D — nuclease and copy-number artifact risk

- cell-line gene copy number and local segment copy number;
- number of predicted genomic cutting sites;
- local density of targeted sites;
- raw-to-CNV-corrected score/rank shift;
- CRISPRcleanR segment correction magnitude;
- gene-independent segment-response flag;
- chromosome arm/amplicon indicator;
- mismatch between gene expression and apparent effect;
- untreated/control fitness effect;
- pan-essential/common-essential prior as a confounding flag.

Store raw and corrected values. [CRISPRcleanR](https://doi.org/10.1186/s12864-018-4989-y) explicitly models gene-independent genomic-segment responses; [Chronos](https://doi.org/10.1186/s13059-021-02540-7) models population dynamics and shares information across compatible screens. Neither should be treated as universally correct for every single drug-contrast design, so correction-method disagreement is itself informative.

### 6.5 Group E — cell-line and cancer context

- baseline gene expression and expression percentile;
- gene copy number and damaging/activating variants;
- lineage/tissue;
- relevant molecular subtype;
- baseline gene dependency;
- drug sensitivity of the cell line;
- target/pathway activity;
- genomic biomarkers of the drug’s mechanism;
- gene-by-context interactions derived only from training/reference data allowed by the split.

Use continuous biological descriptors where possible. Do not use cell-line IDs as free memorization tokens in the main generalization model.

### 6.6 Group F — drug and mechanism context

- canonical drug identifier and synonyms;
- target set and target confidence;
- mechanism/class;
- chemical or target-pathway descriptors;
- cytostatic/cytotoxic exposure class where curated;
- dose relative to a reported cell-line response measure;
- duration and schedule;
- monotherapy versus combination;
- distance from gene to drug targets in a frozen interaction/pathway graph;
- pathway overlap with the drug mechanism.

For drug-out evaluation, exact drug identity cannot carry predictive information. Mechanistic descriptors may remain because they are available for a new drug, but must be constructed without held-out outcome data.

### 6.7 Group G — cross-screen reproducibility

- recurrence in independent screens of the same drug;
- recurrence within drug class;
- recurrence across cell lines and lineages;
- sign concordance across screens;
- heterogeneity of effect;
- independent-screen meta-analytic summary;
- evidence in CRISPR versus RNAi or other perturbation modalities.

These are high-risk leakage features. For every outer fold, recompute them from training studies only. In temporal evaluation, include only evidence public before the test date. Exclude reanalyses of the same raw accession.

### 6.8 Group H — pre-existing mechanistic and literature evidence

- pathway membership and distance to known drug-response machinery;
- physical/genetic interaction with drug targets;
- pre-existing synthetic-lethal evidence;
- cancer-gene evidence;
- number and type of relevant studies available before the cutoff;
- prior non-screen perturbation evidence, separated by modality and direction.

Avoid raw current citation count as a timeless feature. It encodes publication age and future information. Literature and knowledge-graph features must be “as of” the outer-fold cutoff. The validation paper’s own text, figures, and conclusions are forbidden model inputs.

### 6.9 Group I — missingness and applicability

- input mode: FASTQ/count/MAGeCK-only;
- missingness mask for every modality;
- age/version of each external dataset;
- gene identifier mapping confidence;
- distance or density in the training feature space;
- categorical “outside training support” flags for new drug class, lineage, library, or assay.

Do not replace missing biological values with zero. Use fold-fitted imputation plus missingness indicators, or a model with explicit missing-value handling, and validate each supported modality pattern.

## 7. Model strategy

### 7.1 Baseline supervised classification

Primary binary task among adjudicated outcomes:

\[
Y=1:\ V2/V3;\qquad Y=0:\ F0/D.
\]

Models:

1. regularized logistic regression;
2. gradient-boosted trees, preferably LightGBM or XGBoost;
3. optional hierarchical logistic model as a diagnostic.

The deployment model should avoid gene, study, and cell-line IDs as unrestricted categorical features. A hierarchical logistic diagnostic may include study/drug random intercepts to quantify between-study heterogeneity, but a random effect unavailable for a new study defaults to the population distribution and should not be confused with transferable prediction.

Use study-balanced observation weights, with class weighting estimated only inside training folds. Hyperparameters, feature selection, imputation, propensity fitting, and calibration all occur inside nested training data.

For v0, pool resistance and sensitization only after converting every feature to a prediction-aligned sign and adding a direction indicator. Report direction-stratified results. If calibration or feature effects differ materially, use separate direction-specific heads in v1 rather than forcing one pooled probability model.

### 7.2 Learning-to-rank

Each screen contrast is a ranking query. A LambdaMART-style model can optimize a top-weighted ranking objective and naturally aligns with choosing the top \(k\) hits. The original LambdaMART overview describes the boosted-tree implementation of LambdaRank.

Use graded relevance only for adjudicated instances:

- V3: 3;
- V2: 2;
- V1: 1 in sensitivity analysis only;
- F0/D: 0;
- A/T/U: unjudged, not zero.

Limitations:

- many historical screens have only positive validations and no explicit failures;
- queries with only one judged relevance class provide little or no within-query ranking information;
- treating U as relevance 0 would train the authors’ selection pattern and introduce false negatives;
- LTR scores are not probabilities.

Therefore:

- **v0:** classification is primary; pairwise/listwise ranking is secondary where a query has adequate judged positives and negatives;
- **v1:** train LambdaMART after the registry contains enough judged within-screen comparisons or a prospective panel supplies them;
- compare classifier-derived ranks and LambdaMART ranks on identical outer folds.

### 7.3 A hybrid deployment score

Do not average unrelated objectives into one opaque number. Recommended output:

\[
\text{ReproScore}_i =
f_{\mathrm{validation}}(X_i),
\]

plus separate:

- artifact-risk probability/flag;
- context-support score;
- evidence coverage;
- novelty;
- therapeutic priority.

If a single experimental-priority list is needed, apply a transparent user-defined utility rule after prediction, for example:

\[
U_i = \text{ReproScore}_i
\lambda_1 \text{Novelty}_i
\lambda_2 \text{Tractability}_i
\lambda_3 \text{ContextSupport}_i,
\]

with the \(\lambda\) values visible and adjustable. This utility is a decision score, not a validation probability.

## 8. Training, tuning, and test splits

### 8.1 Main benchmark: leave-one-study-out

The principal retrospective benchmark is leave-one-study-out (LOSO), or grouped \(K\)-fold by study when the number of studies makes full LOSO impractical.

Holding out a study means holding out:

- all of its screens;
- every cell line, drug arm, time point, and replicate-derived contrast;
- all validation events;
- all reanalyses of the same raw dataset/accession.

The inner loop for hyperparameter selection is also grouped by study. Nested cross-validation is required because tuning and evaluating on the same folds produces optimistic estimates; Varma and Simon demonstrated this bias and the value of nested evaluation.

### 8.2 Mandatory stress tests

| Split | Held out | Scientific question |
|---|---|---|
| `study-out` | Entire publication/raw-data family | Does the pipeline generalize to a genuinely new study? |
| `temporal` | Studies first public after cutoff | Would the frozen model have predicted later validations using only earlier knowledge? |
| `compound-out` | Exact canonical drug and synonyms | Can it transfer to a new compound? |
| `drug-class-out` | Entire mechanism/class | Can it transfer beyond familiar pharmacology? |
| `cell-line-out` | All instances from one line | Can it transfer to an unseen cell line? |
| `lineage-out` | Entire tissue/lineage | Can it transfer to a new cancer context? |
| `cold-gene` | A gene is absent from every training screen row used for supervision | Does it generalize to genes never labeled during training rather than memorize known genes? |
| `cold-family` | Optional gene-family/paralog cluster held out | Does it transfer beyond close gene homologs? |

`cold-gene` is not a replacement for study-out. Report both:

- study-out with genes potentially observed elsewhere;
- cold-gene, preferably nested within held-out studies or as a separate benchmark matrix.

### 8.3 Temporal evaluation

Use the earliest public date of data/preprint/article as the screen timestamp. For each temporal cutoff:

- train only on earlier studies;
- construct literature, pathway, drug, and cross-screen features from versions available by that date;
- exclude later database annotations;
- exclude citations to or derived knowledge from the test study;
- freeze code, ontology, and hyperparameters before scoring the later set.

A current database snapshot with a historical paper split is not a true temporal benchmark.

### 8.4 Split-aware preprocessing

Within every outer fold:

1. remove held-out raw-data families;
2. fit identifier harmonization rules that are not globally fixed;
3. fit imputation/scaling on training only;
4. compute training-only cross-screen recurrence and meta-features;
5. fit selection propensity by cross-fitting within training;
6. tune model and feature set in grouped inner folds;
7. create out-of-fold training predictions for calibration;
8. lock the fitted pipeline;
9. transform and predict the untouched outer test study.

## 9. Leakage-prevention register

The following are prohibited or must be explicitly controlled:

1. **Guide leakage:** guides from one gene–screen instance split across train and test.
2. **Contrast leakage:** replicate pairs or time points from the same experiment split across train and test.
3. **Study leakage:** different screens from the same paper in both sets during main evaluation.
4. **Raw-data duplication:** the same GEO/SRA/ENA accession reprocessed in multiple papers and treated as independent.
5. **Validation-paper leakage:** abstracts, figures, supplementary validation tables, or conclusions used as predictors.
6. **Future-knowledge leakage:** present-day database and literature evidence used in a historical temporal test.
7. **Cross-screen feature leakage:** recurrence/meta-analysis computed using the held-out study.
8. **Outcome-derived features:** number of validation assays, rescue status, figure prominence, whether the gene appears in the title, or validation grade used as input.
9. **Selection leakage:** “authors chose this gene” used as a reproducibility feature. It is allowed only in the separate selection model or as a clearly labeled baseline.
10. **Gene identity memorization:** one-hot gene IDs, learned target encoding from all data, or embeddings trained using held-out validation labels.
11. **Database circularity:** a target database entry whose evidence ultimately comes from the held-out screen.
12. **Global preprocessing:** imputation, scaling, feature filtering, calibration, or class-prior estimation fitted before the split.
13. **Test-set iteration:** changing labels, feature definitions, or hyperparameters after inspecting test performance.

Maintain a `provenance_graph` linking every derived feature to source accessions and dates. Automated tests should assert that no held-out study contributes to training-derived features.

## 10. Evaluation metrics

### 10.1 Two evaluation populations

Because U is unjudged, publish two distinct evaluations.

#### A. Adjudicated-outcome evaluation

Population: V2/V3 versus F0/D only.

Measures:

- PR-AUC as the primary threshold-free classification metric;
- AUROC as secondary;
- sensitivity, specificity, positive predictive value at prespecified operating points;
- Brier score and log loss for probabilities;
- calibration intercept/slope and reliability plot;
- performance stratified by direction, drug class, cell line, input mode, and validation quality.

This measures discrimination among evaluated historical outcomes. It does not estimate performance over all U genes.

Freeze the exact PR-AUC implementation. Average precision and trapezoidal interpolation of a precision–recall curve are not numerically interchangeable; the benchmark should designate one, preferably average precision, and use it for every model.

#### B. Screen-level retrieval evaluation

Population: all QC-eligible genes or a frozen candidate zone.

Measures:

- Precision@5, @10, and @20 using V2/V3 as observed relevant hits;
- Recall@5, @10, and @20 of known V2/V3 hits;
- NDCG@5, @10, and @20 with graded V3/V2 relevance;
- mean reciprocal rank of the first V2/V3 hit;
- top-\(k\) uplift over MAGeCK/drugZ;
- validation-budget curves: number of known V2/V3 hits recovered versus genes tested.

In incomplete historical judgments, call P@k **observed validated precision@k** or **validated-hit yield@k**. U is unjudged, so this is not the true biological precision. NDCG should be computed:

- on judged candidates for a clean comparison; and
- on the full ranking as a clearly labeled incomplete-judgment retrieval measure.

NDCG is appropriate for top-weighted graded relevance, following the cumulative-gain framework introduced by Järvelin and Kekäläinen.

### 10.2 Aggregation

- Compute metrics within screen wherever defined.
- Macro-average screens, then studies, so a study with many genes/contrasts does not dominate.
- Also report pooled metrics for transparency.
- Report the number of eligible screens for each metric; a screen with no judged positive cannot contribute to recall/NDCG in the usual way.
- Use paired per-screen differences when comparing models.

### 10.3 Confidence intervals and hypothesis testing

- Resample studies as clusters for bootstrap confidence intervals.
- Preserve all nested screens and genes when a study is sampled.
- Add a two-way study/gene clustered sensitivity analysis when recurrent genes contribute many judged instances.
- For a prospective panel, additionally report binomial or bootstrap intervals for top-\(k\) yield.
- Compare models using paired study/screen-level differences, not millions of gene rows as if independent.
- Predefine one primary metric, recommended macro NDCG@10 for ranking, and one primary classification metric, PR-AUC among adjudicated outcomes. Treat the rest as secondary to avoid metric shopping.

PR curves are preferable to relying only on ROC curves when adequate failures/successes are imbalanced; Davis and Goadrich give the formal relationship between PR and ROC evaluation.

## 11. Calibration, uncertainty, and applicability

### 11.1 Calibration

Only a classifier can directly provide a probability. A LambdaMART score must not be labeled as one.

Calibration procedure:

1. within each outer training set, generate grouped inner out-of-fold predictions;
2. fit the calibrator only to those out-of-fold predictions;
3. compare sigmoid/Platt and beta calibration; use isotonic only when the number and coverage of calibration outcomes are adequate;
4. apply the frozen calibrator to the outer test study;
5. assess Brier score, log loss, calibration slope/intercept, and reliability by risk bin.

Scikit-learn’s official calibration documentation also stresses that the calibrator must be fitted on data disjoint from base-model fitting. Beta calibration is a useful parametric alternative described by Kull and colleagues.

Historical calibration has a restricted interpretation:

\[
P(Y=1\mid R=1,X),
\]

not automatically \(P(Y^\star=1\mid X)\) for all screen genes. Until prospective calibration exists, label the output `relative_reproducibility_score`.

### 11.2 Predictive uncertainty

Report at least:

- median prediction across outer-training bootstrap models;
- 5th–95th percentile score/rank interval across study-cluster bootstrap fits;
- rank stability: fraction of fits in which a gene appears in top 5/10/20;
- model-disagreement between logistic, boosted-tree, and ranking models;
- sensitivity to label definitions and selection assumptions;
- data-coverage score and nearest-domain diagnostics.

Separate:

- **aleatoric/label ambiguity:** V1/A-prone evidence, assay heterogeneity;
- **epistemic uncertainty:** sparse or out-of-domain features, model disagreement;
- **selection uncertainty:** unknown outcomes for U and unreported failures.

A narrow bootstrap interval does not remove selection bias. Likewise, a high SHAP contribution is not uncertainty.

### 11.3 Out-of-distribution guardrails

Flag or abstain when:

- drug class was unseen;
- lineage was unseen;
- library/assay design is outside training support;
- required core features are missing;
- feature-space distance exceeds a threshold set using training data;
- different model families disagree substantially;
- the prediction depends on a single unstable guide or artifact-prone region.

For an abstained instance, provide the screen statistics and evidence profile but no calibrated probability.

## 12. Missing modalities and user input modes

Support separate, benchmarked models rather than pretending all inputs have equal information:

| Model | Minimum input | Intended output |
|---|---|---|
| `Lite` | MAGeCK/drugZ gene table plus design metadata | Limited reranking; no guide-level confidence claim |
| `Core` | sgRNA count table, library annotation, sample/design sheet | Harmonized analysis, guide/replicate QC, reproducibility ranking |
| `Full` | FASTQ or counts plus library, design, cell-line CN/expression | Counting/QC, artifact correction, context-aware ranking |

Implementation principles:

- fit a separate validated pipeline for each mode, or use a prespecified late-fusion ensemble of modality-specific models;
- include missingness masks;
- fit all imputation inside training folds;
- evaluate simulated modality removal;
- report performance stratified by mode;
- do not silently backfill missing multi-omics values with population means and present unchanged confidence;
- display evidence coverage next to every score.

If FASTQ and count-table users ultimately converge on the same harmonized count object, their prediction should agree apart from FASTQ-derived mapping/counting QC features.

## 13. Explainability and scientific audit

Use explanations at three levels:

1. **Raw evidence card**
   - guide effects;
   - replicate consistency;
   - raw and corrected gene ranks;
   - copy-number/artifact flags;
   - missing data.

2. **Model explanation**
   - local SHAP values for the selected model;
   - global, outer-fold-only feature importance;
   - grouped importance by feature family;
   - partial dependence or accumulated local effects for major continuous variables.

3. **Evidence provenance**
   - exact database/version/date;
   - training-only cross-screen sources;
   - literature citations;
   - whether evidence is same drug, same class, or external.

SHAP provides additive feature attributions, but it does not establish causality. Correlated feature groups can redistribute attribution, so show grouped and ablation evidence, not only a colorful single-gene SHAP plot.

Include counterfactual-style diagnostics:

- score with CNV/artifact features removed;
- score using screen-only features;
- score without literature/network priors;
- score after leave-best-guide-out perturbation;
- rank stability across correction methods.

An LLM may summarize this evidence after scoring, but the LLM must not determine the rank or fabricate citations.

## 14. Baselines and ablations

### 14.1 Required baselines

Evaluate every baseline on identical outer splits and candidate universes:

1. random within-screen ranking;
2. absolute/signed median sgRNA log fold change;
3. MAGeCK RRA rank/FDR;
4. MAGeCK-MLE beta or rank where available;
5. drugZ normalized Z/rank;
6. CRISPRcleanR-corrected effect/rank;
7. simple consensus of within-screen MAGeCK and drugZ percentiles;
8. screen-only regularized logistic regression;
9. external-evidence-only logistic regression;
10. literature popularity/known-pathway prior;
11. author-selection propensity score;
12. final regularized logistic model;
13. final boosted-tree model;
14. LambdaMART where label structure permits.

The author-selection model is a particularly important negative control. A model that merely predicts which genes papers discuss may look strong on published positives while failing prospectively.

### 14.2 Feature-family ablations

Predefine:

- screen signal only;
- + guide/replicate QC;
- + CNV/artifact;
- + cell/drug context;
- + cross-screen evidence;
- + pathway/network;
- full minus literature;
- full minus gene-level priors;
- raw scores versus corrected scores;
- exact-context features versus broad-context features.

Report both performance and calibration changes. A small performance gain from external evidence may still be valuable if it improves cold-gene or drug-out transfer.

## 15. Feasible releases

### 15.1 v0 — honest pilot

**Goal:** demonstrate a reproducible registry and test whether guide-aware/context-aware ranking adds value beyond screen statistics.

Recommended scope:

- 20–30 publications, aiming for multiple contrasts per study;
- human CRISPR-KO drug-conditioned fitness screens only;
- full gene universe and study metadata;
- manually curated ValidationEvent registry;
- explicit separation of V2/V3, F0/D, V1/A/T/U;
- harmonized MAGeCK and drugZ features where counts permit;
- raw/CNV-corrected feature pairs;
- logistic and boosted-tree models;
- study-out nested CV plus one temporal holdout;
- cold-gene and compound-out stress tests;
- no claim of unconditional probability.

Primary v0 deliverables:

1. data model and curation rubric;
2. baseline benchmark;
3. `relative_reproducibility_score`;
4. selection-model diagnostic;
5. study-cluster uncertainty;
6. interpretable evidence cards;
7. locked external case study on the held-out private screen.

If explicit adequate failures are fewer than roughly several dozen across independent studies, treat classification as exploratory. Emphasize ranking stability and prospective-data acquisition rather than fitting a complex model.

### 15.2 v1 — publishable model with probability claim

**Goal:** estimate prospective reproducibility under a defined protocol.

Add:

- 50–100 studies or the largest feasible curated compendium;
- stronger retrieval of explicit negative and discordant validations;
- curator duplication and adjudication;
- prospective panel sampled across score and MAGeCK-rank strata;
- protocol-standardized validation with complete outcome reporting;
- frozen temporal test;
- cross-fitted selection model and propensity sensitivity;
- PNU/SAR-PU sensitivity analyses;
- LambdaMART trained on sufficiently judged screen queries;
- prospective recalibration;
- external replication in at least one lab/context if possible.

Prospective sampling should include:

- high model / high MAGeCK;
- high model / moderate MAGeCK;
- low model / high MAGeCK;
- intermediate and low-score controls;
- both resistance and sensitization;
- multiple drugs or at least distinct mechanistic groups.

Sampling only the model’s top predictions estimates showcase yield, not calibration. A stratified/random panel across the score range is required to test probability calibration.

## 16. Recommended benchmark contract

Freeze before final evaluation:

- inclusion/exclusion rules;
- candidate-zone definition;
- label rubric;
- context-match rules;
- feature dictionary and timestamps;
- exact split manifests;
- primary and secondary metrics;
- hyperparameter-search space;
- calibration method selection rule;
- missing-data strategy;
- weight truncation and overlap rules;
- prospective panel sampling;
- minimum meaningful validation effect;
- model-selection criterion.

Publish:

- immutable study/accession manifest;
- per-feature provenance;
- deduplication map;
- outer-fold predictions for every model and baseline;
- adjudicated labels with source locations;
- code/environment lockfile;
- data and model cards;
- negative and null results.

## 17. Decision gates

Proceed from v0 to v1 only if:

1. at least two curators can apply the ontology reproducibly;
2. the registry contains explicit failures from multiple independent studies;
3. the full model improves macro top-\(k\) retrieval over MAGeCK/drugZ with study-cluster uncertainty that excludes a trivial gain;
4. the advantage persists in temporal and at least one cold-domain split;
5. the model is not simply reproducing the author-selection propensity;
6. score direction is stable across reasonable label and selection-bias sensitivity analyses;
7. prospective validation is feasible before any absolute-probability claim.

If these conditions fail, the project can still be valuable as a transparent evidence/QC framework, but should not be marketed as a learned validator.

## 18. Primary methodological references

1. Li W, et al. [MAGeCK enables robust identification of essential genes from genome-scale CRISPR/Cas9 knockout screens](https://doi.org/10.1186/s13059-014-0554-4). *Genome Biology*. 2014.
2. Colic M, et al. [Identifying chemogenetic interactions from CRISPR screens with drugZ](https://doi.org/10.1186/s13073-019-0665-3). *Genome Medicine*. 2019.
3. Iorio F, et al. [Unsupervised correction of gene-independent cell responses to CRISPR-Cas9 targeting](https://doi.org/10.1186/s12864-018-4989-y). *BMC Genomics*. 2018.
4. Dempster JM, et al. [Chronos: a cell population dynamics model of CRISPR experiments that improves inference of gene fitness effects](https://doi.org/10.1186/s13059-021-02540-7). *Genome Biology*. 2021.
5. Elkan C, Noto K. [Learning classifiers from only positive and unlabeled data](https://cseweb.ucsd.edu/~elkan/posonly.pdf). *KDD*. 2008.
6. Bekker J, Davis J. [Learning from Positive and Unlabeled Data under the Selected At Random Assumption](https://proceedings.mlr.press/v94/bekker18a.html). *PMLR*. 2018.
7. Sakai T, et al. [Semi-Supervised Classification Based on Classification from Positive and Unlabeled Data](https://proceedings.mlr.press/v70/sakai17a.html). *ICML*. 2017.
8. Kiryo R, et al. [Positive-Unlabeled Learning with Non-Negative Risk Estimator](https://proceedings.neurips.cc/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html). *NeurIPS*. 2017.
9. Bang H, Robins JM. [Doubly robust estimation in missing data and causal inference models](https://doi.org/10.1111/j.1541-0420.2005.00377.x). *Biometrics*. 2005.
10. Burges CJC. [From RankNet to LambdaRank to LambdaMART: An Overview](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/). Microsoft Research Technical Report. 2010.
11. Ke G, et al. [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree). *NeurIPS*. 2017.
12. Chen T, Guestrin C. [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785). *KDD*. 2016.
13. Varma S, Simon R. [Bias in error estimation when using cross-validation for model selection](https://doi.org/10.1186/1471-2105-7-91). *BMC Bioinformatics*. 2006.
14. Järvelin K, Kekäläinen J. [Cumulated gain-based evaluation of IR techniques](https://doi.org/10.1145/582415.582418). *ACM TOIS*. 2002.
15. Davis J, Goadrich M. [The relationship between Precision-Recall and ROC curves](https://dl.acm.org/doi/10.1145/1143844.1143874). *ICML*. 2006.
16. Brier GW. [Verification of forecasts expressed in terms of probability](https://doi.org/10.1175/1520-0493%281950%29078%3C0001:VOFEIT%3E2.0.CO;2). *Monthly Weather Review*. 1950.
17. Kull M, Silva Filho TM, Flach P. [Beta calibration: a well-founded and easily implemented improvement on logistic calibration](https://proceedings.mlr.press/v54/kull17a.html). *AISTATS*. 2017.
18. Lundberg SM, Lee S-I. [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions). *NeurIPS*. 2017.
19. Scikit-learn developers. [Probability calibration documentation](https://scikit-learn.org/stable/modules/calibration.html). Official documentation.
