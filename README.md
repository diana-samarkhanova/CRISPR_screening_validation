# CRISPR-EvidenceRank

CRISPR-EvidenceRank is a guide-aware, experimental-design-aware, and
context-aware framework for prioritizing genes from human CRISPR-Cas9 knockout
drug-response screens for orthogonal experimental follow-up.

The scientific target is deliberately narrow:

> rank genes within a completed drug-response screen by their relative
> likelihood of reproducing under independent perturbation, while separating
> screen signal, artifact risk, mechanistic support, therapeutic priority, and
> novelty.

The project does **not** treat a language-model answer, literature frequency,
or a single database score as experimental validation.

## Status

Version `0.2.0.dev0` is a reproducible development foundation, not a released
pretrained model. It retains the executable v0.1 screen-signal baseline and
defines the v0.2 normalized experimental-design, provenance, feature-profile,
and leakage-control contract required before training on real studies. It
includes:

- normalized records for studies, screens, contrasts, samples, gene scores,
  validation events, and external evidence;
- an explicit validation-label ontology (`V3`, `V2`, `V1`, `F0`, `D`, `A`,
  `T`, `U`);
- raw-count validation, robust median-ratio normalization, and guide-to-gene
  feature extraction;
- MAGeCK gene-summary normalization;
- storage for raw and CNV-corrected scores without silently substituting a
  homemade correction;
- a two-stage baseline that models both author selection for testing and
  validation success among tested genes, with unweighted primary results and
  inverse-propensity weighting reported as a sensitivity analysis;
- source/raw-family- and study-grouped evaluation specifications and ranking
  metrics;
- deterministic synthetic data and automated tests.

Training on a harmonized public compendium and prospective validation are the
next scientific milestones. Until those are complete, output is a
**relative reproducibility score**, not a calibrated probability.

The current ORCS intake milestone pins the human screen archive to BioGRID
ORCS `2.0.18`, compiled on `2025-09-09` and publicly available on
`2025-10-07`. The observed index contains 1,952 human screen records. Index-only
triage assigns 278 to `exclude` and 1,674 to `metadata_only`; within the latter,
435 are confirmed-scope candidates and 1,239 require manual scope review.
There are zero `benchmark_ready` screens at this stage. These counts describe
an outcome-blind curation queue, not a harmonized or trainable biological
corpus.

The first full-text pilot freezes queue ranks 1-10 from ten publication
families before outcome review. Nine screens provide author-derived score
tables only; one has public raw sequencing reads (`SRP158611`). All ten remain
`metadata_only` because count-level ingestion, rights, raw-data-family, and/or
independent validation adjudication are incomplete. Candidate validation
grades are stored downstream of selection and are not yet training labels.

Development from v0.2 onward is versioned in a private GitHub repository.
Public visibility is a later release gate, not a prerequisite for using Git
history and continuous integration. See `docs/REPOSITORY_POLICY.md`.

## V1 scope

Included:

- human pooled CRISPR-Cas9 knockout screens;
- drug-treated versus control competitive/viability screens;
- gene-level resistance or sensitization outcomes;
- count-table input as the primary supported mode;
- MAGeCK/drugZ/CRISPRcleanR-derived summaries as optional features;
- cancer cell-line context and study-level grouping.

Deferred to separate model branches:

- CRISPR activation and CRISPR interference;
- single-cell perturbation readouts;
- non-drug phenotypes;
- clinical target prioritization as the primary endpoint.

This separation prevents incompatible perturbation mechanisms and phenotype
definitions from being pooled into one label.

## Experimental-design hierarchy

Every observation is normalized at four distinct levels:

| Level | Meaning |
|---|---|
| `study` | Publication, preprint, or author-defined research source. |
| `screen` | Shared cell model, CRISPR modality/library, delivery, and pooled experiment. |
| `contrast` | One defined treatment-versus-comparator question, including dose, schedule, endpoint, and intended direction. |
| `sample` | One sequenced biological or technical observation assigned to a contrast. |

A gene prediction remains `gene × screen × contrast × direction`. Conditions
that differ by drug, dose, time point, comparator, or cell model must not be
silently collapsed into one contrast.

## Input modes

| Mode | Input | Current status |
|---|---|---|
| Standard | sgRNA count table + guide annotations + sample sheet | Implemented |
| Lite | MAGeCK gene summary + screen metadata | Implemented adapter |
| Full | FASTQ + library + sample sheet | Contract defined; workflow pending |

The Standard mode is the primary modeling input because it preserves
guide-level evidence while avoiding mandatory redistribution of large FASTQ
files.

## Feature profiles

The model is evaluated as nested information profiles rather than pretending
that every user supplies equivalent data:

| Profile | Evidence used |
|---|---|
| `screen_only` | Gene scores/count-derived guide signal and technical QC. |
| `screen_plus_design` | `screen_only` plus structured study/screen/contrast/sample metadata. |
| `context_aware` | `screen_plus_design` plus versioned artifact/CNV, cell, drug, pathway, and training-only cross-screen evidence. |
| `selection_model` | A separate model for which genes authors chose to test; its selection-only fields are forbidden in the success model. |

Missing modalities retain explicit availability indicators. Reported zero is
kept distinct from “not reported,” and a reduced profile or abstention is used
when required inputs lie outside training support.

## Screen discovery and metadata sources

[BioGRID ORCS](https://orcs.thebiogrid.org/) is used to discover published
screens, recover structured metadata, and import author-method gene scores,
ranks, and hit calls. Intake is pinned to the
[ORCS 2.0.18 release archive](https://downloads.thebiogrid.org/BioGRID-ORCS/Release-Archive/BIOGRID-ORCS-2.0.18/)
rather than the mutable latest-release path. The compiled date, public
availability date, and actual retrieval date are recorded separately. ORCS is
not an orthogonal-validation registry: an ORCS hit call can be a screen-signal
feature or baseline, but never a `V2/V3/F0/D` label.

Primary papers, supplements, raw repositories, and author-supplied matrices are
used to verify exact treatment and control conditions. Library references such
as Joung et al. can identify the library design, but their example protocol is
not copied into a historical screen. Unreported MOI, coverage, selection,
treatment, duration, or comparator fields remain missing.

## Architecture

1. `ingest`: normalize the study → screen → contrast → sample hierarchy,
   validate identifiers, and record provenance.
2. `screen signal`: apply declared normalization, compute guide agreement,
   count quality, direction alignment, effect summaries, and within-screen
   ranks.
3. `artifact layer`: join copy number, expression, corrected scores, and
   off-target/multi-target flags.
4. `evidence layer`: add recurrence and context features using versioned
   sources.
5. `deduplication`: group publications, reanalyses, repositories, and ORCS
   records derived from the same experimental material into a source/raw family.
6. `selection model`: estimate which hits authors chose to test.
7. `reproducibility model`: estimate relative success among genuinely tested
   hits using grouped validation and selection-aware weights.
8. `report`: return separate component scores, uncertainty, missingness, and
   source provenance.

## Quick start

```bash
python -m pip install -e ".[dev]"
python scripts/generate_synthetic.py
crispr-evidencerank validate \
  --table examples/synthetic/sample_sheet.csv \
  --contract sample
crispr-evidencerank featurize-counts \
  --counts examples/synthetic/guide_counts_screen_01.csv \
  --samples examples/synthetic/sample_sheet_screen_01.csv \
  --output data/processed/synthetic_gene_features.csv
crispr-evidencerank featurize-design \
  --screens examples/synthetic/screens.csv \
  --screen-designs examples/synthetic/screen_designs.csv \
  --contrasts examples/synthetic/contrasts.csv \
  --samples examples/synthetic/sample_sheet.csv \
  --output data/processed/synthetic_design_features.csv
pytest
```

The synthetic data demonstrate the software contract only and must never be
used as scientific evidence.

`featurize-counts` expects integer raw read counts. A positive-count
median-ratio variant is the default; all-zero and otherwise uninformative rows
are excluded from size-factor estimation. This avoids the compositional
artifacts that total-count CPM can create when a small number of guides expand
strongly. Use `--normalization-method cpm` only as a declared sensitivity
analysis.

For a release-pinned BioGRID ORCS archive:

```bash
crispr-evidencerank prepare-orcs-release \
  --release-registry config/orcs_releases.yaml \
  --release 2.0.18 \
  --retrieved-date 2026-07-31 \
  --cache-dir data/external/orcs_2.0.18 \
  --output-dir data/processed/orcs_2.0.18/release_intake
```

This command downloads or reuses only the pinned archive, verifies its
project-computed SHA-256 and exact tar inventory, cross-checks all index IDs
against the per-screen member IDs, then atomically writes normalization,
provenance, triage, and curation-queue outputs. It refuses to mix a rerun into
an existing output directory.

The lower-level adapters remain available for focused reprocessing:

```bash
crispr-evidencerank ingest-orcs-index \
  --table BIOGRID-ORCS-SCREEN_INDEX-2.0.18.index.tab.txt \
  --release 2.0.18 \
  --available-date 2025-10-07 \
  --retrieved-date 2026-07-31 \
  --organism-scope "Homo sapiens" \
  --output-dir data/processed/orcs_2.0.18/index

crispr-evidencerank triage-orcs-index \
  --table BIOGRID-ORCS-SCREEN_INDEX-2.0.18.index.tab.txt \
  --release 2.0.18 \
  --available-date 2025-10-07 \
  --retrieved-date 2026-07-31 \
  --organism-scope "Homo sapiens" \
  --output-dir data/processed/orcs_2.0.18/triage

crispr-evidencerank ingest-orcs-screen \
  --table BIOGRID-ORCS-SCREEN_95-2.0.18.screen.tab.txt \
  --index-metadata BIOGRID-ORCS-SCREEN_INDEX-2.0.18.index.tab.txt \
  --release 2.0.18 \
  --output-dir data/processed/orcs_2.0.18/screen_95
```

The adapter preserves lossless raw tables alongside normalized registry
records and parsing issues. `HIT` remains `author_hit`; unreported comparator
and score direction remain unknown.

Index triage writes one screen-level status and one auditable row per
eligibility rule. Explicit non-human, non-KO, non-drug, or arrayed records may
be excluded. Missing metadata remains `metadata_only`; index metadata can
never promote a screen directly to `benchmark_ready`.

For ORCS 2.0.18, the observed deterministic triage is:

| Intake result | Screen records |
|---|---:|
| Total | 1,952 |
| `exclude` | 278 |
| `metadata_only` | 1,674 |
| Confirmed-scope queue | 435 |
| Manual-scope-review queue | 1,239 |
| `benchmark_ready` | 0 |

The queue is outcome-blind: ranking uses scope and metadata completeness, not
ORCS `HIT`, author score magnitude, validation outcomes, or later evidence.
Queue rank is therefore a curation priority only. It must not be interpreted as
a biological hit rank or model-training label.

`Toxin Exposure` is not treated as equivalent to `Drug Exposure`, because the
ORCS condition field may describe a toxin, pathogen, medium, or another
non-drug exposure. Such records remain candidates for manual review. A curated
screen becomes `benchmark_ready` only when the complete policy-v2 rule set is
present and passing, including comparator/sample reconstruction, count-level
signal, source and raw-data families, rights, and adjudicated validation
events. A draft `single_curator` event remains useful for curation but cannot
open the benchmark gate until it receives an approved consensus status.

## Training-label policy

- `V3`: causal validation with independent perturbation plus rescue or a
  comparably strong causal reversal.
- `V2`: phenotype reproduced with independent reagent(s), confirmed
  perturbation, and an appropriate control.
- `V1`: supportive but incomplete follow-up.
- `F0`: perturbation succeeded, but the screen phenotype did not reproduce.
- `D`: a verified result in the opposite direction.
- `A`: ambiguous interpretation.
- `T`: technical failure; phenotype cannot be judged.
- `U`: untested or testing status unknown; `testing_status` records
  `not_tested` and `unknown` separately.

Primary supervised labels are `V2/V3` (positive) and `F0/D` (negative).
`V1/A/T/U` remain missing or auxiliary. A gene absent from a paper's validation
section is not a negative.

## Leakage controls

- split by source/raw-data family and study, never by gene row;
- keep every reanalysis of the same accession or author matrix in one fold;
- report leave-one-study-out and leave-one-drug-class-out performance;
- freeze the model before evaluating any held-out private screen;
- recompute cross-screen evidence from outer-training families only and apply a
  source-availability time cutoff;
- keep exact screen-derived features separate from post-publication evidence;
- use ORCS screen IDs as provenance/join keys and ORCS scores as screen evidence,
  not validation outcomes;
- never use the validation paper's prose or follow-up result as an input
  feature for that same label.

## Held-out prospective case study

An unpublished CRISPR-KO drug-response screen is reserved as a prospective
case study. Its cell model, treatment, anchor genes, count tables, and results
are intentionally excluded from this repository. Any non-KO branch will be
evaluated separately after the KO model is frozen.

## Data and copyright

Code is licensed under Apache-2.0. The ORCS-distributed 2.0.18 archive and its
index/score files are covered by the ORCS download terms recorded in the data
manifest. That license does not transfer to publisher supplements, GEO/SRA
assets, author count matrices, FASTQ files, or other upstream source data;
those retain their own terms and require separate rights review. This
repository should contain accession manifests, download scripts, checksums,
derived features that are lawful to redistribute, and exact provenance—not
copied article text, figures, or unrestricted copies of source matrices.

The recorded SHA-256 for the ORCS archive is computed by this project after
download and is not represented as a checksum published by BioGRID. The
retrieval timestamp records when that exact byte stream was obtained; it is not
the release's compiled or availability date.

For ORCS-derived records, preserve the ORCS release, screen ID, original
publication, author scoring method, and raw/source-family mapping. Multiple
methods or repository copies of one experiment do not create independent
training screens.

The local annotation workbook is a private research artifact and is excluded
from Git by default. Before a public release, verify each dataset version,
license, institutional approval, authorship, embargo, and patent position.

## Intended publication benchmark

The minimum credible paper requires:

- a versioned multi-study training compendium;
- explicit successful and failed validation events;
- MAGeCK/drugZ and effect-size baselines;
- study-, drug-, cell-line-, and gene-cold evaluation;
- observed success yield@5/10/20, NDCG, PR-AUC, calibration, and bootstrap
  uncertainty;
- ablation of guide, CNV/artifact, recurrence, context, and literature layers;
- a frozen-model prospective validation study.

See `docs/SCIENTIFIC_PROTOCOL.md` and `docs/MODEL_CARD.md`.

For the adjacent pre-screen task, [AssayBench](https://github.com/Genentech/AssayBench)
benchmarks prediction of a screen gene ranking from an assay description. It is
an important landscape comparator, but it predicts the screen result before
screening rather than the orthogonal reproducibility of an already observed
guide-level hit.
