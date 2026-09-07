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

Version `0.5.0.dev0` is a reproducible development system, not a released
pretrained model. It includes one-command MAGeCK/count reporting, an auxiliary
ICRAFT-inspired immune-context method, and a separate checksum-bound clinical
treatment-by-cancer registry report. Post-checkpoint development also adds a
separate frozen ClinicalTrials.gov API intake; its unreviewed retrieval output
cannot enter that report. These components sit on top of the normalized
experimental-design, provenance, feature-profile, and leakage-control
foundation. The project includes:

- normalized records for studies, screens, contrasts, samples, gene scores,
  validation events, and external evidence;
- an explicit validation-label ontology (`V3`, `V2`, `V1`, `F0`, `D`, `A`,
  `T`, `U`);
- a checksum-bound event-level Gold workflow for randomized review packets,
  two independent human reviews, field-level comparison, and separately
  recorded named-human adjudication;
- raw-count validation, robust median-ratio normalization, and guide-to-gene
  feature extraction;
- a self-contained `rank-screen` bundle for MAGeCK summaries, count tables, or
  both, including QC, input checksums, declared direction semantics, and a
  human-readable report;
- a report-only immune-screen evidence contract, tumor/immune dual-action
  classification, provenance-aware recurrence, and verified-full-list RRA;
- a report-only clinical-trial contract and exact concept-ID treatment-by-cancer
  landscape that is source-asset bound and cannot change gene ranking;
- a bounded ClinicalTrials.gov v2 snapshot intake that retains every raw JSON
  page, the before/after API version envelope, exact query and field projection,
  pagination lineage, and SHA-256-bound derived inventories;
- storage for raw and CNV-corrected scores without silently substituting a
  homemade correction;
- a two-stage baseline that models both author selection for testing and
  validation success among tested genes, with independent-provenance-group
  balancing in the primary non-selection-IPW analysis and clipped
  inverse-propensity weighting reported as a sensitivity analysis;
- source/raw-family- and study-grouped evaluation specifications and ranking
  metrics;
- deterministic synthetic data and automated tests.

Training on a harmonized public compendium and prospective validation are the
next scientific milestones. Until those are complete, `rank-screen` returns a
**screen-signal priority**, not a reproducibility score or calibrated
probability. The existing grouped model code remains a benchmark harness for
properly labeled data and synthetic tests.

The current ORCS intake milestone pins the human screen archive to BioGRID
ORCS `2.0.18`, compiled on `2025-09-09` and publicly available on
`2025-10-07`. The observed index contains 1,952 human screen records. Index-only
triage assigns 278 to `exclude` and 1,674 to `metadata_only`; within the latter,
435 are confirmed-scope candidates and 1,239 require manual scope review.
There are zero `benchmark_ready` screens at this stage. These counts describe
an outcome-blind curation queue, not a harmonized or trainable biological
corpus.

The first full-text pilot freezes queue ranks 1-10 from ten publication
families before outcome review. Eight screens provide author-derived score
tables only, one has a public sgRNA count matrix, and one has public amplicon
reads (`SRP158611`). Independent second review is complete for all ten screens:
the frozen ranks 6-10 checkpoint, its checksum-bound ranks 1, 3, 4, and 5
progress addendum, and an independently reviewed rank 2 were combined through
the authenticated completion workflow. The full comparison contains 20
gene-level rows: 17 provisional agreements, one evidence-level disagreement,
and two single-curator observations. All require human adjudication.
Adjudication-relevant metadata disagreements remain reported field by field
across source, design, quantitative-data, rights, validation, and blocker
metadata.
All ten screens remain `metadata_only` because count ingestion/QC, rights,
sample mapping, and/or adjudication are incomplete. Candidate validation
grades are stored downstream of selection and are not training labels.

### Event-level Gold review

The event-level workflow is infrastructure for creating a Gold validation set;
it is not evidence that a real Gold set already exists. The checked-in
event-level records used by automated tests are artificial. No real human
review packet, adjudicated biological label set, or `benchmark_ready` screen is
released by this repository checkpoint.

`prepare-event-review-packet` creates one randomized, checksum-bound **core
table bundle** shown identically to Curators A and B. Its four generated files
are:

- `event_evidence_packet.tsv`: frozen event cards with packet-event IDs,
  source/study/screen/contrast linkage, gene, the screen-predicted direction,
  screen drug and cell model, reported validation context, and exact source URL
  and locator;
- `event_review_template.tsv`: the same event roster with all review decisions
  blank;
- `event_review_rubric.yaml`: the frozen label and decision rules; and
- `event_packet_manifest.json`: record, roster, input, output, randomization,
  and blinding checksums. Only the seed hash is recorded.

Do not send those four files as the complete Curator B handoff. The operational
packet must additionally contain the frozen handbook/workbook, complete source
index, lawful full text and supplements (or access-controlled immutable source
snapshot), source version/checksums, and the human independence/identity/COI
declaration. These access-controlled materials remain outside this repository
and the four-file writer; the coordinator must checksum-bind them in the
operational batch manifest.

The packet content excludes model predictions, prior labels, and co-reviewer
decisions, but this is **procedural prediction and co-reviewer blinding**, not
full outcome blinding or cryptographic non-disclosure. Gene identity,
treatment context, the frozen screen-predicted direction, and joinable source
identifiers remain visible because the reviewer must verify the cited source
and judge concordant versus opposite validation. Curator B must therefore work
in an access-isolated workspace without repository, confidential-crosswalk,
or legacy-curation access. Curator B must not receive Curator A's review,
comparison or adjudication output, EvidenceRank/model predictions, MAGeCK rank
or score, prior/proposed/final labels, `benchmark_ready`, or unstructured
pipeline notes. The actor registry is checksum-bound input but is not copied
into the distributable packet. Automated text guards catch obvious leakage,
but cannot prove that prose is cognitively neutral; packet preparation
therefore also requires a recorded coordinator attestation that every card was
manually audited for judgment-bearing text.

Both review streams must cover the same frozen packet and come from distinct,
verified human reviewers working in separate sessions. Their comparison
reports exact and field-level agreement, a confusion matrix, Cohen's kappa,
and Gwet's AC1, but it releases no final label. **Named human adjudication is
required for every event, including exact reviewer agreement.** Finalization
requires complete named-human adjudication of the frozen comparison, preserves
its audit lineage, and derives validation-event records. The actor registry and
attestation fields are audit metadata, not digital-signature verification.
The workflow never creates or promotes `benchmark_ready`; screen readiness
remains a separate registry
decision requiring all design, quantitative-data, rights, family, sample-map,
and label gates.

The three implemented CLI stages are shown below. Set
`EVENT_PACKET_RANDOMIZATION_SEED` to a batch-specific secret outside Git before
preparing the packet.

```bash
crispr-evidencerank prepare-event-review-packet \
  --evidence-cards ../curation-secure/batch_001/evidence_cards.tsv \
  --actor-registry ../curation-secure/curation_actors.tsv \
  --rubric config/event_review_rubric.yaml \
  --randomization-seed "$EVENT_PACKET_RANDOMIZATION_SEED" \
  --access-isolation-attested \
  --neutral-card-audit-attested \
  --output-dir ../curation-secure/batch_001/reviewer_packet

PACKET_MANIFEST_SHA256=$(sha256sum \
  ../curation-secure/batch_001/reviewer_packet/event_packet_manifest.json \
  | cut -d ' ' -f 1)
crispr-evidencerank compare-event-reviews \
  --packet-dir ../curation-secure/batch_001/reviewer_packet \
  --primary-reviews ../curation-secure/batch_001/curator_a.tsv \
  --secondary-reviews ../curation-secure/batch_001/curator_b.tsv \
  --actor-registry ../curation-secure/curation_actors.tsv \
  --compared-at-utc 2026-09-04T12:00:00+00:00 \
  --expected-packet-manifest-sha256 "$PACKET_MANIFEST_SHA256" \
  --output-dir ../curation-secure/batch_001/comparison

COMPARISON_MANIFEST_SHA256=$(sha256sum \
  ../curation-secure/batch_001/comparison/event_review_comparison_manifest.json \
  | cut -d ' ' -f 1)
crispr-evidencerank finalize-event-adjudication \
  --comparison-bundle-dir ../curation-secure/batch_001/comparison \
  --adjudications ../curation-secure/batch_001/signed_adjudications.tsv \
  --released-at-utc 2026-09-04T14:00:00+00:00 \
  --expected-comparison-manifest-sha256 "$COMPARISON_MANIFEST_SHA256" \
  --output-dir ../curation-secure/batch_001/final
```

These paths deliberately place operational curation outside the repository.
Private source PDFs, copied figures, source workbooks, completed reviewer
forms, and access-controlled extracts must remain outside Git. Git may retain
the protocol, schemas, code, artificial/synthetic fixtures, lawful source
links and locators, and only those derived records approved for release.

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
| Lite | MAGeCK gene summary + declared tail semantics | Implemented end-to-end report |
| Full | FASTQ + library + sample sheet | Contract defined; workflow pending |

The Standard mode is the primary modeling input because it preserves
guide-level evidence while avoiding mandatory redistribution of large FASTQ
files.

## Feature profiles

The model is evaluated as nested information profiles rather than pretending
that every user supplies equivalent data:

| Profile or endpoint | Current status | Evidence used or intended role |
|---|---|---|
| `screen_only` | Implemented benchmark profile | Gene scores/count-derived guide signal and technical QC. |
| `screen_plus_design` | Implemented benchmark profile | `screen_only` plus structured study/screen/contrast/sample metadata. |
| `context_aware` | Specified; modeling profile not implemented | Intended to add versioned artifact/CNV, cell, drug, pathway, and outer-training-only cross-screen evidence. The report-only immune and clinical modules do not populate this profile. |
| `selection_model` | Baseline endpoint implemented | A separate model for which genes authors chose to test. It currently receives the same reviewed pre-outcome feature columns as the paired success model; dedicated selection-only covariates are planned, not implemented. |

Only `screen_only` and `screen_plus_design` are currently selectable through
`benchmark --feature-profile`. The success-model allowlist prevents
author-selection, validation-outcome, clinical-registry, and other report-only
fields from entering either implemented success profile. Adding dedicated
selection-only covariates requires a separate runtime interface and leakage
review; the configuration declaration alone does not make those covariates an
implemented model input.

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
   The ICRAFT-inspired immune-context layer remains a separate report-only layer;
   it cannot enter the success model from a mutable current snapshot.
   Clinical trial-registry context is a separate treatment-by-cancer report and
   is never repeated as gene-level support.
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
  --positive-lfc-means resistance \
  --output data/processed/synthetic_gene_features.csv
crispr-evidencerank featurize-design \
  --screens examples/synthetic/screens.csv \
  --screen-designs examples/synthetic/screen_designs.csv \
  --contrasts examples/synthetic/contrasts.csv \
  --samples examples/synthetic/sample_sheet.csv \
  --output data/processed/synthetic_design_features.csv
crispr-evidencerank rank-screen \
  --counts examples/synthetic/guide_counts_screen_01.csv \
  --samples examples/synthetic/sample_sheet_screen_01.csv \
  --positive-lfc-means resistance \
  --output-dir data/processed/synthetic_screen_report
pytest
```

The synthetic data demonstrate the software contract only and must never be
used as scientific evidence.

`featurize-counts` expects integer raw read counts and an explicit declaration
of whether a positive treatment-minus-control LFC means `resistance` or
`sensitization`; it never infers this from a comparator name. The native LFC
sign is retained separately from the biological direction. A positive-count
median-ratio variant is the default; all-zero and otherwise uninformative rows
are excluded from size-factor estimation. This avoids the compositional
artifacts that total-count CPM can create when a small number of guides expand
strongly. Use `--normalization-method cpm` only as a declared sensitivity
analysis.

For a MAGeCK `gene_summary` file, direction semantics are mandatory rather than
guessed:

```bash
crispr-evidencerank rank-screen \
  --mageck-summary gene_summary.txt \
  --screen-id mda_mb_468_olaparib \
  --contrast-id olaparib_vs_vehicle \
  --positive-tail-means resistance \
  --output-dir results/olaparib_screen
```

The bundle contains `ranked_candidates.tsv`, `qc_summary.json`,
`run_manifest.json`, and `report.md`; the manifest checksum-binds the three
non-manifest outputs. If counts and a sample sheet are supplied
with the MAGeCK file, native MAGeCK rank remains primary and guide agreement,
low-count fractions, zero-count fractions, and replicate correlations are
added as QC evidence. The command preserves all observed positive and negative
tail rows; it does not silently apply an FDR or top-*N* candidate filter.

## Auxiliary immune-context analysis

The ICRAFT-inspired module asks a separate translational question after the
primary screen-signal ranking: does the same perturbation appear favorable or
harmful in tumor and immune cells? It preserves the native CRISPRa sign, keeps
modalities separate, collapses correlated source/raw families, enforces a
temporal cutoff, and abstains from RRA unless complete ranked-list rosters are
verified.

```bash
crispr-evidencerank summarize-immuno-context \
  --evidence immune_screen_evidence.tsv \
  --candidates results/olaparib_screen/ranked_candidates.tsv \
  --cutoff-date 2026-08-24 \
  --target-modality CRISPR_KO \
  --exclude-raw-data-family TARGET_SCREEN_RAW_FAMILY \
  --dual-action-group-id ko_antitumor_function \
  --dual-action-group-version reviewed-2026-08-24 \
  --output-dir results/olaparib_screen/immune_context
```

The repository implements the contract and analysis engine but does not bundle
an ICRAFT database export. A real import requires a frozen checksum-pinned
export, row-level original-source provenance, source/raw-family mapping, and
reuse-rights review. The immune bundle contains `immune_context.tsv`,
`immune_context_exclusions.tsv`, `immune_context_used_evidence.tsv`,
`rank_list_audit.tsv`, and `summary.json`; primary screen axes are retained in
the joined report. See `docs/IMMUNE_CONTEXT_METHOD.md`.

## Clinical treatment-by-cancer context

The clinical module describes the interventional-study landscape for one exact
normalized treatment concept and one exact normalized cancer concept. It does
not join the same trial metadata to every gene, calculate efficacy, or modify a
CRISPR rank. Every normalized row must resolve to a checksum-pinned source asset,
and only evidence available by the declared cutoff is used.

ClinicalTrials.gov acquisition is a separate operation. The search strings are
recall-oriented registry queries: a returned study is not an exact treatment or
cancer concept match. The intake saves exact API response bytes and produces an
unreviewed study inventory and curation queue. Every intervention/condition row
in that queue is only a study-level co-mention; it has no reviewed same-arm or
same-cohort linkage and is ineligible for `summarize-clinical-context`, gene
ranking, model features, and validation labels.

```bash
crispr-evidencerank fetch-clinicaltrials-gov \
  --condition-query "triple-negative breast cancer" \
  --intervention-query olaparib \
  --output-dir data/external/ctgov_olaparib_tnbc_snapshot

crispr-evidencerank verify-clinicaltrials-gov \
  --snapshot-dir data/external/ctgov_olaparib_tnbc_snapshot
```

The snapshot contains `version_start.json`, every numbered raw page,
`version_end.json`, `study_inventory.tsv`, `curation_queue.tsv`,
`data_assets.tsv`, and `manifest.json`. Offline verification reparses every raw
page, regenerates all three TSV files, and requires exact byte equality.
The TSV inventory is a deterministic projection: raw JSON, not the TSV, is
authoritative for distinctions among missing, null, and empty collections.
Completeness means only that the exact
recorded query and field projection were traversed through the recorded opaque
page-token chain, with unique NCT identifiers and an observed count equal to
the API `totalCount`. Stable API version and data-timestamp values before and
after pagination are an integrity envelope, not transactional snapshot
isolation and not proof that a broader or differently phrased query would find
no additional records. Real snapshots remain outside Git; only explicitly
synthetic registry fixtures may be checked in.

```bash
crispr-evidencerank summarize-clinical-context \
  --evidence examples/synthetic/clinical_context/evidence.tsv \
  --assets examples/synthetic/clinical_context/assets.tsv \
  --treatment-concept-id NCIT:C71721 \
  --treatment-mapping-source NCIt \
  --treatment-mapping-version 2026-08-01 \
  --cancer-concept-id NCIT:C71732 \
  --cancer-mapping-source NCIt \
  --cancer-mapping-version 2026-08-01 \
  --cutoff-date 2026-08-31 \
  --output-dir data/processed/synthetic_clinical_context
```

The example represents olaparib and triple-negative breast carcinoma with
synthetic registry rows only. Curator-reviewed exact concept matching excludes
broad breast cancer, other PARP inhibitors, observational records, and
non-experimental intervention mentions. A zero count means
`not observed in the supplied snapshot`, not `no trials exist`. Registry status,
phase, and the presence of posted aggregate results are descriptive metadata,
not evidence that an endpoint was met. See `docs/CLINICAL_CONTEXT_METHOD.md`.

Any retained or redistributed ClinicalTrials.gov material must follow the
[official Terms and Conditions](https://clinicaltrials.gov/about-site/terms-conditions),
including ClinicalTrials.gov attribution, display of the date the source data
were processed, and disclosure of project modifications and their date. The
intake manifest records the registry data timestamp, retrieval times, raw and
derived checksums, and transformation identity; it does not itself grant
redistribution rights or imply endorsement.

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
- one human-approved, preregistered primary estimand; prevalence-normalized AP,
  top-k coverage, direction strata, and documented-success diagnostics;
- source-family cluster intervals, paired model-minus-baseline intervals, and
  a prespecified split/model seed sensitivity analysis;
- ablation of guide, CNV/artifact, recurrence, context, and literature layers;
- a frozen-model prospective validation study.

The current `benchmark` command emits a strict JSON report containing point
estimates, informative-query denominators, prevalence, calibrated-score
diagnostics, direction strata, available fixed-score baselines, paired
candidate-minus-baseline bootstrap differences, seed sensitivity, and an audit
of features invariant within ranking queries. `documented_success_*` metrics
count missing outcomes only as *not documented*; they are not biological
failure rates and are not an eligible primary endpoint without an
outcome-complete denominator.

Top-k ties that cross a cutoff use equal fractional inclusion rather than row
order or outcomes. Queries with only one adjudicated class are counted but
excluded from inferential ranking macros because they have no within-query
ordering resolution. MAGeCK and drugZ baseline columns are accepted only after
explicit direction semantics, source version/locator, and source SHA-256 have
been supplied; a suggestive column name is not treated as evidence of correct
orientation.

Fold assignment and metric aggregation are separate axes: connected
source/raw-data families are kept in one outer fold even when a custom split
group is requested, while `study_macro_*` values remain aggregated by the
explicit study identifier. Source-family bootstrap draws never silently turn a
study-macro estimand into a family-macro estimand.

Benchmark input bytes are snapshotted and rechecked before no-clobber
publication. The JSON records input, prediction, packaged-build, and dependency
lock hashes. Prediction and metrics paths cannot overwrite the input or each
other, and a statistical failure produces no partial output.

After the real training roster, primary estimand, features, and model choice are
approved, `freeze-model` writes an atomic checksum-bound joblib bundle. Its
manifest pins the full Git SHA, training table, embedded feature registry,
embedded environment lock, and model bytes. For a repository release, pass
`--current-checkout-path` so the declared revision is checked against a clean
Git checkout both before and after fitting. `verify-frozen-model` checks the
fixed bundle roster without
deserializing it. The freeze command reports the manifest SHA-256; record that
hash in an external registration and pass it back as
`--expected-manifest-sha256` to verify bundle identity as well as internal file
integrity. Without an external anchor, no self-contained bundle can prove that
its manifest was not replaced. External timestamped deposition and the wet-lab
experiment remain separate human-controlled gates.

See `docs/SCIENTIFIC_PROTOCOL.md`, `docs/MODEL_CARD.md`, and
`docs/REPORTING_STANDARDS.md`.

For the adjacent pre-screen task, [AssayBench](https://github.com/Genentech/AssayBench)
benchmarks prediction of a screen gene ranking from an assay description. It is
an important landscape comparator, but it predicts the screen result before
screening rather than the orthogonal reproducibility of an already observed
guide-level hit.
