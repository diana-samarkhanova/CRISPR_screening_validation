# CRISPR-EvidenceRank seed study manifest

## Purpose

This is a **seed discovery manifest**, not a frozen training cohort. It identifies published human pooled CRISPR-Cas9 knockout drug-response or chemogenetic screens that are promising for:

1. reconstruction from FASTQ or sgRNA count tables;
2. extraction of screen-level and guide-level features;
3. manual annotation of targeted experimental validation; and
4. grouped benchmarking across studies, drugs, cancer types and cell lines.

The manifest was assembled on 2026-07-30 from primary papers and official repositories where available. BioGRID ORCS is used only as a discovery/index source when a primary repository accession was not located. A downloadable supplementary file is **not** assumed to be legally redistributable.

## Scope of this seed

- Human cell systems.
- Pooled CRISPR-Cas9 knockout screens.
- Drug-response, resistance, sensitization or chemical-genetic interaction phenotypes.
- Genome-wide screens are prioritized, but several focused DDR or metabolic libraries are retained because they have unusually useful validation evidence.
- CRISPRa arms in mixed papers are mentioned but should be excluded from the first KO model.
- Essentiality-only, infection, differentiation, immune-cell killing and non-drug phenotypes are excluded from this seed.

There are 27 study-level records. A record can contain several drugs, cell lines or screen arms; the next curation pass must expand it into one row per `screen_id`.

## Verification status

| Status | Meaning | Appropriate use now |
|---|---|---|
| `VERIFIED_RAW` | A public raw-data project or series was located in an official repository and linked to the study. | Eligible for sample-level accession curation and reprocessing. |
| `VERIFIED_COUNTS` | An official source/supplementary count or score file was located, but raw reads were not confirmed. | Eligible for processed-data ingestion after schema and license review. |
| `VERIFIED_METADATA` | The primary screen and validation are verified, but a reusable raw/count location is unresolved. | Literature/validation registry only; do not place in the trainable cohort yet. |
| `RESTRICTED_RAW` | A raw accession exists but access or redistribution is controlled. | Use only after authorization; public source data may still support limited derived features. |

`VERIFIED_*` is field-level evidence, not a judgment that every column is complete. The `verified_fields` and `unresolved_fields` columns are therefore mandatory.

## Immediate high-value ingestion set

The best first ingestion wave is:

1. **Lau 2020 (`CER-S016`)** — 27 anticancer compounds, public BioProject, sgRNA-level and gene-level tables, MAGeCK baseline and extensive validation.
2. **Olivieri 2020 (`CER-S015`)** — large DNA-damage chemical-genetic map with a versioned public dataset.
3. **Clements 2020 (`CER-S014`)** — directly supplied screen read-count tables, two olaparib contexts and strong KO validation.
4. **Nechiporuk 2019 (`CER-S008`)** — public raw reads and clear successful targeted validations.
5. **Chen 2019 (`CER-S007`)** — bidirectional screen signal and mechanistic follow-up; use the supplementary screen data, because `GSE125403` is companion RNA-seq rather than CRISPR-screen sequencing.
6. **Damnernsawad 2022 (`CER-S022`)** — public GEO raw data, explicit time points and MAGeCK RRA.
7. **Pettitt 2018 (`CER-S004`)** — public ENA project and unusually strong causal validation.
8. **Tsujino 2023 (`CER-S024`)** — four prostate cancer models, source data and rescue-level follow-up.
9. **Lin 2024 (`CER-S025`)** — public BioProject and useful time-resolved screen QC, but no positive validation labels.

This wave deliberately mixes studies with positive validation labels and a study useful mainly for reproducibility/QC. It should not be treated as a balanced classification dataset.

## Required normalization before model training

Each study-level row must be expanded into:

```text
Study
  └── ScreenArm
        ├── cell_line
        ├── genetic_background
        ├── drug
        ├── dose
        ├── duration
        ├── replicate
        ├── control
        ├── library
        ├── count_file / run_accession
        └── analysis_output
```

Then create a separate `ValidationEvent` table keyed by:

```text
study_id × screen_id × gene × direction × reagent × assay × context
```

The paper-level flag `individual_validation_reported=yes` must **never** become a gene-level label by itself. Every positive or failed label needs a figure/table/methods locator and enough detail to distinguish:

- independent sgRNA versus reuse of a screening guide;
- perturbation confirmed versus assumed;
- phenotype reproduced versus not reproduced;
- same drug/cell context versus a different context;
- rescue or orthogonal validation versus supportive association only.

## Label cautions

- Hits selected for follow-up are affected by author choice, tractability, novelty and prior literature.
- Untested genes are unlabeled, not negatives.
- Papers rarely report all failed validations. Explicit failures should be captured whenever authors state that a candidate was tested and did not reproduce.
- Mixed KO/CRISPRa papers require modality-specific labels and sample tables.
- Focused libraries are not directly comparable to genome-wide libraries without a library-membership feature.
- Studies with only processed gene scores cannot contribute guide-consistency or reanalysis-derived CNV features.
- `CER-S025` is valuable for reproducibility and time-course feature engineering but should not supply positive targeted-validation labels.
- `CER-S020` has public raw data, but the strongest experimental validation is from the CRISPRa arm; KO validation labels need conservative review.

## Reuse and copyright handling

Use a data manifest rather than copying third-party data into the repository by default.

- Keep repository accessions, source URLs, checksums and download scripts.
- Store locally generated count matrices and derived features only when repository and publisher terms permit.
- Do not assume that an open-access article license automatically applies to deposited sequencing reads, supplementary spreadsheets or third-party database annotations.
- Do not assume that public access equals permission to redistribute.
- Keep controlled-access data such as `HRA002646` outside any public release unless explicit authorization permits it.
- Record a license/provenance row for every imported file, including retrieval date and original filename.
- For publisher supplements with unclear reuse terms, extract factual annotations with precise citations instead of mirroring the files.

## Unresolved metadata queue

Priority manual checks:

1. Resolve sample/run accessions and sample labels for every `VERIFIED_RAW` record.
2. Inspect `GSE92742` before using it; it may correspond to RNA-seq rather than CRISPR-screen sequencing.
3. Confirm exact drug/dose/cell-line matrices for multi-arm studies (`CER-S002`, `CER-S015`, `CER-S016`, `CER-S026`).
4. Locate raw accessions for canonical but supplement-only studies (`CER-S001`, `CER-S005`, `CER-S014`, `CER-S018`, `CER-S024`).
5. Record exact licenses for Mendeley, AACR Figshare, BioStudies-hosted publisher supplements and every ORCS-derived file.
6. Recheck the exact screening cell line and analysis method for `CER-S013` and `CER-S027`.
7. Verify whether any supplementary table already contains guide-level counts versus only gene-level rankings.

## Inclusion decision for the first benchmark

A study should enter the first training benchmark only when all of the following are satisfied:

- at least one treated arm and a compatible control are identifiable;
- sample identities and replicate structure are recoverable;
- library guide sequences or an unambiguous library version are available;
- the sign convention for resistance/sensitization is known;
- processed values can be reproduced or traced to a documented method;
- at least one gene-level validation event can be curated, or the study is explicitly assigned to a QC-only role;
- reuse is permitted for the intended release mode.

Until those checks are complete, this file should be described as a **candidate-study manifest**, not a training dataset.
