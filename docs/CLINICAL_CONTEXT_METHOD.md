# Clinical treatment-by-cancer context

## Purpose and scientific boundary

The clinical-context method reports the observed interventional-trial landscape
for one exact normalized treatment concept and one exact normalized cancer
concept. It is independent of candidate genes and cannot alter a CRISPR screen
score, rank, validation label, or validation-success model.

ClinicalTrials.gov records are sponsor- or investigator-submitted registry
records. A registered or completed study is not evidence that a treatment is
effective; `hasResults`/`results_posted` means that aggregate result tables are
present, not that an endpoint was met and not that participant-level data are
available. See the official [ClinicalTrials.gov API
documentation](https://clinicaltrials.gov/data-api/about-api) and [site
disclaimer](https://clinicaltrials.gov/about-site/disclaimer).

Version 0.4 is an offline, frozen-input report engine. It does not perform a live
API query. A future intake adapter must preserve every raw response page,
pagination token, API version, data timestamp, query, retrieval time, checksum,
and exclusion. This separation prevents a mutable current registry snapshot
from silently entering a retrospective analysis.

## Canonical row and source binding

One `clinical_trial_evidence` row represents:

```text
source study × treatment concept × cancer concept
```

Names are display and provenance fields. Matching uses only exact, versioned
concept identifiers. Each treatment and cancer mapping also records its
relation to the source term and review status. Only `exact` +
`curator_reviewed` mappings are eligible. Class-only, substring, brand,
approximate, source-asserted-only, or unreviewed automated matches cannot pass
the report filter by themselves. A curator-reviewed mapping must carry a review
event identifier and date; that date cannot follow transformation availability.
The query pins both mapping sources and releases, so rows from another ontology
or release are excluded even when the bare concept ID is identical.

Every row is bound to a `DataAssetRecord` through `source_asset_id` and
`source_asset_sha256`. The source asset must exist, carry a locally observed
SHA-256, and agree on source name and version. This establishes which frozen
bytes were normalized; it does not establish publisher authenticity or transfer
rights.

Required temporal fields distinguish:

- source snapshot date;
- source record last-update date;
- first availability date;
- transformation availability date;
- retrieval date;
- results-first-posted date, when aggregate results are present.

The effective evidence date is the latest of the source snapshot, record
availability, and transformation availability dates. Evidence after the
declared cutoff is excluded. A current record cannot reconstruct a historical
status; retrospective work requires a genuinely historical snapshot or an
official record version available by the cutoff.

## Eligibility and exclusions

Filtering is deterministic and reason-coded in this order:

1. evidence unavailable by the cutoff;
2. explicitly excluded source family;
3. treatment concept mismatch;
4. cancer concept mismatch;
5. treatment mapping source or version mismatch;
6. cancer mapping source or version mismatch;
7. treatment mapping is not exact or is not curator-reviewed;
8. cancer mapping is not exact or is not curator-reviewed;
9. non-interventional study;
10. treatment is not assigned the experimental intervention role.

Unknown requested source-family exclusions are rejected. Duplicate normalized
rows for the same source study, treatment concept, and cancer concept are
rejected rather than counted twice. A source study cannot map to multiple source
families.

Zero eligible rows means only `not observed in the supplied frozen snapshot`.
It does not mean that no trials exist. Snapshot completeness will require a
future pagination manifest and is not inferred from a TSV.

## Output and interpretation

The atomic bundle contains:

- `clinical_context.tsv`: one-row treatment-by-cancer summary;
- `clinical_context_studies.tsv`: eligible normalized study rows;
- `clinical_context_exclusions.tsv`: all excluded rows with reason codes;
- `clinical_context_used_assets.tsv`: checksum-pinned source assets;
- `summary.json`: parameters, input/output checksums, and method metadata.

`summary.json` also records the package and Python versions, exact runtime
dependency versions, the packaged dependency-lock and clinical-schema hashes,
and an optional build revision supplied by the build environment. The same
metadata file is included in installed wheels, so provenance does not depend on
running from a source checkout.

Apart from the six pinned query identity fields, all summary columns begin with
`report_only_clinical_`. Counts are distinct observed source-family counts.
Phase, status, regimen, and results-posted counts are descriptive strata only.
The method does not compute a clinical score, efficacy score, maximum-phase
score, meta-analysis, or treatment recommendation.

The writer refuses an existing destination, cannot overwrite an input file,
rechecks input hashes before publication, writes to a same-parent staging
directory, and publishes with one atomic directory rename.

## Olaparib and TNBC reference query

The synthetic example uses:

- olaparib: NCIt `C71721`;
- triple-negative breast carcinoma: NCIt `C71732`.

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
  --output-dir results/olaparib_tnbc_clinical_context
```

Exact TNBC matching deliberately excludes a broad breast-cancer row and a
different PARP inhibitor. An active-comparator or prior-exposure mention is not
treated as the experimental intervention. A combination trial is reported as a
complete regimen; its outcome cannot be attributed to olaparib alone.

The current US olaparib breast indications are conditional on germline
pathogenic or suspected pathogenic BRCA1/2, HER2-negative disease, and a
specific early-high-risk or metastatic setting. TNBC alone, HRD alone, somatic
BRCA alone, or generic BRCAness is not an unconditional breast-cancer approval
predicate. The current label must be checked by jurisdiction and revision; see
the [DailyMed Lynparza label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=741ff3e3-dc1a-45a6-84e5-2481b27131aa).

For MDA-MB-468, the defensible statement is `TNBC phenotype matched; approved
patient biomarker match not established`. A cell-line BRCA1 LOH/HRD-like state
is not equivalent to a confirmed germline pathogenic BRCA1/2 variant in a
patient.

## Separate future evidence strata

Trial registry metadata must not be collapsed with:

- jurisdiction-specific regulatory indications;
- arm- and endpoint-level aggregate results;
- primary publications;
- treatment-exposed patient molecular cohorts;
- untreated tumor multi-omics such as TCGA/GDC;
- paired pre/post-treatment translational evidence.

TCGA-BRCA can supply disease biology, expression, CNV, mutation, and prognostic
context. Without documented olaparib exposure and response, it cannot validate
an olaparib-response biomarker. Likewise, a clinical treatment-by-cancer report
does not support any particular CRISPR gene hit.
