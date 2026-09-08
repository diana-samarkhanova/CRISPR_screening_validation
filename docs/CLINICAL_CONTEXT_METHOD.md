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

The treatment-by-cancer report engine remains offline: `summarize-clinical-context`
never queries a registry. A separate snapshot command may retrieve current
ClinicalTrials.gov records, but its outputs are an unreviewed intake queue, not
`clinical_trial_evidence`. This separation prevents a mutable current registry
search from silently entering a retrospective analysis.

## Frozen ClinicalTrials.gov intake

`fetch-clinicaltrials-gov` queries the official ClinicalTrials.gov v2 studies
endpoint with one condition search string and one intervention search string.
It pins the exact query text, response format, markup format, page size, and
scientific field projection. The client follows the API's opaque
`nextPageToken` values rather than synthesizing offsets or assuming a sort
order. Retrieval is bounded by page, study, intervention-by-condition candidate
rows, response bytes, total raw bytes, total derived bytes, elapsed time,
request timeout, and retry limits.

```bash
crispr-evidencerank fetch-clinicaltrials-gov \
  --condition-query "triple-negative breast cancer" \
  --intervention-query olaparib \
  --output-dir data/external/ctgov_olaparib_tnbc_snapshot

crispr-evidencerank verify-clinicaltrials-gov \
  --snapshot-dir data/external/ctgov_olaparib_tnbc_snapshot
```

The atomically published bundle contains:

- `version_start.json` and `version_end.json`: exact API version-response bytes
  obtained immediately before and after pagination;
- `pages/page_XXXXXX.json`: every exact studies-response byte stream;
- `study_inventory.tsv`: deterministic projected study metadata with JSON source
  locators and a binding to the containing raw page;
- `curation_queue.tsv`: fail-closed treatment/condition search co-mentions;
- `data_assets.tsv`: SHA-256, byte size, URL, retrieval time, and raw-family
  provenance for every retained API response;
- `manifest.json`: query identity, field projection, page-token lineage,
  per-page NCT roster and checksums, version envelope, safety limits, software
  identity, output hashes, and scientific boundary.

`verify-clinicaltrials-gov` operates offline. It checks the manifest roster,
file paths and hashes, page-token chain, unique NCT identifiers, `totalCount`,
before/after version envelope, and source-asset bindings. It also reparses every
frozen page, regenerates all three TSV outputs, and compares their exact bytes;
changing and rehashing a derived row therefore fails verification. Verification
establishes internal consistency of retained bytes and their derivatives; it is
not publisher authentication, a rights decision, or clinical curation.
The projected TSV intentionally normalizes an absent, explicit-null, or empty
source collection to the same empty list; only the retained raw JSON preserves
that structural distinction and is authoritative for source-level missingness.

Programmatic injected transports are recorded as injected, with their clock
modes stated explicitly; they never attest a live HTTPS retrieval. Checked-in
synthetic fixtures additionally require explicit markers in every version/page
response and use project-only identifiers and reference URLs.

### Search and completeness semantics

ClinicalTrials.gov `query.cond` and `query.intr` are recall-oriented text-search
inputs. A hit is not an exact ontology mapping. Source conditions may be broad
(`Solid Tumor`, for example), while subtype, biomarker, disease setting, or
cohort detail appears only in a title, description, or eligibility criteria.
Conversely, a treatment and disease term may occur in different arms or
cohorts. The intake therefore does not infer any of the following:

- exact treatment or cancer concepts;
- treatment intervention role or complete regimen;
- same-arm or same-cohort treatment-cancer linkage;
- subtype, biomarker, line-of-therapy, or disease-setting eligibility;
- efficacy or whether an endpoint was met.

Each generated queue row is explicitly `not_performed` for mapping and linkage
review, `co_mention_only`, ineligible, and unused for labels. It cannot be fed
directly to `summarize-clinical-context`. A later human-reviewed transformation
must establish exact, versioned concept mappings, intervention role, regimen,
population scope, and arm/cohort linkage with source locators before creating a
candidate `clinical_trial_evidence` row.

An intake bundle may call its pagination complete only when the recorded token
chain terminates, tokens do not repeat, NCT identifiers do not repeat across
pages, the observed unique count equals the first-page `totalCount`, and the
API version plus data timestamp are unchanged before and after traversal. This
claim is scoped strictly to the exact recorded query and field projection. It
does not establish ontology recall, completeness for synonyms or broader
queries, or absence of records outside the returned result set.

The stable before/after version values are an envelope check, not proof of
transactional snapshot isolation. ClinicalTrials.gov is mutable, and records
could in principle change during a multi-page retrieval even when the envelope
values remain stable. Historical analysis still requires a snapshot genuinely
available by its cutoff; a current intake does not reconstruct past registry
state.

### Retention, terms, and redistribution

Real API pages and complete real snapshots remain outside Git, even when small.
The repository may include only clearly labeled synthetic registry fixtures
for deterministic tests. Local retention and any later distribution must be
reviewed against the current [ClinicalTrials.gov Terms and
Conditions](https://clinicaltrials.gov/about-site/terms-conditions). When
ClinicalTrials.gov content is reproduced, preserve source attribution, clearly
display the date the data were processed by ClinicalTrials.gov, and state the
project's modifications and the date of those modifications. The project also
records its own retrieval and transformation times rather than substituting
them for the registry processing timestamp.

The presence of a government-hosted registry record does not establish that
every submitted field is free of third-party or international copyright. A
checksum and terms URL document integrity and applicable conditions; neither
is a redistribution license, a publisher-authenticity guarantee, or an NLM
endorsement.

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
It does not mean that no trials exist. The report never infers snapshot
completeness from a TSV. When its reviewed inputs were derived from a verified
intake bundle, the manifest can support only the narrower claim that the exact
recorded query's page-token chain was completely traversed.

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
