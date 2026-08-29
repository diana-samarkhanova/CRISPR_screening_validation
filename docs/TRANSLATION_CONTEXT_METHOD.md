# Translation-context method

## Purpose

The translation-context layer answers questions that are intentionally
different from CRISPR-hit reproducibility:

1. How extensively is the treatment being studied in the stated disease?
2. Is there independently curated gene-specific evidence in treatment-matched
   patient cohorts?
3. Is there independently curated in-vitro, organoid/ex-vivo, or in-vivo
   evidence for the same gene, treatment, disease, and phenotype direction?

It does not filter or reorder the candidate table, modify the primary screen
rank, create a validation label, calculate a clinical/preclinical mega-score,
or estimate a validation/reproducibility probability.

## Typed separation

Four contracts prevent treatment-level and gene-level evidence from being
silently mixed:

- `treatment_disease_context`: one explicit treatment, disease, subtype,
  typed biomarker, screen modality, perturbed compartment, and endpoint-category
  question;
- `clinical_trial_context`: one unique NCT record, with no gene field and
  `used_for_gene_ranking=false`;
- `patient_molecular_evidence`: one curated gene/outcome claim from a cohort
  in which treatment, a timepoint-explicit molecular measurement, and clinical
  outcome coexist;
- `preclinical_evidence`: one curated treatment/model or gene-perturbation
  claim with explicit comparator, perturbed compartment, endpoint category,
  endpoint, provenance, and model type.

Ontology identifiers are optional only for a broad/compatible report, and an
identifier cannot be supplied without its ontology name and version. Strict
registry status and exact curated context require versioned canonical treatment
and cancer IDs; name-only identity remains compatible non-exact. TNBC, generic
breast cancer, BRCA1/2 mutation, and HRD are not interchangeable fields.
Scientific attestations accept literal booleans only; numeric or string truthy
values are rejected rather than coerced.

A biomarker request is an all-or-none typed tuple:
`biomarker_context`, `biomarker_feature_type`, `biomarker_state`,
`biomarker_specimen_type`, `biomarker_measurement_timepoint`, and
`biomarker_axes_observation_status`. A term such as
“BRCA” alone cannot substitute for mutation versus expression, altered state,
specimen, or measurement timing. Every supplied tuple also requires an explicit
`biomarker_axes_informative_verified` curator decision. An `exact_typed` match
requires `observation_status=observed` and a true attestation in both the
requested context and curated row; missing/unmeasured status or false
attestation remains unresolved and cannot become exact. The CLI records these
with `--biomarker-axes-observation-status observed` and
`--biomarker-axes-informative-verified`.

Subtype identity has an explicit ontology parent binding. A verified subtype
requires a subtype ID, cancer ID, matching `disease_subtype_parent_id`, and a
versioned disease ontology. Regimen identity likewise uses canonical active
exposure CURIEs, a component relation, identifier source/version, and curator
verification. Free-text labels alone never create exact subtype or regimen
identity. In v1, a verified requested or preclinical regimen is the singleton
canonical treatment ID; fixed combinations remain non-exact until a versioned
combination concept registry is implemented.

## ClinicalTrials.gov v2 adapter

The live adapter queries the official v2 `/studies` endpoint and calls
`/version` before and after pagination. A change in `apiVersion` or
`dataTimestamp` aborts publication, preventing a mixed source snapshot.
Pagination is bounded and follows `nextPageToken`; empty continuation pages,
repeated tokens, page-size violations, and a complete-page count inconsistent
with `totalCount` fail closed. An unfinished crawl is reported as truncated
rather than as absence of trials. Parsed pages have deterministic canonical
SHA-256 values; request URLs, exact API version, source data timestamp,
retrieval UTC, the frozen input bytes, and normalized outputs are
checksum-bound in the atomic bundle. Before returning, every live single-query
or concept snapshot is loaded through the frozen-snapshot validator and must
reproduce the same canonical document exactly. Duplicate NCT IDs within one
query fail closed. Identical payloads found through different query lanes are
deduplicated in the top-level NCT union, while conflicting payloads for one NCT
abort publication; frozen replay recomputes and verifies that union. Report
construction replays the snapshot again before normalization and rejects nested
document or metadata mutation after initial validation. Each typed query has one
validated before/after version audit. Only the stock live transport with an
actual completion timestamp receives a non-serializable capability bound to the
snapshot digest. Serialized replay, an injected transport, or an injected clock
is source-provenance-unverified and cannot emit a strict registry count.

`query.intr` and `query.cond` are broad discovery searches, not exact database
joins. Live mode separately paginates the role-tagged declared query set and
then deduplicates exact NCT records. Treatment terms are divided into canonical
name, same-entity aliases, and broader class terms; cancer terms are divided
into canonical name, same-entity aliases, and broader ancestor terms; subtype
queries likewise distinguish the canonical subtype from same-subtype aliases.
Class and ancestor terms expand discovery but never inherit entity-alias
semantics. Query identity and local disease matching preserve signed subtype
state: `HER2+`, `HER2`, `HER2-`, and spaced labels such as `HER2 + breast
cancer` are distinct. Typed curated matching applies the same sign-preserving
normalization to biomarker state and specimen values, including
`positive (+)`/`positive (-)` and `CD3+`/`CD3-` cells.

The saved typed query cross-product is bound to the requested context during
replay. Any typed term/role mismatch fails closed. Legacy or raw frozen pages
without role-tagged query provenance remain reportable, but their clinical lane
is `frozen_query_context_unverified` and their strict registry count is zero.
“Complete” therefore means complete for that declared query set; it does not
prove exhaustive ontology-level recall for undisclosed synonyms or unstructured
eligibility text. Local matches use only structured active intervention
names/other names, conditions, and keywords. Placebo-only mentions and
narrative background do not become intervention matches.

Treatment matching is stored independently from disease and biomarker
matching:

- `exact_canonical`, `explicit_alias` (alias-unverified), `explicit_component`,
  `declared_class_term`, or `no_structured_match` for intervention;
- `explicit_subtype_term`, `explicit_subtype_alias`, `cancer_type_term_only`,
  `cancer_entity_alias`,
  `declared_ancestor_term`, or `no_structured_match` for disease;
- explicit structured biomarker term, not reported in structured terms, or
  not requested;
- no additional active agent listed, additional active agent listed, or
  unresolved for the study-level regimen relation.

The study-level intervention list cannot prove that every participant received
every listed intervention. An additional agent or embedded combination-product
name is flagged, but is not called a participant-level combination arm without
arm-level adjudication.

The registry's strict status is computed only from requested axes that the
adapter can resolve from structured intervention and condition fields.
Only `exact_canonical` may satisfy the strict treatment entity axis. Alias,
`explicit_component`, and declared class matches remain non-exact discovery
context. A requested subtype needs an explicit subtype term and a positive,
versioned `disease_subtype_parent_binding_verified` assertion in the requested
context; a declared cancer ancestor cannot satisfy the disease entity axis.
The subtype term must also be bound to the requested parent cancer in the
structured condition list as a separate exact parent-cancer condition. A parent
name embedded in or inferred as a substring of the subtype label is not
accepted without a versioned, curator-attested parent-ID binding. A bare
subtype term is not strict disease evidence. A
structured biomarker keyword is retained as discovery metadata, but it does not
encode the typed feature/state/specimen/timepoint tuple and is never an exact
biomarker match. A biomarker-constrained question therefore cannot be promoted
to a strict typed registry match. Requested regimen, stage, or line of therapy
also remains unresolved and forces the strict registry count to zero until an
arm-assignment and eligibility parser can verify those axes; study-level listed
interventions alone are insufficient.

ClinicalTrials.gov `hasResults=true` means registry summary results are posted.
It does not imply patient-level RNA-seq, a reusable molecular cohort, or a
gene-specific response association. `COMPLETED` does not imply efficacy, and a
terminated trial is not automatically a negative efficacy result.

The current API exposes the latest record. A current record is marked
`current_snapshot_only` and cannot become a historical model feature unless
the exact record version at the historical cutoff is independently archived.

## Patient molecular evidence

A patient record must link the gene measurement and outcome to the same
treatment-matched cohort. The requested cohort-context biomarker tuple is not
the candidate gene predictor. Each row separately binds `gene_symbol` to the
same `predictor_gene_symbol`, a controlled versioned `gene_id`, predictor
feature/state/specimen, measurement type/platform, and measurement timepoint.
The predictor feature must be compatible with the declared assay. The required
`predictor_identity_curator_verified=true` value is an audited curator
attestation, not proof that an external identifier resolver was queried.
Interpretation remains typed:

- `predictive_interaction` requires a pretreatment measurement, versioned and
  curator-verified treatment/comparator active-exposure sets and relations,
  distinct source-native assignment IDs, per-arm/evaluable/model counts,
  predictor variation in both arms, verified estimability, a controlled effect
  scale, and a versioned inference rule whose supplied departure p-value and/or
  interval consistently exclude the controlled null;
- `interaction_tested_null` requires a confidence interval wholly inside
  prespecified equivalence bounds and, when supplied, a significant
  equivalence-to-null p-value;
- `interaction_tested_inconclusive` requires discordant departure p-value and
  confidence-interval decisions under the declared rule;
- `interaction_tested_unsupported` requires that no supplied metric satisfy the
  departure rule;
- `treated_cohort_association` is an association within treated patients and
  is not described as treatment-predictive;
- `prognostic_only`, `pharmacodynamic`, `acquired_resistance`,
  `on_treatment_association`, `post_progression_association`,
  `eligibility_only`, `descriptive_only`, and `unresolved` remain distinct.

Free-text arm names cannot override the controlled active-exposure sets, and
identical source-native assignment IDs do not establish a treatment-by-
predictor interaction. V1 formal claims support canonical monotherapy versus a
placebo/no-active comparator; active and unresolved comparators abstain until a
versioned concept registry exists. Significance in treated patients plus
non-significance in controls is not an interaction test. On-treatment RNA-seq
is not a pretreatment predictor, and a post-progression association is not
called acquired resistance. A
`pharmacodynamic` or `acquired_resistance` claim additionally requires paired
baseline material and a longitudinal change test; acquired resistance and the
post-progression lane require documented progression. A `prognostic_only`
predictor must be measured pretreatment/baseline; a later measurement belongs
in an explicitly time-dependent lane.

Exact patient context in v1 requires exact agreement on every strict axis:
treatment/disease ontology identity, subtype, the full cohort-context typed
biomarker tuple, regimen, stage, and line of therapy. Omission on either side is
not a wildcard and cannot become exact; consequently an intentionally broad or
incompletely specified question remains compatible non-exact. The separate
candidate-gene predictor fields identify the tested molecular variable and do
not substitute for any cohort-context axis. A patient row with unverified
treatment exposure cannot establish exact regimen/treatment context even when
its labels and active-exposure IDs match. Non-exact context is partitioned
instead of pooled: name-only or ontology-version-unresolved identity and missing
requested setting fields remain `context_axis_unresolved` or
`evidence_narrower_than_context`, while an
explicit subtype, biomarker, regimen, stage, or line-of-therapy contradiction
is `conflicting_context`. A treatment or cancer identity mismatch is excluded
from the matched evidence lane. Neither compatible non-exact nor conflicting
evidence can produce an exact predictive status.

If no valid treatment-matched patient record is supplied, the result is
`insufficient_matched_patient_data` or
`no_match_in_provided_curated_table`. It is never a biological negative.

## Preclinical evidence

Model types remain separate rather than forming an assumed quality ladder:

- 2D/3D cell line and immune co-culture;
- organoid, PDX-derived organoid, and ex-vivo tissue;
- cell-line xenograft, PDX, syngeneic, genetically engineered, humanized-mouse,
  and other in-vivo models.

Claim types also remain separate:

- `direct_perturbational_interaction`;
- `natural_biomarker_association`;
- `treatment_activity_only`;
- `mechanistic_only`.

Every gene-specific preclinical row binds its gene symbol to a versioned gene
CURIE/source/release and a curator identity attestation. The attestation records
manual curation and is not external resolver authentication. Treatment-context
rows cannot carry this gene identity bundle.

A direct gene-perturbation claim always requires a vehicle/baseline-growth
control and an explicit genotype-by-treatment test. Every non-unknown direction
also requires a stable versioned direction rule, a curator-verified direction
inference, a numeric effect, and a reported sample size. Resistance,
sensitization, and discordant calls map to `direction_supported`, while neutral
maps to `neutral_supported`. An unknown direction must instead be explicitly
`inconclusive`, `unsupported`, or `not_assessed`. Neutral or discordant calls require a prespecified
decision rule. These statuses document how the curator mapped the reported
experiment to a direction; they are not themselves a new statistical test or a
calibrated confidence score. Non-direct claims cannot carry a perturbation
modality. Natural low expression, copy-number loss, drug-response correlation,
RNAi, CRISPRa, and CRISPR KO are not silently treated as equivalent
perturbations, and directional concordance is counted only for the same
perturbation modality as the target screen and a contract-valid directional
inference.

Every preclinical row also declares `perturbed_compartment` and
`endpoint_category`. Exact preclinical context requires both axes to match the
target screen in addition to treatment, disease/subtype, typed biomarker, and
regimen. An explicit compartment or endpoint-category difference is conflicting
context and cannot contribute to exact-context or direction-concordant counts.
Missing or ontology-version-unresolved non-contradictory axes remain compatible
non-exact context. An axis omitted from the requested context is not a wildcard:
evidence with a narrower declared axis remains non-exact.

Candidate summaries expose the partitions separately. The principal columns
include `report_only_preclinical_compatible_nonexact_context_family_n` and
`report_only_preclinical_conflicting_context_family_n`, plus corresponding
patient counts. Predictive patient evidence is separately reported as
`report_only_patient_predictive_exact_context_family_n`,
`report_only_patient_predictive_compatible_nonexact_context_family_n`, and
`report_only_patient_predictive_conflicting_context_family_n`. Statuses likewise
distinguish `compatible_nonexact_context_only` from `conflicting_context_only`
and the corresponding patient forms
`compatible_nonexact_patient_context_only` and
`conflicting_patient_context_only`. When both non-exact partitions are non-empty
and no higher-priority exact or independence status applies, the status is
instead `compatible_and_conflicting_context_present` or
`compatible_and_conflicting_patient_context_present`; it is never mislabeled
with an `_only` status.

Supported, formal-null, inconclusive, and unsupported interaction masks each
have exact, compatible-non-exact, and conflicting family counts. Supported plus
any nonconfirmatory exact result emits
`supported_and_nonconfirmatory_interactions_present`; two or more distinct
nonconfirmatory exact result types emit
`multiple_nonconfirmatory_interaction_results_present` rather than hiding the
mixture behind one priority status.

## Leakage, dates, and independence

Patient and preclinical claims are filtered by `available_date`, not retrieval
date. Both source and curator-assigned raw-data family identifiers are required
and collapsed transitively. Publications or reanalyses known to use the same
cohort, experiment, count matrix, or patient-level dataset must share one
`raw_data_family_id`; only then are they guaranteed not to vote repeatedly.
`cohort_id` is source-local descriptive metadata, not a global deduplication
key. Independence therefore remains conditional on the audited family mapping.
The target source/raw family is excluded transitively across linked source and
raw-data identifiers. Merely supplying an identifier that is absent from the
table does not verify independence; the identifier must resolve in the family
graph or target absence must be explicitly attested. A contradictory absence
attestation fails closed.

Same-study targeted validation belongs in `validation_event` as a potential
label. It cannot re-enter this report as prior support for its own label.

All candidate-derived columns begin with `report_only_`. The incoming row order,
candidate membership, screen/contrast axes, direction, and rank are preserved.
The CLI treats a candidate TSV as ranked only when it supplies both
`ranking_type` and `screen_signal_rank`; supplying just one fails closed. Such a
table must use `ranking_type=screen_signal_baseline` and provide the complete
versioned `rank-screen` bundle. Validation covers the manifest/method schema,
input mode and parameters, all output hashes, ordered candidate schema and row
count, identifiers, tail/direction/rank/percentile/neutral semantics, duplicate
keys, and canonical ordering. This is an unsigned internal-consistency check,
not producer authentication. `--candidate-manifest` cannot be attached to a
structurally unranked table. Without both ranking columns, order is explicitly
recorded as unranked or unverified input order.
The success-model feature validator derives its field deny-list from every
report-only evidence contract and also applies reserved leakage-token checks.
No translation-context count or status is a probability.

## Bundle

`summarize-translation-context` writes:

- `context.json`;
- `clinical_trials.tsv`;
- `clinicaltrials_snapshot.json`;
- separate used and excluded preclinical and patient evidence tables;
- `candidate_translation_context.tsv`;
- `missingness.tsv`;
- `report.md`;
- `summary.json` with input and output SHA-256 values.

The command reads each file from the same bytes that it hashes, rechecks inputs
both before bundle construction and immediately before publication, refuses to
overwrite an existing directory or an input, and atomically renames a complete
staging bundle into place. A live retrieval cannot be backdated by CLI option,
and a wrapped frozen snapshot cannot be restamped.

## Model applicability

The current reproducibility model is in scope only for a small-molecule,
tumor-cell, CRISPR-KO, drug-response/viability screen. A CAR-T trial landscape
can still be retrieved, but a CAR-T, immune-killing, immune-cell, CRISPRa, or
CRISPRi query is marked `not_applicable` to the V1 model. Clinical context is
not a justification for forcing an out-of-distribution screen through the
small-molecule predictor or for reranking its candidates.
