# Gold-standard validation-event curation

## Purpose and status

This protocol defines the independent human review needed to create a Gold
validation set for CRISPR screen candidates. It separates source extraction,
independent annotation, reviewer comparison, and final human adjudication.

The unit of review is a **validation event**, not an entire paper and not a
gene in the abstract. An event is one coherent test of a defined
`gene × perturbation × cell model × treatment × endpoint` claim. Different
drugs, cell models, endpoints, or materially different experiments must be
recorded as separate events. Conflicting events must not be collapsed by a
curator.

Agreement between two curators is provisional. A Gold label exists only after
the two submissions have been frozen and a named human adjudicator has
verified the source evidence and recorded the final decision.

## Roles

- The **coordinator** freezes the source snapshot, prepares neutral evidence
  packets, maintains the confidential event crosswalk, and prevents exchange
  of reviewer decisions.
- **Curator A** and **Curator B** independently review the same evidence
  snapshot under the same protocol version.
- The **adjudicator** reviews the frozen annotations, discrepancies, and
  original sources and creates a named, time-stamped adjudication record. If a third person
  is unavailable, A and B may hold a documented consensus meeting only after
  both independent submissions have been frozen.

## Calibration before Gold review

Before reviewing a Gold batch, both curators must independently annotate the
same 10–15 calibration events covering positive, negative, ambiguous, and
technical cases. Calibration events must be maintained in a separate packet
and must be excluded from the Gold test set and all reported Gold performance
estimates.

The curators may discuss the calibration results. Any clarification produced
by that discussion must be added to this manual, assigned a new protocol
version, and frozen before the Gold batch begins. Label rules must not be
changed in response to results from the Gold batch.

## What Curator B receives

Curator B receives a self-contained, version-locked packet containing:

1. this manual and a one-page label reference;
2. a packet manifest, protocol version, source-snapshot date, and SHA-256
   checksums;
3. one neutral evidence card per event, identified by an opaque
   `packet_event_id` and presented in randomized order;
4. a source index containing DOI/PMID/accession, lawful access links, and exact
   figure, table, page, Methods, and supplement locators;
5. a blank event-level review form with no proposed answers or labels; and
6. a batch independence declaration for signature.

Both curators must receive the same frozen payload and SHA-256 for each card;
only presentation order may differ.

The normalized card must match `EventEvidenceCardRecord`. Its contract fields
are the packet identity/version/hash; batch, source-family, study, screen, and
contrast identifiers; gene; screen-predicted direction, drug, and cell model;
optional reported validation drug, cell model, perturbation modality, and assay
endpoint; source URL and locator; preparation time and actor; and optional
source-facing notes. The distributable packet requires the `notes` field to be
empty.

Each evidence card may pre-populate only verifiable descriptive facts and must
state the source locator for each fact. It should include:

- the gene and the claim being tested, stated without a quality judgment, for
  example, “loss of *GENE X* is expected to cause olaparib resistance”;
- original-screen perturbation, cell model, treatment, comparator, endpoint,
  and expected phenotype direction;
- validation perturbation modality and reagent identities;
- reported perturbation-confirmation methods and results;
- validation cell model, treatment, dose, exposure, controls, assay, endpoint,
  and replication;
- reported effect estimates, uncertainty, and statistics;
- any rescue or causal-reversal experiment;
- exact source locators; and
- explicit `not reported` entries and source discrepancies.

Judgment-bearing fields—including confirmation adequacy, context match,
outcome relation, technical adequacy, and proposed label—must be absent from
the evidence card, not pre-populated with neutral-looking defaults. They remain
blank only in the separate review form. Curator B must verify the card against
the cited figure, legend, Methods, and supplement rather than accepting the
extraction summary as ground truth.

Evidence cards must contain structured, source-backed facts only. They must not
copy unstructured pipeline notes, free-form extraction logs, model reasoning,
TODOs, blocker commentary, interpretive summaries, or notes from earlier
curation tables into any field. Curator B derives scientific judgments directly
from the cited source material.

Automated text filters are a backstop, not evidence of cognitive neutrality.
Before distribution, a named coordinator who has not copied outcome-aware
language into the cards must manually audit every populated field for reviewer,
model, label, rank, and interpretive-outcome leakage and record the audit
attestation in the packet manifest.

### Source access and redistribution

The coordinator must provide lawful access to the full text and relevant
supplements, either through open links or an authorized institutional/private
workspace. Paywalled PDFs, copyrighted figures, source workbooks, and raw data
must not be copied into Git or redistributed publicly unless the applicable
license explicitly permits it. Review-only extracts remain access-controlled.
The public Gold release should retain citations, exact locators, derived
annotations, provenance, and permitted assets only.

## What Curator B must not receive

Before submitting the frozen review, Curator B must not receive or access:

- Curator A's form, labels, notes, confidence ratings, or source judgments;
- candidate or final Gold labels and reviewer-comparison files;
- EvidenceRank/model scores, predicted probabilities, or model explanations;
- cross-screen recurrence, validation-success estimates, or downstream
  rankings;
- MAGeCK/DrugZ/STARS rank or effect magnitude when it is not required to state
  the claim being tested;
- the outcome-dependent rationale for selecting an event for review; or
- an adjudication record from an earlier pass of the same event.

**The current
`data/manifests/orcs_2.0.18/curation_batches/batch_001/reviews.tsv` must never be
sent to Curator B.** It is a prior outcome-aware, single-curator extraction
containing candidate evidence grades, genes, source judgments, and notes. It is
not a neutral packet. The same prohibition applies to renamed, reformatted, or
partially redacted derivatives of that file and to the current
`review_comparison*.tsv` and `reviews_curator_2*.tsv` files.

Gene identity, treatment, cell context, perturbation action, and the expected
screen direction are not blinded because they are required to judge
reproduction versus opposition. Because the primary article may reveal the
authors' interpretation, the design must be described as
**prediction-blinded and co-reviewer-blinded independent review**, not as fully
outcome-blinded review.

The event card also contains source/study/screen/contrast identifiers that can
be joinable to internal curation tables. An opaque `packet_event_id` alone does
not prevent that join. Curator B must therefore receive the packet through a
separate access-isolated workspace and must not have access to the project
repository, confidential crosswalk, legacy review tables, Curator A's area, or
comparison/adjudication outputs until the submission is frozen. The packet
manifest records the coordinator's access-isolation attestation; it does not
technically prove that no outside access occurred.

## Curator workflow

For every event, Curator B must:

1. confirm that the cited source material is available and complete;
2. determine whether the gene received a relevant targeted test;
3. verify that the perturbation action matches the claim;
4. evaluate cell-model, treatment, comparator, and endpoint correspondence;
5. assess perturbation confirmation, independent reagents, orthogonal
   perturbation, controls, assay adequacy, and biological replication;
6. determine the phenotype direction relative to the predeclared screen
   claim;
7. assess whether a rescue or causal reversal was successful;
8. explicitly report whether one or more relevant validation attempts appear
   to be missing from the packet;
9. assign one label using the decision tree below; and
10. provide an exact source locator and a concise rationale for every decisive
   judgment.

If the evidence card is inaccurate, Curator B must correct it in the form and
cite the source. Missing material is requested through the coordinator. Any
addendum must be versioned and delivered to both curators without revealing
either curator's decisions.

### Packet-completeness check and addenda

The richer review workbook must include the required field
`relevant_validation_attempts_missing_from_packet` with values `yes`, `no`, or
`unclear`. If the answer is `yes` or `unclear`, Curator B must provide a short
description and an exact source locator for each potentially omitted attempt.
This completeness check is distinct from the label assigned to the frozen
card.

Curator B must not silently merge an omitted experiment into the current card.
The coordinator verifies the cited source and then either:

- creates a separate opaque event card when the omitted attempt is a distinct
  `gene × perturbation × cell model × treatment × endpoint` event; or
- issues a versioned addendum when it corrects or completes the existing
  event.

Any changed card requires an incremented `packet_version` and a new
`packet_payload_sha256`. The identical revised card or new card must be sent to
both curators. Reviews tied to different packet versions or payload checksums
must never be compared. Superseded cards and reviews remain archived rather
than overwritten.

## Label decision tree

```text
Was a relevant targeted validation test performed?
├── No
│   └── event_scope=not_validation; testing_status=not_tested; U
├── Cannot be established from the sources
│   └── event_scope=unclear; testing_status=unknown; U
└── Yes
    ├── outcome_relation=not_interpretable because of technical failure → T
    ├── outcome_relation=mixed/not_reported or direction remains ambiguous → A
    ├── outcome_relation=concordant
    │   ├── qualifying_validation + context_match=exact + all core quality
    │   │   criteria satisfied
    │   │   ├── rescue_or_causal_reversal=yes → V3
    │   │   └── otherwise → V2
    │   └── qualifying or external support, but a primary-label criterion is
    │       absent, not_reported, unclear, or context is not exact → V1
    ├── outcome_relation=no_effect
    │   ├── qualifying_validation + context_match=exact + all core quality
    │   │   criteria + null_result_evaluable=yes → F0
    │   └── otherwise → A
    └── outcome_relation=opposite
        ├── qualifying_validation + context_match=exact + all core quality
        │   criteria satisfied → D
        └── otherwise → A

Core quality criteria for V3/V2/F0/D:
perturbation_confirmed=yes; appropriate_control=yes; assay_adequate=yes; and
at least one primary reagent, with independent_reagent_count>=2 OR
orthogonal_perturbation=yes.
```

## Label definitions

| Label | Definition | Primary model role |
|---|---|---|
| `V3` | `V2` requirements plus `rescue_or_causal_reversal=yes`, meaning a **successful** causal reversal in the relevant phenotype assay. | Positive |
| `V2` | `event_scope=qualifying_validation`, `context_match=exact`, `outcome_relation=concordant`, and all core quality criteria satisfied. | Positive |
| `V1` | Concordant qualifying or external-support evidence that does not satisfy every primary-label requirement. | Auxiliary/unlabeled |
| `F0` | Exact qualifying validation with `outcome_relation=no_effect`, all core quality criteria, and `null_result_evaluable=yes`. | Negative |
| `D` | Exact qualifying validation with `outcome_relation=opposite` and all core quality criteria. | Negative |
| `A` | Testing occurred, but the result is mixed, biologically ambiguous, or lacks the requirements needed to interpret a no-effect/opposite result as `F0/D`. | Auxiliary/unlabeled |
| `T` | Testing occurred and `outcome_relation=not_interpretable` because a technical failure prevents a biological conclusion. | Auxiliary/unlabeled |
| `U` | No relevant targeted test was reported, or whether testing occurred is unknown. Preserve `not_tested` and `unknown` separately; `event_scope=not_validation` can only receive `U`. | Auxiliary/unlabeled |

Only `V2/V3` and `F0/D` enter the primary binary success model. `V1`, `A`,
`T`, and `U` must never be silently converted to negative labels.

## Interpretation rules

- Absence from a paper's validation section is `U`, never `F0`.
- `p > 0.05` alone is not evidence of `F0`. The assay must be capable of
  detecting the prespecified phenotype; otherwise use `A`.
- Statistical significance alone is not sufficient for `V2/V3`. Record the
  reported effect, uncertainty, and biological replication.
- Technical wells and repeated measurements are not independent biological
  replicates.
- Repeated use of one sgRNA is one reagent. Multiple clones derived using the
  same sgRNA do not satisfy the two-independent-reagent rule.
- Missing perturbation confirmation with an otherwise supportive phenotype
  generally limits evidence to `V1`. Demonstrated failure to perturb the
  target, when it prevents interpretation, is `T`.
- For CRISPR knockout, qPCR alone does not ordinarily establish a functional
  knockout. Genotype/allelic evidence, protein loss, or a validated functional
  readout is required. Modality-appropriate RNA or protein confirmation may be
  used for RNAi, CRISPRi, CRISPRa, or overexpression.
- A rescue qualifies for `V3` only when restoration of the target or an
  equivalent causal intervention successfully reverses the relevant phenotype.
  Merely performing rescue, overexpressing an unrelated construct, or showing
  a pathway association is insufficient.
- Drug response must be distinguished from generic fitness. A higher cell
  count under drug without a matched vehicle/untreated and targeting control
  does not by itself establish resistance.
- A mechanistic readout such as pathway activation or DNA damage, without the
  relevant screen-phenotype endpoint, is nonqualifying mechanistic evidence;
  it cannot create `V2`, `V3`, `F0`, or `D`.
- Same-drug dose-response validation may use doses or exposure schedules that
  differ from the screen if the comparison remains biologically relevant and
  interpretable. Material context differences must be recorded.
- Evidence only from a different cell model is `context_match=external` and is
  supportive at most (`V1`) for the original screen. It must not create `F0`
  or `D` for the original model.
- Conflicting technically adequate events remain separate. Candidate-level
  aggregation is performed later under the adjudication policy, not by either
  curator.

## Review-form fields

### Contract-required event-review core

The normalized event-review table must match `EventReviewRecord`. Its core
fields are:

- linkage and identity: `event_review_id`, `packet_id`, `packet_version`,
  `packet_event_id`, `packet_payload_sha256`, `review_session_id`,
  `reviewer_id`, `rubric_version`, and `reviewed_at_utc`;
- attestations, all required to be true: `independence_attested`,
  `peer_review_blinded`, `model_score_blinded`, and `prior_label_blinded`;
- label-driving judgments: `event_scope`, `context_match`, `testing_status`,
  `perturbation_confirmed`, `independent_reagent_count`,
  `orthogonal_perturbation`, `appropriate_control`, `assay_adequate`,
  `outcome_relation`, `null_result_evaluable`, and
  `rescue_or_causal_reversal`;
- decision and provenance: `proposed_label_code`, `confidence`,
  `source_locator`, sorted unique pipe-delimited `reason_codes`, and optional
  `notes`.

Use only the contract vocabulary:

- `event_scope`: `qualifying_validation`, `external_support`,
  `not_validation`, or `unclear`;
- `context_match`: `exact`, `close`, `external`, or `unclear`;
- criterion assessments: `yes`, `no`, `unclear`, `not_reported`, or
  `not_applicable`;
- `outcome_relation`: `concordant`, `opposite`, `no_effect`, `mixed`,
  `not_interpretable`, or `not_reported`;
- `testing_status`: `tested`, `not_tested`, or `unknown`; and
- `confidence`: `high`, `moderate`, or `low`.

`rescue_or_causal_reversal=yes` means that reversal was successful; merely
attempting or reporting a rescue experiment does not satisfy this criterion.

### Richer workbook extraction fields

The human-facing workbook may—and for auditability should—capture additional
source facts that are not columns in the normalized `EventReviewRecord`. Keep
these fields separate from the contract-required core and map them explicitly
during normalization. They include:

- `source_material_complete`;
- the required `relevant_validation_attempts_missing_from_packet` check,
  missing-attempt description, and source locator;
- perturbation modality, reagent identities, and confirmation method;
- validation drug, dose/unit, exposure schedule, cell model, controls, assay,
  endpoint, and biological/technical replicate details;
- reported effect size, metric, uncertainty, and p-value;
- a concise human rationale and any card correction;
- additional sources used, prior involvement, conflicts of interest, and
  free-text notes.

Richer extraction fields support audit and adjudication but cannot override an
inconsistent core assessment or substitute for a required core field.

## Independence, signing, and amendments

Curator B must work independently, must not discuss Gold cases with Curator A
before submission freeze, and must not search reviewer-specific repository
files. Prior familiarity with a study is allowed but must be disclosed.

Each batch submission must contain the completed event table and the following
signed declaration:

> I independently reviewed the listed source materials under rubric version __
> and packet __ version __. Before submission, I did not access Curator A
> annotations, prior candidate or final labels, model or EvidenceRank outputs,
> or reviewer-comparison results. I have documented all additional sources,
> prior involvement, and potential conflicts of interest. This submission
> represents my independent assessment.

The declaration records the curator's name, stable curator identifier,
affiliation, optional ORCID, signature, and UTC date/time. The coordinator
verifies the identity through an institutional channel and records that check
outside the distributable dataset. Repository fields such as
`identity_verified` are coordinator attestations, not cryptographic signatures
or independent authentication. The coordinator calculates a checksum
immediately after receipt. The frozen submission is never overwritten.
Corrections are stored as named, versioned amendments with a reason and
timestamp and must be completed before unblinding whenever possible.

## Suggested file layout

```text
gold_curation/
├── protocol/
│   ├── GOLD_STANDARD_CURATION.md
│   ├── label_reference.pdf
│   └── label_ontology.tsv
├── batch_001/
│   ├── MANIFEST.json
│   ├── SHA256SUMS.txt
│   ├── curator_B_packet/
│   │   ├── evidence_cards/
│   │   ├── source_index.tsv
│   │   ├── curator_B_review_form.xlsx
│   │   └── batch_declaration.pdf
│   ├── locked_submissions/
│   │   ├── curator_A/
│   │   └── curator_B/
│   ├── comparison/
│   ├── adjudication/
│   └── release/
└── confidential_crosswalk/
```

Curator B receives only the frozen protocol and `curator_B_packet`. The
confidential crosswalk, Curator A submission, comparison, adjudication, and
release directories remain inaccessible until the independent submission has
been signed and frozen.
