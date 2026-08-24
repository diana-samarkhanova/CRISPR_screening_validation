# Auxiliary immune-context method

## Purpose and boundary

The immune-context method is a post-ranking analysis for genes already observed
in a CRISPR screen. It asks whether the same perturbation action has favorable,
unfavorable, or conflicting functional consequences in tumor and immune cells.
It does **not** modify MAGeCK statistics, create `V2/V3/F0/D` labels, or estimate
a probability of orthogonal validation.

The design adapts useful concepts from
[ICRAFT](https://doi.org/10.1016/j.immuni.2025.02.007): standardized
immune-screen comparison, recurrence across ranked lists, explicit tumor and
immune compartments, and dual-action target hypotheses. It does not copy the
mutable ICRAFT portal database or treat its aggregate RRA as ground truth. The
portal is available at <https://icraft.pku-genomics.org/> and the public
crawler/parser repository at
<https://github.com/zenglab-pku/ICRAFT_A20_paper>.

## Project comparison

| Dimension | CRISPR-EvidenceRank | ICRAFT |
|---|---|---|
| Primary estimand | Relative priority for orthogonal follow-up after a specific drug-response screen | Recurrence of immune-related CRISPR phenotypes and dual-cell-compartment hypotheses |
| Input detail | Count/guide QC, exact screen and treatment-control contrast | Large harmonized immune-screen collection |
| Ground truth | Explicit `V3/V2/V1/F0/D/A/T/U` validation-event ontology | No systematic success/failure validation benchmark |
| Main strength | Provenance, negative outcomes, leakage control, user-screen workflow | Breadth, RRA, tumor/immune comparison, in vivo and multi-omic context |
| Current limitation | No released real-data success model; zero benchmark-ready screens | Recurrence is not validation probability; heterogeneous phenotypes and correlated comparisons |

## What is adopted

- separate tumor-cell, immune-cell, and in-vivo evidence lanes;
- exact modality and phenotype strata;
- recurrence across compatible full ranked lists;
- dual-action and immune-liability categories;
- context roles for scRNA-seq, DepMap/CCLE, TCGA, and ICB associations.

The project adds stricter safeguards:

- CRISPR-KO, CRISPRi, and CRISPRa are never pooled by default;
- the native effect sign is immutable; the ICRAFT CRISPRa display inversion is
  accepted only as a registered, numeric LFC sign-pair transformation and is
  never propagated into biological interpretation;
- related comparisons are collapsed through both `raw_data_family_id` and
  `source_family_id`; one external study cannot claim multiple source families;
- evidence is excluded after the declared temporal cutoff, including later
  mappings or transformations;
- ambiguous mouse-to-human mappings remain audit-only;
- exact direction mappings use registered native-sign rules, while conditional
  mappings abstain because the engine does not execute free-text conditions;
- directional and dual-action support requires a source FDR at or below the
  declared threshold (default `0.05`); a sign alone is insufficient;
- de novo RRA requires a verified complete rank roster, a recomputed canonical
  checksum, factual cross-list compatibility, one list per provenance
  component, candidate coverage in every list, and at least two independent
  components;
- report columns are prefixed `report_only_` and are blocked from the current
  validation-success model.

## Direction model

Four concepts remain separate:

1. perturbation modality, such as `CRISPR_KO` or `CRISPRa`;
2. native observation, such as guide enrichment or depletion;
3. assay consequence, such as tumor immune escape or immune-effector gain;
4. endpoint polarity: whether enrichment or depletion is favorable for
   antitumor activity.

A numeric sign alone is never sufficient. For example:

| Contrast | Native KO observation | Interpretation |
|---|---|---|
| Olaparib versus vehicle | Enrichment | Loss-induced olaparib resistance |
| Olaparib versus vehicle | Depletion | Candidate loss-induced sensitization, provided baseline fitness is separated |
| Perturbed tumor cells with T cells versus tumor-only | Depletion | Candidate increase in susceptibility to immune killing |
| Perturbed immune cells, high-activity versus low-activity gate | Enrichment | Candidate immune-effector gain if marker polarity is curated |
| In-vivo abundance | Either direction | Unresolved without perturbed compartment and contrast semantics |

An olaparib-resistance KO hit with favorable immune evidence is therefore a
`PARPi–immune trade-off`, not automatically an inhibitor target.

An `exact` mapping must declare `native_enrichment_is_favorable_v1` or
`native_depletion_is_favorable_v1`, matching its endpoint polarity. A
`conditional` or unresolved mapping remains `unknown`. Directional fractions
and dual-action calls also require `source_fdr <= --max-source-fdr`; evidence
without source FDR remains visible as context but cannot create a favorable
call. A numeric raw effect must also declare controlled sign semantics:
`positive_is_enrichment`, `positive_is_depletion`, or
`unsigned_or_not_applicable`. Unsigned effects cannot create directional
support; signed effects must agree with `native_effect_direction`.

## Dual-action classification

Classification is performed only for an explicitly selected, versioned
`dual_action_group_id` + `dual_action_group_version` adjudication key and one
perturbation modality. The version identifies the caller-declared grouping
rule; the engine checks compatible categories, consequences, compartments, and
context strata but cannot prove that a human review occurred. The key is not
itself biological evidence.

| Tumor-cell evidence | Immune-cell evidence | Output |
|---|---|---|
| Favorable | Favorable | `dual_benefit_candidate` |
| Favorable | Unfavorable | `immune_liability` |
| Unfavorable | Favorable | `tumor_liability` |
| Conflicting or unresolved | Any | `context_dependent` |
| Either compartment absent | — | `insufficient_evidence` |

`dual_benefit_candidate` is hypothesis-generating, not a validated target. A
`recurrent` tier requires at least two independent families in each
compartment; one per compartment is `preliminary`. Tumor support is restricted
to curated tumor-immune escape/sensitization consequences and immune support
to effector- or fitness-gain/loss consequences. Drug-response, marker-only,
ambiguous, and generic selection rows cannot silently create a dual-action
call. Mixed phenotype strata or unresolved evidence within a family yield
abstention or `context_dependent`. Every dual-benefit or liability comparison
requires at least two independent source/raw provenance components.

## RRA eligibility

The implementation uses a dependency-free order-statistic RRA baseline. It is
computed only when every contributing `rank_list_id` passes all checks:

- declared as `full_ranked_list`;
- observed row and unique-gene counts equal `gene_universe_size`;
- ranks are exactly `1..N` with no missing or duplicated position;
- the declared checksum equals a recomputed canonical roster checksum over
  `gene_symbol<TAB>source_rank<LF>` rows ordered by rank;
- modality, compartment, setting, phenotype category/consequence, endpoint
  polarity, source-score semantics, recurrence stratum, and analysis tail
  match across lists;
- every candidate is unambiguously present in every selected list;
- no provenance component supplies more than one list;
- any failed declared-full list blocks the selected stratum rather than being
  silently dropped;
- at least two independent source/raw provenance components remain after
  cutoff and self-family exclusion.

Otherwise the output contains `rra_eligible=false`, a null p-value, and an
explicit reason. `p=1` is not used for abstention. RRA measures recurrence of
rank positions, not effect size, causality, validation, or therapeutic benefit.

## Command

The engine accepts a canonical comparison-by-gene table validated against
`schemas/immune_screen_evidence.schema.json`:

```bash
crispr-evidencerank summarize-immuno-context \
  --evidence immune_screen_evidence.tsv \
  --candidates ranked_candidates.tsv \
  --cutoff-date 2026-08-24 \
  --target-modality CRISPR_KO \
  --exclude-raw-data-family TARGET_SCREEN_RAW_FAMILY \
  --recurrence-stratum-id ko_tumor_immune_killing_negative_tail \
  --dual-action-group-id ko_antitumor_function \
  --dual-action-group-version reviewed-2026-08-24 \
  --max-source-fdr 0.05 \
  --output-dir results/immune_context
```

For a new private target screen that is provably absent from the frozen
compendium, replace the family exclusion with the explicit user attestation
`--target-not-in-compendium`. RRA abstains when neither a matched exclusion nor
that attestation is supplied. Misspelled or nonexistent exclusion IDs are
rejected.

The command validates every evidence row before filtering candidates and
writes:

- `immune_context.tsv`;
- `immune_context_exclusions.tsv`;
- `immune_context_used_evidence.tsv`;
- `rank_list_audit.tsv`;
- `summary.json`.

The bundle is published with one atomic directory rename and refuses to
overwrite an existing directory. `summary.json` records package version, input
paths and SHA-256 values, SHA-256 values for the four TSV outputs, cutoff,
threshold, exact excluded IDs, and the target-absence attestation. The
used-evidence table exposes the rows and provenance components behind the
summary.

Every supplied candidate row is retained. When the candidate input is the
output of `rank-screen`, the screen, contrast, phenotype direction, analysis
tail, native MAGeCK statistics, rank, percentile, and ranking type are carried
into `immune_context.tsv`. The immune method adds a separate axis; it does not
filter or reorder those primary rows. Because `rank-screen` retains
non-significant tails, the report never turns a joined row into a compound
"primary-screen hit plus immune benefit" claim; the primary and immune axes
remain explicitly separate until a downstream significance policy is declared.

Absence of a gene from the evidence collection is represented as unavailable
context, never as a negative screen or failed validation.

## Current data status

The contract, summarizer, RRA eligibility audit, CLI, and synthetic tests are
implemented. No ICRAFT database dump is bundled. A real import remains blocked
until a frozen export has row-level original-source provenance, checksum,
version, source/raw-family mapping, and documented reuse rights. scRNA, TCGA,
ICB, and DepMap associations will use the generic external-evidence lane and
cannot count as functional immune-screen support.

## Limitations

Immune CRISPR studies differ in species, cell model, library, modality,
treatment, selection duration, and phenotype. Standardized MAGeCK processing
cannot eliminate this biological heterogeneity. Cross-model evidence may not
transfer to MDA-MB-468 or olaparib treatment. CRISPR knockout, CRISPRi,
CRISPRa, and pharmacological inhibition are not interchangeable. Marker and
in-vivo abundance endpoints often require manual semantic review. Clinical and
expression associations are non-causal and may reflect batch effects, dropout,
tumor purity, or immune-cell composition. Enhancement of immune-cell activity
does not by itself establish safety or a therapeutic window.
The ICRAFT source search described in the paper covered 2014–2022 and does not
guarantee current completeness; publication and data-availability bias remain.
The raw gene-wise order-statistic p-values are a project baseline, not ICRAFT's
exact implementation or thresholds, and are not adjusted for multiple testing.
The primary resistance/sensitization rows are joined to the report but never
reinterpreted as immune effects. Tumor-cell and immune-cell in-vivo counts are
reported separately; mixed endpoint strata suppress directional fractions.
