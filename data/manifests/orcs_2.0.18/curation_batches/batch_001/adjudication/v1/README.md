# Human validation adjudication packet

Packet: `orcs-2.0.18-batch-001-adjudication-v1`  
Items: 20

This bundle is unsigned and releases no validation labels. Reviewer evidence
levels are curator extractions, not final decisions. Do not infer `V2`, `V3`,
`F0`, or `D` from reviewer agreement.

For every row in `adjudication_decisions.template.tsv`, a named human must
inspect the cited source and choose exactly one disposition:

- `release_validation_event`: provide one fully populated event row;
- `no_qualifying_event`: the cited material is not a qualifying validation
  event; this is not `F0` and not an untested negative;
- `defer_unresolved`: evidence is insufficient and no label is released.

The decision template intentionally contains no prefilled disposition or label.
The event template is a worksheet only; delete non-release rows before
finalization. Each release decision must bind the canonical event row SHA-256,
and the finalizer separately pins the complete event-table SHA-256. The human
must attest that model outputs were unseen, no automated label was assigned,
and the adjudicator is independent of both reviewers. Because the frozen review
records lack stable person identifiers, independence remains a human
attestation plus a display-name sanity check, not cryptographic identity proof.
