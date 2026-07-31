# Contributing

CRISPR-EvidenceRank is currently developed in a private repository. Changes
should preserve scientific provenance, prevent information leakage, and keep
unpublished or non-redistributable data outside Git history.

## Development workflow

1. Create a short feature branch from `main`.
2. Make one scientifically coherent change.
3. Add or update tests and schemas when a contract changes.
4. Run:

   ```bash
   python scripts/check_repository.py
   ruff check .
   pytest
   python scripts/smoke_test.py
   ```

5. Use the pull-request checklist even for solo review.
6. Merge only after CI passes.

## Scientific curation

Each curated fact must include the source version, stable accession or URL,
retrieval date, exact source locator, and curator status. Unknown and
unreported values remain missing; they must not be inferred from a protocol,
related article, or database default.

Validation labels require evidence that the perturbation was actually tested.
Absence from a validation section is not a failed validation. ORCS author hit
calls, MAGeCK significance, recurrence, pathway membership, and database
evidence are features or baselines, not validation labels.

## Data boundaries

Do not commit:

- unpublished screen identities, biological context, counts, or results;
- private annotation workbooks;
- copied article text, figures, PDFs, or supplementary matrices;
- FASTQ/BAM/CRAM files or downloaded repository archives;
- third-party data without verified redistribution terms;
- credentials, tokens, signed URLs, or local environment files.

Commit accession manifests, checksums, download/transform scripts, schemas,
small synthetic fixtures, and redistributable derived records with explicit
terms. See `docs/DATA_PROVENANCE.md` and `docs/REPOSITORY_POLICY.md`.
