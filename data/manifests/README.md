# Data manifests

This directory stores provenance records, not downloaded scientific datasets.
One row in `data_assets.tsv` represents one versioned external or derived
asset. Unknown values remain empty.

Every asset must record:

- stable source/version/accession and exact URL;
- available and retrieval dates;
- SHA-256 and byte size after retrieval;
- license or terms URL and raw/derived redistribution decisions;
- study, screen, source-family, and raw-data-family identifiers;
- download and transformation entry points;
- curator verification status.

Downloaded matrices, reads, article files, private workbooks, and unpublished
screens remain outside Git. A manifest row does not imply permission to
redistribute the corresponding bytes.
