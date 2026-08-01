# Data manifests

This directory stores provenance records, not downloaded scientific datasets.
One row in `data_assets.tsv` represents one versioned external or derived
asset. Unknown values remain empty.

Every asset must record:

- stable source/version/accession and exact URL;
- available date when reported, plus the metadata or byte-retrieval date;
- license or terms URL and raw/derived redistribution decisions;
- study, screen, source-family, and raw-data-family identifiers;
- download and transformation entry points;
- curator verification status.

Every retrieved byte asset must additionally record its UTC retrieval
timestamp, SHA-256, byte size, and checksum provenance. An accession-only
pointer whose bytes were not retrieved must instead use an explicit
non-retrieved curator status and `not_retrieved_no_local_checksum`; its SHA-256
and byte size remain empty rather than being invented.

Downloaded matrices, reads, article files, private workbooks, and unpublished
screens remain outside Git. A manifest row does not imply permission to
redistribute the corresponding bytes.
