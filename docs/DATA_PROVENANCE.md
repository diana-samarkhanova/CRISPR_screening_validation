# Data provenance policy

Every external datum must be traceable to:

- source name and version;
- stable accession or URL;
- retrieval date;
- license or terms link;
- transformation script and version;
- checksum for downloaded files;
- study, screen, sample, guide, gene, and validation-event identifiers.

## Normalized experimental levels

Provenance is attached to four distinct levels:

- `study`: publication, preprint, or author-defined research source;
- `screen`: shared cell model, CRISPR library/modality, delivery, and pooled
  experiment;
- `contrast`: one treatment-versus-comparator question;
- `sample`: one sequenced biological or technical observation.

The registry preserves both `source_family_id` and `raw_data_family_id`. A
source family links article/preprint versions, repository mirrors, and curated
database records. A raw-data family links every result derived from the same
experimental counts or author-supplied matrix. Alternative scoring methods do
not create independent training data.

## Repository policy

Commit:

- manifests;
- schemas;
- curation rules;
- download instructions;
- checksums;
- small synthetic examples;
- redistributable derived data with documented terms.

Do not commit by default:

- copied article text or figures;
- third-party matrices lacking redistribution permission;
- unpublished internal screen data;
- private annotation workbooks;
- credentials or signed download URLs.

## Source roles

`training_label` is restricted to documented targeted validation events.
DepMap/Project Score, tumor multi-omics, pathway resources, and literature are
features or contextual evidence, not ground-truth validation labels.

[BioGRID ORCS](https://orcs.thebiogrid.org/) is used for screen discovery,
structured metadata, and author-method gene scores. Record its release,
retrieval date, ORCS screen ID, original publication, scoring method, and
source/raw-family mapping. ORCS ranks and hit calls are screen evidence, not
orthogonal-validation labels. Versioned releases are available from the
[official ORCS download repository](https://downloads.thebiogrid.org/BioGRID-ORCS/Latest-Release/).

Primary methods, supplements, raw-repository metadata, and author files verify
historical conditions. A library-design paper, including Joung et al., may
support library identity and design fields, but its original MOI, coverage,
selection, treatment, duration, or control protocol is never copied into a
later screen when that screen did not report those values.

## Temporal provenance

Every evidence row carries `available_date`. For temporal validation, a feature
is visible only if it existed before the test study's cutoff date. Current
database snapshots must not leak later knowledge into historical prediction.

Cross-screen recurrence and other source-derived features are reconstructed
inside each outer training fold after source/raw-family deduplication. The
held-out family and any evidence first available after the test cutoff are
excluded.
