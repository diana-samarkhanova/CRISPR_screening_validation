## Purpose

Describe the scientific or software question addressed by this change.

## Validation

- [ ] `python scripts/check_repository.py`
- [ ] `ruff check .`
- [ ] `pytest`
- [ ] `python scripts/smoke_test.py`
- [ ] JSON Schemas remain synchronized with the runtime contracts.

## Data and leakage review

- [ ] No unpublished screen data, private annotation workbook, copied
      supplement, credential, or signed URL is included.
- [ ] Every new external datum has a version, source locator, retrieval date,
      license/terms record, and transformation provenance.
- [ ] Records derived from shared experimental material retain the same
      `source_family_id` and `raw_data_family_id`.
- [ ] No validation outcome or post-publication evidence leaks into features
      for the corresponding held-out study.
