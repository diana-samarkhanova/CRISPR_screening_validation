"""Export immutable source-artifact hashes that are packaged in the wheel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "src" / "crispr_evidencerank" / "_build_metadata.json"
    payload = {
        "format_version": 2,
        "dependency_lock_sha256": _sha256(root / "uv.lock"),
        "clinical_schema_sha256": _sha256(
            root / "schemas" / "clinical_trial_evidence.schema.json"
        ),
        "clinicaltrials_gov_curation_candidate_schema_sha256": _sha256(
            root / "schemas" / "clinicaltrials_gov_curation_candidate.schema.json"
        ),
        "clinicaltrials_gov_snapshot_manifest_schema_sha256": _sha256(
            root / "schemas" / "clinicaltrials_gov_snapshot_manifest.schema.json"
        ),
        "clinicaltrials_gov_study_inventory_schema_sha256": _sha256(
            root / "schemas" / "clinicaltrials_gov_study_inventory.schema.json"
        ),
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
