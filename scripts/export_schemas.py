"""Export versioned JSON Schemas from the Pydantic contracts."""

from __future__ import annotations

import json
from pathlib import Path

from crispr_evidencerank.contracts import CONTRACTS, DOCUMENT_CONTRACTS


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "schemas"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlapping_names = CONTRACTS.keys() & DOCUMENT_CONTRACTS.keys()
    if overlapping_names:
        names = ", ".join(sorted(overlapping_names))
        raise RuntimeError(f"duplicate tabular/document contract names: {names}")
    for name, model in (CONTRACTS | DOCUMENT_CONTRACTS).items():
        path = output_dir / f"{name}.schema.json"
        path.write_text(
            json.dumps(model.model_json_schema(), indent=2),
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()
