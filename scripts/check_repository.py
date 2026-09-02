"""Fail when repository candidates contain private data or likely secrets."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_SUFFIXES = (
    ".fastq",
    ".fastq.gz",
    ".fq",
    ".fq.gz",
    ".bam",
    ".cram",
    ".sra",
    ".xlsx",
    ".xls",
    ".docx",
    ".pdf",
    ".h5",
    ".h5ad",
    ".loom",
    ".parquet",
    ".feather",
    ".arrow",
    ".npy",
    ".npz",
    ".mtx",
    ".mtx.gz",
    ".rds",
    ".rdata",
    ".zip",
    ".tar",
    ".tar.gz",
)
FORBIDDEN_NAMES = {
    ".env",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
}
FORBIDDEN_PARTS = {"private", "outputs", "artifacts", "models"}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024
ALLOWED_TABULAR_PREFIXES = {
    "data/manifests/",
    "examples/synthetic/",
}
ALLOWED_TABULAR_PATHS = {
    "research/seed_study_manifest.tsv",
}
RAW_DATA_PREFIX = "data/raw/"
RAW_DATA_PLACEHOLDER = "data/raw/.gitkeep"
CLINICALTRIALS_GOV_SYNTHETIC_PREFIX = "examples/synthetic/clinicaltrials_gov_snapshot/"
CLINICALTRIALS_GOV_SYNTHETIC_FIXED_FILES = {
    "curation_queue.tsv",
    "data_assets.tsv",
    "manifest.json",
    "study_inventory.tsv",
    "version_end.json",
    "version_start.json",
}
CLINICALTRIALS_GOV_SYNTHETIC_PAGE = re.compile(r"^pages/page_\d{6}\.json$")

SECRET_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"(?:github_pat_|ghp_)[A-Za-z0-9_]{20,}"),
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
}


def _is_clinicaltrials_gov_page(value: object) -> bool:
    """Identify a studies API payload without relying on its path."""

    if not isinstance(value, dict) or not isinstance(value.get("studies"), list):
        return False
    has_study_shape = any(
        isinstance(study, dict)
        and isinstance(study.get("protocolSection"), dict)
        and isinstance(study["protocolSection"].get("identificationModule"), dict)
        and "nctId" in study["protocolSection"]["identificationModule"]
        for study in value["studies"]
    )
    return has_study_shape or "totalCount" in value or "nextPageToken" in value


def _is_clinicaltrials_gov_manifest(value: object) -> bool:
    return (
        isinstance(value, dict)
        and value.get("bundle_type") == "clinicaltrials_gov_api_snapshot"
    )


def _clinicaltrials_gov_json_kind(content: bytes) -> str | None:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, ValueError):
        return None
    if _is_clinicaltrials_gov_manifest(value):
        return "manifest"
    if _is_clinicaltrials_gov_page(value):
        return "studies page"
    return None


def _synthetic_ctgov_manifest_is_attested(content: bytes) -> bool:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, ValueError):
        return False
    if not _is_clinicaltrials_gov_manifest(value):
        return False
    source = value.get("source")
    scientific_boundary = value.get("scientific_boundary")
    return (
        isinstance(source, dict)
        and source.get("name") == "ClinicalTrials.gov synthetic fixture"
        and source.get("synthetic_fixture") is True
        and source.get("transport_mode") == "injected"
        and source.get("clock_mode") == "injected"
        and source.get("elapsed_clock_mode") == "injected"
        and source.get("mutable_registry") is False
        and isinstance(scientific_boundary, dict)
        and scientific_boundary.get("synthetic_fixture") is True
    )


def _ctgov_page_is_wholly_synthetic(content: bytes) -> bool:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, ValueError):
        return False
    if not _is_clinicaltrials_gov_page(value):
        return False
    if value.get("syntheticFixture") is not True:
        return False
    for study in value["studies"]:
        if not isinstance(study, dict):
            return False
        protocol = study.get("protocolSection")
        if not isinstance(protocol, dict):
            return False
        identification = protocol.get("identificationModule")
        if not isinstance(identification, dict):
            return False
        titles = (
            identification.get("briefTitle"),
            identification.get("officialTitle"),
        )
        nct_id = identification.get("nctId")
        if not (
            isinstance(nct_id, str)
            and nct_id.startswith("NCT9")
            and len(nct_id) == 11
            and nct_id[3:].isdigit()
            and any(
                isinstance(title, str) and "synthetic" in title.lower()
                for title in titles
            )
        ):
            return False
    return bool(value["studies"])


def _clinicaltrials_gov_tsv_kind(content: bytes) -> str | None:
    try:
        header = set(content.splitlines()[0].decode("utf-8").split("\t"))
    except (IndexError, UnicodeDecodeError):
        return None
    if {
        "snapshot_id",
        "source_study_id",
        "source_version_holder",
        "source_asset_sha256",
    }.issubset(header):
        return "study inventory"
    if {
        "candidate_id",
        "snapshot_id",
        "source_study_id",
        "treatment_mapping_review_status",
        "eligible_for_clinical_context",
    }.issubset(header):
        return "curation queue"
    return None


def repository_candidates() -> list[Path]:
    """Return files Git would consider for a commit."""
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return [ROOT / line for line in result.stdout.splitlines() if line.strip()]

    ignored_parts = {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        ".venv",
    }
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not ignored_parts.intersection(path.parts)
    ]


def inspect_path(path: Path) -> list[str]:
    """Return repository-policy violations for one path."""
    relative = path.relative_to(ROOT)
    relative_lower = relative.as_posix().lower()
    findings: list[str] = []

    if relative_lower.startswith(RAW_DATA_PREFIX) and (
        relative_lower != RAW_DATA_PLACEHOLDER
    ):
        findings.append(
            f"{relative}: raw data must remain outside the repository; only "
            f"{RAW_DATA_PLACEHOLDER} may be committed"
        )
    if relative_lower.startswith(CLINICALTRIALS_GOV_SYNTHETIC_PREFIX):
        fixture_relative = relative_lower.removeprefix(
            CLINICALTRIALS_GOV_SYNTHETIC_PREFIX
        )
        if (
            fixture_relative not in CLINICALTRIALS_GOV_SYNTHETIC_FIXED_FILES
            and CLINICALTRIALS_GOV_SYNTHETIC_PAGE.fullmatch(fixture_relative) is None
        ):
            findings.append(
                f"{relative}: unexpected file in the frozen ClinicalTrials.gov "
                "synthetic snapshot"
            )

    if path.name in FORBIDDEN_NAMES or path.name.startswith(".env."):
        findings.append(f"{relative}: forbidden credential/environment file")
    if FORBIDDEN_PARTS.intersection(relative.parts):
        findings.append(f"{relative}: forbidden private/artifact directory")
    if any(relative_lower.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
        findings.append(f"{relative}: forbidden source or binary data type")
    if relative_lower.endswith((".csv", ".tsv")) and not (
        any(relative_lower.startswith(prefix) for prefix in ALLOWED_TABULAR_PREFIXES)
        or relative_lower in ALLOWED_TABULAR_PATHS
    ):
        findings.append(
            f"{relative}: tabular data is outside an approved manifest or "
            "synthetic-fixture path"
        )
    if path.stat().st_size > MAX_FILE_SIZE_BYTES:
        findings.append(
            f"{relative}: exceeds {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MiB"
        )

    if path.stat().st_size <= MAX_FILE_SIZE_BYTES:
        content = path.read_bytes()
        if path.suffix.lower() == ".json":
            ctgov_kind = _clinicaltrials_gov_json_kind(content)
            if ctgov_kind is not None and not relative_lower.startswith(
                CLINICALTRIALS_GOV_SYNTHETIC_PREFIX
            ):
                findings.append(
                    f"{relative}: ClinicalTrials.gov {ctgov_kind} is only "
                    "permitted in the frozen synthetic fixture"
                )
            elif ctgov_kind == "manifest" and not (
                _synthetic_ctgov_manifest_is_attested(content)
            ):
                findings.append(
                    f"{relative}: ClinicalTrials.gov synthetic manifest must "
                    "carry the source and scientific-boundary attestations"
                )
            elif ctgov_kind == "studies page" and not (
                _ctgov_page_is_wholly_synthetic(content)
            ):
                findings.append(
                    f"{relative}: every study in the ClinicalTrials.gov fixture "
                    "must be explicitly titled synthetic"
                )
        if path.suffix.lower() == ".tsv":
            ctgov_kind = _clinicaltrials_gov_tsv_kind(content)
            if ctgov_kind is not None and not relative_lower.startswith(
                CLINICALTRIALS_GOV_SYNTHETIC_PREFIX
            ):
                findings.append(
                    f"{relative}: ClinicalTrials.gov {ctgov_kind} is only "
                    "permitted in the frozen synthetic fixture"
                )
        if (
            relative_lower.startswith(CLINICALTRIALS_GOV_SYNTHETIC_PREFIX)
            and b"synthetic" not in content.lower()
        ):
            findings.append(
                f"{relative}: frozen ClinicalTrials.gov fixture files must be "
                "explicitly marked synthetic"
            )
        if (
            relative_lower.startswith(CLINICALTRIALS_GOV_SYNTHETIC_PREFIX)
            and path.suffix.lower() == ".tsv"
            and any(
                b"synthetic" not in line.lower()
                for line in content.splitlines()[1:]
                if line
            )
        ):
            findings.append(
                f"{relative}: every frozen ClinicalTrials.gov fixture row must "
                "be explicitly marked synthetic"
            )
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                findings.append(f"{relative}: possible {label}")

    return findings


def main() -> int:
    findings = [
        finding for path in repository_candidates() for finding in inspect_path(path)
    ]
    if findings:
        print("Repository policy check failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("Repository policy check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
