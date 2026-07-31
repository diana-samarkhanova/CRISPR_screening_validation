"""Fail when repository candidates contain private data or likely secrets."""

from __future__ import annotations

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

SECRET_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"(?:github_pat_|ghp_)[A-Za-z0-9_]{20,}"),
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
}


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
