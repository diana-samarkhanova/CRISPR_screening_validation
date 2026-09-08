"""Checksum-bound frozen-model bundles for prospective evaluation."""

from __future__ import annotations

import hashlib
import io
import json
import math
import platform
import re
import shutil
import subprocess
import tempfile
import tomllib
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import joblib
import pandas as pd
import yaml
from pandas.testing import assert_frame_equal

from . import __version__
from .clinicaltrials_gov import _publish_directory_noreplace
from .modeling import (
    ReproducibilityModel,
    _study_family_groups,
    validate_success_feature_columns,
)

MODEL_FILENAME = "reproducibility_model.joblib"
FEATURES_FILENAME = "feature_columns.json"
FEATURE_REGISTRY_FILENAME = "features.yaml"
ENVIRONMENT_LOCK_FILENAME = "uv.lock"
MANIFEST_FILENAME = "frozen_model_manifest.json"
RUNTIME_DEPENDENCIES = ("joblib", "numpy", "pandas", "scikit-learn")
TRAINING_IDENTITY_ROLES = ("study_id", "source_family_id", "raw_data_family_id")
TRAINING_IDENTITY_HASH_ALGORITHM = "sha256_utf8_role_nul_stripped_value_v1"
MODEL_BUNDLE_FILENAMES = (
    MODEL_FILENAME,
    FEATURES_FILENAME,
    FEATURE_REGISTRY_FILENAME,
    ENVIRONMENT_LOCK_FILENAME,
    MANIFEST_FILENAME,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains a duplicate key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON must not contain non-finite constants: {value}")


def _utc_timestamp(value: datetime) -> datetime:
    resolved = value.astimezone(UTC) if value.tzinfo is not None else value
    if value.tzinfo is None or resolved.utcoffset() is None:
        raise ValueError("created_at_utc must include a timezone")
    if resolved.utcoffset().total_seconds() != 0:
        raise ValueError("created_at_utc must resolve to UTC")
    return resolved


def _validate_revision(value: str) -> str:
    normalized = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", normalized):
        raise ValueError("code_revision must be a full 40-character Git SHA")
    return normalized


def _validate_feature_registry_bytes(
    content: bytes,
    feature_columns: list[str],
) -> None:
    try:
        registry = yaml.safe_load(content.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("feature registry must be valid UTF-8 YAML") from exc
    if not isinstance(registry, dict):
        raise ValueError("feature registry must contain a YAML mapping")
    feature_groups = registry.get("feature_groups")
    if not isinstance(feature_groups, dict) or not feature_groups:
        raise ValueError("feature registry must define nonempty feature_groups")
    declared: set[str] = set()
    disallowed: set[str] = set()
    for group_name, group in feature_groups.items():
        if not isinstance(group_name, str) or not isinstance(group, dict):
            raise ValueError("feature registry groups must be named mappings")
        values = group.get("features", [])
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise ValueError(
                f"feature registry group {group_name!r} has an invalid features list"
            )
        declared.update(values)
        if group.get("allowed_for_primary_model") is not True:
            disallowed.update(values)
    missing = sorted(set(feature_columns) - declared)
    if missing:
        raise ValueError(
            f"feature registry does not declare frozen feature columns: {missing}"
        )
    forbidden = sorted(set(feature_columns) & disallowed)
    if forbidden:
        raise ValueError(
            "feature registry does not allow frozen features in the primary model: "
            f"{forbidden}"
        )
    validate_success_feature_columns(feature_columns)


def _validate_environment_lock_bytes(content: bytes) -> dict[str, set[str]]:
    try:
        lock = tomllib.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError("environment lock must be valid UTF-8 TOML") from exc
    if not isinstance(lock.get("version"), int) or isinstance(
        lock.get("version"), bool
    ):
        raise ValueError("environment lock must declare an integer lock version")
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        raise ValueError("environment lock must contain at least one package")
    versions_by_name: dict[str, set[str]] = {}
    for package in packages:
        if (
            not isinstance(package, dict)
            or not isinstance(package.get("name"), str)
            or not package["name"]
        ):
            raise ValueError("environment lock contains an invalid package entry")
        version = package.get("version")
        if isinstance(version, str) and version:
            versions_by_name.setdefault(package["name"], set()).add(version)
    return versions_by_name


def _validate_runtime_dependencies_against_lock(
    runtime_dependencies: object,
    lock_versions: dict[str, set[str]],
) -> None:
    if not isinstance(runtime_dependencies, dict) or not runtime_dependencies:
        raise ValueError("runtime_dependencies must be a nonempty mapping")
    if set(runtime_dependencies) != set(RUNTIME_DEPENDENCIES):
        raise ValueError(
            "runtime_dependencies must declare exactly joblib, numpy, pandas, "
            "and scikit-learn"
        )
    mismatches = sorted(
        name
        for name, version in runtime_dependencies.items()
        if not isinstance(name, str)
        or not isinstance(version, str)
        or not version
        or version not in lock_versions.get(name, set())
    )
    if mismatches:
        raise ValueError(
            "runtime dependency versions are absent from the environment lock: "
            f"{mismatches}"
        )


def _validate_loading_environment(manifest: dict[str, object]) -> None:
    """Fail before unpickling if the recorded inference environment differs."""

    if manifest.get("package_version") != __version__:
        raise ValueError(
            "frozen model package version differs from the loading runtime"
        )
    recorded_python = str(manifest.get("python_version", ""))
    if recorded_python.split(".")[:2] != platform.python_version().split(".")[:2]:
        raise ValueError(
            "frozen model Python major/minor version differs from the loading runtime"
        )
    recorded = manifest["runtime_dependencies"]
    mismatches = [
        name
        for name in RUNTIME_DEPENDENCIES
        if importlib_metadata.version(name) != recorded[name]
    ]
    if mismatches:
        raise ValueError(
            "frozen model dependency versions differ from the loading runtime: "
            f"{mismatches}"
        )


def _run_git(checkout: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ValueError("Git is required for source-tree attestation") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "Git command failed"
        raise ValueError(f"source-tree attestation failed: {detail}")
    return result.stdout


def _attest_clean_checkout(
    checkout_path: str | Path,
    *,
    expected_revision: str,
) -> dict[str, object]:
    checkout = Path(checkout_path)
    if not checkout.is_dir():
        raise ValueError("current_checkout_path must be an existing directory")
    root_text = _run_git(checkout, "rev-parse", "--show-toplevel").strip()
    if not root_text:
        raise ValueError("source-tree attestation could not resolve a Git root")
    root = Path(root_text).resolve()
    head = _run_git(root, "rev-parse", "--verify", "HEAD").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("source-tree HEAD is not a full 40-character Git SHA")
    if head != expected_revision:
        raise ValueError("code_revision does not match the attested source-tree HEAD")
    status = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if status.strip():
        raise ValueError("attested source-tree checkout is dirty")
    return {
        "mode": "git_checkout",
        "status": "verified_clean",
        "head_revision": head,
        "tracked_and_untracked_changes_absent": True,
        "path_not_recorded": True,
    }


def _read_training_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"unsupported training table extension: {path.suffix}")


def _independent_training_groups(
    frame: pd.DataFrame,
    *,
    study_column: str,
    source_family_column: str,
    raw_family_column: str,
) -> pd.Series:
    family_columns = tuple(
        column
        for column in (source_family_column, raw_family_column)
        if column in frame and frame[column].notna().any()
    )
    if family_columns:
        return _study_family_groups(
            frame,
            study_column=study_column,
            family_columns=family_columns,
        )
    if study_column not in frame:
        raise ValueError(f"training frame is missing group column: {study_column}")
    return frame[study_column].astype(str)


def training_identity_sha256(role: str, value: object) -> str:
    """Hash a nonempty study/family identifier with a stable semantic namespace."""

    if role not in TRAINING_IDENTITY_ROLES:
        raise ValueError(f"unknown training identity role: {role}")
    if pd.isna(value) or not str(value).strip():
        raise ValueError("training identity value must be nonmissing and nonempty")
    return _sha256_bytes(f"{role}\0{str(value).strip()}".encode())


def _training_identity_inventory(
    frame: pd.DataFrame,
    columns: tuple[str, str, str],
) -> dict[str, object]:
    identities: dict[str, object] = {}
    for role, column in zip(TRAINING_IDENTITY_ROLES, columns, strict=True):
        values = (
            frame[column].astype("string").str.strip()
            if column in frame
            else pd.Series(pd.NA, index=frame.index, dtype="string")
        )
        present = values.notna() & values.ne("").fillna(False)
        missing = int((~present).sum())
        identities[role] = {
            "source_column": column,
            "status": (
                "not_attested"
                if missing == len(frame)
                else "partial"
                if missing
                else "complete"
            ),
            "missing_row_count": missing,
            "identity_sha256": sorted(
                {training_identity_sha256(role, value) for value in values[present]}
            ),
        }
    return {
        "identity_hash_algorithm": TRAINING_IDENTITY_HASH_ALGORITHM,
        "identities": identities,
    }


def _validate_training_identity_inventory(value: object, row_count: int) -> None:
    if (
        not isinstance(value, dict)
        or value.get("identity_hash_algorithm") != TRAINING_IDENTITY_HASH_ALGORITHM
        or not isinstance(value.get("identities"), dict)
        or set(value["identities"]) != set(TRAINING_IDENTITY_ROLES)
    ):
        raise ValueError("frozen model has invalid training_provenance structure")
    for role, inventory in value["identities"].items():
        if not isinstance(inventory, dict):
            raise ValueError(f"invalid training_provenance inventory for {role}")
        column = inventory.get("source_column")
        missing = inventory.get("missing_row_count")
        hashes = inventory.get("identity_sha256")
        if (
            not isinstance(column, str)
            or not column.strip()
            or type(missing) is not int
            or not 0 <= missing <= row_count
            or not isinstance(hashes, list)
            or not all(
                isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
                for digest in hashes
            )
            or hashes != sorted(set(hashes))
            or len(hashes) > row_count - missing
            or bool(hashes) != (missing < row_count)
        ):
            raise ValueError(f"invalid training_provenance inventory for {role}")
        expected_status = (
            "not_attested"
            if missing == row_count
            else "partial"
            if missing
            else "complete"
        )
        if inventory.get("status") != expected_status:
            raise ValueError(f"inconsistent training_provenance status for {role}")
        if role == "study_id" and expected_status != "complete":
            raise ValueError("training_provenance requires complete study identities")


def freeze_model_bundle(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    feature_columns: list[str],
    training_table_sha256: str,
    training_table_name: str,
    training_table_path: str | Path,
    feature_registry_path: str | Path,
    environment_lock_path: str | Path,
    code_revision: str,
    created_at_utc: datetime,
    current_checkout_path: str | Path | None = None,
    study_column: str = "study_id",
    source_family_column: str = "source_family_id",
    raw_family_column: str = "raw_data_family_id",
    label_column: str = "label_code",
    tested_column: str = "testing_status",
    model_kind: str = "logistic",
    logistic_c: float = 1.0,
    random_state: int = 17,
    propensity_clip: tuple[float, float] = (0.1, 0.95),
) -> dict[str, object]:
    """Fit once and atomically publish a byte-pinned local model bundle."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"frozen model output directory exists: {output_dir}")
    if not feature_columns or not all(
        isinstance(column, str) and column for column in feature_columns
    ):
        raise ValueError("feature_columns must contain at least one nonempty string")
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("feature_columns must not contain duplicates")
    if not re.fullmatch(r"[0-9a-f]{64}", training_table_sha256):
        raise ValueError("training_table_sha256 must be a lowercase SHA-256")
    training_table_path = Path(training_table_path)
    if not training_table_path.is_file():
        raise FileNotFoundError(
            f"training table file does not exist: {training_table_path}"
        )
    if _sha256(training_table_path) != training_table_sha256:
        raise ValueError("training table does not match training_table_sha256")
    try:
        assert_frame_equal(
            frame.reset_index(drop=True),
            _read_training_table(training_table_path).reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as exc:
        raise ValueError(
            "training frame does not exactly match the checksum-bound table"
        ) from exc
    revision = _validate_revision(code_revision)
    created = _utc_timestamp(created_at_utc)
    feature_registry_path = Path(feature_registry_path)
    environment_lock_path = Path(environment_lock_path)
    for path, role in (
        (feature_registry_path, "feature registry"),
        (environment_lock_path, "environment lock"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{role} file does not exist: {path}")
    feature_registry_bytes = feature_registry_path.read_bytes()
    environment_lock_bytes = environment_lock_path.read_bytes()
    _validate_feature_registry_bytes(feature_registry_bytes, feature_columns)
    lock_versions = _validate_environment_lock_bytes(environment_lock_bytes)
    runtime_dependencies = {
        name: importlib_metadata.version(name) for name in RUNTIME_DEPENDENCIES
    }
    _validate_runtime_dependencies_against_lock(runtime_dependencies, lock_versions)
    feature_registry_sha256 = _sha256_bytes(feature_registry_bytes)
    environment_lock_sha256 = _sha256_bytes(environment_lock_bytes)
    if current_checkout_path is None:
        source_tree_attestation: dict[str, object] = {
            "mode": "not_requested",
            "status": "not_verified",
            "reason": (
                "no source checkout was supplied; this preserves installed-wheel use"
            ),
        }
    else:
        source_tree_attestation = _attest_clean_checkout(
            current_checkout_path,
            expected_revision=revision,
        )

    groups = _independent_training_groups(
        frame,
        study_column=study_column,
        source_family_column=source_family_column,
        raw_family_column=raw_family_column,
    )
    training_provenance = _training_identity_inventory(
        frame,
        (study_column, source_family_column, raw_family_column),
    )
    _validate_training_identity_inventory(training_provenance, len(frame))
    model = ReproducibilityModel(
        list(feature_columns),
        model_kind=model_kind,
        logistic_c=logistic_c,
        random_state=random_state,
        propensity_clip=propensity_clip,
    ).fit(
        frame,
        label_column=label_column,
        tested_column=tested_column,
        selection_weighting=False,
        selection_groups=groups,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        feature_registry_copy = staging / FEATURE_REGISTRY_FILENAME
        feature_registry_copy.write_bytes(feature_registry_bytes)
        environment_lock_copy = staging / ENVIRONMENT_LOCK_FILENAME
        environment_lock_copy.write_bytes(environment_lock_bytes)
        features_path = staging / FEATURES_FILENAME
        features_path.write_bytes(_canonical_json_bytes(list(feature_columns)))
        model_path = staging / MODEL_FILENAME
        joblib.dump(model, model_path, compress=3)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "bundle_type": "frozen_reproducibility_model",
            "status": "frozen_not_yet_prospectively_evaluated",
            "created_at_utc": created.isoformat(),
            "package_version": __version__,
            "code_revision": revision,
            "source_tree_attestation": source_tree_attestation,
            "python_version": platform.python_version(),
            "runtime_dependencies": runtime_dependencies,
            "training_table": {
                "filename": Path(training_table_name).name,
                "sha256": training_table_sha256,
                "row_count": len(frame),
                "label_counts": {
                    str(key): int(value)
                    for key, value in frame[label_column]
                    .fillna("<missing>")
                    .astype(str)
                    .value_counts(dropna=False)
                    .sort_index()
                    .items()
                },
            },
            "training_provenance": training_provenance,
            "feature_registry": {
                "filename": FEATURE_REGISTRY_FILENAME,
                "source_filename": feature_registry_path.name,
                "sha256": _sha256(feature_registry_copy),
                "byte_size": feature_registry_copy.stat().st_size,
            },
            "environment_lock": {
                "filename": ENVIRONMENT_LOCK_FILENAME,
                "source_filename": environment_lock_path.name,
                "sha256": _sha256(environment_lock_copy),
                "byte_size": environment_lock_copy.stat().st_size,
            },
            "feature_columns": {
                "filename": FEATURES_FILENAME,
                "sha256": _sha256(features_path),
                "byte_size": features_path.stat().st_size,
                "count": len(feature_columns),
            },
            "model": {
                "filename": MODEL_FILENAME,
                "sha256": _sha256(model_path),
                "byte_size": model_path.stat().st_size,
                "model_kind": model_kind,
                "logistic_c": logistic_c,
                "random_state": random_state,
                "propensity_clip": list(propensity_clip),
                "selection_weighting": False,
                "independent_group_count": int(groups.nunique()),
            },
            "security": {
                "serialization": "joblib_pickle_protocol",
                "warning": (
                    "hash verification establishes integrity, not trust; never "
                    "load a model bundle from an untrusted source"
                ),
            },
            "prospective_boundary": {
                "predictions_must_be_frozen_before_experiment": True,
                "external_timestamp_or_deposition_not_created_by_this_bundle": True,
                "wet_lab_outcomes_absent": True,
            },
        }
        (staging / MANIFEST_FILENAME).write_bytes(_canonical_json_bytes(manifest))
        if _sha256(training_table_path) != training_table_sha256:
            raise ValueError("training table changed while the model was being frozen")
        if _sha256(feature_registry_path) != feature_registry_sha256:
            raise ValueError(
                "feature registry changed while the model was being frozen"
            )
        if _sha256(environment_lock_path) != environment_lock_sha256:
            raise ValueError(
                "environment lock changed while the model was being frozen"
            )
        if current_checkout_path is not None:
            final_attestation = _attest_clean_checkout(
                current_checkout_path,
                expected_revision=revision,
            )
            if final_attestation != source_tree_attestation:
                raise ValueError(
                    "source-tree attestation changed during model freezing"
                )
        _publish_directory_noreplace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def verify_model_bundle(
    bundle_dir: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Verify bundle structure and all file hashes without deserializing it."""

    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise ValueError(f"frozen model bundle directory does not exist: {bundle_dir}")
    expected_roster = set(MODEL_BUNDLE_FILENAMES)
    observed_roster = {entry.name for entry in bundle_dir.iterdir()}
    missing = [name for name in MODEL_BUNDLE_FILENAMES if name not in observed_roster]
    if missing:
        raise ValueError(f"frozen model bundle is missing files: {missing}")
    unexpected = sorted(observed_roster - expected_roster)
    if unexpected:
        raise ValueError(f"frozen model bundle contains unexpected files: {unexpected}")
    invalid_files = sorted(
        name
        for name in MODEL_BUNDLE_FILENAMES
        if (bundle_dir / name).is_symlink() or not (bundle_dir / name).is_file()
    )
    if invalid_files:
        raise ValueError(
            "frozen model bundle entries must be regular, non-symlink files: "
            f"{invalid_files}"
        )
    manifest_path = bundle_dir / MANIFEST_FILENAME
    manifest_content = manifest_path.read_bytes()
    if expected_manifest_sha256 is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
            raise ValueError("expected_manifest_sha256 must be a lowercase SHA-256")
        if _sha256_bytes(manifest_content) != expected_manifest_sha256:
            raise ValueError("frozen model manifest identity hash mismatch")
    try:
        manifest = json.loads(
            manifest_content.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("frozen model manifest must contain valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("frozen model manifest must be a JSON object")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("unsupported frozen model manifest schema_version")
    if manifest.get("bundle_type") != "frozen_reproducibility_model":
        raise ValueError("unexpected frozen model bundle_type")
    if manifest.get("status") != "frozen_not_yet_prospectively_evaluated":
        raise ValueError("unexpected frozen model status")
    code_revision = manifest.get("code_revision")
    try:
        validated_revision = _validate_revision(code_revision)
    except ValueError as exc:
        raise ValueError("frozen model manifest has an invalid code_revision") from exc
    if code_revision != validated_revision:
        raise ValueError(
            "frozen model manifest code_revision must be canonical lowercase"
        )

    file_specs = (
        ("feature_columns", FEATURES_FILENAME),
        ("feature_registry", FEATURE_REGISTRY_FILENAME),
        ("environment_lock", ENVIRONMENT_LOCK_FILENAME),
        ("model", MODEL_FILENAME),
    )
    file_manifests: dict[str, dict[str, object]] = {}
    verified_contents: dict[str, bytes] = {}
    for role, filename in file_specs:
        file_manifest = manifest.get(role)
        if not isinstance(file_manifest, dict):
            raise ValueError("frozen model manifest is missing file metadata")
        if file_manifest.get("filename") != filename:
            raise ValueError(
                f"frozen model manifest fixed filename mismatch for {role}"
            )
        expected_sha256 = file_manifest.get("sha256")
        if not isinstance(expected_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_sha256
        ):
            raise ValueError(f"frozen model manifest has an invalid hash for {role}")
        byte_size = file_manifest.get("byte_size")
        if (
            not isinstance(byte_size, int)
            or isinstance(byte_size, bool)
            or byte_size < 1
        ):
            raise ValueError(
                f"frozen model manifest has an invalid byte_size for {role}"
            )
        path = bundle_dir / filename
        content = path.read_bytes()
        if len(content) != byte_size:
            raise ValueError(f"frozen model bundle byte_size mismatch: {filename}")
        if _sha256_bytes(content) != expected_sha256:
            raise ValueError(f"frozen model bundle hash mismatch: {filename}")
        file_manifests[role] = file_manifest
        verified_contents[role] = content

    training_manifest = manifest.get("training_table")
    if not isinstance(training_manifest, dict):
        raise ValueError("frozen model manifest is missing training table metadata")
    training_filename = training_manifest.get("filename")
    if (
        not isinstance(training_filename, str)
        or not training_filename
        or Path(training_filename).name != training_filename
    ):
        raise ValueError("frozen model manifest has an invalid training table filename")
    training_sha256 = training_manifest.get("sha256")
    if not isinstance(training_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", training_sha256
    ):
        raise ValueError("frozen model manifest has an invalid training table hash")
    training_row_count = training_manifest.get("row_count")
    if (
        not isinstance(training_row_count, int)
        or isinstance(training_row_count, bool)
        or training_row_count < 1
    ):
        raise ValueError("frozen model manifest has an invalid training row count")
    label_counts = training_manifest.get("label_counts")
    if (
        not isinstance(label_counts, dict)
        or not label_counts
        or not all(
            isinstance(label, str) and type(count) is int and count > 0
            for label, count in label_counts.items()
        )
        or sum(label_counts.values()) != training_row_count
    ):
        raise ValueError("frozen model manifest has inconsistent training label counts")
    if "training_provenance" in manifest:
        _validate_training_identity_inventory(
            manifest["training_provenance"], training_row_count
        )

    model_manifest = file_manifests["model"]
    model_kind = model_manifest.get("model_kind")
    logistic_c = model_manifest.get("logistic_c", 1.0)
    clipping = model_manifest.get("propensity_clip")
    if model_kind not in {"logistic", "hist_gradient_boosting"}:
        raise ValueError("frozen model manifest has an invalid model_kind")
    if (
        type(logistic_c) not in {int, float}
        or not math.isfinite(logistic_c)
        or logistic_c <= 0
        or (model_kind != "logistic" and logistic_c != 1.0)
    ):
        raise ValueError("frozen model manifest has an invalid logistic_c")
    if (
        type(model_manifest.get("random_state")) is not int
        or not 0 <= model_manifest["random_state"] < 2**32
        or model_manifest.get("selection_weighting") is not False
        or type(model_manifest.get("independent_group_count")) is not int
        or not 1 <= model_manifest["independent_group_count"] <= training_row_count
        or not isinstance(clipping, list)
        or len(clipping) != 2
        or not all(type(value) in {int, float} for value in clipping)
        or not 0 < clipping[0] < clipping[1] <= 1
    ):
        raise ValueError("frozen model manifest has invalid model parameters")

    attestation = manifest.get("source_tree_attestation")
    if not isinstance(attestation, dict):
        raise ValueError("frozen model manifest is missing source-tree attestation")
    if attestation.get("mode") == "not_requested":
        if (
            attestation.get("status") != "not_verified"
            or not isinstance(attestation.get("reason"), str)
            or not attestation["reason"]
        ):
            raise ValueError("invalid not-requested source-tree attestation")
    elif attestation.get("mode") == "git_checkout":
        if (
            attestation.get("status") != "verified_clean"
            or attestation.get("head_revision") != validated_revision
            or attestation.get("tracked_and_untracked_changes_absent") is not True
            or attestation.get("path_not_recorded") is not True
        ):
            raise ValueError("invalid verified source-tree attestation")
    else:
        raise ValueError("unknown source-tree attestation mode")

    features_content = verified_contents["feature_columns"]
    try:
        features = json.loads(features_content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("feature_columns.json must contain valid UTF-8 JSON") from exc
    if (
        not isinstance(features, list)
        or not features
        or not all(isinstance(value, str) and value for value in features)
    ):
        raise ValueError("feature_columns.json must contain a nonempty string list")
    if len(features) != len(set(features)):
        raise ValueError("feature_columns.json must not contain duplicates")
    if _canonical_json_bytes(features) != features_content:
        raise ValueError("feature_columns.json must use canonical JSON encoding")
    count = file_manifests["feature_columns"].get("count")
    if type(count) is not int or len(features) != count:
        raise ValueError("frozen model feature count does not match manifest")
    _validate_feature_registry_bytes(
        verified_contents["feature_registry"],
        features,
    )
    lock_versions = _validate_environment_lock_bytes(
        verified_contents["environment_lock"]
    )
    _validate_runtime_dependencies_against_lock(
        manifest.get("runtime_dependencies"),
        lock_versions,
    )
    return manifest


def load_model_bundle(
    bundle_dir: str | Path,
    *,
    trusted_local_bundle: bool = False,
    expected_manifest_sha256: str | None = None,
) -> tuple[ReproducibilityModel, dict[str, object]]:
    """Verify then load a model only after an explicit local-trust decision."""

    manifest = verify_model_bundle(
        bundle_dir,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if not trusted_local_bundle:
        raise ValueError(
            "joblib model loading can execute code; set trusted_local_bundle=True "
            "only for a locally controlled, verified bundle"
        )
    _validate_loading_environment(manifest)
    # Hash the exact immutable bytes passed to joblib; a later path replacement
    # must never turn verification into deserialization of a different pickle.
    model_content = (Path(bundle_dir) / MODEL_FILENAME).read_bytes()
    features_content = (Path(bundle_dir) / FEATURES_FILENAME).read_bytes()
    for role, content in (
        ("model", model_content),
        ("feature_columns", features_content),
    ):
        if _sha256_bytes(content) != manifest[role]["sha256"]:
            raise ValueError(f"frozen model {role} changed before loading")
    features = json.loads(features_content.decode("utf-8"))
    model = joblib.load(io.BytesIO(model_content))
    if not isinstance(model, ReproducibilityModel):
        raise TypeError("frozen model artifact has an unexpected object type")
    if model.feature_columns != features:
        raise ValueError("loaded model feature order differs from the frozen contract")
    model_manifest = manifest["model"]
    for attribute in ("model_kind", "random_state"):
        if getattr(model, attribute, None) != model_manifest[attribute]:
            raise ValueError(
                f"loaded model {attribute} differs from the frozen contract"
            )
    if list(model.propensity_clip) != model_manifest["propensity_clip"]:
        raise ValueError(
            "loaded model propensity_clip differs from the frozen contract"
        )
    if model.logistic_c != model_manifest.get("logistic_c", 1.0):
        raise ValueError("loaded model logistic_c differs from the frozen contract")
    if model.success_model is None:
        raise ValueError("loaded frozen model has no fitted success model")
    for role, pipeline in (
        ("success", model.success_model),
        ("selection", model.selection_model),
    ):
        if pipeline is None:
            continue
        if getattr(pipeline, "n_features_in_", None) != len(features):
            raise ValueError(
                f"loaded {role} pipeline has an incompatible feature count"
            )
        fitted_names = getattr(pipeline, "feature_names_in_", None)
        if fitted_names is not None and list(fitted_names) != features:
            raise ValueError(
                f"loaded {role} pipeline has an incompatible feature order"
            )
        if list(getattr(pipeline, "classes_", [])) != [0, 1]:
            raise ValueError(f"loaded {role} pipeline has incompatible target classes")
        estimator = pipeline.named_steps["model"]
        if role == "success" and model.model_kind == "logistic":
            if getattr(estimator, "C", None) != model.logistic_c:
                raise ValueError("loaded success estimator differs from logistic_c")
    return model, manifest


__all__ = [
    "ENVIRONMENT_LOCK_FILENAME",
    "FEATURES_FILENAME",
    "FEATURE_REGISTRY_FILENAME",
    "MANIFEST_FILENAME",
    "MODEL_BUNDLE_FILENAMES",
    "MODEL_FILENAME",
    "TRAINING_IDENTITY_HASH_ALGORITHM",
    "TRAINING_IDENTITY_ROLES",
    "freeze_model_bundle",
    "load_model_bundle",
    "training_identity_sha256",
    "verify_model_bundle",
]
