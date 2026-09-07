import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from crispr_evidencerank.model_artifact import (
    ENVIRONMENT_LOCK_FILENAME,
    FEATURE_REGISTRY_FILENAME,
    FEATURES_FILENAME,
    MANIFEST_FILENAME,
    MODEL_BUNDLE_FILENAMES,
    MODEL_FILENAME,
    freeze_model_bundle,
    load_model_bundle,
    verify_model_bundle,
)
from crispr_evidencerank.modeling import DEFAULT_FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_fixture(
    output_dir: Path,
    *,
    code_revision: str = "a" * 40,
    current_checkout_path: Path | None = None,
) -> dict[str, object]:
    training_path = ROOT / "examples/synthetic/labeled_gene_features.csv"
    return freeze_model_bundle(
        pd.read_csv(training_path),
        output_dir,
        feature_columns=list(DEFAULT_FEATURE_COLUMNS),
        training_table_sha256=_sha256(training_path),
        training_table_name=training_path.name,
        training_table_path=training_path,
        feature_registry_path=ROOT / "config/features.yaml",
        environment_lock_path=ROOT / "uv.lock",
        code_revision=code_revision,
        created_at_utc=datetime(2026, 9, 7, 12, 0, tzinfo=UTC),
        current_checkout_path=current_checkout_path,
    )


def _rewrite_manifest(output_dir: Path, manifest: dict[str, object]) -> None:
    (output_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _init_clean_checkout(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.org"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test Curator"],
        check=True,
    )
    (path / "tracked.txt").write_text("frozen source\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-q", "-m", "frozen source"],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_frozen_model_bundle_is_hash_bound_and_requires_explicit_trust(tmp_path):
    training_path = ROOT / "examples/synthetic/labeled_gene_features.csv"
    frame = pd.read_csv(training_path)
    output_dir = tmp_path / "frozen-model"
    manifest = freeze_model_bundle(
        frame,
        output_dir,
        feature_columns=list(DEFAULT_FEATURE_COLUMNS),
        training_table_sha256=_sha256(training_path),
        training_table_name=training_path.name,
        training_table_path=training_path,
        feature_registry_path=ROOT / "config/features.yaml",
        environment_lock_path=ROOT / "uv.lock",
        code_revision="a" * 40,
        created_at_utc=datetime(2026, 9, 7, 12, 0, tzinfo=UTC),
    )
    assert manifest["status"] == "frozen_not_yet_prospectively_evaluated"
    assert manifest["training_table"]["row_count"] == len(frame)
    assert manifest["prospective_boundary"]["wet_lab_outcomes_absent"] is True
    assert manifest["source_tree_attestation"]["mode"] == "not_requested"
    assert manifest["feature_registry"]["filename"] == FEATURE_REGISTRY_FILENAME
    assert manifest["environment_lock"]["filename"] == ENVIRONMENT_LOCK_FILENAME
    assert (
        manifest["model"]["byte_size"] == (output_dir / MODEL_FILENAME).stat().st_size
    )
    assert set(path.name for path in output_dir.iterdir()) == set(
        MODEL_BUNDLE_FILENAMES
    )
    assert (output_dir / FEATURE_REGISTRY_FILENAME).read_bytes() == (
        ROOT / "config/features.yaml"
    ).read_bytes()
    assert (output_dir / ENVIRONMENT_LOCK_FILENAME).read_bytes() == (
        ROOT / "uv.lock"
    ).read_bytes()
    assert verify_model_bundle(output_dir) == manifest

    with pytest.raises(ValueError, match="can execute code"):
        load_model_bundle(output_dir)
    model, loaded_manifest = load_model_bundle(
        output_dir,
        trusted_local_bundle=True,
    )
    assert loaded_manifest == manifest
    predictions = model.predict(frame.head(3))
    assert predictions["reproducibility_score"].between(0, 1).all()

    with pytest.raises(FileExistsError):
        freeze_model_bundle(
            frame,
            output_dir,
            feature_columns=list(DEFAULT_FEATURE_COLUMNS),
            training_table_sha256=_sha256(training_path),
            training_table_name=training_path.name,
            training_table_path=training_path,
            feature_registry_path=ROOT / "config/features.yaml",
            environment_lock_path=ROOT / "uv.lock",
            code_revision="a" * 40,
            created_at_utc=datetime(2026, 9, 7, 12, 0, tzinfo=UTC),
        )


@pytest.mark.parametrize(
    "filename",
    [
        MODEL_FILENAME,
        FEATURES_FILENAME,
        FEATURE_REGISTRY_FILENAME,
        ENVIRONMENT_LOCK_FILENAME,
    ],
)
def test_frozen_model_verification_rejects_mutation(tmp_path, filename):
    training_path = ROOT / "examples/synthetic/labeled_gene_features.csv"
    frame = pd.read_csv(training_path)
    output_dir = tmp_path / "frozen-model"
    freeze_model_bundle(
        frame,
        output_dir,
        feature_columns=list(DEFAULT_FEATURE_COLUMNS),
        training_table_sha256=_sha256(training_path),
        training_table_name=training_path.name,
        training_table_path=training_path,
        feature_registry_path=ROOT / "config/features.yaml",
        environment_lock_path=ROOT / "uv.lock",
        code_revision="b" * 40,
        created_at_utc=datetime(2026, 9, 7, 12, 0, tzinfo=UTC),
    )
    path = output_dir / filename
    path.write_bytes(path.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="(?:hash|byte_size) mismatch"):
        verify_model_bundle(output_dir)


def test_frozen_model_manifest_is_strict_json(tmp_path):
    training_path = ROOT / "examples/synthetic/labeled_gene_features.csv"
    frame = pd.read_csv(training_path)
    output_dir = tmp_path / "frozen-model"
    freeze_model_bundle(
        frame,
        output_dir,
        feature_columns=list(DEFAULT_FEATURE_COLUMNS),
        training_table_sha256=_sha256(training_path),
        training_table_name=training_path.name,
        training_table_path=training_path,
        feature_registry_path=ROOT / "config/features.yaml",
        environment_lock_path=ROOT / "uv.lock",
        code_revision="c" * 40,
        created_at_utc=datetime(2026, 9, 7, 12, 0, tzinfo=UTC),
    )
    parsed = json.loads(
        (output_dir / "frozen_model_manifest.json").read_text(encoding="utf-8")
    )
    assert parsed["code_revision"] == "c" * 40


@pytest.mark.parametrize("features", [[], ["median_guide_lfc", "median_guide_lfc"]])
def test_frozen_model_rejects_empty_or_duplicate_feature_contract(tmp_path, features):
    training_path = ROOT / "examples/synthetic/labeled_gene_features.csv"
    frame = pd.read_csv(training_path)
    with pytest.raises(ValueError, match="feature_columns"):
        freeze_model_bundle(
            frame,
            tmp_path / "frozen-model",
            feature_columns=features,
            training_table_sha256=_sha256(training_path),
            training_table_name=training_path.name,
            training_table_path=training_path,
            feature_registry_path=ROOT / "config/features.yaml",
            environment_lock_path=ROOT / "uv.lock",
            code_revision="d" * 40,
            created_at_utc=datetime(2026, 9, 7, 12, 0, tzinfo=UTC),
        )


def test_frozen_model_verifier_rejects_malformed_file_metadata(tmp_path):
    training_path = ROOT / "examples/synthetic/labeled_gene_features.csv"
    frame = pd.read_csv(training_path)
    output_dir = tmp_path / "frozen-model"
    freeze_model_bundle(
        frame,
        output_dir,
        feature_columns=list(DEFAULT_FEATURE_COLUMNS),
        training_table_sha256=_sha256(training_path),
        training_table_name=training_path.name,
        training_table_path=training_path,
        feature_registry_path=ROOT / "config/features.yaml",
        environment_lock_path=ROOT / "uv.lock",
        code_revision="e" * 40,
        created_at_utc=datetime(2026, 9, 7, 12, 0, tzinfo=UTC),
    )
    manifest_path = output_dir / "frozen_model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_columns"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="file metadata"):
        verify_model_bundle(output_dir)


def test_frozen_model_rejects_frame_not_derived_from_pinned_table(tmp_path):
    training_path = ROOT / "examples/synthetic/labeled_gene_features.csv"
    frame = pd.read_csv(training_path)
    frame.loc[0, "label_code"] = "F0"
    with pytest.raises(ValueError, match="does not exactly match"):
        freeze_model_bundle(
            frame,
            tmp_path / "frozen-model",
            feature_columns=list(DEFAULT_FEATURE_COLUMNS),
            training_table_sha256=_sha256(training_path),
            training_table_name=training_path.name,
            training_table_path=training_path,
            feature_registry_path=ROOT / "config/features.yaml",
            environment_lock_path=ROOT / "uv.lock",
            code_revision="f" * 40,
            created_at_utc=datetime(2026, 9, 7, 12, 0, tzinfo=UTC),
        )


def test_frozen_model_manifest_can_be_verified_against_external_anchor(tmp_path):
    training_path = ROOT / "examples/synthetic/labeled_gene_features.csv"
    frame = pd.read_csv(training_path)
    output_dir = tmp_path / "frozen-model"
    freeze_model_bundle(
        frame,
        output_dir,
        feature_columns=list(DEFAULT_FEATURE_COLUMNS),
        training_table_sha256=_sha256(training_path),
        training_table_name=training_path.name,
        training_table_path=training_path,
        feature_registry_path=ROOT / "config/features.yaml",
        environment_lock_path=ROOT / "uv.lock",
        code_revision="1" * 40,
        created_at_utc=datetime(2026, 9, 7, 12, 0, tzinfo=UTC),
    )
    manifest_path = output_dir / "frozen_model_manifest.json"
    expected_hash = _sha256(manifest_path)
    assert (
        verify_model_bundle(
            output_dir,
            expected_manifest_sha256=expected_hash,
        )["code_revision"]
        == "1" * 40
    )
    with pytest.raises(ValueError, match="identity hash mismatch"):
        verify_model_bundle(
            output_dir,
            expected_manifest_sha256="2" * 64,
        )


@pytest.mark.parametrize(
    ("role", "filename", "replacement", "message"),
    [
        (
            "feature_registry",
            FEATURE_REGISTRY_FILENAME,
            b"version: 2\nfeature_groups: {}\n",
            "nonempty feature_groups",
        ),
        (
            "environment_lock",
            ENVIRONMENT_LOCK_FILENAME,
            b"version = 1\n",
            "at least one package",
        ),
    ],
)
def test_frozen_model_verifier_checks_embedded_registry_and_lock_content(
    tmp_path,
    role,
    filename,
    replacement,
    message,
):
    output_dir = tmp_path / "frozen-model"
    _freeze_fixture(output_dir)
    path = output_dir / filename
    path.write_bytes(replacement)
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[role]["sha256"] = _sha256(path)
    manifest[role]["byte_size"] = path.stat().st_size
    _rewrite_manifest(output_dir, manifest)

    with pytest.raises(ValueError, match=message):
        verify_model_bundle(output_dir)


@pytest.mark.parametrize(
    ("role", "wrong_filename"),
    [
        ("feature_columns", "columns.json"),
        ("feature_registry", "registry.yaml"),
        ("environment_lock", "requirements.lock"),
        ("model", "model.joblib"),
    ],
)
def test_frozen_model_verifier_requires_fixed_manifest_filenames(
    tmp_path,
    role,
    wrong_filename,
):
    output_dir = tmp_path / "frozen-model"
    _freeze_fixture(output_dir)
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[role]["filename"] = wrong_filename
    _rewrite_manifest(output_dir, manifest)

    with pytest.raises(ValueError, match="fixed filename mismatch"):
        verify_model_bundle(output_dir)


def test_frozen_model_verifier_rejects_wrong_model_byte_size(tmp_path):
    output_dir = tmp_path / "frozen-model"
    _freeze_fixture(output_dir)
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model"]["byte_size"] += 1
    _rewrite_manifest(output_dir, manifest)

    with pytest.raises(ValueError, match="byte_size mismatch"):
        verify_model_bundle(output_dir)


def test_frozen_model_verifier_binds_runtime_versions_to_lock(tmp_path):
    output_dir = tmp_path / "frozen-model"
    _freeze_fixture(output_dir)
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_dependencies"]["numpy"] = "999.0.0"
    _rewrite_manifest(output_dir, manifest)

    with pytest.raises(ValueError, match="absent from the environment lock"):
        verify_model_bundle(output_dir)


@pytest.mark.parametrize("revision", ["abc", "A" * 40, 17])
def test_frozen_model_verifier_rejects_noncanonical_code_revision(
    tmp_path,
    revision,
):
    output_dir = tmp_path / "frozen-model"
    _freeze_fixture(output_dir)
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["code_revision"] = revision
    _rewrite_manifest(output_dir, manifest)

    with pytest.raises(ValueError, match="code_revision"):
        verify_model_bundle(output_dir)


def test_frozen_model_verifier_rejects_unexpected_bundle_file(tmp_path):
    output_dir = tmp_path / "frozen-model"
    _freeze_fixture(output_dir)
    (output_dir / "unbound-notes.txt").write_text("not hash bound\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected files"):
        verify_model_bundle(output_dir)


def test_frozen_model_can_attest_matching_clean_source_checkout(tmp_path):
    checkout = tmp_path / "checkout"
    revision = _init_clean_checkout(checkout)
    output_dir = tmp_path / "frozen-model"
    manifest = _freeze_fixture(
        output_dir,
        code_revision=revision,
        current_checkout_path=checkout,
    )

    assert manifest["code_revision"] == revision
    assert manifest["source_tree_attestation"] == {
        "mode": "git_checkout",
        "status": "verified_clean",
        "head_revision": revision,
        "tracked_and_untracked_changes_absent": True,
        "path_not_recorded": True,
    }
    assert verify_model_bundle(output_dir) == manifest


def test_frozen_model_source_attestation_rejects_revision_mismatch(tmp_path):
    checkout = tmp_path / "checkout"
    revision = _init_clean_checkout(checkout)
    wrong_revision = ("0" if revision[0] != "0" else "1") + revision[1:]

    with pytest.raises(ValueError, match="does not match"):
        _freeze_fixture(
            tmp_path / "frozen-model",
            code_revision=wrong_revision,
            current_checkout_path=checkout,
        )


def test_frozen_model_source_attestation_rejects_dirty_checkout(tmp_path):
    checkout = tmp_path / "checkout"
    revision = _init_clean_checkout(checkout)
    (checkout / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checkout is dirty"):
        _freeze_fixture(
            tmp_path / "frozen-model",
            code_revision=revision,
            current_checkout_path=checkout,
        )
