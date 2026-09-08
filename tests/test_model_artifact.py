import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml

import crispr_evidencerank.model_artifact as artifact_module
from crispr_evidencerank.model_artifact import (
    ENVIRONMENT_LOCK_FILENAME,
    FEATURE_REGISTRY_FILENAME,
    FEATURES_FILENAME,
    MANIFEST_FILENAME,
    MODEL_BUNDLE_FILENAMES,
    MODEL_FILENAME,
    freeze_model_bundle,
    load_model_bundle,
    training_identity_sha256,
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
    logistic_c: float = 1.0,
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
        logistic_c=logistic_c,
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


def _repin_model(output_dir, manifest, model):
    path = output_dir / MODEL_FILENAME
    joblib.dump(model, path, compress=3)
    manifest["model"]["sha256"] = _sha256(path)
    manifest["model"]["byte_size"] = path.stat().st_size
    _rewrite_manifest(output_dir, manifest)


def test_loaded_payload_feature_order_must_match_bound_feature_contract(tmp_path):
    output_dir = tmp_path / "bundle"
    manifest = _freeze_fixture(output_dir)
    model = joblib.load(output_dir / MODEL_FILENAME)
    model.feature_columns = list(reversed(model.feature_columns))
    _repin_model(output_dir, manifest, model)

    with pytest.raises(ValueError, match="feature order differs"):
        load_model_bundle(output_dir, trusted_local_bundle=True)


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("model_kind", "hist_gradient_boosting"),
        ("random_state", 99),
        ("logistic_c", 2.0),
    ],
)
def test_loaded_payload_parameters_must_match_bound_manifest(
    tmp_path, attribute, value
):
    output_dir = tmp_path / "bundle"
    manifest = _freeze_fixture(output_dir)
    model = joblib.load(output_dir / MODEL_FILENAME)
    setattr(model, attribute, value)
    _repin_model(output_dir, manifest, model)

    with pytest.raises(ValueError, match=f"{attribute} differs"):
        load_model_bundle(output_dir, trusted_local_bundle=True)


def test_loaded_pipeline_feature_order_must_match_wrapper_and_manifest(tmp_path):
    output_dir = tmp_path / "bundle"
    manifest = _freeze_fixture(output_dir)
    model = joblib.load(output_dir / MODEL_FILENAME)
    preprocessor = model.success_model.named_steps["preprocess"]
    preprocessor.feature_names_in_ = preprocessor.feature_names_in_[::-1]
    _repin_model(output_dir, manifest, model)

    with pytest.raises(ValueError, match="incompatible feature order"):
        load_model_bundle(output_dir, trusted_local_bundle=True)


def test_loader_checks_replaced_pickle_before_deserializing(tmp_path, monkeypatch):
    output_dir = tmp_path / "bundle"
    _freeze_fixture(output_dir)
    real_verify = artifact_module.verify_model_bundle

    def replace_after_verification(*args, **kwargs):
        verified = real_verify(*args, **kwargs)
        (output_dir / MODEL_FILENAME).write_bytes(b"replacement-pickle")
        return verified

    def forbidden_load(*args, **kwargs):
        pytest.fail("unverified replacement was passed to joblib")

    monkeypatch.setattr(
        artifact_module, "verify_model_bundle", replace_after_verification
    )
    monkeypatch.setattr(artifact_module.joblib, "load", forbidden_load)
    with pytest.raises(ValueError, match="changed before loading"):
        load_model_bundle(output_dir, trusted_local_bundle=True)


def test_loader_deserializes_verified_bytes_even_when_path_is_replaced(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "bundle"
    _freeze_fixture(output_dir)
    real_load = joblib.load

    def replace_when_loading(source):
        assert hasattr(source, "read")
        (output_dir / MODEL_FILENAME).write_bytes(b"replacement-pickle")
        return real_load(source)

    monkeypatch.setattr(artifact_module.joblib, "load", replace_when_loading)
    model, _ = load_model_bundle(output_dir, trusted_local_bundle=True)
    assert model.feature_columns == list(DEFAULT_FEATURE_COLUMNS)


def test_loader_rejects_incompatible_runtime_before_deserializing(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "bundle"
    _freeze_fixture(output_dir)
    real_version = artifact_module.importlib_metadata.version
    monkeypatch.setattr(
        artifact_module.importlib_metadata,
        "version",
        lambda name: "0.0.0" if name == "scikit-learn" else real_version(name),
    )
    monkeypatch.setattr(
        artifact_module.joblib,
        "load",
        lambda *args, **kwargs: pytest.fail("incompatible runtime reached joblib"),
    )
    # Integrity verification must remain portable, since it never unpickles.
    assert verify_model_bundle(output_dir)["schema_version"] == 1
    with pytest.raises(ValueError, match="dependency versions differ"):
        load_model_bundle(output_dir, trusted_local_bundle=True)


@pytest.mark.parametrize(
    "replacement,message",
    [
        ('{"schema_version":1,"schema_version":1}', "duplicate key"),
        ('{"schema_version":NaN}', "non-finite"),
    ],
)
def test_verifier_rejects_ambiguous_or_nonstandard_json(tmp_path, replacement, message):
    output_dir = tmp_path / "bundle"
    _freeze_fixture(output_dir)
    (output_dir / MANIFEST_FILENAME).write_text(replacement)
    with pytest.raises(ValueError, match=message):
        verify_model_bundle(output_dir)


def test_frozen_inventory_records_hashes_without_plaintext_family_ids(tmp_path):
    output_dir = tmp_path / "bundle"
    manifest = _freeze_fixture(output_dir)
    frame = pd.read_csv(ROOT / "examples/synthetic/labeled_gene_features.csv")
    inventory = manifest["training_provenance"]["identities"]
    assert inventory["study_id"]["status"] == "complete"
    assert inventory["study_id"]["identity_sha256"] == sorted(
        {training_identity_sha256("study_id", value) for value in frame.study_id}
    )
    for role in ("source_family_id", "raw_data_family_id"):
        if role not in frame:
            assert inventory[role]["status"] == "not_attested"
            assert inventory[role]["missing_row_count"] == len(frame)
    assert training_identity_sha256("study_id", " S1 ") == training_identity_sha256(
        "study_id", "S1"
    )
    assert training_identity_sha256("study_id", "S1") != training_identity_sha256(
        "source_family_id", "S1"
    )
    assert (
        verify_model_bundle(output_dir)["training_provenance"]
        == manifest["training_provenance"]
    )


def test_verifier_accepts_legacy_v1_without_identity_inventory(tmp_path):
    output_dir = tmp_path / "bundle"
    manifest = _freeze_fixture(output_dir)
    del manifest["training_provenance"]
    del manifest["model"]["logistic_c"]
    _rewrite_manifest(output_dir, manifest)
    _, verified = load_model_bundle(output_dir, trusted_local_bundle=True)
    assert "training_provenance" not in verified


def test_verifier_rejects_inconsistent_inventory_and_missing_dependency(tmp_path):
    output_dir = tmp_path / "bundle"
    manifest = _freeze_fixture(output_dir)
    manifest["training_provenance"]["identities"]["study_id"]["missing_row_count"] = 1
    _rewrite_manifest(output_dir, manifest)
    with pytest.raises(ValueError, match="training_provenance"):
        verify_model_bundle(output_dir)
    manifest["training_provenance"]["identities"]["study_id"]["missing_row_count"] = 0
    del manifest["runtime_dependencies"]["numpy"]
    _rewrite_manifest(output_dir, manifest)
    with pytest.raises(ValueError, match="must declare exactly"):
        verify_model_bundle(output_dir)


def test_freeze_does_not_replace_concurrently_created_empty_destination(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "bundle"
    real_publish = artifact_module._publish_directory_noreplace

    def raced_publish(source, destination):
        destination.mkdir()
        return real_publish(source, destination)

    monkeypatch.setattr(artifact_module, "_publish_directory_noreplace", raced_publish)
    with pytest.raises(FileExistsError):
        _freeze_fixture(output_dir)
    assert output_dir.is_dir()
    assert not list(output_dir.iterdir())
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_frozen_model_preserves_selected_logistic_regularization(tmp_path):
    output_dir = tmp_path / "bundle"
    manifest = _freeze_fixture(output_dir, logistic_c=0.25)
    assert manifest["model"]["logistic_c"] == 0.25
    model, _ = load_model_bundle(output_dir, trusted_local_bundle=True)
    assert model.logistic_c == 0.25
    assert model.success_model.named_steps["model"].C == 0.25
    if model.selection_model is not None:
        assert model.selection_model.named_steps["model"].C == 1.0


def test_verifier_honors_embedded_registry_primary_model_policy(tmp_path):
    output_dir = tmp_path / "bundle"
    manifest = _freeze_fixture(output_dir)
    path = output_dir / FEATURE_REGISTRY_FILENAME
    registry = yaml.safe_load(path.read_text())
    registry["feature_groups"]["screen_signal"]["allowed_for_primary_model"] = False
    path.write_text(yaml.safe_dump(registry))
    manifest["feature_registry"]["sha256"] = _sha256(path)
    manifest["feature_registry"]["byte_size"] = path.stat().st_size
    _rewrite_manifest(output_dir, manifest)
    with pytest.raises(ValueError, match="does not allow frozen features"):
        verify_model_bundle(output_dir)


def test_training_inventory_marks_partial_family_coverage_without_imputing(tmp_path):
    frame = pd.DataFrame(
        {
            "study_id": ["S1", "S2", "S3"],
            "source_family_id": [" A ", pd.NA, " "],
        }
    )
    inventory = artifact_module._training_identity_inventory(
        frame, ("study_id", "source_family_id", "raw_data_family_id")
    )
    source = inventory["identities"]["source_family_id"]
    assert source["status"] == "partial"
    assert source["missing_row_count"] == 2
    assert source["identity_sha256"] == [
        training_identity_sha256("source_family_id", "A")
    ]
    assert inventory["identities"]["raw_data_family_id"]["status"] == "not_attested"
    artifact_module._validate_training_identity_inventory(inventory, len(frame))
