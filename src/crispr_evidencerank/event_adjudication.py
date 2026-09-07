"""Human review and adjudication of validation-event evidence.

The functions in this module deliberately separate three immutable stages:

1. a label-free packet shown to both reviewers;
2. a deterministic comparison of two independently attested human reviews; and
3. a named human adjudication that can emit validation-event records.

No stage in this module establishes screen-level benchmark readiness.  Bundle
manifests preserve byte-level lineage, while directory publication is atomic
for cooperating writers and always refuses to overwrite an existing target.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from io import BytesIO
from numbers import Integral, Real
from pathlib import Path

import pandas as pd
import yaml

from .contracts import (
    AdjudicationResolutionMethod,
    CriterionAssessment,
    CurationActorRecord,
    CurationActorType,
    EventAdjudicationRecord,
    EventEvidenceCardRecord,
    EventReviewComparisonRecord,
    EventReviewComparisonStatus,
    EventReviewRecord,
    PerturbationModality,
    ValidationContextMatch,
    ValidationEventRecord,
    ValidationOutcomeRelation,
    validate_records,
)
from .labels import LabelCode

PACKET_FILENAMES = (
    "event_evidence_packet.tsv",
    "event_review_template.tsv",
    "event_review_rubric.yaml",
    "event_packet_manifest.json",
)

COMPARISON_FILENAMES = (
    *PACKET_FILENAMES,
    "curation_actors.tsv",
    "event_reviews_a.tsv",
    "event_reviews_b.tsv",
    "event_review_comparison.tsv",
    "event_review_agreement.json",
    "adjudication_template.tsv",
    "event_review_comparison_manifest.json",
)

EVENT_REVIEW_DECISION_FIELDS = (
    "event_scope",
    "context_match",
    "testing_status",
    "perturbation_confirmed",
    "independent_reagent_count",
    "orthogonal_perturbation",
    "appropriate_control",
    "assay_adequate",
    "outcome_relation",
    "null_result_evaluable",
    "rescue_or_causal_reversal",
    "proposed_label_code",
    "confidence",
    "source_locator",
    "reason_codes",
)

LABEL_ORDER = tuple(code.value for code in LabelCode)
PRIMARY_EVENT_LABELS = frozenset(
    {LabelCode.V2.value, LabelCode.V3.value, LabelCode.F0.value, LabelCode.D.value}
)

# Column names are checked before Pydantic validation so case, punctuation, or
# nested JSON cannot be used to smuggle an outcome/model field into a packet.
_FORBIDDEN_PACKET_FIELD_FRAGMENTS = frozenset(
    {
        "ailabel",
        "aiprediction",
        "aiprobability",
        "aiscore",
        "assignedlabel",
        "benchmarkready",
        "candidatev1",
        "candidatev2",
        "candidatev3",
        "comparisonstatus",
        "curatora",
        "curatoradecision",
        "curatorb",
        "curatorbdecision",
        "evidencerank",
        "finallabel",
        "labelcode",
        "mageckrank",
        "mageckscore",
        "modelprobability",
        "modelprediction",
        "modelscore",
        "peerlabel",
        "predictedprobability",
        "proposedlabel",
        "reviewera",
        "reviewerb",
        "reviewerdecision",
        "reviewerlabel",
        "validationsuccessestimate",
    }
)

_FORBIDDEN_PACKET_TEXT_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\bcandidate[_ -]?v[123]\b",
        r"\bevidence\s*rank\b",
        r"\blabel[_ -]?code\b",
        r"\b(?:ai|mageck|model)[_ -]+(?:rank|score|probability|prediction)\b",
        r"\b(?:rank|score|probability|prediction)[_ -]+(?:from|by)"
        r"[_ -]+(?:the[_ -]+)?(?:ai|model)\b",
        r"\bpredicted[_ -]?probability\b",
        r"\bvalidation[_ -]?success[_ -]?estimate\b",
        r"\b(?:ai|assigned|curator|final|peer|proposed|reviewer)"
        r"[_ -]+label\b",
        r"\blabel[_ -]+(?:assigned|chosen|proposed|selected)\b",
        r"\b(?:curator|reviewer)[_ -]*[ab]\b",
        r"\b(?:class|grade|rated|rating)(?:[_ -]+as)?"
        r"[^a-z0-9]{0,8}(?:v[123]|f0|d)\b",
        r"\b(?:v[123]|f0|d)[^a-z0-9]{0,8}(?:class|grade|rating)\b",
    )
)

_REVIEW_FORBIDDEN_COLUMNS = frozenset(
    {
        "benchmark_ready",
        "adjudicated_label",
        "final_label",
        "model_score",
        "peer_label",
        "predicted_probability",
    }
)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256(path: str | Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _file_fact(filename: str, content: bytes) -> dict[str, object]:
    return {
        "filename": filename,
        "sha256": _sha256_bytes(content),
        "byte_size": len(content),
    }


def _content_fact(content: bytes) -> dict[str, object]:
    """Describe an input without persisting a potentially leaky source name."""

    return {
        "sha256": _sha256_bytes(content),
        "byte_size": len(content),
    }


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _require_sha256(value: str, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return normalized


def _require_utc(value: datetime, *, field_name: str) -> datetime:
    timestamp = pd.Timestamp(value).to_pydatetime()
    offset = timestamp.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f"{field_name} must include the UTC timezone")
    return timestamp


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


def _json_output_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _read_table_bytes(
    content: bytes,
    suffix: str,
    *,
    contract: type | None = None,
    role: str = "table",
) -> pd.DataFrame:
    if suffix.lower() == ".csv":
        frame = pd.read_csv(BytesIO(content), dtype="string", keep_default_na=False)
    elif suffix.lower() in {".tsv", ".txt"}:
        frame = pd.read_csv(
            BytesIO(content),
            sep="\t",
            dtype="string",
            keep_default_na=False,
        )
    else:
        raise ValueError(f"unsupported table extension: {suffix}")
    if contract is not None:
        frame = _decode_strict_bool_columns(frame, contract, role=role)
    return frame


def _decode_strict_bool_columns(
    frame: pd.DataFrame,
    contract: type,
    *,
    role: str,
) -> pd.DataFrame:
    """Decode only canonical TSV/CSV spellings for StrictBool fields.

    Tabular serialization has no native boolean type, so pandas necessarily
    reads booleans as text when every column is loaded losslessly.  Restricting
    this conversion to contract fields marked ``Strict`` keeps values such as
    ``1``, ``yes``, or whitespace-padded tokens from being silently coerced.
    Programmatic DataFrames bypass this decoder and retain Pydantic's native
    StrictBool behavior.
    """

    strict_bool_fields = {
        name
        for name, field in contract.model_fields.items()
        if field.annotation is bool
        and any(getattr(metadata, "strict", False) for metadata in field.metadata)
    }
    decoded = frame.copy()
    canonical_tokens = {
        "True": True,
        "False": False,
        "true": True,
        "false": False,
    }
    for column in sorted(strict_bool_fields & set(decoded.columns)):
        values: list[object] = []
        for row_index, value in decoded[column].items():
            if value is None or pd.isna(value):
                values.append(None)
            elif isinstance(value, str) and value in canonical_tokens:
                values.append(canonical_tokens[value])
            elif isinstance(value, bool):
                values.append(value)
            else:
                raise ValueError(
                    f"{role} has noncanonical boolean token in {column} "
                    f"at row {int(row_index) + 2}"
                )
        decoded[column] = values
    return decoded


def _table_output_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        sep="\t",
        index=False,
        lineterminator="\n",
    ).encode("utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _parse_json_object(content: bytes, *, role: str) -> dict[str, object]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _parse_rubric(content: bytes) -> tuple[dict[str, object], str]:
    try:
        rubric = yaml.safe_load(content.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("event review rubric is not valid UTF-8 YAML") from exc
    if not isinstance(rubric, dict):
        raise ValueError("event review rubric must be a YAML mapping")
    rubric_id = rubric.get("rubric_id")
    if not isinstance(rubric_id, str) or not rubric_id.strip():
        raise ValueError("event review rubric requires rubric_id")
    release_policy = rubric.get("release_policy")
    required_guards = (
        "exact_reviewer_agreement_is_not_a_released_label",
        "named_human_adjudication_required",
        "reviewer_submissions_are_immutable",
        "benchmark_ready_is_never_created_by_review_or_adjudication",
    )
    if not isinstance(release_policy, dict) or any(
        release_policy.get(name) is not True for name in required_guards
    ):
        raise ValueError("event review rubric lacks required fail-closed release rules")
    return rubric, rubric_id.strip()


def _normalized_field_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _walk_mapping_keys(value: object) -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_mapping_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_mapping_keys(nested)


def _has_forbidden_packet_text(value: str) -> bool:
    normalized = _normalized_field_name(value)
    return any(
        fragment in normalized for fragment in _FORBIDDEN_PACKET_FIELD_FRAGMENTS
    ) or any(pattern.search(value) for pattern in _FORBIDDEN_PACKET_TEXT_PATTERNS)


def _assert_packet_value_is_label_free(value: object, *, location: str) -> None:
    """Recursively reject review/model/label content in any card value."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            normalized_key = _normalized_field_name(key_text)
            normalized_nested = (
                _normalized_field_name(nested) if isinstance(nested, str) else ""
            )
            if normalized_key in {"curator", "reviewer"} and normalized_nested in {
                "a",
                "b",
            }:
                raise ValueError(
                    f"distributable packet contains reviewer identity leakage at "
                    f"{location}"
                )
            if _has_forbidden_packet_text(key_text):
                raise ValueError(
                    f"distributable packet contains forbidden nested key at {location}"
                )
            _assert_packet_value_is_label_free(
                nested,
                location=f"{location}.{key_text}",
            )
        if _has_forbidden_packet_text(
            json.dumps(value, default=str, ensure_ascii=True, sort_keys=True)
        ):
            raise ValueError(
                f"distributable packet contains reviewer/model/label leakage at "
                f"{location}"
            )
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_packet_value_is_label_free(
                nested,
                location=f"{location}[{index}]",
            )
        if _has_forbidden_packet_text(
            json.dumps(value, default=str, ensure_ascii=True)
        ):
            raise ValueError(
                f"distributable packet contains reviewer/model/label leakage at "
                f"{location}"
            )
        return
    if not isinstance(value, str) or not value.strip():
        return

    stripped = value.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            nested = json.loads(value)
        except json.JSONDecodeError:
            nested = None
        if isinstance(nested, (Mapping, list)):
            _assert_packet_value_is_label_free(nested, location=location)
    if _has_forbidden_packet_text(value):
        raise ValueError(
            f"distributable packet contains reviewer/model/label leakage at {location}"
        )


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    if not pd.api.types.is_scalar(value):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _assert_packet_is_label_free(frame: pd.DataFrame) -> None:
    allowed = set(EventEvidenceCardRecord.model_fields)
    unexpected = sorted(set(frame.columns) - allowed)
    if unexpected:
        raise ValueError(
            f"evidence cards contain non-allowlisted columns: {unexpected}"
        )
    if "notes" in frame:
        notes_present = frame["notes"].map(
            lambda value: not _is_missing_scalar(value) and bool(str(value).strip())
        )
        if notes_present.any():
            raise ValueError(
                "distributable evidence cards require empty notes; unstructured "
                "pipeline notes are not distributable"
            )
    for column in frame.columns:
        normalized = _normalized_field_name(column)
        if any(
            fragment in normalized for fragment in _FORBIDDEN_PACKET_FIELD_FRAGMENTS
        ):
            raise ValueError(
                f"distributable packet contains forbidden column: {column}"
            )

    for column in frame.columns:
        for row_index, value in frame[column].items():
            if _is_missing_scalar(value):
                continue
            _assert_packet_value_is_label_free(
                value,
                location=f"{column} row {int(row_index) + 2}",
            )


def _assert_no_readiness(value: object) -> None:
    for key in _walk_mapping_keys(value):
        if "benchmarkready" in _normalized_field_name(key):
            raise RuntimeError(
                "review/adjudication outputs cannot emit benchmark_ready"
            )


def _validate_frame(
    frame: pd.DataFrame,
    contract: type,
    *,
    role: str,
) -> pd.DataFrame:
    forbidden = sorted(_REVIEW_FORBIDDEN_COLUMNS & set(frame.columns))
    if forbidden:
        raise ValueError(f"{role} contains forbidden release columns: {forbidden}")
    normalized_input = frame.replace("", None)
    valid, errors = validate_records(normalized_input, contract)
    if not errors.empty:
        details = errors.head(5).to_dict(orient="records")
        raise ValueError(f"{role} failed contract validation: {details}")
    return valid.loc[:, list(contract.model_fields)]


def _validate_actors(frame: pd.DataFrame) -> pd.DataFrame:
    actors = _validate_frame(frame, CurationActorRecord, role="curation actors")
    canonical_ids = actors["actor_id"].astype(str).str.strip().str.casefold()
    if canonical_ids.duplicated().any():
        raise ValueError("curation actor IDs must be unique after case normalization")
    persistent = actors["persistent_id"].dropna().astype(str).str.strip().str.casefold()
    persistent = persistent.loc[persistent.ne("")]
    if persistent.duplicated().any():
        raise ValueError("curation actors cannot reuse a persistent identity")
    return actors


def _actor_lookup(actors: pd.DataFrame) -> dict[str, dict[str, object]]:
    return {
        str(row["actor_id"]).strip().casefold(): row.to_dict()
        for _, row in actors.iterrows()
    }


def _require_actor(
    actor_id: object,
    actors: pd.DataFrame,
    *,
    role: str,
    require_human: bool,
) -> dict[str, object]:
    lookup = _actor_lookup(actors)
    actor = lookup.get(str(actor_id).strip().casefold())
    if actor is None:
        raise ValueError(f"{role} is absent from the curation actor registry")
    if actor.get("identity_verified") is not True:
        raise ValueError(f"{role} requires identity_verified=true")
    if require_human and actor.get("actor_type") != CurationActorType.HUMAN.value:
        raise ValueError(f"{role} must resolve to a verified human actor")
    return actor


def _actor_identity_key(actor: Mapping[str, object]) -> str:
    persistent = actor.get("persistent_id")
    if persistent is not None and str(persistent).strip():
        return f"persistent:{str(persistent).strip().casefold()}"
    return f"name:{str(actor['display_name']).strip().casefold()}"


def _card_payload(record: Mapping[str, object]) -> str:
    payload = dict(record)
    payload.pop("packet_payload_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(payload))


def _normalize_evidence_cards(frame: pd.DataFrame) -> pd.DataFrame:
    _assert_packet_is_label_free(frame)
    required_without_hash = {
        name
        for name, field in EventEvidenceCardRecord.model_fields.items()
        if field.is_required() and name != "packet_payload_sha256"
    }
    missing = sorted(required_without_hash - set(frame.columns))
    if missing:
        raise ValueError(f"evidence cards are missing required columns: {missing}")

    records: list[dict[str, object]] = []
    for row_number, raw in enumerate(frame.to_dict(orient="records"), start=2):
        clean = {
            key: (None if value is None or pd.isna(value) or value == "" else value)
            for key, value in raw.items()
        }
        supplied_hash = clean.get("packet_payload_sha256")
        clean["packet_payload_sha256"] = "0" * 64
        try:
            provisional = EventEvidenceCardRecord.model_validate(clean).model_dump(
                mode="json"
            )
        except Exception as exc:
            raise ValueError(
                f"evidence card row {row_number} failed contract validation: {exc}"
            ) from exc
        expected_hash = _card_payload(provisional)
        if supplied_hash is not None and str(supplied_hash).strip() != expected_hash:
            raise ValueError(
                f"evidence card row {row_number} packet_payload_sha256 mismatch"
            )
        provisional["packet_payload_sha256"] = expected_hash
        records.append(
            EventEvidenceCardRecord.model_validate(provisional).model_dump(mode="json")
        )

    if not records:
        raise ValueError("event review packet requires at least one evidence card")
    normalized = pd.DataFrame.from_records(records)
    key = list(EventEvidenceCardRecord.primary_key)
    if normalized.duplicated(key).any():
        raise ValueError(f"evidence cards contain duplicate primary key: {tuple(key)}")
    return normalized.loc[:, list(EventEvidenceCardRecord.model_fields)]


def _deterministic_packet_order(
    cards: pd.DataFrame,
    *,
    randomization_seed: str,
) -> pd.DataFrame:
    if not randomization_seed:
        raise ValueError("randomization_seed cannot be empty")
    ordered = cards.copy()
    ordered["_blind_order"] = ordered.apply(
        lambda row: hashlib.sha256(
            (
                f"{randomization_seed}\0{row['packet_id']}\0"
                f"{row['packet_version']}\0{row['packet_event_id']}\0"
                f"{row['packet_payload_sha256']}"
            ).encode()
        ).hexdigest(),
        axis=1,
    )
    ordered = ordered.sort_values("_blind_order", kind="stable").drop(
        columns="_blind_order"
    )
    return ordered.reset_index(drop=True)


def _blank_review_template(cards: pd.DataFrame, *, rubric_version: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, card in cards.iterrows():
        row = {name: "" for name in EventReviewRecord.model_fields}
        for name in (
            "packet_id",
            "packet_version",
            "packet_event_id",
            "packet_payload_sha256",
        ):
            row[name] = card[name]
        row["rubric_version"] = rubric_version
        rows.append(row)
    return pd.DataFrame(rows, columns=list(EventReviewRecord.model_fields))


def prepare_event_review_packet(
    evidence_cards: pd.DataFrame,
    actors: pd.DataFrame,
    *,
    rubric_version: str,
    randomization_seed: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a randomized, content-filtered packet and blank review template."""

    if not rubric_version.strip():
        raise ValueError("rubric_version cannot be empty")
    valid_actors = _validate_actors(actors)
    cards = _normalize_evidence_cards(evidence_cards)
    for actor_id in cards["prepared_by_actor_id"].astype(str).unique():
        _require_actor(
            actor_id,
            valid_actors,
            role="evidence-card preparer",
            require_human=False,
        )
    ordered = _deterministic_packet_order(
        cards,
        randomization_seed=randomization_seed,
    )
    template = _blank_review_template(ordered, rubric_version=rubric_version)
    return ordered, template


@contextmanager
def _atomic_output_directory(
    output_dir: str | Path,
) -> Iterator[tuple[Path, Path]]:
    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.publish.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"publisher lock already exists: {lock_path}") from exc
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        if _path_lexists(destination):
            raise FileExistsError(f"output directory already exists: {destination}")
        with tempfile.TemporaryDirectory(
            dir=destination.parent,
            prefix=f".{destination.name}.work-",
        ) as work_name:
            work_dir = Path(work_name)
            publish_dir = work_dir / "publish"
            publish_dir.mkdir()
            yield work_dir, publish_dir
            if _path_lexists(destination):
                raise FileExistsError(
                    f"output destination appeared while staging: {destination}"
                )
            publish_dir.rename(destination)
    finally:
        os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _snapshot_files(
    paths: Mapping[str, Path],
    snapshot_dir: Path,
) -> tuple[dict[str, bytes], dict[str, Path]]:
    contents = {role: path.read_bytes() for role, path in paths.items()}
    snapshots: dict[str, Path] = {}
    for role, content in contents.items():
        snapshot = snapshot_dir / role / paths[role].name
        _write_bytes(snapshot, content)
        snapshots[role] = snapshot
    return contents, snapshots


def _packet_roster_sha256(cards: pd.DataFrame) -> str:
    roster = [
        [
            str(row["packet_id"]),
            int(row["packet_version"]),
            str(row["packet_event_id"]),
            str(row["packet_payload_sha256"]),
        ]
        for _, row in cards.iterrows()
    ]
    return _sha256_bytes(_canonical_json_bytes(roster))


def write_event_review_packet(
    evidence_cards_path: str | Path,
    actor_registry_path: str | Path,
    rubric_path: str | Path,
    output_dir: str | Path,
    *,
    randomization_seed: str,
    access_isolation_attested: bool,
    neutral_card_audit_attested: bool,
) -> dict[str, object]:
    """Publish after explicit access-isolation and neutral-card attestations."""

    if access_isolation_attested is not True:
        raise ValueError(
            "access_isolation_attested must be exactly true before distributing "
            "joinable evidence identifiers"
        )
    if neutral_card_audit_attested is not True:
        raise ValueError(
            "neutral_card_audit_attested must be exactly true after manual "
            "review of every distributable evidence card"
        )

    evidence_path = Path(evidence_cards_path)
    actor_path = Path(actor_registry_path)
    rubric_source = Path(rubric_path)
    source_paths = {
        "evidence_cards": evidence_path,
        "actors": actor_path,
        "rubric": rubric_source,
    }

    with _atomic_output_directory(output_dir) as (work_dir, publish_dir):
        contents, _ = _snapshot_files(
            source_paths,
            work_dir / "snapshots",
        )
        _, rubric_version = _parse_rubric(contents["rubric"])
        evidence = _read_table_bytes(contents["evidence_cards"], evidence_path.suffix)
        actors = _read_table_bytes(
            contents["actors"],
            actor_path.suffix,
            contract=CurationActorRecord,
            role="curation actors",
        )
        valid_actors = _validate_actors(actors)
        canonical_actor_content = _table_output_bytes(valid_actors)
        cards, template = prepare_event_review_packet(
            evidence,
            valid_actors,
            rubric_version=rubric_version,
            randomization_seed=randomization_seed,
        )
        batch_ids = sorted(set(cards["batch_id"].astype(str)))
        if len(batch_ids) != 1:
            raise ValueError("one review packet requires exactly one batch_id")

        packet_bytes = _table_output_bytes(cards)
        template_bytes = _table_output_bytes(template)
        output_contents = {
            "event_evidence_packet.tsv": packet_bytes,
            "event_review_template.tsv": template_bytes,
            "event_review_rubric.yaml": contents["rubric"],
        }
        for filename, content in output_contents.items():
            _write_bytes(publish_dir / filename, content)

        manifest: dict[str, object] = {
            "schema_version": 1,
            "status": "event_review_packet_prepared",
            "batch_id": batch_ids[0],
            "rubric_version": rubric_version,
            "rubric_sha256": _sha256_bytes(contents["rubric"]),
            "record_count": len(cards),
            "packet_roster_sha256": _packet_roster_sha256(cards),
            "randomization": {
                "method": "sha256_seeded_card_order_v1",
                "seed_sha256": _sha256_bytes(randomization_seed.encode()),
            },
            "content_exclusions": {
                "peer_review_decisions_not_ingested": True,
                "model_scores_not_ingested": True,
                "prior_labels_not_ingested": True,
                "screen_predicted_direction_visible": True,
            },
            "access_isolation": {
                "attested": True,
                "evidence": "caller_attestation_only",
                "scope": (
                    "reviewer workspace is isolated from joinable prior-review, "
                    "outcome-aware, and model-output artifacts"
                ),
            },
            "neutral_card_audit": {
                "attested": True,
                "evidence": "caller_attestation_only",
                "method": "manual_review_of_every_distributable_card",
                "limitation": (
                    "automated leakage patterns do not prove cognitive neutrality"
                ),
            },
            "source_evidence_cards": _content_fact(contents["evidence_cards"]),
            "source_actor_registry": _content_fact(contents["actors"]),
            "actor_registry": _content_fact(canonical_actor_content),
            "outputs": {
                filename: _file_fact(filename, content)
                for filename, content in output_contents.items()
            },
            "scientific_boundary": {
                "agreement_does_not_release_a_label": True,
                "human_adjudication_required": True,
                "packet_is_not_a_benchmark_release": True,
                "prior_review_artifacts_not_ingested": True,
                "identifier_aliasing_not_applied": True,
                "external_source_and_declaration_bundle_required": True,
            },
        }
        _assert_no_readiness(manifest)
        manifest_bytes = _json_output_bytes(manifest)
        _write_bytes(publish_dir / "event_packet_manifest.json", manifest_bytes)
    return manifest


def _verify_packet_bundle(
    contents: Mapping[str, bytes],
    *,
    expected_manifest_sha256: str,
    actor_content: bytes | None = None,
) -> tuple[dict[str, object], pd.DataFrame, str]:
    expected_sha = _require_sha256(
        expected_manifest_sha256,
        field_name="expected_packet_manifest_sha256",
    )
    manifest_content = contents["event_packet_manifest.json"]
    if _sha256_bytes(manifest_content) != expected_sha:
        raise ValueError("event packet manifest SHA-256 does not match expected")
    manifest = _parse_json_object(manifest_content, role="event packet manifest")
    _assert_no_readiness(manifest)
    if manifest.get("status") != "event_review_packet_prepared":
        raise ValueError("event packet manifest has an unexpected status")
    content_exclusions = manifest.get("content_exclusions")
    if not isinstance(content_exclusions, dict) or any(
        content_exclusions.get(name) is not True
        for name in (
            "peer_review_decisions_not_ingested",
            "model_scores_not_ingested",
            "prior_labels_not_ingested",
        )
    ):
        raise ValueError("event packet manifest lacks required content exclusions")
    access_isolation = manifest.get("access_isolation")
    if not isinstance(access_isolation, dict) or (
        access_isolation.get("attested") is not True
    ):
        raise ValueError("event packet manifest lacks access-isolation attestation")
    neutral_card_audit = manifest.get("neutral_card_audit")
    if not isinstance(neutral_card_audit, dict) or (
        neutral_card_audit.get("attested") is not True
        or neutral_card_audit.get("evidence") != "caller_attestation_only"
        or neutral_card_audit.get("method")
        != "manual_review_of_every_distributable_card"
    ):
        raise ValueError("event packet manifest lacks neutral-card audit attestation")
    scientific_boundary = manifest.get("scientific_boundary")
    if not isinstance(scientific_boundary, dict) or (
        scientific_boundary.get("external_source_and_declaration_bundle_required")
        is not True
    ):
        raise ValueError(
            "event packet manifest must require the external source and "
            "declaration bundle"
        )

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("event packet manifest lacks output checksums")
    for filename in PACKET_FILENAMES[:-1]:
        fact = outputs.get(filename)
        if not isinstance(fact, dict):
            raise ValueError(f"event packet manifest lacks output: {filename}")
        if fact.get("filename") != filename:
            raise ValueError(f"event packet manifest has noncanonical name: {filename}")
        if fact.get("sha256") != _sha256_bytes(contents[filename]):
            raise ValueError(f"event packet output checksum mismatch: {filename}")
        if fact.get("byte_size") != len(contents[filename]):
            raise ValueError(f"event packet output byte size mismatch: {filename}")

    if actor_content is not None:
        actor_fact = manifest.get("actor_registry")
        if not isinstance(actor_fact, dict) or actor_fact.get("sha256") != (
            _sha256_bytes(actor_content)
        ):
            raise ValueError("actor registry differs from the packet lineage")

    cards_raw = _read_table_bytes(contents["event_evidence_packet.tsv"], ".tsv")
    cards = _normalize_evidence_cards(cards_raw)
    # Packet rows are already frozen in their presentation order.  Recomputing
    # payloads above verifies every row without re-randomizing it.
    if _packet_roster_sha256(cards) != manifest.get("packet_roster_sha256"):
        raise ValueError("event packet roster checksum mismatch")
    _, rubric_version = _parse_rubric(contents["event_review_rubric.yaml"])
    if rubric_version != manifest.get("rubric_version"):
        raise ValueError("rubric version differs from the packet manifest")
    if _sha256_bytes(contents["event_review_rubric.yaml"]) != manifest.get(
        "rubric_sha256"
    ):
        raise ValueError("rubric checksum differs from the packet manifest")
    return manifest, cards, rubric_version


def _review_key(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(
        frame[["packet_id", "packet_version", "packet_event_id"]].astype(str)
    )


def _validate_review_stream(
    reviews: pd.DataFrame,
    cards: pd.DataFrame,
    actors: pd.DataFrame,
    *,
    role: str,
    rubric_version: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    unexpected = sorted(_REVIEW_FORBIDDEN_COLUMNS & set(reviews.columns))
    if unexpected:
        raise ValueError(f"{role} reviews contain release columns: {unexpected}")
    valid = _validate_frame(reviews, EventReviewRecord, role=f"{role} reviews")
    card_keys = _review_key(cards)
    review_keys = _review_key(valid)
    if review_keys.duplicated().any():
        raise ValueError(f"{role} reviews require one row per packet event")
    if set(review_keys) != set(card_keys) or len(valid) != len(cards):
        raise ValueError(f"{role} reviews do not exactly cover the frozen packet")

    sessions = sorted(set(valid["review_session_id"].astype(str)))
    reviewers = sorted(set(valid["reviewer_id"].astype(str)))
    rubrics = sorted(set(valid["rubric_version"].astype(str)))
    if len(sessions) != 1 or len(reviewers) != 1:
        raise ValueError(f"{role} review file requires one reviewer and one session")
    if rubrics != [rubric_version]:
        raise ValueError(f"{role} reviews use the wrong rubric version")
    actor = _require_actor(
        reviewers[0],
        actors,
        role=f"{role} reviewer",
        require_human=True,
    )
    reviewer_identity = _actor_identity_key(actor)
    for preparer_id in cards["prepared_by_actor_id"].astype(str).unique():
        preparer = _require_actor(
            preparer_id,
            actors,
            role="evidence-card preparer",
            require_human=False,
        )
        if _actor_identity_key(preparer) == reviewer_identity:
            raise ValueError(
                f"{role} reviewer cannot also prepare a reviewed evidence card"
            )

    aligned_cards = cards.set_index(
        ["packet_id", "packet_version", "packet_event_id"], drop=False
    )
    for _, review in valid.iterrows():
        key = (
            str(review["packet_id"]),
            int(review["packet_version"]),
            str(review["packet_event_id"]),
        )
        card = aligned_cards.loc[key]
        if str(review["packet_payload_sha256"]) != str(card["packet_payload_sha256"]):
            raise ValueError(f"{role} review is bound to the wrong card payload")
        reviewed_at = pd.Timestamp(review["reviewed_at_utc"])
        prepared_at = pd.Timestamp(card["prepared_at_utc"])
        if reviewed_at < prepared_at:
            raise ValueError(f"{role} review predates its evidence card")
    return valid, actor


def compare_event_reviews(
    primary_reviews: pd.DataFrame,
    secondary_reviews: pd.DataFrame,
    packet: pd.DataFrame,
    actors: pd.DataFrame,
    *,
    rubric_version: str,
    compared_at_utc: datetime,
) -> pd.DataFrame:
    """Compare two complete human review streams without releasing labels."""

    comparison_time = _require_utc(compared_at_utc, field_name="compared_at_utc")
    valid_actors = _validate_actors(actors)
    cards = _normalize_evidence_cards(packet)
    primary, primary_actor = _validate_review_stream(
        primary_reviews,
        cards,
        valid_actors,
        role="primary",
        rubric_version=rubric_version,
    )
    secondary, secondary_actor = _validate_review_stream(
        secondary_reviews,
        cards,
        valid_actors,
        role="secondary",
        rubric_version=rubric_version,
    )
    if _actor_identity_key(primary_actor) == _actor_identity_key(secondary_actor):
        raise ValueError("primary and secondary reviews require distinct humans")
    if set(primary["event_review_id"].astype(str)) & set(
        secondary["event_review_id"].astype(str)
    ):
        raise ValueError("primary and secondary event review IDs must be disjoint")
    if set(primary["review_session_id"].astype(str)) & set(
        secondary["review_session_id"].astype(str)
    ):
        raise ValueError("primary and secondary review sessions must be distinct")
    latest_review = max(
        pd.to_datetime(
            pd.concat([primary["reviewed_at_utc"], secondary["reviewed_at_utc"]]),
            utc=True,
        )
    )
    if pd.Timestamp(comparison_time) < latest_review:
        raise ValueError("compared_at_utc cannot precede a source review")

    key_fields = ["packet_id", "packet_version", "packet_event_id"]
    primary_by_key = primary.set_index(key_fields, drop=False)
    secondary_by_key = secondary.set_index(key_fields, drop=False)
    records: list[dict[str, object]] = []
    for _, card in cards.iterrows():
        key = (
            str(card["packet_id"]),
            int(card["packet_version"]),
            str(card["packet_event_id"]),
        )
        first = primary_by_key.loc[key]
        second = secondary_by_key.loc[key]
        disagreement_fields = sorted(
            name
            for name in EVENT_REVIEW_DECISION_FIELDS
            if _comparison_value(first[name]) != _comparison_value(second[name])
        )
        labels_agree = first["proposed_label_code"] == second["proposed_label_code"]
        if labels_agree and not disagreement_fields:
            status = EventReviewComparisonStatus.FULL_AGREEMENT
        elif labels_agree:
            status = EventReviewComparisonStatus.LABEL_AGREEMENT_FACT_DISAGREEMENT
        else:
            status = EventReviewComparisonStatus.LABEL_DISAGREEMENT
        identity = {
            "packet_id": key[0],
            "packet_version": key[1],
            "packet_event_id": key[2],
            "packet_payload_sha256": str(card["packet_payload_sha256"]),
            "primary_review_id": str(first["event_review_id"]),
            "secondary_review_id": str(second["event_review_id"]),
            "compared_at_utc": comparison_time.isoformat(),
        }
        digest = _sha256_bytes(_canonical_json_bytes(identity))
        records.append(
            {
                "comparison_id": f"event-review-comparison:sha256:{digest}",
                "packet_id": key[0],
                "packet_version": key[1],
                "packet_event_id": key[2],
                "packet_payload_sha256": str(card["packet_payload_sha256"]),
                "rubric_version": rubric_version,
                "primary_review_id": str(first["event_review_id"]),
                "secondary_review_id": str(second["event_review_id"]),
                "primary_reviewer_id": str(first["reviewer_id"]),
                "secondary_reviewer_id": str(second["reviewer_id"]),
                "primary_label_code": str(first["proposed_label_code"]),
                "secondary_label_code": str(second["proposed_label_code"]),
                "label_agreement": bool(labels_agree),
                "disagreement_fields": (
                    "|".join(disagreement_fields) if disagreement_fields else None
                ),
                "comparison_status": status.value,
                "human_adjudication_required": True,
                "compared_at_utc": comparison_time,
                "notes": (
                    "Pre-adjudication comparison only; agreement cannot release a "
                    "validation label."
                ),
            }
        )
    frame = pd.DataFrame.from_records(
        records,
        columns=list(EventReviewComparisonRecord.model_fields),
    )
    return _validate_frame(
        frame,
        EventReviewComparisonRecord,
        role="event review comparison",
    )


def _comparison_value(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return ("number", int(value))
    if isinstance(value, Real):
        numeric = float(value)
        if numeric.is_integer():
            return ("number", int(numeric))
        return ("number", numeric)
    return str(value)


def summarize_event_review_agreement(
    primary_reviews: pd.DataFrame,
    secondary_reviews: pd.DataFrame,
) -> dict[str, object]:
    """Calculate pre-adjudication exact agreement, Cohen kappa, and Gwet AC1."""

    key_fields = ["packet_id", "packet_version", "packet_event_id"]
    for role, reviews in (
        ("primary", primary_reviews),
        ("secondary", secondary_reviews),
    ):
        if reviews.duplicated(key_fields).any():
            raise ValueError(
                "agreement statistics reject duplicate composite review keys "
                f"in the {role} stream"
            )
    first = primary_reviews.set_index(key_fields)
    second = secondary_reviews.set_index(key_fields)
    if set(first.index) != set(second.index):
        raise ValueError("agreement statistics require identical review rosters")
    second = second.loc[first.index]
    labels_a = first["proposed_label_code"].astype(str).tolist()
    labels_b = second["proposed_label_code"].astype(str).tolist()
    n = len(labels_a)
    if n == 0:
        return {
            "record_count": 0,
            "label_order": list(LABEL_ORDER),
            "exact_label_agreement_count": 0,
            "exact_label_agreement_fraction": None,
            "cohen_kappa": None,
            "gwet_ac1": None,
            "confusion_matrix_axes": {
                "rows": "primary_reviews.proposed_label_code",
                "columns": "secondary_reviews.proposed_label_code",
            },
            "confusion_matrix": {
                label: {other: 0 for other in LABEL_ORDER} for label in LABEL_ORDER
            },
            "field_agreement": {},
        }

    matrix = {label: {other: 0 for other in LABEL_ORDER} for label in LABEL_ORDER}
    for label_a, label_b in zip(labels_a, labels_b, strict=True):
        matrix[label_a][label_b] += 1
    observed = sum(matrix[label][label] for label in LABEL_ORDER) / n
    row_totals = {label: sum(matrix[label].values()) for label in LABEL_ORDER}
    column_totals = {
        label: sum(matrix[row][label] for row in LABEL_ORDER) for label in LABEL_ORDER
    }
    expected_kappa = sum(
        row_totals[label] * column_totals[label] for label in LABEL_ORDER
    ) / (n * n)
    kappa_denominator = 1.0 - expected_kappa
    kappa = (
        None
        if abs(kappa_denominator) < 1e-15
        else (observed - expected_kappa) / kappa_denominator
    )

    category_count = len(LABEL_ORDER)
    marginal = {
        label: (row_totals[label] + column_totals[label]) / (2.0 * n)
        for label in LABEL_ORDER
    }
    expected_ac1 = sum(value * (1.0 - value) for value in marginal.values()) / (
        category_count - 1
    )
    ac1_denominator = 1.0 - expected_ac1
    ac1 = (
        None
        if abs(ac1_denominator) < 1e-15
        else (observed - expected_ac1) / ac1_denominator
    )

    field_agreement: dict[str, dict[str, object]] = {}
    for field_name in EVENT_REVIEW_DECISION_FIELDS:
        matches = sum(
            _comparison_value(a) == _comparison_value(b)
            for a, b in zip(first[field_name], second[field_name], strict=True)
        )
        field_agreement[field_name] = {
            "agreement_count": matches,
            "agreement_fraction": matches / n,
        }
    return {
        "record_count": n,
        "label_order": list(LABEL_ORDER),
        "exact_label_agreement_count": int(round(observed * n)),
        "exact_label_agreement_fraction": observed,
        "cohen_kappa": kappa,
        "gwet_ac1": ac1,
        "confusion_matrix_axes": {
            "rows": "primary_reviews.proposed_label_code",
            "columns": "secondary_reviews.proposed_label_code",
        },
        "confusion_matrix": matrix,
        "field_agreement": field_agreement,
        "calculation_boundary": "pre_adjudication_reviews_only",
    }


def _blank_adjudication_template(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    linkage = {
        "comparison_id",
        "packet_id",
        "packet_version",
        "packet_event_id",
        "packet_payload_sha256",
        "primary_review_id",
        "secondary_review_id",
        "primary_reviewer_id",
        "secondary_reviewer_id",
        "rubric_version",
    }
    for _, compared in comparison.iterrows():
        row = {name: "" for name in EventAdjudicationRecord.model_fields}
        for name in linkage:
            row[name] = compared[name]
        rows.append(row)
    return pd.DataFrame(rows, columns=list(EventAdjudicationRecord.model_fields))


def write_event_review_comparison_bundle(
    packet_dir: str | Path,
    primary_reviews_path: str | Path,
    secondary_reviews_path: str | Path,
    actor_registry_path: str | Path,
    output_dir: str | Path,
    *,
    compared_at_utc: datetime,
    expected_packet_manifest_sha256: str,
) -> dict[str, object]:
    """Atomically freeze two human review streams and their agreement results."""

    packet_root = Path(packet_dir)
    primary_path = Path(primary_reviews_path)
    secondary_path = Path(secondary_reviews_path)
    actor_path = Path(actor_registry_path)
    paths = {filename: packet_root / filename for filename in PACKET_FILENAMES} | {
        "primary_input": primary_path,
        "secondary_input": secondary_path,
        "actors_input": actor_path,
    }
    comparison_time = _require_utc(compared_at_utc, field_name="compared_at_utc")

    with _atomic_output_directory(output_dir) as (work_dir, publish_dir):
        contents, _ = _snapshot_files(paths, work_dir / "snapshots")
        actors_raw = _read_table_bytes(
            contents["actors_input"],
            actor_path.suffix,
            contract=CurationActorRecord,
            role="curation actors",
        )
        actors = _validate_actors(actors_raw)
        canonical_actor_content = _table_output_bytes(actors)
        packet_contents = {name: contents[name] for name in PACKET_FILENAMES}
        packet_manifest, cards, rubric_version = _verify_packet_bundle(
            packet_contents,
            expected_manifest_sha256=expected_packet_manifest_sha256,
            actor_content=canonical_actor_content,
        )
        primary_raw = _read_table_bytes(
            contents["primary_input"],
            primary_path.suffix,
            contract=EventReviewRecord,
            role="primary reviews",
        )
        secondary_raw = _read_table_bytes(
            contents["secondary_input"],
            secondary_path.suffix,
            contract=EventReviewRecord,
            role="secondary reviews",
        )
        comparison = compare_event_reviews(
            primary_raw,
            secondary_raw,
            cards,
            actors,
            rubric_version=rubric_version,
            compared_at_utc=comparison_time,
        )
        primary, primary_actor = _validate_review_stream(
            primary_raw,
            cards,
            actors,
            role="primary",
            rubric_version=rubric_version,
        )
        secondary, secondary_actor = _validate_review_stream(
            secondary_raw,
            cards,
            actors,
            role="secondary",
            rubric_version=rubric_version,
        )
        agreement = summarize_event_review_agreement(primary, secondary)
        adjudication_template = _blank_adjudication_template(comparison)

        copied_contents = {name: contents[name] for name in PACKET_FILENAMES} | {
            "curation_actors.tsv": canonical_actor_content,
            "event_reviews_a.tsv": _table_output_bytes(primary),
            "event_reviews_b.tsv": _table_output_bytes(secondary),
        }
        derived_contents = {
            "event_review_comparison.tsv": _table_output_bytes(comparison),
            "event_review_agreement.json": _json_output_bytes(agreement),
            "adjudication_template.tsv": _table_output_bytes(adjudication_template),
        }
        all_contents = copied_contents | derived_contents
        for filename, content in all_contents.items():
            _write_bytes(publish_dir / filename, content)

        manifest: dict[str, object] = {
            "schema_version": 1,
            "status": "dual_human_event_review_complete_requires_adjudication",
            "compared_at_utc": comparison_time.isoformat(),
            "rubric_version": rubric_version,
            "record_count": len(comparison),
            "packet_roster_sha256": packet_manifest["packet_roster_sha256"],
            "primary_reviewer_id": str(primary["reviewer_id"].iloc[0]),
            "secondary_reviewer_id": str(secondary["reviewer_id"].iloc[0]),
            "primary_reviewer_identity_key_sha256": _sha256_bytes(
                _actor_identity_key(primary_actor).encode("utf-8")
            ),
            "secondary_reviewer_identity_key_sha256": _sha256_bytes(
                _actor_identity_key(secondary_actor).encode("utf-8")
            ),
            "source_packet_manifest": _file_fact(
                "event_packet_manifest.json",
                contents["event_packet_manifest.json"],
            ),
            "source_actor_registry": _content_fact(contents["actors_input"]),
            "source_primary_review": _content_fact(contents["primary_input"]),
            "source_secondary_review": _content_fact(contents["secondary_input"]),
            "files": {
                filename: _file_fact(filename, content)
                for filename, content in all_contents.items()
            },
            "agreement": agreement,
            "scientific_boundary": {
                "all_rows_require_human_adjudication": True,
                "comparison_releases_no_label": True,
                "agreement_statistics_are_pre_adjudication": True,
            },
        }
        _assert_no_readiness(manifest)
        _write_bytes(
            publish_dir / "event_review_comparison_manifest.json",
            _json_output_bytes(manifest),
        )
    return manifest


def _verify_comparison_bundle(
    contents: Mapping[str, bytes],
    *,
    expected_manifest_sha256: str,
) -> tuple[
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    str,
]:
    expected_sha = _require_sha256(
        expected_manifest_sha256,
        field_name="expected_comparison_manifest_sha256",
    )
    manifest_content = contents["event_review_comparison_manifest.json"]
    if _sha256_bytes(manifest_content) != expected_sha:
        raise ValueError(
            "event review comparison manifest SHA-256 does not match expected"
        )
    manifest = _parse_json_object(
        manifest_content,
        role="event review comparison manifest",
    )
    _assert_no_readiness(manifest)
    if manifest.get("status") != (
        "dual_human_event_review_complete_requires_adjudication"
    ):
        raise ValueError("comparison bundle is not ready for adjudication")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("comparison manifest lacks file checksums")
    for filename in COMPARISON_FILENAMES[:-1]:
        fact = files.get(filename)
        if not isinstance(fact, dict):
            raise ValueError(f"comparison manifest lacks file: {filename}")
        if fact.get("filename") != filename:
            raise ValueError(f"comparison manifest has noncanonical file: {filename}")
        if fact.get("sha256") != _sha256_bytes(contents[filename]):
            raise ValueError(f"comparison bundle checksum mismatch: {filename}")
        if fact.get("byte_size") != len(contents[filename]):
            raise ValueError(f"comparison bundle byte size mismatch: {filename}")

    packet_manifest_sha = _sha256_bytes(contents["event_packet_manifest.json"])
    source_packet = manifest.get("source_packet_manifest")
    if not isinstance(source_packet, dict) or source_packet.get("sha256") != (
        packet_manifest_sha
    ):
        raise ValueError("comparison manifest is not bound to its packet manifest")
    _, cards, rubric_version = _verify_packet_bundle(
        {name: contents[name] for name in PACKET_FILENAMES},
        expected_manifest_sha256=packet_manifest_sha,
        actor_content=contents["curation_actors.tsv"],
    )
    actors = _validate_actors(
        _read_table_bytes(
            contents["curation_actors.tsv"],
            ".tsv",
            contract=CurationActorRecord,
            role="curation actors",
        )
    )
    primary = _read_table_bytes(
        contents["event_reviews_a.tsv"],
        ".tsv",
        contract=EventReviewRecord,
        role="primary reviews",
    )
    secondary = _read_table_bytes(
        contents["event_reviews_b.tsv"],
        ".tsv",
        contract=EventReviewRecord,
        role="secondary reviews",
    )
    compared_at = _require_utc(
        pd.Timestamp(manifest["compared_at_utc"]).to_pydatetime(),
        field_name="comparison manifest compared_at_utc",
    )
    expected_comparison = compare_event_reviews(
        primary,
        secondary,
        cards,
        actors,
        rubric_version=rubric_version,
        compared_at_utc=compared_at,
    )
    observed_comparison = _validate_frame(
        _read_table_bytes(
            contents["event_review_comparison.tsv"],
            ".tsv",
            contract=EventReviewComparisonRecord,
            role="review comparison",
        ),
        EventReviewComparisonRecord,
        role="stored event review comparison",
    )
    if not _frames_equal(expected_comparison, observed_comparison):
        raise ValueError("stored event comparison is not the deterministic derivation")
    valid_primary, _ = _validate_review_stream(
        primary,
        cards,
        actors,
        role="primary",
        rubric_version=rubric_version,
    )
    valid_secondary, _ = _validate_review_stream(
        secondary,
        cards,
        actors,
        role="secondary",
        rubric_version=rubric_version,
    )
    expected_agreement = summarize_event_review_agreement(
        valid_primary,
        valid_secondary,
    )
    observed_agreement = _parse_json_object(
        contents["event_review_agreement.json"],
        role="event review agreement",
    )
    if observed_agreement != expected_agreement or manifest.get("agreement") != (
        expected_agreement
    ):
        raise ValueError("agreement statistics are not the deterministic derivation")
    return (
        manifest,
        cards,
        valid_primary,
        valid_secondary,
        observed_comparison,
        rubric_version,
    )


def _frames_equal(first: pd.DataFrame, second: pd.DataFrame) -> bool:
    if first.columns.tolist() != second.columns.tolist():
        return False
    return first.fillna("").astype(str).equals(second.fillna("").astype(str))


def _criterion_bool(value: object) -> bool | None:
    token = str(value)
    if token == CriterionAssessment.YES.value:
        return True
    if token == CriterionAssessment.NO.value:
        return False
    return None


def _event_adjudication_status(method: object) -> str:
    if str(method) == AdjudicationResolutionMethod.TWO_REVIEWER_CONSENSUS.value:
        return "dual_curator_consensus"
    return "adjudicated"


def _validation_event_from_adjudication(
    card: Mapping[str, object],
    adjudication: Mapping[str, object],
) -> dict[str, object]:
    relation = str(adjudication["final_outcome_relation"])
    if relation == ValidationOutcomeRelation.CONCORDANT.value:
        phenotype_reproduced: bool | None = True
        opposite_reproduced: bool | None = False
    elif relation == ValidationOutcomeRelation.NO_EFFECT.value:
        phenotype_reproduced = False
        opposite_reproduced = False
    elif relation == ValidationOutcomeRelation.OPPOSITE.value:
        phenotype_reproduced = False
        opposite_reproduced = True
    else:
        phenotype_reproduced = None
        opposite_reproduced = None

    identity = {
        "adjudication_id": str(adjudication["adjudication_id"]),
        "packet_id": str(adjudication["packet_id"]),
        "packet_version": int(adjudication["packet_version"]),
        "packet_event_id": str(adjudication["packet_event_id"]),
    }
    digest = _sha256_bytes(_canonical_json_bytes(identity))
    modality = card.get("reported_perturbation_modality")
    if modality is None or pd.isna(modality) or not str(modality).strip():
        modality = PerturbationModality.OTHER.value
    rescue = _criterion_bool(adjudication["final_rescue_or_causal_reversal"])
    notes_parts = [
        f"gold_adjudication_id={adjudication['adjudication_id']}",
        f"primary_event_review_id={adjudication['primary_review_id']}",
        f"secondary_event_review_id={adjudication['secondary_review_id']}",
        f"context_match={adjudication['final_context_match']}",
        f"confidence={adjudication['final_confidence']}",
        f"reason_codes={adjudication['adjudication_reason_codes']}",
    ]
    if adjudication.get("notes") is not None and str(adjudication["notes"]).strip():
        notes_parts.append(str(adjudication["notes"]).strip())
    return {
        "event_id": f"validation-event:sha256:{digest}",
        "study_id": str(card["study_id"]),
        "screen_id": str(card["screen_id"]),
        "contrast_id": str(card["contrast_id"]),
        "gene_symbol": str(card["gene_symbol"]),
        "drug_name": str(card.get("validation_drug") or card["screen_drug"]),
        "cell_line": str(
            card.get("validation_cell_model") or card["screen_cell_model"]
        ),
        "perturbation_modality": str(modality),
        "phenotype_direction": str(card["screen_predicted_direction"]),
        "label_code": str(adjudication["final_label_code"]),
        "testing_status": str(adjudication["final_testing_status"]),
        "perturbation_confirmed": _criterion_bool(
            adjudication["final_perturbation_confirmed"]
        ),
        "independent_reagent_count": adjudication["final_independent_reagent_count"],
        "orthogonal_perturbation": _criterion_bool(
            adjudication["final_orthogonal_perturbation"]
        ),
        "appropriate_control": _criterion_bool(
            adjudication["final_appropriate_control"]
        ),
        "assay_adequate": _criterion_bool(adjudication["final_assay_adequate"]),
        "null_result_evaluable": _criterion_bool(
            adjudication["final_null_result_evaluable"]
        ),
        "phenotype_reproduced": phenotype_reproduced,
        "opposite_direction_reproduced": opposite_reproduced,
        "rescue_performed": rescue,
        "causal_reversal_performed": None,
        "effect_size": None,
        "effect_metric": None,
        "p_value": None,
        "validation_treatment_dose": None,
        "validation_treatment_unit": None,
        "validation_exposure_days": None,
        "validation_assay_endpoint": card.get("reported_assay_endpoint"),
        "validation_control_type": None,
        "validation_biological_replicate_count": None,
        "validation_guide_source": None,
        "protein_confirmed": None,
        "second_model_tested": None,
        "source_url": str(card["source_url"]),
        "source_locator": str(adjudication["final_source_locator"]),
        "curator": str(adjudication["adjudicator_id"]),
        "adjudication_status": _event_adjudication_status(
            adjudication["resolution_method"]
        ),
        "notes": "; ".join(notes_parts),
    }


def _require_primary_event_context(
    card: Mapping[str, object],
    *,
    label_code: object,
    context_match: object,
) -> None:
    """Prevent primary labels from inheriting or inventing validation context."""

    if str(label_code) not in PRIMARY_EVENT_LABELS:
        return
    missing = [
        name
        for name in ("validation_drug", "validation_cell_model")
        if card.get(name) is None
        or pd.isna(card.get(name))
        or not str(card.get(name)).strip()
    ]
    if missing:
        raise ValueError(
            "primary validation labels require explicit evidence-card context: "
            f"{missing}"
        )
    if str(context_match) == ValidationContextMatch.EXACT.value:
        mismatches = [
            (validation_name, screen_name)
            for validation_name, screen_name in (
                ("validation_drug", "screen_drug"),
                ("validation_cell_model", "screen_cell_model"),
            )
            if str(card[validation_name]).strip().casefold()
            != str(card[screen_name]).strip().casefold()
        ]
        if mismatches:
            details = [f"{validation}!={screen}" for validation, screen in mismatches]
            raise ValueError(
                "primary label with exact context requires normalized equality: "
                f"{details}"
            )
    modality = card.get("reported_perturbation_modality")
    if (
        modality is None
        or pd.isna(modality)
        or not str(modality).strip()
        or str(modality) == PerturbationModality.OTHER.value
    ):
        raise ValueError(
            "primary validation labels require an explicit non-other "
            "perturbation modality"
        )


def finalize_event_adjudication(
    packet: pd.DataFrame,
    comparison: pd.DataFrame,
    primary_reviews: pd.DataFrame,
    secondary_reviews: pd.DataFrame,
    adjudications: pd.DataFrame,
    actors: pd.DataFrame,
    *,
    rubric_version: str,
    released_at_utc: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate complete named-human adjudication and derive validation events."""

    release_time = _require_utc(released_at_utc, field_name="released_at_utc")
    cards = _normalize_evidence_cards(packet)
    valid_actors = _validate_actors(actors)
    primary, _ = _validate_review_stream(
        primary_reviews,
        cards,
        valid_actors,
        role="primary",
        rubric_version=rubric_version,
    )
    secondary, _ = _validate_review_stream(
        secondary_reviews,
        cards,
        valid_actors,
        role="secondary",
        rubric_version=rubric_version,
    )
    valid_comparison = _validate_frame(
        comparison,
        EventReviewComparisonRecord,
        role="event review comparison",
    )
    comparison_keys = _review_key(valid_comparison)
    card_keys = _review_key(cards)
    if (
        comparison_keys.duplicated().any()
        or len(valid_comparison) != len(cards)
        or set(comparison_keys) != set(card_keys)
    ):
        raise ValueError("comparison does not exactly cover the frozen packet")
    comparison_times = pd.to_datetime(
        valid_comparison["compared_at_utc"],
        utc=True,
    )
    if comparison_times.nunique() != 1:
        raise ValueError("comparison requires one frozen compared_at_utc timestamp")
    recomputed_comparison = compare_event_reviews(
        primary,
        secondary,
        cards,
        valid_actors,
        rubric_version=rubric_version,
        compared_at_utc=comparison_times.iloc[0].to_pydatetime(),
    )
    if not _frames_equal(recomputed_comparison, valid_comparison):
        raise ValueError(
            "comparison is not the deterministic derivation of the two reviews"
        )
    valid_adjudications = _validate_frame(
        adjudications,
        EventAdjudicationRecord,
        role="event adjudications",
    )
    if valid_adjudications["comparison_id"].duplicated().any():
        raise ValueError("adjudication requires exactly one row per comparison")
    expected_ids = set(valid_comparison["comparison_id"].astype(str))
    observed_ids = set(valid_adjudications["comparison_id"].astype(str))
    if observed_ids != expected_ids or len(valid_adjudications) != len(
        valid_comparison
    ):
        raise ValueError("adjudications do not exactly cover the frozen comparison")

    comparison_by_id = valid_comparison.set_index("comparison_id", drop=False)
    cards_by_key = cards.set_index(
        ["packet_id", "packet_version", "packet_event_id"], drop=False
    )
    review_times = pd.concat(
        [primary["reviewed_at_utc"], secondary["reviewed_at_utc"]],
        ignore_index=True,
    )
    latest_review = max(pd.to_datetime(review_times, utc=True))
    events: list[dict[str, object]] = []
    for _, decision in valid_adjudications.iterrows():
        compared = comparison_by_id.loc[str(decision["comparison_id"])]
        linkage_fields = (
            "packet_id",
            "packet_version",
            "packet_event_id",
            "packet_payload_sha256",
            "primary_review_id",
            "secondary_review_id",
            "primary_reviewer_id",
            "secondary_reviewer_id",
            "rubric_version",
        )
        mismatches = [
            name
            for name in linkage_fields
            if _comparison_value(decision[name]) != _comparison_value(compared[name])
        ]
        if mismatches:
            raise ValueError(
                f"adjudication differs from comparison linkage fields: {mismatches}"
            )
        adjudicated_at = pd.Timestamp(decision["adjudicated_at_utc"])
        if adjudicated_at < pd.Timestamp(compared["compared_at_utc"]):
            raise ValueError("adjudication cannot precede its comparison")
        if adjudicated_at < latest_review:
            raise ValueError("adjudication cannot precede a source review")
        if pd.Timestamp(release_time) < adjudicated_at:
            raise ValueError("released_at_utc cannot precede adjudication")

        adjudicator = _require_actor(
            decision["adjudicator_id"],
            valid_actors,
            role="adjudicator",
            require_human=True,
        )
        reviewer_actors = [
            _require_actor(
                decision[name],
                valid_actors,
                role=name,
                require_human=True,
            )
            for name in ("primary_reviewer_id", "secondary_reviewer_id")
        ]
        co_signer_id = decision.get("co_signer_id")
        if co_signer_id is not None and not pd.isna(co_signer_id):
            _require_actor(
                co_signer_id,
                valid_actors,
                role="adjudication co-signer",
                require_human=True,
            )
        method = str(decision["resolution_method"])
        if (
            method == AdjudicationResolutionMethod.INSUFFICIENT_EVIDENCE.value
            and str(decision["final_label_code"]) in PRIMARY_EVENT_LABELS
        ):
            raise ValueError(
                "insufficient-evidence adjudication cannot release a primary label"
            )
        if method != AdjudicationResolutionMethod.TWO_REVIEWER_CONSENSUS.value:
            adjudicator_key = _actor_identity_key(adjudicator)
            if adjudicator_key in {
                _actor_identity_key(actor) for actor in reviewer_actors
            }:
                raise ValueError(
                    "non-consensus adjudicator must be an independent human"
                )

        card_key = (
            str(decision["packet_id"]),
            int(decision["packet_version"]),
            str(decision["packet_event_id"]),
        )
        card = cards_by_key.loc[card_key]
        _require_primary_event_context(
            card,
            label_code=decision["final_label_code"],
            context_match=decision["final_context_match"],
        )
        event = _validation_event_from_adjudication(card, decision)
        try:
            parsed_event = ValidationEventRecord.model_validate(event).model_dump(
                mode="json"
            )
        except Exception as exc:
            raise ValueError(
                "adjudication cannot produce a valid validation event for "
                f"{card_key}: {exc}"
            ) from exc
        events.append(parsed_event)

    validation_events = pd.DataFrame.from_records(
        events,
        columns=list(ValidationEventRecord.model_fields),
    )
    if "benchmark_ready" in validation_events.columns:
        raise RuntimeError("final adjudication cannot emit benchmark_ready")
    return valid_adjudications, validation_events


def write_final_adjudication_bundle(
    comparison_bundle_dir: str | Path,
    adjudications_path: str | Path,
    output_dir: str | Path,
    *,
    released_at_utc: datetime,
    expected_comparison_manifest_sha256: str,
) -> dict[str, object]:
    """Atomically publish named adjudications and derived validation events."""

    comparison_root = Path(comparison_bundle_dir)
    adjudication_path = Path(adjudications_path)
    paths = {
        filename: comparison_root / filename for filename in COMPARISON_FILENAMES
    } | {"adjudications_input": adjudication_path}
    release_time = _require_utc(released_at_utc, field_name="released_at_utc")

    with _atomic_output_directory(output_dir) as (work_dir, publish_dir):
        contents, _ = _snapshot_files(paths, work_dir / "snapshots")
        comparison_contents = {name: contents[name] for name in COMPARISON_FILENAMES}
        (
            comparison_manifest,
            cards,
            primary,
            secondary,
            comparison,
            rubric_version,
        ) = _verify_comparison_bundle(
            comparison_contents,
            expected_manifest_sha256=expected_comparison_manifest_sha256,
        )
        actors = _read_table_bytes(
            contents["curation_actors.tsv"],
            ".tsv",
            contract=CurationActorRecord,
            role="curation actors",
        )
        adjudications_raw = _read_table_bytes(
            contents["adjudications_input"],
            adjudication_path.suffix,
            contract=EventAdjudicationRecord,
            role="event adjudications",
        )
        adjudications, validation_events = finalize_event_adjudication(
            cards,
            comparison,
            primary,
            secondary,
            adjudications_raw,
            actors,
            rubric_version=rubric_version,
            released_at_utc=release_time,
        )

        copied_contents = {name: contents[name] for name in COMPARISON_FILENAMES} | {
            "event_adjudications.tsv": _table_output_bytes(adjudications)
        }
        derived_contents = {
            "validation_events.tsv": _table_output_bytes(validation_events),
        }
        all_contents = copied_contents | derived_contents
        for filename, content in all_contents.items():
            _write_bytes(publish_dir / filename, content)

        label_counts = {
            str(key): int(value)
            for key, value in validation_events["label_code"]
            .value_counts()
            .sort_index()
            .items()
        }
        manifest: dict[str, object] = {
            "schema_version": 1,
            "status": "human_event_adjudication_complete",
            "released_at_utc": release_time.isoformat(),
            "rubric_version": rubric_version,
            "record_count": len(adjudications),
            "validation_event_count": len(validation_events),
            "label_counts": label_counts,
            "source_comparison_manifest": _file_fact(
                "event_review_comparison_manifest.json",
                contents["event_review_comparison_manifest.json"],
            ),
            "source_packet_manifest_sha256": comparison_manifest[
                "source_packet_manifest"
            ]["sha256"],
            "source_adjudications": _content_fact(contents["adjudications_input"]),
            "files": {
                filename: _file_fact(filename, content)
                for filename, content in all_contents.items()
            },
            "scientific_boundary": {
                "released_records_are_validation_events_only": True,
                "screen_readiness_requires_separate_registry_gates": True,
                "no_automatic_benchmark_promotion": True,
            },
        }
        _assert_no_readiness(manifest)
        _write_bytes(
            publish_dir / "final_adjudication_manifest.json",
            _json_output_bytes(manifest),
        )
    return manifest


__all__ = [
    "EVENT_REVIEW_DECISION_FIELDS",
    "compare_event_reviews",
    "finalize_event_adjudication",
    "prepare_event_review_packet",
    "summarize_event_review_agreement",
    "write_event_review_comparison_bundle",
    "write_event_review_packet",
    "write_final_adjudication_bundle",
]
