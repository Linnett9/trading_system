from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.research.ml.selector_dataset_lineage import (
    logical_manifest_checksum,
    verify_dataset_lineage_manifest,
)
from core.research.ml.stock_level.selector_dataset import (
    SELECTOR_DATASET_CONTRACT_VERSION,
    SELECTOR_DATASET_MANIFEST_VERSION,
)
from core.research.ml.stock_level.selector_lineage import (
    CURRENT_ECONOMIC_TARGET_ID,
    CURRENT_TARGET_PROVENANCE_VERSION,
    SELECTOR_ROW_ID_CONTRACT_VERSION,
    TARGET_COLUMNS,
    TARGET_TIMESTAMP_COLUMNS,
)
from core.research.ml.stock_level.stock_alpha_news_feature_store import (
    FEATURE_FIELDS,
    FEATURE_SCHEMA_CONTRACT,
    FEATURE_STORE_CONTRACT,
)
from core.research.ml.stock_level.stock_alpha_news_pit_policy import (
    STRICT_COLLECTED_AT,
)


REQUEST_CONTRACT = "paired_selector_dataset_request.v1"
PAIR_CONTRACT = "matched_selector_dataset_pair.v1"
PAIR_MANIFEST_SCHEMA = "matched_selector_dataset_pair_manifest.v1"
MEMBER_MANIFEST_SCHEMA = "matched_selector_dataset_member_manifest.v1"
EVIDENCE_SCHEMA = "matched_selector_dataset_evidence.v1"
JOIN_KEY_CONTRACT = "asset_id+decision_session_date+lookback_days.v1"
MISSING_NEWS_POLICY = "PRESERVE_SELECTOR_ROW_WITH_EXPLICIT_MISSING_NEWS"
UNEXPECTED_NEWS_POLICY = "ALLOW_CERTIFIED_SUPERSET_REPORT_BOUNDED_FAIL_ON_CONFLICT"
NEWS_OUTPUT_FIELDS = tuple(
    field for field in FEATURE_FIELDS
    if field not in {
        "asset_id", "symbol", "decision_session_date",
        "decision_timestamp", "lookback_days",
    }
)
NEWS_MEMBER_FIELDS = ("news_lookback_days", *NEWS_OUTPUT_FIELDS)
REASON_CODES = (
    "NO_ELIGIBLE_NEWS", "SELECTOR_PARENT_INVALID", "NEWS_PARENT_INVALID",
    "DUPLICATE_SELECTOR_ROW", "DUPLICATE_NEWS_JOIN_KEY",
    "SELECTOR_NEWS_ASSET_MISMATCH", "SELECTOR_NEWS_SYMBOL_MISMATCH",
    "DECISION_DATE_MISMATCH", "DECISION_CUTOFF_VIOLATION",
    "TARGET_INVARIANT_MISMATCH", "TARGET_TIMESTAMP_MISMATCH",
    "ELIGIBILITY_MISMATCH", "MEMBER_POPULATION_MISMATCH",
    "UNEXPECTED_NEWS_ROW",
)
EXAMPLE_LIMIT = 10


@dataclass(frozen=True)
class PairedSelectorDatasetRequest:
    selector_dataset_root: str
    selector_dataset_identity: str
    selector_logical_manifest_checksum: str
    selector_artifact_checksums: tuple[tuple[str, str], ...]
    news_feature_store_root: str
    news_store_identity: str
    news_logical_manifest_checksum: str
    news_schema_checksum: str
    lookback_days: int
    output_root: str
    request_contract_version: str = REQUEST_CONTRACT
    join_key_contract: str = JOIN_KEY_CONTRACT
    missing_news_policy: str = MISSING_NEWS_POLICY
    unexpected_news_row_policy: str = UNEXPECTED_NEWS_POLICY
    implementation_identity: str = (
        "core.research.ml.stock_level.paired_selector_datasets:"
        "publish_paired_selector_datasets"
    )

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["selector_artifact_checksums"] = [
            {"path": path, "checksum": checksum}
            for path, checksum in self.selector_artifact_checksums
        ]
        return value

    def identity_payload(self) -> dict[str, Any]:
        return {
            key: value for key, value in self.payload().items()
            if key not in {
                "selector_dataset_root",
                "news_feature_store_root",
                "output_root",
            }
        }

    @property
    def checksum(self) -> str:
        return canonical_hash(self.identity_payload())


@dataclass(frozen=True)
class PairedSelectorDatasetResult:
    pair_root: Path
    pair_manifest: Path
    pair_identity: str
    price_only_identity: str
    price_plus_news_identity: str
    row_count: int
    covered_news_row_count: int
    missing_news_row_count: int
    reused: bool

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["pair_root"] = str(self.pair_root)
        value["pair_manifest"] = str(self.pair_manifest)
        return value


def publish_paired_selector_datasets(
    *,
    selector_dataset_root: Path,
    news_feature_store_root: Path,
    output_root: Path,
    lookback_days: int,
    reuse: bool = False,
    source_commit: str | None = None,
) -> PairedSelectorDatasetResult:
    selector = verify_selector_parent(Path(selector_dataset_root))
    news = verify_news_parent(Path(news_feature_store_root), int(lookback_days))
    request = build_request(
        selector=selector, news=news, output_root=Path(output_root),
        lookback_days=int(lookback_days),
    )
    rows, population = read_selector_population(selector)
    news_index, unexpected_candidates = read_news_rows(
        news, lookback_days=int(lookback_days)
    )
    price_rows, augmented_rows, evidence = align_members(
        rows, news_index, unexpected_candidates,
        lookback_days=int(lookback_days),
    )
    schemas = {
        "price_only": schema_identity(price_rows),
        "price_plus_news": schema_identity(augmented_rows),
    }
    identity_payload = {
        "pair_contract_version": PAIR_CONTRACT,
        "request_checksum": request.checksum,
        "selector_parent": selector["identity"],
        "news_parent": news["identity"],
        "lookback_days": int(lookback_days),
        "join_key_contract": request.join_key_contract,
        "missing_news_policy": request.missing_news_policy,
        "unexpected_news_row_policy": request.unexpected_news_row_policy,
        "ordered_economic_row_population_checksum": population["checksum"],
        "economic_target_id": selector["manifest"]["economic_target_id"],
        "target_provenance_contract_version": selector["manifest"][
            "target_provenance_contract_version"
        ],
        "member_feature_schema_identities": schemas,
    }
    pair_identity = canonical_hash(identity_payload)
    member_identities = {
        name: canonical_hash({
            "pair_identity": pair_identity, "member": name,
            "schema_identity": schemas[name],
            "population_checksum": population["checksum"],
        })
        for name in schemas
    }
    pair_root = Path(output_root).resolve() / pair_identity
    if pair_root.exists():
        if not reuse:
            raise FileExistsError(
                f"Immutable paired selector dataset already exists: {pair_root}"
            )
        validated = validate_paired_selector_publication(
            pair_root, expected_request_checksum=request.checksum,
            expected_pair_identity=pair_identity,
        )
        return _result(validated, pair_root=pair_root, reused=True)

    temporary = pair_root.with_name(f".{pair_identity}.{uuid.uuid4().hex}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Paired publication temporary path exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        _write_json(temporary / "request.json", {
            **request.payload(), "request_checksum": request.checksum,
        })
        evidence_payload = _evidence_payload(evidence, population)
        _write_json(temporary / "exclusion_report.json", evidence_payload)
        member_manifests = {}
        for name, member_rows in (
            ("price_only", price_rows),
            ("price_plus_news", augmented_rows),
        ):
            member_manifests[name] = _publish_member(
                temporary / name, rows=member_rows, name=name,
                identity=member_identities[name], schema_identity=schemas[name],
                population=population, pair_identity=pair_identity,
            )
        pair_manifest = {
            "pair_manifest_schema_version": PAIR_MANIFEST_SCHEMA,
            "pair_contract_version": PAIR_CONTRACT,
            "pair_identity": pair_identity,
            "request_checksum": request.checksum,
            "request_path": "request.json",
            "selector_parent": selector["identity"],
            "news_parent": news["identity"],
            "selected_lookback_days": int(lookback_days),
            "join_key_contract": request.join_key_contract,
            "missing_news_policy": request.missing_news_policy,
            "unexpected_news_row_policy": request.unexpected_news_row_policy,
            "economic_target_id": selector["manifest"]["economic_target_id"],
            "target_provenance_contract_version": selector["manifest"][
                "target_provenance_contract_version"
            ],
            "row_identity_contract": selector["manifest"]["row_id_contract"],
            "ordered_economic_row_population_checksum": population["checksum"],
            "canonical_row_count": population["row_count"],
            "asset_count": population["asset_count"],
            "decision_date_count": population["decision_date_count"],
            "members": member_manifests,
            "invariant_validation": {
                "status": "VERIFIED",
                "economic_population_equal": True,
                "targets_equal": True,
                "target_timestamps_equal": True,
                "eligibility_equal": True,
                "non_news_features_equal": True,
            },
            "missing_news_row_count": evidence["counts"]["NO_ELIGIBLE_NEWS"],
            "covered_news_row_count": len(rows) - evidence["counts"]["NO_ELIGIBLE_NEWS"],
            "exclusion_report_identity": evidence_payload["report_identity"],
            "exclusion_report_path": "exclusion_report.json",
            "source_commit": source_commit or _git_commit(),
            "publication_status": "complete",
            "validation_status": "VERIFIED",
        }
        pair_manifest["logical_checksum"] = _logical_checksum(pair_manifest)
        _write_json(temporary / "pair_manifest.json", pair_manifest)
        validate_paired_selector_publication(
            temporary, expected_request_checksum=request.checksum,
            expected_pair_identity=pair_identity,
        )
        pair_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, pair_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    validated = validate_paired_selector_publication(pair_root)
    return _result(validated, pair_root=pair_root, reused=False)


def build_request(
    *, selector: Mapping[str, Any], news: Mapping[str, Any],
    output_root: Path, lookback_days: int,
) -> PairedSelectorDatasetRequest:
    return PairedSelectorDatasetRequest(
        selector_dataset_root=str(Path(selector["root"]).resolve()),
        selector_dataset_identity=str(selector["manifest"]["dataset_id"]),
        selector_logical_manifest_checksum=str(
            selector["manifest"]["logical_checksum"]
        ),
        selector_artifact_checksums=tuple(sorted(
            (str(path), str(checksum))
            for path, checksum in selector["manifest"]["checksums"].items()
        )),
        news_feature_store_root=str(Path(news["root"]).resolve()),
        news_store_identity=str(news["identity"]["news_store_identity"]),
        news_logical_manifest_checksum=str(news["manifest"]["logical_checksum"]),
        news_schema_checksum=str(news["manifest"]["feature_schema_checksum"]),
        lookback_days=int(lookback_days),
        output_root=str(Path(output_root).resolve()),
    )


def verify_selector_parent(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    try:
        verified = verify_dataset_lineage_manifest(manifest_path, dataset_root=root)
        manifest = _read_json(manifest_path)
    except Exception as exc:
        raise ValueError(f"SELECTOR_PARENT_INVALID: {exc}") from exc
    checks = {
        manifest.get("manifest_schema_version") == SELECTOR_DATASET_MANIFEST_VERSION,
        manifest.get("feature_contract") == SELECTOR_DATASET_CONTRACT_VERSION,
        manifest.get("publication_status") == "complete",
        manifest.get("validation_status") == "VERIFIED",
        manifest.get("economic_target_id") == CURRENT_ECONOMIC_TARGET_ID,
        manifest.get("target_provenance_contract_version")
        == CURRENT_TARGET_PROVENANCE_VERSION,
        manifest.get("row_id_contract") == SELECTOR_ROW_ID_CONTRACT_VERSION,
    }
    if False in checks:
        raise ValueError("SELECTOR_PARENT_INVALID: unsupported selector contract")
    for relative, checksum in manifest["checksums"].items():
        artifact = root / str(relative)
        if not artifact.is_file() or _sha256(artifact) != checksum:
            raise ValueError(
                f"SELECTOR_PARENT_INVALID: artifact checksum mismatch: {relative}"
            )
    return {
        "root": root, "manifest": manifest, "verified": verified,
        "identity": {
            "dataset_id": manifest["dataset_id"],
            "logical_checksum": manifest["logical_checksum"],
            "dataset_checksum": manifest["dataset_checksum"],
            "row_population_checksum": manifest["row_population_checksum"],
            "feature_schema_checksum": manifest["feature_schema_checksum"],
            "target_schema_checksum": manifest["target_schema_checksum"],
        },
    }


def verify_news_parent(root: Path, lookback_days: int) -> dict[str, Any]:
    root = root.resolve()
    try:
        manifest = _read_json(root / "manifest.json")
    except Exception as exc:
        raise ValueError(f"NEWS_PARENT_INVALID: {exc}") from exc
    comparable = {
        key: value for key, value in manifest.items() if key != "logical_checksum"
    }
    required_parents = (
        "canonical_corpus_identity", "canonical_corpus_checksum",
        "score_store_identity", "score_store_checksum",
        "canonical_daily_spine_identity", "canonical_daily_spine_checksum",
        "ticker_mapping_identity", "ticker_mapping_checksum",
    )
    windows = [
        int(item["days"]) for item in manifest.get("aggregation_windows", [])
        if isinstance(item, Mapping) and "days" in item
    ]
    policy = manifest.get("pit_eligibility_policy") or {}
    checks = {
        manifest.get("feature_store_contract") == FEATURE_STORE_CONTRACT,
        (manifest.get("feature_schema") or {}).get("contract")
        == FEATURE_SCHEMA_CONTRACT,
        (manifest.get("feature_schema") or {}).get("fields")
        == list(FEATURE_FIELDS),
        manifest.get("feature_schema_checksum")
        == canonical_hash(manifest.get("feature_schema")),
        manifest.get("logical_checksum") == canonical_hash(comparable),
        manifest.get("production_finbert_scoring_proven") is True,
        manifest.get("finbert_scoring_invoked") is False,
        policy.get("pit_policy") == STRICT_COLLECTED_AT,
        policy.get("eligibility_timestamp_field") == "collected_at_utc",
        policy.get("production_pit_validated") is True,
        windows.count(int(lookback_days)) == 1,
        all(manifest.get(field) for field in required_parents),
        bool(manifest.get("partitions")),
    }
    if False in checks:
        raise ValueError(
            "NEWS_PARENT_INVALID: incomplete, uncertified, ambiguous, or "
            "non-production-PIT feature store"
        )
    for partition in manifest["partitions"]:
        path = root / str(partition["relative_path"])
        if not path.is_file() or _sha256(path) != partition["artifact_checksum"]:
            raise ValueError("NEWS_PARENT_INVALID: partition checksum mismatch")
    identity = {
        "news_store_identity": canonical_hash({
            "feature_store_contract": manifest["feature_store_contract"],
            "logical_checksum": manifest["logical_checksum"],
            "artifact_checksum": manifest["feature_store_artifact_checksum"],
        }),
        "logical_checksum": manifest["logical_checksum"],
        "feature_store_artifact_checksum": manifest[
            "feature_store_artifact_checksum"
        ],
        "feature_schema_checksum": manifest["feature_schema_checksum"],
        **{field: manifest[field] for field in required_parents},
    }
    return {"root": root, "manifest": manifest, "identity": identity}


def read_selector_population(
    selector: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(Path(selector["root"]) / "rows.parquet").to_pylist()
    required = {
        "row_id", "asset_id", "decision_session_date", "decision_timestamp",
        "symbol", "actual_forward_return_10d", "target_start_timestamp",
        "label_end_timestamp", "label_available_timestamp", "selector_eligible",
        "target_status",
    }
    missing = sorted(required - set(rows[0] if rows else {}))
    if not rows or missing:
        raise ValueError(f"SELECTOR_PARENT_INVALID: missing row fields {missing}")
    row_ids: set[str] = set()
    keys: set[tuple[str, str]] = set()
    for row in rows:
        row_id = str(row["row_id"])
        key = (str(row["asset_id"]), str(row["decision_session_date"]))
        if not row_id or row_id in row_ids or key in keys:
            raise ValueError("DUPLICATE_SELECTOR_ROW")
        row_ids.add(row_id)
        keys.add(key)
        _instant(row["decision_timestamp"], "decision_timestamp")
        for field in ("target_start_timestamp", "label_end_timestamp",
                      "label_available_timestamp"):
            _instant(row[field], field)
        if not str(row["asset_id"]).strip():
            raise ValueError("SELECTOR_PARENT_INVALID: canonical asset missing")
    ordered = sorted(
        (
            str(row["decision_session_date"]), str(row["asset_id"]),
            str(row["row_id"]),
        )
        for row in rows
    )
    checksum = canonical_hash(ordered)
    expected = selector["manifest"]["row_population_checksum"]
    if checksum != expected:
        raise ValueError("SELECTOR_PARENT_INVALID: row population checksum mismatch")
    return rows, {
        "checksum": checksum, "row_count": len(rows),
        "asset_count": len({str(row["asset_id"]) for row in rows}),
        "decision_date_count": len({
            str(row["decision_session_date"]) for row in rows
        }),
    }


def read_news_rows(
    news: Mapping[str, Any], *, lookback_days: int,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    for partition in news["manifest"]["partitions"]:
        path = Path(news["root"]) / partition["relative_path"]
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if int(row.get("lookback_days", -1)) == int(lookback_days):
                selected.append(row)
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in selected:
        key = (str(row.get("asset_id")), str(row.get("decision_session_date")))
        if key in index:
            raise ValueError("DUPLICATE_NEWS_JOIN_KEY")
        if not all(field in row for field in FEATURE_FIELDS):
            raise ValueError("NEWS_PARENT_INVALID: feature row schema incomplete")
        decision = _instant(row["decision_timestamp"], "decision_timestamp")
        latest_publication = row.get("latest_eligible_publication_timestamp")
        latest_collection = row.get("latest_eligible_collection_timestamp")
        if latest_publication and _instant(
            latest_publication, "latest_eligible_publication_timestamp"
        ) > decision:
            raise ValueError("DECISION_CUTOFF_VIOLATION")
        if latest_collection and _instant(
            latest_collection, "latest_eligible_collection_timestamp"
        ) > decision:
            raise ValueError("DECISION_CUTOFF_VIOLATION")
        index[key] = row
    return index, selected


def align_members(
    selector_rows: Iterable[Mapping[str, Any]],
    news_index: Mapping[tuple[str, str], Mapping[str, Any]],
    news_rows: Iterable[Mapping[str, Any]],
    *, lookback_days: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    price_rows = [dict(row) for row in selector_rows]
    augmented = []
    counts = {reason: 0 for reason in REASON_CODES}
    examples = {reason: [] for reason in REASON_CODES}
    selector_keys = {
        (str(row["asset_id"]), str(row["decision_session_date"]))
        for row in price_rows
    }
    for row in price_rows:
        key = (str(row["asset_id"]), str(row["decision_session_date"]))
        feature = news_index.get(key)
        output = dict(row)
        output["news_lookback_days"] = int(lookback_days)
        if feature is None:
            counts["NO_ELIGIBLE_NEWS"] += 1
            _example(examples, "NO_ELIGIBLE_NEWS", row)
            output.update(_missing_news_values())
        else:
            if str(feature["asset_id"]) != str(row["asset_id"]):
                raise ValueError("SELECTOR_NEWS_ASSET_MISMATCH")
            if str(feature["decision_session_date"]) != str(
                row["decision_session_date"]
            ):
                raise ValueError("DECISION_DATE_MISMATCH")
            if _symbol(feature["symbol"]) != _symbol(
                row.get("canonical_symbol") or row.get("symbol")
            ):
                raise ValueError("SELECTOR_NEWS_SYMBOL_MISMATCH")
            if _instant(feature["decision_timestamp"], "news decision_timestamp") != (
                _instant(row["decision_timestamp"], "selector decision_timestamp")
            ):
                raise ValueError("DECISION_DATE_MISMATCH")
            output.update({
                field: feature[field] for field in NEWS_OUTPUT_FIELDS
            })
        augmented.append(output)
    unexpected = sorted(
        (
            (str(row["asset_id"]), str(row["decision_session_date"]))
            for row in news_rows
            if (str(row["asset_id"]), str(row["decision_session_date"]))
            not in selector_keys
        )
    )
    counts["UNEXPECTED_NEWS_ROW"] = len(unexpected)
    examples["UNEXPECTED_NEWS_ROW"] = [
        {"asset_id": asset, "decision_session_date": date}
        for asset, date in unexpected[:EXAMPLE_LIMIT]
    ]
    _validate_in_memory_members(price_rows, augmented)
    return price_rows, augmented, {"counts": counts, "examples": examples}


def validate_paired_selector_publication(
    pair_root: Path, *, expected_request_checksum: str | None = None,
    expected_pair_identity: str | None = None,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    root = Path(pair_root)
    manifest = _read_json(root / "pair_manifest.json")
    if (
        manifest.get("pair_manifest_schema_version") != PAIR_MANIFEST_SCHEMA
        or manifest.get("pair_contract_version") != PAIR_CONTRACT
        or manifest.get("publication_status") != "complete"
        or manifest.get("validation_status") != "VERIFIED"
        or manifest.get("logical_checksum") != _logical_checksum(manifest)
    ):
        raise ValueError("Incomplete or incompatible pair manifest")
    if expected_request_checksum and manifest.get(
        "request_checksum"
    ) != expected_request_checksum:
        raise ValueError("Paired request identity mismatch")
    if expected_pair_identity and manifest.get(
        "pair_identity"
    ) != expected_pair_identity:
        raise ValueError("Paired dataset identity mismatch")
    request = _read_json(root / manifest["request_path"])
    if (
        request.get("request_checksum") != manifest["request_checksum"]
        or _request_identity_from_payload(request) != request["request_checksum"]
    ):
        raise ValueError("Paired request checksum mismatch")
    members = {}
    for name in ("price_only", "price_plus_news"):
        member = manifest["members"][name]
        member_root = root / name
        member_manifest = _read_json(member_root / "manifest.json")
        if member != member_manifest:
            raise ValueError("Pair/member manifest mismatch")
        if member_manifest.get("logical_checksum") != _logical_checksum(
            member_manifest
        ):
            raise ValueError("Member logical checksum mismatch")
        rows_path = member_root / member_manifest["rows_path"]
        if _sha256(rows_path) != member_manifest["artifact_checksums"]["rows.parquet"]:
            raise ValueError("Member artifact checksum mismatch")
        rows = pq.read_table(rows_path).to_pylist()
        if schema_identity(rows) != member_manifest["feature_schema_identity"]:
            raise ValueError("Member schema identity mismatch")
        members[name] = rows
    _validate_in_memory_members(
        members["price_only"], members["price_plus_news"]
    )
    if len(members["price_only"]) != manifest["canonical_row_count"]:
        raise ValueError("MEMBER_POPULATION_MISMATCH")
    return manifest


def schema_identity(rows: list[Mapping[str, Any]]) -> str:
    if not rows:
        raise ValueError("Empty paired selector member")
    fields = sorted(rows[0])
    shapes = []
    for field in fields:
        types = sorted({
            type(row.get(field)).__name__ for row in rows
            if row.get(field) is not None
        })
        shapes.append({"field": field, "non_null_types": types})
    return canonical_hash({"fields": shapes})


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest().upper()


def _publish_member(
    root: Path, *, rows: list[dict[str, Any]], name: str, identity: str,
    schema_identity: str, population: Mapping[str, Any], pair_identity: str,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root.mkdir()
    path = root / "rows.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    manifest = {
        "member_manifest_schema_version": MEMBER_MANIFEST_SCHEMA,
        "pair_identity": pair_identity,
        "member_name": name,
        "dataset_identity": identity,
        "feature_schema_identity": schema_identity,
        "rows_path": "rows.parquet",
        "artifact_checksums": {"rows.parquet": _sha256(path)},
        "ordered_economic_row_population_checksum": population["checksum"],
        "row_count": population["row_count"],
        "asset_count": population["asset_count"],
        "decision_date_count": population["decision_date_count"],
        "contains_news_features": name == "price_plus_news",
        "publication_status": "complete",
        "validation_status": "VERIFIED",
    }
    manifest["logical_checksum"] = _logical_checksum(manifest)
    _write_json(root / "manifest.json", manifest)
    return manifest


def _validate_in_memory_members(
    price_rows: list[Mapping[str, Any]],
    augmented_rows: list[Mapping[str, Any]],
) -> None:
    if len(price_rows) != len(augmented_rows):
        raise ValueError("MEMBER_POPULATION_MISMATCH")
    if any(field in price_rows[0] for field in NEWS_MEMBER_FIELDS):
        raise ValueError("Price-only member contains news columns")
    if any(field not in augmented_rows[0] for field in NEWS_MEMBER_FIELDS):
        raise ValueError("Price-plus-news member schema incomplete")
    for left, right in zip(price_rows, augmented_rows):
        if list(left) != [field for field in right if field not in NEWS_MEMBER_FIELDS]:
            raise ValueError("MEMBER_POPULATION_MISMATCH")
        for field, value in left.items():
            if not _same_value(value, right.get(field)):
                if field in TARGET_TIMESTAMP_COLUMNS:
                    raise ValueError("TARGET_TIMESTAMP_MISMATCH")
                if field in TARGET_COLUMNS:
                    raise ValueError("TARGET_INVARIANT_MISMATCH")
                if field == "selector_eligible":
                    raise ValueError("ELIGIBILITY_MISMATCH")
                raise ValueError("MEMBER_POPULATION_MISMATCH")


def _same_value(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, float) and left != left:
        return isinstance(right, float) and right != right
    return type(left) is type(right) and left == right


def _missing_news_values() -> dict[str, Any]:
    return {
        "news_missing": True,
        "eligible_article_count": 0,
        "mean_signed_sentiment": None,
        "mean_positive_probability": None,
        "mean_negative_probability": None,
        "latest_eligible_publication_timestamp": None,
        "latest_eligible_collection_timestamp": None,
        "eligible_article_set_checksum": canonical_hash([]),
    }


def _evidence_payload(
    evidence: Mapping[str, Any], population: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "evidence_schema_version": EVIDENCE_SCHEMA,
        "reason_counts": dict(evidence["counts"]),
        "bounded_examples": dict(evidence["examples"]),
        "example_limit": EXAMPLE_LIMIT,
        "canonical_row_count": population["row_count"],
        "rows_dropped_for_missing_news": 0,
        "ordinary_missing_news_preserves_population": True,
    }
    payload["report_identity"] = canonical_hash(payload)
    return payload


def _result(
    manifest: Mapping[str, Any], *, pair_root: Path, reused: bool
) -> PairedSelectorDatasetResult:
    return PairedSelectorDatasetResult(
        pair_root=pair_root,
        pair_manifest=pair_root / "pair_manifest.json",
        pair_identity=str(manifest["pair_identity"]),
        price_only_identity=str(manifest["members"]["price_only"]["dataset_identity"]),
        price_plus_news_identity=str(
            manifest["members"]["price_plus_news"]["dataset_identity"]
        ),
        row_count=int(manifest["canonical_row_count"]),
        covered_news_row_count=int(manifest["covered_news_row_count"]),
        missing_news_row_count=int(manifest["missing_news_row_count"]),
        reused=reused,
    )


def _logical_checksum(value: Mapping[str, Any]) -> str:
    return logical_manifest_checksum(value)


def _request_identity_from_payload(value: Mapping[str, Any]) -> str:
    return canonical_hash({
        key: item for key, item in value.items()
        if key not in {
            "request_checksum",
            "selector_dataset_root",
            "news_feature_store_root",
            "output_root",
        }
    })


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _instant(value: Any, field: str) -> datetime:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} requires a valid timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} requires an explicit timezone")
    return parsed.astimezone(timezone.utc)


def _symbol(value: Any) -> str:
    return str(value or "").strip().upper().replace(".", "-")


def _example(
    examples: dict[str, list[dict[str, Any]]], reason: str,
    row: Mapping[str, Any],
) -> None:
    if len(examples[reason]) < EXAMPLE_LIMIT:
        examples[reason].append({
            "row_id": str(row.get("row_id") or ""),
            "asset_id": str(row.get("asset_id") or ""),
            "decision_session_date": str(
                row.get("decision_session_date") or ""
            ),
        })


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
