from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import pyarrow.parquet as pq

from core.research.ml.ds24 import master_5m_schema_r7 as r7
from core.research.ml.ds24.master_5m_validation_r7v import (
    COMPONENT_REL,
    STAGE_REL,
    find_or_create_run as find_r7v_run,
    load_benchmarks,
    load_frame,
    partition_inventory,
    stable_hash,
)
from core.research.ml.ds24.master_5m_validation_stats import compute_stock_features


AUTHORITY_ID = "CANONICAL_5M_FEATURE_AUTHORITY_FULL_V1"
RUN_PREFIX = "ds24_p6_r8_cloud_master_"
R2_PREFIX_ROOT = "ds24/master-5m/version=v1"
R2_PREFLIGHT_PREFIX = "ds24/_r8_preflight"
R2_STORAGE_CLASS = "STANDARD"
R8_REMOTE_PREFIX_MAX_GIB = 80
R8_CLASS_A_LOGICAL_CALL_LIMIT = 50_000
R8_CLASS_B_LOGICAL_CALL_LIMIT = 100_000
R8_MAX_API_ATTEMPTS_PER_OPERATION = 5
LOCAL_RESERVE_GIB = 10.0
SINGLE_PUT_MAX_BYTES = 5 * 1024**3
EXPECTED_LIMITATION_FEATURES = {
    "relative_volume_tod_pit",
    "cumulative_volume_ratio_tod_pit",
    "vol_percentile_20d_tod_pit",
    "relative_strength_rank_60m",
    "relative_strength_rank_120m",
    "range_expansion_session",
}


class R8Blocked(RuntimeError):
    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


class S3Like(Protocol):
    def head_bucket(self, *, Bucket: str) -> Any: ...
    def put_object(self, **kwargs: Any) -> Any: ...
    def head_object(self, *, Bucket: str, Key: str) -> Any: ...
    def get_object(self, *, Bucket: str, Key: str, Range: str | None = None) -> Any: ...
    def delete_object(self, *, Bucket: str, Key: str) -> Any: ...
    def list_objects_v2(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class R2Environment:
    bucket: str
    endpoint: str
    access_key_id: str
    secret_access_key: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    write_json(tmp, payload)
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def disk(root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(str(root.anchor or root))
    return {"free_bytes": usage.free, "free_gib": round(usage.free / 1024**3, 3)}


def require_r2_environment(env: dict[str, str] | None = None) -> R2Environment:
    source = env if env is not None else os.environ
    missing = [name for name in ("R2_BUCKET", "R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY") if not source.get(name)]
    if missing:
        raise R8Blocked("BLOCKED_DS24_R8_CREDENTIALS", f"missing required R2 environment variables: {', '.join(missing)}")
    return R2Environment(
        bucket=str(source["R2_BUCKET"]),
        endpoint=str(source["R2_ENDPOINT"]),
        access_key_id=str(source["R2_ACCESS_KEY_ID"]),
        secret_access_key=str(source["R2_SECRET_ACCESS_KEY"]),
    )


def create_s3_client(r2_env: R2Environment) -> S3Like:
    try:
        import boto3  # type: ignore
        from botocore.config import Config  # type: ignore
    except ImportError as exc:
        raise R8Blocked("BLOCKED_DS24_R8_R2_PREFLIGHT", "boto3/botocore are required for R2 S3-compatible access") from exc
    return boto3.client(
        "s3",
        endpoint_url=r2_env.endpoint,
        aws_access_key_id=r2_env.access_key_id,
        aws_secret_access_key=r2_env.secret_access_key,
        region_name="auto",
        config=Config(retries={"max_attempts": R8_MAX_API_ATTEMPTS_PER_OPERATION, "mode": "standard"}),
    )


class OperationLedger:
    CLASS_A = {"PutObject", "ListObjectsV2"}
    CLASS_B = {"HeadObject", "GetObject"}

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.rows: list[dict[str, Any]] = []
        self.counts = {
            "PutObject": 0,
            "HeadObject": 0,
            "GetObject": 0,
            "ListObjectsV2": 0,
            "DeleteObject": 0,
            "other_s3_calls": 0,
        }
        self.class_a = 0
        self.class_b = 0
        self.delete_calls = 0
        self.remote_verified_bytes = 0

    def can_call(self, operation: str) -> bool:
        class_a_after = self.class_a + (1 if operation in self.CLASS_A else 0)
        class_b_after = self.class_b + (1 if operation in self.CLASS_B else 0)
        return class_a_after <= R8_CLASS_A_LOGICAL_CALL_LIMIT and class_b_after <= R8_CLASS_B_LOGICAL_CALL_LIMIT

    def ensure_can_call(self, operation: str) -> None:
        if not self.can_call(operation):
            raise R8Blocked("BLOCKED_DS24_R8_OPERATION_GUARD", f"{operation} would exceed R8 operation guard")

    def ensure_remote_bytes(self, additional_bytes: int) -> None:
        max_bytes = R8_REMOTE_PREFIX_MAX_GIB * 1024**3
        if self.remote_verified_bytes + additional_bytes > max_bytes:
            raise R8Blocked("BLOCKED_DS24_R8_REMOTE_PREFIX_SIZE_GUARD", "remote verified bytes would exceed R8 run guard")

    def record(
        self,
        operation: str,
        *,
        key: str = "",
        prefix: str = "",
        bytes_sent: int = 0,
        bytes_received: int = 0,
        success: bool = True,
        attempts: int | None = None,
        error: str = "",
    ) -> None:
        self.counts[operation if operation in self.counts else "other_s3_calls"] += 1
        if operation in self.CLASS_A:
            self.class_a += 1
        elif operation in self.CLASS_B:
            self.class_b += 1
        elif operation == "DeleteObject":
            self.delete_calls += 1
        row = {
            "timestamp_utc": utc_now(),
            "operation": operation,
            "logical_call_count": 1,
            "attempt_count": attempts if attempts is not None else "",
            "object_key_or_prefix": key or prefix,
            "bytes_sent": bytes_sent,
            "bytes_received": bytes_received,
            "success": success,
            "failure": "" if success else error,
        }
        self.rows.append(row)

    def persist(self) -> None:
        write_json(self.run_dir / "r8_r2_operation_ledger.json", self.rows)
        write_csv(self.run_dir / "r8_r2_operation_ledger.csv", self.rows)
        self.write_cost_guard("DS24P6_R8_CLOUD_BUILD_RUNNING")

    def write_cost_guard(self, classification: str) -> None:
        payload = {
            "remote_verified_bytes": self.remote_verified_bytes,
            "remote_verified_GiB": round(self.remote_verified_bytes / 1024**3, 6),
            "Class A logical calls": self.class_a,
            "Class B logical calls": self.class_b,
            "delete calls": self.delete_calls,
            "hard_limits": {
                "R8_REMOTE_PREFIX_MAX_GIB": R8_REMOTE_PREFIX_MAX_GIB,
                "R8_CLASS_A_LOGICAL_CALL_LIMIT": R8_CLASS_A_LOGICAL_CALL_LIMIT,
                "R8_CLASS_B_LOGICAL_CALL_LIMIT": R8_CLASS_B_LOGICAL_CALL_LIMIT,
                "R8_MAX_API_ATTEMPTS_PER_OPERATION": R8_MAX_API_ATTEMPTS_PER_OPERATION,
            },
            "fraction_of_run_limits_consumed": {
                "remote_bytes": self.remote_verified_bytes / (R8_REMOTE_PREFIX_MAX_GIB * 1024**3),
                "Class A": self.class_a / R8_CLASS_A_LOGICAL_CALL_LIMIT,
                "Class B": self.class_b / R8_CLASS_B_LOGICAL_CALL_LIMIT,
            },
            "classification": classification,
        }
        write_json(self.run_dir / "r8_cost_guard.json", payload)


class GuardedS3:
    def __init__(self, client: S3Like, ledger: OperationLedger) -> None:
        self.client = client
        self.ledger = ledger

    def head_bucket(self, *, Bucket: str) -> Any:
        operation = "HeadBucket"
        self.ledger.ensure_can_call(operation)
        try:
            result = self.client.head_bucket(Bucket=Bucket)
            self.ledger.record(operation, success=True)
            return result
        except Exception as exc:
            self.ledger.record(operation, success=False, error=str(exc))
            raise

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, Metadata: dict[str, str], ContentType: str | None = None) -> Any:
        operation = "PutObject"
        self.ledger.ensure_can_call(operation)
        if len(Body) > SINGLE_PUT_MAX_BYTES:
            raise R8Blocked("BLOCKED_DS24_R8_OBJECT_TOO_LARGE_FOR_SINGLE_PUT", f"{Key} is too large for frozen single PutObject policy")
        try:
            result = self.client.put_object(
                Bucket=Bucket,
                Key=Key,
                Body=Body,
                Metadata=Metadata,
                ContentType=ContentType or "application/octet-stream",
                ChecksumAlgorithm="SHA256",
            )
            self.ledger.record(operation, key=Key, bytes_sent=len(Body), success=True)
            return result
        except Exception as exc:
            self.ledger.record(operation, key=Key, bytes_sent=len(Body), success=False, error=str(exc))
            raise

    def head_object(self, *, Bucket: str, Key: str) -> Any:
        operation = "HeadObject"
        self.ledger.ensure_can_call(operation)
        try:
            result = self.client.head_object(Bucket=Bucket, Key=Key)
            self.ledger.record(operation, key=Key, success=True)
            return result
        except Exception as exc:
            self.ledger.record(operation, key=Key, success=False, error=str(exc))
            raise

    def get_object(self, *, Bucket: str, Key: str, Range: str | None = None) -> Any:
        operation = "GetObject"
        self.ledger.ensure_can_call(operation)
        try:
            kwargs: dict[str, Any] = {"Bucket": Bucket, "Key": Key}
            if Range:
                kwargs["Range"] = Range
            result = self.client.get_object(**kwargs)
            received = int(result.get("ContentLength") or 0)
            self.ledger.record(operation, key=Key, bytes_received=received, success=True)
            return result
        except Exception as exc:
            self.ledger.record(operation, key=Key, success=False, error=str(exc))
            raise

    def delete_object(self, *, Bucket: str, Key: str) -> Any:
        operation = "DeleteObject"
        self.ledger.ensure_can_call(operation)
        try:
            result = self.client.delete_object(Bucket=Bucket, Key=Key)
            self.ledger.record(operation, key=Key, success=True)
            return result
        except Exception as exc:
            self.ledger.record(operation, key=Key, success=False, error=str(exc))
            raise

    def list_objects_v2(self, **kwargs: Any) -> Any:
        operation = "ListObjectsV2"
        self.ledger.ensure_can_call(operation)
        try:
            result = self.client.list_objects_v2(**kwargs)
            self.ledger.record(operation, prefix=kwargs.get("Prefix", ""), success=True)
            return result
        except Exception as exc:
            self.ledger.record(operation, prefix=kwargs.get("Prefix", ""), success=False, error=str(exc))
            raise


def run_r2_preflight(s3: GuardedS3, bucket: str, run_candidate: str) -> dict[str, Any]:
    key = f"{R2_PREFLIGHT_PREFIX}/{run_candidate}/{uuid.uuid4().hex}.txt"
    body = f"ds24-r8-preflight-{uuid.uuid4().hex}".encode("ascii")
    expected_sha = hashlib.sha256(body).hexdigest()
    try:
        s3.head_bucket(Bucket=bucket)
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            Metadata={"logical-authority-id": AUTHORITY_ID, "r8-preflight": "true", "sha256": expected_sha},
            ContentType="text/plain",
        )
        head = s3.head_object(Bucket=bucket, Key=key)
        if int(head.get("ContentLength", -1)) != len(body):
            raise RuntimeError("preflight HEAD size mismatch")
        got = s3.get_object(Bucket=bucket, Key=key)
        payload = got["Body"].read()
        if hashlib.sha256(payload).hexdigest() != expected_sha:
            raise RuntimeError("preflight GET SHA-256 mismatch")
        s3.delete_object(Bucket=bucket, Key=key)
    except Exception as exc:
        raise R8Blocked("BLOCKED_DS24_R8_R2_PREFLIGHT", str(exc)) from exc
    return {"classification": "R2_PREFLIGHT_PASS", "key": key, "sha256": expected_sha}


def latest_r7v_run(root: Path) -> Path:
    run_dir = find_r7v_run(root)
    decision = json.loads((run_dir / "34_r7v_final_decision.json").read_text(encoding="utf-8"))
    if decision.get("classification") != "DS24P6_R7V_FULL_HISTORY_VALIDATION_PASS_WITH_LIMITATIONS":
        raise R8Blocked("BLOCKED_DS24_R8_SOURCE_IDENTITY", "R7V parent is not terminal pass-with-limitations")
    return run_dir


def find_or_create_r8_run(root: Path, run_candidate: str | None = None) -> Path:
    stage = root / STAGE_REL
    stage.mkdir(parents=True, exist_ok=True)
    existing = sorted(path for path in stage.iterdir() if path.is_dir() and path.name.startswith(RUN_PREFIX))
    if existing:
        return existing[0]
    run_id = run_candidate or RUN_PREFIX + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = stage / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_registries(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    component = root / COMPONENT_REL
    contract = json.loads((component / "r7_master_feature_contract.json").read_text(encoding="utf-8"))
    stock = list(csv.DictReader((component / "r7_master_stock_features.csv").open(newline="", encoding="utf-8")))
    shared = list(csv.DictReader((component / "r7_master_shared_context.csv").open(newline="", encoding="utf-8")))
    if len(stock) != 72 or len(shared) != 29:
        raise R8Blocked("BLOCKED_DS24_R8_MASTER_SCHEMA_IDENTITY", "R7 feature counts are not 72 stock / 29 shared context")
    observed_limitations = {row["semantic_feature_id"] for row in stock if row["semantic_feature_id"] in EXPECTED_LIMITATION_FEATURES}
    if observed_limitations != EXPECTED_LIMITATION_FEATURES:
        raise R8Blocked("BLOCKED_DS24_R8_MASTER_SCHEMA_IDENTITY", "expected-limitation feature set drifted")
    return contract, stock, shared


def source_identity_from_r7v(root: Path) -> str:
    r7v = latest_r7v_run(root)
    registration = json.loads((r7v / "04_r7v_run_registration.json").read_text(encoding="utf-8"))
    return str(registration["source_identity"])


def registration_payload(root: Path, run_id: str, bucket: str, prefix: str) -> dict[str, Any]:
    contract, stock, shared = load_registries(root)
    r7v = latest_r7v_run(root)
    r7v_registration = json.loads((r7v / "04_r7v_run_registration.json").read_text(encoding="utf-8"))
    code_identity = stable_hash(
        {
            "module": Path(__file__).read_text(encoding="utf-8"),
            "feature_stats_module": "core.research.ml.ds24.master_5m_validation_stats",
            "schema_module": "core.research.ml.ds24.master_5m_schema_r7",
        }
    )
    return {
        "run_id": run_id,
        "classification": "DS24P6_R8_CLOUD_BUILD_REGISTERED",
        "canonical_source_identity": r7v_registration["source_identity"],
        "r7_broad_master_schema_identity": contract["schema_identity"],
        "r7_authority_identity": contract["authority_identity"],
        "accepted_parents": [
            "DS24P6_R7_MASTER_SCHEMA_READY_LOCAL_CAPACITY_BLOCKED",
            "DS24P6_R7V_FULL_HISTORY_VALIDATION_PASS_WITH_LIMITATIONS",
        ],
        "stock_feature_count": len(stock),
        "shared_context_feature_count": len(shared),
        "development_cutoff": r7.DEVELOPMENT_END,
        "locked_holdout_start": r7.HOLDOUT_START,
        "pit_timing_contract": "timestamp_utc bar start; feature availability is timestamp_utc + 5 minutes",
        "physical_parquet_layout": "stock/asset=<IDENTITY>/year=<YEAR>/features.parquet and shared-context/year=<YEAR>/context.parquet",
        "r2_bucket": bucket,
        "r2_prefix": prefix,
        "storage_class": R2_STORAGE_CLASS,
        "billing_guard_contract": {
            "R8_REMOTE_PREFIX_MAX_GIB": R8_REMOTE_PREFIX_MAX_GIB,
            "R8_CLASS_A_LOGICAL_CALL_LIMIT": R8_CLASS_A_LOGICAL_CALL_LIMIT,
            "R8_CLASS_B_LOGICAL_CALL_LIMIT": R8_CLASS_B_LOGICAL_CALL_LIMIT,
            "R8_MAX_API_ATTEMPTS_PER_OPERATION": R8_MAX_API_ATTEMPTS_PER_OPERATION,
            "single_put_only": True,
            "multipart_uploads": False,
            "remote_copy_promotion_loop": False,
        },
        "code_capsule_identity": code_identity,
    }


def checkpoint_payload(run_id: str, classification: str, planned_stock: int, planned_context: int) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "classification": classification,
        "partitions_planned": planned_stock + planned_context,
        "stock_partitions_planned": planned_stock,
        "context_partitions_planned": planned_context,
        "partitions_built": 0,
        "partitions_uploaded": 0,
        "partitions_verified": 0,
        "stock_partitions_verified": 0,
        "context_partitions_verified": 0,
        "partitions_failed": 0,
        "source_rows_processed": 0,
        "stock_feature_rows": 0,
        "context_rows": 0,
        "remote_bytes_verified": 0,
        "current_partition": "",
        "Class A logical calls": 0,
        "Class B logical calls": 0,
        "delete calls": 0,
        "last_checkpoint_utc": utc_now(),
        "local free disk": {},
        "last_error": "",
    }


def load_checkpoint(path: Path, run_id: str, planned_stock: int, planned_context: int) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return checkpoint_payload(run_id, "DS24P6_R8_CLOUD_BUILD_RUNNING", planned_stock, planned_context)


def update_checkpoint(run_dir: Path, checkpoint: dict[str, Any], ledger: OperationLedger, root: Path) -> None:
    checkpoint["remote_bytes_verified"] = ledger.remote_verified_bytes
    checkpoint["Class A logical calls"] = ledger.class_a
    checkpoint["Class B logical calls"] = ledger.class_b
    checkpoint["delete calls"] = ledger.delete_calls
    checkpoint["last_checkpoint_utc"] = utc_now()
    checkpoint["local free disk"] = disk(root)
    atomic_json(run_dir / "r8_checkpoint.json", checkpoint)
    ledger.persist()


def object_metadata(registration: dict[str, Any], partition_identity: str, row_count: int, sha256: str) -> dict[str, str]:
    return {
        "logical-authority-id": AUTHORITY_ID,
        "r8-run-id": registration["run_id"],
        "partition-identity": partition_identity,
        "source-identity": registration["canonical_source_identity"],
        "schema-identity": registration["r7_broad_master_schema_identity"],
        "feature-contract-identity": registration["r7_authority_identity"],
        "row-count": str(row_count),
        "sha256": sha256,
        "development-boundary-authority": f"development_cutoff={r7.DEVELOPMENT_END};holdout_start={r7.HOLDOUT_START}",
        "storage-class": R2_STORAGE_CLASS,
    }


def validate_head(head: dict[str, Any], expected_size: int, metadata: dict[str, str]) -> bool:
    if int(head.get("ContentLength", -1)) != expected_size:
        return False
    observed = {str(k).lower(): str(v) for k, v in (head.get("Metadata") or {}).items()}
    for key, value in metadata.items():
        if observed.get(key.lower()) != value:
            return False
    storage = str(head.get("StorageClass") or R2_STORAGE_CLASS).upper()
    return storage in {"", R2_STORAGE_CLASS}


def parquet_schema_string(path: Path) -> str:
    return str(pq.ParquetFile(path).schema_arrow)


def write_stock_partition(root: Path, temp_dir: Path, part: Any, stock_feature_ids: list[str], benchmarks: dict[str, pd.Series]) -> tuple[Path, int, int, str]:
    frame = load_frame(root / part.relative_path)
    duplicates = int(frame.duplicated(["asset_id", "timestamp_utc"]).sum()) if len(frame) else 0
    if duplicates:
        raise R8Blocked("BLOCKED_DS24_R8_VALIDATION", f"duplicate stock keys in {part.relative_path}: {duplicates}")
    features = compute_stock_features(frame, benchmarks)
    missing = [feature_id for feature_id in stock_feature_ids if feature_id not in features.columns]
    if missing:
        raise R8Blocked("BLOCKED_DS24_R8_VALIDATION", f"missing stock features in {part.relative_path}: {missing}")
    out = frame[["asset_id", "canonical_symbol", "provider_symbol", "timestamp_utc", "session_date", "session_type"]].copy()
    out["decision_timestamp"] = pd.to_datetime(out["timestamp_utc"], utc=True) + pd.Timedelta(minutes=5)
    for feature_id in stock_feature_ids:
        out[feature_id] = pd.to_numeric(features[feature_id], errors="coerce").astype("float32")
    nonfinite = int(out[stock_feature_ids].isin([float("inf"), float("-inf")]).sum().sum())
    if nonfinite:
        raise R8Blocked("BLOCKED_DS24_R8_VALIDATION", f"nonfinite values in {part.relative_path}: {nonfinite}")
    path = temp_dir / "stock" / f"asset={part.asset_id}" / f"year={part.year}" / "features.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False, engine="pyarrow")
    with path.open("ab") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    return path, len(out), int(len(frame)), parquet_schema_string(path)


def upload_verified(
    s3: GuardedS3,
    bucket: str,
    key: str,
    local_path: Path,
    metadata: dict[str, str],
    checkpoint: dict[str, Any],
    ledger: OperationLedger,
) -> str:
    size = local_path.stat().st_size
    ledger.ensure_remote_bytes(size)
    try:
        existing = s3.head_object(Bucket=bucket, Key=key)
        if validate_head(existing, size, metadata):
            ledger.remote_verified_bytes += size
            return "VERIFIED_EXISTING"
        raise R8Blocked("BLOCKED_DS24_R8_CONFLICTING_REMOTE_OBJECT", f"conflicting existing object: {key}")
    except R8Blocked:
        raise
    except Exception:
        pass
    body = local_path.read_bytes()
    s3.put_object(Bucket=bucket, Key=key, Body=body, Metadata=metadata, ContentType="application/vnd.apache.parquet")
    head = s3.head_object(Bucket=bucket, Key=key)
    if not validate_head(head, size, metadata):
        raise R8Blocked("BLOCKED_DS24_R8_REMOTE_INTEGRITY", f"remote HEAD validation failed for {key}")
    ledger.remote_verified_bytes += size
    checkpoint["partitions_uploaded"] = int(checkpoint.get("partitions_uploaded", 0)) + 1
    return "VERIFIED"


def audit_sample_indices(total: int) -> set[int]:
    if total <= 0:
        return set()
    rng = random.Random(240608)
    sample = {0, total - 1}
    if total > 2:
        sample.add(total // 2)
        sample.update(rng.sample(range(total), min(5, total)))
    return sample


def run_campaign(
    root: Path,
    *,
    max_partitions: int | None = None,
    client: S3Like | None = None,
    env: dict[str, str] | None = None,
    run_candidate: str | None = None,
) -> dict[str, Any]:
    r2_env = require_r2_environment(env)
    candidate = run_candidate or RUN_PREFIX + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    preflight_dir = root / STAGE_REL / f"{candidate}_preflight"
    ledger = OperationLedger(preflight_dir)
    s3 = GuardedS3(client or create_s3_client(r2_env), ledger)
    preflight = run_r2_preflight(s3, r2_env.bucket, candidate)
    ledger.persist()

    run_dir = find_or_create_r8_run(root, candidate)
    run_id = run_dir.name
    prefix = f"{R2_PREFIX_ROOT}/run={run_id}"
    registration = registration_payload(root, run_id, r2_env.bucket, prefix)
    write_json(run_dir / "04_r8_run_registration.json", registration)
    write_json(run_dir / "03_r2_preflight.json", preflight)

    contract, stock_registry, shared_registry = load_registries(root)
    stock_feature_ids = [row["semantic_feature_id"] for row in stock_registry]
    inventory = partition_inventory(root)
    planned = inventory[:max_partitions] if max_partitions else inventory
    years = sorted({part.year for part in planned})
    write_csv(run_dir / "05_source_partition_inventory.csv", [part.__dict__ for part in planned])
    write_json(run_dir / "06_stock_feature_registry.json", stock_registry)
    write_json(run_dir / "07_shared_context_registry.json", shared_registry)

    ledger = OperationLedger(run_dir)
    s3 = GuardedS3(client or create_s3_client(r2_env), ledger)
    checkpoint = load_checkpoint(run_dir / "r8_checkpoint.json", run_id, len(planned), len(years))
    temp_dir = run_dir / "_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    benchmarks = load_benchmarks(root)
    stock_manifest: list[dict[str, Any]] = []
    size_values: list[int] = []
    audit_indices = audit_sample_indices(len(planned))
    full_get_audits: list[dict[str, Any]] = []
    start = time.perf_counter()
    max_temp = 0

    for index, part in enumerate(planned):
        partition_id = f"stock/asset={part.asset_id}/year={part.year}"
        if checkpoint.get("verified_partition_identities", {}).get(partition_id):
            continue
        if disk(root)["free_gib"] < LOCAL_RESERVE_GIB:
            checkpoint["classification"] = "DS24P6_R8_CLOUD_BUILD_PAUSED_LOCAL_RESOURCE"
            update_checkpoint(run_dir, checkpoint, ledger, root)
            return checkpoint
        checkpoint["current_partition"] = partition_id
        pstart = time.perf_counter()
        try:
            path, rows, source_rows, schema = write_stock_partition(root, temp_dir, part, stock_feature_ids, benchmarks)
            size = path.stat().st_size
            max_temp = max(max_temp, size)
            if size > SINGLE_PUT_MAX_BYTES:
                raise R8Blocked("BLOCKED_DS24_R8_OBJECT_TOO_LARGE_FOR_SINGLE_PUT", f"{partition_id} size {size}")
            sha = sha256_file(path)
            key = f"{prefix}/stock/asset={part.asset_id}/year={part.year}/features.parquet"
            metadata = object_metadata(registration, partition_id, rows, sha)
            state = upload_verified(s3, r2_env.bucket, key, path, metadata, checkpoint, ledger)
            if index in audit_indices:
                got = s3.get_object(Bucket=r2_env.bucket, Key=key)
                readback = got["Body"].read()
                if hashlib.sha256(readback).hexdigest() != sha:
                    raise R8Blocked("BLOCKED_DS24_R8_REMOTE_INTEGRITY", f"full GET audit failed for {key}")
                full_get_audits.append({"partition_identity": partition_id, "key": key, "bytes": len(readback), "sha256": sha})
            path.unlink(missing_ok=True)
            checkpoint["partitions_built"] += 1
            checkpoint["partitions_verified"] += 1
            checkpoint["stock_partitions_verified"] += 1
            checkpoint["source_rows_processed"] += source_rows
            checkpoint["stock_feature_rows"] += rows
            checkpoint.setdefault("verified_partition_identities", {})[partition_id] = key
            stock_manifest.append({"partition_identity": partition_id, "key": key, "rows": rows, "bytes": size, "sha256": sha, "schema": schema, "verification_state": state})
            size_values.append(size)
            elapsed = max(0.001, time.perf_counter() - pstart)
            checkpoint.setdefault("performance", []).append({"partition_identity": partition_id, "source_rows_per_sec": round(source_rows / elapsed, 3), "feature_rows_per_sec": round(rows / elapsed, 3), "upload_and_verify_sec": round(elapsed, 3)})
            update_checkpoint(run_dir, checkpoint, ledger, root)
        except R8Blocked as exc:
            checkpoint["classification"] = exc.classification
            checkpoint["partitions_failed"] += 1
            checkpoint["last_error"] = str(exc)
            update_checkpoint(run_dir, checkpoint, ledger, root)
            return checkpoint
        finally:
            shutil.rmtree(temp_dir / "stock", ignore_errors=True)

    context_manifest = publish_context_partitions(run_dir, root, s3, r2_env.bucket, prefix, registration, years, shared_registry, checkpoint, ledger)
    all_manifest = stock_manifest + context_manifest
    aggregate = stable_hash([{"key": row["key"], "sha256": row["sha256"], "rows": row["rows"], "bytes": row["bytes"]} for row in all_manifest])
    partition_hash = stable_hash([row["sha256"] for row in all_manifest])
    sizes = sorted(size_values)
    size_distribution = {
        "min": sizes[0] if sizes else 0,
        "median": sizes[len(sizes) // 2] if sizes else 0,
        "p95": sizes[int(len(sizes) * 0.95) - 1] if sizes else 0,
        "max": sizes[-1] if sizes else 0,
    }
    checkpoint["classification"] = (
        "DS24P6_R8_CLOUD_MASTER_BUILD_COMPLETE"
        if checkpoint["stock_partitions_verified"] == len(planned)
        and checkpoint["context_partitions_verified"] == len(years)
        and checkpoint["partitions_failed"] == 0
        else "DS24P6_R8_CLOUD_BUILD_RUNNING"
    )
    checkpoint["remote_object_count"] = len(all_manifest)
    checkpoint["full_GET_audit_sample_count"] = len(full_get_audits)
    checkpoint["aggregate_dataset_identity"] = aggregate
    checkpoint["aggregate_partition_hash_identity"] = partition_hash
    checkpoint["local_max_temp_footprint"] = max_temp
    elapsed = max(0.001, time.perf_counter() - start)
    checkpoint["throughput"] = {
        "source_rows_per_sec": round(checkpoint["source_rows_processed"] / elapsed, 3),
        "stock_feature_rows_per_sec": round(checkpoint["stock_feature_rows"] / elapsed, 3),
    }
    update_checkpoint(run_dir, checkpoint, ledger, root)
    publish_run_manifests(run_dir, registration, stock_manifest, context_manifest, checkpoint, aggregate, partition_hash, size_distribution, full_get_audits, contract)
    return checkpoint


def publish_context_partitions(
    run_dir: Path,
    root: Path,
    s3: GuardedS3,
    bucket: str,
    prefix: str,
    registration: dict[str, Any],
    years: list[int],
    shared_registry: list[dict[str, Any]],
    checkpoint: dict[str, Any],
    ledger: OperationLedger,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    feature_ids = [row["semantic_feature_id"] for row in shared_registry]
    temp_dir = run_dir / "_tmp" / "shared-context"
    benchmarks = load_benchmarks(root)
    for year in years:
        partition_id = f"shared-context/year={year}"
        if checkpoint.get("verified_partition_identities", {}).get(partition_id):
            continue
        timestamps = sorted({ts for series in benchmarks.values() for ts in series.index if int(ts.year) == year})
        frame = pd.DataFrame({"decision_timestamp": timestamps})
        frame["breadth_population_id"] = "DS24_DEV_ELIGIBLE_V1"
        for feature_id in feature_ids:
            if feature_id in benchmarks:
                frame[feature_id] = frame["decision_timestamp"].map(benchmarks[feature_id]).astype("float32")
            else:
                frame[feature_id] = pd.NA
        path = temp_dir / f"year={year}" / "context.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False, engine="pyarrow")
        sha = sha256_file(path)
        size = path.stat().st_size
        key = f"{prefix}/shared-context/year={year}/context.parquet"
        metadata = object_metadata(registration, partition_id, len(frame), sha)
        state = upload_verified(s3, bucket, key, path, metadata, checkpoint, ledger)
        path.unlink(missing_ok=True)
        checkpoint["partitions_built"] += 1
        checkpoint["partitions_verified"] += 1
        checkpoint["context_partitions_verified"] += 1
        checkpoint["context_rows"] += len(frame)
        checkpoint.setdefault("verified_partition_identities", {})[partition_id] = key
        manifest.append({"partition_identity": partition_id, "key": key, "rows": len(frame), "bytes": size, "sha256": sha, "schema": parquet_schema_string(path) if path.exists() else "", "verification_state": state})
        update_checkpoint(run_dir, checkpoint, ledger, root)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return manifest


def publish_run_manifests(
    run_dir: Path,
    registration: dict[str, Any],
    stock_manifest: list[dict[str, Any]],
    context_manifest: list[dict[str, Any]],
    checkpoint: dict[str, Any],
    aggregate: str,
    partition_hash: str,
    size_distribution: dict[str, int],
    full_get_audits: list[dict[str, Any]],
    contract: dict[str, Any],
) -> None:
    write_json(run_dir / "manifests" / "r8_partition_manifest.json", {"stock": stock_manifest, "shared_context": context_manifest})
    write_json(run_dir / "manifests" / "r8_full_get_audit_sample.json", full_get_audits)
    write_json(run_dir / "manifests" / "r8_object_size_distribution.json", size_distribution)
    write_json(
        run_dir / "manifests" / "r8_cloud_authority_manifest.json",
        {
            "authority_id": AUTHORITY_ID,
            "authority_version": "v1",
            "run_id": registration["run_id"],
            "source_authority": registration["canonical_source_identity"],
            "R7_R7V_validation_authorities": registration["accepted_parents"],
            "stock_feature_registry": contract["stock_features"],
            "shared_context_registry": contract["shared_context_features"],
            "daily_context_join_contract": "DAILY_ASOF_CONTEXT PIT as-of join; not physically duplicated",
            "development_cutoff": r7.DEVELOPMENT_END,
            "partition_inventory": {"stock": len(stock_manifest), "shared_context": len(context_manifest)},
            "total_rows": checkpoint["stock_feature_rows"] + checkpoint["context_rows"],
            "total_remote_bytes": checkpoint["remote_bytes_verified"],
            "aggregate_logical_dataset_identity": aggregate,
            "aggregate_partition_hash_identity": partition_hash,
            "R2_bucket": registration["r2_bucket"],
            "R2_prefix": registration["r2_prefix"],
            "operation_totals": {
                "Class A": checkpoint["Class A logical calls"],
                "Class B": checkpoint["Class B logical calls"],
                "delete": checkpoint["delete calls"],
            },
            "validation_result": checkpoint["classification"],
        },
    )
