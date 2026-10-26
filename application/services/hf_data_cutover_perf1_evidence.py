"""Deterministic evidence projections and atomic persistence for PERF1."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from core.research.iex_market_data import IexPartitionRequest
from core.research.iex_market_data import plan_iex_partitions
from application.services.hf_data_cutover_iex_contract import IexBackfillSettings
from application.services.hf_data_cutover_perf1_contract import PERF1_ADOPTION_RATIO, PERF1_SCHEMA
from application.services.hf_data_cutover_service import CutoverPaths, atomic_json, utc_now
from infrastructure.data.iex_market_data_store import IexMarketDataStore, sha256_file


def completed_hash_row(store: IexMarketDataStore, request: IexPartitionRequest) -> dict[str, Any]:
    manifest = store.completed_manifest(request)
    if manifest is None:
        raise ValueError(f"Previously completed partition disappeared: {request.partition_id}")
    return {
        "request": request.identity(),
        "parquet_sha256": manifest["parquet_sha256"],
        "manifest_sha256": sha256_file(store.manifest_path(request)),
    }


def planned_partitions(settings: IexBackfillSettings) -> dict[str, tuple[IexPartitionRequest, ...]]:
    return {
        "5m": plan_iex_partitions(settings.roster, start=settings.start, end=settings.end, timeframe="5m"),
        "1Day": plan_iex_partitions(settings.roster, start=settings.start, end=settings.end, timeframe="1Day"),
    }


def validated_inventory(
    store: IexMarketDataStore,
    plans: Mapping[str, Sequence[IexPartitionRequest]],
) -> tuple[list[IexPartitionRequest], list[IexPartitionRequest]]:
    completed, pending = [], []
    for timeframe in ("5m", "1Day"):
        for request in plans[timeframe]:
            (completed if store.completed_manifest(request) is not None else pending).append(request)
    return completed, pending


def scratch_ceiling(repository_root: Path, maximum_bytes: int) -> int:
    return min(maximum_bytes, int(shutil.disk_usage(repository_root).free * 0.20))


def tree_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def failure_classification_counts(
    store: IexMarketDataStore,
    pending: Sequence[IexPartitionRequest],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for request in pending:
        path = store.partition_dir(request) / "failure.json"
        classification = str(optional_json(path).get("classification", "missing_unattempted"))
        counts[classification] = counts.get(classification, 0) + 1
    return dict(sorted(counts.items()))


def select_transport(paths: CutoverPaths, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline = next(row for row in rows if row["variant"] == "A")
    candidates = [row for row in rows if row["variant"] in {"B", "C"} and row["status"] == "PASSED"]
    admissible = [
        row
        for row in candidates
        if int(row["irrecoverable_partitions"]) == 0
        and int(row["failed_shards"]) == 0
        and int(row["http_429_count"]) == 0
    ]
    best = max(admissible, key=lambda row: float(row["validated_rows_per_second"]), default=None)
    baseline_rate = float(baseline["validated_rows_per_second"])
    ratio = float(best["validated_rows_per_second"]) / max(baseline_rate, 0.000001) if best else 0.0
    selected = best if best is not None and ratio >= PERF1_ADOPTION_RATIO else baseline
    payload = {
        "schema_version": PERF1_SCHEMA,
        "created_at_utc": utc_now(),
        "status": "SELECTED",
        "primary_metric": "validated_rows_per_second",
        "adoption_threshold_ratio": PERF1_ADOPTION_RATIO,
        "baseline_validated_rows_per_second": baseline_rate,
        "best_candidate_validated_rows_per_second": (
            float(best["validated_rows_per_second"]) if best else None
        ),
        "best_candidate_to_baseline_ratio": ratio,
        "selected_variant": selected["variant"],
        "selected_session_mode": selected["session_mode"],
        "selected_shard_geometry": selected["shard_geometry"],
        "selected_workers": int(selected["workers"]),
        "selection_reason": (
            "material throughput improvement with integrity gates passed"
            if selected is not baseline
            else "no candidate cleared the material-improvement threshold"
        ),
        "scientific_contract_changed": False,
    }
    atomic_json(paths.output_root / "PERF1_TRANSPORT_SELECTION.json", payload)
    return payload


def activity_proxy(
    store: IexMarketDataStore,
    completed: Sequence[IexPartitionRequest],
) -> dict[str, int]:
    by_symbol: dict[str, list[int]] = defaultdict(list)
    for request in completed:
        if request.timeframe != "5m":
            continue
        manifest = store.completed_manifest(request)
        assert manifest is not None
        by_symbol[request.symbol].append(int(manifest["quality"]["row_count"]))
    return {symbol: int(median(values)) for symbol, values in by_symbol.items()}


def activity_stratum(value: int | None, population: Iterable[int]) -> str:
    values = sorted(population)
    if value is None or not values:
        return "unknown"
    one_third = values[len(values) // 3]
    two_thirds = values[(2 * len(values)) // 3]
    return "lower" if value <= one_third else "medium" if value <= two_thirds else "high"


def request_from_identity(identity: Mapping[str, Any]) -> IexPartitionRequest:
    return IexPartitionRequest(
        symbol=str(identity["symbol"]),
        timeframe=str(identity["timeframe"]),
        start=datetime.fromisoformat(str(identity["start"]).replace("Z", "+00:00")),
        end=datetime.fromisoformat(str(identity["end"]).replace("Z", "+00:00")),
        year=int(identity["year"]) if identity.get("year") is not None else None,
    )


def retryable_classification(value: str) -> bool:
    return value.casefold() in {
        "missing",
        "missing_unattempted",
        "retryable_failure",
        "transient_network_failure",
        "transient_provider_failure",
        "rate_limit_failure",
        "timeout",
        "connection_failure",
        "network_failure",
    }


def percentile(values: Sequence[float] | Sequence[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def stable_rank(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def optional_json(path: Path) -> dict[str, Any]:
    try:
        return read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


__all__ = [
    "activity_proxy",
    "activity_stratum",
    "atomic_csv",
    "completed_hash_row",
    "failure_classification_counts",
    "json_sha256",
    "optional_json",
    "planned_partitions",
    "percentile",
    "read_json",
    "request_from_identity",
    "retryable_classification",
    "stable_rank",
    "select_transport",
    "scratch_ceiling",
    "tree_bytes",
    "validated_inventory",
]
