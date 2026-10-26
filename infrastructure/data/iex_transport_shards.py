"""Atomic temporary acquisition shards for canonical IEX partition assembly."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from core.research.artifact_publication import JsonByteFormat, atomic_write_json
from core.research.iex_market_data import IexPartitionRequest
from infrastructure.data.iex_market_data_store import sha256_file


_JSON_FORMAT = JsonByteFormat(
    trailing_newline=True,
    format_id="hf_data_cutover_indent2_sort_keys_newline.v1",
)


@dataclass(frozen=True, slots=True)
class IexTransportShardRequest:
    canonical: IexPartitionRequest
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("IEX transport shard bounds must be timezone-aware")
        if not self.canonical.start <= self.start < self.end <= self.canonical.end:
            raise ValueError("IEX transport shard falls outside its canonical partition")

    @property
    def shard_id(self) -> str:
        digest = hashlib.sha256(
            json.dumps(self.identity(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        return f"{self.start:%Y%m}-{digest}"

    def identity(self) -> dict[str, Any]:
        return {
            "canonical_partition_id": self.canonical.partition_id,
            "symbol": self.canonical.symbol,
            "timeframe": self.canonical.timeframe,
            "start": _utc_iso(self.start),
            "end": _utc_iso(self.end),
            "feed": "IEX",
            "adjustment": "raw",
            "currency": "USD",
        }

    def validation_request(self) -> IexPartitionRequest:
        return IexPartitionRequest(
            symbol=self.canonical.symbol,
            timeframe=self.canonical.timeframe,
            start=self.start,
            end=self.end,
            year=self.canonical.year,
        )


def plan_monthly_transport_shards(
    request: IexPartitionRequest,
) -> tuple[IexTransportShardRequest, ...]:
    if request.timeframe != "5m":
        return (IexTransportShardRequest(request, request.start, request.end),)
    shards: list[IexTransportShardRequest] = []
    cursor = request.start
    while cursor < request.end:
        next_month = (
            datetime(cursor.year + 1, 1, 1, tzinfo=timezone.utc)
            if cursor.month == 12
            else datetime(cursor.year, cursor.month + 1, 1, tzinfo=timezone.utc)
        )
        shard_end = min(next_month, request.end)
        shards.append(IexTransportShardRequest(request, cursor, shard_end))
        cursor = shard_end
    return tuple(shards)


class IexTransportShardStore:
    """Persist validated shards without granting them canonical authority."""

    def __init__(self, root: Path, *, scratch_ceiling_bytes: int) -> None:
        if scratch_ceiling_bytes <= 0:
            raise ValueError("scratch_ceiling_bytes must be positive")
        self.root = root
        self.scratch_ceiling_bytes = int(scratch_ceiling_bytes)

    def shard_dir(self, request: IexTransportShardRequest) -> Path:
        return self.root / request.canonical.partition_id / request.shard_id

    def parquet_path(self, request: IexTransportShardRequest) -> Path:
        return self.shard_dir(request) / "bars.parquet"

    def manifest_path(self, request: IexTransportShardRequest) -> Path:
        return self.shard_dir(request) / "manifest.json"

    def completed_manifest(self, request: IexTransportShardRequest) -> dict[str, Any] | None:
        manifest_path = self.manifest_path(request)
        parquet_path = self.parquet_path(request)
        if not manifest_path.exists() or not parquet_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("request") != request.identity():
            raise ValueError(f"IEX transport shard identity mismatch: {manifest_path}")
        if manifest.get("parquet_sha256") != sha256_file(parquet_path):
            raise ValueError(f"IEX transport shard hash mismatch: {parquet_path}")
        return manifest

    def publish(
        self,
        request: IexTransportShardRequest,
        frame: pd.DataFrame,
        *,
        quality: Mapping[str, Any],
        source: Mapping[str, Any],
    ) -> dict[str, Any]:
        existing = self.completed_manifest(request)
        if existing is not None:
            return existing
        estimated_bytes = max(int(frame.memory_usage(index=True, deep=True).sum()), 1)
        self._assert_capacity(estimated_bytes)
        directory = self.shard_dir(request)
        directory.mkdir(parents=True, exist_ok=True)
        parquet_path = self.parquet_path(request)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{parquet_path.name}.", suffix=".tmp", dir=directory
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            frame.to_parquet(temporary, index=False, compression="zstd")
            persisted = pd.read_parquet(temporary)
            if len(persisted) != len(frame) or tuple(persisted.columns) != tuple(frame.columns):
                raise ValueError(f"Temporary shard validation failed: {temporary}")
            os.replace(temporary, parquet_path)
        finally:
            temporary.unlink(missing_ok=True)
        manifest = {
            "schema_version": "hf_data_cutover_r1.perf1_transport_shard.v1",
            "status": "COMPLETE",
            "canonical_authority": False,
            "request": request.identity(),
            "quality": dict(quality),
            "source": dict(source),
            "parquet_path": str(parquet_path.resolve()),
            "parquet_bytes": parquet_path.stat().st_size,
            "parquet_sha256": sha256_file(parquet_path),
        }
        atomic_write_json(self.manifest_path(request), manifest, json_format=_JSON_FORMAT)
        return manifest

    def load_verified(self, request: IexTransportShardRequest) -> pd.DataFrame:
        if self.completed_manifest(request) is None:
            raise ValueError(f"IEX transport shard is incomplete: {request.shard_id}")
        return pd.read_parquet(self.parquet_path(request))

    def bytes_used(self) -> int:
        if not self.root.exists():
            return 0
        return sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())

    def cleanup_partition(self, request: IexPartitionRequest) -> None:
        target = (self.root / request.partition_id).resolve()
        root = self.root.resolve()
        if target.parent != root:
            raise ValueError(f"Refusing unexpected shard cleanup target: {target}")
        if target.exists():
            shutil.rmtree(target)

    def completed_shard_count(self, requests: Iterable[IexTransportShardRequest]) -> int:
        return sum(self.completed_manifest(request) is not None for request in requests)

    def _assert_capacity(self, incoming_bytes: int) -> None:
        current = self.bytes_used()
        if current + incoming_bytes > self.scratch_ceiling_bytes:
            raise OSError(
                "PERF1 scratch ceiling breached: "
                f"current={current}, incoming={incoming_bytes}, ceiling={self.scratch_ceiling_bytes}"
            )


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "IexTransportShardRequest",
    "IexTransportShardStore",
    "plan_monthly_transport_shards",
]
