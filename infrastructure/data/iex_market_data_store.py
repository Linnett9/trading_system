"""Atomic, hash-addressed persistence for canonical IEX dataset partitions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from core.research.artifact_publication import JsonByteFormat, atomic_write_json
from core.research.iex_market_data import IexPartitionRequest


_JSON_FORMAT = JsonByteFormat(trailing_newline=True, format_id="hf_data_cutover_indent2_sort_keys_newline.v1")


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class IexMarketDataStore:
    root: Path

    def partition_dir(self, request: IexPartitionRequest) -> Path:
        timeframe = "1d" if request.timeframe == "1Day" else "5m"
        base = self.root / "raw" / timeframe / f"symbol={_safe_symbol(request.symbol)}"
        return base if request.year is None else base / f"year={request.year}"

    def parquet_path(self, request: IexPartitionRequest) -> Path:
        return self.partition_dir(request) / "bars.parquet"

    def manifest_path(self, request: IexPartitionRequest) -> Path:
        return self.partition_dir(request) / "manifest.json"

    def completed_manifest(self, request: IexPartitionRequest) -> dict[str, Any] | None:
        manifest_path = self.manifest_path(request)
        parquet_path = self.parquet_path(request)
        if not manifest_path.exists() or not parquet_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("request") != request.identity():
            raise ValueError(f"Existing IEX partition request identity mismatch: {manifest_path}")
        if manifest.get("parquet_sha256") != sha256_file(parquet_path):
            raise ValueError(f"Existing IEX partition hash mismatch: {parquet_path}")
        return manifest

    def publish(
        self,
        request: IexPartitionRequest,
        frame: pd.DataFrame,
        *,
        quality: Mapping[str, Any],
        source: Mapping[str, Any],
    ) -> dict[str, Any]:
        directory = self.partition_dir(request)
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
                raise ValueError(f"Temporary IEX partition validation failed: {temporary}")
            os.replace(temporary, parquet_path)
        finally:
            temporary.unlink(missing_ok=True)
        manifest = {
            "schema_version": "hf_data_cutover_r1.iex_raw_partition.v1",
            "status": "COMPLETE",
            "request": request.identity(),
            "quality": dict(quality),
            "source": dict(source),
            "parquet_path": str(parquet_path.resolve()),
            "parquet_bytes": parquet_path.stat().st_size,
            "parquet_sha256": sha256_file(parquet_path),
            "columns": list(frame.columns),
            "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        }
        atomic_write_json(self.manifest_path(request), manifest, json_format=_JSON_FORMAT)
        return manifest

    def write_failure(self, request: IexPartitionRequest, failure: Mapping[str, Any]) -> None:
        path = self.partition_dir(request) / "failure.json"
        payload = {
            "schema_version": "hf_data_cutover_r1.iex_raw_partition_failure.v1",
            "request": request.identity(),
            **dict(failure),
        }
        atomic_write_json(path, payload, json_format=_JSON_FORMAT)


def _safe_symbol(symbol: str) -> str:
    safe = "".join(character if character.isalnum() or character in {"-", "."} else "_" for character in symbol)
    if not safe or safe in {".", ".."}:
        raise ValueError(f"Unsafe symbol partition key: {symbol!r}")
    return safe


__all__ = ["IexMarketDataStore", "sha256_file"]
