"""Atomic partition publication for derived IEX research artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow.parquet as pq

from core.research.artifact_publication import JsonByteFormat, atomic_write_json
from infrastructure.data.iex_market_data_store import sha256_file


_JSON_FORMAT = JsonByteFormat(trailing_newline=True, format_id="hf_data_cutover_indent2_sort_keys_newline.v1")


@dataclass(frozen=True, slots=True)
class IexResearchArtifactStore:
    root: Path

    def completed(self, relative_directory: Path, identity: Mapping[str, Any]) -> dict[str, Any] | None:
        directory = self.root / relative_directory
        parquet_path = directory / "part.parquet"
        manifest_path = directory / "manifest.json"
        if not parquet_path.exists() or not manifest_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != dict(identity):
            raise ValueError(f"IEX research artifact identity mismatch: {manifest_path}")
        if manifest.get("parquet_sha256") != sha256_file(parquet_path):
            raise ValueError(f"IEX research artifact hash mismatch: {parquet_path}")
        return manifest

    def publish(
        self,
        relative_directory: Path,
        frame: pd.DataFrame,
        *,
        identity: Mapping[str, Any],
        lineage: Mapping[str, Any],
        quality: Mapping[str, Any],
    ) -> dict[str, Any]:
        directory = self.root / relative_directory
        directory.mkdir(parents=True, exist_ok=True)
        parquet_path = directory / "part.parquet"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{parquet_path.name}.", suffix=".tmp", dir=directory
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            frame.to_parquet(temporary, index=False, compression="zstd")
            persisted_rows = int(pq.ParquetFile(temporary).metadata.num_rows)
            if persisted_rows != len(frame):
                raise ValueError(f"IEX research artifact row-count validation failed: {temporary}")
            os.replace(temporary, parquet_path)
        finally:
            temporary.unlink(missing_ok=True)
        manifest = {
            "schema_version": "hf_data_cutover_r1.iex_research_partition.v1",
            "status": "COMPLETE",
            "identity": dict(identity),
            "lineage": dict(lineage),
            "quality": dict(quality),
            "row_count": int(len(frame)),
            "columns": list(frame.columns),
            "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
            "parquet_path": str(parquet_path.resolve()),
            "parquet_bytes": parquet_path.stat().st_size,
            "parquet_sha256": sha256_file(parquet_path),
        }
        atomic_write_json(directory / "manifest.json", manifest, json_format=_JSON_FORMAT)
        return manifest


__all__ = ["IexResearchArtifactStore"]
