from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from core.research.ml.features import MLFeatureBuildResult
from core.research.ml.features import write_feature_rows as write_ml_feature_rows


class MLFeatureCache:
    """Manage ML feature-row cache files and cache metadata."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = config

    def load_feature_rows(
        self,
        path: Path,
        cache_key: str,
    ) -> MLFeatureBuildResult | None:
        if not bool(self._ml_config().get("cache_feature_rows", False)):
            return None
        metadata = self.read_metadata(path)
        if metadata.get("cache_key") != cache_key or not path.exists():
            return None
        rows = self.read_csv_rows(path)
        return MLFeatureBuildResult(
            rows=rows,
            dropped_rows=int(metadata.get("dropped_rows_insufficient_lookback", 0)),
            date_range=self._feature_date_range(rows),
        )

    def write_feature_rows(
        self,
        path: Path,
        feature_result: MLFeatureBuildResult,
        cache_key: str,
    ) -> None:
        if not bool(self._ml_config().get("cache_feature_rows", False)):
            return
        write_ml_feature_rows(path, feature_result.rows)
        self.write_metadata(
            path,
            cache_key,
            {
                "cache_type": "historical_feature_rows",
                "row_count": len(feature_result.rows),
                "date_range": feature_result.date_range,
                "dropped_rows_insufficient_lookback": feature_result.dropped_rows,
                "research_only": True,
                "trading_impact": "none",
            },
        )

    def load_expanded_rebalance_rows(
        self,
        path: Path,
        cache_key: str,
        dropped_rows: int,
    ) -> MLFeatureBuildResult | None:
        metadata = self.read_metadata(path)
        if metadata.get("cache_key") != cache_key or not path.exists():
            return None
        rows = self.read_csv_rows(path)
        return MLFeatureBuildResult(
            rows=rows,
            dropped_rows=dropped_rows,
            date_range=self._feature_date_range(rows),
        )

    def feature_cache_key(
        self,
        symbols: list[str],
        benchmark_symbols: tuple[Any, ...],
        lookback_days: int,
        candles_by_symbol: dict[str, list[Any]],
    ) -> str:
        return self.hash_payload({
            "cache_version": 1,
            "cache_type": "historical_feature_rows",
            "symbols": sorted(str(symbol).upper() for symbol in symbols),
            "benchmark_symbols": [str(symbol).upper() for symbol in benchmark_symbols],
            "lookback_days": int(lookback_days),
            "history": self.candles_cache_summary(candles_by_symbol),
            "feature_builder": "HistoricalFeatureBuilder",
        })

    def read_metadata(self, path: Path) -> dict[str, Any]:
        metadata_path = self.metadata_path(path)
        if not metadata_path.exists():
            return {}
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def write_metadata(
        self,
        path: Path,
        cache_key: str,
        metadata: dict[str, Any],
    ) -> None:
        metadata_path = self.metadata_path(path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **metadata,
            "cache_key": cache_key,
            "config_hash": self.hash_payload(self._config),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def metadata_path(path: Path) -> Path:
        return path.with_suffix(path.suffix + ".metadata.json")

    @staticmethod
    def read_csv_rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def rows_hash(rows: list[dict[str, Any]]) -> str:
        return MLFeatureCache.hash_payload(rows)

    @staticmethod
    def candles_cache_summary(
        candles_by_symbol: dict[str, list[Any]],
    ) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        for symbol, candles in sorted(candles_by_symbol.items()):
            ordered = sorted(candles, key=lambda item: item.timestamp)
            summary[symbol] = {
                "count": len(ordered),
                "start": (
                    ordered[0].timestamp.date().isoformat()
                    if ordered
                    else None
                ),
                "end": (
                    ordered[-1].timestamp.date().isoformat()
                    if ordered
                    else None
                ),
                "first_close": float(ordered[0].close) if ordered else None,
                "last_close": float(ordered[-1].close) if ordered else None,
            }
        return summary

    @staticmethod
    def hash_payload(payload: Any) -> str:
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _feature_date_range(
        rows: list[dict[str, str]],
    ) -> tuple[str, str] | None:
        if not rows:
            return None
        return (
            str(rows[0].get("feature_date", "")),
            str(rows[-1].get("feature_date", "")),
        )

    def _ml_config(self) -> Mapping[str, Any]:
        return self._config.get("ml", {}) or {}
