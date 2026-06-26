from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import yaml

from core.research.ml.data_inventory import build_data_inventory


@dataclass(frozen=True)
class UniverseBuildResult:
    path: Path
    name: str
    symbol_count: int
    available_count: int


def build_universe_files(
    inventory_path: str | Path = "reports/ml/data_inventory.json",
    output_dir: str | Path = "data/reference/universes",
    parquet_dir: str | Path = "data/processed/stooq_parquet",
    inventory_output_dir: str | Path = "reports/ml",
    min_history_years: int = 9,
    max_latest_gap_days: int = 14,
    min_average_dollar_volume_252d: float = 50_000_000,
) -> list[UniverseBuildResult]:
    path = Path(inventory_path)
    if not path.exists():
        build_data_inventory(
            parquet_dir=parquet_dir,
            output_dir=inventory_output_dir,
            min_history_years=min_history_years,
            max_latest_gap_days=max_latest_gap_days,
            min_average_dollar_volume_252d=min_average_dollar_volume_252d,
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    symbols = list(payload.get("symbols", []))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    current_symbols = sorted(str(item["symbol"]) for item in symbols)
    passed = [
        item for item in symbols
        if item.get("included") or (
            item.get("passes_min_history_years")
            and item.get("passes_latest_date_check")
            and item.get("passes_liquidity_check")
        )
    ]
    ranked = sorted(
        passed,
        key=lambda item: float(item.get("average_dollar_volume_252d") or 0.0),
        reverse=True,
    )

    results = [
        _write_universe(
            output_path / "current_32.yaml",
            name="current_32",
            symbols=current_symbols,
            available_count=len(current_symbols),
            filters={},
        )
    ]
    filters = {
        "min_history_years": min_history_years,
        "max_latest_gap_days": max_latest_gap_days,
        "min_average_dollar_volume_252d": min_average_dollar_volume_252d,
    }
    for target in (100, 250, 500):
        selected = [str(item["symbol"]) for item in ranked[:target]]
        results.append(
            _write_universe(
                output_path / f"us_liquid_{target}.yaml",
                name=f"us_liquid_{target}",
                symbols=selected,
                available_count=len(ranked),
                filters=filters,
            )
        )
    return results


def _write_universe(
    path: Path,
    name: str,
    symbols: list[str],
    available_count: int,
    filters: dict[str, Any],
) -> UniverseBuildResult:
    payload = {
        "name": name,
        "source": "stooq_parquet_inventory",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "filters": filters,
        "available_count": available_count,
        "symbols": symbols,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return UniverseBuildResult(
        path=path,
        name=name,
        symbol_count=len(symbols),
        available_count=available_count,
    )
