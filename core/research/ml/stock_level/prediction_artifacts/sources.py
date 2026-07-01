from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import yaml

from core.research.framework.data import CsvRowRepository


def _universe_symbols(config: dict[str, Any]) -> list[str]:
    ml_config = config.get("ml", {})
    expanded_config = ml_config.get("expanded_rebalance_dataset", {}) or {}
    paths = ml_config.get("stock_alpha_artifact_universe_paths") or expanded_config.get("universe_paths") or [
        "data/reference/universes/current_32.yaml"
    ]
    symbols: list[str] = []
    for raw_path in paths:
        path = Path(str(raw_path))
        if not path.exists():
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        symbols.extend(str(symbol).upper() for symbol in payload.get("symbols", []))
    unique = list(dict.fromkeys(symbols))
    required = [
        str(symbol).upper()
        for symbol in ml_config.get("stock_alpha_dev_required_symbols", [ml_config.get("stock_ranker_market_symbol", "SPY")])
    ]
    max_symbols = ml_config.get("stock_alpha_artifact_max_symbols", expanded_config.get("max_symbols"))
    method = str(ml_config.get("stock_alpha_artifact_symbol_sample_method", "liquidity_ranked")).lower()
    return _select_symbols(unique, max_symbols=max_symbols, required_symbols=required, method=method)


def _select_symbols(
    symbols: list[str],
    *,
    max_symbols: Any,
    required_symbols: list[str],
    method: str,
) -> list[str]:
    unique = list(dict.fromkeys(str(symbol).upper() for symbol in symbols if symbol))
    required = [symbol for symbol in dict.fromkeys(required_symbols) if symbol in unique]
    remainder = [symbol for symbol in unique if symbol not in set(required)]
    if method == "deterministic_hash":
        remainder = sorted(remainder, key=lambda symbol: hashlib.sha256(symbol.encode("utf-8")).hexdigest())
    elif method not in {"liquidity_ranked", "input_order"}:
        raise ValueError("ml.stock_alpha_artifact_symbol_sample_method must be liquidity_ranked, input_order, or deterministic_hash")
    selected = [*required, *remainder]
    return selected[: int(max_symbols)] if max_symbols else selected


def _load_closes_by_symbol(config: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    parquet_dir = Path(
        str(config.get("ml", {}).get("stooq_parquet_dir", "data/processed/stooq_parquet"))
    )
    closes = {}
    ml = config.get("ml", {})
    required = ml.get("stock_alpha_dev_required_symbols", [ml.get("stock_ranker_market_symbol", "SPY")])
    symbols = {*_universe_symbols(config), *(str(symbol).upper() for symbol in required)}
    for symbol in sorted(symbols):
        path = parquet_dir / f"{symbol.upper()}.parquet"
        if path.exists():
            closes[symbol.upper()] = _read_parquet_closes(path)
    return closes


def _read_parquet_closes(path: Path) -> dict[str, dict[str, float]]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {}
    table = pq.read_table(path, columns=["timestamp", "close", "volume"])
    data = table.to_pydict()
    closes = {}
    dollar_volume = {}
    for value, close, volume in zip(data["timestamp"], data["close"], data["volume"]):
        if close is None or not math.isfinite(float(close)):
            continue
        date = value.date().isoformat() if hasattr(value, "date") else str(value)[:10]
        closes[date] = float(close)
        if volume is not None and math.isfinite(float(volume)):
            dollar_volume[date] = float(close) * float(volume)
    return {"close": closes, "dollar_volume": dollar_volume}


def _output_dir(config: dict[str, Any]) -> Path:
    return Path(
        config.get("ml", {}).get(
            "output_dir",
            Path(config.get("reports", {}).get("ml_dir", "reports/ml"))
            / "regime_transformer_meta_ensemble_v1",
        )
    )


def _expanded_dataset_path(config: dict[str, Any]) -> Path:
    return Path(
        config.get("ml", {}).get(
            "expanded_rebalance_dataset_path",
            Path(config.get("cache", {}).get("ml_dir", "cache/ml"))
            / "expanded_rebalance_dataset.csv",
        )
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    return CsvRowRepository().read(path)
