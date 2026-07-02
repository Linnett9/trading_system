from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from core.interfaces.data_feed import IDataFeed
from core.research.ml.audits.benchmark_relative_validation_baselines import _canonical_schedule


def _output_dir(config: dict[str, Any]) -> Path:
    return Path(config.get("ml", {}).get("output_dir", "reports/ml/regime_transformer_meta_ensemble_v1"))

def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}

def _load_required_closes(
    config: dict[str, Any],
    data_feed: IDataFeed,
    canonical: dict[str, Any],
) -> dict[str, dict[str, float]]:
    schedule = _canonical_schedule(canonical)
    if not schedule:
        return {}
    symbols = {"SPY", "QQQ"}
    symbols.update(
        str(symbol).upper()
        for row in schedule
        for symbol in row.get("selected_symbols", [])
    )
    start = datetime.fromisoformat(schedule[0]["rebalance_date"][:10])
    end = datetime.fromisoformat(schedule[-1]["outcome_end_date"][:10])
    output = {}
    for symbol in sorted(symbols):
        try:
            candles = data_feed.get_historical_bars(symbol, "1Day", start, end)
        except (FileNotFoundError, ValueError):
            continue
        output[symbol] = {
            candle.timestamp.date().isoformat(): float(candle.close)
            for candle in candles
        }
    return output
