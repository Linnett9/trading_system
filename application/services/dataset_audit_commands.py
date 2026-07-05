from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_REQUIRED_TIMEFRAMES = ("5m", "1h")
KNOWN_TIMEFRAMES = ("5m", "1h", "1Day")


def run_dataset_audit(config: dict[str, Any]) -> None:
    ml_config = config.get("ml", {})
    market_config = ml_config.get("market_data", {})
    required_timeframes = _required_timeframes(market_config)
    optional_timeframes = tuple(
        timeframe
        for timeframe in KNOWN_TIMEFRAMES
        if timeframe not in required_timeframes
    )
    processed_root = Path(
        market_config.get(
            "processed_root",
            ml_config.get("parquet_dir", config.get("backtest", {}).get("data_dir", "data/processed")),
        )
    )

    symbol_dirs = sorted(
        path for path in processed_root.iterdir()
        if path.is_dir()
    ) if processed_root.exists() else []

    present_by_timeframe = {timeframe: 0 for timeframe in KNOWN_TIMEFRAMES}
    missing_by_timeframe = {timeframe: 0 for timeframe in KNOWN_TIMEFRAMES}
    missing_parquet_files = 0
    complete_symbols = 0
    partial_symbols = 0

    for symbol_dir in symbol_dirs:
        complete = True
        for timeframe in KNOWN_TIMEFRAMES:
            parquet_path = symbol_dir / timeframe / "bars.parquet"
            if parquet_path.exists():
                present_by_timeframe[timeframe] += 1
                continue
            missing_by_timeframe[timeframe] += 1
            if timeframe in required_timeframes:
                missing_parquet_files += 1
                complete = False
        if complete:
            complete_symbols += 1
        else:
            partial_symbols += 1

    print("\nDATASET AUDIT")
    print(f"processed_root: {processed_root}")
    print(f"required_timeframes: {', '.join(required_timeframes)}")
    print(f"optional_timeframes: {', '.join(optional_timeframes) or 'none'}")
    print(f"total_symbols: {len(symbol_dirs)}")
    print(f"complete_symbols: {complete_symbols}")
    print(f"partial_symbols: {partial_symbols}")
    print(f"missing_parquet_files: {missing_parquet_files}")
    for timeframe in KNOWN_TIMEFRAMES:
        print(f"present_{timeframe}: {present_by_timeframe[timeframe]}")
    for timeframe in KNOWN_TIMEFRAMES:
        print(f"missing_{timeframe}: {missing_by_timeframe[timeframe]}")


def _required_timeframes(market_config: dict[str, Any]) -> tuple[str, ...]:
    audit_config = market_config.get("dataset_audit", {})
    configured = audit_config.get("required_timeframes")
    if configured:
        return tuple(str(timeframe) for timeframe in configured)
    enabled = [
        str(timeframe)
        for timeframe in market_config.get("enabled_timeframes", [])
        if str(timeframe) != "1Day"
    ]
    return tuple(enabled or DEFAULT_REQUIRED_TIMEFRAMES)
