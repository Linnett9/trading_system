from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np

from infrastructure.data.market_parquet import normalize_timeframe


ACTIVE_TIMEFRAMES = ("1Day", "1h", "5m")
BASE_COLUMNS = ("timestamp", "symbol", "timeframe", "open", "high", "low", "close", "volume")
DAILY_COLUMNS = (
    *BASE_COLUMNS,
    "return_1", "return_5", "return_10", "return_20", "return_63", "return_126", "return_252",
    "sma_20", "sma_50", "sma_100", "sma_200",
    "distance_sma_20", "distance_sma_50", "distance_sma_100", "distance_sma_200",
    "ema_12", "ema_26", "ema_50", "ema_200",
    "rsi_14", "macd", "macd_signal", "macd_histogram",
    "atr_14", "atr_pct_14",
    "realized_volatility_21", "realized_volatility_63",
    "rolling_drawdown_63", "rolling_drawdown_126", "rolling_drawdown_252",
    "distance_from_high_20", "distance_from_high_63", "distance_from_high_252",
    "volume_sma_20", "volume_ratio_20", "dollar_volume", "dollar_volume_sma_20",
    "bollinger_zscore_20", "bollinger_width_20",
    "relative_strength_spy_20", "relative_strength_spy_63", "relative_strength_spy_126",
    "relative_strength_qqq_20", "relative_strength_qqq_63", "relative_strength_qqq_126",
    "correlation_spy_63", "beta_spy_63",
    "spy_return_5", "spy_return_20", "spy_return_63", "spy_distance_sma_200",
    "spy_volatility_21", "spy_volatility_63", "spy_drawdown_63", "spy_drawdown_126",
    "qqq_return_5", "qqq_return_20", "qqq_return_63", "qqq_distance_sma_200",
    "qqq_volatility_21", "qqq_volatility_63", "qqq_drawdown_63", "qqq_drawdown_126",
    "qqq_vs_spy_relative_strength_20", "qqq_vs_spy_relative_strength_63",
    "breadth_above_sma_50", "breadth_above_sma_200", "breadth_change_5", "breadth_change_20",
)
HOURLY_COLUMNS = (
    *BASE_COLUMNS,
    "return_1", "return_3", "return_6", "return_12", "return_24", "return_72",
    "sma_6", "sma_12", "sma_24", "sma_72",
    "ema_6", "ema_12", "ema_24",
    "rsi_14", "macd", "macd_signal", "macd_histogram",
    "atr_14", "atr_pct_14",
    "realized_volatility_12", "realized_volatility_24", "realized_volatility_72",
    "volume_ratio_24",
    "relative_strength_spy_24", "relative_strength_spy_72",
    "relative_strength_qqq_24", "relative_strength_qqq_72",
)
FIVE_MIN_COLUMNS = (
    *BASE_COLUMNS,
    "return_1", "return_3", "return_6", "return_12", "return_24", "return_78",
    "sma_6", "sma_12", "sma_24", "sma_78",
    "ema_6", "ema_12", "ema_24",
    "rsi_14",
    "atr_14", "atr_pct_14",
    "realized_volatility_12", "realized_volatility_24", "realized_volatility_78",
    "volume_ratio_12", "volume_ratio_78",
)


@dataclass(frozen=True)
class FeatureBuildResult:
    timeframe: str
    path: str
    row_count: int
    column_count: int
    symbol_count: int
    first_timestamp: str | None
    last_timestamp: str | None
    file_size_bytes: int
    duplicate_key_count: int
    null_count: int
    infinite_value_count: int
    coverage: dict[str, float]


def canonical_bar_paths(data_root: str | Path, timeframes: Iterable[str] = ACTIVE_TIMEFRAMES) -> list[Path]:
    root = Path(data_root)
    wanted = {normalize_timeframe(value) for value in timeframes}
    return sorted(
        path for path in root.glob("*/*/bars.parquet")
        if path.parent.name in wanted
    )


def build_market_data_inventory(
    data_root: str | Path = "data/processed",
    report_dir: str | Path = "reports/data",
    *,
    stale_gap_days: int = 10,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for path in canonical_bar_paths(data_root):
        row = inspect_bar_file(path, stale_gap_days=stale_gap_days)
        rows.append(row)
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_dir / "market_data_inventory.parquet", compression="zstd")
    summary = summarize_market_inventory(rows, stale_gap_days=stale_gap_days)
    (output_dir / "market_data_inventory_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "market_data_inventory.md").write_text(_inventory_markdown(summary), encoding="utf-8")
    return summary


def inspect_bar_file(path: str | Path, *, stale_gap_days: int = 10) -> dict[str, Any]:
    import pyarrow.parquet as pq

    file_path = Path(path)
    parquet_file = pq.ParquetFile(file_path)
    metadata = parquet_file.metadata
    column_nulls = _metadata_null_counts(metadata)
    row_count = metadata.num_rows
    first, last = _metadata_min_max(metadata, "timestamp")
    latest_by_timeframe = {
        "1Day": date(2026, 7, 2),
        "1h": date(2026, 7, 2),
        "5m": date(2026, 7, 2),
    }
    timeframe = file_path.parent.name
    stale = False
    if last is not None:
        last_date = _to_datetime(last).date()
        stale = (latest_by_timeframe.get(timeframe, last_date) - last_date).days > stale_gap_days
    return {
        "symbol": file_path.parent.parent.name.upper(),
        "timeframe": timeframe,
        "path": str(file_path),
        "row_count": row_count,
        "first_timestamp": _iso(first) if first is not None else None,
        "last_timestamp": _iso(last) if last is not None else None,
        "duplicate_timestamp_count": 0,
        "duplicate_check_status": "not_scanned_fast_inventory_assumed_from_canonical_import",
        "null_timestamp_count": column_nulls.get("timestamp", 0),
        "null_open_count": column_nulls.get("open", 0),
        "null_high_count": column_nulls.get("high", 0),
        "null_low_count": column_nulls.get("low", 0),
        "null_close_count": column_nulls.get("close", 0),
        "null_volume_count": column_nulls.get("volume", 0),
        "is_sorted": True,
        "sorted_check_status": "not_scanned_fast_inventory_assumed_from_canonical_import",
        "file_size_bytes": file_path.stat().st_size,
        "is_stale": stale,
    }


def _metadata_null_counts(metadata: Any) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    schema_names = metadata.schema.names
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        for column_index, name in enumerate(schema_names):
            stats = row_group.column(column_index).statistics
            if stats is not None and stats.null_count is not None:
                counts[name] += int(stats.null_count)
    return counts


def _metadata_min_max(metadata: Any, column_name: str) -> tuple[Any | None, Any | None]:
    schema_names = metadata.schema.names
    if column_name not in schema_names:
        return None, None
    column_index = schema_names.index(column_name)
    minimums = []
    maximums = []
    for row_group_index in range(metadata.num_row_groups):
        stats = metadata.row_group(row_group_index).column(column_index).statistics
        if stats is None:
            continue
        if stats.min is not None:
            minimums.append(stats.min)
        if stats.max is not None:
            maximums.append(stats.max)
    return (min(minimums) if minimums else None, max(maximums) if maximums else None)


def summarize_market_inventory(rows: list[dict[str, Any]], *, stale_gap_days: int = 10) -> dict[str, Any]:
    by_timeframe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_timeframe[str(row["timeframe"])].append(row)
    summary: dict[str, Any] = {
        "source": "canonical_market_bar_inventory",
        "stale_gap_days": stale_gap_days,
        "total_files": len(rows),
        "timeframes": {},
        "research_only": True,
        "trading_impact": "none",
    }
    for timeframe, items in sorted(by_timeframe.items()):
        starts = sorted(row["first_timestamp"] for row in items if row["first_timestamp"])
        ends = sorted(row["last_timestamp"] for row in items if row["last_timestamp"])
        counts = sorted(int(row["row_count"]) for row in items)
        summary["timeframes"][timeframe] = {
            "symbol_count": len({row["symbol"] for row in items}),
            "total_rows": sum(int(row["row_count"]) for row in items),
            "total_storage_size_bytes": sum(int(row["file_size_bytes"]) for row in items),
            "earliest_timestamp": starts[0] if starts else None,
            "latest_timestamp": ends[-1] if ends else None,
            "median_start_date": starts[len(starts) // 2] if starts else None,
            "median_end_date": ends[len(ends) // 2] if ends else None,
            "median_row_count": median(counts) if counts else 0,
            "duplicate_timestamp_total": sum(int(row["duplicate_timestamp_count"]) for row in items),
            "duplicate_check_status": "not_scanned_fast_inventory_assumed_from_canonical_import",
            "null_timestamp_total": sum(int(row["null_timestamp_count"]) for row in items),
            "null_open_total": sum(int(row["null_open_count"]) for row in items),
            "null_high_total": sum(int(row["null_high_count"]) for row in items),
            "null_low_total": sum(int(row["null_low_count"]) for row in items),
            "null_close_total": sum(int(row["null_close_count"]) for row in items),
            "null_volume_total": sum(int(row["null_volume_count"]) for row in items),
            "sorted_check_status": "not_scanned_fast_inventory_assumed_from_canonical_import",
            "stale_symbol_count": sum(1 for row in items if row["is_stale"]),
        }
    return summary


def build_stock_feature_bank(
    timeframe: str,
    *,
    data_root: str | Path = "data/processed",
    output_dir: str | Path = "cache/ml/features",
    symbols: Iterable[str] | None = None,
) -> FeatureBuildResult:
    import pyarrow as pa
    import pyarrow.parquet as pq

    canonical = normalize_timeframe(timeframe)
    paths = _feature_source_paths(data_root, canonical, symbols=symbols)
    if not paths:
        raise FileNotFoundError(f"No canonical bar files found for timeframe {canonical}")
    columns = feature_columns(canonical)
    benchmark_maps = {
        name: _feature_map(_load_bars(_path_for_symbol(data_root, name, canonical)), canonical, name)
        for name in ("SPY", "QQQ")
        if _path_for_symbol(data_root, name, canonical).exists()
    }
    breadth = _breadth_maps(paths) if canonical == "1Day" else {}
    output_path = Path(output_dir) / f"stock_features_{canonical}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".parquet.tmp")
    all_rows = []
    for path in sorted(paths, key=lambda item: item.parent.parent.name.upper()):
        symbol = path.parent.parent.name.upper()
        bars = _load_bars(path)
        all_rows.extend(_feature_rows(bars, canonical, symbol, benchmark_maps, breadth))
    all_rows.sort(key=lambda row: (row["timestamp"], row["symbol"]))
    if not all_rows:
        raise RuntimeError(f"No feature rows were generated for {canonical}")
    writer = None
    try:
        for start in range(0, len(all_rows), 50_000):
            chunk = all_rows[start:start + 50_000]
            table = pa.Table.from_pylist([{name: row.get(name) for name in columns} for row in chunk], schema=_schema(columns))
            if writer is None:
                writer = pq.ParquetWriter(temporary_path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    temporary_path.replace(output_path)
    return validate_stock_feature_bank(output_path)


def validate_stock_feature_bank(path: str | Path) -> FeatureBuildResult:
    import pyarrow.parquet as pq

    feature_path = Path(path)
    table = pq.read_table(feature_path)
    data = table.to_pydict()
    keys = list(zip(data["timestamp"], data["symbol"], data["timeframe"]))
    numeric_columns = [
        name for name in table.column_names
        if name not in {"timestamp", "symbol", "timeframe"}
    ]
    infinite_count = 0
    null_count = 0
    coverage: dict[str, float] = {}
    for name in table.column_names:
        values = data[name]
        nulls = _null_count(values)
        null_count += nulls
        if name in numeric_columns:
            for value in values:
                if value is not None and not math.isfinite(float(value)):
                    infinite_count += 1
        if name not in BASE_COLUMNS:
            coverage[name] = 0.0 if not values else (len(values) - nulls) / len(values)
    timestamps = data["timestamp"]
    symbols = {str(value).upper() for value in data["symbol"]}
    timeframes = sorted({str(value) for value in data["timeframe"]})
    return FeatureBuildResult(
        timeframe=timeframes[0] if len(timeframes) == 1 else ",".join(timeframes),
        path=str(feature_path),
        row_count=table.num_rows,
        column_count=table.num_columns,
        symbol_count=len(symbols),
        first_timestamp=_iso(min(timestamps)) if timestamps else None,
        last_timestamp=_iso(max(timestamps)) if timestamps else None,
        file_size_bytes=feature_path.stat().st_size,
        duplicate_key_count=len(keys) - len(set(keys)),
        null_count=null_count,
        infinite_value_count=infinite_count,
        coverage=coverage,
    )


def write_feature_validation_report(
    paths: Iterable[str | Path],
    report_dir: str | Path = "reports/data",
) -> dict[str, Any]:
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [validate_stock_feature_bank(path).__dict__ for path in paths if Path(path).exists()]
    payload = {"source": "stock_feature_bank_validation", "datasets": results, "research_only": True, "trading_impact": "none"}
    (output_dir / "stock_feature_bank_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "stock_feature_bank_validation.md").write_text(_validation_markdown(results), encoding="utf-8")
    return payload


def feature_columns(timeframe: str) -> tuple[str, ...]:
    canonical = normalize_timeframe(timeframe)
    if canonical == "1Day":
        return DAILY_COLUMNS
    if canonical == "1h":
        return HOURLY_COLUMNS
    return FIVE_MIN_COLUMNS


def _feature_source_paths(data_root: str | Path, timeframe: str, symbols: Iterable[str] | None = None) -> list[Path]:
    requested = {symbol.upper() for symbol in symbols or []}
    root = Path(data_root)
    paths = sorted(root.glob(f"*/{timeframe}/bars.parquet"))
    if requested:
        paths = [path for path in paths if path.parent.parent.name.upper() in requested]
    return paths


def _path_for_symbol(data_root: str | Path, symbol: str, timeframe: str) -> Path:
    return Path(data_root) / symbol.upper() / timeframe / "bars.parquet"


def _load_bars(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=list(BASE_COLUMNS[:1] + BASE_COLUMNS[3:]))
    data = table.to_pydict()
    order = sorted(range(table.num_rows), key=lambda index: data["timestamp"][index])
    return {
        "timestamp": [data["timestamp"][index] for index in order],
        "open": np.array([float(data["open"][index]) for index in order], dtype=float),
        "high": np.array([float(data["high"][index]) for index in order], dtype=float),
        "low": np.array([float(data["low"][index]) for index in order], dtype=float),
        "close": np.array([float(data["close"][index]) for index in order], dtype=float),
        "volume": np.array([float(data["volume"][index]) for index in order], dtype=float),
    }


def _feature_rows(
    bars: dict[str, Any],
    timeframe: str,
    symbol: str,
    benchmark_maps: dict[str, dict[Any, dict[str, Any]]],
    breadth: dict[str, dict[Any, float | None]],
) -> list[dict[str, Any]]:
    features = _compute_features(bars, timeframe)
    timestamps = bars["timestamp"]
    rows = []
    for index, timestamp in enumerate(timestamps):
        row = {
            "timestamp": _to_datetime(timestamp),
            "symbol": symbol,
            "timeframe": timeframe,
            "open": float(bars["open"][index]),
            "high": float(bars["high"][index]),
            "low": float(bars["low"][index]),
            "close": float(bars["close"][index]),
            "volume": float(bars["volume"][index]),
        }
        for name, values in features.items():
            row[name] = _finite_or_none(values[index])
        for bench, bench_map in benchmark_maps.items():
            prefix = bench.lower()
            values = bench_map.get(timestamp, {})
            if timeframe == "1Day":
                for name in ("return_5", "return_20", "return_63", "distance_sma_200"):
                    row[f"{prefix}_{_context_name(name)}"] = values.get(name)
                row[f"{prefix}_volatility_21"] = values.get("realized_volatility_21")
                row[f"{prefix}_volatility_63"] = values.get("realized_volatility_63")
                row[f"{prefix}_drawdown_63"] = values.get("rolling_drawdown_63")
                row[f"{prefix}_drawdown_126"] = values.get("rolling_drawdown_126")
                for lookback in (20, 63, 126):
                    left = row.get(f"return_{lookback}")
                    right = values.get(f"return_{lookback}")
                    row[f"relative_strength_{prefix}_{lookback}"] = _difference(left, right)
            else:
                for lookback in ((24, 72) if timeframe == "1h" else ()):
                    row[f"relative_strength_{prefix}_{lookback}"] = _difference(row.get(f"return_{lookback}"), values.get(f"return_{lookback}"))
        if timeframe == "1Day" and "SPY" in benchmark_maps and "QQQ" in benchmark_maps:
            spy = benchmark_maps["SPY"].get(timestamp, {})
            qqq = benchmark_maps["QQQ"].get(timestamp, {})
            for lookback in (20, 63):
                row[f"qqq_vs_spy_relative_strength_{lookback}"] = _difference(qqq.get(f"return_{lookback}"), spy.get(f"return_{lookback}"))
            for name, values in breadth.items():
                row[name] = values.get(timestamp)
        rows.append(row)
    if timeframe == "1Day" and "SPY" in benchmark_maps:
        _add_rolling_spy_alignment(rows, bars, benchmark_maps["SPY"])
    return rows


def _compute_features(bars: dict[str, Any], timeframe: str) -> dict[str, np.ndarray]:
    close = bars["close"]
    high = bars["high"]
    low = bars["low"]
    volume = bars["volume"]
    result: dict[str, np.ndarray] = {}
    if timeframe == "1Day":
        return_windows = (1, 5, 10, 20, 63, 126, 252)
        sma_windows = (20, 50, 100, 200)
        ema_windows = (12, 26, 50, 200)
        vol_windows = (21, 63)
        drawdown_windows = (63, 126, 252)
        high_windows = (20, 63, 252)
    elif timeframe == "1h":
        return_windows = (1, 3, 6, 12, 24, 72)
        sma_windows = (6, 12, 24, 72)
        ema_windows = (6, 12, 24)
        vol_windows = (12, 24, 72)
        drawdown_windows = ()
        high_windows = ()
    else:
        return_windows = (1, 3, 6, 12, 24, 78)
        sma_windows = (6, 12, 24, 78)
        ema_windows = (6, 12, 24)
        vol_windows = (12, 24, 78)
        drawdown_windows = ()
        high_windows = ()
    returns_1 = _returns(close, 1)
    for window in return_windows:
        result[f"return_{window}"] = _returns(close, window)
    for window in sma_windows:
        sma = _rolling_mean(close, window)
        result[f"sma_{window}"] = sma
        if timeframe == "1Day":
            result[f"distance_sma_{window}"] = _ratio_minus_one(close, sma)
    for window in ema_windows:
        result[f"ema_{window}"] = _ema(close, window)
    result["rsi_14"] = _rsi(close, 14)
    if timeframe in {"1Day", "1h"}:
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        macd = ema12 - ema26
        signal = _ema(macd, 9, allow_nan_prefix=True)
        result["macd"] = macd
        result["macd_signal"] = signal
        result["macd_histogram"] = macd - signal
    atr = _atr(high, low, close, 14)
    result["atr_14"] = atr
    result["atr_pct_14"] = _safe_divide(atr, close)
    for window in vol_windows:
        result[f"realized_volatility_{window}"] = _rolling_std(returns_1, window)
    for window in drawdown_windows:
        result[f"rolling_drawdown_{window}"] = _rolling_drawdown(close, window)
    for window in high_windows:
        result[f"distance_from_high_{window}"] = _ratio_minus_one(close, _rolling_max(close, window))
    if timeframe == "1Day":
        volume_sma = _rolling_mean(volume, 20)
        dollar_volume = close * volume
        result["volume_sma_20"] = volume_sma
        result["volume_ratio_20"] = _safe_divide(volume, volume_sma)
        result["dollar_volume"] = dollar_volume
        result["dollar_volume_sma_20"] = _rolling_mean(dollar_volume, 20)
        sma20 = result["sma_20"]
        std20 = _rolling_std(close, 20)
        result["bollinger_zscore_20"] = _safe_divide(close - sma20, std20)
        result["bollinger_width_20"] = _safe_divide(4.0 * std20, sma20)
    elif timeframe == "1h":
        result["volume_ratio_24"] = _safe_divide(volume, _rolling_mean(volume, 24))
    else:
        result["volume_ratio_12"] = _safe_divide(volume, _rolling_mean(volume, 12))
        result["volume_ratio_78"] = _safe_divide(volume, _rolling_mean(volume, 78))
    return result


def _feature_map(bars: dict[str, Any], timeframe: str, symbol: str) -> dict[Any, dict[str, Any]]:
    features = _compute_features(bars, timeframe)
    output = {}
    for index, timestamp in enumerate(bars["timestamp"]):
        output[timestamp] = {name: _finite_or_none(values[index]) for name, values in features.items()}
    return output


def _breadth_maps(paths: list[Path]) -> dict[str, dict[Any, float | None]]:
    counts: dict[Any, int] = defaultdict(int)
    above50: dict[Any, int] = defaultdict(int)
    above200: dict[Any, int] = defaultdict(int)
    for path in paths:
        bars = _load_bars(path)
        close = bars["close"]
        sma50 = _rolling_mean(close, 50)
        sma200 = _rolling_mean(close, 200)
        for index, timestamp in enumerate(bars["timestamp"]):
            counts[timestamp] += 1
            if math.isfinite(sma50[index]) and close[index] > sma50[index]:
                above50[timestamp] += 1
            if math.isfinite(sma200[index]) and close[index] > sma200[index]:
                above200[timestamp] += 1
    breadth50 = {ts: above50[ts] / count if count else None for ts, count in counts.items()}
    breadth200 = {ts: above200[ts] / count if count else None for ts, count in counts.items()}
    ordered = sorted(counts)
    change5 = _change_map(ordered, breadth200, 5)
    change20 = _change_map(ordered, breadth200, 20)
    return {
        "breadth_above_sma_50": breadth50,
        "breadth_above_sma_200": breadth200,
        "breadth_change_5": change5,
        "breadth_change_20": change20,
    }


def _add_rolling_spy_alignment(rows: list[dict[str, Any]], bars: dict[str, Any], spy_map: dict[Any, dict[str, Any]]) -> None:
    spy_returns = np.array([spy_map.get(ts, {}).get("return_1") or np.nan for ts in bars["timestamp"]], dtype=float)
    returns = _returns(bars["close"], 1)
    corr = _rolling_corr(returns, spy_returns, 63)
    beta = _rolling_beta(returns, spy_returns, 63)
    for index, row in enumerate(rows):
        row["correlation_spy_63"] = _finite_or_none(corr[index])
        row["beta_spy_63"] = _finite_or_none(beta[index])


def _returns(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if len(values) > window:
        base = values[:-window]
        valid = base != 0
        out[window:][valid] = values[window:][valid] / base[valid] - 1.0
    return out


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if len(values) < window:
        return out
    csum = np.cumsum(np.insert(values, 0, 0.0))
    out[window - 1:] = (csum[window:] - csum[:-window]) / window
    return out


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for index in range(window - 1, len(values)):
        sample = values[index - window + 1:index + 1]
        sample = sample[np.isfinite(sample)]
        if len(sample) == window:
            out[index] = float(np.std(sample))
    return out


def _rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for index in range(window - 1, len(values)):
        out[index] = float(np.max(values[index - window + 1:index + 1]))
    return out


def _rolling_drawdown(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for index in range(window - 1, len(values)):
        sample = values[index - window + 1:index + 1]
        peak = np.maximum.accumulate(sample)
        out[index] = float(np.min(sample / peak - 1.0))
    return out


def _ema(values: np.ndarray, period: int, *, allow_nan_prefix: bool = False) -> np.ndarray:
    out = np.full(len(values), np.nan)
    clean = np.array(values, dtype=float)
    if allow_nan_prefix:
        valid_indices = np.where(np.isfinite(clean))[0]
        if len(valid_indices) < period:
            return out
        start = valid_indices[0]
        seed_end = start + period
    else:
        start = 0
        seed_end = period
    if len(clean) < seed_end or np.any(~np.isfinite(clean[start:seed_end])):
        return out
    multiplier = 2.0 / (period + 1.0)
    value = float(np.mean(clean[start:seed_end]))
    out[seed_end - 1] = value
    for index in range(seed_end, len(clean)):
        if not math.isfinite(clean[index]):
            continue
        value = (clean[index] - value) * multiplier + value
        out[index] = value
    return out


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    if len(close) < period + 1:
        return out
    changes = np.diff(close)
    gains = np.where(changes > 0, changes, 0.0)
    losses = np.where(changes < 0, -changes, 0.0)
    avg_gain = _rolling_mean(gains, period)
    avg_loss = _rolling_mean(losses, period)
    for index in range(period, len(close)):
        loss = avg_loss[index - 1]
        gain = avg_gain[index - 1]
        if not math.isfinite(loss) or not math.isfinite(gain):
            continue
        out[index] = 100.0 if loss == 0 else 100.0 - (100.0 / (1.0 + gain / loss))
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    if len(close) < period + 1:
        return out
    tr = np.full(len(close), np.nan)
    for index in range(1, len(close)):
        tr[index] = max(high[index] - low[index], abs(high[index] - close[index - 1]), abs(low[index] - close[index - 1]))
    out[period:] = _rolling_mean(tr[1:], period)[period - 1:]
    return out


def _rolling_corr(left: np.ndarray, right: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(left), np.nan)
    for index in range(window - 1, len(left)):
        a = left[index - window + 1:index + 1]
        b = right[index - window + 1:index + 1]
        mask = np.isfinite(a) & np.isfinite(b)
        if np.sum(mask) == window and np.std(a[mask]) > 0 and np.std(b[mask]) > 0:
            out[index] = float(np.corrcoef(a[mask], b[mask])[0, 1])
    return out


def _rolling_beta(left: np.ndarray, right: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(left), np.nan)
    for index in range(window - 1, len(left)):
        a = left[index - window + 1:index + 1]
        b = right[index - window + 1:index + 1]
        mask = np.isfinite(a) & np.isfinite(b)
        if np.sum(mask) == window:
            variance = float(np.var(b[mask]))
            if variance > 0:
                out[index] = float(np.cov(a[mask], b[mask], bias=True)[0, 1] / variance)
    return out


def _ratio_minus_one(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = _safe_divide(left, right) - 1.0
    out[~np.isfinite(out)] = np.nan
    return out


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(len(numerator), np.nan)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    np.divide(numerator, denominator, out=out, where=valid)
    return out


def _difference(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    value = float(left) - float(right)
    return value if math.isfinite(value) else None


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _context_name(name: str) -> str:
    return name if name.startswith("distance") else name


def _change_map(timestamps: list[Any], values: dict[Any, float | None], window: int) -> dict[Any, float | None]:
    out = {}
    for index, timestamp in enumerate(timestamps):
        if index < window or values.get(timestamp) is None or values.get(timestamps[index - window]) is None:
            out[timestamp] = None
        else:
            out[timestamp] = float(values[timestamp]) - float(values[timestamps[index - window]])
    return out


def _schema(columns: tuple[str, ...]):
    import pyarrow as pa

    fields = []
    for name in columns:
        if name == "timestamp":
            fields.append(pa.field(name, pa.timestamp("us", tz="UTC")))
        elif name in {"symbol", "timeframe"}:
            fields.append(pa.field(name, pa.string()))
        else:
            fields.append(pa.field(name, pa.float64()))
    return pa.schema(fields)


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: Any) -> str:
    return _to_datetime(value).isoformat()


def _min_iso(left: str | None, right: Any) -> str:
    value = _iso(right)
    return value if left is None or value < left else left


def _max_iso(left: str | None, right: Any) -> str:
    value = _iso(right)
    return value if left is None or value > left else left


def _null_count(values: Iterable[Any]) -> int:
    return sum(value is None for value in values)


def _inventory_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Market Data Inventory",
        "",
        "Duplicate and sorted timestamp checks use fast-inventory canonical assumptions unless the status fields say otherwise.",
        "",
        "| Timeframe | Symbols | Rows | Earliest | Latest | Duplicates | Duplicate Check | Sorted Check | Stale |",
        "|---|---:|---:|---|---|---:|---|---|---:|",
    ]
    for timeframe, item in summary["timeframes"].items():
        lines.append(
            f"| {timeframe} | {item['symbol_count']} | {item['total_rows']} | "
            f"{item['earliest_timestamp']} | {item['latest_timestamp']} | "
            f"{item['duplicate_timestamp_total']} | {item['duplicate_check_status']} | "
            f"{item['sorted_check_status']} | {item['stale_symbol_count']} |"
        )
    return "\n".join(lines) + "\n"


def _validation_markdown(results: list[dict[str, Any]]) -> str:
    lines = ["# Stock Feature Bank Validation", "", "| Timeframe | Path | Rows | Columns | Symbols | First | Last | Duplicate Keys | Infinite |", "|---|---|---:|---:|---:|---|---|---:|---:|"]
    for item in results:
        lines.append(
            f"| {item['timeframe']} | {item['path']} | {item['row_count']} | "
            f"{item['column_count']} | {item['symbol_count']} | {item['first_timestamp']} | "
            f"{item['last_timestamp']} | {item['duplicate_key_count']} | {item['infinite_value_count']} |"
        )
    return "\n".join(lines) + "\n"
