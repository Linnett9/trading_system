from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EVENT_FEATURES = Path(
    "reports/ml/benchmark/regime_transformer_meta_ensemble_v1/"
    "news_transformer_features_120mo_v1/news_transformer_event_features.csv"
)
DEFAULT_REPORT_DIR = Path(
    "reports/ml/benchmark/regime_transformer_meta_ensemble_v1"
)
INVENTORY_FILENAME = "news_transformer_market_data_inventory.json"
IMPORT_PLAN_FILENAME = "news_transformer_market_data_import_plan.json"
TIMEFRAMES = ("daily", "hourly", "5_minute", "15_minute")
PRICE_EXTENSIONS = {".csv", ".parquet", ".txt"}
TIMESTAMP_COLUMNS = ("timestamp", "datetime", "date", "Date", "Timestamp")
SYMBOL_COLUMNS = ("symbol", "ticker", "Symbol", "Ticker", "<TICKER>")
OHLC_COLUMNS = {
    "open": ("open", "Open", "<OPEN>", "o"),
    "high": ("high", "High", "<HIGH>", "h"),
    "low": ("low", "Low", "<LOW>", "l"),
    "close": ("close", "Close", "<CLOSE>", "c", "adj_close", "adjusted_close"),
}
VOLUME_COLUMNS = ("volume", "Volume", "<VOL>", "vol", "v")
ADJUSTED_COLUMNS = ("adj_close", "adjusted_close", "Adj Close", "AdjClose")


@dataclass
class FileAudit:
    path: str
    symbol: str | None
    rows: int = 0
    readable: bool = False
    invalid_reason: str | None = None
    columns: list[str] = field(default_factory=list)
    date_min: str | None = None
    date_max: str | None = None
    symbols: set[str] = field(default_factory=set)
    required_columns_present: bool = False
    timezone_values: set[str] = field(default_factory=set)
    duplicate_timestamps: int = 0
    duplicate_symbol_timestamps: int = 0
    null_ohlc_count: int = 0
    nonpositive_price_count: int = 0
    negative_volume_count: int = 0
    unsorted: bool = False
    adjusted_price_detected: bool = False
    split_adjustment_detected: bool = False
    dividend_adjustment_detected: bool = False
    regular_hours_count: int = 0
    extended_hours_count: int = 0
    layout: str = "unknown"


def build_market_data_inventory(
    *,
    event_features_path: Path = DEFAULT_EVENT_FEATURES,
    output_dir: Path = DEFAULT_REPORT_DIR,
    candidate_paths: dict[str, list[Path]] | None = None,
) -> dict[str, Any]:
    assert_report_output_dir(output_dir)
    news_symbols, event_date_min, event_date_max = read_news_symbols(event_features_path)
    candidates = candidate_paths or default_candidate_paths()
    timeframe_reports = {
        timeframe: audit_timeframe(timeframe, candidates.get(timeframe, []), news_symbols)
        for timeframe in TIMEFRAMES
    }
    inventory = {
        "mode": "offline_report_only",
        "network_requests": False,
        "price_downloads": False,
        "model_training": False,
        "news_symbol_count": len(news_symbols),
        "news_symbols": news_symbols,
        "event_features_path": str(event_features_path),
        "event_date_min": event_date_min,
        "event_date_max": event_date_max,
        "timeframes": timeframe_reports,
        "news_symbol_mapping": inspect_symbol_mapping_files(),
        "candidate_importers": candidate_importer_findings(),
        "resampling_policy": resampling_policy(timeframe_reports),
        "daily_readiness": daily_readiness(
            timeframe_reports["daily"],
            event_date_min=event_date_min,
            event_date_max=event_date_max,
        ),
        "source_recommendations": source_recommendations(timeframe_reports),
        "research_only": True,
        "trading_impact": "none",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / INVENTORY_FILENAME
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    plan = build_import_plan(inventory)
    plan_path = output_dir / IMPORT_PLAN_FILENAME
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    inventory["inventory_path"] = str(inventory_path)
    inventory["import_plan_path"] = str(plan_path)
    return inventory


def assert_report_output_dir(output_dir: Path) -> None:
    if "reports" not in output_dir.parts:
        raise ValueError("output_dir must be under reports/")


def read_news_symbols(path: Path) -> tuple[list[str], str | None, str | None]:
    if not path.exists():
        return [], None, None
    symbols: set[str] = set()
    event_dates: list[date] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            symbol = normalize_symbol(row.get("symbol") or row.get("ticker"))
            if symbol:
                symbols.add(symbol)
            parsed = parse_datetime(
                row.get("available_at_timestamp") or row.get("event_timestamp")
            )
            if parsed:
                event_dates.append(parsed.date())
    return (
        sorted(symbols),
        min(event_dates).isoformat() if event_dates else None,
        max(event_dates).isoformat() if event_dates else None,
    )


def default_candidate_paths() -> dict[str, list[Path]]:
    return {
        "daily": existing_paths(
            [
                Path("data/reference/adjusted_prices"),
                Path("data/processed/stooq_parquet"),
                Path("data/processed"),
            ]
        ),
        "hourly": existing_paths([Path("data/processed")]),
        "5_minute": existing_paths([Path("data/processed")]),
        "15_minute": existing_paths([Path("data/processed")]),
    }


def existing_paths(paths: Iterable[Path]) -> list[Path]:
    return sorted({path for path in paths if path.exists()}, key=str)


def audit_timeframe(
    timeframe: str,
    candidate_source_paths: list[Path],
    news_symbols: list[str],
) -> dict[str, Any]:
    files = discover_price_files(timeframe, candidate_source_paths)
    audits = [audit_file(path) for path in files]
    readable = [item for item in audits if item.readable]
    symbols = sorted({symbol for item in readable for symbol in item.symbols if symbol})
    news_set = set(news_symbols)
    covered = sorted(news_set & set(symbols))
    missing = sorted(news_set - set(symbols))
    detected_columns = sorted({column for item in readable for column in item.columns})
    timezone_values = sorted({value for item in readable for value in item.timezone_values})
    layouts = Counter(item.layout for item in readable)
    date_values = [
        value
        for item in readable
        for value in (item.date_min, item.date_max)
        if value is not None
    ]
    required_columns_present = bool(readable) and all(
        item.required_columns_present for item in readable
    )
    blocking_reasons = []
    if news_symbols and missing:
        blocking_reasons.append("missing_news_symbol_coverage")
    if files and not readable:
        blocking_reasons.append("no_readable_price_files")
    if readable and not required_columns_present:
        blocking_reasons.append("missing_required_ohlcv_columns")
    if timeframe in {"hourly", "5_minute", "15_minute"} and not readable:
        blocking_reasons.append("no_existing_intraday_files_detected")
    return {
        "timeframe": timeframe,
        "candidate_source_paths": [str(path) for path in candidate_source_paths],
        "file_count": len(files),
        "readable_file_count": len(readable),
        "invalid_file_count": len(files) - len(readable),
        "symbol_count": len(symbols),
        "covered_news_symbol_count": len(covered),
        "missing_news_symbol_count": len(missing),
        "covered_news_symbols": covered,
        "missing_news_symbols": missing,
        "date_min": min(date_values) if date_values else None,
        "date_max": max(date_values) if date_values else None,
        "rows_total": sum(item.rows for item in readable),
        "required_columns_present": required_columns_present,
        "detected_columns": detected_columns,
        "timezone_detected": timezone_values[0] if len(timezone_values) == 1 else (
            "unknown" if not timezone_values else "mixed"
        ),
        "timezone_consistent": len(timezone_values) <= 1,
        "timestamp_duplicates": sum(item.duplicate_timestamps for item in readable),
        "symbol_timestamp_duplicates": sum(
            item.duplicate_symbol_timestamps for item in readable
        ),
        "null_ohlc_count": sum(item.null_ohlc_count for item in readable),
        "nonpositive_price_count": sum(item.nonpositive_price_count for item in readable),
        "negative_volume_count": sum(item.negative_volume_count for item in readable),
        "unsorted_file_count": sum(1 for item in readable if item.unsorted),
        "adjusted_price_detected": any(item.adjusted_price_detected for item in readable),
        "split_adjustment_detected": any(item.split_adjustment_detected for item in readable),
        "dividend_adjustment_detected": any(
            item.dividend_adjustment_detected for item in readable
        ),
        "regular_hours_only_detected": (
            bool(readable)
            and sum(item.regular_hours_count for item in readable) > 0
            and sum(item.extended_hours_count for item in readable) == 0
        ),
        "extended_hours_detected": sum(item.extended_hours_count for item in readable) > 0,
        "bar_timestamp_semantics": "unknown",
        "schema_variants": sorted(
            {
                ",".join(item.columns)
                for item in readable
                if item.columns
            }
        ),
        "layout": layouts.most_common(1)[0][0] if layouts else "unknown",
        "candidate_importers": timeframe_importers(timeframe),
        "blocking_reasons": blocking_reasons,
        "warnings": timeframe_warnings(timeframe, readable),
    }


def discover_price_files(timeframe: str, roots: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() in PRICE_EXTENSIONS:
            if path_matches_timeframe(root, timeframe):
                files.append(root)
            continue
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in PRICE_EXTENSIONS:
                if path_matches_timeframe(path, timeframe):
                    files.append(path)
    return sorted(set(files), key=str)


def path_matches_timeframe(path: Path, timeframe: str) -> bool:
    lowered = "/".join(part.lower() for part in path.parts)
    if timeframe == "daily":
        return (
            "adjusted_prices" in lowered
            or "stooq_parquet" in lowered
            or "/1day/" in lowered
            or "/daily/" in lowered
        )
    if timeframe == "hourly":
        return any(token in lowered for token in ("/1h/", "/hourly/", "_1h", "_60m"))
    if timeframe == "5_minute":
        return any(token in lowered for token in ("/5m/", "/5 min/", "/5min/", "_5m"))
    if timeframe == "15_minute":
        return any(token in lowered for token in ("/15m/", "/15 min/", "/15min/", "_15m"))
    return False


def audit_file(path: Path) -> FileAudit:
    audit = FileAudit(path=str(path), symbol=symbol_from_path(path))
    try:
        rows = read_rows(path)
    except Exception as exc:
        audit.invalid_reason = str(exc)
        return audit
    audit.readable = True
    audit.rows = len(rows)
    if not rows:
        audit.columns = []
        return audit
    audit.columns = sorted(rows[0].keys())
    audit.required_columns_present = has_required_columns(audit.columns)
    audit.adjusted_price_detected = any(column in audit.columns for column in ADJUSTED_COLUMNS)
    audit.dividend_adjustment_detected = any(
        "dividend" in column.lower() for column in audit.columns
    )
    timestamps: list[datetime] = []
    symbol_timestamps: list[tuple[str, datetime]] = []
    previous: datetime | None = None
    close_values: list[float] = []
    for row in rows:
        symbol = normalize_symbol(first_present(row, SYMBOL_COLUMNS)) or audit.symbol
        if symbol:
            audit.symbols.add(symbol)
        ts = parse_row_timestamp(row)
        if ts:
            timestamps.append(ts)
            if symbol:
                symbol_timestamps.append((symbol, ts))
            if previous and ts < previous:
                audit.unsorted = True
            previous = ts
            if ts.tzinfo is None:
                audit.timezone_values.add("naive")
            else:
                audit.timezone_values.add(str(ts.tzinfo))
            if ts.time() >= datetime.strptime("14:30", "%H:%M").time() and ts.time() <= datetime.strptime("21:00", "%H:%M").time():
                audit.regular_hours_count += 1
            elif ts.hour or ts.minute:
                audit.extended_hours_count += 1
        prices = [number(first_present(row, names)) for names in OHLC_COLUMNS.values()]
        if any(value is None for value in prices):
            audit.null_ohlc_count += 1
        if any(value is not None and value <= 0 for value in prices):
            audit.nonpositive_price_count += 1
        close_value = prices[-1]
        if close_value and close_value > 0:
            close_values.append(close_value)
        volume = number(first_present(row, VOLUME_COLUMNS))
        if volume is not None and volume < 0:
            audit.negative_volume_count += 1
    audit.duplicate_timestamps = duplicate_count(timestamps)
    audit.duplicate_symbol_timestamps = duplicate_count(symbol_timestamps)
    if timestamps:
        audit.date_min = min(ts.date() for ts in timestamps).isoformat()
        audit.date_max = max(ts.date() for ts in timestamps).isoformat()
    audit.split_adjustment_detected = audit.adjusted_price_detected or has_large_price_jump(
        close_values
    )
    audit.layout = "multi_symbol_file" if len(audit.symbols) > 1 else "one_file_per_symbol"
    if not audit.symbols and audit.symbol:
        audit.symbols.add(audit.symbol)
    return audit


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to read parquet files") from exc
        table = pq.read_table(path)
        data = table.to_pylist()
        return [dict(row) for row in data]
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def has_required_columns(columns: list[str]) -> bool:
    names = set(columns)
    has_time = any(name in names for name in TIMESTAMP_COLUMNS)
    return has_time and all(any(name in names for name in options) for options in OHLC_COLUMNS.values())


def parse_row_timestamp(row: dict[str, Any]) -> datetime | None:
    value = first_present(row, TIMESTAMP_COLUMNS)
    if value is None and row.get("<DATE>") and row.get("<TIME>"):
        value = f"{row['<DATE>']}{row['<TIME>']}"
        return parse_datetime(value, "%Y%m%d%H%M%S")
    return parse_datetime(value)


def parse_datetime(value: Any, fmt: str | None = None) -> datetime | None:
    if value in {None, ""}:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    text = str(value).strip()
    try:
        return datetime.strptime(text, fmt) if fmt else datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            return datetime.fromisoformat(text[:10])
        except ValueError:
            return None


def first_present(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row and row[name] not in {None, ""}:
            return row[name]
    lowered = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(str(name).lower())
        if value not in {None, ""}:
            return value
    return None


def number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def normalize_symbol(value: Any) -> str:
    if value in {None, ""}:
        return ""
    symbol = str(value).strip().upper()
    if symbol.endswith(".US"):
        symbol = symbol[:-3]
    return symbol.replace("/", "-")


def symbol_from_path(path: Path) -> str:
    stem = path.stem.upper()
    for suffix in ("_5M", "_1H", "_15M", "_60M", "-5M", "-1H", "-15M", "-60M"):
        stem = stem.replace(suffix, "")
    if stem == "BARS":
        return path.parent.parent.name.upper() if path.parent.parent else ""
    return normalize_symbol(stem.split(".")[0])


def duplicate_count(values: Iterable[Any]) -> int:
    items = list(values)
    return len(items) - len(set(items))


def has_large_price_jump(values: list[float]) -> bool:
    for left, right in zip(values, values[1:]):
        if left > 0 and right > 0 and max(left, right) / min(left, right) >= 4.0:
            return True
    return False


def timeframe_importers(timeframe: str) -> list[str]:
    mapping = {
        "daily": [
            "infrastructure/data/stooq_bulk_importer.py",
            "infrastructure/data/yahoo_adjusted_price_importer.py",
            "infrastructure/data/market_parquet.py",
            "application/services/market_parquet_commands.py",
            "application/services/adjusted_price_commands.py",
        ],
        "hourly": [
            "infrastructure/data/market_parquet.py",
            "application/services/market_parquet_commands.py",
        ],
        "5_minute": [
            "infrastructure/data/market_parquet.py",
            "application/services/market_parquet_commands.py",
        ],
        "15_minute": [],
    }
    return mapping[timeframe]


def timeframe_warnings(timeframe: str, readable: list[FileAudit]) -> list[str]:
    warnings = []
    if timeframe == "15_minute":
        warnings.append("no_native_15_minute_importer_or_resampler_detected")
    if timeframe in {"hourly", "5_minute", "15_minute"} and not readable:
        warnings.append("no_populated_existing_intraday_dataset_detected")
    if readable and any("naive" in item.timezone_values for item in readable):
        warnings.append("timestamp_timezone_unknown_for_some_files")
    return warnings


def inspect_symbol_mapping_files() -> dict[str, Any]:
    candidates = [
        Path("data/reference/sector_by_symbol.json"),
        Path("data/reference/universes"),
        Path("scripts/stock_alpha_news_missing_price_symbol_plan.py"),
    ]
    existing = [str(path) for path in candidates if path.exists()]
    return {
        "canonical_symbol_mappings_detected": existing,
        "stooq_symbol_mapping_detected": "unknown",
        "yahoo_symbol_mapping_detected": "unknown",
        "class_share_mapping_detected": "unknown",
        "renamed_or_delisted_mapping_detected": "unknown",
        "notes": [
            "No dedicated provider ticker mapping table was found by the offline inventory script.",
            "Class-share and renamed/delisted mappings require explicit review before import.",
        ],
    }


def candidate_importer_findings() -> list[dict[str, Any]]:
    return [
        {
            "importer_path": "infrastructure/data/stooq_bulk_importer.py",
            "provider": "Stooq bulk ASCII",
            "supported_intervals": ["daily"],
            "cli_api_entry_points": ["--mode import-stooq-bulk"],
            "symbol_list_input_support": True,
            "output_directory": "data/processed/stooq_parquet",
            "overwrite_behavior": "writes parquet unless caller uses resume path",
            "resume_behavior": "import_symbols_with_manifest supports resume=True",
            "rate_limiting": "not applicable to local extracted files",
            "retry_behavior": "none detected",
            "schema": ["timestamp", "open", "high", "low", "close", "volume"],
            "timezone_policy": "daily timestamps are date-like; timezone not explicit",
            "adjustment_policy": "unknown from local metadata",
        },
        {
            "importer_path": "infrastructure/data/market_parquet.py",
            "provider": "local market parquet normalization",
            "supported_intervals": ["1Day", "5m", "1h"],
            "cli_api_entry_points": ["--mode import-market-parquet"],
            "symbol_list_input_support": True,
            "output_directory": "data/processed/{SYMBOL}/{TIMEFRAME}/bars.parquet",
            "overwrite_behavior": "resume=True skips existing output",
            "resume_behavior": "configured by ml.market_data.resume_import",
            "rate_limiting": "not applicable to local files",
            "retry_behavior": "none detected",
            "schema": ["timestamp", "open", "high", "low", "close", "volume", "symbol"],
            "timezone_policy": "source timezone config; normalized to UTC",
            "adjustment_policy": "unknown; preserves input OHLCV",
        },
        {
            "importer_path": "infrastructure/data/yahoo_adjusted_price_importer.py",
            "provider": "Yahoo chart API",
            "supported_intervals": ["daily"],
            "cli_api_entry_points": ["--mode ml-refresh-adjusted-prices"],
            "symbol_list_input_support": True,
            "output_directory": "data/reference/adjusted_prices",
            "overwrite_behavior": "rewrites per-symbol CSV outputs",
            "resume_behavior": "none detected",
            "rate_limiting": "none detected",
            "retry_behavior": "none detected",
            "schema": ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"],
            "timezone_policy": "Yahoo timestamps interpreted as UTC dates",
            "adjustment_policy": "adjusted OHLC using adj_close/raw_close ratio",
        },
    ]


def daily_readiness(
    daily: dict[str, Any],
    *,
    event_date_min: str | None,
    event_date_max: str | None,
) -> dict[str, Any]:
    blocking = list(daily["blocking_reasons"])
    warnings = list(daily["warnings"])
    if not daily["adjusted_price_detected"]:
        warnings.append("adjusted_price_support_not_detected_for_all_daily_sources")
    if event_date_min and daily["date_min"] and daily["date_min"] > event_date_min:
        blocking.append("daily_data_starts_after_first_event")
    if event_date_max and daily["date_max"]:
        needed = (parse_datetime(event_date_max).date() + timedelta(days=30)).isoformat()
        if daily["date_max"] < needed:
            blocking.append("daily_data_may_not_cover_20_trading_day_forward_labels")
    status = "blocked" if blocking else ("approved_with_warnings" if warnings else "approved")
    return {
        "status": status,
        "blocking_reasons": sorted(set(blocking)),
        "warnings": sorted(set(warnings)),
        "checks": {
            "coverage_across_news_symbols": daily["covered_news_symbol_count"],
            "adjusted_close_or_adjustment_support": daily["adjusted_price_detected"],
            "earliest_price_date": daily["date_min"],
            "latest_price_date": daily["date_max"],
            "event_date_min": event_date_min,
            "event_date_max": event_date_max,
            "duplicate_symbol_date_rows": daily["symbol_timestamp_duplicates"],
            "missing_adjusted_prices": not daily["adjusted_price_detected"],
            "obvious_split_discontinuities": daily["split_adjustment_detected"],
            "ticker_mapping_failures": daily["missing_news_symbol_count"],
        },
    }


def resampling_policy(timeframes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    five = timeframes["5_minute"]
    can_derive = (
        five["readable_file_count"] > 0
        and five["timezone_detected"] != "unknown"
        and not five["blocking_reasons"]
    )
    return {
        "preferred_15_minute_policy": "derive_from_5m",
        "derived_15_minute_approved": can_derive,
        "native_15_minute_import_required": False,
        "required_aggregation": {
            "open": "first_valid_open",
            "high": "maximum_high",
            "low": "minimum_low",
            "close": "last_valid_close",
            "volume": "sum_valid_volume",
        },
        "requirements": [
            "never_cross_trading_session_boundaries",
            "respect_exchange_timezone_and_dst",
            "do_not_mix_regular_and_extended_sessions_unless_configured",
            "document_bin_alignment",
            "preserve_symbol_and_session_metadata",
        ],
        "recommendation": (
            "derive_15_minute_from_5m_after_resampler_validation"
            if can_derive
            else "implement_safe_session_aware_5m_to_15m_resampler_before_deriving"
        ),
    }


def source_recommendations(timeframes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    daily = timeframes["daily"]
    five = timeframes["5_minute"]
    hourly = timeframes["hourly"]
    return {
        "daily": (
            "use_existing"
            if not daily["missing_news_symbol_count"] and daily["adjusted_price_detected"]
            else "backfill_missing_only"
            if daily["symbol_count"]
            else "provider_to_inspect"
        ),
        "hourly": "use_existing" if hourly["symbol_count"] else "import_missing",
        "5_minute": "use_existing" if five["symbol_count"] else "import_missing",
        "15_minute": "derive_from_5m",
        "provider_to_inspect": "Stooq intraday local files/import-market-parquet; Yahoo adjusted daily only",
    }


def build_import_plan(inventory: dict[str, Any]) -> dict[str, Any]:
    timeframes = inventory["timeframes"]
    recommendations = inventory["source_recommendations"]
    next_step = "use_existing_daily_and_prepare_intraday_import"
    if timeframes["daily"]["missing_news_symbol_count"]:
        next_step = "prepare_daily_adjustment_layer"
    if not timeframes["5_minute"]["symbol_count"] and not timeframes["hourly"]["symbol_count"]:
        next_step = "prepare_stooq_intraday_import"
    return {
        "news_symbol_count": inventory["news_symbol_count"],
        "daily_covered_symbol_count": timeframes["daily"]["covered_news_symbol_count"],
        "daily_missing_symbols": timeframes["daily"]["missing_news_symbols"],
        "hourly_covered_symbol_count": timeframes["hourly"]["covered_news_symbol_count"],
        "hourly_missing_symbols": timeframes["hourly"]["missing_news_symbols"],
        "five_minute_covered_symbol_count": timeframes["5_minute"]["covered_news_symbol_count"],
        "five_minute_missing_symbols": timeframes["5_minute"]["missing_news_symbols"],
        "fifteen_minute_covered_symbol_count": timeframes["15_minute"]["covered_news_symbol_count"],
        "fifteen_minute_missing_symbols": timeframes["15_minute"]["missing_news_symbols"],
        "recommended_daily_source": recommendations["daily"],
        "recommended_hourly_source": recommendations["hourly"],
        "recommended_five_minute_source": recommendations["5_minute"],
        "recommended_fifteen_minute_policy": recommendations["15_minute"],
        "candidate_importer_paths": sorted(
            {
                path
                for report in timeframes.values()
                for path in report["candidate_importers"]
            }
        ),
        "symbol_mapping_requirements": inventory["news_symbol_mapping"],
        "timezone_requirements": [
            "explicit exchange/source timezone before intraday import",
            "normalize stored bars to UTC",
            "document bar open/close timestamp semantics",
        ],
        "session_requirements": [
            "regular-hours versus extended-hours policy",
            "session-boundary safe resampling",
            "DST-aware exchange calendar handling",
        ],
        "adjustment_requirements": [
            "daily labels require adjusted prices or validated adjustment layer",
            "intraday bars need documented split-adjustment policy",
        ],
        "estimated_file_count": {
            timeframe: report["file_count"] for timeframe, report in timeframes.items()
        },
        "recommended_next_step": next_step,
    }


def print_summary(inventory: dict[str, Any]) -> None:
    timeframes = inventory["timeframes"]
    print("MARKET DATA INVENTORY")
    for name in TIMEFRAMES:
        report = timeframes[name]
        print(
            f"{name}: coverage={report['covered_news_symbol_count']}/"
            f"{inventory['news_symbol_count']} "
            f"symbols={report['symbol_count']} "
            f"range={report['date_min']}..{report['date_max']}"
        )
    print(f"daily_adjustment_status={timeframes['daily']['adjusted_price_detected']}")
    print(
        "candidate_importer_paths="
        + ",".join(build_import_plan(inventory)["candidate_importer_paths"])
    )
    print(
        "derive_15_minute="
        + inventory["resampling_policy"]["recommendation"]
    )
    print(
        "blocking_reasons="
        + json.dumps(
            {
                name: report["blocking_reasons"]
                for name, report in timeframes.items()
                if report["blocking_reasons"]
            },
            sort_keys=True,
        )
    )
    print(f"recommended_next_step={build_import_plan(inventory)['recommended_next_step']}")
    print(f"inventory={inventory['inventory_path']}")
    print(f"import_plan={inventory['import_plan_path']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline report-only market-data inventory for news transformer symbols."
    )
    parser.add_argument("--event-features", default=str(DEFAULT_EVENT_FEATURES))
    parser.add_argument("--output-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--daily-path", action="append", default=[])
    parser.add_argument("--hourly-path", action="append", default=[])
    parser.add_argument("--five-minute-path", action="append", default=[])
    parser.add_argument("--fifteen-minute-path", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    supplied = {
        "daily": [Path(value) for value in args.daily_path],
        "hourly": [Path(value) for value in args.hourly_path],
        "5_minute": [Path(value) for value in args.five_minute_path],
        "15_minute": [Path(value) for value in args.fifteen_minute_path],
    }
    candidate_paths = None if not any(supplied.values()) else supplied
    inventory = build_market_data_inventory(
        event_features_path=Path(args.event_features),
        output_dir=Path(args.output_dir),
        candidate_paths=candidate_paths,
    )
    print_summary(inventory)


if __name__ == "__main__":
    main()
