import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from core.entities.candle import Candle


@dataclass(frozen=True)
class LoadedCandles:
    candles: list[Candle]
    metadata: dict


def load_candles(symbol, config, feed):
    return load_candles_with_metadata(symbol, config, feed).candles


def load_candles_with_metadata(symbol, config, feed) -> LoadedCandles:
    backtest_config = config["backtest"]
    end = datetime.utcnow()
    start = end - timedelta(days=365 * backtest_config["years"])
    cache_config = config.get("cache", {})
    cache_enabled = cache_config.get("enabled", False)
    cache_path = data_cache_path(symbol, backtest_config, cache_config)

    if cache_enabled and cache_path and cache_path.exists():
        cached = read_candle_cache_with_metadata(cache_path)
        if cached is not None and _cache_covers_request(cached.metadata, start, end):
            return LoadedCandles(
                candles=cached.candles,
                metadata={**cached.metadata, "cache_hit": True},
            )

    candles = feed.get_historical_bars(
        symbol=symbol,
        timeframe=backtest_config["timeframe"],
        start=start,
        end=end,
    )

    request_metadata = _request_metadata(symbol, backtest_config, feed, start, end)
    request_metadata.update({
        "actual_start": candles[0].timestamp.isoformat() if candles else None,
        "actual_end": candles[-1].timestamp.isoformat() if candles else None,
        "bar_count": len(candles),
        "cache_hit": False,
    })
    if cache_enabled and cache_path:
        write_candle_cache(cache_path, candles, request_metadata)

    return LoadedCandles(candles=candles, metadata=request_metadata)


def data_cache_path(symbol, backtest_config, cache_config):
    directory = Path(cache_config.get("data_dir", "cache/data"))
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    provider = backtest_config.get("provider", "alpaca")
    data_feed = (
        backtest_config.get("data_feed", "default")
        if provider == "alpaca" else "local"
    )
    bar_limit = (
        backtest_config.get("historical_bar_limit", "default")
        if provider == "alpaca" else "not_applicable"
    )
    adjustment = backtest_config.get("data_adjustment", "raw")
    filename = (
        f"{symbol}_{provider}_{backtest_config['timeframe']}_"
        f"{backtest_config['years']}y_{data_feed}_{adjustment}_{bar_limit}bars.json"
    )
    return directory / filename


def read_candle_cache(path):
    cached = read_candle_cache_with_metadata(path)
    return cached.candles if cached is not None else None


def read_candle_cache_with_metadata(path) -> LoadedCandles | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if isinstance(payload, list):
        return LoadedCandles(candles=_candles_from_payload(payload), metadata={})
    if not isinstance(payload, dict) or not isinstance(payload.get("candles"), list):
        return None
    return LoadedCandles(
        candles=_candles_from_payload(payload["candles"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _candles_from_payload(payload) -> list[Candle]:
    return [
        Candle(
            symbol=item["symbol"],
            timestamp=datetime.fromisoformat(item["timestamp"]),
            open=item["open"],
            high=item["high"],
            low=item["low"],
            close=item["close"],
            volume=item["volume"],
        )
        for item in payload
    ]


def write_candle_cache(path, candles, metadata: dict | None = None):
    candle_payload = [
        {
            "symbol": candle.symbol,
            "timestamp": candle.timestamp.isoformat(),
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
        }
        for candle in candles
    ]

    payload = {"metadata": metadata or {}, "candles": candle_payload}
    try:
        path.write_text(
            json.dumps(payload, separators=(",", ":")),
            encoding="utf-8",
        )
    except OSError:
        return


def _request_metadata(symbol, backtest_config, feed, start, end) -> dict:
    provider_metadata = {}
    if hasattr(feed, "get_last_request_metadata"):
        provider_metadata = feed.get_last_request_metadata(symbol) or {}
    provider = backtest_config.get("provider", "alpaca")
    return {
        "symbol": symbol,
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "provider": provider,
        "feed": backtest_config.get("data_feed") if provider == "alpaca" else None,
        "historical_bar_limit": (
            backtest_config.get("historical_bar_limit")
            if provider == "alpaca" else None
        ),
        **provider_metadata,
    }


def _cache_covers_request(metadata: dict, start: datetime, end: datetime) -> bool:
    actual_start = metadata.get("actual_start")
    actual_end = metadata.get("actual_end")
    if not actual_start or not actual_end:
        return False
    try:
        cached_start = datetime.fromisoformat(actual_start)
        cached_end = datetime.fromisoformat(actual_end)
    except ValueError:
        return False
    requested_start = start.replace(tzinfo=cached_start.tzinfo)
    requested_end = end.replace(tzinfo=cached_end.tzinfo)
    return cached_start <= requested_start + timedelta(days=10) and cached_end >= requested_end - timedelta(days=10)


def latest_prices(candles_by_symbol):
    prices = {}

    for symbol, candles in candles_by_symbol.items():
        if candles:
            prices[symbol] = candles[-1].close

    return prices


def latest_data_freshness(candles_by_symbol, max_age_days=3):
    latest_timestamp = None

    for candles in candles_by_symbol.values():
        if not candles:
            continue

        timestamp = candles[-1].timestamp
        comparable_timestamp = timestamp.replace(tzinfo=None)
        comparable_latest = (
            latest_timestamp.replace(tzinfo=None)
            if latest_timestamp is not None
            else None
        )

        if comparable_latest is None or comparable_timestamp > comparable_latest:
            latest_timestamp = timestamp

    if latest_timestamp is None:
        return {
            "latest_timestamp": None,
            "age_days": None,
            "is_stale": True,
            "max_age_days": max_age_days,
        }

    now = datetime.utcnow()
    latest_naive = latest_timestamp.replace(tzinfo=None)
    age_days = max((now.date() - latest_naive.date()).days, 0)

    return {
        "latest_timestamp": latest_timestamp.isoformat(),
        "age_days": age_days,
        "is_stale": age_days > max_age_days,
        "max_age_days": max_age_days,
    }


def data_quality_report(
    candles_by_symbol,
    min_lookback_bars=252,
    max_latest_gap_percent=0.40,
):
    issues_by_symbol = {}

    for symbol, candles in candles_by_symbol.items():
        issues = []

        if not candles:
            issues.append({
                "reason": "missing_candles",
                "severity": "ERROR",
                "details": {},
            })
            issues_by_symbol[symbol] = issues
            continue

        if len(candles) < min_lookback_bars:
            issues.append({
                "reason": "insufficient_lookback",
                "severity": "ERROR",
                "details": {
                    "bars": len(candles),
                    "min_lookback_bars": min_lookback_bars,
                },
            })

        dates = [candle.timestamp.date().isoformat() for candle in candles]
        duplicate_dates = sorted({
            date for date in dates if dates.count(date) > 1
        })
        if duplicate_dates:
            issues.append({
                "reason": "duplicate_candle_dates",
                "severity": "ERROR",
                "details": {
                    "dates": duplicate_dates[:10],
                    "count": len(duplicate_dates),
                },
            })

        bad_price_count = sum(
            1
            for candle in candles
            if (
                candle.open <= 0
                or candle.high <= 0
                or candle.low <= 0
                or candle.close <= 0
            )
        )
        if bad_price_count:
            issues.append({
                "reason": "zero_or_negative_prices",
                "severity": "ERROR",
                "details": {"count": bad_price_count},
            })

        invalid_ohlc_count = sum(
            1
            for candle in candles
            if candle.high < candle.low
            or candle.high < max(candle.open, candle.close)
            or candle.low > min(candle.open, candle.close)
        )
        if invalid_ohlc_count:
            issues.append({
                "reason": "invalid_ohlc",
                "severity": "ERROR",
                "details": {"count": invalid_ohlc_count},
            })

        if len(candles) >= 2:
            previous_close = candles[-2].close
            latest_close = candles[-1].close
            if previous_close > 0 and latest_close > 0:
                latest_gap = abs((latest_close / previous_close) - 1)
                if latest_gap > max_latest_gap_percent:
                    issues.append({
                        "reason": "large_latest_price_gap",
                        "severity": "ERROR",
                        "details": {
                            "gap": latest_gap,
                            "limit": max_latest_gap_percent,
                            "previous_close": previous_close,
                            "latest_close": latest_close,
                        },
                    })

        if issues:
            issues_by_symbol[symbol] = issues

    return {
        "issues_by_symbol": issues_by_symbol,
        "issue_count": sum(len(issues) for issues in issues_by_symbol.values()),
        "min_lookback_bars": min_lookback_bars,
        "max_latest_gap_percent": max_latest_gap_percent,
    }
