import json
from datetime import datetime, timedelta
from pathlib import Path

from core.entities.candle import Candle


def load_candles(symbol, config, feed):
    backtest_config = config["backtest"]
    end = datetime.utcnow()
    start = end - timedelta(days=365 * backtest_config["years"])
    cache_config = config.get("cache", {})
    cache_enabled = cache_config.get("enabled", False)
    cache_path = data_cache_path(symbol, backtest_config, cache_config)

    if cache_enabled and cache_path and cache_path.exists():
        cached_candles = read_candle_cache(cache_path)
        if cached_candles is not None:
            return cached_candles

    candles = feed.get_historical_bars(
        symbol=symbol,
        timeframe=backtest_config["timeframe"],
        start=start,
        end=end,
    )

    if cache_enabled and cache_path:
        write_candle_cache(cache_path, candles)

    return candles


def data_cache_path(symbol, backtest_config, cache_config):
    directory = Path(cache_config.get("data_dir", "cache/data"))
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    filename = (
        f"{symbol}_{backtest_config['timeframe']}_"
        f"{backtest_config['years']}y.json"
    )
    return directory / filename


def read_candle_cache(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

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


def write_candle_cache(path, candles):
    payload = [
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

    try:
        path.write_text(
            json.dumps(payload, separators=(",", ":")),
            encoding="utf-8",
        )
    except OSError:
        return


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