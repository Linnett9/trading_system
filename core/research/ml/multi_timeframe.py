from __future__ import annotations

from datetime import date, datetime, time, timezone
from math import sqrt
from statistics import pstdev

from core.entities.candle import Candle


def add_intraday_summary_features(
    rows: list[dict[str, float | str]],
    candles_by_timeframe: dict[str, dict[str, list[Candle]]],
    *,
    benchmark_symbols: list[str],
    session_close_utc: str = "21:00",
) -> list[dict[str, float | str]]:
    """Fuse intraday features into daily rows using an as-of, no-lookahead rule."""
    if not rows or not candles_by_timeframe:
        return rows
    close_time = _parse_session_close(session_close_utc)
    output = []
    for row in rows:
        feature_date = date.fromisoformat(str(row["feature_date"]))
        cutoff = datetime.combine(feature_date, close_time, tzinfo=timezone.utc)
        enriched = dict(row)
        for timeframe, candles_by_symbol in sorted(candles_by_timeframe.items()):
            prefix = _feature_prefix(timeframe)
            for symbol in benchmark_symbols:
                candles = [
                    candle for candle in candles_by_symbol.get(symbol, [])
                    if _as_utc(candle.timestamp) <= cutoff
                ]
                if not candles:
                    continue
                closes = [candle.close for candle in candles if candle.close > 0]
                volumes = [candle.volume for candle in candles]
                if len(closes) >= 2:
                    returns = [
                        current / previous - 1.0
                        for previous, current in zip(closes[:-1], closes[1:])
                        if previous > 0
                    ]
                    safe_symbol = _safe_symbol(symbol)
                    enriched[f"{prefix}_{safe_symbol}_return_last_bar"] = returns[-1]
                    enriched[f"{prefix}_{safe_symbol}_realized_vol_last_78_bars"] = (
                        pstdev(returns[-78:]) * sqrt(252 * 78)
                        if len(returns[-78:]) > 1
                        else 0.0
                    )
                    enriched[f"{prefix}_{safe_symbol}_return_last_78_bars"] = (
                        closes[-1] / closes[-min(79, len(closes))] - 1.0
                    )
                enriched[f"{prefix}_{_safe_symbol(symbol)}_volume_last_bar"] = (
                    float(volumes[-1]) if volumes else 0.0
                )
        output.append(enriched)
    return output


def _parse_session_close(value: str) -> time:
    hour, minute = str(value).split(":", 1)
    return time(hour=int(hour), minute=int(minute))


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _feature_prefix(timeframe: str) -> str:
    return str(timeframe).lower().replace("min", "m").replace("day", "d")


def _safe_symbol(symbol: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "_"
        for character in symbol
    ).strip("_")
