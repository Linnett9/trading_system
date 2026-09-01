from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FeatureStats:
    total_rows: int = 0
    non_null_rows: int = 0
    null_rows: int = 0
    finite_rows: int = 0
    nan_rows: int = 0
    pos_inf_rows: int = 0
    neg_inf_rows: int = 0
    min_value: float | None = None
    max_value: float | None = None
    sum_value: float = 0.0
    sumsq_value: float = 0.0
    zero_rows: int = 0
    positive_rows: int = 0
    negative_rows: int = 0
    earliest_valid_timestamp: str = ""
    latest_valid_timestamp: str = ""

    def update(self, values: pd.Series, timestamps: pd.Series) -> None:
        self.total_rows += int(len(values))
        not_null = values.notna()
        self.non_null_rows += int(not_null.sum())
        self.null_rows += int((~not_null).sum())
        numeric = pd.to_numeric(values, errors="coerce")
        finite = np.isfinite(numeric.to_numpy(dtype="float64", na_value=np.nan))
        self.finite_rows += int(finite.sum())
        self.nan_rows += int(np.isnan(numeric.to_numpy(dtype="float64", na_value=np.nan)).sum())
        self.pos_inf_rows += int(np.isposinf(numeric.to_numpy(dtype="float64", na_value=np.nan)).sum())
        self.neg_inf_rows += int(np.isneginf(numeric.to_numpy(dtype="float64", na_value=np.nan)).sum())
        valid = numeric[finite]
        if len(valid):
            current_min = float(valid.min())
            current_max = float(valid.max())
            self.min_value = current_min if self.min_value is None else min(self.min_value, current_min)
            self.max_value = current_max if self.max_value is None else max(self.max_value, current_max)
            self.sum_value += float(valid.sum())
            self.sumsq_value += float((valid * valid).sum())
            self.zero_rows += int((valid == 0).sum())
            self.positive_rows += int((valid > 0).sum())
            self.negative_rows += int((valid < 0).sum())
            valid_ts = timestamps[finite]
            if len(valid_ts):
                low = str(valid_ts.iloc[0])
                high = str(valid_ts.iloc[-1])
                self.earliest_valid_timestamp = low if not self.earliest_valid_timestamp else min(self.earliest_valid_timestamp, low)
                self.latest_valid_timestamp = high if not self.latest_valid_timestamp else max(self.latest_valid_timestamp, high)

    def row(self, feature_id: str) -> dict[str, Any]:
        coverage = self.non_null_rows / self.total_rows if self.total_rows else 0.0
        mean = self.sum_value / self.finite_rows if self.finite_rows else None
        variance = (self.sumsq_value / self.finite_rows - (mean or 0.0) ** 2) if self.finite_rows else None
        return {
            "semantic_feature_id": feature_id,
            "total_considered_rows": self.total_rows,
            "non_null_rows": self.non_null_rows,
            "null_rows": self.null_rows,
            "coverage_fraction": round(coverage, 8),
            "finite_rows": self.finite_rows,
            "NaN_rows": self.nan_rows,
            "pos_inf_rows": self.pos_inf_rows,
            "neg_inf_rows": self.neg_inf_rows,
            "earliest_valid_timestamp": self.earliest_valid_timestamp,
            "latest_valid_timestamp": self.latest_valid_timestamp,
            "min": self.min_value,
            "max": self.max_value,
            "mean": mean,
            "std": math.sqrt(max(0.0, variance)) if variance is not None else None,
            "zero_fraction": round(self.zero_rows / self.finite_rows, 8) if self.finite_rows else 0.0,
            "positive_rows": self.positive_rows,
            "negative_rows": self.negative_rows,
        }


def _safe_div(numerator: pd.Series, denominator: pd.Series | float) -> pd.Series:
    out = numerator / denominator
    return out.replace([np.inf, -np.inf], np.nan)


def _returns(close: pd.Series, bars: int) -> pd.Series:
    return _safe_div(close, close.shift(bars)) - 1.0


def _log_returns(close: pd.Series, bars: int) -> pd.Series:
    return np.log(_safe_div(close, close.shift(bars)))


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = _safe_div(gain, loss)
    return 100.0 - (100.0 / (1.0 + rs))


def _session_minutes(ts: pd.Series) -> pd.Series:
    return ts.dt.hour * 60 + ts.dt.minute


def compute_stock_features(frame: pd.DataFrame, benchmark: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    df = frame.sort_values("timestamp_utc").copy()
    ts = pd.to_datetime(df["timestamp_utc"], utc=True)
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    trade_count = pd.to_numeric(df["trade_count"], errors="coerce") if "trade_count" in df else pd.Series(np.nan, index=df.index)
    vwap = pd.to_numeric(df["vwap"], errors="coerce") if "vwap" in df else pd.Series(np.nan, index=df.index)
    returns_5m = _returns(close, 1)
    session_key = df["session_date"].astype(str)
    grouped = df.assign(_close=close, _high=high, _low=low, _open=open_, _volume=volume).groupby(session_key, sort=False)
    session_open = grouped["_open"].transform("first")
    session_high = grouped["_high"].cummax()
    session_low = grouped["_low"].cummin()
    session_vwap = _safe_div((close * volume).groupby(session_key).cumsum(), volume.groupby(session_key).cumsum())
    minute = _session_minutes(ts)
    first_minute = minute.groupby(session_key).transform("min")
    last_minute = minute.groupby(session_key).transform("max")
    since_open = minute - first_minute
    until_close = last_minute - minute
    session_length = (last_minute - first_minute + 5).replace(0, np.nan)
    dollar_volume = close * volume
    out = pd.DataFrame(index=df.index)
    for bars, name in [(1, "ret_5m"), (3, "ret_15m"), (6, "ret_30m"), (12, "ret_60m"), (24, "ret_120m")]:
        out[name] = _returns(close, bars)
    out["reversal_15m_vs_60m"] = out["ret_15m"] - out["ret_60m"]
    out["momentum_accel_30m_60m"] = out["ret_30m"] - out["ret_60m"]
    out["momentum_persistence_60m"] = np.sign(returns_5m).rolling(12, min_periods=12).mean() * np.sign(out["ret_60m"])
    out["distance_from_intraday_high"] = _safe_div(close, session_high) - 1.0
    out["distance_from_intraday_low"] = _safe_div(close, session_low) - 1.0
    for bars, name in [(3, "realized_vol_15m"), (6, "realized_vol_30m"), (12, "realized_vol_60m"), (24, "realized_vol_120m"), (48, "realized_vol_240m")]:
        out[name] = returns_5m.rolling(bars, min_periods=bars).std(ddof=0)
    out["downside_dev_60m"] = np.sqrt((returns_5m.clip(upper=0) ** 2).rolling(12, min_periods=12).mean())
    out["downside_dev_120m"] = np.sqrt((returns_5m.clip(upper=0) ** 2).rolling(24, min_periods=24).mean())
    out["rolling_drawdown_60m"] = _safe_div(close, close.rolling(12, min_periods=12).max()) - 1.0
    out["rolling_drawdown_120m"] = _safe_div(close, close.rolling(24, min_periods=24).max()) - 1.0
    out["session_drawdown"] = _safe_div(close, session_high) - 1.0
    out["range_pct_60m"] = _safe_div(high.rolling(12, min_periods=12).max() - low.rolling(12, min_periods=12).min(), close)
    out["range_pct_15m"] = _safe_div(high.rolling(3, min_periods=3).max() - low.rolling(3, min_periods=3).min(), close)
    out["vol_trend_30m_vs_120m"] = _safe_div(out["realized_vol_30m"], out["realized_vol_120m"]) - 1.0
    out["atr_pct_70m"] = _safe_div(pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1).rolling(14, min_periods=14).mean(), close)
    out["range_expansion_15m_vs_60m"] = _safe_div(out["range_pct_15m"], out["range_pct_60m"]) - 1.0
    out["vol_percentile_20d_tod_pit"] = out["realized_vol_60m"].groupby(since_open).transform(lambda s: s.shift(1).expanding(min_periods=20).rank(pct=True))
    out["dollar_volume_5m"] = dollar_volume
    out["dollar_volume_60m"] = dollar_volume.rolling(12, min_periods=12).sum()
    out["volume_ratio_60m"] = _safe_div(volume, volume.shift(1).rolling(12, min_periods=12).mean())
    out["volume_accel_15m_vs_60m"] = _safe_div(volume.rolling(3, min_periods=3).sum(), volume.rolling(12, min_periods=12).sum() / 4.0) - 1.0
    out["relative_volume_tod_pit"] = _safe_div(volume, volume.groupby(since_open).transform(lambda s: s.shift(1).expanding(min_periods=20).mean()))
    out["vwap_distance_session"] = _safe_div(close, session_vwap) - 1.0
    out["trade_count_5m"] = trade_count
    out["trade_count_60m"] = trade_count.rolling(12, min_periods=12).sum()
    out["volume_zscore_60m"] = _safe_div(volume - volume.rolling(12, min_periods=12).mean(), volume.rolling(12, min_periods=12).std(ddof=0))
    out["dollar_volume_zscore_60m"] = _safe_div(dollar_volume - dollar_volume.rolling(12, min_periods=12).mean(), dollar_volume.rolling(12, min_periods=12).std(ddof=0))
    out["cumulative_dollar_volume_session"] = dollar_volume.groupby(session_key).cumsum()
    out["dollar_volume_accel_15m_vs_60m"] = _safe_div(dollar_volume.rolling(3, min_periods=3).sum(), dollar_volume.rolling(12, min_periods=12).sum() / 4.0) - 1.0
    out["minutes_since_open"] = since_open.astype("float32")
    out["minutes_until_close"] = until_close.astype("float32")
    out["session_progress"] = _safe_div(since_open, session_length)
    out["early_close_session_flag"] = (session_length < 390).astype("int8")
    out["opening_period_flag"] = (since_open < 30).astype("int8")
    first_close = close.groupby(session_key).transform("first")
    previous_close = close.groupby(session_key).transform("last").shift(1)
    out["overnight_gap"] = _safe_div(session_open, previous_close) - 1.0
    out["session_return_to_date"] = _safe_div(close, session_open) - 1.0
    out["session_range_pct"] = _safe_div(session_high - session_low, close)
    opening_high = high.groupby(session_key).transform(lambda s: s.iloc[:6].max() if len(s) >= 6 else np.nan)
    opening_low = low.groupby(session_key).transform(lambda s: s.iloc[:6].min() if len(s) >= 6 else np.nan)
    out["opening_range_position"] = _safe_div(close - opening_low, opening_high - opening_low)
    out["cumulative_volume_ratio_tod_pit"] = _safe_div(volume.groupby(session_key).cumsum(), volume.groupby(since_open).transform(lambda s: s.shift(1).expanding(min_periods=20).mean()))
    out["session_return_30m"] = _safe_div(close, first_close.groupby(session_key).transform(lambda s: s.iloc[5] if len(s) > 5 else np.nan)) - 1.0
    out["previous_session_return"] = (_safe_div(close.groupby(session_key).transform("last"), session_open) - 1.0).shift(1)
    out["two_session_return"] = _safe_div(close, close.groupby(session_key).transform("last").shift(2)) - 1.0
    out["opening_return_30m"] = _safe_div(first_close.groupby(session_key).transform(lambda s: s.iloc[5] if len(s) > 5 else np.nan), session_open) - 1.0
    out["range_expansion_session"] = _safe_div(out["session_range_pct"], out["session_range_pct"].groupby(since_open).transform(lambda s: s.shift(1).expanding(min_periods=20).mean())) - 1.0
    for bars, name in [(1, "log_ret_5m"), (3, "log_ret_15m"), (12, "log_ret_60m")]:
        out[name] = _log_returns(close, bars)
    out["sma_distance_30m"] = _safe_div(close, close.rolling(6, min_periods=6).mean()) - 1.0
    out["sma_distance_60m"] = _safe_div(close, close.rolling(12, min_periods=12).mean()) - 1.0
    out["ema_distance_60m"] = _safe_div(close, close.ewm(span=12, min_periods=12, adjust=False).mean()) - 1.0
    out["rsi_14_5m"] = _rsi(close, 14)
    macd = close.ewm(span=12, min_periods=12, adjust=False).mean() - close.ewm(span=26, min_periods=26, adjust=False).mean()
    out["macd_histogram_5m"] = macd - macd.ewm(span=9, min_periods=9, adjust=False).mean()
    out["bollinger_zscore_60m"] = _safe_div(close - close.rolling(12, min_periods=12).mean(), close.rolling(12, min_periods=12).std(ddof=0))
    out["high_low_position_60m"] = _safe_div(close - low.rolling(12, min_periods=12).min(), high.rolling(12, min_periods=12).max() - low.rolling(12, min_periods=12).min())
    if benchmark:
        for name in ("spy", "qqq"):
            for bars, horizon in [(3, "15m"), (6, "30m"), (12, "60m"), (24, "120m")]:
                key = f"{name}_ret_{horizon}"
                if key in benchmark:
                    mapped = ts.map(benchmark[key])
                    out[f"relative_strength_{name}_{horizon}"] = out[f"ret_{horizon}"] - mapped
        if "qqq_ret_120m" in benchmark:
            out["relative_strength_qqq_120m"] = out["ret_120m"] - ts.map(benchmark["qqq_ret_120m"])
    for column in (
        "relative_strength_spy_15m",
        "relative_strength_spy_30m",
        "relative_strength_spy_60m",
        "relative_strength_spy_120m",
        "relative_strength_qqq_15m",
        "relative_strength_qqq_60m",
        "relative_strength_qqq_120m",
    ):
        if column not in out:
            out[column] = np.nan
    out["relative_strength_rank_60m"] = np.nan
    out["relative_strength_rank_120m"] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)


def benchmark_context(frames: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    result: dict[str, pd.Series] = {}
    for symbol, frame in frames.items():
        if frame.empty:
            continue
        f = compute_stock_features(frame, None)
        ts = pd.to_datetime(frame.sort_values("timestamp_utc")["timestamp_utc"], utc=True)
        prefix = symbol.lower()
        for horizon in ("5m", "15m", "30m", "60m", "120m"):
            column = f"ret_{horizon}"
            if column in f:
                result[f"{prefix}_ret_{horizon}"] = pd.Series(f[column].to_numpy(), index=ts)
        if "realized_vol_60m" in f:
            result[f"{prefix}_realized_vol_60m"] = pd.Series(f["realized_vol_60m"].to_numpy(), index=ts)
        if "session_drawdown" in f:
            result[f"{prefix}_session_drawdown"] = pd.Series(f["session_drawdown"].to_numpy(), index=ts)
        if "session_return_to_date" in f:
            result[f"{prefix}_session_return_to_date"] = pd.Series(f["session_return_to_date"].to_numpy(), index=ts)
    if "qqq_ret_60m" in result and "spy_ret_60m" in result:
        result["qqq_vs_spy_ret_60m"] = result["qqq_ret_60m"] - result["spy_ret_60m"]
    return result


def time_block() -> float:
    return time.perf_counter()
