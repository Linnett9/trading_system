from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from core.research.ml.ds24.clean_v2_contracts import load_contract, stable_hash
from core.research.ml.ds24.master_5m_validation_stats import compute_stock_features
from infrastructure.data.calendar_authority import default_calendar_authority


CONFIRMED_CROSS_SESSION_REPAIRS = (
    "overnight_gap",
    "previous_session_return",
    "two_session_return",
)
ADDITIONAL_PERTURBATION_REPAIRS = (
    "opening_range_position",
    "session_return_30m",
    "opening_return_30m",
)
CALENDAR_SOURCE_REPAIRS = (
    "minutes_since_open",
    "minutes_until_close",
    "session_progress",
    "early_close_session_flag",
    "opening_period_flag",
)
REPAIRED_FEATURES = (
    CONFIRMED_CROSS_SESSION_REPAIRS
    + ADDITIONAL_PERTURBATION_REPAIRS
    + CALENDAR_SOURCE_REPAIRS
)
SOURCE_TIMESTAMP_SUFFIX = "__max_source_timestamp"
KEY_COLUMNS = ("asset_id", "decision_timestamp")


class FeatureAuthorityError(ValueError):
    """Raised when V2 features cannot be composed without ambiguity."""


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def _session_summary(frame: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
    work = pd.DataFrame(
        {
            "session_date": frame["session_date"].astype(str),
            "open": pd.to_numeric(frame["open"], errors="coerce"),
            "close": pd.to_numeric(frame["close"], errors="coerce"),
            "bar_start": timestamps,
        },
        index=frame.index,
    )
    summary = work.groupby("session_date", sort=False, observed=True).agg(
        session_open=("open", "first"),
        session_close=("close", "last"),
        session_open_bar_start=("bar_start", "first"),
        session_close_bar_start=("bar_start", "last"),
    )
    summary["previous_session_open"] = summary["session_open"].shift(1)
    summary["previous_session_close"] = summary["session_close"].shift(1)
    summary["previous_session_close_bar_start"] = summary["session_close_bar_start"].shift(1)
    summary["two_completed_sessions_ago_close"] = summary["session_close"].shift(2)
    summary["two_completed_sessions_ago_close_bar_start"] = summary["session_close_bar_start"].shift(2)
    return summary


def _regular_bar_mask(frame: pd.DataFrame) -> pd.Series:
    if "session_type" not in frame:
        return pd.Series(True, index=frame.index)
    normalized = frame["session_type"].astype(str).str.strip().str.lower()
    return normalized.isin({"rth", "regular", "regular_session", "early_close"})


@lru_cache(maxsize=4096)
def _calendar_bounds(session_date: str) -> tuple[pd.Timestamp, pd.Timestamp, bool]:
    record = default_calendar_authority().session(session_date, exchange="XNYS")
    if not record.is_trading_day or record.open_timestamp is None or record.close_timestamp is None:
        return pd.NaT, pd.NaT, False
    return (
        pd.Timestamp(record.open_timestamp).tz_convert("UTC"),
        pd.Timestamp(record.close_timestamp).tz_convert("UTC"),
        bool(record.early_close),
    )


def compute_repaired_session_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute V2 session repairs using only bars available at each row's decision.

    `timestamp_utc` is the source bar start and its conservative availability time is
    five minutes later. Input rows may span multiple sessions but must belong to one
    asset and have unique timestamps.
    """

    required = {"timestamp_utc", "session_date", "open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FeatureAuthorityError(f"Raw frame is missing columns: {missing}")
    if frame.empty:
        columns = [*REPAIRED_FEATURES, *(f"{name}{SOURCE_TIMESTAMP_SUFFIX}" for name in REPAIRED_FEATURES)]
        return pd.DataFrame(index=frame.index, columns=columns)

    work = frame.sort_values("timestamp_utc", kind="mergesort").copy()
    timestamps = pd.to_datetime(work["timestamp_utc"], utc=True, errors="raise")
    if timestamps.duplicated().any():
        raise FeatureAuthorityError("Raw frame contains duplicate timestamp_utc values")
    availability = timestamps + pd.Timedelta(minutes=5)
    session_key = work["session_date"].astype(str)
    regular_mask = _regular_bar_mask(work)
    close = pd.to_numeric(work["close"], errors="coerce")
    high = pd.to_numeric(work["high"], errors="coerce")
    low = pd.to_numeric(work["low"], errors="coerce")
    regular_work = work.loc[regular_mask]
    summary = _session_summary(regular_work, timestamps.loc[regular_mask])

    def session_map(column: str) -> pd.Series:
        return session_key.map(summary[column])

    out = pd.DataFrame(index=work.index)
    current_session_open = session_map("session_open")
    previous_open = session_map("previous_session_open")
    previous_close = session_map("previous_session_close")
    two_session_close = session_map("two_completed_sessions_ago_close")

    out["overnight_gap"] = _safe_div(current_session_open, previous_close) - 1.0
    out["previous_session_return"] = _safe_div(previous_close, previous_open) - 1.0
    # The registered R7 formula is close(T) / close(two completed sessions ago) - 1.
    out["two_session_return"] = _safe_div(close, two_session_close) - 1.0

    opening_summary = pd.DataFrame(
        {
            "session_date": session_key.loc[regular_mask],
            "high": high.loc[regular_mask],
            "low": low.loc[regular_mask],
            "close": close.loc[regular_mask],
            "bar_start": timestamps.loc[regular_mask],
        }
    ).groupby("session_date", sort=False, observed=True).agg(
        opening_high=("high", lambda values: values.iloc[:6].max() if len(values) >= 6 else np.nan),
        opening_low=("low", lambda values: values.iloc[:6].min() if len(values) >= 6 else np.nan),
        opening_close=("close", lambda values: values.iloc[5] if len(values) >= 6 else np.nan),
        opening_bar_start=("bar_start", lambda values: values.iloc[5] if len(values) >= 6 else pd.NaT),
    )
    opening_high = session_key.map(opening_summary["opening_high"])
    opening_low = session_key.map(opening_summary["opening_low"])
    opening_close = session_key.map(opening_summary["opening_close"])
    opening_source = pd.to_datetime(session_key.map(opening_summary["opening_bar_start"]), utc=True) + pd.Timedelta(minutes=5)
    opening_complete = regular_mask & opening_source.notna() & (availability >= opening_source)
    out["opening_range_position"] = _safe_div(close - opening_low, opening_high - opening_low).where(opening_complete)
    out["session_return_30m"] = (_safe_div(close, opening_close) - 1.0).where(opening_complete)
    out["opening_return_30m"] = (_safe_div(opening_close, current_session_open) - 1.0).where(opening_complete)

    previous_close_source = pd.to_datetime(
        session_map("previous_session_close_bar_start"), utc=True
    ) + pd.Timedelta(minutes=5)
    two_session_source = pd.to_datetime(
        session_map("two_completed_sessions_ago_close_bar_start"), utc=True
    ) + pd.Timedelta(minutes=5)
    session_open_source = pd.to_datetime(
        session_map("session_open_bar_start"), utc=True
    ) + pd.Timedelta(minutes=5)
    bounds = session_key.map(lambda value: _calendar_bounds(str(value)))
    scheduled_open = pd.to_datetime(bounds.map(lambda value: value[0]), utc=True)
    scheduled_close = pd.to_datetime(bounds.map(lambda value: value[1]), utc=True)
    scheduled_minutes = (scheduled_close - scheduled_open).dt.total_seconds() / 60.0
    minutes_since_open = (availability - scheduled_open).dt.total_seconds() / 60.0
    minutes_until_close = (scheduled_close - availability).dt.total_seconds() / 60.0
    calendar_eligible = regular_mask & (availability > scheduled_open) & (availability <= scheduled_close)
    out["minutes_since_open"] = minutes_since_open.where(calendar_eligible)
    out["minutes_until_close"] = minutes_until_close.where(calendar_eligible)
    out["session_progress"] = _safe_div(minutes_since_open, scheduled_minutes).where(calendar_eligible)
    out["early_close_session_flag"] = bounds.map(lambda value: int(value[2])).where(calendar_eligible)
    out["opening_period_flag"] = (minutes_since_open < 30).astype("float64").where(calendar_eligible)

    out[f"overnight_gap{SOURCE_TIMESTAMP_SUFFIX}"] = pd.concat(
        [session_open_source, previous_close_source], axis=1
    ).max(axis=1)
    out[f"previous_session_return{SOURCE_TIMESTAMP_SUFFIX}"] = previous_close_source
    out[f"two_session_return{SOURCE_TIMESTAMP_SUFFIX}"] = pd.concat(
        [availability, two_session_source], axis=1
    ).max(axis=1)
    out[f"opening_range_position{SOURCE_TIMESTAMP_SUFFIX}"] = availability.where(opening_complete)
    out[f"session_return_30m{SOURCE_TIMESTAMP_SUFFIX}"] = availability.where(opening_complete)
    out[f"opening_return_30m{SOURCE_TIMESTAMP_SUFFIX}"] = opening_source.where(opening_complete)
    for feature in CALENDAR_SOURCE_REPAIRS:
        out[f"{feature}{SOURCE_TIMESTAMP_SUFFIX}"] = availability.where(calendar_eligible)

    for feature in REPAIRED_FEATURES:
        out[feature] = out[feature].where(regular_mask)
        source_column = f"{feature}{SOURCE_TIMESTAMP_SUFFIX}"
        out[source_column] = out[source_column].where(regular_mask)

    decision_timestamps = availability
    for feature in REPAIRED_FEATURES:
        source = out[f"{feature}{SOURCE_TIMESTAMP_SUFFIX}"]
        violation = source.notna() & (source > decision_timestamps)
        if violation.any():
            raise FeatureAuthorityError(f"Future source timestamp detected for {feature}")
    return out.reindex(frame.index)


def compose_v2_stock_features(base: pd.DataFrame, sidecar: pd.DataFrame) -> pd.DataFrame:
    required_sidecar = {*KEY_COLUMNS, *REPAIRED_FEATURES}
    missing = sorted(required_sidecar - set(sidecar.columns))
    if missing:
        raise FeatureAuthorityError(f"V2 sidecar is missing columns: {missing}")
    for label, table in (("base", base), ("sidecar", sidecar)):
        absent_keys = sorted(set(KEY_COLUMNS) - set(table.columns))
        if absent_keys:
            raise FeatureAuthorityError(f"{label} is missing key columns: {absent_keys}")
        if table.duplicated(list(KEY_COLUMNS)).any():
            raise FeatureAuthorityError(f"{label} has duplicate feature keys")
    replacement = sidecar[[*KEY_COLUMNS, *REPAIRED_FEATURES]].copy()
    retained = base.drop(columns=[name for name in REPAIRED_FEATURES if name in base], errors="ignore")
    merged = retained.merge(replacement, on=list(KEY_COLUMNS), how="left", validate="one_to_one", indicator=True)
    if len(merged) != len(base) or not (merged["_merge"] == "both").all():
        raise FeatureAuthorityError("V2 sidecar does not cover the V1 base population exactly")
    return merged.drop(columns="_merge")


def _stock_features_v2(frame: pd.DataFrame) -> pd.DataFrame:
    stock = compute_stock_features(frame, benchmark=None).copy()
    repaired = compute_repaired_session_features(frame)
    for feature in REPAIRED_FEATURES:
        stock[feature] = repaired[feature]
    return stock


def build_complete_feature_panel(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Reference calculator for the complete ordered 101-feature vector.

    This deliberately favors a transparent causal implementation over production
    throughput. It is the value-provenance certification oracle, not the bulk V2
    materializer.
    """

    if not frames:
        raise FeatureAuthorityError("At least one raw symbol frame is required")
    per_symbol: dict[str, pd.DataFrame] = {}
    for symbol, raw in sorted(frames.items()):
        ordered = raw.sort_values("timestamp_utc", kind="mergesort").copy()
        features = _stock_features_v2(ordered)
        features.insert(0, "timestamp_utc", pd.to_datetime(ordered["timestamp_utc"], utc=True).to_numpy())
        features.insert(0, "asset_id", ordered.get("asset_id", pd.Series(symbol, index=ordered.index)).astype(str).to_numpy())
        features.insert(0, "symbol", symbol)
        features["decision_timestamp"] = pd.to_datetime(features["timestamp_utc"], utc=True) + pd.Timedelta(minutes=5)
        per_symbol[symbol] = features.reset_index(drop=True)

    def benchmark_feature(symbol: str, feature: str) -> pd.Series:
        table = per_symbol.get(symbol)
        if table is None:
            return pd.Series(dtype="float64")
        return table.set_index("timestamp_utc")[feature]

    benchmark_columns = {
        "spy_ret_5m": ("SPY", "ret_5m"),
        "spy_ret_15m": ("SPY", "ret_15m"),
        "spy_ret_30m": ("SPY", "ret_30m"),
        "spy_ret_60m": ("SPY", "ret_60m"),
        "spy_ret_120m": ("SPY", "ret_120m"),
        "spy_realized_vol_60m": ("SPY", "realized_vol_60m"),
        "spy_session_drawdown": ("SPY", "session_drawdown"),
        "spy_session_return_to_date": ("SPY", "session_return_to_date"),
        "qqq_ret_5m": ("QQQ", "ret_5m"),
        "qqq_ret_15m": ("QQQ", "ret_15m"),
        "qqq_ret_30m": ("QQQ", "ret_30m"),
        "qqq_ret_60m": ("QQQ", "ret_60m"),
        "qqq_ret_120m": ("QQQ", "ret_120m"),
        "qqq_realized_vol_60m": ("QQQ", "realized_vol_60m"),
        "qqq_session_drawdown": ("QQQ", "session_drawdown"),
        "qqq_session_return_to_date": ("QQQ", "session_return_to_date"),
        "gld_ret_60m": ("GLD", "ret_60m"),
        "tlt_ret_60m": ("TLT", "ret_60m"),
        "xlk_ret_60m": ("XLK", "ret_60m"),
    }
    benchmark_series = {
        column: benchmark_feature(symbol, feature)
        for column, (symbol, feature) in benchmark_columns.items()
    }

    panel = pd.concat(per_symbol.values(), ignore_index=True)
    timestamps = pd.to_datetime(panel["timestamp_utc"], utc=True)
    for column, values in benchmark_series.items():
        panel[column] = timestamps.map(values)
    panel["qqq_vs_spy_ret_60m"] = panel["qqq_ret_60m"] - panel["spy_ret_60m"]
    for horizon in ("15m", "30m", "60m", "120m"):
        panel[f"relative_strength_spy_{horizon}"] = panel[f"ret_{horizon}"] - panel[f"spy_ret_{horizon}"]
    for horizon in ("15m", "60m", "120m"):
        panel[f"relative_strength_qqq_{horizon}"] = panel[f"ret_{horizon}"] - panel[f"qqq_ret_{horizon}"]

    timestamp_group = panel.groupby("timestamp_utc", sort=False, observed=True)
    panel["relative_strength_rank_60m"] = timestamp_group["ret_60m"].rank(pct=True)
    panel["relative_strength_rank_120m"] = timestamp_group["ret_120m"].rank(pct=True)
    for horizon in ("15m", "60m"):
        returns = panel[f"ret_{horizon}"]
        panel[f"breadth_fraction_positive_{horizon}"] = (returns > 0).groupby(panel["timestamp_utc"]).transform("mean")
        panel[f"breadth_median_ret_{horizon}"] = timestamp_group[f"ret_{horizon}"].transform("median")
        panel[f"breadth_return_dispersion_{horizon}"] = timestamp_group[f"ret_{horizon}"].transform(lambda values: values.std(ddof=0))
    observed = timestamp_group["asset_id"].transform("count")
    panel["breadth_observed_symbol_count"] = observed
    panel["breadth_eligible_symbol_count"] = len(per_symbol)
    panel["breadth_coverage_ratio"] = observed / float(len(per_symbol))

    predictors = load_contract("predictor_manifest.json")["predictors"]
    missing = sorted(set(predictors) - set(panel.columns))
    if missing:
        raise FeatureAuthorityError(f"Reference calculator omitted predictors: {missing}")
    return panel[["symbol", "asset_id", "timestamp_utc", "decision_timestamp", *predictors]].sort_values(
        ["decision_timestamp", "symbol"], kind="mergesort"
    ).reset_index(drop=True)


@dataclass(frozen=True)
class PerturbationCase:
    symbol: str
    decision_timestamp: pd.Timestamp
    label: str


def certify_future_bar_invariance(
    frames: Mapping[str, pd.DataFrame],
    cases: Iterable[PerturbationCase],
) -> dict[str, Any]:
    predictors = list(load_contract("predictor_manifest.json")["predictors"])
    baseline = build_complete_feature_panel(frames)
    results: list[dict[str, Any]] = []
    for ordinal, case in enumerate(cases):
        decision = pd.Timestamp(case.decision_timestamp)
        decision = decision.tz_localize("UTC") if decision.tzinfo is None else decision.tz_convert("UTC")
        selected = baseline[(baseline["symbol"] == case.symbol) & (baseline["decision_timestamp"] == decision)]
        if len(selected) != 1:
            raise FeatureAuthorityError(
                f"Expected one baseline row for {case.symbol} at {decision}, got {len(selected)}"
            )
        mutated: dict[str, pd.DataFrame] = {}
        for symbol, raw in frames.items():
            copy = raw.copy(deep=True)
            available = pd.to_datetime(copy["timestamp_utc"], utc=True) + pd.Timedelta(minutes=5)
            future = available > decision
            factor = 1.7 + ordinal * 0.01
            for column in ("open", "high", "low", "close", "vwap"):
                if column in copy:
                    copy.loc[future, column] = pd.to_numeric(copy.loc[future, column], errors="coerce") * factor + 11.0
            for column in ("volume", "trade_count"):
                if column in copy:
                    copy.loc[future, column] = pd.to_numeric(copy.loc[future, column], errors="coerce") * 3 + 7
            mutated[symbol] = copy
        recomputed = build_complete_feature_panel(mutated)
        candidate = recomputed[(recomputed["symbol"] == case.symbol) & (recomputed["decision_timestamp"] == decision)]
        if len(candidate) != 1:
            raise FeatureAuthorityError(f"Perturbed calculation lost case {case.label}")
        before = selected.iloc[0][predictors].to_numpy(dtype="float64")
        after = candidate.iloc[0][predictors].to_numpy(dtype="float64")
        equal = np.isclose(before, after, equal_nan=True, rtol=0.0, atol=1e-12)
        changed = [name for name, unchanged in zip(predictors, equal, strict=True) if not unchanged]
        results.append(
            {
                "label": case.label,
                "symbol": case.symbol,
                "decision_timestamp": decision.isoformat(),
                "changed_features": changed,
                "passed": not changed,
            }
        )
    payload: dict[str, Any] = {
        "certification_id": "DS24_CLEAN_V2_FUTURE_BAR_PERTURBATION_V1",
        "predictor_count": len(predictors),
        "case_count": len(results),
        "cases": results,
        "passed": bool(results) and all(row["passed"] for row in results),
        "terminal_classification": (
            "101 / 101 FEATURES CAUSAL UNDER FUTURE-BAR PERTURBATION"
            if results and all(row["passed"] for row in results)
            else "FEATURE_CAUSALITY_FAILURE"
        ),
    }
    payload["certificate_sha256"] = stable_hash(payload)
    return payload
