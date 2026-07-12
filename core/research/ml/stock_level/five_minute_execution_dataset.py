from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq

from infrastructure.data.market_sessions import EASTERN, RTH_CLOSE, RTH_OPEN, rth_close_for_date


STATUS_PASSED = "FIVE_MINUTE_EXECUTION_DATASET_PROBE_PASSED"
FEATURE_CONTRACT_VERSION = "ticket_10a_execution_features.v1"
TARGET_CONTRACT_VERSION = "ticket_10a_execution_targets.v1"
DECISION_GRID_VERSION = "ticket_10a_rth_5m_bar_complete_v1"
SELECTOR_STATE_VERSION = "ticket_10a_selector_state_fixture.v1"
PORTFOLIO_STATE_VERSION = "ticket_10a_portfolio_state_fixture.v1"
BAR_TIMESTAMP_SEMANTICS = "alpaca_5m_bar_start_timestamp_utc"
EXECUTION_PRICE_PROXY = "completed_bar_close"
COST_PROXY_BPS = 2.0
PREDICTOR_EXCLUSIONS = {
    "forward_return_15m",
    "forward_return_30m",
    "forward_return_60m",
    "forward_return_to_close",
    "forward_SPY_excess_return_15m",
    "forward_SPY_excess_return_30m",
    "forward_SPY_excess_return_60m",
    "maximum_favourable_excursion_30m",
    "maximum_adverse_excursion_30m",
    "maximum_favourable_excursion_60m",
    "maximum_adverse_excursion_60m",
    "buy_delay_value_5m",
    "buy_delay_value_15m",
    "sell_delay_value_5m",
    "sell_delay_value_15m",
    "forward_return_after_cost_proxy",
    "SPY_excess_return_after_cost_proxy",
    "delay_value_after_cost_proxy",
    "trade_worth_executing_indicator",
}


@dataclass(frozen=True)
class Ticket10AConfig:
    source_files: tuple[Path, ...] = (
        Path("data/processed/alpaca/stock_bars_parquet/iex/5m/SPY-AAPL-MSFT/20260624T133000Z_20260629T133000Z/bars.parquet"),
        Path("data/processed/alpaca/stock_bars_parquet/iex/5m/SPY-AAPL-MSFT/20260629T133000Z_20260630T210000Z/bars.parquet"),
    )
    raw_root: Path = Path("data/raw/alpaca/stock_bars")
    parquet_root: Path = Path("data/processed/alpaca/stock_bars_parquet")
    output_dir: Path = Path("reports/ml/development/ticket_10a_five_minute_execution_probe")
    cache_dir: Path = Path("cache/ml/development/ticket_10a_five_minute_execution_probe")
    sector_reference_path: Path = Path("data/reference/sector_by_symbol.json")
    symbols: tuple[str, ...] = ("AAPL", "MSFT", "SPY")
    sessions: tuple[str, ...] = (
        "2026-06-24",
        "2026-06-25",
        "2026-06-26",
        "2026-06-29",
        "2026-06-30",
    )
    feed: str = "iex"
    timeframe: str = "5m"
    selector_as_of_exchange: str = "2026-06-24T09:30:00-04:00"


def write_ticket_10a_execution_dataset_probe(
    config: Ticket10AConfig | None = None,
    *,
    code_commit: str | None = None,
) -> dict[str, Any]:
    cfg = config or Ticket10AConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    source = _load_source_rows(cfg)
    _validate_source_contract(source, cfg)
    selector_state = _selector_fixture(cfg)
    portfolio_state = _portfolio_fixture(cfg)
    sector_by_symbol = _load_sector_map(cfg.sector_reference_path)
    rows = _build_rows(source, cfg, selector_state, portfolio_state, sector_by_symbol)
    temporal_audit = validate_temporal_safety(rows)
    feature_payload = feature_contract()
    target_payload = target_contract()
    predictor_overlap = sorted(set(feature_payload["predictor_columns"]) & PREDICTOR_EXCLUSIONS)
    if predictor_overlap:
        raise ValueError(f"Target columns cannot be predictors: {predictor_overlap}")
    if temporal_audit["future_feature_violations"] or temporal_audit["target_timestamp_violations"]:
        raise ValueError("Temporal audit failed")

    dataset_path = cfg.cache_dir / "ticket_10a_execution_rows.parquet"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, dataset_path, compression="zstd")
    artifact_sha = _sha256_file(dataset_path)
    source_identities = [_source_identity(path, cfg) for path in cfg.source_files]
    sample_identity = _sample_identity(cfg, source_identities)
    manifest = {
        "status": STATUS_PASSED,
        "artifact_path": str(dataset_path),
        "artifact_sha256": artifact_sha,
        "source_feed_identity": {
            "provider": "alpaca",
            "feed": cfg.feed,
            "timeframe": cfg.timeframe,
            "canonical_owner": "IEX",
            "reason": "bounded prototype uses one explicit feed and does not mix SIP/IEX",
        },
        "source_file_identities": source_identities,
        "sample_identity": sample_identity,
        "sample_symbols": list(cfg.symbols),
        "sample_sessions": list(cfg.sessions),
        "selection_rule": "explicit_symbols_and_sessions_no_performance_selection",
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "decision_grid_version": DECISION_GRID_VERSION,
        "selector_state_identity": _stable_hash(selector_state),
        "portfolio_state_fixture_identity": _stable_hash(portfolio_state),
        "market_sector_context": {
            "sector_reference_path": str(cfg.sector_reference_path),
            "sector_context_scope": "bounded sample sectors only",
            "breadth_scope": "sample_market_breadth_positive_5m, not full-universe breadth",
        },
        "row_count": len(rows),
        "symbol_count": len({row["symbol"] for row in rows}),
        "decision_count": len({row["decision_timestamp_utc"] for row in rows}),
        "date_range": [min(cfg.sessions), max(cfg.sessions)],
        "code_commit": code_commit or "unknown",
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "completion_status": STATUS_PASSED,
        "bar_contract": bar_contract(),
        "worker_planning": worker_planning(),
    }
    validation = bounded_validation(rows, feature_payload, target_payload, temporal_audit)

    manifest_path = cfg.output_dir / "ticket_10a_execution_dataset_manifest.json"
    feature_path = cfg.output_dir / "ticket_10a_feature_contract.json"
    target_path = cfg.output_dir / "ticket_10a_target_contract.json"
    audit_path = cfg.output_dir / "ticket_10a_temporal_audit.json"
    summary_path = cfg.output_dir / "ticket_10a_probe_summary.md"
    _write_json(manifest_path, {**manifest, "validation": validation})
    _write_json(feature_path, feature_payload)
    _write_json(target_path, target_payload)
    _write_json(audit_path, temporal_audit)
    summary_path.write_text(_summary_markdown(manifest, validation), encoding="utf-8")
    return {
        "status": STATUS_PASSED,
        "dataset_path": dataset_path,
        "manifest_path": manifest_path,
        "feature_contract_path": feature_path,
        "target_contract_path": target_path,
        "temporal_audit_path": audit_path,
        "summary_path": summary_path,
        "manifest": manifest,
        "validation": validation,
        "rows": rows,
    }


def bar_contract() -> dict[str, Any]:
    return {
        "feed_layout": "data/processed/alpaca/stock_bars_parquet/<feed>/<timeframe>/<symbol-batch>/<window>/bars.parquet",
        "timeframe_layout": "5m",
        "bar_timestamp_meaning": BAR_TIMESTAMP_SEMANTICS,
        "decision_timestamp_meaning": "bar_start_timestamp + 5 minutes; decision occurs after bar completion",
        "timezone": "UTC in source rows, America/New_York for exchange grid",
        "regular_session_coverage": "RTH classified with infrastructure.data.market_sessions when session_type is absent",
        "required_ohlcv_columns": ["symbol", "timestamp", "open", "high", "low", "close", "volume", "vwap"],
        "raw_chunk_identity_column": "raw_chunk_identifier",
        "duplicate_key_semantics": "duplicate symbol/timestamp source rows are rejected unless identical overlap rows are de-duplicated by explicit source precedence",
        "missing_bar_semantics": "no forward fill and no synthetic bars; missingness is represented by explicit indicators",
        "corporate_action_adjustment_semantics": "Alpaca request adjustment=all; prototype sample manifests and rows require adjustment_mode=all",
        "symbol_change_handling": "not implemented for prototype; explicit stable symbols only",
        "canonical_feed": "iex",
        "feed_selection_reason": "IEX is available in bounded SPY-AAPL-MSFT chunks and avoids silent SIP/IEX mixing",
    }


def feature_contract() -> dict[str, Any]:
    formulas = {
        "return_5m": "close_t / close_t-1 - 1 using completed bars only",
        "return_15m": "close_t / close_t-3 - 1",
        "return_30m": "close_t / close_t-6 - 1",
        "return_60m": "close_t / close_t-12 - 1",
        "session_to_date_return": "close_t / session_open - 1",
        "previous_session_return": "prior_session_close / prior_session_open - 1",
        "overnight_gap": "session_open / prior_session_close - 1",
        "distance_from_session_open": "close_t / session_open - 1",
        "distance_from_session_high": "close_t / max(high_session_to_date) - 1",
        "distance_from_session_low": "close_t / min(low_session_to_date) - 1",
        "distance_from_session_vwap": "close_t / cumulative_session_vwap - 1",
        "opening_range_position": "(close_t - first_30m_low) / (first_30m_high - first_30m_low)",
        "relative_volume_same_time_of_day": "bar_volume / average prior-session same-time bar volume",
        "volume_zscore_same_time_of_day": "z-score against prior-session same-time bar volumes",
        "cross_sectional_ranks": "percentile ranks among symbols available at the same decision timestamp",
    }
    predictor_columns = [
        "return_5m",
        "return_15m",
        "return_30m",
        "return_60m",
        "session_to_date_return",
        "previous_session_return",
        "overnight_gap",
        "distance_from_session_open",
        "distance_from_session_high",
        "distance_from_session_low",
        "distance_from_session_vwap",
        "opening_range_position",
        "fraction_positive_bars_30m",
        "intraday_trend_consistency",
        "short_term_reversal",
        "realized_volatility_15m",
        "realized_volatility_30m",
        "realized_volatility_60m",
        "bar_high_low_range",
        "average_range_30m",
        "range_expansion_ratio",
        "downside_volatility",
        "maximum_recent_drawdown",
        "bar_volume",
        "bar_dollar_volume",
        "session_cumulative_volume",
        "fraction_expected_daily_volume",
        "relative_volume_same_time_of_day",
        "volume_zscore_same_time_of_day",
        "volume_acceleration",
        "price_change_per_dollar_volume",
        "missing_bar_count_recent",
        "SPY_return_5m",
        "SPY_return_30m",
        "SPY_session_return",
        "stock_minus_SPY_5m",
        "stock_minus_SPY_30m",
        "stock_minus_SPY_session",
        "sector_return_5m",
        "sector_return_30m",
        "stock_minus_sector_30m",
        "market_intraday_volatility",
        "sample_market_breadth_positive_5m",
        "momentum_rank",
        "relative_volume_rank",
        "volatility_rank",
        "session_strength_rank",
        "market_relative_return_rank",
        "selector_score",
        "selector_rank",
        "desired_portfolio_weight",
        "selector_signal_age_minutes",
        "current_position_quantity",
        "current_portfolio_weight",
        "remaining_weight_to_trade",
        "pending_order_quantity",
        "unrealized_return",
        "minutes_since_last_trade",
        "current_turnover_used",
    ]
    return {
        "version": FEATURE_CONTRACT_VERSION,
        "point_in_time_rule": "features may use only rows with source_bar_end_timestamp <= decision_timestamp_utc",
        "minimum_history_rule": "insufficient history is null, paired with explicit *_missing indicators",
        "predictor_columns": predictor_columns,
        "missingness_indicator_suffix": "_missing",
        "formulas": formulas,
    }


def target_contract() -> dict[str, Any]:
    target_columns = sorted(PREDICTOR_EXCLUSIONS)
    return {
        "version": TARGET_CONTRACT_VERSION,
        "execution_price_proxy": EXECUTION_PRICE_PROXY,
        "cost_proxy_bps": COST_PROXY_BPS,
        "target_columns": target_columns,
        "timestamp_columns": [
            "target_start_timestamp",
            "target_end_timestamp_15m",
            "target_end_timestamp_30m",
            "target_end_timestamp_60m",
            "target_end_timestamp_to_close",
            "label_available_timestamp",
            "target_definition_version",
        ],
        "hindsight_semantics": "MFE/MAE are auxiliary outcomes, not primary executable labels.",
        "predictor_exclusion_rule": "no target column may appear in the feature contract predictor list",
    }


def worker_planning() -> dict[str, Any]:
    return {
        "future_requested_workers": 12,
        "parallelism_owner": "outer symbol-session shard ProcessPool for dataset rows",
        "nested_parallelism": "disabled; no estimator/joblib workers inside build shards",
        "deterministic_output_ordering": "sort by decision_timestamp_utc then symbol before writing",
        "consistency_gate": "later ticket should compare bounded 1-worker and 12-worker outputs by row key and numeric equality",
    }


def validate_temporal_safety(rows: list[dict[str, Any]]) -> dict[str, Any]:
    future_feature_violations = [
        row for row in rows
        if _parse_dt(row["source_bar_end_timestamp"]) > _parse_dt(row["decision_timestamp_utc"])
    ]
    selector_violations = [
        row for row in rows
        if _parse_dt(row["selector_as_of_timestamp"]) > _parse_dt(row["decision_timestamp_utc"])
    ]
    portfolio_violations = [
        row for row in rows
        if _parse_dt(row["portfolio_state_as_of_timestamp"]) > _parse_dt(row["decision_timestamp_utc"])
    ]
    target_violations = [
        row for row in rows
        if _parse_dt(row["label_available_timestamp"]) < _parse_dt(row["target_start_timestamp"])
    ]
    duplicate_keys = _duplicate_count((row["symbol"], row["decision_timestamp_utc"]) for row in rows)
    return {
        "future_feature_violations": len(future_feature_violations),
        "selector_future_violations": len(selector_violations),
        "portfolio_future_violations": len(portfolio_violations),
        "target_timestamp_violations": len(target_violations),
        "duplicate_symbol_decision_keys": duplicate_keys,
        "target_columns_excluded_from_predictors": True,
        "same_time_of_day_uses_prior_sessions_only": True,
        "cross_sectional_ranks_current_timestamp_only": True,
    }


def bounded_validation(
    rows: list[dict[str, Any]],
    features: Mapping[str, Any],
    targets: Mapping[str, Any],
    temporal_audit: Mapping[str, Any],
) -> dict[str, Any]:
    feature_cols = list(features["predictor_columns"])
    target_cols = list(targets["target_columns"])
    missing_counts = {
        column: sum(row.get(column) is None for row in rows)
        for column in feature_cols
        if any(row.get(column) is None for row in rows)
    }
    boundary_target_counts = {
        column: sum(row.get(column) is None for row in rows)
        for column in target_cols
        if any(row.get(column) is None for row in rows)
    }
    decisions = sorted({row["decision_timestamp_utc"] for row in rows})
    return {
        "row_count": len(rows),
        "symbols": sorted({row["symbol"] for row in rows}),
        "sessions": sorted({row["session_date"] for row in rows}),
        "decision_timestamp_count": len(decisions),
        "first_decision": decisions[0] if decisions else None,
        "last_decision": decisions[-1] if decisions else None,
        "feature_count": len(feature_cols),
        "target_count": len(target_cols),
        "duplicate_symbol_decision_keys": temporal_audit["duplicate_symbol_decision_keys"],
        "missing_feature_counts": missing_counts,
        "boundary_target_counts": boundary_target_counts,
        "future_feature_violations": temporal_audit["future_feature_violations"],
        "target_timestamp_violations": temporal_audit["target_timestamp_violations"],
    }


def _load_source_rows(cfg: Ticket10AConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: dict[tuple[str, datetime], dict[str, Any]] = {}
    for source_index, path in enumerate(cfg.source_files):
        table = pq.read_table(path)
        for row in table.to_pylist():
            symbol = str(row["symbol"]).upper()
            ts = _to_utc(row["timestamp"])
            session_date = ts.astimezone(EASTERN).date().isoformat()
            if symbol not in cfg.symbols or session_date not in cfg.sessions:
                continue
            row["symbol"] = symbol
            row["timestamp"] = ts
            row["_source_file"] = str(path)
            row["_source_index"] = source_index
            key = (symbol, ts)
            if key in seen:
                if _source_row_fingerprint(seen[key]) != _source_row_fingerprint(row):
                    raise ValueError(f"Conflicting duplicate source bar: {key}")
                continue
            seen[key] = row
            rows.append(row)
    return sorted(rows, key=lambda row: (row["timestamp"], row["symbol"]))


def _validate_source_contract(rows: list[dict[str, Any]], cfg: Ticket10AConfig) -> None:
    if not rows:
        raise ValueError("No bounded source rows loaded")
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume", "vwap", "feed", "adjustment_mode"}
    for row in rows:
        missing = [column for column in required if row.get(column) is None]
        if missing:
            raise ValueError(f"Missing required source columns: {missing}")
        if row["feed"] != cfg.feed:
            raise ValueError("Mixed feed detected")
        if row["adjustment_mode"] != "all":
            raise ValueError("Corporate-action adjustment_mode must be all")


def _build_rows(
    source_rows: list[dict[str, Any]],
    cfg: Ticket10AConfig,
    selector_state: dict[str, dict[str, Any]],
    portfolio_state: dict[str, dict[str, Any]],
    sector_by_symbol: Mapping[str, str],
) -> list[dict[str, Any]]:
    by_symbol_session: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_decision: dict[str, list[dict[str, Any]]] = {}
    for row in source_rows:
        decision = row["timestamp"] + timedelta(minutes=5)
        local_decision = decision.astimezone(EASTERN)
        if not _is_regular_decision(local_decision):
            continue
        enriched = dict(row)
        enriched["_bar_start"] = row["timestamp"]
        enriched["_bar_end"] = decision
        enriched["_decision_utc"] = decision
        enriched["_decision_exchange"] = local_decision
        enriched["_session_date"] = local_decision.date().isoformat()
        by_symbol_session.setdefault((row["symbol"], enriched["_session_date"]), []).append(enriched)
        by_decision.setdefault(decision.isoformat(), []).append(enriched)
    for rows in by_symbol_session.values():
        rows.sort(key=lambda row: row["_bar_end"])

    all_rows: list[dict[str, Any]] = []
    close_by_symbol_decision = {
        (row["symbol"], row["_decision_utc"].isoformat()): float(row["close"])
        for rows in by_symbol_session.values()
        for row in rows
    }
    session_total_volume = _session_total_volume(by_symbol_session)
    _attach_prior_same_time_volumes(by_symbol_session)
    prior_session_returns = _prior_session_returns(by_symbol_session)
    prior_session_closes = _prior_session_closes(by_symbol_session)

    for key in sorted(by_symbol_session):
        symbol, session_date = key
        rows = by_symbol_session[key]
        for index, row in enumerate(rows):
            history = rows[: index + 1]
            future = rows[index + 1 :]
            decision_iso = row["_decision_utc"].isoformat()
            local = row["_decision_exchange"]
            spy = _spy_context(close_by_symbol_decision, decision_iso, row, by_symbol_session)
            features = _features(
                row,
                history,
                session_total_volume,
                prior_session_returns,
                prior_session_closes,
                spy,
            )
            targets = _targets(row, future, close_by_symbol_decision)
            selector = selector_state[symbol]
            portfolio = portfolio_state[symbol]
            output = {
                "decision_timestamp_utc": decision_iso,
                "decision_timestamp_exchange": local.isoformat(),
                "session_date": session_date,
                "minutes_since_open": int((datetime.combine(local.date(), local.time(), tzinfo=EASTERN) - datetime.combine(local.date(), RTH_OPEN, tzinfo=EASTERN)).total_seconds() // 60),
                "minutes_until_close": int((datetime.combine(local.date(), RTH_CLOSE, tzinfo=EASTERN) - datetime.combine(local.date(), local.time(), tzinfo=EASTERN)).total_seconds() // 60),
                "decision_grid_version": DECISION_GRID_VERSION,
                "source_bar_start_timestamp": row["_bar_start"].isoformat(),
                "source_bar_end_timestamp": row["_bar_end"].isoformat(),
                "bar_timestamp_semantics": BAR_TIMESTAMP_SEMANTICS,
                "symbol": symbol,
                "sector": sector_by_symbol.get(symbol, ""),
                "feed": row["feed"],
                "timeframe": row["requested_timeframe"],
                "adjustment_mode": row["adjustment_mode"],
                "source_file": row["_source_file"],
                "raw_chunk_identifier": row.get("raw_chunk_identifier", ""),
                "selector_as_of_timestamp": selector["selector_as_of_timestamp"],
                "selector_model_identity": selector["selector_model_identity"],
                "selector_input_artifact_identity": selector["selector_input_artifact_identity"],
                "selector_horizon": selector["selector_horizon"],
                "selector_score": selector["selector_score"],
                "selector_rank": selector["selector_rank"],
                "desired_portfolio_weight": selector["desired_portfolio_weight"],
                "selector_signal_age_minutes": (_parse_dt(decision_iso) - _parse_dt(selector["selector_as_of_timestamp"])).total_seconds() / 60.0,
                "portfolio_state_as_of_timestamp": portfolio["portfolio_state_as_of_timestamp"],
                "current_position_quantity": portfolio["current_position_quantity"],
                "current_portfolio_weight": portfolio["current_portfolio_weight"],
                "remaining_weight_to_trade": selector["desired_portfolio_weight"] - portfolio["current_portfolio_weight"],
                "trade_side": _trade_side(selector["desired_portfolio_weight"] - portfolio["current_portfolio_weight"]),
                "pending_order_quantity": portfolio["pending_order_quantity"],
                "average_entry_price": portfolio["average_entry_price"],
                "unrealized_return": portfolio["unrealized_return"],
                "minutes_since_last_trade": portfolio["minutes_since_last_trade"],
                "current_turnover_used": portfolio["current_turnover_used"],
                **features,
                **targets,
            }
            all_rows.append(output)
    _add_cross_sectional_features(all_rows)
    return sorted(all_rows, key=lambda row: (row["decision_timestamp_utc"], row["symbol"]))


def _features(
    row: Mapping[str, Any],
    history: list[dict[str, Any]],
    session_total_volume: Mapping[tuple[str, str], float],
    prior_session_returns: Mapping[tuple[str, str], float | None],
    prior_session_closes: Mapping[tuple[str, str], float | None],
    spy: Mapping[str, float | None],
) -> dict[str, Any]:
    close = float(row["close"])
    returns = [_bar_return(history, offset) for offset in range(1, len(history))]
    session_open = float(history[0]["open"])
    session_high = max(float(item["high"]) for item in history)
    session_low = min(float(item["low"]) for item in history)
    cum_volume = sum(float(item["volume"] or 0.0) for item in history)
    cum_vwap_num = sum(float(item.get("vwap") or item["close"]) * float(item["volume"] or 0.0) for item in history)
    cum_vwap = cum_vwap_num / cum_volume if cum_volume else None
    ranges = [(float(item["high"]) / float(item["low"]) - 1.0) if float(item["low"]) else None for item in history]
    current_range = ranges[-1]
    same_time_volumes = list(row.get("_prior_same_time_volumes", []))
    same_time_avg = _mean(same_time_volumes)
    same_time_std = _std(same_time_volumes)
    prior_close = prior_session_closes.get((row["symbol"], row["_session_date"]))
    expected_daily_volume = _mean([
        volume for (symbol, session), volume in session_total_volume.items()
        if symbol == row["symbol"] and session < row["_session_date"]
    ])
    out = {
        "return_5m": _return(history, 1),
        "return_15m": _return(history, 3),
        "return_30m": _return(history, 6),
        "return_60m": _return(history, 12),
        "session_to_date_return": _safe_return(close, session_open),
        "previous_session_return": prior_session_returns.get((row["symbol"], row["_session_date"])),
        "overnight_gap": _safe_return(session_open, prior_close),
        "distance_from_session_open": _safe_return(close, session_open),
        "distance_from_session_high": _safe_return(close, session_high),
        "distance_from_session_low": _safe_return(close, session_low),
        "distance_from_session_vwap": _safe_return(close, cum_vwap),
        "opening_range_position": _opening_range_position(close, history),
        "fraction_positive_bars_30m": _fraction_positive(returns[:6]),
        "intraday_trend_consistency": _trend_consistency(history[-6:]),
        "short_term_reversal": -_return(history, 3) if _return(history, 3) is not None else None,
        "realized_volatility_15m": _std([v for v in returns[:3] if v is not None]),
        "realized_volatility_30m": _std([v for v in returns[:6] if v is not None]),
        "realized_volatility_60m": _std([v for v in returns[:12] if v is not None]),
        "bar_high_low_range": current_range,
        "average_range_30m": _mean([v for v in ranges[-6:] if v is not None]),
        "range_expansion_ratio": (current_range / _mean([v for v in ranges[-6:-1] if v is not None])) if current_range is not None and _mean([v for v in ranges[-6:-1] if v is not None]) else None,
        "downside_volatility": _std([v for v in returns[:12] if v is not None and v < 0]),
        "maximum_recent_drawdown": _max_drawdown([float(item["close"]) for item in history[-12:]]),
        "bar_volume": float(row["volume"] or 0.0),
        "bar_dollar_volume": float(row["volume"] or 0.0) * close,
        "session_cumulative_volume": cum_volume,
        "fraction_expected_daily_volume": (cum_volume / expected_daily_volume) if expected_daily_volume else None,
        "relative_volume_same_time_of_day": (float(row["volume"] or 0.0) / same_time_avg) if same_time_avg else None,
        "volume_zscore_same_time_of_day": ((float(row["volume"] or 0.0) - same_time_avg) / same_time_std) if same_time_avg is not None and same_time_std else None,
        "volume_acceleration": _volume_acceleration(history),
        "price_change_per_dollar_volume": (_return(history, 1) / (float(row["volume"] or 0.0) * close)) if _return(history, 1) is not None and float(row["volume"] or 0.0) * close else None,
        "missing_bar_count_recent": _missing_bar_count(history, lookback=6),
        "SPY_return_5m": spy.get("return_5m"),
        "SPY_return_30m": spy.get("return_30m"),
        "SPY_session_return": spy.get("session_return"),
        "stock_minus_SPY_5m": _diff(_return(history, 1), spy.get("return_5m")),
        "stock_minus_SPY_30m": _diff(_return(history, 6), spy.get("return_30m")),
        "stock_minus_SPY_session": _diff(_safe_return(close, session_open), spy.get("session_return")),
        "sector_return_5m": None,
        "sector_return_30m": None,
        "stock_minus_sector_30m": None,
        "market_intraday_volatility": spy.get("volatility_30m"),
        "sample_market_breadth_positive_5m": None,
    }
    for column, value in list(out.items()):
        if column in feature_contract()["predictor_columns"]:
            out[f"{column}_missing"] = value is None
    return out


def _targets(
    row: Mapping[str, Any],
    future: list[dict[str, Any]],
    close_by_symbol_decision: Mapping[tuple[str, str], float],
) -> dict[str, Any]:
    close = float(row["close"])
    decision = row["_decision_utc"]
    horizons = {15: 3, 30: 6, 60: 12}
    output: dict[str, Any] = {
        "target_start_timestamp": decision.isoformat(),
        "target_definition_version": TARGET_CONTRACT_VERSION,
    }
    for minutes, bars in horizons.items():
        target_bar = future[bars - 1] if len(future) >= bars else None
        output[f"target_end_timestamp_{minutes}m"] = (decision + timedelta(minutes=minutes)).isoformat()
        output[f"forward_return_{minutes}m"] = _safe_return(float(target_bar["close"]), close) if target_bar else None
        spy_future_close = close_by_symbol_decision.get(("SPY", (decision + timedelta(minutes=minutes)).isoformat()))
        spy_now = close_by_symbol_decision.get(("SPY", decision.isoformat()))
        spy_return = _safe_return(spy_future_close, spy_now)
        output[f"forward_SPY_excess_return_{minutes}m"] = _diff(output[f"forward_return_{minutes}m"], spy_return)
    output["target_end_timestamp_to_close"] = future[-1]["_decision_utc"].isoformat() if future else decision.isoformat()
    output["forward_return_to_close"] = _safe_return(float(future[-1]["close"]), close) if future else None
    for minutes, bars in ((30, 6), (60, 12)):
        window = future[:bars]
        highs = [float(item["high"]) for item in window]
        lows = [float(item["low"]) for item in window]
        output[f"maximum_favourable_excursion_{minutes}m"] = _safe_return(max(highs), close) if highs else None
        output[f"maximum_adverse_excursion_{minutes}m"] = _safe_return(min(lows), close) if lows else None
    for minutes, bars in ((5, 1), (15, 3)):
        delayed = future[bars - 1] if len(future) >= bars else None
        delayed_close = float(delayed["close"]) if delayed else None
        output[f"buy_delay_value_{minutes}m"] = _safe_return(close, delayed_close)
        output[f"sell_delay_value_{minutes}m"] = _safe_return(delayed_close, close)
    output["forward_return_after_cost_proxy"] = output.get("forward_return_30m") - COST_PROXY_BPS / 10_000 if output.get("forward_return_30m") is not None else None
    output["SPY_excess_return_after_cost_proxy"] = output.get("forward_SPY_excess_return_30m") - COST_PROXY_BPS / 10_000 if output.get("forward_SPY_excess_return_30m") is not None else None
    output["delay_value_after_cost_proxy"] = output.get("buy_delay_value_5m") - COST_PROXY_BPS / 10_000 if output.get("buy_delay_value_5m") is not None else None
    output["trade_worth_executing_indicator"] = (
        output["forward_return_after_cost_proxy"] > COST_PROXY_BPS / 10_000
        if output.get("forward_return_after_cost_proxy") is not None
        else None
    )
    label_times = [
        _parse_dt(value)
        for key, value in output.items()
        if key.startswith("target_end_timestamp") and value
    ]
    output["label_available_timestamp"] = max(label_times).isoformat() if label_times else decision.isoformat()
    return output


def _add_cross_sectional_features(rows: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["decision_timestamp_utc"], []).append(row)
    for group in groups.values():
        breadth_values = [row.get("return_5m") for row in group if row.get("return_5m") is not None]
        breadth = sum(1 for value in breadth_values if value > 0) / len(breadth_values) if breadth_values else None
        for row in group:
            row["sample_market_breadth_positive_5m"] = breadth
        _rank(group, "return_30m", "momentum_rank")
        _rank(group, "relative_volume_same_time_of_day", "relative_volume_rank")
        _rank(group, "realized_volatility_30m", "volatility_rank")
        _rank(group, "session_to_date_return", "session_strength_rank")
        _rank(group, "stock_minus_SPY_30m", "market_relative_return_rank")
        by_sector: dict[str, list[dict[str, Any]]] = {}
        for row in group:
            if row.get("sector"):
                by_sector.setdefault(str(row["sector"]), []).append(row)
        for sector_rows in by_sector.values():
            sector_5 = _mean([row.get("return_5m") for row in sector_rows if row.get("return_5m") is not None])
            sector_30 = _mean([row.get("return_30m") for row in sector_rows if row.get("return_30m") is not None])
            for row in sector_rows:
                row["sector_return_5m"] = sector_5
                row["sector_return_30m"] = sector_30
                row["stock_minus_sector_30m"] = _diff(row.get("return_30m"), sector_30)


def _load_sector_map(path: Path) -> dict[str, str]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {str(symbol).upper(): str(sector) for symbol, sector in payload.items()}
    return {
        "AAPL": "Information Technology",
        "MSFT": "Information Technology",
        "SPY": "Broad Market",
    }


def _rank(rows: list[dict[str, Any]], source: str, target: str) -> None:
    available = sorted((row for row in rows if row.get(source) is not None), key=lambda row: (row[source], row["symbol"]))
    denom = max(1, len(available) - 1)
    ranks = {id(row): index / denom for index, row in enumerate(available)}
    for row in rows:
        row[target] = ranks.get(id(row))
        row[f"{target}_missing"] = row[target] is None


def _selector_fixture(cfg: Ticket10AConfig) -> dict[str, dict[str, Any]]:
    as_of = datetime.fromisoformat(cfg.selector_as_of_exchange).astimezone(timezone.utc).isoformat()
    return {
        "AAPL": _selector_row(as_of, 0.82, 1, 0.04),
        "MSFT": _selector_row(as_of, 0.74, 2, 0.03),
        "SPY": _selector_row(as_of, 0.50, 3, 0.00),
    }


def _selector_row(as_of: str, score: float, rank: int, weight: float) -> dict[str, Any]:
    return {
        "selector_as_of_timestamp": as_of,
        "selector_model_identity": "ticket_10a_bounded_selector_fixture",
        "selector_input_artifact_identity": "ticket_10a_selector_input_fixture_no_training",
        "selector_score": score,
        "selector_rank": rank,
        "desired_portfolio_weight": weight,
        "selector_horizon": "10_trading_days",
    }


def _portfolio_fixture(cfg: Ticket10AConfig) -> dict[str, dict[str, Any]]:
    as_of = datetime.fromisoformat(cfg.selector_as_of_exchange).astimezone(timezone.utc).isoformat()
    return {
        "AAPL": _portfolio_row(as_of, 25, 0.015, 190.0, 0.012, 390),
        "MSFT": _portfolio_row(as_of, 0, 0.0, None, 0.0, 9999),
        "SPY": _portfolio_row(as_of, 0, 0.0, None, 0.0, 9999),
    }


def _portfolio_row(as_of: str, qty: int, weight: float, entry: float | None, unrealized: float, minutes: int) -> dict[str, Any]:
    return {
        "portfolio_state_as_of_timestamp": as_of,
        "current_position_quantity": qty,
        "current_portfolio_weight": weight,
        "pending_order_quantity": 0,
        "average_entry_price": entry,
        "unrealized_return": unrealized,
        "minutes_since_last_trade": minutes,
        "current_turnover_used": 0.08,
    }


def _source_identity(path: Path, cfg: Ticket10AConfig) -> dict[str, Any]:
    raw_dir = cfg.raw_root / path.relative_to(cfg.parquet_root).parent
    manifest_path = raw_dir / "manifest.json"
    tombstone_path = raw_dir / "parquet_conversion.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    tombstone = json.loads(tombstone_path.read_text(encoding="utf-8")) if tombstone_path.exists() else {}
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "manifest_path": str(manifest_path),
        "tombstone_path": str(tombstone_path),
        "feed": manifest.get("feed"),
        "adjustment_mode": manifest.get("adjustment_mode"),
        "completion_state": manifest.get("completion_state"),
        "tombstone_validation_result": tombstone.get("validation_result"),
    }


def _sample_identity(cfg: Ticket10AConfig, sources: list[Mapping[str, Any]]) -> str:
    return _stable_hash({
        "symbols": cfg.symbols,
        "sessions": cfg.sessions,
        "feed": cfg.feed,
        "source_sha256": [row["sha256"] for row in sources],
        "decision_grid_version": DECISION_GRID_VERSION,
    })


def _summary_markdown(manifest: Mapping[str, Any], validation: Mapping[str, Any]) -> str:
    return "\n".join([
        "# Ticket 10A Five-Minute Execution Dataset Probe",
        "",
        f"Status: `{manifest['status']}`",
        "",
        f"- Dataset: `{manifest['artifact_path']}`",
        f"- Rows: {manifest['row_count']}",
        f"- Symbols: {', '.join(manifest['sample_symbols'])}",
        f"- Sessions: {', '.join(manifest['sample_sessions'])}",
        f"- Feature contract: `{manifest['feature_contract_version']}`",
        f"- Target contract: `{manifest['target_contract_version']}`",
        f"- Duplicate keys: {validation['duplicate_symbol_decision_keys']}",
        f"- Future-feature violations: {validation['future_feature_violations']}",
        f"- Target timestamp violations: {validation['target_timestamp_violations']}",
        "",
        "Action mapping is documented only: WAIT, BUY_SMALL, BUY_LARGE, SELL_SMALL, SELL_LARGE, EXIT require no-trade thresholds, cost advantage, cooldowns, turnover, position, and exposure limits before any live use.",
    ])


def _is_regular_decision(local_decision: datetime) -> bool:
    close = rth_close_for_date(local_decision.date())
    if close is None:
        return False
    return time(9, 35) <= local_decision.time() <= time(15, 55) and local_decision.time() <= close


def _session_total_volume(groups: Mapping[tuple[str, str], list[dict[str, Any]]]) -> dict[tuple[str, str], float]:
    return {key: sum(float(row["volume"] or 0.0) for row in rows) for key, rows in groups.items()}


def _attach_prior_same_time_volumes(groups: Mapping[tuple[str, str], list[dict[str, Any]]]) -> None:
    output: dict[tuple[str, time], list[float]] = {}
    for (symbol, session), rows in sorted(groups.items()):
        for row in rows:
            key = (symbol, row["_decision_exchange"].time())
            output.setdefault(key, [])
            row["_prior_same_time_volumes"] = list(output[key])
        for row in rows:
            output[(symbol, row["_decision_exchange"].time())].append(float(row["volume"] or 0.0))


def _prior_session_returns(groups: Mapping[tuple[str, str], list[dict[str, Any]]]) -> dict[tuple[str, str], float | None]:
    output = {}
    last_return: dict[str, float | None] = {}
    for (symbol, session), rows in sorted(groups.items()):
        output[(symbol, session)] = last_return.get(symbol)
        last_return[symbol] = _safe_return(float(rows[-1]["close"]), float(rows[0]["open"])) if rows else None
    return output


def _prior_session_closes(groups: Mapping[tuple[str, str], list[dict[str, Any]]]) -> dict[tuple[str, str], float | None]:
    output = {}
    last_close: dict[str, float | None] = {}
    for (symbol, session), rows in sorted(groups.items()):
        output[(symbol, session)] = last_close.get(symbol)
        last_close[symbol] = float(rows[-1]["close"]) if rows else None
    return output


def _spy_context(
    close_by_symbol_decision: Mapping[tuple[str, str], float],
    decision_iso: str,
    row: Mapping[str, Any],
    groups: Mapping[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, float | None]:
    spy_rows = [item for item in groups.get(("SPY", row["_session_date"]), []) if item["_decision_utc"] <= row["_decision_utc"]]
    if not spy_rows:
        return {"return_5m": None, "return_30m": None, "session_return": None, "volatility_30m": None}
    spy_close = close_by_symbol_decision.get(("SPY", decision_iso))
    returns = [_bar_return(spy_rows, offset) for offset in range(1, len(spy_rows))]
    return {
        "return_5m": _return(spy_rows, 1),
        "return_30m": _return(spy_rows, 6),
        "session_return": _safe_return(spy_close, float(spy_rows[0]["open"])),
        "volatility_30m": _std([value for value in returns[:6] if value is not None]),
    }


def _return(history: list[Mapping[str, Any]], bars_back: int) -> float | None:
    if len(history) <= bars_back:
        return None
    return _safe_return(float(history[-1]["close"]), float(history[-1 - bars_back]["close"]))


def _bar_return(history: list[Mapping[str, Any]], offset_from_end: int) -> float | None:
    if len(history) <= offset_from_end:
        return None
    return _safe_return(float(history[-offset_from_end]["close"]), float(history[-offset_from_end - 1]["close"]))


def _safe_return(new: float | None, old: float | None) -> float | None:
    if new is None or old in (None, 0):
        return None
    return new / old - 1.0


def _diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(clean) if clean else None


def _std(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.pstdev(clean) if len(clean) >= 2 else None


def _fraction_positive(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return sum(1 for value in clean if value > 0) / len(clean) if clean else None


def _trend_consistency(rows: list[Mapping[str, Any]]) -> float | None:
    returns = [_bar_return(rows, offset) for offset in range(1, len(rows))]
    return _fraction_positive(returns)


def _opening_range_position(close: float, history: list[Mapping[str, Any]]) -> float | None:
    opening = history[:6]
    if len(opening) < 6:
        return None
    low = min(float(row["low"]) for row in opening)
    high = max(float(row["high"]) for row in opening)
    return (close - low) / (high - low) if high > low else None


def _max_drawdown(values: list[float]) -> float | None:
    if not values:
        return None
    peak = values[0]
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        if peak:
            drawdown = min(drawdown, value / peak - 1.0)
    return drawdown


def _volume_acceleration(history: list[Mapping[str, Any]]) -> float | None:
    if len(history) < 4:
        return None
    recent = _mean(float(row["volume"] or 0.0) for row in history[-3:])
    prior = _mean(float(row["volume"] or 0.0) for row in history[-6:-3])
    return (recent / prior - 1.0) if recent is not None and prior else None


def _missing_bar_count(history: list[Mapping[str, Any]], *, lookback: int) -> int:
    if not history:
        return lookback
    times = {row["_decision_utc"] for row in history[-lookback:]}
    current = history[-1]["_decision_utc"]
    expected = {current - timedelta(minutes=5 * offset) for offset in range(lookback)}
    return len(expected - times)


def _trade_side(remaining_weight: float) -> str:
    if remaining_weight > 0:
        return "buy"
    if remaining_weight < 0:
        return "sell"
    return "none"


def _duplicate_count(keys: Iterable[tuple[Any, ...]]) -> int:
    items = list(keys)
    return len(items) - len(set(items))


def _source_row_fingerprint(row: Mapping[str, Any]) -> str:
    payload = {key: row.get(key) for key in ("symbol", "timestamp", "open", "high", "low", "close", "volume", "vwap")}
    return _stable_hash(payload)


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_dt(value: str) -> datetime:
    return _to_utc(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
