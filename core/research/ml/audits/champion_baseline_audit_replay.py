from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from core.entities.candle import Candle
from core.research.dual_momentum.factory import build_dual_momentum_tester
from core.research.ml.audits.champion_baseline_audit_io import (
    _champion_config_path,
    _read_yaml,
)
from core.research.ml.audits.champion_baseline_audit_math import (
    _annualized_return,
    _compound_returns,
    _equity_curve,
    _max_drawdown,
    _sharpe,
    _sortino,
)
from core.research.ml.audits.champion_baseline_audit_types import RESEARCH_METADATA
from core.research.performance_metrics import calmar_ratio


def exact_champion_replay_from_equity(
    *,
    periods: list[dict[str, str]],
    equity_curve: list[Any],
    selections: list[Any],
    champion_config: dict[str, Any],
    candles_by_symbol: dict[str, list[Candle]] | None = None,
) -> dict[str, Any]:
    equity_by_date = {
        point.timestamp.date().isoformat(): float(point.equity)
        for point in equity_curve
    }
    selection_by_date = _selection_lookup(selections)
    rows = []
    for period in periods:
        start = period["rebalance_date"]
        end = period["outcome_end_date"]
        start_equity = equity_by_date.get(start)
        end_equity = equity_by_date.get(end)
        if start_equity is None or end_equity is None or start_equity <= 0:
            continue
        selection = _selection_at_or_before(selection_by_date, start)
        selected_symbols = list(getattr(selection, "symbols", []) or [])
        target_weights = dict(getattr(selection, "target_weights", {}) or {})
        period_return = (end_equity / start_equity) - 1.0
        rows.append({
            "rebalance_date": start,
            "outcome_end_date": end,
            "period_return": period_return,
            "start_equity": start_equity,
            "end_equity": end_equity,
            "selected_symbols": selected_symbols,
            "target_weights": target_weights,
            "exposure_target": getattr(selection, "exposure_target", None),
            "regime_label": getattr(selection, "regime_label", None),
            "symbol_return_anomalies": _symbol_return_anomalies(
                selected_symbols,
                start,
                end,
                candles_by_symbol or {},
            ),
        })
    period_summary = _period_grid_summary(rows)
    continuous = _continuous_summary(equity_by_date, periods)
    target_exposure = champion_config.get("overrides", {}).get(
        "target_exposure",
        champion_config.get("target_exposure"),
    )
    return {
        "available": bool(rows),
        "availability_reason": None if rows else "no exact replay rows matched periods",
        "stooq_adjusted_status": "unknown",
        "summary": {
            "baseline_name": "exact_champion_replay",
            "semantic_type": "exact_champion_replay",
            "available": bool(rows),
            "is_exact_champion_replay": True,
            "target_exposure": target_exposure,
            "total_return": period_summary.get("total_return"),
            "continuous_total_return": continuous.get("total_return"),
            "max_drawdown": period_summary.get("max_drawdown"),
            "turnover": None,
            "costs": None,
            "cost_turnover_status": (
                "handled_inside_dual_momentum_backtester_equity_curve; "
                "period-level cost attribution unavailable from replay artifact"
            ),
            **RESEARCH_METADATA,
        },
        "champion_config": champion_config,
        "period_grid_summary": period_summary,
        "continuous_equity_summary": continuous,
        "period_rows": rows,
        **RESEARCH_METADATA,
    }


def _try_exact_champion_replay(
    config: dict[str, Any],
    periods: list[dict[str, str]],
) -> dict[str, Any]:
    champion_config = _read_yaml(_champion_config_path(config))
    if not periods:
        return _unavailable_exact_replay(
            champion_config,
            "no holdout evaluation periods were found",
        )
    try:
        dual_config = _active_champion_config(config, champion_config)
        candles_by_symbol = _load_replay_candles(config, dual_config)
        result = build_dual_momentum_tester(config, dual_config).run(candles_by_symbol)
        replay = exact_champion_replay_from_equity(
            periods=periods,
            equity_curve=result.result.equity_curve,
            selections=result.selections,
            champion_config=champion_config,
            candles_by_symbol=candles_by_symbol,
        )
        replay["replay_metadata"] = {
            "source": "research_only_dual_momentum_replay",
            "symbol_count": len(candles_by_symbol),
            "available_symbols": sorted(candles_by_symbol),
            "benchmark_symbol": dual_config.get("benchmark_symbol", "SPY"),
            "universe_path": dual_config.get("universe_path"),
            "max_symbols": dual_config.get("max_symbols"),
        }
        return replay
    except Exception as exc:
        return _unavailable_exact_replay(champion_config, str(exc))


def _active_champion_config(
    config: dict[str, Any],
    champion_config: dict[str, Any],
) -> dict[str, Any]:
    dual_config = deepcopy(config.get("research", {}).get("dual_momentum", {}))
    dual_config.update(champion_config.get("overrides", {}))
    dual_config["champion_id"] = champion_config.get(
        "champion_id",
        dual_config.get("champion_id"),
    )
    dual_config["champion_source_config_name"] = champion_config.get(
        "source_config_name",
        dual_config.get("champion_source_config_name"),
    )
    dual_config["champion_config_path"] = str(_champion_config_path(config))
    universe_path = Path(str(dual_config.get("universe_path", "")))
    if universe_path.exists():
        payload = _read_yaml(universe_path)
        symbols = [str(symbol).upper() for symbol in payload.get("symbols", [])]
        max_symbols = int(dual_config.get("max_symbols") or len(symbols))
        dual_config["symbols"] = symbols[:max_symbols]
    return dual_config


def _load_replay_candles(
    config: dict[str, Any],
    dual_config: dict[str, Any],
) -> dict[str, list[Candle]]:
    parquet_dir = Path(
        config.get("ml", {}).get("stooq_parquet_dir", "data/processed/stooq_parquet")
    )
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Stooq parquet directory not found: {parquet_dir}")
    required_symbols = _required_replay_symbols(dual_config)
    candles_by_symbol = {}
    missing = []
    for symbol in required_symbols:
        path = parquet_dir / f"{symbol.upper()}.parquet"
        if not path.exists():
            missing.append(symbol)
            continue
        candles = _read_parquet_candles(path, symbol.upper())
        if candles:
            candles_by_symbol[symbol.upper()] = candles
    benchmark = str(dual_config.get("benchmark_symbol", "SPY")).upper()
    if benchmark not in candles_by_symbol:
        raise RuntimeError(f"Benchmark symbol {benchmark} was not available")
    if len(candles_by_symbol) < 2:
        raise RuntimeError(
            "Exact replay needs at least benchmark plus one tradable symbol"
        )
    if missing:
        # Missing symbols are expected when a nominal 500-symbol universe has
        # fewer local Stooq histories; the replay uses available local symbols.
        pass
    return candles_by_symbol


def _read_parquet_candles(path: Path, symbol: str) -> list[Candle]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Exact champion replay requires pyarrow to read Stooq parquet data"
        ) from exc
    table = pq.read_table(path, columns=[
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ])
    data = table.to_pydict()
    return sorted(
        [
            Candle(
                symbol=symbol,
                timestamp=value,
                open=float(open_price),
                high=float(high_price),
                low=float(low_price),
                close=float(close_price),
                volume=float(volume),
            )
            for value, open_price, high_price, low_price, close_price, volume in zip(
                data["timestamp"],
                data["open"],
                data["high"],
                data["low"],
                data["close"],
                data["volume"],
            )
        ],
        key=lambda candle: candle.timestamp,
    )


def _required_replay_symbols(dual_config: dict[str, Any]) -> list[str]:
    symbols = set(str(symbol).upper() for symbol in dual_config.get("symbols", []))
    for key in (
        "benchmark_symbol",
        "regime_symbol",
        "relative_strength_symbol",
        "volatility_shock_symbol",
        "leadership_symbol",
        "relative_strength_filter_symbol",
    ):
        if dual_config.get(key):
            symbols.add(str(dual_config[key]).upper())
    for key in (
        "regime_confirmation_symbols",
        "risk_off_symbols",
        "fallback_symbols",
        "benchmark_sleeve_symbols",
        "fast_reentry_symbols",
    ):
        symbols.update(str(symbol).upper() for symbol in dual_config.get(key, []) or [])
    return sorted(symbols)


def _period_grid_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [float(row["period_return"]) for row in rows]
    equity_curve = _equity_curve(returns)
    total = _compound_returns(returns)
    drawdown = _max_drawdown([1.0] + equity_curve)
    annualized = _annualized_return(total, rows)
    return {
        "evaluation_mode": "same_forward_period_grid_as_ml_allocation",
        "start_date": rows[0]["rebalance_date"] if rows else None,
        "end_date": rows[-1]["rebalance_date"] if rows else None,
        "last_outcome_end_date": rows[-1]["outcome_end_date"] if rows else None,
        "period_count": len(rows),
        "total_return": total,
        "annualized_return": annualized,
        "max_drawdown": drawdown,
        "sharpe": _sharpe(returns, rows),
        "sortino": _sortino(returns, rows),
        "calmar": calmar_ratio(annualized if annualized is not None else total, drawdown),
        "largest_positive_period": max(returns, default=None),
        "largest_negative_period": min(returns, default=None),
        "top_20_rebalance_dates": _top_periods(rows, reverse=True),
        "worst_20_rebalance_dates": _top_periods(rows, reverse=False),
    }


def _continuous_summary(
    equity_by_date: dict[str, float],
    periods: list[dict[str, str]],
) -> dict[str, Any]:
    if not periods:
        return {"available": False}
    start = periods[0]["rebalance_date"]
    end = periods[-1]["outcome_end_date"]
    start_equity = equity_by_date.get(start)
    end_equity = equity_by_date.get(end)
    if start_equity is None or end_equity is None or start_equity <= 0:
        return {
            "available": False,
            "reason": "exact start/end equity dates were unavailable",
            "start_date": start,
            "end_date": end,
        }
    path = [
        {"date": date, "equity": equity}
        for date, equity in sorted(equity_by_date.items())
        if start <= date <= end
    ]
    return {
        "available": True,
        "evaluation_mode": "continuous_strategy_equity_start_to_last_outcome_end",
        "start_date": start,
        "end_date": end,
        "starting_equity": start_equity,
        "ending_equity": end_equity,
        "total_return": (end_equity / start_equity) - 1.0,
        "max_drawdown": _max_drawdown([row["equity"] for row in path]),
        "equity_points": len(path),
    }


def _unavailable_exact_replay(
    champion_config: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "available": False,
        "availability_reason": reason,
        "stooq_adjusted_status": "unknown",
        "summary": {
            "baseline_name": "exact_champion_replay",
            "semantic_type": "exact_champion_replay",
            "available": False,
            "skip_reason": reason,
            "is_exact_champion_replay": True,
            **RESEARCH_METADATA,
        },
        "champion_config": champion_config,
        "period_grid_summary": {},
        "continuous_equity_summary": {},
        "period_rows": [],
        **RESEARCH_METADATA,
    }


def _selection_lookup(selections: list[Any]) -> dict[str, Any]:
    return {
        selection.timestamp.date().isoformat(): selection
        for selection in selections
    }


def _selection_at_or_before(
    selection_by_date: dict[str, Any],
    date: str,
) -> Any | None:
    candidates = [
        key for key in selection_by_date
        if key <= date
    ]
    if not candidates:
        return None
    return selection_by_date[max(candidates)]


def _symbol_return_anomalies(
    symbols: list[str],
    start: str,
    end: str,
    candles_by_symbol: dict[str, list[Candle]],
) -> list[dict[str, Any]]:
    anomalies = []
    for symbol in symbols:
        closes = {
            candle.timestamp.date().isoformat(): candle.close
            for candle in candles_by_symbol.get(symbol, [])
        }
        start_close = closes.get(start)
        end_close = closes.get(end)
        if start_close is None or end_close is None or start_close <= 0:
            continue
        return_value = (end_close / start_close) - 1.0
        if return_value > 1.0 or return_value < -0.50:
            anomalies.append({
                "symbol": symbol,
                "start_date": start,
                "end_date": end,
                "start_close": start_close,
                "end_close": end_close,
                "return": return_value,
            })
    return anomalies


def _top_periods(rows: list[dict[str, Any]], *, reverse: bool) -> list[dict[str, Any]]:
    return [
        {
            "rebalance_date": row["rebalance_date"],
            "outcome_end_date": row["outcome_end_date"],
            "period_return": row["period_return"],
            "selected_symbols": row.get("selected_symbols", []),
            "target_weights": row.get("target_weights", {}),
            "exposure_target": row.get("exposure_target"),
            "regime_label": row.get("regime_label"),
            "symbol_return_anomalies": row.get("symbol_return_anomalies", []),
        }
        for row in sorted(
            rows,
            key=lambda item: float(item["period_return"]),
            reverse=reverse,
        )[:20]
    ]
