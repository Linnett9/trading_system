from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Iterable, Mapping

from core.research.ml.stock_level.news_risk_overlay_research_accounting import (
    adverse_excursion as _adverse_excursion,
    expected_shortfall_cvar as _expected_shortfall,
)
from core.research.ml.stock_level.news_risk_overlay_research_parallel import (
    NewsRiskParallelConfig,
    chunks as _chunks,
    parallel_config as _parallel_config,
    record_parallel_phase as _record_parallel_phase,
    record_worker_failures as _record_worker_failures,
    should_parallelize as _should_parallelize,
)

RETURN_COLUMNS = (
    "actual_forward_return_10d",
    "actual_forward_return_5d",
    "forward_return",
)


def _run_open_trade_replay(
    rows: list[Mapping[str, Any]],
    *,
    bars_by_symbol: Mapping[str, list[Mapping[str, Any]]],
    price_score_column: str,
    variant: str,
    variant_settings: Mapping[str, Any],
    replay_config: Mapping[str, Any],
) -> dict[str, Any]:
    by_date: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        symbol = str(row.get("symbol", "")).upper()
        if symbol in bars_by_symbol and _number(row.get(price_score_column)) is not None:
            by_date.setdefault(str(row.get("decision_timestamp", row.get("rebalance_date", "")))[:10], []).append(row)
    bar_lookup = _bar_lookup(bars_by_symbol)
    next_lookup = _next_bar_lookup(bars_by_symbol)
    first_decision = min(by_date, default="9999-12-31")
    all_dates = sorted(
        set(by_date)
        | {
            bar["date"]
            for bars in bars_by_symbol.values()
            for bar in bars
            if bar["date"] >= first_decision
        }
    )
    cash = float(replay_config["starting_equity"])
    open_positions: list[dict[str, Any]] = []
    pending_entries: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    daily_equity: list[dict[str, Any]] = []
    action_events: list[dict[str, Any]] = []
    trade_counter = 0
    previous_equity = cash
    for current_date in all_dates:
        pending_now = [item for item in pending_entries if item["entry_date"] == current_date]
        pending_entries = [item for item in pending_entries if item["entry_date"] != current_date]
        for item in pending_now:
            bar = _bar_on_fast(item["symbol"], current_date, bar_lookup)
            if not bar:
                continue
            entry_price = float(bar["open"]) * (1.0 + float(replay_config["entry_slippage_bps"]) / 10_000.0)
            commission = item["cash_committed"] * float(replay_config["commission_bps"]) / 10_000.0
            if item["cash_committed"] + commission > cash + 1e-12:
                item["skip_reason"] = "insufficient_cash_at_entry"
                continue
            shares = item["cash_committed"] / entry_price if entry_price > 0 else 0.0
            cash -= item["cash_committed"] + commission
            item.update({"entry_price": entry_price, "shares": shares, "entry_commission": commission, "entry_timestamp": current_date})
            open_positions.append(item)
        still_open = []
        for position in open_positions:
            bar = _bar_on_fast(position["symbol"], current_date, bar_lookup)
            if not bar:
                still_open.append(position)
                continue
            position["bars_held"] += 1
            position["maximum_adverse_excursion"] = min(position["maximum_adverse_excursion"], float(bar["low"]) / position["entry_price"] - 1.0)
            position["maximum_favourable_excursion"] = max(position["maximum_favourable_excursion"], float(bar["high"]) / position["entry_price"] - 1.0)
            exit_price, exit_reason = _exit_decision(position, bar, replay_config)
            if exit_price is None:
                still_open.append(position)
                continue
            exit_price *= 1.0 - float(replay_config["exit_slippage_bps"]) / 10_000.0
            gross_pnl = (exit_price - position["entry_price"]) * position["shares"]
            exit_commission = (exit_price * position["shares"]) * float(replay_config["commission_bps"]) / 10_000.0
            total_costs = position["entry_commission"] + exit_commission
            net_pnl = gross_pnl - total_costs
            cash += exit_price * position["shares"] - exit_commission
            ledger.append(_ledger_row(position, exit_price, exit_reason, gross_pnl, total_costs, net_pnl, current_date))
        open_positions = still_open
        if current_date in by_date:
            equity = _equity_fast(cash, open_positions, bar_lookup, current_date)
            ranked = sorted(
                by_date[current_date],
                key=lambda row: (-_variant_sort_value(row, price_score_column, variant_settings), str(row.get("symbol", ""))),
            )
            for rank, candidate in enumerate(ranked, start=1):
                if len(open_positions) + len(pending_entries) >= int(replay_config["max_positions"]):
                    break
                if _has_symbol(candidate, open_positions, pending_entries):
                    continue
                action = str(candidate.get("news_action") or "NO_COVERAGE")
                multiplier, blocked = _variant_multiplier(action, variant_settings, replay_config)
                action_events.append(_action_event(candidate, variant, action, blocked, rank))
                if blocked:
                    if bool(variant_settings.get("replace_blocked")):
                        continue
                    if len(action_events) >= int(replay_config["top_n"]):
                        break
                    continue
                entry_date = _next_bar_date_fast(str(candidate["symbol"]).upper(), current_date, next_lookup)
                if not entry_date:
                    continue
                allocation = min(
                    cash,
                    equity * float(replay_config["max_position_weight"]) * multiplier,
                )
                if allocation <= 0:
                    continue
                trade_counter += 1
                pending_entries.append(
                    _pending_trade(
                        trade_counter,
                        candidate,
                        variant,
                        price_score_column,
                        entry_date,
                        allocation,
                        replay_config,
                        rank,
                    )
                )
                if len([p for p in pending_entries if p.get("decision_timestamp", "")[:10] == current_date]) >= int(replay_config["top_n"]):
                    break
        equity = _equity_fast(cash, open_positions, bar_lookup, current_date)
        daily_return = equity / previous_equity - 1.0 if previous_equity else 0.0
        previous_equity = equity
        daily_equity.append(
            {
                "date": current_date,
                "strategy_variant": variant,
                "cash": cash,
                "position_market_value": equity - cash,
                "total_equity": equity,
                "daily_return": daily_return,
                "gross_exposure": (equity - cash) / equity if equity else 0.0,
                "net_exposure": (equity - cash) / equity if equity else 0.0,
                "concurrent_positions": len(open_positions),
            }
        )
    for position in open_positions:
        last_bar = _last_bar(position["symbol"], bars_by_symbol)
        if last_bar:
            exit_price = float(last_bar["close"])
            gross_pnl = (exit_price - position["entry_price"]) * position["shares"]
            exit_commission = (exit_price * position["shares"]) * float(replay_config["commission_bps"]) / 10_000.0
            total_costs = position["entry_commission"] + exit_commission
            ledger.append(_ledger_row(position, exit_price, "end_of_data", gross_pnl, total_costs, gross_pnl - total_costs, last_bar["date"]))
    return {"ledger": ledger, "daily_equity": daily_equity, "action_events": action_events}


def _load_daily_price_bars(
    symbols: list[str],
    processed_root: Path,
    *,
    parallel_config: NewsRiskParallelConfig | None = None,
    parallel_report: dict[str, Any] | None = None,
    load_daily_price_bar_file_fn: Callable[[str, Path], dict[str, Any]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    ordered_symbols = sorted(dict.fromkeys(str(symbol).upper() for symbol in symbols))
    loader = load_daily_price_bar_file_fn or _load_daily_price_bar_file
    bars_by_symbol: dict[str, list[dict[str, Any]]] = {}
    missing = []
    failures = []
    task_durations = []
    config = parallel_config or _parallel_config({})
    use_parallel = _should_parallelize(config, len(ordered_symbols), phase="bar_loading", report=parallel_report)
    if use_parallel:
        if config.progress:
            print(
                f"Loading bars: 0 / {len(ordered_symbols)} symbols using {config.actual_workers} workers",
                flush=True,
            )
        completed = 0
        effective_chunk_size = min(config.chunk_size, config.batch_limit)
        for chunk in _chunks(ordered_symbols, effective_chunk_size):
            with ThreadPoolExecutor(max_workers=config.actual_workers) as executor:
                futures = {
                    executor.submit(loader, symbol, processed_root): symbol
                    for symbol in chunk
                }
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        failures.append({"task_id": symbol, "error": str(exc)})
                        continue
                    completed += 1
                    task_durations.append(float(result["elapsed_seconds"]))
                    if result["status"] == "MISSING":
                        missing.append(symbol)
                    elif result["status"] == "OK":
                        bars_by_symbol[symbol] = result["rows"]
                    else:
                        failures.append({"task_id": symbol, "error": result.get("error", "malformed worker result")})
                    if config.progress and (completed % max(effective_chunk_size, 1) == 0 or completed == len(ordered_symbols)):
                        print(
                            f"Loading bars: {completed} / {len(ordered_symbols)} symbols using {config.actual_workers} workers",
                            flush=True,
                        )
    else:
        for symbol in ordered_symbols:
            result = loader(symbol, processed_root)
            task_durations.append(float(result["elapsed_seconds"]))
            if result["status"] == "MISSING":
                missing.append(symbol)
            elif result["status"] == "OK":
                bars_by_symbol[symbol] = result["rows"]
            else:
                failures.append({"task_id": symbol, "error": result.get("error", "malformed worker result")})
    if failures:
        _record_worker_failures(parallel_report, "bar_loading", failures)
        first = failures[0]
        raise ValueError(f"daily bar worker failed for {first['task_id']}: {first['error']}")
    bars_by_symbol = {
        symbol: sorted(rows, key=lambda row: row["date"])
        for symbol, rows in sorted(bars_by_symbol.items())
    }
    _record_parallel_phase(
        parallel_report,
        "bar_loading",
        task_count=len(ordered_symbols),
        task_durations=task_durations,
        parallelized=use_parallel,
    )
    audit = {
        "processed_root": str(processed_root),
        "requested_symbol_count": len(ordered_symbols),
        "loaded_symbol_count": len(bars_by_symbol),
        "missing_symbol_count": len(missing),
        "missing_symbols": missing[:100],
        "timeframe": "1Day",
        "required_columns": ["timestamp", "open", "high", "low", "close", "volume", "symbol"],
        "adjusted_status": "local canonical bars; adjustment metadata not explicit in parquet schema",
        "split_handling": "not explicit in parquet schema",
        "dividend_handling": "not explicit in parquet schema",
        "entry_price_available_at_decision": False,
        "entry_convention": "next available daily bar open after decision date",
    }
    return bars_by_symbol, audit


def _variant_multiplier(
    action: str,
    settings: Mapping[str, Any],
    replay_config: Mapping[str, Any],
) -> tuple[float, bool]:
    if not settings.get("use_news"):
        return 1.0, False
    if settings.get("inverted"):
        if action == "ALLOW":
            return 0.0, True
        if action == "REDUCE":
            return float(replay_config["reduce_multiplier"]), False
        return 1.0, False
    if action == "BLOCK":
        return 0.0, True
    if action == "REDUCE":
        if settings.get("reduce"):
            return float(replay_config["reduce_multiplier"]), False
        if settings.get("strict_gate"):
            return 0.0, True
    return 1.0, False


def _variant_sort_value(
    row: Mapping[str, Any],
    price_score_column: str,
    settings: Mapping[str, Any],
) -> float:
    price_score = _number(row.get(price_score_column)) or 0.0
    if not settings.get("contrarian_rerank"):
        return price_score
    news_shock_score = _number(row.get("price_plus_news_risk_probability")) or 0.0
    weight = float(settings.get("contrarian_weight", 0.0))
    return price_score + weight * news_shock_score


def _pending_trade(
    trade_number: int,
    candidate: Mapping[str, Any],
    variant: str,
    price_score_column: str,
    entry_date: str,
    allocation: float,
    replay_config: Mapping[str, Any],
    ranking_after_news: int | None = None,
) -> dict[str, Any]:
    symbol = str(candidate["symbol"]).upper()
    return {
        "trade_id": f"{variant}-{trade_number:08d}",
        "strategy_variant": variant,
        "candidate_id": candidate.get("candidate_id", ""),
        "decision_timestamp": candidate.get("decision_timestamp", candidate.get("rebalance_date", "")),
        "symbol": symbol,
        "direction": "LONG",
        "price_score": _number(candidate.get(price_score_column)) or 0.0,
        "news_risk_probability": _number(candidate.get("price_plus_news_risk_probability")),
        "combined_score": (_number(candidate.get(price_score_column)) or 0.0) - (_number(candidate.get("price_plus_news_risk_probability")) or 0.0),
        "news_action": candidate.get("news_action", "NO_COVERAGE"),
        "news_coverage": candidate.get("news_coverage_status", "NO_COVERAGE"),
        "ranking_before_news": "",
        "ranking_after_news": ranking_after_news if ranking_after_news is not None else "",
        "replaced_candidate": False,
        "entry_date": entry_date,
        "cash_committed": allocation,
        "proposed_size": allocation,
        "actual_size": allocation,
        "stop": "",
        "target": "",
        "maximum_holding_period": replay_config["max_holding_bars"],
        "model_version": candidate.get("model_version", "news-risk-overlay-research-v1"),
        "price_feature_timestamp": candidate.get("decision_timestamp", ""),
        "news_feature_timestamp": candidate.get("news_feature_timestamp", ""),
        "bars_held": 0,
        "maximum_adverse_excursion": 0.0,
        "maximum_favourable_excursion": 0.0,
    }


def _exit_decision(
    position: Mapping[str, Any],
    bar: Mapping[str, Any],
    replay_config: Mapping[str, Any],
) -> tuple[float | None, str | None]:
    stop_loss = replay_config.get("stop_loss_pct")
    profit_target = replay_config.get("profit_target_pct")
    entry = float(position["entry_price"])
    stop_price = entry * (1.0 - float(stop_loss)) if stop_loss is not None else None
    target_price = entry * (1.0 + float(profit_target)) if profit_target is not None else None
    stop_hit = stop_price is not None and float(bar["low"]) <= stop_price
    target_hit = target_price is not None and float(bar["high"]) >= target_price
    if stop_hit:
        return float(stop_price), "stop_hit_conservative_before_target"
    if target_hit:
        return float(target_price), "target_hit"
    if int(position["bars_held"]) >= int(replay_config["max_holding_bars"]):
        return float(bar["close"]), "time_exit"
    return None, None


def _ledger_row(
    position: Mapping[str, Any],
    exit_price: float,
    exit_reason: str,
    gross_pnl: float,
    costs: float,
    net_pnl: float,
    exit_date: str,
) -> dict[str, Any]:
    committed = float(position["cash_committed"])
    return {
        "trade_id": position["trade_id"],
        "strategy_variant": position["strategy_variant"],
        "candidate_id": position.get("candidate_id", ""),
        "decision_timestamp": position["decision_timestamp"],
        "symbol": position["symbol"],
        "direction": position["direction"],
        "price_score": position["price_score"],
        "news_risk_probability": position["news_risk_probability"],
        "combined_score": position["combined_score"],
        "news_action": position["news_action"],
        "news_coverage": position["news_coverage"],
        "ranking_before_news": position["ranking_before_news"],
        "ranking_after_news": position["ranking_after_news"],
        "replaced_candidate": position["replaced_candidate"],
        "entry_timestamp": position["entry_timestamp"],
        "entry_price": position["entry_price"],
        "proposed_size": position["proposed_size"],
        "actual_size": position["actual_size"],
        "cash_committed": committed,
        "stop": position["stop"],
        "target": position["target"],
        "maximum_holding_period": position["maximum_holding_period"],
        "exit_timestamp": exit_date,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_pnl": gross_pnl,
        "transaction_costs": costs,
        "slippage": "",
        "net_pnl": net_pnl,
        "gross_return": gross_pnl / committed if committed else 0.0,
        "net_return": net_pnl / committed if committed else 0.0,
        "maximum_adverse_excursion": position["maximum_adverse_excursion"],
        "maximum_favourable_excursion": position["maximum_favourable_excursion"],
        "holding_period": position["bars_held"],
        "model_versions": position["model_version"],
        "price_feature_timestamp": position["price_feature_timestamp"],
        "news_feature_timestamp": position["news_feature_timestamp"],
    }


def _daily_risk_metrics(
    curve: list[Mapping[str, Any]],
    ledger: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if not curve:
        return {}
    returns = [float(row["daily_return"]) for row in curve]
    equity = [float(row["total_equity"]) for row in curve]
    start = equity[0] / (1.0 + returns[0]) if returns else equity[0]
    end = equity[-1]
    drawdowns = _drawdowns(equity)
    wins = [float(row["net_pnl"]) for row in ledger if float(row["net_pnl"]) > 0]
    losses = [float(row["net_pnl"]) for row in ledger if float(row["net_pnl"]) < 0]
    years = max(len(returns) / 252.0, 1.0 / 252.0)
    wealth = end / start if start else 0.0
    vol = pstdev(returns) * math.sqrt(252.0) if len(returns) > 1 else 0.0
    downside = [min(value, 0.0) for value in returns]
    downside_vol = pstdev(downside) * math.sqrt(252.0) if len(downside) > 1 else 0.0
    cagr = wealth ** (1.0 / years) - 1.0 if wealth > 0 else -1.0
    value_at_risk = sorted(returns)[max(0, math.ceil(len(returns) * 0.05) - 1)]
    total_costs = sum(float(row["transaction_costs"]) for row in ledger)
    average_exposure = mean(float(row["gross_exposure"]) for row in curve)
    turnover = sum(abs(float(row.get("daily_return", 0.0))) for row in curve)
    return {
        "starting_equity": start,
        "ending_equity": end,
        "total_return_decimal": wealth - 1.0,
        "total_return_percent": (wealth - 1.0) * 100.0,
        "wealth_multiple": wealth,
        "CAGR": cagr,
        "annualised_volatility": vol,
        "maximum_drawdown": min(drawdowns),
        "average_drawdown": mean(drawdowns),
        "longest_drawdown_duration": _longest_drawdown_duration(drawdowns),
        "Sharpe_ratio": (mean(returns) * 252.0) / vol if vol else 0.0,
        "Sortino_ratio": (mean(returns) * 252.0) / downside_vol if downside_vol else 0.0,
        "Calmar_ratio": cagr / abs(min(drawdowns)) if min(drawdowns) else 0.0,
        "Value_at_Risk_5pct": value_at_risk,
        "VaR_5pct": value_at_risk,
        "expected_shortfall_CVaR_5pct": _expected_shortfall(returns),
        "CVaR_5pct": _expected_shortfall(returns),
        "worst_day": min(returns),
        "worst_week": _worst_rolling_return(returns, 5),
        "worst_trade": min((float(row["net_return"]) for row in ledger), default=0.0),
        "maximum_adverse_excursion": min((float(row["maximum_adverse_excursion"]) for row in ledger), default=0.0),
        "maximum_favourable_excursion": max((float(row["maximum_favourable_excursion"]) for row in ledger), default=0.0),
        "profit_factor": sum(wins) / abs(sum(losses)) if losses else 0.0,
        "hit_rate": len(wins) / max(len(ledger), 1),
        "average_win": mean(wins) if wins else 0.0,
        "average_loss": mean(losses) if losses else 0.0,
        "turnover": turnover,
        "average_exposure": average_exposure,
        "exposure": average_exposure,
        "average_concurrent_positions": mean(float(row["concurrent_positions"]) for row in curve),
        "total_costs": total_costs,
        "slippage": 0.0,
        "number_of_trades": len(ledger),
    }


def _action_attribution(events: list[Mapping[str, Any]], ledger: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_action: dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        by_action.setdefault(str(event["news_action"]), []).append(event)
    report = {}
    for action, rows in by_action.items():
        forward = [_number(row.get("candidate_forward_return")) or 0.0 for row in rows]
        blocked = [row for row in rows if row.get("blocked")]
        report[action] = {
            "candidate_count": len(rows),
            "average_candidate_forward_return": mean(forward) if forward else 0.0,
            "profitable_trades_blocked": sum((_number(row.get("candidate_forward_return")) or 0.0) > 0 for row in blocked),
            "losing_trades_blocked": sum((_number(row.get("candidate_forward_return")) or 0.0) < 0 for row in blocked),
            "pnl_saved": abs(sum((_number(row.get("candidate_forward_return")) or 0.0) for row in blocked if (_number(row.get("candidate_forward_return")) or 0.0) < 0)),
            "pnl_missed": sum((_number(row.get("candidate_forward_return")) or 0.0) for row in blocked if (_number(row.get("candidate_forward_return")) or 0.0) > 0),
        }
    report["executed_trade_count"] = len(ledger)
    return report


def _load_daily_price_bar_file(symbol: str, processed_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    path = processed_root / str(symbol).upper() / "1Day" / "bars.parquet"
    if not path.exists():
        return {"symbol": str(symbol).upper(), "status": "MISSING", "rows": [], "elapsed_seconds": time.perf_counter() - started}
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"])
        payload = table.to_pydict()
        rows = []
        for idx, timestamp in enumerate(payload["timestamp"]):
            high = float(payload["high"][idx])
            low = float(payload["low"][idx])
            if high < low:
                return {
                    "symbol": str(symbol).upper(),
                    "status": "FAILED",
                    "rows": [],
                    "error": f"malformed OHLC row high < low in {path}",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            rows.append(
                {
                    "date": _date_key(timestamp),
                    "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
                    "open": float(payload["open"][idx]),
                    "high": high,
                    "low": low,
                    "close": float(payload["close"][idx]),
                    "volume": float(payload["volume"][idx] or 0.0),
                    "symbol": str(payload["symbol"][idx]).upper(),
                }
            )
    except ImportError as exc:
        return {"symbol": str(symbol).upper(), "status": "FAILED", "rows": [], "error": "pyarrow unavailable to read daily bars", "elapsed_seconds": time.perf_counter() - started}
    except Exception as exc:
        return {"symbol": str(symbol).upper(), "status": "FAILED", "rows": [], "error": str(exc), "elapsed_seconds": time.perf_counter() - started}
    return {"symbol": str(symbol).upper(), "status": "OK", "rows": sorted(rows, key=lambda row: row["date"]), "elapsed_seconds": time.perf_counter() - started}


def _bar_sets_equal(
    left: Mapping[str, list[Mapping[str, Any]]],
    right: Mapping[str, list[Mapping[str, Any]]],
) -> bool:
    return _bar_set_digest(left) == _bar_set_digest(right)


def _bar_set_digest(payload: Mapping[str, list[Mapping[str, Any]]]) -> list[tuple[Any, ...]]:
    digest = []
    for symbol in sorted(payload):
        for row in sorted(payload[symbol], key=lambda item: str(item.get("date", ""))):
            digest.append(
                (
                    symbol,
                    row.get("date"),
                    row.get("timestamp"),
                    float(row.get("open", 0.0)),
                    float(row.get("high", 0.0)),
                    float(row.get("low", 0.0)),
                    float(row.get("close", 0.0)),
                    float(row.get("volume", 0.0)),
                )
            )
    return digest


def _replay_assumptions(
    replay_config: Mapping[str, Any],
    price_score_column: str,
    processed_root: Path,
) -> dict[str, Any]:
    return {
        "research_only": True,
        "broker_invoked": False,
        "orders_submitted": False,
        "price_data_root": str(processed_root),
        "price_timeframe": "1Day",
        "candidate_ranking": f"descending {price_score_column}, tie-break by symbol",
        "direction": "long_only",
        "entry_timing": "next available daily bar after decision date",
        "entry_price_convention": "next-session open plus entry_slippage_bps",
        "exit_rules": "stop if configured, then target if configured, otherwise max_holding_bars close",
        "same_bar_stop_target_ordering": "conservative_stop_first",
        "position_sizing": "min(available cash, equity * max_position_weight * news multiplier)",
        "maximum_positions": replay_config["max_positions"],
        "cash_allocation": "cash is debited at entry and unused cash remains in portfolio",
        "stop_loss_pct": replay_config["stop_loss_pct"],
        "profit_target_pct": replay_config["profit_target_pct"],
        "maximum_holding_bars": replay_config["max_holding_bars"],
        "commission_bps": replay_config["commission_bps"],
        "entry_slippage_bps": replay_config["entry_slippage_bps"],
        "exit_slippage_bps": replay_config["exit_slippage_bps"],
        "remaining_approximations": [
            "No intraday ordering is available for daily bars.",
            "Stop/target defaults are unset because the selected stock-alpha artifact has no explicit stop/target columns.",
            "Delisting handling exits at end_of_data when no later bars are available.",
        ],
    }


def _date_key(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value)
    return text[:10]


def _bar_on(
    symbol: str,
    date_key: str,
    bars_by_symbol: Mapping[str, list[Mapping[str, Any]]],
) -> Mapping[str, Any] | None:
    for bar in bars_by_symbol.get(str(symbol).upper(), []):
        if bar["date"] == date_key:
            return bar
    return None


def _bar_lookup(
    bars_by_symbol: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    return {
        symbol: {str(row["date"]): row for row in rows}
        for symbol, rows in bars_by_symbol.items()
    }


def _next_bar_lookup(
    bars_by_symbol: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for symbol, rows in bars_by_symbol.items():
        dates = [str(row["date"]) for row in rows]
        mapping = {}
        for index, value in enumerate(dates[:-1]):
            mapping[value] = dates[index + 1]
        for index, value in enumerate(dates):
            mapping.setdefault(value, dates[index + 1] if index + 1 < len(dates) else "")
        output[symbol] = mapping
    return output


def _bar_on_fast(
    symbol: str,
    date_key: str,
    lookup: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Mapping[str, Any] | None:
    return lookup.get(str(symbol).upper(), {}).get(date_key)


def _next_bar_date_fast(
    symbol: str,
    decision_date: str,
    lookup: Mapping[str, Mapping[str, str]],
) -> str | None:
    symbol_lookup = lookup.get(str(symbol).upper(), {})
    direct = symbol_lookup.get(decision_date)
    if direct:
        return direct
    later = [value for value in symbol_lookup if value > decision_date]
    if not later:
        return None
    return min(later)


def _next_bar_date(
    symbol: str,
    decision_date: str,
    bars_by_symbol: Mapping[str, list[Mapping[str, Any]]],
) -> str | None:
    for bar in bars_by_symbol.get(str(symbol).upper(), []):
        if bar["date"] > decision_date:
            return str(bar["date"])
    return None


def _last_bar(
    symbol: str,
    bars_by_symbol: Mapping[str, list[Mapping[str, Any]]],
) -> Mapping[str, Any] | None:
    rows = bars_by_symbol.get(str(symbol).upper(), [])
    return rows[-1] if rows else None


def _has_symbol(
    candidate: Mapping[str, Any],
    open_positions: list[Mapping[str, Any]],
    pending_entries: list[Mapping[str, Any]],
) -> bool:
    symbol = str(candidate.get("symbol", "")).upper()
    return any(row["symbol"] == symbol for row in open_positions) or any(
        row["symbol"] == symbol for row in pending_entries
    )


def _equity(
    cash: float,
    open_positions: list[Mapping[str, Any]],
    bars_by_symbol: Mapping[str, list[Mapping[str, Any]]],
    current_date: str,
) -> float:
    value = cash
    for position in open_positions:
        bar = _bar_on(position["symbol"], current_date, bars_by_symbol)
        mark = float(bar["close"]) if bar else float(position["entry_price"])
        value += float(position["shares"]) * mark
    return value


def _equity_fast(
    cash: float,
    open_positions: list[Mapping[str, Any]],
    lookup: Mapping[str, Mapping[str, Mapping[str, Any]]],
    current_date: str,
) -> float:
    value = cash
    for position in open_positions:
        bar = _bar_on_fast(position["symbol"], current_date, lookup)
        mark = float(bar["close"]) if bar else float(position["entry_price"])
        value += float(position["shares"]) * mark
    return value


def _action_event(
    candidate: Mapping[str, Any],
    variant: str,
    action: str,
    blocked: bool,
    ranking_after_news: int | None = None,
) -> dict[str, Any]:
    return {
        "strategy_variant": variant,
        "candidate_id": candidate.get("candidate_id", ""),
        "decision_timestamp": candidate.get("decision_timestamp", candidate.get("rebalance_date", "")),
        "symbol": candidate.get("symbol", ""),
        "news_action": action,
        "blocked": blocked,
        "candidate_forward_return": candidate.get("actual_forward_return_10d", candidate.get("actual_forward_return_5d", "")),
        "news_coverage": candidate.get("news_coverage_status", "NO_COVERAGE"),
        "news_risk_probability": candidate.get("price_plus_news_risk_probability", ""),
        "price_model_score": candidate.get("score", ""),
        "maximum_adverse_excursion": _adverse_excursion(candidate),
        "maximum_favourable_excursion": _favourable_excursion(candidate),
        "ranking_after_news": ranking_after_news if ranking_after_news is not None else "",
    }


def _drawdowns(equity: list[float]) -> list[float]:
    peak = equity[0] if equity else 0.0
    values = []
    for value in equity:
        peak = max(peak, value)
        values.append(value / peak - 1.0 if peak else 0.0)
    return values


def _longest_drawdown_duration(drawdowns: list[float]) -> int:
    longest = 0
    current = 0
    for value in drawdowns:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _worst_rolling_return(returns: list[float], window: int) -> float:
    if not returns:
        return 0.0
    if len(returns) < window:
        return math.prod(1.0 + value for value in returns) - 1.0
    return min(
        math.prod(1.0 + value for value in returns[index : index + window]) - 1.0
        for index in range(0, len(returns) - window + 1)
    )


def _number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _first_numeric(row: Mapping[str, Any], columns: Iterable[str]) -> float | None:
    for column in columns:
        value = _number(row.get(column))
        if value is not None:
            return value
    return None


def _favourable_excursion(row: Mapping[str, Any]) -> float | None:
    for column in (
        "actual_max_favourable_excursion",
        "forward_max_favourable_excursion",
        "max_favourable_excursion",
        "actual_max_favorable_excursion",
        "forward_max_favorable_excursion",
        "max_favorable_excursion",
    ):
        value = _number(row.get(column))
        if value is not None:
            return value
    forward = _first_numeric(row, RETURN_COLUMNS)
    return max(forward, 0.0) if forward is not None else None
