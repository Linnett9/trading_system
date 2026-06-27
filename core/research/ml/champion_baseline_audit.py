from __future__ import annotations

import csv
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from core.entities.candle import Candle
from core.research.dual_momentum_factory import build_dual_momentum_tester
from core.research.performance_metrics import calmar_ratio


RESEARCH_METADATA = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
}


@dataclass(frozen=True)
class ChampionBaselineAuditPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path


def write_champion_baseline_audit(config: dict[str, Any]) -> ChampionBaselineAuditPaths:
    output_dir = _meta_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    expanded_rows = _read_csv(_expanded_dataset_path(config))
    meta_rows = _read_csv(_meta_dataset_path(config))
    return_audit = _read_json(output_dir / "benchmark_return_audit.json")
    periods = _evaluation_periods(meta_rows, expanded_rows)
    diagnostic_rows = _diagnostic_baseline_rows(return_audit)
    exact_replay = _try_exact_champion_replay(config, periods)
    top_date_report = _top_date_report(
        return_audit,
        exact_replay,
        expanded_rows,
    )
    payload = {
        "mode": "champion_baseline_audit_research_only",
        "baseline_semantics": {
            "champion_return_next_period_created_in": (
                "core/research/ml/rebalance_dataset.py::build_champion_rebalance_rows"
            ),
            "champion_return_next_period_represents": (
                "dual-momentum backtester equity-curve return for each expanded "
                "variant row over the configured label horizon"
            ),
            "current_allocation_champion_baseline": (
                "full allocation exposure applied to the date-averaged expanded "
                "variant return series"
            ),
            "current_champion_baseline_is_exact_champion_replay": False,
            "why_champion_baseline_equals_always_full_exposure": (
                "both use constant allocation exposure of 1.0 in allocation_v2"
            ),
            "equality_is_misleading": True,
        },
        "baseline_rows": diagnostic_rows + [exact_replay["summary"]],
        "exact_champion_replay": exact_replay,
        "v2_vs_exact_champion": _v2_vs_exact(return_audit, exact_replay),
        "top_date_report": top_date_report,
        "stooq_adjustment_audit": _stooq_adjustment_audit(
            config,
            exact_replay,
        ),
        "red_flags": _red_flags(exact_replay, return_audit),
        **RESEARCH_METADATA,
    }
    paths = ChampionBaselineAuditPaths(
        csv_path=output_dir / "champion_baseline_audit.csv",
        json_path=output_dir / "champion_baseline_audit.json",
        markdown_path=output_dir / "champion_baseline_audit.md",
    )
    _write_csv(paths.csv_path, payload["baseline_rows"])
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths.markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return paths


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


def _evaluation_periods(
    meta_rows: list[dict[str, str]],
    expanded_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    holdout_dates = {
        row.get("rebalance_date")
        for row in meta_rows
        if row.get("split") == "holdout" and row.get("rebalance_date")
    }
    end_by_date = {}
    for row in expanded_rows:
        date = row.get("rebalance_date")
        if date in holdout_dates and row.get("outcome_end_date"):
            end_by_date[date] = row["outcome_end_date"]
    return [
        {"rebalance_date": date, "outcome_end_date": end_by_date[date]}
        for date in sorted(end_by_date)
    ]


def _diagnostic_baseline_rows(return_audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    by_name = {
        row.get("candidate_name"): row
        for row in return_audit.get("candidates", [])
        if isinstance(row, dict)
    }
    for source_name, baseline_name in (
        ("champion_baseline", "champion_full_exposure_diagnostic"),
        ("always_full_exposure", "always_full_exposure"),
    ):
        source = by_name.get(source_name, {})
        rows.append({
            "baseline_name": baseline_name,
            "source_candidate_name": source_name,
            "semantic_type": "diagnostic_full_allocation_exposure",
            "available": bool(source),
            "target_exposure": 1.0,
            "total_return": source.get("total_return"),
            "continuous_total_return": None,
            "max_drawdown": source.get("max_drawdown"),
            "turnover": source.get("turnover"),
            "costs": source.get("costs"),
            "cost_turnover_status": "allocation overlay exposure turnover only",
            "is_exact_champion_replay": False,
            **RESEARCH_METADATA,
        })
    return rows


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


def _top_date_report(
    return_audit: dict[str, Any],
    exact_replay: dict[str, Any],
    expanded_rows: list[dict[str, str]],
) -> dict[str, Any]:
    expanded_by_date = _expanded_rows_by_date(expanded_rows)
    output = {}
    for name in (
        "champion_baseline",
        "return_only_allocation",
        "selected_bayesian_optimizer_diagnostic_policy",
    ):
        candidate = _return_audit_candidate(return_audit, name)
        output[name] = {
            "top_20": _attach_expanded_symbols(
                candidate.get("top_20_contributing_rebalance_dates", []),
                expanded_by_date,
            ),
            "worst_20": _attach_expanded_symbols(
                candidate.get("worst_20_contributing_rebalance_dates", []),
                expanded_by_date,
            ),
        }
    output["exact_champion_replay"] = {
        "top_20": exact_replay.get("period_grid_summary", {}).get(
            "top_20_rebalance_dates",
            [],
        ),
        "worst_20": exact_replay.get("period_grid_summary", {}).get(
            "worst_20_rebalance_dates",
            [],
        ),
    }
    output["late_2025_early_2026_dominance"] = _late_period_dominance(return_audit)
    return output


def _attach_expanded_symbols(
    records: list[dict[str, Any]],
    expanded_by_date: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    output = []
    for record in records:
        date = str(record.get("date") or record.get("rebalance_date") or "")
        variants = sorted(
            expanded_by_date.get(date, []),
            key=lambda row: float(row.get("champion_return_next_period", 0.0) or 0.0),
            reverse=True,
        )
        output.append({
            **record,
            "expanded_variant_count": len(variants),
            "top_expanded_variants": [
                {
                    "variant_id": row.get("variant_id"),
                    "selected_symbols": row.get("selected_symbols"),
                    "exposure_target": _number(row.get("exposure_target")),
                    "champion_return_next_period": _number(
                        row.get("champion_return_next_period")
                    ),
                }
                for row in variants[:5]
            ],
        })
    return output


def _stooq_adjustment_audit(
    config: dict[str, Any],
    exact_replay: dict[str, Any],
) -> dict[str, Any]:
    parquet_dir = config.get("ml", {}).get(
        "stooq_parquet_dir",
        "data/processed/stooq_parquet",
    )
    anomaly_rows = [
        anomaly
        for row in exact_replay.get("period_rows", [])
        for anomaly in row.get("symbol_return_anomalies", [])
    ]
    return {
        "data_path": parquet_dir,
        "data_source": "local Stooq parquet research data",
        "price_column_used": "close",
        "adjusted_status": (
            "unknown_from_repo_metadata; code reads Stooq close/Close columns and "
            "does not persist split/dividend adjustment metadata"
        ),
        "top_symbol_anomaly_count": len(anomaly_rows),
        "top_symbol_anomalies": anomaly_rows[:50],
    }


def _v2_vs_exact(
    return_audit: dict[str, Any],
    exact_replay: dict[str, Any],
) -> dict[str, Any]:
    exact_total = exact_replay.get("period_grid_summary", {}).get("total_return")
    continuous_total = exact_replay.get("continuous_equity_summary", {}).get(
        "total_return"
    )
    comparisons = {}
    for name in (
        "return_only_allocation",
        "selected_bayesian_optimizer_diagnostic_policy",
        "meta_ensemble_allocation",
        "binary_exposure_overlay",
    ):
        candidate = _return_audit_candidate(return_audit, name)
        candidate_return = candidate.get("reported_total_return") or candidate.get(
            "total_return"
        )
        comparisons[name] = {
            "candidate_return": candidate_return,
            "beats_exact_period_grid": (
                candidate_return is not None
                and exact_total is not None
                and float(candidate_return) > float(exact_total)
            ),
            "return_delta_vs_exact_period_grid": (
                float(candidate_return) - float(exact_total)
                if candidate_return is not None and exact_total is not None
                else None
            ),
            "beats_exact_continuous_equity": (
                candidate_return is not None
                and continuous_total is not None
                and float(candidate_return) > float(continuous_total)
            ),
        }
    return comparisons


def _red_flags(
    exact_replay: dict[str, Any],
    return_audit: dict[str, Any],
) -> list[str]:
    flags = [
        "current_champion_baseline_is_diagnostic_not_exact_replay",
        "current_allocation_baseline_compounds_overlapping_forward_periods",
    ]
    if not exact_replay.get("available"):
        flags.append("exact_champion_replay_unavailable")
    if _return_audit_candidate(return_audit, "champion_baseline").get("total_return"):
        flags.append("old_champion_baseline_name_is_misleading")
    if exact_replay.get("stooq_adjusted_status") == "unknown":
        flags.append("stooq_adjustment_status_unknown")
    return sorted(set(flags))


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


def _late_period_dominance(return_audit: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for name in ("champion_baseline", "return_only_allocation"):
        candidate = _return_audit_candidate(return_audit, name)
        top = candidate.get("top_20_contributing_rebalance_dates", [])
        if not top:
            output[name] = None
            continue
        late_count = sum(
            "2025-08-01" <= str(row.get("date", "")) <= "2026-02-28"
            for row in top
        )
        output[name] = {
            "top_20_dates_in_2025_08_to_2026_02": late_count,
            "share": late_count / len(top),
        }
    return output


def _expanded_rows_by_date(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        output.setdefault(row.get("rebalance_date", ""), []).append(row)
    return output


def _return_audit_candidate(
    return_audit: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    return next(
        (
            row for row in return_audit.get("candidates", [])
            if row.get("candidate_name") == name
        ),
        {},
    )


def _compound_returns(returns: list[float]) -> float:
    equity = 1.0
    for value in returns:
        equity *= 1.0 + value
    return equity - 1.0


def _equity_curve(returns: list[float]) -> list[float]:
    equity = 1.0
    curve = []
    for value in returns:
        equity *= 1.0 + value
        curve.append(equity)
    return curve


def _max_drawdown(values: list[float]) -> float:
    peak = values[0] if values else 1.0
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = max(drawdown, (peak - value) / peak if peak else 0.0)
    return drawdown


def _annualized_return(total_return: float, rows: list[dict[str, Any]]) -> float | None:
    if len(rows) < 2 or total_return <= -1.0:
        return None
    try:
        start = datetime.fromisoformat(rows[0]["rebalance_date"][:10])
        end = datetime.fromisoformat(rows[-1]["outcome_end_date"][:10])
    except ValueError:
        return None
    elapsed_days = (end - start).days
    if elapsed_days <= 0:
        return None
    return (1.0 + total_return) ** (365.25 / elapsed_days) - 1.0


def _observed_periods_per_year(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return 1.0
    try:
        start = datetime.fromisoformat(rows[0]["rebalance_date"][:10])
        end = datetime.fromisoformat(rows[-1]["outcome_end_date"][:10])
    except ValueError:
        return 1.0
    elapsed_days = (end - start).days
    if elapsed_days <= 0:
        return 1.0
    return max(1.0, len(rows) * 365.25 / elapsed_days)


def _sharpe(returns: list[float], rows: list[dict[str, Any]]) -> float:
    if not returns:
        return 0.0
    average = mean(returns)
    std = math.sqrt(mean((value - average) ** 2 for value in returns))
    if std == 0.0:
        return 0.0
    return average / std * math.sqrt(_observed_periods_per_year(rows))


def _sortino(returns: list[float], rows: list[dict[str, Any]]) -> float:
    if not returns:
        return 0.0
    downside = [min(0.0, value) for value in returns]
    downside_deviation = math.sqrt(sum(value * value for value in downside) / len(returns))
    if downside_deviation == 0.0:
        return 0.0
    return mean(returns) / downside_deviation * math.sqrt(
        _observed_periods_per_year(rows)
    )


def _meta_output_dir(config: dict[str, Any]) -> Path:
    ml_config = config.get("ml", {})
    return Path(
        ml_config.get(
            "output_dir",
            Path(config.get("reports", {}).get("ml_dir", "reports/ml"))
            / "regime_transformer_meta_ensemble_v1",
        )
    )


def _expanded_dataset_path(config: dict[str, Any]) -> Path:
    ml_config = config.get("ml", {})
    return Path(
        ml_config.get(
            "expanded_rebalance_dataset_path",
            Path(config.get("cache", {}).get("ml_dir", "cache/ml"))
            / "expanded_rebalance_dataset.csv",
        )
    )


def _meta_dataset_path(config: dict[str, Any]) -> Path:
    ml_config = config.get("ml", {})
    return Path(
        ml_config.get(
            "meta_dataset_path",
            Path(config.get("cache", {}).get("ml_dir", "cache/ml"))
            / "meta_ensemble_dataset.csv",
        )
    )


def _champion_config_path(config: dict[str, Any]) -> Path:
    return Path(
        str(
            config.get("research", {})
            .get("dual_momentum", {})
            .get(
                "champion_config_path",
                "configs/champions/ranked_top5_monthly_exposure90_v1.yaml",
            )
        )
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "baseline_name",
        "source_candidate_name",
        "semantic_type",
        "available",
        "target_exposure",
        "total_return",
        "continuous_total_return",
        "max_drawdown",
        "turnover",
        "costs",
        "cost_turnover_status",
        "is_exact_champion_replay",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Champion Baseline Audit",
        "",
        "Research only. Trading impact: none. Production validated: false.",
        "",
        "## Semantics",
        "",
        f"- Current champion baseline exact replay: {payload['baseline_semantics']['current_champion_baseline_is_exact_champion_replay']}",
        f"- Why equal to always full: {payload['baseline_semantics']['why_champion_baseline_equals_always_full_exposure']}",
        "",
        "## Baselines",
        "",
        "|baseline|semantic type|available|target exposure|total return|continuous return|drawdown|turnover|costs|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["baseline_rows"]:
        lines.append(
            "|{name}|{kind}|{available}|{target}|{total}|{continuous}|"
            "{drawdown}|{turnover}|{costs}|".format(
                name=row.get("baseline_name"),
                kind=row.get("semantic_type"),
                available=row.get("available"),
                target=_fmt(row.get("target_exposure")),
                total=_fmt(row.get("total_return")),
                continuous=_fmt(row.get("continuous_total_return")),
                drawdown=_fmt(row.get("max_drawdown")),
                turnover=_fmt(row.get("turnover")),
                costs=_fmt(row.get("costs")),
            )
        )
    exact = payload["exact_champion_replay"]
    lines.extend([
        "",
        "## Exact Replay",
        "",
        f"- Available: {exact.get('available')}",
        f"- Reason: {exact.get('availability_reason') or 'ok'}",
        f"- Period-grid return: {_fmt(exact.get('period_grid_summary', {}).get('total_return'))}",
        f"- Continuous equity return: {_fmt(exact.get('continuous_equity_summary', {}).get('total_return'))}",
        "",
        "## Red Flags",
        "",
    ])
    lines.extend(f"- {flag}" for flag in payload.get("red_flags", []))
    lines.extend([
        "",
        "Research only. Trading impact: none. Production validated: false.",
        "",
    ])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.6f}"
