from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.data import CsvRowRepository, JsonRepository
from core.research.framework.ranking import finite_number
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_alpha_run_profile import apply_stock_alpha_run_profile
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_report_metadata

TARGET = "actual_forward_return_10d"
LONG_POLICIES = ("long_only_top_decile_equal_weight", "long_only_top_n_equal_weight", "long_only_score_weighted")
SHORT_POLICIES = ("long_short_top_bottom_decile_equal_weight", "long_short_score_weighted")
GUARDRAILS = {"research_only": True, "trading_impact": "none", "production_validated": False, "promotion_thresholds_changed": False}


@dataclass(frozen=True)
class StockLevelPortfolioReplayPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path
    equity_curves_path: Path
    holdings_path: Path


def write_stock_level_portfolio_replay(config: dict[str, Any]) -> StockLevelPortfolioReplayPaths:
    settings = StockLevelResearchConfig.from_mapping(config)
    if not settings.portfolio_replay_enabled:
        raise ValueError("ml.stock_portfolio_replay_enabled is false")
    rows = CsvRowRepository().read(settings.oos_predictions_path)
    benchmark = JsonRepository().read(settings.benchmark_path)
    rows, profile = apply_stock_alpha_run_profile(rows, settings)
    summary, curves, holdings, payload = build_stock_level_portfolio_replay(
        rows,
        benchmark=benchmark,
        signal_columns=settings.portfolio_signal_columns,
        top_n=settings.portfolio_top_n,
        cost_bps=settings.portfolio_cost_bps,
        slippage_bps=settings.portfolio_slippage_bps,
        max_position_weight=settings.portfolio_max_position_weight,
        min_position_weight=settings.portfolio_min_position_weight,
        allow_short=settings.portfolio_allow_short,
    )
    payload.update(profile)
    payload.update(stock_alpha_report_metadata(config, settings.output_dir, source_artifact_path=settings.oos_predictions_path))
    output = settings.output_dir
    paths = StockLevelPortfolioReplayPaths(
        output / "stock_level_portfolio_replay_summary.csv",
        output / "stock_level_portfolio_replay_summary.json",
        output / "stock_level_portfolio_replay_summary.md",
        output / "stock_level_portfolio_replay_equity_curves.csv",
        output / "stock_level_portfolio_replay_holdings.csv",
    )
    writer = ResearchArtifactWriter()
    writer.write_csv(paths.csv_path, summary, fieldnames=list(summary[0]) if summary else ["signal_column"])
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))
    writer.write_csv(paths.equity_curves_path, curves, fieldnames=list(curves[0]) if curves else ["rebalance_date"])
    writer.write_csv(paths.holdings_path, holdings, fieldnames=list(holdings[0]) if holdings else ["rebalance_date"])
    return paths


def build_stock_level_portfolio_replay(
    rows: list[dict[str, Any]], *, benchmark: dict[str, Any], signal_columns: tuple[str, ...] | list[str],
    top_n: int = 25, cost_bps: float = 10.0, slippage_bps: float = 5.0,
    max_position_weight: float = 0.05, min_position_weight: float = 0.0, allow_short: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if benchmark.get("walk_forward", {}).get("out_of_sample_only") is not True:
        raise ValueError("Benchmark metadata must confirm out_of_sample_only")
    eligible = [row for row in rows if str(row.get("fold_id", "")).strip() and finite_number(row.get(TARGET)) is not None]
    policies = (*LONG_POLICIES, *(SHORT_POLICIES if allow_short else ()))
    summaries: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    holdings: list[dict[str, Any]] = []
    for signal in signal_columns:
        if not any(finite_number(row.get(signal)) is not None for row in eligible):
            continue
        for policy in policies:
            periods, strategy_holdings = _replay(eligible, signal, policy, top_n, cost_bps, slippage_bps, max_position_weight, min_position_weight)
            summaries.append(_metrics(signal, policy, periods, strategy_holdings))
            curves.extend(periods)
            holdings.extend(strategy_holdings)
    winners = _winners(summaries)
    payload = {
        "mode": "stock_level_portfolio_replay_research_only", "target_column": TARGET,
        "oos_only": True, "training_performed": False, "signal_columns": list(signal_columns),
        "policies": list(policies), "summary": summaries, "winners": winners,
        "best_ml_vs_momentum_120d": _ml_vs_momentum(summaries), **GUARDRAILS,
    }
    return summaries, curves, holdings, payload


def _replay(rows: list[dict[str, Any]], signal: str, policy: str, top_n: int, cost_bps: float, slippage_bps: float, cap: float, floor: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_date: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if finite_number(row.get(signal)) is not None:
            by_date.setdefault(str(row["rebalance_date"]), []).append(row)
    previous: dict[str, float] = {}
    equity = 1.0
    periods, holdings = [], []
    for rebalance_date, group in sorted(by_date.items()):
        weights = _weights(group, signal, policy, top_n, cap, floor)
        turnover = sum(abs(weights.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in set(weights) | set(previous))
        gross = sum(weight * float(next(row[TARGET] for row in group if str(row["symbol"]).upper() == symbol)) for symbol, weight in weights.items())
        drag = turnover * (cost_bps + slippage_bps) / 10_000.0
        net = gross - drag
        equity *= 1.0 + net
        key = f"{signal}|{policy}"
        periods.append({"rebalance_date": rebalance_date, "strategy_id": key, "signal_column": signal, "policy": policy, "gross_return": gross, "transaction_cost_drag": drag, "net_return": net, "turnover": turnover, "equity": equity})
        for symbol, weight in sorted(weights.items()):
            holdings.append({"rebalance_date": rebalance_date, "strategy_id": key, "signal_column": signal, "policy": policy, "symbol": symbol, "weight": weight, "side": "long" if weight > 0 else "short"})
        previous = weights
    return periods, holdings


def _weights(rows: list[dict[str, Any]], signal: str, policy: str, top_n: int, cap: float, floor: float) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: (-float(row[signal]), str(row["symbol"]).upper()))
    bucket = max(1, math.ceil(len(ordered) * 0.1))
    if policy == "long_only_top_n_equal_weight":
        return _equal(ordered[:top_n], 1.0, cap, floor)
    if policy == "long_only_top_decile_equal_weight":
        return _equal(ordered[:bucket], 1.0, cap, floor)
    if policy == "long_only_score_weighted":
        return _score(ordered[:top_n], signal, 1.0, cap, floor, positive=True)
    if policy == "long_short_top_bottom_decile_equal_weight":
        return {**_equal(ordered[:bucket], 0.5, cap, floor), **_equal(ordered[-bucket:], -0.5, cap, floor)}
    weights = _score(ordered[:top_n], signal, 0.5, cap, floor, positive=True)
    weights.update(_score(ordered[-top_n:], signal, -0.5, cap, floor, positive=False))
    return weights


def _equal(rows: list[dict[str, Any]], exposure: float, cap: float, floor: float) -> dict[str, float]:
    weight = min(cap, abs(exposure) / len(rows)) if rows else 0.0
    if weight < floor:
        return {}
    return {str(row["symbol"]).upper(): math.copysign(weight, exposure) for row in rows}


def _score(rows: list[dict[str, Any]], signal: str, exposure: float, cap: float, floor: float, *, positive: bool) -> dict[str, float]:
    raw = [max(float(row[signal]), 0.0) if positive else max(-float(row[signal]), 0.0) for row in rows]
    total = sum(raw)
    if total <= 0.0:
        return _equal(rows, exposure, cap, floor)
    return {str(row["symbol"]).upper(): math.copysign(min(cap, abs(exposure) * value / total), exposure) for row, value in zip(rows, raw) if min(cap, abs(exposure) * value / total) >= floor}


def _metrics(signal: str, policy: str, periods: list[dict[str, Any]], holdings: list[dict[str, Any]]) -> dict[str, Any]:
    values = [row["net_return"] for row in periods]
    gross = sum(row["gross_return"] for row in periods)
    drag = sum(row["transaction_cost_drag"] for row in periods)
    equity = [1.0, *[row["equity"] for row in periods]]
    peak, max_dd = equity[0], 0.0
    for value in equity:
        peak = max(peak, value); max_dd = min(max_dd, value / peak - 1.0)
    annualization = _annualization([row["rebalance_date"] for row in periods])
    total = equity[-1] - 1.0
    annualized = (equity[-1] ** (annualization / len(values)) - 1.0) if values and annualization else None
    vol = pstdev(values) if len(values) > 1 else 0.0
    by_date = {d: [h for h in holdings if h["rebalance_date"] == d] for d in {h["rebalance_date"] for h in holdings}}
    weights = [abs(float(row["weight"])) for row in holdings]
    kind = "baseline" if signal in {"predicted_momentum_120d", "predicted_risk_adjusted_momentum"} else "ml_model"
    return {"strategy_id": f"{signal}|{policy}", "signal_column": signal, "kind": kind, "policy": policy, "total_return": total, "annualized_return": annualized, "mean_period_return": mean(values) if values else None, "volatility": vol, "sharpe": (mean(values) / vol * math.sqrt(annualization) if vol > 0 and annualization else None), "max_drawdown": max_dd, "calmar_ratio": (annualized / abs(max_dd) if annualized is not None and max_dd < 0 else None), "hit_rate": mean([v > 0 for v in values]) if values else None, "average_turnover": mean([row["turnover"] for row in periods]) if periods else None, "average_number_of_positions": mean([len(v) for v in by_date.values()]) if by_date else 0.0, "average_position_weight": mean(weights) if weights else 0.0, "max_position_weight": max(weights, default=0.0), "transaction_cost_drag": drag, "gross_return": gross, "net_return": sum(values), "best_period_return": max(values, default=None), "worst_period_return": min(values, default=None), "date_count": len(periods), "symbol_count": len({row["symbol"] for row in holdings})}


def _annualization(dates: list[str]) -> float | None:
    parsed = [date.fromisoformat(value) for value in dates]
    gaps = [(right - left).days for left, right in zip(parsed, parsed[1:]) if right > left]
    return 365.25 / mean(gaps) if gaps else None


def _best(rows: list[dict[str, Any]], metric: str, *, lowest: bool = False, kind: str | None = None) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get(metric) is not None and (kind is None or row["kind"] == kind)]
    return (min if lowest else max)(candidates, key=lambda row: float(row[metric]), default=None)


def _winners(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"best_by_total_return": _best(rows, "total_return"), "best_by_sharpe": _best(rows, "sharpe"), "best_by_max_drawdown": _best(rows, "max_drawdown"), "best_by_calmar": _best(rows, "calmar_ratio"), "best_by_net_return_after_costs": _best(rows, "net_return"), "best_baseline": _best(rows, "net_return", kind="baseline"), "best_ml_model": _best(rows, "net_return", kind="ml_model")}


def _ml_vs_momentum(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ml = _best(rows, "net_return", kind="ml_model")
    momentum = _best([row for row in rows if row["signal_column"] == "predicted_momentum_120d"], "net_return")
    return {"ml_strategy_id": ml.get("strategy_id") if ml else None, "momentum_strategy_id": momentum.get("strategy_id") if momentum else None, "net_return_delta": (ml["net_return"] - momentum["net_return"] if ml and momentum else None), "beats_momentum_120d": bool(ml and momentum and ml["net_return"] > momentum["net_return"])}


def _markdown(payload: dict[str, Any]) -> str:
    lines = ["# Stock-Level Portfolio Replay", "", "Research only. Trading impact: none. Production validated: false.", "", f"- Run size: `{payload.get('run_size', 'benchmark')}`", f"- OOS only: {payload['oos_only']}", "- Promotion thresholds changed: false", "", "| Signal | Policy | Net return | Sharpe | Max drawdown | Turnover | Cost drag |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in payload["summary"]:
        lines.append(f"| {row['signal_column']} | {row['policy']} | {row['net_return']} | {row['sharpe']} | {row['max_drawdown']} | {row['average_turnover']} | {row['transaction_cost_drag']} |")
    return "\n".join(lines) + "\n"
