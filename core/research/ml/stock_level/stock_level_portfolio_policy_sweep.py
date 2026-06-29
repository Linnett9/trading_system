from __future__ import annotations

import itertools
import math
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.data import CsvRowRepository, JsonRepository
from core.research.framework.ranking import finite_number
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_alpha_run_profile import apply_stock_alpha_run_profile
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_report_metadata
from core.research.ml.runtime_parallelism import apply_stock_alpha_worker_caps
from core.research.ml.stock_level.stock_level_portfolio_replay import TARGET, _metrics

POLICIES = (
    "long_only_top_decile_equal_weight", "long_only_top_n_equal_weight",
    "long_only_score_weighted", "long_only_volatility_scaled",
    "long_only_score_x_inverse_volatility", "long_short_top_bottom_decile_equal_weight",
    "long_short_score_weighted",
)
SIZING = ("equal_weight", "score_weighted", "rank_weighted", "softmax_score_weighted", "inverse_volatility_weighted", "score_times_inverse_volatility")
GUARDRAILS = {"research_only": True, "trading_impact": "none", "production_validated": False, "promotion_thresholds_changed": False}


@dataclass(frozen=True)
class StockLevelPortfolioPolicySweepPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path
    equity_curves_path: Path
    top_holdings_path: Path


def build_policy_grid(config: dict[str, Any], available_signals: list[str]) -> list[dict[str, Any]]:
    ml = config.get("ml", {})
    run_size = str(ml.get("stock_alpha_run_size", "benchmark"))
    requested = list(ml.get("stock_portfolio_policy_sweep_signals", available_signals))
    configured_baselines = list(ml.get("stock_portfolio_policy_sweep_baseline_signals", ["predicted_momentum_120d", "predicted_risk_adjusted_momentum"]))
    requested.extend(name for name in configured_baselines if name not in requested)
    signals = [name for name in requested if name in available_signals]
    policies = [p for p in POLICIES if ml.get("stock_portfolio_policy_sweep_allow_short", False) or not p.startswith("long_short")]
    values = itertools.product(
        policies, SIZING,
        ml.get("stock_portfolio_policy_sweep_top_n_values", [10, 25, 50]),
        ml.get("stock_portfolio_policy_sweep_max_position_weights", [0.03, 0.05, 0.10]),
        ml.get("stock_portfolio_policy_sweep_cost_bps_values", [5, 10, 25]),
        ml.get("stock_portfolio_policy_sweep_slippage_bps_values", [5]),
        ml.get("stock_portfolio_policy_sweep_turnover_caps", [None, 0.5, 1.0]),
        ml.get("stock_portfolio_policy_sweep_volatility_targets", [None, 0.10, 0.15]),
        signals,
    )
    grid = []
    for index, value in enumerate(values):
        policy, sizing, top_n, cap, cost, slippage, turnover_cap, vol_target, signal = value
        grid.append({"config_id": f"policy_{index:06d}", "signal_column": signal, "policy": policy, "sizing_method": sizing, "top_n": int(top_n), "max_position_weight_limit": float(cap), "cost_bps": float(cost), "slippage_bps": float(slippage), "turnover_cap": turnover_cap, "volatility_target": vol_target})
    limit = int(ml.get(f"stock_portfolio_policy_sweep_max_configs_{run_size}", {"dev": 40, "benchmark": 250, "full": 1000}[run_size]))
    return grid[:limit]


def write_stock_level_portfolio_policy_sweep(config: dict[str, Any]) -> StockLevelPortfolioPolicySweepPaths:
    settings = StockLevelResearchConfig.from_mapping(config)
    thread_caps = apply_stock_alpha_worker_caps(config)
    stage_started = time.perf_counter(); stage_started_at = datetime.now(timezone.utc).isoformat()
    if not bool(config.get("ml", {}).get("stock_portfolio_policy_sweep_enabled", True)):
        raise ValueError("ml.stock_portfolio_policy_sweep_enabled is false")
    rows, profile = apply_stock_alpha_run_profile(CsvRowRepository().read(settings.oos_predictions_path), settings)
    benchmark = JsonRepository().read(settings.benchmark_path)
    if benchmark.get("walk_forward", {}).get("out_of_sample_only") is not True:
        raise ValueError("Benchmark metadata must confirm out_of_sample_only")
    available = sorted(name for name in (rows[0] if rows else {}) if name.startswith("stock_level_predicted_"))
    available.extend(name for name in ("predicted_momentum_120d", "predicted_risk_adjusted_momentum") if rows and name in rows[0])
    grid = build_policy_grid(config, available)
    ml = config.get("ml", {})
    workers = min(max(1, int(ml.get("stock_portfolio_policy_sweep_n_jobs", 1))), len(grid) or 1)
    kwargs = {"min_positions": int(ml.get("stock_portfolio_policy_sweep_min_positions", 1)), "max_positions": int(ml.get("stock_portfolio_policy_sweep_max_positions", 1000)), "cash_buffer": float(ml.get("stock_portfolio_policy_sweep_cash_buffer", 0.0)), "allow_partial_cash": bool(ml.get("stock_portfolio_policy_sweep_allow_partial_cash_when_constraints_bind", True)), "borrow_cost_bps": float(ml.get("stock_portfolio_policy_sweep_borrow_cost_bps", 0.0))}
    def action(item: dict[str, Any]):
        started = time.perf_counter()
        result = _evaluate_config(rows, dict(item), **kwargs)
        result[0]["elapsed_seconds"] = time.perf_counter() - started
        print(f"[stock-alpha] completed policy={item['config_id']} elapsed={result[0]['elapsed_seconds']:.3f}s")
        return result
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(action, grid))
    else:
        results = [action(item) for item in grid]
    summaries = [result[0] for result in results]
    curves = [row for result in results for row in result[1]]
    holdings = [row for result in results for row in result[2]]
    baseline_names = list(ml.get("stock_portfolio_policy_sweep_baseline_signals", ["predicted_momentum_120d", "predicted_risk_adjusted_momentum"]))
    found_baselines = [name for name in baseline_names if name in available]
    coverage = {"baseline_signal_available": bool(found_baselines), "baseline_signal_columns_found": found_baselines, "baseline_signal_columns_missing": [name for name in baseline_names if name not in available], "baseline_missing_reason": None if found_baselines else "No configured baseline signal columns are present in OOS predictions"}
    winners = _winners(summaries)
    warnings = _negative_return_warnings(summaries)
    payload = {"mode": "stock_level_portfolio_policy_sweep_research_only", "started_at": stage_started_at, "completed_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - stage_started, **profile, **stock_alpha_report_metadata(config, settings.output_dir, source_artifact_path=settings.oos_predictions_path), "policy_config_count": len(grid), "policy_timings": {row["config_id"]: row["elapsed_seconds"] for row in summaries}, "parallelism": {"requested_workers": int(ml.get("stock_portfolio_policy_sweep_n_jobs", 1)), "effective_workers": workers, "nested_workers": 1, "nested_sklearn_n_jobs": 1, "nested_torch_num_threads": 1}, "thread_caps": thread_caps, "baseline_coverage": coverage, **warnings, "summary": summaries, "winners": winners, **GUARDRAILS}
    output = settings.output_dir
    paths = StockLevelPortfolioPolicySweepPaths(output / "stock_level_portfolio_policy_sweep.csv", output / "stock_level_portfolio_policy_sweep.json", output / "stock_level_portfolio_policy_sweep.md", output / "stock_level_portfolio_policy_sweep_equity_curves.csv", output / "stock_level_portfolio_policy_sweep_top_holdings.csv")
    writer = ResearchArtifactWriter()
    writer.write_csv(paths.csv_path, summaries, fieldnames=list(summaries[0]) if summaries else ["config_id"])
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))
    writer.write_csv(paths.equity_curves_path, curves, fieldnames=list(curves[0]) if curves else ["rebalance_date"])
    writer.write_csv(paths.top_holdings_path, holdings, fieldnames=list(holdings[0]) if holdings else ["rebalance_date"])
    return paths


def build_sizing_weights(rows: list[dict[str, Any]], signal: str, method: str, exposure: float, cap: float) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: (-float(row[signal]), str(row["symbol"])))
    count = len(ordered)
    if not count:
        return {}
    scores = [float(row[signal]) for row in ordered]
    # Sizing must remain point-in-time: future/actual volatility labels are forbidden.
    vols = [max(finite_number(row.get("predicted_volatility_20d")) or finite_number(row.get("predicted_future_volatility")) or 1.0, 1e-6) for row in ordered]
    if method == "equal_weight": raw = [1.0] * count
    elif method == "rank_weighted": raw = list(range(count, 0, -1))
    elif method == "softmax_score_weighted":
        maximum = max(scores); raw = [math.exp(min(50.0, score - maximum)) for score in scores]
    elif method == "inverse_volatility_weighted": raw = [1.0 / vol for vol in vols]
    elif method == "score_times_inverse_volatility": raw = [max(score, 0.0) / vol for score, vol in zip(scores, vols)]
    else: raw = [max(score, 0.0) for score in scores]
    if sum(raw) <= 0: raw = [1.0] * count
    return {str(row["symbol"]): math.copysign(min(cap, abs(exposure) * value / sum(raw)), exposure) for row, value in zip(ordered, raw)}


def _evaluate_config(rows: list[dict[str, Any]], spec: dict[str, Any], *, min_positions: int, max_positions: int, cash_buffer: float, allow_partial_cash: bool, borrow_cost_bps: float) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    signal = spec["signal_column"]
    eligible = [row for row in rows if str(row.get("fold_id", "")) and finite_number(row.get(signal)) is not None and finite_number(row.get(TARGET)) is not None]
    by_date: dict[str, list[dict[str, Any]]] = {}
    for row in eligible: by_date.setdefault(str(row["rebalance_date"]), []).append(row)
    previous: dict[str, float] = {}; equity = 1.0; periods = []; holdings = []; infeasible = None
    for rebalance_date, group in sorted(by_date.items()):
        ordered = sorted(group, key=lambda row: (-float(row[signal]), str(row["symbol"])))
        size = max(1, math.ceil(len(ordered) * 0.1)) if "decile" in spec["policy"] else spec["top_n"]
        selected = ordered[:min(size, max_positions)]
        if len(selected) < min_positions: infeasible = "minimum_positions_not_met"; break
        exposure = 1.0 - cash_buffer
        weights = build_sizing_weights(selected, signal, spec["sizing_method"], exposure, spec["max_position_weight_limit"])
        if spec["policy"].startswith("long_short"):
            bottom = ordered[-min(size, max_positions):]
            weights = build_sizing_weights(selected, signal, spec["sizing_method"], exposure / 2, spec["max_position_weight_limit"])
            weights.update(build_sizing_weights(list(reversed(bottom)), signal, spec["sizing_method"], -exposure / 2, spec["max_position_weight_limit"]))
        if spec["volatility_target"] is not None:
            estimated = sum(abs(weight) * max(finite_number(next(row for row in group if str(row["symbol"]) == symbol).get("predicted_volatility_20d")) or 0.0, 0.0) for symbol, weight in weights.items())
            if estimated > 0:
                scale = min(1.0, float(spec["volatility_target"]) / estimated)
                weights = {symbol: weight * scale for symbol, weight in weights.items()}
        if not allow_partial_cash and sum(abs(v) for v in weights.values()) + 1e-9 < exposure: infeasible = "position_cap_prevents_target_exposure"; break
        turnover = sum(abs(weights.get(s, 0) - previous.get(s, 0)) for s in set(weights) | set(previous))
        if spec["turnover_cap"] is not None and turnover > float(spec["turnover_cap"]) + 1e-12: infeasible = "turnover_cap_exceeded"; break
        gross = sum(weight * float(next(row[TARGET] for row in group if str(row["symbol"]) == symbol)) for symbol, weight in weights.items())
        cost_drag = turnover * spec["cost_bps"] / 10000; slippage_drag = turnover * spec["slippage_bps"] / 10000
        borrow_drag = sum(abs(v) for v in weights.values() if v < 0) * borrow_cost_bps / 10000
        net = gross - cost_drag - slippage_drag - borrow_drag; equity *= 1 + net
        periods.append({"rebalance_date": rebalance_date, "strategy_id": spec["config_id"], "gross_return": gross, "net_return": net, "transaction_cost_drag": cost_drag + slippage_drag + borrow_drag, "slippage_drag": slippage_drag, "turnover": turnover, "equity": equity})
        holdings.extend({"rebalance_date": rebalance_date, "strategy_id": spec["config_id"], "symbol": symbol, "weight": weight} for symbol, weight in sorted(weights.items()))
        previous = weights
    if infeasible:
        return ({**spec, "status": "infeasible", "infeasible_reason": infeasible}, [], [])
    metric = _metrics(signal, spec["policy"], periods, [{**row, "signal_column": signal, "policy": spec["policy"], "side": "long" if row["weight"] > 0 else "short"} for row in holdings])
    returns = [row["net_return"] for row in periods]; turns = sorted(row["turnover"] for row in periods); concentrations = sorted(max((abs(h["weight"]) for h in holdings if h["rebalance_date"] == row["rebalance_date"]), default=0) for row in periods)
    metric.update(spec); metric.update({"status": "completed", "infeasible_reason": None, "slippage_drag": sum(row["slippage_drag"] for row in periods), "performance_by_year": _by_year(periods), "worst_20_period_drawdown": min((math.prod(1 + value for value in returns[i:i+20]) - 1 for i in range(len(returns))), default=None), "best_20_period_return": max((math.prod(1 + value for value in returns[i:i+20]) - 1 for i in range(len(returns))), default=None), "percentage_of_periods_in_cash": mean([sum(abs(h["weight"]) for h in holdings if h["rebalance_date"] == row["rebalance_date"]) < 0.999 for row in periods]) if periods else None, "turnover_percentile_95": _percentile(turns, .95), "position_concentration_percentile_95": _percentile(concentrations, .95)})
    return metric, periods, holdings


def _by_year(periods: list[dict[str, Any]]) -> dict[str, float]:
    output: dict[str, list[float]] = {}
    for row in periods: output.setdefault(str(row["rebalance_date"])[:4], []).append(row["net_return"])
    return {year: math.prod(1 + value for value in values) - 1 for year, values in output.items()}


def _percentile(values: list[float], fraction: float) -> float | None:
    return values[min(len(values) - 1, math.ceil(len(values) * fraction) - 1)] if values else None


def _best(rows: list[dict[str, Any]], metric: str, *, kind: str | None = None, lowest: bool = False) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get("status") == "completed" and row.get(metric) is not None and (kind is None or row.get("kind") == kind)]
    return (min if lowest else max)(valid, key=lambda row: float(row[metric]), default=None)


def _winners(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ml = _best(rows, "net_return", kind="ml_model"); momentum = _best([r for r in rows if r.get("signal_column") == "predicted_momentum_120d"], "net_return")
    return {"best_by_total_return": _best(rows, "total_return"), "best_by_net_return_after_costs": _best(rows, "net_return"), "best_by_sharpe": _best(rows, "sharpe"), "best_by_calmar": _best(rows, "calmar_ratio"), "best_by_max_drawdown": _best(rows, "max_drawdown"), "best_by_lowest_turnover": _best(rows, "average_turnover", lowest=True), "best_ml_policy": ml, "best_baseline_policy": _best(rows, "net_return", kind="baseline"), "best_ml_vs_momentum_120d": {"ml_available": ml is not None, "momentum_120d_available": momentum is not None, "comparison_available": ml is not None and momentum is not None, "beats_momentum_120d": bool(ml and momentum and ml["net_return"] > momentum["net_return"]), "net_return_delta": ml["net_return"] - momentum["net_return"] if ml and momentum else None, "comparison_missing_reason": None if ml and momentum else "Both a feasible ML policy and momentum_120d baseline policy are required"}, "best_policy_under_turnover_cap": _best([r for r in rows if r.get("turnover_cap") is not None], "net_return"), "best_policy_under_max_drawdown_limit": _best([r for r in rows if r.get("max_drawdown", -1) >= -0.20], "net_return")}


def _negative_return_warnings(rows: list[dict[str, Any]]) -> dict[str, bool]:
    values = [float(row["net_return"]) for row in rows if row.get("status") == "completed" and row.get("net_return") is not None]
    return {"all_candidate_net_returns_negative": bool(values) and all(value < 0 for value in values), "best_return_is_negative": bool(values) and max(values) < 0}


def _markdown(payload: dict[str, Any]) -> str:
    winner = payload["winners"].get("best_by_net_return_after_costs") or {}
    return "\n".join(["# Stock-Level Portfolio Policy Sweep", "", "Research only. Trading impact: none. Production validated: false.", "", f"- Run size: `{payload['run_size']}`", f"- Policy configs: {payload['policy_config_count']}", f"- Baseline available: {payload['baseline_coverage']['baseline_signal_available']}", f"- Baselines found: {payload['baseline_coverage']['baseline_signal_columns_found']}", f"- All candidate net returns negative: {payload['all_candidate_net_returns_negative']}", f"- Best return is negative: {payload['best_return_is_negative']}", f"- Best signal: {winner.get('signal_column')}", f"- Best policy: {winner.get('policy')}", f"- Best sizing: {winner.get('sizing_method')}", "- Promotion thresholds changed: false", ""])
