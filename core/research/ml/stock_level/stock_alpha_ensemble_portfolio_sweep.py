from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping

from core.research.framework.data import CsvRowRepository
from core.research.framework.ranking import finite_number
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir


TARGET_COLUMN = "actual_forward_return_10d"
GUARDRAILS = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
    "promotion_thresholds_changed": False,
}
_WORKER_ROWS: list[Mapping[str, Any]] = []
_WORKER_TARGET_COLUMN = TARGET_COLUMN
_WORKER_COST_BPS = 10.0
_WORKER_SLIPPAGE_BPS = 5.0
_WORKER_UNDERINVESTMENT_THRESHOLD = 0.75


@dataclass(frozen=True)
class StockAlphaEnsemblePortfolioSweepPaths:
    csv_path: Path
    ranked_csv_path: Path
    json_path: Path
    markdown_path: Path
    equity_curves_path: Path
    holdings_path: Path
    trades_path: Path
    best_equity_curve_path: Path


def write_stock_alpha_ensemble_portfolio_sweep(
    config: Mapping[str, Any],
) -> StockAlphaEnsemblePortfolioSweepPaths:
    ml = dict(config.get("ml", {}) or {})
    source = Path(str(ml["stock_alpha_portfolio_sweep_source_predictions_path"]))
    if not source.exists():
        raise ValueError(f"stock-alpha portfolio sweep source predictions file not found: {source}")
    rows = CsvRowRepository().read(source)
    if not rows:
        raise ValueError(f"stock-alpha portfolio sweep source has no rows: {source}")
    signals = [str(value) for value in ml.get("stock_alpha_portfolio_sweep_signal_columns", [])]
    if not signals:
        raise ValueError("ml.stock_alpha_portfolio_sweep_signal_columns must list at least one signal")
    all_policies = build_ensemble_portfolio_policy_grid(config, rows)
    estimated_policy_count = len(all_policies)
    policies = all_policies
    max_configs = ml.get("stock_alpha_portfolio_sweep_max_policy_configs")
    if max_configs is not None:
        policies = policies[: int(max_configs)]
    progress_every = int(ml.get("stock_alpha_portfolio_sweep_progress_every", 50))
    n_jobs = max(1, int(ml.get("stock_alpha_portfolio_sweep_n_jobs", 1)))
    underinvestment_threshold = float(ml.get("stock_alpha_portfolio_sweep_underinvestment_threshold", 0.75))
    summaries, curves, holdings, trades, payload = build_ensemble_portfolio_policy_sweep(
        rows,
        policies=policies,
        target_column=str(ml.get("stock_alpha_portfolio_sweep_target_column", TARGET_COLUMN)),
        cost_bps=float(ml.get("stock_alpha_portfolio_sweep_cost_bps", 10.0)),
        slippage_bps=float(ml.get("stock_alpha_portfolio_sweep_slippage_bps", 5.0)),
        collect_all_details=bool(ml.get("stock_alpha_portfolio_sweep_write_all_equity_curves", False))
        or bool(ml.get("stock_alpha_portfolio_sweep_write_all_holdings", False))
        or bool(ml.get("stock_alpha_portfolio_sweep_write_all_trades", False)),
        progress_every=progress_every,
        n_jobs=n_jobs,
        underinvestment_threshold=underinvestment_threshold,
    )
    detail_count = int(ml.get("stock_alpha_portfolio_sweep_top_policy_detail_count", 10))
    detail_ids = [
        str(row["strategy_id"])
        for row in payload.get("ranked_policies", [])[:detail_count]
    ] if bool(ml.get("stock_alpha_portfolio_sweep_write_top_policy_details", True)) else []
    if detail_ids:
        detail_curves, detail_holdings, detail_trades = replay_policy_details(
            rows,
            policies=[policy for policy in policies if policy["strategy_id"] in set(detail_ids)],
            target_column=str(ml.get("stock_alpha_portfolio_sweep_target_column", TARGET_COLUMN)),
            cost_bps=float(ml.get("stock_alpha_portfolio_sweep_cost_bps", 10.0)),
            slippage_bps=float(ml.get("stock_alpha_portfolio_sweep_slippage_bps", 5.0)),
        )
    else:
        detail_curves, detail_holdings, detail_trades = [], [], []
    write_all_equity = bool(ml.get("stock_alpha_portfolio_sweep_write_all_equity_curves", False))
    write_all_holdings = bool(ml.get("stock_alpha_portfolio_sweep_write_all_holdings", False))
    write_all_trades = bool(ml.get("stock_alpha_portfolio_sweep_write_all_trades", False))
    output_curves = curves if write_all_equity else detail_curves
    output_holdings = holdings if write_all_holdings else detail_holdings
    output_trades = trades if write_all_trades else detail_trades
    payload.update({"source_predictions_path": str(source), "candidate_signal_columns": signals})
    payload["experiment_stage"] = str(ml.get("stock_alpha_portfolio_sweep_experiment_stage", "full"))
    payload["estimated_policy_count"] = estimated_policy_count
    payload["output_controls"] = {
        "write_all_equity_curves": write_all_equity,
        "write_all_holdings": write_all_holdings,
        "write_all_trades": write_all_trades,
        "write_top_policy_details": bool(ml.get("stock_alpha_portfolio_sweep_write_top_policy_details", True)),
        "top_policy_detail_count": detail_count,
        "max_policy_configs": max_configs,
        "progress_every": progress_every,
        "n_jobs": n_jobs,
        "underinvestment_threshold": underinvestment_threshold,
        "turnover_cap_initial_investment": bool(ml.get("stock_alpha_portfolio_sweep_turnover_cap_initial_investment", True)),
        "detail_strategy_ids": detail_ids,
    }
    output = stock_alpha_output_dir(config) / "portfolio_sweep" / "ensemble"
    paths = StockAlphaEnsemblePortfolioSweepPaths(
        csv_path=output / "policy_sweep_raw.csv",
        ranked_csv_path=output / "policy_sweep_ranked.csv",
        json_path=output / "stock_alpha_ensemble_portfolio_policy_sweep.json",
        markdown_path=output / "stock_alpha_ensemble_portfolio_policy_sweep.md",
        equity_curves_path=output / "policy_sweep_top_policy_equity_curves.csv",
        holdings_path=output / "policy_sweep_top_policy_holdings.csv",
        trades_path=output / "policy_sweep_top_policy_trades.csv",
        best_equity_curve_path=output / "policy_sweep_best_policy_equity_curve.csv",
    )
    best_id = (payload["best_policy_summary"] or {}).get("strategy_id")
    best_curve = [row for row in output_curves if row.get("strategy_id") == best_id]
    writer = ResearchArtifactWriter()
    writer.write_csv(paths.csv_path, summaries, fieldnames=list(summaries[0]) if summaries else ["strategy_id"])
    ranked = payload.get("ranked_policies", [])
    writer.write_csv(paths.ranked_csv_path, ranked, fieldnames=list(ranked[0]) if ranked else ["strategy_id"])
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))
    writer.write_csv(paths.equity_curves_path, output_curves, fieldnames=list(output_curves[0]) if output_curves else ["rebalance_date"])
    writer.write_csv(paths.holdings_path, output_holdings, fieldnames=list(output_holdings[0]) if output_holdings else ["rebalance_date"])
    writer.write_csv(paths.trades_path, output_trades, fieldnames=list(output_trades[0]) if output_trades else ["rebalance_date"])
    writer.write_csv(paths.best_equity_curve_path, best_curve, fieldnames=list(best_curve[0]) if best_curve else ["rebalance_date"])
    return paths


def build_ensemble_portfolio_policy_grid(
    config: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ml = dict(config.get("ml", {}) or {})
    available = set(rows[0]) if rows else set()
    signals = [
        signal
        for signal in ml.get("stock_alpha_portfolio_sweep_signal_columns", [])
        if signal in available
    ]
    if not signals:
        raise ValueError("No configured portfolio sweep signal columns are present in source predictions")
    thresholds = ml.get("stock_alpha_portfolio_sweep_minimum_signal_thresholds", [None])
    turnover_mode = str(ml.get("stock_alpha_portfolio_sweep_turnover_mode", "strict_top_n"))
    if turnover_mode not in {"strict_top_n", "gradual_transition"}:
        raise ValueError("ml.stock_alpha_portfolio_sweep_turnover_mode must be strict_top_n or gradual_transition")
    policies = []
    initial_investment = bool(ml.get("stock_alpha_portfolio_sweep_turnover_cap_initial_investment", True))
    policy_index = 0
    for signal in signals:
        for top_n in ml.get("stock_alpha_portfolio_sweep_top_n_values", [5, 10, 20, 30]):
            for max_weight in ml.get("stock_alpha_portfolio_sweep_max_position_weights", [0.05, 0.075, 0.10]):
                for cash_buffer in ml.get("stock_alpha_portfolio_sweep_cash_buffers", [0.0, 0.05, 0.10]):
                    for turnover_cap in ml.get("stock_alpha_portfolio_sweep_turnover_caps", [None, 0.25, 0.50]):
                        for threshold in thresholds:
                            policies.append(
                                {
                                    "policy_index": policy_index,
                                    "strategy_id": f"{signal}|top{int(top_n)}|w{float(max_weight):.3f}|cash{float(cash_buffer):.2f}|turn{turnover_cap}|thr{threshold}",
                                    "signal_column": str(signal),
                                    "top_n": int(top_n),
                                    "max_position_weight": float(max_weight),
                                    "cash_buffer": float(cash_buffer),
                                    "target_gross_exposure": target_gross_exposure(int(top_n), float(max_weight), float(cash_buffer)),
                                    "target_exposure_bucket": exposure_bucket(target_gross_exposure(int(top_n), float(max_weight), float(cash_buffer))),
                                    "exposure_bucket": exposure_bucket(target_gross_exposure(int(top_n), float(max_weight), float(cash_buffer))),
                                    "cash_buffer_inactive": target_gross_exposure(int(top_n), float(max_weight), float(cash_buffer)) < 1.0 - float(cash_buffer),
                                    "turnover_mode": turnover_mode,
                                    "turnover_cap_initial_investment": bool(ml.get("stock_alpha_portfolio_sweep_turnover_cap_initial_investment", True)),
                                    "turnover_cap": None if turnover_cap is None else float(turnover_cap),
                                    "minimum_signal_threshold": threshold,
                                }
                            )
                            policy_index += 1
    return policies


def build_ensemble_portfolio_policy_sweep(
    rows: list[Mapping[str, Any]],
    *,
    policies: list[dict[str, Any]],
    target_column: str = TARGET_COLUMN,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    collect_all_details: bool = True,
    progress_every: int = 0,
    n_jobs: int = 1,
    underinvestment_threshold: float = 0.75,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    holdings: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    if n_jobs <= 1:
        summaries, curves, holdings, trades = _evaluate_policies_sequential(
            rows,
            policies,
            target_column=target_column,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            collect_all_details=collect_all_details,
            progress_every=progress_every,
            underinvestment_threshold=underinvestment_threshold,
        )
    else:
        summaries = _evaluate_policies_parallel(
            rows,
            policies,
            target_column=target_column,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            progress_every=progress_every,
            n_jobs=n_jobs,
            underinvestment_threshold=underinvestment_threshold,
        )
        if collect_all_details:
            curves, holdings, trades = replay_policy_details(
                rows,
                policies=policies,
                target_column=target_column,
                cost_bps=cost_bps,
                slippage_bps=slippage_bps,
            )
    ranked = rank_policies(summaries)
    top_20 = ranked[:20]
    diagnostics = policy_grid_diagnostics(summaries)
    payload = {
        "mode": "stock_alpha_ensemble_portfolio_policy_sweep_research_only",
        "policy_config_count": len(policies),
        "policy_grid_size": len(policies),
        "parallelism": {"n_jobs": n_jobs, "policy_level_parallelism": n_jobs > 1},
        "underinvestment_threshold": underinvestment_threshold,
        "ranking_method": [
            "max_drawdown_constraint",
            "cost_adjusted_sharpe",
            "cost_adjusted_return",
            "turnover",
            "concentration",
        ],
        "summary": summaries,
        "ranked_policies": ranked,
        "top_20_ranked_policies": top_20,
        "best_policy_summary": ranked[0] if ranked else None,
        "best_policy_per_signal": best_policy_per_signal(summaries),
        "best_policy_per_exposure_bucket": best_policy_per_exposure_bucket(summaries),
        "best_policy_per_signal_per_exposure_bucket": best_policy_per_signal_per_exposure_bucket(summaries),
        "best_policy_per_target_exposure_bucket": best_policy_per_target_exposure_bucket(summaries),
        "best_policy_per_realized_exposure_bucket": best_policy_per_realized_exposure_bucket(summaries),
        "best_policy_per_signal_per_realized_exposure_bucket": best_policy_per_signal_per_realized_exposure_bucket(summaries),
        "underinvested_policy_count": diagnostics["underinvested_policy_count"],
        "most_underinvested_policies": diagnostics["most_underinvested_policies"],
        "drawdown_frontier": drawdown_frontier(summaries),
        "exposure_explanation": exposure_explanation(summaries),
        "effective_exposure_buckets": diagnostics["effective_exposure_buckets"],
        "duplicate_or_inactive_config_warnings": diagnostics["duplicate_or_inactive_config_warnings"],
        "cash_buffer_inactive_count": diagnostics["cash_buffer_inactive_count"],
        **GUARDRAILS,
    }
    return summaries, curves, holdings, trades, payload


def _evaluate_policies_sequential(
    rows: list[Mapping[str, Any]],
    policies: list[dict[str, Any]],
    *,
    target_column: str,
    cost_bps: float,
    slippage_bps: float,
    collect_all_details: bool,
    progress_every: int,
    underinvestment_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    holdings: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    started = time.perf_counter()
    total = len(policies)
    for index, policy in enumerate(policies, start=1):
        if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == total):
            elapsed = time.perf_counter() - started
            print(
                "[stock-alpha portfolio sweep] "
                f"policy {index}/{total} signal={policy['signal_column']} "
                f"top_n={policy['top_n']} elapsed={elapsed:.1f}s"
            )
        summary, periods, policy_holdings, policy_trades = _evaluate_one_policy(
            rows,
            policy,
            target_column=target_column,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            underinvestment_threshold=underinvestment_threshold,
        )
        summaries.append(summary)
        if collect_all_details:
            curves.extend(periods)
            holdings.extend(policy_holdings)
            trades.extend(policy_trades)
    return summaries, curves, holdings, trades


def _evaluate_policies_parallel(
    rows: list[Mapping[str, Any]],
    policies: list[dict[str, Any]],
    *,
    target_column: str,
    cost_bps: float,
    slippage_bps: float,
    progress_every: int,
    n_jobs: int,
    underinvestment_threshold: float,
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    total = len(policies)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=n_jobs,
        initializer=_init_policy_worker,
        initargs=(rows, target_column, cost_bps, slippage_bps, underinvestment_threshold),
    ) as executor:
        futures = {
            executor.submit(_evaluate_policy_worker, policy): policy
            for policy in policies
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            policy = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                raise RuntimeError(
                    "stock-alpha portfolio sweep worker failed: "
                    f"strategy_id={policy.get('strategy_id')} "
                    f"policy_index={policy.get('policy_index')} "
                    f"error={exc}"
                ) from exc
            if progress_every > 0 and (completed == 1 or completed % progress_every == 0 or completed == total):
                elapsed = time.perf_counter() - started
                print(
                    "[stock-alpha portfolio sweep] "
                    f"completed {completed}/{total} n_jobs={n_jobs} "
                    f"elapsed={elapsed:.1f}s latest={policy.get('strategy_id')}"
                )
    return sorted(results, key=lambda row: int(row.get("policy_index", 0)))


def _init_policy_worker(
    rows: list[Mapping[str, Any]],
    target_column: str,
    cost_bps: float,
    slippage_bps: float,
    underinvestment_threshold: float,
) -> None:
    global _WORKER_ROWS, _WORKER_TARGET_COLUMN, _WORKER_COST_BPS, _WORKER_SLIPPAGE_BPS, _WORKER_UNDERINVESTMENT_THRESHOLD
    _WORKER_ROWS = rows
    _WORKER_TARGET_COLUMN = target_column
    _WORKER_COST_BPS = cost_bps
    _WORKER_SLIPPAGE_BPS = slippage_bps
    _WORKER_UNDERINVESTMENT_THRESHOLD = underinvestment_threshold


def _evaluate_policy_worker(policy: dict[str, Any]) -> dict[str, Any]:
    summary, _, _, _ = _evaluate_one_policy(
        _WORKER_ROWS,
        policy,
        target_column=_WORKER_TARGET_COLUMN,
        cost_bps=_WORKER_COST_BPS,
        slippage_bps=_WORKER_SLIPPAGE_BPS,
        underinvestment_threshold=_WORKER_UNDERINVESTMENT_THRESHOLD,
    )
    return summary


def _evaluate_one_policy(
    rows: list[Mapping[str, Any]],
    policy: Mapping[str, Any],
    *,
    target_column: str,
    cost_bps: float,
    slippage_bps: float,
    underinvestment_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    periods, policy_holdings, policy_trades = replay_signal_portfolio(
        rows,
        signal_column=policy["signal_column"],
        target_column=target_column,
        top_n=int(policy["top_n"]),
        max_position_weight=float(policy["max_position_weight"]),
        cash_buffer=float(policy["cash_buffer"]),
        minimum_signal_threshold=policy.get("minimum_signal_threshold"),
        turnover_cap=policy.get("turnover_cap"),
        turnover_mode=str(policy.get("turnover_mode", "strict_top_n")),
        turnover_cap_initial_investment=bool(policy.get("turnover_cap_initial_investment", True)),
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        strategy_id=policy["strategy_id"],
    )
    return portfolio_metrics(policy, periods, policy_holdings, policy_trades, underinvestment_threshold=underinvestment_threshold), periods, policy_holdings, policy_trades


def replay_policy_details(
    rows: list[Mapping[str, Any]],
    *,
    policies: list[dict[str, Any]],
    target_column: str = TARGET_COLUMN,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    curves: list[dict[str, Any]] = []
    holdings: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    for policy in policies:
        periods, policy_holdings, policy_trades = replay_signal_portfolio(
            rows,
            signal_column=policy["signal_column"],
            target_column=target_column,
            top_n=int(policy["top_n"]),
            max_position_weight=float(policy["max_position_weight"]),
            cash_buffer=float(policy["cash_buffer"]),
            minimum_signal_threshold=policy.get("minimum_signal_threshold"),
            turnover_cap=policy.get("turnover_cap"),
            turnover_mode=str(policy.get("turnover_mode", "strict_top_n")),
            turnover_cap_initial_investment=bool(policy.get("turnover_cap_initial_investment", True)),
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            strategy_id=policy["strategy_id"],
        )
        curves.extend(periods)
        holdings.extend(policy_holdings)
        trades.extend(policy_trades)
    return curves, holdings, trades


def target_gross_exposure(top_n: int, max_position_weight: float, cash_buffer: float) -> float:
    return min(max(0.0, int(top_n) * float(max_position_weight)), max(0.0, 1.0 - float(cash_buffer)))


def exposure_bucket(exposure: float) -> str:
    if exposure <= 0.25 + 1e-12:
        return "low_0_25"
    if exposure <= 0.50 + 1e-12:
        return "medium_0_50"
    if exposure <= 0.75 + 1e-12:
        return "high_0_75"
    return "full_1_00"


def replay_signal_portfolio(
    rows: list[Mapping[str, Any]],
    *,
    signal_column: str,
    target_column: str = TARGET_COLUMN,
    top_n: int,
    max_position_weight: float,
    cash_buffer: float,
    minimum_signal_threshold: Any = None,
    turnover_cap: float | None = None,
    turnover_mode: str = "strict_top_n",
    turnover_cap_initial_investment: bool = True,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    strategy_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_date: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if finite_number(row.get(signal_column)) is not None and finite_number(row.get(target_column)) is not None:
            by_date.setdefault(str(row.get("rebalance_date")), []).append(row)
    previous: dict[str, float] = {}
    equity = 1.0
    periods: list[dict[str, Any]] = []
    holdings: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    strategy = strategy_id or signal_column
    if turnover_mode not in {"strict_top_n", "gradual_transition"}:
        raise ValueError("turnover_mode must be strict_top_n or gradual_transition")
    for rebalance_date, group in sorted(by_date.items()):
        selected = select_top_signal_rows(group, signal_column, top_n, minimum_signal_threshold)
        target_symbols = {str(row["symbol"]).upper() for row in selected}
        weights = equal_weight_with_caps(selected, max_position_weight=max_position_weight, cash_buffer=cash_buffer)
        turnover = sum(abs(weights.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in set(weights) | set(previous))
        apply_cap = turnover_cap is not None and (previous or not turnover_cap_initial_investment)
        if apply_cap and turnover > turnover_cap:
            weights = _apply_turnover_cap(previous, weights, float(turnover_cap), mode=turnover_mode)
            turnover = sum(abs(weights.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in set(weights) | set(previous))
        legacy_symbols = set(weights) - target_symbols
        legacy_weight = sum(abs(weights[symbol]) for symbol in legacy_symbols)
        gross = sum(float(row[target_column]) * weights.get(str(row["symbol"]).upper(), 0.0) for row in group)
        transaction_cost = turnover * (cost_bps + slippage_bps) / 10_000.0
        net = gross - transaction_cost
        equity *= 1.0 + net
        periods.append(
            {
                "rebalance_date": rebalance_date,
                "strategy_id": strategy,
                "signal_column": signal_column,
                "gross_return": gross,
                "net_return": net,
                "cost_adjusted_return": net,
                "estimated_transaction_cost": transaction_cost,
                "turnover": turnover,
                "equity": equity,
                "holding_count": len(weights),
                "target_holding_count": len(target_symbols),
                "legacy_holding_count": len(legacy_symbols),
                "legacy_holding_weight": legacy_weight,
                "top_n_violation": len(weights) > top_n,
                "gross_exposure": sum(abs(value) for value in weights.values()),
            }
        )
        for symbol, weight in sorted(weights.items()):
            row = next(item for item in group if str(item["symbol"]).upper() == symbol)
            holdings.append(
                {
                    "rebalance_date": rebalance_date,
                    "strategy_id": strategy,
                    "signal_column": signal_column,
                    "symbol": symbol,
                    "sector": row.get("sector", ""),
                    "weight": weight,
                    "is_legacy_holding": symbol not in target_symbols,
                }
            )
        for symbol in sorted(set(weights) | set(previous)):
            delta = weights.get(symbol, 0.0) - previous.get(symbol, 0.0)
            if abs(delta) > 1e-12:
                trades.append(
                    {
                        "rebalance_date": rebalance_date,
                        "strategy_id": strategy,
                        "symbol": symbol,
                        "weight_delta": delta,
                        "turnover_contribution": abs(delta),
                    }
                )
        previous = weights
    return periods, holdings, trades


def select_top_signal_rows(
    rows: list[Mapping[str, Any]],
    signal_column: str,
    top_n: int,
    minimum_signal_threshold: Any = None,
) -> list[Mapping[str, Any]]:
    values = [row for row in rows if finite_number(row.get(signal_column)) is not None]
    if minimum_signal_threshold is not None:
        threshold = _resolve_threshold(values, signal_column, minimum_signal_threshold)
        values = [row for row in values if float(row[signal_column]) >= threshold]
    return sorted(values, key=lambda row: (-float(row[signal_column]), str(row["symbol"]).upper()))[:top_n]


def equal_weight_with_caps(
    rows: list[Mapping[str, Any]],
    *,
    max_position_weight: float,
    cash_buffer: float,
) -> dict[str, float]:
    if not rows:
        return {}
    exposure = max(0.0, min(1.0, 1.0 - cash_buffer))
    weight = min(float(max_position_weight), exposure / len(rows))
    return {str(row["symbol"]).upper(): weight for row in rows}


def portfolio_metrics(
    policy: Mapping[str, Any],
    periods: list[Mapping[str, Any]],
    holdings: list[Mapping[str, Any]],
    trades: list[Mapping[str, Any]],
    *,
    underinvestment_threshold: float = 0.75,
) -> dict[str, Any]:
    returns = [float(row["net_return"]) for row in periods]
    gross_returns = [float(row["gross_return"]) for row in periods]
    equity = [1.0, *[float(row["equity"]) for row in periods]]
    max_drawdown = _max_drawdown(equity)
    volatility = pstdev(returns) if len(returns) > 1 else 0.0
    annualized = _annualized_return(equity[-1], len(returns))
    sharpe = mean(returns) / volatility * math.sqrt(26.0) if volatility > 0 else None
    by_date = {row["rebalance_date"]: [h for h in holdings if h["rebalance_date"] == row["rebalance_date"]] for row in periods}
    sector_concentration = [
        value
        for value in (_sector_concentration(items) for items in by_date.values())
        if value is not None
    ]
    weights = [abs(float(row["weight"])) for row in holdings]
    costs = [float(row["estimated_transaction_cost"]) for row in periods]
    gross_exposures = [float(row["gross_exposure"]) for row in periods]
    turnovers = [float(row["turnover"]) for row in periods]
    legacy_counts = [int(row.get("legacy_holding_count", 0)) for row in periods]
    legacy_weights = [float(row.get("legacy_holding_weight", 0.0)) for row in periods]
    target_counts = [int(row.get("target_holding_count", 0)) for row in periods]
    actual_counts = [int(row.get("holding_count", 0)) for row in periods]
    top_n_violations = [bool(row.get("top_n_violation")) for row in periods]
    target_exposure = float(policy.get("target_gross_exposure", target_gross_exposure(int(policy.get("top_n", 0)), float(policy.get("max_position_weight", 0.0)), float(policy.get("cash_buffer", 0.0)))))
    average_gross_exposure = mean(gross_exposures) if gross_exposures else 0.0
    utilization = average_gross_exposure / target_exposure if target_exposure > 0.0 else None
    underinvested = bool(utilization is not None and utilization < underinvestment_threshold)
    return {
        **dict(policy),
        "status": "completed" if periods else "no_eligible_periods",
        "target_gross_exposure": target_exposure,
        "target_exposure_bucket": str(policy.get("target_exposure_bucket", policy.get("exposure_bucket", exposure_bucket(target_exposure)))),
        "realized_average_gross_exposure": average_gross_exposure,
        "realized_max_gross_exposure": max(gross_exposures, default=0.0),
        "realized_exposure_bucket": exposure_bucket(average_gross_exposure),
        "exposure_utilization_ratio": utilization,
        "underinvested_policy": underinvested,
        "underinvestment_warning": (
            f"realized_average_gross_exposure {average_gross_exposure:.4f} is below "
            f"{underinvestment_threshold:.2f} * target_gross_exposure {target_exposure:.4f}"
            if underinvested
            else None
        ),
        "unused_cash_estimate": max(0.0, 1.0 - average_gross_exposure),
        "exposure_bucket": str(policy.get("exposure_bucket", exposure_bucket(target_exposure))),
        "cash_buffer_inactive": bool(policy.get("cash_buffer_inactive", target_exposure < 1.0 - float(policy.get("cash_buffer", 0.0)))),
        "turnover_mode": str(policy.get("turnover_mode", "strict_top_n")),
        "turnover_cap_initial_investment": bool(policy.get("turnover_cap_initial_investment", True)),
        "cumulative_return": equity[-1] - 1.0,
        "annualized_return": annualized,
        "volatility": volatility,
        "sharpe": sharpe,
        "cost_adjusted_sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "hit_rate": mean([value > 0.0 for value in returns]) if returns else None,
        "average_turnover": mean([float(row["turnover"]) for row in periods]) if periods else None,
        "total_turnover": sum(float(row["turnover"]) for row in periods),
        "realized_max_turnover": max(turnovers, default=0.0),
        "estimated_transaction_cost": sum(costs),
        "cost_adjusted_return": sum(returns),
        "gross_return": sum(gross_returns),
        "average_target_holdings": mean(target_counts) if target_counts else 0.0,
        "average_actual_holdings": mean(actual_counts) if actual_counts else 0.0,
        "max_actual_holdings": max(actual_counts, default=0),
        "average_legacy_holding_count": mean(legacy_counts) if legacy_counts else 0.0,
        "max_legacy_holding_count": max(legacy_counts, default=0),
        "average_legacy_holding_weight": mean(legacy_weights) if legacy_weights else 0.0,
        "max_legacy_holding_weight": max(legacy_weights, default=0.0),
        "top_n_violation_count": sum(top_n_violations),
        "top_n_violation_ratio": mean(top_n_violations) if top_n_violations else 0.0,
        "average_number_of_holdings": mean([len(items) for items in by_date.values()]) if by_date else 0.0,
        "max_single_name_concentration": max(weights, default=0.0),
        "max_sector_concentration": max(sector_concentration) if sector_concentration else None,
        "date_count": len(periods),
        "symbol_count": len({row["symbol"] for row in holdings}),
        "trade_count": len(trades),
    }


def rank_policies(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    completed = [dict(row) for row in rows if row.get("status") == "completed"]
    return sorted(
        completed,
        key=lambda row: (
            -float(row.get("max_drawdown") or -1.0),
            -float(row.get("cost_adjusted_sharpe") or -1e9),
            -float(row.get("cost_adjusted_return") or -1e9),
            float(row.get("average_turnover") or 1e9),
            float(row.get("max_single_name_concentration") or 1e9),
        ),
    )


def best_policy_per_signal(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for signal in sorted({str(row.get("signal_column")) for row in rows if row.get("signal_column")}):
        ranked = rank_policies([row for row in rows if row.get("signal_column") == signal])
        if ranked:
            output[signal] = ranked[0]
    return output


def best_policy_per_exposure_bucket(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for bucket in ("low_0_25", "medium_0_50", "high_0_75", "full_1_00"):
        ranked = rank_policies([row for row in rows if row.get("exposure_bucket") == bucket])
        if ranked:
            output[bucket] = ranked[0]
    return output


def best_policy_per_signal_per_exposure_bucket(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for signal in sorted({str(row.get("signal_column")) for row in rows if row.get("signal_column")}):
        bucketed = best_policy_per_exposure_bucket([row for row in rows if row.get("signal_column") == signal])
        if bucketed:
            output[signal] = bucketed
    return output


def best_policy_per_target_exposure_bucket(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return _best_policy_per_bucket(rows, "target_exposure_bucket")


def best_policy_per_realized_exposure_bucket(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return _best_policy_per_bucket(rows, "realized_exposure_bucket")


def best_policy_per_signal_per_realized_exposure_bucket(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for signal in sorted({str(row.get("signal_column")) for row in rows if row.get("signal_column")}):
        bucketed = _best_policy_per_bucket([row for row in rows if row.get("signal_column") == signal], "realized_exposure_bucket")
        if bucketed:
            output[signal] = bucketed
    return output


def _best_policy_per_bucket(rows: list[Mapping[str, Any]], bucket_column: str) -> dict[str, dict[str, Any]]:
    output = {}
    for bucket in ("low_0_25", "medium_0_50", "high_0_75", "full_1_00"):
        ranked = rank_policies([row for row in rows if row.get(bucket_column) == bucket])
        if ranked:
            output[bucket] = ranked[0]
    return output


def drawdown_frontier(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any] | None]:
    return {
        "max_drawdown_gte_-0.20": _best_under_drawdown(rows, -0.20),
        "max_drawdown_gte_-0.30": _best_under_drawdown(rows, -0.30),
        "max_drawdown_gte_-0.40": _best_under_drawdown(rows, -0.40),
    }


def exposure_explanation(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    inactive = [
        {
            "strategy_id": row.get("strategy_id"),
            "signal_column": row.get("signal_column"),
            "target_gross_exposure": row.get("target_gross_exposure"),
            "cash_buffer": row.get("cash_buffer"),
            "reason": "top_n * max_position_weight is below 1 - cash_buffer, so cash buffer is inactive because the portfolio is already underinvested.",
        }
        for row in rows
        if row.get("cash_buffer_inactive")
    ]
    return {
        "cash_buffer_inactive_policy_count": len(inactive),
        "cash_buffer_inactive_examples": inactive[:20],
    }


def policy_grid_diagnostics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    cash_inactive = [row for row in rows if row.get("cash_buffer_inactive")]
    turnover_inactive = [
        row
        for row in completed
        if row.get("turnover_cap") is not None
        and row.get("realized_max_turnover") is not None
        and float(row["realized_max_turnover"]) <= float(row["turnover_cap"])
    ]
    underinvested = sorted(
        [row for row in completed if row.get("underinvested_policy")],
        key=lambda row: float(row.get("exposure_utilization_ratio") or 0.0),
    )
    wide_books = [
        row for row in completed
        if row.get("average_actual_holdings") is not None
        and float(row["average_actual_holdings"]) > float(row.get("top_n", 0)) * 1.5
    ]
    strict_legacy = [
        row for row in completed
        if row.get("turnover_mode", "strict_top_n") == "strict_top_n"
        and float(row.get("max_legacy_holding_weight") or 0.0) > 1e-12
    ]
    stale_full_exposure = [
        row for row in completed
        if row.get("exposure_bucket") == "full_1_00"
        and float(row.get("average_legacy_holding_count") or 0.0) > float(row.get("top_n", 0)) * 1.5
        and float(row.get("realized_average_gross_exposure") or 0.0) >= 0.9
    ]
    duplicates = _duplicate_result_groups(completed)
    buckets = sorted({str(row.get("realized_exposure_bucket")) for row in completed if row.get("realized_exposure_bucket")})
    warnings = {
        "cash_buffer_inactive_count": len(cash_inactive),
        "cash_buffer_inactive_examples": [row.get("strategy_id") for row in cash_inactive[:20]],
        "turnover_cap_inactive_count": len(turnover_inactive),
        "turnover_cap_inactive_examples": [row.get("strategy_id") for row in turnover_inactive[:20]],
        "underinvested_policy_count": len(underinvested),
        "most_underinvested_policies": [
            {
                "strategy_id": row.get("strategy_id"),
                "target_gross_exposure": row.get("target_gross_exposure"),
                "realized_average_gross_exposure": row.get("realized_average_gross_exposure"),
                "exposure_utilization_ratio": row.get("exposure_utilization_ratio"),
                "underinvestment_warning": row.get("underinvestment_warning"),
            }
            for row in underinvested[:20]
        ],
        "duplicate_result_group_count": len(duplicates),
        "duplicate_result_groups": duplicates[:20],
        "wide_actual_holdings_count": len(wide_books),
        "wide_actual_holdings_examples": [row.get("strategy_id") for row in wide_books[:20]],
        "strict_mode_legacy_holding_count": len(strict_legacy),
        "strict_mode_legacy_holding_examples": [row.get("strategy_id") for row in strict_legacy[:20]],
        "full_exposure_stale_holding_count": len(stale_full_exposure),
        "full_exposure_stale_holding_examples": [row.get("strategy_id") for row in stale_full_exposure[:20]],
        "exposure_bucket_coverage": buckets,
    }
    return {
        "effective_exposure_buckets": buckets,
        "duplicate_or_inactive_config_warnings": warnings,
        "cash_buffer_inactive_count": len(cash_inactive),
        "underinvested_policy_count": len(underinvested),
        "most_underinvested_policies": warnings["most_underinvested_policies"],
    }


def _duplicate_result_groups(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[str]] = {}
    for row in rows:
        key = (
            row.get("signal_column"),
            round(float(row.get("cost_adjusted_return") or 0.0), 12),
            round(float(row.get("max_drawdown") or 0.0), 12),
            round(float(row.get("average_turnover") or 0.0), 12),
            round(float(row.get("realized_average_gross_exposure") or 0.0), 12),
        )
        grouped.setdefault(key, []).append(str(row.get("strategy_id")))
    return [
        {"signature": list(key), "strategy_ids": values}
        for key, values in grouped.items()
        if len(values) > 1
    ]


def _best_under_drawdown(rows: list[Mapping[str, Any]], threshold: float) -> dict[str, Any] | None:
    ranked = rank_policies([row for row in rows if row.get("max_drawdown") is not None and float(row["max_drawdown"]) >= threshold])
    return ranked[0] if ranked else None


def _resolve_threshold(rows: list[Mapping[str, Any]], signal_column: str, value: Any) -> float:
    if isinstance(value, str) and value.startswith("top_") and value.endswith("_pct"):
        pct = float(value.removeprefix("top_").removesuffix("_pct")) / 100.0
        ordered = sorted(float(row[signal_column]) for row in rows)
        index = max(0, min(len(ordered) - 1, math.floor(len(ordered) * (1.0 - pct))))
        return ordered[index]
    return float(value)


def _apply_turnover_cap(previous: dict[str, float], desired: dict[str, float], cap: float, *, mode: str) -> dict[str, float]:
    if mode == "strict_top_n":
        stale_turnover = sum(abs(weight) for symbol, weight in previous.items() if symbol not in desired)
        target_previous = {symbol: previous.get(symbol, 0.0) for symbol in desired}
        target_turnover = sum(abs(desired.get(symbol, 0.0) - target_previous.get(symbol, 0.0)) for symbol in desired)
        remaining = max(0.0, cap - stale_turnover)
        scale = min(1.0, remaining / target_turnover) if target_turnover > 0.0 else 1.0
        return {
            symbol: target_previous.get(symbol, 0.0) + (desired[symbol] - target_previous.get(symbol, 0.0)) * scale
            for symbol in desired
            if abs(target_previous.get(symbol, 0.0) + (desired[symbol] - target_previous.get(symbol, 0.0)) * scale) > 1e-12
        }
    turnover = sum(abs(desired.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in set(desired) | set(previous))
    if turnover <= cap or turnover <= 0.0:
        return desired
    scale = cap / turnover
    symbols = set(desired) | set(previous)
    return {symbol: previous.get(symbol, 0.0) + (desired.get(symbol, 0.0) - previous.get(symbol, 0.0)) * scale for symbol in symbols if abs(previous.get(symbol, 0.0) + (desired.get(symbol, 0.0) - previous.get(symbol, 0.0)) * scale) > 1e-12}


def _max_drawdown(equity: list[float]) -> float:
    peak = equity[0] if equity else 1.0
    drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        drawdown = min(drawdown, value / peak - 1.0)
    return drawdown


def _annualized_return(final_equity: float, periods: int) -> float | None:
    if periods <= 0:
        return None
    return final_equity ** (26.0 / periods) - 1.0


def _sector_concentration(holdings: list[Mapping[str, Any]]) -> float | None:
    sectors: dict[str, float] = {}
    for row in holdings:
        sector = str(row.get("sector", "") or "")
        if not sector:
            continue
        sectors[sector] = sectors.get(sector, 0.0) + abs(float(row["weight"]))
    return max(sectors.values()) if sectors else None


def _markdown(payload: Mapping[str, Any]) -> str:
    best = payload.get("best_policy_summary") or {}
    top_lines = [
        f"- {row.get('strategy_id')}: bucket={row.get('exposure_bucket')}, max_drawdown={row.get('max_drawdown')}, cost_adjusted_sharpe={row.get('cost_adjusted_sharpe')}, cost_adjusted_return={row.get('cost_adjusted_return')}"
        for row in payload.get("top_20_ranked_policies", [])[:20]
    ]
    target_bucket_lines = [
        f"- {bucket}: {row.get('strategy_id')} cost_adjusted_sharpe={row.get('cost_adjusted_sharpe')}"
        for bucket, row in payload.get("best_policy_per_target_exposure_bucket", {}).items()
    ]
    realized_bucket_lines = [
        f"- {bucket}: {row.get('strategy_id')} cost_adjusted_sharpe={row.get('cost_adjusted_sharpe')}"
        for bucket, row in payload.get("best_policy_per_realized_exposure_bucket", {}).items()
    ]
    frontier = payload.get("drawdown_frontier", {})
    frontier_lines = [
        f"- {name}: {(row or {}).get('strategy_id')}"
        for name, row in frontier.items()
    ]
    return "\n".join(
        [
            "# Stock Alpha Ensemble Portfolio Policy Sweep",
            "",
            "Research only. Trading impact: none. Production validated: false.",
            "",
            f"- Experiment stage: {payload.get('experiment_stage')}",
            f"- Policy configs: {payload['policy_config_count']}",
            f"- Estimated policy count: {payload.get('estimated_policy_count')}",
            f"- Effective exposure buckets: {payload.get('effective_exposure_buckets')}",
            f"- Cash-buffer inactive count: {payload.get('cash_buffer_inactive_count')}",
            f"- Underinvested policy count: {payload.get('underinvested_policy_count')}",
            f"- Best signal: {best.get('signal_column')}",
            f"- Best top_n: {best.get('top_n')}",
            f"- Best max drawdown: {best.get('max_drawdown')}",
            f"- Best cost-adjusted Sharpe: {best.get('cost_adjusted_sharpe')}",
            f"- Best cost-adjusted return: {best.get('cost_adjusted_return')}",
            f"- Best target exposure bucket: {best.get('target_exposure_bucket')}",
            f"- Best realized exposure bucket: {best.get('realized_exposure_bucket')}",
            f"- Best exposure utilization: {best.get('exposure_utilization_ratio')}",
            f"- Turnover mode: {best.get('turnover_mode')}",
            f"- Best top-n violation ratio: {best.get('top_n_violation_ratio')}",
            f"- Cash-buffer inactive policy count: {payload.get('exposure_explanation', {}).get('cash_buffer_inactive_policy_count')}",
            "- Promotion thresholds changed: false",
            "",
            "Turnover modes: `strict_top_n` keeps final holdings inside the current top-n target list. `gradual_transition` lets legacy holdings persist while turnover limits are phased in.",
            "",
            "## Top 20 Ranked Policies",
            "",
            *top_lines,
            "",
            "## Best Policy Per Target Exposure Bucket",
            "",
            *target_bucket_lines,
            "",
            "## Best Policy Per Realized Exposure Bucket",
            "",
            *realized_bucket_lines,
            "",
            "## Drawdown Frontier",
            "",
            *frontier_lines,
            "",
        ]
    )
