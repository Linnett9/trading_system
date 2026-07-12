from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from core.research.ml.artifacts.artifact_writers import MLCoreArtifactWriter
from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.data import CsvRowRepository, JsonRepository
from core.research.framework.ranking import finite_number
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.immutable_runs import (
    deterministic_run_id,
    file_digest,
    preserve_immutable_run,
)
from core.research.ml.stock_level.stock_alpha_run_profile import apply_stock_alpha_run_profile
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_report_metadata

TARGET = "actual_forward_return_10d"
BENCHMARK_RETURN_COLUMNS = (
    "benchmark_return_next_period",
    "actual_benchmark_return_10d",
    "spy_forward_return_10d",
    "benchmark_forward_return_10d",
)
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


@dataclass(frozen=True)
class StockSelectorRebalanceDatasetPaths:
    dataset_path: Path
    metadata_path: Path
    row_count: int


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
    _preserve_stock_selector_replay_run(
        output,
        paths,
        payload,
        config=config,
        predictions_path=settings.oos_predictions_path,
        benchmark_path=settings.benchmark_path,
    )
    return paths


def _preserve_stock_selector_replay_run(
    output_dir: Path,
    paths: StockLevelPortfolioReplayPaths,
    payload: dict[str, Any],
    *,
    config: dict[str, Any],
    predictions_path: Path,
    benchmark_path: Path,
) -> None:
    identity = {
        "predictions_path": str(predictions_path),
        "predictions_sha256": file_digest(predictions_path),
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": file_digest(benchmark_path),
        "target_column": payload.get("target_column"),
        "signal_columns": payload.get("signal_columns"),
        "policies": payload.get("policies"),
        "summary": payload.get("summary"),
        "winners": payload.get("winners"),
        "config_hash": MLCoreArtifactWriter.hash_payload(config),
    }
    run_id = deterministic_run_id("stock_selector_replay", identity)
    preserve_immutable_run(
        output_dir=output_dir,
        run_id=run_id,
        kind="stock_selector_replay",
        identity=identity,
        artifact_paths=(
            paths.csv_path,
            paths.json_path,
            paths.markdown_path,
            paths.equity_curves_path,
            paths.holdings_path,
        ),
        extra_manifest={
            "best_ml_strategy_id": (
                (payload.get("winners") or {}).get("best_ml_model") or {}
            ).get("strategy_id"),
        },
    )


def write_stock_selector_rebalance_dataset(
    config: dict[str, Any],
) -> StockSelectorRebalanceDatasetPaths:
    settings = StockLevelResearchConfig.from_mapping(config)
    if not settings.stock_selector_rebalance_selected_signal:
        raise ValueError("ml.stock_selector_rebalance_selected_signal is required")
    if not settings.stock_selector_rebalance_selected_policy:
        raise ValueError("ml.stock_selector_rebalance_selected_policy is required")

    source_dir = settings.stock_selector_rebalance_source_dir
    predictions_path = Path(
        config.get("ml", {}).get(
            "stock_selector_rebalance_predictions_path",
            source_dir / "stock_level_model_oos_predictions.csv",
        )
    )
    summary_path = source_dir / "stock_level_portfolio_replay_summary.json"
    equity_path = source_dir / "stock_level_portfolio_replay_equity_curves.csv"
    holdings_path = source_dir / "stock_level_portfolio_replay_holdings.csv"

    rows, metadata = build_stock_selector_rebalance_dataset_from_artifacts(
        predictions_path=predictions_path,
        summary_path=summary_path,
        equity_curves_path=equity_path,
        holdings_path=holdings_path,
        selected_signal=settings.stock_selector_rebalance_selected_signal,
        selected_policy=settings.stock_selector_rebalance_selected_policy,
        outcome_horizon_days=settings.stock_selector_rebalance_outcome_horizon_days,
    )
    dataset_hash = _stable_dataset_hash(rows)
    for row in rows:
        row["dataset_hash"] = dataset_hash
    metadata.update({
        "dataset_hash": dataset_hash,
        "output_row_count": len(rows),
        "earliest_feature_date": rows[0]["feature_date"] if rows else None,
        "latest_feature_date": rows[-1]["feature_date"] if rows else None,
        "generated_at": _utc_timestamp(),
    })

    writer = ResearchArtifactWriter()
    writer.write_csv(
        settings.stock_selector_rebalance_dataset_path,
        rows,
        fieldnames=list(rows[0]) if rows else ["feature_id"],
    )
    writer.write_json(settings.stock_selector_rebalance_metadata_path, metadata)
    return StockSelectorRebalanceDatasetPaths(
        dataset_path=settings.stock_selector_rebalance_dataset_path,
        metadata_path=settings.stock_selector_rebalance_metadata_path,
        row_count=len(rows),
    )


def build_stock_selector_rebalance_dataset_from_artifacts(
    *,
    predictions_path: Path,
    summary_path: Path,
    equity_curves_path: Path,
    holdings_path: Path,
    selected_signal: str,
    selected_policy: str,
    outcome_horizon_days: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require_existing_artifacts(
        predictions_path,
        summary_path,
        equity_curves_path,
        holdings_path,
    )
    predictions = CsvRowRepository().read(predictions_path)
    summary = JsonRepository().read(summary_path)
    equity_rows = CsvRowRepository().read(equity_curves_path)
    holding_rows = CsvRowRepository().read(holdings_path)
    _require_columns(predictions_path, predictions, {"rebalance_date", "symbol"})
    _require_columns(equity_curves_path, equity_rows, {
        "rebalance_date",
        "strategy_id",
        "signal_column",
        "policy",
        "gross_return",
        "transaction_cost_drag",
        "net_return",
        "turnover",
        "equity",
        "benchmark_return",
    })
    _require_columns(holdings_path, holding_rows, {
        "rebalance_date",
        "strategy_id",
        "signal_column",
        "policy",
        "symbol",
        "weight",
    })

    available_signals = {
        str(row.get("signal_column", ""))
        for row in equity_rows
    } | set(summary.get("signal_columns", []))
    if selected_signal not in available_signals:
        raise ValueError(
            "Unknown stock selector signal for rebalance dataset: "
            f"{selected_signal}; available_signals={sorted(available_signals)}"
        )
    available_policies = {
        str(row.get("policy", ""))
        for row in equity_rows
    } | set(summary.get("policies", []))
    if selected_policy not in available_policies:
        raise ValueError(
            "Unknown stock selector portfolio policy for rebalance dataset: "
            f"{selected_policy}; available_policies={sorted(available_policies)}"
        )
    _require_columns(predictions_path, predictions, {"rebalance_date", "symbol", selected_signal})

    strategy_id = f"{selected_signal}|{selected_policy}"
    selected_equity = [
        row for row in equity_rows
        if str(row.get("signal_column")) == selected_signal
        and str(row.get("policy")) == selected_policy
        and str(row.get("strategy_id")) == strategy_id
    ]
    if not selected_equity:
        raise ValueError(
            "No equity curve rows match selected stock selector strategy: "
            f"signal={selected_signal}; policy={selected_policy}; strategy_id={strategy_id}"
        )
    _reject_duplicate_equity_rows(selected_equity)
    selected_holdings = [
        row for row in holding_rows
        if str(row.get("signal_column")) == selected_signal
        and str(row.get("policy")) == selected_policy
        and str(row.get("strategy_id")) == strategy_id
    ]
    if not selected_holdings:
        raise ValueError(
            "No holding rows match selected stock selector strategy: "
            f"signal={selected_signal}; policy={selected_policy}; strategy_id={strategy_id}"
        )

    holdings_by_date = _holdings_by_date(selected_holdings, holdings_path)
    equity_dates = {str(row["rebalance_date"]) for row in selected_equity}
    holding_dates = set(holdings_by_date)
    missing_holdings = sorted(equity_dates - holding_dates)
    unmatched_holdings = sorted(holding_dates - equity_dates)
    if missing_holdings:
        raise ValueError(
            "Missing holdings for selected stock selector rebalance dates: "
            f"{missing_holdings}; strategy_id={strategy_id}"
        )
    if unmatched_holdings:
        raise ValueError(
            "Holdings have no matching selected equity outcomes: "
            f"{unmatched_holdings}; strategy_id={strategy_id}"
        )
    rows = _selector_rebalance_rows(
        selected_equity,
        holdings_by_date,
        selected_signal=selected_signal,
        selected_policy=selected_policy,
        strategy_id=strategy_id,
        outcome_horizon_days=outcome_horizon_days,
        source_artifacts={
            "predictions": predictions_path,
            "summary": summary_path,
            "equity_curves": equity_curves_path,
            "holdings": holdings_path,
        },
    )
    metadata = {
        "mode": "stock_selector_rebalance_dataset",
        "contract_version": 1,
        "research_only": True,
        "trading_impact": "none",
        "training_performed": False,
        "selected_signal": selected_signal,
        "selected_policy": selected_policy,
        "strategy_id": strategy_id,
        "source_artifacts": {
            "predictions_path": str(predictions_path),
            "summary_path": str(summary_path),
            "equity_curves_path": str(equity_curves_path),
            "holdings_path": str(holdings_path),
        },
        "source_prediction_row_count": len(predictions),
        "source_equity_row_count": len(selected_equity),
        "source_holding_row_count": len(selected_holdings),
        "source_dataset_hash": _source_dataset_hash(predictions, summary),
        "input_contract": {
            "predictions_required_columns": ["rebalance_date", "symbol", selected_signal],
            "equity_curves_required_columns": [
                "rebalance_date",
                "strategy_id",
                "signal_column",
                "policy",
                "gross_return",
                "transaction_cost_drag",
                "net_return",
                "turnover",
                "equity",
                "benchmark_return",
            ],
            "holdings_required_columns": [
                "rebalance_date",
                "strategy_id",
                "signal_column",
                "policy",
                "symbol",
                "weight",
            ],
        },
    }
    return rows, metadata


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
        benchmark_return = _benchmark_return_for_group(group)
        drag = turnover * (cost_bps + slippage_bps) / 10_000.0
        net = gross - drag
        equity *= 1.0 + net
        key = f"{signal}|{policy}"
        periods.append({"rebalance_date": rebalance_date, "strategy_id": key, "signal_column": signal, "policy": policy, "gross_return": gross, "transaction_cost_drag": drag, "net_return": net, "turnover": turnover, "equity": equity, "benchmark_return": benchmark_return})
        for symbol, weight in sorted(weights.items()):
            holdings.append({"rebalance_date": rebalance_date, "strategy_id": key, "signal_column": signal, "policy": policy, "symbol": symbol, "weight": weight, "side": "long" if weight > 0 else "short"})
        previous = weights
    return periods, holdings


def _benchmark_return_for_group(rows: list[dict[str, Any]]) -> float:
    values: list[float] = []
    missing_symbols: list[str] = []
    timing_columns = (
        "benchmark_target_start_timestamp",
        "benchmark_label_start_timestamp",
        "benchmark_label_end_timestamp",
        "benchmark_label_available_timestamp",
    )
    timing_values: dict[str, set[str]] = {column: set() for column in timing_columns}
    for row in rows:
        found: float | None = None
        for column in BENCHMARK_RETURN_COLUMNS:
            value = finite_number(row.get(column))
            if value is not None:
                found = float(value)
                break
        if found is None:
            missing_symbols.append(str(row.get("symbol", "")))
        else:
            values.append(found)
        for column in timing_columns:
            value = str(row.get(column) or "").strip()
            if value:
                timing_values[column].add(value)
    if missing_symbols:
        raise ValueError(
            "Missing benchmark return for one or more symbols on a rebalance date: "
            f"rebalance_date={rows[0].get('rebalance_date') if rows else ''}; "
            f"symbols={sorted(missing_symbols)}"
        )
    unique = {round(value, 12) for value in values}
    if len(unique) > 1:
        raise ValueError("Benchmark return must be identical for all symbols on a rebalance date")
    conflicts = {
        column: sorted(values)
        for column, values in timing_values.items()
        if len(values) > 1
    }
    if conflicts:
        raise ValueError(
            "Benchmark target timestamps must be identical for all symbols on a "
            f"rebalance date; conflicts={conflicts}"
        )
    if not values:
        raise ValueError("Missing benchmark return for replay period")
    return values[0]


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


def _selector_rebalance_rows(
    equity_rows: list[dict[str, Any]],
    holdings_by_date: dict[str, list[dict[str, Any]]],
    *,
    selected_signal: str,
    selected_policy: str,
    strategy_id: str,
    outcome_horizon_days: int,
    source_artifacts: dict[str, Path],
) -> list[dict[str, Any]]:
    ordered = sorted(equity_rows, key=lambda row: str(row["rebalance_date"]))
    rows: list[dict[str, Any]] = []
    feature_ids: set[str] = set()
    peak_start_equity = 1.0
    previous_end_equity = 1.0
    previous_symbols: set[str] = set()
    for index, row in enumerate(ordered):
        feature_date = str(row["rebalance_date"])
        holdings = holdings_by_date.get(feature_date)
        if not holdings:
            raise ValueError(
                "Missing holdings for selected stock selector rebalance date: "
                f"{feature_date}; strategy_id={strategy_id}"
            )
        weights = _validated_weights(holdings, feature_date, strategy_id)
        symbols = sorted(weights)
        net_return = float(row["net_return"])
        gross_return = float(row["gross_return"])
        transaction_cost_drag = float(row["transaction_cost_drag"])
        turnover = float(row["turnover"])
        end_equity = float(row["equity"])
        benchmark_return = finite_number(row.get("benchmark_return"))
        if benchmark_return is None:
            raise ValueError(
                "Missing benchmark_return for stock selector replay outcome: "
                f"rebalance_date={feature_date}; strategy_id={strategy_id}"
            )
        benchmark_return = float(benchmark_return)
        champion_excess_return = net_return - benchmark_return
        start_equity = end_equity / (1.0 + net_return) if net_return > -1.0 else previous_end_equity
        peak_start_equity = max(peak_start_equity, start_equity)
        current_drawdown = (start_equity / peak_start_equity) - 1.0 if peak_start_equity else 0.0
        future_drawdown = min(0.0, (end_equity / max(start_equity, end_equity)) - 1.0) if start_equity > 0 else 0.0
        label_end_date = _label_end_date(ordered, index, feature_date, outcome_horizon_days)
        label_start_date = min(
            date.fromisoformat(feature_date) + timedelta(days=1),
            date.fromisoformat(label_end_date),
        ).isoformat()
        label_available_timestamp = _label_available_timestamp(
            ordered,
            label_end_date,
            outcome_horizon_days,
        )
        gross_exposure = sum(abs(value) for value in weights.values())
        net_exposure = sum(weights.values())
        normalized_abs = _normalized_abs_weights(weights)
        feature_id = _selector_feature_id(
            selected_signal,
            selected_policy,
            strategy_id,
            feature_date,
            label_end_date,
        )
        if feature_id in feature_ids:
            raise ValueError(f"Duplicate stock selector feature_id: {feature_id}")
        feature_ids.add(feature_id)
        enriched = {
            "feature_id": feature_id,
            "feature_date": feature_date,
            "rebalance_date": feature_date,
            "label_start_date": label_start_date,
            "label_end_date": label_end_date,
            "label_available_timestamp": label_available_timestamp,
            "outcome_end_date": label_end_date,
            "selector_signal": selected_signal,
            "portfolio_policy": selected_policy,
            "strategy_id": strategy_id,
            "selected_symbols": ",".join(symbols),
            "selected_weights": json.dumps(weights, sort_keys=True, separators=(",", ":")),
            "selection_count": len(symbols),
            "portfolio_gross_exposure": gross_exposure,
            "portfolio_net_exposure": net_exposure,
            "cash_weight": max(0.0, 1.0 - gross_exposure),
            "largest_weight": max((abs(value) for value in weights.values()), default=0.0),
            "selection_weight_herfindahl": sum(value * value for value in normalized_abs.values()),
            "selection_overlap_with_prior": (
                len(set(symbols) & previous_symbols) / len(previous_symbols)
                if previous_symbols else 0.0
            ),
            "replacements": len(set(symbols) - previous_symbols),
            "current_drawdown": current_drawdown,
            "recent_champion_return": (
                (start_equity / previous_end_equity) - 1.0
                if previous_end_equity > 0 and index > 0 else 0.0
            ),
            "turnover": turnover,
            "transaction_cost_drag": transaction_cost_drag,
            "portfolio_return_next_period": net_return,
            "portfolio_gross_return_next_period": gross_return,
            "champion_return_next_period": net_return,
            "benchmark_return_next_period": benchmark_return,
            "champion_excess_return": champion_excess_return,
            "forward_return_5d": None,
            "forward_return_10d": net_return,
            "future_volatility": abs(net_return - benchmark_return),
            "future_drawdown": future_drawdown,
            "future_max_drawdown": future_drawdown,
            "max_adverse_excursion": min(0.0, net_return),
            "max_favourable_excursion": max(0.0, net_return),
            "volatility_adjusted_excess_return": (
                champion_excess_return / max(abs(benchmark_return), 1e-9)
            ),
            "good_period": int(net_return > 0),
            "bad_period": int(net_return < -0.03),
            "underperforms_spy": int(champion_excess_return < 0.0),
            "drawdown_event": int(future_drawdown <= -0.10),
            "source_predictions_path": str(source_artifacts["predictions"]),
            "source_equity_curves_path": str(source_artifacts["equity_curves"]),
            "source_holdings_path": str(source_artifacts["holdings"]),
        }
        enriched["should_reduce_exposure"] = _should_reduce_exposure_label(enriched)
        rows.append(enriched)
        previous_symbols = set(symbols)
        previous_end_equity = end_equity
        peak_start_equity = max(peak_start_equity, end_equity)
    return rows


def _selector_feature_id(
    selected_signal: str,
    selected_policy: str,
    strategy_id: str,
    feature_date: str,
    label_end_date: str,
) -> str:
    payload = {
        "source_workflow": "stock_selector_rebalance_dataset",
        "selector_signal": selected_signal,
        "portfolio_policy": selected_policy,
        "strategy_id": strategy_id,
        "feature_date": feature_date,
        "label_end_date": label_end_date,
        "target": "should_reduce_exposure",
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"stock_selector_{digest}"


def _reject_duplicate_equity_rows(equity_rows: list[dict[str, Any]]) -> None:
    seen: set[tuple[str, str]] = set()
    for row in equity_rows:
        key = (str(row.get("strategy_id", "")), str(row.get("rebalance_date", "")))
        if key in seen:
            raise ValueError(
                "Duplicate equity curve outcome for stock selector strategy/date: "
                f"strategy_id={key[0]}; rebalance_date={key[1]}"
            )
        seen.add(key)


def _validated_weights(
    holdings: list[dict[str, Any]],
    rebalance_date: str,
    strategy_id: str,
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in holdings:
        symbol = str(item["symbol"]).upper()
        value = finite_number(item.get("weight"))
        if value is None:
            raise ValueError(
                "Invalid non-numeric stock selector holding weight: "
                f"rebalance_date={rebalance_date}; strategy_id={strategy_id}; symbol={symbol}"
            )
        weights[symbol] = float(value)
    gross = sum(abs(value) for value in weights.values())
    if gross > 1.000001:
        raise ValueError(
            "Stock selector holding gross exposure exceeds supported bounds: "
            f"rebalance_date={rebalance_date}; strategy_id={strategy_id}; gross_exposure={gross}"
        )
    return weights


def _holdings_by_date(
    holding_rows: list[dict[str, Any]],
    holdings_path: Path,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    for row in holding_rows:
        rebalance_date = str(row["rebalance_date"])
        symbol = str(row["symbol"]).upper()
        key = (rebalance_date, symbol)
        if key in seen:
            raise ValueError(
                "Duplicate holding for stock selector rebalance dataset: "
                f"rebalance_date={rebalance_date}; symbol={symbol}; path={holdings_path}"
            )
        seen.add(key)
        grouped.setdefault(rebalance_date, []).append({**row, "symbol": symbol})
    return {
        rebalance_date: sorted(rows, key=lambda item: str(item["symbol"]))
        for rebalance_date, rows in grouped.items()
    }


def _normalized_abs_weights(weights: dict[str, float]) -> dict[str, float]:
    absolute = {symbol: abs(weight) for symbol, weight in weights.items()}
    total = sum(absolute.values())
    return {symbol: weight / total for symbol, weight in absolute.items()} if total else {}


def _label_end_date(
    ordered_rows: list[dict[str, Any]],
    index: int,
    feature_date: str,
    horizon_days: int,
) -> str:
    if index + 1 < len(ordered_rows):
        next_date = str(ordered_rows[index + 1]["rebalance_date"])
        if next_date > feature_date:
            return next_date
    return (date.fromisoformat(feature_date) + timedelta(days=horizon_days)).isoformat()


def _label_available_timestamp(
    ordered_rows: list[dict[str, Any]],
    label_end_date: str,
    horizon_days: int,
) -> str:
    for row in ordered_rows:
        candidate = str(row["rebalance_date"])
        if candidate > label_end_date:
            return candidate
    return (date.fromisoformat(label_end_date) + timedelta(days=max(1, horizon_days))).isoformat()


def _require_existing_artifacts(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing stock selector rebalance source artifacts: "
            + ", ".join(missing)
        )


def _require_columns(path: Path, rows: list[dict[str, Any]], required: set[str]) -> None:
    if not rows:
        raise ValueError(f"Stock selector rebalance source artifact has no rows: {path}")
    available = set(rows[0])
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            "Stock selector rebalance source artifact is missing required columns: "
            f"path={path}; missing_columns={missing}; available_columns={sorted(available)}"
        )


def _source_dataset_hash(
    predictions: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    for row in predictions:
        for key in ("dataset_hash", "source_dataset_hash", "data_hash"):
            if row.get(key):
                return str(row[key])
    for key in ("dataset_hash", "source_dataset_hash", "data_hash"):
        if summary.get(key):
            return str(summary[key])
    return ""


def _should_reduce_exposure_label(
    row: dict[str, Any],
    drawdown_threshold: float = 0.08,
    excess_return_threshold: float = -0.01,
    volatility_adjusted_threshold: float = -0.10,
) -> int:
    return int(
        float(row["future_max_drawdown"]) <= -abs(drawdown_threshold)
        or float(row["champion_excess_return"]) <= excess_return_threshold
        or float(row["volatility_adjusted_excess_return"]) <= volatility_adjusted_threshold
    )


def _stable_dataset_hash(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
