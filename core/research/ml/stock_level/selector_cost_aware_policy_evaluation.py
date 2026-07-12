from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping

from core.research.framework.data import CsvRowRepository
from core.research.framework.ranking import finite_number
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.artifacts.artifact_writers import MLCoreArtifactWriter
from core.research.ml.immutable_runs import file_digest
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir
from core.research.ml.stock_level.stock_level_artifact_io import (
    artifact_identity,
    read_stock_level_artifact,
    write_stock_level_artifact,
)
from core.research.ml.stock_level.stock_level_portfolio_replay import (
    GUARDRAILS,
    TARGET,
    _benchmark_return_for_group,
    _metrics,
    _replay,
)


POLICY_CONTRACT_VERSION = "selector_cost_aware_policy_contract_v1"
EVALUATION_SCHEMA_VERSION = "selector_cost_aware_policy_evaluation_v1"
BASELINE_POLICY_ID = "exact_top_n"
RETURN_CALIBRATED_SEMANTICS = {
    "expected_return",
    "raw_expected_return",
    "return_calibrated",
    "cross_sectional_expected_return",
    "raw_return_10d",
}


@dataclass(frozen=True)
class SelectorCostAwarePolicyEvaluationPaths:
    output_dir: Path
    metrics_path: Path
    period_returns_path: Path
    holdings_path: Path
    trades_path: Path
    decisions_path: Path
    comparison_json_path: Path
    comparison_markdown_path: Path


def write_selector_cost_aware_policy_evaluation(config: Mapping[str, Any]) -> SelectorCostAwarePolicyEvaluationPaths:
    settings = _settings(config)
    if not settings["enabled"]:
        raise ValueError("ml.selector_cost_aware_policy_evaluation.enabled is false")
    source_path = Path(settings["prediction_artifact_path"])
    if not source_path.exists():
        raise FileNotFoundError(f"Selector cost-aware prediction artifact does not exist: {source_path}")
    rows = _read_prediction_artifact(source_path, allow_csv_fallback=bool(settings["allow_csv_fallback"]))
    payload = build_selector_cost_aware_policy_evaluation(
        rows,
        config=config,
        settings=settings,
        source_path=source_path,
    )
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = SelectorCostAwarePolicyEvaluationPaths(
        output_dir=output_dir,
        metrics_path=output_dir / "selector_cost_aware_policy_metrics.csv",
        period_returns_path=output_dir / "selector_cost_aware_policy_period_returns.csv",
        holdings_path=output_dir / "selector_cost_aware_policy_holdings.csv",
        trades_path=output_dir / "selector_cost_aware_policy_trades.csv",
        decisions_path=output_dir / "selector_cost_aware_policy_decisions.parquet",
        comparison_json_path=output_dir / "selector_cost_aware_policy_comparison.json",
        comparison_markdown_path=output_dir / "selector_cost_aware_policy_comparison.md",
    )
    writer = ResearchArtifactWriter()
    writer.write_csv(paths.metrics_path, payload["policy_metrics"], fieldnames=_fields(payload["policy_metrics"], ["policy_id"]))
    writer.write_csv(paths.period_returns_path, payload["period_returns"], fieldnames=_fields(payload["period_returns"], ["policy_id", "rebalance_date"]))
    writer.write_csv(paths.holdings_path, payload["holdings"], fieldnames=_fields(payload["holdings"], ["policy_id", "rebalance_date", "symbol"]))
    writer.write_csv(paths.trades_path, payload["trades"], fieldnames=_fields(payload["trades"], ["policy_id", "rebalance_date", "symbol"]))
    write_stock_level_artifact(
        paths.decisions_path,
        payload["decisions"],
        fieldnames=_decision_fields(payload["decisions"]),
        config={"ml": {"stock_level_artifact_format": "parquet", "stock_level_parquet_compression": "zstd"}},
    )
    writer.write_json(paths.comparison_json_path, payload)
    writer.write_markdown(paths.comparison_markdown_path, _markdown(payload))
    return paths


def build_selector_cost_aware_policy_evaluation(
    rows: list[dict[str, Any]],
    *,
    config: Mapping[str, Any],
    settings: Mapping[str, Any],
    source_path: Path | None,
) -> dict[str, Any]:
    policies = [_policy_contract(policy) for policy in settings["policies"]]
    _validate_policies(policies, settings)
    normalized = _normalize_rows(rows, settings)
    bounded = _bounded_rows(normalized, settings)
    if not bounded:
        raise ValueError("No strict-OOS prediction rows are available for cost-aware policy evaluation")
    baseline_policy = next(policy for policy in policies if policy["policy_id"] == BASELINE_POLICY_ID)
    signal = "selector_policy_signal"
    if _benchmark_available(bounded)["available"]:
        periods, holdings = _replay(
            bounded,
            signal,
            "long_only_top_n_equal_weight",
            int(baseline_policy["selection"]["target_holdings"]),
            float(settings["cost_bps"]),
            float(settings["slippage_bps"]),
            float(settings["max_position_weight"]),
            float(settings["min_position_weight"]),
        )
        baseline_periods = [_tag_period(row, baseline_policy) for row in periods]
        baseline_holdings = [_tag_holding(row, baseline_policy) for row in holdings]
    else:
        baseline = _replay_cost_aware_policy(bounded, baseline_policy, settings, [])
        baseline_periods = baseline["periods"]
        baseline_holdings = baseline["holdings"]
    baseline_decisions = _baseline_decisions(bounded, baseline_policy, baseline_holdings, settings)
    baseline_trades = _trades_from_holdings(baseline_holdings, baseline_policy)

    all_periods = list(baseline_periods)
    all_holdings = list(baseline_holdings)
    all_decisions = list(baseline_decisions)
    all_trades = list(baseline_trades)
    metrics = [_policy_metrics(baseline_policy, baseline_periods, baseline_holdings, [], baseline_periods)]

    for policy in policies:
        if policy["policy_id"] == BASELINE_POLICY_ID:
            continue
        result = _replay_cost_aware_policy(bounded, policy, settings, baseline_periods)
        all_periods.extend(result["periods"])
        all_holdings.extend(result["holdings"])
        all_decisions.extend(result["decisions"])
        all_trades.extend(result["trades"])
        metrics.append(_policy_metrics(policy, result["periods"], result["holdings"], result["decisions"], baseline_periods))

    comparison = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "mode": "selector_cost_aware_policy_evaluation_research_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_status": "BOUNDED DIAGNOSTIC ONLY / NOT POLICY PROMOTION EVIDENCE",
        "source_prediction_artifact_identity": _source_artifact_identity(source_path, rows),
        "candidate_identity": _candidate_identity(settings, bounded),
        "dataset_identity": _dataset_identity(bounded),
        "target_identity": _target_identity(bounded),
        "fold_plan_identity": _first_present(bounded, "fold_plan_identity"),
        "policy_contract_version": POLICY_CONTRACT_VERSION,
        "policy_contracts": policies,
        "policy_metrics": sorted(metrics, key=lambda row: row["policy_id"]),
        "period_returns": sorted(all_periods, key=lambda row: (row["policy_id"], row["rebalance_date"])),
        "holdings": sorted(all_holdings, key=lambda row: (row["policy_id"], row["rebalance_date"], row["symbol"])),
        "trades": sorted(all_trades, key=lambda row: (row["policy_id"], row["rebalance_date"], row["symbol"])),
        "decisions": sorted(all_decisions, key=lambda row: (row["policy_id"], row["rebalance_date"], row["symbol"])),
        "matched_comparison": _matched_comparison(metrics),
        "cost_model_identity": _hash({"cost_bps": settings["cost_bps"], "slippage_bps": settings["slippage_bps"], "source": "stock_level_portfolio_replay"}),
        "benchmark_identity": _benchmark_identity(bounded),
        "evaluation_date_range": {
            "first": min({row["rebalance_date"] for row in bounded}),
            "last": max({row["rebalance_date"] for row in bounded}),
            "decision_date_count": len({row["rebalance_date"] for row in bounded}),
        },
        "configuration_hash": _hash(settings),
        "code_commit": MLCoreArtifactWriter.git_commit(),
        "warnings": _warnings(rows, bounded, settings),
        "training_performed": False,
        "final_fit_performed": False,
        **GUARDRAILS,
    }
    return comparison


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    raw = dict(ml.get("selector_cost_aware_policy_evaluation", {}) or {})
    output_dir = stock_alpha_output_dir(config) / "selector_cost_aware_policy_evaluation"
    top_n = int(raw.get("top_n", ml.get("stock_portfolio_replay_top_n", 25)))
    return {
        "enabled": bool(raw.get("enabled", False)),
        "prediction_artifact_path": str(raw.get("prediction_artifact_path", raw.get("predictions_path", ml.get("selector_cost_aware_predictions_path", "")))),
        "candidate_id": raw.get("candidate_id"),
        "prediction_column": raw.get("prediction_column"),
        "prediction_semantics": str(raw.get("prediction_semantics", "rank_score")),
        "allow_csv_fallback": bool(raw.get("allow_csv_fallback", False)),
        "output_dir": str(raw.get("output_dir", output_dir)),
        "top_n": top_n,
        "cost_bps": float(raw.get("cost_bps", ml.get("stock_portfolio_replay_cost_bps", 10.0))),
        "slippage_bps": float(raw.get("slippage_bps", ml.get("stock_portfolio_replay_slippage_bps", 5.0))),
        "max_position_weight": float(raw.get("max_position_weight", ml.get("stock_portfolio_replay_max_position_weight", 0.05))),
        "min_position_weight": float(raw.get("min_position_weight", ml.get("stock_portfolio_replay_min_position_weight", 0.0))),
        "maximum_decision_dates": raw.get("maximum_decision_dates"),
        "maximum_symbols": raw.get("maximum_symbols"),
        "policies": list(raw.get("policies") or _default_policies(top_n)),
        "development_period": dict(raw.get("development_period", {}) or {}),
        "evaluation_period": dict(raw.get("evaluation_period", {}) or {}),
    }


def _default_policies(top_n: int) -> list[dict[str, Any]]:
    return [
        {
            "policy_id": BASELINE_POLICY_ID,
            "construction_mode": "exact_top_n",
            "selection": {"target_holdings": top_n, "entry_rank_max": top_n, "retention_rank_max": top_n},
            "trading": {"minimum_trade_weight": 0.0, "rebalance_fraction": 1.0},
            "edge_filter": {"enabled": False, "mode": "rank_only", "minimum_percentile_advantage": None, "cost_multiplier": 1.0},
            "retention": {"enabled": False, "existing_position_bonus": 0.0},
            "liquidity": {"enabled": False},
            "costs": {"reuse_replay_cost_model": True},
        },
        {
            "policy_id": "rank_hysteresis_min_trade",
            "construction_mode": "cost_aware",
            "selection": {"target_holdings": top_n, "entry_rank_max": top_n, "retention_rank_max": max(top_n, int(math.ceil(top_n * 1.5)))},
            "trading": {"minimum_trade_weight": 0.0025, "rebalance_fraction": 1.0},
            "edge_filter": {"enabled": False, "mode": "rank_only", "minimum_percentile_advantage": None, "cost_multiplier": 1.0},
            "retention": {"enabled": True, "existing_position_bonus": 0.0},
            "liquidity": {"enabled": False},
            "costs": {"reuse_replay_cost_model": True},
        },
        {
            "policy_id": "hysteresis_edge_partial",
            "construction_mode": "cost_aware",
            "selection": {"target_holdings": top_n, "entry_rank_max": top_n, "retention_rank_max": max(top_n, int(math.ceil(top_n * 1.5)))},
            "trading": {"minimum_trade_weight": 0.0025, "rebalance_fraction": 0.5},
            "edge_filter": {"enabled": True, "mode": "standardized_score", "minimum_percentile_advantage": 0.05, "minimum_score_advantage": None, "cost_multiplier": 1.5},
            "retention": {"enabled": True, "existing_position_bonus": 0.0},
            "liquidity": {"enabled": False},
            "costs": {"reuse_replay_cost_model": True},
        },
    ]


def _read_prediction_artifact(path: Path, *, allow_csv_fallback: bool) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        return read_stock_level_artifact(path, required_columns={"rebalance_date", "symbol"}, allow_csv_fallback=allow_csv_fallback)
    if path.suffix.lower() == ".csv" and allow_csv_fallback:
        return CsvRowRepository().read(path)
    return read_stock_level_artifact(path, required_columns={"rebalance_date", "symbol"}, allow_csv_fallback=allow_csv_fallback)


def _policy_contract(policy: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "contract_version": POLICY_CONTRACT_VERSION,
        "policy_id": str(policy["policy_id"]),
        "construction_mode": str(policy.get("construction_mode", "cost_aware")),
        "selection": dict(policy.get("selection", {}) or {}),
        "trading": dict(policy.get("trading", {}) or {}),
        "edge_filter": dict(policy.get("edge_filter", {}) or {}),
        "retention": dict(policy.get("retention", {}) or {}),
        "liquidity": dict(policy.get("liquidity", {}) or {}),
        "costs": dict(policy.get("costs", {}) or {}),
    }
    contract["selection"].setdefault("target_holdings", 25)
    contract["selection"].setdefault("entry_rank_max", contract["selection"]["target_holdings"])
    contract["selection"].setdefault("retention_rank_max", contract["selection"]["entry_rank_max"])
    contract["trading"].setdefault("minimum_trade_weight", 0.0)
    contract["trading"].setdefault("rebalance_fraction", 1.0)
    contract["edge_filter"].setdefault("enabled", False)
    contract["edge_filter"].setdefault("mode", "rank_only")
    contract["edge_filter"].setdefault("cost_multiplier", 1.0)
    contract["retention"].setdefault("enabled", contract["construction_mode"] == "cost_aware")
    contract["liquidity"].setdefault("enabled", False)
    contract["costs"].setdefault("reuse_replay_cost_model", True)
    contract["policy_identity"] = _hash(contract)
    return contract


def _validate_policies(policies: list[dict[str, Any]], settings: Mapping[str, Any]) -> None:
    if not any(policy["policy_id"] == BASELINE_POLICY_ID for policy in policies):
        raise ValueError("Cost-aware policy evaluation requires an exact_top_n baseline policy")
    ids = [policy["policy_id"] for policy in policies]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate policy_id in selector cost-aware evaluation: {ids}")
    semantics = str(settings["prediction_semantics"])
    for policy in policies:
        edge = policy["edge_filter"]
        if edge.get("mode") == "return_calibrated" and semantics not in RETURN_CALIBRATED_SEMANTICS:
            raise ValueError(
                "return_calibrated edge filtering requires return-compatible prediction_semantics; "
                f"prediction_semantics={semantics}; policy_id={policy['policy_id']}"
            )
        if policy["liquidity"].get("enabled"):
            reference = str(policy["liquidity"].get("reference_column", ""))
            if not reference:
                raise ValueError(f"Liquidity-aware thresholds require reference_column; policy_id={policy['policy_id']}")


def _normalize_rows(rows: list[dict[str, Any]], settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidate_id = settings.get("candidate_id")
    prediction_column = settings.get("prediction_column") or ("prediction" if rows and "prediction" in rows[0] else None)
    if not prediction_column:
        raise ValueError("selector_cost_aware_policy_evaluation.prediction_column is required for wide prediction artifacts")
    filtered = [row for row in rows if not candidate_id or str(row.get("candidate_id")) == str(candidate_id)]
    output: list[dict[str, Any]] = []
    for row in filtered:
        prediction = finite_number(row.get(prediction_column))
        actual = finite_number(row.get(TARGET))
        if actual is None:
            actual = finite_number(row.get("actual_investable_return_10d"))
        if prediction is None or actual is None or not str(row.get("rebalance_date", "")).strip() or not str(row.get("symbol", "")).strip():
            continue
        normalized = dict(row)
        normalized["symbol"] = str(row["symbol"]).upper()
        normalized["selector_policy_signal"] = float(prediction)
        normalized[TARGET] = float(actual)
        benchmark = finite_number(row.get("actual_benchmark_return_10d"))
        if benchmark is not None:
            normalized["actual_benchmark_return_10d"] = float(benchmark)
        else:
            normalized.pop("actual_benchmark_return_10d", None)
        normalized["fold_id"] = str(row.get("fold_id") or "strict_oos")
        output.append(normalized)
    _reject_duplicate_keys(output)
    return sorted(output, key=lambda row: (str(row["rebalance_date"]), str(row["symbol"])))


def _bounded_rows(rows: list[dict[str, Any]], settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    dates = sorted({str(row["rebalance_date"]) for row in rows})
    if settings.get("maximum_decision_dates"):
        dates = dates[: int(settings["maximum_decision_dates"])]
    symbols = sorted({str(row["symbol"]).upper() for row in rows})
    if settings.get("maximum_symbols"):
        symbols = symbols[: int(settings["maximum_symbols"])]
    return [row for row in rows if row["rebalance_date"] in dates and row["symbol"] in symbols]


def _replay_cost_aware_policy(
    rows: list[dict[str, Any]],
    policy: Mapping[str, Any],
    settings: Mapping[str, Any],
    baseline_periods: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_date: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_date.setdefault(str(row["rebalance_date"]), []).append(row)
    previous: dict[str, float] = {}
    equity = 1.0
    periods: list[dict[str, Any]] = []
    holdings: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    for rebalance_date, group in sorted(by_date.items()):
        ranked = _ranked(group)
        desired, selection_reasons, blocked = _cost_aware_desired_weights(ranked, previous, policy, settings)
        executed, trade_reasons = _executed_weights(previous, desired, policy, settings, ranked)
        turnover = sum(abs(executed.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in set(executed) | set(previous))
        gross = sum(executed.get(str(row["symbol"]), 0.0) * float(row[TARGET]) for row in group)
        drag = turnover * (float(settings["cost_bps"]) + float(settings["slippage_bps"])) / 10_000.0
        net = gross - drag
        equity *= 1.0 + net
        benchmark_return = _benchmark_return_for_group(group) if _benchmark_available(group)["available"] else None
        periods.append({
            "rebalance_date": rebalance_date,
            "strategy_id": f"selector_policy_signal|{policy['policy_id']}",
            "signal_column": "selector_policy_signal",
            "policy": policy["policy_id"],
            "policy_id": policy["policy_id"],
            "gross_return": gross,
            "transaction_cost_drag": drag,
            "net_return": net,
            "turnover": turnover,
            "equity": equity,
            "benchmark_return": benchmark_return,
        })
        for symbol, weight in sorted(executed.items()):
            if abs(weight) > 1e-12:
                holdings.append({
                    "rebalance_date": rebalance_date,
                    "strategy_id": f"selector_policy_signal|{policy['policy_id']}",
                    "signal_column": "selector_policy_signal",
                    "policy": policy["policy_id"],
                    "policy_id": policy["policy_id"],
                    "symbol": symbol,
                    "weight": weight,
                    "side": "long",
                })
        decisions.extend(_decision_rows(rebalance_date, ranked, previous, desired, executed, policy, settings, selection_reasons, trade_reasons, blocked))
        trades.extend(_trade_rows(rebalance_date, previous, executed, policy, settings, trade_reasons))
        previous = executed
    return {"periods": periods, "holdings": holdings, "decisions": decisions, "trades": trades}


def _ranked(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(group, key=lambda row: (-float(row["selector_policy_signal"]), str(row["symbol"])))
    count = len(ordered)
    output = []
    for index, row in enumerate(ordered, start=1):
        percentile = 1.0 - ((index - 1) / max(1, count - 1)) if count > 1 else 1.0
        output.append({**row, "rank": index, "score_percentile": percentile})
    return output


def _cost_aware_desired_weights(
    ranked: list[dict[str, Any]],
    previous: Mapping[str, float],
    policy: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, str], dict[str, dict[str, Any]]]:
    target = int(policy["selection"]["target_holdings"])
    entry_rank = int(policy["selection"]["entry_rank_max"])
    retention_rank = int(policy["selection"]["retention_rank_max"])
    by_symbol = {row["symbol"]: row for row in ranked}
    reasons: dict[str, str] = {}
    blocked: dict[str, dict[str, Any]] = {}
    retained = [
        symbol for symbol in previous
        if symbol in by_symbol and int(by_symbol[symbol]["rank"]) <= retention_rank
    ]
    retained = sorted(retained, key=lambda symbol: (int(by_symbol[symbol]["rank"]), symbol))[:target]
    selected = set(retained)
    for symbol in retained:
        reasons[symbol] = "retained_within_band"
    for row in ranked:
        symbol = row["symbol"]
        if len(selected) >= target:
            break
        if int(row["rank"]) <= entry_rank and symbol not in selected:
            selected.add(symbol)
            reasons[symbol] = "entered_top_rank"
    for row in ranked:
        symbol = row["symbol"]
        if symbol in selected or int(row["rank"]) > entry_rank:
            continue
        weakest = _weakest_selected(selected, by_symbol)
        if weakest is None:
            continue
        weakest_inside_retention = int(by_symbol[weakest]["rank"]) <= retention_rank
        if weakest_inside_retention and not policy["edge_filter"].get("enabled"):
            blocked[symbol] = {"decision_reason": "blocked_insufficient_edge", "estimated_edge": None, "edge_threshold": None, "replacement_candidate": weakest}
            continue
        edge = _edge(row, by_symbol[weakest], policy)
        estimated_cost = _estimated_replacement_cost(settings, target)
        threshold = _edge_threshold(policy, estimated_cost)
        if _edge_passes(policy, edge, threshold):
            selected.remove(weakest)
            selected.add(symbol)
            reasons[weakest] = "replaced_by_stronger_candidate"
            reasons[symbol] = "replaced_by_stronger_candidate"
        else:
            blocked[symbol] = {"decision_reason": "blocked_insufficient_edge", "estimated_edge": edge, "edge_threshold": threshold, "replacement_candidate": weakest}
    for symbol in previous:
        if symbol not in selected:
            reasons.setdefault(symbol, "exited_outside_retention_band" if symbol in by_symbol else "forced_exit_ineligible")
    selected_rows = [by_symbol[symbol] for symbol in selected if symbol in by_symbol]
    weight = min(float(settings["max_position_weight"]), 1.0 / len(selected_rows)) if selected_rows else 0.0
    if weight < float(settings["min_position_weight"]):
        return {}, reasons, blocked
    return {row["symbol"]: weight for row in sorted(selected_rows, key=lambda item: item["symbol"])}, reasons, blocked


def _executed_weights(
    previous: Mapping[str, float],
    desired: Mapping[str, float],
    policy: Mapping[str, Any],
    settings: Mapping[str, Any],
    ranked: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, str]]:
    fraction = float(policy["trading"]["rebalance_fraction"])
    minimum = float(policy["trading"]["minimum_trade_weight"])
    by_symbol = {row["symbol"]: row for row in ranked}
    executed: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for symbol in sorted(set(previous) | set(desired)):
        current = float(previous.get(symbol, 0.0))
        target = float(desired.get(symbol, 0.0))
        forced_exit = symbol not in by_symbol and current != 0.0
        proposed = 0.0 if forced_exit else current + fraction * (target - current)
        change = proposed - current
        if forced_exit:
            reasons[symbol] = "forced_exit_ineligible"
        elif abs(change) < minimum and abs(target - current) > 0.0:
            proposed = current
            reasons[symbol] = "blocked_below_trade_size"
        elif fraction < 1.0 and abs(target - current) > 0.0:
            reasons[symbol] = "partial_rebalance"
        elif abs(target - current) == 0.0:
            reasons[symbol] = "held_no_change"
        else:
            reasons[symbol] = "executed_rebalance"
        if abs(proposed) >= 1e-12:
            executed[symbol] = min(float(settings["max_position_weight"]), max(0.0, proposed))
    gross = sum(executed.values())
    if gross > 1.000001:
        scale = 1.0 / gross
        executed = {symbol: weight * scale for symbol, weight in executed.items()}
        for symbol in executed:
            reasons[symbol] = "forced_constraint_adjustment"
    return executed, reasons


def _weakest_selected(selected: set[str], by_symbol: Mapping[str, Mapping[str, Any]]) -> str | None:
    if not selected:
        return None
    return max(selected, key=lambda symbol: (int(by_symbol[symbol]["rank"]), symbol))


def _edge(candidate: Mapping[str, Any], held: Mapping[str, Any], policy: Mapping[str, Any]) -> float | None:
    mode = str(policy["edge_filter"].get("mode", "rank_only"))
    if not policy["edge_filter"].get("enabled"):
        return None
    if mode == "rank_only":
        return None
    if mode == "return_calibrated":
        return float(candidate["selector_policy_signal"]) - float(held["selector_policy_signal"])
    return float(candidate["score_percentile"]) - float(held["score_percentile"])


def _edge_threshold(policy: Mapping[str, Any], estimated_cost: float) -> float | None:
    edge = policy["edge_filter"]
    mode = str(edge.get("mode", "rank_only"))
    if not edge.get("enabled") or mode == "rank_only":
        return None
    if mode == "return_calibrated":
        return estimated_cost * float(edge.get("cost_multiplier", 1.0))
    return finite_number(edge.get("minimum_percentile_advantage")) or finite_number(edge.get("minimum_score_advantage")) or 0.0


def _edge_passes(policy: Mapping[str, Any], edge: float | None, threshold: float | None) -> bool:
    if not policy["edge_filter"].get("enabled") or str(policy["edge_filter"].get("mode")) == "rank_only":
        return True
    return edge is not None and threshold is not None and edge > threshold


def _estimated_replacement_cost(settings: Mapping[str, Any], target_holdings: int) -> float:
    weight = min(float(settings["max_position_weight"]), 1.0 / max(1, target_holdings))
    return 2.0 * weight * (float(settings["cost_bps"]) + float(settings["slippage_bps"])) / 10_000.0


def _decision_rows(
    rebalance_date: str,
    ranked: list[dict[str, Any]],
    previous: Mapping[str, float],
    desired: Mapping[str, float],
    executed: Mapping[str, float],
    policy: Mapping[str, Any],
    settings: Mapping[str, Any],
    selection_reasons: Mapping[str, str],
    trade_reasons: Mapping[str, str],
    blocked: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    desired_top = _frictionless_weights(ranked, int(policy["selection"]["target_holdings"]), settings)
    for row in ranked:
        symbol = row["symbol"]
        previous_weight = float(previous.get(symbol, 0.0))
        desired_weight = float(desired.get(symbol, 0.0))
        executed_weight = float(executed.get(symbol, 0.0))
        blocked_info = dict(blocked.get(symbol, {}) or {})
        trade_reason = trade_reasons.get(symbol)
        reason = blocked_info.get("decision_reason")
        if reason is None and trade_reason and (trade_reason.startswith("blocked") or trade_reason.startswith("forced") or trade_reason == "partial_rebalance"):
            reason = trade_reason
        if reason is None:
            reason = selection_reasons.get(symbol) or trade_reason or "held_no_change"
        rows.append({
            "candidate_id": row.get("candidate_id"),
            "policy_id": policy["policy_id"],
            "policy_identity": policy["policy_identity"],
            "decision_date": rebalance_date,
            "rebalance_date": rebalance_date,
            "symbol": symbol,
            "prediction": row["selector_policy_signal"],
            "rank": row["rank"],
            "previous_weight": previous_weight,
            "frictionless_desired_weight": desired_top.get(symbol, 0.0),
            "cost_aware_desired_weight": desired_weight,
            "executed_weight": executed_weight,
            "weight_change": executed_weight - previous_weight,
            "existing_holding": previous_weight > 0.0,
            "entry_eligible": int(row["rank"]) <= int(policy["selection"]["entry_rank_max"]),
            "retention_eligible": int(row["rank"]) <= int(policy["selection"]["retention_rank_max"]),
            "replacement_candidate": blocked_info.get("replacement_candidate"),
            "estimated_edge": blocked_info.get("estimated_edge"),
            "estimated_cost": _estimated_replacement_cost(settings, int(policy["selection"]["target_holdings"])),
            "edge_threshold": blocked_info.get("edge_threshold"),
            "minimum_trade_threshold": policy["trading"]["minimum_trade_weight"],
            "trade_allowed": reason not in {"blocked_insufficient_edge", "blocked_below_trade_size"},
            "trade_blocked": reason in {"blocked_insufficient_edge", "blocked_below_trade_size"},
            "trade_action": _trade_action(previous_weight, executed_weight),
            "decision_reason": reason,
            "forced_or_discretionary": "forced" if reason.startswith("forced") else "discretionary",
            "prediction_semantics": settings["prediction_semantics"],
            "edge_filter_mode": policy["edge_filter"].get("mode"),
        })
    return rows


def _baseline_decisions(rows: list[dict[str, Any]], policy: Mapping[str, Any], holdings: list[dict[str, Any]], settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_date = {date: _ranked([row for row in rows if row["rebalance_date"] == date]) for date in sorted({row["rebalance_date"] for row in rows})}
    holdings_by_date = {
        date: {row["symbol"]: float(row["weight"]) for row in holdings if row["rebalance_date"] == date}
        for date in by_date
    }
    decisions = []
    previous: dict[str, float] = {}
    for date, ranked in by_date.items():
        executed = holdings_by_date.get(date, {})
        desired = _frictionless_weights(ranked, int(policy["selection"]["target_holdings"]), settings)
        decisions.extend(_decision_rows(date, ranked, previous, desired, executed, policy, settings, {}, {}, {}))
        previous = executed
    return decisions


def _frictionless_weights(ranked: list[dict[str, Any]], top_n: int, settings: Mapping[str, Any]) -> dict[str, float]:
    selected = ranked[:top_n]
    weight = min(float(settings["max_position_weight"]), 1.0 / len(selected)) if selected else 0.0
    return {row["symbol"]: weight for row in selected if weight >= float(settings["min_position_weight"])}


def _trade_rows(rebalance_date: str, previous: Mapping[str, float], executed: Mapping[str, float], policy: Mapping[str, Any], settings: Mapping[str, Any], reasons: Mapping[str, str]) -> list[dict[str, Any]]:
    rows = []
    cost_rate = (float(settings["cost_bps"]) + float(settings["slippage_bps"])) / 10_000.0
    for symbol in sorted(set(previous) | set(executed)):
        before = float(previous.get(symbol, 0.0))
        after = float(executed.get(symbol, 0.0))
        change = after - before
        if abs(change) <= 1e-12:
            continue
        rows.append({
            "policy_id": policy["policy_id"],
            "rebalance_date": rebalance_date,
            "symbol": symbol,
            "previous_weight": before,
            "executed_weight": after,
            "executed_weight_change": change,
            "absolute_trade_weight": abs(change),
            "estimated_cost": abs(change) * cost_rate,
            "decision_reason": reasons.get(symbol, "executed_rebalance"),
            "forced_or_discretionary": "forced" if str(reasons.get(symbol, "")).startswith("forced") else "discretionary",
        })
    return rows


def _trades_from_holdings(holdings: list[dict[str, Any]], policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_date = sorted({row["rebalance_date"] for row in holdings})
    previous: dict[str, float] = {}
    trades = []
    for date in by_date:
        current = {row["symbol"]: float(row["weight"]) for row in holdings if row["rebalance_date"] == date}
        trades.extend(_trade_rows(date, previous, current, policy, {"cost_bps": 0.0, "slippage_bps": 0.0}, {}))
        previous = current
    return trades


def _tag_period(row: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(row), "policy": policy["policy_id"], "policy_id": policy["policy_id"], "strategy_id": f"selector_policy_signal|{policy['policy_id']}"}


def _tag_holding(row: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(row), "policy": policy["policy_id"], "policy_id": policy["policy_id"], "strategy_id": f"selector_policy_signal|{policy['policy_id']}"}


def _policy_metrics(
    policy: Mapping[str, Any],
    periods: list[dict[str, Any]],
    holdings: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    baseline_periods: list[dict[str, Any]],
) -> dict[str, Any]:
    base = _metrics("selector_policy_signal", policy["policy_id"], periods, holdings)
    net = [float(row["net_return"]) for row in periods]
    gross = [float(row["gross_return"]) for row in periods]
    annualization = _annualization([row["rebalance_date"] for row in periods])
    baseline_turnover = sum(float(row["turnover"]) for row in baseline_periods)
    baseline_costs = sum(float(row["transaction_cost_drag"]) for row in baseline_periods)
    turnover = sum(float(row["turnover"]) for row in periods)
    costs = sum(float(row["transaction_cost_drag"]) for row in periods)
    gross_baseline = sum(float(row["gross_return"]) for row in baseline_periods)
    by_date = {date: [row for row in holdings if row["rebalance_date"] == date] for date in {row["rebalance_date"] for row in periods}}
    trade_decisions = [row for row in decisions if abs(float(row.get("weight_change") or 0.0)) > 1e-12]
    return {
        "policy_id": policy["policy_id"],
        "policy_identity": policy["policy_identity"],
        "construction_mode": policy["construction_mode"],
        "gross_cumulative_return": math.prod(1.0 + value for value in gross) - 1.0 if gross else None,
        "net_cumulative_return": math.prod(1.0 + value for value in net) - 1.0 if net else None,
        "net_cagr": _cagr(net, annualization),
        "net_sharpe": _sharpe(net, annualization),
        "sortino": _sortino(net, annualization),
        "maximum_drawdown": _drawdown(net),
        "annualised_volatility": pstdev(net) * math.sqrt(annualization) if len(net) > 1 and annualization else 0.0,
        "annualised_turnover": mean([float(row["turnover"]) for row in periods]) * annualization if periods and annualization else None,
        "transaction_costs": costs,
        "cost_drag": costs,
        "number_of_trades": len(trade_decisions) if decisions else sum(1 for row in periods if float(row["turnover"]) > 0.0),
        "average_trade_size": mean([abs(float(row["weight_change"])) for row in trade_decisions]) if trade_decisions else None,
        "average_holding_period": None,
        "turnover_avoided_vs_baseline": baseline_turnover - turnover,
        "costs_avoided_vs_baseline": baseline_costs - costs,
        "gross_return_sacrificed_vs_baseline": gross_baseline - sum(gross),
        "net_return_gained_vs_baseline": sum(net) - sum(float(row["net_return"]) for row in baseline_periods),
        "tracking_difference_vs_frictionless_target": mean([abs(float(row.get("executed_weight") or 0.0) - float(row.get("frictionless_desired_weight") or 0.0)) for row in decisions]) if decisions else 0.0,
        "average_cash": mean([max(0.0, 1.0 - sum(abs(float(item["weight"])) for item in rows)) for rows in by_date.values()]) if by_date else 0.0,
        "average_holdings": base.get("average_number_of_positions"),
        "entries": sum(1 for row in decisions if row.get("decision_reason") == "entered_top_rank"),
        "exits": sum(1 for row in decisions if str(row.get("decision_reason", "")).startswith("exited")),
        "retentions": sum(1 for row in decisions if row.get("decision_reason") == "retained_within_band"),
        "replacements": sum(1 for row in decisions if row.get("decision_reason") == "replaced_by_stronger_candidate"),
        "no_trade_decisions": sum(1 for row in decisions if str(row.get("decision_reason", "")).startswith("blocked") or row.get("decision_reason") == "held_no_change"),
        "forced_trades": sum(1 for row in decisions if row.get("forced_or_discretionary") == "forced"),
        "discretionary_trades": sum(1 for row in decisions if row.get("forced_or_discretionary") == "discretionary" and abs(float(row.get("weight_change") or 0.0)) > 1e-12),
        "period_count": len(periods),
    }


def _matched_comparison(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next((row for row in metrics if row["policy_id"] == BASELINE_POLICY_ID), None)
    if not baseline:
        return []
    return [
        {
            "policy_id": row["policy_id"],
            "net_cumulative_return_delta_vs_baseline": _delta(row, baseline, "net_cumulative_return"),
            "turnover_avoided_vs_baseline": row["turnover_avoided_vs_baseline"],
            "costs_avoided_vs_baseline": row["costs_avoided_vs_baseline"],
            "number_of_trades_delta_vs_baseline": (row.get("number_of_trades") or 0) - (baseline.get("number_of_trades") or 0),
        }
        for row in metrics
        if row["policy_id"] != BASELINE_POLICY_ID
    ]


def _trade_action(previous_weight: float, executed_weight: float) -> str:
    if previous_weight == 0.0 and executed_weight > 0.0:
        return "enter"
    if previous_weight > 0.0 and executed_weight == 0.0:
        return "exit"
    if executed_weight > previous_weight:
        return "increase"
    if executed_weight < previous_weight:
        return "decrease"
    return "none"


def _source_artifact_identity(path: Path | None, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if path and path.exists() and path.suffix.lower() == ".parquet" and rows:
        return artifact_identity(path, rows=rows, fieldnames=list(rows[0]))
    return {"path": str(path) if path else None, "sha256": file_digest(path) if path and path.exists() else None, "row_count": len(rows)}


def _candidate_identity(settings: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_id": settings.get("candidate_id") or _first_present(rows, "candidate_id"),
        "prediction_column": settings.get("prediction_column") or "prediction",
        "prediction_semantics": settings["prediction_semantics"],
        "target_contract_identity": _first_present(rows, "target_contract_identity"),
    }


def _dataset_identity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "decision_date_count": len({row["rebalance_date"] for row in rows}),
        "symbol_count": len({row["symbol"] for row in rows}),
        "first_decision_date": min((row["rebalance_date"] for row in rows), default=None),
        "last_decision_date": max((row["rebalance_date"] for row in rows), default=None),
        "logical_content_hash": _hash([{k: row.get(k) for k in ("rebalance_date", "symbol", "fold_id", "selector_policy_signal", TARGET)} for row in rows]),
    }


def _target_identity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "target_column": TARGET,
        "selected_target_id": _first_present(rows, "target_id"),
        "target_contract_identity": _first_present(rows, "target_contract_identity"),
    }


def _benchmark_identity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    availability = _benchmark_available(rows)
    values = sorted({
        round(float(row["actual_benchmark_return_10d"]), 12)
        for row in rows
        if finite_number(row.get("actual_benchmark_return_10d")) is not None
    })
    return {
        "benchmark_return_column": "actual_benchmark_return_10d",
        "benchmark_non_null_count": availability["non_null_count"],
        "benchmark_missing_count": availability["missing_count"],
        "benchmark_date_coverage": availability["date_coverage"],
        "benchmark_relative_metrics_available": availability["available"],
        "unique_period_return_values": values[:10],
        "value_count": len(values),
    }


def _warnings(source_rows: list[dict[str, Any]], rows: list[dict[str, Any]], settings: Mapping[str, Any]) -> list[str]:
    warnings = ["BOUNDED DIAGNOSTIC ONLY", "NOT POLICY PROMOTION EVIDENCE"]
    if not _benchmark_available(rows)["available"]:
        warnings.append("benchmark_returns_unavailable; benchmark_relative_metrics_disabled")
    if settings.get("development_period") or settings.get("evaluation_period"):
        warnings.append("policy_development_and_evaluation_periods_disclosed; no automatic promotion")
    else:
        warnings.append("same_history_used_for_policy_diagnostic; do_not_promote")
    if any(policy.get("liquidity", {}).get("enabled") for policy in settings["policies"]):
        warnings.append("liquidity_thresholds_enabled_only_when configured point-in-time column is present")
    else:
        warnings.append("liquidity_sensitive_thresholds_disabled; no reliable configured point-in-time liquidity field")
    return warnings


def _benchmark_available(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dates = sorted({str(row.get("rebalance_date", "")) for row in rows})
    dates_with_benchmark = {
        str(row.get("rebalance_date", ""))
        for row in rows
        if finite_number(row.get("actual_benchmark_return_10d")) is not None
    }
    non_null = sum(1 for row in rows if finite_number(row.get("actual_benchmark_return_10d")) is not None)
    missing = len(rows) - non_null
    return {
        "available": bool(dates) and dates_with_benchmark == set(dates),
        "non_null_count": non_null,
        "missing_count": missing,
        "date_coverage": len(dates_with_benchmark) / len(dates) if dates else 0.0,
    }


def _reject_duplicate_keys(rows: list[dict[str, Any]]) -> None:
    keys = [(row["rebalance_date"], row["symbol"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Cost-aware policy evaluation rows must be unique by rebalance_date and symbol after candidate filtering")


def _annualization(dates: list[str]) -> float | None:
    if len(dates) < 2:
        return None
    from datetime import date

    parsed = [date.fromisoformat(str(value)[:10]) for value in dates]
    gaps = [(right - left).days for left, right in zip(parsed, parsed[1:]) if right > left]
    return 365.25 / mean(gaps) if gaps else None


def _cagr(values: list[float], annualization: float | None) -> float | None:
    if not values or not annualization:
        return None
    return math.prod(1.0 + value for value in values) ** (annualization / len(values)) - 1.0


def _sharpe(values: list[float], annualization: float | None) -> float | None:
    if len(values) < 2 or not annualization:
        return None
    vol = pstdev(values)
    return mean(values) / vol * math.sqrt(annualization) if vol > 0.0 else None


def _sortino(values: list[float], annualization: float | None) -> float | None:
    if not values or not annualization:
        return None
    downside = [min(0.0, value) for value in values]
    vol = math.sqrt(sum(value * value for value in downside) / len(downside))
    return mean(values) / vol * math.sqrt(annualization) if vol > 0.0 else None


def _drawdown(values: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for value in values:
        equity *= 1.0 + value
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1.0)
    return worst


def _delta(left: Mapping[str, Any], right: Mapping[str, Any], key: str) -> float | None:
    a = finite_number(left.get(key))
    b = finite_number(right.get(key))
    return a - b if a is not None and b is not None else None


def _first_present(rows: list[dict[str, Any]], column: str) -> str | None:
    return next((str(row.get(column)) for row in rows if str(row.get(column, "")).strip()), None)


def _hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _fields(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    return [*preferred, *[key for key in dict.fromkeys(key for row in rows for key in row) if key not in preferred]] if rows else preferred


def _decision_fields(rows: list[dict[str, Any]]) -> list[str]:
    return _fields(rows, [
        "candidate_id",
        "policy_id",
        "decision_date",
        "symbol",
        "prediction",
        "rank",
        "previous_weight",
        "frictionless_desired_weight",
        "cost_aware_desired_weight",
        "executed_weight",
        "weight_change",
        "existing_holding",
        "entry_eligible",
        "retention_eligible",
        "replacement_candidate",
        "estimated_edge",
        "estimated_cost",
        "edge_threshold",
        "minimum_trade_threshold",
        "trade_allowed",
        "trade_blocked",
        "trade_action",
        "decision_reason",
        "forced_or_discretionary",
    ])


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Selector Cost-Aware Policy Evaluation",
        "",
        "BOUNDED DIAGNOSTIC ONLY. NOT POLICY PROMOTION EVIDENCE.",
        "",
        f"- Candidate: `{payload['candidate_identity'].get('candidate_id')}`",
        f"- Decision dates: {payload['evaluation_date_range']['decision_date_count']}",
        f"- Policies compared: {', '.join(row['policy_id'] for row in payload['policy_contracts'])}",
        "",
        "| Policy | Net return | Turnover avoided | Costs avoided | Trades | No-trade decisions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["policy_metrics"]:
        lines.append(
            f"| {row['policy_id']} | {row['net_cumulative_return']} | {row['turnover_avoided_vs_baseline']} | {row['costs_avoided_vs_baseline']} | {row['number_of_trades']} | {row['no_trade_decisions']} |"
        )
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in payload.get("warnings", []))
    return "\n".join(lines) + "\n"
