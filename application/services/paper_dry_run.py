from __future__ import annotations

import json
from pathlib import Path

from core.research.ml.sector_reference import load_sector_by_symbol


def write_dry_run_rebalance_plan(config, result) -> tuple[Path, Path]:
    decision = result.decision
    sectors = load_sector_by_symbol(
        config.get("ml", {}).get("sector_reference_path"),
        config.get("ml", {}).get("sector_by_symbol", {}),
    )
    effective_weights = {
        symbol: weight * decision.exposure_target
        for symbol, weight in decision.target_weights.items()
    }
    sector_weights = {}
    for symbol, weight in effective_weights.items():
        sector = sectors.get(symbol, "UNMAPPED")
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
    risk = config.get("risk", {}).get("paper", {})
    max_weight = max(effective_weights.values(), default=0.0)
    max_sector = max(sector_weights.values(), default=0.0)
    turnover = sum(abs(order.dollar_delta) for order in decision.orders) / decision.equity if decision.equity else 0.0
    blockers = [check.reason for check in result.risk_checks if not check.passed]
    order_symbols = {order.symbol for order in decision.orders}
    unpriced_holdings = sorted(
        symbol for symbol, quantity in decision.current_positions.items()
        if quantity and symbol not in decision.target_weights and symbol not in order_symbols
    )
    if unpriced_holdings:
        blockers.append("unpriced_current_holdings:" + ",".join(unpriced_holdings))
    if max_weight > risk.get("max_position_weight", 0.25): blockers.append("max_single_name_weight_exceeded")
    if max_sector > risk.get("max_sector_weight", 0.60): blockers.append("max_sector_weight_exceeded")
    if turnover > risk.get("max_turnover", 0.50): blockers.append("max_turnover_exceeded")
    blockers = list(dict.fromkeys(blockers))
    context = decision.model_context or {}
    final_decision = "blocked" if blockers else "dry_run_only"
    payload = {
        "timestamp": decision.timestamp.isoformat(), "data_provider": config["backtest"].get("provider"),
        "broker_mode": config.get("broker", {}).get("adapter"), "paper_enabled": config.get("paper_trading", {}).get("enabled", False),
        "paper_submit_orders": False, "current_holdings": decision.current_positions,
        "target_holdings": effective_weights, "proposed_orders": [order.to_dict() for order in decision.orders],
        "rejected_orders": blockers, "cash_weight": 1 - decision.exposure_target,
        "unpriced_current_holdings": unpriced_holdings,
        "gross_exposure": sum(abs(value) for value in effective_weights.values()), "sector_weights": sector_weights,
        "max_sector_weight": max_sector, "max_single_name_weight": max_weight, "turnover": turnover,
        "benchmark_freshness": decision.data_freshness,
        "benchmark_available": context.get("benchmark_available", False),
        "max_pairwise_correlation": context.get("max_pairwise_correlation"),
        "correlation_limit": risk.get("max_pairwise_correlation", 0.90),
        "kill_switch_status": result.blocked_reason,
        "final_decision": final_decision,
    }
    directory = Path(config.get("paper_trading", {}).get("report_dir", "reports/paper"))
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "dry_run_rebalance_plan.json"
    markdown_path = directory / "dry_run_rebalance_plan.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(f"# Paper Dry-Run Rebalance Plan\n\nDecision: `{final_decision}`\n\nBlockers: {', '.join(blockers) or 'none'}\n", encoding="utf-8")
    return json_path, markdown_path
