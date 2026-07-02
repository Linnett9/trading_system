from __future__ import annotations

from typing import Any

from core.research.ml.allocation.types import AllocationPolicyResult


def _drawdown_aware_objective(
    result: AllocationPolicyResult,
    champion: AllocationPolicyResult,
    config: dict[str, Any],
) -> float:
    weights = config.get("allocation_grid_objective", {})
    return (
        float(weights.get("total_return_weight", 1.0)) * result.total_return
        + float(weights.get("max_drawdown_improvement_weight", 0.5))
        * (champion.max_drawdown - result.max_drawdown)
        + float(weights.get("sharpe_weight", 0.25)) * result.sharpe
        - float(weights.get("turnover_penalty_weight", 0.1)) * result.turnover
        - float(weights.get("transaction_cost_penalty_weight", 1.0))
        * result.estimated_transaction_costs
    )
