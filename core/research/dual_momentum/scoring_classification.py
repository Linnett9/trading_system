from core.research.dual_momentum.scoring_constraints import production_constraint_failures
from core.research.dual_momentum.scoring_helpers import _annual_return


def classify_walk_forward_fold_result(result):
    if result.result.total_return <= 0:
        return "rejected: return"

    if result.result.max_drawdown > 0.18:
        return "rejected: drawdown"

    if result.annualized_turnover_percent > 6.0:
        return "rejected: turnover"

    if result.result.sharpe < 1.0:
        return "rejected: sharpe"

    if result.excess_return <= 0:
        return "rejected: benchmark"

    if result.excess_vs_equal_weight <= 0:
        return "rejected: equal-weight"

    return "fold pass"
def _paper_candidate_floor(result):
    return (
        result.result.total_return >= 1.10
        and result.result.max_drawdown <= 0.20
        and result.result.sharpe >= 1.0
        and result.annualized_turnover_percent <= 10
    )
def classify_dual_momentum_result(result):
    if result.result.max_drawdown > 0.25:
        return "rejected: drawdown"

    if result.result.max_drawdown > 0.20:
        return "rejected: drawdown"

    if _annual_return(result, 2026) < -0.05:
        return "rejected: bad 2026"

    if result.annualized_turnover_percent > 10:
        return "rejected: turnover"

    if result.excess_return <= 0:
        return "rejected: benchmark"

    if result.excess_vs_equal_weight <= 0:
        return "rejected: equal-weight"

    production_failures = production_constraint_failures(result)
    if not production_failures:
        return "production candidate"

    if _paper_candidate_floor(result) and len(production_failures) == 1:
        return "near-production candidate"

    if _paper_candidate_floor(result):
        return "paper candidate"

    return "research candidate"
