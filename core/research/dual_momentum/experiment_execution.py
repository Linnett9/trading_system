from core.research.dual_momentum.factory import build_dual_momentum_tester
from core.research.dual_momentum.scoring import (
    dual_momentum_quality_score,
    paper_safe_dual_momentum_score,
    walk_forward_selection_score,
)
from core.research.dual_momentum.experiment_candidates import dual_momentum_candidate_configs

def run_dual_momentum_experiments(config, dual_config, candles_by_symbol):
    results = []

    for candidate_config in dual_momentum_candidate_configs(dual_config):
        tester = build_dual_momentum_tester(config, candidate_config)
        results.append(tester.run(candles_by_symbol))

    sorted_results = sorted(
        results,
        key=lambda result: (
            walk_forward_selection_score(result),
            paper_safe_dual_momentum_score(result),
            dual_momentum_quality_score(result),
            result.result.sharpe,
            result.calmar,
            -result.result.max_drawdown,
            -result.annualized_turnover_percent,
        ),
        reverse=True,
    )

    return sorted_results

