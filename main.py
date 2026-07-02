from typing import Any

from application.cli import run_cli
from application.services.market_data_loader import latest_data_freshness
from application.services.paper_service import (
    paper_benchmark_metrics,
    paper_drift_rows,
)
from application.services.runtime_overrides import (
    apply_runtime_overrides,
    apply_fast_mode,
    limit_strategy_grids,
    limited_grid,
)


def dual_momentum_candidate_configs(dual_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Compatibility export that loads optional research code on demand."""
    from core.research.dual_momentum_experiments import (
        dual_momentum_candidate_configs as build_candidate_configs,
    )

    return build_candidate_configs(dual_config)


def dual_momentum_walk_forward_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compatibility export that loads optional research code on demand."""
    from core.research.dual_momentum_scoring import (
        dual_momentum_walk_forward_summary as build_walk_forward_summary,
    )

    return build_walk_forward_summary(results)


if __name__ == "__main__":
    run_cli()
