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
from core.research.dual_momentum.experiments import (
    dual_momentum_candidate_configs,
)
from core.research.dual_momentum.scoring import (
    dual_momentum_walk_forward_summary,
)


if __name__ == "__main__":
    run_cli()