from core.research.ml.meta.meta_ensemble import run_meta_ensemble


def run_ml_meta_ensemble(config):
    result = run_meta_ensemble(config)
    print("\nML META ENSEMBLE")
    print("mode=research | trading_impact=none")
    print(f"Output dir: {result.output_dir}")
    print(f"Meta dataset: {result.meta_dataset_path}")
    print(f"Meta audit: {result.audit_path}")
    print(f"Metrics: {result.metrics_path}")
    print(f"Leaderboard: {result.leaderboard_path}")
    print(f"Allocation v2: {result.allocation_policy_comparison_json_path}")
    print(f"Allocation v2 leaderboard: {result.allocation_policy_leaderboard_path}")
    print(f"Allocation v2 diagnostics: {result.allocation_policy_diagnostics_json_path}")
    print(f"Allocation v2 grid search: {result.allocation_policy_grid_search_json_path}")
    print(f"Meta auxiliary metrics: {result.meta_auxiliary_metrics_json_path}")
    print(f"Allocation optimizer: {result.allocation_optimizer_results_path}")
    print(
        "Selected optimizer exposure path: "
        f"{result.selected_optimizer_exposure_path_json}"
    )
    print(
        "Trading research leaderboard: "
        f"{result.trading_research_leaderboard_json_path}"
    )
