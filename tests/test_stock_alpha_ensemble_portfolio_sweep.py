from __future__ import annotations

import csv
import inspect
import json

import pytest

from config.config_loader import load_config
from core.research.ml.stock_level import stock_alpha_ensemble_portfolio_sweep
from core.research.ml.stock_level.stock_alpha_ensemble_portfolio_sweep import (
    build_ensemble_portfolio_policy_grid,
    build_ensemble_portfolio_policy_sweep,
    equal_weight_with_caps,
    exposure_bucket,
    rank_policies,
    replay_signal_portfolio,
    select_top_signal_rows,
    target_gross_exposure,
    write_stock_alpha_ensemble_portfolio_sweep,
)
from core.research.ml.stock_level.stock_alpha_ensemble import write_stock_alpha_ensemble


def test_replay_from_synthetic_predictions_produces_metrics():
    policies = [{"strategy_id": "p1", "signal_column": "signal", "top_n": 2, "max_position_weight": 0.5, "cash_buffer": 0.0, "turnover_cap": None, "minimum_signal_threshold": None}]
    summary, curves, holdings, trades, payload = build_ensemble_portfolio_policy_sweep(_rows(), policies=policies)

    assert len(summary) == 1
    assert summary[0]["date_count"] == 3
    assert summary[0]["symbol_count"] == 4
    assert summary[0]["cumulative_return"] != 0
    assert curves
    assert holdings
    assert trades
    assert payload["research_only"] is True
    assert payload["trading_impact"] == "none"
    assert payload["production_validated"] is False
    assert payload["promotion_thresholds_changed"] is False
    assert payload["top_20_ranked_policies"]
    assert payload["best_policy_per_signal"]["signal"]["signal_column"] == "signal"
    assert payload["best_policy_per_exposure_bucket"]
    assert payload["best_policy_per_signal_per_exposure_bucket"]["signal"]
    assert "max_drawdown_gte_-0.20" in payload["drawdown_frontier"]


def test_n_jobs_one_and_two_produce_identical_summaries_and_rankings():
    policies = _policy_grid_for_parallel_test()
    sequential, _, _, _, sequential_payload = build_ensemble_portfolio_policy_sweep(
        _rows(),
        policies=policies,
        collect_all_details=False,
        n_jobs=1,
    )
    parallel, _, _, _, parallel_payload = build_ensemble_portfolio_policy_sweep(
        _rows(),
        policies=policies,
        collect_all_details=False,
        n_jobs=2,
    )

    assert sequential == parallel
    assert sequential_payload["ranked_policies"] == parallel_payload["ranked_policies"]
    assert parallel_payload["parallelism"] == {"n_jobs": 2, "policy_level_parallelism": True}


def test_parallel_summary_order_preserves_policy_index():
    policies = list(reversed(_policy_grid_for_parallel_test()))
    summaries, _, _, _, _ = build_ensemble_portfolio_policy_sweep(
        _rows(),
        policies=policies,
        collect_all_details=False,
        n_jobs=2,
    )

    assert [row["policy_index"] for row in summaries] == sorted(row["policy_index"] for row in policies)


def test_target_gross_exposure_calculation_and_bucket_assignment():
    assert target_gross_exposure(5, 0.05, 0.0) == 0.25
    assert target_gross_exposure(20, 0.10, 0.05) == 0.95
    assert exposure_bucket(0.25) == "low_0_25"
    assert exposure_bucket(0.50) == "medium_0_50"
    assert exposure_bucket(0.75) == "high_0_75"
    assert exposure_bucket(0.95) == "full_1_00"


def test_top_n_selection_uses_signal_descending_with_symbol_tiebreak():
    selected = select_top_signal_rows(_rows()[:4], "signal", top_n=2)
    assert [row["symbol"] for row in selected] == ["AAA", "BBB"]


def test_weight_caps_and_cash_buffer_are_enforced():
    weights = equal_weight_with_caps(_rows()[:4], max_position_weight=0.2, cash_buffer=0.1)
    assert max(weights.values()) <= 0.2
    assert sum(weights.values()) <= 0.9


def test_realized_gross_exposure_and_cash_buffer_inactive_warning():
    policies = [{"strategy_id": "p1", "signal_column": "signal", "top_n": 2, "max_position_weight": 0.25, "cash_buffer": 0.1, "target_gross_exposure": 0.5, "exposure_bucket": "medium_0_50", "cash_buffer_inactive": True, "turnover_cap": None, "minimum_signal_threshold": None}]
    summary, _, _, _, payload = build_ensemble_portfolio_policy_sweep(_rows(), policies=policies)
    assert summary[0]["realized_average_gross_exposure"] == 0.5
    assert summary[0]["realized_max_gross_exposure"] == 0.5
    assert summary[0]["unused_cash_estimate"] == 0.5
    assert summary[0]["cash_buffer_inactive"] is True
    assert payload["exposure_explanation"]["cash_buffer_inactive_policy_count"] == 1


def test_realized_exposure_bucket_can_differ_from_target_when_underinvested():
    policies = [{"policy_index": 0, "strategy_id": "under", "signal_column": "signal", "top_n": 10, "max_position_weight": 0.10, "cash_buffer": 0.0, "target_gross_exposure": 1.0, "target_exposure_bucket": "full_1_00", "exposure_bucket": "full_1_00", "turnover_mode": "strict_top_n", "turnover_cap_initial_investment": False, "turnover_cap": 0.15, "minimum_signal_threshold": None}]
    summary, _, _, _, payload = build_ensemble_portfolio_policy_sweep(
        _rotating_rows(symbol_count=20, dates=3),
        policies=policies,
        collect_all_details=False,
        underinvestment_threshold=0.75,
    )

    row = summary[0]
    assert row["target_exposure_bucket"] == "full_1_00"
    assert row["realized_exposure_bucket"] != "full_1_00"
    assert row["underinvested_policy"] is True
    assert row["underinvestment_warning"]
    assert payload["underinvested_policy_count"] == 1
    assert payload["most_underinvested_policies"][0]["strategy_id"] == "under"


def test_best_policy_per_realized_exposure_bucket_reported():
    policies = [
        {"policy_index": 0, "strategy_id": "low", "signal_column": "signal", "top_n": 1, "max_position_weight": 0.20, "cash_buffer": 0.0, "target_gross_exposure": 0.2, "target_exposure_bucket": "low_0_25", "exposure_bucket": "low_0_25", "turnover_mode": "strict_top_n", "turnover_cap_initial_investment": True, "turnover_cap": None, "minimum_signal_threshold": None},
        {"policy_index": 1, "strategy_id": "full", "signal_column": "signal", "top_n": 4, "max_position_weight": 0.25, "cash_buffer": 0.0, "target_gross_exposure": 1.0, "target_exposure_bucket": "full_1_00", "exposure_bucket": "full_1_00", "turnover_mode": "strict_top_n", "turnover_cap_initial_investment": True, "turnover_cap": None, "minimum_signal_threshold": None},
    ]
    _, _, _, _, payload = build_ensemble_portfolio_policy_sweep(_rows(), policies=policies)
    assert "low_0_25" in payload["best_policy_per_realized_exposure_bucket"]
    assert "full_1_00" in payload["best_policy_per_realized_exposure_bucket"]
    assert payload["best_policy_per_signal_per_realized_exposure_bucket"]["signal"]


def test_initial_investment_option_allows_first_rebalance_to_reach_target_exposure():
    periods, _, _ = replay_signal_portfolio(
        _rotating_rows(symbol_count=20, dates=2),
        signal_column="signal",
        top_n=10,
        max_position_weight=0.10,
        cash_buffer=0.0,
        turnover_cap=0.15,
        turnover_mode="strict_top_n",
        turnover_cap_initial_investment=True,
    )
    capped_periods, _, _ = replay_signal_portfolio(
        _rotating_rows(symbol_count=20, dates=2),
        signal_column="signal",
        top_n=10,
        max_position_weight=0.10,
        cash_buffer=0.0,
        turnover_cap=0.15,
        turnover_mode="strict_top_n",
        turnover_cap_initial_investment=False,
    )

    assert periods[0]["gross_exposure"] == pytest.approx(1.0)
    assert capped_periods[0]["gross_exposure"] == pytest.approx(0.15)


def test_turnover_and_transaction_cost_are_calculated():
    periods, _, trades = replay_signal_portfolio(
        _rows(),
        signal_column="signal",
        top_n=2,
        max_position_weight=0.5,
        cash_buffer=0.0,
        cost_bps=10,
        slippage_bps=5,
    )

    assert periods[0]["turnover"] == 1.0
    assert periods[0]["estimated_transaction_cost"] == 0.0015
    assert sum(row["turnover_contribution"] for row in trades if row["rebalance_date"] == periods[0]["rebalance_date"]) == 1.0


def test_strict_top_n_turnover_cap_never_exceeds_top_n():
    periods, holdings, _ = replay_signal_portfolio(
        _rotating_rows(),
        signal_column="signal",
        top_n=10,
        max_position_weight=0.10,
        cash_buffer=0.0,
        turnover_cap=0.25,
        turnover_mode="strict_top_n",
    )

    assert max(row["holding_count"] for row in periods) <= 10
    assert max(row["legacy_holding_count"] for row in periods) == 0
    assert not any(row["top_n_violation"] for row in periods)
    assert not any(row["is_legacy_holding"] for row in holdings)


def test_strict_top_n_removes_stale_names_before_reallocation():
    periods, holdings, _ = replay_signal_portfolio(
        _rotating_rows(),
        signal_column="signal",
        top_n=3,
        max_position_weight=0.20,
        cash_buffer=0.0,
        turnover_cap=0.10,
        turnover_mode="strict_top_n",
    )

    second_date = periods[1]["rebalance_date"]
    second_symbols = {row["symbol"] for row in holdings if row["rebalance_date"] == second_date}
    assert second_symbols <= {"S03", "S04", "S05"}
    assert periods[1]["legacy_holding_count"] == 0


def test_gradual_transition_can_retain_legacy_holdings_and_reports_metrics():
    policies = [{"policy_index": 0, "strategy_id": "gradual", "signal_column": "signal", "top_n": 3, "max_position_weight": 0.20, "cash_buffer": 0.0, "turnover_mode": "gradual_transition", "turnover_cap": 0.10, "minimum_signal_threshold": None}]
    summary, periods, holdings, _, _ = build_ensemble_portfolio_policy_sweep(
        _rotating_rows(),
        policies=policies,
        collect_all_details=True,
    )

    assert max(row["legacy_holding_count"] for row in periods) > 0
    assert any(row["is_legacy_holding"] for row in holdings)
    assert summary[0]["max_legacy_holding_count"] > 0
    assert summary[0]["average_legacy_holding_weight"] > 0


def test_full_exposure_top10_strict_mode_does_not_become_wide_legacy_book():
    periods, _, _ = replay_signal_portfolio(
        _rotating_rows(symbol_count=40, dates=5),
        signal_column="signal",
        top_n=10,
        max_position_weight=0.10,
        cash_buffer=0.0,
        turnover_cap=0.25,
        turnover_mode="strict_top_n",
    )

    assert max(row["holding_count"] for row in periods) <= 10


def test_no_turnover_cap_behavior_is_unchanged():
    periods, _, _ = replay_signal_portfolio(
        _rotating_rows(),
        signal_column="signal",
        top_n=4,
        max_position_weight=0.25,
        cash_buffer=0.0,
        turnover_cap=None,
    )

    assert all(row["holding_count"] == 4 for row in periods)
    assert all(row["gross_exposure"] == 1.0 for row in periods)


def test_minimum_signal_threshold_filters_candidates():
    selected = select_top_signal_rows(_rows()[:4], "signal", top_n=4, minimum_signal_threshold=0.85)
    assert [row["symbol"] for row in selected] == ["AAA"]


def test_top_percentile_threshold_filters_candidates():
    selected = select_top_signal_rows(_rows()[:4], "signal", top_n=4, minimum_signal_threshold="top_50_pct")
    assert [row["symbol"] for row in selected] == ["AAA", "BBB"]


def test_drawdown_calculation_is_reported():
    policies = [{"strategy_id": "p1", "signal_column": "signal", "top_n": 1, "max_position_weight": 1.0, "cash_buffer": 0.0, "turnover_cap": None, "minimum_signal_threshold": None}]
    summary, _, _, _, _ = build_ensemble_portfolio_policy_sweep(_drawdown_rows(), policies=policies)
    assert summary[0]["max_drawdown"] < 0.0


def test_policy_ranking_prefers_lower_drawdown_when_returns_are_similar():
    risky = {"status": "completed", "strategy_id": "risky", "max_drawdown": -0.5, "cost_adjusted_sharpe": 1.0, "cost_adjusted_return": 0.2, "average_turnover": 0.1, "max_single_name_concentration": 0.5}
    controlled = {**risky, "strategy_id": "controlled", "max_drawdown": -0.1, "cost_adjusted_return": 0.19}
    assert rank_policies([risky, controlled])[0]["strategy_id"] == "controlled"


def test_best_policy_per_bucket_and_signal_bucket_are_reported():
    policies = [
        {"strategy_id": "low", "signal_column": "signal", "top_n": 1, "max_position_weight": 0.25, "cash_buffer": 0.0, "target_gross_exposure": 0.25, "exposure_bucket": "low_0_25", "cash_buffer_inactive": True, "turnover_cap": None, "minimum_signal_threshold": None},
        {"strategy_id": "medium", "signal_column": "signal", "top_n": 2, "max_position_weight": 0.25, "cash_buffer": 0.0, "target_gross_exposure": 0.5, "exposure_bucket": "medium_0_50", "cash_buffer_inactive": True, "turnover_cap": None, "minimum_signal_threshold": None},
    ]
    _, _, _, _, payload = build_ensemble_portfolio_policy_sweep(_rows(), policies=policies)
    assert payload["best_policy_per_exposure_bucket"]["low_0_25"]["strategy_id"] == "low"
    assert payload["best_policy_per_signal_per_exposure_bucket"]["signal"]["medium_0_50"]["strategy_id"] == "medium"


def test_missing_optional_sector_and_liquidity_columns_are_handled():
    rows = [{key: value for key, value in row.items() if key != "sector"} for row in _rows()]
    policies = [{"strategy_id": "p1", "signal_column": "signal", "top_n": 2, "max_position_weight": 0.5, "cash_buffer": 0.0, "turnover_cap": None, "minimum_signal_threshold": None}]
    summary, _, _, _, _ = build_ensemble_portfolio_policy_sweep(rows, policies=policies)
    assert summary[0]["max_sector_concentration"] is None


def test_writer_outputs_reports_and_best_policy(tmp_path):
    source = tmp_path / "predictions.csv"
    _write_rows(source, _rows())
    paths = write_stock_alpha_ensemble_portfolio_sweep(
        {
            "ml": {
                "stock_alpha_report_root": str(tmp_path / "reports"),
                "stock_alpha_run_size": "benchmark",
                "stock_alpha_portfolio_sweep_source_predictions_path": str(source),
                "stock_alpha_portfolio_sweep_signal_columns": ["signal", "missing"],
                "stock_alpha_portfolio_sweep_top_n_values": [2],
                "stock_alpha_portfolio_sweep_max_position_weights": [0.5],
                "stock_alpha_portfolio_sweep_cash_buffers": [0.0],
                "stock_alpha_portfolio_sweep_turnover_caps": [None],
                "stock_alpha_portfolio_sweep_minimum_signal_thresholds": [None],
                "stock_alpha_portfolio_sweep_top_policy_detail_count": 1,
            }
        }
    )
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert paths.csv_path.exists()
    assert paths.ranked_csv_path.exists()
    assert paths.best_equity_curve_path.exists()
    assert paths.equity_curves_path.name == "policy_sweep_top_policy_equity_curves.csv"
    assert paths.holdings_path.name == "policy_sweep_top_policy_holdings.csv"
    assert paths.trades_path.name == "policy_sweep_top_policy_trades.csv"
    assert payload["best_policy_summary"]["signal_column"] == "signal"
    assert payload["output_controls"]["write_all_holdings"] is False
    assert payload["output_controls"]["write_all_trades"] is False


def test_default_writer_does_not_write_all_holdings_or_trades(tmp_path):
    source = tmp_path / "predictions.csv"
    _write_rows(source, _rows())
    paths = write_stock_alpha_ensemble_portfolio_sweep(
        {
            "ml": {
                "stock_alpha_report_root": str(tmp_path / "reports"),
                "stock_alpha_run_size": "benchmark",
                "stock_alpha_portfolio_sweep_source_predictions_path": str(source),
                "stock_alpha_portfolio_sweep_signal_columns": ["signal"],
                "stock_alpha_portfolio_sweep_top_n_values": [1, 2],
                "stock_alpha_portfolio_sweep_max_position_weights": [0.5],
                "stock_alpha_portfolio_sweep_cash_buffers": [0.0],
                "stock_alpha_portfolio_sweep_turnover_caps": [None],
                "stock_alpha_portfolio_sweep_minimum_signal_thresholds": [None],
                "stock_alpha_portfolio_sweep_top_policy_detail_count": 1,
            }
        }
    )
    raw_rows = _read_csv(paths.csv_path)
    holding_rows = _read_csv(paths.holdings_path)
    trade_rows = _read_csv(paths.trades_path)
    assert len(raw_rows) == 2
    assert holding_rows
    assert trade_rows
    assert {row["strategy_id"] for row in holding_rows} == {json.loads(paths.json_path.read_text(encoding="utf-8"))["best_policy_summary"]["strategy_id"]}


def test_max_policy_configs_limits_run_size(tmp_path):
    source = tmp_path / "predictions.csv"
    _write_rows(source, _rows())
    paths = write_stock_alpha_ensemble_portfolio_sweep(
        {
            "ml": {
                "stock_alpha_report_root": str(tmp_path / "reports"),
                "stock_alpha_run_size": "benchmark",
                "stock_alpha_portfolio_sweep_source_predictions_path": str(source),
                "stock_alpha_portfolio_sweep_signal_columns": ["signal"],
                "stock_alpha_portfolio_sweep_top_n_values": [1, 2, 3],
                "stock_alpha_portfolio_sweep_max_position_weights": [0.25],
                "stock_alpha_portfolio_sweep_cash_buffers": [0.0],
                "stock_alpha_portfolio_sweep_turnover_caps": [None],
                "stock_alpha_portfolio_sweep_minimum_signal_thresholds": [None],
                "stock_alpha_portfolio_sweep_max_policy_configs": 2,
                "stock_alpha_portfolio_sweep_n_jobs": 2,
            }
        }
    )
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["policy_config_count"] == 2
    assert payload["parallelism"]["n_jobs"] == 2
    assert len(_read_csv(paths.csv_path)) == 2


def test_ranked_csv_is_sorted_like_payload(tmp_path):
    source = tmp_path / "predictions.csv"
    _write_rows(source, _rows())
    paths = write_stock_alpha_ensemble_portfolio_sweep(
        {
            "ml": {
                "stock_alpha_report_root": str(tmp_path / "reports"),
                "stock_alpha_run_size": "benchmark",
                "stock_alpha_portfolio_sweep_source_predictions_path": str(source),
                "stock_alpha_portfolio_sweep_signal_columns": ["signal"],
                "stock_alpha_portfolio_sweep_top_n_values": [1, 2],
                "stock_alpha_portfolio_sweep_max_position_weights": [0.25, 0.5],
                "stock_alpha_portfolio_sweep_cash_buffers": [0.0],
                "stock_alpha_portfolio_sweep_turnover_caps": [None],
                "stock_alpha_portfolio_sweep_minimum_signal_thresholds": [None],
            }
        }
    )
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    ranked_rows = _read_csv(paths.ranked_csv_path)
    assert ranked_rows[0]["strategy_id"] == payload["ranked_policies"][0]["strategy_id"]


def test_config_loads_and_grid_uses_available_signals():
    config = load_config("config/config.stock_alpha_portfolio_sweep_ensemble_benchmark_fast.yaml", overlay_project_config=True)
    rows = [{"stock_level_ensemble_average_rank_score": "0.1", "predicted_momentum_120d": "0.2"}]
    grid = build_ensemble_portfolio_policy_grid(config, rows)
    assert grid
    assert {row["signal_column"] for row in grid} == {"stock_level_ensemble_average_rank_score", "predicted_momentum_120d"}
    assert {row["exposure_bucket"] for row in grid} >= {"low_0_25", "medium_0_50", "high_0_75", "full_1_00"}
    assert config["ml"]["stock_alpha_portfolio_sweep_write_all_holdings"] is False
    assert config["ml"]["stock_alpha_portfolio_sweep_write_all_trades"] is False
    assert config["ml"]["stock_alpha_portfolio_sweep_write_top_policy_details"] is True
    assert config["ml"]["stock_alpha_portfolio_sweep_progress_every"] == 50
    assert config["ml"]["stock_alpha_portfolio_sweep_n_jobs"] == 4
    assert config["ml"]["stock_alpha_portfolio_sweep_turnover_mode"] == "strict_top_n"
    assert config["ml"]["stock_alpha_portfolio_sweep_turnover_cap_initial_investment"] is True
    assert config["ml"]["stock_alpha_portfolio_sweep_underinvestment_threshold"] == 0.75


def test_coarse_config_loads_with_expected_policy_count():
    config = load_config("config/config.stock_alpha_portfolio_sweep_ensemble_benchmark_fast_coarse.yaml", overlay_project_config=True)
    grid = build_ensemble_portfolio_policy_grid(config, [_all_signal_row()])
    assert config["ml"]["stock_alpha_portfolio_sweep_experiment_stage"] == "coarse"
    assert len(grid) == 48
    assert config["ml"]["stock_alpha_portfolio_sweep_progress_every"] == 10
    assert config["ml"]["stock_alpha_portfolio_sweep_write_all_holdings"] is False
    assert config["ml"]["stock_alpha_portfolio_sweep_write_all_trades"] is False


def test_refine_config_loads_with_winner_focused_grid():
    config = load_config("config/config.stock_alpha_portfolio_sweep_ensemble_benchmark_fast_refine.yaml", overlay_project_config=True)
    grid = build_ensemble_portfolio_policy_grid(config, [_all_signal_row()])
    assert config["ml"]["stock_alpha_portfolio_sweep_experiment_stage"] == "refine"
    assert config["ml"]["stock_alpha_portfolio_sweep_signal_columns"] == [
        "predicted_risk_adjusted_momentum",
        "predicted_momentum_120d",
        "stock_level_ensemble_trimmed_mean_rank_score",
        "stock_level_ensemble_average_rank_score",
    ]
    assert len(grid) == 256
    assert config["ml"]["stock_alpha_portfolio_sweep_turnover_mode"] == "strict_top_n"
    assert config["ml"]["stock_alpha_portfolio_sweep_n_jobs"] == 4
    assert config["ml"]["stock_alpha_portfolio_sweep_write_top_policy_details"] is True
    assert config["ml"]["stock_alpha_portfolio_sweep_top_policy_detail_count"] == 10
    assert config["ml"]["stock_alpha_portfolio_sweep_write_all_holdings"] is False
    assert config["ml"]["stock_alpha_portfolio_sweep_write_all_trades"] is False
    assert config["ml"]["stock_alpha_portfolio_sweep_write_all_equity_curves"] is False


def test_full_grid_config_still_loads_and_is_marked():
    config = load_config("config/config.stock_alpha_portfolio_sweep_ensemble_benchmark_fast_full_grid.yaml", overlay_project_config=True)
    grid = build_ensemble_portfolio_policy_grid(config, [_all_signal_row()])
    assert config["ml"]["stock_alpha_portfolio_sweep_experiment_stage"] == "full_grid"
    assert len(grid) == 1080


def test_staged_config_output_roots_are_isolated():
    coarse = load_config("config/config.stock_alpha_portfolio_sweep_ensemble_benchmark_fast_coarse.yaml", overlay_project_config=True)
    refine = load_config("config/config.stock_alpha_portfolio_sweep_ensemble_benchmark_fast_refine.yaml", overlay_project_config=True)
    full = load_config("config/config.stock_alpha_portfolio_sweep_ensemble_benchmark_fast_full_grid.yaml", overlay_project_config=True)
    roots = {
        coarse["ml"]["stock_alpha_report_root"],
        refine["ml"]["stock_alpha_report_root"],
        full["ml"]["stock_alpha_report_root"],
    }
    assert len(roots) == 3
    assert all("stock_alpha_portfolio_sweep_ensemble_benchmark_fast" in root for root in roots)


def test_stage_metadata_appears_in_json_and_markdown(tmp_path):
    source = tmp_path / "predictions.csv"
    _write_rows(source, _rows())
    paths = write_stock_alpha_ensemble_portfolio_sweep(
        {
            "ml": {
                "stock_alpha_report_root": str(tmp_path / "reports"),
                "stock_alpha_run_size": "benchmark",
                "stock_alpha_portfolio_sweep_experiment_stage": "coarse",
                "stock_alpha_portfolio_sweep_source_predictions_path": str(source),
                "stock_alpha_portfolio_sweep_signal_columns": ["signal"],
                "stock_alpha_portfolio_sweep_top_n_values": [1],
                "stock_alpha_portfolio_sweep_max_position_weights": [0.5],
                "stock_alpha_portfolio_sweep_cash_buffers": [0.0],
                "stock_alpha_portfolio_sweep_turnover_caps": [None],
                "stock_alpha_portfolio_sweep_minimum_signal_thresholds": [None],
            }
        }
    )
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    markdown = paths.markdown_path.read_text(encoding="utf-8")
    assert payload["experiment_stage"] == "coarse"
    assert payload["policy_grid_size"] == 1
    assert payload["estimated_policy_count"] == 1
    assert payload["effective_exposure_buckets"]
    assert "duplicate_or_inactive_config_warnings" in payload
    assert "- Experiment stage: coarse" in markdown


def test_parallel_worker_failures_surface_clear_errors():
    policies = [{"policy_index": 0, "strategy_id": "bad", "signal_column": "signal", "top_n": "bad", "max_position_weight": 0.5, "cash_buffer": 0.0, "turnover_cap": None, "minimum_signal_threshold": None}]
    with pytest.raises(RuntimeError, match="stock-alpha portfolio sweep worker failed"):
        build_ensemble_portfolio_policy_sweep(
            _rows(),
            policies=policies,
            collect_all_details=False,
            n_jobs=2,
        )


def test_full_enriched_configs_load_without_existing_future_files():
    ensemble = load_config("config/config.stock_alpha_ensemble_full_enriched.yaml", overlay_project_config=True)
    sweep = load_config("config/config.stock_alpha_portfolio_sweep_ensemble_full_enriched.yaml", overlay_project_config=True)
    assert ensemble["ml"]["stock_alpha_run_size"] == "full"
    assert sweep["ml"]["stock_alpha_run_size"] == "full"


def test_full_enriched_staged_configs_load_with_safe_defaults():
    coarse = load_config("config/config.stock_alpha_portfolio_sweep_ensemble_full_enriched_coarse.yaml", overlay_project_config=True)
    refine = load_config("config/config.stock_alpha_portfolio_sweep_ensemble_full_enriched_refine.yaml", overlay_project_config=True)
    full = load_config("config/config.stock_alpha_portfolio_sweep_ensemble_full_enriched_full_grid.yaml", overlay_project_config=True)

    configs = [coarse, refine, full]
    roots = {config["ml"]["stock_alpha_report_root"] for config in configs}
    source_paths = {config["ml"]["stock_alpha_portfolio_sweep_source_predictions_path"] for config in configs}

    assert len(roots) == 3
    assert len(source_paths) == 1
    assert next(iter(source_paths)) == "reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha_ensemble_full/full/ensemble/average_rank/stock_alpha_ensemble_average_rank_predictions.csv"
    assert [config["ml"]["stock_alpha_portfolio_sweep_experiment_stage"] for config in configs] == ["coarse", "refine", "full_grid"]

    for config in configs:
        ml = config["ml"]
        assert ml["stock_alpha_run_size"] == "full"
        assert ml["stock_alpha_portfolio_sweep_turnover_mode"] == "strict_top_n"
        assert ml["stock_alpha_portfolio_sweep_turnover_cap_initial_investment"] is True
        assert ml["stock_alpha_portfolio_sweep_underinvestment_threshold"] == 0.75
        assert ml["stock_alpha_portfolio_sweep_write_all_holdings"] is False
        assert ml["stock_alpha_portfolio_sweep_write_all_trades"] is False
        assert ml["stock_alpha_portfolio_sweep_write_top_policy_details"] is True
        assert ml["stock_alpha_portfolio_sweep_top_policy_detail_count"] == 10
        assert ml["stock_alpha_portfolio_sweep_n_jobs"] == 4
        assert ml["research_only"] is True
        assert ml["trading_impact"] == "none"
        assert ml["production_validated"] is False
        assert ml["promotion_thresholds_changed"] is False

    assert len(build_ensemble_portfolio_policy_grid(coarse, [_all_signal_row()])) == 72
    assert len(build_ensemble_portfolio_policy_grid(refine, [_all_signal_row()])) == 256
    assert len(build_ensemble_portfolio_policy_grid(full, [_all_signal_row()])) == 1080


def test_future_full_ensemble_execution_fails_clearly_if_source_missing(tmp_path):
    missing = tmp_path / "future_full.csv"
    try:
        write_stock_alpha_ensemble(
            {
                "ml": {
                    "stock_alpha_report_root": str(tmp_path / "reports"),
                    "stock_alpha_run_size": "full",
                    "stock_alpha_ensemble_source_predictions_path": str(missing),
                    "stock_alpha_ensemble_component_signal_columns": ["a", "b"],
                    "stock_alpha_ensemble_min_component_count": 2,
                }
            }
        )
    except ValueError as exc:
        assert "source predictions path does not exist" in str(exc)
    else:
        raise AssertionError("missing full ensemble source should fail clearly")


def test_future_full_portfolio_sweep_execution_fails_clearly_if_source_missing(tmp_path):
    missing = tmp_path / "future_full_ensemble.csv"
    try:
        write_stock_alpha_ensemble_portfolio_sweep(
            {
                "ml": {
                    "stock_alpha_report_root": str(tmp_path / "reports"),
                    "stock_alpha_run_size": "full",
                    "stock_alpha_portfolio_sweep_source_predictions_path": str(missing),
                    "stock_alpha_portfolio_sweep_signal_columns": ["signal"],
                }
            }
        )
    except ValueError as exc:
        assert "source predictions file not found" in str(exc)
    else:
        raise AssertionError("missing full portfolio source should fail clearly")


def test_no_broker_or_order_execution_imports_touched():
    source = inspect.getsource(stock_alpha_ensemble_portfolio_sweep)
    assert "infrastructure.broker" not in source
    assert "paper_trading" not in source
    assert "live_trading" not in source
    assert "core.entities.order" not in source
    assert "order_execution" not in source


def _rows():
    rows = []
    returns = {
        "2024-01-01": {"AAA": 0.03, "BBB": 0.02, "CCC": -0.01, "DDD": -0.02},
        "2024-01-08": {"AAA": -0.02, "BBB": 0.01, "CCC": 0.03, "DDD": -0.01},
        "2024-01-15": {"AAA": 0.01, "BBB": -0.02, "CCC": 0.02, "DDD": 0.03},
    }
    signals = {
        "2024-01-01": {"AAA": 0.9, "BBB": 0.8, "CCC": 0.2, "DDD": 0.1},
        "2024-01-08": {"AAA": 0.1, "BBB": 0.8, "CCC": 0.9, "DDD": 0.2},
        "2024-01-15": {"AAA": 0.6, "BBB": 0.1, "CCC": 0.8, "DDD": 0.9},
    }
    for date, values in signals.items():
        for symbol, signal in values.items():
            rows.append({"rebalance_date": date, "symbol": symbol, "signal": str(signal), "actual_forward_return_10d": str(returns[date][symbol]), "sector": "tech" if symbol in {"AAA", "BBB"} else "industrial"})
    return rows


def _all_signal_row():
    return {
        "stock_level_ensemble_average_rank_score": "0.1",
        "stock_level_ensemble_trimmed_mean_rank_score": "0.1",
        "predicted_momentum_120d": "0.1",
        "predicted_risk_adjusted_momentum": "0.1",
    }


def _policy_grid_for_parallel_test():
    return [
        {"policy_index": 0, "strategy_id": "p0", "signal_column": "signal", "top_n": 1, "max_position_weight": 0.5, "cash_buffer": 0.0, "target_gross_exposure": 0.5, "exposure_bucket": "medium_0_50", "cash_buffer_inactive": True, "turnover_cap": None, "minimum_signal_threshold": None},
        {"policy_index": 1, "strategy_id": "p1", "signal_column": "signal", "top_n": 2, "max_position_weight": 0.5, "cash_buffer": 0.0, "target_gross_exposure": 1.0, "exposure_bucket": "full_1_00", "cash_buffer_inactive": False, "turnover_cap": None, "minimum_signal_threshold": None},
        {"policy_index": 2, "strategy_id": "p2", "signal_column": "signal", "top_n": 2, "max_position_weight": 0.25, "cash_buffer": 0.1, "target_gross_exposure": 0.5, "exposure_bucket": "medium_0_50", "cash_buffer_inactive": True, "turnover_cap": 0.5, "minimum_signal_threshold": "top_50_pct"},
    ]


def _drawdown_rows():
    return [
        {"rebalance_date": "2024-01-01", "symbol": "AAA", "signal": "1.0", "actual_forward_return_10d": "0.1"},
        {"rebalance_date": "2024-01-08", "symbol": "AAA", "signal": "1.0", "actual_forward_return_10d": "-0.2"},
        {"rebalance_date": "2024-01-15", "symbol": "AAA", "signal": "1.0", "actual_forward_return_10d": "0.05"},
    ]


def _rotating_rows(symbol_count=12, dates=4):
    rows = []
    symbols = [f"S{index:02d}" for index in range(symbol_count)]
    for date_index in range(dates):
        date = f"2024-02-{date_index + 1:02d}"
        leaders = symbols[(date_index * 3) % symbol_count :] + symbols[: (date_index * 3) % symbol_count]
        ranks = {symbol: symbol_count - index for index, symbol in enumerate(leaders)}
        for symbol in symbols:
            rows.append(
                {
                    "rebalance_date": date,
                    "symbol": symbol,
                    "signal": str(ranks[symbol]),
                    "actual_forward_return_10d": "0.01",
                }
            )
    return rows


def _write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
