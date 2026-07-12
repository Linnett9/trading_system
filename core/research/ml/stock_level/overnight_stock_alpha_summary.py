from __future__ import annotations

from pathlib import Path
from typing import Any

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.data import JsonRepository
from core.research.ml.stock_level.overnight_stock_alpha_types import RESEARCH_GUARDRAILS, SUMMARY_MODELS, METRIC_ALIASES
from core.research.ml.stock_level.overnight_stock_alpha_reporting import _path_payload


def _build_summary(
    *,
    config: dict[str, Any],
    base_output_dir: Path,
    output_dir: Path,
    settings: StockLevelResearchConfig,
    artifact_status: dict[str, Any],
    artifact_paths: Any,
    feature_paths: Any,
    baseline_paths: Any,
    enriched_paths: Any,
    target_paths: Any,
    portfolio_paths: Any,
    sweep_paths: Any,
    attribution_paths: Any,
    timings: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    stock_artifact = _read_optional_json(getattr(artifact_paths, "json_path", None))
    baseline = _read_optional_json(getattr(baseline_paths, "json_path", None)) or {}
    enriched = _read_optional_json(getattr(enriched_paths, "json_path", None)) or {}
    portfolio = JsonRepository().read(Path(portfolio_paths.json_path)) if portfolio_paths else {}
    sweep = JsonRepository().read(Path(sweep_paths.json_path)) if sweep_paths else {}
    baseline_rows = _rows_by_name(baseline)
    enriched_rows = _rows_by_name(enriched)

    comparisons = {
        "original_baseline_artifact_benchmark": _named_metrics(
            "original_baseline_artifact_benchmark",
            _best_row(baseline_rows),
        ),
        "enriched_feature_benchmark": _named_metrics(
            "enriched_feature_benchmark",
            _best_row(enriched_rows),
        ),
    }
    for name in SUMMARY_MODELS[2:]:
        comparisons[name] = _named_metrics(name, enriched_rows.get(name, {}))
    selector_grid = _daily_selector_decision_grid(settings)
    regeneration_gate = _twelve_worker_regeneration_gate(
        config=config,
        settings=settings,
        stock_artifact=stock_artifact,
    )

    return {
        "mode": "overnight_stock_alpha_experiment_research_only",
        "base_output_dir": str(base_output_dir),
        "output_dir": str(output_dir),
        "summary_models": list(SUMMARY_MODELS),
        "metrics": list(METRIC_ALIASES),
        "comparisons": comparisons,
        "winners": {
            "best_by_spearman": _winner(comparisons, "spearman_ic"),
            "best_by_spread": _winner(comparisons, "top_minus_bottom_spread"),
            "best_by_sharpe": _winner(comparisons, "spread_sharpe"),
            "best_by_risk_adjusted_spread": _winner(
                comparisons, "risk_adjusted_spread"
            ),
            "did_enriched_features_help": _did_enriched_help(comparisons),
        },
        "artifacts": {
            "stock_artifact": _path_payload(artifact_paths),
            "alpha_features": _path_payload(feature_paths),
            "baseline_benchmark": _path_payload(baseline_paths),
            "enriched_benchmark": _path_payload(enriched_paths),
            "target_comparison": _path_payload(target_paths),
            "portfolio_replay": _path_payload(portfolio_paths),
            "portfolio_policy_sweep": _path_payload(sweep_paths),
            "attribution": _path_payload(attribution_paths),
        },
        "artifact_status": artifact_status,
        "stage_timings": timings,
        "run_size": settings.run_size,
        "effective_row_count": enriched.get("effective_row_count", enriched.get("eligible_row_count")),
        "effective_date_count": enriched.get("effective_date_count", enriched.get("input_date_count")),
        "effective_symbol_count": enriched.get("effective_symbol_count", enriched.get("input_symbol_count")),
        "portfolio_replay": _portfolio_summary(portfolio),
        "portfolio_policy_sweep": _portfolio_sweep_summary(sweep),
        "daily_selector_decision_grid": selector_grid,
        "twelve_worker_regeneration_gate": regeneration_gate,
        "parallelism": _parallelism_payload(settings, baseline, enriched, stock_artifact),
        **RESEARCH_GUARDRAILS,
    }

def _daily_selector_decision_grid(
    settings: StockLevelResearchConfig,
) -> list[dict[str, Any]]:
    horizon = settings.stock_selector_rebalance_outcome_horizon_days
    return [
        {
            "layer": "daily_selector",
            "decision": "rank_stock_ownership_candidates",
            "cadence": "each_eligible_trading_day",
            "output": "stock_rankings_and_oos_selector_predictions",
            "target": settings.target_column,
            "intended_holding_period_trading_days": horizon,
            "historical_evaluation_policy": "strict_oos_only",
            "final_fitted_selector_used_for_historical_evaluation": False,
        },
        {
            "layer": "daily_portfolio_construction",
            "decision": "convert_selector_rankings_to_target_weights",
            "cadence": "each_eligible_trading_day",
            "output": "target_portfolio_weights",
            "target": "portfolio_weight",
            "intended_holding_period_trading_days": horizon,
            "historical_evaluation_policy": "matched_selector_and_exposure_comparison",
            "final_fitted_selector_used_for_historical_evaluation": False,
        },
        {
            "layer": "intraday_exposure_control",
            "decision": "adjust_portfolio_risk",
            "cadence": "later_work",
            "output": "risk_exposure_multiplier",
            "target": "exposure",
            "intended_holding_period_trading_days": None,
            "historical_evaluation_policy": "out_of_scope_for_ticket_7b3",
            "final_fitted_selector_used_for_historical_evaluation": False,
        },
        {
            "layer": "five_minute_execution",
            "decision": "enter_wait_add_reduce_or_exit",
            "cadence": "later_work_ticket_10",
            "output": "execution_action",
            "target": "execution_timing",
            "intended_holding_period_trading_days": None,
            "historical_evaluation_policy": "out_of_scope_for_ticket_7b3",
            "final_fitted_selector_used_for_historical_evaluation": False,
        },
    ]

def _twelve_worker_regeneration_gate(
    *,
    config: dict[str, Any],
    settings: StockLevelResearchConfig,
    stock_artifact: dict[str, Any] | None,
) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    stages = dict(ml.get("stock_alpha_stages", {}) or {})
    dataset_parallelism = dict((stock_artifact or {}).get("dataset_parallelism", {}) or {})
    decision_grid = dict((stock_artifact or {}).get("decision_grid", {}) or {})
    canonical_artifact = dict((stock_artifact or {}).get("canonical_artifact", {}) or {})
    disabled_stages = (
        "baseline_benchmark",
        "enriched_benchmark",
        "target_comparison",
        "portfolio_replay",
        "portfolio_policy_sweep",
        "experiment_report",
        "attribution",
    )
    checks = [
        _gate_check("profile_is_ticket_7b3_daily", "ticket_7b3_daily" in settings.run_profile, settings.run_profile, "contains ticket_7b3_daily"),
        _gate_check("artifact_format_is_parquet", settings.artifact_format == "parquet", settings.artifact_format, "parquet"),
        _gate_check("parquet_compression_is_zstd", settings.parquet_compression == "zstd", settings.parquet_compression, "zstd"),
        _gate_check("configured_decision_frequency_is_daily", str(ml.get("stock_level_decision_frequency", "")).lower() == "daily", ml.get("stock_level_decision_frequency"), "daily"),
        _gate_check("dataset_inner_threads_is_one", settings.dataset_inner_threads == 1, settings.dataset_inner_threads, 1),
        _gate_check("force_refresh_enabled", settings.force_refresh is True, settings.force_refresh, True),
        _gate_check("resume_existing_outputs_disabled", settings.resume_existing_outputs is False, settings.resume_existing_outputs, False),
        _gate_check("legacy_output_paths_disabled", bool(ml.get("stock_alpha_allow_legacy_output_paths", False)) is False, bool(ml.get("stock_alpha_allow_legacy_output_paths", False)), False),
        _gate_check(
            "only_regeneration_stages_enabled",
            stages.get("stock_artifact") is True
            and stages.get("alpha_features") is True
            and all(stages.get(stage) is False for stage in disabled_stages),
            {stage: stages.get(stage) for stage in ("stock_artifact", "alpha_features", *disabled_stages)},
            "stock_artifact and alpha_features only",
        ),
    ]
    if decision_grid:
        checks.extend(
            [
                _gate_check("artifact_decision_frequency_is_daily", decision_grid.get("decision_frequency") == "daily", decision_grid.get("decision_frequency"), "daily"),
                _gate_check("artifact_decision_date_count_positive", int(decision_grid.get("decision_date_count") or 0) > 0, decision_grid.get("decision_date_count"), "> 0"),
                _gate_check("artifact_calendar_identity_recorded", bool(decision_grid.get("exchange_calendar_identity")), decision_grid.get("exchange_calendar_identity"), "non-empty"),
                _gate_check("artifact_daily_grid_identity_recorded", bool(decision_grid.get("decision_grid_identity")), decision_grid.get("decision_grid_identity"), "non-empty"),
                _gate_check("artifact_completion_status_complete", canonical_artifact.get("completion_status") == "complete", canonical_artifact.get("completion_status"), "complete"),
            ]
        )
    if dataset_parallelism:
        checks.extend(
            [
                _gate_check("artifact_worker_request_recorded", dataset_parallelism.get("requested_workers") == settings.dataset_workers, dataset_parallelism.get("requested_workers"), settings.dataset_workers),
                _gate_check("artifact_effective_worker_count_positive", int(dataset_parallelism.get("effective_workers") or 0) > 0, dataset_parallelism.get("effective_workers"), "> 0"),
                _gate_check("artifact_worker_inner_thread_cap_recorded", dataset_parallelism.get("inner_thread_limit") == 1, dataset_parallelism.get("inner_thread_limit"), 1),
                _gate_check(
                    "artifact_worker_tasks_completed",
                    dataset_parallelism.get("completed_task_count") == dataset_parallelism.get("task_count"),
                    {
                        "completed_task_count": dataset_parallelism.get("completed_task_count"),
                        "task_count": dataset_parallelism.get("task_count"),
                    },
                    "completed_task_count == task_count",
                ),
                _gate_check("artifact_worker_tasks_not_failed", dataset_parallelism.get("failed_task_count") == 0, dataset_parallelism.get("failed_task_count"), 0),
            ]
        )
    passed = all(check["passed"] for check in checks)
    return {
        "gate": "ticket_7b3_twelve_worker_regeneration",
        "passed": passed,
        "status": "passed" if passed else "failed",
        "scope": "stock_level_dataset_regeneration_only",
        "selector_training_invoked": False,
        "news_backfill_invoked": False,
        "portfolio_replay_invoked": False,
        "checks": checks,
    }

def _gate_check(
    name: str,
    passed: bool,
    actual: Any,
    expected: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }

def _parallelism_payload(
    settings: StockLevelResearchConfig,
    baseline: dict[str, Any],
    enriched: dict[str, Any],
    stock_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    enriched_parallelism = dict(enriched.get("parallelism", {}) or {})
    baseline_parallelism = dict(baseline.get("parallelism", {}) or {})
    dataset_parallelism = dict((stock_artifact or {}).get("dataset_parallelism", {}) or {})
    requested_stage_workers = settings.overnight_stage_n_jobs
    return {
        "stock_level_dataset": dataset_parallelism,
        "stock_level_dataset_workers": settings.dataset_workers,
        "stock_level_dataset_inner_threads": settings.dataset_inner_threads,
        "stock_alpha_feature_n_jobs": settings.alpha_feature_n_jobs,
        "stock_ranker_model_n_jobs": settings.model_n_jobs,
        "sklearn_n_jobs": settings.sklearn_n_jobs,
        "effective_model_workers": enriched_parallelism.get(
            "effective_model_workers",
            baseline_parallelism.get(
                "effective_model_workers",
                settings.model_n_jobs,
            ),
        ),
        "stock_alpha_overnight_stage_n_jobs": requested_stage_workers,
        "effective_stage_workers": 1,
        "stages": "sequential",
        "stage_parallelism_enabled": False,
        "oversubscription_policy": (
            "Overnight stages remain sequential by default; stock-level dataset "
            "generation owns symbol-level dataset workers, alpha feature generation "
            "and each benchmark use their own bounded worker settings, and nested "
            "native thread pools are capped by the active stage."
        ),
    }


def _read_optional_json(path: Any) -> dict[str, Any] | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return JsonRepository().read(candidate)

def _portfolio_summary(payload: dict[str, Any]) -> dict[str, Any]:
    best = payload.get("winners", {}).get("best_by_net_return_after_costs") or {}
    return {
        "best_portfolio_signal": best.get("signal_column"),
        "best_portfolio_policy": best.get("policy"),
        "best_ml_vs_momentum_120d": payload.get("best_ml_vs_momentum_120d", {}),
        "net_return_after_costs": best.get("net_return"),
        "sharpe": best.get("sharpe"),
        "max_drawdown": best.get("max_drawdown"),
        "turnover": best.get("average_turnover"),
        "cost_drag": best.get("transaction_cost_drag"),
    }

def _portfolio_sweep_summary(payload: dict[str, Any]) -> dict[str, Any]:
    best = payload.get("winners", {}).get("best_by_net_return_after_costs") or {}
    comparison = payload.get("winners", {}).get("best_ml_vs_momentum_120d") or {}
    return {
        "best_portfolio_sweep_signal": best.get("signal_column"),
        "best_portfolio_sweep_policy": best.get("policy"),
        "best_portfolio_sweep_sizing_method": best.get("sizing_method"),
        "net_return_after_costs": best.get("net_return"),
        "sharpe": best.get("sharpe"),
        "max_drawdown": best.get("max_drawdown"),
        "turnover": best.get("average_turnover"),
        "cost_drag": best.get("transaction_cost_drag"),
        "ml_beats_momentum_120d_after_costs": comparison.get("beats_momentum_120d"),
    }

def _rows_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("name")): row
        for row in payload.get("leaderboard", []) or []
        if row.get("name")
    }

def _best_row(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows.values() if row.get("name") in SUMMARY_MODELS[2:]]
    if not candidates:
        return {}
    return max(
        candidates,
        key=lambda row: (
            _metric(row, "spearman_ic"),
            _metric(row, "top_minus_bottom_spread"),
        ),
    )

def _named_metrics(name: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "source_name": row.get("name"),
        **{metric: _metric(row, metric) for metric in METRIC_ALIASES},
    }

def _metric(row: dict[str, Any], metric: str) -> float | None:
    raw = row.get(METRIC_ALIASES[metric])
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None

def _winner(comparisons: dict[str, dict[str, Any]], metric: str) -> str | None:
    candidates = [
        (name, row[metric])
        for name, row in comparisons.items()
        if name not in SUMMARY_MODELS[:2] and row.get(metric) is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])[0]

def _did_enriched_help(comparisons: dict[str, dict[str, Any]]) -> bool:
    original = comparisons["original_baseline_artifact_benchmark"]
    enriched = comparisons["enriched_feature_benchmark"]
    return any(
        (enriched.get(metric) is not None)
        and (original.get(metric) is not None)
        and enriched[metric] > original[metric]
        for metric in (
            "spearman_ic",
            "top_minus_bottom_spread",
            "spread_sharpe",
            "risk_adjusted_spread",
        )
    )
