from __future__ import annotations

from pathlib import Path
from typing import Any

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.data import JsonRepository
from core.research.ml.stock_level.overnight_stock_alpha_types import RESEARCH_GUARDRAILS, SUMMARY_MODELS, METRIC_ALIASES
from core.research.ml.stock_level.overnight_stock_alpha_reporting import _path_payload


def _build_summary(
    *,
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
    baseline = JsonRepository().read(Path(baseline_paths.json_path))
    enriched = JsonRepository().read(Path(enriched_paths.json_path))
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
        "parallelism": _parallelism_payload(settings, baseline, enriched),
        **RESEARCH_GUARDRAILS,
    }

def _parallelism_payload(
    settings: StockLevelResearchConfig,
    baseline: dict[str, Any],
    enriched: dict[str, Any],
) -> dict[str, Any]:
    enriched_parallelism = dict(enriched.get("parallelism", {}) or {})
    baseline_parallelism = dict(baseline.get("parallelism", {}) or {})
    requested_stage_workers = settings.overnight_stage_n_jobs
    return {
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
            "Overnight stages remain sequential by default; alpha feature generation "
            "and each benchmark use their own bounded worker settings."
        ),
    }

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
