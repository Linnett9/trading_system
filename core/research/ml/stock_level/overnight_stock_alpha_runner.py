from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from types import SimpleNamespace
import csv

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.data import JsonRepository
from core.research.framework.logging import ResearchStageLogger
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_level_alpha_features import (
    write_stock_level_alpha_features,
)
from core.research.ml.stock_level.stock_level_feature_attribution import (
    write_stock_level_feature_attribution,
)
from core.research.ml.stock_level.stock_level_model_ranking_benchmark import (
    write_stock_level_model_ranking_benchmark,
)
from core.research.ml.stock_level.stock_level_prediction_artifacts import (
    write_stock_level_prediction_artifacts,
)
from core.research.ml.stock_level.stock_level_target_comparison import write_stock_level_target_comparison
from core.research.ml.stock_level.stock_level_portfolio_replay import write_stock_level_portfolio_replay
from core.research.ml.stock_level.stock_level_portfolio_policy_sweep import write_stock_level_portfolio_policy_sweep
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir, stock_alpha_report_metadata


RESEARCH_GUARDRAILS = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
    "promotion_thresholds_changed": False,
}
SUMMARY_MODELS = (
    "original_baseline_artifact_benchmark",
    "enriched_feature_benchmark",
    "momentum_120d",
    "ridge",
    "elastic_net",
    "random_forest",
    "gradient_boosting",
)
METRIC_ALIASES = {
    "spearman_ic": "mean_spearman_ic",
    "top_minus_bottom_spread": "top_minus_bottom_spread",
    "spread_sharpe": "spread_sharpe",
    "risk_adjusted_spread": "risk_adjusted_spread",
    "top_decile_hit_rate": "top_decile_hit_rate",
}


@dataclass(frozen=True)
class OvernightStockAlphaPaths:
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class OvernightStockAlphaStages:
    stock_artifact: Callable[[dict[str, Any]], Any] = write_stock_level_prediction_artifacts
    alpha_features: Callable[[dict[str, Any]], Any] = write_stock_level_alpha_features
    benchmark: Callable[[dict[str, Any]], Any] = write_stock_level_model_ranking_benchmark
    attribution: Callable[[dict[str, Any]], Any] = write_stock_level_feature_attribution
    target_comparison: Callable[[dict[str, Any]], Any] | None = None
    portfolio_replay: Callable[[dict[str, Any]], Any] | None = None
    portfolio_policy_sweep: Callable[[dict[str, Any]], Any] | None = None


def write_overnight_stock_alpha_experiment(
    config: dict[str, Any],
    *,
    stages: OvernightStockAlphaStages | None = None,
    clock: Callable[[], float] | None = None,
) -> OvernightStockAlphaPaths:
    stages = stages or OvernightStockAlphaStages(target_comparison=write_stock_level_target_comparison, portfolio_replay=write_stock_level_portfolio_replay, portfolio_policy_sweep=write_stock_level_portfolio_policy_sweep)
    clock = clock or time.perf_counter
    logger = ResearchStageLogger("overnight_stock_alpha_experiment")
    output_dir = stock_alpha_output_dir(config)
    run_config = _with_ml_overrides(config, output_dir=str(output_dir))
    settings = StockLevelResearchConfig.from_mapping(run_config)
    base_output_dir = output_dir
    legacy_output_paths_allowed = bool(run_config.get("ml", {}).get("stock_alpha_allow_legacy_output_paths", False))
    output_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, dict[str, Any]] = {}

    def timed(stage_name: str, action: Callable[[], Any]) -> Any:
        label = stage_name.replace("_", " ")
        print(f"[stock-alpha] START {label}")
        with logger.stage(stage_name):
            started = clock()
            result = action()
            elapsed = max(0.0, clock() - started)
        _validate_stage_output_root(stage_name, result, output_dir, legacy_output_paths_allowed)
        output = _path_payload(result)
        timings[stage_name] = {"status": "executed", "seconds": elapsed, "output_paths": output, "skipped": False}
        print(f"[stock-alpha] END {label} elapsed={elapsed:.1f}s output={output}")
        return result

    def resumable(stage_name: str, expected: dict[str, Path], required: dict[str, set[str]], action: Callable[[], Any]) -> Any:
        valid = all(_valid_output(expected[name], required.get(name, set())) for name in expected)
        if settings.resume_existing_outputs and not settings.force_refresh and valid:
            timings[stage_name] = {"status": "skipped_existing", "seconds": 0.0, "output_paths": {name: str(path) for name, path in expected.items()}, "validation_passed": True, "skipped": True}
            print(f"[stock-alpha] SKIP {stage_name.replace('_', ' ')} existing output validated")
            return SimpleNamespace(**{name: path for name, path in expected.items()})
        result = timed(stage_name, action)
        timings[stage_name]["validation_passed"] = all(_valid_output(expected[name], required.get(name, set())) for name in expected)
        return result

    artifact_status = _artifact_status(run_config, settings)
    if settings.force_refresh or not settings.resume_existing_outputs or artifact_status["refresh_required"]:
        artifact_paths = timed(
            "stock_artifact",
            lambda: stages.stock_artifact(run_config),
        )
        artifact_status["action"] = "generated"
    else:
        artifact_paths = None
        timings["stock_artifact"] = {"status": "skipped_existing", "seconds": 0.0, "output_paths": {"csv_path": str(settings.base_artifact_path), "json_path": str(output_dir / "stock_level_prediction_artifacts.json"), "markdown_path": str(output_dir / "stock_level_prediction_artifacts.md")}, "validation_passed": True, "skipped": True}
        print("[stock-alpha] SKIP stock artifact existing output validated")
        artifact_status["action"] = "reused"

    feature_config = _with_ml_overrides(
        run_config,
        output_dir=str(output_dir),
        stock_level_base_prediction_artifacts_path=str(settings.base_artifact_path),
    )
    feature_paths = resumable(
        "alpha_features",
        {"enriched_csv_path": output_dir / "stock_level_prediction_artifacts_enriched.csv", "audit_json_path": output_dir / "stock_level_alpha_feature_audit.json"},
        {"enriched_csv_path": {"rebalance_date", "symbol"}, "audit_json_path": {"features"}},
        lambda: stages.alpha_features(feature_config),
    )
    enriched_path = Path(feature_paths.enriched_csv_path)

    baseline_config = _with_ml_overrides(
        run_config,
        output_dir=str(output_dir / "baseline"),
        stock_level_prediction_artifacts_path=str(settings.base_artifact_path),
        stock_ranker_include_engineered_features=False,
    )
    baseline_paths = resumable(
        "baseline_benchmark",
        {"json_path": output_dir / "baseline" / "stock_level_model_ranking_benchmark.json", "predictions_path": output_dir / "baseline" / "stock_level_model_oos_predictions.csv"},
        {"json_path": {"leaderboard", "target_column"}, "predictions_path": {"rebalance_date", "symbol"}},
        lambda: stages.benchmark(baseline_config),
    )

    enriched_config = _with_ml_overrides(
        run_config,
        output_dir=str(output_dir / "enriched"),
        stock_level_prediction_artifacts_path=str(enriched_path),
        stock_ranker_include_engineered_features=True,
    )
    enriched_paths = resumable(
        "enriched_benchmark",
        {"json_path": output_dir / "enriched" / "stock_level_model_ranking_benchmark.json", "predictions_path": output_dir / "enriched" / "stock_level_model_oos_predictions.csv"},
        {"json_path": {"leaderboard", "target_column"}, "predictions_path": {"rebalance_date", "symbol"}},
        lambda: stages.benchmark(enriched_config),
    )

    target_config = _with_ml_overrides(enriched_config, output_dir=str(output_dir / "target_comparison"))
    if stages.target_comparison is not None and settings.target_comparison_enabled:
        target_paths = resumable(
            "target_comparison",
            {"csv_path": output_dir / "target_comparison" / "stock_level_target_comparison.csv", "json_path": output_dir / "target_comparison" / "stock_level_target_comparison.json", "markdown_path": output_dir / "target_comparison" / "stock_level_target_comparison.md"},
            {"csv_path": {"target_column"}, "json_path": {"targets", "promotion_thresholds_changed"}, "markdown_path": set()},
            lambda: stages.target_comparison(target_config),
        )
    else:
        target_paths = None
        timings["target_comparison"] = {"status": "disabled", "seconds": 0.0, "output_paths": {}, "validation_passed": False, "skipped": True}

    portfolio_config = _with_ml_overrides(
        enriched_config,
        output_dir=str(output_dir / "portfolio_replay"),
        stock_level_model_ranking_benchmark_path=str(enriched_paths.json_path),
        stock_level_model_oos_predictions_path=str(enriched_paths.predictions_path),
    )
    if stages.portfolio_replay is not None and settings.overnight_run_portfolio_replay:
        portfolio_paths = resumable(
            "portfolio_replay",
            {"csv_path": output_dir / "portfolio_replay" / "stock_level_portfolio_replay_summary.csv", "json_path": output_dir / "portfolio_replay" / "stock_level_portfolio_replay_summary.json", "markdown_path": output_dir / "portfolio_replay" / "stock_level_portfolio_replay_summary.md", "equity_curves_path": output_dir / "portfolio_replay" / "stock_level_portfolio_replay_equity_curves.csv", "holdings_path": output_dir / "portfolio_replay" / "stock_level_portfolio_replay_holdings.csv"},
            {"csv_path": {"signal_column", "policy"}, "json_path": {"summary", "winners", "promotion_thresholds_changed"}, "markdown_path": set(), "equity_curves_path": {"rebalance_date", "equity"}, "holdings_path": {"rebalance_date", "symbol", "weight"}},
            lambda: stages.portfolio_replay(portfolio_config),
        )
    else:
        portfolio_paths = None
        timings["portfolio_replay"] = {"status": "disabled", "seconds": 0.0, "output_paths": {}, "validation_passed": False, "skipped": True}

    sweep_config = _with_ml_overrides(
        enriched_config,
        output_dir=str(output_dir / "portfolio_policy_sweep"),
        stock_level_model_ranking_benchmark_path=str(enriched_paths.json_path),
        stock_level_model_oos_predictions_path=str(enriched_paths.predictions_path),
    )
    ml_config = run_config.get("ml", {})
    run_sweep = bool(ml_config.get("stock_alpha_overnight_run_portfolio_policy_sweep", settings.run_size != "full"))
    if stages.portfolio_policy_sweep is not None and run_sweep:
        sweep_paths = resumable(
            "portfolio_policy_sweep",
            {"csv_path": output_dir / "portfolio_policy_sweep" / "stock_level_portfolio_policy_sweep.csv", "json_path": output_dir / "portfolio_policy_sweep" / "stock_level_portfolio_policy_sweep.json", "markdown_path": output_dir / "portfolio_policy_sweep" / "stock_level_portfolio_policy_sweep.md", "equity_curves_path": output_dir / "portfolio_policy_sweep" / "stock_level_portfolio_policy_sweep_equity_curves.csv", "top_holdings_path": output_dir / "portfolio_policy_sweep" / "stock_level_portfolio_policy_sweep_top_holdings.csv"},
            {"csv_path": {"config_id", "status"}, "json_path": {"summary", "winners", "promotion_thresholds_changed"}, "markdown_path": set(), "equity_curves_path": {"rebalance_date", "equity"}, "top_holdings_path": {"rebalance_date", "symbol", "weight"}},
            lambda: stages.portfolio_policy_sweep(sweep_config),
        )
    else:
        sweep_paths = None
        timings["portfolio_policy_sweep"] = {"status": "disabled", "seconds": 0.0, "output_paths": {}, "validation_passed": False, "skipped": True}

    attribution_paths = None

    summary_paths = OvernightStockAlphaPaths(
        json_path=output_dir / "overnight_stock_alpha_summary.json",
        markdown_path=output_dir / "overnight_stock_alpha_summary.md",
    )
    summary = _build_summary(
        base_output_dir=base_output_dir,
        output_dir=output_dir,
        settings=settings,
        artifact_status=artifact_status,
        artifact_paths=artifact_paths,
        feature_paths=feature_paths,
        baseline_paths=baseline_paths,
        enriched_paths=enriched_paths,
        target_paths=target_paths,
        portfolio_paths=portfolio_paths,
        sweep_paths=sweep_paths,
        attribution_paths=attribution_paths,
        timings=timings,
    )
    summary.update(stock_alpha_report_metadata(run_config, output_dir, source_artifact_path=settings.base_artifact_path))
    with logger.stage("summary_write"):
        started = clock()
        summary["stage_timings"]["summary_write"] = {"seconds": 0.0}
        _write_summary(summary_paths, summary)
        summary["stage_timings"]["summary_write"] = {
            "seconds": max(0.0, clock() - started)
        }
        _write_summary(summary_paths, summary)
    if settings.overnight_run_attribution:
        attribution_config = _with_ml_overrides(
            enriched_config,
            stock_level_model_ranking_benchmark_path=str(enriched_paths.json_path),
            stock_level_model_oos_predictions_path=str(enriched_paths.predictions_path),
        )
        attribution_paths = timed("attribution", lambda: stages.attribution(attribution_config))
        summary["artifacts"]["attribution"] = _path_payload(attribution_paths)
        summary["stage_timings"] = timings
        _write_summary(summary_paths, summary)
    else:
        summary["stage_timings"]["attribution"] = {"status": "disabled", "seconds": 0.0, "output_paths": {}, "validation_passed": False, "skipped": True}
        _write_summary(summary_paths, summary)
    return summary_paths


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


def _valid_output(path: Path, required_fields: set[str]) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if not required_fields:
        return True
    try:
        if path.suffix == ".json":
            payload = JsonRepository().read(path)
            return isinstance(payload, dict) and required_fields.issubset(payload)
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                fields = set(next(csv.reader(handle), []))
            return required_fields.issubset(fields)
    except (OSError, ValueError):
        return False
    return True


def _artifact_status(
    config: dict[str, Any],
    settings: StockLevelResearchConfig,
) -> dict[str, Any]:
    source_paths = _artifact_source_paths(config, settings.output_dir)
    existing = settings.base_artifact_path
    existing_mtime = existing.stat().st_mtime if existing.exists() else None
    newest_source = max(
        (path.stat().st_mtime for path in source_paths if path.exists()),
        default=None,
    )
    refresh_required = (
        not existing.exists()
        or (
            newest_source is not None
            and existing_mtime is not None
            and newest_source > existing_mtime
        )
    )
    return {
        "path": str(existing),
        "refresh_required": refresh_required,
        "source_paths": [str(path) for path in source_paths],
    }


def _artifact_source_paths(config: dict[str, Any], output_dir: Path) -> list[Path]:
    ml = dict(config.get("ml", {}) or {})
    cache = dict(config.get("cache", {}) or {})
    return [
        Path(
            ml.get(
                "expanded_rebalance_dataset_path",
                Path(cache.get("ml_dir", "cache/ml")) / "expanded_rebalance_dataset.csv",
            )
        ),
        output_dir / "meta_auxiliary_predictions.csv",
    ]


def _with_ml_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    updated = dict(config)
    if "output_dir" in overrides:
        overrides["stock_alpha_output_dir_override"] = True
    updated["ml"] = {**dict(config.get("ml", {}) or {}), **overrides}
    return updated


def _validate_stage_output_root(
    stage_name: str,
    paths: Any,
    output_root: Path,
    legacy_output_paths_allowed: bool,
) -> None:
    if paths is None:
        return
    root = output_root.resolve()
    for key, value in _path_payload(paths).items():
        path = Path(value)
        resolved = path.resolve()
        if root == resolved or root in resolved.parents:
            continue
        is_legacy = "reports/ml/benchmark/ml" in path.as_posix()
        if is_legacy and legacy_output_paths_allowed:
            continue
        raise ValueError(
            "Output-root validation failed for "
            f"{stage_name}.{key}: {path} is outside canonical output dir {output_root}"
        )


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


def _path_payload(paths: Any) -> dict[str, str]:
    if paths is None:
        return {}
    return {
        key: str(value)
        for key, value in vars(paths).items()
        if isinstance(value, Path)
    }


def _write_summary(paths: OvernightStockAlphaPaths, payload: dict[str, Any]) -> None:
    writer = ResearchArtifactWriter()
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Overnight Stock Alpha Summary",
        "",
        "Research only. Trading impact: none. Production validated: false.",
        "",
        f"- Run size: `{payload.get('run_size', 'benchmark')}`",
        f"- Effective rows/dates/symbols: {payload.get('effective_row_count')}/{payload.get('effective_date_count')}/{payload.get('effective_symbol_count')}",
        "",
        "## Winners",
    ]
    for key, value in payload["winners"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Comparison"])
    lines.append(
        "| model | spearman_ic | top_minus_bottom_spread | spread_sharpe | risk_adjusted_spread | top_decile_hit_rate |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in SUMMARY_MODELS:
        row = payload["comparisons"].get(name, {})
        lines.append(
            "| {name} | {spearman} | {spread} | {sharpe} | {risk} | {hit} |".format(
                name=name,
                spearman=_fmt(row.get("spearman_ic")),
                spread=_fmt(row.get("top_minus_bottom_spread")),
                sharpe=_fmt(row.get("spread_sharpe")),
                risk=_fmt(row.get("risk_adjusted_spread")),
                hit=_fmt(row.get("top_decile_hit_rate")),
            )
        )
    lines.extend(["", "## Stage Timings"])
    for stage, timing in payload["stage_timings"].items():
        lines.append(f"- {stage}: status={timing.get('status', 'executed')} seconds={_fmt(timing.get('seconds'))} skipped={timing.get('skipped', False)} output={timing.get('output_paths', {})}")
    portfolio = payload.get("portfolio_replay", {})
    lines.extend(["", "## Portfolio Replay"])
    for key in ("best_portfolio_signal", "best_portfolio_policy", "net_return_after_costs", "sharpe", "max_drawdown", "turnover", "cost_drag"):
        lines.append(f"- {key}: {portfolio.get(key)}")
    sweep = payload.get("portfolio_policy_sweep", {})
    lines.extend(["", "## Portfolio Policy Sweep"])
    for key, value in sweep.items():
        lines.append(f"- {key}: {value}")
    parallelism = payload.get("parallelism", {})
    lines.extend(["", "## Parallelism"])
    for key in (
        "stock_alpha_feature_n_jobs",
        "stock_ranker_model_n_jobs",
        "sklearn_n_jobs",
        "effective_model_workers",
        "stock_alpha_overnight_stage_n_jobs",
        "effective_stage_workers",
        "stages",
    ):
        lines.append(f"- {key}: {parallelism.get(key)}")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)
