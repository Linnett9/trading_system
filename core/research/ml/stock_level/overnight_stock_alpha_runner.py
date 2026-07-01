from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable
from types import SimpleNamespace

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.logging import ResearchStageLogger
from core.research.ml.stock_level.stock_level_target_comparison import write_stock_level_target_comparison
from core.research.ml.stock_level.stock_level_portfolio_replay import write_stock_level_portfolio_replay
from core.research.ml.stock_level.stock_level_portfolio_policy_sweep import write_stock_level_portfolio_policy_sweep
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir, stock_alpha_report_metadata
from core.research.ml.stock_level.run_manifest.service import StockAlphaRunManifestTracker, expected_stage_output_paths
from core.research.ml.stock_level.overnight_stock_alpha_types import RESEARCH_GUARDRAILS, SUMMARY_MODELS, METRIC_ALIASES, OvernightStockAlphaPaths, OvernightStockAlphaStages
from core.research.ml.stock_level.overnight_stock_alpha_config import _artifact_source_paths, _artifact_status, _with_ml_overrides
from core.research.ml.stock_level.overnight_stock_alpha_validation import _stage_model_set_compatibility, _valid_output, _validate_stage_output_root, stock_alpha_stage_stale_reason
from core.research.ml.stock_level.overnight_stock_alpha_summary import _best_row, _build_summary, _did_enriched_help, _metric, _named_metrics, _parallelism_payload, _portfolio_summary, _portfolio_sweep_summary, _rows_by_name, _winner
from core.research.ml.stock_level.overnight_stock_alpha_reporting import _fmt, _markdown, _path_payload, _write_summary
from core.research.ml.stock_level.stock_alpha_stage_selection import StockAlphaStageSelector


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
    manifest = StockAlphaRunManifestTracker(run_config, output_dir)
    expected_manifest_outputs = expected_stage_output_paths(run_config, output_dir)

    timings: dict[str, dict[str, Any]] = {}
    stale_stage_reasons: dict[str, str] = {}

    def compatible_existing(stage_name: str) -> bool:
        expected = expected_manifest_outputs.get(stage_name, {})
        if not expected:
            return False
        dependency_keys = {
            "stock_artifact": ("csv_path", "json_path"),
            "alpha_features": ("enriched_csv_path", "audit_json_path"),
            "baseline_benchmark": ("json_path", "predictions_path"),
            "enriched_benchmark": ("json_path", "predictions_path"),
        }
        required_paths = {
            key: path
            for key, path in expected.items()
            if key in dependency_keys.get(stage_name, tuple(expected))
        }
        if not all(_valid_output(Path(path), set()) for path in required_paths.values()):
            return False
        stale_reason = stock_alpha_stage_stale_reason(
            stage_name,
            {name: Path(path) for name, path in required_paths.items()},
            settings,
            run_config,
        )
        model_set_valid = stale_reason is None
        if stale_reason:
            stale_stage_reasons[stage_name] = stale_reason
        return model_set_valid

    stage_selection = StockAlphaStageSelector(
        run_config,
        settings,
        output_exists=compatible_existing,
    ).resolve()

    def timed(
        stage_name: str,
        action: Callable[[], Any],
        *,
        manifest_stage_name: str | None = None,
    ) -> Any:
        manifest_stage = manifest_stage_name or stage_name
        label = stage_name.replace("_", " ")
        print(f"[stock-alpha] START {label}")
        started = clock()
        manifest.mark_running(manifest_stage)
        try:
            with logger.stage(stage_name):
                result = action()
        except KeyboardInterrupt:
            manifest.mark_interrupted(
                manifest_stage,
                elapsed_seconds=max(0.0, clock() - started),
            )
            raise
        except Exception as exc:
            manifest.mark_failed(
                manifest_stage,
                error=exc,
                elapsed_seconds=max(0.0, clock() - started),
            )
            raise
        elapsed = max(0.0, clock() - started)
        try:
            _validate_stage_output_root(
                stage_name,
                result,
                output_dir,
                legacy_output_paths_allowed,
            )
        except Exception as exc:
            manifest.mark_failed(manifest_stage, error=exc, elapsed_seconds=elapsed)
            raise
        output = _path_payload(result)
        manifest.mark_completed(
            manifest_stage,
            output_paths=output,
            elapsed_seconds=elapsed,
        )
        timings[stage_name] = {"status": "executed", "seconds": elapsed, "output_paths": output, "skipped": False}
        print(f"[stock-alpha] END {label} elapsed={elapsed:.1f}s output={output}")
        return result

    stale_dependencies = False

    def resumable(stage_name: str, expected: dict[str, Path], required: dict[str, set[str]], action: Callable[[], Any]) -> Any:
        nonlocal stale_dependencies
        valid = all(_valid_output(expected[name], required.get(name, set())) for name in expected)
        stale_reason = stock_alpha_stage_stale_reason(stage_name, expected, settings, run_config)
        model_set_valid = stale_reason is None
        if stage_name in {"target_comparison", "portfolio_replay", "portfolio_policy_sweep"} and stale_dependencies:
            model_set_valid, stale_reason = False, "upstream benchmark output is stale due to artifact/run-profile setting mismatch"
        valid = valid and model_set_valid
        if settings.resume_existing_outputs and not settings.force_refresh and valid:
            output_paths = {name: str(path) for name, path in expected.items()}
            timings[stage_name] = {"status": "skipped_existing", "seconds": 0.0, "output_paths": output_paths, "validation_passed": True, "skipped": True}
            manifest.mark_skipped(
                stage_name,
                output_paths=output_paths,
                skip_reason="existing output validated",
            )
            print(f"[stock-alpha] SKIP {stage_name.replace('_', ' ')} existing output validated")
            return SimpleNamespace(**{name: path for name, path in expected.items()})
        if stale_reason:
            timings[stage_name] = {"status": "stale_existing", "seconds": 0.0, "output_paths": {name: str(path) for name, path in expected.items()}, "validation_passed": False, "skipped": False, "stale_reason": stale_reason}
            manifest.mark_stale(stage_name, stale_reason=stale_reason)
            print(f"[stock-alpha] STALE {stage_name.replace('_', ' ')}: {stale_reason}")
            if stage_name in {"baseline_benchmark", "enriched_benchmark"}:
                stale_dependencies = True
        result = timed(stage_name, action)
        timings[stage_name]["validation_passed"] = all(_valid_output(expected[name], required.get(name, set())) for name in expected)
        return result

    def mark_disabled(stage_name: str) -> None:
        timings[stage_name] = {"status": "skipped_by_user", "seconds": 0.0, "output_paths": {}, "validation_passed": False, "skipped": True, "skip_reason": "disabled by ml.stock_alpha_stages"}
        manifest.mark_skipped(
            stage_name,
            output_paths=expected_manifest_outputs.get(stage_name, {}),
            skip_reason="disabled by ml.stock_alpha_stages",
        )

    artifact_status = _artifact_status(run_config, settings)
    artifact_stale_reason = stock_alpha_stage_stale_reason(
        "stock_artifact",
        expected_manifest_outputs["stock_artifact"],
        settings,
        run_config,
    )
    if artifact_stale_reason:
        artifact_status["refresh_required"] = True
        artifact_status["stale_reason"] = artifact_stale_reason
    if not stage_selection.enabled("stock_artifact"):
        artifact_paths = None
        mark_disabled("stock_artifact")
        artifact_status["action"] = "skipped_by_user"
    elif settings.force_refresh or not settings.resume_existing_outputs or artifact_status["refresh_required"]:
        if artifact_stale_reason:
            manifest.mark_stale("stock_artifact", stale_reason=artifact_stale_reason)
            print(f"[stock-alpha] STALE stock artifact: {artifact_stale_reason}")
        artifact_paths = timed(
            "stock_artifact",
            lambda: stages.stock_artifact(run_config),
        )
        artifact_status["action"] = "generated"
    else:
        artifact_paths = None
        stock_artifact_outputs = {"csv_path": str(settings.base_artifact_path), "json_path": str(output_dir / "stock_level_prediction_artifacts.json"), "markdown_path": str(output_dir / "stock_level_prediction_artifacts.md")}
        timings["stock_artifact"] = {"status": "skipped_existing", "seconds": 0.0, "output_paths": stock_artifact_outputs, "validation_passed": True, "skipped": True}
        manifest.mark_skipped(
            "stock_artifact",
            output_paths=stock_artifact_outputs,
            skip_reason="existing output validated",
        )
        print("[stock-alpha] SKIP stock artifact existing output validated")
        artifact_status["action"] = "reused"

    feature_config = _with_ml_overrides(
        run_config,
        output_dir=str(output_dir),
        stock_level_base_prediction_artifacts_path=str(settings.base_artifact_path),
    )
    feature_expected = {"enriched_csv_path": output_dir / "stock_level_prediction_artifacts_enriched.csv", "audit_json_path": output_dir / "stock_level_alpha_feature_audit.json"}
    if stage_selection.enabled("alpha_features"):
        feature_paths = resumable(
            "alpha_features",
            feature_expected,
            {"enriched_csv_path": {"rebalance_date", "symbol"}, "audit_json_path": {"features"}},
            lambda: stages.alpha_features(feature_config),
        )
    else:
        mark_disabled("alpha_features")
        configured_artifact = settings.artifact_path
        if configured_artifact.name == "stock_level_prediction_artifacts_enriched.csv" and configured_artifact.exists():
            feature_paths = SimpleNamespace(
                enriched_csv_path=configured_artifact,
                audit_json_path=feature_expected["audit_json_path"],
            )
        else:
            feature_paths = SimpleNamespace(**feature_expected)
    enriched_path = Path(feature_paths.enriched_csv_path)

    baseline_config = _with_ml_overrides(
        run_config,
        output_dir=str(output_dir / "baseline"),
        stock_level_prediction_artifacts_path=str(settings.base_artifact_path),
        stock_ranker_include_engineered_features=False,
    )
    baseline_expected = {"json_path": output_dir / "baseline" / "stock_level_model_ranking_benchmark.json", "predictions_path": output_dir / "baseline" / "stock_level_model_oos_predictions.csv"}
    if stage_selection.enabled("baseline_benchmark"):
        baseline_paths = resumable(
            "baseline_benchmark",
            baseline_expected,
            {"json_path": {"leaderboard", "target_column"}, "predictions_path": {"rebalance_date", "symbol"}},
            lambda: stages.benchmark(baseline_config),
        )
    else:
        mark_disabled("baseline_benchmark")
        baseline_paths = SimpleNamespace(**baseline_expected)

    enriched_config = _with_ml_overrides(
        run_config,
        output_dir=str(output_dir / "enriched"),
        stock_level_prediction_artifacts_path=str(enriched_path),
        stock_ranker_include_engineered_features=True,
    )
    enriched_expected = {"json_path": output_dir / "enriched" / "stock_level_model_ranking_benchmark.json", "predictions_path": output_dir / "enriched" / "stock_level_model_oos_predictions.csv"}
    if stage_selection.enabled("enriched_benchmark"):
        enriched_paths = resumable(
            "enriched_benchmark",
            enriched_expected,
            {"json_path": {"leaderboard", "target_column"}, "predictions_path": {"rebalance_date", "symbol"}},
            lambda: stages.benchmark(enriched_config),
        )
    else:
        mark_disabled("enriched_benchmark")
        enriched_paths = SimpleNamespace(**enriched_expected)

    target_config = _with_ml_overrides(enriched_config, output_dir=str(output_dir / "target_comparison"))
    if stages.target_comparison is not None and settings.target_comparison_enabled and stage_selection.enabled("target_comparison"):
        target_paths = resumable(
            "target_comparison",
            {"csv_path": output_dir / "target_comparison" / "stock_level_target_comparison.csv", "json_path": output_dir / "target_comparison" / "stock_level_target_comparison.json", "markdown_path": output_dir / "target_comparison" / "stock_level_target_comparison.md"},
            {"csv_path": {"target_column"}, "json_path": {"targets", "promotion_thresholds_changed"}, "markdown_path": set()},
            lambda: stages.target_comparison(target_config),
        )
    else:
        target_paths = None
        mark_disabled("target_comparison")

    portfolio_config = _with_ml_overrides(
        enriched_config,
        output_dir=str(output_dir / "portfolio_replay"),
        stock_level_model_ranking_benchmark_path=str(enriched_paths.json_path),
        stock_level_model_oos_predictions_path=str(enriched_paths.predictions_path),
    )
    if stages.portfolio_replay is not None and settings.overnight_run_portfolio_replay and stage_selection.enabled("portfolio_replay"):
        portfolio_paths = resumable(
            "portfolio_replay",
            {"csv_path": output_dir / "portfolio_replay" / "stock_level_portfolio_replay_summary.csv", "json_path": output_dir / "portfolio_replay" / "stock_level_portfolio_replay_summary.json", "markdown_path": output_dir / "portfolio_replay" / "stock_level_portfolio_replay_summary.md", "equity_curves_path": output_dir / "portfolio_replay" / "stock_level_portfolio_replay_equity_curves.csv", "holdings_path": output_dir / "portfolio_replay" / "stock_level_portfolio_replay_holdings.csv"},
            {"csv_path": {"signal_column", "policy"}, "json_path": {"summary", "winners", "promotion_thresholds_changed"}, "markdown_path": set(), "equity_curves_path": {"rebalance_date", "equity"}, "holdings_path": {"rebalance_date", "symbol", "weight"}},
            lambda: stages.portfolio_replay(portfolio_config),
        )
    else:
        portfolio_paths = None
        mark_disabled("portfolio_replay")

    sweep_config = _with_ml_overrides(
        enriched_config,
        output_dir=str(output_dir / "portfolio_policy_sweep"),
        stock_level_model_ranking_benchmark_path=str(enriched_paths.json_path),
        stock_level_model_oos_predictions_path=str(enriched_paths.predictions_path),
    )
    ml_config = run_config.get("ml", {})
    run_sweep = bool(ml_config.get("stock_alpha_overnight_run_portfolio_policy_sweep", settings.run_size != "full"))
    if stages.portfolio_policy_sweep is not None and run_sweep and stage_selection.enabled("portfolio_policy_sweep"):
        sweep_paths = resumable(
            "portfolio_policy_sweep",
            {"csv_path": output_dir / "portfolio_policy_sweep" / "stock_level_portfolio_policy_sweep.csv", "json_path": output_dir / "portfolio_policy_sweep" / "stock_level_portfolio_policy_sweep.json", "markdown_path": output_dir / "portfolio_policy_sweep" / "stock_level_portfolio_policy_sweep.md", "equity_curves_path": output_dir / "portfolio_policy_sweep" / "stock_level_portfolio_policy_sweep_equity_curves.csv", "top_holdings_path": output_dir / "portfolio_policy_sweep" / "stock_level_portfolio_policy_sweep_top_holdings.csv"},
            {"csv_path": {"config_id", "status"}, "json_path": {"summary", "winners", "promotion_thresholds_changed"}, "markdown_path": set(), "equity_curves_path": {"rebalance_date", "equity"}, "top_holdings_path": {"rebalance_date", "symbol", "weight"}},
            lambda: stages.portfolio_policy_sweep(sweep_config),
        )
    else:
        sweep_paths = None
        mark_disabled("portfolio_policy_sweep")

    attribution_paths = None

    summary_paths = OvernightStockAlphaPaths(
        json_path=output_dir / "overnight_stock_alpha_summary.json",
        markdown_path=output_dir / "overnight_stock_alpha_summary.md",
    )
    manifest.mark_running("overnight_summary")
    summary_started = clock()
    try:
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
        summary["stage_selection"] = stage_selection.payload()
        summary["stage_selection"]["stale_stages"] = dict(stale_stage_reasons)
        summary.update(stock_alpha_report_metadata(run_config, output_dir, source_artifact_path=settings.base_artifact_path))
        with logger.stage("summary_write"):
            started = clock()
            summary["stage_timings"]["summary_write"] = {"seconds": 0.0}
            _write_summary(summary_paths, summary)
            summary["stage_timings"]["summary_write"] = {
                "seconds": max(0.0, clock() - started)
            }
            _write_summary(summary_paths, summary)
    except KeyboardInterrupt:
        manifest.mark_interrupted(
            "overnight_summary",
            elapsed_seconds=max(0.0, clock() - summary_started),
        )
        raise
    except Exception as exc:
        manifest.mark_failed(
            "overnight_summary",
            error=exc,
            elapsed_seconds=max(0.0, clock() - summary_started),
        )
        raise
    manifest.mark_completed(
        "overnight_summary",
        output_paths={
            "json_path": summary_paths.json_path,
            "markdown_path": summary_paths.markdown_path,
        },
        elapsed_seconds=max(0.0, clock() - summary_started),
    )
    if settings.overnight_run_attribution and stage_selection.enabled("attribution"):
        attribution_config = _with_ml_overrides(
            enriched_config,
            stock_level_model_ranking_benchmark_path=str(enriched_paths.json_path),
            stock_level_model_oos_predictions_path=str(enriched_paths.predictions_path),
        )
        attribution_paths = timed(
            "attribution",
            lambda: stages.attribution(attribution_config),
            manifest_stage_name="optional_attribution",
        )
        summary["artifacts"]["attribution"] = _path_payload(attribution_paths)
        summary["stage_timings"] = timings
        _write_summary(summary_paths, summary)
    else:
        manifest.mark_skipped(
            "optional_attribution",
            output_paths=expected_manifest_outputs.get("optional_attribution", {}),
            skip_reason="disabled by ml.stock_alpha_stages" if not stage_selection.enabled("attribution") else "disabled",
        )
        summary["stage_timings"]["attribution"] = {"status": "skipped_by_user" if not stage_selection.enabled("attribution") else "disabled", "seconds": 0.0, "output_paths": {}, "validation_passed": False, "skipped": True}
        _write_summary(summary_paths, summary)
    return summary_paths
