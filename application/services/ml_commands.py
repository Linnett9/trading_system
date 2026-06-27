from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from config.config_loader import load_config
from application.services.research_profiles import apply_research_profile
from core.research.ml.experiment_runner import MLExperimentRunner
from core.research.ml.data_inventory import build_data_inventory
from core.research.ml.universe_builder import build_universe_files
from core.research.ml.meta_ensemble import run_meta_ensemble
from core.research.ml.artifact_validator import validate_prediction_artifact_dirs
from core.research.ml.leaderboard import write_source_leaderboard
from core.research.ml.model_contract_audit import write_model_contract_audit
from core.research.ml.canonical_continuous_equity_replay import (
    write_canonical_continuous_equity_replay,
)
from core.research.ml.champion_baseline_audit import write_champion_baseline_audit
from core.research.ml.benchmark_relative_validation import (
    write_benchmark_relative_validation,
)
from core.research.ml.data_adjustment_validation import (
    write_clean_data_replay,
    write_data_adjustment_audit,
    write_independent_period_validation,
)
from core.research.ml.data_anomaly_quarantine import write_data_anomaly_quarantine
from core.research.ml.profit_concentration_audit import (
    write_profit_concentration_audit,
)
from core.research.ml.return_mechanics_audit import write_return_mechanics_audit
from core.research.ml.trading_research_leaderboard import (
    write_trading_research_leaderboard,
)
from core.research.ml.runtime_parallelism import (
    apply_runtime_parallelism,
    apply_worker_thread_environment,
    format_runtime_settings,
)
from infrastructure.data.stooq_parquet_data_feed import StooqParquetDataFeed


@dataclass(frozen=True)
class MLResearchBatchItem:
    config_path: Path
    output_dir: Path


@dataclass(frozen=True)
class MLResearchBatchResult:
    config_path: str
    output_dir: str
    success: bool
    metrics_path: str | None = None
    prediction_artifacts_path: str | None = None
    error: str | None = None


def run_ml_research(config, feed=None):
    runtime_settings = apply_runtime_parallelism(config)
    result = MLExperimentRunner(config, feed=feed).run()
    leaderboard_markdown_path, leaderboard_json_path = _update_source_leaderboard(
        config,
        result.output_dir,
    )
    print("\nML RESEARCH")
    print(
        "mode=research | "
        f"label={config.get('ml', {}).get('research_label', 'UNSPECIFIED_RESEARCH')} | "
        "trading_impact=none"
    )
    print(f"Runtime: {format_runtime_settings(runtime_settings)}")
    print(f"Output dir: {result.output_dir}")
    print(f"Metrics: {result.metrics_path}")
    print(f"Predictions: {result.predictions_path}")
    print(f"Feature importance: {result.feature_importance_path}")
    print(f"Confusion matrix: {result.confusion_matrix_path}")
    print(f"Metadata: {result.metadata_path}")
    print(f"Model: {result.model_path}")
    print(f"Features: {result.features_path}")
    print(f"Feature summary: {result.feature_summary_path}")
    print(f"Labels: {result.labels_path}")
    print(f"Dataset: {result.dataset_path}")
    print(f"Dataset audit: {result.dataset_audit_path}")
    print(f"Walk-forward metrics: {result.walk_forward_metrics_path}")
    print(f"Threshold sweep: {result.threshold_sweep_path}")
    print(f"Model comparison: {result.model_comparison_path}")
    print(f"Shadow overlay: {result.shadow_overlay_path}")
    print(f"Holdout shadow overlay: {result.holdout_shadow_overlay_path}")
    print(f"Champion rebalance dataset: {result.rebalance_dataset_path}")
    print(f"Champion rebalance audit: {result.rebalance_dataset_audit_path}")
    print(f"History coverage: {result.history_coverage_path}")
    print(f"Drawdown event review: {result.drawdown_event_review_path}")
    print(f"Rule exposure study: {result.rule_exposure_study_path}")
    print(f"Probability calibration: {result.probability_calibration_path}")
    print(
        "Walk-forward probability calibration: "
        f"{result.walk_forward_probability_calibration_path}"
    )
    print(f"Baseline model comparison: {result.baseline_model_comparison_path}")
    print(f"Ranking diagnostics: {result.ranking_diagnostics_path}")
    print(f"Leaderboard: {leaderboard_markdown_path}")
    print(f"Leaderboard JSON: {leaderboard_json_path}")


def run_ml_research_batch(
    config: dict[str, Any],
    *,
    executor_cls: type[ProcessPoolExecutor] = ProcessPoolExecutor,
    worker_fn: Callable[[str, int, str], MLResearchBatchResult] | None = None,
) -> list[MLResearchBatchResult]:
    batch_config = config.get("ml_research_batch", {})
    runtime_settings = apply_runtime_parallelism(config)
    items = validate_ml_research_batch_config(config)
    max_workers = int(batch_config.get("max_workers", runtime_settings.num_workers))
    model_threads = int(batch_config.get("model_threads", runtime_settings.model_threads))
    fail_fast = bool(batch_config.get("fail_fast", True))
    shared_dataset_path = str(_expanded_rebalance_dataset_path(config))
    profile_name = str(config.get("research_profile", {}).get("name", "") or "")
    worker = worker_fn or _run_ml_research_batch_worker

    print("\nML RESEARCH BATCH")
    print("mode=research | trading_impact=none")
    print(f"Configs: {len(items)}")
    print(f"Workers: {max_workers}")
    print(f"Model threads: {model_threads}")
    print(f"Runtime: {format_runtime_settings(runtime_settings)}")
    print(f"Shared expanded dataset: {shared_dataset_path}")

    results: list[MLResearchBatchResult] = []
    with executor_cls(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                worker,
                str(item.config_path),
                model_threads,
                shared_dataset_path,
                profile_name,
            ): item
            for item in items
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive process boundary
                result = MLResearchBatchResult(
                    config_path=str(item.config_path),
                    output_dir=str(item.output_dir),
                    success=False,
                    error=str(exc),
                )
            results.append(result)
            status = "ok" if result.success else "failed"
            print(f"{status}: {result.config_path} -> {result.output_dir}")
            if result.error:
                print(f"  error: {result.error}")
            if result.success:
                leaderboard_markdown_path, _ = _update_source_leaderboard(
                    config,
                    Path(result.output_dir),
                )
                print(f"  leaderboard: {leaderboard_markdown_path}")
            if fail_fast and not result.success:
                for pending in futures:
                    pending.cancel()
                break

    failures = [result for result in results if not result.success]
    if failures:
        details = "; ".join(
            f"{Path(result.config_path).name}: {result.error}" for result in failures
        )
        raise RuntimeError(f"ML research batch failed: {details}")
    return results


def validate_ml_research_batch_config(config: dict[str, Any]) -> list[MLResearchBatchItem]:
    batch_config = config.get("ml_research_batch")
    if not isinstance(batch_config, dict):
        raise RuntimeError("ml-research-batch requires ml_research_batch config")
    config_paths = batch_config.get("config_paths", [])
    if not config_paths:
        raise RuntimeError("ml_research_batch.config_paths must contain at least one config")
    max_workers = int(batch_config.get("max_workers", 1))
    model_threads = int(batch_config.get("model_threads", 1))
    if max_workers < 1:
        raise RuntimeError("ml_research_batch.max_workers must be at least one")
    if model_threads < 1:
        raise RuntimeError("ml_research_batch.model_threads must be at least one")

    expanded_dataset_path = _expanded_rebalance_dataset_path(config)
    if not expanded_dataset_path.exists():
        raise RuntimeError(
            "ml-research-batch requires existing expanded rebalance dataset: "
            f"{expanded_dataset_path}"
        )

    items = []
    output_dirs: dict[Path, Path] = {}
    shared_cache_dir = Path(config.get("cache", {}).get("ml_dir", "cache/ml")).resolve()
    profile_name = str(config.get("research_profile", {}).get("name", "") or "")
    for raw_path in config_paths:
        config_path = Path(str(raw_path))
        if not config_path.exists():
            raise RuntimeError(f"Batch research config does not exist: {config_path}")
        child_config = apply_research_profile(
            load_config(str(config_path), overlay_project_config=True),
            profile_name or None,
        )
        child_cache_dir = Path(
            child_config.get("cache", {}).get("ml_dir", "cache/ml")
        ).resolve()
        if child_cache_dir != shared_cache_dir:
            raise RuntimeError(
                "All ml-research-batch configs must use the shared cache ml_dir "
                f"{shared_cache_dir}; {config_path} uses {child_cache_dir}"
            )
        output_dir = Path(
            child_config.get("ml", {}).get("output_dir", "reports/ml")
        ).resolve()
        if output_dir in output_dirs:
            raise RuntimeError(
                "Duplicate ml.output_dir in batch configs: "
                f"{output_dir} used by {output_dirs[output_dir]} and {config_path}"
            )
        output_dirs[output_dir] = config_path
        items.append(MLResearchBatchItem(config_path=config_path, output_dir=output_dir))
    return items


def _expanded_rebalance_dataset_path(config: dict[str, Any]) -> Path:
    ml_config = config.get("ml", {})
    return Path(
        ml_config.get(
            "expanded_rebalance_dataset_path",
            Path(config.get("cache", {}).get("ml_dir", "cache/ml"))
            / "expanded_rebalance_dataset.csv",
        )
    )


def _run_ml_research_batch_worker(
    config_path: str,
    model_threads: int,
    expanded_dataset_path: str,
    profile_name: str = "",
) -> MLResearchBatchResult:
    apply_worker_thread_environment(model_threads)

    config = apply_research_profile(
        load_config(config_path, overlay_project_config=True),
        profile_name or None,
    )
    ml_config = config.setdefault("ml", {})
    ml_config.setdefault("model_threads", model_threads)
    ml_config.setdefault("torch_num_threads", model_threads)
    ml_config.setdefault("sklearn_n_jobs", model_threads)
    apply_runtime_parallelism(config)
    output_dir = Path(config.get("ml", {}).get("output_dir", "reports/ml"))
    worker_config = _batch_worker_config(config, expanded_dataset_path)
    try:
        result = MLExperimentRunner(
            worker_config,
            feed=_build_research_feed(worker_config),
        ).run()
        return MLResearchBatchResult(
            config_path=config_path,
            output_dir=str(result.output_dir),
            success=True,
            metrics_path=str(result.metrics_path),
            prediction_artifacts_path=str(result.prediction_artifacts_path),
        )
    except Exception as exc:
        return MLResearchBatchResult(
            config_path=config_path,
            output_dir=str(output_dir),
            success=False,
            error=str(exc),
        )


def _batch_worker_config(
    config: dict[str, Any],
    expanded_dataset_path: str,
) -> dict[str, Any]:
    worker_config = deepcopy(config)
    ml_config = worker_config.setdefault("ml", {})
    output_dir = Path(ml_config.get("output_dir", "reports/ml"))
    worker_config.setdefault("cache", {})["enabled"] = False
    worker_config.setdefault("cache", {})["ml_dir"] = str(output_dir / "_batch_cache")
    ml_config["read_existing_expanded_rebalance_dataset"] = True
    ml_config["expanded_rebalance_dataset_path"] = expanded_dataset_path
    return worker_config


def _build_research_feed(config: dict[str, Any]):
    provider = config.get("backtest", {}).get("provider", "alpaca").lower()
    if provider == "stooq_parquet":
        from infrastructure.data.stooq_parquet_data_feed import StooqParquetDataFeed

        return StooqParquetDataFeed(
            data_dir=config.get("backtest", {}).get(
                "data_dir", "data/processed/stooq_parquet"
            )
        )
    if provider == "stooq_csv":
        from infrastructure.data.stooq_csv_data_feed import StooqCsvDataFeed

        return StooqCsvDataFeed(
            data_dir=config.get("backtest", {}).get("data_dir", "data/raw/stooq")
        )
    if provider == "stooq":
        from infrastructure.data.stooq_data_feed import StooqDataFeed

        return StooqDataFeed()
    raise RuntimeError(
        "ml-research-batch supports local research data providers only; "
        f"unsupported provider '{provider}'"
    )


def run_ml_data_inventory(config):
    ml_config = config.get("ml", {})
    inventories = build_data_inventory(
        parquet_dir=ml_config.get("parquet_dir", ml_config.get(
            "stooq_parquet_dir", "data/processed/stooq_parquet"
        )),
        output_dir=ml_config.get("inventory_output_dir", "reports/ml"),
        min_history_years=int(ml_config.get("min_history_years", 9)),
        max_latest_gap_days=int(ml_config.get("max_latest_gap_days", 14)),
        min_average_dollar_volume_252d=float(
            ml_config.get("min_average_dollar_volume_252d", 50_000_000)
        ),
    )
    included_count = sum(1 for item in inventories if item.included)
    missing_count = len(inventories) - included_count
    output_dir = ml_config.get("inventory_output_dir", "reports/ml")
    print("\nML DATA INVENTORY")
    print("mode=research | trading_impact=none")
    print(f"Scanned symbols: {len(inventories)}")
    print(f"Included symbols: {included_count}")
    print(f"Excluded symbols: {missing_count}")
    print(f"Inventory: {output_dir}/data_inventory.json")
    print(f"Coverage CSV: {output_dir}/symbol_coverage.csv")


def run_ml_build_universes(config):
    ml_config = config.get("ml", {})
    inventory_output_dir = ml_config.get("inventory_output_dir", "reports/ml")
    results = build_universe_files(
        inventory_path=f"{inventory_output_dir}/data_inventory.json",
        output_dir=ml_config.get("universe_output_dir", "data/reference/universes"),
        parquet_dir=ml_config.get("parquet_dir", ml_config.get(
            "stooq_parquet_dir", "data/processed/stooq_parquet"
        )),
        inventory_output_dir=inventory_output_dir,
        min_history_years=int(ml_config.get("min_history_years", 9)),
        max_latest_gap_days=int(ml_config.get("max_latest_gap_days", 14)),
        min_average_dollar_volume_252d=float(
            ml_config.get("min_average_dollar_volume_252d", 50_000_000)
        ),
    )
    print("\nML UNIVERSE BUILD")
    print("mode=research | trading_impact=none")
    for result in results:
        print(
            f"{result.name}: {result.symbol_count} symbols "
            f"(available={result.available_count}) -> {result.path}"
        )


def run_ml_expanded_rebalance_dataset(config, feed):
    runtime_settings = apply_runtime_parallelism(config)
    dataset_path, audit_path, row_count = MLExperimentRunner(
        config,
        feed=feed,
    ).build_expanded_rebalance_dataset()
    print("\nML EXPANDED REBALANCE DATASET")
    print("mode=research | trading_impact=none")
    print(f"Runtime: {format_runtime_settings(runtime_settings)}")
    print(f"Rows: {row_count}")
    print(f"Dataset: {dataset_path}")
    print(f"Audit: {audit_path}")


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


def run_ml_return_mechanics_audit(config):
    result = write_return_mechanics_audit(config)
    champion_result = write_champion_baseline_audit(config)
    canonical_result = write_canonical_continuous_equity_replay(config)
    anomaly_result = write_data_anomaly_quarantine(config)
    concentration_result = write_profit_concentration_audit(config)
    research_feed = StooqParquetDataFeed(
        str(
            config.get("ml", {}).get(
                "stooq_parquet_dir",
                "data/processed/stooq_parquet",
            )
        )
    )
    adjustment_result = write_data_adjustment_audit(config)
    independent_result = write_independent_period_validation(config)
    clean_replay_result = write_clean_data_replay(config, research_feed)
    validation_result = write_benchmark_relative_validation(
        config,
        research_feed,
    )
    leaderboard_result = _refresh_trading_research_leaderboard(config)
    print("\nML RETURN MECHANICS AUDIT")
    print("mode=research | trading_impact=none")
    print(f"CSV: {result.csv_path}")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")
    print(f"Champion baseline CSV: {champion_result.csv_path}")
    print(f"Champion baseline JSON: {champion_result.json_path}")
    print(f"Champion baseline Markdown: {champion_result.markdown_path}")
    print(f"Canonical replay JSON: {canonical_result.json_path}")
    print(f"Anomaly quarantine JSON: {anomaly_result.json_path}")
    print(f"Profit concentration JSON: {concentration_result.json_path}")
    print(f"Data adjustment audit JSON: {adjustment_result.json_path}")
    print(f"Independent-period validation JSON: {independent_result.json_path}")
    print(f"Clean-data replay JSON: {clean_replay_result.json_path}")
    print(f"Benchmark-relative validation JSON: {validation_result.json_path}")
    print(f"Promotion readiness: {validation_result.promotion_readiness_path}")
    print(f"Trading research leaderboard JSON: {leaderboard_result.json_path}")


def _refresh_trading_research_leaderboard(config):
    output_dir = Path(
        config.get("ml", {}).get(
            "output_dir",
            Path(config.get("reports", {}).get("ml_dir", "reports/ml"))
            / "regime_transformer_meta_ensemble_v1",
        )
    )
    return write_trading_research_leaderboard(
        output_dir=output_dir,
        classification_leaderboard_path=output_dir / "leaderboard.json",
        allocation_comparison_path=output_dir / "allocation_policy_comparison.json",
        optimizer_results_path=output_dir / "allocation_optimizer_results.json",
        auxiliary_metrics_path=output_dir / "meta_auxiliary_metrics.json",
    )


def run_ml_validate_artifacts(config):
    source_dirs = _artifact_source_dirs(config)
    results = validate_prediction_artifact_dirs(source_dirs)
    print("\nML ARTIFACT VALIDATION")
    print("mode=research | trading_impact=none")
    for result in results:
        status = "legacy" if result.legacy_warnings else "ok"
        print(
            f"{status}: {result.csv_path.parent} | "
            f"rows={result.row_count} | dataset_hash={result.dataset_hash}"
        )
        for warning in result.legacy_warnings:
            print(f"  warning: {warning}")
    meta_output_dir = _meta_ensemble_output_dir(config)
    if not (meta_output_dir / "prediction_artifacts.json").exists():
        print(f"not run yet: {meta_output_dir}")


def run_ml_run_inventory(config):
    print("\nML RUN INVENTORY")
    print("mode=research | trading_impact=none")
    for source_dir in _artifact_source_dirs(config, require_exists=False):
        csv_path = source_dir / "prediction_artifacts.csv"
        metadata_path = source_dir / "prediction_artifacts.json"
        status = "missing"
        if csv_path.exists() and metadata_path.exists():
            try:
                result = validate_prediction_artifact_dirs([source_dir])[0]
            except RuntimeError as exc:
                status = f"invalid: {exc}"
            else:
                status = (
                    "legacy"
                    if result.legacy_warnings
                    else f"complete rows={result.row_count} hash={result.dataset_hash}"
                )
        print(f"{source_dir}: {status}")


def run_ml_clean_incomplete_runs(config):
    report_dir = Path(config.get("reports", {}).get("ml_dir", "reports/ml"))
    incomplete = incomplete_ml_run_dirs(report_dir)
    print("\nML INCOMPLETE RUNS")
    print("mode=research | trading_impact=none")
    if not incomplete:
        print("No incomplete run directories found.")
        return
    for path in incomplete:
        print(path)
    print("No files were deleted. Remove listed directories manually if desired.")


def incomplete_ml_run_dirs(report_dir: Path) -> list[Path]:
    return [
        path
        for path in _artifact_child_dirs(report_dir)
        if path.name != "regime_transformer_meta_ensemble_v1"
        and _is_incomplete_run_dir(path)
    ]


def _is_incomplete_run_dir(path: Path) -> bool:
    required_files = (
        "metrics.json",
        "metadata.json",
        "dataset_audit.json",
        "prediction_artifacts.csv",
        "prediction_artifacts.json",
    )
    if not any(child.is_file() for child in path.iterdir()):
        return False
    if not all((path / name).exists() for name in required_files):
        return True
    return not _is_valid_source_artifact_dir(path)


def run_ml_model_contract_audit(config):
    output_dir = config.get("reports", {}).get(
        "ml_dir",
        config.get("ml", {}).get("output_dir", "reports/ml"),
    )
    markdown_path, json_path = write_model_contract_audit(output_dir)
    print("\nML MODEL CONTRACT AUDIT")
    print("mode=research | trading_impact=none")
    print(f"Markdown: {markdown_path}")
    print(f"JSON: {json_path}")


def _update_source_leaderboard(
    config: dict[str, Any],
    completed_output_dir: Path,
) -> tuple[Path, Path]:
    report_dir = _leaderboard_report_dir(config, completed_output_dir)
    leaderboard_dir = report_dir / "regime_transformer_meta_ensemble_v1"
    source_dirs = _valid_source_leaderboard_dirs(report_dir)
    if completed_output_dir not in source_dirs and _is_valid_source_artifact_dir(
        completed_output_dir
    ):
        source_dirs.append(completed_output_dir)
    source_dirs = sorted(set(source_dirs))
    markdown_path = leaderboard_dir / "leaderboard.md"
    json_path = leaderboard_dir / "leaderboard.json"
    write_source_leaderboard(json_path, markdown_path, source_dirs)
    return markdown_path, json_path


def _leaderboard_report_dir(
    config: dict[str, Any],
    completed_output_dir: Path,
) -> Path:
    return Path(
        str(
            config.get("reports", {}).get(
                "ml_dir",
                completed_output_dir.parent,
            )
        )
    )


def _valid_source_leaderboard_dirs(report_dir: Path) -> list[Path]:
    return [
        child
        for child in _artifact_child_dirs(report_dir)
        if child.name != "regime_transformer_meta_ensemble_v1"
        and _is_valid_source_artifact_dir(child)
    ]


def _is_valid_source_artifact_dir(path: Path) -> bool:
    csv_path = path / "prediction_artifacts.csv"
    metadata_path = path / "prediction_artifacts.json"
    if not csv_path.exists() or not metadata_path.exists():
        return False
    try:
        result = validate_prediction_artifact_dirs([path])[0]
    except RuntimeError:
        return False
    return not result.legacy_warnings


def _artifact_source_dirs(
    config: dict[str, Any],
    *,
    require_exists: bool = True,
) -> list[Path]:
    ml_config = config.get("ml", {})
    explicit_dirs = ml_config.get("source_prediction_dirs")
    if explicit_dirs:
        source_dirs = [Path(str(path)) for path in explicit_dirs if path]
    else:
        output_dir = Path(
            str(
                ml_config.get(
                    "output_dir",
                    config.get("reports", {}).get("ml_dir", "reports/ml"),
                )
            )
        )
        if (output_dir / "prediction_artifacts.csv").exists():
            source_dirs = [output_dir]
        else:
            source_dirs = _artifact_child_dirs(output_dir)
    if not source_dirs:
        report_dir = Path(config.get("reports", {}).get("ml_dir", "reports/ml"))
        source_dirs = _artifact_child_dirs(report_dir)
    source_dirs = [
        path
        for path in source_dirs
        if path.name != "regime_transformer_meta_ensemble_v1"
    ]
    if require_exists:
        missing = [path for path in source_dirs if not path.exists()]
        if missing:
            raise RuntimeError(
                "Prediction artifact directories do not exist: "
                + ", ".join(str(path) for path in missing)
            )
    return source_dirs


def _meta_ensemble_output_dir(config: dict[str, Any]) -> Path:
    report_dir = Path(config.get("reports", {}).get("ml_dir", "reports/ml"))
    return report_dir / "regime_transformer_meta_ensemble_v1"


def _artifact_child_dirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [
        child
        for child in sorted(path.iterdir())
        if child.is_dir()
    ]
