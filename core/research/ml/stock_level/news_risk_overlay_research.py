from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable, Mapping

from core.research.ml.stock_level.news_risk_overlay import (
    DECISION_TIMESTAMP_COLUMNS,
    TIMESTAMP_COLUMNS,
    NewsRiskOverlayConfig,
    build_news_risk_labels,
    chronological_splits,
    evaluate_candidate,
    join_news_to_stock_alpha_observations,
    shadow_decision_row,
)
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir


LABEL_SOURCE_COLUMNS = (
    "actual_max_adverse_excursion",
    "forward_max_adverse_excursion",
    "max_adverse_excursion",
    "actual_forward_return_20d",
    "actual_forward_return_10d",
    "actual_forward_return_5d",
    "forward_return",
    "stop_hit_before_target",
)
RETURN_COLUMNS = (
    "actual_forward_return_10d",
    "actual_forward_return_5d",
    "forward_return",
)
PRICE_SCORE_COLUMNS = (
    "stock_level_predicted_forward_return_10d_elastic_net",
    "stock_level_predicted_forward_return_10d_gradient_boosting",
    "stock_level_predicted_forward_return_10d_random_forest",
    "stock_level_predicted_forward_return_10d_ridge",
    "predicted_forward_return_10d",
    "predicted_momentum_120d",
    "predicted_risk_adjusted_momentum",
)
EXCLUDED_FEATURE_PREFIXES = ("actual_", "forward_", "news_risk_", "price_only_", "price_plus_news_")
EXCLUDED_FEATURE_COLUMNS = {
    "symbol",
    "sector",
    "fold_id",
    "source",
    "source_feature_id",
    "source_model_type",
    "source_split",
    "source_dataset_hash",
    "true_stock_level_row",
    "decision_timestamp",
    "rebalance_date",
    "feature_date",
    "date",
    "news_feature_timestamp",
    "news_coverage_status",
    "news_missing_coverage",
    "news_has_coverage_30d",
    "news_news_has_coverage_30d",
}


@dataclass(frozen=True)
class NewsRiskResearchPaths:
    output_dir: Path
    dataset_csv_path: Path
    coverage_json_path: Path
    leakage_json_path: Path
    metrics_json_path: Path
    portfolio_json_path: Path
    accounting_json_path: Path
    accounting_audit_json_path: Path
    equity_curve_csv_path: Path
    drawdown_curve_csv_path: Path
    trade_ledger_csv_path: Path
    daily_equity_price_only_csv_path: Path
    daily_equity_news_cash_csv_path: Path
    daily_equity_news_replacement_csv_path: Path
    daily_equity_news_reduced_size_csv_path: Path
    open_trade_portfolio_json_path: Path
    replay_risk_metrics_json_path: Path
    action_attribution_json_path: Path
    score_direction_audit_json_path: Path
    news_score_deciles_csv_path: Path
    news_score_direction_report_json_path: Path
    news_score_direction_summary_md_path: Path
    replay_action_attribution_json_path: Path
    event_category_analysis_json_path: Path
    contrarian_strategy_comparison_json_path: Path
    contrarian_trade_ledger_csv_path: Path
    price_stabilisation_comparison_json_path: Path
    resilience_filter_analysis_json_path: Path
    extreme_event_archive_csv_path: Path
    extreme_event_memory_report_json_path: Path
    cost_scenario_comparison_json_path: Path
    parallel_execution_report_json_path: Path
    replay_assumptions_json_path: Path
    replay_data_audit_json_path: Path
    shadow_csv_path: Path
    manifest_json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class NewsRiskParallelConfig:
    enabled: bool
    requested_workers: int | None
    actual_workers: int
    backend: str
    min_items: int
    chunk_size: int
    batch_limit: int
    progress: bool
    cpu_count: int
    fallback_reason: str | None = None


@dataclass(frozen=True)
class NewsRiskParallelBenchmarkPaths:
    output_dir: Path
    report_json_path: Path


@dataclass(frozen=True)
class NewsRiskOverlayInspection:
    output_dir: Path
    summary: dict[str, Any]
    artifact_status: list[dict[str, Any]]


def write_stock_alpha_news_risk_overlay_research(
    config: Mapping[str, Any],
) -> NewsRiskResearchPaths:
    run_started = time.perf_counter()
    ml = dict(config.get("ml", {}) or {})
    parallel_config = _parallel_config(ml)
    parallel_report = _parallel_report_skeleton(parallel_config)
    output_dir = Path(
        str(
            ml.get(
                "stock_alpha_news_risk_overlay_output_dir",
                "research-results/stock_alpha_news_risk_overlay",
            )
        )
    )
    with _timed_phase(parallel_report, "sequential_preflight"):
        _check_output_disk_space(output_dir, ml)
    output_dir.mkdir(parents=True, exist_ok=True)

    with _timed_phase(parallel_report, "input_loading"):
        price_path = _locate_price_candidates(config)
        news_path = _locate_news_features(config)
        price_rows = _read_csv(price_path)
        news_rows = _read_csv(news_path)
        _validate_source_rows(price_rows, news_rows, price_path, news_path)

    overlay_config = NewsRiskOverlayConfig(
        decision_timestamp_column=_optional_str(
            ml.get("stock_alpha_news_risk_overlay_decision_timestamp_column")
        ),
        news_timestamp_preference=tuple(dict.fromkeys((*TIMESTAMP_COLUMNS, *DECISION_TIMESTAMP_COLUMNS))),
        adverse_return_threshold=float(
            ml.get("stock_alpha_news_risk_overlay_adverse_return_threshold", -0.05)
        ),
        block_threshold=float(ml.get("stock_alpha_news_risk_overlay_block_threshold", 0.70)),
        reduce_threshold=float(ml.get("stock_alpha_news_risk_overlay_reduce_threshold", 0.50)),
        reduce_multiplier=float(ml.get("stock_alpha_news_risk_overlay_reduce_multiplier", 0.50)),
        model_version=str(ml.get("stock_alpha_news_risk_overlay_model_version", "news-risk-overlay-research-v1")),
    )
    with _timed_phase(parallel_report, "point_in_time_join"):
        joined, leakage = join_news_to_stock_alpha_observations(price_rows, news_rows, overlay_config)
        labeled = build_news_risk_labels(joined, overlay_config)
    coverage = _coverage_report(labeled, leakage)
    min_coverage = float(ml.get("stock_alpha_news_risk_overlay_min_coverage_ratio", 0.01))
    if coverage["covered_row_count"] <= 0 or coverage["row_coverage_ratio"] < min_coverage:
        raise ValueError(
            "stock-alpha news risk overlay coverage unavailable: "
            f"covered_row_count={coverage['covered_row_count']} "
            f"row_coverage_ratio={coverage['row_coverage_ratio']:.4f} "
            f"required_min={min_coverage:.4f}"
        )
    if leakage.get("leakage_violation_count", 0):
        raise ValueError("timestamp leakage detected in joined news features")

    folds = int(ml.get("stock_alpha_news_risk_overlay_walk_forward_folds", 3))
    embargo_days = int(ml.get("stock_alpha_news_risk_overlay_embargo_days", 0))
    learning_rate = float(ml.get("stock_alpha_news_risk_overlay_learning_rate", 0.05))
    epochs = int(ml.get("stock_alpha_news_risk_overlay_epochs", 60))
    l2 = float(ml.get("stock_alpha_news_risk_overlay_l2", 0.01))
    max_train_rows = int(ml.get("stock_alpha_news_risk_overlay_max_train_rows", 12000))
    max_features = int(ml.get("stock_alpha_news_risk_overlay_max_features", 48))
    dataset_max_rows = int(ml.get("stock_alpha_news_risk_overlay_dataset_max_rows", 5000))
    shadow_max_rows = int(ml.get("stock_alpha_news_risk_overlay_shadow_max_rows", 5000))
    audit_detail_max_rows = int(ml.get("stock_alpha_news_risk_overlay_audit_detail_max_rows", 1000))
    price_score_column = _choose_column(
        labeled,
        _configured_first(ml.get("stock_alpha_news_risk_overlay_price_score_column"), PRICE_SCORE_COLUMNS),
        "price score",
    )
    return_column = _choose_column(
        labeled,
        _configured_first(ml.get("stock_alpha_news_risk_overlay_return_column"), RETURN_COLUMNS),
        "portfolio return",
    )
    price_feature_columns = _limit_features(
        _feature_columns(labeled, include_news=False),
        labeled,
        max_features=max_features,
    )
    price_news_feature_columns = _limit_features(
        _feature_columns(labeled, include_news=True),
        labeled,
        max_features=max_features,
        require_news=True,
    )
    if not price_feature_columns:
        raise ValueError("no numeric price candidate features available for price-only baseline")
    if not any(column.startswith("news_") for column in price_news_feature_columns):
        raise ValueError("no numeric joined news features available for price-plus-news baseline")

    splits = chronological_splits(labeled, folds=folds, embargo_days=embargo_days)
    if not splits:
        raise ValueError("not enough timestamped rows for chronological walk-forward splits")

    with _timed_phase(parallel_report, "model_diagnostics"):
        price_metrics, price_probs = _walk_forward_logistic(
            labeled,
            price_feature_columns,
            splits,
            learning_rate=learning_rate,
            epochs=epochs,
            l2=l2,
            max_train_rows=max_train_rows,
        )
        news_metrics, news_probs = _walk_forward_logistic(
            labeled,
            price_news_feature_columns,
            splits,
            learning_rate=learning_rate,
            epochs=epochs,
            l2=l2,
            max_train_rows=max_train_rows,
        )
        _apply_probabilities(labeled, price_probs, "price_only_news_risk_probability")
        _apply_probabilities(labeled, news_probs, "price_plus_news_risk_probability")
        decision_rows = _apply_news_decisions(labeled, overlay_config, price_score_column)
        oos_rows = [row for index, row in enumerate(labeled) if index in news_probs]
    portfolio = _portfolio_comparison(
        oos_rows,
        price_score_column=price_score_column,
        return_column=return_column,
        top_n=int(ml.get("stock_alpha_news_risk_overlay_portfolio_top_n", 25)),
        starting_equity=float(ml.get("stock_alpha_news_risk_overlay_starting_equity", 1.0)),
        transaction_cost_bps=float(ml.get("stock_alpha_news_risk_overlay_transaction_cost_bps", 0.0)),
        slippage_bps=float(ml.get("stock_alpha_news_risk_overlay_slippage_bps", 0.0)),
    )
    with _timed_phase(parallel_report, "replay"):
        replay = _build_open_trade_replay(
            oos_rows,
            config=ml,
            price_score_column=price_score_column,
            output_dir=output_dir,
            parallel_config=parallel_config,
            parallel_report=parallel_report,
        )
    with _timed_phase(parallel_report, "model_diagnostic_reports"):
        score_direction_audit = _score_direction_audit(
            rows=oos_rows,
            config=overlay_config,
            target_column="news_risk_label",
        )
        _assert_score_direction_contract(score_direction_audit, oos_rows)
        score_decile_rows, score_direction_report = _news_score_decile_diagnostics(
            oos_rows,
            replay["trade_ledger"],
            price_score_column=price_score_column,
        )
        replay_action_attribution = _replay_action_attribution(
            replay["action_events"],
            replay["trade_ledger"],
            replay.get("hypothetical_trade_ledger", []),
        )
        event_category_analysis = _event_category_analysis(oos_rows, replay["trade_ledger"])
        contrarian_report = _contrarian_strategy_report(
            replay["risk_metrics"],
            replay.get("variant_settings", {}),
            ml,
        )
        price_stabilisation = _price_stabilisation_report(ml)
        resilience_analysis = _resilience_filter_analysis(oos_rows)
        extreme_archive_rows, extreme_memory_report = _extreme_event_archive(oos_rows, ml)
    with _timed_phase(parallel_report, "cost_scenarios"):
        cost_scenarios = _cost_scenario_comparison(
            oos_rows,
            bars_by_symbol=replay["bars_by_symbol"],
            price_score_column=price_score_column,
            base_replay_config=replay["replay_config"],
            parallel_config=parallel_config,
            parallel_report=parallel_report,
        )

    paths = NewsRiskResearchPaths(
        output_dir=output_dir,
        dataset_csv_path=output_dir / "stock_alpha_news_risk_overlay_dataset.csv",
        coverage_json_path=output_dir / "coverage_report.json",
        leakage_json_path=output_dir / "leakage_report.json",
        metrics_json_path=output_dir / "logistic_regression_metrics.json",
        portfolio_json_path=output_dir / "price_vs_news_portfolio_report.json",
        accounting_json_path=output_dir / "accounting_definitions.json",
        accounting_audit_json_path=output_dir / "accounting_audit.json",
        equity_curve_csv_path=output_dir / "equity_curve.csv",
        drawdown_curve_csv_path=output_dir / "drawdown_curve.csv",
        trade_ledger_csv_path=output_dir / "trade_ledger.csv",
        daily_equity_price_only_csv_path=output_dir / "daily_equity_price_only.csv",
        daily_equity_news_cash_csv_path=output_dir / "daily_equity_news_cash.csv",
        daily_equity_news_replacement_csv_path=output_dir / "daily_equity_news_replacement.csv",
        daily_equity_news_reduced_size_csv_path=output_dir / "daily_equity_news_reduced_size.csv",
        open_trade_portfolio_json_path=output_dir / "portfolio_comparison.json",
        replay_risk_metrics_json_path=output_dir / "risk_metrics.json",
        action_attribution_json_path=output_dir / "action_attribution.json",
        score_direction_audit_json_path=output_dir / "score_direction_audit.json",
        news_score_deciles_csv_path=output_dir / "news_score_deciles.csv",
        news_score_direction_report_json_path=output_dir / "news_score_direction_report.json",
        news_score_direction_summary_md_path=output_dir / "news_score_direction_summary.md",
        replay_action_attribution_json_path=output_dir / "replay_action_attribution.json",
        event_category_analysis_json_path=output_dir / "event_category_analysis.json",
        contrarian_strategy_comparison_json_path=output_dir / "contrarian_strategy_comparison.json",
        contrarian_trade_ledger_csv_path=output_dir / "contrarian_trade_ledger.csv",
        price_stabilisation_comparison_json_path=output_dir / "price_stabilisation_comparison.json",
        resilience_filter_analysis_json_path=output_dir / "resilience_filter_analysis.json",
        extreme_event_archive_csv_path=output_dir / "extreme_event_archive.csv",
        extreme_event_memory_report_json_path=output_dir / "extreme_event_memory_report.json",
        cost_scenario_comparison_json_path=output_dir / "cost_scenario_comparison.json",
        parallel_execution_report_json_path=output_dir / "parallel_execution_report.json",
        replay_assumptions_json_path=output_dir / "replay_assumptions.json",
        replay_data_audit_json_path=output_dir / "replay_data_audit.json",
        shadow_csv_path=output_dir / "shadow_decision_log.csv",
        manifest_json_path=output_dir / "model_manifest.json",
        markdown_path=output_dir / "README.md",
    )
    metrics = {
        "model_type": "in_repo_logistic_regression",
        "chronological_walk_forward": True,
        "transformer_trained": False,
        "paper_orders_enabled": False,
        "price_only": price_metrics,
        "price_plus_news": news_metrics,
        "price_feature_columns": price_feature_columns,
        "price_plus_news_feature_columns": price_news_feature_columns,
    }
    manifest = {
        "mode": "ml-stock-alpha-news-risk-overlay-research",
        "research_only": True,
        "trading_impact": "none",
        "price_candidates_path": str(price_path),
        "news_features_path": str(news_path),
        "output_dir": str(output_dir),
        "price_score_column": price_score_column,
        "return_column": return_column,
        "label_source_columns": [column for column in LABEL_SOURCE_COLUMNS if column in labeled[0]],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "full_joined_row_count": len(labeled),
        "dataset_csv_row_count": len(_limited_rows(labeled, dataset_max_rows)),
        "shadow_csv_row_count": len(_limited_rows(decision_rows, shadow_max_rows)),
        "transformer_trained": False,
        "paper_orders_enabled": False,
    }
    with _timed_phase(parallel_report, "report_writing"):
        _write_csv(paths.dataset_csv_path, _limited_rows(labeled, dataset_max_rows))
        _write_csv(paths.shadow_csv_path, _limited_rows(decision_rows, shadow_max_rows))
        _write_json(paths.coverage_json_path, coverage)
        _write_json(paths.leakage_json_path, _limited_audit_details(leakage, audit_detail_max_rows))
        _write_json(paths.metrics_json_path, metrics)
        _write_json(paths.portfolio_json_path, portfolio)
        _write_json(paths.accounting_json_path, _accounting_definitions())
        _write_json(paths.accounting_audit_json_path, _accounting_audit(portfolio))
        _write_csv(paths.equity_curve_csv_path, portfolio["equity_curve"])
        _write_csv(paths.drawdown_curve_csv_path, portfolio["drawdown_curve"])
        _write_csv(paths.trade_ledger_csv_path, replay["trade_ledger"])
        _write_csv(paths.daily_equity_price_only_csv_path, replay["daily_equity"]["price_only"])
        _write_csv(paths.daily_equity_news_cash_csv_path, replay["daily_equity"]["news_cash"])
        _write_csv(paths.daily_equity_news_replacement_csv_path, replay["daily_equity"]["news_replacement"])
        _write_csv(paths.daily_equity_news_reduced_size_csv_path, replay["daily_equity"]["news_reduced_size"])
        _write_json(paths.open_trade_portfolio_json_path, replay["portfolio_comparison"])
        _write_json(paths.replay_risk_metrics_json_path, replay["risk_metrics"])
        _write_json(paths.action_attribution_json_path, replay["action_attribution"])
        _write_json(paths.score_direction_audit_json_path, score_direction_audit)
        _write_csv(paths.news_score_deciles_csv_path, score_decile_rows)
        _write_json(paths.news_score_direction_report_json_path, score_direction_report)
        paths.news_score_direction_summary_md_path.write_text(
            _score_direction_markdown(score_direction_report),
            encoding="utf-8",
        )
        _write_json(paths.replay_action_attribution_json_path, replay_action_attribution)
        _write_json(paths.event_category_analysis_json_path, event_category_analysis)
        _write_json(paths.contrarian_strategy_comparison_json_path, contrarian_report)
        _write_csv(paths.contrarian_trade_ledger_csv_path, replay.get("contrarian_trade_ledger", []))
        _write_json(paths.price_stabilisation_comparison_json_path, price_stabilisation)
        _write_json(paths.resilience_filter_analysis_json_path, resilience_analysis)
        _write_csv(paths.extreme_event_archive_csv_path, extreme_archive_rows)
        _write_json(paths.extreme_event_memory_report_json_path, extreme_memory_report)
        _write_json(paths.cost_scenario_comparison_json_path, cost_scenarios)
        _write_json(paths.replay_assumptions_json_path, replay["replay_assumptions"])
        _write_json(paths.replay_data_audit_json_path, replay["replay_data_audit"])
        _write_json(paths.manifest_json_path, manifest)
        paths.markdown_path.write_text(
            _markdown(
                manifest,
                coverage,
                metrics,
                portfolio,
                replay,
                score_direction_report,
                contrarian_report,
                cost_scenarios,
            ),
            encoding="utf-8",
        )
    parallel_report["elapsed_seconds_total"] = time.perf_counter() - run_started
    parallel_report["determinism_status"] = _parallel_determinism_status(parallel_report)
    _write_json(paths.parallel_execution_report_json_path, parallel_report)
    return paths


def write_stock_alpha_news_risk_overlay_parallel_benchmark(
    config: Mapping[str, Any],
) -> NewsRiskParallelBenchmarkPaths:
    ml = dict(config.get("ml", {}) or {})
    output_dir = Path(
        str(
            ml.get(
                "stock_alpha_news_risk_overlay_output_dir",
                "research-results/stock_alpha_news_risk_overlay",
            )
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    price_path = _locate_price_candidates(config)
    price_rows = _read_csv(price_path)
    if not price_rows:
        raise ValueError(f"price candidates are empty: {price_path}")
    symbol_limit = int(ml.get("news_risk_parallel_benchmark_symbol_limit", 50))
    symbols = sorted(dict.fromkeys(str(row.get("symbol", "")).upper() for row in price_rows if row.get("symbol")))[: max(1, symbol_limit)]
    processed_root = Path(str(ml.get("stock_alpha_news_risk_overlay_market_data_root", "data/processed")))
    sequential_report = _parallel_report_skeleton(_parallel_config({"news_risk_parallel_enabled": False}))
    sequential_started = time.perf_counter()
    sequential_bars, sequential_audit = _load_daily_price_bars(
        symbols,
        processed_root,
        parallel_config=_parallel_config({"news_risk_parallel_enabled": False}),
        parallel_report=sequential_report,
    )
    sequential_runtime = time.perf_counter() - sequential_started
    parallel_config = _parallel_config({**ml, "news_risk_parallel_enabled": True})
    parallel_report = _parallel_report_skeleton(parallel_config)
    parallel_started = time.perf_counter()
    parallel_bars, parallel_audit = _load_daily_price_bars(
        symbols,
        processed_root,
        parallel_config=parallel_config,
        parallel_report=parallel_report,
    )
    parallel_runtime = time.perf_counter() - parallel_started
    equivalent = _bar_sets_equal(sequential_bars, parallel_bars)
    report = {
        "schema_name": "stock_alpha_news_risk_overlay_parallel_benchmark",
        "schema_version": "1.0",
        "research_only": True,
        "broker_invoked": False,
        "orders_submitted": False,
        "price_candidates_path": str(price_path),
        "processed_root": str(processed_root),
        "symbol_limit": symbol_limit,
        "symbol_count": len(symbols),
        "symbols": symbols,
        "sequential_runtime_seconds": sequential_runtime,
        "parallel_runtime_seconds": parallel_runtime,
        "speedup_ratio": (sequential_runtime / parallel_runtime) if equivalent and parallel_runtime > 0 else None,
        "speedup_claim_allowed": equivalent,
        "worker_count": parallel_config.actual_workers,
        "backend": parallel_config.backend,
        "output_equivalence_result": equivalent,
        "sequential_audit": sequential_audit,
        "parallel_audit": parallel_audit,
        "sequential_parallel_report": sequential_report,
        "parallel_execution_report": parallel_report,
    }
    path = output_dir / "parallel_benchmark_report.json"
    _write_json(path, report)
    return NewsRiskParallelBenchmarkPaths(output_dir=output_dir, report_json_path=path)


def inspect_stock_alpha_news_risk_overlay_results(
    config: Mapping[str, Any],
) -> NewsRiskOverlayInspection:
    ml = dict(config.get("ml", {}) or {})
    output_dir = Path(
        str(
            ml.get(
                "stock_alpha_news_risk_overlay_output_dir",
                "research-results/stock_alpha_news_risk_overlay",
            )
        )
    )
    summary, artifact_status = _build_executive_summary(output_dir)
    return NewsRiskOverlayInspection(
        output_dir=output_dir,
        summary=summary,
        artifact_status=artifact_status,
    )


def format_news_risk_overlay_summary(
    summary: Mapping[str, Any],
    artifact_status: list[Mapping[str, Any]],
    *,
    mode: str = "summary",
) -> str:
    if mode == "json":
        return json.dumps(
            {"summary": summary, "artifact_status": artifact_status},
            indent=2,
            sort_keys=True,
            default=str,
        )
    if mode == "artifact-list":
        return "\n".join(
            [
                "STOCK-ALPHA NEWS RISK OVERLAY ARTIFACTS",
                *[
                    f"{row['status']:>18}  {row['name']}  {row['path']}"
                    for row in artifact_status
                ],
            ]
        )
    lines = _summary_lines(summary)
    if mode == "verbose":
        lines.extend(["", "Artifacts:"])
        lines.extend(f"- {row['name']}: {row['status']} ({row['path']})" for row in artifact_status)
    return "\n".join(lines)


def _locate_price_candidates(config: Mapping[str, Any]) -> Path:
    ml = dict(config.get("ml", {}) or {})
    configured = _optional_path(ml.get("stock_alpha_news_risk_overlay_price_candidates_path"))
    if configured:
        return _existing(configured, "configured price candidates")
    output = stock_alpha_output_dir(config)
    candidates = [
        output / "enriched" / "stock_level_model_oos_predictions.csv",
        output / "baseline" / "stock_level_model_oos_predictions.csv",
        output / "stock_level_model_oos_predictions.csv",
        output / "stock_level_prediction_artifacts_enriched.csv",
        output / "stock_level_prediction_artifacts.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "historical price-strategy candidates not found; set "
        "ml.stock_alpha_news_risk_overlay_price_candidates_path"
    )


def _locate_news_features(config: Mapping[str, Any]) -> Path:
    ml = dict(config.get("ml", {}) or {})
    configured = _optional_path(
        ml.get("stock_alpha_news_risk_overlay_news_features_path")
        or ml.get("stock_alpha_news_features_path")
    )
    if configured:
        return _existing(configured, "configured news features")
    candidates = [
        Path("reports/ml/benchmark/regime_transformer_meta_ensemble_v1/news_transformer_features_120mo_v1/news_transformer_event_features.csv"),
        stock_alpha_output_dir(config) / "news_features" / "stock_alpha_news_features.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "stock-alpha news features not found; set "
        "ml.stock_alpha_news_risk_overlay_news_features_path"
    )


def _validate_source_rows(
    price_rows: list[dict[str, str]],
    news_rows: list[dict[str, str]],
    price_path: Path,
    news_path: Path,
) -> None:
    if not price_rows:
        raise ValueError(f"price candidates are empty: {price_path}")
    if not news_rows:
        raise ValueError(f"news features are empty: {news_path}")
    price_columns = set(price_rows[0])
    news_columns = set(news_rows[0])
    if "symbol" not in price_columns:
        raise ValueError(f"price candidates missing symbol column: {price_path}")
    if "symbol" not in news_columns:
        raise ValueError(f"news features missing symbol column: {news_path}")
    if not price_columns.intersection(DECISION_TIMESTAMP_COLUMNS):
        raise ValueError(f"price candidates missing decision timestamp column: {price_path}")
    if not news_columns.intersection((*TIMESTAMP_COLUMNS, *DECISION_TIMESTAMP_COLUMNS)):
        raise ValueError(f"news features missing point-in-time timestamp column: {news_path}")
    if not price_columns.intersection(LABEL_SOURCE_COLUMNS):
        raise ValueError(f"price candidates missing configurable adverse-outcome label source: {price_path}")


def _walk_forward_logistic(
    rows: list[Mapping[str, Any]],
    feature_columns: list[str],
    splits: list[tuple[list[int], list[int]]],
    *,
    learning_rate: float,
    epochs: int,
    l2: float,
    max_train_rows: int,
) -> tuple[dict[str, Any], dict[int, float]]:
    probabilities: dict[int, float] = {}
    fold_reports = []
    for fold_id, (train_index, test_index) in enumerate(splits, start=1):
        if max_train_rows > 0:
            train_index = train_index[-max_train_rows:]
        train = [rows[index] for index in train_index]
        test = [rows[index] for index in test_index]
        labels = [int(row["news_risk_label"]) for row in train]
        if len(set(labels)) < 2:
            continue
        model = _fit_logistic(train, feature_columns, learning_rate=learning_rate, epochs=epochs, l2=l2)
        fold_probs = [_predict_logistic(model, row) for row in test]
        for index, probability in zip(test_index, fold_probs):
            probabilities[index] = probability
        fold_reports.append(
            {
                "fold_id": fold_id,
                "train_rows": len(train),
                "test_rows": len(test),
                **_classification_metrics([int(row["news_risk_label"]) for row in test], fold_probs),
            }
        )
    if not probabilities:
        raise ValueError("walk-forward logistic regression produced no out-of-sample predictions")
    y_true = [int(rows[index]["news_risk_label"]) for index in probabilities]
    y_prob = [probabilities[index] for index in probabilities]
    return {
        "oos_rows": len(probabilities),
        "folds_completed": len(fold_reports),
        "folds": fold_reports,
        **_classification_metrics(y_true, y_prob),
    }, probabilities


def _fit_logistic(
    rows: list[Mapping[str, Any]],
    feature_columns: list[str],
    *,
    learning_rate: float,
    epochs: int,
    l2: float,
) -> dict[str, Any]:
    matrix = [[_number(row.get(column)) or 0.0 for column in feature_columns] for row in rows]
    labels = [float(row["news_risk_label"]) for row in rows]
    means = [mean(column) for column in zip(*matrix)]
    stdevs = [pstdev(column) or 1.0 for column in zip(*matrix)]
    weights = [0.0 for _ in feature_columns]
    intercept = 0.0
    for _ in range(max(1, epochs)):
        grad = [0.0 for _ in weights]
        intercept_grad = 0.0
        for features, label in zip(matrix, labels):
            scaled = [(value - means[i]) / stdevs[i] for i, value in enumerate(features)]
            prediction = _sigmoid(intercept + sum(w * x for w, x in zip(weights, scaled)))
            error = prediction - label
            intercept_grad += error
            for i, value in enumerate(scaled):
                grad[i] += error * value
        n = max(len(matrix), 1)
        intercept -= learning_rate * intercept_grad / n
        for i in range(len(weights)):
            weights[i] -= learning_rate * ((grad[i] / n) + l2 * weights[i])
    return {"columns": feature_columns, "means": means, "stdevs": stdevs, "weights": weights, "intercept": intercept}


def _predict_logistic(model: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    total = float(model["intercept"])
    for column, avg, scale, weight in zip(model["columns"], model["means"], model["stdevs"], model["weights"]):
        total += float(weight) * (((_number(row.get(column)) or 0.0) - float(avg)) / float(scale))
    return _sigmoid(total)


def _classification_metrics(y_true: list[int], y_prob: list[float]) -> dict[str, float]:
    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "brier": 0.0, "roc_auc": 0.0}
    predictions = [1 if value >= 0.5 else 0 for value in y_prob]
    tp = sum(1 for y, p in zip(y_true, predictions) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, predictions) if y == 0 and p == 1)
    tn = sum(1 for y, p in zip(y_true, predictions) if y == 0 and p == 0)
    fn = sum(1 for y, p in zip(y_true, predictions) if y == 1 and p == 0)
    return {
        "accuracy": (tp + tn) / len(y_true),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "brier": mean([(p - y) ** 2 for y, p in zip(y_true, y_prob)]),
        "roc_auc": _roc_auc(y_true, y_prob),
        "positive_rate": sum(y_true) / len(y_true),
    }


def _roc_auc(y_true: list[int], y_prob: list[float]) -> float:
    positive_count = sum(1 for label in y_true if label == 1)
    negative_count = len(y_true) - positive_count
    if not positive_count or not negative_count:
        return 0.0
    ranked = sorted(zip(y_prob, y_true), key=lambda item: item[0])
    rank_sum = 0.0
    rank = 1
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (rank + end) / 2.0
        rank_sum += average_rank * sum(1 for _, label in ranked[index:end] if label == 1)
        rank = end + 1
        index = end
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def _apply_probabilities(rows: list[dict[str, Any]], probabilities: Mapping[int, float], column: str) -> None:
    for index, value in probabilities.items():
        rows[index][column] = value


def _apply_news_decisions(
    rows: list[dict[str, Any]],
    config: NewsRiskOverlayConfig,
    price_score_column: str,
) -> list[dict[str, Any]]:
    decision_rows = []
    for row in rows:
        probability = _number(row.get("price_plus_news_risk_probability"))
        decision = evaluate_candidate(
            symbol=str(row.get("symbol", "")),
            decision_timestamp=_timestamp(row),
            base_position_size=1.0,
            price_model_score=_number(row.get(price_score_column)) or 0.0,
            recent_features=row,
            risk_probability=probability,
            config=config,
        )
        row["news_action"] = decision.action
        row["news_position_multiplier"] = decision.recommended_position_multiplier
        decision_rows.append(
            shadow_decision_row(
                timestamp=_timestamp(row),
                symbol=str(row.get("symbol", "")),
                price_score=_number(row.get(price_score_column)) or 0.0,
                price_only_position_size=1.0,
                decision=decision,
                order_submitted=False,
                relevant_news_features={key: row[key] for key in row if key.startswith("news_")},
            )
        )
    return decision_rows


def _portfolio_comparison(
    rows: Iterable[Mapping[str, Any]],
    *,
    price_score_column: str,
    return_column: str,
    top_n: int,
    starting_equity: float,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    by_date: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_date.setdefault(str(row.get("decision_timestamp", ""))[:10], []).append(row)
    control_periods = []
    experiment_periods = []
    for date_key in sorted(by_date):
        ranked = sorted(
            by_date[date_key],
            key=lambda row: _number(row.get(price_score_column)) or float("-inf"),
            reverse=True,
        )[: max(1, top_n)]
        if not ranked:
            continue
        control_periods.append(
            _period_accounting_row(
                date_key,
                ranked,
                return_column=return_column,
                overlay=False,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
            )
        )
        experiment_periods.append(
            _period_accounting_row(
                date_key,
                ranked,
                return_column=return_column,
                overlay=True,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
            )
        )
    control_curve = _equity_curve(control_periods, starting_equity=starting_equity, prefix="price_only")
    experiment_curve = _equity_curve(
        experiment_periods,
        starting_equity=starting_equity,
        prefix="price_plus_news",
    )
    control_stats = _portfolio_stats(control_curve, return_column="price_only_period_return_net")
    experiment_stats = _portfolio_stats(experiment_curve, return_column="price_plus_news_period_return_net")
    return {
        "price_score_column": price_score_column,
        "return_column": return_column,
        "top_n": top_n,
        "starting_equity": starting_equity,
        "transaction_cost_bps": transaction_cost_bps,
        "slippage_bps": slippage_bps,
        "accounting_approximation": (
            "Decision-level marked-to-market approximation using realized forward returns "
            "from candidate artifacts. Overlapping holdings are approximated by one "
            "equal-weight decision-period basket per timestamp because the artifacts do "
            "not include full open-position daily mark-to-market paths."
        ),
        "price_only": control_stats,
        "price_plus_news": experiment_stats,
        "news_overlay_lowered_drawdown": (
            experiment_stats["maximum_drawdown"] > control_stats["maximum_drawdown"]
        ),
        "incremental_total_return_decimal": (
            experiment_stats["total_return_decimal"] - control_stats["total_return_decimal"]
        ),
        "equity_curve": _merge_curves(control_curve, experiment_curve),
        "drawdown_curve": _drawdown_curve(control_curve, experiment_curve),
    }


def _period_accounting_row(
    date_key: str,
    rows: list[Mapping[str, Any]],
    *,
    return_column: str,
    overlay: bool,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    # trade_return_net is the candidate realized forward return after applying
    # overlay exposure and subtracting one-way transaction cost plus slippage.
    # It is not a portfolio total return until basket returns are compounded
    # through _equity_curve as ending_equity / starting_equity - 1.
    selected = []
    total_cost_bps = transaction_cost_bps + slippage_bps
    for row in rows:
        multiplier = float(row.get("news_position_multiplier", 1.0) or 0.0) if overlay else 1.0
        gross_return = (_number(row.get(return_column)) or 0.0) * multiplier
        trade_cost = abs(multiplier) * total_cost_bps / 10_000.0
        selected.append(
            {
                "symbol": row.get("symbol", ""),
                "gross_return": gross_return,
                "trade_return_net": gross_return - trade_cost,
                "gross_exposure": abs(multiplier),
                "transaction_cost": trade_cost,
                "max_adverse_excursion": _adverse_excursion(row),
            }
        )
    denominator = max(len(selected), 1)
    return {
        "date": date_key,
        "period_return_net": sum(row["trade_return_net"] for row in selected) / denominator,
        "gross_exposure": sum(row["gross_exposure"] for row in selected) / denominator,
        "net_exposure": sum(row["gross_exposure"] for row in selected) / denominator,
        "transaction_costs": sum(row["transaction_cost"] for row in selected) / denominator,
        "number_of_positions": sum(row["gross_exposure"] > 0 for row in selected),
        "worst_trade": min((row["trade_return_net"] for row in selected), default=0.0),
        "maximum_adverse_excursion": min(
            (row["max_adverse_excursion"] for row in selected if row["max_adverse_excursion"] is not None),
            default=0.0,
        ),
    }


def _equity_curve(
    period_rows: list[Mapping[str, Any]],
    *,
    starting_equity: float,
    prefix: str,
) -> list[dict[str, Any]]:
    equity = starting_equity
    peak = starting_equity
    curve = []
    previous_exposure = 0.0
    drawdown_duration = 0
    for row in period_rows:
        period_return = float(row["period_return_net"])
        equity *= 1.0 + period_return
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0 if peak else 0.0
        drawdown_duration = drawdown_duration + 1 if drawdown < 0 else 0
        exposure = float(row["gross_exposure"])
        curve.append(
            {
                "date": row["date"],
                f"{prefix}_period_return_net": period_return,
                f"{prefix}_ending_equity": equity,
                f"{prefix}_drawdown": drawdown,
                f"{prefix}_drawdown_duration": drawdown_duration,
                f"{prefix}_gross_exposure": exposure,
                f"{prefix}_net_exposure": float(row["net_exposure"]),
                f"{prefix}_turnover": abs(exposure - previous_exposure),
                f"{prefix}_transaction_costs": float(row["transaction_costs"]),
                f"{prefix}_number_of_positions": int(row["number_of_positions"]),
                f"{prefix}_worst_trade": float(row["worst_trade"]),
                f"{prefix}_maximum_adverse_excursion": float(row["maximum_adverse_excursion"]),
            }
        )
        previous_exposure = exposure
    return curve


def _portfolio_stats(curve: list[Mapping[str, Any]], *, return_column: str) -> dict[str, float]:
    if not curve:
        return {
            "periods": 0,
            "starting_equity": 0.0,
            "ending_equity": 0.0,
            "total_return_decimal": 0.0,
            "total_return_percent": 0.0,
            "wealth_multiple": 0.0,
            "CAGR": 0.0,
            "annualised_volatility": 0.0,
            "maximum_drawdown": 0.0,
            "average_drawdown": 0.0,
            "longest_drawdown_duration": 0.0,
            "Sharpe_ratio": 0.0,
            "Sortino_ratio": 0.0,
            "Calmar_ratio": 0.0,
            "worst_day": 0.0,
            "worst_trade": 0.0,
            "maximum_adverse_excursion": 0.0,
            "expected_shortfall_CVaR_5pct": 0.0,
            "average_gross_exposure": 0.0,
            "average_net_exposure": 0.0,
            "turnover": 0.0,
            "transaction_costs": 0.0,
            "number_of_positions": 0.0,
        }
    prefix = return_column.replace("_period_return_net", "")
    returns = [float(row[return_column]) for row in curve]
    starting_equity = float(curve[0][f"{prefix}_ending_equity"]) / (1.0 + returns[0])
    ending_equity = float(curve[-1][f"{prefix}_ending_equity"])
    wealth_multiple = ending_equity / starting_equity if starting_equity else 0.0
    total_return_decimal = wealth_multiple - 1.0
    periods_per_year = 252.0
    years = max(len(returns) / periods_per_year, 1.0 / periods_per_year)
    downside = [min(value, 0.0) for value in returns]
    volatility = pstdev(returns) * math.sqrt(periods_per_year) if len(returns) > 1 else 0.0
    downside_volatility = pstdev(downside) * math.sqrt(periods_per_year) if len(downside) > 1 else 0.0
    average_period_return = mean(returns)
    maximum_drawdown = min(float(row[f"{prefix}_drawdown"]) for row in curve)
    expected_shortfall = _expected_shortfall(returns)
    return {
        "periods": len(returns),
        "starting_equity": starting_equity,
        "ending_equity": ending_equity,
        "total_return_decimal": total_return_decimal,
        "total_return_percent": total_return_decimal * 100.0,
        "wealth_multiple": wealth_multiple,
        "CAGR": wealth_multiple ** (1.0 / years) - 1.0 if wealth_multiple > 0 else -1.0,
        "annualised_volatility": volatility,
        "maximum_drawdown": maximum_drawdown,
        "average_drawdown": mean(float(row[f"{prefix}_drawdown"]) for row in curve),
        "longest_drawdown_duration": max(float(row[f"{prefix}_drawdown_duration"]) for row in curve),
        "Sharpe_ratio": (average_period_return * periods_per_year) / volatility if volatility else 0.0,
        "Sortino_ratio": (average_period_return * periods_per_year) / downside_volatility if downside_volatility else 0.0,
        "Calmar_ratio": (wealth_multiple ** (1.0 / years) - 1.0) / abs(maximum_drawdown) if maximum_drawdown else 0.0,
        "worst_day": min(returns),
        "worst_trade": min(float(row[f"{prefix}_worst_trade"]) for row in curve),
        "maximum_adverse_excursion": min(float(row[f"{prefix}_maximum_adverse_excursion"]) for row in curve),
        "expected_shortfall_CVaR_5pct": expected_shortfall,
        "average_gross_exposure": mean(float(row[f"{prefix}_gross_exposure"]) for row in curve),
        "average_net_exposure": mean(float(row[f"{prefix}_net_exposure"]) for row in curve),
        "turnover": sum(float(row[f"{prefix}_turnover"]) for row in curve),
        "transaction_costs": sum(float(row[f"{prefix}_transaction_costs"]) for row in curve),
        "number_of_positions": sum(float(row[f"{prefix}_number_of_positions"]) for row in curve),
    }


def _feature_columns(rows: list[Mapping[str, Any]], *, include_news: bool) -> list[str]:
    columns = []
    all_columns = list(dict.fromkeys(key for row in rows for key in row))
    for column in all_columns:
        if column in EXCLUDED_FEATURE_COLUMNS or column in LABEL_SOURCE_COLUMNS:
            continue
        if any(column.startswith(prefix) for prefix in EXCLUDED_FEATURE_PREFIXES):
            continue
        if column.startswith("news_") != include_news and column.startswith("news_"):
            continue
        values = [_number(row.get(column)) for row in rows[:200]]
        if any(value is not None and math.isfinite(value) for value in values):
            columns.append(column)
    if include_news:
        price = _feature_columns(rows, include_news=False)
        news = [column for column in columns if column.startswith("news_")]
        return [*price, *news]
    return columns


def _merge_curves(
    control_curve: list[Mapping[str, Any]],
    experiment_curve: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for control, experiment in zip(control_curve, experiment_curve):
        rows.append({**control, **{k: v for k, v in experiment.items() if k != "date"}})
    return rows


def _drawdown_curve(
    control_curve: list[Mapping[str, Any]],
    experiment_curve: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for control, experiment in zip(control_curve, experiment_curve):
        rows.append(
            {
                "date": control["date"],
                "price_only_drawdown": control["price_only_drawdown"],
                "price_plus_news_drawdown": experiment["price_plus_news_drawdown"],
                "price_only_drawdown_duration": control["price_only_drawdown_duration"],
                "price_plus_news_drawdown_duration": experiment["price_plus_news_drawdown_duration"],
            }
        )
    return rows


def _expected_shortfall(returns: list[float], tail_fraction: float = 0.05) -> float:
    if not returns:
        return 0.0
    count = max(1, math.ceil(len(returns) * tail_fraction))
    return mean(sorted(returns)[:count])


def _adverse_excursion(row: Mapping[str, Any]) -> float | None:
    for column in (
        "actual_max_adverse_excursion",
        "forward_max_adverse_excursion",
        "max_adverse_excursion",
    ):
        value = _number(row.get(column))
        if value is not None:
            return value
    drawdown = _number(row.get("actual_future_drawdown"))
    if drawdown is not None:
        return -abs(drawdown)
    return None


def _limit_features(
    columns: list[str],
    rows: list[Mapping[str, Any]],
    *,
    max_features: int,
    require_news: bool = False,
) -> list[str]:
    if max_features <= 0 or len(columns) <= max_features:
        return columns
    scored = sorted(
        columns,
        key=lambda column: (
            sum(_number(row.get(column)) is not None for row in rows),
            column.startswith("news_"),
        ),
        reverse=True,
    )
    selected = scored[:max_features]
    if require_news and not any(column.startswith("news_") for column in selected):
        first_news = next((column for column in scored if column.startswith("news_")), None)
        if first_news:
            selected[-1] = first_news
    return list(dict.fromkeys(selected))


def _coverage_report(rows: list[Mapping[str, Any]], audit: Mapping[str, Any]) -> dict[str, Any]:
    total = len(rows)
    covered = sum(str(row.get("news_coverage_status")) == "COVERED" for row in rows)
    return {
        "stock_row_count": total,
        "covered_row_count": covered,
        "row_coverage_ratio": covered / max(total, 1),
        "label_positive_rate": sum(int(row.get("news_risk_label", 0)) for row in rows) / max(total, 1),
        "symbol_coverage": audit.get("symbol_coverage", {}),
        "date_coverage": audit.get("date_coverage", {}),
        "future_news_rows_rejected": audit.get("future_news_rows_rejected", 0),
        "leakage_violation_count": audit.get("leakage_violation_count", 0),
    }


def _choose_column(rows: list[Mapping[str, Any]], candidates: Iterable[str], label: str) -> str:
    available = set().union(*(row.keys() for row in rows))
    for column in candidates:
        if column in available and any(_number(row.get(column)) is not None for row in rows):
            return column
    raise ValueError(f"no usable {label} column found; tried {list(candidates)}")


def _build_open_trade_replay(
    rows: list[dict[str, Any]],
    *,
    config: Mapping[str, Any],
    price_score_column: str,
    output_dir: Path,
    parallel_config: NewsRiskParallelConfig | None = None,
    parallel_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("open-trade replay requires out-of-sample candidate rows")
    symbols = sorted({str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")})
    processed_root = Path(str(config.get("market_data", {}).get("processed_root", "data/processed"))) if isinstance(config.get("market_data"), Mapping) else Path(str(config.get("stock_alpha_news_risk_overlay_market_data_root", "data/processed")))
    with _timed_phase(parallel_report, "bar_loading"):
        bars_by_symbol, data_audit = _load_daily_price_bars(
            symbols,
            processed_root,
            parallel_config=parallel_config,
            parallel_report=parallel_report,
        )
    missing = sorted(set(symbols) - set(bars_by_symbol))
    min_coverage = float(config.get("stock_alpha_news_risk_overlay_replay_min_price_symbol_coverage", 0.95))
    coverage = 1.0 - (len(missing) / max(len(symbols), 1))
    if coverage < min_coverage:
        raise ValueError(
            "open-trade replay price-bar coverage unavailable: "
            f"coverage={coverage:.4f}, required_min={min_coverage:.4f}, "
            f"missing_symbols={missing[:20]}"
        )
    replay_config = {
        "starting_equity": float(config.get("stock_alpha_news_risk_overlay_replay_starting_equity", 1.0)),
        "top_n": int(config.get("stock_alpha_news_risk_overlay_replay_top_n", config.get("stock_alpha_news_risk_overlay_portfolio_top_n", 25))),
        "max_positions": int(config.get("stock_alpha_news_risk_overlay_replay_max_positions", config.get("stock_alpha_news_risk_overlay_portfolio_top_n", 25))),
        "max_position_weight": float(config.get("stock_alpha_news_risk_overlay_replay_max_position_weight", 0.05)),
        "max_holding_bars": int(config.get("stock_alpha_news_risk_overlay_replay_max_holding_bars", 10)),
        "entry_slippage_bps": float(config.get("stock_alpha_news_risk_overlay_replay_entry_slippage_bps", config.get("stock_alpha_news_risk_overlay_slippage_bps", 0.0))),
        "exit_slippage_bps": float(config.get("stock_alpha_news_risk_overlay_replay_exit_slippage_bps", config.get("stock_alpha_news_risk_overlay_slippage_bps", 0.0))),
        "commission_bps": float(config.get("stock_alpha_news_risk_overlay_replay_commission_bps", config.get("stock_alpha_news_risk_overlay_transaction_cost_bps", 0.0))),
        "stop_loss_pct": _number(config.get("stock_alpha_news_risk_overlay_replay_stop_loss_pct")),
        "profit_target_pct": _number(config.get("stock_alpha_news_risk_overlay_replay_profit_target_pct")),
        "reduce_multiplier": float(config.get("stock_alpha_news_risk_overlay_reduce_multiplier", 0.50)),
    }
    contrarian_weight = float(config.get("stock_alpha_news_risk_overlay_contrarian_weight", 0.25))
    variants = {
        "price_only": {"use_news": False, "replace_blocked": False, "reduce": False, "strict_gate": False},
        "news_risk_gate": {"use_news": True, "replace_blocked": False, "reduce": False, "strict_gate": True},
        "news_cash": {"use_news": True, "replace_blocked": False, "reduce": False, "strict_gate": True},
        "news_replacement": {"use_news": True, "replace_blocked": True, "reduce": False, "strict_gate": True},
        "news_reduced_size": {"use_news": True, "replace_blocked": False, "reduce": True, "strict_gate": False},
        "news_inverted_gate": {
            "use_news": True,
            "inverted": True,
            "replace_blocked": False,
            "reduce": False,
            "strict_gate": True,
            "diagnostic_only": True,
        },
        "news_contrarian_rerank": {
            "use_news": False,
            "contrarian_rerank": True,
            "contrarian_weight": contrarian_weight,
            "diagnostic_only": True,
        },
    }
    ledgers: list[dict[str, Any]] = []
    curves: dict[str, list[dict[str, Any]]] = {}
    attribution_inputs: list[dict[str, Any]] = []
    for variant, settings in variants.items():
        result = _run_open_trade_replay(
            rows,
            bars_by_symbol=bars_by_symbol,
            price_score_column=price_score_column,
            variant=variant,
            variant_settings=settings,
            replay_config=replay_config,
        )
        ledgers.extend(result["ledger"])
        curves[variant] = result["daily_equity"]
        attribution_inputs.extend(result["action_events"])
    risk = {variant: _daily_risk_metrics(curve, [row for row in ledgers if row["strategy_variant"] == variant]) for variant, curve in curves.items()}
    hypothetical = _hypothetical_trade_ledger(
        rows,
        bars_by_symbol=bars_by_symbol,
        price_score_column=price_score_column,
        replay_config=replay_config,
    )
    assumptions = _replay_assumptions(replay_config, price_score_column, processed_root)
    comparison = {
        "mode": "open_trade_marked_to_market_replay",
        "is_genuine_marked_to_market_portfolio_replay": True,
        "paper_orders_enabled": False,
        "live_orders_enabled": False,
        "strategy_rules_source": "core.research.ml.stock_level.stock_level_portfolio_replay ranking/top_n/max_position_weight conventions",
        "variants": risk,
        "news_overlay_reduced_max_drawdown": (
            risk.get("news_cash", {}).get("maximum_drawdown", 0.0)
            > risk.get("price_only", {}).get("maximum_drawdown", 0.0)
        ),
        "outputs_under": str(output_dir),
    }
    return {
        "trade_ledger": ledgers,
        "daily_equity": curves,
        "portfolio_comparison": comparison,
        "risk_metrics": risk,
        "action_attribution": _action_attribution(attribution_inputs, ledgers),
        "replay_config": replay_config,
        "variant_settings": variants,
        "bars_by_symbol": bars_by_symbol,
        "action_events": attribution_inputs,
        "hypothetical_trade_ledger": hypothetical,
        "contrarian_trade_ledger": [row for row in ledgers if row["strategy_variant"] in {"news_inverted_gate", "news_contrarian_rerank"}],
        "replay_assumptions": assumptions,
        "replay_data_audit": data_audit,
    }


def _run_open_trade_replay(
    rows: list[Mapping[str, Any]],
    *,
    bars_by_symbol: Mapping[str, list[Mapping[str, Any]]],
    price_score_column: str,
    variant: str,
    variant_settings: Mapping[str, Any],
    replay_config: Mapping[str, Any],
) -> dict[str, Any]:
    by_date: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        symbol = str(row.get("symbol", "")).upper()
        if symbol in bars_by_symbol and _number(row.get(price_score_column)) is not None:
            by_date.setdefault(str(row.get("decision_timestamp", row.get("rebalance_date", "")))[:10], []).append(row)
    bar_lookup = _bar_lookup(bars_by_symbol)
    next_lookup = _next_bar_lookup(bars_by_symbol)
    first_decision = min(by_date, default="9999-12-31")
    all_dates = sorted(
        set(by_date)
        | {
            bar["date"]
            for bars in bars_by_symbol.values()
            for bar in bars
            if bar["date"] >= first_decision
        }
    )
    cash = float(replay_config["starting_equity"])
    open_positions: list[dict[str, Any]] = []
    pending_entries: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    daily_equity: list[dict[str, Any]] = []
    action_events: list[dict[str, Any]] = []
    trade_counter = 0
    previous_equity = cash
    for current_date in all_dates:
        pending_now = [item for item in pending_entries if item["entry_date"] == current_date]
        pending_entries = [item for item in pending_entries if item["entry_date"] != current_date]
        for item in pending_now:
            bar = _bar_on_fast(item["symbol"], current_date, bar_lookup)
            if not bar:
                continue
            entry_price = float(bar["open"]) * (1.0 + float(replay_config["entry_slippage_bps"]) / 10_000.0)
            commission = item["cash_committed"] * float(replay_config["commission_bps"]) / 10_000.0
            if item["cash_committed"] + commission > cash + 1e-12:
                item["skip_reason"] = "insufficient_cash_at_entry"
                continue
            shares = item["cash_committed"] / entry_price if entry_price > 0 else 0.0
            cash -= item["cash_committed"] + commission
            item.update({"entry_price": entry_price, "shares": shares, "entry_commission": commission, "entry_timestamp": current_date})
            open_positions.append(item)
        still_open = []
        for position in open_positions:
            bar = _bar_on_fast(position["symbol"], current_date, bar_lookup)
            if not bar:
                still_open.append(position)
                continue
            position["bars_held"] += 1
            position["maximum_adverse_excursion"] = min(position["maximum_adverse_excursion"], float(bar["low"]) / position["entry_price"] - 1.0)
            position["maximum_favourable_excursion"] = max(position["maximum_favourable_excursion"], float(bar["high"]) / position["entry_price"] - 1.0)
            exit_price, exit_reason = _exit_decision(position, bar, replay_config)
            if exit_price is None:
                still_open.append(position)
                continue
            exit_price *= 1.0 - float(replay_config["exit_slippage_bps"]) / 10_000.0
            gross_pnl = (exit_price - position["entry_price"]) * position["shares"]
            exit_commission = (exit_price * position["shares"]) * float(replay_config["commission_bps"]) / 10_000.0
            total_costs = position["entry_commission"] + exit_commission
            net_pnl = gross_pnl - total_costs
            cash += exit_price * position["shares"] - exit_commission
            ledger.append(_ledger_row(position, exit_price, exit_reason, gross_pnl, total_costs, net_pnl, current_date))
        open_positions = still_open
        if current_date in by_date:
            equity = _equity_fast(cash, open_positions, bar_lookup, current_date)
            ranked = sorted(
                by_date[current_date],
                key=lambda row: (-_variant_sort_value(row, price_score_column, variant_settings), str(row.get("symbol", ""))),
            )
            for rank, candidate in enumerate(ranked, start=1):
                if len(open_positions) + len(pending_entries) >= int(replay_config["max_positions"]):
                    break
                if _has_symbol(candidate, open_positions, pending_entries):
                    continue
                action = str(candidate.get("news_action") or "NO_COVERAGE")
                multiplier, blocked = _variant_multiplier(action, variant_settings, replay_config)
                action_events.append(_action_event(candidate, variant, action, blocked, rank))
                if blocked:
                    if bool(variant_settings.get("replace_blocked")):
                        continue
                    if len(action_events) >= int(replay_config["top_n"]):
                        break
                    continue
                entry_date = _next_bar_date_fast(str(candidate["symbol"]).upper(), current_date, next_lookup)
                if not entry_date:
                    continue
                allocation = min(
                    cash,
                    equity * float(replay_config["max_position_weight"]) * multiplier,
                )
                if allocation <= 0:
                    continue
                trade_counter += 1
                pending_entries.append(
                    _pending_trade(
                        trade_counter,
                        candidate,
                        variant,
                        price_score_column,
                        entry_date,
                        allocation,
                        replay_config,
                        rank,
                    )
                )
                if len([p for p in pending_entries if p.get("decision_timestamp", "")[:10] == current_date]) >= int(replay_config["top_n"]):
                    break
        equity = _equity_fast(cash, open_positions, bar_lookup, current_date)
        daily_return = equity / previous_equity - 1.0 if previous_equity else 0.0
        previous_equity = equity
        daily_equity.append(
            {
                "date": current_date,
                "strategy_variant": variant,
                "cash": cash,
                "position_market_value": equity - cash,
                "total_equity": equity,
                "daily_return": daily_return,
                "gross_exposure": (equity - cash) / equity if equity else 0.0,
                "net_exposure": (equity - cash) / equity if equity else 0.0,
                "concurrent_positions": len(open_positions),
            }
        )
    for position in open_positions:
        last_bar = _last_bar(position["symbol"], bars_by_symbol)
        if last_bar:
            exit_price = float(last_bar["close"])
            gross_pnl = (exit_price - position["entry_price"]) * position["shares"]
            exit_commission = (exit_price * position["shares"]) * float(replay_config["commission_bps"]) / 10_000.0
            total_costs = position["entry_commission"] + exit_commission
            ledger.append(_ledger_row(position, exit_price, "end_of_data", gross_pnl, total_costs, gross_pnl - total_costs, last_bar["date"]))
    return {"ledger": ledger, "daily_equity": daily_equity, "action_events": action_events}


def _configured_first(value: Any, defaults: Iterable[str]) -> list[str]:
    return [str(value), *defaults] if value else list(defaults)


def _load_daily_price_bars(
    symbols: list[str],
    processed_root: Path,
    *,
    parallel_config: NewsRiskParallelConfig | None = None,
    parallel_report: dict[str, Any] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    ordered_symbols = sorted(dict.fromkeys(str(symbol).upper() for symbol in symbols))
    bars_by_symbol: dict[str, list[dict[str, Any]]] = {}
    missing = []
    failures = []
    task_durations = []
    config = parallel_config or _parallel_config({})
    use_parallel = _should_parallelize(config, len(ordered_symbols), phase="bar_loading", report=parallel_report)
    if use_parallel:
        if config.progress:
            print(
                f"Loading bars: 0 / {len(ordered_symbols)} symbols using {config.actual_workers} workers",
                flush=True,
            )
        completed = 0
        effective_chunk_size = min(config.chunk_size, config.batch_limit)
        for chunk in _chunks(ordered_symbols, effective_chunk_size):
            with ThreadPoolExecutor(max_workers=config.actual_workers) as executor:
                futures = {
                    executor.submit(_load_daily_price_bar_file, symbol, processed_root): symbol
                    for symbol in chunk
                }
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        failures.append({"task_id": symbol, "error": str(exc)})
                        continue
                    completed += 1
                    task_durations.append(float(result["elapsed_seconds"]))
                    if result["status"] == "MISSING":
                        missing.append(symbol)
                    elif result["status"] == "OK":
                        bars_by_symbol[symbol] = result["rows"]
                    else:
                        failures.append({"task_id": symbol, "error": result.get("error", "malformed worker result")})
                    if config.progress and (completed % max(effective_chunk_size, 1) == 0 or completed == len(ordered_symbols)):
                        print(
                            f"Loading bars: {completed} / {len(ordered_symbols)} symbols using {config.actual_workers} workers",
                            flush=True,
                        )
    else:
        for symbol in ordered_symbols:
            result = _load_daily_price_bar_file(symbol, processed_root)
            task_durations.append(float(result["elapsed_seconds"]))
            if result["status"] == "MISSING":
                missing.append(symbol)
            elif result["status"] == "OK":
                bars_by_symbol[symbol] = result["rows"]
            else:
                failures.append({"task_id": symbol, "error": result.get("error", "malformed worker result")})
    if failures:
        _record_worker_failures(parallel_report, "bar_loading", failures)
        first = failures[0]
        raise ValueError(f"daily bar worker failed for {first['task_id']}: {first['error']}")
    bars_by_symbol = {
        symbol: sorted(rows, key=lambda row: row["date"])
        for symbol, rows in sorted(bars_by_symbol.items())
    }
    _record_parallel_phase(
        parallel_report,
        "bar_loading",
        task_count=len(ordered_symbols),
        task_durations=task_durations,
        parallelized=use_parallel,
    )
    audit = {
        "processed_root": str(processed_root),
        "requested_symbol_count": len(ordered_symbols),
        "loaded_symbol_count": len(bars_by_symbol),
        "missing_symbol_count": len(missing),
        "missing_symbols": missing[:100],
        "timeframe": "1Day",
        "required_columns": ["timestamp", "open", "high", "low", "close", "volume", "symbol"],
        "adjusted_status": "local canonical bars; adjustment metadata not explicit in parquet schema",
        "split_handling": "not explicit in parquet schema",
        "dividend_handling": "not explicit in parquet schema",
        "entry_price_available_at_decision": False,
        "entry_convention": "next available daily bar open after decision date",
    }
    return bars_by_symbol, audit


def _variant_multiplier(
    action: str,
    settings: Mapping[str, Any],
    replay_config: Mapping[str, Any],
) -> tuple[float, bool]:
    if not settings.get("use_news"):
        return 1.0, False
    if settings.get("inverted"):
        if action == "ALLOW":
            return 0.0, True
        if action == "REDUCE":
            return float(replay_config["reduce_multiplier"]), False
        return 1.0, False
    if action == "BLOCK":
        return 0.0, True
    if action == "REDUCE":
        if settings.get("reduce"):
            return float(replay_config["reduce_multiplier"]), False
        if settings.get("strict_gate"):
            return 0.0, True
    return 1.0, False


def _variant_sort_value(
    row: Mapping[str, Any],
    price_score_column: str,
    settings: Mapping[str, Any],
) -> float:
    price_score = _number(row.get(price_score_column)) or 0.0
    if not settings.get("contrarian_rerank"):
        return price_score
    news_shock_score = _number(row.get("price_plus_news_risk_probability")) or 0.0
    weight = float(settings.get("contrarian_weight", 0.0))
    return price_score + weight * news_shock_score


def _pending_trade(
    trade_number: int,
    candidate: Mapping[str, Any],
    variant: str,
    price_score_column: str,
    entry_date: str,
    allocation: float,
    replay_config: Mapping[str, Any],
    ranking_after_news: int | None = None,
) -> dict[str, Any]:
    symbol = str(candidate["symbol"]).upper()
    return {
        "trade_id": f"{variant}-{trade_number:08d}",
        "strategy_variant": variant,
        "decision_timestamp": candidate.get("decision_timestamp", candidate.get("rebalance_date", "")),
        "symbol": symbol,
        "direction": "LONG",
        "price_score": _number(candidate.get(price_score_column)) or 0.0,
        "news_risk_probability": _number(candidate.get("price_plus_news_risk_probability")),
        "combined_score": (_number(candidate.get(price_score_column)) or 0.0) - (_number(candidate.get("price_plus_news_risk_probability")) or 0.0),
        "news_action": candidate.get("news_action", "NO_COVERAGE"),
        "news_coverage": candidate.get("news_coverage_status", "NO_COVERAGE"),
        "ranking_before_news": "",
        "ranking_after_news": ranking_after_news if ranking_after_news is not None else "",
        "replaced_candidate": False,
        "entry_date": entry_date,
        "cash_committed": allocation,
        "proposed_size": allocation,
        "actual_size": allocation,
        "stop": "",
        "target": "",
        "maximum_holding_period": replay_config["max_holding_bars"],
        "model_version": candidate.get("model_version", "news-risk-overlay-research-v1"),
        "price_feature_timestamp": candidate.get("decision_timestamp", ""),
        "news_feature_timestamp": candidate.get("news_feature_timestamp", ""),
        "bars_held": 0,
        "maximum_adverse_excursion": 0.0,
        "maximum_favourable_excursion": 0.0,
    }


def _exit_decision(
    position: Mapping[str, Any],
    bar: Mapping[str, Any],
    replay_config: Mapping[str, Any],
) -> tuple[float | None, str | None]:
    stop_loss = replay_config.get("stop_loss_pct")
    profit_target = replay_config.get("profit_target_pct")
    entry = float(position["entry_price"])
    stop_price = entry * (1.0 - float(stop_loss)) if stop_loss is not None else None
    target_price = entry * (1.0 + float(profit_target)) if profit_target is not None else None
    stop_hit = stop_price is not None and float(bar["low"]) <= stop_price
    target_hit = target_price is not None and float(bar["high"]) >= target_price
    if stop_hit:
        return float(stop_price), "stop_hit_conservative_before_target"
    if target_hit:
        return float(target_price), "target_hit"
    if int(position["bars_held"]) >= int(replay_config["max_holding_bars"]):
        return float(bar["close"]), "time_exit"
    return None, None


def _ledger_row(
    position: Mapping[str, Any],
    exit_price: float,
    exit_reason: str,
    gross_pnl: float,
    costs: float,
    net_pnl: float,
    exit_date: str,
) -> dict[str, Any]:
    committed = float(position["cash_committed"])
    return {
        "trade_id": position["trade_id"],
        "strategy_variant": position["strategy_variant"],
        "decision_timestamp": position["decision_timestamp"],
        "symbol": position["symbol"],
        "direction": position["direction"],
        "price_score": position["price_score"],
        "news_risk_probability": position["news_risk_probability"],
        "combined_score": position["combined_score"],
        "news_action": position["news_action"],
        "news_coverage": position["news_coverage"],
        "ranking_before_news": position["ranking_before_news"],
        "ranking_after_news": position["ranking_after_news"],
        "replaced_candidate": position["replaced_candidate"],
        "entry_timestamp": position["entry_timestamp"],
        "entry_price": position["entry_price"],
        "proposed_size": position["proposed_size"],
        "actual_size": position["actual_size"],
        "cash_committed": committed,
        "stop": position["stop"],
        "target": position["target"],
        "maximum_holding_period": position["maximum_holding_period"],
        "exit_timestamp": exit_date,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_pnl": gross_pnl,
        "transaction_costs": costs,
        "slippage": "",
        "net_pnl": net_pnl,
        "gross_return": gross_pnl / committed if committed else 0.0,
        "net_return": net_pnl / committed if committed else 0.0,
        "maximum_adverse_excursion": position["maximum_adverse_excursion"],
        "maximum_favourable_excursion": position["maximum_favourable_excursion"],
        "holding_period": position["bars_held"],
        "model_versions": position["model_version"],
        "price_feature_timestamp": position["price_feature_timestamp"],
        "news_feature_timestamp": position["news_feature_timestamp"],
    }


def _daily_risk_metrics(
    curve: list[Mapping[str, Any]],
    ledger: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if not curve:
        return {}
    returns = [float(row["daily_return"]) for row in curve]
    equity = [float(row["total_equity"]) for row in curve]
    start = equity[0] / (1.0 + returns[0]) if returns else equity[0]
    end = equity[-1]
    drawdowns = _drawdowns(equity)
    wins = [float(row["net_pnl"]) for row in ledger if float(row["net_pnl"]) > 0]
    losses = [float(row["net_pnl"]) for row in ledger if float(row["net_pnl"]) < 0]
    years = max(len(returns) / 252.0, 1.0 / 252.0)
    wealth = end / start if start else 0.0
    vol = pstdev(returns) * math.sqrt(252.0) if len(returns) > 1 else 0.0
    downside = [min(value, 0.0) for value in returns]
    downside_vol = pstdev(downside) * math.sqrt(252.0) if len(downside) > 1 else 0.0
    cagr = wealth ** (1.0 / years) - 1.0 if wealth > 0 else -1.0
    value_at_risk = sorted(returns)[max(0, math.ceil(len(returns) * 0.05) - 1)]
    total_costs = sum(float(row["transaction_costs"]) for row in ledger)
    average_exposure = mean(float(row["gross_exposure"]) for row in curve)
    turnover = sum(abs(float(row.get("daily_return", 0.0))) for row in curve)
    return {
        "starting_equity": start,
        "ending_equity": end,
        "total_return_decimal": wealth - 1.0,
        "total_return_percent": (wealth - 1.0) * 100.0,
        "wealth_multiple": wealth,
        "CAGR": cagr,
        "annualised_volatility": vol,
        "maximum_drawdown": min(drawdowns),
        "average_drawdown": mean(drawdowns),
        "longest_drawdown_duration": _longest_drawdown_duration(drawdowns),
        "Sharpe_ratio": (mean(returns) * 252.0) / vol if vol else 0.0,
        "Sortino_ratio": (mean(returns) * 252.0) / downside_vol if downside_vol else 0.0,
        "Calmar_ratio": cagr / abs(min(drawdowns)) if min(drawdowns) else 0.0,
        "Value_at_Risk_5pct": value_at_risk,
        "VaR_5pct": value_at_risk,
        "expected_shortfall_CVaR_5pct": _expected_shortfall(returns),
        "CVaR_5pct": _expected_shortfall(returns),
        "worst_day": min(returns),
        "worst_week": _worst_rolling_return(returns, 5),
        "worst_trade": min((float(row["net_return"]) for row in ledger), default=0.0),
        "maximum_adverse_excursion": min((float(row["maximum_adverse_excursion"]) for row in ledger), default=0.0),
        "maximum_favourable_excursion": max((float(row["maximum_favourable_excursion"]) for row in ledger), default=0.0),
        "profit_factor": sum(wins) / abs(sum(losses)) if losses else 0.0,
        "hit_rate": len(wins) / max(len(ledger), 1),
        "average_win": mean(wins) if wins else 0.0,
        "average_loss": mean(losses) if losses else 0.0,
        "turnover": turnover,
        "average_exposure": average_exposure,
        "exposure": average_exposure,
        "average_concurrent_positions": mean(float(row["concurrent_positions"]) for row in curve),
        "total_costs": total_costs,
        "slippage": 0.0,
        "number_of_trades": len(ledger),
    }


def _action_attribution(events: list[Mapping[str, Any]], ledger: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_action: dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        by_action.setdefault(str(event["news_action"]), []).append(event)
    report = {}
    for action, rows in by_action.items():
        forward = [_number(row.get("candidate_forward_return")) or 0.0 for row in rows]
        blocked = [row for row in rows if row.get("blocked")]
        report[action] = {
            "candidate_count": len(rows),
            "average_candidate_forward_return": mean(forward) if forward else 0.0,
            "profitable_trades_blocked": sum((_number(row.get("candidate_forward_return")) or 0.0) > 0 for row in blocked),
            "losing_trades_blocked": sum((_number(row.get("candidate_forward_return")) or 0.0) < 0 for row in blocked),
            "pnl_saved": abs(sum((_number(row.get("candidate_forward_return")) or 0.0) for row in blocked if (_number(row.get("candidate_forward_return")) or 0.0) < 0)),
            "pnl_missed": sum((_number(row.get("candidate_forward_return")) or 0.0) for row in blocked if (_number(row.get("candidate_forward_return")) or 0.0) > 0),
        }
    report["executed_trade_count"] = len(ledger)
    return report


def _score_direction_audit(
    *,
    rows: list[Mapping[str, Any]],
    config: NewsRiskOverlayConfig,
    target_column: str,
) -> dict[str, Any]:
    return {
        "target_column": target_column,
        "target_definition": (
            "Label is 1 when stop_hit_before_target is true, maximum adverse excursion "
            f"is <= {config.adverse_return_threshold}, or forward return is <= "
            f"{config.adverse_return_threshold}."
        ),
        "label_1_means": "adverse downside outcome / higher news risk",
        "label_0_means": "no configured adverse downside outcome",
        "forward_horizon": "from configured source column, typically actual_forward_return_10d when present",
        "thresholds": {
            "adverse_return_threshold": config.adverse_return_threshold,
            "reduce_threshold": config.reduce_threshold,
            "block_threshold": config.block_threshold,
        },
        "drawdown_sign_convention": "negative values are adverse; thresholds must be <= 0",
        "higher_model_probability_means": "higher probability of label 1, therefore higher intended risk",
        "reduce_comparison_operator": ">=",
        "block_comparison_operator": ">=",
        "probabilities_inverted_anywhere": False,
        "model_output_interpretation": "logistic sigmoid probability of news_risk_label == 1",
        "combined_score_formula": "price_score - price_plus_news_risk_probability for ledger diagnostics",
        "fallback_score_when_no_trained_model": "none; command fails if walk-forward probabilities are unavailable",
        "out_of_sample_probability_column": "price_plus_news_risk_probability",
        "row_count_checked": len(rows),
    }


def _assert_score_direction_contract(
    audit: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> None:
    thresholds = dict(audit.get("thresholds", {}) or {})
    adverse_threshold = float(thresholds.get("adverse_return_threshold", 0.0))
    reduce_threshold = float(thresholds.get("reduce_threshold", 0.0))
    block_threshold = float(thresholds.get("block_threshold", 0.0))
    if adverse_threshold > 0:
        raise ValueError("news-risk adverse threshold must be <= 0 so it identifies downside, not positive returns")
    if not (0.0 <= reduce_threshold <= block_threshold <= 1.0):
        raise ValueError("news-risk action thresholds must satisfy 0 <= reduce <= block <= 1")
    if audit.get("higher_model_probability_means") != "higher probability of label 1, therefore higher intended risk":
        raise ValueError("news-risk probability direction is not documented as higher risk")
    for row in rows:
        probability = _number(row.get("price_plus_news_risk_probability"))
        if probability is not None and not (0.0 <= probability <= 1.0):
            raise ValueError("news-risk probability outside [0, 1]")
        label = int(row.get("news_risk_label", 0))
        adverse = _adverse_excursion(row)
        forward = _first_numeric(row, RETURN_COLUMNS)
        stop_hit = _boolish(row.get("stop_hit_before_target"))
        if label == 1 and adverse is not None and adverse > 0 and not stop_hit and (forward is None or forward > adverse_threshold):
            raise ValueError("news-risk label is inconsistent with negative adverse-excursion sign convention")
        if label == 1 and forward is not None and forward > 0 and adverse is None and not stop_hit:
            raise ValueError("news-risk label marks a positive return without a downside source")


def _news_score_decile_diagnostics(
    rows: list[Mapping[str, Any]],
    ledger: list[Mapping[str, Any]],
    *,
    price_score_column: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored = [
        row for row in rows
        if _number(row.get("price_plus_news_risk_probability")) is not None
    ]
    scored.sort(key=lambda row: _timestamp(row))
    if not scored:
        return [], _empty_score_direction_report()
    by_trade_key: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for trade in ledger:
        by_trade_key.setdefault((str(trade.get("strategy_variant")), str(trade.get("symbol", "")).upper()), []).append(trade)
    ranked = sorted(scored, key=lambda row: _number(row.get("price_plus_news_risk_probability")) or 0.0)
    decile_by_payload: dict[str, int] = {}
    for index, row in enumerate(ranked):
        decile_by_payload[_row_key(row)] = min(10, int(index * 10 / max(len(ranked), 1)) + 1)
    deciles = []
    for decile in range(1, 11):
        members = [row for row in scored if decile_by_payload.get(_row_key(row)) == decile]
        returns = [_first_numeric(row, RETURN_COLUMNS) or 0.0 for row in members]
        maes = [_adverse_excursion(row) for row in members]
        mfes = [_favourable_excursion(row) for row in members]
        probabilities = [_number(row.get("price_plus_news_risk_probability")) or 0.0 for row in members]
        price_scores = [_number(row.get(price_score_column)) or 0.0 for row in members]
        executed = [
            trade for trade in ledger
            if str(trade.get("symbol", "")).upper() in {str(row.get("symbol", "")).upper() for row in members}
            and str(trade.get("strategy_variant")) == "price_only"
        ]
        net_returns = [float(trade.get("net_return", 0.0)) for trade in executed]
        deciles.append(
            {
                "decile": decile,
                "candidate_count": len(members),
                "executed_trade_count": len(executed),
                "average_news_risk_probability": mean(probabilities) if probabilities else 0.0,
                "average_forward_return": mean(returns) if returns else 0.0,
                "median_forward_return": median(returns) if returns else 0.0,
                "average_replay_net_return": mean(net_returns) if net_returns else 0.0,
                "median_replay_net_return": median(net_returns) if net_returns else 0.0,
                "hit_rate": sum(value > 0 for value in returns) / max(len(returns), 1),
                "maximum_adverse_excursion": min((value for value in maes if value is not None), default=0.0),
                "maximum_favourable_excursion": max((value for value in mfes if value is not None), default=0.0),
                "worst_trade": min(returns, default=0.0),
                "volatility": pstdev(returns) if len(returns) > 1 else 0.0,
                "stop_hit_rate": sum(_boolish(row.get("stop_hit_before_target")) for row in members) / max(len(members), 1),
                "event_category_mix": _category_mix(members),
                "news_coverage": sum(str(row.get("news_coverage_status")) == "COVERED" for row in members) / max(len(members), 1),
                "average_price_model_score": mean(price_scores) if price_scores else 0.0,
            }
        )
    probabilities = [_number(row.get("price_plus_news_risk_probability")) or 0.0 for row in scored]
    returns = [_first_numeric(row, RETURN_COLUMNS) or 0.0 for row in scored]
    maes = [_adverse_excursion(row) or 0.0 for row in scored]
    mfes = [_favourable_excursion(row) or 0.0 for row in scored]
    return deciles, {
        "uses_out_of_sample_predictions_only": True,
        "candidate_count": len(scored),
        "spearman_news_score_vs_future_return": _spearman(probabilities, returns),
        "correlation_news_score_vs_maximum_adverse_excursion": _pearson(probabilities, maes),
        "correlation_news_score_vs_maximum_favourable_excursion": _pearson(probabilities, mfes),
        "monotonicity": _monotonicity(deciles, "average_forward_return"),
        "confidence_intervals": {
            "method": "normal approximation by decile where practical",
            "average_forward_return_95pct": {
                str(row["decile"]): _mean_ci([_first_numeric(member, RETURN_COLUMNS) or 0.0 for member in scored if decile_by_payload.get(_row_key(member)) == row["decile"]])
                for row in deciles
            },
        },
        "answers": {
            "higher_score_predicts_lower_return": _spearman(probabilities, returns) < 0,
            "higher_score_predicts_deeper_temporary_drawdown": _pearson(probabilities, maes) < 0,
            "higher_score_predicts_greater_movement_both_directions": (
                abs(_pearson(probabilities, maes)) > 0.05 and abs(_pearson(probabilities, mfes)) > 0.05
            ),
            "relationship_supports_inversion": _spearman(probabilities, returns) > 0.05,
        },
    }


def _replay_action_attribution(
    events: list[Mapping[str, Any]],
    ledger: list[Mapping[str, Any]],
    hypothetical: list[Mapping[str, Any]],
) -> dict[str, Any]:
    actual_by_variant_action: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for trade in ledger:
        actual_by_variant_action.setdefault(
            (str(trade.get("strategy_variant")), str(trade.get("news_action"))),
            [],
        ).append(trade)
    hypothetical_by_symbol_date = {
        (str(row.get("symbol", "")).upper(), str(row.get("decision_timestamp", ""))[:10]): row
        for row in hypothetical
    }
    by_action: dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        if str(event.get("strategy_variant")) not in {"news_risk_gate", "news_cash", "news_reduced_size", "news_replacement"}:
            continue
        by_action.setdefault(str(event.get("news_action")), []).append(event)
    report: dict[str, Any] = {
        "units": {
            "pnl": "portfolio currency using starting_equity units",
            "return": "decimal return on committed capital",
            "mae_mfe": "decimal return from entry price",
        }
    }
    for action, rows in by_action.items():
        actual_trades = [
            trade
            for key, trades in actual_by_variant_action.items()
            if key[1] == action
            for trade in trades
        ]
        hypothetical_rows = [
            hypothetical_by_symbol_date.get((str(row.get("symbol", "")).upper(), str(row.get("decision_timestamp", ""))[:10]))
            for row in rows
        ]
        hypothetical_rows = [row for row in hypothetical_rows if row]
        blocked_hypothetical = [row for event, row in zip(rows, hypothetical_rows) if event.get("blocked")]
        report[action] = {
            "decision_count": len(rows),
            "actual_executed_trade_count": len(actual_trades),
            "hypothetical_trade_count": len(hypothetical_rows),
            "gross_pnl": sum(float(row.get("gross_pnl", 0.0)) for row in hypothetical_rows),
            "net_pnl": sum(float(row.get("net_pnl", 0.0)) for row in hypothetical_rows),
            "average_return": mean([float(row.get("net_return", 0.0)) for row in hypothetical_rows]) if hypothetical_rows else 0.0,
            "maximum_adverse_excursion": min((float(row.get("maximum_adverse_excursion", 0.0)) for row in hypothetical_rows), default=0.0),
            "maximum_favourable_excursion": max((float(row.get("maximum_favourable_excursion", 0.0)) for row in hypothetical_rows), default=0.0),
            "capital_used": sum(float(row.get("cash_committed", 0.0)) for row in hypothetical_rows),
            "average_holding_period": mean([float(row.get("holding_period", 0.0)) for row in hypothetical_rows]) if hypothetical_rows else 0.0,
            "profitable_trades_blocked": sum(float(row.get("net_pnl", 0.0)) > 0 for row in blocked_hypothetical),
            "losing_trades_blocked": sum(float(row.get("net_pnl", 0.0)) < 0 for row in blocked_hypothetical),
            "actual_portfolio_currency_pnl_saved": abs(sum(float(row.get("net_pnl", 0.0)) for row in blocked_hypothetical if float(row.get("net_pnl", 0.0)) < 0)),
            "actual_portfolio_currency_pnl_missed": sum(float(row.get("net_pnl", 0.0)) for row in blocked_hypothetical if float(row.get("net_pnl", 0.0)) > 0),
        }
    report["candidate_return_vs_replay_attribution_note"] = (
        "action_attribution.json keeps candidate-forward-return attribution; this file uses "
        "hypothetical replay entries and exits with identical replay rules where daily bars exist."
    )
    return report


def _event_category_analysis(
    rows: list[Mapping[str, Any]],
    ledger: list[Mapping[str, Any]],
) -> dict[str, Any]:
    categories = sorted({_event_category(row) for row in rows})
    trade_returns_by_symbol = {
        str(row.get("symbol", "")).upper(): float(row.get("net_return", 0.0))
        for row in ledger
        if row.get("strategy_variant") == "price_only"
    }
    policies = _event_category_policies()
    report: dict[str, Any] = {
        "policy_defaults": policies,
        "category_source": "best available event/category columns; unavailable rows use general_negative_sentiment_or_uncategorized",
    }
    for category in categories:
        members = [row for row in rows if _event_category(row) == category]
        returns = [_first_numeric(row, RETURN_COLUMNS) or 0.0 for row in members]
        maes = [_adverse_excursion(row) for row in members]
        mfes = [_favourable_excursion(row) for row in members]
        replay_returns = [trade_returns_by_symbol.get(str(row.get("symbol", "")).upper(), 0.0) for row in members]
        report[category] = {
            "policy": policies.get(category, "RISK_ONLY"),
            "count": len(members),
            "average_return": mean(returns) if returns else 0.0,
            "median_return": median(returns) if returns else 0.0,
            "maximum_adverse_excursion": min((value for value in maes if value is not None), default=0.0),
            "maximum_favourable_excursion": max((value for value in mfes if value is not None), default=0.0),
            "hit_rate": sum(value > 0 for value in returns) / max(len(returns), 1),
            "recovery_duration": "unavailable_without_explicit_recovery_field",
            "immediate_entry_result": mean(replay_returns) if replay_returns else 0.0,
            "delayed_entry_result": "not_computed_without_enabled_stabilisation_variant",
            "contrarian_suitability": _contrarian_suitability(category, returns, maes),
        }
    return report


def _contrarian_strategy_report(
    risk_metrics: Mapping[str, Any],
    variant_settings: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    enabled = bool(config.get("stock_alpha_news_risk_overlay_extreme_event_entry_enabled", False))
    return {
        "research_only": True,
        "paper_orders_enabled": False,
        "live_orders_enabled": False,
        "raw_probabilities_changed": False,
        "diagnostic_variants": {
            "price_only": risk_metrics.get("price_only", {}),
            "news_risk_gate": risk_metrics.get("news_risk_gate", risk_metrics.get("news_cash", {})),
            "news_inverted_gate": risk_metrics.get("news_inverted_gate", {}),
            "news_contrarian_rerank": risk_metrics.get("news_contrarian_rerank", {}),
        },
        "variant_settings": {
            name: settings
            for name, settings in variant_settings.items()
            if name in {"news_inverted_gate", "news_contrarian_rerank"}
        },
        "extreme_event_entry": {
            "enabled": enabled,
            "implemented_as": "disabled policy scaffold only unless explicitly enabled in config",
            "candidate_universe_rule": "no arbitrary symbols; eligible rows must already be in the joined price-model candidate universe",
            "safeguards": {
                "minimum_price_model_score": config.get("stock_alpha_news_risk_overlay_extreme_entry_min_price_score", "unavailable"),
                "minimum_liquidity": config.get("stock_alpha_news_risk_overlay_extreme_entry_min_dollar_volume", "unavailable"),
                "minimum_price": config.get("stock_alpha_news_risk_overlay_extreme_entry_min_price", "unavailable"),
                "maximum_position_size": config.get("stock_alpha_news_risk_overlay_extreme_entry_max_position_weight", 0.0),
                "maximum_contrarian_positions": config.get("stock_alpha_news_risk_overlay_extreme_entry_max_positions", 0),
                "bankruptcy_delisting_fraud_accounting_excluded_by_default": True,
                "point_in_time_news_required": True,
            },
        },
    }


def _price_stabilisation_report(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(config.get("stock_alpha_news_risk_overlay_price_stabilisation_enabled", False)),
        "point_in_time_only": True,
        "rules": {
            "next_session_open": "available",
            "one_session_delay": "configured_but_not_run_until_enabled",
            "two_session_delay": "configured_but_not_run_until_enabled",
            "first_close_above_prior_close": "configured_but_not_run_until_enabled",
            "first_positive_daily_return_after_event": "configured_but_not_run_until_enabled",
            "short_moving_average_reclaim": "unavailable_without_existing_ma_utility_wiring",
            "continued_fall_threshold_no_entry": config.get("stock_alpha_news_risk_overlay_stabilisation_no_entry_fall_threshold", "unavailable"),
        },
        "headline_answer": "Immediate versus delayed stabilisation is not compared until extreme-event entry is explicitly enabled.",
    }


def _resilience_filter_analysis(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    fields = {
        "market_cap": ("market_cap", "market_capitalization", "news_market_cap"),
        "dollar_trading_volume": ("dollar_volume", "avg_dollar_volume", "news_dollar_volume"),
        "profitability": ("profit_margin", "return_on_equity", "news_profitability"),
        "free_cash_flow": ("free_cash_flow", "fcf"),
        "leverage": ("debt_to_equity", "net_debt_to_ebitda"),
        "interest_coverage": ("interest_coverage",),
        "bankruptcy_distance": ("distance_to_default", "altman_z_score"),
        "index_membership": ("index_member", "sp500_member", "russell_1000_member"),
        "sector": ("sector",),
        "analyst_coverage": ("analyst_count", "news_analyst_coverage"),
        "prior_recovery_behaviour": ("prior_recovery_rate", "prior_event_recovery_days"),
    }
    availability = {
        name: next((column for column in candidates if any(row.get(column) not in {None, ""} for row in rows)), None)
        for name, candidates in fields.items()
    }
    return {
        "field_availability": {name: (column or "unavailable") for name, column in availability.items()},
        "all_companies": _row_group_summary(rows),
        "large_liquid_companies": _filtered_group_summary(rows, availability, large_liquid=True),
        "financially_resilient_companies": _filtered_group_summary(rows, availability, resilient=True),
        "smaller_or_less_liquid_companies": _filtered_group_summary(rows, availability, smaller_or_less_liquid=True),
        "unsupported_values_imputed": False,
    }


def _extreme_event_archive(
    rows: list[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    threshold = float(config.get("stock_alpha_news_risk_overlay_extreme_event_probability_threshold", 0.90))
    archive = []
    for index, row in enumerate(rows, start=1):
        probability = _number(row.get("price_plus_news_risk_probability")) or 0.0
        if probability < threshold:
            continue
        archive.append(
            {
                "event_id": row.get("event_id") or f"news-risk-{index:08d}",
                "symbol": row.get("symbol", ""),
                "category": _event_category(row),
                "publication_time": row.get("published_at_utc", row.get("published_at", "")),
                "effective_availability_time": row.get("news_feature_timestamp", ""),
                "severity": probability,
                "sentiment_shock": _first_numeric(row, ("news_sentiment", "sentiment", "news_sentiment_score")),
                "volume_shock": _first_numeric(row, ("news_volume_shock", "volume_shock")),
                "source_count": _first_numeric(row, ("news_source_count", "source_count")),
                "source_diversity": _first_numeric(row, ("news_source_diversity", "source_diversity")),
                "relevance": _first_numeric(row, ("news_relevance", "relevance")),
                "novelty": _first_numeric(row, ("news_novelty", "novelty")),
                "price_before_event": _first_numeric(row, ("price_before_event", "previous_close", "close")),
                "model_version": row.get("model_version", "news-risk-overlay-research-v1"),
                "future_1d_return": _first_numeric(row, ("actual_forward_return_1d",)),
                "future_3d_return": _first_numeric(row, ("actual_forward_return_3d",)),
                "future_5d_return": _first_numeric(row, ("actual_forward_return_5d",)),
                "future_10d_return": _first_numeric(row, ("actual_forward_return_10d",)),
                "future_20d_return": _first_numeric(row, ("actual_forward_return_20d",)),
                "maximum_adverse_excursion": _adverse_excursion(row),
                "maximum_favourable_excursion": _favourable_excursion(row),
                "recovery_duration": row.get("recovery_duration", ""),
            }
        )
    return archive, {
        "point_in_time_archive_rows": len(archive),
        "future_outcomes_attached_for_historical_research_only": True,
        "future_outcomes_exposed_to_decisions": False,
        "decayed_memory_features": {
            "time_since_latest_extreme_negative_event": "planned_feature_from_archive",
            "latest_event_severity": "planned_feature_from_archive",
            "cumulative_severity": "planned_feature_from_archive",
            "repeated_event_count": "planned_feature_from_archive",
            "unresolved_event_flag": "unavailable_without_resolution_data",
            "event_category_specific_decay": "planned_feature_from_archive",
        },
    }


def _cost_scenario_comparison(
    rows: list[Mapping[str, Any]],
    *,
    bars_by_symbol: Mapping[str, list[Mapping[str, Any]]],
    price_score_column: str,
    base_replay_config: Mapping[str, Any],
    parallel_config: NewsRiskParallelConfig | None = None,
    parallel_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    variants = {
        "price_only": {"use_news": False, "replace_blocked": False, "reduce": False, "strict_gate": False},
        "news_risk_gate": {"use_news": True, "replace_blocked": False, "reduce": False, "strict_gate": True},
        "news_inverted_gate": {"use_news": True, "inverted": True, "replace_blocked": False, "reduce": False, "strict_gate": True},
        "news_contrarian_rerank": {"use_news": False, "contrarian_rerank": True, "contrarian_weight": 0.25},
    }
    round_trips = (0.0, 5.0, 10.0, 20.0)
    config = parallel_config or _parallel_config({})
    use_parallel = (
        _should_parallelize(config, len(round_trips), phase="cost_scenarios", report=parallel_report)
        and config.backend == "thread"
    )
    if config.backend == "process" and config.enabled:
        _record_fallback(
            parallel_report,
            "cost_scenarios",
            "process backend not used because daily bars are large shared read-only inputs",
        )
    task_durations = []
    results: dict[float, dict[str, Any]] = {}
    if use_parallel:
        with ThreadPoolExecutor(max_workers=config.actual_workers) as executor:
            futures = {
                executor.submit(
                    _cost_scenario_task,
                    round_trip_bps,
                    rows,
                    bars_by_symbol,
                    price_score_column,
                    base_replay_config,
                    variants,
                ): round_trip_bps
                for round_trip_bps in round_trips
            }
            for future in as_completed(futures):
                round_trip_bps = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    _record_worker_failures(
                        parallel_report,
                        "cost_scenarios",
                        [{"task_id": f"{round_trip_bps:g}_bps_round_trip", "error": str(exc)}],
                    )
                    raise ValueError(f"cost scenario worker failed for {round_trip_bps:g}_bps_round_trip: {exc}") from exc
                task_durations.append(float(result.pop("elapsed_seconds")))
                results[round_trip_bps] = result
    else:
        for round_trip_bps in round_trips:
            result = _cost_scenario_task(
                round_trip_bps,
                rows,
                bars_by_symbol,
                price_score_column,
                base_replay_config,
                variants,
            )
            task_durations.append(float(result.pop("elapsed_seconds")))
            results[round_trip_bps] = result
    _record_parallel_phase(
        parallel_report,
        "cost_scenarios",
        task_count=len(round_trips),
        task_durations=task_durations,
        parallelized=use_parallel,
    )
    scenarios = {
        f"{round_trip_bps:g}_bps_round_trip": results[round_trip_bps]
        for round_trip_bps in sorted(results)
    }
    return {
        "cost_model": "round_trip_bps split equally into entry and exit commissions; slippage set to 0.0 unless configured elsewhere",
        "zero_costs_recorded_as": 0.0,
        "scenarios": scenarios,
    }


def _cost_scenario_task(
    round_trip_bps: float,
    rows: list[Mapping[str, Any]],
    bars_by_symbol: Mapping[str, list[Mapping[str, Any]]],
    price_score_column: str,
    base_replay_config: Mapping[str, Any],
    variants: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    started = time.perf_counter()
    replay_config = dict(base_replay_config)
    one_way = round_trip_bps / 2.0
    replay_config["commission_bps"] = one_way
    replay_config["entry_slippage_bps"] = 0.0
    replay_config["exit_slippage_bps"] = 0.0
    scenario_metrics = {}
    for variant, settings in sorted(variants.items()):
        result = _run_open_trade_replay(
            rows,
            bars_by_symbol=bars_by_symbol,
            price_score_column=price_score_column,
            variant=variant,
            variant_settings=settings,
            replay_config=replay_config,
        )
        scenario_metrics[variant] = _daily_risk_metrics(result["daily_equity"], result["ledger"])
    return {
        "round_trip_bps": round_trip_bps,
        "one_way_commission_bps": one_way,
        "variants": scenario_metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _build_executive_summary(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifacts = _news_risk_artifact_paths(output_dir)
    status = [_artifact_status(name, path) for name, path in artifacts.items()]
    risk = _read_json_if_available(artifacts["risk_metrics"])
    coverage = _read_json_if_available(artifacts["coverage_report"])
    direction = _read_json_if_available(artifacts["news_score_direction_report"])
    cost = _read_json_if_available(artifacts["cost_scenario_comparison"])
    event = _read_json_if_available(artifacts["event_category_analysis"])
    extreme = _read_json_if_available(artifacts["extreme_event_memory_report"])
    comparison = _read_json_if_available(artifacts["portfolio_comparison"])
    replay_audit = _read_json_if_available(artifacts["replay_data_audit"])
    deciles = _read_csv_if_available(artifacts["news_score_deciles"])
    strategy_rows = _strategy_summary_rows(risk)
    cost_rows = _cost_robustness_rows(cost)
    diagnostics = _diagnostics_summary(
        coverage=coverage,
        direction=direction,
        event=event,
        extreme=extreme,
        risk=risk,
        cost_rows=cost_rows,
        comparison=comparison,
    )
    warnings = _executive_warnings(
        artifact_status=status,
        deciles=deciles,
        diagnostics=diagnostics,
        cost_rows=cost_rows,
        replay_audit=replay_audit,
        risk=risk,
    )
    return {
        "output_dir": str(output_dir),
        "strategy_comparison": strategy_rows,
        "cost_robustness": cost_rows,
        "diagnostics": diagnostics,
        "winners": _winner_summary(strategy_rows, cost_rows),
        "warnings": warnings,
        "paper_orders_enabled": bool(comparison.get("paper_orders_enabled", False)),
        "live_orders_enabled": bool(comparison.get("live_orders_enabled", False)),
    }, status


def _news_risk_artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "coverage_report": output_dir / "coverage_report.json",
        "risk_metrics": output_dir / "risk_metrics.json",
        "portfolio_comparison": output_dir / "portfolio_comparison.json",
        "news_score_deciles": output_dir / "news_score_deciles.csv",
        "news_score_direction_report": output_dir / "news_score_direction_report.json",
        "cost_scenario_comparison": output_dir / "cost_scenario_comparison.json",
        "event_category_analysis": output_dir / "event_category_analysis.json",
        "extreme_event_memory_report": output_dir / "extreme_event_memory_report.json",
        "replay_data_audit": output_dir / "replay_data_audit.json",
        "parallel_execution_report": output_dir / "parallel_execution_report.json",
        "README": output_dir / "README.md",
    }


def _artifact_status(name: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"name": name, "path": str(path), "status": "MISSING", "bytes": 0}
    stat = path.stat()
    size = stat.st_size
    base = {"name": name, "path": str(path), "bytes": size, "modified_timestamp": stat.st_mtime}
    if size <= 0:
        return {**base, "status": "EMPTY_PLACEHOLDER"}
    if path.suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {**base, "status": "FAILED_VALIDATION", "reason": str(exc)}
        if not payload:
            return {**base, "status": "EMPTY_VALID"}
    if path.suffix == ".csv":
        rows = _read_csv_if_available(path)
        if not rows:
            return {**base, "status": "EMPTY_VALID"}
        return {**base, "status": "COMPLETE", "row_count": len(rows)}
    return {**base, "status": "COMPLETE"}


def _read_json_if_available(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv_if_available(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    try:
        return _read_csv(path)
    except (OSError, csv.Error):
        return []


def _strategy_summary_rows(risk: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, metrics in sorted(dict(risk).items()):
        if not isinstance(metrics, Mapping) or not metrics:
            continue
        rows.append(
            {
                "strategy": name,
                "ending_wealth": _metric(metrics, "ending_equity", "wealth_multiple"),
                "total_return": _metric(metrics, "total_return_percent"),
                "cagr": _metric(metrics, "cagr", "CAGR"),
                "maximum_drawdown": _metric(metrics, "maximum_drawdown"),
                "sharpe_ratio": _metric(metrics, "sharpe_ratio", "Sharpe_ratio"),
                "calmar_ratio": _metric(metrics, "calmar_ratio", "Calmar_ratio"),
                "cvar": _metric(metrics, "cvar_5pct", "CVaR_5pct", "expected_shortfall_CVaR_5pct"),
                "trade_count": int(_metric(metrics, "number_of_trades") or 0),
                "total_costs": _metric(metrics, "total_costs"),
            }
        )
    return rows


def _cost_robustness_rows(cost: Mapping[str, Any]) -> list[dict[str, Any]]:
    scenarios = dict(cost.get("scenarios", {}) or {})
    rows = []
    for key, payload in sorted(scenarios.items(), key=lambda item: _number(str(item[1].get("round_trip_bps", 0))) or 0.0):
        variants = dict(payload.get("variants", {}) or {})
        price = dict(variants.get("price_only", {}) or {})
        contrarian = dict(variants.get("news_contrarian_rerank", variants.get("news_inverted_gate", {})) or {})
        price_wealth = _metric(price, "ending_equity", "wealth_multiple")
        contrarian_wealth = _metric(contrarian, "ending_equity", "wealth_multiple")
        rows.append(
            {
                "round_trip_bps": _metric(payload, "round_trip_bps"),
                "price_only_ending_wealth": price_wealth,
                "contrarian_ending_wealth": contrarian_wealth,
                "difference": (
                    contrarian_wealth - price_wealth
                    if contrarian_wealth is not None and price_wealth is not None
                    else None
                ),
                "price_only_maximum_drawdown": _metric(price, "maximum_drawdown"),
                "contrarian_maximum_drawdown": _metric(contrarian, "maximum_drawdown"),
                "scenario": key,
            }
        )
    return rows


def _diagnostics_summary(
    *,
    coverage: Mapping[str, Any],
    direction: Mapping[str, Any],
    event: Mapping[str, Any],
    extreme: Mapping[str, Any],
    risk: Mapping[str, Any],
    cost_rows: list[Mapping[str, Any]],
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    event_counts = _event_counts(event)
    categorized = sum(count for category, count in event_counts.items() if category != "general_negative_sentiment_or_uncategorized")
    uncategorized = event_counts.get("general_negative_sentiment_or_uncategorized", 0)
    total_events = categorized + uncategorized
    price = dict(risk.get("price_only", {}) or {})
    contrarian = dict(risk.get("news_contrarian_rerank", {}) or {})
    price_wealth = _metric(price, "ending_equity", "wealth_multiple")
    contrarian_wealth = _metric(contrarian, "ending_equity", "wealth_multiple")
    return {
        "news_coverage_percentage": float(coverage.get("row_coverage_ratio", 0.0)) * 100.0,
        "categorized_event_percentage": categorized / max(total_events, 1) * 100.0,
        "uncategorized_event_percentage": uncategorized / max(total_events, 1) * 100.0,
        "extreme_event_count": int(extreme.get("point_in_time_archive_rows", 0) or 0),
        "score_direction_conclusion": _score_direction_conclusion(direction),
        "contrarian_reranking_beat_price_only": (
            contrarian_wealth > price_wealth
            if contrarian_wealth is not None and price_wealth is not None
            else None
        ),
        "superior_after_5_10_20_bps": _superior_after_costs(cost_rows, (5.0, 10.0, 20.0)),
        "untouched_holdout_used": False,
        "untouched_holdout_status": "unavailable_in_current_artifacts",
        "paper_orders_enabled": bool(comparison.get("paper_orders_enabled", False)),
        "live_orders_enabled": bool(comparison.get("live_orders_enabled", False)),
    }


def _event_counts(event: Mapping[str, Any]) -> dict[str, int]:
    counts = {}
    for key, payload in event.items():
        if key in {"policy_defaults", "category_source"} or not isinstance(payload, Mapping):
            continue
        counts[key] = int(payload.get("count", 0) or 0)
    return counts


def _score_direction_conclusion(direction: Mapping[str, Any]) -> str:
    answers = dict(direction.get("answers", {}) or {})
    if answers.get("relationship_supports_inversion"):
        return "supports diagnostic inversion"
    if answers.get("higher_score_predicts_lower_return") or answers.get("higher_score_predicts_deeper_temporary_drawdown"):
        return "higher score aligns with downside risk"
    if answers.get("higher_score_predicts_greater_movement_both_directions"):
        return "higher score appears volatility-like"
    return "inconclusive"


def _superior_after_costs(rows: list[Mapping[str, Any]], bps_values: Iterable[float]) -> dict[str, bool | None]:
    by_bps = {float(row.get("round_trip_bps", 0.0) or 0.0): row for row in rows}
    output = {}
    for value in bps_values:
        row = by_bps.get(float(value))
        output[f"{value:g}_bps"] = (row.get("difference") > 0) if row and row.get("difference") is not None else None
    return output


def _executive_warnings(
    *,
    artifact_status: list[Mapping[str, Any]],
    deciles: list[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    cost_rows: list[Mapping[str, Any]],
    replay_audit: Mapping[str, Any],
    risk: Mapping[str, Any],
) -> list[str]:
    warnings = []
    bad_artifacts = [row for row in artifact_status if row.get("status") not in {"COMPLETE", "EMPTY_VALID"}]
    if bad_artifacts:
        warnings.append("failed or missing output validation artifacts are present")
    if any(row.get("status") == "EMPTY_PLACEHOLDER" for row in artifact_status):
        warnings.append("empty placeholder report files are present")
    mtimes = [
        float(row["modified_timestamp"])
        for row in artifact_status
        if row.get("status") == "COMPLETE" and row.get("modified_timestamp") is not None
    ]
    if mtimes and max(mtimes) - min(mtimes) > 24 * 60 * 60:
        warnings.append("stale artifacts: output modification times span more than one day")
    if _decile_values_repeated(deciles, "average_forward_return"):
        warnings.append("repeated or identical metrics across score deciles")
    if _decile_values_repeated(deciles, "executed_trade_count"):
        warnings.append("identical executed-trade counts across multiple deciles")
    if int(diagnostics.get("extreme_event_count", 0) or 0) < 30:
        warnings.append("insufficient extreme-event sample size")
    if float(diagnostics.get("uncategorized_event_percentage", 0.0) or 0.0) >= 80.0:
        warnings.append("mostly uncategorized events")
    if "not explicit" in str(replay_audit.get("adjusted_status", "")).lower():
        warnings.append("missing corporate-action adjustment information")
    zero_cost = next((row for row in cost_rows if float(row.get("round_trip_bps", 0.0) or 0.0) == 0.0), None)
    positive_zero = zero_cost and (zero_cost.get("contrarian_ending_wealth") or 0.0) > (zero_cost.get("price_only_ending_wealth") or 0.0)
    positive_realistic = any(
        (row.get("difference") or 0.0) > 0 and float(row.get("round_trip_bps", 0.0) or 0.0) > 0.0
        for row in cost_rows
    )
    if positive_zero and not positive_realistic:
        warnings.append("zero-cost-only profitability")
    if diagnostics.get("untouched_holdout_used") is False:
        warnings.append("in-sample or post-hypothesis evaluation: untouched holdout unavailable")
    if not risk:
        warnings.append("risk metrics unavailable")
    return warnings


def _decile_values_repeated(rows: list[Mapping[str, Any]], field: str) -> bool:
    values = [str(row.get(field, "")) for row in rows if str(row.get("candidate_count", "0")) not in {"0", ""}]
    return len(values) >= 3 and len(set(values)) == 1


def _winner_summary(
    strategy_rows: list[Mapping[str, Any]],
    cost_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "best_absolute_return": _best_strategy(strategy_rows, "ending_wealth", higher=True),
        "best_risk_adjusted_result": _best_strategy(strategy_rows, "sharpe_ratio", higher=True),
        "lowest_maximum_drawdown": _best_strategy(strategy_rows, "maximum_drawdown", higher=True),
        "best_result_after_realistic_costs": _best_after_costs(cost_rows),
    }


def _best_strategy(rows: list[Mapping[str, Any]], field: str, *, higher: bool) -> str | None:
    usable = [row for row in rows if row.get(field) is not None]
    if not usable:
        return None
    return str(max(usable, key=lambda row: row[field])["strategy"] if higher else min(usable, key=lambda row: row[field])["strategy"])


def _best_after_costs(rows: list[Mapping[str, Any]]) -> str:
    realistic = [row for row in rows if float(row.get("round_trip_bps", 0.0) or 0.0) in {5.0, 10.0, 20.0}]
    if not realistic:
        return "unavailable"
    wins = sum((row.get("difference") or 0.0) > 0 for row in realistic)
    return "contrarian" if wins == len(realistic) else "price_only_or_mixed"


def _summary_lines(summary: Mapping[str, Any]) -> list[str]:
    lines = [
        "STOCK-ALPHA NEWS RISK OVERLAY SUMMARY",
        f"Output: {summary.get('output_dir', '')}",
        "",
        "Strategy comparison:",
        _table(
            ["strategy", "wealth", "return", "cagr", "max_dd", "sharpe", "calmar", "cvar", "trades", "costs"],
            [
                [
                    row["strategy"],
                    _fmt(row.get("ending_wealth")),
                    _fmt_pct(row.get("total_return")),
                    _fmt_pct_decimal(row.get("cagr")),
                    _fmt_pct_decimal(row.get("maximum_drawdown")),
                    _fmt(row.get("sharpe_ratio")),
                    _fmt(row.get("calmar_ratio")),
                    _fmt_pct_decimal(row.get("cvar")),
                    str(row.get("trade_count", 0)),
                    _fmt(row.get("total_costs")),
                ]
                for row in summary.get("strategy_comparison", [])
            ],
        ),
        "",
        "Cost robustness:",
        _table(
            ["bps", "price_w", "contra_w", "diff", "price_dd", "contra_dd"],
            [
                [
                    _fmt(row.get("round_trip_bps"), digits=0),
                    _fmt(row.get("price_only_ending_wealth")),
                    _fmt(row.get("contrarian_ending_wealth")),
                    _fmt(row.get("difference")),
                    _fmt_pct_decimal(row.get("price_only_maximum_drawdown")),
                    _fmt_pct_decimal(row.get("contrarian_maximum_drawdown")),
                ]
                for row in summary.get("cost_robustness", [])
            ],
        ),
    ]
    diagnostics = dict(summary.get("diagnostics", {}) or {})
    winners = dict(summary.get("winners", {}) or {})
    lines.extend(
        [
            "",
            "Diagnostics:",
            f"- news coverage: {_fmt_pct(diagnostics.get('news_coverage_percentage'))}",
            f"- categorized/uncategorized events: {_fmt_pct(diagnostics.get('categorized_event_percentage'))} / {_fmt_pct(diagnostics.get('uncategorized_event_percentage'))}",
            f"- extreme events: {diagnostics.get('extreme_event_count', 0)}",
            f"- score direction: {diagnostics.get('score_direction_conclusion', 'unavailable')}",
            f"- contrarian beat price-only: {diagnostics.get('contrarian_reranking_beat_price_only')}",
            f"- superior after 5/10/20 bps: {diagnostics.get('superior_after_5_10_20_bps')}",
            f"- untouched holdout used: {diagnostics.get('untouched_holdout_used')}",
            f"- paper/live trading enabled: {diagnostics.get('paper_orders_enabled')} / {diagnostics.get('live_orders_enabled')}",
            "",
            "Winners:",
            f"- best absolute return: {winners.get('best_absolute_return')}",
            f"- best risk-adjusted: {winners.get('best_risk_adjusted_result')}",
            f"- lowest max drawdown: {winners.get('lowest_maximum_drawdown')}",
            f"- best after realistic costs: {winners.get('best_result_after_realistic_costs')}",
        ]
    )
    warnings = list(summary.get("warnings", []) or [])
    if warnings:
        lines.extend(["", "WARNINGS:"])
        lines.extend(f"- {warning}" for warning in warnings[:6])
    return lines


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "(unavailable)"
    widths = [
        max(len(str(value)) for value in [header, *[row[index] for row in rows]])
        for index, header in enumerate(headers)
    ]
    lines = ["  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend(
        "  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def _metric(payload: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = _number(payload.get(name))
        if value is not None:
            return value
    return None


def _fmt(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def _fmt_pct(value: Any) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.1f}%"


def _fmt_pct_decimal(value: Any) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number * 100.0:.1f}%"


def _hypothetical_trade_ledger(
    rows: list[Mapping[str, Any]],
    *,
    bars_by_symbol: Mapping[str, list[Mapping[str, Any]]],
    price_score_column: str,
    replay_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    config = dict(replay_config)
    config["max_positions"] = max(int(config.get("max_positions", 1)), len(rows), 1)
    config["top_n"] = max(int(config.get("top_n", 1)), len(rows), 1)
    config["max_position_weight"] = min(float(config.get("max_position_weight", 0.05)), 0.01)
    result = _run_open_trade_replay(
        rows,
        bars_by_symbol=bars_by_symbol,
        price_score_column=price_score_column,
        variant="hypothetical_candidate",
        variant_settings={"use_news": False},
        replay_config=config,
    )
    return result["ledger"]


def _empty_score_direction_report() -> dict[str, Any]:
    return {
        "uses_out_of_sample_predictions_only": True,
        "candidate_count": 0,
        "spearman_news_score_vs_future_return": 0.0,
        "correlation_news_score_vs_maximum_adverse_excursion": 0.0,
        "correlation_news_score_vs_maximum_favourable_excursion": 0.0,
        "monotonicity": {"direction": "unavailable", "violations": 0},
        "confidence_intervals": {},
        "answers": {
            "higher_score_predicts_lower_return": False,
            "higher_score_predicts_deeper_temporary_drawdown": False,
            "higher_score_predicts_greater_movement_both_directions": False,
            "relationship_supports_inversion": False,
        },
    }


def _first_numeric(row: Mapping[str, Any], columns: Iterable[str]) -> float | None:
    for column in columns:
        value = _number(row.get(column))
        if value is not None:
            return value
    return None


def _favourable_excursion(row: Mapping[str, Any]) -> float | None:
    for column in (
        "actual_max_favourable_excursion",
        "forward_max_favourable_excursion",
        "max_favourable_excursion",
        "actual_max_favorable_excursion",
        "forward_max_favorable_excursion",
        "max_favorable_excursion",
    ):
        value = _number(row.get(column))
        if value is not None:
            return value
    forward = _first_numeric(row, RETURN_COLUMNS)
    return max(forward, 0.0) if forward is not None else None


def _row_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("decision_timestamp", row.get("rebalance_date", "")))[:19],
            str(row.get("symbol", "")).upper(),
            str(row.get("price_plus_news_risk_probability", "")),
        ]
    )


def _category_mix(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    output: dict[str, int] = {}
    for row in rows:
        category = _event_category(row)
        output[category] = output.get(category, 0) + 1
    return output


def _event_category(row: Mapping[str, Any]) -> str:
    for column in ("news_event_category", "event_category", "category", "news_category"):
        value = str(row.get(column, "")).strip().lower()
        if value:
            return _normalise_event_category(value)
    return "general_negative_sentiment_or_uncategorized"


def _normalise_event_category(value: str) -> str:
    text = value.replace("-", "_").replace(" ", "_")
    category_map = {
        "earnings": "earnings_miss",
        "earnings_miss": "earnings_miss",
        "guidance": "guidance_cut",
        "guidance_cut": "guidance_cut",
        "downgrade": "analyst_downgrade",
        "analyst_downgrade": "analyst_downgrade",
        "litigation": "litigation",
        "regulatory": "regulatory_investigation",
        "regulatory_investigation": "regulatory_investigation",
        "fraud": "fraud_allegation",
        "fraud_allegation": "fraud_allegation",
        "accounting": "accounting_restatement",
        "accounting_restatement": "accounting_restatement",
        "bankruptcy": "bankruptcy_or_liquidity_warning",
        "liquidity_warning": "bankruptcy_or_liquidity_warning",
        "management": "management_departure",
        "management_departure": "management_departure",
        "merger": "merger_or_acquisition",
        "acquisition": "merger_or_acquisition",
        "product_failure": "product_failure",
        "clinical_trial_failure": "clinical_trial_failure",
        "operational_disruption": "temporary_operational_disruption",
        "temporary_operational_disruption": "temporary_operational_disruption",
    }
    return category_map.get(text, text or "general_negative_sentiment_or_uncategorized")


def _event_category_policies() -> dict[str, str]:
    return {
        "earnings_miss": "REQUIRE_CONFIRMATION",
        "guidance_cut": "REQUIRE_CONFIRMATION",
        "analyst_downgrade": "CONTRARIAN_ALLOWED",
        "litigation": "RISK_ONLY",
        "regulatory_investigation": "RISK_ONLY",
        "fraud_allegation": "EXCLUDED",
        "accounting_restatement": "EXCLUDED",
        "bankruptcy_or_liquidity_warning": "EXCLUDED",
        "management_departure": "REQUIRE_CONFIRMATION",
        "merger_or_acquisition": "RISK_ONLY",
        "product_failure": "REQUIRE_CONFIRMATION",
        "clinical_trial_failure": "RISK_ONLY",
        "temporary_operational_disruption": "CONTRARIAN_ALLOWED",
        "general_negative_sentiment_or_uncategorized": "RISK_ONLY",
    }


def _contrarian_suitability(
    category: str,
    returns: list[float],
    maes: list[float | None],
) -> str:
    policy = _event_category_policies().get(category, "RISK_ONLY")
    if policy == "EXCLUDED":
        return "excluded_by_policy"
    if not returns:
        return "unavailable"
    if mean(returns) > 0 and min((value for value in maes if value is not None), default=0.0) > -0.10:
        return "possible_with_confirmation"
    return "risk_only"


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    x_mean = mean(x)
    y_mean = mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_var = sum((a - x_mean) ** 2 for a in x)
    y_var = sum((b - y_mean) ** 2 for b in y)
    denominator = math.sqrt(x_var * y_var)
    return numerator / denominator if denominator else 0.0


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    return _pearson(_ranks(x), _ranks(y))


def _ranks(values: list[float]) -> list[float]:
    ranked = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0 for _ in values]
    position = 0
    while position < len(ranked):
        end = position + 1
        while end < len(ranked) and ranked[end][0] == ranked[position][0]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for _, index in ranked[position:end]:
            ranks[index] = average_rank
        position = end
    return ranks


def _monotonicity(rows: list[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row.get(field, 0.0)) for row in rows if row.get("candidate_count", 0)]
    if len(values) < 2:
        return {"direction": "unavailable", "violations": 0}
    increasing_violations = sum(1 for previous, current in zip(values, values[1:]) if current < previous)
    decreasing_violations = sum(1 for previous, current in zip(values, values[1:]) if current > previous)
    direction = "increasing" if increasing_violations <= decreasing_violations else "decreasing"
    return {
        "direction": direction,
        "violations": min(increasing_violations, decreasing_violations),
        "value_by_decile": values,
    }


def _mean_ci(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "lower_95": 0.0, "upper_95": 0.0}
    avg = mean(values)
    if len(values) < 2:
        return {"mean": avg, "lower_95": avg, "upper_95": avg}
    half_width = 1.96 * pstdev(values) / math.sqrt(len(values))
    return {"mean": avg, "lower_95": avg - half_width, "upper_95": avg + half_width}


def _row_group_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    returns = [_first_numeric(row, RETURN_COLUMNS) or 0.0 for row in rows]
    return {
        "count": len(rows),
        "average_return": mean(returns) if returns else 0.0,
        "median_return": median(returns) if returns else 0.0,
        "hit_rate": sum(value > 0 for value in returns) / max(len(returns), 1),
    }


def _filtered_group_summary(
    rows: list[Mapping[str, Any]],
    availability: Mapping[str, str | None],
    *,
    large_liquid: bool = False,
    resilient: bool = False,
    smaller_or_less_liquid: bool = False,
) -> dict[str, Any]:
    if large_liquid and not availability.get("dollar_trading_volume"):
        return {"status": "unavailable", "reason": "dollar trading volume field unavailable"}
    if resilient and not any(availability.get(name) for name in ("profitability", "free_cash_flow", "leverage", "interest_coverage")):
        return {"status": "unavailable", "reason": "financial resilience fields unavailable"}
    if smaller_or_less_liquid and not availability.get("dollar_trading_volume"):
        return {"status": "unavailable", "reason": "dollar trading volume field unavailable"}
    field = availability.get("dollar_trading_volume") if (large_liquid or smaller_or_less_liquid) else None
    selected = rows
    if field:
        values = sorted(_number(row.get(field)) or 0.0 for row in rows)
        cutoff = values[int(0.7 * (len(values) - 1))] if values else 0.0
        selected = [
            row for row in rows
            if ((_number(row.get(field)) or 0.0) >= cutoff) != smaller_or_less_liquid
        ]
    summary = _row_group_summary(selected)
    summary["status"] = "computed"
    return summary


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parallel_config(config: Mapping[str, Any]) -> NewsRiskParallelConfig:
    cpu_count = os.cpu_count() or 1
    requested = _optional_int(
        config.get("news_risk_max_workers")
        or config.get("stock_alpha_news_risk_overlay_parallel_max_workers")
    )
    configured_backend = str(
        config.get("news_risk_parallel_backend")
        or config.get("stock_alpha_news_risk_overlay_parallel_backend")
        or "thread"
    ).lower()
    backend = configured_backend if configured_backend in {"thread", "process"} else "thread"
    max_allowed = max(cpu_count - 1, 1)
    requested_workers = requested if requested is not None else max_allowed
    actual_workers = max(1, min(int(requested_workers), max_allowed))
    enabled = bool(
        config.get("news_risk_parallel_enabled")
        or config.get("stock_alpha_news_risk_overlay_parallel_enabled")
        or False
    )
    fallback_reason = None
    if not enabled:
        fallback_reason = "parallel disabled by configuration"
    elif actual_workers <= 1:
        fallback_reason = "single-worker mode requested"
    return NewsRiskParallelConfig(
        enabled=enabled,
        requested_workers=requested,
        actual_workers=actual_workers,
        backend=backend,
        min_items=max(
            1,
            int(
                config.get("news_risk_parallel_min_items")
                or config.get("stock_alpha_news_risk_overlay_parallel_min_items")
                or 16
            ),
        ),
        chunk_size=max(
            1,
            int(
                config.get("news_risk_parallel_chunk_size")
                or config.get("stock_alpha_news_risk_overlay_parallel_chunk_size")
                or 32
            ),
        ),
        batch_limit=max(
            1,
            int(
                config.get("news_risk_parallel_batch_limit")
                or config.get("stock_alpha_news_risk_overlay_parallel_batch_limit")
                or 128
            ),
        ),
        progress=bool(
            config.get("news_risk_parallel_progress")
            or config.get("stock_alpha_news_risk_overlay_parallel_progress")
            or False
        ),
        cpu_count=cpu_count,
        fallback_reason=fallback_reason,
    )


def _parallel_report_skeleton(config: NewsRiskParallelConfig) -> dict[str, Any]:
    return {
        "parallel_enabled": config.enabled,
        "backend": config.backend,
        "requested_workers": config.requested_workers,
        "actual_workers": config.actual_workers,
        "cpu_count": config.cpu_count,
        "task_count": 0,
        "chunk_size": config.chunk_size,
        "batch_limit": config.batch_limit,
        "phases_parallelised": [],
        "phases_kept_sequential": [],
        "elapsed_seconds_by_phase": {},
        "worker_count_used": config.actual_workers,
        "number_of_tasks": {},
        "average_task_duration_seconds": {},
        "slowest_tasks": {},
        "worker_failures": [],
        "fallback_events": (
            [{"phase": "global", "reason": config.fallback_reason}]
            if config.fallback_reason
            else []
        ),
        "determinism_status": "PENDING",
        "phases_forced_sequential": [
            "chronological_model_fitting",
            "point_in_time_join",
            "single_strategy_daily_portfolio_replay",
            "shared_ledger_or_equity_state",
            "broker_paper_live_order_paths",
        ],
        "paper_orders_enabled": False,
        "live_orders_enabled": False,
    }


@contextmanager
def _timed_phase(report: dict[str, Any] | None, phase: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        if report is not None:
            elapsed = time.perf_counter() - started
            report.setdefault("elapsed_seconds_by_phase", {})[phase] = (
                report.setdefault("elapsed_seconds_by_phase", {}).get(phase, 0.0) + elapsed
            )


def _should_parallelize(
    config: NewsRiskParallelConfig,
    item_count: int,
    *,
    phase: str,
    report: dict[str, Any] | None,
) -> bool:
    if not config.enabled:
        _record_fallback(report, phase, "parallel disabled by configuration")
        return False
    if config.actual_workers <= 1:
        _record_fallback(report, phase, "single-worker mode")
        return False
    if item_count < config.min_items:
        _record_fallback(report, phase, f"item_count {item_count} below min_items {config.min_items}")
        return False
    if phase == "bar_loading" and config.backend != "thread":
        _record_fallback(report, phase, "bar loading uses thread backend only because parquet reads are I/O-bound")
        return False
    return True


def _record_parallel_phase(
    report: dict[str, Any] | None,
    phase: str,
    *,
    task_count: int,
    task_durations: list[float],
    parallelized: bool,
) -> None:
    if report is None:
        return
    report["task_count"] = int(report.get("task_count", 0)) + task_count
    report.setdefault("number_of_tasks", {})[phase] = task_count
    if task_durations:
        report.setdefault("average_task_duration_seconds", {})[phase] = mean(task_durations)
        report.setdefault("slowest_tasks", {})[phase] = sorted(task_durations, reverse=True)[:5]
    target = "phases_parallelised" if parallelized else "phases_kept_sequential"
    if phase not in report.setdefault(target, []):
        report[target].append(phase)


def _record_fallback(report: dict[str, Any] | None, phase: str, reason: str) -> None:
    if report is None:
        return
    event = {"phase": phase, "reason": reason}
    if event not in report.setdefault("fallback_events", []):
        report["fallback_events"].append(event)
    if phase not in report.setdefault("phases_kept_sequential", []):
        report["phases_kept_sequential"].append(phase)


def _record_worker_failures(
    report: dict[str, Any] | None,
    phase: str,
    failures: list[Mapping[str, Any]],
) -> None:
    if report is None:
        return
    for failure in failures:
        report.setdefault("worker_failures", []).append({"phase": phase, **dict(failure)})


def _parallel_determinism_status(report: Mapping[str, Any]) -> str:
    return "FAILED_WORKER" if report.get("worker_failures") else "STABLE_ORDERING_ENFORCED"


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), max(size, 1)):
        yield items[index : index + max(size, 1)]


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_daily_price_bar_file(symbol: str, processed_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    path = processed_root / str(symbol).upper() / "1Day" / "bars.parquet"
    if not path.exists():
        return {"symbol": str(symbol).upper(), "status": "MISSING", "rows": [], "elapsed_seconds": time.perf_counter() - started}
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"])
        payload = table.to_pydict()
        rows = []
        for idx, timestamp in enumerate(payload["timestamp"]):
            high = float(payload["high"][idx])
            low = float(payload["low"][idx])
            if high < low:
                return {
                    "symbol": str(symbol).upper(),
                    "status": "FAILED",
                    "rows": [],
                    "error": f"malformed OHLC row high < low in {path}",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            rows.append(
                {
                    "date": _date_key(timestamp),
                    "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
                    "open": float(payload["open"][idx]),
                    "high": high,
                    "low": low,
                    "close": float(payload["close"][idx]),
                    "volume": float(payload["volume"][idx] or 0.0),
                    "symbol": str(payload["symbol"][idx]).upper(),
                }
            )
    except ImportError as exc:
        return {"symbol": str(symbol).upper(), "status": "FAILED", "rows": [], "error": "pyarrow unavailable to read daily bars", "elapsed_seconds": time.perf_counter() - started}
    except Exception as exc:
        return {"symbol": str(symbol).upper(), "status": "FAILED", "rows": [], "error": str(exc), "elapsed_seconds": time.perf_counter() - started}
    return {"symbol": str(symbol).upper(), "status": "OK", "rows": sorted(rows, key=lambda row: row["date"]), "elapsed_seconds": time.perf_counter() - started}


def _bar_sets_equal(
    left: Mapping[str, list[Mapping[str, Any]]],
    right: Mapping[str, list[Mapping[str, Any]]],
) -> bool:
    return _bar_set_digest(left) == _bar_set_digest(right)


def _bar_set_digest(payload: Mapping[str, list[Mapping[str, Any]]]) -> list[tuple[Any, ...]]:
    digest = []
    for symbol in sorted(payload):
        for row in sorted(payload[symbol], key=lambda item: str(item.get("date", ""))):
            digest.append(
                (
                    symbol,
                    row.get("date"),
                    row.get("timestamp"),
                    float(row.get("open", 0.0)),
                    float(row.get("high", 0.0)),
                    float(row.get("low", 0.0)),
                    float(row.get("close", 0.0)),
                    float(row.get("volume", 0.0)),
                )
            )
    return digest


def _replay_assumptions(
    replay_config: Mapping[str, Any],
    price_score_column: str,
    processed_root: Path,
) -> dict[str, Any]:
    return {
        "research_only": True,
        "broker_invoked": False,
        "orders_submitted": False,
        "price_data_root": str(processed_root),
        "price_timeframe": "1Day",
        "candidate_ranking": f"descending {price_score_column}, tie-break by symbol",
        "direction": "long_only",
        "entry_timing": "next available daily bar after decision date",
        "entry_price_convention": "next-session open plus entry_slippage_bps",
        "exit_rules": "stop if configured, then target if configured, otherwise max_holding_bars close",
        "same_bar_stop_target_ordering": "conservative_stop_first",
        "position_sizing": "min(available cash, equity * max_position_weight * news multiplier)",
        "maximum_positions": replay_config["max_positions"],
        "cash_allocation": "cash is debited at entry and unused cash remains in portfolio",
        "stop_loss_pct": replay_config["stop_loss_pct"],
        "profit_target_pct": replay_config["profit_target_pct"],
        "maximum_holding_bars": replay_config["max_holding_bars"],
        "commission_bps": replay_config["commission_bps"],
        "entry_slippage_bps": replay_config["entry_slippage_bps"],
        "exit_slippage_bps": replay_config["exit_slippage_bps"],
        "remaining_approximations": [
            "No intraday ordering is available for daily bars.",
            "Stop/target defaults are unset because the selected stock-alpha artifact has no explicit stop/target columns.",
            "Delisting handling exits at end_of_data when no later bars are available.",
        ],
    }


def _timestamp(row: Mapping[str, Any]) -> datetime:
    for column in ("decision_timestamp", *DECISION_TIMESTAMP_COLUMNS):
        value = row.get(column)
        if value:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    raise ValueError("row missing decision timestamp")


def _date_key(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value)
    return text[:10]


def _bar_on(
    symbol: str,
    date_key: str,
    bars_by_symbol: Mapping[str, list[Mapping[str, Any]]],
) -> Mapping[str, Any] | None:
    for bar in bars_by_symbol.get(str(symbol).upper(), []):
        if bar["date"] == date_key:
            return bar
    return None


def _bar_lookup(
    bars_by_symbol: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    return {
        symbol: {str(row["date"]): row for row in rows}
        for symbol, rows in bars_by_symbol.items()
    }


def _next_bar_lookup(
    bars_by_symbol: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for symbol, rows in bars_by_symbol.items():
        dates = [str(row["date"]) for row in rows]
        mapping = {}
        for index, value in enumerate(dates[:-1]):
            mapping[value] = dates[index + 1]
        for index, value in enumerate(dates):
            mapping.setdefault(value, dates[index + 1] if index + 1 < len(dates) else "")
        output[symbol] = mapping
    return output


def _bar_on_fast(
    symbol: str,
    date_key: str,
    lookup: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Mapping[str, Any] | None:
    return lookup.get(str(symbol).upper(), {}).get(date_key)


def _next_bar_date_fast(
    symbol: str,
    decision_date: str,
    lookup: Mapping[str, Mapping[str, str]],
) -> str | None:
    symbol_lookup = lookup.get(str(symbol).upper(), {})
    direct = symbol_lookup.get(decision_date)
    if direct:
        return direct
    later = [value for value in symbol_lookup if value > decision_date]
    if not later:
        return None
    return min(later)


def _next_bar_date(
    symbol: str,
    decision_date: str,
    bars_by_symbol: Mapping[str, list[Mapping[str, Any]]],
) -> str | None:
    for bar in bars_by_symbol.get(str(symbol).upper(), []):
        if bar["date"] > decision_date:
            return str(bar["date"])
    return None


def _last_bar(
    symbol: str,
    bars_by_symbol: Mapping[str, list[Mapping[str, Any]]],
) -> Mapping[str, Any] | None:
    rows = bars_by_symbol.get(str(symbol).upper(), [])
    return rows[-1] if rows else None


def _has_symbol(
    candidate: Mapping[str, Any],
    open_positions: list[Mapping[str, Any]],
    pending_entries: list[Mapping[str, Any]],
) -> bool:
    symbol = str(candidate.get("symbol", "")).upper()
    return any(row["symbol"] == symbol for row in open_positions) or any(
        row["symbol"] == symbol for row in pending_entries
    )


def _equity(
    cash: float,
    open_positions: list[Mapping[str, Any]],
    bars_by_symbol: Mapping[str, list[Mapping[str, Any]]],
    current_date: str,
) -> float:
    value = cash
    for position in open_positions:
        bar = _bar_on(position["symbol"], current_date, bars_by_symbol)
        mark = float(bar["close"]) if bar else float(position["entry_price"])
        value += float(position["shares"]) * mark
    return value


def _equity_fast(
    cash: float,
    open_positions: list[Mapping[str, Any]],
    lookup: Mapping[str, Mapping[str, Mapping[str, Any]]],
    current_date: str,
) -> float:
    value = cash
    for position in open_positions:
        bar = _bar_on_fast(position["symbol"], current_date, lookup)
        mark = float(bar["close"]) if bar else float(position["entry_price"])
        value += float(position["shares"]) * mark
    return value


def _action_event(
    candidate: Mapping[str, Any],
    variant: str,
    action: str,
    blocked: bool,
    ranking_after_news: int | None = None,
) -> dict[str, Any]:
    return {
        "strategy_variant": variant,
        "decision_timestamp": candidate.get("decision_timestamp", candidate.get("rebalance_date", "")),
        "symbol": candidate.get("symbol", ""),
        "news_action": action,
        "blocked": blocked,
        "candidate_forward_return": candidate.get("actual_forward_return_10d", candidate.get("actual_forward_return_5d", "")),
        "news_coverage": candidate.get("news_coverage_status", "NO_COVERAGE"),
        "news_risk_probability": candidate.get("price_plus_news_risk_probability", ""),
        "price_model_score": candidate.get("score", ""),
        "maximum_adverse_excursion": _adverse_excursion(candidate),
        "maximum_favourable_excursion": _favourable_excursion(candidate),
        "ranking_after_news": ranking_after_news if ranking_after_news is not None else "",
    }


def _drawdowns(equity: list[float]) -> list[float]:
    peak = equity[0] if equity else 0.0
    values = []
    for value in equity:
        peak = max(peak, value)
        values.append(value / peak - 1.0 if peak else 0.0)
    return values


def _longest_drawdown_duration(drawdowns: list[float]) -> int:
    longest = 0
    current = 0
    for value in drawdowns:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _worst_rolling_return(returns: list[float], window: int) -> float:
    if not returns:
        return 0.0
    if len(returns) < window:
        return math.prod(1.0 + value for value in returns) - 1.0
    return min(
        math.prod(1.0 + value for value in returns[index : index + window]) - 1.0
        for index in range(0, len(returns) - window + 1)
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _check_output_disk_space(output_dir: Path, ml: Mapping[str, Any]) -> None:
    estimated_bytes = int(ml.get("stock_alpha_news_risk_overlay_estimated_output_bytes", 10_000_000))
    minimum_free_bytes = int(
        ml.get(
            "stock_alpha_news_risk_overlay_min_free_bytes",
            max(25_000_000, estimated_bytes * 2),
        )
    )
    target = output_dir if output_dir.exists() else output_dir.parent
    while not target.exists() and target != target.parent:
        target = target.parent
    free_bytes = shutil.disk_usage(target).free
    if free_bytes < minimum_free_bytes:
        raise ValueError(
            "insufficient disk space for stock-alpha news risk overlay outputs: "
            f"free_bytes={free_bytes}, required_free_bytes={minimum_free_bytes}, "
            f"estimated_output_bytes={estimated_bytes}. "
            "No files were deleted automatically. Safe cleanup options: remove or move "
            "old untracked research-results outputs, old generated reports under reports/ml, "
            "or external local caches after reviewing them."
        )


def _limited_rows(rows: list[Mapping[str, Any]], limit: int) -> list[Mapping[str, Any]]:
    if limit <= 0:
        return rows
    return rows[:limit]


def _limited_audit_details(payload: Mapping[str, Any], limit: int) -> dict[str, Any]:
    output = dict(payload)
    details = output.get("max_news_timestamp_by_decision")
    if isinstance(details, list) and limit > 0:
        output["max_news_timestamp_by_decision_total_count"] = len(details)
        output["max_news_timestamp_by_decision"] = details[:limit]
    return output


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _accounting_definitions() -> dict[str, Any]:
    return {
        "trade_return_net": (
            "Realized candidate forward return after overlay exposure multiplier, "
            "transaction cost and slippage. This is trade-level, not portfolio total return."
        ),
        "starting_equity": "Equity assigned to the first decision-period basket.",
        "ending_equity": "Equity after compounding decision-period portfolio returns.",
        "total_return_decimal": "ending_equity / starting_equity - 1",
        "total_return_percent": "100 * total_return_decimal",
        "wealth_multiple": "ending_equity / starting_equity",
        "CAGR": "wealth_multiple ** (1 / years) - 1, using 252 decision periods per year.",
        "transaction_costs": "Configured transaction_cost_bps plus slippage_bps, applied to absolute exposure.",
        "blocked_trade_handling": "BLOCK sets exposure to 0, so gross/net return and costs are 0 for that candidate.",
        "reduced_trade_handling": "REDUCE multiplies gross return, exposure and costs by the configured exposure multiplier.",
        "overlapping_positions": (
            "Approximated as one equal-weight basket per decision timestamp because the current "
            "candidate artifacts do not include full open-position daily mark-to-market paths."
        ),
    }


def _accounting_audit(portfolio: Mapping[str, Any]) -> dict[str, Any]:
    price_only = dict(portfolio["price_only"])
    price_plus_news = dict(portfolio["price_plus_news"])
    return {
        "accounting_audit_version": "stock-alpha-news-risk-overlay-accounting-v1",
        "is_full_marked_to_market_portfolio_backtest": False,
        "return_series_type": "compounded_decision_period_basket_returns",
        "return_arithmetic_or_compounded": "compounded",
        "total_return_formula": "ending_equity / starting_equity - 1",
        "wealth_multiple_formula": "ending_equity / starting_equity",
        "decision_frequency": "one decision-period basket per unique decision timestamp/date",
        "candidate_aggregation_method": (
            "Candidates are sorted by the configured price score, the top_n rows are selected, "
            "and each selected candidate receives equal basket weight for that decision period."
        ),
        "trade_return_net_definition": (
            "candidate forward return * exposure multiplier - transaction_cost - slippage"
        ),
        "blocked_trade_treatment": "BLOCK sets exposure multiplier to 0.0 and contributes no return or cost.",
        "reduced_trade_treatment": (
            "REDUCE multiplies candidate return and costs by the configured reduce multiplier."
        ),
        "overlapping_trades_represented": False,
        "overlapping_trade_note": (
            "The current report does not maintain an open-position book. It compounds "
            "decision-period baskets and therefore remains an approximation."
        ),
        "unused_cash_represented": "partially",
        "unused_cash_note": (
            "Blocked/reduced exposure lowers basket exposure, but idle cash earns zero and "
            "cash constraints are not yet simulated with an explicit cash ledger."
        ),
        "replacement_candidates_selected": False,
        "replacement_candidate_note": (
            "Blocked candidates are not replaced by the next-ranked candidate in the current approximation."
        ),
        "transaction_cost_bps": portfolio["transaction_cost_bps"],
        "slippage_bps": portfolio["slippage_bps"],
        "price_only": _accounting_summary(price_only),
        "price_plus_news": _accounting_summary(price_plus_news),
        "plain_english_answer": {
            "question": "What exactly does an ending equity of 120.2441 mean?",
            "answer": _ending_equity_answer(price_only),
        },
        "news_overlay_lowered_drawdown": portfolio["news_overlay_lowered_drawdown"],
        "drawdown_change_percentage_points": (
            price_plus_news["maximum_drawdown"] - price_only["maximum_drawdown"]
        )
        * 100.0,
        "accounting_approximation": portfolio["accounting_approximation"],
    }


def _accounting_summary(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "starting_equity": stats["starting_equity"],
        "ending_equity": stats["ending_equity"],
        "total_return_decimal": stats["total_return_decimal"],
        "total_return_percent": stats["total_return_percent"],
        "wealth_multiple": stats["wealth_multiple"],
        "CAGR": stats["CAGR"],
        "maximum_drawdown": stats["maximum_drawdown"],
    }


def _ending_equity_answer(stats: Mapping[str, Any]) -> str:
    return (
        f"When starting_equity is {stats['starting_equity']:.4f}, ending_equity "
        f"of {stats['ending_equity']:.4f} means a {stats['wealth_multiple']:.4f} "
        f"wealth multiple and a {stats['total_return_percent']:.2f}% total return. "
        "This is compounded from decision-period basket returns; it is not a full "
        "marked-to-market open-position portfolio backtest."
    )


def _score_direction_markdown(report: Mapping[str, Any]) -> str:
    answers = dict(report.get("answers", {}) or {})
    return "\n".join(
        [
            "# News Score Direction Summary",
            "",
            f"- Candidate count: `{report.get('candidate_count', 0)}`",
            f"- Spearman score vs future return: `{report.get('spearman_news_score_vs_future_return', 0.0)}`",
            f"- Correlation score vs MAE: `{report.get('correlation_news_score_vs_maximum_adverse_excursion', 0.0)}`",
            f"- Correlation score vs MFE: `{report.get('correlation_news_score_vs_maximum_favourable_excursion', 0.0)}`",
            f"- Higher score predicts lower return: `{answers.get('higher_score_predicts_lower_return', False)}`",
            f"- Higher score predicts deeper temporary drawdown: `{answers.get('higher_score_predicts_deeper_temporary_drawdown', False)}`",
            f"- Higher score predicts movement both ways: `{answers.get('higher_score_predicts_greater_movement_both_directions', False)}`",
            f"- Observed relationship supports inversion: `{answers.get('relationship_supports_inversion', False)}`",
            "",
        ]
    )


def _markdown(
    manifest: Mapping[str, Any],
    coverage: Mapping[str, Any],
    metrics: Mapping[str, Any],
    portfolio: Mapping[str, Any],
    replay: Mapping[str, Any],
    score_direction_report: Mapping[str, Any],
    contrarian_report: Mapping[str, Any],
    cost_scenarios: Mapping[str, Any],
) -> str:
    replay_metrics = replay.get("risk_metrics", {})
    price_only_replay = replay_metrics.get("price_only", {})
    news_cash_replay = replay_metrics.get("news_cash", {})
    news_replacement_replay = replay_metrics.get("news_replacement", {})
    attribution = replay.get("action_attribution", {})
    blocked = attribution.get("BLOCK", {})
    score_answers = dict(score_direction_report.get("answers", {}) or {})
    cost_keys = sorted(dict(cost_scenarios.get("scenarios", {}) or {}).keys())
    return "\n".join(
        [
            "# Stock-Alpha News Risk Overlay Research",
            "",
            "Research-only historical comparison. Transformer training and paper orders are disabled.",
            "",
            f"- Price candidates: `{manifest['price_candidates_path']}`",
            f"- News features: `{manifest['news_features_path']}`",
            f"- Coverage ratio: `{coverage['row_coverage_ratio']:.4f}`",
            f"- Price-only ROC AUC: `{metrics['price_only']['roc_auc']:.4f}`",
            f"- Price-plus-news ROC AUC: `{metrics['price_plus_news']['roc_auc']:.4f}`",
            f"- Price-only total return: `{portfolio['price_only']['total_return_percent']:.2f}%`",
            f"- Price-plus-news total return: `{portfolio['price_plus_news']['total_return_percent']:.2f}%`",
            f"- News lowered drawdown: `{portfolio['news_overlay_lowered_drawdown']}`",
            f"- Incremental total return: `{portfolio['incremental_total_return_decimal']:.6f}`",
            f"- Transaction cost bps: `{portfolio['transaction_cost_bps']}`",
            f"- Slippage bps: `{portfolio['slippage_bps']}`",
            "",
            "Accounting: total return means `ending_equity / starting_equity - 1`; "
            "wealth multiple means `ending_equity / starting_equity`. Summed trade "
            "returns are not labelled as portfolio total return.",
            "",
            "What exactly does the return number mean?",
            "",
            _ending_equity_answer(portfolio["price_only"]),
            "",
            "Are the decision-period accounting returns based on a genuine portfolio replay?",
            "",
            "No. The decision-period accounting returns are still an approximation, not a full marked-to-market "
            "open-trade replay with overlapping positions, explicit cash and replacement logic.",
            "",
            f"Approximation: {portfolio['accounting_approximation']}",
            "",
            "## Phase 2 Open-Trade Replay",
            "",
            f"- Genuine marked-to-market replay: `{replay['portfolio_comparison']['is_genuine_marked_to_market_portfolio_replay']}`",
            f"- Price-only ending equity: `{price_only_replay.get('ending_equity')}`",
            f"- News-cash ending equity: `{news_cash_replay.get('ending_equity')}`",
            f"- News-replacement ending equity: `{news_replacement_replay.get('ending_equity')}`",
            f"- Price-only max drawdown: `{price_only_replay.get('maximum_drawdown')}`",
            f"- News-cash max drawdown: `{news_cash_replay.get('maximum_drawdown')}`",
            f"- Replacement max drawdown: `{news_replacement_replay.get('maximum_drawdown')}`",
            f"- Trade ledger rows: `{len(replay.get('trade_ledger', []))}`",
            f"- Losing blocked candidates: `{blocked.get('losing_trades_blocked', 0)}`",
            f"- Profitable blocked candidates: `{blocked.get('profitable_trades_blocked', 0)}`",
            f"- Score supports inversion: `{score_answers.get('relationship_supports_inversion', False)}`",
            f"- Extreme event entry enabled: `{contrarian_report.get('extreme_event_entry', {}).get('enabled', False)}`",
            f"- Cost scenarios: `{', '.join(cost_keys)}`",
            "",
            "The open-trade replay uses next-session open entries, daily close marks, "
            "cash debits at entry, unused cash preservation, max-position limits, "
            "and time exits after the configured holding period. Paper and live "
            "order control remain disabled.",
            "",
            "## Executive Questions",
            "",
            f"1. Current score direction correct: `{not score_answers.get('relationship_supports_inversion', False)}`",
            f"2. Predicting downside: `{score_answers.get('higher_score_predicts_lower_return', False)}`; "
            f"temporary drawdown: `{score_answers.get('higher_score_predicts_deeper_temporary_drawdown', False)}`; "
            f"movement both ways: `{score_answers.get('higher_score_predicts_greater_movement_both_directions', False)}`",
            f"3. Reversing actions improves results: inspect `contrarian_strategy_comparison.json` `diagnostic_variants.news_inverted_gate` vs `price_only`.",
            f"4. Contrarian re-ranking improves on price-only: inspect `diagnostic_variants.news_contrarian_rerank` vs `price_only`.",
            "5. Extreme negative news can originate trades: disabled by default; inspect `extreme_event_entry.enabled` and safeguards.",
            "6. Event categories with rebound behaviour: inspect `event_category_analysis.json` `contrarian_suitability`.",
            "7. Delayed entry safer than immediate: inspect `price_stabilisation_comparison.json`; not computed until enabled.",
            "8. Resilient company response: inspect `resilience_filter_analysis.json`; unavailable fields are reported explicitly.",
            "9. Robust after costs: inspect `cost_scenario_comparison.json`.",
            "10. Proceed to scikit-learn/transformer comparisons: only if score direction and cost scenarios are stable.",
            "",
            "Run:",
            "",
            "```bash",
            'PYTHONDONTWRITEBYTECODE=1 "$PY" main.py --mode ml-stock-alpha-news-risk-overlay-research',
            "```",
            "",
        ]
    )


def _existing(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _optional_path(value: Any) -> Path | None:
    text = str(value or "").strip()
    return Path(text).expanduser() if text else None


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _sigmoid(value: float) -> float:
    value = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-value))
