from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Callable, Iterable, Mapping, Sequence

from core.research.ml.stock_level.news_risk_overlay import (
    build_news_risk_labels,
    chronological_splits,
    join_news_to_stock_alpha_observations,
)
from core.research.ml.stock_level import news_risk_overlay_research_inputs as _inputs_module
from core.research.ml.stock_level.news_risk_overlay_research_accounting import (
    adverse_excursion as _adverse_excursion,
    drawdown_curve as _drawdown_curve,
    equity_curve as _equity_curve,
    expected_shortfall_cvar as _expected_shortfall,
    merge_curves as _merge_curves,
    period_accounting_row as _period_accounting_row,
    portfolio_comparison as _portfolio_comparison,
    portfolio_stats as _portfolio_stats,
)
from core.research.ml.stock_level.news_risk_overlay_research_artifacts import write_news_risk_research_artifacts
from core.research.ml.stock_level.news_risk_overlay_research_audit import *
from core.research.ml.stock_level.news_risk_overlay_research_audit import (
    _hypothetical_trade_ledger,
    _missing_news_bias,
    _stable_hash,
    _text_model_readiness,
)
from core.research.ml.stock_level.news_risk_overlay_research_catastrophic import (
    CATASTROPHIC_POLICY_VARIANTS,
    DISTRESSED_DILUTION_EVENT_CATEGORIES,
    DISTRESSED_DILUTION_TERMS,
    EXTREME_DISTRESS_EVENT_CATEGORIES,
    EXTREME_DISTRESS_ONLY_TERMS,
    EXTREME_DISTRESS_TERMS,
    FRAUD_EVENT_CATEGORIES,
    FRAUD_TERMS,
    SEVERE_LOSS_AVOIDANCE_TERMS,
    SOFT_RISK_REDUCE_TERMS,
    _as_optional_float,
    _avg_keyword,
    _bounceback_label,
    _case_keyword_scores,
    _case_value,
    _casebook_case,
    _catastrophic_classification_for_candidate,
    _catastrophic_classification_for_trade,
    _catastrophic_news_artifacts,
    _catastrophic_news_evidence_quality_artifacts,
    _catastrophic_policy_variant_artifacts,
    _catastrophic_trade_return,
    _catastrophic_veto_bounceback_artifacts,
    _catastrophic_veto_extreme_only_policy_proposal,
    _catastrophic_veto_full_replay_blocker,
    _catastrophic_veto_loser_bounceback_casebook_artifacts,
    _catastrophic_veto_policy,
    _catastrophic_veto_replay_seam_report,
    _catastrophic_veto_removal_reason,
    _catastrophic_veto_removed_symbol_rows,
    _catastrophic_veto_strategy_artifacts,
    _category_attribution_rows,
    _classify_event_taxonomy_from_headline,
    _difference,
    _event_category_for_candidate,
    _feature_diff_row,
    _filing_forms_detected,
    _headline_text,
    _is_generic_filing_headline,
    _mapping_first,
    _metric_delta,
    _metric_value,
    _news_duplicate_grouping_artifacts,
    _news_event_taxonomy_artifacts,
    _news_point_in_time_text_safety_artifacts,
    _news_text_keyword_baseline_artifacts,
    _normalized_headline,
    _parse_optional_timestamp,
    _policy_variant_blocks_candidate,
    _policy_variant_examples,
    _policy_variant_spec,
    _policy_variant_trade_rows,
    _rate,
    _severity_group_for_candidate,
    _strict_veto_breadth_diagnostic,
    _trade_key,
    apply_catastrophic_policy_variant_to_candidates,
    apply_catastrophic_veto_to_candidates,
)
from core.research.ml.stock_level.news_risk_overlay_research_decisions import (
    apply_news_decisions as _apply_news_decisions,
    apply_probabilities as _apply_probabilities,
    assign_candidate_ids as _assign_candidate_ids,
)
from core.research.ml.stock_level.news_risk_overlay_research_diagnostics import *
from core.research.ml.stock_level.news_risk_overlay_research_diagnostics import (
    _assert_score_direction_contract,
    _contrarian_strategy_report,
    _cost_scenario_comparison,
    _event_category_analysis,
    _extreme_event_archive,
    _news_score_decile_diagnostics,
    _price_stabilisation_report,
    _replay_action_attribution,
    _resilience_filter_analysis,
    _score_direction_audit,
)
from core.research.ml.stock_level.news_risk_overlay_research_evidence import *
from core.research.ml.stock_level.news_risk_overlay_research_evidence import (
    _news_evidence_lineage_artifacts,
    _optional_csv_stage,
)
from core.research.ml.stock_level.news_risk_overlay_research_inputs import *
from core.research.ml.stock_level.news_risk_overlay_research_inputs import (
    _build_labeled_news_risk_dataset,
    _build_news_risk_overlay_config,
    _load_news_risk_research_inputs,
    _locate_price_candidates,
    _resolve_news_risk_runtime_config,
    _select_news_risk_features,
)
from core.research.ml.stock_level.news_risk_overlay_research_inspection import (
    _catastrophic_veto_summary_lines,
)
from core.research.ml.stock_level.news_risk_overlay_research_manifest import build_news_risk_metrics_and_manifest
from core.research.ml.stock_level.news_risk_overlay_research_model import (
    classification_metrics as _classification_metrics,
    fit_logistic as _fit_logistic,
    predict_logistic as _predict_logistic,
    roc_auc as _roc_auc,
    walk_forward_logistic as _walk_forward_logistic,
)
from core.research.ml.stock_level.news_risk_overlay_research_parallel import (
    NewsRiskParallelConfig,
    chunks as _chunks,
    optional_int as _optional_int,
    parallel_config as _parallel_config,
    parallel_determinism_status as _parallel_determinism_status,
    parallel_report_skeleton as _parallel_report_skeleton,
    record_fallback as _record_fallback,
    record_parallel_phase as _record_parallel_phase,
    record_worker_failures as _record_worker_failures,
    should_parallelize as _should_parallelize,
    timed_phase as _timed_phase,
)
from core.research.ml.stock_level.news_risk_overlay_research_paths import NewsRiskResearchPaths, build_news_risk_research_paths
from core.research.ml.stock_level.news_risk_overlay_research_replay import (
    _action_attribution,
    _bar_set_digest,
    _bar_sets_equal,
    _daily_risk_metrics,
    _load_daily_price_bar_file,
    _load_daily_price_bars as _load_daily_price_bars_impl,
    _replay_assumptions,
    _run_open_trade_replay,
    _variant_multiplier,
)
from core.research.ml.stock_level.news_risk_overlay_research_reports import build_news_risk_validation_and_evidence_reports
from core.research.ml.stock_level.news_risk_overlay_research_robustness import *
from core.research.ml.stock_level.news_risk_overlay_research_selection import *
from core.research.ml.stock_level.news_risk_overlay_research_selection import (
    _append_experiment_registry_entry,
)
from core.research.ml.stock_level.news_risk_overlay_research_summary import *
from core.research.ml.stock_level.news_risk_overlay_research_summary import (
    _accounting_audit,
    _accounting_definitions,
    _markdown,
    _score_direction_markdown,
)
from core.research.ml.stock_level.news_risk_overlay_research_utils import *
from core.research.ml.stock_level.news_risk_overlay_research_utils import (
    _check_output_disk_space,
    _limited_audit_details,
    _limited_rows,
    _number,
    _read_csv,
    _write_csv,
    _write_json,
)
from core.research.ml.stock_level.news_risk_overlay_research_validation import _contrarian_validation_stage_reports
from core.research.ml.stock_level.news_risk_overlay_research_variants import (
    ResearchCandidateFilterSpec,
    ResearchStrategyVariantSpec,
    build_news_risk_research_variants,
    build_research_strategy_variant_inputs,
)
from core.research.ml.stock_level.news_risk_overlay_research_workflow import (
    _artifact_lineage_report,
    _artifact_validation_report,
    _news_validation_gap_analysis,
    _news_validation_workflow_map,
    _research_artifact_manifest,
    _validation_dependency_graph,
    _validation_readiness_dashboard,
    _validation_stage_placeholders,
    _workflow_node_warnings,
)


def _build_labeled_news_risk_dataset(
    price_rows: list[dict[str, str]],
    news_rows: list[dict[str, str]],
    overlay_config: Any,
    ml: Mapping[str, Any],
    parallel_report: dict[str, Any],
) -> dict[str, Any]:
    _inputs_module.join_news_to_stock_alpha_observations = join_news_to_stock_alpha_observations
    _inputs_module.build_news_risk_labels = build_news_risk_labels
    return _inputs_module._build_labeled_news_risk_dataset(
        price_rows,
        news_rows,
        overlay_config,
        ml,
        parallel_report,
    )


@dataclass(frozen=True)
class NewsRiskParallelBenchmarkPaths:
    output_dir: Path
    report_json_path: Path


def write_stock_alpha_news_risk_overlay_research(
    config: Mapping[str, Any],
) -> NewsRiskResearchPaths:
    run_started = time.perf_counter()
    runtime = _resolve_news_risk_runtime_config(config)
    ml = runtime["ml"]
    parallel_config = runtime["parallel_config"]
    parallel_report = runtime["parallel_report"]
    output_dir = runtime["output_dir"]
    with _timed_phase(parallel_report, "sequential_preflight"):
        _check_output_disk_space(output_dir, ml)
    output_dir.mkdir(parents=True, exist_ok=True)

    price_path, news_path, price_rows, news_rows = _load_news_risk_research_inputs(
        config,
        ml,
        parallel_report,
    )
    overlay_config = _build_news_risk_overlay_config(ml)
    labeled_dataset = _build_labeled_news_risk_dataset(
        price_rows,
        news_rows,
        overlay_config,
        ml,
        parallel_report,
    )
    labeled = labeled_dataset["labeled"]
    leakage = labeled_dataset["leakage"]
    coverage = labeled_dataset["coverage"]

    folds = int(ml.get("stock_alpha_news_risk_overlay_walk_forward_folds", 3))
    embargo_days = int(ml.get("stock_alpha_news_risk_overlay_embargo_days", 0))
    dataset_max_rows = int(ml.get("stock_alpha_news_risk_overlay_dataset_max_rows", 5000))
    shadow_max_rows = int(ml.get("stock_alpha_news_risk_overlay_shadow_max_rows", 5000))
    audit_detail_max_rows = int(ml.get("stock_alpha_news_risk_overlay_audit_detail_max_rows", 1000))
    selected_features = _select_news_risk_features(labeled, ml)
    price_score_column = selected_features["price_score_column"]
    return_column = selected_features["return_column"]
    price_feature_columns = selected_features["price_feature_columns"]
    price_news_feature_columns = selected_features["price_news_feature_columns"]

    splits = chronological_splits(labeled, folds=folds, embargo_days=embargo_days)
    if not splits:
        raise ValueError("not enough timestamped rows for chronological walk-forward splits")

    diagnostics = _run_news_risk_model_diagnostics(
        labeled,
        splits,
        price_feature_columns,
        price_news_feature_columns,
        ml,
        overlay_config,
        price_score_column,
        parallel_report,
    )
    price_metrics = diagnostics["price_metrics"]
    news_metrics = diagnostics["news_metrics"]
    decision_rows = diagnostics["decision_rows"]
    oos_rows = diagnostics["oos_rows"]
    portfolio = _portfolio_comparison(
        oos_rows,
        price_score_column=price_score_column,
        return_column=return_column,
        top_n=int(ml.get("stock_alpha_news_risk_overlay_portfolio_top_n", 25)),
        starting_equity=float(ml.get("stock_alpha_news_risk_overlay_starting_equity", 1.0)),
        transaction_cost_bps=float(ml.get("stock_alpha_news_risk_overlay_transaction_cost_bps", 0.0)),
        slippage_bps=float(ml.get("stock_alpha_news_risk_overlay_slippage_bps", 0.0)),
    )
    variants = build_news_risk_research_variants(
        oos_rows,
        apply_catastrophic_veto_to_candidates=apply_catastrophic_veto_to_candidates,
        apply_catastrophic_policy_variant_to_candidates=apply_catastrophic_policy_variant_to_candidates,
        catastrophic_policy_variants=CATASTROPHIC_POLICY_VARIANTS,
    )
    replay = _run_news_risk_replay(
        oos_rows=oos_rows,
        ml=ml,
        price_score_column=price_score_column,
        output_dir=output_dir,
        extra_research_variants=variants["extra_research_variants"],
        parallel_config=parallel_config,
        parallel_report=parallel_report,
    )
    diagnostic_reports = _build_news_risk_model_diagnostic_reports(
        oos_rows=oos_rows,
        replay=replay,
        overlay_config=overlay_config,
        price_score_column=price_score_column,
        ml=ml,
        parallel_report=parallel_report,
    )
    score_direction_audit = diagnostic_reports["score_direction_audit"]
    score_decile_rows = diagnostic_reports["score_decile_rows"]
    score_direction_report = diagnostic_reports["score_direction_report"]
    decile_join_audit = diagnostic_reports["decile_join_audit"]
    decile_reconciliation = diagnostic_reports["decile_reconciliation"]
    replay_action_attribution = diagnostic_reports["replay_action_attribution"]
    event_category_analysis = diagnostic_reports["event_category_analysis"]
    contrarian_report = diagnostic_reports["contrarian_report"]
    price_stabilisation = diagnostic_reports["price_stabilisation"]
    resilience_analysis = diagnostic_reports["resilience_analysis"]
    extreme_archive_rows = diagnostic_reports["extreme_archive_rows"]
    extreme_memory_report = diagnostic_reports["extreme_memory_report"]
    cost_scenarios = _build_news_risk_cost_scenarios(
        oos_rows=oos_rows,
        replay=replay,
        price_score_column=price_score_column,
        parallel_config=parallel_config,
        parallel_report=parallel_report,
    )
    validation = build_news_risk_validation_and_evidence_reports(
        oos_rows=oos_rows,
        replay=replay,
        price_score_column=price_score_column,
        ml=ml,
        cost_scenarios=cost_scenarios,
        news_path=news_path,
        news_rows=news_rows,
        labeled=labeled,
        contrarian_validation_stage_reports=_contrarian_validation_stage_reports,
        optional_csv_stage=_optional_csv_stage,
        news_evidence_lineage_artifacts=_news_evidence_lineage_artifacts,
    )
    paths = build_news_risk_research_paths(output_dir)
    metrics, manifest = build_news_risk_metrics_and_manifest(
        price_path=price_path,
        news_path=news_path,
        output_dir=output_dir,
        price_score_column=price_score_column,
        return_column=return_column,
        labeled=labeled,
        decision_rows=decision_rows,
        dataset_max_rows=dataset_max_rows,
        shadow_max_rows=shadow_max_rows,
        price_metrics=price_metrics,
        news_metrics=news_metrics,
        price_feature_columns=price_feature_columns,
        price_news_feature_columns=price_news_feature_columns,
        label_source_columns=LABEL_SOURCE_COLUMNS,
        limited_rows=_limited_rows,
    )
    with _timed_phase(parallel_report, "report_writing"):
        write_news_risk_research_artifacts(
            paths=paths,
            labeled=labeled,
            dataset_max_rows=dataset_max_rows,
            decision_rows=decision_rows,
            shadow_max_rows=shadow_max_rows,
            coverage=coverage,
            leakage=leakage,
            audit_detail_max_rows=audit_detail_max_rows,
            metrics=metrics,
            portfolio=portfolio,
            replay=replay,
            score_direction_audit=score_direction_audit,
            score_decile_rows=score_decile_rows,
            decile_join_audit=decile_join_audit,
            decile_reconciliation=decile_reconciliation,
            score_direction_report=score_direction_report,
            replay_action_attribution=replay_action_attribution,
            event_category_analysis=event_category_analysis,
            contrarian_report=contrarian_report,
            price_stabilisation=price_stabilisation,
            resilience_analysis=resilience_analysis,
            extreme_archive_rows=extreme_archive_rows,
            extreme_memory_report=extreme_memory_report,
            cost_scenarios=cost_scenarios,
            validation=validation,
            oos_rows=oos_rows,
            manifest=manifest,
            config=ml,
            write_csv=_write_csv,
            write_json=_write_json,
            limited_rows=_limited_rows,
            limited_audit_details=_limited_audit_details,
            accounting_definitions=_accounting_definitions,
            accounting_audit=_accounting_audit,
            score_direction_markdown=_score_direction_markdown,
            append_experiment_registry_entry=_append_experiment_registry_entry,
            research_artifact_manifest=_research_artifact_manifest,
            artifact_validation_report=_artifact_validation_report,
            news_validation_workflow_map=_news_validation_workflow_map,
            validation_dependency_graph=_validation_dependency_graph,
            validation_readiness_dashboard=_validation_readiness_dashboard,
            artifact_lineage_report=_artifact_lineage_report,
            news_validation_gap_analysis=_news_validation_gap_analysis,
            markdown=_markdown,
        )
    parallel_report["elapsed_seconds_total"] = time.perf_counter() - run_started
    parallel_report["determinism_status"] = _parallel_determinism_status(parallel_report)
    _write_json(paths.parallel_execution_report_json_path, parallel_report)
    return paths


def _run_news_risk_model_diagnostics(
    labeled: list[dict[str, Any]],
    splits: list[tuple[list[int], list[int]]],
    price_feature_columns: list[str],
    price_news_feature_columns: list[str],
    ml: Mapping[str, Any],
    overlay_config: NewsRiskOverlayConfig,
    price_score_column: str,
    parallel_report: dict[str, Any],
) -> dict[str, Any]:
    learning_rate = float(ml.get("stock_alpha_news_risk_overlay_learning_rate", 0.05))
    epochs = int(ml.get("stock_alpha_news_risk_overlay_epochs", 60))
    l2 = float(ml.get("stock_alpha_news_risk_overlay_l2", 0.01))
    max_train_rows = int(ml.get("stock_alpha_news_risk_overlay_max_train_rows", 12000))
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
        _assign_candidate_ids(labeled, price_score_column)
        decision_rows = _apply_news_decisions(labeled, overlay_config, price_score_column)
        oos_rows = [row for index, row in enumerate(labeled) if index in news_probs]
    return {
        "price_metrics": price_metrics,
        "news_metrics": news_metrics,
        "decision_rows": decision_rows,
        "oos_rows": oos_rows,
    }


def _run_news_risk_replay(
    *,
    oos_rows: list[dict[str, Any]],
    ml: Mapping[str, Any],
    price_score_column: str,
    output_dir: Path,
    extra_research_variants: Sequence[ResearchStrategyVariantSpec],
    parallel_config: NewsRiskParallelConfig,
    parallel_report: dict[str, Any],
) -> Mapping[str, Any]:
    with _timed_phase(parallel_report, "replay"):
        return _build_open_trade_replay(
            oos_rows,
            config=ml,
            price_score_column=price_score_column,
            output_dir=output_dir,
            extra_research_variants=extra_research_variants,
            parallel_config=parallel_config,
            parallel_report=parallel_report,
        )


def _build_news_risk_model_diagnostic_reports(
    *,
    oos_rows: list[dict[str, Any]],
    replay: Mapping[str, Any],
    overlay_config: NewsRiskOverlayConfig,
    price_score_column: str,
    ml: Mapping[str, Any],
    parallel_report: dict[str, Any],
) -> dict[str, Any]:
    with _timed_phase(parallel_report, "model_diagnostic_reports"):
        score_direction_audit = _score_direction_audit(
            rows=oos_rows,
            config=overlay_config,
            target_column="news_risk_label",
        )
        _assert_score_direction_contract(score_direction_audit, oos_rows)
        score_decile_rows, score_direction_report, decile_join_audit, decile_reconciliation = _news_score_decile_diagnostics(
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
    return {
        "score_direction_audit": score_direction_audit,
        "score_decile_rows": score_decile_rows,
        "score_direction_report": score_direction_report,
        "decile_join_audit": decile_join_audit,
        "decile_reconciliation": decile_reconciliation,
        "replay_action_attribution": replay_action_attribution,
        "event_category_analysis": event_category_analysis,
        "contrarian_report": contrarian_report,
        "price_stabilisation": price_stabilisation,
        "resilience_analysis": resilience_analysis,
        "extreme_archive_rows": extreme_archive_rows,
        "extreme_memory_report": extreme_memory_report,
    }


def _build_news_risk_cost_scenarios(
    *,
    oos_rows: list[dict[str, Any]],
    replay: Mapping[str, Any],
    price_score_column: str,
    parallel_config: NewsRiskParallelConfig,
    parallel_report: dict[str, Any],
) -> Mapping[str, Any]:
    with _timed_phase(parallel_report, "cost_scenarios"):
        return _cost_scenario_comparison(
            oos_rows,
            bars_by_symbol=replay["bars_by_symbol"],
            price_score_column=price_score_column,
            base_replay_config=replay["replay_config"],
            parallel_config=parallel_config,
            parallel_report=parallel_report,
        )


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


def _build_open_trade_replay(
    rows: list[dict[str, Any]],
    *,
    config: Mapping[str, Any],
    price_score_column: str,
    output_dir: Path,
    extra_research_variants: Sequence[ResearchStrategyVariantSpec] | None = None,
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
    extra_variant_metadata = _append_extra_research_variant_results(
        base_rows=rows,
        extra_research_variants=extra_research_variants,
        base_variant_settings=variants,
        bars_by_symbol=bars_by_symbol,
        price_score_column=price_score_column,
        replay_config=replay_config,
        ledgers=ledgers,
        curves=curves,
        attribution_inputs=attribution_inputs,
    )
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
        **(
            {"extra_research_variant_metadata": extra_variant_metadata}
            if extra_variant_metadata
            else {}
        ),
        "bars_by_symbol": bars_by_symbol,
        "action_events": attribution_inputs,
        "hypothetical_trade_ledger": hypothetical,
        "contrarian_trade_ledger": [row for row in ledgers if row["strategy_variant"] in {"news_inverted_gate", "news_contrarian_rerank"}],
        "replay_assumptions": assumptions,
        "replay_data_audit": data_audit,
    }


def _append_extra_research_variant_results(
    *,
    base_rows: Sequence[Mapping[str, Any]],
    extra_research_variants: Sequence[ResearchStrategyVariantSpec] | None,
    base_variant_settings: Mapping[str, Mapping[str, Any]],
    bars_by_symbol: Mapping[str, list[Mapping[str, Any]]],
    price_score_column: str,
    replay_config: Mapping[str, Any],
    ledgers: list[dict[str, Any]],
    curves: dict[str, list[dict[str, Any]]],
    attribution_inputs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for spec in extra_research_variants or ():
        if not spec.enabled_for_research:
            continue
        if not spec.research_only or spec.enabled_for_paper_trading or spec.enabled_for_live_trading:
            raise ValueError("extra replay variants must be research-only with paper/live disabled")
        if spec.new_variant_name in base_variant_settings or spec.new_variant_name in curves:
            raise ValueError(f"duplicate research strategy variant: {spec.new_variant_name}")
        if spec.base_variant_name not in base_variant_settings:
            raise ValueError(f"unknown base research strategy variant: {spec.base_variant_name}")

        variant_input = build_research_strategy_variant_inputs(base_rows, spec)
        settings = dict(base_variant_settings[spec.base_variant_name])
        settings.update(
            {
                "diagnostic_only": True,
                "research_only": True,
                "base_variant_name": spec.base_variant_name,
            }
        )
        result = _run_open_trade_replay(
            variant_input["candidate_rows"],
            bars_by_symbol=bars_by_symbol,
            price_score_column=price_score_column,
            variant=spec.new_variant_name,
            variant_settings=settings,
            replay_config=replay_config,
        )
        ledgers.extend(result["ledger"])
        curves[spec.new_variant_name] = result["daily_equity"]
        attribution_inputs.extend(result["action_events"])
        metadata[spec.new_variant_name] = {
            **variant_input,
            "candidate_rows": "COPIED_RESEARCH_INPUT",
        }
    return metadata


def _load_daily_price_bars(
    symbols: list[str],
    processed_root: Path,
    *,
    parallel_config: NewsRiskParallelConfig | None = None,
    parallel_report: dict[str, Any] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    return _load_daily_price_bars_impl(
        symbols,
        processed_root,
        parallel_config=parallel_config,
        parallel_report=parallel_report,
        load_daily_price_bar_file_fn=_load_daily_price_bar_file,
    )
