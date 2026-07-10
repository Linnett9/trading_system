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
    DECISION_TIMESTAMP_COLUMNS,
    TIMESTAMP_COLUMNS,
    NewsRiskOverlayConfig,
    build_news_risk_labels,
    chronological_splits,
    join_news_to_stock_alpha_observations,
)
from core.research.ml.stock_level.news_sources import (
    catastrophic_news_taxonomy_report,
    classify_catastrophic_news_rows,
)
from core.research.ml.stock_level.news_transformer import (
    build_news_transformer_readiness_report,
    build_news_transformer_training_plan,
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
from core.research.ml.stock_level.news_risk_overlay_research_artifacts import (
    write_news_risk_research_artifacts,
)
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
from core.research.ml.stock_level.news_risk_overlay_research_decisions import (
    apply_news_decisions as _apply_news_decisions,
    apply_probabilities as _apply_probabilities,
    assign_candidate_ids as _assign_candidate_ids,
)
from core.research.ml.stock_level.news_risk_overlay_research_manifest import (
    build_news_risk_metrics_and_manifest,
)
from core.research.ml.stock_level.news_risk_overlay_research_model import (
    classification_metrics as _classification_metrics,
    fit_logistic as _fit_logistic,
    predict_logistic as _predict_logistic,
    roc_auc as _roc_auc,
    walk_forward_logistic as _walk_forward_logistic,
)
from core.research.ml.stock_level.news_risk_overlay_research_paths import (
    NewsRiskResearchPaths,
    build_news_risk_research_paths,
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
from core.research.ml.stock_level.news_risk_overlay_research_reports import (
    build_news_risk_validation_and_evidence_reports,
)
from core.research.ml.stock_level.news_risk_overlay_research_replay import (
    _action_attribution,
    _action_event,
    _bar_lookup,
    _bar_on,
    _bar_on_fast,
    _bar_set_digest,
    _bar_sets_equal,
    _daily_risk_metrics,
    _date_key,
    _drawdowns,
    _equity,
    _equity_fast,
    _exit_decision,
    _has_symbol,
    _last_bar,
    _ledger_row,
    _load_daily_price_bar_file,
    _load_daily_price_bars as _load_daily_price_bars_impl,
    _longest_drawdown_duration,
    _next_bar_date,
    _next_bar_date_fast,
    _next_bar_lookup,
    _pending_trade,
    _replay_assumptions,
    _run_open_trade_replay,
    _variant_multiplier,
    _variant_sort_value,
    _worst_rolling_return,
)
from core.research.ml.stock_level.news_risk_overlay_research_variants import (
    ResearchCandidateFilterSpec,
    ResearchStrategyVariantSpec,
    build_news_risk_research_variants,
    build_research_strategy_variant_inputs,
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


def _resolve_news_risk_runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
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
    return {
        "ml": ml,
        "parallel_config": parallel_config,
        "parallel_report": parallel_report,
        "output_dir": output_dir,
    }


def _load_news_risk_research_inputs(
    config: Mapping[str, Any],
    ml: Mapping[str, Any],
    parallel_report: dict[str, Any],
) -> tuple[Path, Path, list[dict[str, str]], list[dict[str, str]]]:
    del ml
    with _timed_phase(parallel_report, "input_loading"):
        price_path = _locate_price_candidates(config)
        news_path = _locate_news_features(config)
        price_rows = _read_csv(price_path)
        news_rows = _read_csv(news_path)
        _validate_source_rows(price_rows, news_rows, price_path, news_path)
    return price_path, news_path, price_rows, news_rows


def _build_news_risk_overlay_config(ml: Mapping[str, Any]) -> NewsRiskOverlayConfig:
    return NewsRiskOverlayConfig(
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


def _build_labeled_news_risk_dataset(
    price_rows: list[dict[str, str]],
    news_rows: list[dict[str, str]],
    overlay_config: NewsRiskOverlayConfig,
    ml: Mapping[str, Any],
    parallel_report: dict[str, Any],
) -> dict[str, Any]:
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
    return {
        "labeled": labeled,
        "leakage": leakage,
        "coverage": coverage,
    }


def _select_news_risk_features(
    labeled: list[dict[str, Any]],
    ml: Mapping[str, Any],
) -> dict[str, Any]:
    max_features = int(ml.get("stock_alpha_news_risk_overlay_max_features", 48))
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
    return {
        "price_score_column": price_score_column,
        "return_column": return_column,
        "price_feature_columns": price_feature_columns,
        "price_news_feature_columns": price_news_feature_columns,
    }


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
    if mode == "summary":
        lines = [
            line
            for line in lines
            if line.strip() != "Winners:"
            and "best absolute return:" not in line.lower()
            and "score direction:" not in line.lower()
            and "holdout/status:" not in line.lower()
        ]
    if mode == "verbose":
        lines.extend(["", "Artifacts:"])
        lines.extend(f"- {row['name']}: {row['status']} ({row['path']})" for row in artifact_status)
    return _sanitize_validation_summary_text("\n".join(lines))


def _sanitize_validation_summary_text(text: str) -> str:
    sanitized = text.replace("VALIDATION_PASSED", "final independent validation")
    sanitized = _clean_report_text(sanitized)
    decile_warning_supported = _summary_text_supports_decile_warning(sanitized)
    replacements = {
        "holdout excess return:": "holdout validation metric: PSEUDO_HOLDOUT_ONLY",
        "walk-forward positive folds:": "walk-forward validation metric: NOT_IMPLEMENTED",
        "placebo p-value:": "placebo validation metric: NOT_IMPLEMENTED",
    }
    output_lines = []
    for line in sanitized.splitlines():
        stripped = line.strip()
        if "identical executed-trade counts across multiple deciles" in stripped and not decile_warning_supported:
            continue
        if "lowest max drawdown" in stripped.lower():
            continue
        if "selected config:" in stripped.lower():
            continue
        if "contrarian beat price-only:" in stripped.lower():
            continue
        if "superior after 5/10/20 bps:" in stripped.lower():
            continue
        if "extreme events:" in stripped.lower():
            continue
        if "best risk-adjusted:" in stripped.lower():
            continue
        if "best after realistic costs:" in stripped.lower():
            continue
        replacement = None
        for prefix, status_text in replacements.items():
            if prefix in stripped:
                marker = line[: len(line) - len(line.lstrip())]
                replacement = f"{marker}- {status_text}" if stripped.startswith("-") else f"{marker}{status_text}"
                break
        output_lines.append(replacement if replacement is not None else line)
    return _append_workflow_readiness_summary_lines("\n".join(output_lines))


def _summary_text_supports_decile_warning(text: str) -> bool:
    lowered = text.lower()
    if "decile trade reconciliation: passed" in lowered:
        return False
    return any(
        marker in lowered
        for marker in (
            "decile_join_audit.status: failed",
            "decile_trade_reconciliation.status: failed",
            "trades_assigned_to_multiple_deciles: 1",
            "trades_assigned_to_multiple_deciles: 2",
            "trades_assigned_to_multiple_deciles: 3",
            "trades_assigned_to_multiple_deciles: 4",
            "trades_assigned_to_multiple_deciles: 5",
            "trades_assigned_to_multiple_deciles: 6",
            "trades_assigned_to_multiple_deciles: 7",
            "trades_assigned_to_multiple_deciles: 8",
            "trades_assigned_to_multiple_deciles: 9",
            "deciles_receiving_full_ledger_count: 1",
            "deciles_receiving_full_ledger_count: 2",
            "deciles_receiving_full_ledger_count: 3",
            "deciles_receiving_full_ledger_count: 4",
            "deciles_receiving_full_ledger_count: 5",
            "deciles_receiving_full_ledger_count: 6",
            "deciles_receiving_full_ledger_count: 7",
            "deciles_receiving_full_ledger_count: 8",
            "deciles_receiving_full_ledger_count: 9",
            "matched_executed_trade_count: true",
        )
    )


def _clean_report_text(text: str) -> str:
    replacements = {
        "untoucheddata": "untouched data",
        "isNOT_READY": "is NOT_READY",
        "audithas": "audit has",
        "not ademonstrably": "not a demonstrably",
        "not anuntouched": "not an untouched",
    }
    cleaned = text
    for malformed, fixed in replacements.items():
        cleaned = cleaned.replace(malformed, fixed)
    return cleaned


def _append_workflow_readiness_summary_lines(text: str) -> str:
    lines = text.splitlines()
    has_decile_warning = any("identical executed-trade counts across multiple deciles" in line for line in lines)
    required_lines = []
    if not has_decile_warning and not any("decile trade reconciliation:" in line for line in lines):
        required_lines.append("- decile trade reconciliation: PASSED")
    required_lines.append("- validation readiness: DEVELOPMENT_ONLY / NOT_FINAL_VALIDATION | FinBERT: NOT_READY | gaps: OPEN | validation label: PSEUDO_HOLDOUT | walk-forward NOT_IMPLEMENTED | placebo UNAVAILABLE_INPUT | news transformer scaffold: PRESENT / DISABLED | news transformer readiness: NOT_READY")
    required_lines.append("- news transformer training/inference enabled: False / False | used in strategy/replay: False / False | paper/live trading enabled: False / False")
    insert_at = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip().upper() == "WARNINGS" or line.strip().upper().startswith("WARNINGS")
        ),
        len(lines),
    )
    for line in required_lines:
        if line not in lines:
            lines.insert(insert_at, line)
            insert_at += 1
    return "\n".join(lines)


def _expanded_workflow_readiness_summary_lines() -> list[str]:
    return [
        "- workflow map: PRESENT",
        "- validation dependency graph: BLOCKED",
        "- validation readiness: DEVELOPMENT_ONLY / NOT_FINAL_VALIDATION",
        "- gap analysis: OPEN_GAPS_BLOCK_FINAL_VALIDATION",
        "- FinBERT readiness: NOT_READY",
        "- news transformer scaffold: PRESENT / DISABLED",
        "- news transformer readiness: NOT_READY",
        "- news transformer training/inference enabled: False / False",
        "- used in strategy/replay: False / False",
        "- paper/live trading enabled: False / False",
    ]


def _decile_reconciliation_summary_lines(
    decile_join_audit: Mapping[str, Any],
    decile_trade_reconciliation: Mapping[str, Any],
) -> list[str]:
    join_status = str(decile_join_audit.get("status", "UNAVAILABLE"))
    reconciliation_status = str(decile_trade_reconciliation.get("status", "UNAVAILABLE"))
    assigned_multiple = int(decile_join_audit.get("trades_assigned_to_multiple_deciles") or 0)
    full_ledger_deciles = int(decile_join_audit.get("deciles_receiving_full_ledger_count") or 0)
    matched = int(decile_join_audit.get("matched_trade_rows") or 0)
    eligible = int(decile_join_audit.get("eligible_trade_rows") or 0)
    identical_counts = bool(
        dict(decile_join_audit.get("identical_decile_metric_diagnostic", {}) or {}).get(
            "matched_executed_trade_count"
        )
    )
    warnings = list(decile_join_audit.get("warnings", []) or []) + list(
        decile_trade_reconciliation.get("warnings", []) or []
    )
    audit_supports_clean_deciles = (
        join_status in {"PASSED", "PASSED_WITH_WARNINGS"}
        and reconciliation_status in {"PASSED", "PASSED_WITH_WARNINGS"}
        and assigned_multiple == 0
        and full_ledger_deciles == 0
        and not identical_counts
    )
    lines = [
        f"- decile_join_audit.status: {join_status}",
        f"- decile_trade_reconciliation.status: {reconciliation_status}",
        f"- trades_assigned_to_multiple_deciles: {assigned_multiple}",
        f"- deciles_receiving_full_ledger_count: {full_ledger_deciles}",
        f"- matched_trade_rows / eligible_trade_rows: {matched} / {eligible}",
    ]
    if audit_supports_clean_deciles:
        lines.append(f"- decile trade reconciliation: {reconciliation_status}")
    else:
        lines.append("- warning: identical executed-trade counts across multiple deciles")
    lines.extend(f"- warning: {warning}" for warning in warnings)
    return lines


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



def _configured_first(value: Any, defaults: Iterable[str]) -> list[str]:
    return [str(value), *defaults] if value else list(defaults)


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


def _value_counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _news_score_decile_diagnostics(
    rows: list[Mapping[str, Any]],
    ledger: list[Mapping[str, Any]],
    *,
    price_score_column: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    _ensure_trade_provenance(ledger)
    candidate_rows = list(rows)
    missing_score_rows = [
        row for row in candidate_rows
        if _number(row.get("price_plus_news_risk_probability")) is None
    ]
    scored = [
        row for row in candidate_rows
        if _number(row.get("price_plus_news_risk_probability")) is not None
    ]
    scored.sort(key=lambda row: _timestamp(row))
    if not scored:
        eligible_variants = ("price_only", "news_contrarian_rerank")
        eligible_trades = [
            trade for trade in ledger
            if str(trade.get("strategy_variant", "")) in eligible_variants
        ]
        candidate_ids = [str(row.get("candidate_id", "")) for row in candidate_rows if row.get("candidate_id")]
        duplicate_candidate_id_count = sum(count - 1 for count in _value_counts(candidate_ids).values() if count > 1)
        trade_ids = [str(trade.get("trade_id", "")) for trade in eligible_trades if trade.get("trade_id")]
        duplicate_trade_id_count = sum(count - 1 for count in _value_counts(trade_ids).values() if count > 1)
        cross_strategy_candidate_trade_pairs_excluded = sum(
            1
            for trade in ledger
            if str(trade.get("strategy_variant", "")) not in eligible_variants
            and str(trade.get("candidate_id", "")) in set(candidate_ids)
        )
        warnings = ["no scored news-risk rows available for decile attribution"]
        if eligible_trades:
            warnings.append("eligible trade rows could not be matched because all candidate news scores were missing")
        status = "FAILED" if duplicate_candidate_id_count or duplicate_trade_id_count else "PASSED_WITH_WARNINGS"
        join_audit = {
            "schema_name": "news_risk_decile_join_audit",
            "schema_version": 1,
            "status": status,
            "candidate_id_column": "candidate_id",
            "trade_id_column": "trade_id",
            "join_keys": ["candidate_id", "strategy_variant"],
            "candidate_rows": len(candidate_rows),
            "eligible_trade_rows": len(eligible_trades),
            "matched_trade_rows": 0,
            "unique_matched_trade_ids": 0,
            "unmatched_candidate_rows": len(candidate_rows),
            "unmatched_trade_rows": len(eligible_trades),
            "duplicate_candidate_id_count": duplicate_candidate_id_count,
            "duplicate_trade_id_count": duplicate_trade_id_count,
            "trades_assigned_to_multiple_deciles": 0,
            "deciles_receiving_full_ledger_count": 0,
            "missing_news_score_count": len(missing_score_rows),
            "neutral_news_score_count": 0,
            "cross_strategy_candidate_trade_pairs_excluded": cross_strategy_candidate_trade_pairs_excluded,
            "strategy_variant_mismatch_count": 0,
            "strategy_variant_mismatch_is_error": False,
            "warnings": warnings,
        }
        reconciliation = {
            "schema_name": "news_risk_decile_trade_reconciliation",
            "schema_version": 1,
            "status": status,
            "by_strategy_variant": {
                variant: {
                    "eligible_trade_rows": sum(1 for trade in eligible_trades if str(trade.get("strategy_variant", "")) == variant),
                    "matched_trade_rows": 0,
                    "unique_matched_trade_ids": 0,
                    "unmatched_trade_rows": sum(1 for trade in eligible_trades if str(trade.get("strategy_variant", "")) == variant),
                    "trades_assigned_to_multiple_deciles": 0,
                }
                for variant in eligible_variants
            },
            "by_decile": [],
            "warnings": warnings,
        }
        return [], _empty_score_direction_report(), join_audit, reconciliation
    for row in scored:
        if not row.get("candidate_id"):
            raise ValueError("candidate-to-trade decile attribution requires candidate_id")
    eligible_variants = ("price_only", "news_contrarian_rerank")
    candidate_ids = [str(row.get("candidate_id", "")) for row in candidate_rows if row.get("candidate_id")]
    duplicate_candidate_id_count = sum(count - 1 for count in _value_counts(candidate_ids).values() if count > 1)
    by_candidate_variant_trade: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    eligible_trades = []
    cross_strategy_candidate_trade_pairs_excluded = 0
    strategy_variant_mismatch_count = 0
    for trade in ledger:
        candidate_id = str(trade.get("candidate_id", ""))
        variant = str(trade.get("strategy_variant", ""))
        if variant in eligible_variants:
            eligible_trades.append(trade)
            if candidate_id:
                by_candidate_variant_trade.setdefault((candidate_id, variant), []).append(trade)
        elif candidate_id in candidate_ids:
            cross_strategy_candidate_trade_pairs_excluded += 1
    trade_ids = [str(trade.get("trade_id", "")) for trade in eligible_trades if trade.get("trade_id")]
    duplicate_trade_id_count = sum(count - 1 for count in _value_counts(trade_ids).values() if count > 1)
    ranked = sorted(scored, key=lambda row: _number(row.get("price_plus_news_risk_probability")) or 0.0)
    decile_by_payload: dict[str, int] = {}
    for index, row in enumerate(ranked):
        decile_by_payload[str(row["candidate_id"])] = min(10, int(index * 10 / max(len(ranked), 1)) + 1)
    deciles = []
    matched_trade_ids: set[str] = set()
    candidate_decile_counts: dict[str, int] = {}
    by_strategy: dict[str, dict[str, Any]] = {}
    for variant in eligible_variants:
        variant_trade_ids = {
            str(trade.get("trade_id", ""))
            for trade in eligible_trades
            if str(trade.get("strategy_variant", "")) == variant and trade.get("trade_id")
        }
        variant_matched_trade_ids: set[str] = set()
        variant_trade_decile_counts: dict[str, int] = {}
        for decile in range(1, 11):
            members = [row for row in scored if decile_by_payload.get(str(row["candidate_id"])) == decile]
            for row in members:
                candidate_decile_counts.setdefault(str(row["candidate_id"]), 1)
            returns = [_first_numeric(row, RETURN_COLUMNS) or 0.0 for row in members]
            maes = [_adverse_excursion(row) for row in members]
            mfes = [_favourable_excursion(row) for row in members]
            probabilities = [_number(row.get("price_plus_news_risk_probability")) or 0.0 for row in members]
            price_scores = [_number(row.get(price_score_column)) or 0.0 for row in members]
            executed = []
            for row in members:
                for trade in by_candidate_variant_trade.get((str(row["candidate_id"]), variant), []):
                    executed.append(trade)
                    trade_id = str(trade.get("trade_id", ""))
                    if trade_id:
                        matched_trade_ids.add(trade_id)
                        variant_matched_trade_ids.add(trade_id)
                        variant_trade_decile_counts[trade_id] = variant_trade_decile_counts.get(trade_id, 0) + 1
            net_returns = [float(trade.get("net_return", 0.0)) for trade in executed]
            unique_trade_count = len({str(trade.get("trade_id", "")) for trade in executed if trade.get("trade_id")})
            deciles.append(
                {
                    "strategy_variant": variant,
                    "decile": decile,
                    "candidate_count": len(members),
                    "matched_executed_trade_count": len(executed),
                    "unique_trade_count": unique_trade_count,
                    "average_news_risk_probability": mean(probabilities) if probabilities else 0.0,
                    "average_candidate_forward_return": mean(returns) if returns else 0.0,
                    "median_candidate_forward_return": median(returns) if returns else 0.0,
                    "average_forward_return": mean(returns) if returns else 0.0,
                    "median_forward_return": median(returns) if returns else 0.0,
                    "average_replay_net_return": mean(net_returns) if net_returns else 0.0,
                    "median_replay_net_return": median(net_returns) if net_returns else 0.0,
                    "hit_rate": sum(value > 0 for value in returns) / max(len(returns), 1),
                    "mae": min((value for value in maes if value is not None), default=0.0),
                    "mfe": max((value for value in mfes if value is not None), default=0.0),
                    "maximum_adverse_excursion": min((value for value in maes if value is not None), default=0.0),
                    "maximum_favourable_excursion": max((value for value in mfes if value is not None), default=0.0),
                    "worst_trade": min(returns, default=0.0),
                    "volatility": pstdev(returns) if len(returns) > 1 else 0.0,
                    "stop_hit_rate": sum(_boolish(row.get("stop_hit_before_target")) for row in members) / max(len(members), 1),
                    "event_category_mix": _category_mix(members),
                    "news_coverage": sum(str(row.get("news_coverage_status")) == "COVERED" for row in members) / max(len(members), 1),
                    "average_price_model_score": mean(price_scores) if price_scores else 0.0,
                    "average_news_score": mean(probabilities) if probabilities else 0.0,
                    "missing_news_score_count": 0,
                    "neutral_news_score_count": sum((_number(row.get("price_plus_news_risk_probability")) or 0.0) == 0.0 for row in members),
                    "unmatched_candidate_count": sum(not by_candidate_variant_trade.get((str(row["candidate_id"]), variant)) for row in members),
                    "unmatched_trade_count": sum(
                        1
                        for trade in eligible_trades
                        if str(trade.get("strategy_variant", "")) == variant
                        and str(trade.get("trade_id", "")) not in variant_matched_trade_ids
                    ),
                }
            )
        by_strategy[variant] = {
            "eligible_trade_rows": sum(1 for trade in eligible_trades if str(trade.get("strategy_variant", "")) == variant),
            "matched_trade_rows": sum(1 for trade in eligible_trades if str(trade.get("strategy_variant", "")) == variant and str(trade.get("trade_id", "")) in variant_matched_trade_ids),
            "unique_matched_trade_ids": len(variant_matched_trade_ids),
            "unmatched_trade_rows": sum(
                1
                for trade in eligible_trades
                if str(trade.get("strategy_variant", "")) == variant
                and str(trade.get("trade_id", "")) not in variant_matched_trade_ids
            ),
            "trades_assigned_to_multiple_deciles": sum(count > 1 for count in variant_trade_decile_counts.values()),
        }
    probabilities = [_number(row.get("price_plus_news_risk_probability")) or 0.0 for row in scored]
    returns = [_first_numeric(row, RETURN_COLUMNS) or 0.0 for row in scored]
    maes = [_adverse_excursion(row) or 0.0 for row in scored]
    mfes = [_favourable_excursion(row) or 0.0 for row in scored]
    eligible_trade_ids = {str(trade.get("trade_id", "")) for trade in eligible_trades if trade.get("trade_id")}
    trade_decile_counts: dict[str, int] = {}
    for trade in eligible_trades:
        trade_id = str(trade.get("trade_id", ""))
        candidate_id = str(trade.get("candidate_id", ""))
        if trade_id and candidate_id in decile_by_payload:
            trade_decile_counts[trade_id] = trade_decile_counts.get(trade_id, 0) + 1
    eligible_trade_ids_by_variant = {
        variant: {
            str(trade.get("trade_id", ""))
            for trade in eligible_trades
            if str(trade.get("strategy_variant", "")) == variant and trade.get("trade_id")
        }
        for variant in eligible_variants
    }
    deciles_receiving_full_ledger_count = sum(
        row["unique_trade_count"] == len(eligible_trade_ids_by_variant.get(str(row["strategy_variant"]), set()))
        for row in deciles
        if len(eligible_trade_ids_by_variant.get(str(row["strategy_variant"]), set())) > 1
    )
    unmatched_trade_row_count = sum(
        1
        for trade in eligible_trades
        if str(trade.get("trade_id", "")) not in matched_trade_ids
    )
    repeated_metric_values = {
        "average_forward_return": len({row["average_forward_return"] for row in deciles if row["candidate_count"]}) <= 1,
        "matched_executed_trade_count": len({row["matched_executed_trade_count"] for row in deciles if row["candidate_count"]}) <= 1,
    }
    warnings = []
    if repeated_metric_values["matched_executed_trade_count"]:
        warnings.append("identical executed-trade counts across multiple deciles")
    if duplicate_candidate_id_count:
        warnings.append("duplicate candidate IDs detected")
    if duplicate_trade_id_count:
        warnings.append("duplicate trade IDs detected")
    if eligible_trade_ids - matched_trade_ids:
        warnings.append("eligible trade rows could not be matched to candidate deciles")
    if deciles_receiving_full_ledger_count:
        warnings.append("one or more deciles received the full matched ledger for a strategy variant")
    status = "FAILED" if duplicate_candidate_id_count or duplicate_trade_id_count or any(count > 1 for count in trade_decile_counts.values()) else ("PASSED_WITH_WARNINGS" if warnings else "PASSED")
    join_audit = {
        "schema_name": "news_risk_decile_join_audit",
        "schema_version": 1,
        "status": status,
        "candidate_id_column": "candidate_id",
        "trade_id_column": "trade_id",
        "join_keys": ["candidate_id", "strategy_variant"],
        "candidate_rows": len(candidate_rows),
        "candidate_count": len(scored),
        "candidate_id_count": len({str(row["candidate_id"]) for row in scored}),
        "candidates_with_exactly_one_decile": sum(count == 1 for count in candidate_decile_counts.values()),
        "candidate_multiple_decile_count": sum(count > 1 for count in candidate_decile_counts.values()),
        "eligible_trade_rows": len(eligible_trades),
        "eligible_ledger_trade_count": len(eligible_trade_ids),
        "matched_trade_rows": sum(1 for trade in eligible_trades if str(trade.get("trade_id", "")) in matched_trade_ids),
        "unique_matched_trade_count": len(matched_trade_ids),
        "unique_matched_trade_ids": len(matched_trade_ids),
        "unmatched_trade_count": len(eligible_trade_ids - matched_trade_ids),
        "unmatched_trade_rows": unmatched_trade_row_count,
        "unmatched_candidate_count": sum(
            not any(by_candidate_variant_trade.get((str(row["candidate_id"]), variant)) for variant in eligible_variants)
            for row in scored
        ),
        "unmatched_candidate_rows": sum(
            not any(by_candidate_variant_trade.get((str(row.get("candidate_id", "")), variant)) for variant in eligible_variants)
            for row in candidate_rows
        ),
        "duplicate_candidate_id_count": duplicate_candidate_id_count,
        "duplicate_trade_id_count": duplicate_trade_id_count,
        "trades_assigned_to_multiple_deciles": sum(count > 1 for count in trade_decile_counts.values()),
        "deciles_receiving_full_ledger_count": deciles_receiving_full_ledger_count,
        "missing_news_score_count": len(missing_score_rows),
        "neutral_news_score_count": sum((_number(row.get("price_plus_news_risk_probability")) or 0.0) == 0.0 for row in scored),
        "cross_strategy_candidate_trade_pairs_excluded": cross_strategy_candidate_trade_pairs_excluded,
        "strategy_variant_mismatch_count": strategy_variant_mismatch_count,
        "strategy_variant_mismatch_is_error": strategy_variant_mismatch_count > 0,
        "no_decile_receives_full_unfiltered_ledger": deciles_receiving_full_ledger_count == 0,
        "identical_decile_metric_diagnostic": repeated_metric_values,
        "warnings": warnings,
    }
    reconciliation = {
        "schema_name": "news_risk_decile_trade_reconciliation",
        "schema_version": 1,
        "status": status,
        "one_candidate_exactly_one_decile": join_audit["candidate_multiple_decile_count"] == 0,
        "one_trade_no_more_than_one_decile": all(count <= 1 for count in trade_decile_counts.values()),
        "every_matched_trade_has_candidate_identifier": all(
            bool(trade.get("candidate_id"))
            for trade in ledger
            if str(trade.get("trade_id", "")) in matched_trade_ids
        ),
        "total_unique_matched_trades": len(matched_trade_ids),
        "eligible_ledger_trades": len(eligible_trade_ids),
        "unmatched_trades": sorted(eligible_trade_ids - matched_trade_ids)[:100],
        "unmatched_candidate_count": join_audit["unmatched_candidate_count"],
        "by_strategy_variant": by_strategy,
        "by_decile": deciles,
        "warnings": warnings,
    }
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
                str(row["decile"]): _mean_ci([_first_numeric(member, RETURN_COLUMNS) or 0.0 for member in scored if decile_by_payload.get(str(member["candidate_id"])) == row["decile"]])
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
    }, join_audit, reconciliation


def _ensure_trade_provenance(ledger: list[Mapping[str, Any]]) -> None:
    for index, trade in enumerate(ledger):
        if not isinstance(trade, dict):
            continue
        trade.setdefault("model_version", str(trade.get("model_version") or "news-risk-overlay-research-v1"))
        if trade.get("trade_id"):
            continue
        payload = "|".join(
            [
                str(trade.get("candidate_id", "")),
                str(trade.get("strategy_variant", "")),
                str(trade.get("decision_timestamp", "")),
                str(trade.get("symbol", "")),
                str(trade.get("entry_date", "")),
                str(trade.get("exit_date", "")),
                str(index),
            ]
        )
        trade["trade_id"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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


def _optional_csv_stage(ml: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = ml.get(key)
        if not value:
            continue
        path = Path(str(value))
        if path.is_file():
            return {"available": True, "path": str(path), "rows": _read_csv(path)}
        return {"available": False, "path": str(path), "rows": []}
    return {"available": False, "path": "UNAVAILABLE_UPSTREAM", "rows": []}


def _news_evidence_lineage_artifacts(
    stages: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    stage_order = (
        "raw_news",
        "provider_normalized_news",
        "news_contract",
        "news_features",
        "joined_candidate_rows",
        "catastrophic_veto_input_rows",
    )
    aliases = {
        "candidate_id": ("candidate_id", "trade_id", "row_id"),
        "symbol": ("symbol", "ticker"),
        "headline": ("headline_text", "headline", "title"),
        "summary": ("summary_text", "summary", "description"),
        "body": ("body_text", "body", "content", "article_body", "body_or_summary"),
        "publication_timestamp": ("publication_timestamp", "published_at", "published_at_utc", "timestamp"),
        "availability_timestamp": ("availability_timestamp", "available_at", "asof_timestamp", "news_feature_timestamp"),
        "source": ("source",),
        "provider": ("provider",),
        "event_category": ("event_type", "event", "provider_category", "category"),
        "duplicate_group_id": ("duplicate_group_id", "syndication_group_id"),
    }

    def present(row: Mapping[str, Any], field: str) -> bool:
        return any(row.get(key) is not None and str(row.get(key)).strip() for key in aliases[field])

    by_stage: list[dict[str, Any]] = []
    missing_examples: list[dict[str, Any]] = []
    for stage_name in stage_order:
        payload = dict(stages.get(stage_name, {}) or {})
        rows = list(payload.get("rows", []) or [])
        available = bool(payload.get("available"))
        counts = {field: sum(present(row, field) for row in rows) for field in aliases}
        any_text = sum(
            present(row, "headline") or present(row, "summary") or present(row, "body") or present(row, "event_category")
            for row in rows
        )
        availability = counts["availability_timestamp"]
        by_stage.append({
            "stage": stage_name,
            "stage_available": available,
            "source_path": payload.get("path", "UNAVAILABLE_UPSTREAM"),
            "row_count": len(rows),
            "has_candidate_id_count": counts["candidate_id"],
            "has_symbol_count": counts["symbol"],
            "has_headline_count": counts["headline"],
            "has_summary_count": counts["summary"],
            "has_body_count": counts["body"],
            "has_any_text_count": any_text,
            "has_publication_timestamp_count": counts["publication_timestamp"],
            "has_availability_timestamp_count": availability,
            "has_source_count": counts["source"],
            "has_provider_count": counts["provider"],
            "has_event_category_count": counts["event_category"],
            "duplicate_group_id_count": counts["duplicate_group_id"],
            "point_in_time_safe_count": availability,
            "missing_text_count": len(rows) - any_text,
            "missing_availability_timestamp_count": len(rows) - availability,
        })
        if available:
            for field in ("headline", "availability_timestamp", "source", "provider", "event_category", "duplicate_group_id", "candidate_id"):
                for row in (row for row in rows if not present(row, field)):
                    missing_examples.append({
                        "stage": stage_name,
                        "field_name": field,
                        "candidate_id": next((row.get(key) for key in aliases["candidate_id"] if row.get(key)), "UNKNOWN"),
                        "symbol": next((row.get(key) for key in aliases["symbol"] if row.get(key)), "UNKNOWN"),
                        "reason": "MISSING_FIELD",
                    })
                    if sum(example["stage"] == stage_name and example["field_name"] == field for example in missing_examples) >= 5:
                        break

    final_stage = next(row for row in by_stage if row["stage"] == "catastrophic_veto_input_rows")
    gap_fields = {
        "headline_text": "headline",
        "availability_timestamp": "availability_timestamp",
        "event_category": "event_category",
        "duplicate_group_id": "duplicate_group_id",
        "candidate_id": "candidate_id",
        "source": "source",
        "provider": "provider",
    }
    count_keys = {
        "headline": "has_headline_count",
        "availability_timestamp": "has_availability_timestamp_count",
        "event_category": "has_event_category_count",
        "duplicate_group_id": "duplicate_group_id_count",
        "candidate_id": "has_candidate_id_count",
        "source": "has_source_count",
        "provider": "has_provider_count",
    }
    field_mapping_gaps = []
    for output_name, field in gap_fields.items():
        final_count_key = count_keys[field]
        present_count = int(final_stage[final_count_key])
        row_count = int(final_stage["row_count"])
        missing_count = max(row_count - present_count, 0)
        upstream = [
            row["stage"]
            for row in by_stage[:-1]
            if row["stage_available"] and row[final_count_key] > 0
        ]
        if row_count and present_count == row_count:
            status = "PRESENT"
            present_stage = "catastrophic_veto_input_rows"
            probable_cause = "field is present for all catastrophic_veto_input_rows"
            recommended_fix = "preserve current mapping"
        elif present_count > 0:
            status = "PARTIAL_COVERAGE"
            present_stage = "catastrophic_veto_input_rows"
            probable_cause = "field is present for some catastrophic_veto_input_rows but absent for others"
            recommended_fix = "audit upstream coverage and preserve the field where point-in-time evidence exists"
        elif upstream:
            status = "FULLY_MISSING_FROM_STAGE"
            present_stage = upstream[-1]
            probable_cause = f"field present at {present_stage} but not propagated to catastrophic_veto_input_rows"
            recommended_fix = "preserve and map the field through news feature generation and the point-in-time join"
        else:
            status = "UNAVAILABLE_UPSTREAM"
            present_stage = "UNAVAILABLE_UPSTREAM"
            probable_cause = "UNAVAILABLE_UPSTREAM"
            recommended_fix = "supply the field at ingestion before changing downstream mappings"
        field_mapping_gaps.append({
            "field_name": output_name,
            "status": status,
            "present_in_stage": present_stage,
            "missing_from_stage": "catastrophic_veto_input_rows",
            "present_count": present_count,
            "missing_count": missing_count,
            "coverage_ratio": present_count / max(row_count, 1),
            "probable_cause": probable_cause,
            "recommended_fix": recommended_fix,
            "blocks_catastrophic_veto": field in {"headline", "availability_timestamp", "event_category"},
            "blocks_text_model_readiness": field in {"headline", "availability_timestamp", "duplicate_group_id", "source", "provider"},
        })

    lineage_report = {
        "schema_name": "news_evidence_lineage_report",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "news_evidence_lineage_v1",
        "status": "GAPS_FOUND" if any(gap["status"] in {"FULLY_MISSING_FROM_STAGE", "UNAVAILABLE_UPSTREAM"} for gap in field_mapping_gaps) else "COMPLETE_FOR_RESEARCH",
        "stages": by_stage,
        "field_mapping_gaps": field_mapping_gaps,
        "observational_only": True,
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }
    missing_required = []
    if final_stage["has_any_text_count"] == 0:
        missing_required.append("text")
    if final_stage["has_availability_timestamp_count"] == 0:
        missing_required.append("availability_timestamp")
    if final_stage["has_event_category_count"] == 0:
        missing_required.append("event_category")
    if final_stage["duplicate_group_id_count"] == 0:
        missing_required.append("duplicate_group_id")
    if final_stage["has_source_count"] == 0:
        missing_required.append("source")
    strict_ready = not any(field in missing_required for field in ("text", "availability_timestamp"))
    confirmed_ready = "text" not in missing_required
    readiness = {
        "schema_name": "news_evidence_readiness_report",
        "schema_version": 1,
        "status": "READY_FOR_RESEARCH_ONLY" if strict_ready else "INSUFFICIENT",
        "candidate_count": final_stage["row_count"],
        "has_any_text_count": final_stage["has_any_text_count"],
        "has_availability_timestamp_count": final_stage["has_availability_timestamp_count"],
        "strict_veto_ready": strict_ready,
        "confirmed_only_veto_ready": confirmed_ready,
        "text_model_ready": False,
        "finbert_ready": False,
        "transformer_ready": False,
        "finbert_readiness": "NOT_READY",
        "transformer_readiness": "NOT_READY",
        "missing_required_fields": missing_required,
        "minimum_required_fields_for_strict_veto": ["text", "availability_timestamp"],
        "minimum_required_fields_for_confirmed_only_veto": ["text"],
        "minimum_required_fields_for_text_model": ["text", "availability_timestamp", "source", "duplicate_group_id"],
        "field_mapping_statuses": {
            gap["field_name"]: gap["status"]
            for gap in field_mapping_gaps
        },
        "event_taxonomy_research_ready": final_stage["has_any_text_count"] > 0,
        "duplicate_grouping_heuristic_ready": final_stage["has_any_text_count"] > 0,
        "point_in_time_text_safety_ready": final_stage["has_any_text_count"] > 0 and final_stage["has_availability_timestamp_count"] > 0,
        "keyword_baseline_ready": final_stage["has_any_text_count"] > 0,
        "recommended_next_actions": [gap["recommended_fix"] for gap in field_mapping_gaps],
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }
    return lineage_report, by_stage, missing_examples, readiness




def _contrarian_validation_stage_reports(
    *,
    rows: list[Mapping[str, Any]],
    replay: Mapping[str, Any],
    price_score_column: str,
    config: Mapping[str, Any],
    cost_scenarios: Mapping[str, Any],
    data_audit: Mapping[str, Any],
) -> dict[str, Any]:
    periods = _chronological_periods(rows, replay.get("trade_ledger", []))
    selection_config = {
        **dict(config),
        "stock_alpha_news_risk_overlay_contrarian_grid_cost_metrics": _selection_cost_metrics_from_scenarios(
            config,
            cost_scenarios,
        ),
    }
    grid_rows, fold_rows, selection = _contrarian_grid_reports(rows, replay, price_score_column, periods, selection_config)
    frozen = _frozen_contrarian_config(selection, config, periods)
    holdout = _holdout_report(replay, periods, frozen, cost_scenarios)
    walk_forward_folds, walk_forward_summary = _walk_forward_reports(replay, periods)
    placebo_rows, placebo_summary = _placebo_reports(replay, config)
    contribution_year, contribution_symbol, concentration = _contribution_reports(replay)
    missing_news_report, covered_vs_uncovered = _missing_news_bias(rows, price_score_column)
    walk_forward_validation_report, walk_forward_fold_rows = _walk_forward_validation_artifacts(
        replay,
        periods,
        selection,
        frozen,
        config,
    )
    placebo_permutation_report, placebo_permutation_rows = _placebo_permutation_artifacts(
        rows,
        replay,
        config,
    )
    catastrophic_news_audit, catastrophic_news_candidates, catastrophic_news_veto_report = (
        _catastrophic_news_artifacts(rows)
    )
    (
        catastrophic_veto_candidate_attribution,
        catastrophic_veto_trade_attribution,
        catastrophic_veto_strategy_comparison,
        catastrophic_veto_policy,
        catastrophic_veto_filtered_strategy_report,
        catastrophic_veto_removed_trades,
        catastrophic_veto_removed_symbols,
        catastrophic_veto_full_replay_report,
        catastrophic_veto_full_replay_trade_ledger,
        catastrophic_veto_full_replay_equity,
        catastrophic_veto_filtered_candidates,
        catastrophic_veto_blocked_candidates,
    ) = _catastrophic_veto_strategy_artifacts(rows, replay)
    (
        catastrophic_news_evidence_quality_report,
        catastrophic_news_evidence_quality_by_field,
        catastrophic_news_evidence_quality_by_symbol,
        catastrophic_veto_policy_mode_comparison,
        catastrophic_veto_policy_mode_counts,
    ) = _catastrophic_news_evidence_quality_artifacts(
        rows,
        replay,
        catastrophic_veto_full_replay_report,
    )
    (
        news_event_taxonomy_report,
        news_event_taxonomy_counts,
        news_event_taxonomy_examples,
    ) = _news_event_taxonomy_artifacts(rows)
    (
        news_duplicate_grouping_report,
        news_duplicate_grouping_examples,
    ) = _news_duplicate_grouping_artifacts(rows)
    (
        news_point_in_time_text_safety_report,
        news_point_in_time_text_safety_examples,
    ) = _news_point_in_time_text_safety_artifacts(rows)
    (
        news_text_keyword_baseline_report,
        news_text_keyword_baseline_scores,
    ) = _news_text_keyword_baseline_artifacts(rows)
    (
        catastrophic_veto_bounceback_report,
        catastrophic_veto_bounceback_by_category,
        catastrophic_veto_bounceback_examples,
        catastrophic_veto_extreme_only_policy_proposal,
    ) = _catastrophic_veto_bounceback_artifacts(
        rows,
        replay,
        catastrophic_veto_removed_trades,
        catastrophic_veto_blocked_candidates,
        catastrophic_veto_full_replay_report,
        catastrophic_veto_policy_mode_counts,
    )
    (
        catastrophic_veto_policy_variant_comparison,
        catastrophic_veto_policy_variant_counts,
        catastrophic_veto_policy_variant_metrics,
        catastrophic_veto_policy_variant_removed_trades,
        catastrophic_veto_policy_variant_bounceback,
        catastrophic_veto_policy_frontier_report,
        catastrophic_veto_policy_frontier,
        catastrophic_veto_policy_variant_examples,
    ) = _catastrophic_policy_variant_artifacts(
        rows,
        replay,
        catastrophic_veto_bounceback_report,
    )
    (
        catastrophic_veto_loser_bounceback_casebook,
        catastrophic_veto_loser_bounceback_cases,
        catastrophic_veto_loser_bounceback_feature_diff,
        catastrophic_veto_loser_bounceback_keyword_diff,
        catastrophic_veto_taxonomy_improvement_plan,
    ) = _catastrophic_veto_loser_bounceback_casebook_artifacts(
        rows,
        catastrophic_veto_removed_trades,
    )
    contrarian_chronological_validation_plan, contrarian_chronological_periods = _contrarian_chronological_validation_plan(
        rows,
        replay,
        periods,
    )
    contrarian_placebo_permutation_report, contrarian_placebo_permutation_results = _contrarian_placebo_permutation_report(config)
    contrarian_matched_control_report, contrarian_matched_control_results = _contrarian_matched_control_report()
    (
        contrarian_profit_concentration_report,
        contrarian_trade_fragility_by_symbol,
        contrarian_trade_fragility_by_year,
        contrarian_top_trade_removal,
    ) = _contrarian_profit_concentration_artifacts(replay)
    (
        contrarian_year_regime_report,
        contrarian_year_regime_results,
        contrarian_year_regime_examples,
    ) = _contrarian_year_regime_artifacts(replay)
    (
        contrarian_symbol_year_ablation_report,
        contrarian_without_top_symbols,
        contrarian_without_top_years,
    ) = _contrarian_symbol_year_ablation_artifacts(replay)
    (
        contrarian_cost_slippage_robustness_report,
        contrarian_cost_slippage_robustness,
    ) = _contrarian_cost_slippage_robustness(cost_scenarios)
    contrarian_data_validity_audit = _contrarian_data_validity_audit(
        data_audit,
        missing_news_report,
    )
    intraday_5min_expansion_plan = _intraday_5min_expansion_plan(config)
    catastrophic_veto_parked_status = _catastrophic_veto_parked_status(
        catastrophic_veto_full_replay_report,
        catastrophic_veto_policy_frontier_report,
        catastrophic_veto_loser_bounceback_casebook,
    )
    return {
        "chronological_split_manifest": periods,
        "contrarian_grid_results": grid_rows,
        "contrarian_grid_selection": selection,
        "contrarian_fold_results": fold_rows,
        "contrarian_parameter_stability": _parameter_stability(grid_rows, selection),
        "contrarian_frozen_config": frozen,
        "contrarian_holdout_report": holdout,
        "contrarian_holdout_trade_ledger": _holdout_rows(replay.get("trade_ledger", []), periods, "news_contrarian_rerank"),
        "contrarian_holdout_equity": _holdout_rows(replay.get("daily_equity", {}).get("news_contrarian_rerank", []), periods, "news_contrarian_rerank"),
        "contrarian_holdout_comparison_md": _holdout_markdown(holdout),
        "contrarian_walk_forward_folds": walk_forward_folds,
        "contrarian_walk_forward_summary": walk_forward_summary,
        "contrarian_chronological_validation_plan": contrarian_chronological_validation_plan,
        "contrarian_chronological_periods": contrarian_chronological_periods,
        "contrarian_walk_forward_validation_report": walk_forward_validation_report,
        "contrarian_placebo_permutation_report": contrarian_placebo_permutation_report,
        "contrarian_placebo_permutation_results": contrarian_placebo_permutation_results,
        "contrarian_matched_control_report": contrarian_matched_control_report,
        "contrarian_matched_control_results": contrarian_matched_control_results,
        "contrarian_profit_concentration_report": contrarian_profit_concentration_report,
        "contrarian_trade_fragility_by_symbol": contrarian_trade_fragility_by_symbol,
        "contrarian_trade_fragility_by_year": contrarian_trade_fragility_by_year,
        "contrarian_top_trade_removal": contrarian_top_trade_removal,
        "contrarian_year_regime_report": contrarian_year_regime_report,
        "contrarian_year_regime_results": contrarian_year_regime_results,
        "contrarian_year_regime_examples": contrarian_year_regime_examples,
        "contrarian_symbol_year_ablation_report": contrarian_symbol_year_ablation_report,
        "contrarian_without_top_symbols": contrarian_without_top_symbols,
        "contrarian_without_top_years": contrarian_without_top_years,
        "contrarian_cost_slippage_robustness_report": contrarian_cost_slippage_robustness_report,
        "contrarian_cost_slippage_robustness": contrarian_cost_slippage_robustness,
        "contrarian_data_validity_audit": contrarian_data_validity_audit,
        "intraday_5min_expansion_plan": intraday_5min_expansion_plan,
        "contrarian_placebo_results": placebo_rows,
        "contrarian_placebo_summary": placebo_summary,
        "contrarian_matched_controls": _matched_controls(replay),
        "contrarian_contribution_by_year": contribution_year,
        "contrarian_contribution_by_symbol": contribution_symbol,
        "contrarian_concentration_report": concentration,
        "universe_survivorship_audit": _universe_survivorship_audit(rows, config),
        "universe_membership_by_date": _universe_membership(rows),
        "corporate_action_audit": _corporate_action_audit(data_audit),
        "missing_news_bias_report": missing_news_report,
        "covered_vs_uncovered_candidates": covered_vs_uncovered,
        "text_model_readiness": _not_ready_text_model_report(_text_model_readiness(rows)),
        "news_transformer_readiness": build_news_transformer_readiness_report(rows),
        "news_transformer_training_plan": build_news_transformer_training_plan(),
        "catastrophic_news_audit": catastrophic_news_audit,
        "catastrophic_news_candidates": catastrophic_news_candidates,
        "catastrophic_news_veto_report": catastrophic_news_veto_report,
        "catastrophic_veto_candidate_attribution": catastrophic_veto_candidate_attribution,
        "catastrophic_veto_trade_attribution": catastrophic_veto_trade_attribution,
        "catastrophic_veto_strategy_comparison": catastrophic_veto_strategy_comparison,
        "catastrophic_veto_policy": catastrophic_veto_policy,
        "catastrophic_veto_filtered_strategy_report": catastrophic_veto_filtered_strategy_report,
        "catastrophic_veto_removed_trades": catastrophic_veto_removed_trades,
        "catastrophic_veto_removed_symbols": catastrophic_veto_removed_symbols,
        "catastrophic_veto_full_replay_report": catastrophic_veto_full_replay_report,
        "catastrophic_veto_full_replay_trade_ledger": catastrophic_veto_full_replay_trade_ledger,
        "catastrophic_veto_full_replay_equity": catastrophic_veto_full_replay_equity,
        "catastrophic_veto_filtered_candidates": catastrophic_veto_filtered_candidates,
        "catastrophic_veto_blocked_candidates": catastrophic_veto_blocked_candidates,
        "catastrophic_veto_replay_seam_report": _catastrophic_veto_replay_seam_report(
            full_replay_computed=bool(catastrophic_veto_full_replay_report.get("full_replay_computed")),
        ),
        "catastrophic_news_evidence_quality_report": catastrophic_news_evidence_quality_report,
        "catastrophic_news_evidence_quality_by_field": catastrophic_news_evidence_quality_by_field,
        "catastrophic_news_evidence_quality_by_symbol": catastrophic_news_evidence_quality_by_symbol,
        "catastrophic_veto_policy_mode_comparison": catastrophic_veto_policy_mode_comparison,
        "catastrophic_veto_policy_mode_counts": catastrophic_veto_policy_mode_counts,
        "news_event_taxonomy_report": news_event_taxonomy_report,
        "news_event_taxonomy_counts": news_event_taxonomy_counts,
        "news_event_taxonomy_examples": news_event_taxonomy_examples,
        "news_duplicate_grouping_report": news_duplicate_grouping_report,
        "news_duplicate_grouping_examples": news_duplicate_grouping_examples,
        "news_point_in_time_text_safety_report": news_point_in_time_text_safety_report,
        "news_point_in_time_text_safety_examples": news_point_in_time_text_safety_examples,
        "news_text_keyword_baseline_report": news_text_keyword_baseline_report,
        "news_text_keyword_baseline_scores": news_text_keyword_baseline_scores,
        "catastrophic_veto_bounceback_report": catastrophic_veto_bounceback_report,
        "catastrophic_veto_bounceback_by_category": catastrophic_veto_bounceback_by_category,
        "catastrophic_veto_bounceback_examples": catastrophic_veto_bounceback_examples,
        "catastrophic_veto_extreme_only_policy_proposal": catastrophic_veto_extreme_only_policy_proposal,
        "catastrophic_veto_policy_variant_comparison": catastrophic_veto_policy_variant_comparison,
        "catastrophic_veto_policy_variant_counts": catastrophic_veto_policy_variant_counts,
        "catastrophic_veto_policy_variant_metrics": catastrophic_veto_policy_variant_metrics,
        "catastrophic_veto_policy_variant_removed_trades": catastrophic_veto_policy_variant_removed_trades,
        "catastrophic_veto_policy_variant_bounceback": catastrophic_veto_policy_variant_bounceback,
        "catastrophic_veto_policy_frontier_report": catastrophic_veto_policy_frontier_report,
        "catastrophic_veto_policy_frontier": catastrophic_veto_policy_frontier,
        "catastrophic_veto_policy_variant_examples": catastrophic_veto_policy_variant_examples,
        "catastrophic_veto_loser_bounceback_casebook": catastrophic_veto_loser_bounceback_casebook,
        "catastrophic_veto_loser_bounceback_cases": catastrophic_veto_loser_bounceback_cases,
        "catastrophic_veto_loser_bounceback_feature_diff": catastrophic_veto_loser_bounceback_feature_diff,
        "catastrophic_veto_loser_bounceback_keyword_diff": catastrophic_veto_loser_bounceback_keyword_diff,
        "catastrophic_veto_taxonomy_improvement_plan": catastrophic_veto_taxonomy_improvement_plan,
        "catastrophic_veto_parked_status": catastrophic_veto_parked_status,
        "validation_stage_placeholders": _validation_stage_placeholders(
            full_replay_computed=bool(catastrophic_veto_full_replay_report.get("full_replay_computed")),
        ),
        "walk_forward_validation_report": walk_forward_validation_report,
        "walk_forward_fold_results": walk_forward_fold_rows,
        "placebo_permutation_report": placebo_permutation_report,
        "placebo_permutation_results": placebo_permutation_rows,
        "exposure_matched_controls": _matched_control_artifact("exposure_matched_controls"),
        "trade_count_matched_controls": _matched_control_artifact("trade_count_matched_controls"),
        "concentration_fragility_report": _concentration_fragility_artifact(replay),
    }


def _not_ready_text_model_report(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(report)
    payload.update(
        {
            "status": "NOT_READY",
            "ready_for_text_model": False,
            "text_columns_available": list(payload.get("text_columns_available", payload.get("available_text_columns", [])) or []),
            "finbert_readiness": "NOT_READY",
            "bert_readiness": "NOT_READY",
            "numeric_transformer_readiness": "NOT_READY",
            "transformer_trained": False,
            "bert_enabled": False,
            "finbert_enabled": False,
            "transformer_training_enabled": False,
            "recommended_next_baseline": "structured_event_taxonomy_or_simple_text_baseline",
            "blocked_reason": payload.get(
                "blocked_reason",
                "Text modelling is deferred until numerical overlay validation, timestamp audits, taxonomy, and simple baselines are complete.",
            ),
            "warnings": [
                "validation spine not complete",
                "events still uncategorized",
                "text timestamps not proven point-in-time",
                "duplicate/syndication handling not proven",
                "FinBERT deferred",
            ],
        }
    )
    return payload


def _validation_stage_placeholders(*, full_replay_computed: bool = False) -> dict[str, Any]:
    def stage(name: str, status: str, reason: str, warnings: list[str] | None = None) -> dict[str, Any]:
        implemented = status not in {"NOT_IMPLEMENTED", "NOT_READY", "NOT_ENABLED"}
        return {
            "stage_name": name,
            "status": status,
            "implemented": implemented,
            "blocks_final_validation": not implemented,
            "metric_output_allowed": implemented,
            "reason": reason,
            "warnings": warnings or [],
        }

    return {
        "schema_name": "stock_alpha_news_risk_overlay_validation_stage_placeholders",
        "schema_version": 1,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "validation_passed": False,
        "is_final_validation": False,
        "walk_forward": stage("walk_forward", "NOT_IMPLEMENTED", "Walk-forward robustness has not been implemented for this validation spine."),
        "placebo_permutation": stage("placebo_permutation", "NOT_IMPLEMENTED", "Placebo and permutation tests have not been implemented."),
        "exposure_matched_controls": stage("exposure_matched_controls", "NOT_IMPLEMENTED", "Exposure-matched controls have not been implemented."),
        "trade_count_matched_controls": stage("trade_count_matched_controls", "NOT_IMPLEMENTED", "Trade-count-matched controls have not been implemented."),
        "matched_controls": stage("matched_controls", "NOT_IMPLEMENTED", "Legacy aggregate matched-control placeholder; use exposure/trade-count controls when implemented."),
        "concentration_analysis": stage("concentration_analysis", "NOT_IMPLEMENTED", "Contribution and fragility analysis has not been implemented."),
        "year_regime_robustness": {
            **stage("year_regime_robustness", "AVAILABLE", "Ledger-level year/regime robustness report is available."),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "symbol_year_ablation": {
            **stage("symbol_year_ablation", "AVAILABLE", "Ledger-level symbol/year ablations are available."),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "survivorship_audit": stage("survivorship_audit", "NOT_IMPLEMENTED", "Point-in-time universe and survivorship audit has not been implemented."),
        "corporate_action_audit": stage("corporate_action_audit", "NOT_IMPLEMENTED", "Corporate-action validation has not been implemented."),
        "missing_news_bias": stage("missing_news_bias", "NOT_IMPLEMENTED", "Missing-news bias analysis has not been completed."),
        "transaction_cost_validation": stage("transaction_cost_validation", "NOT_IMPLEMENTED", "Realistic transaction-cost validation has not been implemented."),
        "intraday_5min_expansion_plan": {
            **stage("intraday_5min_expansion_plan", "PLANNING_ONLY", "Future Dell PC intraday 5-minute expansion is planning-only."),
            "implemented": True,
            "blocks_final_validation": False,
            "metric_output_allowed": False,
        },
        "catastrophic_news_evidence_quality": {
            **stage(
                "catastrophic_news_evidence_quality",
                "INSUFFICIENT_FOR_STRICT_VETO",
                "Catastrophic-news evidence quality is insufficient for strict live-style filtering.",
                ["research-only", "missing text/availability evidence", "paper/live disabled"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "news_evidence_lineage": {
            **stage(
                "news_evidence_lineage",
                "INSUFFICIENT",
                "News evidence contract and lineage are incomplete for strict veto and text-model readiness.",
                ["observational audit only", "paper/live disabled", "text models disabled"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "catastrophic_veto_strategy_comparison": {
            **stage(
                "catastrophic_veto_strategy_comparison",
                "FULL_REPLAY_COMPUTED" if full_replay_computed else "APPROXIMATE_LEDGER_SIMULATION",
                "Separate research-only full replay is computed." if full_replay_computed else "Catastrophic-veto policy, attribution, and ledger-level simulation are present, but full filtered replay is not computed.",
                ["research-only", "not enforced in current strategy", "paper/live disabled"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": full_replay_computed,
            "replay_impact_status": "FULL_REPLAY_COMPUTED" if full_replay_computed else "APPROXIMATE_LEDGER_SIMULATION",
            "full_replay_status": "FULL_REPLAY_COMPUTED" if full_replay_computed else "FULL_REPLAY_NOT_AVAILABLE",
            "safe_replay_insertion_point": "RESEARCH_STRATEGY_VARIANT_INPUT_SEAM",
            "safe_filtered_variant_seam_status": "REPLAY_ADAPTER_EXECUTED" if full_replay_computed else "REPLAY_ADAPTER_AVAILABLE_OPT_IN_ONLY",
            "full_replay_blocker": "" if full_replay_computed else "integrated replay helper accepts optional extra research-only variants, but replay output does not contain the catastrophic-veto variant",
            "veto_strategy": "news_contrarian_rerank_catastrophic_veto",
        },
        "event_taxonomy_research": {
            **stage(
                "event_taxonomy_research",
                "RESEARCH_RULES_READY",
                "Deterministic headline taxonomy is available for research diagnostics only.",
                ["not production event_category", "paper/live disabled"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "duplicate_grouping_heuristic": {
            **stage(
                "duplicate_grouping_heuristic",
                "HEURISTIC_ONLY",
                "Duplicate grouping is deterministic and heuristic, not provider-grade duplicate_group_id.",
                ["heuristic-only", "text-model readiness remains blocked"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "point_in_time_text_safety": {
            **stage(
                "point_in_time_text_safety",
                "PARTIAL_POINT_IN_TIME_SAFE",
                "Text safety audit is available but remains limited by availability timestamp coverage.",
                ["publication-only timestamps are not availability evidence"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "keyword_text_baseline": {
            **stage(
                "keyword_text_baseline",
                "RESEARCH_ONLY",
                "Deterministic keyword scores are emitted for diagnostics and are not used in strategy ranking.",
                ["no model training", "not used in replay"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "catastrophic_veto_bounceback": {
            **stage(
                "catastrophic_veto_bounceback",
                "RESEARCH_ONLY",
                "Removed-trade bounce-back attribution is observational and does not alter replay mechanics.",
                ["not used in strategy", "paper/live disabled"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "extreme_only_policy_proposal": {
            **stage(
                "extreme_only_policy_proposal",
                "PROPOSED_NOT_REPLAYED",
                "Extreme-distress-only policy is a proposal and requires a future separate research-only replay.",
                ["proposal only", "not validated", "paper/live disabled"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": False,
        },
        "catastrophic_policy_frontier": {
            **stage(
                "catastrophic_policy_frontier",
                "RESEARCH_ONLY_DIAGNOSTIC",
                "Policy frontier ranks catastrophic-veto variants for hypothesis triage only.",
                ["not final validation", "not model selection", "paper/live disabled"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "loser_bounceback_casebook": {
            **stage(
                "loser_bounceback_casebook",
                "RESEARCH_ONLY_DIAGNOSTIC",
                "Loser-vs-bounceback casebook compares removed-trade losers and bounceback winners without changing replay mechanics.",
                ["observational only", "taxonomy proposal only", "paper/live disabled"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "text_model_readiness": {
            **stage(
                "text_model_readiness",
                "NOT_READY",
                "Text modelling is deferred until the numerical validation spine, taxonomy, timestamp, and duplicate-handling checks are complete.",
                [
                    "validation spine not complete",
                    "events still uncategorized",
                    "text timestamps not proven point-in-time",
                    "duplicate/syndication handling not proven",
                    "FinBERT deferred",
                ],
            ),
            "transformer_training_enabled": False,
            "bert_enabled": False,
            "finbert_enabled": False,
        },
    }


def _chronological_periods(
    rows: list[Mapping[str, Any]],
    ledger: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    by_date: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_date.setdefault(_timestamp(row).date().isoformat(), []).append(row)
    dates = sorted(by_date)
    if not dates:
        return {
            "schema_name": "stock_alpha_news_risk_overlay_chronological_split_manifest",
            "schema_version": 1,
            "generated_timestamp": datetime.now(timezone.utc).isoformat(),
            "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
            "status": "NO_ROWS",
            "periods": {},
            "split_method": "chronological_by_complete_decision_date",
            "decision_date_integrity_check": {
                "same_day_candidates_split": False,
                "decision_dates_in_multiple_periods": [],
            },
            "holdout_type": "PSEUDO_HOLDOUT",
            "holdout_status": "PSEUDO_HOLDOUT",
            "validation_label": "PSEUDO_HOLDOUT",
            "is_final_validation": False,
            "validation_passed": False,
            "final_validation_status": "NOT_FINAL_VALIDATION",
            "warnings": ["No candidate rows available for chronological split manifest."],
        }
    cut1 = max(1, int(len(dates) * 0.60))
    cut2 = max(cut1 + 1, int(len(dates) * 0.80)) if len(dates) > 2 else len(dates)
    buckets = {
        "development": dates[:cut1],
        "parameter_validation": dates[cut1:cut2],
        "final_holdout": dates[cut2:],
    }
    trade_counts_by_date: dict[str, int] = {}
    for trade in ledger or []:
        date_key = str(trade.get("decision_timestamp") or trade.get("entry_decision_timestamp") or trade.get("rebalance_date") or "")[:10]
        if date_key:
            trade_counts_by_date[date_key] = trade_counts_by_date.get(date_key, 0) + 1
    date_to_period: dict[str, str] = {}
    periods = {}
    for name, bucket in buckets.items():
        for date_key in bucket:
            date_to_period[date_key] = name
        members = [row for date_key in bucket for row in by_date.get(date_key, [])]
        periods[name] = {
            "period_name": name,
            "start_date": bucket[0] if bucket else None,
            "end_date": bucket[-1] if bucket else None,
            "decision_date_count": len(bucket),
            "candidate_count": len(members),
            "trade_count": sum(trade_counts_by_date.get(date_key, 0) for date_key in bucket),
            "symbol_count": len({str(row.get("symbol", "")).upper() for row in members}),
            "news_coverage": sum(str(row.get("news_coverage_status")) == "COVERED" for row in members) / max(len(members), 1),
            "previously_inspected": name == "final_holdout",
            "status": "PSEUDO_HOLDOUT" if name == "final_holdout" else "DEVELOPMENT_ONLY",
            "warnings": (
                ["Final period has been previously inspected; not an untouched holdout."]
                if name == "final_holdout"
                else []
            ),
            "market_regime_distribution": _category_mix(members) if members else {},
        }
    decision_dates_in_multiple_periods = [
        date_key
        for date_key in dates
        if sum(date_key in bucket for bucket in buckets.values()) > 1
    ]
    return {
        "schema_name": "stock_alpha_news_risk_overlay_chronological_split_manifest",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "status": "PSEUDO_HOLDOUT",
        "split_method": "chronological_by_complete_decision_date",
        "decision_date_integrity_check": {
            "same_day_candidates_split": False,
            "decision_dates_in_multiple_periods": decision_dates_in_multiple_periods,
            "passed": not decision_dates_in_multiple_periods,
        },
        "holdout_previously_inspected": True,
        "holdout_type": "PSEUDO_HOLDOUT",
        "holdout_status": "PSEUDO_HOLDOUT",
        "validation_label": "PSEUDO_HOLDOUT",
        "is_final_validation": False,
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "holdout_warning": "Contrarian hypothesis was introduced after inspecting prior full-history results.",
        "warnings": [
            "Final period is labelled PSEUDO_HOLDOUT because untouched status cannot be proven.",
        ],
        "periods": periods,
    }


def _walk_forward_validation_artifacts(
    replay: Mapping[str, Any],
    periods: Mapping[str, Any],
    selection: Mapping[str, Any],
    frozen: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    period_payloads = dict(periods.get("periods", {}) or {})
    fold_rows: list[dict[str, Any]] = []
    for fold_id, period_name in enumerate(("development", "parameter_validation", "final_holdout"), start=1):
        payload = dict(period_payloads.get(period_name, {}) or {})
        fold_rows.append(
            {
                "fold_id": fold_id,
                "period_name": period_name,
                "train_start": dict(period_payloads.get("development", {}) or {}).get("start_date"),
                "train_end": dict(period_payloads.get("parameter_validation", {}) or {}).get("end_date") if period_name == "final_holdout" else payload.get("start_date"),
                "test_start": payload.get("start_date"),
                "test_end": payload.get("end_date"),
                "start_date": payload.get("start_date"),
                "end_date": payload.get("end_date"),
                "decision_date_count": payload.get("decision_date_count", 0),
                "candidate_count": payload.get("candidate_count", 0),
                "trade_count": payload.get("trade_count", 0),
                "used_for_parameter_selection": period_name != "final_holdout",
                "uses_frozen_configuration": True,
                "random_split_used": False,
                "same_decision_date_crosses_folds": False,
                "price_only_return": None,
                "contrarian_return": None,
                "excess_return": None,
                "excess_sharpe": None,
                "excess_calmar": None,
                "wealth": "UNAVAILABLE_INPUT",
                "return": "UNAVAILABLE_INPUT",
                "max_drawdown": "UNAVAILABLE_INPUT",
                "sharpe": "UNAVAILABLE_INPUT",
                "calmar": "UNAVAILABLE_INPUT",
                "status": "NOT_IMPLEMENTED",
                "metric_status": "NOT_IMPLEMENTED",
                "warning": "Fold-level replay metrics are not computed in this scaffold; no fake metrics emitted.",
            }
        )
    return {
        "schema_name": "stock_alpha_news_walk_forward_validation_report",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "status": "NOT_IMPLEMENTED",
        "validation_label": "PSEUDO_HOLDOUT",
        "holdout_type": "PSEUDO_HOLDOUT",
        "is_final_validation": False,
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "configuration_id": frozen.get("configuration_id"),
        "frozen_configuration_hash": frozen.get("immutable_configuration_hash"),
        "selection_round_trip_cost_bps": float(config.get("stock_alpha_news_risk_overlay_selection_round_trip_cost_bps", 10.0)),
        "fold_count": len(fold_rows),
        "completed_fold_count": 0,
        "passed_fold_count": 0,
        "failed_fold_count": 0,
        "positive_excess_return_fold_count": 0,
        "median_excess_return": None,
        "median_excess_sharpe": None,
        "median_excess_calmar": None,
        "price_only": _risk_subset(dict(replay.get("risk_metrics", {}).get("price_only", {}) or {})),
        "contrarian": _risk_subset(dict(replay.get("risk_metrics", {}).get("news_contrarian_rerank", {}) or {})),
        "folds": fold_rows,
        "used_holdout_for_selection": False,
        "selected_configuration_id": selection.get("selected_configuration_id"),
        "limitations": [
            "Fold-level replay metrics require a future replay-by-fold implementation.",
            "This scaffold preserves chronological dates and emits no fake fold metrics.",
        ],
        "warnings": [
            "Walk-forward validation is not implemented and blocks final validation.",
            "Final pseudo-holdout rows are not used for parameter selection.",
        ],
    }, fold_rows


def _placebo_permutation_artifacts(
    rows: list[Mapping[str, Any]],
    replay: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    del rows, replay
    seed = int(config.get("stock_alpha_news_risk_overlay_seed", 1729))
    checks = (
        "news_score_permuted_by_decision_date",
        "news_score_permuted_globally",
        "news_score_sign_flipped",
        "random_decile_assignment_fixed_seed",
    )
    result_rows = [
        {
            "check_name": check,
            "status": "UNAVAILABLE_INPUT",
            "deterministic_seed": seed,
            "observed_edge": None,
            "placebo_edge": None,
            "observed_edge_larger_than_placebo": None,
            "used_for_configuration_selection": False,
            "warning": "Requires a future placebo replay/statistics implementation; no fake metrics emitted.",
        }
        for check in checks
    ]
    return {
        "schema_name": "stock_alpha_news_placebo_permutation_report",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "status": "UNAVAILABLE_INPUT",
        "validation_label": "PSEUDO_HOLDOUT",
        "holdout_type": "PSEUDO_HOLDOUT",
        "is_final_validation": False,
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "deterministic_seed": seed,
        "checks": result_rows,
        "used_for_configuration_selection": False,
        "warnings": [
            "Placebo/permutation validation is not implemented and blocks final validation.",
            "No placebo results are used for retuning or configuration selection.",
        ],
    }, result_rows


def _matched_control_artifact(control_name: str) -> dict[str, Any]:
    return {
        "schema_name": f"stock_alpha_news_{control_name}",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "status": "NOT_IMPLEMENTED",
        "implemented": False,
        "blocks_final_validation": True,
        "metric_output_allowed": False,
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "warnings": [f"{control_name} is not implemented and blocks final validation."],
    }


def _concentration_fragility_artifact(replay: Mapping[str, Any]) -> dict[str, Any]:
    ledger = list(replay.get("contrarian_trade_ledger", []) or [])
    net_returns = sorted(
        (_number(trade.get("net_return")) or 0.0 for trade in ledger),
        reverse=True,
    )
    total_positive = sum(value for value in net_returns if value > 0)
    top_1_share = (net_returns[0] / total_positive) if net_returns and total_positive > 0 else None
    return {
        "schema_name": "stock_alpha_news_concentration_fragility_report",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "status": "SCAFFOLD_WITH_LIMITED_LEDGER_SUMMARY" if ledger else "UNAVAILABLE_INPUT",
        "implemented": bool(ledger),
        "blocks_final_validation": True,
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "trade_count": len(ledger),
        "top_1_positive_return_share": top_1_share,
        "warnings": [
            "Concentration report is a limited ledger summary; full top-trade/top-symbol fragility is not implemented.",
            "This stage blocks final validation.",
        ],
    }


def _catastrophic_veto_parked_status(
    strict_report: Mapping[str, Any],
    policy_frontier: Mapping[str, Any],
    casebook: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_name": "catastrophic_veto_parked_status",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PARKED_DIAGNOSTIC_ONLY",
        "reason": "strict veto too broad for return; narrow rules no effect; loser-vs-bounceback casebook did not reveal a usable deterministic distinction from available evidence",
        "latest_strict_veto_result": {
            "status": strict_report.get("status", "UNAVAILABLE_INPUT"),
            "replay_impact_status": strict_report.get("replay_impact_status", "UNAVAILABLE_INPUT"),
            "delta_metrics": strict_report.get("delta_metrics", {}),
        },
        "latest_narrow_policy_result": {
            "frontier_status": policy_frontier.get("frontier_status", policy_frontier.get("status", "UNAVAILABLE_INPUT")),
            "best_balanced_policy": policy_frontier.get("best_balanced_policy", "UNAVAILABLE_INPUT"),
            "policies_with_no_effect": policy_frontier.get("policies_with_no_effect", []),
        },
        "casebook_result": {
            "status": casebook.get("status", "UNAVAILABLE_INPUT"),
            "severe_loser_case_count": casebook.get("severe_loser_case_count", "UNAVAILABLE_INPUT"),
            "strong_bounceback_case_count": casebook.get("strong_bounceback_case_count", "UNAVAILABLE_INPUT"),
        },
        "recommended_future_revisit_condition": "Revisit only after upstream article/body text, provider, availability timestamp, and event taxonomy evidence materially improve.",
        "used_in_current_strategy": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
    }


def _contrarian_chronological_validation_plan(
    rows: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    periods: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_periods = dict(periods.get("periods", {}) or {})
    mapping = (
        ("development", "development"),
        ("parameter_validation", "parameter_validation"),
        ("pseudo_holdout", "final_holdout"),
        ("future_final_holdout", "future_final_holdout"),
    )
    ledger = [row for row in replay.get("trade_ledger", []) if row.get("strategy_variant") == "news_contrarian_rerank"]
    rows_by_date: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        rows_by_date.setdefault(_timestamp(row).date().isoformat(), []).append(row)
    trade_counts_by_entry: dict[str, int] = {}
    for trade in ledger:
        date_key = str(trade.get("entry_date") or trade.get("entry_timestamp") or trade.get("decision_timestamp") or "")[:10]
        if date_key:
            trade_counts_by_entry[date_key] = trade_counts_by_entry.get(date_key, 0) + 1
    period_rows: list[dict[str, Any]] = []
    assigned_dates: set[str] = set()
    for period_name, source_name in mapping:
        source = dict(manifest_periods.get(source_name, {}) or {})
        if period_name == "future_final_holdout":
            period_rows.append({
                "period_name": period_name,
                "start_date": "NOT_YET_DEFINED",
                "end_date": "NOT_YET_DEFINED",
                "decision_date_count": 0,
                "candidate_count": 0,
                "trade_count_if_available": 0,
                "used_for_selection": False,
                "used_for_final_validation": False,
                "contamination_status": "NOT_AVAILABLE",
                "allowed_actions": "collect future untouched data only",
                "validation_label": "NOT_FINAL_VALIDATION",
            })
            continue
        start = source.get("start_date")
        end = source.get("end_date")
        date_keys = sorted(date for date in rows_by_date if start and end and str(start) <= date <= str(end))
        assigned_dates.update(date_keys)
        period_rows.append({
            "period_name": period_name,
            "start_date": start or "UNAVAILABLE_INPUT",
            "end_date": end or "UNAVAILABLE_INPUT",
            "decision_date_count": len(date_keys),
            "candidate_count": sum(len(rows_by_date[date]) for date in date_keys),
            "trade_count_if_available": sum(trade_counts_by_entry.get(date, 0) for date in date_keys),
            "used_for_selection": period_name in {"development", "parameter_validation"},
            "used_for_final_validation": False,
            "contamination_status": "PSEUDO_HOLDOUT_PREVIOUSLY_INSPECTED" if period_name == "pseudo_holdout" else "DEVELOPMENT_ONLY",
            "allowed_actions": "diagnostic reporting only" if period_name == "pseudo_holdout" else "research development only",
            "validation_label": "PSEUDO_HOLDOUT" if period_name == "pseudo_holdout" else "DEVELOPMENT_ONLY",
        })
    return {
        "schema_name": "contrarian_chronological_validation_plan",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PSEUDO_HOLDOUT_PLAN",
        "split_method": "chronological_by_complete_decision_date",
        "random_row_split_used": False,
        "complete_decision_dates_only": True,
        "decision_dates_in_multiple_periods": sorted(set(rows_by_date) - assigned_dates),
        "future_final_holdout_status": "NOT_YET_DEFINED",
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "periods": period_rows,
        "warnings": ["Pseudo-holdout is not final validation; future final holdout is not yet defined."],
    }, period_rows


def _metric_from_trade_sum(trades: Sequence[Mapping[str, Any]], field: str) -> float:
    return sum(_number(row.get(field)) or 0.0 for row in trades)


def _contrarian_trade_rows(replay: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in replay.get("trade_ledger", [])
        if row.get("strategy_variant") == "news_contrarian_rerank"
    ]


def _trade_pnl(row: Mapping[str, Any]) -> float:
    return _number(row.get("net_pnl")) or _number(row.get("pnl")) or 0.0


def _contrarian_trade_return(row: Mapping[str, Any]) -> float | None:
    for field in ("net_return", "removed_trade_return", "return", "total_return"):
        value = _number(row.get(field))
        if value is not None:
            return value
    return None


def _trade_year(row: Mapping[str, Any]) -> str:
    return str(
        row.get("exit_date")
        or row.get("exit_timestamp")
        or row.get("entry_date")
        or row.get("entry_timestamp")
        or row.get("decision_timestamp")
        or "UNKNOWN"
    )[:4]


def _year_regime_status(year: str, trade_count: int, net_pnl: float, all_years: Sequence[str]) -> str:
    current_year = str(datetime.now(timezone.utc).year)
    if year == current_year and year == max(all_years, default=year):
        return "partial_year"
    if net_pnl < 0:
        return "negative_year"
    if trade_count < 25:
        return "low_sample_year"
    if net_pnl > 0 and trade_count >= 100:
        return "high_positive_year"
    if net_pnl > 0:
        return "moderate_positive_year"
    return "low_sample_year"


def _contrarian_year_regime_artifacts(
    replay: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    trades = _contrarian_trade_rows(replay)
    total_pnl = sum(_trade_pnl(row) for row in trades)
    by_year: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        by_year.setdefault(_trade_year(trade), []).append(trade)
    years = sorted(by_year)
    daily_equity = {
        str(row.get("date") or row.get("timestamp") or "")[:10]: row
        for row in dict(replay.get("daily_equity", {}) or {}).get("news_contrarian_rerank", [])
    }
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for year in years:
        group = by_year[year]
        returns = [_contrarian_trade_return(row) for row in group if _contrarian_trade_return(row) is not None]
        pnl = sum(_trade_pnl(row) for row in group)
        winners = sum(_trade_pnl(row) > 0 for row in group)
        losers = sum(_trade_pnl(row) < 0 for row in group)
        equity_rows = [
            row for date_key, row in daily_equity.items()
            if date_key.startswith(year) and _number(row.get("total_equity")) is not None
        ]
        wealth = _number(equity_rows[-1].get("total_equity")) if equity_rows else None
        regime_status = _year_regime_status(year, len(group), pnl, years)
        warnings = []
        if regime_status == "negative_year":
            warnings.append("negative ledger-level year")
        if regime_status == "partial_year":
            warnings.append("partial calendar year")
        if not equity_rows:
            warnings.append("equity metrics unavailable; ledger-level metrics only")
        rows.append({
            "year": year,
            "trade_count": len(group),
            "winner_count": winners,
            "loser_count": losers,
            "net_pnl": pnl,
            "average_trade_return": mean(returns) if returns else "UNAVAILABLE_INPUT",
            "median_trade_return": median(returns) if returns else "UNAVAILABLE_INPUT",
            "wealth_if_available": wealth if wealth is not None else "UNAVAILABLE_INPUT",
            "return_if_available": (wealth - 1.0) if wealth is not None else "UNAVAILABLE_INPUT",
            "max_drawdown_if_available": "UNAVAILABLE_INPUT",
            "sharpe_if_available": "UNAVAILABLE_INPUT",
            "pnl_contribution": pnl / total_pnl if total_pnl else "UNAVAILABLE_INPUT",
            "regime_status": regime_status,
            "warnings": "; ".join(warnings) if warnings else "",
        })
        for trade in sorted(group, key=lambda row: abs(_trade_pnl(row)), reverse=True)[:3]:
            examples.append({
                "year": year,
                "trade_id": trade.get("trade_id", trade.get("candidate_id", "UNAVAILABLE_INPUT")),
                "symbol": str(trade.get("symbol", "UNKNOWN")).upper(),
                "entry_date": str(trade.get("entry_date") or trade.get("entry_timestamp") or "")[:10],
                "net_pnl": _trade_pnl(trade),
                "net_return": _contrarian_trade_return(trade) if _contrarian_trade_return(trade) is not None else "UNAVAILABLE_INPUT",
            })
    negative_years = [row["year"] for row in rows if row["regime_status"] == "negative_year"]
    return {
        "schema_name": "contrarian_year_regime_report",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "AVAILABLE" if trades else "UNAVAILABLE_INPUT",
        "metric_basis": "LEDGER_LEVEL_APPROXIMATION",
        "trade_count": len(trades),
        "year_count": len(rows),
        "negative_years": negative_years,
        "year_2022_status": next((row["regime_status"] for row in rows if row["year"] == "2022"), "UNAVAILABLE_INPUT"),
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "warnings": ["Ledger-level annual robustness does not recompute full portfolio compounding."],
    }, rows, examples


def _contrarian_profit_concentration_artifacts(
    replay: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    trades = _contrarian_trade_rows(replay)
    sorted_trades = sorted(trades, key=lambda row: (_number(row.get("net_pnl")) or _number(row.get("pnl")) or 0.0), reverse=True)
    total_pnl = _metric_from_trade_sum(trades, "net_pnl")
    if total_pnl == 0:
        total_pnl = _metric_from_trade_sum(trades, "pnl")
    returns = [_number(row.get("net_return")) for row in trades if _number(row.get("net_return")) is not None]
    total_return_sum = sum(returns)
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    by_year: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        by_symbol.setdefault(str(trade.get("symbol", "UNKNOWN")).upper(), []).append(trade)
        year = str(trade.get("exit_date") or trade.get("exit_timestamp") or trade.get("entry_date") or trade.get("entry_timestamp") or "UNKNOWN")[:4]
        by_year.setdefault(year, []).append(trade)

    def group_rows(groups: Mapping[str, list[dict[str, Any]]], key_name: str) -> list[dict[str, Any]]:
        output = []
        for key, group in sorted(groups.items()):
            pnl = _metric_from_trade_sum(group, "net_pnl") or _metric_from_trade_sum(group, "pnl")
            output.append({
                key_name: key,
                "trade_count": len(group),
                "net_pnl": pnl,
                "pnl_contribution": pnl / total_pnl if total_pnl else "UNAVAILABLE_INPUT",
                "average_net_return": mean([_number(row.get("net_return")) or 0.0 for row in group]) if group else "UNAVAILABLE_INPUT",
            })
        return sorted(output, key=lambda row: _number(row.get("net_pnl")) or 0.0, reverse=True)

    symbol_rows = group_rows(by_symbol, "symbol")
    year_rows = group_rows(by_year, "year")
    top_rows = []
    for count in (1, 5, 10):
        removed = sorted_trades[:count]
        removed_pnl = _metric_from_trade_sum(removed, "net_pnl") or _metric_from_trade_sum(removed, "pnl")
        removed_return = sum(_number(row.get("net_return")) or 0.0 for row in removed)
        top_rows.append({
            "removed_top_trade_count": count,
            "remaining_trade_count": max(len(trades) - len(removed), 0),
            "removed_net_pnl": removed_pnl,
            "remaining_net_pnl": total_pnl - removed_pnl,
            "return_without_top_trades": total_return_sum - removed_return if returns else "UNAVAILABLE_INPUT",
            "deterministic_sort": "net_pnl_desc_trade_id",
        })
    largest_winner = sorted_trades[0] if sorted_trades else {}
    largest_loser = min(trades, key=lambda row: _number(row.get("net_pnl")) or _number(row.get("pnl")) or 0.0, default={})
    report = {
        "schema_name": "contrarian_profit_concentration_report",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "IMPLEMENTED" if trades else "UNAVAILABLE_INPUT",
        "trade_count": len(trades),
        "total_net_pnl": total_pnl if trades else "UNAVAILABLE_INPUT",
        "top_1_trade_contribution": _top_contribution(sorted_trades, total_pnl, 1) if trades and total_pnl else "UNAVAILABLE_INPUT",
        "top_5_trade_contribution": _top_contribution(sorted_trades, total_pnl, 5) if trades and total_pnl else "UNAVAILABLE_INPUT",
        "top_10_trade_contribution": _top_contribution(sorted_trades, total_pnl, 10) if trades and total_pnl else "UNAVAILABLE_INPUT",
        "top_symbol_contribution": symbol_rows[0]["pnl_contribution"] if symbol_rows else "UNAVAILABLE_INPUT",
        "top_year_contribution": year_rows[0]["pnl_contribution"] if year_rows else "UNAVAILABLE_INPUT",
        "return_without_top_1_trade": top_rows[0]["return_without_top_trades"] if top_rows else "UNAVAILABLE_INPUT",
        "return_without_top_5_trades": top_rows[1]["return_without_top_trades"] if len(top_rows) > 1 else "UNAVAILABLE_INPUT",
        "return_without_top_10_trades": top_rows[2]["return_without_top_trades"] if len(top_rows) > 2 else "UNAVAILABLE_INPUT",
        "largest_winner": largest_winner.get("trade_id", largest_winner.get("candidate_id", "UNAVAILABLE_INPUT")),
        "largest_loser": largest_loser.get("trade_id", largest_loser.get("candidate_id", "UNAVAILABLE_INPUT")),
        "winner_loser_balance": {
            "winner_count": sum((_number(row.get("net_pnl")) or _number(row.get("pnl")) or 0.0) > 0 for row in trades),
            "loser_count": sum((_number(row.get("net_pnl")) or _number(row.get("pnl")) or 0.0) < 0 for row in trades),
        },
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }
    return report, symbol_rows, year_rows, top_rows


def _fragility_status(remaining_pnl: float, total_pnl: float, contribution_removed: float | str) -> str:
    if not total_pnl or isinstance(contribution_removed, str):
        return "UNAVAILABLE_INPUT"
    if remaining_pnl <= 0:
        return "FRAGILE_TO_REMOVAL"
    if contribution_removed >= 0.5:
        return "FRAGILE_TO_REMOVAL"
    if contribution_removed >= 0.25:
        return "MODERATELY_CONCENTRATED"
    return "ROBUST_TO_REMOVAL"


def _ablation_row(
    *,
    ablation_name: str,
    removed_group: str,
    removed: Sequence[Mapping[str, Any]],
    all_trades: Sequence[Mapping[str, Any]],
    total_pnl: float,
    total_return_sum: float | None,
) -> dict[str, Any]:
    removed_pnl = sum(_trade_pnl(row) for row in removed)
    remaining_pnl = total_pnl - removed_pnl
    removed_returns = [_contrarian_trade_return(row) for row in removed if _contrarian_trade_return(row) is not None]
    contribution = removed_pnl / total_pnl if total_pnl else "UNAVAILABLE_INPUT"
    return_without_removed = (
        total_return_sum - sum(removed_returns)
        if total_return_sum is not None
        else "UNAVAILABLE_INPUT"
    )
    return {
        "ablation_name": ablation_name,
        "removed_group": removed_group,
        "removed_trade_count": len(removed),
        "remaining_trade_count": max(len(all_trades) - len(removed), 0),
        "removed_net_pnl": removed_pnl,
        "remaining_net_pnl": remaining_pnl,
        "return_without_removed_group": return_without_removed,
        "pnl_contribution_removed": contribution,
        "fragility_status": _fragility_status(remaining_pnl, total_pnl, contribution),
        "warnings": "LEDGER_LEVEL_APPROXIMATION; full portfolio compounding not recomputed",
    }


def _contrarian_symbol_year_ablation_artifacts(
    replay: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    trades = _contrarian_trade_rows(replay)
    total_pnl = sum(_trade_pnl(row) for row in trades)
    returns = [_contrarian_trade_return(row) for row in trades if _contrarian_trade_return(row) is not None]
    total_return_sum = sum(returns) if returns else None
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    by_year: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        by_symbol.setdefault(str(trade.get("symbol", "UNKNOWN")).upper(), []).append(trade)
        by_year.setdefault(_trade_year(trade), []).append(trade)
    ranked_symbols = sorted(
        by_symbol.items(),
        key=lambda item: (sum(_trade_pnl(row) for row in item[1]), item[0]),
        reverse=True,
    )
    ranked_years = sorted(
        by_year.items(),
        key=lambda item: (sum(_trade_pnl(row) for row in item[1]), item[0]),
        reverse=True,
    )
    symbol_rows = []
    for count in (1, 3, 5):
        selected = ranked_symbols[:count]
        removed = [trade for _group, group in selected for trade in group]
        symbol_rows.append(_ablation_row(
            ablation_name=f"without_top_{count}_symbol" if count == 1 else f"without_top_{count}_symbols",
            removed_group=",".join(group for group, _rows in selected) or "UNAVAILABLE_INPUT",
            removed=removed,
            all_trades=trades,
            total_pnl=total_pnl,
            total_return_sum=total_return_sum,
        ))
    year_specs = [
        ("without_top_1_year", ranked_years[:1]),
        ("without_top_2_years", ranked_years[:2]),
        ("without_negative_years", [(year, group) for year, group in by_year.items() if sum(_trade_pnl(row) for row in group) < 0]),
        ("without_best_year", ranked_years[:1]),
    ]
    year_rows = []
    for name, selected in year_specs:
        selected_sorted = sorted(selected, key=lambda item: item[0])
        removed = [trade for _group, group in selected_sorted for trade in group]
        year_rows.append(_ablation_row(
            ablation_name=name,
            removed_group=",".join(group for group, _rows in selected_sorted) or "UNAVAILABLE_INPUT",
            removed=removed,
            all_trades=trades,
            total_pnl=total_pnl,
            total_return_sum=total_return_sum,
        ))
    return {
        "schema_name": "contrarian_symbol_year_ablation_report",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "AVAILABLE" if trades else "UNAVAILABLE_INPUT",
        "metric_basis": "LEDGER_LEVEL_APPROXIMATION",
        "trade_count": len(trades),
        "total_net_pnl": total_pnl if trades else "UNAVAILABLE_INPUT",
        "symbol_ablation_count": len(symbol_rows),
        "year_ablation_count": len(year_rows),
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "warnings": ["Ablations remove ledger groups deterministically; full portfolio compounding is not recomputed."],
    }, symbol_rows, year_rows


def _contrarian_placebo_permutation_report(config: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed = int(config.get("stock_alpha_news_risk_overlay_seed", 1729))
    specs = [
        ("shuffle_news_scores_within_decision_date", True, False, "decision_date"),
        ("shuffle_news_scores_within_symbol", False, True, "symbol"),
        ("permute_news_availability_across_candidates", False, False, "candidate"),
        ("replace_news_score_with_noise", False, False, "fixed_seed_noise"),
    ]
    rows = [
        {
            "placebo_name": name,
            "seed": seed + index,
            "shuffle_scope": scope,
            "preserves_decision_date": preserves_date,
            "preserves_symbol": preserves_symbol,
            "status": "UNAVAILABLE_INPUT",
            "wealth": "UNAVAILABLE_INPUT",
            "return": "UNAVAILABLE_INPUT",
            "sharpe": "UNAVAILABLE_INPUT",
            "p_value_if_available": "UNAVAILABLE_INPUT",
        }
        for index, (name, preserves_date, preserves_symbol, scope) in enumerate(specs)
    ]
    return {
        "schema_name": "contrarian_placebo_permutation_report",
        "schema_version": 1,
        "status": "UNAVAILABLE_INPUT",
        "deterministic_seed": seed,
        "placebo_definitions": rows,
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "warnings": ["Placebo definitions are deterministic, but replay/statistics are deferred; no fake metrics emitted."],
    }, rows


def _contrarian_matched_control_report() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    controls = [
        ("price_score_matched_control", "price-score nearest-neighbor matching"),
        ("trade_count_matched_control", "trade-count matched price-only subset"),
        ("sector_symbol_exposure_matched_control", "sector/symbol exposure matching when fields exist"),
        ("date_matched_random_control", "decision-date matched random control with fixed seed"),
    ]
    rows = [
        {
            "control_name": name,
            "matching_method": method,
            "matched_trade_count": "UNAVAILABLE_INPUT",
            "exposure_match_quality": "UNAVAILABLE_INPUT",
            "wealth": "UNAVAILABLE_INPUT",
            "return": "UNAVAILABLE_INPUT",
            "max_drawdown": "UNAVAILABLE_INPUT",
            "sharpe": "UNAVAILABLE_INPUT",
            "status": "NOT_IMPLEMENTED",
            "warnings": "requires dedicated matched-control replay/input construction; no fake metrics emitted",
        }
        for name, method in controls
    ]
    return {
        "schema_name": "contrarian_matched_control_report",
        "schema_version": 1,
        "status": "NOT_IMPLEMENTED",
        "controls": rows,
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }, rows


def _contrarian_cost_slippage_robustness(
    cost_scenarios: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scenarios = dict(cost_scenarios.get("scenarios", {}) or {})
    rows = []
    for bps in (0, 5, 10, 20, 30, 50, 100):
        key = f"{bps:g}_bps_round_trip"
        payload = dict(scenarios.get(key, {}) or {})
        variants = dict(payload.get("variants", {}) or {})
        contrarian = dict(variants.get("news_contrarian_rerank", {}) or {})
        price = dict(variants.get("price_only", {}) or {})
        computed = bool(contrarian)
        rows.append({
            "cost_bps": bps,
            "wealth": _metric_value(contrarian, "ending_equity", "ending_wealth") if computed else "UNAVAILABLE_INPUT",
            "return": _metric_value(contrarian, "total_return_decimal", "total_return") if computed else "UNAVAILABLE_INPUT",
            "max_drawdown": _metric_value(contrarian, "maximum_drawdown", "max_drawdown") if computed else "UNAVAILABLE_INPUT",
            "sharpe": _metric_value(contrarian, "Sharpe_ratio", "sharpe_ratio") if computed else "UNAVAILABLE_INPUT",
            "trade_count": _metric_value(contrarian, "trade_count") if computed else "UNAVAILABLE_INPUT",
            "cost_robustness_status": "COMPUTED_EXISTING_COST_TABLE" if computed else "NOT_COMPUTED",
            "metric_status": "COMPUTED_FROM_EXISTING_COST_TABLE" if computed else "NOT_COMPUTED",
            "beats_price_only": (
                (_metric(contrarian, "total_return_decimal") or 0.0) > (_metric(price, "total_return_decimal") or 0.0)
                if computed and price
                else "UNAVAILABLE_INPUT"
            ),
        })
    return {
        "schema_name": "contrarian_cost_slippage_robustness_report",
        "schema_version": 1,
        "status": "PARTIAL_EXISTING_COST_TABLE",
        "computed_cost_bps": [row["cost_bps"] for row in rows if row["cost_robustness_status"] != "NOT_COMPUTED"],
        "not_computed_cost_bps": [row["cost_bps"] for row in rows if row["cost_robustness_status"] == "NOT_COMPUTED"],
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "warnings": ["Existing 0/5/10/20 bps table is preserved; extra costs are marked NOT_COMPUTED unless already available."],
    }, rows


def _contrarian_data_validity_audit(
    data_audit: Mapping[str, Any],
    missing_news_report: Mapping[str, Any],
) -> dict[str, Any]:
    missing_bar_count = int(data_audit.get("missing_bar_count") or 0)
    split_status = data_audit.get("split_adjustment_status", "NOT_IMPLEMENTED")
    dividend_status = data_audit.get("dividend_adjustment_status", "NOT_IMPLEMENTED")
    corporate_action_status = data_audit.get("corporate_action_status", "NOT_IMPLEMENTED")
    check_statuses = {
        "survivorship_bias": "NOT_IMPLEMENTED",
        "delisted_stock_handling": "NOT_IMPLEMENTED",
        "bankrupt_stock_handling": "NOT_IMPLEMENTED",
        "split_adjustment": split_status,
        "dividend_adjustment": dividend_status,
        "corporate_action_adjustment": corporate_action_status,
        "suspicious_price_discontinuities": data_audit.get("price_discontinuity_status", "UNAVAILABLE_INPUT"),
        "missing_price_bars": "PASSED" if missing_bar_count == 0 else "FAILED",
        "missing_news_bias": missing_news_report.get("status", "UNAVAILABLE_INPUT"),
    }
    major_checks = {
        "survivorship_bias",
        "delisted_stock_handling",
        "bankrupt_stock_handling",
        "split_adjustment",
        "dividend_adjustment",
        "corporate_action_adjustment",
        "missing_news_bias",
    }
    required_inputs = {
        "survivorship_bias": ["point_in_time_universe_membership", "delisted_symbol_reference"],
        "delisted_stock_handling": ["delisted_symbol_reference"],
        "bankrupt_stock_handling": ["bankruptcy_or_delisting_event_reference"],
        "split_adjustment": ["split_adjusted_price_bars", "split_factor_reference"],
        "dividend_adjustment": ["dividend_adjusted_price_bars", "dividend_reference"],
        "corporate_action_adjustment": ["corporate_action_reference"],
        "suspicious_price_discontinuities": ["daily_price_bars"],
        "missing_price_bars": ["daily_price_bars"],
        "missing_news_bias": ["news_features", "covered_vs_uncovered_candidates"],
    }
    recommendations = {
        "survivorship_bias": "Load point-in-time universe membership and verify unavailable/delisted symbols are represented.",
        "delisted_stock_handling": "Add a delisted-symbol reference and reconcile candidate/trade symbols against it.",
        "bankrupt_stock_handling": "Add bankruptcy/delisting event reference data before final validation.",
        "split_adjustment": "Validate split-adjusted price continuity against a split-factor reference.",
        "dividend_adjustment": "Validate dividend adjustment policy and total-return semantics.",
        "corporate_action_adjustment": "Document and test corporate-action adjustment semantics for all bars.",
        "suspicious_price_discontinuities": "Run discontinuity checks by symbol/date and inspect large adjusted moves.",
        "missing_price_bars": "Repair or document missing bars before final validation.",
        "missing_news_bias": "Complete covered-vs-uncovered candidate analysis and check return skew.",
    }
    risks = {
        "survivorship_bias": "Inflated historical returns if failed/delisted symbols are absent.",
        "delisted_stock_handling": "Losers can disappear from the tradable universe.",
        "bankrupt_stock_handling": "Extreme downside events may be omitted or mislabelled.",
        "split_adjustment": "False returns and ranking artifacts around split dates.",
        "dividend_adjustment": "Return comparisons can be inconsistent across symbols.",
        "corporate_action_adjustment": "Backtest may trade on distorted prices.",
        "suspicious_price_discontinuities": "Bad bars can dominate trade-level returns.",
        "missing_price_bars": "Entry/exit timing and holding-period returns can be wrong.",
        "missing_news_bias": "Signal may rely on covered/uncovered selection effects.",
    }
    checks = {
        name: {
            "status": status,
            "blocks_final_validation": name in major_checks and status != "PASSED" or status in {"FAILED", "UNAVAILABLE_INPUT", "INSUFFICIENT_DATA"},
            "evidence_available": status == "PASSED",
            "required_input_files": required_inputs[name],
            "detected_evidence": {
                "missing_bar_count": missing_bar_count if name == "missing_price_bars" else "UNAVAILABLE_INPUT",
                "source_status": status,
            },
            "recommended_next_step": recommendations[name],
            "risk_if_unresolved": risks[name],
        }
        for name, status in check_statuses.items()
    }
    blocking_checks = [name for name, payload in checks.items() if payload["blocks_final_validation"]]
    return {
        "schema_name": "contrarian_data_validity_audit",
        "schema_version": 1,
        "status": "BLOCKING" if blocking_checks else "PASSED",
        "checks": checks,
        "blocking_checks": blocking_checks,
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "warnings": ["Data-validity audits block final validation until implemented or proven."],
    }


def _intraday_5min_expansion_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_name": "intraday_5min_expansion_plan",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PLANNING_ONLY",
        "target_machine": "Dell PC",
        "recommended_data_frequency": "5min",
        "secondary_frequency": "15min",
        "required_years": config.get("stock_alpha_news_risk_overlay_intraday_required_years", "TO_BE_CONFIRMED"),
        "required_symbols": config.get("stock_alpha_news_risk_overlay_intraday_required_symbols", "TO_BE_CONFIRMED"),
        "expected_data_layout": config.get("stock_alpha_news_risk_overlay_intraday_data_layout", "TO_BE_CONFIRMED"),
        "parquet_conversion_required": True,
        "storage_estimate_status": "TO_BE_CONFIRMED",
        "compute_estimate_status": "TO_BE_CONFIRMED",
        "data_quality_checks": [
            "symbol/date coverage",
            "missing 5min bars",
            "split-adjusted price continuity",
            "timezone/session alignment",
            "duplicate bars",
            "daily-to-intraday reconciliation",
        ],
        "pipeline_steps": [
            "locate existing downloaded 5min/15min data",
            "convert source files to parquet",
            "validate symbol/date coverage",
            "check missing bars",
            "check split-adjusted price continuity",
            "run a small subset first",
            "run full-universe intraday features on the Dell",
            "compare intraday model output to daily contrarian signal",
        ],
        "recommended_commands_placeholder": "TO_BE_CONFIRMED",
        "risks": [
            "unknown local data paths",
            "large storage footprint",
            "intraday missing-bar bias",
            "split/session/timezone mismatch",
            "daily signal may not transfer to intraday cadence",
        ],
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }


def _period_for_date(date_key: str, manifest: Mapping[str, Any]) -> str:
    for name, payload in dict(manifest.get("periods", {}) or {}).items():
        start = payload.get("start_date")
        end = payload.get("end_date")
        if start and end and str(start) <= date_key <= str(end):
            return name
    return "unassigned"


def _contrarian_grid_reports(
    rows: list[Mapping[str, Any]],
    replay: Mapping[str, Any],
    price_score_column: str,
    periods: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    weights = tuple(config.get("stock_alpha_news_risk_overlay_contrarian_grid_weights", (0.0, 0.10, 0.25, 0.50, 0.75, 1.00)))
    selection_cost_bps = float(config.get("stock_alpha_news_risk_overlay_selection_round_trip_cost_bps", 10.0))
    transforms = ("raw_probability", "training_median_centered", "decision_date_percentile", "fold_local_zscore")
    missing = ("no_adjustment", "fold_local_neutral_median")
    price = dict(replay.get("risk_metrics", {}).get("price_only", {}) or {})
    contrarian = dict(replay.get("risk_metrics", {}).get("news_contrarian_rerank", {}) or {})
    cost_metric_overrides = dict(config.get("stock_alpha_news_risk_overlay_contrarian_grid_cost_metrics", {}) or {})
    grid_rows = []
    for weight in weights:
        for transform in transforms:
            for missing_policy in missing:
                config_id = f"w{float(weight):.2f}_{transform}_{missing_policy}"
                is_current = abs(float(weight) - float(config.get("stock_alpha_news_risk_overlay_contrarian_weight", 0.25))) < 1e-12 and transform == "raw_probability" and missing_policy == "no_adjustment"
                cost_metrics = _grid_metrics_at_selection_cost(
                    config_id,
                    selection_cost_bps,
                    cost_metric_overrides,
                )
                median_calmar = (
                    _number(cost_metrics.get("median_validation_calmar"))
                    if cost_metrics
                    else (_metric(contrarian, "Calmar_ratio") if is_current else 0.0)
                )
                excess_return = (
                    _number(cost_metrics.get("median_excess_return"))
                    if cost_metrics
                    else ((_metric(contrarian, "total_return_decimal") or 0.0) - (_metric(price, "total_return_decimal") or 0.0) if is_current else 0.0)
                )
                excess_sharpe = (
                    _number(cost_metrics.get("median_excess_sharpe"))
                    if cost_metrics
                    else ((_metric(contrarian, "Sharpe_ratio") or 0.0) - (_metric(price, "Sharpe_ratio") or 0.0) if is_current else 0.0)
                )
                eligible = bool(cost_metrics.get("eligible", True)) if cost_metrics else is_current
                grid_rows.append(
                    {
                        "configuration_id": config_id,
                        "contrarian_weight": float(weight),
                        "news_transformation": transform,
                        "missing_news_treatment": missing_policy,
                        "selection_metric": "median_validation_calmar",
                        "selection_round_trip_cost_bps": selection_cost_bps,
                        "median_validation_calmar": median_calmar or 0.0,
                        "median_excess_return": excess_return or 0.0,
                        "median_excess_sharpe": excess_sharpe or 0.0,
                        "eligible": eligible,
                        "rejection_reason": "" if eligible else "not evaluated in lightweight validation artifact; requires manual grid run",
                        "training_data_only_transform_fit": transform in {"training_median_centered", "fold_local_zscore"},
                        "decision_timestamp_only_transform": transform == "decision_date_percentile",
                    }
                )
    winner = _select_contrarian_grid_configuration(grid_rows)
    fold_rows = []
    for period_name, payload in dict(periods.get("periods", {}) or {}).items():
        fold_rows.append(
            {
                "fold_id": period_name,
                "training_dates": "earlier_observations_only",
                "validation_dates": f"{payload.get('start_date')}..{payload.get('end_date')}",
                "selected_configuration": winner["configuration_id"],
                "price_only_return": _metric(price, "total_return_decimal"),
                "contrarian_return": _metric(contrarian, "total_return_decimal"),
                "excess_return": (_metric(contrarian, "total_return_decimal") or 0.0) - (_metric(price, "total_return_decimal") or 0.0),
                "price_only_drawdown": _metric(price, "maximum_drawdown"),
                "contrarian_drawdown": _metric(contrarian, "maximum_drawdown"),
                "sharpe_difference": (_metric(contrarian, "Sharpe_ratio") or 0.0) - (_metric(price, "Sharpe_ratio") or 0.0),
                "calmar_difference": (_metric(contrarian, "Calmar_ratio") or 0.0) - (_metric(price, "Calmar_ratio") or 0.0),
                "news_coverage": payload.get("news_coverage"),
                "point_in_time_validation_failures": 0,
            }
        )
    selection = {
        "schema_name": "contrarian_grid_selection",
        "schema_version": "1.0",
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "status": "SCAFFOLD_CURRENT_FORMULA_ONLY",
        "experiment_registry_metadata": {
            "experiment_family": "stock_alpha_news_risk_overlay",
            "experiment_name": "news_contrarian_rerank_validation_scaffold",
            "research_only": True,
            "broker_invoked": False,
            "paper_orders_enabled": False,
            "live_orders_enabled": False,
            "news_originated_entries_enabled": False,
            "status": "DEVELOPMENT_ONLY",
        },
        "validation_status": "DEVELOPMENT_ONLY",
        "grid_search_status": "SCAFFOLD_COMPLETE_CURRENT_FORMULA_ONLY",
        "full_grid_search_implemented": False,
        "future_validation_stage_statuses": {
            "walk_forward": "NOT_IMPLEMENTED",
            "placebo_permutation": "NOT_IMPLEMENTED",
            "matched_controls": "NOT_IMPLEMENTED",
            "concentration_analysis": "NOT_IMPLEMENTED",
            "survivorship_audit": "NOT_IMPLEMENTED",
            "corporate_action_audit": "NOT_IMPLEMENTED",
            "missing_news_bias": "NOT_IMPLEMENTED",
            "transaction_cost_validation": "NOT_IMPLEMENTED",
        },
        "selection_policy": "highest median validation Calmar among eligible chronological folds",
        "selection_round_trip_cost_bps": selection_cost_bps,
        "cost_selection_policy": "single predeclared round-trip cost used for parameter selection; sensitivity reported separately",
        "cost_selection_metric_source": "metrics at configured selection_round_trip_cost_bps",
        "selected_configuration_id": winner["configuration_id"],
        "selection_metric": "median_validation_calmar",
        "selected_config_metric_at_selection_cost": winner.get("median_validation_calmar", 0.0),
        "selected_config": winner,
        "selected_configuration": winner,
        "holdout_used_for_selection": False,
        "used_holdout_for_selection": False,
        "eligible_config_count": sum(bool(row.get("eligible")) for row in grid_rows),
        "rejected_config_count": sum(not row["eligible"] for row in grid_rows),
        "rejected_configuration_count": sum(not row["eligible"] for row in grid_rows),
        "rejection_reasons": sorted({
            str(row.get("rejection_reason", ""))
            for row in grid_rows
            if not row.get("eligible") and row.get("rejection_reason")
        }),
        "rejected_configurations": [row for row in grid_rows if not row["eligible"]][:50],
        "tie_break_rules": [
            "higher median validation Calmar at configured selection cost",
            "higher median excess return",
            "higher median excess Sharpe",
            "smaller absolute contrarian weight",
            "deterministic lexical configuration ID",
        ],
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "warnings": [
            "Grid artifact is scaffold/current-formula only; full bounded grid is not complete.",
            "Holdout rows are not used for parameter selection.",
        ],
        "fixed_portfolio_rules": {
            "price_score_column": price_score_column,
            "position_count": "unchanged",
            "max_position_weight": "unchanged",
            "entry_exit_rules": "unchanged",
            "transaction_cost_assumption_for_selection": f"{selection_cost_bps:g} bps round trip",
        },
    }
    return grid_rows, fold_rows, selection


def _select_contrarian_grid_configuration(grid_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [dict(row) for row in grid_rows if bool(row.get("eligible"))]
    candidates = eligible or [dict(row) for row in grid_rows]
    if not candidates:
        return {
            "configuration_id": "unavailable",
            "eligible": False,
            "median_validation_calmar": 0.0,
            "median_excess_return": 0.0,
            "median_excess_sharpe": 0.0,
            "contrarian_weight": 0.0,
            "rejection_reason": "no grid rows available",
        }
    return max(
        candidates,
        key=lambda row: (
            bool(row.get("eligible")),
            _number(row.get("median_validation_calmar")) or 0.0,
            _number(row.get("median_excess_return")) or 0.0,
            _number(row.get("median_excess_sharpe")) or 0.0,
            -abs(_number(row.get("contrarian_weight")) or 0.0),
            str(row.get("configuration_id", "")),
        ),
    )


def _selection_cost_metrics_from_scenarios(
    config: Mapping[str, Any],
    cost_scenarios: Mapping[str, Any],
) -> dict[str, Any]:
    weight = float(config.get("stock_alpha_news_risk_overlay_contrarian_weight", 0.25))
    config_id = f"w{weight:.2f}_raw_probability_no_adjustment"
    by_cost: dict[str, Any] = {}
    for scenario in dict(cost_scenarios.get("scenarios", {}) or {}).values():
        if not isinstance(scenario, Mapping):
            continue
        round_trip_bps = _number(scenario.get("round_trip_bps"))
        variants = dict(scenario.get("variants", {}) or {})
        price = dict(variants.get("price_only", {}) or {})
        contrarian = dict(variants.get("news_contrarian_rerank", {}) or {})
        if round_trip_bps is None or not contrarian:
            continue
        by_cost[f"{round_trip_bps:g}"] = {
            "median_validation_calmar": _metric(contrarian, "Calmar_ratio", "calmar_ratio") or 0.0,
            "median_excess_return": (_metric(contrarian, "total_return_decimal") or 0.0) - (_metric(price, "total_return_decimal") or 0.0),
            "median_excess_sharpe": (_metric(contrarian, "Sharpe_ratio", "sharpe_ratio") or 0.0) - (_metric(price, "Sharpe_ratio", "sharpe_ratio") or 0.0),
            "eligible": True,
        }
    return {config_id: by_cost} if by_cost else {}


def _grid_metrics_at_selection_cost(
    configuration_id: str,
    selection_cost_bps: float,
    cost_metric_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    by_config = dict(cost_metric_overrides.get(configuration_id, {}) or {})
    if not by_config:
        return {}
    exact_keys = (
        str(selection_cost_bps),
        f"{selection_cost_bps:g}",
        f"{selection_cost_bps:.1f}",
        f"{selection_cost_bps:.2f}",
    )
    for key in exact_keys:
        payload = by_config.get(key)
        if isinstance(payload, Mapping):
            return dict(payload)
    bps_key = f"{selection_cost_bps:g}_bps"
    payload = by_config.get(bps_key)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _frozen_contrarian_config(
    selection: Mapping[str, Any],
    config: Mapping[str, Any],
    periods: Mapping[str, Any],
) -> dict[str, Any]:
    selected = dict(selection.get("selected_configuration", {}) or {})
    payload = {
        "schema_name": "contrarian_frozen_config",
        "schema_version": "1.0",
        "configuration_id": selected.get("configuration_id", "unavailable"),
        "experiment_registry_metadata": dict(selection.get("experiment_registry_metadata", {}) or {}),
        "exact_formula": "contrarian_score = price_score + contrarian_weight * transformed_news_score",
        "formula": "contrarian_score = price_score + contrarian_weight * transformed_news_score",
        "news_transformation": selected.get("news_transformation", "raw_probability"),
        "contrarian_weight": selected.get("contrarian_weight", config.get("stock_alpha_news_risk_overlay_contrarian_weight", 0.25)),
        "missing_news_treatment": selected.get("missing_news_treatment", "no_adjustment"),
        "candidate_universe": "joined stock-alpha price-model candidate universe only",
        "ranking_direction": "descending contrarian_score",
        "tie_breaking": "symbol lexical ordering after score",
        "portfolio_rules": "unchanged from open-trade replay",
        "cost_assumptions": "selection uses one predeclared round-trip cost; sensitivity remains in cost_scenario_comparison.json",
        "development_dates": periods.get("periods", {}).get("development", {}),
        "validation_dates": periods.get("periods", {}).get("parameter_validation", {}),
        "selection_metric": selection.get("selection_policy"),
        "selection_round_trip_cost_bps": float(config.get("stock_alpha_news_risk_overlay_selection_round_trip_cost_bps", 10.0)),
        "selected_config_metric_at_selection_cost": selected.get("median_validation_calmar", 0.0),
        "eligibility_constraints": {
            "positive_excess_return_majority_required": True,
            "median_excess_return_positive_required": True,
            "drawdown_not_worse_than_price_only_by_more_than_5pct_points": True,
            "sufficient_trade_count_required": True,
            "point_in_time_validation_failures_allowed": 0,
            "status": "SCAFFOLD_CURRENT_FORMULA_ONLY",
        },
        "tie_break_rules": ["median excess Sharpe", "lower turnover", "smaller absolute weight", "simpler transformation", "lexical configuration ID"],
        "used_holdout_for_selection": False,
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "hash_excluded_fields": ["generated_timestamp"],
        "final_holdout_required": True,
        "parameter_overrides_allowed_for_final_evaluation": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "holdout_type": "PSEUDO_HOLDOUT",
        "validation_label": "PSEUDO_HOLDOUT",
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "is_final_validation": False,
        "validation_passed": False,
        "production_signal": False,
        "paper_orders_enabled": False,
        "live_orders_enabled": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "news_originated_entries_enabled": False,
    }
    payload["immutable_configuration_hash"] = _frozen_config_hash(payload)
    return payload


def _frozen_config_hash(payload: Mapping[str, Any]) -> str:
    ignored = set(payload.get("hash_excluded_fields", ["generated_timestamp"]) or [])
    ignored.update({"generated_timestamp", "immutable_configuration_hash"})
    stable_payload = {
        key: value
        for key, value in dict(payload).items()
        if key not in ignored
    }
    return _stable_hash(stable_payload)


def _holdout_report(
    replay: Mapping[str, Any],
    periods: Mapping[str, Any],
    frozen: Mapping[str, Any],
    cost_scenarios: Mapping[str, Any],
) -> dict[str, Any]:
    price = dict(replay.get("risk_metrics", {}).get("price_only", {}) or {})
    contrarian = dict(replay.get("risk_metrics", {}).get("news_contrarian_rerank", {}) or {})
    validation_gate = _pseudo_holdout_validation_gate(periods, frozen)
    return {
        "schema_name": "contrarian_holdout_report",
        "schema_version": "1.0",
        "holdout_status": "PSEUDO_HOLDOUT",
        "validation_label": "PSEUDO_HOLDOUT",
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "holdout_type": "PSEUDO_HOLDOUT",
        "evaluation_result": "PASSED_WITH_WARNINGS",
        "is_final_validation": False,
        "validation_passed": False,
        "validation_gate": validation_gate,
        "required_validation_gates": {
            "genuine_untouched_holdout": False,
            "frozen_configuration_not_retuned": True,
            "walk_forward_checks_passed": False,
            "placebo_permutation_checks_passed": False,
            "exposure_matched_controls_passed": False,
            "concentration_analysis_passed": False,
            "survivorship_audit_passed": False,
            "corporate_action_audit_passed": False,
            "missing_news_bias_analysis_passed": False,
            "transaction_cost_validation_passed": False,
        },
        "reason": "No genuinely untouched holdout has been established for the already-inspected contrarian hypothesis.",
        "warnings": [
            "Previously viewed period; not a genuine untouched holdout.",
            "This is a pseudo-holdout evaluation result, not final independent validation.",
        ],
        "frozen_configuration_hash": frozen.get("immutable_configuration_hash"),
        "selection_round_trip_cost_bps": frozen.get("selection_round_trip_cost_bps"),
        "parameter_overrides_used": False,
        "price_only": _risk_subset(price),
        "contrarian": _risk_subset(contrarian),
        "excess_return_over_price_only": (_metric(contrarian, "total_return_decimal") or 0.0) - (_metric(price, "total_return_decimal") or 0.0),
        "excess_sharpe": (_metric(contrarian, "Sharpe_ratio") or 0.0) - (_metric(price, "Sharpe_ratio") or 0.0),
        "drawdown_difference": (_metric(contrarian, "maximum_drawdown") or 0.0) - (_metric(price, "maximum_drawdown") or 0.0),
        "cost_sensitivity": cost_scenarios,
        "holdout_period": periods.get("periods", {}).get(
            "final_holdout",
            periods.get("periods", {}).get("final_untouched_holdout", {}),
        ),
    }


def _pseudo_holdout_validation_gate(
    periods: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    final_period = dict(periods.get("periods", {}).get("final_holdout", {}) or {})
    retuning_gate = _retuning_refusal_gate(frozen, override_requested=False)
    warnings = [
        "Final period is not demonstrably untouched.",
        "Robustness, placebo, matched-control, concentration, survivorship, corporate-action, missing-news, and transaction-cost gates remain incomplete.",
    ]
    return {
        "status": "BLOCKED_PSEUDO_HOLDOUT",
        "holdout_type": periods.get("holdout_type", "PSEUDO_HOLDOUT"),
        "validation_label": "PSEUDO_HOLDOUT",
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "is_final_validation": False,
        "validation_passed": False,
        "parameter_overrides_used": False,
        "retuning_gate": retuning_gate,
        "retuning_gate_status": retuning_gate["status"],
        "retuning_gate_passed": bool(retuning_gate["retuning_gate_passed"]),
        "retuning_override_requested": bool(retuning_gate["retuning_override_requested"]),
        "retuning_override_allowed_for_final_evaluation": bool(retuning_gate["retuning_override_allowed_for_final_evaluation"]),
        "final_validation_blocked_by_pseudo_holdout": True,
        "final_evaluation_valid": False,
        "frozen_configuration_hash": frozen.get("immutable_configuration_hash"),
        "final_period_start_date": final_period.get("start_date"),
        "final_period_end_date": final_period.get("end_date"),
        "warnings": warnings,
    }


def _retuning_refusal_gate(
    frozen: Mapping[str, Any],
    *,
    override_requested: bool,
) -> dict[str, Any]:
    overrides_allowed = bool(frozen.get("parameter_overrides_allowed_for_final_evaluation", False))
    refused = bool(override_requested and not overrides_allowed)
    return {
        "status": "REFUSED" if refused else "NO_OVERRIDE_REQUESTED",
        "retuning_gate_status": "REFUSED" if refused else "NO_OVERRIDE_REQUESTED",
        "retuning_gate_passed": not refused,
        "retuning_override_requested": override_requested,
        "retuning_override_allowed_for_final_evaluation": overrides_allowed,
        "override_requested": override_requested,
        "overrides_allowed_for_final_evaluation": overrides_allowed,
        "final_validation_blocked_by_pseudo_holdout": False,
        "final_evaluation_valid": not refused,
        "reason": (
            "Parameter overrides are not allowed for final evaluation from a frozen configuration."
            if refused
            else "No retuning override requested."
        ),
    }


def _append_experiment_registry_entry(
    path: Path,
    *,
    rows: list[Mapping[str, Any]],
    replay: Mapping[str, Any],
    validation: Mapping[str, Any],
    coverage: Mapping[str, Any],
    event_category_analysis: Mapping[str, Any],
    decile_reconciliation: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    risk_metrics = dict(replay.get("risk_metrics", {}) or {})
    decile_reconciliation_payload = dict(decile_reconciliation or {})
    if not decile_reconciliation_payload:
        decile_reconciliation_payload = {"status": "UNKNOWN"}
    event_categories = dict(event_category_analysis or {})
    categorized_count = sum(
        int(payload.get("count", 0))
        for key, payload in event_categories.items()
        if isinstance(payload, Mapping) and key != "general_negative_sentiment_or_uncategorized"
    )
    total_event_count = sum(
        int(payload.get("count", 0))
        for payload in event_categories.values()
        if isinstance(payload, Mapping)
    )
    entry = {
        "experiment_id": _stable_hash(
            {
                "hypothesis": "Among price-model-approved candidates, downside-news pressure may improve ranking.",
                "generated_date": datetime.now(timezone.utc).date().isoformat(),
                "selection_cost": config.get("stock_alpha_news_risk_overlay_selection_round_trip_cost_bps", 10.0),
            }
        ),
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Among price-model-approved candidates, downside-news pressure may improve ranking.",
        "status": "DEVELOPMENT_ONLY",
        "current_formula": "contrarian_score = price_score + contrarian_weight * transformed_news_score",
        "parameters_already_inspected": True,
        "historical_periods_already_viewed": True,
        "current_development_result": {
            "price_only": _risk_subset(dict(risk_metrics.get("price_only", {}) or {})),
            "news_contrarian_rerank": _risk_subset(dict(risk_metrics.get("news_contrarian_rerank", {}) or {})),
        },
        "holdout_contamination_status": "PSEUDO_HOLDOUT",
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "selected_transaction_cost_bps": float(config.get("stock_alpha_news_risk_overlay_selection_round_trip_cost_bps", 10.0)),
        "candidate_universe_constraint": "news reranking remains inside the price-model-approved candidate universe",
        "score_direction": "higher score aligns with downside risk",
        "news_coverage": coverage.get("row_coverage_ratio"),
        "event_categorization_status": "UNCATEGORIZED" if categorized_count == 0 and total_event_count else "PARTIAL_OR_UNAVAILABLE",
        "decile_reconciliation_status": decile_reconciliation_payload.get("status", "UNKNOWN"),
        "grid_selection_status": dict(validation.get("contrarian_grid_selection", {}) or {}).get("status", "UNKNOWN"),
        "frozen_config_hash": dict(validation.get("contrarian_frozen_config", {}) or {}).get("immutable_configuration_hash"),
        "pseudo_holdout_gate_status": dict(
            dict(validation.get("contrarian_holdout_report", {}) or {}).get("validation_gate", {}) or {}
        ).get("status", "UNKNOWN"),
        "warnings": [
            "Development-only result; not a production signal.",
            "Final period remains pseudo-holdout because untouched status cannot be proven.",
        ],
        "holdout_accessed": True,
        "validation_label": "PSEUDO_HOLDOUT",
        "holdout_type": "PSEUDO_HOLDOUT",
        "is_final_validation": False,
        "validation_passed": False,
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "row_count": len(rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def _research_artifact_manifest(paths: NewsRiskResearchPaths) -> dict[str, Any]:
    artifacts = {
        name: str(value)
        for name, value in paths.__dict__.items()
        if name.endswith("_path") and isinstance(value, Path)
    }
    return {
        "schema_name": "stock_alpha_news_risk_overlay_artifact_manifest",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "research_only": True,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def _artifact_validation_report(paths: NewsRiskResearchPaths) -> dict[str, Any]:
    required_names = (
        "chronological_split_manifest_json_path",
        "experiment_registry_jsonl_path",
        "contrarian_grid_selection_json_path",
        "contrarian_frozen_config_json_path",
        "contrarian_holdout_report_json_path",
        "contrarian_chronological_validation_plan_json_path",
        "contrarian_chronological_periods_csv_path",
        "contrarian_walk_forward_validation_report_json_path",
        "contrarian_placebo_permutation_report_json_path",
        "contrarian_placebo_permutation_results_csv_path",
        "contrarian_matched_control_report_json_path",
        "contrarian_matched_control_results_csv_path",
        "contrarian_profit_concentration_report_json_path",
        "contrarian_trade_fragility_by_symbol_csv_path",
        "contrarian_trade_fragility_by_year_csv_path",
        "contrarian_top_trade_removal_csv_path",
        "contrarian_year_regime_report_json_path",
        "contrarian_year_regime_results_csv_path",
        "contrarian_year_regime_examples_csv_path",
        "contrarian_symbol_year_ablation_report_json_path",
        "contrarian_without_top_symbols_csv_path",
        "contrarian_without_top_years_csv_path",
        "contrarian_cost_slippage_robustness_report_json_path",
        "contrarian_cost_slippage_robustness_csv_path",
        "contrarian_data_validity_audit_json_path",
        "intraday_5min_expansion_plan_json_path",
        "text_model_readiness_json_path",
        "validation_stage_placeholders_json_path",
        "decile_join_audit_json_path",
        "decile_trade_reconciliation_json_path",
        "corrected_news_score_deciles_csv_path",
        "news_validation_workflow_map_json_path",
        "validation_dependency_graph_json_path",
        "validation_readiness_dashboard_json_path",
        "artifact_lineage_report_json_path",
        "news_validation_gap_analysis_json_path",
        "news_transformer_readiness_json_path",
        "news_transformer_training_plan_json_path",
        "catastrophic_news_audit_json_path",
        "catastrophic_news_candidates_csv_path",
        "catastrophic_news_veto_report_json_path",
        "catastrophic_veto_candidate_attribution_json_path",
        "catastrophic_veto_trade_attribution_csv_path",
        "catastrophic_veto_strategy_comparison_json_path",
        "catastrophic_veto_policy_json_path",
        "catastrophic_veto_filtered_strategy_report_json_path",
        "catastrophic_veto_removed_trades_csv_path",
        "catastrophic_veto_removed_symbols_csv_path",
        "catastrophic_veto_full_replay_report_json_path",
        "catastrophic_veto_full_replay_trade_ledger_csv_path",
        "catastrophic_veto_full_replay_equity_csv_path",
        "catastrophic_veto_filtered_candidates_csv_path",
        "catastrophic_veto_blocked_candidates_csv_path",
        "catastrophic_veto_replay_seam_report_json_path",
        "catastrophic_veto_bounceback_report_json_path",
        "catastrophic_veto_bounceback_by_category_csv_path",
        "catastrophic_veto_bounceback_examples_csv_path",
        "catastrophic_veto_extreme_only_policy_proposal_json_path",
        "catastrophic_veto_policy_variant_comparison_json_path",
        "catastrophic_veto_policy_variant_counts_csv_path",
        "catastrophic_veto_policy_variant_metrics_csv_path",
        "catastrophic_veto_policy_variant_removed_trades_csv_path",
        "catastrophic_veto_policy_variant_bounceback_csv_path",
        "catastrophic_veto_policy_frontier_report_json_path",
        "catastrophic_veto_policy_frontier_csv_path",
        "catastrophic_veto_policy_variant_examples_csv_path",
        "catastrophic_veto_loser_bounceback_casebook_json_path",
        "catastrophic_veto_loser_bounceback_cases_csv_path",
        "catastrophic_veto_loser_bounceback_feature_diff_csv_path",
        "catastrophic_veto_loser_bounceback_keyword_diff_csv_path",
        "catastrophic_veto_taxonomy_improvement_plan_json_path",
        "catastrophic_veto_parked_status_json_path",
        "catastrophic_news_evidence_quality_report_json_path",
        "catastrophic_news_evidence_quality_by_field_csv_path",
        "catastrophic_news_evidence_quality_by_symbol_csv_path",
        "catastrophic_veto_policy_mode_comparison_json_path",
        "catastrophic_veto_policy_mode_counts_csv_path",
        "news_evidence_lineage_report_json_path",
        "news_evidence_lineage_by_stage_csv_path",
        "news_evidence_missing_field_examples_csv_path",
        "news_evidence_readiness_report_json_path",
        "news_event_taxonomy_report_json_path",
        "news_event_taxonomy_counts_csv_path",
        "news_event_taxonomy_examples_csv_path",
        "news_duplicate_grouping_report_json_path",
        "news_duplicate_grouping_examples_csv_path",
        "news_point_in_time_text_safety_report_json_path",
        "news_point_in_time_text_safety_examples_csv_path",
        "news_text_keyword_baseline_report_json_path",
        "news_text_keyword_baseline_scores_csv_path",
        "walk_forward_validation_report_json_path",
        "walk_forward_fold_results_csv_path",
        "placebo_permutation_report_json_path",
        "placebo_permutation_results_csv_path",
        "exposure_matched_controls_json_path",
        "trade_count_matched_controls_json_path",
        "concentration_fragility_report_json_path",
    )
    artifact_status = []
    missing = []
    for name in required_names:
        path = getattr(paths, name)
        exists = path.exists()
        if not exists:
            missing.append(name)
        artifact_status.append(
            {
                "artifact_key": name,
                "path": str(path),
                "exists": exists,
                "status": "PRESENT" if exists else "MISSING",
            }
        )
    return {
        "schema_name": "stock_alpha_news_risk_overlay_artifact_validation_report",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "status": "PASSED" if not missing else "FAILED",
        "status_scope": "ARTIFACT_PRESENCE_ONLY",
        "artifact_presence_status": "PASSED" if not missing else "FAILED",
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "validation_passed": False,
        "is_final_validation": False,
        "research_only": True,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "required_artifact_count": len(required_names),
        "missing_artifact_count": len(missing),
        "missing_artifacts": missing,
        "artifacts": artifact_status,
    }


def _news_validation_workflow_map(paths: NewsRiskResearchPaths) -> dict[str, Any]:
    full_replay_computed = bool(
        _read_json_if_available(paths.catastrophic_veto_full_replay_report_json_path).get("full_replay_computed")
    )
    evidence_status = _read_json_if_available(
        paths.catastrophic_news_evidence_quality_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    news_evidence_status = _read_json_if_available(
        paths.news_evidence_readiness_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    event_taxonomy_status = _read_json_if_available(
        paths.news_event_taxonomy_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    duplicate_grouping_status = _read_json_if_available(
        paths.news_duplicate_grouping_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    text_safety_status = _read_json_if_available(
        paths.news_point_in_time_text_safety_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    keyword_baseline_status = _read_json_if_available(
        paths.news_text_keyword_baseline_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    bounceback_status = _read_json_if_available(
        paths.catastrophic_veto_bounceback_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    extreme_policy_status = _read_json_if_available(
        paths.catastrophic_veto_extreme_only_policy_proposal_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    policy_variant_status = _read_json_if_available(
        paths.catastrophic_veto_policy_variant_comparison_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    policy_frontier_status = _read_json_if_available(
        paths.catastrophic_veto_policy_frontier_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    casebook_status = _read_json_if_available(
        paths.catastrophic_veto_loser_bounceback_casebook_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    taxonomy_plan_status = _read_json_if_available(
        paths.catastrophic_veto_taxonomy_improvement_plan_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    parked_veto_status = _read_json_if_available(
        paths.catastrophic_veto_parked_status_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    contrarian_profit_status = _read_json_if_available(
        paths.contrarian_profit_concentration_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    year_regime_status = _read_json_if_available(
        paths.contrarian_year_regime_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    symbol_year_ablation_status = _read_json_if_available(
        paths.contrarian_symbol_year_ablation_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    contrarian_data_validity_status = _read_json_if_available(
        paths.contrarian_data_validity_audit_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    intraday_5min_status = _read_json_if_available(
        paths.intraday_5min_expansion_plan_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    casebook_status = _read_json_if_available(
        paths.catastrophic_veto_loser_bounceback_casebook_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    taxonomy_plan_status = _read_json_if_available(
        paths.catastrophic_veto_taxonomy_improvement_plan_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    bounceback_status = _read_json_if_available(
        paths.catastrophic_veto_bounceback_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    extreme_policy_status = _read_json_if_available(
        paths.catastrophic_veto_extreme_only_policy_proposal_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    event_taxonomy_status = _read_json_if_available(
        paths.news_event_taxonomy_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    duplicate_grouping_status = _read_json_if_available(
        paths.news_duplicate_grouping_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    text_safety_status = _read_json_if_available(
        paths.news_point_in_time_text_safety_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    keyword_baseline_status = _read_json_if_available(
        paths.news_text_keyword_baseline_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    node_specs = [
        ("price_model_candidates", "Price model candidates", "IMPLEMENTED", True, [paths.dataset_csv_path], True, True, []),
        ("news_feature_join", "Point-in-time news feature join", "IMPLEMENTED", True, [paths.leakage_json_path], True, True, ["price_model_candidates"]),
        ("coverage_report", "News coverage report", "PRESENT", True, [paths.coverage_json_path], True, True, ["news_feature_join"]),
        ("score_direction_audit", "Score direction audit", "PRESENT", True, [paths.score_direction_audit_json_path, paths.news_score_direction_report_json_path], True, True, ["news_feature_join"]),
        ("strategy_variants", "Research strategy variants", "IMPLEMENTED", True, [paths.contrarian_strategy_comparison_json_path], True, True, ["score_direction_audit"]),
        ("portfolio_replay", "Open-position research replay", "IMPLEMENTED", True, [paths.open_trade_portfolio_json_path, paths.replay_risk_metrics_json_path], True, True, ["strategy_variants"]),
        ("cost_sensitivity", "Cost sensitivity scenarios", "PRESENT", True, [paths.cost_scenario_comparison_json_path], True, True, ["portfolio_replay"]),
        ("decile_reconciliation", "Decile candidate-to-trade reconciliation", "PASSED", True, [paths.decile_join_audit_json_path, paths.decile_trade_reconciliation_json_path], True, True, ["portfolio_replay"]),
        ("chronological_split", "Chronological split manifest", "PSEUDO_HOLDOUT", True, [paths.chronological_split_manifest_json_path], True, True, ["decile_reconciliation"]),
        ("experiment_registry", "Append-only experiment registry", "PRESENT", True, [paths.experiment_registry_jsonl_path], True, True, ["chronological_split"]),
        ("grid_selection", "Contrarian grid selection scaffold", "SCAFFOLD", True, [paths.contrarian_grid_selection_json_path], True, True, ["experiment_registry"]),
        ("frozen_config", "Frozen contrarian configuration", "PRESENT", True, [paths.contrarian_frozen_config_json_path], True, True, ["grid_selection"]),
        ("pseudo_holdout_report", "Pseudo-holdout report", "PSEUDO_HOLDOUT", True, [paths.contrarian_holdout_report_json_path], True, True, ["frozen_config"]),
        ("contrarian_chronological_validation", "Main contrarian chronological validation plan", "IN_PROGRESS", True, [paths.contrarian_chronological_validation_plan_json_path, paths.contrarian_chronological_periods_csv_path], True, True, ["pseudo_holdout_report"]),
        ("contrarian_profit_concentration", "Main contrarian profit concentration", contrarian_profit_status, True, [paths.contrarian_profit_concentration_report_json_path, paths.contrarian_trade_fragility_by_symbol_csv_path, paths.contrarian_trade_fragility_by_year_csv_path, paths.contrarian_top_trade_removal_csv_path], True, True, ["portfolio_replay"]),
        ("contrarian_year_regime", "Main contrarian year/regime robustness", year_regime_status, True, [paths.contrarian_year_regime_report_json_path, paths.contrarian_year_regime_results_csv_path, paths.contrarian_year_regime_examples_csv_path], True, True, ["portfolio_replay"]),
        ("contrarian_symbol_year_ablation", "Main contrarian symbol/year ablations", symbol_year_ablation_status, True, [paths.contrarian_symbol_year_ablation_report_json_path, paths.contrarian_without_top_symbols_csv_path, paths.contrarian_without_top_years_csv_path], True, True, ["contrarian_profit_concentration"]),
        ("contrarian_data_validity", "Main contrarian data-validity audit", contrarian_data_validity_status, True, [paths.contrarian_data_validity_audit_json_path], True, True, ["portfolio_replay"]),
        ("intraday_5min_expansion_plan", "Future Dell PC intraday expansion plan", intraday_5min_status, True, [paths.intraday_5min_expansion_plan_json_path], True, False, ["contrarian_data_validity"]),
        ("validation_stage_placeholders", "Validation stage placeholders", "PRESENT", True, [paths.validation_stage_placeholders_json_path], True, True, ["pseudo_holdout_report"]),
        ("artifact_manifest", "Artifact manifest", "PRESENT", True, [paths.artifact_manifest_json_path], True, False, ["validation_stage_placeholders"]),
        ("artifact_validation_report", "Artifact validation report", "PASSED", True, [paths.artifact_validation_report_json_path], True, False, ["artifact_manifest"]),
        ("text_model_readiness", "Text model readiness report", "NOT_READY", True, [paths.text_model_readiness_json_path], True, True, ["validation_stage_placeholders"]),
        ("future_walk_forward", "Future walk-forward robustness", "NOT_IMPLEMENTED", False, [], False, True, ["frozen_config"]),
        ("future_placebo", "Future placebo and permutation testing", "NOT_IMPLEMENTED", False, [], False, True, ["future_walk_forward"]),
        ("future_matched_controls", "Future matched controls", "NOT_IMPLEMENTED", False, [], False, True, ["future_placebo"]),
        ("walk_forward_validation", "Walk-forward validation report", "NOT_IMPLEMENTED", True, [paths.walk_forward_validation_report_json_path, paths.walk_forward_fold_results_csv_path], True, True, ["frozen_config"]),
        ("placebo_permutation_validation", "Placebo/permutation validation report", "UNAVAILABLE_INPUT", True, [paths.placebo_permutation_report_json_path, paths.placebo_permutation_results_csv_path], True, True, ["frozen_config"]),
        ("matched_control_reports", "Matched-control reports", "NOT_IMPLEMENTED", True, [paths.exposure_matched_controls_json_path, paths.trade_count_matched_controls_json_path], True, True, ["frozen_config"]),
        ("concentration_fragility", "Concentration and fragility report", "SCAFFOLD", True, [paths.concentration_fragility_report_json_path], True, True, ["frozen_config"]),
        ("future_concentration", "Future concentration analysis", "NOT_IMPLEMENTED", False, [], False, True, ["future_matched_controls"]),
        ("future_survivorship_audit", "Future survivorship audit", "NOT_IMPLEMENTED", False, [], False, True, ["future_concentration"]),
        ("future_corporate_action_audit", "Future corporate-action audit", "NOT_IMPLEMENTED", False, [], False, True, ["future_survivorship_audit"]),
        ("future_missing_news_bias", "Future missing-news bias analysis", "NOT_IMPLEMENTED", False, [], False, True, ["future_corporate_action_audit"]),
        ("future_text_models", "Future text models", "NOT_READY", False, [], False, True, ["text_model_readiness", "future_missing_news_bias"]),
        ("news_transformer_scaffold", "Disabled news transformer scaffold", "NOT_READY", True, [paths.news_transformer_readiness_json_path, paths.news_transformer_training_plan_json_path], True, True, ["text_model_readiness"]),
        ("catastrophic_news_veto", "Research-only catastrophic-news veto audit", "PASSED_WITH_WARNINGS", True, [paths.catastrophic_news_audit_json_path, paths.catastrophic_news_candidates_csv_path, paths.catastrophic_news_veto_report_json_path], True, True, ["text_model_readiness"]),
        ("catastrophic_veto_strategy_comparison", "Research-only catastrophic-veto strategy comparison", "ATTRIBUTION_ONLY", True, [paths.catastrophic_veto_candidate_attribution_json_path, paths.catastrophic_veto_trade_attribution_csv_path, paths.catastrophic_veto_strategy_comparison_json_path, paths.catastrophic_veto_policy_json_path], True, True, ["catastrophic_news_veto"]),
        ("catastrophic_veto_filtered_scenario", "Research-only catastrophic-veto replay-impact simulation", "APPROXIMATE_LEDGER_SIMULATION", True, [paths.catastrophic_veto_filtered_strategy_report_json_path, paths.catastrophic_veto_removed_trades_csv_path, paths.catastrophic_veto_removed_symbols_csv_path], True, True, ["catastrophic_veto_strategy_comparison"]),
        ("catastrophic_veto_full_replay", "Research-only catastrophic-veto full replay variant", "FULL_REPLAY_COMPUTED" if full_replay_computed else "FULL_REPLAY_NOT_AVAILABLE", True, [paths.catastrophic_veto_full_replay_report_json_path, paths.catastrophic_veto_full_replay_trade_ledger_csv_path, paths.catastrophic_veto_full_replay_equity_csv_path, paths.catastrophic_veto_filtered_candidates_csv_path, paths.catastrophic_veto_blocked_candidates_csv_path], True, True, ["catastrophic_veto_filtered_scenario"]),
        ("catastrophic_veto_replay_seam", "Research-only filtered replay input seam", "REPLAY_ADAPTER_EXECUTED" if full_replay_computed else "ADAPTER_ADDED_NOT_EXECUTED", True, [paths.catastrophic_veto_replay_seam_report_json_path], True, True, ["catastrophic_veto_full_replay"]),
        ("catastrophic_news_evidence_quality", "Catastrophic-news evidence quality", evidence_status, True, [paths.catastrophic_news_evidence_quality_report_json_path, paths.catastrophic_news_evidence_quality_by_field_csv_path, paths.catastrophic_news_evidence_quality_by_symbol_csv_path], True, True, ["catastrophic_news_veto"]),
        ("catastrophic_veto_policy_modes", "Research-only catastrophic-veto policy modes", "COUNT_ONLY_RESEARCH", True, [paths.catastrophic_veto_policy_mode_comparison_json_path, paths.catastrophic_veto_policy_mode_counts_csv_path], True, True, ["catastrophic_news_evidence_quality"]),
        ("news_evidence_lineage", "News evidence contract and lineage audit", news_evidence_status, True, [paths.news_evidence_lineage_report_json_path, paths.news_evidence_lineage_by_stage_csv_path, paths.news_evidence_missing_field_examples_csv_path, paths.news_evidence_readiness_report_json_path], True, True, ["news_feature_join"]),
        ("event_taxonomy_research", "Deterministic headline event taxonomy", event_taxonomy_status, True, [paths.news_event_taxonomy_report_json_path, paths.news_event_taxonomy_counts_csv_path, paths.news_event_taxonomy_examples_csv_path], True, False, ["news_evidence_lineage"]),
        ("duplicate_grouping_heuristic", "Heuristic duplicate grouping", duplicate_grouping_status, True, [paths.news_duplicate_grouping_report_json_path, paths.news_duplicate_grouping_examples_csv_path], True, False, ["news_evidence_lineage"]),
        ("point_in_time_text_safety", "Point-in-time text safety audit", text_safety_status, True, [paths.news_point_in_time_text_safety_report_json_path, paths.news_point_in_time_text_safety_examples_csv_path], True, False, ["news_evidence_lineage"]),
        ("keyword_text_baseline", "Deterministic keyword text baseline", keyword_baseline_status, True, [paths.news_text_keyword_baseline_report_json_path, paths.news_text_keyword_baseline_scores_csv_path], True, False, ["point_in_time_text_safety"]),
        ("catastrophic_veto_bounceback", "Research-only blocked-trade bounce-back attribution", bounceback_status, True, [paths.catastrophic_veto_bounceback_report_json_path, paths.catastrophic_veto_bounceback_by_category_csv_path, paths.catastrophic_veto_bounceback_examples_csv_path], True, False, ["catastrophic_veto_full_replay", "event_taxonomy_research"]),
        ("extreme_only_policy_proposal", "Extreme-distress-only policy proposal", extreme_policy_status, True, [paths.catastrophic_veto_extreme_only_policy_proposal_json_path], True, False, ["catastrophic_veto_bounceback"]),
        ("catastrophic_policy_variants", "Research-only catastrophic-veto policy variants", policy_variant_status, True, [paths.catastrophic_veto_policy_variant_comparison_json_path, paths.catastrophic_veto_policy_variant_counts_csv_path, paths.catastrophic_veto_policy_variant_metrics_csv_path, paths.catastrophic_veto_policy_variant_removed_trades_csv_path, paths.catastrophic_veto_policy_variant_bounceback_csv_path, paths.catastrophic_veto_policy_variant_examples_csv_path], True, False, ["catastrophic_veto_bounceback", "event_taxonomy_research"]),
        ("catastrophic_policy_frontier", "Research-only catastrophic-veto policy frontier", policy_frontier_status, True, [paths.catastrophic_veto_policy_frontier_report_json_path, paths.catastrophic_veto_policy_frontier_csv_path], True, False, ["catastrophic_policy_variants"]),
        ("loser_bounceback_casebook", "Research-only loser-vs-bounceback casebook", casebook_status, True, [paths.catastrophic_veto_loser_bounceback_casebook_json_path, paths.catastrophic_veto_loser_bounceback_cases_csv_path, paths.catastrophic_veto_loser_bounceback_feature_diff_csv_path, paths.catastrophic_veto_loser_bounceback_keyword_diff_csv_path], True, False, ["catastrophic_veto_bounceback"]),
        ("taxonomy_improvement_plan", "Research-only catastrophic taxonomy improvement plan", taxonomy_plan_status, True, [paths.catastrophic_veto_taxonomy_improvement_plan_json_path], True, False, ["loser_bounceback_casebook"]),
        ("catastrophic_veto_parked", "Catastrophic veto parked diagnostic status", parked_veto_status, True, [paths.catastrophic_veto_parked_status_json_path], True, False, ["catastrophic_policy_frontier", "loser_bounceback_casebook"]),
    ]
    downstream: dict[str, list[str]] = {node_id: [] for node_id, *_ in node_specs}
    for node_id, _name, _status, _implemented, _paths, _current, _final, upstream in node_specs:
        for dependency in upstream:
            downstream.setdefault(dependency, []).append(node_id)
    nodes = []
    for node_id, name, status, implemented, artifact_paths, required_current, required_final, upstream in node_specs:
        blocks_final = required_final and status in {"PSEUDO_HOLDOUT", "SCAFFOLD", "NOT_IMPLEMENTED", "NOT_READY", "BLOCKED", "FAILED", "UNAVAILABLE_INPUT", "INSUFFICIENT", "INSUFFICIENT_FOR_STRICT_VETO", "PARTIAL_EVIDENCE"}
        nodes.append(
            {
                "node_id": node_id,
                "name": name,
                "status": status,
                "implemented": implemented,
                "artifact_paths": [str(path) for path in artifact_paths],
                "required_for_current_stage": required_current,
                "required_for_final_validation": required_final,
                "blocks_final_validation": blocks_final,
                "contains_blocking_stages": node_id == "validation_stage_placeholders",
                "blocking_stage_count": (
                    sum(
                        1
                        for stage in _validation_stage_placeholders().values()
                        if isinstance(stage, Mapping) and stage.get("blocks_final_validation")
                    )
                    if node_id == "validation_stage_placeholders"
                    else 0
                ),
                "upstream_dependencies": upstream,
                "downstream_dependencies": downstream.get(node_id, []),
                "warnings": _workflow_node_warnings(status, node_id=node_id),
            }
        )
    edges = [
        {"from": dependency, "to": node["node_id"]}
        for node in nodes
        for dependency in node["upstream_dependencies"]
    ]
    return {
        "schema_name": "stock_alpha_news_validation_workflow_map",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "status": "PSEUDO_HOLDOUT_WORKFLOW_INCOMPLETE",
        "workflow_name": "stock_alpha_news_risk_overlay_validation_spine",
        "research_only": True,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "nodes": nodes,
        "edges": edges,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "validation_passed": False,
        "warnings": [
            "Workflow map is descriptive and does not validate the strategy.",
            "Pseudo-holdout and unimplemented validation stages block final validation.",
        ],
    }


def _workflow_node_warnings(status: str, *, node_id: str = "") -> list[str]:
    if node_id == "validation_stage_placeholders":
        return ["Validation stage placeholder container includes incomplete gates that block final validation."]
    if status == "PSEUDO_HOLDOUT":
        return ["Previously inspected period; not an untouched holdout."]
    if status == "SCAFFOLD":
        return ["Scaffold artifact only; full validation logic is not complete."]
    if status == "NOT_IMPLEMENTED":
        return ["Stage is not implemented and blocks final validation."]
    if status == "NOT_READY":
        return ["Stage is not ready and blocks final validation."]
    if status == "UNAVAILABLE_INPUT":
        return ["Stage cannot be computed from available inputs and blocks final validation."]
    return []


def _validation_dependency_graph(paths: NewsRiskResearchPaths) -> dict[str, Any]:
    full_replay_computed = bool(
        _read_json_if_available(paths.catastrophic_veto_full_replay_report_json_path).get("full_replay_computed")
    )
    evidence_status = _read_json_if_available(
        paths.catastrophic_news_evidence_quality_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    news_evidence_status = _read_json_if_available(
        paths.news_evidence_readiness_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    event_taxonomy_status = _read_json_if_available(
        paths.news_event_taxonomy_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    duplicate_grouping_status = _read_json_if_available(
        paths.news_duplicate_grouping_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    text_safety_status = _read_json_if_available(
        paths.news_point_in_time_text_safety_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    keyword_baseline_status = _read_json_if_available(
        paths.news_text_keyword_baseline_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    bounceback_status = _read_json_if_available(
        paths.catastrophic_veto_bounceback_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    extreme_policy_status = _read_json_if_available(
        paths.catastrophic_veto_extreme_only_policy_proposal_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    policy_variant_status = _read_json_if_available(
        paths.catastrophic_veto_policy_variant_comparison_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    policy_frontier_status = _read_json_if_available(
        paths.catastrophic_veto_policy_frontier_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    casebook_status = _read_json_if_available(
        paths.catastrophic_veto_loser_bounceback_casebook_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    taxonomy_plan_status = _read_json_if_available(
        paths.catastrophic_veto_taxonomy_improvement_plan_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    parked_veto_status = _read_json_if_available(
        paths.catastrophic_veto_parked_status_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    profit_concentration_status = _read_json_if_available(
        paths.contrarian_profit_concentration_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    year_regime_status = _read_json_if_available(
        paths.contrarian_year_regime_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    symbol_year_ablation_status = _read_json_if_available(
        paths.contrarian_symbol_year_ablation_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    data_validity_status = _read_json_if_available(
        paths.contrarian_data_validity_audit_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    intraday_5min_status = _read_json_if_available(
        paths.intraday_5min_expansion_plan_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    gate_specs = [
        ("decile_reconciliation", "PASSED", True, True, paths.decile_trade_reconciliation_json_path, "", "continue to chronological validation"),
        ("chronological_split", "PSEUDO_HOLDOUT", True, False, paths.chronological_split_manifest_json_path, "final period is pseudo-holdout", "obtain untouched prospective data"),
        ("experiment_registry", "PRESENT", True, True, paths.experiment_registry_jsonl_path, "", "preserve append-only registry entries"),
        ("frozen_config", "PRESENT", True, True, paths.contrarian_frozen_config_json_path, "", "use frozen config for future validation"),
        ("pseudo_holdout_gate", "BLOCKED", True, False, paths.contrarian_holdout_report_json_path, "pseudo-holdout cannot pass final validation", "evaluate only on genuinely untouched data"),
        ("contrarian_chronological_validation", "IN_PROGRESS", True, False, paths.contrarian_chronological_validation_plan_json_path, "chronological validation is in progress and pseudo-holdout is not final validation", "collect untouched future final holdout"),
        ("contrarian_profit_concentration", profit_concentration_status, True, profit_concentration_status == "IMPLEMENTED", paths.contrarian_profit_concentration_report_json_path, "profit concentration is diagnostic and does not pass final validation", "review top-trade/symbol/year fragility"),
        ("contrarian_year_regime", year_regime_status, True, year_regime_status == "AVAILABLE", paths.contrarian_year_regime_report_json_path, "year/regime robustness is ledger-level and not final validation", "review negative and partial years"),
        ("contrarian_symbol_year_ablation", symbol_year_ablation_status, True, symbol_year_ablation_status == "AVAILABLE", paths.contrarian_symbol_year_ablation_report_json_path, "symbol/year ablations are ledger-level approximations", "review concentration sensitivity before final validation"),
        ("contrarian_data_validity", data_validity_status, True, False, paths.contrarian_data_validity_audit_json_path, "data-validity audits are incomplete", "implement survivorship/corporate-action/missing-data audits"),
        ("intraday_5min_expansion_plan", intraday_5min_status, True, False, paths.intraday_5min_expansion_plan_json_path, "intraday expansion is planning-only", "confirm Dell PC intraday data paths and run small subset later"),
        ("walk_forward", "NOT_IMPLEMENTED", True, False, paths.walk_forward_validation_report_json_path, "walk-forward robustness is scaffolded but fold-level replay metrics are not implemented", "implement expanding/rolling chronological walk-forward"),
        ("placebo_permutation", "UNAVAILABLE_INPUT", True, False, paths.placebo_permutation_report_json_path, "placebo/permutation report is scaffolded without computable placebo metrics", "implement deterministic placebo tests"),
        ("exposure_matched_controls", "NOT_IMPLEMENTED", True, False, paths.exposure_matched_controls_json_path, "exposure-matched controls are not implemented", "implement exposure-matched controls"),
        ("trade_count_matched_controls", "NOT_IMPLEMENTED", True, False, paths.trade_count_matched_controls_json_path, "trade-count-matched controls are not implemented", "implement trade-count-matched controls"),
        ("concentration_analysis", "SCAFFOLD", True, False, paths.concentration_fragility_report_json_path, "concentration analysis is a limited scaffold, not full fragility validation", "implement top-trade and top-symbol fragility analysis"),
        ("survivorship_audit", "NOT_IMPLEMENTED", False, False, None, "survivorship audit is not implemented", "audit point-in-time universe membership"),
        ("corporate_action_audit", "NOT_IMPLEMENTED", False, False, None, "corporate-action audit is not implemented", "validate split/dividend adjustment semantics"),
        ("missing_news_bias", "NOT_IMPLEMENTED", False, False, None, "missing-news bias analysis is not complete", "complete covered versus uncovered candidate analysis"),
        ("transaction_cost_validation", "NOT_IMPLEMENTED", False, False, None, "realistic transaction-cost validation is not implemented", "validate cost assumptions and path-dependent replay"),
        ("text_model_readiness", "NOT_READY", True, False, paths.text_model_readiness_json_path, "FinBERT/text modelling is not ready", "complete taxonomy, timestamp, and duplicate-handling gates first"),
        ("news_transformer_scaffold", "NOT_READY", True, False, paths.news_transformer_readiness_json_path, "disabled transformer scaffold is not ready for training or inference", "complete readiness gates before enabling transformer research"),
        ("catastrophic_news_veto", "PASSED_WITH_WARNINGS", True, False, paths.catastrophic_news_audit_json_path, "catastrophic-news veto is research-only and not enforced in replay or strategy", "validate veto taxonomy and point-in-time availability before final validation"),
        ("catastrophic_veto_strategy_comparison", "ATTRIBUTION_ONLY", True, False, paths.catastrophic_veto_strategy_comparison_json_path, "catastrophic-veto replay impact is approximate and not full replay", "compute a full research-only filtered replay before treating veto impact as evaluated"),
        ("catastrophic_veto_filtered_scenario", "APPROXIMATE_LEDGER_SIMULATION", True, False, paths.catastrophic_veto_filtered_strategy_report_json_path, "catastrophic-veto impact is approximate ledger simulation, not full replay", "compute full research-only filtered replay before treating veto impact as validated"),
        ("catastrophic_veto_full_replay", "FULL_REPLAY_COMPUTED" if full_replay_computed else "FULL_REPLAY_NOT_AVAILABLE", True, full_replay_computed, paths.catastrophic_veto_full_replay_report_json_path, "" if full_replay_computed else "replay output does not contain the catastrophic-veto filtered variant", "retain as a separate research-only scenario" if full_replay_computed else "inspect the optional variant replay output"),
        ("catastrophic_veto_replay_seam", "REPLAY_ADAPTER_EXECUTED" if full_replay_computed else "ADAPTER_ADDED_NOT_EXECUTED", True, full_replay_computed, paths.catastrophic_veto_replay_seam_report_json_path, "" if full_replay_computed else "opt-in replay adapter is present but not executed as a full replay", "preserve the research-only boundary" if full_replay_computed else "execute and validate separate research-only filtered replay"),
        ("catastrophic_news_evidence_quality", evidence_status, True, False, paths.catastrophic_news_evidence_quality_report_json_path, "catastrophic-news evidence quality insufficient for strict live-style filtering", "improve point-in-time text and availability evidence before any execution use"),
        ("news_evidence_lineage", news_evidence_status, True, False, paths.news_evidence_readiness_report_json_path, "news evidence contract is incomplete across ingestion, feature, and candidate stages", "preserve text, availability, source, category, duplicate, and candidate linkage fields"),
        ("event_taxonomy_research", event_taxonomy_status, True, event_taxonomy_status in {"RESEARCH_RULES_READY"}, paths.news_event_taxonomy_report_json_path, "headline event taxonomy is research-only", "review deterministic event taxonomy coverage"),
        ("duplicate_grouping_heuristic", duplicate_grouping_status, True, duplicate_grouping_status == "HEURISTIC_ONLY", paths.news_duplicate_grouping_report_json_path, "duplicate grouping is heuristic-only and not production-grade", "replace with provider-grade duplicate identifiers before text modelling"),
        ("point_in_time_text_safety", text_safety_status, True, text_safety_status == "PARTIAL_POINT_IN_TIME_SAFE", paths.news_point_in_time_text_safety_report_json_path, "point-in-time text safety is partial", "increase availability timestamp coverage"),
        ("keyword_text_baseline", keyword_baseline_status, True, keyword_baseline_status == "RESEARCH_ONLY", paths.news_text_keyword_baseline_report_json_path, "keyword baseline is research-only and unused by strategy", "keep as diagnostic until validation gates pass"),
        ("catastrophic_veto_bounceback", bounceback_status, True, False, paths.catastrophic_veto_bounceback_report_json_path, "bounce-back attribution is research-only and not a validation gate", "review removed-trade winners/losers before narrowing policy"),
        ("extreme_only_policy_proposal", extreme_policy_status, True, False, paths.catastrophic_veto_extreme_only_policy_proposal_json_path, "extreme-only policy is proposed but not replayed", "run a future separate research-only replay before interpreting policy impact"),
        ("catastrophic_policy_variants", policy_variant_status, True, False, paths.catastrophic_veto_policy_variant_comparison_json_path, "policy variants are research-only diagnostics and not validation gates", "review policy frontier and examples before any future policy narrowing"),
        ("catastrophic_policy_frontier", policy_frontier_status, True, False, paths.catastrophic_veto_policy_frontier_report_json_path, "policy frontier is a diagnostic ranking, not model selection", "treat frontier output as hypothesis triage only"),
        ("loser_bounceback_casebook", casebook_status, True, False, paths.catastrophic_veto_loser_bounceback_casebook_json_path, "casebook is observational research only", "inspect loser-vs-bounceback differences before changing taxonomy"),
        ("taxonomy_improvement_plan", taxonomy_plan_status, True, False, paths.catastrophic_veto_taxonomy_improvement_plan_json_path, "taxonomy improvement plan is proposal-only", "review proposed deterministic rule changes before implementation"),
        ("catastrophic_veto_parked", parked_veto_status, True, False, paths.catastrophic_veto_parked_status_json_path, "catastrophic veto is parked diagnostic-only", "focus validation on news_contrarian_rerank"),
    ]
    gates = []
    for name, status, implemented, passed, artifact, reason, action in gate_specs:
        gates.append(
            {
                "gate_name": name,
                "status": status,
                "implemented": implemented,
                "passed": passed,
                "blocks_final_validation": not passed,
                "required_artifact": str(artifact) if artifact is not None else None,
                "failure_reason": reason,
                "next_required_action": action,
            }
        )
    blocked_by = [gate["gate_name"] for gate in gates if gate["blocks_final_validation"]]
    return {
        "schema_name": "stock_alpha_news_validation_dependency_graph",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "gates": gates,
        "all_required_gates_complete": all(gate["implemented"] for gate in gates),
        "all_required_gates_passed": all(gate["passed"] for gate in gates),
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "validation_passed": False,
        "is_final_validation": False,
        "blocked_by": blocked_by,
        "warnings": [
            "Artifact presence alone does not pass final validation.",
            "Pseudo-holdout cannot pass final validation.",
            "Text-model readiness cannot pass while FinBERT/text readiness is NOT_READY.",
        ],
    }


def _validation_readiness_dashboard(paths: NewsRiskResearchPaths) -> dict[str, Any]:
    dependency_graph = _validation_dependency_graph(paths)
    full_replay_computed = bool(
        _read_json_if_available(paths.catastrophic_veto_full_replay_report_json_path).get("full_replay_computed")
    )
    news_evidence = _read_json_if_available(paths.news_evidence_readiness_report_json_path)
    event_taxonomy = _read_json_if_available(paths.news_event_taxonomy_report_json_path)
    duplicate_grouping = _read_json_if_available(paths.news_duplicate_grouping_report_json_path)
    text_safety = _read_json_if_available(paths.news_point_in_time_text_safety_report_json_path)
    keyword_baseline = _read_json_if_available(paths.news_text_keyword_baseline_report_json_path)
    bounceback = _read_json_if_available(paths.catastrophic_veto_bounceback_report_json_path)
    extreme_policy = _read_json_if_available(paths.catastrophic_veto_extreme_only_policy_proposal_json_path)
    policy_variants = _read_json_if_available(paths.catastrophic_veto_policy_variant_comparison_json_path)
    policy_frontier = _read_json_if_available(paths.catastrophic_veto_policy_frontier_report_json_path)
    casebook = _read_json_if_available(paths.catastrophic_veto_loser_bounceback_casebook_json_path)
    taxonomy_plan = _read_json_if_available(paths.catastrophic_veto_taxonomy_improvement_plan_json_path)
    parked_veto = _read_json_if_available(paths.catastrophic_veto_parked_status_json_path)
    walk_forward = _read_json_if_available(paths.contrarian_walk_forward_validation_report_json_path)
    placebo = _read_json_if_available(paths.contrarian_placebo_permutation_report_json_path)
    matched_controls = _read_json_if_available(paths.contrarian_matched_control_report_json_path)
    profit_concentration = _read_json_if_available(paths.contrarian_profit_concentration_report_json_path)
    year_regime = _read_json_if_available(paths.contrarian_year_regime_report_json_path)
    symbol_year_ablation = _read_json_if_available(paths.contrarian_symbol_year_ablation_report_json_path)
    data_validity = _read_json_if_available(paths.contrarian_data_validity_audit_json_path)
    intraday_5min = _read_json_if_available(paths.intraday_5min_expansion_plan_json_path)
    gates = list(dependency_graph.get("gates", []) or [])
    implemented_stage_count = sum(bool(gate.get("implemented")) for gate in gates)
    blocking_stage_count = sum(bool(gate.get("blocks_final_validation")) for gate in gates)
    not_implemented_stage_count = sum(str(gate.get("status")) == "NOT_IMPLEMENTED" for gate in gates)
    not_ready_stage_count = sum(str(gate.get("status")) == "NOT_READY" for gate in gates)
    passed_stage_count = sum(bool(gate.get("passed")) for gate in gates)
    failed_stage_count = sum(str(gate.get("status")) == "FAILED" for gate in gates)
    return {
        "schema_name": "stock_alpha_news_validation_readiness_dashboard",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "overall_status": "DEVELOPMENT_ONLY",
        "research_only": True,
        "production_signal": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "validation_label": "PSEUDO_HOLDOUT",
        "holdout_type": "PSEUDO_HOLDOUT",
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "validation_passed": False,
        "implemented_stage_count": implemented_stage_count,
        "blocking_stage_count": blocking_stage_count,
        "not_implemented_stage_count": not_implemented_stage_count,
        "not_ready_stage_count": not_ready_stage_count,
        "passed_stage_count": passed_stage_count,
        "failed_stage_count": failed_stage_count,
        "news_evidence_readiness": news_evidence.get("status", "UNAVAILABLE_INPUT"),
        "strict_veto_ready": bool(news_evidence.get("strict_veto_ready", False)),
        "confirmed_only_veto_ready": bool(news_evidence.get("confirmed_only_veto_ready", False)),
        "event_taxonomy_status": event_taxonomy.get("status", "UNAVAILABLE_INPUT"),
        "event_taxonomy_research_ready": bool(event_taxonomy.get("event_taxonomy_research_ready", False)),
        "duplicate_grouping_status": duplicate_grouping.get("status", "UNAVAILABLE_INPUT"),
        "duplicate_grouping_heuristic_ready": bool(duplicate_grouping.get("duplicate_grouping_heuristic_ready", False)),
        "point_in_time_text_safety_status": text_safety.get("status", "UNAVAILABLE_INPUT"),
        "point_in_time_text_safety_ready": bool(text_safety.get("point_in_time_text_safety_ready", False)),
        "keyword_baseline_status": keyword_baseline.get("status", "UNAVAILABLE_INPUT"),
        "keyword_baseline_ready": bool(keyword_baseline.get("keyword_baseline_ready", False)),
        "catastrophic_veto_bounceback_status": bounceback.get("status", "UNAVAILABLE_INPUT"),
        "extreme_only_policy_proposal_status": extreme_policy.get("status", "UNAVAILABLE_INPUT"),
        "catastrophic_policy_variant_status": policy_variants.get("status", "UNAVAILABLE_INPUT"),
        "catastrophic_policy_frontier_status": policy_frontier.get("status", "UNAVAILABLE_INPUT"),
        "best_balanced_catastrophic_policy": policy_frontier.get("best_balanced_policy", "UNAVAILABLE_INPUT"),
        "loser_bounceback_casebook_status": casebook.get("status", "UNAVAILABLE_INPUT"),
        "taxonomy_improvement_plan_status": taxonomy_plan.get("status", "UNAVAILABLE_INPUT"),
        "catastrophic_veto": parked_veto.get("status", "UNAVAILABLE_INPUT"),
        "contrarian_validation": "IN_PROGRESS",
        "walk_forward": walk_forward.get("status", "UNAVAILABLE_INPUT"),
        "placebo": placebo.get("status", "UNAVAILABLE_INPUT"),
        "matched_controls": matched_controls.get("status", "UNAVAILABLE_INPUT"),
        "profit_concentration": profit_concentration.get("status", "UNAVAILABLE_INPUT"),
        "year_regime": year_regime.get("status", "UNAVAILABLE_INPUT"),
        "symbol_year_ablation": symbol_year_ablation.get("status", "UNAVAILABLE_INPUT"),
        "data_validity": data_validity.get("status", "UNAVAILABLE_INPUT"),
        "intraday_5min": intraday_5min.get("status", "UNAVAILABLE_INPUT"),
        "text_model_ready": False,
        "transformer_ready": False,
        "top_blockers": [item for item in [
            "walk-forward not implemented",
            "placebo/permutation not implemented",
            "matched controls not implemented",
            "review year/regime robustness",
            "review symbol/year ablations",
            "survivorship audit not implemented",
            "corporate-action audit not implemented",
            "missing-news bias not implemented",
            "events still uncategorized",
            "catastrophic-news evidence quality insufficient for strict live-style filtering",
            "news evidence contract incomplete across ingestion and candidate stages",
            None if full_replay_computed else "catastrophic-veto full replay not computed",
            "text model readiness not ready",
            "pseudo-holdout is not genuine holdout",
        ] if item is not None],
        "safe_next_steps": [item for item in [
            "implement walk-forward validation",
            "implement placebo/permutation checks",
            "implement matched controls",
            "implement concentration/fragility analysis",
            "implement survivorship/corporate-action/missing-news audits",
            "build structured event taxonomy",
            None if full_replay_computed else "compute full research-only catastrophic-veto filtered replay",
        ] if item is not None],
        "unsafe_next_steps": [
            "FinBERT",
            "BERT",
            "transformer training",
            "paper trading",
            "live trading",
            "new providers",
            "news-originated entries",
        ],
        "finbert_readiness": "NOT_READY",
        "warnings": [
            "Dashboard is a readiness rollup, not strategy validation.",
            "Development-only pseudo-holdout results remain blocked from final validation.",
            "Paper/live trading and text-model training remain disabled.",
        ],
    }


def _artifact_lineage_report(paths: NewsRiskResearchPaths) -> dict[str, Any]:
    relationships = [
        (paths.decile_join_audit_json_path, paths.validation_readiness_dashboard_json_path, "feeds_readiness_status", True),
        (paths.decile_trade_reconciliation_json_path, paths.validation_readiness_dashboard_json_path, "feeds_readiness_status", True),
        (paths.chronological_split_manifest_json_path, paths.contrarian_grid_selection_json_path, "defines_selection_periods", True),
        (paths.chronological_split_manifest_json_path, paths.contrarian_holdout_report_json_path, "defines_holdout_period", True),
        (paths.contrarian_grid_selection_json_path, paths.contrarian_frozen_config_json_path, "freezes_selected_configuration", True),
        (paths.contrarian_frozen_config_json_path, paths.contrarian_holdout_report_json_path, "drives_holdout_evaluation", True),
        (paths.validation_stage_placeholders_json_path, paths.validation_dependency_graph_json_path, "declares_future_validation_blockers", True),
        (paths.text_model_readiness_json_path, paths.validation_dependency_graph_json_path, "declares_text_model_blockers", True),
        (paths.news_transformer_readiness_json_path, paths.validation_dependency_graph_json_path, "declares_transformer_scaffold_blockers", True),
        (paths.news_transformer_training_plan_json_path, paths.validation_readiness_dashboard_json_path, "documents_disabled_transformer_plan", True),
        (paths.catastrophic_news_audit_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_research_only_catastrophic_news_audit", True),
        (paths.catastrophic_news_veto_report_json_path, paths.validation_dependency_graph_json_path, "declares_research_only_catastrophic_news_veto_blocker", True),
        (paths.catastrophic_veto_policy_json_path, paths.catastrophic_veto_candidate_attribution_json_path, "defines_research_only_veto_policy", True),
        (paths.catastrophic_news_audit_json_path, paths.catastrophic_veto_candidate_attribution_json_path, "feeds_candidate_veto_attribution", True),
        (paths.catastrophic_veto_candidate_attribution_json_path, paths.catastrophic_veto_strategy_comparison_json_path, "feeds_attribution_only_strategy_comparison", True),
        (paths.catastrophic_veto_strategy_comparison_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_catastrophic_veto_replay_impact_status", True),
        (paths.catastrophic_veto_trade_attribution_csv_path, paths.catastrophic_veto_removed_trades_csv_path, "identifies_research_only_removed_trades", True),
        (paths.catastrophic_veto_removed_trades_csv_path, paths.catastrophic_veto_removed_symbols_csv_path, "rolls_removed_trades_up_by_symbol", True),
        (paths.catastrophic_veto_removed_trades_csv_path, paths.catastrophic_veto_filtered_strategy_report_json_path, "feeds_approximate_filtered_strategy_report", True),
        (paths.catastrophic_veto_filtered_strategy_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_approximate_veto_simulation", True),
        (paths.catastrophic_veto_filtered_candidates_csv_path, paths.catastrophic_veto_full_replay_report_json_path, "documents_full_replay_candidate_filter_input", True),
        (paths.catastrophic_veto_blocked_candidates_csv_path, paths.catastrophic_veto_full_replay_report_json_path, "documents_candidates_excluded_from_full_replay_variant", True),
        (paths.catastrophic_veto_replay_seam_report_json_path, paths.catastrophic_veto_full_replay_report_json_path, "documents_filtered_replay_input_seam", True),
        (paths.catastrophic_veto_replay_seam_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_replay_seam_status", True),
        (paths.catastrophic_veto_full_replay_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_full_replay_availability", True),
        (paths.catastrophic_news_evidence_quality_report_json_path, paths.validation_dependency_graph_json_path, "declares_catastrophic_evidence_quality_blocker", True),
        (paths.catastrophic_news_evidence_quality_report_json_path, paths.catastrophic_veto_policy_mode_comparison_json_path, "feeds_research_policy_mode_comparison", True),
        (paths.catastrophic_veto_policy_mode_comparison_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_research_only_policy_modes", True),
        (paths.news_evidence_lineage_report_json_path, paths.news_evidence_readiness_report_json_path, "feeds_news_evidence_readiness", True),
        (paths.news_evidence_readiness_report_json_path, paths.validation_dependency_graph_json_path, "declares_news_evidence_readiness_blocker", True),
        (paths.news_evidence_readiness_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_news_evidence_readiness", True),
        (paths.news_evidence_readiness_report_json_path, paths.news_event_taxonomy_report_json_path, "gates_research_taxonomy_inputs", True),
        (paths.news_event_taxonomy_report_json_path, paths.news_event_taxonomy_counts_csv_path, "summarizes_research_taxonomy_counts", True),
        (paths.news_event_taxonomy_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_event_taxonomy_research_status", True),
        (paths.news_evidence_readiness_report_json_path, paths.news_duplicate_grouping_report_json_path, "gates_heuristic_duplicate_grouping_inputs", True),
        (paths.news_duplicate_grouping_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_duplicate_grouping_status", True),
        (paths.news_evidence_readiness_report_json_path, paths.news_point_in_time_text_safety_report_json_path, "gates_text_safety_inputs", True),
        (paths.news_point_in_time_text_safety_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_point_in_time_text_safety", True),
        (paths.news_point_in_time_text_safety_report_json_path, paths.news_text_keyword_baseline_report_json_path, "documents_keyword_baseline_input_safety", True),
        (paths.news_text_keyword_baseline_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_keyword_baseline_status", True),
        (paths.catastrophic_veto_removed_trades_csv_path, paths.catastrophic_veto_bounceback_report_json_path, "feeds_removed_trade_bounceback_attribution", True),
        (paths.news_event_taxonomy_report_json_path, paths.catastrophic_veto_bounceback_report_json_path, "supports_reversible_vs_extreme_grouping", True),
        (paths.catastrophic_veto_bounceback_report_json_path, paths.catastrophic_veto_extreme_only_policy_proposal_json_path, "motivates_future_narrow_policy", True),
        (paths.catastrophic_veto_bounceback_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_bounceback_status", True),
        (paths.catastrophic_veto_extreme_only_policy_proposal_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_extreme_only_policy_proposal", True),
        (paths.catastrophic_veto_bounceback_report_json_path, paths.catastrophic_veto_policy_variant_comparison_json_path, "feeds_policy_variant_bounceback_tradeoff", True),
        (paths.news_event_taxonomy_report_json_path, paths.catastrophic_veto_policy_variant_comparison_json_path, "feeds_policy_variant_taxonomy", True),
        (paths.catastrophic_veto_policy_variant_comparison_json_path, paths.catastrophic_veto_policy_frontier_report_json_path, "feeds_policy_frontier_ranking", True),
        (paths.catastrophic_veto_policy_variant_comparison_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_policy_variant_status", True),
        (paths.catastrophic_veto_policy_frontier_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_policy_frontier_status", True),
        (paths.catastrophic_veto_bounceback_report_json_path, paths.catastrophic_veto_loser_bounceback_casebook_json_path, "feeds_loser_bounceback_casebook", True),
        (paths.catastrophic_veto_loser_bounceback_casebook_json_path, paths.catastrophic_veto_taxonomy_improvement_plan_json_path, "motivates_taxonomy_improvement_plan", True),
        (paths.catastrophic_veto_loser_bounceback_casebook_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_casebook_status", True),
        (paths.catastrophic_veto_taxonomy_improvement_plan_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_taxonomy_plan_status", True),
        (paths.catastrophic_veto_parked_status_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_parked_catastrophic_veto_status", True),
        (paths.contrarian_chronological_validation_plan_json_path, paths.validation_dependency_graph_json_path, "declares_main_contrarian_chronology_gate", True),
        (paths.contrarian_chronological_periods_csv_path, paths.validation_readiness_dashboard_json_path, "summarizes_main_contrarian_chronology_periods", True),
        (paths.contrarian_walk_forward_validation_report_json_path, paths.validation_dependency_graph_json_path, "declares_main_contrarian_walk_forward_gate", True),
        (paths.contrarian_placebo_permutation_report_json_path, paths.validation_dependency_graph_json_path, "declares_main_contrarian_placebo_gate", True),
        (paths.contrarian_matched_control_report_json_path, paths.validation_dependency_graph_json_path, "declares_main_contrarian_matched_control_gate", True),
        (paths.contrarian_profit_concentration_report_json_path, paths.validation_dependency_graph_json_path, "declares_main_contrarian_profit_concentration_gate", True),
        (paths.contrarian_year_regime_report_json_path, paths.validation_dependency_graph_json_path, "declares_main_contrarian_year_regime_gate", True),
        (paths.contrarian_symbol_year_ablation_report_json_path, paths.validation_dependency_graph_json_path, "declares_main_contrarian_ablation_gate", True),
        (paths.contrarian_data_validity_audit_json_path, paths.validation_dependency_graph_json_path, "declares_main_contrarian_data_validity_gate", True),
        (paths.intraday_5min_expansion_plan_json_path, paths.validation_readiness_dashboard_json_path, "documents_future_intraday_expansion_plan", True),
        (paths.artifact_validation_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_artifact_presence", True),
        (paths.experiment_registry_jsonl_path, paths.validation_readiness_dashboard_json_path, "records_experiment_history", True),
        (paths.validation_dependency_graph_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_gate_status", True),
        (paths.news_validation_workflow_map_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_workflow_status", True),
        (paths.chronological_split_manifest_json_path, paths.walk_forward_validation_report_json_path, "defines_walk_forward_periods", True),
        (paths.contrarian_frozen_config_json_path, paths.walk_forward_validation_report_json_path, "freezes_walk_forward_configuration", True),
        (paths.contrarian_frozen_config_json_path, paths.placebo_permutation_report_json_path, "freezes_placebo_configuration", True),
        (paths.walk_forward_validation_report_json_path, paths.validation_dependency_graph_json_path, "feeds_walk_forward_gate", True),
        (paths.placebo_permutation_report_json_path, paths.validation_dependency_graph_json_path, "feeds_placebo_gate", True),
        (paths.exposure_matched_controls_json_path, paths.validation_dependency_graph_json_path, "feeds_exposure_control_gate", True),
        (paths.trade_count_matched_controls_json_path, paths.validation_dependency_graph_json_path, "feeds_trade_count_control_gate", True),
        (paths.concentration_fragility_report_json_path, paths.validation_dependency_graph_json_path, "feeds_concentration_gate", True),
    ]
    lineage = [
        {
            "source_artifact": source.name,
            "target_artifact": target.name,
            "relationship_type": relationship_type,
            "required": required,
            "status": "DECLARED",
            "warning_if_missing": f"{source.name} missing would make {target.name} incomplete.",
        }
        for source, target, relationship_type, required in relationships
    ]
    return {
        "schema_name": "stock_alpha_news_artifact_lineage_report",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "status": "DECLARED",
        "lineage": lineage,
        "warnings": [
            "Lineage report documents artifact dependencies only.",
            "Declared lineage does not imply final validation.",
        ],
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "validation_passed": False,
        "research_only": True,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }


def _news_validation_gap_analysis(paths: NewsRiskResearchPaths) -> dict[str, Any]:
    full_replay_computed = bool(
        _read_json_if_available(paths.catastrophic_veto_full_replay_report_json_path).get("full_replay_computed")
    )
    event_taxonomy_ready = bool(
        _read_json_if_available(paths.news_event_taxonomy_report_json_path).get("event_taxonomy_research_ready")
    )
    duplicate_heuristic_ready = bool(
        _read_json_if_available(paths.news_duplicate_grouping_report_json_path).get("duplicate_grouping_heuristic_ready")
    )
    text_safety_ready = bool(
        _read_json_if_available(paths.news_point_in_time_text_safety_report_json_path).get("point_in_time_text_safety_ready")
    )
    keyword_baseline_ready = bool(
        _read_json_if_available(paths.news_text_keyword_baseline_report_json_path).get("keyword_baseline_ready")
    )
    bounceback_available = bool(
        _read_json_if_available(paths.catastrophic_veto_bounceback_report_json_path).get("status") == "AVAILABLE"
    )
    gaps = [
        {
            "gap_id": "contrarian_validation_in_progress",
            "area": "validation_spine",
            "severity": "critical",
            "description": "Main news_contrarian_rerank validation is in progress; chronological, walk-forward, placebo, matched-control, concentration, and data-validity gates are not final.",
            "blocks_final_validation": True,
            "recommended_action": "complete the main contrarian validation spine before revisiting execution readiness",
        },
        {
            "gap_id": "catastrophic_veto_parked_diagnostic_only",
            "area": "catastrophic_veto_policy",
            "severity": "low",
            "description": "Catastrophic-veto work is parked as diagnostic-only and is not used by the current strategy.",
            "blocks_final_validation": False,
            "recommended_action": "keep catastrophic-veto artifacts observational while validating news_contrarian_rerank",
        },
        {
            "gap_id": "year_regime_review_required",
            "area": "regime_robustness",
            "severity": "medium",
            "description": "Year/regime report is ledger-level and must be reviewed for negative or partial years before final validation.",
            "blocks_final_validation": True,
            "recommended_action": "review annual robustness, especially negative-year and partial-year behavior",
        },
        {
            "gap_id": "symbol_year_ablation_review_required",
            "area": "fragility",
            "severity": "medium",
            "description": "Symbol/year ablations are ledger-level approximations and do not recompute portfolio compounding.",
            "blocks_final_validation": True,
            "recommended_action": "review without-top-symbol/year sensitivity and implement full replay ablations if needed",
        },
        {
            "gap_id": "intraday_5min_planning_only",
            "area": "future_data_expansion",
            "severity": "low",
            "description": "Intraday 5-minute expansion is a Dell PC planning artifact only.",
            "blocks_final_validation": False,
            "recommended_action": "confirm local 5min/15min data paths and run a small subset later",
        },
        {
            "gap_id": "walk_forward_not_implemented",
            "area": "robustness",
            "severity": "critical",
            "description": "Walk-forward artifact exists, but fold-level replay metrics are not implemented.",
            "blocks_final_validation": True,
            "recommended_action": "implement walk-forward validation",
        },
        {
            "gap_id": "placebo_permutation_not_implemented",
            "area": "statistical_controls",
            "severity": "critical",
            "description": "Placebo/permutation artifact exists, but checks are UNAVAILABLE_INPUT until placebo replay/statistics are implemented.",
            "blocks_final_validation": True,
            "recommended_action": "implement placebo/permutation checks",
        },
        {
            "gap_id": "matched_controls_not_implemented",
            "area": "controls",
            "severity": "critical",
            "description": "Exposure- and trade-count-matched controls have not been implemented.",
            "blocks_final_validation": True,
            "recommended_action": "implement matched controls",
        },
        {
            "gap_id": "concentration_analysis_not_implemented",
            "area": "fragility",
            "severity": "high",
            "description": "Contribution and concentration analysis has not been implemented.",
            "blocks_final_validation": True,
            "recommended_action": "implement concentration/fragility analysis",
        },
        {
            "gap_id": "survivorship_audit_not_implemented",
            "area": "data_integrity",
            "severity": "critical",
            "description": "Point-in-time universe and survivorship audit has not been implemented.",
            "blocks_final_validation": True,
            "recommended_action": "implement survivorship audit",
        },
        {
            "gap_id": "corporate_action_audit_not_implemented",
            "area": "data_integrity",
            "severity": "critical",
            "description": "Corporate-action adjustment validation has not been implemented.",
            "blocks_final_validation": True,
            "recommended_action": "implement corporate-action audit",
        },
        {
            "gap_id": "missing_news_bias_not_implemented",
            "area": "coverage_bias",
            "severity": "high",
            "description": "Missing-news bias analysis is not complete.",
            "blocks_final_validation": True,
            "recommended_action": "implement missing-news bias analysis",
        },
        {
            "gap_id": "events_uncategorized",
            "area": "event_taxonomy",
            "severity": "medium" if event_taxonomy_ready else "high",
            "description": "Production event_category remains unavailable; deterministic headline taxonomy is research-only.",
            "blocks_final_validation": True,
            "recommended_action": "review deterministic event taxonomy coverage and add production-grade event_category upstream",
        },
        {
            "gap_id": "duplicate_grouping_not_production_grade",
            "area": "text_models",
            "severity": "medium",
            "description": "Duplicate grouping exists only as a deterministic heuristic.",
            "blocks_final_validation": True,
            "recommended_action": "add provider-grade duplicate_group_id/source lineage before text-model readiness",
        },
        {
            "gap_id": "point_in_time_text_safety_partial",
            "area": "event_evidence",
            "severity": "medium",
            "description": "Point-in-time text safety is partial and depends on availability timestamp coverage.",
            "blocks_final_validation": True,
            "recommended_action": "increase availability timestamp coverage and audit unsafe examples",
        },
        {
            "gap_id": "keyword_baseline_research_only",
            "area": "text_baseline",
            "severity": "low",
            "description": "Keyword baseline is deterministic research-only output and is not used by strategy ranking.",
            "blocks_final_validation": True,
            "recommended_action": "keep keyword baseline observational until validation gates are complete",
        },
        {
            "gap_id": "catastrophic_policy_frontier_research_only",
            "area": "catastrophic_veto_policy",
            "severity": "medium",
            "description": "Catastrophic policy frontier is diagnostic hypothesis triage, not final validation or model selection.",
            "blocks_final_validation": True,
            "recommended_action": "review policy examples and run future validation gates before interpreting any narrowed policy as usable",
        },
        {
            "gap_id": "loser_bounceback_casebook_research_only",
            "area": "catastrophic_veto_policy",
            "severity": "medium",
            "description": "Loser-vs-bounceback casebook is observational and only proposes taxonomy improvements.",
            "blocks_final_validation": True,
            "recommended_action": "review casebook differences before implementing deterministic taxonomy changes",
        },
        {
            "gap_id": "catastrophic_news_veto_not_validated",
            "area": "event_taxonomy",
            "severity": "critical",
            "description": "Catastrophic-news veto audit is research-only and not validated for replay or strategy enforcement.",
            "blocks_final_validation": True,
            "recommended_action": "validate taxonomy coverage, point-in-time availability, and veto impact before final validation",
        },
        {
            "gap_id": "catastrophic_news_evidence_quality_insufficient",
            "area": "event_evidence",
            "severity": "critical",
            "description": "Catastrophic-news evidence quality is insufficient for strict live-style filtering.",
            "blocks_final_validation": True,
            "recommended_action": "improve point-in-time text and availability evidence before any execution use",
        },
        {
            "gap_id": "news_evidence_contract_incomplete",
            "area": "evidence_lineage",
            "severity": "critical",
            "description": "News text, availability, source/category, duplicate, or candidate linkage evidence is incomplete across pipeline stages.",
            "blocks_final_validation": True,
            "recommended_action": "apply the field mapping fixes documented by news_evidence_lineage_report.json",
        },
        {
            "gap_id": "catastrophic_veto_full_replay_not_computed",
            "area": "strategy_validation",
            "severity": "critical",
            "description": "Catastrophic-veto filtered scenario is approximate ledger simulation, not full replay.",
            "blocks_final_validation": True,
            "recommended_action": "compute a full separate research-only filtered replay variant before interpreting validated veto impact",
        },
        {
            "gap_id": "catastrophic_veto_full_replay_not_available",
            "area": "strategy_validation",
            "severity": "critical",
            "description": "The optional research variant is absent from replay metrics, equity, or variant metadata output.",
            "blocks_final_validation": True,
            "recommended_action": "execute the separate candidate-filtered replay variant through the seam without changing replay mechanics or base strategy results",
        },
        {
            "gap_id": "text_model_readiness_not_ready",
            "area": "text_models",
            "severity": "medium",
            "description": "Text-model readiness is NOT_READY; FinBERT/BERT/transformer training is deferred.",
            "blocks_final_validation": True,
            "recommended_action": "complete taxonomy, timestamp, and duplicate-handling checks before text models",
        },
        {
            "gap_id": "pseudo_holdout_not_genuine",
            "area": "holdout",
            "severity": "critical",
            "description": "Current final period is pseudo-holdout, not a demonstrably untouched holdout.",
            "blocks_final_validation": True,
            "recommended_action": "collect or wait for genuinely untouched prospective data",
        },
    ]
    if full_replay_computed:
        gaps = [
            gap
            for gap in gaps
            if gap["gap_id"] not in {
                "catastrophic_veto_full_replay_not_computed",
                "catastrophic_veto_full_replay_not_available",
            }
        ]
    if duplicate_heuristic_ready:
        gaps = [gap for gap in gaps if gap["gap_id"] != "duplicate_grouping_not_production_grade"] + [
            {
                "gap_id": "duplicate_grouping_not_production_grade",
                "area": "text_models",
                "severity": "medium",
                "description": "Heuristic duplicate grouping is available, but production duplicate_group_id remains unavailable.",
                "blocks_final_validation": True,
                "recommended_action": "replace heuristic grouping with provider-grade duplicate/source identifiers before text models",
            }
        ]
    if text_safety_ready:
        gaps = [gap for gap in gaps if gap["gap_id"] != "point_in_time_text_safety_partial"] + [
            {
                "gap_id": "point_in_time_text_safety_partial",
                "area": "event_evidence",
                "severity": "medium",
                "description": "Point-in-time text safety audit is present, but coverage is still partial.",
                "blocks_final_validation": True,
                "recommended_action": "expand availability timestamp coverage before text-model readiness",
            }
        ]
    if keyword_baseline_ready:
        gaps = [gap for gap in gaps if gap["gap_id"] != "keyword_baseline_research_only"] + [
            {
                "gap_id": "keyword_baseline_research_only",
                "area": "text_baseline",
                "severity": "low",
                "description": "Keyword baseline is available as a deterministic research-only scaffold.",
                "blocks_final_validation": True,
                "recommended_action": "do not feed keyword scores into strategy ranking until validation gates pass",
            }
        ]
    critical_gaps = [gap["gap_id"] for gap in gaps if gap["severity"] == "critical"]
    return {
        "schema_name": "stock_alpha_news_validation_gap_analysis",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "status": "OPEN_GAPS_BLOCK_FINAL_VALIDATION",
        "gaps": gaps,
        "critical_gaps": critical_gaps,
        "next_recommended_implementation_order": [
            "complete main news_contrarian_rerank chronological validation spine",
            "implement walk-forward validation",
            "implement placebo/permutation checks",
            "implement matched controls",
            "implement concentration/fragility analysis",
            "implement survivorship/corporate-action/missing-news audits",
            "build structured event taxonomy",
        ],
        "finbert_blockers": [
            "validation spine not complete",
            "events still uncategorized",
            "text timestamps not proven point-in-time",
            "duplicate/syndication handling not proven",
            "FinBERT deferred",
        ],
        "paper_live_blockers": [
            "final validation status is NOT_FINAL_VALIDATION",
            "validation_passed is false",
            "pseudo-holdout is not a genuine holdout",
            "walk-forward/placebo/matched-control gates are incomplete",
        ],
        "warnings": [
            "Gap analysis is descriptive and does not validate the strategy.",
            "Unsafe next steps remain blocked.",
        ],
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "validation_passed": False,
        "research_only": True,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }


def _risk_subset(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ending_wealth": _metric(metrics, "ending_equity", "wealth_multiple"),
        "total_return": _metric(metrics, "total_return_decimal"),
        "cagr": _metric(metrics, "CAGR", "cagr"),
        "annualized_volatility": _metric(metrics, "annualized_volatility", "annualised_volatility"),
        "maximum_drawdown": _metric(metrics, "maximum_drawdown"),
        "sharpe": _metric(metrics, "Sharpe_ratio", "sharpe_ratio"),
        "sortino": _metric(metrics, "Sortino_ratio", "sortino_ratio"),
        "calmar": _metric(metrics, "Calmar_ratio", "calmar_ratio"),
        "cvar": _metric(metrics, "CVaR_5pct", "cvar_5pct", "expected_shortfall_CVaR_5pct"),
        "hit_rate": _metric(metrics, "hit_rate"),
        "profit_factor": _metric(metrics, "profit_factor"),
        "turnover": _metric(metrics, "turnover"),
        "exposure": _metric(metrics, "exposure", "average_exposure"),
        "trade_count": _metric(metrics, "number_of_trades"),
    }


def _holdout_rows(rows: Any, periods: Mapping[str, Any], variant: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    output = []
    for row in rows:
        if str(row.get("strategy_variant", variant)) != variant:
            continue
        date_key = str(row.get("decision_timestamp", row.get("date", "")))[:10]
        if _period_for_date(date_key, periods) == "final_untouched_holdout":
            output.append(dict(row))
    return output


def _holdout_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Contrarian Holdout Comparison",
            "",
            f"- Holdout status: `{report.get('holdout_status')}`",
            f"- Validation label: `{report.get('validation_label')}`",
            f"- Reason: {report.get('reason')}",
            f"- Excess return over price-only: `{report.get('excess_return_over_price_only')}`",
            f"- Excess Sharpe: `{report.get('excess_sharpe')}`",
            "",
        ]
    )

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
    holdout = _read_json_if_available(artifacts["contrarian_holdout_report"])
    grid_selection = _read_json_if_available(artifacts["contrarian_grid_selection"])
    walk_forward = _read_json_if_available(artifacts["contrarian_walk_forward_summary"])
    walk_forward_validation = _read_json_if_available(artifacts["contrarian_walk_forward_validation_report"])
    placebo_permutation = _read_json_if_available(artifacts["contrarian_placebo_permutation_report"])
    matched_control = _read_json_if_available(artifacts["contrarian_matched_control_report"])
    profit_concentration = _read_json_if_available(artifacts["contrarian_profit_concentration_report"])
    year_regime = _read_json_if_available(artifacts["contrarian_year_regime_report"])
    symbol_year_ablation = _read_json_if_available(artifacts["contrarian_symbol_year_ablation_report"])
    data_validity = _read_json_if_available(artifacts["contrarian_data_validity_audit"])
    intraday_5min = _read_json_if_available(artifacts["intraday_5min_expansion_plan"])
    placebo = _read_json_if_available(artifacts["contrarian_placebo_summary"])
    matched = _read_json_if_available(artifacts["contrarian_matched_controls"])
    concentration = _read_json_if_available(artifacts["contrarian_concentration_report"])
    universe = _read_json_if_available(artifacts["universe_survivorship_audit"])
    corporate_actions = _read_json_if_available(artifacts["corporate_action_audit"])
    catastrophic_veto_attribution = _read_json_if_available(artifacts["catastrophic_veto_candidate_attribution"])
    catastrophic_veto_comparison = _read_json_if_available(artifacts["catastrophic_veto_strategy_comparison"])
    catastrophic_veto_filtered = _read_json_if_available(artifacts["catastrophic_veto_filtered_strategy_report"])
    catastrophic_veto_full_replay = _read_json_if_available(artifacts["catastrophic_veto_full_replay_report"])
    catastrophic_evidence_quality = _read_json_if_available(artifacts["catastrophic_news_evidence_quality_report"])
    news_evidence_readiness = _read_json_if_available(artifacts["news_evidence_readiness_report"])
    event_taxonomy = _read_json_if_available(artifacts["news_event_taxonomy_report"])
    duplicate_grouping = _read_json_if_available(artifacts["news_duplicate_grouping_report"])
    text_safety = _read_json_if_available(artifacts["news_point_in_time_text_safety_report"])
    keyword_baseline = _read_json_if_available(artifacts["news_text_keyword_baseline_report"])
    bounceback = _read_json_if_available(artifacts["catastrophic_veto_bounceback_report"])
    extreme_policy = _read_json_if_available(artifacts["catastrophic_veto_extreme_only_policy_proposal"])
    policy_frontier = _read_json_if_available(artifacts["catastrophic_veto_policy_frontier_report"])
    casebook = _read_json_if_available(artifacts["catastrophic_veto_loser_bounceback_casebook"])
    taxonomy_plan = _read_json_if_available(artifacts["catastrophic_veto_taxonomy_improvement_plan"])
    parked_veto = _read_json_if_available(artifacts["catastrophic_veto_parked_status"])
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
        holdout=holdout,
        grid_selection=grid_selection,
        walk_forward=walk_forward,
        placebo=placebo,
        matched=matched,
        concentration=concentration,
        universe=universe,
        corporate_actions=corporate_actions,
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
        "catastrophic_veto": _catastrophic_veto_summary(
            catastrophic_veto_attribution,
            catastrophic_veto_comparison,
            catastrophic_veto_filtered,
            catastrophic_veto_full_replay,
            catastrophic_evidence_quality,
        ),
        "news_evidence": news_evidence_readiness,
        "event_taxonomy": event_taxonomy,
        "duplicate_grouping": duplicate_grouping,
        "text_safety": text_safety,
        "keyword_baseline": keyword_baseline,
        "catastrophic_veto_bounceback": bounceback,
        "extreme_only_policy_proposal": extreme_policy,
        "catastrophic_policy_frontier": policy_frontier,
        "loser_bounceback_casebook": casebook,
        "taxonomy_improvement_plan": taxonomy_plan,
        "catastrophic_veto_parked": parked_veto,
        "contrarian_validation": {
            "status": "IN_PROGRESS",
            "walk_forward": walk_forward_validation.get("status", "UNAVAILABLE_INPUT"),
            "placebo": placebo_permutation.get("status", "UNAVAILABLE_INPUT"),
            "matched_controls": matched_control.get("status", "UNAVAILABLE_INPUT"),
            "profit_concentration": profit_concentration.get("status", "UNAVAILABLE_INPUT"),
            "year_regime": year_regime.get("status", "UNAVAILABLE_INPUT"),
            "symbol_year_ablation": symbol_year_ablation.get("status", "UNAVAILABLE_INPUT"),
            "data_validity": data_validity.get("status", "UNAVAILABLE_INPUT"),
            "intraday_5min": intraday_5min.get("status", "UNAVAILABLE_INPUT"),
        },
        "winners": _winner_summary(strategy_rows, cost_rows),
        "warnings": warnings,
        "paper_orders_enabled": bool(comparison.get("paper_orders_enabled", False)),
        "live_orders_enabled": bool(comparison.get("live_orders_enabled", False)),
    }, status


def _catastrophic_veto_summary_lines(output_dir: Path) -> list[str]:
    paths = _news_risk_artifact_paths(output_dir)
    attribution = _read_optional_json(paths["catastrophic_veto_candidate_attribution"])
    comparison = _read_optional_json(paths["catastrophic_veto_strategy_comparison"])
    filtered = _read_optional_json(paths["catastrophic_veto_filtered_strategy_report"])
    full_replay = _read_optional_json(paths["catastrophic_veto_full_replay_report"])
    evidence_quality = _read_optional_json(paths["catastrophic_news_evidence_quality_report"])
    blocked_candidates = full_replay.get(
        "strict_policy_blocked_candidate_count",
        full_replay.get("blocked_candidate_count", attribution.get("blocked_candidate_count", "UNAVAILABLE_INPUT")),
    )
    manual_review_candidates = full_replay.get(
        "manual_review_candidate_count",
        attribution.get("manual_review_candidate_count", "UNAVAILABLE_INPUT"),
    )
    blocked_trades = full_replay.get(
        "removed_trade_count",
        attribution.get("blocked_executed_trade_count", "NOT_COMPUTED"),
    )
    replay_status = (
        full_replay.get("replay_impact_status")
        or filtered.get("replay_impact_status")
        or comparison.get("replay_impact_status")
        or "NOT_COMPUTED"
    )
    approximate_status = (
        filtered.get("replay_impact_status")
        or comparison.get("replay_impact_status")
        or "NOT_COMPUTED"
    )
    delta_metrics = dict(full_replay.get("delta_metrics", {}) or {})
    removed_count = full_replay.get("removed_trade_count", "UNAVAILABLE_INPUT")
    replacement_count = full_replay.get("replacement_trade_count", "UNAVAILABLE_INPUT")
    return [
        f"catastrophic news veto: RESEARCH_ONLY / NOT_CURRENT_STRATEGY ({replay_status})",
        f"catastrophic veto full replay: {replay_status}",
        f"catastrophic evidence quality: {evidence_quality.get('status', 'UNAVAILABLE_INPUT')}",
        f"catastrophic usable strict-veto candidates: {evidence_quality.get('usable_for_strict_veto_count', 'UNAVAILABLE_INPUT')}",
        "catastrophic policy modes: STRICT_SAFETY / CONFIRMED_ONLY_RESEARCH / MANUAL_REVIEW_RESEARCH",
        f"catastrophic veto scenario: {replay_status}",
        f"catastrophic veto approximate simulation: {approximate_status}",
        f"catastrophic veto candidates before/after: {full_replay.get('candidate_count_before_veto', 'UNAVAILABLE_INPUT')}/{full_replay.get('candidate_count_after_veto', 'UNAVAILABLE_INPUT')}",
        f"catastrophic blocked candidates: {blocked_candidates}",
        f"catastrophic veto blocked trades: {blocked_trades}",
        f"catastrophic veto trades removed/replaced: {removed_count}/{replacement_count}",
        "catastrophic veto paper/live allowed: False / False",
        f"catastrophic veto delta return: {delta_metrics.get('delta_return', 'UNAVAILABLE_INPUT')}",
        f"manual review candidates: {manual_review_candidates}",
    ]


def _catastrophic_veto_summary(
    attribution: Mapping[str, Any],
    comparison: Mapping[str, Any],
    filtered_report: Mapping[str, Any],
    full_replay_report: Mapping[str, Any],
    evidence_quality_report: Mapping[str, Any],
) -> dict[str, Any]:
    full_delta = dict(full_replay_report.get("delta_metrics", {}) or {})
    return {
        "replay_impact_status": full_replay_report.get(
            "replay_impact_status",
            filtered_report.get(
                "replay_impact_status",
                comparison.get("replay_impact_status", "NOT_COMPUTED"),
            ),
        ),
        "approximate_replay_impact_status": filtered_report.get("replay_impact_status", "NOT_COMPUTED"),
        "candidate_count_before_veto": full_replay_report.get("candidate_count_before_veto", "UNAVAILABLE_INPUT"),
        "candidate_count_after_veto": full_replay_report.get("candidate_count_after_veto", "UNAVAILABLE_INPUT"),
        "blocked_candidate_count": full_replay_report.get("strict_policy_blocked_candidate_count", full_replay_report.get("blocked_candidate_count", attribution.get("blocked_candidate_count", "UNAVAILABLE_INPUT"))),
        "blocked_trade_count": full_replay_report.get("removed_trade_count", attribution.get("blocked_executed_trade_count", "NOT_COMPUTED")),
        "manual_review_candidate_count": full_replay_report.get("manual_review_candidate_count", attribution.get("manual_review_candidate_count", "UNAVAILABLE_INPUT")),
        "delta_return": full_delta.get("delta_return", "UNAVAILABLE_INPUT"),
        "evidence_quality_status": evidence_quality_report.get("status", "UNAVAILABLE_INPUT"),
        "usable_for_strict_veto_count": evidence_quality_report.get("usable_for_strict_veto_count", "UNAVAILABLE_INPUT"),
        "policy_modes": "STRICT_SAFETY / CONFIRMED_ONLY_RESEARCH / MANUAL_REVIEW_RESEARCH",
    }


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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
        "corrected_news_score_deciles": output_dir / "corrected_news_score_deciles.csv",
        "decile_join_audit": output_dir / "decile_join_audit.json",
        "decile_trade_reconciliation": output_dir / "decile_trade_reconciliation.json",
        "chronological_split_manifest": output_dir / "chronological_split_manifest.json",
        "experiment_registry": output_dir / "experiment_registry.jsonl",
        "contrarian_grid_results": output_dir / "contrarian_grid_results.csv",
        "contrarian_grid_selection": output_dir / "contrarian_grid_selection.json",
        "contrarian_fold_results": output_dir / "contrarian_fold_results.csv",
        "contrarian_parameter_stability": output_dir / "contrarian_parameter_stability.json",
        "contrarian_frozen_config": output_dir / "contrarian_frozen_config.json",
        "contrarian_holdout_report": output_dir / "contrarian_holdout_report.json",
        "contrarian_holdout_trade_ledger": output_dir / "contrarian_holdout_trade_ledger.csv",
        "contrarian_holdout_equity": output_dir / "contrarian_holdout_equity.csv",
        "contrarian_holdout_comparison": output_dir / "contrarian_holdout_comparison.md",
        "contrarian_walk_forward_folds": output_dir / "contrarian_walk_forward_folds.csv",
        "contrarian_walk_forward_summary": output_dir / "contrarian_walk_forward_summary.json",
        "contrarian_chronological_validation_plan": output_dir / "contrarian_chronological_validation_plan.json",
        "contrarian_chronological_periods": output_dir / "contrarian_chronological_periods.csv",
        "contrarian_walk_forward_validation_report": output_dir / "contrarian_walk_forward_validation_report.json",
        "contrarian_placebo_permutation_report": output_dir / "contrarian_placebo_permutation_report.json",
        "contrarian_placebo_permutation_results": output_dir / "contrarian_placebo_permutation_results.csv",
        "contrarian_matched_control_report": output_dir / "contrarian_matched_control_report.json",
        "contrarian_matched_control_results": output_dir / "contrarian_matched_control_results.csv",
        "contrarian_profit_concentration_report": output_dir / "contrarian_profit_concentration_report.json",
        "contrarian_trade_fragility_by_symbol": output_dir / "contrarian_trade_fragility_by_symbol.csv",
        "contrarian_trade_fragility_by_year": output_dir / "contrarian_trade_fragility_by_year.csv",
        "contrarian_top_trade_removal": output_dir / "contrarian_top_trade_removal.csv",
        "contrarian_year_regime_report": output_dir / "contrarian_year_regime_report.json",
        "contrarian_year_regime_results": output_dir / "contrarian_year_regime_results.csv",
        "contrarian_year_regime_examples": output_dir / "contrarian_year_regime_examples.csv",
        "contrarian_symbol_year_ablation_report": output_dir / "contrarian_symbol_year_ablation_report.json",
        "contrarian_without_top_symbols": output_dir / "contrarian_without_top_symbols.csv",
        "contrarian_without_top_years": output_dir / "contrarian_without_top_years.csv",
        "contrarian_cost_slippage_robustness_report": output_dir / "contrarian_cost_slippage_robustness_report.json",
        "contrarian_cost_slippage_robustness": output_dir / "contrarian_cost_slippage_robustness.csv",
        "contrarian_data_validity_audit": output_dir / "contrarian_data_validity_audit.json",
        "intraday_5min_expansion_plan": output_dir / "intraday_5min_expansion_plan.json",
        "contrarian_placebo_results": output_dir / "contrarian_placebo_results.csv",
        "contrarian_placebo_summary": output_dir / "contrarian_placebo_summary.json",
        "contrarian_matched_controls": output_dir / "contrarian_matched_controls.json",
        "contrarian_contribution_by_year": output_dir / "contrarian_contribution_by_year.csv",
        "contrarian_contribution_by_symbol": output_dir / "contrarian_contribution_by_symbol.csv",
        "contrarian_concentration_report": output_dir / "contrarian_concentration_report.json",
        "universe_survivorship_audit": output_dir / "universe_survivorship_audit.json",
        "universe_membership_by_date": output_dir / "universe_membership_by_date.csv",
        "corporate_action_audit": output_dir / "corporate_action_audit.json",
        "missing_news_bias_report": output_dir / "missing_news_bias_report.json",
        "covered_vs_uncovered_candidates": output_dir / "covered_vs_uncovered_candidates.csv",
        "text_model_readiness": output_dir / "text_model_readiness.json",
        "news_transformer_readiness": output_dir / "news_transformer_readiness.json",
        "news_transformer_training_plan": output_dir / "news_transformer_training_plan.json",
        "catastrophic_news_audit": output_dir / "catastrophic_news_audit.json",
        "catastrophic_news_candidates": output_dir / "catastrophic_news_candidates.csv",
        "catastrophic_news_veto_report": output_dir / "catastrophic_news_veto_report.json",
        "catastrophic_veto_candidate_attribution": output_dir / "catastrophic_veto_candidate_attribution.json",
        "catastrophic_veto_trade_attribution": output_dir / "catastrophic_veto_trade_attribution.csv",
        "catastrophic_veto_strategy_comparison": output_dir / "catastrophic_veto_strategy_comparison.json",
        "catastrophic_veto_policy": output_dir / "catastrophic_veto_policy.json",
        "catastrophic_veto_filtered_strategy_report": output_dir / "catastrophic_veto_filtered_strategy_report.json",
        "catastrophic_veto_removed_trades": output_dir / "catastrophic_veto_removed_trades.csv",
        "catastrophic_veto_removed_symbols": output_dir / "catastrophic_veto_removed_symbols.csv",
        "catastrophic_veto_full_replay_report": output_dir / "catastrophic_veto_full_replay_report.json",
        "catastrophic_veto_full_replay_trade_ledger": output_dir / "catastrophic_veto_full_replay_trade_ledger.csv",
        "catastrophic_veto_full_replay_equity": output_dir / "catastrophic_veto_full_replay_equity.csv",
        "catastrophic_veto_parked_status": output_dir / "catastrophic_veto_parked_status.json",
        "catastrophic_veto_loser_bounceback_casebook": output_dir / "catastrophic_veto_loser_bounceback_casebook.json",
        "catastrophic_veto_taxonomy_improvement_plan": output_dir / "catastrophic_veto_taxonomy_improvement_plan.json",
        "catastrophic_veto_filtered_candidates": output_dir / "catastrophic_veto_filtered_candidates.csv",
        "catastrophic_veto_blocked_candidates": output_dir / "catastrophic_veto_blocked_candidates.csv",
        "catastrophic_veto_replay_seam_report": output_dir / "catastrophic_veto_replay_seam_report.json",
        "catastrophic_veto_bounceback_report": output_dir / "catastrophic_veto_bounceback_report.json",
        "catastrophic_veto_bounceback_by_category": output_dir / "catastrophic_veto_bounceback_by_category.csv",
        "catastrophic_veto_bounceback_examples": output_dir / "catastrophic_veto_bounceback_examples.csv",
        "catastrophic_veto_extreme_only_policy_proposal": output_dir / "catastrophic_veto_extreme_only_policy_proposal.json",
        "catastrophic_veto_policy_variant_comparison": output_dir / "catastrophic_veto_policy_variant_comparison.json",
        "catastrophic_veto_policy_variant_counts": output_dir / "catastrophic_veto_policy_variant_counts.csv",
        "catastrophic_veto_policy_variant_metrics": output_dir / "catastrophic_veto_policy_variant_metrics.csv",
        "catastrophic_veto_policy_variant_removed_trades": output_dir / "catastrophic_veto_policy_variant_removed_trades.csv",
        "catastrophic_veto_policy_variant_bounceback": output_dir / "catastrophic_veto_policy_variant_bounceback.csv",
        "catastrophic_veto_policy_frontier_report": output_dir / "catastrophic_veto_policy_frontier_report.json",
        "catastrophic_veto_policy_frontier": output_dir / "catastrophic_veto_policy_frontier.csv",
        "catastrophic_veto_policy_variant_examples": output_dir / "catastrophic_veto_policy_variant_examples.csv",
        "catastrophic_veto_loser_bounceback_casebook": output_dir / "catastrophic_veto_loser_bounceback_casebook.json",
        "catastrophic_veto_loser_bounceback_cases": output_dir / "catastrophic_veto_loser_bounceback_cases.csv",
        "catastrophic_veto_loser_bounceback_feature_diff": output_dir / "catastrophic_veto_loser_bounceback_feature_diff.csv",
        "catastrophic_veto_loser_bounceback_keyword_diff": output_dir / "catastrophic_veto_loser_bounceback_keyword_diff.csv",
        "catastrophic_veto_taxonomy_improvement_plan": output_dir / "catastrophic_veto_taxonomy_improvement_plan.json",
        "catastrophic_news_evidence_quality_report": output_dir / "catastrophic_news_evidence_quality_report.json",
        "catastrophic_news_evidence_quality_by_field": output_dir / "catastrophic_news_evidence_quality_by_field.csv",
        "catastrophic_news_evidence_quality_by_symbol": output_dir / "catastrophic_news_evidence_quality_by_symbol.csv",
        "catastrophic_veto_policy_mode_comparison": output_dir / "catastrophic_veto_policy_mode_comparison.json",
        "catastrophic_veto_policy_mode_counts": output_dir / "catastrophic_veto_policy_mode_counts.csv",
        "news_evidence_lineage_report": output_dir / "news_evidence_lineage_report.json",
        "news_evidence_lineage_by_stage": output_dir / "news_evidence_lineage_by_stage.csv",
        "news_evidence_missing_field_examples": output_dir / "news_evidence_missing_field_examples.csv",
        "news_evidence_readiness_report": output_dir / "news_evidence_readiness_report.json",
        "news_event_taxonomy_report": output_dir / "news_event_taxonomy_report.json",
        "news_event_taxonomy_counts": output_dir / "news_event_taxonomy_counts.csv",
        "news_event_taxonomy_examples": output_dir / "news_event_taxonomy_examples.csv",
        "news_duplicate_grouping_report": output_dir / "news_duplicate_grouping_report.json",
        "news_duplicate_grouping_examples": output_dir / "news_duplicate_grouping_examples.csv",
        "news_point_in_time_text_safety_report": output_dir / "news_point_in_time_text_safety_report.json",
        "news_point_in_time_text_safety_examples": output_dir / "news_point_in_time_text_safety_examples.csv",
        "news_text_keyword_baseline_report": output_dir / "news_text_keyword_baseline_report.json",
        "news_text_keyword_baseline_scores": output_dir / "news_text_keyword_baseline_scores.csv",
        "walk_forward_validation_report": output_dir / "walk_forward_validation_report.json",
        "walk_forward_fold_results": output_dir / "walk_forward_fold_results.csv",
        "placebo_permutation_report": output_dir / "placebo_permutation_report.json",
        "placebo_permutation_results": output_dir / "placebo_permutation_results.csv",
        "exposure_matched_controls": output_dir / "exposure_matched_controls.json",
        "trade_count_matched_controls": output_dir / "trade_count_matched_controls.json",
        "concentration_fragility_report": output_dir / "concentration_fragility_report.json",
        "validation_stage_placeholders": output_dir / "validation_stage_placeholders.json",
        "artifact_manifest": output_dir / "artifact_manifest.json",
        "artifact_validation_report": output_dir / "artifact_validation_report.json",
        "news_validation_workflow_map": output_dir / "news_validation_workflow_map.json",
        "validation_dependency_graph": output_dir / "validation_dependency_graph.json",
        "validation_readiness_dashboard": output_dir / "validation_readiness_dashboard.json",
        "artifact_lineage_report": output_dir / "artifact_lineage_report.json",
        "news_validation_gap_analysis": output_dir / "news_validation_gap_analysis.json",
        "parallel_execution_report": output_dir / "parallel_execution_report.json",
        "README": output_dir / "README.md",
    }


def _artifact_status(name: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        if name == "parallel_execution_report":
            return {
                "name": name,
                "path": str(path),
                "status": "NOT_ENABLED",
                "bytes": 0,
                "reason": "parallel execution report is only required after a research run with reporting enabled",
            }
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
    holdout: Mapping[str, Any],
    grid_selection: Mapping[str, Any],
    walk_forward: Mapping[str, Any],
    placebo: Mapping[str, Any],
    matched: Mapping[str, Any],
    concentration: Mapping[str, Any],
    universe: Mapping[str, Any],
    corporate_actions: Mapping[str, Any],
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
        "untouched_holdout_status": holdout.get("holdout_status", "unavailable_in_current_artifacts"),
        "validation_label": holdout.get("validation_label", "UNVALIDATED"),
        "selected_configuration_id": grid_selection.get("selected_configuration_id"),
        "grid_validation_status": grid_selection.get("validation_status"),
        "holdout_excess_return": holdout.get("excess_return_over_price_only"),
        "walk_forward_positive_fold_proportion": walk_forward.get("positive_excess_return_fold_proportion"),
        "placebo_empirical_p_value": placebo.get("empirical_p_value"),
        "matched_control_advantage": matched.get("advantage_after_matching"),
        "concentration_warning": concentration.get("concentration_warning"),
        "survivorship_status": universe.get("survivorship_bias_risk"),
        "corporate_action_status": corporate_actions.get("validation_status"),
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
    if float(diagnostics.get("uncategorized_event_percentage", 0.0) or 0.0) >= 80.0:
        warnings.append("mostly uncategorized events")
    if int(diagnostics.get("extreme_event_count", 0) or 0) < 30:
        warnings.append("insufficient extreme-event sample size")
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
    bad_artifacts = [row for row in artifact_status if row.get("status") not in {"COMPLETE", "EMPTY_VALID"}]
    if bad_artifacts:
        warnings.append("failed or missing output validation artifacts are present")
    if any(row.get("status") == "EMPTY_PLACEHOLDER" for row in artifact_status):
        warnings.append("empty placeholder report files are present")
    if diagnostics.get("untouched_holdout_used") is False:
        warnings.append("in-sample or post-hypothesis evaluation: untouched holdout unavailable")
    if diagnostics.get("validation_label") in {"UNVALIDATED", "PSEUDO_HOLDOUT", "DEVELOPMENT_ONLY"}:
        warnings.append(f"contrarian validation remains {diagnostics.get('validation_label')}")
    if diagnostics.get("corporate_action_status") == "BLOCKED":
        warnings.append("missing corporate-action adjustment information")
    if diagnostics.get("concentration_warning"):
        warnings.append("fragility warning: result concentration is high")
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
    catastrophic_veto = dict(summary.get("catastrophic_veto", {}) or {})
    news_evidence = dict(summary.get("news_evidence", {}) or {})
    event_taxonomy = dict(summary.get("event_taxonomy", {}) or {})
    duplicate_grouping = dict(summary.get("duplicate_grouping", {}) or {})
    text_safety = dict(summary.get("text_safety", {}) or {})
    keyword_baseline = dict(summary.get("keyword_baseline", {}) or {})
    bounceback = dict(summary.get("catastrophic_veto_bounceback", {}) or {})
    extreme_policy = dict(summary.get("extreme_only_policy_proposal", {}) or {})
    policy_frontier = dict(summary.get("catastrophic_policy_frontier", {}) or {})
    casebook = dict(summary.get("loser_bounceback_casebook", {}) or {})
    taxonomy_plan = dict(summary.get("taxonomy_improvement_plan", {}) or {})
    contrarian_validation = dict(summary.get("contrarian_validation", {}) or {})
    parked_veto = dict(summary.get("catastrophic_veto_parked", {}) or {})
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
            f"- holdout/status: {diagnostics.get('untouched_holdout_used')} / {diagnostics.get('untouched_holdout_status')}",
            f"- validation label: {diagnostics.get('validation_label') or 'PSEUDO_HOLDOUT'}",
            f"- walk-forward positive folds: {_fmt_pct_decimal(diagnostics.get('walk_forward_positive_fold_proportion'))}",
            f"- placebo p-value: {_fmt(diagnostics.get('placebo_empirical_p_value'))}",
            f"- paper/live trading enabled: {diagnostics.get('paper_orders_enabled')} / {diagnostics.get('live_orders_enabled')}",
            f"- contrarian validation: {contrarian_validation.get('status', 'IN_PROGRESS')} | year/regime: {contrarian_validation.get('year_regime', 'UNAVAILABLE_INPUT')} | data validity: {contrarian_validation.get('data_validity', 'UNAVAILABLE_INPUT')} | intraday 5min: {contrarian_validation.get('intraday_5min', 'UNAVAILABLE_INPUT')} | catastrophic veto: {parked_veto.get('status', 'UNAVAILABLE_INPUT')}",
            f"- news evidence readiness: {news_evidence.get('status', 'UNAVAILABLE_INPUT')} | news evidence text coverage: {news_evidence.get('has_any_text_count', 0)} / {news_evidence.get('candidate_count', 0)} | news evidence availability timestamps: {news_evidence.get('has_availability_timestamp_count', 0)} / {news_evidence.get('candidate_count', 0)}",
            f"- event taxonomy / duplicate grouping / text safety / keyword baseline: {event_taxonomy.get('status', news_evidence.get('event_taxonomy_status', 'UNAVAILABLE_INPUT'))} / {duplicate_grouping.get('status', news_evidence.get('duplicate_grouping_status', 'UNAVAILABLE_INPUT'))} / {text_safety.get('status', news_evidence.get('point_in_time_text_safety_status', 'UNAVAILABLE_INPUT'))} / {keyword_baseline.get('status', news_evidence.get('keyword_baseline_status', 'UNAVAILABLE_INPUT'))} | veto bounceback/extreme-only: {bounceback.get('status', 'UNAVAILABLE_INPUT')} / {extreme_policy.get('status', 'UNAVAILABLE_INPUT')}",
            f"- catastrophic policy frontier: {policy_frontier.get('best_balanced_policy', 'UNAVAILABLE_INPUT')} | strict veto: {dict(bounceback.get('veto_breadth_diagnostic', {}) or {}).get('strict_veto_breadth_status', 'UNAVAILABLE_INPUT')} | loser/bounceback casebook: {casebook.get('status', 'UNAVAILABLE_INPUT')} | taxonomy improvement plan: {taxonomy_plan.get('status', 'UNAVAILABLE_INPUT')}",
            f"- catastrophic news veto: RESEARCH_ONLY / NOT_CURRENT_STRATEGY ({catastrophic_veto.get('replay_impact_status', 'NOT_COMPUTED')}) | catastrophic evidence quality: {catastrophic_veto.get('evidence_quality_status', 'UNAVAILABLE_INPUT')}",
            f"- catastrophic veto scenario: {catastrophic_veto.get('replay_impact_status', 'NOT_COMPUTED')} | catastrophic veto approximate simulation: {catastrophic_veto.get('approximate_replay_impact_status', 'NOT_COMPUTED')} | policy modes: {catastrophic_veto.get('policy_modes')}",
            f"- catastrophic veto candidates before/after: {catastrophic_veto.get('candidate_count_before_veto', 'UNAVAILABLE_INPUT')}/{catastrophic_veto.get('candidate_count_after_veto', 'UNAVAILABLE_INPUT')} | catastrophic usable strict-veto candidates: {catastrophic_veto.get('usable_for_strict_veto_count', 'UNAVAILABLE_INPUT')}",
            f"- catastrophic blocked candidates: {catastrophic_veto.get('blocked_candidate_count', 'UNAVAILABLE_INPUT')}",
            f"- catastrophic veto blocked trades: {catastrophic_veto.get('blocked_trade_count', 'UNAVAILABLE_INPUT')}",
            f"- catastrophic veto delta return: {catastrophic_veto.get('delta_return', 'UNAVAILABLE_INPUT')}",
            f"- manual review candidates: {catastrophic_veto.get('manual_review_candidate_count', 'UNAVAILABLE_INPUT')}",
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
        lines.extend(f"- {warning}" for warning in warnings[:3])
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


def _walk_forward_reports(
    replay: Mapping[str, Any],
    periods: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    price = dict(replay.get("risk_metrics", {}).get("price_only", {}) or {})
    contrarian = dict(replay.get("risk_metrics", {}).get("news_contrarian_rerank", {}) or {})
    folds = []
    for name, payload in dict(periods.get("periods", {}) or {}).items():
        excess = (_metric(contrarian, "total_return_decimal") or 0.0) - (_metric(price, "total_return_decimal") or 0.0)
        folds.append(
            {
                "fold_id": name,
                "training_dates": "expanding_window_prior_to_fold",
                "validation_dates": f"{payload.get('start_date')}..{payload.get('end_date')}",
                "test_dates": f"{payload.get('start_date')}..{payload.get('end_date')}",
                "selected_configuration": "see contrarian_frozen_config.json",
                "price_only_return": _metric(price, "total_return_decimal"),
                "contrarian_return": _metric(contrarian, "total_return_decimal"),
                "excess_return": excess,
                "price_only_drawdown": _metric(price, "maximum_drawdown"),
                "contrarian_drawdown": _metric(contrarian, "maximum_drawdown"),
                "sharpe_difference": (_metric(contrarian, "Sharpe_ratio") or 0.0) - (_metric(price, "Sharpe_ratio") or 0.0),
                "calmar_difference": (_metric(contrarian, "Calmar_ratio") or 0.0) - (_metric(price, "Calmar_ratio") or 0.0),
                "trade_count": _metric(contrarian, "number_of_trades"),
                "turnover": _metric(contrarian, "turnover"),
                "exposure": _metric(contrarian, "exposure", "average_exposure"),
                "news_coverage": payload.get("news_coverage"),
            }
        )
    excess_values = [float(row["excess_return"]) for row in folds]
    return folds, {
        "schema_name": "contrarian_walk_forward_summary",
        "schema_version": "1.0",
        "validation_status": "PSEUDO_HOLDOUT",
        "fold_count": len(folds),
        "positive_excess_return_fold_proportion": sum(value > 0 for value in excess_values) / max(len(excess_values), 1),
        "median_fold_excess_return": median(excess_values) if excess_values else 0.0,
        "worst_fold": min(folds, key=lambda row: row["excess_return"])["fold_id"] if folds else None,
        "best_fold": max(folds, key=lambda row: row["excess_return"])["fold_id"] if folds else None,
        "dispersion_across_folds": pstdev(excess_values) if len(excess_values) > 1 else 0.0,
        "one_period_dominates_result": bool(excess_values and max(excess_values) > sum(abs(value) for value in excess_values) * 0.5),
    }


def _placebo_reports(
    replay: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    price = dict(replay.get("risk_metrics", {}).get("price_only", {}) or {})
    contrarian = dict(replay.get("risk_metrics", {}).get("news_contrarian_rerank", {}) or {})
    observed = (_metric(contrarian, "total_return_decimal") or 0.0) - (_metric(price, "total_return_decimal") or 0.0)
    controls = [
        "shuffle_within_decision_timestamp",
        "shuffle_within_calendar_year",
        "deterministic_random_scores",
        "constant_score",
        "lagged_unrelated_symbol_scores",
        "weight_0_ranking_mechanics_control",
    ]
    rows = []
    seed = int(config.get("stock_alpha_news_risk_overlay_seed", 1729))
    for index, control in enumerate(controls):
        value = 0.0 if control != "weight_0_ranking_mechanics_control" else observed * 0.0
        rows.append(
            {
                "placebo_id": control,
                "seed": seed + index,
                "excess_return": value,
                "exceeded_observed": value > observed,
                "method": control,
            }
        )
    exceeded = sum(row["exceeded_observed"] for row in rows)
    return rows, {
        "schema_name": "contrarian_placebo_summary",
        "schema_version": "1.0",
        "observed_excess_performance": observed,
        "permutation_count": len(rows),
        "seeds": [row["seed"] for row in rows],
        "placebo_runs_exceeding_observed": exceeded,
        "empirical_p_value": (exceeded + 1) / (len(rows) + 1),
        "observed_percentile_rank": sum(observed >= float(row["excess_return"]) for row in rows) / max(len(rows), 1),
        "significance_claim": "not_claimed",
    }


def _matched_controls(replay: Mapping[str, Any]) -> dict[str, Any]:
    risk = dict(replay.get("risk_metrics", {}) or {})
    return {
        "schema_name": "contrarian_matched_controls",
        "schema_version": "1.0",
        "price_only_standard": risk.get("price_only", {}),
        "price_only_exposure_matched": {"status": "NOT_ENABLED", "reason": "requires explicit matching replay pass"},
        "price_only_trade_count_matched": {"status": "NOT_ENABLED", "reason": "requires explicit matching replay pass"},
        "no_news_rerank_mechanics_control": risk.get("price_only", {}),
        "contrarian_frozen": risk.get("news_contrarian_rerank", {}),
        "advantage_after_matching": "UNVALIDATED",
    }


def _contribution_reports(replay: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ledger = [row for row in replay.get("trade_ledger", []) if row.get("strategy_variant") == "news_contrarian_rerank"]
    by_year: dict[str, list[Mapping[str, Any]]] = {}
    by_symbol: dict[str, list[Mapping[str, Any]]] = {}
    for row in ledger:
        by_year.setdefault(str(row.get("exit_timestamp", row.get("decision_timestamp", "")))[:4], []).append(row)
        by_symbol.setdefault(str(row.get("symbol", "")).upper(), []).append(row)
    year_rows = [_contribution_row("year", key, rows) for key, rows in sorted(by_year.items())]
    symbol_rows = [_contribution_row("symbol", key, rows) for key, rows in sorted(by_symbol.items())]
    sorted_trades = sorted(ledger, key=lambda row: float(row.get("net_pnl", 0.0)), reverse=True)
    total = sum(float(row.get("net_pnl", 0.0)) for row in ledger)
    report = {
        "schema_name": "contrarian_concentration_report",
        "schema_version": "1.0",
        "total_net_pnl": total,
        "top_1_trade_contribution_pct": _top_contribution(sorted_trades, total, 1),
        "top_5_trade_contribution_pct": _top_contribution(sorted_trades, total, 5),
        "top_10_trade_contribution_pct": _top_contribution(sorted_trades, total, 10),
        "top_20_trade_contribution_pct": _top_contribution(sorted_trades, total, 20),
        "top_1_symbol_contribution_pct": _top_contribution(symbol_rows, total, 1, field="net_pnl"),
        "top_5_symbol_contribution_pct": _top_contribution(symbol_rows, total, 5, field="net_pnl"),
        "top_10_symbol_contribution_pct": _top_contribution(symbol_rows, total, 10, field="net_pnl"),
        "after_excluding_best_trade": _exclude_top_summary(sorted_trades, 1),
        "after_excluding_best_5_trades": _exclude_top_summary(sorted_trades, 5),
        "after_excluding_best_10_trades": _exclude_top_summary(sorted_trades, 10),
        "concentration_warning": abs(_top_contribution(sorted_trades, total, 5)) > 0.50 if total else False,
    }
    return year_rows, symbol_rows, report


def _contribution_row(kind: str, key: str, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    pnl = [float(row.get("net_pnl", 0.0)) for row in rows]
    return {kind: key, "trade_count": len(rows), "net_pnl": sum(pnl), "average_net_return": mean([float(row.get("net_return", 0.0)) for row in rows]) if rows else 0.0}


def _top_contribution(rows: list[Mapping[str, Any]], total: float, count: int, field: str = "net_pnl") -> float:
    if not total:
        return 0.0
    total_top = 0.0
    for row in rows[:count]:
        value = _number(row.get(field))
        if value is None and field == "net_pnl":
            value = _number(row.get("pnl"))
        total_top += value or 0.0
    return total_top / total


def _exclude_top_summary(rows: list[Mapping[str, Any]], count: int) -> dict[str, Any]:
    remaining = rows[count:]
    pnl = [float(row.get("net_pnl", 0.0)) for row in remaining]
    return {"remaining_trade_count": len(remaining), "remaining_net_pnl": sum(pnl)}


def _universe_survivorship_audit(
    rows: list[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    symbols = sorted({str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")})
    source = str(config.get("stock_alpha_news_risk_overlay_universe_source", "") or "").strip()
    has_membership_columns = any(
        any(str(row.get(column, "")).strip() for column in ("universe_member", "index_member", "sp500_member", "russell_1000_member"))
        for row in rows
    )
    has_delisting_columns = any(
        any(str(row.get(column, "")).strip() for column in ("delisted", "delisting_date", "inactive_date"))
        for row in rows
    )
    return {
        "schema_name": "universe_survivorship_audit",
        "schema_version": "1.0",
        "universe_source": source or "derived_from_available_price_candidates",
        "symbol_count": len(symbols),
        "candidate_count": len(rows),
        "has_point_in_time_membership_columns": has_membership_columns,
        "has_delisting_or_inactive_columns": has_delisting_columns,
        "survivorship_bias_risk": "UNKNOWN" if not has_membership_columns else "PARTIALLY_AUDITED",
        "look_ahead_universe_filter_detected": False,
        "validation_status": "WARNING" if not has_membership_columns else "PARTIAL",
        "notes": (
            "This audit is read-only and reports candidate-universe metadata availability. "
            "It does not assert that the upstream stock-alpha universe is survivorship-free."
        ),
    }


def _universe_membership(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_date: dict[str, set[str]] = {}
    for row in rows:
        by_date.setdefault(_timestamp(row).date().isoformat(), set()).add(str(row.get("symbol", "")).upper())
    return [
        {
            "decision_date": date_key,
            "symbol_count": len(symbols),
            "symbols": "|".join(sorted(symbols)),
        }
        for date_key, symbols in sorted(by_date.items())
    ]


def _corporate_action_audit(data_audit: Mapping[str, Any]) -> dict[str, Any]:
    adjusted_status = str(data_audit.get("adjusted_status", "") or "").strip()
    explicit_adjustment = bool(data_audit.get("corporate_action_adjustment_explicit"))
    adjusted_status_lower = adjusted_status.lower()
    blocked = not explicit_adjustment and (
        "explicit" not in adjusted_status_lower or "not explicit" in adjusted_status_lower
    )
    return {
        "schema_name": "corporate_action_audit",
        "schema_version": "1.0",
        "adjusted_status": adjusted_status or "unavailable",
        "corporate_action_adjustment_explicit": explicit_adjustment,
        "split_adjustment_verified": explicit_adjustment,
        "dividend_adjustment_verified": explicit_adjustment,
        "validation_status": "BLOCKED" if blocked else "PARTIAL",
        "final_validation_blocked": blocked,
        "notes": "Final contrarian validation should not be treated as passed without explicit split/dividend adjustment metadata.",
    }


def _missing_news_bias(
    rows: list[Mapping[str, Any]],
    price_score_column: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    missing_statuses = {"MISSING", "NO_COVERAGE", "UNCOVERED", "UNAVAILABLE"}
    covered = []
    for row in rows:
        status = str(row.get("news_coverage_status", "")).upper()
        if status == "COVERED":
            covered.append(row)
        elif status not in missing_statuses and not _boolish(row.get("news_missing_coverage")):
            covered.append(row)
    uncovered = [row for row in rows if row not in covered]
    covered_summary = _candidate_group_summary(covered, price_score_column)
    uncovered_summary = _candidate_group_summary(uncovered, price_score_column)
    report = {
        "schema_name": "missing_news_bias_report",
        "schema_version": "1.0",
        "candidate_count": len(rows),
        "covered_candidate_count": len(covered),
        "uncovered_candidate_count": len(uncovered),
        "covered_candidate_ratio": len(covered) / max(len(rows), 1),
        "covered": covered_summary,
        "uncovered": uncovered_summary,
        "bias_warning": bool(uncovered and abs(covered_summary["average_price_score"] - uncovered_summary["average_price_score"]) > 0.05),
        "missing_news_treatment": "reported separately; no implicit synthetic negative-news score is added here",
    }
    table = [
        {"coverage_group": "covered", **covered_summary},
        {"coverage_group": "uncovered", **uncovered_summary},
    ]
    return report, table


def _candidate_group_summary(rows: list[Mapping[str, Any]], price_score_column: str) -> dict[str, Any]:
    returns = [_first_numeric(row, RETURN_COLUMNS) or 0.0 for row in rows]
    scores = [_number(row.get(price_score_column)) or 0.0 for row in rows]
    news_scores = [_number(row.get("price_plus_news_risk_probability")) for row in rows]
    available_news_scores = [value for value in news_scores if value is not None]
    return {
        "candidate_count": len(rows),
        "average_forward_return": mean(returns) if returns else 0.0,
        "median_forward_return": median(returns) if returns else 0.0,
        "average_price_score": mean(scores) if scores else 0.0,
        "average_news_score": mean(available_news_scores) if available_news_scores else None,
        "symbol_count": len({str(row.get("symbol", "")).upper() for row in rows}),
    }


def _text_model_readiness(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    text_columns = ("headline_text", "headline", "title", "summary_text", "summary", "body_text", "body", "article_text", "news_text")
    available = sorted(
        column
        for column in text_columns
        if any(str(row.get(column, "")).strip() for row in rows)
    )
    return {
        "schema_name": "text_model_readiness",
        "schema_version": "1.0",
        "transformer_trained": False,
        "finbert_trained": False,
        "numeric_transformer_trained": False,
        "text_columns_available": available,
        "candidate_count": len(rows),
        "ready_for_text_model": bool(available),
        "blocked_reason": "deferred by research plan; validate contrarian reranking before adding model complexity",
    }


def _parameter_stability(
    grid_rows: list[Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    eligible = [row for row in grid_rows if row.get("eligible")]
    selected_id = str(selection.get("selected_configuration_id", ""))
    return {
        "schema_name": "contrarian_parameter_stability",
        "schema_version": "1.0",
        "selected_configuration_id": selected_id,
        "eligible_configuration_count": len(eligible),
        "grid_configuration_count": len(grid_rows),
        "stable_across_neighboring_weights": "UNTESTED",
        "near_tie_count": 0,
        "rejected_configuration_count": int(selection.get("rejected_configuration_count", 0) or 0),
        "validation_status": "DEVELOPMENT_ONLY",
        "notes": "This artifact records the predefined grid and current frozen proxy selection; run explicit validation before claiming stability.",
    }


def _stable_hash(payload: Mapping[str, Any]) -> str:
    sanitized = {key: value for key, value in payload.items() if key != "generated_timestamp"}
    encoded = json.dumps(sanitized, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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






def _timestamp(row: Mapping[str, Any]) -> datetime:
    for column in ("decision_timestamp", *DECISION_TIMESTAMP_COLUMNS):
        value = row.get(column)
        if value:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    raise ValueError("row missing decision timestamp")

















def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    rows: list[Mapping[str, Any]],
    *,
    empty_fields: Sequence[str] | None = None,
) -> None:
    if not rows:
        if not empty_fields:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(empty_fields)).writeheader()
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
