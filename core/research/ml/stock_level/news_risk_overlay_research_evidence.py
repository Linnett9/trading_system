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

from core.research.ml.stock_level.news_risk_overlay_research_audit import (
    _category_mix,
    _risk_subset,
)
from core.research.ml.stock_level.news_risk_overlay_research_utils import (
    _number,
    _read_csv,
    _timestamp,
)

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


__all__ = [name for name in globals() if not name.startswith("__")]
