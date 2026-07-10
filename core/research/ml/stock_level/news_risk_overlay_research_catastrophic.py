from __future__ import annotations

import math
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from statistics import mean, median
from typing import Any, Callable, Mapping, Sequence

from core.research.ml.stock_level.news_sources import (
    catastrophic_news_taxonomy_report,
    classify_catastrophic_news_rows,
)

RETURN_COLUMNS = (
    "actual_forward_return_10d",
    "actual_forward_return_5d",
    "forward_return",
)


def _catastrophic_news_artifacts(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    classifications = classify_catastrophic_news_rows(rows)
    blocked_candidate_ids = {
        row["candidate_id"]
        for row in classifications
        if row.get("blocks_contrarian_entry") and row.get("candidate_id") != "UNKNOWN"
    }
    manual_review_candidate_ids = {
        row["candidate_id"]
        for row in classifications
        if row.get("requires_manual_review") and row.get("candidate_id") != "UNKNOWN"
    }
    unknown_or_unavailable_count = sum(
        1
        for row in classifications
        if row.get("highest_severity") == "UNKNOWN"
        or row.get("classification_method") == "UNAVAILABLE_INPUT"
    )
    point_in_time_safe_count = sum(1 for row in classifications if row.get("point_in_time_safe"))
    point_in_time_unsafe_count = len(classifications) - point_in_time_safe_count
    warnings = [
        "Research-only catastrophic-news layer; not enforced in replay, strategy, paper trading, or live trading.",
    ]
    if unknown_or_unavailable_count:
        warnings.append("UNAVAILABLE_INPUT rows require manual review and are not classified as safe.")
    if point_in_time_unsafe_count:
        warnings.append("Availability timestamp is missing for at least one row; point-in-time safety is not established.")

    status = "UNAVAILABLE_INPUT" if not classifications else "PASSED_WITH_WARNINGS"
    audit = {
        "schema": "catastrophic_news_audit_v1",
        "status": status,
        "research_only": True,
        "taxonomy": catastrophic_news_taxonomy_report(),
        "categories": catastrophic_news_taxonomy_report()["categories"],
        "candidate_count": len(rows),
        "news_event_count": len(classifications),
        "matched_event_count": sum(1 for row in classifications if row.get("matched")),
        "catastrophic_event_count": sum(
            1 for row in classifications if row.get("highest_severity") == "CATASTROPHIC"
        ),
        "blocked_candidate_count": len(blocked_candidate_ids),
        "manual_review_candidate_count": len(manual_review_candidate_ids),
        "unknown_or_unavailable_count": unknown_or_unavailable_count,
        "point_in_time_safe_count": point_in_time_safe_count,
        "point_in_time_unsafe_count": point_in_time_unsafe_count,
        "blocked_candidate_ids": sorted(blocked_candidate_ids),
        "manual_review_candidate_ids": sorted(manual_review_candidate_ids),
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "warnings": warnings,
    }
    candidate_rows = [
        {
            "candidate_id": row.get("candidate_id", "UNKNOWN"),
            "symbol": row.get("symbol", "UNKNOWN"),
            "matched": row.get("matched", False),
            "matched_categories": "|".join(row.get("matched_categories", [])),
            "highest_severity": row.get("highest_severity", "UNKNOWN"),
            "blocks_contrarian_entry": row.get("blocks_contrarian_entry", False),
            "requires_manual_review": row.get("requires_manual_review", False),
            "matched_terms": "|".join(row.get("matched_terms", [])),
            "matched_patterns": "|".join(row.get("matched_patterns", [])),
            "classification_method": row.get("classification_method", "UNKNOWN"),
            "availability_timestamp_present": row.get("availability_timestamp_present", False),
            "point_in_time_safe": row.get("point_in_time_safe", False),
            "source": row.get("source", "UNKNOWN"),
            "publication_timestamp": row.get("publication_timestamp", "UNKNOWN"),
            "availability_timestamp": row.get("availability_timestamp", "UNKNOWN"),
            "warnings": "|".join(row.get("warnings", [])),
            "research_only_veto_would_apply": row.get("blocks_contrarian_entry", False),
        }
        for row in classifications
    ]
    veto_report = {
        "schema": "catastrophic_news_veto_report_v1",
        "status": status,
        "research_only": True,
        "veto_enabled_in_strategy": False,
        "used_in_replay": False,
        "training_enabled": False,
        "inference_enabled": False,
        "would_block_candidate_count": len(blocked_candidate_ids),
        "would_require_manual_review_count": len(manual_review_candidate_ids),
        "blocked_candidate_ids": sorted(blocked_candidate_ids),
        "manual_review_candidate_ids": sorted(manual_review_candidate_ids),
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "warnings": warnings,
    }
    return audit, candidate_rows, veto_report


def _catastrophic_veto_policy() -> dict[str, Any]:
    taxonomy = catastrophic_news_taxonomy_report()
    categories = list(taxonomy["categories"])
    return {
        "schema_name": "catastrophic_veto_policy",
        "schema_version": 1,
        "policy_name": "research_only_catastrophic_news_contrarian_veto",
        "policy_version": "v1",
        "policy_stage": "RESEARCH_ONLY",
        "catastrophic_veto_policy_mode": "STRICT_SAFETY",
        "allowed_policy_modes": ["STRICT_SAFETY", "CONFIRMED_ONLY_RESEARCH", "MANUAL_REVIEW_RESEARCH"],
        "enforcement_stage": "AUDIT_OR_RESEARCH_SIMULATION_ONLY",
        "enabled_for_research": True,
        "enabled_for_paper_trading": False,
        "enabled_for_live_trading": False,
        "paper_trading_allowed": False,
        "live_trading_allowed": False,
        "manual_review_required_before_any_execution": True,
        "unknown_text_default": "DO_NOT_TREAT_AS_SAFE",
        "missing_availability_timestamp_default": "NOT_POINT_IN_TIME_SAFE",
        "default_action_for_catastrophic": "BLOCK_CONTRARIAN_ENTRY",
        "default_action_for_manual_review": "BLOCK_UNTIL_REVIEWED",
        "default_action_for_unknown": "DO_NOT_TREAT_AS_SAFE",
        "missing_availability_timestamp": "NOT_POINT_IN_TIME_SAFE",
        "categories": categories,
        "manual_review_required_categories": [
            category["category_id"]
            for category in categories
            if category.get("requires_manual_review")
        ],
        "point_in_time_requirements": {
            "availability_timestamp_required": True,
            "missing_availability_timestamp": "NOT_POINT_IN_TIME_SAFE",
            "missing_text": "UNAVAILABLE_INPUT",
        },
        "warnings": [
            "Research-only policy; not enforced in current replay, paper trading, or live trading.",
            "UNKNOWN and manual-review rows are not treated as safe.",
        ],
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
    }


def apply_catastrophic_veto_to_candidates(
    candidate_rows: Sequence[Mapping[str, Any]],
    classification_rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    policy_mode: str = "STRICT_SAFETY",
) -> dict[str, Any]:
    if policy_mode not in {"STRICT_SAFETY", "CONFIRMED_ONLY_RESEARCH", "MANUAL_REVIEW_RESEARCH"}:
        raise ValueError(f"unknown catastrophic veto policy mode: {policy_mode}")
    classifications = (
        [dict(row) for row in classification_rows]
        if classification_rows is not None
        else classify_catastrophic_news_rows(candidate_rows)
    )
    by_candidate_id = {
        str(row.get("candidate_id")): row
        for row in classifications
        if row.get("candidate_id") not in {None, "", "UNKNOWN"}
    }
    by_symbol: dict[str, list[Mapping[str, Any]]] = {}
    for row in classifications:
        symbol = str(row.get("symbol", "UNKNOWN"))
        if symbol != "UNKNOWN":
            by_symbol.setdefault(symbol, []).append(row)

    filtered_candidates: list[dict[str, Any]] = []
    blocked_candidates: list[dict[str, Any]] = []
    manual_review_candidates: list[dict[str, Any]] = []
    unknown_candidates: list[dict[str, Any]] = []
    confirmed_catastrophic_candidates: list[dict[str, Any]] = []
    unknown_text_candidates: list[dict[str, Any]] = []
    missing_availability_candidates: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        candidate_copy = dict(candidate)
        classification = _catastrophic_classification_for_candidate(
            candidate_copy,
            by_candidate_id,
            by_symbol,
        )
        taxonomy_matched = bool(classification.get("matched_categories"))
        confirmed_catastrophic = taxonomy_matched and bool(classification.get("blocks_contrarian_entry"))
        unknown_text = classification.get("classification_method") in {"UNAVAILABLE_INPUT", "UNKNOWN"}
        missing_availability = not bool(classification.get("availability_timestamp_present"))
        manual_review = (
            taxonomy_matched
            and bool(classification.get("requires_manual_review"))
            and not unknown_text
        )
        unknown = unknown_text or (
            classification.get("highest_severity") == "UNKNOWN"
            and not taxonomy_matched
        )
        if policy_mode == "CONFIRMED_ONLY_RESEARCH":
            blocked = confirmed_catastrophic
        elif policy_mode == "MANUAL_REVIEW_RESEARCH":
            blocked = confirmed_catastrophic or manual_review
        else:
            blocked = confirmed_catastrophic or manual_review or unknown or missing_availability
        enriched = {
            **candidate_copy,
            "catastrophic_veto_action": "EXCLUDE_FROM_RESEARCH_VARIANT" if blocked else "KEEP",
            "catastrophic_veto_reason": _catastrophic_veto_removal_reason(
                bool(classification.get("blocks_contrarian_entry")),
                manual_review,
                unknown,
                missing_availability,
            ),
            "catastrophic_veto_matched_categories": "|".join(classification.get("matched_categories", [])),
            "catastrophic_veto_highest_severity": classification.get("highest_severity", "UNKNOWN"),
            "catastrophic_veto_point_in_time_safe": classification.get("point_in_time_safe", False),
            "catastrophic_veto_classification_method": classification.get("classification_method", "UNKNOWN"),
            "catastrophic_veto_confirmed_catastrophic": confirmed_catastrophic,
            "catastrophic_veto_manual_review": manual_review,
            "catastrophic_veto_unknown_text": unknown_text,
            "catastrophic_veto_missing_availability": missing_availability,
            "catastrophic_veto_policy_mode": policy_mode,
        }
        if blocked:
            blocked_candidates.append(enriched)
        else:
            filtered_candidates.append(enriched)
        if manual_review:
            manual_review_candidates.append(enriched)
        if unknown:
            unknown_candidates.append(enriched)
        if confirmed_catastrophic:
            confirmed_catastrophic_candidates.append(enriched)
        if unknown_text:
            unknown_text_candidates.append(enriched)
        if missing_availability:
            missing_availability_candidates.append(enriched)

    filter_audit = {
        "schema_name": "catastrophic_veto_candidate_filter_audit",
        "schema_version": 1,
        "status": "PASSED_WITH_WARNINGS",
        "deterministic": True,
        "candidate_count_before_veto": len(candidate_rows),
        "candidate_count_after_veto": len(filtered_candidates),
        "blocked_candidate_count": len(blocked_candidates),
        "strict_policy_blocked_candidate_count": len(blocked_candidates),
        "confirmed_catastrophic_candidate_count": len(confirmed_catastrophic_candidates),
        "confirmed_catastrophic_blocked_candidate_count": len(confirmed_catastrophic_candidates),
        "manual_review_candidate_count": len(manual_review_candidates),
        "unknown_text_candidate_count": len(unknown_text_candidates),
        "missing_availability_candidate_count": len(missing_availability_candidates),
        "unknown_candidate_count": len(unknown_candidates),
        "catastrophic_veto_policy_mode": policy_mode,
        "base_candidates_mutated": False,
        "warnings": [
            (
                "Manual-review, unknown, missing-text, and not-point-in-time-safe rows are excluded from the research-only veto variant."
                if policy_mode == "STRICT_SAFETY"
                else "Unknown or missing evidence is reported separately and is not approved for paper/live use."
            ),
        ],
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
    }
    return {
        "filtered_candidates": filtered_candidates,
        "blocked_candidates": blocked_candidates,
        "manual_review_candidates": manual_review_candidates,
        "unknown_candidates": unknown_candidates,
        "confirmed_catastrophic_candidates": confirmed_catastrophic_candidates,
        "unknown_text_candidates": unknown_text_candidates,
        "missing_availability_candidates": missing_availability_candidates,
        "filter_audit": filter_audit,
    }


def _catastrophic_veto_full_replay_blocker() -> dict[str, Any]:
    return {
        "safe_replay_insertion_point": "RESEARCH_STRATEGY_VARIANT_INPUT_SEAM",
        "full_replay_blocker": "catastrophic-veto extra variant is absent from replay metrics, equity, or variant metadata output",
        "full_replay_limitation": (
            "Full filtered replay is unavailable unless the separate extra variant completes through the existing "
            "replay mechanics and exposes metrics, equity, and variant metadata."
        ),
        "candidate_input_source": "joined stock-alpha/news candidate rows inside write_stock_alpha_news_risk_overlay_research",
        "strategy_variant_builder": "existing research replay variant assembly for price_only/news_contrarian_rerank plus opt-in ResearchStrategyVariantSpec seam",
        "replay_engine_entrypoint": "_build_open_trade_replay optional extra_research_variants adapter",
        "metrics_writer_entrypoint": "portfolio_comparison and risk/metrics JSON writers in write_stock_alpha_news_risk_overlay_research",
        "trade_ledger_writer_entrypoint": "trade ledger CSV writers in write_stock_alpha_news_risk_overlay_research",
        "equity_writer_entrypoint": "daily equity CSV writers in write_stock_alpha_news_risk_overlay_research",
    }


def _catastrophic_veto_replay_seam_report(*, full_replay_computed: bool = False) -> dict[str, Any]:
    blocker = _catastrophic_veto_full_replay_blocker()
    return {
        "schema_name": "catastrophic_veto_replay_seam_report",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "catastrophic_veto_research_v1",
        "status": "FULL_REPLAY_COMPUTED" if full_replay_computed else "ADAPTER_ADDED_NOT_EXECUTED",
        "candidate_construction_entrypoint": blocker["candidate_input_source"],
        "strategy_variant_construction_entrypoint": blocker["strategy_variant_builder"],
        "replay_input_entrypoint": blocker["replay_engine_entrypoint"],
        "metrics_writer_entrypoint": blocker["metrics_writer_entrypoint"],
        "trade_ledger_writer_entrypoint": blocker["trade_ledger_writer_entrypoint"],
        "equity_writer_entrypoint": blocker["equity_writer_entrypoint"],
        "safe_filtered_variant_seam_status": "REPLAY_ADAPTER_EXECUTED" if full_replay_computed else "REPLAY_ADAPTER_AVAILABLE_OPT_IN_ONLY",
        "full_replay_possible_after_this_pass": full_replay_computed,
        "full_replay_computed": full_replay_computed,
        "remaining_blocker": "" if full_replay_computed else blocker["full_replay_blocker"],
        "seam_changes": [
            "Added ResearchCandidateFilterSpec",
            "Added ResearchStrategyVariantSpec",
            "Added build_research_strategy_variant_inputs copy-only opt-in input builder",
            "Added _build_open_trade_replay extra_research_variants adapter",
        ],
        "default_behavior_unchanged": True,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "warnings": [
            "Seam is research-only and not used by broker, provider, paper, or live paths.",
            "Full replay is a separate research-only scenario and is not the current strategy." if full_replay_computed else "Full replay remains unavailable because replay output does not contain the catastrophic-veto variant.",
        ],
    }


def _catastrophic_classification_for_candidate(
    candidate: Mapping[str, Any],
    by_candidate_id: Mapping[str, Mapping[str, Any]],
    by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Mapping[str, Any]:
    candidate_id = _mapping_first(candidate, "candidate_id", "trade_id", "row_id")
    if candidate_id and candidate_id in by_candidate_id:
        return by_candidate_id[candidate_id]
    symbol = _mapping_first(candidate, "symbol", "ticker")
    if symbol and symbol in by_symbol:
        symbol_rows = list(by_symbol[symbol])
        blocking_rows = [
            row
            for row in symbol_rows
            if row.get("blocks_contrarian_entry")
            or row.get("requires_manual_review")
            or row.get("classification_method") == "UNAVAILABLE_INPUT"
            or not row.get("point_in_time_safe", False)
        ]
        return blocking_rows[0] if blocking_rows else symbol_rows[0]
    return {
        "candidate_id": candidate_id or "UNKNOWN",
        "symbol": symbol or "UNKNOWN",
        "matched": False,
        "matched_categories": [],
        "highest_severity": "UNKNOWN",
        "blocks_contrarian_entry": False,
        "requires_manual_review": True,
        "classification_method": "UNKNOWN",
        "point_in_time_safe": False,
    }


def _catastrophic_veto_strategy_artifacts(
    rows: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    classifications = classify_catastrophic_news_rows(rows)
    filter_result = apply_catastrophic_veto_to_candidates(rows, classifications)
    filtered_candidates = list(filter_result["filtered_candidates"])
    blocked_candidate_rows = list(filter_result["blocked_candidates"])
    by_candidate_id = {
        row["candidate_id"]: row
        for row in classifications
        if row.get("candidate_id") not in {None, "", "UNKNOWN"}
    }
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in classifications:
        symbol = str(row.get("symbol", "UNKNOWN"))
        if symbol != "UNKNOWN":
            by_symbol.setdefault(symbol, []).append(row)

    blocked_candidates = [row for row in classifications if row.get("blocks_contrarian_entry")]
    manual_review_candidates = [
        row
        for row in classifications
        if row.get("matched_categories")
        and row.get("requires_manual_review")
        and row.get("classification_method") not in {"UNAVAILABLE_INPUT", "UNKNOWN"}
    ]
    unknown_candidates = [
        row
        for row in classifications
        if row.get("highest_severity") == "UNKNOWN"
        or row.get("classification_method") == "UNAVAILABLE_INPUT"
    ]
    point_in_time_safe_count = sum(1 for row in classifications if row.get("point_in_time_safe"))
    point_in_time_unsafe_count = len(classifications) - point_in_time_safe_count

    trade_rows: list[dict[str, Any]] = []
    removed_trade_rows: list[dict[str, Any]] = []
    all_replay_trades = list(replay.get("trade_ledger", []) or [])
    executed_trades = [
        trade
        for trade in all_replay_trades
        if str(trade.get("strategy_variant", "")) == "news_contrarian_rerank"
    ]
    blocked_trade_count = 0
    manual_review_trade_count = 0
    unknown_trade_count = 0
    for trade in executed_trades:
        classification = _catastrophic_classification_for_trade(trade, by_candidate_id, by_symbol)
        blocked = bool(classification.get("blocks_contrarian_entry"))
        taxonomy_matched = bool(classification.get("matched_categories"))
        unknown_text = classification.get("classification_method") in {"UNAVAILABLE_INPUT", "UNKNOWN"}
        missing_availability = not bool(classification.get("availability_timestamp_present"))
        manual_review = taxonomy_matched and bool(classification.get("requires_manual_review")) and not unknown_text
        unknown = unknown_text or (
            classification.get("highest_severity") == "UNKNOWN"
            and not taxonomy_matched
        )
        blocked_trade_count += int(blocked)
        manual_review_trade_count += int(manual_review)
        unknown_trade_count += int(unknown)
        would_remove = blocked or manual_review or unknown or missing_availability
        trade_row = {
            "trade_id": _mapping_first(trade, "trade_id", "id", "row_id") or "UNKNOWN",
            "candidate_id": _mapping_first(trade, "candidate_id", "trade_id", "row_id") or "UNKNOWN",
            "symbol": _mapping_first(trade, "symbol", "ticker") or classification.get("symbol", "UNKNOWN"),
            "strategy_variant": _mapping_first(trade, "strategy_variant", "variant", "strategy") or "UNKNOWN",
            "entry_date": _mapping_first(trade, "entry_date", "entry_timestamp", "open_date") or "UNAVAILABLE_INPUT",
            "exit_date": _mapping_first(trade, "exit_date", "exit_timestamp", "close_date") or "UNAVAILABLE_INPUT",
            "net_return": _mapping_first(trade, "net_return", "trade_return_net", "return") or "UNAVAILABLE_INPUT",
            "pnl": _mapping_first(trade, "pnl", "net_pnl", "profit_loss") or "UNAVAILABLE_INPUT",
            "matched": classification.get("matched", False),
            "matched_categories": "|".join(classification.get("matched_categories", [])),
            "highest_severity": classification.get("highest_severity", "UNKNOWN"),
            "blocks_contrarian_entry": blocked,
            "requires_manual_review": manual_review,
            "unknown_or_unavailable": unknown,
            "unknown_text": unknown_text,
            "missing_availability": missing_availability,
            "point_in_time_safe": classification.get("point_in_time_safe", False),
            "classification_method": classification.get("classification_method", "UNKNOWN"),
            "removal_reason": _catastrophic_veto_removal_reason(
                blocked,
                manual_review,
                unknown,
                missing_availability,
            ),
            "research_only_veto_would_apply": would_remove,
        }
        trade_rows.append(trade_row)
        if would_remove:
            removed_trade_rows.append(trade_row)

    blocked_categories = sorted(
        {
            category
            for row in blocked_candidates
            for category in row.get("matched_categories", [])
        }
    )
    blocked_symbols = sorted({str(row.get("symbol")) for row in blocked_candidates if row.get("symbol") != "UNKNOWN"})
    warnings = [
        "Research-only attribution; catastrophic veto is not enforced in the current strategy.",
        "Missing text and UNKNOWN rows are not treated as safe.",
    ]
    if point_in_time_unsafe_count:
        warnings.append("At least one candidate lacks point-in-time-safe availability evidence.")
    replay_impact_status = "APPROXIMATE_LEDGER_SIMULATION" if executed_trades else "UNAVAILABLE_INPUT"
    available_removed_pnl = [_as_optional_float(row.get("pnl")) for row in removed_trade_rows]
    available_removed_returns = [_as_optional_float(row.get("net_return")) for row in removed_trade_rows]
    removed_pnl_values = [value for value in available_removed_pnl if value is not None]
    removed_return_values = [value for value in available_removed_returns if value is not None]
    removed_pnl_contribution: float | str = sum(removed_pnl_values) if removed_pnl_values else "UNAVAILABLE_INPUT"
    removed_return_contribution: float | str = sum(removed_return_values) if removed_return_values else "UNAVAILABLE_INPUT"
    removed_symbol_rows = _catastrophic_veto_removed_symbol_rows(
        removed_trade_rows,
        blocked_candidates,
    )
    full_replay_blocker = _catastrophic_veto_full_replay_blocker()
    veto_variant = "news_contrarian_rerank_catastrophic_veto"
    confirmed_only_variant = "news_contrarian_rerank_catastrophic_veto_confirmed_only"
    manual_review_variant = "news_contrarian_rerank_catastrophic_veto_manual_review"
    risk_metrics = replay.get("risk_metrics", {})
    daily_equity = replay.get("daily_equity", {})
    extra_variant_metadata = replay.get("extra_research_variant_metadata", {})
    full_replay_computed = (
        isinstance(risk_metrics, Mapping)
        and veto_variant in risk_metrics
        and isinstance(daily_equity, Mapping)
        and veto_variant in daily_equity
        and isinstance(extra_variant_metadata, Mapping)
        and veto_variant in extra_variant_metadata
    )
    veto_trade_ledger = [
        dict(trade)
        for trade in all_replay_trades
        if str(trade.get("strategy_variant", "")) == veto_variant
    ]
    veto_equity = (
        [dict(row) for row in daily_equity.get(veto_variant, [])]
        if full_replay_computed
        else []
    )
    filter_audit = dict(filter_result["filter_audit"])
    if not full_replay_computed:
        full_replay_status = "FULL_REPLAY_NOT_AVAILABLE"
        veto_metrics_status = "UNAVAILABLE_REPLAY_NOT_COMPUTED"
        empty_output_reason = "FULL_REPLAY_NOT_AVAILABLE"
    elif not filtered_candidates:
        full_replay_status = "FULL_REPLAY_COMPUTED_ZERO_CANDIDATES"
        veto_metrics_status = "UNAVAILABLE_EMPTY_CANDIDATE_SET"
        empty_output_reason = "STRICT_POLICY_BLOCKED_ALL_CANDIDATES"
    elif not veto_trade_ledger:
        full_replay_status = "FULL_REPLAY_COMPUTED_ZERO_TRADES"
        veto_metrics_status = "UNAVAILABLE_ZERO_TRADES"
        empty_output_reason = "FILTERED_CANDIDATES_PRODUCED_NO_TRADES"
    else:
        full_replay_status = "FULL_REPLAY_COMPUTED"
        veto_metrics_status = "AVAILABLE"
        empty_output_reason = ""

    candidate_attribution = {
        "schema_name": "catastrophic_veto_candidate_attribution",
        "schema_version": 1,
        "status": "PASSED_WITH_WARNINGS" if classifications else "UNAVAILABLE_INPUT",
        "candidate_count": len(rows),
        "classified_candidate_count": len(classifications),
        "blocked_candidate_count": len(blocked_candidates),
        "strict_policy_blocked_candidate_count": filter_audit["strict_policy_blocked_candidate_count"],
        "confirmed_catastrophic_candidate_count": filter_audit["confirmed_catastrophic_candidate_count"],
        "confirmed_catastrophic_blocked_candidate_count": filter_audit["confirmed_catastrophic_blocked_candidate_count"],
        "manual_review_candidate_count": len(manual_review_candidates),
        "unknown_text_candidate_count": filter_audit["unknown_text_candidate_count"],
        "missing_availability_candidate_count": filter_audit["missing_availability_candidate_count"],
        "unknown_candidate_count": len(unknown_candidates),
        "catastrophic_veto_policy_mode": "STRICT_SAFETY",
        "executed_trade_count": len(executed_trades),
        "blocked_executed_trade_count": blocked_trade_count,
        "manual_review_executed_trade_count": manual_review_trade_count,
        "unknown_executed_trade_count": unknown_trade_count,
        "blocked_categories": blocked_categories,
        "blocked_symbols": blocked_symbols,
        "point_in_time_safe_count": point_in_time_safe_count,
        "point_in_time_unsafe_count": point_in_time_unsafe_count,
        "warnings": warnings,
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }
    comparison_metrics = replay.get("portfolio_comparison", {})
    if not isinstance(comparison_metrics, Mapping):
        comparison_metrics = {}
    if not isinstance(risk_metrics, Mapping):
        risk_metrics = {}
    comparison_variants = (
        comparison_metrics.get("variants", {})
        if isinstance(comparison_metrics, Mapping)
        else {}
    )
    base_metrics = dict(
        risk_metrics.get("news_contrarian_rerank", {})
        or comparison_variants.get("news_contrarian_rerank", {})
        or comparison_metrics.get("news_contrarian_rerank", {})
        or {}
    )
    price_metrics = dict(
        risk_metrics.get("price_only", {})
        or comparison_variants.get("price_only", {})
        or comparison_metrics.get("price_only", {})
        or {}
    )
    full_veto_metrics = dict(risk_metrics.get(veto_variant, {}) or {}) if full_replay_computed else {}
    extra_veto_variants = [
        name
        for name in (confirmed_only_variant, manual_review_variant)
        if isinstance(risk_metrics, Mapping) and name in risk_metrics
    ]
    base_trade_keys = {
        (str(row.get("candidate_id", "")), str(row.get("symbol", "")), str(row.get("entry_date", "")))
        for row in executed_trades
    }
    veto_trade_keys = {
        (str(row.get("candidate_id", "")), str(row.get("symbol", "")), str(row.get("entry_date", "")))
        for row in veto_trade_ledger
    }
    full_replay_delta_return: float | str = "UNAVAILABLE_INPUT"
    full_replay_delta_pnl: float | str = "UNAVAILABLE_INPUT"
    if full_replay_computed:
        base_return = _number(base_metrics.get("total_return_decimal"))
        veto_return = _number(full_veto_metrics.get("total_return_decimal"))
        base_equity = _number(base_metrics.get("ending_equity"))
        veto_equity_value = _number(full_veto_metrics.get("ending_equity"))
        if base_return is not None and veto_return is not None:
            full_replay_delta_return = veto_return - base_return
        if base_equity is not None and veto_equity_value is not None:
            full_replay_delta_pnl = veto_equity_value - base_equity
    delta_metrics = {
        "removed_pnl_contribution": removed_pnl_contribution,
        "removed_return_contribution": removed_return_contribution,
        "full_replay_delta_return": full_replay_delta_return,
        "full_replay_delta_pnl": full_replay_delta_pnl,
        "calculation": (
            "separate full portfolio replay through the existing accounting engine"
            if full_replay_computed
            else "ledger-level removal attribution only; no portfolio path recomputation"
        ),
    }
    veto_metrics = {
        "approximate_removed_trade_count": len(removed_trade_rows),
        "approximate_removed_pnl_contribution": removed_pnl_contribution,
        "approximate_removed_return_contribution": removed_return_contribution,
    }
    strategy_comparison = {
        "schema_name": "catastrophic_veto_strategy_comparison",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "catastrophic_veto_research_v1",
        "status": full_replay_status,
        "replay_impact_status": full_replay_status,
        "approximate_replay_impact_status": replay_impact_status,
        "safe_replay_insertion_point": full_replay_blocker["safe_replay_insertion_point"],
        "full_replay_blocker": "" if full_replay_computed else full_replay_blocker["full_replay_blocker"],
        "full_replay_limitation": "" if full_replay_computed else full_replay_blocker["full_replay_limitation"],
        "candidate_input_source": full_replay_blocker["candidate_input_source"],
        "strategy_variant_builder": full_replay_blocker["strategy_variant_builder"],
        "replay_engine_entrypoint": full_replay_blocker["replay_engine_entrypoint"],
        "metrics_writer_entrypoint": full_replay_blocker["metrics_writer_entrypoint"],
        "trade_ledger_writer_entrypoint": full_replay_blocker["trade_ledger_writer_entrypoint"],
        "equity_writer_entrypoint": full_replay_blocker["equity_writer_entrypoint"],
        "base_strategy": "news_contrarian_rerank",
        "veto_strategy": "news_contrarian_rerank_catastrophic_veto",
        "strategy_names": [
            "price_only",
            "news_contrarian_rerank",
            "news_contrarian_rerank_catastrophic_veto",
            *extra_veto_variants,
        ],
        "additional_research_variants": {
            name: {
                "metrics": dict(risk_metrics.get(name, {}) or {}),
                "paper_trading_enabled": False,
                "live_trading_enabled": False,
                "validation_passed": False,
                "final_validation_status": "NOT_FINAL_VALIDATION",
            }
            for name in extra_veto_variants
        },
        "veto_enabled_for_research": True,
        "veto_enabled_for_paper_trading": False,
        "veto_enabled_for_live_trading": False,
        "used_in_current_replay": False,
        "full_replay_computed": full_replay_computed,
        "approximate_simulation_used": not full_replay_computed and bool(executed_trades),
        "blocked_candidate_count": filter_audit["strict_policy_blocked_candidate_count"],
        "blocked_trade_count": len(base_trade_keys - veto_trade_keys) if full_replay_computed else blocked_trade_count,
        "full_replay_blocked_candidate_count": filter_audit["strict_policy_blocked_candidate_count"],
        "full_replay_removed_trade_count": len(base_trade_keys - veto_trade_keys) if full_replay_computed else "UNAVAILABLE_INPUT",
        "approximate_blocked_candidate_count": len(blocked_candidates),
        "approximate_removed_trade_count": len(removed_trade_rows),
        "catastrophic_veto_policy_mode": "STRICT_SAFETY",
        "veto_metrics_status": veto_metrics_status,
        "manual_review_count": len(manual_review_candidates),
        "unknown_count": len(unknown_candidates),
        "base_metrics": {
            "price_only": price_metrics,
            "news_contrarian_rerank": base_metrics,
        },
        "veto_metrics": full_veto_metrics if full_replay_computed else (veto_metrics if executed_trades else {}),
        "delta_metrics": delta_metrics,
        "warnings": warnings + ([] if full_replay_computed else ["Full filtered replay variant has not been computed; impact is approximate ledger attribution."]),
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
    }
    filtered_report = {
        "schema_name": "catastrophic_veto_filtered_strategy_report",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "catastrophic_veto_research_v1",
        "status": replay_impact_status,
        "replay_impact_status": replay_impact_status,
        "approximate_simulation_superseded_by_full_replay": full_replay_computed,
        "base_strategy": "news_contrarian_rerank",
        "veto_strategy": "news_contrarian_rerank_catastrophic_veto",
        "veto_policy_version": "v1",
        "veto_enabled_for_research": True,
        "veto_enabled_for_paper_trading": False,
        "veto_enabled_for_live_trading": False,
        "used_in_current_replay": False,
        "full_replay_computed": full_replay_computed,
        "approximate_simulation_used": not full_replay_computed and bool(executed_trades),
        "blocked_candidate_count": len(blocked_candidates),
        "blocked_trade_count": blocked_trade_count,
        "manual_review_candidate_count": len(manual_review_candidates),
        "manual_review_trade_count": manual_review_trade_count,
        "unknown_candidate_count": len(unknown_candidates),
        "unknown_trade_count": unknown_trade_count,
        "price_only_metrics": price_metrics,
        "base_contrarian_metrics": base_metrics,
        "veto_contrarian_metrics": full_veto_metrics if full_replay_computed else (veto_metrics if executed_trades else {}),
        "delta_metrics": delta_metrics,
        "removed_trade_summary": {
            "removed_trade_count": len(removed_trade_rows),
            "available_pnl_contribution": removed_pnl_contribution,
            "available_return_contribution": removed_return_contribution,
        },
        "removed_symbol_summary": removed_symbol_rows,
        "limitations": ([] if full_replay_computed else [
            "Approximate ledger simulation removes attributed trades but does not recompute portfolio path, cash drag, sizing, or replacement selections.",
            "Full filtered replay has not been computed.",
        ]),
        "warnings": warnings,
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }
    full_replay_report = {
        "schema_name": "catastrophic_veto_full_replay_report",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "catastrophic_veto_research_v1",
        "status": full_replay_status,
        "replay_impact_status": full_replay_status,
        "base_strategy": "news_contrarian_rerank",
        "veto_strategy": "news_contrarian_rerank_catastrophic_veto",
        "full_replay_computed": full_replay_computed,
        "approximate_simulation_used": not full_replay_computed,
        "veto_enabled_for_research": True,
        "veto_enabled_for_paper_trading": False,
        "veto_enabled_for_live_trading": False,
        "used_in_current_replay": False,
        "candidate_count_before_veto": len(rows),
        "candidate_count_after_veto": len(filtered_candidates),
        "blocked_candidate_count": len(blocked_candidate_rows),
        "strict_policy_blocked_candidate_count": filter_audit["strict_policy_blocked_candidate_count"],
        "confirmed_catastrophic_candidate_count": filter_audit["confirmed_catastrophic_candidate_count"],
        "confirmed_catastrophic_blocked_candidate_count": filter_audit["confirmed_catastrophic_blocked_candidate_count"],
        "manual_review_candidate_count": len(filter_result["manual_review_candidates"]),
        "unknown_text_candidate_count": filter_audit["unknown_text_candidate_count"],
        "missing_availability_candidate_count": filter_audit["missing_availability_candidate_count"],
        "unknown_candidate_count": len(filter_result["unknown_candidates"]),
        "catastrophic_veto_policy_mode": "STRICT_SAFETY",
        "base_trade_count": len(executed_trades),
        "veto_trade_count": len(veto_trade_ledger) if full_replay_computed else "UNAVAILABLE_INPUT",
        "removed_trade_count": len(base_trade_keys - veto_trade_keys) if full_replay_computed else "UNAVAILABLE_INPUT",
        "replacement_trade_count": len(veto_trade_keys - base_trade_keys) if full_replay_computed else "UNAVAILABLE_INPUT",
        "price_only_metrics": price_metrics,
        "base_contrarian_metrics": base_metrics,
        "veto_contrarian_metrics": full_veto_metrics,
        "veto_metrics_status": veto_metrics_status,
        "empty_output_reason": empty_output_reason,
        "delta_metrics": {
            "delta_return": full_replay_delta_return,
            "delta_pnl": full_replay_delta_pnl,
            "reason": (
                "computed from the separate research-only replay variant"
                if full_replay_computed
                else "full filtered replay variant is absent from replay output"
            ),
        },
        "equity_curve_path": "catastrophic_veto_full_replay_equity.csv",
        "trade_ledger_path": "catastrophic_veto_full_replay_trade_ledger.csv",
        "filtered_candidate_path": "catastrophic_veto_filtered_candidates.csv",
        "blocked_candidate_path": "catastrophic_veto_blocked_candidates.csv",
        "candidate_input_source": full_replay_blocker["candidate_input_source"],
        "strategy_variant_builder": full_replay_blocker["strategy_variant_builder"],
        "replay_engine_entrypoint": full_replay_blocker["replay_engine_entrypoint"],
        "safe_replay_insertion_point": full_replay_blocker["safe_replay_insertion_point"],
        "full_replay_blocker": "" if full_replay_computed else full_replay_blocker["full_replay_blocker"],
        "full_replay_limitation": "" if full_replay_computed else full_replay_blocker["full_replay_limitation"],
        "limitations": ["No replay accounting, sizing, cash, entry/exit, or base candidate-selection mechanics were changed."],
        "warnings": (["Paper/live trading remain disabled."] if full_replay_computed else [
            "Research-only full replay variant is not available from the current safe insertion point.",
            "Paper/live trading remain disabled.",
        ]),
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }
    return (
        candidate_attribution,
        trade_rows,
        strategy_comparison,
        _catastrophic_veto_policy(),
        filtered_report,
        removed_trade_rows,
        removed_symbol_rows,
        full_replay_report,
        veto_trade_ledger if full_replay_computed else [],
        veto_equity,
        filtered_candidates,
        blocked_candidate_rows,
    )


def _catastrophic_classification_for_trade(
    trade: Mapping[str, Any],
    by_candidate_id: Mapping[str, Mapping[str, Any]],
    by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Mapping[str, Any]:
    candidate_id = _mapping_first(trade, "candidate_id", "trade_id", "row_id")
    if candidate_id and candidate_id in by_candidate_id:
        return by_candidate_id[candidate_id]
    symbol = _mapping_first(trade, "symbol", "ticker")
    if symbol and symbol in by_symbol:
        symbol_rows = list(by_symbol[symbol])
        blocking_rows = [
            row
            for row in symbol_rows
            if row.get("blocks_contrarian_entry")
            or row.get("requires_manual_review")
            or row.get("classification_method") == "UNAVAILABLE_INPUT"
        ]
        return blocking_rows[0] if blocking_rows else symbol_rows[0]
    return {
        "candidate_id": candidate_id or "UNKNOWN",
        "symbol": symbol or "UNKNOWN",
        "matched": False,
        "matched_categories": [],
        "highest_severity": "UNKNOWN",
        "blocks_contrarian_entry": False,
        "requires_manual_review": True,
        "classification_method": "UNKNOWN",
        "point_in_time_safe": False,
    }


def _catastrophic_news_evidence_quality_artifacts(
    rows: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    full_replay_report: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    classifications = classify_catastrophic_news_rows(rows)
    filter_result = apply_catastrophic_veto_to_candidates(rows, classifications)
    audit = dict(filter_result["filter_audit"])

    def present(row: Mapping[str, Any], *keys: str) -> bool:
        return any(row.get(key) is not None and str(row.get(key)).strip() for key in keys)

    field_specs = {
        "headline": ("headline_text", "headline", "title"),
        "summary": ("summary_text", "summary", "description"),
        "body": ("body_text", "body", "content", "article_body"),
        "publication_timestamp": ("publication_timestamp", "published_at", "timestamp"),
        "availability_timestamp": ("availability_timestamp", "available_at", "asof_timestamp"),
        "event_category": ("event_type", "event", "provider_category", "category"),
    }
    field_counts = {
        name: sum(present(row, *keys) for row in rows)
        for name, keys in field_specs.items()
    }
    has_any_text_count = sum(
        present(row, "headline_text", "headline", "title", "summary_text", "summary", "description", "body_text", "body", "content", "article_body", "event_type", "event", "provider_category", "category")
        for row in rows
    )
    point_in_time_safe_count = sum(bool(row.get("point_in_time_safe")) for row in classifications)
    usable_strict = sum(
        bool(row.get("point_in_time_safe"))
        and row.get("classification_method") != "UNAVAILABLE_INPUT"
        for row in classifications
    )
    usable_confirmed_only = sum(row.get("classification_method") != "UNAVAILABLE_INPUT" for row in classifications)
    status = "INSUFFICIENT_FOR_STRICT_VETO" if usable_strict == 0 else (
        "PARTIAL_EVIDENCE" if usable_strict < len(rows) else "SUFFICIENT_FOR_RESEARCH"
    )
    report = {
        "schema_name": "catastrophic_news_evidence_quality_report",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "catastrophic_veto_research_v1",
        "status": status,
        "candidate_count": len(rows),
        "has_headline_count": field_counts["headline"],
        "has_summary_count": field_counts["summary"],
        "has_body_count": field_counts["body"],
        "has_any_text_count": has_any_text_count,
        "missing_text_count": len(rows) - has_any_text_count,
        "has_publication_timestamp_count": field_counts["publication_timestamp"],
        "has_availability_timestamp_count": field_counts["availability_timestamp"],
        "missing_availability_timestamp_count": len(rows) - field_counts["availability_timestamp"],
        "point_in_time_safe_count": point_in_time_safe_count,
        "point_in_time_unsafe_count": len(rows) - point_in_time_safe_count,
        "has_event_category_count": field_counts["event_category"],
        "uncategorized_event_count": len(rows) - field_counts["event_category"],
        "confirmed_catastrophic_candidate_count": audit["confirmed_catastrophic_candidate_count"],
        "manual_review_candidate_count": audit["manual_review_candidate_count"],
        "unknown_candidate_count": audit["unknown_candidate_count"],
        "strict_policy_blocked_candidate_count": audit["strict_policy_blocked_candidate_count"],
        "usable_for_strict_veto_count": usable_strict,
        "usable_for_confirmed_only_veto_count": usable_confirmed_only,
        "warnings": [
            "Publication timestamps alone are not point-in-time availability evidence.",
            "Unknown or missing evidence remains blocked by STRICT_SAFETY.",
            "Evidence-quality reporting is observational and does not alter replay inputs.",
        ],
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }
    by_field = [
        {
            "field": name,
            "available_count": count,
            "missing_count": len(rows) - count,
            "availability_ratio": count / max(len(rows), 1),
        }
        for name, count in field_counts.items()
    ]

    by_symbol_groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_symbol_groups.setdefault(str(row.get("symbol", row.get("ticker", "UNKNOWN"))), []).append(row)
    by_symbol = []
    for symbol, symbol_rows in sorted(by_symbol_groups.items()):
        symbol_classifications = classify_catastrophic_news_rows(symbol_rows)
        symbol_filter = apply_catastrophic_veto_to_candidates(symbol_rows, symbol_classifications)["filter_audit"]
        symbol_text = sum(
            present(row, "headline_text", "headline", "title", "summary_text", "summary", "description", "body_text", "body", "content", "article_body", "event_type", "event", "provider_category", "category")
            for row in symbol_rows
        )
        symbol_availability = sum(present(row, "availability_timestamp", "available_at", "asof_timestamp") for row in symbol_rows)
        by_symbol.append({
            "symbol": symbol,
            "candidate_count": len(symbol_rows),
            "has_any_text_count": symbol_text,
            "missing_text_count": len(symbol_rows) - symbol_text,
            "has_availability_timestamp_count": symbol_availability,
            "missing_availability_timestamp_count": len(symbol_rows) - symbol_availability,
            "confirmed_catastrophic_candidate_count": symbol_filter["confirmed_catastrophic_candidate_count"],
            "strict_policy_blocked_candidate_count": symbol_filter["strict_policy_blocked_candidate_count"],
        })

    base_trades = [
        trade for trade in replay.get("trade_ledger", [])
        if str(trade.get("strategy_variant", "")) == "news_contrarian_rerank"
    ]
    candidate_id_sets = {
        "STRICT_SAFETY": {str(row.get("candidate_id")) for row in filter_result["blocked_candidates"]},
        "CONFIRMED_ONLY_RESEARCH": {str(row.get("candidate_id")) for row in filter_result["confirmed_catastrophic_candidates"]},
        "MANUAL_REVIEW_RESEARCH": {
            str(row.get("candidate_id"))
            for row in [*filter_result["confirmed_catastrophic_candidates"], *filter_result["manual_review_candidates"]]
        },
    }
    counts = []
    mode_variant_names = {
        "STRICT_SAFETY": "news_contrarian_rerank_catastrophic_veto",
        "CONFIRMED_ONLY_RESEARCH": "news_contrarian_rerank_catastrophic_veto_confirmed_only",
        "MANUAL_REVIEW_RESEARCH": "news_contrarian_rerank_catastrophic_veto_manual_review",
    }
    risk_metrics = replay.get("risk_metrics", {})
    daily_equity = replay.get("daily_equity", {})
    extra_variant_metadata = replay.get("extra_research_variant_metadata", {})
    for mode in ("STRICT_SAFETY", "CONFIRMED_ONLY_RESEARCH", "MANUAL_REVIEW_RESEARCH"):
        blocked_ids = candidate_id_sets[mode]
        total_blocked = len(blocked_ids)
        strict_mode = mode == "STRICT_SAFETY"
        variant_name = mode_variant_names[mode]
        replayed = (
            isinstance(risk_metrics, Mapping)
            and variant_name in risk_metrics
            and isinstance(daily_equity, Mapping)
            and variant_name in daily_equity
            and isinstance(extra_variant_metadata, Mapping)
            and variant_name in extra_variant_metadata
        )
        counts.append({
            "policy_mode": mode,
            "strategy_variant": variant_name,
            "candidate_count_before_veto": len(rows),
            "candidate_count_after_veto": len(rows) - total_blocked,
            "confirmed_catastrophic_blocked_candidate_count": audit["confirmed_catastrophic_candidate_count"],
            "manual_review_blocked_candidate_count": audit["manual_review_candidate_count"] if mode == "MANUAL_REVIEW_RESEARCH" else (audit["manual_review_candidate_count"] if strict_mode else 0),
            "unknown_text_blocked_candidate_count": audit["unknown_text_candidate_count"] if strict_mode else 0,
            "missing_availability_blocked_candidate_count": audit["missing_availability_candidate_count"] if strict_mode else 0,
            "total_blocked_candidate_count": total_blocked,
            "estimated_removed_trade_count": sum(str(trade.get("candidate_id")) in blocked_ids for trade in base_trades),
            "full_replay_computed": bool(full_replay_report.get("full_replay_computed")) if strict_mode else replayed,
            "full_replay_status": full_replay_report.get("replay_impact_status", "FULL_REPLAY_NOT_AVAILABLE") if strict_mode else ("FULL_REPLAY_COMPUTED" if replayed else "COUNT_ONLY_NOT_REPLAYED"),
            "veto_metrics_status": full_replay_report.get("veto_metrics_status", "UNAVAILABLE_INPUT") if strict_mode else ("AVAILABLE" if replayed else "COUNT_ONLY_NOT_REPLAYED"),
            "paper_trading_allowed": False,
            "live_trading_allowed": False,
            "validation_passed": False,
            "final_validation_status": "NOT_FINAL_VALIDATION",
            "warnings": ["Research-only policy comparison; unknown evidence is not approved for paper/live use."],
        })
    comparison = {
        "schema_name": "catastrophic_veto_policy_mode_comparison",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "catastrophic_veto_research_v1",
        "status": "RESEARCH_ONLY_COUNT_COMPARISON",
        "active_full_replay_policy_mode": "STRICT_SAFETY",
        "policy_modes": counts,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "warnings": ["Policy-mode variants are research-only; unknown evidence is not approved for paper/live use."],
    }
    return report, by_field, by_symbol, comparison, counts


EVENT_TAXONOMY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("catastrophic_or_distress", ("bankruptcy", "chapter 11", "insolvency", "liquidation", "going concern", "delisting", "default", "administration", "winding-up petition", "trading suspension", "suspended trading")),
    ("fraud_or_accounting", ("fraud", "accounting irregular", "restatement", "misstatement", "auditor resign", "qualified audit", "criminal probe", "enforcement action")),
    ("distressed_dilution", ("deep discount raise", "emergency capital raise", "rescue financing", "distressed dilution", "highly dilutive", "covenant breach")),
    ("guidance_cut", ("cuts guidance", "lowers guidance", "withdraws guidance", "guidance cut", "reduces outlook", "large guidance cut")),
    ("earnings_miss", ("earnings miss", "misses estimates", "misses expectations", "profit warning", "revenue miss")),
    ("analyst_downgrade", ("downgrade", "downgrades", "price target cut", "rating cut")),
    ("capital_raise_or_dilution", ("stock offering", "share offering", "capital raise", "dilution", "dilutive", "secondary offering")),
    ("litigation_or_regulatory", ("lawsuit", "litigation", "class action", "regulatory", "investigation", "probe", "sec charges", "major enforcement")),
    ("management_change", ("ceo resigns", "cfo resigns", "management change", "steps down", "departure")),
    ("operational_issue", ("plant shutdown", "production halt", "recall", "supply disruption", "cyberattack", "outage")),
    ("macro_or_sector", ("sector", "industry", "tariff", "rates", "inflation", "macro", "commodity")),
)


KEYWORD_BASELINE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("distress_score", ("bankruptcy", "insolvency", "going concern", "default", "delisting", "liquidation")),
    ("earnings_negative_score", ("earnings miss", "misses estimates", "profit warning", "revenue miss")),
    ("guidance_negative_score", ("cuts guidance", "lowers guidance", "withdraws guidance", "reduces outlook")),
    ("litigation_score", ("lawsuit", "litigation", "class action", "regulatory", "investigation", "probe")),
    ("dilution_score", ("offering", "capital raise", "dilution", "dilutive", "secondary")),
    ("management_change_score", ("ceo resigns", "cfo resigns", "steps down", "departure")),
    ("generic_negative_score", ("falls", "drops", "warning", "cuts", "misses", "lower", "weak")),
)


def _headline_text(row: Mapping[str, Any]) -> str:
    return _mapping_first(row, "headline_text", "headline", "title") or ""


def _normalized_headline(row: Mapping[str, Any]) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", _headline_text(row).lower())).strip()


def _classify_event_taxonomy_from_headline(headline: str) -> tuple[str, list[str]]:
    normalized = re.sub(r"\s+", " ", headline.lower()).strip()
    if not normalized:
        return "uncategorized", []
    for category, terms in EVENT_TAXONOMY_RULES:
        matched = [term for term in terms if term in normalized]
        if matched:
            return category, matched
    return "uncategorized", []


def _news_event_taxonomy_artifacts(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    counts: dict[str, int] = {}
    examples_by_category: dict[str, list[dict[str, Any]]] = {}
    text_count = 0
    for row in rows:
        headline = _headline_text(row)
        if headline.strip():
            text_count += 1
        category, matched_terms = _classify_event_taxonomy_from_headline(headline)
        counts[category] = counts.get(category, 0) + 1
        example = {
            "candidate_id": _mapping_first(row, "candidate_id", "trade_id", "row_id") or "UNKNOWN",
            "symbol": _mapping_first(row, "symbol", "ticker") or "UNKNOWN",
            "decision_timestamp": _mapping_first(row, "decision_timestamp", "rebalance_timestamp", "feature_timestamp", "timestamp") or "UNKNOWN",
            "event_category_research": category,
            "matched_terms": "|".join(matched_terms),
            "headline_text": headline,
            "research_only": True,
        }
        examples_by_category.setdefault(category, [])
        if len(examples_by_category[category]) < 5:
            examples_by_category[category].append(example)
    count_rows = [
        {
            "event_category_research": category,
            "candidate_count": count,
            "coverage_ratio": count / max(len(rows), 1),
        }
        for category, count in sorted(counts.items())
    ]
    examples = [
        example
        for category in sorted(examples_by_category)
        for example in examples_by_category[category]
    ]
    categorized_count = len(rows) - counts.get("uncategorized", 0)
    report = {
        "schema_name": "news_event_taxonomy_report",
        "schema_version": 1,
        "status": "RESEARCH_RULES_READY" if text_count else "UNAVAILABLE_INPUT",
        "taxonomy_method": "DETERMINISTIC_HEADLINE_RULES",
        "candidate_count": len(rows),
        "has_headline_count": text_count,
        "categorized_count": categorized_count,
        "uncategorized_count": counts.get("uncategorized", 0),
        "event_taxonomy_research_ready": text_count > 0,
        "production_event_category_ready": False,
        "categories": [category for category, _terms in EVENT_TAXONOMY_RULES] + ["uncategorized"],
        "warnings": [
            "Research-only headline taxonomy; not a production event_category field.",
            "Catastrophic taxonomy remains the conservative blocking taxonomy.",
        ],
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }
    return report, count_rows, examples


def _news_duplicate_grouping_artifacts(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        symbol = (_mapping_first(row, "symbol", "ticker") or "UNKNOWN").upper()
        headline = _normalized_headline(row)
        availability = _mapping_first(row, "availability_timestamp", "available_at", "asof_timestamp", "news_feature_timestamp") or "UNKNOWN"
        provider = (_mapping_first(row, "provider") or "UNKNOWN").lower()
        date_key = availability[:10] if availability != "UNKNOWN" else "UNKNOWN"
        key = "|".join((symbol, headline or "NO_HEADLINE", date_key, provider))
        group_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        groups.setdefault(group_id, []).append(row)
    duplicate_groups = {group_id: members for group_id, members in groups.items() if len(members) > 1}
    examples = []
    for group_id, members in sorted(duplicate_groups.items()):
        for row in members[:5]:
            examples.append({
                "duplicate_group_id_heuristic": group_id,
                "duplicate_group_method": "symbol_normalized_headline_availability_date_provider",
                "duplicate_group_size": len(members),
                "candidate_id": _mapping_first(row, "candidate_id", "trade_id", "row_id") or "UNKNOWN",
                "symbol": _mapping_first(row, "symbol", "ticker") or "UNKNOWN",
                "headline_text": _headline_text(row),
            })
    duplicate_candidate_count = sum(len(members) for members in duplicate_groups.values())
    report = {
        "schema_name": "news_duplicate_grouping_report",
        "schema_version": 1,
        "status": "HEURISTIC_ONLY",
        "duplicate_group_method": "symbol_normalized_headline_availability_date_provider",
        "candidate_count": len(rows),
        "duplicate_group_count": len(duplicate_groups),
        "singleton_count": sum(1 for members in groups.values() if len(members) == 1),
        "duplicate_candidate_count": duplicate_candidate_count,
        "duplicate_grouping_heuristic_ready": True,
        "production_duplicate_group_id_ready": False,
        "warnings": [
            "Research-only heuristic; not provider-grade deduplication.",
            "Does not mark production duplicate_group_id as fully available.",
        ],
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }
    return report, examples


def _parse_optional_timestamp(value: Any) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _news_point_in_time_text_safety_artifacts(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    examples: list[dict[str, Any]] = []
    has_text_count = 0
    has_availability_count = 0
    safe_text_count = 0
    unsafe_text_count = 0
    publication_only_count = 0
    availability_after_decision_count = 0
    missing_decision_count = 0
    missing_availability_count = 0
    for row in rows:
        has_text = bool(_headline_text(row).strip() or _mapping_first(row, "summary_text", "summary", "body_text", "body", "content"))
        availability_text = _mapping_first(row, "availability_timestamp", "available_at", "asof_timestamp")
        publication_text = _mapping_first(row, "publication_timestamp", "published_at", "published_at_utc", "timestamp")
        decision_text = _mapping_first(row, "decision_timestamp", "rebalance_timestamp", "feature_timestamp")
        availability = _parse_optional_timestamp(availability_text)
        decision = _parse_optional_timestamp(decision_text)
        if has_text:
            has_text_count += 1
        if availability is not None:
            has_availability_count += 1
        else:
            missing_availability_count += 1
        if decision is None:
            missing_decision_count += 1
        if publication_text and availability is None:
            publication_only_count += 1
        safe = has_text and availability is not None and decision is not None and availability <= decision
        after_decision = has_text and availability is not None and decision is not None and availability > decision
        if safe:
            safe_text_count += 1
        elif has_text:
            unsafe_text_count += 1
        if after_decision:
            availability_after_decision_count += 1
        if (after_decision or (has_text and not safe)) and len(examples) < 50:
            examples.append({
                "candidate_id": _mapping_first(row, "candidate_id", "trade_id", "row_id") or "UNKNOWN",
                "symbol": _mapping_first(row, "symbol", "ticker") or "UNKNOWN",
                "decision_timestamp": decision_text or "UNKNOWN",
                "availability_timestamp": availability_text or "UNKNOWN",
                "publication_timestamp": publication_text or "UNKNOWN",
                "has_text": has_text,
                "safe_text": safe,
                "reason": "AVAILABILITY_AFTER_DECISION" if after_decision else "MISSING_POINT_IN_TIME_EVIDENCE",
            })
    report = {
        "schema_name": "news_point_in_time_text_safety_report",
        "schema_version": 1,
        "status": "PARTIAL_POINT_IN_TIME_SAFE" if safe_text_count else "INSUFFICIENT",
        "candidate_count": len(rows),
        "has_text_count": has_text_count,
        "has_availability_timestamp_count": has_availability_count,
        "safe_text_count": safe_text_count,
        "unsafe_text_count": unsafe_text_count,
        "publication_only_count": publication_only_count,
        "availability_after_decision_count": availability_after_decision_count,
        "missing_decision_timestamp_count": missing_decision_count,
        "missing_availability_timestamp_count": missing_availability_count,
        "point_in_time_text_safety_ready": safe_text_count > 0 and availability_after_decision_count == 0,
        "warnings": [
            "Publication timestamp alone is not availability evidence.",
            "Rows without availability_timestamp are not point-in-time safe for text use.",
        ],
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }
    return report, examples


def _news_text_keyword_baseline_artifacts(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    score_rows: list[dict[str, Any]] = []
    scored_count = 0
    for row in rows:
        headline = _headline_text(row)
        normalized = headline.lower()
        scores = {
            score_name: sum(1 for term in terms if term in normalized)
            for score_name, terms in KEYWORD_BASELINE_RULES
        }
        if headline.strip():
            scored_count += 1
        score_rows.append({
            "candidate_id": _mapping_first(row, "candidate_id", "trade_id", "row_id") or "UNKNOWN",
            "symbol": _mapping_first(row, "symbol", "ticker") or "UNKNOWN",
            "decision_timestamp": _mapping_first(row, "decision_timestamp", "rebalance_timestamp", "feature_timestamp", "timestamp") or "UNKNOWN",
            "headline_text": headline,
            **scores,
            "keyword_baseline_total_score": sum(scores.values()),
            "used_in_strategy": False,
            "research_only": True,
        })
    report = {
        "schema_name": "news_text_keyword_baseline_report",
        "schema_version": 1,
        "status": "RESEARCH_ONLY" if scored_count else "UNAVAILABLE_INPUT",
        "candidate_count": len(rows),
        "scored_headline_count": scored_count,
        "score_columns": [name for name, _terms in KEYWORD_BASELINE_RULES],
        "keyword_baseline_ready": scored_count > 0,
        "used_in_strategy": False,
        "sklearn_required": False,
        "finbert_enabled": False,
        "transformer_enabled": False,
        "warnings": [
            "Deterministic keyword baseline only; not used in ranking or replay.",
        ],
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }
    return report, score_rows


REVERSIBLE_EVENT_CATEGORIES = {
    "earnings_miss",
    "guidance_cut",
    "analyst_downgrade",
    "operational_issue",
    "macro_or_sector",
}
SERIOUS_AMBIGUOUS_EVENT_CATEGORIES = {
    "litigation_or_regulatory",
    "management_change",
    "capital_raise_or_dilution",
    "distressed_dilution",
}
EXTREME_DISTRESS_EVENT_CATEGORIES = {
    "catastrophic_or_distress",
}
EXTREME_DISTRESS_TERMS = (
    "bankruptcy",
    "chapter 11",
    "chapter 7",
    "insolven",
    "liquidat",
    "administration",
    "default",
    "going concern",
    "delisting",
    "suspension",
    "trading suspension",
    "suspended trading",
    "fraud",
    "accounting irregular",
    "material misstatement",
    "rescue financing",
)

FRAUD_EVENT_CATEGORIES = {"fraud_or_accounting"}
DISTRESSED_DILUTION_EVENT_CATEGORIES = {"distressed_dilution"}

EXTREME_DISTRESS_ONLY_TERMS = (
    "bankruptcy",
    "chapter 11",
    "chapter 7",
    "insolven",
    "liquidat",
    "administration",
    "default",
    "debt default",
    "going concern",
    "delisting",
    "trading suspension",
    "suspended trading",
    "winding-up petition",
    "rescue financing",
    "emergency rescue financing",
)
FRAUD_TERMS = (
    "fraud",
    "accounting irregular",
    "auditor resign",
    "qualified audit",
    "criminal probe",
    "enforcement action",
    "major enforcement",
    "material misstatement",
)
DISTRESSED_DILUTION_TERMS = (
    "deep discount raise",
    "emergency capital raise",
    "highly dilutive",
    "distressed dilution",
    "covenant breach",
    "rescue financing",
)
SEVERE_LOSS_AVOIDANCE_TERMS = (
    "bankruptcy",
    "default",
    "delisting",
    "suspension",
    "fraud",
    "investigation",
    "liquidation",
    "going concern",
    "covenant breach",
    "rescue financing",
)
SOFT_RISK_REDUCE_TERMS = (
    "major litigation",
    "regulatory investigation",
    "distressed dilution",
    "management crisis",
    "large guidance cut",
)

CATASTROPHIC_POLICY_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "policy_name": "EXTREME_DISTRESS_ONLY",
        "variant_name": "news_contrarian_rerank_extreme_distress_only_veto",
        "policy_stage": "FULL_REPLAY_RESEARCH",
        "block_groups": ("EXTREME_DISTRESS",),
    },
    {
        "policy_name": "EXTREME_DISTRESS_OR_FRAUD",
        "variant_name": "news_contrarian_rerank_extreme_distress_or_fraud_veto",
        "policy_stage": "FULL_REPLAY_RESEARCH",
        "block_groups": ("EXTREME_DISTRESS", "EXTREME_DISTRESS_OR_FRAUD"),
    },
    {
        "policy_name": "DISTRESS_OR_DILUTION",
        "variant_name": "news_contrarian_rerank_distress_or_dilution_veto",
        "policy_stage": "FULL_REPLAY_RESEARCH",
        "block_groups": ("EXTREME_DISTRESS", "EXTREME_DISTRESS_OR_FRAUD", "DISTRESS_OR_DILUTION"),
    },
    {
        "policy_name": "SEVERE_LOSS_AVOIDANCE",
        "variant_name": "news_contrarian_rerank_severe_loss_avoidance_veto",
        "policy_stage": "FULL_REPLAY_RESEARCH",
        "heuristic_terms": SEVERE_LOSS_AVOIDANCE_TERMS,
    },
    {
        "policy_name": "SOFT_RISK_REDUCE",
        "variant_name": "news_contrarian_rerank_soft_risk_reduce_veto",
        "policy_stage": "COUNT_ONLY_PROPOSAL",
        "heuristic_terms": SOFT_RISK_REDUCE_TERMS,
    },
)


def _bounceback_label(row: Mapping[str, Any]) -> str:
    trade_return = _as_optional_float(
        _mapping_first(
            row,
            "net_return",
            "trade_return_net",
            "return",
            "total_return",
            "removed_trade_return",
            "actual_forward_return_10d",
        )
    )
    if trade_return is None:
        return "UNAVAILABLE_OUTCOME"
    if trade_return > 0.10:
        return "BOUNCED_BACK_STRONGLY"
    if trade_return > 0.0:
        return "BOUNCED_BACK_WEAKLY"
    if trade_return <= -0.10:
        return "SEVERE_LOSS"
    return "DID_NOT_BOUNCE"


def _catastrophic_trade_return(row: Mapping[str, Any]) -> float | None:
    return _as_optional_float(
        _mapping_first(
            row,
            "net_return",
            "trade_return_net",
            "return",
            "total_return",
            "removed_trade_return",
            "actual_forward_return_10d",
        )
    )


def _event_category_for_candidate(row: Mapping[str, Any]) -> str:
    category, _terms = _classify_event_taxonomy_from_headline(_headline_text(row))
    return category


def _severity_group_for_candidate(row: Mapping[str, Any]) -> str:
    headline = _headline_text(row).lower()
    has_text = bool(headline.strip())
    has_availability = bool(_mapping_first(row, "availability_timestamp", "available_at", "asof_timestamp"))
    category = _event_category_for_candidate(row)
    if not has_text or not has_availability or category == "uncategorized":
        return "UNKNOWN_OR_INSUFFICIENT_EVIDENCE"
    matched_categories = str(row.get("matched_categories", row.get("catastrophic_veto_matched_categories", ""))).lower()
    if (
        category in DISTRESSED_DILUTION_EVENT_CATEGORIES
        or any(term in headline for term in DISTRESSED_DILUTION_TERMS)
        or "distressed_dilution" in matched_categories
    ):
        return "DISTRESS_OR_DILUTION"
    if (
        category in FRAUD_EVENT_CATEGORIES
        or any(term in headline for term in FRAUD_TERMS)
        or any(term in matched_categories for term in ("fraud", "accounting", "auditor", "enforcement"))
    ):
        return "EXTREME_DISTRESS_OR_FRAUD"
    if (
        category in EXTREME_DISTRESS_EVENT_CATEGORIES
        or any(term in headline for term in EXTREME_DISTRESS_ONLY_TERMS)
        or any(term in matched_categories for term in ("bankruptcy", "insolvency", "default", "delisting", "going_concern"))
    ):
        return "EXTREME_DISTRESS"
    if category in REVERSIBLE_EVENT_CATEGORIES:
        return "REVERSIBLE_BAD_NEWS"
    if category in SERIOUS_AMBIGUOUS_EVENT_CATEGORIES:
        return "SERIOUS_BUT_AMBIGUOUS"
    return "UNKNOWN_OR_INSUFFICIENT_EVIDENCE"


def _policy_variant_spec(policy_name: str) -> dict[str, Any]:
    for spec in CATASTROPHIC_POLICY_VARIANTS:
        if spec["policy_name"] == policy_name:
            return dict(spec)
    raise ValueError(f"unknown catastrophic policy variant: {policy_name}")


def _policy_variant_blocks_candidate(row: Mapping[str, Any], policy_name: str) -> tuple[bool, str, str]:
    spec = _policy_variant_spec(policy_name)
    headline = _headline_text(row).lower()
    severity_group = _severity_group_for_candidate(row)
    if spec.get("block_groups") and severity_group in set(spec["block_groups"]):
        return True, severity_group, f"severity_group={severity_group}"
    for term in spec.get("heuristic_terms", ()):
        if term in headline:
            if policy_name == "SOFT_RISK_REDUCE":
                return True, "SERIOUS_BUT_AMBIGUOUS", f"soft_risk_term={term}"
            return True, severity_group, f"severe_loss_heuristic_term={term}"
    return False, severity_group, "ALLOW_OR_REPORT_SEPARATELY"


def apply_catastrophic_policy_variant_to_candidates(
    candidate_rows: Sequence[Mapping[str, Any]],
    policy_name: str,
) -> dict[str, Any]:
    spec = _policy_variant_spec(policy_name)
    filtered_candidates: list[dict[str, Any]] = []
    blocked_candidates: list[dict[str, Any]] = []
    unknown_candidates: list[dict[str, Any]] = []
    proposed_soft_risk_candidates: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        blocked, severity_group, reason = _policy_variant_blocks_candidate(candidate, policy_name)
        is_unknown = severity_group == "UNKNOWN_OR_INSUFFICIENT_EVIDENCE"
        category = _event_category_for_candidate(candidate)
        enriched = {
            **dict(candidate),
            "policy_name": policy_name,
            "catastrophic_policy_variant_action": "PROPOSE_SIZE_REDUCTION" if policy_name == "SOFT_RISK_REDUCE" and blocked else ("EXCLUDE_FROM_RESEARCH_VARIANT" if blocked else "KEEP"),
            "catastrophic_policy_variant_reason": reason,
            "event_category_research": category,
            "severity_group": severity_group,
            "unknown_or_insufficient_evidence": is_unknown,
            "paper_trading_enabled": False,
            "live_trading_enabled": False,
            "validation_passed": False,
            "final_validation_status": "NOT_FINAL_VALIDATION",
        }
        if policy_name == "SOFT_RISK_REDUCE":
            filtered_candidates.append(enriched)
            if blocked:
                proposed_soft_risk_candidates.append(enriched)
        elif blocked:
            blocked_candidates.append(enriched)
        else:
            filtered_candidates.append(enriched)
        if is_unknown:
            unknown_candidates.append(enriched)
    return {
        "policy_name": policy_name,
        "variant_name": spec["variant_name"],
        "policy_stage": spec["policy_stage"],
        "filtered_candidates": filtered_candidates,
        "blocked_candidates": blocked_candidates,
        "unknown_candidates": unknown_candidates,
        "proposed_soft_risk_candidates": proposed_soft_risk_candidates,
        "filter_audit": {
            "schema_name": "catastrophic_veto_policy_variant_filter_audit",
            "schema_version": 1,
            "policy_name": policy_name,
            "variant_name": spec["variant_name"],
            "policy_stage": spec["policy_stage"],
            "candidate_count_before": len(candidate_rows),
            "candidate_count_after": len(filtered_candidates),
            "blocked_candidate_count": len(blocked_candidates),
            "unknown_evidence_candidate_count": len(unknown_candidates),
            "proposed_soft_risk_reduce_candidate_count": len(proposed_soft_risk_candidates),
            "unknown_evidence_policy": "REPORT_SEPARATELY_NOT_APPROVED_FOR_PAPER_LIVE",
            "research_only": True,
            "paper_trading_enabled": False,
            "live_trading_enabled": False,
            "validation_passed": False,
            "final_validation_status": "NOT_FINAL_VALIDATION",
        },
    }


def _category_attribution_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("event_category_research", "uncategorized")), str(row.get("severity_group", "UNKNOWN_OR_INSUFFICIENT_EVIDENCE")))
        groups.setdefault(key, []).append(row)
    output = []
    for (category, severity), group_rows in sorted(groups.items()):
        returns = [value for value in (_catastrophic_trade_return(row) for row in group_rows) if value is not None]
        unavailable_count = len(group_rows) - len(returns)
        best = max(group_rows, key=lambda row: _catastrophic_trade_return(row) if _catastrophic_trade_return(row) is not None else -math.inf)
        worst = min(group_rows, key=lambda row: _catastrophic_trade_return(row) if _catastrophic_trade_return(row) is not None else math.inf)
        output.append({
            "event_category_research": category,
            "severity_group": severity,
            "candidate_count": len(group_rows),
            "removed_trade_count": len(group_rows),
            "mean_removed_trade_return": mean(returns) if returns else "UNAVAILABLE_OUTCOME",
            "median_removed_trade_return": median(returns) if returns else "UNAVAILABLE_OUTCOME",
            "total_removed_pnl_or_return": sum(returns) if returns else "UNAVAILABLE_OUTCOME",
            "positive_removed_trade_count": sum(value > 0 for value in returns),
            "negative_removed_trade_count": sum(value < 0 for value in returns),
            "severe_loss_count": sum(row.get("bounceback_label") == "SEVERE_LOSS" for row in group_rows),
            "strong_bounceback_count": sum(row.get("bounceback_label") == "BOUNCED_BACK_STRONGLY" for row in group_rows),
            "weak_bounceback_count": sum(row.get("bounceback_label") == "BOUNCED_BACK_WEAKLY" for row in group_rows),
            "unavailable_outcome_count": unavailable_count,
            "best_removed_trade": best.get("trade_id", best.get("candidate_id", "UNKNOWN")),
            "worst_removed_trade": worst.get("trade_id", worst.get("candidate_id", "UNKNOWN")),
        })
    return output


def _metric_delta(base: Mapping[str, Any], candidate: Mapping[str, Any], *names: str) -> float | str:
    base_value = _metric(base, *names)
    candidate_value = _metric(candidate, *names)
    if base_value is None or candidate_value is None:
        return "UNAVAILABLE_INPUT"
    return candidate_value - base_value


def _strict_veto_breadth_diagnostic(
    replay: Mapping[str, Any],
    removed_rows: Sequence[Mapping[str, Any]],
    policy_mode_counts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    risk_metrics = replay.get("risk_metrics", {}) if isinstance(replay.get("risk_metrics", {}), Mapping) else {}
    base_metrics = dict(risk_metrics.get("news_contrarian_rerank", {}) or {})
    strict_metrics = dict(risk_metrics.get("news_contrarian_rerank_catastrophic_veto", {}) or {})
    strict_delta = _metric_delta(base_metrics, strict_metrics, "total_return_decimal")
    drawdown_delta = _metric_delta(base_metrics, strict_metrics, "maximum_drawdown")
    sharpe_delta = _metric_delta(base_metrics, strict_metrics, "Sharpe_ratio", "sharpe_ratio")
    counts_by_mode = {str(row.get("policy_mode")): row for row in policy_mode_counts}
    confirmed_removed = int(counts_by_mode.get("CONFIRMED_ONLY_RESEARCH", {}).get("estimated_removed_trade_count", 0) or 0)
    manual_removed = int(counts_by_mode.get("MANUAL_REVIEW_RESEARCH", {}).get("estimated_removed_trade_count", 0) or 0)
    if strict_delta == "UNAVAILABLE_INPUT":
        status = "INSUFFICIENT_OUTCOME_DATA" if removed_rows else "NEEDS_CONFIRMED_ONLY_COMPARISON"
    elif strict_delta < 0 and (drawdown_delta == "UNAVAILABLE_INPUT" or drawdown_delta >= 0):
        status = "TOO_BROAD_FOR_RETURN"
    elif drawdown_delta != "UNAVAILABLE_INPUT" and drawdown_delta > 0:
        status = "POSSIBLY_USEFUL_RISK_FILTER"
    else:
        status = "NEEDS_CONFIRMED_ONLY_COMPARISON"
    return {
        "strict_veto_removed_trade_count": len(removed_rows),
        "confirmed_only_removed_trade_count": confirmed_removed,
        "manual_review_removed_trade_count": manual_removed,
        "strict_veto_return_delta": strict_delta,
        "strict_veto_drawdown_delta": drawdown_delta,
        "strict_veto_sharpe_delta": sharpe_delta,
        "strict_veto_breadth_status": status,
        "recommended_policy_next_step": (
            "compare confirmed-only/manual-review variants and evaluate an extreme-distress-only replay proposal"
            if status in {"TOO_BROAD_FOR_RETURN", "NEEDS_CONFIRMED_ONLY_COMPARISON"}
            else "review category-level severe-loss concentration before narrowing policy"
        ),
    }


def _catastrophic_veto_extreme_only_policy_proposal() -> dict[str, Any]:
    return {
        "schema_name": "catastrophic_veto_extreme_only_policy_proposal",
        "schema_version": 1,
        "status": "PROPOSED_NOT_REPLAYED",
        "policy_name": "EXTREME_DISTRESS_ONLY_RESEARCH",
        "policy_stage": "PROPOSED_NOT_REPLAYED",
        "blocks_categories": [
            "bankruptcy",
            "insolvency",
            "liquidation",
            "administration",
            "default",
            "going_concern_warning",
            "delisting",
            "trading_suspension",
            "emergency_rescue_financing",
        ],
        "manual_review_categories": [
            "fraud_or_accounting_irregularity",
            "major_litigation_existential",
            "regulatory_enforcement_or_criminal_probe",
            "distressed_dilution_or_deep_discount_raise",
        ],
        "does_not_block_categories": [
            "earnings_miss",
            "guidance_cut",
            "analyst_downgrade",
            "operational_issue",
            "macro_or_sector",
        ],
        "unknown_evidence_policy": "REPORT_SEPARATELY_NOT_APPROVED_FOR_PAPER_LIVE",
        "paper_trading_allowed": False,
        "live_trading_allowed": False,
        "requires_future_replay": True,
        "requires_manual_review": True,
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "warnings": [
            "Proposal only; no replay variant or strategy behavior has changed.",
            "Unknown evidence is not approved for paper/live trading.",
        ],
    }


def _catastrophic_veto_bounceback_artifacts(
    rows: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    removed_trade_rows: Sequence[Mapping[str, Any]],
    blocked_candidate_rows: Sequence[Mapping[str, Any]],
    full_replay_report: Mapping[str, Any],
    policy_mode_counts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidates_by_id = {
        str(row.get("candidate_id")): row
        for row in rows
        if row.get("candidate_id") not in {None, ""}
    }
    enriched_removed = []
    for trade in removed_trade_rows:
        candidate = candidates_by_id.get(str(trade.get("candidate_id")), {})
        merged = {**dict(candidate), **dict(trade)}
        category = _event_category_for_candidate(merged)
        severity_group = _severity_group_for_candidate(merged)
        label = _bounceback_label(merged)
        enriched_removed.append({
            **merged,
            "event_category_research": category,
            "severity_group": severity_group,
            "bounceback_label": label,
            "removed_trade_return": _catastrophic_trade_return(merged) if _catastrophic_trade_return(merged) is not None else "UNAVAILABLE_OUTCOME",
            "research_only": True,
        })
    by_category = _category_attribution_rows(enriched_removed)
    returns = [value for value in (_catastrophic_trade_return(row) for row in enriched_removed) if value is not None]
    keyword_summary = {
        "available": bool(enriched_removed),
        "distress_removed_trade_count": sum(int(row.get("distress_score", 0) or 0) > 0 for row in enriched_removed),
        "litigation_removed_trade_count": sum(int(row.get("litigation_score", 0) or 0) > 0 for row in enriched_removed),
        "dilution_removed_trade_count": sum(int(row.get("dilution_score", 0) or 0) > 0 for row in enriched_removed),
        "source": "headline keyword scores when present on candidate rows; otherwise recomputation is not inferred",
    }
    extreme_rows = [row for row in enriched_removed if row["severity_group"] == "EXTREME_DISTRESS"]
    reversible_rows = [row for row in enriched_removed if row["severity_group"] == "REVERSIBLE_BAD_NEWS"]
    winners = sorted(
        [row for row in enriched_removed if _catastrophic_trade_return(row) is not None],
        key=lambda row: _catastrophic_trade_return(row) or 0.0,
        reverse=True,
    )[:10]
    losers = sorted(
        [row for row in enriched_removed if _catastrophic_trade_return(row) is not None],
        key=lambda row: _catastrophic_trade_return(row) or 0.0,
    )[:10]
    diagnostic = _strict_veto_breadth_diagnostic(replay, enriched_removed, policy_mode_counts)
    report = {
        "schema_name": "catastrophic_veto_bounceback_report",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "AVAILABLE" if enriched_removed else "UNAVAILABLE_INPUT",
        "base_strategy": "news_contrarian_rerank",
        "veto_strategy": "news_contrarian_rerank_catastrophic_veto",
        "removed_trade_count": len(enriched_removed),
        "blocked_candidate_count": len(blocked_candidate_rows),
        "candidate_count_before_veto": full_replay_report.get("candidate_count_before_veto", len(rows)),
        "candidate_count_after_veto": full_replay_report.get("candidate_count_after_veto", "UNAVAILABLE_INPUT"),
        "analysis_scope": "removed news_contrarian_rerank trades from research-only catastrophic-veto attribution; no replay recomputation",
        "bounceback_definition": {
            "BOUNCED_BACK_STRONGLY": "removed trade return > +10%",
            "BOUNCED_BACK_WEAKLY": "removed trade return between 0% and +10%",
            "DID_NOT_BOUNCE": "removed trade return between -10% and 0%",
            "SEVERE_LOSS": "removed trade return <= -10%",
            "UNAVAILABLE_OUTCOME": "required return fields unavailable",
        },
        "lookahead_windows": "uses only outcome fields already present in trade ledgers; no new lookahead windows computed",
        "category_summary": by_category,
        "keyword_summary": keyword_summary,
        "extreme_distress_summary": {
            "removed_trade_count": len(extreme_rows),
            "severe_loss_count": sum(row["bounceback_label"] == "SEVERE_LOSS" for row in extreme_rows),
            "strong_bounceback_count": sum(row["bounceback_label"] == "BOUNCED_BACK_STRONGLY" for row in extreme_rows),
        },
        "reversible_bad_news_summary": {
            "removed_trade_count": len(reversible_rows),
            "severe_loss_count": sum(row["bounceback_label"] == "SEVERE_LOSS" for row in reversible_rows),
            "strong_bounceback_count": sum(row["bounceback_label"] == "BOUNCED_BACK_STRONGLY" for row in reversible_rows),
        },
        "veto_breadth_diagnostic": diagnostic,
        "top_removed_winners": [
            {"trade_id": row.get("trade_id", "UNKNOWN"), "candidate_id": row.get("candidate_id", "UNKNOWN"), "removed_trade_return": _catastrophic_trade_return(row), "event_category_research": row.get("event_category_research")}
            for row in winners
        ],
        "top_removed_losers": [
            {"trade_id": row.get("trade_id", "UNKNOWN"), "candidate_id": row.get("candidate_id", "UNKNOWN"), "removed_trade_return": _catastrophic_trade_return(row), "event_category_research": row.get("event_category_research")}
            for row in losers
        ],
        "warnings": [
            "Research-only attribution; no base replay mechanics were recomputed.",
            "Unavailable trade outcomes are reported as UNAVAILABLE_OUTCOME.",
        ],
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }
    examples = [
        {
            "trade_id": row.get("trade_id", "UNKNOWN"),
            "candidate_id": row.get("candidate_id", "UNKNOWN"),
            "symbol": row.get("symbol", "UNKNOWN"),
            "event_category_research": row.get("event_category_research", "uncategorized"),
            "severity_group": row.get("severity_group", "UNKNOWN_OR_INSUFFICIENT_EVIDENCE"),
            "bounceback_label": row.get("bounceback_label", "UNAVAILABLE_OUTCOME"),
            "removed_trade_return": row.get("removed_trade_return", "UNAVAILABLE_OUTCOME"),
            "headline_text": _headline_text(row),
        }
        for row in enriched_removed[:100]
    ]
    return report, by_category, examples, _catastrophic_veto_extreme_only_policy_proposal()


def _metric_value(metrics: Mapping[str, Any], *names: str) -> Any:
    value = _metric(metrics, *names)
    return value if value is not None else "UNAVAILABLE_INPUT"


def _trade_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(_mapping_first(row, "candidate_id", "trade_id", "row_id") or ""),
        str(_mapping_first(row, "symbol", "ticker") or ""),
        str(_mapping_first(row, "entry_date", "entry_timestamp", "open_date") or ""),
    )


def _policy_variant_trade_rows(
    *,
    policy_name: str,
    variant_name: str,
    base_trades: Sequence[Mapping[str, Any]],
    variant_trades: Sequence[Mapping[str, Any]],
    candidates_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    variant_keys = {_trade_key(row) for row in variant_trades}
    removed = [dict(row) for row in base_trades if _trade_key(row) not in variant_keys]
    output = []
    for trade in removed:
        candidate = candidates_by_id.get(str(trade.get("candidate_id")), {})
        merged = {**dict(candidate), **trade}
        output.append({
            "policy_name": policy_name,
            "variant_name": variant_name,
            "trade_id": _mapping_first(merged, "trade_id", "id", "row_id") or "UNKNOWN",
            "candidate_id": _mapping_first(merged, "candidate_id", "trade_id", "row_id") or "UNKNOWN",
            "symbol": _mapping_first(merged, "symbol", "ticker") or "UNKNOWN",
            "entry_date": _mapping_first(merged, "entry_date", "entry_timestamp", "open_date") or "UNAVAILABLE_INPUT",
            "exit_date": _mapping_first(merged, "exit_date", "exit_timestamp", "close_date") or "UNAVAILABLE_INPUT",
            "headline_text": _headline_text(merged),
            "event_category_research": _event_category_for_candidate(merged),
            "severity_group": _severity_group_for_candidate(merged),
            "removed_trade_return": _catastrophic_trade_return(merged) if _catastrophic_trade_return(merged) is not None else "UNAVAILABLE_INPUT",
            "bounceback_label": _bounceback_label(merged),
            "research_only": True,
            "paper_trading_enabled": False,
            "live_trading_enabled": False,
        })
    return output


def _policy_variant_examples(
    policy_name: str,
    removed_rows: Sequence[Mapping[str, Any]],
    allowed_trades: Sequence[Mapping[str, Any]],
    candidates_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    def example(row: Mapping[str, Any], example_type: str, reason: str) -> dict[str, Any]:
        candidate = candidates_by_id.get(str(row.get("candidate_id")), {})
        merged = {**dict(candidate), **dict(row)}
        return {
            "policy_name": policy_name,
            "example_type": example_type,
            "trade_id": _mapping_first(merged, "trade_id", "id", "row_id") or "UNKNOWN",
            "candidate_id": _mapping_first(merged, "candidate_id", "trade_id", "row_id") or "UNKNOWN",
            "symbol": _mapping_first(merged, "symbol", "ticker") or "UNKNOWN",
            "headline_text": _headline_text(merged),
            "event_category_research": _event_category_for_candidate(merged),
            "severity_group": _severity_group_for_candidate(merged),
            "removed_trade_return": _catastrophic_trade_return(merged) if _catastrophic_trade_return(merged) is not None else "UNAVAILABLE_INPUT",
            "bounceback_label": _bounceback_label(merged),
            "reason": reason,
        }

    removed_with_returns = [row for row in removed_rows if _catastrophic_trade_return(row) is not None]
    winners = sorted([row for row in removed_with_returns if (_catastrophic_trade_return(row) or 0.0) > 0], key=lambda row: _catastrophic_trade_return(row) or 0.0, reverse=True)[:3]
    losers = sorted([row for row in removed_with_returns if (_catastrophic_trade_return(row) or 0.0) <= 0], key=lambda row: _catastrophic_trade_return(row) or 0.0)[:3]
    severe = [row for row in losers if _bounceback_label(row) == "SEVERE_LOSS"][:3]
    allowed_with_returns = [row for row in allowed_trades if _catastrophic_trade_return(row) is not None]
    allowed_winners = sorted([row for row in allowed_with_returns if (_catastrophic_trade_return(row) or 0.0) > 0], key=lambda row: _catastrophic_trade_return(row) or 0.0, reverse=True)[:2]
    allowed_losers = sorted([row for row in allowed_with_returns if (_catastrophic_trade_return(row) or 0.0) <= 0], key=lambda row: _catastrophic_trade_return(row) or 0.0)[:2]
    rows = []
    rows.extend(example(row, "blocked_winner", "bounceback winner accidentally removed") for row in winners)
    rows.extend(example(row, "blocked_loser", "losing trade removed") for row in losers)
    rows.extend(example(row, "allowed_winner", "winner remained tradable") for row in allowed_winners)
    rows.extend(example(row, "allowed_loser", "loser remained tradable") for row in allowed_losers)
    rows.extend(example(row, "top_severe_loss_avoided", "severe loss removed") for row in severe)
    rows.extend(example(row, "top_bounceback_winner_accidentally_removed", "strong bounceback removed") for row in winners if _bounceback_label(row) == "BOUNCED_BACK_STRONGLY")
    return rows


def _catastrophic_policy_variant_artifacts(
    rows: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    strict_bounceback_report: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    risk_metrics = replay.get("risk_metrics", {}) if isinstance(replay.get("risk_metrics", {}), Mapping) else {}
    daily_equity = replay.get("daily_equity", {}) if isinstance(replay.get("daily_equity", {}), Mapping) else {}
    extra_metadata = replay.get("extra_research_variant_metadata", {}) if isinstance(replay.get("extra_research_variant_metadata", {}), Mapping) else {}
    all_trades = [dict(row) for row in replay.get("trade_ledger", []) or []]
    base_variant = "news_contrarian_rerank"
    base_trades = [row for row in all_trades if str(row.get("strategy_variant")) == base_variant]
    base_metrics = dict(risk_metrics.get(base_variant, {}) or {})
    candidates_by_id = {
        str(row.get("candidate_id")): row
        for row in rows
        if row.get("candidate_id") not in {None, ""}
    }

    counts: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []
    bounceback_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    frontier_rows: list[dict[str, Any]] = []
    comparison_policies: list[dict[str, Any]] = []

    for spec in CATASTROPHIC_POLICY_VARIANTS:
        policy_name = str(spec["policy_name"])
        variant_name = str(spec["variant_name"])
        filter_result = apply_catastrophic_policy_variant_to_candidates(rows, policy_name)
        blocked_candidate_rows = list(filter_result["blocked_candidates"])
        full_replay_computed = (
            spec["policy_stage"] == "FULL_REPLAY_RESEARCH"
            and variant_name in risk_metrics
            and variant_name in daily_equity
            and variant_name in extra_metadata
        )
        variant_metrics = dict(risk_metrics.get(variant_name, {}) or {}) if full_replay_computed else {}
        variant_trades = [row for row in all_trades if str(row.get("strategy_variant")) == variant_name] if full_replay_computed else []
        policy_removed_rows = _policy_variant_trade_rows(
            policy_name=policy_name,
            variant_name=variant_name,
            base_trades=base_trades,
            variant_trades=variant_trades,
            candidates_by_id=candidates_by_id,
        ) if full_replay_computed else []
        if not full_replay_computed and spec["policy_stage"] == "COUNT_ONLY_PROPOSAL":
            policy_removed_rows = []
        removed_rows.extend(policy_removed_rows)
        removed_returns = [value for value in (_catastrophic_trade_return(row) for row in policy_removed_rows) if value is not None]
        positive_removed = sum(value > 0 for value in removed_returns)
        negative_removed = sum(value < 0 for value in removed_returns)
        strong_bounceback = sum(row.get("bounceback_label") == "BOUNCED_BACK_STRONGLY" for row in policy_removed_rows)
        severe_loss = sum(row.get("bounceback_label") == "SEVERE_LOSS" for row in policy_removed_rows)
        return_delta = _metric_delta(base_metrics, variant_metrics, "total_return_decimal") if full_replay_computed else "UNAVAILABLE_INPUT"
        drawdown_delta = _metric_delta(base_metrics, variant_metrics, "maximum_drawdown") if full_replay_computed else "UNAVAILABLE_INPUT"
        sharpe_delta = _metric_delta(base_metrics, variant_metrics, "Sharpe_ratio", "sharpe_ratio") if full_replay_computed else "UNAVAILABLE_INPUT"
        return_loss_penalty = abs(return_delta) if isinstance(return_delta, (int, float)) and return_delta < 0 else 0.0
        drawdown_improvement = drawdown_delta if isinstance(drawdown_delta, (int, float)) and drawdown_delta > 0 else 0.0
        risk_benefit_score = (
            drawdown_improvement
            + severe_loss * 0.02
            - return_loss_penalty
            - strong_bounceback * 0.01
        ) if full_replay_computed else "UNAVAILABLE_INPUT"
        too_broad_score = (
            (positive_removed / max(len(policy_removed_rows), 1)) + return_loss_penalty
            if full_replay_computed and policy_removed_rows
            else "UNAVAILABLE_INPUT"
        )
        recommended = (
            "count-only size-reduction proposal; requires separate sizing-safe adapter"
            if spec["policy_stage"] == "COUNT_ONLY_PROPOSAL"
            else (
                "candidate for further review"
                if full_replay_computed and not (isinstance(return_delta, (int, float)) and return_delta < -0.05)
                else "too broad or unavailable; inspect examples before use"
            )
        )
        common = {
            "policy_name": policy_name,
            "variant_name": variant_name,
            "policy_stage": spec["policy_stage"],
            "full_replay_computed": full_replay_computed,
            "candidate_count_before": len(rows),
            "candidate_count_after": len(filter_result["filtered_candidates"]),
            "blocked_candidate_count": len(blocked_candidate_rows),
            "removed_trade_count": len(policy_removed_rows) if full_replay_computed else "UNAVAILABLE_INPUT",
            "wealth": _metric_value(variant_metrics, "ending_wealth", "ending_equity"),
            "return": _metric_value(variant_metrics, "total_return_decimal", "total_return"),
            "cagr": _metric_value(variant_metrics, "cagr"),
            "max_drawdown": _metric_value(variant_metrics, "maximum_drawdown", "max_drawdown"),
            "sharpe": _metric_value(variant_metrics, "Sharpe_ratio", "sharpe_ratio"),
            "calmar": _metric_value(variant_metrics, "Calmar_ratio", "calmar_ratio"),
            "cvar": _metric_value(variant_metrics, "cvar"),
            "trade_count": _metric_value(variant_metrics, "trade_count"),
            "return_delta_vs_original": return_delta,
            "drawdown_delta_vs_original": drawdown_delta,
            "sharpe_delta_vs_original": sharpe_delta,
            "removed_trade_mean_return": mean(removed_returns) if removed_returns else "UNAVAILABLE_INPUT",
            "removed_trade_median_return": median(removed_returns) if removed_returns else "UNAVAILABLE_INPUT",
            "removed_trade_positive_count": positive_removed if full_replay_computed else "UNAVAILABLE_INPUT",
            "removed_trade_negative_count": negative_removed if full_replay_computed else "UNAVAILABLE_INPUT",
            "removed_trade_strong_bounceback_count": strong_bounceback if full_replay_computed else "UNAVAILABLE_INPUT",
            "removed_trade_severe_loss_count": severe_loss if full_replay_computed else "UNAVAILABLE_INPUT",
            "too_broad_score": too_broad_score,
            "risk_benefit_score": risk_benefit_score,
            "recommended_next_step": recommended,
            "validation_passed": False,
            "final_validation_status": "NOT_FINAL_VALIDATION",
            "paper_trading_enabled": False,
            "live_trading_enabled": False,
            "warnings": "research-only; unknown evidence reported separately and not approved for paper/live",
        }
        counts.append({
            "policy_name": policy_name,
            "variant_name": variant_name,
            "policy_stage": spec["policy_stage"],
            "candidate_count_before": len(rows),
            "candidate_count_after": len(filter_result["filtered_candidates"]),
            "blocked_candidate_count": len(blocked_candidate_rows),
            "unknown_evidence_candidate_count": len(filter_result["unknown_candidates"]),
            "proposed_soft_risk_reduce_candidate_count": len(filter_result["proposed_soft_risk_candidates"]),
            "full_replay_computed": full_replay_computed,
            "paper_trading_enabled": False,
            "live_trading_enabled": False,
            "validation_passed": False,
            "final_validation_status": "NOT_FINAL_VALIDATION",
        })
        metrics_rows.append(common)
        bounceback_rows.append({
            "policy_name": policy_name,
            "variant_name": variant_name,
            "removed_trade_count": common["removed_trade_count"],
            "removed_trade_positive_count": common["removed_trade_positive_count"],
            "removed_trade_negative_count": common["removed_trade_negative_count"],
            "strong_bounceback_count": common["removed_trade_strong_bounceback_count"],
            "severe_loss_count": common["removed_trade_severe_loss_count"],
            "mean_removed_trade_return": common["removed_trade_mean_return"],
            "median_removed_trade_return": common["removed_trade_median_return"],
            "too_broad_score": too_broad_score,
        })
        allowed_trades = [row for row in variant_trades if full_replay_computed]
        examples.extend(_policy_variant_examples(policy_name, policy_removed_rows, allowed_trades, candidates_by_id))
        frontier_row = {
            "policy_name": policy_name,
            "variant_name": variant_name,
            "return_preservation": (1.0 + return_delta) if isinstance(return_delta, (int, float)) else "UNAVAILABLE_INPUT",
            "drawdown_improvement": drawdown_improvement if full_replay_computed else "UNAVAILABLE_INPUT",
            "sharpe_delta_vs_original": sharpe_delta,
            "severe_loss_removed_count": severe_loss if full_replay_computed else "UNAVAILABLE_INPUT",
            "bounceback_winner_removed_count": strong_bounceback if full_replay_computed else "UNAVAILABLE_INPUT",
            "risk_benefit_score": risk_benefit_score,
            "full_replay_computed": full_replay_computed,
        }
        frontier_rows.append(frontier_row)
        comparison_policies.append({**common, "warnings": [common["warnings"]]})

    scored = [row for row in frontier_rows if isinstance(row.get("risk_benefit_score"), (int, float))]
    best_balanced = max(scored, key=lambda row: (float(row["risk_benefit_score"]), str(row["policy_name"])), default={})
    return_scored = [row for row in metrics_rows if isinstance(row.get("return_delta_vs_original"), (int, float))]
    drawdown_scored = [row for row in frontier_rows if isinstance(row.get("drawdown_improvement"), (int, float))]
    best_return = max(return_scored, key=lambda row: (float(row["return_delta_vs_original"]), str(row["policy_name"])), default={})
    best_drawdown = max(drawdown_scored, key=lambda row: (float(row["drawdown_improvement"]), str(row["policy_name"])), default={})
    policies_too_broad = [
        row["policy_name"]
        for row in metrics_rows
        if isinstance(row.get("return_delta_vs_original"), (int, float)) and row["return_delta_vs_original"] < -0.05
    ]
    policies_no_effect = [
        row["policy_name"]
        for row in metrics_rows
        if row.get("removed_trade_count") == 0 or row.get("blocked_candidate_count") == 0
    ]
    full_replay_policy_rows = [row for row in metrics_rows if row.get("full_replay_computed") is True]
    no_effect_frontier = bool(full_replay_policy_rows) and all(
        row.get("blocked_candidate_count") == 0
        and row.get("removed_trade_count") == 0
        and row.get("return_delta_vs_original") == 0
        and row.get("drawdown_delta_vs_original") == 0
        for row in full_replay_policy_rows
    )
    comparison = {
        "schema_name": "catastrophic_veto_policy_variant_comparison",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "RESEARCH_ONLY_POLICY_VARIANTS",
        "strict_veto_breadth_status": dict(strict_bounceback_report.get("veto_breadth_diagnostic", {}) or {}).get("strict_veto_breadth_status", "UNAVAILABLE_INPUT"),
        "policies": comparison_policies,
        "policy_names": [str(spec["policy_name"]) for spec in CATASTROPHIC_POLICY_VARIANTS],
        "full_replay_variants": [row["variant_name"] for row in counts if row["full_replay_computed"]],
        "count_only_variants": [row["variant_name"] for row in counts if not row["full_replay_computed"]],
        "soft_risk_reduce_status": "COUNT_ONLY_PROPOSAL",
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "warnings": [
            "Research-only policy variants; not enforced in current strategy, paper trading, or live trading.",
            "SOFT_RISK_REDUCE is count-only because the safe adapter filters candidates but does not adjust position sizing.",
        ],
    }
    frontier_report = {
        "schema_name": "catastrophic_veto_policy_frontier_report",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "NO_EFFECT_FRONTIER" if no_effect_frontier else "RESEARCH_ONLY_DIAGNOSTIC",
        "frontier_status": "NO_EFFECT_FRONTIER" if no_effect_frontier else "RESEARCH_ONLY_DIAGNOSTIC",
        "scoring_formula": "drawdown_improvement + severe_loss_removed_bonus - return_loss_penalty - bounceback_winner_removed_penalty",
        "best_return_preserving_policy": "UNAVAILABLE_NO_EFFECT" if no_effect_frontier else best_return.get("policy_name", "UNAVAILABLE_INPUT"),
        "best_drawdown_reduction_policy": "UNAVAILABLE_NO_EFFECT" if no_effect_frontier else best_drawdown.get("policy_name", "UNAVAILABLE_INPUT"),
        "best_balanced_policy": "UNAVAILABLE_NO_EFFECT" if no_effect_frontier else best_balanced.get("policy_name", "UNAVAILABLE_INPUT"),
        "policies_too_broad_for_return": policies_too_broad,
        "policies_with_no_effect": policies_no_effect,
        "policies_requiring_more_taxonomy": [
            row["policy_name"]
            for row in counts
            if int(row.get("unknown_evidence_candidate_count", 0) or 0) > 0
        ],
        "recommended_next_step": (
            "inspect loser-vs-bounceback cases and improve taxonomy/source evidence"
            if no_effect_frontier
            else "review policy frontier examples before any future policy narrowing"
        ),
        "diagnostic_only": True,
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "warnings": ["Frontier ranking is deterministic and diagnostic only; it is not model selection or final validation."],
    }
    return comparison, counts, metrics_rows, removed_rows, bounceback_rows, frontier_report, frontier_rows, examples


FILING_FORM_PATTERNS: tuple[tuple[str, str], ...] = (
    ("NT_10_Q", r"\bnt\s*-?\s*10\s*-?\s*q\b"),
    ("10_Q", r"\b10\s*-?\s*q\b"),
    ("8_K", r"\b8\s*-?\s*k\b"),
)


def _filing_forms_detected(text: str) -> list[str]:
    normalized = text.lower()
    return [name for name, pattern in FILING_FORM_PATTERNS if re.search(pattern, normalized)]


def _is_generic_filing_headline(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if not normalized:
        return False
    has_form = bool(_filing_forms_detected(normalized))
    generic_terms = ("filed by", "files form", "form 10-q", "10-q filed", "8-k filed", "filed form")
    risk_terms = ("late", "nt 10-q", "going concern", "default", "delisting", "fraud", "investigation", "bankruptcy", "restatement")
    return has_form and any(term in normalized for term in generic_terms) and not any(term in normalized for term in risk_terms)


def _case_value(row: Mapping[str, Any], *keys: str) -> Any:
    value = _mapping_first(row, *keys)
    return value if value is not None else "UNAVAILABLE_INPUT"


def _case_keyword_scores(row: Mapping[str, Any]) -> dict[str, int]:
    headline = _headline_text(row).lower()
    return {
        score_name: int(row.get(score_name, 0) or 0) if _number(row.get(score_name)) is not None else sum(1 for term in terms if term in headline)
        for score_name, terms in KEYWORD_BASELINE_RULES
    }


def _casebook_case(row: Mapping[str, Any], case_type: str, reason: str) -> dict[str, Any]:
    keyword_scores = _case_keyword_scores(row)
    return {
        "case_type": case_type,
        "trade_id": _case_value(row, "trade_id", "id", "row_id"),
        "candidate_id": _case_value(row, "candidate_id", "trade_id", "row_id"),
        "symbol": _case_value(row, "symbol", "ticker"),
        "decision_timestamp": _case_value(row, "decision_timestamp", "rebalance_timestamp", "feature_timestamp"),
        "entry_date": _case_value(row, "entry_date", "entry_timestamp", "open_date"),
        "exit_date": _case_value(row, "exit_date", "exit_timestamp", "close_date"),
        "headline_text": _headline_text(row) or "UNAVAILABLE_INPUT",
        "summary_text": _case_value(row, "summary_text", "summary", "body_text", "body", "content"),
        "provider": _case_value(row, "provider", "source", "news_provider"),
        "availability_timestamp": _case_value(row, "availability_timestamp", "available_at", "asof_timestamp"),
        "publication_timestamp": _case_value(row, "publication_timestamp", "published_at", "published_at_utc", "timestamp"),
        "event_category_research": _event_category_for_candidate(row),
        "severity_group": _severity_group_for_candidate(row),
        "keyword_scores": json.dumps(keyword_scores, sort_keys=True),
        "news_score": _case_value(row, "news_score", "price_plus_news_risk_probability", "news_risk_score"),
        "price_model_score": _case_value(row, "price_model_score", "price_score", "price_only_news_risk_probability"),
        "removed_trade_return": _catastrophic_trade_return(row) if _catastrophic_trade_return(row) is not None else "UNAVAILABLE_INPUT",
        "bounceback_label": _bounceback_label(row),
        "max_adverse_excursion_if_available": _case_value(row, "maximum_adverse_excursion", "max_adverse_excursion"),
        "max_favourable_excursion_if_available": _case_value(row, "maximum_favourable_excursion", "max_favourable_excursion"),
        "reason_for_selection": reason,
        "headline_is_generic_filing": _is_generic_filing_headline(_headline_text(row)),
        "filing_forms_detected": "|".join(_filing_forms_detected(_headline_text(row))) or "UNAVAILABLE_INPUT",
    }


def _rate(rows: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]) -> float | str:
    if not rows:
        return "UNAVAILABLE_INPUT"
    return sum(1 for row in rows if predicate(row)) / len(rows)


def _difference(left: Any, right: Any) -> Any:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left - right
    return "UNAVAILABLE_INPUT"


def _feature_diff_row(feature: str, loser_value: Any, winner_value: Any, interpretation: str) -> dict[str, Any]:
    diff = _difference(loser_value, winner_value)
    direction = "UNAVAILABLE_INPUT"
    if isinstance(diff, (int, float)):
        direction = "HIGHER_IN_SEVERE_LOSERS" if diff > 0 else ("HIGHER_IN_STRONG_BOUNCEBACK" if diff < 0 else "NO_DIFFERENCE")
    return {
        "feature_name": feature,
        "severe_loser_value_or_rate": loser_value,
        "strong_bounceback_value_or_rate": winner_value,
        "difference": diff,
        "direction": direction,
        "interpretation": interpretation,
    }


def _avg_keyword(rows: Sequence[Mapping[str, Any]], score_name: str) -> float | str:
    if not rows:
        return "UNAVAILABLE_INPUT"
    return mean(_case_keyword_scores(row).get(score_name, 0) for row in rows)


def _catastrophic_veto_loser_bounceback_casebook_artifacts(
    rows: Sequence[Mapping[str, Any]],
    removed_trade_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidates_by_id = {
        str(row.get("candidate_id")): row
        for row in rows
        if row.get("candidate_id") not in {None, ""}
    }
    enriched = []
    for trade in removed_trade_rows:
        candidate = candidates_by_id.get(str(trade.get("candidate_id")), {})
        merged = {**dict(candidate), **dict(trade)}
        enriched.append(merged)
    severe_losers = sorted(
        [row for row in enriched if _bounceback_label(row) == "SEVERE_LOSS"],
        key=lambda row: (_catastrophic_trade_return(row) if _catastrophic_trade_return(row) is not None else math.inf, str(row.get("trade_id", ""))),
    )[:25]
    strong_winners = sorted(
        [row for row in enriched if _bounceback_label(row) == "BOUNCED_BACK_STRONGLY"],
        key=lambda row: (-(_catastrophic_trade_return(row) if _catastrophic_trade_return(row) is not None else -math.inf), str(row.get("trade_id", ""))),
    )[:25]
    cases = [
        _casebook_case(row, "top_severe_loser", "lowest removed trade returns among strict-veto removed trades")
        for row in severe_losers
    ] + [
        _casebook_case(row, "top_strong_bounceback_winner", "highest positive removed trade returns among strict-veto removed trades")
        for row in strong_winners
    ]
    features: list[tuple[str, Callable[[Mapping[str, Any]], bool], str]] = [
        ("has_headline_text", lambda row: bool(_headline_text(row).strip()), "headline text coverage"),
        ("has_provider", lambda row: _case_value(row, "provider", "source", "news_provider") != "UNAVAILABLE_INPUT", "provider/source coverage"),
        ("has_availability_timestamp", lambda row: _case_value(row, "availability_timestamp", "available_at", "asof_timestamp") != "UNAVAILABLE_INPUT", "point-in-time availability evidence"),
        ("headline_is_generic_filing", lambda row: _is_generic_filing_headline(_headline_text(row)), "generic filing headline rate"),
        ("headline_mentions_10q", lambda row: "10-q" in _headline_text(row).lower() or "10q" in _headline_text(row).lower(), "routine 10-Q headline mention"),
        ("headline_mentions_8k", lambda row: "8-k" in _headline_text(row).lower() or "8k" in _headline_text(row).lower(), "8-K headline mention"),
        ("headline_mentions_nt_10q", lambda row: "nt 10-q" in _headline_text(row).lower() or "nt10q" in _headline_text(row).lower(), "late filing signal"),
        ("headline_mentions_going_concern", lambda row: "going concern" in _headline_text(row).lower(), "going-concern signal"),
        ("headline_mentions_default", lambda row: "default" in _headline_text(row).lower(), "default signal"),
        ("headline_mentions_delisting", lambda row: "delisting" in _headline_text(row).lower(), "delisting signal"),
        ("headline_mentions_suspension", lambda row: "suspension" in _headline_text(row).lower(), "trading-suspension signal"),
        ("headline_mentions_fraud", lambda row: "fraud" in _headline_text(row).lower(), "fraud signal"),
        ("headline_mentions_investigation", lambda row: "investigation" in _headline_text(row).lower(), "investigation signal"),
        ("headline_mentions_dilution", lambda row: "dilution" in _headline_text(row).lower() or "dilutive" in _headline_text(row).lower(), "dilution signal"),
        ("headline_mentions_offering", lambda row: "offering" in _headline_text(row).lower(), "offering signal"),
        ("headline_mentions_bankruptcy", lambda row: "bankruptcy" in _headline_text(row).lower(), "bankruptcy signal"),
    ]
    feature_diff = [
        _feature_diff_row(name, _rate(severe_losers, predicate), _rate(strong_winners, predicate), interpretation)
        for name, predicate, interpretation in features
    ]
    for name in ("event_category_research", "severity_group"):
        loser_value = Counter(_event_category_for_candidate(row) if name == "event_category_research" else _severity_group_for_candidate(row) for row in severe_losers).most_common(1)
        winner_value = Counter(_event_category_for_candidate(row) if name == "event_category_research" else _severity_group_for_candidate(row) for row in strong_winners).most_common(1)
        feature_diff.append(_feature_diff_row(
            name,
            loser_value[0][0] if loser_value else "UNAVAILABLE_INPUT",
            winner_value[0][0] if winner_value else "UNAVAILABLE_INPUT",
            "most common categorical value",
        ))
    keyword_diff = [
        _feature_diff_row(score_name, _avg_keyword(severe_losers, score_name), _avg_keyword(strong_winners, score_name), f"average {score_name}")
        for score_name, _terms in KEYWORD_BASELINE_RULES
    ]
    feature_diff.extend([
        _feature_diff_row("keyword_distress_score", _avg_keyword(severe_losers, "distress_score"), _avg_keyword(strong_winners, "distress_score"), "average distress keyword score"),
        _feature_diff_row("keyword_dilution_score", _avg_keyword(severe_losers, "dilution_score"), _avg_keyword(strong_winners, "dilution_score"), "average dilution keyword score"),
        _feature_diff_row("keyword_litigation_score", _avg_keyword(severe_losers, "litigation_score"), _avg_keyword(strong_winners, "litigation_score"), "average litigation keyword score"),
        _feature_diff_row("keyword_generic_negative_score", _avg_keyword(severe_losers, "generic_negative_score"), _avg_keyword(strong_winners, "generic_negative_score"), "average generic-negative keyword score"),
        _feature_diff_row("news_score_decile_if_available", "UNAVAILABLE_INPUT", "UNAVAILABLE_INPUT", "requires explicit decile field on case rows"),
        _feature_diff_row("price_model_score_bucket_if_available", "UNAVAILABLE_INPUT", "UNAVAILABLE_INPUT", "requires explicit bucket field on case rows"),
    ])
    generic_cases = [case for case in cases if case["headline_is_generic_filing"]]
    filing_forms = sorted({
        form
        for case in cases
        for form in str(case.get("filing_forms_detected", "")).split("|")
        if form and form != "UNAVAILABLE_INPUT"
    })
    generic_diagnostic = {
        "generic_filing_case_count": len(generic_cases),
        "generic_filing_severe_loser_count": sum(case["case_type"] == "top_severe_loser" for case in generic_cases),
        "generic_filing_strong_bounceback_count": sum(case["case_type"] == "top_strong_bounceback_winner" for case in generic_cases),
        "filing_forms_detected": filing_forms,
        "needs_filing_content_not_just_headline": bool(generic_cases),
        "recommended_next_step": "Treat routine 10-Q/8-K headlines as weak evidence unless filing body, summary, or late-filing terms add risk context.",
    }
    report = {
        "schema_name": "catastrophic_veto_loser_bounceback_casebook",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "AVAILABLE" if cases else "UNAVAILABLE_INPUT",
        "case_selection": "top severe losers and top strong bounce-back winners among strict-veto removed trades",
        "severe_loser_case_count": len(severe_losers),
        "strong_bounceback_case_count": len(strong_winners),
        "generic_filing_diagnostic": generic_diagnostic,
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "warnings": ["Research-only casebook; no replay, ranking, sizing, or execution behavior changed."],
    }
    plan = {
        "schema_name": "catastrophic_veto_taxonomy_improvement_plan",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PROPOSED",
        "proposal_only": True,
        "recommended_rule_improvements": [
            "routine 10-Q headline alone is not enough",
            "NT 10-Q / late filing may be riskier than normal 10-Q",
            "going concern text requires filing body or summary, not just generic headline",
            "distressed financing needs terms like emergency, rescue, going concern, liquidity",
            "ordinary capital raise should not equal distressed dilution",
        ],
        "generic_filing_diagnostic": generic_diagnostic,
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "finbert_readiness": "NOT_READY",
        "transformer_enabled": False,
        "warnings": ["Proposal artifact only; deterministic taxonomy rules were not changed by this plan."],
    }
    return report, cases, feature_diff, keyword_diff, plan


def _catastrophic_veto_removal_reason(
    blocked: bool,
    manual_review: bool,
    unknown: bool,
    missing_availability: bool = False,
) -> str:
    if blocked:
        return "BLOCK_CONTRARIAN_ENTRY"
    if manual_review:
        return "BLOCK_UNTIL_REVIEWED"
    if unknown:
        return "DO_NOT_TREAT_AS_SAFE"
    if missing_availability:
        return "NOT_POINT_IN_TIME_SAFE"
    return "NOT_REMOVED"


def _catastrophic_veto_removed_symbol_rows(
    removed_trade_rows: Sequence[Mapping[str, Any]],
    blocked_candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidate_counts: dict[str, int] = {}
    for candidate in blocked_candidates:
        symbol = str(candidate.get("symbol", "UNKNOWN"))
        candidate_counts[symbol] = candidate_counts.get(symbol, 0) + 1
    by_symbol: dict[str, list[Mapping[str, Any]]] = {}
    for row in removed_trade_rows:
        symbol = str(row.get("symbol", "UNKNOWN"))
        by_symbol.setdefault(symbol, []).append(row)
    output = []
    for symbol, rows in sorted(by_symbol.items()):
        pnl_values = [_as_optional_float(row.get("pnl")) for row in rows]
        return_values = [_as_optional_float(row.get("net_return")) for row in rows]
        available_pnl = [value for value in pnl_values if value is not None]
        available_returns = [value for value in return_values if value is not None]
        severities = [str(row.get("highest_severity", "UNKNOWN")) for row in rows]
        output.append(
            {
                "symbol": symbol,
                "blocked_trade_count": len(rows),
                "blocked_candidate_count": candidate_counts.get(symbol, 0),
                "matched_categories": "|".join(
                    sorted(
                        {
                            category
                            for row in rows
                            for category in str(row.get("matched_categories", "")).split("|")
                            if category
                        }
                    )
                ),
                "highest_severity": "CATASTROPHIC" if "CATASTROPHIC" in severities else (severities[0] if severities else "UNKNOWN"),
                "available_pnl_contribution": sum(available_pnl) if available_pnl else "UNAVAILABLE_INPUT",
                "available_return_contribution": sum(available_returns) if available_returns else "UNAVAILABLE_INPUT",
                "limitations": "ledger-level attribution only; portfolio path was not recomputed",
            }
        )
    return output


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() in {"", "UNAVAILABLE_INPUT", "UNKNOWN"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mapping_first(row: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _metric(payload: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = _number(payload.get(name))
        if value is not None:
            return value
    return None
