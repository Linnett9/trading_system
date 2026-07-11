from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from core.research.ml.stock_level.news_sources import (
    catastrophic_news_taxonomy_report,
    classify_catastrophic_news_rows,
)
from core.research.ml.stock_level.news_risk_overlay_research_catastrophic_utils import (
    _as_optional_float,
    _catastrophic_trade_return,
    _catastrophic_veto_removal_reason,
    _catastrophic_veto_removed_symbol_rows,
    _mapping_first,
    _metric_delta,
    _number,
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
