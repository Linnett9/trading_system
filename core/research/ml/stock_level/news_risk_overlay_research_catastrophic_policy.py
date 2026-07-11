from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from statistics import mean, median
from typing import Any, Mapping, Sequence

from core.research.ml.stock_level.news_risk_overlay_research_catastrophic_utils import (
    DISTRESSED_DILUTION_EVENT_CATEGORIES,
    DISTRESSED_DILUTION_TERMS,
    EXTREME_DISTRESS_EVENT_CATEGORIES,
    EXTREME_DISTRESS_ONLY_TERMS,
    EXTREME_DISTRESS_TERMS,
    FRAUD_EVENT_CATEGORIES,
    FRAUD_TERMS,
    SEVERE_LOSS_AVOIDANCE_TERMS,
    SOFT_RISK_REDUCE_TERMS,
    _bounceback_label,
    _catastrophic_trade_return,
    _event_category_for_candidate,
    _headline_text,
    _mapping_first,
    _metric,
    _metric_delta,
    _number,
    _rate,
    _severity_group_for_candidate,
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
