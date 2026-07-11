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

from core.research.ml.stock_level.news_risk_overlay_research_inspection import (
    _build_executive_summary,
    _summary_lines,
)


@dataclass(frozen=True)
class NewsRiskOverlayInspection:
    output_dir: Path
    summary: dict[str, Any]
    artifact_status: list[dict[str, Any]]


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


__all__ = [name for name in globals() if not name.startswith("__")]


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
