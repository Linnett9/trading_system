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

from core.research.ml.stock_level.news_risk_overlay_research_accounting import portfolio_stats as _portfolio_stats
from core.research.ml.stock_level.news_risk_overlay_research_inspection import _metric
from core.research.ml.stock_level.news_risk_overlay_research_replay import (
    _run_open_trade_replay,
)
from core.research.ml.stock_level.news_risk_overlay_research_utils import (
    RETURN_COLUMNS,
    _boolish,
    _first_numeric,
    _number,
    _timestamp,
)

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
    from core.research.ml.stock_level.news_risk_overlay_research_selection import _period_for_date

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


__all__ = [name for name in globals() if not name.startswith("__")]
