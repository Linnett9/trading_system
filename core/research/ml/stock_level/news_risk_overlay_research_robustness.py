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

from core.research.ml.stock_level.news_risk_overlay_research_catastrophic import _metric_value
from core.research.ml.stock_level.news_risk_overlay_research_audit import _top_contribution
from core.research.ml.stock_level.news_risk_overlay_research_inspection import _metric
from core.research.ml.stock_level.news_risk_overlay_research_utils import (
    _number,
    _timestamp,
)

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


__all__ = [name for name in globals() if not name.startswith("__")]
