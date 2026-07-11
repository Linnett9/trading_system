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

from core.research.ml.stock_level.news_risk_overlay import NewsRiskOverlayConfig
from core.research.ml.stock_level.news_risk_overlay_research_accounting import adverse_excursion as _adverse_excursion
from core.research.ml.stock_level.news_risk_overlay_research_audit import (
    _category_mix,
    _contrarian_suitability,
    _event_category,
    _event_category_policies,
    _filtered_group_summary,
    _mean_ci,
    _monotonicity,
    _pearson,
    _row_group_summary,
    _spearman,
)
from core.research.ml.stock_level.news_risk_overlay_research_parallel import (
    NewsRiskParallelConfig,
    parallel_config as _parallel_config,
    record_fallback as _record_fallback,
    record_parallel_phase as _record_parallel_phase,
    record_worker_failures as _record_worker_failures,
    should_parallelize as _should_parallelize,
)
from core.research.ml.stock_level.news_risk_overlay_research_replay import _daily_risk_metrics, _run_open_trade_replay
from core.research.ml.stock_level.news_risk_overlay_research_utils import (
    RETURN_COLUMNS,
    _boolish,
    _empty_score_direction_report,
    _first_numeric,
    _favourable_excursion,
    _number,
    _timestamp,
    _value_counts,
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


__all__ = [name for name in globals() if not name.startswith("__")]
