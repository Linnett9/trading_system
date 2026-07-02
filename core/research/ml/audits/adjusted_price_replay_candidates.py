from __future__ import annotations

from typing import Any

from core.research.ml.audits.adjusted_data_loading import _number
from core.research.ml.audits.adjusted_price_replay_coverage import _coverage_summary, _period_adjusted_coverage
from core.research.ml.audits.adjusted_price_replay_prices import _weighted_period_return


def _adjusted_champion_audit(
    champion_audit: dict[str, Any],
    adjusted_closes: dict[str, dict[str, float]],
    raw_closes: dict[str, dict[str, float]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    period_rows = []
    coverage_rows = []
    for row in champion_audit.get("exact_champion_replay", {}).get("period_rows", []) or []:
        if not isinstance(row, dict):
            continue
        coverage = _period_adjusted_coverage(
            row,
            adjusted_closes,
            raw_closes,
            config,
        )
        coverage_rows.append(coverage)
        if not coverage["valid_adjusted_period"]:
            continue
        adjusted_return = _weighted_period_return(
            row,
            adjusted_closes,
            raw_closes,
            config,
        )
        if adjusted_return is None:
            continue
        period_rows.append({
            **row,
            "period_return": adjusted_return * float(row.get("exposure_target") or 1.0),
            "adjusted_symbol_period_return": adjusted_return,
            "symbol_return_anomalies": [],
        })
    return (
        {
            **champion_audit,
            "exact_champion_replay": {
                **champion_audit.get("exact_champion_replay", {}),
                "period_rows": period_rows,
            },
        },
        _coverage_summary("exact_champion_replay", coverage_rows, config),
    )

def _adjusted_selected_optimizer(
    selected_optimizer: dict[str, Any],
    champion_audit: dict[str, Any],
    adjusted_champion: dict[str, Any],
    adjusted_closes: dict[str, dict[str, float]],
    raw_closes: dict[str, dict[str, float]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_period_by_date = {
        str(row.get("rebalance_date")): row
        for row in champion_audit.get("exact_champion_replay", {}).get("period_rows", []) or []
        if isinstance(row, dict)
    }
    adjusted_period_by_date = {
        str(row.get("rebalance_date")): row
        for row in adjusted_champion.get("exact_champion_replay", {}).get("period_rows", []) or []
    }
    rows = []
    coverage_rows = []
    for row in selected_optimizer.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        raw_period = raw_period_by_date.get(str(row.get("rebalance_date")))
        if not raw_period:
            continue
        exposure = _number(row.get("exposure"))
        coverage_period = {
            **raw_period,
            "exposure": exposure,
            "selected_symbols": raw_period.get("selected_symbols", []) or [],
            "target_weights": raw_period.get("target_weights", {}) or {},
        }
        coverage = _period_adjusted_coverage(
            coverage_period,
            adjusted_closes,
            raw_closes,
            config,
        )
        coverage_rows.append(coverage)
        if not coverage["valid_adjusted_period"]:
            continue
        period = adjusted_period_by_date.get(str(row.get("rebalance_date")))
        if not period:
            continue
        adjusted_return = _weighted_period_return(
            period,
            adjusted_closes,
            raw_closes,
            config,
        )
        if adjusted_return is None or exposure is None:
            continue
        cost = _number(row.get("cost")) or 0.0
        rows.append({
            **row,
            "period_return": adjusted_return,
            "net_return": adjusted_return * exposure - cost,
        })
    return (
        {**selected_optimizer, "rows": rows},
        _coverage_summary(
            "selected_bayesian_optimizer_diagnostic_policy",
            coverage_rows,
            config,
        ),
    )
