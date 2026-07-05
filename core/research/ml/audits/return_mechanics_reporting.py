from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "candidate_name",
        "available",
        "period_source",
        "exact_period_path",
        "start_date",
        "end_date",
        "number_of_periods",
        "total_return",
        "reported_total_return",
        "total_return_delta_vs_reported",
        "compounded_return",
        "arithmetic_mean_period_return",
        "geometric_mean_period_return",
        "annualized_return",
        "max_drawdown",
        "reported_max_drawdown",
        "sharpe",
        "sortino",
        "calmar",
        "turnover",
        "costs",
        "exposure_mean",
        "exposure_median",
        "exposure_min",
        "exposure_max",
        "largest_positive_period_contribution",
        "largest_negative_period_contribution",
        "forecast_source",
        "transaction_cost_bps",
        "red_flags",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                name: (
                    json.dumps(row.get(name))
                    if isinstance(row.get(name), (list, dict))
                    else row.get(name)
                )
                for name in fieldnames
            })


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Return Mechanics Audit",
        "",
        "Research only. Trading impact: none. Production validated: false.",
        "",
        "## Mechanics",
        "",
        f"- Aggregation: {payload['mechanics']['aggregation_method']}",
        f"- Return unit inference: {payload['mechanics']['return_unit_inference']}",
        f"- Transaction costs: {payload['mechanics']['transaction_cost_method']}",
        f"- Turnover: {payload['mechanics']['turnover_method']}",
        f"- Meta holdout rows: {payload['mechanics']['meta_holdout_row_count']}",
        f"- Meta holdout rebalance dates: {payload['mechanics']['meta_holdout_unique_rebalance_dates']}",
        "",
        "## Candidate Summary",
        "",
        "|candidate|periods|total return|reported total|drawdown|Sharpe|turnover|costs|mean exposure|largest + period|largest - period|flags|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["candidates"]:
        lines.append(
            "|{name}|{periods}|{total}|{reported}|{drawdown}|{sharpe}|"
            "{turnover}|{costs}|{exposure}|{positive}|{negative}|{flags}|".format(
                name=row["candidate_name"],
                periods=row.get("number_of_periods", ""),
                total=_fmt(row.get("total_return")),
                reported=_fmt(row.get("reported_total_return")),
                drawdown=_fmt(row.get("max_drawdown")),
                sharpe=_fmt(row.get("sharpe")),
                turnover=_fmt(row.get("turnover")),
                costs=_fmt(row.get("costs")),
                exposure=_fmt(row.get("exposure_mean")),
                positive=_fmt(row.get("largest_positive_period_contribution")),
                negative=_fmt(row.get("largest_negative_period_contribution")),
                flags=", ".join(row.get("red_flags", [])) or "",
            )
        )
    lines.extend([
        "",
        "## Concentration",
        "",
    ])
    for row in payload["candidates"]:
        if not row.get("available"):
            continue
        top_1 = row.get("return_concentration", {}).get("top_1", {})
        top_20 = row.get("return_concentration", {}).get("top_20", {})
        lines.append(
            "- {name}: top 1 share of positive period returns={top1}, "
            "top 20 share={top20}".format(
                name=row["candidate_name"],
                top1=_fmt(top_1.get("share_of_positive_period_returns")),
                top20=_fmt(top_20.get("share_of_positive_period_returns")),
            )
        )
    lines.extend([
        "",
        "## Champion Baseline",
        "",
        f"- Equals always_full_exposure: {payload['champion_baseline_audit']['champion_baseline_equals_always_full_exposure']}",
        f"- Champion config: {payload['champion_baseline_audit'].get('champion_config_path')}",
        f"- Represents full YAML replay: {payload['champion_baseline_audit']['represents_full_frozen_champion_yaml_replay']}",
        f"- Note: {payload['champion_baseline_audit']['should_have_turnover_costs_flag']}",
        "",
        "## Leakage Check",
        "",
        f"- Optimizer protocol: {payload['leakage_check']['optimizer_selection_protocol']}",
        f"- Out-of-fold optimizer selection: {payload['leakage_check']['optimizer_selects_parameters_on_out_of_fold_data']}",
        f"- Actual columns used as forecasts: {payload['leakage_check']['actual_columns_used_as_forecasts']}",
        "",
        "## Red Flags",
        "",
    ])
    if payload["red_flags"]:
        lines.extend(f"- {flag}" for flag in payload["red_flags"])
    else:
        lines.append("- none")
    lines.extend(["", "## Top Dates", ""])
    for row in payload["candidates"]:
        if not row.get("available"):
            continue
        lines.append(f"### {row['candidate_name']}")
        lines.append("")
        lines.append("|date|net return|baseline return|exposure|")
        lines.append("|---|---:|---:|---:|")
        for record in row.get("top_20_contributing_rebalance_dates", [])[:20]:
            lines.append(
                "|{date}|{net}|{base}|{exposure}|".format(
                    date=record["date"],
                    net=_fmt(record["net_return"]),
                    base=_fmt(record["baseline_return"]),
                    exposure=_fmt(record["exposure"]),
                )
            )
        lines.append("")
    lines.append("Research only. Trading impact: none. Production validated: false.")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    number = _number_for_format(value)
    return "" if number is None else f"{number:.6f}"


def _number_for_format(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
