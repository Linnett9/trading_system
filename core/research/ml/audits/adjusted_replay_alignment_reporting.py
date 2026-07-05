from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from core.research.ml.audits.adjusted_data_comparison import NOTICE
from core.research.ml.audits.adjusted_replay_alignment_math import _fmt


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    fieldnames = [
        "candidate",
        "rebalance_date",
        "outcome_end_date",
        "adjusted_outcome_end_date",
        "symbol",
        "raw_return",
        "adjusted_return",
        "return_delta",
        "raw_close_start",
        "raw_close_end",
        "adjusted_close_start",
        "adjusted_close_end",
        "adjustment_ratio_start",
        "adjustment_ratio_end",
        "adjustment_ratio_change",
        "adjustment_ratio_split_like_factor",
        "expected_adjusted_return_from_ratio",
        "adjusted_return_matches_ratio",
        "exposure",
        "adjusted_exposure",
        "raw_candidate_net_return",
        "adjusted_candidate_net_return",
        "candidate_net_return_delta",
        "included_in_canonical_replay",
        "adjusted_included_in_canonical_replay",
        "missing_adjusted_prices",
        "date_misalignment",
        "symbol_mismatch",
        "exposure_mismatch",
        "label_window_mismatch",
        "non_overlap_mismatch",
        "return_delta_above_threshold",
        "candidate_net_return_delta_above_threshold",
        "adjustment_ratio_jump",
        "unexplained_adjusted_delta",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload.get("rows", []) or []:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _markdown(payload: dict[str, Any]) -> str:
    alignment = payload.get("alignment", {})
    lines = [
        "# Adjusted Replay Alignment Audit",
        "",
        NOTICE,
        "",
        f"Aligned correctly: {alignment.get('aligned_correctly')}",
        f"Explanation verdict: {alignment.get('explanation_verdict')}",
        f"Missing adjusted price rows: {alignment.get('missing_adjusted_price_row_count')}",
        f"Invalid adjusted periods: {alignment.get('invalid_adjusted_period_count')}",
        f"Valid adjusted periods: {alignment.get('valid_adjusted_period_count')}",
        "Valid adjusted independent periods: "
        f"{alignment.get('valid_adjusted_independent_period_count')}",
        f"Date misalignment rows: {alignment.get('date_misalignment_row_count')}",
        f"Symbol mismatch rows: {alignment.get('symbol_mismatch_row_count')}",
        f"Large return-delta rows: {alignment.get('large_return_delta_row_count')}",
        "Large candidate net-return delta rows: "
        f"{alignment.get('large_candidate_net_return_delta_row_count')}",
        f"Adjustment-ratio jump rows: {alignment.get('adjustment_ratio_jump_row_count')}",
        "",
        "|candidate|rows|coverage|valid periods|invalid periods|missing adjusted|date mismatch|symbol mismatch|large delta|max abs delta|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("candidate_summaries", {}).values():
        lines.append(
            "|{candidate}|{rows}|{coverage}|{valid}|{invalid}|{missing}|{dates}|{symbols}|{large}|{delta}|".format(
                candidate=row.get("candidate"),
                rows=row.get("row_count"),
                coverage=_fmt(row.get("adjusted_coverage_ratio")),
                valid=row.get("valid_adjusted_period_count"),
                invalid=row.get("invalid_adjusted_period_count"),
                missing=row.get("missing_adjusted_price_row_count"),
                dates=row.get("date_misalignment_row_count"),
                symbols=row.get("symbol_mismatch_row_count"),
                large=row.get("large_return_delta_row_count"),
                delta=_fmt(row.get("max_abs_return_delta")),
            )
        )
    lines.extend([
        "",
        "## Biggest Return Deltas",
        "",
        "|candidate|rebalance|symbol|raw return|adjusted return|delta|ratio start|ratio end|missing adjusted|",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in payload.get("biggest_return_deltas", [])[:10]:
        lines.append(
            "|{candidate}|{date}|{symbol}|{raw}|{adjusted}|{delta}|{rs}|{re}|{missing}|".format(
                candidate=row.get("candidate"),
                date=row.get("rebalance_date"),
                symbol=row.get("symbol"),
                raw=_fmt(row.get("raw_return")),
                adjusted=_fmt(row.get("adjusted_return")),
                delta=_fmt(row.get("return_delta")),
                rs=_fmt(row.get("adjustment_ratio_start")),
                re=_fmt(row.get("adjustment_ratio_end")),
                missing=row.get("missing_adjusted_prices"),
            )
        )
    lines.extend([
        "",
        "## Biggest Candidate Net-Return Deltas",
        "",
        "|candidate|rebalance|symbol|raw net|adjusted net|delta|missing adjusted|",
        "|---|---|---|---:|---:|---:|---|",
    ])
    for row in payload.get("biggest_candidate_net_return_deltas", [])[:10]:
        lines.append(
            "|{candidate}|{date}|{symbol}|{raw}|{adjusted}|{delta}|{missing}|".format(
                candidate=row.get("candidate"),
                date=row.get("rebalance_date"),
                symbol=row.get("symbol"),
                raw=_fmt(row.get("raw_candidate_net_return")),
                adjusted=_fmt(row.get("adjusted_candidate_net_return")),
                delta=_fmt(row.get("candidate_net_return_delta")),
                missing=row.get("missing_adjusted_prices"),
            )
        )
    lines.extend(["", NOTICE, ""])
    return "\n".join(lines)
