from __future__ import annotations

import json
from typing import Any

from core.research.ml.replay.canonical_replay_candidates import (
    _candidate_payload,
    _champion_rows,
    _exclusion_reason,
    _non_overlapping_rows,
    _optimizer_replay_invalid_reason,
    _selected_optimizer_rows,
)
from core.research.ml.replay.canonical_replay_io import (
    _fmt,
    _markdown,
    _meta_output_dir,
    _read_json,
    _write_csv,
)
from core.research.ml.replay.canonical_replay_math import (
    _annualized_return,
    _compound_returns,
    _compound_rows,
    _date,
    _max_drawdown,
    _number,
    _periods_per_year,
    _sharpe,
    _sortino,
)
from core.research.ml.replay.canonical_replay_metrics import _equity_rows, _summary
from core.research.ml.replay.canonical_replay_types import (
    NOTICE,
    RESEARCH_METADATA,
    CanonicalContinuousReplayPaths,
)


def score_candidate_exposure_path(
    rows: list[dict[str, Any]],
    *,
    candidate_name: str = "optimizer_candidate",
    excluded_dates: set[str] | None = None,
    cost_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Score one exposure path with canonical non-overlap mechanics, without I/O."""
    if cost_multiplier < 0.0:
        raise ValueError("cost_multiplier must be non-negative")
    normalized_rows = []
    for row in rows:
        period_return = _number(row.get("period_return"))
        exposure = _number(row.get("exposure"))
        if period_return is None or exposure is None:
            continue
        base_cost = _number(row.get("cost")) or 0.0
        stressed_cost = base_cost * cost_multiplier
        normalized_rows.append({
            **row,
            "candidate_name": candidate_name,
            "period_return": period_return,
            "exposure": exposure,
            "turnover": _number(row.get("turnover")) or 0.0,
            "cost": stressed_cost,
            "net_return": (period_return * exposure) - stressed_cost,
            "selected_symbols": list(row.get("selected_symbols", []) or []),
            "target_weights": dict(row.get("target_weights", {}) or {}),
            "source": row.get("source", "optimizer_candidate_exposure_path"),
        })
    return _candidate_payload(
        candidate_name,
        normalized_rows,
        excluded_dates=excluded_dates or set(),
        excluded_symbols=set(),
        period_return_semantics=(
            "allocation overlay return: baseline period return * candidate exposure "
            "minus turnover cost"
        ),
        period_cost_semantics=(
            "explicit allocation turnover cost multiplied by "
            f"{cost_multiplier:g}"
        ),
    )


def write_canonical_continuous_equity_replay(
    config: dict[str, Any],
) -> CanonicalContinuousReplayPaths:
    output_dir = _meta_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_optimizer = _read_json(output_dir / "selected_optimizer_exposure_path.json")
    champion_audit = _read_json(output_dir / "champion_baseline_audit.json")
    payload = build_canonical_replay(
        selected_optimizer=selected_optimizer,
        champion_audit=champion_audit,
    )
    paths = CanonicalContinuousReplayPaths(
        csv_path=output_dir / "canonical_continuous_equity_replay.csv",
        json_path=output_dir / "canonical_continuous_equity_replay.json",
        markdown_path=output_dir / "canonical_continuous_equity_replay.md",
    )
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(paths.csv_path, payload)
    paths.markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return paths


def build_canonical_replay(
    *,
    selected_optimizer: dict[str, Any],
    champion_audit: dict[str, Any],
    excluded_dates: set[str] | None = None,
    excluded_symbols: set[str] | None = None,
) -> dict[str, Any]:
    excluded_dates = excluded_dates or set()
    excluded_symbols = excluded_symbols or set()
    champion_rows = _champion_rows(champion_audit)
    period_by_date = {
        str(row["rebalance_date"]): row
        for row in champion_rows
        if row.get("rebalance_date")
    }
    selected_rows = _selected_optimizer_rows(selected_optimizer, period_by_date)
    candidates = {
        "selected_bayesian_optimizer_diagnostic_policy": _candidate_payload(
            "selected_bayesian_optimizer_diagnostic_policy",
            selected_rows,
            excluded_dates=excluded_dates,
            excluded_symbols=excluded_symbols,
            period_return_semantics=(
                "allocation overlay return: baseline period return * selected "
                "optimizer exposure minus turnover cost"
            ),
            period_cost_semantics="explicit allocation turnover cost",
        ),
        "exact_champion_replay": _candidate_payload(
            "exact_champion_replay",
            champion_rows,
            excluded_dates=excluded_dates,
            excluded_symbols=excluded_symbols,
            period_return_semantics=(
                "frozen champion backtester equity change over the period; "
                "exposure, cash drag, and strategy costs are already embedded"
            ),
            period_cost_semantics=(
                "embedded in champion backtester equity curve; period attribution "
                "is unavailable"
            ),
        ),
    }
    return {
        "mode": "canonical_continuous_equity_replay_research_only",
        "canonical_definition": {
            "canonical_tradable_total_return": (
                "non-overlapping compounded equity return from one rebalance "
                "state at a time; cash return assumed zero"
            ),
            "diagnostic_period_grid_return": (
                "all saved rebalance rows compounded, including overlapping "
                "forward windows; diagnostic only"
            ),
            "paper_tradable_equity_return": None,
            "non_overlap_rule": (
                "keep rows sorted by rebalance_date only when rebalance_date is "
                "on or after the prior kept row's outcome_end_date"
            ),
        },
        "exclusions": {
            "excluded_dates": sorted(excluded_dates),
            "excluded_symbols": sorted(excluded_symbols),
        },
        "candidates": candidates,
        **RESEARCH_METADATA,
    }
