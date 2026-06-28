from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from core.research.performance_metrics import calmar_ratio


RESEARCH_METADATA = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
}
NOTICE = "Research only. Trading impact: none. Production validated: false."


@dataclass(frozen=True)
class CanonicalContinuousReplayPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path


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


def _champion_rows(champion_audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in (
        champion_audit.get("exact_champion_replay", {}).get("period_rows", [])
        or []
    ):
        if not isinstance(row, dict):
            continue
        period_return = _number(row.get("period_return"))
        if period_return is None:
            continue
        selected_symbols = [str(symbol) for symbol in row.get("selected_symbols", [])]
        target_weights = {
            str(symbol): float(weight)
            for symbol, weight in (row.get("target_weights", {}) or {}).items()
            if _number(weight) is not None
        }
        rows.append({
            "candidate_name": "exact_champion_replay",
            "rebalance_date": str(row.get("rebalance_date", "")),
            "outcome_end_date": str(row.get("outcome_end_date", "")),
            "period_return": period_return,
            "exposure": _number(row.get("exposure_target")),
            "turnover": None,
            "cost": None,
            "net_return": period_return,
            "selected_symbols": selected_symbols,
            "target_weights": target_weights,
            "max_position_weight": max(target_weights.values(), default=None),
            "source": "exact_champion_replay",
        })
    return rows


def _selected_optimizer_rows(
    selected_optimizer: dict[str, Any],
    period_by_date: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for row in selected_optimizer.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        date = str(row.get("rebalance_date", ""))
        period = period_by_date.get(date, {})
        period_return = _number(row.get("period_return"))
        exposure = _number(row.get("exposure"))
        if period_return is None or exposure is None:
            continue
        cost = _number(row.get("cost")) or 0.0
        net_return = _number(row.get("net_return"))
        if net_return is None:
            net_return = period_return * exposure - cost
        target_weights = dict(period.get("target_weights", {}) or {})
        selected_symbols = list(period.get("selected_symbols", []) or [])
        invalid_reason = _optimizer_replay_invalid_reason(
            exposure,
            selected_symbols,
            target_weights,
        )
        rows.append({
            "candidate_name": "selected_bayesian_optimizer_diagnostic_policy",
            "rebalance_date": date,
            "outcome_end_date": str(period.get("outcome_end_date", "")),
            "period_return": period_return,
            "exposure": exposure,
            "turnover": _number(row.get("turnover")),
            "cost": cost,
            "net_return": 0.0 if invalid_reason else net_return,
            "selected_symbols": selected_symbols,
            "target_weights": target_weights,
            "max_position_weight": max(
                (_number(value) or 0.0 for value in target_weights.values()),
                default=None,
            ),
            "replay_valid": invalid_reason is None,
            "replay_invalid_reason": invalid_reason,
            "empty_selection_with_positive_exposure": (
                invalid_reason == "empty_selection_with_positive_exposure"
            ),
            "empty_selection_resolution": "invalidated" if invalid_reason else None,
            "source": "selected_optimizer_exposure_path",
            "score": _number(row.get("score")),
            "predicted_forward_return": _number(
                row.get("predicted_forward_return")
            ),
            "predicted_future_drawdown": _number(
                row.get("predicted_future_drawdown")
            ),
            "predicted_future_volatility": _number(
                row.get("predicted_future_volatility")
            ),
        })
    return rows


def _optimizer_replay_invalid_reason(
    exposure: float,
    selected_symbols: list[Any],
    target_weights: dict[str, Any],
) -> str | None:
    if exposure > 0.0 and not selected_symbols and not target_weights:
        return "empty_selection_with_positive_exposure"
    return None


def _candidate_payload(
    candidate_name: str,
    rows: list[dict[str, Any]],
    *,
    excluded_dates: set[str],
    excluded_symbols: set[str],
    period_return_semantics: str,
    period_cost_semantics: str,
) -> dict[str, Any]:
    filtered_rows = []
    replay_rows = []
    for row in sorted(rows, key=lambda item: str(item["rebalance_date"])):
        reason = _exclusion_reason(row, excluded_dates, excluded_symbols)
        replay_rows.append({**row, "exclusion_reason": reason})
        if reason is None:
            filtered_rows.append(row)
    non_overlap_rows = _non_overlapping_rows(filtered_rows)
    empty_selection_dates = [
        row["rebalance_date"] for row in replay_rows
        if row.get("empty_selection_with_positive_exposure")
    ]
    return {
        "candidate_name": candidate_name,
        "available": bool(rows),
        "period_return_semantics": period_return_semantics,
        "period_cost_semantics": period_cost_semantics,
        "diagnostic_period_grid": _summary(filtered_rows, all_rows=True),
        "canonical_continuous_equity": _summary(non_overlap_rows, all_rows=False),
        "rows": _equity_rows(replay_rows, non_overlap_rows),
        "empty_selection_with_positive_exposure_count": len(empty_selection_dates),
        "empty_selection_with_positive_exposure_dates": empty_selection_dates,
        "empty_selection_resolution": (
            "invalidated" if empty_selection_dates else "unchanged"
        ),
        **RESEARCH_METADATA,
    }


def _exclusion_reason(
    row: dict[str, Any],
    excluded_dates: set[str],
    excluded_symbols: set[str],
) -> str | None:
    invalid_reason = row.get("replay_invalid_reason")
    if invalid_reason:
        return str(invalid_reason)
    date = str(row.get("rebalance_date", ""))
    if date in excluded_dates:
        return "excluded_rebalance_date"
    symbols = {str(symbol) for symbol in row.get("selected_symbols", [])}
    if symbols & excluded_symbols:
        return "excluded_symbol"
    return None


def _non_overlapping_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = []
    previous_end: datetime | None = None
    for row in sorted(rows, key=lambda item: str(item["rebalance_date"])):
        start = _date(row.get("rebalance_date"))
        end = _date(row.get("outcome_end_date")) or start
        if start is None:
            continue
        if previous_end is None or start >= previous_end:
            kept.append(row)
            previous_end = end
    return kept


def _summary(rows: list[dict[str, Any]], *, all_rows: bool) -> dict[str, Any]:
    equity_rows = _compound_rows(rows)
    returns = [float(row["net_return"]) for row in equity_rows]
    total = _compound_returns(returns)
    drawdown = _max_drawdown([1.0] + [float(row["equity"]) for row in equity_rows])
    annualized = _annualized_return(total, equity_rows)
    return {
        "evaluation_mode": (
            "diagnostic_period_grid" if all_rows else "canonical_non_overlapping"
        ),
        "row_count": len(rows),
        "start_date": rows[0]["rebalance_date"] if rows else None,
        "end_date": rows[-1]["rebalance_date"] if rows else None,
        "last_outcome_end_date": rows[-1].get("outcome_end_date") if rows else None,
        "total_return": total,
        "canonical_tradable_total_return": None if all_rows else total,
        "annualized_return": annualized,
        "max_drawdown": drawdown,
        "sharpe": _sharpe(returns, equity_rows),
        "sortino": _sortino(returns, equity_rows),
        "calmar": calmar_ratio(annualized if annualized is not None else total, drawdown),
        "turnover": sum(
            float(row["turnover"])
            for row in rows
            if _number(row.get("turnover")) is not None
        ),
        "estimated_transaction_costs": sum(
            float(row["cost"])
            for row in rows
            if _number(row.get("cost")) is not None
        ),
        "largest_positive_period": max(returns, default=None),
        "largest_negative_period": min(returns, default=None),
    }


def _equity_rows(
    rows: list[dict[str, Any]],
    non_overlap_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    non_overlap_dates = {row["rebalance_date"] for row in non_overlap_rows}
    compounded = {
        row["rebalance_date"]: row
        for row in _compound_rows([
            row for row in rows
            if row.get("exclusion_reason") is None
            and row["rebalance_date"] in non_overlap_dates
        ])
    }
    output = []
    for row in rows:
        date = row["rebalance_date"]
        compound = compounded.get(date, {})
        output.append({
            "candidate_name": row["candidate_name"],
            "rebalance_date": date,
            "outcome_end_date": row.get("outcome_end_date"),
            "included_in_canonical": (
                date in non_overlap_dates and row.get("exclusion_reason") is None
            ),
            "exclusion_reason": row.get("exclusion_reason"),
            "period_return": row.get("period_return"),
            "exposure": row.get("exposure"),
            "turnover": row.get("turnover"),
            "cost": row.get("cost"),
            "net_return": row.get("net_return"),
            "equity": compound.get("equity"),
            "drawdown": compound.get("drawdown"),
            "selected_symbols": row.get("selected_symbols", []),
            "target_weights": row.get("target_weights", {}),
            "max_position_weight": row.get("max_position_weight"),
            "replay_valid": row.get("replay_valid", True),
            "replay_invalid_reason": row.get("replay_invalid_reason"),
            "empty_selection_with_positive_exposure": bool(
                row.get("empty_selection_with_positive_exposure", False)
            ),
            "empty_selection_resolution": row.get("empty_selection_resolution"),
            "source": row.get("source"),
            **RESEARCH_METADATA,
        })
    return output


def _compound_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    equity = 1.0
    peak = 1.0
    output = []
    for row in rows:
        net_return = float(row.get("net_return") or 0.0)
        equity *= 1.0 + net_return
        peak = max(peak, equity)
        output.append({
            **row,
            "equity": equity,
            "drawdown": (peak - equity) / peak if peak else 0.0,
        })
    return output


def _compound_returns(returns: list[float]) -> float:
    equity = 1.0
    for value in returns:
        equity *= 1.0 + value
    return equity - 1.0


def _max_drawdown(values: list[float]) -> float:
    peak = values[0] if values else 1.0
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = max(drawdown, (peak - value) / peak if peak else 0.0)
    return drawdown


def _annualized_return(total_return: float, rows: list[dict[str, Any]]) -> float | None:
    if len(rows) < 2 or total_return <= -1.0:
        return None
    start = _date(rows[0].get("rebalance_date"))
    end = _date(rows[-1].get("outcome_end_date")) or _date(rows[-1].get("rebalance_date"))
    if start is None or end is None:
        return None
    elapsed_days = (end - start).days
    if elapsed_days <= 0:
        return None
    return (1.0 + total_return) ** (365.25 / elapsed_days) - 1.0


def _periods_per_year(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return 1.0
    start = _date(rows[0].get("rebalance_date"))
    end = _date(rows[-1].get("outcome_end_date")) or _date(rows[-1].get("rebalance_date"))
    if start is None or end is None:
        return 1.0
    elapsed_days = (end - start).days
    return max(1.0, len(rows) * 365.25 / elapsed_days) if elapsed_days > 0 else 1.0


def _sharpe(returns: list[float], rows: list[dict[str, Any]]) -> float:
    if not returns:
        return 0.0
    average = mean(returns)
    variance = mean((value - average) ** 2 for value in returns)
    if variance <= 0:
        return 0.0
    return average / math.sqrt(variance) * math.sqrt(_periods_per_year(rows))


def _sortino(returns: list[float], rows: list[dict[str, Any]]) -> float:
    if not returns:
        return 0.0
    downside = [min(0.0, value) for value in returns]
    deviation = math.sqrt(sum(value * value for value in downside) / len(returns))
    if deviation <= 0:
        return 0.0
    return mean(returns) / deviation * math.sqrt(_periods_per_year(rows))


def _date(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _meta_output_dir(config: dict[str, Any]) -> Path:
    ml_config = config.get("ml", {})
    return Path(
        ml_config.get(
            "output_dir",
            Path(config.get("reports", {}).get("ml_dir", "reports/ml"))
            / "regime_transformer_meta_ensemble_v1",
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    rows = [
        row
        for candidate in payload.get("candidates", {}).values()
        for row in candidate.get("rows", [])
    ]
    fieldnames = [
        "candidate_name",
        "rebalance_date",
        "outcome_end_date",
        "included_in_canonical",
        "exclusion_reason",
        "period_return",
        "exposure",
        "turnover",
        "cost",
        "net_return",
        "equity",
        "drawdown",
        "selected_symbols",
        "max_position_weight",
        "source",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = {name: row.get(name) for name in fieldnames}
            output["selected_symbols"] = ",".join(row.get("selected_symbols", []))
            writer.writerow(output)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Canonical Continuous Equity Replay",
        "",
        NOTICE,
        "",
        "Old period-grid return is diagnostic only. Canonical return uses non-overlapping periods.",
        "",
        "|candidate|diagnostic period-grid return|canonical continuous return|rows|non-overlap rows|max drawdown|turnover|costs|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, candidate in payload.get("candidates", {}).items():
        diagnostic = candidate.get("diagnostic_period_grid", {})
        canonical = candidate.get("canonical_continuous_equity", {})
        lines.append(
            "|{name}|{diag}|{canonical}|{rows}|{kept}|{drawdown}|{turnover}|{costs}|".format(
                name=name,
                diag=_fmt(diagnostic.get("total_return")),
                canonical=_fmt(canonical.get("total_return")),
                rows=diagnostic.get("row_count", 0),
                kept=canonical.get("row_count", 0),
                drawdown=_fmt(canonical.get("max_drawdown")),
                turnover=_fmt(canonical.get("turnover")),
                costs=_fmt(canonical.get("estimated_transaction_costs")),
            )
        )
    lines.extend(["", NOTICE, ""])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.6f}"
