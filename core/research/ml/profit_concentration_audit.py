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
class ProfitConcentrationAuditPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path


def write_profit_concentration_audit(
    config: dict[str, Any],
) -> ProfitConcentrationAuditPaths:
    output_dir = _meta_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = _read_json(output_dir / "canonical_continuous_equity_replay.json")
    anomaly = _read_json(output_dir / "anomaly_quarantine_report.json")
    payload = build_profit_concentration_audit(
        canonical_replay=canonical,
        anomaly_report=anomaly,
    )
    paths = ProfitConcentrationAuditPaths(
        csv_path=output_dir / "profit_concentration_audit.csv",
        json_path=output_dir / "profit_concentration_audit.json",
        markdown_path=output_dir / "profit_concentration_audit.md",
    )
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(paths.csv_path, payload)
    paths.markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return paths


def build_profit_concentration_audit(
    *,
    canonical_replay: dict[str, Any],
    anomaly_report: dict[str, Any],
) -> dict[str, Any]:
    flagged_dates = set(anomaly_report.get("flagged_rebalance_dates", []) or [])
    candidates = {}
    for name, candidate in canonical_replay.get("candidates", {}).items():
        rows = [
            row for row in candidate.get("rows", [])
            if not row.get("exclusion_reason")
        ]
        candidates[name] = _candidate_audit(name, rows, flagged_dates)
    return {
        "mode": "profit_concentration_audit_research_only",
        "scenario_definitions": [
            "baseline",
            "remove_best_date",
            "remove_best_week",
            "remove_best_month",
            "remove_best_symbol",
            "remove_top_3_symbols",
            "remove_anomaly_dates",
        ],
        "candidates": candidates,
        "comparison": _comparison(candidates),
        **RESEARCH_METADATA,
    }


def _candidate_audit(
    name: str,
    rows: list[dict[str, Any]],
    flagged_dates: set[str],
) -> dict[str, Any]:
    baseline_rows = _included_rows(rows)
    baseline = _summary(baseline_rows)
    symbol_contributions = _symbol_contributions(baseline_rows)
    best_symbol = _top_symbols(symbol_contributions, 1)
    top_3_symbols = _top_symbols(symbol_contributions, 3)
    best_date = _best_date(baseline_rows)
    best_week = _week_key(best_date)
    best_month = best_date[:7] if best_date else None
    scenarios = [
        _scenario("baseline", rows, baseline, set(), set()),
        _scenario("remove_best_date", rows, baseline, {best_date} if best_date else set(), set()),
        _scenario("remove_best_week", rows, baseline, set(), set(), week=best_week),
        _scenario("remove_best_month", rows, baseline, set(), set(), month=best_month),
        _scenario("remove_best_symbol", rows, baseline, set(), set(best_symbol)),
        _scenario("remove_top_3_symbols", rows, baseline, set(), set(top_3_symbols)),
        _scenario("remove_anomaly_dates", rows, baseline, flagged_dates, set()),
    ]
    concentration = _concentration_summary(baseline_rows, symbol_contributions)
    return {
        "candidate_name": name,
        "baseline": baseline,
        "profit_concentration": concentration,
        "top_symbols": [
            {"symbol": symbol, "contribution": contribution}
            for symbol, contribution in sorted(
                symbol_contributions.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:20]
        ],
        "scenarios": scenarios,
        **RESEARCH_METADATA,
    }


def _scenario(
    scenario_name: str,
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    excluded_dates: set[str],
    excluded_symbols: set[str],
    *,
    week: str | None = None,
    month: str | None = None,
) -> dict[str, Any]:
    filtered = []
    for row in rows:
        date = str(row.get("rebalance_date", ""))
        symbols = {str(symbol) for symbol in row.get("selected_symbols", [])}
        if date in excluded_dates:
            continue
        if excluded_symbols and symbols & excluded_symbols:
            continue
        if week and _week_key(date) == week:
            continue
        if month and date.startswith(month):
            continue
        filtered.append(row)
    included = _non_overlapping_rows(filtered)
    summary = _summary(included)
    remaining_symbol_contributions = _symbol_contributions(included)
    removed_symbols = sorted(excluded_symbols)
    return {
        "scenario_name": scenario_name,
        "excluded_dates": sorted(date for date in excluded_dates if date),
        "excluded_symbols": removed_symbols,
        "excluded_week": week,
        "excluded_month": month,
        "summary": summary,
        "return_delta_vs_baseline": (
            _number(summary.get("total_return"))
            - _number(baseline.get("total_return"))
            if _number(summary.get("total_return")) is not None
            and _number(baseline.get("total_return")) is not None
            else None
        ),
        "removed_contributor_remaining_contribution": {
            symbol: remaining_symbol_contributions.get(symbol, 0.0)
            for symbol in removed_symbols
        },
    }


def _included_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row.get("included_in_canonical")
    ]


def _non_overlapping_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = []
    previous_end: datetime | None = None
    for row in sorted(rows, key=lambda item: str(item.get("rebalance_date", ""))):
        start = _date(row.get("rebalance_date"))
        end = _date(row.get("outcome_end_date")) or start
        if start is None:
            continue
        if previous_end is None or start >= previous_end:
            kept.append(row)
            previous_end = end
    return kept


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    equity = 1.0
    peak = 1.0
    curve = [1.0]
    returns = []
    for row in rows:
        net_return = float(row.get("net_return") or 0.0)
        returns.append(net_return)
        equity *= 1.0 + net_return
        peak = max(peak, equity)
        curve.append(equity)
    total = equity - 1.0
    drawdown = max(((max(curve[: index + 1]) - value) / max(curve[: index + 1])) for index, value in enumerate(curve)) if curve else 0.0
    annualized = _annualized_return(total, rows)
    return {
        "row_count": len(rows),
        "total_return": total,
        "canonical_tradable_total_return": total,
        "annualized_return": annualized,
        "max_drawdown": drawdown,
        "sharpe": _sharpe(returns, rows),
        "sortino": _sortino(returns, rows),
        "calmar": calmar_ratio(annualized if annualized is not None else total, drawdown),
        "largest_positive_period": max(returns, default=None),
        "largest_negative_period": min(returns, default=None),
    }


def _concentration_summary(
    rows: list[dict[str, Any]],
    symbol_contributions: dict[str, float],
) -> dict[str, Any]:
    positive_returns = sorted(
        [float(row.get("net_return") or 0.0) for row in rows if float(row.get("net_return") or 0.0) > 0],
        reverse=True,
    )
    positive_total = sum(positive_returns)
    top_symbols = sorted(symbol_contributions.values(), reverse=True)
    symbol_positive_total = sum(value for value in top_symbols if value > 0)
    return {
        "top_1_date_positive_return_share": (
            positive_returns[0] / positive_total
            if positive_returns and positive_total
            else None
        ),
        "top_5_date_positive_return_share": (
            sum(positive_returns[:5]) / positive_total
            if positive_returns and positive_total
            else None
        ),
        "top_1_symbol_contribution_share": (
            top_symbols[0] / symbol_positive_total
            if top_symbols and symbol_positive_total
            else None
        ),
        "top_5_symbol_contribution_share": (
            sum(top_symbols[:5]) / symbol_positive_total
            if top_symbols and symbol_positive_total
            else None
        ),
    }


def _symbol_contributions(rows: list[dict[str, Any]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for row in rows:
        symbols = [str(symbol) for symbol in row.get("selected_symbols", [])]
        if not symbols:
            continue
        weights = row.get("target_weights", {}) or {}
        net_return = float(row.get("net_return") or 0.0)
        fallback_weight = 1.0 / len(symbols)
        for symbol in symbols:
            weight = _number(weights.get(symbol))
            output[symbol] = output.get(symbol, 0.0) + net_return * (
                weight if weight is not None else fallback_weight
            )
    return output


def _top_symbols(contributions: dict[str, float], count: int) -> list[str]:
    return [
        symbol for symbol, _ in sorted(
            contributions.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:count]
    ]


def _best_date(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    row = max(rows, key=lambda item: float(item.get("net_return") or 0.0))
    return str(row.get("rebalance_date", "")) or None


def _week_key(date: str | None) -> str | None:
    if not date:
        return None
    parsed = _date(date)
    if parsed is None:
        return None
    year, week, _ = parsed.isocalendar()
    return f"{year}-W{week:02d}"


def _comparison(candidates: dict[str, Any]) -> dict[str, Any]:
    selected = candidates.get("selected_bayesian_optimizer_diagnostic_policy", {})
    champion = candidates.get("exact_champion_replay", {})
    return {
        "selected_optimizer_canonical_return": selected.get("baseline", {}).get(
            "total_return"
        ),
        "exact_champion_canonical_return": champion.get("baseline", {}).get(
            "total_return"
        ),
        "selected_beats_exact_champion": (
            _number(selected.get("baseline", {}).get("total_return")) is not None
            and _number(champion.get("baseline", {}).get("total_return")) is not None
            and float(selected["baseline"]["total_return"])
            > float(champion["baseline"]["total_return"])
        ),
    }


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
    fieldnames = [
        "candidate_name",
        "scenario_name",
        "total_return",
        "max_drawdown",
        "row_count",
        "return_delta_vs_baseline",
        "excluded_dates",
        "excluded_symbols",
        "excluded_week",
        "excluded_month",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate_name, candidate in payload.get("candidates", {}).items():
            for scenario in candidate.get("scenarios", []):
                summary = scenario.get("summary", {})
                writer.writerow({
                    "candidate_name": candidate_name,
                    "scenario_name": scenario.get("scenario_name"),
                    "total_return": summary.get("total_return"),
                    "max_drawdown": summary.get("max_drawdown"),
                    "row_count": summary.get("row_count"),
                    "return_delta_vs_baseline": scenario.get(
                        "return_delta_vs_baseline"
                    ),
                    "excluded_dates": ",".join(scenario.get("excluded_dates", [])),
                    "excluded_symbols": ",".join(
                        scenario.get("excluded_symbols", [])
                    ),
                    "excluded_week": scenario.get("excluded_week"),
                    "excluded_month": scenario.get("excluded_month"),
                    **RESEARCH_METADATA,
                })


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Profit Concentration Audit",
        "",
        NOTICE,
        "",
        "|candidate|baseline canonical return|top 1 date share|top 5 date share|top 1 symbol share|remove anomaly dates return|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, candidate in payload.get("candidates", {}).items():
        concentration = candidate.get("profit_concentration", {})
        anomaly = next(
            (
                row for row in candidate.get("scenarios", [])
                if row.get("scenario_name") == "remove_anomaly_dates"
            ),
            {},
        )
        lines.append(
            "|{name}|{baseline}|{top1_date}|{top5_date}|{top1_symbol}|{anomaly}|".format(
                name=name,
                baseline=_fmt(candidate.get("baseline", {}).get("total_return")),
                top1_date=_fmt(
                    concentration.get("top_1_date_positive_return_share")
                ),
                top5_date=_fmt(
                    concentration.get("top_5_date_positive_return_share")
                ),
                top1_symbol=_fmt(
                    concentration.get("top_1_symbol_contribution_share")
                ),
                anomaly=_fmt(
                    anomaly.get("summary", {}).get("total_return")
                ),
            )
        )
    lines.extend(["", NOTICE, ""])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.6f}"
