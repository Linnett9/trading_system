from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from core.interfaces.data_feed import IDataFeed


COST_STRESS_BPS = (5, 10, 25, 50, 100)
RESEARCH_METADATA = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
}


@dataclass(frozen=True)
class BenchmarkRelativeValidationPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path
    promotion_readiness_path: Path


def write_benchmark_relative_validation(
    config: dict[str, Any],
    data_feed: IDataFeed,
) -> BenchmarkRelativeValidationPaths:
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = _read_json(output_dir / "canonical_continuous_equity_replay.json")
    anomaly = _read_json(output_dir / "anomaly_quarantine_report.json")
    concentration = _read_json(output_dir / "profit_concentration_audit.json")
    external_reports = {
        "data_adjustment_audit": _read_json(output_dir / "data_adjustment_audit.json"),
        "clean_data_replay": _read_json(output_dir / "clean_data_replay.json"),
        "independent_period_validation": _read_json(
            output_dir / "independent_period_validation.json"
        ),
    }
    closes = _load_required_closes(config, data_feed, canonical)
    payload = build_benchmark_relative_validation(
        canonical_replay=canonical,
        anomaly_report=anomaly,
        concentration_report=concentration,
        closes_by_symbol=closes,
        validation_config=config.get("ml", {}).get(
            "benchmark_relative_validation",
            {},
        ),
        external_reports=external_reports,
    )
    paths = BenchmarkRelativeValidationPaths(
        csv_path=output_dir / "benchmark_relative_validation.csv",
        json_path=output_dir / "benchmark_relative_validation.json",
        markdown_path=output_dir / "benchmark_relative_validation.md",
        promotion_readiness_path=output_dir / "promotion_readiness_report.md",
    )
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(paths.csv_path, payload.get("candidates", []))
    paths.markdown_path.write_text(_markdown(payload), encoding="utf-8")
    paths.promotion_readiness_path.write_text(
        _promotion_markdown(payload),
        encoding="utf-8",
    )
    return paths


def build_benchmark_relative_validation(
    *,
    canonical_replay: dict[str, Any],
    anomaly_report: dict[str, Any],
    concentration_report: dict[str, Any] | None = None,
    closes_by_symbol: dict[str, dict[str, float]],
    validation_config: dict[str, Any] | None = None,
    external_reports: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = validation_config or {}
    external_reports = external_reports or {}
    schedule = _canonical_schedule(canonical_replay)
    flagged_dates = set(anomaly_report.get("flagged_rebalance_dates", []) or [])
    candidates = [
        _market_baseline("spy_buy_and_hold", schedule, closes_by_symbol, "SPY"),
        _market_baseline("qqq_buy_and_hold", schedule, closes_by_symbol, "QQQ"),
        _selected_universe_baseline(
            "equal_weight_selected_universe",
            schedule,
            closes_by_symbol,
            equal_weight=True,
        ),
        _selected_universe_baseline(
            "always_full_champion_universe",
            schedule,
            closes_by_symbol,
            equal_weight=False,
        ),
        _canonical_candidate(canonical_replay, "exact_champion_replay"),
        _canonical_candidate(
            canonical_replay,
            "selected_bayesian_optimizer_diagnostic_policy",
        ),
    ]
    scored = [
        _merge_existing_concentration(
            _score_candidate(candidate, flagged_dates),
            (concentration_report or {}).get("candidates", {}).get(
                candidate["candidate_name"],
                {},
            ),
        )
        for candidate in candidates
    ]
    by_name = {row["candidate_name"]: row for row in scored}
    benchmark_returns = {
        "spy": _return(by_name.get("spy_buy_and_hold")),
        "qqq": _return(by_name.get("qqq_buy_and_hold")),
        "equal_weight": _return(by_name.get("equal_weight_selected_universe")),
    }
    spy_drawdown = _number(
        by_name.get("spy_buy_and_hold", {}).get("max_drawdown")
    )
    gated = [
        _apply_gates(
            row,
            benchmark_returns=benchmark_returns,
            spy_drawdown=spy_drawdown,
            config=config,
            external_reports=external_reports,
        )
        for row in scored
    ]
    passing = [
        row["candidate_name"]
        for row in gated
        if row.get("promotion_candidate_status") == "pass"
    ]
    return {
        "mode": "benchmark_relative_tradability_validation_research_only",
        "canonical_alignment": (
            "all candidates use exact champion canonical non-overlapping windows"
        ),
        "cost_stress_semantics": (
            "incremental cost = estimated one-way effective-weight turnover * bps"
        ),
        "cost_stress_bps": list(COST_STRESS_BPS),
        "benchmark_returns": benchmark_returns,
        "gate_config": {
            "max_anomaly_dependency_ratio": float(
                config.get("max_anomaly_dependency_ratio", 0.25)
            ),
            "max_top_5_date_profit_share": float(
                config.get("max_top_5_date_profit_share", 0.50)
            ),
            "max_drawdown_worse_than_spy": float(
                config.get("max_drawdown_worse_than_spy", 0.05)
            ),
            "min_independent_periods": int(
                config.get("min_independent_periods", 36)
            ),
            "acceptable_adjusted_price_statuses": list(
                config.get(
                    "acceptable_adjusted_price_statuses",
                    [
                        "known_adjusted",
                        "appears_adjusted",
                        "raw_adjusted_identical",
                    ],
                )
            ),
            "allow_unknown_adjusted_price_status": bool(
                config.get("allow_unknown_adjusted_price_status", False)
            ),
        },
        "external_gate_reports_available": {
            name: bool(report) for name, report in external_reports.items()
        },
        "candidates": gated,
        "promotion_candidates": passing,
        "any_candidate_passes": bool(passing),
        **RESEARCH_METADATA,
    }


def _canonical_schedule(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    exact = canonical.get("candidates", {}).get("exact_champion_replay", {})
    return [
        dict(row)
        for row in exact.get("rows", []) or []
        if row.get("included_in_canonical") and not row.get("exclusion_reason")
    ]


def _market_baseline(
    name: str,
    schedule: list[dict[str, Any]],
    closes: dict[str, dict[str, float]],
    symbol: str,
) -> dict[str, Any]:
    rows = []
    for period in schedule:
        value = _price_return(closes, symbol, period)
        if value is None:
            continue
        rows.append(_baseline_row(period, value, {symbol: 1.0}))
    return {"candidate_name": name, "rows": rows}


def _selected_universe_baseline(
    name: str,
    schedule: list[dict[str, Any]],
    closes: dict[str, dict[str, float]],
    *,
    equal_weight: bool,
) -> dict[str, Any]:
    rows = []
    for period in schedule:
        symbols = [str(symbol) for symbol in period.get("selected_symbols", [])]
        source_weights = dict(period.get("target_weights", {}) or {})
        returns = {
            symbol: value
            for symbol in symbols
            if (value := _price_return(closes, symbol, period)) is not None
        }
        if not returns:
            continue
        if equal_weight:
            weights = {symbol: 1.0 / len(returns) for symbol in returns}
        else:
            weights = {
                symbol: float(source_weights.get(symbol, 0.0) or 0.0)
                for symbol in returns
            }
            total_weight = sum(weights.values())
            weights = (
                {symbol: weight / total_weight for symbol, weight in weights.items()}
                if total_weight > 0.0
                else {symbol: 1.0 / len(returns) for symbol in returns}
            )
        period_return = sum(returns[symbol] * weights[symbol] for symbol in returns)
        rows.append(_baseline_row(period, period_return, weights))
    return {"candidate_name": name, "rows": rows}


def _baseline_row(
    period: dict[str, Any],
    period_return: float,
    weights: dict[str, float],
) -> dict[str, Any]:
    return {
        "rebalance_date": str(period.get("rebalance_date", "")),
        "outcome_end_date": str(period.get("outcome_end_date", "")),
        "period_return": period_return,
        "net_return": period_return,
        "exposure": 1.0,
        "selected_symbols": sorted(weights),
        "target_weights": weights,
    }


def _canonical_candidate(
    canonical: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    candidate = canonical.get("candidates", {}).get(name, {})
    rows = [
        dict(row)
        for row in candidate.get("rows", []) or []
        if row.get("included_in_canonical") and not row.get("exclusion_reason")
    ]
    return {"candidate_name": name, "rows": rows}


def _score_candidate(
    candidate: dict[str, Any],
    flagged_dates: set[str],
) -> dict[str, Any]:
    rows = sorted(candidate.get("rows", []), key=lambda row: row["rebalance_date"])
    if not rows:
        return {
            "candidate_name": candidate["candidate_name"],
            "available": False,
            "skip_reason": "no aligned canonical rows",
            **RESEARCH_METADATA,
        }
    returns = [float(row.get("net_return") or 0.0) for row in rows]
    anomaly_returns = [
        float(row.get("net_return") or 0.0)
        for row in rows
        if str(row.get("rebalance_date")) not in flagged_dates
    ]
    turnovers = _turnover_by_row(rows)
    total_return = _compound(returns)
    anomaly_return = _compound(anomaly_returns)
    positive_returns = sorted((value for value in returns if value > 0.0), reverse=True)
    positive_total = sum(positive_returns)
    symbol_contributions = _symbol_contributions(rows)
    positive_symbol_total = sum(
        value for value in symbol_contributions.values() if value > 0.0
    )
    cost_returns = {
        f"cost_stressed_return_{bps}bps": _compound([
            period_return - (turnover * bps / 10_000.0)
            for period_return, turnover in zip(returns, turnovers)
        ])
        for bps in COST_STRESS_BPS
    }
    ratio = max(
        0.0,
        (total_return - anomaly_return) / max(abs(total_return), 1e-12),
    )
    curve = _equity_curve(returns)
    return {
        "candidate_name": candidate["candidate_name"],
        "available": True,
        "canonical_non_overlap_return": total_return,
        "anomaly_adjusted_return": anomaly_return,
        "anomaly_dependency_ratio": ratio,
        "max_drawdown": _max_drawdown(curve),
        "sharpe": _sharpe(returns, rows),
        "sortino": _sortino(returns, rows),
        "turnover": sum(turnovers),
        **cost_returns,
        "top_1_date_profit_share": (
            positive_returns[0] / positive_total
            if positive_returns and positive_total else None
        ),
        "top_5_date_profit_share": (
            sum(positive_returns[:5]) / positive_total
            if positive_returns and positive_total else None
        ),
        "top_1_symbol_profit_share": (
            max(symbol_contributions.values(), default=0.0) / positive_symbol_total
            if positive_symbol_total else None
        ),
        "canonical_period_count": len(rows),
        "flagged_period_count": sum(
            str(row.get("rebalance_date")) in flagged_dates for row in rows
        ),
        **RESEARCH_METADATA,
    }


def _merge_existing_concentration(
    row: dict[str, Any],
    concentration: dict[str, Any],
) -> dict[str, Any]:
    if not row.get("available") or not concentration:
        return row
    anomaly_return = next(
        (
            _number(scenario.get("summary", {}).get("total_return"))
            for scenario in concentration.get("scenarios", []) or []
            if scenario.get("scenario_name") == "remove_anomaly_dates"
        ),
        None,
    )
    metrics = concentration.get("profit_concentration", {})
    total_return = float(row["canonical_non_overlap_return"])
    if anomaly_return is not None:
        row["anomaly_adjusted_return"] = anomaly_return
        row["anomaly_dependency_ratio"] = max(
            0.0,
            (total_return - anomaly_return) / max(abs(total_return), 1e-12),
        )
    mappings = {
        "top_1_date_profit_share": "top_1_date_positive_return_share",
        "top_5_date_profit_share": "top_5_date_positive_return_share",
        "top_1_symbol_profit_share": "top_1_symbol_contribution_share",
    }
    for output_name, source_name in mappings.items():
        value = _number(metrics.get(source_name))
        if value is not None:
            row[output_name] = value
    return row


def _apply_gates(
    row: dict[str, Any],
    *,
    benchmark_returns: dict[str, float | None],
    spy_drawdown: float | None,
    config: dict[str, Any],
    external_reports: dict[str, Any],
) -> dict[str, Any]:
    if not row.get("available"):
        return {
            **row,
            "benchmark_relative_pass": False,
            "tradability_validation_pass": False,
            "promotion_candidate_status": "unavailable",
            "failed_gates": ["candidate_unavailable"],
        }
    candidate_return = float(row["canonical_non_overlap_return"])
    excess = {
        name: (
            candidate_return - benchmark
            if benchmark is not None else None
        )
        for name, benchmark in benchmark_returns.items()
    }
    gates = {
        "anomaly_dependency": float(row["anomaly_dependency_ratio"])
        <= float(config.get("max_anomaly_dependency_ratio", 0.25)),
        "top_5_date_concentration": (
            _number(row.get("top_5_date_profit_share")) is not None
            and float(row["top_5_date_profit_share"])
            <= float(config.get("max_top_5_date_profit_share", 0.50))
        ),
        "positive_excess_vs_spy": excess["spy"] is not None and excess["spy"] > 0.0,
        "positive_excess_vs_qqq": excess["qqq"] is not None and excess["qqq"] > 0.0,
        "positive_excess_vs_equal_weight": (
            excess["equal_weight"] is not None and excess["equal_weight"] > 0.0
        ),
        "survives_25bps": float(row["cost_stressed_return_25bps"]) > 0.0,
        "survives_50bps": float(row["cost_stressed_return_50bps"]) > 0.0,
        "drawdown_not_materially_worse_than_spy": (
            spy_drawdown is not None
            and float(row["max_drawdown"])
            <= spy_drawdown + float(config.get("max_drawdown_worse_than_spy", 0.05))
        ),
    }
    external_context = _external_promotion_gate_context(
        str(row["candidate_name"]),
        external_reports,
        config,
    )
    gates.update(external_context.get("gates", {}))
    benchmark_pass = all(
        gates[name]
        for name in (
            "positive_excess_vs_spy",
            "positive_excess_vs_qqq",
            "positive_excess_vs_equal_weight",
        )
    )
    tradability_pass = all(
        value for name, value in gates.items() if not name.startswith("positive_excess")
    )
    return {
        **row,
        "excess_return_vs_spy": excess["spy"],
        "excess_return_vs_qqq": excess["qqq"],
        "excess_return_vs_equal_weight": excess["equal_weight"],
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "external_gate_context": external_context.get("context", {}),
        "benchmark_relative_pass": benchmark_pass,
        "tradability_validation_pass": tradability_pass,
        "promotion_candidate_status": (
            "pass" if benchmark_pass and tradability_pass else "blocked"
        ),
    }


def _external_promotion_gate_context(
    candidate_name: str,
    external_reports: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    gates: dict[str, bool] = {}
    context: dict[str, Any] = {}
    adjustment = external_reports.get("data_adjustment_audit") or {}
    if adjustment:
        status = str(
            adjustment.get("adjusted_price_status")
            or adjustment.get("adjusted_status")
            or "unknown"
        )
        dependencies = (
            adjustment.get("candidate_dependencies", {}).get(candidate_name, {})
            if isinstance(adjustment.get("candidate_dependencies"), dict)
            else {}
        )
        dependency_count = int(dependencies.get("suspicious_dependency_count") or 0)
        gates["adjusted_price_status"] = _adjusted_price_status_acceptable(
            status,
            adjustment,
            config,
        )
        gates["no_suspicious_split_like_rows"] = dependency_count == 0
        context["data_adjustment_audit"] = {
            "adjusted_price_status": status,
            "suspicious_dependency_count": dependency_count,
            "suspicious_rebalance_dates": dependencies.get(
                "suspicious_rebalance_dates",
                [],
            ),
        }
    independent = external_reports.get("independent_period_validation") or {}
    if independent:
        gate = independent.get("gate", {})
        actual = int(
            gate.get(
                "actual",
                independent.get("independent_canonical_period_count", 0),
            )
            or 0
        )
        minimum = int(
            gate.get(
                "minimum",
                independent.get(
                    "minimum_independent_periods",
                    config.get("min_independent_periods", 36),
                ),
            )
            or 0
        )
        gates["minimum_independent_periods"] = bool(
            gate.get("passed", actual >= minimum)
        )
        context["independent_period_validation"] = {
            "actual": actual,
            "minimum": minimum,
        }
    clean = external_reports.get("clean_data_replay") or {}
    if clean:
        clean_candidates = clean.get("candidates", {})
        clean_row = (
            clean_candidates.get(candidate_name, {})
            if isinstance(clean_candidates, dict)
            else {}
        )
        gates["clean_data_return_positive"] = bool(
            clean_row.get("clean_data_return_positive", False)
        )
        gates["clean_data_benchmark_relative"] = bool(
            clean_row.get("clean_data_benchmark_relative", False)
        )
        context["clean_data_replay"] = {
            "clean_canonical_return": clean_row.get("clean_canonical_return"),
            "clean_data_verdict": clean_row.get("clean_data_verdict"),
        }
    return {"gates": gates, "context": context}


def _adjusted_price_status_acceptable(
    status: str,
    adjustment_report: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    gate = adjustment_report.get("promotion_gate", {})
    if "adjusted_price_status_acceptable" in gate:
        return bool(gate["adjusted_price_status_acceptable"])
    acceptable = set(
        str(value)
        for value in config.get(
            "acceptable_adjusted_price_statuses",
            ["known_adjusted", "appears_adjusted", "raw_adjusted_identical"],
        )
    )
    if status in acceptable:
        return True
    return bool(config.get("allow_unknown_adjusted_price_status", False)) and (
        status.startswith("unknown")
    )


def _turnover_by_row(rows: list[dict[str, Any]]) -> list[float]:
    previous: dict[str, float] = {}
    turnovers = []
    for row in rows:
        exposure = float(row.get("exposure", 1.0) or 0.0)
        weights = {
            str(symbol): float(weight) * exposure
            for symbol, weight in (row.get("target_weights", {}) or {}).items()
        }
        if not weights:
            symbols = [str(symbol) for symbol in row.get("selected_symbols", [])]
            weights = {
                symbol: exposure / len(symbols) for symbol in symbols
            } if symbols else {}
        assets = set(previous) | set(weights)
        asset_change = sum(abs(weights.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in assets)
        cash_change = abs((1.0 - sum(weights.values())) - (1.0 - sum(previous.values())))
        turnovers.append(0.5 * (asset_change + cash_change))
        previous = weights
    return turnovers


def _symbol_contributions(rows: list[dict[str, Any]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for row in rows:
        symbols = [str(symbol) for symbol in row.get("selected_symbols", [])]
        if not symbols:
            continue
        weights = dict(row.get("target_weights", {}) or {})
        total_weight = sum(float(weights.get(symbol, 0.0) or 0.0) for symbol in symbols)
        for symbol in symbols:
            weight = (
                float(weights.get(symbol, 0.0) or 0.0) / total_weight
                if total_weight > 0.0 else 1.0 / len(symbols)
            )
            output[symbol] = output.get(symbol, 0.0) + float(
                row.get("net_return") or 0.0
            ) * weight
    return output


def _price_return(
    closes: dict[str, dict[str, float]],
    symbol: str,
    period: dict[str, Any],
) -> float | None:
    values = closes.get(symbol.upper(), {})
    start = values.get(str(period.get("rebalance_date", "")))
    end = values.get(str(period.get("outcome_end_date", "")))
    if start is None or end is None or start <= 0.0:
        return None
    return (end / start) - 1.0


def _load_required_closes(
    config: dict[str, Any],
    data_feed: IDataFeed,
    canonical: dict[str, Any],
) -> dict[str, dict[str, float]]:
    schedule = _canonical_schedule(canonical)
    if not schedule:
        return {}
    symbols = {"SPY", "QQQ"}
    symbols.update(
        str(symbol).upper()
        for row in schedule
        for symbol in row.get("selected_symbols", [])
    )
    start = datetime.fromisoformat(schedule[0]["rebalance_date"][:10])
    end = datetime.fromisoformat(schedule[-1]["outcome_end_date"][:10])
    output = {}
    for symbol in sorted(symbols):
        try:
            candles = data_feed.get_historical_bars(symbol, "1Day", start, end)
        except (FileNotFoundError, ValueError):
            continue
        output[symbol] = {
            candle.timestamp.date().isoformat(): float(candle.close)
            for candle in candles
        }
    return output


def _compound(returns: list[float]) -> float:
    equity = 1.0
    for value in returns:
        equity *= 1.0 + value
    return equity - 1.0


def _equity_curve(returns: list[float]) -> list[float]:
    equity = 1.0
    output = [equity]
    for value in returns:
        equity *= 1.0 + value
        output.append(equity)
    return output


def _max_drawdown(values: list[float]) -> float:
    peak = values[0] if values else 1.0
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = max(drawdown, (peak - value) / peak if peak else 0.0)
    return drawdown


def _sharpe(returns: list[float], rows: list[dict[str, Any]]) -> float:
    if len(returns) < 2:
        return 0.0
    average = mean(returns)
    variance = sum((value - average) ** 2 for value in returns) / len(returns)
    return average / math.sqrt(variance) * math.sqrt(_periods_per_year(rows)) if variance > 0 else 0.0


def _sortino(returns: list[float], rows: list[dict[str, Any]]) -> float:
    downside = [min(value, 0.0) for value in returns]
    downside_deviation = math.sqrt(sum(value * value for value in downside) / len(downside)) if downside else 0.0
    return mean(returns) / downside_deviation * math.sqrt(_periods_per_year(rows)) if downside_deviation > 0 else 0.0


def _periods_per_year(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return 1.0
    start = datetime.fromisoformat(str(rows[0]["rebalance_date"])[:10])
    end = datetime.fromisoformat(str(rows[-1]["outcome_end_date"])[:10])
    years = max((end - start).days / 365.25, 1.0 / 365.25)
    return len(rows) / years


def _return(row: dict[str, Any] | None) -> float | None:
    return _number((row or {}).get("canonical_non_overlap_return"))


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _output_dir(config: dict[str, Any]) -> Path:
    return Path(config.get("ml", {}).get("output_dir", "reports/ml/regime_transformer_meta_ensemble_v1"))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "candidate_name", "available", "canonical_non_overlap_return",
        "anomaly_adjusted_return", "anomaly_dependency_ratio", "max_drawdown",
        "sharpe", "sortino", "turnover",
        *[f"cost_stressed_return_{bps}bps" for bps in COST_STRESS_BPS],
        "top_1_date_profit_share", "top_5_date_profit_share",
        "top_1_symbol_profit_share", "excess_return_vs_spy",
        "excess_return_vs_qqq", "excess_return_vs_equal_weight",
        "benchmark_relative_pass", "tradability_validation_pass",
        "promotion_candidate_status", "failed_gates", "research_only",
        "trading_impact", "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({
            **{name: row.get(name) for name in fieldnames},
            "failed_gates": json.dumps(row.get("failed_gates", [])),
        } for row in rows)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Benchmark-Relative and Tradability Validation", "",
        "Research only. Trading impact: none. Production validated: false.", "",
        "|candidate|canonical return|anomaly-adjusted|drawdown|Sharpe|turnover|top 5 dates|excess SPY|excess QQQ|excess equal-weight|status|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("candidates", []):
        lines.append(
            f"|{row['candidate_name']}|{_fmt(row.get('canonical_non_overlap_return'))}|"
            f"{_fmt(row.get('anomaly_adjusted_return'))}|{_fmt(row.get('max_drawdown'))}|"
            f"{_fmt(row.get('sharpe'))}|{_fmt(row.get('turnover'))}|"
            f"{_fmt(row.get('top_5_date_profit_share'))}|{_fmt(row.get('excess_return_vs_spy'))}|"
            f"{_fmt(row.get('excess_return_vs_qqq'))}|{_fmt(row.get('excess_return_vs_equal_weight'))}|"
            f"{row.get('promotion_candidate_status')}|"
        )
    return "\n".join(lines) + "\n"


def _promotion_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Promotion Readiness", "",
        "Research only. No automatic promotion or trading impact.", "",
        f"Any candidate passes all gates: {payload.get('any_candidate_passes', False)}", "",
    ]
    for row in payload.get("candidates", []):
        lines.extend([
            f"## {row['candidate_name']}", "",
            f"Status: {row.get('promotion_candidate_status')}",
            f"Failed gates: {', '.join(row.get('failed_gates', [])) or 'none'}", "",
        ])
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.6f}"
