from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


RESEARCH_METADATA = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
}
REPORT_CANDIDATES = (
    "exact_champion_replay",
    "selected_bayesian_optimizer_diagnostic_policy",
)


@dataclass(frozen=True)
class IndependentPeriodExpansionAuditPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path


def write_independent_period_expansion_audit(
    config: dict[str, Any],
) -> IndependentPeriodExpansionAuditPaths:
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    adjusted_replay = _read_json(output_dir / "adjusted_price_replay.json")
    canonical_replay = _read_json(output_dir / "canonical_continuous_equity_replay.json")
    data_adjustment = _read_json(output_dir / "data_adjustment_audit.json")
    benchmark_validation = _read_json(output_dir / "benchmark_relative_validation.json")
    adjusted_alignment = _read_json(output_dir / "adjusted_replay_alignment_audit.json")
    adjusted_closes = _load_adjusted_closes(config)
    payload = build_independent_period_expansion_audit(
        adjusted_price_replay=adjusted_replay,
        canonical_replay=canonical_replay,
        data_adjustment_audit=data_adjustment,
        benchmark_relative_validation=benchmark_validation,
        adjusted_replay_alignment_audit=adjusted_alignment,
        adjusted_closes_by_symbol=adjusted_closes,
        validation_config=_validation_config(config),
        expansion_config=_expansion_config(config),
    )
    paths = IndependentPeriodExpansionAuditPaths(
        csv_path=output_dir / "independent_period_expansion_audit.csv",
        json_path=output_dir / "independent_period_expansion_audit.json",
        markdown_path=output_dir / "independent_period_expansion_audit.md",
    )
    _write_csv(paths.csv_path, payload["rows"])
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths.markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return paths


def build_independent_period_expansion_audit(
    *,
    adjusted_price_replay: dict[str, Any],
    canonical_replay: dict[str, Any],
    data_adjustment_audit: dict[str, Any] | None = None,
    benchmark_relative_validation: dict[str, Any] | None = None,
    adjusted_replay_alignment_audit: dict[str, Any] | None = None,
    adjusted_closes_by_symbol: dict[str, dict[str, float]] | None = None,
    validation_config: dict[str, Any] | None = None,
    expansion_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = validation_config or {}
    expansion = _normalized_expansion_config(expansion_config or {})
    minimum = int(config.get("min_independent_periods", 36))
    suspicious_dates = set(
        (data_adjustment_audit or {}).get("suspicious_rebalance_dates", []) or []
    )
    alignment = (adjusted_replay_alignment_audit or {}).get("alignment", {})
    adjusted_closes_by_symbol = adjusted_closes_by_symbol or {}
    no_selection_rows = _no_selected_symbol_rows(
        adjusted_price_replay=adjusted_price_replay,
        canonical_replay=canonical_replay,
    )
    rows: list[dict[str, Any]] = []
    settings: dict[str, dict[str, Any]] = {}
    for candidate_name in REPORT_CANDIDATES:
        adjusted_rows = _candidate_adjusted_rows(adjusted_price_replay, candidate_name)
        coverage = _candidate_coverage(adjusted_price_replay, candidate_name)
        candidate_settings = []
        for setting in expansion["settings"]:
            selected, skipped = _select_periods(adjusted_rows, setting)
            metrics = _setting_metrics(
                candidate_name=candidate_name,
                setting=setting,
                selected_rows=selected,
                skipped_rows=skipped,
                coverage=coverage,
                suspicious_dates=suspicious_dates,
                adjusted_closes_by_symbol=adjusted_closes_by_symbol,
                minimum_independent_periods=minimum,
                adjusted_alignment=alignment,
                benchmark_relative_validation=benchmark_relative_validation or {},
            )
            rows.append(metrics)
            candidate_settings.append(metrics)
        settings[candidate_name] = {
            "candidate_name": candidate_name,
            "settings": candidate_settings,
            "safest_expansion": _safest_expansion(candidate_settings, minimum),
            **RESEARCH_METADATA,
        }
    return {
        "mode": "independent_adjusted_period_expansion_audit_research_only",
        "purpose": (
            "Compare leakage-safe adjusted replay period selections without "
            "lowering promotion gates or rerunning models."
        ),
        "minimum_independent_periods": minimum,
        "promotion_thresholds_changed": False,
        "leakage_safety_rule": (
            "A selected period must start after the previous selected label window "
            "ends plus the configured minimum gap."
        ),
        "current_valid_adjusted_independent_periods": {
            name: int(
                _candidate_summary(adjusted_price_replay, name).get(
                    "valid_adjusted_independent_period_count"
                )
                or 0
            )
            for name in REPORT_CANDIDATES
        },
        "no_selected_symbols": no_selection_rows,
        "no_selected_symbol_summary": _no_selected_symbol_summary(no_selection_rows),
        "candidate_settings": settings,
        "rows": rows,
        "red_flags": _red_flags(rows, minimum),
        **RESEARCH_METADATA,
    }


def _no_selected_symbol_rows(
    *,
    adjusted_price_replay: dict[str, Any],
    canonical_replay: dict[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for candidate_name in REPORT_CANDIDATES:
        raw_by_date = {
            str(row.get("rebalance_date")): row
            for row in (
                canonical_replay.get("candidates", {})
                .get(candidate_name, {})
                .get("rows", [])
                or []
            )
            if isinstance(row, dict)
        }
        for period in (
            _candidate_summary(adjusted_price_replay, candidate_name)
            .get("coverage", {})
            .get("periods", [])
            or []
        ):
            if period.get("fail_closed_reason") != "no_selected_symbols":
                continue
            date = str(period.get("rebalance_date"))
            raw = raw_by_date.get(date, {})
            exposure = _number(raw.get("exposure"))
            selected_symbols = raw.get("selected_symbols", []) or []
            expected_no_position = (
                len(selected_symbols) == 0
                and (exposure is None or abs(exposure) <= 1e-12)
            )
            positive_exposure_without_symbols = (
                len(selected_symbols) == 0
                and exposure is not None
                and exposure > 1e-12
            )
            output.append({
                "candidate": candidate_name,
                "rebalance_date": date,
                "outcome_end_date": period.get("outcome_end_date"),
                "selected_symbols": [],
                "selected_symbol_count": 0,
                "exposure": exposure,
                "included_in_raw_canonical": bool(raw.get("included_in_canonical")),
                "why_no_symbols": (
                    "source replay row has zero exposure and empty target weights"
                    if expected_no_position
                    else "source replay row has positive exposure but no selected symbols"
                    if positive_exposure_without_symbols
                    else "source replay row has no selected symbols"
                ),
                "expected_no_position": expected_no_position,
                "replay_bug_suspected": not expected_no_position,
                **RESEARCH_METADATA,
            })
    return output


def _setting_metrics(
    *,
    candidate_name: str,
    setting: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
    coverage: dict[str, Any],
    suspicious_dates: set[str],
    adjusted_closes_by_symbol: dict[str, dict[str, float]],
    minimum_independent_periods: int,
    adjusted_alignment: dict[str, Any],
    benchmark_relative_validation: dict[str, Any],
) -> dict[str, Any]:
    returns = [_number(row.get("net_return")) or 0.0 for row in selected_rows]
    total_return = _compound(returns)
    anomaly_rows = [
        row for row in selected_rows
        if str(row.get("rebalance_date")) not in suspicious_dates
    ]
    anomaly_adjusted_return = _compound(
        [_number(row.get("net_return")) or 0.0 for row in anomaly_rows]
    )
    spy_return = _benchmark_return(selected_rows, adjusted_closes_by_symbol, "SPY")
    excess = None if total_return is None or spy_return is None else total_return - spy_return
    top_5_share = _top_positive_share(returns, 5)
    drawdown = _max_drawdown(returns)
    overlap_risk = "none" if not skipped_rows else "controlled_by_filter"
    failed_gates = []
    if len(selected_rows) < minimum_independent_periods:
        failed_gates.append("minimum_adjusted_independent_periods")
    if not bool(adjusted_alignment.get("aligned_correctly", False)):
        failed_gates.append("adjusted_replay_alignment")
    if not bool(coverage.get("adjusted_full_symbol_coverage", False)):
        failed_gates.append("adjusted_replay_full_symbol_coverage")
    if excess is None or excess <= 0.0:
        failed_gates.append("positive_excess_vs_spy")
    if top_5_share is not None and top_5_share > 0.50:
        failed_gates.append("top_5_date_concentration")
    return {
        "candidate": candidate_name,
        "setting": setting["name"],
        "description": setting["description"],
        "spacing": setting["spacing"],
        "minimum_gap_days": setting["minimum_gap_days"],
        "enforce_non_overlap": setting["enforce_non_overlap"],
        "leakage_safe": True,
        "overlap_risk": overlap_risk,
        "independent_period_count": len(selected_rows),
        "overlap_skipped_period_count": len(skipped_rows),
        "adjusted_coverage_ratio": coverage.get("adjusted_coverage_ratio"),
        "valid_adjusted_period_count": coverage.get("valid_adjusted_period_count"),
        "invalid_adjusted_period_count": coverage.get("invalid_adjusted_period_count"),
        "canonical_return": total_return,
        "anomaly_adjusted_return": anomaly_adjusted_return,
        "max_drawdown": drawdown,
        "top_5_positive_return_share": top_5_share,
        "benchmark_return": spy_return,
        "benchmark_excess_return": excess,
        "promotion_gate_status": "blocked" if failed_gates else "pass",
        "failed_gates": sorted(set(failed_gates)),
        "selected_rebalance_dates": [
            str(row.get("rebalance_date")) for row in selected_rows
        ],
        "skipped_overlap_dates": [
            str(row.get("rebalance_date")) for row in skipped_rows
        ],
        "source_promotion_gates_preserved": _source_gates_preserved(
            benchmark_relative_validation,
            candidate_name,
        ),
        **RESEARCH_METADATA,
    }


def _select_periods(
    rows: list[dict[str, Any]],
    setting: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = [row for row in rows if row.get("exclusion_reason") is None]
    if setting["spacing"] == "strict_non_overlap":
        candidates = [row for row in eligible if row.get("included_in_canonical")]
    else:
        candidates = _first_periods_by_bucket(eligible, setting["spacing"])
    if not setting["enforce_non_overlap"]:
        return candidates, []
    selected = []
    skipped = []
    previous_end = None
    gap = timedelta(days=int(setting["minimum_gap_days"]))
    for row in sorted(candidates, key=lambda item: str(item.get("rebalance_date"))):
        start = _date(row.get("rebalance_date"))
        end = _date(row.get("outcome_end_date")) or start
        if start is None or end is None:
            skipped.append(row)
            continue
        if previous_end is None or start >= previous_end + gap:
            selected.append(row)
            previous_end = end
        else:
            skipped.append(row)
    return selected, skipped


def _first_periods_by_bucket(
    rows: list[dict[str, Any]],
    spacing: str,
) -> list[dict[str, Any]]:
    output = {}
    for row in sorted(rows, key=lambda item: str(item.get("rebalance_date"))):
        start = _date(row.get("rebalance_date"))
        if start is None:
            continue
        if spacing == "monthly":
            key = start.strftime("%Y-%m")
        elif spacing == "quarterly":
            key = f"{start.year}-Q{((start.month - 1) // 3) + 1}"
        elif spacing == "all_valid_min_gap":
            key = str(row.get("rebalance_date"))
        else:
            key = start.strftime("%Y-%m")
        output.setdefault(key, row)
    return list(output.values())


def _candidate_adjusted_rows(
    adjusted_price_replay: dict[str, Any],
    candidate_name: str,
) -> list[dict[str, Any]]:
    return [
        row for row in (
            adjusted_price_replay.get("adjusted_canonical_replay", {})
            .get("candidates", {})
            .get(candidate_name, {})
            .get("rows", [])
            or []
        )
        if isinstance(row, dict)
    ]


def _candidate_summary(
    adjusted_price_replay: dict[str, Any],
    candidate_name: str,
) -> dict[str, Any]:
    row = adjusted_price_replay.get("candidates", {}).get(candidate_name, {})
    return row if isinstance(row, dict) else {}


def _candidate_coverage(
    adjusted_price_replay: dict[str, Any],
    candidate_name: str,
) -> dict[str, Any]:
    coverage = _candidate_summary(adjusted_price_replay, candidate_name).get(
        "coverage",
        {},
    )
    return coverage if isinstance(coverage, dict) else {}


def _benchmark_return(
    rows: list[dict[str, Any]],
    closes: dict[str, dict[str, float]],
    symbol: str,
) -> float | None:
    symbol_closes = closes.get(symbol, {})
    returns = []
    for row in rows:
        start = symbol_closes.get(str(row.get("rebalance_date")))
        end = symbol_closes.get(str(row.get("outcome_end_date")))
        if start is None or end is None or start <= 0:
            return None
        returns.append((end / start) - 1.0)
    return _compound(returns)


def _compound(returns: list[float]) -> float | None:
    if not returns:
        return None
    equity = 1.0
    for value in returns:
        equity *= 1.0 + float(value)
    return equity - 1.0


def _max_drawdown(returns: list[float]) -> float | None:
    if not returns:
        return None
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in returns:
        equity *= 1.0 + float(value)
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    return max_drawdown


def _top_positive_share(returns: list[float], top_n: int) -> float | None:
    positives = sorted((value for value in returns if value > 0.0), reverse=True)
    total = sum(positives)
    if total <= 0.0:
        return None
    return sum(positives[:top_n]) / total


def _no_selected_symbol_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_candidate: dict[str, int] = {}
    bug_rows = 0
    for row in rows:
        by_candidate[row["candidate"]] = by_candidate.get(row["candidate"], 0) + 1
        if row.get("replay_bug_suspected"):
            bug_rows += 1
    return {
        "row_count": len(rows),
        "by_candidate": by_candidate,
        "replay_bug_suspected_count": bug_rows,
        "verdict": (
            "expected_no_position_periods"
            if rows and bug_rows == 0
            else "review_required"
            if bug_rows
            else "none"
        ),
    }


def _safest_expansion(
    rows: list[dict[str, Any]],
    minimum_independent_periods: int,
) -> dict[str, Any]:
    safe_rows = [row for row in rows if row["leakage_safe"]]
    if not safe_rows:
        return {"setting": None, "reason": "no_leakage_safe_setting"}
    best = max(
        safe_rows,
        key=lambda row: (
            int(row["independent_period_count"]),
            float(row.get("benchmark_excess_return") or -999.0),
        ),
    )
    reason = (
        "best_available_but_still_below_minimum"
        if int(best["independent_period_count"]) < minimum_independent_periods
        else "meets_minimum_independent_periods"
    )
    return {
        "setting": best["setting"],
        "independent_period_count": best["independent_period_count"],
        "reason": reason,
    }


def _red_flags(rows: list[dict[str, Any]], minimum: int) -> list[str]:
    flags = []
    if not any(int(row["independent_period_count"]) >= minimum for row in rows):
        flags.append("no_expansion_setting_reaches_minimum_independent_periods")
    if any(row["promotion_gate_status"] == "blocked" for row in rows):
        flags.append("promotion_gates_remain_blocked")
    return flags


def _source_gates_preserved(
    benchmark_relative_validation: dict[str, Any],
    candidate_name: str,
) -> bool:
    candidates = benchmark_relative_validation.get("candidates", []) or []
    row = next(
        (
            item for item in candidates
            if isinstance(item, dict) and item.get("candidate_name") == candidate_name
        ),
        {},
    )
    return bool(row.get("promotion_candidate_status") == "blocked")


def _normalized_expansion_config(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("settings")
    if not isinstance(settings, list) or not settings:
        settings = [
            {
                "name": "current_strict_non_overlap",
                "description": "Current adjusted canonical non-overlap selection.",
                "spacing": "strict_non_overlap",
                "minimum_gap_days": 0,
                "enforce_non_overlap": True,
            },
            {
                "name": "monthly_leakage_safe",
                "description": "First valid adjusted period each month, then remove overlaps.",
                "spacing": "monthly",
                "minimum_gap_days": 0,
                "enforce_non_overlap": True,
            },
            {
                "name": "quarterly_leakage_safe",
                "description": "First valid adjusted period each quarter, then remove overlaps.",
                "spacing": "quarterly",
                "minimum_gap_days": 0,
                "enforce_non_overlap": True,
            },
            {
                "name": "all_valid_min_gap_0",
                "description": "All valid adjusted periods filtered by label-window non-overlap.",
                "spacing": "all_valid_min_gap",
                "minimum_gap_days": 0,
                "enforce_non_overlap": True,
            },
        ]
    normalized = []
    for item in settings:
        if not isinstance(item, dict):
            continue
        normalized.append({
            "name": str(item.get("name") or item.get("spacing") or "setting"),
            "description": str(item.get("description") or ""),
            "spacing": str(item.get("spacing") or "monthly"),
            "minimum_gap_days": int(item.get("minimum_gap_days", 0)),
            "enforce_non_overlap": bool(item.get("enforce_non_overlap", True)),
        })
    return {"settings": normalized}


def _validation_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("ml", {}).get("benchmark_relative_validation", {}) or {})


def _expansion_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("ml", {}).get("independent_period_expansion", {}) or {})


def _output_dir(config: dict[str, Any]) -> Path:
    return Path(
        config.get("ml", {}).get(
            "output_dir",
            "reports/ml/regime_transformer_meta_ensemble_v1",
        )
    )


def _load_adjusted_closes(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    adjusted = config.get("ml", {}).get("adjusted_data_source", {}) or {}
    data_dir = Path(str(adjusted.get("adjusted_data_dir", "data/reference/adjusted_prices")))
    output: dict[str, dict[str, float]] = {}
    if not data_dir.exists():
        return output
    for path in data_dir.glob("*.csv"):
        if path.name == "manifest.json":
            continue
        symbol = path.stem.upper()
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = {}
            for row in reader:
                day = row.get("date") or row.get("Date")
                close = _number(
                    row.get("adj_close")
                    or row.get("adjusted_close")
                    or row.get("Adj Close")
                    or row.get("close")
                )
                if day and close is not None and close > 0:
                    rows[str(day)[:10]] = close
            output[symbol] = rows
    return output


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "candidate",
        "setting",
        "spacing",
        "minimum_gap_days",
        "leakage_safe",
        "overlap_risk",
        "independent_period_count",
        "overlap_skipped_period_count",
        "adjusted_coverage_ratio",
        "valid_adjusted_period_count",
        "invalid_adjusted_period_count",
        "canonical_return",
        "anomaly_adjusted_return",
        "max_drawdown",
        "top_5_positive_return_share",
        "benchmark_return",
        "benchmark_excess_return",
        "promotion_gate_status",
        "failed_gates",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                name: ";".join(row.get(name, []))
                if isinstance(row.get(name), list)
                else row.get(name)
                for name in fieldnames
            })


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Independent Adjusted Period Expansion Audit",
        "",
        "Research only. Trading impact: none. Production validated: false.",
        "",
        f"Minimum independent periods: {payload.get('minimum_independent_periods')}",
        f"Promotion thresholds changed: {payload.get('promotion_thresholds_changed')}",
        "",
        "## No Selected Symbols",
        "",
        (
            f"Rows: {payload.get('no_selected_symbol_summary', {}).get('row_count')} | "
            f"Verdict: {payload.get('no_selected_symbol_summary', {}).get('verdict')}"
        ),
        "",
        "## Expansion Settings",
        "",
        "|candidate|setting|periods|coverage|return|anomaly-adjusted|drawdown|excess vs SPY|status|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("rows", []):
        lines.append(
            "|{candidate}|{setting}|{periods}|{coverage}|{ret}|{clean}|{dd}|{excess}|{status}|".format(
                candidate=row.get("candidate"),
                setting=row.get("setting"),
                periods=row.get("independent_period_count"),
                coverage=_fmt(row.get("adjusted_coverage_ratio")),
                ret=_fmt(row.get("canonical_return")),
                clean=_fmt(row.get("anomaly_adjusted_return")),
                dd=_fmt(row.get("max_drawdown")),
                excess=_fmt(row.get("benchmark_excess_return")),
                status=row.get("promotion_gate_status"),
            )
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.4f}"


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _date(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)[:10])
    except ValueError:
        return None
