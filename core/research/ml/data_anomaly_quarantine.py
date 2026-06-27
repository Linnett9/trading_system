from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.research.ml.canonical_continuous_equity_replay import (
    build_canonical_replay,
)


RESEARCH_METADATA = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
}
NOTICE = "Research only. Trading impact: none. Production validated: false."


@dataclass(frozen=True)
class DataAnomalyQuarantinePaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path


def detect_period_anomalies(
    period_rows: list[dict[str, Any]],
    *,
    large_symbol_return_abs: float = 1.0,
    large_portfolio_return_abs: float = 0.50,
) -> list[dict[str, Any]]:
    """Detect existing quarantine anomalies in period rows without report I/O."""
    anomalies = []
    for row in period_rows:
        for anomaly in row.get("symbol_return_anomalies", []) or []:
            symbol_return = _number(anomaly.get("return"))
            if symbol_return is None or abs(symbol_return) <= large_symbol_return_abs:
                continue
            anomalies.append({
                "anomaly_type": "large_symbol_period_return",
                "rebalance_date": row.get("rebalance_date"),
                "outcome_end_date": row.get("outcome_end_date"),
                "symbol": anomaly.get("symbol"),
                "symbol_return": symbol_return,
                "portfolio_period_return": _number(row.get("period_return")),
                "start_close": _number(anomaly.get("start_close")),
                "end_close": _number(anomaly.get("end_close")),
                "severity": _severity(symbol_return),
                "adjusted_status": "unknown",
                **RESEARCH_METADATA,
            })
        period_return = _number(row.get("period_return"))
        if period_return is None or abs(period_return) <= large_portfolio_return_abs:
            continue
        anomalies.append({
            "anomaly_type": "large_portfolio_period_return",
            "rebalance_date": row.get("rebalance_date"),
            "outcome_end_date": row.get("outcome_end_date"),
            "symbol": None,
            "symbol_return": None,
            "portfolio_period_return": period_return,
            "start_close": None,
            "end_close": None,
            "severity": _severity(period_return),
            "adjusted_status": "unknown",
            **RESEARCH_METADATA,
        })
    return anomalies


def write_data_anomaly_quarantine(
    config: dict[str, Any],
    *,
    exclude_flagged: bool = False,
) -> DataAnomalyQuarantinePaths:
    output_dir = _meta_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    champion_audit = _read_json(output_dir / "champion_baseline_audit.json")
    canonical = _read_json(output_dir / "canonical_continuous_equity_replay.json")
    selected_optimizer = _read_json(output_dir / "selected_optimizer_exposure_path.json")
    payload = build_anomaly_quarantine_report(
        champion_audit=champion_audit,
        canonical_replay=canonical,
        selected_optimizer=selected_optimizer,
        exclude_flagged=exclude_flagged,
    )
    paths = DataAnomalyQuarantinePaths(
        csv_path=output_dir / "anomaly_quarantine_report.csv",
        json_path=output_dir / "anomaly_quarantine_report.json",
        markdown_path=output_dir / "anomaly_quarantine_report.md",
    )
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(paths.csv_path, payload.get("anomalies", []))
    paths.markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return paths


def build_anomaly_quarantine_report(
    *,
    champion_audit: dict[str, Any],
    canonical_replay: dict[str, Any],
    selected_optimizer: dict[str, Any],
    exclude_flagged: bool = False,
) -> dict[str, Any]:
    anomalies = detect_period_anomalies(_champion_period_rows(champion_audit))
    flagged_dates = sorted({
        str(row["rebalance_date"])
        for row in anomalies
        if row.get("rebalance_date")
    })
    flagged_symbols = sorted({
        str(row["symbol"])
        for row in anomalies
        if row.get("symbol")
    })
    exclusion_preview = None
    if exclude_flagged:
        exclusion_preview = build_canonical_replay(
            selected_optimizer=selected_optimizer,
            champion_audit=champion_audit,
            excluded_dates=set(flagged_dates),
        )
    return {
        "mode": "data_anomaly_quarantine_research_only",
        "exclude_flagged": exclude_flagged,
        "thresholds": {
            "large_symbol_period_return_abs": 1.0,
            "large_portfolio_period_return_abs": 0.50,
            "close_must_be_positive": True,
        },
        "adjusted_status": champion_audit.get("stooq_adjustment_audit", {}).get(
            "adjusted_status",
            "unknown",
        ),
        "data_path": champion_audit.get("stooq_adjustment_audit", {}).get(
            "data_path"
        ),
        "price_column_used": champion_audit.get("stooq_adjustment_audit", {}).get(
            "price_column_used"
        ),
        "anomaly_count": len(anomalies),
        "flagged_rebalance_dates": flagged_dates,
        "flagged_symbols": flagged_symbols,
        "top_return_contributors": _top_return_contributors(canonical_replay),
        "exclusion_preview": _preview_summary(exclusion_preview),
        "anomalies": anomalies,
        "red_flags": _red_flags(anomalies, champion_audit),
        **RESEARCH_METADATA,
    }


def _champion_period_rows(champion_audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = champion_audit.get("exact_champion_replay", {}).get("period_rows", [])
    return [row for row in rows if isinstance(row, dict)]


def _top_return_contributors(canonical_replay: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for name, candidate in canonical_replay.get("candidates", {}).items():
        rows = [
            row for row in candidate.get("rows", [])
            if row.get("included_in_canonical")
        ]
        rows = sorted(
            rows,
            key=lambda row: float(row.get("net_return") or 0.0),
            reverse=True,
        )
        output[name] = [
            {
                "rebalance_date": row.get("rebalance_date"),
                "outcome_end_date": row.get("outcome_end_date"),
                "net_return": _number(row.get("net_return")),
                "selected_symbols": row.get("selected_symbols", []),
            }
            for row in rows[:20]
        ]
    return output


def _preview_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    return {
        name: {
            "canonical_continuous_return": candidate.get(
                "canonical_continuous_equity", {}
            ).get("total_return"),
            "non_overlap_rows": candidate.get(
                "canonical_continuous_equity", {}
            ).get("row_count"),
        }
        for name, candidate in payload.get("candidates", {}).items()
    }


def _red_flags(
    anomalies: list[dict[str, Any]],
    champion_audit: dict[str, Any],
) -> list[str]:
    flags = []
    if anomalies:
        flags.append("anomalies_present")
    adjusted = str(
        champion_audit.get("stooq_adjustment_audit", {}).get("adjusted_status", "")
    )
    if "unknown" in adjusted.lower():
        flags.append("stooq_adjustment_status_unknown")
    if any(row.get("severity") == "extreme" for row in anomalies):
        flags.append("extreme_symbol_or_period_moves_present")
    return sorted(set(flags))


def _severity(value: float | None) -> str:
    if value is None:
        return "unknown"
    magnitude = abs(value)
    if magnitude >= 1.0:
        return "extreme"
    if magnitude >= 0.50:
        return "large"
    return "moderate"


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "anomaly_type",
        "rebalance_date",
        "outcome_end_date",
        "symbol",
        "symbol_return",
        "portfolio_period_return",
        "start_close",
        "end_close",
        "severity",
        "adjusted_status",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({name: row.get(name) for name in fieldnames} for row in rows)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Data Anomaly Quarantine",
        "",
        NOTICE,
        "",
        f"Anomaly count: {payload.get('anomaly_count', 0)}",
        f"Flagged dates: {len(payload.get('flagged_rebalance_dates', []))}",
        f"Flagged symbols: {', '.join(payload.get('flagged_symbols', [])) or 'none'}",
        f"Adjusted status: {payload.get('adjusted_status')}",
        "",
        "## Red Flags",
        "",
    ]
    lines.extend(f"- {flag}" for flag in payload.get("red_flags", []))
    lines.extend(["", NOTICE, ""])
    return "\n".join(lines)
