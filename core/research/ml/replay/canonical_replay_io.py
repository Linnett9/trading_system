from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from core.research.ml.replay.canonical_replay_math import _number
from core.research.ml.replay.canonical_replay_types import NOTICE


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
