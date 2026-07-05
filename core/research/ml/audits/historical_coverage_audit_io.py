from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from core.research.ml.audits.historical_coverage_audit_types import NOTICE


def _audit_config(config: dict[str, Any]) -> dict[str, Any]:
    validation = dict(
        config.get("ml", {}).get("benchmark_relative_validation", {}) or {}
    )
    return validation
def _output_dir(config: dict[str, Any]) -> Path:
    return Path(
        config.get("ml", {}).get(
            "output_dir",
            "reports/ml/regime_transformer_meta_ensemble_v1",
        )
    )
def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}
def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    fieldnames = [
        "layer",
        "earliest_date",
        "latest_date",
        "rebalance_date_count",
        "independent_period_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload.get("rows", []))
def _markdown(payload: dict[str, Any]) -> str:
    bottleneck = payload.get("historical_bottleneck", {})
    lines = [
        "# Historical Coverage Audit",
        "",
        NOTICE,
        "",
        f"Current bottleneck: {bottleneck.get('limiting_layer')}",
        f"Minimum independent periods: {payload.get('minimum_independent_periods')}",
        f"Full model rerun required: {payload.get('full_model_rerun_required')}",
        "",
        "|layer|earliest|latest|rebalance dates|independent periods|",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload.get("rows", []):
        lines.append(
            "|{layer}|{earliest}|{latest}|{rebalance}|{independent}|".format(
                layer=row.get("layer"),
                earliest=row.get("earliest_date"),
                latest=row.get("latest_date"),
                rebalance=row.get("rebalance_date_count"),
                independent=row.get("independent_period_count"),
            )
        )
    lines.extend([
        "",
        "## Blockers",
        "",
        *[f"- {item}" for item in payload.get("blockers", [])],
        "",
        "## Overnight Command",
        "",
        f"`{payload.get('overnight_command_if_rerun_justified')}`",
        "",
    ])
    return "\n".join(lines)
