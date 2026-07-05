from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from core.research.ml.independent_period_expansion_audit_math import _number


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
