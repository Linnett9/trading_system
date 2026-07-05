from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_stooq_price_rows_by_symbol(
    data_dir: Path,
    symbols: list[str],
) -> dict[str, list[dict[str, Any]]]:
    return {
        symbol: _load_stooq_price_rows(data_dir / f"{symbol}.parquet")
        for symbol in symbols
    }


def _load_stooq_price_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Stooq adjustment audit requires pyarrow to read local parquet data"
        ) from exc
    table = pq.read_table(path)
    columns = table.to_pydict()
    names = list(columns)
    row_count = len(columns[names[0]]) if names else 0
    rows = []
    for index in range(row_count):
        rows.append({name: columns[name][index] for name in names})
    return rows
