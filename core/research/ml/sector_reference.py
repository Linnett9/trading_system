from __future__ import annotations

import json
from pathlib import Path


def load_sector_by_symbol(
    reference_path: str | None,
    inline_mapping: dict[str, str] | None = None,
) -> dict[str, str]:
    """Load a local sector reference, preserving legacy inline mappings."""
    sectors = {
        str(symbol).upper(): str(sector)
        for symbol, sector in (inline_mapping or {}).items()
        if str(sector).strip()
    }
    if not reference_path:
        return sectors

    path = Path(reference_path)
    if not path.exists():
        raise FileNotFoundError(f"Sector reference file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(
        isinstance(symbol, str) and isinstance(sector, str)
        for symbol, sector in payload.items()
    ):
        raise ValueError(
            "Sector reference must be a JSON object mapping symbols to sector names"
        )
    sectors.update({
        symbol.upper(): sector
        for symbol, sector in payload.items()
        if sector.strip()
    })
    return sectors
