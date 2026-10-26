"""Frozen operational contract shared by IEX raw acquisition implementations."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from application.services.hf_data_cutover_service import CutoverAuditError, CutoverPaths


MINIMUM_FINAL_RESERVE_BYTES = 15 * 1024**3
BACKFILL_REQUESTS_PER_MINUTE = 150
BACKFILL_PACING_REQUESTS_PER_MINUTE = 60
INITIAL_VALIDATED_BACKFILL_WORKERS = 32
BACKFILL_WORKERS = 8
TRANSPORT_PROBE_WORKERS = 1
TRANSPORT_PROBE_PARTITIONS = 3
BACKFILL_ASOF = "2026-10-25"


@dataclass(frozen=True, slots=True)
class IexBackfillSettings:
    start: datetime
    end: datetime
    roster: tuple[str, ...]
    data_root: Path
    provider_symbol_by_canonical: tuple[tuple[str, str], ...]


def load_iex_backfill_settings(paths: CutoverPaths) -> IexBackfillSettings:
    coverage = _read_json(paths.output_root / "IEX_HISTORICAL_COVERAGE_PROBE.json")
    roster_payload = _read_json(paths.output_root / "IEX_ACQUISITION_ROSTER.json")
    start = _parse_datetime(str(coverage["selected_uniform_acquisition_start"]))
    end = _parse_datetime(str(coverage["selected_acquisition_end_exclusive"]))
    roster = tuple(str(symbol).upper() for symbol in roster_payload["symbols"])
    if len(roster) != 514 or len(set(roster)) != 514 or not {"SPY", "QQQ"}.issubset(roster):
        raise CutoverAuditError("Frozen IEX acquisition roster failed its identity invariants")
    return IexBackfillSettings(
        start=start,
        end=end,
        roster=roster,
        data_root=paths.repository_root / "data/market/iex",
        provider_symbol_by_canonical=alpaca_symbol_aliases(paths.repository_root, roster),
    )


def alpaca_symbol_aliases(
    repository_root: Path,
    roster: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    assets_path = repository_root / "data/reference/assets/canonical_asset_registry.csv"
    aliases_path = repository_root / "data/reference/assets/provider_symbol_aliases.csv"
    if not assets_path.exists() or not aliases_path.exists():
        raise CutoverAuditError("Canonical asset and provider alias registries are required for IEX acquisition")
    with assets_path.open(encoding="utf-8-sig", newline="") as handle:
        canonical_by_asset = {
            str(row["asset_id"]): str(row["canonical_symbol"]).upper()
            for row in csv.DictReader(handle)
            if str(row.get("canonical_symbol") or "").strip()
        }
    resolved: dict[str, str] = {}
    with aliases_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("provider") or "").lower() != "alpaca":
                continue
            if str(row.get("is_primary") or "").lower() not in {"1", "true", "yes"}:
                continue
            canonical = canonical_by_asset.get(str(row.get("asset_id") or ""))
            provider_symbol = str(row.get("provider_symbol") or "").upper()
            if canonical in roster and provider_symbol and provider_symbol != canonical:
                resolved[canonical] = provider_symbol
    unresolved_special = sorted(symbol for symbol in roster if "-" in symbol and symbol not in resolved)
    if unresolved_special:
        raise CutoverAuditError(f"Alpaca aliases missing for canonical special symbols: {unresolved_special}")
    return tuple(sorted(resolved.items()))


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timezone required: {value}")
    return parsed


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


__all__ = [
    "BACKFILL_ASOF",
    "BACKFILL_PACING_REQUESTS_PER_MINUTE",
    "BACKFILL_REQUESTS_PER_MINUTE",
    "BACKFILL_WORKERS",
    "INITIAL_VALIDATED_BACKFILL_WORKERS",
    "IexBackfillSettings",
    "MINIMUM_FINAL_RESERVE_BYTES",
    "TRANSPORT_PROBE_PARTITIONS",
    "TRANSPORT_PROBE_WORKERS",
    "alpaca_symbol_aliases",
    "load_iex_backfill_settings",
]
