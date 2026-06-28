from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from infrastructure.data.yahoo_adjusted_price_importer import (
    AdjustedPriceImportManifest,
    YahooAdjustedPriceImporter,
)


DEFAULT_ADJUSTED_SYMBOLS = ["AMSC", "AXTI", "LEU", "LUMN", "MRVL", "MU", "SPY", "QQQ"]


def run_refresh_adjusted_prices(
    config: dict,
    *,
    symbols: list[str] | None = None,
    auto_discover_replay_symbols: bool = False,
) -> AdjustedPriceImportManifest:
    ml_config = config.get("ml", {})
    adjusted_config = ml_config.get("adjusted_data_source", {}) or {}
    auto_discover = bool(
        auto_discover_replay_symbols
        or adjusted_config.get("auto_discover_replay_symbols", False)
    )
    resolved_symbols = resolve_adjusted_price_symbols(
        config,
        symbols=symbols,
        auto_discover_replay_symbols=auto_discover,
    )
    output_dir = str(
        adjusted_config.get("adjusted_data_dir", "data/reference/adjusted_prices")
    )
    start = _date(
        adjusted_config.get("start_date")
        or ml_config.get("adjusted_price_start_date")
        or "1990-01-01"
    )
    end = _date(
        adjusted_config.get("end_date")
        or ml_config.get("adjusted_price_end_date")
        or date.today().isoformat()
    )
    importer = YahooAdjustedPriceImporter(output_dir)
    manifest = importer.import_symbols(
        resolved_symbols,
        start=start,
        end=end,
        manifest_path=Path(output_dir) / "manifest.json",
    )
    print("\nADJUSTED PRICE REFRESH")
    print("mode=research | trading_impact=none")
    print(f"Source: {manifest.source}")
    print(f"Output dir: {manifest.output_dir}")
    print(f"Requested symbols: {manifest.requested_symbol_count}")
    print(f"Imported symbols: {manifest.imported_symbol_count}")
    print(f"Failed symbols: {manifest.failed_symbol_count}")
    print(f"Manifest: {Path(output_dir) / 'manifest.json'}")
    print(f"Auto-discovered replay symbols: {auto_discover}")
    for item in manifest.failed_symbols:
        print(f"Failed {item['symbol']}: {item['reason']}")
    return manifest


def resolve_adjusted_price_symbols(
    config: dict,
    *,
    symbols: list[str] | None = None,
    auto_discover_replay_symbols: bool = False,
) -> list[str]:
    ml_config = config.get("ml", {})
    adjusted_config = ml_config.get("adjusted_data_source", {}) or {}
    base_symbols = (
        symbols
        or adjusted_config.get("symbols")
        or adjusted_config.get("inspect_symbols")
        or DEFAULT_ADJUSTED_SYMBOLS
    )
    resolved = {_symbol(symbol) for symbol in base_symbols}
    if auto_discover_replay_symbols:
        resolved.update(discover_required_adjusted_symbols(config))
    return sorted(symbol for symbol in resolved if symbol)


def discover_required_adjusted_symbols(config: dict) -> list[str]:
    output_dir = _ml_output_dir(config)
    ml_config = config.get("ml", {})
    adjusted_config = ml_config.get("adjusted_data_source", {}) or {}
    symbols = {"SPY", "QQQ"}
    symbols.update(_symbol(symbol) for symbol in adjusted_config.get("inspect_symbols", []))
    symbols.update(
        _selected_replay_symbols(
            _read_json(output_dir / "canonical_continuous_equity_replay.json")
        )
    )
    symbols.update(
        _alignment_missing_symbols(
            _read_json(output_dir / "adjusted_replay_alignment_audit.json")
        )
    )
    symbols.update(
        _adjusted_replay_missing_symbols(
            _read_json(output_dir / "adjusted_price_replay.json")
        )
    )
    return sorted(symbol for symbol in symbols if symbol)


def _selected_replay_symbols(payload: dict[str, Any]) -> set[str]:
    symbols: set[str] = set()
    candidates = payload.get("candidates", {})
    if not isinstance(candidates, dict):
        return symbols
    for name in (
        "exact_champion_replay",
        "selected_bayesian_optimizer_diagnostic_policy",
    ):
        for row in candidates.get(name, {}).get("rows", []) or []:
            if not isinstance(row, dict):
                continue
            symbols.update(_symbol(symbol) for symbol in row.get("selected_symbols", []))
    return {symbol for symbol in symbols if symbol}


def _alignment_missing_symbols(payload: dict[str, Any]) -> set[str]:
    output = set()
    summaries = payload.get("candidate_summaries", {})
    if not isinstance(summaries, dict):
        return output
    for row in summaries.values():
        if not isinstance(row, dict):
            continue
        output.update(
            _symbol(symbol)
            for symbol in (
                row.get("missing_symbols")
                or row.get("missing_adjusted_symbols")
                or []
            )
        )
    return {symbol for symbol in output if symbol}


def _adjusted_replay_missing_symbols(payload: dict[str, Any]) -> set[str]:
    output = set()
    candidates = payload.get("candidates", {})
    if not isinstance(candidates, dict):
        return output
    for row in candidates.values():
        if not isinstance(row, dict):
            continue
        output.update(
            _symbol(symbol)
            for symbol in (
                row.get("missing_symbols")
                or row.get("missing_adjusted_symbols")
                or []
            )
        )
    return {symbol for symbol in output if symbol}


def _ml_output_dir(config: dict) -> Path:
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


def _symbol(value: Any) -> str:
    return str(value).upper().strip()


def _date(value: str) -> date:
    return datetime.fromisoformat(str(value)[:10]).date()
