from __future__ import annotations

import inspect
import json
from datetime import date

from application.services import adjusted_price_commands
from application.services.adjusted_price_commands import (
    discover_required_adjusted_symbols,
    resolve_adjusted_price_symbols,
    run_refresh_adjusted_prices,
)
from infrastructure.data.yahoo_adjusted_price_importer import (
    AdjustedPriceImportManifest,
)


def test_required_replay_symbol_discovery_reads_synthetic_artifacts(tmp_path):
    (tmp_path / "canonical_continuous_equity_replay.json").write_text(
        json.dumps({
            "candidates": {
                "exact_champion_replay": {
                    "rows": [
                        {"selected_symbols": ["AAA", "BBB"]},
                    ],
                },
                "selected_bayesian_optimizer_diagnostic_policy": {
                    "rows": [
                        {"selected_symbols": ["CCC", "AAA"]},
                    ],
                },
            }
        }),
        encoding="utf-8",
    )
    (tmp_path / "adjusted_replay_alignment_audit.json").write_text(
        json.dumps({
            "candidate_summaries": {
                "exact_champion_replay": {
                    "missing_symbols": ["MISS"],
                }
            }
        }),
        encoding="utf-8",
    )
    (tmp_path / "adjusted_price_replay.json").write_text(
        json.dumps({
            "candidates": {
                "exact_champion_replay": {
                    "missing_adjusted_symbols": ["LATE"],
                }
            }
        }),
        encoding="utf-8",
    )
    config = {
        "ml": {
            "output_dir": str(tmp_path),
            "adjusted_data_source": {
                "symbols": ["BASE"],
                "inspect_symbols": ["AMSC"],
            },
        }
    }

    discovered = discover_required_adjusted_symbols(config)
    resolved = resolve_adjusted_price_symbols(
        config,
        auto_discover_replay_symbols=True,
    )

    assert discovered == [
        "AAA",
        "AMSC",
        "BBB",
        "CCC",
        "LATE",
        "MISS",
        "QQQ",
        "SPY",
    ]
    assert resolved == [
        "AAA",
        "AMSC",
        "BASE",
        "BBB",
        "CCC",
        "LATE",
        "MISS",
        "QQQ",
        "SPY",
    ]


def test_refresh_adjusted_prices_records_missing_downloads_in_manifest(
    monkeypatch,
    tmp_path,
):
    captured: dict[str, object] = {}

    class FakeImporter:
        def __init__(self, output_dir: str):
            captured["output_dir"] = output_dir

        def import_symbols(
            self,
            symbols: list[str],
            *,
            start: date,
            end: date,
            manifest_path,
        ) -> AdjustedPriceImportManifest:
            captured["symbols"] = symbols
            captured["start"] = start
            captured["end"] = end
            captured["manifest_path"] = manifest_path
            return AdjustedPriceImportManifest(
                source="fake_adjusted_source",
                output_dir=str(tmp_path / "adjusted"),
                download_date="2026-06-27",
                requested_symbols=symbols,
                requested_symbol_count=len(symbols),
                imported_symbol_count=1,
                failed_symbol_count=1,
                symbols=[
                    {
                        "symbol": "GOOD",
                        "output_path": "GOOD.csv",
                        "row_count": 2,
                        "first_date": "2024-01-01",
                        "last_date": "2024-01-02",
                        "source": "fake_adjusted_source",
                        "adjusted_ohlc": True,
                    }
                ],
                failed_symbols=[
                    {"symbol": "MISS", "reason": "not available"},
                ],
            )

    monkeypatch.setattr(
        adjusted_price_commands,
        "YahooAdjustedPriceImporter",
        FakeImporter,
    )
    config = {
        "ml": {
            "adjusted_data_source": {
                "adjusted_data_dir": str(tmp_path / "adjusted"),
                "symbols": ["GOOD", "MISS"],
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
            }
        }
    }

    manifest = run_refresh_adjusted_prices(config)

    assert captured["symbols"] == ["GOOD", "MISS"]
    assert manifest.failed_symbol_count == 1
    assert manifest.failed_symbols == [{"symbol": "MISS", "reason": "not available"}]


def test_refresh_adjusted_prices_does_not_overwrite_raw_stooq_data(
    monkeypatch,
    tmp_path,
):
    raw_dir = tmp_path / "stooq_parquet"
    raw_dir.mkdir()
    raw_path = raw_dir / "GOOD.parquet"
    raw_path.write_text("raw-stooq-data", encoding="utf-8")

    class FakeImporter:
        def __init__(self, output_dir: str):
            self.output_dir = output_dir

        def import_symbols(
            self,
            symbols: list[str],
            *,
            start: date,
            end: date,
            manifest_path,
        ) -> AdjustedPriceImportManifest:
            return AdjustedPriceImportManifest(
                source="fake_adjusted_source",
                output_dir=self.output_dir,
                download_date="2026-06-27",
                requested_symbols=symbols,
                requested_symbol_count=len(symbols),
                imported_symbol_count=len(symbols),
                failed_symbol_count=0,
                symbols=[],
                failed_symbols=[],
            )

    monkeypatch.setattr(
        adjusted_price_commands,
        "YahooAdjustedPriceImporter",
        FakeImporter,
    )

    run_refresh_adjusted_prices({
        "data": {"data_dir": str(raw_dir)},
        "ml": {
            "adjusted_data_source": {
                "adjusted_data_dir": str(tmp_path / "adjusted"),
                "symbols": ["GOOD"],
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
            }
        },
    })

    assert raw_path.read_text(encoding="utf-8") == "raw-stooq-data"


def test_adjusted_price_commands_have_no_operational_imports_or_references():
    source = inspect.getsource(adjusted_price_commands)

    assert "infrastructure.broker" not in source
    assert "paper_trading" not in source
    assert "paper_commands" not in source
    assert "live_trading" not in source
    assert "core.entities.order" not in source
    assert "order_execution" not in source
