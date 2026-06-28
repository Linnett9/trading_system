from __future__ import annotations

import csv
import inspect
import json
from datetime import date, datetime, timezone

from infrastructure.data import yahoo_adjusted_price_importer
from infrastructure.data.yahoo_adjusted_price_importer import (
    SOURCE_NAME,
    YahooAdjustedPriceImporter,
)


class FakeYahooChartClient:
    def __init__(self, payloads: dict[str, dict]):
        self.payloads = payloads
        self.calls: list[tuple[str, date, date]] = []

    def fetch_chart(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> dict:
        self.calls.append((symbol, start, end))
        return self.payloads[symbol]


def test_importer_writes_adjusted_ohlcv_csv_from_mocked_yahoo_data(tmp_path):
    client = FakeYahooChartClient(
        {
            "AAA": _chart_payload(
                open_values=[90.0, 110.0],
                high_values=[120.0, 130.0],
                low_values=[80.0, 100.0],
                close_values=[100.0, 120.0],
                adjusted_close_values=[50.0, 60.0],
                volumes=[1000, 1200],
            )
        }
    )
    importer = YahooAdjustedPriceImporter(
        str(tmp_path / "adjusted_prices"),
        client=client,
    )

    manifest = importer.import_symbols(
        ["AAA"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 2),
    )

    output_path = tmp_path / "adjusted_prices" / "AAA.csv"
    rows = list(csv.DictReader(output_path.open("r", encoding="utf-8")))

    assert [field.lower() for field in rows[0].keys()] == [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert rows[0]["symbol"] == "AAA"
    assert rows[0]["date"] == "2024-01-01"
    assert float(rows[0]["open"]) == 45.0
    assert float(rows[0]["high"]) == 60.0
    assert float(rows[0]["low"]) == 40.0
    assert float(rows[0]["close"]) == 50.0
    assert float(rows[0]["adj_close"]) == 50.0
    assert float(rows[0]["volume"]) == 1000.0
    assert manifest.imported_symbol_count == 1
    assert client.calls == [("AAA", date(2024, 1, 1), date(2024, 1, 2))]


def test_importer_manifest_records_source_download_date_counts_and_ranges(tmp_path):
    importer = YahooAdjustedPriceImporter(
        str(tmp_path / "adjusted_prices"),
        client=FakeYahooChartClient({"AAA": _chart_payload()}),
    )

    manifest = importer.import_symbols(
        ["AAA"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 2),
    )
    manifest_payload = json.loads(
        (tmp_path / "adjusted_prices" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest.source == SOURCE_NAME
    assert manifest_payload["source"] == SOURCE_NAME
    assert manifest_payload["download_date"]
    assert manifest_payload["requested_symbol_count"] == 1
    assert manifest_payload["imported_symbol_count"] == 1
    assert manifest_payload["failed_symbol_count"] == 0
    assert manifest_payload["research_only"] is True
    assert manifest_payload["trading_impact"] == "none"
    assert manifest_payload["symbols"][0]["symbol"] == "AAA"
    assert manifest_payload["symbols"][0]["first_date"] == "2024-01-01"
    assert manifest_payload["symbols"][0]["last_date"] == "2024-01-02"


def test_importer_records_failed_symbol_when_adjusted_source_is_missing(tmp_path):
    importer = YahooAdjustedPriceImporter(
        str(tmp_path / "adjusted_prices"),
        client=FakeYahooChartClient(
            {
                "GOOD": _chart_payload(),
                "MISS": {"chart": {"result": [], "error": None}},
            }
        ),
    )

    manifest = importer.import_symbols(
        ["GOOD", "MISS"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 2),
    )

    assert manifest.imported_symbol_count == 1
    assert manifest.failed_symbol_count == 1
    assert manifest.failed_symbols[0]["symbol"] == "MISS"
    assert "returned no result" in manifest.failed_symbols[0]["reason"]
    assert not (tmp_path / "adjusted_prices" / "MISS.csv").exists()


def test_importer_preserves_raw_stooq_parquet_separately(tmp_path):
    raw_dir = tmp_path / "raw_stooq"
    raw_dir.mkdir()
    raw_path = raw_dir / "AAA.parquet"
    raw_path.write_text("raw stooq bytes stay put", encoding="utf-8")
    importer = YahooAdjustedPriceImporter(
        str(tmp_path / "adjusted_prices"),
        client=FakeYahooChartClient({"AAA": _chart_payload()}),
    )

    importer.import_symbols(
        ["AAA"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 2),
    )

    assert raw_path.read_text(encoding="utf-8") == "raw stooq bytes stay put"
    assert (tmp_path / "adjusted_prices" / "AAA.csv").exists()


def test_yahoo_adjusted_price_importer_has_no_operational_imports_or_references():
    source = inspect.getsource(yahoo_adjusted_price_importer)

    assert "infrastructure.broker" not in source
    assert "paper_trading" not in source
    assert "paper_commands" not in source
    assert "live_trading" not in source
    assert "core.entities.order" not in source
    assert "order_execution" not in source


def _chart_payload(
    *,
    open_values: list[float] | None = None,
    high_values: list[float] | None = None,
    low_values: list[float] | None = None,
    close_values: list[float] | None = None,
    adjusted_close_values: list[float] | None = None,
    volumes: list[int] | None = None,
) -> dict:
    timestamps = [
        int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()),
        int(datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp()),
    ]
    return {
        "chart": {
            "result": [
                {
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": open_values or [100.0, 110.0],
                                "high": high_values or [101.0, 111.0],
                                "low": low_values or [99.0, 109.0],
                                "close": close_values or [100.0, 110.0],
                                "volume": volumes or [1000, 1100],
                            }
                        ],
                        "adjclose": [
                            {
                                "adjclose": adjusted_close_values
                                or [100.0, 110.0],
                            }
                        ],
                    },
                }
            ],
            "error": None,
        }
    }
