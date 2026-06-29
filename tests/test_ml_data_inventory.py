from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

import application.cli as cli
from core.research.ml.data.data_inventory import build_data_inventory, inspect_symbol_file


def test_data_inventory_writes_reports_for_small_parquet_files(tmp_path):
    pytest.importorskip("pyarrow")
    parquet_dir = tmp_path / "parquet"
    output_dir = tmp_path / "reports"
    _write_parquet(
        parquet_dir / "AAPL.parquet",
        start=date(2015, 1, 1),
        days=2600,
        close=100.0,
        volume=1_000_000,
    )

    inventory = build_data_inventory(
        parquet_dir=parquet_dir,
        output_dir=output_dir,
        min_history_years=5,
        max_latest_gap_days=20,
        min_average_dollar_volume_252d=10_000_000,
        as_of_date=date(2022, 2, 1),
    )

    assert len(inventory) == 1
    assert inventory[0].symbol == "AAPL"
    assert inventory[0].passes_liquidity_check
    assert (output_dir / "data_inventory.json").exists()
    assert (output_dir / "symbol_coverage.csv").exists()


def test_missing_volume_does_not_crash_inventory(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "NO_VOLUME.parquet"
    _write_parquet(
        path,
        start=date(2015, 1, 1),
        days=2600,
        close=100.0,
        volume=None,
    )

    item = inspect_symbol_file(
        path,
        min_history_years=5,
        max_latest_gap_days=20,
        as_of_date=date(2022, 2, 1),
    )

    assert item.zero_volume_days is None
    assert item.average_dollar_volume_252d is None
    assert item.excluded_reason == "missing_liquidity_data"


def test_ml_data_inventory_cli_mode_does_not_build_feed(monkeypatch, tmp_path):
    pytest.importorskip("pyarrow")
    parquet_dir = tmp_path / "parquet"
    _write_parquet(
        parquet_dir / "AAPL.parquet",
        start=date(2015, 1, 1),
        days=2600,
        close=100.0,
        volume=1_000_000,
    )
    args = SimpleNamespace(
        mode="ml-data-inventory",
        config="config/config.yaml",
        symbols=None,
    )
    config = {
        "ml": {
            "parquet_dir": str(parquet_dir),
            "inventory_output_dir": str(tmp_path / "reports"),
            "min_history_years": 5,
            "max_latest_gap_days": 20,
            "min_average_dollar_volume_252d": 10_000_000,
        }
    }
    monkeypatch.setattr(cli, "parse_args", lambda: args)
    monkeypatch.setattr(cli, "load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(cli, "apply_runtime_overrides", lambda loaded, parsed: loaded)
    monkeypatch.setattr(
        cli,
        "build_feed",
        lambda config: (_ for _ in ()).throw(AssertionError("feed should not build")),
    )

    cli.run_cli()

    assert (tmp_path / "reports" / "data_inventory.json").exists()


def _write_parquet(
    path,
    start: date,
    days: int,
    close: float,
    volume: int | None,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(days):
        row = {
            "timestamp": datetime.combine(start + timedelta(days=index), datetime.min.time()),
            "close": close + index * 0.01,
        }
        if volume is not None:
            row["volume"] = volume
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), path)
