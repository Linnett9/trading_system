from __future__ import annotations

import inspect
import math
from datetime import date, timedelta

import pytest

from core.research.ml.stock_level import stock_level_alpha_features
from core.research.ml.stock_level.stock_level_alpha_features import (
    ENGINEERED_FEATURE_COLUMNS,
    build_stock_level_alpha_features,
)


@pytest.mark.parametrize("feature", ENGINEERED_FEATURE_COLUMNS)
def test_each_engineered_feature_is_populated(feature):
    rows, _ = build_stock_level_alpha_features(_rows(), _histories())

    assert all(row[feature] != "" for row in rows)
    assert all(math.isfinite(float(row[feature])) for row in rows)


def test_momentum_250d_uses_prior_prices():
    histories = _histories()
    rows, _ = build_stock_level_alpha_features(_rows(), histories)
    aaa = next(row for row in rows if row["symbol"] == "AAA")
    closes = [float(row["close"]) for row in histories["AAA"]]

    expected = closes[-1] / closes[-251] - 1.0

    assert aaa["momentum_250d"] == pytest.approx(expected)


def test_relative_features_use_same_date_cross_section():
    rows, _ = build_stock_level_alpha_features(_rows(), _histories())
    tech = [row for row in rows if row["sector"] == "Technology"]

    assert sum(float(row["relative_momentum_vs_sector"]) for row in tech) == pytest.approx(0.0)
    assert {float(row["sector_relative_strength"]) for row in tech} == {0.25, 0.75}
    assert all(0.0 <= float(row["momentum_percentile"]) <= 1.0 for row in rows)


def test_future_prices_do_not_change_earlier_features():
    histories = _histories()
    first, _ = build_stock_level_alpha_features(_rows(), histories)
    changed = {symbol: [dict(row) for row in history] for symbol, history in histories.items()}
    for symbol in changed:
        changed[symbol].append(
            {
                "date": "2025-12-31",
                "close": 100_000.0,
                "high": 110_000.0,
                "low": 90_000.0,
            }
        )
    second, _ = build_stock_level_alpha_features(_rows(), changed)

    assert [
        {feature: row[feature] for feature in ENGINEERED_FEATURE_COLUMNS}
        for row in first
    ] == [
        {feature: row[feature] for feature in ENGINEERED_FEATURE_COLUMNS}
        for row in second
    ]


def test_enrichment_preserves_source_columns_and_audits_availability():
    source = _rows()
    rows, audit = build_stock_level_alpha_features(source, _histories())

    assert len(rows) == len(source)
    assert audit["source_columns_preserved"] is True
    assert audit["unique_symbol_date_rows"] is True
    assert audit["engineered_feature_count"] == len(ENGINEERED_FEATURE_COLUMNS)
    assert all(item["missing_count"] == 0 for item in audit["features"])
    assert all(
        rows[index][column] == source[index][column]
        for index in range(len(source))
        for column in source[index]
    )


def test_industry_relative_strength_is_missing_when_industry_is_unavailable():
    source = [{key: value for key, value in row.items() if key != "industry"} for row in _rows()]
    rows, audit = build_stock_level_alpha_features(source, _histories())
    industry_audit = next(
        row for row in audit["features"] if row["feature"] == "industry_relative_strength"
    )

    assert all(row["industry_relative_strength"] == "" for row in rows)
    assert industry_audit["availability_rate"] == 0.0
    assert audit["industry_metadata_available"] is False


def test_alpha_feature_pipeline_has_no_operational_imports():
    source = inspect.getsource(stock_level_alpha_features)

    assert "infrastructure.broker" not in source
    assert "paper_trading" not in source
    assert "paper_commands" not in source
    assert "live_trading" not in source
    assert "core.entities.order" not in source
    assert "order_execution" not in source


def _rows() -> list[dict[str, str]]:
    definitions = (
        ("AAA", "Technology", "Software", "0.30"),
        ("BBB", "Technology", "Hardware", "0.15"),
        ("CCC", "Materials", "Mining", "-0.05"),
    )
    return [
        {
            "rebalance_date": "2025-03-01",
            "symbol": symbol,
            "sector": sector,
            "industry": industry,
            "predicted_momentum_120d": momentum,
            "sentinel": f"preserve-{symbol}",
        }
        for symbol, sector, industry, momentum in definitions
    ]


def _histories() -> dict[str, list[dict[str, float | str]]]:
    return {
        "AAA": _history(100.0, 0.0012, 0.025),
        "BBB": _history(80.0, 0.0007, 0.018),
        "CCC": _history(60.0, -0.0001, 0.030),
        "SPY": _history(120.0, 0.0005, 0.012),
    }


def _history(
    start: float,
    trend: float,
    oscillation: float,
) -> list[dict[str, float | str]]:
    output = []
    start_date = date(2024, 1, 1)
    for index in range(420):
        value = start * (1.0 + trend * index + oscillation * math.sin(index / 7.0))
        output.append(
            {
                "date": (start_date + timedelta(days=index)).isoformat(),
                "close": value,
                "high": value * (1.01 + 0.002 * math.sin(index / 5.0)),
                "low": value * (0.99 - 0.002 * math.cos(index / 6.0)),
            }
        )
    return output
