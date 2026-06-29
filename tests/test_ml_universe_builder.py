from __future__ import annotations

import json

import yaml

from core.research.ml.data.universe_builder import build_universe_files


def test_universe_builder_filters_by_history_latest_date_and_liquidity(tmp_path):
    inventory_path = tmp_path / "data_inventory.json"
    output_dir = tmp_path / "universes"
    inventory_path.write_text(
        json.dumps({
            "symbols": [
                _symbol("AAA", included=True, adv=300_000_000),
                _symbol("BBB", included=True, adv=100_000_000),
                _symbol("CCC", included=False, adv=500_000_000),
            ]
        }),
        encoding="utf-8",
    )

    results = build_universe_files(
        inventory_path=inventory_path,
        output_dir=output_dir,
        min_history_years=9,
        max_latest_gap_days=14,
        min_average_dollar_volume_252d=50_000_000,
    )

    liquid_100 = yaml.safe_load((output_dir / "us_liquid_100.yaml").read_text())
    current = yaml.safe_load((output_dir / "current_32.yaml").read_text())

    assert [result.name for result in results] == [
        "current_32",
        "us_liquid_100",
        "us_liquid_250",
        "us_liquid_500",
    ]
    assert liquid_100["symbols"] == ["AAA", "BBB"]
    assert liquid_100["available_count"] == 2
    assert current["symbols"] == ["AAA", "BBB", "CCC"]


def _symbol(symbol: str, included: bool, adv: float) -> dict:
    return {
        "symbol": symbol,
        "included": included,
        "passes_min_history_years": included,
        "passes_latest_date_check": included,
        "passes_liquidity_check": included,
        "average_dollar_volume_252d": adv,
    }
