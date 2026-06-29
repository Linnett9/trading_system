from __future__ import annotations

import inspect
import json

import pytest

from core.research.ml import cross_sectional_ranking_diagnostics
from core.research.ml.metrics.cross_sectional_ranking_diagnostics import (
    build_cross_sectional_ranking_diagnostics,
    write_cross_sectional_ranking_diagnostics,
)


def test_top_decile_selection_scores_best_rows_within_date():
    payload = build_cross_sectional_ranking_diagnostics(
        [
            _row("2024-01-01", index, signal=10 - index, target=0.10 - index * 0.01)
            for index in range(10)
        ]
    )

    signal = _signal(payload, "meta_predicted_forward_return_10d")

    assert signal["evaluated_date_count"] == 1
    assert signal["top_decile_forward_return"] == pytest.approx(0.10)
    assert signal["bottom_decile_forward_return"] == pytest.approx(0.01)
    assert signal["top_minus_bottom_spread"] == pytest.approx(0.09)
    assert signal["hit_rate_top_decile"] == 1.0


def test_rank_correlation_is_grouped_by_rebalance_date():
    rows = [
        _row("2024-01-01", 0, signal=1.0, target=0.03),
        _row("2024-01-01", 1, signal=2.0, target=0.02),
        _row("2024-01-01", 2, signal=3.0, target=0.01),
        _row("2024-02-01", 0, signal=1.0, target=0.01),
        _row("2024-02-01", 1, signal=2.0, target=0.02),
        _row("2024-02-01", 2, signal=3.0, target=0.03),
    ]

    payload = build_cross_sectional_ranking_diagnostics(rows)
    signal = _signal(payload, "meta_predicted_forward_return_10d")

    assert signal["evaluated_date_count"] == 2
    assert signal["mean_spearman"] == 0.0
    assert [row["rebalance_date"] for row in signal["date_metrics"]] == [
        "2024-01-01",
        "2024-02-01",
    ]


def test_actual_columns_are_only_used_as_evaluation_targets():
    rows = [
        {
            **_row("2024-01-01", 0, signal=0.1, target=-0.05),
            "actual_cheating_signal": "999",
        },
        {
            **_row("2024-01-01", 1, signal=0.2, target=0.05),
            "actual_cheating_signal": "-999",
        },
    ]

    payload = build_cross_sectional_ranking_diagnostics(rows)

    assert "actual_cheating_signal" not in {
        signal["signal"] for signal in payload["signals"]
    }
    assert payload["time_series_leakage_guard"].startswith(
        "Signals are evaluated only against same-row"
    )


def test_ranking_diagnostics_use_stock_level_rows_when_available(tmp_path):
    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    (output_dir / "stock_level_prediction_artifacts.csv").write_text(
        "\n".join(
            [
                "rebalance_date,symbol,predicted_momentum_20d,actual_forward_return_10d,actual_future_drawdown,actual_future_volatility",
                "2024-01-01,AAA,0.3,0.05,-0.01,0.02",
                "2024-01-01,BBB,0.1,-0.02,-0.04,0.03",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "meta_auxiliary_predictions.csv").write_text(
        "\n".join(
            [
                "rebalance_date,variant_id,meta_predicted_forward_return_10d,actual_forward_return_10d,actual_future_drawdown,actual_future_volatility",
                "2024-01-01,variant-a,-1.0,-1.0,-0.01,0.02",
                "2024-01-01,variant-b,-2.0,-2.0,-0.01,0.02",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    paths = write_cross_sectional_ranking_diagnostics(
        {"ml": {"output_dir": str(output_dir)}}
    )

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["row_entity_type"] == "symbol"
    assert payload["stock_level_available"] is True
    assert payload["source_path"].endswith("stock_level_prediction_artifacts.csv")
    assert _signal(payload, "momentum_20d")["top_minus_bottom_spread"] > 0.0


def test_cross_sectional_ranking_diagnostics_has_no_operational_imports():
    source = inspect.getsource(cross_sectional_ranking_diagnostics)

    assert "infrastructure.broker" not in source
    assert "paper_trading" not in source
    assert "paper_commands" not in source
    assert "live_trading" not in source
    assert "core.entities.order" not in source
    assert "order_execution" not in source


def _signal(payload: dict, name: str) -> dict:
    return next(row for row in payload["signals"] if row["signal"] == name)


def _row(date: str, index: int, *, signal: float, target: float) -> dict[str, str]:
    return {
        "feature_id": f"{date}-{index}",
        "rebalance_date": date,
        "variant_id": f"variant-{index}",
        "meta_predicted_forward_return_10d": str(signal),
        "meta_predicted_forward_return_5d": str(signal / 2.0),
        "meta_predicted_future_drawdown": "-0.05",
        "meta_predicted_future_volatility": "0.02",
        "actual_forward_return_10d": str(target),
        "actual_forward_return_5d": str(target / 2.0),
        "actual_future_drawdown": "-0.05",
        "actual_future_volatility": "0.02",
    }
