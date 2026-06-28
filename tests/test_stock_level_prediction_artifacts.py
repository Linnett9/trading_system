from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import yaml

from core.research.ml.stock_level import stock_level_prediction_artifacts
from core.research.ml.stock_level.stock_level_prediction_artifacts import (
    build_stock_level_prediction_artifacts,
    write_stock_level_prediction_artifacts,
)


def test_stock_level_artifacts_create_one_row_per_symbol_date():
    rows, audit = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-01"), _expanded("2024-01-08")],
        artifact_rows=[],
        universe_symbols=["AAA", "BBB"],
        closes_by_symbol={
            "AAA": _closes(100.0),
            "BBB": _closes(50.0),
        },
    )

    keys = {(row["rebalance_date"], row["symbol"]) for row in rows}

    assert len(rows) == 4
    assert keys == {
        ("2024-01-01", "AAA"),
        ("2024-01-01", "BBB"),
        ("2024-01-08", "AAA"),
        ("2024-01-08", "BBB"),
    }
    assert audit["true_stock_level_rows"] is True
    assert audit["row_count"] == 4
    assert audit["symbol_count"] == 2
    assert audit["rebalance_date_count"] == 2


def test_missing_predictions_are_reported_explicitly():
    rows, audit = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-01")],
        artifact_rows=[],
        universe_symbols=["AAA", "BBB"],
        closes_by_symbol={
            "AAA": _closes(100.0),
            "BBB": _closes(50.0),
        },
    )

    assert all(row["predicted_probability"] == "" for row in rows)
    assert audit["missing_prediction_counts"]["predicted_probability"] == 2
    assert audit["missing_prediction_counts"]["predicted_momentum_20d"] == 2
    assert audit["artifact_rows_with_symbol_predictions"] == 0
    assert audit["suitable_for_true_stock_level_ranking_diagnostics"] is False
    assert "do not contain symbol-level predictions" in audit["suitability_reason"]


def test_baseline_features_use_only_prices_before_rebalance_date():
    before = _history_around_rebalance(after_jump=100.0)
    after_changed = _history_around_rebalance(after_jump=10_000.0)

    first_rows, first_audit = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-04-01")],
        artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol={"AAA": before},
    )
    second_rows, _ = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-04-01")],
        artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol={"AAA": after_changed},
    )

    assert first_rows[0]["predicted_momentum_20d"] == second_rows[0][
        "predicted_momentum_20d"
    ]
    assert first_rows[0]["predicted_momentum_60d"] == second_rows[0][
        "predicted_momentum_60d"
    ]
    assert first_rows[0]["predicted_volatility_20d"] == second_rows[0][
        "predicted_volatility_20d"
    ]
    assert first_audit["populated_prediction_counts"]["predicted_momentum_20d"] == 1
    assert first_audit["usable_for_stock_level_ranking"] is True


def test_missing_early_history_baseline_features_are_reported():
    _, audit = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-05")],
        artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol={"AAA": _history_around_rebalance(after_jump=100.0)},
    )

    assert audit["missing_prediction_counts"]["predicted_momentum_120d"] == 1


def test_existing_artifact_level_files_are_preserved(tmp_path):
    output_dir = tmp_path / "reports" / "meta"
    cache_dir = tmp_path / "cache"
    universe_path = tmp_path / "universe.yaml"
    expanded_path = cache_dir / "expanded_rebalance_dataset.csv"
    old_artifact = output_dir / "prediction_artifacts.csv"
    output_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    old_artifact.write_text("sentinel\n", encoding="utf-8")
    universe_path.write_text(
        yaml.safe_dump({"symbols": ["AAA", "BBB"]}),
        encoding="utf-8",
    )
    _write_csv(expanded_path, [_expanded("2024-01-01")])
    (output_dir / "meta_auxiliary_predictions.csv").write_text(
        "rebalance_date,feature_id,symbol\n2024-01-01,feature-a,\n",
        encoding="utf-8",
    )

    paths = write_stock_level_prediction_artifacts(
        {
            "cache": {"ml_dir": str(cache_dir)},
            "ml": {
                "output_dir": str(output_dir),
                "expanded_rebalance_dataset_path": str(expanded_path),
                "expanded_rebalance_dataset": {
                    "universe_paths": [str(universe_path)],
                },
                "stooq_parquet_dir": str(tmp_path / "missing_parquet"),
            },
        }
    )

    assert paths.csv_path.exists()
    assert paths.json_path.exists()
    assert paths.markdown_path.exists()
    assert old_artifact.read_text(encoding="utf-8") == "sentinel\n"
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["existing_artifact_level_files_preserved"] is True


def test_stock_level_prediction_artifacts_has_no_operational_imports():
    source = inspect.getsource(stock_level_prediction_artifacts)

    assert "infrastructure.broker" not in source
    assert "paper_trading" not in source
    assert "paper_commands" not in source
    assert "live_trading" not in source
    assert "core.entities.order" not in source
    assert "order_execution" not in source


def _expanded(date: str) -> dict[str, str]:
    return {
        "rebalance_date": date,
        "feature_date": date,
        "breadth_above_sma_200": "0.5",
        "spy_realized_volatility_21d": "0.1",
        "spy_realized_volatility_63d": "0.2",
        "spy_max_drawdown_63d": "-0.03",
        "spy_max_drawdown_126d": "-0.05",
    }


def _closes(start: float) -> dict[str, dict[str, float]]:
    close = {}
    dollar_volume = {}
    for index in range(20):
        day = index + 1
        date = f"2024-01-{day:02d}"
        close[date] = start + index
        dollar_volume[date] = (start + index) * 1_000_000
    return {"close": close, "dollar_volume": dollar_volume}


def _history_around_rebalance(after_jump: float) -> dict[str, dict[str, float]]:
    close = {}
    dollar_volume = {}
    for index in range(90):
        date = f"2024-01-{index + 1:02d}" if index < 31 else (
            f"2024-02-{index - 30:02d}" if index < 60 else f"2024-03-{index - 59:02d}"
        )
        value = 100.0 + index
        close[date] = value
        dollar_volume[date] = value * 1_000_000
    close["2024-04-02"] = after_jump
    dollar_volume["2024-04-02"] = after_jump * 1_000_000
    return {"close": close, "dollar_volume": dollar_volume}


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
