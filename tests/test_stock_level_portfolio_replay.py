import inspect
import json

from core.research.ml.stock_level import stock_level_portfolio_replay
from core.research.ml.stock_level.stock_level_portfolio_replay import (
    build_stock_level_portfolio_replay,
    write_stock_level_portfolio_replay,
)


def _rows():
    rows = []
    for date_index, rebalance_date in enumerate(("2024-01-01", "2024-01-11", "2024-01-21")):
        for index, symbol in enumerate(("AAA", "BBB", "CCC", "DDD")):
            score = 4 - index + date_index * 0.01
            rows.append({"rebalance_date": rebalance_date, "symbol": symbol, "fold_id": date_index + 1, "actual_forward_return_10d": (3 - index) / 100, "ml_signal": score, "predicted_momentum_120d": score})
    return rows


def _build(**overrides):
    arguments = {"benchmark": {"walk_forward": {"out_of_sample_only": True}}, "signal_columns": ("ml_signal", "predicted_momentum_120d"), "top_n": 2, "max_position_weight": 0.5}
    arguments.update(overrides)
    return build_stock_level_portfolio_replay(_rows(), **arguments)


def test_replay_is_oos_only_and_selection_is_deterministic():
    rows = _rows() + [{"rebalance_date": "2024-02-01", "symbol": "ZZZ", "fold_id": "", "actual_forward_return_10d": 9, "ml_signal": 99}]
    summary, _, holdings, payload = build_stock_level_portfolio_replay(rows, benchmark={"walk_forward": {"out_of_sample_only": True}}, signal_columns=("ml_signal",), top_n=2, max_position_weight=0.5)
    selected = [row["symbol"] for row in holdings if row["policy"] == "long_only_top_n_equal_weight" and row["rebalance_date"] == "2024-01-01"]
    assert selected == ["AAA", "BBB"]
    assert "ZZZ" not in {row["symbol"] for row in holdings}
    assert payload["training_performed"] is False
    assert summary


def test_equal_weights_turnover_costs_and_caps():
    summary, curves, holdings, _ = _build()
    selected = [row for row in holdings if row["strategy_id"] == "ml_signal|long_only_top_n_equal_weight"]
    for rebalance_date in {row["rebalance_date"] for row in selected}:
        assert sum(row["weight"] for row in selected if row["rebalance_date"] == rebalance_date) == 1.0
    row = next(row for row in summary if row["strategy_id"] == "ml_signal|long_only_top_n_equal_weight")
    assert row["max_position_weight"] == 0.5
    assert row["transaction_cost_drag"] > 0
    assert row["net_return"] < row["gross_return"]
    first = next(row for row in curves if row["strategy_id"] == row["strategy_id"] and row["signal_column"] == "ml_signal" and row["policy"] == "long_only_top_n_equal_weight")
    assert first["turnover"] == 1.0


def test_long_short_has_expected_exposure():
    _, _, holdings, _ = _build(allow_short=True)
    rows = [row for row in holdings if row["strategy_id"] == "ml_signal|long_short_top_bottom_decile_equal_weight" and row["rebalance_date"] == "2024-01-01"]
    assert sum(row["weight"] for row in rows) == 0.0
    assert sum(abs(row["weight"]) for row in rows) == 1.0


def test_writer_creates_all_artifacts(tmp_path):
    predictions = tmp_path / "predictions.csv"
    fields = list(_rows()[0])
    predictions.write_text(",".join(fields) + "\n" + "\n".join(",".join(str(row[field]) for field in fields) for row in _rows()), encoding="utf-8")
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps({"walk_forward": {"out_of_sample_only": True}}), encoding="utf-8")
    paths = write_stock_level_portfolio_replay({"ml": {"output_dir": str(tmp_path), "stock_level_model_oos_predictions_path": str(predictions), "stock_level_model_ranking_benchmark_path": str(benchmark), "stock_portfolio_replay_signal_columns": ["ml_signal", "predicted_momentum_120d"], "stock_portfolio_replay_top_n": 2, "stock_portfolio_replay_max_position_weight": 0.5}})
    assert all(path.exists() for path in (paths.csv_path, paths.json_path, paths.markdown_path, paths.equity_curves_path, paths.holdings_path))
    payload = json.loads(paths.json_path.read_text())
    assert payload["promotion_thresholds_changed"] is False


def test_replay_has_no_operational_imports():
    source = inspect.getsource(stock_level_portfolio_replay)
    assert all(token not in source for token in ("core.interfaces.broker", "core.paper", "core.entities.order", "paper_trading"))
