import csv
import json
from types import SimpleNamespace

from core.research.ml.stock_level.stock_level_target_comparison import write_stock_level_target_comparison


def test_target_with_zero_eligible_dates_is_skipped(tmp_path):
    artifact = tmp_path / "artifact.csv"
    with artifact.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rebalance_date", "symbol", "actual_forward_return_10d", "actual_market_residual_return_10d"])
        writer.writeheader(); writer.writerow({"rebalance_date": "2024-01-01", "symbol": "AAA", "actual_forward_return_10d": .1, "actual_market_residual_return_10d": ""})
    paths = write_stock_level_target_comparison({"ml": {"output_dir": str(tmp_path), "stock_level_prediction_artifacts_path": str(artifact), "stock_alpha_run_size": "dev", "stock_ranker_min_train_dates": 2, "stock_ranker_embargo_dates": 1, "stock_ranker_target_columns": ["actual_market_residual_return_10d"], "stock_ranker_include_sequence_models": False}})
    payload = json.loads(paths.json_path.read_text())
    row = payload["targets"][0]
    assert payload["status"] == "completed_with_skips"
    assert row["status"] == "skipped_insufficient_data"
    assert row["eligible_date_count"] == 0
    assert row["skip_reason"]
    assert row["promotion_thresholds_changed"] is False


def test_target_comparison_module_has_no_operational_imports():
    import inspect
    from core.research.ml.stock_level import stock_level_target_comparison
    source = inspect.getsource(stock_level_target_comparison)
    assert all(token not in source for token in ("core.interfaces.broker", "core.paper", "core.entities.order", "paper_trading"))


def test_preflight_distinguishes_missing_and_all_null_columns(tmp_path):
    artifact = tmp_path / "artifact.csv"
    artifact.write_text("rebalance_date,symbol,all_null\n2024-01-01,AAA,\n", encoding="utf-8")
    paths = write_stock_level_target_comparison({"ml": {"output_dir": str(tmp_path), "stock_level_prediction_artifacts_path": str(artifact), "stock_alpha_run_size": "dev", "stock_ranker_min_train_dates": 1, "stock_ranker_embargo_dates": 0, "stock_ranker_target_columns": ["missing", "all_null"], "stock_ranker_include_sequence_models": False}})
    rows = {row["target_column"]: row for row in json.loads(paths.json_path.read_text())["targets"]}
    assert rows["missing"]["skip_reason_code"] == "column_missing"
    assert rows["missing"]["target_column_present"] is False
    assert rows["all_null"]["skip_reason_code"] == "column_present_all_null"
    assert rows["all_null"]["target_column_present"] is True
