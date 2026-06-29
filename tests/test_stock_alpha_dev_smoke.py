from pathlib import Path
import json
from types import SimpleNamespace

from core.research.ml.stock_level.stock_alpha_dev_smoke import StockAlphaDevSmokeStages, write_stock_alpha_dev_smoke
from core.research.ml.stock_level.stock_level_prediction_artifacts import TARGET_TYPES


def _artifact_header() -> str:
    return "rebalance_date,symbol," + ",".join(TARGET_TYPES) + "\n"


def test_dev_smoke_forces_small_dev_caps_and_disables_attribution(tmp_path):
    seen = []
    base = tmp_path / "stock_level_prediction_artifacts.csv"; base.write_text(_artifact_header())
    def stage(name, result):
        def action(config): seen.append((name, dict(config["ml"]))); return result
        return action
    feature = SimpleNamespace(enriched_csv_path=tmp_path / "enriched.csv")
    benchmark = SimpleNamespace(json_path=tmp_path / "benchmark.json", predictions_path=tmp_path / "predictions.csv")
    report = SimpleNamespace(json_path=tmp_path / "report.json")
    path = write_stock_alpha_dev_smoke({"ml": {"output_dir": str(tmp_path), "stock_alpha_run_size": "dev", "stock_alpha_dev_max_dates": 100, "stock_alpha_dev_max_symbols": 100, "stock_portfolio_policy_sweep_max_configs_dev": 100}}, stages=StockAlphaDevSmokeStages(features=stage("features", feature), benchmark=stage("benchmark", benchmark), targets=stage("targets", object()), portfolio=stage("portfolio", object()), sweep=stage("sweep", object()), report=stage("report", report)))
    assert path.exists()
    assert all(config["stock_alpha_run_size"] == "dev" for _, config in seen)
    assert all(config["stock_alpha_overnight_run_attribution"] is False for _, config in seen)
    assert max(config["stock_alpha_dev_max_dates"] for _, config in seen) <= 24
    assert max(config["stock_portfolio_policy_sweep_max_configs_dev"] for _, config in seen) <= 8
    assert path.with_suffix(".md").exists()
    markdown = path.with_suffix(".md").read_text()
    assert "## Target availability" in markdown
    assert "## Policy sweep baseline coverage" in markdown


def test_dev_smoke_completes_when_target_is_skipped(tmp_path):
    base = tmp_path / "stock_level_prediction_artifacts.csv"; base.write_text(_artifact_header())
    feature = SimpleNamespace(enriched_csv_path=tmp_path / "enriched.csv")
    benchmark = SimpleNamespace(json_path=tmp_path / "benchmark.json", predictions_path=tmp_path / "predictions.csv")
    target_json = tmp_path / "targets.json"
    target_json.write_text(json.dumps({"status": "completed_with_skips", "skipped_targets": [{"target_column": "missing", "status": "skipped_insufficient_data", "skip_reason": "no rows"}]}))
    target = SimpleNamespace(json_path=target_json)
    report_path = tmp_path / "report.json"; report_path.write_text(json.dumps({"validation": {"errors": [{"check": "synthetic"}], "warnings": [{"check": "coverage"}]}}))
    report = SimpleNamespace(json_path=report_path)
    sweep_path = tmp_path / "sweep.json"; sweep_path.write_text(json.dumps({"baseline_coverage": {"baseline_signal_available": True, "baseline_signal_columns_found": ["predicted_momentum_120d"], "baseline_signal_columns_missing": []}, "winners": {"best_baseline_policy": {"signal_column": "predicted_momentum_120d"}, "best_ml_vs_momentum_120d": {"net_return_delta": .1}}}))
    sweep = SimpleNamespace(json_path=sweep_path)
    stages = StockAlphaDevSmokeStages(features=lambda _: feature, benchmark=lambda _: benchmark, targets=lambda _: target, portfolio=lambda _: object(), sweep=lambda _: sweep, report=lambda _: report)
    path = write_stock_alpha_dev_smoke({"ml": {"output_dir": str(tmp_path), "stock_alpha_run_size": "dev"}}, stages=stages)
    payload = json.loads(path.read_text())
    assert payload["status"] == "completed_with_skips"
    assert payload["timings"]["target comparison"]["status"] == "completed_with_skips"
    assert payload["experiment_report_path"] == str(report.json_path)
    assert payload["experiment_validation"]["errors"]
    assert payload["experiment_validation"]["warnings"]
    assert payload["policy_sweep_baseline_coverage"]["baseline_signal_available"] is True


def test_dev_smoke_uses_canonical_dev_root(tmp_path):
    root = tmp_path / "stock_alpha"
    output = root / "dev"; output.mkdir(parents=True)
    (output / "stock_level_prediction_artifacts.csv").write_text(_artifact_header())
    feature = SimpleNamespace(enriched_csv_path=output / "enriched.csv")
    benchmark = SimpleNamespace(json_path=output / "benchmark.json", predictions_path=output / "predictions.csv")
    report = SimpleNamespace(json_path=output / "report.json")
    stages = StockAlphaDevSmokeStages(features=lambda _: feature, benchmark=lambda _: benchmark, targets=lambda _: object(), portfolio=lambda _: object(), sweep=lambda _: object(), report=lambda _: report)
    path = write_stock_alpha_dev_smoke({"ml": {"stock_alpha_report_root": str(root), "stock_alpha_run_size": "benchmark"}}, stages=stages)
    assert path.parent == root / "dev"
