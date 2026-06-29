import json
import os
from pathlib import Path

from core.research.ml.stock_level.stock_alpha_experiment_report import validate_stock_alpha_outputs, write_stock_alpha_experiment_report

GUARDS = {"research_only": True, "trading_impact": "none", "production_validated": False, "promotion_thresholds_changed": False, "run_size": "dev"}


def _write(path: Path, **values):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps({**GUARDS, **values}), encoding="utf-8"); return path


def test_validator_catches_missing_inconsistent_and_promotion(tmp_path):
    paths = {"benchmark": _write(tmp_path / "benchmark.json", oos_date_count=3), "portfolio_replay": tmp_path / "missing.json"}
    _, checks = validate_stock_alpha_outputs(paths, expected_root=tmp_path, expected_run_size="benchmark", require_all=True, max_age_hours=24)
    assert {row["check"] for row in checks["errors"]} >= {"file_exists", "run_size"}
    _write(paths["benchmark"], promotion_thresholds_changed=True, oos_date_count=3)
    _, checks = validate_stock_alpha_outputs({"benchmark": paths["benchmark"]}, expected_root=tmp_path, expected_run_size="dev", require_all=False, max_age_hours=24)
    assert "guardrail" in {row["check"] for row in checks["errors"]}


def test_validator_catches_stale_mix_and_unexpected_root(tmp_path):
    old = _write(tmp_path / "old.json", oos_date_count=2); new = _write(tmp_path / "new.json", oos_date_count=2)
    os.utime(old, (1, 1))
    _, checks = validate_stock_alpha_outputs({"benchmark": old, "portfolio_replay": new}, expected_root=tmp_path / "expected", expected_run_size="dev", require_all=True, max_age_hours=1)
    names = {row["check"] for row in checks["errors"]}
    assert {"stale_mixed_outputs", "output_root"}.issubset(names)


def test_validator_rejects_zero_date_or_infeasible_winner(tmp_path):
    path = _write(tmp_path / "sweep.json", effective_date_count=2, winners={"best": {"status": "infeasible", "date_count": 0}})
    _, checks = validate_stock_alpha_outputs({"portfolio_policy_sweep": path}, expected_root=tmp_path, expected_run_size="dev", require_all=True, max_age_hours=24)
    assert {row["check"] for row in checks["errors"]} >= {"winner_eligibility", "winner_date_count"}


def test_registry_writes_from_minimal_outputs(tmp_path):
    _write(tmp_path / "stock_level_model_ranking_benchmark.json", oos_date_count=2, effective_row_count=4, effective_date_count=2, effective_symbol_count=2, best_ml_model={"name": "ridge", "mean_spearman_ic": .1, "top_minus_bottom_spread": .02})
    result = write_stock_alpha_experiment_report({"ml": {"output_dir": str(tmp_path), "stock_alpha_run_size": "dev", "stock_alpha_experiment_registry_path": str(tmp_path / "registry.csv")}})
    assert result.json_path.exists() and result.markdown_path.exists() and result.registry_path.exists()
    assert "run_id" in result.registry_path.read_text()


def test_dev_smoke_level_ignores_missing_overnight_but_overnight_level_reports_it(tmp_path):
    common = {"output_dir": str(tmp_path), "stock_alpha_run_size": "dev", "stock_alpha_experiment_registry_path": str(tmp_path / "registry.csv")}
    dev = write_stock_alpha_experiment_report({"ml": {**common, "stock_alpha_experiment_report_level": "dev_smoke"}})
    dev_payload = json.loads(dev.json_path.read_text())
    assert all(row.get("artifact") != "overnight_summary" for row in dev_payload["validation"]["warnings"])
    overnight = write_stock_alpha_experiment_report({"ml": {**common, "stock_alpha_experiment_report_level": "overnight"}})
    overnight_payload = json.loads(overnight.json_path.read_text())
    assert any(row.get("artifact") == "overnight_summary" for row in overnight_payload["validation"]["errors"])


def test_legacy_output_is_flagged_when_disabled(tmp_path):
    legacy = tmp_path / "reports/ml/benchmark/ml/benchmark.json"
    _write(legacy, oos_date_count=2)
    _, checks = validate_stock_alpha_outputs({"benchmark": legacy}, expected_root=tmp_path / "canonical/dev", expected_run_size="dev", require_all=True, max_age_hours=24, legacy_output_paths_allowed=False)
    assert checks["output_root_validation_passed"] is False
    assert checks["legacy_output_paths_detected"] == [str(legacy)]
