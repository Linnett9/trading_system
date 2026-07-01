import csv
import json

import pytest

from application.services.research_profiles import apply_research_profile
from core.research.ml.stock_level.stock_alpha_candidate_report import build_stock_alpha_candidate_report, write_stock_alpha_candidate_report
from core.research.ml.stock_level.stock_alpha_model_sets import resolve_stock_alpha_model_set


GUARDRAILS = {"research_only": True, "trading_impact": "none", "production_validated": False, "promotion_thresholds_changed": False}


def _write_group(root, group, values):
    directory = root / group; directory.mkdir(parents=True)
    models = list(values)
    ranking = {**GUARDRAILS, "requested_models": models + ["news_analysis_transformer"], "completed_models": models, "unavailable_models": [{"model": "news_analysis_transformer", "reason": "dependency"}], "leaderboard": [{"name": model, "mean_spearman_ic": index, "top_minus_bottom_spread": index, "spread_sharpe": index} for index, model in enumerate(models, 1)]}
    (directory / "stock_level_model_ranking_benchmark.json").write_text(json.dumps(ranking))
    columns = ["rebalance_date", "symbol", *[f"stock_level_predicted_forward_return_10d_{model}" for model in models]]
    with (directory / "stock_level_model_oos_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
        for index in range(4):
            writer.writerow({"rebalance_date": f"2024-01-0{index + 1}", "symbol": "AAA", **{f"stock_level_predicted_forward_return_10d_{model}": series[index] for model, series in values.items()}})


def test_coverage_uses_only_finite_numeric_predictions_and_classifies_unavailable(tmp_path):
    values = {"valid": [1, 2, 3, 4], "all_nan": [float("nan")] * 4, "blank": [" "] * 4, "nan_string": ["nan"] * 4, "mixed": [1, 2, "bad", float("inf")]}
    for group in ("baseline", "enriched"): _write_group(tmp_path, group, values)
    report = build_stock_alpha_candidate_report(tmp_path)
    coverage = {row["model"]: row for row in report["model_validation"]["enriched"]["prediction_coverage"]}
    for model in ("all_nan", "blank", "nan_string"):
        assert coverage[model]["status"] == "invalid_empty_predictions"
        assert coverage[model]["finite_prediction_count"] == 0
        assert coverage[model]["prediction_coverage_ratio"] == 0.0
    assert coverage["mixed"]["status"] == "invalid_non_finite_predictions"
    assert coverage["mixed"]["raw_non_null_count"] == 4
    assert coverage["mixed"]["numeric_non_null_count"] == 3
    assert coverage["mixed"]["finite_prediction_count"] == 2
    unavailable = [row for row in report["candidate_table"] if row["model_or_signal"] == "news_analysis_transformer"]
    assert unavailable and all(row["category"] == "unavailable_model" for row in unavailable)
    assert report["model_validation"]["enriched"]["winners"]["mean_spearman_ic"]["name"] == "valid"


def test_missing_prediction_column(tmp_path):
    values = {"valid": [1, 2, 3, 4]}
    for group in ("baseline", "enriched"): _write_group(tmp_path, group, values)
    path = tmp_path / "enriched" / "stock_level_model_ranking_benchmark.json"
    ranking = json.loads(path.read_text()); ranking["requested_models"].append("missing"); ranking["completed_models"].append("missing"); ranking["leaderboard"].append({"name": "missing", "mean_spearman_ic": 999, "top_minus_bottom_spread": 999, "spread_sharpe": 999}); path.write_text(json.dumps(ranking))
    report = build_stock_alpha_candidate_report(tmp_path)
    coverage = {row["model"]: row for row in report["model_validation"]["enriched"]["prediction_coverage"]}
    assert coverage["missing"]["status"] == "invalid_missing_prediction_column"
    assert report["model_validation"]["enriched"]["winners"]["mean_spearman_ic"]["name"] == "valid"


def test_validated_standard_prediction_coverage_accepts_all_executable_models(tmp_path):
    model_set = resolve_stock_alpha_model_set("validated_standard")
    values = {
        model: [index + 1, index + 2, index + 3, index + 4]
        for index, model in enumerate(model_set.included_models)
    }
    for group in ("baseline", "enriched"):
        _write_group(tmp_path, group, values)
        ranking_path = tmp_path / group / "stock_level_model_ranking_benchmark.json"
        ranking = json.loads(ranking_path.read_text())
        ranking.update(model_set.metadata())
        ranking["stock_ranker_model_set"] = "validated_standard"
        ranking["requested_models"] = list(model_set.included_models)
        ranking["completed_models"] = list(model_set.included_models)
        ranking["unavailable_models"] = []
        ranking_path.write_text(json.dumps(ranking))

    report = build_stock_alpha_candidate_report(tmp_path)
    enriched = report["model_validation"]["enriched"]
    coverage = {row["model"]: row for row in enriched["prediction_coverage"]}

    assert enriched["effective_model_set"] == "validated_standard"
    assert set(enriched["valid_completed_models"]) == set(model_set.included_models)
    assert all(coverage[model]["status"] == "valid" for model in model_set.included_models)
    assert "news_analysis_transformer" not in coverage
    excluded = {row["name"]: row["exclusion_reason"] for row in enriched["excluded_models"]}
    assert excluded["news_analysis_transformer"] == "conditional_news_features_unavailable"


def test_development_profile_writes_and_inspects_dev_flat_layout(tmp_path):
    root = tmp_path / "stock_alpha"; dev = root / "dev"
    _write_group(dev, "source", {"valid": [1, 2, 3, 4]})
    source = dev / "source"
    for name in ("stock_level_model_ranking_benchmark.json", "stock_level_model_oos_predictions.csv"):
        (source / name).replace(dev / name)
    source.rmdir()
    config = apply_research_profile({"ml": {"stock_alpha_report_root": str(root), "stock_alpha_run_size": "benchmark"}}, "development")
    paths = write_stock_alpha_candidate_report(config); payload = json.loads(paths.json_path.read_text())
    assert paths.json_path.parent == dev
    assert not (root / "benchmark" / "stock_alpha_candidate_report.json").exists()
    assert (payload["resolved_profile"], payload["resolved_run_size"], payload["layout_detected"]) == ("development", "dev", "dev_flat")
    assert payload["output_dir"] == str(dev)


def test_benchmark_profile_uses_nested_benchmark_layout(tmp_path):
    root = tmp_path / "stock_alpha"; benchmark = root / "benchmark"
    for group in ("baseline", "enriched"): _write_group(benchmark, group, {"valid": [1, 2, 3, 4]})
    config = apply_research_profile({"ml": {"stock_alpha_report_root": str(root), "stock_alpha_run_size": "dev"}}, "benchmark")
    paths = write_stock_alpha_candidate_report(config); payload = json.loads(paths.json_path.read_text())
    assert paths.json_path.parent == benchmark
    assert payload["resolved_run_size"] == "benchmark"
    assert payload["layout_detected"] == "overnight_nested"
    assert payload["research_only"] is True and payload["production_validated"] is False


def test_mixed_layout_prefers_nested_outputs_and_warns(tmp_path):
    _write_group(tmp_path, "flat_source", {"flat_model": [1, 2, 3, 4]})
    flat = tmp_path / "flat_source"
    for name in ("stock_level_model_ranking_benchmark.json", "stock_level_model_oos_predictions.csv"):
        (flat / name).replace(tmp_path / name)
    flat.rmdir()
    for group in ("baseline", "enriched"):
        _write_group(tmp_path, group, {"nested_model": [1, 2, 3, 4]})

    report = build_stock_alpha_candidate_report(tmp_path)

    assert report["layout_detected"] == "mixed"
    assert set(report["model_validation"]) == {"baseline", "enriched"}
    assert str(tmp_path / "stock_level_model_ranking_benchmark.json") not in report["files_inspected"]
    assert str(tmp_path / "enriched" / "stock_level_model_ranking_benchmark.json") in report["files_inspected"]
    assert any("using nested baseline/enriched outputs" in warning for warning in report["warnings"])
