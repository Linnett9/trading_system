from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path

from core.research.ml.stock_level import overnight_stock_alpha_runner
from core.research.ml.stock_level.overnight_stock_alpha_runner import (
    OvernightStockAlphaStages,
    write_overnight_stock_alpha_experiment,
)


@dataclass(frozen=True)
class ArtifactPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class FeaturePaths:
    enriched_csv_path: Path
    audit_csv_path: Path
    audit_json_path: Path
    audit_markdown_path: Path


@dataclass(frozen=True)
class BenchmarkPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path
    predictions_path: Path


@dataclass(frozen=True)
class AttributionPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path


def test_summary_compares_original_and_enriched_metrics(tmp_path):
    summary = _run(tmp_path)

    comparisons = summary["comparisons"]
    winners = summary["winners"]

    assert comparisons["original_baseline_artifact_benchmark"]["source_name"] == "ridge"
    assert comparisons["enriched_feature_benchmark"]["source_name"] == "gradient_boosting"
    assert comparisons["momentum_120d"]["spearman_ic"] == 0.02
    assert comparisons["ridge"]["top_minus_bottom_spread"] == 0.045
    assert comparisons["gradient_boosting"]["spread_sharpe"] == 1.4
    assert winners["best_by_spearman"] == "gradient_boosting"
    assert winners["best_by_spread"] == "gradient_boosting"
    assert winners["best_by_sharpe"] == "gradient_boosting"
    assert winners["best_by_risk_adjusted_spread"] == "gradient_boosting"
    assert winners["did_enriched_features_help"] is True


def test_cli_and_sector_reference_compatibility_imports():
    import application.cli  # noqa: F401
    from core.research.ml.data.sector_reference import (
        load_sector_by_symbol as refactored_loader,
    )
    from core.research.ml.sector_reference import load_sector_by_symbol

    assert load_sector_by_symbol is refactored_loader


def test_stage_timing_and_report_writing_work(tmp_path):
    summary = _run(tmp_path)

    summary_dir = tmp_path / "stock_alpha" / "benchmark"
    assert (summary_dir / "overnight_stock_alpha_summary.json").exists()
    assert (summary_dir / "overnight_stock_alpha_summary.md").exists()
    assert set(summary["stage_timings"]) == {
        "stock_artifact",
        "alpha_features",
        "baseline_benchmark",
        "enriched_benchmark",
        "target_comparison",
        "portfolio_replay",
        "portfolio_policy_sweep",
        "attribution",
        "summary_write",
    }
    assert summary["stage_timings"]["baseline_benchmark"]["seconds"] > 0.0
    assert summary["research_only"] is True
    assert summary["trading_impact"] == "none"
    assert summary["production_validated"] is False


def test_overnight_summary_records_effective_parallelism(tmp_path):
    summary = _run(tmp_path)

    assert summary["parallelism"] == {
        "stock_alpha_feature_n_jobs": 2,
        "stock_ranker_model_n_jobs": 4,
                    "sklearn_n_jobs": 2,
        "effective_model_workers": 4,
        "stock_alpha_overnight_stage_n_jobs": 1,
        "effective_stage_workers": 1,
        "stages": "sequential",
        "stage_parallelism_enabled": False,
        "oversubscription_policy": (
            "Overnight stages remain sequential by default; alpha feature generation "
            "and each benchmark use their own bounded worker settings."
        ),
    }


def test_overnight_runner_has_no_execution_or_trading_imports():
    source = inspect.getsource(overnight_stock_alpha_runner)

    forbidden = (
        "infrastructure.broker",
        "paper_trading",
        "paper_commands",
        "live_trading",
        "core.entities.order",
        "order_execution",
        "core.execution",
        "core.paper",
        "Broker",
        "Order",
    )
    assert not any(item in source for item in forbidden)


def test_overnight_runner_does_not_change_promotion_gates(tmp_path):
    summary = _run(tmp_path)

    assert summary["promotion_thresholds_changed"] is False
    assert "promotion_reduced_exposure" not in json.dumps(summary)
    assert "promotion_candidate_status" not in json.dumps(summary)


def test_overnight_benchmark_stock_artifact_writes_to_canonical_output_dir(tmp_path):
    root = tmp_path / "stock_alpha"
    seen: dict[str, dict] = {}

    def stock_artifact(config):
        seen["stock_artifact"] = dict(config["ml"])
        output = Path(config["ml"]["output_dir"])
        return ArtifactPaths(
            output / "stock_level_prediction_artifacts.csv",
            output / "stock_level_prediction_artifacts.json",
            output / "stock_level_prediction_artifacts.md",
        )

    summary = _run(tmp_path, root=root, stock_artifact=stock_artifact)
    canonical = root / "benchmark"

    assert Path(seen["stock_artifact"]["output_dir"]) == canonical
    assert summary["artifacts"]["stock_artifact"] == {
        "csv_path": str(canonical / "stock_level_prediction_artifacts.csv"),
        "json_path": str(canonical / "stock_level_prediction_artifacts.json"),
        "markdown_path": str(canonical / "stock_level_prediction_artifacts.md"),
    }


def test_alpha_feature_stage_reads_canonical_stock_artifact_path(tmp_path):
    root = tmp_path / "stock_alpha"
    seen: dict[str, dict] = {}

    def alpha_features(config):
        seen["alpha_features"] = dict(config["ml"])
        output = Path(config["ml"]["output_dir"])
        path = output / "stock_level_prediction_artifacts_enriched.csv"
        path.write_text("rebalance_date,symbol\n2024-01-01,AAA\n", encoding="utf-8")
        return FeaturePaths(
            path,
            output / "stock_level_alpha_feature_audit.csv",
            output / "stock_level_alpha_feature_audit.json",
            output / "stock_level_alpha_feature_audit.md",
        )

    _run(tmp_path, root=root, alpha_features=alpha_features)

    assert Path(seen["alpha_features"]["stock_level_base_prediction_artifacts_path"]) == (
        root / "benchmark" / "stock_level_prediction_artifacts.csv"
    )


def test_overnight_dev_benchmark_full_output_dirs_are_separated(tmp_path):
    root = tmp_path / "stock_alpha"
    summaries = {
        run_size: _run(tmp_path / run_size, root=root, run_size=run_size)
        for run_size in ("dev", "benchmark", "full")
    }

    assert {summary["output_dir"] for summary in summaries.values()} == {
        str(root / "dev"),
        str(root / "benchmark"),
        str(root / "full"),
    }


def test_resume_does_not_use_legacy_ml_output_when_disabled(tmp_path):
    root = tmp_path / "stock_alpha"
    legacy = tmp_path / "reports" / "ml" / "benchmark" / "ml"
    legacy.mkdir(parents=True)
    (legacy / "stock_level_prediction_artifacts.csv").write_text("rebalance_date,symbol\n2024-01-01,OLD\n", encoding="utf-8")

    summary = _run(tmp_path, root=root)

    assert summary["artifact_status"]["path"] == str(root / "benchmark" / "stock_level_prediction_artifacts.csv")
    assert "reports/ml/benchmark/ml" not in json.dumps(summary)


def test_stage_output_outside_canonical_root_fails_validation(tmp_path):
    root = tmp_path / "stock_alpha"
    outside = tmp_path / "outside" / "stock_level_prediction_artifacts.csv"

    def stock_artifact(_config):
        return ArtifactPaths(outside, outside.with_suffix(".json"), outside.with_suffix(".md"))

    try:
        _run(tmp_path, root=root, stock_artifact=stock_artifact)
    except ValueError as exc:
        assert "Output-root validation failed" in str(exc)
        assert "stock_artifact.csv_path" in str(exc)
    else:
        raise AssertionError("Expected output-root validation failure")


def _run(
    tmp_path: Path,
    *,
    root: Path | None = None,
    run_size: str = "benchmark",
    stock_artifact=None,
    alpha_features=None,
) -> dict:
    root = root or tmp_path / "stock_alpha"
    output_dir = root / run_size
    cache_dir = tmp_path / "cache"
    base_artifact = output_dir / "stock_level_prediction_artifacts.csv"
    base_artifact.parent.mkdir(parents=True)
    base_artifact.write_text("rebalance_date,symbol\n2024-01-01,AAA\n", encoding="utf-8")
    (cache_dir / "expanded_rebalance_dataset.csv").parent.mkdir(parents=True)
    (cache_dir / "expanded_rebalance_dataset.csv").write_text(
        "feature_date,symbol\n2024-01-01,AAA\n",
        encoding="utf-8",
    )
    (output_dir / "meta_auxiliary_predictions.csv").write_text(
        "rebalance_date,symbol\n2024-01-01,AAA\n",
        encoding="utf-8",
    )

    calls: list[str] = []

    def default_stock_artifact(config):
        calls.append("stock_artifact")
        path = Path(config["ml"]["output_dir"]) / "stock_level_prediction_artifacts.csv"
        return ArtifactPaths(path, path.with_suffix(".json"), path.with_suffix(".md"))

    def default_alpha_features(config):
        calls.append("alpha_features")
        path = Path(config["ml"]["output_dir"]) / "stock_level_prediction_artifacts_enriched.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("rebalance_date,symbol\n2024-01-01,AAA\n", encoding="utf-8")
        return FeaturePaths(
            path,
            path.with_name("stock_level_alpha_feature_audit.csv"),
            path.with_name("stock_level_alpha_feature_audit.json"),
            path.with_name("stock_level_alpha_feature_audit.md"),
        )
    stock_artifact_impl = stock_artifact or default_stock_artifact
    alpha_features_impl = alpha_features or default_alpha_features

    def stock_artifact_stage(config):
        if stock_artifact is not None:
            calls.append("stock_artifact")
        return stock_artifact_impl(config)

    def alpha_features_stage(config):
        if alpha_features is not None:
            calls.append("alpha_features")
        return alpha_features_impl(config)

    def benchmark(config):
        calls.append(f"benchmark:{config['ml']['output_dir']}")
        output = Path(config["ml"]["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "stock_level_model_ranking_benchmark.json"
        payload = _enriched_payload() if output.name == "enriched" else _baseline_payload()
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        return BenchmarkPaths(
            output / "stock_level_model_ranking_benchmark.csv",
            json_path,
            output / "stock_level_model_ranking_benchmark.md",
            output / "stock_level_model_oos_predictions.csv",
        )

    def attribution(config):
        calls.append("attribution")
        output = Path(config["ml"]["output_dir"])
        return AttributionPaths(
            output / "stock_level_feature_attribution.csv",
            output / "stock_level_feature_attribution.json",
            output / "stock_level_feature_attribution.md",
        )

    ticks = iter(float(index) for index in range(20))
    result = write_overnight_stock_alpha_experiment(
        {
            "cache": {"ml_dir": str(cache_dir)},
            "ml": {
                "stock_alpha_report_root": str(root),
                "stock_alpha_run_size": run_size,
                "stock_ranker_include_sequence_models": False,
                "stock_alpha_feature_n_jobs": 2,
                "stock_ranker_model_n_jobs": 4,
                "sklearn_n_jobs": 2,
                "stock_alpha_overnight_run_attribution": True,
            },
        },
        stages=OvernightStockAlphaStages(
            stock_artifact=stock_artifact_stage,
            alpha_features=alpha_features_stage,
            benchmark=benchmark,
            attribution=attribution,
        ),
        clock=lambda: next(ticks),
    )
    assert calls == [
        "stock_artifact",
        "alpha_features",
        f"benchmark:{output_dir / 'baseline'}",
        f"benchmark:{output_dir / 'enriched'}",
        "attribution",
    ]
    return json.loads(result.json_path.read_text(encoding="utf-8"))


def _baseline_payload() -> dict:
    return {
        "parallelism": {"effective_model_workers": 4},
        "leaderboard": [
            _row("momentum_120d", 0.02, 0.02, 0.5, 0.01, 0.52),
            _row("ridge", 0.04, 0.04, 0.9, 0.03, 0.55),
            _row("elastic_net", 0.03, 0.03, 0.7, 0.02, 0.53),
            _row("random_forest", 0.01, 0.01, 0.2, 0.01, 0.50),
            _row("gradient_boosting", 0.035, 0.035, 0.8, 0.025, 0.54),
        ]
    }


def _enriched_payload() -> dict:
    return {
        "parallelism": {"effective_model_workers": 4},
        "leaderboard": [
            _row("momentum_120d", 0.02, 0.02, 0.5, 0.01, 0.52),
            _row("ridge", 0.05, 0.045, 1.0, 0.04, 0.57),
            _row("elastic_net", 0.06, 0.05, 1.1, 0.045, 0.58),
            _row("random_forest", 0.07, 0.055, 1.2, 0.05, 0.59),
            _row("gradient_boosting", 0.08, 0.07, 1.4, 0.065, 0.61),
        ]
    }


def _row(
    name: str,
    spearman: float,
    spread: float,
    sharpe: float,
    risk_adjusted: float,
    hit_rate: float,
) -> dict:
    return {
        "name": name,
        "mean_spearman_ic": spearman,
        "top_minus_bottom_spread": spread,
        "spread_sharpe": sharpe,
        "risk_adjusted_spread": risk_adjusted,
        "top_decile_hit_rate": hit_rate,
    }
