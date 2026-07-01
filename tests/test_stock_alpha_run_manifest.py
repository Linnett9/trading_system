from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from core.research.ml.stock_level import stock_alpha_run_manifest
from core.research.ml.stock_level.overnight_stock_alpha_runner import (
    OvernightStockAlphaStages,
    write_overnight_stock_alpha_experiment,
)
from core.research.ml.stock_level.run_manifest.service import (
    GUARDRAILS,
    StockAlphaRunManifestTracker,
    expected_stage_output_paths,
    inspect_stock_alpha_run_status,
    validate_canonical_output_paths,
    write_stock_alpha_run_status,
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


def test_manifest_writes_json_markdown_and_stage_transitions(tmp_path):
    config = _config(tmp_path)
    output_dir = tmp_path / "stock_alpha" / "benchmark"
    output_dir.mkdir(parents=True)
    expected = expected_stage_output_paths(config, output_dir)
    _write_outputs(expected["stock_artifact"])

    tracker = StockAlphaRunManifestTracker(config, output_dir)
    tracker.mark_running("stock_artifact")
    tracker.mark_completed(
        "stock_artifact",
        output_paths=expected["stock_artifact"],
        elapsed_seconds=1.25,
    )

    payload = json.loads((output_dir / "stock_alpha_run_manifest.json").read_text())
    stages = {stage["name"]: stage for stage in payload["stages"]}
    assert (output_dir / "stock_alpha_run_manifest.md").exists()
    assert stages["stock_artifact"]["status"] == "completed"
    assert stages["stock_artifact"]["elapsed_seconds"] == 1.25
    assert all(stages["stock_artifact"]["output_path_exists"].values())
    assert payload["guardrails"] == GUARDRAILS
    assert payload["research_only"] is True
    assert payload["trading_impact"] == "none"
    assert payload["production_validated"] is False
    assert payload["promotion_thresholds_changed"] is False


def test_run_status_detects_completed_missing_and_partial_stages(tmp_path):
    config = _config(tmp_path)
    output_dir = tmp_path / "stock_alpha" / "benchmark"
    expected = expected_stage_output_paths(config, output_dir)
    _write_outputs(expected["stock_artifact"])
    _write_text(expected["baseline_benchmark"]["json_path"], "{}\n")

    status = inspect_stock_alpha_run_status(config)

    assert "stock_artifact" in status["completed_stages"]
    assert "alpha_features" in status["missing_stages"]
    assert "baseline_benchmark" in status["partial_stages"]
    assert status["next_recommended_stage"] == "alpha_features"
    assert status["guardrails"] == GUARDRAILS


def test_run_status_command_writes_inspection_outputs(tmp_path):
    config = _config(tmp_path)

    paths = write_stock_alpha_run_status(config)
    payload = json.loads(paths.json_path.read_text())

    assert paths.json_path == tmp_path / "stock_alpha" / "benchmark" / "stock_alpha_run_status.json"
    assert paths.markdown_path.exists()
    assert payload["mode"] == "ml-stock-alpha-run-status"
    assert payload["resume_command"].endswith(
        "main.py --mode ml-overnight-stock-alpha --config config/config.yaml --profile benchmark"
    )


def test_run_status_reports_legacy_path_warnings_when_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / "reports" / "ml" / "benchmark" / "ml"
    _write_text(legacy / "stock_level_prediction_artifacts.csv", "old\n")

    status = inspect_stock_alpha_run_status(_config(tmp_path))

    assert status["legacy_output_paths_allowed"] is False
    assert status["legacy_path_warnings"]
    assert "reports/ml/benchmark/ml" in status["legacy_path_warnings"][0]["path"]


def test_canonical_output_root_validation_rejects_outside_paths(tmp_path):
    output_dir = tmp_path / "stock_alpha" / "benchmark"
    output_dir.mkdir(parents=True)

    validate_canonical_output_paths(
        output_dir,
        {"json_path": output_dir / "stock_alpha_run_manifest.json"},
        legacy_output_paths_allowed=False,
    )
    with pytest.raises(ValueError, match="Output-root validation failed"):
        validate_canonical_output_paths(
            output_dir,
            {"json_path": tmp_path / "outside" / "report.json"},
            legacy_output_paths_allowed=False,
        )


def test_interrupted_summary_written_on_simulated_keyboard_interrupt(tmp_path):
    config = _config(tmp_path)
    output_dir = tmp_path / "stock_alpha" / "benchmark"

    def stock_artifact(stage_config):
        output = Path(stage_config["ml"]["output_dir"])
        paths = ArtifactPaths(
            output / "stock_level_prediction_artifacts.csv",
            output / "stock_level_prediction_artifacts.json",
            output / "stock_level_prediction_artifacts.md",
        )
        _write_outputs({
            "csv_path": paths.csv_path,
            "json_path": paths.json_path,
            "markdown_path": paths.markdown_path,
        })
        return paths

    def alpha_features(_stage_config):
        raise KeyboardInterrupt()

    ticks = iter(float(index) for index in range(20))
    with pytest.raises(KeyboardInterrupt):
        write_overnight_stock_alpha_experiment(
            config,
            stages=OvernightStockAlphaStages(
                stock_artifact=stock_artifact,
                alpha_features=alpha_features,
                benchmark=lambda _config: None,
                attribution=lambda _config: None,
            ),
            clock=lambda: next(ticks),
        )

    summary = json.loads(
        (output_dir / "overnight_stock_alpha_interrupted_summary.json").read_text()
    )
    manifest = json.loads((output_dir / "stock_alpha_run_manifest.json").read_text())
    stage_statuses = {stage["name"]: stage["status"] for stage in manifest["stages"]}

    assert (output_dir / "overnight_stock_alpha_interrupted_summary.md").exists()
    assert summary["interrupted_stage"] == "alpha_features"
    assert "stock_artifact" in summary["completed_stages"]
    assert "alpha_features" in summary["missing_stages"]
    assert summary["resume_appears_safe"] is True
    assert "not deleted or modified" in summary["note"]
    assert summary["guardrails"] == GUARDRAILS
    assert stage_statuses["stock_artifact"] == "completed"
    assert stage_statuses["alpha_features"] == "interrupted"


def test_run_manifest_has_no_execution_or_order_imports():
    source = inspect.getsource(stock_alpha_run_manifest)

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


def _config(tmp_path: Path, *, run_size: str = "benchmark") -> dict:
    return {
        "config_path": "config/config.yaml",
        "cache": {"ml_dir": str(tmp_path / "cache")},
        "research": {"profile": "benchmark"},
        "ml": {
            "stock_alpha_report_root": str(tmp_path / "stock_alpha"),
            "stock_alpha_run_size": run_size,
        },
    }


def _write_outputs(paths: dict[str, Path]) -> None:
    for key, path in paths.items():
        if path.suffix == ".csv":
            _write_text(path, "rebalance_date,symbol\n2024-01-01,AAA\n")
        elif path.suffix == ".json":
            _write_text(path, "{}\n")
        else:
            _write_text(path, f"{key}\n")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
