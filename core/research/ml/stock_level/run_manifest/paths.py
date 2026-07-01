from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.research.ml.stock_level.run_manifest.types import (
    LEGACY_OUTPUT_ROOT,
    StockAlphaInterruptedSummaryPaths,
    StockAlphaRunManifestPaths,
    StockAlphaRunStatusPaths,
)


def manifest_paths(output_dir: Path) -> StockAlphaRunManifestPaths:
    return StockAlphaRunManifestPaths(
        json_path=output_dir / "stock_alpha_run_manifest.json",
        markdown_path=output_dir / "stock_alpha_run_manifest.md",
    )


def interrupted_summary_paths(output_dir: Path) -> StockAlphaInterruptedSummaryPaths:
    return StockAlphaInterruptedSummaryPaths(
        json_path=output_dir / "overnight_stock_alpha_interrupted_summary.json",
        markdown_path=output_dir / "overnight_stock_alpha_interrupted_summary.md",
    )


def run_status_paths(output_dir: Path) -> StockAlphaRunStatusPaths:
    return StockAlphaRunStatusPaths(
        json_path=output_dir / "stock_alpha_run_status.json",
        markdown_path=output_dir / "stock_alpha_run_status.md",
    )


def expected_stage_output_paths(
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, dict[str, Path]]:
    return {
        "stock_artifact": {
            "csv_path": output_dir / "stock_level_prediction_artifacts.csv",
            "json_path": output_dir / "stock_level_prediction_artifacts.json",
            "markdown_path": output_dir / "stock_level_prediction_artifacts.md",
        },
        "alpha_features": {
            "enriched_csv_path": output_dir / "stock_level_prediction_artifacts_enriched.csv",
            "audit_csv_path": output_dir / "stock_level_alpha_feature_audit.csv",
            "audit_json_path": output_dir / "stock_level_alpha_feature_audit.json",
            "audit_markdown_path": output_dir / "stock_level_alpha_feature_audit.md",
        },
        "baseline_benchmark": {
            "csv_path": output_dir / "baseline" / "stock_level_model_ranking_benchmark.csv",
            "json_path": output_dir / "baseline" / "stock_level_model_ranking_benchmark.json",
            "markdown_path": output_dir / "baseline" / "stock_level_model_ranking_benchmark.md",
            "predictions_path": output_dir / "baseline" / "stock_level_model_oos_predictions.csv",
        },
        "enriched_benchmark": {
            "csv_path": output_dir / "enriched" / "stock_level_model_ranking_benchmark.csv",
            "json_path": output_dir / "enriched" / "stock_level_model_ranking_benchmark.json",
            "markdown_path": output_dir / "enriched" / "stock_level_model_ranking_benchmark.md",
            "predictions_path": output_dir / "enriched" / "stock_level_model_oos_predictions.csv",
        },
        "target_comparison": {
            "csv_path": output_dir / "target_comparison" / "stock_level_target_comparison.csv",
            "json_path": output_dir / "target_comparison" / "stock_level_target_comparison.json",
            "markdown_path": output_dir / "target_comparison" / "stock_level_target_comparison.md",
        },
        "portfolio_replay": {
            "csv_path": output_dir / "portfolio_replay" / "stock_level_portfolio_replay_summary.csv",
            "json_path": output_dir / "portfolio_replay" / "stock_level_portfolio_replay_summary.json",
            "markdown_path": output_dir / "portfolio_replay" / "stock_level_portfolio_replay_summary.md",
            "equity_curves_path": output_dir / "portfolio_replay" / "stock_level_portfolio_replay_equity_curves.csv",
            "holdings_path": output_dir / "portfolio_replay" / "stock_level_portfolio_replay_holdings.csv",
        },
        "portfolio_policy_sweep": {
            "csv_path": output_dir / "portfolio_policy_sweep" / "stock_level_portfolio_policy_sweep.csv",
            "json_path": output_dir / "portfolio_policy_sweep" / "stock_level_portfolio_policy_sweep.json",
            "markdown_path": output_dir / "portfolio_policy_sweep" / "stock_level_portfolio_policy_sweep.md",
            "equity_curves_path": output_dir / "portfolio_policy_sweep" / "stock_level_portfolio_policy_sweep_equity_curves.csv",
            "top_holdings_path": output_dir / "portfolio_policy_sweep" / "stock_level_portfolio_policy_sweep_top_holdings.csv",
        },
        "experiment_report": {
            "json_path": output_dir / "stock_alpha_experiment_report.json",
            "markdown_path": output_dir / "stock_alpha_experiment_report.md",
        },
        "optional_attribution": {
            "csv_path": output_dir / "enriched" / "stock_level_feature_attribution.csv",
            "json_path": output_dir / "enriched" / "stock_level_feature_attribution.json",
            "markdown_path": output_dir / "enriched" / "stock_level_feature_attribution.md",
        },
        "overnight_summary": {
            "json_path": output_dir / "overnight_stock_alpha_summary.json",
            "markdown_path": output_dir / "overnight_stock_alpha_summary.md",
        },
    }


def validate_canonical_output_paths(
    output_dir: Path,
    output_paths: Mapping[str, Any],
    *,
    legacy_output_paths_allowed: bool,
) -> list[dict[str, str]]:
    root = output_dir.resolve()
    warnings: list[dict[str, str]] = []
    for key, value in _string_paths(output_paths).items():
        path = Path(value)
        resolved = path.resolve()
        if root == resolved or root in resolved.parents:
            continue
        is_legacy = LEGACY_OUTPUT_ROOT.as_posix() in path.as_posix()
        if is_legacy and legacy_output_paths_allowed:
            warnings.append({"path": str(path), "warning": "legacy output path allowed"})
            continue
        raise ValueError(
            f"Output-root validation failed for {key}: "
            f"{path} is outside canonical output dir {output_dir}"
        )
    return warnings

def _artifact_source_paths(config: Mapping[str, Any], output_dir: Path) -> list[Path]:
    ml = dict(config.get("ml", {}) or {})
    cache = dict(config.get("cache", {}) or {})
    return [
        Path(
            ml.get(
                "expanded_rebalance_dataset_path",
                Path(cache.get("ml_dir", "cache/ml")) / "expanded_rebalance_dataset.csv",
            )
        ),
        output_dir / "meta_auxiliary_predictions.csv",
    ]


def _string_paths(output_paths: Mapping[str, Any] | None) -> dict[str, str]:
    if not output_paths:
        return {}
    return {
        str(key): str(value)
        for key, value in output_paths.items()
        if isinstance(value, (str, Path))
    }


def _all_exist(output_paths: Mapping[str, str]) -> bool:
    return bool(output_paths) and all(
        Path(path).exists() and Path(path).stat().st_size > 0
        for path in output_paths.values()
    )
