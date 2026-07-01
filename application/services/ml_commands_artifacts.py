from pathlib import Path
from typing import Any

from core.research.ml.artifact_validator import validate_prediction_artifact_dirs
from core.research.ml.metrics.leaderboard import write_source_leaderboard
from core.research.ml.trading_research_leaderboard import write_trading_research_leaderboard


def _refresh_trading_research_leaderboard(config):
    output_dir = Path(
        config.get("ml", {}).get(
            "output_dir",
            Path(config.get("reports", {}).get("ml_dir", "reports/ml"))
            / "regime_transformer_meta_ensemble_v1",
        )
    )
    return write_trading_research_leaderboard(
        output_dir=output_dir,
        classification_leaderboard_path=output_dir / "leaderboard.json",
        allocation_comparison_path=output_dir / "allocation_policy_comparison.json",
        optimizer_results_path=output_dir / "allocation_optimizer_results.json",
        auxiliary_metrics_path=output_dir / "meta_auxiliary_metrics.json",
    )

def run_ml_validate_artifacts(config):
    source_dirs = _artifact_source_dirs(config)
    results = validate_prediction_artifact_dirs(source_dirs)
    print("\nML ARTIFACT VALIDATION")
    print("mode=research | trading_impact=none")
    for result in results:
        status = "legacy" if result.legacy_warnings else "ok"
        print(
            f"{status}: {result.csv_path.parent} | "
            f"rows={result.row_count} | dataset_hash={result.dataset_hash}"
        )
        for warning in result.legacy_warnings:
            print(f"  warning: {warning}")
    meta_output_dir = _meta_ensemble_output_dir(config)
    if not (meta_output_dir / "prediction_artifacts.json").exists():
        print(f"not run yet: {meta_output_dir}")

def run_ml_run_inventory(config):
    print("\nML RUN INVENTORY")
    print("mode=research | trading_impact=none")
    for source_dir in _artifact_source_dirs(config, require_exists=False):
        csv_path = source_dir / "prediction_artifacts.csv"
        metadata_path = source_dir / "prediction_artifacts.json"
        status = "missing"
        if csv_path.exists() and metadata_path.exists():
            try:
                result = validate_prediction_artifact_dirs([source_dir])[0]
            except RuntimeError as exc:
                status = f"invalid: {exc}"
            else:
                status = (
                    "legacy"
                    if result.legacy_warnings
                    else f"complete rows={result.row_count} hash={result.dataset_hash}"
                )
        print(f"{source_dir}: {status}")

def run_ml_clean_incomplete_runs(config):
    report_dir = Path(config.get("reports", {}).get("ml_dir", "reports/ml"))
    incomplete = incomplete_ml_run_dirs(report_dir)
    print("\nML INCOMPLETE RUNS")
    print("mode=research | trading_impact=none")
    if not incomplete:
        print("No incomplete run directories found.")
        return
    for path in incomplete:
        print(path)
    print("No files were deleted. Remove listed directories manually if desired.")

def incomplete_ml_run_dirs(report_dir: Path) -> list[Path]:
    return [
        path
        for path in _artifact_child_dirs(report_dir)
        if path.name != "regime_transformer_meta_ensemble_v1"
        and _is_incomplete_run_dir(path)
    ]

def _is_incomplete_run_dir(path: Path) -> bool:
    required_files = (
        "metrics.json",
        "metadata.json",
        "dataset_audit.json",
        "prediction_artifacts.csv",
        "prediction_artifacts.json",
    )
    if not any(child.is_file() for child in path.iterdir()):
        return False
    if not all((path / name).exists() for name in required_files):
        return True
    return not _is_valid_source_artifact_dir(path)

def _update_source_leaderboard(
    config: dict[str, Any],
    completed_output_dir: Path,
) -> tuple[Path, Path]:
    report_dir = _leaderboard_report_dir(config, completed_output_dir)
    leaderboard_dir = report_dir / "regime_transformer_meta_ensemble_v1"
    source_dirs = _valid_source_leaderboard_dirs(report_dir)
    if completed_output_dir not in source_dirs and _is_valid_source_artifact_dir(
        completed_output_dir
    ):
        source_dirs.append(completed_output_dir)
    source_dirs = sorted(set(source_dirs))
    markdown_path = leaderboard_dir / "leaderboard.md"
    json_path = leaderboard_dir / "leaderboard.json"
    write_source_leaderboard(json_path, markdown_path, source_dirs)
    return markdown_path, json_path

def _leaderboard_report_dir(
    config: dict[str, Any],
    completed_output_dir: Path,
) -> Path:
    return Path(
        str(
            config.get("reports", {}).get(
                "ml_dir",
                completed_output_dir.parent,
            )
        )
    )

def _valid_source_leaderboard_dirs(report_dir: Path) -> list[Path]:
    return [
        child
        for child in _artifact_child_dirs(report_dir)
        if child.name != "regime_transformer_meta_ensemble_v1"
        and _is_valid_source_artifact_dir(child)
    ]

def _is_valid_source_artifact_dir(path: Path) -> bool:
    csv_path = path / "prediction_artifacts.csv"
    metadata_path = path / "prediction_artifacts.json"
    if not csv_path.exists() or not metadata_path.exists():
        return False
    try:
        result = validate_prediction_artifact_dirs([path])[0]
    except RuntimeError:
        return False
    return not result.legacy_warnings

def _artifact_source_dirs(
    config: dict[str, Any],
    *,
    require_exists: bool = True,
) -> list[Path]:
    ml_config = config.get("ml", {})
    explicit_dirs = ml_config.get("source_prediction_dirs")
    if explicit_dirs:
        source_dirs = [Path(str(path)) for path in explicit_dirs if path]
    else:
        output_dir = Path(
            str(
                ml_config.get(
                    "output_dir",
                    config.get("reports", {}).get("ml_dir", "reports/ml"),
                )
            )
        )
        if (output_dir / "prediction_artifacts.csv").exists():
            source_dirs = [output_dir]
        else:
            source_dirs = _artifact_child_dirs(output_dir)
    if not source_dirs:
        report_dir = Path(config.get("reports", {}).get("ml_dir", "reports/ml"))
        source_dirs = _artifact_child_dirs(report_dir)
    source_dirs = [
        path
        for path in source_dirs
        if path.name != "regime_transformer_meta_ensemble_v1"
    ]
    if require_exists:
        missing = [path for path in source_dirs if not path.exists()]
        if missing:
            raise RuntimeError(
                "Prediction artifact directories do not exist: "
                + ", ".join(str(path) for path in missing)
            )
    return source_dirs

def _meta_ensemble_output_dir(config: dict[str, Any]) -> Path:
    report_dir = Path(config.get("reports", {}).get("ml_dir", "reports/ml"))
    return report_dir / "regime_transformer_meta_ensemble_v1"

def _artifact_child_dirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [
        child
        for child in sorted(path.iterdir())
        if child.is_dir()
    ]
