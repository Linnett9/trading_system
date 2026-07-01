from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def stock_alpha_output_dir(config: Mapping[str, Any]) -> Path:
    ml = dict(config.get("ml", {}) or {})
    run_size = str(ml.get("stock_alpha_run_size", "benchmark")).lower()
    if run_size not in {"dev", "benchmark", "full"}:
        raise ValueError("ml.stock_alpha_run_size must be dev, benchmark, or full")
    if ml.get("stock_alpha_output_dir_override"):
        return Path(ml["output_dir"])
    if "stock_alpha_report_root" not in ml and "output_dir" in ml:
        return Path(ml["output_dir"])
    root = Path(ml.get("stock_alpha_report_root", "reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha"))
    return root / run_size


def stock_alpha_report_metadata(config: Mapping[str, Any], output_dir: Path, *, source_artifact_path: Path | None = None, generated_artifact_paths: list[Path] | None = None) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    expanded = dict(ml.get("expanded_rebalance_dataset", {}) or {})
    return {
        "output_root": str(Path(ml.get("stock_alpha_report_root", "reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha"))),
        "output_dir": str(output_dir),
        "run_size": str(ml.get("stock_alpha_run_size", "benchmark")),
        "profile": str(config.get("research", {}).get("profile", ml.get("stock_alpha_run_size", "benchmark"))),
        "config_path": str(config.get("config_path", "config/config.yaml")),
        "source_artifact_path": str(source_artifact_path) if source_artifact_path else None,
        "generated_artifact_paths": [str(path) for path in (generated_artifact_paths or [])],
        "legacy_output_paths_allowed": bool(ml.get("stock_alpha_allow_legacy_output_paths", False)),
        "stock_alpha_run_profile": {
            "stock_alpha_dev_max_dates": ml.get("stock_alpha_dev_max_dates"),
            "stock_alpha_dev_max_symbols": ml.get("stock_alpha_dev_max_symbols"),
            "stock_alpha_dev_recent_dates_only": ml.get("stock_alpha_dev_recent_dates_only"),
            "stock_alpha_dev_symbol_sample_method": ml.get("stock_alpha_dev_symbol_sample_method", "sorted"),
            "stock_alpha_dev_required_symbols": list(ml.get("stock_alpha_dev_required_symbols", [])),
        },
        "stock_alpha_artifact_profile": {
            "stock_alpha_artifact_universe_paths": list(ml.get("stock_alpha_artifact_universe_paths", expanded.get("universe_paths", [])) or []),
            "stock_alpha_artifact_max_symbols": ml.get("stock_alpha_artifact_max_symbols", expanded.get("max_symbols")),
            "stock_alpha_artifact_symbol_sample_method": ml.get(
                "stock_alpha_artifact_symbol_sample_method",
                ml.get("stock_alpha_artifact_sampling_method", "liquidity_ranked"),
            ),
        },
    }
