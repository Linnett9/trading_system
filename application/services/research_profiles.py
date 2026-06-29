from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROFILE_DIR = Path("configs/research/profiles")


def load_research_profile(profile_name: str) -> dict[str, Any]:
    path = PROFILE_DIR / f"{profile_name}.yaml"
    if not path.exists():
        raise RuntimeError(f"Research profile does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Research profile must be a mapping: {path}")
    required = {
        "research_years",
        "universe",
        "max_symbols",
        "output_suffix",
        "cache_dir",
        "report_dir",
        "batch_workers",
        "model_threads",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(
            f"Research profile {profile_name} is missing required keys: {missing}"
        )
    payload["name"] = profile_name
    return payload


def apply_research_profile(config: dict[str, Any], profile_name: str | None) -> dict[str, Any]:
    if not profile_name:
        return config
    profile = load_research_profile(profile_name)
    updated = deepcopy(config)
    cache_dir = Path(str(profile["cache_dir"]))
    report_dir = Path(str(profile["report_dir"]))
    universe_path = Path("data/reference/universes") / f"{profile['universe']}.yaml"

    updated["research_profile"] = profile
    updated.setdefault("backtest", {})["years"] = int(profile["research_years"])
    updated.setdefault("cache", {})["ml_dir"] = str(cache_dir)
    updated.setdefault("reports", {})["ml_dir"] = str(report_dir)

    ml_config = updated.setdefault("ml", {})
    ml_config["research_years"] = int(profile["research_years"])
    ml_config["profile"] = profile_name
    ml_config["profile_output_suffix"] = str(profile["output_suffix"])
    ml_config["num_workers"] = int(profile["batch_workers"])
    ml_config["model_threads"] = int(profile["model_threads"])
    ml_config["torch_num_threads"] = int(
        profile.get("torch_num_threads", profile["model_threads"])
    )
    ml_config["sklearn_n_jobs"] = int(
        profile.get("sklearn_n_jobs", profile["model_threads"])
    )
    ml_config["feature_workers"] = int(profile.get("feature_workers", 1))
    for key in (
        "stock_alpha_feature_n_jobs",
        "stock_ranker_model_n_jobs",
        "stock_target_comparison_n_jobs",
        "stock_portfolio_policy_sweep_n_jobs",
        "numpy_num_threads",
    ):
        if key in profile:
            ml_config[key] = int(profile[key])
    ml_config["expanded_rebalance_dataset_path"] = str(
        cache_dir / "expanded_rebalance_dataset.csv"
    )
    ml_config["expanded_rebalance_audit_path"] = str(
        report_dir / "expanded_rebalance_dataset_audit.json"
    )
    ml_config["inventory_output_dir"] = str(report_dir)

    if "output_dir" in ml_config:
        ml_config["output_dir"] = str(_profiled_path(ml_config["output_dir"], report_dir))
    if "meta_dataset_path" in ml_config:
        ml_config["meta_dataset_path"] = str(cache_dir / Path(str(ml_config["meta_dataset_path"])).name)
    if "source_prediction_dirs" in ml_config:
        ml_config["source_prediction_dirs"] = [
            str(_profiled_path(path, report_dir))
            for path in ml_config.get("source_prediction_dirs", [])
        ]

    expanded_config = ml_config.setdefault("expanded_rebalance_dataset", {})
    expanded_config["universe_paths"] = [str(universe_path)]
    expanded_config["max_symbols"] = int(profile["max_symbols"])

    dual_momentum = updated.setdefault("research", {}).setdefault("dual_momentum", {})
    dual_momentum["universe_path"] = str(universe_path)
    dual_momentum["max_symbols"] = int(profile["max_symbols"])

    batch_config = updated.get("ml_research_batch")
    if isinstance(batch_config, dict):
        batch_config["max_workers"] = int(profile["batch_workers"])
        batch_config["model_threads"] = int(profile["model_threads"])
        batch_config["profile"] = profile_name

    return updated


def _profiled_path(path: str | Path, report_dir: Path) -> Path:
    raw_path = Path(str(path))
    return report_dir / raw_path.name
