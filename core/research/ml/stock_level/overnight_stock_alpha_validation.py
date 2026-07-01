from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.data import JsonRepository
from core.research.ml.stock_level.stock_alpha_model_sets import resolve_stock_alpha_model_set
from core.research.ml.stock_level.stock_alpha_model_sets import resolve_stock_alpha_target_model_set
from core.research.ml.stock_level.overnight_stock_alpha_reporting import _path_payload


def _valid_output(path: Path, required_fields: set[str]) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if not required_fields:
        return True
    try:
        if path.suffix == ".json":
            payload = JsonRepository().read(path)
            return isinstance(payload, dict) and required_fields.issubset(payload)
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                fields = set(next(csv.reader(handle), []))
            return required_fields.issubset(fields)
    except (OSError, ValueError):
        return False
    return True

def _stage_model_set_compatibility(
    stage_name: str,
    expected: dict[str, Path],
    settings: StockLevelResearchConfig,
) -> tuple[bool, str | None]:
    requested = None
    config_key = None
    if stage_name in {"baseline_benchmark", "enriched_benchmark"}:
        requested, config_key = settings.ranker_model_set, "stock_ranker_model_set"
    elif stage_name == "target_comparison":
        requested, config_key = settings.target_comparison_model_set, "stock_target_comparison_model_set"
    else:
        return True, None
    json_path = expected.get("json_path")
    if json_path is None or not json_path.exists():
        return True, None
    try:
        payload = JsonRepository().read(json_path)
    except (OSError, ValueError):
        return False, f"existing {stage_name} output model-set metadata could not be read"
    output_set = payload.get("effective_model_set") or payload.get("requested_model_set")
    if output_set is None:
        return False, f"existing output was created without model-set metadata; current config requests {config_key}={requested}"
    if output_set != requested:
        return False, (
            f"existing output model set is {output_set}; active {config_key} is {requested}; "
            f"current config requests {config_key}={requested}"
        )
    resolver = resolve_stock_alpha_target_model_set if stage_name == "target_comparison" else resolve_stock_alpha_model_set
    resolved = resolver(requested, include_sequence_models=settings.include_sequence_models)
    included = payload.get("included_models")
    if not isinstance(included, list) or set(included) != set(resolved.included_models):
        return False, f"included models do not match effective model set {requested}"
    return True, None


def stock_alpha_stage_stale_reason(
    stage_name: str,
    expected: dict[str, Path],
    settings: StockLevelResearchConfig,
    config: dict[str, Any],
) -> str | None:
    if stage_name == "stock_artifact":
        return _stock_artifact_stale_reason(expected.get("json_path"), config)
    if stage_name == "alpha_features":
        return _source_path_stale_reason(
            expected.get("audit_json_path"),
            expected_key="source_artifact_path",
            expected_path=settings.base_artifact_path,
            label="base stock artifact",
        ) or _run_profile_stale_reason(expected.get("audit_json_path"), settings)
    if stage_name in {"baseline_benchmark", "enriched_benchmark"}:
        model_valid, model_reason = _stage_model_set_compatibility(stage_name, expected, settings)
        if not model_valid:
            return model_reason
        expected_source = settings.base_artifact_path if stage_name == "baseline_benchmark" else settings.artifact_path
        return _source_path_stale_reason(
            expected.get("json_path"),
            expected_key="source_path",
            expected_path=expected_source,
            label="source artifact",
        ) or _ranker_split_stale_reason(expected.get("json_path"), settings) or _run_profile_stale_reason(expected.get("json_path"), settings)
    if stage_name == "target_comparison":
        model_valid, model_reason = _stage_model_set_compatibility(stage_name, expected, settings)
        if not model_valid:
            return model_reason
        return _source_path_stale_reason(
            expected.get("json_path"),
            expected_key="source_artifact_path",
            expected_path=settings.artifact_path,
            label="enriched artifact",
        )
    if stage_name in {"portfolio_replay", "portfolio_policy_sweep"}:
        expected_predictions = settings.output_dir / "enriched" / "stock_level_model_oos_predictions.csv"
        return _source_path_stale_reason(
            expected.get("json_path"),
            expected_key="source_artifact_path",
            expected_path=expected_predictions,
            label="enriched OOS predictions",
        )
    return None


def _stock_artifact_stale_reason(json_path: Path | None, config: dict[str, Any]) -> str | None:
    if json_path is None or not json_path.exists():
        return None
    try:
        payload = JsonRepository().read(json_path)
    except (OSError, ValueError):
        return "existing stock artifact metadata could not be read"
    active = _active_artifact_profile(config)
    if not _has_explicit_artifact_profile(config):
        return None
    existing = dict(payload.get("stock_alpha_artifact_profile") or {})
    if not existing and active:
        existing_symbols = payload.get("symbol_count", "unknown")
        existing_universe = payload.get("stock_alpha_artifact_universe_paths") or "unknown"
        active_symbols = active.get("stock_alpha_artifact_max_symbols", "unbounded")
        active_universe = active.get("stock_alpha_artifact_universe_paths") or "default universe"
        return (
            "existing stock artifact was generated without artifact-profile metadata "
            f"({existing_symbols} symbols/{existing_universe}); active config requests "
            f"{active_symbols} symbols/{active_universe}"
        )
    mismatches = _profile_mismatches(existing, active)
    if mismatches:
        return "existing stock artifact profile differs from active config: " + "; ".join(mismatches)
    return None


def _source_path_stale_reason(
    json_path: Path | None,
    *,
    expected_key: str,
    expected_path: Path,
    label: str,
) -> str | None:
    if json_path is None or not json_path.exists():
        return None
    try:
        payload = JsonRepository().read(json_path)
    except (OSError, ValueError):
        return f"existing {label} metadata could not be read"
    existing = payload.get(expected_key)
    if existing is None:
        return f"existing output was created without {expected_key}; active {label} is {expected_path}"
    if _normalize_path(existing) != _normalize_path(expected_path):
        return f"existing output {expected_key} is {existing}; active {label} is {expected_path}"
    return None


def _ranker_split_stale_reason(json_path: Path | None, settings: StockLevelResearchConfig) -> str | None:
    if json_path is None or not json_path.exists():
        return None
    try:
        payload = JsonRepository().read(json_path)
    except (OSError, ValueError):
        return "existing ranker split metadata could not be read"
    walk_forward = dict(payload.get("walk_forward") or {})
    expected = {
        "min_train_dates": settings.min_train_dates,
        "test_window_dates": settings.test_window_dates,
        "embargo_rebalance_dates": settings.embargo_dates,
    }
    mismatches = _profile_mismatches(walk_forward, expected)
    if mismatches:
        return "existing ranker split differs from active config: " + "; ".join(mismatches)
    return None


def _run_profile_stale_reason(json_path: Path | None, settings: StockLevelResearchConfig) -> str | None:
    if settings.run_size != "dev" or json_path is None or not json_path.exists():
        return None
    try:
        payload = JsonRepository().read(json_path)
    except (OSError, ValueError):
        return "existing run-profile metadata could not be read"
    requested = {
        "stock_alpha_dev_max_dates": settings.dev_max_dates,
        "stock_alpha_dev_max_symbols": settings.dev_max_symbols,
        "stock_alpha_dev_recent_dates_only": settings.dev_recent_dates_only,
        "stock_alpha_dev_symbol_sample_method": settings.dev_symbol_sample_method,
        "stock_alpha_dev_required_symbols": list(settings.dev_required_symbols),
    }
    existing_profile = dict(payload.get("stock_alpha_run_profile") or {})
    if existing_profile:
        mismatches = _profile_mismatches(existing_profile, requested)
        if mismatches:
            return "existing dev run profile differs from active config: " + "; ".join(mismatches)
        return None
    existing_dates = payload.get("effective_date_count")
    existing_symbols = payload.get("effective_symbol_count")
    if existing_dates is None or existing_symbols is None:
        return (
            "existing output was created without dev run-profile metadata; active config requests "
            f"stock_alpha_dev_max_dates={settings.dev_max_dates}, "
            f"stock_alpha_dev_max_symbols={settings.dev_max_symbols}"
        )
    if int(existing_dates) < settings.dev_max_dates or int(existing_symbols) < settings.dev_max_symbols:
        return (
            "existing output run profile is smaller than active config: "
            f"existing effective_date_count={existing_dates}, effective_symbol_count={existing_symbols}; "
            f"active stock_alpha_dev_max_dates={settings.dev_max_dates}, "
            f"stock_alpha_dev_max_symbols={settings.dev_max_symbols}"
        )
    return None


def _active_artifact_profile(config: dict[str, Any]) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    expanded = dict(ml.get("expanded_rebalance_dataset", {}) or {})
    return {
        "stock_alpha_artifact_universe_paths": list(
            ml.get("stock_alpha_artifact_universe_paths", expanded.get("universe_paths", [])) or []
        ),
        "stock_alpha_artifact_max_symbols": ml.get("stock_alpha_artifact_max_symbols", expanded.get("max_symbols")),
        "stock_alpha_artifact_symbol_sample_method": ml.get(
            "stock_alpha_artifact_symbol_sample_method",
            ml.get("stock_alpha_artifact_sampling_method", "liquidity_ranked"),
        ),
    }


def _has_explicit_artifact_profile(config: dict[str, Any]) -> bool:
    ml = dict(config.get("ml", {}) or {})
    return any(
        key in ml
        for key in (
            "stock_alpha_artifact_universe_paths",
            "stock_alpha_artifact_max_symbols",
            "stock_alpha_artifact_symbol_sample_method",
            "stock_alpha_artifact_sampling_method",
        )
    )


def _profile_mismatches(existing: dict[str, Any], active: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for key, active_value in active.items():
        existing_value = existing.get(key)
        if _normalize_value(existing_value) != _normalize_value(active_value):
            mismatches.append(f"{key} existing={existing_value!r} active={active_value!r}")
    return mismatches


def _normalize_path(value: Any) -> str:
    return Path(str(value)).as_posix()


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value

def _validate_stage_output_root(
    stage_name: str,
    paths: Any,
    output_root: Path,
    legacy_output_paths_allowed: bool,
) -> None:
    if paths is None:
        return
    root = output_root.resolve()
    for key, value in _path_payload(paths).items():
        path = Path(value)
        resolved = path.resolve()
        if root == resolved or root in resolved.parents:
            continue
        is_legacy = "reports/ml/benchmark/ml" in path.as_posix()
        if is_legacy and legacy_output_paths_allowed:
            continue
        raise ValueError(
            "Output-root validation failed for "
            f"{stage_name}.{key}: {path} is outside canonical output dir {output_root}"
        )
