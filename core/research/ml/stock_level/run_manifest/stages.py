from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.research.framework.config import StockLevelResearchConfig
from core.research.ml.stock_level.stock_alpha_model_sets import (
    resolve_stock_alpha_model_set,
    resolve_stock_alpha_target_model_set,
)
from core.research.ml.stock_level.overnight_stock_alpha_validation import stock_alpha_stage_stale_reason
from core.research.ml.stock_level.run_manifest.paths import (
    _all_exist,
    _string_paths,
    expected_stage_output_paths,
)
from core.research.ml.stock_level.run_manifest.types import (
    FINAL_STATUSES,
    STAGE_LABELS,
    STAGE_ORDER,
)
from core.research.ml.stock_level.stock_alpha_stage_selection import StockAlphaStageSelector


def _initial_stage_states(
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    settings = StockLevelResearchConfig.from_mapping(config)
    expected = expected_stage_output_paths(config, output_dir)
    enabled = _stage_enabled(config, settings)
    states: dict[str, dict[str, Any]] = {}
    for name in STAGE_ORDER:
        paths = _string_paths(expected[name])
        status = "pending"
        skip_reason = None
        if not enabled[name] and not _all_exist(paths):
            status = "skipped"
            skip_reason = "disabled"
        elif _all_exist(paths):
            status = "completed"
        stale_reason = stock_alpha_stage_stale_reason(
            name,
            expected[name],
            settings,
            dict(config),
        ) or _manifest_model_set_stale_reason(name, expected[name], settings)
        if stale_reason:
            status = "stale"
        states[name] = {
            "name": name,
            "label": STAGE_LABELS[name],
            "status": status,
            "started_at": None,
            "ended_at": None,
            "elapsed_seconds": None,
            "output_paths": paths,
            "skip_reason": skip_reason,
            "stale_reason": stale_reason,
        }
    return states


def _manifest_model_set_stale_reason(name: str, paths: Mapping[str, Any], settings: StockLevelResearchConfig) -> str | None:
    if name in {"baseline_benchmark", "enriched_benchmark"}:
        requested, key = settings.ranker_model_set, "stock_ranker_model_set"
    elif name == "target_comparison":
        requested, key = settings.target_comparison_model_set, "stock_target_comparison_model_set"
    else:
        return None
    path = Path(paths["json_path"])
    if not path.exists(): return None
    try: payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return "existing output model-set metadata could not be read"
    output_set = payload.get("effective_model_set") or payload.get("requested_model_set")
    if output_set is None: return f"existing output was created without model-set metadata; current config requests {key}={requested}"
    if output_set != requested:
        return f"existing output model set is {output_set}; active {key} is {requested}"
    resolver = resolve_stock_alpha_target_model_set if name == "target_comparison" else resolve_stock_alpha_model_set
    included = payload.get("included_models"); expected_models = resolver(requested, include_sequence_models=settings.include_sequence_models).included_models
    if not isinstance(included, list) or set(included) != set(expected_models): return f"included models do not match effective model set {requested}"
    return None


def _detected_stage_states(
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    return _initial_stage_states(config, output_dir)


def _stage_enabled(
    config: Mapping[str, Any],
    settings: StockLevelResearchConfig,
) -> dict[str, bool]:
    selection = StockAlphaStageSelector(config, settings).resolve()
    flags = dict(selection.requested)
    flags["optional_attribution"] = flags.pop("attribution")
    flags["overnight_summary"] = True
    return flags


def _stage_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    output_paths = dict(state.get("output_paths", {}) or {})
    exists = {
        key: Path(path).exists() and Path(path).stat().st_size > 0
        for key, path in output_paths.items()
    }
    return {
        **dict(state),
        "output_paths": output_paths,
        "output_path_exists": exists,
        "all_outputs_exist": bool(output_paths) and all(exists.values()),
        "any_output_exists": any(exists.values()),
    }


def _next_recommended_stage(stages: list[dict[str, Any]]) -> str | None:
    for stage in stages:
        if stage["status"] not in FINAL_STATUSES:
            return str(stage["name"])
    return None


def _next_recommended_action(stage_name: str | None) -> str:
    if stage_name is None:
        return "No missing stock-alpha stages detected."
    if stage_name == "experiment_report":
        return "Run ml-stock-alpha-experiment-report after completed overnight outputs are available."
    return f"Resume ml-overnight-stock-alpha from stage: {stage_name}."
