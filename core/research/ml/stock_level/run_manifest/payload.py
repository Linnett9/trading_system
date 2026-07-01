from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.research.framework.config import StockLevelResearchConfig
from core.research.ml.stock_level.stock_alpha_paths import (
    stock_alpha_report_metadata,
)
from core.research.ml.stock_level.run_manifest.paths import (
    _artifact_source_paths,
)
from core.research.ml.stock_level.run_manifest.stages import (
    _next_recommended_action,
    _next_recommended_stage,
    _stage_payload,
)
from core.research.ml.stock_level.run_manifest.time import _utc_now
from core.research.ml.stock_level.run_manifest.types import (
    DEFAULT_PYTHON,
    FINAL_STATUSES,
    GUARDRAILS,
    LEGACY_OUTPUT_ROOT,
    STAGE_ORDER,
)


def build_manifest_payload(
    config: Mapping[str, Any],
    output_dir: Path,
    stage_states: Mapping[str, Mapping[str, Any]],
    *,
    mode: str,
    started_at: str | None,
    interrupted_stage: str | None = None,
    failed_stage: str | None = None,
) -> dict[str, Any]:
    settings = StockLevelResearchConfig.from_mapping(config)
    ml = dict(config.get("ml", {}) or {})
    legacy_allowed = bool(ml.get("stock_alpha_allow_legacy_output_paths", False))
    stages = [_stage_payload(stage_states[name]) for name in STAGE_ORDER]
    completed = [stage["name"] for stage in stages if stage["status"] == "completed"]
    missing = [
        stage["name"]
        for stage in stages
        if stage["status"] not in FINAL_STATUSES and not stage["all_outputs_exist"]
    ]
    partial = [
        stage["name"]
        for stage in stages
        if stage["any_output_exists"] and not stage["all_outputs_exist"]
    ]
    existing_paths = {
        stage["name"]: {
            key: path
            for key, path in stage["output_paths"].items()
            if stage["output_path_exists"].get(key)
        }
        for stage in stages
    }
    missing_warnings = [
        {"stage": stage["name"], "path_key": key, "path": path}
        for stage in stages
        if stage["status"] not in {"skipped"}
        for key, path in stage["output_paths"].items()
        if not stage["output_path_exists"].get(key, False)
    ]
    stale_warnings = stale_output_warnings(config, output_dir)
    legacy_warnings = legacy_output_warnings(
        legacy_output_paths_allowed=legacy_allowed,
    )
    next_stage = _next_recommended_stage(stages)
    payload = {
        "mode": mode,
        "profile": str(config.get("research", {}).get("profile", settings.run_size)),
        "run_size": settings.run_size,
        "output_root": stock_alpha_report_metadata(config, output_dir)["output_root"],
        "output_dir": str(output_dir),
        "legacy_output_paths_allowed": legacy_allowed,
        "command_mode": mode,
        "command": build_command(config, mode=mode),
        "started_at": started_at,
        "updated_at": _utc_now(),
        "stage_order": list(STAGE_ORDER),
        "stages": stages,
        "completed_stages": completed,
        "missing_stages": missing,
        "partial_stages": partial,
        "existing_output_paths": existing_paths,
        "missing_output_warnings": missing_warnings,
        "stale_output_warnings": stale_warnings,
        "stale_stage_guidance": stale_stage_guidance(stages),
        "legacy_path_warnings": legacy_warnings,
        "interrupted_stage": interrupted_stage,
        "failed_stage": failed_stage,
        "next_recommended_stage": next_stage,
        "next_recommended_action": _next_recommended_action(next_stage),
        "resume_command": build_command(config, mode="ml-overnight-stock-alpha"),
        "status_command": build_command(config, mode="ml-stock-alpha-run-status"),
        "guardrails": dict(GUARDRAILS),
        **GUARDRAILS,
    }
    return payload


def stale_output_warnings(
    config: Mapping[str, Any],
    output_dir: Path,
) -> list[dict[str, str]]:
    artifact = output_dir / "stock_level_prediction_artifacts.csv"
    if not artifact.exists():
        return []
    artifact_mtime = artifact.stat().st_mtime
    warnings: list[dict[str, str]] = []
    for source in _artifact_source_paths(config, output_dir):
        if source.exists() and source.stat().st_mtime > artifact_mtime:
            warnings.append(
                {
                    "stage": "stock_artifact",
                    "path": str(artifact),
                    "source_path": str(source),
                    "warning": "stock artifact is older than a source artifact",
                }
            )
    return warnings


def stale_stage_guidance(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    guidance = []
    for stage in stages:
        stale_reason = str(stage.get("stale_reason") or "")
        if not stale_reason:
            continue
        actions = ["Rerun this stage with the current config."]
        if stage.get("name") == "target_comparison" and "model set" in stale_reason:
            actions = [
                "Rerun target_comparison with the current config.",
                "Change ml.stock_target_comparison_model_set back to the existing output model set to reuse this output.",
                "Disable target_comparison through ml.stock_alpha_stages if target comparison is not needed.",
            ]
        guidance.append(
            {
                "stage": str(stage.get("name")),
                "reason": stale_reason,
                "recommended_actions": actions,
            }
        )
    return guidance


def legacy_output_warnings(*, legacy_output_paths_allowed: bool) -> list[dict[str, str]]:
    if legacy_output_paths_allowed or not LEGACY_OUTPUT_ROOT.exists():
        return []
    files = [
        path
        for path in sorted(LEGACY_OUTPUT_ROOT.rglob("*"))
        if path.is_file()
    ][:20]
    return [
        {
            "path": str(path),
            "warning": "legacy stock-alpha output path detected but legacy paths are disabled",
        }
        for path in files
    ]


def build_command(config: Mapping[str, Any], *, mode: str) -> str:
    config_path = str(config.get("config_path", "config/config.yaml"))
    profile = str(config.get("research", {}).get("profile", ""))
    pieces = [
        "PYTHONDONTWRITEBYTECODE=1",
        DEFAULT_PYTHON,
        "main.py",
        "--mode",
        mode,
        "--config",
        config_path,
    ]
    if profile in {"development", "benchmark"}:
        pieces.extend(["--profile", profile])
    return " ".join(pieces)
