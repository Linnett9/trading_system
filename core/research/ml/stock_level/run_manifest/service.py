from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.run_manifest.markdown import (
    _fmt,
    _interrupted_markdown,
    _markdown,
)
from core.research.ml.stock_level.run_manifest.paths import (
    _all_exist,
    _artifact_source_paths,
    _string_paths,
    expected_stage_output_paths,
    interrupted_summary_paths,
    manifest_paths,
    run_status_paths,
    validate_canonical_output_paths,
)
from core.research.ml.stock_level.run_manifest.payload import (
    build_command,
    build_manifest_payload,
    legacy_output_warnings,
    stale_output_warnings,
)
from core.research.ml.stock_level.run_manifest.stages import (
    _detected_stage_states,
    _initial_stage_states,
    _manifest_model_set_stale_reason,
    _next_recommended_action,
    _next_recommended_stage,
    _stage_enabled,
    _stage_payload,
)
from core.research.ml.stock_level.run_manifest.time import _utc_now
from core.research.ml.stock_level.run_manifest.types import (
    DEFAULT_PYTHON,
    FINAL_STATUSES,
    GUARDRAILS,
    LEGACY_OUTPUT_ROOT,
    STAGE_LABELS,
    STAGE_ORDER,
    StockAlphaInterruptedSummaryPaths,
    StockAlphaRunManifestPaths,
    StockAlphaRunStatusPaths,
)


class StockAlphaRunManifestTracker:
    def __init__(
        self,
        config: Mapping[str, Any],
        output_dir: Path,
        *,
        command_mode: str = "ml-overnight-stock-alpha",
    ) -> None:
        self.config = dict(config)
        self.output_dir = output_dir
        self.command_mode = command_mode
        self.settings = StockLevelResearchConfig.from_mapping(self.config)
        self.started_at = _utc_now()
        self.interrupted_stage: str | None = None
        self.failed_stage: str | None = None
        self.stage_states = _initial_stage_states(self.config, self.output_dir)
        self.write()

    def mark_running(self, stage_name: str) -> None:
        state = self._stage(stage_name)
        state.update(
            {
                "status": "running",
                "started_at": _utc_now(),
                "ended_at": None,
                "elapsed_seconds": None,
            }
        )
        self.write()

    def mark_completed(
        self,
        stage_name: str,
        *,
        output_paths: Mapping[str, Any] | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        state = self._stage(stage_name)
        state.update(
            {
                "status": "completed",
                "ended_at": _utc_now(),
                "elapsed_seconds": elapsed_seconds,
                "output_paths": _string_paths(output_paths) or state["output_paths"],
            }
        )
        self.write()

    def mark_skipped(
        self,
        stage_name: str,
        *,
        output_paths: Mapping[str, Any] | None = None,
        elapsed_seconds: float | None = 0.0,
        skip_reason: str | None = None,
    ) -> None:
        state = self._stage(stage_name)
        state.update(
            {
                "status": "skipped",
                "ended_at": _utc_now(),
                "elapsed_seconds": elapsed_seconds,
                "output_paths": _string_paths(output_paths) or state["output_paths"],
                "skip_reason": skip_reason,
            }
        )
        self.write()

    def mark_stale(self, stage_name: str, *, stale_reason: str) -> None:
        state = self._stage(stage_name)
        state.update({"status": "stale", "stale_reason": stale_reason, "skip_reason": None})
        self.write()

    def mark_failed(
        self,
        stage_name: str,
        *,
        error: BaseException | str,
        elapsed_seconds: float | None = None,
    ) -> None:
        self.failed_stage = stage_name
        state = self._stage(stage_name)
        state.update(
            {
                "status": "failed",
                "ended_at": _utc_now(),
                "elapsed_seconds": elapsed_seconds,
                "error": str(error),
            }
        )
        self.write()

    def mark_interrupted(
        self,
        stage_name: str,
        *,
        elapsed_seconds: float | None = None,
    ) -> None:
        self.interrupted_stage = stage_name
        state = self._stage(stage_name)
        state.update(
            {
                "status": "interrupted",
                "ended_at": _utc_now(),
                "elapsed_seconds": elapsed_seconds,
            }
        )
        self.write()
        self.write_interrupted_summary()

    def write(self) -> StockAlphaRunManifestPaths:
        paths = manifest_paths(self.output_dir)
        payload = build_manifest_payload(
            self.config,
            self.output_dir,
            self.stage_states,
            mode=self.command_mode,
            started_at=self.started_at,
            interrupted_stage=self.interrupted_stage,
            failed_stage=self.failed_stage,
        )
        writer = ResearchArtifactWriter()
        writer.write_json(paths.json_path, payload)
        writer.write_markdown(paths.markdown_path, _markdown(payload, title="Stock Alpha Run Manifest"))
        return paths

    def write_interrupted_summary(self) -> StockAlphaInterruptedSummaryPaths:
        paths = interrupted_summary_paths(self.output_dir)
        manifest = build_manifest_payload(
            self.config,
            self.output_dir,
            self.stage_states,
            mode=self.command_mode,
            started_at=self.started_at,
            interrupted_stage=self.interrupted_stage,
            failed_stage=self.failed_stage,
        )
        payload = {
            "mode": "overnight_stock_alpha_interrupted_summary_research_only",
            "output_root": manifest["output_root"],
            "output_dir": manifest["output_dir"],
            "run_size": manifest["run_size"],
            "completed_stages": manifest["completed_stages"],
            "interrupted_stage": self.interrupted_stage,
            "missing_stages": manifest["missing_stages"],
            "existing_output_paths": manifest["existing_output_paths"],
            "resume_appears_safe": self.interrupted_stage is not None,
            "resume_command": manifest["resume_command"],
            "note": "Completed outputs were not deleted or modified by this interrupted summary.",
            "guardrails": dict(GUARDRAILS),
            **GUARDRAILS,
        }
        writer = ResearchArtifactWriter()
        writer.write_json(paths.json_path, payload)
        writer.write_markdown(paths.markdown_path, _interrupted_markdown(payload))
        return paths

    def _stage(self, stage_name: str) -> dict[str, Any]:
        if stage_name not in self.stage_states:
            raise KeyError(f"Unknown stock-alpha stage: {stage_name}")
        return self.stage_states[stage_name]

def write_stock_alpha_run_status(config: Mapping[str, Any]) -> StockAlphaRunStatusPaths:
    settings = StockLevelResearchConfig.from_mapping(config)
    output_dir = settings.output_dir
    paths = run_status_paths(output_dir)
    payload = inspect_stock_alpha_run_status(config)
    writer = ResearchArtifactWriter()
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload, title="Stock Alpha Run Status"))
    return paths


def inspect_stock_alpha_run_status(config: Mapping[str, Any]) -> dict[str, Any]:
    settings = StockLevelResearchConfig.from_mapping(config)
    output_dir = settings.output_dir
    stage_states = _detected_stage_states(config, output_dir)
    return build_manifest_payload(
        config,
        output_dir,
        stage_states,
        mode="ml-stock-alpha-run-status",
        started_at=None,
    )
