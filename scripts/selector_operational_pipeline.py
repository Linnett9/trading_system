from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.research.ml.selector_operational_inputs import (
    resolve_outcome_maturity_cutoff,
)


CONTRACT = "selector_operational_pipeline_state.v2"
STAGES = (
    "parent_stages_1_10",
    "operational_inputs",
    "stage_11_component_batch",
    "stage_12_component_validation",
    "stage_12_panel_resolution",
    "stage_12_evaluation_preflight",
)
RETRYABLE = {"FAILED_RETRYABLE", "INTERRUPTED_RETRYABLE"}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


class DirectSelectorPipeline:
    def __init__(
        self,
        args: argparse.Namespace,
        executor: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ):
        self.args = args
        self.executor = executor
        cutoff = resolve_outcome_maturity_cutoff(
            outcome_maturity_cutoff=getattr(
                args, "outcome_maturity_cutoff", None
            ),
            evaluation_cutoff=getattr(args, "evaluation_cutoff", None),
        )
        args.outcome_maturity_cutoff = cutoff["outcome_maturity_cutoff"]
        settings = self._settings()
        if args.state_path.exists():
            if not args.resume:
                raise ValueError("Existing selector pipeline state requires --resume")
            self.state = json.loads(args.state_path.read_text(encoding="utf-8"))
            if self.state.get("contract_version") != CONTRACT:
                raise ValueError("Incompatible selector pipeline state")
            if self.state.get("run_id") != args.run_id:
                raise ValueError("Selector pipeline run ID changed on resume")
            if self.state.get("settings") != settings:
                raise ValueError("Selector pipeline settings changed on resume")
            stages = self.state.get("stages")
            if (
                self.state.get("stage_order") != list(STAGES)
                or not isinstance(stages, dict)
                or set(stages) != set(STAGES)
                or any(
                    status not in {
                        "WAITING", "RUNNING", "COMPLETED", *RETRYABLE
                    }
                    for status in stages.values()
                )
            ):
                raise ValueError("Incompatible selector pipeline stage state")
            for stage in STAGES:
                if stages[stage] == "RUNNING":
                    stages[stage] = "INTERRUPTED_RETRYABLE"
            atomic_json(args.state_path, self.state)
        else:
            if args.resume:
                raise ValueError("Selector pipeline resume state does not exist")
            self.state = {
                "contract_version": CONTRACT,
                "run_id": args.run_id,
                **cutoff,
                "settings": settings,
                "stage_order": list(STAGES),
                "stages": {stage: "WAITING" for stage in STAGES},
                "attempts": [],
                "production_completion_claimed": False,
            }
            atomic_json(args.state_path, self.state)

    def _settings(self) -> dict[str, Any]:
        return {
            "selector_run_root": str(self.args.selector_run_root),
            "operational_inputs_root": str(self.args.operational_inputs_root),
            "report_root": str(self.args.report_root),
            "component_root": str(self.args.component_root),
            "selector_config": str(self.args.selector_config),
            "panel_config": str(self.args.panel_config),
            "frozen_panel": str(self.args.frozen_panel),
            "outcome_maturity_cutoff": self.args.outcome_maturity_cutoff,
            "operational_input_workers": self.args.operational_input_workers,
            "max_component_workers": self.args.max_component_workers,
            "weighted_capacity": self.args.weighted_capacity,
        }

    def command(
        self, stage: str, command: Sequence[str], transcript: Path
    ) -> None:
        status = self.state["stages"][stage]
        if status == "COMPLETED":
            return
        if status not in {"WAITING", *RETRYABLE}:
            raise ValueError(f"Stage is not safely runnable: {stage}:{status}")
        self.state["stages"][stage] = "RUNNING"
        attempt = {
            "stage": stage,
            "command": subprocess.list2cmdline(list(command)),
            "started_at": _now(),
            "exit_code": None,
            "transcript": str(transcript),
        }
        self.state["attempts"].append(attempt)
        atomic_json(self.args.state_path, self.state)
        result = self.executor(
            list(command),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript.write_text(
            (result.stdout or "") + (result.stderr or ""), encoding="utf-8"
        )
        attempt["exit_code"] = result.returncode
        attempt["finished_at"] = _now()
        self.state["stages"][stage] = (
            "COMPLETED" if result.returncode == 0 else "FAILED_RETRYABLE"
        )
        atomic_json(self.args.state_path, self.state)
        if result.returncode:
            raise RuntimeError(f"{stage} failed with exit code {result.returncode}")

    def run(self) -> None:
        transcript_root = self.args.report_root / "transcripts"
        self.command(
            "parent_stages_1_10",
            self._parent_command(transcript_root / "parent_stages_1_10.txt"),
            transcript_root / "parent_stages_1_10.txt",
        )
        if self._stop("parent_stages_1_10"):
            return
        artifacts = self._parent_artifacts()
        inventory_path = self.args.operational_inputs_root / "inventory.json"
        self.command(
            "operational_inputs",
            [
                sys.executable,
                "scripts/build_selector_operational_inputs.py",
                "--plan",
                artifacts["component_preflight"],
                "--selector-dataset",
                str(Path(artifacts["dataset_manifest"]).parent),
                "--parent-gate",
                artifacts["parent_gate"],
                "--output-root",
                str(self.args.operational_inputs_root),
                "--outcome-maturity-cutoff",
                self.args.outcome_maturity_cutoff,
                "--selector-run-id",
                self.args.run_id,
                "--max-workers",
                str(self.args.operational_input_workers),
            ],
            transcript_root / "operational_inputs.txt",
        )
        if self._stop("operational_inputs"):
            return
        self.command(
            "stage_11_component_batch",
            [
                sys.executable,
                "scripts/run_selector_component_batch.py",
                "--readiness",
                artifacts["component_preflight"],
                "--input-inventory",
                str(inventory_path),
                "--parent-gate",
                artifacts["parent_gate"],
                "--experiment-ledger",
                str(self.args.report_root / "experiment_ledger.jsonl"),
                "--output-root",
                str(self.args.report_root / "component_batch"),
                "--max-component-workers",
                str(self.args.max_component_workers),
                "--weighted-capacity",
                str(self.args.weighted_capacity),
            ],
            transcript_root / "stage_11_component_batch.txt",
        )
        if self._stop("stage_11_component_batch"):
            return
        self.command(
            "stage_12_component_validation",
            [
                sys.executable,
                "main.py",
                "--mode",
                "ml-selector-component-preflight",
                "--parent-gate",
                artifacts["parent_gate"],
                "--selector-dataset-root",
                str(Path(artifacts["dataset_manifest"]).parent),
                "--component-output-root",
                str(self.args.component_root),
                "--approved-component-root",
                str(self.args.component_root),
                "--config",
                str(self.args.selector_config),
                "--verification-output",
                artifacts["component_preflight"],
            ],
            transcript_root / "stage_12_component_validation.txt",
        )
        if self._stop("stage_12_component_validation"):
            return
        self.command(
            "stage_12_panel_resolution",
            [
                sys.executable,
                "main.py",
                "--mode",
                "ml-selector-panel-resolve",
                "--selector-manifest-root",
                str(self.args.component_root),
                "--panel-config",
                str(self.args.panel_config),
                "--panel-output-root",
                str(self.args.frozen_panel.parent),
            ],
            transcript_root / "stage_12_panel_resolution.txt",
        )
        if self._stop("stage_12_panel_resolution"):
            return
        self.command(
            "stage_12_evaluation_preflight",
            [
                sys.executable,
                "main.py",
                "--mode",
                "ml-selector-evaluation-preflight",
                "--frozen-panel",
                str(self.args.frozen_panel),
                "--evaluation-output-root",
                str(self.args.report_root / "evaluation"),
                "--verification-output",
                str(self.args.report_root / "evaluation_preflight"),
            ],
            transcript_root / "stage_12_evaluation_preflight.txt",
        )

    def _parent_command(self, transcript: Path) -> list[str]:
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            "scripts/selector_parent_publication_runbook.ps1",
            "-FromStage",
            "1",
            "-ThroughStage",
            "10",
            "-RunId",
            self.args.run_id,
            "-TranscriptPath",
            str(transcript),
        ]
        if self.args.resume:
            command.insert(6, "-Resume")
        return command

    def _parent_artifacts(self) -> dict[str, str]:
        path = self.args.selector_run_root / "run_state.json"
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            artifacts = state["artifacts"]
            return {
                key: str(artifacts[key])
                for key in ("component_preflight", "dataset_manifest", "parent_gate")
            }
        except (OSError, KeyError, ValueError, TypeError) as exc:
            raise ValueError("Parent runbook artifact state is incomplete") from exc

    def _stop(self, stage: str) -> bool:
        return self.args.stop_after == stage


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Thin authoritative selector operational command wrapper."
    )
    value.add_argument("--run-id", required=True)
    value.add_argument("--outcome-maturity-cutoff")
    value.add_argument("--evaluation-cutoff")
    value.add_argument("--selector-run-root", required=True, type=Path)
    value.add_argument("--operational-inputs-root", required=True, type=Path)
    value.add_argument("--report-root", required=True, type=Path)
    value.add_argument("--component-root", required=True, type=Path)
    value.add_argument("--state-path", required=True, type=Path)
    value.add_argument("--panel-config", required=True, type=Path)
    value.add_argument("--frozen-panel", required=True, type=Path)
    value.add_argument(
        "--selector-config",
        type=Path,
        default=Path(
            "config/config.ticket_7b3_daily_large_history_regeneration_canonical_v2.yaml"
        ),
    )
    value.add_argument("--resume", action="store_true")
    value.add_argument("--operational-input-workers", type=int, default=4)
    value.add_argument("--max-component-workers", type=int, default=3)
    value.add_argument("--weighted-capacity", type=int, default=4)
    value.add_argument("--stop-after", choices=STAGES)
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    args = parser().parse_args()
    DirectSelectorPipeline(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
