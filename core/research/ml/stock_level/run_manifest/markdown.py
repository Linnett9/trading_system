from __future__ import annotations

from typing import Any


def _markdown(payload: dict[str, Any], *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "Research only. Trading impact: none. Production validated: false.",
        "",
        f"- Mode: `{payload.get('mode')}`",
        f"- Run size: `{payload.get('run_size')}`",
        f"- Output root: `{payload.get('output_root')}`",
        f"- Output dir: `{payload.get('output_dir')}`",
        f"- Legacy output paths allowed: {payload.get('legacy_output_paths_allowed')}",
        f"- Interrupted stage: {payload.get('interrupted_stage')}",
        f"- Next recommended stage: {payload.get('next_recommended_stage')}",
        f"- Resume command: `{payload.get('resume_command')}`",
        "",
        "## Stages",
        "",
        "| stage | status | elapsed_seconds | outputs_exist |",
        "|---|---|---:|---|",
    ]
    for stage in payload.get("stages", []):
        exists = stage.get("output_path_exists", {})
        existing = f"{sum(1 for value in exists.values() if value)}/{len(exists)}"
        lines.append(
            f"| {stage['name']} | {stage['status']} | {_fmt(stage.get('elapsed_seconds'))} | {existing} |"
        )
    lines.extend(["", "## Missing Output Warnings"])
    for warning in payload.get("missing_output_warnings", [])[:25]:
        lines.append(f"- {warning['stage']}.{warning['path_key']}: `{warning['path']}`")
    lines.extend(["", "## Stale Stage Guidance"])
    for item in payload.get("stale_stage_guidance", [])[:25]:
        lines.append(f"- {item['stage']}: {item['reason']}")
        for action in item.get("recommended_actions", []):
            lines.append(f"  - {action}")
    lines.extend(["", "## Legacy Path Warnings"])
    for warning in payload.get("legacy_path_warnings", [])[:25]:
        lines.append(f"- `{warning['path']}`")
    return "\n".join(lines) + "\n"


def _interrupted_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Overnight Stock Alpha Interrupted Summary",
        "",
        "Research only. Trading impact: none. Production validated: false.",
        "",
        f"- Run size: `{payload.get('run_size')}`",
        f"- Output dir: `{payload.get('output_dir')}`",
        f"- Interrupted stage: `{payload.get('interrupted_stage')}`",
        f"- Resume appears safe: {payload.get('resume_appears_safe')}",
        f"- Resume command: `{payload.get('resume_command')}`",
        "- Completed outputs were not deleted or modified.",
        "",
        "## Completed Stages",
    ]
    lines.extend(f"- {stage}" for stage in payload.get("completed_stages", []))
    lines.extend(["", "## Missing Stages"])
    lines.extend(f"- {stage}" for stage in payload.get("missing_stages", []))
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)
