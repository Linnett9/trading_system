from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.research.ml.allocation.types import AllocationPolicyDefinition, AllocationV2Paths


def _write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe_csv_row(row))


def _write_leaderboard(
    path: Path,
    ranked_payloads: list[dict[str, Any]],
    skipped_payloads: list[dict[str, Any]],
) -> None:
    lines = [
        "# Allocation Policy Leaderboard",
        "",
        "Trading impact: none",
        "",
    ]
    for row in ranked_payloads:
        lines.append(
            f"{row['rank']}. {row['policy_name']} - total_return={row['total_return']}"
        )
    if skipped_payloads:
        lines.extend(["", "Skipped policies:"])
        for row in skipped_payloads:
            lines.append(f"- {row['policy_name']}: {row.get('skip_reason')}")
    lines.extend(["", "Research only. Trading impact: none. Production validated: false."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_diagnostics_markdown(
    path: Path,
    ranked_payloads: list[dict[str, Any]],
    skipped_payloads: list[dict[str, Any]],
) -> None:
    lines = ["# Allocation Policy Diagnostics", ""]
    lines.append(f"Ranked policies: {len(ranked_payloads)}")
    lines.append(f"Skipped policies: {len(skipped_payloads)}")
    lines.extend(["", "Research only. Trading impact: none. Production validated: false."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_grid_search_reports(
    paths: AllocationV2Paths,
    grid_search: dict[str, Any],
    config: dict[str, Any],
    *,
    result_payload: Any,
) -> None:
    del config
    candidates = grid_search.get("candidates", [])
    with paths.grid_search_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=sorted({key for row in candidates for key in row}),
        )
        writer.writeheader()
        for row in candidates:
            writer.writerow(_json_safe_csv_row(row))
    selected = grid_search.get("selected")
    selected_diagnostic_policy = None
    if selected:
        selected_diagnostic_policy = {
            "candidate_id": selected["candidate"]["candidate_id"],
            "objective": selected["objective"],
            "selection_protocol": grid_search.get("selection_protocol"),
            "selection_notice": grid_search.get("selection_notice"),
            "selection_metrics": result_payload(selected["selection_result"]),
            "holdout_metrics": result_payload(selected["result"]),
            "overfit_warning": selected["definition"].overfit_warning,
        }
    paths.grid_search_json.write_text(
        json.dumps(
            {
                **_json_safe_payload(grid_search),
                "grid_size": grid_search.get("grid_size", len(candidates)),
                "selected_diagnostic_policy": selected_diagnostic_policy,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths.grid_search_markdown.write_text(
        "# Allocation Policy Grid Search\n\n"
        f"Grid size: {grid_search.get('grid_size', len(candidates))}\n\n"
        "Research only. Trading impact: none. Production validated: false.\n",
        encoding="utf-8",
    )


def _validate_output_consistency(paths: AllocationV2Paths) -> None:
    comparison = _load_json(paths.comparison_json)
    diagnostics = _load_json(paths.diagnostics_json)
    grid_search = _load_json(paths.grid_search_json)
    optimizer = _load_json(paths.optimizer_results_json)

    for payload in (comparison, diagnostics, optimizer):
        if not isinstance(payload, dict):
            raise ValueError("Allocation v2 outputs contain inconsistent payloads")
        if payload.get("research_only") is not True:
            raise ValueError("Allocation v2 outputs contain inconsistent payloads")
        if payload.get("trading_impact") != "none":
            raise ValueError("Allocation v2 outputs contain inconsistent payloads")
        if payload.get("production_validated") is not False:
            raise ValueError("Allocation v2 outputs contain inconsistent payloads")

    if not isinstance(grid_search, dict):
        raise ValueError("Allocation v2 outputs contain inconsistent payloads")

    for path in (
        paths.comparison_csv,
        paths.leaderboard_markdown,
        paths.shadow_overlay_json,
        paths.diagnostics_markdown,
        paths.grid_search_csv,
        paths.grid_search_markdown,
        paths.optimizer_candidates_csv,
        paths.optimizer_report_markdown,
        paths.selected_optimizer_exposure_path_csv,
        paths.selected_optimizer_exposure_path_json,
    ):
        if not path.exists():
            raise ValueError("Allocation v2 outputs contain inconsistent payloads")

    if comparison.get("mode") != "allocation_policy_comparison_v2_research_only":
        raise ValueError("Allocation v2 outputs contain inconsistent payloads")
    if diagnostics.get("mode") != "allocation_policy_diagnostics_v2_research_only":
        raise ValueError("Allocation v2 outputs contain inconsistent payloads")
    if grid_search.get("selection_protocol") is None:
        raise ValueError("Allocation v2 outputs contain inconsistent payloads")
    if grid_search.get("grid_size") is None:
        raise ValueError("Allocation v2 outputs contain inconsistent payloads")
    if "selected_diagnostic_policy" not in grid_search:
        raise ValueError("Allocation v2 outputs contain inconsistent payloads")
    if optimizer.get("mode") != "allocation_optimizer_research_only":
        raise ValueError("Allocation v2 outputs contain inconsistent payloads")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe_payload(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _json_safe_payload(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _json_safe_payload(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_payload(item) for item in value]
    if callable(value):
        return str(value)
    return value


def _json_safe_csv_row(row: dict[str, Any]) -> dict[str, str]:
    payload = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            payload[key] = json.dumps(_json_safe_payload(value), sort_keys=True)
        elif value is None:
            payload[key] = ""
        else:
            payload[key] = str(value)
    return payload


def _unavailable_policy_payload(
    definition: AllocationPolicyDefinition,
    reason: str,
) -> dict[str, Any]:
    return {
        "policy_name": definition.policy_name,
        "policy_version": definition.policy_version,
        "policy_kind": definition.policy_kind,
        "available": False,
        "skip_reason": reason,
        "research_only": True,
        "production_validated": False,
        "trading_impact": "none",
        "robustness_flags": {"exposure_is_constant": False},
    }


def _shadow_policy_payload(
    definition: AllocationPolicyDefinition,
    rows: list[dict[str, str]],
    exposures: list[float] | None,
    skip_reason: str | None,
) -> dict[str, Any]:
    del rows
    return {
        "policy_name": definition.policy_name,
        "available": exposures is not None and skip_reason is None,
        "skip_reason": skip_reason,
        "exposure_count": len(exposures or []),
    }
