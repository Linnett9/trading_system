from __future__ import annotations

import csv
import io
import json
from typing import Any, Mapping, Sequence

from .run_contracts import checksum

LEADERBOARD_CONTRACT = "compute_model_leaderboard.v1"


def results_payload(
    run_identity: str, records: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], str]:
    ordered = sorted((dict(row) for row in records), key=lambda row: (
        str(row.get("item_identity")), str(row.get("result_identity"))
    ))
    payload = {
        "contract_version": "compute_results_snapshot.v1",
        "run_identity": run_identity, "records": ordered,
    }
    payload["logical_checksum"] = checksum(payload)
    columns = {
        "run_identity", "item_identity", "result_identity", "result_kind",
        "pipeline", "stage", "status",
    }
    metric_ids = sorted({
        metric_id for row in ordered for metric_id in row.get("metrics", {})
    })
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=sorted(columns) + metric_ids)
    writer.writeheader()
    for row in ordered:
        flat = {column: row.get(column) for column in columns}
        for metric_id in metric_ids:
            metric = row.get("metrics", {}).get(metric_id)
            flat[metric_id] = (
                metric.get("value") if metric and metric.get("availability") == "AVAILABLE"
                else None
            )
        writer.writerow(flat)
    return payload, output.getvalue()


def build_leaderboard(
    *, run_identity: str, campaign_identity: str | None,
    population_identity: str, ranking_metric: str, ranking_direction: str,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if ranking_direction not in {"HIGHER_IS_BETTER", "LOWER_IS_BETTER"}:
        raise ValueError("Leaderboard ranking direction is invalid")
    eligible, excluded = [], []
    for source in entries:
        row = dict(source)
        reasons = []
        for flag, reason in (
            ("fitted_model_valid", "FITTED_MODEL_MISSING_OR_INVALID"),
            ("prediction_valid", "PREDICTION_MISSING_OR_INVALID"),
            ("matched_evaluation_valid", "MATCHED_EVALUATION_MISSING"),
            ("safeguards_passed", "SAFEGUARDS_FAILED"),
            ("population_compatible", "POPULATION_INCOMPATIBLE"),
            ("promotion_gates_complete", "PROMOTION_GATES_INCOMPLETE"),
        ):
            if row.get(flag) is not True:
                reasons.append(reason)
        metric = row.get("metrics", {}).get(ranking_metric, {})
        if metric.get("availability") != "AVAILABLE" or metric.get("value") is None:
            reasons.append("RANKING_METRIC_UNAVAILABLE")
        if reasons:
            row["rank_eligible"] = False
            row["exclusion_reasons"] = reasons
            row["rank"] = None
            excluded.append(row)
        else:
            row["rank_eligible"] = True
            eligible.append(row)
    reverse = ranking_direction == "HIGHER_IS_BETTER"
    eligible.sort(key=lambda row: (
        -float(row["metrics"][ranking_metric]["value"]) if reverse
        else float(row["metrics"][ranking_metric]["value"]),
        str(row.get("model_component_identity")),
    ))
    for rank, row in enumerate(eligible, 1):
        row["rank"] = rank
        row["exclusion_reasons"] = []
    payload = {
        "contract_version": LEADERBOARD_CONTRACT,
        "run_identity": run_identity, "campaign_identity": campaign_identity,
        "population_identity": population_identity,
        "ranking_metric": ranking_metric, "ranking_direction": ranking_direction,
        "required_eligibility_gates": [
            "fitted_model", "prediction", "matched_evaluation", "safeguards",
            "population", "promotion_gates",
        ],
        "deterministic_tie_policy": "metric_then_model_component_identity_ascending",
        "ordered_entries": eligible, "excluded_entries": sorted(
            excluded, key=lambda row: str(row.get("model_component_identity"))
        ),
    }
    payload["logical_checksum"] = checksum(payload)
    return payload


def render_summary(
    manifest: Mapping[str, Any], status: Mapping[str, Any],
    *, artifact_inventory: Sequence[Mapping[str, Any]] = (),
    leaderboard: Mapping[str, Any] | None = None,
    important_paths: Sequence[str] = (),
) -> str:
    counts = status.get("counts", {})
    winner = (
        leaderboard.get("ordered_entries", [])[0].get("model_component_identity")
        if leaderboard and leaderboard.get("ordered_entries") else None
    )
    lines = [
        "# Compute Run Summary", "",
        "## Run identity", "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Run identity: `{manifest['run_identity']}`",
        f"- Pipeline and stage: `{manifest['pipeline']}` / `{manifest['stage']}`",
        f"- Source commit: `{manifest['source_git_commit']}`",
        f"- Current status: `{status['current_status']}`",
        f"- Started: `{status.get('started_timestamp')}`",
        f"- Completed: `{status.get('completed_timestamp')}`", "",
        "## Progress", "",
    ]
    for name in (
        "expected", "completed", "running", "waiting", "blocked", "failed",
        "cancelled", "skipped_compatible", "incomplete",
    ):
        lines.append(f"- {name.replace('_', ' ').title()}: `{counts.get(name, 0)}`")
    lines.extend([
        "", "## Resources", "",
        f"- Machine profile: `{manifest['machine_profile_identity']}`",
        f"- Reserved RAM bytes: `{status.get('reserved_ram_bytes')}`",
        f"- Measured peak RAM bytes: `{status.get('measured_peak_ram_bytes')}`",
        f"- Resource wait seconds: `{status.get('resource_wait_seconds')}`",
        "", "## Artifact completeness", "",
        f"- Valid artifact entries: `{sum(row.get('compatibility_validation_status') == 'VALID' for row in artifact_inventory)}`",
        "", "## Leaderboard status", "",
        f"- Eligible winner: `{winner}`" if winner else "- No eligible winner exists before complete evaluation evidence.",
        "", "## Failures", "",
        *(f"- {row}" for row in status.get("failure_summary", []) or ["None"]),
        "", "## Blockers", "",
        *(f"- {row}" for row in status.get("blocker_summary", []) or ["None"]),
        "", "## Next required operational action", "",
        f"- {status.get('next_required_action') or 'Review structured status and pipeline-owned evidence.'}",
        "", "## Important artifact paths", "",
        *(f"- `{path}`" for path in important_paths),
    ])
    return "\n".join(lines) + "\n"
