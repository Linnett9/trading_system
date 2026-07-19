from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Callable, Mapping, Sequence


WEIGHTS = {
    "ridge": 1, "elastic_net": 1, "ordered_logit_ranker": 2,
    "huber": 1, "contextual_elastic_net": 1,
    "multi_horizon_ridge": 1, "multi_horizon_elastic_net": 1,
    "lightgbm_rank_xendcg": 2, "lightgbm_lambdarank": 2,
}
EXPECTED_MODELS = ("ridge", "elastic_net", "ordered_logit_ranker")
EXPECTED_JOB_COUNT = 15


def validate_component_plan(
    jobs: Sequence[Mapping[str, Any]],
    *,
    campaign_manifest: Mapping[str, Any] | None = None,
) -> list[Mapping[str, Any]]:
    supplied = list(jobs)
    expected_matrix = None
    if campaign_manifest is None:
        if len(supplied) != EXPECTED_JOB_COUNT:
            raise ValueError("Stage-10 production plan must contain exactly 15 jobs")
    else:
        from core.research.ml.selector_research_campaign import (
            validate_selector_campaign,
        )
        validate_selector_campaign(campaign_manifest)
        expected_matrix = list(campaign_manifest["fitted_component_matrix"])
        if len(supplied) != len(expected_matrix):
            raise ValueError("Runtime component count differs from campaign manifest")
    required = {
        "job_id", "model_id", "prediction_date", "selector_dataset_root",
        "authoritative_output_root", "feature_schema", "target_contract",
        "expected_parent_gate_checksum", "expected_dataset_checksum",
        "dependency_state", "overwrite_policy", "resume_policy", "logical_checksum",
    }
    job_ids: set[str] = set()
    owners: set[tuple[str, str, str]] = set()
    model_dates: dict[str, set[str]] = {model: set() for model in EXPECTED_MODELS}
    by_job_id: dict[str, Mapping[str, Any]] = {}
    for job in supplied:
        missing = sorted(required - set(job))
        if missing:
            raise ValueError(f"Stage-10 job fields missing: {','.join(missing)}")
        job_id = str(job["job_id"])
        if job_id in job_ids:
            raise ValueError(f"Duplicate Stage-10 job ID: {job_id}")
        job_ids.add(job_id)
        by_job_id[job_id] = job
        model = str(job["model_id"])
        date = str(job["prediction_date"])
        horizon = str(job.get("horizon_id") or "")
        owner = (model, date, horizon)
        if owner in owners:
            raise ValueError(f"Duplicate Stage-10 component owner: {model}:{date}:{horizon}")
        owners.add(owner)
        if campaign_manifest is None and model not in model_dates:
            raise ValueError(f"Unsupported Stage-10 model: {model}")
        expected_job_id = (
            f"selector:{date}:{model}:{horizon}" if horizon
            else f"selector:{date}:{model}"
        )
        if job_id != expected_job_id:
            raise ValueError(f"Stage-10 job identity mismatch: {job_id}")
        if model in model_dates:
            model_dates[model].add(date)
    if campaign_manifest is None and any(
        len(dates) != 5 for dates in model_dates.values()
    ):
        raise ValueError("Stage-10 plan must contain five dates for each base model")
    if expected_matrix is not None:
        expected_ids = [str(row["job_id"]) for row in expected_matrix]
        if set(by_job_id) != set(expected_ids):
            missing = sorted(set(expected_ids) - set(by_job_id))
            unexpected = sorted(set(by_job_id) - set(expected_ids))
            raise ValueError(
                "Runtime components differ from campaign manifest: "
                f"missing={','.join(missing)};unexpected={','.join(unexpected)}"
            )
        for expected in expected_matrix:
            actual = by_job_id[str(expected["job_id"])]
            if (
                str(actual["model_id"]) != str(expected["model_id"])
                or str(actual["prediction_date"])
                != str(expected["prediction_date"])
                or (actual.get("horizon_id") or None)
                != (expected.get("horizon_id") or None)
            ):
                raise ValueError("Runtime component identity differs from campaign")
            if actual.get("campaign_identity") != campaign_manifest.get(
                "campaign_identity"
            ):
                raise ValueError("Runtime component campaign identity mismatch")
        return [by_job_id[job_id] for job_id in expected_ids]
    return supplied


def run_component_jobs(
    jobs: Sequence[Mapping[str, Any]],
    *,
    runner: Callable[[Mapping[str, Any]], Any],
    max_component_workers: int = 3,
    capacity: int = 4,
    weights: Mapping[str, int] = WEIGHTS,
    campaign_manifest: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    ordered = validate_component_plan(
        jobs, campaign_manifest=campaign_manifest
    )
    if max_component_workers < 1 or capacity < 1:
        raise ValueError("Invalid scheduler bounds")
    indexed = list(enumerate(ordered))
    for _, job in indexed:
        _weight(job, weights, capacity)
    pending = list(indexed)
    active: dict[Future[Any], tuple[int, Mapping[str, Any], int]] = {}
    evidence: dict[int, dict[str, Any]] = {}
    used_capacity = 0
    halted = False

    with ThreadPoolExecutor(max_workers=max_component_workers) as pool:
        while pending or active:
            while not halted and pending and len(active) < max_component_workers:
                launch_index = next(
                    (
                        index for index, (_, job) in enumerate(pending)
                        if used_capacity + _weight(job, weights, capacity) <= capacity
                    ),
                    None,
                )
                if launch_index is None:
                    break
                plan_index, job = pending.pop(launch_index)
                weight = _weight(job, weights, capacity)
                future = pool.submit(runner, job)
                active[future] = (plan_index, job, weight)
                used_capacity += weight

            if not active:
                if pending and not halted:
                    raise RuntimeError("Scheduler deadlock")
                break

            done, _ = wait(active, return_when=FIRST_COMPLETED)
            completed = sorted(
                ((active.pop(future), future) for future in done),
                key=lambda item: item[0][0],
            )
            for (plan_index, job, weight), future in completed:
                used_capacity -= weight
                try:
                    result = future.result()
                    runner_status = (
                        str(result.get("status", "COMPLETED"))
                        if isinstance(result, Mapping) else "COMPLETED"
                    )
                    status = {
                        "COMPLETE": "COMPLETED",
                        "COMPLETED": "COMPLETED",
                        "SKIPPED_COMPATIBLE": "SKIPPED_COMPATIBLE",
                        "WAITING_FOR_RESOURCES": "WAITING_FOR_RESOURCES",
                        "INCOMPLETE": "INCOMPLETE",
                        "BLOCKED": "BLOCKED",
                        "CORRUPT": "CORRUPT",
                        "CANCELLED": "CANCELLED",
                    }.get(runner_status)
                    if status is None:
                        raise ValueError(
                            f"Unsupported component runner status: {runner_status}"
                        )
                    evidence[plan_index] = _evidence(job, status, weight, result=result)
                except Exception as exc:
                    halted = True
                    evidence[plan_index] = _evidence(
                        job,
                        "FAILED",
                        weight,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )

        for plan_index, job in pending:
            evidence[plan_index] = _evidence(
                job, "NOT_STARTED", _weight(job, weights, capacity)
            )

    return [evidence[index] for index in range(len(ordered))]


def _weight(
    job: Mapping[str, Any], weights: Mapping[str, int], capacity: int
) -> int:
    model = str(job.get("model_id"))
    weight = weights.get(model)
    if not isinstance(weight, int) or weight < 1 or weight > capacity:
        raise ValueError(f"Invalid or over-capacity component weight: {model}")
    return weight


def _evidence(
    job: Mapping[str, Any],
    status: str,
    weight: int,
    *,
    result: Any = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "job_id": str(job["job_id"]),
        "component_identity": {
            "model_id": str(job["model_id"]),
            "prediction_date": str(job["prediction_date"]),
            "horizon_id": job.get("horizon_id"),
        },
        "status": status,
        "weight": weight,
        "runner_result": result,
        "error_type": error_type,
        "error_message": error_message,
    }
