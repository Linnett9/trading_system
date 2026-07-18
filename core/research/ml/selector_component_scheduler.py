from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Callable, Mapping, Sequence


WEIGHTS = {"ridge": 1, "elastic_net": 1, "ordered_logit_ranker": 2}
EXPECTED_MODELS = tuple(WEIGHTS)
EXPECTED_JOB_COUNT = 15


def validate_component_plan(jobs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    ordered = list(jobs)
    if len(ordered) != EXPECTED_JOB_COUNT:
        raise ValueError("Stage-10 production plan must contain exactly 15 jobs")
    required = {
        "job_id", "model_id", "prediction_date", "selector_dataset_root",
        "authoritative_output_root", "feature_schema", "target_contract",
        "expected_parent_gate_checksum", "expected_dataset_checksum",
        "dependency_state", "overwrite_policy", "resume_policy", "logical_checksum",
    }
    job_ids: set[str] = set()
    owners: set[tuple[str, str, str]] = set()
    model_dates: dict[str, set[str]] = {model: set() for model in EXPECTED_MODELS}
    for job in ordered:
        missing = sorted(required - set(job))
        if missing:
            raise ValueError(f"Stage-10 job fields missing: {','.join(missing)}")
        job_id = str(job["job_id"])
        if job_id in job_ids:
            raise ValueError(f"Duplicate Stage-10 job ID: {job_id}")
        job_ids.add(job_id)
        model = str(job["model_id"])
        date = str(job["prediction_date"])
        horizon = str(job.get("horizon_id") or "")
        owner = (model, date, horizon)
        if owner in owners:
            raise ValueError(f"Duplicate Stage-10 component owner: {model}:{date}:{horizon}")
        owners.add(owner)
        if model not in model_dates:
            raise ValueError(f"Unsupported Stage-10 model: {model}")
        if job_id != f"selector:{date}:{model}":
            raise ValueError(f"Stage-10 job identity mismatch: {job_id}")
        model_dates[model].add(date)
    if any(len(dates) != 5 for dates in model_dates.values()):
        raise ValueError("Stage-10 plan must contain five dates for each base model")
    return ordered


def run_component_jobs(
    jobs: Sequence[Mapping[str, Any]],
    *,
    runner: Callable[[Mapping[str, Any]], Any],
    max_component_workers: int = 3,
    capacity: int = 4,
    weights: Mapping[str, int] = WEIGHTS,
) -> list[dict[str, Any]]:
    ordered = validate_component_plan(jobs)
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
                    status = (
                        "SKIPPED_COMPATIBLE"
                        if runner_status == "SKIPPED_COMPATIBLE" else "COMPLETED"
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
