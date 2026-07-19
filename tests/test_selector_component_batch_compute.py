from __future__ import annotations

from core.research.ml.selector_component_scheduler import run_component_jobs


def test_scheduler_preserves_compute_waiting_and_incomplete_states():
    jobs = [
        {
            "job_id": f"selector:2025-01-{day:02d}:{model}",
            "model_id": model,
            "prediction_date": f"2025-01-{day:02d}",
            "horizon_id": None,
            "selector_dataset_root": "dataset",
            "authoritative_output_root": "output",
            "feature_schema": "features",
            "target_contract": "target",
            "economic_target_id": "forward_return_10d",
            "target_provenance_contract_version": "stock_level_target_provenance_v2",
            "expected_parent_gate_checksum": "gate",
            "expected_dataset_checksum": "dataset",
            "dependency_state": "ready",
            "overwrite_policy": "never",
            "resume_policy": "compatible",
            "logical_checksum": f"{day}-{model}",
        }
        for day in range(1, 6)
        for model in ("ridge", "elastic_net", "ordered_logit_ranker")
    ]
    statuses = iter(
        ["WAITING_FOR_RESOURCES", "INCOMPLETE"] + ["COMPLETE"] * 13
    )
    evidence = run_component_jobs(
        jobs,
        runner=lambda job: {"status": next(statuses)},
        max_component_workers=1,
    )
    assert evidence[0]["status"] == "WAITING_FOR_RESOURCES"
    assert evidence[1]["status"] == "INCOMPLETE"
    assert all(row["status"] == "COMPLETED" for row in evidence[2:])
