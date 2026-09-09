from __future__ import annotations

import json
from pathlib import Path

from scripts.local import ds24_r42_forward_metrics_and_ready_queue as r42


def test_r42_matrix_repairs_forward_metrics_without_runtime_mutation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(r42, "CAPABILITY_ROOT", tmp_path / "capabilities")
    monkeypatch.setattr(r42, "OUTPUT_JSON", tmp_path / "matrix.json")
    monkeypatch.setattr(r42, "OUTPUT_CSV", tmp_path / "matrix.csv")
    monkeypatch.setattr(r42, "REPAIR_RESULTS_JSON", tmp_path / "results.json")
    monkeypatch.setattr(r42, "READY_QUEUE_JSON", tmp_path / "queue.json")
    monkeypatch.setattr(
        r42,
        "previous_r41_rows",
        lambda: {
            "elastic_net": {"state": "FORWARD_METRICS_CONTRACT_REQUIRED", "blocker": "missing"},
            "random_forest": {"state": "READY_TO_LAUNCH", "blocker": "", "evidence_paths": []},
            "Temporal Fusion Transformer": {"state": "CONFIGURATION_AUTHORITY_REQUIRED", "blocker": "authority"},
        },
    )
    monkeypatch.setattr(
        r42,
        "certification_evidence",
        lambda family: {
            "state": "V3_SEQUENCE_WORKER_CERTIFIED_READY"
            if family in r42.SEQUENCE_FAMILIES
            else "V3_CERTIFIED_READY",
            "path": f"stage/{family}.json",
            "record": {"family": family},
        },
    )

    matrix = r42.build_matrix()
    r42.publish(matrix)
    rows = {row["family"]: row for row in matrix["families"]}
    queue = json.loads((tmp_path / "queue.json").read_text(encoding="utf-8"))

    assert rows["elastic_net"]["current_r42_state"] == "READY_TO_LAUNCH"
    assert rows["Temporal Fusion Transformer"]["current_r42_state"] == "CONFIGURATION_AUTHORITY_REQUIRED"
    assert rows["elastic_net"]["safety"]["worker_launches"] == 0
    assert matrix["live_supervisor_queue_modified"] is False
    assert "elastic_net" in queue["ready_family_queue"]
    assert (tmp_path / "capabilities" / "elastic_net.forward_metrics_capability.json").exists()
    assert (tmp_path / "matrix.csv").read_text(encoding="utf-8").splitlines()[0].split(",") == list(r42.CSV_COLUMNS)
