from __future__ import annotations

import json
from pathlib import Path

from core.research.ml.ds24_forward_metrics_capability import (
    CAPABILITY_SCHEMA_ID,
    build_forward_metrics_capability,
    write_forward_metrics_capability,
)
from core.research.ml.ds24_metrics_only_evaluator import validate_extended_metrics_writer_capability


def test_build_capability_uses_current_extended_metrics_registry() -> None:
    capability = build_forward_metrics_capability(
        family="elastic_net",
        worker_kind="TABULAR",
        worker_script="scripts/local/ds24_p8_r14_e3g_c2_r7_r14_policy_worker.py",
        metrics_root_name="metrics_only_v3_r40_elastic_net",
        evidence_basis={"ticket": "test"},
    )

    result = validate_extended_metrics_writer_capability(capability, family="elastic_net")

    assert capability["schema_id"] == CAPABILITY_SCHEMA_ID
    assert capability["writer_initialised_before_first_prediction"] is True
    assert capability["atomic_retention_gate_available"] is True
    assert capability["paper_orders"] == 0
    assert capability["live_orders"] == 0
    assert capability["holdout_accessed"] is False
    assert result["admitted"] is True
    assert capability["validation"]["admitted"] is True
    assert capability["capability_hash"]


def test_write_capability_revalidates_before_atomic_publish(tmp_path: Path) -> None:
    path = tmp_path / "elastic_net.forward_metrics_capability.json"
    capability = build_forward_metrics_capability(
        family="elastic_net",
        worker_kind="TABULAR",
        worker_script="scripts/local/ds24_p8_r14_e3g_c2_r7_r14_policy_worker.py",
        metrics_root_name="metrics_only_v3_r40_elastic_net",
    )

    publication = write_forward_metrics_capability(path, capability)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert publication["published"] is True
    assert payload["family"] == "elastic_net"
    assert payload["validation"]["admitted"] is True
