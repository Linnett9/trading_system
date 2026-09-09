from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.local import ds24_cross_host_result_import as importer
from scripts.local import ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor as supervisor


R42_QUEUE = [
    "random_forest",
    "elastic_net",
    "lightgbm_rank_xendcg",
    "lightgbm_lambdarank",
    "DLinear",
    "PatchTST",
    "Transformer",
    "iTransformer",
    "Momentum Transformer",
    "Market Context Encoder",
]


def _configure_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    model_root = stage / "model_artifacts"
    model_root.mkdir()
    r42 = stage / "R42_ready_family_queue.json"
    matrix = stage / "R42_full_family_readiness_matrix.json"
    r44 = stage / "R44_cross_host_family_ownership.json"
    r42.write_text(
        json.dumps({"ticket": supervisor.R42_READY_QUEUE_AUTHORITY_ID, "ready_family_queue": R42_QUEUE}),
        encoding="utf-8",
    )
    matrix.write_text(
        json.dumps({"families": [{"family": family, "current_r42_state": "READY_TO_LAUNCH"} for family in R42_QUEUE]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(supervisor, "R42_READY_QUEUE_PATH", r42)
    monkeypatch.setattr(supervisor, "R42_READINESS_MATRIX_PATH", matrix)
    monkeypatch.setattr(supervisor, "R44_CROSS_HOST_OWNERSHIP_PATH", r44)
    supervisor.build_cross_host_ownership_authority(updated_at_utc="2026-09-09T01:00:00+00:00", path=r44)
    monkeypatch.setattr(importer, "STAGE", stage)
    monkeypatch.setattr(importer, "MODEL_ARTIFACT_ROOT", model_root)
    monkeypatch.setattr(importer, "OOF_ROOT", stage / "oof_predictions")
    monkeypatch.setattr(importer, "R47_PRE_IMPORT_SNAPSHOT_PATH", stage / "R47_cross_host_pre_import_snapshot.json")
    monkeypatch.setattr(importer, "R47_IMPORT_AUTHORITY_PATH", stage / "R47_cross_host_import_authority.json")
    monkeypatch.setattr(importer, "R47_IMPORT_LEDGER_PATH", stage / "R47_cross_host_import_ledger.jsonl")
    monkeypatch.setattr(importer, "R47_FAMILY_IMPORT_STATUS_PATH", stage / "R47_family_import_status.json")
    monkeypatch.setattr(importer, "R47_OWNERSHIP_STATE_PATH", stage / "R47_cross_host_ownership_state.json")
    monkeypatch.setattr(importer, "R47_DELL_EFFECTIVE_READY_QUEUE_PATH", stage / "R47_dell_effective_ready_queue.json")
    monkeypatch.setattr(importer, "R47_MAC_TRANSFER_CONTRACT_PATH", stage / "R47_mac_future_transfer_contract.json")
    monkeypatch.setattr(importer, "XENDCG_EXPECTED_MODEL_ARTIFACTS", 2)


def _complete_xendcg_status(stage: Path) -> None:
    (stage / "54_rankxendcg_status.json").write_text(
        json.dumps(
            {
                "producer_source_hash": "source-hash",
                "model_config_authority_hash": "config-hash",
                "metrics_manifest": {"rows": 10},
                "metrics_hashes": {"metrics": "hash"},
                "score_oof_population": {"rows": 10},
                "date_range": {"start": "2022-01-01", "end": "2024-12-31"},
                "feature_hash": "feature-hash",
                "holdout_accessed": False,
                "paper_orders": 0,
                "live_orders": 0,
            }
        ),
        encoding="utf-8",
    )


def test_xendcg_import_validation_succeeds_with_complete_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_stage(tmp_path, monkeypatch)
    _complete_xendcg_status(importer.STAGE)
    (importer.MODEL_ARTIFACT_ROOT / "lightgbm_rank_xendcg_a.pkl").write_bytes(b"a")
    (importer.MODEL_ARTIFACT_ROOT / "lightgbm_rank_xendcg_b.pkl").write_bytes(b"b")
    (importer.MODEL_ARTIFACT_ROOT / "._lightgbm_rank_xendcg_b.pkl").write_bytes(b"sidecar")

    result = importer.run_import("lightgbm_rank_xendcg")
    ownership = json.loads(importer.R47_OWNERSHIP_STATE_PATH.read_text(encoding="utf-8"))

    assert result["classification"] == "COMPLETE_IMPORTED"
    assert ownership["by_family"]["lightgbm_rank_xendcg"]["owner_state"] == "COMPLETE_IMPORTED"
    assert "lightgbm_rank_xendcg" not in result["effective_queue"]
    assert importer.verify_family_import("lightgbm_rank_xendcg").apple_double_sidecar_count == 1


def test_duplicate_identical_import_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_stage(tmp_path, monkeypatch)
    _complete_xendcg_status(importer.STAGE)
    (importer.MODEL_ARTIFACT_ROOT / "lightgbm_rank_xendcg_a.pkl").write_bytes(b"a")
    (importer.MODEL_ARTIFACT_ROOT / "lightgbm_rank_xendcg_b.pkl").write_bytes(b"b")
    first = importer.run_import("lightgbm_rank_xendcg")

    second = importer.run_import("lightgbm_rank_xendcg")

    assert first["classification"] == "COMPLETE_IMPORTED"
    assert second["classification"] == "ALREADY_IMPORTED_IDENTICAL"
    assert len(importer.ledger_events()) == 1


def test_hash_conflict_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_stage(tmp_path, monkeypatch)
    _complete_xendcg_status(importer.STAGE)
    (importer.MODEL_ARTIFACT_ROOT / "lightgbm_rank_xendcg_a.pkl").write_bytes(b"a")
    (importer.MODEL_ARTIFACT_ROOT / "lightgbm_rank_xendcg_b.pkl").write_bytes(b"b")
    importer.run_import("lightgbm_rank_xendcg")
    (importer.MODEL_ARTIFACT_ROOT / "lightgbm_rank_xendcg_b.pkl").write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="CROSS_HOST_IMPORT_HASH_CONFLICT"):
        importer.run_import("lightgbm_rank_xendcg")


def test_lambdarank_awaits_mac_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_stage(tmp_path, monkeypatch)

    status = importer.verify_family_import("lightgbm_lambdarank")

    assert status.classification == "IMPORT_BLOCKED_MISSING_AUTHORITY_OR_ARTIFACTS"
    assert "mac_completion_artifacts_not_present" in status.missing_requirements


def test_dlinear_remains_reserved_without_completion_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_stage(tmp_path, monkeypatch)

    status = importer.verify_family_import("DLinear")

    assert status.classification == "IMPORT_BLOCKED_MISSING_AUTHORITY_OR_ARTIFACTS"
    assert "mac_dlinear_not_started_or_completed" in status.missing_requirements


def test_complete_imported_is_terminal_for_supervisor_admission() -> None:
    row = {
        "family": "lightgbm_rank_xendcg",
        "state": "COMPLETE_IMPORTED",
        "pid_alive": False,
        "duplicate_worker_count": 0,
    }

    assert supervisor.family_specific_skip_reason(row, allowed_states={"V3_CERTIFIED_READY"}) == "COMPLETE"
