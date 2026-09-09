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
    mac_aux_root = stage / "mac_aux" / "queue=DS24_MAC_AUX_NINE_FAMILY_R1" / "family=lightgbm_rank_xendcg"
    (mac_aux_root / "model_artifacts").mkdir(parents=True)
    (mac_aux_root / "metrics_only_v3").mkdir()
    (mac_aux_root / "checkpoints").mkdir()
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
    monkeypatch.setattr(importer, "MAC_AUX_ROOT", stage / "mac_aux" / "queue=DS24_MAC_AUX_NINE_FAMILY_R1")
    monkeypatch.setattr(importer, "XENDCG_MAC_AUX_FAMILY_ROOT", mac_aux_root)
    monkeypatch.setattr(importer, "LAMBDARANK_MAC_AUX_FAMILY_ROOT", stage / "mac_aux" / "queue=DS24_MAC_AUX_NINE_FAMILY_R1" / "family=lightgbm_lambdarank")
    monkeypatch.setattr(importer, "PRODUCER_AUTHORITY_ROOT", stage / "producer_authority")
    monkeypatch.setattr(importer, "R47_PRE_IMPORT_SNAPSHOT_PATH", stage / "R47_cross_host_pre_import_snapshot.json")
    monkeypatch.setattr(importer, "R47_IMPORT_AUTHORITY_PATH", stage / "R47_cross_host_import_authority.json")
    monkeypatch.setattr(importer, "R47_IMPORT_LEDGER_PATH", stage / "R47_cross_host_import_ledger.jsonl")
    monkeypatch.setattr(importer, "R47_FAMILY_IMPORT_STATUS_PATH", stage / "R47_family_import_status.json")
    monkeypatch.setattr(importer, "R47_OWNERSHIP_STATE_PATH", stage / "R47_cross_host_ownership_state.json")
    monkeypatch.setattr(importer, "R47_DELL_EFFECTIVE_READY_QUEUE_PATH", stage / "R47_dell_effective_ready_queue.json")
    monkeypatch.setattr(importer, "R47_MAC_TRANSFER_CONTRACT_PATH", stage / "R47_mac_future_transfer_contract.json")
    monkeypatch.setattr(importer, "R47A_XENDCG_SOURCE_DISCOVERY_PATH", stage / "R47A_xendcg_source_discovery.json")
    monkeypatch.setattr(importer, "R47A_XENDCG_ARTIFACT_VALIDATION_PATH", stage / "R47A_xendcg_artifact_validation.json")
    monkeypatch.setattr(importer, "R47A_XENDCG_IMPORT_AUTHORITY_PATH", stage / "R47A_xendcg_import_authority.json")
    monkeypatch.setattr(importer, "R47A_XENDCG_IMPORT_RESULT_PATH", stage / "R47A_xendcg_import_result.json")
    monkeypatch.setattr(importer, "R47A_OWNERSHIP_STATE_PATH", stage / "R47A_cross_host_ownership_state.json")
    monkeypatch.setattr(importer, "R47A_DELL_EFFECTIVE_READY_QUEUE_PATH", stage / "R47A_dell_effective_ready_queue.json")
    monkeypatch.setattr(importer, "XENDCG_EXPECTED_MODEL_ARTIFACTS", 2)
    monkeypatch.setattr(importer, "XENDCG_EXPECTED_FIRST_ORDINAL", 2)
    monkeypatch.setattr(importer, "XENDCG_EXPECTED_LAST_ORDINAL", 3)
    monkeypatch.setattr(importer, "XENDCG_FIXTURE_REFITS", ("000002", "000003"))


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


def _complete_mac_aux_xendcg_root() -> None:
    root = importer.XENDCG_MAC_AUX_FAMILY_ROOT
    (root / "family_execution_summary.json").write_text(
        json.dumps(
            {
                "family": "lightgbm_rank_xendcg",
                "live_orders": 0,
                "metrics_rows": 2,
                "paper_orders": 0,
                "run_id": "DS24_MAC_AUX_NINE_FAMILY_R1",
                "status": "PASS",
                "validation": {"status": "PASS"},
                "zero_holdout": True,
            }
        ),
        encoding="utf-8",
    )
    (root / "checkpoints" / "latest.json").write_text(
        json.dumps({"cursor": "family_complete", "family": "lightgbm_rank_xendcg"}),
        encoding="utf-8",
    )
    for refit, body in (("000002", b"a"), ("000003", b"b")):
        (root / "model_artifacts" / f"lightgbm_rank_xendcg_refit={refit}.pkl").write_bytes(body)
    (root / "model_artifacts" / "._lightgbm_rank_xendcg_refit=000003.pkl").write_bytes(b"sidecar")
    oof_file = root / "ensemble_oof_scores_v2" / "decision_date=2016-01-05" / "part-refit=000002.parquet"
    oof_file.parent.mkdir(parents=True)
    oof_file.write_bytes(b"oof-a")
    second_oof = root / "ensemble_oof_scores_v2" / "decision_date=2016-01-06" / "part-refit=000003.parquet"
    second_oof.parent.mkdir(parents=True)
    second_oof.write_bytes(b"oof-b")
    manifest = {
        "distinct_assets": 2,
        "distinct_decision_timestamps": 2,
        "family": "lightgbm_rank_xendcg",
        "files": [
            {
                "decision_date": "2016-01-05",
                "refit_ordinal": 2,
                "relative_path": "ensemble_oof_scores_v2/decision_date=2016-01-05/part-refit=000002.parquet",
                "row_count": 2,
                "sha256": importer.file_hash(oof_file),
            },
            {
                "decision_date": "2016-01-06",
                "refit_ordinal": 3,
                "relative_path": "ensemble_oof_scores_v2/decision_date=2016-01-06/part-refit=000003.parquet",
                "row_count": 2,
                "sha256": importer.file_hash(second_oof),
            },
        ],
        "row_count": 4,
        "run_id": "DS24_MAC_AUX_NINE_FAMILY_R1",
    }
    (root / "ensemble_oof_scores_manifest_v2.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "metrics_only_v3" / "resolved_performance_summary_v3.json").write_text(
        json.dumps(
            {
                "decision_rows": 40,
                "family": "lightgbm_rank_xendcg",
                "holdout_rows": 0,
                "live_orders": 0,
                "mean_spearman_rank_ic": 0.25,
                "paper_orders": 0,
                "rank_ic_rows": 2,
                "status": "PASS",
                "target_contract": "forward_return_60m__decision_5m",
            }
        ),
        encoding="utf-8",
    )
    (root / "metrics_only_v3" / "per_t_metrics.parquet").write_bytes(b"per-t")
    (root / "metrics_only_v3" / "decision_trace.parquet").write_bytes(b"trace")
    authority = importer.PRODUCER_AUTHORITY_ROOT
    authority.mkdir(parents=True)
    (authority / "source_manifest.json").write_text(
        json.dumps(
            {
                "mac_head": importer.XENDCG_MAC_HEAD,
                "producer_source_classification": "MAC_WORKTREE_UNCOMMITTED_SOURCE_OVER_ADVERTISED_HEAD",
                "feature_authority": {
                    "feature_order_sha256": importer.XENDCG_FEATURE_ORDER_SHA,
                    "predictor_count": 101,
                },
                "estimator_configuration_source": {
                    "params": {
                        "learning_rate": 0.05,
                        "min_child_samples": 10,
                        "n_estimators": 25,
                        "n_jobs": 4,
                        "num_leaves": 15,
                        "num_threads": 4,
                        "objective": "rank_xendcg",
                        "random_state": 1729,
                    }
                },
                "runtime_arguments": {"lookback_sessions": 20, "max_training_rows": 24000},
                "source_files": [
                    {
                        "relative_path": "core/research/ml/ds24/mac_aux_queue_r44f2.py",
                        "sha256": importer.XENDCG_PRODUCER_SHA,
                    }
                ],
                "xendcg_artifact_hashes": [
                    {
                        "refit": "000002",
                        "model_sha256": importer.file_hash(root / "model_artifacts" / "lightgbm_rank_xendcg_refit=000002.pkl"),
                        "model_size_bytes": 1,
                        "oof_sha256": importer.file_hash(oof_file),
                        "oof_size_bytes": 5,
                    },
                    {
                        "refit": "000003",
                        "model_sha256": importer.file_hash(root / "model_artifacts" / "lightgbm_rank_xendcg_refit=000003.pkl"),
                        "model_size_bytes": 1,
                        "oof_sha256": importer.file_hash(second_oof),
                        "oof_size_bytes": 5,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (authority / "three_refit_parity.json").write_text(
        json.dumps(
            {
                "classification": "XENDCG_PRODUCTION_PARITY_PASS",
                "results": [
                    {
                        "refit_ordinal": 2,
                        "fresh_vs_saved_model_max_abs_diff": 0.0,
                        "fresh_vs_oof_max_abs_diff": 1e-8,
                        "saved_model_vs_oof_max_abs_diff": 1e-8,
                        "fresh_top20_membership_match": True,
                        "fresh_top20_order_match": True,
                        "saved_model_top20_membership_match": True,
                        "saved_model_top20_order_match": True,
                    },
                    {
                        "refit_ordinal": 3,
                        "fresh_vs_saved_model_max_abs_diff": 0.0,
                        "fresh_vs_oof_max_abs_diff": 1e-8,
                        "saved_model_vs_oof_max_abs_diff": 1e-8,
                        "fresh_top20_membership_match": True,
                        "fresh_top20_order_match": True,
                        "saved_model_top20_membership_match": True,
                        "saved_model_top20_order_match": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_xendcg_import_validation_succeeds_with_complete_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_stage(tmp_path, monkeypatch)
    _complete_mac_aux_xendcg_root()

    result = importer.run_import("lightgbm_rank_xendcg")
    ownership = json.loads(importer.R47_OWNERSHIP_STATE_PATH.read_text(encoding="utf-8"))

    assert result["classification"] == "COMPLETE_IMPORTED"
    assert ownership["by_family"]["lightgbm_rank_xendcg"]["owner_state"] == "COMPLETE_IMPORTED"
    assert "lightgbm_rank_xendcg" not in result["effective_queue"]
    assert importer.verify_family_import("lightgbm_rank_xendcg").apple_double_sidecar_count == 1


def test_duplicate_identical_import_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_stage(tmp_path, monkeypatch)
    _complete_mac_aux_xendcg_root()
    first = importer.run_import("lightgbm_rank_xendcg")

    second = importer.run_import("lightgbm_rank_xendcg")

    assert first["classification"] == "COMPLETE_IMPORTED"
    assert second["classification"] == "ALREADY_IMPORTED_IDENTICAL"
    assert len(importer.ledger_events()) == 1


def test_hash_conflict_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_stage(tmp_path, monkeypatch)
    _complete_mac_aux_xendcg_root()
    importer.run_import("lightgbm_rank_xendcg")
    (importer.XENDCG_MAC_AUX_FAMILY_ROOT / "model_artifacts" / "lightgbm_rank_xendcg_refit=000003.pkl").write_bytes(b"changed")

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


def test_mac_aux_artifact_inventory_counts_ordinals_and_sidecars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_stage(tmp_path, monkeypatch)
    _complete_mac_aux_xendcg_root()

    inventory = importer.xendcg_model_inventory()

    assert inventory["genuine_pkl_count"] == 2
    assert inventory["apple_double_sidecar_count"] == 1
    assert inventory["missing_ordinal_count"] == 0
    assert inventory["duplicate_ordinal_count"] == 0
    assert inventory["first_ordinal"] == "000002"
    assert inventory["last_ordinal"] == "000003"


def test_r47a_import_appends_success_then_duplicate_noops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_stage(tmp_path, monkeypatch)
    _complete_mac_aux_xendcg_root()
    importer.append_ledger_event(
        {
            "family": "lightgbm_rank_xendcg",
            "source_run_id": importer.XENDCG_SOURCE_RUN_ID,
            "validation_classification": "IMPORT_BLOCKED_MISSING_AUTHORITY_OR_ARTIFACTS",
            "artifact_hash": "",
        }
    )

    first = importer.run_r47a_xendcg_import()
    second = importer.run_r47a_xendcg_import()

    assert first["classification"] == "COMPLETE_IMPORTED"
    assert first["ledger_appended"] is True
    assert second["classification"] == "ALREADY_IMPORTED_IDENTICAL"
    assert second["ledger_appended"] is False
    assert len(importer.ledger_events()) == 2


def test_r47a_hash_conflict_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_stage(tmp_path, monkeypatch)
    _complete_mac_aux_xendcg_root()
    importer.run_r47a_xendcg_import()
    (importer.XENDCG_MAC_AUX_FAMILY_ROOT / "model_artifacts" / "lightgbm_rank_xendcg_refit=000003.pkl").write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="CROSS_HOST_IMPORT_HASH_CONFLICT"):
        importer.run_r47a_xendcg_import()
