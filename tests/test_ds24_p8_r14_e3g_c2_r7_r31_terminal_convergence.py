from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from core.research.ml.ds24_metrics_only_evaluator import append_parquet_log, publish_resolved_performance_contract_v3
from scripts.local import ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor as supervisor
from scripts.local import ds24_p8_r14_e3g_c2_r7_r31_terminal_convergence as r31


def test_policy_terminal_authority_accepts_2000_and_rejects_extra_timestamp() -> None:
    checkpoint = {"last_completed_T": "2026-06-30T20:00:00+00:00"}
    spine = ["2026-06-30T19:55:00+00:00", "2026-06-30T20:00:00+00:00"]

    result = r31.has_reached_registered_terminal("ridge_policy_v1_control", checkpoint, spine)

    assert result["reached"] is True
    assert result["timestamp_identity_match"] is True
    assert r31.has_reached_registered_terminal("ridge_policy_v1_control", {"last_completed_T": "2026-06-30T20:05:00+00:00"}, spine)["reached"] is False


def test_exact_terminal_authority_remains_1900() -> None:
    result = r31.has_reached_registered_terminal("exact_ridge_pca", {"last_completed_T": "2026-06-30T19:00:00Z"})

    assert result["reached"] is True
    assert r31.terminal_authority("exact_ridge_pca")["authority_kind"] == "LAST_TRAINABLE_TARGET_DECISION"


def test_r31_reads_v3_manifest_backed_terminal_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    family = "rff_ridge"
    family_root = tmp_path / family
    metrics_root = family_root / "metrics_only_v3_r37_rff_retry"
    metrics_root.mkdir(parents=True)
    publish_resolved_performance_contract_v3(metrics_root / "resolved_performance_contract_v3.json")
    (family_root / "progress.json").write_text(
        json.dumps({"last_completed_T": "2026-06-30T20:00:00+00:00", "metrics_root_name": metrics_root.name}),
        encoding="utf-8",
    )
    (metrics_root / "checkpoint.json").write_text(
        json.dumps({"last_completed_T": "2026-06-30T20:00:00+00:00"}),
        encoding="utf-8",
    )
    (metrics_root / "resolved_performance_checkpoint_v3.json").write_text(
        json.dumps({"pending_score_rows": 0}),
        encoding="utf-8",
    )
    append_parquet_log(
        metrics_root,
        "rank_ic_v3",
        pd.DataFrame(
            [
                {
                    "family": family,
                    "decision_timestamp": "2024-01-02T14:35:00+00:00",
                    "session_date": "2024-01-02",
                    "spearman_rank_ic": 0.25,
                    "eligible_asset_count": 2,
                    "resolved_asset_count": 2,
                }
            ]
        ),
    )
    append_parquet_log(
        metrics_root,
        "daily_portfolio_returns_v3",
        pd.DataFrame(
            [
                {
                    "session_date": "2024-01-02",
                    "gross_daily_return": 0.01,
                    "net_daily_return": 0.01,
                }
            ]
        ),
        timestamp_col="session_date",
    )
    append_parquet_log(
        metrics_root,
        "sleeve_maturity_ledger_v3",
        pd.DataFrame(
            [
                {
                    "decision_timestamp": "2024-01-02T14:35:00+00:00",
                    "maturity_timestamp": "2024-01-02T15:35:00+00:00",
                    "turnover": 0.5,
                    "transaction_cost_contribution": 0.0,
                    "gross_return_contribution": 0.01,
                    "net_return_contribution": 0.01,
                }
            ]
        ),
    )
    append_parquet_log(
        metrics_root,
        "decision_trace",
        pd.DataFrame(
            [
                {
                    "family": family,
                    "decision_timestamp": "2024-01-02T14:35:00+00:00",
                    "asset_id": "A",
                    "score": 1.0,
                }
            ]
        ),
    )

    monkeypatch.setattr(r31, "POLICY_ROOT", tmp_path)

    inventory = r31.inventory_evidence(family)
    summary = r31.performance_summary(family)

    assert inventory["metrics_layout"] == "V3_APPEND_ONLY_MANIFEST"
    assert inventory["metrics"]["rows"] == 0
    assert inventory["rank_ic_v3"]["rows"] == 1
    assert summary["classification"] == "DS24_R31_V3_TERMINAL_PERFORMANCE_SUMMARY_PUBLISHED"
    assert summary["information_coefficient"]["mean_spearman_rank_ic"] == pytest.approx(0.25)
    assert summary["portfolio_performance"]["top_n_equal_weight"]["cumulative_net_return"] == pytest.approx(0.01)


def test_complete_marker_prevents_supervisor_readmission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    family = "ridge_policy_v1_control"
    root = tmp_path / family
    (root / "metrics_only").mkdir(parents=True)
    (root / "progress.json").write_text(
        json.dumps({"last_completed_T": "2026-06-30T20:00:00+00:00", "family": family}),
        encoding="utf-8",
    )
    (root / r31.COMPLETE_MARKER_NAME).write_text(
        json.dumps({"family": family, "terminal_cursor": "2026-06-30T20:00:00+00:00", "terminal_validation_state": "COMPLETE"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(supervisor, "POLICY_ROOT", tmp_path)
    monkeypatch.setattr(r31, "POLICY_ROOT", tmp_path)
    monkeypatch.setattr(supervisor, "parquet_rows", lambda path: 10 if path.name != "pending_buffer.parquet" else 0)
    monkeypatch.setattr(supervisor, "family_processes", lambda: {item: [] for item in supervisor.ALL_FAMILIES})

    row = supervisor.classify_family(family, {item: [] for item in supervisor.ALL_FAMILIES})

    assert row["state"] == "COMPLETE"
    assert supervisor.next_ready_family([row, {"family": "rff_ridge", "state": "CERTIFIED_READY"}]) == "rff_ridge"


def test_active_quarantine_is_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    family = "huber"
    root = tmp_path / family
    root.mkdir(parents=True)
    (root / r31.NAMESPACE_QUARANTINE_NAME).write_text(
        json.dumps({"family": family, "status": "ACTIVE", "release_required": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(r31, "POLICY_ROOT", tmp_path)

    assert r31.active_quarantine(family)["release_required"] is True


def test_valid_quarantine_release_unblocks_same_metrics_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    family = "huber"
    root = tmp_path / family
    metrics_root = root / "metrics_only_v3_r37_huber_replay"
    metrics_root.mkdir(parents=True)
    publish_resolved_performance_contract_v3(metrics_root / "resolved_performance_contract_v3.json")
    (root / "progress.json").write_text(json.dumps({"metrics_root_name": metrics_root.name}), encoding="utf-8")
    (root / r31.NAMESPACE_QUARANTINE_NAME).write_text(
        json.dumps({"family": family, "status": "ACTIVE", "release_required": True}),
        encoding="utf-8",
    )
    (root / r31.QUARANTINE_RELEASE_NAME).write_text(
        json.dumps(
            {
                "release_id": "DS24_HUBER_R36_QUARANTINE_SCIENTIFIC_RELEASE_V1",
                "family": family,
                "status": "ACTIVE",
                "decision": "HUBER_R36_QUARANTINE_RELEASE_SCIENTIFICALLY_JUSTIFIED",
                "released_metrics_root_name": metrics_root.name,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(r31, "POLICY_ROOT", tmp_path)

    assert r31.quarantine_release_valid(family) is True
    assert r31.active_quarantine(family) == {}


def test_terminal_cursor_without_marker_enters_terminal_validating(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    family = "pca_ridge_policy_v1_control"
    root = tmp_path / family
    (root / "metrics_only").mkdir(parents=True)
    (root / "progress.json").write_text(json.dumps({"last_completed_T": "2026-06-30T20:00:00+00:00"}), encoding="utf-8")
    monkeypatch.setattr(supervisor, "POLICY_ROOT", tmp_path)
    monkeypatch.setattr(r31, "POLICY_ROOT", tmp_path)
    monkeypatch.setattr(supervisor, "parquet_log_rows", lambda root, name, legacy_path=None: 10)
    monkeypatch.setattr(supervisor, "parquet_rows", lambda path: 1 if path.name == "pending_buffer.parquet" else 10)

    row = supervisor.classify_family(family, {item: [] for item in supervisor.ALL_FAMILIES})

    assert row["state"] == "TERMINAL_VALIDATING"
    assert supervisor.active_running([row]) == []


def test_pending_outcome_reconciliation_classifies_all_terminal_reasons() -> None:
    pending = pd.DataFrame(
        {
            "family": ["ridge_policy_v1_control"] * 6,
            "decision_timestamp": [
                "2026-06-30T19:00:00Z",
                "2026-06-30T19:05:00Z",
                "2026-06-30T18:55:00Z",
                "2026-06-30T18:50:00Z",
                "2026-06-30T18:45:00Z",
                "2026-06-30T18:45:00Z",
            ],
            "asset_id": ["A", "A", "MISSING", "BADCHRON", "DUP", "DUP"],
            "prediction": [1, 1, 1, 1, 1, 2],
        }
    )
    targets = pd.DataFrame(
        {
            "asset_id": ["A", "A", "BADCHRON", "DUP"],
            "decision_timestamp": [
                "2026-06-30T19:00:00Z",
                "2026-06-30T19:05:00Z",
                "2026-06-30T18:50:00Z",
                "2026-06-30T18:45:00Z",
            ],
            "target_available_timestamp": [
                "2026-06-30T20:00:00Z",
                "2026-06-30T20:05:00Z",
                "2026-06-30T18:49:00Z",
                "2026-06-30T19:45:00Z",
            ],
            "target_is_trainable": [True, False, True, True],
            "target_value": [0.1, None, 0.2, 0.3],
        }
    )

    _, counts = r31.classify_pending_frame("ridge_policy_v1_control", pending, targets)

    assert counts["RESOLVED_MATURED"] == 1
    assert counts["RIGHT_CENSORED_AT_DATA_TERMINAL"] == 1
    assert counts["SOURCE_MISSING_BLOCKED"] == 1
    assert counts["CHRONOLOGY_INVALID"] == 1
    assert counts["DUPLICATE_INVALID"] == 2


def test_target_loader_resolves_hashed_asset_id_through_canonical_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_path = tmp_path / "targets" / "symbol=KBH" / "year=2026" / "target_rows.parquet"
    target_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "asset_id": ["asset_hash"],
            "decision_timestamp": ["2026-06-30T19:00:00Z"],
            "target_available_timestamp": ["2026-06-30T20:00:00Z"],
            "target_is_trainable": [True],
            "target_value": [0.01],
        }
    ).to_parquet(target_path, index=False)
    registry = tmp_path / "canonical_asset_registry.csv"
    registry.write_text("asset_id,canonical_symbol\nasset_hash,KBH\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "asset_id,year,target_partition\nKBH,2026,targets/symbol=KBH/year=2026/target_rows.parquet\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(r31, "ROOT", tmp_path)
    monkeypatch.setattr(r31, "CANONICAL_ASSET_REGISTRY", registry)
    monkeypatch.setattr(r31, "MODEL_PARTITION_MANIFEST", manifest)
    monkeypatch.setattr(r31, "TARGET_ROOT", tmp_path / "unused")

    targets, meta = r31.load_targets_for_keys(
        pd.DataFrame({"asset_id": ["asset_hash"], "decision_timestamp": ["2026-06-30T19:00:00Z"]})
    )

    assert len(targets) == 1
    assert targets.iloc[0]["target_value"] == pytest.approx(0.01)
    assert meta["target_files_read"] == 1
    assert meta["target_manifest_resolved_files"] == 1
    assert meta["target_files_missing"] == 0


def test_hac_confidence_interval_is_finite_for_rank_ic_series() -> None:
    ci = r31.hac_mean_ci([0.1, 0.2, -0.1, 0.0], lag=1)

    assert ci["observations"] == 4
    assert ci["lower"] < ci["mean"] < ci["upper"]


def test_portfolio_summary_uses_session_aggregation_and_drawdown() -> None:
    returns = pd.DataFrame(
        {
            "decision_timestamp": ["2026-01-02T14:35:00Z", "2026-01-02T14:40:00Z", "2026-01-05T14:35:00Z"],
            "gross_return": [0.02, 0.00, -0.01],
            "net_return": [0.02, 0.00, -0.01],
            "turnover": [0.0, 0.5, 0.25],
            "transaction_cost": [0.0, 0.0, 0.0],
            "top_n": [20, 20, 20],
        }
    )

    summary = r31.portfolio_summary_from_returns(returns)

    assert summary["aggregation_policy"] == "session_mean_of_overlapping_60m_forward_returns"
    assert summary["resolved_session_count"] == 2
    assert summary["maximum_drawdown"] <= 0
    assert summary["turnover"] == pytest.approx(0.25)


def test_rank_ic_not_invented_when_metrics_only_contract_did_not_retain_it() -> None:
    metrics = pd.DataFrame({"family": ["ridge"], "decision_timestamp": ["2026-01-02T14:35:00Z"], "mature_target_rows": [0]})

    result = r31.rank_ic_summary(metrics)

    assert result["status"] == "NOT_RETAINED_BY_METRICS_ONLY_CONTRACT"
    assert result["mean_spearman_rank_ic"] is None


def test_resource_gate_counts_complete_live_process_until_contained(monkeypatch: pytest.MonkeyPatch) -> None:
    board = [{"family": "ridge_policy_v1_control", "state": "COMPLETE", "pid_alive": True, "working_set": 1, "private_memory": 1, "heavy": False, "duplicate_worker_count": 0}]
    monkeypatch.setattr(supervisor, "system_snapshot", lambda: {"available_ram_bytes": 16 * 1024**3, "disk_free_bytes": 20 * 1024**3, "system_commit_percent": 10})
    monkeypatch.setattr(supervisor, "exact_manifest", lambda: {"pid_alive": False, "terminal_complete": True, "working_set": 0, "private_memory": 0})
    monkeypatch.setattr(supervisor, "zero_full_prediction_guard", lambda: {"paper_orders": 0, "live_orders": 0, "holdout_accessed": False, "full_prediction_files_in_metrics_namespaces": 0})
    monkeypatch.setattr(supervisor, "disallowed_process_manifest", lambda: {"ds26_workers": [], "r20_compactor": []})

    gate = supervisor.resource_gate(board, supervisor.GateConfig(max_active_model_processes=1, max_policy_workers=1))

    assert gate["active_model_processes"] == 1
    assert "MAX_ACTIVE_MODEL_PROCESSES" in gate["blocked_reasons"]
