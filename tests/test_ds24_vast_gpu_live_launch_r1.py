from __future__ import annotations

import copy
from pathlib import Path

import pytest

from core.research.ml.ds24 import vast_gpu_live_launch_r1 as r51
from core.research.ml.ds24 import vast_reverse_queue_r1 as reverse_queue
from core.research.ml.stock_level.stock_level_sequence_regressors import (
    SequenceRegressorConfig,
    TorchSequenceReturnRegressor,
)


ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-09-03T12:00:00Z"


def _queue_definition() -> dict:
    return r51.read_json(ROOT / r51.DEFAULT_QUEUE_AUTHORITY_ROOT_REL / "vast_reverse_queue_definition.json")


def _admission(root: Path) -> None:
    r51.write_json_atomic(root / "telemetry/gpu_admission.json", r51.synthetic_passing_gpu_admission_evidence())


def _snapshots() -> tuple[dict, dict]:
    queue_definition = _queue_definition()
    dell, mac = reverse_queue.synthetic_external_status_fixture(queue_definition, now_utc=r51.utc_now())["snapshots"]
    return dell, mac


def _write_snapshots(tmp_path: Path, dell: dict, mac: dict) -> list[Path]:
    paths = [tmp_path / "dell.json", tmp_path / "mac.json"]
    r51.write_json_atomic(paths[0], dell)
    r51.write_json_atomic(paths[1], mac)
    return paths


def _update_snapshot(snapshot: dict, family: str, **updates: object) -> dict:
    out = copy.deepcopy(snapshot)
    for row in out["families"]:
        if row["family_id"] == family:
            row.update(updates)
    out["snapshot_hash"] = reverse_queue.snapshot_hash(out)
    return out


def test_r49_family_adapter_gpu_audit_covers_all_families() -> None:
    audit = r51.family_adapter_gpu_audit(ROOT)
    assert audit["status"] == "PASS"
    assert audit["accepted_reverse_order"] == list(r51.ACCEPTED_REVERSE_ORDER)
    assert [row["family"] for row in audit["families"]] == list(r51.ACCEPTED_REVERSE_ORDER)
    sequence = [row for row in audit["families"] if row["family"] in r51.GPU_SEQUENCE_FAMILIES]
    rankers = [row for row in audit["families"] if row["family"] in r51.LIGHTGBM_RANKING_FAMILIES]
    assert all(row["device_behavior"]["cuda_required_before_queue_release"] for row in sequence)
    assert all("run-live-sequence-family" in row["real_training_path"] for row in sequence)
    assert all(row["device_behavior"]["lightgbm_gpu_preferred"] for row in rankers)
    assert all(row["device_behavior"]["lightgbm_cpu_fallback_explicit"] for row in rankers)


def test_lightgbm_gpu_policy_has_safe_cpu_fallback() -> None:
    gpu = r51.lightgbm_ranking_runtime_policy(gpu_supported=True)
    cpu = r51.lightgbm_ranking_runtime_policy(gpu_supported=False)
    assert {row["parameters"]["device_type"] for row in gpu["families"]} == {"gpu"}
    assert {row["runtime_policy"] for row in cpu["families"]} == {"CPU_FALLBACK"}
    assert all(row["safe_cpu_fallback_reason"] for row in cpu["families"])


@pytest.mark.parametrize(
    ("mutator", "failed_check"),
    [
        (lambda evidence: evidence["torch_probe"].update(cuda_available=False), "torch_cuda_available"),
        (lambda evidence: evidence["torch_probe"].update(device_name="NVIDIA A100"), "expected_rtx_gpu_identity"),
        (lambda evidence: evidence["model_process"].update(pid=999, command_marker="definitely-not-present"), "model_process_in_nvidia_smi"),
        (lambda evidence: evidence["torch_probe"].update(memory_allocated_mib=0) or [sample.update(memory_used_mib=0) for sample in evidence["nvidia_smi_samples"]], "meaningful_gpu_memory_allocation"),
        (lambda evidence: [sample.update(utilization_gpu_percent=0) for sample in evidence["nvidia_smi_samples"]], "repeated_nonzero_gpu_utilisation"),
        (lambda evidence: evidence.update(cuda_oom_observed=True), "no_cuda_oom"),
        (lambda evidence: evidence["checkpoint_resume"].update(status="FAIL"), "valid_checkpoint_resume"),
        (lambda evidence: evidence["compact_artifacts"].update(forbidden_artifact_count=1), "correct_compact_artifacts"),
    ],
)
def test_gpu_admission_validation_fails_each_required_condition(mutator, failed_check: str) -> None:
    evidence = r51.synthetic_passing_gpu_admission_evidence()
    mutator(evidence)
    result = r51.validate_gpu_admission_evidence(evidence)
    assert result["status"] == "FAIL"
    assert result["checks"][failed_check] is False


def test_gpu_admission_synthetic_fixture_passes() -> None:
    result = r51.synthetic_passing_gpu_admission_evidence()["validation"]
    assert result["status"] == "PASS"
    assert result["checks"]["torch_cuda_available"] is True
    assert result["checks"]["repeated_nonzero_gpu_utilisation"] is True


def test_family_commands_use_r51_live_runners() -> None:
    seq = r51._family_command("dlinear", ROOT, Path("/data"), Path("/run"))
    rank = r51._family_command("lightgbm_lambdarank", ROOT, Path("/data"), Path("/run"))
    assert "run-live-sequence-family" in seq
    assert "scripts/local/ds24_v3_sequence_policy_worker.py" not in seq
    assert "run-live-lightgbm-family" in rank


def test_bootstrap_script_is_jupyter_proxy_safe_and_resumable() -> None:
    script = r51.render_vast_jupyter_proxy_bootstrap()
    assert "DS24_BOOTSTRAP_COMMIT:?" in script
    assert "DS24_DELL_STATUS_SNAPSHOT_JSON_B64" in script
    assert "DS24_MAC_STATUS_SNAPSHOT_JSON_B64" in script
    assert script.index("validate-snapshot") < script.index("Downloading TradingSystemDataset44/ds24/full_data_r1")
    assert "rclone copy" in script
    assert script.index("run-gpu-admission") < script.index("run-vast-reverse-queue")
    assert "tmux new-session" in script
    assert "set +x" in script
    assert "ssh " not in script
    assert "scp " not in script
    assert "rsync" not in script


def test_missing_dell_mac_snapshots_fail_closed(tmp_path: Path) -> None:
    _admission(tmp_path)
    with pytest.raises(r51.VastGpuLiveLaunchError, match="DELL_MAC_OWNERSHIP_SNAPSHOTS_REQUIRED"):
        r51.run_vast_reverse_queue(
            repo_root=ROOT,
            dataset_root=tmp_path / "dataset",
            run_root=tmp_path,
            execute_live=False,
            confirm_token="",
        )


def test_neutral_snapshot_dry_queue_starts_at_bottom_without_execution(tmp_path: Path) -> None:
    _admission(tmp_path)
    result = r51.run_vast_reverse_queue(
        repo_root=ROOT,
        dataset_root=tmp_path / "dataset",
        run_root=tmp_path,
        execute_live=False,
        confirm_token="",
        allow_neutral_synthetic_ownership=True,
    )
    assert result["status"] == "PASS"
    assert result["launched"][0]["family"] == "temporal_fusion_transformer"
    assert result["execute_live"] is False
    assert result["external_ownership_gate"]["neutral_synthetic_ownership_used"] is True


def test_external_running_family_blocks_queue(tmp_path: Path) -> None:
    _admission(tmp_path)
    dell, mac = _snapshots()
    dell = _update_snapshot(
        dell,
        "temporal_fusion_transformer",
        family_state="RUNNING",
        ownership_state="RUNNING",
        active_owner="dell",
        pid=100,
        pid_alive=True,
    )
    paths = _write_snapshots(tmp_path, dell, mac)
    with pytest.raises(r51.VastGpuLiveLaunchError, match="FAMILY_NOT_CLAIMABLE"):
        r51.run_vast_reverse_queue(
            repo_root=ROOT,
            dataset_root=tmp_path / "dataset",
            run_root=tmp_path,
            execute_live=False,
            confirm_token="",
            external_snapshot_paths=paths,
        )


def test_external_completed_family_is_skipped(tmp_path: Path) -> None:
    _admission(tmp_path)
    dell, mac = _snapshots()
    dell = _update_snapshot(
        dell,
        "temporal_fusion_transformer",
        family_state="COMPLETE",
        ownership_state="COMPLETE",
        result_artifact_manifest_hash="verified-result-manifest",
    )
    paths = _write_snapshots(tmp_path, dell, mac)
    result = r51.run_vast_reverse_queue(
        repo_root=ROOT,
        dataset_root=tmp_path / "dataset",
        run_root=tmp_path,
        execute_live=False,
        confirm_token="",
        external_snapshot_paths=paths,
    )
    assert result["launched"][0]["family"] == "market_context_encoder"
    assert "temporal_fusion_transformer" in result["completed"]


def test_publisher_resource_gate_skips_without_rclone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "metrics_only_v3").mkdir()
    monkeypatch.setattr(
        r51,
        "current_resource_snapshot",
        lambda _root: {
            "cpu_total_percent": 99.0,
            "ram_free_gb": 1.0,
            "disk_free_gb": 1.0,
            "disk_busy_percent": 99.0,
            "publisher_backlog_gb": 0.0,
        },
    )
    result = r51.publisher_once(tmp_path, bucket="bucket", remote_prefix="prefix")
    assert result["status"] == "SKIPPED_RESOURCE_GATE"
    assert result["resource_gate"]["allowed_overlap"] is False


def test_budget_and_single_gpu_gates() -> None:
    stop = r51.budget_self_stop_guard(
        max_runtime_hours=1,
        max_estimated_cost_usd=1,
        hourly_price_usd=2,
        started_at_utc="2026-09-03T00:00:00Z",
        now_utc="2026-09-03T01:00:00Z",
    )
    busy = r51.single_gpu_family_admission(
        "dlinear",
        [{"status": "ACTIVE", "family": "patchtst", "expires_at_utc": "2026-09-03T13:00:00Z"}],
        now_utc=NOW,
    )
    assert stop["status"] == "STOP_REQUIRED"
    assert busy["status"] == "FAIL"


def test_materialized_configs_and_launchers_are_safe(tmp_path: Path) -> None:
    manifest = r51.write_materialized_live_configs(ROOT, tmp_path, bootstrap_commit="abc123")
    assert manifest["status"] == "PASS"
    for name in [
        "VAST_BOOTSTRAP_CONFIG_JSON.example.json",
        "PUBLISHER_CONFIG_JSON.example.json",
        "DELL_REPATRIATION_CONFIG_JSON.example.json",
        "b2_remote_inventory.example.json",
        "DATASET_COMPLETE.example.json",
        "dell_status_snapshot.example.json",
        "mac_status_snapshot.example.json",
        "vast_jupyter_proxy_bootstrap.sh",
        "dell_repatriate_vast_outputs.ps1",
        "vast_show_status.sh",
    ]:
        assert (tmp_path / name).exists()
    bootstrap = (tmp_path / "VAST_BOOTSTRAP_CONFIG_JSON.example.json").read_text(encoding="utf-8")
    assert r51.B2_BUCKET in bootstrap
    assert str(r51.EXPECTED_DATASET_OBJECT_COUNT) in bootstrap
    assert "<Backblaze application key>" not in bootstrap
    repatriation = (tmp_path / "dell_repatriate_vast_outputs.ps1").read_text(encoding="utf-8")
    assert "metrics_only_v3/**" in repatriation
    assert "ensemble_oof_scores_v2/**" in repatriation
    assert "*full_prediction*" in repatriation
    assert "rclone copy" in repatriation


def test_sequence_regressor_preprocess_fit_predict_cpu_smoke() -> None:
    config = SequenceRegressorConfig(
        architecture="dlinear",
        sequence_length=3,
        epochs=1,
        batch_size=2,
        torch_num_threads=1,
    )
    regressor = TorchSequenceReturnRegressor(config)
    sequences = [
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
        [[2.0, 1.0], [3.0, 2.0], [4.0, 3.0]],
        [[3.0, 2.0], [4.0, 3.0], [5.0, 4.0]],
    ]
    regressor.fit(sequences, [0.1, 0.2, 0.3])
    preds = regressor.predict(sequences[:2])
    assert len(preds) == 2
    assert regressor.feature_means is not None
    assert regressor.feature_stds is not None
