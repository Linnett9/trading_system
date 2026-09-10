from __future__ import annotations

import argparse

import pandas as pd
import pytest

from scripts.local import ds24_v3_sequence_policy_worker as worker


def test_registry_resolution_for_all_six_families() -> None:
    for family in worker.SUPPORTED_FAMILIES:
        spec = worker.resolve_family_spec(family, threads=1, cache_budget_gb=1.0)

        assert spec.family == family
        assert spec.implementation_owner
        assert spec.sequence_length >= 2
        assert spec.config.device == "cpu"
        assert spec.config.torch_num_threads == 1
        assert spec.loss_function == "SmoothL1Loss"
        assert spec.optimizer["name"] == "AdamW"
        assert spec.configuration_hash
        assert spec.selector_registry_hash


@pytest.mark.parametrize("family", worker.SUPPORTED_FAMILIES)
def test_sequence_shapes_masks_and_one_update(family: str) -> None:
    spec = worker.resolve_family_spec(family)
    proof = worker.train_one_bounded_update(spec)

    assert proof["status"] == "PASS"
    assert proof["input_shape"] == [
        worker.SYNTHETIC_ASSET_COUNT,
        spec.sequence_length,
        len(worker.SYNTHETIC_FEATURES),
    ]
    assert proof["orientation"] == "[batch, sequence_length, feature_or_channel]"
    assert proof["padding_mask_shape"] == [worker.SYNTHETIC_ASSET_COUNT, spec.sequence_length]
    assert proof["output_score_shape"] == [worker.SYNTHETIC_ASSET_COUNT]
    assert proof["scores_finite"] is True
    assert proof["loss_finite"] is True
    assert proof["one_backward_update"] is True


@pytest.mark.parametrize("family", worker.SUPPORTED_FAMILIES)
def test_pit_chronology_target_maturity_and_refit_policy(family: str) -> None:
    spec = worker.resolve_family_spec(family)
    proof = worker.chronology_proof(spec)

    assert proof["status"] == "PASS"
    assert proof["features_available_before_decision"] is True
    assert proof["training_targets_mature_by_refit_cutoff"] is True
    assert proof["never_crosses_holdout_boundary"] is True


@pytest.mark.parametrize("family", worker.SUPPORTED_FAMILIES)
def test_synthetic_checkpoint_resume_v3_commit_and_guards(family: str, tmp_path) -> None:
    result = worker.run_synthetic_contract(
        family,
        tmp_path,
        resume_generation=1,
        threads=1,
        cache_budget_gb=1.0,
        prediction_batch_decisions=2,
    )

    assert result["state"] == "V3_SEQUENCE_WORKER_CERTIFIED_READY"
    assert result["checkpoint_resume_result"]["status"] == "PASS"
    assert result["checkpoint_resume_result"]["deterministic_resumed_predictions"] is True
    assert result["checkpoint_resume_result"]["configuration_hash_verified"] is True
    assert result["commit_result"]["gap_free_metric_commits"] is True
    assert result["commit_result"]["duplicate_free_metric_commits"] is True
    assert result["commit_result"]["duplicate_metric_rows_added"] == 0
    assert result["duplicate_writer_refusal"]["duplicate_writer_refused"] is True
    assert result["v3_metrics_only"]["full_prediction_files"] == 0
    assert result["v3_metrics_only"]["paper_orders"] == 0
    assert result["v3_metrics_only"]["live_orders"] == 0
    assert result["v3_metrics_only"]["holdout_accessed"] is False
    assert result["v3_metrics_only"]["rank_ic_rows"] >= 1
    assert result["ensemble_trace_compatibility"]["has_raw_score"] is True
    assert result["ensemble_trace_compatibility"]["has_percentile_rank"] is True
    assert result["guards"]["worker_launches"] == 0
    assert result["guards"]["full_training_runs"] == 0
    assert result["guards"]["live_namespaces_modified"] is False


def test_resource_preflight_fails_before_allocation() -> None:
    spec = worker.resolve_family_spec("patchtst", cache_budget_gb=1.0)

    with pytest.raises(worker.ResourcePreflightError, match="INSUFFICIENT_RAM"):
        worker.resource_preflight(spec, available_memory_bytes=1, free_disk_bytes=worker.MIN_FREE_BYTES + 1)


def test_worker_refuses_tft_and_lightgbm_ranking_families() -> None:
    for family in ("temporal_fusion_transformer", "tft", "lightgbm_rank_xendcg", "lightgbm_lambdarank"):
        with pytest.raises(worker.UnsupportedFamilyError):
            worker.resolve_family_spec(family)


def test_supported_cli_contract_is_exposed() -> None:
    for option in (
        "--family",
        "--resume",
        "--resume-generation",
        "--threads",
        "--cache-budget-gb",
        "--prediction-batch-decisions",
        "--evaluation-version",
        "--metrics-root-name",
        "--refit-policy",
    ):
        assert option in worker.CLI_OPTIONS


def test_readiness_manifest_certifies_all_six_and_refuses_excluded(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "R7_LOW_USAGE_V3_SEQUENCE_WORKER_READINESS.json"

    manifest = worker.publish_readiness_manifest(manifest_path)

    assert manifest["success"] is True
    assert manifest["worker_launches"] == 0
    assert manifest["live_namespaces_modified"] is False
    assert set(manifest["family_states"]) == set(worker.SUPPORTED_FAMILIES)
    assert set(manifest["family_states"].values()) == {"V3_SEQUENCE_WORKER_CERTIFIED_READY"}
    assert "temporal_fusion_transformer" in manifest["refused_families"]
    assert "lightgbm_rank_xendcg" in manifest["refused_families"]
    assert manifest_path.exists()


def test_supervised_patchtst_routes_to_real_historical_path(monkeypatch) -> None:
    calls: dict[str, int] = {"real": 0, "synthetic": 0}

    def fake_real(args: argparse.Namespace, family: str) -> dict[str, object]:
        calls["real"] += 1
        assert family == "patchtst"
        return {"classification": "PATCHTST_REAL_HISTORICAL_EXECUTION_BOUNDED_PROOF_COMPLETE"}

    def fake_synthetic(*args, **kwargs):  # noqa: ANN002, ANN003
        calls["synthetic"] += 1
        raise AssertionError("supervised production mode must not use synthetic certification")

    monkeypatch.setattr(worker, "run_real_historical_sequence_worker", fake_real)
    monkeypatch.setattr(worker, "run_synthetic_contract", fake_synthetic)

    result = worker.run_supervised_sequence_worker(
        worker.parse_args(["--family", "PatchTST", "--metrics-root-name", "metrics_only_v3_r52_test"]),
        "patchtst",
    )

    assert result == 0
    assert calls == {"real": 1, "synthetic": 0}


def test_synthetic_certify_mode_remains_isolated(monkeypatch, tmp_path) -> None:
    calls: dict[str, int] = {"real": 0, "synthetic": 0}

    def fake_real(*args, **kwargs):  # noqa: ANN002, ANN003
        calls["real"] += 1
        raise AssertionError("synthetic certification mode must not run historical worker")

    def fake_synthetic(*args, **kwargs):  # noqa: ANN002, ANN003
        calls["synthetic"] += 1
        return {"family": "patchtst", "state": "V3_SEQUENCE_WORKER_CERTIFIED_READY", "checkpoint_resume_result": {"status": "PASS"}}

    monkeypatch.setattr(worker, "run_real_historical_sequence_worker", fake_real)
    monkeypatch.setattr(worker, "run_synthetic_contract", fake_synthetic)

    result = worker.main(
        [
            "--family",
            "PatchTST",
            "--synthetic-certify",
            "--output-root",
            str(tmp_path),
            "--metrics-root-name",
            "metrics_only_v3_r52_synthetic",
        ]
    )

    assert result == 0
    assert calls == {"real": 0, "synthetic": 1}


def test_real_sequence_examples_use_chronological_windows_only() -> None:
    rows = []
    for asset in ("A", "B"):
        for index in range(5):
            rows.append(
                {
                    "asset_id": asset,
                    "decision_timestamp": pd.Timestamp("2020-01-02T14:30:00Z") + pd.Timedelta(minutes=5 * index),
                    "f1": float(index),
                    "f2": float(index + 10),
                    "target_value": float(index) / 100.0,
                }
            )
    panel = pd.DataFrame(rows)
    eligible = panel["decision_timestamp"] >= pd.Timestamp("2020-01-02T14:40:00Z")

    sequences, targets, metadata = worker.sequence_examples_from_panel(
        panel,
        ["f1", "f2"],
        sequence_length=3,
        eligible_mask=eligible,
        include_targets=True,
    )

    assert len(sequences) == 6
    assert len(targets) == 6
    assert metadata["decision_timestamp"].min() == "2020-01-02T14:40:00+00:00"
    assert sequences[0] == [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]]


def test_real_sequence_examples_apply_construction_cap() -> None:
    panel = pd.DataFrame(
        [
            {
                "asset_id": "A",
                "decision_timestamp": pd.Timestamp("2020-01-02T14:30:00Z") + pd.Timedelta(minutes=5 * index),
                "f1": float(index),
                "target_value": float(index) / 100.0,
            }
            for index in range(8)
        ]
    )

    sequences, targets, metadata = worker.sequence_examples_from_panel(
        panel,
        ["f1"],
        sequence_length=3,
        eligible_mask=pd.Series([True] * len(panel)),
        include_targets=True,
        max_examples=2,
    )

    assert len(sequences) == 2
    assert len(targets) == 2
    assert metadata["decision_timestamp"].tolist() == [
        "2020-01-02T15:00:00+00:00",
        "2020-01-02T15:05:00+00:00",
    ]


def test_lazy_daily_sequence_schedule_matches_canonical_prefix() -> None:
    spine = []
    for day_index in range(24):
        day = pd.Timestamp("2020-01-02T14:30:00Z") + pd.Timedelta(days=day_index)
        spine.extend([day, day + pd.Timedelta(minutes=5)])

    expected = worker.canonical_worker.build_daily_session_refit_schedule(spine, max_refits=3)
    actual = list(worker.iter_daily_session_refit_schedule(spine))[:3]

    assert [(item.ordinal, item.refit_T, item.training_session_dates, item.score_session_dates) for item in actual] == [
        (item.ordinal, item.refit_T, item.training_session_dates, item.score_session_dates) for item in expected
    ]


def test_canonical_sequence_progress_schema_contains_required_fields() -> None:
    payload = worker.canonical_sequence_progress_payload(
        family="patchtst",
        display_family="PatchTST",
        pid=123,
        package_state="RUNNING",
        resume_generation="7",
        metrics_root_name="metrics_only_v3_r52",
        current_refit_session="2020-01-02T14:30:00+00:00",
        current_scoring_cursor="2020-01-02T14:35:00+00:00",
        last_committed_decision_timestamp="2020-01-02T14:35:00+00:00",
        metrics_rows=1,
        resolved_performance_rows=1,
        partition_count=1,
    )

    for field in (
        "pid",
        "family",
        "package_state",
        "heartbeat",
        "metrics_rows",
        "resolved_performance_rows",
        "current_refit_session",
        "current_scoring_cursor",
        "partition_count",
        "resume_generation",
        "last_committed_decision_timestamp",
    ):
        assert field in payload
    assert payload["execution_mode"] == "REAL_DS24_HISTORICAL_WALK_FORWARD"
    assert payload["full_prediction_files"] == 0


def test_sequence_full_prediction_guard_ignores_metrics_only_files(tmp_path) -> None:
    metrics = tmp_path / "PatchTST" / "metrics_only_v3" / "pending_scores_v3.parquet"
    metrics.parent.mkdir(parents=True)
    metrics.write_bytes(b"metrics-only")
    metrics_audit = tmp_path / "PatchTST" / "metrics_only_v3_r40_patchtst" / "prediction_audit.json"
    metrics_audit.parent.mkdir(parents=True)
    metrics_audit.write_bytes(b"metrics-only")
    assert worker.sequence_worker_full_prediction_files(tmp_path / "PatchTST") == 0

    prediction = tmp_path / "PatchTST" / "predictions" / "full_prediction.parquet"
    prediction.parent.mkdir(parents=True)
    prediction.write_bytes(b"not-allowed")
    assert worker.sequence_worker_full_prediction_files(tmp_path / "PatchTST") == 1
