from __future__ import annotations

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
