from __future__ import annotations

import json
from copy import deepcopy

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import core.research.ml.stock_level.selector_parent_chain as chain
from core.research.ml.stock_level.selector_parent_chain import (
    ParentChainInputs,
    build_production_enriched_child,
    prepare_parent_chain_plan,
    publish_bounded_smoke,
)


def _row(symbol="A", date="2026-01-02"):
    return {
        "symbol": symbol,
        "rebalance_date": date,
        "decision_session_date": date,
        "decision_timestamp": f"{date}T21:05:00Z",
        "feature_data_cutoff_timestamp": f"{date}T21:00:00Z",
        "target_provenance_contract_version": "stock_level_target_provenance_v2",
        "actual_forward_return_10d": 0.1,
        "actual_benchmark_return_10d": 0.02,
        "actual_market_residual_return_10d": 0.08,
        "target_status": "realized",
        "target_start_timestamp": f"{date}T21:00:00Z",
        "label_start_timestamp": "2026-01-05T21:00:00Z",
        "label_end_timestamp": "2026-01-16T21:00:00Z",
        "label_available_timestamp": "2026-01-20T21:00:00Z",
        "benchmark_target_start_timestamp": f"{date}T21:00:00Z",
        "benchmark_label_start_timestamp": "2026-01-05T21:00:00Z",
        "benchmark_label_end_timestamp": "2026-01-16T21:00:00Z",
        "benchmark_label_available_timestamp": "2026-01-20T21:00:00Z",
    }


def _fixture(tmp_path, monkeypatch):
    base = tmp_path / "base.parquet"
    rows = [_row("A"), _row("B"), _row("A", "2026-01-05"), _row("B", "2026-01-05")]
    pq.write_table(pa.Table.from_pylist(rows), base)
    base_hash = chain._sha256(base)
    monkeypatch.setattr(chain, "APPROVED_BASE_SHA256", base_hash)
    monkeypatch.setattr(chain, "PRODUCTION_ROW_COUNT", len(rows))
    base_manifest = tmp_path / "base.json"
    base_manifest.write_text(json.dumps({
        "economic_target_id": "forward_return_10d",
        "economic_key_sha256": "e" * 64,
        "target_provenance_contract_version": "stock_level_target_provenance_v2",
        "canonical_artifact": {
            "logical_content_sha256": chain.APPROVED_BASE_LOGICAL_HASH,
        },
    }))
    daily = tmp_path / "daily.json"
    daily.write_text(json.dumps({
        "dataset_logical_partition_hash": chain.APPROVED_CANONICAL_DAILY_HASH,
    }))
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({
        "version": chain.APPROVED_ASSET_REGISTRY_VERSION,
        "row_identity_checksum": chain.APPROVED_ASSET_REGISTRY_CHECKSUM,
    }))
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"feature_columns": ["smoke_feature"]}))
    inputs = ParentChainInputs(
        run_id="smoke-1",
        output_root=tmp_path / "run=smoke-1",
        base_artifact=base,
        base_manifest=base_manifest,
        canonical_daily_manifest=daily,
        asset_registry_manifest=registry,
        feature_schema=schema,
        path_length_limit=1_000,
    )
    return inputs


def test_preflight_is_deterministic_and_read_only(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    first = prepare_parent_chain_plan(inputs)
    second = prepare_parent_chain_plan(inputs)
    assert first == second
    assert first["status"] == "READY"
    assert first["mutation_performed"] is False
    assert not inputs.output_root.exists()


def test_parent_preflight_plans_authoritative_bounded_namespaces(
    tmp_path, monkeypatch
):
    inputs = _fixture(tmp_path, monkeypatch)

    plan = prepare_parent_chain_plan(inputs)
    namespaces = plan["alpha_namespaces"]

    assert namespaces["layout"] == "bounded_v1"
    assert namespaces["owner"].endswith("_planned_bounded_alpha_namespaces")
    for namespace in (namespaces["base"], namespaces["partitions"]):
        path = chain.Path(namespace["path"])
        assert path.name == f"id-{namespace['namespace_key']}"
        assert len(namespace["namespace_key"]) == 20
    assert len(namespaces["base"]["source_base_sha256"]) == 64
    assert len(namespaces["partitions"]["base_artifact_sha256"]) == 64
    parts = chain.Path(namespaces["partitions"]["path"]).parts
    assert not any(len(part) == 64 for part in parts)


def test_parent_path_budget_blocks_before_mutation_or_worker_submission(
    tmp_path, monkeypatch
):
    inputs = _fixture(tmp_path, monkeypatch)
    blocked = chain.ParentChainInputs(
        **{
            **inputs.__dict__,
            "production": True,
            "canonical_daily_root": tmp_path,
            "path_length_limit": 80,
        }
    )
    config = tmp_path / "config.yaml"
    config.write_text("ml: {}\n")
    called = False

    import infrastructure.data.canonical_v2_alpha_enrichment as enrichment

    def forbidden(_payload):
        nonlocal called
        called = True
        raise AssertionError("worker owner must not be called")

    monkeypatch.setattr(
        enrichment, "write_partitioned_canonical_v2_alpha_features", forbidden
    )

    plan = prepare_parent_chain_plan(blocked)
    assert plan["path_budget"]["blocker"] == "PATH_LENGTH_BUDGET_EXCEEDED"
    assert plan["path_budget"]["calculated_maximum_path_length"] > 80
    assert plan["path_budget"]["longest_representative_path"]
    assert plan["path_budget"]["recommended_shorter_output_root"]
    assert not blocked.output_root.exists()
    with pytest.raises(ValueError, match="PATH_LENGTH_BUDGET_EXCEEDED"):
        build_production_enriched_child(blocked, config_path=config)
    assert called is False
    assert not blocked.output_root.exists()


def test_parent_safe_path_budget_proceeds(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)

    plan = prepare_parent_chain_plan(inputs)

    assert plan["path_budget"]["status"] == "READY"
    assert (
        plan["path_budget"]["calculated_maximum_path_length"]
        <= plan["path_budget"]["configured_limit"]
    )


def test_parent_chain_module_entry_point_preflights(
    tmp_path, monkeypatch, capsys
):
    inputs = _fixture(tmp_path, monkeypatch)

    status = chain.main(
        [
            "preflight",
            "--run-id",
            inputs.run_id,
            "--output-root",
            str(inputs.output_root),
            "--base-artifact",
            str(inputs.base_artifact),
            "--base-manifest",
            str(inputs.base_manifest),
            "--canonical-daily-manifest",
            str(inputs.canonical_daily_manifest),
            "--asset-registry-manifest",
            str(inputs.asset_registry_manifest),
            "--feature-schema",
            str(inputs.feature_schema),
            "--path-length-limit",
            "1000",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert payload["status"] == "READY"
    assert payload["mutation_performed"] is False


@pytest.mark.parametrize(
    ("attribute", "value", "blocker"),
    [
        ("APPROVED_BASE_SHA256", "0" * 64, "BASE_HASH_MISMATCH"),
        ("APPROVED_BASE_LOGICAL_HASH", "1" * 64, "BASE_LOGICAL_HASH_MISMATCH"),
        ("PRODUCTION_ROW_COUNT", 99, "BASE_POPULATION_MISMATCH"),
        ("APPROVED_CANONICAL_DAILY_HASH", "2" * 64, "CANONICAL_DAILY_HASH_MISMATCH"),
    ],
)
def test_parent_identity_failures(tmp_path, monkeypatch, attribute, value, blocker):
    inputs = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(chain, attribute, value)
    assert blocker in prepare_parent_chain_plan(inputs)["blockers"]


@pytest.mark.parametrize("provenance", ["stock_level_target_provenance_v1", "stock_level_target_provenance_v4", ""])
def test_unsupported_provenance_fails(tmp_path, monkeypatch, provenance):
    inputs = _fixture(tmp_path, monkeypatch)
    payload = json.loads(inputs.base_manifest.read_text())
    payload["target_provenance_contract_version"] = provenance
    inputs.base_manifest.write_text(json.dumps(payload))
    assert "TARGET_IDENTITY_MISMATCH" in prepare_parent_chain_plan(inputs)["blockers"]


def test_namespace_confusion_fails(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    payload = json.loads(inputs.base_manifest.read_text())
    payload["economic_target_id"] = "stock_level_target_provenance_v2"
    payload["target_provenance_contract_version"] = "forward_return_10d"
    inputs.base_manifest.write_text(json.dumps(payload))
    assert "TARGET_IDENTITY_MISMATCH" in prepare_parent_chain_plan(inputs)["blockers"]


def test_feature_target_and_memory_fail_closed(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    inputs.feature_schema.write_text(json.dumps({"feature_columns": ["actual_forward_return_10d"]}))
    bad = chain.ParentChainInputs(**{**inputs.__dict__, "memory_budget_gib": 31})
    blockers = prepare_parent_chain_plan(bad)["blockers"]
    assert "TARGET_COLUMN_IN_FEATURE_ALLOWLIST" in blockers
    assert "MEMORY_BUDGET_INVALID" in blockers


def test_object_list_feature_schema_is_resolved_and_checked(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    inputs.feature_schema.write_text(json.dumps({
        "features": [{"name": "actual_forward_return_10d"}],
    }))
    blockers = prepare_parent_chain_plan(inputs)["blockers"]
    assert "TARGET_COLUMN_IN_FEATURE_ALLOWLIST" in blockers
    assert "FEATURE_SCHEMA_UNRESOLVED" not in blockers


def test_empty_feature_schema_fails_closed(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    inputs.feature_schema.write_text(json.dumps({"features": []}))
    assert "FEATURE_SCHEMA_UNRESOLVED" in prepare_parent_chain_plan(inputs)["blockers"]


def test_output_conflict_and_concurrent_owner(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    inputs.output_root.mkdir()
    (inputs.output_root / "owner.json").write_text(json.dumps({"run_id": "other"}))
    assert "ACTIVE_OWNER_CONFLICT" in prepare_parent_chain_plan(inputs)["blockers"]


def test_active_smoke_owner_lock_fails_closed(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    lock = inputs.output_root.with_name(f".{inputs.output_root.name}.owner.lock")
    lock.write_text("other")
    with pytest.raises(ValueError, match="ACTIVE_OWNER_CONFLICT"):
        publish_bounded_smoke(inputs, decision_dates=["2026-01-02"])


def test_protected_output_root_fails(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    protected = chain.ParentChainInputs(**{
        **inputs.__dict__,
        "output_root": tmp_path / "regeneration_canonical_v2" / "alpha_enrichment",
    })
    assert "PROTECTED_OUTPUT_ROOT" in prepare_parent_chain_plan(protected)["blockers"]


def test_smoke_population_spine_frozen_and_resume(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    first = publish_bounded_smoke(inputs, decision_dates=["2026-01-02", "2026-01-05"])
    assert first["status"] == "READY"
    assert first["row_count"] == 4
    assert first["symbol_count"] == 2
    assert first["lineage"]["status"] == "READY"
    assert first["daily_spine"]["status"] == "READY"
    assert first["frozen_preflight"]["status"] == "READY"
    assert len(first["enriched_logical_checksum"]) == 64
    assert len(first["enriched_physical_sha256"]) == 64
    assert first["benchmark"]["write_bytes"] > 0
    resumed = chain.ParentChainInputs(**{**inputs.__dict__, "resume": True})
    second = publish_bounded_smoke(resumed, decision_dates=["2026-01-02", "2026-01-05"])
    assert second["resume_action"] == "REUSED_COMPATIBLE"
    assert second["row_id_checksum"] == first["row_id_checksum"]
    assert second["target_checksum"] == first["target_checksum"]


def test_owned_incomplete_output_resumes_safely(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    inputs.output_root.mkdir()
    plan = prepare_parent_chain_plan(inputs)
    (inputs.output_root / "owner.json").write_text(json.dumps(chain._owner(plan)))
    resumed = chain.ParentChainInputs(**{**inputs.__dict__, "resume": True})
    result = publish_bounded_smoke(resumed, decision_dates=["2026-01-02"])
    assert result["resume_action"] == "RESUMED_INCOMPLETE"
    assert result["resume_history"] == [
        {"action": "RESTARTED_OWNED_INCOMPLETE_OUTPUT"}
    ]


def test_incompatible_incomplete_output_fails(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    inputs.output_root.mkdir()
    (inputs.output_root / "owner.json").write_text(json.dumps({"run_id": "other"}))
    resumed = chain.ParentChainInputs(**{**inputs.__dict__, "resume": True})
    with pytest.raises(ValueError, match="preflight blocked"):
        publish_bounded_smoke(resumed, decision_dates=["2026-01-02"])


def test_unknown_duplicate_and_changed_protected_values_fail_or_preserve(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    rows = pq.read_table(inputs.base_artifact).to_pylist()
    unknown = deepcopy(rows[0])
    unknown["symbol"] = "UNKNOWN"
    with pytest.raises(ValueError, match="unknown enrichment"):
        chain.merge_enrichment_preserving_base(rows, [unknown])
    with pytest.raises(ValueError, match="ROW_ID_DUPLICATE"):
        chain.merge_enrichment_preserving_base(rows, [rows[0], rows[0]])
    poisoned = {**rows[0], "actual_forward_return_10d": 999, "smoke_feature": 1}
    output = chain.merge_enrichment_preserving_base(rows, [poisoned])
    assert output[0]["actual_forward_return_10d"] == rows[0]["actual_forward_return_10d"]


def test_production_build_delegates_to_authoritative_owner(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    labeled_root = tmp_path / "labeled"
    labeled_root.mkdir()
    labeled_manifest = tmp_path / "labeled.json"
    labeled_manifest.write_text(json.dumps({"status": "BUILT"}))
    inference_manifest = tmp_path / "inference.json"
    inference_manifest.write_text(json.dumps({"status": "BUILT"}))
    inputs.canonical_daily_manifest.write_text(json.dumps({
        "status": "COMPLETE",
        "completed_partitions": 514,
        "dataset_logical_partition_hash": chain.APPROVED_CANONICAL_DAILY_HASH,
    }))
    inputs.canonical_daily_manifest.with_name("validation.json").write_text(
        json.dumps({"valid": True})
    )
    inputs = chain.ParentChainInputs(**{
        **inputs.__dict__,
        "production": True,
        "canonical_daily_root": canonical_root,
        "labeled_spine_root": labeled_root,
        "labeled_spine_manifest": labeled_manifest,
        "inference_spine_manifest": inference_manifest,
    })
    config = tmp_path / "config.yaml"
    config.write_text("ml: {}\n")
    observed = {}

    class Paths:
        enriched_parquet_path = tmp_path / "child.parquet"

    def build(payload):
        observed.update(payload["ml"])
        return Paths()

    import infrastructure.data.canonical_v2_alpha_enrichment as enrichment
    monkeypatch.setattr(
        enrichment, "write_partitioned_canonical_v2_alpha_features", build
    )
    result = build_production_enriched_child(inputs, config_path=config)
    assert result["status"] == "COMPLETE"
    assert observed["stock_level_base_prediction_artifacts_path"] == str(
        inputs.base_artifact.resolve()
    )
    assert observed["stock_alpha_feature_n_jobs"] == 6
    assert observed["economic_target_id"] == "forward_return_10d"
    assert observed["canonical_v2_labeled_spine_root"] == str(labeled_root.resolve())
    assert observed["canonical_v2_alpha_base_manifest_path"] == str(
        inputs.base_manifest.resolve()
    )
    assert result["alpha_namespaces"]["layout"] == "bounded_v1"
    assert (
        result["parent_lineage"]["canonical_daily_logical_checksum"]
        == chain.APPROVED_CANONICAL_DAILY_HASH
    )
    assert (
        result["parent_lineage"]["target_provenance_contract_version"]
        == "stock_level_target_provenance_v2"
    )


def test_parent_publication_runbook_requires_explicit_v2_child():
    text = (
        chain.Path("scripts/selector_parent_publication_runbook.ps1")
        .read_text(encoding="utf-8")
    )
    assert "require explicit -EnrichedArtifact and -EnrichedManifest" in text
    assert "--base-artifact" in text
    assert "--base-manifest" in text
    assert "--enriched-manifest" in text
    assert "python -m scripts.build_canonical_v2_selector_dataset" in text


def test_explicit_spine_authorities_override_relative_yaml(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    root = tmp_path / "authority" / "labeled"
    root.mkdir(parents=True)
    labeled = tmp_path / "authority" / "labeled.json"
    inference = tmp_path / "authority" / "inference.json"
    labeled.write_text('{"status":"BUILT"}')
    inference.write_text('{"status":"BUILT"}')
    config = tmp_path / "config.yaml"
    config.write_text(
        "ml:\n"
        "  canonical_v2_labeled_spine_root: relative/labeled\n"
        "  canonical_v2_labeled_spine_manifest_path: relative/labeled.json\n"
        "  canonical_v2_inference_spine_manifest_path: relative/inference.json\n"
    )
    production = chain.ParentChainInputs(**{
        **inputs.__dict__,
        "production": True,
        "canonical_daily_root": tmp_path,
        "labeled_spine_root": root,
        "labeled_spine_manifest": labeled,
        "inference_spine_manifest": inference,
    })
    adapted = chain._parent_alpha_config(
        production, base_identity={"sha256": "a" * 64}, config_path=config
    )
    assert adapted["ml"]["canonical_v2_labeled_spine_root"] == str(root.resolve())
    assert adapted["ml"]["canonical_v2_labeled_spine_manifest_path"] == str(labeled.resolve())
    assert adapted["ml"]["canonical_v2_inference_spine_manifest_path"] == str(inference.resolve())
    assert adapted["ml"]["stooq_parquet_dir"] == str(tmp_path.resolve())
    assert config.read_text().startswith("ml:\n")


def test_relative_spine_yaml_without_data_root_fails_closed(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    config = tmp_path / "config.yaml"
    config.write_text(
        "ml:\n  canonical_v2_labeled_spine_root: relative\n"
        "  canonical_v2_labeled_spine_manifest_path: labeled.json\n"
        "  canonical_v2_inference_spine_manifest_path: inference.json\n"
    )
    production = chain.ParentChainInputs(**{
        **inputs.__dict__, "production": True, "canonical_daily_root": tmp_path
    })
    with pytest.raises(ValueError, match="--data-root"):
        chain._parent_alpha_config(
            production, base_identity={"sha256": "a" * 64}, config_path=config
        )


def test_economic_key_and_authoritative_inputs_are_bound_to_plan(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    plan = prepare_parent_chain_plan(inputs)
    assert len(plan["base"]["economic_key_sha256"]) == 64
    assert len(plan["adapted_configuration_sha256"]) == 64
    changed = chain.ParentChainInputs(**{
        **inputs.__dict__, "canonical_daily_root": tmp_path / "different"
    })
    assert (
        prepare_parent_chain_plan(changed)["adapted_configuration_sha256"]
        != plan["adapted_configuration_sha256"]
    )


def _alpha_resolution_config(tmp_path, base, recovered=None):
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    canonical_manifest = tmp_path / "canonical.json"
    canonical_manifest.write_text(json.dumps({
        "status": "COMPLETE",
        "completed_partitions": 514,
        "dataset_logical_partition_hash": chain.APPROVED_CANONICAL_DAILY_HASH,
    }))
    canonical_manifest.with_name("validation.json").write_text('{"valid":true}')
    labeled_root = tmp_path / "labeled"
    labeled_root.mkdir()
    labeled = tmp_path / "labeled.json"
    inference = tmp_path / "inference.json"
    labeled.write_text('{"status":"BUILT"}')
    inference.write_text('{"status":"BUILT"}')
    ml = {
        "canonical_daily_v2_root": str(canonical_root),
        "canonical_daily_v2_manifest_path": str(canonical_manifest),
        "canonical_v2_labeled_spine_root": str(labeled_root),
        "canonical_v2_labeled_spine_manifest_path": str(labeled),
        "canonical_v2_inference_spine_manifest_path": str(inference),
        "stock_level_base_prediction_artifacts_path": str(base),
        "stock_selector_market_data_source": "canonical_daily_v2",
        "stooq_parquet_dir": str(canonical_root),
        "output_dir": str(tmp_path / "output"),
    }
    if recovered is not None:
        ml["canonical_v2_recovered_reference_path"] = str(recovered)
    return {"ml": ml}


def test_missing_optional_recovered_reference_is_not_a_hash_mismatch(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    import infrastructure.data.canonical_v2_alpha_enrichment as enrichment

    result = enrichment.resolve_inputs(
        _alpha_resolution_config(tmp_path, inputs.base_artifact)
    )
    assert "recovered_artifact_hash_mismatch" not in result["blocking_issues"]
    assert result["recovered_artifact_policy"]["result"] == "PASS"


def test_genuine_two_sided_recovered_hash_mismatch_has_complete_payload(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    recovered = tmp_path / "recovered.parquet"
    recovered.write_bytes(b"different")
    import infrastructure.data.canonical_v2_alpha_enrichment as enrichment

    result = enrichment.resolve_inputs(
        _alpha_resolution_config(tmp_path, inputs.base_artifact, recovered)
    )
    policy = result["recovered_artifact_policy"]
    assert "recovered_artifact_hash_mismatch" in result["blocking_issues"]
    assert policy["blocker"] == "RECOVERED_ARTIFACT_HASH_MISMATCH"
    assert all(policy[key] for key in (
        "expected_path", "expected_sha256", "observed_path",
        "observed_sha256", "authority_source",
    ))
