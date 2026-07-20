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


def test_final_schema_plan_distinguishes_120_column_alpha_child_from_182_final(
    tmp_path, monkeypatch
):
    from core.research.ml.stock_level.stock_fundamentals import (
        FUNDAMENTAL_FEATURE_COLUMNS,
        FUNDAMENTAL_METADATA_COLUMNS,
    )
    from infrastructure.data.canonical_v2_alpha_enrichment import (
        ALPHA_OUTPUT_SCHEMA,
        _time_series_features,
    )

    missing_spine = {
        "asset_id",
        "canonical_symbol",
        "model_close",
        "source_provider",
        "compatibility_tier",
        "eligibility_reason",
        "selector_eligible",
        "provider_transition_flag",
        "provider_transition_id",
    }
    missing_fundamentals = set(FUNDAMENTAL_FEATURE_COLUMNS) | set(
        FUNDAMENTAL_METADATA_COLUMNS
    )
    expected_missing = missing_spine | missing_fundamentals
    producer = set(_time_series_features([], []))
    base_columns = set(ALPHA_OUTPUT_SCHEMA) - producer - expected_missing
    assert len(base_columns) == 70
    base = tmp_path / "physical_base.parquet"
    pq.write_table(
        pa.table({column: pa.array([None]) for column in sorted(base_columns)}),
        base,
    )
    labeled = tmp_path / "labeled" / "symbol=A" / "spine.parquet"
    labeled.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "session_date": ["2026-01-02"],
                **{
                    column: pa.array([None])
                    for column in sorted(missing_spine - {"model_close"})
                },
                "model_close": [100.0],
            }
        ),
        labeled,
    )
    inputs = ParentChainInputs(
        run_id="ownership",
        output_root=tmp_path / "out",
        base_artifact=base,
        base_manifest=tmp_path / "base.json",
        canonical_daily_manifest=tmp_path / "daily.json",
        asset_registry_manifest=tmp_path / "registry.json",
        feature_schema=tmp_path / "schema.json",
        labeled_spine_root=labeled.parents[1],
    )
    plan = chain._final_schema_assembly_plan(inputs)

    assert plan["contract_decision"] == "B"
    assert plan["alpha_child_physical_column_count"] == 120
    assert plan["final_selector_parent_contract"]["column_count"] == 182
    assert plan["missing_from_alpha_child_count"] == 62
    assert set(plan["missing_from_alpha_child"]) == expected_missing
    assert plan["unresolved_column_count"] == 53
    assert {
        row["column"] for row in plan["unresolved_columns"]
    } == missing_fundamentals


def _final_assembly_fixture(tmp_path, *, future_fundamentals=False):
    import hashlib

    from core.research.ml.stock_level.stock_fundamentals import (
        FUNDAMENTAL_FEATURE_COLUMNS,
        FUNDAMENTAL_METADATA_COLUMNS,
    )
    from infrastructure.data.canonical_v2_alpha_enrichment import (
        ALPHA_OUTPUT_SCHEMA,
        _schema_for_fieldnames,
        _time_series_features,
    )

    fundamental_columns = set(FUNDAMENTAL_FEATURE_COLUMNS) | set(
        FUNDAMENTAL_METADATA_COLUMNS
    )
    spine_columns = {
        "asset_id",
        "canonical_symbol",
        "model_close",
        "source_provider",
        "compatibility_tier",
        "eligibility_reason",
        "selector_eligible",
        "provider_transition_flag",
        "provider_transition_id",
    }
    alpha_columns = set(ALPHA_OUTPUT_SCHEMA) - fundamental_columns - spine_columns
    assert len(alpha_columns) == 120
    row = {}
    for column in alpha_columns:
        kind, nullable = ALPHA_OUTPUT_SCHEMA[column]
        if column == "true_stock_level_row":
            value = True
        elif nullable:
            value = None
        elif kind == "bool":
            value = True
        elif kind == "int":
            value = 1
        elif kind == "float":
            value = 1.0
        else:
            value = "value"
        row[column] = value
    row.update(
        {
            "symbol": "A",
            "rebalance_date": "2026-01-02",
            "decision_timestamp": "2026-01-02T20:05:00Z",
            "target_provenance_contract_version": (
                "stock_level_target_provenance_v2"
            ),
        }
    )
    alpha = tmp_path / "alpha.parquet"
    schema = _schema_for_fieldnames(
        [name for name in ALPHA_OUTPUT_SCHEMA if name in alpha_columns]
    )
    pq.write_table(pa.Table.from_pylist([row], schema=schema), alpha)

    spine = tmp_path / "spines" / "symbol=A" / "spine.parquet"
    spine.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "asset_id": "asset-a",
                    "canonical_symbol": "A",
                    "session_date": "2026-01-02",
                    "model_close": 100.0,
                    "source_provider": "canonical",
                    "compatibility_tier": "TIER_A",
                    "eligibility_reason": "eligible",
                    "selector_eligible": True,
                    "provider_transition_flag": False,
                    "provider_transition_id": "",
                }
            ]
        ),
        spine,
    )
    fundamental_row = {
        column: None for column in fundamental_columns
    }
    fundamental_row.update(
        {
            "asset_id": "asset-a",
            "rebalance_date": "2026-01-02",
            "fundamentals_available_timestamp": (
                "2026-01-03T20:05:00Z"
                if future_fundamentals
                else "2026-01-02T19:00:00Z"
            ),
        }
    )
    fundamentals = tmp_path / "fundamentals.parquet"
    pq.write_table(pa.Table.from_pylist([fundamental_row]), fundamentals)
    key = hashlib.sha256(b"2026-01-02\x1fA\n").hexdigest()
    return alpha, spine.parents[1], fundamentals, {
        "row_count": 1,
        "economic_key_sha256": key,
    }


def test_final_assembly_is_pit_safe_row_preserving_and_deterministic(tmp_path):
    from infrastructure.data.canonical_v2_alpha_enrichment import ALPHA_OUTPUT_SCHEMA

    alpha, spine_root, fundamentals, expected = _final_assembly_fixture(tmp_path)
    output = tmp_path / "selector_parent.parquet"
    result = chain._assemble_final_selector_parent(
        alpha_child=alpha,
        labeled_spine_root=spine_root,
        fundamentals_artifact=fundamentals,
        output_path=output,
        expected_base=expected,
    )

    assert result["status"] == "COMPLETE"
    assert result["row_count"] == 1
    assert result["column_count"] == 182
    assert pq.ParquetFile(output).schema_arrow.names == list(ALPHA_OUTPUT_SCHEMA)


def test_final_assembly_rejects_future_fundamentals(tmp_path):
    alpha, spine_root, fundamentals, expected = _final_assembly_fixture(
        tmp_path, future_fundamentals=True
    )
    with pytest.raises(ValueError, match="future PIT fundamentals"):
        chain._assemble_final_selector_parent(
            alpha_child=alpha,
            labeled_spine_root=spine_root,
            fundamentals_artifact=fundamentals,
            output_path=tmp_path / "selector_parent.parquet",
            expected_base=expected,
        )
    assert not (tmp_path / "selector_parent.parquet").exists()


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


def test_production_build_blocks_when_final_schema_producer_is_unresolved(
    tmp_path, monkeypatch
):
    inputs = _fixture(tmp_path, monkeypatch)
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    labeled_root = tmp_path / "labeled"
    labeled_root.mkdir()
    spine_path = labeled_root / "symbol=A" / "spine.parquet"
    spine_path.parent.mkdir()
    pq.write_table(
        pa.table(
            {
                "asset_id": ["asset-a"],
                "canonical_symbol": ["A"],
                "session_date": ["2026-01-02"],
                "model_close": [100.0],
                "source_provider": ["canonical"],
                "compatibility_tier": ["TIER_A"],
                "eligibility_reason": ["eligible"],
                "selector_eligible": [True],
                "provider_transition_flag": [False],
                "provider_transition_id": [""],
            }
        ),
        spine_path,
    )
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
    plan = prepare_parent_chain_plan(
        chain.ParentChainInputs(
            **{**inputs.__dict__, "config_path": config}
        )
    )
    assert plan["status"] == "BLOCKED"
    assert "FINAL_SCHEMA_PRODUCER_UNRESOLVED" in plan["blockers"]
    assert sum(
        row["expected_owner"] == "stock_fundamentals_pit_enrichment"
        for row in plan["final_schema_assembly"]["unresolved_columns"]
    ) == 53
    with pytest.raises(ValueError, match="FINAL_SCHEMA_PRODUCER_UNRESOLVED"):
        build_production_enriched_child(inputs, config_path=config)
    assert not inputs.output_root.exists()


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
