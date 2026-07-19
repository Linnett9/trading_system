from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.registries.adapters import selector_model_adapter, verify_registry_capabilities
from core.research.ml.registries.io import canonical_hash
from core.research.ml.selector_component_readiness import (
    DATES, MODELS, WAVE4_CHALLENGERS, assess_selector_component_readiness,
)
from core.research.ml.stock_level.huber_selector import huber_selector_input
from core.research.ml.stock_level.wave4_selector_integration import (
    HORIZON_TARGETS, build_multi_horizon_evidence, publish_wave4_component,
)


def _gate():
    gate = {
        "gate_contract_version": "selector_parent_publication_gate.v1",
        "status": "READY", "selector_dataset_id": "synthetic-dataset",
        "selector_dataset_artifact_checksum": "D" * 64,
        "selector_feature_schema_checksum": "F" * 64,
        "canonical_registry_id": "registry-v1", "daily_spine_id": "spine-v1",
    }
    gate["logical_checksum"] = canonical_hash(gate)
    return gate


def _huber_input():
    rows = []
    for index in range(12):
        training = index < 8
        decision = f"2024-01-{index + 1:02d}" if training else "2024-01-20"
        rows.append({
            "row_id": f"row-{index:02d}", "asset_id": f"A{index:02d}",
            "decision_timestamp": decision,
            "feature_availability_timestamp": decision,
            "feature_ids": ["feature_a", "feature_b"],
            "feature_values": [float(index), float((index % 3) - 1)],
            "target_value": 0.02 * index + 0.01 * (index % 2),
            "target_maturity_timestamp": f"2024-01-{min(index + 2, 9):02d}",
            "sample_weight": 1.0, "split": "TRAINING" if training else "VALIDATION",
        })
    return huber_selector_input(
        rows, target_horizon="10_sessions", target_contract_identity="forward_return_10d",
        feature_schema_identity="canonical_v2_daily_tabular_v1",
        training_fold_identity="train-fold", validation_fold_identity="validation-fold",
        dataset_identity="synthetic-dataset", source_population_checksum="D" * 64,
    )


def test_registry_resolves_all_wave4_models_and_targets_truthfully():
    resolver = RegistryResolver(load_registry_bundle())
    for model in WAVE4_CHALLENGERS:
        resolution = resolver.resolve("selector_models", model, role="selector")
        assert resolution.entry.payload["ordinary_runner_support"] is True
        assert selector_model_adapter(model, runner="ordinary").constructor_owner.endswith(
            "wave4_selector_integration:publish_wave4_component"
        )
    for target in HORIZON_TARGETS.values():
        assert resolver.resolve("target_contracts", target, role="selector").canonical_id == target
    assert verify_registry_capabilities()["ordinary"] >= len(WAVE4_CHALLENGERS)


def test_contextual_schema_has_exact_bounded_contract():
    payload = json.loads(Path("config/selector_features/contextual_elastic_net_v1.json").read_text())
    assert len(payload["stock_features"]) == 21
    assert len(payload["market_context_features"]) == 8
    assert payload["interactions"] == [
        ["momentum", "market_volatility"], ["momentum", "market_trend"],
        ["drawdown_recovery", "market_drawdown"],
        ["risk_adjusted_momentum", "market_volatility"],
        ["liquidity", "market_volatility"],
        ["stock_volatility", "market_volatility"],
    ]


def test_base_campaign_is_unchanged_and_challengers_are_explicit(tmp_path):
    gate_path = tmp_path / "gate.json"; gate_path.write_text(json.dumps(_gate()))
    common = {
        "parent_gate_path": gate_path, "authoritative_root": tmp_path / "components",
        "selector_dataset_root": tmp_path / "dataset", "config_path": tmp_path / "config.json",
        "approved_component_roots": (tmp_path / "components",),
    }
    base = assess_selector_component_readiness(**common)
    challenger = assess_selector_component_readiness(**common, campaign="wave4_challengers")
    assert base["required_models"] == list(MODELS)
    assert base["expected_component_count"] == 15
    assert challenger["required_models"] == list(WAVE4_CHALLENGERS)
    assert challenger["expected_component_count"] == len(DATES) * (2 + 3 * 4)
    assert all(job["model_id"] in WAVE4_CHALLENGERS for job in challenger["production_plan"])


def test_huber_authoritative_publication_is_deterministic_and_resumable(tmp_path):
    kwargs = {
        "model_id": "huber", "prediction_date": "2024-02-01",
        "fit_input": _huber_input(), "output_root": tmp_path / "components",
        "parent_gate": _gate(), "ledger_path": tmp_path / "ledger.jsonl",
        "fit_options": {"maximum_iterations": 500, "minimum_rank_diversity": 2},
    }
    first = publish_wave4_component(**kwargs)
    manifest = json.loads(Path(first["manifest_path"]).read_text())
    assert manifest["component_schema_version"] == "authoritative_selector_component_v1"
    assert manifest["validation_status"] == "VERIFIED_STRICT_OOS"
    assert manifest["economic_target_id"] == "forward_return_10d"
    assert manifest["target_provenance_contract_version"] == (
        "stock_level_target_provenance_v2"
    )
    metrics = json.loads(Path(manifest["metrics_path"]).read_text())
    assert metrics["diagnostics"]["convergence_status"] == "CONVERGED"
    assert metrics["gate_w4_evidence"]["gate_passed"] is False
    second = publish_wave4_component(**kwargs)
    assert second["status"] == "SKIPPED_COMPLETE"
    statuses = [json.loads(line)["event_status"] for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert statuses == ["STARTED", "COMPLETED", "SKIPPED_COMPLETE"]


def _fake_horizon_component(tmp_path, horizon, scores):
    owner = tmp_path / horizon; owner.mkdir()
    prediction = owner / "predictions.csv"
    with prediction.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "selector_score"])
        writer.writeheader()
        for index, score in enumerate(scores): writer.writerow({"row_id": f"r{index}", "selector_score": score})
    manifest = {
        "horizon_id": horizon, "prediction_artifact_path": str(prediction),
        "manifest_checksum": canonical_hash({"horizon": horizon, "scores": scores}),
    }
    path = owner / "manifest.json"; path.write_text(json.dumps(manifest))
    return path


def test_multi_horizon_evidence_is_deterministic_and_missing_horizon_fails(tmp_path):
    paths = [
        _fake_horizon_component(tmp_path, horizon, [index + 0.1, -index - 0.2, index + 0.3])
        for index, horizon in enumerate(("return_1s", "return_5s", "return_10s", "return_20s"))
    ]
    first = build_multi_horizon_evidence(paths)
    second = build_multi_horizon_evidence(list(reversed(paths)))
    assert first["logical_checksum"] == second["logical_checksum"]
    assert first["persistence_equation"].startswith("0.5*sign_agreement")
    assert all(row["persistence_score"] is not None for row in first["rows"])
    assert all(row["horizon_disagreement"] is not None for row in first["rows"])
    with pytest.raises(ValueError, match="four horizon"):
        build_multi_horizon_evidence(paths[:-1])


def test_integration_module_has_no_legacy_replay_exposure_or_pyarrow_import():
    source = Path("core/research/ml/stock_level/wave4_selector_integration.py").read_text()
    for forbidden in (
        "stock_level_benchmark_execution", "stock_level_model_ranking_benchmark",
        "portfolio_replay", "policy_sweep", "allocation.exposures", "pyarrow",
    ):
        assert forbidden not in source
