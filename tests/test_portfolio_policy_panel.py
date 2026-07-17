from __future__ import annotations

import json

import pytest

from core.research.ml.portfolio_policy_panel import (
    POLICY_IDS, build_policy_panel, clip_trades_to_adv_capacity,
    daily_top_k_target_weights, hysteresis_target_weights, policy_checksum,
    staggered_cohort_target_weights, transaction_cost,
    turnover_limited_aim_weights, validate_replay_lineage,
)


def test_exact_six_registered_and_hashes_are_deterministic(tmp_path):
    first = build_policy_panel(source_commit="abc", output_path=tmp_path / "panel.json")
    second = build_policy_panel(source_commit="abc")
    assert first == second
    assert tuple(row["policy_id"] for row in first["policies"]) == POLICY_IDS
    assert len({row["policy_checksum"] for row in first["policies"]}) == 6


@pytest.mark.parametrize("field,value", [
    ("cost_bps", 25), ("adv_participation_limit", 0.05), ("selection_size", 19),
])
def test_material_policy_changes_change_identity(field, value):
    policy = build_policy_panel(source_commit="abc")["policies"][0]["policy"]
    changed = {**policy, field: value}
    assert policy_checksum(policy) != policy_checksum(changed)


def test_daily_top20_equal_weights():
    weights = daily_top_k_target_weights([f"S{i:02}" for i in range(30)], k=20)
    assert len(weights) == 20
    assert set(weights.values()) == {0.05}
    assert sum(weights.values()) == pytest.approx(1.0)


def test_staggered_cohorts_allocate_tenths_and_expire():
    cohorts = [{"entry_session": index, "symbols": [f"S{index}"]} for index in range(11)]
    active = staggered_cohort_target_weights(cohorts, current_session=10)
    assert len(active) == 10 and set(active.values()) == {0.1}
    assert "S0" not in active and sum(active.values()) == pytest.approx(1.0)


def test_top10_top20_top40_are_distinct():
    symbols = [f"S{i}" for i in range(50)]
    assert [len(daily_top_k_target_weights(symbols, k=k)) for k in (10, 20, 40)] == [10, 20, 40]


def test_hysteresis_entry_retention_and_exit():
    ranks = {"NEW25": 25, "HELD25": 25, "HELD31": 31, "NEW10": 10}
    weights = hysteresis_target_weights(ranks, {"HELD25": 0.5, "HELD31": 0.5})
    assert set(weights) == {"HELD25", "NEW10"}


def test_aim_stock_sector_and_turnover_caps_bind():
    alpha = {"A": 4, "B": 3, "C": 2, "D": 1}
    covariance = {symbol: {symbol: 1.0} for symbol in alpha}
    result = turnover_limited_aim_weights(
        alpha, covariance, {}, {"A": "X", "B": "X", "C": "Y", "D": "Y"},
        transaction_cost_penalty=0.01, maximum_stock_weight=0.20,
        maximum_sector_weight=0.25, maximum_turnover=0.30,
        liquidity_eligible={symbol: True for symbol in alpha},
    )
    assert max(result.values()) <= 0.20
    assert result["A"] + result["B"] <= 0.25
    assert sum(result.values()) <= 0.30 + 1e-12


def test_adv_clipping_and_transaction_cost_known_examples():
    assert clip_trades_to_adv_capacity(
        {"A": 1000, "B": -500}, {"A": 10_000, "B": 10_000},
        participation_limit=0.025,
    ) == {"A": 250.0, "B": -250.0}
    assert transaction_cost(0.5, 10, portfolio_value=1_000_000) == 500.0


def _lineage(**updates):
    value = {
        "wave4_gate_status": "READY_FOR_PORTFOLIO_REPLAY",
        "component_plan_checksum": "plan", "campaign_id": "campaign",
        "experiment_ledger_checksum": "ledger",
        "strict_oos_verification": "VERIFIED_STRICT_OOS",
        "dataset_id": "dataset", "daily_spine_id": "spine",
        "symbol_registry_id": "registry", "feature_schema_hash": "feature",
        "target_contract_hash": "target",
        "target_provenance_contract_version": "stock_level_target_provenance_v2",
        "model_id": "ridge", "decision_date": "2026-03-16",
        "source_commit": "abc", "complete_finite_prediction_coverage": True,
        "experiment_ledger_identity_matches": True,
    }
    value.update(updates)
    return value


@pytest.mark.parametrize("updates,match", [
    ({"strict_oos_verification": None}, "Strict-OOS"),
    ({"wave4_gate_status": "REJECTED"}, "not READY"),
    ({"target_provenance_contract_version": "stock_level_target_provenance_v1"}, "provenance v2"),
    ({"experiment_ledger_identity_matches": False}, "ledger"),
])
def test_lineage_blocks_incompatible_evidence(updates, match):
    with pytest.raises(ValueError, match=match):
        validate_replay_lineage(_lineage(**updates))


def test_atomic_write_failure_preserves_previous_manifest(tmp_path, monkeypatch):
    path = tmp_path / "panel.json"
    build_policy_panel(source_commit="abc", output_path=path)
    before = path.read_bytes()
    def fail(*_args):
        raise OSError("synthetic")
    monkeypatch.setattr("core.research.ml.portfolio_policy_panel.os.replace", fail)
    with pytest.raises(OSError, match="synthetic"):
        build_policy_panel(source_commit="def", output_path=path)
    assert path.read_bytes() == before


def test_module_has_no_execution_imports():
    source = open("core/research/ml/portfolio_policy_panel.py", encoding="utf-8").read().lower()
    for forbidden in ("subprocess", "portfolio_replay", "policy_sweep", "exposure", "news", "five_minute"):
        assert f"import {forbidden}" not in source
