from __future__ import annotations

import numpy as np
import pandas as pd

from application.services.hf_data_cutover_retirement import _active_consumer_gate, _directory_identity
from core.research.feature_autopsy import feature_quality_rows, rank_ic_by_period, redundancy_rows
from core.research.market_data_cutover import (
    AvailabilityClass,
    FeatureFamily,
    classify_feature,
    iex_feature_bank_payload,
    iex_feature_bank_v1,
    old_feature_availability,
)


def test_iex_feature_bank_is_compact_unique_and_separates_context() -> None:
    features = iex_feature_bank_v1()

    assert len(features) == 80
    assert len({feature.name for feature in features}) == 80
    assert {feature.table for feature in features} == {
        "5m",
        "1h",
        "1d",
        "cross_sectional",
        "cross_timeframe",
        "market_state",
    }
    assert "sip01_trend_minus_pullback" in {feature.name for feature in features}
    assert all("target" not in feature.source_columns for feature in features)


def test_iex_feature_payload_hash_is_deterministic() -> None:
    left = iex_feature_bank_payload(evidence={"audit": "fixed"})
    right = iex_feature_bank_payload(evidence={"audit": "fixed"})

    assert left == right
    assert len(left["spec_sha256"]) == 64
    assert left["status"] == "FROZEN_BEFORE_IEX_TARGET_OUTCOME_ANALYSIS"


def test_taxonomy_and_repaired_availability_are_explicit() -> None:
    assert classify_feature("relative_strength_spy_60m") == FeatureFamily.MARKET_RELATIVE
    assert classify_feature("breadth_return_dispersion_60m") == FeatureFamily.BREADTH_STATE
    assert classify_feature("rsi_14_5m") == FeatureFamily.CLASSIC_TECHNICAL
    assert old_feature_availability("overnight_gap", repaired=False) == AvailabilityClass.INVALID
    assert old_feature_availability("overnight_gap", repaired=True) == AvailabilityClass.CAUSAL
    assert old_feature_availability("breadth_fraction_positive_60m", repaired=True) == AvailabilityClass.AMBIGUOUS


def test_quality_and_redundancy_diagnostics_surface_null_and_duplicate_features() -> None:
    timestamps = pd.date_range("2024-01-02 15:00", periods=20, freq="5min", tz="UTC").repeat(3)
    assets = np.tile(["A", "B", "C"], 20)
    base = np.arange(len(timestamps), dtype=float)
    frame = pd.DataFrame(
        {
            "decision_timestamp": timestamps,
            "asset_id": assets,
            "signal": base,
            "duplicate_signal": base,
            "all_null": np.nan,
        }
    )

    quality = {row["feature_name"]: row for row in feature_quality_rows(frame, ["signal", "duplicate_signal", "all_null"])}
    edges, summary = redundancy_rows(frame, ["signal", "duplicate_signal", "all_null"])

    assert quality["all_null"]["missingness_rate"] == 1.0
    assert quality["signal"]["unique_value_count"] == len(frame)
    assert edges[0]["absolute_spearman"] == 1.0
    assert summary["pair_fraction_abs_spearman_gt_0_99"] == 1.0


def test_vectorized_period_rank_ic_matches_direct_spearman() -> None:
    timestamps = pd.to_datetime(["2024-01-02T15:00:00Z"] * 12 + ["2024-01-02T15:30:00Z"] * 12)
    signal = np.tile(np.arange(12, dtype=float), 2)
    target = np.concatenate([signal[:12] * 2.0, -signal[:12]])
    frame = pd.DataFrame({"decision_timestamp": timestamps, "signal": signal, "target": target})

    result = rank_ic_by_period(frame, "signal", "target")

    assert result.tolist() == [1.0, -1.0]


def test_retirement_identity_changes_when_candidate_changes(tmp_path) -> None:
    candidate = tmp_path / "sip"
    candidate.mkdir()
    (candidate / "manifest.json").write_text("one", encoding="utf-8")
    first_stats, first_identity = _directory_identity(candidate)

    (candidate / "manifest.json").write_text("two-two", encoding="utf-8")
    second_stats, second_identity = _directory_identity(candidate)

    assert first_stats.bytes != second_stats.bytes
    assert first_identity != second_identity


def test_active_consumer_gate_blocks_ds24_but_preserves_unrelated_ds26(tmp_path) -> None:
    candidate = tmp_path / "sip"
    state = {
        "relevant_processes": [
            {"pid": 10, "command_line": "python scripts/ds24_clean_worker.py"},
            {"pid": 11, "command_line": "python scripts/ds26_prospective_capture_worker.py"},
        ]
    }

    result = _active_consumer_gate(state, [candidate])

    assert result["passed"] is False
    assert result["blocking_consumers"][0]["pid"] == 10
    assert result["protected_unrelated_processes"][0]["pid"] == 11
