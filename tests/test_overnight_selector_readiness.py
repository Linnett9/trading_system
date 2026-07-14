from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from infrastructure.data import overnight_selector_readiness as readiness


def test_residual_trigger_explanations_are_deterministic() -> None:
    row = {
        "classification": "GENUINE_LARGE_RETURN_DISAGREEMENT",
        "return_abs_diff": "0.25",
        "close_rel_diff": "0.30",
        "price_ratio": "1.0",
        "price_ratio_deviation": "0.0",
        "alpaca_return": "0.2",
        "stooq_return": "-0.05",
    }

    first = readiness._trigger_reasons(row, {"price_ratio": "1.0", "session_date": "2026-05-26"})
    second = readiness._trigger_reasons(row, {"price_ratio": "1.0", "session_date": "2026-05-26"})

    assert first == second
    assert readiness._explanation_from_triggers(first) == "genuine_close_to_close_disagreement"


def test_bridge_invalidates_unbridged_transition_and_preserves_raw_fields() -> None:
    asset = {"asset_id": "asset_x", "canonical_symbol": "XYZ"}
    rows = [
        readiness._canonical_row(asset, {"session_date": "2026-03-26", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100, "source_path": "stooq"}, "TIER_A_NATIVE_COMPATIBLE", "stooq", False, "", 1.0, {}),
        readiness._canonical_row(asset, {"session_date": "2026-03-27", "open": 20, "high": 22, "low": 19, "close": 21, "volume": 200, "source_path": "alpaca"}, "TIER_A_NATIVE_COMPATIBLE", "alpaca", True, "2026-03-27", 1.0, {}),
    ]

    readiness._add_returns_and_volume_controls(rows)

    assert rows[1]["raw_close"] == 21
    assert rows[1]["model_close"] == 21
    assert rows[1]["return_valid"] is False
    assert rows[1]["return_invalid_reason"] == "unbridged_provider_transition"


def test_price_bridge_is_separate_from_raw_fields_and_allows_transition_return() -> None:
    asset = {"asset_id": "asset_x", "canonical_symbol": "XYZ"}
    rows = [
        readiness._canonical_row(asset, {"session_date": "2026-03-26", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100, "source_path": "stooq"}, "TIER_B_COMPATIBLE_WITH_PRICE_BRIDGE", "stooq", False, "", 1.0, {}),
        readiness._canonical_row(asset, {"session_date": "2026-03-27", "open": 20, "high": 22, "low": 19, "close": 21, "volume": 200, "source_path": "alpaca"}, "TIER_B_COMPATIBLE_WITH_PRICE_BRIDGE", "alpaca", True, "2026-03-27", 0.5, {}),
    ]

    readiness._add_returns_and_volume_controls(rows)

    assert rows[1]["raw_close"] == 21
    assert rows[1]["model_close"] == 10.5
    assert rows[1]["price_bridge_factor"] == 0.5
    assert rows[1]["return_valid"] is True


def test_quarantined_rows_cannot_be_return_valid() -> None:
    asset = {"asset_id": "asset_x", "canonical_symbol": "XYZ"}
    quarantine = {("XYZ", "2026-03-27"): {"quarantine_reason": "test"}}
    rows = [
        readiness._canonical_row(asset, {"session_date": "2026-03-26", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100, "source_path": "stooq"}, "TIER_A_NATIVE_COMPATIBLE", "stooq", False, "", 1.0, quarantine),
        readiness._canonical_row(asset, {"session_date": "2026-03-27", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100, "source_path": "stooq"}, "TIER_A_NATIVE_COMPATIBLE", "stooq", False, "", 1.0, quarantine),
    ]

    readiness._add_returns_and_volume_controls(rows)

    assert rows[1]["quarantine_flag"] is True
    assert rows[1]["return_valid"] is False
    assert rows[1]["return_invalid_reason"] == "quarantined_current_or_previous_row"


def test_large_artifact_resolution_rejects_legacy_candidates() -> None:
    candidates = readiness._legacy_candidates()

    assert candidates
    assert all(item["rejected_reason"] == "row/symbol count smaller than recovered large baseline" for item in candidates)


def test_smoke_outputs_keep_labeled_and_inference_spines_separate() -> None:
    labeled = Path("reports/ml/readiness/selector_spine_extension/labeled_selector_spine.parquet")
    inference = Path("reports/ml/readiness/selector_spine_extension/current_inference_spine.parquet")

    assert labeled.exists()
    assert inference.exists()
    labeled_rows = pq.read_table(labeled, columns=["is_labeled", "is_inference_only"]).to_pylist()
    inference_rows = pq.read_table(inference, columns=["is_labeled", "is_inference_only", "label_unavailable_reason"]).to_pylist()

    assert labeled_rows
    assert inference_rows
    assert all(row["is_labeled"] is True and row["is_inference_only"] is False for row in labeled_rows)
    assert all(row["is_labeled"] is False and row["is_inference_only"] is True for row in inference_rows)
    assert all(row["label_unavailable_reason"] for row in inference_rows)

