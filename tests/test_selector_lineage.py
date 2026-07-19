from __future__ import annotations

from copy import deepcopy

import pytest

from core.research.ml.stock_level.selector_lineage import (
    LineageStatus,
    build_ready_daily_spine_manifest,
    canonical_timestamp,
    merge_enrichment_preserving_base,
    preflight_frozen_selector_dataset,
    row_id_checksum,
    selector_row_id,
    target_checksum,
    timestamp_at_or_before,
    validate_selector_parent_child_lineage,
)
from core.research.ml.stock_level.stock_level_alpha_features_builder import (
    build_stock_level_alpha_features,
)


def _manifest() -> dict:
    return {
        "status": "READY",
        "economic_target_id": "forward_return_10d",
        "target_provenance_contract_version": "stock_level_target_provenance_v2",
        "canonical_daily_dataset_version": "canonical_daily_v2.partitioned.v1",
        "canonical_daily_logical_checksum": "daily-hash",
        "asset_registry_version": "registry-v1",
        "asset_registry_checksum": "registry-hash",
        "daily_spine_identity": "spine-v1",
        "daily_spine_logical_checksum": "spine-hash",
        "calendar_version": "nyse-v1",
        "decision_timing_contract": "close-plus-five-minutes-v1",
        "configuration_hash": "config-hash",
        "git_commit": "a" * 40,
        "builder_contract_version": "builder-v1",
        "feature_schema_version": "features-v1",
    }


def _row(symbol: str = "A", day: int = 2) -> dict:
    date = f"2026-01-{day:02d}"
    row = {
        "asset_id": f"asset-{symbol.lower()}",
        "canonical_symbol": symbol,
        "symbol": symbol,
        "rebalance_date": date,
        "decision_session_date": date,
        "decision_timestamp": f"{date}T21:05:00Z",
        "feature_data_cutoff_timestamp": f"{date}T21:00:00Z",
        "economic_target_id": "forward_return_10d",
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
        "target_horizon_trading_days": 10,
        "overlapping_targets": True,
        "required_purge_horizon_trading_days": 10,
    }
    row["row_id"] = selector_row_id(row)
    return row


def _ready(parent_rows=None, child_rows=None, parent_manifest=None, child_manifest=None):
    parent = parent_rows or [_row("A"), _row("B")]
    child = child_rows or deepcopy(parent)
    return validate_selector_parent_child_lineage(
        parent_manifest=parent_manifest or _manifest(),
        child_manifest=child_manifest or _manifest(),
        parent_rows=parent,
        child_rows=child,
    )


def test_identical_selector_lineage_is_ready_and_deterministic() -> None:
    first = _ready()
    second = _ready()
    assert first.status is LineageStatus.READY
    assert first.as_dict() == second.as_dict()
    assert first.row_id_checksum == second.row_id_checksum
    assert first.target_checksum == second.target_checksum


@pytest.mark.parametrize(
    ("parent", "child", "status"),
    [
        ([_row("A"), _row("B")], [_row("A")], LineageStatus.ROW_COUNT_MISMATCH),
        ([_row("A")], [_row("A"), _row("B")], LineageStatus.ROW_COUNT_MISMATCH),
        ([_row("A")], [_row("B")], LineageStatus.ROW_POPULATION_MISMATCH),
        ([_row("A"), _row("A")], [_row("A")], LineageStatus.ROW_ID_DUPLICATE),
        ([_row("A")], [_row("A"), _row("A")], LineageStatus.ROW_ID_DUPLICATE),
    ],
)
def test_population_failures(parent, child, status) -> None:
    result = _ready(parent, child)
    assert status.value in result.blockers
    assert not result.ready


@pytest.mark.parametrize(
    ("field", "value", "status"),
    [
        ("economic_target_id", "other", LineageStatus.ECONOMIC_TARGET_MISMATCH),
        (
            "target_provenance_contract_version",
            "stock_level_target_provenance_v1",
            LineageStatus.PROVENANCE_CONTRACT_MISMATCH,
        ),
        (
            "target_provenance_contract_version",
            "stock_level_target_provenance_v4",
            LineageStatus.PROVENANCE_CONTRACT_MISMATCH,
        ),
    ],
)
def test_manifest_identity_failures(field, value, status) -> None:
    child = _manifest()
    child[field] = value
    result = _ready(child_manifest=child)
    assert status.value in result.blockers


def test_missing_parent_identity_fails() -> None:
    child = _manifest()
    child.pop("daily_spine_identity")
    assert (
        LineageStatus.PARENT_IDENTITY_MISSING.value
        in _ready(child_manifest=child).blockers
    )


def test_deprecated_generic_identity_cannot_contradict_explicit_identities() -> None:
    child = {**_manifest(), "target_contract": "stock_level_target_provenance_v4"}
    assert (
        LineageStatus.MANIFEST_ROW_CONTRADICTION.value
        in _ready(child_manifest=child).blockers
    )


def test_namespace_confused_and_source_configuration_identities_fail_closed() -> None:
    confused = {
        **_manifest(),
        "economic_target_id": "stock_level_target_provenance_v2",
        "target_provenance_contract_version": "forward_return_10d",
    }
    result = _ready(child_manifest=confused)
    assert LineageStatus.ECONOMIC_TARGET_MISMATCH.value in result.blockers
    assert LineageStatus.PROVENANCE_CONTRACT_MISMATCH.value in result.blockers
    changed_config = {**_manifest(), "configuration_hash": "different"}
    assert (
        LineageStatus.SOURCE_DATASET_MISMATCH.value
        in _ready(child_manifest=changed_config).blockers
    )


@pytest.mark.parametrize(
    ("field", "value", "status"),
    [
        ("actual_forward_return_10d", 0.2, LineageStatus.TARGET_VALUE_MISMATCH),
        (
            "actual_benchmark_return_10d",
            0.03,
            LineageStatus.BENCHMARK_VALUE_MISMATCH,
        ),
        ("target_status", "missing", LineageStatus.TARGET_VALUE_MISMATCH),
        (
            "label_end_timestamp",
            "2026-01-17T21:00:00Z",
            LineageStatus.TARGET_TIMESTAMP_MISMATCH,
        ),
        (
            "label_available_timestamp",
            "2026-01-21T21:00:00Z",
            LineageStatus.TARGET_TIMESTAMP_MISMATCH,
        ),
        (
            "decision_timestamp",
            "2026-01-02T22:05:00Z",
            LineageStatus.DECISION_TIMESTAMP_MISMATCH,
        ),
    ],
)
def test_target_and_timestamp_mutations_fail(field, value, status) -> None:
    parent = [_row()]
    child = deepcopy(parent)
    child[0][field] = value
    result = _ready(parent, child)
    assert status.value in result.blockers


def test_equivalent_utc_timestamps_compare_semantically() -> None:
    assert canonical_timestamp("2026-01-01T16:00:00Z") == canonical_timestamp(
        "2026-01-01T16:00:00+00:00"
    )
    assert timestamp_at_or_before(
        "2026-01-01T16:00:00+00:00", "2026-01-01T16:00:00Z"
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        canonical_timestamp("2026-01-01T16:00:00")


def test_partial_enrichment_preserves_population_order_and_targets() -> None:
    base = [_row("B"), _row("A")]
    features = [{**_row("A"), "momentum_250d": 0.4}]
    output = merge_enrichment_preserving_base(base, features)
    assert [row["symbol"] for row in output] == ["B", "A"]
    assert len(output) == len(base)
    assert output[0]["feature_coverage_status"] == "MISSING"
    assert output[1]["feature_coverage_status"] == "AVAILABLE"
    assert output[1]["momentum_250d"] == 0.4
    assert output[1]["actual_forward_return_10d"] == base[1][
        "actual_forward_return_10d"
    ]


def test_enrichment_rejects_unknown_and_duplicate_rows() -> None:
    with pytest.raises(ValueError, match="unknown enrichment"):
        merge_enrichment_preserving_base([_row("A")], [_row("B")])
    with pytest.raises(ValueError, match="ROW_ID_DUPLICATE"):
        merge_enrichment_preserving_base([_row("A")], [_row("A"), _row("A")])


def test_feature_input_cannot_overwrite_target() -> None:
    base = [_row("A")]
    poison = [{**_row("A"), "actual_forward_return_10d": 999.0, "alpha": 1.0}]
    output = merge_enrichment_preserving_base(base, poison)
    assert output[0]["actual_forward_return_10d"] == 0.1
    assert output[0]["alpha"] == 1.0


def test_poisoned_future_price_does_not_change_historical_feature() -> None:
    rows = [_row("A")]
    history = [
        {"date": f"2025-12-{day:02d}", "close": 100.0 + day}
        for day in range(1, 32)
    ]
    first, _ = build_stock_level_alpha_features(rows, {"A": history, "SPY": history})
    poisoned = history + [{"date": "2026-02-01", "close": 1_000_000.0}]
    second, _ = build_stock_level_alpha_features(
        rows, {"A": poisoned, "SPY": poisoned}
    )
    comparable = {
        key: value
        for key, value in first[0].items()
        if key != "feature_coverage_status"
    }
    assert comparable == {
        key: value
        for key, value in second[0].items()
        if key != "feature_coverage_status"
    }


def test_daily_spine_fixture_is_ready_and_deterministic() -> None:
    rows = [_row("A"), _row("B")]
    parent = {
        key: _manifest()[key]
        for key in (
            "canonical_daily_dataset_version",
            "canonical_daily_logical_checksum",
            "asset_registry_version",
            "asset_registry_checksum",
            "calendar_version",
            "decision_timing_contract",
        )
    }
    first = build_ready_daily_spine_manifest(
        rows,
        parent_identity=parent,
        physical_sha256="physical",
        configuration_hash="config",
        git_commit="a" * 40,
    )
    second = build_ready_daily_spine_manifest(
        rows,
        parent_identity=parent,
        physical_sha256="physical",
        configuration_hash="config",
        git_commit="a" * 40,
    )
    assert first == second
    assert first["status"] == "READY"
    assert first["row_id_checksum"] == row_id_checksum(rows)
    assert first["economic_target_id"] == "forward_return_10d"


def test_frozen_preflight_ready_and_blockers() -> None:
    base = [_row("A"), _row("B")]
    enriched = merge_enrichment_preserving_base(base, [{**row, "alpha": 1.0} for row in base])
    spine = {"status": "READY", "logical_checksum": "spine-hash"}
    ready = preflight_frozen_selector_dataset(
        daily_spine_manifest=spine,
        base_manifest=_manifest(),
        enriched_manifest=_manifest(),
        base_rows=base,
        enriched_rows=enriched,
        feature_columns=["alpha"],
    )
    assert ready["status"] == "READY"
    assert ready["lineage"]["target_checksum"] == target_checksum(base)

    blocked = preflight_frozen_selector_dataset(
        daily_spine_manifest={"status": "BLOCKED"},
        base_manifest=_manifest(),
        enriched_manifest={
            **_manifest(),
            "target_provenance_contract_version": "stock_level_target_provenance_v1",
        },
        base_rows=base,
        enriched_rows=enriched[:-1],
        feature_columns=["actual_forward_return_10d"],
    )
    assert blocked["status"] == "BLOCKED"
    assert "DAILY_SPINE_NOT_READY" in blocked["blockers"]
    assert "PROVENANCE_CONTRACT_MISMATCH" in blocked["blockers"]
    assert "ROW_COUNT_MISMATCH" in blocked["blockers"]
    assert "TARGET_COLUMN_IN_FEATURE_SCHEMA" in blocked["blockers"]

    timestamp_leak = preflight_frozen_selector_dataset(
        daily_spine_manifest=spine,
        base_manifest=_manifest(),
        enriched_manifest=_manifest(),
        base_rows=base,
        enriched_rows=enriched,
        feature_columns=["label_available_timestamp"],
    )
    assert "TARGET_COLUMN_IN_FEATURE_SCHEMA" in timestamp_leak["blockers"]
