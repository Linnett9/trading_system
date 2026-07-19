from __future__ import annotations

import json
import inspect
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import infrastructure.data.canonical_v2_alpha_enrichment as alpha_enrichment
from core.research.ml.stock_level.prediction_artifacts.sources import (
    _load_canonical_daily_v2_closes,
)
from infrastructure.data.canonical_daily_v2_builder import (
    _canonical_row,
    _completed_symbols,
)
from infrastructure.data.canonical_daily_v2_spines import build_selector_spines
from infrastructure.data.canonical_v2_alpha_enrichment import (
    BOOL_COLUMNS,
    INTERMEDIATE_NUMERIC_COLUMNS,
    PartitionBuildError,
    _column_type_inventory,
    _consolidate_partition_parquets,
    _base_partition_identity,
    _base_partition_identity_path,
    _completed_partition_paths,
    _completed_compatible_symbols,
    _partition_compatibility_identity,
    _failure_record,
    _normalize_partition_rows,
    _read_symbol_source_rows_from_spine,
    _read_parquet_file,
    _resolve_symbol_source,
    _schema_failure_payload,
    _schema_for_fieldnames,
    _symbol_spine_index,
    _validate_partition_dataset,
)


def test_tier_b_bridge_keeps_raw_fields_and_bridges_model_fields() -> None:
    row = _canonical_row(
        {"asset_id": "asset-bbw", "canonical_symbol": "BBW"},
        {
            "session_date": "2026-04-01",
            "open": 10,
            "high": 11,
            "low": 9,
            "close": 10.5,
            "volume": 100,
            "feed": "sip",
            "adjustment_policy": "all",
        },
        "TIER_B_COMPATIBLE_WITH_PRICE_BRIDGE",
        "alpaca",
        True,
        "2026-04-01",
        2.0,
        {},
    )

    assert row["raw_close"] == 10.5
    assert row["model_close"] == 21.0
    assert row["raw_volume"] == 100.0
    assert row["price_bridge_method"] == "median_overlap_ratio"


def test_tier_c_quarantine_only_removes_explicit_dates() -> None:
    asset = {"asset_id": "asset-abc", "canonical_symbol": "ABC"}
    base = {"open": 10, "high": 11, "low": 9, "close": 10, "volume": 1}

    quarantined = _canonical_row(
        asset,
        {"session_date": "2026-05-27", **base},
        "TIER_C_COMPATIBLE_WITH_DATE_QUARANTINE",
        "alpaca",
        False,
        "",
        1.0,
        {"2026-05-27": "missing_or_misaligned_previous_session"},
    )
    clean = _canonical_row(
        asset,
        {"session_date": "2026-05-28", **base},
        "TIER_C_COMPATIBLE_WITH_DATE_QUARANTINE",
        "alpaca",
        False,
        "",
        1.0,
        {"2026-05-27": "missing_or_misaligned_previous_session"},
    )

    assert quarantined["selector_eligible"] is False
    assert clean["selector_eligible"] is True


def test_tier_d_is_never_selector_eligible() -> None:
    row = _canonical_row(
        {"asset_id": "asset-aap", "canonical_symbol": "AAP"},
        {"session_date": "2026-05-28", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        "TIER_D_SYMBOL_QUARANTINE",
        "alpaca",
        False,
        "",
        1.0,
        {},
    )

    assert row["selector_eligible"] is False
    assert row["eligibility_reason"] == "tier_d_symbol_quarantine"


def test_completed_symbols_skips_only_complete_manifests(tmp_path: Path) -> None:
    (tmp_path / "AAPL.json").write_text(json.dumps({"symbol": "AAPL", "status": "COMPLETE"}))
    (tmp_path / "MSFT.json").write_text(json.dumps({"symbol": "MSFT", "status": "FAILED"}))

    assert _completed_symbols(tmp_path) == {"AAPL"}


def test_canonical_source_requires_complete_manifest_without_fallback(tmp_path: Path) -> None:
    root = tmp_path / "canonical" / "full"
    _write_partition(
        root / "symbol=AAPL" / "year=2026" / "bars.parquet",
        [
            {
                "session_date": "2026-01-02",
                "model_close": 100.0,
                "raw_volume": 10.0,
                "selector_eligible": True,
            }
        ],
    )
    manifest = tmp_path / "build_manifest.json"
    manifest.write_text(json.dumps({"status": "INCOMPLETE"}))
    config = {
        "ml": {
            "canonical_daily_v2_root": str(root),
            "canonical_daily_v2_manifest_path": str(manifest),
            "stock_alpha_dev_required_symbols": ["AAPL"],
        }
    }

    with pytest.raises(ValueError, match="not COMPLETE"):
        _load_canonical_daily_v2_closes(config)

    manifest.write_text(json.dumps({"status": "COMPLETE"}))
    closes = _load_canonical_daily_v2_closes(config)
    assert closes["AAPL"]["close"] == {"2026-01-02": 100.0}


def test_selector_spine_derives_cutoff_and_does_not_fabricate_inference_labels(tmp_path: Path) -> None:
    root = tmp_path / "canonical" / "full"
    rows = []
    for index in range(15):
        rows.append(
            {
                "asset_id": "asset-a",
                "canonical_symbol": "AAA",
                "session_date": f"2026-01-{index + 1:02d}",
                "model_close": float(100 + index),
                "source_provider": "stooq",
                "compatibility_tier": "TIER_A_NATIVE_COMPATIBLE",
                "selector_eligible": True,
                "eligibility_reason": "eligible",
                "return_valid": True,
                "return_invalid_reason": "",
                "quarantine_flag": False,
                "provider_transition_flag": False,
                "provider_transition_id": "",
            }
        )
    _write_partition(root / "symbol=AAA" / "year=2026" / "bars.parquet", rows)

    result = build_selector_spines(
        dataset_root=root,
        report_root=tmp_path / "spines",
        target_horizon_sessions=3,
    )

    assert result["validation"]["target_complete_maximum_date"] == "2026-01-12"
    assert result["validation"]["inference_rows_with_fabricated_targets"] == 0
    assert result["labeled"]["row_count"] == 12
    assert result["inference"]["row_count"] == 3


def _write_partition(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_alpha_numeric_placeholders_normalise_to_float_nulls() -> None:
    rows = [
        {"symbol": "AAA", "rebalance_date": "2026-01-01", "momentum_250d": ""},
        {"symbol": "AAA", "rebalance_date": "2026-01-02", "momentum_250d": None},
        {"symbol": "AAA", "rebalance_date": "2026-01-03", "momentum_250d": 1},
        {"symbol": "AAA", "rebalance_date": "2026-01-04", "momentum_250d": 1.5},
    ]

    normalized, report = _normalize_partition_rows(rows)

    assert [row["momentum_250d"] for row in normalized] == [None, None, 1.0, 1.5]
    assert report["values_coerced_to_null_by_column"] == {"momentum_250d": 1}


def test_alpha_text_column_rejects_float_with_column_name() -> None:
    rows = [{"symbol": "AAA", "rebalance_date": "2026-01-01", "source_provider": 1.5}]

    with pytest.raises(ValueError, match="source_provider"):
        _normalize_partition_rows(rows)


def test_alpha_dictionary_encoded_strings_read_as_strings(tmp_path: Path) -> None:
    path = tmp_path / "symbol=AAPL" / "rows.parquet"
    path.parent.mkdir(parents=True)
    table = pa.table({"symbol": pa.array(["AAPL", "AAPL"]).dictionary_encode(), "rebalance_date": ["2026-01-01", "2026-01-02"]})
    pq.write_table(table, path)

    rows = _read_parquet_file(path)

    assert rows[0]["symbol"] == "AAPL"


def test_alpha_datetime_columns_normalise_to_strings() -> None:
    rows = [{"symbol": "AAA", "rebalance_date": "2026-01-01", "decision_timestamp": "2026-01-01 20:05:00+00:00"}]

    normalized, _ = _normalize_partition_rows(rows)
    schema = _schema_for_fieldnames(list(normalized[0]))

    assert str(schema.field("decision_timestamp").type) == "string"


def test_alpha_normalisation_preserves_row_count_and_keys() -> None:
    rows = [
        {"symbol": "AAA", "rebalance_date": "2026-01-01", "actual_forward_return_10d": 0.1},
        {"symbol": "AAA", "rebalance_date": "2026-01-02", "actual_forward_return_10d": ""},
    ]

    normalized, report = _normalize_partition_rows(rows)

    assert len(normalized) == len(rows)
    assert [(row["symbol"], row["rebalance_date"]) for row in normalized] == [("AAA", "2026-01-01"), ("AAA", "2026-01-02")]
    assert report["duplicate_symbol_date_keys"] == 0


def test_alpha_schema_failure_payload_preserves_inventory() -> None:
    rows = [{"symbol": "AAA", "rebalance_date": "2026-01-01", "source_provider": 1.5}]
    inventory = _column_type_inventory(rows)

    try:
        _normalize_partition_rows(rows)
    except Exception as exc:
        payload = _schema_failure_payload("AAA", exc, inventory, phase="schema_construction")

    assert payload["symbol"] == "AAA"
    assert payload["phase"] == "schema_construction"
    assert "source_provider" in payload["column_type_inventory"]


def test_alpha_valid_partition_writes_atomically(tmp_path: Path) -> None:
    rows = [{"symbol": "AAA", "rebalance_date": "2026-01-01", "momentum_250d": 1.2}]
    normalized, _ = _normalize_partition_rows(rows)
    target = tmp_path / "rows.parquet"
    tmp = target.with_suffix(".parquet.tmp")

    pq.write_table(pa.Table.from_pylist(normalized, schema=_schema_for_fieldnames(list(normalized[0]))), tmp)
    tmp.replace(target)

    assert target.exists()
    assert not tmp.exists()


def test_alpha_failed_schema_build_leaves_no_promoted_parquet(tmp_path: Path) -> None:
    rows = [{"symbol": "AAA", "rebalance_date": "2026-01-01", "source_provider": 1.2}]
    target = tmp_path / "rows.parquet"

    with pytest.raises(ValueError):
        _normalize_partition_rows(rows)

    assert not target.exists()


def test_alpha_trend_indicator_is_explicit_numeric_contract() -> None:
    rows = [
        {"symbol": "AAA", "rebalance_date": "2026-01-01", "_stock_above_200d_average": 0.0},
        {"symbol": "AAA", "rebalance_date": "2026-01-02", "_stock_above_200d_average": 1.0},
        {"symbol": "AAA", "rebalance_date": "2026-01-03", "_stock_above_200d_average": ""},
    ]

    normalized, report = _normalize_partition_rows(rows)
    schema = _schema_for_fieldnames(list(normalized[0]))

    assert "_stock_above_200d_average" in INTERMEDIATE_NUMERIC_COLUMNS
    assert "_stock_above_200d_average" not in BOOL_COLUMNS
    assert [row["_stock_above_200d_average"] for row in normalized] == [0.0, 1.0, None]
    assert str(schema.field("_stock_above_200d_average").type) == "double"
    assert {
        column["name"]: column["kind"]
        for column in report["columns"]
    }["_stock_above_200d_average"] == "float"


def test_alpha_explicit_boolean_accepts_only_binary_values() -> None:
    normalized, _ = _normalize_partition_rows(
        [
            {"symbol": "AAA", "rebalance_date": "2026-01-01", "selector_eligible": 0.0},
            {"symbol": "AAA", "rebalance_date": "2026-01-02", "selector_eligible": 1},
            {"symbol": "AAA", "rebalance_date": "2026-01-03", "selector_eligible": True},
        ]
    )

    assert [row["selector_eligible"] for row in normalized] == [False, True, True]
    with pytest.raises(ValueError, match="selector_eligible"):
        _normalize_partition_rows([{"symbol": "AAA", "rebalance_date": "2026-01-04", "selector_eligible": 0.5}])


@pytest.mark.parametrize("value", [True, False])
def test_alpha_true_stock_level_row_preserves_boolean(value: bool) -> None:
    normalized, report = _normalize_partition_rows(
        [
            {
                "symbol": "AAP",
                "rebalance_date": "2026-01-01",
                "true_stock_level_row": value,
            }
        ]
    )
    schema = _schema_for_fieldnames(list(normalized[0]))

    assert normalized[0]["true_stock_level_row"] is value
    assert schema.field("true_stock_level_row").type == pa.bool_()
    assert schema.field("true_stock_level_row").nullable is False
    assert {
        column["name"]: column["kind"] for column in report["columns"]
    }["true_stock_level_row"] == "bool"


@pytest.mark.parametrize("value", [None, "True", "False", 1, 0])
def test_alpha_true_stock_level_row_rejects_non_boolean_values(value) -> None:
    with pytest.raises(ValueError, match="true_stock_level_row"):
        _normalize_partition_rows(
            [
                {
                    "symbol": "AAP",
                    "rebalance_date": "2026-01-01",
                    "true_stock_level_row": value,
                }
            ]
        )


def test_alpha_overlapping_targets_preserves_boolean_type() -> None:
    normalized, _ = _normalize_partition_rows(
        [
            {
                "symbol": "AAP",
                "rebalance_date": "2026-01-01",
                "overlapping_targets": True,
            },
            {
                "symbol": "AAP",
                "rebalance_date": "2026-01-02",
                "overlapping_targets": False,
            },
        ]
    )
    schema = _schema_for_fieldnames(list(normalized[0]))

    assert [row["overlapping_targets"] for row in normalized] == [True, False]
    assert schema.field("overlapping_targets").type == pa.bool_()


def test_alpha_integer_and_nullable_prediction_contracts_are_explicit() -> None:
    normalized, _ = _normalize_partition_rows(
        [
            {
                "symbol": "AAP",
                "rebalance_date": "2026-01-01",
                "target_horizon_trading_days": 10,
                "context_age_calendar_days": "",
                "fundamental_coverage_count": "",
                "predicted_forward_return_10d": "",
                "momentum_250d": "",
                "revenue_growth_yoy": "",
                "industry_mapping_available": "",
            },
            {
                "symbol": "AAP",
                "rebalance_date": "2026-01-02",
                "target_horizon_trading_days": 10.0,
                "context_age_calendar_days": 2,
                "fundamental_coverage_count": 14,
                "predicted_forward_return_10d": 0.25,
                "momentum_250d": 0.5,
                "revenue_growth_yoy": 0.12,
                "industry_mapping_available": 1.0,
            },
        ]
    )
    schema = _schema_for_fieldnames(list(normalized[0]))

    assert [row["target_horizon_trading_days"] for row in normalized] == [10, 10]
    assert [row["context_age_calendar_days"] for row in normalized] == [None, 2]
    assert [row["fundamental_coverage_count"] for row in normalized] == [None, 14]
    assert [row["predicted_forward_return_10d"] for row in normalized] == [
        None,
        0.25,
    ]
    assert [row["momentum_250d"] for row in normalized] == [None, 0.5]
    assert [row["revenue_growth_yoy"] for row in normalized] == [None, 0.12]
    assert [row["industry_mapping_available"] for row in normalized] == [
        None,
        1.0,
    ]
    assert schema.field("target_horizon_trading_days").type == pa.int64()
    assert schema.field("context_age_calendar_days").type == pa.int64()
    assert schema.field("fundamental_coverage_count").type == pa.int64()
    assert schema.field("predicted_forward_return_10d").type == pa.float64()
    assert schema.field("predicted_forward_return_10d").nullable is True
    assert schema.field("momentum_250d").type == pa.float64()
    assert schema.field("revenue_growth_yoy").type == pa.float64()
    assert schema.field("industry_mapping_available").type == pa.float64()


def test_alpha_average_dollar_volume_is_nullable_numeric() -> None:
    normalized, _ = _normalize_partition_rows(
        [
            {
                "symbol": "ACGL",
                "rebalance_date": "2026-01-01",
                "average_dollar_volume_21d": "",
                "average_dollar_volume_63d": None,
            },
            {
                "symbol": "ACGL",
                "rebalance_date": "2026-01-02",
                "average_dollar_volume_21d": 71_335_728.87585714,
                "average_dollar_volume_63d": 68_000_000,
            },
        ]
    )
    schema = _schema_for_fieldnames(list(normalized[0]))

    assert [row["average_dollar_volume_21d"] for row in normalized] == [
        None,
        71_335_728.87585714,
    ]
    assert schema.field("average_dollar_volume_21d").type == pa.float64()
    assert schema.field("average_dollar_volume_21d").nullable is True
    assert schema.field("average_dollar_volume_63d").type == pa.float64()


def test_alpha_output_schema_map_is_exhaustive_and_conflict_checked() -> None:
    assert set(alpha_enrichment.ALPHA_OUTPUT_SCHEMA) == set(
        alpha_enrichment.ALPHA_EXPECTED_OUTPUT_COLUMNS
    )
    assert all(
        column in alpha_enrichment.ALPHA_OUTPUT_SCHEMA
        for column in alpha_enrichment.FEATURE_DEFINITIONS
    )
    assert alpha_enrichment._validate_alpha_output_schema_coverage(
        alpha_enrichment.BASE_ARTIFACT_FIXED_COLUMNS
    )["status"] == "COMPLETE"

    with pytest.raises(ValueError, match="conflicting.*x"):
        alpha_enrichment._build_alpha_output_schema_map(
            expected_columns=["x"],
            bool_columns=["x"],
            int_columns=["x"],
            numeric_columns=[],
        )


def test_alpha_feature_producer_outputs_have_explicit_schema_kinds() -> None:
    from core.research.ml.stock_level.stock_level_alpha_features_builder import (
        _time_series_features,
    )

    produced = set(_time_series_features([], []))

    assert produced <= set(alpha_enrichment.ALPHA_OUTPUT_SCHEMA)
    assert set(alpha_enrichment.ENGINEERED_FEATURE_COLUMNS) <= set(
        alpha_enrichment.ALPHA_OUTPUT_SCHEMA
    )
    assert set(alpha_enrichment.ENRICHMENT_METADATA_COLUMNS) <= set(
        alpha_enrichment.ALPHA_OUTPUT_SCHEMA
    )


def test_alpha_unknown_base_column_fails_schema_preflight_before_pool() -> None:
    with pytest.raises(ValueError, match="unknown_base_columns=.*mystery_output"):
        alpha_enrichment._validate_alpha_output_schema_coverage(
            ["rebalance_date", "symbol", "mystery_output"]
        )

    source = inspect.getsource(
        alpha_enrichment._write_partitioned_canonical_v2_alpha_features
    )
    assert source.index("_validate_alpha_output_schema_coverage(") < source.index(
        "_execute_alpha_process_pool("
    )


def test_alpha_production_style_acgl_partition_round_trip(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "symbol": "ACGL",
            "rebalance_date": f"2026-{(index // 28) + 1:02d}-{(index % 28) + 1:02d}",
            "true_stock_level_row": True,
            "overlapping_targets": True,
            "average_dollar_volume_21d": 71_335_728.87585714,
            "average_dollar_volume_63d": (
                None if index < 63 else 70_000_000.0
            ),
            "predicted_liquidity_score": (
                None if index < 21 else 18.08
            ),
            "market_volatility_20d": (
                None if index < 20 else 0.22
            ),
            "relative_momentum_vs_sector": (
                "" if index == 0 else 0.03
            ),
        }
        for index in range(2_206)
    ]
    normalized, report = _normalize_partition_rows(rows)
    schema = _schema_for_fieldnames(list(normalized[0]))
    path = tmp_path / "partitions" / "symbol=ACGL" / "rows.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(normalized, schema=schema), path)
    observed = pq.ParquetFile(path).read()

    assert report["valid"] is True
    assert observed.num_rows == 2_206
    for column in (
        "average_dollar_volume_21d",
        "average_dollar_volume_63d",
        "predicted_liquidity_score",
        "market_volatility_20d",
        "relative_momentum_vs_sector",
    ):
        assert observed.schema.field(column).type == pa.float64()
    assert observed.schema.field("true_stock_level_row").type == pa.bool_()
    assert observed.schema.field("overlapping_targets").type == pa.bool_()


def test_alpha_aap_boolean_partition_parquet_round_trip(tmp_path: Path) -> None:
    rows = [
        {
            "symbol": "AAP",
            "rebalance_date": f"2026-01-{(index % 28) + 1:02d}-{index:04d}",
            "true_stock_level_row": True,
            "overlapping_targets": index % 2 == 0,
            "target_horizon_trading_days": 10,
            "predicted_forward_return_10d": (
                None if index % 3 == 0 else index / 10_000
            ),
            "momentum_250d": "" if index == 0 else index / 1_000,
        }
        for index in range(2_206)
    ]
    normalized, report = _normalize_partition_rows(rows)
    path = tmp_path / "symbol=AAP" / "rows.parquet"
    path.parent.mkdir(parents=True)
    table = pa.Table.from_pylist(
        normalized, schema=_schema_for_fieldnames(list(normalized[0]))
    )
    pq.write_table(table, path)
    observed = pq.ParquetFile(path).read()

    assert report["valid"] is True
    assert observed.num_rows == 2_206
    assert observed.schema.field("true_stock_level_row").type == pa.bool_()
    assert observed.schema.field("true_stock_level_row").nullable is False
    assert observed.column("true_stock_level_row").to_pylist() == [True] * 2_206
    assert observed.schema.field("overlapping_targets").type == pa.bool_()
    assert observed.schema.field("predicted_forward_return_10d").type == pa.float64()


def test_alpha_name_fragments_do_not_make_boolean_schema() -> None:
    rows = [{"symbol": "AAA", "rebalance_date": "2026-01-01", "has_float_flag_name": 0.5}]

    with pytest.raises(ValueError, match="has_float_flag_name"):
        _normalize_partition_rows(rows)


def test_alpha_symbol_spine_index_and_missing_partition(tmp_path: Path) -> None:
    spine = tmp_path / "spines" / "symbol=AAPL" / "spine.parquet"
    _write_partition(spine, [{"symbol": "AAPL", "session_date": "2026-01-01"}])
    manifest = tmp_path / "labeled_spine_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "BUILT",
                "partition_manifests": [
                    {"canonical_symbol": "AAPL", "status": "BUILT", "path": str(spine), "row_count": 1}
                ],
            }
        )
    )
    config = {"ml": {"canonical_v2_labeled_spine_manifest_path": str(manifest)}}

    index = _symbol_spine_index(config)

    assert index["AAPL"]["path"] == str(spine)
    assert _resolve_symbol_source(config, "AAPL")["monolithic_base_read"] is False
    with pytest.raises(FileNotFoundError, match="MSFT"):
        _resolve_symbol_source(config, "MSFT")


def test_alpha_symbol_rows_reuse_existing_base_partition_without_spine_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spine = tmp_path / "spines" / "symbol=AAPL" / "spine.parquet"
    base = tmp_path / "base" / "symbol=AAPL" / "rows.parquet"
    config = {"ml": {}}
    input_resolution = {}
    _write_partition(spine, [{"symbol": "AAPL", "session_date": "2026-01-01"}])
    _write_partition(base, [{"symbol": "AAPL", "rebalance_date": "2026-01-01", "actual_forward_return_10d": 0.1}])
    _base_partition_identity_path(base).write_text(json.dumps(_base_partition_identity(base, config, input_resolution=input_resolution)))

    def fail_spine_read(path: Path, columns: object = None) -> list[dict[str, object]]:
        if Path(path) == spine:
            raise AssertionError("spine should not be opened when base partition is reusable")
        return [{"symbol": "AAPL", "rebalance_date": "2026-01-01", "actual_forward_return_10d": 0.1}]

    monkeypatch.setattr(alpha_enrichment, "_read_parquet_file", fail_spine_read)
    rows, meta = _read_symbol_source_rows_from_spine(
        "AAPL",
        spine,
        base,
        config=config,
        input_resolution=input_resolution,
    )

    assert rows[0]["symbol"] == "AAPL"
    assert meta["base_partition_reused"] is True
    assert meta["spine_read_seconds"] == 0.0


def test_alpha_symbol_rows_reject_v1_base_partition_and_rebuild(tmp_path: Path) -> None:
    spine_root = tmp_path / "spines"
    spine = spine_root / "symbol=AAPL" / "spine.parquet"
    spy = spine_root / "symbol=SPY" / "spine.parquet"
    base = tmp_path / "base" / "symbol=AAPL" / "rows.parquet"
    _write_partition(
        spine,
        [
            {
                "symbol": "AAPL",
                "canonical_symbol": "AAPL",
                "asset_id": "asset-aapl",
                "session_date": "2026-01-02",
                "target_end_session_date": "2026-01-16",
                "actual_forward_return_10d": 0.1,
                "selector_eligible": True,
                "is_labeled": True,
                "provider_transition_flag": False,
            },
            {
                "symbol": "AAPL",
                "canonical_symbol": "AAPL",
                "asset_id": "asset-aapl",
                "session_date": "2026-01-05",
                "target_end_session_date": "2026-01-20",
                "actual_forward_return_10d": 0.2,
                "selector_eligible": True,
                "is_labeled": True,
                "provider_transition_flag": False,
            },
        ],
    )
    _write_partition(spy, [{"session_date": "2026-01-02", "actual_forward_return_10d": 0.02}, {"session_date": "2026-01-05", "actual_forward_return_10d": 0.03}])
    _write_partition(base, [{"symbol": "AAPL", "rebalance_date": "2026-01-02", "target_provenance_contract_version": "stock_level_target_provenance_v1"}])

    rows, meta = _read_symbol_source_rows_from_spine(
        "AAPL",
        spine,
        base,
        config={"ml": {"canonical_v2_labeled_spine_root": str(spine_root)}},
        input_resolution={"canonical_dataset": {"hash": "hash"}},
    )

    assert meta["base_partition_reused"] is False
    assert rows[0]["target_provenance_contract_version"] == "stock_level_target_provenance_v2"
    assert rows[0]["label_start_timestamp"] == "2026-01-05T21:00:00Z"
    assert rows[0]["actual_market_residual_return_10d"] == pytest.approx(0.08)


def test_alpha_symbol_rows_build_base_from_single_spine_partition(tmp_path: Path) -> None:
    spine_root = tmp_path / "spines"
    spine = spine_root / "symbol=AAPL" / "spine.parquet"
    spy = spine_root / "symbol=SPY" / "spine.parquet"
    base = tmp_path / "base" / "symbol=AAPL" / "rows.parquet"
    _write_partition(
        spine,
        [
            {
                "symbol": "AAPL",
                "canonical_symbol": "AAPL",
                "asset_id": "asset-aapl",
                "session_date": "2026-01-01",
                "target_end_session_date": "2026-01-15",
                "actual_forward_return_10d": 0.1,
                "selector_eligible": True,
                "is_labeled": True,
                "provider_transition_flag": False,
            }
        ],
    )
    _write_partition(spy, [{"session_date": "2026-01-01", "actual_forward_return_10d": 0.02}])

    rows, meta = _read_symbol_source_rows_from_spine(
        "AAPL",
        spine,
        base,
        config={"ml": {"canonical_v2_labeled_spine_root": str(spine_root)}},
        input_resolution={"canonical_dataset": {"hash": "hash"}},
    )

    assert base.exists()
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["actual_benchmark_return_10d"] == 0.02
    assert meta["base_partition_reused"] is False
    assert meta["source_rows_read"] == 1


def test_alpha_base_row_missing_benchmark_fails_closed(tmp_path: Path) -> None:
    spine_root = tmp_path / "spines"
    spine = spine_root / "symbol=AAPL" / "spine.parquet"
    base = tmp_path / "base" / "symbol=AAPL" / "rows.parquet"
    _write_partition(
        spine,
        [
            {
                "symbol": "AAPL",
                "canonical_symbol": "AAPL",
                "asset_id": "asset-aapl",
                "session_date": "2026-01-02",
                "target_end_session_date": "2026-01-16",
                "actual_forward_return_10d": 0.1,
                "selector_eligible": True,
                "is_labeled": True,
                "provider_transition_flag": False,
            },
            {
                "symbol": "AAPL",
                "canonical_symbol": "AAPL",
                "asset_id": "asset-aapl",
                "session_date": "2026-01-05",
                "target_end_session_date": "2026-01-20",
                "actual_forward_return_10d": 0.2,
                "selector_eligible": True,
                "is_labeled": True,
                "provider_transition_flag": False,
            },
        ],
    )

    rows, _ = _read_symbol_source_rows_from_spine(
        "AAPL",
        spine,
        base,
        config={"ml": {"canonical_v2_labeled_spine_root": str(spine_root)}},
        input_resolution={"canonical_dataset": {"hash": "hash"}},
    )

    assert rows[0]["actual_benchmark_return_10d"] == ""
    assert rows[0]["actual_market_residual_return_10d"] == ""
    assert rows[0]["benchmark_label_start_timestamp"] == ""


def test_alpha_partition_build_records_paths_timings_and_no_monolith(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spine_root = tmp_path / "spines"
    spine = spine_root / "symbol=AAPL" / "spine.parquet"
    manifest = tmp_path / "labeled_spine_manifest.json"
    price_root = tmp_path / "prices"
    partition_root = tmp_path / "partitions"
    manifest_root = tmp_path / "manifests"
    base_root = tmp_path / "base"
    _write_partition(spine, [{"symbol": "AAPL", "session_date": "2026-01-01"}])
    manifest.write_text(
        json.dumps(
            {
                "status": "BUILT",
                "partition_manifests": [
                    {"canonical_symbol": "AAPL", "status": "BUILT", "path": str(spine), "row_count": 1}
                ],
            }
        )
    )
    config = {
        "ml": {
            "canonical_v2_labeled_spine_manifest_path": str(manifest),
            "canonical_v2_labeled_spine_root": str(spine_root),
            "canonical_v2_base_partition_root": str(base_root),
            "stooq_parquet_dir": str(price_root),
            "output_dir": str(tmp_path / "out"),
        }
    }
    base_path = base_root / "symbol=AAPL" / "rows.parquet"
    _write_partition(base_path, [{"symbol": "AAPL", "rebalance_date": "2026-01-01", "actual_forward_return_10d": 0.1}])
    _base_partition_identity_path(base_path).write_text(json.dumps(_base_partition_identity(base_path, config, input_resolution={"canonical_dataset": {"hash": "hash"}})))
    monkeypatch.setattr(alpha_enrichment, "resolve_inputs", lambda config: {"canonical_dataset": {"hash": "hash"}})
    monkeypatch.setattr(alpha_enrichment, "_load_price_histories", lambda root, symbols: {"AAPL": [{"date": "2025-12-31", "close": 100.0}]})
    monkeypatch.setattr(
        alpha_enrichment,
        "_build_symbol_rows",
        lambda payload: [{"symbol": "AAPL", "rebalance_date": "2026-01-01", "actual_forward_return_10d": 0.1, "_stock_above_200d_average": 1.0}],
    )

    manifest_payload = alpha_enrichment._build_partition("AAPL", config, [], partition_root, manifest_root)

    assert manifest_payload["source_mode"] == "labeled_spine_partition"
    assert manifest_payload["source_spine_path"].endswith("symbol=AAPL\\spine.parquet") or manifest_payload["source_spine_path"].endswith("symbol=AAPL/spine.parquet")
    assert manifest_payload["monolithic_base_read"] is False
    assert manifest_payload["base_partition_reused"] is True
    assert manifest_payload["price_history_rows_read"] == 1
    assert {"normalisation_seconds", "parquet_write_seconds", "total_seconds"} <= set(manifest_payload["phase_timings"])
    assert Path(manifest_payload["path"]).exists()
    assert manifest_payload["compatibility_identity"] == _partition_compatibility_identity(
        "AAPL",
        config,
        source_base_partition_path=str(base_path),
    )
    assert _completed_compatible_symbols(manifest_root, config) == {"AAPL"}


def test_alpha_partition_enrichment_preserves_v2_target_metadata_order_independent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spine_root = tmp_path / "spines"
    spine = spine_root / "symbol=AAPL" / "spine.parquet"
    spy = spine_root / "symbol=SPY" / "spine.parquet"
    manifest = tmp_path / "labeled_spine_manifest.json"
    partition_root = tmp_path / "partitions"
    manifest_root = tmp_path / "manifests"
    price_root = tmp_path / "prices"
    _write_partition(
        spine,
        [
                {"symbol": "AAPL", "canonical_symbol": "AAPL", "asset_id": "asset-aapl", "session_date": "2026-01-02", "target_end_session_date": "2026-01-16", "actual_forward_return_10d": 0.1, "selector_eligible": True, "is_labeled": True, "provider_transition_flag": False},
                {"symbol": "AAPL", "canonical_symbol": "AAPL", "asset_id": "asset-aapl", "session_date": "2026-01-05", "target_end_session_date": "2026-01-20", "actual_forward_return_10d": 0.2, "selector_eligible": True, "is_labeled": True, "provider_transition_flag": False},
                {"symbol": "AAPL", "canonical_symbol": "AAPL", "asset_id": "asset-aapl", "session_date": "2026-01-06", "target_end_session_date": "2026-01-21", "actual_forward_return_10d": 0.3, "selector_eligible": True, "is_labeled": True, "provider_transition_flag": False},
            ],
        )
    _write_partition(spy, [{"session_date": "2026-01-02", "actual_forward_return_10d": 0.02}, {"session_date": "2026-01-05", "actual_forward_return_10d": 0.03}, {"session_date": "2026-01-06", "actual_forward_return_10d": 0.04}])
    manifest.write_text(json.dumps({"status": "BUILT", "partition_manifests": [{"canonical_symbol": "AAPL", "status": "BUILT", "path": str(spine), "row_count": 2}]}))
    config = {"ml": {"canonical_v2_labeled_spine_manifest_path": str(manifest), "canonical_v2_labeled_spine_root": str(spine_root), "canonical_v2_base_partition_root": str(tmp_path / "base"), "stooq_parquet_dir": str(price_root), "output_dir": str(tmp_path / "out")}}
    monkeypatch.setattr(alpha_enrichment, "resolve_inputs", lambda config: {"canonical_dataset": {"hash": "hash"}})
    monkeypatch.setattr(alpha_enrichment, "_load_price_histories", lambda root, symbols: {"AAPL": []})

    def enrich(payload):
        rows, _history, _spy = payload
        return [{**row, "_stock_above_200d_average": 1.0} for row in reversed(rows)]

    monkeypatch.setattr(alpha_enrichment, "_build_symbol_rows", enrich)

    alpha_enrichment._build_partition("AAPL", config, [], partition_root, manifest_root)
    base_rows = _read_parquet_file(tmp_path / "base" / "symbol=AAPL" / "rows.parquet")
    enriched_rows = _read_parquet_file(partition_root / "symbol=AAPL" / "rows.parquet")
    by_key = {(row["symbol"], row["rebalance_date"]): row for row in enriched_rows}

    for base_row in base_rows[:-1]:
        enriched = by_key[(base_row["symbol"], base_row["rebalance_date"])]
        for column in (
            "actual_forward_return_10d",
            "actual_benchmark_return_10d",
            "actual_market_residual_return_10d",
            "target_provenance_contract_version",
            "target_start_timestamp",
            "label_start_timestamp",
            "label_end_timestamp",
            "label_available_timestamp",
            "benchmark_label_start_timestamp",
            "benchmark_label_end_timestamp",
            "benchmark_label_available_timestamp",
        ):
            assert enriched[column] == base_row[column]
        assert base_row["target_provenance_contract_version"] == "stock_level_target_provenance_v2"
        assert base_row["decision_timestamp"] < base_row["label_start_timestamp"] <= base_row["label_end_timestamp"] <= base_row["label_available_timestamp"]


def test_alpha_partition_resume_rejects_wrong_base_identity_and_feature_schema(tmp_path: Path) -> None:
    manifest_root = tmp_path / "manifests"
    part = tmp_path / "partitions" / "symbol=AAPL" / "rows.parquet"
    base = tmp_path / "base" / "symbol=AAPL" / "rows.parquet"
    config = {"ml": {"output_dir": str(tmp_path / "out")}}
    _write_partition(part, [{"symbol": "AAPL", "rebalance_date": "2026-01-01"}])
    _write_partition(base, [{"symbol": "AAPL", "rebalance_date": "2026-01-01"}])
    manifest_root.mkdir(parents=True)
    good = {
        "symbol": "AAPL",
        "status": "COMPLETE",
        "path": str(part),
        "source_base_partition_path": str(base),
        "compatibility_identity": _partition_compatibility_identity("AAPL", config, source_base_partition_path=str(base)),
    }
    (manifest_root / "AAPL.json").write_text(json.dumps(good))
    assert _completed_compatible_symbols(manifest_root, config) == {"AAPL"}

    wrong_base = dict(good)
    wrong_base["compatibility_identity"] = {**good["compatibility_identity"], "source_base_partition_sha256": "wrong"}
    (manifest_root / "AAPL.json").write_text(json.dumps(wrong_base))
    assert _completed_compatible_symbols(manifest_root, config) == set()

    wrong_schema = dict(good)
    wrong_schema["compatibility_identity"] = {**good["compatibility_identity"], "feature_schema_identity": "wrong"}
    (manifest_root / "AAPL.json").write_text(json.dumps(wrong_schema))
    assert _completed_compatible_symbols(manifest_root, config) == set()

    missing_evidence = {key: value for key, value in good.items() if key != "compatibility_identity"}
    (manifest_root / "AAPL.json").write_text(json.dumps(missing_evidence))
    with pytest.raises(FileNotFoundError, match="missing completed alpha partitions"):
        _completed_partition_paths(manifest_root, expected_symbols=["AAPL"], config=config)


def test_alpha_partition_failure_persists_payload_and_no_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spine_root = tmp_path / "spines"
    spine = spine_root / "symbol=AAPL" / "spine.parquet"
    manifest = tmp_path / "labeled_spine_manifest.json"
    partition_root = tmp_path / "partitions"
    manifest_root = tmp_path / "manifests"
    base_root = tmp_path / "base"
    _write_partition(spine, [{"symbol": "AAPL", "session_date": "2026-01-01"}])
    manifest.write_text(
        json.dumps(
            {
                "status": "BUILT",
                "partition_manifests": [
                    {"canonical_symbol": "AAPL", "status": "BUILT", "path": str(spine), "row_count": 1}
                ],
            }
        )
    )
    config = {
        "ml": {
            "canonical_v2_labeled_spine_manifest_path": str(manifest),
            "canonical_v2_labeled_spine_root": str(spine_root),
            "canonical_v2_base_partition_root": str(base_root),
            "stooq_parquet_dir": str(tmp_path / "prices"),
            "output_dir": str(tmp_path / "out"),
        }
    }
    base_path = base_root / "symbol=AAPL" / "rows.parquet"
    _write_partition(base_path, [{"symbol": "AAPL", "rebalance_date": "2026-01-01", "actual_forward_return_10d": 0.1}])
    _base_partition_identity_path(base_path).write_text(json.dumps(_base_partition_identity(base_path, config, input_resolution={"canonical_dataset": {"hash": "hash"}})))
    monkeypatch.setattr(alpha_enrichment, "resolve_inputs", lambda config: {"canonical_dataset": {"hash": "hash"}})
    monkeypatch.setattr(alpha_enrichment, "_load_price_histories", lambda root, symbols: {"AAPL": []})
    monkeypatch.setattr(
        alpha_enrichment,
        "_build_symbol_rows",
        lambda payload: [{"symbol": "AAPL", "rebalance_date": "2026-01-01", "source_provider": 1.5}],
    )

    with pytest.raises(PartitionBuildError) as exc:
        alpha_enrichment._build_partition("AAPL", config, [], partition_root, manifest_root)

    failure_path = manifest_root.parent / "partition_failures" / "AAPL.json"
    failure = json.loads(failure_path.read_text())
    assert failure["phase"] == "normalisation"
    assert failure["source_spine_path"].endswith("spine.parquet")
    assert failure["monolithic_base_read"] is False
    assert failure["phase_timings"]["price_history_read_seconds"] >= 0.0
    assert _failure_record("AAPL", exc.value)["traceback"]
    assert not (partition_root / "symbol=AAPL" / "rows.parquet").exists()


def test_alpha_clean_typed_partitions_stream_consolidate_without_table_pylist(tmp_path: Path) -> None:
    part_a = tmp_path / "partitions" / "symbol=AAA" / "rows.parquet"
    part_b = tmp_path / "partitions" / "symbol=BBB" / "rows.parquet"
    _write_partition(part_a, [{"symbol": "AAA", "rebalance_date": "2026-01-01", "actual_forward_return_10d": 0.1}])
    _write_partition(part_b, [{"symbol": "BBB", "rebalance_date": "2026-01-01", "actual_forward_return_10d": None}])
    output = tmp_path / "out" / "stock_level_prediction_artifacts_enriched.parquet"

    source = inspect.getsource(_consolidate_partition_parquets)
    assert ".to_pylist()" not in source
    assert "from_pylist" not in source
    identity = _consolidate_partition_parquets(
        [part_a, part_b],
        output,
        config={"ml": {}},
        sample_path=None,
        expected_row_count=2,
    )

    assert identity["row_count"] == 2
    assert output.exists()
    assert pq.ParquetFile(output).read()["actual_forward_return_10d"].to_pylist() == [0.1, None]


def test_alpha_consolidation_rejects_numeric_empty_string_with_partition_details(tmp_path: Path) -> None:
    path = tmp_path / "partitions" / "symbol=AAA" / "rows.parquet"
    path.parent.mkdir(parents=True)
    schema = pa.schema(
        [
            pa.field("symbol", pa.string()),
            pa.field("rebalance_date", pa.string()),
            pa.field("actual_forward_return_10d", pa.string()),
        ]
    )
    pq.write_table(pa.Table.from_pylist([{"symbol": "AAA", "rebalance_date": "2026-01-01", "actual_forward_return_10d": ""}], schema=schema), path)

    with pytest.raises(ValueError, match="actual_forward_return_10d.*row_index=0"):
        _validate_partition_dataset([path])


def test_alpha_consolidation_casts_dictionary_strings_to_canonical_strings(tmp_path: Path) -> None:
    first = tmp_path / "partitions" / "symbol=AAA" / "rows.parquet"
    second = tmp_path / "partitions" / "symbol=BBB" / "rows.parquet"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    pq.write_table(pa.table({"symbol": ["AAA"], "rebalance_date": ["2026-01-01"], "actual_forward_return_10d": [0.1]}), first)
    pq.write_table(
        pa.table(
            {
                "symbol": pa.array(["BBB"]).dictionary_encode(),
                "rebalance_date": pa.array(["2026-01-01"]).dictionary_encode(),
                "actual_forward_return_10d": [0.2],
            }
        ),
        second,
    )
    output = tmp_path / "out.parquet"

    identity = _consolidate_partition_parquets([first, second], output, config={"ml": {}}, sample_path=None, expected_row_count=2)

    assert identity["row_count"] == 2
    assert str(pq.ParquetFile(output).schema_arrow.field("symbol").type) == "string"


def test_alpha_consolidation_duplicate_symbol_date_fails_and_does_not_promote(tmp_path: Path) -> None:
    first = tmp_path / "partitions" / "symbol=AAA" / "rows.parquet"
    second = tmp_path / "partitions" / "symbol=AAA_DUP" / "rows.parquet"
    output = tmp_path / "out.parquet"
    _write_partition(first, [{"symbol": "AAA", "rebalance_date": "2026-01-01", "actual_forward_return_10d": 0.1}])
    _write_partition(second, [{"symbol": "AAA", "rebalance_date": "2026-01-01", "actual_forward_return_10d": 0.2}])

    with pytest.raises(ValueError, match="duplicate symbol/date"):
        _consolidate_partition_parquets([first, second], output, config={"ml": {}}, sample_path=None, expected_row_count=2)

    assert not output.exists()
    assert not output.with_suffix(output.suffix + ".tmp").exists()


def test_alpha_failed_consolidation_preserves_existing_final_artifact(tmp_path: Path) -> None:
    first = tmp_path / "partitions" / "symbol=AAA" / "rows.parquet"
    second = tmp_path / "partitions" / "symbol=AAA_DUP" / "rows.parquet"
    output = tmp_path / "out.parquet"
    _write_partition(output, [{"symbol": "OLD", "rebalance_date": "2026-01-01", "actual_forward_return_10d": 9.0}])
    before = output.read_bytes()
    _write_partition(first, [{"symbol": "AAA", "rebalance_date": "2026-01-01", "actual_forward_return_10d": 0.1}])
    _write_partition(second, [{"symbol": "AAA", "rebalance_date": "2026-01-01", "actual_forward_return_10d": 0.2}])

    with pytest.raises(ValueError, match="duplicate symbol/date"):
        _consolidate_partition_parquets([first, second], output, config={"ml": {}}, sample_path=None, expected_row_count=2)

    assert output.read_bytes() == before


def test_alpha_consolidation_successful_retry_publishes_after_failure(tmp_path: Path) -> None:
    bad = tmp_path / "bad" / "symbol=AAA" / "rows.parquet"
    good = tmp_path / "good" / "symbol=AAA" / "rows.parquet"
    output = tmp_path / "out.parquet"
    _write_partition(bad, [{"symbol": "AAA", "rebalance_date": "2026-01-01", "actual_forward_return_10d": 0.1}])
    _write_partition(good, [{"symbol": "AAA", "rebalance_date": "2026-01-02", "actual_forward_return_10d": 0.2}])

    with pytest.raises(ValueError):
        _consolidate_partition_parquets([bad, good], output, config={"ml": {}}, sample_path=None, expected_row_count=3)
    assert not output.exists()

    identity = _consolidate_partition_parquets([bad, good], output, config={"ml": {}}, sample_path=None, expected_row_count=2)

    assert identity["row_count"] == 2
    assert output.exists()


def _alpha_smoke_current_contract_issues(path: Path) -> list[str]:
    if not path.is_file():
        return ["artifact_missing"]
    schema = pq.ParquetFile(path).schema_arrow
    issues: list[str] = []
    for name in ("true_stock_level_row", "overlapping_targets"):
        if name not in schema.names:
            issues.append(f"{name}:missing")
        elif schema.field(name).type != pa.bool_():
            issues.append(f"{name}:{schema.field(name).type}")
    smoke_root = path.parents[2]
    manifest_path = (
        smoke_root / "partition_manifests" / f"{path.parent.name.removeprefix('symbol=')}.json"
    )
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    compatibility = dict(manifest.get("compatibility_identity") or {})
    if not compatibility:
        issues.append("compatibility_identity:missing")
    else:
        if (
            compatibility.get("alpha_enrichment_contract_version")
            != alpha_enrichment.ALPHA_ENRICHMENT_CONTRACT_VERSION
        ):
            issues.append("alpha_enrichment_contract_version:mismatch")
        if (
            compatibility.get("feature_schema_identity")
            != alpha_enrichment._feature_schema_identity()
        ):
            issues.append("feature_schema_identity:mismatch")
    return issues


def _write_current_alpha_smoke_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True)
    normalized, _ = _normalize_partition_rows(
        [
            {
                "symbol": path.parent.name.removeprefix("symbol="),
                "rebalance_date": "2026-01-01",
                "true_stock_level_row": True,
                "overlapping_targets": False,
                "actual_forward_return_10d": 0.1,
            }
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(
            normalized, schema=_schema_for_fieldnames(list(normalized[0]))
        ),
        path,
    )
    manifest_path = (
        path.parents[2]
        / "partition_manifests"
        / f"{path.parent.name.removeprefix('symbol=')}.json"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "compatibility_identity": {
                    "alpha_enrichment_contract_version": (
                        alpha_enrichment.ALPHA_ENRICHMENT_CONTRACT_VERSION
                    ),
                    "feature_schema_identity": (
                        alpha_enrichment._feature_schema_identity()
                    ),
                },
            }
        ),
        encoding="utf-8",
    )


def test_alpha_smoke_contract_detection_is_consistent_when_absent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "smoke" / "partitions" / "symbol=AAPL" / "rows.parquet"

    assert _alpha_smoke_current_contract_issues(path) == ["artifact_missing"]


def test_alpha_smoke_contract_detection_rejects_legacy_boolean_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "smoke" / "partitions" / "symbol=AAPL" / "rows.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "symbol": ["AAPL"],
                "rebalance_date": ["2026-01-01"],
                "overlapping_targets": [""],
                "actual_forward_return_10d": [0.1],
            }
        ),
        path,
    )

    issues = _alpha_smoke_current_contract_issues(path)

    assert "true_stock_level_row:missing" in issues
    assert "overlapping_targets:string" in issues
    assert "compatibility_identity:missing" in issues


def test_alpha_smoke_contract_detection_accepts_current_boolean_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "smoke" / "partitions" / "symbol=AAPL" / "rows.parquet"
    _write_current_alpha_smoke_fixture(path)

    assert _alpha_smoke_current_contract_issues(path) == []
    report = _validate_partition_dataset([path])
    assert report["row_count"] == 1
    assert report["partitions"][0]["type_mismatches"] == []


def test_alpha_validate_real_aapl_partition_has_no_numeric_empty_strings() -> None:
    path = Path(
        "reports/ml/development/ticket_7b3_daily_large_history/regeneration_canonical_v2/"
        "alpha_enrichment_smoke_1/partitions/symbol=AAPL/rows.parquet"
    )
    issues = _alpha_smoke_current_contract_issues(path)
    if issues == ["artifact_missing"]:
        pytest.skip("AAPL smoke partition is not present in this checkout")
    if issues:
        pytest.skip(
            "legacy AAPL smoke partition is not current-schema authoritative: "
            + ", ".join(issues)
        )

    report = _validate_partition_dataset([path])

    assert report["row_count"] == 10528
    assert report["duplicate_symbol_date_keys"] == 0
    assert report["partitions"][0]["type_mismatches"] == []
