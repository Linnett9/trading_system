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
    _write_partition(spine, [{"symbol": "AAPL", "session_date": "2026-01-01"}])
    _write_partition(base, [{"symbol": "AAPL", "rebalance_date": "2026-01-01", "actual_forward_return_10d": 0.1}])

    def fail_spine_read(path: Path, columns: object = None) -> list[dict[str, object]]:
        if Path(path) == spine:
            raise AssertionError("spine should not be opened when base partition is reusable")
        return [{"symbol": "AAPL", "rebalance_date": "2026-01-01", "actual_forward_return_10d": 0.1}]

    monkeypatch.setattr(alpha_enrichment, "_read_parquet_file", fail_spine_read)
    rows, meta = _read_symbol_source_rows_from_spine(
        "AAPL",
        spine,
        base,
        config={"ml": {}},
        input_resolution={},
    )

    assert rows[0]["symbol"] == "AAPL"
    assert meta["base_partition_reused"] is True
    assert meta["spine_read_seconds"] == 0.0


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
    _write_partition(base_root / "symbol=AAPL" / "rows.parquet", [{"symbol": "AAPL", "rebalance_date": "2026-01-01", "actual_forward_return_10d": 0.1}])
    config = {
        "ml": {
            "canonical_v2_labeled_spine_manifest_path": str(manifest),
            "canonical_v2_labeled_spine_root": str(spine_root),
            "canonical_v2_base_partition_root": str(base_root),
            "stooq_parquet_dir": str(price_root),
            "output_dir": str(tmp_path / "out"),
        }
    }
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
    _write_partition(base_root / "symbol=AAPL" / "rows.parquet", [{"symbol": "AAPL", "rebalance_date": "2026-01-01", "actual_forward_return_10d": 0.1}])
    config = {
        "ml": {
            "canonical_v2_labeled_spine_manifest_path": str(manifest),
            "canonical_v2_labeled_spine_root": str(spine_root),
            "canonical_v2_base_partition_root": str(base_root),
            "stooq_parquet_dir": str(tmp_path / "prices"),
            "output_dir": str(tmp_path / "out"),
        }
    }
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


def test_alpha_validate_real_aapl_partition_has_no_numeric_empty_strings() -> None:
    path = Path(
        "reports/ml/development/ticket_7b3_daily_large_history/regeneration_canonical_v2/"
        "alpha_enrichment_smoke_1/partitions/symbol=AAPL/rows.parquet"
    )
    if not path.exists():
        pytest.skip("AAPL smoke partition is not present in this checkout")

    report = _validate_partition_dataset([path])

    assert report["row_count"] == 10528
    assert report["duplicate_symbol_date_keys"] == 0
    assert report["partitions"][0]["type_mismatches"] == []
