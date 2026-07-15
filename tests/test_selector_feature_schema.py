from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.research.ml.stock_level.bounded_selector_runner import _resolve_features, _validate_features
from core.research.ml.stock_level.selector_feature_schema import (
    classify_column,
    load_feature_schema,
    schema_hash,
)


def _write_schema(path: Path, names: list[str], *, rule: str = "strictly prior") -> Path:
    payload = {
        "contract_version": "test_v1",
        "features": [{"name": name, "data_type": "double", "availability_rule": rule} for name in names],
    }
    payload["schema_hash"] = schema_hash(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_feature_classification_and_outcome_exclusion():
    assert classify_column("momentum_250d", predictor_names=("momentum_250d",))[0] == "safe predictor"
    assert classify_column("industry_relative_strength", predictor_names=("industry_relative_strength",))[0] == "conditionally available"
    assert classify_column("actual_forward_return_10d", predictor_names=("actual_forward_return_10d",))[0] == "target/outcome"


def test_feature_ordering_and_schema_hash_are_stable(tmp_path: Path):
    path = _write_schema(tmp_path / "schema.json", ["second", "first"])
    loaded = load_feature_schema(path)
    assert [row["name"] for row in loaded["features"]] == ["second", "first"]
    assert loaded["schema_hash"] == schema_hash(loaded)
    loaded["features"].reverse()
    assert loaded["schema_hash"] != schema_hash(loaded)


def test_schema_rejects_outcome(tmp_path: Path):
    path = _write_schema(tmp_path / "schema.json", ["actual_forward_return_10d"])
    with pytest.raises(RuntimeError, match="Outcome columns"):
        load_feature_schema(path)


def test_explicit_engineered_resolution_and_missing_feature_failure(tmp_path: Path):
    rows = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"momentum_250d": pa.array([0.1], type=pa.float64())}), rows)
    schema = _write_schema(tmp_path / "schema.json", ["momentum_250d"])
    features, identity = _resolve_features(rows, False, schema)
    assert features == ("momentum_250d",)
    assert identity["schema_hash"] == load_feature_schema(schema)["schema_hash"]
    missing = _write_schema(tmp_path / "missing.json", ["not_present"])
    with pytest.raises(RuntimeError, match="missing"):
        _resolve_features(rows, False, missing)


def test_all_null_feature_failure():
    table = pa.table({"feature": pa.array([None, None], type=pa.float64())})
    with pytest.raises(RuntimeError, match="entirely null"):
        _validate_features(table, ("feature",))


def test_schema_identity_changes_when_availability_contract_changes(tmp_path: Path):
    first = _write_schema(tmp_path / "first.json", ["feature"], rule="strictly prior")
    second = _write_schema(tmp_path / "second.json", ["feature"], rule="prior and published")
    assert load_feature_schema(first)["schema_hash"] != load_feature_schema(second)["schema_hash"]


def test_forward_and_label_fields_never_classify_as_safe():
    for name in ("actual_future_volatility", "label_end_timestamp", "target_start_timestamp", "benchmark_label_available_timestamp"):
        assert classify_column(name, predictor_names=())[0] in {"target/outcome", "potentially leaking", "timestamp"}
