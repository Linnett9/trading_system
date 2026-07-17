from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.profile_daily_spine_preflight import build_bounded_sample


def _artifact(path, rows, *, enriched=False):
    values = [
        {
            "rebalance_date": f"2024-01-{index % 28 + 1:02d}",
            "symbol": f"S{index % 5}",
            "decision_timestamp": f"2024-01-{index % 28 + 1:02d}T21:00:00Z",
            **({"feature": float(index)} if enriched else {}),
        }
        for index in range(rows)
    ]
    pq.write_table(pa.Table.from_pylist(values), path, row_group_size=10)
    return path


def test_bounded_sampler_preserves_schema_and_beginning_middle_end_groups(tmp_path):
    base = _artifact(tmp_path / "base.parquet", 40)
    enriched = _artifact(tmp_path / "enriched.parquet", 40, enriched=True)
    manifest = build_bounded_sample(
        base_source=base, enriched_source=enriched,
        output_root=tmp_path / "samples", rows=12,
    )
    assert manifest["production_certification_evidence"] is False
    assert manifest["base"]["selected_row_groups"] == [0, 2, 3]
    assert manifest["enriched"]["selected_row_groups"] == [0, 2, 3]
    assert manifest["base"]["schema_preserved"] is True
    assert manifest["enriched"]["schema_preserved"] is True
    assert manifest["base"]["sample_rows"] == 12
    assert manifest["matched_population_strategy"] == "same_global_ordinal_ranges"
    assert manifest["physical_order_match_required_for_certification"] is False
    assert manifest["sample_row_key_overlap"]["overlap_count"] == 12
    assert len(manifest["base"]["sample_sha256"]) == 64


def test_sampler_reads_only_explicit_selected_row_groups(monkeypatch, tmp_path):
    base = _artifact(tmp_path / "base.parquet", 40)
    enriched = _artifact(tmp_path / "enriched.parquet", 40, enriched=True)
    calls = []
    original = pq.ParquetFile.iter_batches

    def observed(self, *args, **kwargs):
        calls.append(tuple(kwargs.get("row_groups") or ()))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", observed)
    build_bounded_sample(
        base_source=base, enriched_source=enriched,
        output_root=tmp_path / "samples", rows=12,
    )
    assert calls
    assert all(len(groups) == 1 for groups in calls)
    assert {groups[0] for groups in calls} == {0, 2, 3}
    assert not list((tmp_path / "samples").glob("**/*.tmp"))


def test_sampler_refuses_population_mismatch(tmp_path):
    base = _artifact(tmp_path / "base.parquet", 40)
    enriched = _artifact(tmp_path / "enriched.parquet", 39, enriched=True)
    with pytest.raises(ValueError, match="equal row counts"):
        build_bounded_sample(
            base_source=base, enriched_source=enriched,
            output_root=tmp_path / "samples", rows=12,
        )
