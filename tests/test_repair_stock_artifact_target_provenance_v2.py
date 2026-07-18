from __future__ import annotations

import inspect
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import repair_stock_artifact_target_provenance_v2 as repair


def _row(
    symbol: str,
    *,
    status: str,
    provenance: str | None,
    date: str = "2026-01-02",
) -> dict[str, object]:
    return {
        "rebalance_date": date,
        "symbol": symbol,
        "target_status": status,
        "target_provenance_contract_version": provenance,
        "actual_forward_return_10d": 0.1 if status == "realized" else None,
        "label_available_timestamp": (
            "2026-01-16T21:00:00Z" if status == "realized" else None
        ),
        "predicted_momentum_20d": 0.25,
        "momentum_250d": 0.5,
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("rebalance_date", pa.string()),
            pa.field("symbol", pa.string()),
            pa.field("target_status", pa.string()),
            pa.field("target_provenance_contract_version", pa.string()),
            pa.field("actual_forward_return_10d", pa.float64()),
            pa.field("label_available_timestamp", pa.string()),
            pa.field("predicted_momentum_20d", pa.float64()),
            pa.field("momentum_250d", pa.float64()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def _run(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    dry_run: bool = False,
    promote: bool = False,
    output: Path | None = None,
    writer_factory=repair.pq.ParquetWriter,
) -> tuple[dict[str, object], Path, Path]:
    source = tmp_path / "source.parquet"
    repaired = output or tmp_path / "repaired.parquet"
    _write(source, rows)
    if promote:
        identity = repair.bounded_parquet_artifact_identity(
            source, batch_rows=1, resolved_artifact_path=source
        )
        (tmp_path / "stock_level_prediction_artifacts.json").write_text(
            json.dumps(repair._refreshed_publication({}, identity)),
            encoding="utf-8",
        )
    report = repair.repair_stock_artifact_target_provenance_v2(
        input_path=source,
        output_path=repaired,
        report_root=tmp_path / "report",
        dry_run=dry_run,
        promote=promote,
        batch_rows=1,
        writer_factory=writer_factory,
    )
    return report, source, repaired


def _versions(path: Path) -> list[str | None]:
    return pq.read_table(
        path, columns=["target_provenance_contract_version"]
    ).column(0).to_pylist()


def test_approved_unrealized_boundary_rows_are_repaired(tmp_path: Path) -> None:
    report, _source, output = _run(
        tmp_path,
        [
            _row("AAA", status="realized", provenance=repair.TARGET_PROVENANCE_V2),
            _row("BBB", status="unrealized_boundary", provenance=None),
        ],
    )

    assert _versions(output) == [
        repair.TARGET_PROVENANCE_V2,
        repair.TARGET_PROVENANCE_V2,
    ]
    assert report["approved_rows_repaired"] == 1


def test_approved_missing_source_price_row_is_repaired(tmp_path: Path) -> None:
    report, _source, output = _run(
        tmp_path,
        [_row("AAA", status="missing_source_price", provenance=None)],
    )

    assert _versions(output) == [repair.TARGET_PROVENANCE_V2]
    assert report["approved_rows_repaired"] == 1


def test_mixed_approved_status_populations_repair_successfully(
    tmp_path: Path,
) -> None:
    report, _source, output = _run(
        tmp_path,
        [
            _row("AAA", status="missing_source_price", provenance=None),
            _row("BBB", status="unrealized_boundary", provenance=""),
        ],
    )

    assert _versions(output) == [repair.TARGET_PROVENANCE_V2] * 2
    assert report["approved_rows_repaired"] == 2


def test_realized_blank_row_blocks(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    _write(source, [_row("AAA", status="realized", provenance=None)])

    with pytest.raises(repair.RepairBlockedError) as exc:
        repair.repair_stock_artifact_target_provenance_v2(
            input_path=source,
            output_path=tmp_path / "out.parquet",
            report_root=tmp_path / "report",
            dry_run=False,
            promote=False,
            batch_rows=2,
        )

    assert "UNAPPROVED_BLANK_TARGET_STATUS" in exc.value.report["blockers"]
    assert not (tmp_path / "out.parquet").exists()


@pytest.mark.parametrize("status", ["future_unknown", ""])
def test_unknown_or_empty_target_status_blocks(
    tmp_path: Path, status: str
) -> None:
    source = tmp_path / "source.parquet"
    _write(source, [_row("AAA", status=status, provenance="")])

    with pytest.raises(repair.RepairBlockedError) as exc:
        repair.repair_stock_artifact_target_provenance_v2(
            input_path=source,
            output_path=tmp_path / "out.parquet",
            report_root=tmp_path / "report",
            dry_run=False,
            promote=False,
            batch_rows=2,
        )

    assert exc.value.report["unapproved_blank_rows"] == 1


def test_v1_or_mixed_nonblank_provenance_blocks(tmp_path: Path) -> None:
    for name, rows in {
        "v1": [_row("AAA", status="realized", provenance="stock_level_target_provenance_v1")],
        "mixed": [
            _row("AAA", status="realized", provenance=repair.TARGET_PROVENANCE_V2),
            _row("BBB", status="realized", provenance="stock_level_target_provenance_v1"),
        ],
    }.items():
        root = tmp_path / name
        source = root / "source.parquet"
        _write(source, rows)
        with pytest.raises(repair.RepairBlockedError) as exc:
            repair.repair_stock_artifact_target_provenance_v2(
                input_path=source,
                output_path=root / "out.parquet",
                report_root=root / "report",
                dry_run=False,
                promote=False,
                batch_rows=2,
            )
        assert "NON_V2_PROVENANCE_PRESENT" in exc.value.report["blockers"]


def test_all_v2_input_is_idempotent(tmp_path: Path) -> None:
    rows = [
        _row("AAA", status="realized", provenance=repair.TARGET_PROVENANCE_V2),
        _row("BBB", status="unrealized_boundary", provenance=repair.TARGET_PROVENANCE_V2),
    ]
    report, _source, output = _run(tmp_path, rows)

    assert _versions(output) == [repair.TARGET_PROVENANCE_V2] * 2
    assert report["approved_rows_repaired"] == 0


def test_row_count_and_ordering_are_preserved(tmp_path: Path) -> None:
    rows = [
        _row("BBB", status="unrealized_boundary", provenance=None, date="2026-01-03"),
        _row("AAA", status="realized", provenance=repair.TARGET_PROVENANCE_V2),
    ]
    report, source, output = _run(tmp_path, rows)

    source_keys = pq.read_table(source, columns=["rebalance_date", "symbol"]).to_pydict()
    output_keys = pq.read_table(output, columns=["rebalance_date", "symbol"]).to_pydict()
    assert source_keys == output_keys
    assert report["total_rows"] == 2
    assert report["economic_key_match"] is True


def test_every_non_provenance_value_remains_identical(tmp_path: Path) -> None:
    rows = [
        _row("AAA", status="realized", provenance=repair.TARGET_PROVENANCE_V2),
        _row("BBB", status="missing_source_price", provenance=None),
    ]
    report, source, output = _run(tmp_path, rows)
    columns = [
        name
        for name in pq.read_schema(source).names
        if name != repair.PROVENANCE_COLUMN
    ]

    assert pq.read_table(source, columns=columns).equals(
        pq.read_table(output, columns=columns)
    )
    assert report["invariant_column_checksum_match"] is True
    assert report["null_populations_outside_provenance_match"] is True


def test_duplicate_economic_keys_block(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    _write(
        source,
        [
            _row("AAA", status="unrealized_boundary", provenance=None),
            _row("AAA", status="unrealized_boundary", provenance=None),
        ],
    )

    with pytest.raises(repair.RepairBlockedError) as exc:
        repair.repair_stock_artifact_target_provenance_v2(
            input_path=source,
            output_path=tmp_path / "out.parquet",
            report_root=tmp_path / "report",
            dry_run=False,
            promote=False,
            batch_rows=1,
        )

    assert "DUPLICATE_ECONOMIC_KEYS" in exc.value.report["blockers"]


def test_dry_run_writes_report_but_no_repaired_artifact(tmp_path: Path) -> None:
    report, source, output = _run(
        tmp_path,
        [
            _row("AAA", status="unrealized_boundary", provenance=None),
            _row("BBB", status="missing_source_price", provenance=None),
        ],
        dry_run=True,
    )

    assert source.exists()
    assert not output.exists()
    assert report["status"] == "DRY_RUN_COMPLETE"
    assert report["blank_or_null_rows_found"] == 2
    assert report["approved_boundary_population"] == 2
    assert (tmp_path / "report" / "target_provenance_repair_report.json").exists()


def test_side_by_side_repair_preserves_source(tmp_path: Path) -> None:
    rows = [_row("AAA", status="unrealized_boundary", provenance=None)]
    report, source, output = _run(tmp_path, rows)

    assert _versions(source) == [None]
    assert _versions(output) == [repair.TARGET_PROVENANCE_V2]
    assert report["promotion_status"] == "side_by_side"


def test_explicit_promotion_atomically_replaces_source(tmp_path: Path) -> None:
    report, source, output = _run(
        tmp_path,
        [_row("AAA", status="unrealized_boundary", provenance=None)],
        promote=True,
    )

    assert _versions(source) == [repair.TARGET_PROVENANCE_V2]
    assert not output.exists()
    assert report["promotion_status"] == "promoted"
    assert report["promoted_at"]
    assert Path(report["backup_path"]).exists()
    sidecar = json.loads(
        (tmp_path / "stock_level_prediction_artifacts.json").read_text()
    )
    assert sidecar["canonical_artifact"]["file_size_bytes"] == source.stat().st_size
    assert sidecar["canonical_artifact"]["sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert sidecar["artifact_sha256"] == sidecar["canonical_artifact"]["sha256"]


def test_promotion_refreshes_exact_stale_sidecar_and_is_idempotent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    _write(source, [_row("AAA", status="unrealized_boundary", provenance=None)])
    stale = repair.bounded_parquet_artifact_identity(
        source, batch_rows=1, resolved_artifact_path=source
    )
    stale["file_size_bytes"] = 1
    stale["sha256"] = "0" * 64
    sidecar_path = tmp_path / "stock_level_prediction_artifacts.json"
    sidecar_path.write_text(
        json.dumps(repair._refreshed_publication({}, stale)), encoding="utf-8"
    )

    first = repair.repair_stock_artifact_target_provenance_v2(
        input_path=source,
        output_path=tmp_path / "repaired.parquet",
        report_root=tmp_path / "report",
        dry_run=False,
        promote=True,
        batch_rows=1,
    )
    published = json.loads(sidecar_path.read_text())
    assert first["previous_publication_identity"]["sha256"] == "0" * 64
    assert published["canonical_artifact"] == first["promoted_publication_identity"]
    assert published["canonical_artifact"]["row_count"] == 1
    assert published["canonical_artifact"]["stable_column_order"] == pq.read_schema(source).names

    second = repair.repair_stock_artifact_target_provenance_v2(
        input_path=source,
        output_path=tmp_path / "repaired.parquet",
        report_root=tmp_path / "report-second",
        dry_run=False,
        promote=True,
        batch_rows=1,
    )
    assert second["promotion_status"] == "already_consistent"
    assert json.loads(sidecar_path.read_text()) == published


def test_publication_generation_failure_preserves_parquet_and_sidecar(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    _write(source, [_row("AAA", status="unrealized_boundary", provenance=None)])
    identity = repair.bounded_parquet_artifact_identity(
        source, batch_rows=1, resolved_artifact_path=source
    )
    sidecar_path = tmp_path / "stock_level_prediction_artifacts.json"
    sidecar_path.write_text(
        json.dumps(repair._refreshed_publication({}, identity)), encoding="utf-8"
    )
    parquet_before = source.read_bytes()
    sidecar_before = sidecar_path.read_bytes()

    def fail_publication(_path: Path, _payload: dict[str, object]) -> None:
        raise OSError("synthetic publication failure")

    with pytest.raises(OSError, match="synthetic publication failure"):
        repair.repair_stock_artifact_target_provenance_v2(
            input_path=source,
            output_path=tmp_path / "repaired.parquet",
            report_root=tmp_path / "report",
            dry_run=False,
            promote=True,
            batch_rows=1,
            publication_writer=fail_publication,
        )

    assert source.read_bytes() == parquet_before
    assert sidecar_path.read_bytes() == sidecar_before


def test_failed_write_preserves_source_and_prior_output(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "repaired.parquet"
    _write(source, [_row("AAA", status="unrealized_boundary", provenance=None)])
    _write(output, [_row("OLD", status="realized", provenance=repair.TARGET_PROVENANCE_V2)])
    source_before = source.read_bytes()
    output_before = output.read_bytes()

    class FailingWriter:
        def __init__(self, *_args, **_kwargs):
            raise OSError("synthetic write failure")

    with pytest.raises(OSError, match="synthetic write failure"):
        repair.repair_stock_artifact_target_provenance_v2(
            input_path=source,
            output_path=output,
            report_root=tmp_path / "report",
            dry_run=False,
            promote=False,
            batch_rows=1,
            writer_factory=FailingWriter,
        )

    assert source.read_bytes() == source_before
    assert output.read_bytes() == output_before


def test_report_records_source_and_repaired_checksums(tmp_path: Path) -> None:
    report, source, output = _run(
        tmp_path,
        [_row("AAA", status="unrealized_boundary", provenance=None)],
    )
    persisted = json.loads(
        (tmp_path / "report" / "target_provenance_repair_report.json").read_text()
    )

    assert persisted["source_checksum"] == repair._file_sha256(source)
    assert persisted["repaired_checksum"] == repair._file_sha256(output)
    assert persisted["source_checksum"] != persisted["repaired_checksum"]
    assert report == persisted


def test_implementation_uses_bounded_arrow_batches_without_full_materialization() -> None:
    source = inspect.getsource(repair)

    assert "iter_batches" in source
    assert "batch_size=batch_rows" in source
    assert "pandas" not in source
    assert ".to_pylist()" not in source
    assert ".read()" not in inspect.getsource(repair._scan_projected)
    assert ".read()" not in inspect.getsource(repair._invariant_fingerprint)
