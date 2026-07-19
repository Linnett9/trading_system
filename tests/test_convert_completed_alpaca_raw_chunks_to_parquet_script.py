from __future__ import annotations

import json
import sys
import hashlib
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Barrier
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from infrastructure.data.historical_bar_providers import CollectionManifest
import infrastructure.data.alpaca_parquet_conversion_compute as compute_execution
import scripts.convert_completed_alpaca_raw_chunks_to_parquet as conversion_script
from scripts.convert_completed_alpaca_raw_chunks_to_parquet import convert_one, find_candidates, run_conversions, scan_candidates
from infrastructure.data.alpaca_parquet_conversion_compute import (
    ConversionComputeOptions,
    execute_conversion_run,
)


def test_converts_completed_chunk_and_preserves_payloads_by_default(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    chunk = _write_chunk(raw_root)
    candidate = find_candidates(raw_root, parquet_root)[0]

    result = convert_one(candidate, row_group_size=2)

    assert result["status"] == "converted"
    assert result["source_row_count"] == 2
    assert result["parquet_row_count"] == 2
    assert (chunk / "normalized_rows.json").exists()
    assert (chunk / "provider_pages.json").exists()
    assert (chunk / "manifest.json").exists()
    tombstone = json.loads((chunk / "parquet_conversion.json").read_text(encoding="utf-8"))
    assert tombstone["validation_result"] == "passed"
    assert tombstone["json_payloads_preserved"] is True
    assert tombstone["source_bytes_deleted"] == 0
    assert Path(tombstone["parquet_path"]).exists()
    table = pq.read_table(tombstone["parquet_path"])
    assert "source_raw_chunk_path" in table.schema.names
    assert "conversion_timestamp" in table.schema.names


def test_delete_json_requires_explicit_dangerous_flag(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    chunk = _write_chunk(raw_root)
    candidate = find_candidates(raw_root, parquet_root)[0]

    result = convert_one(candidate, row_group_size=2, delete_json_after_validate=True)

    assert result["status"] == "converted"
    assert not (chunk / "normalized_rows.json").exists()
    assert not (chunk / "provider_pages.json").exists()
    tombstone = json.loads((chunk / "parquet_conversion.json").read_text(encoding="utf-8"))
    assert tombstone["json_payloads_preserved"] is False
    assert tombstone["deleted_payload_files"]


def test_skips_tmp_in_progress_and_already_converted_chunks(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    _write_chunk(raw_root, suffix=".tmp")
    _write_chunk(raw_root, batch="MSFT", completion_state="in_progress")
    converted = _write_chunk(raw_root, batch="AAPL")
    candidate = find_candidates(raw_root, parquet_root)[0]
    assert convert_one(candidate, row_group_size=2)["status"] == "converted"

    candidates = find_candidates(raw_root, parquet_root)
    scan = scan_candidates(raw_root, parquet_root)

    assert candidates == []
    assert scan["skipped_existing_count"] == 1
    assert (converted / "manifest.json").exists()


def test_failed_conversion_retains_payloads(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    chunk = _write_chunk(raw_root)
    (chunk / "normalized_rows.json").write_text("[{", encoding="utf-8")
    candidate = find_candidates(raw_root, parquet_root)[0]

    result = convert_one(candidate, row_group_size=2)

    assert result["status"] == "failed"
    assert (chunk / "normalized_rows.json").exists()
    assert (chunk / "provider_pages.json").exists()
    assert not (chunk / "parquet_conversion.json").exists()


def test_parallel_runner_converts_distinct_chunks(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    _write_chunk(raw_root, batch="AAPL")
    _write_chunk(raw_root, batch="MSFT")
    candidates = find_candidates(raw_root, parquet_root)

    results = run_conversions(candidates, row_group_size=2, workers=2)

    assert sorted(result["status"] for result in results) == ["converted", "converted"]
    assert find_candidates(raw_root, parquet_root) == []


def test_dry_run_candidate_selection_uses_completed_json_chunks_only(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    _write_chunk(raw_root, batch="BRK.A")
    _write_chunk(raw_root, batch="AAPL", completion_state="in_progress")
    converted = _write_chunk(raw_root, batch="MSFT")
    msft = [candidate for candidate in find_candidates(raw_root, parquet_root) if "MSFT" in candidate["source_path"]][0]
    assert convert_one(msft, row_group_size=2)["status"] == "converted"

    candidates = find_candidates(raw_root, parquet_root)

    assert [Path(candidate["source_path"]).parent.name for candidate in candidates] == ["BRK.A"]
    assert (converted / "parquet_conversion.json").exists()


def test_recovered_brk_chunk_conversion_records_canonical_and_provider_provenance(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    chunk = _write_chunk(
        raw_root,
        batch="BOOM-BP-BRK.A-BRK.B",
        canonical_symbols=["BOOM", "BP", "BRK-A", "BRK-B"],
        provider_symbol_map={"BRK-A": "BRK.A", "BRK-B": "BRK.B"},
        row_symbol="BRK-B",
        provider_symbol="BRK.B",
    )
    candidate = find_candidates(raw_root, parquet_root)[0]

    result = convert_one(candidate, row_group_size=2)

    assert result["status"] == "converted"
    assert (chunk / "normalized_rows.json").exists()
    tombstone = json.loads((chunk / "parquet_conversion.json").read_text(encoding="utf-8"))
    assert tombstone["canonical_symbol_batch"] == ["BOOM", "BP", "BRK-A", "BRK-B"]
    assert tombstone["provider_symbol_batch"] == ["BOOM", "BP", "BRK.A", "BRK.B"]
    assert tombstone["provider_symbol_map"] == {"BRK-A": "BRK.A", "BRK-B": "BRK.B"}
    table = pq.read_table(tombstone["parquet_path"])
    assert table.column("symbol").to_pylist() == ["BRK-B", "BRK-B"]
    assert table.column("provider_symbol").to_pylist() == ["BRK.B", "BRK.B"]


def test_collection_manifest_filter_excludes_pilot_chunks(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    production = _write_chunk(raw_root, batch="BRK.A", canonical_symbols=["BRK-A"])
    _write_chunk(raw_root, batch="SPY-AAPL-MSFT")
    manifest_path = tmp_path / "collection_manifest.json"
    manifest = CollectionManifest(manifest_path)
    manifest.update(
        "alpaca-sip-5m-BRK-A-20260102T143000Z-20260102T150000Z",
        "completed",
        {"rows": 2},
    )

    candidates = find_candidates(raw_root, parquet_root, collection_manifest_path=manifest_path)

    assert [Path(candidate["source_path"]) for candidate in candidates] == [production]


def test_compute_execution_publishes_run_artifact_and_releases_lease(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    chunk = _write_chunk(raw_root)
    scan = scan_candidates(raw_root, parquet_root)
    options = ConversionComputeOptions(
        row_group_size=2,
        requested_workers=9,
        runs_root=tmp_path / "runs",
        resource_ledger_path=tmp_path / "leases.json",
        artifact_root=tmp_path / "artifacts",
        registry_path=tmp_path / "registry.json",
        invocation={"raw_root": str(raw_root), "max_chunks": 1},
    )

    execution = execute_conversion_run(
        candidates=scan["candidates"],
        compatible_skips=scan["compatible_skips"],
        convert_one=convert_one,
        options=options,
    )

    assert execution["summary"]["converted_count"] == 1
    assert execution["summary"]["effective_workers"] == 1
    assert (chunk / "normalized_rows.json").exists()
    assert (chunk / "parquet_conversion.json").exists()
    run_root = Path(execution["run_root"])
    assert (run_root / "run_manifest.json").exists()
    assert (run_root / "resource_summary.json").exists()
    artifact_manifest = next((tmp_path / "artifacts").glob("*/manifest.json"))
    payload = json.loads(artifact_manifest.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "DATA_STAGE_ARTIFACT"
    assert {row["relative_path"] for row in payload["file_inventory"]} == {
        "evidence/parquet_conversion.json",
    }
    authoritative = Path(execution["results"][0]["parquet_path"])
    assert authoritative.exists()
    assert not list((tmp_path / "artifacts").rglob("bars.parquet"))
    metadata = payload["conversion_metadata"]
    assert metadata["destination_identity"]
    assert metadata["parquet_output_size"] == authoritative.stat().st_size
    assert metadata["parquet_output_hash"]
    evidence = chunk / "parquet_conversion.json"
    assert metadata["evidence_identity"]
    assert metadata["evidence_size"] == evidence.stat().st_size
    assert metadata["evidence_hash"] == hashlib.sha256(evidence.read_bytes()).hexdigest()
    assert metadata["parquet_output_hash"] == hashlib.sha256(authoritative.read_bytes()).hexdigest()
    assert metadata["row_count"] == 2
    assert metadata["source_identity"]
    assert metadata["source_size"] > 0
    assert metadata["source_hash"]
    ledger = json.loads((tmp_path / "leases.json").read_text(encoding="utf-8"))
    assert ledger["active_leases"] == []
    assert ledger["recent_history"][-1]["request"]["estimated_peak_ram_bytes"] == 8 * 1024**3
    assert ledger["recent_history"][-1]["request"]["cpu_weight"] == 2


def test_all_compatible_compute_invocation_uses_no_lease_or_pool(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    _write_chunk(raw_root)
    candidate = find_candidates(raw_root, parquet_root)[0]
    assert convert_one(candidate, row_group_size=2)["status"] == "converted"
    scan = scan_candidates(raw_root, parquet_root)
    parquet = Path(scan["compatible_skips"][0]["parquet_path"])
    evidence = Path(scan["compatible_skips"][0]["evidence_path"])
    before = {
        "parquet": (parquet.stat().st_mtime_ns, hashlib.sha256(parquet.read_bytes()).hexdigest()),
        "evidence": (evidence.stat().st_mtime_ns, hashlib.sha256(evidence.read_bytes()).hexdigest()),
    }

    def forbidden_pool(**_kwargs):
        raise AssertionError("pool must not be constructed")

    execution = execute_conversion_run(
        candidates=scan["candidates"],
        compatible_skips=scan["compatible_skips"],
        convert_one=convert_one,
        options=ConversionComputeOptions(
            row_group_size=2,
            requested_workers=2,
            runs_root=tmp_path / "runs",
            resource_ledger_path=tmp_path / "leases.json",
            artifact_root=tmp_path / "artifacts",
            registry_path=tmp_path / "registry.json",
            invocation={"raw_root": str(raw_root), "max_chunks": 1},
            executor_factory=forbidden_pool,
        ),
    )

    assert execution["summary"]["compatible_skip_count"] == 1
    assert execution["summary"]["conversion_required_count"] == 0
    assert not (tmp_path / "leases.json").exists()
    assert not (tmp_path / "artifacts").exists()
    assert before == {
        "parquet": (parquet.stat().st_mtime_ns, hashlib.sha256(parquet.read_bytes()).hexdigest()),
        "evidence": (evidence.stat().st_mtime_ns, hashlib.sha256(evidence.read_bytes()).hexdigest()),
    }


def test_repeated_invocation_has_stable_logical_identity_and_distinct_attempt_roots(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    _write_chunk(raw_root)
    common = {
        "row_group_size": 2,
        "requested_workers": 1,
        "runs_root": tmp_path / "runs",
        "resource_ledger_path": tmp_path / "leases.json",
        "artifact_root": tmp_path / "artifacts",
        "registry_path": tmp_path / "registry.json",
        "invocation": {"raw_root": str(raw_root), "max_chunks": 1},
    }
    first_scan = scan_candidates(raw_root, parquet_root)
    first = execute_conversion_run(
        candidates=first_scan["candidates"], compatible_skips=[],
        convert_one=convert_one, options=ConversionComputeOptions(**common),
    )
    first_root = Path(first["run_root"])
    first_evidence = {
        name: (first_root / name).read_bytes()
        for name in ("run_manifest.json", "run_status.json", "results.json")
    }
    second_scan = scan_candidates(raw_root, parquet_root)
    second = execute_conversion_run(
        candidates=[], compatible_skips=second_scan["compatible_skips"],
        convert_one=convert_one, options=ConversionComputeOptions(**common),
    )

    first_manifest = json.loads((Path(first["run_root"]) / "run_manifest.json").read_text())
    second_manifest = json.loads((Path(second["run_root"]) / "run_manifest.json").read_text())
    assert first_manifest["campaign_identity"] == second_manifest["campaign_identity"]
    assert first_manifest["run_id"].endswith("attempt-0001")
    assert second_manifest["run_id"].endswith("attempt-0002")
    assert first["run_root"] != second["run_root"]
    assert first_evidence == {
        name: (first_root / name).read_bytes()
        for name in ("run_manifest.json", "run_status.json", "results.json")
    }
    registry = json.loads((tmp_path / "registry.json").read_text())
    assert len(registry["runs"]) == 2


def test_concurrent_identical_invocations_claim_distinct_attempts_and_registry_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root, parquet_root = tmp_path / "raw", tmp_path / "parquet"
    _write_chunk(raw_root)
    candidate = find_candidates(raw_root, parquet_root)[0]
    assert convert_one(candidate, row_group_size=2)["status"] == "converted"
    compatible = scan_candidates(raw_root, parquet_root)["compatible_skips"]
    options = _options(tmp_path, raw_root)
    barrier = Barrier(2)
    original_claim = compute_execution._claim_attempt_run_id

    def synchronized_claim(runs_root, logical_run_id):
        barrier.wait(timeout=5)
        return original_claim(runs_root, logical_run_id)

    monkeypatch.setattr(compute_execution, "_claim_attempt_run_id", synchronized_claim)

    def invoke():
        return execute_conversion_run(
            candidates=[], compatible_skips=compatible,
            convert_one=convert_one, options=options,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda _index: invoke(), range(2)))

    manifests = [
        json.loads((Path(result["run_root"]) / "run_manifest.json").read_text())
        for result in (first, second)
    ]
    assert manifests[0]["campaign_identity"] == manifests[1]["campaign_identity"]
    assert manifests[0]["run_id"] != manifests[1]["run_id"]
    assert {row["run_id"].rsplit("-", 1)[-1] for row in manifests} == {"0001", "0002"}
    assert first["run_root"] != second["run_root"]
    assert all(Path(result["run_root"]).is_dir() for result in (first, second))
    assert all((Path(result["run_root"]) / "run_status.json").exists() for result in (first, second))
    registry = json.loads((tmp_path / "registry.json").read_text())
    assert {row["run_id"] for row in registry["runs"]} == {
        manifests[0]["run_id"], manifests[1]["run_id"],
    }


def test_failed_initialisation_claim_is_never_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root, parquet_root = tmp_path / "raw", tmp_path / "parquet"
    _write_chunk(raw_root)
    candidate = find_candidates(raw_root, parquet_root)[0]
    assert convert_one(candidate, row_group_size=2)["status"] == "converted"
    compatible = scan_candidates(raw_root, parquet_root)["compatible_skips"]
    options = _options(tmp_path, raw_root)
    original_initialise = compute_execution.initialise_run

    monkeypatch.setattr(
        compute_execution, "initialise_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("initialisation failed")),
    )
    with pytest.raises(RuntimeError, match="initialisation failed"):
        execute_conversion_run(
            candidates=[], compatible_skips=compatible,
            convert_one=convert_one, options=options,
        )
    claimed = next((tmp_path / "runs").rglob("*attempt-0001"))
    assert claimed.is_dir()

    monkeypatch.setattr(compute_execution, "initialise_run", original_initialise)
    later = execute_conversion_run(
        candidates=[], compatible_skips=compatible,
        convert_one=convert_one, options=options,
    )
    assert Path(later["run_root"]).name.endswith("attempt-0002")
    assert claimed.is_dir()


def test_telemetry_start_failure_releases_lease_and_fails_run(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    _write_chunk(raw_root)
    scan = scan_candidates(raw_root, parquet_root)

    def broken_telemetry():
        ledger = json.loads((tmp_path / "leases.json").read_text())
        assert ledger["active_leases"][0]["status"] == "ACTIVE"
        raise RuntimeError("telemetry unavailable")

    execution = execute_conversion_run(
        candidates=scan["candidates"],
        compatible_skips=[],
        convert_one=convert_one,
        options=ConversionComputeOptions(
            row_group_size=2,
            requested_workers=2,
            runs_root=tmp_path / "runs",
            resource_ledger_path=tmp_path / "leases.json",
            artifact_root=tmp_path / "artifacts",
            registry_path=tmp_path / "registry.json",
            invocation={"raw_root": str(raw_root)},
            telemetry_factory=broken_telemetry,
        ),
    )

    assert "telemetry unavailable" in execution["error"]
    ledger = json.loads((tmp_path / "leases.json").read_text(encoding="utf-8"))
    assert ledger["active_leases"] == []
    status = json.loads((Path(execution["run_root"]) / "run_status.json").read_text())
    assert status["current_status"] == "FAILED"


def test_lease_acquisition_failure_never_starts_telemetry_or_pool(tmp_path: Path) -> None:
    raw_root, parquet_root = tmp_path / "raw", tmp_path / "parquet"
    _write_chunk(raw_root)
    scan = scan_candidates(raw_root, parquet_root)
    calls = []

    class DeniedLedger:
        def request_persisted_lease(self, _request):
            raise RuntimeError("lease denied")

    def ledger_factory(**_kwargs):
        return DeniedLedger()

    def telemetry_factory():
        calls.append("telemetry")
        raise AssertionError

    def pool_factory(**_kwargs):
        calls.append("pool")
        raise AssertionError

    execution = execute_conversion_run(
        candidates=scan["candidates"], compatible_skips=[], convert_one=convert_one,
        options=ConversionComputeOptions(
            **{
                **_options(tmp_path, raw_root).__dict__,
                "ledger_factory": ledger_factory,
                "telemetry_factory": telemetry_factory,
                "executor_factory": pool_factory,
            }
        ),
    )

    assert "lease denied" in execution["error"]
    assert calls == []
    assert json.loads((Path(execution["run_root"]) / "run_status.json").read_text())["current_status"] == "FAILED"


def test_pool_is_capped_and_constructed_only_after_active_lease(
    tmp_path: Path,
) -> None:
    raw_root, parquet_root = tmp_path / "raw", tmp_path / "parquet"
    _write_chunk(raw_root, batch="AAPL")
    _write_chunk(raw_root, batch="MSFT")
    scan = scan_candidates(raw_root, parquet_root)
    observed = {}

    class RecordingPool:
        def __init__(self, *, max_workers):
            ledger = json.loads((tmp_path / "leases.json").read_text())
            observed["lease_status"] = ledger["active_leases"][0]["status"]
            observed["max_workers"] = max_workers

        def submit(self, function, *args, **kwargs):
            future = Future()
            future.set_result(function(*args, **kwargs))
            return future

        def shutdown(self, **_kwargs):
            observed["stopped"] = True

    execution = execute_conversion_run(
        candidates=scan["candidates"], compatible_skips=[], convert_one=convert_one,
        options=_options(tmp_path, raw_root, requested_workers=99, executor_factory=RecordingPool),
    )

    assert execution["summary"]["converted_count"] == 2
    assert observed == {"lease_status": "ACTIVE", "max_workers": 2, "stopped": True}


def test_pool_construction_failure_stops_telemetry_and_releases_lease(
    tmp_path: Path,
) -> None:
    raw_root, parquet_root = tmp_path / "raw", tmp_path / "parquet"
    _write_chunk(raw_root, batch="AAPL")
    _write_chunk(raw_root, batch="MSFT")
    scan = scan_candidates(raw_root, parquet_root)

    def broken_pool(**_kwargs):
        raise RuntimeError("pool construction failed")

    execution = execute_conversion_run(
        candidates=scan["candidates"], compatible_skips=[], convert_one=convert_one,
        options=_options(tmp_path, raw_root, requested_workers=2, executor_factory=broken_pool),
    )

    assert "pool construction failed" in execution["error"]
    assert (Path(execution["run_root"]) / "resource_summary.json").exists()
    ledger = json.loads((tmp_path / "leases.json").read_text())
    assert ledger["active_leases"] == []
    assert json.loads((Path(execution["run_root"]) / "run_status.json").read_text())["current_status"] == "FAILED"


def test_mixed_skip_and_conversion_and_worker_failure_are_item_scoped(
    tmp_path: Path,
) -> None:
    raw_root, parquet_root = tmp_path / "raw", tmp_path / "parquet"
    _write_chunk(raw_root, batch="AAPL")
    _write_chunk(raw_root, batch="MSFT")
    aapl = next(row for row in find_candidates(raw_root, parquet_root) if "AAPL" in row["source_path"])
    assert convert_one(aapl, 2)["status"] == "converted"
    scan = scan_candidates(raw_root, parquet_root)

    def failed_worker(candidate, _row_group_size, **_kwargs):
        return {
            "status": "failed", "source_path": candidate["source_path"],
            "parquet_path": candidate["parquet_path"], "errors": ["fixture failure"],
        }

    execution = execute_conversion_run(
        candidates=scan["candidates"], compatible_skips=scan["compatible_skips"],
        convert_one=failed_worker, options=_options(tmp_path, raw_root),
    )

    assert execution["summary"]["compatible_skip_count"] == 1
    assert execution["summary"]["failed_count"] == 1
    status_rows = (Path(execution["run_root"]) / "component_status.csv").read_text()
    assert "SKIPPED_COMPATIBLE" in status_rows
    assert "FAILED" in status_rows


def test_artifact_publication_failure_preserves_authoritative_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root, parquet_root = tmp_path / "raw", tmp_path / "parquet"
    chunk = _write_chunk(raw_root)
    scan = scan_candidates(raw_root, parquet_root)
    monkeypatch.setattr(
        compute_execution, "publish_artifact_package",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("artifact publication failed")),
    )

    execution = execute_conversion_run(
        candidates=scan["candidates"], compatible_skips=[], convert_one=convert_one,
        options=_options(tmp_path, raw_root),
    )

    assert "artifact publication failed" in execution["error"]
    assert Path(scan["candidates"][0]["parquet_path"]).exists()
    assert (chunk / "parquet_conversion.json").exists()
    assert (chunk / "normalized_rows.json").exists()
    assert json.loads((tmp_path / "leases.json").read_text())["active_leases"] == []


def test_registry_publication_failure_is_visible_after_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root, parquet_root = tmp_path / "raw", tmp_path / "parquet"
    _write_chunk(raw_root)
    scan = scan_candidates(raw_root, parquet_root)
    monkeypatch.setattr(
        compute_execution, "update_global_registry_snapshot",
        lambda *_args, **_kwargs: {"health": "DEGRADED_REGISTRY", "error": "registry locked"},
    )

    execution = execute_conversion_run(
        candidates=scan["candidates"], compatible_skips=[], convert_one=convert_one,
        options=_options(tmp_path, raw_root),
    )

    assert "registry locked" in execution["error"]
    assert (Path(execution["run_root"]) / "publication_failure.json").exists()
    assert json.loads((tmp_path / "leases.json").read_text())["active_leases"] == []


def test_keyboard_interrupt_cancels_items_and_releases_lease(tmp_path: Path) -> None:
    raw_root, parquet_root = tmp_path / "raw", tmp_path / "parquet"
    chunk = _write_chunk(raw_root)
    scan = scan_candidates(raw_root, parquet_root)

    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute_conversion_run(
            candidates=scan["candidates"], compatible_skips=[],
            convert_one=interrupted, options=_options(tmp_path, raw_root),
        )

    assert json.loads((tmp_path / "leases.json").read_text())["active_leases"] == []
    assert (chunk / "normalized_rows.json").exists()
    status_file = next((tmp_path / "runs").rglob("run_status.json"))
    assert json.loads(status_file.read_text())["current_status"] == "CANCELLED"


def test_delete_flag_is_rejected_before_discovery_or_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "normalized_rows.json"
    source.write_text("[]")
    monkeypatch.setattr(
        conversion_script, "scan_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("discovery must not run")),
    )
    monkeypatch.setattr(
        sys, "argv",
        ["convert", "--execute", "--delete-source-json-after-validation"],
    )
    with pytest.raises(SystemExit, match="source JSON is always preserved"):
        conversion_script.main()
    assert source.read_text() == "[]"


def test_candidate_discovery_failure_occurs_before_compute_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        conversion_script, "scan_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("discovery failed")),
    )
    monkeypatch.setattr(sys, "argv", ["convert", "--execute"])
    with pytest.raises(RuntimeError, match="discovery failed"):
        conversion_script.main()


def _options(
    tmp_path: Path,
    raw_root: Path,
    *,
    requested_workers: int = 1,
    executor_factory=compute_execution.ProcessPoolExecutor,
) -> ConversionComputeOptions:
    return ConversionComputeOptions(
        row_group_size=2,
        requested_workers=requested_workers,
        runs_root=tmp_path / "runs",
        resource_ledger_path=tmp_path / "leases.json",
        artifact_root=tmp_path / "artifacts",
        registry_path=tmp_path / "registry.json",
        invocation={"raw_root": str(raw_root), "max_chunks": 0},
        executor_factory=executor_factory,
    )


def _write_chunk(
    raw_root: Path,
    *,
    batch: str = "AAPL",
    suffix: str = "",
    completion_state: str = "completed",
    canonical_symbols: list[str] | None = None,
    provider_symbol_map: dict[str, str] | None = None,
    row_symbol: str | None = None,
    provider_symbol: str | None = None,
) -> Path:
    chunk = raw_root / "sip" / "5m" / batch / f"20260102T143000Z_20260102T150000Z{suffix}"
    chunk.mkdir(parents=True, exist_ok=True)
    row_symbol = row_symbol or batch
    rows = [
        _row(row_symbol, "2026-01-02 14:30:00+00:00", provider_symbol=provider_symbol),
        _row(row_symbol, "2026-01-02 14:35:00+00:00", provider_symbol=provider_symbol),
    ]
    manifest = {
        "provider": "alpaca",
        "feed": "sip",
        "timeframe_requested": "5m",
        "native_timeframe": "5Min",
        "symbol_batch": batch.split("-"),
        "requested_start": "2026-01-02T14:30:00+00:00",
        "requested_end": "2026-01-02T15:00:00+00:00",
        "row_count": len(rows),
        "page_count": 1,
        "collection_timestamp": "2026-01-02T15:01:00+00:00",
        "adjustment_mode": "all",
        "session_policy": "regular_session_default",
        "normalizer_version": "historical_bar_provider_v1",
        "completion_state": completion_state,
    }
    if canonical_symbols is not None:
        manifest["canonical_symbol_batch"] = canonical_symbols
    if provider_symbol_map is not None:
        manifest["provider_symbol_map"] = provider_symbol_map
    (chunk / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (chunk / "normalized_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    (chunk / "provider_pages.json").write_text(json.dumps([{"bars": {}}]), encoding="utf-8")
    return chunk


def _row(symbol: str, timestamp: str, *, provider_symbol: str | None = None) -> dict:
    row = {
        "symbol": symbol,
        "timestamp": timestamp,
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "volume": 100.0,
        "trade_count": 1,
        "vwap": 10.25,
        "provider": "alpaca",
        "feed": "sip",
        "collection_timestamp": "2026-01-02T15:01:00+00:00",
        "requested_timeframe": "5m",
        "native_timeframe": "5Min",
        "adjustment_mode": "all",
        "extended_hours": False,
        "session_policy": "all_returned_bars_preserved",
        "session_type": "rth",
        "raw_chunk_identifier": f"chunk-{symbol}",
        "normalizer_version": "historical_bar_provider_v1",
    }
    if provider_symbol:
        row["provider_symbol"] = provider_symbol
    return row
