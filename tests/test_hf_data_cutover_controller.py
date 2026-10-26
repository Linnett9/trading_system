from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import application.services.hf_data_cutover_feature_validation as feature_validation
import application.services.hf_data_cutover_controller as controller_module
import application.services.hf_data_cutover_iex_backfill as backfill_module
from application.services.hf_data_cutover_controller import (
    STAGES,
    ControllerLease,
    ProcessIdentity,
    clear_control_requests,
    controller_status,
    freeze_operational_configuration,
    raw_backfill_assessment,
    request_control,
)
from application.services.hf_data_cutover_controller_validation import (
    StageGateError,
    validate_feature_quality,
    validate_raw_authority,
)
from application.services.hf_data_cutover_iex_backfill import RawBackfillDisposition, assess_raw_backfill_manifests
from application.services.hf_data_cutover_service import CutoverPaths, atomic_json
from core.research.iex_market_data import IexPartitionRequest
from infrastructure.data.iex_market_data_store import sha256_file


def _json(path: Path, payload: dict[str, object]) -> None:
    atomic_json(path, payload)


def test_stage_graph_is_fixed_and_daily_targets_require_adjusted_authority() -> None:
    assert [stage.command for stage in STAGES] == [
        "backfill",
        "adjust-daily",
        "derive-1h",
        "features",
        "feature-validate",
        "targets",
        "validate",
    ]
    targets = next(stage for stage in STAGES if stage.command == "targets")
    assert targets.prerequisites == ("raw_iex", "adjusted_daily", "feature_quality")
    assert STAGES[0].provider_retryable and STAGES[1].provider_retryable
    assert not any(stage.provider_retryable for stage in STAGES[2:])


def test_controller_lease_is_singleton_and_pid_reuse_safe(tmp_path: Path) -> None:
    lease_path = tmp_path / "controller.lock"
    first = ControllerLease(lease_path)
    first.acquire()
    with pytest.raises(RuntimeError, match="already active"):
        ControllerLease(lease_path).acquire()
    first.release()
    _json(lease_path, {"pid": 999999, "create_time": 1.0})
    replacement = ControllerLease(lease_path)
    replacement.acquire()
    assert json.loads(lease_path.read_text(encoding="utf-8"))["pid"] == ProcessIdentity.current().pid
    replacement.release()


def test_status_is_read_only_and_control_requests_are_explicit(tmp_path: Path) -> None:
    audit = tmp_path / "HF_DATA_CUTOVER_R1_TEST"
    control = audit / "controller"
    control.mkdir(parents=True)
    _json(audit / "RUN_MANIFEST.json", {"state": "IEX_RAW_BACKFILL_IN_PROGRESS"})
    _json(control / "CONTROLLER_STATE.json", {"status": "RUNNING", "current_stage": "raw_iex"})
    before = {path: path.read_bytes() for path in audit.rglob("*") if path.is_file()}
    status = controller_status(audit)
    after = {path: path.read_bytes() for path in audit.rglob("*") if path.is_file()}
    assert status["status"] == "RUNNING"
    assert status["run_state"] == "IEX_RAW_BACKFILL_IN_PROGRESS"
    assert before == after
    pause = request_control(audit, "pause")
    assert json.loads(pause.read_text(encoding="utf-8"))["action"] == "PAUSE"
    assert clear_control_requests(audit)
    assert not pause.exists()


def test_operational_freeze_pins_validated_concurrency_limiter_and_authorities(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    audit = repo / "audit"
    audit.mkdir(parents=True)
    _json(
        audit / "RUN_MANIFEST.json",
        {
            "iex_raw_backfill": {
                "max_in_flight_requests": 32,
                "requests_per_minute": 150,
                "minimum_final_reserve_bytes": 15 * 1024**3,
                "plan_identity": "plan",
                "planned_5m_partitions": 2,
                "planned_daily_partitions": 1,
            }
        },
    )
    for name in (
        "IEX_HISTORICAL_COVERAGE_PROBE.json",
        "IEX_ACQUISITION_ROSTER.json",
        "IEX_SOURCE_USE_AUTHORITY.json",
        "IEX_FEATURE_BANK_V1_SPEC.json",
    ):
        _json(audit / name, {"name": name})
    frozen = freeze_operational_configuration(repo, audit)
    assert frozen["backfill"]["max_in_flight_requests"] == 32
    assert frozen["backfill"]["shared_requests_per_minute"] == 150
    assert frozen["scientific_authorities_refrozen"] is False
    recovery = json.loads(
        (audit / "IEX_TRANSPORT_RECOVERY_CONFIGURATION.json").read_text(encoding="utf-8")
    )
    assert recovery["initial_validated_max_in_flight_requests"] == 32
    assert recovery["recovery_max_in_flight_requests"] == 8
    assert recovery["transport_probe_workers"] == 1
    assert recovery["shared_requests_per_minute_ceiling"] == 150
    assert recovery["pacing_requests_per_minute"] == 60
    assert recovery["zero_progress_invocation_limit"] == 3
    _json(audit / "IEX_SOURCE_USE_AUTHORITY.json", {"name": "drift"})
    with pytest.raises(StageGateError, match="drifted"):
        freeze_operational_configuration(repo, audit)
    (audit / "IEX_SOURCE_USE_AUTHORITY.json").unlink()
    _json(audit / "IEX_SOURCE_AUTHORITY.json", {"name": "wrong legacy filename"})
    with pytest.raises(StageGateError, match="IEX_SOURCE_USE_AUTHORITY.json"):
        freeze_operational_configuration(repo, audit)


def _raw_manifest(*, planned: int, completed: int, failures: list[dict[str, object]]) -> dict[str, object]:
    return {
        "planned_partition_count": planned,
        "completed_partition_count": completed,
        "failures": failures,
    }


def test_legacy_ssl_failures_and_missing_partitions_are_resumable_not_fatal(tmp_path: Path) -> None:
    legacy_ssl = {
        "classification": "unexpected_failure",
        "error_type": "URLError",
        "error_message": "<urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol>",
    }
    assessment = assess_raw_backfill_manifests(
        _raw_manifest(planned=2, completed=1, failures=[legacy_ssl]),
        _raw_manifest(planned=1, completed=0, failures=[{"classification": "missing_unattempted"}]),
    )
    assert assessment == {
        "disposition": RawBackfillDisposition.RESUMABLE_INCOMPLETE.value,
        "planned_partition_count": 3,
        "completed_partition_count": 1,
        "unresolved_partition_count": 2,
        "resumable_incomplete_partition_count": 2,
        "irrecoverable_failure_partition_count": 0,
        "classification_counts": {
            "legacy_transient_network_failure": 1,
            "missing_unattempted": 1,
        },
    }
    audit = tmp_path / "audit"
    _json(audit / "IEX_RAW_5M_MANIFEST.json", _raw_manifest(planned=2, completed=1, failures=[legacy_ssl]))
    _json(audit / "IEX_DAILY_MANIFEST.json", _raw_manifest(planned=1, completed=0, failures=[{"classification": "missing_unattempted"}]))
    assert raw_backfill_assessment(audit)["disposition"] == RawBackfillDisposition.RESUMABLE_INCOMPLETE.value


def test_genuine_raw_integrity_failure_remains_fail_closed() -> None:
    assessment = assess_raw_backfill_manifests(
        _raw_manifest(
            planned=1,
            completed=0,
            failures=[{"classification": "integrity_failure", "error_type": "ValueError", "error_message": "hash mismatch"}],
        )
    )
    assert assessment["disposition"] == RawBackfillDisposition.IRRECOVERABLE_FAILURE.value
    assert assessment["irrecoverable_failure_partition_count"] == 1


def test_controller_automatically_continues_resumable_raw_invocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    audit = repo / "audit"
    repo.mkdir()
    audit.mkdir()
    resumable = _raw_manifest(
        planned=1,
        completed=0,
        failures=[{"classification": "transient_network_failure", "error_type": "URLError", "error_message": "TLS EOF"}],
    )
    _json(audit / "IEX_RAW_5M_MANIFEST.json", resumable)
    _json(audit / "IEX_DAILY_MANIFEST.json", _raw_manifest(planned=0, completed=0, failures=[]))
    _json(audit / "RUN_MANIFEST.json", {"state": "IEX_RAW_DATA_INCOMPLETE", "iex_raw_backfill": {}})
    controller = controller_module.CutoverController(repo, audit)
    stage = controller_module.Stage("raw_iex", "backfill", lambda _repo, _audit: {"status": "COMPLETE"}, True)
    return_codes = iter((1, 0))
    launches: list[list[str]] = []

    class FakeProcess:
        pid = 123

    def fake_popen(command: list[str], **_kwargs: object) -> FakeProcess:
        launches.append(command)
        return FakeProcess()

    monkeypatch.setattr(controller_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(controller_module.ProcessIdentity, "from_pid", classmethod(lambda _cls, pid: controller_module.ProcessIdentity(pid, 1.0)))
    monkeypatch.setattr(controller, "_wait_for_process", lambda *_args, **_kwargs: next(return_codes))
    monkeypatch.setattr(controller, "_discover_stage_process", lambda _stage: None)
    monkeypatch.setattr(controller, "_assert_reserve", lambda *args, **kwargs: None)
    monkeypatch.setattr(controller, "_control_request", lambda: None)
    monkeypatch.setattr(controller, "_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(controller, "_write_attempt_count", lambda *args, **kwargs: None)
    monkeypatch.setattr(controller, "_attempt_count", lambda _stage: 0)
    monkeypatch.setattr(controller, "_interruptible_wait", lambda *_args, **_kwargs: None)
    assert controller._execute_stage(stage) == {"status": "COMPLETE"}
    assert len(launches) == 2


def test_controller_blocks_genuine_raw_failure_without_relaunch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    audit = repo / "audit"
    repo.mkdir()
    audit.mkdir()
    fatal = _raw_manifest(
        planned=1,
        completed=0,
        failures=[{"classification": "validation_failure", "error_type": "ValueError", "error_message": "schema drift"}],
    )
    _json(audit / "IEX_RAW_5M_MANIFEST.json", fatal)
    _json(audit / "IEX_DAILY_MANIFEST.json", _raw_manifest(planned=0, completed=0, failures=[]))
    controller = controller_module.CutoverController(repo, audit)
    stage = controller_module.Stage("raw_iex", "backfill", lambda _repo, _audit: {}, True)
    launches = 0

    class FakeProcess:
        pid = 123

    def fake_popen(_command: list[str], **_kwargs: object) -> FakeProcess:
        nonlocal launches
        launches += 1
        return FakeProcess()

    monkeypatch.setattr(controller_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(controller_module.ProcessIdentity, "from_pid", classmethod(lambda _cls, pid: controller_module.ProcessIdentity(pid, 1.0)))
    monkeypatch.setattr(controller, "_wait_for_process", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(controller, "_discover_stage_process", lambda _stage: None)
    monkeypatch.setattr(controller, "_assert_reserve", lambda *args, **kwargs: None)
    monkeypatch.setattr(controller, "_control_request", lambda: None)
    monkeypatch.setattr(controller, "_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(controller, "_write_attempt_count", lambda *args, **kwargs: None)
    monkeypatch.setattr(controller, "_attempt_count", lambda _stage: 0)
    with pytest.raises(StageGateError, match="irrecoverable=1"):
        controller._execute_stage(stage)
    assert launches == 1


def test_three_repeated_zero_progress_cycles_open_transport_circuit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    audit = repo / "audit"
    repo.mkdir()
    audit.mkdir()
    controller = controller_module.CutoverController(repo, audit)
    controller._reset_raw_progress(2443, source="test_baseline")
    assessment = {"completed_partition_count": 2443}

    first = controller._record_raw_progress(assessment, source="backfill")
    second = controller._record_raw_progress(assessment, source="backfill")
    third = controller._record_raw_progress(assessment, source="backfill")

    assert not first["circuit_open"]
    assert not second["circuit_open"]
    assert third["circuit_open"]
    assert third["consecutive_zero_progress_invocations"] == 3
    assert controller._raw_recovery_required()


def test_sequential_transport_probe_recovers_only_incomplete_partitions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    audit = repo / "audit"
    repo.mkdir()
    audit.mkdir()
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 1, tzinfo=timezone.utc)
    completed = IexPartitionRequest("DONE", "5m", start, end, year=2024)
    pending = tuple(
        IexPartitionRequest(symbol, "5m", start, end, year=2024)
        for symbol in ("A", "B", "C")
    )
    _json(
        audit / "RUN_MANIFEST.json",
        {"state": "IEX_RAW_DATA_INCOMPLETE", "iex_raw_backfill": {"plan_identity": "plan"}},
    )
    settings = backfill_module.IexBackfillSettings(start, end, ("A",), repo / "data", ())
    provider_instances: list[object] = []
    requested: list[IexPartitionRequest] = []

    class FakeStore:
        def completed_manifest(self, request: IexPartitionRequest) -> dict[str, object] | None:
            return {"status": "COMPLETE"} if request == completed else None

        def write_failure(self, *_args: object) -> None:
            raise AssertionError("successful probe must not publish failure evidence")

    class FakeProvider:
        def __init__(self, **kwargs: object) -> None:
            provider_instances.append(self)
            assert kwargs["opener"] is not None
            self.metrics = SimpleNamespace(as_dict=lambda: {"requests_attempted": 1})

        @staticmethod
        def check_authentication() -> dict[str, bool]:
            return {"can_attempt_authenticated_request": True}

    inventories = iter((( [completed], list(pending)), ([completed, *pending], [])))
    monkeypatch.setattr(backfill_module, "load_iex_backfill_settings", lambda _paths: settings)
    monkeypatch.setattr(backfill_module, "_preflight", lambda *_args: None)
    monkeypatch.setattr(backfill_module, "IexMarketDataStore", lambda _root: FakeStore())
    monkeypatch.setattr(backfill_module, "AlpacaBasicHistoricalBarProvider", FakeProvider)
    monkeypatch.setattr(backfill_module, "plan_iex_partitions", lambda *_args, **_kwargs: pending)
    monkeypatch.setattr(backfill_module, "_plan_identity", lambda _plans: "plan")
    monkeypatch.setattr(backfill_module, "_validated_partition_inventory", lambda *_args: next(inventories))
    monkeypatch.setattr(backfill_module, "_completed_manifest_set_sha256", lambda *_args: "a" * 64)

    def collect_batch(**kwargs: object) -> list[tuple[IexPartitionRequest, dict[str, object]]]:
        requests = kwargs["requests"]
        assert isinstance(requests, tuple) and len(requests) == 1
        request = requests[0]
        assert request != completed
        requested.append(request)
        return [(request, {"parquet_sha256": "b" * 64, "quality": {"row_count": 1}})]

    def publish(**_kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        manifest = json.loads((audit / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
        manifest["state"] = "IEX_RAW_DATA_INCOMPLETE"
        _json(audit / "RUN_MANIFEST.json", manifest)
        return manifest, {}

    monkeypatch.setattr(backfill_module, "_collect_batch", collect_batch)
    monkeypatch.setattr(backfill_module, "_publish_raw_invocation", publish)
    paths = CutoverPaths(repo, audit, tmp_path / "r1", tmp_path / "r2")
    report = backfill_module.run_iex_transport_connectivity_probe(paths)

    assert report["status"] == "PASSED"
    assert report["probe_workers"] == 1
    assert requested == list(pending)
    assert len(provider_instances) == 3
    assert report["completed_partition_count_before"] == 1
    assert report["completed_partition_count_after"] == 4


def test_transport_probe_fatal_failure_remains_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FatalAuthFailure(RuntimeError):
        classification = "authentication_failure"

    monkeypatch.setattr(
        backfill_module,
        "_exception_classification",
        lambda _exc: FatalAuthFailure.classification,
    )
    assert backfill_module._exception_classification(FatalAuthFailure()) not in (
        backfill_module._RETRYABLE_FAILURE_CLASSIFICATIONS
    )


def test_stale_resumable_manifest_cannot_mask_current_invocation_abort(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    audit = repo / "audit"
    repo.mkdir()
    audit.mkdir()
    controller = controller_module.CutoverController(repo, audit)
    assessment = {
        "disposition": RawBackfillDisposition.RESUMABLE_INCOMPLETE.value,
        "unresolved_partition_count": 1,
        "irrecoverable_failure_partition_count": 0,
    }
    _json(audit / "RUN_MANIFEST.json", {"state": "IEX_RAW_BACKFILL_IN_PROGRESS", "iex_raw_backfill": {}})
    assert not controller._raw_stage_is_resumable(assessment)
    _json(
        audit / "RUN_MANIFEST.json",
        {
            "state": "IEX_RAW_DATA_INCOMPLETE",
            "iex_raw_backfill": {"completion_disposition": RawBackfillDisposition.RESUMABLE_INCOMPLETE.value},
        },
    )
    assert controller._raw_stage_is_resumable(assessment)


def test_collect_all_never_redownloads_completed_partitions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 1, tzinfo=timezone.utc)
    completed = IexPartitionRequest("A", "5m", start, end, year=2024)
    pending = IexPartitionRequest("B", "5m", start, end, year=2024)
    requested: list[IexPartitionRequest] = []

    class FakeStore:
        root = tmp_path

        @staticmethod
        def completed_manifest(request: IexPartitionRequest) -> dict[str, object] | None:
            return {"quality": {"row_count": 10}} if request == completed else None

        @staticmethod
        def write_failure(_request: IexPartitionRequest, _failure: dict[str, object]) -> None:
            raise AssertionError("no failure expected")

    def fake_collect_batch(**kwargs: object) -> list[tuple[IexPartitionRequest, dict[str, object]]]:
        batch = kwargs["requests"]
        assert isinstance(batch, tuple)
        requested.extend(batch)
        return [(request, {"quality": {"row_count": 5}}) for request in batch]

    monkeypatch.setattr(backfill_module, "_collect_batch", fake_collect_batch)
    summary = backfill_module._collect_all(
        store=FakeStore(),
        provider=SimpleNamespace(metrics=SimpleNamespace(requests_attempted=0, http_429_count=0, as_dict=lambda: {})),
        plans={"5m": (completed, pending), "1Day": ()},
        provider_symbol_map={},
    )
    assert requested == [pending]
    assert summary["resumed_partitions"] == 1
    assert summary["completed_partitions"] == 2


def test_raw_gate_requires_top_level_and_partition_hash_identity(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    audit = repo / "audit"
    five_partition = repo / "data/market/iex/raw/5m/symbol=A/year=2024/manifest.json"
    daily_partition = repo / "data/market/iex/raw/1d/symbol=A/manifest.json"
    _json(five_partition, {"status": "COMPLETE", "id": "five"})
    _json(daily_partition, {"status": "COMPLETE", "id": "daily"})
    five = {
        "dataset": "raw_5m",
        "feed": "IEX",
        "adjustment": "raw",
        "planned_partition_count": 1,
        "completed_partition_count": 1,
        "failed_partition_count": 0,
        "duplicate_count": 0,
        "off_grid_count": 0,
        "partition_manifest_sha256": [sha256_file(five_partition)],
    }
    daily = {**five, "dataset": "raw_daily", "partition_manifest_sha256": [sha256_file(daily_partition)]}
    _json(audit / "IEX_RAW_5M_MANIFEST.json", five)
    _json(audit / "IEX_DAILY_MANIFEST.json", daily)
    _json(
        audit / "RUN_MANIFEST.json",
        {
            "state": "IEX_RAW_DATA_COMPLETE_FEATURE_BUILD_PENDING",
            "iex_raw_backfill": {
                "planned_5m_partitions": 1,
                "planned_daily_partitions": 1,
                "raw_5m_manifest_sha256": sha256_file(audit / "IEX_RAW_5M_MANIFEST.json"),
                "daily_manifest_sha256": sha256_file(audit / "IEX_DAILY_MANIFEST.json"),
            },
        },
    )
    assert validate_raw_authority(repo, audit)["partitions"] == 2
    _json(five_partition, {"status": "COMPLETE", "id": "corrupted"})
    with pytest.raises(StageGateError, match="hashes/counts drifted"):
        validate_raw_authority(repo, audit)


def test_pretarget_feature_validation_rejects_quality_drift_and_reads_no_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    audit = repo / "audit"
    feature_root = repo / "data/market/iex/features/v1"
    partition_path = feature_root / "5m/symbol=A/year=2024/manifest.json"
    parquet_path = partition_path.parent / "part.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.write_bytes(b"synthetic parquet identity")
    spec_hash = "s" * 64
    _json(audit / "RUN_MANIFEST.json", {"state": "IEX_FEATURE_V1_COMPLETE_TARGET_BUILD_PENDING"})
    _json(
        audit / "IEX_FEATURE_V1_MANIFEST.json",
        {
            "status": "COMPLETE",
            "missing_tables": [],
            "predictor_count": 80,
            "feature_spec_sha256": spec_hash,
            "targets_read_by_feature_pipeline": False,
            "raw_ohlcv_duplicated_in_feature_tables": False,
            "tables": {"5m": {"partitions": 1, "rows": 3, "bytes": 10}},
        },
    )
    partition = {
        "status": "COMPLETE",
        "identity": {"table": "5m", "feature_spec_sha256": spec_hash},
        "lineage": {"targets_read": False},
        "quality": {
            "duplicate_count": 0,
            "availability_violation_count": 0,
            "nonfinite_rate_by_feature": {"return": 0.0},
        },
        "row_count": 3,
        "parquet_bytes": 10,
        "parquet_path": str(parquet_path),
        "parquet_sha256": sha256_file(parquet_path),
    }
    _json(partition_path, partition)
    monkeypatch.setattr(
        feature_validation,
        "_feature_determinism_sentinel",
        lambda _paths: {"feature_exact": True, "sample": {"symbol": "AAPL", "year": 2024}},
    )
    paths = CutoverPaths(repo, audit, tmp_path / "r1", tmp_path / "r2")
    report = feature_validation.run_iex_feature_validation(paths)
    assert report["status"] == "PASSED" and report["target_data_read"] is False
    assert validate_feature_quality(repo, audit)["partitions"] == 1
    partition["quality"]["nonfinite_rate_by_feature"] = {"return": 0.1}
    _json(partition_path, partition)
    with pytest.raises(Exception, match="non-finite"):
        feature_validation.run_iex_feature_validation(paths)


def test_powershell_surface_is_detached_and_monitor_has_no_control_side_effect() -> None:
    text = Path("scripts/local/Invoke-HFDataCutoverR1.ps1").read_text(encoding="utf-8")
    assert 'ValidateSet("Start", "Status", "Monitor", "Pause", "Resume", "Stop", "Foreground")' in text
    assert "Start-Process" in text and "-WindowStyle Hidden" in text
    assert "RedirectStandardOutput" in text and "RedirectStandardError" in text
    assert "Start-Job" not in text
    monitor_block = text[text.index('"Monitor"'): text.index('"Pause"')]
    assert "stop" not in monitor_block.casefold() and "pause" not in monitor_block.casefold()
