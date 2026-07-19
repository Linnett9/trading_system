from __future__ import annotations

import hashlib
import inspect
import json
import os
import pickle
import subprocess
import sys
import tempfile
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import infrastructure.data.canonical_v2_alpha_enrichment as alpha
from application import cli_dispatch
from core.research.ml.stock_level.stock_level_alpha_features_types import (
    StockLevelAlphaFeaturePaths,
)


def _base_fixture(
    tmp_path: Path,
    *,
    provenance: str = alpha.TARGET_PROVENANCE_V2,
) -> tuple[dict[str, object], Path]:
    output = tmp_path / "benchmark"
    output.mkdir(parents=True)
    path = output / "stock_level_prediction_artifacts.parquet"
    rows = [
        {
            "rebalance_date": "2026-01-02",
            "symbol": "AAA",
            "asset_id": "asset-aaa",
            "decision_timestamp": "2026-01-02T20:05:00Z",
            "target_provenance_contract_version": provenance,
            "actual_forward_return_10d": 0.1,
        },
        {
            "rebalance_date": "2026-01-02",
            "symbol": "BBB",
            "asset_id": "asset-bbb",
            "decision_timestamp": "2026-01-02T20:05:00Z",
            "target_provenance_contract_version": provenance,
            "actual_forward_return_10d": 0.2,
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = {
        "canonical_artifact": {
            "completion_status": "complete",
            "resolved_artifact_path": str(path),
            "row_count": len(rows),
            "file_size_bytes": path.stat().st_size,
            "sha256": sha,
            "logical_content_sha256": sha,
        }
    }
    (output / "stock_level_prediction_artifacts.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    config: dict[str, object] = {
        "ml": {
            "output_dir": str(output),
            "stock_level_base_prediction_artifacts_path": str(path),
            "stock_level_artifact_format": "parquet",
            "canonical_v2_alpha_report_root": str(tmp_path / "alpha"),
            "stooq_parquet_dir": str(tmp_path / "prices"),
        }
    }
    return config, path


def test_certified_alpha_base_uses_metadata_and_projected_batches(tmp_path: Path) -> None:
    config, path = _base_fixture(tmp_path)

    result = alpha.validate_alpha_base_artifact(config)

    assert result["status"] == "VALID"
    assert result["row_count"] == 2
    assert result["full_table_materialized"] is False
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_certified_alpha_base_missing_corrupt_and_v1_fail_closed(tmp_path: Path) -> None:
    config, path = _base_fixture(tmp_path)
    path.unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        alpha.validate_alpha_base_artifact(config)

    corrupt_config, corrupt = _base_fixture(tmp_path / "corrupt")
    corrupt.write_bytes(b"not parquet")
    with pytest.raises(ValueError, match="metadata is unreadable"):
        alpha.validate_alpha_base_artifact(corrupt_config)

    v1_config, _ = _base_fixture(
        tmp_path / "v1", provenance="stock_level_target_provenance_v1"
    )
    with pytest.raises(ValueError, match="provenance"):
        alpha.validate_alpha_base_artifact(v1_config)

    blank_config, blank_path = _base_fixture(tmp_path / "blank")
    table = pq.read_table(blank_path)
    table = table.set_column(
        table.schema.get_field_index("target_provenance_contract_version"),
        "target_provenance_contract_version",
        pa.array([alpha.TARGET_PROVENANCE_V2, None], type=pa.string()),
    )
    pq.write_table(table, blank_path)
    sidecar_path = blank_path.with_name("stock_level_prediction_artifacts.json")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["canonical_artifact"]["file_size_bytes"] = blank_path.stat().st_size
    sidecar["canonical_artifact"]["sha256"] = hashlib.sha256(
        blank_path.read_bytes()
    ).hexdigest()
    sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="provenance"):
        alpha.validate_alpha_base_artifact(blank_config)


def test_base_partition_preparation_is_streaming_checksum_owned_and_reusable(
    tmp_path: Path,
) -> None:
    config, _ = _base_fixture(tmp_path)
    validation = alpha.validate_alpha_base_artifact(config)

    first = alpha.prepare_alpha_base_partitions(
        config, base_validation=validation
    )
    second = alpha.prepare_alpha_base_partitions(
        config, base_validation=validation
    )

    assert first == second
    assert first["source_base_sha256"] == validation["sha256"]
    assert first["partition_count"] == 2
    assert first["full_table_materialized"] is False
    assert all(Path(row["path"]).exists() for row in first["partitions"])


def test_base_partition_attempt_path_is_windows_safe_under_long_report_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_root = (
        tmp_path
        / "ticket_7b3_daily_large_history"
        / "regeneration_canonical_v2"
        / "alpha_enrichment"
    )
    config, _ = _base_fixture(tmp_path / "fixture")
    validation = alpha.validate_alpha_base_artifact(config)
    target_root = report_root / "alpha_base_partitions_v2" / validation["sha256"]
    attempt_root = alpha._alpha_base_attempt_root(target_root, validation["sha256"])
    configured_target = (
        Path(config["ml"]["canonical_v2_alpha_report_root"])
        / "alpha_base_partitions_v2"
        / validation["sha256"]
    )
    replacements: list[tuple[Path, Path]] = []
    real_replace = alpha.os.replace

    def record_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    monkeypatch.setattr(alpha.os, "replace", record_replace)
    result = alpha.prepare_alpha_base_partitions(config, base_validation=validation)

    assert len(str(attempt_root.resolve())) < 260
    assert validation["sha256"] not in attempt_root.name
    assert Path(result["path"]).name == validation["sha256"]
    assert replacements[-1][0].name.startswith(
        f".attempt-{validation['sha256'][:8]}-"
    )
    assert replacements[-1][1] == configured_target


def test_base_partition_interruption_cleans_attempt_and_can_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _base_fixture(tmp_path)
    validation = alpha.validate_alpha_base_artifact(config)
    real_replace = alpha.os.replace

    def interrupt_publication(source, destination):
        if Path(destination).name == validation["sha256"]:
            raise KeyboardInterrupt("synthetic interruption")
        return real_replace(source, destination)

    monkeypatch.setattr(alpha.os, "replace", interrupt_publication)
    with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
        alpha.prepare_alpha_base_partitions(config, base_validation=validation)

    partition_parent = (
        Path(config["ml"]["canonical_v2_alpha_report_root"])
        / "alpha_base_partitions_v2"
    )
    assert not list(partition_parent.glob(".attempt-*"))
    assert not (partition_parent / validation["sha256"]).exists()

    monkeypatch.setattr(alpha.os, "replace", real_replace)
    resumed = alpha.prepare_alpha_base_partitions(config, base_validation=validation)
    assert Path(resumed["path"]).is_dir()


def test_bounded_alpha_partition_namespace_is_deterministic_and_full_identity(
    tmp_path: Path,
) -> None:
    config, _ = _base_fixture(tmp_path)
    validation = alpha.validate_alpha_base_artifact(config)

    first = alpha._alpha_partition_namespace_identity(
        config, base_validation=validation
    )
    second = alpha._alpha_partition_namespace_identity(
        config, base_validation=validation
    )

    assert first == second
    assert len(first["namespace_key"]) == 20
    assert first["base_artifact_sha256"] == validation["sha256"]
    assert len(first["base_artifact_sha256"]) == 64
    assert first["feature_schema_sha256"] == alpha._feature_schema_identity()
    assert len(first["feature_schema_sha256"]) == 64
    assert len(first["configuration_sha256"]) == 64


def test_bounded_alpha_partition_namespace_stays_within_windows_path_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _base_fixture(tmp_path)
    validation = alpha.validate_alpha_base_artifact(config)
    replacements: list[tuple[Path, Path]] = []
    real_replace = alpha.os.replace

    with tempfile.TemporaryDirectory(prefix="alpha-ns-") as temporary:
        report_root = (
            Path(temporary)
            / "reports"
            / "ml"
            / "development"
            / "ticket_7b3_daily_large_history"
            / "regeneration_canonical_v2"
            / "alpha_enrichment"
        )

        def record_replace(source, destination):
            replacements.append((Path(source), Path(destination)))
            return real_replace(source, destination)

        monkeypatch.setattr(alpha.os, "replace", record_replace)
        namespace, identity = alpha._resolve_alpha_partition_namespace(
            report_root, config, base_validation=validation
        )
        deepest_new_paths = [
            namespace / "partitions" / "symbol=BRK.B" / "rows.parquet.tmp",
            namespace / "partition_manifests" / "BRK.B.json",
            namespace / "partition_failures" / "BRK.B.json",
            namespace / "schema_diagnostics" / "BRK.B.json",
        ]

        assert identity["layout"] == "bounded_v1"
        assert all(len(str(path.resolve())) < 240 for path in deepest_new_paths)
        assert all(len(str(source.resolve())) < 240 for source, _ in replacements)
        manifest = json.loads(
            (namespace / "namespace_manifest.json").read_text()
        )
        assert manifest["base_artifact_sha256"] == validation["sha256"]
        assert manifest["feature_schema_sha256"] == alpha._feature_schema_identity()


def test_bounded_alpha_partition_namespace_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    config, _ = _base_fixture(tmp_path)
    validation = alpha.validate_alpha_base_artifact(config)
    namespace, identity = alpha._resolve_alpha_partition_namespace(
        tmp_path / "report", config, base_validation=validation
    )
    manifest_path = namespace / "namespace_manifest.json"
    mismatched = {**identity, "base_artifact_sha256": "f" * 64}
    manifest_path.write_text(json.dumps(mismatched), encoding="utf-8")

    with pytest.raises(ValueError, match="namespace identity mismatch"):
        alpha._resolve_alpha_partition_namespace(
            tmp_path / "report", config, base_validation=validation
        )


def test_bounded_alpha_partition_namespace_resume_and_incompatible_identity(
    tmp_path: Path,
) -> None:
    config, _ = _base_fixture(tmp_path)
    validation = alpha.validate_alpha_base_artifact(config)
    first, first_identity = alpha._resolve_alpha_partition_namespace(
        tmp_path / "report", config, base_validation=validation
    )
    resumed, resumed_identity = alpha._resolve_alpha_partition_namespace(
        tmp_path / "report", config, base_validation=validation
    )
    incompatible = {**config, "ml": dict(config["ml"])}
    incompatible["ml"]["stock_level_parquet_compression"] = "snappy"
    other, other_identity = alpha._resolve_alpha_partition_namespace(
        tmp_path / "report", incompatible, base_validation=validation
    )

    assert resumed == first
    assert resumed_identity == first_identity
    assert other != first
    assert other_identity["configuration_sha256"] != first_identity[
        "configuration_sha256"
    ]


def test_bounded_alpha_partition_namespace_interruption_cleans_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _base_fixture(tmp_path)
    validation = alpha.validate_alpha_base_artifact(config)
    report_root = tmp_path / "report"
    real_replace = alpha.os.replace

    def interrupt(_source, _destination):
        raise KeyboardInterrupt("synthetic namespace interruption")

    monkeypatch.setattr(alpha.os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt, match="synthetic namespace interruption"):
        alpha._resolve_alpha_partition_namespace(
            report_root, config, base_validation=validation
        )

    namespace_parent = report_root / "alpha_partitions_v2"
    assert not list(namespace_parent.glob(".attempt-*"))
    assert not list(namespace_parent.glob("id-*"))

    monkeypatch.setattr(alpha.os, "replace", real_replace)
    resumed, _ = alpha._resolve_alpha_partition_namespace(
        report_root, config, base_validation=validation
    )
    assert resumed.is_dir()


def test_alpha_partition_namespace_reuses_existing_legacy_layout(
    tmp_path: Path,
) -> None:
    config, _ = _base_fixture(tmp_path)
    validation = alpha.validate_alpha_base_artifact(config)
    with tempfile.TemporaryDirectory(prefix="al-") as temporary:
        report_root = Path(temporary) / "r"
        legacy = (
            report_root
            / "alpha_partitions_v2"
            / validation["sha256"]
            / alpha._feature_schema_identity()
        )
        legacy.mkdir(parents=True)

        namespace, identity = alpha._resolve_alpha_partition_namespace(
            report_root, config, base_validation=validation
        )

        assert namespace == legacy
        assert identity["layout"] == "legacy_full_hash_v2"
        assert not list((report_root / "alpha_partitions_v2").glob("id-*"))


def test_subprocess_regression_avoids_production_style_full_hash_final_path() -> None:
    script = """
import json
import tempfile
from pathlib import Path
from infrastructure.data import canonical_v2_alpha_enrichment as alpha

config = {"ml": {
    "stock_alpha_feature_n_jobs": 1,
    "stock_level_artifact_format": "parquet",
    "stock_level_parquet_compression": "zstd",
}}
validation = {"sha256": "c2487d7f378121069ea5e92a1d0cf0444f42dfc1da237566d24c650ae8558d38"}
with tempfile.TemporaryDirectory(prefix="alpha-final-path-") as temporary:
    report = Path(temporary) / (
        "reports/ml/development/ticket_7b3_daily_large_history/"
        "regeneration_canonical_v2/alpha_enrichment"
    )
    legacy = (
        report / "alpha_partitions_v2" / validation["sha256"]
        / alpha._feature_schema_identity() / "partitions"
    )
    namespace, identity = alpha._resolve_alpha_partition_namespace(
        report, config, base_validation=validation
    )
    durable = namespace / "partitions" / "symbol=AAA" / "rows.parquet.tmp"
    print(json.dumps({
        "legacy": len(str(legacy.resolve())),
        "durable": len(str(durable.resolve())),
        "layout": identity["layout"],
    }))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["legacy"] >= 260
    assert observed["durable"] < 248
    assert observed["layout"] == "bounded_v1"


def test_worker_payload_is_small_and_excludes_unrelated_configuration(
    tmp_path: Path,
) -> None:
    config, _ = _base_fixture(tmp_path)
    config["ml"]["unrelated_large_value"] = "x" * 1_000_000

    payload = alpha._small_alpha_worker_config(config)

    assert "unrelated_large_value" not in payload["ml"]
    assert len(pickle.dumps(payload)) < 131_072


def test_worker_startup_failure_persists_original_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _base_fixture(tmp_path)
    worker_config = alpha._small_alpha_worker_config(config)

    def fail(*_args, **_kwargs):
        raise RuntimeError("original worker startup failure")

    monkeypatch.setattr(alpha, "_load_price_histories", fail)
    with pytest.raises(RuntimeError, match="original worker startup failure"):
        alpha._alpha_worker_initialize(worker_config, {}, str(tmp_path / "startup"))

    payload = json.loads(next((tmp_path / "startup").glob("*.json")).read_text())
    assert payload["status"] == "FAILED"
    assert payload["exception_type"] == "RuntimeError"
    assert "original worker startup failure" in payload["traceback"]


class _BrokenInlineExecutor:
    def __init__(self, **_kwargs):
        self._processes = {}

    def submit(self, _function, _task):
        future = Future()
        future.set_exception(BrokenProcessPool("worker terminated before task execution"))
        return future

    def shutdown(self, *, wait: bool, cancel_futures: bool):
        assert wait is True
        assert cancel_futures is True


def test_broken_pool_report_is_bounded_and_preserves_primary_context(
    tmp_path: Path,
) -> None:
    config, _ = _base_fixture(tmp_path)

    result = alpha._execute_alpha_process_pool(
        ["AAA", "BBB", "CCC", "DDD"],
        config=config,
        input_resolution={},
        partition_root=tmp_path / "partitions",
        manifest_root=tmp_path / "manifests",
        report_root=tmp_path / "report",
        workers=2,
        fail_fast=alpha._fail_fast_settings({}),
        planned_partitions=4,
        started=0.0,
        executor_cls=_BrokenInlineExecutor,
    )

    lifecycle = json.loads(
        (tmp_path / "report" / "executor_lifecycle.json").read_text()
    )
    assert result["aborted_early"] is True
    assert len(result["failures"]) == 3
    assert lifecycle["multiprocessing_start_method"] == "spawn"
    assert lifecycle["submitted"] == 4
    assert lifecycle["failed"] == 3
    assert lifecycle["executor_phase"] == "shutdown_complete"


def test_spawn_pool_initializes_worker_local_resources_and_publishes_partitions(
    tmp_path: Path,
) -> None:
    config, _ = _base_fixture(tmp_path)
    validation = alpha.validate_alpha_base_artifact(config)
    base_partitions = alpha.prepare_alpha_base_partitions(
        config, base_validation=validation
    )
    config["ml"].update(
        {
            "canonical_v2_alpha_base_partition_root": base_partitions["path"],
            "canonical_v2_alpha_validated_base_sha256": validation["sha256"],
            "canonical_v2_alpha_validated_base_key_sha256": validation[
                "economic_key_sha256"
            ],
            "stock_alpha_feature_n_jobs": 2,
        }
    )
    partition_root = tmp_path / "enriched_partitions"
    manifest_root = tmp_path / "enriched_manifests"

    result = alpha._execute_alpha_process_pool(
        ["AAA", "BBB"],
        config=config,
        input_resolution={"validated_alpha_base": validation},
        partition_root=partition_root,
        manifest_root=manifest_root,
        report_root=tmp_path / "report",
        workers=2,
        fail_fast=alpha._fail_fast_settings({}),
        planned_partitions=2,
        started=0.0,
    )

    assert result["failures"] == []
    assert result["rows_processed"] == 2
    assert (partition_root / "symbol=AAA" / "rows.parquet").exists()
    assert (partition_root / "symbol=BBB" / "rows.parquet").exists()
    lifecycle = json.loads(
        (tmp_path / "report" / "executor_lifecycle.json").read_text()
    )
    assert lifecycle["status"] == "COMPLETE"
    assert lifecycle["multiprocessing_start_method"] == "spawn"
    assert lifecycle["completed"] == 2
    assert lifecycle["worker_startup_diagnostics"]


def test_alpha_only_cli_dispatches_only_the_alpha_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    commands = SimpleNamespace(
        run_ml_stock_level_alpha_features=lambda _config: calls.append("alpha")
    )
    monkeypatch.setattr(cli_dispatch, "_commands", lambda _name: commands)

    cli_dispatch.dispatch(
        SimpleNamespace(mode="ml-stock-level-alpha-features"),
        {"ml": {}},
        None,
    )

    assert calls == ["alpha"]


def test_alpha_only_run_manifest_records_immutable_parent_and_separate_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, path = _base_fixture(tmp_path)
    parent = alpha.validate_alpha_base_artifact(config)
    output = tmp_path / "published"
    paths = StockLevelAlphaFeaturePaths(
        enriched_parquet_path=output / "enriched.parquet",
        audit_csv_path=output / "audit.csv",
        audit_json_path=output / "audit.json",
        audit_markdown_path=output / "audit.md",
        enriched_sample_csv_path=output / "sample.csv",
    )
    monkeypatch.setattr(alpha, "validate_alpha_base_artifact", lambda _config: parent)
    monkeypatch.setattr(
        alpha,
        "_write_partitioned_canonical_v2_alpha_features",
        lambda _config: paths,
    )

    result = alpha.write_partitioned_canonical_v2_alpha_features(config)

    assert result == paths
    state = json.loads(
        (
            tmp_path / "alpha" / "alpha_only_run_manifest.json"
        ).read_text()
    )
    assert state["status"] == "COMPLETE"
    assert state["stages"]["stock_artifact"]["status"] == "completed_existing"
    assert state["stages"]["stock_artifact"]["path"] == str(path)
    assert state["stages"]["alpha_features"]["status"] == "completed"
    assert state["stock_artifact_generation_invoked"] is False


def test_alpha_resume_owner_does_not_import_unrelated_processing_owners() -> None:
    source = inspect.getsource(alpha)

    for forbidden in (
        "stock_level_portfolio_replay",
        "allocation.exposures",
        "stock_alpha_news",
        "alpaca_5m",
    ):
        assert forbidden not in source
