from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from config.config_loader import load_config
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_alpha_news_historical_backfill import (
    SCHEMA_VERSION,
    build_stock_alpha_news_historical_backfill,
    build_stock_alpha_news_historical_corpus_assembly,
    generate_historical_news_partitions,
    write_stock_alpha_news_historical_backfill,
)


class FakeNewsSource:
    api_key_required = False

    def __init__(self, rows=None, *, fail=False, more=False):
        self.rows = list([_row("1", "AAPL")] if rows is None else rows)
        self.fail = fail
        self.more = more
        self.calls = 0
        self.last_provider_config = {}
        self.last_collect_kwargs = {}
        self.last_batch_diagnostic = {}

    def with_provider_config(self, provider_config):
        self.last_provider_config = dict(provider_config)
        return self

    def collect(self, **kwargs):
        self.calls += 1
        self.last_collect_kwargs = dict(kwargs)
        if self.fail:
            raise RuntimeError("boom")
        provider = "alpaca_benzinga"
        self.last_batch_diagnostic = {
            f"{provider}_pages_requested": 1,
            f"{provider}_pages_completed": 1,
            f"{provider}_provider_records_returned": len(self.rows),
            f"{provider}_unique_provider_articles": len({row["provider_article_id"] for row in self.rows}),
            f"{provider}_multi_symbol_expansion_row_count": max(0, len(self.rows) - len({row["provider_article_id"] for row in self.rows})),
            f"{provider}_stopped_with_more_results_available": self.more,
            f"{provider}_termination_reason": "max_rows_per_batch" if self.more else "end_of_results",
        }
        return self.rows


class KeyedFakeNewsSource(FakeNewsSource):
    api_key_required = True


def test_historical_backfill_generates_deterministic_monthly_symbol_batch_partitions(tmp_path):
    config = _config(tmp_path, start="2024-01-15", end="2024-02-02", symbols=["AAPL", "MSFT", "NVDA"], batch_size=2)

    first = generate_historical_news_partitions(config)
    second = generate_historical_news_partitions(config)

    assert first == second
    assert [item["start_date"] for item in first] == ["2024-01-15", "2024-01-15", "2024-02-01", "2024-02-01"]
    assert [item["end_date"] for item in first] == ["2024-01-31", "2024-01-31", "2024-02-02", "2024-02-02"]
    assert first[0]["symbol_batch_id"] == "symbol_batch_001"
    assert first[1]["symbols"] == ["NVDA"]
    assert first[0]["schema_version"] == SCHEMA_VERSION
    assert first[0]["provider_config_hash"]


def test_historical_backfill_partition_id_changes_when_symbols_or_provider_config_change(tmp_path):
    base = _config(tmp_path, symbols=["AAPL", "MSFT"], batch_size=2)
    changed_symbols = _config(tmp_path, symbols=["AAPL", "NVDA"], batch_size=2)
    changed_config = _config(tmp_path, symbols=["AAPL", "MSFT"], batch_size=2)
    changed_config["ml"]["stock_alpha_news_historical_backfill"]["provider_config"]["page_size"] = 25

    assert generate_historical_news_partitions(base)[0]["partition_id"] != generate_historical_news_partitions(changed_symbols)[0]["partition_id"]
    assert generate_historical_news_partitions(base)[0]["partition_id"] != generate_historical_news_partitions(changed_config)[0]["partition_id"]


def test_historical_backfill_resolves_canonical_universe_and_batches(tmp_path):
    stock_rows = tmp_path / "stocks.csv"
    ResearchArtifactWriter().write_csv(
        stock_rows,
        [{"symbol": "MSFT"}, {"symbol": "AAPL"}, {"symbol": "NVDA"}],
        fieldnames=["symbol"],
    )
    config = _config(tmp_path, symbols=[], batch_size=2)
    config["ml"]["stock_alpha_stock_rows_path"] = str(stock_rows)
    settings = config["ml"]["stock_alpha_news_historical_backfill"]
    settings["use_canonical_universe"] = True
    settings["only_symbols"] = ["AAPL", "NVDA"]

    partitions = generate_historical_news_partitions(config)

    assert [partition["symbols"] for partition in partitions] == [["AAPL", "NVDA"]]


def test_historical_backfill_completes_partition_and_writes_manifest_and_artifact(tmp_path):
    config = _config(tmp_path)
    source = FakeNewsSource(rows=[_row("1", "AAPL"), _row("1", "MSFT")])

    paths = write_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": source})
    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    record = manifest["partitions"][0]

    assert record["status"] == "complete"
    assert record["attempt_count"] == 1
    assert record["article_symbol_rows"] == 2
    assert record["multi_symbol_expansion_rows"] == 1
    assert record["artifact_size"] > 0
    assert Path(record["output_artifact"]).is_file()
    assert Path(record["audit_artifact"]).is_file()
    assert "year=2024/month=01/symbol_batch=001" in record["output_artifact"]
    assert not list(Path(record["output_artifact"]).parent.glob("*.tmp"))
    audit = json.loads(Path(record["audit_artifact"]).read_text(encoding="utf-8"))
    assert audit["partition"]["status"] == "complete"
    rows = list(csv.DictReader(Path(record["output_artifact"]).open(encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["publisher"] == "Benzinga"
    assert rows[0]["author"] == "Amit Nag"
    assert rows[0]["raw_source"] == "benzinga"
    assert rows[0]["summary"] == "Short summary"
    assert rows[0]["body_or_full_text"] == ""
    assert rows[0]["body_or_summary_kind"] == "summary"


def test_historical_backfill_attempts_alpaca_with_partition_config(tmp_path):
    config = _config(
        tmp_path,
        start="2016-01-03",
        end="2016-01-07",
        symbols=["AAPL", "MSFT", "AMZN"],
        batch_size=3,
    )
    settings = config["ml"]["stock_alpha_news_historical_backfill"]
    settings["provider_request_limit"] = 25
    settings["max_rows_per_partition"] = 2000
    settings["provider_config"] = {
        "enabled": True,
        "api_key_env": "ALPACA_API_KEY_ID",
        "secret_key_env": "ALPACA_SECRET_KEY",
        "page_size": 50,
        "max_pages_per_batch": 80,
    }
    source = FakeNewsSource(rows=[_row("attempted", "AAPL", published="2016-01-04T10:00:00Z")])

    payload = build_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": source})
    record = payload["manifest"]["partitions"][0]

    assert record["status"] == "complete"
    assert record["provider_requested"] is True
    assert record["provider_attempted"] is True
    assert source.calls == 1
    assert source.last_provider_config["api_key_env"] == "ALPACA_API_KEY_ID"
    assert source.last_provider_config["secret_key_env"] == "ALPACA_SECRET_KEY"
    assert source.last_provider_config["page_size"] == 50
    assert source.last_provider_config["max_pages_per_batch"] == 80
    assert source.last_collect_kwargs["symbols"] == ["AAPL", "MSFT", "AMZN"]
    assert source.last_collect_kwargs["start_date"] == "2016-01-03"
    assert source.last_collect_kwargs["end_date"] == "2016-01-07"
    assert source.last_collect_kwargs["limit"] == 25
    assert record["provider_batch_count"] == 1
    assert record["pages_requested"] == 1
    assert record["termination_reason"] == "end_of_results"


def test_historical_backfill_missing_key_skip_cannot_complete(tmp_path, monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    config = _config(tmp_path)
    source = KeyedFakeNewsSource(rows=[])

    payload = build_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": source})
    record = payload["manifest"]["partitions"][0]
    audit = json.loads(Path(record["audit_artifact"]).read_text(encoding="utf-8"))

    assert source.calls == 0
    assert record["status"] == "failed"
    assert record["last_error"] == "provider_skipped_missing_key"
    assert record["provider_requested"] is True
    assert record["provider_attempted"] is False
    assert record["provider_skipped_missing_key"] is True
    assert audit["provider_execution"]["providers_skipped_missing_key"] == ["alpaca_benzinga"]


def test_historical_backfill_provider_not_selected_cannot_complete(tmp_path):
    config = _config(tmp_path)
    config["ml"]["stock_alpha_news_historical_backfill"]["provider_config"] = {"enabled": False}
    source = FakeNewsSource(rows=[])

    payload = build_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": source})
    record = payload["manifest"]["partitions"][0]

    assert source.calls == 0
    assert record["status"] == "failed"
    assert record["last_error"] == "provider_not_requested"
    assert record["provider_requested"] is False
    assert record["provider_attempted"] is False


def test_historical_backfill_missing_provider_adapter_cannot_complete(tmp_path):
    config = _config(tmp_path)

    payload = build_stock_alpha_news_historical_backfill(config, sources={"other": FakeNewsSource(rows=[])})
    record = payload["manifest"]["partitions"][0]

    assert record["status"] == "failed"
    assert record["last_error"] == "provider_adapter_unavailable"
    assert record["provider_requested"] is True
    assert record["provider_attempted"] is False
    assert record["provider_failed"] is True


def test_historical_backfill_attempted_zero_rows_can_complete(tmp_path):
    config = _config(tmp_path)
    source = FakeNewsSource(rows=[])

    payload = build_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": source})
    record = payload["manifest"]["partitions"][0]
    audit = json.loads(Path(record["audit_artifact"]).read_text(encoding="utf-8"))

    assert source.calls == 1
    assert record["status"] == "complete"
    assert record["provider_requested"] is True
    assert record["provider_attempted"] is True
    assert record["provider_returned_zero_rows"] is True
    assert record["provider_zero_row_reason"] == "all_batches_returned_zero_rows"
    assert record["pages_requested"] == 1
    assert record["pages_completed"] == 1
    assert record["termination_reason"] == "end_of_results"
    assert record["output_artifact"] == ""
    assert audit["provider_execution"]["providers_attempted"] == ["alpaca_benzinga"]


def test_historical_backfill_existing_zero_row_complete_without_attempt_evidence_is_retried(tmp_path):
    config = _config(tmp_path)
    partition = generate_historical_news_partitions(config)[0]
    old_record = {
        **partition,
        "status": "complete",
        "attempt_count": 1,
        "output_artifact": "",
        "output_row_count": 0,
        "checksum": "",
        "pages_requested": 0,
        "termination_reason": "",
    }
    _write_manifest(config, [old_record])
    source = FakeNewsSource(rows=[])

    payload = build_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": source})
    record = payload["manifest"]["partitions"][0]

    assert source.calls == 1
    assert payload["summary"]["skipped_complete"] == 0
    assert record["status"] == "complete"
    assert record["attempt_count"] == 2
    assert record["provider_attempted"] is True
    assert record["pages_requested"] == 1


def test_historical_backfill_resume_skips_complete_partition_without_duplicate_append(tmp_path):
    config = _config(tmp_path)
    source = FakeNewsSource(rows=[_row("1", "AAPL")])

    first = write_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": source})
    second_payload = build_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": source})
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    artifact = Path(manifest["partitions"][0]["output_artifact"])

    assert source.calls == 1
    assert second_payload["summary"]["skipped_complete"] == 1
    assert len(list(csv.DictReader(artifact.open(encoding="utf-8")))) == 1


def test_historical_backfill_failed_partition_is_recorded_and_retry_can_complete(tmp_path):
    config = _config(tmp_path)

    first = write_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": FakeNewsSource(fail=True)})
    failed_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    failed_record = failed_manifest["partitions"][0]
    assert failed_record["status"] == "failed"
    assert failed_record["provider_attempted"] is True
    assert failed_record["provider_failed"] is True

    payload = build_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": FakeNewsSource(rows=[_row("2", "AAPL")])})
    record = payload["manifest"]["partitions"][0]

    assert record["status"] == "complete"
    assert record["attempt_count"] == 2
    assert record["last_error"] == ""


def test_historical_backfill_interrupted_running_partition_is_retried(tmp_path):
    config = _config(tmp_path)
    partition = generate_historical_news_partitions(config)[0]
    manifest_path = Path(config["ml"]["stock_alpha_news_historical_backfill"]["work_dir"]) / "stock_alpha_news_historical_backfill_manifest.json"
    interrupted = {
        "schema_version": SCHEMA_VERSION,
        "partitions": [
            {
                **partition,
                "status": "running",
                "attempt_count": 1,
                "output_artifact": "",
                "output_row_count": 0,
            }
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(interrupted), encoding="utf-8")

    payload = build_stock_alpha_news_historical_backfill(
        config,
        sources={"alpaca_benzinga": FakeNewsSource(rows=[_row("retry", "AAPL")])},
    )
    record = payload["manifest"]["partitions"][0]

    assert record["status"] == "complete"
    assert record["attempt_count"] == 2


def test_historical_backfill_partial_partition_not_marked_complete_when_more_results_available(tmp_path):
    config = _config(tmp_path)
    config["ml"]["stock_alpha_news_historical_backfill"]["max_partitions_per_run"] = 1
    payload = build_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": FakeNewsSource(more=True)})
    record = payload["manifest"]["partitions"][0]

    assert record["status"] == "partial"
    assert record["stopped_with_more_results_available"] is True
    assert record["output_artifact"] == ""
    assert record["child_partition_ids"]
    child_records = [row for row in payload["manifest"]["partitions"] if row.get("parent_partition_id") == record["partition_id"]]
    assert [row["status"] for row in child_records] == ["pending", "pending"]
    assert child_records[0]["start_date"] == "2024-01-01"
    assert child_records[0]["end_date"] == "2024-01-16"


def test_historical_backfill_existing_artifact_without_complete_manifest_is_rebuilt(tmp_path):
    config = _config(tmp_path)
    partition = generate_historical_news_partitions(config)[0]
    artifact = (
        Path(config["ml"]["stock_alpha_news_historical_backfill"]["work_dir"])
        / "partitions"
        / "alpaca_benzinga"
        / "year=2024"
        / "month=01"
        / "symbol_batch=001"
        / f"{partition['partition_id']}.csv"
    )
    ResearchArtifactWriter().write_csv(artifact, [_row("old", "AAPL")], fieldnames=list(_row("old", "AAPL")))

    payload = build_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": FakeNewsSource(rows=[_row("new", "AAPL")])})

    assert payload["manifest"]["partitions"][0]["status"] == "complete"
    rows = list(csv.DictReader(artifact.open(encoding="utf-8")))
    assert [row["provider_article_id"] for row in rows] == ["new"]


def test_historical_backfill_filters_publication_window_and_preserves_valid_rows(tmp_path):
    config = _config(tmp_path, start="2024-01-01", end="2024-01-31")
    rows = [
        _row("before", "AAPL", published="2023-12-31T23:59:59Z"),
        _row("start", "AAPL", published="2024-01-01T00:00:00Z"),
        _row("end", "AAPL", published="2024-01-31T23:59:59Z"),
        _row("after", "AAPL", published="2024-02-01T00:00:00Z"),
    ]

    payload = build_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": FakeNewsSource(rows=rows)})
    record = payload["manifest"]["partitions"][0]

    assert record["out_of_window_rejected_count"] == 2
    assert record["article_symbol_rows"] == 2
    artifact_rows = list(csv.DictReader(Path(record["output_artifact"]).open(encoding="utf-8")))
    assert [row["provider_article_id"] for row in artifact_rows] == ["start", "end"]


def test_historical_corpus_assembly_uses_only_complete_partitions_and_deduplicates(tmp_path):
    config = _config(tmp_path)
    write_stock_alpha_news_historical_backfill(
        config,
        sources={"alpaca_benzinga": FakeNewsSource(rows=[_row("1", "AAPL"), _row("1", "AAPL")])},
    )

    payload, rows = build_stock_alpha_news_historical_corpus_assembly(config)

    assert payload["incomplete_partition_count"] == 0
    assert len(rows) == 1
    assert payload["row_count"] == 1
    assert payload["publisher_availability_rate"] == 1.0
    assert payload["author_availability_rate"] == 1.0
    assert payload["text_availability_by_year"]["2024"]["summary_availability_rate"] == 1.0
    assert payload["text_availability_by_year"]["2024"]["body_or_full_text_availability_rate"] == 0.0
    assert payload["invalid_required_field_row_count"] == 0


def test_historical_corpus_assembly_rejects_incomplete_partitions(tmp_path):
    config = _config(tmp_path)
    config["ml"]["stock_alpha_news_historical_backfill"]["max_partitions_per_run"] = 1
    build_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": FakeNewsSource(more=True)})

    payload, rows = build_stock_alpha_news_historical_corpus_assembly(config)

    assert rows == []
    assert payload["incomplete_partition_count"] >= 1
    assert payload["row_count"] == 0


def test_historical_corpus_assembly_rejects_blank_artifact_for_data_bearing_complete_partition(tmp_path):
    config = _config(tmp_path)
    _write_manifest(config, [_complete_record("blank", output_artifact="", output_row_count=1)])

    try:
        build_stock_alpha_news_historical_corpus_assembly(config)
    except ValueError as exc:
        assert "complete partition blank has blank output_artifact" in str(exc)
    else:
        raise AssertionError("expected manifest integrity failure")


def test_historical_corpus_assembly_rejects_missing_artifact_for_data_bearing_complete_partition(tmp_path):
    config = _config(tmp_path)
    record = _complete_record("missing-field", output_row_count=1)
    record.pop("output_artifact")
    _write_manifest(config, [record])

    try:
        build_stock_alpha_news_historical_corpus_assembly(config)
    except ValueError as exc:
        assert "complete partition missing-field has blank output_artifact" in str(exc)
    else:
        raise AssertionError("expected manifest integrity failure")


def test_historical_corpus_assembly_rejects_directory_artifact_path(tmp_path):
    config = _config(tmp_path)
    artifact_dir = tmp_path / "artifact.csv"
    artifact_dir.mkdir()
    _write_manifest(config, [
        _complete_record("directory", output_artifact=str(artifact_dir), output_row_count=1, checksum="abc")
    ])

    try:
        build_stock_alpha_news_historical_corpus_assembly(config)
    except ValueError as exc:
        assert "complete partition directory output_artifact is not a regular file" in str(exc)
    else:
        raise AssertionError("expected manifest integrity failure")


def test_historical_corpus_assembly_rejects_nonexistent_artifact(tmp_path):
    config = _config(tmp_path)
    artifact = tmp_path / "missing.csv"
    _write_manifest(config, [
        _complete_record("nonexistent", output_artifact=str(artifact), output_row_count=1, checksum="abc")
    ])

    try:
        build_stock_alpha_news_historical_corpus_assembly(config)
    except ValueError as exc:
        assert "complete partition nonexistent output_artifact does not exist" in str(exc)
    else:
        raise AssertionError("expected manifest integrity failure")


def test_historical_corpus_assembly_valid_complete_artifact_assembles(tmp_path):
    config = _config(tmp_path)
    artifact, checksum = _write_partition_csv(tmp_path, "valid", [_row("1", "AAPL")])
    _write_manifest(config, [
        _complete_record("valid", output_artifact=str(artifact), output_row_count=1, checksum=checksum)
    ])

    payload, rows = build_stock_alpha_news_historical_corpus_assembly(config)

    assert payload["row_count"] == 1
    assert [row["provider_article_id"] for row in rows] == ["1"]


def test_historical_corpus_assembly_write_records_csv_path_checksum_and_row_count(tmp_path):
    config = _config(tmp_path)
    config["ml"]["stock_alpha_news_historical_backfill"]["action"] = "assemble"
    first_artifact, first_checksum = _write_partition_csv(tmp_path, "first", [_row("1", "AAPL")])
    second_artifact, second_checksum = _write_partition_csv(tmp_path, "second", [_row("2", "MSFT")])
    _write_manifest(config, [
        _complete_record("first", output_artifact=str(first_artifact), output_row_count=1, checksum=first_checksum),
        _complete_record("second", output_artifact=str(second_artifact), output_row_count=1, checksum=second_checksum),
    ])

    paths = write_stock_alpha_news_historical_backfill(config)
    payload = json.loads(paths.assembly_json_path.read_text(encoding="utf-8"))
    csv_rows = list(csv.DictReader(paths.assembly_csv_path.open(encoding="utf-8")))

    assert payload["assembly_csv_path"] == str(paths.assembly_csv_path)
    assert paths.assembly_csv_path.is_file()
    assert payload["assembly_checksum"]
    assert payload["checksum"] == payload["assembly_checksum"]
    assert payload["assembly_checksum"] == _file_sha256(paths.assembly_csv_path)
    assert payload["row_count"] == len(csv_rows) == 2
    assert payload["incomplete_partition_count"] == 0


def test_historical_corpus_assembly_excludes_partial_parent_partition(tmp_path):
    config = _config(tmp_path)
    config["ml"]["stock_alpha_news_historical_backfill"]["assembly_require_all_complete"] = False
    artifact, checksum = _write_partition_csv(tmp_path, "child", [_row("child", "AAPL")])
    _write_manifest(config, [
        _partial_record("parent", output_artifact=".", output_row_count=99),
        _complete_record("child", output_artifact=str(artifact), output_row_count=1, checksum=checksum, parent_partition_id="parent"),
    ])

    payload, rows = build_stock_alpha_news_historical_corpus_assembly(config)

    assert payload["incomplete_partition_count"] == 1
    assert payload["row_count"] == 1
    assert [row["provider_article_id"] for row in rows] == ["child"]


def test_historical_corpus_assembly_write_excludes_failed_partial_and_incomplete_partitions(tmp_path):
    config = _config(tmp_path)
    config["ml"]["stock_alpha_news_historical_backfill"]["action"] = "assemble"
    config["ml"]["stock_alpha_news_historical_backfill"]["assembly_require_all_complete"] = False
    artifact, checksum = _write_partition_csv(tmp_path, "complete", [_row("complete", "AAPL")])
    failed = _partial_record("failed")
    failed["status"] = "failed"
    pending = _partial_record("pending")
    pending["status"] = "pending"
    _write_manifest(config, [
        _complete_record("complete", output_artifact=str(artifact), output_row_count=1, checksum=checksum),
        _partial_record("partial"),
        failed,
        pending,
    ])

    paths = write_stock_alpha_news_historical_backfill(config)
    payload = json.loads(paths.assembly_json_path.read_text(encoding="utf-8"))
    csv_rows = list(csv.DictReader(paths.assembly_csv_path.open(encoding="utf-8")))

    assert payload["incomplete_partition_count"] == 3
    assert payload["row_count"] == len(csv_rows) == 1
    assert [row["provider_article_id"] for row in csv_rows] == ["complete"]
    assert payload["assembly_checksum"] == _file_sha256(paths.assembly_csv_path)


def test_historical_corpus_assembly_adaptive_split_complete_children_assemble(tmp_path):
    config = _config(tmp_path)
    config["ml"]["stock_alpha_news_historical_backfill"]["assembly_require_all_complete"] = False
    first_artifact, first_checksum = _write_partition_csv(tmp_path, "child-1", [_row("child-1", "AAPL")])
    second_artifact, second_checksum = _write_partition_csv(tmp_path, "child-2", [_row("child-2", "MSFT")])
    _write_manifest(config, [
        _partial_record("parent", stopped_with_more_results_available=True),
        _complete_record("child-1", output_artifact=str(first_artifact), output_row_count=1, checksum=first_checksum, parent_partition_id="parent"),
        _complete_record("child-2", output_artifact=str(second_artifact), output_row_count=1, checksum=second_checksum, parent_partition_id="parent"),
    ])

    payload, rows = build_stock_alpha_news_historical_corpus_assembly(config)

    assert payload["row_count"] == 2
    assert {row["provider_article_id"] for row in rows} == {"child-1", "child-2"}


def test_historical_corpus_assembly_zero_row_complete_record_is_not_read_as_csv(tmp_path):
    config = _config(tmp_path)
    _write_manifest(config, [_complete_record("zero", output_artifact="", output_row_count=0)])

    payload, rows = build_stock_alpha_news_historical_corpus_assembly(config)

    assert payload["complete_partition_count"] == 1
    assert payload["row_count"] == 0
    assert rows == []


def test_historical_corpus_assembly_rejects_checksum_mismatch(tmp_path):
    config = _config(tmp_path)
    artifact, _checksum = _write_partition_csv(tmp_path, "bad-checksum", [_row("bad-checksum", "AAPL")])
    _write_manifest(config, [
        _complete_record("bad-checksum", output_artifact=str(artifact), output_row_count=1, checksum="not-the-checksum")
    ])

    try:
        build_stock_alpha_news_historical_corpus_assembly(config)
    except ValueError as exc:
        assert "complete partition bad-checksum checksum mismatch" in str(exc)
    else:
        raise AssertionError("expected manifest integrity failure")


def test_historical_corpus_assembly_rejects_complete_record_stopped_with_more_results(tmp_path):
    config = _config(tmp_path)
    artifact, checksum = _write_partition_csv(tmp_path, "stopped", [_row("stopped", "AAPL")])
    _write_manifest(config, [
        _complete_record(
            "stopped",
            output_artifact=str(artifact),
            output_row_count=1,
            checksum=checksum,
            stopped_with_more_results_available=True,
        )
    ])

    try:
        build_stock_alpha_news_historical_corpus_assembly(config)
    except ValueError as exc:
        assert "complete partition stopped has stopped_with_more_results_available=true" in str(exc)
    else:
        raise AssertionError("expected manifest integrity failure")


def test_historical_corpus_assembly_blank_path_never_resolves_to_current_directory(tmp_path):
    config = _config(tmp_path)
    _write_manifest(config, [_complete_record("dot-guard", output_artifact="", output_row_count=1)])

    try:
        build_stock_alpha_news_historical_corpus_assembly(config)
    except ValueError as exc:
        assert "blank output_artifact" in str(exc)
        assert "Is a directory" not in str(exc)
    else:
        raise AssertionError("expected manifest integrity failure")


def test_historical_corpus_assembly_pilot_style_zero_row_manifest_assembles(tmp_path):
    config = _config(tmp_path)
    _write_manifest(config, [
        _complete_record("symbol-batch-001", output_artifact="", output_row_count=0),
        _complete_record("symbol-batch-002", output_artifact="", output_row_count=0),
    ])

    payload, rows = build_stock_alpha_news_historical_corpus_assembly(config)

    assert payload["complete_partition_count"] == 2
    assert payload["row_count"] == 0
    assert rows == []


def test_historical_backfill_configs_parse_without_enabling_models_or_trading():
    for path in [
        "config/config.stock_alpha_news_historical_backfill_alpaca_benzinga_tiny_mock_smoke.yaml",
        "config/config.stock_alpha_news_historical_backfill_alpaca_benzinga_pilot.yaml",
        "config/config.stock_alpha_news_historical_backfill_alpaca_benzinga_known_positive_pilot.yaml",
        "config/config.stock_alpha_news_historical_backfill_alpaca_benzinga_known_positive_pilot_assembly.yaml",
        "config/config.stock_alpha_news_historical_backfill_alpaca_benzinga_full_template.yaml",
    ]:
        config = load_config(path, overlay_project_config=True)
        ml = config["ml"]
        settings = ml["stock_alpha_news_historical_backfill"]
        assert settings["provider"] == "alpaca_benzinga"
        assert ml["stock_alpha_news_enable_transformer"] is False
        assert ml["trading_impact"] == "none"
        assert ml["production_validated"] is False


def test_known_positive_historical_backfill_configs_are_isolated_and_research_only():
    collect = load_config(
        "config/config.stock_alpha_news_historical_backfill_alpaca_benzinga_known_positive_pilot.yaml",
        overlay_project_config=True,
    )
    assembly = load_config(
        "config/config.stock_alpha_news_historical_backfill_alpaca_benzinga_known_positive_pilot_assembly.yaml",
        overlay_project_config=True,
    )
    collect_settings = collect["ml"]["stock_alpha_news_historical_backfill"]
    assembly_settings = assembly["ml"]["stock_alpha_news_historical_backfill"]
    work_dir = collect_settings["work_dir"]

    assert assembly_settings["work_dir"] == work_dir
    assert collect_settings["action"] == "collect"
    assert collect_settings["dry_run"] is False
    assert assembly_settings["action"] == "assemble"
    assert collect_settings["symbols"] == ["AAPL", "MSFT", "AMZN"]
    assert collect_settings["start_date"] == "2016-01-03"
    assert collect_settings["end_date"] == "2016-01-07"
    assert assembly_settings["start_date"] == "2016-01-03"
    assert assembly_settings["end_date"] == "2016-01-07"
    assert collect_settings["symbol_batch_size"] == 3
    assert collect_settings["max_partitions_per_run"] == 1
    assert collect_settings["provider_request_limit"] == 250
    assert collect_settings["max_rows_per_partition"] == 2000
    assert collect_settings["provider_config"]["page_size"] == 50
    assert collect_settings["provider_config"]["max_pages_per_batch"] == 80
    assert collect["ml"]["stock_alpha_news_enable_transformer"] is False
    assert assembly["ml"]["stock_alpha_news_enable_transformer"] is False
    assert collect["ml"]["trading_impact"] == "none"
    assert assembly["ml"]["trading_impact"] == "none"
    assert collect["ml"]["production_validated"] is False
    assert assembly["ml"]["production_validated"] is False
    assert "known_positive_pilot" in work_dir
    assert "stock_alpha_news_historical_backfill_alpaca_benzinga_pilot/dev" not in work_dir
    assert "stock_alpha_news_historical_backfill_alpaca_benzinga_full/dev" not in work_dir


def test_known_positive_historical_backfill_uses_collector_safe_request_limit(tmp_path):
    config = load_config(
        "config/config.stock_alpha_news_historical_backfill_alpaca_benzinga_known_positive_pilot.yaml",
        overlay_project_config=True,
    )
    config["ml"]["stock_alpha_news_historical_backfill"]["work_dir"] = str(tmp_path / "known_positive_pilot")
    source = FakeNewsSource(rows=[_row("known-positive", "AAPL", published="2016-01-04T10:00:00Z")])

    payload = build_stock_alpha_news_historical_backfill(
        config,
        sources={"alpaca_benzinga": source},
    )

    record = payload["manifest"]["partitions"][0]
    assert record["status"] == "complete"
    assert source.last_collect_kwargs["limit"] == 250


def _config(
    tmp_path: Path,
    *,
    start: str = "2024-01-01",
    end: str = "2024-01-31",
    symbols: list[str] | None = None,
    batch_size: int = 2,
) -> dict:
    return {
        "ml": {
            "stock_alpha_news_historical_backfill": {
                "action": "collect",
                "dry_run": False,
                "work_dir": str(tmp_path / "backfill"),
                "provider": "alpaca_benzinga",
                "start_date": start,
                "end_date": end,
                "partition_frequency": "monthly",
                "symbols": ["AAPL", "MSFT"] if symbols is None else symbols,
                "symbol_batch_size": batch_size,
                "max_partitions_per_run": 10,
                "provider_request_limit": 20,
                "max_rows_per_partition": 100,
                "request_timeout_seconds": 2,
                "rate_limit_sleep_seconds": 0.0,
                "provider_config": {"enabled": True},
            },
            "stock_alpha_news_enable_transformer": False,
        }
    }


def _write_manifest(config: dict, partitions: list[dict]) -> None:
    manifest_path = Path(config["ml"]["stock_alpha_news_historical_backfill"]["work_dir"]) / "stock_alpha_news_historical_backfill_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({
            "schema_version": SCHEMA_VERSION,
            "action": "collect",
            "dry_run": False,
            "partition_count": len(partitions),
            "partitions": partitions,
        }),
        encoding="utf-8",
    )


def _complete_record(
    partition_id: str,
    *,
    output_artifact: str = "",
    output_row_count: int = 0,
    checksum: str = "",
    parent_partition_id: str = "",
    stopped_with_more_results_available: bool = False,
) -> dict:
    return {
        "partition_id": partition_id,
        "status": "complete",
        "parent_partition_id": parent_partition_id,
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "symbol_batch_id": "symbol_batch_001",
        "output_artifact": output_artifact,
        "output_row_count": output_row_count,
        "checksum": checksum,
        "stopped_with_more_results_available": stopped_with_more_results_available,
    }


def _partial_record(
    partition_id: str,
    *,
    output_artifact: str = "",
    output_row_count: int = 0,
    stopped_with_more_results_available: bool = False,
) -> dict:
    record = _complete_record(
        partition_id,
        output_artifact=output_artifact,
        output_row_count=output_row_count,
        stopped_with_more_results_available=stopped_with_more_results_available,
    )
    record["status"] = "partial"
    return record


def _write_partition_csv(tmp_path: Path, stem: str, rows: list[dict]) -> tuple[Path, str]:
    artifact = tmp_path / f"{stem}.csv"
    ResearchArtifactWriter().write_csv(artifact, rows, fieldnames=list(rows[0]), extrasaction="ignore")
    return artifact, _file_sha256(artifact)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row(
    provider_id: str,
    symbol: str,
    *,
    published: str = "2024-01-02T10:00:00Z",
    summary: str = "Short summary",
    body: str = "",
) -> dict:
    return {
        "article_id": f"alpaca_benzinga:{provider_id}:{symbol}",
        "symbol": symbol,
        "published_at_utc": published,
        "source": "Benzinga",
        "headline": f"Headline {provider_id}",
        "body_or_summary": body or summary,
        "sentiment_score": "",
        "relevance_score": "",
        "novelty_score": "",
        "event_type": "editorial_news",
        "language": "en",
        "ingested_at": "2024-01-02T10:01:00Z",
        "provider": "alpaca_benzinga",
        "provider_article_id": provider_id,
        "provider_url": f"https://example.test/{provider_id}",
        "provider_symbols": symbol,
        "updated_at_utc": "2024-01-02T10:02:00Z",
        "collected_at_utc": "2024-01-02T10:01:00Z",
        "publisher": "Benzinga",
        "author": "Amit Nag",
        "raw_source": "benzinga",
        "summary": summary,
        "body_or_full_text": body,
        "body_or_summary_kind": "body_or_full_text" if body else "summary",
    }
