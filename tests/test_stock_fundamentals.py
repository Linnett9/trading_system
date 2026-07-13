from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import application.cli_dispatch as cli_dispatch
from application.cli_runtime import FEEDLESS_MODES
from core.research.ml.stock_level.selector_feature_ablation import build_feature_family_contracts
from core.research.ml.stock_level.stock_fundamentals import (
    SecCompanyFactsProvider,
    build_fundamental_snapshots,
    build_stock_fundamentals_pipeline,
    canonical_fact_dictionary,
    normalize_sec_company_facts,
    validate_cached_companyfacts,
    write_stock_fundamentals_collect,
    write_stock_fundamentals_enrich,
    write_stock_fundamentals_normalize,
    write_stock_fundamentals_preflight,
    write_stock_fundamentals_snapshots,
)
from core.research.ml.stock_level.stock_level_artifact_io import write_stock_level_artifact


def test_sec_companyfacts_cache_is_deterministic_and_resumable(tmp_path):
    calls = []

    def fake_get(url, headers, timeout):
        calls.append((url, headers["User-Agent"], timeout))
        return json.dumps(_companyfacts_payload()).encode(), {"content-type": "application/json", "etag": "v1"}

    provider = SecCompanyFactsProvider(
        raw_root=tmp_path / "raw",
        user_agent="Research Bot contact@example.com",
        request_delay_seconds=0,
        http_get=fake_get,
    )

    first = provider.fetch_company_facts("CIK0000320193")
    second = provider.fetch_company_facts("0000320193")

    assert first["status"] == "downloaded"
    assert second["status"] == "skipped_cached"
    assert calls == [
        (
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
            "Research Bot contact@example.com",
            30,
        )
    ]
    assert first["path"] == tmp_path / "raw" / "official_sec_companyfacts" / "CIK0000320193" / "companyfacts.json"


def test_normalization_uses_filing_availability_not_period_end(tmp_path):
    raw_path = tmp_path / "companyfacts.json"
    raw_path.write_text(json.dumps(_companyfacts_payload()), encoding="utf-8")
    mapping = _mapping()

    facts, audit = normalize_sec_company_facts(
        [{"path": raw_path, "metadata": {"sha256": "abc", "retrieval_timestamp": "2024-01-01T00:00:00Z"}, "payload": _companyfacts_payload()}],
        mapping,
        fact_dictionary=canonical_fact_dictionary(),
    )

    revenue = [row for row in facts if row["canonical_fact_id"] == "revenue" and row["filing_accession"] == "orig-q1"][0]
    assert revenue["period_end"] == "2023-03-31"
    assert revenue["available_timestamp"] == "2023-05-01T23:59:59Z"
    assert audit["availability_rule"].startswith("available_timestamp is SEC filed date")


def test_snapshot_excludes_future_filing_and_does_not_leak_amendment(tmp_path):
    raw_path = tmp_path / "companyfacts.json"
    raw_path.write_text(json.dumps(_companyfacts_payload()), encoding="utf-8")
    facts, _audit = normalize_sec_company_facts(
        [{"path": raw_path, "metadata": {"sha256": "abc", "retrieval_timestamp": "2024-01-01T00:00:00Z"}, "payload": _companyfacts_payload()}],
        _mapping(),
        fact_dictionary=canonical_fact_dictionary(),
    )
    base_rows = [
        {"rebalance_date": "2023-06-15", "symbol": "AAPL", "close": 10.0},
        {"rebalance_date": "2023-09-15", "symbol": "AAPL", "close": 10.0},
        {"rebalance_date": "2023-06-15", "symbol": "MSFT", "close": 10.0},
    ]

    snapshots, audit = build_fundamental_snapshots(
        base_rows,
        _mapping(),
        facts,
        maximum_data_age_days=None,
        minimum_denominator=1e-9,
    )

    june = [row for row in snapshots if row["symbol"] == "AAPL" and row["decision_timestamp"].startswith("2023-06")][0]
    september = [row for row in snapshots if row["symbol"] == "AAPL" and row["decision_timestamp"].startswith("2023-09")][0]
    missing = [row for row in snapshots if row["symbol"] == "MSFT"][0]

    assert "orig-q1" in june["selected_source_document_identities"]
    assert "amend-q1" not in june["selected_source_document_identities"]
    assert "amend-q1" in september["selected_source_document_identities"]
    assert audit["future_filing_exclusion_count"] > 0
    assert missing["snapshot_status"] == "unresolved_entity"
    assert missing["gross_margin"] is None


def test_pipeline_enriches_artifact_with_non_null_real_features_and_lineage(tmp_path):
    base_path = tmp_path / "base.parquet"
    base_rows = [
        _base_row("2023-06-15", "AAPL"),
        _base_row("2023-09-15", "AAPL"),
        _base_row("2023-09-15", "MSFT"),
    ]
    write_stock_level_artifact(
        base_path,
        base_rows,
        fieldnames=list(base_rows[0]),
        config={"ml": {"stock_level_artifact_format": "parquet", "stock_level_parquet_compression": "zstd"}},
    )

    def fake_get(url, headers, timeout):
        assert headers["User-Agent"] == "Research Bot contact@example.com"
        return json.dumps(_companyfacts_payload()).encode(), {"content-type": "application/json"}

    payload = build_stock_fundamentals_pipeline(
        {
            "ml": {
                "stock_fundamentals": {
                    "enabled": True,
                    "source_dataset_path": str(base_path),
                    "output_dir": str(tmp_path / "out"),
                    "symbols": ["AAPL", "MSFT"],
                    "cik_by_symbol": {"AAPL": "320193"},
                    "user_agent": "Research Bot contact@example.com",
                    "collection": {"raw_root": str(tmp_path / "raw"), "request_delay_seconds": 0},
                }
            }
        },
        http_get=fake_get,
    )

    enriched = payload["enriched_rows"]
    aapl = [row for row in enriched if row["symbol"] == "AAPL" and row["rebalance_date"] == "2023-09-15"][0]
    msft = [row for row in enriched if row["symbol"] == "MSFT"][0]
    assert len(enriched) == len(base_rows)
    assert aapl["gross_margin"] == 0.5
    assert aapl["fundamentals_contract_identity"]
    assert msft["fundamentals_snapshot_status"] == "unresolved_entity"
    assert msft["gross_margin"] is None
    assert payload["analyst_estimate_status"] == "source_not_configured"


def test_fundamentals_families_resolve_only_non_null_present_columns():
    rows = [
        {"symbol": "AAPL", "gross_margin": 0.5, "earnings_quality_score": None, "fundamental_coverage_count": 2},
        {"symbol": "MSFT", "gross_margin": None, "earnings_quality_score": None, "fundamental_coverage_count": 0},
    ]
    available = ("gross_margin", "fundamental_coverage_count")
    families = build_feature_family_contracts(rows, available, settings={})
    by_id = {row["family_id"]: row for row in families}

    assert by_id["fundamental_profitability"]["resolved_ordered_columns"] == ["gross_margin"]
    assert "earnings_quality_score" not in by_id["fundamental_quality"]["resolved_ordered_columns"]
    assert by_id["fundamental_growth"]["resolved_ordered_columns"] == []
    assert by_id["fundamental_growth"]["point_in_time_status"] == "SAFE WITH CONDITIONS"


def test_stock_fundamentals_cli_modes_are_feedless_and_dispatch(monkeypatch):
    for mode in [
        "ml-stock-fundamentals-preflight",
        "ml-stock-fundamentals-collect",
        "ml-stock-fundamentals-normalize",
        "ml-stock-fundamentals-audit",
        "ml-stock-fundamentals-snapshots",
        "ml-stock-fundamentals-enrich",
        "ml-stock-fundamentals-pipeline",
    ]:
        assert mode in FEEDLESS_MODES

    captured = {}

    class Commands:
        @staticmethod
        def run_ml_stock_fundamentals_preflight(config):
            captured["mode"] = "preflight"

        @staticmethod
        def run_ml_stock_fundamentals_collect(config):
            captured["mode"] = "collect"

        @staticmethod
        def run_ml_stock_fundamentals_pipeline(config):
            captured["mode"] = "pipeline"
            captured["config"] = config

    monkeypatch.setattr(cli_dispatch, "import_module", lambda name: Commands)
    cli_dispatch.dispatch(SimpleNamespace(mode="ml-stock-fundamentals-preflight"), {"ml": {}}, None)
    assert captured["mode"] == "preflight"
    cli_dispatch.dispatch(SimpleNamespace(mode="ml-stock-fundamentals-collect"), {"ml": {}}, None)
    assert captured["mode"] == "collect"
    cli_dispatch.dispatch(SimpleNamespace(mode="ml-stock-fundamentals-pipeline"), {"ml": {}}, None)
    assert captured == {"mode": "pipeline", "config": {"ml": {}}}


def test_preflight_blocks_missing_live_user_agent_but_env_unblocks(monkeypatch, tmp_path):
    config = {"ml": {"stock_fundamentals": {"enabled": True, "output_dir": str(tmp_path / "out")}}}
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    paths = write_stock_fundamentals_preflight(config)
    payload = json.loads(paths.preflight_path.read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED"
    assert "SEC_USER_AGENT" in payload["blocking_reasons"][0]

    monkeypatch.setenv("SEC_USER_AGENT", "Fixture Bot contact@example.com")
    paths = write_stock_fundamentals_preflight(config)
    payload = json.loads(paths.preflight_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["user_agent_configured"] is True
    assert "contact@example.com" not in payload["user_agent_redacted"]


def test_stage_separation_collect_normalize_snapshots_enrich(tmp_path):
    raw_root = tmp_path / "raw"
    base_path = tmp_path / "base.parquet"
    rows = [_base_row("2023-06-15", "AAPL"), _base_row("2023-09-15", "AAPL")]
    write_stock_level_artifact(
        base_path,
        rows,
        fieldnames=list(rows[0]),
        config={"ml": {"stock_level_artifact_format": "parquet", "stock_level_parquet_compression": "zstd"}},
    )
    entity_dir = raw_root / "official_sec_companyfacts" / "CIK0000320193"
    entity_dir.mkdir(parents=True)
    raw = entity_dir / "companyfacts.json"
    raw.write_text(json.dumps(_companyfacts_payload()), encoding="utf-8")
    raw_hash = __import__("hashlib").sha256(raw.read_bytes()).hexdigest()
    raw.with_suffix(".metadata.json").write_text(
        json.dumps({"sha256": raw_hash, "retrieval_timestamp": "2024-01-01T00:00:00Z", "content_type": "application/json"}),
        encoding="utf-8",
    )
    config = {
        "ml": {
            "stock_fundamentals": {
                "enabled": True,
                "source_dataset_path": str(base_path),
                "output_dir": str(tmp_path / "out"),
                "symbols": ["AAPL"],
                "cik_by_symbol": {"AAPL": "320193"},
                "user_agent": "Fixture Bot contact@example.com",
                "collection": {"raw_root": str(raw_root), "request_delay_seconds": 0},
            }
        }
    }

    write_stock_fundamentals_collect(config)
    out = tmp_path / "out"
    assert (out / "fundamentals_raw_collection_manifest.json").exists()
    assert not (out / "fundamentals_normalized_facts.parquet").exists()

    write_stock_fundamentals_normalize(config)
    assert (out / "fundamentals_normalized_facts.parquet").exists()
    assert not (out / "fundamentals_point_in_time_snapshots.parquet").exists()

    write_stock_fundamentals_snapshots(config)
    assert (out / "fundamentals_point_in_time_snapshots.parquet").exists()
    assert not (out / "stock_level_prediction_artifacts_fundamentals_enriched.parquet").exists()

    write_stock_fundamentals_enrich(config)
    assert (out / "stock_level_prediction_artifacts_fundamentals_enriched.parquet").exists()


def test_cache_validation_detects_corrupt_or_identity_mismatch(tmp_path):
    entity_dir = tmp_path / "official_sec_companyfacts" / "CIK0000320193"
    entity_dir.mkdir(parents=True)
    raw = entity_dir / "companyfacts.json"
    raw.write_text(json.dumps(_companyfacts_payload()), encoding="utf-8")
    raw_hash = __import__("hashlib").sha256(raw.read_bytes()).hexdigest()
    raw.with_suffix(".metadata.json").write_text(json.dumps({"sha256": raw_hash, "retrieval_timestamp": "2024-01-01T00:00:00Z"}), encoding="utf-8")
    assert validate_cached_companyfacts(raw, expected_cik="320193")["cache_state"] == "valid_cached"
    assert validate_cached_companyfacts(raw, expected_cik="789019")["cache_state"] == "identity_mismatch"
    raw.write_text("{bad", encoding="utf-8")
    assert validate_cached_companyfacts(raw, expected_cik="320193")["cache_state"] == "corrupt"


def _mapping():
    return [
        {
            "symbol": "AAPL",
            "reporting_entity_id": "CIK0000320193",
            "provider_entity_id": "0000320193",
            "security_mapping_identity": "map-aapl",
            "mapping_status": "CURRENT STATIC",
        },
        {
            "symbol": "MSFT",
            "reporting_entity_id": "",
            "provider_entity_id": "",
            "security_mapping_identity": "map-msft",
            "mapping_status": "UNRESOLVED",
        },
    ]


def _base_row(date: str, symbol: str) -> dict:
    return {
        "rebalance_date": date,
        "symbol": symbol,
        "close": 10.0,
        "predicted_momentum_20d": 0.1,
        "predicted_momentum_60d": 0.2,
        "predicted_momentum_120d": 0.3,
        "predicted_risk_adjusted_momentum": 0.4,
        "predicted_volatility_20d": 0.2,
        "predicted_drawdown_60d": -0.1,
        "predicted_liquidity_score": 1.0,
        "actual_forward_return_10d": 0.01,
    }


def _companyfacts_payload():
    return {
        "cik": "0000320193",
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": [
                    _duration("2022-01-01", "2022-03-31", 80.0, 2022, "Q1", "10-Q", "2022-05-01", "prior-q1"),
                    _duration("2023-01-01", "2023-03-31", 100.0, 2023, "Q1", "10-Q", "2023-05-01", "orig-q1"),
                    _duration("2023-01-01", "2023-03-31", 120.0, 2023, "Q1", "10-Q/A", "2023-08-01", "amend-q1"),
                    _duration("2023-04-01", "2023-06-30", 140.0, 2023, "Q2", "10-Q", "2023-11-01", "future-q2"),
                ]}},
                "GrossProfit": {"units": {"USD": [
                    _duration("2023-01-01", "2023-03-31", 50.0, 2023, "Q1", "10-Q", "2023-05-01", "orig-q1"),
                    _duration("2023-01-01", "2023-03-31", 60.0, 2023, "Q1", "10-Q/A", "2023-08-01", "amend-q1"),
                ]}},
                "OperatingIncomeLoss": {"units": {"USD": [_duration("2023-01-01", "2023-03-31", 30.0, 2023, "Q1", "10-Q", "2023-05-01", "orig-q1")]}},
                "NetIncomeLoss": {"units": {"USD": [_duration("2023-01-01", "2023-03-31", 20.0, 2023, "Q1", "10-Q", "2023-05-01", "orig-q1")]}},
                "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [_duration("2023-01-01", "2023-03-31", 25.0, 2023, "Q1", "10-Q", "2023-05-01", "orig-q1")]}},
                "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [_duration("2023-01-01", "2023-03-31", 5.0, 2023, "Q1", "10-Q", "2023-05-01", "orig-q1")]}},
                "Assets": {"units": {"USD": [_instant("2023-03-31", 200.0, 2023, "Q1", "10-Q", "2023-05-01", "orig-q1")]}},
                "AssetsCurrent": {"units": {"USD": [_instant("2023-03-31", 100.0, 2023, "Q1", "10-Q", "2023-05-01", "orig-q1")]}},
                "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [_instant("2023-03-31", 30.0, 2023, "Q1", "10-Q", "2023-05-01", "orig-q1")]}},
                "Liabilities": {"units": {"USD": [_instant("2023-03-31", 90.0, 2023, "Q1", "10-Q", "2023-05-01", "orig-q1")]}},
                "LiabilitiesCurrent": {"units": {"USD": [_instant("2023-03-31", 40.0, 2023, "Q1", "10-Q", "2023-05-01", "orig-q1")]}},
                "LongTermDebtNoncurrent": {"units": {"USD": [_instant("2023-03-31", 50.0, 2023, "Q1", "10-Q", "2023-05-01", "orig-q1")]}},
                "StockholdersEquity": {"units": {"USD": [_instant("2023-03-31", 110.0, 2023, "Q1", "10-Q", "2023-05-01", "orig-q1")]}},
                "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": [_duration("2023-01-01", "2023-03-31", 10.0, 2023, "Q1", "10-Q", "2023-05-01", "orig-q1")]}},
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {"units": {"shares": [_instant("2023-03-31", 10.0, 2023, "Q1", "10-Q", "2023-05-01", "orig-q1")]}}
            },
        },
    }


def _duration(start, end, val, fy, fp, form, filed, accn):
    return {"start": start, "end": end, "val": val, "fy": fy, "fp": fp, "form": form, "filed": filed, "accn": accn}


def _instant(end, val, fy, fp, form, filed, accn):
    return {"end": end, "val": val, "fy": fy, "fp": fp, "form": form, "filed": filed, "accn": accn}
