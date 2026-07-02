from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from application.services.ml_commands_stock import (
    run_ml_stock_alpha_news_contract_ingest,
    run_ml_stock_alpha_news_coverage_audit,
    run_ml_stock_alpha_news_feature_diagnostics,
    run_ml_stock_alpha_news_pipeline_preflight,
    run_ml_stock_alpha_news_pipeline_inspect,
    run_ml_stock_alpha_news_provider_audit,
    run_ml_stock_alpha_news_provider_sample_check,
    run_ml_stock_alpha_news_source_diagnostics,
)
from config.config_loader import load_config
from core.research.framework.data import CsvRowRepository

from core.research.ml.stock_level.stock_alpha_news_contract import (
    REQUIRED_NEWS_AGGREGATE_FEATURES,
    REQUIRED_NEWS_CONTRACT_COLUMNS,
    REQUIRED_NEWS_FEATURE_COLUMNS,
    build_stock_alpha_news_features,
    check_news_transformer_readiness,
    validate_news_contract,
    write_stock_alpha_news_features,
    write_stock_alpha_news_features_from_config,
    write_stock_alpha_news_contract_validation,
)
from core.research.ml.stock_level.stock_alpha_news_contract_ingest import (
    write_stock_alpha_news_contract_ingest,
)
from core.research.ml.stock_level.stock_alpha_news_coverage_audit import (
    write_stock_alpha_news_coverage_audit,
)
from core.research.ml.stock_level.stock_alpha_news_feature_diagnostics import (
    write_stock_alpha_news_feature_diagnostics,
)
from core.research.ml.stock_level.news_sources import AlphaVantageNewsSource, CompanyPressReleaseRssSource, GdeltNewsSource, MassiveStockNewsSource, PROVIDER_METADATA, SecEdgarNewsSource
from urllib.error import HTTPError
from core.research.ml.stock_level.stock_alpha_news_free_source_collect import (
    write_stock_alpha_news_free_source_collect,
)
from core.research.ml.stock_level.stock_alpha_news_collection_plan import (
    write_stock_alpha_news_collection_plan,
)
from core.research.ml.stock_level.stock_alpha_news_daily_confirmation import (
    write_stock_alpha_news_daily_confirmation,
)
from core.research.ml.stock_level.stock_alpha_news_provider_audit import (
    write_stock_alpha_news_provider_audit,
)
from core.research.ml.stock_level.stock_alpha_news_provider_sample_check import (
    write_stock_alpha_news_provider_sample_check,
)
from core.research.ml.stock_level.stock_alpha_news_pipeline_preflight import (
    build_stock_alpha_news_pipeline_preflight,
    write_stock_alpha_news_pipeline_preflight,
)
from core.research.ml.stock_level.stock_alpha_news_pipeline_inspect import (
    write_stock_alpha_news_pipeline_inspect,
)
from core.research.ml.stock_level.stock_alpha_news_readiness_preflight import (
    build_stock_alpha_news_readiness_preflight,
    write_stock_alpha_news_readiness_preflight,
)
from core.research.ml.stock_level.stock_alpha_news_source_diagnostics import (
    write_stock_alpha_news_source_diagnostics,
)
from core.research.ml.stock_level.stock_alpha_news_source_setup_check import (
    write_stock_alpha_news_source_setup_check,
)


def test_valid_contract_passes_when_features_and_transformer_enabled(tmp_path):
    news = tmp_path / "news.csv"
    features = tmp_path / "features.csv"
    _write_news(news)
    _write_features(features)

    validation = validate_news_contract(
        _config(news, features, enable=True, min_symbol=1.0, min_date=1.0),
        _stock_rows(),
    )

    assert validation.available is True
    assert validation.contract_valid is True
    assert validation.aggregate_features_valid is True
    assert validation.symbol_coverage == 1.0
    assert validation.date_coverage == 1.0
    assert validation.payload()["research_only"] is True


def test_missing_required_fields_fails(tmp_path):
    news = tmp_path / "news.csv"
    news.write_text("article_id,symbol,published_at_utc\n1,AAPL,2024-01-01T00:00:00Z\n", encoding="utf-8")

    validation = validate_news_contract(_config(news), _stock_rows())

    assert validation.available is False
    assert validation.reason == "missing required news contract fields"
    assert "ingested_at" in validation.missing_fields


def test_phase6_required_raw_fields_include_event_relevance_and_novelty(tmp_path):
    news = tmp_path / "news.csv"
    news.write_text(
        "article_id,symbol,published_at_utc,ingested_at,source,headline,body_or_summary,sentiment_score,language\n"
        "1,AAPL,2024-01-01T00:00:00Z,2024-01-01T00:00:00Z,vendor,h,b,0.1,en\n",
        encoding="utf-8",
    )

    validation = validate_news_contract(_config(news), _stock_rows())

    assert set(REQUIRED_NEWS_CONTRACT_COLUMNS) >= {"event_type", "relevance_score", "novelty_score"}
    assert validation.available is False
    assert validation.reason == "missing required news contract fields"
    assert set(validation.missing_fields) >= {"event_type", "relevance_score", "novelty_score"}


def test_future_published_at_is_a_diagnostic_candidate_not_contract_failure(tmp_path):
    news = tmp_path / "news.csv"
    _write_news(news, published="2024-01-03T00:00:00Z", ingested="2024-01-01T00:00:00Z")

    validation = validate_news_contract(_config(news, min_symbol=0.0, min_date=0.0), _stock_rows())

    assert validation.contract_valid is True
    assert validation.future_article_count == 1


def test_future_ingested_at_is_a_diagnostic_candidate_not_contract_failure(tmp_path):
    news = tmp_path / "news.csv"
    _write_news(news, published="2024-01-01T00:00:00Z", ingested="2024-01-03T00:00:00Z")

    validation = validate_news_contract(_config(news, min_symbol=0.0, min_date=0.0), _stock_rows())

    assert validation.contract_valid is True
    assert validation.future_article_count == 1


def test_insufficient_coverage_fails(tmp_path):
    news = tmp_path / "news.csv"
    _write_news(news, symbol="AAPL")

    validation = validate_news_contract(_config(news, min_symbol=1.0, min_date=1.0), _stock_rows(symbols=("AAPL", "MSFT")))

    assert validation.available is False
    assert validation.reason == "news contract symbol coverage below minimum"
    assert validation.symbol_coverage == 0.5
    assert validation.missing_symbol_coverage == ("MSFT",)


def test_synthetic_zero_news_fake_coverage_is_rejected(tmp_path):
    news = tmp_path / "news.csv"
    _write_news(
        news,
        article_id="synthetic-zero-news-AAPL",
        source="synthetic",
        headline="",
        body="placeholder no_news coverage",
        sentiment="0.0",
        relevance="0.0",
        novelty="0.0",
        event_type="zero_news",
    )

    validation = validate_news_contract(_config(news, min_symbol=0.0, min_date=0.0), _stock_rows())

    assert validation.available is False
    assert validation.reason == "news contract contains synthetic zero-news fake coverage"
    assert validation.synthetic_zero_news_count == 1


def test_transformer_remains_unavailable_without_valid_contract(tmp_path):
    validation = validate_news_contract(
        {"ml": {"stock_alpha_news_enable_transformer": True}},
        _stock_rows(),
    )

    assert validation.available is False
    assert validation.reason == "missing ml.stock_alpha_news_contract_path"


def test_news_transformer_readiness_unavailable_when_features_missing(tmp_path):
    readiness = check_news_transformer_readiness(
        {
            "ml": {
                "stock_alpha_news_features_path": str(tmp_path / "missing_features.csv"),
                "stock_alpha_news_enable_transformer": True,
                "stock_alpha_news_min_symbol_coverage": 1.0,
                "stock_alpha_news_min_date_coverage": 1.0,
                **_guardrails(),
            }
        },
        _stock_rows(),
    )

    assert readiness.transformer_available is False
    assert readiness.unavailable_reason == "news_features_file_not_found"


def test_news_transformer_readiness_requires_enable_flag(tmp_path):
    features = tmp_path / "features.csv"
    _write_features(features)

    readiness = check_news_transformer_readiness(
        {
            "ml": {
                "stock_alpha_news_features_path": str(features),
                "stock_alpha_news_enable_transformer": False,
                **_guardrails(),
            }
        },
        _stock_rows(),
    )

    assert readiness.transformer_available is False
    assert readiness.unavailable_reason == "stock_alpha_news_enable_transformer_false"


def test_news_transformer_readiness_rejects_missing_required_columns(tmp_path):
    features = tmp_path / "features.csv"
    features.write_text("rebalance_date,symbol,news_random_score\n2024-01-02,AAPL,1\n", encoding="utf-8")

    readiness = check_news_transformer_readiness(
        {
            "ml": {
                "stock_alpha_news_features_path": str(features),
                "stock_alpha_news_enable_transformer": True,
                **_guardrails(),
            }
        },
        _stock_rows(),
    )

    assert readiness.transformer_available is False
    assert readiness.unavailable_reason == "missing_required_news_feature_columns"
    assert "news_count_1d" in readiness.required_columns_missing


def test_news_transformer_readiness_rejects_alignment_and_coverage_failures(tmp_path):
    features = tmp_path / "features.csv"
    _write_features(features, symbol="MSFT")

    readiness = check_news_transformer_readiness(
        {
            "ml": {
                "stock_alpha_news_features_path": str(features),
                "stock_alpha_news_enable_transformer": True,
                "stock_alpha_news_min_symbol_coverage": 1.0,
                "stock_alpha_news_min_date_coverage": 1.0,
                **_guardrails(),
            }
        },
        _stock_rows(symbols=("AAPL",)),
    )

    assert readiness.transformer_available is False
    assert readiness.unavailable_reason == "news_feature_symbol_coverage_below_minimum"
    assert readiness.aligned_stock_row_count == 0


def test_news_transformer_readiness_available_only_with_valid_features_and_enable_flag(tmp_path):
    features = tmp_path / "features.csv"
    _write_features(features)

    readiness = check_news_transformer_readiness(
        {
            "ml": {
                "stock_alpha_news_features_path": str(features),
                "stock_alpha_news_enable_transformer": True,
                "stock_alpha_news_min_symbol_coverage": 1.0,
                "stock_alpha_news_min_date_coverage": 1.0,
                **_guardrails(),
            }
        },
        _stock_rows(),
    )

    assert readiness.transformer_available is True
    assert readiness.unavailable_reason == ""
    assert set(readiness.required_columns_found) == set(REQUIRED_NEWS_FEATURE_COLUMNS)
    assert readiness.payload()["research_only"] is True


def test_news_transformer_readiness_rejects_feature_timestamp_leakage(tmp_path):
    features = tmp_path / "features.csv"
    _write_features(features, extra_fields={"published_at_utc": "2024-01-03T00:00:00Z"})

    readiness = check_news_transformer_readiness(
        {
            "ml": {
                "stock_alpha_news_features_path": str(features),
                "stock_alpha_news_enable_transformer": True,
                "stock_alpha_news_min_symbol_coverage": 1.0,
                "stock_alpha_news_min_date_coverage": 1.0,
                **_guardrails(),
            }
        },
        _stock_rows(),
    )

    assert readiness.transformer_available is False
    assert readiness.unavailable_reason == "news_feature_rows_contain_future_timestamps"
    assert readiness.pit_violation_count == 1


def test_news_readiness_preflight_blocks_when_transformer_disabled(tmp_path):
    features = tmp_path / "features.csv"
    _write_features(features)

    payload = build_stock_alpha_news_readiness_preflight(
        {
            "ml": {
                "stock_alpha_news_features_path": str(features),
                "stock_alpha_news_stock_rows_path": str(_write_stock_rows_csv(tmp_path)),
                "stock_alpha_news_enable_transformer": False,
                "stock_alpha_news_min_symbol_coverage": 1.0,
                "stock_alpha_news_min_date_coverage": 1.0,
                **_guardrails(),
            }
        }
    )

    assert payload["safe_to_train_news_transformer"] is False
    assert payload["readiness_available"] is False
    assert payload["enable_flag"] is False
    assert "stock_alpha_news_enable_transformer_false" in payload["blocking_issues"]
    assert payload["row_count"] == 1
    assert payload["symbol_count"] == 1
    assert payload["date_count"] == 1


def test_news_readiness_preflight_safe_only_with_enabled_valid_features_and_audit(tmp_path):
    features = tmp_path / "features.csv"
    _write_features(features)
    audit_dir = tmp_path / "news_features"
    audit_dir.mkdir()
    (audit_dir / "stock_alpha_news_features_audit.json").write_text(
        json.dumps({"pit_violation_count": 0}),
        encoding="utf-8",
    )

    payload = build_stock_alpha_news_readiness_preflight(
        {
            "ml": {
                "stock_alpha_news_features_path": str(features),
                "stock_alpha_news_stock_rows_path": str(_write_stock_rows_csv(tmp_path)),
                "stock_alpha_news_enable_transformer": True,
                "stock_alpha_news_min_symbol_coverage": 1.0,
                "stock_alpha_news_min_date_coverage": 1.0,
                **_guardrails(),
            }
        }
    )

    assert payload["safe_to_train_news_transformer"] is True
    assert payload["readiness_available"] is True
    assert payload["blocking_issues"] == []
    assert payload["pit_audit_summary"]["audit_metadata_available"] is True


def test_news_readiness_preflight_blocks_true_pit_violation_reported_by_audit(tmp_path):
    features = tmp_path / "features.csv"
    _write_features(features)
    audit_dir = tmp_path / "news_features"
    audit_dir.mkdir()
    (audit_dir / "stock_alpha_news_features_audit.json").write_text(
        json.dumps(
            {
                "future_article_candidate_count": 3,
                "future_article_excluded_count": 3,
                "pit_violation_count": 1,
            }
        ),
        encoding="utf-8",
    )

    payload = build_stock_alpha_news_readiness_preflight(
        {
            "ml": {
                "stock_alpha_news_features_path": str(features),
                "stock_alpha_news_stock_rows_path": str(_write_stock_rows_csv(tmp_path)),
                "stock_alpha_news_enable_transformer": True,
                "stock_alpha_news_min_symbol_coverage": 1.0,
                "stock_alpha_news_min_date_coverage": 1.0,
                **_guardrails(),
            }
        }
    )

    assert payload["safe_to_train_news_transformer"] is False
    assert "news_features_audit_reports_pit_violations" in payload["blocking_issues"]
    assert payload["pit_audit_summary"]["audit_pit_violation_count"] == 1


def test_news_readiness_preflight_writes_json_and_markdown(tmp_path):
    features = tmp_path / "features.csv"
    _write_features(features)

    paths = write_stock_alpha_news_readiness_preflight(
        {
            "ml": {
                "stock_alpha_report_root": str(tmp_path / "reports"),
                "stock_alpha_run_size": "dev",
                "stock_alpha_news_features_path": str(features),
                "stock_alpha_news_stock_rows_path": str(_write_stock_rows_csv(tmp_path)),
                "stock_alpha_news_enable_transformer": False,
                **_guardrails(),
            }
        }
    )

    assert paths.json_path.exists()
    assert paths.markdown_path.exists()
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["safe_to_train_news_transformer"] is False


def test_news_contract_ingest_writes_canonical_csv_and_audit(tmp_path):
    raw = tmp_path / "raw_news.csv"
    contract = tmp_path / "stock_alpha_news_contract.csv"
    audit_dir = tmp_path / "audit"
    _write_raw_news_csv(raw, symbol=" aapl ", event_type="EPS beat")

    paths = write_stock_alpha_news_contract_ingest(_ingest_config(raw, contract, audit_dir))

    rows = CsvRowRepository().read(paths.contract_path)
    audit = json.loads(paths.audit_json_path.read_text(encoding="utf-8"))
    assert paths.contract_path == contract
    assert paths.audit_markdown_path.exists()
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["event_type"] == "earnings"
    assert rows[0]["published_at_utc"].endswith("Z")
    assert list(rows[0]) == list(REQUIRED_NEWS_CONTRACT_COLUMNS)
    assert audit["raw_row_count"] == 1
    assert audit["valid_row_count"] == 1
    assert audit["safe_to_generate_features"] is True
    assert audit["event_type_counts"] == {"earnings": 1}


def test_news_contract_ingest_missing_source_path_fails_without_contract(tmp_path):
    raw = tmp_path / "missing.csv"
    contract = tmp_path / "stock_alpha_news_contract.csv"

    with pytest.raises(FileNotFoundError, match="raw source file not found"):
        write_stock_alpha_news_contract_ingest(_ingest_config(raw, contract, tmp_path / "audit"))

    assert not contract.exists()


def test_news_contract_ingest_wrapper_reports_missing_source_cleanly(tmp_path, capsys):
    raw = tmp_path / "missing.csv"
    contract = tmp_path / "stock_alpha_news_contract.csv"

    with pytest.raises(SystemExit) as exc:
        run_ml_stock_alpha_news_contract_ingest(
            _ingest_config(raw, contract, tmp_path / "audit")
        )

    output = capsys.readouterr().out
    assert exc.value.code == 1
    assert "STOCK-ALPHA NEWS CONTRACT INGEST" in output
    assert "mode=research" in output
    assert "safe_to_generate_features=false" in output
    assert "blocking_issue=stock-alpha news raw source file not found" in output
    assert str(raw) in output
    assert not contract.exists()


def test_news_contract_ingest_dedupes_duplicate_article_id(tmp_path):
    raw = tmp_path / "raw_news.csv"
    contract = tmp_path / "stock_alpha_news_contract.csv"
    _write_raw_news_csv(raw, rows=[_raw_news_row(article_id="dup-1"), _raw_news_row(article_id="dup-1", symbol="MSFT")])

    paths = write_stock_alpha_news_contract_ingest(_ingest_config(raw, contract, tmp_path / "audit"))

    rows = CsvRowRepository().read(paths.contract_path)
    audit = json.loads(paths.audit_json_path.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAPL"
    assert audit["duplicate_article_id_count"] == 1


def test_news_contract_ingest_rejects_ingested_before_published(tmp_path):
    raw = tmp_path / "raw_news.csv"
    contract = tmp_path / "stock_alpha_news_contract.csv"
    _write_raw_news_csv(
        raw,
        published="2024-01-02T10:00:00Z",
        ingested="2024-01-02T09:59:00Z",
    )

    paths = write_stock_alpha_news_contract_ingest(_ingest_config(raw, contract, tmp_path / "audit"))

    rows = CsvRowRepository().read(paths.contract_path)
    audit = json.loads(paths.audit_json_path.read_text(encoding="utf-8"))
    assert rows == []
    assert audit["ingested_before_published_count"] == 1
    assert audit["safe_to_generate_features"] is False


def test_news_contract_ingest_missing_required_input_columns_blocks_generation(tmp_path):
    raw = tmp_path / "raw_news.csv"
    contract = tmp_path / "stock_alpha_news_contract.csv"
    raw.write_text("article_id,symbol,published_at_utc\n1,AAPL,2024-01-01T00:00:00Z\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        write_stock_alpha_news_contract_ingest(_ingest_config(raw, contract, tmp_path / "audit"))

    assert not contract.exists()


def test_news_contract_ingest_uppercases_symbols(tmp_path):
    raw = tmp_path / "raw_news.csv"
    contract = tmp_path / "stock_alpha_news_contract.csv"
    _write_raw_news_csv(raw, symbol=" msft ")

    paths = write_stock_alpha_news_contract_ingest(_ingest_config(raw, contract, tmp_path / "audit"))

    rows = CsvRowRepository().read(paths.contract_path)
    assert rows[0]["symbol"] == "MSFT"


def test_news_contract_ingest_normalizes_event_types(tmp_path):
    raw = tmp_path / "raw_news.csv"
    contract = tmp_path / "stock_alpha_news_contract.csv"
    _write_raw_news_csv(
        raw,
        rows=[
            _raw_news_row(article_id="1", event_type="M&A"),
            _raw_news_row(article_id="2", event_type="CEO change"),
            _raw_news_row(article_id="3", event_type="unexpected blob"),
        ],
    )

    paths = write_stock_alpha_news_contract_ingest(_ingest_config(raw, contract, tmp_path / "audit"))

    rows = CsvRowRepository().read(paths.contract_path)
    assert [row["event_type"] for row in rows] == ["mna", "management", "other"]


def test_tiny_raw_provider_ingest_smoke_runs_contract_features_and_preflight():
    ingest_config = load_config(
        "config/config.stock_alpha_news_contract_ingest_tiny_fixture.yaml",
        overlay_project_config=True,
    )
    ingest_paths = write_stock_alpha_news_contract_ingest(ingest_config)

    contract_rows = CsvRowRepository().read(ingest_paths.contract_path)
    ingest_audit = json.loads(ingest_paths.audit_json_path.read_text(encoding="utf-8"))
    assert ingest_paths.audit_markdown_path.exists()
    assert len(contract_rows) == 5
    assert ingest_audit["raw_row_count"] == 7
    assert ingest_audit["valid_row_count"] == 5
    assert ingest_audit["duplicate_article_id_count"] == 1
    assert ingest_audit["ingested_before_published_count"] == 1
    assert ingest_audit["event_type_counts"] == {
        "analyst": 1,
        "earnings": 1,
        "guidance": 1,
        "litigation": 1,
        "mna": 1,
    }
    assert {row["symbol"] for row in contract_rows} == {"AAPL", "MSFT"}
    assert "tiny-bad-msft-1" not in {row["article_id"] for row in contract_rows}
    assert sum(1 for row in contract_rows if row["article_id"] == "tiny-analyst-aapl-1") == 1

    features_config = load_config(
        "config/config.stock_alpha_news_features_tiny_ingest_fixture.yaml",
        overlay_project_config=True,
    )
    features_paths = write_stock_alpha_news_features_from_config(features_config)
    feature_rows = CsvRowRepository().read(features_paths.features_csv_path)

    assert features_paths.audit_markdown_path.exists()
    assert set(REQUIRED_NEWS_FEATURE_COLUMNS).issubset(feature_rows[0])
    assert len(feature_rows) == 10
    assert _feature_row(feature_rows, "2024-01-06", "AAPL")["mna_news_count_30d"] == "1"
    assert _feature_row(feature_rows, "2024-01-03", "AAPL")["analyst_news_count_14d"] == "1"
    assert _feature_row(feature_rows, "2024-01-03", "AAPL")["negative_news_count_7d"] == "1"
    assert _feature_row(feature_rows, "2024-01-05", "MSFT")["guidance_news_count_30d"] == "1"
    assert _feature_row(feature_rows, "2024-01-05", "MSFT")["litigation_news_count_30d"] == "1"
    assert _feature_row(feature_rows, "2024-01-05", "MSFT")["negative_news_count_7d"] == "1"
    no_news_row = _feature_row(feature_rows, "2024-01-02", "MSFT")
    assert no_news_row["avg_sentiment_1d"] == ""
    assert no_news_row["news_has_coverage_30d"] == "False"

    preflight_config = load_config(
        "config/config.stock_alpha_news_readiness_preflight_tiny_ingest_fixture.yaml",
        overlay_project_config=True,
    )
    preflight_paths = write_stock_alpha_news_readiness_preflight(preflight_config)
    preflight = json.loads(preflight_paths.json_path.read_text(encoding="utf-8"))
    assert preflight_paths.markdown_path.exists()
    assert preflight["source_features_exists"] is True
    assert preflight["stock_rows_exists"] is True
    assert preflight["safe_to_train_news_transformer"] is False
    assert preflight["enable_flag"] is False
    assert "stock_alpha_news_enable_transformer_false" in preflight["blocking_issues"]


def test_alias_provider_column_map_ingest_writes_canonical_contract():
    config = load_config(
        "config/config.stock_alpha_news_contract_ingest_alias_tiny_fixture.yaml",
        overlay_project_config=True,
    )

    paths = write_stock_alpha_news_contract_ingest(config)

    rows = CsvRowRepository().read(paths.contract_path)
    audit = json.loads(paths.audit_json_path.read_text(encoding="utf-8"))
    assert paths.audit_markdown_path.exists()
    assert len(rows) == 4
    assert list(rows[0]) == list(REQUIRED_NEWS_CONTRACT_COLUMNS)
    assert "ticker" not in rows[0]
    assert "published_at" not in rows[0]
    assert {row["symbol"] for row in rows} == {"AAPL", "MSFT"}
    assert _contract_row(rows, "alias-guidance-msft-1")["symbol"] == "MSFT"
    assert _contract_row(rows, "alias-mna-aapl-1")["event_type"] == "mna"
    assert sum(1 for row in rows if row["article_id"] == "alias-analyst-aapl-1") == 1
    assert "alias-bad-msft-1" not in {row["article_id"] for row in rows}
    assert audit["provider_column_map_used"] is True
    assert audit["provider_column_map"]["article_id"] == "id"
    assert audit["missing_mapped_provider_columns"] == []
    assert audit["unmapped_provider_columns"] == ["provider_note"]
    assert audit["canonical_columns_after_mapping"] == list(REQUIRED_NEWS_CONTRACT_COLUMNS)
    assert audit["duplicate_article_id_count"] == 1
    assert audit["ingested_before_published_count"] == 1


def test_alias_provider_column_map_missing_provider_column_fails_without_contract(tmp_path):
    raw = tmp_path / "alias_missing.csv"
    contract = tmp_path / "stock_alpha_news_contract.csv"
    raw.write_text(
        "id,ticker,published_at,provider,title,summary,sentiment,relevance,novelty,category,lang\n"
        "1,AAPL,2024-01-01T00:00:00Z,vendor,title,summary,0.1,0.9,0.8,earnings,en\n",
        encoding="utf-8",
    )

    config = _ingest_config(raw, contract, tmp_path / "audit")
    config["ml"]["stock_alpha_news_provider_column_map"] = {
        "article_id": "id",
        "symbol": "ticker",
        "published_at_utc": "published_at",
        "source": "provider",
        "headline": "title",
        "body_or_summary": "summary",
        "sentiment_score": "sentiment",
        "relevance_score": "relevance",
        "novelty_score": "novelty",
        "event_type": "category",
        "language": "lang",
        "ingested_at": "collected_at",
    }

    with pytest.raises(ValueError, match="missing mapped provider columns"):
        write_stock_alpha_news_contract_ingest(config)

    audit_path = tmp_path / "audit" / "stock_alpha_news_contract_ingest_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert not contract.exists()
    assert audit["provider_column_map_used"] is True
    assert audit["missing_mapped_provider_columns"] == ["collected_at"]
    assert audit["safe_to_generate_features"] is False


def test_provider_audit_alias_fixture_reports_quality_metrics():
    config = load_config(
        "config/config.stock_alpha_news_provider_audit_alias_tiny_fixture.yaml",
        overlay_project_config=True,
    )

    paths = write_stock_alpha_news_provider_audit(config)

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert paths.markdown_path.exists()
    assert payload["safe_for_pit_research"] is True
    assert payload["provider_column_map_used"] is True
    assert payload["provider_column_map"]["article_id"] == "id"
    assert payload["missing_mapped_provider_columns"] == []
    assert payload["raw_row_count"] == 6
    assert payload["article_id_count"] == 5
    assert payload["duplicate_article_id_count"] == 1
    assert payload["ingested_before_published_count"] == 1
    assert payload["invalid_timestamp_count"] == 0
    assert payload["symbol_count"] == 2
    assert payload["source_count"] == 1
    assert payload["language_counts"] == {"en": 6}
    assert payload["event_type_counts"]["mna"] == 1
    assert payload["article_count_by_symbol"] == {"AAPL": 4, "MSFT": 2}
    assert payload["article_count_by_source"] == {"alias_vendor": 6}
    assert payload["sentiment_present_count"] == 6
    assert payload["sentiment_missing_count"] == 0
    assert payload["relevance_present_count"] == 6
    assert payload["novelty_present_count"] == 6


def test_provider_audit_explains_sec_edgar_duplicate_headlines(tmp_path):
    raw = tmp_path / "provider.csv"
    rows = [
        _collected_news_row(f"sec-{index}", "sec_edgar")
        | {
            "symbol": "AAPL",
            "source": "SEC EDGAR",
            "headline": "AAPL filed Form 4",
        }
        for index in range(3)
    ] + [
        _collected_news_row("alpha-1", "alpha_vantage")
        | {"symbol": "MSFT", "headline": "MSFT reports earnings"}
    ]
    with raw.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[*REQUIRED_NEWS_CONTRACT_COLUMNS, "provider", "provider_article_id", "provider_url"],
        )
        writer.writeheader()
        writer.writerows(rows)
    config = {"ml": {
        "stock_alpha_news_raw_path": str(raw),
        "stock_alpha_news_provider_audit_dir": str(tmp_path / "audit"),
        "stock_alpha_news_provider_audit_min_symbol_count": 25,
        "stock_alpha_news_provider_audit_max_duplicate_headline_rate": 0.05,
    }}

    paths = write_stock_alpha_news_provider_audit(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))

    assert payload["threshold_comparisons"]["symbol_count"] == {
        "actual": 2,
        "minimum": 25.0,
        "passes": False,
    }
    assert payload["threshold_comparisons"]["duplicate_headline_rate"]["passes"] is False
    assert payload["duplicate_headline_count_by_provider"] == {
        "alpha_vantage": 0,
        "sec_edgar": 2,
    }
    assert payload["symbol_distribution_by_provider"]["sec_edgar"] == {"AAPL": 3}
    assert payload["duplicate_headline_examples"][0]["headline"] == "AAPL filed Form 4"
    assert payload["sec_edgar_generic_headline_count"] == 3
    assert payload["duplicates_mainly_sec_edgar_generic_filings"] is True


def test_provider_audit_missing_mapped_provider_column_blocks(tmp_path):
    raw = tmp_path / "alias_missing.csv"
    raw.write_text(
        "id,ticker,published_at,provider,title,summary,sentiment,relevance,novelty,category,lang\n"
        "1,AAPL,2024-01-01T00:00:00Z,vendor,title,summary,0.1,0.9,0.8,earnings,en\n",
        encoding="utf-8",
    )
    config = {
        "ml": {
            "stock_alpha_news_raw_path": str(raw),
            "stock_alpha_news_provider_audit_dir": str(tmp_path / "audit"),
            "stock_alpha_news_provider_column_map": {
                "article_id": "id",
                "symbol": "ticker",
                "published_at_utc": "published_at",
                "source": "provider",
                "headline": "title",
                "body_or_summary": "summary",
                "sentiment_score": "sentiment",
                "relevance_score": "relevance",
                "novelty_score": "novelty",
                "event_type": "category",
                "language": "lang",
                "ingested_at": "collected_at",
            },
        }
    }

    paths = write_stock_alpha_news_provider_audit(config)

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["safe_for_pit_research"] is False
    assert payload["missing_mapped_provider_columns"] == ["collected_at"]
    assert any("missing mapped provider columns" in issue for issue in payload["blocking_issues"])


def test_provider_audit_wrapper_reports_missing_source_cleanly(tmp_path, capsys):
    raw = tmp_path / "missing_provider.csv"
    contract = tmp_path / "stock_alpha_news_contract.csv"
    features = tmp_path / "stock_alpha_news_features.csv"

    run_ml_stock_alpha_news_provider_audit(
        {
            "ml": {
                "stock_alpha_news_raw_path": str(raw),
                "stock_alpha_news_provider_audit_dir": str(tmp_path / "audit"),
            }
        }
    )

    output = capsys.readouterr().out
    payload = json.loads((tmp_path / "audit" / "stock_alpha_news_provider_audit.json").read_text(encoding="utf-8"))
    assert "STOCK-ALPHA NEWS PROVIDER AUDIT" in output
    assert "mode=research | inspection_only=true" in output
    assert "safe_for_pit_research=false" in output
    assert "blocking_issue=raw source file not found" in output
    assert str(raw) in output
    assert payload["safe_for_pit_research"] is False
    assert not contract.exists()
    assert not features.exists()


def test_tiny_news_coverage_audit_reports_alignment_metrics():
    ingest_config = load_config(
        "config/config.stock_alpha_news_contract_ingest_tiny_fixture.yaml",
        overlay_project_config=True,
    )
    write_stock_alpha_news_contract_ingest(ingest_config)
    config = load_config(
        "config/config.stock_alpha_news_coverage_audit_tiny_ingest_fixture.yaml",
        overlay_project_config=True,
    )

    paths = write_stock_alpha_news_coverage_audit(config)

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert paths.markdown_path.exists()
    assert payload["safe_for_feature_generation"] is True
    assert payload["news_row_count"] == 5
    assert payload["stock_row_count"] == 10
    assert payload["news_symbol_count"] == 2
    assert payload["stock_symbol_count"] == 2
    assert payload["pre_pit_symbol_overlap_count"] == 2
    assert payload["pre_pit_symbol_overlap"] == ["AAPL", "MSFT"]
    assert payload["rebalance_date_count"] == 5
    assert payload["news_published_at_utc_min"] == "2024-01-01T10:00:00Z"
    assert payload["news_published_at_utc_max"] == "2024-01-05T13:00:00Z"
    assert payload["stock_rebalance_date_min"] == "2024-01-02"
    assert payload["stock_rebalance_date_max"] == "2024-01-06"
    assert payload["covered_symbol_count"] == 2
    assert payload["symbol_coverage"] == 1.0
    assert payload["covered_rebalance_date_count"] == 5
    assert payload["date_coverage"] == 1.0
    assert payload["covered_stock_row_count"] == 8
    assert payload["stock_row_coverage"] == 0.8
    assert payload["no_news_stock_row_count"] == 2
    assert payload["no_news_stock_row_rate"] == 0.2
    assert payload["event_type_counts"] == {
        "analyst": 1,
        "earnings": 1,
        "guidance": 1,
        "litigation": 1,
        "mna": 1,
    }
    assert payload["event_type_covered_stock_rows"]["mna"] == 1
    assert payload["freshness_bucket_counts"]["30d"] > 0
    assert payload["same_symbol_stock_article_pair_count"] == 25
    assert payload["future_article_candidate_count"] == 10
    assert payload["future_article_excluded_count"] == 10
    assert payload["published_after_rebalance_count"] == 10
    assert payload["ingested_after_rebalance_count"] == 10
    assert payload["pit_violation_count"] == 0
    assert "future article candidates correctly excluded" in payload["warning_issues"][1]


def test_news_coverage_audit_reports_ingestion_time_alignment_blocker(tmp_path):
    contract = tmp_path / "contract.csv"
    stock_rows = tmp_path / "stock_rows.csv"
    contract.write_text(
        "\n".join(
            [
                "article_id,symbol,published_at_utc,source,headline,body_or_summary,sentiment_score,relevance_score,novelty_score,event_type,language,ingested_at",
                "a1,AAPL,2026-01-01T10:00:00Z,vendor,AAPL update,body,0.1,0.9,0.5,analyst,en,2026-07-02T08:00:00Z",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    stock_rows.write_text(
        "rebalance_date,symbol\n2026-04-20,AAPL\n",
        encoding="utf-8",
    )

    paths = write_stock_alpha_news_coverage_audit(
        {
            "ml": {
                "stock_alpha_news_contract_path": str(contract),
                "stock_alpha_news_stock_rows_path": str(stock_rows),
                "stock_alpha_news_coverage_audit_dir": str(tmp_path / "audit"),
            }
        }
    )

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["safe_for_feature_generation"] is False
    assert payload["pit_policy"] == "strict_collected_at"
    assert payload["eligibility_timestamp_field"] == "collected_at_utc"
    assert payload["historical_provider_availability_assumed"] is False
    assert payload["production_pit_validated"] is True
    assert payload["pre_pit_symbol_overlap_count"] == 1
    assert payload["same_symbol_stock_article_pair_count"] == 1
    assert payload["future_article_candidate_count"] == 1
    assert payload["published_after_rebalance_count"] == 0
    assert payload["ingested_after_rebalance_count"] == 1
    assert payload["covered_stock_row_count"] == 0
    assert "all same-symbol articles fail PIT ingested_at alignment" in payload["blocking_issues"]
    assert any("ingested_at is after rebalance_date" in issue for issue in payload["warning_issues"])


def test_news_coverage_audit_provider_available_policy_uses_explicit_availability_lag(tmp_path):
    contract = tmp_path / "contract.csv"
    stock_rows = tmp_path / "stock_rows.csv"
    contract.write_text(
        "\n".join(
            [
                "article_id,symbol,published_at_utc,source,headline,body_or_summary,sentiment_score,relevance_score,novelty_score,event_type,language,ingested_at",
                "a1,AAPL,2026-01-01T10:00:00Z,vendor,AAPL update,body,0.1,0.9,0.5,analyst,en,2026-07-02T08:00:00Z",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    stock_rows.write_text(
        "rebalance_date,symbol\n2026-01-03,AAPL\n",
        encoding="utf-8",
    )

    paths = write_stock_alpha_news_coverage_audit(
        {
            "ml": {
                "stock_alpha_news_contract_path": str(contract),
                "stock_alpha_news_stock_rows_path": str(stock_rows),
                "stock_alpha_news_coverage_audit_dir": str(tmp_path / "audit"),
                "stock_alpha_news_pit_policy": "provider_available_at",
                "stock_alpha_news_availability_lag_hours": 24,
                "stock_alpha_news_historical_provider_availability_enabled": True,
            }
        }
    )

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["safe_for_feature_generation"] is True
    assert payload["pit_policy"] == "provider_available_at"
    assert payload["eligibility_timestamp_field"] == "available_at_utc"
    assert payload["historical_provider_availability_assumed"] is True
    assert payload["production_pit_validated"] is False
    assert payload["news_available_at_utc_min"] == "2026-01-02T10:00:00Z"
    assert payload["covered_stock_row_count"] == 1
    assert payload["available_after_rebalance_count"] == 0
    assert payload["ingested_after_rebalance_count"] == 0
    assert payload["warning_issues"] == []


def test_news_coverage_audit_provider_available_policy_requires_explicit_research_flag(tmp_path):
    contract = tmp_path / "contract.csv"
    stock_rows = tmp_path / "stock_rows.csv"
    contract.write_text(
        "\n".join(
            [
                "article_id,symbol,published_at_utc,source,headline,body_or_summary,sentiment_score,relevance_score,novelty_score,event_type,language,ingested_at",
                "a1,AAPL,2026-01-01T10:00:00Z,vendor,AAPL update,body,0.1,0.9,0.5,analyst,en,2026-07-02T08:00:00Z",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    stock_rows.write_text(
        "rebalance_date,symbol\n2026-01-03,AAPL\n",
        encoding="utf-8",
    )

    paths = write_stock_alpha_news_coverage_audit(
        {
            "ml": {
                "stock_alpha_news_contract_path": str(contract),
                "stock_alpha_news_stock_rows_path": str(stock_rows),
                "stock_alpha_news_coverage_audit_dir": str(tmp_path / "audit"),
                "stock_alpha_news_pit_policy": "provider_available_at",
                "stock_alpha_news_availability_lag_hours": 24,
            }
        }
    )

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["safe_for_feature_generation"] is False
    assert payload["historical_provider_availability_assumed"] is False
    assert (
        "provider_available_at PIT policy requires "
        "stock_alpha_news_historical_provider_availability_enabled=true"
    ) in payload["blocking_issues"]


def test_news_coverage_audit_missing_contract_path_blocks_cleanly(tmp_path, capsys):
    contract = tmp_path / "missing_contract.csv"
    features = tmp_path / "stock_alpha_news_features.csv"

    run_ml_stock_alpha_news_coverage_audit(
        {
            "ml": {
                "stock_alpha_news_contract_path": str(contract),
                "stock_alpha_news_stock_rows_path": "tests/fixtures/stock_alpha_news/stock_rows_tiny.csv",
                "stock_alpha_news_coverage_audit_dir": str(tmp_path / "audit"),
            }
        }
    )

    output = capsys.readouterr().out
    payload = json.loads((tmp_path / "audit" / "stock_alpha_news_coverage_audit.json").read_text(encoding="utf-8"))
    assert "STOCK-ALPHA NEWS COVERAGE AUDIT" in output
    assert "safe_for_feature_generation=false" in output
    assert "blocking_issue=news contract file not found" in output
    assert payload["safe_for_feature_generation"] is False
    assert not contract.exists()
    assert not features.exists()


def test_news_coverage_audit_missing_stock_rows_path_blocks_cleanly(tmp_path, capsys):
    contract = tmp_path / "contract.csv"
    _write_news(contract)

    run_ml_stock_alpha_news_coverage_audit(
        {
            "ml": {
                "stock_alpha_news_contract_path": str(contract),
                "stock_alpha_news_stock_rows_path": str(tmp_path / "missing_stock_rows.csv"),
                "stock_alpha_news_coverage_audit_dir": str(tmp_path / "audit"),
            }
        }
    )

    output = capsys.readouterr().out
    payload = json.loads((tmp_path / "audit" / "stock_alpha_news_coverage_audit.json").read_text(encoding="utf-8"))
    assert "safe_for_feature_generation=false" in output
    assert "blocking_issue=stock rows file not found" in output
    assert payload["safe_for_feature_generation"] is False


def test_news_coverage_audit_missing_required_columns_blocks(tmp_path):
    contract = tmp_path / "contract.csv"
    stock_rows = tmp_path / "stock_rows.csv"
    contract.write_text("article_id,symbol,published_at_utc\n1,AAPL,2024-01-01T00:00:00Z\n", encoding="utf-8")
    stock_rows.write_text("rebalance_date,not_symbol\n2024-01-02,AAPL\n", encoding="utf-8")

    paths = write_stock_alpha_news_coverage_audit(
        {
            "ml": {
                "stock_alpha_news_contract_path": str(contract),
                "stock_alpha_news_stock_rows_path": str(stock_rows),
                "stock_alpha_news_coverage_audit_dir": str(tmp_path / "audit"),
            }
        }
    )

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["safe_for_feature_generation"] is False
    assert "source" in payload["missing_required_news_columns"]
    assert payload["missing_required_stock_row_columns"] == ["symbol"]
    assert any("missing required news contract columns" in issue for issue in payload["blocking_issues"])


def test_contract_report_writes_json_markdown_and_coverage(tmp_path):
    news = tmp_path / "news.csv"
    _write_news(news)
    paths = write_stock_alpha_news_contract_validation(
        {
            "ml": {
                "stock_alpha_report_root": str(tmp_path / "reports"),
                "stock_alpha_run_size": "dev",
                "stock_alpha_news_contract_path": str(news),
                "stock_alpha_news_min_symbol_coverage": 1.0,
                "stock_alpha_news_min_date_coverage": 1.0,
            }
        },
        _stock_rows(),
    )

    assert paths.json_path.exists()
    assert paths.markdown_path.exists()
    assert paths.coverage_csv_path.exists()


def test_news_feature_aggregation_enforces_point_in_time_windows():
    news_rows = [
        _news_row("a1", "AAPL", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "0.5", "earnings"),
        _news_row("a2", "AAPL", "2024-01-03T00:00:00Z", "2024-01-03T00:00:00Z", "-0.5", "analyst"),
    ]
    stock_rows = [
        {"rebalance_date": "2024-01-02", "symbol": "AAPL"},
        {"rebalance_date": "2024-01-04", "symbol": "AAPL"},
    ]

    features, audit = build_stock_alpha_news_features(news_rows, stock_rows)

    first, second = features
    assert first["news_count_3d"] == 1
    assert first["avg_sentiment_3d"] == 0.5
    assert first["analyst_news_count_14d"] == 0
    assert second["news_count_3d"] == 2
    assert second["negative_news_count_7d"] == 1
    assert second["analyst_news_count_14d"] == 1
    assert audit["point_in_time_filters"]["future_statistics_used"] is False
    assert audit["synthetic_news_features_created"] is False
    assert audit["raw_article_count"] == 2
    assert audit["valid_article_count"] == 2
    assert audit["event_type_coverage"] == {"analyst": 1, "earnings": 1}
    assert audit["symbol_coverage"] == 1.0
    assert audit["date_coverage"] == 1.0


def test_news_feature_aggregation_does_not_count_future_articles_in_windows():
    news_rows = [
        _news_row("a1", "AAPL", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "0.5", "earnings"),
        _news_row("a2", "AAPL", "2024-01-03T00:00:00Z", "2024-01-03T00:00:00Z", "-0.5", "analyst"),
    ]

    features, audit = build_stock_alpha_news_features(
        news_rows,
        [{"rebalance_date": "2024-01-02", "symbol": "AAPL"}],
    )

    assert features[0]["news_count_7d"] == 1
    assert features[0]["analyst_news_count_14d"] == 0
    assert audit["future_article_candidate_count"] == 1
    assert audit["future_article_excluded_count"] == 1
    assert audit["pit_violation_count"] == 0
    assert audit["pit_violations_count"] == 0


def test_negative_news_count_uses_configurable_threshold():
    news_rows = [
        _news_row("a1", "AAPL", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "-0.05", "analyst"),
        _news_row("a2", "AAPL", "2024-01-01T01:00:00Z", "2024-01-01T01:00:00Z", "-0.25", "guidance"),
    ]

    default_features, _ = build_stock_alpha_news_features(
        news_rows,
        [{"rebalance_date": "2024-01-02", "symbol": "AAPL"}],
    )
    threshold_features, _ = build_stock_alpha_news_features(
        news_rows,
        [{"rebalance_date": "2024-01-02", "symbol": "AAPL"}],
        negative_sentiment_threshold=-0.10,
    )

    assert default_features[0]["negative_news_count_7d"] == 2
    assert threshold_features[0]["negative_news_count_7d"] == 1


def test_news_feature_aggregation_keeps_missing_news_as_missing_not_neutral():
    features, _ = build_stock_alpha_news_features(
        [_news_row("a1", "MSFT", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "0.5", "earnings")],
        [{"rebalance_date": "2024-01-02", "symbol": "AAPL"}],
    )

    row = features[0]
    assert row["news_count_1d"] == 0
    assert row["avg_sentiment_1d"] == ""
    assert row["avg_sentiment_3d"] == ""
    assert row["sentiment_change_3d"] == ""
    assert row["news_has_coverage_30d"] is False
    assert row["news_count_7d"] == 0


def test_news_feature_writer_outputs_features_and_audit_then_gate_can_validate(tmp_path):
    news = tmp_path / "news.csv"
    features_path = tmp_path / "features.csv"
    _write_news(news)

    paths = write_stock_alpha_news_features(
        {
            "ml": {
                "stock_alpha_report_root": str(tmp_path / "reports"),
                "stock_alpha_run_size": "dev",
                "stock_alpha_news_contract_path": str(news),
                "stock_alpha_news_features_path": str(features_path),
                "stock_alpha_news_min_symbol_coverage": 1.0,
                "stock_alpha_news_min_date_coverage": 1.0,
                **_guardrails(),
            }
        },
        _stock_rows(),
    )

    assert paths.features_csv_path == features_path
    assert paths.features_csv_path.exists()
    assert paths.audit_json_path.exists()
    assert paths.audit_markdown_path.exists()
    audit = json.loads(paths.audit_json_path.read_text(encoding="utf-8"))
    assert audit["source_news_path"] == str(news)
    assert audit["output_features_path"] == str(features_path)
    assert audit["article_count"] == 1
    assert audit["symbol_count"] == 1
    assert audit["rebalance_date_count"] == 1
    assert audit["feature_row_count"] == 1
    assert audit["pit_violations_count"] == 0
    assert audit["pit_violation_count"] == 0
    assert audit["missing_or_invalid_timestamp_count"] == 0
    assert audit["missing_feature_counts"]["avg_sentiment_1d"] == 0
    assert audit["required_news_contract_columns"] == list(REQUIRED_NEWS_CONTRACT_COLUMNS)
    assert audit["required_news_feature_columns"] == list(REQUIRED_NEWS_FEATURE_COLUMNS)
    assert audit["transformer_available"] is False

    disabled = validate_news_contract(
        _config(news, features_path, enable=False, min_symbol=1.0, min_date=1.0),
        _stock_rows(),
    )
    enabled = validate_news_contract(
        _config(news, features_path, enable=True, min_symbol=1.0, min_date=1.0),
        _stock_rows(),
    )
    assert disabled.available is False
    assert disabled.contract_valid is True
    assert disabled.aggregate_features_valid is True
    assert enabled.available is True
    assert enabled.aggregate_features_valid is True


def test_news_feature_writer_from_config_reads_stock_rows_file(tmp_path):
    news = tmp_path / "news.csv"
    stock_rows = tmp_path / "stock_rows.csv"
    features_path = tmp_path / "features.csv"
    _write_news(news)
    with stock_rows.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rebalance_date", "symbol"])
        writer.writeheader()
        writer.writerow({"rebalance_date": "2024-01-02", "symbol": "AAPL"})

    paths = write_stock_alpha_news_features_from_config(
        {
            "ml": {
                "stock_alpha_report_root": str(tmp_path / "reports"),
                "stock_alpha_run_size": "dev",
                "stock_alpha_news_contract_path": str(news),
                "stock_alpha_news_stock_rows_path": str(stock_rows),
                "stock_alpha_news_features_path": str(features_path),
                "stock_alpha_news_min_symbol_coverage": 1.0,
                "stock_alpha_news_min_date_coverage": 1.0,
                **_guardrails(),
            }
        }
    )

    assert paths.features_csv_path.exists()


def test_news_feature_writer_fails_clearly_when_news_contract_missing(tmp_path):
    stock_rows = tmp_path / "stock_rows.csv"
    with stock_rows.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rebalance_date", "symbol"])
        writer.writeheader()
        writer.writerow({"rebalance_date": "2024-01-02", "symbol": "AAPL"})

    try:
        write_stock_alpha_news_features_from_config(
            {
                "ml": {
                    "stock_alpha_report_root": str(tmp_path / "reports"),
                    "stock_alpha_run_size": "dev",
                    "stock_alpha_news_contract_path": str(tmp_path / "missing_news.csv"),
                    "stock_alpha_news_stock_rows_path": str(stock_rows),
                    "stock_alpha_news_min_symbol_coverage": 0.0,
                    "stock_alpha_news_min_date_coverage": 0.0,
                    **_guardrails(),
                }
            }
        )
    except ValueError as exc:
        assert "news contract file not found" in str(exc)
    else:
        raise AssertionError("missing news contract should fail clearly")


def test_news_feature_writer_fails_clearly_when_stock_rows_missing(tmp_path):
    news = tmp_path / "news.csv"
    _write_news(news)

    try:
        write_stock_alpha_news_features_from_config(
            {
                "ml": {
                    "stock_alpha_report_root": str(tmp_path / "reports"),
                    "stock_alpha_run_size": "dev",
                    "stock_alpha_news_contract_path": str(news),
                    "stock_alpha_news_stock_rows_path": str(tmp_path / "missing_stock_rows.csv"),
                    **_guardrails(),
                }
            }
        )
    except ValueError as exc:
        assert "stock rows file not found" in str(exc)
    else:
        raise AssertionError("missing stock rows should fail clearly")


def test_news_feature_writer_fails_clearly_when_stock_row_fields_missing(tmp_path):
    news = tmp_path / "news.csv"
    stock_rows = tmp_path / "stock_rows.csv"
    _write_news(news)
    stock_rows.write_text("symbol\nAAPL\n", encoding="utf-8")

    try:
        write_stock_alpha_news_features_from_config(
            {
                "ml": {
                    "stock_alpha_report_root": str(tmp_path / "reports"),
                    "stock_alpha_run_size": "dev",
                    "stock_alpha_news_contract_path": str(news),
                    "stock_alpha_news_stock_rows_path": str(stock_rows),
                    **_guardrails(),
                }
            }
        )
    except ValueError as exc:
        assert "missing required fields: rebalance_date" in str(exc)
    else:
        raise AssertionError("missing stock row fields should fail clearly")


def test_news_feature_writer_fails_clearly_when_guardrails_invalid(tmp_path):
    news = tmp_path / "news.csv"
    _write_news(news)

    try:
        write_stock_alpha_news_features(
            {
                "ml": {
                    "stock_alpha_report_root": str(tmp_path / "reports"),
                    "stock_alpha_run_size": "dev",
                    "stock_alpha_news_contract_path": str(news),
                    "stock_alpha_news_min_symbol_coverage": 1.0,
                    "stock_alpha_news_min_date_coverage": 1.0,
                }
            },
            _stock_rows(),
        )
    except ValueError as exc:
        assert "requires research-only guardrails" in str(exc)
    else:
        raise AssertionError("invalid guardrails should fail clearly")


def test_news_feature_template_configs_load():
    benchmark = load_config("config/config.stock_alpha_news_features_benchmark_fast_template.yaml", overlay_project_config=True)
    full = load_config("config/config.stock_alpha_news_features_full_template.yaml", overlay_project_config=True)

    assert benchmark["ml"]["stock_alpha_run_size"] == "benchmark"
    assert full["ml"]["stock_alpha_run_size"] == "full"
    assert benchmark["ml"]["stock_alpha_news_enable_transformer"] is False
    assert full["ml"]["stock_alpha_news_enable_transformer"] is False
    assert benchmark["ml"]["research_only"] is True
    assert full["ml"]["trading_impact"] == "none"


def test_tiny_news_fixture_contract_is_real_shaped_and_unique():
    news_rows = CsvRowRepository().read(Path("tests/fixtures/stock_alpha_news/news_contract_tiny.csv"))
    stock_rows = CsvRowRepository().read(Path("tests/fixtures/stock_alpha_news/stock_rows_tiny.csv"))

    assert set(REQUIRED_NEWS_CONTRACT_COLUMNS) <= set(news_rows[0])
    assert {"rebalance_date", "symbol"} <= set(stock_rows[0])
    article_ids = [row["article_id"] for row in news_rows]
    assert len(article_ids) == len(set(article_ids))
    assert {row["symbol"] for row in stock_rows} == {"AAPL", "MSFT"}


def test_tiny_news_fixture_generates_features_and_readiness(tmp_path):
    config = load_config("config/config.stock_alpha_news_features_tiny_fixture.yaml", overlay_project_config=True)
    features_path = tmp_path / "stock_alpha_news_features.csv"
    config["ml"]["stock_alpha_report_root"] = str(tmp_path / "reports")
    config["ml"]["stock_alpha_news_features_path"] = str(features_path)

    paths = write_stock_alpha_news_features_from_config(config)
    feature_rows = CsvRowRepository().read(paths.features_csv_path)
    audit = json.loads(paths.audit_json_path.read_text(encoding="utf-8"))

    assert paths.features_csv_path.exists()
    assert paths.audit_json_path.exists()
    assert paths.audit_markdown_path.exists()
    assert set(REQUIRED_NEWS_FEATURE_COLUMNS) <= set(feature_rows[0])
    by_key = {(row["rebalance_date"], row["symbol"]): row for row in feature_rows}
    assert int(by_key[("2024-01-04", "AAPL")]["negative_news_count_7d"]) == 1
    assert int(by_key[("2024-01-04", "AAPL")]["earnings_news_count_14d"]) == 1
    assert int(by_key[("2024-01-04", "AAPL")]["analyst_news_count_14d"]) == 1
    assert int(by_key[("2024-01-06", "AAPL")]["guidance_news_count_30d"]) == 1
    assert int(by_key[("2024-01-06", "MSFT")]["litigation_news_count_30d"]) == 1
    assert int(by_key[("2024-01-06", "MSFT")]["mna_news_count_30d"]) == 1
    assert by_key[("2024-01-02", "AAPL")]["news_volume_zscore"] == ""
    assert by_key[("2024-01-02", "MSFT")]["avg_sentiment_1d"] == ""
    assert int(by_key[("2024-01-04", "MSFT")]["guidance_news_count_30d"]) == 0
    assert audit["future_article_excluded_count"] > 0
    assert audit["pit_violation_count"] == 0
    assert audit["transformer_available"] is False

    stock_rows = CsvRowRepository().read(Path("tests/fixtures/stock_alpha_news/stock_rows_tiny.csv"))
    disabled = check_news_transformer_readiness(
        {**config, "ml": {**config["ml"], "stock_alpha_news_enable_transformer": False}},
        stock_rows,
    )
    enabled = check_news_transformer_readiness(
        {**config, "ml": {**config["ml"], "stock_alpha_news_enable_transformer": True}},
        stock_rows,
    )
    assert disabled.transformer_available is False
    assert disabled.unavailable_reason == "stock_alpha_news_enable_transformer_false"
    assert enabled.transformer_available is True
    assert set(enabled.required_columns_found) == set(REQUIRED_NEWS_FEATURE_COLUMNS)


def test_news_feature_writer_excludes_future_archive_rows(tmp_path):
    news = tmp_path / "news.csv"
    _write_news(news, published="2024-01-03T00:00:00Z")

    paths = write_stock_alpha_news_features(
        {
            "ml": {
                "stock_alpha_report_root": str(tmp_path / "reports"),
                "stock_alpha_run_size": "dev",
                "stock_alpha_news_contract_path": str(news),
                "stock_alpha_news_min_symbol_coverage": 0.0,
                "stock_alpha_news_min_date_coverage": 0.0,
                **_guardrails(),
            }
        },
        _stock_rows(),
    )

    audit = json.loads(paths.audit_json_path.read_text(encoding="utf-8"))
    assert audit["future_article_excluded_count"] == 1
    assert audit["pit_violation_count"] == 0


def test_stock_alpha_news_pipeline_preflight_tiny_fixture_stops_at_disabled_transformer(tmp_path):
    config = load_config(
        "config/config.stock_alpha_news_pipeline_preflight_tiny_fixture.yaml",
        overlay_project_config=True,
    )
    _redirect_pipeline_outputs(config, tmp_path)

    paths = write_stock_alpha_news_pipeline_preflight(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    stages = payload["stages"]

    assert payload["pipeline_safe_for_news_transformer_training"] is False
    assert payload["pipeline_completed"] is True
    assert payload["stopped_stage"] is None
    assert payload["model_training_invoked"] is False
    assert payload["diagnostics_invoked"] is False
    assert stages["provider_audit"]["attempted"] is True
    assert stages["provider_audit"]["completed"] is True
    assert stages["provider_audit"]["safe"] is True
    assert stages["contract_ingest"]["completed"] is True
    assert stages["contract_ingest"]["safe"] is True
    assert stages["coverage_audit"]["completed"] is True
    assert stages["coverage_audit"]["safe"] is True
    assert stages["feature_generation"]["completed"] is True
    assert stages["feature_generation"]["safe"] is True
    assert stages["readiness_preflight"]["completed"] is True
    assert stages["readiness_preflight"]["safe"] is False
    assert stages["readiness_preflight"]["blocking_issues"] == [
        "stock_alpha_news_enable_transformer_false"
    ]


def test_provider_sample_check_canonical_tiny_is_compatible_and_read_only(tmp_path):
    config = load_config(
        "config/config.stock_alpha_news_provider_sample_check_tiny_fixture.yaml",
        overlay_project_config=True,
    )
    config["ml"]["stock_alpha_news_provider_sample_check_output_dir"] = str(tmp_path / "check")
    contract = tmp_path / "contract.csv"
    features = tmp_path / "features.csv"
    config["ml"]["stock_alpha_news_contract_path"] = str(contract)
    config["ml"]["stock_alpha_news_features_path"] = str(features)

    paths = write_stock_alpha_news_provider_sample_check(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))

    assert payload["compatible_with_contract_ingest"] is True
    assert payload["next_action"] == "run_provider_audit"
    assert set(payload["canonical_columns_present_directly"]) == set(REQUIRED_NEWS_CONTRACT_COLUMNS)
    assert payload["timestamp_parseability"]["published_at_utc"]["invalid_count"] == 0
    assert payload["ingested_before_published_count"] == 1
    assert payload["article_id_uniqueness_preview"]["duplicate_count"] == 1
    assert payload["canonical_contract_written"] is False
    assert payload["features_generated"] is False
    assert payload["model_training_invoked"] is False
    assert payload["diagnostics_invoked"] is False
    assert not contract.exists()
    assert not features.exists()


def test_provider_sample_check_alias_mapping_is_compatible(tmp_path):
    config = load_config(
        "config/config.stock_alpha_news_contract_ingest_alias_tiny_fixture.yaml",
        overlay_project_config=True,
    )
    config["ml"]["stock_alpha_news_provider_sample_check_output_dir"] = str(tmp_path)

    paths = write_stock_alpha_news_provider_sample_check(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))

    assert payload["compatible_with_contract_ingest"] is True
    assert payload["provider_mapping_used"] is True
    assert payload["missing_mapped_provider_columns"] == []
    assert payload["symbol_normalization_preview"][0]["normalized"] == "AAPL"


def test_provider_sample_check_missing_file_blocks_cleanly(tmp_path, capsys):
    config = load_config(
        "config/config.stock_alpha_news_provider_sample_check_real_template.yaml",
        overlay_project_config=True,
    )
    config["ml"]["stock_alpha_news_raw_path"] = str(tmp_path / "missing.csv")
    config["ml"]["stock_alpha_news_provider_sample_check_output_dir"] = str(tmp_path / "check")

    run_ml_stock_alpha_news_provider_sample_check(config)
    output = capsys.readouterr().out
    payload = json.loads(
        (tmp_path / "check" / "stock_alpha_news_provider_sample_check.json").read_text(encoding="utf-8")
    )

    assert payload["compatible_with_contract_ingest"] is False
    assert payload["next_action"] == "provide_raw_news_file"
    assert "next_action=provide_raw_news_file" in output


def test_provider_sample_check_missing_mapped_column_is_incompatible(tmp_path):
    config = load_config(
        "config/config.stock_alpha_news_contract_ingest_alias_tiny_fixture.yaml",
        overlay_project_config=True,
    )
    config["ml"]["stock_alpha_news_provider_column_map"]["article_id"] = "missing_id"
    config["ml"]["stock_alpha_news_provider_sample_check_output_dir"] = str(tmp_path)

    paths = write_stock_alpha_news_provider_sample_check(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))

    assert payload["compatible_with_contract_ingest"] is False
    assert payload["next_action"] == "fix_missing_provider_columns"
    assert payload["missing_mapped_provider_columns"] == ["missing_id"]


def test_provider_sample_check_bad_timestamps_are_incompatible(tmp_path):
    raw = tmp_path / "raw.csv"
    _write_raw_news_csv(raw, published="not-a-timestamp")
    config = {
        "ml": {
            "stock_alpha_news_raw_path": str(raw),
            "stock_alpha_news_provider_sample_check_output_dir": str(tmp_path / "check"),
        }
    }

    paths = write_stock_alpha_news_provider_sample_check(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))

    assert payload["compatible_with_contract_ingest"] is False
    assert payload["next_action"] == "fix_timestamp_columns"
    assert payload["timestamp_parseability"]["published_at_utc"]["invalid_count"] == 1


def test_news_feature_diagnostics_missing_inputs_block_cleanly(tmp_path, capsys):
    stock = _write_stock_rows_csv(tmp_path)
    run_ml_stock_alpha_news_feature_diagnostics(
        _feature_diagnostics_config(tmp_path / "missing.csv", stock, tmp_path / "report")
    )
    output = capsys.readouterr().out
    payload = json.loads((tmp_path / "report" / "stock_alpha_news_feature_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["next_action"] == "provide_news_features"
    assert "blocking_issue=news_features_file_not_found" in output

    features = tmp_path / "features.csv"
    _write_features(features)
    paths = write_stock_alpha_news_feature_diagnostics(
        _feature_diagnostics_config(features, tmp_path / "missing-stock.csv", tmp_path / "report-2")
    )
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["next_action"] == "provide_stock_rows"


def test_news_feature_diagnostics_tiny_report_is_read_only(tmp_path):
    config = load_config(
        "config/config.stock_alpha_news_feature_diagnostics_tiny_fixture.yaml",
        overlay_project_config=True,
    )
    config["ml"]["stock_alpha_news_feature_diagnostics_report_dir"] = str(tmp_path)
    paths = write_stock_alpha_news_feature_diagnostics(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["input_status"]["required_columns_present"] is True
    assert payload["pit_safety_diagnostics"]["future_article_excluded_count"] > 0
    assert payload["pit_safety_diagnostics"]["pit_violation_count"] == 0
    assert payload["exploratory_correlations"]["status"] == "computed"
    for key in ("features_generated", "files_ingested", "readiness_invoked", "diagnostics_invoked", "model_training_invoked", "news_transformer_enabled"):
        assert payload[key] is False


def test_news_feature_diagnostics_enforces_columns_and_audit_safety(tmp_path):
    stock = _write_stock_rows_csv(tmp_path)
    incomplete = tmp_path / "incomplete.csv"
    incomplete.write_text("rebalance_date,symbol\n2024-01-02,AAPL\n", encoding="utf-8")
    paths = write_stock_alpha_news_feature_diagnostics(_feature_diagnostics_config(incomplete, stock, tmp_path / "missing-columns"))
    assert json.loads(paths.json_path.read_text(encoding="utf-8"))["next_action"] == "fix_missing_required_feature_columns"

    for name, audit, action in (
        ("pit", {"pit_violation_count": 1, "future_article_excluded_count": 3}, "investigate_pit_violation"),
        ("synthetic", {"pit_violation_count": 0, "synthetic_news_features_created": True}, "investigate_synthetic_features"),
    ):
        directory = tmp_path / name
        directory.mkdir()
        features = directory / "features.csv"
        _write_features(features)
        audit_dir = directory / "news_features"
        audit_dir.mkdir()
        (audit_dir / "stock_alpha_news_features_audit.json").write_text(json.dumps(audit), encoding="utf-8")
        paths = write_stock_alpha_news_feature_diagnostics(_feature_diagnostics_config(features, stock, directory / "report"))
        payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
        assert payload["next_action"] == action
        assert payload["model_readiness_diagnostics"]["suitable_for_disabled_diagnostic_merge_check"] is False


def test_news_feature_diagnostics_no_news_missing_sentiment_and_labels_skip(tmp_path):
    features = tmp_path / "features.csv"
    _write_features(features)
    rows = list(csv.DictReader(features.open(encoding="utf-8")))
    rows[0]["news_has_coverage_30d"] = "false"
    for column in ("avg_sentiment_1d", "avg_sentiment_3d", "sentiment_change_3d"):
        rows[0][column] = ""
    with features.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stock = tmp_path / "stock.csv"
    stock.write_text("rebalance_date,symbol\n2024-01-02,AAPL\n", encoding="utf-8")
    paths = write_stock_alpha_news_feature_diagnostics(_feature_diagnostics_config(features, stock, tmp_path / "report"))
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["missingness_diagnostics"]["no_news_fake_neutral_sentiment_row_count"] == 0
    assert payload["exploratory_correlations"]["status"] == "skipped_labels_absent"


def test_news_source_diagnostics_missing_inputs_block_cleanly(tmp_path, capsys):
    stock = _write_stock_rows_csv(tmp_path)
    config = _source_diagnostics_config(tmp_path / "missing.csv", stock, tmp_path / "report")
    run_ml_stock_alpha_news_source_diagnostics(config)
    output = capsys.readouterr().out
    payload = json.loads((tmp_path / "report" / "stock_alpha_news_source_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["next_action"] == "provide_news_contract"
    assert "blocking_issue=news_contract_file_not_found" in output

    paths = write_stock_alpha_news_source_diagnostics(
        _source_diagnostics_config(Path("tests/fixtures/stock_alpha_news/news_contract_tiny.csv"), tmp_path / "missing-stock.csv", tmp_path / "report-2")
    )
    assert json.loads(paths.json_path.read_text(encoding="utf-8"))["next_action"] == "provide_stock_rows"


def test_news_source_diagnostics_tiny_is_read_only_and_exploratory(tmp_path):
    config = load_config("config/config.stock_alpha_news_source_diagnostics_tiny_fixture.yaml", overlay_project_config=True)
    config["ml"]["stock_alpha_news_source_diagnostics_report_dir"] = str(tmp_path)
    paths = write_stock_alpha_news_source_diagnostics(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["input_status"]["source_identifier_column"] == "source"
    assert payload["pit_safety"]["future_article_excluded_count"] > 0
    assert payload["pit_safety"]["included_future_article_count"] == 0
    assert payload["exploratory_label_relationship"]["status"] == "computed"
    for key in ("features_generated", "files_ingested", "readiness_invoked", "model_training_invoked", "diagnostics_invoked", "news_transformer_enabled"):
        assert payload[key] is False


def test_news_source_diagnostics_multiple_sources_agreement_duplication_and_label_skip(tmp_path):
    contract = tmp_path / "contract.csv"
    rows = [
        _raw_news_row(article_id="a1", source="one", headline="same", sentiment="0.8"),
        _raw_news_row(article_id="a2", source="one", headline="same", sentiment="0.7"),
        _raw_news_row(article_id="b1", source="two", headline="different", sentiment="-0.8"),
    ]
    _write_raw_news_csv(contract, rows=rows)
    stock = tmp_path / "stock.csv"
    stock.write_text("rebalance_date,symbol\n2024-01-02,AAPL\n", encoding="utf-8")
    paths = write_stock_alpha_news_source_diagnostics(_source_diagnostics_config(contract, stock, tmp_path / "report"))
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["cross_source_agreement"]["multi_source_window_count"] == 1
    assert payload["cross_source_agreement"]["windows"][0]["sentiment_disagreement_score"] > 1.0
    assert "one" in payload["source_quality"]["sources_with_duplicate_headlines"]
    assert "source_duplication_detected" in payload["warning_issues"]
    assert payload["exploratory_label_relationship"]["status"] == "skipped_labels_absent"


def test_news_source_diagnostics_true_timestamp_leakage_blocks(tmp_path):
    contract = tmp_path / "contract.csv"
    _write_raw_news_csv(contract, published="2024-01-02T00:00:00Z", ingested="2024-01-01T00:00:00Z")
    paths = write_stock_alpha_news_source_diagnostics(
        _source_diagnostics_config(contract, _write_stock_rows_csv(tmp_path), tmp_path / "report")
    )
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["next_action"] == "fix_timestamp_leakage"
    assert "timestamp_leakage_detected" in payload["blocking_issues"]


def test_gdelt_fake_response_normalizes_without_invented_scores():
    requested_urls = []

    def fake_get(url, timeout):
        requested_urls.append(url)
        return {"articles": [{
            "url": "https://example.test/a", "seendate": "20240101T120000Z",
            "domain": "example.test", "title": "AAPL headline",
            "snippet": "Provider summary.", "language": "English",
        }]}

    source = GdeltNewsSource(fake_get)
    rows = source.collect(symbols=["AAPL"], start_date="2024-01-01", end_date="2024-01-02", limit=5, timeout=2)
    query = parse_qs(urlparse(requested_urls[0]).query)
    assert query["query"] == ['Apple OR "Apple Inc"']
    assert source.last_batch_diagnostic["query_terms"] == {"AAPL": ["Apple", "Apple Inc"]}
    assert rows[0]["provider"] == "gdelt"
    assert rows[0]["source"] == "example.test"
    assert rows[0]["published_at_utc"] == "2024-01-01T12:00:00Z"
    assert rows[0]["body_or_summary"] == "Provider summary."
    assert rows[0]["event_type"] == "news"
    assert rows[0]["sentiment_score"] == ""
    assert rows[0]["relevance_score"] == ""
    assert rows[0]["novelty_score"] == ""


def test_gdelt_ambiguous_short_symbol_is_skipped_without_company_alias():
    def fake_get(url, timeout):
        raise AssertionError("ambiguous ticker should not be queried")

    source = GdeltNewsSource(fake_get)
    rows = source.collect(
        symbols=["IT"],
        start_date="2024-01-01",
        end_date="2024-01-02",
        limit=5,
        timeout=2,
    )

    assert rows == []
    assert source.api_key_required is False
    assert source.last_batch_diagnostic["query_terms"] == {}
    assert source.last_batch_diagnostic["skipped_symbols"] == {
        "IT": "skipped_ambiguous_symbol"
    }


def test_alpha_vantage_news_source_honors_configured_date_window():
    requested_urls = []

    def fake_get(url, timeout):
        requested_urls.append(url)
        return {"feed": []}

    AlphaVantageNewsSource(fake_get).collect(
        symbols=["AAPL"],
        start_date="2026-01-01",
        end_date="2026-04-20",
        limit=5,
        timeout=2,
        api_key="test-key",
    )

    query = parse_qs(urlparse(requested_urls[0]).query)
    assert query["function"] == ["NEWS_SENTIMENT"]
    assert query["tickers"] == ["AAPL"]
    assert query["time_from"] == ["20260101T0000"]
    assert query["time_to"] == ["20260420T2359"]
    assert query["limit"] == ["5"]


def test_massive_stock_news_source_maps_response_to_canonical_rows():
    requested_urls = []

    def fake_get(url, timeout):
        requested_urls.append(url)
        return {
            "status": "OK",
            "results": [
                {
                    "id": "news-1",
                    "publisher": {"name": "Example Wire"},
                    "title": "Apple expands test program",
                    "description": "A concise provider summary.",
                    "article_url": "https://example.test/news-1",
                    "published_utc": "2026-04-20T14:30:00Z",
                    "tickers": ["AAPL", "MSFT"],
                    "insights": [{"ticker": "AAPL", "sentiment": "positive"}],
                }
            ],
        }

    rows = MassiveStockNewsSource(fake_get).collect(
        symbols=["AAPL"],
        start_date="2023-12-01",
        end_date="2026-04-20",
        limit=5,
        timeout=2,
        api_key="test-key",
    )

    query = parse_qs(urlparse(requested_urls[0]).query)
    assert urlparse(requested_urls[0]).netloc == "api.massive.com"
    assert query["ticker"] == ["AAPL"]
    assert query["published_utc.gte"] == ["2023-12-01"]
    assert query["published_utc.lte"] == ["2026-04-20"]
    assert query["sort"] == ["published_utc"]
    assert query["order"] == ["asc"]
    assert query["limit"] == ["5"]
    assert query["apiKey"] == ["test-key"]
    assert rows[0]["article_id"] == "massive_stock_news:news-1:AAPL"
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["published_at_utc"] == "2026-04-20T14:30:00Z"
    assert rows[0]["source"] == "Example Wire"
    assert rows[0]["headline"] == "Apple expands test program"
    assert rows[0]["body_or_summary"] == "A concise provider summary."
    assert rows[0]["sentiment_score"] == ""
    assert rows[0]["relevance_score"] == ""
    assert rows[0]["novelty_score"] == ""
    assert rows[0]["event_type"] == "news"
    assert rows[0]["provider"] == "massive_stock_news"
    assert rows[0]["provider_article_id"] == "news-1:AAPL"
    assert rows[0]["provider_url"] == "https://example.test/news-1"


def test_massive_stock_news_zero_results_are_empty_not_failure():
    rows = MassiveStockNewsSource(lambda url, timeout: {"status": "OK", "results": []}).collect(
        symbols=["AAPL"],
        start_date="2023-12-01",
        end_date="2026-04-20",
        limit=5,
        timeout=2,
        api_key="test-key",
    )
    assert rows == []


def test_company_press_release_rss_source_maps_sample_to_canonical_without_scores():
    requested_urls = []
    rss = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Apple Newsroom</title>
        <language>en-US</language>
        <item>
          <title>Apple announces test expansion</title>
          <link>https://example.test/apple-press-release</link>
          <pubDate>Mon, 20 Apr 2026 14:30:00 GMT</pubDate>
          <description>Official RSS summary only.</description>
        </item>
      </channel>
    </rss>
    """

    def fake_get(url, timeout):
        requested_urls.append((url, timeout))
        return rss

    source = CompanyPressReleaseRssSource(fake_get).with_provider_config({
        "max_rows_per_feed": 5,
        "feeds": {
            "AAPL": [{
                "name": "Apple Newsroom",
                "url": "https://example.test/apple/rss.xml",
                "enabled": True,
                "event_type": "press_release",
            }]
        },
    })

    rows = source.collect(
        symbols=["AAPL"],
        start_date="2026-04-19",
        end_date="2026-04-21",
        limit=5,
        timeout=2,
    )

    assert requested_urls == [("https://example.test/apple/rss.xml", 2)]
    assert source.api_key_required is False
    assert PROVIDER_METADATA["company_press_release_rss"]["api_key_required"] is False
    assert rows[0]["article_id"].startswith("company_press_release_rss:rss:AAPL:")
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["published_at_utc"] == "2026-04-20T14:30:00Z"
    assert rows[0]["source"] == "Apple Newsroom"
    assert rows[0]["headline"] == "Apple announces test expansion"
    assert rows[0]["body_or_summary"] == "Official RSS summary only."
    assert rows[0]["sentiment_score"] == ""
    assert rows[0]["relevance_score"] == ""
    assert rows[0]["novelty_score"] == ""
    assert rows[0]["event_type"] == "press_release"
    assert rows[0]["language"] == "en-US"
    assert rows[0]["provider"] == "company_press_release_rss"
    assert rows[0]["provider_article_id"].startswith("rss:AAPL:")
    assert rows[0]["provider_url"] == "https://example.test/apple-press-release"
    assert source.last_batch_diagnostic["feed_diagnostics"][0] == {
        "provider": "company_press_release_rss",
        "symbol": "AAPL",
        "feed_name": "Apple Newsroom",
        "feed_url": "https://example.test/apple/rss.xml",
        "response_row_count": 1,
        "normalized_row_count": 1,
        "zero_row_reason": "",
        "error_type": "",
        "error_message": "",
        "rate_limited": False,
    }


def test_company_press_release_rss_empty_feed_reports_zero_reason_not_failure():
    source = CompanyPressReleaseRssSource(lambda url, timeout: "<rss><channel /></rss>").with_provider_config({
        "feeds": {
            "AAPL": [{
                "name": "Apple Newsroom",
                "url": "https://example.test/apple/rss.xml",
                "enabled": True,
            }]
        },
    })

    rows = source.collect(
        symbols=["AAPL"],
        start_date="2026-04-19",
        end_date="2026-04-21",
        limit=5,
        timeout=2,
    )

    diagnostic = source.last_batch_diagnostic["feed_diagnostics"][0]
    assert rows == []
    assert diagnostic["zero_row_reason"] == "empty_feed"
    assert diagnostic["error_type"] == ""
    assert diagnostic["error_message"] == ""
    assert diagnostic["rate_limited"] is False


def test_company_press_release_rss_invalid_feed_url_is_diagnostic_not_collection_failure(tmp_path):
    def fake_get(url, timeout):
        raise AssertionError("invalid URL should fail before network fetch")

    config = _collection_config(tmp_path, dry_run=True)
    settings = config["ml"]["stock_alpha_news_collect"]
    settings["providers"] = {
        "company_press_release_rss": {
            "enabled": True,
            "feeds": {
                "AAPL": [{
                    "name": "Invalid feed",
                    "url": "not-a-url",
                    "enabled": True,
                }]
            },
        }
    }
    paths = write_stock_alpha_news_free_source_collect(
        config,
        sources={"company_press_release_rss": CompanyPressReleaseRssSource(fake_get)},
    )
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    feed_diagnostic = payload["provider_batch_diagnostics"][0]["feed_diagnostics"][0]

    assert payload["providers_failed"] == {}
    assert payload["provider_row_counts"] == {"company_press_release_rss": 0}
    assert payload["providers_returned_zero_rows"] == ["company_press_release_rss"]
    assert payload["provider_zero_row_reasons"] == {
        "company_press_release_rss": "all_batches_returned_zero_rows"
    }
    assert feed_diagnostic["provider"] == "company_press_release_rss"
    assert feed_diagnostic["symbol"] == "AAPL"
    assert feed_diagnostic["feed_name"] == "Invalid feed"
    assert feed_diagnostic["feed_url"] == "not-a-url"
    assert feed_diagnostic["zero_row_reason"] == "provider_error"
    assert feed_diagnostic["error_type"] == "ValueError"
    assert feed_diagnostic["error_message"] == "invalid RSS feed URL"
    assert feed_diagnostic["rate_limited"] is False


def test_free_source_collection_reports_rss_registry_metrics(tmp_path):
    rss = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <item>
          <title>Apple announces test expansion</title>
          <link>https://example.test/apple-press-release</link>
          <pubDate>Mon, 20 Apr 2026 14:30:00 GMT</pubDate>
          <description>Official RSS summary only.</description>
        </item>
      </channel>
    </rss>
    """

    def fake_get(url, timeout):
        return rss

    config = _collection_config(tmp_path, dry_run=True)
    settings = config["ml"]["stock_alpha_news_collect"]
    settings["symbols"] = ["AAPL", "TSLA", "BAD"]
    settings["start_date"] = "2026-04-19"
    settings["end_date"] = "2026-04-21"
    settings["providers"] = {
        "company_press_release_rss": {
            "enabled": True,
            "feeds": {
                "AAPL": [{
                    "name": "Apple Newsroom",
                    "url": "https://example.test/apple/rss.xml",
                    "enabled": True,
                    "official": True,
                    "verified_source_url": "https://example.test/apple/",
                }],
                "TSLA": [{
                    "name": "Tesla RSS unavailable",
                    "enabled": False,
                }],
                "BAD": [{
                    "name": "Broken official RSS",
                    "url": "not-a-url",
                    "enabled": True,
                    "official": True,
                    "verified_source_url": "https://example.test/bad/",
                }],
            },
        }
    }
    paths = write_stock_alpha_news_free_source_collect(
        config,
        sources={"company_press_release_rss": CompanyPressReleaseRssSource(fake_get)},
    )
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    markdown = paths.markdown_path.read_text(encoding="utf-8")

    assert payload["requested_symbol_count"] == 3
    assert payload["verified_feed_symbol_count"] == 2
    assert payload["enabled_feed_symbol_count"] == 2
    assert payload["disabled_symbol_count"] == 1
    assert payload["symbols_without_verified_feed"] == ["TSLA"]
    assert payload["symbols_with_feed_errors"] == ["BAD"]
    assert payload["rows_by_symbol"] == {"AAPL": 1}
    assert payload["provider_symbol_coverage"] == {"company_press_release_rss": 1 / 3}
    assert "- Verified feed symbols: 2" in markdown
    assert "- Symbols without verified feed: ['TSLA']" in markdown
    assert "- Symbols with feed errors: ['BAD']" in markdown


def test_free_source_collection_missing_keys_skip_and_dry_run_is_safe(tmp_path, monkeypatch):
    for name in ("ALPHA_VANTAGE_API_KEY", "FINNHUB_API_KEY", "FMP_API_KEY", "NEWSAPI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    config = load_config("config/config.stock_alpha_news_collect_free_sources_dry_run.yaml", overlay_project_config=True)
    config["ml"]["stock_alpha_news_collect_report_dir"] = str(tmp_path / "report")
    config["ml"]["stock_alpha_news_collect_output_path"] = str(tmp_path / "raw.csv")
    paths = write_stock_alpha_news_free_source_collect(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert set(payload["providers_skipped_missing_key"]) == {"alpha_vantage", "finnhub", "fmp", "newsapi"}
    assert payload["output_written"] is False
    assert not paths.output_path.exists()
    assert "API_KEY" not in paths.json_path.read_text(encoding="utf-8")
    for key in ("files_ingested", "features_generated", "readiness_invoked", "diagnostics_invoked", "model_training_invoked", "news_transformer_enabled"):
        assert payload[key] is False


def test_free_source_collection_writes_deduplicated_canonical_csv(tmp_path):
    class FakeSource:
        api_key_required = False
        def collect(self, **kwargs):
            row = _collected_news_row("same", "gdelt")
            return [row, dict(row)]
    config = _collection_config(tmp_path, dry_run=False)
    paths = write_stock_alpha_news_free_source_collect(config, sources={"gdelt": FakeSource()})
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(paths.output_path.open(encoding="utf-8")))
    assert payload["total_rows_collected"] == 2
    assert payload["deduplicated_row_count"] == 1
    assert payload["output_written"] is True
    assert len(rows) == 1
    assert set(REQUIRED_NEWS_CONTRACT_COLUMNS) <= set(rows[0])
    assert rows[0]["provider"] == "gdelt"
    assert rows[0]["source"] == "gdelt-source"


def test_free_source_collection_batches_symbols_and_reports_provider_coverage(tmp_path):
    calls = []

    class Batched:
        api_key_required = False

        def collect(self, **kwargs):
            calls.append(kwargs)
            return [
                _collected_news_row(f"{symbol}-{len(calls)}", "gdelt")
                | {"symbol": symbol}
                for symbol in kwargs["symbols"]
            ]

    config = _collection_config(tmp_path, dry_run=True)
    settings = config["ml"]["stock_alpha_news_collect"]
    settings["symbols"] = ["AAPL", "MSFT", "NVDA", "AMZN", "META"]
    settings["symbols_per_batch"] = 2
    settings["provider_request_limit"] = 5
    settings["max_rows_per_provider"] = 10
    settings["rate_limit_sleep_seconds"] = 0

    paths = write_stock_alpha_news_free_source_collect(config, sources={"gdelt": Batched()})
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))

    assert [call["symbols"] for call in calls] == [
        ["AAPL", "MSFT"],
        ["NVDA", "AMZN"],
        ["META"],
    ]
    assert payload["requested_symbol_count"] == 5
    assert payload["symbols_per_batch"] == 2
    assert payload["provider_request_limit"] == 5
    assert payload["max_rows_per_provider"] == 10
    assert payload["provider_batch_counts"] == {"gdelt": 3}
    assert payload["provider_batch_diagnostics"] == [
        {
            "provider": "gdelt",
            "batch_index": 1,
            "symbol_count": 2,
            "symbols": ["AAPL", "MSFT"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "requested_limit": 5,
            "response_row_count": 2,
            "normalized_row_count": 2,
            "zero_row_reason": "",
            "rate_limited": False,
        },
        {
            "provider": "gdelt",
            "batch_index": 2,
            "symbol_count": 2,
            "symbols": ["NVDA", "AMZN"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "requested_limit": 5,
            "response_row_count": 2,
            "normalized_row_count": 2,
            "zero_row_reason": "",
            "rate_limited": False,
        },
        {
            "provider": "gdelt",
            "batch_index": 3,
            "symbol_count": 1,
            "symbols": ["META"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "requested_limit": 5,
            "response_row_count": 1,
            "normalized_row_count": 1,
            "zero_row_reason": "",
            "rate_limited": False,
        },
    ]
    assert payload["provider_zero_row_reasons"] == {}
    assert payload["rows_by_provider"] == {"gdelt": 5}
    assert payload["provider_symbol_counts"] == {"gdelt": 5}
    assert payload["provider_symbol_coverage"] == {"gdelt": 1.0}
    assert payload["rows_by_symbol"] == {
        "AAPL": 1,
        "AMZN": 1,
        "META": 1,
        "MSFT": 1,
        "NVDA": 1,
    }
    assert payload["published_at_utc_range_by_provider"]["gdelt"] == {
        "min_published_at_utc": "2024-01-01T10:00:00Z",
        "max_published_at_utc": "2024-01-01T10:00:00Z",
    }


def test_gdelt_collection_reports_query_terms_and_ambiguous_skips(tmp_path):
    def fake_get(url, timeout):
        return {"articles": [{
            "url": "https://example.test/aapl",
            "seendate": "20240101T120000Z",
            "domain": "example.test",
            "title": "Apple headline",
            "snippet": "Provider summary.",
            "language": "English",
        }]}

    config = _collection_config(tmp_path, dry_run=True)
    settings = config["ml"]["stock_alpha_news_collect"]
    settings["symbols"] = ["AAPL", "IT"]
    settings["symbols_per_batch"] = 2
    settings["provider_request_limit"] = 4
    settings["rate_limit_sleep_seconds"] = 0

    paths = write_stock_alpha_news_free_source_collect(
        config, sources={"gdelt": GdeltNewsSource(fake_get)}
    )
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    diagnostic = payload["provider_batch_diagnostics"][0]

    assert payload["provider_row_counts"] == {"gdelt": 1}
    assert diagnostic["query_terms"] == {"AAPL": ["Apple", "Apple Inc"]}
    assert diagnostic["skipped_symbols"] == {"IT": "skipped_ambiguous_symbol"}
    assert diagnostic["response_row_count"] == 1
    assert diagnostic["normalized_row_count"] == 1


def test_free_source_collection_merges_with_backup_and_stable_deduplication(tmp_path):
    existing = _collected_news_row("existing", "gdelt")
    new = _collected_news_row("new", "gdelt") | {
        "published_at_utc": "2024-01-02T10:00:00Z",
        "ingested_at": "2024-01-02T10:05:00Z",
    }
    config = _collection_config(tmp_path, dry_run=False)
    settings = config["ml"]["stock_alpha_news_collect"]
    settings["merge_existing"] = True
    settings["backup_existing"] = True
    output = Path(config["ml"]["stock_alpha_news_collect_output_path"])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[*REQUIRED_NEWS_CONTRACT_COLUMNS, "provider", "provider_article_id", "provider_url"],
        )
        writer.writeheader()
        writer.writerow(existing)

    class FakeSource:
        api_key_required = False

        def collect(self, **kwargs):
            return [dict(existing) | {"headline": "Refreshed headline"}, new]

    paths = write_stock_alpha_news_free_source_collect(
        config, sources={"gdelt": FakeSource()}
    )
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    output_rows = list(csv.DictReader(output.open(encoding="utf-8")))

    assert payload["existing_input_row_count"] == 1
    assert payload["new_row_count"] == 2
    assert payload["merge_deduplicated_row_count"] == 1
    assert payload["output_row_count"] == 2
    assert payload["output_written"] is True
    assert Path(payload["backup_path"]).exists()
    assert len(output_rows) == 2
    assert output_rows[0]["headline"] == "Refreshed headline"


def test_free_source_collection_protects_output_and_isolates_provider_error(tmp_path):
    class Good:
        api_key_required = False
        def collect(self, **kwargs): return [_collected_news_row("good", "gdelt")]
    class Bad:
        api_key_required = False
        def collect(self, **kwargs): raise RuntimeError("provider unavailable")
    config = _collection_config(tmp_path, dry_run=False)
    output = Path(config["ml"]["stock_alpha_news_collect_output_path"])
    output.write_text("preserve-me", encoding="utf-8")
    config["ml"]["stock_alpha_news_collect"]["providers"]["bad"] = {"enabled": True}
    paths = write_stock_alpha_news_free_source_collect(config, sources={"gdelt": Good(), "bad": Bad()})
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert output.read_text(encoding="utf-8") == "preserve-me"
    assert "bad" in payload["providers_failed"]
    assert payload["output_written"] is False

    config["ml"]["stock_alpha_news_collect"]["allow_overwrite"] = True
    paths = write_stock_alpha_news_free_source_collect(config, sources={"gdelt": Good(), "bad": Bad()})
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["output_written"] is True
    assert output.read_text(encoding="utf-8").startswith("article_id,symbol,")


def test_free_source_collection_report_redacts_api_key(tmp_path, monkeypatch):
    secret = "super-secret-test-key"
    monkeypatch.setenv("TEST_NEWS_KEY", secret)
    class FailingKeyed:
        api_key_required = True
        def collect(self, **kwargs): raise RuntimeError(f"request failed with {kwargs['api_key']}")
    config = _collection_config(tmp_path, dry_run=True)
    config["ml"]["stock_alpha_news_collect"]["providers"] = {"keyed": {"enabled": True, "api_key_env": "TEST_NEWS_KEY"}}
    paths = write_stock_alpha_news_free_source_collect(config, sources={"keyed": FailingKeyed()})
    assert secret not in paths.json_path.read_text(encoding="utf-8")
    assert secret not in paths.markdown_path.read_text(encoding="utf-8")
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    diagnostic = payload["provider_batch_diagnostics"][0]
    assert diagnostic["error_message"] == "request failed with [REDACTED]"
    assert diagnostic["zero_row_reason"] == "provider_error"


def test_news_source_setup_check_gdelt_only_needs_no_key_and_is_read_only(tmp_path):
    config = load_config("config/config.stock_alpha_news_source_setup_check_free_sources.yaml", overlay_project_config=True)
    config["ml"]["stock_alpha_news_source_setup_check_report_dir"] = str(tmp_path / "report")
    config["ml"]["stock_alpha_news_collect_output_path"] = str(tmp_path / "raw.csv")
    paths = write_stock_alpha_news_source_setup_check(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["providers_enabled"] == ["gdelt"]
    assert payload["enabled_providers_missing_key"] == []
    assert payload["next_action"] == "run_free_source_dry_collection"
    assert payload["collection_invoked"] is False
    assert payload["raw_export_written"] is False
    assert not Path(config["ml"]["stock_alpha_news_collect_output_path"]).exists()
    for key in ("files_ingested", "features_generated", "readiness_invoked", "diagnostics_invoked", "model_training_invoked", "news_transformer_enabled"):
        assert payload[key] is False


def test_news_source_setup_check_key_presence_without_value_disclosure(tmp_path, monkeypatch):
    config = load_config("config/config.stock_alpha_news_collect_free_sources_keyed_dry_run.yaml", overlay_project_config=True)
    config["ml"]["stock_alpha_news_source_setup_check_report_dir"] = str(tmp_path)
    for name in ("ALPHA_VANTAGE_API_KEY", "FINNHUB_API_KEY", "FMP_API_KEY", "NEWSAPI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    paths = write_stock_alpha_news_source_setup_check(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["next_action"] == "set_alpha_vantage_api_key"
    assert set(payload["enabled_providers_missing_key"]) == {"alpha_vantage", "finnhub", "fmp", "newsapi"}

    secret = "do-not-print-this-value"
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", secret)
    paths = write_stock_alpha_news_source_setup_check(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["provider_setup"]["alpha_vantage"]["environment_variable_present"] is True
    assert secret not in paths.json_path.read_text(encoding="utf-8")
    assert secret not in paths.markdown_path.read_text(encoding="utf-8")


def test_news_source_setup_check_disabled_keyed_provider_does_not_block_and_literal_is_flagged(tmp_path, monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    config = load_config("config/config.stock_alpha_news_source_setup_check_free_sources.yaml", overlay_project_config=True)
    config["ml"]["stock_alpha_news_source_setup_check_report_dir"] = str(tmp_path)
    paths = write_stock_alpha_news_source_setup_check(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert "finnhub" not in payload["enabled_providers_missing_key"]

    config["ml"]["stock_alpha_news_collect"]["providers"]["finnhub"]["api_key"] = "literal-secret"
    paths = write_stock_alpha_news_source_setup_check(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["next_action"] == "remove_key_values_from_config"
    assert "key_like_literal_present_in_config" in payload["blocking_issues"]
    assert "literal-secret" not in paths.json_path.read_text(encoding="utf-8")


def test_daily_confirmation_reports_negative_news_and_recent_sec_filing(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-alpha-key")

    class Alpha:
        api_key_required = True

        def collect(self, **kwargs):
            return [
                _collected_news_row("alpha-negative", "alpha_vantage")
                | {
                    "symbol": kwargs["symbols"][0],
                    "headline": "AAPL supplier warning",
                    "sentiment_score": "-0.42",
                    "published_at_utc": "2026-04-20T10:00:00Z",
                    "provider_url": "https://example.test/alpha-negative",
                }
            ]

    class Sec:
        api_key_required = False

        def collect(self, **kwargs):
            return [
                _collected_news_row("sec-8k", "sec_edgar")
                | {
                    "symbol": kwargs["symbols"][0],
                    "headline": f"{kwargs['symbols'][0]} SEC 8-K filing accepted 2026-04-20 accession 0001",
                    "published_at_utc": "2026-04-20T12:00:00Z",
                    "provider_url": "https://sec.test/8k",
                }
            ]

    paths = write_stock_alpha_news_daily_confirmation(
        _daily_confirmation_config(tmp_path),
        sources={"alpha_vantage": Alpha(), "sec_edgar": Sec()},
    )
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    report = payload["symbol_reports"][0]

    assert payload["confirmation_only"] is True
    assert payload["trading_impact"] == "none"
    assert payload["orders_generated"] is False
    assert payload["broker_invoked"] is False
    assert payload["model_training_invoked"] is False
    assert payload["news_transformer_enabled"] is False
    assert report["confirmation_status"] == "negative_news_review"
    assert report["negative_news_flag"] is True
    assert report["sec_recent_filing"] is True
    assert report["recent_filing_count"] == 1
    assert report["latest_filing_form"] == "8-K"
    assert report["article_count"] == 1
    assert report["provider_sentiment_summary"]["min"] == -0.42
    assert "BUY" not in paths.markdown_path.read_text(encoding="utf-8")
    assert "SELL" not in paths.markdown_path.read_text(encoding="utf-8")


def test_daily_confirmation_alpha_rate_limit_is_provider_limited_and_redacted(tmp_path, monkeypatch):
    secret = "secret-alpha-confirmation-key"
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", secret)

    class AlphaLimited:
        api_key_required = True

        def collect(self, **kwargs):
            raise RuntimeError(f"standard API rate limit for {kwargs['api_key']}")

    class EmptySec:
        api_key_required = False

        def collect(self, **kwargs):
            return []

    paths = write_stock_alpha_news_daily_confirmation(
        _daily_confirmation_config(tmp_path),
        sources={"alpha_vantage": AlphaLimited(), "sec_edgar": EmptySec()},
    )
    report_text = paths.json_path.read_text(encoding="utf-8")
    payload = json.loads(report_text)
    symbol_report = payload["symbol_reports"][0]

    assert secret not in report_text
    assert payload["providers_rate_limited"] == ["alpha_vantage"]
    assert payload["providers_failed"] == {}
    assert symbol_report["confirmation_status"] == "provider_limited"
    assert symbol_report["rate_limit_flag"] is True
    assert symbol_report["zero_row_reason"] == "provider_limited"
    assert symbol_report["provider_notes"]["alpha_vantage"]["error_message"].endswith("[REDACTED]")


def test_daily_confirmation_zero_recent_news_is_not_provider_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-alpha-key")

    class Empty:
        api_key_required = False

        def collect(self, **kwargs):
            return []

    class EmptyAlpha(Empty):
        api_key_required = True

    paths = write_stock_alpha_news_daily_confirmation(
        _daily_confirmation_config(tmp_path),
        sources={"alpha_vantage": EmptyAlpha(), "sec_edgar": Empty()},
    )
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    symbol_report = payload["symbol_reports"][0]

    assert payload["providers_failed"] == {}
    assert payload["providers_rate_limited"] == []
    assert symbol_report["confirmation_status"] == "no_recent_news"
    assert symbol_report["zero_row_reason"] == "no_recent_news"
    assert symbol_report["article_count"] == 0
    assert symbol_report["sec_recent_filing"] is False


def test_free_source_collection_reports_zero_rows_and_practical_next_actions(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-alpha-key")
    monkeypatch.setenv("FINNHUB_API_KEY", "test-finnhub-key")
    class Alpha:
        api_key_required = True
        def collect(self, **kwargs): return [_collected_news_row("alpha-1", "alpha_vantage")]
    class Finnhub:
        api_key_required = True
        def collect(self, **kwargs): return []
    config = load_config("config/config.stock_alpha_news_collect_alpha_vantage_finnhub_dry_run.yaml", overlay_project_config=True)
    config["ml"]["stock_alpha_news_collect_report_dir"] = str(tmp_path / "report")
    config["ml"]["stock_alpha_news_collect_output_path"] = str(tmp_path / "raw.csv")
    paths = write_stock_alpha_news_free_source_collect(config, sources={"alpha_vantage": Alpha(), "finnhub": Finnhub()})
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["providers_returned_zero_rows"] == ["finnhub"]
    assert payload["provider_zero_row_reasons"] == {
        "finnhub": "all_batches_returned_zero_rows"
    }
    assert payload["provider_batch_diagnostics"][1]["zero_row_reason"] == (
        "empty_provider_response_or_no_matching_articles"
    )
    assert payload["next_action"] == "adjust_finnhub_symbols_or_date_range"

    config["ml"]["stock_alpha_news_collect"]["providers"]["finnhub"]["enabled"] = False
    paths = write_stock_alpha_news_free_source_collect(config, sources={"alpha_vantage": Alpha()})
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["next_action"] == "write_alpha_vantage_bounded_export"


def test_free_source_collection_paid_upgrade_failures_have_specific_action(tmp_path, monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-fmp-key")
    monkeypatch.setenv("NEWSAPI_API_KEY", "test-newsapi-key")
    class Paid:
        api_key_required = True
        def __init__(self, status): self.status = status
        def collect(self, **kwargs): raise RuntimeError(f"HTTP {self.status}")
    config = _collection_config(tmp_path, dry_run=True)
    config["ml"]["stock_alpha_news_collect"]["providers"] = {
        "fmp": {"enabled": True, "api_key_env": "FMP_API_KEY"},
        "newsapi": {"enabled": True, "api_key_env": "NEWSAPI_API_KEY"},
    }
    paths = write_stock_alpha_news_free_source_collect(config, sources={"fmp": Paid(402), "newsapi": Paid(426)})
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["next_action"] == "disable_paid_or_upgrade_required_sources"


def test_news_source_adapter_distributes_hard_cap_across_symbols():
    requested_limits = []
    def fake_get(url, timeout):
        requested_limits.append(url)
        return {"articles": [{"url": url, "seendate": "20240101T120000Z", "domain": "test", "title": "headline"}]}
    rows = GdeltNewsSource(fake_get).collect(
        symbols=["AAPL", "MSFT", "NVDA"], start_date="2024-01-01", end_date="2024-01-02", limit=5, timeout=2
    )
    assert len(requested_limits) == 3
    assert len(rows) == 3
    assert {row["symbol"] for row in rows} == {"AAPL", "MSFT", "NVDA"}


def test_news_collection_planner_reads_raw_gaps_and_is_read_only(tmp_path):
    raw = tmp_path / "raw.csv"
    rows = [_collected_news_row("a1", "alpha_vantage"), _collected_news_row("m1", "alpha_vantage")]
    rows[1]["symbol"] = "MSFT"
    with raw.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    audit_config = tmp_path / "audit.yaml"
    audit_config.write_text(
        "ml:\n"
        f"  stock_alpha_news_raw_path: {raw}\n"
        "  stock_alpha_news_provider_audit_min_article_count: 10\n"
        "  stock_alpha_news_provider_audit_min_symbol_count: 5\n",
        encoding="utf-8",
    )
    stock = tmp_path / "stock.csv"
    stock.write_text("rebalance_date,symbol\n2024-01-02,AAPL\n2024-01-02,MSFT\n2024-01-02,NVDA\n2024-01-02,AMZN\n2024-01-02,META\n", encoding="utf-8")
    report = tmp_path / "report"
    config = _collection_plan_config(stock, audit_config, report)
    before = raw.read_text(encoding="utf-8")
    paths = write_stock_alpha_news_collection_plan(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["current_raw_export_row_count"] == 2
    assert payload["current_raw_export_symbol_count"] == 2
    assert payload["article_threshold_gap"] == 8
    assert payload["symbol_threshold_gap"] == 3
    assert payload["collection_invoked"] is False
    assert payload["raw_export_written"] is False
    assert raw.read_text(encoding="utf-8") == before


def test_news_collection_planner_handles_missing_stock_rows(tmp_path):
    audit_config = tmp_path / "audit.yaml"
    audit_config.write_text(
        "ml:\n  stock_alpha_news_raw_path: missing.csv\n"
        "  stock_alpha_news_provider_audit_min_article_count: 100\n"
        "  stock_alpha_news_provider_audit_min_symbol_count: 25\n",
        encoding="utf-8",
    )
    paths = write_stock_alpha_news_collection_plan(
        _collection_plan_config(tmp_path / "missing-stock.csv", audit_config, tmp_path / "report")
    )
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["input_status"]["stock_rows_exists"] is False
    assert payload["stock_row_symbols_available"] == []
    assert payload["recommended_symbol_list"]


def test_sec_edgar_fake_response_normalizes_official_filings_without_sentiment():
    payload = {"filings": {"recent": {
        "accessionNumber": ["0000320193-26-000001", "0000320193-26-000002"],
        "filingDate": ["2026-01-02", "2026-01-03"],
        "acceptanceDateTime": ["2026-01-02T12:30:00Z", "2026-01-03T13:30:00Z"],
        "form": ["8-K", "10-Q"],
        "primaryDocument": ["event.htm", "quarter.htm"],
    }}}
    source = SecEdgarNewsSource(lambda url, timeout: payload)
    rows = source.collect(symbols=["AAPL"], start_date="2026-01-01", end_date="2026-01-31", limit=5, timeout=2)
    assert source.api_key_required is False
    assert [row["event_type"] for row in rows] == ["company_event", "earnings"]
    assert all(row["provider"] == "sec_edgar" for row in rows)
    assert all(row["source"] == "SEC EDGAR" for row in rows)
    assert all(row["sentiment_score"] == "" for row in rows)
    assert all(row["relevance_score"] == "" and row["novelty_score"] == "" for row in rows)
    assert rows[0]["headline"] == (
        "AAPL SEC 8-K filing accepted 2026-01-02 "
        "accession 0000320193-26-000001"
    )
    assert rows[1]["headline"] == (
        "AAPL SEC 10-Q filing accepted 2026-01-03 "
        "accession 0000320193-26-000002"
    )
    assert len({row["article_id"] for row in rows}) == 2
    assert "official_filings_source" in PROVIDER_METADATA["sec_edgar"]["statuses"]


def test_sec_edgar_repeated_forms_have_distinct_metadata_headlines():
    payload = {"filings": {"recent": {
        "accessionNumber": [
            "0000320193-26-000010",
            "0000320193-26-000011",
            "0000320193-26-000012",
        ],
        "filingDate": ["2026-02-01", "2026-02-02", "2026-02-03"],
        "acceptanceDateTime": ["", "", ""],
        "form": ["4", "4", "4"],
        "primaryDocument": ["one.xml", "two.xml", "three.xml"],
    }}}
    rows = SecEdgarNewsSource(lambda url, timeout: payload).collect(
        symbols=["AAPL"],
        start_date="2026-02-01",
        end_date="2026-02-28",
        limit=5,
        timeout=2,
    )

    assert len(rows) == 3
    assert len({row["headline"] for row in rows}) == 3
    assert all("AAPL SEC 4 filing filed 2026-02-" in row["headline"] for row in rows)
    assert all(" accession 0000320193-26-" in row["headline"] for row in rows)
    assert all(row["sentiment_score"] == "" for row in rows)


def test_gdelt_http_429_is_rate_limited_with_clean_next_action(tmp_path):
    class RateLimited:
        api_key_required = False
        def collect(self, **kwargs):
            raise HTTPError("https://gdelt.test", 429, "Too Many Requests", None, None)
    config = _collection_config(tmp_path, dry_run=True)
    paths = write_stock_alpha_news_free_source_collect(config, sources={"gdelt": RateLimited()})
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["providers_rate_limited"] == ["gdelt"]
    assert payload["provider_row_counts"]["gdelt"] == 0
    assert payload["providers_returned_zero_rows"] == ["gdelt"]
    assert payload["provider_zero_row_reasons"] == {"gdelt": "rate_limited"}
    assert payload["provider_batch_diagnostics"][0]["rate_limited"] is True
    assert payload["provider_batch_diagnostics"][0]["zero_row_reason"] == "rate_limited"
    assert "gdelt" not in payload["providers_failed"]
    assert payload["next_action"] == "retry_gdelt_later_or_reduce_request"
    assert "rate_limited_or_retry_later" in payload["provider_policy"]["gdelt"]["statuses"]


def test_gdelt_non_rate_limit_failure_reports_zero_rows(tmp_path):
    class Unavailable:
        api_key_required = False

        def collect(self, **kwargs):
            raise TimeoutError("bounded request timed out")

    config = _collection_config(tmp_path, dry_run=True)
    paths = write_stock_alpha_news_free_source_collect(
        config, sources={"gdelt": Unavailable()}
    )
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))

    assert payload["provider_row_counts"]["gdelt"] == 0
    assert payload["providers_returned_zero_rows"] == ["gdelt"]
    assert payload["providers_rate_limited"] == []
    assert payload["providers_failed"]["gdelt"].startswith("TimeoutError:")
    assert payload["provider_zero_row_reasons"] == {"gdelt": "provider_error"}


def test_alpha_vantage_rate_limit_payload_is_reported_without_key_disclosure(tmp_path, monkeypatch):
    secret = "test-alpha-secret"
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", secret)

    class AlphaRateLimited:
        api_key_required = True

        def collect(self, **kwargs):
            raise RuntimeError(
                "standard API call frequency exceeded for "
                f"{kwargs['api_key']}"
            )

    config = _collection_config(tmp_path, dry_run=True)
    config["ml"]["stock_alpha_news_collect"]["providers"] = {
        "alpha_vantage": {"enabled": True, "api_key_env": "ALPHA_VANTAGE_API_KEY"}
    }

    paths = write_stock_alpha_news_free_source_collect(
        config, sources={"alpha_vantage": AlphaRateLimited()}
    )
    report_text = paths.json_path.read_text(encoding="utf-8")
    payload = json.loads(report_text)

    assert secret not in report_text
    assert payload["providers_rate_limited"] == ["alpha_vantage"]
    assert payload["provider_zero_row_reasons"] == {"alpha_vantage": "rate_limited"}
    assert payload["provider_batch_diagnostics"][0]["rate_limited"] is True
    assert payload["provider_batch_diagnostics"][0]["error_message"].endswith(
        "for [REDACTED]"
    )


def test_massive_collection_uses_polygon_key_fallback_and_redacts_rate_limit(tmp_path, monkeypatch):
    secret = "test-polygon-secret"
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.setenv("POLYGON_API_KEY", secret)

    class MassiveRateLimited:
        api_key_required = True

        def collect(self, **kwargs):
            raise RuntimeError(f"HTTP 429 rate limit for apiKey={kwargs['api_key']}")

    config = _collection_config(tmp_path, dry_run=True)
    config["ml"]["stock_alpha_news_collect"]["providers"] = {
        "massive_stock_news": {
            "enabled": True,
            "api_key_env": "MASSIVE_API_KEY",
            "api_key_env_fallbacks": ["POLYGON_API_KEY"],
        }
    }

    paths = write_stock_alpha_news_free_source_collect(
        config, sources={"massive_stock_news": MassiveRateLimited()}
    )
    report_text = paths.json_path.read_text(encoding="utf-8")
    payload = json.loads(report_text)

    assert secret not in report_text
    assert payload["providers_skipped_missing_key"] == []
    assert payload["providers_rate_limited"] == ["massive_stock_news"]
    assert payload["provider_zero_row_reasons"] == {"massive_stock_news": "rate_limited"}
    diagnostic = payload["provider_batch_diagnostics"][0]
    assert diagnostic["rate_limited"] is True
    assert diagnostic["error_message"].endswith("apiKey=[REDACTED]")


def test_massive_collection_missing_primary_and_fallback_keys_skips(tmp_path, monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)

    config = _collection_config(tmp_path, dry_run=True)
    config["ml"]["stock_alpha_news_collect"]["providers"] = {
        "massive_stock_news": {
            "enabled": True,
            "api_key_env": "MASSIVE_API_KEY",
            "api_key_env_fallbacks": ["POLYGON_API_KEY"],
        }
    }

    paths = write_stock_alpha_news_free_source_collect(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))

    assert payload["providers_requested"] == ["massive_stock_news"]
    assert payload["providers_attempted"] == []
    assert payload["providers_skipped_missing_key"] == ["massive_stock_news"]
    assert payload["output_written"] is False


def test_news_pipeline_inspect_tiny_fixture_is_read_only(tmp_path):
    config = load_config(
        "config/config.stock_alpha_news_pipeline_inspect_tiny_fixture.yaml",
        overlay_project_config=True,
    )
    ml = config["ml"]
    dev_dir = tmp_path / "dev"
    ml["stock_alpha_report_root"] = str(tmp_path)
    ml["stock_alpha_news_contract_path"] = str(dev_dir / "stock_alpha_news_contract.csv")
    ml["stock_alpha_news_features_path"] = str(dev_dir / "stock_alpha_news_features.csv")
    ml["stock_alpha_news_coverage_audit_dir"] = str(dev_dir / "news_coverage_audit")
    ml["stock_alpha_news_pipeline_preflight_output_dir"] = str(dev_dir)
    ml["stock_alpha_news_pipeline_inspect_output_dir"] = str(dev_dir)

    paths = write_stock_alpha_news_pipeline_inspect(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))

    assert payload["next_action"] == "run_pipeline_preflight"
    assert payload["existence_checks"]["raw_provider_file_exists"] is True
    assert payload["existence_checks"]["stock_rows_file_exists"] is True
    assert payload["existence_checks"]["nearby_feature_audit_path_coherent"] is True
    assert payload["config_summary"]["stock_alpha_news_enable_transformer"] is False
    assert payload["stage_order"] == [
        "provider_audit", "contract_ingest", "coverage_audit",
        "feature_generation", "readiness_preflight",
    ]
    assert payload["inspection_only"] is True
    assert payload["files_ingested"] is False
    assert payload["features_generated"] is False
    assert payload["model_training_invoked"] is False
    assert payload["diagnostics_invoked"] is False
    assert not Path(ml["stock_alpha_news_contract_path"]).exists()
    assert not Path(ml["stock_alpha_news_features_path"]).exists()


def test_news_pipeline_inspect_real_missing_raw_reports_next_action(tmp_path, capsys):
    config = load_config(
        "config/config.stock_alpha_news_pipeline_inspect_real_template.yaml",
        overlay_project_config=True,
    )
    config["ml"]["stock_alpha_news_raw_path"] = str(tmp_path / "missing.csv")
    config["ml"]["stock_alpha_news_pipeline_inspect_output_dir"] = str(tmp_path / "inspect")

    run_ml_stock_alpha_news_pipeline_inspect(config)
    output = capsys.readouterr().out
    payload = json.loads(
        (tmp_path / "inspect" / "stock_alpha_news_pipeline_inspect.json").read_text(encoding="utf-8")
    )

    assert payload["next_action"] == "provide_raw_news_file"
    assert "next_action=provide_raw_news_file" in output
    assert "stock_alpha_news_enable_transformer=false" in output
    assert "model_training_invoked=false" in output
    assert "diagnostics_invoked=false" in output


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("stock_alpha_news_coverage_min_symbol_coverage", "bad", "must be numeric"),
        ("stock_alpha_news_unknown_min_rate", 0.5, "unknown stock-alpha news threshold"),
    ],
)
def test_news_pipeline_inspect_malformed_thresholds_fail_cleanly(key, value, message, capsys):
    config = load_config(
        "config/config.stock_alpha_news_pipeline_inspect_tiny_fixture.yaml",
        overlay_project_config=True,
    )
    config["ml"][key] = value

    with pytest.raises(SystemExit) as exc:
        run_ml_stock_alpha_news_pipeline_inspect(config)

    assert exc.value.code == 1
    assert message in capsys.readouterr().out


def test_stock_alpha_news_pipeline_preflight_real_template_missing_raw_is_clean(tmp_path, capsys):
    config = load_config(
        "config/config.stock_alpha_news_pipeline_preflight_real_template.yaml",
        overlay_project_config=True,
    )
    _redirect_pipeline_outputs(config, tmp_path)
    config["ml"]["stock_alpha_news_raw_path"] = str(tmp_path / "missing_raw.csv")

    run_ml_stock_alpha_news_pipeline_preflight(config)
    output = capsys.readouterr().out
    payload = json.loads((tmp_path / "dev" / "stock_alpha_news_pipeline_preflight.json").read_text(encoding="utf-8"))

    assert "STOCK-ALPHA NEWS PIPELINE PREFLIGHT" in output
    assert "pipeline_safe_for_news_transformer_training=false" in output
    assert "stopped_stage=provider_audit" in output
    assert "missing_raw.csv" in output
    assert payload["stopped_stage"] == "provider_audit"
    assert payload["stages"]["contract_ingest"]["attempted"] is False
    assert not Path(config["ml"]["stock_alpha_news_contract_path"]).exists()
    assert not Path(config["ml"]["stock_alpha_news_features_path"]).exists()


def test_stock_alpha_news_pipeline_preflight_stops_before_features_when_coverage_unsafe(tmp_path):
    config = load_config(
        "config/config.stock_alpha_news_pipeline_preflight_tiny_fixture.yaml",
        overlay_project_config=True,
    )
    _redirect_pipeline_outputs(config, tmp_path)
    config["ml"]["stock_alpha_news_coverage_min_symbol_coverage"] = 1.01

    payload = build_stock_alpha_news_pipeline_preflight(config)

    assert payload["stopped_stage"] == "coverage_audit"
    assert payload["stages"]["coverage_audit"]["attempted"] is True
    assert payload["stages"]["coverage_audit"]["safe"] is False
    assert "coverage_audit: symbol coverage below minimum" in payload["blocking_issues"]
    assert payload["stages"]["feature_generation"]["attempted"] is False
    assert payload["stages"]["readiness_preflight"]["attempted"] is False
    assert not Path(config["ml"]["stock_alpha_news_features_path"]).exists()


def _config(
    news,
    features=None,
    *,
    enable=False,
    min_symbol=0.8,
    min_date=0.8,
):
    ml = {
        "stock_alpha_news_contract_path": str(news),
        "stock_alpha_news_enable_transformer": enable,
        "stock_alpha_news_min_symbol_coverage": min_symbol,
        "stock_alpha_news_min_date_coverage": min_date,
    }
    if features is not None:
        ml["stock_alpha_news_features_path"] = str(features)
    return {"ml": ml}


def _stock_rows(symbols=("AAPL",)):
    return [
        {"rebalance_date": "2024-01-02", "symbol": symbol}
        for symbol in symbols
    ]


def _write_news(
    path,
    *,
    article_id="real-1",
    symbol="AAPL",
    published="2024-01-01T00:00:00Z",
    ingested="2024-01-01T00:00:00Z",
    source="vendor",
    headline="AAPL reports earnings",
    body="A real vendor news summary.",
    sentiment="0.2",
    relevance="0.9",
    novelty="0.8",
    event_type="earnings",
):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "article_id",
                "symbol",
                "published_at_utc",
                "ingested_at",
                "source",
                "headline",
                "body_or_summary",
                "sentiment_score",
                "relevance_score",
                "novelty_score",
                "event_type",
                "language",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "article_id": article_id,
                "symbol": symbol,
                "published_at_utc": published,
                "ingested_at": ingested,
                "source": source,
                "headline": headline,
                "body_or_summary": body,
                "sentiment_score": sentiment,
                "relevance_score": relevance,
                "novelty_score": novelty,
                "event_type": event_type,
                "language": "en",
            }
        )


def _write_features(path, *, symbol="AAPL", extra_fields=None):
    extra_fields = dict(extra_fields or {})
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["rebalance_date", "symbol", *REQUIRED_NEWS_AGGREGATE_FEATURES, "news_has_coverage_30d", *extra_fields]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {column: "0.1" for column in fieldnames}
            | {"rebalance_date": "2024-01-02", "symbol": symbol, "news_has_coverage_30d": "true"}
            | extra_fields
        )


def _guardrails():
    return {
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
        "promotion_thresholds_changed": False,
    }


def _feature_diagnostics_config(features: Path, stock_rows: Path, report_dir: Path) -> dict:
    return {"ml": {
        "stock_alpha_news_features_path": str(features),
        "stock_alpha_stock_rows_path": str(stock_rows),
        "stock_alpha_news_feature_diagnostics_report_dir": str(report_dir),
    }}


def _source_diagnostics_config(contract: Path, stock_rows: Path, report_dir: Path) -> dict:
    return {"ml": {
        "stock_alpha_news_contract_path": str(contract),
        "stock_alpha_stock_rows_path": str(stock_rows),
        "stock_alpha_news_source_diagnostics_report_dir": str(report_dir),
        "stock_alpha_news_source_column": "source",
        "stock_alpha_news_provider_column": "provider",
    }}


def _collection_config(tmp_path: Path, *, dry_run: bool) -> dict:
    return {"ml": {
        "stock_alpha_news_collect_report_dir": str(tmp_path / "report"),
        "stock_alpha_news_collect_output_path": str(tmp_path / "raw.csv"),
        "stock_alpha_news_collect": {
            "enabled": True, "dry_run": dry_run, "allow_overwrite": False,
            "max_articles_per_provider": 5, "request_timeout_seconds": 2,
            "start_date": "2024-01-01", "end_date": "2024-01-02", "symbols": ["AAPL"],
            "providers": {"gdelt": {"enabled": True}},
        },
    }}


def _daily_confirmation_config(tmp_path: Path) -> dict:
    return {"ml": {
        "stock_alpha_news_daily_confirmation_report_dir": str(tmp_path / "daily"),
        "stock_alpha_news_confirmation": {
            "enabled": True,
            "dry_run": True,
            "inspection_only": True,
            "as_of_utc": "2026-04-21T00:00:00Z",
            "lookback_hours": 72,
            "max_symbols": 5,
            "max_articles_per_symbol": 3,
            "max_provider_requests": 10,
            "request_timeout_seconds": 2,
            "symbols": ["AAPL"],
            "providers": {
                "alpha_vantage": {
                    "enabled": True,
                    "api_key_env": "ALPHA_VANTAGE_API_KEY",
                },
                "sec_edgar": {"enabled": True},
            },
        },
    }}


def _collected_news_row(article_id: str, provider: str) -> dict:
    return {
        "article_id": article_id, "symbol": "AAPL", "published_at_utc": "2024-01-01T10:00:00Z",
        "source": f"{provider}-source", "headline": "Headline", "body_or_summary": "Summary",
        "sentiment_score": "", "relevance_score": "", "novelty_score": "", "event_type": "",
        "language": "en", "ingested_at": "2024-01-01T10:05:00Z", "provider": provider,
        "provider_article_id": article_id, "provider_url": f"https://example.test/{article_id}",
    }


def _collection_plan_config(stock_rows: Path, audit_config: Path, report_dir: Path) -> dict:
    return {"ml": {
        "stock_alpha_stock_rows_path": str(stock_rows),
        "stock_alpha_news_collection_plan_report_dir": str(report_dir),
        "stock_alpha_news_provider_audit_config_path": str(audit_config),
        "stock_alpha_news_collect": {
            "symbols": ["AAPL", "MSFT", "NVDA", "AMZN", "META"],
            "max_articles_per_provider": 10,
            "providers": {"alpha_vantage": {"enabled": True}, "finnhub": {"enabled": False}},
        },
    }}


def _write_stock_rows_csv(tmp_path: Path) -> Path:
    path = tmp_path / "stock_rows.csv"
    path.write_text("rebalance_date,symbol\n2024-01-02,AAPL\n", encoding="utf-8")
    return path


def _feature_row(rows: list[dict], rebalance_date: str, symbol: str) -> dict:
    for row in rows:
        if row["rebalance_date"] == rebalance_date and row["symbol"] == symbol:
            return row
    raise AssertionError(f"missing feature row for {rebalance_date} {symbol}")


def _contract_row(rows: list[dict], article_id: str) -> dict:
    for row in rows:
        if row["article_id"] == article_id:
            return row
    raise AssertionError(f"missing contract row for {article_id}")


def _ingest_config(raw_path: Path, contract_path: Path, audit_dir: Path) -> dict:
    return {
        "ml": {
            "stock_alpha_news_raw_path": str(raw_path),
            "stock_alpha_news_contract_path": str(contract_path),
            "stock_alpha_news_contract_ingest_audit_dir": str(audit_dir),
            **_guardrails(),
        }
    }


def _redirect_pipeline_outputs(config: dict, output_dir: Path) -> None:
    ml = config["ml"]
    ml["stock_alpha_report_root"] = str(output_dir)
    dev_dir = output_dir / "dev"
    ml["stock_alpha_news_contract_path"] = str(dev_dir / "stock_alpha_news_contract.csv")
    ml["stock_alpha_news_features_path"] = str(dev_dir / "stock_alpha_news_features.csv")
    ml["stock_alpha_news_provider_audit_dir"] = str(dev_dir / "news_provider_audit")
    ml["stock_alpha_news_contract_ingest_audit_dir"] = str(dev_dir / "news_contract_ingest")
    ml["stock_alpha_news_coverage_audit_dir"] = str(dev_dir / "news_coverage_audit")
    ml["stock_alpha_news_pipeline_preflight_output_dir"] = str(dev_dir)


def _write_raw_news_csv(
    path: Path,
    *,
    rows: list[dict] | None = None,
    symbol: str = "AAPL",
    published: str = "2024-01-01T00:00:00Z",
    ingested: str = "2024-01-01T00:01:00Z",
    event_type: str = "earnings",
) -> None:
    rows = rows or [
        _raw_news_row(
            symbol=symbol,
            published=published,
            ingested=ingested,
            event_type=event_type,
        )
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_NEWS_CONTRACT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _raw_news_row(
    *,
    article_id: str = "raw-1",
    symbol: str = "AAPL",
    published: str = "2024-01-01T00:00:00Z",
    ingested: str = "2024-01-01T00:01:00Z",
    source: str = "vendor",
    headline: str = "AAPL reports earnings",
    body: str = "A real vendor news summary.",
    sentiment: str = "0.2",
    relevance: str = "0.9",
    novelty: str = "0.8",
    event_type: str = "earnings",
    language: str = "en",
) -> dict:
    return {
        "article_id": article_id,
        "symbol": symbol,
        "published_at_utc": published,
        "ingested_at": ingested,
        "source": source,
        "headline": headline,
        "body_or_summary": body,
        "sentiment_score": sentiment,
        "relevance_score": relevance,
        "novelty_score": novelty,
        "event_type": event_type,
        "language": language,
    }


def _news_row(article_id, symbol, published, ingested, sentiment, event_type):
    return {
        "article_id": article_id,
        "symbol": symbol,
        "published_at_utc": published,
        "ingested_at": ingested,
        "source": "vendor",
        "headline": "Real headline",
        "body_or_summary": "Real summary",
        "sentiment_score": sentiment,
        "relevance_score": "0.9",
        "novelty_score": "0.8",
        "event_type": event_type,
        "language": "en",
    }
