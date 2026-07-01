from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from application.services.ml_commands_stock import run_ml_stock_alpha_news_contract_ingest
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
from core.research.ml.stock_level.stock_alpha_news_readiness_preflight import (
    build_stock_alpha_news_readiness_preflight,
    write_stock_alpha_news_readiness_preflight,
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


def test_future_published_at_fails(tmp_path):
    news = tmp_path / "news.csv"
    _write_news(news, published="2024-01-03T00:00:00Z", ingested="2024-01-01T00:00:00Z")

    validation = validate_news_contract(_config(news, min_symbol=0.0, min_date=0.0), _stock_rows())

    assert validation.available is False
    assert validation.reason == "news contract contains future articles"
    assert validation.future_article_count == 1


def test_future_ingested_at_fails(tmp_path):
    news = tmp_path / "news.csv"
    _write_news(news, published="2024-01-01T00:00:00Z", ingested="2024-01-03T00:00:00Z")

    validation = validate_news_contract(_config(news, min_symbol=0.0, min_date=0.0), _stock_rows())

    assert validation.available is False
    assert validation.reason == "news contract contains future articles"
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
    assert audit["pit_violation_count"] == 1


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
    assert audit["pit_violation_count"] > 0
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


def test_news_feature_writer_rejects_invalid_contract(tmp_path):
    news = tmp_path / "news.csv"
    _write_news(news, published="2024-01-03T00:00:00Z")

    try:
        write_stock_alpha_news_features(
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
    except ValueError as exc:
        assert "cannot aggregate stock-alpha news features" in str(exc)
    else:
        raise AssertionError("invalid point-in-time news contract should not aggregate")


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
