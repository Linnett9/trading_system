from __future__ import annotations

import csv
import json

from config.config_loader import load_config

from core.research.ml.stock_level.stock_alpha_news_contract import (
    REQUIRED_NEWS_AGGREGATE_FEATURES,
    build_stock_alpha_news_features,
    validate_news_contract,
    write_stock_alpha_news_features,
    write_stock_alpha_news_features_from_config,
    write_stock_alpha_news_contract_validation,
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
    assert audit["missing_or_invalid_timestamp_count"] == 0
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


def _write_features(path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["rebalance_date", "symbol", *REQUIRED_NEWS_AGGREGATE_FEATURES]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({column: "0.1" for column in fieldnames} | {"rebalance_date": "2024-01-02", "symbol": "AAPL"})


def _guardrails():
    return {
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
        "promotion_thresholds_changed": False,
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
