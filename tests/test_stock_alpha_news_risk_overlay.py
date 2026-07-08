from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.research.ml.stock_level.news_risk_overlay import (
    NewsRiskOverlayConfig,
    append_shadow_decision_log,
    build_news_risk_labels,
    chronological_splits,
    evaluate_candidate,
    join_news_to_stock_alpha_observations,
    shadow_decision_row,
    validate_news_paper_safety,
)


def test_future_news_cannot_enter_earlier_decisions() -> None:
    stock = [_stock_row("AAPL", "2024-01-03T14:30:00+00:00")]
    news = [
        _news_row("AAPL", "2024-01-03T14:00:00+00:00", sentiment=-0.5),
        _news_row("AAPL", "2024-01-03T15:00:00+00:00", sentiment=-0.9),
    ]

    enriched, audit = join_news_to_stock_alpha_observations(stock, news)

    assert enriched[0]["news_sentiment"] == -0.5
    assert audit["future_news_rows_rejected"] == 1
    assert audit["leakage_violation_count"] == 0


def test_ingestion_timestamp_is_preferred_over_publication_timestamp() -> None:
    stock = [_stock_row("AAPL", "2024-01-03T14:30:00+00:00")]
    news = [
        {
            "symbol": "AAPL",
            "published_at": "2024-01-03T13:00:00+00:00",
            "ingested_at": "2024-01-03T15:00:00+00:00",
            "sentiment": "-0.7",
        }
    ]

    enriched, audit = join_news_to_stock_alpha_observations(stock, news)

    assert enriched[0]["news_coverage_status"] == "NO_COVERAGE"
    assert audit["future_news_rows_rejected"] == 1


def test_after_hours_news_maps_to_later_decision() -> None:
    stock = [
        _stock_row("AAPL", "2024-01-05T16:00:00+00:00"),
        _stock_row("AAPL", "2024-01-08T14:30:00+00:00"),
    ]
    news = [_news_row("AAPL", "2024-01-05T22:00:00+00:00", sentiment=-0.8)]

    enriched, _ = join_news_to_stock_alpha_observations(stock, news)

    assert enriched[0]["news_coverage_status"] == "NO_COVERAGE"
    assert enriched[1]["news_sentiment"] == -0.8


def test_point_in_time_join_propagates_matched_news_evidence() -> None:
    stock = [_stock_row("AAPL", "2024-01-03T14:30:00+00:00")]
    news = [{
        "symbol": "AAPL",
        "event_id": "event-1",
        "ingested_at": "2024-01-03T14:00:00+00:00",
        "published_at_utc": "2024-01-03T13:55:00+00:00",
        "headline_text": "Company announces ordinary update",
        "summary_text": "Point-in-time summary",
        "body_text": "Point-in-time body",
        "provider": "official_feed",
        "sentiment": "-0.2",
    }]

    enriched, audit = join_news_to_stock_alpha_observations(stock, news)
    row = enriched[0]

    assert row["headline_text"] == "Company announces ordinary update"
    assert row["summary_text"] == "Point-in-time summary"
    assert row["body_text"] == "Point-in-time body"
    assert row["provider"] == "official_feed"
    assert row["availability_timestamp"] == "2024-01-03T14:00:00+00:00"
    assert row["availability_timestamp_source"] == "ingested_at"
    assert row["publication_timestamp"] == "2024-01-03T13:55:00+00:00"
    assert row["availability_timestamp"] <= row["decision_timestamp"]
    assert audit["leakage_violation_count"] == 0


def test_join_does_not_fabricate_missing_news_evidence() -> None:
    enriched, _ = join_news_to_stock_alpha_observations(
        [_stock_row("AAPL", "2024-01-03T14:30:00+00:00")],
        [{"symbol": "AAPL", "event_id": "event-1", "published_at": "2024-01-03T14:00:00+00:00", "sentiment": "0.0"}],
    )

    row = enriched[0]
    assert "headline_text" not in row
    assert "provider" not in row
    assert "availability_timestamp" not in row
    assert row["news_feature_timestamp"] == "2024-01-03T14:00:00+00:00"


def test_duplicate_news_does_not_multiply_features() -> None:
    stock = [_stock_row("AAPL", "2024-01-03T14:30:00+00:00")]
    news = [
        _news_row("AAPL", "2024-01-03T14:00:00+00:00", sentiment=-0.5, event_id="same"),
        _news_row("AAPL", "2024-01-03T14:00:00+00:00", sentiment=-0.1, event_id="same"),
    ]

    enriched, _ = join_news_to_stock_alpha_observations(stock, news)

    assert enriched[0]["news_sentiment"] == -0.1


def test_missing_coverage_is_not_neutral_sentiment() -> None:
    enriched, _ = join_news_to_stock_alpha_observations(
        [_stock_row("MSFT", "2024-01-03T14:30:00+00:00")],
        [_news_row("AAPL", "2024-01-03T14:00:00+00:00", sentiment=0.0)],
    )

    assert enriched[0]["news_coverage_status"] == "NO_COVERAGE"
    assert enriched[0]["news_missing_coverage"] is True
    assert "news_sentiment" not in enriched[0]


def test_risk_label_uses_configurable_loss_threshold() -> None:
    rows = build_news_risk_labels(
        [{"symbol": "AAPL", "actual_forward_return_10d": "-0.04"}],
        NewsRiskOverlayConfig(adverse_return_threshold=-0.03),
    )

    assert rows[0]["news_risk_label"] == 1


def test_chronological_splits_never_shuffle() -> None:
    rows = [
        _stock_row("AAPL", "2024-01-01T00:00:00+00:00"),
        _stock_row("AAPL", "2024-01-02T00:00:00+00:00"),
        _stock_row("AAPL", "2024-01-03T00:00:00+00:00"),
        _stock_row("AAPL", "2024-01-04T00:00:00+00:00"),
    ]

    splits = chronological_splits(rows, folds=2)

    assert splits
    for train, test in splits:
        assert max(train) < min(test)


def test_shadow_mode_never_submits_orders(tmp_path: Path) -> None:
    decision = evaluate_candidate(
        symbol="AAPL",
        decision_timestamp=datetime(2024, 1, 3, tzinfo=timezone.utc),
        base_position_size=100.0,
        price_model_score=0.8,
        recent_features={"news_coverage_status": "COVERED"},
        risk_probability=0.9,
    )
    row = shadow_decision_row(
        timestamp=datetime(2024, 1, 3, tzinfo=timezone.utc),
        symbol="AAPL",
        price_score=0.8,
        price_only_position_size=100.0,
        decision=decision,
    )
    path = tmp_path / "shadow.csv"
    append_shadow_decision_log(path, [row])

    with path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert written[0]["order_submitted"] == "False"
    assert written[0]["news_action"] == "BLOCK"


def test_paper_mode_rejects_live_endpoint() -> None:
    with pytest.raises(ValueError, match="paper Alpaca endpoint"):
        validate_news_paper_safety(
            paper_orders=True,
            alpaca_endpoint="https://api.alpaca.markets",
            allow_env="1",
            readiness_ok=True,
            leakage_ok=True,
            model_loaded=True,
            inputs_fresh=True,
        )


def test_identical_inputs_produce_deterministic_decisions() -> None:
    kwargs = {
        "symbol": "AAPL",
        "decision_timestamp": datetime(2024, 1, 3, tzinfo=timezone.utc),
        "base_position_size": 100.0,
        "price_model_score": 0.8,
        "recent_features": {"news_coverage_status": "COVERED"},
        "risk_probability": 0.6,
    }

    first = evaluate_candidate(**kwargs)
    second = evaluate_candidate(**kwargs)

    assert first == second
    assert first.action == "REDUCE"


def _stock_row(symbol: str, timestamp: str) -> dict[str, str]:
    return {
        "symbol": symbol,
        "decision_timestamp": timestamp,
        "price_model_score": "0.8",
        "price_only_position_size": "100",
    }


def _news_row(
    symbol: str,
    timestamp: str,
    *,
    sentiment: float,
    event_id: str = "event",
) -> dict[str, str]:
    return {
        "symbol": symbol,
        "event_id": event_id,
        "ingested_at": timestamp,
        "sentiment": str(sentiment),
        "event_count": "1",
    }
