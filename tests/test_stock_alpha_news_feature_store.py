from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.research.ml.stock_level.stock_alpha_news_feature_store import (
    FEATURE_STORE_CONTRACT,
    publish_pit_news_feature_store,
)


MODEL = {
    "model_id": "ProsusAI/finbert",
    "model_revision": "0123456789abcdef",
    "tokenizer_id": "ProsusAI/finbert",
    "tokenizer_revision": "fedcba9876543210",
}


def _hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest().upper()


def _parents(scored_articles, spine_rows, aliases):
    normalised_aliases = {
        str(alias).upper().replace(".", "-"):
        str(canonical).upper().replace(".", "-")
        for alias, canonical in aliases.items()
    }
    return {
        "canonical_corpus_manifest": {
            "canonical_corpus_identity": "canonical-news-v1",
            "canonical_corpus_checksum": "CORPUS",
        },
        "score_store_manifest": {
            "score_store_identity": "finbert-scores-v1",
            "score_store_checksum": "SCORES",
            "production_scoring_complete": True,
            "finbert_model_identity": MODEL,
            "scored_rows_logical_checksum": _hash(scored_articles),
        },
        "daily_spine_manifest": {
            "daily_spine_identity": "daily-spine-v2",
            "daily_spine_checksum": "SPINE",
            "spine_rows_logical_checksum": _hash(spine_rows),
        },
        "ticker_mapping_manifest": {
            "ticker_mapping_identity": "canonical-symbol-map-v1",
            "ticker_mapping_checksum": "MAPPING",
            "ticker_aliases_logical_checksum": _hash(normalised_aliases),
        },
    }


def _article(
    article_id,
    symbol,
    published,
    collected,
    signed,
    *,
    model=MODEL,
):
    return {
        "article_id": article_id,
        "symbol": symbol,
        "published_at_utc": published,
        "collected_at_utc": collected,
        "finbert_model_id": model["model_id"],
        "finbert_model_revision": model["model_revision"],
        "tokenizer_id": model["tokenizer_id"],
        "tokenizer_revision": model["tokenizer_revision"],
        "signed_sentiment_score": signed,
        "positive_probability": 0.8 if signed > 0 else 0.1,
        "negative_probability": 0.1 if signed > 0 else 0.8,
    }


def _spine():
    return [
        {
            "asset_id": "asset_BRK-B",
            "symbol": "BRK-B",
            "decision_session_date": "2024-01-03",
            "decision_timestamp": "2024-01-03T21:00:00-05:00",
        },
        {
            "asset_id": "asset_TSLA",
            "symbol": "TSLA",
            "decision_session_date": "2024-01-03",
            "decision_timestamp": "2024-01-04T02:00:00Z",
        },
    ]


def _publish(tmp_path, **updates):
    scored_articles = [
        _article(
            "eligible", "BRK.B",
            "2024-01-03T10:00:00Z", "2024-01-03T11:00:00Z", 0.7,
        ),
        _article(
            "future", "BRK.B",
            "2024-01-04T03:00:00Z", "2024-01-04T03:01:00Z", -0.9,
        ),
    ]
    spine_rows = _spine()
    aliases = {"BRK.B": "BRK-B"}
    values = {
        **_parents(scored_articles, spine_rows, aliases),
        "scored_articles": scored_articles,
        "spine_rows": spine_rows,
        "ticker_aliases": aliases,
        "output_root": tmp_path / "store",
        "finbert_model_identity": MODEL,
        "source_commit": "abc123",
        "lookback_days": (3, 1),
    }
    values.update(updates)
    return publish_pit_news_feature_store(**values)


def _rows(manifest, root):
    result = []
    for partition in manifest["partitions"]:
        path = root / partition["relative_path"]
        result.extend(json.loads(line) for line in path.read_text().splitlines())
    return result


def test_identity_pit_alias_timezone_missingness_and_ordering(tmp_path):
    first = _publish(tmp_path)
    rows = _rows(first, tmp_path / "store")
    brk = next(
        row for row in rows
        if row["symbol"] == "BRK-B" and row["lookback_days"] == 3
    )
    tsla = next(
        row for row in rows
        if row["symbol"] == "TSLA" and row["lookback_days"] == 3
    )
    assert first["feature_store_contract"] == FEATURE_STORE_CONTRACT
    assert first["ticker_mapping_identity"] == "canonical-symbol-map-v1"
    assert first["timezone"] == "UTC"
    assert first["aggregation_windows"] == [
        {"window_id": "trailing_1_calendar_days", "days": 1},
        {"window_id": "trailing_3_calendar_days", "days": 3},
    ]
    assert brk["decision_timestamp"] == "2024-01-04T02:00:00Z"
    assert brk["eligible_article_count"] == 1
    assert brk["mean_signed_sentiment"] == pytest.approx(0.7)
    assert tsla["news_missing"] is True
    assert tsla["mean_signed_sentiment"] is None
    assert first["eligibility_evidence"]["post_decision_article_exclusions"] > 0
    assert [row["decision_date"] for row in first["partitions"]] == sorted(
        row["decision_date"] for row in first["partitions"]
    )

    second = _publish(tmp_path)
    assert second["logical_checksum"] == first["logical_checksum"]
    assert all(
        row["publication_result"] == "SKIPPED_COMPATIBLE"
        for row in second["partition_publication_results"]
    )
    assert second["manifest_publication_result"] == "SKIPPED_COMPATIBLE"


def test_incompatible_partition_fails_closed(tmp_path):
    manifest = _publish(tmp_path)
    partition = tmp_path / "store" / manifest["partitions"][0]["relative_path"]
    partition.write_text("incompatible\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Incompatible existing news partition"):
        _publish(tmp_path)


@pytest.mark.parametrize(
    "model_update,error",
    [
        ({"model_revision": "main"}, "must be pinned"),
        ({"model_revision": "different"}, "model identity mismatch"),
    ],
)
def test_unpinned_or_mismatched_finbert_identity_rejected(
    tmp_path, model_update, error
):
    requested = {**MODEL, **model_update}
    with pytest.raises(ValueError, match=error):
        _publish(tmp_path, finbert_model_identity=requested)


def test_publisher_contains_no_scorer_or_transformer_training_execution():
    source = Path(
        "core/research/ml/stock_level/stock_alpha_news_feature_store.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "score_finbert_articles",
        "HuggingFaceFinBertAdapter",
        "news_analysis_transformer",
        "torch",
        "fit(",
    ):
        assert forbidden not in source
