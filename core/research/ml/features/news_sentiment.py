from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from typing import Sequence


@dataclass(frozen=True)
class NewsEvent:
    event_id: str
    symbol: str
    headline: str
    published_at: datetime | None
    first_seen_at: datetime | None
    retrieved_at: datetime | None
    source_name: str
    reliability_tier: int = 2
    sentiment_score: float | None = None

    @property
    def event_time(self) -> datetime | None:
        if self.published_at and self.first_seen_at:
            return max(_utc(self.published_at), _utc(self.first_seen_at))
        if self.first_seen_at:
            return _utc(self.first_seen_at)
        if self.published_at:
            return _utc(self.published_at)
        return None

    @property
    def body_hash(self) -> str:
        return hashlib.sha256(self.headline.strip().lower().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SentimentAggregationAudit:
    eligible_event_count: int
    excluded_future_event_count: int
    excluded_low_reliability_count: int
    missing_event_time_count: int
    deduplicated_event_count: int


def aggregate_news_sentiment_features(
    events: Sequence[NewsEvent],
    feature_timestamp: datetime,
    symbol: str,
    lookback_days: Sequence[int] = (1, 5, 10, 21),
    min_reliability_tier: int = 2,
) -> tuple[dict[str, float], SentimentAggregationAudit]:
    """Aggregate timestamp-disciplined sentiment features for one row.

    Events are eligible only when their canonical event time is known, no later
    than the feature timestamp, inside the lookback window, and reliable enough.
    """

    cutoff = _utc(feature_timestamp)
    windows = [int(window) for window in lookback_days]
    features: dict[str, float] = {}
    excluded_future = 0
    excluded_reliability = 0
    missing_time = 0
    deduplicated = 0
    eligible: list[NewsEvent] = []
    seen_hashes: set[str] = set()

    for event in events:
        if event.symbol.upper() != symbol.upper():
            continue
        event_time = event.event_time
        if event_time is None:
            missing_time += 1
            continue
        if event_time > cutoff:
            excluded_future += 1
            continue
        if event.reliability_tier > min_reliability_tier:
            excluded_reliability += 1
            continue
        if event.body_hash in seen_hashes:
            deduplicated += 1
            continue
        seen_hashes.add(event.body_hash)
        eligible.append(event)

    for window in windows:
        window_seconds = window * 24 * 60 * 60
        window_events = [
            event
            for event in eligible
            if event.event_time is not None
            and 0 <= (cutoff - event.event_time).total_seconds() <= window_seconds
        ]
        scores = [
            float(event.sentiment_score)
            if event.sentiment_score is not None
            else score_headline_sentiment(event.headline)
            for event in window_events
        ]
        prefix = f"news_sentiment_{window}d"
        features[f"{prefix}_mean"] = sum(scores) / len(scores) if scores else 0.0
        features[f"{prefix}_count"] = float(len(scores))
        features[f"{prefix}_negative_shock"] = min(scores) if scores else 0.0

    audit = SentimentAggregationAudit(
        eligible_event_count=len(eligible),
        excluded_future_event_count=excluded_future,
        excluded_low_reliability_count=excluded_reliability,
        missing_event_time_count=missing_time,
        deduplicated_event_count=deduplicated,
    )
    return features, audit


def score_headline_sentiment(headline: str) -> float:
    lower = headline.lower()
    positive_terms = {"beats", "raises", "upgrade", "growth", "profit", "record"}
    negative_terms = {"misses", "cuts", "downgrade", "loss", "probe", "fraud", "warning"}
    positive = sum(1 for term in positive_terms if term in lower)
    negative = sum(1 for term in negative_terms if term in lower)
    if positive == negative:
        return 0.0
    return max(-1.0, min(1.0, (positive - negative) / max(positive + negative, 1)))


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
