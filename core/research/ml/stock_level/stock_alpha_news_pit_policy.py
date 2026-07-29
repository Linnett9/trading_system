from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping

from core.research.ml.reference.market_information_availability_authority import (
    AVAILABLE,
    MarketInformationAvailabilityAuthority,
    MarketInformationEvent,
    any_timestamp_session_policy,
)


STRICT_COLLECTED_AT = "strict_collected_at"
PROVIDER_AVAILABLE_AT = "provider_available_at"
SUPPORTED_PIT_POLICIES = (STRICT_COLLECTED_AT, PROVIDER_AVAILABLE_AT)
NEWS_AVAILABILITY_AUTHORITY = MarketInformationAvailabilityAuthority(
    session_policy=any_timestamp_session_policy("stock_alpha_news_legacy_decision_timestamp_policy_v1")
)


@dataclass(frozen=True)
class StockAlphaNewsPitPolicy:
    mode: str
    availability_lag_hours: float
    historical_provider_availability_assumed: bool

    @property
    def eligibility_timestamp_field(self) -> str:
        if self.mode == PROVIDER_AVAILABLE_AT:
            return "available_at_utc"
        return "collected_at_utc"

    @property
    def production_pit_validated(self) -> bool:
        return self.mode == STRICT_COLLECTED_AT


def resolve_stock_alpha_news_pit_policy(config: Mapping[str, Any]) -> StockAlphaNewsPitPolicy:
    ml = dict(config.get("ml", {}) or {})
    mode = str(ml.get("stock_alpha_news_pit_policy", STRICT_COLLECTED_AT)).strip() or STRICT_COLLECTED_AT
    if mode not in SUPPORTED_PIT_POLICIES:
        raise ValueError(
            "ml.stock_alpha_news_pit_policy must be one of: "
            + ", ".join(SUPPORTED_PIT_POLICIES)
        )
    lag = float(ml.get("stock_alpha_news_availability_lag_hours", 24.0))
    if lag < 0:
        raise ValueError("ml.stock_alpha_news_availability_lag_hours must be non-negative")
    return StockAlphaNewsPitPolicy(
        mode=mode,
        availability_lag_hours=lag,
        historical_provider_availability_assumed=bool(
            ml.get("stock_alpha_news_historical_provider_availability_enabled", False)
        )
        and mode == PROVIDER_AVAILABLE_AT,
    )


def pit_policy_payload(policy: StockAlphaNewsPitPolicy) -> dict[str, Any]:
    return {
        "pit_policy": policy.mode,
        "availability_lag_hours": policy.availability_lag_hours,
        "eligibility_timestamp_field": policy.eligibility_timestamp_field,
        "historical_provider_availability_assumed": policy.historical_provider_availability_assumed,
        "provider_availability_research_mode": policy.mode == PROVIDER_AVAILABLE_AT,
        "production_pit_validated": policy.production_pit_validated,
        "ingested_at_semantics": "local_collection_time_alias_collected_at_utc",
        "availability_authority_version": NEWS_AVAILABILITY_AUTHORITY.authority_version,
        "session_policy": NEWS_AVAILABILITY_AUTHORITY.session_policy.policy_id,
    }


def enrich_news_row_with_pit_timestamps(
    row: Mapping[str, Any],
    policy: StockAlphaNewsPitPolicy,
) -> dict[str, Any]:
    published = row.get("published_at_utc")
    collected = row.get("collected_at_utc") or row.get("ingested_at")
    enriched = {
        **dict(row),
        "collected_at_utc": collected,
        "available_at_utc": _available_at(published, policy),
    }
    return enriched


def article_is_pit_eligible(
    row: Mapping[str, Any],
    rebalance: datetime,
    policy: StockAlphaNewsPitPolicy,
) -> bool:
    return _article_availability_result(row, rebalance, policy)["status"] == AVAILABLE


def article_pit_exclusion_flags(
    row: Mapping[str, Any],
    rebalance: datetime,
    policy: StockAlphaNewsPitPolicy,
) -> dict[str, bool]:
    published = row.get("published_at_utc")
    available = row.get("available_at_utc")
    collected = row.get("collected_at_utc")
    eligibility_timestamp = row.get(policy.eligibility_timestamp_field)
    return {
        "published_after_rebalance": isinstance(published, datetime)
        and published > rebalance,
        "available_after_rebalance": isinstance(available, datetime)
        and available > rebalance,
        "collected_after_rebalance": isinstance(collected, datetime)
        and collected > rebalance,
        "eligibility_after_rebalance": isinstance(eligibility_timestamp, datetime)
        and eligibility_timestamp > rebalance,
    }


def _available_at(
    published: Any,
    policy: StockAlphaNewsPitPolicy,
) -> datetime | None:
    if not isinstance(published, datetime):
        return None
    if policy.mode == PROVIDER_AVAILABLE_AT:
        return published + timedelta(hours=policy.availability_lag_hours)
    return None


def _article_availability_result(
    row: Mapping[str, Any],
    rebalance: datetime,
    policy: StockAlphaNewsPitPolicy,
) -> dict[str, Any]:
    published = row.get("published_at_utc")
    collected = row.get("collected_at_utc") or row.get("ingested_at")
    available = row.get("available_at_utc") or row.get("provider_available_at_utc")
    if policy.mode == PROVIDER_AVAILABLE_AT:
        event = MarketInformationEvent(
            source_kind="news",
            decision_timestamp=rebalance,
            source_event_timestamp=published,
            provider_published_timestamp=published,
            provider_received_timestamp=available,
            source_version=str(row.get("article_id") or row.get("provider_article_id") or ""),
            required_knowledge_fields=("provider_published_timestamp", "provider_received_timestamp"),
            availability_basis_fields=("provider_published_timestamp", "provider_received_timestamp"),
        )
    else:
        event = MarketInformationEvent(
            source_kind="news",
            decision_timestamp=rebalance,
            source_event_timestamp=published,
            provider_published_timestamp=published,
            first_seen_timestamp=collected,
            ingestion_timestamp=collected,
            source_version=str(row.get("article_id") or row.get("provider_article_id") or ""),
            required_knowledge_fields=("provider_published_timestamp", "first_seen_timestamp"),
            availability_basis_fields=("provider_published_timestamp", "first_seen_timestamp"),
        )
    return NEWS_AVAILABILITY_AUTHORITY.evaluate(event)
