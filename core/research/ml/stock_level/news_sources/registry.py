from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


NEWS_SOURCE_CLASSIFICATIONS = frozenset(
    {
        "verified_rss_feed",
        "known_error_feed",
        "no_verified_official_rss",
        "sec_only_candidate",
        "sec_company_filings_candidate",
        "company_investor_relations_candidate",
        "free_source_candidate",
        "paid_provider_candidate",
        "disabled_pending_review",
    }
)

RSS_SOURCE_CLASSIFICATIONS = frozenset(
    {
        "verified_rss_feed",
        "known_error_feed",
        "no_verified_official_rss",
        "sec_only_candidate",
        "disabled_pending_review",
    }
)

NEWS_SOURCE_PLANNING_SCHEMA_VERSION = "stock_alpha_news.source_planning.v1"
NEWS_SOURCE_PLANS = {
    "alpaca_benzinga": {
        "source_category": "editorial_market_data_provider",
        "authoritative": False,
        "editorial": True,
        "intended_event_coverage": ("editorial_news",),
        "historical_support_status": "implemented_provider_adapter",
        "timestamp_semantics": "published_at_utc from provider article time; provider availability not proven historically",
        "expected_text_fields": ("headline", "summary"),
        "current_implementation_status": "adapter_available",
        "collection_enabled": False,
        "canonical_ingest_enabled": False,
        "feature_generation_enabled": False,
        "known_limitations": ("requires entitlement/API keys", "provider symbol tags may be weak"),
        "terms_or_licensing_review_status": "required_before_broader_use",
        "priority": 1,
    },
    "sec_edgar": {
        "source_category": "official_filings",
        "authoritative": True,
        "editorial": False,
        "intended_event_coverage": ("filings", "8-K", "10-Q", "10-K", "ownership"),
        "historical_support_status": "implemented_provider_adapter",
        "timestamp_semantics": "accepted datetime when supplied, otherwise filing date",
        "expected_text_fields": ("filing_metadata",),
        "current_implementation_status": "adapter_available",
        "collection_enabled": False,
        "canonical_ingest_enabled": False,
        "feature_generation_enabled": False,
        "known_limitations": ("SEC form type is not economic event classification",),
        "terms_or_licensing_review_status": "review_required",
        "priority": 2,
    },
    "company_ir_or_rss": {
        "source_category": "official_company_rss_or_ir",
        "authoritative": True,
        "editorial": False,
        "intended_event_coverage": ("press_release", "investor_relations"),
        "historical_support_status": "rss_adapter_available_registry_required",
        "timestamp_semantics": "RSS item publication timestamp when available",
        "expected_text_fields": ("title", "summary"),
        "current_implementation_status": "adapter_available",
        "collection_enabled": False,
        "canonical_ingest_enabled": False,
        "feature_generation_enabled": False,
        "known_limitations": ("feed registry coverage incomplete", "some verified feeds may error"),
        "terms_or_licensing_review_status": "source_by_source_review_required",
        "priority": 3,
    },
    "financial_modeling_prep": {
        "source_category": "paid_market_data_provider",
        "authoritative": False,
        "editorial": True,
        "intended_event_coverage": ("stock_news",),
        "historical_support_status": "adapter_available_disabled_by_default",
        "timestamp_semantics": "provider published date",
        "expected_text_fields": ("title", "text"),
        "current_implementation_status": "adapter_available",
        "collection_enabled": False,
        "canonical_ingest_enabled": False,
        "feature_generation_enabled": False,
        "known_limitations": ("paid/provider terms review required",),
        "terms_or_licensing_review_status": "required",
        "priority": 5,
    },
    "alpha_vantage": {
        "source_category": "market_data_provider_news_sentiment",
        "authoritative": False,
        "editorial": True,
        "intended_event_coverage": ("news_sentiment",),
        "historical_support_status": "adapter_available_disabled_by_default",
        "timestamp_semantics": "provider time_published",
        "expected_text_fields": ("title", "summary"),
        "current_implementation_status": "adapter_available",
        "collection_enabled": False,
        "canonical_ingest_enabled": False,
        "feature_generation_enabled": False,
        "known_limitations": ("requires key", "rate limits"),
        "terms_or_licensing_review_status": "required",
        "priority": 4,
    },
    "gdelt": {
        "source_category": "free_web_news_index",
        "authoritative": False,
        "editorial": True,
        "intended_event_coverage": ("web_news",),
        "historical_support_status": "experimental_adapter_available",
        "timestamp_semantics": "GDELT seen date, not guaranteed original publication",
        "expected_text_fields": ("title", "snippet"),
        "current_implementation_status": "experimental_adapter_available",
        "collection_enabled": False,
        "canonical_ingest_enabled": False,
        "feature_generation_enabled": False,
        "known_limitations": ("ambiguous ticker queries", "rate limiting", "weak symbol relevance"),
        "terms_or_licensing_review_status": "required",
        "priority": 6,
    },
}


def load_validated_rss_registry(
    universe_path: str | Path,
    registry_path: str | Path,
) -> tuple[list[str], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Load a static RSS registry and prove that it classifies the full universe."""
    universe_payload = _read_yaml_mapping(Path(universe_path))
    registry = _read_yaml_mapping(Path(registry_path))
    symbols = _symbols(universe_payload)
    classifications = _classifications(registry)

    missing = sorted(set(symbols) - set(classifications))
    extra = sorted(set(classifications) - set(symbols))
    if missing or extra:
        raise ValueError(
            "news source registry must classify the universe exactly; "
            f"missing={missing}, extra={extra}"
        )

    feeds: dict[str, list[dict[str, Any]]] = {}
    for symbol in symbols:
        classification = classifications[symbol]
        entry = registry.get(symbol, {})
        sources = entry.get("sources", []) if isinstance(entry, Mapping) else []
        normalised_sources = [dict(source) for source in sources or [] if isinstance(source, Mapping)]
        if classification in {"verified_rss_feed", "known_error_feed"}:
            if not normalised_sources:
                raise ValueError(f"{symbol} is {classification} but has no source")
            for source in normalised_sources:
                if not (
                    str(source.get("url", "")).strip()
                    and bool(source.get("official", False))
                    and bool(source.get("enabled", False))
                    and str(source.get("verified_source_url", "")).strip()
                ):
                    raise ValueError(f"{symbol} has an enabled RSS source without complete verification metadata")
                if classification == "known_error_feed" and not bool(source.get("known_error", False)):
                    raise ValueError(f"{symbol} known-error source is not marked known_error")
            feeds[symbol] = normalised_sources
        elif normalised_sources:
            raise ValueError(f"{symbol} has RSS sources but is classified as {classification}")

    counts = {
        status: sum(value == status for value in classifications.values())
        for status in sorted(RSS_SOURCE_CLASSIFICATIONS)
    }
    report = {
        "universe_path": str(universe_path),
        "registry_path": str(registry_path),
        "universe_symbol_count": len(symbols),
        "registry_symbol_count": len(classifications),
        "registry_missing_symbols": missing,
        "registry_extra_symbols": extra,
        "classification_counts": counts,
        "enabled_feed_symbol_count": len(feeds),
        "verified_rss_feed_symbols": sorted(
            symbol for symbol, value in classifications.items() if value == "verified_rss_feed"
        ),
        "known_error_feed_symbols": sorted(
            symbol for symbol, value in classifications.items() if value == "known_error_feed"
        ),
        "no_verified_official_rss_symbols": sorted(
            symbol for symbol, value in classifications.items() if value == "no_verified_official_rss"
        ),
        "sec_only_candidate_symbols": sorted(
            symbol for symbol, value in classifications.items() if value == "sec_only_candidate"
        ),
        "disabled_pending_review_symbols": sorted(
            symbol for symbol, value in classifications.items() if value == "disabled_pending_review"
        ),
        "registry_complete": True,
    }
    return symbols, feeds, report


def reconcile_news_source_registry(
    *,
    universe_symbols: list[str] | tuple[str, ...],
    registry_classifications: Mapping[str, str],
) -> dict[str, Any]:
    """Validate registry coverage for any provider family, not only RSS."""

    symbols = _symbol_values(universe_symbols)
    classifications = {
        str(symbol).strip().upper(): str(classification).strip()
        for symbol, classification in registry_classifications.items()
        if str(symbol).strip()
    }
    unsupported = sorted(
        {
            classification
            for classification in classifications.values()
            if classification not in NEWS_SOURCE_CLASSIFICATIONS
        }
    )
    if unsupported:
        raise ValueError("unsupported news source classification: " + ", ".join(unsupported))
    missing = sorted(set(symbols) - set(classifications))
    extra = sorted(set(classifications) - set(symbols))
    counts = {
        status: sum(value == status for value in classifications.values())
        for status in sorted(NEWS_SOURCE_CLASSIFICATIONS)
    }
    return {
        "universe_symbol_count": len(symbols),
        "registry_symbol_count": len(classifications),
        "registry_missing_symbols": missing,
        "registry_extra_symbols": extra,
        "classification_counts": counts,
        "registry_complete": not missing and not extra,
        "supports_non_rss_source_planning": True,
        "scope": "provider_independent_news_source_registry",
    }


def news_source_planning_registry() -> dict[str, dict[str, Any]]:
    """Return disabled-by-default provider planning metadata.

    This is static planning metadata only. It does not create clients, sessions,
    credentials, requests, or enabled collection configuration.
    """

    return {name: dict(plan) for name, plan in NEWS_SOURCE_PLANS.items()}


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected YAML mapping: {path}")
    return dict(payload)


def _symbols(payload: Mapping[str, Any]) -> list[str]:
    values = payload.get("symbols", [])
    symbols = _symbol_values(values)
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError("universe symbols must be non-empty and unique")
    available_count = int(payload.get("available_count", len(symbols)))
    if available_count != len(symbols):
        raise ValueError("universe available_count does not match symbols")
    return symbols


def _symbol_values(values: Any) -> list[str]:
    return [str(value).strip().upper() for value in values or [] if str(value).strip()]


def _classifications(registry: Mapping[str, Any]) -> dict[str, str]:
    grouped = registry.get("_classifications", {})
    if not isinstance(grouped, Mapping):
        raise ValueError("news source registry requires _classifications")
    result: dict[str, str] = {}
    for classification, values in grouped.items():
        if classification not in NEWS_SOURCE_CLASSIFICATIONS:
            raise ValueError(f"unsupported news source classification: {classification}")
        for value in values or []:
            symbol = str(value).strip().upper()
            if symbol in result:
                raise ValueError(f"news source registry classifies {symbol} more than once")
            result[symbol] = str(classification)
    overrides = registry.get("_classification_overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("news source registry _classification_overrides must be a mapping")
    for value, classification in overrides.items():
        symbol = str(value).strip().upper()
        if classification not in NEWS_SOURCE_CLASSIFICATIONS:
            raise ValueError(f"unsupported news source classification override: {classification}")
        if symbol not in result:
            raise ValueError(f"news source classification override is outside the registry: {symbol}")
        result[symbol] = str(classification)
    return result
