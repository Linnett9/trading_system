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
        "disabled_pending_review",
    }
)


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
        for status in sorted(NEWS_SOURCE_CLASSIFICATIONS)
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


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected YAML mapping: {path}")
    return dict(payload)


def _symbols(payload: Mapping[str, Any]) -> list[str]:
    values = payload.get("symbols", [])
    symbols = [str(value).strip().upper() for value in values or [] if str(value).strip()]
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError("universe symbols must be non-empty and unique")
    available_count = int(payload.get("available_count", len(symbols)))
    if available_count != len(symbols):
        raise ValueError("universe available_count does not match symbols")
    return symbols


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
