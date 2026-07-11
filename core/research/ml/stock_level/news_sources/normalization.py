"""Deterministic provider-independent news normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_WHITESPACE_RE = re.compile(r"\s+")
_TRACKING_QUERY_PREFIXES = ("utm_",)
_TRACKING_QUERY_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


@dataclass(frozen=True)
class TimestampValidationResult:
    raw_value: str
    parsed_at_utc: str | None
    valid: bool
    reason: str


def normalize_whitespace(value: str | None) -> str | None:
    """Collapse repeated whitespace while preserving missing values."""

    if value is None:
        return None
    return _WHITESPACE_RE.sub(" ", value).strip()


def normalize_source_name(value: str | None) -> str | None:
    """Normalize source or publisher casing deterministically."""

    cleaned = normalize_whitespace(value)
    if cleaned is None or cleaned == "":
        return cleaned
    known = {
        "alpaca": "Alpaca",
        "benzinga": "Benzinga",
        "sec": "SEC",
        "sec edgar": "SEC EDGAR",
        "edgar": "SEC EDGAR",
        "fmp": "FMP",
        "gdelt": "GDELT",
        "alpha vantage": "Alpha Vantage",
    }
    return known.get(cleaned.casefold(), cleaned.title())


def normalize_headline(value: str | None) -> str | None:
    """Normalize headlines for matching, not for display."""

    cleaned = normalize_whitespace(value)
    if cleaned is None or cleaned == "":
        return cleaned
    return cleaned.casefold()


def normalize_language(value: str | None) -> str | None:
    """Normalize safe language identifiers to lowercase ISO-like tags."""

    cleaned = normalize_whitespace(value)
    if cleaned is None or cleaned == "":
        return cleaned
    aliases = {"english": "en", "en-us": "en", "en_us": "en", "eng": "en"}
    folded = cleaned.replace("_", "-").casefold()
    return aliases.get(folded, folded)


def normalize_symbol(value: str | None) -> str | None:
    """Normalize ticker-like symbols without changing economic identity.

    The helper uppercases and trims only. It intentionally does not rewrite
    share classes, exchange suffixes, ADR markers, or delimiters.
    """

    cleaned = normalize_whitespace(value)
    if cleaned is None or cleaned == "":
        return cleaned
    return cleaned.upper()


def normalize_url(value: str | None) -> str | None:
    """Normalize URL structure while preserving the original elsewhere."""

    cleaned = normalize_whitespace(value)
    if cleaned is None or cleaned == "":
        return cleaned
    parts = urlsplit(cleaned)
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    if (scheme == "http" and netloc.endswith(":80")) or (scheme == "https" and netloc.endswith(":443")):
        netloc = netloc.rsplit(":", 1)[0]
    query_pairs = sorted(
        (key, val)
        for key, val in parse_qsl(parts.query, keep_blank_values=True)
        if key not in _TRACKING_QUERY_KEYS and not key.startswith(_TRACKING_QUERY_PREFIXES)
    )
    query = urlencode(query_pairs, doseq=True)
    path = parts.path or "/"
    return urlunsplit((scheme, netloc, path, query, ""))


def normalize_url_pair(value: str | None) -> tuple[str | None, str | None]:
    """Return ``(original_url, normalized_url)`` for derived artifacts."""

    return value, normalize_url(value)


def parse_utc_timestamp(value: str | datetime | None) -> datetime | None:
    """Parse and validate a timestamp as timezone-aware UTC.

    Unknown provider availability must remain ``None``; callers should not pass
    collection timestamps as substitutes.
    """

    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        cleaned = normalize_whitespace(value)
        if cleaned is None or cleaned == "":
            return None
        parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include timezone information")
    return parsed.astimezone(timezone.utc)


def format_utc_timestamp(value: datetime | None) -> str | None:
    """Format a timezone-aware datetime as an ISO UTC string."""

    if value is None:
        return None
    return parse_utc_timestamp(value).isoformat().replace("+00:00", "Z")


def validate_utc_timestamp(value: str | datetime | None) -> TimestampValidationResult:
    """Validate a timestamp without silently treating invalid values as UTC."""

    raw = "" if value is None else str(value)
    if value is None or normalize_whitespace(raw) == "":
        return TimestampValidationResult(raw, None, False, "missing")
    try:
        parsed = parse_utc_timestamp(value)
    except ValueError as exc:
        return TimestampValidationResult(raw, None, False, str(exc))
    return TimestampValidationResult(raw, format_utc_timestamp(parsed), True, "")
