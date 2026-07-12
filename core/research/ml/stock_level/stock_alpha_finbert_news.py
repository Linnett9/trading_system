from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Mapping, Protocol, Sequence

from core.research.framework.data import CsvRowRepository
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.artifacts.artifact_writers import MLCoreArtifactWriter
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir


FINBERT_INFERENCE_CONTRACT_VERSION = "stock_alpha_finbert_article_inference.v1"
FINBERT_TEXT_SELECTION_CONTRACT_VERSION = "stock_alpha_finbert_text_selection.v1"
FINBERT_STOCK_FEATURE_CONTRACT_VERSION = "stock_alpha_finbert_stock_features.v1"
FINBERT_EXPOSURE_FEATURE_CONTRACT_VERSION = "stock_alpha_finbert_exposure_features.v1"
FINBERT_SELECTOR_JOIN_CONTRACT_VERSION = "stock_alpha_finbert_selector_join.v1"

ARTICLE_LEVEL_FIELDS = (
    "article_id",
    "symbol",
    "provider",
    "source",
    "headline",
    "selected_text_hash",
    "selected_text_source_fields",
    "published_at",
    "first_seen_at",
    "ingested_at",
    "news_available_timestamp",
    "news_available_timestamp_source",
    "finbert_model_id",
    "finbert_model_revision",
    "tokenizer_id",
    "tokenizer_revision",
    "inference_contract_version",
    "text_selection_contract_version",
    "max_token_length",
    "truncation_applied",
    "positive_probability",
    "neutral_probability",
    "negative_probability",
    "sentiment_label",
    "signed_sentiment_score",
    "confidence",
    "inference_device",
    "inference_batch_identity",
    "scored_at",
)

STOCK_NEWS_FEATURE_COLUMNS = (
    "finbert_news_feature_contract_version",
    "finbert_model_id",
    "decision_timestamp",
    "symbol",
    "finbert_lookback_days",
    "finbert_lookback_start",
    "finbert_lookback_end",
    "finbert_latest_news_available_timestamp",
    "finbert_article_count",
    "finbert_scored_article_count",
    "finbert_news_coverage_indicator",
    "finbert_input_article_set_hash",
    "finbert_mean_signed_sentiment",
    "finbert_median_signed_sentiment",
    "finbert_min_signed_sentiment",
    "finbert_max_signed_sentiment",
    "finbert_negative_probability_mean",
    "finbert_positive_probability_mean",
    "finbert_high_confidence_negative_count",
    "finbert_high_confidence_positive_count",
    "finbert_sentiment_stddev",
    "finbert_negative_news_breadth",
    "finbert_sentiment_agreement",
    "finbert_sentiment_disagreement",
    "finbert_news_freshness_days",
    "finbert_source_count",
    "finbert_source_diversity",
    "contrarian_adverse_news_risk",
    "contrarian_negative_news_after_drawdown",
    "contrarian_sentiment_price_disagreement",
    "contrarian_potential_overreaction",
    "contrarian_potential_sentiment_reversal",
)

EXPOSURE_NEWS_FEATURE_COLUMNS = (
    "finbert_exposure_feature_contract_version",
    "feature_date",
    "finbert_lookback_days",
    "portfolio_finbert_weighted_mean_sentiment",
    "portfolio_finbert_weighted_negative_probability",
    "portfolio_finbert_negative_news_breadth",
    "portfolio_weight_with_recent_news",
    "portfolio_weight_with_high_confidence_negative_news",
    "portfolio_max_single_stock_adverse_news_concentration",
    "portfolio_finbert_source_diversity",
    "portfolio_finbert_sentiment_disagreement",
    "portfolio_largest_negative_sentiment_holding_weight",
    "portfolio_finbert_covered_holding_count",
    "portfolio_finbert_total_holding_count",
    "portfolio_finbert_input_set_hash",
)


@dataclass(frozen=True)
class TextSelection:
    text: str
    text_hash: str
    source_fields: tuple[str, ...]
    truncation_applied: bool


@dataclass(frozen=True)
class AvailabilityResolution:
    timestamp: str
    source: str
    same_decision_day_ambiguous: bool = False


@dataclass(frozen=True)
class FinBertPrediction:
    positive_probability: float
    neutral_probability: float
    negative_probability: float
    sentiment_label: str


@dataclass(frozen=True)
class FinBertModelIdentity:
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    inference_device: str


class FinBertAdapter(Protocol):
    @property
    def identity(self) -> FinBertModelIdentity:
        ...

    def score_batch(self, texts: Sequence[str]) -> list[FinBertPrediction]:
        ...


@dataclass(frozen=True)
class FinBertScoringPaths:
    scored_articles_csv_path: Path
    audit_json_path: Path
    audit_markdown_path: Path
    chunk_manifest_csv_path: Path
    rejected_articles_csv_path: Path | None


@dataclass(frozen=True)
class FinBertSmokePaths:
    output_dir: Path
    scoring: FinBertScoringPaths
    stock_features_csv_path: Path
    stock_features_audit_json_path: Path
    selector_join_audit_json_path: Path
    exposure_features_csv_path: Path
    exposure_audit_json_path: Path
    summary_json_path: Path
    summary_markdown_path: Path


class DeterministicFinBertFixtureAdapter:
    """Small deterministic scorer for unit tests and offline smoke fixtures."""

    def __init__(self, *, model_id: str = "deterministic-finbert-fixture") -> None:
        self._identity = FinBertModelIdentity(
            model_id=model_id,
            model_revision="fixture-v1",
            tokenizer_id=model_id,
            tokenizer_revision="fixture-v1",
            inference_device="cpu",
        )

    @property
    def identity(self) -> FinBertModelIdentity:
        return self._identity

    def score_batch(self, texts: Sequence[str]) -> list[FinBertPrediction]:
        return [_fixture_prediction(text) for text in texts]


class HuggingFaceFinBertAdapter:
    """Local-files-only HuggingFace FinBERT adapter.

    This adapter intentionally does not download model assets unless the caller
    explicitly sets ``local_files_only`` false in configuration.
    """

    def __init__(
        self,
        *,
        model_id: str,
        tokenizer_id: str | None = None,
        model_revision: str = "main",
        tokenizer_revision: str | None = None,
        device: str = "cpu",
        max_token_length: int = 256,
        local_files_only: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("FinBERT inference requires torch and transformers") from exc

        tokenizer_id = tokenizer_id or model_id
        tokenizer_revision = tokenizer_revision or model_revision
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("FinBERT CUDA requested but CUDA is unavailable")
        resolved_device = "cuda" if device == "cuda" else "cpu"
        self._torch = torch
        self._max_token_length = max_token_length
        self._identity = FinBertModelIdentity(
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            inference_device=resolved_device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            revision=tokenizer_revision,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            revision=model_revision,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        ).to(resolved_device)
        self._model.eval()

    @property
    def identity(self) -> FinBertModelIdentity:
        return self._identity

    def score_batch(self, texts: Sequence[str]) -> list[FinBertPrediction]:
        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self._max_token_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._identity.inference_device) for key, value in encoded.items()}
        with self._torch.no_grad():
            logits = self._model(**encoded).logits
            probabilities = self._torch.softmax(logits, dim=-1).detach().cpu().tolist()
        labels = [self._label_for_index(index) for index in range(len(probabilities[0]))]
        output = []
        for row in probabilities:
            values = {label: float(value) for label, value in zip(labels, row)}
            output.append(
                FinBertPrediction(
                    positive_probability=values.get("positive", 0.0),
                    neutral_probability=values.get("neutral", 0.0),
                    negative_probability=values.get("negative", 0.0),
                    sentiment_label=max(values, key=values.get),
                )
            )
        return output

    def _label_for_index(self, index: int) -> str:
        config = getattr(self._model, "config", None)
        raw = getattr(config, "id2label", {}).get(index, str(index)).lower()
        if "pos" in raw:
            return "positive"
        if "neg" in raw:
            return "negative"
        return "neutral"


def select_article_text(row: Mapping[str, Any], *, max_characters: int = 10_000) -> TextSelection:
    candidates = (
        ("headline", "body_or_full_text"),
        ("headline", "body"),
        ("headline", "summary"),
        ("headline", "body_or_summary"),
        ("headline",),
        ("summary",),
        ("body_or_summary",),
        ("body_or_full_text",),
        ("body",),
    )
    for fields in candidates:
        parts = [_clean_text(row.get(field)) for field in fields]
        parts = [part for part in parts if part]
        if not parts:
            continue
        if len(parts) == 2 and _same_or_repeated(parts[0], parts[1]):
            parts = [parts[1]]
        text = _normalise_whitespace(" ".join(parts))
        if not text:
            continue
        truncated = len(text) > max_characters
        if truncated:
            text = text[:max_characters].rstrip()
        return TextSelection(
            text=text,
            text_hash=_sha256_text(text),
            source_fields=fields,
            truncation_applied=truncated,
        )
    raise ValueError("empty selected article text")


def resolve_news_available_timestamp(row: Mapping[str, Any]) -> AvailabilityResolution:
    candidates = (
        ("provider_available_at_utc", False),
        ("available_at_utc", False),
        ("first_seen_at", False),
        ("collected_at_utc", False),
        ("ingested_at", False),
        ("published_at_utc", True),
        ("published_at", True),
    )
    parsed: dict[str, tuple[datetime, bool]] = {}
    failures = []
    for field, publication_like in candidates:
        raw = _text(row.get(field))
        if not raw:
            continue
        try:
            parsed[field] = (_parse_timestamp(raw), _date_only(raw) and publication_like)
        except ValueError as exc:
            failures.append(f"{field}:{exc}")
    for field in ("provider_available_at_utc", "available_at_utc", "first_seen_at", "collected_at_utc", "ingested_at"):
        if field in parsed:
            timestamp, _ambiguous = parsed[field]
            return AvailabilityResolution(_format_timestamp(timestamp), field)
    for field in ("published_at_utc", "published_at"):
        if field in parsed:
            timestamp, ambiguous = parsed[field]
            if ambiguous:
                raise ValueError("date-only publication timestamp is ambiguous without first-seen or ingestion fallback")
            return AvailabilityResolution(_format_timestamp(timestamp), field)
    raise ValueError("missing valid news availability timestamp" + (f"; parse_failures={failures}" if failures else ""))


def validate_finbert_prediction(prediction: FinBertPrediction) -> dict[str, float]:
    values = {
        "positive": float(prediction.positive_probability),
        "neutral": float(prediction.neutral_probability),
        "negative": float(prediction.negative_probability),
    }
    for label, value in values.items():
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"invalid FinBERT probability: {label}={value}")
    if not math.isclose(sum(values.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("FinBERT probabilities must sum to 1")
    expected = max(values, key=values.get)
    if prediction.sentiment_label != expected:
        raise ValueError("FinBERT sentiment label must match maximum probability")
    score = values["positive"] - values["negative"]
    return {
        "positive_probability": values["positive"],
        "neutral_probability": values["neutral"],
        "negative_probability": values["negative"],
        "signed_sentiment_score": score,
        "confidence": values[expected],
    }


def score_finbert_articles(
    rows: Sequence[Mapping[str, Any]],
    *,
    adapter: FinBertAdapter,
    output_dir: Path,
    config: Mapping[str, Any],
    max_token_length: int = 256,
    max_characters: int = 10_000,
    batch_size: int = 8,
    scored_at: str | None = None,
) -> FinBertScoringPaths:
    writer = ResearchArtifactWriter()
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    scored_at = scored_at or _format_timestamp(datetime.now(timezone.utc))
    valid_items: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_articles: dict[str, dict[str, Any]] = {}
    availability_counts = {
        "valid": 0,
        "missing": 0,
        "publication_timestamp": 0,
        "first_seen_timestamp": 0,
        "ingestion_fallback": 0,
        "same_decision_day_ambiguous": 0,
        "timezone_normalisation_failures": 0,
    }
    for index, row in enumerate(rows, start=1):
        article_id = _article_id(row)
        try:
            if not article_id:
                raise ValueError("missing article_id")
            symbol = _symbol(row)
            if not symbol:
                raise ValueError("missing symbol")
            text = select_article_text(row, max_characters=max_characters)
            availability = resolve_news_available_timestamp(row)
            availability_counts["valid"] += 1
            if availability.source in {"published_at_utc", "published_at"}:
                availability_counts["publication_timestamp"] += 1
            elif availability.source == "first_seen_at":
                availability_counts["first_seen_timestamp"] += 1
            elif availability.source in {"ingested_at", "collected_at_utc"}:
                availability_counts["ingestion_fallback"] += 1
            duplicate_key = f"{article_id}|{symbol}"
            duplicate_identity = {
                "selected_text_hash": text.text_hash,
                "news_available_timestamp": availability.timestamp,
            }
            existing = seen_articles.get(duplicate_key)
            if existing and existing != duplicate_identity:
                raise ValueError("conflicting duplicate article identity")
            seen_articles[duplicate_key] = duplicate_identity
            valid_items.append(
                {
                    "source_row_number": index,
                    "source_row": dict(row),
                    "article_id": article_id,
                    "symbol": symbol,
                    "text": text,
                    "availability": availability,
                }
            )
        except Exception as exc:
            if "timestamp" in str(exc).lower():
                availability_counts["missing"] += 1
            if "timezone" in str(exc).lower():
                availability_counts["timezone_normalisation_failures"] += 1
            rejected.append(_rejected_row(row, index, type(exc).__name__, str(exc)))

    scored: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    config_hash = MLCoreArtifactWriter.hash_payload(config)
    for chunk_index, start in enumerate(range(0, len(valid_items), batch_size), start=1):
        chunk_items = valid_items[start : start + batch_size]
        chunk_identity = _chunk_identity(chunk_items, adapter.identity, max_token_length, config_hash)
        chunk_path = chunk_dir / f"{chunk_identity['chunk_id']}.json"
        reused = False
        chunk_payload = _read_completed_chunk(chunk_path, chunk_identity)
        if chunk_payload is None:
            predictions = adapter.score_batch([item["text"].text for item in chunk_items])
            scored_rows = [
                _scored_row(
                    item,
                    prediction,
                    adapter.identity,
                    max_token_length=max_token_length,
                    chunk_id=chunk_identity["chunk_id"],
                    scored_at=scored_at,
                )
                for item, prediction in zip(chunk_items, predictions)
            ]
            chunk_payload = {"status": "completed", "identity": chunk_identity, "rows": scored_rows}
            _write_json_atomic(chunk_path, chunk_payload)
        else:
            reused = True
        scored.extend(chunk_payload["rows"])
        chunk_rows.append(
            {
                "chunk_id": chunk_identity["chunk_id"],
                "status": "completed",
                "article_count": len(chunk_items),
                "reused": str(reused).lower(),
                "chunk_path": str(chunk_path),
            }
        )

    scored_path = output_dir / "finbert_scored_articles.csv"
    writer.write_csv(scored_path, scored, fieldnames=ARTICLE_LEVEL_FIELDS, extrasaction="ignore")
    chunk_manifest = output_dir / "finbert_chunk_manifest.csv"
    writer.write_csv(
        chunk_manifest,
        chunk_rows,
        fieldnames=("chunk_id", "status", "article_count", "reused", "chunk_path"),
    )
    rejection_path = output_dir / "finbert_rejected_articles.csv" if rejected else None
    if rejection_path:
        writer.write_csv(
            rejection_path,
            rejected,
            fieldnames=("source_row_number", "article_id", "symbol", "error_type", "error_message"),
        )
    audit = _article_audit(rows, scored, rejected, chunk_rows, adapter.identity, availability_counts)
    audit_path = output_dir / "finbert_scoring_audit.json"
    writer.write_json(audit_path, audit)
    markdown_path = output_dir / "finbert_scoring_audit.md"
    writer.write_markdown(markdown_path, _article_audit_markdown(audit))
    return FinBertScoringPaths(scored_path, audit_path, markdown_path, chunk_manifest, rejection_path)


def build_stock_finbert_features(
    scored_articles: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    *,
    lookback_days: Sequence[int] = (1, 3, 7),
    high_confidence_threshold: float = 0.75,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    articles = [_normalise_scored_article(row) for row in scored_articles]
    feature_rows: list[dict[str, Any]] = []
    future_exclusions = 0
    for decision in decision_rows:
        symbol = _symbol(decision)
        decision_time = _parse_timestamp(_decision_timestamp(decision))
        recent_return = _number(
            decision.get("price_return_20d")
            or decision.get("predicted_momentum_20d")
            or decision.get("recent_price_return")
        ) or 0.0
        for days in lookback_days:
            window_start = decision_time.timestamp() - (days * 86400)
            eligible = []
            for article in articles:
                if article["symbol"] != symbol:
                    continue
                available = _parse_timestamp(article["news_available_timestamp"])
                if available > decision_time:
                    future_exclusions += 1
                    continue
                if available.timestamp() >= window_start:
                    eligible.append(article)
            feature_rows.append(
                _stock_feature_row(
                    symbol=symbol,
                    decision_time=decision_time,
                    lookback_days=days,
                    articles=eligible,
                    recent_return=recent_return,
                    high_confidence_threshold=high_confidence_threshold,
                )
            )
    audit = {
        "feature_contract_version": FINBERT_STOCK_FEATURE_CONTRACT_VERSION,
        "decision_row_count": len(decision_rows),
        "feature_row_count": len(feature_rows),
        "lookback_days": list(lookback_days),
        "future_article_exclusion_count": future_exclusions,
        "duplicate_join_key_count": _duplicate_key_count(feature_rows, ("symbol", "decision_timestamp", "finbert_lookback_days")),
        "rows_with_any_eligible_news": sum(int(row["finbert_article_count"]) > 0 for row in feature_rows),
        "rows_with_no_eligible_news": sum(int(row["finbert_article_count"]) == 0 for row in feature_rows),
        "coverage_by_symbol": _coverage_by(feature_rows, "symbol"),
        "coverage_by_year": _coverage_by_year(feature_rows, "decision_timestamp"),
        "coverage_by_decision_date": _coverage_by(feature_rows, "decision_timestamp"),
    }
    return feature_rows, audit


def join_finbert_features_for_selector(
    rows: Sequence[Mapping[str, Any]],
    stock_feature_rows: Sequence[Mapping[str, Any]],
    *,
    include_news: bool = False,
    missing_policy: str = "zero_with_indicators",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not include_news:
        return [dict(row) for row in rows], {
            "contract_version": FINBERT_SELECTOR_JOIN_CONTRACT_VERSION,
            "include_news": False,
            "row_count": len(rows),
            "model_input_identity": "price_only",
        }
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    duplicate_join_keys = 0
    for feature in stock_feature_rows:
        key = (str(feature.get("decision_timestamp", ""))[:10], _symbol(feature))
        if key in by_key:
            duplicate_join_keys += 1
        by_key[key] = feature
    joined = []
    rows_with_news = 0
    future_exclusions = 0
    for row in rows:
        date_key = _decision_timestamp(row)[:10]
        key = (date_key, _symbol(row))
        feature = by_key.get(key)
        output = dict(row)
        if feature is None:
            if missing_policy != "zero_with_indicators":
                raise ValueError(f"missing FinBERT news feature for selector row: {key}")
            output.update(_empty_stock_feature_values(date_key, _symbol(row)))
        else:
            if str(feature.get("decision_timestamp", ""))[:10] > date_key:
                future_exclusions += 1
                output.update(_empty_stock_feature_values(date_key, _symbol(row)))
            else:
                rows_with_news += int(float(feature.get("finbert_news_coverage_indicator", 0)) > 0)
                output.update({column: feature.get(column, "") for column in STOCK_NEWS_FEATURE_COLUMNS})
        joined.append(output)
    identity = MLCoreArtifactWriter.hash_payload(
        {
            "contract": FINBERT_SELECTOR_JOIN_CONTRACT_VERSION,
            "include_news": True,
            "missing_policy": missing_policy,
            "feature_rows_hash": _stable_rows_hash(stock_feature_rows),
        }
    )
    audit = {
        "contract_version": FINBERT_SELECTOR_JOIN_CONTRACT_VERSION,
        "include_news": True,
        "selector_rows": len(rows),
        "selector_rows_with_any_eligible_news": rows_with_news,
        "selector_rows_with_no_eligible_news": len(rows) - rows_with_news,
        "duplicate_join_keys": duplicate_join_keys,
        "future_news_exclusions": future_exclusions,
        "coverage_by_symbol": _coverage_by(joined, "symbol", indicator="finbert_news_coverage_indicator"),
        "coverage_by_year": _coverage_by_year(joined, "rebalance_date", indicator="finbert_news_coverage_indicator"),
        "model_input_identity": identity,
    }
    return joined, audit


def build_exposure_finbert_features(
    holdings_rows: Sequence[Mapping[str, Any]],
    stock_feature_rows: Sequence[Mapping[str, Any]],
    *,
    include_news: bool = False,
    lookback_days: int = 7,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not include_news:
        return [dict(row) for row in holdings_rows], {
            "contract_version": FINBERT_EXPOSURE_FEATURE_CONTRACT_VERSION,
            "include_news": False,
            "portfolio_decisions": len({row.get("rebalance_date") for row in holdings_rows}),
            "model_input_identity": "price_only",
        }
    feature_by_key = {
        (str(row.get("decision_timestamp", ""))[:10], _symbol(row), int(row.get("finbert_lookback_days") or 0)): row
        for row in stock_feature_rows
    }
    by_date: dict[str, list[Mapping[str, Any]]] = {}
    for row in holdings_rows:
        by_date.setdefault(str(row.get("rebalance_date") or row.get("feature_date")), []).append(row)
    output = []
    mismatches = 0
    for feature_date, holdings in sorted(by_date.items()):
        output.append(_exposure_feature_row(feature_date, holdings, feature_by_key, lookback_days, mismatches_ref=[mismatches]))
        mismatches += output[-1].pop("_mismatches")
    identity = MLCoreArtifactWriter.hash_payload(
        {
            "contract": FINBERT_EXPOSURE_FEATURE_CONTRACT_VERSION,
            "lookback_days": lookback_days,
            "stock_feature_rows_hash": _stable_rows_hash(stock_feature_rows),
            "holdings_hash": _stable_rows_hash(holdings_rows),
        }
    )
    for row in output:
        row["portfolio_finbert_input_set_hash"] = identity
    audit = {
        "contract_version": FINBERT_EXPOSURE_FEATURE_CONTRACT_VERSION,
        "include_news": True,
        "portfolio_decisions": len(output),
        "portfolio_decisions_with_any_covered_holding": sum(row["portfolio_finbert_covered_holding_count"] > 0 for row in output),
        "average_portfolio_weight_covered_by_news": mean([row["portfolio_weight_with_recent_news"] for row in output]) if output else 0.0,
        "holdings_news_key_mismatches": mismatches,
        "coverage_by_year": _coverage_by_year(output, "feature_date", indicator="portfolio_weight_with_recent_news"),
        "model_input_identity": identity,
    }
    return output, audit


def write_finbert_news_probe(config: Mapping[str, Any]) -> FinBertSmokePaths:
    ml = dict(config.get("ml", {}) or {})
    settings = dict(ml.get("stock_alpha_finbert_news", {}) or {})
    output = Path(settings.get("output_dir") or stock_alpha_output_dir(config) / "finbert_news_probe")
    output.mkdir(parents=True, exist_ok=True)
    adapter_kind = str(settings.get("adapter", "fixture")).lower()
    max_token_length = int(settings.get("max_token_length", 128))
    batch_size = int(settings.get("batch_size", 3))
    adapter: FinBertAdapter
    if adapter_kind == "fixture":
        adapter = DeterministicFinBertFixtureAdapter()
    elif adapter_kind == "huggingface":
        adapter = HuggingFaceFinBertAdapter(
            model_id=str(settings.get("model_id", "ProsusAI/finbert")),
            tokenizer_id=str(settings.get("tokenizer_id", settings.get("model_id", "ProsusAI/finbert"))),
            model_revision=str(settings.get("model_revision", "main")),
            tokenizer_revision=str(settings.get("tokenizer_revision", settings.get("model_revision", "main"))),
            device=str(settings.get("device", "cpu")),
            max_token_length=max_token_length,
            local_files_only=bool(settings.get("local_files_only", True)),
            cache_dir=settings.get("cache_dir"),
        )
    else:
        raise ValueError("stock_alpha_finbert_news.adapter must be fixture or huggingface")
    articles = _fixture_articles()
    scoring = score_finbert_articles(
        articles,
        adapter=adapter,
        output_dir=output / "article_scoring",
        config=config,
        max_token_length=max_token_length,
        batch_size=batch_size,
    )
    scored = CsvRowRepository().read(scoring.scored_articles_csv_path)
    decisions = _fixture_decision_rows()
    stock_features, stock_audit = build_stock_finbert_features(scored, decisions, lookback_days=(1, 3, 7))
    writer = ResearchArtifactWriter()
    stock_features_path = output / "stock_finbert_features.csv"
    writer.write_csv(stock_features_path, stock_features, fieldnames=STOCK_NEWS_FEATURE_COLUMNS)
    stock_audit_path = output / "stock_finbert_features_audit.json"
    writer.write_json(stock_audit_path, stock_audit)
    joined, selector_audit = join_finbert_features_for_selector(decisions, [r for r in stock_features if int(r["finbert_lookback_days"]) == 7], include_news=True)
    selector_audit_path = output / "selector_news_alignment_audit.json"
    writer.write_json(selector_audit_path, selector_audit)
    exposure_features, exposure_audit = build_exposure_finbert_features(_fixture_holdings_rows(), stock_features, include_news=True, lookback_days=7)
    exposure_path = output / "exposure_finbert_features.csv"
    writer.write_csv(exposure_path, exposure_features, fieldnames=EXPOSURE_NEWS_FEATURE_COLUMNS)
    exposure_audit_path = output / "exposure_news_alignment_audit.json"
    writer.write_json(exposure_audit_path, exposure_audit)
    scoring_audit = json.loads(scoring.audit_json_path.read_text(encoding="utf-8"))
    status = "FINBERT_PROBE_PASSED"
    summary = {
        "status": status,
        "articles": len(articles),
        "symbols": len({row["symbol"] for row in articles if row.get("symbol")}),
        "date_range": ["2024-01-02T10:00:00+00:00", "2024-01-09T14:00:00+00:00"],
        "successful_scores": scoring_audit["successfully_scored_articles"],
        "rejections": scoring_audit["failed_articles"],
        "future_exclusions": stock_audit["future_article_exclusion_count"],
        "selector_joined_rows": len(joined),
        "exposure_aggregated_rows": len(exposure_features),
        "model_identity": adapter.identity.__dict__,
        "article_scoring_audit": str(scoring.audit_json_path),
        "stock_feature_audit": str(stock_audit_path),
        "selector_alignment_audit": str(selector_audit_path),
        "exposure_alignment_audit": str(exposure_audit_path),
    }
    summary_path = output / "ticket_9a_finbert_news_probe_summary.json"
    writer.write_json(summary_path, summary)
    markdown_path = output / "ticket_9a_finbert_news_probe_summary.md"
    writer.write_markdown(markdown_path, _smoke_markdown(summary))
    return FinBertSmokePaths(
        output,
        scoring,
        stock_features_path,
        stock_audit_path,
        selector_audit_path,
        exposure_path,
        exposure_audit_path,
        summary_path,
        markdown_path,
    )


def _scored_row(
    item: Mapping[str, Any],
    prediction: FinBertPrediction,
    identity: FinBertModelIdentity,
    *,
    max_token_length: int,
    chunk_id: str,
    scored_at: str,
) -> dict[str, Any]:
    values = validate_finbert_prediction(prediction)
    row = item["source_row"]
    text: TextSelection = item["text"]
    availability: AvailabilityResolution = item["availability"]
    return {
        "article_id": item["article_id"],
        "symbol": item["symbol"],
        "provider": _text(row.get("provider") or row.get("delivery_provider")),
        "source": _text(row.get("source") or row.get("publisher")),
        "headline": _text(row.get("headline") or row.get("title")),
        "selected_text_hash": text.text_hash,
        "selected_text_source_fields": ",".join(text.source_fields),
        "published_at": _text(row.get("published_at_utc") or row.get("published_at")),
        "first_seen_at": _text(row.get("first_seen_at") or row.get("provider_available_at_utc")),
        "ingested_at": _text(row.get("ingested_at") or row.get("collected_at_utc")),
        "news_available_timestamp": availability.timestamp,
        "news_available_timestamp_source": availability.source,
        "finbert_model_id": identity.model_id,
        "finbert_model_revision": identity.model_revision,
        "tokenizer_id": identity.tokenizer_id,
        "tokenizer_revision": identity.tokenizer_revision,
        "inference_contract_version": FINBERT_INFERENCE_CONTRACT_VERSION,
        "text_selection_contract_version": FINBERT_TEXT_SELECTION_CONTRACT_VERSION,
        "max_token_length": max_token_length,
        "truncation_applied": str(text.truncation_applied).lower(),
        "positive_probability": values["positive_probability"],
        "neutral_probability": values["neutral_probability"],
        "negative_probability": values["negative_probability"],
        "sentiment_label": prediction.sentiment_label,
        "signed_sentiment_score": values["signed_sentiment_score"],
        "confidence": values["confidence"],
        "inference_device": identity.inference_device,
        "inference_batch_identity": chunk_id,
        "scored_at": scored_at,
    }


def _stock_feature_row(
    *,
    symbol: str,
    decision_time: datetime,
    lookback_days: int,
    articles: list[Mapping[str, Any]],
    recent_return: float,
    high_confidence_threshold: float,
) -> dict[str, Any]:
    sentiments = [_number(row.get("signed_sentiment_score")) for row in articles]
    sentiments = [value for value in sentiments if value is not None]
    negative_probs = [_number(row.get("negative_probability")) for row in articles]
    negative_probs = [value for value in negative_probs if value is not None]
    positive_probs = [_number(row.get("positive_probability")) for row in articles]
    positive_probs = [value for value in positive_probs if value is not None]
    high_neg = [row for row in articles if _number(row.get("negative_probability")) is not None and _number(row.get("negative_probability")) >= high_confidence_threshold]
    high_pos = [row for row in articles if _number(row.get("positive_probability")) is not None and _number(row.get("positive_probability")) >= high_confidence_threshold]
    latest = max((_parse_timestamp(row["news_available_timestamp"]) for row in articles), default=None)
    source_count = len({_text(row.get("source") or row.get("provider")) for row in articles if _text(row.get("source") or row.get("provider"))})
    article_count = len(articles)
    negative_breadth = len([value for value in sentiments if value < 0.0]) / article_count if article_count else 0.0
    sentiment_std = pstdev(sentiments) if len(sentiments) > 1 else 0.0
    agreement = 1.0 - min(1.0, sentiment_std) if article_count else 0.0
    adverse_news_risk = negative_breadth * abs(min(sentiments, default=0.0))
    negative_after_drawdown = adverse_news_risk * max(0.0, -recent_return)
    disagreement = (mean(sentiments) if sentiments else 0.0) - recent_return
    potential_overreaction = max(0.0, -mean(sentiments)) * max(0.0, -recent_return) if sentiments else 0.0
    potential_reversal = max(0.0, mean(sentiments)) * max(0.0, -recent_return) if sentiments else 0.0
    decision = _format_timestamp(decision_time)
    lookback_start = _format_timestamp(datetime.fromtimestamp(decision_time.timestamp() - lookback_days * 86400, tz=timezone.utc))
    return {
        "finbert_news_feature_contract_version": FINBERT_STOCK_FEATURE_CONTRACT_VERSION,
        "finbert_model_id": _text(articles[0].get("finbert_model_id")) if articles else "",
        "decision_timestamp": decision,
        "symbol": symbol,
        "finbert_lookback_days": lookback_days,
        "finbert_lookback_start": lookback_start,
        "finbert_lookback_end": decision,
        "finbert_latest_news_available_timestamp": _format_timestamp(latest) if latest else "",
        "finbert_article_count": article_count,
        "finbert_scored_article_count": article_count,
        "finbert_news_coverage_indicator": int(article_count > 0),
        "finbert_input_article_set_hash": _article_set_hash(articles),
        "finbert_mean_signed_sentiment": mean(sentiments) if sentiments else 0.0,
        "finbert_median_signed_sentiment": median(sentiments) if sentiments else 0.0,
        "finbert_min_signed_sentiment": min(sentiments) if sentiments else 0.0,
        "finbert_max_signed_sentiment": max(sentiments) if sentiments else 0.0,
        "finbert_negative_probability_mean": mean(negative_probs) if negative_probs else 0.0,
        "finbert_positive_probability_mean": mean(positive_probs) if positive_probs else 0.0,
        "finbert_high_confidence_negative_count": len(high_neg),
        "finbert_high_confidence_positive_count": len(high_pos),
        "finbert_sentiment_stddev": sentiment_std,
        "finbert_negative_news_breadth": negative_breadth,
        "finbert_sentiment_agreement": agreement,
        "finbert_sentiment_disagreement": sentiment_std,
        "finbert_news_freshness_days": ((decision_time - latest).total_seconds() / 86400.0) if latest else "",
        "finbert_source_count": source_count,
        "finbert_source_diversity": source_count / article_count if article_count else 0.0,
        "contrarian_adverse_news_risk": adverse_news_risk,
        "contrarian_negative_news_after_drawdown": negative_after_drawdown,
        "contrarian_sentiment_price_disagreement": disagreement,
        "contrarian_potential_overreaction": potential_overreaction,
        "contrarian_potential_sentiment_reversal": potential_reversal,
    }


def _exposure_feature_row(
    feature_date: str,
    holdings: Sequence[Mapping[str, Any]],
    feature_by_key: Mapping[tuple[str, str, int], Mapping[str, Any]],
    lookback_days: int,
    *,
    mismatches_ref: list[int],
) -> dict[str, Any]:
    gross = sum(abs(_number(row.get("weight")) or 0.0) for row in holdings) or 1.0
    weighted_sentiment = 0.0
    weighted_negative = 0.0
    weight_with_news = 0.0
    weight_high_neg = 0.0
    max_concentration = 0.0
    largest_negative_weight = 0.0
    covered = 0
    sources: set[str] = set()
    disagreements = []
    mismatches = 0
    for holding in holdings:
        symbol = _symbol(holding)
        weight = abs(_number(holding.get("weight")) or 0.0) / gross
        feature = feature_by_key.get((feature_date, symbol, lookback_days))
        if feature is None:
            mismatches += 1
            continue
        coverage = float(feature.get("finbert_news_coverage_indicator") or 0.0)
        sentiment = float(feature.get("finbert_mean_signed_sentiment") or 0.0)
        negative = float(feature.get("finbert_negative_probability_mean") or 0.0)
        high_neg = float(feature.get("finbert_high_confidence_negative_count") or 0.0)
        weighted_sentiment += weight * sentiment
        weighted_negative += weight * negative
        if coverage > 0:
            covered += 1
            weight_with_news += weight
            sources.add(str(feature.get("finbert_input_article_set_hash") or ""))
        if high_neg > 0:
            weight_high_neg += weight
            largest_negative_weight = max(largest_negative_weight, weight)
        max_concentration = max(max_concentration, weight * max(0.0, -sentiment))
        disagreements.append(float(feature.get("finbert_sentiment_disagreement") or 0.0))
    return {
        "finbert_exposure_feature_contract_version": FINBERT_EXPOSURE_FEATURE_CONTRACT_VERSION,
        "feature_date": feature_date,
        "finbert_lookback_days": lookback_days,
        "portfolio_finbert_weighted_mean_sentiment": weighted_sentiment,
        "portfolio_finbert_weighted_negative_probability": weighted_negative,
        "portfolio_finbert_negative_news_breadth": weight_high_neg,
        "portfolio_weight_with_recent_news": weight_with_news,
        "portfolio_weight_with_high_confidence_negative_news": weight_high_neg,
        "portfolio_max_single_stock_adverse_news_concentration": max_concentration,
        "portfolio_finbert_source_diversity": len(sources) / max(covered, 1) if covered else 0.0,
        "portfolio_finbert_sentiment_disagreement": mean(disagreements) if disagreements else 0.0,
        "portfolio_largest_negative_sentiment_holding_weight": largest_negative_weight,
        "portfolio_finbert_covered_holding_count": covered,
        "portfolio_finbert_total_holding_count": len(holdings),
        "portfolio_finbert_input_set_hash": "",
        "_mismatches": mismatches,
    }


def _fixture_prediction(text: str) -> FinBertPrediction:
    lowered = text.lower()
    positive_words = ("beat", "growth", "upgrade", "profit", "strong", "record", "positive")
    negative_words = ("fraud", "miss", "lawsuit", "downgrade", "weak", "loss", "negative", "risk")
    pos = sum(word in lowered for word in positive_words)
    neg = sum(word in lowered for word in negative_words)
    if pos > neg:
        return FinBertPrediction(0.82, 0.13, 0.05, "positive")
    if neg > pos:
        return FinBertPrediction(0.06, 0.14, 0.80, "negative")
    return FinBertPrediction(0.12, 0.78, 0.10, "neutral")


def _article_audit(
    input_rows: Sequence[Mapping[str, Any]],
    scored: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    chunk_rows: Sequence[Mapping[str, Any]],
    identity: FinBertModelIdentity,
    availability_counts: Mapping[str, int],
) -> dict[str, Any]:
    sentiments = [_number(row.get("signed_sentiment_score")) for row in scored]
    sentiments = [value for value in sentiments if value is not None]
    dates = [row["news_available_timestamp"] for row in scored if row.get("news_available_timestamp")]
    return {
        "schema_version": "stock_alpha_finbert_scoring_audit.v1",
        "input_rows": len(input_rows),
        "unique_article_ids": len({_text(row.get("article_id")) for row in input_rows if _text(row.get("article_id"))}),
        "duplicate_article_rows": len(input_rows) - len({_article_id(row) + '|' + _symbol(row) for row in input_rows if _article_id(row) and _symbol(row)}),
        "eligible_articles": len(scored),
        "empty_text_articles": sum(row.get("error_message") == "empty selected article text" for row in rejected),
        "missing_timestamp_articles": sum("timestamp" in str(row.get("error_message", "")).lower() for row in rejected),
        "successfully_scored_articles": len(scored),
        "failed_articles": len(rejected),
        "completed_chunks": len(chunk_rows),
        "partial_chunks": 0,
        "resumed_chunks": sum(str(row.get("reused")) == "true" for row in chunk_rows),
        "model_identity": identity.__dict__,
        "date_range": [min(dates), max(dates)] if dates else None,
        "symbol_count": len({_symbol(row) for row in scored if _symbol(row)}),
        "sentiment_class_counts": {
            label: sum(row.get("sentiment_label") == label for row in scored)
            for label in ("positive", "neutral", "negative")
        },
        "average_signed_sentiment": mean(sentiments) if sentiments else 0.0,
        "probability_validation_failures": 0,
        "temporal_availability_audit": dict(availability_counts),
    }


def _article_audit_markdown(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# FinBERT Scoring Audit",
            "",
            f"- Input rows: `{payload['input_rows']}`",
            f"- Successfully scored: `{payload['successfully_scored_articles']}`",
            f"- Failed articles: `{payload['failed_articles']}`",
            f"- Completed chunks: `{payload['completed_chunks']}`",
            f"- Average signed sentiment: `{payload['average_signed_sentiment']}`",
        ]
    ) + "\n"


def _chunk_identity(items: Sequence[Mapping[str, Any]], identity: FinBertModelIdentity, max_token_length: int, config_hash: str) -> dict[str, Any]:
    payload = {
        "article_identities": [
            {
                "article_id": item["article_id"],
                "symbol": item["symbol"],
                "selected_text_hash": item["text"].text_hash,
            }
            for item in items
        ],
        "model_id": identity.model_id,
        "model_revision": identity.model_revision,
        "tokenizer_id": identity.tokenizer_id,
        "tokenizer_revision": identity.tokenizer_revision,
        "inference_contract_version": FINBERT_INFERENCE_CONTRACT_VERSION,
        "text_selection_contract_version": FINBERT_TEXT_SELECTION_CONTRACT_VERSION,
        "max_token_length": max_token_length,
        "configuration_hash": config_hash,
    }
    payload["chunk_id"] = MLCoreArtifactWriter.hash_payload(payload)[:24]
    return payload


def _read_completed_chunk(path: Path, identity: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if payload.get("status") != "completed":
        return None
    return payload if payload.get("identity") == identity else None


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _normalise_scored_article(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(row),
        "symbol": _symbol(row),
        "news_available_timestamp": _format_timestamp(_parse_timestamp(row["news_available_timestamp"])),
    }


def _empty_stock_feature_values(decision_timestamp: str, symbol: str) -> dict[str, Any]:
    row = {column: 0.0 for column in STOCK_NEWS_FEATURE_COLUMNS}
    row.update(
        {
            "finbert_news_feature_contract_version": FINBERT_STOCK_FEATURE_CONTRACT_VERSION,
            "decision_timestamp": decision_timestamp,
            "symbol": symbol,
            "finbert_news_coverage_indicator": 0,
            "finbert_article_count": 0,
            "finbert_scored_article_count": 0,
            "finbert_input_article_set_hash": _sha256_text("empty"),
        }
    )
    return row


def _fixture_articles() -> list[dict[str, Any]]:
    return [
        {"article_id": "a1", "symbol": "AAPL", "provider": "fixture", "source": "wire", "headline": "AAPL beats estimates with strong growth", "summary": "Record profit and positive guidance.", "published_at_utc": "2024-01-02T10:00:00Z", "provider_available_at_utc": "2024-01-02T10:02:00Z", "ingested_at": "2024-01-02T10:03:00Z"},
        {"article_id": "a2", "symbol": "AAPL", "provider": "fixture", "source": "wire", "headline": "AAPL faces lawsuit risk", "summary": "Negative legal risk weighs on shares.", "published_at_utc": "2024-01-03T11:00:00Z", "first_seen_at": "2024-01-03T11:05:00Z", "ingested_at": "2024-01-03T11:06:00Z"},
        {"article_id": "a2", "symbol": "AAPL", "provider": "fixture", "source": "wire", "headline": "AAPL faces lawsuit risk", "summary": "Negative legal risk weighs on shares.", "published_at_utc": "2024-01-03T11:00:00Z", "first_seen_at": "2024-01-03T11:05:00Z", "ingested_at": "2024-01-03T11:06:00Z"},
        {"article_id": "a3", "symbol": "MSFT", "provider": "fixture", "source": "wire", "headline": "MSFT holds investor day", "summary": "Management discussed product roadmap.", "published_at_utc": "2024-01-04T15:00:00Z", "ingested_at": "2024-01-04T15:10:00Z"},
        {"article_id": "a4", "symbol": "MSFT", "provider": "fixture", "source": "wire", "headline": "MSFT downgrade after weak demand", "summary": "Analysts cite weak demand and loss risk.", "published_at_utc": "2024-01-09T14:00:00Z", "ingested_at": "2024-01-09T14:05:00Z"},
        {"article_id": "missing_text", "symbol": "AAPL", "provider": "fixture", "published_at_utc": "2024-01-05T10:00:00Z", "ingested_at": "2024-01-05T10:01:00Z"},
        {"article_id": "missing_time", "symbol": "MSFT", "provider": "fixture", "headline": "MSFT neutral note", "summary": "No change to rating."},
    ]


def _fixture_decision_rows() -> list[dict[str, Any]]:
    return [
        {"rebalance_date": "2024-01-04T21:00:00+00:00", "symbol": "AAPL", "predicted_momentum_20d": "-0.08"},
        {"rebalance_date": "2024-01-04T21:00:00+00:00", "symbol": "MSFT", "predicted_momentum_20d": "0.02"},
        {"rebalance_date": "2024-01-08T21:00:00+00:00", "symbol": "AAPL", "predicted_momentum_20d": "-0.10"},
        {"rebalance_date": "2024-01-08T21:00:00+00:00", "symbol": "MSFT", "predicted_momentum_20d": "0.01"},
    ]


def _fixture_holdings_rows() -> list[dict[str, Any]]:
    return [
        {"rebalance_date": "2024-01-04", "symbol": "AAPL", "weight": "0.6"},
        {"rebalance_date": "2024-01-04", "symbol": "MSFT", "weight": "0.4"},
        {"rebalance_date": "2024-01-08", "symbol": "AAPL", "weight": "0.5"},
        {"rebalance_date": "2024-01-08", "symbol": "MSFT", "weight": "0.5"},
    ]


def _smoke_markdown(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Ticket 9A FinBERT News Probe",
            "",
            f"- Status: `{summary['status']}`",
            f"- Articles: `{summary['articles']}`",
            f"- Successful scores: `{summary['successful_scores']}`",
            f"- Rejections: `{summary['rejections']}`",
            f"- Future exclusions: `{summary['future_exclusions']}`",
            f"- Selector joined rows: `{summary['selector_joined_rows']}`",
            f"- Exposure aggregated rows: `{summary['exposure_aggregated_rows']}`",
        ]
    ) + "\n"


def _clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return _normalise_whitespace(text)


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _same_or_repeated(headline: str, body: str) -> bool:
    left = headline.strip().lower()
    right = body.strip().lower()
    return bool(left) and (left == right or right.startswith(left + " "))


def _parse_timestamp(value: Any) -> datetime:
    text = _text(value)
    if not text:
        raise ValueError("empty timestamp")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _date_only(value: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()))


def _text(value: Any) -> str:
    return str(value or "").strip()


def _symbol(row: Mapping[str, Any]) -> str:
    return _text(row.get("symbol") or row.get("ticker")).upper()


def _article_id(row: Mapping[str, Any]) -> str:
    return _text(row.get("article_id") or row.get("provider_article_id") or row.get("id"))


def _decision_timestamp(row: Mapping[str, Any]) -> str:
    return _text(row.get("decision_timestamp") or row.get("rebalance_date") or row.get("feature_date"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_rows_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return MLCoreArtifactWriter.hash_payload([dict(row) for row in rows])


def _article_set_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return MLCoreArtifactWriter.hash_payload(
        [
            {
                "article_id": row.get("article_id"),
                "symbol": row.get("symbol"),
                "selected_text_hash": row.get("selected_text_hash"),
            }
            for row in rows
        ]
    )


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _rejected_row(row: Mapping[str, Any], source_row_number: int, error_type: str, error_message: str) -> dict[str, Any]:
    return {
        "source_row_number": source_row_number,
        "article_id": _article_id(row),
        "symbol": _symbol(row),
        "error_type": error_type,
        "error_message": error_message,
    }


def _duplicate_key_count(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> int:
    seen = set()
    duplicates = 0
    for row in rows:
        key = tuple(str(row.get(column, "")) for column in keys)
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def _coverage_by(rows: Sequence[Mapping[str, Any]], key: str, *, indicator: str = "finbert_news_coverage_indicator") -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        bucket = output.setdefault(value, {"rows": 0, "covered_rows": 0})
        bucket["rows"] += 1
        bucket["covered_rows"] += int(float(row.get(indicator) or 0.0) > 0)
    for bucket in output.values():
        bucket["coverage_rate"] = bucket["covered_rows"] / bucket["rows"] if bucket["rows"] else 0.0
    return output


def _coverage_by_year(rows: Sequence[Mapping[str, Any]], key: str, *, indicator: str = "finbert_news_coverage_indicator") -> dict[str, dict[str, Any]]:
    projected = [{**dict(row), "year": str(row.get(key, ""))[:4]} for row in rows]
    return _coverage_by(projected, "year", indicator=indicator)
