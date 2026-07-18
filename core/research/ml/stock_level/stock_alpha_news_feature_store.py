from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from core.research.ml.stock_level.stock_alpha_news_pit_policy import (
    STRICT_COLLECTED_AT,
    StockAlphaNewsPitPolicy,
    article_is_pit_eligible,
    pit_policy_payload,
)


FEATURE_STORE_CONTRACT = "canonical_partitioned_pit_news_feature_store.v1"
FEATURE_SCHEMA_CONTRACT = "canonical_daily_pit_finbert_features.v1"
SESSION_ASSIGNMENT_POLICY = (
    "canonical_daily_spine decision_timestamp UTC; features are assigned to "
    "the spine decision_session_date"
)
PARTITIONING_SCHEME = "decision_date=YYYY-MM-DD/part-00000.jsonl"
FEATURE_FIELDS = (
    "asset_id",
    "symbol",
    "decision_session_date",
    "decision_timestamp",
    "lookback_days",
    "news_missing",
    "eligible_article_count",
    "mean_signed_sentiment",
    "mean_positive_probability",
    "mean_negative_probability",
    "latest_eligible_publication_timestamp",
    "latest_eligible_collection_timestamp",
    "eligible_article_set_checksum",
)


def publish_pit_news_feature_store(
    *,
    canonical_corpus_manifest: Mapping[str, Any],
    score_store_manifest: Mapping[str, Any],
    daily_spine_manifest: Mapping[str, Any],
    ticker_mapping_manifest: Mapping[str, Any],
    scored_articles: Sequence[Mapping[str, Any]],
    spine_rows: Sequence[Mapping[str, Any]],
    ticker_aliases: Mapping[str, str],
    output_root: Path,
    finbert_model_identity: Mapping[str, str],
    source_commit: str | None = None,
    lookback_days: Sequence[int] = (1, 3, 7),
    pit_policy: StockAlphaNewsPitPolicy | None = None,
) -> dict[str, Any]:
    policy = pit_policy or StockAlphaNewsPitPolicy(
        mode=STRICT_COLLECTED_AT,
        availability_lag_hours=0.0,
        historical_provider_availability_assumed=False,
    )
    parents = _validate_parents(
        canonical_corpus_manifest=canonical_corpus_manifest,
        score_store_manifest=score_store_manifest,
        daily_spine_manifest=daily_spine_manifest,
        ticker_mapping_manifest=ticker_mapping_manifest,
        finbert_model_identity=finbert_model_identity,
    )
    windows = tuple(sorted({int(value) for value in lookback_days}))
    if not windows or any(value < 1 for value in windows):
        raise ValueError("Aggregation windows must be positive days")
    aliases = {
        _symbol(alias): _symbol(canonical)
        for alias, canonical in ticker_aliases.items()
    }
    _validate_parent_content(
        score_store_manifest=score_store_manifest,
        daily_spine_manifest=daily_spine_manifest,
        ticker_mapping_manifest=ticker_mapping_manifest,
        scored_articles=scored_articles,
        spine_rows=spine_rows,
        aliases=aliases,
    )
    normalised_articles, article_evidence = _normalise_articles(
        scored_articles,
        aliases=aliases,
        policy=policy,
        finbert_model_identity=finbert_model_identity,
    )
    rows, eligibility = _build_rows(
        spine_rows=spine_rows,
        articles=normalised_articles,
        windows=windows,
        policy=policy,
    )
    schema = {
        "contract": FEATURE_SCHEMA_CONTRACT,
        "fields": list(FEATURE_FIELDS),
        "missing_news_semantics": (
            "news_missing=true, eligible_article_count=0, sentiment and "
            "probability aggregates are null"
        ),
    }
    schema_checksum = _hash(schema)
    partition_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        partition_rows.setdefault(row["decision_session_date"], []).append(row)
    partition_records = []
    partition_publication_results = []
    for decision_date in sorted(partition_rows):
        ordered = sorted(
            partition_rows[decision_date],
            key=lambda row: (
                str(row["asset_id"]), str(row["symbol"]), int(row["lookback_days"])
            ),
        )
        payload = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
            for row in ordered
        ).encode("utf-8")
        checksum = hashlib.sha256(payload).hexdigest().upper()
        relative = Path(f"decision_date={decision_date}") / "part-00000.jsonl"
        publication = _publish_partition(output_root / relative, payload, checksum)
        partition_records.append(
            {
                "decision_date": decision_date,
                "relative_path": relative.as_posix(),
                "artifact_checksum": checksum,
                "row_count": len(ordered),
            }
        )
        partition_publication_results.append(
            {
                "relative_path": relative.as_posix(),
                "publication_result": publication,
            }
        )
    coverage = {
        "decision_date_min": min(partition_rows) if partition_rows else None,
        "decision_date_max": max(partition_rows) if partition_rows else None,
        "decision_date_count": len(partition_rows),
        "asset_count": len({row["asset_id"] for row in rows}),
    }
    manifest = {
        "feature_store_contract": FEATURE_STORE_CONTRACT,
        "feature_store_version": "v1",
        **parents,
        "finbert_model_identity": dict(finbert_model_identity),
        "pit_eligibility_policy_identity": _hash(pit_policy_payload(policy)),
        "pit_eligibility_policy": pit_policy_payload(policy),
        "feature_schema": schema,
        "feature_schema_checksum": schema_checksum,
        "aggregation_windows": [
            {"window_id": f"trailing_{days}_calendar_days", "days": days}
            for days in windows
        ],
        "timezone": "UTC",
        "session_assignment_policy": SESSION_ASSIGNMENT_POLICY,
        "publication_partitioning_scheme": PARTITIONING_SCHEME,
        "source_code_commit": source_commit or _git_commit(),
        "row_count": len(rows),
        "coverage": coverage,
        "missingness_evidence": {
            "rows_with_missing_news": sum(row["news_missing"] for row in rows),
            "rows_with_eligible_news": sum(not row["news_missing"] for row in rows),
        },
        "eligibility_evidence": {**article_evidence, **eligibility},
        "partitions": partition_records,
        "production_finbert_scoring_proven": True,
        "finbert_scoring_invoked": False,
        "transformer_training_invoked": False,
    }
    manifest["feature_store_artifact_checksum"] = _hash(
        [
            (row["relative_path"], row["artifact_checksum"], row["row_count"])
            for row in partition_records
        ]
    )
    manifest["logical_checksum"] = _hash(manifest)
    manifest_path = output_root / "manifest.json"
    manifest_publication_result = _publish_manifest(manifest_path, manifest)
    return {
        **manifest,
        "partition_publication_results": partition_publication_results,
        "manifest_publication_result": manifest_publication_result,
    }


def _validate_parents(
    *,
    canonical_corpus_manifest: Mapping[str, Any],
    score_store_manifest: Mapping[str, Any],
    daily_spine_manifest: Mapping[str, Any],
    ticker_mapping_manifest: Mapping[str, Any],
    finbert_model_identity: Mapping[str, str],
) -> dict[str, Any]:
    required_model = (
        "model_id", "model_revision", "tokenizer_id", "tokenizer_revision"
    )
    if any(not str(finbert_model_identity.get(key) or "").strip() for key in required_model):
        raise ValueError("Pinned FinBERT model and tokenizer identity is required")
    if any(
        str(finbert_model_identity[key]).lower() in {"main", "master", "latest"}
        for key in ("model_revision", "tokenizer_revision")
    ):
        raise ValueError("FinBERT revisions must be pinned")
    if score_store_manifest.get("production_scoring_complete") is not True:
        raise ValueError("Production FinBERT score-store completion is not proven")
    if dict(score_store_manifest.get("finbert_model_identity") or {}) != dict(
        finbert_model_identity
    ):
        raise ValueError("FinBERT score-store model identity mismatch")
    identities = {
        "canonical_corpus_identity": canonical_corpus_manifest.get(
            "canonical_corpus_identity"
        ),
        "canonical_corpus_checksum": canonical_corpus_manifest.get(
            "canonical_corpus_checksum"
        ),
        "score_store_identity": score_store_manifest.get("score_store_identity"),
        "score_store_checksum": score_store_manifest.get("score_store_checksum"),
        "canonical_daily_spine_identity": daily_spine_manifest.get(
            "daily_spine_identity"
        ),
        "canonical_daily_spine_checksum": daily_spine_manifest.get(
            "daily_spine_checksum"
        ),
        "ticker_mapping_identity": ticker_mapping_manifest.get(
            "ticker_mapping_identity"
        ),
        "ticker_mapping_checksum": ticker_mapping_manifest.get(
            "ticker_mapping_checksum"
        ),
    }
    missing = sorted(key for key, value in identities.items() if not value)
    if missing:
        raise ValueError(f"Feature-store parent identities missing: {','.join(missing)}")
    return identities


def _normalise_articles(
    rows: Sequence[Mapping[str, Any]],
    *,
    aliases: Mapping[str, str],
    policy: StockAlphaNewsPitPolicy,
    finbert_model_identity: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output = []
    evidence = {
        "scored_article_rows": len(rows),
        "unmapped_ticker_rows": 0,
        "invalid_timestamp_rows": 0,
        "model_identity_mismatch_rows": 0,
    }
    for source in rows:
        row = dict(source)
        identity_fields = {
            "finbert_model_id": "model_id",
            "finbert_model_revision": "model_revision",
            "tokenizer_id": "tokenizer_id",
            "tokenizer_revision": "tokenizer_revision",
        }
        if any(
            str(row.get(row_key) or "")
            != str(finbert_model_identity[identity_key])
            for row_key, identity_key in identity_fields.items()
        ):
            evidence["model_identity_mismatch_rows"] += 1
            raise ValueError("Scored article FinBERT identity mismatch")
        raw_symbol = _symbol(row.get("symbol") or row.get("ticker"))
        symbol = aliases.get(raw_symbol, raw_symbol)
        if not symbol:
            evidence["unmapped_ticker_rows"] += 1
            continue
        try:
            published = _timestamp(
                row.get("published_at_utc") or row.get("published_at"),
                field="published_at_utc",
            )
            collected = _timestamp(
                row.get("collected_at_utc")
                or row.get("ingested_at")
                or row.get("news_available_timestamp"),
                field="collected_at_utc",
            )
        except ValueError:
            evidence["invalid_timestamp_rows"] += 1
            raise
        row.update(
            symbol=symbol,
            published_at_utc=published,
            collected_at_utc=collected,
            available_at_utc=(
                published + timedelta(hours=policy.availability_lag_hours)
                if policy.eligibility_timestamp_field == "available_at_utc"
                else None
            ),
        )
        output.append(row)
    return output, evidence


def _validate_parent_content(
    *,
    score_store_manifest: Mapping[str, Any],
    daily_spine_manifest: Mapping[str, Any],
    ticker_mapping_manifest: Mapping[str, Any],
    scored_articles: Sequence[Mapping[str, Any]],
    spine_rows: Sequence[Mapping[str, Any]],
    aliases: Mapping[str, str],
) -> None:
    checks = {
        "scored_rows_logical_checksum": (
            score_store_manifest.get("scored_rows_logical_checksum"),
            _hash([dict(row) for row in scored_articles]),
        ),
        "spine_rows_logical_checksum": (
            daily_spine_manifest.get("spine_rows_logical_checksum"),
            _hash([dict(row) for row in spine_rows]),
        ),
        "ticker_aliases_logical_checksum": (
            ticker_mapping_manifest.get("ticker_aliases_logical_checksum"),
            _hash(dict(aliases)),
        ),
    }
    for field, (expected, actual) in checks.items():
        if not expected:
            raise ValueError(f"Feature-store parent content checksum missing: {field}")
        if str(expected) != actual:
            raise ValueError(f"Feature-store parent content checksum mismatch: {field}")


def _build_rows(
    *,
    spine_rows: Sequence[Mapping[str, Any]],
    articles: Sequence[Mapping[str, Any]],
    windows: Sequence[int],
    policy: StockAlphaNewsPitPolicy,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output = []
    evidence = {
        "article_decision_candidates": 0,
        "eligible_article_contributions": 0,
        "post_decision_article_exclusions": 0,
        "pre_window_article_exclusions": 0,
    }
    for spine in spine_rows:
        asset_id = str(spine.get("asset_id") or "").strip()
        symbol = _symbol(spine.get("symbol"))
        decision_date = str(spine.get("decision_session_date") or "")
        decision = _timestamp(
            spine.get("decision_timestamp"), field="decision_timestamp"
        )
        if not asset_id or not symbol or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}", decision_date
        ):
            raise ValueError("Canonical spine row identity is incomplete")
        symbol_articles = [row for row in articles if row["symbol"] == symbol]
        for days in windows:
            start = decision - timedelta(days=days)
            eligible = []
            for article in symbol_articles:
                evidence["article_decision_candidates"] += 1
                if not article_is_pit_eligible(article, decision, policy):
                    evidence["post_decision_article_exclusions"] += 1
                    continue
                timestamp = article[policy.eligibility_timestamp_field]
                if timestamp < start:
                    evidence["pre_window_article_exclusions"] += 1
                    continue
                eligible.append(article)
            evidence["eligible_article_contributions"] += len(eligible)
            output.append(
                _feature_row(
                    asset_id=asset_id,
                    symbol=symbol,
                    decision_date=decision_date,
                    decision=decision,
                    days=days,
                    articles=eligible,
                )
            )
    return output, evidence


def _feature_row(
    *,
    asset_id: str,
    symbol: str,
    decision_date: str,
    decision: datetime,
    days: int,
    articles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    signed = [_number(row["signed_sentiment_score"]) for row in articles]
    positive = [_number(row["positive_probability"]) for row in articles]
    negative = [_number(row["negative_probability"]) for row in articles]
    identities = sorted(
        f"{row.get('article_id')}|{row.get('symbol')}|"
        f"{row['collected_at_utc'].isoformat()}"
        for row in articles
    )
    return {
        "asset_id": asset_id,
        "symbol": symbol,
        "decision_session_date": decision_date,
        "decision_timestamp": _format_timestamp(decision),
        "lookback_days": days,
        "news_missing": not bool(articles),
        "eligible_article_count": len(articles),
        "mean_signed_sentiment": mean(signed) if signed else None,
        "mean_positive_probability": mean(positive) if positive else None,
        "mean_negative_probability": mean(negative) if negative else None,
        "latest_eligible_publication_timestamp": (
            _format_timestamp(max(row["published_at_utc"] for row in articles))
            if articles else None
        ),
        "latest_eligible_collection_timestamp": (
            _format_timestamp(max(row["collected_at_utc"] for row in articles))
            if articles else None
        ),
        "eligible_article_set_checksum": _hash(identities),
    }


def _publish_partition(path: Path, payload: bytes, checksum: str) -> str:
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest().upper() != checksum:
            raise FileExistsError(f"Incompatible existing news partition: {path}")
        return "SKIPPED_COMPATIBLE"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return "PUBLISHED"


def _publish_manifest(path: Path, manifest: Mapping[str, Any]) -> str:
    comparable = dict(manifest)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != comparable:
            raise FileExistsError("Incompatible existing news feature-store manifest")
        return "SKIPPED_COMPATIBLE"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(comparable, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)
    return "PUBLISHED"


def _timestamp(value: Any, *, field: str) -> datetime:
    text = str(value or "").strip()
    if not text or re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ValueError(f"{field} requires an explicit timezone timestamp")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Malformed {field}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} requires an explicit timezone timestamp")
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _symbol(value: Any) -> str:
    return str(value or "").strip().upper().replace(".", "-")


def _number(value: Any) -> float:
    result = float(value)
    if not (-1e308 < result < 1e308):
        raise ValueError("Nonfinite news score")
    return result


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest().upper()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
