from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.ml.artifacts.artifact_writers import MLCoreArtifactWriter
from core.research.ml.stock_level.news_sources.historical_canonical_corpus import (
    HISTORICAL_CANONICAL_CORPUS_SCHEMA_VERSION,
    canonical_rows_logical_checksum,
    verify_canonical_corpus_inventory,
)
from core.research.ml.stock_level.stock_alpha_finbert_news import (
    ARTICLE_LEVEL_FIELDS,
    FINBERT_INFERENCE_CONTRACT_VERSION,
    FINBERT_TEXT_SELECTION_CONTRACT_VERSION,
    FinBertModelIdentity,
    finbert_chunk_identity,
    resolve_news_available_timestamp,
    select_article_text,
)


SCORING_PLAN_CONTRACT = "stock_alpha_finbert_production_scoring_plan.v1"
ELIGIBILITY_CONTRACT = "canonical_finbert_article_eligibility.v1"
ORDERING_POLICY = "article_id,symbol,selected_text_hash ascending"
PARTITIONING_ALGORITHM = (
    "stable ordered eligible inventory split into contiguous fixed-size chunks"
)
UNPINNED_REVISIONS = {"", "main", "master", "latest"}


def publish_finbert_scoring_plan(
    *,
    corpus_manifest: Mapping[str, Any],
    corpus_path: Path,
    canonical_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    model_identity: FinBertModelIdentity,
    scoring_config: Mapping[str, Any],
    chunk_size: int,
    max_token_length: int = 256,
    max_characters: int = 10_000,
    scope: str = "production",
    source_commit: str | None = None,
) -> dict[str, Any]:
    if scope != "production":
        raise ValueError("Only explicit production scope may publish this scoring plan")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if max_token_length < 1 or max_characters < 1:
        raise ValueError("token limits must be positive")
    _validate_pinned_identity(model_identity)
    verification = verify_canonical_corpus_inventory(
        corpus_manifest, corpus_path=corpus_path
    )
    if (
        corpus_manifest.get("schema_version")
        != HISTORICAL_CANONICAL_CORPUS_SCHEMA_VERSION
        or not verification["inventory_certified"]
    ):
        raise ValueError(
            "Inventory-certified canonical corpus v2 is required: "
            + ",".join(verification["reasons"])
        )
    supplied_checksum = canonical_rows_logical_checksum(canonical_rows)
    if supplied_checksum != corpus_manifest["canonical_rows_logical_checksum"]:
        raise ValueError("Supplied canonical article inventory checksum mismatch")

    eligible, exclusions = build_eligible_scoring_inventory(
        canonical_rows, max_characters=max_characters,
        canonical_identity=corpus_manifest["canonical_corpus_identity"],
    )
    invalid_count = len(exclusions)
    configuration_checksum = MLCoreArtifactWriter.hash_payload(scoring_config)
    chunks = []
    for index, start in enumerate(range(0, len(eligible), chunk_size), start=1):
        items = eligible[start : start + chunk_size]
        identity = finbert_chunk_identity(
            items, model_identity, max_token_length, configuration_checksum
        )
        chunks.append(
            {
                "ordinal": index,
                "chunk_id": identity["chunk_id"],
                "article_count": len(items),
                "identity": identity,
            }
        )
    score_schema = {
        "inference_contract": FINBERT_INFERENCE_CONTRACT_VERSION,
        "fields": list(ARTICLE_LEVEL_FIELDS),
    }
    model_payload = {
        "model_id": model_identity.model_id,
        "model_revision": model_identity.model_revision,
        "tokenizer_id": model_identity.tokenizer_id,
        "tokenizer_revision": model_identity.tokenizer_revision,
    }
    manifest = {
        "scoring_plan_contract": SCORING_PLAN_CONTRACT,
        "scoring_plan_version": "v1",
        "scope": "production",
        "canonical_corpus_identity": corpus_manifest["canonical_corpus_identity"],
        "canonical_corpus_manifest_checksum": corpus_manifest[
            "logical_manifest_checksum"
        ],
        "canonical_corpus_checksum": corpus_manifest["canonical_corpus_checksum"],
        "eligible_article_inventory_logical_checksum": _hash(
            [_article_identity(item) for item in eligible]
        ),
        "source_canonical_rows_logical_checksum": supplied_checksum,
        "eligibility_validation_contract": ELIGIBILITY_CONTRACT,
        "text_selection_contract": FINBERT_TEXT_SELECTION_CONTRACT_VERSION,
        "inference_contract": FINBERT_INFERENCE_CONTRACT_VERSION,
        "finbert_model_identity": model_payload,
        "maximum_token_length": max_token_length,
        "maximum_selected_text_characters": max_characters,
        "chunk_size": chunk_size,
        "deterministic_ordering_policy": ORDERING_POLICY,
        "deterministic_partitioning_algorithm": PARTITIONING_ALGORITHM,
        "score_schema": score_schema,
        "score_schema_checksum": _hash(score_schema),
        "expected_chunks": chunks,
        "expected_chunk_count": len(chunks),
        "expected_article_count": len(eligible),
        "first_eligible_article_identity": (
            _article_identity(eligible[0]) if eligible else None
        ),
        "last_eligible_article_identity": (
            _article_identity(eligible[-1]) if eligible else None
        ),
        "duplicate_article_count": 0,
        "invalid_or_excluded_article_count": invalid_count,
        "configuration_checksum": configuration_checksum,
        "source_code_commit": source_commit or _git_commit(),
        "production_scoring_complete": False,
        "model_loading_invoked": False,
        "model_download_invoked": False,
        "inference_invoked": False,
        "score_chunks_inspected": False,
    }
    manifest["logical_checksum"] = _hash(manifest)
    manifest["plan_artifact_checksum"] = _hash(manifest)
    publication_result = _publish_atomic(output_path, manifest)
    return {**manifest, "publication_result": publication_result}


def build_eligible_scoring_inventory(
    rows: Sequence[Mapping[str, Any]], *, max_characters: int,
    canonical_identity: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    eligible = []
    exclusions: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row_number, source in enumerate(rows, 1):
        row = dict(source)
        article_id = str(
            row.get("article_id")
            or row.get("provider_article_id")
            or row.get("id")
            or ""
        ).strip()
        symbol = str(row.get("symbol") or row.get("ticker") or "").strip().upper()
        if not article_id or not symbol:
            raise ValueError("Canonical scoring owner identity is incomplete")
        owner = (article_id, symbol)
        if owner in seen:
            raise ValueError(f"Duplicate canonical article identity: {article_id}|{symbol}")
        seen.add(owner)
        try:
            text = select_article_text(row, max_characters=max_characters)
        except ValueError:
            owner = _hash({
                "canonical_identity": canonical_identity,
                "article_id": article_id, "symbol": symbol,
                "source_row_number": row_number,
            }).lower()
            exclusion = {
                "owner_identity": owner,
                "provider_article_identity_hash": _hash(article_id).lower(),
                "symbol": symbol,
                "reason_code": "NO_SELECTABLE_SCORING_TEXT",
                "canonical_identity": canonical_identity,
                "text_selection_contract":
                    FINBERT_TEXT_SELECTION_CONTRACT_VERSION,
            }
            exclusion["exclusion_identity"] = _hash(exclusion).lower()
            exclusions.append(exclusion)
            continue
        availability = resolve_news_available_timestamp(row)
        eligible.append(
            {
                "article_id": article_id,
                "symbol": symbol,
                "text": text,
                "availability": availability,
                "source_row": row,
            }
        )
    eligible.sort(
        key=lambda item: (
            item["article_id"], item["symbol"], item["text"].text_hash
        )
    )
    return eligible, exclusions


def _validate_pinned_identity(identity: FinBertModelIdentity) -> None:
    for field in ("model_id", "model_revision", "tokenizer_id", "tokenizer_revision"):
        value = str(getattr(identity, field) or "").strip()
        if not value:
            raise ValueError(f"Pinned FinBERT identity field is required: {field}")
    if identity.model_revision.lower() in UNPINNED_REVISIONS:
        raise ValueError("FinBERT model revision must be pinned")
    if identity.tokenizer_revision.lower() in UNPINNED_REVISIONS:
        raise ValueError("FinBERT tokenizer revision must be pinned")


def _article_identity(item: Mapping[str, Any]) -> dict[str, str]:
    return {
        "article_id": str(item["article_id"]),
        "symbol": str(item["symbol"]),
        "selected_text_hash": item["text"].text_hash,
    }


def _publish_atomic(path: Path, manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError("Incompatible existing FinBERT scoring plan")
        return "SKIPPED_COMPATIBLE"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return "PUBLISHED"


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
