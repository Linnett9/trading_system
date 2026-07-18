from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.ml.stock_level.stock_alpha_finbert_news import (
    _chunk_manifest_logical_checksum,
    _chunk_metadata_logical_checksum,
    _scored_rows_logical_checksum,
    _validate_chunk_row_ownership,
)


SCORE_STORE_CONTRACT = "stock_alpha_finbert_production_score_store.v1"
SCORING_PLAN_CONTRACT = "stock_alpha_finbert_production_scoring_plan.v1"


def certify_finbert_score_store(
    *,
    scoring_plan: Mapping[str, Any],
    chunk_manifest_path: Path,
    output_path: Path,
    generated_at: str | None = None,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Validate bounded planned chunks and publish their top-level certificate."""
    plan = dict(scoring_plan)
    _validate_plan(plan)
    bounded_rows = _read_bounded_manifest(chunk_manifest_path)
    expected_chunks = list(plan["expected_chunks"])
    expected_by_id = {row["chunk_id"]: row for row in expected_chunks}
    supplied_by_id: dict[str, dict[str, str]] = {}
    duplicate_chunks: list[str] = []
    unexpected_chunks: list[str] = []
    for row in bounded_rows:
        chunk_id = str(row.get("chunk_id") or "")
        if chunk_id in supplied_by_id:
            duplicate_chunks.append(chunk_id)
            continue
        supplied_by_id[chunk_id] = row
        if chunk_id not in expected_by_id:
            unexpected_chunks.append(chunk_id)
    if duplicate_chunks:
        raise ValueError(
            "Duplicate FinBERT chunk ownership: " + ",".join(sorted(duplicate_chunks))
        )
    if unexpected_chunks:
        raise ValueError(
            "Unexpected FinBERT chunk evidence: "
            + ",".join(sorted(unexpected_chunks))
        )

    missing_chunks = []
    incomplete_chunks = []
    certified_chunks = []
    scored_rows: list[dict[str, Any]] = []
    for expected in expected_chunks:
        chunk_id = expected["chunk_id"]
        evidence = supplied_by_id.get(chunk_id)
        if evidence is None:
            missing_chunks.append(chunk_id)
            continue
        if evidence.get("status") != "completed":
            incomplete_chunks.append(
                {"chunk_id": chunk_id, "status": evidence.get("status") or "unknown"}
            )
            continue
        if evidence.get("production_scope") != "true":
            raise ValueError("Non-production chunk supplied for score-store certification")
        chunk_path = Path(str(evidence.get("chunk_path") or ""))
        if not chunk_path.is_file():
            missing_chunks.append(chunk_id)
            continue
        artifact_bytes = chunk_path.read_bytes()
        artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest().upper()
        if artifact_sha256 != evidence.get("chunk_artifact_sha256"):
            raise ValueError(f"FinBERT chunk artifact checksum mismatch: {chunk_id}")
        try:
            payload = json.loads(artifact_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid FinBERT chunk artifact: {chunk_id}") from exc
        _validate_completed_chunk(
            payload=payload,
            evidence=evidence,
            expected=expected,
            plan=plan,
        )
        chunk_rows = [dict(row) for row in payload["rows"]]
        scored_rows.extend(chunk_rows)
        certified_chunks.append(
            {
                "ordinal": expected["ordinal"],
                "chunk_id": chunk_id,
                "chunk_artifact_path": str(chunk_path),
                "chunk_artifact_sha256": artifact_sha256,
                "scored_rows_logical_checksum": evidence[
                    "scored_rows_logical_checksum"
                ],
                "chunk_metadata_logical_checksum": evidence[
                    "chunk_metadata_logical_checksum"
                ],
                "scoring_plan_identity": evidence["scoring_plan_identity"],
                "status": "completed",
            }
        )

    complete = not missing_chunks and not incomplete_chunks
    if complete:
        _validate_combined_inventory(scored_rows, expected_chunks)
    aggregate_artifact_checksum = _hash(
        [
            (
                row["ordinal"],
                row["chunk_id"],
                row["chunk_artifact_sha256"],
                row["scored_rows_logical_checksum"],
                row["chunk_metadata_logical_checksum"],
            )
            for row in certified_chunks
        ]
    )
    all_rows_checksum = _hash(scored_rows)
    score_store_identity = _hash(
        {
            "contract": SCORE_STORE_CONTRACT,
            "scoring_plan_logical_checksum": plan["logical_checksum"],
            "aggregate_artifact_checksum": aggregate_artifact_checksum,
            "production_scoring_complete": complete,
        }
    )
    manifest = {
        "score_store_contract": SCORE_STORE_CONTRACT,
        "score_store_version": "v1",
        "scope": "production",
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "score_store_identity": score_store_identity,
        "score_store_checksum": aggregate_artifact_checksum,
        "production_scoring_plan_identity": plan["logical_checksum"],
        "production_scoring_plan_logical_checksum": plan["logical_checksum"],
        "production_scoring_plan_artifact_checksum": plan[
            "plan_artifact_checksum"
        ],
        "canonical_corpus_identity": plan["canonical_corpus_identity"],
        "canonical_corpus_artifact_checksum": plan["canonical_corpus_checksum"],
        "canonical_corpus_checksum": plan["canonical_corpus_checksum"],
        "canonical_corpus_logical_rows_checksum": plan[
            "source_canonical_rows_logical_checksum"
        ],
        "finbert_model_identity": dict(plan["finbert_model_identity"]),
        "text_selection_contract": plan["text_selection_contract"],
        "inference_contract": plan["inference_contract"],
        "score_schema": plan["score_schema"],
        "score_schema_checksum": plan["score_schema_checksum"],
        "scoring_configuration_checksum": plan["configuration_checksum"],
        "maximum_token_length": plan["maximum_token_length"],
        "maximum_selected_text_characters": plan[
            "maximum_selected_text_characters"
        ],
        "chunk_size": plan["chunk_size"],
        "expected_chunk_count": plan["expected_chunk_count"],
        "certified_completed_chunk_count": len(certified_chunks),
        "expected_article_count": plan["expected_article_count"],
        "certified_scored_row_count": len(scored_rows),
        "ordered_expected_chunk_identities": [
            {
                "ordinal": row["ordinal"],
                "chunk_id": row["chunk_id"],
                "identity": row["identity"],
            }
            for row in expected_chunks
        ],
        "ordered_certified_chunks": certified_chunks,
        "aggregate_artifact_checksum": aggregate_artifact_checksum,
        "scored_rows_logical_checksum": all_rows_checksum,
        "first_planned_article_identity": plan[
            "first_eligible_article_identity"
        ],
        "last_planned_article_identity": plan["last_eligible_article_identity"],
        "missing_chunk_evidence": missing_chunks,
        "failed_or_incomplete_chunk_evidence": incomplete_chunks,
        "duplicate_chunk_evidence": [],
        "unexpected_chunk_evidence": [],
        "incompatible_chunk_evidence": [],
        "bounded_manifest_logical_checksum": (
            bounded_rows[0]["manifest_logical_checksum"] if bounded_rows else None
        ),
        "source_git_commit": source_commit or _git_commit(),
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "production_scoring_complete": complete,
        "model_loading_performed": False,
        "model_download_performed": False,
        "inference_performed": False,
        "scoring_chunks_modified": False,
    }
    manifest["logical_manifest_checksum"] = _logical_manifest_checksum(manifest)
    publication_result, artifact_checksum = _publish_atomic(output_path, manifest)
    return {
        **manifest,
        "certificate_artifact_checksum": artifact_checksum,
        "publication_result": publication_result,
    }


def _validate_plan(plan: Mapping[str, Any]) -> None:
    if (
        plan.get("scoring_plan_contract") != SCORING_PLAN_CONTRACT
        or plan.get("scoring_plan_version") != "v1"
        or plan.get("scope") != "production"
    ):
        raise ValueError("Invalid production FinBERT scoring plan")
    artifact_payload = {
        key: value for key, value in plan.items()
        if key not in {"plan_artifact_checksum", "publication_result"}
    }
    if plan.get("plan_artifact_checksum") != _hash(artifact_payload):
        raise ValueError("FinBERT scoring-plan artifact checksum mismatch")
    logical_payload = {
        key: value for key, value in artifact_payload.items()
        if key != "logical_checksum"
    }
    if plan.get("logical_checksum") != _hash(logical_payload):
        raise ValueError("FinBERT scoring-plan logical checksum mismatch")
    required = (
        "canonical_corpus_identity",
        "canonical_corpus_checksum",
        "source_canonical_rows_logical_checksum",
        "finbert_model_identity",
        "text_selection_contract",
        "inference_contract",
        "score_schema",
        "score_schema_checksum",
        "configuration_checksum",
    )
    if any(not plan.get(field) for field in required):
        raise ValueError("FinBERT scoring plan lacks certification lineage")
    if plan.get("score_schema_checksum") != _hash(plan["score_schema"]):
        raise ValueError("FinBERT scoring-plan score schema checksum mismatch")
    model = dict(plan["finbert_model_identity"])
    required_model = (
        "model_id", "model_revision", "tokenizer_id", "tokenizer_revision"
    )
    if any(not str(model.get(field) or "").strip() for field in required_model):
        raise ValueError("Pinned FinBERT model and tokenizer identity is required")
    if any(
        str(model[field]).lower() in {"main", "master", "latest"}
        for field in ("model_revision", "tokenizer_revision")
    ):
        raise ValueError("FinBERT model and tokenizer revisions must be pinned")
    chunks = list(plan.get("expected_chunks") or [])
    if (
        plan.get("expected_chunk_count") != len(chunks)
        or [row.get("ordinal") for row in chunks]
        != list(range(1, len(chunks) + 1))
        or len({row.get("chunk_id") for row in chunks}) != len(chunks)
        or plan.get("expected_article_count")
        != sum(int(row.get("article_count") or 0) for row in chunks)
    ):
        raise ValueError("Invalid planned FinBERT chunk inventory")
    article_identities = []
    for chunk in chunks:
        identity = dict(chunk.get("identity") or {})
        expected_identity_fields = {
            **model,
            "inference_contract_version": plan["inference_contract"],
            "text_selection_contract_version": plan["text_selection_contract"],
            "max_token_length": plan["maximum_token_length"],
            "configuration_hash": plan["configuration_checksum"],
        }
        if (
            chunk.get("chunk_id") != identity.get("chunk_id")
            or chunk.get("article_count")
            != len(identity.get("article_identities") or [])
            or any(
                identity.get(field) != value
                for field, value in expected_identity_fields.items()
            )
        ):
            raise ValueError("Planned FinBERT chunk identity mismatch")
        article_identities.extend(identity["article_identities"])
    if (
        (article_identities[0] if article_identities else None)
        != plan.get("first_eligible_article_identity")
        or (article_identities[-1] if article_identities else None)
        != plan.get("last_eligible_article_identity")
    ):
        raise ValueError("Planned FinBERT first or last article identity mismatch")


def _read_bounded_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        return []
    checksums = {row.get("manifest_logical_checksum") for row in rows}
    if len(checksums) != 1 or checksums.pop() != _chunk_manifest_logical_checksum(
        rows
    ):
        raise ValueError("FinBERT bounded chunk manifest checksum mismatch")
    return rows


def _validate_completed_chunk(
    *,
    payload: Mapping[str, Any],
    evidence: Mapping[str, str],
    expected: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    chunk_id = expected["chunk_id"]
    if payload.get("status") != "completed":
        raise ValueError(f"Expected completed FinBERT chunk: {chunk_id}")
    if payload.get("production_scope") is not True:
        raise ValueError("Non-production chunk supplied for score-store certification")
    if payload.get("identity") != expected.get("identity"):
        raise ValueError(f"FinBERT chunk identity mismatch: {chunk_id}")
    binding = payload.get("scoring_plan")
    expected_binding = {
        "logical_checksum": plan["logical_checksum"],
        "plan_artifact_checksum": plan["plan_artifact_checksum"],
        "planned_ordinal": expected["ordinal"],
    }
    if binding != expected_binding:
        raise ValueError(f"FinBERT scoring-plan binding mismatch: {chunk_id}")
    if (
        evidence.get("planned_ordinal") != str(expected["ordinal"])
        or evidence.get("scoring_plan_identity") != plan["logical_checksum"]
    ):
        raise ValueError(f"FinBERT bounded chunk ordinal mismatch: {chunk_id}")
    _validate_chunk_row_ownership(payload, expected["identity"])
    rows_checksum = _scored_rows_logical_checksum(
        payload.get("rows") or [],
        chunk_id=chunk_id,
        planned_ordinal=expected["ordinal"],
    )
    if (
        payload.get("scored_rows_logical_checksum") != rows_checksum
        or evidence.get("scored_rows_logical_checksum") != rows_checksum
    ):
        raise ValueError(f"FinBERT scored-row checksum mismatch: {chunk_id}")
    metadata_checksum = _chunk_metadata_logical_checksum(payload)
    if (
        payload.get("chunk_metadata_logical_checksum") != metadata_checksum
        or evidence.get("chunk_metadata_logical_checksum") != metadata_checksum
    ):
        raise ValueError(f"FinBERT chunk metadata checksum mismatch: {chunk_id}")


def _validate_combined_inventory(
    rows: Sequence[Mapping[str, Any]],
    expected_chunks: Sequence[Mapping[str, Any]],
) -> None:
    expected = [
        identity
        for chunk in expected_chunks
        for identity in chunk["identity"]["article_identities"]
    ]
    actual = [
        {
            "article_id": row.get("article_id"),
            "symbol": row.get("symbol"),
            "selected_text_hash": row.get("selected_text_hash"),
        }
        for row in rows
    ]
    if actual != expected:
        raise ValueError("Certified FinBERT scored-row inventory mismatch")


def _logical_manifest_checksum(manifest: Mapping[str, Any]) -> str:
    return _hash(
        {
            key: value for key, value in manifest.items()
            if key not in {
                "generated_at",
                "logical_manifest_checksum",
                "certificate_artifact_checksum",
                "publication_result",
            }
        }
    )


def _publish_atomic(
    path: Path, manifest: Mapping[str, Any]
) -> tuple[str, str]:
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FileExistsError(
                "Invalid existing FinBERT score-store certificate"
            ) from exc
        if (
            existing.get("logical_manifest_checksum")
            != manifest["logical_manifest_checksum"]
            or _logical_manifest_checksum(existing)
            != existing.get("logical_manifest_checksum")
        ):
            raise FileExistsError(
                "Incompatible existing FinBERT score-store certificate"
            )
        return (
            "SKIPPED_COMPATIBLE",
            hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return "PUBLISHED", hashlib.sha256(payload).hexdigest().upper()


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Certify an explicit planned FinBERT chunk inventory."
    )
    parser.add_argument("--scoring-plan", required=True, type=Path)
    parser.add_argument("--chunk-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = certify_finbert_score_store(
        scoring_plan=json.loads(args.scoring_plan.read_text(encoding="utf-8")),
        chunk_manifest_path=args.chunk_manifest,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
