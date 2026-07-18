from __future__ import annotations

import csv
import hashlib
import json

import pytest

from core.research.ml.artifacts.artifact_writers import MLCoreArtifactWriter
from core.research.ml.stock_level.stock_alpha_finbert_news import (
    ARTICLE_LEVEL_FIELDS,
    FINBERT_INFERENCE_CONTRACT_VERSION,
    FINBERT_TEXT_SELECTION_CONTRACT_VERSION,
    DeterministicFinBertFixtureAdapter,
    _chunk_manifest_logical_checksum,
    finbert_chunk_identity,
    score_finbert_articles,
    select_article_text,
)
from core.research.ml.stock_level.stock_alpha_finbert_score_store import (
    certify_finbert_score_store,
)


def test_complete_planned_inventory_certifies_deterministically(tmp_path):
    fixture = _scored_fixture(tmp_path / "chunks", chunk_size=1)
    output = tmp_path / "score_store.json"

    first = certify_finbert_score_store(
        scoring_plan=fixture["plan"],
        chunk_manifest_path=fixture["manifest"],
        output_path=output,
        generated_at="2024-01-10T00:00:00+00:00",
        source_commit="fixture-commit",
    )
    second = certify_finbert_score_store(
        scoring_plan=fixture["plan"],
        chunk_manifest_path=fixture["manifest"],
        output_path=output,
        generated_at="2025-02-11T00:00:00+00:00",
        source_commit="fixture-commit",
    )

    assert first["production_scoring_complete"] is True
    assert first["status"] == "COMPLETE"
    assert first["certified_completed_chunk_count"] == 2
    assert first["certified_scored_row_count"] == 2
    assert first["logical_manifest_checksum"] == second["logical_manifest_checksum"]
    assert second["publication_result"] == "SKIPPED_COMPATIBLE"
    assert [row["ordinal"] for row in first["ordered_certified_chunks"]] == [1, 2]
    assert first["model_loading_performed"] is False
    assert first["model_download_performed"] is False
    assert first["inference_performed"] is False
    assert first["scoring_chunks_modified"] is False


def test_reordered_supplied_manifest_is_certified_in_planned_order(tmp_path):
    fixture = _scored_fixture(tmp_path / "chunks", chunk_size=1)
    rows = _manifest_rows(fixture["manifest"])
    _write_manifest(fixture["manifest"], list(reversed(rows)))

    result = certify_finbert_score_store(
        scoring_plan=fixture["plan"],
        chunk_manifest_path=fixture["manifest"],
        output_path=tmp_path / "store.json",
        generated_at="2024-01-10T00:00:00+00:00",
        source_commit="fixture-commit",
    )

    assert [row["ordinal"] for row in result["ordered_certified_chunks"]] == [1, 2]


def test_missing_and_failed_planned_chunks_are_incomplete(tmp_path):
    missing = certify_finbert_score_store(
        scoring_plan=_plan_fixture()[0],
        chunk_manifest_path=tmp_path / "absent.csv",
        output_path=tmp_path / "missing.json",
        generated_at="2024-01-10T00:00:00+00:00",
        source_commit="fixture-commit",
    )
    assert missing["status"] == "INCOMPLETE"
    assert missing["production_scoring_complete"] is False
    assert len(missing["missing_chunk_evidence"]) == 2

    fixture = _scored_fixture(tmp_path / "failed-chunks", chunk_size=1)
    rows = _manifest_rows(fixture["manifest"])
    rows[0]["status"] = "failed"
    _write_manifest(fixture["manifest"], rows)
    failed = certify_finbert_score_store(
        scoring_plan=fixture["plan"],
        chunk_manifest_path=fixture["manifest"],
        output_path=tmp_path / "failed.json",
        generated_at="2024-01-10T00:00:00+00:00",
        source_commit="fixture-commit",
    )
    assert failed["production_scoring_complete"] is False
    assert failed["failed_or_incomplete_chunk_evidence"][0]["status"] == "failed"


@pytest.mark.parametrize("status", ["running", "partial"])
def test_partial_planned_chunk_is_incomplete(tmp_path, status):
    fixture = _scored_fixture(tmp_path / status, chunk_size=2)
    rows = _manifest_rows(fixture["manifest"])
    rows[0]["status"] = status
    _write_manifest(fixture["manifest"], rows)

    result = certify_finbert_score_store(
        scoring_plan=fixture["plan"],
        chunk_manifest_path=fixture["manifest"],
        output_path=tmp_path / f"{status}.json",
        source_commit="fixture-commit",
    )

    assert result["status"] == "INCOMPLETE"
    assert result["failed_or_incomplete_chunk_evidence"][0]["status"] == status


@pytest.mark.parametrize("case", ["duplicate", "unexpected", "smoke"])
def test_invalid_manifest_ownership_fails_closed(tmp_path, case):
    fixture = _scored_fixture(tmp_path / case, chunk_size=1)
    rows = _manifest_rows(fixture["manifest"])
    if case == "duplicate":
        rows.append(dict(rows[0]))
        message = "Duplicate"
    elif case == "unexpected":
        rows[0]["chunk_id"] = "unexpected"
        message = "Unexpected"
    else:
        rows[0]["production_scope"] = "false"
        message = "Non-production"
    _write_manifest(fixture["manifest"], rows)

    with pytest.raises(ValueError, match=message):
        certify_finbert_score_store(
            scoring_plan=fixture["plan"],
            chunk_manifest_path=fixture["manifest"],
            output_path=tmp_path / "invalid.json",
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("ordinal", "ordinal mismatch"),
        ("plan", "scoring-plan binding mismatch"),
        ("identity", "chunk identity mismatch"),
        ("rows_checksum", "scored-row checksum mismatch"),
        ("metadata_checksum", "metadata checksum mismatch"),
        ("artifact_checksum", "artifact checksum mismatch"),
        ("manifest_checksum", "manifest checksum mismatch"),
        ("coverage", "row ownership mismatch"),
    ],
)
def test_incompatible_chunk_evidence_fails_closed(tmp_path, case, message):
    fixture = _scored_fixture(tmp_path / case, chunk_size=2)
    rows = _manifest_rows(fixture["manifest"])
    chunk_path = fixture["chunk_paths"][0]
    payload = json.loads(chunk_path.read_text())
    if case == "ordinal":
        rows[0]["planned_ordinal"] = "9"
        _write_manifest(fixture["manifest"], rows)
    elif case == "plan":
        payload["scoring_plan"]["logical_checksum"] = "wrong"
        _rewrite_chunk_and_manifest(chunk_path, fixture["manifest"], rows, payload)
    elif case == "identity":
        payload["identity"]["model_revision"] = "wrong"
        _rewrite_chunk_and_manifest(chunk_path, fixture["manifest"], rows, payload)
    elif case == "rows_checksum":
        payload["scored_rows_logical_checksum"] = "wrong"
        _rewrite_chunk_and_manifest(chunk_path, fixture["manifest"], rows, payload)
    elif case == "metadata_checksum":
        payload["chunk_metadata_logical_checksum"] = "wrong"
        _rewrite_chunk_and_manifest(chunk_path, fixture["manifest"], rows, payload)
    elif case == "artifact_checksum":
        rows[0]["chunk_artifact_sha256"] = "wrong"
        _write_manifest(fixture["manifest"], rows)
    elif case == "manifest_checksum":
        rows[0]["manifest_logical_checksum"] = "wrong"
        _write_csv(fixture["manifest"], rows)
    else:
        payload["rows"][0]["article_id"] = "wrong"
        _rewrite_chunk_and_manifest(chunk_path, fixture["manifest"], rows, payload)

    with pytest.raises(ValueError, match=message):
        certify_finbert_score_store(
            scoring_plan=fixture["plan"],
            chunk_manifest_path=fixture["manifest"],
            output_path=tmp_path / "invalid.json",
        )


@pytest.mark.parametrize(
    "field",
    [
        "canonical_corpus_identity",
        "canonical_corpus_checksum",
    ],
)
def test_changed_corpus_lineage_cannot_bind_existing_chunks(tmp_path, field):
    fixture = _scored_fixture(tmp_path / field, chunk_size=2)
    plan = json.loads(json.dumps(fixture["plan"]))
    plan[field] = "wrong"
    _resign_plan(plan)

    with pytest.raises(ValueError, match="scoring-plan binding mismatch"):
        certify_finbert_score_store(
            scoring_plan=plan,
            chunk_manifest_path=fixture["manifest"],
            output_path=tmp_path / "invalid.json",
        )


@pytest.mark.parametrize("field", ["model_revision", "tokenizer_revision"])
def test_changed_model_lineage_cannot_bind_existing_chunks(tmp_path, field):
    fixture = _scored_fixture(tmp_path / field, chunk_size=2)
    plan = json.loads(json.dumps(fixture["plan"]))
    plan["finbert_model_identity"][field] = "wrong"
    _resign_plan(plan)

    with pytest.raises(ValueError, match="identity mismatch"):
        certify_finbert_score_store(
            scoring_plan=plan,
            chunk_manifest_path=fixture["manifest"],
            output_path=tmp_path / "invalid.json",
        )


def test_certificate_satisfies_news_pit_parent_fields(tmp_path):
    fixture = _scored_fixture(tmp_path / "pit", chunk_size=2)
    result = certify_finbert_score_store(
        scoring_plan=fixture["plan"],
        chunk_manifest_path=fixture["manifest"],
        output_path=tmp_path / "store.json",
        source_commit="fixture-commit",
    )
    scored_rows = json.loads(fixture["chunk_paths"][0].read_text())["rows"]

    assert result["production_scoring_complete"] is True
    assert result["score_store_identity"]
    assert result["score_store_checksum"]
    assert result["finbert_model_identity"] == fixture["plan"]["finbert_model_identity"]
    assert result["canonical_corpus_identity"] == fixture["plan"]["canonical_corpus_identity"]
    assert result["canonical_corpus_checksum"] == fixture["plan"]["canonical_corpus_checksum"]
    assert result["scored_rows_logical_checksum"] == _hash(scored_rows)


def test_incompatible_existing_certificate_fails_closed(tmp_path):
    fixture = _scored_fixture(tmp_path / "chunks", chunk_size=2)
    output = tmp_path / "store.json"
    certify_finbert_score_store(
        scoring_plan=fixture["plan"],
        chunk_manifest_path=fixture["manifest"],
        output_path=output,
        source_commit="fixture-commit",
    )
    existing = json.loads(output.read_text())
    existing["score_store_identity"] = "wrong"
    output.write_text(json.dumps(existing), encoding="utf-8")

    with pytest.raises(FileExistsError, match="Incompatible"):
        certify_finbert_score_store(
            scoring_plan=fixture["plan"],
            chunk_manifest_path=fixture["manifest"],
            output_path=output,
            source_commit="fixture-commit",
        )


def _scored_fixture(root, *, chunk_size):
    plan, rows, adapter, config = _plan_fixture(chunk_size=chunk_size)
    paths = score_finbert_articles(
        list(reversed(rows)),
        adapter=adapter,
        output_dir=root,
        config=config,
        batch_size=chunk_size,
        scope="production",
        scoring_plan=plan,
        scored_at="2024-01-10T00:00:00+00:00",
    )
    chunks = sorted(
        (root / "chunks").glob("*.json"),
        key=lambda path: json.loads(path.read_text())["scoring_plan"]["planned_ordinal"],
    )
    return {
        "plan": plan,
        "manifest": paths.chunk_manifest_csv_path,
        "chunk_paths": chunks,
    }


def _plan_fixture(*, chunk_size=1):
    rows = [
        _article("a1", "AAPL", "growth", "2024-01-02T10:00:00Z"),
        _article("a2", "MSFT", "demand", "2024-01-02T11:00:00Z"),
    ]
    adapter = DeterministicFinBertFixtureAdapter()
    config = {"ticket": "NEWS-SCORE-1"}
    items = [
        {
            "article_id": row["article_id"],
            "symbol": row["symbol"],
            "text": select_article_text(row),
        }
        for row in rows
    ]
    config_hash = MLCoreArtifactWriter.hash_payload(config)
    chunks = []
    for ordinal, start in enumerate(range(0, len(items), chunk_size), start=1):
        selected = items[start : start + chunk_size]
        identity = finbert_chunk_identity(
            selected, adapter.identity, 256, config_hash
        )
        chunks.append(
            {
                "ordinal": ordinal,
                "chunk_id": identity["chunk_id"],
                "article_count": len(selected),
                "identity": identity,
            }
        )
    model = {
        "model_id": adapter.identity.model_id,
        "model_revision": adapter.identity.model_revision,
        "tokenizer_id": adapter.identity.tokenizer_id,
        "tokenizer_revision": adapter.identity.tokenizer_revision,
    }
    score_schema = {
        "inference_contract": FINBERT_INFERENCE_CONTRACT_VERSION,
        "fields": list(ARTICLE_LEVEL_FIELDS),
    }
    identities = [
        identity
        for chunk in chunks
        for identity in chunk["identity"]["article_identities"]
    ]
    plan = {
        "scoring_plan_contract": "stock_alpha_finbert_production_scoring_plan.v1",
        "scoring_plan_version": "v1",
        "scope": "production",
        "canonical_corpus_identity": "fixture-corpus",
        "canonical_corpus_manifest_checksum": "A" * 64,
        "canonical_corpus_checksum": "B" * 64,
        "source_canonical_rows_logical_checksum": "C" * 64,
        "eligible_article_inventory_logical_checksum": "D" * 64,
        "text_selection_contract": FINBERT_TEXT_SELECTION_CONTRACT_VERSION,
        "inference_contract": FINBERT_INFERENCE_CONTRACT_VERSION,
        "finbert_model_identity": model,
        "maximum_token_length": 256,
        "maximum_selected_text_characters": 10_000,
        "chunk_size": chunk_size,
        "score_schema": score_schema,
        "score_schema_checksum": _hash(score_schema),
        "expected_chunks": chunks,
        "expected_chunk_count": len(chunks),
        "expected_article_count": len(items),
        "first_eligible_article_identity": identities[0],
        "last_eligible_article_identity": identities[-1],
        "configuration_checksum": config_hash,
        "production_scoring_complete": False,
    }
    _resign_plan(plan)
    return plan, rows, adapter, config


def _resign_plan(plan):
    plan.pop("logical_checksum", None)
    plan.pop("plan_artifact_checksum", None)
    plan["logical_checksum"] = _hash(plan)
    plan["plan_artifact_checksum"] = _hash(plan)


def _manifest_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_manifest(path, rows):
    checksum = _chunk_manifest_logical_checksum(rows)
    for row in rows:
        row["manifest_logical_checksum"] = checksum
    _write_csv(path, rows)


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rewrite_chunk_and_manifest(chunk_path, manifest_path, rows, payload):
    chunk_path.write_text(json.dumps(payload), encoding="utf-8")
    rows[0]["chunk_artifact_sha256"] = hashlib.sha256(
        chunk_path.read_bytes()
    ).hexdigest().upper()
    _write_manifest(manifest_path, rows)


def _article(article_id, symbol, headline, timestamp):
    return {
        "article_id": article_id,
        "symbol": symbol,
        "provider": "fixture",
        "source": "wire",
        "headline": headline,
        "published_at_utc": timestamp,
        "ingested_at": timestamp,
    }


def _hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest().upper()
