from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from core.research.ml.artifacts.artifact_writers import MLCoreArtifactWriter
from core.research.ml.stock_level.news_sources.historical_canonical_corpus import (
    CANONICAL_CORPUS_CSV,
    materialize_historical_canonical_corpus,
    sha256_file,
)
from core.research.ml.stock_level.stock_alpha_finbert_news import (
    FinBertModelIdentity,
    finbert_chunk_identity,
    select_article_text,
)
from core.research.ml.stock_level.stock_alpha_finbert_scoring_plan import (
    publish_finbert_scoring_plan,
)


MODEL = FinBertModelIdentity(
    model_id="ProsusAI/finbert",
    model_revision="0123456789abcdef",
    tokenizer_id="ProsusAI/finbert",
    tokenizer_revision="fedcba9876543210",
    inference_device="plan-only",
)


def _assembly(article_id, symbol, headline):
    return {
        "provider": "provider",
        "article_id": article_id,
        "provider_article_id": article_id,
        "provider_original_article_id": article_id,
        "provider_symbols": symbol,
        "symbol": symbol,
        "published_at_utc": "2024-01-02T14:30:00Z",
        "collected_at_utc": "2024-01-02T15:00:00Z",
        "headline": headline,
        "summary": f"{headline} summary",
        "source": "source",
    }


def _corpus(tmp_path, rows):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.csv"
    metadata = tmp_path / "source.json"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checksum = sha256_file(source)
    metadata.write_text(
        json.dumps({"assembly_checksum": checksum}), encoding="utf-8"
    )
    root = tmp_path / "corpus"
    manifest = materialize_historical_canonical_corpus(
        source_assembly_csv_path=source,
        source_assembly_metadata_json_path=metadata,
        output_dir=root,
        expected_source_checksum=checksum,
        write_enabled=True,
        ingested_at_utc="2026-07-18T00:00:00Z",
    )
    corpus_path = root / CANONICAL_CORPUS_CSV
    with corpus_path.open("r", encoding="utf-8", newline="") as handle:
        canonical_rows = list(csv.DictReader(handle))
    return manifest, corpus_path, canonical_rows


def _publish(tmp_path, rows=None, **updates):
    rows = rows or [
        _assembly("a3", "MSFT", "third"),
        _assembly("a1", "AAPL", "first"),
        _assembly("a2", "AAPL", "second"),
    ]
    manifest, corpus_path, canonical_rows = _corpus(tmp_path, rows)
    values = {
        "corpus_manifest": manifest,
        "corpus_path": corpus_path,
        "canonical_rows": canonical_rows,
        "output_path": tmp_path / "plan.json",
        "model_identity": MODEL,
        "scoring_config": {"scope": "production", "seed": 7},
        "chunk_size": 2,
        "source_commit": "abc123",
    }
    values.update(updates)
    return publish_finbert_scoring_plan(**values), values


def test_deterministic_ordered_plan_and_scorer_identity_match(tmp_path):
    first, values = _publish(tmp_path)
    assert first["scope"] == "production"
    assert first["production_scoring_complete"] is False
    assert first["expected_article_count"] == 3
    assert first["expected_chunk_count"] == 2
    assert [row["ordinal"] for row in first["expected_chunks"]] == [1, 2]
    assert first["first_eligible_article_identity"]["article_id"] == "a1"
    assert first["last_eligible_article_identity"]["article_id"] == "a3"

    ordered = sorted(
        values["canonical_rows"],
        key=lambda row: (
            row["provider_article_id"], row["symbol"],
            select_article_text(row).text_hash,
        ),
    )
    items = [
        {
            "article_id": row["provider_article_id"],
            "symbol": row["symbol"],
            "text": select_article_text(row),
        }
        for row in ordered[:2]
    ]
    expected = finbert_chunk_identity(
        items,
        MODEL,
        256,
        MLCoreArtifactWriter.hash_payload(values["scoring_config"]),
    )
    assert first["expected_chunks"][0]["identity"] == expected
    assert first["model_loading_invoked"] is False
    assert first["inference_invoked"] is False

    second = publish_finbert_scoring_plan(**values)
    assert second["logical_checksum"] == first["logical_checksum"]
    assert second["publication_result"] == "SKIPPED_COMPATIBLE"


def test_reordered_input_same_plan_and_changes_affect_identity(tmp_path):
    rows = [
        _assembly("a1", "AAPL", "first"),
        _assembly("a2", "MSFT", "second"),
    ]
    left, values = _publish(tmp_path / "left", rows=rows)
    reordered_values = {
        **values,
        "canonical_rows": list(reversed(values["canonical_rows"])),
        "output_path": tmp_path / "reordered-plan.json",
    }
    right = publish_finbert_scoring_plan(**reordered_values)
    assert right["logical_checksum"] == left["logical_checksum"]
    changed, _ = _publish(
        tmp_path / "changed",
        rows=[{**rows[0], "headline": "changed"}, rows[1]],
    )
    assert changed["logical_checksum"] != left["logical_checksum"]
    resized, _ = _publish(
        tmp_path / "resized", rows=rows, chunk_size=1
    )
    assert resized["logical_checksum"] != left["logical_checksum"]


def test_duplicate_unpinned_scope_and_inventory_mismatch_fail_closed(tmp_path):
    duplicate = [
        _assembly("same", "AAPL", "one"),
        _assembly("same", "AAPL", "two"),
    ]
    with pytest.raises(ValueError, match="Duplicate canonical article"):
        _publish(tmp_path / "duplicate", rows=duplicate)
    with pytest.raises(ValueError, match="must be pinned"):
        _publish(
            tmp_path / "unpinned",
            model_identity=FinBertModelIdentity(
                "model", "main", "tokenizer", "revision", "plan-only"
            ),
        )
    with pytest.raises(ValueError, match="production scope"):
        _publish(tmp_path / "smoke", scope="smoke")

    _, values = _publish(tmp_path / "mismatch-source")
    values["canonical_rows"] = [
        {**values["canonical_rows"][0], "headline": "tampered"},
        *values["canonical_rows"][1:],
    ]
    values["output_path"] = tmp_path / "mismatch.json"
    with pytest.raises(ValueError, match="inventory checksum mismatch"):
        publish_finbert_scoring_plan(**values)


def test_changed_model_tokenizer_and_incompatible_existing_plan(tmp_path):
    first, values = _publish(tmp_path / "base")
    changed_model = FinBertModelIdentity(
        MODEL.model_id,
        "different-pinned-model",
        MODEL.tokenizer_id,
        MODEL.tokenizer_revision,
        "plan-only",
    )
    changed, _ = _publish(
        tmp_path / "changed-model", model_identity=changed_model
    )
    assert changed["logical_checksum"] != first["logical_checksum"]
    changed_tokenizer = FinBertModelIdentity(
        MODEL.model_id,
        MODEL.model_revision,
        MODEL.tokenizer_id,
        "different-pinned-tokenizer",
        "plan-only",
    )
    tokenized, _ = _publish(
        tmp_path / "changed-tokenizer", model_identity=changed_tokenizer
    )
    assert tokenized["logical_checksum"] != first["logical_checksum"]

    values["output_path"].write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Incompatible existing"):
        publish_finbert_scoring_plan(**values)


def test_legacy_or_corrupt_corpus_and_model_execution_are_rejected(tmp_path):
    _, values = _publish(tmp_path / "valid")
    legacy = dict(values["corpus_manifest"])
    legacy["schema_version"] = "stock_alpha_news.historical_canonical_corpus.v1"
    values["corpus_manifest"] = legacy
    values["output_path"] = tmp_path / "legacy.json"
    with pytest.raises(ValueError, match="v2 is required"):
        publish_finbert_scoring_plan(**values)

    _, corrupt = _publish(tmp_path / "corrupt")
    corrupt["corpus_path"].write_text("corrupt\n", encoding="utf-8")
    corrupt["output_path"] = tmp_path / "corrupt-plan.json"
    with pytest.raises(ValueError, match="ARTIFACT_CHECKSUM_MISMATCH"):
        publish_finbert_scoring_plan(**corrupt)

    source = Path(
        "core/research/ml/stock_level/stock_alpha_finbert_scoring_plan.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "HuggingFaceFinBertAdapter",
        "score_finbert_articles",
        "score_batch",
        "from_pretrained",
        "transformers",
        "torch",
    ):
        assert forbidden not in source
