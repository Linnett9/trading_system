from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.selector_dataset_lineage import logical_manifest_checksum
from core.research.ml.stock_level.paired_selector_datasets import (
    NEWS_MEMBER_FIELDS,
    PairedSelectorDatasetRequest,
    _validate_in_memory_members,
    canonical_hash,
    publish_paired_selector_datasets,
    validate_paired_selector_publication,
    verify_news_parent,
)
from core.research.ml.stock_level.selector_dataset import (
    SELECTOR_DATASET_CONTRACT_VERSION,
    SELECTOR_DATASET_MANIFEST_VERSION,
)
from core.research.ml.stock_level.selector_lineage import (
    SELECTOR_ROW_ID_CONTRACT_VERSION,
)
from core.research.ml.stock_level.stock_alpha_news_feature_store import (
    publish_pit_news_feature_store,
)


MODEL = {
    "model_id": "ProsusAI/finbert",
    "model_revision": "0123456789abcdef",
    "tokenizer_id": "ProsusAI/finbert",
    "tokenizer_revision": "fedcba9876543210",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _rows():
    common = {
        "economic_target_id": "forward_return_10d",
        "target_provenance_contract_version": "stock_level_target_provenance_v2",
        "selector_eligible": True,
        "target_status": "realized",
        "price_momentum_20d": 0.2,
        "fundamental_quality": 0.7,
    }
    return [
        {
            **common, "row_id": "01" * 32, "asset_id": "asset_AAPL",
            "symbol": "AAPL", "canonical_symbol": "AAPL",
            "decision_session_date": "2024-01-03",
            "decision_timestamp": "2024-01-04T02:00:00Z",
            "actual_forward_return_10d": 0.1,
            "target_start_timestamp": "2024-01-04T14:30:00Z",
            "label_start_timestamp": "2024-01-04T21:00:00Z",
            "label_end_timestamp": "2024-01-18T21:00:00Z",
            "label_available_timestamp": "2024-01-18T21:00:00Z",
        },
        {
            **common, "row_id": "02" * 32, "asset_id": "asset_TSLA",
            "symbol": "TSLA", "canonical_symbol": "TSLA",
            "decision_session_date": "2024-01-03",
            "decision_timestamp": "2024-01-04T02:00:00Z",
            "actual_forward_return_10d": -0.2,
            "target_start_timestamp": "2024-01-04T14:30:00Z",
            "label_start_timestamp": "2024-01-04T21:00:00Z",
            "label_end_timestamp": "2024-01-18T21:00:00Z",
            "label_available_timestamp": "2024-01-18T21:00:00Z",
        },
    ]


def _selector(root: Path, rows=None):
    root.mkdir()
    rows = list(rows or _rows())
    pq.write_table(pa.Table.from_pylist(rows), root / "rows.parquet")
    (root / "feature_schema.json").write_text('{"schema":"fixture"}')
    (root / "target_schema.json").write_text('{"target":"fixture"}')
    target = RegistryResolver(load_registry_bundle()).resolve(
        "target_contracts", "forward_return_10d", role="selector"
    )
    population = canonical_hash(sorted(
        (r["decision_session_date"], r["asset_id"], r["row_id"]) for r in rows
    ))
    checksums = {
        "rows.parquet": _sha(root / "rows.parquet"),
        "feature_schema.json": _sha(root / "feature_schema.json"),
        "target_schema.json": _sha(root / "target_schema.json"),
    }
    manifest = {
        "manifest_schema_version": SELECTOR_DATASET_MANIFEST_VERSION,
        "dataset_id": SELECTOR_DATASET_CONTRACT_VERSION + "_fixture",
        "dataset_checksum": checksums["rows.parquet"],
        "row_population_checksum": population,
        "row_count": len(rows),
        "symbol_registry_identity": "symbols",
        "symbol_registry_checksum": "SYMBOLS",
        "daily_stock_spine_identity": "spine",
        "daily_stock_spine_checksum": "SPINE",
        "daily_feature_store_identity": "price-features",
        "daily_feature_store_checksum": "FEATURES",
        "target_contract": target.canonical_id,
        "target_contract_checksum": target.entry.entry_hash,
        "economic_target_id": target.canonical_id,
        "target_provenance_contract_version": "stock_level_target_provenance_v2",
        "row_id_contract": SELECTOR_ROW_ID_CONTRACT_VERSION,
        "feature_contract": SELECTOR_DATASET_CONTRACT_VERSION,
        "feature_schema_checksum": checksums["feature_schema.json"],
        "target_schema_checksum": checksums["target_schema.json"],
        "builder_run_identity": "fixture-builder",
        "git_commit": "fixture",
        "checksums": checksums,
        "publication_status": "complete",
        "validation_status": "VERIFIED",
    }
    manifest["logical_checksum"] = logical_manifest_checksum(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return root


def _hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest().upper()


def _news(root: Path, *, spine=None):
    spine = list(spine or [{
        "asset_id": "asset_AAPL", "symbol": "AAPL",
        "decision_session_date": "2024-01-03",
        "decision_timestamp": "2024-01-04T02:00:00Z",
    }])
    articles = [{
        "article_id": "a1", "symbol": "AAPL",
        "published_at_utc": "2024-01-03T10:00:00Z",
        "collected_at_utc": "2024-01-03T11:00:00Z",
        "finbert_model_id": MODEL["model_id"],
        "finbert_model_revision": MODEL["model_revision"],
        "tokenizer_id": MODEL["tokenizer_id"],
        "tokenizer_revision": MODEL["tokenizer_revision"],
        "signed_sentiment_score": 0.7,
        "positive_probability": 0.8,
        "negative_probability": 0.1,
    }]
    aliases = {}
    parents = {
        "canonical_corpus_manifest": {
            "canonical_corpus_identity": "corpus", "canonical_corpus_checksum": "C"
        },
        "score_store_manifest": {
            "score_store_identity": "scores", "score_store_checksum": "S",
            "production_scoring_complete": True, "finbert_model_identity": MODEL,
            "scored_rows_logical_checksum": _hash(articles),
        },
        "daily_spine_manifest": {
            "daily_spine_identity": "spine", "daily_spine_checksum": "D",
            "spine_rows_logical_checksum": _hash(spine),
        },
        "ticker_mapping_manifest": {
            "ticker_mapping_identity": "mapping", "ticker_mapping_checksum": "M",
            "ticker_aliases_logical_checksum": _hash(aliases),
        },
    }
    publish_pit_news_feature_store(
        **parents, scored_articles=articles, spine_rows=spine,
        ticker_aliases=aliases, output_root=root,
        finbert_model_identity=MODEL, source_commit="fixture", lookback_days=(3,),
    )
    return root


def _fixture(tmp_path):
    return _selector(tmp_path / "selector"), _news(tmp_path / "news")


def _publish(tmp_path, **kwargs):
    selector, news = _fixture(tmp_path)
    values = {
        "selector_dataset_root": selector,
        "news_feature_store_root": news,
        "output_root": tmp_path / "pairs",
        "lookback_days": 3,
    }
    values.update(kwargs)
    return publish_paired_selector_datasets(**values)


def _member_rows(result, member):
    manifest = json.loads(
        (result.pair_root / member / "manifest.json").read_text()
    )
    return pq.read_table(result.pair_root / member / manifest["rows_path"]).to_pylist()


def test_valid_exact_matched_publication_and_missingness(tmp_path):
    result = _publish(tmp_path)
    left = _member_rows(result, "price_only")
    right = _member_rows(result, "price_plus_news")
    assert result.row_count == 2
    assert result.covered_news_row_count == 1
    assert result.missing_news_row_count == 1
    assert [r["row_id"] for r in left] == [r["row_id"] for r in right]
    assert right[1]["news_missing"] is True
    assert right[1]["eligible_article_count"] == 0
    assert right[1]["mean_signed_sentiment"] is None
    assert not any(field in left[0] for field in NEWS_MEMBER_FIELDS)
    assert result.price_only_identity != result.price_plus_news_identity
    validate_paired_selector_publication(result.pair_root)


def test_identity_stability_and_compatible_reuse(tmp_path):
    first = _publish(tmp_path)
    second = publish_paired_selector_datasets(
        selector_dataset_root=tmp_path / "selector",
        news_feature_store_root=tmp_path / "news",
        output_root=tmp_path / "pairs", lookback_days=3, reuse=True,
    )
    assert second.reused is True
    assert second.pair_identity == first.pair_identity
    assert second.pair_root == first.pair_root


def test_existing_output_requires_explicit_reuse(tmp_path):
    first = _publish(tmp_path)
    with pytest.raises(FileExistsError):
        publish_paired_selector_datasets(
            selector_dataset_root=tmp_path / "selector",
            news_feature_store_root=tmp_path / "news",
            output_root=tmp_path / "pairs", lookback_days=3,
        )
    assert first.pair_root.exists()


def test_corrupt_artifact_prevents_reuse(tmp_path):
    first = _publish(tmp_path)
    path = first.pair_root / "price_only" / "rows.parquet"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact checksum"):
        publish_paired_selector_datasets(
            selector_dataset_root=tmp_path / "selector",
            news_feature_store_root=tmp_path / "news",
            output_root=tmp_path / "pairs", lookback_days=3, reuse=True,
        )


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda rows: rows + [dict(rows[0])], "DUPLICATE_SELECTOR_ROW"),
        (lambda rows: [{**rows[0], "asset_id": ""}, *rows[1:]], "canonical asset"),
        (lambda rows: [{**rows[0], "decision_timestamp": "bad"}, *rows[1:]], "valid timestamp"),
        (lambda rows: [{**rows[0], "target_start_timestamp": "bad"}, *rows[1:]], "valid timestamp"),
        (lambda rows: [{**rows[0], "label_end_timestamp": "bad"}, *rows[1:]], "valid timestamp"),
        (lambda rows: [{**rows[0], "label_available_timestamp": "bad"}, *rows[1:]], "valid timestamp"),
    ],
)
def test_invalid_selector_population_fails_closed(tmp_path, mutation, error):
    selector = _selector(tmp_path / "selector", mutation(_rows()))
    news = _news(tmp_path / "news")
    with pytest.raises(ValueError, match=error):
        publish_paired_selector_datasets(
            selector_dataset_root=selector, news_feature_store_root=news,
            output_root=tmp_path / "pairs", lookback_days=3,
        )


def _mutate_news(root, mutate):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    partition = root / manifest["partitions"][0]["relative_path"]
    rows = [json.loads(line) for line in partition.read_text().splitlines()]
    rows = mutate(rows)
    payload = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)
    partition.write_text(payload)
    manifest["partitions"][0]["artifact_checksum"] = _sha(partition)
    manifest["partitions"][0]["row_count"] = len(rows)
    manifest["feature_store_artifact_checksum"] = _hash([
        (p["relative_path"], p["artifact_checksum"], p["row_count"])
        for p in manifest["partitions"]
    ])
    manifest.pop("logical_checksum", None)
    manifest["logical_checksum"] = _hash(manifest)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))


@pytest.mark.parametrize(
    "mutate,error",
    [
        (lambda rows: rows + [dict(rows[0])], "DUPLICATE_NEWS_JOIN_KEY"),
        (lambda rows: [{**rows[0], "symbol": "WRONG"}], "SYMBOL_MISMATCH"),
        (lambda rows: [{**rows[0], "decision_timestamp": "2024-01-05T02:00:00Z"}], "DECISION_DATE_MISMATCH"),
        (lambda rows: [{**rows[0], "latest_eligible_collection_timestamp": "2024-01-05T02:00:00Z"}], "DECISION_CUTOFF_VIOLATION"),
    ],
)
def test_invalid_news_rows_fail_closed(tmp_path, mutate, error):
    selector, news = _fixture(tmp_path)
    _mutate_news(news, mutate)
    with pytest.raises(ValueError, match=error):
        publish_paired_selector_datasets(
            selector_dataset_root=selector, news_feature_store_root=news,
            output_root=tmp_path / "pairs", lookback_days=3,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("production_finbert_scoring_proven", False),
        ("feature_store_contract", "legacy"),
        ("feature_schema_checksum", "bad"),
        ("logical_checksum", "bad"),
    ],
)
def test_uncertified_or_incomplete_news_parent_fails(tmp_path, field, value):
    _, news = _fixture(tmp_path)
    manifest = json.loads((news / "manifest.json").read_text())
    manifest[field] = value
    (news / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="NEWS_PARENT_INVALID"):
        verify_news_parent(news, 3)


def test_ambiguous_or_absent_lookback_fails(tmp_path):
    _, news = _fixture(tmp_path)
    with pytest.raises(ValueError, match="NEWS_PARENT_INVALID"):
        verify_news_parent(news, 99)
    manifest = json.loads((news / "manifest.json").read_text())
    manifest["aggregation_windows"].append(dict(manifest["aggregation_windows"][0]))
    manifest.pop("logical_checksum")
    manifest["logical_checksum"] = _hash(manifest)
    (news / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="NEWS_PARENT_INVALID"):
        verify_news_parent(news, 3)


def test_unexpected_news_rows_reported_but_do_not_create_rows(tmp_path):
    selector = _selector(tmp_path / "selector")
    news = _news(tmp_path / "news", spine=[
        {
            "asset_id": "asset_AAPL", "symbol": "AAPL",
            "decision_session_date": "2024-01-03",
            "decision_timestamp": "2024-01-04T02:00:00Z",
        },
        {
            "asset_id": "asset_EXTRA", "symbol": "EXTRA",
            "decision_session_date": "2024-01-03",
            "decision_timestamp": "2024-01-04T02:00:00Z",
        },
    ])
    result = publish_paired_selector_datasets(
        selector_dataset_root=selector, news_feature_store_root=news,
        output_root=tmp_path / "pairs", lookback_days=3,
    )
    evidence = json.loads((result.pair_root / "exclusion_report.json").read_text())
    assert result.row_count == 2
    assert evidence["reason_counts"]["UNEXPECTED_NEWS_ROW"] == 1
    assert evidence["rows_dropped_for_missing_news"] == 0


def test_pair_identity_ignores_operational_output_location(tmp_path):
    selector, news = _fixture(tmp_path)
    first = publish_paired_selector_datasets(
        selector_dataset_root=selector, news_feature_store_root=news,
        output_root=tmp_path / "one", lookback_days=3,
    )
    second = publish_paired_selector_datasets(
        selector_dataset_root=selector, news_feature_store_root=news,
        output_root=tmp_path / "two", lookback_days=3,
    )
    assert first.pair_identity == second.pair_identity


def test_request_normalisation_and_checksum_stable(tmp_path):
    request = PairedSelectorDatasetRequest(
        selector_dataset_root=str((tmp_path / "selector").resolve()),
        selector_dataset_identity="selector",
        selector_logical_manifest_checksum="S",
        selector_artifact_checksums=(("b", "2"), ("a", "1")),
        news_feature_store_root=str((tmp_path / "news").resolve()),
        news_store_identity="news", news_logical_manifest_checksum="N",
        news_schema_checksum="NS", lookback_days=3,
        output_root=str((tmp_path / "out").resolve()),
    )
    assert request.checksum == canonical_hash(request.identity_payload())
    assert request.payload()["request_contract_version"].endswith(".v1")


def test_partial_publication_not_accepted(tmp_path):
    root = tmp_path / "partial"
    root.mkdir()
    (root / "pair_manifest.json").write_text("{}")
    with pytest.raises(ValueError):
        validate_paired_selector_publication(root)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("actual_forward_return_10d", 999.0, "TARGET_INVARIANT_MISMATCH"),
        ("target_start_timestamp", "2024-01-05T14:30:00Z", "TARGET_TIMESTAMP_MISMATCH"),
        ("label_end_timestamp", "2024-01-19T21:00:00Z", "TARGET_TIMESTAMP_MISMATCH"),
        ("label_available_timestamp", "2024-01-19T21:00:00Z", "TARGET_TIMESTAMP_MISMATCH"),
        ("selector_eligible", False, "ELIGIBILITY_MISMATCH"),
        ("price_momentum_20d", 999.0, "MEMBER_POPULATION_MISMATCH"),
        ("fundamental_quality", 999.0, "MEMBER_POPULATION_MISMATCH"),
    ],
)
def test_economic_member_invariant_mismatches_fail(field, value, error):
    left = _rows()
    right = [{**row, "news_lookback_days": 3, **{
        "news_missing": True, "eligible_article_count": 0,
        "mean_signed_sentiment": None, "mean_positive_probability": None,
        "mean_negative_probability": None,
        "latest_eligible_publication_timestamp": None,
        "latest_eligible_collection_timestamp": None,
        "eligible_article_set_checksum": canonical_hash([]),
    }} for row in left]
    right[0][field] = value
    with pytest.raises(ValueError, match=error):
        _validate_in_memory_members(left, right)


def test_pair_manifest_and_evidence_are_deterministic(tmp_path):
    result = _publish(tmp_path)
    manifest = json.loads(result.pair_manifest.read_text())
    evidence = json.loads((result.pair_root / "exclusion_report.json").read_text())
    assert manifest["members"]["price_only"] == json.loads(
        (result.pair_root / "price_only" / "manifest.json").read_text()
    )
    identity = evidence.pop("report_identity")
    assert identity == canonical_hash(evidence)


def test_cli_help_and_bounded_execution(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "build_paired_selector_datasets.py"
    help_run = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True
    )
    assert help_run.returncode == 0
    selector, news = _fixture(tmp_path)
    run = subprocess.run([
        sys.executable, str(script),
        "--selector-dataset-root", str(selector),
        "--news-feature-store-root", str(news),
        "--output-root", str(tmp_path / "pairs"),
        "--lookback-days", "3",
    ], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["row_count"] == 2


def test_cli_nonzero_on_invariant_failure(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "build_paired_selector_datasets.py"
    selector, news = _fixture(tmp_path)
    run = subprocess.run([
        sys.executable, str(script),
        "--selector-dataset-root", str(selector),
        "--news-feature-store-root", str(news),
        "--output-root", str(tmp_path / "pairs"),
        "--lookback-days", "99",
    ], capture_output=True, text=True)
    assert run.returncode != 0
