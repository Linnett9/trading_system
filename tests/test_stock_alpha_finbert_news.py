from __future__ import annotations

import csv
import json
import hashlib
import math
import sys
import types

import pytest

from core.research.ml.artifacts.artifact_writers import MLCoreArtifactWriter
from core.research.ml.stock_level.stock_alpha_finbert_news import (
    DeterministicFinBertFixtureAdapter,
    FINBERT_INFERENCE_CONTRACT_VERSION,
    FINBERT_EXPOSURE_FEATURE_CONTRACT_VERSION,
    FINBERT_SELECTOR_JOIN_CONTRACT_VERSION,
    FINBERT_STOCK_FEATURE_CONTRACT_VERSION,
    FINBERT_TEXT_SELECTION_CONTRACT_VERSION,
    FinBertPrediction,
    HuggingFaceFinBertAdapter,
    build_exposure_finbert_features,
    build_stock_finbert_features,
    finbert_chunk_identity,
    join_finbert_features_for_selector,
    resolve_news_available_timestamp,
    score_finbert_articles,
    select_article_text,
    validate_finbert_prediction,
)


def test_text_selection_is_deterministic_and_hashes_selected_text():
    row = {
        "article_id": "a1",
        "headline": "  <b>Strong profit</b>  ",
        "body_or_full_text": "Strong profit  Strong profit continues.",
    }

    first = select_article_text(row)
    second = select_article_text(dict(row))

    assert first == second
    assert first.source_fields == ("headline", "body_or_full_text")
    assert len(first.text_hash) == 64
    assert "<b>" not in first.text


def test_probability_validation_requires_sum_label_and_score_formula():
    values = validate_finbert_prediction(FinBertPrediction(0.7, 0.2, 0.1, "positive"))

    assert values["signed_sentiment_score"] == pytest.approx(0.6)
    assert values["confidence"] == pytest.approx(0.7)

    with pytest.raises(ValueError, match="sum"):
        validate_finbert_prediction(FinBertPrediction(0.7, 0.2, 0.2, "positive"))
    with pytest.raises(ValueError, match="maximum"):
        validate_finbert_prediction(FinBertPrediction(0.7, 0.2, 0.1, "negative"))


def test_huggingface_adapter_uses_config_label_indices_and_inference_contract(monkeypatch):
    state = _install_fake_huggingface_stack(
        monkeypatch,
        id2label={0: "neutral", 1: "negative", 2: "positive"},
        logits=[[0.0, 1.0, 3.0], [0.0, 4.0, 1.0]],
    )

    adapter = HuggingFaceFinBertAdapter(
        model_id="local-finbert",
        tokenizer_id="local-tokenizer",
        model_revision="model-sha",
        tokenizer_revision="tokenizer-sha",
        model_path="local-model-path",
        tokenizer_path="local-tokenizer-path",
        device="cpu",
        max_token_length=7,
        local_files_only=True,
        cache_dir="cache-dir",
    )

    predictions = adapter.score_batch(["first article", "second article"])

    assert state["model"].to_device == "cpu"
    assert state["model"].eval_called is True
    assert state["model_from_pretrained"]["args"] == ("local-model-path",)
    assert state["tokenizer_from_pretrained"]["args"] == ("local-tokenizer-path",)
    assert state["no_grad_entered"] is True
    assert state["tokenizer"].calls == [
        {
            "texts": ["first article", "second article"],
            "padding": True,
            "truncation": True,
            "max_length": 7,
            "return_tensors": "pt",
        }
    ]
    assert state["encoded_tensors"]["input_ids"].devices == ["cpu"]
    assert predictions[0].sentiment_label == "positive"
    assert predictions[0].positive_probability > predictions[0].negative_probability
    assert predictions[1].sentiment_label == "negative"
    assert predictions[1].negative_probability > predictions[1].positive_probability
    assert adapter.score_batch([]) == []


def test_huggingface_adapter_fails_closed_on_ambiguous_model_labels(monkeypatch):
    _install_fake_huggingface_stack(
        monkeypatch,
        id2label={0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
        logits=[[0.0, 1.0, 3.0]],
    )

    with pytest.raises(RuntimeError, match="cannot be mapped unambiguously"):
        HuggingFaceFinBertAdapter(model_id="local-finbert")


def test_timestamp_precedence_and_date_only_publication_fails_closed():
    row = {
        "published_at_utc": "2024-01-02",
        "first_seen_at": "2024-01-03T10:00:00Z",
        "ingested_at": "2024-01-03T10:01:00Z",
    }

    resolved = resolve_news_available_timestamp(row)

    assert resolved.source == "first_seen_at"
    assert resolved.timestamp == "2024-01-03T10:00:00+00:00"
    with pytest.raises(ValueError, match="date-only"):
        resolve_news_available_timestamp({"published_at_utc": "2024-01-02"})


def test_scoring_writes_completed_chunks_and_resumes_compatible_chunks(tmp_path):
    rows = [
        _article("a1", "AAPL", "AAPL strong growth", "2024-01-02T10:00:00Z"),
        _article("a2", "MSFT", "MSFT weak demand risk", "2024-01-02T11:00:00Z"),
        {"article_id": "bad", "symbol": "MSFT", "headline": "", "ingested_at": "2024-01-02T12:00:00Z"},
    ]
    adapter = DeterministicFinBertFixtureAdapter()

    first = score_finbert_articles(
        rows,
        adapter=adapter,
        output_dir=tmp_path,
        config={"ticket": "test"},
        batch_size=1,
        scored_at="2024-01-10T00:00:00+00:00",
    )
    second = score_finbert_articles(
        rows,
        adapter=adapter,
        output_dir=tmp_path,
        config={"ticket": "test"},
        batch_size=1,
        scored_at="2024-01-10T00:00:00+00:00",
    )
    audit = json.loads(second.audit_json_path.read_text(encoding="utf-8"))

    assert first.scored_articles_csv_path.exists()
    assert second.rejected_articles_csv_path is not None
    assert audit["successfully_scored_articles"] == 2
    assert audit["failed_articles"] == 1
    assert audit["resumed_chunks"] == 2


def test_partial_chunk_is_not_reused(tmp_path):
    rows = [_article("a1", "AAPL", "AAPL strong growth", "2024-01-02T10:00:00Z")]
    adapter = DeterministicFinBertFixtureAdapter()
    score_finbert_articles(rows, adapter=adapter, output_dir=tmp_path, config={}, batch_size=1)
    chunk_path = next((tmp_path / "chunks").glob("*.json"))
    payload = json.loads(chunk_path.read_text(encoding="utf-8"))
    payload["status"] = "running"
    chunk_path.write_text(json.dumps(payload), encoding="utf-8")

    score_finbert_articles(rows, adapter=adapter, output_dir=tmp_path, config={}, batch_size=1)
    repaired = json.loads(chunk_path.read_text(encoding="utf-8"))

    assert repaired["status"] == "completed"


def test_production_scoring_requires_exact_plan_and_uses_planned_order(tmp_path):
    rows = [
        _article("a2", "MSFT", "MSFT weak demand risk", "2024-01-02T11:00:00Z"),
        _article("a1", "AAPL", "AAPL strong growth", "2024-01-02T10:00:00Z"),
    ]
    adapter = DeterministicFinBertFixtureAdapter()
    config = {"ticket": "NEWS-PLAN-2"}
    plan = _production_plan(rows, adapter, config, chunk_size=1)

    with pytest.raises(ValueError, match="requires an authoritative plan"):
        score_finbert_articles(
            rows, adapter=adapter, output_dir=tmp_path / "missing",
            config=config, batch_size=1, scope="production",
        )

    paths = score_finbert_articles(
        rows, adapter=adapter, output_dir=tmp_path / "planned",
        config=config, batch_size=1, scope="production", scoring_plan=plan,
    )
    chunk_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((tmp_path / "planned" / "chunks").glob("*.json"))
    ]
    scored_rows = paths.scored_articles_csv_path.read_text(encoding="utf-8").splitlines()

    assert scored_rows[1].startswith("a1,")
    assert all(payload["production_scope"] is True for payload in chunk_payloads)
    assert {
        payload["scoring_plan"]["logical_checksum"] for payload in chunk_payloads
    } == {plan["logical_checksum"]}
    assert sorted(
        payload["scoring_plan"]["planned_ordinal"] for payload in chunk_payloads
    ) == [1, 2]
    with paths.chunk_manifest_csv_path.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        manifest_rows = list(csv.DictReader(handle))
    for payload, artifact in zip(
        sorted(chunk_payloads, key=lambda row: row["scoring_plan"]["planned_ordinal"]),
        sorted(
            (tmp_path / "planned" / "chunks").glob("*.json"),
            key=lambda path: next(
                row["scoring_plan"]["planned_ordinal"]
                for row in chunk_payloads
                if row["identity"]["chunk_id"] == path.stem
            ),
        ),
    ):
        evidence = next(
            row for row in manifest_rows
            if row["chunk_id"] == payload["identity"]["chunk_id"]
        )
        assert payload["scored_rows_logical_checksum"]
        assert payload["chunk_metadata_logical_checksum"]
        assert evidence["chunk_artifact_sha256"] == hashlib.sha256(
            artifact.read_bytes()
        ).hexdigest().upper()
        assert (
            evidence["scored_rows_logical_checksum"]
            == payload["scored_rows_logical_checksum"]
        )
        assert (
            evidence["chunk_metadata_logical_checksum"]
            == payload["chunk_metadata_logical_checksum"]
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda plan: plan.__setitem__("chunk_size", 2), "limits mismatch"),
        (
            lambda plan: plan["finbert_model_identity"].__setitem__(
                "model_revision", "wrong"
            ),
            "model or tokenizer identity mismatch",
        ),
        (
            lambda plan: plan["finbert_model_identity"].__setitem__(
                "tokenizer_revision", "wrong"
            ),
            "model or tokenizer identity mismatch",
        ),
        (
            lambda plan: plan.__setitem__("configuration_checksum", "wrong"),
            "configuration mismatch",
        ),
        (
            lambda plan: plan["expected_chunks"][0].__setitem__("identity", {}),
            "chunk identity",
        ),
    ],
)
def test_production_scoring_rejects_incompatible_plan(
    tmp_path, mutation, message
):
    rows = [_article("a1", "AAPL", "growth", "2024-01-02T10:00:00Z")]
    adapter = DeterministicFinBertFixtureAdapter()
    config = {"ticket": "NEWS-PLAN-2"}
    plan = _production_plan(rows, adapter, config, chunk_size=1)
    mutation(plan)
    _resign_plan(plan)

    with pytest.raises(ValueError, match=message):
        score_finbert_articles(
            rows, adapter=adapter, output_dir=tmp_path, config=config,
            batch_size=1, scope="production", scoring_plan=plan,
        )


@pytest.mark.parametrize("change", ["missing", "unexpected"])
def test_production_scoring_rejects_runtime_inventory_drift(tmp_path, change):
    planned = [
        _article("a1", "AAPL", "growth", "2024-01-02T10:00:00Z"),
        _article("a2", "MSFT", "demand", "2024-01-02T11:00:00Z"),
    ]
    runtime = planned[:1] if change == "missing" else [
        *planned,
        _article("a3", "TSLA", "deliveries", "2024-01-02T12:00:00Z"),
    ]
    adapter = DeterministicFinBertFixtureAdapter()
    config = {"ticket": "NEWS-PLAN-2"}
    plan = _production_plan(planned, adapter, config, chunk_size=1)

    with pytest.raises(ValueError, match="inventory mismatch"):
        score_finbert_articles(
            runtime, adapter=adapter, output_dir=tmp_path, config=config,
            batch_size=1, scope="production", scoring_plan=plan,
        )


def test_production_resume_skips_compatible_and_fails_closed_on_artifacts(
    tmp_path, monkeypatch
):
    rows = [_article("a1", "AAPL", "growth", "2024-01-02T10:00:00Z")]
    adapter = DeterministicFinBertFixtureAdapter()
    config = {"ticket": "NEWS-PLAN-2"}
    plan = _production_plan(rows, adapter, config, chunk_size=1)
    score_finbert_articles(
        rows, adapter=adapter, output_dir=tmp_path, config=config,
        batch_size=1, scope="production", scoring_plan=plan,
    )
    monkeypatch.setattr(
        adapter, "score_batch",
        lambda texts: pytest.fail("compatible production chunk was rescored"),
    )
    resumed = score_finbert_articles(
        rows, adapter=adapter, output_dir=tmp_path, config=config,
        batch_size=1, scope="production", scoring_plan=plan,
    )
    audit = json.loads(resumed.audit_json_path.read_text(encoding="utf-8"))
    assert audit["resumed_chunks"] == 1

    chunk_path = next((tmp_path / "chunks").glob("*.json"))
    incompatible = json.loads(chunk_path.read_text(encoding="utf-8"))
    incompatible["scoring_plan"]["logical_checksum"] = "wrong"
    chunk_path.write_text(json.dumps(incompatible), encoding="utf-8")
    with pytest.raises(ValueError, match="scoring-plan binding"):
        score_finbert_articles(
            rows, adapter=adapter, output_dir=tmp_path, config=config,
            batch_size=1, scope="production", scoring_plan=plan,
        )
    chunk_path.write_text(json.dumps({
        **incompatible,
        "scoring_plan": {
            "logical_checksum": plan["logical_checksum"],
            "plan_artifact_checksum": plan["plan_artifact_checksum"],
            "planned_ordinal": 1,
        },
    }), encoding="utf-8")
    (tmp_path / "chunks" / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected production"):
        score_finbert_articles(
            rows, adapter=adapter, output_dir=tmp_path, config=config,
            batch_size=1, scope="production", scoring_plan=plan,
        )


def test_production_chunk_checksums_are_deterministic_and_tamper_evident(
    tmp_path
):
    rows = [_article("a1", "AAPL", "growth", "2024-01-02T10:00:00Z")]
    adapter = DeterministicFinBertFixtureAdapter()
    config = {"ticket": "NEWS-CHUNK-1"}
    plan = _production_plan(rows, adapter, config, chunk_size=1)
    outputs = []
    for name in ("left", "right"):
        score_finbert_articles(
            rows, adapter=adapter, output_dir=tmp_path / name, config=config,
            batch_size=1, scope="production", scoring_plan=plan,
            scored_at="2024-01-10T00:00:00+00:00",
        )
        outputs.append(json.loads(
            next((tmp_path / name / "chunks").glob("*.json")).read_text()
        ))
    assert (
        outputs[0]["scored_rows_logical_checksum"]
        == outputs[1]["scored_rows_logical_checksum"]
    )
    assert (
        outputs[0]["chunk_metadata_logical_checksum"]
        == outputs[1]["chunk_metadata_logical_checksum"]
    )

    chunk_path = next((tmp_path / "left" / "chunks").glob("*.json"))
    tampered = json.loads(chunk_path.read_text())
    tampered["rows"][0]["positive_probability"] = 0.123
    chunk_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="scored-row checksum"):
        score_finbert_articles(
            rows, adapter=adapter, output_dir=tmp_path / "left", config=config,
            batch_size=1, scope="production", scoring_plan=plan,
        )


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    [
        (
            "label",
            lambda payload: payload["rows"][0].__setitem__(
                "sentiment_label", "negative"
            ),
            "scored-row checksum",
        ),
        (
            "article",
            lambda payload: payload["rows"][0].__setitem__("article_id", "wrong"),
            "row ownership",
        ),
        (
            "metadata",
            lambda payload: payload["scoring_plan"].__setitem__(
                "planned_ordinal", 99
            ),
            "scoring-plan binding",
        ),
        (
            "missing_checksum",
            lambda payload: payload.pop("scored_rows_logical_checksum"),
            "scored-row checksum",
        ),
    ],
)
def test_production_chunk_logical_tampering_fails_resume(
    tmp_path, name, mutate, message
):
    rows = [_article("a1", "AAPL", "growth", "2024-01-02T10:00:00Z")]
    adapter = DeterministicFinBertFixtureAdapter()
    config = {"ticket": "NEWS-CHUNK-1"}
    plan = _production_plan(rows, adapter, config, chunk_size=1)
    output = tmp_path / name
    score_finbert_articles(
        rows, adapter=adapter, output_dir=output, config=config,
        batch_size=1, scope="production", scoring_plan=plan,
    )
    chunk_path = next((output / "chunks").glob("*.json"))
    payload = json.loads(chunk_path.read_text())
    mutate(payload)
    chunk_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        score_finbert_articles(
            rows, adapter=adapter, output_dir=output, config=config,
            batch_size=1, scope="production", scoring_plan=plan,
        )


def test_production_chunk_exact_bytes_and_row_order_are_tamper_evident(tmp_path):
    rows = [
        _article("a1", "AAPL", "growth", "2024-01-02T10:00:00Z"),
        _article("a2", "MSFT", "demand", "2024-01-02T11:00:00Z"),
    ]
    adapter = DeterministicFinBertFixtureAdapter()
    config = {"ticket": "NEWS-CHUNK-1"}
    plan = _production_plan(rows, adapter, config, chunk_size=2)
    for name, reorder in (("bytes", False), ("order", True)):
        output = tmp_path / name
        score_finbert_articles(
            rows, adapter=adapter, output_dir=output, config=config,
            batch_size=2, scope="production", scoring_plan=plan,
        )
        chunk_path = next((output / "chunks").glob("*.json"))
        if reorder:
            payload = json.loads(chunk_path.read_text())
            payload["rows"].reverse()
            chunk_path.write_text(json.dumps(payload), encoding="utf-8")
            message = "row ownership"
        else:
            chunk_path.write_bytes(chunk_path.read_bytes() + b"\n")
            message = "artifact checksum"
        with pytest.raises(ValueError, match=message):
            score_finbert_articles(
                rows, adapter=adapter, output_dir=output, config=config,
                batch_size=2, scope="production", scoring_plan=plan,
            )


def test_production_chunk_requires_external_evidence_and_retries_incomplete(
    tmp_path
):
    rows = [_article("a1", "AAPL", "growth", "2024-01-02T10:00:00Z")]
    adapter = DeterministicFinBertFixtureAdapter()
    config = {"ticket": "NEWS-CHUNK-1"}
    plan = _production_plan(rows, adapter, config, chunk_size=1)
    score_finbert_articles(
        rows, adapter=adapter, output_dir=tmp_path, config=config,
        batch_size=1, scope="production", scoring_plan=plan,
    )
    manifest_path = tmp_path / "finbert_chunk_manifest.csv"
    manifest_path.unlink()
    with pytest.raises(ValueError, match="external checksum evidence"):
        score_finbert_articles(
            rows, adapter=adapter, output_dir=tmp_path, config=config,
            batch_size=1, scope="production", scoring_plan=plan,
        )

    chunk_path = next((tmp_path / "chunks").glob("*.json"))
    partial = json.loads(chunk_path.read_text())
    partial["status"] = "running"
    chunk_path.write_text(json.dumps(partial), encoding="utf-8")
    repaired = score_finbert_articles(
        rows, adapter=adapter, output_dir=tmp_path, config=config,
        batch_size=1, scope="production", scoring_plan=plan,
    )
    assert repaired.chunk_manifest_csv_path.exists()
    assert json.loads(chunk_path.read_text())["status"] == "completed"


def test_smoke_scope_remains_explicitly_nonproduction(tmp_path):
    rows = [_article("a1", "AAPL", "growth", "2024-01-02T10:00:00Z")]
    adapter = DeterministicFinBertFixtureAdapter()
    paths = score_finbert_articles(
        rows, adapter=adapter, output_dir=tmp_path, config={}, batch_size=1,
        scope="smoke",
    )
    chunk = json.loads(next((tmp_path / "chunks").glob("*.json")).read_text())
    audit = json.loads(paths.audit_json_path.read_text(encoding="utf-8"))

    assert chunk["production_scope"] is False
    assert chunk["scoring_plan"] is None
    assert audit["production_planned"] is False


def test_stock_aggregation_excludes_future_news_and_distinguishes_no_news():
    scored = [
        _scored("a1", "AAPL", "2024-01-02T10:00:00+00:00", 0.8, 0.1, 0.1, "positive"),
        _scored("a2", "AAPL", "2024-01-05T10:00:00+00:00", 0.1, 0.1, 0.8, "negative"),
        _scored("a3", "MSFT", "2024-01-02T10:00:00+00:00", 0.1, 0.8, 0.1, "neutral"),
    ]
    decisions = [
        {"rebalance_date": "2024-01-04T21:00:00+00:00", "symbol": "AAPL", "predicted_momentum_20d": "-0.05"},
        {"rebalance_date": "2024-01-04T21:00:00+00:00", "symbol": "TSLA", "predicted_momentum_20d": "0.02"},
    ]

    features, audit = build_stock_finbert_features(scored, decisions, lookback_days=(3,))

    aapl = next(row for row in features if row["symbol"] == "AAPL")
    tsla = next(row for row in features if row["symbol"] == "TSLA")
    assert aapl["finbert_article_count"] == 1
    assert tsla["finbert_news_coverage_indicator"] == 0
    assert audit["future_article_exclusion_count"] == 1
    assert aapl["finbert_news_feature_contract_version"] == FINBERT_STOCK_FEATURE_CONTRACT_VERSION


def test_contrarian_features_use_price_conditioning_not_negative_sentiment_only():
    scored = [_scored("a1", "AAPL", "2024-01-02T10:00:00+00:00", 0.05, 0.05, 0.9, "negative")]
    down, _ = build_stock_finbert_features(
        scored,
        [{"rebalance_date": "2024-01-03T21:00:00+00:00", "symbol": "AAPL", "predicted_momentum_20d": "-0.20"}],
        lookback_days=(3,),
    )
    up, _ = build_stock_finbert_features(
        scored,
        [{"rebalance_date": "2024-01-03T21:00:00+00:00", "symbol": "AAPL", "predicted_momentum_20d": "0.20"}],
        lookback_days=(3,),
    )

    assert down[0]["contrarian_potential_overreaction"] > up[0]["contrarian_potential_overreaction"]


def test_selector_join_disabled_is_unchanged_and_enabled_changes_identity():
    rows = [{"rebalance_date": "2024-01-04", "symbol": "AAPL", "price_feature": "1.0"}]
    features = [
        {
            **{column: 0 for column in _stock_feature_columns()},
            "decision_timestamp": "2024-01-04T21:00:00+00:00",
            "symbol": "AAPL",
            "finbert_news_coverage_indicator": 1,
        }
    ]

    disabled, disabled_audit = join_finbert_features_for_selector(rows, features, include_news=False)
    enabled, enabled_audit = join_finbert_features_for_selector(rows, features, include_news=True)

    assert disabled == rows
    assert disabled_audit["model_input_identity"] == "price_only"
    assert enabled[0]["finbert_news_coverage_indicator"] == 1
    assert enabled_audit["contract_version"] == FINBERT_SELECTOR_JOIN_CONTRACT_VERSION
    assert enabled_audit["model_input_identity"] != "price_only"


def test_exposure_weighted_aggregation_and_disabled_identity():
    holdings = [
        {"rebalance_date": "2024-01-04", "symbol": "AAPL", "weight": "0.6"},
        {"rebalance_date": "2024-01-04", "symbol": "MSFT", "weight": "0.4"},
    ]
    features = [
        _stock_feature("2024-01-04", "AAPL", 7, sentiment=-0.5, negative=0.8, high_neg=1),
        _stock_feature("2024-01-04", "MSFT", 7, sentiment=0.25, negative=0.1, high_neg=0),
    ]

    disabled, disabled_audit = build_exposure_finbert_features(holdings, features, include_news=False)
    enabled, enabled_audit = build_exposure_finbert_features(holdings, features, include_news=True)

    assert disabled == holdings
    assert disabled_audit["model_input_identity"] == "price_only"
    assert enabled[0]["portfolio_finbert_weighted_mean_sentiment"] == pytest.approx(-0.2)
    assert enabled[0]["finbert_exposure_feature_contract_version"] == FINBERT_EXPOSURE_FEATURE_CONTRACT_VERSION
    assert enabled_audit["portfolio_decisions_with_any_covered_holding"] == 1


def _production_plan(rows, adapter, config, *, chunk_size):
    items = []
    for row in rows:
        items.append(
            {
                "article_id": row["article_id"],
                "symbol": row["symbol"],
                "text": select_article_text(row),
            }
        )
    items.sort(
        key=lambda item: (
            item["article_id"], item["symbol"], item["text"].text_hash
        )
    )
    config_hash = MLCoreArtifactWriter.hash_payload(config)
    chunks = []
    for ordinal, start in enumerate(range(0, len(items), chunk_size), start=1):
        identity = finbert_chunk_identity(
            items[start : start + chunk_size],
            adapter.identity,
            256,
            config_hash,
        )
        chunks.append(
            {
                "ordinal": ordinal,
                "chunk_id": identity["chunk_id"],
                "article_count": len(items[start : start + chunk_size]),
                "identity": identity,
            }
        )
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
        "finbert_model_identity": {
            "model_id": adapter.identity.model_id,
            "model_revision": adapter.identity.model_revision,
            "tokenizer_id": adapter.identity.tokenizer_id,
            "tokenizer_revision": adapter.identity.tokenizer_revision,
        },
        "maximum_token_length": 256,
        "maximum_selected_text_characters": 10_000,
        "chunk_size": chunk_size,
        "expected_chunks": chunks,
        "expected_chunk_count": len(chunks),
        "expected_article_count": len(items),
        "configuration_checksum": config_hash,
        "production_scoring_complete": False,
    }
    _resign_plan(plan)
    return plan


def _resign_plan(plan):
    plan.pop("logical_checksum", None)
    plan.pop("plan_artifact_checksum", None)
    plan["logical_checksum"] = _test_hash(plan)
    plan["plan_artifact_checksum"] = _test_hash(plan)


def _test_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest().upper()


def _article(article_id: str, symbol: str, headline: str, timestamp: str) -> dict[str, str]:
    return {
        "article_id": article_id,
        "symbol": symbol,
        "provider": "fixture",
        "source": "wire",
        "headline": headline,
        "published_at_utc": timestamp,
        "ingested_at": timestamp,
    }


def _scored(
    article_id: str,
    symbol: str,
    timestamp: str,
    positive: float,
    neutral: float,
    negative: float,
    label: str,
) -> dict[str, object]:
    return {
        "article_id": article_id,
        "symbol": symbol,
        "provider": "fixture",
        "source": "wire",
        "selected_text_hash": article_id,
        "news_available_timestamp": timestamp,
        "finbert_model_id": "fixture",
        "positive_probability": positive,
        "neutral_probability": neutral,
        "negative_probability": negative,
        "signed_sentiment_score": positive - negative,
        "sentiment_label": label,
        "confidence": max(positive, neutral, negative),
    }


def _stock_feature(date: str, symbol: str, lookback: int, *, sentiment: float, negative: float, high_neg: int) -> dict[str, object]:
    row = {column: 0 for column in _stock_feature_columns()}
    row.update(
        {
            "decision_timestamp": date,
            "symbol": symbol,
            "finbert_lookback_days": lookback,
            "finbert_news_coverage_indicator": 1,
            "finbert_mean_signed_sentiment": sentiment,
            "finbert_negative_probability_mean": negative,
            "finbert_high_confidence_negative_count": high_neg,
            "finbert_sentiment_disagreement": abs(sentiment),
        }
    )
    return row


def _stock_feature_columns() -> tuple[str, ...]:
    from core.research.ml.stock_level.stock_alpha_finbert_news import STOCK_NEWS_FEATURE_COLUMNS

    return STOCK_NEWS_FEATURE_COLUMNS


def _install_fake_huggingface_stack(
    monkeypatch: pytest.MonkeyPatch,
    *,
    id2label: dict[int, str],
    logits: list[list[float]],
) -> dict[str, object]:
    state: dict[str, object] = {}

    class FakeTensor:
        def __init__(self, name: str) -> None:
            self.name = name
            self.devices: list[str] = []

        def to(self, device: str) -> "FakeTensor":
            self.devices.append(device)
            return self

    class FakeProbabilities:
        def __init__(self, values: list[list[float]]) -> None:
            self._values = values

        def detach(self) -> "FakeProbabilities":
            return self

        def cpu(self) -> "FakeProbabilities":
            return self

        def tolist(self) -> list[list[float]]:
            return self._values

    class FakeNoGrad:
        def __enter__(self) -> None:
            state["no_grad_entered"] = True

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

    class FakeTorch(types.ModuleType):
        def __init__(self) -> None:
            super().__init__("torch")
            self.cuda = types.SimpleNamespace(is_available=lambda: False)

        def no_grad(self) -> FakeNoGrad:
            return FakeNoGrad()

        def softmax(self, values: list[list[float]], dim: int = -1) -> FakeProbabilities:
            assert dim == -1
            probabilities = []
            for row in values:
                exponentials = [math.exp(value) for value in row]
                denominator = sum(exponentials)
                probabilities.append([value / denominator for value in exponentials])
            return FakeProbabilities(probabilities)

    class FakeTokenizer:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def __call__(
            self,
            texts: list[str],
            *,
            padding: bool,
            truncation: bool,
            max_length: int,
            return_tensors: str,
        ) -> dict[str, FakeTensor]:
            self.calls.append(
                {
                    "texts": texts,
                    "padding": padding,
                    "truncation": truncation,
                    "max_length": max_length,
                    "return_tensors": return_tensors,
                }
            )
            encoded = {"input_ids": FakeTensor("input_ids"), "attention_mask": FakeTensor("attention_mask")}
            state["encoded_tensors"] = encoded
            return encoded

    class FakeModel:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(id2label=id2label)
            self.to_device: str | None = None
            self.eval_called = False

        def to(self, device: str) -> "FakeModel":
            self.to_device = device
            return self

        def eval(self) -> None:
            self.eval_called = True

        def __call__(self, **kwargs: FakeTensor) -> object:
            state["model_call_kwargs"] = kwargs
            return types.SimpleNamespace(logits=logits)

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeTokenizer:
            tokenizer = FakeTokenizer()
            state["tokenizer"] = tokenizer
            state["tokenizer_from_pretrained"] = {"args": args, "kwargs": kwargs}
            return tokenizer

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeModel:
            model = FakeModel()
            state["model"] = model
            state["model_from_pretrained"] = {"args": args, "kwargs": kwargs}
            return model

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForSequenceClassification = FakeAutoModel
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return state
