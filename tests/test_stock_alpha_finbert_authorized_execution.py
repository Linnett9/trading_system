from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

import core.research.ml.stock_level.stock_alpha_finbert_authorized_execution as owner
from core.research.ml.stock_level.stock_alpha_finbert_authorized_execution import (
    COMPLETE_STATUS,
    ExecutionCallbacks,
    ExecutionFailure,
    ProductionExecutionFactory,
    immutable_model_activation_options,
    execute_authorized_scoring,
)
from core.research.ml.stock_level.stock_alpha_finbert_news import (
    FinBertModelIdentity, FinBertPrediction,
)
from core.research.ml.stock_level.stock_alpha_finbert_scoring_plan import (
    build_eligible_scoring_inventory,
)
from core.research.ml.stock_level.stock_alpha_finbert_scoring_foundation import (
    authorization_template,
    deterministic_run_id,
    logical_identity,
)
from tests.test_stock_alpha_finbert_scoring_foundation import (
    REVISION, _package, _request,
)
from scripts.run_authorized_stock_alpha_finbert_scoring import main as cli_main


@dataclass
class FakeLease:
    events: list
    closed: bool = False

    def release_success(self):
        self.events.append("lease_success")

    def close_failure(self, code):
        self.closed = True
        self.events.append("lease_failure:" + code)


def _fixture(tmp_path, monkeypatch, *, chunks=2):
    package = _package(tmp_path)
    request = _request(tmp_path, package)
    request.update({
        "canonical_row_count": chunks + 1,
        "expected_eligible_row_count": chunks,
        "expected_excluded_row_count": 1,
        "chunk_size": 1,
    })
    request["scoring_request_identity"] = logical_identity({
        key: value for key, value in request.items()
        if key != "scoring_request_identity"
    })
    runtime = {"runtime_contract": "stock_alpha_news_finbert_scoring_runtime.v1"}
    runtime_checksum = logical_identity(runtime)
    run_id = deterministic_run_id(request, runtime_checksum)
    authorization = authorization_template(
        request, runtime_checksum=runtime_checksum, expected_run_id=run_id
    )
    plan = {
        "scoring_request_identity": request["scoring_request_identity"],
        "expected_eligible_row_count": chunks, "chunk_size": 1,
        "expected_chunk_count": chunks,
        "chunks": [
            {
                "ordinal": index + 1, "start_inclusive": index,
                "stop_exclusive": index + 1, "row_count": 1,
                "chunk_identity": f"chunk-{index}", "status": "PLANNED",
            }
            for index in range(chunks)
        ],
    }
    plan["chunk_plan_identity"] = logical_identity(plan)
    monkeypatch.setattr(
        owner, "validate_model_package",
        lambda *args, **kwargs: package,
    )
    return request, runtime, runtime_checksum, authorization, plan


def _callbacks(events, request, *, fail_at=None, reuse_first=False):
    lease = FakeLease(events)

    def result(chunk):
        return {
            "status": "COMPLETE",
            "chunk_identity": chunk["chunk_identity"],
            "output_sha256": "a" * 64,
            "logical_checksum": "b" * 64,
            "row_count": chunk["row_count"],
            "scoring_request_identity": request["scoring_request_identity"],
        }

    def infer(chunk, source, tokenizer, model):
        del source, tokenizer, model
        events.append("infer:" + chunk["chunk_identity"])
        if fail_at == chunk["chunk_identity"]:
            raise RuntimeError("synthetic")
        return [{
            "article_id": chunk["chunk_identity"], "symbol": "AAPL",
            "positive_probability": 0.7, "negative_probability": 0.1,
            "neutral_probability": 0.2, "sentiment_label": "positive",
            "model_package_identity": request["model_package_identity"],
            "canonical_identity": request["canonical_identity"],
            "chunk_identity": chunk["chunk_identity"],
            "scored_at": request["scored_at_utc"],
        }]

    return ExecutionCallbacks(
        persist_resource_request=lambda run: events.append("persist"),
        acquire_lease=lambda run: events.append("lease") or lease,
        activate_source=lambda: events.append("source") or {
            "canonical_row_count": request["canonical_row_count"],
            "eligible_row_count": request["expected_eligible_row_count"],
            "excluded_row_count": 1,
            "exclusions": [{
                "owner_identity": "excluded-owner",
                "reason_code": "NO_SELECTABLE_TEXT",
            }],
        },
        activate_tokenizer=lambda: events.append("tokenizer") or object(),
        activate_model=lambda: events.append("model") or object(),
        compatible_chunk=lambda chunk: (
            events.append("compatible:" + chunk["chunk_identity"])
            or (result(chunk) if reuse_first and chunk["ordinal"] == 1 else None)
        ),
        infer_chunk=infer,
        publish_chunk=lambda chunk, rows: (
            events.append("publish:" + chunk["chunk_identity"]) or result(chunk)
        ),
        publish_failure=lambda chunk, code: events.append("failure:" + chunk),
        publish_completion=lambda payload: (
            events.append("completion") or payload
        ),
    )


def test_true_authorization_orders_full_synthetic_lifecycle(tmp_path, monkeypatch):
    request, runtime, checksum, authorization, plan = _fixture(
        tmp_path, monkeypatch
    )
    authorization["execution_authorized"] = True
    events = []
    result = execute_authorized_scoring(
        request=request, runtime=runtime, runtime_checksum=checksum,
        authorization=authorization, chunk_plan=plan,
        callbacks=_callbacks(events, request),
    )
    assert result["status"] == COMPLETE_STATUS
    assert events[:5] == ["persist", "lease", "source", "tokenizer", "model"]
    assert events[-2:] == ["completion", "lease_success"]


def test_invalid_authorization_and_plan_only_have_zero_side_effects(
    tmp_path, monkeypatch,
):
    request, runtime, checksum, authorization, plan = _fixture(
        tmp_path, monkeypatch
    )
    events = []
    callbacks = _callbacks(events, request)
    result = execute_authorized_scoring(
        request=request, runtime=runtime, runtime_checksum=checksum,
        authorization=authorization, chunk_plan=plan, callbacks=callbacks,
        plan_only=True,
    )
    assert result["status"] == "PLAN_VALIDATED_NOT_AUTHORIZED_FOR_EXECUTION"
    assert events == []
    authorization["canonical_identity"] = "wrong"
    with pytest.raises(ValueError):
        execute_authorized_scoring(
            request=request, runtime=runtime, runtime_checksum=checksum,
            authorization=authorization, chunk_plan=plan, callbacks=callbacks,
        )
    assert events == []


def test_failure_stops_later_chunks_and_closes_lease(tmp_path, monkeypatch):
    request, runtime, checksum, authorization, plan = _fixture(
        tmp_path, monkeypatch, chunks=3
    )
    authorization["execution_authorized"] = True
    events = []
    with pytest.raises(ExecutionFailure):
        execute_authorized_scoring(
            request=request, runtime=runtime, runtime_checksum=checksum,
            authorization=authorization, chunk_plan=plan,
            callbacks=_callbacks(events, request, fail_at="chunk-1"),
        )
    assert "infer:chunk-2" not in events
    assert "completion" not in events
    assert any(event.startswith("lease_failure:") for event in events)


def test_resume_and_probability_validation(tmp_path, monkeypatch):
    request, runtime, checksum, authorization, plan = _fixture(
        tmp_path, monkeypatch
    )
    authorization["execution_authorized"] = True
    events = []
    execute_authorized_scoring(
        request=request, runtime=runtime, runtime_checksum=checksum,
        authorization=authorization, chunk_plan=plan,
        callbacks=_callbacks(events, request, reuse_first=True),
    )
    assert "infer:chunk-0" not in events
    callbacks = _callbacks([], request)
    callbacks = ExecutionCallbacks(
        **{**callbacks.__dict__, "infer_chunk": lambda *args: [{
            "article_id": "bad", "symbol": "AAPL",
            "positive_probability": 0.8, "negative_probability": 0.8,
            "neutral_probability": 0.1, "sentiment_label": "positive",
            "model_package_identity": request["model_package_identity"],
            "canonical_identity": request["canonical_identity"],
            "chunk_identity": args[0]["chunk_identity"],
            "scored_at": request["scored_at_utc"],
        }]}
    )
    with pytest.raises(ExecutionFailure, match="NORMALIZATION"):
        execute_authorized_scoring(
            request=request, runtime=runtime, runtime_checksum=checksum,
            authorization=authorization, chunk_plan=plan, callbacks=callbacks,
        )


class _FakeProductionModel:
    def __init__(self, request):
        self.identity = FinBertModelIdentity(
            model_id=request["model_name"],
            model_revision=request["model_revision"],
            tokenizer_id=request["model_name"],
            tokenizer_revision=request["model_revision"],
            inference_device="cpu",
        )

    def score_batch(self, texts):
        return [
            FinBertPrediction(0.7, 0.2, 0.1, "positive") for _ in texts
        ]


def test_concrete_factory_runs_synthetic_shared_compute_end_to_end(
    tmp_path, monkeypatch,
):
    request, runtime, checksum, authorization, plan = _fixture(
        tmp_path, monkeypatch
    )
    authorization["execution_authorized"] = True
    rows = [
        {
            "provider_article_id": "article-b", "symbol": "MSFT",
            "headline": "second", "published_at_utc": "2026-01-01T00:00:00Z",
            "collected_at_utc": "2026-01-01T01:00:00Z",
            "provider": "fixture", "source": "fixture",
        },
        {
            "provider_article_id": "article-a", "symbol": "AAPL",
            "headline": "first", "published_at_utc": "2026-01-01T00:00:00Z",
            "collected_at_utc": "2026-01-01T01:00:00Z",
            "provider": "fixture", "source": "fixture",
        },
        {
            "provider_article_id": "article-x", "symbol": "AAPL",
            "headline": "", "body": "",
            "published_at_utc": "2026-01-01T00:00:00Z",
            "collected_at_utc": "2026-01-01T01:00:00Z",
            "provider": "fixture", "source": "fixture",
        },
    ]
    factory = ProductionExecutionFactory(
        request=request, runtime=runtime, chunk_plan=plan,
        runs_root=tmp_path / "runs", ledger_path=tmp_path / "ledger.json",
        registry_path=tmp_path / "registry.json",
        dependencies={
            "read_canonical_rows": lambda: rows,
            "activate_tokenizer": lambda: {"local_files_only": True},
            "activate_model": lambda: _FakeProductionModel(request),
        },
    )
    result = execute_authorized_scoring(
        request=request, runtime=runtime, runtime_checksum=checksum,
        authorization=authorization, chunk_plan=plan,
        callbacks=factory.callbacks(),
    )
    assert result["status"] == COMPLETE_STATUS
    assert factory.events.index("lease_acquisition") < factory.events.index(
        "source_activation"
    )
    assert factory.events.index("eligible_inventory_construction") < (
        factory.events.index("model_activation")
    )
    assert (Path(request["score_output_root"]) /
            "score_store_completion.json").exists()
    completion = json.loads(
        (Path(request["score_output_root"]) /
         "score_store_completion.json").read_text()
    )
    assert completion["certified"] is False
    assert completion["pit_generation_status"] == (
        "BLOCKED_AWAITING_CERTIFICATION"
    )
    assert (tmp_path / "ledger.json").exists()
    assert (tmp_path / "registry.json").exists()
    assert list((tmp_path / "runs").rglob("results.json"))
    assert factory.source["exclusions"][0]["reason_code"] == (
        "NO_SELECTABLE_SCORING_TEXT"
    )


def test_cli_execute_uses_concrete_injected_factory(tmp_path, monkeypatch):
    request, runtime, checksum, authorization, plan = _fixture(
        tmp_path, monkeypatch
    )
    authorization["execution_authorized"] = True
    rows = [
        {
            "provider_article_id": f"article-{index}", "symbol": "AAPL",
            "headline": f"fixture {index}",
            "published_at_utc": "2026-01-01T00:00:00Z",
            "collected_at_utc": "2026-01-01T01:00:00Z",
        }
        for index in range(2)
    ] + [{
        "provider_article_id": "excluded", "symbol": "AAPL", "headline": "",
        "published_at_utc": "2026-01-01T00:00:00Z",
        "collected_at_utc": "2026-01-01T01:00:00Z",
    }]
    paths = {}
    for name, value in (
        ("request", request), ("runtime", runtime),
        ("authorization", authorization), ("plan", plan),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    created = []

    def builder(**kwargs):
        factory = ProductionExecutionFactory(
            **kwargs, dependencies={
                "read_canonical_rows": lambda: rows,
                "activate_tokenizer": lambda: {"local_files_only": True},
                "activate_model": lambda: _FakeProductionModel(request),
            },
        )
        created.append(factory)
        return factory

    assert cli_main([
        "--request", str(paths["request"]),
        "--runtime-config", str(paths["runtime"]),
        "--authorization", str(paths["authorization"]),
        "--chunk-plan", str(paths["plan"]),
        "--run-root", str(tmp_path / "runs"),
        "--resource-ledger", str(tmp_path / "ledger.json"),
        "--registry", str(tmp_path / "registry.json"), "--execute",
    ], factory_builder=builder) == 0
    assert created
    assert "source_activation" in created[0].events
    assert "eligible_inventory_construction" in created[0].events


def test_inventory_order_and_bounded_exclusion_are_deterministic():
    rows = [
        {
            "provider_article_id": "b", "symbol": "MSFT", "headline": "b",
            "published_at_utc": "2026-01-01T00:00:00Z",
        },
        {
            "provider_article_id": "a", "symbol": "AAPL", "headline": "a",
            "published_at_utc": "2026-01-01T00:00:00Z",
        },
        {
            "provider_article_id": "x", "symbol": "AAPL", "headline": "",
            "published_at_utc": "2026-01-01T00:00:00Z",
        },
    ]
    first = build_eligible_scoring_inventory(
        rows, max_characters=10_000, canonical_identity="canonical"
    )
    second = build_eligible_scoring_inventory(
        rows, max_characters=10_000, canonical_identity="canonical"
    )
    assert first == second
    eligible, exclusions = first
    assert [(item["article_id"], item["symbol"]) for item in eligible] == [
        ("a", "AAPL"), ("b", "MSFT")
    ]
    assert len(exclusions) == 1
    assert exclusions[0]["reason_code"] == "NO_SELECTABLE_SCORING_TEXT"
    assert not {
        "headline", "body", "selected_text", "source_row"
    } & set(exclusions[0])


def test_immutable_model_activation_is_explicit_offline_and_cpu(
    tmp_path, monkeypatch,
):
    package = _package(tmp_path)
    request = _request(tmp_path, package)
    options = immutable_model_activation_options(request)
    assert options["model_path"] == package["package_root"]
    assert options["tokenizer_path"] == package["package_root"]
    assert options["cache_dir"] == package["package_root"]
    assert options["local_files_only"] is True
    assert options["device"] == "cpu"
    assert options["model_revision"] == REVISION


@pytest.mark.parametrize("phase", [
    "source", "tokenizer", "model", "infer", "publish", "completion",
])
def test_each_activation_or_publication_failure_closes_lease(
    tmp_path, monkeypatch, phase,
):
    request, runtime, checksum, authorization, plan = _fixture(
        tmp_path, monkeypatch, chunks=1
    )
    authorization["execution_authorized"] = True
    events = []
    callbacks = _callbacks(events, request)

    def fail(*args):
        raise RuntimeError("synthetic")

    mutations = {
        "source": {"activate_source": fail},
        "tokenizer": {"activate_tokenizer": fail},
        "model": {"activate_model": fail},
        "infer": {"infer_chunk": fail},
        "publish": {"publish_chunk": fail},
        "completion": {"publish_completion": fail},
    }
    callbacks = ExecutionCallbacks(
        **{**callbacks.__dict__, **mutations[phase]}
    )
    with pytest.raises((RuntimeError, ExecutionFailure)):
        execute_authorized_scoring(
            request=request, runtime=runtime, runtime_checksum=checksum,
            authorization=authorization, chunk_plan=plan,
            callbacks=callbacks,
        )
    assert any(event.startswith("lease_failure:") for event in events)
