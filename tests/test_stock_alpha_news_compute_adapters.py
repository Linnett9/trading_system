from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.research.compute.machine_profile import GIB, dell_i5_10500_profile
from core.research.compute.resource_lease_ledger import ResourceLeaseLedger
from core.research.ml.stock_level.news_sources.historical_canonical_corpus import (
    CANONICAL_CORPUS_CSV,
    sha256_file,
)
from core.research.ml.stock_level.stock_alpha_finbert_compute import (
    execute_finbert_compute_run,
)
from core.research.ml.stock_level.stock_alpha_finbert_news import (
    DeterministicFinBertFixtureAdapter,
)
from core.research.ml.stock_level.stock_alpha_news_compute_adapters import (
    AuthoritativeCertificationAdapter,
    AuthoritativeFinBertChunkAdapter,
    AuthoritativeNewsDataAdapter,
    CanonicalCorpusBinding,
    PitFeatureStoreBinding,
)
from core.research.ml.stock_level.stock_alpha_news_data_compute import (
    CORPUS,
    FEATURES,
    NewsDataMaterialisationPlan,
    execute_news_data_compute_run,
)
from core.research.ml.stock_level.stock_alpha_news_pit_policy import (
    STRICT_COLLECTED_AT,
    StockAlphaNewsPitPolicy,
)
from tests.test_stock_alpha_finbert_score_store import _plan_fixture
from tests.test_stock_alpha_news_feature_store import (
    MODEL, _article, _parents, _spine,
)


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest().upper()


def _services(tmp_path):
    profile = dell_i5_10500_profile(
        source_git_commit="commit", generated_at="fixed"
    )
    ledger = ResourceLeaseLedger(
        profile=profile, path=tmp_path / "ledger.json",
        available_memory=lambda: 32 * GIB,
    )
    ledger.initialise_ledger()
    return profile, ledger


def _data_plan(stages, corpus_id="expected", corpus_checksum="EXPECTED"):
    return NewsDataMaterialisationPlan(
        selected_stages=tuple(stages), source_inventory_identity="source",
        source_inventory_checksum="SOURCE",
        canonical_corpus_contract_identity="canonical-v2",
        canonical_output_compatibility_identity="canonical-compatible",
        canonical_parent_identity=corpus_id,
        canonical_parent_checksum=corpus_checksum,
        date_boundary_identity="2024-01-01_2024-01-04",
        universe_identity="AAPL_MSFT",
        availability_policy_identity="placeholder",
        pit_feature_contract_identity="pit-v1",
        feature_output_compatibility_identity="features-compatible",
        configuration_checksum="CONFIG", source_git_commit="commit",
    )


def _assembly(provider, article, symbol, published, collected):
    return {
        "provider": provider, "article_id": f"{provider}:{article}:{symbol}",
        "provider_article_id": article,
        "provider_original_article_id": article,
        "provider_symbols": symbol, "symbol": symbol,
        "published_at_utc": published, "updated_at_utc": "",
        "collected_at_utc": collected, "headline": f"{symbol} synthetic",
        "summary": "", "body_or_full_text": "", "source": provider,
        "publisher": provider, "author": "", "provider_url": "",
        "language": "en", "event_type": "",
    }


def test_authoritative_corpus_adapter_materialises_validates_and_resumes(tmp_path):
    source = tmp_path / "source.csv"
    metadata = tmp_path / "source.json"
    output = tmp_path / "canonical"
    rows = [
        _assembly("provider-a", "shared", "AAPL",
                  "2024-01-02T10:00:00Z", "2024-01-02T11:00:00Z"),
        _assembly("provider-b", "b2", "MSFT",
                  "2024-01-03T10:00:00Z", "2024-01-03T11:00:00Z"),
    ]
    _write_csv(source, rows)
    checksum = sha256_file(source)
    metadata.write_text(json.dumps({"assembly_checksum": checksum}),
                        encoding="utf-8")
    adapter = AuthoritativeNewsDataAdapter(canonical=CanonicalCorpusBinding(
        source, metadata, output, checksum, "2026-07-18T00:00:00Z"
    ))
    profile, ledger = _services(tmp_path)
    first = execute_news_data_compute_run(
        plan=_data_plan((CORPUS,)), adapter=adapter,
        machine_profile=profile, lease_ledger=ledger,
        runs_root=tmp_path / "runs", registry_path=tmp_path / "registry.json",
    )
    assert first["summary"]["newly_materialised_items"] == 1
    assert (output / CANONICAL_CORPUS_CSV).exists()
    second = execute_news_data_compute_run(
        plan=_data_plan((CORPUS,)), adapter=adapter,
        machine_profile=profile, lease_ledger=ledger,
        runs_root=tmp_path / "runs", registry_path=tmp_path / "registry.json",
    )
    assert second["run_identity"] == first["run_identity"]
    assert second["summary"]["reused_skipped_items"] == 1
    assert ledger.read_ledger_status()["active_leases"] == []


def test_authoritative_finbert_adapter_scores_certifies_and_resumes(tmp_path):
    plan, rows, fake, config = _plan_fixture(chunk_size=1)
    output = tmp_path / "scoring"
    adapter = AuthoritativeFinBertChunkAdapter(
        scoring_plan=plan, source_rows=rows, output_dir=output,
        scoring_config=config, model_factory=lambda reference, policy: fake,
        scored_at="2026-07-18T00:00:00Z",
    )
    certifier = AuthoritativeCertificationAdapter(
        chunk_manifest_path=output / "finbert_chunk_manifest.csv",
        output_path=tmp_path / "certificate.json", source_commit="commit",
    )
    profile, ledger = _services(tmp_path)
    first = execute_finbert_compute_run(
        scoring_plan=plan, adapter=adapter, certify=certifier,
        machine_profile=profile, lease_ledger=ledger,
        runs_root=tmp_path / "runs", registry_path=tmp_path / "registry.json",
    )
    assert first["summary"]["newly_scored_chunks"] == 2
    assert first["summary"]["certification_status"] == "COMPLETE"
    assert adapter.model_load_count == 2
    second = execute_finbert_compute_run(
        scoring_plan=plan, adapter=adapter, certify=certifier,
        machine_profile=profile, lease_ledger=ledger,
        runs_root=tmp_path / "runs", registry_path=tmp_path / "registry.json",
    )
    assert second["run_identity"] == first["run_identity"]
    assert second["summary"]["compatible_reused_chunks"] == 2
    assert adapter.model_load_count == 2
    assert ledger.read_ledger_status()["active_leases"] == []


def test_finbert_model_failure_releases_lease(tmp_path):
    plan, rows, _, config = _plan_fixture(chunk_size=2)
    def fail(reference, policy):
        raise RuntimeError("synthetic model load failure")
    adapter = AuthoritativeFinBertChunkAdapter(
        scoring_plan=plan, source_rows=rows, output_dir=tmp_path / "scoring",
        scoring_config=config, model_factory=fail,
        scored_at="2026-07-18T00:00:00Z",
    )
    profile, ledger = _services(tmp_path)
    result = execute_finbert_compute_run(
        scoring_plan=plan, adapter=adapter,
        certify=lambda _: pytest.fail("certification must not run"),
        machine_profile=profile, lease_ledger=ledger,
        runs_root=tmp_path / "runs", registry_path=tmp_path / "registry.json",
    )
    assert result["summary"]["final_run_status"] == "FAILED"
    assert ledger.read_ledger_status()["active_leases"] == []


def test_scripts_help_plan_only_and_invalid_exit(tmp_path):
    for script in (
        "scripts/run_stock_alpha_finbert_compute.py",
        "scripts/run_stock_alpha_news_data_compute.py",
    ):
        help_result = subprocess.run(
            [sys.executable, script, "--help"], capture_output=True, text=True
        )
        assert help_result.returncode == 0
    invalid = subprocess.run(
        [sys.executable, "scripts/run_stock_alpha_news_data_compute.py"],
        capture_output=True, text=True,
    )
    assert invalid.returncode != 0

    plan, _, _, _ = _plan_fixture(chunk_size=2)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps({"scoring_plan_path": str(plan_path)}),
                            encoding="utf-8")
    planned = subprocess.run([
        sys.executable, "scripts/run_stock_alpha_finbert_compute.py",
        "--request", str(request_path), "--run-root", str(tmp_path / "runs"),
        "--resource-ledger", str(tmp_path / "ledger.json"),
        "--registry", str(tmp_path / "registry.json"), "--plan-only",
    ], capture_output=True, text=True)
    assert planned.returncode == 0
    assert json.loads(planned.stdout)["status"] == "PLAN_VALID"
    assert not (tmp_path / "ledger.json").exists()
    data_request = tmp_path / "data-request.json"
    data_request.write_text(json.dumps({
        "plan": {
            **_data_plan((CORPUS,)).__dict__,
            "selected_stages": [CORPUS],
            "corpus_work_units": [],
            "feature_work_units": [],
        }
    }), encoding="utf-8")
    data_planned = subprocess.run([
        sys.executable, "scripts/run_stock_alpha_news_data_compute.py",
        "--request", str(data_request), "--run-root", str(tmp_path / "data-runs"),
        "--resource-ledger", str(tmp_path / "data-ledger.json"),
        "--registry", str(tmp_path / "data-registry.json"), "--plan-only",
    ], capture_output=True, text=True)
    assert data_planned.returncode == 0
    assert json.loads(data_planned.stdout)["status"] == "PLAN_VALID"
    assert not (tmp_path / "data-ledger.json").exists()


def test_shared_json_records_do_not_leak_payload_text(tmp_path):
    plan, rows, fake, config = _plan_fixture(chunk_size=2)
    secret = "growth"
    adapter = AuthoritativeFinBertChunkAdapter(
        scoring_plan=plan, source_rows=rows, output_dir=tmp_path / "scoring",
        scoring_config=config, model_factory=lambda reference, policy: fake,
        scored_at="2026-07-18T00:00:00Z",
    )
    profile, ledger = _services(tmp_path)
    execute_finbert_compute_run(
        scoring_plan=plan, adapter=adapter,
        certify=AuthoritativeCertificationAdapter(
            chunk_manifest_path=tmp_path / "scoring" / "finbert_chunk_manifest.csv",
            output_path=tmp_path / "certificate.json", source_commit="commit"),
        machine_profile=profile, lease_ledger=ledger,
        runs_root=tmp_path / "runs", registry_path=tmp_path / "registry.json",
    )
    shared = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "runs").rglob("*") if path.is_file()
    )
    assert secret not in shared


def test_authoritative_pit_adapter_publishes_exact_ancestry_and_resumes(tmp_path):
    from core.research.ml.stock_level.stock_alpha_news_feature_store import _hash
    from core.research.ml.stock_level.stock_alpha_news_pit_policy import (
        pit_policy_payload,
    )
    policy = StockAlphaNewsPitPolicy(STRICT_COLLECTED_AT, 0.0, False)
    articles = [
        _article("eligible", "AAPL", "2024-01-02T10:00:00Z",
                 "2024-01-02T11:00:00Z", 0.7),
        _article("late", "AAPL", "2024-01-02T12:00:00Z",
                 "2024-01-05T11:00:00Z", -0.8),
    ]
    spine = [{
        "asset_id": "asset_AAPL", "symbol": "AAPL",
        "decision_session_date": "2024-01-03",
        "decision_timestamp": "2024-01-03T21:00:00Z",
    }]
    aliases = {}
    _write_csv(tmp_path / "articles.csv", articles)
    _write_csv(tmp_path / "spine.csv", spine)
    with (tmp_path / "articles.csv").open(newline="", encoding="utf-8") as handle:
        persisted_articles = list(csv.DictReader(handle))
    with (tmp_path / "spine.csv").open(newline="", encoding="utf-8") as handle:
        persisted_spine = list(csv.DictReader(handle))
    parents = _parents(persisted_articles, persisted_spine, aliases)
    paths = {}
    for name, payload in (
        ("score", parents["score_store_manifest"]),
        ("spine", parents["daily_spine_manifest"]),
        ("mapping", parents["ticker_mapping_manifest"]),
        ("aliases", aliases),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    binding = PitFeatureStoreBinding(
        paths["score"], paths["spine"], paths["mapping"],
        tmp_path / "articles.csv", tmp_path / "spine.csv", paths["aliases"],
        tmp_path / "features", MODEL, policy, "commit",
    )
    plan_value = NewsDataMaterialisationPlan(
        **{
            **_data_plan(
                (FEATURES,), "canonical-news-v1", "CORPUS"
            ).__dict__,
            "availability_policy_identity": _hash(pit_policy_payload(policy)),
        }
    )
    profile, ledger = _services(tmp_path)
    adapter = AuthoritativeNewsDataAdapter(feature_store=binding)
    first = execute_news_data_compute_run(
        plan=plan_value, adapter=adapter, machine_profile=profile,
        lease_ledger=ledger, runs_root=tmp_path / "runs",
        registry_path=tmp_path / "registry.json",
    )
    manifest = json.loads(
        (tmp_path / "features" / "manifest.json").read_text()
    )
    assert manifest["canonical_corpus_identity"] == "canonical-news-v1"
    assert manifest["eligibility_evidence"][
        "post_decision_article_exclusions"
    ] > 0
    second = execute_news_data_compute_run(
        plan=plan_value, adapter=adapter, machine_profile=profile,
        lease_ledger=ledger, runs_root=tmp_path / "runs",
        registry_path=tmp_path / "registry.json",
    )
    assert first["run_identity"] == second["run_identity"]
    assert second["summary"]["reused_skipped_items"] == 1


def test_synthetic_authoritative_end_to_end_flow(tmp_path):
    from core.research.ml.stock_level.stock_alpha_finbert_scoring_plan import (
        publish_finbert_scoring_plan,
    )
    from core.research.ml.stock_level.stock_alpha_news_feature_store import _hash
    from core.research.ml.stock_level.stock_alpha_news_pit_policy import (
        pit_policy_payload,
    )
    source = tmp_path / "assembly.csv"
    metadata = tmp_path / "assembly.json"
    canonical_root = tmp_path / "canonical"
    source_rows = [
        _assembly("provider-a", "a1", "AAPL", "2024-01-02T10:00:00Z",
                  "2024-01-02T11:00:00Z"),
        _assembly("provider-b", "b1", "MSFT", "2024-01-03T10:00:00Z",
                  "2024-01-05T11:00:00Z"),
    ]
    _write_csv(source, source_rows)
    source_checksum = sha256_file(source)
    metadata.write_text(json.dumps({"assembly_checksum": source_checksum}),
                        encoding="utf-8")
    profile, data_ledger = _services(tmp_path / "corpus-compute")
    corpus_run = execute_news_data_compute_run(
        plan=_data_plan((CORPUS,)),
        adapter=AuthoritativeNewsDataAdapter(canonical=CanonicalCorpusBinding(
            source, metadata, canonical_root, source_checksum,
            "2026-07-18T00:00:00Z")),
        machine_profile=profile, lease_ledger=data_ledger,
        runs_root=tmp_path / "corpus-runs",
        registry_path=tmp_path / "corpus-registry.json",
    )
    corpus_manifest = json.loads(
        (canonical_root / "stock_alpha_news_canonical_corpus_manifest.json")
        .read_text()
    )
    with (canonical_root / CANONICAL_CORPUS_CSV).open(
        newline="", encoding="utf-8"
    ) as handle:
        canonical_rows = list(csv.DictReader(handle))
    fake = DeterministicFinBertFixtureAdapter()
    config = {"ticket": "synthetic-e2e"}
    scoring_plan = publish_finbert_scoring_plan(
        corpus_manifest=corpus_manifest,
        corpus_path=canonical_root / CANONICAL_CORPUS_CSV,
        canonical_rows=canonical_rows,
        output_path=tmp_path / "scoring-plan.json",
        model_identity=fake.identity, scoring_config=config, chunk_size=1,
        scope="production", source_commit="commit",
    )
    scoring_output = tmp_path / "scoring"
    scoring_adapter = AuthoritativeFinBertChunkAdapter(
        scoring_plan=scoring_plan, source_rows=canonical_rows,
        output_dir=scoring_output, scoring_config=config,
        model_factory=lambda reference, policy: fake,
        scored_at="2026-07-18T00:00:00Z",
    )
    scoring_profile, scoring_ledger = _services(tmp_path / "score-compute")
    scoring_run = execute_finbert_compute_run(
        scoring_plan=scoring_plan, adapter=scoring_adapter,
        certify=AuthoritativeCertificationAdapter(
            chunk_manifest_path=scoring_output / "finbert_chunk_manifest.csv",
            output_path=tmp_path / "score-certificate.json",
            source_commit="commit"),
        machine_profile=scoring_profile, lease_ledger=scoring_ledger,
        runs_root=tmp_path / "scoring-runs",
        registry_path=tmp_path / "scoring-registry.json",
    )
    certificate = json.loads(
        (tmp_path / "score-certificate.json").read_text()
    )
    scored = []
    for path in sorted((scoring_output / "chunks").glob("*.json")):
        scored.extend(json.loads(path.read_text())["rows"])
    scored_path = tmp_path / "scored.csv"
    _write_csv(scored_path, scored)
    with scored_path.open(newline="", encoding="utf-8") as handle:
        persisted_scored = list(csv.DictReader(handle))
    spine = [
        {"asset_id": "asset_AAPL", "symbol": "AAPL",
         "decision_session_date": "2024-01-03",
         "decision_timestamp": "2024-01-03T21:00:00Z"},
        {"asset_id": "asset_MSFT", "symbol": "MSFT",
         "decision_session_date": "2024-01-04",
         "decision_timestamp": "2024-01-04T21:00:00Z"},
    ]
    spine_path = tmp_path / "spine.csv"
    _write_csv(spine_path, spine)
    with spine_path.open(newline="", encoding="utf-8") as handle:
        persisted_spine = list(csv.DictReader(handle))
    aliases = {}
    alias_path = tmp_path / "aliases.json"
    alias_path.write_text("{}", encoding="utf-8")
    score_manifest_path = tmp_path / "score-parent.json"
    certificate["scored_rows_logical_checksum"] = _hash(persisted_scored)
    score_manifest_path.write_text(json.dumps(certificate), encoding="utf-8")
    spine_manifest_path = tmp_path / "spine.json"
    spine_manifest_path.write_text(json.dumps({
        "daily_spine_identity": "spine", "daily_spine_checksum": "SPINE",
        "spine_rows_logical_checksum": _hash(persisted_spine),
    }), encoding="utf-8")
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps({
        "ticker_mapping_identity": "mapping",
        "ticker_mapping_checksum": "MAPPING",
        "ticker_aliases_logical_checksum": _hash(aliases),
    }), encoding="utf-8")
    policy = StockAlphaNewsPitPolicy(STRICT_COLLECTED_AT, 0.0, False)
    feature_plan = NewsDataMaterialisationPlan(
        **{
            **_data_plan((FEATURES,), corpus_manifest[
                "canonical_corpus_identity"], corpus_manifest[
                "canonical_corpus_checksum"]).__dict__,
            "availability_policy_identity": _hash(pit_policy_payload(policy)),
        }
    )
    feature_profile, feature_ledger = _services(tmp_path / "feature-compute")
    feature_run = execute_news_data_compute_run(
        plan=feature_plan,
        adapter=AuthoritativeNewsDataAdapter(feature_store=PitFeatureStoreBinding(
            score_manifest_path, spine_manifest_path, mapping_path, scored_path,
            spine_path, alias_path, tmp_path / "features",
            scoring_plan["finbert_model_identity"], policy, "commit")),
        machine_profile=feature_profile, lease_ledger=feature_ledger,
        runs_root=tmp_path / "feature-runs",
        registry_path=tmp_path / "feature-registry.json",
    )
    feature_manifest = json.loads(
        (tmp_path / "features" / "manifest.json").read_text()
    )
    assert corpus_run["summary"]["final_run_status"] == "COMPONENTS_COMPLETE"
    assert scoring_run["summary"]["certification_status"] == "COMPLETE"
    assert feature_run["summary"]["final_run_status"] == "COMPONENTS_COMPLETE"
    assert feature_manifest["canonical_corpus_identity"] == (
        corpus_manifest["canonical_corpus_identity"]
    )
    assert feature_manifest["eligibility_evidence"][
        "post_decision_article_exclusions"
    ] > 0
