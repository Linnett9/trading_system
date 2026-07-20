from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from core.research.ml.stock_level.news_sources.historical_canonical_corpus import (
    CANONICAL_CORPUS_CSV,
    CANONICAL_CORPUS_MANIFEST_JSON,
    materialize_historical_canonical_corpus,
    verify_canonical_corpus_inventory,
)
from core.research.ml.stock_level.stock_alpha_finbert_compute import (
    FinBertExecutionPolicy,
)
from core.research.ml.stock_level.stock_alpha_finbert_news import (
    FinBertAdapter,
    _article_id,
    _chunk_metadata_logical_checksum,
    _chunk_manifest_logical_checksum,
    _read_completed_chunk,
    _read_chunk_manifest_evidence,
    _scored_row,
    _scored_rows_logical_checksum,
    _symbol,
    _validate_chunk_row_ownership,
    _write_json_atomic,
    resolve_news_available_timestamp,
    select_article_text,
)
from core.research.ml.stock_level.stock_alpha_finbert_score_store import (
    certify_finbert_score_store,
)
from core.research.ml.stock_level.stock_alpha_news_data_compute import (
    CORPUS,
    FEATURES,
)
from core.research.ml.stock_level.stock_alpha_news_feature_store import (
    publish_pit_news_feature_store,
)
from core.research.ml.stock_level.stock_alpha_news_pit_policy import (
    StockAlphaNewsPitPolicy,
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True)
class CanonicalCorpusBinding:
    source_csv: Path
    source_metadata_json: Path
    output_dir: Path
    expected_source_checksum: str
    ingested_at_utc: str


@dataclass(frozen=True)
class PitFeatureStoreBinding:
    score_store_manifest_path: Path
    daily_spine_manifest_path: Path
    ticker_mapping_manifest_path: Path
    scored_articles_path: Path
    spine_rows_path: Path
    ticker_aliases_path: Path
    output_root: Path
    finbert_model_identity: Mapping[str, str]
    pit_policy: StockAlphaNewsPitPolicy
    source_commit: str


class AuthoritativeNewsDataAdapter:
    """Bind 1B lifecycle callbacks to existing corpus/PIT publishers."""

    def __init__(
        self, *, canonical: CanonicalCorpusBinding | None = None,
        feature_store: PitFeatureStoreBinding | None = None,
    ) -> None:
        self.canonical = canonical
        self.feature_store = feature_store
        self._corpus_result: dict[str, Any] | None = None
        self._feature_result: dict[str, Any] | None = None

    def compatible(self, stage, work_unit, corpus_parent):
        del work_unit
        if stage == CORPUS:
            binding = self._require_corpus()
            manifest_path = binding.output_dir / CANONICAL_CORPUS_MANIFEST_JSON
            corpus_path = binding.output_dir / CANONICAL_CORPUS_CSV
            if not manifest_path.exists():
                return None
            manifest = _json(manifest_path)
            validation = verify_canonical_corpus_inventory(
                manifest, corpus_path=corpus_path
            )
            if not validation["inventory_certified"]:
                raise ValueError("Existing canonical corpus is incompatible")
            return _bounded_corpus(manifest, validation)
        binding = self._require_features()
        manifest_path = binding.output_root / "manifest.json"
        if not manifest_path.exists():
            return None
        manifest = _json(manifest_path)
        _validate_feature_reference(manifest, corpus_parent, binding.pit_policy)
        return _bounded_features(manifest)

    def resolve_corpus_input(self, work_unit):
        del work_unit
        binding = self._require_corpus()
        if not binding.source_csv.is_file() or not binding.source_metadata_json.is_file():
            raise FileNotFoundError("Canonical source assembly inputs are missing")
        return {"input_rows": None, "binding": binding}

    def read_corpus_source(self, resolved):
        return resolved

    def canonicalise(self, source):
        binding = source["binding"]
        self._corpus_result = materialize_historical_canonical_corpus(
            source_assembly_csv_path=binding.source_csv,
            source_assembly_metadata_json_path=binding.source_metadata_json,
            output_dir=binding.output_dir,
            expected_source_checksum=binding.expected_source_checksum,
            write_enabled=True, ingested_at_utc=binding.ingested_at_utc,
        )
        return self._corpus_result

    def validate_corpus_identity(self, canonical):
        binding = self._require_corpus()
        result = verify_canonical_corpus_inventory(
            canonical, corpus_path=binding.output_dir / CANONICAL_CORPUS_CSV
        )
        return {"passed": result["inventory_certified"],
                "validation_status": "PASSED" if result["inventory_certified"]
                else "FAILED"}

    def validate_corpus_availability(self, canonical):
        passed = bool(
            canonical.get("row_count_reconciled")
            and canonical.get("ingested_at_utc")
        )
        return {"passed": passed,
                "validation_status": "PASSED" if passed else "FAILED"}

    def publish_corpus(self, canonical):
        validation = verify_canonical_corpus_inventory(
            canonical,
            corpus_path=self._require_corpus().output_dir / CANONICAL_CORPUS_CSV,
        )
        if not validation["inventory_certified"]:
            raise ValueError("Published canonical corpus inventory is invalid")
        return _bounded_corpus(canonical, validation)

    def resolve_corpus_parent(self, parent, work_unit):
        del work_unit
        if not parent.get("canonical_corpus_identity"):
            raise ValueError("Canonical corpus parent is required")
        return dict(parent)

    def prepare_pit_inputs(self, parent, work_unit):
        del work_unit
        binding = self._require_features()
        return {
            "parent": dict(parent), "binding": binding,
            "canonical_manifest": _canonical_manifest(parent),
            "score_manifest": _json(binding.score_store_manifest_path),
            "spine_manifest": _json(binding.daily_spine_manifest_path),
            "mapping_manifest": _json(binding.ticker_mapping_manifest_path),
        }

    def calculate_features(self, prepared):
        binding = prepared["binding"]
        self._feature_result = publish_pit_news_feature_store(
            canonical_corpus_manifest=prepared["canonical_manifest"],
            score_store_manifest=prepared["score_manifest"],
            daily_spine_manifest=prepared["spine_manifest"],
            ticker_mapping_manifest=prepared["mapping_manifest"],
            scored_articles=_csv(binding.scored_articles_path),
            spine_rows=_csv(binding.spine_rows_path),
            ticker_aliases=_json(binding.ticker_aliases_path),
            output_root=binding.output_root,
            finbert_model_identity=binding.finbert_model_identity,
            source_commit=binding.source_commit,
            pit_policy=binding.pit_policy,
        )
        return self._feature_result

    def validate_feature_store(self, features):
        _validate_feature_reference(
            features, {
                "canonical_corpus_identity": features["canonical_corpus_identity"],
                "canonical_corpus_checksum": features["canonical_corpus_checksum"],
            }, self._require_features().pit_policy,
        )
        return {"passed": True, "validation_status": "PASSED"}

    def publish_feature_store(self, features):
        return _bounded_features(features)

    def _require_corpus(self) -> CanonicalCorpusBinding:
        if self.canonical is None:
            raise ValueError("Canonical corpus binding was not configured")
        return self.canonical

    def _require_features(self) -> PitFeatureStoreBinding:
        if self.feature_store is None:
            raise ValueError("PIT feature-store binding was not configured")
        return self.feature_store


class AuthoritativeFinBertChunkAdapter:
    """Use authoritative chunk compatibility, row construction, and publisher."""

    def __init__(
        self, *, scoring_plan: Mapping[str, Any],
        source_rows: Sequence[Mapping[str, Any]] | None, output_dir: Path,
        scoring_config: Mapping[str, Any], model_factory: Callable[
            [Mapping[str, Any], FinBertExecutionPolicy], FinBertAdapter],
        scored_at: str,
        prepared_items_by_chunk: Mapping[str, Sequence[Mapping[str, Any]]]
        | None = None,
    ) -> None:
        self.plan = dict(scoring_plan)
        self.output_dir = output_dir
        self.config = dict(scoring_config)
        self.model_factory = model_factory
        self.scored_at = scored_at
        self.model_load_count = 0
        self._active_identity = None
        if prepared_items_by_chunk is None:
            if source_rows is None:
                raise ValueError("Source rows or explicit prepared items are required")
            self._items = _prepared_items(source_rows, self.plan)
        else:
            self._items = {
                str(key): [dict(item) for item in values]
                for key, values in prepared_items_by_chunk.items()
            }
            expected = {
                str(chunk["chunk_id"]): int(chunk["article_count"])
                for chunk in self.plan["expected_chunks"]
            }
            if set(self._items) != set(expected) or any(
                len(self._items[key]) != count for key, count in expected.items()
            ):
                raise ValueError("Explicit FinBERT chunk item population mismatch")

    def compatible_output(self, chunk):
        path = self.output_dir / "chunks" / f"{chunk['chunk_id']}.json"
        evidence = _read_chunk_manifest_evidence(
            self.output_dir / "finbert_chunk_manifest.csv"
        ).get(chunk["chunk_id"])
        payload = _read_completed_chunk(
            path, chunk["identity"],
            plan_binding={
                "logical_checksum": self.plan["logical_checksum"],
                "plan_artifact_checksum": self.plan["plan_artifact_checksum"],
            },
            planned_ordinal=int(chunk["ordinal"]), external_evidence=evidence,
        )
        return _chunk_reference(path, payload) if payload else None

    def load_model(self, model_reference, policy):
        self.model_load_count += 1
        model = self.model_factory(model_reference, policy)
        self._active_identity = model.identity
        return model

    def tokenize(self, model, chunk):
        del model
        items = self._items[chunk["chunk_id"]]
        return [item["text"].text for item in items]

    def infer(self, model, tokenized, chunk):
        del chunk
        return model.score_batch(tokenized)

    def publish(self, chunk, predictions):
        return self.publish_rows(chunk, self.build_rows(chunk, predictions))

    def build_rows(self, chunk, predictions):
        items = self._items[chunk["chunk_id"]]
        rows = [
            _scored_row(
                item, prediction,
                self._active_identity or _identity_from_plan(self.plan),
                max_token_length=int(self.plan["maximum_token_length"]),
                chunk_id=chunk["chunk_id"], scored_at=self.scored_at,
            )
            for item, prediction in zip(items, predictions)
        ]
        if len(rows) != len(items):
            raise ValueError("FinBERT prediction count mismatch")
        return rows

    def publish_rows(self, chunk, rows):
        items = self._items[chunk["chunk_id"]]
        identity = chunk["identity"]
        ordinal = int(chunk["ordinal"])
        if len(rows) != len(items):
            raise ValueError("FinBERT scored row count mismatch")
        payload = {
            "status": "completed", "identity": identity, "rows": rows,
            "scoring_plan": {
                "logical_checksum": self.plan["logical_checksum"],
                "plan_artifact_checksum": self.plan["plan_artifact_checksum"],
                "planned_ordinal": ordinal,
            },
            "production_scope": True,
            "scored_rows_logical_checksum": _scored_rows_logical_checksum(
                rows, chunk_id=chunk["chunk_id"], planned_ordinal=ordinal
            ),
        }
        _validate_chunk_row_ownership(payload, identity)
        payload["chunk_metadata_logical_checksum"] = (
            _chunk_metadata_logical_checksum(payload)
        )
        path = self.output_dir / "chunks" / f"{chunk['chunk_id']}.json"
        _write_json_atomic(path, payload)
        self._write_chunk_manifest()
        return _chunk_reference(path, payload)

    def _write_chunk_manifest(self):
        from core.research.framework.reporting import ResearchArtifactWriter
        records = []
        for chunk in self.plan["expected_chunks"]:
            path = self.output_dir / "chunks" / f"{chunk['chunk_id']}.json"
            if not path.exists():
                continue
            payload = _json(path)
            _validate_chunk_row_ownership(payload, chunk["identity"])
            reference = _chunk_reference(path, payload)
            records.append({
                "chunk_id": chunk["chunk_id"], "status": "completed",
                "article_count": len(payload["rows"]), "reused": "false",
                "chunk_path": str(path),
                "scoring_plan_identity": self.plan["logical_checksum"],
                "planned_ordinal": chunk["ordinal"],
                "production_scope": "true",
                "chunk_artifact_sha256": reference["chunk_artifact_sha256"],
                "scored_rows_logical_checksum":
                    payload["scored_rows_logical_checksum"],
                "chunk_metadata_logical_checksum":
                    payload["chunk_metadata_logical_checksum"],
            })
        manifest_checksum = _chunk_manifest_logical_checksum(records)
        for record in records:
            record["manifest_logical_checksum"] = manifest_checksum
        ResearchArtifactWriter().write_csv(
            self.output_dir / "finbert_chunk_manifest.csv", records,
            fieldnames=(
                "chunk_id", "status", "article_count", "reused", "chunk_path",
                "scoring_plan_identity", "planned_ordinal", "production_scope",
                "chunk_artifact_sha256", "scored_rows_logical_checksum",
                "chunk_metadata_logical_checksum", "manifest_logical_checksum",
            ),
        )


class AuthoritativeCertificationAdapter:
    def __init__(self, *, chunk_manifest_path: Path, output_path: Path,
                 source_commit: str) -> None:
        self.chunk_manifest_path = chunk_manifest_path
        self.output_path = output_path
        self.source_commit = source_commit

    def __call__(self, scoring_plan):
        result = certify_finbert_score_store(
            scoring_plan=scoring_plan,
            chunk_manifest_path=self.chunk_manifest_path,
            output_path=self.output_path,
            source_commit=self.source_commit,
        )
        return result


def _prepared_items(rows, plan):
    output = {}
    by_identity = {}
    for index, row in enumerate(rows, 1):
        item = {
            "source_row_number": index, "source_row": dict(row),
            "article_id": _article_id(row), "symbol": _symbol(row),
            "text": select_article_text(
                row, max_characters=int(
                    plan["maximum_selected_text_characters"])
            ),
            "availability": resolve_news_available_timestamp(row),
        }
        key = (item["article_id"], item["symbol"], item["text"].text_hash)
        by_identity[key] = item
    for chunk in plan["expected_chunks"]:
        items = []
        for identity in chunk["identity"]["article_identities"]:
            key = (identity["article_id"], identity["symbol"],
                   identity["selected_text_hash"])
            if key not in by_identity:
                raise ValueError("Scoring source rows do not match planned inventory")
            items.append(by_identity[key])
        output[chunk["chunk_id"]] = items
    return output


def _identity_from_plan(plan):
    from core.research.ml.stock_level.stock_alpha_finbert_news import (
        FinBertModelIdentity,
    )
    model = plan["finbert_model_identity"]
    return FinBertModelIdentity(
        model_id=model["model_id"], model_revision=model["model_revision"],
        tokenizer_id=model["tokenizer_id"],
        tokenizer_revision=model["tokenizer_revision"],
        inference_device="cpu",
    )


def _chunk_reference(path, payload):
    if payload is None:
        return None
    import hashlib
    return {
        "chunk_id": payload["identity"]["chunk_id"],
        "chunk_path": str(path),
        "chunk_artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        "scored_rows_logical_checksum": payload["scored_rows_logical_checksum"],
        "chunk_metadata_logical_checksum":
            payload["chunk_metadata_logical_checksum"],
        "row_count": len(payload["rows"]), "publication_result": "PUBLISHED",
    }


def _bounded_corpus(manifest, validation):
    return {
        key: manifest[key] for key in (
            "canonical_corpus_identity", "canonical_corpus_checksum",
            "logical_manifest_checksum", "canonical_schema_version",
            "canonical_schema_checksum", "canonical_row_count",
            "source_row_count", "duplicate_group_count", "publication_result",
        ) if key in manifest
    } | {"validation_status": "PASSED",
         "inventory_certified": validation["inventory_certified"]}


def _bounded_features(manifest):
    return {
        key: manifest[key] for key in (
            "feature_store_artifact_checksum", "logical_checksum",
            "feature_store_contract", "feature_schema_checksum", "row_count",
            "canonical_corpus_identity", "canonical_corpus_checksum",
            "canonical_daily_spine_identity", "ticker_mapping_identity",
            "pit_eligibility_policy_identity", "manifest_publication_result",
        ) if key in manifest
    } | {"validation_status": "PASSED"}


def _validate_feature_reference(manifest, parent, policy):
    from core.research.ml.stock_level.stock_alpha_news_feature_store import _hash
    from core.research.ml.stock_level.stock_alpha_news_pit_policy import (
        pit_policy_payload,
    )
    expected = {
        "canonical_corpus_identity": parent["canonical_corpus_identity"],
        "canonical_corpus_checksum": parent["canonical_corpus_checksum"],
        "pit_eligibility_policy_identity": _hash(pit_policy_payload(policy)),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("PIT feature-store ancestry or policy mismatch")


def _canonical_manifest(parent):
    return {
        "canonical_corpus_identity": parent["canonical_corpus_identity"],
        "canonical_corpus_checksum": parent["canonical_corpus_checksum"],
    }
