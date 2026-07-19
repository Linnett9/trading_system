from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.research.compute.machine_profile import GIB, dell_i5_10500_profile
from core.research.compute.resource_lease_ledger import ResourceLeaseLedger
from core.research.ml.stock_level.stock_alpha_news_data_compute import (
    CORPUS,
    FEATURES,
    NewsDataMaterialisationPlan,
    build_news_data_resource_request,
    deterministic_news_data_item_id,
    deterministic_news_data_run_id,
    execute_news_data_compute_run,
)


def plan(*stages):
    return NewsDataMaterialisationPlan(
        selected_stages=tuple(stages),
        source_inventory_identity="providers-v1",
        source_inventory_checksum="SOURCE",
        canonical_corpus_contract_identity="canonical-contract-v2",
        canonical_output_compatibility_identity="canonical-compatible-v1",
        canonical_parent_identity="canonical-v1",
        canonical_parent_checksum="CORPUS",
        date_boundary_identity="2020-01-01_2024-12-31",
        universe_identity="universe-v1",
        availability_policy_identity="strict-collected-at-v1",
        pit_feature_contract_identity="pit-features-v1",
        feature_output_compatibility_identity="features-compatible-v1",
        configuration_checksum="CONFIG",
        source_git_commit="commit",
    )


class Adapter:
    def __init__(
        self, *, compatible=(), fail=None, validation_failure=None,
        wrong_ancestry=False,
    ):
        self.compatible_stages = set(compatible)
        self.fail = fail
        self.validation_failure = validation_failure
        self.wrong_ancestry = wrong_ancestry
        self.calls = []

    def compatible(self, stage, unit, parent):
        self.calls.append((stage, "compatible"))
        if stage not in self.compatible_stages:
            return None
        return self._corpus() if stage == CORPUS else self._features()

    def resolve_corpus_input(self, unit):
        return self._call(CORPUS, "input_inventory_resolution", {"input_rows": 2})

    def read_corpus_source(self, resolved):
        return self._call(CORPUS, "source_reading", [{"article_id": "bounded"}])

    def canonicalise(self, source):
        return self._call(CORPUS, "canonicalisation", [{"canonical": True}])

    def validate_corpus_identity(self, canonical):
        return self._validation(CORPUS, "identity_deduplication_validation")

    def validate_corpus_availability(self, canonical):
        return self._validation(CORPUS, "availability_time_validation")

    def publish_corpus(self, canonical):
        return self._call(CORPUS, "atomic_publication", self._corpus())

    def resolve_corpus_parent(self, parent, unit):
        return self._call(FEATURES, "corpus_parent_resolution", dict(parent))

    def prepare_pit_inputs(self, parent, unit):
        return self._call(FEATURES, "pit_filtering_join_preparation", {"rows": 2})

    def calculate_features(self, prepared):
        return self._call(FEATURES, "feature_calculation", [{"feature": 1}])

    def validate_feature_store(self, features):
        return self._validation(FEATURES, "feature_store_validation")

    def publish_feature_store(self, features):
        return self._call(FEATURES, "atomic_publication", self._features())

    def _call(self, stage, phase, result):
        self.calls.append((stage, phase))
        if self.fail == phase:
            raise RuntimeError(f"{phase} failed")
        return result

    def _validation(self, stage, phase):
        self.calls.append((stage, phase))
        return {
            "passed": self.validation_failure != phase,
            "validation_status": "PASSED",
        }

    def _corpus(self):
        return {
            "canonical_corpus_identity": "canonical-v1",
            "canonical_corpus_checksum": "CORPUS",
            "logical_manifest_checksum": "canonical-compatible-v1",
            "canonical_row_count": 2,
            "publication_result": "PUBLISHED",
            "validation_status": "PASSED",
        }

    def _features(self):
        return {
            "feature_store_artifact_checksum": "FEATURES",
            "canonical_corpus_identity": (
                "wrong" if self.wrong_ancestry else "canonical-v1"
            ),
            "canonical_corpus_checksum": "CORPUS",
            "pit_eligibility_policy_identity": "strict-collected-at-v1",
            "feature_schema_checksum": "SCHEMA",
            "row_count": 3,
            "manifest_publication_result": "PUBLISHED",
            "validation_status": "PASSED",
        }


def services(tmp_path):
    profile = dell_i5_10500_profile(
        source_git_commit="commit", generated_at="fixed"
    )
    ledger = ResourceLeaseLedger(
        profile=profile, path=tmp_path / "ledger.json",
        available_memory=lambda: 32 * GIB,
    )
    ledger.initialise_ledger()
    return profile, ledger


def run(tmp_path, requested, adapter):
    profile, ledger = services(tmp_path)
    result = execute_news_data_compute_run(
        plan=plan(*requested), adapter=adapter, machine_profile=profile,
        lease_ledger=ledger, runs_root=tmp_path / "runs",
        registry_path=tmp_path / "registry.json",
    )
    return result, ledger


def test_deterministic_run_stage_items_and_resource_defaults():
    value = plan(CORPUS, FEATURES)
    run_id = deterministic_news_data_run_id(value)
    assert run_id == deterministic_news_data_run_id(value)
    corpus = deterministic_news_data_item_id(
        run_id=run_id, stage=CORPUS, work_unit={"work_unit_id": "stage"},
        plan=value,
    )
    feature = deterministic_news_data_item_id(
        run_id=run_id, stage=FEATURES,
        work_unit={"work_unit_id": "stage"}, plan=value,
    )
    assert corpus != feature
    assert corpus == deterministic_news_data_item_id(
        run_id=run_id, stage=CORPUS, work_unit={"work_unit_id": "stage"},
        plan=value,
    )
    corpus_request = build_news_data_resource_request(
        run_id=run_id, item_id=corpus, stage=CORPUS,
        attempt_identity="corpus-attempt",
    )
    feature_request = build_news_data_resource_request(
        run_id=run_id, item_id=feature, stage=FEATURES,
        attempt_identity="feature-attempt",
    )
    assert (corpus_request.estimated_peak_ram_bytes,
            corpus_request.cpu_weight) == (4 * GIB, 1)
    assert (feature_request.estimated_peak_ram_bytes,
            feature_request.cpu_weight) == (6 * GIB, 2)
    assert corpus_request.estimate_source == "CONSERVATIVE_DEFAULT"
    assert feature_request.gpu_required is False


def test_empty_or_malformed_plan_fails_closed():
    with pytest.raises(ValueError, match="At least one"):
        plan()
    with pytest.raises(ValueError, match="Unsupported"):
        plan("UNKNOWN")


@pytest.mark.parametrize("stage", [CORPUS, FEATURES])
def test_compatible_stage_skips_before_activation(tmp_path, stage):
    adapter = Adapter(compatible=(stage,))
    result, ledger = run(tmp_path, (stage,), adapter)
    assert result["summary"]["reused_skipped_items"] == 1
    assert result["resource_summary"]["requests_created"] == 0
    assert ledger.read_ledger_status()["active_leases"] == []
    assert all(phase == "compatible" for _, phase in adapter.calls)


def test_new_corpus_then_feature_has_exact_parent_and_releases_leases(tmp_path):
    adapter = Adapter()
    result, ledger = run(tmp_path, (CORPUS, FEATURES), adapter)
    assert result["summary"]["newly_materialised_items"] == 2
    assert result["summary"]["corpus_parent_identity"] == "canonical-v1"
    assert result["summary"]["feature_store_parent_identities"][
        "canonical_corpus_identity"
    ] == "canonical-v1"
    assert result["resource_summary"]["requests_created"] == 2
    assert ledger.read_ledger_status()["active_leases"] == []
    assert result["registry"]["health"] == "HEALTHY"


def test_compatible_corpus_then_new_feature(tmp_path):
    result, _ = run(
        tmp_path, (CORPUS, FEATURES), Adapter(compatible=(CORPUS,))
    )
    assert result["summary"]["reused_skipped_items"] == 1
    assert result["summary"]["newly_materialised_items"] == 1
    assert result["resource_summary"]["requests_created"] == 1


@pytest.mark.parametrize(
    "phase",
    ["source_reading", "canonicalisation", "atomic_publication"],
)
def test_corpus_failure_releases_lease_and_blocks_feature(tmp_path, phase):
    adapter = Adapter(fail=phase)
    result, ledger = run(tmp_path, (CORPUS, FEATURES), adapter)
    assert result["summary"]["failed_items"] == 2
    assert result["summary"]["final_run_status"] == "FAILED"
    assert ledger.read_ledger_status()["active_leases"] == []
    assert (FEATURES, "feature_calculation") not in adapter.calls


def test_corpus_validation_failure_blocks_feature(tmp_path):
    adapter = Adapter(validation_failure="identity_deduplication_validation")
    result, ledger = run(tmp_path, (CORPUS, FEATURES), adapter)
    assert result["summary"]["failed_items"] == 2
    assert ledger.read_ledger_status()["active_leases"] == []
    assert (FEATURES, "feature_calculation") not in adapter.calls


@pytest.mark.parametrize(
    "phase", ["feature_calculation", "atomic_publication"]
)
def test_feature_failure_releases_lease(tmp_path, phase):
    result, ledger = run(tmp_path, (FEATURES,), Adapter(fail=phase))
    assert result["summary"]["failed_items"] == 1
    assert result["summary"]["final_run_status"] == "FAILED"
    assert ledger.read_ledger_status()["active_leases"] == []


def test_pit_validation_and_incompatible_ancestry_fail_closed(tmp_path):
    failed, _ = run(
        tmp_path / "validation", (FEATURES,),
        Adapter(validation_failure="feature_store_validation"),
    )
    assert failed["summary"]["final_run_status"] == "FAILED"
    wrong, _ = run(
        tmp_path / "ancestry", (FEATURES,), Adapter(wrong_ancestry=True)
    )
    assert wrong["summary"]["final_run_status"] == "FAILED"


def test_artifact_references_telemetry_summary_are_bounded(tmp_path):
    result, _ = run(tmp_path, (CORPUS, FEATURES), Adapter())
    root = Path(result["run_root"])
    serialized = (
        (root / "telemetry_spans.json").read_text()
        + (root / "news_data_summary.json").read_text()
    )
    assert "article text must never appear" not in serialized
    spans = json.loads((root / "telemetry_spans.json").read_text())["spans"]
    phases = {row["phase"] for row in spans}
    assert {
        "compatibility_resume_resolution", "input_inventory_resolution",
        "source_reading", "canonicalisation",
        "identity_deduplication_validation",
        "availability_time_validation", "atomic_publication",
        "corpus_parent_resolution", "pit_filtering_join_preparation",
        "feature_calculation", "feature_store_validation",
    } <= phases
    status = (root / "component_status.csv").read_text()
    assert "canonical-v1" in status
    assert "FEATURES" in status


def test_compatible_rerun_is_deterministic(tmp_path):
    first, _ = run(tmp_path / "first", (CORPUS,), Adapter(compatible=(CORPUS,)))
    second, _ = run(
        tmp_path / "second", (CORPUS,), Adapter(compatible=(CORPUS,))
    )
    assert first["run_id"] == second["run_id"]
    assert first["run_identity"] == second["run_identity"]


def _result_records(result):
    return json.loads(
        (Path(result["run_root"]) / "results.json").read_text()
    )["records"]


def test_production_shaped_canonical_rows_reach_shared_result(tmp_path):
    class ProductionShapedAdapter(Adapter):
        def _corpus(self):
            return {
                **super()._corpus(),
                "canonical_row_count": 486757,
                "source_row_count": 486757,
                "duplicate_group_count": 369509,
            }

    result, _ = run(tmp_path, (CORPUS,), ProductionShapedAdapter())
    record = _result_records(result)[0]
    assert record["status"] == "COMPLETE"
    assert record["counts"] == {
        "input_rows": 486757,
        "output_rows": 486757,
        "rows": 486757,
    }
    assert record["metrics"]["row_count"][
        "value"
    ] == 486757
    assert record["metrics"]["row_count"]["availability"] == "AVAILABLE"
    assert result["summary"]["failed_items"] == 0
    assert result["resource_summary"]["active_lease_count"] == 0
    assert result["registry"]["health"] == "HEALTHY"


def test_unknown_or_failed_rows_are_not_reported_as_zero(tmp_path):
    class UnknownRowsAdapter(Adapter):
        def _corpus(self):
            value = super()._corpus()
            value.pop("canonical_row_count")
            return value

    unknown, _ = run(tmp_path / "unknown", (CORPUS,), UnknownRowsAdapter())
    unknown_record = _result_records(unknown)[0]
    assert unknown_record["counts"] == {}
    assert unknown_record["metrics"]["row_count"]["availability"] == (
        "NOT_COMPUTED"
    )
    assert unknown_record["metrics"]["row_count"]["value"] is None

    failed, _ = run(
        tmp_path / "failed", (CORPUS,),
        Adapter(fail="canonicalisation"),
    )
    failed_record = _result_records(failed)[0]
    assert failed_record["status"] == "FAILED"
    assert failed_record["counts"] == {}
    assert failed_record["metrics"]["row_count"]["value"] is None
    assert failed_record["metrics"]["row_count"][
        "availability"
    ] == "NOT_COMPUTED"


def test_feature_row_mapping_and_result_checksum_remain_deterministic(
    tmp_path,
):
    first, _ = run(tmp_path / "first", (FEATURES,), Adapter())
    second, _ = run(tmp_path / "second", (FEATURES,), Adapter())
    first_record = _result_records(first)[0]
    second_record = _result_records(second)[0]
    assert first_record["counts"] == {"output_rows": 3, "rows": 3}
    assert first_record["metrics"]["row_count"]["value"] == 3
    assert first_record["logical_checksum"] == second_record[
        "logical_checksum"
    ]
