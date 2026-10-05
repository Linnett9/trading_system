from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from core.research.ml.ds24.cross_host_throughput import (
    PackageProfile,
    PackageProfileBuilder,
    ProfileContractError,
    StageObservation,
    StageStatus,
    ThroughputStage,
    StageResourceObservation,
    WorkloadCounters,
    distributed_critical_path,
    summarize_family_profiles,
    uninstrumented_stage_observations,
)
from scripts.local.ds24_v11x_profile_report import (
    _load_profiles,
    _parse_remaining,
    build_report,
)
from scripts.local import ds24_clean_v2_family_benchmark as family_benchmark


def _profile(
    *, host: str = "dell", family: str = "random_forest", wall: float = 100.0
) -> PackageProfile:
    stages = list(uninstrumented_stage_observations())
    stages[0] = StageObservation(
        ThroughputStage.SCHEDULER_DELAY, StageStatus.MEASURED, 10.0
    )
    stages[11] = StageObservation(
        ThroughputStage.MODEL_FIT, StageStatus.MEASURED, 50.0
    )
    stages[18] = StageObservation(
        ThroughputStage.ATOMIC_LEDGER_PUBLICATION, StageStatus.MEASURED, 5.0
    )
    return PackageProfile(
        host=host,
        family=family,
        run_id="run",
        package_id=f"package-{wall}",
        refit_timestamp="2026-01-01T00:00:00+00:00",
        source_hash="source",
        scientific_authority_hash="science",
        wall_seconds=wall,
        stages=tuple(stages),
    )


def test_profile_requires_explicit_status_for_all_23_stages() -> None:
    profile = _profile()
    assert len(profile.stages) == 23
    round_trip = PackageProfile.from_payload(profile.payload())
    assert round_trip == profile

    with pytest.raises(ProfileContractError, match="Stage coverage mismatch"):
        replace(profile, stages=profile.stages[:-1])


def test_profile_rejects_wrong_host_ownership_and_overlapping_totals() -> None:
    with pytest.raises(ProfileContractError, match="not owned"):
        _profile(host="mac", family="random_forest")

    profile = _profile(wall=100.0)
    with pytest.raises(ProfileContractError, match="exceeds package wall"):
        replace(profile, wall_seconds=59.0)


def test_family_summary_has_required_percentiles_and_stage_percentages() -> None:
    summary = summarize_family_profiles(
        [_profile(wall=100.0), _profile(wall=200.0)], remaining_packages=4
    )

    assert summary["sec_per_refit_p50"] == 150.0
    assert summary["sec_per_refit_p90"] == 190.0
    assert summary["sec_per_refit_p99"] == 199.0
    assert summary["wait_percent"] == pytest.approx(20.0 / 300.0 * 100.0)
    assert summary["fit_percent"] == pytest.approx(100.0 / 300.0 * 100.0)
    assert summary["publish_percent"] == pytest.approx(10.0 / 300.0 * 100.0)
    assert summary["primary_bottleneck"] == "model_fit"
    assert summary["estimated_remaining_hours"] == pytest.approx(1.0 / 6.0)
    assert "data_parquet_read" in summary["uninstrumented_stages"]


def test_summary_refuses_mixed_authority_or_host_family_evidence() -> None:
    with pytest.raises(ProfileContractError, match="authority"):
        summarize_family_profiles(
            [_profile(), replace(_profile(wall=101.0), source_hash="changed")]
        )
    with pytest.raises(ProfileContractError, match="host/family"):
        summarize_family_profiles(
            [
                _profile(),
                _profile(host="dell", family="huber", wall=101.0),
            ]
        )


def test_distributed_path_reports_partial_evidence_without_guessing() -> None:
    known = summarize_family_profiles([_profile()], remaining_packages=10)
    unknown = summarize_family_profiles(
        [_profile(host="dell", family="huber")], remaining_packages=None
    )
    result = distributed_critical_path([known, unknown])

    assert result["classification"] == "PARTIAL_EVIDENCE_DO_NOT_DEPLOY"
    assert result["expected_final_straggler"] == {
        "host": "dell",
        "family": "random_forest",
    }


def test_uninstrumented_stage_cannot_carry_inferred_timing() -> None:
    with pytest.raises(ProfileContractError, match="cannot have wall_seconds"):
        StageObservation(
            ThroughputStage.DATA_PARQUET_READ,
            StageStatus.UNINSTRUMENTED,
            wall_seconds=1.0,
        )


def test_builder_completes_missing_stages_without_inference() -> None:
    builder = PackageProfileBuilder(
        host="mac",
        family="momentum_transformer",
        run_id="run",
        package_id="package",
        refit_timestamp="2026-01-01T00:00:00+00:00",
        source_hash="source",
        scientific_authority_hash="science",
    )
    builder.record(
        ThroughputStage.TENSOR_ARRAY_CONVERSION,
        wall_seconds=2.0,
        resource=StageResourceObservation(
            cpu_seconds=1.5,
            rss_growth_bytes=-1024,
            swap_growth_bytes=0,
        ),
    )
    builder.mark_not_applicable(ThroughputStage.CACHE_WRITE_EVICTION)

    profile = builder.build(
        wall_seconds=3.0,
        workload=WorkloadCounters(sequence_endpoint_count=128, batch_count=4),
    )
    by_stage = {row.stage: row for row in profile.stages}
    assert by_stage[ThroughputStage.TENSOR_ARRAY_CONVERSION].status is StageStatus.MEASURED
    assert by_stage[ThroughputStage.CACHE_WRITE_EVICTION].status is StageStatus.NOT_APPLICABLE
    assert by_stage[ThroughputStage.DATA_PARQUET_READ].status is StageStatus.UNINSTRUMENTED

    with pytest.raises(ProfileContractError, match="already recorded"):
        builder.record(ThroughputStage.TENSOR_ARRAY_CONVERSION, wall_seconds=1.0)


def test_summary_rejects_duplicate_package_evidence() -> None:
    profile = _profile()
    with pytest.raises(ProfileContractError, match="Duplicate package"):
        summarize_family_profiles([profile, profile])


def test_workload_counters_reject_inconsistent_cache_evidence() -> None:
    with pytest.raises(ProfileContractError, match="cache_miss_reason"):
        WorkloadCounters(cache_hit=True, cache_miss_reason="NOT_FOUND")
    with pytest.raises(ProfileContractError, match="rows_read"):
        WorkloadCounters(rows_read=-1)


def test_read_only_reporter_loads_profile_and_preserves_partial_classification(
    tmp_path: Path,
) -> None:
    profile = _profile()
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile.payload()), encoding="utf-8")

    loaded = _load_profiles(path)
    report = build_report(
        loaded,
        _parse_remaining(["dell:random_forest=411"]),
    )

    assert loaded == [profile]
    assert report["classification"] == "V11X_CROSS_HOST_PROFILE_PARTIAL"
    assert report["distributed_critical_path"]["classification"] == (
        "PARTIAL_EVIDENCE_DO_NOT_DEPLOY"
    )
    random_forest = next(
        row
        for row in report["performance_matrix"]
        if row["family"] == "random_forest"
    )
    assert random_forest["remaining_packages"] == 411
    assert len(report["performance_matrix"]) == 7


def test_rf_thread_override_is_explicit_bounded_and_does_not_mutate_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = {"parameters": {"n_jobs": 1, "random_state": 23}}
    monkeypatch.setattr(
        family_benchmark,
        "_family_contract",
        lambda _family: (authority, "FULL_CLEAN"),
    )

    resolved, lane = family_benchmark._family_contract_with_runtime_overrides(
        "random_forest", random_forest_n_jobs=3
    )

    assert lane == "FULL_CLEAN"
    assert resolved["parameters"]["n_jobs"] == 3
    assert authority["parameters"]["n_jobs"] == 1
    with pytest.raises(ValueError, match="only for random_forest"):
        family_benchmark._family_contract_with_runtime_overrides(
            "huber", random_forest_n_jobs=2
        )
    with pytest.raises(ValueError, match="one of 1, 2, or 3"):
        family_benchmark._family_contract_with_runtime_overrides(
            "random_forest", random_forest_n_jobs=4
        )
