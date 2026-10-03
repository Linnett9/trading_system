from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.research.ml.ds24.clean_v2_contracts import stable_hash
from core.research.ml.ds24.clean_v2_package_cache import (
    DEFAULT_PACKAGE_CACHE_MAX_BYTES,
)
from core.research.ml.ds24.clean_v2_resources import (
    DELL_DISK_ADMISSION_MINIMUM_FREE_BYTES,
    DELL_DISK_ADMISSION_POLICY_ID,
    DELL_PREVIOUS_DISK_ADMISSION_POLICY_ID,
    DELL_R1_RECOVERY_RUN_ID,
    GIB,
    RESOURCE_POLICY,
    cache_minimum_post_write_free_bytes,
    disk_admission_policy,
    evaluate_disk_admission,
    evaluate_disk_write_feasibility,
    operational_scope_policy,
)
from scripts.local import ds24_clean_v2_supervisor as supervisor


STORAGE_CONTRACT = {
    "minimum_post_launch_free_gib": 15.0,
    "projected_tournament_gib_upper_bound": 8.0,
    "projected_sidecar_gib_upper_bound": 12.0,
}


def test_dell_total_floor_rejects_below_and_accepts_exactly_ten_gib() -> None:
    policy = disk_admission_policy(
        host="dell", storage_contract=STORAGE_CONTRACT, sidecar_complete=True
    )

    below = evaluate_disk_admission(policy, disk_free_bytes=10 * GIB - 1)
    exact = evaluate_disk_admission(policy, disk_free_bytes=10 * GIB)

    assert policy.policy_id == DELL_DISK_ADMISSION_POLICY_ID
    assert policy.previous_policy_id == DELL_PREVIOUS_DISK_ADMISSION_POLICY_ID
    assert policy.required_free_disk_bytes == 10_737_418_240
    assert below.safe is False
    assert below.margin_bytes == -1
    assert exact.safe is True
    assert exact.margin_bytes == 0


@pytest.mark.parametrize("free_gib", [11, 15, 18, 22])
def test_dell_above_ten_gib_has_no_hidden_legacy_floor(free_gib: int) -> None:
    policy = disk_admission_policy(
        host="dell", storage_contract=STORAGE_CONTRACT, sidecar_complete=True
    )

    decision = evaluate_disk_admission(
        policy, disk_free_bytes=free_gib * GIB
    )

    assert decision.safe is True
    assert policy.declared_campaign_growth_bytes == 8 * GIB
    assert policy.growth_forecasts_added_to_required_free_bytes is False
    assert policy.required_free_disk_bytes == 10 * GIB


def test_mac_disk_formula_is_unchanged() -> None:
    complete = disk_admission_policy(
        host="mac", storage_contract=STORAGE_CONTRACT, sidecar_complete=True
    )
    incomplete = disk_admission_policy(
        host="mac", storage_contract=STORAGE_CONTRACT, sidecar_complete=False
    )

    assert complete.required_free_disk_bytes == 23 * GIB
    assert incomplete.required_free_disk_bytes == 35 * GIB
    assert complete.growth_forecasts_added_to_required_free_bytes is True


def test_actual_next_write_still_fails_closed_at_remaining_floor() -> None:
    one_kib_margin = evaluate_disk_write_feasibility(
        disk_free_bytes=10 * GIB + 1024,
        required_remaining_bytes=10 * GIB,
        additional_write_bytes=1025,
    )
    unknown = evaluate_disk_write_feasibility(
        disk_free_bytes=20 * GIB,
        required_remaining_bytes=10 * GIB,
        additional_write_bytes=None,
    )

    assert one_kib_margin.safe is False
    assert one_kib_margin.projected_free_bytes == 10 * GIB - 1
    assert unknown.safe is False
    assert unknown.payload()["blocking_reasons"] == [
        "NEXT_WRITE_INCREMENTAL_BYTES_UNKNOWN"
    ]


def test_cache_and_memory_safeguards_are_unchanged() -> None:
    assert DEFAULT_PACKAGE_CACHE_MAX_BYTES == 7 * GIB // 4
    assert cache_minimum_post_write_free_bytes("dell") == 10 * GIB
    assert cache_minimum_post_write_free_bytes("mac") == 4 * GIB
    assert RESOURCE_POLICY.system_commit_headroom_gib == 5
    assert RESOURCE_POLICY.windows_job_aggregate_commit_limit_gib == 24
    assert RESOURCE_POLICY.soft_pause_available_physical_gib == 4
    assert RESOURCE_POLICY.emergency_available_physical_gib == 3
    assert RESOURCE_POLICY.readmission_available_physical_gib == 7


def test_admission_token_is_bound_to_exact_disk_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = disk_admission_policy(
        host="dell", storage_contract=STORAGE_CONTRACT, sidecar_complete=True
    ).payload()
    scope = supervisor._effective_operational_scope_policy("dell").payload()
    report = {
        "ready": True,
        "blocking_reasons": [],
        "run_id": "run",
        "host_role": "dell",
        "feature_authority_hash": "feature",
        "target_authority_hash": "target",
        "target_contract_hash": "target-contract",
        "static_authority_bundle_sha256": "bundle",
        "clean_source_hash": "source",
        "refit_policy": "refit",
        "launch_families": ["random_forest", "momentum"],
        "maximum_model_workers": 2,
        "operational_scope_policy": scope,
        "resource_policy": {"resource_policy_id": "memory-policy"},
        "disk_admission_policy": policy,
    }
    admission_core = {
        key: report[key]
        for key in (
            "run_id",
            "host_role",
            "feature_authority_hash",
            "target_authority_hash",
            "target_contract_hash",
            "static_authority_bundle_sha256",
            "clean_source_hash",
            "refit_policy",
            "launch_families",
            "maximum_model_workers",
            "operational_scope_policy",
            "resource_policy",
            "disk_admission_policy",
        )
    }
    admission_core["hostname"] = "test-host"
    admission_core["issued_at_utc"] = "2026-10-02T00:00:00+00:00"
    admission_path = tmp_path / "manual_admission_dell.json"
    admission_path.write_text(
        json.dumps(
            {
                **admission_core,
                "admission_token": stable_hash(admission_core),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(supervisor, "_host_path", lambda _kind, _host: admission_path)

    validated = supervisor._validate_admission_against_report("dell", report)
    assert validated["disk_admission_policy"] == policy

    stale = json.loads(admission_path.read_text(encoding="utf-8"))
    stale.pop("admission_token")
    stale["disk_admission_policy"] = {
        **policy,
        "required_free_disk_bytes": 23 * GIB,
    }
    stale["admission_token"] = stable_hash(stale)
    admission_path.write_text(json.dumps(stale), encoding="utf-8")

    with pytest.raises(
        RuntimeError, match="MANUAL_PREFLIGHT_ADMISSION_STALE:disk_admission_policy"
    ):
        supervisor._validate_admission_against_report("dell", report)


def test_dell_floor_constant_matches_operator_authorisation() -> None:
    assert DELL_DISK_ADMISSION_MINIMUM_FREE_BYTES == 10_737_418_240


def test_r1_dell_scope_is_rf_momentum_only_and_two_workers() -> None:
    policy = supervisor._effective_operational_scope_policy("dell")

    assert policy.run_id == DELL_R1_RECOVERY_RUN_ID
    assert policy.launch_families == ("random_forest", "momentum")
    assert policy.maximum_model_workers == 2
    assert supervisor._launch_families("dell") == [
        "random_forest",
        "momentum",
    ]
    assert supervisor._maximum_model_workers("dell") == 2
    assert "equal_weight_no_model" in policy.excluded_owned_families


def test_r1_scope_payload_is_stable_across_owned_family_orderings() -> None:
    supervisor_order = (
        "random_forest",
        "huber",
        "elastic_net_C5",
        "momentum",
        "transformer",
    )
    worker_contract_order = (
        "random_forest",
        "transformer",
        "huber",
        "elastic_net_C5",
        "momentum",
    )

    supervisor_scope = operational_scope_policy(
        host="dell",
        run_id=DELL_R1_RECOVERY_RUN_ID,
        ordered_owned_families=supervisor_order,
        configured_maximum_workers=3,
    )
    worker_scope = operational_scope_policy(
        host="dell",
        run_id=DELL_R1_RECOVERY_RUN_ID,
        ordered_owned_families=worker_contract_order,
        configured_maximum_workers=3,
    )

    assert supervisor_scope.payload() == worker_scope.payload()


def test_non_r1_dell_and_mac_scopes_remain_contract_driven() -> None:
    dell = operational_scope_policy(
        host="dell",
        run_id="DS24_CLEAN_V2_TOURNAMENT_R2_20261002",
        ordered_owned_families=("random_forest", "huber", "momentum"),
        configured_maximum_workers=3,
    )
    mac = operational_scope_policy(
        host="mac",
        run_id=DELL_R1_RECOVERY_RUN_ID,
        ordered_owned_families=("lightgbm_rank_xendcg", "momentum_transformer"),
        configured_maximum_workers=1,
    )

    assert dell.launch_families == ("random_forest", "huber", "momentum")
    assert dell.maximum_model_workers == 3
    assert mac.launch_families == (
        "lightgbm_rank_xendcg",
        "momentum_transformer",
    )
    assert mac.maximum_model_workers == 1
