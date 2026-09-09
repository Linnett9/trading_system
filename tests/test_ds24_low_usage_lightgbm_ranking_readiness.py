from __future__ import annotations

from pathlib import Path

from core.research.ml.registries.io import canonical_hash
from scripts.local import ds24_lightgbm_ranking_readiness as readiness


def test_lightgbm_dependency_and_registered_objectives() -> None:
    dep = readiness.dependency_state()
    entries = readiness.registry_entries()

    assert dep["installed"] is True
    assert dep["version"] == "4.6.0"
    assert entries["lightgbm_rank_xendcg"]["objective"] == "rank_xendcg"
    assert entries["lightgbm_lambdarank"]["objective"] == "lambdarank"
    for family, entry in entries.items():
        config = readiness.fixed_configuration(family)
        assert canonical_hash(config) == entry["fitting_configuration_checksum"]
        assert config["parameters"]["n_jobs"] == 1
        assert config["parameters"]["random_state"] == 1729


def test_query_groups_are_five_minute_timestamp_cross_sections() -> None:
    dataset, labels = readiness.grouped_dataset()
    proof = readiness.query_group_proof(dataset)

    assert dataset["valid"] is True
    assert labels["label_contract_identity"] == "within_date_quintile_relevance_v1"
    assert labels["contract_version"] == "mature_training_integer_relevance.v1"
    assert proof["group_lengths_reconcile_to_rows"] is True
    assert proof["strict_timestamp_groups"] is True
    assert proof["contiguous_single_timestamp_groups"] is True
    assert proof["chronological_order"] is True
    assert proof["pit_eligible"] is True
    assert proof["group_size_vector"] == [6, 6, 6, 6]


def test_tiny_synthetic_objective_smokes_return_one_score_per_symbol(tmp_path: Path) -> None:
    dataset, _ = readiness.grouped_dataset()
    for family in readiness.SCOPED_FAMILIES:
        smoke = readiness.objective_smoke(family, dataset, tmp_path)

        assert smoke["status"] == "PASS"
        assert smoke["objective"] == readiness.registry_entries()[family]["objective"]
        assert smoke["training_rows"] == 12
        assert smoke["training_group_sizes"] == [6, 6]
        assert smoke["prediction_rows"] == 12
        assert smoke["continuous_score_per_eligible_symbol"] is True


def test_focused_resume_and_v3_namespace_probe(tmp_path: Path) -> None:
    resume = readiness.resume_proof()
    probe = readiness.v3_namespace_probe(tmp_path, "lightgbm_rank_xendcg")

    assert resume["status"] == "PASS"
    assert resume["daily_session_refit"] is True
    assert resume["five_minute_scoring"] is True
    assert resume["fully_completed_refit_packages"] == 1
    assert resume["partial_or_uncommitted_refit_packages"] == 1
    assert resume["first_uncommitted_T"]
    assert probe["v3_metrics_only_supported"] is True
    assert probe["namespace_lease_supported"] is True
    assert probe["paper_orders"] == 0
    assert probe["live_orders"] == 0
    assert probe["holdout_accessed"] is False


def test_manifest_classifies_two_lightgbm_rankers_ready() -> None:
    manifest = readiness.build_manifest()
    states = {row["family"]: row["state"] for row in manifest["families"]}

    assert states == {
        "lightgbm_rank_xendcg": "V3_CERTIFIED_READY",
        "lightgbm_lambdarank": "V3_CERTIFIED_READY",
    }
    assert manifest["worker_launches"] == 0
    assert manifest["tournament_training_runs"] == 0
    assert manifest["future_launch_command"] == ""
    assert "generic tabular policy worker" in manifest["launch_note"]
