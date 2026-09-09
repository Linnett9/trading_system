from __future__ import annotations

from pathlib import Path

import pytest

from scripts.local import ds24_v3_lightgbm_ranking_policy_worker as worker


def test_ranking_worker_route_supports_only_group_aware_lightgbm_families() -> None:
    assert worker.assert_supported_family("lightgbm-rank-xendcg") == "lightgbm_rank_xendcg"
    assert worker.assert_supported_family("lightgbm_lambdarank") == "lightgbm_lambdarank"

    with pytest.raises(worker.LightGbmRankingWorkerError, match="UNSUPPORTED_FAMILY"):
        worker.assert_supported_family("random_forest")


def test_bounded_certification_preserves_query_group_contract(tmp_path: Path) -> None:
    result = worker.run_bounded_certification("lightgbm_rank_xendcg", tmp_path)

    assert result["state"] == "V3_CERTIFIED_READY"
    assert result["query_group_authority"]["contiguous_single_timestamp_groups"] is True
    assert result["query_group_authority"]["pit_eligible"] is True
    assert result["objective_smoke"]["continuous_score_per_eligible_symbol"] is True
    assert result["resume_proof"]["five_minute_scoring"] is True
    assert result["runtime_guards"]["paper_orders"] == 0
    assert result["runtime_guards"]["live_orders"] == 0
    assert result["runtime_guards"]["holdout_accessed"] is False


def test_non_certification_cli_fails_closed_without_launching_history(capsys) -> None:
    code = worker.main(["--family", "lightgbm_lambdarank"])
    captured = capsys.readouterr()

    assert code == 0
    assert "READY_FOR_BOUNDED_CERTIFICATION_ONLY" in captured.out
