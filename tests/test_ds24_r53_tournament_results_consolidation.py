from __future__ import annotations

import json
from pathlib import Path

from scripts.local.ds24_r53_tournament_results_consolidation import (
    OUTPUT_FILES,
    REQUESTED_FAMILIES,
    R53_DIRNAME,
    build_rows,
    run,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def v3_summary(family: str, mean_ic: float = 0.2, sharpe: float = 1.5, start: str = "2025-01-02T14:35:00+00:00") -> dict[str, object]:
    return {
        "family": family,
        "evaluation_contract_hash": "c5d24ca5ba4182d6e9080a7ca98b0c32a054bffe77a56fd09052de516cae84bc",
        "evaluation_contract_id": "DS24_RESOLVED_PERFORMANCE_V3_STAGGERED_SLEEVE_RANK_IC_CONTRACT",
        "evaluation_contract_version": "V3",
        "first_resolved_decision_timestamp": start,
        "last_resolved_decision_timestamp": "2026-06-30T19:00:00+00:00",
        "coverage": {"pending_score_rows": 0, "terminal_censored_rows": 12, "eligible_resolved_fraction": 0.75},
        "resolved_performance_rows": 100,
        "rank_ic": {
            "mean_spearman_rank_ic": mean_ic,
            "median_spearman_rank_ic": mean_ic - 0.01,
            "daily_mean_spearman_rank_ic": mean_ic,
            "positive_fraction": 0.75,
            "valid_timestamps": 100,
            "dependence_aware_95_ci": {"lower": mean_ic - 0.1, "upper": mean_ic + 0.1, "mean": mean_ic, "standard_error": 0.01, "lag": 12},
            "inference_method": "newey_west_hac_lag_12",
        },
        "returns": {
            "annualized_return_from_daily_returns": 0.4,
            "annualized_volatility_from_daily_returns": 0.2,
            "cumulative_gross_return": 1.5,
            "cumulative_net_return": 1.5,
            "daily_return_rows": 80,
            "daily_sharpe": sharpe,
            "maximum_drawdown": -0.02,
            "mean_turnover": 0.3,
            "portfolio_contract": "twelve_staggered_equal_capital_sleeves",
            "total_estimated_costs": 0.0,
            "win_rate": 0.6,
        },
        "status": "PROVISIONAL",
    }


def legacy_summary(family: str) -> dict[str, object]:
    return {
        "family": family,
        "accepted_performance_results": {"accepted": False, "status": "PROVISIONAL_UNVALIDATED_R33"},
        "coverage": {
            "first_decision_timestamp": "2024-01-02T14:35:00+00:00",
            "last_decision_timestamp": "2026-06-30T20:00:00+00:00",
        },
        "information_coefficient": {"mean_spearman_rank_ic": None, "pearson_ic": None, "ic_observation_count": 0},
        "portfolio_performance": {
            "top_n_equal_weight": {
                "annualized_return": 1.0,
                "annualized_volatility": 0.2,
                "cumulative_gross_return": 2.0,
                "cumulative_net_return": 2.0,
                "estimated_transaction_costs": 0.0,
                "first_resolved_decision_timestamp": "2024-01-03T14:35:00+00:00",
                "hit_rate": 0.7,
                "last_resolved_decision_timestamp": "2026-06-30T19:00:00+00:00",
                "maximum_drawdown": -0.1,
                "resolved_portfolio_observations": 30,
                "resolved_session_count": 30,
                "sharpe_ratio": 5.0,
                "turnover": 0.2,
            }
        },
        "status": "PROVISIONAL_UNVALIDATED_R33",
    }


def make_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    stage = tmp_path / "stage"
    worker = stage / "r7_r14_policy_workers"
    mac = tmp_path / "mac_aux_runs" / "queue=DS24_MAC_AUX_NINE_FAMILY_R1"
    output = stage / R53_DIRNAME
    families = []
    for family, _ in REQUESTED_FAMILIES:
        owner = "DELL"
        owner_state = "DELL_OWNED"
        ownership_state = "DELL_OWNED"
        readiness = "READY_TO_LAUNCH"
        if family in {"lightgbm_rank_xendcg", "lightgbm_lambdarank", "DLinear"}:
            owner = "MAC"
            owner_state = "MAC_RUNNING"
            ownership_state = "MAC_OWNED"
        if family == "lightgbm_rank_xendcg":
            owner_state = "COMPLETE_IMPORTED"
            ownership_state = "COMPLETE_IMPORTED"
        if family == "Temporal Fusion Transformer":
            owner = "UNASSIGNED"
            owner_state = "UNASSIGNED"
            ownership_state = "UNASSIGNED"
            readiness = "CONFIGURATION_AUTHORITY_REQUIRED"
        families.append(
            {
                "family": family,
                "execution_owner": owner,
                "owner_state": owner_state,
                "ownership_state": ownership_state,
                "readiness_state": readiness,
                "dell_eligible": owner == "DELL",
                "mac_eligible": owner == "MAC",
                "ownership_reason": "R51 LambdaRank transfer files missing" if family == "lightgbm_lambdarank" else "",
            }
        )
    write_json(stage / "R51_cross_host_ownership_state.json", {"families": families})
    write_json(stage / "R42_full_family_readiness_matrix.json", {"families": families})
    for family in ["elastic_net", "huber", "rff_ridge"]:
        metrics_root = worker / family / f"metrics_only_v3_terminal_{family}"
        start = "2025-04-02T14:35:00+00:00" if family == "elastic_net" else "2016-02-02T14:35:00+00:00"
        write_json(metrics_root / "resolved_performance_summary_v3.json", v3_summary(family, mean_ic=0.3, sharpe=2.0, start=start))
        write_json(worker / family / "progress.json", {"family": family, "last_completed_T": "2026-06-30T20:00:00+00:00", "metrics_root": metrics_root.as_posix()})
    stale = worker / "elastic_net" / "metrics_only_v3_stale"
    write_json(stale / "resolved_performance_summary_v3.json", v3_summary("elastic_net", mean_ic=0.99, sharpe=9.0, start="2025-09-11T14:35:00+00:00"))
    write_json(worker / "random_forest" / "metrics_only_v3_running" / "resolved_performance_summary_v3.json", v3_summary("random_forest", mean_ic=0.4, sharpe=3.0))
    write_json(worker / "random_forest" / "progress.json", {"family": "random_forest", "last_completed_T": "2025-01-02T20:00:00+00:00", "metrics_root": (worker / "random_forest" / "metrics_only_v3_running").as_posix()})
    for family in ["ridge_policy_v1_control", "pca_ridge_policy_v1_control", "spline_additive_ridge"]:
        write_json(worker / family / "r31_performance_summary.json", legacy_summary(family))
    write_json(mac / "family=lightgbm_rank_xendcg" / "metrics_only_v3" / "resolved_performance_summary_v3.json", v3_summary("lightgbm_rank_xendcg", mean_ic=0.5, sharpe=4.0))
    write_json(mac / "family=lightgbm_rank_xendcg" / "ensemble_oof_scores_manifest_v2.json", {"family": "lightgbm_rank_xendcg"})
    write_json(worker / "PatchTST" / "metrics_only_v3_synthetic" / "resolved_performance_summary_v3.json", v3_summary("PatchTST", mean_ic=0.8, sharpe=8.0))
    write_json(stage / "R52_patchtst_real_path_smoke" / "PatchTST" / "metrics_only_v3_r52_smoke" / "resolved_performance_summary_v3.json", v3_summary("PatchTST", mean_ic=0.1, sharpe=1.0))
    write_json(worker / "PatchTST" / "progress.json", {"family": "PatchTST", "last_completed_T": "2025-01-02T20:00:00+00:00", "metrics_root": (stage / "R52_patchtst_real_path_smoke" / "PatchTST" / "metrics_only_v3_r52_smoke").as_posix()})
    return stage, worker, mac, output


def row_by_family(rows: list[dict[str, object]], family: str) -> dict[str, object]:
    return next(row for row in rows if row["family"] == family)


def test_exactly_19_requested_families_and_deterministic_order(tmp_path: Path) -> None:
    stage, worker, mac, _ = make_fixture(tmp_path)
    rows, _ = build_rows(stage, worker, mac)
    assert [row["family"] for row in rows] == [family for family, _ in REQUESTED_FAMILIES]
    assert len(rows) == 19


def test_terminal_v3_beats_stale_summary_and_zero_cost_warning(tmp_path: Path) -> None:
    stage, worker, mac, _ = make_fixture(tmp_path)
    rows, _ = build_rows(stage, worker, mac)
    elastic = row_by_family(rows, "elastic_net")
    assert "metrics_only_v3_terminal_elastic_net" in str(elastic["result_source_path"])
    assert elastic["performance_status"] == "ACCEPTED_FINAL"
    assert elastic["economic_cost_classification"] == "ZERO_COST_ECONOMICS"
    assert "ZERO_COST_WITH_NONZERO_TURNOVER" in str(elastic["warnings"])


def test_imported_mac_result_and_parity_failure_does_not_overwrite_xendcg(tmp_path: Path) -> None:
    stage, worker, mac, _ = make_fixture(tmp_path)
    write_json(stage / "R999_xendcg_prospective_paper_parity_failure.json", {"family": "lightgbm_rank_xendcg", "classification": "FAIL"})
    rows, _ = build_rows(stage, worker, mac)
    xendcg = row_by_family(rows, "lightgbm_rank_xendcg")
    assert xendcg["performance_status"] == "ACCEPTED_IMPORTED_FINAL"
    assert xendcg["source_host"] == "MAC"
    assert xendcg["imported_result"] is True


def test_running_and_legacy_excluded_from_accepted_final_leaderboard(tmp_path: Path) -> None:
    stage, worker, mac, output = make_fixture(tmp_path)
    summary = run(stage, worker, mac, output)
    leaderboard = (output / "R53_completed_final_results_leaderboard.csv").read_text(encoding="utf-8")
    assert "random_forest" not in leaderboard
    assert "ridge_policy_v1_control" not in leaderboard
    assert summary["requested_families"] == 19


def test_missing_external_lambdarank_separated_from_scientific_mac_state(tmp_path: Path) -> None:
    stage, worker, mac, _ = make_fixture(tmp_path)
    rows, _ = build_rows(stage, worker, mac)
    lambdarank = row_by_family(rows, "lightgbm_lambdarank")
    assert lambdarank["r53_result_state"] == "EXTERNAL_MAC_RESULT_NOT_LOCALLY_AVAILABLE"
    assert lambdarank["execution_host"] == "MAC"


def test_sequence_synthetic_evidence_excluded_when_real_post_r52_namespace_selected(tmp_path: Path) -> None:
    stage, worker, mac, _ = make_fixture(tmp_path)
    rows, _ = build_rows(stage, worker, mac)
    patchtst = row_by_family(rows, "PatchTST")
    assert "R52_patchtst_real_path_smoke" in str(patchtst["result_source_path"])
    assert patchtst["mean_spearman_rank_ic"] == 0.1


def test_missing_evidence_retained_and_no_metrics_invented(tmp_path: Path) -> None:
    stage, worker, mac, _ = make_fixture(tmp_path)
    rows, _ = build_rows(stage, worker, mac)
    tft = row_by_family(rows, "Temporal Fusion Transformer")
    assert tft["r53_result_state"] == "CONFIGURATION_AUTHORITY_REQUIRED"
    assert tft.get("sharpe") is None


def test_differing_evaluation_window_classification(tmp_path: Path) -> None:
    stage, worker, mac, _ = make_fixture(tmp_path)
    rows, _ = build_rows(stage, worker, mac)
    imported = row_by_family(rows, "lightgbm_rank_xendcg")
    elastic = row_by_family(rows, "elastic_net")
    assert imported["comparability_classification"] == "IMPORTED_RETAINED_OOF_DIFFERENT_WINDOW"
    assert elastic["comparability_classification"] == "V3_DIFFERENT_EVALUATION_WINDOWS"
    assert "DIFFERENT_WINDOW_FROM_RANKING_FAMILIES" in str(elastic["warnings"])


def test_sha_manifest_and_no_writes_outside_r53_output_root(tmp_path: Path) -> None:
    stage, worker, mac, output = make_fixture(tmp_path)
    run(stage, worker, mac, output)
    manifest = json.loads((output / "R53_source_evidence_manifest.json").read_text(encoding="utf-8"))
    assert manifest["families"]["elastic_net"]["selected"]["file_sha256"]
    produced = sorted(path.name for path in output.iterdir())
    assert produced == sorted(OUTPUT_FILES)
    outside_files = [path for path in stage.rglob("R53_*") if R53_DIRNAME not in path.parts]
    assert outside_files == []
