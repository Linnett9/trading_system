from __future__ import annotations

import sys

from scripts.local import ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor as supervisor
from scripts.local import ds24_p8_r14_e3g_c2_r7_r40_full_tournament_activation as r40
from scripts.local import monitor_ds24_full_family_tournament as monitor


def test_r40_default_queue_covers_all_19_families_without_duplicates() -> None:
    queue = supervisor.selected_queue("")

    assert queue[:3] == ["rff_ridge", "huber", "mlp"]
    assert len(queue) == 19
    assert len(set(queue)) == 19
    assert set(queue) == set(supervisor.ALL_FAMILIES)


def test_r40_metrics_root_is_family_specific_for_v3_replays() -> None:
    assert supervisor.default_metrics_root_name("rff_ridge", "metrics_only_v3") == "metrics_only_v3_r37_rff_retry"
    assert supervisor.default_metrics_root_name("huber", "metrics_only") == "metrics_only_v3_r37_huber_replay"
    assert supervisor.default_metrics_root_name("mlp", "metrics_only_v3_r37_rff_retry") == "metrics_only_v3_r37_mlp_direct_gen7"
    assert supervisor.default_metrics_root_name("random_forest", "metrics_only_v3").startswith("metrics_only_v3_r40_random_forest")
    assert supervisor.default_metrics_root_name("elastic_net", "custom_namespace") == "custom_namespace"


def test_r40_execution_registry_routes_only_real_launch_workers() -> None:
    tabular = supervisor.execution_registry_row("random_forest")
    sequence = supervisor.execution_registry_row("DLinear")
    tft = supervisor.execution_registry_row("Temporal Fusion Transformer")

    assert tabular["worker_kind"] == "TABULAR"
    assert tabular["launch_enabled"] is True
    assert sequence["worker_kind"] == "PYTORCH_SEQUENCE"
    assert sequence["launch_enabled"] is True
    assert tft["worker_kind"] == "PYTORCH_SEQUENCE"
    assert tft["launch_enabled"] is False


def test_r40_terminal_fails_closed_when_v3_resolution_is_invalid() -> None:
    board = [
        {"family": "huber", "state": "RUNNING", "pid_alive": True},
        {"family": "DLinear", "state": "V3_CERTIFICATION_REQUIRED", "pid_alive": False},
    ]
    performance = {"status": "BLOCKED_NO_REAL_TARGET_RESOLUTION", "leaderboard_enabled": False}
    queue = {"resource_gate": {"blocked_reasons": []}}
    watchdog = {"persistent_watchdog_active": True}

    assert (
        r40.terminal_classification(board=board, performance=performance, queue=queue, watchdog=watchdog, sample={})
        == r40.TERMINAL_PERF_INVALID
    )


def test_r40_terminal_records_partial_blocks_when_tournament_can_continue(monkeypatch) -> None:
    monkeypatch.setattr(r40, "zero_guard", lambda: {"paper_orders_zero": True, "live_orders_zero": True, "holdout_accessed_false": True, "full_prediction_files_in_metrics_namespaces_zero": True})
    board = [
        {"family": "huber", "state": "RUNNING", "pid_alive": True},
        {"family": "DLinear", "state": "V3_CERTIFICATION_REQUIRED", "pid_alive": False},
    ]
    performance = {"status": "PASS_REAL_TARGET_PROBE", "leaderboard_enabled": False}
    queue = {"resource_gate": {"blocked_reasons": []}}
    watchdog = {"persistent_watchdog_active": True}

    assert (
        r40.terminal_classification(board=board, performance=performance, queue=queue, watchdog=watchdog, sample={})
        == r40.TERMINAL_PARTIAL
    )


def test_monitor_accepts_compact_eta_arguments(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["monitor", "--compact", "--eta-window-minutes", "30"])

    args = monitor.parse_args()

    assert args.compact is True
    assert args.eta_window_minutes == 30
