from __future__ import annotations

from pathlib import Path

import pytest

from scripts.local import ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor as supervisor


def _row(family: str, state: str) -> dict:
    return {
        "family": family,
        "state": state,
        "pid_alive": False,
        "duplicate_worker_count": 0,
        "heavy": family in supervisor.HEAVY_FAMILIES,
    }


def test_excluded_family_is_not_admitted_from_ready_queue(tmp_path: Path) -> None:
    board = [_row("PatchTST", "V3_CERTIFIED_READY"), _row("iTransformer", "V3_CERTIFIED_READY")]

    plan = supervisor.certified_queue_admission_plan(
        board,
        family_queue="PatchTST,iTransformer",
        excluded_families=["PatchTST"],
        retired_families_path=tmp_path / "retired.json",
    )

    assert plan["first_eligible_family"] == "iTransformer"
    assert [row["family"] for row in plan["eligible_families"]] == ["iTransformer"]
    assert {row["family"]: row["reason"] for row in plan["skipped_families"]}["PatchTST"] == "EXCLUDED_BY_OPERATOR"


def test_excluded_family_is_not_recovered_from_crashed_recoverable_queue(tmp_path: Path) -> None:
    board = [_row("PatchTST", "CRASHED_RECOVERABLE"), _row("iTransformer", "V3_CERTIFIED_READY")]

    plan = supervisor.certified_queue_admission_plan(
        board,
        family_queue="PatchTST,iTransformer",
        admit_crashed_recoverable=True,
        excluded_families=["PatchTST"],
        retired_families_path=tmp_path / "retired.json",
    )

    assert plan["first_eligible_family"] == "iTransformer"
    assert {row["family"]: row["reason"] for row in plan["skipped_families"]}["PatchTST"] == "EXCLUDED_BY_OPERATOR"


def test_excluded_sequence_family_is_not_restarted(tmp_path: Path, monkeypatch) -> None:
    spawned = {"called": False}

    def fake_popen(*_args, **_kwargs):
        spawned["called"] = True
        raise AssertionError("excluded sequence family must not spawn")

    monkeypatch.setattr(supervisor.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="DS24_R54_EXCLUDED_FAMILY_LAUNCH_REFUSED:PatchTST"):
        supervisor.launch_family(
            "PatchTST",
            7,
            evaluation_version="v3",
            metrics_root_name="metrics_only_v3",
            forward_contract_admission={"admitted": True},
            excluded_families=["PatchTST"],
            retired_families_path=tmp_path / "retired.json",
        )

    assert spawned["called"] is False


def test_exclusion_wins_when_explicit_queue_includes_family(tmp_path: Path) -> None:
    board = [
        _row("PatchTST", "V3_CERTIFIED_READY"),
        {"family": "Transformer", "state": "RUNNING", "pid_alive": True, "duplicate_worker_count": 0, "heavy": True},
        _row("iTransformer", "V3_CERTIFIED_READY"),
    ]

    assert (
        supervisor.next_ready_family(
            board,
            family_queue="PatchTST,Transformer,iTransformer",
            excluded_families=["PatchTST"],
            retired_families_path=tmp_path / "retired.json",
        )
        == "iTransformer"
    )


def test_exclusion_matching_is_case_and_spacing_normalized(tmp_path: Path) -> None:
    board = [_row("PatchTST", "V3_CERTIFIED_READY"), _row("iTransformer", "V3_CERTIFIED_READY")]

    plan = supervisor.certified_queue_admission_plan(
        board,
        family_queue="PatchTST,iTransformer",
        excluded_families=[" patchtst "],
        retired_families_path=tmp_path / "retired.json",
    )

    assert plan["excluded_families"] == ["PatchTST"]
    assert plan["first_eligible_family"] == "iTransformer"


def test_itransformer_remains_launchable_when_patchtst_is_excluded(tmp_path: Path) -> None:
    assert supervisor.family_is_launchable(
        "iTransformer",
        excluded_families=["patchtst"],
        retired_families_path=tmp_path / "retired.json",
    )
