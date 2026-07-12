from __future__ import annotations

import json

import pytest

from core.research.ml.immutable_runs import (
    champion_pointer_path,
    deterministic_run_id,
    immutable_run_dir,
    is_complete_run_dir,
    latest_completed_pointer_path,
    preserve_immutable_run,
    update_champion_pointer,
)


def test_deterministic_run_id_changes_with_identity_fields():
    identity = {
        "dataset_hash": "dataset-a",
        "model_input_hash": "input-a",
        "feature_columns": ["alpha", "beta"],
        "target_label_name": "should_reduce_exposure",
    }

    assert deterministic_run_id("exposure_ml", identity) == deterministic_run_id(
        "exposure_ml",
        dict(identity),
    )
    assert deterministic_run_id("exposure_ml", identity) != deterministic_run_id(
        "exposure_ml",
        {**identity, "dataset_hash": "dataset-b"},
    )
    assert deterministic_run_id("exposure_ml", identity) != deterministic_run_id(
        "exposure_ml",
        {**identity, "feature_columns": ["beta", "alpha"]},
    )
    assert deterministic_run_id("exposure_ml", identity) != deterministic_run_id(
        "exposure_ml",
        {**identity, "target_label_name": "other_target"},
    )


def test_preserve_immutable_run_marks_complete_and_updates_latest(tmp_path):
    output_dir = tmp_path / "reports" / "model"
    artifact = output_dir / "metrics.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({"ok": True}), encoding="utf-8")
    identity = {"dataset_hash": "dataset-a"}
    run_id = deterministic_run_id("exposure_ml", identity)

    record = preserve_immutable_run(
        output_dir=output_dir,
        run_id=run_id,
        kind="exposure_ml",
        identity=identity,
        artifact_paths=(artifact,),
    )

    manifest = json.loads(record.manifest_path.read_text(encoding="utf-8"))
    latest = json.loads(latest_completed_pointer_path(output_dir).read_text())
    assert manifest["run_status"] == "complete"
    assert manifest["run_id"] == run_id
    assert latest["run_id"] == run_id
    assert (record.run_dir / "metrics.json").exists()
    assert is_complete_run_dir(record.run_dir) is True


def test_completed_immutable_run_is_not_overwritten(tmp_path):
    output_dir = tmp_path / "reports" / "model"
    artifact = output_dir / "metrics.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({"version": 1}), encoding="utf-8")
    identity = {"dataset_hash": "dataset-a", "config_hash": "config-a"}
    run_id = deterministic_run_id("exposure_ml", identity)
    first = preserve_immutable_run(
        output_dir=output_dir,
        run_id=run_id,
        kind="exposure_ml",
        identity=identity,
        artifact_paths=(artifact,),
    )
    artifact.write_text(json.dumps({"version": 2}), encoding="utf-8")

    second = preserve_immutable_run(
        output_dir=output_dir,
        run_id=run_id,
        kind="exposure_ml",
        identity=identity,
        artifact_paths=(artifact,),
    )

    assert second.run_dir == first.run_dir
    preserved = json.loads((first.run_dir / "metrics.json").read_text())
    assert preserved == {"version": 1}


def test_partial_immutable_run_is_not_complete_and_cannot_be_championed(tmp_path):
    output_dir = tmp_path / "reports" / "model"
    identity = {"dataset_hash": "dataset-a"}
    run_id = deterministic_run_id("stock_selector_benchmark", identity)
    run_dir = immutable_run_dir(output_dir, run_id)
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"run_id": run_id, "run_status": "writing"}),
        encoding="utf-8",
    )

    assert is_complete_run_dir(run_dir) is False
    with pytest.raises(RuntimeError, match="Cannot champion incomplete"):
        update_champion_pointer(
            output_dir=output_dir,
            run_id=run_id,
            kind="stock_selector_benchmark",
            model_name="ridge",
            identity=identity,
        )
    assert not champion_pointer_path(output_dir).exists()


def test_champion_pointer_updates_only_through_explicit_call(tmp_path):
    output_dir = tmp_path / "reports" / "selector"
    artifact = output_dir / "stock_level_model_ranking_benchmark.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({"leaderboard": []}), encoding="utf-8")
    identity = {"completed_models": ["ridge"], "dataset_hash": "dataset-a"}
    run_id = deterministic_run_id("stock_selector_benchmark", identity)

    preserve_immutable_run(
        output_dir=output_dir,
        run_id=run_id,
        kind="stock_selector_benchmark",
        identity=identity,
        artifact_paths=(artifact,),
    )

    assert not champion_pointer_path(output_dir).exists()
    update_champion_pointer(
        output_dir=output_dir,
        run_id=run_id,
        kind="stock_selector_benchmark",
        model_name="ridge",
        identity=identity,
        reason="explicit test promotion",
    )
    champion = json.loads(champion_pointer_path(output_dir).read_text())
    assert champion["run_id"] == run_id
    assert champion["model_name"] == "ridge"
