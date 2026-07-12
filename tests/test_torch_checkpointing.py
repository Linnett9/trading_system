from __future__ import annotations

from pathlib import Path

import pytest

from core.research.ml.models.dlinear_model import DLinearSequenceMLModel
from core.research.ml.models.torch_checkpointing import (
    TorchCheckpointSettings,
    attach_torch_checkpoint_settings,
    checkpoint_settings_from_config,
)


def _rows(count: int = 48) -> list[dict[str, float]]:
    return [
        {
            "return_1": i / 100.0,
            "volatility_10": (i % 5) / 10.0,
            "breadth": 1.0 if i % 2 else 0.0,
        }
        for i in range(count)
    ]


def _labels(count: int = 48) -> list[int]:
    return [1 if i % 7 in {0, 1, 2} else 0 for i in range(count)]


def _checkpointed_model(tmp_path: Path, *, epochs: int = 2) -> DLinearSequenceMLModel:
    model = DLinearSequenceMLModel(
        sequence_length=6,
        epochs=epochs,
        batch_size=8,
        random_seed=11,
        learning_rate=0.01,
    )
    attach_torch_checkpoint_settings(
        model,
        settings=TorchCheckpointSettings(
            enabled=True,
            resume_enabled=True,
            checkpoint_dir=tmp_path,
            frequency_epochs=1,
            validation_fraction=0.25,
        ),
        resolved_config_hash="config-a",
        target_label="should_reduce_exposure",
        run_identity="run-a",
    )
    return model


def test_torch_training_checkpoint_resumes_next_epoch_and_restores_state(tmp_path: Path):
    torch = pytest.importorskip("torch")
    first = _checkpointed_model(tmp_path, epochs=1)
    first.fit(_rows(), _labels())

    last_payload = torch.load(tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    assert last_payload["completed_epoch"] == 0
    assert last_payload["next_epoch"] == 1
    assert last_payload["optimizer_state_dict"]["state"]
    assert (tmp_path / "best_validation_checkpoint.pt").exists()

    second = _checkpointed_model(tmp_path, epochs=3)
    second.fit(_rows(), _labels())

    metadata = second._torch_checkpoint_metadata
    assert metadata["resumed"] is True
    assert metadata["resume_start_epoch"] == 1
    assert metadata["best_validation_metric"] is not None
    assert metadata["best_validation_epoch"] in {0, 1, 2}
    assert metadata["final_weights_source"] in {"best_validation_checkpoint", "last_epoch"}

    final_payload = torch.load(tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    assert final_payload["completed_epoch"] == 2
    assert final_payload["early_stopping_state"]["counter"] >= 0
    assert final_payload["rng_state"]["torch_cpu"] is not None


def test_incompatible_dataset_hash_rejects_resume_without_loading(tmp_path: Path):
    pytest.importorskip("torch")
    first = _checkpointed_model(tmp_path, epochs=1)
    first.fit(_rows(), _labels())

    second = _checkpointed_model(tmp_path, epochs=2)
    second.fit(_rows(49), _labels(49))

    metadata = second._torch_checkpoint_metadata
    assert metadata["resumed"] is False
    assert metadata["resume_reason"] == "dataset_hash_mismatch"


def test_changed_feature_order_rejects_resume(tmp_path: Path):
    pytest.importorskip("torch")
    first = _checkpointed_model(tmp_path, epochs=1)
    first.fit(_rows(), _labels())

    class ReverseFeatureDLinear(DLinearSequenceMLModel):
        def fit(self, x_train, y_train):
            original = sorted
            try:
                import builtins

                builtins.sorted = lambda values: list(reversed(original(values)))
                return super().fit(x_train, y_train)
            finally:
                builtins.sorted = original

    second = ReverseFeatureDLinear(
        sequence_length=6,
        epochs=2,
        batch_size=8,
        random_seed=11,
        learning_rate=0.01,
    )
    attach_torch_checkpoint_settings(
        second,
        settings=TorchCheckpointSettings(
            enabled=True,
            resume_enabled=True,
            checkpoint_dir=tmp_path,
            frequency_epochs=1,
            validation_fraction=0.25,
        ),
        resolved_config_hash="config-a",
        target_label="should_reduce_exposure",
        run_identity="run-a",
    )

    second.fit(_rows(), _labels())

    assert second._torch_checkpoint_metadata["resume_reason"] == "feature_order_mismatch"


def test_corrupt_and_partial_checkpoint_files_are_rejected(tmp_path: Path):
    pytest.importorskip("torch")
    (tmp_path / "last_checkpoint.pt").write_bytes(b"not a torch checkpoint")
    (tmp_path / ".last_checkpoint.pt.partial.tmp").write_bytes(b"partial")

    model = _checkpointed_model(tmp_path, epochs=1)
    model.fit(_rows(), _labels())

    metadata = model._torch_checkpoint_metadata
    assert metadata["resumed"] is False
    assert metadata["resume_reason"].startswith("checkpoint_unreadable")
    assert (tmp_path / ".last_checkpoint.pt.partial.tmp").exists()


def test_resume_disabled_and_checkpointing_disabled_keep_clean_training(tmp_path: Path):
    pytest.importorskip("torch")
    first = _checkpointed_model(tmp_path, epochs=1)
    first.fit(_rows(), _labels())

    resume_disabled = _checkpointed_model(tmp_path, epochs=2)
    attach_torch_checkpoint_settings(
        resume_disabled,
        settings=TorchCheckpointSettings(
            enabled=True,
            resume_enabled=False,
            checkpoint_dir=tmp_path,
        ),
        resolved_config_hash="config-a",
        target_label="should_reduce_exposure",
        run_identity="run-a",
    )
    resume_disabled.fit(_rows(), _labels())
    assert resume_disabled._torch_checkpoint_metadata["resume_reason"] == "resume_disabled"

    disabled = DLinearSequenceMLModel(sequence_length=6, epochs=1, batch_size=8)
    disabled.fit(_rows(), _labels())
    assert not hasattr(disabled, "_torch_checkpoint_metadata")


def test_checkpointing_enabled_does_not_create_validation_holdback_without_policy(tmp_path: Path):
    settings = checkpoint_settings_from_config(
        {
            "ml": {
                "output_dir": str(tmp_path),
                "torch_checkpointing_enabled": True,
            }
        },
        model_type="dlinear",
        output_dir=str(tmp_path),
    )

    assert settings.enabled is True
    assert settings.validation_fraction == 0.0
    assert settings.temporal_policy["validation_policy"] == "none"

    explicit = checkpoint_settings_from_config(
        {
            "ml": {
                "output_dir": str(tmp_path),
                "torch_checkpointing_enabled": True,
                "temporal_validation": {
                    "validation_policy": "chronological_tail",
                    "validation_fraction": 0.25,
                    "purge_policy": "label_end_before_validation_start",
                },
            }
        },
        model_type="dlinear",
        output_dir=str(tmp_path),
    )

    assert explicit.validation_fraction == 0.25
    assert explicit.temporal_policy["validation_policy"] == "chronological_tail"
