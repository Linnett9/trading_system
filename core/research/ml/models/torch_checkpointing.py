from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from core.research.ml.data.datasets import MODEL_INPUT_CONTRACT_VERSION


CHECKPOINT_SCHEMA_VERSION = 1
LOGGER = logging.getLogger(__name__)
TORCH_EXPOSURE_MODEL_TYPES = {
    "dlinear",
    "patchtst",
    "transformer",
    "itransformer",
    "momentum_transformer",
    "multitask_transformer",
    "market_context_encoder",
    "news_analysis_transformer",
    "temporal_fusion_transformer",
}


@dataclass(frozen=True)
class TorchCheckpointSettings:
    enabled: bool = False
    resume_enabled: bool = True
    checkpoint_dir: Path | None = None
    frequency_epochs: int = 1
    retain_best_validation: bool = True
    validation_fraction: float = 0.0
    temporal_policy: dict[str, Any] | None = None


@dataclass(frozen=True)
class ResumeResult:
    resumed: bool
    start_epoch: int = 0
    reason: str = ""
    path: Path | None = None


def attach_torch_checkpoint_settings(
    model: Any,
    *,
    settings: TorchCheckpointSettings,
    resolved_config_hash: str,
    target_label: str,
    run_identity: str,
    news_contract: dict[str, Any] | None = None,
) -> None:
    if getattr(model, "model_type", "") not in TORCH_EXPOSURE_MODEL_TYPES:
        return
    model._torch_checkpoint_settings = settings
    model._torch_checkpoint_resolved_config_hash = resolved_config_hash
    model._torch_checkpoint_target_label = target_label
    model._torch_checkpoint_run_identity = run_identity
    model._torch_checkpoint_news_contract = news_contract or {}


def checkpoint_settings_from_config(
    config: dict[str, Any],
    *,
    model_type: str,
    output_dir: str,
) -> TorchCheckpointSettings:
    ml_config = config.get("ml", {}) or {}
    nested = ml_config.get("torch_checkpointing", {})
    nested = nested if isinstance(nested, dict) else {}
    temporal_validation = ml_config.get("temporal_validation", {})
    temporal_validation = temporal_validation if isinstance(temporal_validation, dict) else {}
    validation_policy = str(temporal_validation.get("validation_policy", "none"))
    validation_fraction = (
        float(temporal_validation.get("validation_fraction", 0.0))
        if validation_policy not in {"", "none", "disabled"}
        else 0.0
    )
    enabled = bool(
        nested.get("enabled", ml_config.get("torch_checkpointing_enabled", False))
    )
    resume_enabled = bool(
        nested.get("resume_enabled", ml_config.get("torch_resume_enabled", True))
    )
    raw_dir = nested.get("checkpoint_dir", ml_config.get("torch_checkpoint_dir"))
    checkpoint_dir = (
        Path(raw_dir)
        if raw_dir
        else Path(output_dir) / "checkpoints" / str(model_type)
    )
    return TorchCheckpointSettings(
        enabled=enabled,
        resume_enabled=resume_enabled,
        checkpoint_dir=checkpoint_dir,
        frequency_epochs=max(
            1,
            int(nested.get("frequency_epochs", ml_config.get("torch_checkpoint_frequency_epochs", 1))),
        ),
        retain_best_validation=bool(
            nested.get(
                "retain_best_validation",
                ml_config.get("torch_retain_best_validation_checkpoint", True),
            )
        ),
        validation_fraction=max(0.0, min(0.5, validation_fraction)),
        temporal_policy={
            "version": 1,
            "validation_policy": validation_policy,
            "validation_fraction": max(0.0, min(0.5, validation_fraction)),
            "minimum_validation_rows": int(temporal_validation.get("minimum_validation_rows", 0)),
            "purge_policy": str(temporal_validation.get("purge_policy", "none")),
            "embargo_rows": int(temporal_validation.get("embargo_rows", 0)),
        },
    )


def model_architecture_params(model: Any) -> dict[str, Any]:
    keys = (
        "sequence_length",
        "patch_length",
        "patch_stride",
        "d_model",
        "nhead",
        "num_layers",
        "dim_feedforward",
        "hidden_size",
        "attention_heads",
        "dropout",
        "pos_weight",
        "size_multiplier_floor",
        "size_multiplier_ceiling",
        "risk_multiplier_floor",
        "risk_multiplier_ceiling",
        "known_future_features",
        "regression_targets",
        "classification_weight",
        "regression_loss",
        "huber_delta",
        "regression_weights",
    )
    return {key: getattr(model, key) for key in keys if hasattr(model, key)}


def ordered_feature_columns(model: Any) -> list[str]:
    if hasattr(model, "feature_names"):
        return list(getattr(model, "feature_names"))
    return [
        *list(getattr(model, "observed_feature_names", [])),
        *list(getattr(model, "known_feature_names", [])),
    ]


def feature_scaler_state(model: Any) -> dict[str, Any]:
    return {
        "feature_means": getattr(model, "feature_means", None),
        "feature_stds": getattr(model, "feature_stds", None),
        "means": getattr(model, "means", None),
        "stds": getattr(model, "stds", None),
        "target_means": getattr(model, "target_means", None),
        "target_stds": getattr(model, "target_stds", None),
    }


def checkpoint_identity(
    model: Any,
    *,
    x_train: list[dict[str, float]],
    y_train: list[int],
) -> dict[str, Any]:
    feature_columns = ordered_feature_columns(model)
    dataset_hash = _stable_hash({"features": x_train, "labels": y_train})
    model_input = {
        "feature_columns": feature_columns,
        "features": [
            [float(row.get(column, 0.0) or 0.0) for column in feature_columns]
            for row in x_train
        ],
        "labels": [int(value) for value in y_train],
    }
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": getattr(model, "model_type", type(model).__name__),
        "architecture_params": model_architecture_params(model),
        "target": getattr(model, "_torch_checkpoint_target_label", "unknown"),
        "dataset_hash": dataset_hash,
        "model_input_hash": _stable_hash(model_input),
        "model_input_contract_version": MODEL_INPUT_CONTRACT_VERSION,
        "feature_columns": feature_columns,
        "feature_scaler_contract": "standardizer_v1",
        "feature_scaler_state": feature_scaler_state(model),
        "sequence_length": getattr(model, "sequence_length", None),
        "resolved_config_hash": getattr(
            model,
            "_torch_checkpoint_resolved_config_hash",
            "",
        ),
        "random_seed": getattr(model, "random_seed", None),
        "news_contract": getattr(model, "_torch_checkpoint_news_contract", {}),
        "run_identity": getattr(model, "_torch_checkpoint_run_identity", ""),
        "temporal_policy": getattr(
            getattr(model, "_torch_checkpoint_settings", TorchCheckpointSettings()),
            "temporal_policy",
            None,
        ),
    }


class TorchCheckpointSession:
    def __init__(
        self,
        *,
        torch: Any,
        model_owner: Any,
        network: Any,
        optimizer: Any,
        identity: dict[str, Any],
        total_epochs: int,
        scheduler: Any = None,
        metric_name: str = "validation_loss",
        metric_direction: str = "minimize",
    ) -> None:
        settings = getattr(
            model_owner,
            "_torch_checkpoint_settings",
            TorchCheckpointSettings(),
        )
        self.enabled = bool(settings.enabled and settings.checkpoint_dir)
        self.resume_enabled = bool(settings.resume_enabled)
        self.frequency_epochs = max(1, int(settings.frequency_epochs))
        self.retain_best_validation = bool(settings.retain_best_validation)
        self.torch = torch
        self.model_owner = model_owner
        self.network = network
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.identity = identity
        self.total_epochs = int(total_epochs)
        self.metric_name = metric_name
        self.metric_direction = metric_direction
        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self.early_stopping_state = {"counter": 0, "patience": None}
        self.resume_result = ResumeResult(False, reason="checkpointing_disabled")
        self.dir = Path(settings.checkpoint_dir) if settings.checkpoint_dir else None
        self.last_path = self.dir / "last_checkpoint.pt" if self.dir else None
        self.best_path = self.dir / "best_validation_checkpoint.pt" if self.dir else None
        if self.enabled:
            self.model_owner._torch_checkpoint_metadata = {
                "resumed": False,
                "resume_reason": "checkpoint_missing",
                "resume_start_epoch": 0,
                "checkpoint_path": None,
                "final_weights_source": "last_epoch",
            }

    def restore_if_compatible(self) -> ResumeResult:
        if not self.enabled:
            return self.resume_result
        if not self.resume_enabled:
            self.resume_result = ResumeResult(False, reason="resume_disabled")
            self._record_resume_metadata()
            self._log("checkpoint_load_rejected", reason=self.resume_result.reason)
            return self.resume_result
        assert self.last_path is not None
        if not self.last_path.exists():
            self.resume_result = ResumeResult(False, reason="checkpoint_missing")
            self._record_resume_metadata()
            self._log("clean_restart", reason=self.resume_result.reason)
            return self.resume_result
        try:
            payload = _torch_load(self.torch, self.last_path)
        except Exception as exc:
            self.resume_result = ResumeResult(
                False,
                reason=f"checkpoint_unreadable:{type(exc).__name__}",
                path=self.last_path,
            )
            self._record_resume_metadata()
            self._log("corrupt_checkpoint_rejected", reason=self.resume_result.reason)
            return self.resume_result
        reason = self._incompatibility_reason(payload)
        if reason:
            self.resume_result = ResumeResult(False, reason=reason, path=self.last_path)
            self._record_resume_metadata()
            self._log("checkpoint_load_rejected", reason=reason, path=str(self.last_path))
            return self.resume_result
        self.network.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if self.scheduler is not None and payload.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        self.best_value = payload.get("best_validation_metric")
        self.best_epoch = payload.get("best_validation_epoch")
        self.early_stopping_state = dict(
            payload.get("early_stopping_state", self.early_stopping_state)
        )
        _restore_rng_state(self.torch, payload.get("rng_state", {}))
        start_epoch = int(payload.get("next_epoch", 0))
        self.resume_result = ResumeResult(True, start_epoch, path=self.last_path)
        self._record_resume_metadata()
        self._log(
            "checkpoint_load_accepted",
            path=str(self.last_path),
            resume_start_epoch=start_epoch,
        )
        return self.resume_result

    def save_epoch(self, *, completed_epoch: int, validation_metric: float) -> None:
        if not self.enabled:
            return
        improved = self._is_improved(validation_metric)
        if improved:
            self.best_value = float(validation_metric)
            self.best_epoch = int(completed_epoch)
            self.early_stopping_state = {**self.early_stopping_state, "counter": 0}
        else:
            counter = int(self.early_stopping_state.get("counter") or 0) + 1
            self.early_stopping_state = {**self.early_stopping_state, "counter": counter}
        if (completed_epoch + 1) % self.frequency_epochs == 0:
            self._write_checkpoint(self.last_path, completed_epoch)
            self._log("last_checkpoint_updated", completed_epoch=completed_epoch)
        if improved and self.retain_best_validation:
            self._write_checkpoint(self.best_path, completed_epoch)
            self._log(
                "best_checkpoint_updated",
                completed_epoch=completed_epoch,
                validation_metric=validation_metric,
            )
        self.model_owner._torch_checkpoint_metadata = {
            **getattr(self.model_owner, "_torch_checkpoint_metadata", {}),
            "best_validation_metric": self.best_value,
            "best_validation_epoch": self.best_epoch,
            "validation_metric_name": self.metric_name,
            "validation_metric_direction": self.metric_direction,
        }

    def restore_best_weights(self) -> str:
        if not self.enabled:
            return "last_epoch"
        if not self.best_path or not self.best_path.exists():
            return self._record_final_weights_source("last_epoch")
        try:
            payload = _torch_load(self.torch, self.best_path)
        except Exception:
            return self._record_final_weights_source("last_epoch")
        if self._incompatibility_reason(payload):
            return self._record_final_weights_source("last_epoch")
        self.network.load_state_dict(payload["model_state_dict"])
        return self._record_final_weights_source("best_validation_checkpoint")

    def _record_resume_metadata(self) -> None:
        self.model_owner._torch_checkpoint_metadata = {
            **getattr(self.model_owner, "_torch_checkpoint_metadata", {}),
            "resumed": self.resume_result.resumed,
            "resume_reason": self.resume_result.reason,
            "resume_start_epoch": self.resume_result.start_epoch,
            "checkpoint_path": str(self.resume_result.path) if self.resume_result.path else None,
        }

    def _record_final_weights_source(self, source: str) -> str:
        self.model_owner._torch_checkpoint_metadata = {
            **getattr(self.model_owner, "_torch_checkpoint_metadata", {}),
            "final_weights_source": source,
        }
        return source

    def _write_checkpoint(self, path: Path | None, completed_epoch: int) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "identity": self.identity,
            "model_type": self.identity.get("model_type"),
            "architecture_params": self.identity.get("architecture_params"),
            "model_state_dict": self.network.state_dict(),
            "optimizer_type": type(self.optimizer).__name__,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_type": type(self.scheduler).__name__ if self.scheduler else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "completed_epoch": int(completed_epoch),
            "next_epoch": int(completed_epoch) + 1,
            "total_epochs": self.total_epochs,
            "best_validation_metric": self.best_value,
            "best_validation_epoch": self.best_epoch,
            "validation_metric_name": self.metric_name,
            "validation_metric_direction": self.metric_direction,
            "early_stopping_state": self.early_stopping_state,
            "feature_columns": self.identity.get("feature_columns"),
            "feature_scaler_contract": self.identity.get("feature_scaler_contract"),
            "feature_scaler_state": self.identity.get("feature_scaler_state"),
            "target": self.identity.get("target"),
            "sequence_length": self.identity.get("sequence_length"),
            "dataset_hash": self.identity.get("dataset_hash"),
            "model_input_hash": self.identity.get("model_input_hash"),
            "model_input_contract_version": self.identity.get("model_input_contract_version"),
            "resolved_config_hash": self.identity.get("resolved_config_hash"),
            "random_seed": self.identity.get("random_seed"),
            "news_contract": self.identity.get("news_contract"),
            "rng_state": _rng_state(self.torch),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_identity": self.identity.get("run_identity"),
        }
        self._log("checkpoint_save_started", path=str(path), completed_epoch=completed_epoch)
        _atomic_torch_save(self.torch, payload, path)
        self._log("checkpoint_save_completed", path=str(path), completed_epoch=completed_epoch)

    def _log(self, event: str, **fields: Any) -> None:
        LOGGER.info(
            "torch_checkpoint_event",
            extra={
                "event": event,
                "model_type": self.identity.get("model_type"),
                "run_identity": self.identity.get("run_identity"),
                **fields,
            },
        )

    def _is_improved(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.metric_direction == "maximize":
            return float(value) > float(self.best_value)
        return float(value) < float(self.best_value)

    def _incompatibility_reason(self, payload: Any) -> str:
        if not isinstance(payload, dict):
            return "checkpoint_payload_not_mapping"
        if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            return "checkpoint_schema_version_mismatch"
        saved_identity = payload.get("identity")
        if not isinstance(saved_identity, dict):
            return "missing_identity"
        checks = (
            "model_type",
            "architecture_params",
            "target",
            "dataset_hash",
            "model_input_contract_version",
            "feature_columns",
            "model_input_hash",
            "feature_scaler_contract",
            "feature_scaler_state",
            "sequence_length",
            "resolved_config_hash",
            "news_contract",
            "run_identity",
            "temporal_policy",
        )
        for key in checks:
            if saved_identity.get(key) != self.identity.get(key):
                if key == "feature_columns" and set(saved_identity.get(key, [])) == set(
                    self.identity.get(key, [])
                ):
                    return "feature_order_mismatch"
                return f"{key}_mismatch"
        if payload.get("optimizer_type") != type(self.optimizer).__name__:
            return "optimizer_type_mismatch"
        expected_scheduler = type(self.scheduler).__name__ if self.scheduler else None
        if payload.get("scheduler_type") != expected_scheduler:
            return "scheduler_type_mismatch"
        if "model_state_dict" not in payload or "optimizer_state_dict" not in payload:
            return "missing_training_state"
        return ""


def validation_split_tensors(
    tensors: tuple[Any, ...],
    labels: Any,
    *,
    fraction: float,
) -> tuple[tuple[Any, ...], Any, tuple[Any, ...], Any]:
    total = int(labels.shape[0])
    if total < 3 or fraction <= 0.0:
        return tensors, labels, tensors, labels
    validation_count = max(1, int(total * fraction))
    if validation_count >= total:
        validation_count = 1
    train_count = total - validation_count
    train_tensors = tuple(tensor[:train_count] for tensor in tensors)
    validation_tensors = tuple(tensor[train_count:] for tensor in tensors)
    return train_tensors, labels[:train_count], validation_tensors, labels[train_count:]


def checkpoint_validation_fraction(model: Any) -> float:
    settings = getattr(model, "_torch_checkpoint_settings", TorchCheckpointSettings())
    if not settings.enabled:
        return 0.0
    return settings.validation_fraction


def checkpoint_enabled(model: Any) -> bool:
    settings = getattr(model, "_torch_checkpoint_settings", TorchCheckpointSettings())
    return bool(settings.enabled)


def binary_validation_loss(
    torch: Any,
    model: Any,
    criterion: Any,
    validation_tensors: tuple[Any, ...],
    validation_labels: Any,
    device: str,
    forward: Callable[[Any, tuple[Any, ...]], Any] | None = None,
) -> float:
    model.eval()
    with torch.no_grad():
        logits = (
            forward(model, validation_tensors)
            if forward is not None
            else model(validation_tensors[0].to(device))
        )
        loss = criterion(logits.squeeze(-1), validation_labels.to(device))
    model.train()
    return float(loss.detach().cpu().item())


def _rng_state(torch: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(torch: Any, state: dict[str, Any]) -> None:
    if not isinstance(state, dict):
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    try:
        import numpy as np

        if state.get("numpy") is not None:
            np.random.set_state(state["numpy"])
    except ImportError:
        pass
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])
    if (
        state.get("torch_cuda") is not None
        and hasattr(torch, "cuda")
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(torch: Any, payload: dict[str, Any], path: Path) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _torch_load(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
