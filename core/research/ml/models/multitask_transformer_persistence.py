from __future__ import annotations

from pathlib import Path

from core.research.ml.models.multitask_transformer_network import _make_multitask_transformer_module
from core.research.ml.models.multitask_transformer_types import MultiTaskTransformerTrainingSummary
from core.research.ml.models.transformer_model import _torch_dependencies


class MultiTaskTransformerPersistenceMixin:
    def save(self, path: Path) -> None:
        torch, _, _, _ = _torch_dependencies()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": self.model_type,
            "sequence_length": self.sequence_length,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "random_seed": self.random_seed,
            "device": self.device,
            "regression_targets": self.regression_targets,
            "classification_weight": self.classification_weight,
            "regression_loss": self.regression_loss,
            "huber_delta": self.huber_delta,
            "regression_weights": self.regression_weights,
            "feature_names": self.feature_names,
            "means": self.means,
            "stds": self.stds,
            "target_means": self.target_means,
            "target_stds": self.target_stds,
            "prior_probability": self.prior_probability,
            "training_summary": self.training_summary.__dict__,
            "state_dict": self.model.state_dict() if self.model is not None else None,
        }
        torch.save(payload, path)
    @classmethod
    def load(cls, path: Path) -> "MultiTaskTransformerSequenceMLModel":
        torch, _, _, _ = _torch_dependencies()
        payload = torch.load(path, map_location="cpu")
        if payload.get("model_type") != cls.model_type:
            raise ValueError(f"Unsupported model payload: {payload.get('model_type')}")

        model = cls(
            sequence_length=int(payload["sequence_length"]),
            d_model=int(payload["d_model"]),
            nhead=int(payload["nhead"]),
            num_layers=int(payload["num_layers"]),
            dim_feedforward=int(payload["dim_feedforward"]),
            dropout=float(payload["dropout"]),
            epochs=int(payload["epochs"]),
            batch_size=int(payload["batch_size"]),
            learning_rate=float(payload["learning_rate"]),
            weight_decay=float(payload["weight_decay"]),
            random_seed=int(payload["random_seed"]),
            device=str(payload.get("device", "cpu")),
            regression_targets=list(payload.get("regression_targets", [])),
            classification_weight=float(payload.get("classification_weight", 1.0)),
            regression_loss=str(payload.get("regression_loss", "huber")),
            huber_delta=float(payload.get("huber_delta", 1.0)),
            regression_weights={
                key: float(value)
                for key, value in payload.get("regression_weights", {}).items()
            },
        )
        model.feature_names = list(payload.get("feature_names", []))
        model.means = {
            key: float(value)
            for key, value in payload.get("means", {}).items()
        }
        model.stds = {
            key: float(value)
            for key, value in payload.get("stds", {}).items()
        }
        model.target_means = {
            key: float(value)
            for key, value in payload.get("target_means", {}).items()
        }
        model.target_stds = {
            key: float(value)
            for key, value in payload.get("target_stds", {}).items()
        }
        model.prior_probability = float(payload.get("prior_probability", 0.5))
        summary = payload.get("training_summary", {})
        model.training_summary = MultiTaskTransformerTrainingSummary(
            trained=bool(summary.get("trained", False)),
            sequence_count=int(summary.get("sequence_count", 0)),
            feature_count=int(summary.get("feature_count", len(model.feature_names))),
            positive_rate=float(summary.get("positive_rate", model.prior_probability)),
            regression_targets=list(
                summary.get("regression_targets", model.regression_targets)
            ),
            missing_target_counts={
                key: int(value)
                for key, value in summary.get("missing_target_counts", {}).items()
            },
        )

        if payload.get("state_dict") is not None and model.feature_names:
            network_cls = _make_multitask_transformer_module()
            network = network_cls(
                feature_count=len(model.feature_names),
                sequence_length=model.sequence_length,
                d_model=model.d_model,
                nhead=model.nhead,
                num_layers=model.num_layers,
                dim_feedforward=model.dim_feedforward,
                dropout=model.dropout,
                regression_head_count=len(model.regression_targets),
            )
            network.load_state_dict(payload["state_dict"])
            model.model = network.cpu()
        return model
