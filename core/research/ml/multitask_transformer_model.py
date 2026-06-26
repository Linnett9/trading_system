from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from core.research.ml.models import IMLModel
from core.research.ml.sequence_dataset import (
    build_sequence_indices,
    sequence_group_ids_from_metadata,
)
from core.research.ml.transformer_model import _torch_dependencies


DEFAULT_REGRESSION_TARGETS = [
    "forward_return_5d",
    "forward_return_10d",
    "future_volatility",
    "future_drawdown",
    "max_adverse_excursion",
    "max_favourable_excursion",
]

LEAKAGE_FEATURE_PREFIXES = (
    "actual_",
    "forward_",
    "future_",
    "max_adverse_",
    "max_favourable_",
)
LEAKAGE_FEATURE_NAMES = {
    "research_label",
    "should_reduce_exposure",
    *DEFAULT_REGRESSION_TARGETS,
}


@dataclass(frozen=True)
class MultiTaskTransformerTrainingSummary:
    trained: bool
    sequence_count: int
    feature_count: int
    positive_rate: float
    regression_targets: list[str]
    missing_target_counts: dict[str, int]


class MultiTaskTransformerSequenceMLModel(IMLModel):
    """Research-only transformer with a classification head and optional regressions.

    The public `fit`, `predict_proba`, and `predict` methods intentionally keep
    the existing single-task model contract. Multi-task training is opt-in via
    `fit_multitask`, so MLExperimentRunner can run this model before artifact
    and meta-ensemble plumbing learn about the auxiliary heads.
    """

    model_type = "multitask_transformer"

    def __init__(
        self,
        sequence_length: int = 63,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.10,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        random_seed: int = 42,
        device: str = "cpu",
        regression_targets: list[str] | None = None,
        classification_weight: float = 1.0,
        regression_loss: str = "huber",
        huber_delta: float = 1.0,
        regression_weights: dict[str, float] | None = None,
    ):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if d_model < 4:
            raise ValueError("d_model must be at least 4")
        if nhead < 1:
            raise ValueError("nhead must be at least one")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if classification_weight <= 0:
            raise ValueError("classification_weight must be positive")
        if regression_loss not in {"huber", "mse"}:
            raise ValueError("regression_loss must be one of: huber, mse")
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")

        self.sequence_length = int(sequence_length)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.random_seed = int(random_seed)
        self.device = str(device)
        self.regression_targets = list(regression_targets or DEFAULT_REGRESSION_TARGETS)
        self.classification_weight = float(classification_weight)
        self.regression_loss = str(regression_loss)
        self.huber_delta = float(huber_delta)
        self.regression_weights = {
            target: float((regression_weights or {}).get(target, 0.2))
            for target in self.regression_targets
        }

        self.feature_names: list[str] = []
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.target_means: dict[str, float] = {name: 0.0 for name in self.regression_targets}
        self.target_stds: dict[str, float] = {name: 1.0 for name in self.regression_targets}
        self.prior_probability = 0.5
        self.model: Any = None
        self._sequence_group_ids: list[str] = []
        self.training_summary = MultiTaskTransformerTrainingSummary(
            False,
            0,
            0,
            0.5,
            list(self.regression_targets),
            {name: 0 for name in self.regression_targets},
        )

    def set_sequence_context(
        self,
        metadata: list[dict[str, str]] | None = None,
        feature_dates: list[str] | None = None,
    ) -> None:
        del feature_dates
        self._sequence_group_ids = sequence_group_ids_from_metadata(
            metadata,
            len(metadata or []),
        )

    def fit(self, x_train: list[dict[str, float]], y_train: list[int]) -> None:
        self.fit_multitask(x_train, y_train, {})

    def fit_multitask(
        self,
        x_train: list[dict[str, float]],
        y_train: list[int],
        regression_targets: dict[str, list[float | None]],
    ) -> None:
        if len(x_train) != len(y_train):
            raise ValueError("Features and labels must have the same length")
        for target_name, values in regression_targets.items():
            if target_name not in self.regression_targets:
                raise ValueError(f"Unsupported regression target '{target_name}'")
            if len(values) != len(x_train):
                raise ValueError(
                    f"Regression target '{target_name}' must match feature length"
                )
        if not x_train:
            self.training_summary = MultiTaskTransformerTrainingSummary(
                False,
                0,
                0,
                0.5,
                list(self.regression_targets),
                {name: 0 for name in self.regression_targets},
            )
            return

        self.feature_names = _safe_feature_names(x_train[0])
        self.prior_probability = sum(int(value) for value in y_train) / len(y_train)
        self._fit_scaler(x_train)

        sequences, labels, regression_values, regression_mask = self._build_training_tensors(
            x_train,
            y_train,
            regression_targets,
        )
        self._fit_target_scalers(regression_values, regression_mask)
        normalized_regression_values = self._normalize_regression_targets(
            regression_values,
            regression_mask,
        )
        missing_counts = self._missing_target_counts(regression_mask)

        if not sequences or len(set(labels)) < 2:
            self.training_summary = MultiTaskTransformerTrainingSummary(
                False,
                len(sequences),
                len(self.feature_names),
                self.prior_probability,
                list(self.regression_targets),
                missing_counts,
            )
            return

        torch, nn, DataLoader, TensorDataset = _torch_dependencies()
        torch.manual_seed(self.random_seed)

        device = torch.device(self.device)
        x_tensor = torch.tensor(sequences, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
        regression_tensor = torch.tensor(
            normalized_regression_values,
            dtype=torch.float32,
            device=device,
        )
        mask_tensor = torch.tensor(regression_mask, dtype=torch.float32, device=device)
        dataset = TensorDataset(x_tensor, y_tensor, regression_tensor, mask_tensor)
        loader = DataLoader(
            dataset,
            batch_size=max(1, self.batch_size),
            shuffle=True,
            generator=torch.Generator(device="cpu").manual_seed(self.random_seed),
        )

        network_cls = _make_multitask_transformer_module()
        model = network_cls(
            feature_count=len(self.feature_names),
            sequence_length=self.sequence_length,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            regression_head_count=len(self.regression_targets),
        ).to(device)

        positive_count = sum(labels)
        negative_count = len(labels) - positive_count
        pos_weight = torch.tensor(
            [negative_count / max(positive_count, 1)],
            dtype=torch.float32,
            device=device,
        )
        classification_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        regression_loss_fn = (
            nn.SmoothL1Loss(reduction="none", beta=self.huber_delta)
            if self.regression_loss == "huber"
            else nn.MSELoss(reduction="none")
        )
        regression_weights = torch.tensor(
            [self.regression_weights[target] for target in self.regression_targets],
            dtype=torch.float32,
            device=device,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        model.train()
        for _ in range(max(1, self.epochs)):
            for batch_x, batch_y, batch_regression, batch_mask in loader:
                optimizer.zero_grad(set_to_none=True)
                classification_logits, regression_outputs = model(batch_x)
                classification_loss = classification_loss_fn(
                    classification_logits,
                    batch_y,
                )
                regression_loss = self._masked_regression_loss(
                    regression_loss_fn(
                        regression_outputs,
                        batch_regression,
                    ),
                    batch_mask,
                    regression_weights,
                )
                loss = self.classification_weight * classification_loss + regression_loss
                loss.backward()
                optimizer.step()

        self.model = model.cpu()
        self.training_summary = MultiTaskTransformerTrainingSummary(
            True,
            len(sequences),
            len(self.feature_names),
            self.prior_probability,
            list(self.regression_targets),
            missing_counts,
        )

    def predict(self, x: list[dict[str, float]]) -> list[int]:
        return [int(probability >= 0.5) for probability in self.predict_proba(x)]

    def predict_proba(self, x: list[dict[str, float]]) -> list[float]:
        return [
            row["probability_should_reduce_exposure"]
            for row in self.predict_multitask(x)
        ]

    def predict_multitask(self, x: list[dict[str, float]]) -> list[dict[str, float]]:
        if not x:
            return []

        predictions = [
            self._prior_prediction_row()
            for _ in x
        ]
        if self.model is None or not self.training_summary.trained:
            return predictions

        sequences, end_indices = self._build_prediction_sequences(x)
        if not sequences:
            return predictions

        torch, _, _, _ = _torch_dependencies()
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(sequences, dtype=torch.float32)
            classification_logits, regression_outputs = self.model(x_tensor)
            probabilities = torch.sigmoid(classification_logits).cpu().tolist()
            regression_rows = regression_outputs.cpu().tolist()

        for index, probability, regression_row in zip(
            end_indices,
            probabilities,
            regression_rows,
        ):
            row = {
                "probability_should_reduce_exposure": float(
                    max(0.0, min(1.0, probability))
                )
            }
            for target, normalized_value in zip(self.regression_targets, regression_row):
                value = (
                    float(normalized_value) * self.target_stds.get(target, 1.0)
                    + self.target_means.get(target, 0.0)
                )
                row[f"predicted_{target}"] = value if math.isfinite(value) else 0.0
            predictions[index] = row
        return predictions

    def feature_importances(self) -> dict[str, float]:
        return {}

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

    def _fit_scaler(self, rows: list[dict[str, float]]) -> None:
        self.means = {}
        self.stds = {}
        for name in self.feature_names:
            values = [float(row.get(name, 0.0) or 0.0) for row in rows]
            mean_value = sum(values) / len(values)
            variance = sum((value - mean_value) ** 2 for value in values) / len(values)
            std_value = variance ** 0.5
            self.means[name] = mean_value
            self.stds[name] = std_value if std_value > 1e-12 else 1.0

    def _fit_target_scalers(
        self,
        regression_values: list[list[float]],
        regression_mask: list[list[float]],
    ) -> None:
        self.target_means = {}
        self.target_stds = {}
        for target_index, target in enumerate(self.regression_targets):
            values = [
                row[target_index]
                for row, mask in zip(regression_values, regression_mask)
                if mask[target_index] > 0.0
            ]
            if not values:
                self.target_means[target] = 0.0
                self.target_stds[target] = 1.0
                continue
            mean_value = sum(values) / len(values)
            variance = sum((value - mean_value) ** 2 for value in values) / len(values)
            std_value = variance ** 0.5
            self.target_means[target] = mean_value
            self.target_stds[target] = std_value if std_value > 1e-12 else 1.0

    def _row_vector(self, row: dict[str, float]) -> list[float]:
        return [
            (float(row.get(name, 0.0) or 0.0) - self.means.get(name, 0.0))
            / self.stds.get(name, 1.0)
            for name in self.feature_names
        ]

    def _build_training_tensors(
        self,
        rows: list[dict[str, float]],
        labels: list[int],
        regression_targets: dict[str, list[float | None]],
    ) -> tuple[list[list[list[float]]], list[int], list[list[float]], list[list[float]]]:
        matrix = [self._row_vector(row) for row in rows]
        sequences: list[list[list[float]]] = []
        targets: list[int] = []
        regression_values: list[list[float]] = []
        regression_mask: list[list[float]] = []
        for indices in build_sequence_indices(
            self._context_group_ids(len(matrix)),
            self.sequence_length,
        ):
            end_index = indices[-1]
            sequences.append([matrix[index] for index in indices])
            targets.append(int(labels[end_index]))
            values: list[float] = []
            mask: list[float] = []
            for target_name in self.regression_targets:
                raw_values = regression_targets.get(target_name, [])
                raw_value = raw_values[end_index] if raw_values else None
                if raw_value is None or not math.isfinite(float(raw_value)):
                    values.append(0.0)
                    mask.append(0.0)
                else:
                    values.append(float(raw_value))
                    mask.append(1.0)
            regression_values.append(values)
            regression_mask.append(mask)
        return sequences, targets, regression_values, regression_mask

    def _normalize_regression_targets(
        self,
        regression_values: list[list[float]],
        regression_mask: list[list[float]],
    ) -> list[list[float]]:
        normalized: list[list[float]] = []
        for values, mask in zip(regression_values, regression_mask):
            row: list[float] = []
            for target, value, mask_value in zip(self.regression_targets, values, mask):
                if mask_value <= 0.0:
                    row.append(0.0)
                    continue
                row.append(
                    (value - self.target_means.get(target, 0.0))
                    / self.target_stds.get(target, 1.0)
                )
            normalized.append(row)
        return normalized

    def _build_prediction_sequences(
        self,
        rows: list[dict[str, float]],
    ) -> tuple[list[list[list[float]]], list[int]]:
        matrix = [self._row_vector(row) for row in rows]
        sequences: list[list[list[float]]] = []
        end_indices: list[int] = []
        for indices in build_sequence_indices(
            self._context_group_ids(len(matrix)),
            self.sequence_length,
        ):
            sequences.append([matrix[index] for index in indices])
            end_indices.append(indices[-1])
        return sequences, end_indices

    def _context_group_ids(self, sample_count: int) -> list[str]:
        if len(self._sequence_group_ids) == sample_count:
            return list(self._sequence_group_ids)
        return ["global" for _ in range(sample_count)]

    def _prior_prediction_row(self) -> dict[str, float]:
        row = {"probability_should_reduce_exposure": float(self.prior_probability)}
        for target in self.regression_targets:
            row[f"predicted_{target}"] = float(self.target_means.get(target, 0.0))
        return row

    def _missing_target_counts(
        self,
        regression_mask: list[list[float]],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for target_index, target in enumerate(self.regression_targets):
            counts[target] = sum(
                1
                for row in regression_mask
                if target_index >= len(row) or row[target_index] <= 0.0
            )
        return counts

    @staticmethod
    def _masked_regression_loss(loss_values, mask, regression_weights):
        weighted = loss_values * mask * regression_weights
        denominator = (mask * regression_weights).sum().clamp_min(1.0)
        return weighted.sum() / denominator


def _safe_feature_names(row: dict[str, float]) -> list[str]:
    return [
        name
        for name in sorted(row)
        if name not in LEAKAGE_FEATURE_NAMES
        and not any(name.startswith(prefix) for prefix in LEAKAGE_FEATURE_PREFIXES)
    ]


def _make_multitask_transformer_module() -> type:
    torch, nn, _, _ = _torch_dependencies()

    class MultiTaskTransformerModule(nn.Module):
        def __init__(
            self,
            feature_count: int,
            sequence_length: int,
            d_model: int,
            nhead: int,
            num_layers: int,
            dim_feedforward: int,
            dropout: float,
            regression_head_count: int,
        ):
            super().__init__()
            self.input_projection = nn.Linear(feature_count, d_model)
            self.position_embedding = nn.Parameter(
                torch.zeros(1, sequence_length, d_model)
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 1),
            )
            self.regression_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, regression_head_count),
            )

        def forward(self, x):
            encoded = self.input_projection(x) + self.position_embedding[:, : x.shape[1], :]
            encoded = self.encoder(encoded)
            pooled = encoded[:, -1, :]
            return (
                self.classifier(pooled).squeeze(-1),
                self.regression_head(pooled),
            )

    return MultiTaskTransformerModule
