from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.research.ml.models import IMLModel


@dataclass(frozen=True)
class TransformerTrainingSummary:
    trained: bool
    sequence_count: int
    feature_count: int
    positive_rate: float


class TransformerSequenceMLModel(IMLModel):
    """Small PyTorch transformer classifier behind the existing ML interface.

    This is intentionally research-only and shadow-safe. It accepts the current
    tabular interface used by MLExperimentRunner, builds rolling sequences
    internally, and returns one probability per input row. Rows before a full
    sequence exists receive the training prior probability.
    """

    model_type = "transformer"

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
    ):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if d_model < 4:
            raise ValueError("d_model must be at least 4")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

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

        self.feature_names: list[str] = []
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.prior_probability = 0.5
        self.model: Any = None
        self.training_summary = TransformerTrainingSummary(False, 0, 0, 0.5)

    def fit(self, x_train: list[dict[str, float]], y_train: list[int]) -> None:
        if len(x_train) != len(y_train):
            raise ValueError("Features and labels must have the same length")
        if not x_train:
            self.training_summary = TransformerTrainingSummary(False, 0, 0, 0.5)
            return

        self.feature_names = sorted(x_train[0])
        self.prior_probability = sum(int(value) for value in y_train) / len(y_train)
        self._fit_scaler(x_train)

        sequences, labels = self._build_sequences(x_train, y_train)
        if not sequences:
            self.training_summary = TransformerTrainingSummary(
                False,
                0,
                len(self.feature_names),
                self.prior_probability,
            )
            return

        # A single-class training window cannot teach a binary classifier. Keep
        # the calibrated prior so the runner still emits valid probabilities.
        if len(set(labels)) < 2:
            self.training_summary = TransformerTrainingSummary(
                False,
                len(sequences),
                len(self.feature_names),
                self.prior_probability,
            )
            return

        torch, nn, DataLoader, TensorDataset = _torch_dependencies()
        torch.manual_seed(self.random_seed)

        device = torch.device(self.device)
        x_tensor = torch.tensor(sequences, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=max(1, self.batch_size),
            shuffle=True,
            generator=torch.Generator(device="cpu").manual_seed(self.random_seed),
        )

        network_cls = _make_tiny_transformer_classifier()
        model = network_cls(
            feature_count=len(self.feature_names),
            sequence_length=self.sequence_length,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(device)

        positive_count = sum(labels)
        negative_count = len(labels) - positive_count
        pos_weight = torch.tensor(
            [negative_count / max(positive_count, 1)],
            dtype=torch.float32,
            device=device,
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        model.train()
        for _ in range(max(1, self.epochs)):
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()

        self.model = model.cpu()
        self.training_summary = TransformerTrainingSummary(
            True,
            len(sequences),
            len(self.feature_names),
            self.prior_probability,
        )

    def predict(self, x: list[dict[str, float]]) -> list[int]:
        return [int(probability >= 0.5) for probability in self.predict_proba(x)]

    def predict_proba(self, x: list[dict[str, float]]) -> list[float]:
        if not x:
            return []
        if self.model is None or not self.training_summary.trained:
            return [float(self.prior_probability) for _ in x]

        sequences, end_indices = self._build_prediction_sequences(x)
        probabilities = [float(self.prior_probability) for _ in x]
        if not sequences:
            return probabilities

        torch, _, _, _ = _torch_dependencies()
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(sequences, dtype=torch.float32)
            logits = self.model(x_tensor)
            predicted = torch.sigmoid(logits).cpu().tolist()

        for index, probability in zip(end_indices, predicted):
            probabilities[index] = float(max(0.0, min(1.0, probability)))
        return probabilities

    def feature_importances(self) -> dict[str, float]:
        # Transformer attention weights are not reliable feature importances.
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
            "feature_names": self.feature_names,
            "means": self.means,
            "stds": self.stds,
            "prior_probability": self.prior_probability,
            "training_summary": self.training_summary.__dict__,
            "state_dict": self.model.state_dict() if self.model is not None else None,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> "TransformerSequenceMLModel":
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
        )
        model.feature_names = list(payload.get("feature_names", []))
        model.means = {key: float(value) for key, value in payload.get("means", {}).items()}
        model.stds = {key: float(value) for key, value in payload.get("stds", {}).items()}
        model.prior_probability = float(payload.get("prior_probability", 0.5))
        summary = payload.get("training_summary", {})
        model.training_summary = TransformerTrainingSummary(
            trained=bool(summary.get("trained", False)),
            sequence_count=int(summary.get("sequence_count", 0)),
            feature_count=int(summary.get("feature_count", len(model.feature_names))),
            positive_rate=float(summary.get("positive_rate", model.prior_probability)),
        )

        if payload.get("state_dict") is not None and model.feature_names:
            network_cls = _make_tiny_transformer_classifier()
            network = network_cls(
                feature_count=len(model.feature_names),
                sequence_length=model.sequence_length,
                d_model=model.d_model,
                nhead=model.nhead,
                num_layers=model.num_layers,
                dim_feedforward=model.dim_feedforward,
                dropout=model.dropout,
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

    def _row_vector(self, row: dict[str, float]) -> list[float]:
        return [
            (float(row.get(name, 0.0) or 0.0) - self.means.get(name, 0.0))
            / self.stds.get(name, 1.0)
            for name in self.feature_names
        ]

    def _build_sequences(
        self,
        rows: list[dict[str, float]],
        labels: list[int],
    ) -> tuple[list[list[list[float]]], list[int]]:
        matrix = [self._row_vector(row) for row in rows]
        sequences: list[list[list[float]]] = []
        targets: list[int] = []
        for end_index in range(self.sequence_length - 1, len(matrix)):
            start_index = end_index - self.sequence_length + 1
            sequences.append(matrix[start_index : end_index + 1])
            targets.append(int(labels[end_index]))
        return sequences, targets

    def _build_prediction_sequences(
        self,
        rows: list[dict[str, float]],
    ) -> tuple[list[list[list[float]]], list[int]]:
        matrix = [self._row_vector(row) for row in rows]
        sequences: list[list[list[float]]] = []
        end_indices: list[int] = []
        for end_index in range(self.sequence_length - 1, len(matrix)):
            start_index = end_index - self.sequence_length + 1
            sequences.append(matrix[start_index : end_index + 1])
            end_indices.append(end_index)
        return sequences, end_indices


def _make_tiny_transformer_classifier() -> type:
    torch, nn, _, _ = _torch_dependencies()

    class TinyTransformerClassifier(nn.Module):
        def __init__(
            self,
            feature_count: int,
            sequence_length: int,
            d_model: int,
            nhead: int,
            num_layers: int,
            dim_feedforward: int,
            dropout: float,
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

        def forward(self, x):
            encoded = self.input_projection(x) + self.position_embedding[:, : x.shape[1], :]
            encoded = self.encoder(encoded)
            return self.classifier(encoded[:, -1, :]).squeeze(-1)

    return TinyTransformerClassifier


def _torch_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "Transformer ML model requires PyTorch. Install it with: "
            "python -m pip install torch"
        ) from exc
    return torch, nn, DataLoader, TensorDataset

