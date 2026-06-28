from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def _torch_dependencies() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "iTransformer research model requires PyTorch. Install it with: python -m pip install torch"
        ) from exc
    return torch, nn


class ITransformerSequenceMLModel:
    """Research-only iTransformer-style sequence classifier.

    The existing ML runner supplies chronological tabular feature rows. This
    model converts them into rolling windows, treats each feature/asset column
    as a token, embeds that token's recent temporal history, and applies
    cross-token self-attention before producing one probability per row.
    """

    model_type = "itransformer"

    def __init__(
        self,
        sequence_length: int = 126,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.10,
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        random_seed: int = 42,
        device: str = "cpu",
        pos_weight: str | float | None = "auto",
    ):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if d_model < 4:
            raise ValueError("d_model must be at least 4")
        if nhead < 1:
            raise ValueError("nhead must be at least one")
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
        self.pos_weight = pos_weight

        self.feature_names: list[str] = []
        self.feature_means: list[float] = []
        self.feature_stds: list[float] = []
        self.training_prior: float = 0.5
        self.model: Any = None

    def fit(self, x_train: list[dict[str, float]], y_train: list[int]) -> None:
        if len(x_train) != len(y_train):
            raise ValueError("Features and labels must have the same length")
        if not x_train:
            return

        torch, nn = _torch_dependencies()
        torch.manual_seed(self.random_seed)

        self.feature_names = sorted(x_train[0])
        self.training_prior = float(sum(y_train) / len(y_train)) if y_train else 0.5
        matrix = self._matrix(x_train)
        self.feature_means, self.feature_stds = self._fit_standardizer(matrix)
        scaled = self._scale_matrix(matrix)

        sequences, labels = self._build_training_tensors(torch, scaled, y_train)
        if sequences is None or labels is None:
            return

        model = _build_itransformer_module(
            torch=torch,
            nn=nn,
            sequence_length=self.sequence_length,
            feature_count=len(self.feature_names),
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight_tensor(torch, y_train))
        dataset = torch.utils.data.TensorDataset(sequences, labels)
        generator = torch.Generator().manual_seed(self.random_seed)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=max(1, self.batch_size),
            shuffle=True,
            generator=generator,
        )

        model.train()
        for _ in range(max(1, self.epochs)):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        self.model = model.cpu()

    def predict(self, x: list[dict[str, float]]) -> list[int]:
        return [int(probability >= 0.5) for probability in self.predict_proba(x)]

    def predict_proba(self, x: list[dict[str, float]]) -> list[float]:
        if not x:
            return []
        if self.model is None or not self.feature_names:
            return [self.training_prior for _ in x]

        torch, _ = _torch_dependencies()
        matrix = self._matrix(x)
        scaled = self._scale_matrix(matrix)
        sequences, sequence_indices = self._build_prediction_tensor(torch, scaled)
        probabilities = [self.training_prior for _ in x]
        if sequences is None:
            return probabilities

        self.model.eval()
        with torch.no_grad():
            logits = self.model(sequences)
            values = torch.sigmoid(logits).cpu().tolist()
        for index, probability in zip(sequence_indices, values):
            probabilities[index] = float(max(0.0, min(1.0, probability)))
        return probabilities

    def feature_importances(self) -> dict[str, float]:
        return {}

    def save(self, path: Path) -> None:
        torch, _ = _torch_dependencies()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": self.model_type,
                "params": {
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
                    "pos_weight": self.pos_weight,
                },
                "feature_names": self.feature_names,
                "feature_means": self.feature_means,
                "feature_stds": self.feature_stds,
                "training_prior": self.training_prior,
                "state_dict": self.model.state_dict() if self.model is not None else None,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "ITransformerSequenceMLModel":
        torch, nn = _torch_dependencies()
        payload = torch.load(path, map_location="cpu")
        if payload.get("model_type") != cls.model_type:
            raise ValueError(f"Unsupported model payload: {payload.get('model_type')}")
        model = cls(**payload.get("params", {}))
        model.feature_names = list(payload.get("feature_names", []))
        model.feature_means = [float(value) for value in payload.get("feature_means", [])]
        model.feature_stds = [float(value) for value in payload.get("feature_stds", [])]
        model.training_prior = float(payload.get("training_prior", 0.5))
        state_dict = payload.get("state_dict")
        if state_dict is not None and model.feature_names:
            model.model = _build_itransformer_module(
                torch=torch,
                nn=nn,
                sequence_length=model.sequence_length,
                feature_count=len(model.feature_names),
                d_model=model.d_model,
                nhead=model.nhead,
                num_layers=model.num_layers,
                dim_feedforward=model.dim_feedforward,
                dropout=model.dropout,
            )
            model.model.load_state_dict(state_dict)
            model.model.eval()
        return model

    def _matrix(self, rows: list[dict[str, float]]) -> list[list[float]]:
        return [[float(row.get(name, 0.0) or 0.0) for name in self.feature_names] for row in rows]

    def _fit_standardizer(self, matrix: list[list[float]]) -> tuple[list[float], list[float]]:
        if not matrix:
            return [], []
        columns = list(zip(*matrix))
        means = [sum(column) / len(column) for column in columns]
        stds = []
        for column, mean in zip(columns, means):
            variance = sum((value - mean) ** 2 for value in column) / max(1, len(column) - 1)
            stds.append(math.sqrt(variance) if variance > 1e-12 else 1.0)
        return means, stds

    def _scale_matrix(self, matrix: list[list[float]]) -> list[list[float]]:
        return [
            [
                (float(value) - self.feature_means[index]) / self.feature_stds[index]
                for index, value in enumerate(row)
            ]
            for row in matrix
        ]

    def _build_training_tensors(self, torch: Any, matrix: list[list[float]], labels: list[int]) -> tuple[Any | None, Any | None]:
        if len(matrix) < self.sequence_length:
            return None, None
        sequences = []
        sequence_labels = []
        for end_index in range(self.sequence_length - 1, len(matrix)):
            start_index = end_index - self.sequence_length + 1
            sequences.append(matrix[start_index : end_index + 1])
            sequence_labels.append(float(labels[end_index]))
        return torch.tensor(sequences, dtype=torch.float32), torch.tensor(sequence_labels, dtype=torch.float32)

    def _build_prediction_tensor(self, torch: Any, matrix: list[list[float]]) -> tuple[Any | None, list[int]]:
        if len(matrix) < self.sequence_length:
            return None, []
        sequences = []
        indices = []
        for end_index in range(self.sequence_length - 1, len(matrix)):
            start_index = end_index - self.sequence_length + 1
            sequences.append(matrix[start_index : end_index + 1])
            indices.append(end_index)
        return torch.tensor(sequences, dtype=torch.float32), indices

    def _pos_weight_tensor(self, torch: Any, labels: list[int]) -> Any | None:
        if self.pos_weight is None or str(self.pos_weight).lower() in {"none", "false", "0"}:
            return None
        if str(self.pos_weight).lower() == "auto":
            positives = sum(1 for label in labels if int(label) == 1)
            negatives = len(labels) - positives
            if positives <= 0 or negatives <= 0:
                return None
            return torch.tensor([negatives / positives], dtype=torch.float32, device=self.device)
        return torch.tensor([float(self.pos_weight)], dtype=torch.float32, device=self.device)


def _build_itransformer_module(
    torch: Any,
    nn: Any,
    sequence_length: int,
    feature_count: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
) -> Any:
    class ITransformerClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.temporal_projection = nn.Linear(sequence_length, d_model)
            self.variable_embedding = nn.Parameter(torch.zeros(1, feature_count, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

        def forward(self, x: Any) -> Any:
            # x: [batch, sequence_length, feature_count]. Invert dimensions so
            # each feature/asset column becomes a token with temporal context.
            tokens = x.transpose(1, 2)
            encoded = self.temporal_projection(tokens) + self.variable_embedding[:, : tokens.shape[1], :]
            encoded = self.encoder(encoded)
            pooled = self.norm(encoded.mean(dim=1))
            return self.head(pooled).squeeze(-1)

    return ITransformerClassifier()
