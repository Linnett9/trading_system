from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from core.research.ml.sequence_dataset import (
    build_sequence_indices,
    sequence_group_ids_from_metadata,
)


LEAKAGE_PREFIXES = (
    "actual_",
    "forward_",
    "future_",
    "max_adverse_",
    "max_favourable_",
)


def _torch_dependencies() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "Market context encoder requires PyTorch. Install it with: python -m pip install torch"
        ) from exc
    return torch, nn


class MarketContextEncoderMLModel:
    """Research-only temporal encoder for market regime and risk compatibility."""

    model_type = "market_context_encoder"

    def __init__(
        self,
        sequence_length: int = 63,
        hidden_size: int = 32,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        dropout: float = 0.10,
        random_seed: int = 42,
        device: str = "cpu",
        risk_multiplier_floor: float = 0.25,
        risk_multiplier_ceiling: float = 1.25,
    ):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if hidden_size < 4:
            raise ValueError("hidden_size must be at least 4")
        if risk_multiplier_floor <= 0:
            raise ValueError("risk_multiplier_floor must be positive")
        if risk_multiplier_ceiling < risk_multiplier_floor:
            raise ValueError("risk_multiplier_ceiling must be >= risk_multiplier_floor")

        self.sequence_length = int(sequence_length)
        self.hidden_size = int(hidden_size)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)
        self.random_seed = int(random_seed)
        self.device = str(device)
        self.risk_multiplier_floor = float(risk_multiplier_floor)
        self.risk_multiplier_ceiling = float(risk_multiplier_ceiling)

        self.feature_names: list[str] = []
        self.feature_means: list[float] = []
        self.feature_stds: list[float] = []
        self.training_prior = 0.5
        self.model: Any = None
        self._sequence_group_ids: list[str] = []

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
        if len(x_train) != len(y_train):
            raise ValueError("Features and labels must have the same length")
        if not x_train:
            return

        torch, nn = _torch_dependencies()
        torch.manual_seed(self.random_seed)

        self.feature_names = _safe_feature_names(x_train[0])
        self.training_prior = float(sum(y_train) / len(y_train)) if y_train else 0.5
        matrix = self._matrix(x_train)
        self.feature_means, self.feature_stds = self._fit_standardizer(matrix)
        scaled = self._scale_matrix(matrix)
        sequences, labels = self._build_training_tensors(torch, scaled, y_train)
        if sequences is None or labels is None or len(set(y_train)) < 2:
            return

        model = _build_market_context_module(
            torch=torch,
            nn=nn,
            feature_count=len(self.feature_names),
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        pos_weight = self._pos_weight_tensor(torch, labels)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        dataset = torch.utils.data.TensorDataset(sequences, labels)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=max(1, self.batch_size),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_seed),
        )

        model.train()
        for _ in range(max(1, self.epochs)):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits, _ = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        self.model = model.cpu()

    def predict(self, x: list[dict[str, float]]) -> list[int]:
        return [int(value >= 0.5) for value in self.predict_proba(x)]

    def predict_proba(self, x: list[dict[str, float]]) -> list[float]:
        return [
            row["market_regime_probability_risk_off"]
            for row in self.predict_context(x)
        ]

    def predict_context(self, x: list[dict[str, float]]) -> list[dict[str, float]]:
        if not x:
            return []
        rows = [self._prior_context_row() for _ in x]
        if self.model is None or not self.feature_names:
            return rows

        torch, _ = _torch_dependencies()
        matrix = self._matrix(x)
        scaled = self._scale_matrix(matrix)
        sequences, indices = self._build_prediction_tensor(torch, scaled)
        if sequences is None:
            return rows

        self.model.eval()
        with torch.no_grad():
            logits, embeddings = self.model(sequences)
            probabilities = torch.sigmoid(logits).cpu().tolist()
            embedding_rows = embeddings.cpu().tolist()

        for index, probability, embedding in zip(indices, probabilities, embedding_rows):
            probability = float(max(0.0, min(1.0, probability)))
            rows[index] = {
                "market_regime_probability_risk_off": probability,
                "market_regime_class": float(int(probability >= 0.5)),
                "risk_multiplier": self._risk_multiplier(probability),
                "volatility_regime_score": probability,
                "drawdown_regime_score": probability,
                "breadth_regime_score": 1.0 - probability,
                "liquidity_regime_score": 1.0 - probability,
                "context_embedding_001": float(embedding[0]) if len(embedding) > 0 else 0.0,
                "context_embedding_002": float(embedding[1]) if len(embedding) > 1 else 0.0,
                "context_embedding_003": float(embedding[2]) if len(embedding) > 2 else 0.0,
            }
        return rows

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
                    "hidden_size": self.hidden_size,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "dropout": self.dropout,
                    "random_seed": self.random_seed,
                    "device": self.device,
                    "risk_multiplier_floor": self.risk_multiplier_floor,
                    "risk_multiplier_ceiling": self.risk_multiplier_ceiling,
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
    def load(cls, path: Path) -> "MarketContextEncoderMLModel":
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
            network = _build_market_context_module(
                torch=torch,
                nn=nn,
                feature_count=len(model.feature_names),
                hidden_size=model.hidden_size,
                dropout=model.dropout,
            )
            network.load_state_dict(state_dict)
            model.model = network.cpu()
        return model

    def _matrix(self, rows: list[dict[str, float]]) -> list[list[float]]:
        return [[float(row.get(name, 0.0) or 0.0) for name in self.feature_names] for row in rows]

    def _fit_standardizer(self, matrix: list[list[float]]) -> tuple[list[float], list[float]]:
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
        indices = build_sequence_indices(self._context_group_ids(len(matrix)), self.sequence_length)
        if not indices:
            return None, None
        sequences = [[matrix[index] for index in row] for row in indices]
        sequence_labels = [float(labels[row[-1]]) for row in indices]
        return torch.tensor(sequences, dtype=torch.float32), torch.tensor(sequence_labels, dtype=torch.float32)

    def _build_prediction_tensor(self, torch: Any, matrix: list[list[float]]) -> tuple[Any | None, list[int]]:
        indices = build_sequence_indices(self._context_group_ids(len(matrix)), self.sequence_length)
        if not indices:
            return None, []
        sequences = [[matrix[index] for index in row] for row in indices]
        return torch.tensor(sequences, dtype=torch.float32), [row[-1] for row in indices]

    def _context_group_ids(self, sample_count: int) -> list[str]:
        if len(self._sequence_group_ids) == sample_count:
            return list(self._sequence_group_ids)
        return ["global" for _ in range(sample_count)]

    def _prior_context_row(self) -> dict[str, float]:
        probability = float(self.training_prior)
        return {
            "market_regime_probability_risk_off": probability,
            "market_regime_class": float(int(probability >= 0.5)),
            "risk_multiplier": self._risk_multiplier(probability),
            "volatility_regime_score": probability,
            "drawdown_regime_score": probability,
            "breadth_regime_score": 1.0 - probability,
            "liquidity_regime_score": 1.0 - probability,
            "context_embedding_001": 0.0,
            "context_embedding_002": 0.0,
            "context_embedding_003": 0.0,
        }

    def _risk_multiplier(self, probability: float) -> float:
        span = self.risk_multiplier_ceiling - self.risk_multiplier_floor
        return self.risk_multiplier_ceiling - span * float(max(0.0, min(1.0, probability)))

    @staticmethod
    def _pos_weight_tensor(torch: Any, labels: Any) -> Any:
        positive_count = float(labels.sum().item())
        negative_count = float(labels.numel() - positive_count)
        return torch.tensor([negative_count / max(positive_count, 1.0)], dtype=torch.float32)


def _safe_feature_names(row: dict[str, float]) -> list[str]:
    return [
        name
        for name in sorted(row)
        if not any(name.startswith(prefix) for prefix in LEAKAGE_PREFIXES)
    ]


def _build_market_context_module(torch: Any, nn: Any, feature_count: int, hidden_size: int, dropout: float) -> Any:
    class MarketContextModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(feature_count, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
            )
            self.classifier = nn.Linear(hidden_size, 1)

        def forward(self, x):
            pooled = x.mean(dim=1)
            embedding = self.encoder(pooled)
            return self.classifier(embedding).squeeze(-1), embedding

    return MarketContextModule()
