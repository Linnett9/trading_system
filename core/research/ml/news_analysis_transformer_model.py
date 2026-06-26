from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from core.research.ml.market_context_encoder_model import _torch_dependencies
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


class NewsAnalysisTransformerMLModel:
    """Research-only sequence model for market rows enriched with news features."""

    model_type = "news_analysis_transformer"

    def __init__(
        self,
        sequence_length: int = 63,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.10,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        random_seed: int = 42,
        device: str = "cpu",
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

        model = _build_news_transformer_module(
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
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight_tensor(torch, labels))
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
            row["news_probability_should_reduce_exposure"]
            for row in self.predict_news_components(x)
        ]

    def predict_news_components(self, x: list[dict[str, float]]) -> list[dict[str, float]]:
        if not x:
            return []
        rows = [self._prior_news_row(row) for row in x]
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
                **self._news_feature_summary(x[index]),
                "news_probability_should_reduce_exposure": probability,
                "news_attention_proxy": float(max(0.0, min(1.0, abs(embedding[0]))))
                if embedding else 0.0,
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
    def load(cls, path: Path) -> "NewsAnalysisTransformerMLModel":
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
            network = _build_news_transformer_module(
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
        return (
            torch.tensor([[matrix[index] for index in row] for row in indices], dtype=torch.float32),
            torch.tensor([float(labels[row[-1]]) for row in indices], dtype=torch.float32),
        )

    def _build_prediction_tensor(self, torch: Any, matrix: list[list[float]]) -> tuple[Any | None, list[int]]:
        indices = build_sequence_indices(self._context_group_ids(len(matrix)), self.sequence_length)
        if not indices:
            return None, []
        return torch.tensor([[matrix[index] for index in row] for row in indices], dtype=torch.float32), [row[-1] for row in indices]

    def _context_group_ids(self, sample_count: int) -> list[str]:
        if len(self._sequence_group_ids) == sample_count:
            return list(self._sequence_group_ids)
        return ["global" for _ in range(sample_count)]

    def _prior_news_row(self, row: dict[str, float]) -> dict[str, float]:
        return {
            **self._news_feature_summary(row),
            "news_probability_should_reduce_exposure": float(self.training_prior),
            "news_attention_proxy": 0.0,
        }

    @staticmethod
    def _news_feature_summary(row: dict[str, float]) -> dict[str, float]:
        sentiment_values = [
            float(value)
            for name, value in row.items()
            if "sentiment" in name and "count" not in name
        ]
        count_values = [
            float(value)
            for name, value in row.items()
            if "news" in name and "count" in name
        ]
        return {
            "news_sentiment_score": sum(sentiment_values) / len(sentiment_values)
            if sentiment_values else 0.0,
            "news_volume_score": sum(count_values) if count_values else 0.0,
        }

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


def _build_news_transformer_module(
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
    class NewsTransformerModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_projection = nn.Linear(feature_count, d_model)
            self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.classifier = nn.Linear(d_model, 1)

        def forward(self, x):
            encoded = self.input_projection(x) + self.position_embedding[:, : x.shape[1], :]
            encoded = self.encoder(encoded)
            pooled = encoded[:, -1, :]
            return self.classifier(pooled).squeeze(-1), pooled

    return NewsTransformerModule()
