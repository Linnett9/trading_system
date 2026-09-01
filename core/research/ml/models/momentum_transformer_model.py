from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from core.research.ml.data.sequence_window_authority import (
    build_sequence_indices_from_context,
    sequence_context_rows_from_metadata,
)
from core.research.ml.models.torch_checkpointing import (
    TorchCheckpointSession,
    binary_validation_loss,
    checkpoint_identity,
    checkpoint_validation_fraction,
    validation_split_tensors,
)


def _torch_dependencies() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "Momentum Transformer research model requires PyTorch. "
            "Install it with: python -m pip install torch"
        ) from exc
    return torch, nn


class MomentumTransformerSequenceMLModel:
    """Research-only trend/regime-sensitive sequence classifier.

    The first version keeps the existing MLExperimentRunner contract: fit,
    predict_proba, predict, feature_importances, save, and load. Extra research
    outputs are exposed through predict_components but are not required by the
    runner or meta ensemble yet.
    """

    model_type = "momentum_transformer"

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
        size_multiplier_floor: float = 0.25,
        size_multiplier_ceiling: float = 1.25,
        strict_context_required: bool = False,
    ):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if d_model < 4:
            raise ValueError("d_model must be at least 4")
        if nhead < 1:
            raise ValueError("nhead must be at least one")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if size_multiplier_floor <= 0:
            raise ValueError("size_multiplier_floor must be positive")
        if size_multiplier_ceiling < size_multiplier_floor:
            raise ValueError("size_multiplier_ceiling must be >= size_multiplier_floor")

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
        self.size_multiplier_floor = float(size_multiplier_floor)
        self.size_multiplier_ceiling = float(size_multiplier_ceiling)
        self.strict_context_required = bool(strict_context_required)

        self.feature_names: list[str] = []
        self.feature_means: list[float] = []
        self.feature_stds: list[float] = []
        self.training_prior: float = 0.5
        self.model: Any = None
        self._sequence_context_rows: list[dict[str, Any]] = []

    def set_sequence_context(
        self,
        metadata: list[dict[str, str]] | None = None,
        feature_dates: list[str] | None = None,
        feature_ids: list[str] | None = None,
        label_start_dates: list[str] | None = None,
        label_end_dates: list[str] | None = None,
    ) -> None:
        sample_count = len(feature_dates or metadata or [])
        self._sequence_context_rows = sequence_context_rows_from_metadata(
            metadata,
            sample_count,
            feature_dates=feature_dates,
            feature_ids=feature_ids,
            label_start_dates=label_start_dates,
            label_end_dates=label_end_dates,
        )

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

        model = _build_momentum_transformer_module(
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
        train_tensors, train_labels, validation_tensors, validation_labels = (
            validation_split_tensors(
                (sequences,),
                labels,
                fraction=checkpoint_validation_fraction(self),
            )
        )
        dataset = torch.utils.data.TensorDataset(train_tensors[0], train_labels)
        generator = torch.Generator().manual_seed(self.random_seed)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=max(1, self.batch_size),
            shuffle=True,
            generator=generator,
        )
        checkpoint = TorchCheckpointSession(
            torch=torch,
            model_owner=self,
            network=model,
            optimizer=optimizer,
            identity=checkpoint_identity(self, x_train=x_train, y_train=y_train),
            total_epochs=max(1, self.epochs),
        )
        resume = checkpoint.restore_if_compatible()

        model.train()
        for epoch in range(resume.start_epoch, max(1, self.epochs)):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits, _, _ = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            checkpoint.save_epoch(
                completed_epoch=epoch,
                validation_metric=binary_validation_loss(
                    torch,
                    model,
                    criterion,
                    validation_tensors,
                    validation_labels,
                    self.device,
                    forward=lambda network, tensors: network(tensors[0].to(self.device))[0],
                ),
            )

        checkpoint.restore_best_weights()
        self.model = model.cpu()

    def predict(self, x: list[dict[str, float]]) -> list[int]:
        return [int(probability >= 0.5) for probability in self.predict_proba(x)]

    def predict_proba(self, x: list[dict[str, float]]) -> list[float]:
        components = self.predict_components(x)
        return [row["probability_should_reduce_exposure"] for row in components]

    def predict_components(self, x: list[dict[str, float]]) -> list[dict[str, float]]:
        if not x:
            return []
        components = [
            {
                "probability_should_reduce_exposure": float(self.training_prior),
                "trend_score": 0.0,
                "regime_score": 0.5,
                "size_multiplier": self._size_multiplier_from_probability(
                    float(self.training_prior)
                ),
            }
            for _ in x
        ]
        if self.model is None or not self.feature_names:
            return components

        torch, _ = _torch_dependencies()
        matrix = self._matrix(x)
        scaled = self._scale_matrix(matrix)
        sequences, sequence_indices = self._build_prediction_tensor(torch, scaled)
        if sequences is None:
            return components

        self.model.eval()
        with torch.no_grad():
            logits, trend_logits, regime_logits = self.model(sequences)
            probabilities = torch.sigmoid(logits).cpu().tolist()
            trend_scores = torch.tanh(trend_logits).cpu().tolist()
            regime_scores = torch.sigmoid(regime_logits).cpu().tolist()

        for index, probability, trend_score, regime_score in zip(
            sequence_indices,
            probabilities,
            trend_scores,
            regime_scores,
        ):
            probability = float(max(0.0, min(1.0, probability)))
            components[index] = {
                "probability_should_reduce_exposure": probability,
                "trend_score": float(max(-1.0, min(1.0, trend_score))),
                "regime_score": float(max(0.0, min(1.0, regime_score))),
                "size_multiplier": self._size_multiplier_from_probability(probability),
            }
        return components

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
                    "size_multiplier_floor": self.size_multiplier_floor,
                    "size_multiplier_ceiling": self.size_multiplier_ceiling,
                    "strict_context_required": self.strict_context_required,
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
    def load(cls, path: Path) -> "MomentumTransformerSequenceMLModel":
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
            model.model = _build_momentum_transformer_module(
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
        indices = build_sequence_indices_from_context(
            self._sequence_context_rows,
            len(matrix),
            self.sequence_length,
            strict_context_required=self.strict_context_required,
        )
        if not indices:
            return None, None
        sequences = []
        sequence_labels = []
        for row in indices:
            sequences.append([matrix[index] for index in row])
            sequence_labels.append(float(labels[row[-1]]))
        return torch.tensor(sequences, dtype=torch.float32), torch.tensor(sequence_labels, dtype=torch.float32)

    def _build_prediction_tensor(self, torch: Any, matrix: list[list[float]]) -> tuple[Any | None, list[int]]:
        sequence_indices = build_sequence_indices_from_context(
            self._sequence_context_rows,
            len(matrix),
            self.sequence_length,
            strict_context_required=self.strict_context_required,
        )
        if not sequence_indices:
            return None, []
        sequences = []
        indices = []
        for row in sequence_indices:
            sequences.append([matrix[index] for index in row])
            indices.append(row[-1])
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

    def _size_multiplier_from_probability(self, probability: float) -> float:
        risk_on_score = 1.0 - float(max(0.0, min(1.0, probability)))
        return self.size_multiplier_floor + (
            self.size_multiplier_ceiling - self.size_multiplier_floor
        ) * risk_on_score


def _build_momentum_transformer_module(
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
    class MomentumTransformerClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(feature_count, d_model)
            self.position = nn.Parameter(torch.zeros(1, sequence_length, d_model))
            self.delta_projection = nn.Linear(feature_count, d_model)
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
            self.classification_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))
            self.trend_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))
            self.regime_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

        def forward(self, x: Any, padding_mask: Any | None = None) -> tuple[Any, Any, Any]:
            deltas = x[:, -1, :] - x[:, 0, :]
            tokens = self.input_projection(x) + self.position[:, : x.shape[1], :]
            tokens = tokens + self.delta_projection(deltas).unsqueeze(1)
            causal_mask = torch.triu(
                torch.ones(x.shape[1], x.shape[1], device=tokens.device, dtype=torch.bool),
                diagonal=1,
            )
            if padding_mask is not None:
                padding_mask = padding_mask.to(device=tokens.device, dtype=torch.bool)
            encoded = self.encoder(tokens, mask=causal_mask, src_key_padding_mask=padding_mask)
            encoded = torch.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0)
            if padding_mask is None:
                pooled = encoded[:, -1, :]
            else:
                valid_lengths = (~padding_mask).sum(dim=1).clamp(min=1) - 1
                pooled = encoded[torch.arange(encoded.shape[0], device=encoded.device), valid_lengths]
            pooled = self.norm(pooled)
            return (
                self.classification_head(pooled).squeeze(-1),
                self.trend_head(pooled).squeeze(-1),
                self.regime_head(pooled).squeeze(-1),
            )

    return MomentumTransformerClassifier()
