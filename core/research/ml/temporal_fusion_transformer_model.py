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


DEFAULT_KNOWN_FUTURE_FEATURES = [
    "day_of_week",
    "month",
    "is_month_end",
    "rebalance_frequency",
    "days_until_next_rebalance",
]


class TemporalFusionTransformerMLModel:
    """Compact research-only TFT-style model.

    This is an intentionally small first implementation: observed historical
    features pass through a variable-selection gate and temporal encoder, known
    future/calendar features pass through a separate projection, and the fused
    representation drives the existing binary classifier contract plus optional
    auxiliary forecasts/diagnostics.
    """

    model_type = "temporal_fusion_transformer"

    def __init__(
        self,
        sequence_length: int = 64,
        hidden_size: int = 64,
        attention_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.15,
        epochs: int = 30,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0005,
        random_seed: int = 42,
        device: str = "cpu",
        known_future_features: list[str] | None = None,
    ):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if hidden_size < 4:
            raise ValueError("hidden_size must be at least 4")
        if attention_heads < 1:
            raise ValueError("attention_heads must be at least one")
        if hidden_size % attention_heads != 0:
            raise ValueError("hidden_size must be divisible by attention_heads")

        self.sequence_length = int(sequence_length)
        self.hidden_size = int(hidden_size)
        self.attention_heads = int(attention_heads)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.random_seed = int(random_seed)
        self.device = str(device)
        self.known_future_features = list(known_future_features or DEFAULT_KNOWN_FUTURE_FEATURES)

        self.observed_feature_names: list[str] = []
        self.known_feature_names: list[str] = []
        self.feature_means: dict[str, float] = {}
        self.feature_stds: dict[str, float] = {}
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

        self.known_feature_names = [
            name for name in self.known_future_features if name in x_train[0]
        ]
        self.observed_feature_names = [
            name
            for name in sorted(x_train[0])
            if name not in self.known_feature_names
            and not any(name.startswith(prefix) for prefix in LEAKAGE_PREFIXES)
        ]
        self.training_prior = float(sum(y_train) / len(y_train)) if y_train else 0.5
        self._fit_standardizer(x_train)
        observed, known = self._matrices(x_train)
        sequences, known_tensor, labels = self._build_training_tensors(
            torch,
            observed,
            known,
            y_train,
        )
        if sequences is None or labels is None or known_tensor is None or len(set(y_train)) < 2:
            return

        model = _build_tft_module(
            torch=torch,
            nn=nn,
            observed_feature_count=len(self.observed_feature_names),
            known_feature_count=max(1, len(self.known_feature_names)),
            sequence_length=self.sequence_length,
            hidden_size=self.hidden_size,
            attention_heads=self.attention_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight_tensor(torch, labels))
        dataset = torch.utils.data.TensorDataset(sequences, known_tensor, labels)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=max(1, self.batch_size),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_seed),
        )
        model.train()
        for _ in range(max(1, self.epochs)):
            for batch_observed, batch_known, batch_y in loader:
                batch_observed = batch_observed.to(self.device)
                batch_known = batch_known.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits, _, _ = model(batch_observed, batch_known)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        self.model = model.cpu()

    def predict(self, x: list[dict[str, float]]) -> list[int]:
        return [int(value >= 0.5) for value in self.predict_proba(x)]

    def predict_proba(self, x: list[dict[str, float]]) -> list[float]:
        return [
            row["probability_should_reduce_exposure"]
            for row in self.predict_tft_outputs(x)
        ]

    def predict_tft_outputs(self, x: list[dict[str, float]]) -> list[dict[str, float]]:
        if not x:
            return []
        rows = [self._prior_output_row() for _ in x]
        if self.model is None or not self.observed_feature_names:
            return rows
        torch, _ = _torch_dependencies()
        observed, known = self._matrices(x)
        sequences, known_tensor, indices = self._build_prediction_tensors(torch, observed, known)
        if sequences is None or known_tensor is None:
            return rows
        self.model.eval()
        with torch.no_grad():
            logits, auxiliary, diagnostics = self.model(sequences, known_tensor)
            probabilities = torch.sigmoid(logits).cpu().tolist()
            auxiliary_rows = auxiliary.cpu().tolist()
            diagnostic_rows = diagnostics.cpu().tolist()
        for index, probability, aux, diagnostics_row in zip(
            indices,
            probabilities,
            auxiliary_rows,
            diagnostic_rows,
        ):
            probability = float(max(0.0, min(1.0, probability)))
            rows[index] = {
                "probability_should_reduce_exposure": probability,
                "predicted_forward_return_5d": float(aux[0]),
                "predicted_forward_return_10d": float(aux[1]),
                "predicted_future_volatility": abs(float(aux[2])),
                "predicted_future_drawdown": -abs(float(aux[3])),
                "tft_feature_selection_entropy": float(max(0.0, diagnostics_row[0])),
                "tft_attention_concentration": float(max(0.0, min(1.0, diagnostics_row[1]))),
                "tft_gating_saturation": float(max(0.0, min(1.0, diagnostics_row[2]))),
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
                    "attention_heads": self.attention_heads,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "random_seed": self.random_seed,
                    "device": self.device,
                    "known_future_features": self.known_future_features,
                },
                "observed_feature_names": self.observed_feature_names,
                "known_feature_names": self.known_feature_names,
                "feature_means": self.feature_means,
                "feature_stds": self.feature_stds,
                "training_prior": self.training_prior,
                "state_dict": self.model.state_dict() if self.model is not None else None,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "TemporalFusionTransformerMLModel":
        torch, nn = _torch_dependencies()
        payload = torch.load(path, map_location="cpu")
        if payload.get("model_type") != cls.model_type:
            raise ValueError(f"Unsupported model payload: {payload.get('model_type')}")
        model = cls(**payload.get("params", {}))
        model.observed_feature_names = list(payload.get("observed_feature_names", []))
        model.known_feature_names = list(payload.get("known_feature_names", []))
        model.feature_means = {key: float(value) for key, value in payload.get("feature_means", {}).items()}
        model.feature_stds = {key: float(value) for key, value in payload.get("feature_stds", {}).items()}
        model.training_prior = float(payload.get("training_prior", 0.5))
        state_dict = payload.get("state_dict")
        if state_dict is not None and model.observed_feature_names:
            network = _build_tft_module(
                torch=torch,
                nn=nn,
                observed_feature_count=len(model.observed_feature_names),
                known_feature_count=max(1, len(model.known_feature_names)),
                sequence_length=model.sequence_length,
                hidden_size=model.hidden_size,
                attention_heads=model.attention_heads,
                num_layers=model.num_layers,
                dropout=model.dropout,
            )
            network.load_state_dict(state_dict)
            model.model = network.cpu()
        return model

    def _fit_standardizer(self, rows: list[dict[str, float]]) -> None:
        names = [*self.observed_feature_names, *self.known_feature_names]
        self.feature_means = {}
        self.feature_stds = {}
        for name in names:
            values = [float(row.get(name, 0.0) or 0.0) for row in rows]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
            self.feature_means[name] = mean
            self.feature_stds[name] = math.sqrt(variance) if variance > 1e-12 else 1.0

    def _matrices(self, rows: list[dict[str, float]]) -> tuple[list[list[float]], list[list[float]]]:
        observed = [[self._scaled(row, name) for name in self.observed_feature_names] for row in rows]
        known_names = self.known_feature_names or ["__known_bias"]
        known = [
            [self._scaled(row, name) if name != "__known_bias" else 0.0 for name in known_names]
            for row in rows
        ]
        return observed, known

    def _scaled(self, row: dict[str, float], name: str) -> float:
        return (
            float(row.get(name, 0.0) or 0.0) - self.feature_means.get(name, 0.0)
        ) / self.feature_stds.get(name, 1.0)

    def _build_training_tensors(self, torch: Any, observed: list[list[float]], known: list[list[float]], labels: list[int]) -> tuple[Any | None, Any | None, Any | None]:
        indices = build_sequence_indices(self._context_group_ids(len(observed)), self.sequence_length)
        if not indices:
            return None, None, None
        sequences = [[observed[index] for index in row] for row in indices]
        known_rows = [known[row[-1]] for row in indices]
        sequence_labels = [float(labels[row[-1]]) for row in indices]
        return (
            torch.tensor(sequences, dtype=torch.float32),
            torch.tensor(known_rows, dtype=torch.float32),
            torch.tensor(sequence_labels, dtype=torch.float32),
        )

    def _build_prediction_tensors(self, torch: Any, observed: list[list[float]], known: list[list[float]]) -> tuple[Any | None, Any | None, list[int]]:
        indices = build_sequence_indices(self._context_group_ids(len(observed)), self.sequence_length)
        if not indices:
            return None, None, []
        return (
            torch.tensor([[observed[index] for index in row] for row in indices], dtype=torch.float32),
            torch.tensor([known[row[-1]] for row in indices], dtype=torch.float32),
            [row[-1] for row in indices],
        )

    def _context_group_ids(self, sample_count: int) -> list[str]:
        if len(self._sequence_group_ids) == sample_count:
            return list(self._sequence_group_ids)
        return ["global" for _ in range(sample_count)]

    def _prior_output_row(self) -> dict[str, float]:
        return {
            "probability_should_reduce_exposure": float(self.training_prior),
            "predicted_forward_return_5d": 0.0,
            "predicted_forward_return_10d": 0.0,
            "predicted_future_volatility": 0.0,
            "predicted_future_drawdown": 0.0,
            "tft_feature_selection_entropy": 0.0,
            "tft_attention_concentration": 0.0,
            "tft_gating_saturation": 0.0,
        }

    @staticmethod
    def _pos_weight_tensor(torch: Any, labels: Any) -> Any:
        positive_count = float(labels.sum().item())
        negative_count = float(labels.numel() - positive_count)
        return torch.tensor([negative_count / max(positive_count, 1.0)], dtype=torch.float32)


def _build_tft_module(
    torch: Any,
    nn: Any,
    observed_feature_count: int,
    known_feature_count: int,
    sequence_length: int,
    hidden_size: int,
    attention_heads: int,
    num_layers: int,
    dropout: float,
) -> Any:
    class TinyTFTModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.variable_gate = nn.Linear(observed_feature_count, observed_feature_count)
            self.observed_projection = nn.Linear(observed_feature_count, hidden_size)
            self.known_projection = nn.Linear(known_feature_count, hidden_size)
            self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length, hidden_size))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=attention_heads,
                dim_feedforward=hidden_size * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.gate = nn.Linear(hidden_size * 2, hidden_size)
            self.classifier = nn.Linear(hidden_size, 1)
            self.auxiliary = nn.Linear(hidden_size, 4)

        def forward(self, observed, known):
            variable_weights = torch.sigmoid(self.variable_gate(observed))
            selected = observed * variable_weights
            encoded = self.observed_projection(selected) + self.position_embedding[:, : observed.shape[1], :]
            temporal = self.encoder(encoded)[:, -1, :]
            known_encoded = self.known_projection(known)
            gate_values = torch.sigmoid(self.gate(torch.cat([temporal, known_encoded], dim=1)))
            fused = temporal * gate_values + known_encoded * (1.0 - gate_values)
            selection_mean = variable_weights.mean(dim=1)
            entropy = -(
                selection_mean * torch.log(selection_mean.clamp_min(1e-6))
                + (1.0 - selection_mean) * torch.log((1.0 - selection_mean).clamp_min(1e-6))
            ).mean(dim=1)
            concentration = selection_mean.max(dim=1).values
            saturation = (gate_values - 0.5).abs().mean(dim=1) * 2.0
            diagnostics = torch.stack([entropy, concentration, saturation], dim=1)
            return self.classifier(fused).squeeze(-1), self.auxiliary(fused), diagnostics

    return TinyTFTModule()
