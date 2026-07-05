from __future__ import annotations

from core.research.ml.models.multitask_transformer_types import (
    LEAKAGE_FEATURE_NAMES,
    LEAKAGE_FEATURE_PREFIXES,
)
from core.research.ml.models.transformer_model import _torch_dependencies


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
