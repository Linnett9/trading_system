from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SequenceRegressorConfig:
    architecture: str
    sequence_length: int = 13
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    random_seed: int = 42
    device: str = "cpu"
    d_model: int = 32
    nhead: int = 4
    num_layers: int = 1
    dim_feedforward: int = 64
    dropout: float = 0.10
    patch_length: int = 4
    patch_stride: int = 2
    torch_num_threads: int | None = None


class TorchSequenceReturnRegressor:
    """Regression adapter over the existing research sequence backbones.

    The adapter owns no trading behavior. It accepts already leakage-safe,
    per-symbol feature sequences and predicts a standardized forward return.
    Existing classification model classes and checkpoints remain untouched.
    """

    def __init__(self, config: SequenceRegressorConfig):
        self.config = config
        self.model: Any = None
        self.feature_means: Any = None
        self.feature_stds: Any = None
        self.target_mean = 0.0
        self.target_std = 1.0
        self.auxiliary_means: Any = None
        self.auxiliary_stds: Any = None

    def fit(
        self,
        sequences: list[list[list[float]]],
        targets: list[float],
        auxiliary_targets: list[list[float]] | None = None,
    ) -> None:
        if len(sequences) != len(targets):
            raise ValueError("Sequence features and targets must have the same length")
        if not sequences:
            return
        torch, nn = _torch_dependencies()
        if self.config.torch_num_threads is not None:
            torch.set_num_threads(max(1, self.config.torch_num_threads))
        torch.manual_seed(self.config.random_seed)
        x = torch.tensor(sequences, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32)
        self.feature_means = x.mean(dim=(0, 1), keepdim=True)
        self.feature_stds = x.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        x = (x - self.feature_means) / self.feature_stds
        self.target_mean = float(y.mean().item())
        target_std = float(y.std().item()) if len(y) > 1 else 1.0
        self.target_std = target_std if target_std > 1e-6 else 1.0
        normalized_y = (y - self.target_mean) / self.target_std

        auxiliary = None
        if self.config.architecture == "multitask_transformer":
            if not auxiliary_targets or len(auxiliary_targets) != len(sequences):
                raise ValueError("Multi-task transformer requires aligned auxiliary targets")
            auxiliary = torch.tensor(auxiliary_targets, dtype=torch.float32)
            self.auxiliary_means = auxiliary.mean(dim=0, keepdim=True)
            self.auxiliary_stds = auxiliary.std(dim=0, keepdim=True).clamp_min(1e-6)
            auxiliary = (auxiliary - self.auxiliary_means) / self.auxiliary_stds

        model = self._build_model(torch, nn, feature_count=x.shape[2]).to(
            self.config.device
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        dataset_tensors = (x, normalized_y) if auxiliary is None else (x, normalized_y, auxiliary)
        dataset = torch.utils.data.TensorDataset(*dataset_tensors)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=max(1, self.config.batch_size),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.config.random_seed),
        )
        loss_fn = nn.SmoothL1Loss()
        model.train()
        for _ in range(max(1, self.config.epochs)):
            for batch in loader:
                batch = tuple(value.to(self.config.device) for value in batch)
                optimizer.zero_grad(set_to_none=True)
                primary, model_auxiliary = self._forward(model, batch[0])
                loss = loss_fn(primary, batch[1])
                if len(batch) == 3 and model_auxiliary is not None:
                    loss = loss + 0.2 * loss_fn(model_auxiliary, batch[2])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        self.model = model.cpu()

    def predict(self, sequences: list[list[list[float]]]) -> list[float]:
        if not sequences:
            return []
        if self.model is None:
            return [self.target_mean for _ in sequences]
        torch, _ = _torch_dependencies()
        x = torch.tensor(sequences, dtype=torch.float32)
        x = (x - self.feature_means) / self.feature_stds
        self.model.eval()
        with torch.no_grad():
            normalized, _ = self._forward(self.model, x)
        return [
            float(value) * self.target_std + self.target_mean
            for value in normalized.cpu().tolist()
        ]

    def _build_model(self, torch: Any, nn: Any, *, feature_count: int) -> Any:
        config = self.config
        if config.architecture == "dlinear":
            return _DLinearRegressor(nn, config.sequence_length, feature_count)
        if config.architecture == "transformer":
            from core.research.ml.models.transformer_model import (
                _make_tiny_transformer_classifier,
            )

            return _make_tiny_transformer_classifier()(
                feature_count=feature_count,
                sequence_length=config.sequence_length,
                d_model=config.d_model,
                nhead=config.nhead,
                num_layers=config.num_layers,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
            )
        if config.architecture == "patchtst":
            from core.research.ml.models.patchtst_model import _build_patchtst_module

            return _build_patchtst_module(
                torch,
                nn,
                config.sequence_length,
                feature_count,
                config.patch_length,
                config.patch_stride,
                config.d_model,
                config.nhead,
                config.num_layers,
                config.dim_feedforward,
                config.dropout,
            )
        if config.architecture == "itransformer":
            from core.research.ml.models.itransformer_model import _build_itransformer_module

            return _build_itransformer_module(
                torch,
                nn,
                config.sequence_length,
                feature_count,
                config.d_model,
                config.nhead,
                config.num_layers,
                config.dim_feedforward,
                config.dropout,
            )
        if config.architecture == "momentum_transformer":
            from core.research.ml.models.momentum_transformer_model import (
                _build_momentum_transformer_module,
            )

            return _build_momentum_transformer_module(
                torch,
                nn,
                config.sequence_length,
                feature_count,
                config.d_model,
                config.nhead,
                config.num_layers,
                config.dim_feedforward,
                config.dropout,
            )
        if config.architecture == "multitask_transformer":
            from core.research.ml.models.multitask_transformer_model import (
                _make_multitask_transformer_module,
            )

            return _make_multitask_transformer_module()(
                feature_count=feature_count,
                sequence_length=config.sequence_length,
                d_model=config.d_model,
                nhead=config.nhead,
                num_layers=config.num_layers,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                regression_head_count=4,
            )
        if config.architecture == "market_context_encoder":
            from core.research.ml.models.market_context_encoder_model import (
                _build_market_context_module,
            )

            return _build_market_context_module(
                torch,
                nn,
                feature_count,
                config.d_model,
                config.dropout,
            )
        if config.architecture == "news_analysis_transformer":
            from core.research.ml.models.news_analysis_transformer_model import (
                _build_news_transformer_module,
            )

            return _build_news_transformer_module(
                torch,
                nn,
                config.sequence_length,
                feature_count,
                config.d_model,
                config.nhead,
                config.num_layers,
                config.dim_feedforward,
                config.dropout,
            )
        if config.architecture == "temporal_fusion_transformer":
            return _TFTRegressionWrapper(torch, nn, config, feature_count)
        raise ValueError(f"Unsupported stock-level sequence architecture: {config.architecture}")

    def _forward(self, model: Any, x: Any) -> tuple[Any, Any | None]:
        architecture = self.config.architecture
        output = model(x)
        if architecture == "momentum_transformer":
            return output[0], None
        if architecture == "multitask_transformer":
            _, regression = output
            return regression[:, 0], regression[:, 1:]
        if architecture in {"market_context_encoder", "news_analysis_transformer"}:
            return output[0], None
        if architecture == "temporal_fusion_transformer":
            return output[0], output[1]
        return output, None


def _DLinearRegressor(nn: Any, sequence_length: int, feature_count: int) -> Any:
    class DLinearRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(sequence_length * feature_count, 1)

        def forward(self, x: Any) -> Any:
            return self.head(x.reshape(x.shape[0], -1)).squeeze(-1)

    return DLinearRegressor()


def _TFTRegressionWrapper(
    torch: Any,
    nn: Any,
    config: SequenceRegressorConfig,
    feature_count: int,
) -> Any:
    from core.research.ml.models.temporal_fusion_transformer_model import _build_tft_module

    backbone = _build_tft_module(
        torch,
        nn,
        feature_count,
        feature_count,
        config.sequence_length,
        config.d_model,
        config.nhead,
        config.num_layers,
        config.dropout,
    )

    class TFTRegressionWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone

        def forward(self, x: Any) -> tuple[Any, Any]:
            primary, auxiliary, _ = self.backbone(x, x[:, -1, :])
            return primary, auxiliary

    return TFTRegressionWrapper()


def _torch_dependencies() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "Stock-level sequence regressors require PyTorch. Install PyTorch "
            "before running the full alpha benchmark suite."
        ) from exc
    return torch, nn
