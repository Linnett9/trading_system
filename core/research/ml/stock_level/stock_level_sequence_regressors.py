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
        self.feature_impute_values: Any = None
        self.target_mean = 0.0
        self.target_std = 1.0
        self.auxiliary_means: Any = None
        self.auxiliary_stds: Any = None
        self.diagnostics: dict[str, Any] = {}

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
        self.diagnostics = _tensor_quality(torch, x, y, prefix="train")
        x = self._preprocess_fit_features(torch, x)
        self.diagnostics.update(_post_preprocess_quality(torch, x, prefix="train"))
        finite_y = y[torch.isfinite(y)]
        if finite_y.numel() != y.numel():
            fill = finite_y.median() if finite_y.numel() else torch.tensor(0.0)
            y = torch.where(torch.isfinite(y), y, fill)
        self.target_mean = float(y.mean().item())
        target_std = float(y.std().item()) if len(y) > 1 else 1.0
        self.target_std = target_std if target_std > 1e-6 else 1.0
        normalized_y = (y - self.target_mean) / self.target_std
        self.diagnostics.update(
            {
                "y_train_mean": self.target_mean,
                "y_train_std": self.target_std,
                "y_train_min": float(y.min().item()) if y.numel() else None,
                "y_train_max": float(y.max().item()) if y.numel() else None,
            }
        )

        auxiliary = None
        if self.config.architecture == "multitask_transformer":
            if not auxiliary_targets or len(auxiliary_targets) != len(sequences):
                raise ValueError("Multi-task transformer requires aligned auxiliary targets")
            auxiliary = torch.tensor(auxiliary_targets, dtype=torch.float32)
            auxiliary = torch.nan_to_num(auxiliary, nan=0.0, posinf=0.0, neginf=0.0)
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
        epoch_losses: list[float] = []
        loss_finite = True
        for _ in range(max(1, self.config.epochs)):
            batch_losses: list[float] = []
            for batch in loader:
                batch = tuple(value.to(self.config.device) for value in batch)
                optimizer.zero_grad(set_to_none=True)
                primary, model_auxiliary = self._forward(model, batch[0])
                loss = loss_fn(primary, batch[1])
                if len(batch) == 3 and model_auxiliary is not None:
                    loss = loss + 0.2 * loss_fn(model_auxiliary, batch[2])
                if not bool(torch.isfinite(loss).item()):
                    loss_finite = False
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))
            epoch_losses.append(sum(batch_losses) / len(batch_losses) if batch_losses else float("nan"))
        self.model = model.cpu()
        params_finite = all(
            bool(torch.isfinite(parameter.detach()).all().item())
            for parameter in self.model.parameters()
        )
        self.diagnostics.update(
            {
                "train_loss_by_epoch": epoch_losses,
                "train_loss_finite": loss_finite and all(_is_finite(value) for value in epoch_losses),
                "model_parameters_finite": params_finite,
            }
        )

    def predict(self, sequences: list[list[list[float]]]) -> list[float]:
        if not sequences:
            return []
        if self.model is None:
            return [self.target_mean for _ in sequences]
        torch, _ = _torch_dependencies()
        x = torch.tensor(sequences, dtype=torch.float32)
        prediction_quality = _tensor_quality(torch, x, None, prefix="test")
        x = self._preprocess_predict_features(torch, x)
        prediction_quality.update(_post_preprocess_quality(torch, x, prefix="test"))
        self.model.eval()
        with torch.no_grad():
            normalized, _ = self._forward(self.model, x)
        raw_values = normalized.cpu().tolist()
        values = [
            float(value) * self.target_std + self.target_mean
            for value in raw_values
        ]
        prediction_quality.update(
            {
                "raw_prediction_count": len(raw_values),
                "raw_finite_prediction_count": sum(_is_finite(value) for value in raw_values),
                "postprocessed_prediction_count": len(values),
                "postprocessed_finite_prediction_count": sum(_is_finite(value) for value in values),
            }
        )
        self.diagnostics.update(prediction_quality)
        return values

    def _preprocess_fit_features(self, torch: Any, x: Any) -> Any:
        finite = torch.isfinite(x)
        feature_count = x.shape[2]
        flat = x.reshape(-1, feature_count)
        medians = []
        for column_index in range(feature_count):
            column = flat[:, column_index]
            finite_column = column[torch.isfinite(column)]
            medians.append(finite_column.median() if finite_column.numel() else torch.tensor(0.0))
        self.feature_impute_values = torch.stack(medians).reshape(1, 1, feature_count)
        imputed = torch.where(finite, x, self.feature_impute_values)
        self.diagnostics.update(
            {
                "feature_imputation_strategy": "train_fold_median_else_zero",
                "feature_nan_count_before_imputation": int((~finite).sum().item()),
                "feature_nan_count_after_imputation": int((~torch.isfinite(imputed)).sum().item()),
            }
        )
        self.feature_means = imputed.mean(dim=(0, 1), keepdim=True)
        self.feature_stds = imputed.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        return (imputed - self.feature_means) / self.feature_stds

    def _preprocess_predict_features(self, torch: Any, x: Any) -> Any:
        finite = torch.isfinite(x)
        impute_values = self.feature_impute_values
        if impute_values is None:
            impute_values = torch.zeros((1, 1, x.shape[2]), dtype=x.dtype)
        imputed = torch.where(finite, x, impute_values)
        self.diagnostics.update(
            {
                "test_feature_nan_count_before_imputation": int((~finite).sum().item()),
                "test_feature_nan_count_after_imputation": int((~torch.isfinite(imputed)).sum().item()),
            }
        )
        return (imputed - self.feature_means) / self.feature_stds

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


def _tensor_quality(torch: Any, x: Any, y: Any | None, *, prefix: str) -> dict[str, Any]:
    finite_x = torch.isfinite(x)
    total_x = int(x.numel())
    quality: dict[str, Any] = {
        f"x_{prefix}_finite_ratio": float(finite_x.sum().item()) / total_x if total_x else 1.0,
        f"x_{prefix}_non_finite_count": int((~finite_x).sum().item()),
    }
    if y is not None:
        finite_y = torch.isfinite(y)
        quality.update(
            {
                "y_train_finite_count": int(finite_y.sum().item()),
                "y_train_non_finite_count": int((~finite_y).sum().item()),
            }
        )
    return quality


def _post_preprocess_quality(torch: Any, x: Any, *, prefix: str) -> dict[str, Any]:
    finite_x = torch.isfinite(x)
    total_x = int(x.numel())
    return {
        f"x_{prefix}_finite_ratio_after_preprocessing": float(finite_x.sum().item()) / total_x if total_x else 1.0,
        f"x_{prefix}_non_finite_count_after_preprocessing": int((~finite_x).sum().item()),
    }


def _is_finite(value: Any) -> bool:
    try:
        import math

        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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
