from __future__ import annotations

from typing import Any

from core.research.ml.data.sequence_dataset import sequence_group_ids_from_metadata
from core.research.ml.models import IMLModel
from core.research.ml.models.multitask_transformer_persistence import (
    MultiTaskTransformerPersistenceMixin,
)
from core.research.ml.models.multitask_transformer_prediction import (
    MultiTaskTransformerPredictionMixin,
)
from core.research.ml.models.multitask_transformer_training import (
    MultiTaskTransformerTrainingMixin,
)
from core.research.ml.models.multitask_transformer_types import (
    DEFAULT_REGRESSION_TARGETS,
    LEAKAGE_FEATURE_NAMES,
    LEAKAGE_FEATURE_PREFIXES,
    MultiTaskTransformerTrainingSummary,
)
from core.research.ml.models.multitask_transformer_network import (
    _make_multitask_transformer_module,
    _safe_feature_names,
)


class MultiTaskTransformerSequenceMLModel(
    MultiTaskTransformerTrainingMixin,
    MultiTaskTransformerPredictionMixin,
    MultiTaskTransformerPersistenceMixin,
    IMLModel,
):
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
