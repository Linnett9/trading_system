from __future__ import annotations

import math

from core.research.ml.data.sequence_dataset import build_sequence_indices
from core.research.ml.models.multitask_transformer_network import (
    _make_multitask_transformer_module,
    _safe_feature_names,
)
from core.research.ml.models.multitask_transformer_types import MultiTaskTransformerTrainingSummary
from core.research.ml.models.transformer_model import _torch_dependencies


class MultiTaskTransformerTrainingMixin:
    def fit(self, x_train: list[dict[str, float]], y_train: list[int]) -> None:
        self.fit_multitask(x_train, y_train, {})
    def fit_multitask(
        self,
        x_train: list[dict[str, float]],
        y_train: list[int],
        regression_targets: dict[str, list[float | None]],
    ) -> None:
        if len(x_train) != len(y_train):
            raise ValueError("Features and labels must have the same length")
        for target_name, values in regression_targets.items():
            if target_name not in self.regression_targets:
                raise ValueError(f"Unsupported regression target '{target_name}'")
            if len(values) != len(x_train):
                raise ValueError(
                    f"Regression target '{target_name}' must match feature length"
                )
        if not x_train:
            self.training_summary = MultiTaskTransformerTrainingSummary(
                False,
                0,
                0,
                0.5,
                list(self.regression_targets),
                {name: 0 for name in self.regression_targets},
            )
            return

        self.feature_names = _safe_feature_names(x_train[0])
        self.prior_probability = sum(int(value) for value in y_train) / len(y_train)
        self._fit_scaler(x_train)

        sequences, labels, regression_values, regression_mask = self._build_training_tensors(
            x_train,
            y_train,
            regression_targets,
        )
        self._fit_target_scalers(regression_values, regression_mask)
        normalized_regression_values = self._normalize_regression_targets(
            regression_values,
            regression_mask,
        )
        missing_counts = self._missing_target_counts(regression_mask)

        if not sequences or len(set(labels)) < 2:
            self.training_summary = MultiTaskTransformerTrainingSummary(
                False,
                len(sequences),
                len(self.feature_names),
                self.prior_probability,
                list(self.regression_targets),
                missing_counts,
            )
            return

        torch, nn, DataLoader, TensorDataset = _torch_dependencies()
        torch.manual_seed(self.random_seed)

        device = torch.device(self.device)
        x_tensor = torch.tensor(sequences, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
        regression_tensor = torch.tensor(
            normalized_regression_values,
            dtype=torch.float32,
            device=device,
        )
        mask_tensor = torch.tensor(regression_mask, dtype=torch.float32, device=device)
        dataset = TensorDataset(x_tensor, y_tensor, regression_tensor, mask_tensor)
        loader = DataLoader(
            dataset,
            batch_size=max(1, self.batch_size),
            shuffle=True,
            generator=torch.Generator(device="cpu").manual_seed(self.random_seed),
        )

        network_cls = _make_multitask_transformer_module()
        model = network_cls(
            feature_count=len(self.feature_names),
            sequence_length=self.sequence_length,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            regression_head_count=len(self.regression_targets),
        ).to(device)

        positive_count = sum(labels)
        negative_count = len(labels) - positive_count
        pos_weight = torch.tensor(
            [negative_count / max(positive_count, 1)],
            dtype=torch.float32,
            device=device,
        )
        classification_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        regression_loss_fn = (
            nn.SmoothL1Loss(reduction="none", beta=self.huber_delta)
            if self.regression_loss == "huber"
            else nn.MSELoss(reduction="none")
        )
        regression_weights = torch.tensor(
            [self.regression_weights[target] for target in self.regression_targets],
            dtype=torch.float32,
            device=device,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        model.train()
        for _ in range(max(1, self.epochs)):
            for batch_x, batch_y, batch_regression, batch_mask in loader:
                optimizer.zero_grad(set_to_none=True)
                classification_logits, regression_outputs = model(batch_x)
                classification_loss = classification_loss_fn(
                    classification_logits,
                    batch_y,
                )
                regression_loss = self._masked_regression_loss(
                    regression_loss_fn(
                        regression_outputs,
                        batch_regression,
                    ),
                    batch_mask,
                    regression_weights,
                )
                loss = self.classification_weight * classification_loss + regression_loss
                loss.backward()
                optimizer.step()

        self.model = model.cpu()
        self.training_summary = MultiTaskTransformerTrainingSummary(
            True,
            len(sequences),
            len(self.feature_names),
            self.prior_probability,
            list(self.regression_targets),
            missing_counts,
        )
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
    def _fit_target_scalers(
        self,
        regression_values: list[list[float]],
        regression_mask: list[list[float]],
    ) -> None:
        self.target_means = {}
        self.target_stds = {}
        for target_index, target in enumerate(self.regression_targets):
            values = [
                row[target_index]
                for row, mask in zip(regression_values, regression_mask)
                if mask[target_index] > 0.0
            ]
            if not values:
                self.target_means[target] = 0.0
                self.target_stds[target] = 1.0
                continue
            mean_value = sum(values) / len(values)
            variance = sum((value - mean_value) ** 2 for value in values) / len(values)
            std_value = variance ** 0.5
            self.target_means[target] = mean_value
            self.target_stds[target] = std_value if std_value > 1e-12 else 1.0
    def _build_training_tensors(
        self,
        rows: list[dict[str, float]],
        labels: list[int],
        regression_targets: dict[str, list[float | None]],
    ) -> tuple[list[list[list[float]]], list[int], list[list[float]], list[list[float]]]:
        matrix = [self._row_vector(row) for row in rows]
        sequences: list[list[list[float]]] = []
        targets: list[int] = []
        regression_values: list[list[float]] = []
        regression_mask: list[list[float]] = []
        for indices in build_sequence_indices(
            self._context_group_ids(len(matrix)),
            self.sequence_length,
        ):
            end_index = indices[-1]
            sequences.append([matrix[index] for index in indices])
            targets.append(int(labels[end_index]))
            values: list[float] = []
            mask: list[float] = []
            for target_name in self.regression_targets:
                raw_values = regression_targets.get(target_name, [])
                raw_value = raw_values[end_index] if raw_values else None
                if raw_value is None or not math.isfinite(float(raw_value)):
                    values.append(0.0)
                    mask.append(0.0)
                else:
                    values.append(float(raw_value))
                    mask.append(1.0)
            regression_values.append(values)
            regression_mask.append(mask)
        return sequences, targets, regression_values, regression_mask
    def _normalize_regression_targets(
        self,
        regression_values: list[list[float]],
        regression_mask: list[list[float]],
    ) -> list[list[float]]:
        normalized: list[list[float]] = []
        for values, mask in zip(regression_values, regression_mask):
            row: list[float] = []
            for target, value, mask_value in zip(self.regression_targets, values, mask):
                if mask_value <= 0.0:
                    row.append(0.0)
                    continue
                row.append(
                    (value - self.target_means.get(target, 0.0))
                    / self.target_stds.get(target, 1.0)
                )
            normalized.append(row)
        return normalized
    def _missing_target_counts(
        self,
        regression_mask: list[list[float]],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for target_index, target in enumerate(self.regression_targets):
            counts[target] = sum(
                1
                for row in regression_mask
                if target_index >= len(row) or row[target_index] <= 0.0
            )
        return counts
    @staticmethod
    def _masked_regression_loss(loss_values, mask, regression_weights):
        weighted = loss_values * mask * regression_weights
        denominator = (mask * regression_weights).sum().clamp_min(1.0)
        return weighted.sum() / denominator
