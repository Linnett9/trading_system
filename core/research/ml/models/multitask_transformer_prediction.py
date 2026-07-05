from __future__ import annotations

import math

from core.research.ml.data.sequence_dataset import build_sequence_indices
from core.research.ml.models.transformer_model import _torch_dependencies


class MultiTaskTransformerPredictionMixin:
    def predict(self, x: list[dict[str, float]]) -> list[int]:
        return [int(probability >= 0.5) for probability in self.predict_proba(x)]
    def predict_proba(self, x: list[dict[str, float]]) -> list[float]:
        return [
            row["probability_should_reduce_exposure"]
            for row in self.predict_multitask(x)
        ]
    def predict_multitask(self, x: list[dict[str, float]]) -> list[dict[str, float]]:
        if not x:
            return []

        predictions = [
            self._prior_prediction_row()
            for _ in x
        ]
        if self.model is None or not self.training_summary.trained:
            return predictions

        sequences, end_indices = self._build_prediction_sequences(x)
        if not sequences:
            return predictions

        torch, _, _, _ = _torch_dependencies()
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(sequences, dtype=torch.float32)
            classification_logits, regression_outputs = self.model(x_tensor)
            probabilities = torch.sigmoid(classification_logits).cpu().tolist()
            regression_rows = regression_outputs.cpu().tolist()

        for index, probability, regression_row in zip(
            end_indices,
            probabilities,
            regression_rows,
        ):
            row = {
                "probability_should_reduce_exposure": float(
                    max(0.0, min(1.0, probability))
                )
            }
            for target, normalized_value in zip(self.regression_targets, regression_row):
                value = (
                    float(normalized_value) * self.target_stds.get(target, 1.0)
                    + self.target_means.get(target, 0.0)
                )
                row[f"predicted_{target}"] = value if math.isfinite(value) else 0.0
            predictions[index] = row
        return predictions
    def feature_importances(self) -> dict[str, float]:
        return {}
    def _row_vector(self, row: dict[str, float]) -> list[float]:
        return [
            (float(row.get(name, 0.0) or 0.0) - self.means.get(name, 0.0))
            / self.stds.get(name, 1.0)
            for name in self.feature_names
        ]
    def _build_prediction_sequences(
        self,
        rows: list[dict[str, float]],
    ) -> tuple[list[list[list[float]]], list[int]]:
        matrix = [self._row_vector(row) for row in rows]
        sequences: list[list[list[float]]] = []
        end_indices: list[int] = []
        for indices in build_sequence_indices(
            self._context_group_ids(len(matrix)),
            self.sequence_length,
        ):
            sequences.append([matrix[index] for index in indices])
            end_indices.append(indices[-1])
        return sequences, end_indices
    def _context_group_ids(self, sample_count: int) -> list[str]:
        if len(self._sequence_group_ids) == sample_count:
            return list(self._sequence_group_ids)
        return ["global" for _ in range(sample_count)]
    def _prior_prediction_row(self) -> dict[str, float]:
        row = {"probability_should_reduce_exposure": float(self.prior_probability)}
        for target in self.regression_targets:
            row[f"predicted_{target}"] = float(self.target_means.get(target, 0.0))
        return row
