from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import pickle
from pathlib import Path
import random
from typing import Any, Protocol

from core.research.ml.evaluation import classification_metrics


@dataclass(frozen=True)
class OnlineObservation:
    """One point-in-time feature row whose label becomes known later."""

    observation_id: str
    observed_at: datetime
    label_available_at: datetime
    features: dict[str, float]
    label: int
    next_bar_return: float = 0.0


@dataclass(frozen=True)
class PrequentialPrediction:
    observation_id: str
    predicted_at: datetime
    label_available_at: datetime
    probability: float
    prediction: int
    label: int
    next_bar_return: float
    trained_through: datetime | None


class OnlineBinaryModel(Protocol):
    name: str
    trained_through: datetime | None

    def predict_probability(self, features: dict[str, float]) -> float: ...

    def update(self, observations: list[OnlineObservation]) -> None: ...


class IncrementalLogisticModel:
    """Incremental scaler plus SGD logistic regression using partial_fit."""

    name = "online_logistic"

    def __init__(
        self,
        *,
        random_seed: int = 42,
        alpha: float = 0.0001,
        update_batch_size: int = 12,
    ) -> None:
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler

        self.feature_names: list[str] = []
        self.scaler = StandardScaler()
        self.model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=float(alpha),
            random_state=random_seed,
            average=True,
        )
        self.update_batch_size = max(1, int(update_batch_size))
        self._update_buffer: list[OnlineObservation] = []
        self.fitted = False
        self.trained_through: datetime | None = None

    def predict_probability(self, features: dict[str, float]) -> float:
        if not self.fitted:
            return 0.5
        matrix = self.scaler.transform([self._row(features)])
        return float(self.model.predict_proba(matrix)[0, 1])

    def update(self, observations: list[OnlineObservation]) -> None:
        self._update_buffer.extend(observations)
        if len(self._update_buffer) < self.update_batch_size:
            return
        observations = self._update_buffer
        self._update_buffer = []
        if not self.feature_names:
            self.feature_names = sorted(observations[0].features)
        matrix = [self._row(item.features) for item in observations]
        labels = [int(item.label) for item in observations]
        self.scaler.partial_fit(matrix)
        scaled = self.scaler.transform(matrix)
        if self.fitted:
            self.model.partial_fit(scaled, labels)
        else:
            self.model.partial_fit(scaled, labels, classes=[0, 1])
            self.fitted = True
        self.trained_through = max(item.label_available_at for item in observations)

    def _row(self, features: dict[str, float]) -> list[float]:
        if set(features) != set(self.feature_names):
            raise ValueError("Online feature columns changed after model initialization")
        return [float(features[name]) for name in self.feature_names]


class FrozenLogisticModel(IncrementalLogisticModel):
    """Fit once on the first matured training batch, then remain frozen."""

    name = "frozen_logistic"

    def __init__(self, *, minimum_training_samples: int = 100, **kwargs: Any) -> None:
        kwargs.setdefault("update_batch_size", minimum_training_samples)
        super().__init__(**kwargs)
        self.minimum_training_samples = int(minimum_training_samples)
        self._pending: list[OnlineObservation] = []
        self._frozen = False

    def update(self, observations: list[OnlineObservation]) -> None:
        if self._frozen:
            return
        self._pending.extend(observations)
        if len(self._pending) >= self.minimum_training_samples:
            super().update(self._pending)
            self._pending = []
            self._frozen = True


class PeriodicRefitLogisticModel:
    """Full logistic refit control using only labels matured so far."""

    name = "periodic_refit_logistic"

    def __init__(
        self,
        *,
        refit_every: int = 78,
        minimum_training_samples: int = 100,
        random_seed: int = 42,
    ) -> None:
        self.refit_every = int(refit_every)
        self.minimum_training_samples = int(minimum_training_samples)
        self.random_seed = random_seed
        self.history: list[OnlineObservation] = []
        self.model: Any = None
        self.scaler: Any = None
        self.feature_names: list[str] = []
        self.trained_through: datetime | None = None
        self._last_refit_size = 0

    def predict_probability(self, features: dict[str, float]) -> float:
        if self.model is None:
            return 0.5
        row = [[float(features[name]) for name in self.feature_names]]
        return float(self.model.predict_proba(self.scaler.transform(row))[0, 1])

    def update(self, observations: list[OnlineObservation]) -> None:
        self.history.extend(observations)
        due = len(self.history) - self._last_refit_size >= self.refit_every
        if len(self.history) < self.minimum_training_samples or not due:
            return
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.feature_names = sorted(self.history[0].features)
        matrix = [
            [float(item.features[name]) for name in self.feature_names]
            for item in self.history
        ]
        labels = [item.label for item in self.history]
        if len(set(labels)) < 2:
            return
        self.scaler = StandardScaler().fit(matrix)
        self.model = LogisticRegression(
            max_iter=5_000,
            C=1.0,
            class_weight="balanced",
            random_state=self.random_seed,
        ).fit(self.scaler.transform(matrix), labels)
        self._last_refit_size = len(self.history)
        self.trained_through = max(item.label_available_at for item in self.history)


class WarmStartNeuralModel:
    """Small online MLP updated from matured labels plus a replay buffer."""

    name = "warm_start_neural"

    def __init__(
        self,
        *,
        hidden_size: int = 16,
        learning_rate: float = 0.001,
        replay_size: int = 512,
        replay_batch_size: int = 64,
        gradient_steps: int = 1,
        random_seed: int = 42,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Warm-start neural online learning requires torch") from exc
        torch.manual_seed(random_seed)
        self._torch = torch
        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)
        self.replay_size = int(replay_size)
        self.replay_batch_size = int(replay_batch_size)
        self.gradient_steps = int(gradient_steps)
        self.random = random.Random(random_seed)
        self.feature_names: list[str] = []
        self.model: Any = None
        self.optimizer: Any = None
        self.replay: list[OnlineObservation] = []
        self.trained_through: datetime | None = None

    def predict_probability(self, features: dict[str, float]) -> float:
        if self.model is None:
            return 0.5
        torch = self._torch
        with torch.no_grad():
            logits = self.model(torch.tensor([self._row(features)], dtype=torch.float32))
            return float(torch.sigmoid(logits).item())

    def update(self, observations: list[OnlineObservation]) -> None:
        if not observations:
            return
        if not self.feature_names:
            self.feature_names = sorted(observations[0].features)
            torch = self._torch
            self.model = torch.nn.Sequential(
                torch.nn.Linear(len(self.feature_names), self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_size, 1),
            )
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            )
        self.replay.extend(observations)
        self.replay = self.replay[-self.replay_size :]
        torch = self._torch
        loss_function = torch.nn.BCEWithLogitsLoss()
        for _ in range(self.gradient_steps):
            batch = self.random.sample(
                self.replay, min(self.replay_batch_size, len(self.replay))
            )
            features = torch.tensor(
                [self._row(item.features) for item in batch], dtype=torch.float32
            )
            labels = torch.tensor(
                [[float(item.label)] for item in batch], dtype=torch.float32
            )
            self.optimizer.zero_grad()
            loss = loss_function(self.model(features), labels)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite online neural loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        self.trained_through = max(item.label_available_at for item in observations)

    def _row(self, features: dict[str, float]) -> list[float]:
        return [float(features[name]) for name in self.feature_names]


class PrequentialEvaluator:
    """Predict first, then train only when each earlier label matures."""

    def __init__(self, model: OnlineBinaryModel, *, threshold: float = 0.5) -> None:
        self.model = model
        self.threshold = float(threshold)

    def run(self, observations: list[OnlineObservation]) -> dict[str, Any]:
        ordered = sorted(observations, key=lambda item: item.observed_at)
        pending: list[OnlineObservation] = []
        evaluated: list[PrequentialPrediction] = []
        updates = 0
        for observation in ordered:
            matured = [
                item for item in pending
                if item.label_available_at <= observation.observed_at
            ]
            if matured:
                self.model.update(matured)
                updates += 1
                pending = [item for item in pending if item not in matured]
            probability = min(
                1.0, max(0.0, float(self.model.predict_probability(observation.features)))
            )
            evaluated.append(PrequentialPrediction(
                observation_id=observation.observation_id,
                predicted_at=observation.observed_at,
                label_available_at=observation.label_available_at,
                probability=probability,
                prediction=int(probability >= self.threshold),
                label=observation.label,
                next_bar_return=observation.next_bar_return,
                trained_through=self.model.trained_through,
            ))
            pending.append(observation)
        scored = [
            row for row in evaluated
            if row.label_available_at <= ordered[-1].observed_at
        ] if ordered else []
        actual = [row.label for row in scored]
        predicted = [row.prediction for row in scored]
        probabilities = [row.probability for row in scored]
        temporal_violations = [
            row.observation_id for row in evaluated
            if row.trained_through is not None and row.trained_through > row.predicted_at
        ]
        result = {
            "model": self.model.name,
            "observation_count": len(ordered),
            "scored_prediction_count": len(scored),
            "pending_label_count": len(ordered) - len(scored),
            "update_count": updates,
            "metrics": classification_metrics(actual, predicted),
            "brier_score": (
                sum((probability - label) ** 2 for probability, label in zip(probabilities, actual))
                / len(actual)
                if actual else None
            ),
            "temporal_leakage_check_passed": not temporal_violations,
            "temporal_leakage_violations": temporal_violations,
            "predictions": [row.__dict__ for row in evaluated],
            "research_only": True,
            "trading_impact": "none",
            "production_validated": False,
        }
        result["trading_simulation"] = self.trading_simulation(scored)
        return result

    @staticmethod
    def trading_simulation(
        predictions: list[PrequentialPrediction],
        *,
        reduced_exposure: float = 0.5,
        transaction_cost_bps: float = 1.0,
    ) -> dict[str, float | int | None]:
        equity = 1.0
        peak = 1.0
        max_drawdown = 0.0
        previous_exposure = 1.0
        net_returns = []
        turnover = 0.0
        reduced_bars = 0
        for row in predictions:
            exposure = reduced_exposure if row.prediction else 1.0
            reduced_bars += int(row.prediction == 1)
            change = abs(exposure - previous_exposure)
            turnover += change
            cost = change * transaction_cost_bps / 10_000.0
            net_return = exposure * row.next_bar_return - cost
            net_returns.append(net_return)
            equity *= 1.0 + net_return
            peak = max(peak, equity)
            max_drawdown = min(max_drawdown, equity / peak - 1.0)
            previous_exposure = exposure
        mean_return = sum(net_returns) / len(net_returns) if net_returns else 0.0
        variance = (
            sum((value - mean_return) ** 2 for value in net_returns) / len(net_returns)
            if net_returns else 0.0
        )
        baseline = PrequentialEvaluator._equity_statistics(
            [row.next_bar_return for row in predictions]
        )
        return {
            "net_total_return": equity - 1.0,
            "max_drawdown": max_drawdown,
            "annualized_sharpe": (
                mean_return / math.sqrt(variance) * math.sqrt(78 * 252)
                if variance > 0 else None
            ),
            "turnover": turnover,
            "reduced_exposure_bars": reduced_bars,
            "transaction_cost_bps": transaction_cost_bps,
            "baseline_total_return": baseline["total_return"],
            "baseline_max_drawdown": baseline["max_drawdown"],
            "baseline_annualized_sharpe": baseline["annualized_sharpe"],
            "return_delta": equity - 1.0 - float(baseline["total_return"]),
            "max_drawdown_improvement": max_drawdown - float(baseline["max_drawdown"]),
        }

    @staticmethod
    def _equity_statistics(returns: list[float]) -> dict[str, float | None]:
        equity = 1.0
        peak = 1.0
        max_drawdown = 0.0
        for value in returns:
            equity *= 1.0 + value
            peak = max(peak, equity)
            max_drawdown = min(max_drawdown, equity / peak - 1.0)
        average = sum(returns) / len(returns) if returns else 0.0
        variance = (
            sum((value - average) ** 2 for value in returns) / len(returns)
            if returns else 0.0
        )
        return {
            "total_return": equity - 1.0,
            "max_drawdown": max_drawdown,
            "annualized_sharpe": (
                average / math.sqrt(variance) * math.sqrt(78 * 252)
                if variance > 0 else None
            ),
        }


def save_online_checkpoint(path: Path, model: OnlineBinaryModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle)


def load_online_checkpoint(path: Path) -> OnlineBinaryModel:
    with path.open("rb") as handle:
        return pickle.load(handle)
