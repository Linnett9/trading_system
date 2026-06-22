from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MLExperimentConfig:
    model_type: str
    feature_set: str
    label_type: str
    train_start: str | None
    train_end: str | None
    test_start: str | None
    test_end: str | None
    prediction_horizon: int
    label_horizon_days: int
    drawdown_risk_threshold: float
    decision_threshold: float
    class_weight_balanced: bool
    test_fraction: float
    walk_forward_folds: int
    random_seed: int
    output_dir: str = "reports/ml"

    @classmethod
    def from_config(cls, config: dict) -> "MLExperimentConfig":
        ml_config = config.get("ml", {})
        return cls(
            model_type=str(ml_config.get("model_type", "logistic_regression")),
            feature_set=str(ml_config.get("feature_set", "price_regime_v1")),
            label_type=str(ml_config.get("label_type", "champion_success")),
            train_start=ml_config.get("train_start"),
            train_end=ml_config.get("train_end"),
            test_start=ml_config.get("test_start"),
            test_end=ml_config.get("test_end"),
            prediction_horizon=int(ml_config.get("prediction_horizon", 42)),
            label_horizon_days=int(
                ml_config.get(
                    "label_horizon_days",
                    ml_config.get("prediction_horizon", 42),
                )
            ),
            drawdown_risk_threshold=float(
                ml_config.get("drawdown_risk_threshold", 0.08)
            ),
            decision_threshold=float(ml_config.get("decision_threshold", 0.50)),
            class_weight_balanced=bool(ml_config.get("class_weight_balanced", True)),
            test_fraction=float(ml_config.get("test_fraction", 0.20)),
            walk_forward_folds=int(ml_config.get("walk_forward_folds", 3)),
            random_seed=int(ml_config.get("random_seed", 42)),
            output_dir=str(
                ml_config.get(
                    "output_dir",
                    config.get("reports", {}).get("ml_dir", "reports/ml"),
                )
            ),
        )

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "feature_set": self.feature_set,
            "label_type": self.label_type,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "prediction_horizon": self.prediction_horizon,
            "label_horizon_days": self.label_horizon_days,
            "drawdown_risk_threshold": self.drawdown_risk_threshold,
            "decision_threshold": self.decision_threshold,
            "class_weight_balanced": self.class_weight_balanced,
            "test_fraction": self.test_fraction,
            "walk_forward_folds": self.walk_forward_folds,
            "random_seed": self.random_seed,
            "output_dir": self.output_dir,
        }
