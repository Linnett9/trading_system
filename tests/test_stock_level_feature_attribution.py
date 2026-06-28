from __future__ import annotations

import inspect

from core.research.ml import stock_level_feature_attribution
from core.research.ml.stock_level_feature_attribution import (
    build_stock_level_feature_attribution,
)


class SyntheticLinearRegressor:
    def fit(self, features, targets):
        target_mean = sum(targets) / len(targets)
        self.intercept_ = target_mean
        self.coef_ = []
        best_index = 0
        best_score = -1.0
        for index, column in enumerate(zip(*features)):
            column_mean = sum(column) / len(column)
            covariance = sum(
                (value - column_mean) * (target - target_mean)
                for value, target in zip(column, targets)
            )
            variance = sum((value - column_mean) ** 2 for value in column)
            slope = covariance / variance if variance > 0.0 else 0.0
            self.coef_.append(slope)
            score = abs(covariance)
            if score > best_score:
                best_score = score
                best_index = index
        self.best_index = best_index
        self.best_slope = self.coef_[best_index]
        selected = [row[best_index] for row in features]
        self.selected_mean = sum(selected) / len(selected)

    def predict(self, features):
        return [
            self.intercept_ + self.best_slope * (row[self.best_index] - self.selected_mean)
            for row in features
        ]


def test_ridge_coefficient_attribution_works():
    payload = _build()
    rows = {row["feature"]: row for row in payload["feature_rows"]}

    assert rows["predicted_momentum_20d"]["coefficient_mean"] > 0.0
    useful_magnitude = rows["predicted_momentum_20d"][
        "normalized_coefficient_or_importance_magnitude"
    ]
    assert useful_magnitude > 0.4
    assert useful_magnitude == max(
        row["normalized_coefficient_or_importance_magnitude"]
        for row in rows.values()
    )
    assert rows["predicted_momentum_20d"]["attribution_method"] == "coefficient"


def test_feature_ablation_is_chronological_and_oos_only():
    payload = _build()
    walk_forward = payload["walk_forward"]

    assert walk_forward["out_of_sample_only"] is True
    assert walk_forward["all_chronological_guards_passed"] is True
    assert all(
        fold["train_end_date"] < fold["test_start_date"]
        for fold in walk_forward["folds"]
    )
    assert all(
        all(fold["train_end_date"] < date for date in fold["embargoed_dates"])
        for fold in walk_forward["folds"]
    )


def test_removing_useful_feature_lowers_ranking_metrics():
    payload = _build()
    useful = next(
        row
        for row in payload["feature_rows"]
        if row["feature"] == "predicted_momentum_20d"
    )

    assert useful["ablated_mean_spearman_ic"] < useful["full_mean_spearman_ic"]
    assert useful["ablation_delta_mean_spearman_ic"] < 0.0
    assert useful["ablation_delta_top_minus_bottom_spread"] < 0.0


def test_feature_attribution_has_no_operational_imports():
    source = inspect.getsource(stock_level_feature_attribution)

    assert "infrastructure.broker" not in source
    assert "paper_trading" not in source
    assert "paper_commands" not in source
    assert "live_trading" not in source
    assert "core.entities.order" not in source
    assert "order_execution" not in source


def _build():
    return build_stock_level_feature_attribution(
        _rows(),
        _benchmark(),
        existing_prediction_rows=[
            {
                "rebalance_date": "2024-01-04",
                "symbol": "S00",
                "stock_level_predicted_forward_return_10d_ridge": "0.0",
            }
        ],
        permutation_repeats=1,
        model_factories={"ridge": SyntheticLinearRegressor},
    )


def _benchmark() -> dict:
    return {
        "completed_models": ["ridge"],
        "walk_forward": {
            "min_train_dates": 2,
            "test_window_dates": 2,
            "embargo_rebalance_dates": 1,
        },
        "leaderboard": [
            {
                "name": "ridge",
                "mean_spearman_ic": 1.0,
                "top_minus_bottom_spread": 0.9,
                "top_decile_return": 0.4,
                "bottom_decile_return": -0.5,
                "top_decile_hit_rate": 1.0,
                "risk_adjusted_spread": 4.5,
                "spread_sharpe": 3.0,
            }
        ],
    }


def _rows() -> list[dict[str, str]]:
    rows = []
    for date_index in range(8):
        date = f"2024-01-{date_index + 1:02d}"
        for symbol_index in range(10):
            useful = float(symbol_index - 5) / 10.0
            noise = [
                ((symbol_index * multiplier + date_index) % 10 - 5) / 10.0
                for multiplier in (3, 7, 9, 2, 4, 6)
            ]
            rows.append(
                {
                    "rebalance_date": date,
                    "symbol": f"S{symbol_index:02d}",
                    "predicted_momentum_20d": str(useful),
                    "predicted_momentum_60d": str(noise[0]),
                    "predicted_momentum_120d": str(noise[1]),
                    "predicted_volatility_20d": str(noise[2]),
                    "predicted_drawdown_60d": str(noise[3]),
                    "predicted_liquidity_score": str(noise[4]),
                    "predicted_risk_adjusted_momentum": str(noise[5]),
                    "actual_forward_return_5d": str(useful * 0.5),
                    "actual_forward_return_10d": str(useful),
                    "actual_future_volatility": "0.1",
                    "actual_future_drawdown": "-0.2",
                }
            )
    return rows
