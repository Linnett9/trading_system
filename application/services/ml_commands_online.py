from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

from core.research.ml.online_learning import (
    FrozenLogisticModel,
    IncrementalLogisticModel,
    OnlineObservation,
    PeriodicRefitLogisticModel,
    PrequentialEvaluator,
    WarmStartNeuralModel,
)


def run_ml_online_intraday_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    ml = config.get("ml", {}) or {}
    online = ml.get("online_intraday", {}) or {}
    source_path = Path(str(online.get("dataset_path", "")))
    if not source_path.is_file() and bool(online.get("build_from_parquet", False)):
        _build_5m_observations_from_parquet(source_path, online)
    if not source_path.is_file():
        raise RuntimeError(
            "ml-online-intraday-benchmark requires ml.online_intraday.dataset_path"
        )
    observations, feature_columns = _read_online_observations(source_path, online)
    seed = int(ml.get("random_seed", 42))
    minimum = int(online.get("minimum_training_samples", 100))
    models = [
        FrozenLogisticModel(minimum_training_samples=minimum, random_seed=seed),
        IncrementalLogisticModel(
            random_seed=seed,
            alpha=float(online.get("online_logistic_alpha", 0.0001)),
            update_batch_size=int(online.get("online_update_batch_size", 12)),
        ),
        PeriodicRefitLogisticModel(
            refit_every=int(online.get("periodic_refit_every_bars", 78)),
            minimum_training_samples=minimum,
            random_seed=seed,
        ),
    ]
    if bool(online.get("include_warm_start_neural", True)):
        models.append(WarmStartNeuralModel(
            hidden_size=int(online.get("neural_hidden_size", 16)),
            learning_rate=float(online.get("neural_learning_rate", 0.001)),
            replay_size=int(online.get("neural_replay_size", 512)),
            replay_batch_size=int(online.get("neural_replay_batch_size", 64)),
            gradient_steps=int(online.get("neural_gradient_steps_per_bar", 1)),
            random_seed=seed,
        ))
    model_workers = min(
        max(1, int(online.get("model_workers", 1))),
        len(models),
    )
    with ThreadPoolExecutor(max_workers=model_workers) as executor:
        results = list(executor.map(
            lambda model: PrequentialEvaluator(
                model, threshold=float(online.get("decision_threshold", 0.5))
            ).run(observations),
            models,
        ))
    payload = {
        "mode": "online_intraday_prequential_benchmark",
        "source_path": str(source_path),
        "feature_columns": feature_columns,
        "observation_count": len(observations),
        "model_workers": model_workers,
        "first_observed_at": observations[0].observed_at.isoformat(),
        "last_observed_at": observations[-1].observed_at.isoformat(),
        "models": results,
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
    }
    output_path = Path(str(
        online.get(
            "output_path",
            "reports/ml/online_intraday/prequential_benchmark.json",
        )
    ))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Online intraday benchmark: {output_path}")
    for result in results:
        metrics = result["metrics"]
        print(
            f"{result['model']}: samples={metrics['samples']} | "
            f"balanced_accuracy={metrics.get('balanced_accuracy')} | "
            f"brier={result['brier_score']} | "
            f"leakage_safe={result['temporal_leakage_check_passed']}"
        )
    return payload


def _build_5m_observations_from_parquet(
    output_path: Path,
    config: dict[str, Any],
) -> None:
    import pyarrow.parquet as pq

    symbol = str(config.get("source_symbol", "SPY")).upper()
    source = (
        Path(str(config.get("processed_root", "data/processed")))
        / symbol / "5m" / "bars.parquet"
    )
    if not source.is_file():
        raise RuntimeError(f"Missing 5-minute source parquet: {source}")
    horizon = int(config.get("label_horizon_bars", 12))
    target_type = str(config.get("target_type", "future_return_below"))
    threshold = float(config.get("target_threshold", config.get("negative_return_threshold", 0.0)))
    rows = pq.read_table(
        source, columns=["timestamp", "open", "high", "low", "close", "volume"]
    ).to_pylist()
    context_symbols = [
        str(value).upper() for value in config.get("context_symbols", ["QQQ"])
        if str(value).upper() != symbol
    ]
    context_features = _context_features_by_timestamp(
        Path(str(config.get("processed_root", "data/processed"))),
        context_symbols,
    )
    closes = [float(row["close"]) for row in rows]
    volumes = [float(row["volume"]) for row in rows]
    session_bounds: dict[Any, tuple[int, int]] = {}
    for row in rows:
        timestamp = row["timestamp"]
        minute = timestamp.hour * 60 + timestamp.minute
        lower, upper = session_bounds.get(timestamp.date(), (minute, minute))
        session_bounds[timestamp.date()] = (min(lower, minute), max(upper, minute))
    records = []
    for index in range(78, len(rows) - horizon):
        current = rows[index]
        timestamp = current["timestamp"]
        future_index = index + horizon
        future_closes = closes[index + 1 : future_index + 1]
        future_returns = [value / closes[index] - 1.0 for value in future_closes]
        future_bar_returns = [
            closes[position] / closes[position - 1] - 1.0
            for position in range(index + 1, future_index + 1)
        ]
        target_values = {
            "future_return_below": closes[future_index] / closes[index] - 1.0,
            "future_drawdown_below": min(future_returns),
            "future_volatility_above": (
                sum((value - sum(future_bar_returns) / len(future_bar_returns)) ** 2
                    for value in future_bar_returns) / len(future_bar_returns)
            ) ** 0.5,
        }
        if target_type not in target_values:
            raise RuntimeError(f"Unsupported online target_type: {target_type}")
        returns_12 = [
            closes[position] / closes[position - 1] - 1.0
            for position in range(index - 11, index + 1)
            if closes[position - 1]
        ]
        volume_window = volumes[index - 77 : index + 1]
        volume_mean = sum(volume_window) / len(volume_window)
        volume_variance = sum(
            (value - volume_mean) ** 2 for value in volume_window
        ) / len(volume_window)
        minute = timestamp.hour * 60 + timestamp.minute
        session_start, session_end = session_bounds[timestamp.date()]
        session_fraction = (
            (minute - session_start) / (session_end - session_start)
            if session_end > session_start else 0.0
        )
        context = context_features.get(timestamp)
        if context_symbols and context is None:
            continue
        records.append({
            "observation_id": f"{symbol}_{timestamp.isoformat()}",
            "observed_at": timestamp.isoformat(),
            "label_available_at": rows[future_index]["timestamp"].isoformat(),
            "label": int(
                target_values[target_type] >= threshold
                if target_type == "future_volatility_above"
                else target_values[target_type] <= threshold
            ),
            "next_bar_return": closes[index + 1] / closes[index] - 1.0,
            "return_1_bar": closes[index] / closes[index - 1] - 1.0,
            "return_3_bars": closes[index] / closes[index - 3] - 1.0,
            "return_12_bars": closes[index] / closes[index - 12] - 1.0,
            "realized_volatility_12_bars": (
                sum((value - sum(returns_12) / len(returns_12)) ** 2 for value in returns_12)
                / len(returns_12)
            ) ** 0.5,
            "distance_sma_12_bars": closes[index]
            / (sum(closes[index - 11 : index + 1]) / 12.0) - 1.0,
            "distance_sma_78_bars": closes[index]
            / (sum(closes[index - 77 : index + 1]) / 78.0) - 1.0,
            "bar_range": (
                float(current["high"]) - float(current["low"])
            ) / closes[index],
            "volume_zscore_78_bars": (
                (volumes[index] - volume_mean) / math.sqrt(volume_variance)
                if volume_variance > 0 else 0.0
            ),
            "session_sin": math.sin(2.0 * math.pi * session_fraction),
            "session_cos": math.cos(2.0 * math.pi * session_fraction),
            **(context or {}),
        })
    if not records:
        raise RuntimeError(f"Insufficient 5-minute history in {source}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    audit_path = output_path.with_suffix(".audit.json")
    audit_path.write_text(json.dumps({
        "source_path": str(source),
        "symbol": symbol,
        "bar_timeframe": "5m",
        "row_count": len(records),
        "label_horizon_bars": horizon,
        "target_type": target_type,
        "target_threshold": threshold,
        "context_symbols": context_symbols,
        "first_observed_at": records[0]["observed_at"],
        "last_observed_at": records[-1]["observed_at"],
        "point_in_time_features_only": True,
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
    }, indent=2), encoding="utf-8")


def _read_online_observations(
    path: Path,
    config: dict[str, Any],
) -> tuple[list[OnlineObservation], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("Online intraday dataset is empty")
    required_metadata = {
        "observation_id", "observed_at", "label_available_at", "label"
    }
    reserved = required_metadata | {"next_bar_return"}
    feature_columns = [
        str(name) for name in config.get("feature_columns", [])
    ] or [name for name in rows[0] if name not in reserved]
    required = required_metadata | set(feature_columns)
    missing = sorted(required - set(rows[0]))
    if missing:
        raise RuntimeError("Online intraday dataset missing columns: " + ", ".join(missing))
    observations = [
        OnlineObservation(
            observation_id=str(row["observation_id"]),
            observed_at=datetime.fromisoformat(str(row["observed_at"]).replace("Z", "+00:00")),
            label_available_at=datetime.fromisoformat(
                str(row["label_available_at"]).replace("Z", "+00:00")
            ),
            features={name: float(row[name]) for name in feature_columns},
            label=int(row["label"]),
            next_bar_return=float(row.get("next_bar_return", 0.0) or 0.0),
        )
        for row in rows
    ]
    if any(item.label_available_at <= item.observed_at for item in observations):
        raise RuntimeError("Every online label must mature after its prediction timestamp")
    return sorted(observations, key=lambda item: item.observed_at), feature_columns


def _context_features_by_timestamp(
    processed_root: Path,
    symbols: list[str],
) -> dict[Any, dict[str, float]]:
    if not symbols:
        return {}
    import pyarrow.parquet as pq

    by_symbol: dict[str, dict[Any, tuple[float, float]]] = {}
    for symbol in symbols:
        path = processed_root / symbol / "5m" / "bars.parquet"
        if not path.is_file():
            raise RuntimeError(f"Missing context 5-minute parquet: {path}")
        rows = pq.read_table(path, columns=["timestamp", "close"]).to_pylist()
        closes = [float(row["close"]) for row in rows]
        by_symbol[symbol] = {
            rows[index]["timestamp"]: (
                closes[index] / closes[index - 1] - 1.0,
                closes[index] / closes[index - 12] - 1.0,
            )
            for index in range(12, len(rows))
        }
    common = set.intersection(*(set(values) for values in by_symbol.values()))
    return {
        timestamp: {
            **{
                f"{symbol.lower()}_return_1_bar": by_symbol[symbol][timestamp][0]
                for symbol in symbols
            },
            **{
                f"{symbol.lower()}_return_12_bars": by_symbol[symbol][timestamp][1]
                for symbol in symbols
            },
            "context_breadth_positive_1_bar": sum(
                by_symbol[symbol][timestamp][0] > 0 for symbol in symbols
            ) / len(symbols),
            "context_breadth_positive_12_bars": sum(
                by_symbol[symbol][timestamp][1] > 0 for symbol in symbols
            ) / len(symbols),
        }
        for timestamp in common
    }
