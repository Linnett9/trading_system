"""Deterministic, provider-free diagnostics for a materialized feature bank."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from core.research.market_data_cutover import FeatureFamily, classify_feature


def _finite(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(np.isfinite(numeric))


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def feature_quality_rows(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    timestamp_column: str = "decision_timestamp",
    asset_column: str = "asset_id",
) -> list[dict[str, Any]]:
    """Measure target-blind quality on an explicitly bounded audit frame."""

    if frame.empty:
        return []
    output: list[dict[str, Any]] = []
    grouped_assets = frame.groupby(asset_column, sort=False) if asset_column in frame else None
    grouped_timestamps = frame.groupby(timestamp_column, sort=False) if timestamp_column in frame else None
    for name in feature_names:
        if name not in frame:
            continue
        raw = pd.to_numeric(frame[name], errors="coerce")
        finite_mask = np.isfinite(raw)
        values = raw[finite_mask]
        missing_rate = float(raw.isna().mean())
        nonfinite_rate = float((~finite_mask & raw.notna()).mean())
        unique_count = int(values.nunique(dropna=True))
        zero_rate = float(values.eq(0).mean()) if len(values) else 1.0
        if len(values):
            quantiles = values.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
            iqr = float(quantiles.loc[0.75] - quantiles.loc[0.25])
            lower = float(quantiles.loc[0.25] - 3.0 * iqr)
            upper = float(quantiles.loc[0.75] + 3.0 * iqr)
            extreme_rate = float(((values < lower) | (values > upper)).mean()) if iqr > 0 else 0.0
            winsorized = values.clip(quantiles.loc[0.01], quantiles.loc[0.99])
            outlier_sensitivity = abs(float(values.mean()) - float(winsorized.mean())) / max(float(values.std(ddof=0)), 1e-12)
        else:
            quantiles = pd.Series({0.01: np.nan, 0.25: np.nan, 0.5: np.nan, 0.75: np.nan, 0.99: np.nan})
            extreme_rate = 0.0
            outlier_sensitivity = 0.0

        stale_numerator = 0
        stale_denominator = 0
        if grouped_assets is not None:
            for _, group in grouped_assets:
                sequence = pd.to_numeric(group[name], errors="coerce")
                valid_pair = sequence.notna() & sequence.shift().notna()
                stale_numerator += int((sequence.eq(sequence.shift()) & valid_pair).sum())
                stale_denominator += int(valid_pair.sum())

        cross_sectional_dispersion = None
        if grouped_timestamps is not None:
            dispersion = grouped_timestamps[name].std(ddof=0).replace([np.inf, -np.inf], np.nan).dropna()
            cross_sectional_dispersion = _safe_float(dispersion.mean())
        time_series_dispersion = None
        if grouped_assets is not None:
            dispersion = grouped_assets[name].std(ddof=0).replace([np.inf, -np.inf], np.nan).dropna()
            time_series_dispersion = _safe_float(dispersion.mean())

        monthly_drift = None
        if timestamp_column in frame and len(values):
            timestamp = pd.to_datetime(frame.loc[finite_mask, timestamp_column], utc=True, errors="coerce")
            monthly = pd.DataFrame({"month": timestamp.dt.strftime("%Y-%m"), "value": values.to_numpy()}).groupby("month")["value"].median()
            if len(monthly) > 1:
                monthly_drift = _safe_float((monthly.max() - monthly.min()) / max(float(values.std(ddof=0)), 1e-12))

        output.append(
            {
                "feature_name": name,
                "feature_family": classify_feature(name).value,
                "rows_audited": int(len(raw)),
                "missingness_rate": missing_rate,
                "nonfinite_rate": nonfinite_rate,
                "zero_rate": zero_rate,
                "constant_or_near_constant": unique_count <= 2 or (len(values) > 0 and zero_rate >= 0.999),
                "unique_value_count": unique_count,
                "mean": _safe_float(values.mean()) if len(values) else None,
                "standard_deviation": _safe_float(values.std(ddof=0)) if len(values) else None,
                "p01": _safe_float(quantiles.loc[0.01]),
                "median": _safe_float(quantiles.loc[0.5]),
                "p99": _safe_float(quantiles.loc[0.99]),
                "extreme_tail_rate_3iqr": extreme_rate,
                "outlier_sensitivity": outlier_sensitivity,
                "stale_adjacent_rate": stale_numerator / stale_denominator if stale_denominator else None,
                "mean_cross_sectional_dispersion": cross_sectional_dispersion,
                "mean_time_series_dispersion": time_series_dispersion,
                "max_monthly_median_drift_std_units": monthly_drift,
            }
        )
    return output


def redundancy_rows(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    maximum_rows: int = 100_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return high-correlation edges and target-blind effective dimension."""

    usable = [name for name in feature_names if name in frame and _finite(frame[name]).notna().any()]
    sample = frame.loc[:, usable]
    if len(sample) > maximum_rows:
        positions = np.linspace(0, len(sample) - 1, maximum_rows, dtype=np.int64)
        sample = sample.iloc[positions]
    numeric = sample.apply(_finite)
    correlation = numeric.corr(method="spearman", min_periods=max(50, len(numeric) // 20))
    edges: list[dict[str, Any]] = []
    counts = {"0.90": 0, "0.95": 0, "0.99": 0}
    valid_pairs = 0
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left_index, left in enumerate(usable):
        for right in usable[left_index + 1 :]:
            value = correlation.at[left, right]
            if not math.isfinite(value):
                continue
            valid_pairs += 1
            absolute = abs(float(value))
            for threshold in (0.90, 0.95, 0.99):
                if absolute > threshold:
                    counts[f"{threshold:.2f}"] += 1
            if absolute > 0.90:
                adjacency[left].add(right)
                adjacency[right].add(left)
                edges.append({"left_feature": left, "right_feature": right, "spearman": float(value), "absolute_spearman": absolute})

    # Connected components are a compact substitute for a persisted dense matrix.
    components: dict[str, int] = {}
    cluster = 0
    for feature in usable:
        if feature in components:
            continue
        cluster += 1
        stack = [feature]
        while stack:
            current = stack.pop()
            if current in components:
                continue
            components[current] = cluster
            stack.extend(adjacency[current] - components.keys())
    for row in edges:
        row["cluster_id"] = components[row["left_feature"]]

    complete = numeric.dropna(axis=1, how="all").fillna(numeric.median(numeric_only=True))
    effective_dimension = 0.0
    if complete.shape[1] and complete.shape[0]:
        standardized = (complete - complete.mean()) / complete.std(ddof=0).replace(0, 1)
        covariance = np.cov(standardized.to_numpy(dtype=float), rowvar=False)
        eigenvalues = np.atleast_1d(np.linalg.eigvalsh(np.atleast_2d(covariance)))
        eigenvalues = eigenvalues[eigenvalues > 0]
        if len(eigenvalues):
            effective_dimension = float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())
    summary = {
        "sample_rows": int(len(numeric)),
        "features": len(usable),
        "valid_pairs": valid_pairs,
        "effective_dimension": effective_dimension,
        "correlation_cluster_count": cluster,
        "pair_fraction_abs_spearman_gt_0_90": counts["0.90"] / valid_pairs if valid_pairs else None,
        "pair_fraction_abs_spearman_gt_0_95": counts["0.95"] / valid_pairs if valid_pairs else None,
        "pair_fraction_abs_spearman_gt_0_99": counts["0.99"] / valid_pairs if valid_pairs else None,
    }
    return sorted(edges, key=lambda item: (-item["absolute_spearman"], item["left_feature"], item["right_feature"])), summary


def rank_ic_by_period(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    *,
    period_column: str = "decision_timestamp",
) -> pd.Series:
    valid = frame[[period_column, feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return pd.Series(dtype=float)
    grouped = valid.groupby(period_column, sort=True, observed=True)
    x = grouped[feature].rank(method="average")
    y = grouped[target].rank(method="average")
    moments = pd.DataFrame(
        {
            period_column: valid[period_column].to_numpy(),
            "x": x.to_numpy(dtype=float),
            "y": y.to_numpy(dtype=float),
        }
    )
    moments["xx"] = moments["x"] * moments["x"]
    moments["yy"] = moments["y"] * moments["y"]
    moments["xy"] = moments["x"] * moments["y"]
    summary = moments.groupby(period_column, sort=True, observed=True).agg(
        n=("x", "size"),
        sum_x=("x", "sum"),
        sum_y=("y", "sum"),
        sum_xx=("xx", "sum"),
        sum_yy=("yy", "sum"),
        sum_xy=("xy", "sum"),
    )
    covariance = summary["sum_xy"] - summary["sum_x"] * summary["sum_y"] / summary["n"]
    variance_x = summary["sum_xx"] - summary["sum_x"].pow(2) / summary["n"]
    variance_y = summary["sum_yy"] - summary["sum_y"].pow(2) / summary["n"]
    denominator = np.sqrt(variance_x * variance_y)
    correlation = covariance / denominator.where(denominator > 0)
    return correlation.where(summary["n"] >= 10).dropna()


def univariate_signal_rows(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    target: str,
    horizon: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    sessions = pd.to_datetime(frame["decision_timestamp"], utc=True).dt.strftime("%Y-%m-%d")
    working = frame.assign(_session=sessions)
    for feature in feature_names:
        if feature not in working:
            continue
        period_ic = rank_ic_by_period(working, feature, target)
        session_ic = period_ic.groupby(pd.to_datetime(period_ic.index, utc=True).strftime("%Y-%m-%d")).mean() if len(period_ic) else pd.Series(dtype=float)
        blocks = np.array_split(session_ic.to_numpy(dtype=float), min(4, len(session_ic))) if len(session_ic) else []
        valid_rows = working[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
        missing_rows = working[working[feature].isna()]
        output.append(
            {
                "feature_name": feature,
                "feature_family": classify_feature(feature).value,
                "target_horizon": horizon,
                "status": "DIAGNOSTIC_REUSED_HISTORY",
                "period_count": int(len(period_ic)),
                "session_count": int(len(session_ic)),
                "mean_rank_ic": _safe_float(period_ic.mean()),
                "median_rank_ic": _safe_float(period_ic.median()),
                "positive_period_fraction": _safe_float(period_ic.gt(0).mean()),
                "sign_stability_session_fraction": _safe_float(session_ic.gt(0).mean()),
                "chronological_block_rank_ic": jsonable_float_list([np.mean(block) for block in blocks]),
                "breadth_rows": int(len(valid_rows)),
                "missingness_conditioned_rows_excluded": int(len(missing_rows)),
            }
        )
    return output


def jsonable_float_list(values: Sequence[Any]) -> str:
    return "[" + ",".join("null" if _safe_float(value) is None else format(float(value), ".12g") for value in values) + "]"


def monotonicity_rows(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    target: str,
    bins: int = 10,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for feature in feature_names:
        valid = frame[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < bins * 20 or valid[feature].nunique() < bins:
            output.append({"feature_name": feature, "status": "INSUFFICIENT_COVERAGE", "relationship": "unknown", "quantile_target_means": "[]"})
            continue
        quantile = pd.qcut(valid[feature], bins, labels=False, duplicates="drop")
        means = valid.assign(_quantile=quantile).groupby("_quantile", observed=True)[target].mean()
        differences = np.diff(means.to_numpy(dtype=float))
        positive = int((differences > 0).sum())
        negative = int((differences < 0).sum())
        if positive == len(differences):
            relationship = "monotonic_increasing"
        elif negative == len(differences):
            relationship = "monotonic_decreasing"
        elif len(means) >= 5 and means.iloc[len(means) // 2] < min(means.iloc[0], means.iloc[-1]):
            relationship = "U-shaped"
        elif len(means) >= 5 and means.iloc[len(means) // 2] > max(means.iloc[0], means.iloc[-1]):
            relationship = "inverted-U"
        elif max(positive, negative) >= max(1, len(differences) - 2):
            relationship = "threshold-like"
        else:
            relationship = "absent_or_unstable"
        output.append(
            {
                "feature_name": feature,
                "status": "DIAGNOSTIC_REUSED_HISTORY",
                "relationship": relationship,
                "quantile_count": int(len(means)),
                "quantile_target_means": jsonable_float_list(means.to_numpy()),
            }
        )
    return output


def _rank_ic_for_predictions(timestamps: pd.Series, target: np.ndarray, predictions: np.ndarray) -> float:
    scored = pd.DataFrame({"timestamp": timestamps.to_numpy(), "target": target, "prediction": predictions})
    values = rank_ic_by_period(scored, "prediction", "target", period_column="timestamp")
    return float(values.mean()) if len(values) else math.nan


def fit_diagnostic_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: Sequence[str],
    *,
    target: str,
    model_name: str,
) -> dict[str, Any]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    selected = [name for name in features if name in train and train[name].notna().any()]
    if not selected:
        return {"model": model_name, "status": "NO_USABLE_FEATURES", "feature_count": 0}
    estimators = {
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=2_000, random_state=20261025),
        "Huber": HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=300),
    }
    estimator = estimators[model_name]
    pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)
    started = time.perf_counter()
    pipeline.fit(train[selected], train[target])
    train_prediction = pipeline.predict(train[selected])
    validation_prediction = pipeline.predict(validation[selected])
    elapsed = time.perf_counter() - started
    train_ic = _rank_ic_for_predictions(train["decision_timestamp"], train[target].to_numpy(dtype=float), train_prediction)
    validation_ic = _rank_ic_for_predictions(validation["decision_timestamp"], validation[target].to_numpy(dtype=float), validation_prediction)
    convergence = bool(getattr(estimator, "n_iter_", 0) != getattr(estimator, "max_iter", -1))
    return {
        "model": model_name,
        "status": "COMPLETE",
        "feature_count": len(selected),
        "train_rank_ic": train_ic,
        "validation_rank_ic": validation_ic,
        "train_to_validation_degradation": train_ic - validation_ic,
        "converged": convergence,
        "fitting_seconds": elapsed,
    }


def model_comparison_rows(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    *,
    target: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for set_name, features in feature_sets.items():
        for model_name in ("Ridge", "ElasticNet", "Huber"):
            result = fit_diagnostic_model(train, validation, features, target=target, model_name=model_name)
            output.append({"feature_set": set_name, **result})
    return output


def fixed_feature_sets(feature_names: Sequence[str]) -> dict[str, list[str]]:
    by_family: dict[FeatureFamily, list[str]] = defaultdict(list)
    for name in feature_names:
        by_family[classify_feature(name)].append(name)
    returns = by_family[FeatureFamily.RETURNS_MOMENTUM]
    trend = by_family[FeatureFamily.TREND] + by_family[FeatureFamily.REVERSAL]
    volatility = by_family[FeatureFamily.VOLATILITY] + by_family[FeatureFamily.DOWNSIDE_RISK] + by_family[FeatureFamily.RANGE_LOCATION]
    relative = by_family[FeatureFamily.MARKET_RELATIVE] + by_family[FeatureFamily.BETA_CORRELATION]
    activity = by_family[FeatureFamily.VOLUME_ACTIVITY] + by_family[FeatureFamily.LIQUIDITY] + by_family[FeatureFamily.VWAP]
    ranks = by_family[FeatureFamily.CROSS_SECTIONAL_RANKS]
    context = by_family[FeatureFamily.TIME_OF_DAY] + by_family[FeatureFamily.BREADTH_STATE]
    cross = by_family[FeatureFamily.DAILY_CONTEXT] + by_family[FeatureFamily.HOURLY_CONTEXT] + by_family[FeatureFamily.CROSS_TIMEFRAME]

    def unique(values: Sequence[str]) -> list[str]:
        return list(dict.fromkeys(values))

    f0 = returns[:8]
    f1 = unique(f0 + trend[:8] + volatility[:8] + relative[:8])
    f2 = unique(f1 + activity[:10])
    f3 = unique(f2 + ranks[:8])
    f4 = unique(f3 + context[:10])
    f5 = unique(f4 + cross[:10])
    return {"F0": f0, "F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5, "FOLD": list(feature_names)}
