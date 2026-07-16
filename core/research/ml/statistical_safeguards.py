from __future__ import annotations

import itertools
import json
import math
import platform
from datetime import datetime, timezone
from hashlib import sha256
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np


RESULT_SCHEMA = "statistical_safeguard_result_v1"
MATCHED_SERIES_SCHEMA = "matched_statistical_series_v1"
BLOCK_POLICY_VERSION = "explicit_circular_block_length_v1"
STATUSES = {
    "VALID", "INSUFFICIENT_DATA", "INVALID_INPUT", "UNMATCHED_POPULATION",
    "UNSUPPORTED_CONFIGURATION", "NUMERICAL_FAILURE",
}


class SafeguardInputError(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_hash(payload: Any) -> str:
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest().upper()


def matched_series(
    observation_ids: Sequence[str],
    benchmark: Sequence[float],
    candidates: Mapping[str, Sequence[float]],
    *,
    orientation: str = "return",
    overlap_horizon: int = 1,
    seed_identities: Mapping[str, str | int] | None = None,
    fold_identity: str | None = None,
    panel_identity: str | None = None,
    minimum_observations: int = 2,
) -> dict[str, Any]:
    try:
        ids = tuple(str(value) for value in observation_ids)
        if orientation not in {"return", "loss"}:
            raise SafeguardInputError("INVALID_INPUT", "ORIENTATION_INVALID")
        if overlap_horizon < 1:
            raise SafeguardInputError("INVALID_INPUT", "OVERLAP_HORIZON_INVALID")
        if len(ids) != len(set(ids)):
            raise SafeguardInputError("UNMATCHED_POPULATION", "OBSERVATION_IDENTITIES_NOT_UNIQUE")
        if ids != tuple(sorted(ids)):
            raise SafeguardInputError("INVALID_INPUT", "OBSERVATIONS_NOT_DETERMINISTICALLY_ORDERED")
        if len(ids) < minimum_observations:
            raise SafeguardInputError("INSUFFICIENT_DATA", "OBSERVATION_COUNT_INSUFFICIENT")
        benchmark_values = _finite_vector(benchmark, "benchmark")
        if len(benchmark_values) != len(ids):
            raise SafeguardInputError("UNMATCHED_POPULATION", "BENCHMARK_POPULATION_MISMATCH")
        if not candidates:
            raise SafeguardInputError("INSUFFICIENT_DATA", "CANDIDATES_MISSING")
        candidate_values: dict[str, list[float]] = {}
        for candidate_id in sorted(candidates):
            values = _finite_vector(candidates[candidate_id], candidate_id)
            if len(values) != len(ids):
                raise SafeguardInputError("UNMATCHED_POPULATION", f"CANDIDATE_POPULATION_MISMATCH:{candidate_id}")
            candidate_values[str(candidate_id)] = values
        population_checksum = canonical_hash({"contract": MATCHED_SERIES_SCHEMA, "observation_ids": ids})
        return {
            "contract_schema": MATCHED_SERIES_SCHEMA,
            "observation_ids": list(ids),
            "benchmark": benchmark_values,
            "candidates": candidate_values,
            "seed_identities": dict(sorted((seed_identities or {}).items())),
            "fold_identity": fold_identity,
            "panel_identity": panel_identity,
            "orientation": orientation,
            "overlap_horizon": overlap_horizon,
            "observation_count": len(ids),
            "candidate_count": len(candidate_values),
            "population_checksum": population_checksum,
        }
    except SafeguardInputError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise SafeguardInputError("INVALID_INPUT", f"SERIES_VALIDATION_FAILED:{type(exc).__name__}") from exc


def circular_block_bootstrap(
    series: Mapping[str, Any],
    *,
    candidate_id: str,
    block_length: int,
    replications: int = 2000,
    random_seed: int = 0,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    method = "circular_block_bootstrap_paired_mean"
    params = {
        "block_length": block_length, "replications": replications,
        "confidence_level": confidence_level, "block_policy_version": BLOCK_POLICY_VERSION,
    }
    try:
        validated = _validated_payload(series, minimum_observations=3)
        if candidate_id not in validated["candidates"]:
            raise SafeguardInputError("INVALID_INPUT", "CANDIDATE_UNKNOWN")
        n = validated["observation_count"]
        _validate_bootstrap_parameters(n, block_length, replications, confidence_level)
        advantage = _advantage(validated, candidate_id)
        if not np.isfinite(advantage).all():
            raise SafeguardInputError("INVALID_INPUT", "NON_FINITE_ADVANTAGE")
        rng = np.random.default_rng(random_seed)
        indices = _circular_indices(n, block_length, replications, rng)
        boot = advantage[indices].mean(axis=1)
        observed = float(advantage.mean())
        alpha = 1.0 - confidence_level
        warnings = _block_warnings(block_length, validated["overlap_horizon"])
        metrics = {
            "candidate_id": candidate_id,
            "mean_difference": observed,
            "bootstrap_standard_error": float(np.std(boot, ddof=1)),
            "confidence_interval": [float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))],
            "one_sided_p_value": float((1 + np.sum(boot - observed >= observed)) / (replications + 1)),
            "two_sided_p_value": float((1 + np.sum(np.abs(boot - observed) >= abs(observed))) / (replications + 1)),
            "resampling": "paired circular blocks with wraparound",
            "iid_inference": False,
        }
        return _result(method, "1.0", validated, params, random_seed, metrics, warnings=warnings)
    except SafeguardInputError as exc:
        return _blocked_result(method, "1.0", series, params, random_seed, exc)
    except Exception as exc:  # pragma: no cover - fail-closed boundary
        return _blocked_result(method, "1.0", series, params, random_seed, SafeguardInputError("NUMERICAL_FAILURE", type(exc).__name__))


def deflated_sharpe_ratio(
    *,
    observed_sharpe: float,
    observation_count: int,
    skewness: float,
    kurtosis: float,
    effective_search_count: int | None,
    variance_convention: str = "bailey_lopez_de_prado_2014",
    annualisation_convention: str = "input_sharpe_already_annualised",
) -> dict[str, Any]:
    method = "deflated_sharpe_ratio"
    params = {
        "observed_sharpe": observed_sharpe, "observation_count": observation_count,
        "skewness": skewness, "kurtosis": kurtosis,
        "effective_search_count": effective_search_count,
        "variance_convention": variance_convention,
        "annualisation_convention": annualisation_convention,
    }
    base = _scalar_series_metadata(observation_count)
    try:
        values = (observed_sharpe, skewness, kurtosis)
        if not all(math.isfinite(float(value)) for value in values):
            raise SafeguardInputError("INVALID_INPUT", "NON_FINITE_DSR_INPUT")
        if observation_count < 3:
            raise SafeguardInputError("INSUFFICIENT_DATA", "DSR_OBSERVATION_COUNT_INSUFFICIENT")
        if effective_search_count is None:
            raise SafeguardInputError("INVALID_INPUT", "EFFECTIVE_SEARCH_COUNT_REQUIRED")
        if not isinstance(effective_search_count, int) or effective_search_count < 1:
            raise SafeguardInputError("INVALID_INPUT", "EFFECTIVE_SEARCH_COUNT_INVALID")
        if variance_convention != "bailey_lopez_de_prado_2014":
            raise SafeguardInputError("UNSUPPORTED_CONFIGURATION", "DSR_VARIANCE_CONVENTION_UNSUPPORTED")
        variance = (1 - skewness * observed_sharpe + ((kurtosis - 1) / 4) * observed_sharpe**2) / (observation_count - 1)
        if variance <= 0 or not math.isfinite(variance):
            raise SafeguardInputError("NUMERICAL_FAILURE", "DSR_VARIANCE_NON_POSITIVE")
        normal = NormalDist()
        gamma = 0.5772156649015329
        trials = float(effective_search_count)
        if effective_search_count == 1:
            expected_maximum = 0.0
        else:
            expected_maximum = math.sqrt(variance) * (
                (1 - gamma) * normal.inv_cdf(1 - 1 / trials)
                + gamma * normal.inv_cdf(1 - 1 / (trials * math.e))
            )
        z_score = (observed_sharpe - expected_maximum) / math.sqrt(variance)
        probability = normal.cdf(z_score)
        metrics = {
            "raw_sharpe": observed_sharpe,
            "expected_maximum_sharpe_under_search": expected_maximum,
            "sharpe_variance": variance,
            "deflated_sharpe_z_score": z_score,
            "deflated_sharpe_probability": probability,
            "effective_search_count": effective_search_count,
        }
        return _result(method, "bailey_lopez_de_prado_2014_v1", base, params, None, metrics)
    except SafeguardInputError as exc:
        return _blocked_result(method, "bailey_lopez_de_prado_2014_v1", base, params, None, exc)


def probability_of_backtest_overfitting(
    series: Mapping[str, Any],
    *,
    partition_budget: int = 100,
    random_seed: int = 0,
) -> dict[str, Any]:
    method = "combinatorially_symmetric_cross_validation_pbo"
    params = {"partition_budget": partition_budget, "selection_policy": "seeded_bounded_lexicographic_cscv"}
    try:
        validated = _validated_payload(series, minimum_observations=6)
        candidate_ids = sorted(validated["candidates"])
        if len(candidate_ids) < 2:
            raise SafeguardInputError("INSUFFICIENT_DATA", "PBO_REQUIRES_MULTIPLE_CANDIDATES")
        if partition_budget < 2:
            raise SafeguardInputError("UNSUPPORTED_CONFIGURATION", "PBO_PARTITION_BUDGET_TOO_SMALL")
        n = validated["observation_count"]
        train_size = n // 2
        full_count = math.comb(n, train_size)
        combinations = itertools.combinations(range(n), train_size)
        if full_count <= partition_budget:
            selected = list(combinations)
        else:
            rng = np.random.default_rng(random_seed)
            chosen = sorted(rng.choice(full_count, size=partition_budget, replace=False).tolist())
            selected, cursor = [], 0
            for index, combination in enumerate(combinations):
                if cursor < len(chosen) and index == chosen[cursor]:
                    selected.append(combination); cursor += 1
                if cursor == len(chosen):
                    break
        advantages = {candidate: _advantage(validated, candidate) for candidate in candidate_ids}
        rows = []
        for train_tuple in selected:
            train = np.asarray(train_tuple, dtype=int)
            test = np.asarray(sorted(set(range(n)) - set(train_tuple)), dtype=int)
            train_means = {candidate: float(advantages[candidate][train].mean()) for candidate in candidate_ids}
            winner = sorted(candidate_ids, key=lambda item: (-train_means[item], item))[0]
            test_means = {candidate: float(advantages[candidate][test].mean()) for candidate in candidate_ids}
            ordered = sorted(candidate_ids, key=lambda item: (test_means[item], item))
            rank = ordered.index(winner) + 1
            relative_rank = rank / (len(candidate_ids) + 1)
            logit = math.log(relative_rank / (1 - relative_rank))
            rows.append({
                "training_indexes": train.tolist(), "test_indexes": test.tolist(),
                "in_sample_winner": winner, "out_of_sample_rank_ascending": rank,
                "relative_rank": relative_rank, "logit": logit,
                "winner_oos_advantage": test_means[winner],
            })
        if len(rows) < 2:
            raise SafeguardInputError("INSUFFICIENT_DATA", "PBO_PARTITIONS_INSUFFICIENT")
        metrics = {
            "partition_count": len(rows), "full_partition_count": full_count,
            "bounded_selection_applied": full_count > partition_budget,
            "probability_of_backtest_overfitting": sum(row["logit"] <= 0 for row in rows) / len(rows),
            "mean_logit": float(np.mean([row["logit"] for row in rows])),
            "partitions": rows,
        }
        return _result(method, "cscv_bounded_v1", validated, params, random_seed, metrics)
    except SafeguardInputError as exc:
        return _blocked_result(method, "cscv_bounded_v1", series, params, random_seed, exc)


def superior_predictive_ability(
    series: Mapping[str, Any],
    *,
    block_length: int,
    replications: int = 2000,
    random_seed: int = 0,
) -> dict[str, Any]:
    method = "hansen_spa_studentized_consistent_recenter"
    params = {"block_length": block_length, "replications": replications, "block_policy_version": BLOCK_POLICY_VERSION}
    try:
        validated = _validated_payload(series, minimum_observations=8)
        n = validated["observation_count"]
        _validate_bootstrap_parameters(n, block_length, replications, 0.95)
        ids = sorted(validated["candidates"])
        matrix = np.vstack([_advantage(validated, candidate) for candidate in ids])
        std = matrix.std(axis=1, ddof=1)
        if np.all(std <= 1e-15):
            raise SafeguardInputError("INSUFFICIENT_DATA", "SPA_ALL_CANDIDATES_CONSTANT")
        valid = std > 1e-15
        warnings = [f"CONSTANT_CANDIDATE_EXCLUDED:{ids[index]}" for index in range(len(ids)) if not valid[index]]
        ids = [candidate for candidate, keep in zip(ids, valid) if keep]
        matrix, std = matrix[valid], std[valid]
        means = matrix.mean(axis=1)
        observed_t = np.sqrt(n) * means / std
        observed_max = float(np.max(observed_t))
        threshold = -std * math.sqrt(2 * math.log(max(math.log(n), 1.0000001))) / math.sqrt(n)
        retained_null = means >= threshold
        centered = matrix - np.where(retained_null, means, 0.0)[:, None]
        rng = np.random.default_rng(random_seed)
        indices = _circular_indices(n, block_length, replications, rng)
        boot_means = centered[:, indices].mean(axis=2)
        boot_t = np.sqrt(n) * boot_means / std[:, None]
        boot_max = np.max(boot_t, axis=0)
        p_value = float((1 + np.sum(boot_max >= observed_max)) / (replications + 1))
        metrics = {
            "candidate_statistics": {
                candidate: {"mean_advantage": float(mean), "studentized_statistic": float(stat)}
                for candidate, mean, stat in zip(ids, means, observed_t)
            },
            "observed_max_statistic": observed_max,
            "spa_p_value": p_value,
            "null_recentered_candidate_count": int(np.sum(retained_null)),
            "variant": "Hansen-style studentized SPA with consistent data-dependent recentering and circular blocks",
        }
        return _result(method, "1.0", validated, params, random_seed, metrics, warnings=warnings + _block_warnings(block_length, validated["overlap_horizon"]))
    except SafeguardInputError as exc:
        return _blocked_result(method, "1.0", series, params, random_seed, exc)


def model_confidence_set(
    series: Mapping[str, Any],
    *,
    block_length: int,
    replications: int = 2000,
    random_seed: int = 0,
    confidence_level: float = 0.90,
) -> dict[str, Any]:
    method = "model_confidence_set_range"
    params = {
        "block_length": block_length, "replications": replications,
        "confidence_level": confidence_level, "block_policy_version": BLOCK_POLICY_VERSION,
    }
    try:
        validated = _validated_payload(series, minimum_observations=8)
        n = validated["observation_count"]
        _validate_bootstrap_parameters(n, block_length, replications, confidence_level)
        if len(validated["candidates"]) < 2:
            raise SafeguardInputError("INSUFFICIENT_DATA", "MCS_REQUIRES_MULTIPLE_MODELS")
        losses = {
            candidate: np.asarray(values if validated["orientation"] == "loss" else [-value for value in values], dtype=float)
            for candidate, values in validated["candidates"].items()
        }
        active = sorted(losses)
        eliminated = []
        rng = np.random.default_rng(random_seed)
        round_number = 0
        while len(active) > 1:
            round_number += 1
            means = {model: float(losses[model].mean()) for model in active}
            observed_range = max(means.values()) - min(means.values())
            if observed_range <= 1e-15:
                break
            indices = _circular_indices(n, block_length, replications, rng)
            centered_means = []
            for model in active:
                centered = losses[model] - means[model]
                centered_means.append(centered[indices].mean(axis=1))
            boot = np.vstack(centered_means)
            boot_ranges = boot.max(axis=0) - boot.min(axis=0)
            p_value = float((1 + np.sum(boot_ranges >= observed_range)) / (replications + 1))
            if p_value >= 1 - confidence_level:
                break
            worst = sorted(active, key=lambda model: (-means[model], model))[0]
            eliminated.append({
                "model_id": worst, "elimination_round": round_number,
                "mean_loss": means[worst], "range_statistic": observed_range,
                "p_value": p_value,
            })
            active.remove(worst)
        metrics = {
            "retained_models": active, "eliminated_models": eliminated,
            "elimination_round_count": round_number,
            "variant": "range-based MCS using circular-block bootstrap",
        }
        return _result(method, "range_v1", validated, params, random_seed, metrics, warnings=_block_warnings(block_length, validated["overlap_horizon"]))
    except SafeguardInputError as exc:
        return _blocked_result(method, "range_v1", series, params, random_seed, exc)


def seed_dispersion(
    records: Sequence[Mapping[str, Any]],
    *,
    instability_cv_threshold: float = 0.25,
    instability_rank_correlation_threshold: float = 0.5,
) -> dict[str, Any]:
    method = "seed_dispersion_summary"
    params = {
        "instability_cv_threshold": instability_cv_threshold,
        "instability_rank_correlation_threshold": instability_rank_correlation_threshold,
    }
    try:
        if not records:
            raise SafeguardInputError("INSUFFICIENT_DATA", "SEED_RECORDS_MISSING")
        latest: dict[tuple[str, str], Mapping[str, Any]] = {}
        retry_count = 0
        for row in records:
            model = str(row.get("model_id") or "")
            seed = str(row.get("seed") if row.get("seed") is not None else "")
            value = float(row.get("value"))
            if not model or not seed or not math.isfinite(value):
                raise SafeguardInputError("INVALID_INPUT", "SEED_RECORD_INVALID")
            key = (model, seed)
            attempt = int(row.get("attempt", 1))
            if key in latest:
                retry_count += 1
                if attempt <= int(latest[key].get("attempt", 1)):
                    continue
            latest[key] = {**row, "value": value, "attempt": attempt}
        by_model: dict[str, list[Mapping[str, Any]]] = {}
        for (model, _), row in latest.items():
            by_model.setdefault(model, []).append(row)
        summaries = {}
        warnings = []
        for model, rows in sorted(by_model.items()):
            ordered = sorted(rows, key=lambda row: str(row["seed"]))
            values = np.asarray([float(row["value"]) for row in ordered])
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else None
            cv = abs(std / mean) if std is not None and abs(mean) > 1e-15 else None
            if len(values) == 1:
                warnings.append(f"SINGLE_SEED:{model}")
            if cv is not None and cv > instability_cv_threshold:
                warnings.append(f"SEED_INSTABILITY:{model}")
            summaries[model] = {
                "seed_count": len(values), "mean": mean, "median": float(np.median(values)),
                "standard_deviation": std, "minimum": float(values.min()), "maximum": float(values.max()),
                "range": float(values.max() - values.min()),
                "interquartile_range": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                "worst_seed": str(ordered[int(np.argmin(values))]["seed"]),
                "best_seed": str(ordered[int(np.argmax(values))]["seed"]),
                "coefficient_of_variation": cv,
            }
        rank_stability = _rank_stability(latest)
        if rank_stability is not None and rank_stability < instability_rank_correlation_threshold:
            warnings.append("CROSS_MODEL_RANK_INSTABILITY")
        metadata = {
            "observation_count": len(latest), "candidate_count": len(by_model),
            "population_checksum": canonical_hash(sorted(f"{model}:{seed}" for model, seed in latest)),
            "orientation": "return", "overlap_horizon": 1,
        }
        metrics = {
            "models": summaries, "rank_stability_mean_spearman": rank_stability,
            "distinct_seed_count": len({seed for _, seed in latest}),
            "retry_record_count_excluded": retry_count,
        }
        return _result(method, "1.0", metadata, params, None, metrics, warnings=warnings)
    except (SafeguardInputError, TypeError, ValueError) as exc:
        error = exc if isinstance(exc, SafeguardInputError) else SafeguardInputError("INVALID_INPUT", type(exc).__name__)
        return _blocked_result(method, "1.0", {}, params, None, error)


def _validated_payload(series: Mapping[str, Any], *, minimum_observations: int) -> dict[str, Any]:
    try:
        return matched_series(
            series["observation_ids"], series["benchmark"], series["candidates"],
            orientation=str(series.get("orientation", "return")),
            overlap_horizon=int(series.get("overlap_horizon", 1)),
            seed_identities=series.get("seed_identities"),
            fold_identity=series.get("fold_identity"),
            panel_identity=series.get("panel_identity"),
            minimum_observations=minimum_observations,
        )
    except KeyError as exc:
        raise SafeguardInputError("INVALID_INPUT", f"MATCHED_SERIES_FIELD_MISSING:{exc.args[0]}") from exc


def _finite_vector(values: Sequence[float], owner: str) -> list[float]:
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise SafeguardInputError("INVALID_INPUT", f"NON_FINITE_VALUE:{owner}")
    return result


def _advantage(series: Mapping[str, Any], candidate_id: str) -> np.ndarray:
    candidate = np.asarray(series["candidates"][candidate_id], dtype=float)
    benchmark = np.asarray(series["benchmark"], dtype=float)
    return candidate - benchmark if series["orientation"] == "return" else benchmark - candidate


def _validate_bootstrap_parameters(n: int, block_length: int, replications: int, confidence_level: float) -> None:
    if block_length < 1 or block_length > n:
        raise SafeguardInputError("UNSUPPORTED_CONFIGURATION", "BLOCK_LENGTH_INVALID")
    if replications < 100:
        raise SafeguardInputError("UNSUPPORTED_CONFIGURATION", "BOOTSTRAP_REPLICATIONS_TOO_SMALL")
    if not 0 < confidence_level < 1:
        raise SafeguardInputError("UNSUPPORTED_CONFIGURATION", "CONFIDENCE_LEVEL_INVALID")


def _circular_indices(n: int, block_length: int, replications: int, rng: np.random.Generator) -> np.ndarray:
    block_count = math.ceil(n / block_length)
    starts = rng.integers(0, n, size=(replications, block_count))
    offsets = np.arange(block_length)
    return ((starts[:, :, None] + offsets) % n).reshape(replications, -1)[:, :n]


def _block_warnings(block_length: int, overlap_horizon: int) -> list[str]:
    warnings = ["DEPENDENCY_AWARE_BLOCK_BOOTSTRAP", "BLOCK_LENGTH_MUST_BE_PRE_SPECIFIED"]
    if overlap_horizon > 1 and block_length < overlap_horizon:
        warnings.append("BLOCK_LENGTH_BELOW_DECLARED_OVERLAP_HORIZON")
    return warnings


def _result(method: str, version: str, series: Mapping[str, Any], params: Mapping[str, Any], random_seed: int | None, metrics: Mapping[str, Any], *, warnings: Sequence[str] = ()) -> dict[str, Any]:
    logical = {
        "contract_schema": RESULT_SCHEMA, "method_id": method, "method_version": version,
        "status": "VALID", "valid": True, "blocking_reasons": [], "warnings": sorted(set(warnings)),
        "observation_count": int(series.get("observation_count", 0)),
        "candidate_count": int(series.get("candidate_count", 0)),
        "population_checksum": series.get("population_checksum"),
        "parameter_checksum": canonical_hash(_identity_safe(params)), "deterministic_random_seed": random_seed,
        "input_orientation": series.get("orientation"), "target_overlap_horizon": series.get("overlap_horizon"),
        "result_metrics": _jsonable(metrics),
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _blocked_result(method: str, version: str, series: Mapping[str, Any], params: Mapping[str, Any], random_seed: int | None, error: SafeguardInputError) -> dict[str, Any]:
    status = error.status if error.status in STATUSES else "INVALID_INPUT"
    logical = {
        "contract_schema": RESULT_SCHEMA, "method_id": method, "method_version": version,
        "status": status, "valid": False, "blocking_reasons": [error.reason], "warnings": [],
        "observation_count": int(series.get("observation_count", len(series.get("observation_ids", ()))) if isinstance(series, Mapping) else 0),
        "candidate_count": int(series.get("candidate_count", len(series.get("candidates", {}))) if isinstance(series, Mapping) else 0),
        "population_checksum": series.get("population_checksum") if isinstance(series, Mapping) else None,
        "parameter_checksum": canonical_hash(_identity_safe(params)), "deterministic_random_seed": random_seed,
        "input_orientation": series.get("orientation") if isinstance(series, Mapping) else None,
        "target_overlap_horizon": series.get("overlap_horizon") if isinstance(series, Mapping) else None,
        "result_metrics": {},
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _scalar_series_metadata(observation_count: int) -> dict[str, Any]:
    return {
        "observation_count": observation_count, "candidate_count": 1,
        "population_checksum": canonical_hash({"scalar_observation_count": observation_count}),
        "orientation": "return", "overlap_horizon": 1,
    }


def _creation_metadata() -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _identity_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _identity_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_identity_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"non_finite_float": repr(value)}
    return value


def _rank_stability(latest: Mapping[tuple[str, str], Mapping[str, Any]]) -> float | None:
    seeds = sorted({seed for _, seed in latest})
    models = sorted({model for model, _ in latest})
    rankings = []
    for seed in seeds:
        if not all((model, seed) in latest for model in models):
            continue
        ordered = sorted(models, key=lambda model: (-float(latest[(model, seed)]["value"]), model))
        rankings.append({model: rank for rank, model in enumerate(ordered)})
    correlations = []
    for left, right in itertools.combinations(rankings, 2):
        x = np.asarray([left[model] for model in models], dtype=float)
        y = np.asarray([right[model] for model in models], dtype=float)
        if np.std(x) > 0 and np.std(y) > 0:
            correlations.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(correlations)) if correlations else None
