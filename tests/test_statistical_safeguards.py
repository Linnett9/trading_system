from __future__ import annotations

import json
import math

import numpy as np
import pytest

from core.research.ml.statistical_safeguards import (
    SafeguardInputError,
    canonical_json,
    circular_block_bootstrap,
    deflated_sharpe_ratio,
    matched_series,
    model_confidence_set,
    probability_of_backtest_overfitting,
    seed_dispersion,
    superior_predictive_ability,
)


def _series(candidates=None, *, n=40, orientation="return", overlap=10):
    ids = [f"2024-01-{index + 1:02d}" for index in range(n)]
    benchmark = np.zeros(n)
    values = candidates or {"candidate": np.sin(np.arange(n) / 3) * 0.01}
    return matched_series(
        ids, benchmark, values, orientation=orientation,
        overlap_horizon=overlap, minimum_observations=2,
    )


def _logical(result):
    return {key: value for key, value in result.items() if key != "creation_metadata"}


def test_matched_contract_is_deterministic_and_fail_closed():
    first = _series()
    second = _series()
    assert first == second
    assert first["population_checksum"] == second["population_checksum"]
    with pytest.raises(SafeguardInputError, match="MISMATCH"):
        matched_series(["a", "b"], [0], {"x": [0, 1]})
    with pytest.raises(SafeguardInputError, match="NON_FINITE"):
        matched_series(["a", "b"], [0, math.nan], {"x": [0, 1]})
    with pytest.raises(SafeguardInputError, match="UNIQUE"):
        matched_series(["a", "a"], [0, 0], {"x": [0, 1]})


def test_stable_json_order_and_logical_identity():
    result = circular_block_bootstrap(_series(), candidate_id="candidate", block_length=10, replications=200, random_seed=7)
    assert result["valid"]
    encoded = canonical_json(_logical(result))
    assert encoded == canonical_json(json.loads(encoded))
    again = circular_block_bootstrap(_series(), candidate_id="candidate", block_length=10, replications=200, random_seed=7)
    assert _logical(result) == _logical(again)


def test_block_bootstrap_null_superior_inferior_and_reproducibility():
    rng = np.random.default_rng(4)
    noise = rng.normal(0, 0.01, 60)
    series = _series({"null": noise, "superior": noise + 0.02, "inferior": noise - 0.02}, n=60)
    null = circular_block_bootstrap(series, candidate_id="null", block_length=10, replications=500, random_seed=8)
    superior = circular_block_bootstrap(series, candidate_id="superior", block_length=10, replications=500, random_seed=8)
    inferior = circular_block_bootstrap(series, candidate_id="inferior", block_length=10, replications=500, random_seed=8)
    assert abs(null["result_metrics"]["mean_difference"]) < 0.01
    assert superior["result_metrics"]["mean_difference"] > 0
    assert inferior["result_metrics"]["mean_difference"] < 0
    assert _logical(superior) == _logical(circular_block_bootstrap(series, candidate_id="superior", block_length=10, replications=500, random_seed=8))


def test_block_bootstrap_preserves_local_dependence_and_warns_on_short_blocks():
    values = np.repeat([0.0, 1.0, -1.0, 0.5], 10)
    result = circular_block_bootstrap(_series({"candidate": values}), candidate_id="candidate", block_length=4, replications=300, random_seed=2)
    iid_like = circular_block_bootstrap(_series({"candidate": values}), candidate_id="candidate", block_length=1, replications=300, random_seed=2)
    assert result["result_metrics"]["bootstrap_standard_error"] != iid_like["result_metrics"]["bootstrap_standard_error"]
    assert "BLOCK_LENGTH_BELOW_DECLARED_OVERLAP_HORIZON" in result["warnings"]
    assert result["result_metrics"]["resampling"].startswith("paired circular")


@pytest.mark.parametrize("block_length", [0, 41])
def test_invalid_block_length_is_rejected(block_length):
    result = circular_block_bootstrap(_series(), candidate_id="candidate", block_length=block_length, replications=200)
    assert result["status"] == "UNSUPPORTED_CONFIGURATION"
    assert not result["valid"]


def test_insufficient_observations_are_explicit():
    series = matched_series(["a", "b"], [0, 0], {"x": [0, 0]}, minimum_observations=2)
    result = circular_block_bootstrap(series, candidate_id="x", block_length=1, replications=200)
    assert result["status"] == "INSUFFICIENT_DATA"


def test_dsr_requires_search_count_and_becomes_more_conservative():
    one = deflated_sharpe_ratio(observed_sharpe=0.2, observation_count=252, skewness=0, kurtosis=3, effective_search_count=1)
    many = deflated_sharpe_ratio(observed_sharpe=0.2, observation_count=252, skewness=0, kurtosis=3, effective_search_count=100)
    missing = deflated_sharpe_ratio(observed_sharpe=0.2, observation_count=252, skewness=0, kurtosis=3, effective_search_count=None)
    assert one["valid"] and many["valid"]
    assert many["result_metrics"]["expected_maximum_sharpe_under_search"] > one["result_metrics"]["expected_maximum_sharpe_under_search"]
    assert many["result_metrics"]["deflated_sharpe_probability"] < one["result_metrics"]["deflated_sharpe_probability"]
    assert missing["status"] == "INVALID_INPUT"


def test_dsr_rejects_nonfinite_and_insufficient_inputs():
    assert deflated_sharpe_ratio(observed_sharpe=math.inf, observation_count=10, skewness=0, kurtosis=3, effective_search_count=2)["status"] == "INVALID_INPUT"
    assert deflated_sharpe_ratio(observed_sharpe=1, observation_count=2, skewness=0, kurtosis=3, effective_search_count=2)["status"] == "INSUFFICIENT_DATA"


def test_pbo_partitions_are_deterministic_and_bounded():
    rng = np.random.default_rng(5)
    candidates = {f"m{index}": rng.normal(index * 0.001, 0.02, 12) for index in range(4)}
    series = _series(candidates, n=12, overlap=1)
    first = probability_of_backtest_overfitting(series, partition_budget=20, random_seed=12)
    second = probability_of_backtest_overfitting(series, partition_budget=20, random_seed=12)
    assert first["valid"] and first["result_metrics"]["partition_count"] == 20
    assert first["result_metrics"]["bounded_selection_applied"]
    assert _logical(first) == _logical(second)


def test_pbo_rejects_one_candidate_and_detects_deliberate_overfit():
    one = probability_of_backtest_overfitting(_series({"only": np.arange(12)}, n=12), partition_budget=20)
    assert one["status"] == "INSUFFICIENT_DATA"
    stable = _series({"good": np.ones(12), "bad": np.zeros(12)}, n=12, overlap=1)
    overfit = _series({
        "first_half": np.r_[np.ones(6), -np.ones(6)],
        "second_half": np.r_[-np.ones(6), np.ones(6)],
        "flat": np.zeros(12),
    }, n=12, overlap=1)
    stable_result = probability_of_backtest_overfitting(stable, partition_budget=100, random_seed=1)
    overfit_result = probability_of_backtest_overfitting(overfit, partition_budget=100, random_seed=1)
    assert overfit_result["result_metrics"]["probability_of_backtest_overfitting"] >= stable_result["result_metrics"]["probability_of_backtest_overfitting"]


def test_spa_null_and_positive_signal():
    rng = np.random.default_rng(9)
    null_noise = rng.normal(0, 0.02, 60)
    null = superior_predictive_ability(_series({"a": null_noise, "b": -null_noise}, n=60), block_length=10, replications=500, random_seed=3)
    positive = superior_predictive_ability(_series({"a": null_noise + 0.03, "b": -null_noise}, n=60), block_length=10, replications=500, random_seed=3)
    assert null["valid"] and positive["valid"]
    assert positive["result_metrics"]["spa_p_value"] < null["result_metrics"]["spa_p_value"]
    assert positive["result_metrics"]["spa_p_value"] < 0.10


def test_spa_constant_series_is_protected():
    result = superior_predictive_ability(_series({"a": np.ones(40), "b": np.ones(40)}, n=40), block_length=10, replications=200)
    assert result["status"] == "INSUFFICIENT_DATA"
    mixed = superior_predictive_ability(_series({"constant": np.ones(40), "variable": np.sin(np.arange(40))}, n=40), block_length=10, replications=200)
    assert mixed["valid"]
    assert "CONSTANT_CANDIDATE_EXCLUDED:constant" in mixed["warnings"]


def test_mcs_retains_identical_and_eliminates_obviously_poor_model():
    identical = model_confidence_set(_series({"a": np.ones(40), "b": np.ones(40)}, n=40), block_length=10, replications=300, random_seed=2)
    assert identical["result_metrics"]["retained_models"] == ["a", "b"]
    losses = _series({"good": np.zeros(40), "middle": np.ones(40), "poor": np.ones(40) * 5}, n=40, orientation="loss", overlap=1)
    first = model_confidence_set(losses, block_length=5, replications=300, random_seed=2)
    second = model_confidence_set(losses, block_length=5, replications=300, random_seed=2)
    assert first["valid"]
    assert first["result_metrics"]["eliminated_models"][0]["model_id"] == "poor"
    assert _logical(first) == _logical(second)


def test_seed_dispersion_single_unstable_retries_and_rank_stability():
    single = seed_dispersion([{"model_id": "a", "seed": 1, "value": 1.0}])
    assert single["valid"] and "SINGLE_SEED:a" in single["warnings"]
    records = [
        {"model_id": "a", "seed": 1, "value": 1.0, "attempt": 1},
        {"model_id": "a", "seed": 1, "value": 2.0, "attempt": 2},
        {"model_id": "a", "seed": 2, "value": -1.0},
        {"model_id": "b", "seed": 1, "value": 0.0},
        {"model_id": "b", "seed": 2, "value": 3.0},
    ]
    result = seed_dispersion(records, instability_cv_threshold=0.1)
    assert result["result_metrics"]["models"]["a"]["seed_count"] == 2
    assert result["result_metrics"]["retry_record_count_excluded"] == 1
    assert "SEED_INSTABILITY:a" in result["warnings"]
    assert result["result_metrics"]["rank_stability_mean_spearman"] == pytest.approx(-1.0)
    assert "CROSS_MODEL_RANK_INSTABILITY" in result["warnings"]
