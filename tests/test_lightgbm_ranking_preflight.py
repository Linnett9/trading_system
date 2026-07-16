from __future__ import annotations

import copy
from importlib import metadata

import numpy as np
import pytest

from core.research.ml.lightgbm_ranking_preflight import (
    canonical_hash,
    deterministic_ranker_configuration,
    run_lightgbm_ranking_preflight,
    validate_grouped_ranking_input,
)


def _valid(**overrides):
    values = {
        "feature_matrix": [[0.0], [1.0], [2.0], [3.0]],
        "labels": [0, 1, 2, 3],
        "group_sizes": [4],
        "row_ids": ["R00", "R01", "R02", "R03"],
        "label_type": "nonnegative_integer_relevance",
        "num_threads": 1,
    }
    values.update(overrides)
    return validate_grouped_ranking_input(**values)


def test_lightgbm_import_and_exact_version():
    import lightgbm
    assert lightgbm.__version__ == metadata.version("lightgbm") == "4.6.0"


def test_rank_xendcg_lambda_group_prediction_determinism_and_serialisation(tmp_path):
    result = run_lightgbm_ranking_preflight(tmp_path)
    assert result["status"] == "READY"
    assert result["rank_xendcg_capability"]["grouped_fit"]
    assert result["lambdarank_capability"]["grouped_fit"]
    assert result["rank_xendcg_capability"]["finite_predictions"]
    assert result["deterministic_repeatability"]
    assert result["model_serialisation_capability"]
    assert result["installation_artifact"] == "prebuilt_wheel"
    assert list(tmp_path.iterdir()) == [tmp_path / "lightgbm_rank_xendcg_preflight.txt"]


def test_invalid_group_sum_and_empty_group_rejected():
    with pytest.raises(ValueError, match="GROUP_SIZE_VECTOR_INVALID"):
        _valid(group_sizes=[3])
    with pytest.raises(ValueError, match="GROUP_SIZE_VECTOR_INVALID"):
        _valid(group_sizes=[4, 0])


@pytest.mark.parametrize("labels", [[0, -1, 2, 3], [0.0, 1.0, 2.0, 3.0]])
def test_negative_and_continuous_relevance_rejected(labels):
    with pytest.raises(ValueError, match="INTEGER_NONNEGATIVE_RELEVANCE_REQUIRED"):
        _valid(labels=labels)


def test_continuous_label_contract_rejected():
    with pytest.raises(ValueError, match="LABEL_TYPE_REJECTED"):
        _valid(label_type="continuous_percentile")


def test_nonfinite_features_and_nondeterministic_row_order_rejected():
    with pytest.raises(ValueError, match="FEATURE_MATRIX_NON_FINITE"):
        _valid(feature_matrix=[[0.0], [np.nan], [2.0], [3.0]])
    with pytest.raises(ValueError, match="ROW_IDS_NOT_UNIQUE_DETERMINISTIC_ORDER"):
        _valid(row_ids=["R01", "R00", "R02", "R03"])


def test_explicit_thread_limit_and_bounded_configuration():
    assert deterministic_ranker_configuration(objective="rank_xendcg", num_threads=2)["n_jobs"] == 2
    with pytest.raises(ValueError, match="THREAD_COUNT"):
        deterministic_ranker_configuration(objective="rank_xendcg", num_threads=3)


def test_result_checksum_stable_when_creation_time_changes(tmp_path):
    first = run_lightgbm_ranking_preflight(tmp_path)
    second = run_lightgbm_ranking_preflight(tmp_path)
    assert first["logical_result_checksum"] == second["logical_result_checksum"]
    changed = copy.deepcopy(first)
    changed["creation_metadata"]["created_at"] = "different"
    assert changed["logical_result_checksum"] == first["logical_result_checksum"]
    logical = {k: v for k, v in first.items() if k not in {"creation_metadata", "logical_result_checksum"}}
    assert first["logical_result_checksum"] == canonical_hash(logical)


def test_controlled_unavailable_dependency_fixture(tmp_path):
    def unavailable(name):
        if name == "lightgbm":
            raise ModuleNotFoundError(name)
        return __import__(name)
    result = run_lightgbm_ranking_preflight(tmp_path, importer=unavailable)
    assert result["status"] == "DEPENDENCY_UNAVAILABLE"
    assert not result["valid"]


def test_controlled_unsupported_objective_fixture(tmp_path):
    result = run_lightgbm_ranking_preflight(tmp_path, objectives=("rank_xendcg", "unsupported"))
    assert result["status"] == "OBJECTIVE_UNAVAILABLE"
    assert not result["valid"]
