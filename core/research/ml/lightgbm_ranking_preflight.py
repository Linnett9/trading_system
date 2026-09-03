from __future__ import annotations

import json
import math
import platform
import warnings
from datetime import datetime, timezone
from hashlib import sha256
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, Callable, Sequence


CONTRACT_VERSION = "lightgbm_ranking_dependency_preflight_v1"
SUPPORTED_OBJECTIVES = ("rank_xendcg", "lambdarank")
SUPPORTED_LABEL_TYPES = ("quintile_integer", "decile_integer", "nonnegative_integer_relevance")
REJECTED_LABEL_TYPES = ("continuous_percentile", "negative_relevance", "non_integer_relevance")
STATUSES = {
    "READY", "DEPENDENCY_UNAVAILABLE", "DEPENDENCY_CONFLICT", "OBJECTIVE_UNAVAILABLE",
    "GROUPED_QUERY_UNAVAILABLE", "NONDETERMINISTIC_CONFIGURATION", "SERIALISATION_FAILURE",
    "INVALID_ENVIRONMENT", "NUMERICAL_FAILURE",
}


class PreflightError(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def deterministic_ranker_configuration(
    *,
    objective: str,
    num_threads: int = 1,
    device_type: str = "cpu",
) -> dict[str, Any]:
    if objective not in SUPPORTED_OBJECTIVES:
        raise PreflightError("OBJECTIVE_UNAVAILABLE", f"OBJECTIVE_UNSUPPORTED:{objective}")
    if isinstance(num_threads, bool) or not isinstance(num_threads, int) or not 1 <= num_threads <= 2:
        raise PreflightError("NONDETERMINISTIC_CONFIGURATION", "THREAD_COUNT_MUST_BE_ONE_OR_TWO")
    device = str(device_type or "cpu").lower()
    if device not in {"cpu", "gpu"}:
        raise PreflightError("INVALID_ENVIRONMENT", f"LIGHTGBM_DEVICE_UNSUPPORTED:{device_type}")
    configuration = {
        "objective": objective,
        "n_estimators": 12,
        "learning_rate": 0.1,
        "num_leaves": 4,
        "max_depth": 3,
        "min_child_samples": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "max_bin": 31,
        "random_state": 1729,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": num_threads,
        "bagging_seed": 1729,
        "feature_fraction_seed": 1729,
        "data_random_seed": 1729,
        "verbosity": -1,
    }
    if device == "gpu":
        configuration.update(
            {
                "device_type": "gpu",
                "gpu_platform_id": 0,
                "gpu_device_id": 0,
                "force_col_wise": False,
                "deterministic": False,
            }
        )
    return configuration


def gpu_preferred_ranker_configuration(
    *,
    objective: str,
    num_threads: int = 1,
    gpu_supported: bool,
    safe_cpu_fallback: bool = True,
    fallback_reason: str = "",
) -> dict[str, Any]:
    if gpu_supported:
        parameters = deterministic_ranker_configuration(
            objective=objective,
            num_threads=num_threads,
            device_type="gpu",
        )
        policy = "GPU"
        fallback = ""
    elif safe_cpu_fallback:
        parameters = deterministic_ranker_configuration(
            objective=objective,
            num_threads=num_threads,
            device_type="cpu",
        )
        policy = "CPU_FALLBACK"
        fallback = fallback_reason or "LIGHTGBM_GPU_NOT_SUPPORTED_BY_CURRENT_BUILD_OR_DRIVER"
    else:
        raise PreflightError("INVALID_ENVIRONMENT", "LIGHTGBM_GPU_REQUIRED_BUT_UNSUPPORTED")
    return {
        "runtime_policy": policy,
        "safe_cpu_fallback_reason": fallback,
        "parameters": parameters,
    }


def validate_grouped_ranking_input(
    feature_matrix: Sequence[Sequence[Any]],
    labels: Sequence[Any],
    group_sizes: Sequence[Any],
    *,
    row_ids: Sequence[Any],
    label_type: str,
    num_threads: int,
) -> tuple[list[list[float]], list[int], list[int], list[str]]:
    deterministic_ranker_configuration(objective="rank_xendcg", num_threads=num_threads)
    if label_type not in SUPPORTED_LABEL_TYPES:
        raise PreflightError("INVALID_ENVIRONMENT", f"LABEL_TYPE_REJECTED:{label_type}")
    matrix = [[float(value) for value in row] for row in feature_matrix]
    integer_labels = list(labels)
    groups = list(group_sizes)
    ordered_ids = [str(value) for value in row_ids]
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise PreflightError("INVALID_ENVIRONMENT", "FEATURE_MATRIX_EMPTY_OR_RAGGED")
    if not all(math.isfinite(value) for row in matrix for value in row):
        raise PreflightError("NUMERICAL_FAILURE", "FEATURE_MATRIX_NON_FINITE")
    if len(matrix) != len(integer_labels) or len(matrix) != len(ordered_ids):
        raise PreflightError("GROUPED_QUERY_UNAVAILABLE", "ROW_LABEL_ID_LENGTH_MISMATCH")
    if ordered_ids != sorted(ordered_ids) or len(set(ordered_ids)) != len(ordered_ids):
        raise PreflightError("INVALID_ENVIRONMENT", "ROW_IDS_NOT_UNIQUE_DETERMINISTIC_ORDER")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in integer_labels):
        raise PreflightError("INVALID_ENVIRONMENT", "INTEGER_NONNEGATIVE_RELEVANCE_REQUIRED")
    if (
        not groups
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in groups)
        or sum(groups) != len(matrix)
    ):
        raise PreflightError("GROUPED_QUERY_UNAVAILABLE", "GROUP_SIZE_VECTOR_INVALID")
    return matrix, integer_labels, groups, ordered_ids


def run_lightgbm_ranking_preflight(
    temporary_directory: str | Path,
    *,
    num_threads: int = 1,
    importer: Callable[[str], Any] = import_module,
    objectives: Sequence[str] = SUPPORTED_OBJECTIVES,
) -> dict[str, Any]:
    try:
        try:
            lgb = importer("lightgbm")
        except (ImportError, ModuleNotFoundError) as exc:
            raise PreflightError("DEPENDENCY_UNAVAILABLE", "LIGHTGBM_IMPORT_FAILED") from exc
        np = importer("numpy")
        matrix, labels, groups, row_ids = validate_grouped_ranking_input(
            [[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [3, 1]],
            [0, 1, 2, 3, 0, 1, 2, 3], [4, 4],
            row_ids=[f"R{i:02d}" for i in range(8)],
            label_type="nonnegative_integer_relevance", num_threads=num_threads,
        )
        objective_results = {}
        rank_xendcg_predictions = None
        serialisation_ready = False
        for objective in objectives:
            if objective not in SUPPORTED_OBJECTIVES:
                raise PreflightError("OBJECTIVE_UNAVAILABLE", f"OBJECTIVE_UNSUPPORTED:{objective}")
            configuration = deterministic_ranker_configuration(objective=objective, num_threads=num_threads)
            try:
                first = lgb.LGBMRanker(**configuration).fit(matrix, labels, group=groups)
                second = lgb.LGBMRanker(**configuration).fit(matrix, labels, group=groups)
            except Exception as exc:
                raise PreflightError("OBJECTIVE_UNAVAILABLE", f"OBJECTIVE_SMOKE_FAILED:{objective}") from exc
            first_predictions = np.asarray(_predict_without_feature_name_warning(first, matrix), dtype=float)
            second_predictions = np.asarray(_predict_without_feature_name_warning(second, matrix), dtype=float)
            if not np.isfinite(first_predictions).all():
                raise PreflightError("NUMERICAL_FAILURE", f"NONFINITE_PREDICTIONS:{objective}")
            repeatable = bool(np.allclose(first_predictions, second_predictions, rtol=0.0, atol=1e-12))
            if not repeatable:
                raise PreflightError("NONDETERMINISTIC_CONFIGURATION", f"REPEATED_FIT_MISMATCH:{objective}")
            objective_results[objective] = {
                "available": True, "grouped_fit": True, "finite_predictions": True,
                "deterministic_repeatability": repeatable,
                "prediction_checksum": canonical_hash(first_predictions.tolist()),
            }
            if objective == "rank_xendcg":
                rank_xendcg_predictions = first_predictions
                target = Path(temporary_directory).resolve()
                target.mkdir(parents=True, exist_ok=True)
                model_path = target / "lightgbm_rank_xendcg_preflight.txt"
                try:
                    first.booster_.save_model(str(model_path))
                    reloaded = lgb.Booster(model_file=str(model_path))
                    reloaded_predictions = np.asarray(reloaded.predict(matrix), dtype=float)
                except Exception as exc:
                    raise PreflightError("SERIALISATION_FAILURE", "MODEL_SAVE_OR_RELOAD_FAILED") from exc
                serialisation_ready = bool(
                    np.allclose(rank_xendcg_predictions, reloaded_predictions, rtol=0.0, atol=1e-12)
                )
                if not serialisation_ready:
                    raise PreflightError("SERIALISATION_FAILURE", "RELOADED_PREDICTIONS_MISMATCH")
        if "rank_xendcg" not in objective_results or "lambdarank" not in objective_results:
            raise PreflightError("OBJECTIVE_UNAVAILABLE", "REQUIRED_OBJECTIVE_NOT_TESTED")
        wheel_state = _wheel_state()
        configuration = deterministic_ranker_configuration(objective="rank_xendcg", num_threads=num_threads)
        logical = {
            "contract_version": CONTRACT_VERSION, "status": "READY", "valid": True,
            "blocking_reasons": [], "warnings": [],
            "lightgbm_version": str(lgb.__version__), "python_version": platform.python_version(),
            "dependency_versions": {
                "numpy": metadata.version("numpy"), "scipy": metadata.version("scipy"),
                "scikit_learn": metadata.version("scikit-learn"),
            },
            "installation_source": "python_package_environment",
            "installation_artifact": wheel_state,
            "openmp_runtime_resolved": True,
            "rank_xendcg_capability": objective_results["rank_xendcg"],
            "lambdarank_capability": objective_results["lambdarank"],
            "grouped_query_capability": True,
            "model_serialisation_capability": serialisation_ready,
            "deterministic_repeatability": True,
            "deterministic_tolerance": {"relative": 0.0, "absolute": 1e-12},
            "thread_policy_identity": f"bounded_cpu_threads_1_to_2:selected_{num_threads}",
            "supported_label_types": list(SUPPORTED_LABEL_TYPES),
            "rejected_label_types": list(REJECTED_LABEL_TYPES),
            "deterministic_configuration": configuration,
            "fixture_identity": canonical_hash({
                "matrix": matrix, "labels": labels, "groups": groups, "row_ids": row_ids,
            }),
        }
        logical["configuration_checksum"] = canonical_hash(configuration)
        logical["logical_result_checksum"] = canonical_hash(logical)
        return {**logical, "creation_metadata": _creation_metadata()}
    except PreflightError as exc:
        return _blocked(exc)


def _wheel_state() -> str:
    distribution = metadata.distribution("lightgbm")
    wheel = distribution.read_text("WHEEL") or ""
    return "prebuilt_wheel" if "Wheel-Version:" in wheel else "source_or_unknown"


def _predict_without_feature_name_warning(model: Any, matrix: Sequence[Sequence[float]]) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMRanker was fitted with feature names",
            category=UserWarning,
        )
        return model.predict(matrix)


def _blocked(error: PreflightError) -> dict[str, Any]:
    logical = {
        "contract_version": CONTRACT_VERSION,
        "status": error.status if error.status in STATUSES else "INVALID_ENVIRONMENT",
        "valid": False, "blocking_reasons": [error.reason], "warnings": [],
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _creation_metadata() -> dict[str, str]:
    return {"created_at": datetime.now(timezone.utc).isoformat()}
