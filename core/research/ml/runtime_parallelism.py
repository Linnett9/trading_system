from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class MLRuntimeSettings:
    num_workers: int
    model_threads: int
    torch_num_threads: int
    sklearn_n_jobs: int
    feature_workers: int

    def to_dict(self) -> dict[str, int]:
        return {
            "num_workers": self.num_workers,
            "model_threads": self.model_threads,
            "torch_num_threads": self.torch_num_threads,
            "sklearn_n_jobs": self.sklearn_n_jobs,
            "feature_workers": self.feature_workers,
        }


def runtime_settings_from_config(config: dict[str, Any]) -> MLRuntimeSettings:
    ml_config = config.get("ml", {})
    model_threads = _positive_int(ml_config.get("model_threads", 1), "model_threads")
    return MLRuntimeSettings(
        num_workers=_positive_int(ml_config.get("num_workers", 1), "num_workers"),
        model_threads=model_threads,
        torch_num_threads=_positive_int(
            ml_config.get("torch_num_threads", model_threads),
            "torch_num_threads",
        ),
        sklearn_n_jobs=_positive_int(
            ml_config.get("sklearn_n_jobs", model_threads),
            "sklearn_n_jobs",
        ),
        feature_workers=_positive_int(
            ml_config.get("feature_workers", 1),
            "feature_workers",
        ),
    )


def apply_runtime_parallelism(config: dict[str, Any]) -> MLRuntimeSettings:
    settings = runtime_settings_from_config(config)
    for name in THREAD_ENV_VARS:
        os.environ[name] = str(settings.model_threads)
    _apply_torch_threads(settings.torch_num_threads)
    return settings


def apply_worker_thread_environment(model_threads: int) -> None:
    model_threads = _positive_int(model_threads, "model_threads")
    for name in THREAD_ENV_VARS:
        os.environ[name] = str(model_threads)


def format_runtime_settings(settings: MLRuntimeSettings) -> str:
    return (
        f"num_workers={settings.num_workers} | "
        f"model_threads={settings.model_threads} | "
        f"torch_num_threads={settings.torch_num_threads} | "
        f"sklearn_n_jobs={settings.sklearn_n_jobs} | "
        f"feature_workers={settings.feature_workers}"
    )


def _apply_torch_threads(torch_num_threads: int) -> None:
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(torch_num_threads)


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise RuntimeError(f"ml.{name} must be at least one")
    return parsed
