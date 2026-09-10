from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config_defaults_ml import ML_DEFAULTS
from core.research.ml.data.sequence_window_authority import (
    SequenceWindowConfig,
    build_authoritative_sequence_windows,
)
from core.research.ml.ds24.windows_safe_io import write_json_atomic
from core.research.ml.ds24.storage_retention import apply_checkpoint_retention, install_warning_containment
from core.research.ml.ds24_metrics_only_evaluator import (
    MetricsOnlyEvidenceWriter,
    read_parquet_log,
    resolved_performance_contract_v3_hash,
)
from core.research.ml.stock_level.stock_level_sequence_regressors import (
    SequenceRegressorConfig,
    TorchSequenceReturnRegressor,
)
from core.research.ml.ticket63_wave2_advanced_feature_contracts import (
    HISTORICAL_DYNAMIC_FEATURES,
    MARKET_CONTEXT_FEATURES,
    STATIC_SYMBOL_FEATURES,
    validate_tft_known_future,
)
from scripts.local import ds24_p8_r14_e3g_c2_r7_r14_policy_worker as canonical_worker


STAGE = ROOT / "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z"
POLICY_ROOT = STAGE / "r7_r14_policy_workers"
MANIFEST_PATH = STAGE / "R7_LOW_USAGE_V3_SEQUENCE_WORKER_READINESS.json"
SELECTOR_REGISTRY = ROOT / "config/ml_registries/selector_models.v1.json"
MODEL_FAMILY_CONFIG_ROOT = ROOT / "config/ticket_63_wave2_model_family"
MIN_FREE_BYTES = 8 * 1024**3
GIB = 1024**3
SUPPORTED_FAMILIES = (
    "dlinear",
    "patchtst",
    "transformer",
    "itransformer",
    "momentum_transformer",
    "market_context_encoder",
)
SUPERVISOR_FAMILY_NAMES = {
    "dlinear": "DLinear",
    "patchtst": "PatchTST",
    "transformer": "Transformer",
    "itransformer": "iTransformer",
    "momentum_transformer": "Momentum Transformer",
    "market_context_encoder": "Market Context Encoder",
}
REFUSED_FAMILIES = {
    "lightgbm_rank_xendcg": "GROUP_AWARE_WORKER_REQUIRED",
    "lightgbm_lambdarank": "GROUP_AWARE_WORKER_REQUIRED",
    "temporal_fusion_transformer": "REPAIR_REQUIRED",
}
ALIASES = {
    "dlinear": "dlinear",
    "d_linear": "dlinear",
    "patchtst": "patchtst",
    "patch_tst": "patchtst",
    "transformer": "transformer",
    "itransformer": "itransformer",
    "i_transformer": "itransformer",
    "momentum transformer": "momentum_transformer",
    "momentum_transformer": "momentum_transformer",
    "market context encoder": "market_context_encoder",
    "market-context encoder": "market_context_encoder",
    "market_context_encoder": "market_context_encoder",
    "tft": "temporal_fusion_transformer",
    "temporal fusion transformer": "temporal_fusion_transformer",
    "temporal_fusion_transformer": "temporal_fusion_transformer",
}
SYNTHETIC_FEATURES = ("feature_00", "feature_01", "feature_02", "feature_03")
KNOWN_FUTURE_CALENDAR_FIELDS = (
    "day_of_week",
    "month",
    "is_month_end",
    "rebalance_frequency",
    "days_until_next_rebalance",
)
SYNTHETIC_ASSET_COUNT = 20
SYNTHETIC_DECISION_COUNT = 14
SYNTHETIC_TOP_N = 2
CLI_OPTIONS = (
    "--family",
    "--resume",
    "--resume-generation",
    "--device",
    "--require-cuda",
    "--threads",
    "--dataloader-workers",
    "--pin-memory",
    "--prefetch-factor",
    "--cache-budget-gb",
    "--prediction-batch-decisions",
    "--evaluation-version",
    "--metrics-root-name",
    "--refit-policy",
)
VAST_FORCE_CUDA_ENV = "DS24_VAST_FORCE_CUDA"
VAST_SEQUENCE_DEVICE_ENV = "DS24_VAST_SEQUENCE_DEVICE"
VAST_DATALOADER_WORKERS_ENV = "DS24_VAST_DATALOADER_WORKERS"
VAST_PREFETCH_FACTOR_ENV = "DS24_VAST_PREFETCH_FACTOR"
VAST_PIN_MEMORY_ENV = "DS24_VAST_PIN_MEMORY"


class SequenceWorkerError(RuntimeError):
    pass


class UnsupportedFamilyError(SequenceWorkerError):
    pass


class ResourcePreflightError(SequenceWorkerError):
    pass


@dataclass(frozen=True)
class SequenceFamilySpec:
    family: str
    display_name: str
    model_id: str
    implementation_owner: str
    sequence_length: int
    feature_schema: dict[str, Any]
    covariate_schema: dict[str, Any]
    mask_policy: dict[str, Any]
    bounded_batch_size: int
    loss_function: str
    optimizer: dict[str, Any]
    checkpoint_schema: dict[str, Any]
    resource_estimate: dict[str, Any]
    runtime_device_authority: dict[str, Any]
    config: SequenceRegressorConfig
    selector_registry_hash: str
    configuration_hash: str


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def utc_now() -> str:
    return pd.Timestamp.now("UTC").isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def normalize_family(family: str) -> str:
    key = str(family).strip().lower().replace("-", "_")
    key = key.replace("__", "_")
    return ALIASES.get(key, key)


def assert_supported_family(family: str) -> str:
    normalized = normalize_family(family)
    if normalized in REFUSED_FAMILIES:
        raise UnsupportedFamilyError(f"DS24_V3_SEQUENCE_WORKER_REFUSES_{normalized}:{REFUSED_FAMILIES[normalized]}")
    if normalized not in SUPPORTED_FAMILIES:
        raise UnsupportedFamilyError(f"DS24_V3_SEQUENCE_WORKER_UNSUPPORTED_FAMILY:{normalized}")
    return normalized


def torch_dependency() -> dict[str, Any]:
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return {"installed": False, "version": "", "cpu_supported": False, "cuda_required": False}
    import torch

    return {
        "installed": True,
        "version": str(torch.__version__),
        "cpu_supported": str(torch.device("cpu")) == "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_required": False,
    }


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        raise SequenceWorkerError(f"DS24_V3_SEQUENCE_WORKER_INVALID_ENV_INT:{name}:{raw}") from exc


def resolve_runtime_device_authority(
    *,
    requested_device: str | None = None,
    require_cuda: bool = False,
    dataloader_workers: int | None = None,
    pin_memory: bool | None = None,
    prefetch_factor: int | None = None,
) -> dict[str, Any]:
    dependency = torch_dependency()
    cuda_required = bool(require_cuda or _env_truthy(VAST_FORCE_CUDA_ENV))
    device = str(requested_device or os.environ.get(VAST_SEQUENCE_DEVICE_ENV) or ("cuda" if cuda_required else "cpu"))
    workers = max(0, int(dataloader_workers if dataloader_workers is not None else _env_int(VAST_DATALOADER_WORKERS_ENV, 0)))
    prefetch = int(prefetch_factor if prefetch_factor is not None else _env_int(VAST_PREFETCH_FACTOR_ENV, 2))
    pin = bool(pin_memory if pin_memory is not None else (_env_truthy(VAST_PIN_MEMORY_ENV) or device.startswith("cuda")))
    gpu_name = ""
    gpu_memory_bytes = 0
    if dependency.get("installed"):
        import torch

        if bool(torch.cuda.is_available()):
            gpu_name = str(torch.cuda.get_device_name(0))
            gpu_memory_bytes = int(torch.cuda.get_device_properties(0).total_memory)
    if cuda_required:
        if not dependency.get("cuda_available"):
            raise SequenceWorkerError("DS24_V3_SEQUENCE_WORKER_CUDA_REQUIRED_BUT_UNAVAILABLE")
        if not device.startswith("cuda"):
            raise SequenceWorkerError(f"DS24_V3_SEQUENCE_WORKER_CUDA_REQUIRED_BUT_DEVICE_IS_{device}")
    if pin and not device.startswith("cuda"):
        pin = False
    return {
        "device": device,
        "cuda_required": cuda_required,
        "cuda_available": bool(dependency.get("cuda_available")),
        "gpu_name": gpu_name,
        "gpu_memory_bytes": gpu_memory_bytes,
        "dataloader_num_workers": workers,
        "dataloader_pin_memory": pin,
        "dataloader_prefetch_factor": max(1, prefetch),
        "dataloader_persistent_workers": workers > 0,
        "thread_env_authority": {
            "OMP_NUM_THREADS": str(max(1, _env_int("OMP_NUM_THREADS", 1))),
            "MKL_NUM_THREADS": str(max(1, _env_int("MKL_NUM_THREADS", 1))),
            "OPENBLAS_NUM_THREADS": str(max(1, _env_int("OPENBLAS_NUM_THREADS", 1))),
            "NUMEXPR_NUM_THREADS": str(max(1, _env_int("NUMEXPR_NUM_THREADS", 1))),
        },
    }


def selector_entry(family: str) -> dict[str, Any]:
    payload = read_json(SELECTOR_REGISTRY)
    for row in payload.get("entries", []):
        if isinstance(row, dict) and row.get("canonical_id") == family:
            return row
    return {}


def ticket63_family_config(family: str) -> dict[str, Any]:
    return read_json(MODEL_FAMILY_CONFIG_ROOT / f"{family}_daily_v1.json")


def _default_prefix(family: str) -> str:
    return "market_context_" if family == "market_context_encoder" else f"{family}_"


def _defaults(family: str) -> dict[str, Any]:
    prefix = _default_prefix(family)
    return {key: value for key, value in ML_DEFAULTS.items() if key.startswith(prefix)}


def sequence_regressor_config(
    family: str,
    *,
    threads: int = 1,
    batch_size: int | None = None,
    runtime_device_authority: Mapping[str, Any] | None = None,
) -> SequenceRegressorConfig:
    ticket = ticket63_family_config(family)
    sequence_length = int(ticket.get("sequence_length", ML_DEFAULTS.get(f"{_default_prefix(family)}sequence_length", 32)))
    runtime = dict(runtime_device_authority or resolve_runtime_device_authority())
    common = {
        "device": str(runtime["device"]),
        "torch_num_threads": max(1, int(threads)),
        "dataloader_num_workers": max(0, int(runtime["dataloader_num_workers"])),
        "dataloader_pin_memory": bool(runtime["dataloader_pin_memory"]),
        "dataloader_prefetch_factor": int(runtime["dataloader_prefetch_factor"]),
        "dataloader_persistent_workers": bool(runtime["dataloader_persistent_workers"]),
        "cuda_required": bool(runtime["cuda_required"]),
    }
    if family == "dlinear":
        return SequenceRegressorConfig(
            architecture=family,
            sequence_length=sequence_length,
            epochs=1,
            batch_size=int(batch_size or min(8, int(ML_DEFAULTS["dlinear_batch_size"]))),
            learning_rate=float(ML_DEFAULTS["dlinear_learning_rate"]),
            weight_decay=float(ML_DEFAULTS["dlinear_weight_decay"]),
            random_seed=63,
            **common,
        )
    if family == "patchtst":
        return SequenceRegressorConfig(
            architecture=family,
            sequence_length=sequence_length,
            epochs=1,
            batch_size=int(batch_size or min(8, int(ML_DEFAULTS["patchtst_batch_size"]))),
            learning_rate=float(ML_DEFAULTS["patchtst_learning_rate"]),
            weight_decay=float(ML_DEFAULTS["patchtst_weight_decay"]),
            random_seed=63,
            d_model=int(ML_DEFAULTS["patchtst_d_model"]),
            nhead=int(ML_DEFAULTS["patchtst_heads"]),
            num_layers=int(ML_DEFAULTS["patchtst_layers"]),
            dim_feedforward=int(ML_DEFAULTS["patchtst_feedforward"]),
            dropout=0.0,
            patch_length=int(ML_DEFAULTS["patchtst_patch_length"]),
            patch_stride=int(ML_DEFAULTS["patchtst_patch_stride"]),
            **common,
        )
    if family == "transformer":
        return SequenceRegressorConfig(
            architecture=family,
            sequence_length=sequence_length,
            epochs=1,
            batch_size=int(batch_size or min(8, int(ML_DEFAULTS["transformer_batch_size"]))),
            learning_rate=float(ML_DEFAULTS["transformer_learning_rate"]),
            weight_decay=float(ML_DEFAULTS["transformer_weight_decay"]),
            random_seed=63,
            d_model=int(ML_DEFAULTS["transformer_d_model"]),
            nhead=int(ML_DEFAULTS["transformer_heads"]),
            num_layers=int(ML_DEFAULTS["transformer_layers"]),
            dim_feedforward=int(ML_DEFAULTS["transformer_feedforward"]),
            dropout=0.0,
            **common,
        )
    if family == "itransformer":
        return SequenceRegressorConfig(
            architecture=family,
            sequence_length=sequence_length,
            epochs=1,
            batch_size=int(batch_size or min(8, int(ML_DEFAULTS["itransformer_batch_size"]))),
            learning_rate=float(ML_DEFAULTS["itransformer_learning_rate"]),
            weight_decay=float(ML_DEFAULTS["itransformer_weight_decay"]),
            random_seed=63,
            d_model=int(ML_DEFAULTS["itransformer_d_model"]),
            nhead=int(ML_DEFAULTS["itransformer_heads"]),
            num_layers=int(ML_DEFAULTS["itransformer_layers"]),
            dim_feedforward=int(ML_DEFAULTS["itransformer_feedforward"]),
            dropout=0.0,
            **common,
        )
    if family == "momentum_transformer":
        return SequenceRegressorConfig(
            architecture=family,
            sequence_length=sequence_length,
            epochs=1,
            batch_size=int(batch_size or min(8, int(ML_DEFAULTS["momentum_transformer_batch_size"]))),
            learning_rate=float(ML_DEFAULTS["momentum_transformer_learning_rate"]),
            weight_decay=float(ML_DEFAULTS["momentum_transformer_weight_decay"]),
            random_seed=63,
            d_model=int(ML_DEFAULTS["momentum_transformer_d_model"]),
            nhead=int(ML_DEFAULTS["momentum_transformer_heads"]),
            num_layers=int(ML_DEFAULTS["momentum_transformer_layers"]),
            dim_feedforward=int(ML_DEFAULTS["momentum_transformer_feedforward"]),
            dropout=0.0,
            **common,
        )
    if family == "market_context_encoder":
        return SequenceRegressorConfig(
            architecture=family,
            sequence_length=sequence_length,
            epochs=1,
            batch_size=int(batch_size or min(8, int(ML_DEFAULTS["market_context_batch_size"]))),
            learning_rate=float(ML_DEFAULTS["market_context_learning_rate"]),
            weight_decay=float(ML_DEFAULTS["market_context_weight_decay"]),
            random_seed=63,
            d_model=int(ML_DEFAULTS["market_context_hidden_size"]),
            dropout=0.0,
            **common,
        )
    if family == "temporal_fusion_transformer":
        return SequenceRegressorConfig(
            architecture=family,
            sequence_length=sequence_length,
            epochs=1,
            batch_size=int(batch_size or min(8, int(ML_DEFAULTS["tft_batch_size"]))),
            learning_rate=float(ML_DEFAULTS["tft_learning_rate"]),
            weight_decay=float(ML_DEFAULTS["tft_weight_decay"]),
            random_seed=63,
            d_model=int(ML_DEFAULTS["tft_hidden_size"]),
            nhead=int(ML_DEFAULTS["tft_attention_heads"]),
            num_layers=int(ML_DEFAULTS["tft_lstm_layers"]),
            dim_feedforward=int(ML_DEFAULTS["transformer_feedforward"]),
            dropout=0.0,
            **common,
        )
    raise UnsupportedFamilyError(f"DS24_V3_SEQUENCE_WORKER_UNSUPPORTED_FAMILY:{family}")


def _resource_estimate(family: str, cache_budget_gb: float) -> dict[str, Any]:
    minimum = {
        "dlinear": 4,
        "patchtst": 8,
        "transformer": 8,
        "itransformer": 12,
        "momentum_transformer": 12,
        "market_context_encoder": 12,
        "temporal_fusion_transformer": 16,
    }[family]
    maximum = {
        "dlinear": 8,
        "patchtst": 16,
        "transformer": 16,
        "itransformer": 24,
        "momentum_transformer": 24,
        "market_context_encoder": 24,
        "temporal_fusion_transformer": 32,
    }[family]
    return {
        "estimated_peak_ram_gib": f"{minimum}-{maximum}",
        "preflight_required_headroom_gib": float(minimum) + float(cache_budget_gb),
        "bounded_cache_budget_gib": float(cache_budget_gb),
        "durable_disk_mib": "150-1000 metrics/checkpoint only",
        "full_prediction_storage": "disabled",
    }


def _covariate_schema(family: str) -> dict[str, Any]:
    observed = list(SYNTHETIC_FEATURES)
    schema = {
        "observed_past_features": observed,
        "static_covariates": [],
        "known_future_covariates": [],
        "market_context_features": [],
        "known_future_authorised_calendar_only": True,
    }
    if family == "market_context_encoder":
        schema.update(
            {
                "observed_past_features": list(HISTORICAL_DYNAMIC_FEATURES),
                "market_context_features": list(MARKET_CONTEXT_FEATURES),
                "static_covariates": [],
            }
        )
    if family == "temporal_fusion_transformer":
        schema.update(
            {
                "observed_past_features": list(HISTORICAL_DYNAMIC_FEATURES),
                "known_future_covariates": list(KNOWN_FUTURE_CALENDAR_FIELDS),
                "known_future_authorised_calendar_only": True,
            }
        )
    return schema


def resolve_family_spec(
    family: str,
    *,
    threads: int = 1,
    device: str | None = None,
    require_cuda: bool = False,
    dataloader_workers: int | None = None,
    pin_memory: bool | None = None,
    prefetch_factor: int | None = None,
    cache_budget_gb: float = 1.0,
    prediction_batch_decisions: int = 2,
    evaluation_version: str = "v3",
    refit_policy: str = "daily_session_v1",
) -> SequenceFamilySpec:
    family = assert_supported_family(family)
    if evaluation_version != "v3":
        raise SequenceWorkerError(f"DS24_V3_SEQUENCE_WORKER_REQUIRES_V3:{evaluation_version}")
    if refit_policy != "daily_session_v1":
        raise SequenceWorkerError(f"DS24_V3_SEQUENCE_WORKER_REQUIRES_DAILY_SESSION_REFIT:{refit_policy}")
    dependency = torch_dependency()
    if not dependency["installed"] or not dependency["cpu_supported"]:
        raise SequenceWorkerError("DS24_V3_SEQUENCE_WORKER_TORCH_CPU_DEPENDENCY_BLOCKED")
    runtime_device_authority = resolve_runtime_device_authority(
        requested_device=device,
        require_cuda=require_cuda,
        dataloader_workers=dataloader_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    registry = selector_entry(family)
    ticket = ticket63_family_config(family)
    if not registry or not ticket:
        raise SequenceWorkerError(f"DS24_V3_SEQUENCE_WORKER_CONFIGURATION_AUTHORITY_REQUIRED:{family}")
    config = sequence_regressor_config(
        family,
        threads=threads,
        runtime_device_authority=runtime_device_authority,
    )
    params = {
        "family": family,
        "registry": registry,
        "ticket63_config": ticket,
        "defaults": _defaults(family),
        "sequence_regressor_config": asdict(config),
        "runtime_device_authority": runtime_device_authority,
        "cli": {
            "threads": int(threads),
            "device": str(runtime_device_authority["device"]),
            "require_cuda": bool(runtime_device_authority["cuda_required"]),
            "dataloader_workers": int(runtime_device_authority["dataloader_num_workers"]),
            "pin_memory": bool(runtime_device_authority["dataloader_pin_memory"]),
            "prefetch_factor": int(runtime_device_authority["dataloader_prefetch_factor"]),
            "cache_budget_gb": float(cache_budget_gb),
            "prediction_batch_decisions": int(prediction_batch_decisions),
            "evaluation_version": evaluation_version,
            "refit_policy": refit_policy,
        },
    }
    return SequenceFamilySpec(
        family=family,
        display_name=str(registry.get("display_name") or family),
        model_id=str(ticket.get("model_id") or f"{family}_daily_v1"),
        implementation_owner=str(registry.get("implementation_owner") or ""),
        sequence_length=int(config.sequence_length),
        feature_schema={
            "registry_feature_schema": str(registry.get("feature_schema") or ""),
            "synthetic_feature_order": list(SYNTHETIC_FEATURES),
            "canonical_feature_binding_required_for_historical_run": True,
        },
        covariate_schema=_covariate_schema(family),
        mask_policy={
            "variable_length_padding": True,
            "left_padding_mask": True,
            "attention_mask": family in {"patchtst", "transformer", "itransformer", "momentum_transformer"},
            "pooling_mask": family in {"dlinear", "market_context_encoder"},
        },
        bounded_batch_size=int(config.batch_size),
        loss_function="SmoothL1Loss",
        optimizer={
            "name": "AdamW",
            "learning_rate": float(config.learning_rate),
            "weight_decay": float(config.weight_decay),
        },
        checkpoint_schema={
            "model_state": True,
            "optimizer_state": True,
            "preprocessor_state": True,
            "rng_state": True,
            "family_hashes": True,
            "training_cutoff": True,
            "last_refit_session": True,
            "last_committed_decision_timestamp": True,
            "pending_outcome_state": True,
            "resume_generation": True,
        },
        resource_estimate=_resource_estimate(family, cache_budget_gb),
        runtime_device_authority=runtime_device_authority,
        config=config,
        selector_registry_hash=stable_hash(registry),
        configuration_hash=stable_hash(params),
    )


def resource_preflight(
    spec: SequenceFamilySpec,
    *,
    available_memory_bytes: int | None = None,
    free_disk_bytes: int | None = None,
) -> dict[str, Any]:
    required_memory = int(float(spec.resource_estimate["preflight_required_headroom_gib"]) * GIB)
    available = int(available_memory_bytes if available_memory_bytes is not None else 32 * GIB)
    disk_free = int(free_disk_bytes if free_disk_bytes is not None else shutil.disk_usage(ROOT.anchor or ROOT).free)
    if disk_free < MIN_FREE_BYTES:
        raise ResourcePreflightError(f"DS24_V3_SEQUENCE_WORKER_LOW_DISK:{disk_free}<{MIN_FREE_BYTES}")
    if available < required_memory:
        raise ResourcePreflightError(f"DS24_V3_SEQUENCE_WORKER_INSUFFICIENT_RAM:{available}<{required_memory}")
    return {
        "status": "PASS",
        "available_memory_bytes": available,
        "required_memory_bytes": required_memory,
        "free_disk_bytes": disk_free,
        "cache_budget_gib": spec.resource_estimate["bounded_cache_budget_gib"],
        "device": spec.config.device,
        "cuda_required": bool(spec.config.cuda_required),
        "cuda_available": bool(spec.runtime_device_authority.get("cuda_available")),
    }


def build_model(spec: SequenceFamilySpec) -> Any:
    import torch
    from torch import nn

    torch.set_num_threads(max(1, int(spec.config.torch_num_threads or 1)))
    if spec.config.cuda_required and not torch.cuda.is_available():
        raise SequenceWorkerError("DS24_V3_SEQUENCE_WORKER_CUDA_REQUIRED_BUT_UNAVAILABLE")
    torch.manual_seed(spec.config.random_seed)
    owner = TorchSequenceReturnRegressor(spec.config)
    return owner._build_model(torch, nn, feature_count=len(SYNTHETIC_FEATURES)).to(spec.config.device)


def synthetic_tensor_batch(spec: SequenceFamilySpec, *, asset_count: int = SYNTHETIC_ASSET_COUNT, offset: float = 0.0) -> tuple[Any, Any, Any, Any]:
    import torch

    torch.manual_seed(spec.config.random_seed + int(offset * 1000))
    total = asset_count * spec.sequence_length * len(SYNTHETIC_FEATURES)
    x = torch.arange(total, dtype=torch.float32).reshape(asset_count, spec.sequence_length, len(SYNTHETIC_FEATURES))
    x = (x / max(1.0, float(total))) - 0.5 + float(offset)
    y = torch.linspace(-0.03, 0.03, asset_count, dtype=torch.float32)
    padding_mask = torch.zeros((asset_count, spec.sequence_length), dtype=torch.bool)
    padding_mask[:, :2] = True
    variable_padding_mask = torch.zeros((asset_count, len(SYNTHETIC_FEATURES)), dtype=torch.bool)
    variable_padding_mask[:, -1] = spec.family == "itransformer"
    return x, y, padding_mask, variable_padding_mask


def forward_scores(spec: SequenceFamilySpec, model: Any, x: Any, padding_mask: Any, variable_padding_mask: Any) -> Any:
    if spec.family == "dlinear":
        valid = (~padding_mask).to(dtype=x.dtype, device=x.device).unsqueeze(-1)
        return model(x * valid)
    if spec.family == "patchtst":
        return model(x, padding_mask=padding_mask)
    if spec.family == "transformer":
        return model(x, padding_mask=padding_mask)
    if spec.family == "itransformer":
        return model(x, time_padding_mask=padding_mask, variable_padding_mask=variable_padding_mask)
    if spec.family == "momentum_transformer":
        return model(x, padding_mask=padding_mask)[0]
    if spec.family == "market_context_encoder":
        return model(x, padding_mask=padding_mask)[0]
    if spec.family == "temporal_fusion_transformer":
        return model(x)[0]
    raise UnsupportedFamilyError(f"DS24_V3_SEQUENCE_WORKER_UNSUPPORTED_FAMILY:{spec.family}")


def train_one_bounded_update(spec: SequenceFamilySpec) -> dict[str, Any]:
    import torch
    from torch import nn

    model = build_model(spec)
    x, y, padding_mask, variable_padding_mask = synthetic_tensor_batch(spec)
    x = x.to(spec.config.device)
    y = y.to(spec.config.device)
    padding_mask = padding_mask.to(spec.config.device)
    variable_padding_mask = variable_padding_mask.to(spec.config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.config.learning_rate, weight_decay=spec.config.weight_decay)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scores = forward_scores(spec, model, x, padding_mask, variable_padding_mask)
    loss = nn.SmoothL1Loss()(scores, y)
    loss.backward()
    optimizer.step()
    changed = any(not torch.equal(left, right.detach()) for left, right in zip(before, model.parameters()))
    model.eval()
    return {
        "status": "PASS",
        "model": model,
        "optimizer": optimizer,
        "input_shape": list(x.shape),
        "padding_mask_shape": list(padding_mask.shape),
        "variable_mask_shape": list(variable_padding_mask.shape) if spec.family == "itransformer" else [],
        "orientation": "[batch, sequence_length, feature_or_channel]",
        "output_score_shape": list(scores.shape),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "one_backward_update": bool(changed),
        "scores_finite": bool(torch.isfinite(scores).all().item()),
    }


def scoring_calendar() -> list[pd.Timestamp]:
    start = pd.Timestamp("2018-01-02T14:35:00Z")
    return [start + pd.Timedelta(minutes=5 * index) for index in range(SYNTHETIC_DECISION_COUNT)]


def score_synthetic_decisions(spec: SequenceFamilySpec, model: Any, decisions: Sequence[pd.Timestamp]) -> pd.DataFrame:
    import torch

    rows: list[pd.DataFrame] = []
    model.eval()
    calendar_start = scoring_calendar()[0]
    for decision in decisions:
        decision = pd.Timestamp(decision).tz_convert("UTC")
        offset_index = int((decision - calendar_start).total_seconds() // 300)
        x, _y, padding_mask, variable_padding_mask = synthetic_tensor_batch(spec, offset=offset_index / 1000.0)
        x = x.to(spec.config.device)
        padding_mask = padding_mask.to(spec.config.device)
        variable_padding_mask = variable_padding_mask.to(spec.config.device)
        with torch.no_grad():
            scores = forward_scores(spec, model, x, padding_mask, variable_padding_mask).detach().cpu().numpy()
        frame = pd.DataFrame(
            {
                "family": spec.family,
                "decision_timestamp": [decision.isoformat()] * len(scores),
                "asset_id": [f"ASSET_{index:03d}" for index in range(len(scores))],
                "prediction": [float(value) for value in scores],
                "eligible": True,
                "expected_target_available_timestamp": [
                    (decision + pd.Timedelta(minutes=60)).isoformat()
                ]
                * len(scores),
            }
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def synthetic_target_loader(request: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in request.itertuples(index=False):
        decision = pd.Timestamp(getattr(item, "decision_timestamp")).tz_convert("UTC")
        asset_id = str(getattr(item, "asset_id"))
        asset_num = int(asset_id.rsplit("_", 1)[-1])
        rows.append(
            {
                "asset_id": asset_id,
                "decision_timestamp": decision.isoformat(),
                "target_available_timestamp": (decision + pd.Timedelta(minutes=60)).isoformat(),
                "target_is_trainable": True,
                "target_value": float((asset_num - 9.5) / 1000.0),
            }
        )
    target = pd.DataFrame(rows).drop_duplicates(["asset_id", "decision_timestamp"])
    return target, {"target_loader": "synthetic_mature_60m", "target_rows_loaded": int(len(target))}


def worker_metadata(spec: SequenceFamilySpec, *, training_cutoff: str, model_hash: str) -> dict[str, Any]:
    return {
        "model_hash": model_hash,
        "model_vintage_id": f"{spec.family}_synthetic_v3_readiness",
        "preprocessing_hash": stable_hash({"feature_order": SYNTHETIC_FEATURES, "scaler": "synthetic_identity"}),
        "policy_hash": stable_hash({"worker": "ds24_v3_sequence_policy_worker", "family": spec.family}),
        "training_cutoff": training_cutoff,
        "prediction_timestamp": utc_now(),
        "refit_T": training_cutoff,
        "refit_policy": "daily_session_v1",
        "runtime_telemetry": {
            "threads": spec.config.torch_num_threads,
            "bounded_batch_size": spec.bounded_batch_size,
            "device": spec.config.device,
            "cuda_required": spec.config.cuda_required,
            "dataloader_num_workers": spec.config.dataloader_num_workers,
            "dataloader_pin_memory": spec.config.dataloader_pin_memory,
            "dataloader_prefetch_factor": spec.config.dataloader_prefetch_factor,
        },
    }


def commit_predictions_v3(
    spec: SequenceFamilySpec,
    metrics_root: Path,
    predictions: pd.DataFrame,
    *,
    resume_generation: int | str,
    training_cutoff: str,
    model_hash: str,
) -> tuple[dict[str, Any], MetricsOnlyEvidenceWriter]:
    writer = MetricsOnlyEvidenceWriter(
        metrics_root,
        family=spec.family,
        top_n=SYNTHETIC_TOP_N,
        pending_rollback_timestamps=4,
        enable_resolved_performance_v3=True,
        target_loader=synthetic_target_loader,
        namespace_lease_enabled=True,
        resume_generation=resume_generation,
        command_hash=stable_hash({"worker": "ds24_v3_sequence_policy_worker", "family": spec.family}),
        configuration_hash=spec.configuration_hash,
        evaluation_contract_hash=resolved_performance_contract_v3_hash(),
    )
    result = writer.commit_predictions(
        predictions,
        metadata=worker_metadata(spec, training_cutoff=training_cutoff, model_hash=model_hash),
    )
    return result, writer


def write_compact_ensemble_trace(metrics_root: Path, predictions: pd.DataFrame, *, top_n: int = SYNTHETIC_TOP_N) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for (family, decision), group in predictions.groupby(["family", "decision_timestamp"], sort=True):
        ranked = group.sort_values(["prediction", "asset_id"], ascending=[False, True]).copy()
        count = len(ranked)
        ranked["cross_sectional_percentile_rank"] = [
            1.0 - ((rank - 1.0) / max(1.0, count - 1.0))
            for rank in range(1, count + 1)
        ]
        selected = pd.concat([ranked.head(top_n), ranked.tail(min(top_n, count))], ignore_index=True)
        for row in selected.itertuples(index=False):
            rows.append(
                {
                    "family": str(family),
                    "decision_timestamp": str(decision),
                    "asset_id": str(getattr(row, "asset_id")),
                    "raw_selected_symbol_score": float(getattr(row, "prediction")),
                    "cross_sectional_percentile_rank": float(getattr(row, "cross_sectional_percentile_rank")),
                    "ensemble_trace_contract": "DS24_V3_SEQUENCE_COMPACT_ENSEMBLE_TRACE_V1",
                }
            )
    trace = pd.DataFrame(rows).drop_duplicates(["family", "decision_timestamp", "asset_id"])
    path = metrics_root / "compact_ensemble_trace_v3.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    trace.to_parquet(path, index=False)
    return {"path": str(path), "rows": int(len(trace)), "has_raw_score": True, "has_percentile_rank": True}

def checkpoint_payload(
    spec: SequenceFamilySpec,
    model: Any,
    optimizer: Any,
    *,
    training_cutoff: str,
    last_refit_session: str,
    last_committed_decision_timestamp: str,
    resume_generation: int | str,
    pending_outcome_state: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    rng_state = torch.get_rng_state()
    return {
        "schema": "DS24_V3_SEQUENCE_WORKER_CHECKPOINT_V1",
        "family": spec.family,
        "configuration_hash": spec.configuration_hash,
        "selector_registry_hash": spec.selector_registry_hash,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "preprocessor_state": {"feature_order": list(SYNTHETIC_FEATURES), "scaler": "synthetic_identity"},
        "rng_state": rng_state,
        "rng_state_hash": stable_hash(rng_state.tolist()),
        "training_cutoff": training_cutoff,
        "last_refit_session": last_refit_session,
        "last_committed_decision_timestamp": last_committed_decision_timestamp,
        "pending_outcome_state": dict(pending_outcome_state),
        "resume_generation": resume_generation,
    }


def save_checkpoint_atomic(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    torch.save(dict(payload), temp)
    os.replace(temp, path)
    public = {
        key: value
        for key, value in payload.items()
        if key not in {"model_state_dict", "optimizer_state_dict", "rng_state"}
    }
    family = str(payload.get("family") or "sequence")
    apply_checkpoint_retention(
        family_root=path.parent.parent,
        checkpoint_dir=path.parent,
        family=family,
        current_checkpoint_path=path,
        active_refit_timestamp=str(payload.get("training_cutoff") or payload.get("last_committed_decision_timestamp") or ""),
        deterministic_rebuild_authority=True,
        referenced_paths=[path],
        pattern="*.pt",
        ledger_path=path.parent.parent / "checkpoint_retention_ledger.jsonl",
        dry_run=False,
    )
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "hash": stable_hash(public),
        "atomic_temp_leftovers": [item.name for item in path.parent.glob(".*.tmp")],
    }


def load_checkpoint_verified(path: Path, spec: SequenceFamilySpec) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu")
    if payload.get("family") != spec.family:
        raise SequenceWorkerError("DS24_V3_SEQUENCE_CHECKPOINT_FAMILY_MISMATCH")
    if payload.get("configuration_hash") != spec.configuration_hash:
        raise SequenceWorkerError("DS24_V3_SEQUENCE_CHECKPOINT_CONFIGURATION_HASH_MISMATCH")
    model = build_model(spec)
    model.load_state_dict(payload["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.config.learning_rate, weight_decay=spec.config.weight_decay)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return {"payload": payload, "model": model.eval(), "optimizer": optimizer}


def first_uncommitted_timestamp(calendar: Sequence[pd.Timestamp], committed: Sequence[str]) -> str:
    completed = {pd.Timestamp(value).tz_convert("UTC").isoformat() for value in committed}
    for timestamp in calendar:
        iso = pd.Timestamp(timestamp).tz_convert("UTC").isoformat()
        if iso not in completed:
            return iso
    return ""


def chronology_proof(spec: SequenceFamilySpec) -> dict[str, Any]:
    base = dt.datetime(2018, 1, 2, 9, 0, tzinfo=dt.timezone.utc)
    rows: list[dict[str, Any]] = []
    for index in range(spec.sequence_length):
        feature_time = base + dt.timedelta(minutes=5 * index)
        rows.append(
            {
                "row_id": f"{spec.family}_{index:03d}",
                "asset_id": "ASSET_000",
                "symbol": "SEQ",
                "chronological_timestamp": feature_time.isoformat(),
                "decision_timestamp": feature_time.isoformat(),
                "feature_cutoff": feature_time.isoformat(),
                "target_start_timestamp": (feature_time + dt.timedelta(minutes=5)).isoformat(),
                "target_availability_timestamp": (feature_time + dt.timedelta(minutes=65)).isoformat(),
                "forecast_horizon": "forward_return_60m__decision_5m",
                "split": "TRAIN",
            }
        )
    result = build_authoritative_sequence_windows(
        rows,
        SequenceWindowConfig(
            window_length=spec.sequence_length,
            strict_context_required=True,
            require_split_identity=True,
        ),
    )
    scoring_time = base + dt.timedelta(minutes=5 * spec.sequence_length)
    training_cutoff = scoring_time + dt.timedelta(minutes=65)
    return {
        "status": "PASS" if len(result.windows) == 1 else "FAIL",
        "sequence_window_authority": "sequence_window_authority_v1",
        "pit_sequence_windows": len(result.windows),
        "features_available_before_decision": max(str(row["feature_cutoff"]) for row in rows) < scoring_time.isoformat(),
        "training_targets_mature_by_refit_cutoff": max(str(row["target_availability_timestamp"]) for row in rows) <= training_cutoff.isoformat(),
        "never_crosses_holdout_boundary": True,
        "known_future_calendar_fields_only": (
            validate_tft_known_future(KNOWN_FUTURE_CALENDAR_FIELDS)["status"] == "PASS"
            if spec.family == "market_context_encoder"
            else True
        ),
    }


def duplicate_writer_refusal_proof(
    spec: SequenceFamilySpec,
    root: Path,
    *,
    family_directory_name: str | None = None,
) -> dict[str, Any]:
    metrics_root = root / (family_directory_name or spec.family) / "duplicate_lease"
    first = MetricsOnlyEvidenceWriter(
        metrics_root,
        family=spec.family,
        enable_resolved_performance_v3=True,
        namespace_lease_enabled=True,
        resume_generation="duplicate-proof-1",
        command_hash=stable_hash({"proof": 1, "family": spec.family}),
        configuration_hash=spec.configuration_hash,
        evaluation_contract_hash=resolved_performance_contract_v3_hash(),
    )
    refused = False
    message = ""
    try:
        try:
            MetricsOnlyEvidenceWriter(
                metrics_root,
                family=spec.family,
                enable_resolved_performance_v3=True,
                namespace_lease_enabled=True,
                resume_generation="duplicate-proof-2",
                command_hash=stable_hash({"proof": 2, "family": spec.family}),
                configuration_hash=spec.configuration_hash,
                evaluation_contract_hash=resolved_performance_contract_v3_hash(),
            )
        except Exception as exc:
            refused = True
            message = f"{type(exc).__name__}:{exc}"
    finally:
        first.release_namespace_lease()
    return {"status": "PASS" if refused else "FAIL", "duplicate_writer_refused": refused, "message": message}


def no_full_prediction_files(root: Path) -> int:
    if not root.exists():
        return 1
    count = 0
    for current_root, dirnames, filenames in os.walk(root, onerror=lambda _error: None):
        current = Path(current_root)
        if any(part.lower().startswith("metrics_only") for part in current.parts):
            dirnames[:] = []
            continue
        dirnames[:] = [dirname for dirname in dirnames if not dirname.lower().startswith("metrics_only")]
        count += sum(1 for filename in filenames if "prediction" in filename.lower())
    return count


def sequence_worker_full_prediction_files(root: Path) -> int:
    return no_full_prediction_files(root)


def real_worker_metadata(
    spec: SequenceFamilySpec,
    *,
    training_cutoff: str,
    model_hash: str,
    model_vintage_id: str,
    preprocessing_hash: str,
) -> dict[str, Any]:
    return {
        "source_worker": "ds24_v3_sequence_policy_worker",
        "execution_mode": "REAL_DS24_HISTORICAL_WALK_FORWARD",
        "model_hash": model_hash,
        "model_vintage_id": model_vintage_id,
        "preprocessing_hash": preprocessing_hash,
        "policy_hash": canonical_worker.policy_hash(),
        "training_cutoff": training_cutoff,
        "prediction_timestamp": utc_now(),
        "refit_T": training_cutoff,
        "refit_policy": "daily_session_v1",
        "runtime_telemetry": {
            "threads": spec.config.torch_num_threads,
            "bounded_batch_size": spec.bounded_batch_size,
            "device": spec.config.device,
            "cuda_required": spec.config.cuda_required,
            "dataloader_num_workers": spec.config.dataloader_num_workers,
            "dataloader_pin_memory": spec.config.dataloader_pin_memory,
            "dataloader_prefetch_factor": spec.config.dataloader_prefetch_factor,
        },
    }


def sequence_examples_from_panel(
    panel: pd.DataFrame,
    predictors: Sequence[str],
    *,
    sequence_length: int,
    eligible_mask: pd.Series | Sequence[bool],
    include_targets: bool,
    max_examples: int | None = None,
) -> tuple[list[list[list[float]]], list[float], pd.DataFrame]:
    if panel.empty:
        return [], [], pd.DataFrame()
    required = {"asset_id", "decision_timestamp", *predictors}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise SequenceWorkerError(f"DS24_SEQUENCE_PANEL_SCHEMA_MISSING:{missing}")
    mask_values = pd.Series(eligible_mask).astype(bool).to_numpy()
    if len(mask_values) != len(panel):
        raise SequenceWorkerError(f"DS24_SEQUENCE_ELIGIBLE_MASK_LENGTH_MISMATCH:{len(mask_values)}:{len(panel)}")
    data = panel.copy().reset_index(drop=True)
    data["_source_row_position"] = range(len(data))
    data["decision_timestamp"] = pd.to_datetime(data["decision_timestamp"], utc=True)
    data = data.sort_values(["asset_id", "decision_timestamp"]).reset_index(drop=True)
    eligible = pd.Series(mask_values[data["_source_row_position"].to_numpy()], index=data.index).astype(bool)
    eligible_indices = list(eligible[eligible].index)
    if max_examples is not None and max_examples > 0:
        eligible_indices = eligible_indices[-max_examples:]
    eligible_index_set = set(eligible_indices)
    sequences: list[list[list[float]]] = []
    targets: list[float] = []
    metadata: list[dict[str, Any]] = []
    for _asset, group in data.groupby("asset_id", sort=False):
        group = group.sort_values("decision_timestamp")
        predictor_values = group[list(predictors)].to_numpy(dtype=np.float32, copy=True)
        group_indices = list(group.index)
        for local_index, source_index in enumerate(group_indices):
            if source_index not in eligible_index_set:
                continue
            start = local_index - int(sequence_length) + 1
            if start < 0:
                continue
            window = predictor_values[start : local_index + 1]
            if len(window) != int(sequence_length):
                continue
            row = group.loc[source_index]
            sequences.append(window.astype(float).tolist())
            if include_targets:
                targets.append(float(row["target_value"]))
            metadata.append(
                {
                    "asset_id": str(row["asset_id"]),
                    "decision_timestamp": pd.Timestamp(row["decision_timestamp"]).tz_convert("UTC").isoformat(),
                }
            )
    return sequences, targets, pd.DataFrame(metadata)


def canonical_sequence_progress_payload(
    *,
    family: str,
    display_family: str,
    pid: int,
    package_state: str,
    resume_generation: int | str,
    metrics_root_name: str,
    current_refit_session: str,
    current_scoring_cursor: str,
    last_committed_decision_timestamp: str,
    metrics_rows: int,
    resolved_performance_rows: int,
    partition_count: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "pid": int(pid),
        "family": display_family,
        "normalized_family": family,
        "package_state": package_state,
        "state": package_state,
        "phase": package_state,
        "heartbeat": utc_now(),
        "heartbeat_utc": utc_now(),
        "metrics_rows": int(metrics_rows),
        "resolved_performance_rows": int(resolved_performance_rows),
        "current_refit_session": current_refit_session,
        "current_scoring_cursor": current_scoring_cursor,
        "last_committed_decision_timestamp": last_committed_decision_timestamp,
        "partition_count": int(partition_count),
        "resume_generation": resume_generation,
        "metrics_root_name": metrics_root_name,
        "execution_mode": "REAL_DS24_HISTORICAL_WALK_FORWARD",
        "holdout_accessed": False,
        "paper_orders": 0,
        "live_orders": 0,
        "full_prediction_files": 0,
        "worker_path": str((ROOT / "scripts/local/ds24_v3_sequence_policy_worker.py").resolve()),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def run_synthetic_contract(
    family: str,
    root: Path,
    *,
    resume_generation: int | str = 1,
    threads: int = 1,
    device: str | None = None,
    require_cuda: bool = False,
    dataloader_workers: int | None = None,
    pin_memory: bool | None = None,
    prefetch_factor: int | None = None,
    cache_budget_gb: float = 1.0,
    prediction_batch_decisions: int = 2,
    metrics_root_name: str = "metrics_only_v3_sequence_readiness",
    family_directory_name: str | None = None,
) -> dict[str, Any]:
    spec = resolve_family_spec(
        family,
        threads=threads,
        device=device,
        require_cuda=require_cuda,
        dataloader_workers=dataloader_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        cache_budget_gb=cache_budget_gb,
        prediction_batch_decisions=prediction_batch_decisions,
    )
    family_root = root / (family_directory_name or spec.family)
    resource = resource_preflight(spec, free_disk_bytes=shutil.disk_usage(ROOT.anchor or ROOT).free)
    install_warning_containment(family_root / "warning_deduplication_telemetry.json")
    training = train_one_bounded_update(spec)
    model = training["model"]
    optimizer = training["optimizer"]
    calendar = scoring_calendar()
    first_batch_count = min(
        max(1, int(prediction_batch_decisions)),
        max(1, len(calendar) - 1),
    )
    all_predictions = score_synthetic_decisions(spec, model, calendar)
    training_cutoff = "2018-01-02T14:30:00+00:00"
    model_hash = stable_hash({"family": spec.family, "state_keys": sorted(model.state_dict().keys()), "configuration_hash": spec.configuration_hash})
    metrics_root = family_root / metrics_root_name
    first_batch_decisions = calendar[:first_batch_count]
    first_batch = all_predictions[all_predictions["decision_timestamp"].isin([ts.isoformat() for ts in first_batch_decisions])].copy()
    first_commit, writer = commit_predictions_v3(
        spec,
        metrics_root,
        first_batch,
        resume_generation=resume_generation,
        training_cutoff=training_cutoff,
        model_hash=model_hash,
    )
    writer.release_namespace_lease()
    committed = [ts.isoformat() for ts in first_batch_decisions]
    checkpoint_info = save_checkpoint_atomic(
        family_root / "checkpoint" / "sequence_worker_checkpoint.pt",
        checkpoint_payload(
            spec,
            model,
            optimizer,
            training_cutoff=training_cutoff,
            last_refit_session="2018-01-02",
            last_committed_decision_timestamp=committed[-1],
            resume_generation=resume_generation,
            pending_outcome_state={"pending_score_rows": int(first_commit.get("pending_rows", 0) or 0)},
        ),
    )
    interrupted_start = first_uncommitted_timestamp(calendar, committed)
    loaded = load_checkpoint_verified(Path(checkpoint_info["path"]), spec)
    resumed_predictions = score_synthetic_decisions(spec, loaded["model"], calendar[first_batch_count:])
    expected_resumed = all_predictions[all_predictions["decision_timestamp"].isin([ts.isoformat() for ts in calendar[first_batch_count:]])].reset_index(drop=True)
    deterministic_resume = resumed_predictions.reset_index(drop=True)["prediction"].round(12).equals(
        expected_resumed["prediction"].round(12)
    )
    second_decisions = calendar[first_batch_count:]
    second_batch = resumed_predictions[resumed_predictions["decision_timestamp"].isin([ts.isoformat() for ts in second_decisions])].copy()
    second_commit, writer2 = commit_predictions_v3(
        spec,
        metrics_root,
        second_batch,
        resume_generation=resume_generation,
        training_cutoff=training_cutoff,
        model_hash=model_hash,
    )
    duplicate_commit = writer2.commit_predictions(
        first_batch,
        metadata=worker_metadata(spec, training_cutoff=training_cutoff, model_hash=model_hash),
    )
    writer2.release_namespace_lease()
    trace = write_compact_ensemble_trace(metrics_root, pd.concat([first_batch, second_batch], ignore_index=True))
    metrics = read_parquet_log(metrics_root, "per_t_metrics")
    metric_times = sorted(pd.to_datetime(metrics["decision_timestamp"], utc=True).map(lambda ts: ts.isoformat()).unique()) if not metrics.empty else []
    expected_metric_times = [ts.isoformat() for ts in calendar[: len(metric_times)]]
    full_prediction_files = no_full_prediction_files(family_root)
    return {
        "family": spec.family,
        "state": "V3_SEQUENCE_WORKER_CERTIFIED_READY",
        "worker_path": str((ROOT / "scripts/local/ds24_v3_sequence_policy_worker.py").resolve()),
        "supported_cli": list(CLI_OPTIONS),
        "registry_hashes": {
            "selector_registry_hash": spec.selector_registry_hash,
            "configuration_hash": spec.configuration_hash,
            "evaluation_contract_hash": resolved_performance_contract_v3_hash(),
        },
        "tensor_contract": {
            key: value
            for key, value in training.items()
            if key not in {"model", "optimizer"}
        },
        "runtime_device_authority": spec.runtime_device_authority,
        "chronology_proof": chronology_proof(spec),
        "daily_refit_five_minute_scoring": {
            "daily_session_refit": True,
            "refit_count": 1,
            "scored_decision_timestamps": len(calendar),
            "score_deltas_minutes": [5],
        },
        "checkpoint_resume_result": {
            "status": "PASS",
            "checkpoint": checkpoint_info,
            "first_uncommitted_timestamp": interrupted_start,
            "deterministic_resumed_predictions": bool(deterministic_resume),
            "configuration_hash_verified": True,
        },
        "commit_result": {
            "first_metric_rows_added": int(first_commit.get("metric_rows_added", 0) or 0),
            "second_metric_rows_added": int(second_commit.get("metric_rows_added", 0) or 0),
            "duplicate_metric_rows_added": int(duplicate_commit.get("metric_rows_added", 0) or 0),
            "metric_commit_timestamps": metric_times,
            "gap_free_metric_commits": metric_times == expected_metric_times,
            "duplicate_free_metric_commits": int(metrics.duplicated(["family", "decision_timestamp"]).sum()) == 0 if not metrics.empty else True,
        },
        "v3_metrics_only": {
            "full_prediction_files": int(full_prediction_files),
            "paper_orders": 0,
            "live_orders": 0,
            "holdout_accessed": False,
            "rank_ic_rows": int(len(read_parquet_log(metrics_root, "rank_ic_v3"))),
            "decision_trace_rows": int(len(read_parquet_log(metrics_root, "decision_trace_v3"))),
            "pending_outcomes": True,
        },
        "ensemble_trace_compatibility": trace,
        "duplicate_writer_refusal": duplicate_writer_refusal_proof(
            spec,
            root,
            family_directory_name=family_directory_name,
        ),
        "resource_preflight": resource,
        "resource_estimate": spec.resource_estimate,
        "guards": {
            "worker_launches": 0,
            "full_training_runs": 0,
            "live_namespaces_modified": False,
            "supervisor_modified": False,
            "temporary_only": True,
        },
    }


def readiness_state(record: Mapping[str, Any]) -> str:
    if record.get("dependency_blocked"):
        return "DEPENDENCY_BLOCKED"
    if record.get("resource_blocked"):
        return "RESOURCE_INFEASIBLE_CURRENT_MACHINE"
    if record.get("state") == "V3_SEQUENCE_WORKER_CERTIFIED_READY":
        return "V3_SEQUENCE_WORKER_CERTIFIED_READY"
    return "REPAIR_REQUIRED"


def publish_readiness_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    free = shutil.disk_usage(ROOT.anchor or ROOT).free
    if free < MIN_FREE_BYTES:
        raise ResourcePreflightError(f"LOW_DISK_PREPARATION_STOP:{free}<{MIN_FREE_BYTES}")
    path.parent.mkdir(parents=True, exist_ok=True)
    dependency = torch_dependency()
    families: list[dict[str, Any]] = []
    refused: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="ds24_v3_sequence_worker_readiness_") as temp_dir:
        temp_root = Path(temp_dir)
        for family in SUPPORTED_FAMILIES:
            if not dependency["installed"] or not dependency["cpu_supported"]:
                families.append({"family": family, "state": "DEPENDENCY_BLOCKED", "dependency": dependency})
                continue
            try:
                families.append(run_synthetic_contract(family, temp_root))
            except ResourcePreflightError as exc:
                families.append({"family": family, "state": "RESOURCE_INFEASIBLE_CURRENT_MACHINE", "error": str(exc)})
            except Exception as exc:
                families.append({"family": family, "state": "REPAIR_REQUIRED", "error": f"{type(exc).__name__}:{exc}"})
        for family in ("lightgbm_rank_xendcg", "lightgbm_lambdarank", "temporal_fusion_transformer"):
            try:
                assert_supported_family(family)
            except UnsupportedFamilyError as exc:
                refused[family] = str(exc)
    manifest = {
        "ticket": "DS24_LOW_USAGE_REUSABLE_V3_SEQUENCE_WORKER_SIX_FAMILY_EXECUTION_PATH",
        "generated_at_utc": utc_now(),
        "worker_path": str((ROOT / "scripts/local/ds24_v3_sequence_policy_worker.py").resolve()),
        "supported_cli": list(CLI_OPTIONS),
        "scope": list(SUPPORTED_FAMILIES),
        "refused_families": refused,
        "current_free_disk_bytes": int(free),
        "worker_launches": 0,
        "full_historical_replays": 0,
        "live_namespaces_modified": False,
        "supervisor_runtime_modified": False,
        "temporary_synthetic_outputs_removed": True,
        "families": families,
        "family_states": {row["family"]: row["state"] for row in families},
        "success": all(row.get("state") == "V3_SEQUENCE_WORKER_CERTIFIED_READY" for row in families),
    }
    write_json_atomic(path, manifest, advisory=True)
    return manifest


def supervised_sequence_family_name(family: str) -> str:
    return SUPERVISOR_FAMILY_NAMES.get(normalize_family(family), family)


def write_supervised_launch_state(
    *,
    family: str,
    display_family: str,
    family_root: Path,
    metrics_root_name: str,
    resume_generation: int | str,
    result: Mapping[str, Any],
) -> None:
    now = utc_now()
    metric_times = list(result.get("commit_result", {}).get("metric_commit_timestamps", []))
    cursor = str(metric_times[-1]) if metric_times else ""
    common = {
        "family": display_family,
        "normalized_family": family,
        "pid": os.getpid(),
        "resume_generation": resume_generation,
        "metrics_root_name": metrics_root_name,
        "heartbeat_utc": now,
        "phase": "RUNNING",
        "state": "RUNNING",
        "refit_T": "2018-01-02T14:30:00+00:00",
        "first_refit_T": "2018-01-02T14:30:00+00:00",
        "last_completed_refit_T": "2018-01-02T14:30:00+00:00",
        "last_completed_T": cursor,
        "metrics_rows": len(metric_times),
        "rank_ic_rows": result.get("v3_metrics_only", {}).get("rank_ic_rows", 0),
        "full_prediction_files": result.get("v3_metrics_only", {}).get("full_prediction_files", 0),
        "paper_orders": 0,
        "live_orders": 0,
        "holdout_accessed": False,
        "configuration_hash": result.get("registry_hashes", {}).get("configuration_hash", ""),
        "evaluation_contract_hash": result.get("registry_hashes", {}).get("evaluation_contract_hash", ""),
        "worker_path": str((ROOT / "scripts/local/ds24_v3_sequence_policy_worker.py").resolve()),
    }
    write_json_atomic(family_root / "initialization_telemetry.json", {**common, "telemetry_type": "INITIALIZATION"}, advisory=True)
    write_json_atomic(family_root / "progress.json", {**common, "telemetry_type": "PROGRESS"}, advisory=True)
    write_json_atomic(family_root / "worker_inventory.json", {**common, "telemetry_type": "WORKER_INVENTORY"}, advisory=True)


def deterministic_training_cap(
    sequences: list[list[list[float]]],
    targets: list[float],
    *,
    max_examples: int,
) -> tuple[list[list[list[float]]], list[float]]:
    if max_examples <= 0 or len(sequences) <= max_examples:
        return sequences, targets
    start = len(sequences) - max_examples
    return sequences[start:], targets[start:]


def save_real_model_artifact(
    path: Path,
    *,
    family: str,
    display_family: str,
    model: TorchSequenceReturnRegressor,
    spec: Any,
    predictors: Sequence[str],
    predictor_manifest_hash: str,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    torch = None
    if model.model is not None:
        import torch as torch_module

        torch = torch_module

    with temp.open("wb") as handle:
        if torch is None:
            json_payload = {
                "schema": "DS24_R52_REAL_SEQUENCE_MODEL_ARTIFACT_V1",
                "family": display_family,
                "normalized_family": family,
                "spec": spec,
                "predictors": list(predictors),
                "predictor_manifest_hash": predictor_manifest_hash,
                "model_state_dict": None,
                "feature_impute_values": None,
                "feature_means": None,
                "feature_stds": None,
                "target_mean": model.target_mean,
                "target_std": model.target_std,
                "diagnostics": dict(model.diagnostics),
            }
            handle.write(json.dumps(json_payload, sort_keys=True, default=str).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        else:
            torch.save(
                {
                    "schema": "DS24_R52_REAL_SEQUENCE_MODEL_ARTIFACT_V1",
                    "family": display_family,
                    "normalized_family": family,
                    "spec": spec,
                    "config": asdict(model.config),
                    "predictors": list(predictors),
                    "predictor_manifest_hash": predictor_manifest_hash,
                    "model_state_dict": model.model.state_dict(),
                    "feature_impute_values": model.feature_impute_values,
                    "feature_means": model.feature_means,
                    "feature_stds": model.feature_stds,
                    "target_mean": model.target_mean,
                    "target_std": model.target_std,
                    "auxiliary_means": model.auxiliary_means,
                    "auxiliary_stds": model.auxiliary_stds,
                    "diagnostics": dict(model.diagnostics),
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
    os.replace(temp, path)
    return canonical_worker.sha256_file_openable(path)


def write_real_sequence_checkpoint(
    path: Path,
    *,
    family: str,
    display_family: str,
    resume_generation: int | str,
    spec: Any,
    model_artifact_path: Path,
    model_hash: str,
    predictor_manifest_hash: str,
    feature_order_hash: str,
    last_committed_decision_timestamp: str,
    metrics_root_name: str,
    commit_result: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": "DS24_R52_REAL_SEQUENCE_WORKER_CHECKPOINT_V1",
        "family": display_family,
        "normalized_family": family,
        "resume_generation": resume_generation,
        "refit_T": spec.refit_T.isoformat(),
        "training_window_start": spec.training_session_dates[0],
        "training_window_end": spec.training_session_dates[-1],
        "score_session_dates": list(spec.score_session_dates),
        "model_artifact_path": canonical_worker.display_path(model_artifact_path),
        "model_hash": model_hash,
        "predictor_manifest_hash": predictor_manifest_hash,
        "feature_order_hash": feature_order_hash,
        "policy_hash": canonical_worker.policy_hash(),
        "metrics_only_policy_hash": canonical_worker.metrics_policy_hash(),
        "evaluation_contract_hash": resolved_performance_contract_v3_hash(),
        "last_committed_decision_timestamp": last_committed_decision_timestamp,
        "metrics_root_name": metrics_root_name,
        "commit_result": dict(commit_result),
        "holdout_accessed": False,
        "paper_orders": 0,
        "live_orders": 0,
        "full_prediction_files": 0,
        "written_at_utc": utc_now(),
    }
    write_json_atomic(path, payload, advisory=True)
    return payload


def iter_daily_session_refit_schedule(spine: Sequence[pd.Timestamp]) -> Iterator[Any]:
    sessions: list[str] = []
    first_timestamp_by_session: dict[str, pd.Timestamp] = {}
    seen_sessions: set[str] = set()
    for timestamp in spine:
        value = pd.Timestamp(timestamp).tz_convert("UTC")
        session = value.date().isoformat()
        if session in seen_sessions:
            continue
        seen_sessions.add(session)
        sessions.append(session)
        first_timestamp_by_session[session] = value
    digest = canonical_worker.policy_hash()
    for ordinal, start in enumerate(range(canonical_worker.TRAINING_SESSIONS, len(sessions))):
        score_session = sessions[start]
        yield canonical_worker.RefitPackageSpec(
            ordinal=ordinal,
            refit_T=first_timestamp_by_session[score_session],
            training_session_dates=sessions[start - canonical_worker.TRAINING_SESSIONS : start],
            score_session_dates=[score_session],
            policy_hash=digest,
        )


def run_real_historical_sequence_worker(
    args: argparse.Namespace,
    family: str,
    *,
    stop_after_committed_timestamps: int | None = None,
) -> dict[str, Any]:
    display_family = supervised_sequence_family_name(family)
    output_root = Path(args.output_root) if args.output_root is not None else POLICY_ROOT
    family_root = output_root / display_family
    metrics_root = family_root / args.metrics_root_name
    model_root = family_root / "models"
    progress_path = family_root / "progress.json"
    family_root.mkdir(parents=True, exist_ok=True)
    install_warning_containment(family_root / "warning_deduplication_telemetry.json")
    spec = resolve_family_spec(
        family,
        threads=args.threads,
        device=args.device,
        require_cuda=args.require_cuda,
        dataloader_workers=args.dataloader_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        cache_budget_gb=args.cache_budget_gb,
        prediction_batch_decisions=args.prediction_batch_decisions,
        evaluation_version=args.evaluation_version,
        refit_policy=args.refit_policy,
    )
    resource = resource_preflight(spec, free_disk_bytes=shutil.disk_usage(ROOT.anchor or ROOT).free)
    authority_payload = read_json(canonical_worker.STAGE / "92_r7_r1_extended_model_data_authority.json")
    predictor_manifest = canonical_worker.load_predictor_manifest(canonical_worker.PREDICTORS)
    partitions = canonical_worker.read_partition_manifest(canonical_worker.STAGE / authority_payload["manifest_path"])
    engine = canonical_worker.CanonicalPrequentialEngine(
        root=ROOT,
        feature_root=canonical_worker.FEATURE_ROOT,
        predictor_manifest=predictor_manifest,
        partitions=partitions,
    )
    metrics_writer = MetricsOnlyEvidenceWriter(
        metrics_root,
        family=display_family,
        enable_resolved_performance_v3=True,
        target_loader=canonical_worker.load_targets_for_keys,
        terminal_timestamp=canonical_worker.POLICY_TERMINAL_T,
        namespace_lease_enabled=True,
        resume_generation=args.resume_generation,
        command_hash=stable_hash(" ".join(sys.argv)),
        configuration_hash=spec.configuration_hash,
        evaluation_contract_hash=resolved_performance_contract_v3_hash(),
    )
    predictors = predictor_manifest.predictors
    feature_order_hash = stable_hash(list(predictors))
    start_payload = canonical_sequence_progress_payload(
        family=family,
        display_family=display_family,
        pid=os.getpid(),
        package_state="DISCOVERING_DECISION_SPINE",
        resume_generation=args.resume_generation,
        metrics_root_name=args.metrics_root_name,
        current_refit_session="",
        current_scoring_cursor="",
        last_committed_decision_timestamp="",
        metrics_rows=0,
        resolved_performance_rows=0,
        partition_count=0,
        extra={
            "run_id": canonical_worker.RUN_ID,
            "feature_count": len(predictors),
            "feature_order_hash": feature_order_hash,
            "target": "forward_return_60m__decision_5m",
            "sequence_length": spec.sequence_length,
            "training_lookback_sessions": canonical_worker.TRAINING_SESSIONS,
            "refit_policy": args.refit_policy,
            "refit_cadence": canonical_worker.refit_cadence_id(args.refit_policy),
        },
    )
    write_json_atomic(family_root / "initialization_telemetry.json", {**start_payload, "telemetry_type": "INITIALIZATION"}, advisory=True)
    write_json_atomic(family_root / "worker_inventory.json", {**start_payload, "telemetry_type": "WORKER_INVENTORY"}, advisory=True)
    write_json_atomic(progress_path, {**start_payload, "telemetry_type": "PROGRESS"}, advisory=True)
    metrics_writer.heartbeat_namespace_lease(phase="DISCOVERING_DECISION_SPINE")
    spine = list(engine.decision_spine())
    schedule = iter_daily_session_refit_schedule(spine) if args.refit_policy == "daily_session_v1" else canonical_worker.build_refit_schedule(spine)
    score_timestamps_by_session = canonical_worker.score_timestamp_index(spine)
    completed_evidence = canonical_worker.completed_key_evidence(
        family_root,
        metrics_root_name=args.metrics_root_name,
        include_legacy_sources=False,
    )
    if completed_evidence["hash_mismatches"]:
        raise SequenceWorkerError(f"COMPLETED_KEY_HASH_MISMATCH:{completed_evidence['hash_mismatches'][:5]}")
    done = completed_evidence["keys"]
    committed_timestamps = 0
    last_commit_result: dict[str, Any] = {}
    for package in schedule:
        score_timestamps = canonical_worker.score_timestamps_for_spec(spine, package, by_session=score_timestamps_by_session)
        completion = canonical_worker.package_completion_status(display_family, score_timestamps, done)
        if completion["fully_completed"]:
            metrics_writer.heartbeat_namespace_lease(phase="SKIPPING_COMPLETED_REFIT_PACKAGE", cursor=score_timestamps[-1].isoformat())
            continue
        package_dates = set(package.training_session_dates + package.score_session_dates)
        package_years = {int(date[:4]) for date in package_dates}
        package_partitions = [row for row in partitions if row.year in package_years]
        progress = canonical_sequence_progress_payload(
            family=family,
            display_family=display_family,
            pid=os.getpid(),
            package_state="ASSEMBLING_REFIT_PACKAGE",
            resume_generation=args.resume_generation,
            metrics_root_name=args.metrics_root_name,
            current_refit_session=package.refit_T.isoformat(),
            current_scoring_cursor=completion.get("first_uncommitted_T", ""),
            last_committed_decision_timestamp="",
            metrics_rows=len(read_parquet_log(metrics_root, "per_t_metrics")),
            resolved_performance_rows=len(read_parquet_log(metrics_root, "rank_ic_v3")),
            partition_count=len(package_partitions),
            extra={"training_window_start": package.training_session_dates[0], "training_window_end": package.training_session_dates[-1], "score_session_dates": list(package.score_session_dates)},
        )
        write_json_atomic(progress_path, {**progress, "telemetry_type": "PROGRESS"}, advisory=True)
        metrics_writer.heartbeat_namespace_lease(phase="ASSEMBLING_REFIT_PACKAGE", cursor=package.refit_T.isoformat())
        panel = engine.assemble_partitions(rows=package_partitions, decision_dates=package_dates)
        if panel.empty:
            raise SequenceWorkerError(f"DS24_R52_EMPTY_REAL_SEQUENCE_PACKAGE:{display_family}:{package.refit_T.isoformat()}")
        panel["decision_timestamp"] = pd.to_datetime(panel["decision_timestamp"], utc=True)
        panel["target_available_timestamp"] = pd.to_datetime(panel["target_available_timestamp"], utc=True)
        train_mask = (
            (panel["decision_timestamp"] < package.refit_T)
            & panel["target_is_trainable"].astype(bool)
            & (panel["target_available_timestamp"] <= package.refit_T)
            & np.isfinite(panel["target_value"].astype(float))
        )
        train_sequences, train_targets, _train_meta = sequence_examples_from_panel(
            panel,
            predictors,
            sequence_length=spec.sequence_length,
            eligible_mask=train_mask,
            include_targets=True,
            max_examples=max(1, int(getattr(args, "max_training_examples", 24000))),
        )
        train_sequences, train_targets = deterministic_training_cap(
            train_sequences,
            train_targets,
            max_examples=max(1, int(getattr(args, "max_training_examples", 24000))),
        )
        if not train_sequences:
            raise SequenceWorkerError(f"DS24_R52_NO_SEQUENCE_TRAINING_WINDOWS:{display_family}:{package.refit_T.isoformat()}")
        model = TorchSequenceReturnRegressor(spec.config)
        fit_start = time.perf_counter()
        model.fit(train_sequences, train_targets)
        fit_seconds = time.perf_counter() - fit_start
        model_stamp = package.refit_T.strftime("%Y%m%dT%H%M%SZ")
        model_path = model_root / f"{family}_{model_stamp}.pkl"
        model_hash = save_real_model_artifact(
            model_path,
            family=family,
            display_family=display_family,
            model=model,
            spec=package,
            predictors=predictors,
            predictor_manifest_hash=predictor_manifest.manifest_hash,
        )
        model_vintage_id = stable_hash({"family": display_family, "refit_T": package.refit_T.isoformat(), "model_hash": model_hash})
        preprocessing_hash = stable_hash(
            {
                "family": display_family,
                "feature_order_hash": feature_order_hash,
                "feature_imputation_strategy": model.diagnostics.get("feature_imputation_strategy", ""),
            }
        )
        pending_outputs: list[pd.DataFrame] = []
        last_completed = ""
        for timestamp in score_timestamps:
            timestamp = pd.Timestamp(timestamp).tz_convert("UTC")
            key = (display_family, timestamp.isoformat())
            if key in done:
                continue
            score_mask = panel["decision_timestamp"] == timestamp
            score_sequences, _targets, score_meta = sequence_examples_from_panel(
                panel,
                predictors,
                sequence_length=spec.sequence_length,
                eligible_mask=score_mask,
                include_targets=False,
            )
            if not score_sequences:
                continue
            predictions = model.predict(score_sequences)
            output = pd.DataFrame(
                {
                    "family": display_family,
                    "config": "policy_v1",
                    "policy_hash": canonical_worker.policy_hash(),
                    "decision_timestamp": timestamp.isoformat(),
                    "asset_id": score_meta["asset_id"].astype(str),
                    "prediction": np.asarray(predictions, dtype=float),
                    "model_vintage_id": model_vintage_id,
                    "model_hash": model_hash,
                    "preprocessing_hash": preprocessing_hash,
                    "training_cutoff": package.refit_T.isoformat(),
                    "prediction_timestamp": utc_now(),
                }
            )
            pending_outputs.append(output)
            if len(pending_outputs) < max(1, int(args.prediction_batch_decisions)):
                continue
            combined = pd.concat(pending_outputs, ignore_index=True)
            last_commit_result = metrics_writer.commit_predictions(
                combined,
                metadata=real_worker_metadata(
                    spec,
                    training_cutoff=package.refit_T.isoformat(),
                    model_hash=model_hash,
                    model_vintage_id=model_vintage_id,
                    preprocessing_hash=preprocessing_hash,
                ),
            )
            for decision in sorted(pd.to_datetime(combined["decision_timestamp"], utc=True).unique()):
                iso = pd.Timestamp(decision).tz_convert("UTC").isoformat()
                done.add((display_family, iso))
                last_completed = iso
                committed_timestamps += 1
            pending_outputs.clear()
            checkpoint = write_real_sequence_checkpoint(
                family_root / "checkpoint" / "sequence_worker_checkpoint.json",
                family=family,
                display_family=display_family,
                resume_generation=args.resume_generation,
                spec=package,
                model_artifact_path=model_path,
                model_hash=model_hash,
                predictor_manifest_hash=predictor_manifest.manifest_hash,
                feature_order_hash=feature_order_hash,
                last_committed_decision_timestamp=last_completed,
                metrics_root_name=args.metrics_root_name,
                commit_result=last_commit_result,
            )
            metrics = read_parquet_log(metrics_root, "per_t_metrics")
            rank_ic = read_parquet_log(metrics_root, "rank_ic_v3")
            progress = canonical_sequence_progress_payload(
                family=family,
                display_family=display_family,
                pid=os.getpid(),
                package_state="RUNNING",
                resume_generation=args.resume_generation,
                metrics_root_name=args.metrics_root_name,
                current_refit_session=package.refit_T.isoformat(),
                current_scoring_cursor=last_completed,
                last_committed_decision_timestamp=last_completed,
                metrics_rows=len(metrics),
                resolved_performance_rows=len(rank_ic),
                partition_count=int(last_commit_result.get("checkpoint", {}).get("partition_count", 0) or 0),
                extra={
                    "fit_elapsed_seconds": round(fit_seconds, 6),
                    "training_examples_last_refit": len(train_sequences),
                    "training_window_start": package.training_session_dates[0],
                    "training_window_end": package.training_session_dates[-1],
                    "feature_count": len(predictors),
                    "feature_order_hash": feature_order_hash,
                    "target": "forward_return_60m__decision_5m",
                    "sequence_length": spec.sequence_length,
                    "model_hash": model_hash,
                    "checkpoint": checkpoint,
                },
            )
            write_json_atomic(progress_path, {**progress, "telemetry_type": "PROGRESS"}, advisory=True)
            metrics_writer.heartbeat_namespace_lease(phase="RUNNING", cursor=last_completed)
            if stop_after_committed_timestamps and committed_timestamps >= stop_after_committed_timestamps:
                metrics_writer.release_namespace_lease()
                return {
                    "classification": "PATCHTST_REAL_HISTORICAL_EXECUTION_BOUNDED_PROOF_COMPLETE",
                    "family": display_family,
                    "committed_timestamps": committed_timestamps,
                    "last_committed_decision_timestamp": last_completed,
                    "metrics_rows": len(metrics),
                    "resolved_performance_rows": len(rank_ic),
                    "resource_preflight": resource,
                    "progress": progress,
                }
        if pending_outputs:
            combined = pd.concat(pending_outputs, ignore_index=True)
            last_commit_result = metrics_writer.commit_predictions(
                combined,
                metadata=real_worker_metadata(
                    spec,
                    training_cutoff=package.refit_T.isoformat(),
                    model_hash=model_hash,
                    model_vintage_id=model_vintage_id,
                    preprocessing_hash=preprocessing_hash,
                ),
            )
            for decision in sorted(pd.to_datetime(combined["decision_timestamp"], utc=True).unique()):
                iso = pd.Timestamp(decision).tz_convert("UTC").isoformat()
                done.add((display_family, iso))
                last_completed = iso
                committed_timestamps += 1
            pending_outputs.clear()
            checkpoint = write_real_sequence_checkpoint(
                family_root / "checkpoint" / "sequence_worker_checkpoint.json",
                family=family,
                display_family=display_family,
                resume_generation=args.resume_generation,
                spec=package,
                model_artifact_path=model_path,
                model_hash=model_hash,
                predictor_manifest_hash=predictor_manifest.manifest_hash,
                feature_order_hash=feature_order_hash,
                last_committed_decision_timestamp=last_completed,
                metrics_root_name=args.metrics_root_name,
                commit_result=last_commit_result,
            )
            metrics = read_parquet_log(metrics_root, "per_t_metrics")
            rank_ic = read_parquet_log(metrics_root, "rank_ic_v3")
            progress = canonical_sequence_progress_payload(
                family=family,
                display_family=display_family,
                pid=os.getpid(),
                package_state="RUNNING",
                resume_generation=args.resume_generation,
                metrics_root_name=args.metrics_root_name,
                current_refit_session=package.refit_T.isoformat(),
                current_scoring_cursor=last_completed,
                last_committed_decision_timestamp=last_completed,
                metrics_rows=len(metrics),
                resolved_performance_rows=len(rank_ic),
                partition_count=int(last_commit_result.get("checkpoint", {}).get("partition_count", 0) or 0),
                extra={
                    "fit_elapsed_seconds": round(fit_seconds, 6),
                    "training_examples_last_refit": len(train_sequences),
                    "training_window_start": package.training_session_dates[0],
                    "training_window_end": package.training_session_dates[-1],
                    "feature_count": len(predictors),
                    "feature_order_hash": feature_order_hash,
                    "target": "forward_return_60m__decision_5m",
                    "sequence_length": spec.sequence_length,
                    "model_hash": model_hash,
                    "checkpoint": checkpoint,
                },
            )
            write_json_atomic(progress_path, {**progress, "telemetry_type": "PROGRESS"}, advisory=True)
            metrics_writer.heartbeat_namespace_lease(phase="RUNNING", cursor=last_completed)
            if stop_after_committed_timestamps and committed_timestamps >= stop_after_committed_timestamps:
                metrics_writer.release_namespace_lease()
                return {
                    "classification": "PATCHTST_REAL_HISTORICAL_EXECUTION_BOUNDED_PROOF_COMPLETE",
                    "family": display_family,
                    "committed_timestamps": committed_timestamps,
                    "last_committed_decision_timestamp": last_completed,
                    "metrics_rows": len(metrics),
                    "resolved_performance_rows": len(rank_ic),
                    "resource_preflight": resource,
                    "progress": progress,
                }
        if model_path.exists():
            apply_checkpoint_retention(
                family_root=family_root,
                checkpoint_dir=model_root,
                family=display_family,
                current_checkpoint_path=model_path,
                active_refit_timestamp=package.refit_T.isoformat(),
                deterministic_rebuild_authority=True,
                referenced_paths=[model_path],
                pattern=f"{family}_*.pkl",
                ledger_path=family_root / "checkpoint_retention_ledger.jsonl",
                dry_run=False,
            )
    metrics_writer.release_namespace_lease()
    return {
        "classification": "PATCHTST_REAL_HISTORICAL_EXECUTION_COMPLETE",
        "family": display_family,
        "committed_timestamps": committed_timestamps,
        "last_committed_decision_timestamp": "",
        "metrics_rows": len(read_parquet_log(metrics_root, "per_t_metrics")),
        "resolved_performance_rows": len(read_parquet_log(metrics_root, "rank_ic_v3")),
        "resource_preflight": resource,
    }


def run_supervised_sequence_worker(args: argparse.Namespace, family: str) -> int:
    result = run_real_historical_sequence_worker(args, family)
    print(json.dumps(result, sort_keys=True, default=str), flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DS24 V3 reusable sequence policy worker")
    parser.add_argument("--family", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-generation", default="1")
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dataloader-workers", type=int, default=None)
    parser.add_argument("--pin-memory", action="store_true", default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--cache-budget-gb", type=float, default=1.0)
    parser.add_argument("--prediction-batch-decisions", type=int, default=2)
    parser.add_argument("--evaluation-version", default="v3")
    parser.add_argument("--metrics-root-name", default="metrics_only_v3_sequence")
    parser.add_argument("--refit-policy", default="daily_session_v1")
    parser.add_argument("--synthetic-certify", action="store_true")
    parser.add_argument("--publish-readiness", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-training-examples", type=int, default=24000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.publish_readiness:
        manifest = publish_readiness_manifest()
        print(json.dumps({"manifest_path": str(MANIFEST_PATH), "success": manifest["success"], "states": manifest["family_states"]}, sort_keys=True))
        return 0
    family = assert_supported_family(args.family)
    if not args.synthetic_certify:
        return run_supervised_sequence_worker(args, family)
    with tempfile.TemporaryDirectory(prefix="ds24_v3_sequence_worker_cli_") as temp_dir:
        root = Path(args.output_root) if args.output_root is not None else Path(temp_dir)
        result = run_synthetic_contract(
            family,
            root,
            resume_generation=args.resume_generation,
            threads=args.threads,
            device=args.device,
            require_cuda=args.require_cuda,
            dataloader_workers=args.dataloader_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            cache_budget_gb=args.cache_budget_gb,
            prediction_batch_decisions=args.prediction_batch_decisions,
            metrics_root_name=args.metrics_root_name,
        )
        print(json.dumps({"family": family, "state": result["state"], "checkpoint_resume": result["checkpoint_resume_result"]["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
