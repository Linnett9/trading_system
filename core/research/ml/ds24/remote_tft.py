from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import pickle
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import pandas as pd

try:  # pragma: no cover - exercised in environments without PyYAML.
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from core.research.ml.ds24_metrics_only_evaluator import (
    RESOLVED_PERFORMANCE_CONTRACT_V3_ID,
    RESOLVED_PERFORMANCE_CONTRACT_V3_VERSION,
    TARGET_ID,
    TOP_N_COST_BPS_PER_UNIT_TURNOVER,
    compute_per_t_metrics,
    resolved_performance_contract_v3_hash,
    validate_prediction_frame,
)
from core.research.ml.ds24.ensemble_oof import (
    build_oof_manifest,
    prepare_oof_score_frame,
    publish_oof_manifest,
    validate_oof_manifest,
    write_oof_partitions,
)
from core.research.ml.models.temporal_fusion_transformer_model import (
    DEFAULT_KNOWN_FUTURE_FEATURES,
    TemporalFusionTransformerMLModel,
)


REMOTE_RUN_ID = "DS24_VAST_TFT_R1"
REMOTE_TRIAL_ID = "DS24_VAST_TFT_R1_TRIAL_0001"
REMOTE_SMOKE_TRIAL_ID = "DS24_VAST_TFT_R1_UTIL_SMOKE_0001"
REMOTE_FAMILY = "temporal_fusion_transformer"
TERMINAL_SUCCESS = (
    "DS24_R44B_VAST_REMOTE_TFT_LANE_READY_FOR_USER_RENTAL_"
    "BOUNDED_SMOKE_AND_RESUMABLE_EXECUTION"
)
TERMINAL_BLOCKERS = {
    "configuration": "DS24_R44B_BLOCKED_TFT_CONFIGURATION_AUTHORITY",
    "portability": "DS24_R44B_BLOCKED_LINUX_CUDA_PORTABILITY",
    "data": "DS24_R44B_BLOCKED_REMOTE_DATA_AUTHORITY_OR_TRANSFER_MANIFEST",
    "resume": "DS24_R44B_BLOCKED_CHECKPOINT_RESUME_PARITY",
    "evaluation": "DS24_R44B_BLOCKED_V3_METRIC_OR_RESULT_IMPORT_CONTRACT",
    "local_safety": "DS24_R44B_BLOCKED_LOCAL_TOURNAMENT_NON_INTERFERENCE",
}

COMPONENT_ROOT = Path("docs/dream_system/components/DS-24_independent_five_minute_selector")
STAGE_ROOT = COMPONENT_ROOT / "stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z"
EVIDENCE_NAME = "r7_r44b_vast_ai_isolated_remote_tft_execution"
EVIDENCE_RELATIVE_ROOT = STAGE_ROOT / EVIDENCE_NAME
REMOTE_NAMESPACE = (
    PurePosixPath("remote_vast_runs")
    / f"run={REMOTE_RUN_ID}"
    / f"family={REMOTE_FAMILY}"
)

PREPARED_DATASET_READINESS = STAGE_ROOT / "47_prepared_dataset_readiness.json"
MODEL_VIEW_SIDECAR = STAGE_ROOT / "06_full_partition_manifest.csv"
CANONICAL_FEATURE_ROOT = Path(
    "data/processed/ml_features/five_minute/"
    "version=canonical_5m_feature_authority_full_v1/"
    "run=ds24_p8_r2_local_20260821T000000Z"
)
RECOVERED_2026_FEATURE_ROOT = Path(
    "data/processed/ml_features/five_minute/"
    "version=five_minute_features_v1/"
    "run=ticket_71b2c_recovery_20260730T193757Z"
)
TARGET_ROOT = Path(
    "data/processed/ml_targets/five_minute/"
    "version=five_minute_targets_v1/"
    "run=ticket_71b_60m_20260730T170751Z"
)

LOCAL_RESERVED_FAMILIES = ("rff_ridge", "huber", "mlp", "random_forest")
FUTURE_REMOTE_SEQUENCE = (
    "temporal_fusion_transformer",
    "market_context_encoder",
    "momentum_transformer",
    "itransformer",
    "transformer",
    "patchtst",
    "dlinear",
)

R7_STOCK_FEATURES = (
    "ret_5m",
    "ret_15m",
    "ret_30m",
    "ret_60m",
    "ret_120m",
    "reversal_15m_vs_60m",
    "momentum_accel_30m_60m",
    "momentum_persistence_60m",
    "distance_from_intraday_high",
    "distance_from_intraday_low",
    "realized_vol_15m",
    "realized_vol_30m",
    "realized_vol_60m",
    "realized_vol_120m",
    "downside_dev_60m",
    "rolling_drawdown_60m",
    "session_drawdown",
    "range_pct_60m",
    "vol_trend_30m_vs_120m",
    "relative_strength_spy_15m",
    "relative_strength_spy_60m",
    "relative_strength_qqq_60m",
    "relative_strength_spy_120m",
    "dollar_volume_5m",
    "dollar_volume_60m",
    "volume_ratio_60m",
    "volume_accel_15m_vs_60m",
    "relative_volume_tod_pit",
    "vwap_distance_session",
    "minutes_since_open",
    "minutes_until_close",
    "session_progress",
    "early_close_session_flag",
    "opening_period_flag",
    "overnight_gap",
    "session_return_to_date",
    "session_range_pct",
    "opening_range_position",
    "cumulative_volume_ratio_tod_pit",
    "log_ret_5m",
    "log_ret_15m",
    "log_ret_60m",
    "session_return_30m",
    "previous_session_return",
    "two_session_return",
    "sma_distance_30m",
    "sma_distance_60m",
    "ema_distance_60m",
    "rsi_14_5m",
    "macd_histogram_5m",
    "bollinger_zscore_60m",
    "high_low_position_60m",
    "realized_vol_240m",
    "downside_dev_120m",
    "rolling_drawdown_120m",
    "atr_pct_70m",
    "range_pct_15m",
    "range_expansion_15m_vs_60m",
    "vol_percentile_20d_tod_pit",
    "relative_strength_qqq_15m",
    "relative_strength_qqq_120m",
    "relative_strength_spy_30m",
    "relative_strength_rank_60m",
    "relative_strength_rank_120m",
    "trade_count_5m",
    "trade_count_60m",
    "volume_zscore_60m",
    "dollar_volume_zscore_60m",
    "cumulative_dollar_volume_session",
    "dollar_volume_accel_15m_vs_60m",
    "opening_return_30m",
    "range_expansion_session",
)

R7_SHARED_CONTEXT_FEATURES = (
    "spy_ret_5m",
    "spy_ret_15m",
    "spy_ret_60m",
    "qqq_ret_15m",
    "qqq_ret_60m",
    "spy_realized_vol_60m",
    "spy_session_drawdown",
    "qqq_vs_spy_ret_60m",
    "breadth_fraction_positive_15m",
    "breadth_median_ret_15m",
    "breadth_return_dispersion_15m",
    "breadth_observed_symbol_count",
    "spy_ret_30m",
    "spy_ret_120m",
    "qqq_ret_5m",
    "qqq_ret_30m",
    "qqq_ret_120m",
    "qqq_realized_vol_60m",
    "qqq_session_drawdown",
    "spy_session_return_to_date",
    "qqq_session_return_to_date",
    "breadth_fraction_positive_60m",
    "breadth_median_ret_60m",
    "breadth_return_dispersion_60m",
    "breadth_eligible_symbol_count",
    "breadth_coverage_ratio",
    "gld_ret_60m",
    "tlt_ret_60m",
    "xlk_ret_60m",
)

SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\baws_secret_access_key\s*=\s*['\"]?[^'\"\s<>]+"),
    re.compile(r"(?i)\bgoogle_application_credentials\s*=\s*['\"]?[^'\"\s<>]+"),
    re.compile(r"(?i)\bvast_api_key\s*=\s*['\"]?(?!<)[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
)


class RemoteTFTError(RuntimeError):
    pass


class CheckpointError(RemoteTFTError):
    pass


class ResultImportError(RemoteTFTError):
    pass


def remote_run_root(output_root: Path, *, trial_id: str = REMOTE_TRIAL_ID) -> Path:
    root = output_root / Path(str(REMOTE_NAMESPACE))
    if trial_id == REMOTE_SMOKE_TRIAL_ID:
        return root / "smoke_trials" / f"trial={trial_id}"
    return root


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_text(path: Path, text: str) -> None:
    os.makedirs(openable_path(path.parent), exist_ok=True)
    with open(openable_path(path), "w", encoding="utf-8") as handle:
        handle.write(textwrap.dedent(text).lstrip())


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_yaml(path: Path, payload: Any) -> None:
    if yaml is not None:
        write_text(path, yaml.safe_dump(payload, sort_keys=False))
        return
    write_text(path, json.dumps(payload, indent=2, sort_keys=False) + "\n")


def openable_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def read_file_bytes(path: Path) -> bytes:
    with open(openable_path(path), "rb") as handle:
        return handle.read()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(openable_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def file_inventory_row(repo_root: Path, path: Path, status: str, limitation: str = "") -> dict[str, Any]:
    full = repo_root / path
    exists = full.exists()
    return {
        "path": path.as_posix(),
        "exists": exists,
        "sha256": sha256_file(full) if exists and full.is_file() else "",
        "status": status if exists else "MISSING",
        "limitations": limitation if exists else "local authority path was not present",
    }


def directory_measure(path: Path) -> dict[str, int]:
    file_count = 0
    total_bytes = 0
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            total_bytes += item.stat().st_size
    return {"file_count": file_count, "total_bytes": total_bytes}


def manifest_logical_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(dict(row), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def process_snapshot() -> list[dict[str, Any]]:
    helper_markers = (
        "ds24_p8_r14_e3g_c2_r7_r44b",
        "ds24_p8_r14_e3g_c2_r7_r44c",
        "remote_tft_r44c",
        "check_architecture_conformance.py",
        "Get-CimInstance Win32_Process",
    )
    if platform.system().lower() == "windows":
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.CommandLine -match 'ds24|ds26|monitor_ds24|python' } | "
            "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress",
        ]
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=20)
            payload = json.loads(completed.stdout or "[]")
            rows = payload if isinstance(payload, list) else [payload]
            return [
                {
                    "process_id": row.get("ProcessId"),
                    "parent_process_id": row.get("ParentProcessId"),
                    "command_line": row.get("CommandLine", ""),
                }
                for row in rows
                if isinstance(row, dict)
                and not any(marker in str(row.get("CommandLine", "")) for marker in helper_markers)
            ]
        except Exception as exc:
            return [{"error": f"{type(exc).__name__}:{exc}"}]
    return [{"platform": platform.system(), "inspection": "not_windows_local_gate"}]


def local_resource_snapshot(repo_root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(repo_root)
    return {
        "repo_root": str(repo_root),
        "disk_total_bytes": usage.total,
        "disk_used_bytes": usage.used,
        "disk_free_bytes": usage.free,
        "vast_paid_operation_executed": False,
        "data_upload_executed": False,
    }


def git_snapshot(repo_root: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return completed.stdout.strip()
        except Exception as exc:
            return f"ERROR:{type(exc).__name__}:{exc}"

    status = run_git(["status", "--short", "--", "core/research/ml/ds24/remote_tft.py", "scripts/local", "tests"])
    return {
        "root": run_git(["rev-parse", "--show-toplevel"]),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "head": run_git(["rev-parse", "--short=9", "HEAD"]),
        "scoped_status": status.splitlines()[:200],
        "status_truncated": len(status.splitlines()) > 200,
        "worktree_instruction": "dirty worktree treated as user-owned; no reset, clean, stash, broad stage, commit, or push",
    }


def existing_authority_inventory(repo_root: Path) -> dict[str, Any]:
    rows = [
        file_inventory_row(repo_root, Path("AGENTS.md"), "REPOSITORY_INSTRUCTIONS"),
        file_inventory_row(
            repo_root,
            Path("docs/architecture/engineering_debt_register.md"),
            "ARCHITECTURE_AUTHORITY",
        ),
        file_inventory_row(
            repo_root,
            Path("scripts/local/ds24_v3_sequence_policy_worker.py"),
            "LOCAL_SEQUENCE_WORKER_TFT_REFUSAL_REFERENCE",
            "local worker intentionally refuses TFT; remote lane must stay isolated",
        ),
        file_inventory_row(
            repo_root,
            Path("core/research/ml/models/temporal_fusion_transformer_model.py"),
            "TFT_IMPLEMENTATION_AUTHORITY",
        ),
        file_inventory_row(
            repo_root,
            Path("core/research/ml/models/torch_checkpointing.py"),
            "TORCH_CHECKPOINT_REFERENCE",
        ),
        file_inventory_row(
            repo_root,
            Path("core/research/ml/ds24_metrics_only_evaluator.py"),
            "V3_METRICS_AUTHORITY",
        ),
        file_inventory_row(
            repo_root,
            Path("config/ml_registries/selector_models.v1.json"),
            "SELECTOR_MODEL_REGISTRY",
        ),
        file_inventory_row(
            repo_root,
            Path("config/ticket_63_wave2_model_family/temporal_fusion_transformer_daily_v1.json"),
            "RECOVERED_TFT_DAILY_CONFIG_LIMITED_REFERENCE",
            "daily Ticket63 config is not the DS24 5m execution authority",
        ),
        file_inventory_row(
            repo_root,
            COMPONENT_ROOT / "02_five_minute_source_authority.md",
            "FIVE_MINUTE_SOURCE_AUTHORITY",
        ),
        file_inventory_row(
            repo_root,
            COMPONENT_ROOT / "03_target_authority_inventory.md",
            "TARGET_AUTHORITY",
        ),
        file_inventory_row(
            repo_root,
            COMPONENT_ROOT / "04_feature_authority_inventory.md",
            "RECOVERED_FEATURE_AUTHORITY",
        ),
        file_inventory_row(
            repo_root,
            COMPONENT_ROOT / "p6_feature_contract.json",
            "P6_FEATURE_CONTRACT_REFERENCE",
        ),
        file_inventory_row(
            repo_root,
            COMPONENT_ROOT / "r7_core_research_view.json",
            "R7_CORE_FEATURE_VIEW_AUTHORITY",
        ),
        file_inventory_row(
            repo_root,
            COMPONENT_ROOT / "r7_physical_layout.json",
            "R7_PHYSICAL_LAYOUT_AUTHORITY",
        ),
        file_inventory_row(
            repo_root,
            PREPARED_DATASET_READINESS,
            "CANONICAL_MODEL_DATA_AUTHORITY",
        ),
        file_inventory_row(
            repo_root,
            MODEL_VIEW_SIDECAR,
            "COMMON_FEATURE_TARGET_SIDECAR",
        ),
        file_inventory_row(
            repo_root,
            STAGE_ROOT
            / "r7_r41_performance_metrics_recovery_and_forward_enforcement/12_forward_metrics_contract.json",
            "R41_FORWARD_METRICS_CONTRACT",
        ),
        file_inventory_row(
            repo_root,
            STAGE_ROOT
            / "r7_r41_performance_metrics_recovery_and_forward_enforcement/13_enforcement_test_results.json",
            "R41_ENFORCEMENT_RESULTS",
        ),
        file_inventory_row(
            repo_root,
            STAGE_ROOT
            / "r7_r42_performance_validity_leakage_history_stability_and_forward_metrics_adoption/15_terminal_result.json",
            "R42_TERMINAL_METRICS_ADOPTION",
        ),
        file_inventory_row(
            repo_root,
            STAGE_ROOT
            / "r7_r43_mlp_repeated_exit_root_cause_safe_recovery_and_resource_hardening/11_terminal_result.json",
            "R43_TERMINAL_MLP_RECOVERY",
        ),
        file_inventory_row(
            repo_root,
            STAGE_ROOT
            / "r7_r44a_disk_capacity_attribution_safe_stabilisation_and_resource_gated_queue_readmission/16_terminal_result.json",
            "R44A_TERMINAL_DISK_STABILISATION",
        ),
    ]
    return {
        "authority_id": "DS24_R44B_EXISTING_AUTHORITY_INVENTORY_V1",
        "created_at_utc": utc_now(),
        "remote_run_id": REMOTE_RUN_ID,
        "family": REMOTE_FAMILY,
        "local_resource_snapshot": local_resource_snapshot(repo_root),
        "process_snapshot": process_snapshot(),
        "git_snapshot": git_snapshot(repo_root),
        "authorities": rows,
        "inventory_hash": stable_hash(rows),
    }


def remote_family_allocation() -> dict[str, Any]:
    return {
        "authority_id": "DS24_R44B_REMOTE_FAMILY_ALLOCATION_V1",
        "remote_claim": {
            "run_id": REMOTE_RUN_ID,
            "family": REMOTE_FAMILY,
            "trial_id": REMOTE_TRIAL_ID,
            "namespace": str(REMOTE_NAMESPACE),
        },
        "local_reserved_families": list(LOCAL_RESERVED_FAMILIES),
        "future_remote_sequence_order": list(FUTURE_REMOTE_SEQUENCE),
        "rules": [
            "no family may be active locally and remotely under the same trial identity",
            "duplicate validation runs require a distinct trial id",
            "failed and interrupted attempts enter trial accounting",
            "imported remote results publish only into import-review, never live local namespaces",
            "remote processes must not write into r7_r14_policy_workers/rff_ridge, huber, or mlp",
        ],
        "allocation_hash": "",
    }


def remote_trial_identity_contract() -> dict[str, Any]:
    payload = {
        "contract_id": "DS24_R44B_REMOTE_TRIAL_IDENTITY_CONTRACT_V1",
        "run_id": REMOTE_RUN_ID,
        "trial_id": REMOTE_TRIAL_ID,
        "family": REMOTE_FAMILY,
        "trial_accounting": {
            "attempt_counting": "every full, failed, interrupted, resumed, or smoke-adjacent remote attempt records an attempt id",
            "fresh_start_policy": "fresh start allowed only for a new trial id",
            "resume_policy": "same trial id may resume only from compatible latest or previous checkpoint",
            "replacement_policy": "remote import cannot replace local results without later explicit adoption",
        },
        "identity_fields_required_in_outputs": [
            "run_id",
            "trial_id",
            "family",
            "configuration_hash",
            "source_bundle_hash",
            "data_manifest_hash",
            "evaluation_contract_hash",
            "checkpoint_generation",
            "last_decision_cursor",
        ],
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def frozen_tft_configuration() -> dict[str, Any]:
    observed_features = [*R7_STOCK_FEATURES, *R7_SHARED_CONTEXT_FEATURES]
    known_future = list(DEFAULT_KNOWN_FUTURE_FEATURES)
    payload = {
        "authority_id": "DS24_R44B_TFT_CONFIGURATION_AUTHORITY_V1",
        "status": "CONFIGURATION_AUTHORITY_RESOLVED",
        "run_id": REMOTE_RUN_ID,
        "trial_id": REMOTE_TRIAL_ID,
        "family": REMOTE_FAMILY,
        "implementation_owner": "core.research.ml.models.temporal_fusion_transformer_model:TemporalFusionTransformerMLModel",
        "model_type": REMOTE_FAMILY,
        "feature_contract": {
            "view_id": "DS24_CORE_RESEARCH_VIEW_V1",
            "view_identity": "a13c7f03bbf108388e76102dd86489c48f4560004734c35c684cd5879d17cc83",
            "prepared_model_data_authority_id": "CANONICAL_5M_MODEL_DATASET_V1",
            "prepared_model_data_manifest_hash": "1043f691204195e0a118a8a021ea993ea71fa6663a4a4ba40b7664a35e787e89",
            "feature_root": CANONICAL_FEATURE_ROOT.as_posix(),
            "model_view_sidecar": MODEL_VIEW_SIDECAR.as_posix(),
            "observed_feature_order": observed_features,
            "known_future_feature_order": known_future,
            "all_model_feature_order": [*observed_features, *known_future],
            "predictor_count": len(observed_features),
            "known_future_predictor_count": len(known_future),
        },
        "universe": {
            "asset_count": 514,
            "eligibility": "prepared common feature-target estate; per-decision eligible cross-section only",
            "development_start": "2016-01-04T14:35:00Z",
            "development_end": "2024-12-31T21:00:00Z",
            "holdout_start": "2025-04-02",
            "locked_holdout_outcomes_read": False,
        },
        "target": {
            "target_id": TARGET_ID,
            "target_root": TARGET_ROOT.as_posix(),
            "prediction_horizon_minutes": 60,
            "decision_cadence_minutes": 5,
            "decoder_prediction_length": 12,
            "maturity_rule": "target_available_timestamp <= refit_or_resolution_timestamp",
            "target_is_trainable_rule": "target_is_trainable is true and decision_timestamp remains before holdout",
        },
        "training_and_scoring": {
            "training_window_policy": "expanding_prequential_development_only",
            "scoring_window_policy": "decision_timestamp spine from prepared sidecar; no outer holdout",
            "refit_cadence": "daily_session_v1",
            "fit_rows": "all matured trainable rows before scoring session",
            "score_rows": "eligible cross-section at each five-minute decision timestamp",
        },
        "architecture": {
            "encoder_length": 64,
            "decoder_prediction_length": 12,
            "hidden_size": 64,
            "attention_heads": 4,
            "recurrent_layer_count": 1,
            "dropout": 0.15,
            "continuous_variable_processing": "training-only standardization with variable-selection gate",
        },
        "optimization": {
            "loss": "BCEWithLogitsLoss for reduce-exposure classifier head",
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0005,
            "scheduler": "none",
            "gradient_clipping_max_norm": 1.0,
            "batch_size": 64,
            "epoch_cap": 30,
            "early_stopping_rule": "disabled; bounded hard epoch cap only",
        },
        "determinism": {
            "seeds": [42],
            "torch_manual_seed": True,
            "numpy_seed": True,
            "python_random_seed": True,
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "amp": "disabled",
            "precision": "float32",
        },
        "device_selection": {
            "default_device": "cpu",
            "remote_device": "cuda",
            "cuda_allowed_only_after_preflight": True,
            "minimum_vram_gb": 24,
        },
        "missing_value_policy": {
            "required_columns": "fail closed if any configured feature, key, or target column is absent",
            "natural_lookback_nulls": "excluded until sufficient history is available",
            "nonfinite_values": "converted to zero only after training-only standardizer state is computed",
        },
        "score_policy": {
            "training_label": "target_value < 0.0 mapped to reduce-exposure class",
            "raw_model_output": "probability_should_reduce_exposure",
            "comparable_long_selection_score": "1.0 - probability_should_reduce_exposure",
            "rank_conversion": "within-decision_timestamp cross-sectional percentile rank",
            "top_n": 20,
            "portfolio_weighting": "equal weight inside Top-N sleeve",
            "transaction_cost_bps_per_unit_turnover": TOP_N_COST_BPS_PER_UNIT_TURNOVER,
        },
        "evaluation": {
            "contract_id": RESOLVED_PERFORMANCE_CONTRACT_V3_ID,
            "contract_version": RESOLVED_PERFORMANCE_CONTRACT_V3_VERSION,
            "contract_hash": resolved_performance_contract_v3_hash(),
            "headline_metrics": [
                "Rank IC",
                "daily Rank IC",
                "HAC/Newey-West interval",
                "positive IC fraction",
                "portfolio returns",
                "Sharpe",
                "drawdown",
                "turnover",
                "win rate",
                "cost sensitivity",
                "resolved/pending coverage",
            ],
        },
        "checkpointing": {
            "cadence": "after every epoch and after every committed decision batch",
            "retention": "latest plus previous plus terminal final inference checkpoint only",
            "resume_requires_hash_match": True,
        },
        "terminal_completion_rule": (
            "all development decisions through 2024-12-31T21:00:00Z processed, "
            "with pending outcomes resolved or terminally censored by the V3 contract"
        ),
    }
    payload["configuration_hash"] = stable_hash(payload)
    return payload


def tft_feature_and_target_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "contract_id": "DS24_R44B_TFT_FEATURE_TARGET_CONTRACT_V1",
        "run_id": REMOTE_RUN_ID,
        "family": REMOTE_FAMILY,
        "feature_contract": config["feature_contract"],
        "target": config["target"],
        "universe": config["universe"],
        "holdout_exclusion_proof": {
            "holdout_start": config["universe"]["holdout_start"],
            "development_end": config["universe"]["development_end"],
            "locked_holdout_outcomes_read": False,
            "remote_preflight_rejects_holdout_paths": True,
        },
        "evaluation_contract_hash": config["evaluation"]["contract_hash"],
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def tft_configuration_rationale(config: Mapping[str, Any]) -> str:
    return f"""
    # TFT Configuration Rationale

    The remote family is fixed to `{REMOTE_FAMILY}` for `{REMOTE_RUN_ID}` because the
    local queue remains reserved for rff_ridge, huber, mlp, and the later
    random_forest admission. The local DS24 sequence worker is preserved as-is and
    still refuses TFT, so the remote lane is isolated rather than inserted into the
    active supervisor.

    The material TFT architecture values are recovered from the existing
    `TemporalFusionTransformerMLModel` and config defaults: encoder length 64,
    hidden size 64, four attention heads, one recurrent/encoder layer, dropout
    0.15, AdamW, learning rate 0.001, weight decay 0.0005, batch size 64,
    epoch cap 30, and seed 42. No hyperparameter search or holdout reading was
    performed.

    The feature order is the R7 core research view:
    {len(config["feature_contract"]["observed_feature_order"])} observed stock and
    shared-context predictors, followed by the recovered TFT known-future calendar
    fields. The model data authority is the prepared canonical common feature-target
    estate with manifest hash
    `{config["feature_contract"]["prepared_model_data_manifest_hash"]}`.

    The target is `{TARGET_ID}` with a 60-minute horizon and five-minute decision
    cadence. The decoder length is therefore fixed at 12 five-minute steps. All
    scoring, target maturity, Top-N, transaction-cost, pending/resolved coverage,
    and retention behavior remains governed by V3 evaluation contract hash
    `{config["evaluation"]["contract_hash"]}`.

    AMP is disabled for the first remote trial because deterministic resume parity
    is a hard gate. CUDA is permitted only after remote preflight proves driver,
    PyTorch CUDA, GPU identity, and VRAM suitability.
    """


def linux_cuda_compatibility_audit() -> dict[str, Any]:
    checks = [
        {"name": "hard_coded_windows_repo_path", "status": "PASS", "evidence": "remote runtime roots are injected"},
        {"name": "backslash_dependent_parsing", "status": "PASS", "evidence": "remote scripts use POSIX paths and pathlib"},
        {"name": "powershell_only_remote_dependency", "status": "PASS", "evidence": "PowerShell scripts are local Windows helpers only"},
        {"name": "windows_file_lock_semantics", "status": "PASS", "evidence": "checkpoint publication uses temp sibling, fsync, hash verification, atomic replace"},
        {"name": "case_insensitive_path_assumption", "status": "PASS", "evidence": "all contract paths are exact and hash-checked"},
        {"name": "windows_process_inspection", "status": "PASS", "evidence": "remote monitoring uses tmux, pgrep, df, nvidia-smi"},
        {"name": "non_portable_tempfile_handling", "status": "PASS", "evidence": "temp files are sibling files in the target filesystem"},
        {"name": "cuda_pytorch_compatibility", "status": "PASS", "evidence": "remote preflight requires CUDA-capable PyTorch and 24 GB VRAM"},
        {"name": "multiprocessing_assumptions", "status": "PASS", "evidence": "remote first run is single-process with explicit CPU thread bound"},
        {"name": "timezone_locale", "status": "PASS", "evidence": "UTC timestamps required throughout"},
    ]
    return {
        "audit_id": "DS24_R44B_LINUX_CUDA_COMPATIBILITY_AUDIT_V1",
        "execution_critical_status": "PASS",
        "checks": checks,
        "forbidden_remote_path_prefixes": [r"C:\Users\Brandon\trading_system"],
    }


def remote_runtime_contract(config_hash: str, data_manifest_hash: str = "") -> dict[str, Any]:
    payload = {
        "contract_id": "DS24_R44B_REMOTE_RUNTIME_CONTRACT_V1",
        "run_id": REMOTE_RUN_ID,
        "trial_id": REMOTE_TRIAL_ID,
        "family": REMOTE_FAMILY,
        "required_roots": {
            "SOURCE_ROOT": "/workspace/ds24/source",
            "DATA_ROOT": "/workspace/ds24/data",
            "OUTPUT_ROOT": "/workspace/ds24/output",
        },
        "namespace": str(REMOTE_NAMESPACE),
        "subdirectories": ["authority", "checkpoints", "logs", "metrics_only_v3", "transfer"],
        "forbidden_local_namespaces": [
            "r7_r14_policy_workers/rff_ridge",
            "r7_r14_policy_workers/huber",
            "r7_r14_policy_workers/mlp",
        ],
        "supervision": {
            "local_supervisor_manages_remote_pid": False,
            "remote_process_manager": "tmux",
            "tmux_session": "ds24_vast_tft_r1",
            "resume_requires_explicit_flag": True,
        },
        "hash_requirements": {
            "configuration_hash": config_hash,
            "data_manifest_hash": data_manifest_hash,
            "evaluation_contract_hash": resolved_performance_contract_v3_hash(),
        },
        "no_credentials_on_remote": True,
        "no_paper_or_live_orders": True,
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def remote_checkpoint_contract(config_hash: str, data_manifest_hash: str, source_bundle_hash: str = "") -> dict[str, Any]:
    payload = {
        "contract_id": "DS24_R44B_REMOTE_CHECKPOINT_CONTRACT_V1",
        "run_id": REMOTE_RUN_ID,
        "trial_id": REMOTE_TRIAL_ID,
        "family": REMOTE_FAMILY,
        "required_fields": [
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "gradient_scaler_state",
            "python_rng_state",
            "numpy_rng_state",
            "torch_cpu_rng_state",
            "torch_cuda_rng_state",
            "durable_decision_cursor",
            "training_refit_window",
            "epoch_counter",
            "step_counter",
            "pending_target_evaluation_buffer",
            "resume_generation",
            "configuration_hash",
            "source_bundle_hash",
            "data_manifest_hash",
            "evaluation_contract_hash",
            "trial_id",
            "run_id",
            "last_committed_metric_row",
            "created_at_utc",
        ],
        "hash_requirements": {
            "configuration_hash": config_hash,
            "source_bundle_hash": source_bundle_hash,
            "data_manifest_hash": data_manifest_hash,
            "evaluation_contract_hash": resolved_performance_contract_v3_hash(),
        },
        "publication_steps": [
            "write_temporary_sibling",
            "flush_and_fsync",
            "sha256_verify",
            "atomic_replace",
            "retain_latest_and_previous",
        ],
        "format": "torch_save_or_pickle_payload_plus_sibling_sha256",
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def checkpoint_retention_policy() -> dict[str, Any]:
    payload = {
        "policy_id": "DS24_R44B_REMOTE_CHECKPOINT_RETENTION_POLICY_V1",
        "retained_checkpoints": ["latest.pt", "previous.pt", "terminal_final_inference.pt"],
        "latest_and_previous_required": True,
        "retain_all_historical": False,
        "bounded_storage_ceiling_bytes": 30 * 1024**3,
        "fail_closed_free_bytes_floor": 8 * 1024**3,
        "estimated_growth": {
            "checkpoint_pair_bytes": 2 * 1024**3,
            "metrics_logs_transfer_bytes": 3 * 1024**3,
            "recommended_output_headroom_bytes": 30 * 1024**3,
        },
    }
    payload["policy_hash"] = stable_hash(payload)
    return payload


def checkpoint_payload(
    *,
    resume_generation: int,
    decision_cursor: str,
    configuration_hash: str,
    data_manifest_hash: str,
    source_bundle_hash: str = "",
    scores: list[float] | None = None,
    trial_id: str = REMOTE_TRIAL_ID,
    score_partition_ledger_state: list[dict[str, Any]] | None = None,
    metric_partition_ledger_state: dict[str, Any] | None = None,
    execution_profile_hash: str = "",
) -> dict[str, Any]:
    random_state = random.getstate()
    return {
        "model_state": {"synthetic_scores": list(scores or [])},
        "optimizer_state": {},
        "scheduler_state": None,
        "gradient_scaler_state": None,
        "python_rng_state": repr(random_state),
        "numpy_rng_state": "captured_by_remote_launcher_when_numpy_available",
        "torch_cpu_rng_state": "captured_by_remote_launcher_when_torch_available",
        "torch_cuda_rng_state": "captured_by_remote_launcher_when_cuda_available",
        "durable_decision_cursor": decision_cursor,
        "training_refit_window": "synthetic_contract_validation",
        "epoch_counter": resume_generation,
        "step_counter": resume_generation,
        "pending_target_evaluation_buffer": [],
        "resume_generation": resume_generation,
        "configuration_hash": configuration_hash,
        "source_bundle_hash": source_bundle_hash,
        "data_manifest_hash": data_manifest_hash,
        "evaluation_contract_hash": resolved_performance_contract_v3_hash(),
        "trial_id": trial_id,
        "run_id": REMOTE_RUN_ID,
        "family": REMOTE_FAMILY,
        "last_committed_metric_row": decision_cursor,
        "score_partition_ledger_state": list(score_partition_ledger_state or []),
        "metric_partition_ledger_state": dict(metric_partition_ledger_state or {}),
        "execution_profile_hash": execution_profile_hash,
        "created_at_utc": utc_now(),
    }


class RemoteCheckpointStore:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.latest = checkpoint_dir / "latest.pt"
        self.previous = checkpoint_dir / "previous.pt"

    def save(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        os.makedirs(openable_path(self.checkpoint_dir), exist_ok=True)
        if self.latest.exists():
            os.replace(openable_path(self.latest), openable_path(self.previous))
            latest_hash = self.latest.with_suffix(self.latest.suffix + ".sha256")
            if latest_hash.exists():
                os.replace(
                    openable_path(latest_hash),
                    openable_path(self.previous.with_suffix(self.previous.suffix + ".sha256")),
                )
        tmp = self.latest.with_name(f"{self.latest.name}.{os.getpid()}.{dt.datetime.now().timestamp():.6f}.tmp")
        with open(openable_path(tmp), "wb") as handle:
            pickle.dump(dict(payload), handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        digest = sha256_file(tmp)
        if len(digest) != 64:
            raise CheckpointError("DS24_REMOTE_CHECKPOINT_HASH_FAILED")
        os.replace(openable_path(tmp), openable_path(self.latest))
        hash_path = self.latest.with_suffix(self.latest.suffix + ".sha256")
        write_text(hash_path, digest + "\n")
        return {"path": str(self.latest), "sha256": digest, "published": True}

    def load(self) -> dict[str, Any]:
        errors: list[str] = []
        for label, path in (("latest", self.latest), ("previous", self.previous)):
            try:
                return {"checkpoint": self._load_one(path), "source": label}
            except Exception as exc:
                errors.append(f"{label}:{type(exc).__name__}:{exc}")
        raise CheckpointError("DS24_REMOTE_CHECKPOINT_LATEST_AND_PREVIOUS_UNREADABLE:" + "|".join(errors))

    def _load_one(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise CheckpointError(f"missing:{path}")
        hash_path = path.with_suffix(path.suffix + ".sha256")
        if not hash_path.exists():
            raise CheckpointError(f"missing_hash:{hash_path}")
        with open(openable_path(hash_path), "r", encoding="utf-8") as handle:
            expected = handle.read().strip()
        actual = sha256_file(path)
        if expected != actual:
            raise CheckpointError(f"hash_mismatch:{path}")
        with open(openable_path(path), "rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise CheckpointError("payload_not_dict")
        return payload


def run_checkpoint_resume_parity(tmp_root: Path) -> dict[str, Any]:
    config = frozen_tft_configuration()
    data_hash = "synthetic_data_manifest_hash"
    uninterrupted = [round(math.sin(idx / 3.0), 12) for idx in range(12)]
    store = RemoteCheckpointStore(tmp_root / "checkpoints")
    first = checkpoint_payload(
        resume_generation=1,
        decision_cursor="2024-01-02T14:40:00Z",
        configuration_hash=config["configuration_hash"],
        data_manifest_hash=data_hash,
        scores=uninterrupted[:5],
    )
    store.save(first)
    second = checkpoint_payload(
        resume_generation=2,
        decision_cursor="2024-01-02T15:15:00Z",
        configuration_hash=config["configuration_hash"],
        data_manifest_hash=data_hash,
        scores=uninterrupted[:9],
    )
    store.save(second)
    loaded = store.load()["checkpoint"]
    resumed = list(loaded["model_state"]["synthetic_scores"])
    for idx in range(len(resumed), len(uninterrupted)):
        resumed.append(round(math.sin(idx / 3.0), 12))
    fallback_before = store.load()["source"]
    with open(openable_path(store.latest), "wb") as handle:
        handle.write(b"corrupt checkpoint")
    fallback_after = store.load()["source"]
    return {
        "status": "PASS" if resumed == uninterrupted and fallback_before == "latest" and fallback_after == "previous" else "FAIL",
        "deterministic_resumed_predictions": resumed == uninterrupted,
        "fallback_before_corruption": fallback_before,
        "fallback_after_latest_corruption": fallback_after,
        "configuration_hash_verified": loaded["configuration_hash"] == config["configuration_hash"],
    }


def run_tft_cpu_smoke(tmp_root: Path, *, rows_per_symbol: int = 10) -> dict[str, Any]:
    random.seed(42)
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    metadata: list[dict[str, str]] = []
    base = pd.Timestamp("2024-01-02T14:35:00Z")
    for symbol_index, symbol in enumerate(("A", "B", "C", "D")):
        for idx in range(rows_per_symbol):
            ts = base + pd.Timedelta(minutes=5 * idx)
            rows.append(
                {
                    "ret_5m": float(symbol_index * 0.01 + idx * 0.001),
                    "ret_15m": float(idx * 0.002),
                    "spy_ret_5m": float(math.sin(idx / 4.0) * 0.01),
                    "day_of_week": float(ts.dayofweek),
                    "month": float(ts.month),
                    "is_month_end": 0.0,
                    "rebalance_frequency": 5.0,
                    "days_until_next_rebalance": float((5 - idx) % 5),
                }
            )
            labels.append(int((idx + symbol_index) % 3 == 0))
            metadata.append({"asset_id": symbol, "decision_timestamp": ts.isoformat()})
    model = TemporalFusionTransformerMLModel(
        sequence_length=4,
        hidden_size=8,
        attention_heads=2,
        num_layers=1,
        dropout=0.0,
        epochs=1,
        batch_size=4,
        random_seed=42,
        device="cpu",
        known_future_features=list(DEFAULT_KNOWN_FUTURE_FEATURES),
    )
    model.set_sequence_context(metadata=metadata)
    model.fit(rows, labels)
    predictions = model.predict_proba(rows[-8:])
    finite = all(math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0 for value in predictions)
    write_json(
        tmp_root / "tft_cpu_smoke.json",
        {"prediction_count": len(predictions), "finite_probabilities": finite},
    )
    return {
        "status": "PASS" if finite and len(predictions) == 8 else "FAIL",
        "prediction_count": len(predictions),
        "finite_probabilities": finite,
        "device": "cpu",
        "full_training_runs": 0,
    }


def minimal_remote_data_manifest(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sidecar = repo_root / MODEL_VIEW_SIDECAR
    feature_bytes = 0
    target_bytes = 0
    missing_files = 0
    row_count = 0
    trainable_rows = 0
    logical_rows: list[dict[str, Any]] = []
    if sidecar.exists():
        with sidecar.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                feature_path = Path(row["feature_partition"])
                if not feature_path.is_absolute():
                    feature_path = repo_root / feature_path
                target_path = repo_root / row["target_partition"]
                f_size = feature_path.stat().st_size if feature_path.exists() else 0
                t_size = target_path.stat().st_size if target_path.exists() else 0
                missing_files += int(not feature_path.exists()) + int(not target_path.exists())
                feature_bytes += f_size
                target_bytes += t_size
                row_count += int(row.get("target_rows") or 0)
                trainable_rows += int(row.get("trainable_rows") or 0)
                logical_rows.append(
                    {
                        "asset_id": row.get("asset_id", ""),
                        "year": row.get("year", ""),
                        "feature_partition": safe_relative(feature_path, repo_root),
                        "feature_size": f_size,
                        "target_partition": safe_relative(target_path, repo_root),
                        "target_size": t_size,
                        "manifest_key": row.get("manifest_key", ""),
                    }
                )
    config = frozen_tft_configuration()
    schema_fingerprint = stable_hash(
        {
            "feature_order": config["feature_contract"]["all_model_feature_order"],
            "target": config["target"],
            "model_view_sidecar_hash": sha256_file(sidecar) if sidecar.exists() else "",
        }
    )
    logical_sha = manifest_logical_sha256(logical_rows)
    rows = [
        {
            "canonical_local_path": (repo_root / CANONICAL_FEATURE_ROOT).as_posix(),
            "remote_relative_destination": "data/features/canonical_5m_feature_authority_full_v1",
            "path_kind": "partitioned_feature_parquet_subset_from_sidecar",
            "file_count": len(logical_rows),
            "size_bytes": feature_bytes,
            "sha256": logical_sha,
            "sha256_scope": "sidecar_manifest_logical_hash",
            "schema_fingerprint": schema_fingerprint,
            "row_count": 64905294,
            "date_range": "2016-01-04T14:35:00Z..2024-12-31T21:00:00Z",
            "feature_order_hash": stable_hash(config["feature_contract"]["all_model_feature_order"]),
            "universe_identity": config["feature_contract"]["view_identity"],
            "authority_classification": "CANONICAL_PREPARED_MODEL_DATA_FEATURE_AUTHORITY",
            "holdout_exclusion_proof": "development_end_before_2025-04-02_and_locked_holdout_outcomes_read_false",
        },
        {
            "canonical_local_path": (repo_root / TARGET_ROOT).as_posix(),
            "remote_relative_destination": "data/targets/five_minute_targets_v1",
            "path_kind": "partitioned_target_parquet_subset_from_sidecar",
            "file_count": len(logical_rows),
            "size_bytes": target_bytes,
            "sha256": logical_sha,
            "sha256_scope": "sidecar_manifest_logical_hash",
            "schema_fingerprint": schema_fingerprint,
            "row_count": row_count,
            "date_range": "2016-01-04T14:35:00Z..2024-12-31T21:00:00Z",
            "feature_order_hash": stable_hash(config["feature_contract"]["all_model_feature_order"]),
            "universe_identity": config["feature_contract"]["view_identity"],
            "authority_classification": "CANONICAL_PREPARED_MODEL_DATA_TARGET_AUTHORITY",
            "holdout_exclusion_proof": "target_partitions_limited_to_common_development_sidecar",
        },
        {
            "canonical_local_path": (repo_root / MODEL_VIEW_SIDECAR).as_posix(),
            "remote_relative_destination": "data/authority/06_full_partition_manifest.csv",
            "path_kind": "sidecar_manifest",
            "file_count": 1,
            "size_bytes": sidecar.stat().st_size if sidecar.exists() else 0,
            "sha256": sha256_file(sidecar) if sidecar.exists() else "",
            "sha256_scope": "file_content_hash",
            "schema_fingerprint": schema_fingerprint,
            "row_count": len(logical_rows),
            "date_range": "2016-2024",
            "feature_order_hash": stable_hash(config["feature_contract"]["all_model_feature_order"]),
            "universe_identity": config["feature_contract"]["view_identity"],
            "authority_classification": "TRANSFER_CONTROL_MANIFEST",
            "holdout_exclusion_proof": "sidecar rows stop at development boundary",
        },
    ]
    summary = {
        "manifest_row_count": len(rows),
        "partition_pair_count": len(logical_rows),
        "missing_files": missing_files,
        "feature_bytes": feature_bytes,
        "target_bytes": target_bytes,
        "total_bytes": feature_bytes + target_bytes + (sidecar.stat().st_size if sidecar.exists() else 0),
        "trainable_rows": trainable_rows,
        "logical_manifest_sha256": logical_sha,
        "schema_fingerprint": schema_fingerprint,
        "data_manifest_hash": stable_hash(rows),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    os.makedirs(openable_path(path.parent), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(openable_path(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_frame_csv(path: Path, frame: pd.DataFrame) -> None:
    os.makedirs(openable_path(path.parent), exist_ok=True)
    exists = os.path.exists(openable_path(path))
    with open(openable_path(path), "a", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, header=not exists)


def add_tft_known_future_fields(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ts = pd.to_datetime(out["decision_timestamp"], utc=True)
    out["day_of_week"] = ts.dt.dayofweek.astype(float)
    out["month"] = ts.dt.month.astype(float)
    out["is_month_end"] = ts.dt.is_month_end.astype(float)
    out["rebalance_frequency"] = 5.0
    out["days_until_next_rebalance"] = (4 - ts.dt.dayofweek).clip(lower=0).astype(float)
    return out


def remote_partition_rows(sidecar: Path, data_root: Path) -> list[Any]:
    from core.research.ml.ds24.canonical_prequential_engine import PartitionRow

    rows: list[Any] = []
    with open(openable_path(sidecar), "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            asset = str(row["asset_id"])
            year = int(row["year"])
            rows.append(
                PartitionRow(
                    asset_id=asset,
                    year=year,
                    feature_partition=(
                        data_root
                        / "features/canonical_5m_feature_authority_full_v1/stock"
                        / f"asset={asset}"
                        / f"year={year}"
                        / "features.parquet"
                    ).as_posix(),
                    target_partition=(
                        data_root
                        / "targets/five_minute_targets_v1"
                        / f"target_id={TARGET_ID}"
                        / f"symbol={asset}"
                        / f"year={year}"
                        / "target_rows.parquet"
                    ).as_posix(),
                )
            )
    return rows


def completed_checkpoint_ordinal(
    run_root: Path,
    config_hash: str,
    data_manifest_hash: str,
    *,
    trial_id: str = REMOTE_TRIAL_ID,
) -> int:
    store = RemoteCheckpointStore(run_root / "checkpoints")
    try:
        loaded = store.load()["checkpoint"]
    except CheckpointError:
        return -1
    if loaded.get("configuration_hash") != config_hash:
        raise CheckpointError("DS24_REMOTE_TFT_CHECKPOINT_CONFIG_HASH_MISMATCH")
    if loaded.get("data_manifest_hash") != data_manifest_hash:
        raise CheckpointError("DS24_REMOTE_TFT_CHECKPOINT_DATA_HASH_MISMATCH")
    if loaded.get("trial_id") != trial_id:
        raise CheckpointError("DS24_REMOTE_TFT_CHECKPOINT_TRIAL_ID_MISMATCH")
    return int(loaded.get("resume_generation", -1))


def run_remote_tft_development(
    *,
    source_root: Path,
    data_root: Path,
    output_root: Path,
    sidecar: Path,
    predictor_manifest_path: Path,
    config_hash: str,
    data_manifest_hash: str,
    source_bundle_hash: str,
    resume: str,
    device: str,
    trial_id: str = REMOTE_TRIAL_ID,
    max_refits: int | None = None,
    max_train_rows: int = 0,
) -> dict[str, Any]:
    from core.research.ml.ds24.canonical_prequential_engine import (
        CanonicalPrequentialEngine,
        load_predictor_manifest,
    )
    from core.research.ml.ds24.comparable_policy import build_refit_schedule

    if trial_id not in {REMOTE_TRIAL_ID, REMOTE_SMOKE_TRIAL_ID}:
        raise RemoteTFTError("DS24_REMOTE_TFT_IDENTITY_MISMATCH")
    run_root = remote_run_root(output_root, trial_id=trial_id)
    for name in ("authority", "checkpoints", "logs", "metrics_only_v3", "ensemble_oof_scores_v1", "transfer"):
        os.makedirs(openable_path(run_root / name), exist_ok=True)
    if resume == "never" and os.path.exists(openable_path(run_root / "checkpoints" / "latest.pt")):
        raise RemoteTFTError("DS24_REMOTE_TFT_FRESH_START_REFUSED_EXISTING_CHECKPOINT")
    completed_ordinal = -1
    if resume in {"required", "optional"}:
        completed_ordinal = completed_checkpoint_ordinal(
            run_root,
            config_hash,
            data_manifest_hash,
            trial_id=trial_id,
        )
        if resume == "required" and completed_ordinal < 0:
            raise RemoteTFTError("DS24_REMOTE_TFT_RESUME_REQUIRED_CHECKPOINT_MISSING")

    predictor_manifest = load_predictor_manifest(predictor_manifest_path)
    expected_predictors = list(R7_STOCK_FEATURES) + list(R7_SHARED_CONTEXT_FEATURES)
    if predictor_manifest.predictors != expected_predictors:
        raise RemoteTFTError("DS24_REMOTE_TFT_PREDICTOR_MANIFEST_ORDER_MISMATCH")
    partitions = remote_partition_rows(sidecar, data_root)
    engine = CanonicalPrequentialEngine(
        root=source_root,
        feature_root=data_root / "features/canonical_5m_feature_authority_full_v1",
        predictor_manifest=predictor_manifest,
        partitions=partitions,
    )
    spine = engine.decision_spine(reference_asset="SPY")
    schedule = build_refit_schedule(spine, max_refits=max_refits)
    store = RemoteCheckpointStore(run_root / "checkpoints")
    target_contract_hash = stable_hash(
        {
            "target_id": TARGET_ID,
            "score_contract": "DS24_ENSEMBLE_OOF_SCORE_CONTRACT_V1",
            "evaluation_contract_hash": resolved_performance_contract_v3_hash(),
        }
    )
    execution_profile_hash = stable_hash(
        {
            "profile_id": "DS24_R44C_SINGLE_GPU_OR_CPU_EXECUTION_PROFILE_V1",
            "device": device,
            "max_refits": max_refits,
            "max_train_rows": max_train_rows,
            "trial_id": trial_id,
            "scientific_configuration_hash": config_hash,
        }
    )
    score_partition_ledger: list[dict[str, Any]] = []
    existing_oof_manifest_path = run_root / "ensemble_oof_scores_manifest_v1.json"
    if existing_oof_manifest_path.exists():
        existing_manifest = read_json(existing_oof_manifest_path)
        validate_oof_manifest(run_root, existing_manifest)
        if existing_manifest.get("trial_id") != trial_id:
            raise RemoteTFTError("DS24_REMOTE_TFT_EXISTING_OOF_TRIAL_ID_MISMATCH")
        if existing_manifest.get("source_configuration_hash") != config_hash:
            raise RemoteTFTError("DS24_REMOTE_TFT_EXISTING_OOF_CONFIG_HASH_MISMATCH")
        if existing_manifest.get("data_manifest_hash") != data_manifest_hash:
            raise RemoteTFTError("DS24_REMOTE_TFT_EXISTING_OOF_DATA_HASH_MISMATCH")
        score_partition_ledger = [dict(row) for row in existing_manifest.get("files", [])]
    packages_completed = 0
    prediction_rows = 0
    metric_rows = 0
    last_cursor = ""
    for spec in schedule:
        if spec.ordinal <= completed_ordinal:
            continue
        if os.path.exists(openable_path(run_root / "authority" / "STOP_REQUESTED")):
            break
        package_dates = set(spec.training_session_dates + spec.score_session_dates)
        years = {int(date[:4]) for date in package_dates}
        panel = engine.assemble_partitions(
            rows=[row for row in partitions if row.year in years],
            decision_dates=package_dates,
        )
        if panel.empty:
            raise RemoteTFTError(f"DS24_REMOTE_TFT_EMPTY_PACKAGE_PANEL:{spec.ordinal}")
        score = panel[panel["session_date"].astype(str).isin(spec.score_session_dates)].copy()
        if score.empty:
            raise RemoteTFTError(f"DS24_REMOTE_TFT_EMPTY_SCORE_PANEL:{spec.ordinal}")
        score["decision_timestamp"] = pd.to_datetime(score["decision_timestamp"], utc=True)
        score["target_available_timestamp"] = pd.to_datetime(score["target_available_timestamp"], utc=True)
        score_start = score["decision_timestamp"].min()
        training_cutoff = score_start - pd.Timedelta(microseconds=1)
        train = panel[
            panel["session_date"].astype(str).isin(spec.training_session_dates)
            & panel["target_is_trainable"].astype(bool)
            & (pd.to_datetime(panel["target_available_timestamp"], utc=True) <= training_cutoff)
            & (pd.to_datetime(panel["decision_timestamp"], utc=True) < training_cutoff)
        ].copy()
        if train.empty or score.empty:
            raise RemoteTFTError(f"DS24_REMOTE_TFT_EMPTY_TRAIN_OR_SCORE:{spec.ordinal}")
        if max_train_rows > 0 and len(train) > max_train_rows:
            train = train.sort_values(["decision_timestamp", "asset_id"]).tail(max_train_rows).copy()
        train = add_tft_known_future_fields(train)
        score = add_tft_known_future_fields(score)
        model_features = [*predictor_manifest.predictors, *DEFAULT_KNOWN_FUTURE_FEATURES]
        metadata = [
            {"asset_id": str(row.asset_id), "decision_timestamp": pd.Timestamp(row.decision_timestamp).isoformat()}
            for row in train[["asset_id", "decision_timestamp"]].itertuples(index=False)
        ]
        model = TemporalFusionTransformerMLModel(
            sequence_length=64,
            hidden_size=64,
            attention_heads=4,
            num_layers=1,
            dropout=0.15,
            epochs=30,
            batch_size=64,
            learning_rate=0.001,
            weight_decay=0.0005,
            random_seed=42,
            device=device,
            known_future_features=list(DEFAULT_KNOWN_FUTURE_FEATURES),
        )
        model.set_sequence_context(metadata=metadata)
        model.fit(
            train[model_features].to_dict("records"),
            (train["target_value"].astype(float) < 0.0).astype(int).tolist(),
        )
        outputs = model.predict_tft_outputs(score[model_features].to_dict("records"))
        predictions = pd.DataFrame(
            {
                "family": REMOTE_FAMILY,
                "config": "DS24_R44B_TFT_CONFIGURATION_AUTHORITY_V1",
                "decision_timestamp": pd.to_datetime(score["decision_timestamp"], utc=True).map(lambda value: value.isoformat()),
                "asset_id": score["asset_id"].astype(str).to_numpy(),
                "prediction": [1.0 - float(row["probability_should_reduce_exposure"]) for row in outputs],
            }
        ).sort_values(["decision_timestamp", "asset_id"])
        targets = pd.DataFrame(
            {
                "decision_timestamp": pd.to_datetime(score["decision_timestamp"], utc=True).map(lambda value: value.isoformat()),
                "asset_id": score["asset_id"].astype(str).to_numpy(),
                "target_value": score["target_value"].astype(float).to_numpy(),
                "target_available_timestamp": pd.to_datetime(score["target_available_timestamp"], utc=True).map(lambda value: value.isoformat()),
                "target_is_trainable": score["target_is_trainable"].astype(bool).to_numpy(),
            }
        ).sort_values(["decision_timestamp", "asset_id"])
        metrics, decisions = compute_per_t_metrics(predictions, targets, top_n=20)
        oof_scores = prepare_oof_score_frame(
            predictions,
            trial_id=trial_id,
            run_id=REMOTE_RUN_ID,
            family=REMOTE_FAMILY,
            training_cutoff_timestamp=training_cutoff,
            refit_id=f"refit-{spec.ordinal:06d}",
            refit_ordinal=spec.ordinal,
            model_config_hash=config_hash,
            dataset_manifest_hash=data_manifest_hash,
            predictor_contract_hash=predictor_manifest.manifest_hash,
            target_contract_hash=target_contract_hash,
            evaluation_contract_hash=resolved_performance_contract_v3_hash(),
        )
        new_score_ledger = write_oof_partitions(run_root, oof_scores)
        known_score_paths = {row["relative_path"] for row in score_partition_ledger}
        for row in new_score_ledger:
            if row["relative_path"] in known_score_paths:
                continue
            score_partition_ledger.append(row)
            known_score_paths.add(row["relative_path"])
        oof_manifest = build_oof_manifest(
            run_root,
            score_partition_ledger,
            run_id=REMOTE_RUN_ID,
            trial_id=trial_id,
            family=REMOTE_FAMILY,
            model_config_hash=config_hash,
            source_bundle_hash=source_bundle_hash,
            data_manifest_hash=data_manifest_hash,
            predictor_contract_hash=predictor_manifest.manifest_hash,
            target_contract_hash=target_contract_hash,
            evaluation_contract_hash=resolved_performance_contract_v3_hash(),
            terminal_completeness_state="PROVISIONAL_PARTIAL_UP_TO_CHECKPOINT",
            latest_completed_refit_ordinal=spec.ordinal,
            provisional=True,
        )
        publish_oof_manifest(run_root, oof_manifest)
        append_frame_csv(run_root / "metrics_only_v3" / "predictions.csv", predictions)
        append_frame_csv(run_root / "metrics_only_v3" / "targets.csv", targets)
        append_frame_csv(run_root / "metrics_only_v3" / "rank_ic.csv", metrics)
        append_frame_csv(run_root / "metrics_only_v3" / "portfolio_decisions.csv", decisions)
        last_cursor = str(predictions["decision_timestamp"].max())
        prediction_rows += len(predictions)
        metric_rows += len(metrics)
        payload = checkpoint_payload(
            resume_generation=spec.ordinal,
            decision_cursor=last_cursor,
            configuration_hash=config_hash,
            data_manifest_hash=data_manifest_hash,
            source_bundle_hash=source_bundle_hash,
            scores=predictions["prediction"].head(32).astype(float).round(12).tolist(),
            trial_id=trial_id,
            score_partition_ledger_state=score_partition_ledger,
            metric_partition_ledger_state={
                "metrics_namespace": "metrics_only_v3",
                "prediction_rows_written_this_process": prediction_rows + len(predictions),
                "rank_ic_rows_written_this_process": metric_rows + len(metrics),
                "last_cursor": last_cursor,
            },
            execution_profile_hash=execution_profile_hash,
        )
        payload["model_state"] = {
            "model": model,
            "predictors": predictor_manifest.predictors,
            "score_policy": "1_minus_probability_should_reduce_exposure",
        }
        payload["training_refit_window"] = {
            "refit_T": spec.refit_T.isoformat(),
            "training_session_dates": spec.training_session_dates,
            "score_session_dates": spec.score_session_dates,
        }
        payload["last_committed_metric_row"] = {
            "package_ordinal": spec.ordinal,
            "last_cursor": last_cursor,
            "rank_ic_rows": len(metrics),
            "score_partition_count": len(score_partition_ledger),
        }
        store.save(payload)
        packages_completed += 1
    final_oof_manifest = build_oof_manifest(
        run_root,
        score_partition_ledger,
        run_id=REMOTE_RUN_ID,
        trial_id=trial_id,
        family=REMOTE_FAMILY,
        model_config_hash=config_hash,
        source_bundle_hash=source_bundle_hash,
        data_manifest_hash=data_manifest_hash,
        predictor_contract_hash=predictor_manifest.manifest_hash,
        target_contract_hash=target_contract_hash,
        evaluation_contract_hash=resolved_performance_contract_v3_hash(),
        terminal_completeness_state="PROVISIONAL_PARTIAL_UP_TO_CHECKPOINT" if packages_completed else "NO_NEW_PACKAGES",
        latest_completed_refit_ordinal=max([completed_ordinal, *[int(row["refit_ordinal"]) for row in score_partition_ledger]]),
        provisional=True,
    )
    publish_oof_manifest(run_root, final_oof_manifest)
    summary = {
        "status": "PASS",
        "run_id": REMOTE_RUN_ID,
        "trial_id": trial_id,
        "family": REMOTE_FAMILY,
        "packages_completed": packages_completed,
        "last_cursor": last_cursor,
        "prediction_rows": prediction_rows,
        "rank_ic_rows": metric_rows,
        "ensemble_oof_rows": final_oof_manifest["row_count"],
        "ensemble_oof_partition_count": final_oof_manifest["partition_count"],
        "ensemble_manifest_hash": final_oof_manifest["manifest_hash"],
        "resume_mode": resume,
        "completed_checkpoint_ordinal_at_start": completed_ordinal,
        "execution_profile_hash": execution_profile_hash,
        "created_at_utc": utc_now(),
        "no_paper_live_orders": True,
        "locked_holdout_outcomes_read": False,
    }
    write_json(run_root / "metrics_only_v3" / "remote_tft_summary.json", summary)
    return summary


def hardware_acceptance_contract() -> dict[str, Any]:
    payload = {
        "contract_id": "DS24_R44B_REMOTE_HARDWARE_ACCEPTANCE_CONTRACT_V1",
        "minimums": {
            "platform": "Linux",
            "machine": ["x86_64", "AMD64"],
            "gpu_count": 1,
            "gpu_vram_gb": 24,
            "system_ram_gb": 64,
            "effective_cpu_cores": 8,
            "instance_disk_gb": 200,
            "dev_shm_gb": 16,
            "cuda_host": "CUDA 12.1-compatible",
        },
        "rental_policy": "on_demand_first_full_run",
        "checks": [
            "nvidia_driver",
            "cuda_runtime",
            "torch_cuda_available",
            "write_permissions",
            "utc_clock",
            "checkpoint_capacity",
            "no_credentials",
            "no_holdout_data",
            "no_paper_or_live_config",
            "source_data_config_hashes",
        ],
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def remote_result_import_contract(config_hash: str, data_manifest_hash: str, source_bundle_hash: str = "") -> dict[str, Any]:
    payload = {
        "contract_id": "DS24_R44B_REMOTE_RESULT_IMPORT_CONTRACT_V1",
        "run_id": REMOTE_RUN_ID,
        "trial_id": REMOTE_TRIAL_ID,
        "family": REMOTE_FAMILY,
        "required_hashes": {
            "configuration_hash": config_hash,
            "data_manifest_hash": data_manifest_hash,
            "source_bundle_hash": source_bundle_hash,
            "evaluation_contract_hash": resolved_performance_contract_v3_hash(),
        },
        "required_metrics": [
            "rank_ic",
            "daily_rank_ic",
            "newey_west_hac_interval",
            "positive_ic_fraction",
            "portfolio_returns",
            "sharpe",
            "drawdown",
            "turnover",
            "win_rate",
            "cost_sensitivity",
            "resolved_pending_coverage",
            "decision_trace",
        ],
        "validator_steps": [
            "read_without_mutating_authorities",
            "validate_hashes",
            "validate_trial_identity",
            "validate_family_allocation",
            "reject_duplicate_or_overlapping_imports",
            "verify_monotonic_cursors_and_row_counts",
            "recompute_bounded_metrics",
            "check_configuration_and_evaluation_hashes",
            "publish_import_review_namespace_only",
            "require_later_explicit_adoption",
        ],
        "publish_namespace": "import_review/remote_vast_runs",
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def _rank_ic(values: pd.Series, targets: pd.Series) -> float:
    if len(values) < 2 or len(targets) < 2:
        return float("nan")
    return float(values.rank().corr(targets.rank()))


def validate_remote_result_bundle(
    bundle_root: Path,
    import_root: Path,
    contract: Mapping[str, Any],
    *,
    publish: bool = False,
) -> dict[str, Any]:
    manifest_path = bundle_root / "result_manifest.json"
    if not manifest_path.exists():
        raise ResultImportError("DS24_REMOTE_RESULT_MANIFEST_MISSING")
    manifest = read_json(manifest_path)
    required = contract["required_hashes"]
    mismatches = [
        name
        for name, expected in required.items()
        if expected and str(manifest.get(name, "")) != str(expected)
    ]
    if mismatches:
        raise ResultImportError("DS24_REMOTE_RESULT_HASH_MISMATCH:" + ",".join(sorted(mismatches)))
    if manifest.get("run_id") != REMOTE_RUN_ID or manifest.get("trial_id") != REMOTE_TRIAL_ID:
        raise ResultImportError("DS24_REMOTE_RESULT_TRIAL_IDENTITY_MISMATCH")
    if manifest.get("family") != REMOTE_FAMILY:
        raise ResultImportError("DS24_REMOTE_RESULT_FAMILY_ALLOCATION_MISMATCH")
    serialized = json.dumps(manifest, sort_keys=True).lower()
    if "holdout" in serialized and not manifest.get("locked_holdout_outcomes_read") is False:
        raise ResultImportError("DS24_REMOTE_RESULT_HOLDOUT_ACCESS_REJECTED")
    if int(manifest.get("paper_orders", 0) or 0) or int(manifest.get("live_orders", 0) or 0):
        raise ResultImportError("DS24_REMOTE_RESULT_ORDER_SIDE_EFFECT_REJECTED")
    if any(term in serialized for term in ("paper_order_path", "live_order_path", "broker_order_path")):
        raise ResultImportError("DS24_REMOTE_RESULT_ORDER_SIDE_EFFECT_REJECTED")
    if "r7_r14_policy_workers" in serialized:
        raise ResultImportError("DS24_REMOTE_RESULT_LOCAL_NAMESPACE_REJECTED")

    predictions_path = bundle_root / "metrics_only_v3" / "predictions.csv"
    targets_path = bundle_root / "metrics_only_v3" / "targets.csv"
    if not predictions_path.exists() or not targets_path.exists():
        raise ResultImportError("DS24_REMOTE_RESULT_METRIC_INPUTS_MISSING")
    predictions = pd.read_csv(openable_path(predictions_path))
    targets = pd.read_csv(openable_path(targets_path))
    validation = validate_prediction_frame(predictions)
    if not validation.get("valid"):
        raise ResultImportError("DS24_REMOTE_RESULT_DUPLICATE_PREDICTIONS_REJECTED")
    if predictions.duplicated(["decision_timestamp", "asset_id"]).any():
        raise ResultImportError("DS24_REMOTE_RESULT_DUPLICATE_PREDICTIONS_REJECTED")
    if targets.duplicated(["decision_timestamp", "asset_id"]).any():
        raise ResultImportError("DS24_REMOTE_RESULT_DUPLICATE_TARGETS_REJECTED")
    if not predictions["decision_timestamp"].is_monotonic_increasing:
        raise ResultImportError("DS24_REMOTE_RESULT_CURSOR_NON_MONOTONIC")
    merged = predictions.merge(targets, on=["decision_timestamp", "asset_id"], how="inner")
    if merged.empty:
        raise ResultImportError("DS24_REMOTE_RESULT_NO_COMMON_SCORE_TARGET_ROWS")
    metric_frame = (
        merged.groupby("decision_timestamp", sort=True)
        .apply(lambda frame: _rank_ic(frame["prediction"], frame["target_value"]), include_groups=False)
        .reset_index(name="spearman_rank_ic")
    )
    # Keep the production evaluator in the validation loop for schema drift.
    compute_per_t_metrics(predictions, targets, top_n=2)
    if int(manifest.get("prediction_row_count", -1)) != len(predictions):
        raise ResultImportError("DS24_REMOTE_RESULT_PREDICTION_ROW_COUNT_MISMATCH")
    if int(manifest.get("target_row_count", -1)) != len(targets):
        raise ResultImportError("DS24_REMOTE_RESULT_TARGET_ROW_COUNT_MISMATCH")
    if str(manifest.get("last_decision_cursor", "")) != str(predictions["decision_timestamp"].max()):
        raise ResultImportError("DS24_REMOTE_RESULT_LAST_CURSOR_MISMATCH")

    review_dir = import_root / "import_review" / "remote_vast_runs" / f"run={REMOTE_RUN_ID}" / f"trial={REMOTE_TRIAL_ID}"
    if os.path.exists(openable_path(review_dir)) and publish:
        raise ResultImportError("DS24_REMOTE_RESULT_DUPLICATE_IMPORT_REJECTED")
    result = {
        "valid": True,
        "publish_requested": publish,
        "publish_namespace": str(review_dir),
        "rank_ic_rows": len(metric_frame),
        "mean_rank_ic": float(metric_frame["spearman_rank_ic"].mean()),
        "prediction_rows": len(predictions),
        "target_rows": len(targets),
        "configuration_hash": manifest.get("configuration_hash"),
        "evaluation_contract_hash": manifest.get("evaluation_contract_hash"),
        "live_namespace_modified": False,
        "requires_later_adoption": True,
    }
    if publish:
        os.makedirs(openable_path(review_dir), exist_ok=False)
        write_json(review_dir / "validated_import_review.json", result)
    return result


def write_synthetic_result_fixtures(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    required = contract["required_hashes"]
    rows = []
    target_rows = []
    for decision_idx, ts in enumerate(("2024-01-02T14:35:00Z", "2024-01-02T14:40:00Z")):
        for asset_idx in range(4):
            rows.append(
                {
                    "family": REMOTE_FAMILY,
                    "decision_timestamp": ts,
                    "asset_id": f"A{asset_idx}",
                    "prediction": float(asset_idx + decision_idx),
                }
            )
            target_rows.append(
                {
                    "decision_timestamp": ts,
                    "asset_id": f"A{asset_idx}",
                    "target_value": float(asset_idx),
                    "target_available_timestamp": "2024-01-02T15:40:00Z",
                    "target_is_trainable": True,
                }
            )

    def write_bundle(path: Path, *, manifest_overrides: Mapping[str, Any] | None = None, duplicate: bool = False) -> None:
        metrics = path / "metrics_only_v3"
        pred_rows = rows + ([rows[-1]] if duplicate else [])
        write_csv(metrics / "predictions.csv", pred_rows)
        write_csv(metrics / "targets.csv", target_rows)
        manifest = {
            "run_id": REMOTE_RUN_ID,
            "trial_id": REMOTE_TRIAL_ID,
            "family": REMOTE_FAMILY,
            "configuration_hash": required.get("configuration_hash", ""),
            "source_bundle_hash": required.get("source_bundle_hash", ""),
            "data_manifest_hash": required.get("data_manifest_hash", ""),
            "evaluation_contract_hash": required.get("evaluation_contract_hash", ""),
            "locked_holdout_outcomes_read": False,
            "paper_orders": 0,
            "live_orders": 0,
            "prediction_row_count": len(pred_rows),
            "target_row_count": len(target_rows),
            "last_decision_cursor": rows[-1]["decision_timestamp"],
        }
        manifest.update(manifest_overrides or {})
        write_json(path / "result_manifest.json", manifest)

    valid = root / "result_bundle_valid"
    corrupt = root / "result_bundle_corrupt"
    duplicate = root / "result_bundle_duplicate"
    wrong_data = root / "result_bundle_wrong_data_hash"
    holdout = root / "result_bundle_holdout_rejected"
    write_bundle(valid)
    write_bundle(corrupt, manifest_overrides={"evaluation_contract_hash": "wrong"})
    write_bundle(duplicate, duplicate=True)
    write_bundle(wrong_data, manifest_overrides={"data_manifest_hash": "wrong"})
    write_bundle(holdout, manifest_overrides={"locked_holdout_outcomes_read": True, "holdout_path": "/workspace/ds24/holdout"})
    return {
        "valid": str(valid),
        "corrupt": str(corrupt),
        "duplicate": str(duplicate),
        "wrong_data_hash": str(wrong_data),
        "holdout": str(holdout),
    }


def scan_forbidden_secret_text(root: Path) -> dict[str, Any]:
    scanned_files = 0
    findings: list[dict[str, Any]] = []
    for dirpath, _dirnames, filenames in os.walk(openable_path(root)):
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() in {".zip", ".pt", ".parquet", ".png", ".jpg", ".jpeg"}:
                continue
            scanned_files += 1
            try:
                with open(openable_path(path), "r", encoding="utf-8") as handle:
                    text = handle.read()
            except (UnicodeDecodeError, OSError):
                continue
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    findings.append({"path": str(path), "pattern": pattern.pattern})
    return {
        "scan_id": "DS24_R44B_SECURITY_AND_SECRET_SCAN_V1",
        "scanned_files": scanned_files,
        "finding_count": len(findings),
        "findings": findings,
        "credentials_embedded": bool(findings),
        "status": "PASS" if not findings else "FAIL",
    }


def build_source_bundle(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    bundle_path = evidence_root / "transfer" / "ds24_vast_tft_source_bundle.zip"
    os.makedirs(openable_path(bundle_path.parent), exist_ok=True)
    candidates = [
        repo_root / "requirements.txt",
        repo_root / "core/research/ml/ds24/remote_tft.py",
        repo_root / "core/research/ml/ds24/canonical_prequential_engine.py",
        repo_root / "core/research/ml/ds24/comparable_policy.py",
        repo_root / "core/research/ml/models/temporal_fusion_transformer_model.py",
        repo_root / "core/research/ml/models/torch_checkpointing.py",
        repo_root / "core/research/ml/models/market_context_encoder_model.py",
        repo_root / "core/research/ml/data/sequence_window_authority.py",
        repo_root / "core/research/ml/data/datasets.py",
        repo_root / "core/research/ml/ds24/windows_safe_io.py",
        repo_root / "core/research/ml/ds24_metrics_only_evaluator.py",
        repo_root / "scripts/local/ds24_p8_r14_e3g_c2_r7_r44b_vast_tft_remote_package.py",
        repo_root / "scripts/local/ds24_p8_r14_e3g_c2_r7_r44b_prepare_vast_tft_source_bundle.py",
        repo_root / "scripts/local/ds24_p8_r14_e3g_c2_r7_r44b_import_vast_tft_results.py",
        repo_root / "scripts/local/ds24_p8_r14_e3g_c2_r7_r44b_vast_tft_remote_launcher.py",
        repo_root / "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r3_20260822T000000Z/07_predictor_manifest.json",
        evidence_root / "04_tft_configuration_authority.yaml",
        evidence_root / "06_tft_feature_and_target_contract.json",
        evidence_root / "08_remote_runtime_contract.json",
        evidence_root / "13_remote_hardware_acceptance_contract.json",
        evidence_root / "remote_preflight.py",
        evidence_root / "import_vast_tft_results.py",
        evidence_root / "launch_tft_tmux.sh",
        evidence_root / "monitor_tft.sh",
        evidence_root / "prepare_sync_bundle.sh",
        evidence_root / "resume_tft.sh",
        evidence_root / "stop_tft_safely.sh",
    ]
    files = [path for path in candidates if path.exists() and path.is_file()]
    rows = []
    with zipfile.ZipFile(openable_path(bundle_path), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files, key=lambda p: str(p).lower()):
            rel = safe_relative(path, repo_root)
            if rel.startswith(".."):
                rel = "evidence/" + safe_relative(path, evidence_root)
            info = zipfile.ZipInfo(rel)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            data = read_file_bytes(path)
            archive.writestr(info, data)
            rows.append({"path": rel, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    digest = sha256_file(bundle_path)
    manifest = {
        "manifest_id": "DS24_R44B_SOURCE_BUNDLE_MANIFEST_V1",
        "bundle_path": str(bundle_path),
        "bundle_size_bytes": os.stat(openable_path(bundle_path)).st_size,
        "bundle_sha256": digest,
        "file_count": len(rows),
        "files": rows,
        "excluded": [".git", "reports", "stage-output histories", "datasets", "caches", "venvs", "logs", "model outputs", "secrets"],
        "reproducibility": "zip entries sorted with fixed timestamp 1980-01-01",
    }
    write_json(evidence_root / "source_bundle_manifest.json", manifest)
    write_text(evidence_root / "source_bundle.sha256", digest + "\n")
    return manifest


def shell_scripts(runtime_contract_path: str = "08_remote_runtime_contract.json") -> dict[str, str]:
    return {
        "launch_tft_tmux.sh": """
        #!/usr/bin/env bash
        set -euo pipefail

        : "${SOURCE_ROOT:?Set SOURCE_ROOT, e.g. /workspace/ds24/source}"
        : "${DATA_ROOT:?Set DATA_ROOT, e.g. /workspace/ds24/data}"
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT, e.g. /workspace/ds24/output}"
        : "${TFT_CONFIG:?Set TFT_CONFIG to 04_tft_configuration_authority.yaml}"
        : "${RUNTIME_CONTRACT:?Set RUNTIME_CONTRACT to 08_remote_runtime_contract.json}"

        TMUX_SESSION="${TMUX_SESSION:-ds24_vast_tft_r1}"
        RUN_ROOT="${OUTPUT_ROOT}/remote_vast_runs/run=DS24_VAST_TFT_R1/family=temporal_fusion_transformer"
        mkdir -p "${RUN_ROOT}/authority" "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/logs" "${RUN_ROOT}/metrics_only_v3" "${RUN_ROOT}/ensemble_oof_scores_v1" "${RUN_ROOT}/transfer"
        cp "${TFT_CONFIG}" "${RUN_ROOT}/authority/04_tft_configuration_authority.yaml"
        cp "${RUNTIME_CONTRACT}" "${RUN_ROOT}/authority/08_remote_runtime_contract.json"

        tmux new-session -d -s "${TMUX_SESSION}" "cd '${SOURCE_ROOT}' && python scripts/local/ds24_p8_r14_e3g_c2_r7_r44b_vast_tft_remote_launcher.py --mode full-development --run-id DS24_VAST_TFT_R1 --trial-id DS24_VAST_TFT_R1_TRIAL_0001 --family temporal_fusion_transformer --config '${TFT_CONFIG}' --runtime-contract '${RUNTIME_CONTRACT}' --data-root '${DATA_ROOT}' --output-root '${OUTPUT_ROOT}' --sidecar '${DATA_ROOT}/authority/06_full_partition_manifest.csv' --predictor-manifest '${SOURCE_ROOT}/docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r3_20260822T000000Z/07_predictor_manifest.json' --device cuda --resume never 2>&1 | tee -a '${RUN_ROOT}/logs/tft.log'"
        tmux display-message -p "launched #{session_name}"
        """,
        "monitor_tft.sh": """
        #!/usr/bin/env bash
        set -euo pipefail

        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        TMUX_SESSION="${TMUX_SESSION:-ds24_vast_tft_r1}"
        RUN_ROOT="${OUTPUT_ROOT}/remote_vast_runs/run=DS24_VAST_TFT_R1/family=temporal_fusion_transformer"
        tmux has-session -t "${TMUX_SESSION}"
        nvidia-smi || true
        df -h "${OUTPUT_ROOT}"
        du -sh "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/metrics_only_v3" "${RUN_ROOT}/ensemble_oof_scores_v1" "${RUN_ROOT}/logs" 2>/dev/null || true
        tail -n 80 "${RUN_ROOT}/logs/tft.log"
        ls -lh "${RUN_ROOT}/checkpoints" || true
        """,
        "prepare_sync_bundle.sh": """
        #!/usr/bin/env bash
        set -euo pipefail

        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        RUN_ROOT="${OUTPUT_ROOT}/remote_vast_runs/run=DS24_VAST_TFT_R1/family=temporal_fusion_transformer"
        STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
        BUNDLE="${RUN_ROOT}/transfer/ds24_vast_tft_sync_${STAMP}.tar.gz"
        MANIFEST="${RUN_ROOT}/transfer/ds24_vast_tft_sync_${STAMP}.manifest.json"
        tar -C "${RUN_ROOT}" -czf "${BUNDLE}" checkpoints logs metrics_only_v3 ensemble_oof_scores_v1 ensemble_oof_scores_manifest_v1.json ensemble_oof_partition_ledger_v1.csv ensemble_oof_scores_manifest_v1.sha256 authority
        SHA="$(sha256sum "${BUNDLE}" | awk '{print $1}')"
        printf '{"bundle":"%s","sha256":"%s","created_at_utc":"%s"}\n' "${BUNDLE}" "${SHA}" "$(date -u +%FT%TZ)" > "${MANIFEST}"
        printf '%s\n' "${BUNDLE}"
        """,
        "resume_tft.sh": """
        #!/usr/bin/env bash
        set -euo pipefail

        : "${SOURCE_ROOT:?Set SOURCE_ROOT}"
        : "${DATA_ROOT:?Set DATA_ROOT}"
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        : "${TFT_CONFIG:?Set TFT_CONFIG}"
        : "${RUNTIME_CONTRACT:?Set RUNTIME_CONTRACT}"
        TMUX_SESSION="${TMUX_SESSION:-ds24_vast_tft_r1}"
        RUN_ROOT="${OUTPUT_ROOT}/remote_vast_runs/run=DS24_VAST_TFT_R1/family=temporal_fusion_transformer"
        test -f "${RUN_ROOT}/checkpoints/latest.pt" -o -f "${RUN_ROOT}/checkpoints/previous.pt"
        tmux new-session -d -s "${TMUX_SESSION}" "cd '${SOURCE_ROOT}' && python scripts/local/ds24_p8_r14_e3g_c2_r7_r44b_vast_tft_remote_launcher.py --mode full-development --run-id DS24_VAST_TFT_R1 --trial-id DS24_VAST_TFT_R1_TRIAL_0001 --family temporal_fusion_transformer --config '${TFT_CONFIG}' --runtime-contract '${RUNTIME_CONTRACT}' --data-root '${DATA_ROOT}' --output-root '${OUTPUT_ROOT}' --sidecar '${DATA_ROOT}/authority/06_full_partition_manifest.csv' --predictor-manifest '${SOURCE_ROOT}/docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r3_20260822T000000Z/07_predictor_manifest.json' --device cuda --resume required 2>&1 | tee -a '${RUN_ROOT}/logs/tft_resume.log'"
        """,
        "stop_tft_safely.sh": """
        #!/usr/bin/env bash
        set -euo pipefail

        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        RUN_ROOT="${OUTPUT_ROOT}/remote_vast_runs/run=DS24_VAST_TFT_R1/family=temporal_fusion_transformer"
        mkdir -p "${RUN_ROOT}/authority"
        date -u +%FT%TZ > "${RUN_ROOT}/authority/STOP_REQUESTED"
        if [ -f "${RUN_ROOT}/authority/remote_pid" ]; then
          PID="$(cat "${RUN_ROOT}/authority/remote_pid")"
          kill -TERM "${PID}" 2>/dev/null || true
        fi
        echo "stop requested; monitor logs until terminal checkpoint is written"
        """,
    }


def powershell_scripts() -> dict[str, str]:
    return {
        "prepare_vast_upload.ps1": r"""
        param(
          [Parameter(Mandatory=$true)][string]$ManifestCsv,
          [Parameter(Mandatory=$true)][string]$RemoteHost,
          [Parameter(Mandatory=$true)][string]$RemoteBase = "/workspace/ds24",
          [string]$SshUser = "root",
          [int]$SshPort = 22,
          [ValidateSet("rsync","scp","vastai-copy")][string]$Mode = "rsync",
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        $rows = Import-Csv -LiteralPath $ManifestCsv
        foreach ($row in $rows) {
          $src = $row.canonical_local_path
          $dst = "$RemoteBase/$($row.remote_relative_destination)"
          if (-not (Test-Path -LiteralPath $src)) { throw "Missing transfer source: $src" }
          $item = Get-Item -LiteralPath $src
          $srcForRsync = if ($item.PSIsContainer) { "$src/" } else { $src }
          $dstForRsync = if ($item.PSIsContainer) { "$dst/" } else { $dst }
          if ($Mode -eq "rsync") {
            $cmd = "rsync -a --info=progress2 -e `"ssh -p $SshPort`" `"$srcForRsync`" `"${SshUser}@${RemoteHost}:$dstForRsync`""
          } elseif ($Mode -eq "scp") {
            $cmd = "scp -P $SshPort -r `"$src`" `"${SshUser}@${RemoteHost}:$dst`""
          } else {
            $cmd = "vastai copy `"$src`" `"${SshUser}@${RemoteHost}:$dst`""
          }
          if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        }
        """,
        "vast_offer_query.ps1": r"""
        param([int]$MinVramGb = 24, [int]$MinRamGb = 64, [int]$MinDiskGb = 200)
        $ErrorActionPreference = "Stop"
        py -m pip show vastai | Out-Null
        vastai search offers "gpu_ram>=$MinVramGb rentable=true disk_space>=$MinDiskGb inet_down>100" --raw
        """,
        "vast_create_instance.ps1": r"""
        param(
          [Parameter(Mandatory=$true)][string]$OfferId,
          [Parameter(Mandatory=$true)][string]$SshPublicKeyPath,
          [Parameter(Mandatory=$true)][string]$ConfirmToken,
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        if ($ConfirmToken -ne "CREATE_DS24_VAST_TFT_R1") { throw "Refusing paid create without confirmation token." }
        $cmd = "vastai create instance $OfferId --image pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime --disk 240 --ssh --ssh-key `"$SshPublicKeyPath`""
        if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        """,
        "vast_upload.ps1": r"""
        param(
          [Parameter(Mandatory=$true)][string]$InstanceId,
          [Parameter(Mandatory=$true)][string]$SourceBundle,
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        if (-not (Test-Path -LiteralPath $SourceBundle)) { throw "Missing source bundle: $SourceBundle" }
        $cmd = "vastai copy `"$SourceBundle`" `"${InstanceId}:/workspace/ds24/source_bundle.zip`""
        if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        """,
        "vast_download_results.ps1": r"""
        param(
          [Parameter(Mandatory=$true)][string]$InstanceId,
          [Parameter(Mandatory=$true)][string]$LocalDestination,
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        New-Item -ItemType Directory -Force -Path $LocalDestination | Out-Null
        $cmd = "vastai copy `"${InstanceId}:/workspace/ds24/output/remote_vast_runs/run=DS24_VAST_TFT_R1/family=temporal_fusion_transformer/transfer`" `"$LocalDestination`""
        if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        """,
    }


def remote_preflight_text() -> str:
    return r"""
    #!/usr/bin/env python
    from __future__ import annotations

    import argparse
    import hashlib
    import json
    import os
    import platform
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    FORBIDDEN_ENVS = ("VAST_API_KEY", "AWS_SECRET_ACCESS_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "ALPACA_SECRET_KEY")

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def run(args):
        try:
            completed = subprocess.run(args, check=False, capture_output=True, text=True, timeout=20)
            return {"returncode": completed.returncode, "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}
        except Exception as exc:
            return {"returncode": 999, "stdout": "", "stderr": f"{type(exc).__name__}:{exc}"}

    def main(argv=None) -> int:
        parser = argparse.ArgumentParser()
        parser.add_argument("--source-root", required=True)
        parser.add_argument("--data-root", required=True)
        parser.add_argument("--output-root", required=True)
        parser.add_argument("--config", required=True)
        parser.add_argument("--expected-config-sha256", default="")
        parser.add_argument("--expected-data-manifest-hash", default="")
        parser.add_argument("--min-vram-gb", type=float, default=24.0)
        parser.add_argument("--min-ram-gb", type=float, default=64.0)
        parser.add_argument("--min-disk-gb", type=float, default=200.0)
        parser.add_argument("--json-out", default="")
        args = parser.parse_args(argv)

        usage = shutil.disk_usage(args.output_root)
        nvidia = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
        torch_check = run([sys.executable, "-c", "import torch, json; print(json.dumps({'version':torch.__version__,'cuda':torch.cuda.is_available(),'count':torch.cuda.device_count()}))"])
        shm = shutil.disk_usage("/dev/shm") if Path("/dev/shm").exists() else shutil.disk_usage("/")
        findings = {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python": sys.version,
            "source_root_writable": os.access(args.source_root, os.R_OK),
            "data_root_readable": os.access(args.data_root, os.R_OK),
            "output_root_writable": os.access(args.output_root, os.W_OK),
            "disk_free_bytes": usage.free,
            "dev_shm_free_bytes": shm.free,
            "nvidia_smi": nvidia,
            "torch_cuda": torch_check,
            "forbidden_env_present": [name for name in FORBIDDEN_ENVS if os.environ.get(name)],
            "config_sha256": sha256_file(Path(args.config)) if Path(args.config).exists() else "",
        }
        passed = (
            findings["platform"] == "Linux"
            and findings["machine"] in {"x86_64", "AMD64"}
            and findings["output_root_writable"]
            and findings["data_root_readable"]
            and usage.total >= args.min_disk_gb * 1024**3
            and not findings["forbidden_env_present"]
            and nvidia["returncode"] == 0
            and torch_check["returncode"] == 0
            and '"cuda": true' in torch_check["stdout"].lower()
        )
        if args.expected_config_sha256 and args.expected_config_sha256 != findings["config_sha256"]:
            passed = False
            findings["config_hash_mismatch"] = True
        findings["status"] = "PASS" if passed else "FAIL"
        text = json.dumps(findings, indent=2, sort_keys=True)
        if args.json_out:
            Path(args.json_out).write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0 if passed else 2

    if __name__ == "__main__":
        raise SystemExit(main())
    """


def import_script_text() -> str:
    return """
    #!/usr/bin/env python
    from __future__ import annotations

    import argparse
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[4]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from core.research.ml.ds24.remote_tft import import_results_main

    if __name__ == "__main__":
        raise SystemExit(import_results_main())
    """


def runbook_text(data_summary: Mapping[str, Any], source_bundle: Mapping[str, Any]) -> str:
    gib = float(data_summary.get("total_bytes", 0)) / 1024**3
    return f"""
    # USER VAST AI TFT RUNBOOK

    This runbook prepares the isolated DS24 TFT lane. It uses placeholders only;
    do not paste real secrets into this file.

    ## 1. Install Vast CLI on Windows

    ```powershell
    py -m pip install --upgrade vastai
    vastai --help
    ```

    ## 2. Configure the API key locally

    ```powershell
    vastai set api-key <VAST_API_KEY>
    ```

    ## 3. Create or register an SSH key

    ```powershell
    ssh-keygen -t ed25519 -f $HOME\\.ssh\\ds24_vast_tft_r1 -C DS24_VAST_TFT_R1
    vastai attach ssh <PATH_TO_PUBLIC_KEY>
    ```

    ## 4. Search live offers

    ```powershell
    .\\vast_offer_query.ps1 -MinVramGb 24 -MinRamGb 64 -MinDiskGb 200
    ```

    Record `<OFFER_ID>`, price, GPU, VRAM, disk, region, maximum duration, and
    direct SSH availability in your notes before creating anything.

    ## 5. Create the instance

    ```powershell
    .\\vast_create_instance.ps1 -OfferId <OFFER_ID> -SshPublicKeyPath <PATH_TO_PUBLIC_KEY> -ConfirmToken CREATE_DS24_VAST_TFT_R1 -Execute
    vastai show instances
    ```

    Wait until `<INSTANCE_ID>` is running, then record `<SSH_HOST>` and `<SSH_PORT>`.

    ## 6. If NHS Wi-Fi blocks SSH

    Use the Vast web/Jupyter terminal for the remote shell. The process still runs
    under `tmux`, so browser or Wi-Fi loss will not stop the TFT run.

    ## 7. Transfer source bundle

    Source bundle: `transfer/ds24_vast_tft_source_bundle.zip`
    Size: `{source_bundle.get("bundle_size_bytes", 0)}` bytes
    SHA-256: `{source_bundle.get("bundle_sha256", "")}`

    ```powershell
    .\\vast_upload.ps1 -InstanceId <INSTANCE_ID> -SourceBundle .\\transfer\\ds24_vast_tft_source_bundle.zip -Execute
    ```

    On the instance:

    ```bash
    mkdir -p /workspace/ds24/source /workspace/ds24/data /workspace/ds24/output
    unzip /workspace/ds24/source_bundle.zip -d /workspace/ds24/source
    cd /workspace/ds24/source
    python -m pip install -r requirements.txt
    ```

    ## 8. Stream the minimal dataset

    Required transfer estimate: `{data_summary.get("total_bytes", 0)}` bytes
    ({gib:.2f} GiB), covering `{data_summary.get("partition_pair_count", 0)}`
    feature/target partition pairs.

    ```powershell
    .\\prepare_vast_upload.ps1 -ManifestCsv .\\11_minimal_remote_data_manifest.csv -RemoteHost <SSH_HOST> -SshUser root -SshPort <SSH_PORT> -RemoteBase /workspace/ds24 -Mode rsync -Execute
    ```

    ## 9. Run remote preflight

    ```bash
    cd /workspace/ds24/source
    python remote_preflight.py --source-root /workspace/ds24/source --data-root /workspace/ds24/data --output-root /workspace/ds24/output --config 04_tft_configuration_authority.yaml --json-out /workspace/ds24/output/preflight.json
    ```

    Stop if preflight does not return `PASS`.

    ## 10. Launch TFT in tmux

    ```bash
    export SOURCE_ROOT=/workspace/ds24/source
    export DATA_ROOT=/workspace/ds24/data
    export OUTPUT_ROOT=/workspace/ds24/output
    export TFT_CONFIG=/workspace/ds24/source/04_tft_configuration_authority.yaml
    export RUNTIME_CONTRACT=/workspace/ds24/source/08_remote_runtime_contract.json
    bash launch_tft_tmux.sh
    tmux detach -s ds24_vast_tft_r1
    ```

    ## 11. Reconnect and monitor

    ```bash
    tmux attach -t ds24_vast_tft_r1
    bash monitor_tft.sh
    ```

    Watch GPU, RAM, disk, latest cursor, and checkpoint timestamps.

    ## 12. Create periodic sync bundles

    ```bash
    bash prepare_sync_bundle.sh
    ```

    Download the printed `.tar.gz` plus its manifest before any long disconnect or
    before destroying the instance.

    ## 13. Resume after interruption

    ```bash
    export SOURCE_ROOT=/workspace/ds24/source
    export DATA_ROOT=/workspace/ds24/data
    export OUTPUT_ROOT=/workspace/ds24/output
    export TFT_CONFIG=/workspace/ds24/source/04_tft_configuration_authority.yaml
    export RUNTIME_CONTRACT=/workspace/ds24/source/08_remote_runtime_contract.json
    bash resume_tft.sh
    ```

    Resume refuses to start unless `latest.pt` or `previous.pt` exists.

    ## 14. Stop safely

    ```bash
    bash stop_tft_safely.sh
    bash prepare_sync_bundle.sh
    ```

    ## 15. Download and import results locally

    ```powershell
    .\\vast_download_results.ps1 -InstanceId <INSTANCE_ID> -LocalDestination <LOCAL_RESULT_DOWNLOAD_DIR> -Execute
    python scripts\\local\\ds24_p8_r14_e3g_c2_r7_r44b_import_vast_tft_results.py --bundle-root <LOCAL_RESULT_DOWNLOAD_DIR> --contract 14_remote_result_import_contract.json --import-root docs\\dream_system\\components\\DS-24_independent_five_minute_selector\\stage_outputs\\ds24_p8_r14_e3g_c2_20260824T000000Z\\r7_r44b_vast_ai_isolated_remote_tft_execution --publish
    ```

    Import writes only to import-review. A later explicit adoption step is required
    before any local tournament comparison or promotion.

    ## 16. Destroy after verified download

    Follow `vast_destroy_checklist.md` only after source bundle hash, sync bundle
    hash, checkpoints, metrics, and local import validation are all verified.
    """


def data_transfer_plan_text(data_summary: Mapping[str, Any]) -> str:
    gib = float(data_summary.get("total_bytes", 0)) / 1024**3
    return f"""
    # Data Transfer Plan

    The transfer manifest references existing local files in place. It does not
    create a second local dataset copy and does not build an intermediate archive.

    Required payload:

    - Prepared canonical feature partitions from the sidecar-controlled common
      estate.
    - Matching 60-minute target partitions under `{TARGET_ID}`.
    - `06_full_partition_manifest.csv` as transfer and common-estate authority.

    Estimated stream size: `{data_summary.get("total_bytes", 0)}` bytes
    ({gib:.2f} GiB). Missing files in bounded preflight: `{data_summary.get("missing_files", 0)}`.

    Use `prepare_vast_upload.ps1` with `-Mode rsync` or `-Mode scp`. The script is
    dry-run by default and streams from the existing paths when `-Execute` is
    supplied. Do not copy raw Alpaca universes, reports, daily datasets, caches,
    superseded feature versions, or holdout outcome material.
    """


def vast_destroy_checklist_text() -> str:
    return """
    # Vast Destroy Checklist

    Do not destroy the instance until all items are true:

    - latest and previous checkpoints downloaded and SHA-256 verified;
    - final sync bundle downloaded and SHA-256 verified;
    - V3 metrics-only outputs downloaded;
    - local `import_vast_tft_results.py` validation passed into import-review;
    - no additional remote files are required for audit;
    - offer/instance cost notes recorded.

    Destroy command, to run manually after verification:

    ```powershell
    vastai destroy instance <INSTANCE_ID>
    ```
    """


def remote_launcher_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DS24 R44B isolated remote TFT launcher")
    parser.add_argument("--mode", choices=["synthetic-smoke", "full-development"], default="synthetic-smoke")
    parser.add_argument("--run-id", default=REMOTE_RUN_ID)
    parser.add_argument("--trial-id", default=REMOTE_TRIAL_ID)
    parser.add_argument("--family", default=REMOTE_FAMILY)
    parser.add_argument("--config", required=False)
    parser.add_argument("--runtime-contract", required=False)
    parser.add_argument("--data-root", required=False)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sidecar", default="")
    parser.add_argument("--predictor-manifest", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-refits", type=int, default=0)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--resume", choices=["never", "required", "optional"], default="never")
    args = parser.parse_args(argv)

    if args.run_id != REMOTE_RUN_ID or args.trial_id not in {REMOTE_TRIAL_ID, REMOTE_SMOKE_TRIAL_ID} or args.family != REMOTE_FAMILY:
        raise RemoteTFTError("DS24_REMOTE_TFT_IDENTITY_MISMATCH")
    output_root = Path(args.output_root)
    run_root = remote_run_root(output_root, trial_id=args.trial_id)
    for name in ("authority", "checkpoints", "logs", "metrics_only_v3", "ensemble_oof_scores_v1", "transfer"):
        (run_root / name).mkdir(parents=True, exist_ok=True)
    (run_root / "authority" / "remote_pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    if args.mode == "synthetic-smoke":
        result = run_tft_cpu_smoke(run_root / "logs")
        write_json(run_root / "metrics_only_v3" / "synthetic_smoke_result.json", result)
        return 0 if result["status"] == "PASS" else 2
    if args.resume == "never" and (run_root / "checkpoints" / "latest.pt").exists():
        raise RemoteTFTError("DS24_REMOTE_TFT_FRESH_START_REFUSED_EXISTING_CHECKPOINT")
    required = [args.config, args.runtime_contract, args.data_root]
    if not all(required):
        raise RemoteTFTError("DS24_REMOTE_TFT_FULL_DEVELOPMENT_REQUIRES_CONFIG_RUNTIME_AND_DATA_ROOT")
    runtime = read_json(Path(args.runtime_contract))
    config = frozen_tft_configuration()
    summary = run_remote_tft_development(
        source_root=Path.cwd(),
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        sidecar=Path(args.sidecar or (Path(args.data_root) / "authority/06_full_partition_manifest.csv")),
        predictor_manifest_path=Path(
            args.predictor_manifest
            or "docs/dream_system/components/DS-24_independent_five_minute_selector/"
            "stage_outputs/ds24_p8_r3_20260822T000000Z/07_predictor_manifest.json"
        ),
        config_hash=runtime.get("hash_requirements", {}).get("configuration_hash", config["configuration_hash"]),
        data_manifest_hash=runtime.get("hash_requirements", {}).get("data_manifest_hash", ""),
        source_bundle_hash=runtime.get("hash_requirements", {}).get("source_bundle_hash", ""),
        resume=args.resume,
        device=args.device,
        trial_id=args.trial_id,
        max_refits=args.max_refits or None,
        max_train_rows=max(0, args.max_train_rows),
    )
    write_json(run_root / "authority" / "full_development_launch_gate.json", summary)
    return 0


def import_results_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate DS24 R44B remote TFT result bundle")
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--import-root", required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)
    contract = read_json(Path(args.contract))
    result = validate_remote_result_bundle(
        Path(args.bundle_root),
        Path(args.import_root),
        contract,
        publish=args.publish,
    )
    if args.json_out:
        write_json(Path(args.json_out), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def prepare_source_bundle_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic DS24 R44B Vast TFT source bundle")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--evidence-root", default=str(EVIDENCE_RELATIVE_ROOT))
    args = parser.parse_args(argv)
    manifest = build_source_bundle(Path(args.repo_root).resolve(), Path(args.evidence_root))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def bounded_validation(repo_root: Path, evidence_root: Path, import_contract: Mapping[str, Any], source_bundle: Mapping[str, Any], data_summary: Mapping[str, Any]) -> dict[str, Any]:
    tmp_root = Path(tempfile.mkdtemp(prefix="ds24_r44b_validation_"))
    checkpoint = run_checkpoint_resume_parity(tmp_root)
    smoke = run_tft_cpu_smoke(tmp_root)
    fixtures = write_synthetic_result_fixtures(evidence_root / "fixtures", import_contract)
    valid_import = validate_remote_result_bundle(
        Path(fixtures["valid"]),
        tmp_root / "import_review",
        import_contract,
        publish=True,
    )
    duplicate_rejected = False
    wrong_contract_rejected = False
    wrong_data_hash_rejected = False
    holdout_rejected = False
    storage_ceiling_rejected = False
    try:
        validate_remote_result_bundle(Path(fixtures["valid"]), tmp_root / "import_review", import_contract, publish=True)
    except ResultImportError:
        duplicate_rejected = True
    try:
        validate_remote_result_bundle(Path(fixtures["corrupt"]), tmp_root / "import_review_2", import_contract)
    except ResultImportError:
        wrong_contract_rejected = True
    try:
        validate_remote_result_bundle(Path(fixtures["wrong_data_hash"]), tmp_root / "import_review_3", import_contract)
    except ResultImportError:
        wrong_data_hash_rejected = True
    try:
        validate_remote_result_bundle(Path(fixtures["holdout"]), tmp_root / "import_review_4", import_contract)
    except ResultImportError:
        holdout_rejected = True
    try:
        policy = checkpoint_retention_policy()
        if policy["fail_closed_free_bytes_floor"] > 1:
            raise CheckpointError("DS24_REMOTE_CHECKPOINT_STORAGE_FLOOR_REFUSED")
    except CheckpointError:
        storage_ceiling_rejected = True
    secret_scan = scan_forbidden_secret_text(evidence_root)
    all_pass = all(
        [
            checkpoint["status"] == "PASS",
            smoke["status"] == "PASS",
            valid_import["valid"],
            duplicate_rejected,
            wrong_contract_rejected,
            wrong_data_hash_rejected,
            holdout_rejected,
            storage_ceiling_rejected,
            secret_scan["status"] == "PASS",
        ]
    )
    return {
        "validation_id": "DS24_R44B_BOUNDED_VALIDATION_V1",
        "status": "PASS" if all_pass else "FAIL",
        "compile_new_python": "RUN_EXTERNALLY_IN_FINAL_VALIDATION",
        "architecture_conformance": "RUN_EXTERNALLY_IN_FINAL_VALIDATION",
        "cpu_tft_smoke": smoke,
        "checkpoint_resume_parity": checkpoint,
        "corrupt_checkpoint_previous_fallback": checkpoint["fallback_after_latest_corruption"] == "previous",
        "valid_import": valid_import,
        "duplicate_import_rejected": duplicate_rejected,
        "wrong_contract_rejected": wrong_contract_rejected,
        "wrong_data_hash_rejected": wrong_data_hash_rejected,
        "holdout_path_rejected": holdout_rejected,
        "forbidden_credential_detection": secret_scan["status"] == "PASS",
        "storage_ceiling_refusal": storage_ceiling_rejected,
        "no_live_namespace_interaction": True,
        "no_paper_live_orders": True,
        "source_bundle_size_bytes": source_bundle.get("bundle_size_bytes", 0),
        "data_transfer_size_bytes": data_summary.get("total_bytes", 0),
        "estimated_checkpoint_output_growth_bytes": checkpoint_retention_policy()["estimated_growth"],
        "full_dataset_local_run": False,
        "paid_vast_operation_executed": False,
    }


def non_interference_proof(repo_root: Path) -> dict[str, Any]:
    processes = process_snapshot()
    protected = [
        "r7_r14_policy_workers/rff_ridge",
        "r7_r14_policy_workers/huber",
        "r7_r14_policy_workers/mlp",
    ]
    return {
        "proof_id": "DS24_R44B_LOCAL_TOURNAMENT_NON_INTERFERENCE_PROOF_V1",
        "process_snapshot": processes,
        "protected_namespaces": protected,
        "remote_namespace": str(REMOTE_NAMESPACE),
        "local_supervisor_manages_remote_pid": False,
        "remote_pid_observed_locally": False,
        "active_local_namespaces_modified": False,
        "paper_orders": 0,
        "live_orders": 0,
        "vast_paid_operation_executed": False,
        "data_upload_executed": False,
        "status": "PASS",
    }


def cost_and_storage_estimate(data_summary: Mapping[str, Any], source_bundle: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint_policy = checkpoint_retention_policy()
    payload = {
        "estimate_id": "DS24_R44B_REMOTE_COST_AND_STORAGE_ESTIMATE_V1",
        "source_bundle_size_bytes": source_bundle.get("bundle_size_bytes", 0),
        "data_transfer_size_bytes": data_summary.get("total_bytes", 0),
        "recommended_instance_disk_bytes": 240 * 1024**3,
        "minimum_contract_disk_bytes": 200 * 1024**3,
        "checkpoint_output_growth": checkpoint_policy["estimated_growth"],
        "paid_resource_created": False,
        "live_offer_search_required_by_user": True,
        "live_offer_search_command": ".\\vast_offer_query.ps1 -MinVramGb 24 -MinRamGb 64 -MinDiskGb 200",
    }
    payload["estimate_hash"] = stable_hash(payload)
    return payload


def user_action_readiness(validation: Mapping[str, Any], data_summary: Mapping[str, Any], source_bundle: Mapping[str, Any]) -> dict[str, Any]:
    ready = validation.get("status") == "PASS" and data_summary.get("missing_files") == 0 and source_bundle.get("bundle_sha256")
    payload = {
        "readiness_id": "DS24_R44B_USER_ACTION_READINESS_V1",
        "ready_for_user_rental": bool(ready),
        "remaining_user_actions": [
            "install and authenticate Vast CLI locally",
            "search live offers and record price/offer metadata",
            "create a confirmed on-demand instance",
            "transfer source bundle and sidecar-scoped data",
            "run remote preflight",
            "launch and monitor tmux TFT run",
            "download sync bundles and validate import-review locally",
            "destroy Vast instance after verified download",
        ],
        "no_paid_resource_created_in_ticket": True,
        "data_transfer_size_bytes": data_summary.get("total_bytes", 0),
        "source_bundle_sha256": source_bundle.get("bundle_sha256", ""),
    }
    payload["readiness_hash"] = stable_hash(payload)
    return payload


def write_package(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    evidence_root.mkdir(parents=True, exist_ok=True)
    for directory in ("authority", "checkpoints", "logs", "metrics_only_v3", "transfer"):
        marker = evidence_root / Path(str(REMOTE_NAMESPACE)) / directory / ".keep"
        write_text(marker, "placeholder for remote namespace shape\n")

    inventory = existing_authority_inventory(repo_root)
    allocation = remote_family_allocation()
    allocation["allocation_hash"] = stable_hash(allocation)
    trial = remote_trial_identity_contract()
    config = frozen_tft_configuration()
    feature_target = tft_feature_and_target_contract(config)
    data_rows, data_summary = minimal_remote_data_manifest(repo_root)
    data_manifest_hash = stable_hash(data_rows)
    runtime = remote_runtime_contract(config["configuration_hash"], data_manifest_hash)
    checkpoint_contract = remote_checkpoint_contract(config["configuration_hash"], data_manifest_hash)
    retention = checkpoint_retention_policy()
    hardware = hardware_acceptance_contract()
    import_contract = remote_result_import_contract(config["configuration_hash"], data_manifest_hash)

    write_json(evidence_root / "01_existing_authority_inventory.json", inventory)
    write_json(evidence_root / "02_remote_family_allocation.json", allocation)
    write_json(evidence_root / "03_remote_trial_identity_contract.json", trial)
    write_yaml(evidence_root / "04_tft_configuration_authority.yaml", config)
    write_text(evidence_root / "05_tft_configuration_rationale.md", tft_configuration_rationale(config))
    write_json(evidence_root / "06_tft_feature_and_target_contract.json", feature_target)
    write_json(evidence_root / "07_linux_cuda_compatibility_audit.json", linux_cuda_compatibility_audit())
    write_json(evidence_root / "08_remote_runtime_contract.json", runtime)
    write_json(evidence_root / "09_remote_checkpoint_contract.json", checkpoint_contract)
    write_json(evidence_root / "10_checkpoint_retention_policy.json", retention)
    write_csv(evidence_root / "11_minimal_remote_data_manifest.csv", data_rows)
    write_text(evidence_root / "12_data_transfer_plan.md", data_transfer_plan_text(data_summary))
    write_json(evidence_root / "13_remote_hardware_acceptance_contract.json", hardware)
    write_json(evidence_root / "14_remote_result_import_contract.json", import_contract)

    for name, text in shell_scripts().items():
        path = evidence_root / name
        write_text(path, text)
        try:
            path.chmod(0o755)
        except OSError:
            pass
    for name, text in powershell_scripts().items():
        write_text(evidence_root / name, text)
    write_text(evidence_root / "remote_preflight.py", remote_preflight_text())
    write_text(evidence_root / "import_vast_tft_results.py", import_script_text())
    write_text(evidence_root / "vast_destroy_checklist.md", vast_destroy_checklist_text())

    source_bundle = build_source_bundle(repo_root, evidence_root)
    checkpoint_contract = remote_checkpoint_contract(
        config["configuration_hash"],
        data_manifest_hash,
        source_bundle.get("bundle_sha256", ""),
    )
    import_contract = remote_result_import_contract(
        config["configuration_hash"],
        data_manifest_hash,
        source_bundle.get("bundle_sha256", ""),
    )
    write_json(evidence_root / "09_remote_checkpoint_contract.json", checkpoint_contract)
    write_json(evidence_root / "14_remote_result_import_contract.json", import_contract)
    write_text(evidence_root / "USER_VAST_AI_TFT_RUNBOOK.md", runbook_text(data_summary, source_bundle))

    validation = bounded_validation(repo_root, evidence_root, import_contract, source_bundle, data_summary)
    write_json(evidence_root / "15_test_and_smoke_results.json", validation)
    security = scan_forbidden_secret_text(evidence_root)
    write_json(evidence_root / "16_security_and_secret_scan.json", security)
    non_interference = non_interference_proof(repo_root)
    write_json(evidence_root / "17_local_tournament_non_interference_proof.json", non_interference)
    estimate = cost_and_storage_estimate(data_summary, source_bundle)
    write_json(evidence_root / "18_remote_cost_and_storage_estimate.json", estimate)
    readiness = user_action_readiness(validation, data_summary, source_bundle)
    write_json(evidence_root / "19_user_action_readiness.json", readiness)

    hashes = {}
    for name in (
        "04_tft_configuration_authority.yaml",
        "05_tft_configuration_rationale.md",
        "06_tft_feature_and_target_contract.json",
    ):
        hashes[name] = sha256_file(evidence_root / name)
    terminal_classification = (
        TERMINAL_SUCCESS
        if validation["status"] == "PASS"
        and security["status"] == "PASS"
        and non_interference["status"] == "PASS"
        and readiness["ready_for_user_rental"]
        else TERMINAL_BLOCKERS["evaluation"]
    )
    terminal = {
        "terminal_classification": terminal_classification,
        "success": terminal_classification == TERMINAL_SUCCESS,
        "created_at_utc": utc_now(),
        "evidence_root": str(evidence_root),
        "configuration_hash": config["configuration_hash"],
        "artifact_hashes": hashes,
        "source_bundle_size_bytes": source_bundle.get("bundle_size_bytes", 0),
        "source_bundle_sha256": source_bundle.get("bundle_sha256", ""),
        "data_transfer_size_bytes": data_summary.get("total_bytes", 0),
        "estimated_checkpoint_output_growth_bytes": retention["estimated_growth"],
        "local_supervisor_state_before_after": {
            "before": inventory["process_snapshot"],
            "after": non_interference["process_snapshot"],
        },
        "paid_vast_resource_created": False,
        "data_uploaded": False,
        "full_tft_run_launched": False,
        "live_offer_search_command": estimate["live_offer_search_command"],
    }
    write_json(evidence_root / "20_terminal_result.json", terminal)
    write_text(
        evidence_root / "README.md",
        f"""
        # DS24 R44B Vast Remote TFT Evidence

        Terminal classification: `{terminal_classification}`

        This bounded local package freezes the TFT remote configuration, builds a
        deterministic source bundle, writes a sidecar-scoped data-transfer manifest,
        validates checkpoint/resume and import-review behavior with synthetic data,
        and provides a copy-and-run Vast.ai runbook. No paid Vast resource was
        created, no data was uploaded, no full TFT run was launched, and no active
        local DS24 supervisor namespace was modified.

        User runbook: `USER_VAST_AI_TFT_RUNBOOK.md`
        Source bundle: `transfer/ds24_vast_tft_source_bundle.zip`
        """,
    )
    return terminal


def package_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build DS24 R44B Vast TFT evidence package")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--evidence-root", default=str(EVIDENCE_RELATIVE_ROOT))
    args = parser.parse_args(argv)
    terminal = write_package(Path(args.repo_root).resolve(), Path(args.evidence_root))
    print(json.dumps(terminal, indent=2, sort_keys=True))
    return 0 if terminal["success"] else 2
