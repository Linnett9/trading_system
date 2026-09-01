from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from core.research.ml.ds24.windows_safe_io import write_json_atomic


POLICY_ID = "DS24_METRICS_ONLY_RESEARCH_OUTPUT_POLICY_V1"
TARGET_ID = "forward_return_60m__decision_5m"
LEGACY_FULL_PREDICTION_CONTRACT = "DS24_COMPACT_DAILY_PREDICTION_BATCH_V1"
METRICS_ONLY_CONTRACT = "DS24_METRICS_ONLY_RESEARCH_OUTPUT_POLICY_V1"
RESOLVED_PERFORMANCE_CONTRACT_V2_ID = "DS24_RESOLVED_PERFORMANCE_METRICS_CONTRACT_V2"
RESOLVED_PERFORMANCE_CONTRACT_V2_VERSION = "V2"
RESOLVED_PERFORMANCE_CONTRACT_V3_ID = "DS24_RESOLVED_PERFORMANCE_V3_STAGGERED_SLEEVE_RANK_IC_CONTRACT"
RESOLVED_PERFORMANCE_CONTRACT_V3_VERSION = "V3"
EXTENDED_PERFORMANCE_METRICS_CONTRACT_ID = "DS24_EXTENDED_PERFORMANCE_METRICS_V1"
EXTENDED_PERFORMANCE_METRICS_CONTRACT_VERSION = "V1"
DEFAULT_TARGET_HORIZON_MINUTES = 60
DEFAULT_DECISION_CADENCE_MINUTES = 5
DEFAULT_MAX_ASSETS_PER_SCORING_TIMESTAMP = 514
DEFAULT_PENDING_SCORE_MAX_TIMESTAMPS = 18
DEFAULT_PENDING_SCORE_LIMIT = 80_000
DEFAULT_V3_SLEEVE_COUNT = int(DEFAULT_TARGET_HORIZON_MINUTES / DEFAULT_DECISION_CADENCE_MINUTES)
DEFAULT_V3_MIN_RANK_IC_CROSS_SECTION = 20
DEFAULT_V3_AUDIT_SAMPLE_ROWS_PER_TIMESTAMP = 20
TRANSIENT_STORAGE_CONTRACT_V1_ID = "DS24_V2_TRANSIENT_STORAGE_CONTRACT_V1"
TRANSIENT_STORAGE_CONTRACT_V1_VERSION = "V1"
MAX_TEMPORARY_DISK_BYTES_PER_WORKER = 1 * 1024**3
PREFERRED_TEMPORARY_DISK_BYTES_PER_WORKER = 512 * 1024**2
MAX_AGGREGATE_TOURNAMENT_TEMPORARY_DISK_BYTES = 3 * 1024**3
MIN_EXECUTION_FREE_DISK_BYTES = 12 * 1024**3
CLEAN_ADMISSION_FREE_DISK_BYTES = 15 * 1024**3
THREE_WORKER_REACTIVATION_FREE_DISK_BYTES = 18 * 1024**3
MIN_PROJECTED_POST_LAUNCH_FREE_DISK_BYTES = 15 * 1024**3
TOP_N_COST_BPS_PER_UNIT_TURNOVER = 0.0
TRADING_SESSIONS_PER_YEAR = 252
NAMESPACE_WRITER_LEASE_NAME = "namespace_writer_lease.json"
NAMESPACE_WRITER_ACQUIRE_LOCK_NAME = "namespace_writer_lease.acquire.lock.json"
APPEND_COMMIT_LOCK_NAME = "append_commit.lock.json"
CURRENT_PROCESS_CREATION_TIME = pd.Timestamp.now("UTC").isoformat()


PREDICTIVE_METRICS = [
    "spearman_rank_ic",
    "pearson_ic",
    "mae",
    "mse",
    "rmse",
    "directional_accuracy",
    "ndcg_at_top_n",
    "prediction_coverage",
    "missing_prediction_count",
    "duplicate_prediction_count",
    "score_count",
    "score_mean",
    "score_std",
    "score_min",
    "score_max",
    "score_q01",
    "score_q05",
    "score_q10",
    "score_q25",
    "score_q50",
    "score_q75",
    "score_q90",
    "score_q95",
    "score_q99",
    "score_tie_count",
]

ECONOMIC_METRICS = [
    "top_n",
    "selected_symbols",
    "weights",
    "gross_return",
    "net_return",
    "turnover",
    "transaction_cost_drag",
    "exposure",
    "cash_weight",
    "benchmark_return",
    "excess_return",
    "hit_rate",
    "cumulative_return",
    "annualized_return",
    "volatility",
    "sharpe",
    "sortino",
    "maximum_drawdown",
    "calmar",
    "win_loss_ratio",
]

ROBUSTNESS_METRICS = [
    "metric_by_year",
    "metric_by_volatility_regime",
    "metric_by_liquidity_bucket",
    "metric_by_sector",
    "metric_by_market_direction",
    "metric_by_time_of_day_bucket",
]

OPERATIONAL_METRICS = [
    "training_rows",
    "eligible_assets",
    "scoring_timestamps",
    "prediction_rows_generated",
    "refit_count",
    "failed_fits",
    "convergence",
    "training_time_seconds",
    "scoring_time_seconds",
    "cpu_seconds",
    "ram_bytes",
    "checkpoint_count",
    "resume_generation",
    "gap_count",
    "duplicate_count",
    "output_bytes",
]

FAMILY_DIAGNOSTICS = [
    "ridge_coefficients_digest",
    "pca_variance_digest",
    "elastic_net_sparsity",
    "tree_feature_importance_digest",
    "ranking_group_validation",
    "sequence_training_loss",
]


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(payload: Mapping[str, Any] | Sequence[Any] | str) -> str:
    if isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def openable_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def openable_exists(path: Path) -> bool:
    return os.path.exists(openable_path(path))


def mkdir_openable(path: Path) -> None:
    os.makedirs(openable_path(path), exist_ok=True)


def _utc_now_iso() -> str:
    return pd.Timestamp.now("UTC").isoformat()


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalise_process_creation_time(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        if isinstance(value, (int, float)):
            return pd.Timestamp(float(value), unit="s", tz="UTC").isoformat()
        return _normalise_timestamp(value).isoformat()
    except Exception:
        return str(value)


def _same_process_creation_time(left: Any, right: Any) -> bool:
    left_text = _normalise_process_creation_time(left)
    right_text = _normalise_process_creation_time(right)
    if not left_text or not right_text:
        return False
    try:
        left_ts = pd.Timestamp(left_text).tz_convert("UTC")
        right_ts = pd.Timestamp(right_text).tz_convert("UTC")
        return abs((left_ts - right_ts).total_seconds()) < 1.0
    except Exception:
        return left_text == right_text


def process_identity_for_namespace_lease(pid: int) -> dict[str, Any]:
    if int(pid or 0) <= 0:
        return {"pid": int(pid or 0), "alive": False, "process_creation_time": ""}
    try:
        import psutil  # type: ignore[import-not-found]

        proc = psutil.Process(int(pid))
        return {
            "pid": int(pid),
            "alive": bool(proc.is_running()),
            "process_creation_time": _normalise_process_creation_time(proc.create_time()),
        }
    except Exception:
        if os.name == "nt":
            script = (
                f"$p=Get-CimInstance Win32_Process -Filter \"ProcessId={int(pid)}\" -ErrorAction SilentlyContinue; "
                "if ($p) { @{alive=$true;process_creation_time=$p.CreationDate.ToUniversalTime().ToString('o')} | ConvertTo-Json -Compress } "
                "else { @{alive=$false;process_creation_time=''} | ConvertTo-Json -Compress }"
            )
            try:
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", script],
                    text=True,
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
                payload = json.loads(result.stdout.strip() or "{}")
                return {
                    "pid": int(pid),
                    "alive": bool(payload.get("alive")),
                    "process_creation_time": _normalise_process_creation_time(payload.get("process_creation_time", "")),
                }
            except Exception:
                pass
        if int(pid) == os.getpid():
            return {"pid": int(pid), "alive": True, "process_creation_time": CURRENT_PROCESS_CREATION_TIME}
        if os.name != "nt":
            try:
                os.kill(int(pid), 0)
                return {"pid": int(pid), "alive": True, "process_creation_time": ""}
            except OSError:
                pass
        return {"pid": int(pid), "alive": False, "process_creation_time": ""}


def _lease_hash_parts(*parts: Any) -> str:
    return stable_hash([str(part) for part in parts])


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    mkdir_openable(path.parent)
    raw = (json.dumps(dict(payload), sort_keys=True, default=str, separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(openable_path(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    try:
        os.write(fd, raw)
    finally:
        os.close(fd)


def _unlink_if_matching_token(path: Path, token: str) -> bool:
    payload = _read_json_mapping(path)
    if str(payload.get("token", "")) != str(token):
        return False
    try:
        os.unlink(openable_path(path))
        return True
    except FileNotFoundError:
        return True


class AtomicJsonProcessLock:
    def __init__(self, path: Path, *, family: str, purpose: str, namespace: str, timeout_seconds: float = 0.0) -> None:
        self.path = path
        self.family = family
        self.purpose = purpose
        self.namespace = namespace
        self.timeout_seconds = float(timeout_seconds)
        self.token = _lease_hash_parts(os.getpid(), time.time_ns(), path, purpose)
        self.payload: dict[str, Any] = {}

    def acquire(self) -> dict[str, Any]:
        started = time.monotonic()
        while True:
            payload = {
                "contract": "DS24_R37_ATOMIC_JSON_PROCESS_LOCK_V1",
                "family": self.family,
                "purpose": self.purpose,
                "namespace": self.namespace,
                "pid": os.getpid(),
                "process_creation_time": process_identity_for_namespace_lease(os.getpid()).get("process_creation_time", ""),
                "hostname": socket.gethostname(),
                "token": self.token,
                "acquired_at_utc": _utc_now_iso(),
            }
            try:
                _write_json_exclusive(self.path, payload)
                self.payload = payload
                return payload
            except FileExistsError:
                existing = _read_json_mapping(self.path)
                owner_pid = int(existing.get("pid", 0) or 0)
                if owner_pid == os.getpid():
                    if str(existing.get("token", "")) == self.token:
                        self.payload = dict(existing)
                        return dict(existing)
                    raise RuntimeError(
                        "DS24_R37_APPEND_COMMIT_LOCK_LIVE_OWNER_REFUSED:"
                        f"family={self.family};namespace={self.namespace};owner_pid={owner_pid};purpose={self.purpose}"
                    )
                identity = process_identity_for_namespace_lease(owner_pid)
                owner_alive = bool(identity.get("alive"))
                stored_creation = existing.get("process_creation_time", "")
                observed_creation = identity.get("process_creation_time", "")
                owner_verified = owner_alive and _same_process_creation_time(stored_creation, observed_creation)
                owner_unverified = owner_alive and (not stored_creation or not observed_creation)
                if owner_verified or owner_unverified:
                    if time.monotonic() - started >= self.timeout_seconds:
                        raise RuntimeError(
                            "DS24_R37_APPEND_COMMIT_LOCK_LIVE_OWNER_REFUSED:"
                            f"family={self.family};namespace={self.namespace};owner_pid={owner_pid};purpose={self.purpose}"
                        )
                    time.sleep(0.1)
                    continue
                try:
                    os.unlink(openable_path(self.path))
                except FileNotFoundError:
                    pass
                except PermissionError:
                    if time.monotonic() - started >= self.timeout_seconds:
                        raise
                    time.sleep(0.1)

    def release(self) -> bool:
        return _unlink_if_matching_token(self.path, self.token)

    def __enter__(self) -> "AtomicJsonProcessLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


class NamespaceWriterLease:
    def __init__(
        self,
        root: Path,
        *,
        family: str,
        resume_generation: int | str | None = None,
        command_hash: str | None = None,
        configuration_hash: str | None = None,
        evaluation_contract_hash: str | None = None,
        namespace: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.family = family
        self.resume_generation = resume_generation
        self.command_hash = command_hash or stable_hash(" ".join(sys.argv))
        self.configuration_hash = configuration_hash or ""
        self.evaluation_contract_hash = evaluation_contract_hash or ""
        self.namespace = namespace or str(self.root.resolve())
        self.path = self.root / NAMESPACE_WRITER_LEASE_NAME
        self._acquire_lock_path = self.root / NAMESPACE_WRITER_ACQUIRE_LOCK_NAME
        self.token = _lease_hash_parts(self.family, os.getpid(), time.time_ns(), self.namespace)
        self.payload: dict[str, Any] = {}

    def acquire(self) -> dict[str, Any]:
        mkdir_openable(self.root)
        with AtomicJsonProcessLock(
            self._acquire_lock_path,
            family=self.family,
            purpose="namespace_writer_lease_acquire",
            namespace=self.namespace,
            timeout_seconds=5.0,
        ):
            existing = _read_json_mapping(self.path)
            previous_owner = dict(existing)
            recovery_reason = ""
            if existing:
                existing_namespace = str(existing.get("namespace", "") or "")
                if existing_namespace and existing_namespace != self.namespace:
                    raise RuntimeError(
                        "DS24_R37_NAMESPACE_LEASE_IDENTITY_MISMATCH:"
                        f"family={self.family};expected={self.namespace};found={existing_namespace}"
                    )
                owner_pid = int(existing.get("pid", 0) or 0)
                if owner_pid == os.getpid():
                    if str(existing.get("token", "")) == self.token:
                        self.payload = dict(existing)
                        return dict(existing)
                    raise RuntimeError(
                        "DS24_R37_NAMESPACE_LEASE_LIVE_OWNER_REFUSED:"
                        f"family={self.family};namespace={self.namespace};owner_pid={owner_pid}"
                    )
                identity = process_identity_for_namespace_lease(owner_pid)
                stored_creation = existing.get("process_creation_time", "")
                observed_creation = identity.get("process_creation_time", "")
                if owner_pid and identity.get("alive"):
                    if not stored_creation or not observed_creation:
                        raise RuntimeError(
                            "DS24_R37_NAMESPACE_LEASE_OWNER_IDENTITY_UNVERIFIED:"
                            f"family={self.family};namespace={self.namespace};owner_pid={owner_pid}"
                        )
                    if _same_process_creation_time(stored_creation, observed_creation):
                        raise RuntimeError(
                            "DS24_R37_NAMESPACE_LEASE_LIVE_OWNER_REFUSED:"
                            f"family={self.family};namespace={self.namespace};owner_pid={owner_pid}"
                        )
                    recovery_reason = "PID_REUSE_VERIFIED_PREVIOUS_OWNER_NOT_MATCHING_CREATION_TIME"
                else:
                    recovery_reason = "STALE_PID_NOT_ALIVE"
            current_identity = process_identity_for_namespace_lease(os.getpid())
            generation = int(existing.get("lease_generation", 0) or 0) + 1 if existing else 1
            payload = {
                "contract": "DS24_R37_V3_NAMESPACE_SINGLE_WRITER_LEASE_V1",
                "family": self.family,
                "pid": int(os.getpid()),
                "process_creation_time": current_identity.get("process_creation_time", ""),
                "hostname": socket.gethostname(),
                "resume_generation": self.resume_generation,
                "command_hash": self.command_hash,
                "configuration_hash": self.configuration_hash,
                "evaluation_contract_hash": self.evaluation_contract_hash,
                "namespace": self.namespace,
                "namespace_root": str(self.root),
                "acquired_at_utc": _utc_now_iso(),
                "heartbeat_utc": _utc_now_iso(),
                "heartbeat_count": 0,
                "phase": "LEASE_ACQUIRED",
                "cursor": "",
                "lease_generation": generation,
                "token": self.token,
                "stale_recovered": bool(existing),
                "stale_recovery_reason": recovery_reason,
                "previous_owner": previous_owner,
            }
            payload["lease_hash"] = stable_hash(payload)
            write_json_atomic(self.path, payload, advisory=False)
            self.payload = payload
            return dict(payload)

    def assert_owner(self) -> None:
        current = _read_json_mapping(self.path)
        if str(current.get("token", "")) != self.token or int(current.get("pid", 0) or 0) != os.getpid():
            raise RuntimeError(
                "DS24_R37_NAMESPACE_LEASE_OWNER_LOST:"
                f"family={self.family};namespace={self.namespace};owner_pid={current.get('pid')}"
            )
        if str(current.get("namespace", "") or "") != self.namespace:
            raise RuntimeError(
                "DS24_R37_NAMESPACE_LEASE_IDENTITY_CHANGED:"
                f"family={self.family};namespace={self.namespace};found={current.get('namespace')}"
            )

    def heartbeat(self, *, phase: str = "", cursor: str = "") -> dict[str, Any]:
        self.assert_owner()
        payload = _read_json_mapping(self.path)
        payload["heartbeat_utc"] = _utc_now_iso()
        payload["heartbeat_count"] = int(payload.get("heartbeat_count", 0) or 0) + 1
        if phase:
            payload["phase"] = phase
        if cursor:
            payload["cursor"] = cursor
        payload["lease_hash"] = stable_hash({key: value for key, value in payload.items() if key != "lease_hash"})
        write_json_atomic(self.path, payload, advisory=False)
        self.payload = payload
        return dict(payload)

    def release(self) -> bool:
        current = _read_json_mapping(self.path)
        if str(current.get("token", "")) != self.token:
            return False
        write_json_atomic(
            self.root / "namespace_writer_lease_released.json",
            {
                "family": self.family,
                "pid": os.getpid(),
                "namespace": self.namespace,
                "lease_generation": current.get("lease_generation"),
                "released_at_utc": _utc_now_iso(),
                "token": self.token,
            },
            advisory=False,
        )
        try:
            os.unlink(openable_path(self.path))
            return True
        except FileNotFoundError:
            return True


def metrics_registry() -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "target_id": TARGET_ID,
        "retention_rule": "discard_full_predictions_after_required_metrics_and_pending_buffer_commit",
        "prediction_generation_required": "full_eligible_universe_at_every_authorised_T",
        "prediction_persistence_forbidden": "unbounded_full_history_asset_level_predictions",
        "predictive_metrics": PREDICTIVE_METRICS,
        "economic_metrics": ECONOMIC_METRICS,
        "robustness_metrics": ROBUSTNESS_METRICS,
        "operational_metrics": OPERATIONAL_METRICS,
        "family_diagnostics": FAMILY_DIAGNOSTICS,
        "top_n_values": [20],
        "score_quantiles": [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
        "pending_buffer_scope": "registered_forward_horizon_plus_one_checkpoint_rollback",
        "checkpoint_retention": "current_plus_previous",
        "holdout_access": "forbidden",
        "paper_orders": 0,
        "live_orders": 0,
    }


def policy_payload() -> dict[str, Any]:
    registry = metrics_registry()
    return {
        "authority_id": POLICY_ID,
        "authority_hash": stable_hash(registry),
        "supersedes": LEGACY_FULL_PREDICTION_CONTRACT,
        "does_not_change": [
            "model_inputs",
            "model_architecture",
            "training_samples",
            "refit_calendar",
            "eligible_universe",
            "prediction_cadence",
            "target_definition",
            "target_availability_chronology",
            "evaluation_population",
            "transaction_cost_assumptions",
            "holdout_boundary",
            "queue_ordering",
            "promotion_rules",
        ],
        "registry": registry,
    }


def policy_hash() -> str:
    return policy_payload()["authority_hash"]


def resolved_performance_contract_v2_registry() -> dict[str, Any]:
    return {
        "contract_id": RESOLVED_PERFORMANCE_CONTRACT_V2_ID,
        "version": RESOLVED_PERFORMANCE_CONTRACT_V2_VERSION,
        "target_id": TARGET_ID,
        "target_horizon_minutes": DEFAULT_TARGET_HORIZON_MINUTES,
        "retention_rule": "bounded_unresolved_cross_sectional_scores_only",
        "resolved_retention_rule": "compact_per_timestamp_metrics_no_asset_level_score_history",
        "pending_buffer_scope": "unresolved_until_target_maturity_or_terminal_censor",
        "pending_score_limit_rows_default": DEFAULT_PENDING_SCORE_LIMIT,
        "rank_ic_population": "complete_resolved_eligible_cross_section",
        "portfolio_policy": "registered_top_bottom_n_equal_weight",
        "top_n_values": [20],
        "transaction_cost_bps_per_unit_turnover": TOP_N_COST_BPS_PER_UNIT_TURNOVER,
        "target_availability_rule": "target_available_timestamp <= resolution_timestamp",
        "forbidden_outputs": [
            "permanent_full_cross_sectional_prediction_history",
            "resolved_asset_level_score_target_history",
            "holdout_outputs",
            "paper_orders",
            "live_orders",
        ],
        "does_not_change": [
            "model_hyperparameters",
            "training_window",
            "feature_inputs",
            "scoring_calendar",
            "refit_cadence",
            "asset_eligibility",
            "target_definition",
            "portfolio_construction",
        ],
    }


def resolved_performance_contract_v2_payload() -> dict[str, Any]:
    registry = resolved_performance_contract_v2_registry()
    return {
        "authority_id": RESOLVED_PERFORMANCE_CONTRACT_V2_ID,
        "authority_hash": stable_hash(registry),
        "version": RESOLVED_PERFORMANCE_CONTRACT_V2_VERSION,
        "supersedes": POLICY_ID,
        "registry": registry,
    }


def resolved_performance_contract_v2_hash() -> str:
    return resolved_performance_contract_v2_payload()["authority_hash"]


def resolved_performance_contract_v3_registry() -> dict[str, Any]:
    return {
        "contract_id": RESOLVED_PERFORMANCE_CONTRACT_V3_ID,
        "version": RESOLVED_PERFORMANCE_CONTRACT_V3_VERSION,
        "target_id": TARGET_ID,
        "target_horizon_minutes": DEFAULT_TARGET_HORIZON_MINUTES,
        "decision_cadence_minutes": DEFAULT_DECISION_CADENCE_MINUTES,
        "training_execution_cadence": "daily_session_refit_with_five_minute_scoring",
        "refit_rule": "fit at most once per market session then reuse that strictly PIT model for all five-minute decisions in the session",
        "rank_ic_population": "eligible_cross_section_at_each_decision_timestamp_with_matured_registered_forward_target",
        "rank_ic_tie_method": "average_rank",
        "minimum_rank_ic_cross_section": DEFAULT_V3_MIN_RANK_IC_CROSS_SECTION,
        "rank_ic_inference": "Newey-West HAC confidence interval over timestamp IC with lag at least horizon/cadence because adjacent 60m outcomes overlap",
        "rank_ic_retention": [
            "per_timestamp_sufficient_statistics",
            "per_timestamp_rank_hash",
            "bounded_deterministic_score_target_rank_audit_sample",
        ],
        "portfolio_contract": "twelve_staggered_equal_capital_sleeves_for_60m_horizon_5m_decisions",
        "sleeve_count": DEFAULT_V3_SLEEVE_COUNT,
        "simultaneous_capital_limit": 1.0,
        "rebalance_rule": "one sleeve rebalances at each five-minute decision timestamp",
        "daily_return_rule": "sum matured sleeve gross/net contributions by maturity session date, then compound only daily portfolio returns",
        "transaction_cost_bps_per_unit_turnover": TOP_N_COST_BPS_PER_UNIT_TURNOVER,
        "top_n_values": [20],
        "bottom_n_trace_retained": True,
        "pending_retention": "bounded_unresolved_cross_sectional_scores_until_target_maturity_or_terminal_censor",
        "pending_timestamp_limit": DEFAULT_PENDING_SCORE_MAX_TIMESTAMPS,
        "audit_sample_rows_per_timestamp": DEFAULT_V3_AUDIT_SAMPLE_ROWS_PER_TIMESTAMP,
        "growing_outputs": "immutable_append_only_parquet_partitions_with_manifest_and_partition_level_atomicity",
        "forbidden_outputs": [
            "permanent_full_cross_sectional_prediction_history",
            "full_history_parquet_rewrite",
            "resolved_asset_level_full_score_target_history",
            "holdout_outputs",
            "paper_orders",
            "live_orders",
        ],
        "does_not_change": [
            "model_hyperparameters",
            "training_window",
            "feature_inputs",
            "scoring_calendar",
            "asset_eligibility",
            "target_definition",
            "family_parameters",
        ],
    }


def resolved_performance_contract_v3_payload() -> dict[str, Any]:
    registry = resolved_performance_contract_v3_registry()
    return {
        "authority_id": RESOLVED_PERFORMANCE_CONTRACT_V3_ID,
        "authority_hash": stable_hash(registry),
        "version": RESOLVED_PERFORMANCE_CONTRACT_V3_VERSION,
        "supersedes": RESOLVED_PERFORMANCE_CONTRACT_V2_ID,
        "registry": registry,
    }


def resolved_performance_contract_v3_hash() -> str:
    return resolved_performance_contract_v3_payload()["authority_hash"]


def extended_performance_metrics_contract_registry() -> dict[str, Any]:
    return {
        "contract_id": EXTENDED_PERFORMANCE_METRICS_CONTRACT_ID,
        "version": EXTENDED_PERFORMANCE_METRICS_CONTRACT_VERSION,
        "extends": RESOLVED_PERFORMANCE_CONTRACT_V3_ID,
        "target_id": TARGET_ID,
        "admission_rule": "future_family_must_initialise_required_metrics_writer_before_first_prediction",
        "activation_scope": "forward_only_safe_for_controlled_adoption",
        "mandatory_per_timestamp_fields": [
            "family",
            "policy_hash",
            "decision_timestamp",
            "target_maturity_timestamp",
            "target_id",
            "evaluation_contract_hash",
            "eligible_asset_count",
            "resolved_asset_count",
            "spearman_rank_ic",
            "pearson_ic",
            "mae",
            "mse",
            "rmse",
            "directional_accuracy",
            "ndcg_at_top_n",
            "prediction_coverage",
            "missing_prediction_count",
            "duplicate_prediction_count",
            "top20_identifiers",
            "top20_ranks",
            "top20_scores",
            "top20_weights",
            "gross_return",
            "benchmark_return",
            "turnover",
            "transaction_cost_drag",
            "net_return",
            "excess_return",
            "training_rows",
            "training_assets",
            "train_start_timestamp",
            "train_end_timestamp",
            "training_history_days",
            "refit_id",
            "convergence_state",
        ],
        "mandatory_artifacts": [
            "rank_ic_v3",
            "pearson_ic",
            "ic_confidence_intervals",
            "time_bucketed_ic",
            "rolling_ic",
            "coverage_and_maturity_counts",
            "economic_performance",
            "turnover",
            "transaction_cost_sensitivity",
            "mae_mse_rmse",
            "directional_accuracy",
            "ndcg_or_family_ranking_metric",
            "refit_training_history_metadata",
            "evaluation_contract_identity",
        ],
        "atomic_retention_gate_requirements": [
            "outcome_mature",
            "eligible_and_resolved_populations_reconcile",
            "full_universe_ic_non_null",
            "required_predictive_metrics_committed",
            "economic_metrics_committed",
            "target_and_evaluation_contract_hashes_recorded",
            "compact_manifest_and_file_hashes_committed",
            "top20_decision_trace_committed",
            "pending_buffer_coverage_sufficient",
            "restart_safe_idempotent_commit",
        ],
        "source_retention_failure_classification": "METRICS_INCOMPLETE_SOURCE_RETAINED",
        "full_prediction_discard_rule": "discard_only_after_atomic_retention_gate_passes",
        "forbidden": [
            "top20_only_ic_as_headline_full_universe_ic",
            "full_prediction_deletion_before_resolved_metrics_commit",
            "holdout_access",
            "paper_orders",
            "live_orders",
        ],
        "storage": {
            "unbounded_full_predictions": "forbidden",
            "allowed_pending_scope": "bounded_unresolved_horizon_plus_rollback_checkpoint",
            "preferred_layout": "bounded_daily_or_session_parquet_shards_with_manifests",
            "one_file_per_timestamp": "forbidden",
        },
        "does_not_change": resolved_performance_contract_v3_registry()["does_not_change"],
    }


def extended_performance_metrics_contract_payload() -> dict[str, Any]:
    registry = extended_performance_metrics_contract_registry()
    return {
        "authority_id": EXTENDED_PERFORMANCE_METRICS_CONTRACT_ID,
        "authority_hash": stable_hash(registry),
        "version": EXTENDED_PERFORMANCE_METRICS_CONTRACT_VERSION,
        "extends": RESOLVED_PERFORMANCE_CONTRACT_V3_ID,
        "base_contract_hash": resolved_performance_contract_v3_hash(),
        "registry": registry,
    }


def extended_performance_metrics_contract_hash() -> str:
    return extended_performance_metrics_contract_payload()["authority_hash"]


def validate_extended_metrics_writer_capability(
    capability: Mapping[str, Any] | None,
    *,
    family: str,
) -> dict[str, Any]:
    evidence = dict(capability or {})
    missing: list[str] = []
    if str(evidence.get("family", family)) != str(family):
        missing.append("family_identity")
    if evidence.get("writer_initialised_before_first_prediction") is not True:
        missing.append("writer_initialised_before_first_prediction")
    if str(evidence.get("base_evaluation_contract_hash", "")) != resolved_performance_contract_v3_hash():
        missing.append("base_evaluation_contract_hash")
    if str(evidence.get("extended_contract_hash", "")) != extended_performance_metrics_contract_hash():
        missing.append("extended_contract_hash")
    fields = {str(item) for item in evidence.get("available_per_timestamp_fields", [])}
    required = set(extended_performance_metrics_contract_registry()["mandatory_per_timestamp_fields"])
    missing.extend(sorted(required - fields))
    artifacts = {str(item) for item in evidence.get("available_artifacts", [])}
    required_artifacts = set(extended_performance_metrics_contract_registry()["mandatory_artifacts"])
    missing.extend(sorted(f"artifact:{item}" for item in (required_artifacts - artifacts)))
    if evidence.get("atomic_retention_gate_available") is not True:
        missing.append("atomic_retention_gate_available")
    return {
        "family": str(family),
        "admitted": not missing,
        "classification": "ADMITTED_EXTENDED_PERFORMANCE_METRICS_READY" if not missing else "FUTURE_FAMILY_ADMISSION_FAILED_METRICS_CAPABILITY_MISSING",
        "missing_requirements": missing,
        "contract_id": EXTENDED_PERFORMANCE_METRICS_CONTRACT_ID,
        "contract_hash": extended_performance_metrics_contract_hash(),
    }


def enforce_future_family_metrics_admission(capability: Mapping[str, Any] | None, *, family: str) -> dict[str, Any]:
    result = validate_extended_metrics_writer_capability(capability, family=family)
    if not result["admitted"]:
        raise RuntimeError(
            "DS24_EXTENDED_PERFORMANCE_METRICS_ADMISSION_FAILED:"
            f"family={family};missing={','.join(result['missing_requirements'])}"
        )
    return result


def evaluate_atomic_retention_gate(evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    state = dict(evidence or {})
    requirements = list(extended_performance_metrics_contract_registry()["atomic_retention_gate_requirements"])
    failed = [name for name in requirements if state.get(name) is not True]
    return {
        "passed": not failed,
        "classification": "METRICS_COMPLETE_SOURCE_DISCARD_ALLOWED" if not failed else "METRICS_INCOMPLETE_SOURCE_RETAINED",
        "failed_requirements": failed,
        "source_must_be_retained": bool(failed),
        "contract_id": EXTENDED_PERFORMANCE_METRICS_CONTRACT_ID,
        "contract_hash": extended_performance_metrics_contract_hash(),
    }


def validate_feature_availability(
    features: pd.DataFrame,
    *,
    decision_col: str = "decision_timestamp",
    availability_col: str = "feature_available_timestamp",
) -> dict[str, Any]:
    required = {decision_col, availability_col}
    missing = sorted(required - set(features.columns))
    if missing:
        return {"valid": False, "reason": "missing_feature_availability_columns", "missing_columns": missing}
    frame = features.copy()
    decisions = _normalise_timestamp_series(frame[decision_col])
    available = _normalise_timestamp_series(frame[availability_col])
    future = available.notna() & decisions.notna() & (available > decisions)
    return {
        "valid": not bool(future.any()),
        "reason": "" if not bool(future.any()) else "future_dated_feature_availability",
        "row_count": int(len(frame)),
        "future_dated_feature_rows": int(future.sum()),
    }


def enforce_feature_availability(
    features: pd.DataFrame,
    *,
    decision_col: str = "decision_timestamp",
    availability_col: str = "feature_available_timestamp",
) -> dict[str, Any]:
    result = validate_feature_availability(features, decision_col=decision_col, availability_col=availability_col)
    if not result["valid"]:
        raise ValueError(f"DS24_FEATURE_AVAILABILITY_VIOLATION:{result['reason']}:{result.get('future_dated_feature_rows', 0)}")
    return result


def validate_train_score_boundary(
    training: pd.DataFrame,
    *,
    score_start_timestamp: Any,
    timestamp_col: str = "decision_timestamp",
    label_available_col: str = "target_available_timestamp",
) -> dict[str, Any]:
    missing = [column for column in [timestamp_col, label_available_col] if column not in training.columns]
    if missing:
        return {"valid": False, "reason": "missing_train_score_boundary_columns", "missing_columns": missing}
    score_start = _normalise_timestamp(score_start_timestamp)
    train_timestamps = _normalise_timestamp_series(training[timestamp_col])
    label_available = _normalise_timestamp_series(training[label_available_col])
    overlap = train_timestamps.notna() & (train_timestamps >= score_start)
    immature_labels = label_available.notna() & (label_available > score_start)
    return {
        "valid": not bool(overlap.any() or immature_labels.any()),
        "reason": "" if not bool(overlap.any() or immature_labels.any()) else "train_score_overlap_or_immature_label",
        "row_count": int(len(training)),
        "train_score_overlap_rows": int(overlap.sum()),
        "immature_training_label_rows": int(immature_labels.sum()),
        "score_start_timestamp": score_start.isoformat(),
    }


def enforce_train_score_boundary(
    training: pd.DataFrame,
    *,
    score_start_timestamp: Any,
    timestamp_col: str = "decision_timestamp",
    label_available_col: str = "target_available_timestamp",
) -> dict[str, Any]:
    result = validate_train_score_boundary(
        training,
        score_start_timestamp=score_start_timestamp,
        timestamp_col=timestamp_col,
        label_available_col=label_available_col,
    )
    if not result["valid"]:
        raise ValueError(
            "DS24_TRAIN_SCORE_BOUNDARY_VIOLATION:"
            f"overlap={result.get('train_score_overlap_rows', 0)};"
            f"immature={result.get('immature_training_label_rows', 0)}"
        )
    return result


def exact_common_timestamp_intersection(
    frames: Mapping[str, pd.DataFrame],
    *,
    timestamp_col: str = "decision_timestamp",
) -> dict[str, Any]:
    timestamp_sets: dict[str, set[str]] = {}
    for family, frame in frames.items():
        if frame.empty or timestamp_col not in frame.columns:
            timestamp_sets[str(family)] = set()
            continue
        timestamps = _normalise_timestamp_series(frame[timestamp_col]).dropna().map(lambda ts: ts.isoformat())
        timestamp_sets[str(family)] = set(timestamps)
    common = set.intersection(*timestamp_sets.values()) if timestamp_sets else set()
    return {
        "families": sorted(timestamp_sets),
        "common_timestamps": sorted(common),
        "common_timestamp_count": int(len(common)),
        "family_timestamp_counts": {family: int(len(values)) for family, values in sorted(timestamp_sets.items())},
    }


def transaction_cost_sensitivity_from_sleeves(
    sleeves: pd.DataFrame,
    bps_values: Sequence[float],
    *,
    trading_sessions_per_year: int = TRADING_SESSIONS_PER_YEAR,
) -> list[dict[str, Any]]:
    if sleeves.empty:
        return []
    frame = sleeves.copy()
    required = {"maturity_session_date", "gross_return_contribution", "turnover", "sleeve_capital_fraction"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"DS24_TRANSACTION_COST_SENSITIVITY_MISSING_COLUMNS:{missing}")
    frame["gross_return_contribution"] = _finite_numeric(frame["gross_return_contribution"]).fillna(0.0)
    frame["turnover"] = _finite_numeric(frame["turnover"]).fillna(0.0)
    frame["sleeve_capital_fraction"] = _finite_numeric(frame["sleeve_capital_fraction"]).fillna(1.0 / DEFAULT_V3_SLEEVE_COUNT)
    results: list[dict[str, Any]] = []
    for bps in bps_values:
        work = frame.copy()
        work["cost_drag"] = work["turnover"] * work["sleeve_capital_fraction"] * (float(bps) / 10000.0)
        work["net_return_contribution"] = work["gross_return_contribution"] - work["cost_drag"]
        daily = work.groupby("maturity_session_date", sort=True).agg(
            gross_daily_return=("gross_return_contribution", "sum"),
            net_daily_return=("net_return_contribution", "sum"),
            transaction_cost_drag=("cost_drag", "sum"),
            mean_turnover=("turnover", "mean"),
        )
        net = daily["net_daily_return"].astype(float)
        gross = daily["gross_daily_return"].astype(float)
        vol = float(net.std(ddof=1) * math.sqrt(trading_sessions_per_year)) if len(net) > 1 else None
        annualized = float(net.mean() * trading_sessions_per_year) if len(net) else None
        row = {
            "transaction_cost_bps": float(bps),
            "daily_return_rows": int(len(daily)),
            "cumulative_gross_return": float((1.0 + gross).prod() - 1.0) if len(gross) else None,
            "cumulative_net_return": float((1.0 + net).prod() - 1.0) if len(net) else None,
            "annualized_net_return_from_daily_returns": annualized,
            "annualized_volatility_from_daily_returns": vol,
            "sharpe_from_daily_returns": annualized / vol if annualized is not None and vol and vol > 0 else None,
            "maximum_drawdown": maximum_drawdown(net),
            "daily_win_rate": float((net > 0).mean()) if len(net) else None,
            "mean_turnover": float(daily["mean_turnover"].mean()) if len(daily) else None,
            "total_transaction_cost_drag": float(daily["transaction_cost_drag"].sum()) if len(daily) else 0.0,
            "raw_overlapping_forward_returns_annualized": False,
        }
        results.append(row)
    return results


def transient_storage_contract_v1_registry() -> dict[str, Any]:
    horizon_timestamps = int(DEFAULT_TARGET_HORIZON_MINUTES / DEFAULT_DECISION_CADENCE_MINUTES)
    pending_timestamp_limit = DEFAULT_PENDING_SCORE_MAX_TIMESTAMPS
    return {
        "contract_id": TRANSIENT_STORAGE_CONTRACT_V1_ID,
        "version": TRANSIENT_STORAGE_CONTRACT_V1_VERSION,
        "maximum_temporary_disk_per_worker_bytes": MAX_TEMPORARY_DISK_BYTES_PER_WORKER,
        "preferred_steady_state_temporary_disk_per_worker_bytes": PREFERRED_TEMPORARY_DISK_BYTES_PER_WORKER,
        "maximum_aggregate_tournament_temporary_disk_bytes": MAX_AGGREGATE_TOURNAMENT_TEMPORARY_DISK_BYTES,
        "minimum_free_disk_during_execution_bytes": MIN_EXECUTION_FREE_DISK_BYTES,
        "clean_admission_floor_bytes": CLEAN_ADMISSION_FREE_DISK_BYTES,
        "three_worker_reactivation_floor_bytes": THREE_WORKER_REACTIVATION_FREE_DISK_BYTES,
        "minimum_projected_post_launch_free_disk_bytes": MIN_PROJECTED_POST_LAUNCH_FREE_DISK_BYTES,
        "target_horizon_minutes": DEFAULT_TARGET_HORIZON_MINUTES,
        "decision_cadence_minutes": DEFAULT_DECISION_CADENCE_MINUTES,
        "unresolved_horizon_timestamps": horizon_timestamps,
        "pending_timestamp_limit": pending_timestamp_limit,
        "pending_logical_unit": "asset_score_row_with_decision_timestamp_asset_id_score_rank_and_model_lineage",
        "legacy_pending_score_limit_rows": DEFAULT_PENDING_SCORE_LIMIT,
        "active_pending_score_limit_rule": "pending_timestamp_limit * observed_max_assets_per_timestamp",
        "default_max_assets_per_timestamp": DEFAULT_MAX_ASSETS_PER_SCORING_TIMESTAMP,
        "default_active_pending_score_limit_rows": pending_timestamp_limit * DEFAULT_MAX_ASSETS_PER_SCORING_TIMESTAMP,
        "resolved_metrics_retention": "append_only_immutable_parquet_partitions_plus_manifest",
        "legacy_single_file_read_support": True,
        "full_history_rewrite_forbidden": True,
        "worker_temporary_root": "metrics_only/transient_tmp",
        "orphan_tmp_recovery": "delete_or_refuse_stale_tmp_files_before_worker_resume_after_hash_inventory",
    }


def transient_storage_contract_v1_payload() -> dict[str, Any]:
    registry = transient_storage_contract_v1_registry()
    return {
        "authority_id": TRANSIENT_STORAGE_CONTRACT_V1_ID,
        "authority_hash": stable_hash(registry),
        "version": TRANSIENT_STORAGE_CONTRACT_V1_VERSION,
        "registry": registry,
    }


def transient_storage_contract_v1_hash() -> str:
    return transient_storage_contract_v1_payload()["authority_hash"]


def publish_transient_storage_contract_v1(path: Path) -> dict[str, Any]:
    payload = transient_storage_contract_v1_payload()
    write_json_atomic(path, payload, advisory=False)
    return payload


def publish_resolved_performance_contract_v2(path: Path) -> dict[str, Any]:
    payload = resolved_performance_contract_v2_payload()
    write_json_atomic(path, payload, advisory=False)
    return payload


def publish_resolved_performance_contract_v3(path: Path) -> dict[str, Any]:
    payload = resolved_performance_contract_v3_payload()
    write_json_atomic(path, payload, advisory=False)
    return payload


def publish_policy(path: Path) -> dict[str, Any]:
    payload = policy_payload()
    write_json_atomic(path, payload, advisory=False)
    return payload


def _normalise_timestamp(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC") if pd.Timestamp(value).tzinfo else pd.Timestamp(value, tz="UTC")


def validate_prediction_frame(
    predictions: pd.DataFrame,
    *,
    expected_timestamp: pd.Timestamp | None = None,
    expected_assets: Sequence[str] | None = None,
) -> dict[str, Any]:
    required = {"family", "decision_timestamp", "asset_id", "prediction"}
    missing_columns = sorted(required - set(predictions.columns))
    if missing_columns:
        return {"valid": False, "reason": "missing_prediction_columns", "missing_columns": missing_columns}
    frame = predictions.copy()
    frame["asset_id"] = frame["asset_id"].astype(str)
    frame["decision_timestamp"] = pd.to_datetime(frame["decision_timestamp"], utc=True)
    if expected_timestamp is not None:
        expected = _normalise_timestamp(expected_timestamp)
        if set(frame["decision_timestamp"]) != {expected}:
            return {"valid": False, "reason": "unexpected_prediction_timestamp", "expected": expected.isoformat()}
    duplicate_count = int(frame.duplicated(["family", "decision_timestamp", "asset_id"]).sum())
    missing_count = 0
    extra_count = 0
    if expected_assets is not None:
        expected_set = {str(asset) for asset in expected_assets}
        observed_set = set(frame["asset_id"])
        missing_count = len(expected_set - observed_set)
        extra_count = len(observed_set - expected_set)
    finite = bool(np.isfinite(pd.to_numeric(frame["prediction"], errors="coerce")).all())
    return {
        "valid": duplicate_count == 0 and missing_count == 0 and extra_count == 0 and finite,
        "row_count": int(len(frame)),
        "duplicate_prediction_count": duplicate_count,
        "missing_prediction_count": missing_count,
        "extra_prediction_count": extra_count,
        "finite_predictions": finite,
    }


def score_distribution(scores: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(scores, errors="coerce").dropna()
    if numeric.empty:
        return {
            "score_count": 0,
            "score_mean": math.nan,
            "score_std": math.nan,
            "score_min": math.nan,
            "score_max": math.nan,
            "score_tie_count": 0,
        }
    quantiles = numeric.quantile(metrics_registry()["score_quantiles"])
    result = {
        "score_count": int(numeric.size),
        "score_mean": float(numeric.mean()),
        "score_std": float(numeric.std(ddof=0)),
        "score_min": float(numeric.min()),
        "score_max": float(numeric.max()),
        "score_tie_count": int(numeric.size - numeric.nunique(dropna=True)),
    }
    for q, value in quantiles.items():
        result[f"score_q{int(round(float(q) * 100)):02d}"] = float(value)
    return result


def _corr(left: pd.Series, right: pd.Series, method: str) -> float:
    valid = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 2 or valid["left"].nunique() < 2 or valid["right"].nunique() < 2:
        return math.nan
    return float(valid["left"].corr(valid["right"], method=method))


def _ndcg(scores: pd.Series, targets: pd.Series, top_n: int) -> float:
    frame = pd.DataFrame({"score": scores, "target": targets}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return math.nan
    take = min(top_n, len(frame))
    ranked = frame.sort_values("score", ascending=False).head(take)
    ideal = frame.sort_values("target", ascending=False).head(take)
    discounts = 1.0 / np.log2(np.arange(2, take + 2))
    dcg = float(np.sum(ranked["target"].to_numpy(dtype=float) * discounts))
    ideal_dcg = float(np.sum(ideal["target"].to_numpy(dtype=float) * discounts))
    if ideal_dcg == 0:
        return math.nan
    return dcg / ideal_dcg


def compute_per_t_metrics(
    predictions: pd.DataFrame,
    targets: pd.DataFrame | None = None,
    *,
    top_n: int = 20,
    expected_assets: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame = predictions.copy()
    frame["decision_timestamp"] = pd.to_datetime(frame["decision_timestamp"], utc=True)
    frame["asset_id"] = frame["asset_id"].astype(str)
    if targets is not None and not targets.empty:
        target_frame = targets.copy()
        target_frame["decision_timestamp"] = pd.to_datetime(target_frame["decision_timestamp"], utc=True)
        target_frame["asset_id"] = target_frame["asset_id"].astype(str)
        target_cols = ["asset_id", "decision_timestamp", "target_value"]
        if "target_available_timestamp" in target_frame.columns:
            target_cols.append("target_available_timestamp")
        frame = frame.merge(target_frame[target_cols], on=["asset_id", "decision_timestamp"], how="left")
    metric_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for (family, timestamp), group in frame.groupby(["family", "decision_timestamp"], sort=True):
        validation = validate_prediction_frame(group, expected_timestamp=timestamp, expected_assets=expected_assets)
        dist = score_distribution(group["prediction"])
        metric: dict[str, Any] = {
            "policy_id": POLICY_ID,
            "policy_hash": policy_hash(),
            "family": family,
            "decision_timestamp": pd.Timestamp(timestamp).isoformat(),
            "prediction_rows_generated": int(len(group)),
            "eligible_assets": int(len(expected_assets) if expected_assets is not None else group["asset_id"].nunique()),
            "prediction_coverage": float(len(group) / len(expected_assets)) if expected_assets else 1.0,
            "missing_prediction_count": validation["missing_prediction_count"],
            "duplicate_prediction_count": validation["duplicate_prediction_count"],
            **dist,
        }
        if "target_value" in group.columns:
            mature = group.dropna(subset=["target_value"]).copy()
            metric["mature_target_rows"] = int(len(mature))
            metric["spearman_rank_ic"] = _corr(mature["prediction"], mature["target_value"], "spearman")
            metric["pearson_ic"] = _corr(mature["prediction"], mature["target_value"], "pearson")
            error = mature["prediction"].astype(float) - mature["target_value"].astype(float)
            metric["mae"] = float(error.abs().mean()) if len(error) else math.nan
            metric["mse"] = float((error**2).mean()) if len(error) else math.nan
            metric["rmse"] = float(math.sqrt(metric["mse"])) if not math.isnan(metric["mse"]) else math.nan
            metric["directional_accuracy"] = float((np.sign(mature["prediction"]) == np.sign(mature["target_value"])).mean()) if len(mature) else math.nan
            metric["ndcg_at_top_n"] = _ndcg(mature["prediction"], mature["target_value"], top_n)
            top = mature.sort_values("prediction", ascending=False).head(top_n)
            metric["gross_return"] = float(top["target_value"].mean()) if len(top) else math.nan
            metric["net_return"] = metric["gross_return"]
            metric["hit_rate"] = float((top["target_value"] > 0).mean()) if len(top) else math.nan
        else:
            metric["mature_target_rows"] = 0
        selected = group.sort_values(["prediction", "asset_id"], ascending=[False, True]).head(top_n)
        weight = 1.0 / len(selected) if len(selected) else 0.0
        for rank, row in enumerate(selected.itertuples(index=False), start=1):
            decision_rows.append(
                {
                    "policy_id": POLICY_ID,
                    "policy_hash": policy_hash(),
                    "family": family,
                    "decision_timestamp": pd.Timestamp(timestamp).isoformat(),
                    "rank": rank,
                    "asset_id": str(getattr(row, "asset_id")),
                    "weight": weight,
                    "score": float(getattr(row, "prediction")),
                    "top_n": top_n,
                }
            )
        metric_rows.append(metric)
    return pd.DataFrame(metric_rows), pd.DataFrame(decision_rows)


def _metadata_required(metadata: Mapping[str, Any] | None, key: str) -> str:
    value = "" if metadata is None else str(metadata.get(key, "") or "")
    if not value:
        raise ValueError(f"RESOLVED_PERFORMANCE_V2_MISSING_{key.upper()}")
    return value


def _normalise_timestamp_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _iso_or_empty(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return _normalise_timestamp(value).isoformat()


def _finite_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def newey_west_mean_ci(values: Sequence[float], *, lag: int | None = None, z: float = 1.96) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    n = int(len(arr))
    if n == 0:
        return {"mean": None, "lower": None, "upper": None, "standard_error": None, "observations": 0, "lag": 0}
    mean = float(arr.mean())
    if n == 1:
        return {"mean": mean, "lower": mean, "upper": mean, "standard_error": 0.0, "observations": 1, "lag": 0}
    demeaned = arr - mean
    if lag is None:
        lag = max(1, int(round(4 * (n / 100.0) ** (2 / 9))))
    lag = min(max(int(lag), 0), n - 1)
    gamma0 = float(np.dot(demeaned, demeaned) / n)
    var = gamma0
    for step in range(1, lag + 1):
        gamma = float(np.dot(demeaned[step:], demeaned[:-step]) / n)
        var += 2.0 * (1.0 - step / (lag + 1.0)) * gamma
    se = math.sqrt(max(var / n, 0.0))
    return {"mean": mean, "lower": mean - z * se, "upper": mean + z * se, "standard_error": se, "observations": n, "lag": lag}


def maximum_drawdown(returns: pd.Series) -> float | None:
    clean = _finite_numeric(returns).dropna()
    if clean.empty:
        return None
    equity = (1.0 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def resolved_v2_expected_columns() -> list[str]:
    return [
        "family",
        "decision_timestamp",
        "asset_id",
        "score",
        "score_rank",
        "eligible",
        "model_vintage_id",
        "model_hash",
        "preprocessing_hash",
        "policy_hash",
        "training_cutoff",
        "target_id",
        "target_horizon_minutes",
        "prediction_timestamp",
        "expected_target_available_timestamp",
        "evaluation_contract_id",
        "evaluation_contract_hash",
    ]


def resolved_v3_expected_columns() -> list[str]:
    return resolved_v2_expected_columns()


def build_pending_score_frame(
    predictions: pd.DataFrame,
    *,
    family: str,
    metadata: Mapping[str, Any] | None = None,
    target_horizon_minutes: int = DEFAULT_TARGET_HORIZON_MINUTES,
) -> pd.DataFrame:
    required = {"decision_timestamp", "asset_id", "prediction"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"RESOLVED_PERFORMANCE_V2_MISSING_PREDICTION_COLUMNS:{missing}")
    model_hash = _metadata_required(metadata, "model_hash")
    model_vintage_id = _metadata_required(metadata, "model_vintage_id")
    preprocessing_hash = _metadata_required(metadata, "preprocessing_hash")
    policy_hash_value = str(metadata.get("policy_hash", "") if metadata else "") or str(predictions.get("policy_hash", pd.Series([""])).iloc[0] or "")
    if not policy_hash_value:
        raise ValueError("RESOLVED_PERFORMANCE_V2_MISSING_POLICY_HASH")
    training_cutoff = _metadata_required(metadata, "training_cutoff")
    prediction_timestamp = str((metadata or {}).get("prediction_timestamp") or pd.Timestamp.now("UTC").isoformat())
    frame = predictions.copy()
    frame["family"] = frame.get("family", family)
    frame["family"] = frame["family"].fillna(family).astype(str)
    frame["decision_timestamp"] = _normalise_timestamp_series(frame["decision_timestamp"])
    frame["asset_id"] = frame["asset_id"].astype(str)
    frame["score"] = _finite_numeric(frame["prediction"])
    if frame["decision_timestamp"].isna().any() or frame["score"].isna().any():
        raise ValueError("RESOLVED_PERFORMANCE_V2_NONFINITE_OR_INVALID_PREDICTION")
    duplicate_count = int(frame.duplicated(["family", "decision_timestamp", "asset_id"]).sum())
    if duplicate_count:
        raise ValueError(f"RESOLVED_PERFORMANCE_V2_DUPLICATE_PENDING_SCORE_KEYS:{duplicate_count}")
    frame = frame.sort_values(["family", "decision_timestamp", "score", "asset_id"], ascending=[True, True, False, True]).copy()
    frame["score_rank"] = frame.groupby(["family", "decision_timestamp"], sort=False).cumcount() + 1
    if "expected_target_available_timestamp" in frame.columns:
        expected = _normalise_timestamp_series(frame["expected_target_available_timestamp"])
    else:
        expected = frame["decision_timestamp"] + pd.Timedelta(minutes=int(target_horizon_minutes))
    out = pd.DataFrame(
        {
            "family": frame["family"],
            "decision_timestamp": frame["decision_timestamp"].map(lambda ts: ts.isoformat()),
            "asset_id": frame["asset_id"],
            "score": frame["score"].astype(float),
            "score_rank": frame["score_rank"].astype(int),
            "eligible": frame.get("eligible", True),
            "model_vintage_id": model_vintage_id,
            "model_hash": model_hash,
            "preprocessing_hash": preprocessing_hash,
            "policy_hash": policy_hash_value,
            "training_cutoff": training_cutoff,
            "target_id": TARGET_ID,
            "target_horizon_minutes": int(target_horizon_minutes),
            "prediction_timestamp": prediction_timestamp,
            "expected_target_available_timestamp": expected.map(lambda ts: ts.isoformat()),
            "evaluation_contract_id": RESOLVED_PERFORMANCE_CONTRACT_V2_ID,
            "evaluation_contract_hash": resolved_performance_contract_v2_hash(),
        }
    )
    return out[resolved_v2_expected_columns()]


def build_pending_score_frame_v3(
    predictions: pd.DataFrame,
    *,
    family: str,
    metadata: Mapping[str, Any] | None = None,
    target_horizon_minutes: int = DEFAULT_TARGET_HORIZON_MINUTES,
) -> pd.DataFrame:
    out = build_pending_score_frame(
        predictions,
        family=family,
        metadata=metadata,
        target_horizon_minutes=target_horizon_minutes,
    ).copy()
    out["evaluation_contract_id"] = RESOLVED_PERFORMANCE_CONTRACT_V3_ID
    out["evaluation_contract_hash"] = resolved_performance_contract_v3_hash()
    return out[resolved_v3_expected_columns()]


def deduplicate_pending_scores(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = frame.copy()
    key_cols = ["family", "decision_timestamp", "asset_id"]
    duplicated = work.duplicated(key_cols, keep=False)
    if not duplicated.any():
        return work
    value_cols = ["score", "model_hash", "preprocessing_hash", "policy_hash", "training_cutoff", "target_id"]
    for _, group in work[duplicated].groupby(key_cols, sort=False):
        comparable = group[value_cols].astype(str).drop_duplicates()
        if len(comparable) > 1:
            raise ValueError("RESOLVED_PERFORMANCE_V2_CONFLICTING_DUPLICATE_PENDING_SCORE_KEYS")
    return work.drop_duplicates(key_cols, keep="last").copy()


def _target_identity_checked(targets: pd.DataFrame) -> pd.DataFrame:
    if targets.empty:
        return pd.DataFrame(columns=["asset_id", "decision_timestamp", "target_available_timestamp", "target_is_trainable", "target_value"])
    target = targets.copy()
    if "target_id" in target.columns and int((target["target_id"].astype(str) != TARGET_ID).sum()):
        raise ValueError("RESOLVED_PERFORMANCE_V2_TARGET_ID_MISMATCH")
    required = {"asset_id", "decision_timestamp", "target_available_timestamp", "target_is_trainable", "target_value"}
    missing = sorted(required - set(target.columns))
    if missing:
        raise ValueError(f"RESOLVED_PERFORMANCE_V2_MISSING_TARGET_COLUMNS:{missing}")
    target["asset_id"] = target["asset_id"].astype(str)
    target["decision_timestamp"] = _normalise_timestamp_series(target["decision_timestamp"])
    target["target_available_timestamp"] = _normalise_timestamp_series(target["target_available_timestamp"])
    duplicate_count = int(target.duplicated(["asset_id", "decision_timestamp"]).sum())
    if duplicate_count:
        raise ValueError(f"RESOLVED_PERFORMANCE_V2_DUPLICATE_TARGET_KEYS:{duplicate_count}")
    return target


def _json_hashable_assets(assets: Sequence[str]) -> str:
    return json.dumps([str(asset) for asset in assets], separators=(",", ":"))


def _weights_from_asset_json(value: Any) -> dict[str, float]:
    if not value:
        return {}
    try:
        assets = [str(asset) for asset in json.loads(str(value))]
    except Exception:
        return {}
    weight = 1.0 / len(assets) if assets else 0.0
    return {asset: weight for asset in assets}


def _last_top_weights(existing_resolved: pd.DataFrame | None) -> dict[str, float]:
    if existing_resolved is None or existing_resolved.empty or "top_selected_assets_json" not in existing_resolved.columns:
        return {}
    work = existing_resolved.copy()
    work["decision_timestamp"] = _normalise_timestamp_series(work["decision_timestamp"])
    work = work.sort_values("decision_timestamp")
    return _weights_from_asset_json(work.iloc[-1].get("top_selected_assets_json"))


def _ic_values(resolved: pd.DataFrame) -> dict[str, Any]:
    clean = resolved[["score", "target_value"]].copy()
    clean["score"] = _finite_numeric(clean["score"])
    clean["target_value"] = _finite_numeric(clean["target_value"])
    clean = clean.dropna()
    obs = int(len(clean))
    score_dispersion = float(clean["score"].std(ddof=0)) if obs else None
    target_dispersion = float(clean["target_value"].std(ddof=0)) if obs else None
    if obs < 2:
        return {
            "spearman_rank_ic": None,
            "pearson_ic": None,
            "rank_ic_observation_count": obs,
            "rank_ic_null_reason": "INSUFFICIENT_RESOLVED_CROSS_SECTION",
            "score_dispersion": score_dispersion,
            "target_dispersion": target_dispersion,
        }
    if clean["score"].nunique() < 2:
        return {
            "spearman_rank_ic": None,
            "pearson_ic": None,
            "rank_ic_observation_count": obs,
            "rank_ic_null_reason": "ZERO_SCORE_DISPERSION",
            "score_dispersion": score_dispersion,
            "target_dispersion": target_dispersion,
        }
    if clean["target_value"].nunique() < 2:
        return {
            "spearman_rank_ic": None,
            "pearson_ic": None,
            "rank_ic_observation_count": obs,
            "rank_ic_null_reason": "ZERO_TARGET_DISPERSION",
            "score_dispersion": score_dispersion,
            "target_dispersion": target_dispersion,
        }
    return {
        "spearman_rank_ic": float(clean["score"].corr(clean["target_value"], method="spearman")),
        "pearson_ic": float(clean["score"].corr(clean["target_value"], method="pearson")),
        "rank_ic_observation_count": obs,
        "rank_ic_null_reason": "",
        "score_dispersion": score_dispersion,
        "target_dispersion": target_dispersion,
    }


def _turnover(new_weights: Mapping[str, float], previous_weights: Mapping[str, float]) -> float:
    if not previous_weights:
        return 0.0
    assets = set(new_weights) | set(previous_weights)
    return float(0.5 * sum(abs(float(new_weights.get(asset, 0.0)) - float(previous_weights.get(asset, 0.0))) for asset in assets))


def resolve_pending_score_frame(
    pending: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    resolution_timestamp: pd.Timestamp,
    top_n: int = 20,
    existing_resolved: pd.DataFrame | None = None,
    terminal_timestamp: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if pending.empty:
        return pd.DataFrame(), pending.copy(), pd.DataFrame(), {"resolved_rows": 0, "pending_rows": 0, "terminal_censored_rows": 0}
    work = deduplicate_pending_scores(pending)
    work["decision_timestamp"] = _normalise_timestamp_series(work["decision_timestamp"])
    work["expected_target_available_timestamp"] = _normalise_timestamp_series(work["expected_target_available_timestamp"])
    work["asset_id"] = work["asset_id"].astype(str)
    work["score"] = _finite_numeric(work["score"])
    if work["decision_timestamp"].isna().any() or work["expected_target_available_timestamp"].isna().any() or work["score"].isna().any():
        raise ValueError("RESOLVED_PERFORMANCE_V2_INVALID_PENDING_SCORE_RECORD")
    resolution = _normalise_timestamp(resolution_timestamp)
    terminal = _normalise_timestamp(terminal_timestamp) if terminal_timestamp else None
    target = _target_identity_checked(targets)
    merged = work.merge(target, on=["asset_id", "decision_timestamp"], how="left", validate="many_to_one")
    merged["target_value"] = _finite_numeric(merged.get("target_value", pd.Series(dtype=float)))
    if "target_available_timestamp" in merged:
        merged["target_available_timestamp"] = _normalise_timestamp_series(merged["target_available_timestamp"])
    if "target_is_trainable" not in merged:
        merged["target_is_trainable"] = False
    chronology = merged["target_available_timestamp"].notna() & (merged["target_available_timestamp"] <= merged["decision_timestamp"])
    if bool(chronology.any()):
        raise ValueError(f"RESOLVED_PERFORMANCE_V2_TARGET_CHRONOLOGY_VIOLATION:{int(chronology.sum())}")

    previous_top_weights = _last_top_weights(existing_resolved)
    resolved_rows: list[dict[str, Any]] = []
    terminal_rows: list[dict[str, Any]] = []
    remaining_parts: list[pd.DataFrame] = []
    for (family, decision_timestamp), group in merged.groupby(["family", "decision_timestamp"], sort=True):
        decision = pd.Timestamp(decision_timestamp).tz_convert("UTC")
        expected_available = group["expected_target_available_timestamp"].max()
        should_censor = bool(terminal is not None and expected_available > terminal)
        target_available_values = group["target_available_timestamp"].dropna()
        actual_available_after_resolution = bool(not target_available_values.empty and target_available_values.max() > resolution)
        should_resolve = bool((expected_available <= resolution and not actual_available_after_resolution) or should_censor)
        original_pending = work[(work["family"] == family) & (work["decision_timestamp"] == decision)]
        if not should_resolve:
            remaining_parts.append(original_pending)
            continue
        if should_censor:
            terminal_rows.append(
                {
                    "family": str(family),
                    "decision_timestamp": decision.isoformat(),
                    "resolution_timestamp": resolution.isoformat(),
                    "terminal_censored_count": int(len(group)),
                    "pending_score_hash": stable_hash(original_pending[resolved_v2_expected_columns()].to_dict(orient="split")),
                    "evaluation_contract_id": RESOLVED_PERFORMANCE_CONTRACT_V2_ID,
                    "evaluation_contract_hash": resolved_performance_contract_v2_hash(),
                }
            )
            continue
        trainable = group["target_is_trainable"].fillna(False).astype(bool)
        available = group["target_available_timestamp"].notna() & (group["target_available_timestamp"] <= resolution)
        finite_target = group["target_value"].notna()
        resolved = group[trainable & available & finite_target].copy()
        missing_target_count = int(len(group) - len(resolved))
        ic = _ic_values(resolved)
        ranked = group.sort_values(["score", "asset_id"], ascending=[False, True]).copy()
        top = ranked.head(top_n)
        bottom = ranked.tail(min(top_n, len(ranked))).copy()
        top_assets = [str(asset) for asset in top["asset_id"].tolist()]
        bottom_assets = [str(asset) for asset in bottom["asset_id"].tolist()]
        top_resolved = resolved[resolved["asset_id"].isin(top_assets)].copy()
        bottom_resolved = resolved[resolved["asset_id"].isin(bottom_assets)].copy()
        selected_weights = {asset: 1.0 / len(top_assets) for asset in top_assets} if top_assets else {}
        turnover = _turnover(selected_weights, previous_top_weights)
        cost = turnover * (TOP_N_COST_BPS_PER_UNIT_TURNOVER / 10000.0)
        top_complete = len(top_assets) > 0 and len(top_resolved) == len(top_assets)
        bottom_complete = len(bottom_assets) > 0 and len(bottom_resolved) == len(bottom_assets)
        top_gross = float(top_resolved["target_value"].mean()) if top_complete else None
        bottom_gross = float(bottom_resolved["target_value"].mean()) if bottom_complete else None
        spread = (top_gross - bottom_gross) if top_gross is not None and bottom_gross is not None else None
        net = (top_gross - cost) if top_gross is not None else None
        row = {
            "family": str(family),
            "decision_timestamp": decision.isoformat(),
            "resolution_timestamp": resolution.isoformat(),
            "target_id": TARGET_ID,
            "model_vintage_id": str(group["model_vintage_id"].iloc[0]),
            "model_hash": str(group["model_hash"].iloc[0]),
            "preprocessing_hash": str(group["preprocessing_hash"].iloc[0]),
            "policy_hash": str(group["policy_hash"].iloc[0]),
            "training_cutoff": str(group["training_cutoff"].iloc[0]),
            "eligible_asset_count": int(group["eligible"].fillna(True).astype(bool).sum()),
            "scored_asset_count": int(len(group)),
            "resolved_asset_count": int(len(resolved)),
            "missing_target_count": missing_target_count,
            "duplicate_count": 0,
            "chronology_violation_count": 0,
            "terminal_censored_count": 0,
            "evaluation_contract_id": RESOLVED_PERFORMANCE_CONTRACT_V2_ID,
            "evaluation_contract_hash": resolved_performance_contract_v2_hash(),
            **ic,
            "top_n": int(top_n),
            "top_n_gross_return": top_gross,
            "bottom_n_gross_return": bottom_gross,
            "long_short_spread_return": spread,
            "net_return": net,
            "turnover": turnover,
            "estimated_transaction_cost": cost,
            "selected_asset_count": int(len(top_assets)),
            "resolved_selected_asset_count": int(len(top_resolved)),
            "bottom_selected_asset_count": int(len(bottom_assets)),
            "resolved_bottom_selected_asset_count": int(len(bottom_resolved)),
            "coverage_fraction": float(len(resolved) / len(group)) if len(group) else 0.0,
            "portfolio_return_null_reason": "" if top_complete else "INCOMPLETE_TOP_N_TARGET_COVERAGE",
            "top_selected_assets_json": _json_hashable_assets(top_assets),
            "bottom_selected_assets_json": _json_hashable_assets(bottom_assets),
        }
        row["row_hash"] = stable_hash(row)
        resolved_rows.append(row)
        previous_top_weights = selected_weights
    remaining = pd.concat(remaining_parts, ignore_index=True) if remaining_parts else pd.DataFrame(columns=work.columns)
    resolved_frame = pd.DataFrame(resolved_rows)
    terminal_frame = pd.DataFrame(terminal_rows)
    return (
        resolved_frame,
        remaining[resolved_v2_expected_columns()] if not remaining.empty else remaining,
        terminal_frame,
        {
            "resolved_rows": int(len(resolved_frame)),
            "pending_rows": int(len(remaining)),
            "terminal_censored_rows": int(len(terminal_frame)),
            "resolution_timestamp": resolution.isoformat(),
        },
    )


def resolved_v2_summary(resolved: pd.DataFrame) -> dict[str, Any]:
    if resolved.empty:
        return {
            "status": "NO_RESOLVED_PERFORMANCE_ROWS",
            "resolved_performance_rows": 0,
            "rank_ic": {"valid_rows": 0, "mean_spearman_rank_ic": None},
            "returns": {"resolved_portfolio_observations": 0, "cumulative_net_return": None},
        }
    work = resolved.copy()
    work["decision_timestamp"] = _normalise_timestamp_series(work["decision_timestamp"])
    rank_ic = _finite_numeric(work.get("spearman_rank_ic", pd.Series(dtype=float))).dropna()
    net = _finite_numeric(work.get("net_return", pd.Series(dtype=float))).dropna()
    gross = _finite_numeric(work.get("top_n_gross_return", pd.Series(dtype=float))).dropna()
    downside = net[net < 0]
    annualized_return = float(net.mean() * TRADING_SESSIONS_PER_YEAR) if len(net) else None
    annualized_volatility = float(net.std(ddof=1) * math.sqrt(TRADING_SESSIONS_PER_YEAR)) if len(net) > 1 else None
    downside_volatility = float(downside.std(ddof=1) * math.sqrt(TRADING_SESSIONS_PER_YEAR)) if len(downside) > 1 else None
    ic_std = float(rank_ic.std(ddof=1)) if len(rank_ic) > 1 else None
    return {
        "status": "PROVISIONAL" if len(work) else "NO_RESOLVED_PERFORMANCE_ROWS",
        "evaluation_contract_id": RESOLVED_PERFORMANCE_CONTRACT_V2_ID,
        "evaluation_contract_hash": resolved_performance_contract_v2_hash(),
        "first_resolved_decision_timestamp": work["decision_timestamp"].min().isoformat(),
        "last_resolved_decision_timestamp": work["decision_timestamp"].max().isoformat(),
        "resolved_performance_rows": int(len(work)),
        "rank_ic": {
            "mean_spearman_rank_ic": float(rank_ic.mean()) if len(rank_ic) else None,
            "median_spearman_rank_ic": float(rank_ic.median()) if len(rank_ic) else None,
            "rank_ic_std": ic_std,
            "rank_ic_positive_fraction": float((rank_ic > 0).mean()) if len(rank_ic) else None,
            "rank_ic_information_ratio": float(rank_ic.mean() / ic_std) if ic_std and ic_std > 0 else None,
            "newey_west_95_ci": newey_west_mean_ci(rank_ic.tolist()) if len(rank_ic) else None,
            "valid_rows": int(len(rank_ic)),
        },
        "returns": {
            "cumulative_gross_return": float((1.0 + gross).prod() - 1.0) if len(gross) else None,
            "cumulative_net_return": float((1.0 + net).prod() - 1.0) if len(net) else None,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe": annualized_return / annualized_volatility if annualized_volatility and annualized_volatility > 0 else None,
            "sortino": annualized_return / downside_volatility if downside_volatility and downside_volatility > 0 else None,
            "maximum_drawdown": maximum_drawdown(net),
            "hit_rate": float((net > 0).mean()) if len(net) else None,
            "turnover": float(_finite_numeric(work.get("turnover", pd.Series(dtype=float))).mean()) if "turnover" in work else None,
            "total_estimated_costs": float(_finite_numeric(work.get("estimated_transaction_cost", pd.Series(dtype=float))).sum()) if "estimated_transaction_cost" in work else 0.0,
            "resolved_portfolio_observations": int(len(net)),
        },
        "annualisation": {
            "method": "per-decision provisional series; terminal reports may aggregate by session for overlapping 60m outcomes",
            "trading_sessions_per_year": TRADING_SESSIONS_PER_YEAR,
        },
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
    }


def _rank_ic_sufficient_statistics(
    resolved: pd.DataFrame,
    *,
    min_cross_section: int = DEFAULT_V3_MIN_RANK_IC_CROSS_SECTION,
) -> dict[str, Any]:
    clean = resolved[["asset_id", "score", "target_value"]].copy() if not resolved.empty else pd.DataFrame(columns=["asset_id", "score", "target_value"])
    clean["asset_id"] = clean["asset_id"].astype(str)
    clean["score"] = _finite_numeric(clean["score"])
    clean["target_value"] = _finite_numeric(clean["target_value"])
    clean = clean.dropna().sort_values("asset_id").copy()
    obs = int(len(clean))
    if obs:
        clean["score_rank_average"] = clean["score"].rank(method="average", ascending=True)
        clean["target_rank_average"] = clean["target_value"].rank(method="average", ascending=True)
    score_ties = int(obs - clean["score"].nunique(dropna=True)) if obs else 0
    target_ties = int(obs - clean["target_value"].nunique(dropna=True)) if obs else 0
    null_reason = ""
    if obs < int(min_cross_section):
        null_reason = "INSUFFICIENT_RESOLVED_CROSS_SECTION"
    elif clean["score"].nunique(dropna=True) < 2:
        null_reason = "ZERO_SCORE_DISPERSION"
    elif clean["target_value"].nunique(dropna=True) < 2:
        null_reason = "ZERO_TARGET_DISPERSION"
    sum_score = float(clean["score_rank_average"].sum()) if obs else 0.0
    sum_target = float(clean["target_rank_average"].sum()) if obs else 0.0
    sum_score_sq = float((clean["score_rank_average"] ** 2).sum()) if obs else 0.0
    sum_target_sq = float((clean["target_rank_average"] ** 2).sum()) if obs else 0.0
    sum_cross = float((clean["score_rank_average"] * clean["target_rank_average"]).sum()) if obs else 0.0
    numerator = float(obs * sum_cross - sum_score * sum_target) if obs else 0.0
    denom_left = float(obs * sum_score_sq - sum_score**2) if obs else 0.0
    denom_right = float(obs * sum_target_sq - sum_target**2) if obs else 0.0
    denominator = math.sqrt(max(denom_left, 0.0) * max(denom_right, 0.0)) if obs else 0.0
    spearman = None if null_reason or denominator <= 0 else float(numerator / denominator)
    pearson = _corr(clean["score"], clean["target_value"], "pearson") if not null_reason else math.nan
    stats = {
        "rank_ic_observation_count": obs,
        "minimum_rank_ic_cross_section": int(min_cross_section),
        "rank_ic_null_reason": null_reason,
        "tie_method": "average_rank",
        "score_tie_count": score_ties,
        "target_tie_count": target_ties,
        "sum_score_rank": sum_score,
        "sum_target_rank": sum_target,
        "sum_score_rank_squared": sum_score_sq,
        "sum_target_rank_squared": sum_target_sq,
        "sum_score_target_rank_product": sum_cross,
        "rank_ic_numerator": numerator,
        "rank_ic_denominator": denominator,
        "spearman_rank_ic": spearman,
        "pearson_ic": None if pd.isna(pearson) else float(pearson),
    }
    stats["sufficient_statistics_hash"] = stable_hash(stats)
    return stats


def _bounded_rank_audit_sample(
    resolved: pd.DataFrame,
    *,
    decision_timestamp: pd.Timestamp,
    family: str,
    max_rows: int = DEFAULT_V3_AUDIT_SAMPLE_ROWS_PER_TIMESTAMP,
) -> pd.DataFrame:
    if resolved.empty or max_rows <= 0:
        return pd.DataFrame()
    work = resolved[["asset_id", "score", "target_value"]].copy()
    work["asset_id"] = work["asset_id"].astype(str)
    work["score"] = _finite_numeric(work["score"])
    work["target_value"] = _finite_numeric(work["target_value"])
    work = work.dropna().sort_values(["score", "asset_id"], ascending=[False, True]).copy()
    if work.empty:
        return pd.DataFrame()
    work["score_rank_average"] = work["score"].rank(method="average", ascending=True)
    work["target_rank_average"] = work["target_value"].rank(method="average", ascending=True)
    if len(work) > max_rows:
        indexes = np.linspace(0, len(work) - 1, max_rows).round().astype(int)
        work = work.iloc[sorted(set(int(idx) for idx in indexes))].copy()
    work.insert(0, "family", str(family))
    work.insert(1, "decision_timestamp", decision_timestamp.isoformat())
    work["audit_sample_ordinal"] = np.arange(1, len(work) + 1)
    work["evaluation_contract_id"] = RESOLVED_PERFORMANCE_CONTRACT_V3_ID
    work["evaluation_contract_hash"] = resolved_performance_contract_v3_hash()
    work["pair_hash"] = work.apply(lambda row: stable_hash(row.to_dict()), axis=1)
    return work


def _sleeve_id_for_timestamp(timestamp: pd.Timestamp, *, sleeve_count: int = DEFAULT_V3_SLEEVE_COUNT) -> int:
    ts = pd.Timestamp(timestamp).tz_convert("UTC")
    return int(((ts.hour * 60 + ts.minute) // DEFAULT_DECISION_CADENCE_MINUTES) % int(sleeve_count))


def _latest_sleeve_weights(existing_sleeves: pd.DataFrame | None, *, sleeve_count: int = DEFAULT_V3_SLEEVE_COUNT) -> dict[int, dict[str, float]]:
    weights: dict[int, dict[str, float]] = {idx: {} for idx in range(int(sleeve_count))}
    if existing_sleeves is None or existing_sleeves.empty:
        return weights
    work = existing_sleeves.copy()
    if "decision_timestamp" not in work or "sleeve_id" not in work:
        return weights
    work["decision_timestamp"] = _normalise_timestamp_series(work["decision_timestamp"])
    work = work.sort_values("decision_timestamp")
    for sleeve_id, group in work.groupby("sleeve_id", sort=False):
        try:
            idx = int(sleeve_id)
        except Exception:
            continue
        weights[idx] = _weights_from_asset_json(group.iloc[-1].get("top_selected_assets_json"))
    return weights


def _daily_rows_from_sleeves(
    sleeves: pd.DataFrame,
    *,
    existing_daily: pd.DataFrame | None = None,
    resolution_timestamp: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if sleeves.empty:
        return pd.DataFrame()
    work = sleeves.copy()
    work["maturity_timestamp"] = _normalise_timestamp_series(work["maturity_timestamp"])
    work["maturity_session_date"] = work["maturity_timestamp"].dt.date.astype(str)
    net = _finite_numeric(work.get("net_return_contribution", pd.Series(dtype=float)))
    work = work[net.notna()].copy()
    if work.empty:
        return pd.DataFrame()
    if resolution_timestamp is not None:
        resolution_date = pd.Timestamp(resolution_timestamp).tz_convert("UTC").date().isoformat()
        work = work[work["maturity_session_date"] < resolution_date].copy()
    if work.empty:
        return pd.DataFrame()
    existing_dates: dict[str, str] = {}
    if existing_daily is not None and not existing_daily.empty and "session_date" in existing_daily:
        existing_dates = dict(zip(existing_daily["session_date"].astype(str), existing_daily.get("daily_return_hash", pd.Series([""] * len(existing_daily))).astype(str)))
    rows: list[dict[str, Any]] = []
    for session_date, group in work.groupby("maturity_session_date", sort=True):
        row = {
            "session_date": str(session_date),
            "sleeve_count": int(group.get("sleeve_count", pd.Series([DEFAULT_V3_SLEEVE_COUNT])).max()),
            "matured_sleeve_decisions": int(len(group)),
            "gross_daily_return": float(_finite_numeric(group["gross_return_contribution"]).sum()),
            "net_daily_return": float(_finite_numeric(group["net_return_contribution"]).sum()),
            "transaction_cost_drag": float(_finite_numeric(group.get("transaction_cost_contribution", pd.Series(dtype=float))).sum()),
            "mean_turnover": float(_finite_numeric(group.get("turnover", pd.Series(dtype=float))).mean()),
            "max_simultaneous_capital_fraction": float(_finite_numeric(group.get("simultaneous_capital_fraction", pd.Series([1.0]))).max()),
            "win": bool(float(_finite_numeric(group["net_return_contribution"]).sum()) > 0),
            "evaluation_contract_id": RESOLVED_PERFORMANCE_CONTRACT_V3_ID,
            "evaluation_contract_hash": resolved_performance_contract_v3_hash(),
        }
        row["daily_return_hash"] = stable_hash(row)
        previous = existing_dates.get(str(session_date))
        if previous and previous != row["daily_return_hash"]:
            raise RuntimeError("RESOLVED_PERFORMANCE_V3_DAILY_RETURN_HASH_MISMATCH")
        if not previous:
            rows.append(row)
    return pd.DataFrame(rows)


def resolve_pending_score_frame_v3(
    pending: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    resolution_timestamp: pd.Timestamp,
    top_n: int = 20,
    existing_rank_ic: pd.DataFrame | None = None,
    existing_sleeves: pd.DataFrame | None = None,
    terminal_timestamp: pd.Timestamp | str | None = None,
    min_rank_ic_cross_section: int = DEFAULT_V3_MIN_RANK_IC_CROSS_SECTION,
    audit_sample_rows: int = DEFAULT_V3_AUDIT_SAMPLE_ROWS_PER_TIMESTAMP,
    sleeve_count: int = DEFAULT_V3_SLEEVE_COUNT,
    transaction_cost_bps_per_unit_turnover: float = TOP_N_COST_BPS_PER_UNIT_TURNOVER,
) -> dict[str, Any]:
    if pending.empty:
        return {
            "rank_ic": pd.DataFrame(),
            "decision_trace": pd.DataFrame(),
            "sleeves": pd.DataFrame(),
            "transaction_costs": pd.DataFrame(),
            "audit_sample": pd.DataFrame(),
            "remaining": pending.copy(),
            "terminal_censored": pd.DataFrame(),
            "meta": {"resolved_timestamps": 0, "pending_rows": 0, "terminal_censored_rows": 0},
        }
    work = deduplicate_pending_scores(pending)
    work["decision_timestamp"] = _normalise_timestamp_series(work["decision_timestamp"])
    work["expected_target_available_timestamp"] = _normalise_timestamp_series(work["expected_target_available_timestamp"])
    work["asset_id"] = work["asset_id"].astype(str)
    work["score"] = _finite_numeric(work["score"])
    if work["decision_timestamp"].isna().any() or work["expected_target_available_timestamp"].isna().any() or work["score"].isna().any():
        raise ValueError("RESOLVED_PERFORMANCE_V3_INVALID_PENDING_SCORE_RECORD")
    resolution = _normalise_timestamp(resolution_timestamp)
    terminal = _normalise_timestamp(terminal_timestamp) if terminal_timestamp else None
    target = _target_identity_checked(targets)
    merged = work.merge(target, on=["asset_id", "decision_timestamp"], how="left", validate="many_to_one")
    merged["target_value"] = _finite_numeric(merged.get("target_value", pd.Series(dtype=float)))
    if "target_available_timestamp" in merged:
        merged["target_available_timestamp"] = _normalise_timestamp_series(merged["target_available_timestamp"])
    if "target_is_trainable" not in merged:
        merged["target_is_trainable"] = False
    chronology = merged["target_available_timestamp"].notna() & (merged["target_available_timestamp"] <= merged["decision_timestamp"])
    if bool(chronology.any()):
        raise ValueError(f"RESOLVED_PERFORMANCE_V3_TARGET_CHRONOLOGY_VIOLATION:{int(chronology.sum())}")

    existing_hashes: dict[str, str] = {}
    if existing_rank_ic is not None and not existing_rank_ic.empty and "decision_timestamp" in existing_rank_ic:
        existing_hashes = dict(
            zip(
                _normalise_timestamp_series(existing_rank_ic["decision_timestamp"]).map(lambda ts: ts.isoformat()),
                existing_rank_ic.get("row_hash", pd.Series([""] * len(existing_rank_ic))).astype(str),
            )
        )
    sleeve_weights = _latest_sleeve_weights(existing_sleeves, sleeve_count=sleeve_count)
    rank_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    sleeve_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    audit_frames: list[pd.DataFrame] = []
    terminal_rows: list[dict[str, Any]] = []
    remaining_parts: list[pd.DataFrame] = []
    skipped_existing = 0
    for (family, decision_timestamp), group in merged.groupby(["family", "decision_timestamp"], sort=True):
        decision = pd.Timestamp(decision_timestamp).tz_convert("UTC")
        expected_available = group["expected_target_available_timestamp"].max()
        should_censor = bool(terminal is not None and expected_available > terminal)
        target_available_values = group["target_available_timestamp"].dropna()
        actual_available_after_resolution = bool(not target_available_values.empty and target_available_values.max() > resolution)
        should_resolve = bool((expected_available <= resolution and not actual_available_after_resolution) or should_censor)
        original_pending = work[(work["family"] == family) & (work["decision_timestamp"] == decision)]
        if not should_resolve:
            remaining_parts.append(original_pending)
            continue
        if should_censor:
            terminal_rows.append(
                {
                    "family": str(family),
                    "decision_timestamp": decision.isoformat(),
                    "resolution_timestamp": resolution.isoformat(),
                    "terminal_censored_count": int(len(group)),
                    "pending_score_hash": stable_hash(original_pending[resolved_v3_expected_columns()].to_dict(orient="split")),
                    "evaluation_contract_id": RESOLVED_PERFORMANCE_CONTRACT_V3_ID,
                    "evaluation_contract_hash": resolved_performance_contract_v3_hash(),
                }
            )
            continue
        trainable = group["target_is_trainable"].fillna(False).astype(bool)
        available = group["target_available_timestamp"].notna() & (group["target_available_timestamp"] <= resolution)
        eligible = group["eligible"].fillna(True).astype(bool)
        finite_target = group["target_value"].notna()
        resolved = group[eligible & trainable & available & finite_target].copy()
        ic = _rank_ic_sufficient_statistics(resolved, min_cross_section=min_rank_ic_cross_section)
        ranked = group.sort_values(["score", "asset_id"], ascending=[False, True]).copy()
        top = ranked.head(top_n).copy()
        bottom = ranked.tail(min(top_n, len(ranked))).copy()
        top_assets = [str(asset) for asset in top["asset_id"].tolist()]
        bottom_assets = [str(asset) for asset in bottom["asset_id"].tolist()]
        top_resolved = resolved[resolved["asset_id"].isin(top_assets)].copy()
        bottom_resolved = resolved[resolved["asset_id"].isin(bottom_assets)].copy()
        top_complete = len(top_assets) > 0 and len(top_resolved) == len(top_assets)
        bottom_complete = len(bottom_assets) > 0 and len(bottom_resolved) == len(bottom_assets)
        top_gross = float(top_resolved["target_value"].mean()) if top_complete else None
        bottom_gross = float(bottom_resolved["target_value"].mean()) if bottom_complete else None
        sleeve_id = _sleeve_id_for_timestamp(decision, sleeve_count=sleeve_count)
        selected_weights = {asset: 1.0 / len(top_assets) for asset in top_assets} if top_assets else {}
        turnover = _turnover(selected_weights, sleeve_weights.get(sleeve_id, {}))
        sleeve_weights[sleeve_id] = selected_weights
        sleeve_capital = 1.0 / int(sleeve_count)
        transaction_cost = turnover * (float(transaction_cost_bps_per_unit_turnover) / 10000.0) * sleeve_capital
        gross_contribution = top_gross * sleeve_capital if top_gross is not None else None
        net_contribution = gross_contribution - transaction_cost if gross_contribution is not None else None
        base = {
            "family": str(family),
            "decision_timestamp": decision.isoformat(),
            "session_date": decision.date().isoformat(),
            "resolution_timestamp": resolution.isoformat(),
            "target_id": TARGET_ID,
            "model_vintage_id": str(group["model_vintage_id"].iloc[0]),
            "model_hash": str(group["model_hash"].iloc[0]),
            "preprocessing_hash": str(group["preprocessing_hash"].iloc[0]),
            "policy_hash": str(group["policy_hash"].iloc[0]),
            "training_cutoff": str(group["training_cutoff"].iloc[0]),
            "eligible_asset_count": int(eligible.sum()),
            "scored_asset_count": int(len(group)),
            "resolved_asset_count": int(len(resolved)),
            "excluded_asset_count": int(len(group) - len(resolved)),
            "duplicate_count": 0,
            "chronology_violation_count": 0,
            "evaluation_contract_id": RESOLVED_PERFORMANCE_CONTRACT_V3_ID,
            "evaluation_contract_hash": resolved_performance_contract_v3_hash(),
        }
        rank_row = {**base, **ic}
        rank_row["row_hash"] = stable_hash(rank_row)
        existing_hash = existing_hashes.get(decision.isoformat())
        if existing_hash:
            if existing_hash != rank_row["row_hash"]:
                raise RuntimeError("RESOLVED_PERFORMANCE_V3_DUPLICATE_TIMESTAMP_HASH_MISMATCH")
            skipped_existing += 1
            continue
        rank_rows.append(rank_row)
        for side, selected in (("TOP", top), ("BOTTOM", bottom)):
            weight = (1.0 / len(selected)) if len(selected) else 0.0
            for rank, row in enumerate(selected.itertuples(index=False), start=1):
                trace = {
                    "family": str(family),
                    "decision_timestamp": decision.isoformat(),
                    "side": side,
                    "rank": int(rank),
                    "asset_id": str(getattr(row, "asset_id")),
                    "score": float(getattr(row, "score")),
                    "weight": float(weight),
                    "top_n": int(top_n),
                    "evaluation_contract_id": RESOLVED_PERFORMANCE_CONTRACT_V3_ID,
                    "evaluation_contract_hash": resolved_performance_contract_v3_hash(),
                }
                trace["row_hash"] = stable_hash(trace)
                trace_rows.append(trace)
        maturity = decision + pd.Timedelta(minutes=DEFAULT_TARGET_HORIZON_MINUTES)
        sleeve_row = {
            **base,
            "sleeve_id": int(sleeve_id),
            "sleeve_count": int(sleeve_count),
            "sleeve_capital_fraction": float(sleeve_capital),
            "simultaneous_capital_fraction": 1.0,
            "maturity_timestamp": maturity.isoformat(),
            "maturity_session_date": maturity.date().isoformat(),
            "top_n": int(top_n),
            "top_n_gross_return": top_gross,
            "bottom_n_gross_return": bottom_gross,
            "long_short_spread_return": (top_gross - bottom_gross) if top_gross is not None and bottom_gross is not None else None,
            "gross_return_contribution": gross_contribution,
            "net_return_contribution": net_contribution,
            "turnover": float(turnover),
            "transaction_cost_contribution": float(transaction_cost),
            "portfolio_return_null_reason": "" if top_complete else "INCOMPLETE_TOP_N_TARGET_COVERAGE",
            "top_selected_assets_json": _json_hashable_assets(top_assets),
            "bottom_selected_assets_json": _json_hashable_assets(bottom_assets),
            "resolved_selected_asset_count": int(len(top_resolved)),
            "resolved_bottom_selected_asset_count": int(len(bottom_resolved)),
        }
        sleeve_row["row_hash"] = stable_hash(sleeve_row)
        sleeve_rows.append(sleeve_row)
        cost_row = {
            "family": str(family),
            "decision_timestamp": decision.isoformat(),
            "sleeve_id": int(sleeve_id),
            "turnover": float(turnover),
            "transaction_cost_bps_per_unit_turnover": float(transaction_cost_bps_per_unit_turnover),
            "transaction_cost_contribution": float(transaction_cost),
            "evaluation_contract_id": RESOLVED_PERFORMANCE_CONTRACT_V3_ID,
            "evaluation_contract_hash": resolved_performance_contract_v3_hash(),
        }
        cost_row["row_hash"] = stable_hash(cost_row)
        cost_rows.append(cost_row)
        sample = _bounded_rank_audit_sample(
            resolved,
            decision_timestamp=decision,
            family=str(family),
            max_rows=audit_sample_rows,
        )
        if not sample.empty:
            audit_frames.append(sample)
    remaining = pd.concat(remaining_parts, ignore_index=True) if remaining_parts else pd.DataFrame(columns=work.columns)
    return {
        "rank_ic": pd.DataFrame(rank_rows),
        "decision_trace": pd.DataFrame(trace_rows),
        "sleeves": pd.DataFrame(sleeve_rows),
        "transaction_costs": pd.DataFrame(cost_rows),
        "audit_sample": pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame(),
        "remaining": remaining[resolved_v3_expected_columns()] if not remaining.empty else remaining,
        "terminal_censored": pd.DataFrame(terminal_rows),
        "meta": {
            "resolved_timestamps": int(len(rank_rows)),
            "skipped_existing_timestamps": int(skipped_existing),
            "pending_rows": int(len(remaining)),
            "terminal_censored_rows": int(len(terminal_rows)),
            "resolution_timestamp": resolution.isoformat(),
        },
    }


def resolved_v3_summary(
    rank_ic: pd.DataFrame,
    daily_returns: pd.DataFrame,
    sleeves: pd.DataFrame,
    *,
    pending_rows: int = 0,
    terminal_censored_rows: int = 0,
) -> dict[str, Any]:
    ic_values = _finite_numeric(rank_ic.get("spearman_rank_ic", pd.Series(dtype=float))).dropna() if not rank_ic.empty else pd.Series(dtype=float)
    daily_net = _finite_numeric(daily_returns.get("net_daily_return", pd.Series(dtype=float))).dropna() if not daily_returns.empty else pd.Series(dtype=float)
    daily_gross = _finite_numeric(daily_returns.get("gross_daily_return", pd.Series(dtype=float))).dropna() if not daily_returns.empty else pd.Series(dtype=float)
    daily_vol = float(daily_net.std(ddof=1)) if len(daily_net) > 1 else None
    annualized_vol = daily_vol * math.sqrt(TRADING_SESSIONS_PER_YEAR) if daily_vol is not None else None
    annualized_return = float(daily_net.mean() * TRADING_SESSIONS_PER_YEAR) if len(daily_net) else None
    daily_ic_rows = 0
    daily_ic_mean = None
    if not rank_ic.empty and "session_date" in rank_ic:
        daily_ic = rank_ic.copy()
        daily_ic["spearman_rank_ic"] = _finite_numeric(daily_ic["spearman_rank_ic"])
        daily_ic = daily_ic.dropna(subset=["spearman_rank_ic"]).groupby("session_date")["spearman_rank_ic"].mean()
        daily_ic_rows = int(len(daily_ic))
        daily_ic_mean = float(daily_ic.mean()) if len(daily_ic) else None
    turnover = _finite_numeric(sleeves.get("turnover", pd.Series(dtype=float))).dropna() if not sleeves.empty else pd.Series(dtype=float)
    coverage = None
    if not rank_ic.empty and "eligible_asset_count" in rank_ic and "resolved_asset_count" in rank_ic:
        eligible_total = float(_finite_numeric(rank_ic["eligible_asset_count"]).sum())
        resolved_total = float(_finite_numeric(rank_ic["resolved_asset_count"]).sum())
        coverage = resolved_total / eligible_total if eligible_total > 0 else None
    return {
        "status": "PROVISIONAL" if len(rank_ic) or len(daily_returns) else "NO_RESOLVED_PERFORMANCE_ROWS",
        "evaluation_contract_id": RESOLVED_PERFORMANCE_CONTRACT_V3_ID,
        "evaluation_contract_hash": resolved_performance_contract_v3_hash(),
        "evaluation_contract_version": RESOLVED_PERFORMANCE_CONTRACT_V3_VERSION,
        "first_resolved_decision_timestamp": _normalise_timestamp_series(rank_ic["decision_timestamp"]).min().isoformat() if not rank_ic.empty else "",
        "last_resolved_decision_timestamp": _normalise_timestamp_series(rank_ic["decision_timestamp"]).max().isoformat() if not rank_ic.empty else "",
        "resolved_performance_rows": int(len(rank_ic)),
        "rank_ic": {
            "valid_timestamps": int(len(ic_values)),
            "mean_spearman_rank_ic": float(ic_values.mean()) if len(ic_values) else None,
            "median_spearman_rank_ic": float(ic_values.median()) if len(ic_values) else None,
            "positive_fraction": float((ic_values > 0).mean()) if len(ic_values) else None,
            "dependence_aware_95_ci": newey_west_mean_ci(ic_values.tolist(), lag=max(DEFAULT_V3_SLEEVE_COUNT, 1)) if len(ic_values) else None,
            "inference_method": "newey_west_hac_lag_12_for_overlapping_60m_targets",
            "daily_ic_rows": daily_ic_rows,
            "daily_mean_spearman_rank_ic": daily_ic_mean,
        },
        "returns": {
            "portfolio_contract": "twelve_staggered_equal_capital_sleeves",
            "sleeve_count": DEFAULT_V3_SLEEVE_COUNT,
            "simultaneous_capital_limit": 1.0,
            "daily_return_rows": int(len(daily_net)),
            "last_daily_net_return": float(daily_net.iloc[-1]) if len(daily_net) else None,
            "cumulative_gross_return": float((1.0 + daily_gross).prod() - 1.0) if len(daily_gross) else None,
            "cumulative_net_return": float((1.0 + daily_net).prod() - 1.0) if len(daily_net) else None,
            "annualized_return_from_daily_returns": annualized_return,
            "annualized_volatility_from_daily_returns": annualized_vol,
            "daily_sharpe": annualized_return / annualized_vol if annualized_return is not None and annualized_vol and annualized_vol > 0 else None,
            "maximum_drawdown": maximum_drawdown(daily_net),
            "win_rate": float((daily_net > 0).mean()) if len(daily_net) else None,
            "mean_turnover": float(turnover.mean()) if len(turnover) else None,
            "total_estimated_costs": float(_finite_numeric(sleeves.get("transaction_cost_contribution", pd.Series(dtype=float))).sum()) if not sleeves.empty else 0.0,
            "raw_overlapping_forward_returns_annualized": False,
        },
        "coverage": {
            "eligible_resolved_fraction": coverage,
            "pending_score_rows": int(pending_rows),
            "terminal_censored_rows": int(terminal_censored_rows),
        },
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
    }


def pending_score_horizon_state(pending: pd.DataFrame, *, max_timestamps: int = DEFAULT_PENDING_SCORE_MAX_TIMESTAMPS) -> dict[str, Any]:
    if pending.empty:
        return {
            "pending_score_rows": 0,
            "pending_timestamp_count": 0,
            "pending_timestamp_limit": int(max_timestamps),
            "observed_max_assets_per_timestamp": 0,
            "active_pending_score_limit_rows": int(max_timestamps * DEFAULT_MAX_ASSETS_PER_SCORING_TIMESTAMP),
            "within_contract": True,
        }
    work = pending.copy()
    work["decision_timestamp"] = _normalise_timestamp_series(work["decision_timestamp"])
    counts = work.groupby("decision_timestamp", dropna=False)["asset_id"].nunique()
    observed_max_assets = int(counts.max()) if len(counts) else 0
    active_limit = int(max_timestamps * max(DEFAULT_MAX_ASSETS_PER_SCORING_TIMESTAMP, observed_max_assets))
    timestamp_count = int(work["decision_timestamp"].nunique(dropna=True))
    return {
        "pending_score_rows": int(len(work)),
        "pending_timestamp_count": timestamp_count,
        "pending_timestamp_limit": int(max_timestamps),
        "observed_max_assets_per_timestamp": observed_max_assets,
        "active_pending_score_limit_rows": active_limit,
        "within_contract": bool(timestamp_count <= max_timestamps and len(work) <= active_limit),
    }


def enforce_pending_score_horizon(pending: pd.DataFrame, *, max_timestamps: int = DEFAULT_PENDING_SCORE_MAX_TIMESTAMPS) -> dict[str, Any]:
    state = pending_score_horizon_state(pending, max_timestamps=max_timestamps)
    if not state["within_contract"]:
        raise RuntimeError(
            "RESOLVED_PERFORMANCE_V2_PENDING_HORIZON_LIMIT_EXCEEDED:"
            f"timestamps={state['pending_timestamp_count']}>{state['pending_timestamp_limit']};"
            f"rows={state['pending_score_rows']}>{state['active_pending_score_limit_rows']}"
        )
    return state


def apply_summary_validation_marker(root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    marker_path = root / "summary_validation_state_r33.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        marker = {}
    if not marker:
        return summary
    out = dict(summary)
    state = str(marker.get("status") or marker.get("validation_state") or "")
    if state:
        out["status"] = state
    out["validation_state"] = marker
    return out


@dataclass
class ResolvedPerformanceV2Writer:
    root: Path
    family: str
    top_n: int = 20
    target_loader: Callable[[pd.DataFrame], tuple[pd.DataFrame, dict[str, Any]]] | None = None
    pending_score_limit_rows: int = DEFAULT_PENDING_SCORE_LIMIT
    terminal_timestamp: str | None = None

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        publish_resolved_performance_contract_v2(self.contract_path)
        publish_transient_storage_contract_v1(self.transient_storage_contract_path)

    @property
    def contract_path(self) -> Path:
        return self.root / "resolved_performance_contract_v2.json"

    @property
    def transient_storage_contract_path(self) -> Path:
        return self.root / "transient_storage_contract_v1.json"

    @property
    def transient_root(self) -> Path:
        return self.root / "transient_tmp"

    @property
    def pending_scores_path(self) -> Path:
        return self.root / "pending_scores_v2.parquet"

    @property
    def resolved_path(self) -> Path:
        return self.root / "resolved_per_t_performance_v2.parquet"

    @property
    def terminal_censored_path(self) -> Path:
        return self.root / "terminal_censored_v2.parquet"

    @property
    def summary_path(self) -> Path:
        return self.root / "resolved_performance_summary_v2.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.root / "resolved_performance_checkpoint_v2.json"

    def _read_existing(self, path: Path) -> pd.DataFrame:
        if openable_exists(path):
            return pd.read_parquet(openable_path(path))
        return pd.DataFrame()

    def _read_resolved_history(self) -> pd.DataFrame:
        return read_parquet_log(self.root, "resolved_per_t_performance_v2", legacy_path=self.resolved_path)

    def _read_terminal_history(self) -> pd.DataFrame:
        return read_parquet_log(self.root, "terminal_censored_v2", legacy_path=self.terminal_censored_path)

    def commit_predictions(self, predictions: pd.DataFrame, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if predictions.empty:
            return {"committed": False, "reason": "empty_predictions", "evaluation_contract_id": RESOLVED_PERFORMANCE_CONTRACT_V2_ID}
        existing_resolved = self._read_resolved_history()
        existing_timestamps = set()
        if not existing_resolved.empty and "decision_timestamp" in existing_resolved:
            existing_timestamps = set(_normalise_timestamp_series(existing_resolved["decision_timestamp"]).map(lambda ts: ts.isoformat()))
        new_pending = build_pending_score_frame(predictions, family=self.family, metadata=metadata)
        new_pending = new_pending[~_normalise_timestamp_series(new_pending["decision_timestamp"]).map(lambda ts: ts.isoformat()).isin(existing_timestamps)].copy()
        existing_pending = self._read_existing(self.pending_scores_path)
        combined_pending = pd.concat([existing_pending, new_pending], ignore_index=True) if not existing_pending.empty else new_pending
        combined_pending = deduplicate_pending_scores(combined_pending)
        if len(combined_pending) > self.pending_score_limit_rows:
            raise RuntimeError(f"RESOLVED_PERFORMANCE_V2_PENDING_SCORE_LIMIT_EXCEEDED:{len(combined_pending)}>{self.pending_score_limit_rows}")
        if combined_pending.empty:
            targets = pd.DataFrame()
            target_meta = {"target_files_read": 0, "target_rows_loaded": 0}
        elif self.target_loader is not None:
            targets, target_meta = self.target_loader(combined_pending[["asset_id", "decision_timestamp"]])
        else:
            targets = pd.DataFrame()
            target_meta = {"target_loader": "ABSENT_PENDING_RETAINED_UNTIL_RESUME"}
        timestamps = _normalise_timestamp_series(predictions["decision_timestamp"])
        resolution_timestamp = timestamps.max()
        resolved_new, pending_after, terminal_new, resolution_meta = resolve_pending_score_frame(
            combined_pending,
            targets,
            resolution_timestamp=resolution_timestamp,
            top_n=self.top_n,
            existing_resolved=existing_resolved,
            terminal_timestamp=self.terminal_timestamp,
        )
        try:
            horizon_state = enforce_pending_score_horizon(pending_after)
        except RuntimeError as exc:
            if "PENDING_HORIZON_LIMIT_EXCEEDED" not in str(exc):
                raise
            horizon_state = pending_score_horizon_state(pending_after)
            horizon_state["within_contract"] = False
            horizon_state["retention_exception"] = "PENDING_HORIZON_OVERFLOW_RETAINED_BOUNDED_UNTIL_TARGET_MATURITY"
            horizon_state["retention_exception_detail"] = str(exc)
        if not existing_resolved.empty and not resolved_new.empty:
            existing_hashes = dict(zip(existing_resolved["decision_timestamp"].astype(str), existing_resolved.get("row_hash", pd.Series([""] * len(existing_resolved))).astype(str)))
            conflicts = [
                row
                for row in resolved_new.itertuples(index=False)
                if str(getattr(row, "decision_timestamp")) in existing_hashes and existing_hashes[str(getattr(row, "decision_timestamp"))] != str(getattr(row, "row_hash"))
            ]
            if conflicts:
                raise RuntimeError("RESOLVED_PERFORMANCE_V2_DUPLICATE_RESOLVED_METRIC_REFUSAL")
            resolved_new = resolved_new[~resolved_new["decision_timestamp"].astype(str).isin(existing_hashes)].copy()
        combined_resolved = pd.concat([existing_resolved, resolved_new], ignore_index=True) if not existing_resolved.empty else resolved_new
        if not combined_resolved.empty:
            combined_resolved["decision_timestamp"] = _normalise_timestamp_series(combined_resolved["decision_timestamp"]).map(lambda ts: ts.isoformat())
            combined_resolved = combined_resolved.sort_values(["family", "decision_timestamp"]).drop_duplicates(["family", "decision_timestamp"], keep="last")
        existing_terminal = self._read_terminal_history()
        combined_terminal = pd.concat([existing_terminal, terminal_new], ignore_index=True) if not existing_terminal.empty else terminal_new
        if not combined_terminal.empty:
            combined_terminal = combined_terminal.drop_duplicates(["family", "decision_timestamp"], keep="last")
        resolved_pub = append_parquet_log(self.root, "resolved_per_t_performance_v2", resolved_new, temp_root=self.transient_root) if not resolved_new.empty else {"appended": False, "rows": 0, "bytes": 0, "parts": []}
        pending_pub = publish_parquet_atomic(self.pending_scores_path, pending_after, temp_root=self.transient_root)
        terminal_pub = append_parquet_log(self.root, "terminal_censored_v2", terminal_new, temp_root=self.transient_root) if not terminal_new.empty else {"appended": False, "rows": 0, "bytes": 0, "parts": []}
        summary = apply_summary_validation_marker(self.root, resolved_v2_summary(combined_resolved))
        write_json_atomic(self.summary_path, summary, advisory=False)
        oldest_pending = ""
        if not pending_after.empty:
            oldest_pending = _normalise_timestamp_series(pending_after["decision_timestamp"]).min().isoformat()
        temp_inventory = temporary_file_inventory(self.transient_root)
        prior_checkpoint = {}
        try:
            prior_checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except Exception:
            prior_checkpoint = {}
        transient_peak = max(int(prior_checkpoint.get("peak_temporary_bytes", 0) or 0), int(temp_inventory["bytes"]))
        durable_bytes = (
            directory_size_bytes(parquet_log_parts_dir(self.root, "resolved_per_t_performance_v2"))
            + directory_size_bytes(parquet_log_parts_dir(self.root, "terminal_censored_v2"))
            + int(os.stat(openable_path(self.pending_scores_path)).st_size if openable_exists(self.pending_scores_path) else 0)
        )
        checkpoint = {
            "family": self.family,
            "evaluation_contract_version": RESOLVED_PERFORMANCE_CONTRACT_V2_VERSION,
            "evaluation_contract_id": RESOLVED_PERFORMANCE_CONTRACT_V2_ID,
            "evaluation_contract_hash": resolved_performance_contract_v2_hash(),
            "transient_storage_contract_id": TRANSIENT_STORAGE_CONTRACT_V1_ID,
            "transient_storage_contract_hash": transient_storage_contract_v1_hash(),
            "pending_score_rows": int(len(pending_after)),
            "pending_logical_unit": "asset_score_row",
            "pending_timestamp_count": int(horizon_state["pending_timestamp_count"]),
            "pending_timestamp_limit": int(horizon_state["pending_timestamp_limit"]),
            "active_pending_score_limit_rows": int(horizon_state["active_pending_score_limit_rows"]),
            "oldest_pending_timestamp": oldest_pending,
            "resolved_performance_rows": int(len(combined_resolved)),
            "rank_ic_valid_rows": int(summary.get("rank_ic", {}).get("valid_rows") or 0),
            "return_valid_rows": int(summary.get("returns", {}).get("resolved_portfolio_observations") or 0),
            "terminal_censored_rows": int(len(combined_terminal)),
            "bounded_pending_storage_bytes": int(pending_pub.get("bytes", 0) or 0),
            "durable_metrics_bytes": durable_bytes,
            "current_temporary_bytes": int(temp_inventory["bytes"]),
            "peak_temporary_bytes": int(transient_peak),
            "current_batch_staging_bytes": int(temp_inventory["bytes"]),
            "last_cleanup_result": "NO_STALE_TMP_FILES" if int(temp_inventory["file_count"]) == 0 else "TMP_FILES_PRESENT",
            "permanent_full_prediction_files_written": 0,
            "last_resolution_timestamp": pd.Timestamp(resolution_timestamp).tz_convert("UTC").isoformat(),
            "target_load": target_meta,
            "resolution_meta": resolution_meta,
            "checkpoint_utc": pd.Timestamp.now("UTC").isoformat(),
        }
        write_json_atomic(self.checkpoint_path, checkpoint, advisory=False)
        return {
            "committed": True,
            "evaluation_contract_id": RESOLVED_PERFORMANCE_CONTRACT_V2_ID,
            "evaluation_contract_hash": resolved_performance_contract_v2_hash(),
            "pending_score_rows": int(len(pending_after)),
            "resolved_rows_added": int(len(resolved_new)),
            "resolved_performance_rows": int(len(combined_resolved)),
            "rank_ic_valid_rows": checkpoint["rank_ic_valid_rows"],
            "return_valid_rows": checkpoint["return_valid_rows"],
            "terminal_censored_rows": checkpoint["terminal_censored_rows"],
            "pending_timestamp_count": checkpoint["pending_timestamp_count"],
            "current_temporary_bytes": checkpoint["current_temporary_bytes"],
            "peak_temporary_bytes": checkpoint["peak_temporary_bytes"],
            "resolved_publication": resolved_pub,
            "pending_publication": pending_pub,
            "terminal_censored_publication": terminal_pub,
            "summary": summary,
            "checkpoint": checkpoint,
        }


@dataclass
class ResolvedPerformanceV3Writer:
    root: Path
    family: str
    top_n: int = 20
    target_loader: Callable[[pd.DataFrame], tuple[pd.DataFrame, dict[str, Any]]] | None = None
    pending_score_limit_rows: int = DEFAULT_PENDING_SCORE_LIMIT
    terminal_timestamp: str | None = None
    min_rank_ic_cross_section: int = DEFAULT_V3_MIN_RANK_IC_CROSS_SECTION
    sleeve_count: int = DEFAULT_V3_SLEEVE_COUNT
    audit_sample_rows: int = DEFAULT_V3_AUDIT_SAMPLE_ROWS_PER_TIMESTAMP
    transaction_cost_bps_per_unit_turnover: float = TOP_N_COST_BPS_PER_UNIT_TURNOVER

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        publish_resolved_performance_contract_v3(self.contract_path)
        publish_transient_storage_contract_v1(self.transient_storage_contract_path)

    @property
    def contract_path(self) -> Path:
        return self.root / "resolved_performance_contract_v3.json"

    @property
    def transient_storage_contract_path(self) -> Path:
        return self.root / "transient_storage_contract_v1.json"

    @property
    def transient_root(self) -> Path:
        return self.root / "transient_tmp"

    @property
    def pending_scores_path(self) -> Path:
        return self.root / "pending_scores_v3.parquet"

    @property
    def checkpoint_path(self) -> Path:
        return self.root / "resolved_performance_checkpoint_v3.json"

    @property
    def summary_path(self) -> Path:
        return self.root / "resolved_performance_summary_v3.json"

    def _read_existing(self, path: Path) -> pd.DataFrame:
        if openable_exists(path):
            return pd.read_parquet(openable_path(path))
        return pd.DataFrame()

    def _read_log(self, stem: str) -> pd.DataFrame:
        return read_parquet_log(self.root, stem)

    def _append_refit_event(self, predictions: pd.DataFrame, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        metadata = metadata or {}
        timestamps = _normalise_timestamp_series(predictions["decision_timestamp"]).dropna().sort_values()
        if timestamps.empty:
            return {"appended": False, "reason": "NO_TIMESTAMPS"}
        diffs = timestamps.drop_duplicates().diff().dropna().dt.total_seconds().astype(int)
        five_minute = bool(diffs.empty or (diffs % (DEFAULT_DECISION_CADENCE_MINUTES * 60) == 0).all())
        training_cutoff = str(metadata.get("training_cutoff") or metadata.get("refit_T") or "")
        existing = self._read_log("refit_events_v3")
        key = {
            "family": self.family,
            "training_cutoff": training_cutoff,
            "model_hash": str(metadata.get("model_hash", "")),
        }
        if not existing.empty:
            duplicates = existing[
                (existing.get("family", pd.Series(dtype=str)).astype(str) == key["family"])
                & (existing.get("training_cutoff", pd.Series(dtype=str)).astype(str) == key["training_cutoff"])
                & (existing.get("model_hash", pd.Series(dtype=str)).astype(str) == key["model_hash"])
            ]
            if not duplicates.empty:
                return {"appended": False, "reason": "DUPLICATE_REFIT_EVENT"}
        row = {
            **key,
            "refit_session_date": pd.Timestamp(training_cutoff).date().isoformat() if training_cutoff else "",
            "scored_session_dates_json": json.dumps(sorted({ts.date().isoformat() for ts in timestamps}), separators=(",", ":")),
            "scored_decision_count": int(timestamps.nunique()),
            "first_scored_decision_timestamp": timestamps.min().isoformat(),
            "last_scored_decision_timestamp": timestamps.max().isoformat(),
            "daily_refit_with_five_minute_scoring": five_minute,
            "no_five_minute_retraining": True,
            "evaluation_contract_id": RESOLVED_PERFORMANCE_CONTRACT_V3_ID,
            "evaluation_contract_hash": resolved_performance_contract_v3_hash(),
        }
        row["row_hash"] = stable_hash(row)
        return append_parquet_log(self.root, "refit_events_v3", pd.DataFrame([row]), temp_root=self.transient_root)

    def _pending_ledger_row(self, *, resolution_timestamp: pd.Timestamp, pending_after: pd.DataFrame, terminal_new: pd.DataFrame, meta: Mapping[str, Any]) -> dict[str, Any]:
        horizon = pending_score_horizon_state(pending_after)
        row = {
            "family": self.family,
            "resolution_timestamp": pd.Timestamp(resolution_timestamp).tz_convert("UTC").isoformat(),
            "pending_score_rows": int(len(pending_after)),
            "pending_timestamp_count": int(horizon["pending_timestamp_count"]),
            "pending_timestamp_limit": int(horizon["pending_timestamp_limit"]),
            "terminal_censored_rows_added": int(len(terminal_new)),
            "resolved_timestamps_added": int(meta.get("resolved_timestamps", 0) or 0),
            "evaluation_contract_id": RESOLVED_PERFORMANCE_CONTRACT_V3_ID,
            "evaluation_contract_hash": resolved_performance_contract_v3_hash(),
        }
        row["row_hash"] = stable_hash(row)
        return row

    def commit_predictions(self, predictions: pd.DataFrame, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        with AtomicJsonProcessLock(
            self.root / APPEND_COMMIT_LOCK_NAME,
            family=self.family,
            purpose="resolved_performance_v3_commit",
            namespace=str(self.root.resolve()),
            timeout_seconds=0.0,
        ):
            return self._commit_predictions_unlocked(predictions, metadata=metadata)

    def _commit_predictions_unlocked(self, predictions: pd.DataFrame, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if predictions.empty:
            return {"committed": False, "reason": "empty_predictions", "evaluation_contract_id": RESOLVED_PERFORMANCE_CONTRACT_V3_ID}
        existing_rank_ic = self._read_log("rank_ic_v3")
        existing_sleeves = self._read_log("sleeve_maturity_ledger_v3")
        existing_daily = self._read_log("daily_portfolio_returns_v3")
        existing_terminal = self._read_log("terminal_censored_v3")
        existing_timestamps: set[str] = set()
        if not existing_rank_ic.empty and "decision_timestamp" in existing_rank_ic:
            existing_timestamps = set(_normalise_timestamp_series(existing_rank_ic["decision_timestamp"]).map(lambda ts: ts.isoformat()))
        new_pending = build_pending_score_frame_v3(predictions, family=self.family, metadata=metadata)
        new_pending = new_pending[~_normalise_timestamp_series(new_pending["decision_timestamp"]).map(lambda ts: ts.isoformat()).isin(existing_timestamps)].copy()
        existing_pending = self._read_existing(self.pending_scores_path)
        combined_pending = pd.concat([existing_pending, new_pending], ignore_index=True) if not existing_pending.empty else new_pending
        combined_pending = deduplicate_pending_scores(combined_pending)
        if len(combined_pending) > self.pending_score_limit_rows:
            raise RuntimeError(f"RESOLVED_PERFORMANCE_V3_PENDING_SCORE_LIMIT_EXCEEDED:{len(combined_pending)}>{self.pending_score_limit_rows}")
        if combined_pending.empty:
            targets = pd.DataFrame()
            target_meta = {"target_files_read": 0, "target_rows_loaded": 0}
        elif self.target_loader is not None:
            targets, target_meta = self.target_loader(combined_pending[["asset_id", "decision_timestamp"]])
        else:
            targets = pd.DataFrame()
            target_meta = {"target_loader": "ABSENT_PENDING_RETAINED_UNTIL_RESUME"}
        timestamps = _normalise_timestamp_series(predictions["decision_timestamp"])
        resolution_timestamp = timestamps.max()
        resolved = resolve_pending_score_frame_v3(
            combined_pending,
            targets,
            resolution_timestamp=resolution_timestamp,
            top_n=self.top_n,
            existing_rank_ic=existing_rank_ic,
            existing_sleeves=existing_sleeves,
            terminal_timestamp=self.terminal_timestamp,
            min_rank_ic_cross_section=self.min_rank_ic_cross_section,
            audit_sample_rows=self.audit_sample_rows,
            sleeve_count=self.sleeve_count,
            transaction_cost_bps_per_unit_turnover=self.transaction_cost_bps_per_unit_turnover,
        )
        pending_after = resolved["remaining"]
        terminal_new = resolved["terminal_censored"]
        try:
            horizon_state = enforce_pending_score_horizon(pending_after)
        except RuntimeError as exc:
            if "PENDING_HORIZON_LIMIT_EXCEEDED" not in str(exc):
                raise
            horizon_state = pending_score_horizon_state(pending_after)
            horizon_state["within_contract"] = False
            horizon_state["retention_exception"] = "PENDING_HORIZON_OVERFLOW_RETAINED_BOUNDED_UNTIL_TARGET_MATURITY"
            horizon_state["retention_exception_detail"] = str(exc)
        rank_pub = append_parquet_log(self.root, "rank_ic_v3", resolved["rank_ic"], temp_root=self.transient_root) if not resolved["rank_ic"].empty else {"appended": False, "rows": 0, "bytes": 0, "parts": []}
        trace_pub = append_parquet_log(self.root, "decision_trace_v3", resolved["decision_trace"], temp_root=self.transient_root) if not resolved["decision_trace"].empty else {"appended": False, "rows": 0, "bytes": 0, "parts": []}
        sleeve_pub = append_parquet_log(self.root, "sleeve_maturity_ledger_v3", resolved["sleeves"], temp_root=self.transient_root) if not resolved["sleeves"].empty else {"appended": False, "rows": 0, "bytes": 0, "parts": []}
        cost_pub = append_parquet_log(self.root, "transaction_costs_v3", resolved["transaction_costs"], temp_root=self.transient_root) if not resolved["transaction_costs"].empty else {"appended": False, "rows": 0, "bytes": 0, "parts": []}
        sample_pub = append_parquet_log(self.root, "rank_ic_audit_sample_v3", resolved["audit_sample"], temp_root=self.transient_root) if not resolved["audit_sample"].empty else {"appended": False, "rows": 0, "bytes": 0, "parts": []}
        terminal_pub = append_parquet_log(self.root, "terminal_censored_v3", terminal_new, temp_root=self.transient_root) if not terminal_new.empty else {"appended": False, "rows": 0, "bytes": 0, "parts": []}
        pending_pub = publish_parquet_atomic(self.pending_scores_path, pending_after, temp_root=self.transient_root)
        pending_ledger_pub = append_parquet_log(
            self.root,
            "pending_outcome_ledger_v3",
            pd.DataFrame([self._pending_ledger_row(resolution_timestamp=resolution_timestamp, pending_after=pending_after, terminal_new=terminal_new, meta=resolved["meta"])]),
            timestamp_col="resolution_timestamp",
            temp_root=self.transient_root,
        )
        refit_pub = self._append_refit_event(predictions, metadata)
        all_sleeves = self._read_log("sleeve_maturity_ledger_v3")
        all_daily = self._read_log("daily_portfolio_returns_v3")
        daily_new = _daily_rows_from_sleeves(all_sleeves, existing_daily=all_daily, resolution_timestamp=resolution_timestamp)
        daily_pub = append_parquet_log(self.root, "daily_portfolio_returns_v3", daily_new, timestamp_col="session_date", temp_root=self.transient_root) if not daily_new.empty else {"appended": False, "rows": 0, "bytes": 0, "parts": []}
        all_rank_ic = self._read_log("rank_ic_v3")
        all_daily = self._read_log("daily_portfolio_returns_v3")
        all_terminal = self._read_log("terminal_censored_v3")
        summary = resolved_v3_summary(
            all_rank_ic,
            all_daily,
            all_sleeves,
            pending_rows=int(len(pending_after)),
            terminal_censored_rows=int(len(all_terminal)),
        )
        write_json_atomic(self.summary_path, summary, advisory=False)
        temp_inventory = temporary_file_inventory(self.transient_root)
        prior_checkpoint = {}
        try:
            prior_checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except Exception:
            prior_checkpoint = {}
        transient_peak = max(int(prior_checkpoint.get("peak_temporary_bytes", 0) or 0), int(temp_inventory["bytes"]))
        stems = [
            "rank_ic_v3",
            "decision_trace_v3",
            "sleeve_maturity_ledger_v3",
            "daily_portfolio_returns_v3",
            "transaction_costs_v3",
            "pending_outcome_ledger_v3",
            "rank_ic_audit_sample_v3",
            "terminal_censored_v3",
            "refit_events_v3",
        ]
        partition_count = sum(len(parquet_log_part_paths(self.root, stem)) for stem in stems)
        duplicate_count = 0
        if not all_rank_ic.empty and "decision_timestamp" in all_rank_ic:
            duplicate_count = int(all_rank_ic.duplicated(["family", "decision_timestamp"]).sum())
        durable_bytes = sum(directory_size_bytes(parquet_log_parts_dir(self.root, stem)) for stem in stems) + int(os.stat(openable_path(self.pending_scores_path)).st_size if openable_exists(self.pending_scores_path) else 0)
        first_uncommitted = ""
        if not pending_after.empty:
            first_uncommitted = _normalise_timestamp_series(pending_after["decision_timestamp"]).min().isoformat()
        refits = self._read_log("refit_events_v3")
        checkpoint = {
            "family": self.family,
            "evaluation_contract_version": RESOLVED_PERFORMANCE_CONTRACT_V3_VERSION,
            "evaluation_contract_id": RESOLVED_PERFORMANCE_CONTRACT_V3_ID,
            "evaluation_contract_hash": resolved_performance_contract_v3_hash(),
            "transient_storage_contract_id": TRANSIENT_STORAGE_CONTRACT_V1_ID,
            "transient_storage_contract_hash": transient_storage_contract_v1_hash(),
            "pending_score_rows": int(len(pending_after)),
            "pending_logical_unit": "asset_score_row",
            "pending_timestamp_count": int(horizon_state["pending_timestamp_count"]),
            "pending_timestamp_limit": int(horizon_state["pending_timestamp_limit"]),
            "active_pending_score_limit_rows": int(horizon_state["active_pending_score_limit_rows"]),
            "pending_horizon_within_contract": bool(horizon_state.get("within_contract", False)),
            "pending_retention_exception": horizon_state.get("retention_exception", ""),
            "pending_retention_exception_detail": horizon_state.get("retention_exception_detail", ""),
            "first_uncommitted_timestamp": first_uncommitted,
            "resolved_performance_rows": int(len(all_rank_ic)),
            "rank_ic_valid_rows": int(summary.get("rank_ic", {}).get("valid_timestamps") or 0),
            "daily_return_rows": int(summary.get("returns", {}).get("daily_return_rows") or 0),
            "return_valid_rows": int(summary.get("returns", {}).get("daily_return_rows") or 0),
            "terminal_censored_rows": int(len(all_terminal)),
            "bounded_pending_storage_bytes": int(pending_pub.get("bytes", 0) or 0),
            "pending_buffer_bytes": int(pending_pub.get("bytes", 0) or 0),
            "durable_metrics_bytes": int(durable_bytes),
            "current_temporary_bytes": int(temp_inventory["bytes"]),
            "peak_temporary_bytes": int(transient_peak),
            "largest_temporary_file_bytes": int(temp_inventory["files"][0]["bytes"]) if temp_inventory.get("files") else 0,
            "partition_count": int(partition_count),
            "duplicate_count": int(duplicate_count),
            "permanent_full_prediction_files_written": 0,
            "last_resolution_timestamp": pd.Timestamp(resolution_timestamp).tz_convert("UTC").isoformat(),
            "target_load": target_meta,
            "resolution_meta": resolved["meta"],
            "daily_refit_count": int(len(refits)),
            "last_refit_session": str(refits["refit_session_date"].iloc[-1]) if not refits.empty and "refit_session_date" in refits else "",
            "daily_refit_with_five_minute_scoring": bool(refits["daily_refit_with_five_minute_scoring"].all()) if not refits.empty and "daily_refit_with_five_minute_scoring" in refits else False,
            "no_five_minute_retraining": bool(refits["no_five_minute_retraining"].all()) if not refits.empty and "no_five_minute_retraining" in refits else False,
            "checkpoint_utc": pd.Timestamp.now("UTC").isoformat(),
        }
        write_json_atomic(self.checkpoint_path, checkpoint, advisory=False)
        return {
            "committed": True,
            "evaluation_contract_id": RESOLVED_PERFORMANCE_CONTRACT_V3_ID,
            "evaluation_contract_hash": resolved_performance_contract_v3_hash(),
            "pending_score_rows": int(len(pending_after)),
            "resolved_rows_added": int(len(resolved["rank_ic"])),
            "resolved_performance_rows": int(len(all_rank_ic)),
            "rank_ic_valid_rows": checkpoint["rank_ic_valid_rows"],
            "return_valid_rows": checkpoint["return_valid_rows"],
            "terminal_censored_rows": checkpoint["terminal_censored_rows"],
            "pending_timestamp_count": checkpoint["pending_timestamp_count"],
            "pending_horizon_within_contract": checkpoint["pending_horizon_within_contract"],
            "pending_retention_exception": checkpoint["pending_retention_exception"],
            "current_temporary_bytes": checkpoint["current_temporary_bytes"],
            "peak_temporary_bytes": checkpoint["peak_temporary_bytes"],
            "partition_count": checkpoint["partition_count"],
            "duplicate_count": checkpoint["duplicate_count"],
            "rank_ic_publication": rank_pub,
            "decision_trace_publication_v3": trace_pub,
            "sleeve_publication": sleeve_pub,
            "daily_return_publication": daily_pub,
            "transaction_cost_publication": cost_pub,
            "pending_ledger_publication": pending_ledger_pub,
            "audit_sample_publication": sample_pub,
            "terminal_censored_publication": terminal_pub,
            "refit_event_publication": refit_pub,
            "summary": summary,
            "checkpoint": checkpoint,
        }


def append_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def publish_parquet_atomic(path: Path, frame: pd.DataFrame, *, temp_root: Path | None = None) -> dict[str, Any]:
    mkdir_openable(path.parent)
    if temp_root is not None:
        mkdir_openable(temp_root)
        tmp = temp_root / f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    else:
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    frame.to_parquet(openable_path(tmp), index=False)
    for attempt in range(10):
        try:
            os.replace(openable_path(tmp), openable_path(path))
            break
        except PermissionError:
            if attempt == 9:
                raise
            gc.collect()
            time.sleep(0.25 * (attempt + 1))
    return {"path": str(path), "rows": int(len(frame)), "bytes": int(os.stat(openable_path(path)).st_size), "sha256": sha256_file(path)}


def parquet_rows(path: Path) -> int:
    if not openable_exists(path):
        return 0
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(openable_path(path)).metadata.num_rows)
    except Exception:
        return 0


def directory_size_bytes(path: Path) -> int:
    if not openable_exists(path):
        return 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(openable_path(path)):
        for name in filenames:
            try:
                total += int(os.path.getsize(os.path.join(dirpath, name)))
            except OSError:
                continue
    return total


def parquet_log_manifest_path(root: Path, stem: str) -> Path:
    return root / f"{stem}_manifest.json"


def parquet_log_parts_dir(root: Path, stem: str) -> Path:
    return root / f"{stem}_parts"


def _read_parquet_log_manifest(root: Path, stem: str) -> dict[str, Any]:
    path = parquet_log_manifest_path(root, stem)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    parts = payload.get("parts", [])
    if not isinstance(parts, list):
        parts = []
    return {
        "schema_version": payload.get("schema_version", "DS24_APPEND_ONLY_PARQUET_LOG_V1"),
        "stem": payload.get("stem", stem),
        "parts": parts,
        "total_rows": int(payload.get("total_rows", 0) or 0),
        "total_bytes": int(payload.get("total_bytes", 0) or 0),
        "updated_at_utc": payload.get("updated_at_utc", ""),
    }


def _write_parquet_log_manifest(root: Path, stem: str, manifest: Mapping[str, Any]) -> None:
    path = parquet_log_manifest_path(root, stem)
    payload = dict(manifest)
    payload["schema_version"] = "DS24_APPEND_ONLY_PARQUET_LOG_V1"
    payload["stem"] = stem
    payload["updated_at_utc"] = pd.Timestamp.now("UTC").isoformat()
    write_json_atomic(path, payload, advisory=False)


def parquet_log_part_paths(root: Path, stem: str) -> list[Path]:
    manifest = _read_parquet_log_manifest(root, stem)
    paths: list[Path] = []
    for part in manifest.get("parts", []):
        path_text = str(part.get("path", ""))
        if path_text:
            path = Path(path_text)
            paths.append(path if path.is_absolute() else root / path)
    existing: list[Path] = []
    parts_root = parquet_log_parts_dir(root, stem)
    if openable_exists(parts_root):
        for dirpath, _dirnames, filenames in os.walk(openable_path(parts_root)):
            for name in filenames:
                if name.endswith(".parquet"):
                    path_text = os.path.join(dirpath, name)
                    if path_text.startswith("\\\\?\\"):
                        path_text = path_text[4:]
                    existing.append(Path(path_text))
        existing = sorted(existing)
    seen = {path.resolve() for path in paths if openable_exists(path)}
    for path in existing:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved not in seen:
            paths.append(path)
            seen.add(resolved)
    return [path for path in paths if openable_exists(path)]


def parquet_log_rows(root: Path, stem: str, *, legacy_path: Path | None = None) -> int:
    rows = parquet_rows(legacy_path) if legacy_path is not None else 0
    manifest = _read_parquet_log_manifest(root, stem)
    part_rows = int(manifest.get("total_rows", 0) or 0)
    if part_rows:
        return rows + part_rows
    return rows + sum(parquet_rows(path) for path in parquet_log_part_paths(root, stem))


def read_parquet_log(root: Path, stem: str, *, legacy_path: Path | None = None, columns: Sequence[str] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if legacy_path is not None and openable_exists(legacy_path):
        frames.append(pd.read_parquet(openable_path(legacy_path), columns=columns))
    for path in parquet_log_part_paths(root, stem):
        frames.append(pd.read_parquet(openable_path(path), columns=columns))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def append_parquet_log(
    root: Path,
    stem: str,
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "decision_timestamp",
    temp_root: Path | None = None,
) -> dict[str, Any]:
    if frame.empty:
        return {"appended": False, "rows": 0, "bytes": 0, "parts": []}
    work = frame.copy()
    if timestamp_col in work.columns:
        timestamps = _normalise_timestamp_series(work[timestamp_col])
        min_ts = timestamps.min()
        max_ts = timestamps.max()
        part_date = min_ts.strftime("%Y-%m-%d") if pd.notna(min_ts) else "unknown"
        min_text = min_ts.isoformat() if pd.notna(min_ts) else ""
        max_text = max_ts.isoformat() if pd.notna(max_ts) else ""
    else:
        part_date = "unknown"
        min_text = ""
        max_text = ""
    part_dir = parquet_log_parts_dir(root, stem) / f"decision_date={part_date}"
    part_name = f"part-{os.getpid()}-{time.time_ns()}.parquet"
    part_path = part_dir / part_name
    pub = publish_parquet_atomic(part_path, work, temp_root=temp_root)
    manifest = _read_parquet_log_manifest(root, stem)
    rel_path = str(part_path.relative_to(root)).replace("\\", "/")
    part_record = {
        "path": rel_path,
        "rows": int(len(work)),
        "bytes": int(pub.get("bytes", 0) or 0),
        "sha256": pub.get("sha256", ""),
        "min_decision_timestamp": min_text,
        "max_decision_timestamp": max_text,
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
    }
    parts = list(manifest.get("parts", [])) + [part_record]
    manifest["parts"] = parts
    manifest["total_rows"] = int(sum(int(part.get("rows", 0) or 0) for part in parts))
    manifest["total_bytes"] = int(sum(int(part.get("bytes", 0) or 0) for part in parts))
    _write_parquet_log_manifest(root, stem, manifest)
    return {"appended": True, "rows": int(len(work)), "bytes": int(pub.get("bytes", 0) or 0), "parts": [part_record], "manifest_path": str(parquet_log_manifest_path(root, stem))}


def temporary_file_inventory(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    total = 0
    if openable_exists(root):
        for dirpath, _dirnames, filenames in os.walk(openable_path(root)):
            for name in filenames:
                if not (name.endswith(".tmp") or name.endswith(".temp") or name.endswith(".partial") or ".parquet." in name and name.endswith(".tmp")):
                    continue
                path_text = os.path.join(dirpath, name)
                if path_text.startswith("\\\\?\\"):
                    path_text = path_text[4:]
                path = Path(path_text)
                try:
                    size = int(path.stat().st_size)
                except OSError:
                    continue
                total += size
                files.append({"path": str(path), "bytes": size, "modified_at": pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat()})
    return {"root": str(root), "bytes": total, "file_count": len(files), "files": sorted(files, key=lambda row: row["bytes"], reverse=True)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(openable_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class MetricsOnlyEvidenceWriter:
    root: Path
    family: str
    top_n: int = 20
    pending_rollback_timestamps: int = 24
    enable_resolved_performance_v2: bool = False
    enable_resolved_performance_v3: bool = False
    target_loader: Callable[[pd.DataFrame], tuple[pd.DataFrame, dict[str, Any]]] | None = None
    pending_score_limit_rows: int = DEFAULT_PENDING_SCORE_LIMIT
    terminal_timestamp: str | None = None
    namespace_lease_enabled: bool = False
    resume_generation: int | str | None = None
    command_hash: str | None = None
    configuration_hash: str | None = None
    evaluation_contract_hash: str | None = None

    def __post_init__(self) -> None:
        self._resolved_v2_writer: ResolvedPerformanceV2Writer | None = None
        self._resolved_v3_writer: ResolvedPerformanceV3Writer | None = None
        self._namespace_lease: NamespaceWriterLease | None = None
        if self.enable_resolved_performance_v2 and self.enable_resolved_performance_v3:
            raise ValueError("RESOLVED_PERFORMANCE_VERSION_CONFLICT")
        if self.namespace_lease_enabled and not self.enable_resolved_performance_v3:
            raise ValueError("DS24_R37_NAMESPACE_LEASE_REQUIRES_RESOLVED_PERFORMANCE_V3")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.namespace_lease_enabled:
            self._namespace_lease = NamespaceWriterLease(
                self.root,
                family=self.family,
                resume_generation=self.resume_generation,
                command_hash=self.command_hash,
                configuration_hash=self.configuration_hash,
                evaluation_contract_hash=self.evaluation_contract_hash or resolved_performance_contract_v3_hash(),
            )
            self._namespace_lease.acquire()
        publish_policy(self.root / "metrics_only_authority.json")
        if self.enable_resolved_performance_v3:
            self._resolved_v3_writer = ResolvedPerformanceV3Writer(
                self.root,
                family=self.family,
                top_n=self.top_n,
                target_loader=self.target_loader,
                pending_score_limit_rows=self.pending_score_limit_rows,
                terminal_timestamp=self.terminal_timestamp,
            )
        elif self.enable_resolved_performance_v2:
            self._resolved_v2_writer = ResolvedPerformanceV2Writer(
                self.root,
                family=self.family,
                top_n=self.top_n,
                target_loader=self.target_loader,
                pending_score_limit_rows=self.pending_score_limit_rows,
                terminal_timestamp=self.terminal_timestamp,
            )

    @property
    def namespace_lease_path(self) -> Path:
        return self.root / NAMESPACE_WRITER_LEASE_NAME

    def namespace_lease_payload(self) -> dict[str, Any]:
        if self._namespace_lease is None:
            return {}
        return _read_json_mapping(self.namespace_lease_path)

    def heartbeat_namespace_lease(self, *, phase: str = "", cursor: str = "") -> dict[str, Any]:
        if self._namespace_lease is None:
            return {}
        return self._namespace_lease.heartbeat(phase=phase, cursor=cursor)

    def release_namespace_lease(self) -> bool:
        if self._namespace_lease is None:
            return False
        return self._namespace_lease.release()

    @property
    def metrics_path(self) -> Path:
        return self.root / "per_t_metrics.parquet"

    @property
    def decisions_path(self) -> Path:
        return self.root / "decision_trace.parquet"

    @property
    def pending_path(self) -> Path:
        return self.root / "pending_buffer.parquet"

    @property
    def checkpoint_path(self) -> Path:
        return self.root / "checkpoint.json"

    @property
    def resolved_v2_checkpoint_path(self) -> Path:
        return self.root / "resolved_performance_checkpoint_v2.json"

    @property
    def resolved_v3_checkpoint_path(self) -> Path:
        return self.root / "resolved_performance_checkpoint_v3.json"

    def _read_existing(self, path: Path) -> pd.DataFrame:
        if openable_exists(path):
            return pd.read_parquet(openable_path(path))
        return pd.DataFrame()

    def _read_metrics_history(self) -> pd.DataFrame:
        if self.enable_resolved_performance_v2 or self.enable_resolved_performance_v3:
            return read_parquet_log(self.root, "per_t_metrics", legacy_path=self.metrics_path)
        return self._read_existing(self.metrics_path)

    def _read_decision_history(self) -> pd.DataFrame:
        if self.enable_resolved_performance_v2 or self.enable_resolved_performance_v3:
            return read_parquet_log(self.root, "decision_trace", legacy_path=self.decisions_path)
        return self._read_existing(self.decisions_path)

    def commit_predictions(
        self,
        predictions: pd.DataFrame,
        *,
        targets: pd.DataFrame | None = None,
        expected_assets: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if predictions.empty:
            return {"committed": False, "reason": "empty_predictions"}
        if self._namespace_lease is not None:
            cursor = ""
            if "decision_timestamp" in predictions.columns:
                try:
                    cursor = _normalise_timestamp_series(predictions["decision_timestamp"]).max().isoformat()
                except Exception:
                    cursor = ""
            self._namespace_lease.heartbeat(phase="COMMIT_PREDICTIONS_START", cursor=cursor)
        predictions = predictions.copy()
        predictions["family"] = predictions.get("family", self.family)
        v2_result: dict[str, Any] = {}
        v3_result: dict[str, Any] = {}
        if self._resolved_v2_writer is not None:
            v2_result = self._resolved_v2_writer.commit_predictions(predictions, metadata=metadata)
        if self._resolved_v3_writer is not None:
            v3_result = self._resolved_v3_writer.commit_predictions(predictions, metadata=metadata)
        metrics, decisions = compute_per_t_metrics(predictions, targets, top_n=self.top_n, expected_assets=expected_assets)
        existing_metrics = self._read_metrics_history()
        existing_decisions = self._read_decision_history()
        if not existing_metrics.empty:
            key = set(zip(existing_metrics["family"], existing_metrics["decision_timestamp"].astype(str)))
            metrics = metrics[~metrics.apply(lambda row: (row["family"], str(row["decision_timestamp"])) in key, axis=1)]
        if not existing_decisions.empty:
            key = set(zip(existing_decisions["family"], existing_decisions["decision_timestamp"].astype(str), existing_decisions["asset_id"].astype(str)))
            decisions = decisions[~decisions.apply(lambda row: (row["family"], str(row["decision_timestamp"]), str(row["asset_id"])) in key, axis=1)]
        combined_metrics = pd.concat([existing_metrics, metrics], ignore_index=True) if not existing_metrics.empty else metrics
        combined_decisions = pd.concat([existing_decisions, decisions], ignore_index=True) if not existing_decisions.empty else decisions
        resolved_writer = self._resolved_v3_writer or self._resolved_v2_writer
        resolved_result = v3_result or v2_result
        transient_root = resolved_writer.transient_root if resolved_writer is not None else None
        if self.enable_resolved_performance_v2 or self.enable_resolved_performance_v3:
            metric_pub = append_parquet_log(self.root, "per_t_metrics", metrics, temp_root=transient_root) if not metrics.empty else {"appended": False, "rows": 0, "bytes": 0, "parts": []}
            decision_pub = append_parquet_log(self.root, "decision_trace", decisions, temp_root=transient_root) if not decisions.empty else {"appended": False, "rows": 0, "bytes": 0, "parts": []}
        else:
            metric_pub = publish_parquet_atomic(self.metrics_path, combined_metrics) if not combined_metrics.empty else {}
            decision_pub = publish_parquet_atomic(self.decisions_path, combined_decisions) if not combined_decisions.empty else {}
        pending = predictions.copy()
        pending["committed_at_utc"] = pd.Timestamp.now("UTC").isoformat()
        pending["policy_id"] = POLICY_ID
        pending["policy_hash"] = policy_hash()
        timestamps = sorted(pd.to_datetime(pending["decision_timestamp"], utc=True).drop_duplicates())
        pending_timestamp_limit = min(self.pending_rollback_timestamps, DEFAULT_PENDING_SCORE_MAX_TIMESTAMPS) if self.enable_resolved_performance_v2 or self.enable_resolved_performance_v3 else self.pending_rollback_timestamps
        keep = set(ts.isoformat() for ts in timestamps[-pending_timestamp_limit:])
        pending["decision_timestamp"] = pd.to_datetime(pending["decision_timestamp"], utc=True).map(lambda ts: ts.isoformat())
        pending = pending[pending["decision_timestamp"].isin(keep)].copy()
        pending_pub = publish_parquet_atomic(self.pending_path, pending, temp_root=transient_root)
        temp_inventory = temporary_file_inventory(transient_root) if transient_root is not None else {"bytes": 0, "file_count": 0}
        prior_checkpoint = {}
        try:
            prior_checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except Exception:
            prior_checkpoint = {}
        transient_peak = max(int(prior_checkpoint.get("peak_temporary_bytes", 0) or 0), int(temp_inventory["bytes"]))
        checkpoint = {
            "policy_id": POLICY_ID,
            "policy_hash": policy_hash(),
            "family": self.family,
            "evaluation_contract_version": (
                RESOLVED_PERFORMANCE_CONTRACT_V3_VERSION
                if self._resolved_v3_writer is not None
                else (RESOLVED_PERFORMANCE_CONTRACT_V2_VERSION if self._resolved_v2_writer is not None else "V1")
            ),
            "evaluation_contract_id": resolved_result.get("evaluation_contract_id", ""),
            "evaluation_contract_hash": resolved_result.get("evaluation_contract_hash", ""),
            "last_completed_T": max(timestamps).isoformat() if timestamps else "",
            "metric_rows": int(len(combined_metrics)),
            "decision_rows": int(len(combined_decisions)),
            "pending_rows": int(len(pending)),
            "pending_logical_unit": "rollback_asset_score_row",
            "pending_timestamp_count": int(len(keep)),
            "pending_timestamp_limit": int(pending_timestamp_limit),
            "pending_bytes": pending_pub["bytes"],
            "v2_pending_score_rows": int(v2_result.get("pending_score_rows", 0) or 0),
            "v2_resolved_performance_rows": int(v2_result.get("resolved_performance_rows", 0) or 0),
            "v2_rank_ic_valid_rows": int(v2_result.get("rank_ic_valid_rows", 0) or 0),
            "v2_return_valid_rows": int(v2_result.get("return_valid_rows", 0) or 0),
            "v2_terminal_censored_rows": int(v2_result.get("terminal_censored_rows", 0) or 0),
            "v3_pending_score_rows": int(v3_result.get("pending_score_rows", 0) or 0),
            "v3_resolved_performance_rows": int(v3_result.get("resolved_performance_rows", 0) or 0),
            "v3_rank_ic_valid_rows": int(v3_result.get("rank_ic_valid_rows", 0) or 0),
            "v3_return_valid_rows": int(v3_result.get("return_valid_rows", 0) or 0),
            "v3_terminal_censored_rows": int(v3_result.get("terminal_censored_rows", 0) or 0),
            "v3_partition_count": int(v3_result.get("partition_count", 0) or 0),
            "v3_duplicate_count": int(v3_result.get("duplicate_count", 0) or 0),
            "transient_storage_contract_id": TRANSIENT_STORAGE_CONTRACT_V1_ID if self.enable_resolved_performance_v2 or self.enable_resolved_performance_v3 else "",
            "current_temporary_bytes": int(temp_inventory["bytes"]),
            "peak_temporary_bytes": int(transient_peak),
            "full_prediction_files_written": 0,
            "namespace_writer_lease_path": str(self.namespace_lease_path) if self._namespace_lease is not None else "",
            "namespace_writer_lease": self.namespace_lease_payload(),
            "metadata": dict(metadata or {}),
            "checkpoint_utc": pd.Timestamp.now("UTC").isoformat(),
        }
        write_json_atomic(self.checkpoint_path, checkpoint, advisory=False)
        if self._namespace_lease is not None:
            self._namespace_lease.heartbeat(phase="COMMIT_PREDICTIONS_COMPLETE", cursor=checkpoint["last_completed_T"])
        return {
            "committed": True,
            "policy_id": POLICY_ID,
            "policy_hash": policy_hash(),
            "metric_rows_added": int(len(metrics)),
            "decision_rows_added": int(len(decisions)),
            "pending_rows": int(len(pending)),
            "current_temporary_bytes": checkpoint["current_temporary_bytes"],
            "peak_temporary_bytes": checkpoint["peak_temporary_bytes"],
            "metric_publication": metric_pub,
            "decision_publication": decision_pub,
            "pending_publication": pending_pub,
            "checkpoint": checkpoint,
            "namespace_writer_lease": self.namespace_lease_payload(),
            "resolved_performance_v2": v2_result,
            "resolved_performance_v3": v3_result,
        }


def fail_if_legacy_full_prediction_persistence(storage_contract: str) -> None:
    if storage_contract == LEGACY_FULL_PREDICTION_CONTRACT:
        raise RuntimeError("FULL_PREDICTION_PERSISTENCE_DISABLED_BY_DS24_METRICS_ONLY_RESEARCH_OUTPUT_POLICY_V1")
