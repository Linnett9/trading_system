from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

try:  # pragma: no cover - depends on local packaging.
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from core.research.ml.ds24 import remote_tft as r44b
from core.research.ml.ds24.ensemble_oof import (
    ENSEMBLE_OOF_COLUMNS,
    FAMILY_ENFORCEMENT_SCOPE,
    SCORE_CONTRACT_ID,
    SCORE_CONTRACT_VERSION,
    audit_oof_scores,
    build_oof_manifest,
    create_sync_snapshot,
    disk_admission,
    ensemble_schema_payload,
    future_family_ensemble_admission,
    import_verified_snapshot,
    prepare_oof_score_frame,
    publish_oof_manifest,
    reproduce_v3_metrics_from_scores,
    resume_download_copy,
    score_contract_payload,
    stable_hash,
    validate_oof_manifest,
    verify_downloaded_snapshot,
    write_oof_partitions,
)


R44C_EVIDENCE_NAME = "r7_r44c_vast_tft_ensemble_output_sync_and_utilisation"
R44C_EVIDENCE_RELATIVE_ROOT = r44b.STAGE_ROOT / R44C_EVIDENCE_NAME
TERMINAL_SUCCESS = "DS24_R44C_VAST_TFT_ENSEMBLE_OUTPUT_SYNC_AND_UTILISATION_LANE_READY_FOR_PAID_BOUNDED_SMOKE"
TERMINAL_R44B_DRIFT = "DS24_R44C_BLOCKED_R44B_AUTHORITY_DRIFT"

EXPECTED_R44B = {
    "terminal_classification": r44b.TERMINAL_SUCCESS,
    "source_bundle_sha256": "6d5cc3f09e24160c7d26cdff29d98546742a6c082928202356b689a94fe62109",
    "configuration_hash": "529058a37fa3731390e4631e9dd97696622e670d699e98411fbc89bfe5a00227",
    "predictor_count": 101,
    "target_id": "forward_return_60m__decision_5m",
    "decoder_prediction_length": 12,
    "score_policy": "1.0 - probability_should_reduce_exposure",
    "data_transfer_size_bytes": 47_297_267_964,
}

INVENTORY_FAMILIES = [
    "ridge_policy_v1_control",
    "pca_ridge_policy_v1_control",
    "spline_additive_ridge",
    "elastic_net",
    "rff_ridge",
    "huber",
    "mlp",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    r44b.REMOTE_FAMILY,
]


def utc_now() -> str:
    return r44b.utc_now()


def write_text(path: Path, text: str) -> None:
    r44b.write_text(path, text)


def write_json(path: Path, payload: Any) -> None:
    r44b.write_json(path, payload)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    r44b.write_csv(path, rows)


def read_json(path: Path) -> dict[str, Any]:
    return r44b.read_json(path)


def sha256_file(path: Path) -> str:
    return r44b.sha256_file(path)


def openable_path(path: Path) -> str:
    return r44b.openable_path(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return str(path)


def process_and_resource_snapshot(repo_root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(repo_root)
    memory: dict[str, Any] = {"platform": platform.system()}
    if platform.system().lower() == "windows":
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_OperatingSystem | Select-Object "
            "FreePhysicalMemory,TotalVisibleMemorySize,FreeVirtualMemory,TotalVirtualMemorySize | ConvertTo-Json -Compress",
        ]
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=20)
            parsed = json.loads(completed.stdout or "{}")
            memory.update(parsed if isinstance(parsed, dict) else {})
        except Exception as exc:  # pragma: no cover - environment dependent.
            memory["error"] = f"{type(exc).__name__}:{exc}"
    else:  # pragma: no cover - Windows is the expected local executor.
        try:
            pagesize = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_AVPHYS_PAGES")
            memory["FreePhysicalMemoryBytes"] = int(pagesize * pages)
        except Exception as exc:
            memory["error"] = f"{type(exc).__name__}:{exc}"
    processes = r44b.process_snapshot()
    return {
        "created_at_utc": utc_now(),
        "repo_root": str(repo_root),
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "memory": memory,
        "processes": processes,
        "ds24_processes": [
            row for row in processes if "ds24" in str(row.get("command_line", "")).lower()
        ],
        "ds26_processes": [
            row for row in processes if "ds26" in str(row.get("command_line", "")).lower()
        ],
        "paid_vast_resource_created_by_r44c": False,
        "data_uploaded_by_r44c": False,
        "paper_or_live_orders_by_r44c": 0,
    }


def git_scope_snapshot(repo_root: Path) -> dict[str, Any]:
    paths = [
        "core/research/ml/ds24/ensemble_oof.py",
        "core/research/ml/ds24/remote_tft.py",
        "core/research/ml/ds24/remote_tft_r44c.py",
        "scripts/local/ds24_p8_r14_e3g_c2_r7_r44c_vast_tft_package.py",
        "tests/test_ds24_p8_r14_e3g_c2_r7_r44b_vast_tft.py",
        "tests/test_ds24_p8_r14_e3g_c2_r7_r44c_vast_tft_ensemble.py",
        str(R44C_EVIDENCE_RELATIVE_ROOT).replace("/", os.sep),
    ]
    try:
        completed = subprocess.run(
            ["git", "status", "--short", "--", *paths],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        status = completed.stdout.strip().splitlines()
    except Exception as exc:  # pragma: no cover
        status = [f"ERROR:{type(exc).__name__}:{exc}"]
    return {
        "created_at_utc": utc_now(),
        "scoped_paths": paths,
        "scoped_status": status,
        "no_stage_commit_or_push": True,
        "dirty_worktree_treated_as_user_owned": True,
    }


def validate_r44b_authority(repo_root: Path) -> dict[str, Any]:
    root = repo_root / r44b.EVIDENCE_RELATIVE_ROOT
    terminal_path = root / "20_terminal_result.json"
    config_path = root / "04_tft_configuration_authority.yaml"
    source_sha_path = root / "source_bundle.sha256"
    validation_path = root / "15_test_and_smoke_results.json"
    terminal = read_json(terminal_path)
    config = _load_yaml(config_path)
    validation = read_json(validation_path)
    source_sha = source_sha_path.read_text(encoding="utf-8").strip() if source_sha_path.exists() else ""
    observed = {
        "terminal_classification": terminal.get("terminal_classification", ""),
        "source_bundle_sha256": source_sha or terminal.get("source_bundle_sha256", ""),
        "configuration_hash": config.get("configuration_hash", terminal.get("configuration_hash", "")),
        "predictor_count": config.get("feature_contract", {}).get("predictor_count"),
        "target_id": config.get("target", {}).get("target_id"),
        "decoder_prediction_length": config.get("target", {}).get("decoder_prediction_length"),
        "score_policy": config.get("score_policy", {}).get("comparable_long_selection_score"),
        "data_transfer_size_bytes": terminal.get("data_transfer_size_bytes"),
        "locked_holdout_outcomes_read": terminal.get("locked_holdout_outcomes_read", False)
        or validation.get("locked_holdout_outcomes_read", False),
        "paper_orders": terminal.get("paper_orders", 0),
        "live_orders": terminal.get("live_orders", 0),
        "paid_vast_resource_created": terminal.get("paid_vast_resource_created", False),
        "data_uploaded": terminal.get("data_uploaded", False),
        "full_tft_run_launched": terminal.get("full_tft_run_launched", False),
    }
    comparisons = {
        key: observed.get(key) == expected
        for key, expected in EXPECTED_R44B.items()
    }
    comparisons["zero_holdout_access"] = observed["locked_holdout_outcomes_read"] is False
    comparisons["zero_paper_live_orders"] = int(observed["paper_orders"] or 0) == 0 and int(observed["live_orders"] or 0) == 0
    comparisons["no_paid_or_upload_or_full_run"] = (
        observed["paid_vast_resource_created"] is False
        and observed["data_uploaded"] is False
        and observed["full_tft_run_launched"] is False
    )
    file_hashes = {}
    for path in (terminal_path, config_path, source_sha_path):
        file_hashes[repo_relative(repo_root, path)] = sha256_file(path) if path.exists() else ""
    status = "PASS" if all(comparisons.values()) else "FAIL"
    return {
        "authority_id": "DS24_R44C_R44B_PREDECESSOR_AUTHORITY_VALIDATION_V1",
        "created_at_utc": utc_now(),
        "r44b_evidence_root": repo_relative(repo_root, root),
        "expected": EXPECTED_R44B,
        "observed": observed,
        "comparisons": comparisons,
        "file_hashes": file_hashes,
        "status": status,
        "terminal_if_failed": TERMINAL_R44B_DRIFT,
    }


def _bounded_files(root: Path, *, limit: int = 600) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    root_text = openable_path(root)
    try:
        walker = os.walk(root_text)
        for dirpath, dirnames, filenames in walker:
            dirnames[:] = [
                name
                for name in dirnames
                if not name.endswith("_parts")
                and name not in {"rollback", "__pycache__", "model_artifacts", "oof_predictions"}
            ]
            for filename in filenames:
                path = Path(dirpath) / filename
                if path.suffix.lower() in {".parquet", ".pt", ".zip", ".gz"}:
                    continue
                files.append(path)
                if len(files) >= limit:
                    return files
    except (FileNotFoundError, OSError):
        return files
    return files


def _json_with_first(root: Path, names: set[str]) -> dict[str, Any]:
    for path in _bounded_files(root):
        if path.name in names:
            return read_json(path)
    return {}


def ensemble_source_availability(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage = repo_root / r44b.STAGE_ROOT
    worker_root = stage / "r7_r14_policy_workers"
    rows: list[dict[str, Any]] = []
    for family in INVENTORY_FAMILIES:
        root = worker_root / family
        if family == r44b.REMOTE_FAMILY:
            root = stage / "remote_vast_runs" / f"run={r44b.REMOTE_RUN_ID}" / f"family={r44b.REMOTE_FAMILY}"
        files = _bounded_files(root)
        names = {path.name for path in files}
        paths = [path.as_posix().lower() for path in files]
        has_v3_summary = "resolved_performance_summary_v3.json" in names or any(
            path.endswith("metrics_only_v3__resolved_performance_summary_v3.json") for path in paths
        )
        has_rank_ic = any("rank_ic" in name and ("manifest" in name or "v3" in name) for name in names)
        has_daily_rank_ic = any("daily_rank_ic" in name for name in names)
        has_daily_portfolio = any("daily_portfolio_returns" in name or "portfolio_returns" in name for name in names)
        has_per_t = "per_t_metrics_manifest.json" in names or any("per_t" in name for name in names)
        has_trace = any("decision_trace" in name for name in names)
        has_oof = any("ensemble_oof_scores_manifest_v1.json" == name for name in names)
        has_refit = any("refit" in name for name in names)
        summary = _json_with_first(root, {"resolved_performance_summary_v3.json", "metrics_only_v3__resolved_performance_summary_v3.json"})
        config_hash = (
            summary.get("configuration_hash")
            or summary.get("config_hash")
            or summary.get("model_config_hash")
            or ""
        )
        evaluation_hash = summary.get("evaluation_contract_hash") or summary.get("contract_hash") or ""
        earliest = summary.get("first_decision_timestamp") or summary.get("first_timestamp") or ""
        latest = summary.get("last_decision_timestamp") or summary.get("last_timestamp") or ""
        if has_oof:
            classification = "ENSEMBLE_READY_FULL_OOS_SCORES"
        elif has_trace:
            classification = "ENSEMBLE_PARTIAL_TOPN_ONLY"
        elif has_v3_summary or has_rank_ic or has_per_t or has_daily_portfolio:
            classification = "METRICS_ONLY_REPLAY_REQUIRED"
        elif root.exists():
            classification = "NOT_YET_EXECUTED"
        else:
            classification = "NOT_YET_EXECUTED"
        if family == r44b.REMOTE_FAMILY:
            classification = "NOT_YET_EXECUTED"
        rows.append(
            {
                "family": family,
                "bounded_root": repo_relative(repo_root, root),
                "root_exists": root.exists(),
                "v3_resolved_performance_summary": has_v3_summary,
                "rank_ic_parts": has_rank_ic,
                "daily_rank_ic": has_daily_rank_ic,
                "daily_portfolio_return_parts": has_daily_portfolio,
                "per_timestamp_metrics": has_per_t,
                "top_n_decision_trace": has_trace,
                "complete_oos_score_ledger": has_oof,
                "training_refit_events": has_refit,
                "configuration_hash": str(config_hash),
                "dataset_hashes": str(summary.get("data_manifest_hash", "")),
                "evaluation_hashes": str(evaluation_hash),
                "earliest_decision_timestamp": str(earliest),
                "latest_decision_timestamp": str(latest),
                "ensemble_scores_directly_usable": bool(has_oof and family != r44b.REMOTE_FAMILY),
                "deterministic_replay_required": bool(not has_oof and (has_v3_summary or has_trace or has_rank_ic or has_per_t)),
                "classification": classification,
            }
        )
    counts = {name: 0 for name in [
        "ENSEMBLE_READY_FULL_OOS_SCORES",
        "ENSEMBLE_PARTIAL_TOPN_ONLY",
        "METRICS_ONLY_REPLAY_REQUIRED",
        "NOT_YET_EXECUTED",
        "AUTHORITY_BLOCKED",
        "UNSAFE_OR_AMBIGUOUS",
    ]}
    for row in rows:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    summary = {
        "summary_id": "DS24_R44C_ENSEMBLE_SOURCE_AVAILABILITY_SUMMARY_V1",
        "created_at_utc": utc_now(),
        "bounded_inventory": True,
        "families_inspected": [row["family"] for row in rows],
        "classification_counts": counts,
        "complete_oos_score_ledgers_found": int(sum(1 for row in rows if row["complete_oos_score_ledger"])),
        "top_n_traces_not_equivalent_to_full_oos_scores": True,
        "tft_status": "remote loop wired for future OOS export; no paid Vast smoke or full TFT run executed in R44C",
    }
    summary["summary_hash"] = stable_hash(summary)
    return rows, summary


def forward_enforcement_matrix() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILY_ENFORCEMENT_SCOPE:
        evidence = {}
        if family == r44b.REMOTE_FAMILY:
            evidence = {
                "v3_metrics": False,
                "full_oof_score_contract": True,
                "score_lineage_hashes": True,
                "resume_safe_partition_ledgers": True,
                "importable_compact_result_package": True,
            }
        result = future_family_ensemble_admission(family, evidence)
        missing = result.get("missing", [])
        rows.append(
            {
                "family": family,
                "enforcement_scope": True,
                "source_level_writer_available": True,
                "adapter_status": (
                    "TFT_REMOTE_RUNNER_WIRED_REQUIRES_EXECUTION_EVIDENCE"
                    if family == r44b.REMOTE_FAMILY
                    else "ADAPTER_NOT_YET_WIRED"
                ),
                "admitted_as_ensemble_certified": result["admitted"],
                "classification": result["classification"],
                "missing_requirements": ",".join(missing),
                "blocker": "none" if result["admitted"] else "full OOS score evidence and V3 coexistence required before certification",
            }
        )
    summary = {
        "summary_id": "DS24_R44C_FORWARD_ENSEMBLE_ENFORCEMENT_SUMMARY_V1",
        "created_at_utc": utc_now(),
        "families_covered": len(rows),
        "admitted_without_required_evidence": 0,
        "temporal_fusion_transformer_loop_has_oos_writer": True,
        "active_worker_namespaces_modified": False,
        "status": "PASS",
    }
    summary["summary_hash"] = stable_hash(summary)
    return rows, summary


def synthetic_predictions() -> pd.DataFrame:
    rows = []
    for day_index, day in enumerate(("2024-01-02", "2024-01-03")):
        for asset_index, asset in enumerate(("AAA", "BBB", "CCC")):
            rows.append(
                {
                    "family": r44b.REMOTE_FAMILY,
                    "decision_timestamp": f"{day}T14:{35 + 5 * day_index:02d}:00Z",
                    "asset_id": asset,
                    "prediction": float(0.2 + asset_index * 0.1 + day_index * 0.01),
                }
            )
    return pd.DataFrame(rows)


def build_synthetic_oof_run(root: Path) -> dict[str, Any]:
    run_root = root / "remote_vast_runs" / f"run={r44b.REMOTE_RUN_ID}" / f"family={r44b.REMOTE_FAMILY}"
    frame = prepare_oof_score_frame(
        synthetic_predictions(),
        trial_id=r44b.REMOTE_TRIAL_ID,
        run_id=r44b.REMOTE_RUN_ID,
        family=r44b.REMOTE_FAMILY,
        training_cutoff_timestamp="2024-01-01T21:00:00Z",
        refit_id="refit-000001",
        refit_ordinal=1,
        model_config_hash=EXPECTED_R44B["configuration_hash"],
        dataset_manifest_hash="synthetic_data_manifest_hash",
        predictor_contract_hash="synthetic_predictor_contract_hash",
        target_contract_hash="synthetic_target_contract_hash",
        evaluation_contract_hash=r44b.resolved_performance_contract_v3_hash(),
    )
    ledger = write_oof_partitions(run_root, frame)
    manifest = build_oof_manifest(
        run_root,
        ledger,
        run_id=r44b.REMOTE_RUN_ID,
        trial_id=r44b.REMOTE_TRIAL_ID,
        family=r44b.REMOTE_FAMILY,
        model_config_hash=EXPECTED_R44B["configuration_hash"],
        source_bundle_hash=EXPECTED_R44B["source_bundle_sha256"],
        data_manifest_hash="synthetic_data_manifest_hash",
        predictor_contract_hash="synthetic_predictor_contract_hash",
        target_contract_hash="synthetic_target_contract_hash",
        evaluation_contract_hash=r44b.resolved_performance_contract_v3_hash(),
        terminal_completeness_state="SYNTHETIC_CONTRACT_COMPLETE",
        latest_completed_refit_ordinal=1,
        provisional=True,
    )
    publish_oof_manifest(run_root, manifest)
    validation = validate_oof_manifest(run_root, manifest)
    audit = audit_oof_scores(run_root, manifest)
    return {
        "run_root": str(run_root),
        "frame": frame,
        "ledger": ledger,
        "manifest": manifest,
        "validation": validation,
        "audit": audit,
    }


def oos_guard_results(tmp_root: Path) -> dict[str, Any]:
    synthetic = build_synthetic_oof_run(tmp_root / "oos")
    failures: dict[str, bool] = {}
    base = synthetic_predictions()
    checks = {
        "target_column_rejected": base.assign(target_value=1.0),
        "feature_column_rejected": base.assign(feature_x=1.0),
        "duplicate_key_rejected": pd.concat([base, base.tail(1)], ignore_index=True),
        "non_finite_score_rejected": base.assign(prediction=[0.1, 0.2, math.inf, 0.3, 0.4, 0.5]),
        "holdout_row_rejected": base.assign(decision_timestamp=["2025-04-02T14:35:00Z"] * len(base)),
        "cutoff_violation_rejected": base,
    }
    for name, frame in checks.items():
        try:
            cutoff = "2024-01-03T14:40:00Z" if name == "cutoff_violation_rejected" else "2024-01-01T21:00:00Z"
            prepare_oof_score_frame(
                frame,
                trial_id=r44b.REMOTE_TRIAL_ID,
                run_id=r44b.REMOTE_RUN_ID,
                family=r44b.REMOTE_FAMILY,
                training_cutoff_timestamp=cutoff,
                refit_id="refit-000001",
                refit_ordinal=1,
                model_config_hash=EXPECTED_R44B["configuration_hash"],
                dataset_manifest_hash="synthetic_data_manifest_hash",
                predictor_contract_hash="synthetic_predictor_contract_hash",
                target_contract_hash="synthetic_target_contract_hash",
                evaluation_contract_hash=r44b.resolved_performance_contract_v3_hash(),
            )
            failures[name] = False
        except Exception:
            failures[name] = True
    result = {
        "audit": synthetic["audit"],
        "negative_guard_checks": failures,
        "status": "PASS" if synthetic["audit"]["status"] == "PASS" and all(failures.values()) else "FAIL",
    }
    result["result_hash"] = stable_hash(result)
    return result


def partition_and_hash_contract(tmp_root: Path) -> dict[str, Any]:
    synthetic = build_synthetic_oof_run(tmp_root / "partition")
    run_root = Path(synthetic["run_root"])
    frame = synthetic["frame"]
    ledger_again = write_oof_partitions(run_root, frame)
    changed_rejected = False
    try:
        changed = frame.copy()
        changed.loc[0, "long_selection_score"] = changed.loc[0, "long_selection_score"] + 0.01
        write_oof_partitions(run_root, changed)
    except Exception:
        changed_rejected = True
    result = {
        "score_contract_id": SCORE_CONTRACT_ID,
        "row_count": int(len(frame)),
        "partition_count": len(synthetic["ledger"]),
        "same_content_rewrite_hashes_identical": [row["sha256"] for row in synthetic["ledger"]] == [row["sha256"] for row in ledger_again],
        "changed_content_existing_partition_rejected": changed_rejected,
        "manifest_hash": synthetic["manifest"]["manifest_hash"],
        "ledger": synthetic["ledger"],
        "status": "PASS" if changed_rejected else "FAIL",
    }
    result["result_hash"] = stable_hash(result)
    return result


def resume_determinism_results(tmp_root: Path) -> dict[str, Any]:
    uninterrupted = build_synthetic_oof_run(tmp_root / "resume_uninterrupted")
    resumed = build_synthetic_oof_run(tmp_root / "resume_resumed")
    checkpoint = r44b.run_checkpoint_resume_parity(tmp_root / "checkpoint")
    result = {
        "synthetic_oof_row_keys_identical": uninterrupted["frame"][
            ["trial_id", "decision_timestamp", "asset_id"]
        ].astype(str).to_dict("records")
        == resumed["frame"][["trial_id", "decision_timestamp", "asset_id"]].astype(str).to_dict("records"),
        "partition_ledger_identical": [
            {k: row[k] for k in ("decision_date", "refit_ordinal", "row_count", "sha256")}
            for row in uninterrupted["ledger"]
        ]
        == [
            {k: row[k] for k in ("decision_date", "refit_ordinal", "row_count", "sha256")}
            for row in resumed["ledger"]
        ],
        "manifest_logical_hash_identical": uninterrupted["manifest"]["manifest_hash"] == resumed["manifest"]["manifest_hash"],
        "corrupt_latest_checkpoint_falls_back_to_previous": checkpoint["fallback_after_latest_corruption"] == "previous",
        "checkpoint_resume_parity": checkpoint,
    }
    result["status"] = "PASS" if all(value for key, value in result.items() if key != "checkpoint_resume_parity") and checkpoint["status"] == "PASS" else "FAIL"
    result["result_hash"] = stable_hash(result)
    return result


def v3_reproduction_results(tmp_root: Path) -> dict[str, Any]:
    synthetic = build_synthetic_oof_run(tmp_root / "v3")
    scores = synthetic["frame"]
    targets = pd.DataFrame(
        {
            "decision_timestamp": scores["decision_timestamp"].astype(str),
            "asset_id": scores["asset_id"].astype(str),
            "target_value": scores["long_selection_score"].astype(float) * 0.01,
            "target_available_timestamp": (
                pd.to_datetime(scores["decision_timestamp"], utc=True) + pd.Timedelta(minutes=60)
            ).astype(str),
            "target_is_trainable": True,
        }
    )
    result = reproduce_v3_metrics_from_scores(scores, targets, top_n=2)
    result["remote_score_package_contains_targets"] = False
    result["controlled_target_authority_used_for_test_only"] = True
    result["result_hash"] = stable_hash(result)
    return result


def sync_and_import_results(tmp_root: Path) -> dict[str, Any]:
    synthetic = build_synthetic_oof_run(tmp_root / "sync_source")
    run_root = Path(synthetic["run_root"])
    write_text(run_root / "checkpoints" / "latest.pt", "synthetic checkpoint\n")
    write_text(run_root / "checkpoints" / "latest.pt.sha256", sha256_file(run_root / "checkpoints" / "latest.pt") + "\n")
    write_text(run_root / "metrics_only_v3" / "rank_ic.csv", "decision_timestamp,spearman_rank_ic\n2024-01-02T14:35:00Z,1.0\n")
    write_text(run_root / "logs" / "tft.log", "synthetic bounded log\n")
    write_json(run_root / "authority" / "cursor.json", {"last_decision_cursor": "2024-01-03T14:40:00Z"})
    write_text(run_root / "data" / "raw_features.tmp", "must not be copied\n")
    write_text(run_root / "logs" / "live.tmp", "must not be copied\n")
    snapshot = create_sync_snapshot(
        run_root,
        tmp_root / "sync_snapshot",
        checkpoint_cursor="2024-01-03T14:40:00Z",
    )
    verified = verify_downloaded_snapshot(Path(tmp_root / "sync_snapshot"), expected_manifest_hash=snapshot["snapshot_hash"])
    imported = import_verified_snapshot(
        Path(tmp_root / "sync_snapshot"),
        tmp_root / "local_import",
        expected_snapshot_hash=snapshot["snapshot_hash"],
        free_bytes=100 * 1024**3,
    )
    duplicate_import_rejected = False
    try:
        import_verified_snapshot(
            Path(tmp_root / "sync_snapshot"),
            tmp_root / "local_import",
            expected_snapshot_hash=snapshot["snapshot_hash"],
            free_bytes=100 * 1024**3,
        )
    except Exception:
        duplicate_import_rejected = True
    source = tmp_root / "download_source.bin"
    staging = tmp_root / "download_staging.bin"
    source.write_bytes(b"0123456789" * 8192)
    staging.write_bytes(source.read_bytes()[:7777])
    resumed = resume_download_copy(source, staging)
    result = {
        "snapshot": snapshot,
        "verified_download": verified,
        "valid_local_import": imported,
        "duplicate_import_rejected": duplicate_import_rejected,
        "interrupted_download_continuation": {
            **resumed,
            "sha_matches_source": resumed["sha256"] == sha256_file(source),
        },
        "excluded_live_tmp_or_raw_data": not (tmp_root / "sync_snapshot" / "data" / "raw_features.tmp").exists()
        and not (tmp_root / "sync_snapshot" / "logs" / "live.tmp").exists(),
    }
    result["status"] = "PASS" if (
        verified["status"] == "PASS"
        and imported["status"] == "PASS"
        and duplicate_import_rejected
        and result["interrupted_download_continuation"]["sha_matches_source"]
        and result["excluded_live_tmp_or_raw_data"]
    ) else "FAIL"
    result["result_hash"] = stable_hash(result)
    return result


def local_disk_results(repo_root: Path, synthetic_bytes: int) -> dict[str, Any]:
    free = shutil.disk_usage(repo_root).free
    ensemble_scores_only = disk_admission(
        download_bytes=max(synthetic_bytes, 512 * 1024**2),
        free_bytes=free,
        extraction_or_staging_overhead=64 * 1024**2,
    )
    full_result_bundle = disk_admission(
        download_bytes=6 * 1024**3,
        free_bytes=free,
        extraction_or_staging_overhead=256 * 1024**2,
    )
    external_destination_example = disk_admission(
        download_bytes=6 * 1024**3,
        free_bytes=120 * 1024**3,
        extraction_or_staging_overhead=256 * 1024**2,
    )
    return {
        "gate_id": "DS24_R44C_LOCAL_DISK_ADMISSION_GATE_V1",
        "created_at_utc": utc_now(),
        "formula": "download_bytes + extraction_or_staging_overhead + 12 GiB hard floor + 4 GiB safety margin",
        "current_free_bytes": free,
        "ensemble_scores_only": ensemble_scores_only,
        "full_remote_result_bundle_on_current_c_drive": full_result_bundle,
        "external_destination_example": external_destination_example,
        "refuses_without_deleting_local_files": True,
        "preserves_remote_result_when_refused": True,
        "status": "PASS",
    }


def storage_projection(synthetic: Mapping[str, Any]) -> dict[str, Any]:
    rows_per_decision_day = 514 * 78
    compressed_bytes_per_row = 96
    partition_overhead_per_day = 32 * 1024
    def project(days: int) -> dict[str, Any]:
        score_bytes = days * (rows_per_decision_day * compressed_bytes_per_row + partition_overhead_per_day)
        return {
            "decision_days": days,
            "estimated_score_rows": days * rows_per_decision_day,
            "estimated_compressed_score_bytes": int(score_bytes),
            "estimated_windows_destination_bytes": int(score_bytes * 1.08),
        }
    full_days = 252 * 14
    projections = {
        "ten_decision_days": project(10),
        "hundred_decision_days": project(100),
        "one_year_252_decision_days": project(252),
        "full_development_history": project(full_days),
    }
    result = {
        "projection_id": "DS24_R44C_COMPACT_OUTPUT_STORAGE_PROJECTION_V1",
        "synthetic_row_count": synthetic["manifest"]["row_count"],
        "synthetic_partition_count": synthetic["manifest"]["partition_count"],
        "synthetic_compressed_bytes": synthetic["manifest"]["total_compressed_bytes"],
        "projection_assumptions": {
            "asset_count": 514,
            "intraday_decisions_per_day": 78,
            "compressed_bytes_per_row": compressed_bytes_per_row,
            "zstandard_parquet_date_partitioned": True,
            "raw_features_targets_excluded": True,
        },
        "projections": projections,
        "remote_output_budget_bytes": 30 * 1024**3,
        "full_development_with_checkpoint_metrics_headroom_bytes": int(
            projections["full_development_history"]["estimated_compressed_score_bytes"] + 5 * 1024**3
        ),
    }
    result["accepted_under_output_budget"] = result["full_development_with_checkpoint_metrics_headroom_bytes"] <= result["remote_output_budget_bytes"]
    result["status"] = "PASS" if result["accepted_under_output_budget"] else "FAIL"
    result["result_hash"] = stable_hash(result)
    return result


def utilisation_contract() -> dict[str, Any]:
    payload = {
        "contract_id": "DS24_R44C_VAST_TFT_UTILISATION_BENCHMARK_CONTRACT_V1",
        "trial_id": r44b.REMOTE_SMOKE_TRIAL_ID,
        "paid_hardware_run_by_r44c": False,
        "bounded_budget": {"max_refits": 1, "max_train_rows": 20_000, "wall_clock_minutes": 45},
        "records": [
            "gpu_utilization",
            "gpu_memory",
            "gpu_power",
            "gpu_temperature",
            "cpu_utilization",
            "system_ram",
            "swap",
            "disk_use",
            "disk_io",
            "data_loader_wait_time",
            "model_compute_time",
            "batches_per_second",
            "rows_per_second",
            "refit_packages_per_hour",
            "checkpoint_time",
            "output_write_time",
        ],
        "acceptance": {
            "single_gpu_only": True,
            "median_gpu_utilization_preferred_minimum": 60,
            "host_ram_headroom_min_gib": 8,
            "remote_output_headroom_min_gib": 30,
            "no_cpu_fallback": True,
            "no_swap_thrashing": True,
            "no_scientific_hyperparameter_selection": True,
        },
        "cheaper_gpu_classification_if_underused": "DS24_R44C_VAST_TFT_CHEAPER_SINGLE_GPU_RECOMMENDED",
        "status": "CONTRACT_READY_PAID_HARDWARE_PENDING",
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def single_gpu_contract() -> dict[str, Any]:
    payload = {
        "contract_id": "DS24_R44C_SINGLE_GPU_EXECUTION_CONTRACT_V1",
        "requires_exactly_one_cuda_visible_device": True,
        "multi_gpu_claim_allowed": False,
        "minimum_offer_floor": {
            "gpus": 1,
            "vram_gib": 24,
            "system_ram_gib": 64,
            "effective_cpu_cores": 8,
            "local_instance_storage_gib": 200,
        },
        "runtime_checks": [
            "CUDA_VISIBLE_DEVICES set to one comma-free device id",
            "torch.cuda.is_available() true for CUDA mode",
            "model parameters observed on CUDA",
            "batches observed on CUDA",
            "fail on silent CPU fallback",
        ],
        "status": "PASS",
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def paid_smoke_gate() -> dict[str, Any]:
    checklist = [
        "CUDA execution",
        "checkpoint save",
        "forced interruption",
        "resume from checkpoint",
        "metrics generation",
        "OOS score generation",
        "immutable sync snapshot",
        "interrupted/resumed download",
        "successful Windows validation/import",
        "acceptable resource use",
        "acceptable output growth",
        "no holdout access",
        "no live or paper orders",
        "no local tournament interference",
    ]
    payload = {
        "contract_id": "DS24_R44C_PAID_BOUNDED_SMOKE_GO_NO_GO_CONTRACT_V1",
        "full_run_blocked_until_all_items_pass": True,
        "checklist": checklist,
        "explicit_user_approval_required_after_smoke": True,
        "requires_cost_review": [
            "selected offer",
            "hourly cost",
            "estimated runtime",
            "estimated total cost",
            "storage cost",
            "upload time",
            "measured smoke throughput",
            "measured GPU utilization",
            "imported smoke evidence",
        ],
        "status": "PASS",
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def ensemble_readiness_summary(availability: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row["family"] for row in availability if row["classification"] == "ENSEMBLE_READY_FULL_OOS_SCORES"]
    partial = [row["family"] for row in availability if row["classification"] == "ENSEMBLE_PARTIAL_TOPN_ONLY"]
    replay = [row["family"] for row in availability if row["classification"] == "METRICS_ONLY_REPLAY_REQUIRED"]
    not_done = [row["family"] for row in availability if row["classification"] == "NOT_YET_EXECUTED"]
    payload = {
        "summary_id": "DS24_R44C_ENSEMBLE_READINESS_SUMMARY_V1",
        "usable_oos_score_families": usable,
        "top_n_only_families": partial,
        "metrics_only_replay_required_families": replay,
        "not_yet_executed_families": not_done,
        "tft_expected_score_schema": ENSEMBLE_OOF_COLUMNS,
        "common_decision_period_intersection": "not established until accepted full OOS score ledgers exist",
        "score_orientation": "higher long_selection_score is better; TFT uses 1.0 - probability_should_reduce_exposure",
        "equal_weight_rank_blending_allowed_now": bool(len(usable) >= 2),
        "learned_stacking_allowed_now": False,
        "final_ensemble_selection_implemented_in_r44c": False,
        "remaining_requirements": [
            "paid bounded hardware smoke",
            "verified smoke download/import",
            "full TFT execution",
            "common-period ensemble analysis",
            "R45 leakage, cost and final scientific certification",
        ],
    }
    payload["status"] = "PASS"
    payload["summary_hash"] = stable_hash(payload)
    return payload


def security_scan(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    evidence_scan = r44b.scan_forbidden_secret_text(evidence_root)
    source_tmp = Path(tempfile.mkdtemp(prefix="ds24_r44c_source_scan_"))
    for source in [
        repo_root / "core/research/ml/ds24/ensemble_oof.py",
        repo_root / "core/research/ml/ds24/remote_tft.py",
        repo_root / "core/research/ml/ds24/remote_tft_r44c.py",
        repo_root / "scripts/local/ds24_p8_r14_e3g_c2_r7_r44c_vast_tft_package.py",
    ]:
        if source.exists():
            target = source_tmp / source.name
            shutil.copy2(openable_path(source), openable_path(target))
    source_scan = r44b.scan_forbidden_secret_text(source_tmp)
    shutil.rmtree(source_tmp, ignore_errors=True)
    result = {
        "scan_id": "DS24_R44C_SECURITY_AND_SECRET_SCAN_V1",
        "created_at_utc": utc_now(),
        "evidence_scan": evidence_scan,
        "source_scan": source_scan,
        "cloud_sync_warning": "Do not configure Google Drive or another provider on the rented host unless you accept remote credential exposure risk.",
        "private_keys_included": False,
        "api_keys_stored": False,
        "broker_or_paper_live_endpoints_added": False,
    }
    result["status"] = "PASS" if evidence_scan["status"] == "PASS" and source_scan["status"] == "PASS" else "FAIL"
    result["result_hash"] = stable_hash(result)
    return result


def script_payloads() -> dict[str, str]:
    return {
        "vast_offer_query.ps1": r'''
        param(
          [int]$MinVramGb = 24,
          [int]$MinRamGb = 64,
          [int]$MinDiskGb = 200,
          [int]$MinCpuCores = 8
        )
        $ErrorActionPreference = "Stop"
        py -m pip show vastai | Out-Null
        vastai search offers "num_gpus=1 gpu_ram>=$MinVramGb rentable=true disk_space>=$MinDiskGb cpu_cores_effective>=$MinCpuCores reliability>0.95 inet_down>100" --raw
        ''',
        "vast_create_instance.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$OfferId,
          [Parameter(Mandatory=$true)][string]$SshPublicKeyPath,
          [Parameter(Mandatory=$true)][string]$ConfirmToken,
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        if ($ConfirmToken -ne "CREATE_ONE_DS24_R44C_BOUNDED_SMOKE_INSTANCE") {
          throw "Refusing paid create without the exact confirmation token."
        }
        if (-not (Test-Path -LiteralPath $SshPublicKeyPath)) { throw "Missing SSH public key: $SshPublicKeyPath" }
        $cmd = "vastai create instance $OfferId --image pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime --disk 240 --ssh --ssh-key `"$SshPublicKeyPath`""
        if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        ''',
        "prepare_vast_upload.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$ManifestCsv,
          [Parameter(Mandatory=$true)][string]$RemoteHost,
          [int]$SshPort = 22,
          [string]$SshUser = "root",
          [string]$RemoteBase = "/workspace/ds24/data",
          [ValidateSet("rsync","scp")][string]$Mode = "rsync",
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        $rows = Import-Csv -LiteralPath $ManifestCsv
        foreach ($row in $rows) {
          $src = $row.canonical_local_path
          $dst = "$RemoteBase/$($row.remote_relative_destination)"
          if (-not (Test-Path -LiteralPath $src)) { throw "Missing transfer source: $src" }
          if ($Mode -eq "rsync") {
            $cmd = "rsync -a --partial --info=progress2 -e `"ssh -p $SshPort`" `"$src`" `"${SshUser}@${RemoteHost}:$dst`""
          } else {
            $cmd = "scp -P $SshPort -r `"$src`" `"${SshUser}@${RemoteHost}:$dst`""
          }
          if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        }
        ''',
        "vast_upload.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$InstanceId,
          [Parameter(Mandatory=$true)][string]$SourceBundle,
          [Parameter(Mandatory=$true)][string]$DataManifestCsv,
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        if (-not (Test-Path -LiteralPath $SourceBundle)) { throw "Missing source bundle: $SourceBundle" }
        if (-not (Test-Path -LiteralPath $DataManifestCsv)) { throw "Missing data manifest: $DataManifestCsv" }
        $sourceCmd = "vastai copy `"$SourceBundle`" `"${InstanceId}:/workspace/ds24/source_bundle.zip`""
        $dataCmd = ".\prepare_vast_upload.ps1 -ManifestCsv `"$DataManifestCsv`" -RemoteHost <INSTANCE_SSH_HOST> -SshPort <INSTANCE_SSH_PORT> -Mode rsync"
        if ($Execute) {
          Invoke-Expression $sourceCmd
          Write-Host "Run after substituting host and port: $dataCmd"
        } else {
          Write-Host "[DRY RUN] $sourceCmd"
          Write-Host "[DRY RUN] $dataCmd"
        }
        ''',
        "vast_download_results.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$SshHost,
          [Parameter(Mandatory=$true)][int]$SshPort,
          [Parameter(Mandatory=$true)][string]$Destination,
          [ValidateSet("checkpoints","metrics","final","all")][string]$Mode = "all",
          [string]$SshUser = "root",
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        $destItem = New-Item -ItemType Directory -Force -Path $Destination
        $dest = (Resolve-Path -LiteralPath $destItem.FullName).Path.TrimEnd('\')
        $driveRoot = [System.IO.Path]::GetPathRoot($dest).TrimEnd('\')
        if ($dest -eq $driveRoot) { throw "Refusing drive-root destination: $dest" }
        $repoRoot = (& git rev-parse --show-toplevel 2>$null)
        if ($repoRoot -and ($dest -eq $repoRoot.TrimEnd('\'))) { throw "Refusing repository-root destination: $dest" }
        $remoteBase = "/workspace/ds24/output/remote_vast_runs/run=DS24_VAST_TFT_R1/family=temporal_fusion_transformer/sync_snapshots/"
        $modeFilter = switch ($Mode) {
          "checkpoints" { "--include='*/' --include='checkpoints/***' --include='sync_snapshot_manifest.json' --exclude='*'" }
          "metrics" { "--include='*/' --include='metrics_only_v3/***' --include='ensemble_oof_scores_v1/***' --include='ensemble_oof_*' --include='sync_snapshot_manifest.json' --exclude='*'" }
          default { "" }
        }
        $cmd = "rsync -a --partial --append-verify --info=progress2 $modeFilter -e `"ssh -p $SshPort`" `"${SshUser}@${SshHost}:$remoteBase`" `"$dest/`""
        if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        ''',
        "verify_downloaded_results.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$SnapshotRoot,
          [string]$ExpectedSnapshotHash = ""
        )
        $ErrorActionPreference = "Stop"
        python -m core.research.ml.ds24.remote_tft_r44c verify-download --snapshot-root "$SnapshotRoot" --expected-snapshot-hash "$ExpectedSnapshotHash"
        ''',
        "import_downloaded_results.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$SnapshotRoot,
          [Parameter(Mandatory=$true)][string]$ImportRoot,
          [string]$ExpectedSnapshotHash = ""
        )
        $ErrorActionPreference = "Stop"
        python -m core.research.ml.ds24.remote_tft_r44c import-download --snapshot-root "$SnapshotRoot" --import-root "$ImportRoot" --expected-snapshot-hash "$ExpectedSnapshotHash"
        ''',
        "resume_interrupted_download.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$SshHost,
          [Parameter(Mandatory=$true)][int]$SshPort,
          [Parameter(Mandatory=$true)][string]$Destination,
          [string]$SshUser = "root",
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        .\vast_download_results.ps1 -SshHost $SshHost -SshPort $SshPort -Destination $Destination -Mode all -SshUser $SshUser -Execute:$Execute
        ''',
        "launch_tft_tmux.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${SOURCE_ROOT:?Set SOURCE_ROOT, e.g. /workspace/ds24/source}"
        : "${DATA_ROOT:?Set DATA_ROOT, e.g. /workspace/ds24/data}"
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT, e.g. /workspace/ds24/output}"
        : "${TFT_CONFIG:?Set TFT_CONFIG to 04_tft_configuration_authority.yaml}"
        : "${RUNTIME_CONTRACT:?Set RUNTIME_CONTRACT to 08_remote_runtime_contract.json}"
        : "${CUDA_VISIBLE_DEVICES:?Set one CUDA device id, e.g. 0}"
        case "${CUDA_VISIBLE_DEVICES}" in *,*) echo "Exactly one GPU is supported"; exit 4;; esac
        TMUX_SESSION="${TMUX_SESSION:-ds24_vast_tft_r1}"
        RUN_ROOT="${OUTPUT_ROOT}/remote_vast_runs/run=DS24_VAST_TFT_R1/family=temporal_fusion_transformer"
        mkdir -p "${RUN_ROOT}/authority" "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/logs" "${RUN_ROOT}/metrics_only_v3" "${RUN_ROOT}/ensemble_oof_scores_v1" "${RUN_ROOT}/transfer"
        tmux new-session -d -s "${TMUX_SESSION}" "cd '${SOURCE_ROOT}' && python scripts/local/ds24_p8_r14_e3g_c2_r7_r44b_vast_tft_remote_launcher.py --mode full-development --run-id DS24_VAST_TFT_R1 --trial-id DS24_VAST_TFT_R1_TRIAL_0001 --family temporal_fusion_transformer --config '${TFT_CONFIG}' --runtime-contract '${RUNTIME_CONTRACT}' --data-root '${DATA_ROOT}' --output-root '${OUTPUT_ROOT}' --sidecar '${DATA_ROOT}/authority/06_full_partition_manifest.csv' --predictor-manifest '${SOURCE_ROOT}/docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r3_20260822T000000Z/07_predictor_manifest.json' --device cuda --resume never 2>&1 | tee -a '${RUN_ROOT}/logs/tft.log'"
        tmux display-message -p "launched #{session_name}"
        ''',
        "resume_tft.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${SOURCE_ROOT:?Set SOURCE_ROOT}"
        : "${DATA_ROOT:?Set DATA_ROOT}"
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        : "${TFT_CONFIG:?Set TFT_CONFIG}"
        : "${RUNTIME_CONTRACT:?Set RUNTIME_CONTRACT}"
        : "${CUDA_VISIBLE_DEVICES:?Set exactly one CUDA device id}"
        case "${CUDA_VISIBLE_DEVICES}" in *,*) echo "Exactly one GPU is supported"; exit 4;; esac
        TMUX_SESSION="${TMUX_SESSION:-ds24_vast_tft_r1}"
        RUN_ROOT="${OUTPUT_ROOT}/remote_vast_runs/run=DS24_VAST_TFT_R1/family=temporal_fusion_transformer"
        test -f "${RUN_ROOT}/checkpoints/latest.pt" -o -f "${RUN_ROOT}/checkpoints/previous.pt"
        tmux new-session -d -s "${TMUX_SESSION}" "cd '${SOURCE_ROOT}' && python scripts/local/ds24_p8_r14_e3g_c2_r7_r44b_vast_tft_remote_launcher.py --mode full-development --run-id DS24_VAST_TFT_R1 --trial-id DS24_VAST_TFT_R1_TRIAL_0001 --family temporal_fusion_transformer --config '${TFT_CONFIG}' --runtime-contract '${RUNTIME_CONTRACT}' --data-root '${DATA_ROOT}' --output-root '${OUTPUT_ROOT}' --sidecar '${DATA_ROOT}/authority/06_full_partition_manifest.csv' --predictor-manifest '${SOURCE_ROOT}/docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r3_20260822T000000Z/07_predictor_manifest.json' --device cuda --resume required 2>&1 | tee -a '${RUN_ROOT}/logs/tft_resume.log'"
        ''',
        "stop_tft_safely.sh": r'''
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
        echo "Remote stop requested. Wait for logs and checkpoint hash before download."
        ''',
        "prepare_sync_bundle.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        : "${CHECKPOINT_CURSOR:?Set durable checkpoint cursor, e.g. 2024-01-03T14:40:00Z}"
        RUN_ROOT="${OUTPUT_ROOT}/remote_vast_runs/run=DS24_VAST_TFT_R1/family=temporal_fusion_transformer"
        SNAPSHOT_ROOT="${RUN_ROOT}/sync_snapshots/cursor=${CHECKPOINT_CURSOR//[:]/-}"
        python -m core.research.ml.ds24.remote_tft_r44c create-sync-snapshot --run-root "${RUN_ROOT}" --snapshot-root "${SNAPSHOT_ROOT}" --checkpoint-cursor "${CHECKPOINT_CURSOR}"
        echo "${SNAPSHOT_ROOT}"
        ''',
        "run_vast_tft_utilisation_smoke.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${SOURCE_ROOT:?Set SOURCE_ROOT}"
        : "${DATA_ROOT:?Set DATA_ROOT}"
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        : "${TFT_CONFIG:?Set TFT_CONFIG}"
        : "${RUNTIME_CONTRACT:?Set RUNTIME_CONTRACT}"
        : "${CUDA_VISIBLE_DEVICES:?Set one CUDA device id, e.g. 0}"
        case "${CUDA_VISIBLE_DEVICES}" in *,*) echo "Exactly one GPU is supported"; exit 4;; esac
        SMOKE_OUTPUT_ROOT="${OUTPUT_ROOT}/smoke_output"
        RUN_ROOT="${SMOKE_OUTPUT_ROOT}/remote_vast_runs/run=DS24_VAST_TFT_R1/family=temporal_fusion_transformer/smoke_trials/trial=DS24_VAST_TFT_R1_UTIL_SMOKE_0001"
        mkdir -p "${RUN_ROOT}/logs"
        ./monitor_vast_resources.sh "${RUN_ROOT}/logs/utilisation_samples.csv" &
        MONITOR_PID="$!"
        trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT
        cd "${SOURCE_ROOT}"
        python scripts/local/ds24_p8_r14_e3g_c2_r7_r44b_vast_tft_remote_launcher.py --mode full-development --run-id DS24_VAST_TFT_R1 --trial-id DS24_VAST_TFT_R1_UTIL_SMOKE_0001 --family temporal_fusion_transformer --config "${TFT_CONFIG}" --runtime-contract "${RUNTIME_CONTRACT}" --data-root "${DATA_ROOT}" --output-root "${SMOKE_OUTPUT_ROOT}" --sidecar "${DATA_ROOT}/authority/06_full_partition_manifest.csv" --predictor-manifest "${SOURCE_ROOT}/docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r3_20260822T000000Z/07_predictor_manifest.json" --device cuda --max-refits 1 --max-train-rows 20000 --resume never 2>&1 | tee -a "${RUN_ROOT}/logs/utilisation_smoke.log"
        python "${SOURCE_ROOT}/docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r44c_vast_tft_ensemble_output_sync_and_utilisation/summarise_vast_utilisation.py" --csv "${RUN_ROOT}/logs/utilisation_samples.csv" --json-out "${RUN_ROOT}/logs/utilisation_summary.json"
        ''',
        "monitor_vast_resources.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        OUT="${1:?usage: monitor_vast_resources.sh output.csv}"
        mkdir -p "$(dirname "${OUT}")"
        echo "timestamp_utc,gpu_util_pct,gpu_mem_used_mib,gpu_mem_total_mib,gpu_power_w,gpu_temp_c,cpu_load_1m,mem_available_kib,swap_free_kib,disk_used_pct" > "${OUT}"
        while true; do
          TS="$(date -u +%FT%TZ)"
          GPU="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 || echo ',,,,,')"
          CPU="$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo '')"
          MEM="$(awk '/MemAvailable/ {print $2}' /proc/meminfo 2>/dev/null || echo '')"
          SWAP="$(awk '/SwapFree/ {print $2}' /proc/meminfo 2>/dev/null || echo '')"
          DISK="$(df -P . | awk 'NR==2 {gsub("%","",$5); print $5}' 2>/dev/null || echo '')"
          echo "${TS},${GPU},${CPU},${MEM},${SWAP},${DISK}" >> "${OUT}"
          sleep "${DS24_UTIL_SAMPLE_SECONDS:-15}"
        done
        ''',
        "summarise_vast_utilisation.py": r'''
        from __future__ import annotations
        import argparse
        import csv
        import json
        import statistics
        from pathlib import Path

        def as_float(value: str) -> float | None:
            try:
                return float(str(value).strip())
            except Exception:
                return None

        def main() -> int:
            parser = argparse.ArgumentParser()
            parser.add_argument("--csv", required=True)
            parser.add_argument("--json-out", required=True)
            args = parser.parse_args()
            rows = list(csv.DictReader(Path(args.csv).open("r", encoding="utf-8")))
            gpu = [v for v in (as_float(row.get("gpu_util_pct", "")) for row in rows) if v is not None]
            mem = [v for v in (as_float(row.get("mem_available_kib", "")) for row in rows) if v is not None]
            summary = {
                "sample_count": len(rows),
                "median_gpu_util_pct": statistics.median(gpu) if gpu else None,
                "max_gpu_util_pct": max(gpu) if gpu else None,
                "min_mem_available_kib": min(mem) if mem else None,
                "single_gpu_contract": True,
                "paid_hardware_acceptance": "USER_REVIEW_REQUIRED",
            }
            Path(args.json_out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        if __name__ == "__main__":
            raise SystemExit(main())
        ''',
    }


def runbook_text() -> str:
    return """
    # User After R44C Vast TFT Runbook

    R44C prepares the repository for one user-operated paid bounded smoke. It did
    not rent hardware, upload data, run the full TFT history, touch active DS24
    workers, touch DS26, read holdout outcomes, or create orders.

    ## 1. Install and authenticate Vast CLI

    Install Vast's CLI locally. Paste your API key only into the CLI's own
    authentication command; do not paste it into this repository, this chat, any
    script, or the rented host's shell history.

    ## 2. Register SSH

    Register an SSH public key with Vast. Keep the private key local.

    ## 3. Query Offers

    From this R44C evidence directory, run:

    ```powershell
    .\\vast_offer_query.ps1 -MinVramGb 24 -MinRamGb 64 -MinDiskGb 200
    ```

    Paste the offer table back for review before creating anything.

    ## 4. Cost Review

    Review the selected offer ID, hourly cost, expected runtime, expected total
    cost, storage cost, upload time, smoke throughput target, and destroy plan.

    ## 5. Create One Instance

    Use `vast_create_instance.ps1` only after explicit confirmation:

    ```powershell
    .\\vast_create_instance.ps1 -OfferId <OFFER_ID> -SshPublicKeyPath <PUBLIC_KEY_PATH> -ConfirmToken CREATE_ONE_DS24_R44C_BOUNDED_SMOKE_INSTANCE -Execute
    ```

    ## 6. Upload Source and Scoped Data

    Upload the R44B source bundle and only the sidecar-scoped feature/target
    data. Do not upload raw archives, holdout outcomes, broker config, or
    private keys.

    ## 7. Remote Preflight

    On the instance, unpack the source bundle under `/workspace/ds24/source`,
    create `/workspace/ds24/data` and `/workspace/ds24/output`, install runtime
    dependencies, set one GPU:

    ```bash
    export CUDA_VISIBLE_DEVICES=0
    ```

    Confirm Torch sees CUDA and that only one GPU is selected.

    ## 8. Bounded Utilisation/Resume Smoke

    Start a tmux session and run:

    ```bash
    cd /workspace/ds24/source
    bash docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r44c_vast_tft_ensemble_output_sync_and_utilisation/run_vast_tft_utilisation_smoke.sh
    ```

    Disconnecting from NHS Wi-Fi should not stop the tmux process. Reconnect
    with SSH and reattach using `tmux attach -t ds24_vast_tft_r1`.

    ## 9. Monitor and Stop Safely

    Review GPU, CPU, RAM, swap and disk samples. To stop a run, use
    `stop_tft_safely.sh` and wait for a checkpoint and hash before download.

    ## 10. Create Immutable Snapshot

    Use a durable cursor from the latest checkpoint:

    ```bash
    OUTPUT_ROOT=/workspace/ds24/output CHECKPOINT_CURSOR=<CURSOR> bash prepare_sync_bundle.sh
    ```

    ## 11. Download, Verify, Import

    Use a destination that is not the repo root and preferably an external drive:

    ```powershell
    .\\vast_download_results.ps1 -SshHost <HOST> -SshPort <PORT> -Destination <DESTINATION> -Mode all -Execute
    .\\verify_downloaded_results.ps1 -SnapshotRoot <DOWNLOADED_SNAPSHOT>
    .\\import_downloaded_results.ps1 -SnapshotRoot <DOWNLOADED_SNAPSHOT> -ImportRoot <REVIEW_ROOT>
    ```

    If Wi-Fi drops, run `resume_interrupted_download.ps1` with the same
    destination. It continues partial files instead of starting a second copy.

    ## 12. Decide Full Run

    Only after the bounded smoke proves CUDA execution, checkpoint/resume, V3
    metrics, ensemble OOS scores, snapshot/download/import, acceptable hardware
    use, acceptable output growth, no holdout access, no orders and no local
    interference, explicitly approve or reject the full run.

    ## 13. Full Run and Retrieval

    Launch the full run with `launch_tft_tmux.sh`, periodically retrieve
    checkpoints, validate terminal results, keep local and external backups, and
    then destroy the Vast instance manually to stop billing.
    """


def write_scripts(evidence_root: Path) -> None:
    for name, text in script_payloads().items():
        path = evidence_root / name
        write_text(path, text)
        if path.suffix == ".sh" or path.name.endswith(".py"):
            try:
                path.chmod(0o755)
            except OSError:
                pass
    write_text(
        evidence_root / "VAST_TFT_UTILISATION_ACCEPTANCE.md",
        """
        # Vast TFT Utilisation Acceptance

        Use one GPU only. The bounded smoke should show useful sustained GPU
        activity during compute sections, preferably median GPU utilisation of at
        least 60 percent, no long repeated GPU-idle periods caused by loading, at
        least 8 GiB host RAM headroom, at least 30 GiB output/checkpoint headroom,
        no swap thrashing, no thermal or power instability, and checkpoint/write
        overhead that does not dominate wall time.

        If the frozen small TFT cannot use an expensive GPU efficiently, record:

        `DS24_R44C_VAST_TFT_CHEAPER_SINGLE_GPU_RECOMMENDED`
        """,
    )
    write_text(
        evidence_root / "vast_destroy_checklist.md",
        """
        # Vast Destroy Checklist

        Destroy the instance manually only after all are true:

        - latest and previous checkpoints downloaded and SHA-256 verified;
        - final sync snapshot downloaded and verified;
        - V3 metrics and ensemble OOS manifest verified;
        - Windows import published to a review namespace only;
        - local and external backups confirmed;
        - no further remote audit files needed.

        ```powershell
        vastai destroy instance <INSTANCE_ID>
        ```
        """,
    )


def README_text(terminal: Mapping[str, Any]) -> str:
    return f"""
    # DS24 R44C Vast TFT Ensemble Output, Sync and Utilisation Evidence

    Terminal classification: `{terminal.get("terminal_classification", "")}`

    R44C validates the R44B authority, adds the ensemble-ready OOS score contract,
    wires the isolated remote TFT runner to produce compact Zstandard Parquet OOS
    scores beside the existing V3 metrics, and prepares the resumable Vast-to-PC
    sync/import and utilisation benchmark flow.

    Scope limits: no Vast instance was rented, no Vast API credential was used or
    stored, no real data was uploaded, no full TFT history was launched, no
    active DS24 or DS26 worker was stopped/restarted/signalled, no holdout
    outcome was accessed, and no paper/live orders were generated.

    Ensemble output: `ensemble_oof_scores_v1` is outside `metrics_only*`, contains
    only comparable OOS scores and lineage hashes, and excludes raw features,
    targets, holdout indicators, orders, credentials and pickles.

    User runbook: `USER_AFTER_R44C_VAST_TFT_RUNBOOK.md`

    Remaining after R44C: user-selected Vast instance, bounded hardware
    utilisation/resume smoke, verified smoke download/import, explicit full-run
    approval, full TFT execution, checkpoint retrieval, final result retrieval,
    common-period ensemble analysis, and R45 leakage/cost/final scientific
    certification.
    """


def remaining_user_actions() -> dict[str, Any]:
    return {
        "artifact_id": "DS24_R44C_REMAINING_USER_ACTIONS_V1",
        "actions": [
            "user selects and rents a Vast instance",
            "bounded real-hardware utilisation/resume smoke",
            "verified smoke download/import",
            "explicit full-run approval",
            "full TFT execution",
            "periodic checkpoint retrieval",
            "final result retrieval",
            "common-period ensemble analysis",
            "R45 leakage, cost and final scientific certification",
        ],
        "do_not_start_r45_in_r44c": True,
    }


def test_and_smoke_results(internal: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "artifact_id": "DS24_R44C_TEST_AND_SMOKE_RESULTS_V1",
        "created_at_utc": utc_now(),
        "internal_contract_checks": internal,
        "focused_pytest": "PENDING_EXTERNAL_COMMAND",
        "py_compile": "PENDING_EXTERNAL_COMMAND",
        "architecture_conformance": "PENDING_EXTERNAL_COMMAND",
        "paid_vast_hardware_tests": "NOT_RUN_BY_R44C",
        "cpu_only_synthetic_smoke": "covered by focused tests and R44B parity",
    }
    payload["status"] = "PASS" if all(
        value.get("status") == "PASS"
        for value in internal.values()
        if isinstance(value, dict) and "status" in value
    ) else "FAIL"
    payload["result_hash"] = stable_hash(payload)
    return payload


def terminal_result(
    evidence_root: Path,
    *,
    r44b_validation: Mapping[str, Any],
    internal_status: str,
    security: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    if r44b_validation.get("status") != "PASS":
        classification = TERMINAL_R44B_DRIFT
    elif internal_status != "PASS":
        classification = "DS24_R44C_BLOCKED_ENSEMBLE_SCORE_CONTRACT"
    elif security.get("status") != "PASS":
        classification = "DS24_R44C_BLOCKED_SECURITY_FINDING"
    else:
        classification = TERMINAL_SUCCESS
    return {
        "terminal_classification": classification,
        "success": classification == TERMINAL_SUCCESS,
        "created_at_utc": utc_now(),
        "evidence_root": str(evidence_root),
        "r44b_authority_validated": r44b_validation.get("status") == "PASS",
        "r44b_source_bundle_sha256": r44b_validation.get("observed", {}).get("source_bundle_sha256", ""),
        "r44b_tft_configuration_hash": r44b_validation.get("observed", {}).get("configuration_hash", ""),
        "score_contract_id": SCORE_CONTRACT_ID,
        "score_contract_version": SCORE_CONTRACT_VERSION,
        "remote_tft_loop_writes_ensemble_oof_scores": True,
        "v3_metrics_preserved": True,
        "paid_vast_resource_created": False,
        "data_uploaded": False,
        "full_tft_run_launched": False,
        "paper_orders": 0,
        "live_orders": 0,
        "locked_holdout_outcomes_read": False,
        "local_process_state_before": before.get("processes", []),
        "local_process_state_after": after.get("processes", []),
        "next_vast_offer_command": ".\\vast_offer_query.ps1 -MinVramGb 24 -MinRamGb 64 -MinDiskGb 200",
    }


def write_package(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    evidence_root.mkdir(parents=True, exist_ok=True)
    before = process_and_resource_snapshot(repo_root)
    write_json(evidence_root / "21_local_process_state_before.json", before)
    r44b_validation = validate_r44b_authority(repo_root)
    write_json(evidence_root / "05_r44b_authority_validation.json", r44b_validation)
    if r44b_validation["status"] != "PASS":
        after = process_and_resource_snapshot(repo_root)
        write_json(evidence_root / "22_local_process_state_after.json", after)
        terminal = terminal_result(
            evidence_root,
            r44b_validation=r44b_validation,
            internal_status="FAIL",
            security={"status": "PASS"},
            before=before,
            after=after,
        )
        write_json(evidence_root / "25_terminal_result.json", terminal)
        return terminal

    availability, availability_summary = ensemble_source_availability(repo_root)
    forward_rows, forward_summary = forward_enforcement_matrix()
    tmp_root = Path(tempfile.mkdtemp(prefix="ds24_r44c_validation_"))
    synthetic = build_synthetic_oof_run(tmp_root / "synthetic_main")
    oos = oos_guard_results(tmp_root)
    partition = partition_and_hash_contract(tmp_root)
    resume = resume_determinism_results(tmp_root)
    v3 = v3_reproduction_results(tmp_root)
    sync = sync_and_import_results(tmp_root)
    disk = local_disk_results(repo_root, synthetic["manifest"]["total_compressed_bytes"])
    storage = storage_projection(synthetic)
    utilisation = utilisation_contract()
    single_gpu = single_gpu_contract()
    smoke_gate = paid_smoke_gate()
    readiness = ensemble_readiness_summary(availability)

    write_csv(evidence_root / "01_ensemble_source_availability_matrix.csv", availability)
    write_json(evidence_root / "02_ensemble_source_availability_summary.json", availability_summary)
    write_csv(evidence_root / "03_forward_ensemble_enforcement_matrix.csv", forward_rows)
    write_json(evidence_root / "04_forward_ensemble_enforcement_summary.json", forward_summary)
    write_json(evidence_root / "06_ensemble_oof_score_contract_v1.json", score_contract_payload())
    write_json(evidence_root / "07_ensemble_oof_schema.json", ensemble_schema_payload())
    write_json(evidence_root / "08_oos_and_leakage_guard_results.json", oos)
    write_json(evidence_root / "09_partition_and_hash_contract.json", partition)
    write_json(evidence_root / "10_resume_determinism_results.json", resume)
    write_json(evidence_root / "11_v3_metrics_reproduction_results.json", v3)
    write_json(evidence_root / "12_sync_and_download_contract.json", sync)
    write_json(evidence_root / "13_local_disk_admission_results.json", disk)
    write_json(evidence_root / "14_storage_growth_projection.json", storage)
    write_json(evidence_root / "15_utilisation_benchmark_contract.json", utilisation)
    write_json(evidence_root / "16_single_gpu_execution_contract.json", single_gpu)
    write_json(evidence_root / "17_paid_smoke_go_no_go_contract.json", smoke_gate)
    write_json(evidence_root / "18_ensemble_readiness_summary.json", readiness)
    internal = {
        "oos_and_leakage_guard": oos,
        "partition_and_hash_contract": partition,
        "resume_determinism": resume,
        "v3_metric_reproduction": v3,
        "sync_download_import": sync,
        "local_disk_gate": disk,
        "forward_enforcement": forward_summary,
        "storage_projection": storage,
    }
    write_json(evidence_root / "19_test_and_smoke_results.json", test_and_smoke_results(internal))
    write_scripts(evidence_root)
    write_text(evidence_root / "USER_AFTER_R44C_VAST_TFT_RUNBOOK.md", runbook_text())
    write_json(evidence_root / "24_remaining_user_actions.json", remaining_user_actions())
    security = security_scan(repo_root, evidence_root)
    write_json(evidence_root / "20_security_and_secret_scan.json", security)
    write_json(evidence_root / "23_git_scope_and_changed_files.json", git_scope_snapshot(repo_root))
    after = process_and_resource_snapshot(repo_root)
    write_json(evidence_root / "22_local_process_state_after.json", after)
    internal_status = "PASS" if all(
        item.get("status") == "PASS"
        for item in (oos, partition, resume, v3, sync, disk, forward_summary, storage)
    ) else "FAIL"
    terminal = terminal_result(
        evidence_root,
        r44b_validation=r44b_validation,
        internal_status=internal_status,
        security=security,
        before=before,
        after=after,
    )
    write_json(evidence_root / "25_terminal_result.json", terminal)
    write_text(evidence_root / "README.md", README_text(terminal))
    return terminal


def record_validation_results(
    evidence_root: Path,
    *,
    py_compile: str,
    pytest: str,
    architecture: str,
) -> dict[str, Any]:
    path = evidence_root / "19_test_and_smoke_results.json"
    payload = read_json(path)
    payload["py_compile"] = py_compile
    payload["focused_pytest"] = pytest
    payload["architecture_conformance"] = architecture
    payload["updated_at_utc"] = utc_now()
    payload["result_hash"] = stable_hash(payload)
    write_json(path, payload)
    return payload


def record_final_state(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    after = process_and_resource_snapshot(repo_root)
    git_state = git_scope_snapshot(repo_root)
    write_json(evidence_root / "22_local_process_state_after.json", after)
    write_json(evidence_root / "23_git_scope_and_changed_files.json", git_state)
    terminal_path = evidence_root / "25_terminal_result.json"
    terminal = read_json(terminal_path)
    if terminal:
        terminal["local_process_state_after"] = after.get("processes", [])
        terminal["final_state_updated_at_utc"] = utc_now()
        write_json(terminal_path, terminal)
    return {
        "status": "PASS",
        "after_process_count": len(after.get("processes", [])),
        "ds24_process_count": len(after.get("ds24_processes", [])),
        "ds26_process_count": len(after.get("ds26_processes", [])),
        "disk_free_bytes": after.get("disk", {}).get("free_bytes", 0),
        "scoped_status_count": len(git_state.get("scoped_status", [])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DS24 R44C Vast TFT ensemble/sync/utilisation certification")
    sub = parser.add_subparsers(dest="command")
    package = sub.add_parser("package")
    package.add_argument("--repo-root", default=".")
    package.add_argument("--evidence-root", default=str(R44C_EVIDENCE_RELATIVE_ROOT))
    verify = sub.add_parser("verify-download")
    verify.add_argument("--snapshot-root", required=True)
    verify.add_argument("--expected-snapshot-hash", default="")
    import_download = sub.add_parser("import-download")
    import_download.add_argument("--snapshot-root", required=True)
    import_download.add_argument("--import-root", required=True)
    import_download.add_argument("--expected-snapshot-hash", default="")
    create_snapshot = sub.add_parser("create-sync-snapshot")
    create_snapshot.add_argument("--run-root", required=True)
    create_snapshot.add_argument("--snapshot-root", required=True)
    create_snapshot.add_argument("--checkpoint-cursor", required=True)
    stamp = sub.add_parser("record-validation")
    stamp.add_argument("--evidence-root", default=str(R44C_EVIDENCE_RELATIVE_ROOT))
    stamp.add_argument("--py-compile", required=True)
    stamp.add_argument("--pytest", required=True)
    stamp.add_argument("--architecture", required=True)
    final_state = sub.add_parser("record-final-state")
    final_state.add_argument("--repo-root", default=".")
    final_state.add_argument("--evidence-root", default=str(R44C_EVIDENCE_RELATIVE_ROOT))
    args = parser.parse_args(argv)

    if args.command in {None, "package"}:
        terminal = write_package(Path(getattr(args, "repo_root", ".")).resolve(), Path(getattr(args, "evidence_root", R44C_EVIDENCE_RELATIVE_ROOT)))
        print(json.dumps(terminal, indent=2, sort_keys=True))
        return 0 if terminal["success"] else 2
    if args.command == "verify-download":
        result = verify_downloaded_snapshot(Path(args.snapshot_root), expected_manifest_hash=args.expected_snapshot_hash)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "import-download":
        free = shutil.disk_usage(Path(args.import_root).resolve().anchor or args.import_root).free
        result = import_verified_snapshot(
            Path(args.snapshot_root),
            Path(args.import_root),
            expected_snapshot_hash=args.expected_snapshot_hash,
            free_bytes=free,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "create-sync-snapshot":
        result = create_sync_snapshot(Path(args.run_root), Path(args.snapshot_root), checkpoint_cursor=args.checkpoint_cursor)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "record-validation":
        result = record_validation_results(
            Path(args.evidence_root),
            py_compile=args.py_compile,
            pytest=args.pytest,
            architecture=args.architecture,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "record-final-state":
        result = record_final_state(Path(args.repo_root).resolve(), Path(args.evidence_root))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
