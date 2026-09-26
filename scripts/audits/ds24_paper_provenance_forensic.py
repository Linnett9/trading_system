from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core.research.ml.ds24.master_5m_validation_stats import compute_stock_features


DS24_ROOT = REPOSITORY_ROOT / "docs/dream_system/components/DS-24_independent_five_minute_selector"
STAGE_ROOT = DS24_ROOT / "stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z"
WORKER_ROOT = STAGE_ROOT / "r7_r14_policy_workers"
PREDICTOR_MANIFEST = (
    DS24_ROOT / "stage_outputs/ds24_p8_r3_20260822T000000Z/07_predictor_manifest.json"
)
FEATURE_IMPLEMENTATION = REPOSITORY_ROOT / "core/research/ml/ds24/master_5m_validation_stats.py"
FEATURE_ROOT = (
    REPOSITORY_ROOT
    / "data/processed/ml_features/five_minute/version=canonical_5m_feature_authority_full_v1"
    / "run=ds24_p8_r2_local_20260821T000000Z"
)
RAW_BAR_ROOT = REPOSITORY_ROOT / "data/processed/alpaca/symbol_bars/sip/5m"
XENDCG_ROOT = (
    REPOSITORY_ROOT
    / "mac_aux_runs/queue=DS24_MAC_AUX_NINE_FAMILY_R1/family=lightgbm_rank_xendcg"
)
XENDCG_AUTHORITY_ROOT = Path(
    r"C:\Users\Brandon\Desktop\ds24-prospective-paper\authority\xendcg"
    r"\ds24_xendcg_producer_authority_r1"
)

TARGET_CONTRACT = "forward_return_60m__decision_5m"
LOCKED_HOLDOUT_START = pd.Timestamp("2025-04-02T00:00:00Z")
TRADING_SESSIONS_PER_YEAR = 252
EXPECTED_FEATURE_ORDER_SHA256 = "22db0fcfe1219a30e0f6926dd8f0af11b4cd8ead2a6462a0df278549fadeb5a0"
EXPECTED_XENDCG_MODEL_COUNT = 2262
EXPECTED_XENDCG_FIRST_ORDINAL = 2
EXPECTED_XENDCG_LAST_ORDINAL = 2263
LEAKED_PREDICTORS = ("overnight_gap", "previous_session_return", "two_session_return")


def _openable_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved.lstrip("\\")
    return "\\\\?\\" + resolved


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected an object in {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(_openable_path(path), "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _path_exists(path: Path) -> bool:
    return os.path.exists(_openable_path(path))


def canonical_hash(payload: Any, *, compact: bool = False) -> str:
    kwargs: dict[str, Any] = {"sort_keys": True, "default": str}
    if compact:
        kwargs["separators"] = (",", ":")
    return hashlib.sha256(json.dumps(payload, **kwargs).encode("utf-8")).hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def file_evidence(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "bytes": os.stat(_openable_path(path)).st_size,
        "sha256": sha256_file(path),
    }


def _finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def newey_west_mean_ci(
    values: Sequence[float], *, lag: int = 12, z: float = 1.96
) -> dict[str, Any]:
    array = np.asarray([float(value) for value in values if math.isfinite(float(value))])
    observations = int(len(array))
    if observations == 0:
        return {
            "mean": None,
            "lower": None,
            "upper": None,
            "standard_error": None,
            "observations": 0,
            "lag": 0,
        }
    mean = float(array.mean())
    if observations == 1:
        return {
            "mean": mean,
            "lower": mean,
            "upper": mean,
            "standard_error": 0.0,
            "observations": 1,
            "lag": 0,
        }
    effective_lag = min(max(int(lag), 0), observations - 1)
    demeaned = array - mean
    long_run_variance = float(np.dot(demeaned, demeaned) / observations)
    for step in range(1, effective_lag + 1):
        covariance = float(
            np.dot(demeaned[step:], demeaned[:-step]) / observations
        )
        long_run_variance += 2.0 * (1.0 - step / (effective_lag + 1.0)) * covariance
    standard_error = math.sqrt(max(long_run_variance / observations, 0.0))
    return {
        "mean": mean,
        "lower": mean - z * standard_error,
        "upper": mean + z * standard_error,
        "standard_error": standard_error,
        "observations": observations,
        "lag": effective_lag,
    }


def maximum_drawdown(returns: pd.Series) -> float | None:
    clean = _finite(returns).dropna()
    if clean.empty:
        return None
    equity = (1.0 + clean).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def _close(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def read_verified_parquet_log(
    namespace: Path,
    stem: str,
    *,
    columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = namespace / f"{stem}_manifest.json"
    manifest = _json(manifest_path)
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    actual_rows = 0
    records: list[dict[str, Any]] = []
    for part in manifest.get("parts", []):
        relative_path = str(part.get("path", ""))
        path = namespace / relative_path
        if not _path_exists(path):
            failures.append({"path": relative_path, "reason": "MISSING"})
            continue
        actual_hash = sha256_file(path)
        expected_hash = str(part.get("sha256", ""))
        if actual_hash != expected_hash:
            failures.append(
                {
                    "path": relative_path,
                    "reason": "SHA256_MISMATCH",
                    "expected": expected_hash,
                    "actual": actual_hash,
                }
            )
        frame = pd.read_parquet(_openable_path(path), columns=columns)
        rows = int(len(frame))
        actual_rows += rows
        if rows != int(part.get("rows", 0) or 0):
            failures.append(
                {
                    "path": relative_path,
                    "reason": "ROW_COUNT_MISMATCH",
                    "expected": int(part.get("rows", 0) or 0),
                    "actual": rows,
                }
            )
        frames.append(frame)
        records.append(
            {
                "path": relative_path,
                "rows": rows,
                "sha256": actual_hash,
            }
        )
    expected_rows = int(manifest.get("total_rows", 0) or 0)
    if actual_rows != expected_rows:
        failures.append(
            {
                "reason": "MANIFEST_TOTAL_ROWS_MISMATCH",
                "expected": expected_rows,
                "actual": actual_rows,
            }
        )
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, {
        "manifest": file_evidence(manifest_path),
        "part_count": len(records),
        "row_count": actual_rows,
        "part_record_digest": canonical_hash(records, compact=True),
        "all_part_hashes_and_rows_match": not failures,
        "failures": failures,
    }


def _daily_from_sleeves(sleeves: pd.DataFrame) -> pd.DataFrame:
    work = sleeves.copy()
    work["net_return_contribution"] = _finite(work["net_return_contribution"])
    work = work[work["net_return_contribution"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    grouped = work.groupby("maturity_session_date", sort=True)
    daily = grouped.agg(
        gross_daily_return=("gross_return_contribution", "sum"),
        net_daily_return=("net_return_contribution", "sum"),
        transaction_cost_drag=("transaction_cost_contribution", "sum"),
        mean_turnover=("turnover", "mean"),
        max_simultaneous_capital_fraction=("simultaneous_capital_fraction", "max"),
        matured_sleeve_decisions=("decision_timestamp", "size"),
    ).reset_index(names="session_date")
    return daily


def _recomputed_v3_summary(
    rank_ic: pd.DataFrame, daily: pd.DataFrame, sleeves: pd.DataFrame
) -> dict[str, Any]:
    ic = _finite(rank_ic["spearman_rank_ic"]).dropna()
    daily_net = _finite(daily["net_daily_return"]).dropna()
    daily_gross = _finite(daily["gross_daily_return"]).dropna()
    annualized_return = float(daily_net.mean() * TRADING_SESSIONS_PER_YEAR)
    annualized_volatility = float(
        daily_net.std(ddof=1) * math.sqrt(TRADING_SESSIONS_PER_YEAR)
    )
    daily_ic = (
        rank_ic.assign(spearman_rank_ic=_finite(rank_ic["spearman_rank_ic"]))
        .dropna(subset=["spearman_rank_ic"])
        .groupby("session_date")["spearman_rank_ic"]
        .mean()
    )
    turnover = _finite(sleeves["turnover"]).dropna()
    return {
        "resolved_performance_rows": int(len(rank_ic)),
        "rank_ic": {
            "valid_timestamps": int(len(ic)),
            "mean_spearman_rank_ic": float(ic.mean()),
            "median_spearman_rank_ic": float(ic.median()),
            "positive_fraction": float((ic > 0).mean()),
            "dependence_aware_95_ci": newey_west_mean_ci(ic.tolist(), lag=12),
            "daily_ic_rows": int(len(daily_ic)),
            "daily_mean_spearman_rank_ic": float(daily_ic.mean()),
        },
        "returns": {
            "daily_return_rows": int(len(daily_net)),
            "last_daily_net_return": float(daily_net.iloc[-1]),
            "cumulative_gross_return": float((1.0 + daily_gross).prod() - 1.0),
            "cumulative_net_return": float((1.0 + daily_net).prod() - 1.0),
            "annualized_return_from_daily_returns": annualized_return,
            "annualized_volatility_from_daily_returns": annualized_volatility,
            "daily_sharpe": annualized_return / annualized_volatility,
            "maximum_drawdown": maximum_drawdown(daily_net),
            "win_rate": float((daily_net > 0).mean()),
            "mean_turnover": float(turnover.mean()),
            "total_estimated_costs": float(
                _finite(sleeves["transaction_cost_contribution"]).sum()
            ),
        },
    }


def _summary_comparison(
    saved: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> dict[str, Any]:
    fields = {
        "resolved_performance_rows": (
            saved["resolved_performance_rows"],
            recomputed["resolved_performance_rows"],
        ),
        "rank_ic.valid_timestamps": (
            saved["rank_ic"]["valid_timestamps"],
            recomputed["rank_ic"]["valid_timestamps"],
        ),
        "rank_ic.mean_spearman_rank_ic": (
            saved["rank_ic"]["mean_spearman_rank_ic"],
            recomputed["rank_ic"]["mean_spearman_rank_ic"],
        ),
        "rank_ic.median_spearman_rank_ic": (
            saved["rank_ic"]["median_spearman_rank_ic"],
            recomputed["rank_ic"]["median_spearman_rank_ic"],
        ),
        "rank_ic.positive_fraction": (
            saved["rank_ic"]["positive_fraction"],
            recomputed["rank_ic"]["positive_fraction"],
        ),
        "rank_ic.newey_west.lower": (
            saved["rank_ic"]["dependence_aware_95_ci"]["lower"],
            recomputed["rank_ic"]["dependence_aware_95_ci"]["lower"],
        ),
        "rank_ic.newey_west.upper": (
            saved["rank_ic"]["dependence_aware_95_ci"]["upper"],
            recomputed["rank_ic"]["dependence_aware_95_ci"]["upper"],
        ),
        "returns.daily_return_rows": (
            saved["returns"]["daily_return_rows"],
            recomputed["returns"]["daily_return_rows"],
        ),
        "returns.cumulative_net_return": (
            saved["returns"]["cumulative_net_return"],
            recomputed["returns"]["cumulative_net_return"],
        ),
        "returns.annualized_return": (
            saved["returns"]["annualized_return_from_daily_returns"],
            recomputed["returns"]["annualized_return_from_daily_returns"],
        ),
        "returns.annualized_volatility": (
            saved["returns"]["annualized_volatility_from_daily_returns"],
            recomputed["returns"]["annualized_volatility_from_daily_returns"],
        ),
        "returns.daily_sharpe": (
            saved["returns"]["daily_sharpe"],
            recomputed["returns"]["daily_sharpe"],
        ),
        "returns.maximum_drawdown": (
            saved["returns"]["maximum_drawdown"],
            recomputed["returns"]["maximum_drawdown"],
        ),
        "returns.mean_turnover": (
            saved["returns"]["mean_turnover"],
            recomputed["returns"]["mean_turnover"],
        ),
        "returns.total_estimated_costs": (
            saved["returns"]["total_estimated_costs"],
            recomputed["returns"]["total_estimated_costs"],
        ),
    }
    rows = {
        name: {
            "saved": saved_value,
            "recomputed": recomputed_value,
            "matches": _close(saved_value, recomputed_value),
        }
        for name, (saved_value, recomputed_value) in fields.items()
    }
    return {"all_match": all(row["matches"] for row in rows.values()), "fields": rows}


def _verify_model_artifacts(
    family: str, family_root: Path, refits: pd.DataFrame
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checked = 0
    unique = refits.drop_duplicates(["training_cutoff", "model_hash"])
    for row in unique.itertuples(index=False):
        cutoff = pd.Timestamp(row.training_cutoff).tz_convert("UTC")
        name = f"{family}_{cutoff.strftime('%Y%m%dT%H%M%SZ')}.pkl"
        model_path = family_root / "models" / name
        if not _path_exists(model_path):
            failures.append({"path": _relative(model_path), "reason": "MISSING"})
            continue
        checked += 1
        actual_hash = sha256_file(model_path)
        if actual_hash != str(row.model_hash):
            failures.append(
                {
                    "path": _relative(model_path),
                    "reason": "MODEL_HASH_MISMATCH",
                    "expected": str(row.model_hash),
                    "actual": actual_hash,
                }
            )
    return {
        "expected_unique_models": int(len(unique)),
        "checked_models": checked,
        "all_referenced_model_hashes_match": not failures and checked == len(unique),
        "failures": failures,
    }


def audit_v3_family(family: str, namespace_name: str) -> dict[str, Any]:
    family_root = WORKER_ROOT / family
    namespace = family_root / namespace_name
    rank_columns = [
        "decision_timestamp",
        "session_date",
        "target_id",
        "training_cutoff",
        "model_hash",
        "eligible_asset_count",
        "resolved_asset_count",
        "duplicate_count",
        "chronology_violation_count",
        "spearman_rank_ic",
    ]
    sleeve_columns = [
        "decision_timestamp",
        "maturity_session_date",
        "sleeve_count",
        "sleeve_capital_fraction",
        "simultaneous_capital_fraction",
        "gross_return_contribution",
        "net_return_contribution",
        "turnover",
        "transaction_cost_contribution",
    ]
    daily_columns = [
        "session_date",
        "matured_sleeve_decisions",
        "gross_daily_return",
        "net_daily_return",
        "transaction_cost_drag",
        "mean_turnover",
        "max_simultaneous_capital_fraction",
    ]
    refit_columns = [
        "training_cutoff",
        "model_hash",
        "refit_session_date",
        "first_scored_decision_timestamp",
        "last_scored_decision_timestamp",
        "scored_decision_count",
        "daily_refit_with_five_minute_scoring",
        "no_five_minute_retraining",
    ]
    cost_columns = [
        "decision_timestamp",
        "transaction_cost_bps_per_unit_turnover",
        "transaction_cost_contribution",
    ]
    rank_ic, rank_verification = read_verified_parquet_log(
        namespace, "rank_ic_v3", columns=rank_columns
    )
    sleeves, sleeve_verification = read_verified_parquet_log(
        namespace, "sleeve_maturity_ledger_v3", columns=sleeve_columns
    )
    daily, daily_verification = read_verified_parquet_log(
        namespace, "daily_portfolio_returns_v3", columns=daily_columns
    )
    refits, refit_verification = read_verified_parquet_log(
        namespace, "refit_events_v3", columns=refit_columns
    )
    costs, cost_verification = read_verified_parquet_log(
        namespace, "transaction_costs_v3", columns=cost_columns
    )

    saved_summary_path = namespace / "resolved_performance_summary_v3.json"
    saved_summary = _json(saved_summary_path)
    recomputed_summary = _recomputed_v3_summary(rank_ic, daily, sleeves)
    derived_daily = _daily_from_sleeves(sleeves)
    ledger_recomputed_summary = _recomputed_v3_summary(
        rank_ic, derived_daily, sleeves
    )
    daily_join = daily.merge(
        derived_daily,
        on="session_date",
        how="outer",
        suffixes=("_saved", "_recomputed"),
        indicator=True,
    )
    daily_value_columns = (
        "gross_daily_return",
        "net_daily_return",
        "transaction_cost_drag",
        "mean_turnover",
        "max_simultaneous_capital_fraction",
        "matured_sleeve_decisions",
    )
    daily_differences: dict[str, Any] = {}
    missing_daily_rows = daily_join.loc[daily_join["_merge"] != "both"].copy()
    for column in daily_value_columns:
        saved_values = _finite(daily_join[f"{column}_saved"])
        recomputed_values = _finite(daily_join[f"{column}_recomputed"])
        difference = (saved_values - recomputed_values).abs()
        daily_differences[column] = {
            "max_abs_difference": float(difference.max()) if difference.notna().any() else 0.0,
            "mismatched_rows": int(
                ((difference.fillna(0.0) > 1e-12) | (daily_join["_merge"] != "both")).sum()
            ),
        }

    missing_daily_records = []
    for row in missing_daily_rows.to_dict(orient="records"):
        missing_daily_records.append(
            {
                "session_date": str(row["session_date"]),
                "presence": str(row["_merge"]),
                "saved_net_daily_return": (
                    None
                    if pd.isna(row["net_daily_return_saved"])
                    else float(row["net_daily_return_saved"])
                ),
                "ledger_net_daily_return": (
                    None
                    if pd.isna(row["net_daily_return_recomputed"])
                    else float(row["net_daily_return_recomputed"])
                ),
                "ledger_matured_sleeve_decisions": (
                    None
                    if pd.isna(row["matured_sleeve_decisions_recomputed"])
                    else int(row["matured_sleeve_decisions_recomputed"])
                ),
            }
        )

    decision_timestamps = pd.to_datetime(rank_ic["decision_timestamp"], utc=True)
    refit_cutoffs = pd.to_datetime(refits["training_cutoff"], utc=True)
    first_scores = pd.to_datetime(
        refits["first_scored_decision_timestamp"], utc=True
    )
    cost_bps = sorted(
        set(_finite(costs["transaction_cost_bps_per_unit_turnover"]).dropna().tolist())
    )
    namespace_verifications = {
        "rank_ic_duplicate_decision_timestamps": int(
            rank_ic.duplicated(["decision_timestamp"]).sum()
        ),
        "sleeve_duplicate_decision_timestamps": int(
            sleeves.duplicated(["decision_timestamp"]).sum()
        ),
        "daily_duplicate_session_dates": int(daily.duplicated(["session_date"]).sum()),
        "refit_duplicate_keys": int(
            refits.duplicated(["training_cutoff", "model_hash"]).sum()
        ),
        "reported_duplicate_count_sum": int(
            _finite(rank_ic["duplicate_count"]).fillna(0).sum()
        ),
        "reported_chronology_violation_count_sum": int(
            _finite(rank_ic["chronology_violation_count"]).fillna(0).sum()
        ),
        "target_contract_values": sorted(set(rank_ic["target_id"].astype(str))),
        "locked_holdout_score_rows": int(
            (decision_timestamps >= LOCKED_HOLDOUT_START).sum()
        ),
        "persisted_training_cutoff_strictly_before_first_score_count": int(
            (refit_cutoffs < first_scores).sum()
        ),
        "persisted_training_cutoff_equal_first_score_count": int(
            (refit_cutoffs == first_scores).sum()
        ),
        "refit_count": int(len(refits)),
        "transaction_cost_bps_values": cost_bps,
        "nonzero_transaction_cost_rows": int(
            (_finite(costs["transaction_cost_contribution"]).fillna(0.0) != 0.0).sum()
        ),
        "sleeve_count_values": sorted(
            set(_finite(sleeves["sleeve_count"]).dropna().astype(int).tolist())
        ),
        "sleeve_capital_fraction_values": sorted(
            set(_finite(sleeves["sleeve_capital_fraction"]).dropna().tolist())
        ),
        "simultaneous_capital_fraction_max": float(
            _finite(sleeves["simultaneous_capital_fraction"]).max()
        ),
        "daily_recomputed_from_sleeves": {
            "row_count_matches": bool((daily_join["_merge"] == "both").all()),
            "missing_or_extra_rows": missing_daily_records,
            "field_comparison": daily_differences,
        },
    }
    return {
        "family": family,
        "namespace": _relative(namespace),
        "summary": file_evidence(saved_summary_path),
        "saved_status": saved_summary["status"],
        "saved_summary": saved_summary,
        "recomputed_summary_from_published_daily_log": recomputed_summary,
        "recomputed_summary_from_complete_sleeve_ledger": ledger_recomputed_summary,
        "saved_summary_comparison": _summary_comparison(
            saved_summary, recomputed_summary
        ),
        "complete_sleeve_ledger_summary_comparison": _summary_comparison(
            saved_summary, ledger_recomputed_summary
        ),
        "log_verification": {
            "rank_ic_v3": rank_verification,
            "sleeve_maturity_ledger_v3": sleeve_verification,
            "daily_portfolio_returns_v3": daily_verification,
            "refit_events_v3": refit_verification,
            "transaction_costs_v3": cost_verification,
        },
        "namespace_verification": namespace_verifications,
        "model_artifacts": _verify_model_artifacts(family, family_root, refits),
    }


def _model_ordinals(model_paths: Iterable[Path]) -> list[int]:
    ordinals: list[int] = []
    for path in model_paths:
        match = re.search(r"refit=(\d{6})\.pkl$", path.name)
        if match:
            ordinals.append(int(match.group(1)))
    return ordinals


def _xendcg_fixture_verification(
    source_manifest: Mapping[str, Any], oof_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    oof_by_ordinal = {
        int(row["refit_ordinal"]): row for row in oof_manifest.get("files", [])
    }
    fixtures: dict[str, Any] = {}
    for expected in source_manifest.get("xendcg_artifact_hashes", []):
        ordinal = int(expected["refit"])
        key = f"{ordinal:06d}"
        model_path = (
            XENDCG_ROOT
            / "model_artifacts"
            / f"lightgbm_rank_xendcg_refit={key}.pkl"
        )
        oof_path = XENDCG_ROOT / oof_by_ordinal[ordinal]["relative_path"]
        model_hash = sha256_file(model_path)
        oof_hash = sha256_file(oof_path)
        fixtures[key] = {
            "model_path": _relative(model_path),
            "expected_model_sha256": expected["model_sha256"],
            "actual_model_sha256": model_hash,
            "model_hash_match": model_hash == expected["model_sha256"],
            "oof_path": _relative(oof_path),
            "expected_oof_sha256": expected["oof_sha256"],
            "actual_oof_sha256": oof_hash,
            "oof_hash_match": oof_hash == expected["oof_sha256"],
        }
    return fixtures


def audit_xendcg() -> dict[str, Any]:
    oof_manifest_path = XENDCG_ROOT / "ensemble_oof_scores_manifest_v2.json"
    oof_manifest = _json(oof_manifest_path)
    oof_failures: list[dict[str, Any]] = []
    observed_decisions: set[str] = set()
    observed_assets: set[str] = set()
    observed_ordinals: list[int] = []
    actual_rows = 0
    verified_records: list[dict[str, Any]] = []
    for part in oof_manifest.get("files", []):
        path = XENDCG_ROOT / str(part["relative_path"])
        actual_hash = sha256_file(path) if _path_exists(path) else ""
        if actual_hash != str(part.get("sha256", "")):
            oof_failures.append(
                {
                    "path": _relative(path),
                    "reason": "SHA256_MISMATCH" if _path_exists(path) else "MISSING",
                    "expected": part.get("sha256", ""),
                    "actual": actual_hash,
                }
            )
            continue
        frame = pd.read_parquet(
            _openable_path(path),
            columns=["decision_timestamp", "asset_id", "refit_ordinal"],
        )
        actual_rows += len(frame)
        timestamps = pd.to_datetime(frame["decision_timestamp"], utc=True)
        timestamp_values = set(timestamps.map(lambda value: value.isoformat()))
        if len(timestamp_values) != 1:
            oof_failures.append(
                {"path": _relative(path), "reason": "MULTIPLE_DECISION_TIMESTAMPS"}
            )
        duplicate_assets = int(frame.duplicated(["decision_timestamp", "asset_id"]).sum())
        if duplicate_assets:
            oof_failures.append(
                {
                    "path": _relative(path),
                    "reason": "DUPLICATE_SCORE_KEYS",
                    "count": duplicate_assets,
                }
            )
        if observed_decisions.intersection(timestamp_values):
            oof_failures.append(
                {"path": _relative(path), "reason": "DUPLICATE_DECISION_PARTITION"}
            )
        observed_decisions.update(timestamp_values)
        observed_assets.update(frame["asset_id"].astype(str))
        ordinal_values = set(frame["refit_ordinal"].astype(int))
        if ordinal_values != {int(part["refit_ordinal"])}:
            oof_failures.append(
                {"path": _relative(path), "reason": "REFIT_ORDINAL_MISMATCH"}
            )
        observed_ordinals.extend(ordinal_values)
        if len(frame) != int(part["row_count"]):
            oof_failures.append(
                {"path": _relative(path), "reason": "ROW_COUNT_MISMATCH"}
            )
        verified_records.append(
            {
                "path": str(part["relative_path"]),
                "rows": int(len(frame)),
                "sha256": actual_hash,
            }
        )

    model_root = XENDCG_ROOT / "model_artifacts"
    model_paths = sorted(
        path
        for path in model_root.glob("lightgbm_rank_xendcg_refit=*.pkl")
        if not path.name.startswith("._")
    )
    sidecars = sorted(model_root.glob("._lightgbm_rank_xendcg_refit=*.pkl"))
    model_ordinals = _model_ordinals(model_paths)
    ordinal_counts = {
        ordinal: model_ordinals.count(ordinal) for ordinal in sorted(set(model_ordinals))
    }
    expected_ordinals = set(
        range(EXPECTED_XENDCG_FIRST_ORDINAL, EXPECTED_XENDCG_LAST_ORDINAL + 1)
    )
    missing_ordinals = sorted(expected_ordinals - set(model_ordinals))
    duplicate_ordinals = sorted(
        ordinal for ordinal, count in ordinal_counts.items() if count > 1
    )

    lineage = {
        int(row["refit_ordinal"]): pd.Timestamp(row["training_cutoff_timestamp"])
        for row in oof_manifest.get("refit_lineage", [])
    }
    strict_cutoff_count = 0
    for part in oof_manifest.get("files", []):
        ordinal = int(part["refit_ordinal"])
        score_timestamp = pd.Timestamp(part["first_decision_timestamp"])
        if lineage[ordinal] < score_timestamp:
            strict_cutoff_count += 1

    source_manifest_path = XENDCG_AUTHORITY_ROOT / "source_manifest.json"
    parity_path = XENDCG_AUTHORITY_ROOT / "three_refit_parity.json"
    producer_path = (
        XENDCG_AUTHORITY_ROOT / "core/research/ml/ds24/mac_aux_queue_r44f2.py"
    )
    engine_path = (
        XENDCG_AUTHORITY_ROOT
        / "core/research/ml/ds24/canonical_prequential_engine.py"
    )
    evaluator_path = XENDCG_AUTHORITY_ROOT / "core/research/ml/ds24_metrics_only_evaluator.py"
    source_manifest = _json(source_manifest_path)
    primary_source = next(
        row
        for row in source_manifest["source_files"]
        if row["relative_path"] == "core/research/ml/ds24/mac_aux_queue_r44f2.py"
    )
    fixtures = _xendcg_fixture_verification(source_manifest, oof_manifest)

    metrics_summary_path = XENDCG_ROOT / "metrics_only_v3/resolved_performance_summary_v3.json"
    per_t_path = XENDCG_ROOT / "metrics_only_v3/per_t_metrics.parquet"
    decision_trace_path = XENDCG_ROOT / "metrics_only_v3/decision_trace.parquet"
    metrics_summary = _json(metrics_summary_path)
    per_t = pd.read_parquet(_openable_path(per_t_path))
    valid_ic = _finite(per_t["spearman_rank_ic"]).dropna()
    net_returns = _finite(per_t["net_return"]).dropna()
    gross_returns = _finite(per_t["gross_return"]).dropna()
    annualized_return = float(net_returns.mean() * TRADING_SESSIONS_PER_YEAR)
    annualized_volatility = float(
        net_returns.std(ddof=1) * math.sqrt(TRADING_SESSIONS_PER_YEAR)
    )
    reproduced_metrics = {
        "rows": int(len(per_t)),
        "valid_rank_ic_rows": int(len(valid_ic)),
        "mean_spearman_rank_ic": float(valid_ic.mean()),
        "positive_ic_fraction": float((valid_ic > 0).mean()),
        "gross_net_max_abs_difference": float(
            (gross_returns - net_returns).abs().max()
        ),
        "annualized_arithmetic_return": annualized_return,
        "annual_volatility": annualized_volatility,
        "daily_sharpe": annualized_return / annualized_volatility,
        "maximum_drawdown": maximum_drawdown(net_returns),
        "mean_daily_net_return": float(net_returns.mean()),
    }

    import_authority_path = STAGE_ROOT / "R47A_xendcg_import_authority.json"
    import_authority = _json(import_authority_path)
    authority_hash = canonical_hash(
        {key: value for key, value in import_authority.items() if key != "authority_hash"}
    )
    validation_path = STAGE_ROOT / "R47A_xendcg_artifact_validation.json"
    validation = _json(validation_path)

    inventory = {
        "root": import_authority["contract"]["artifact_manifest"]["root"],
        "model_root": import_authority["contract"]["artifact_manifest"]["model_root"],
        "model_artifact_count": len(model_paths),
        "genuine_pkl_count": len(model_paths),
        "apple_double_sidecar_count": len(sidecars),
        "missing_ordinals": [f"{ordinal:06d}" for ordinal in missing_ordinals],
        "missing_ordinal_count": len(missing_ordinals),
        "duplicate_ordinals": [f"{ordinal:06d}" for ordinal in duplicate_ordinals],
        "duplicate_ordinal_count": len(duplicate_ordinals),
        "first_ordinal": f"{min(model_ordinals):06d}" if model_ordinals else "",
        "last_ordinal": f"{max(model_ordinals):06d}" if model_ordinals else "",
        "expected_first_ordinal": f"{EXPECTED_XENDCG_FIRST_ORDINAL:06d}",
        "expected_last_ordinal": f"{EXPECTED_XENDCG_LAST_ORDINAL:06d}",
        "expected_genuine_pkl_count": EXPECTED_XENDCG_MODEL_COUNT,
        "unexpected_filenames": [],
    }
    producer_authority = validation["producer_authority"]
    stable_artifact_hash = canonical_hash(
        {
            "source_root_type": "MAC_AUX_FAMILY_ROOT",
            "source_root": import_authority["contract"]["source_root"],
            "model_inventory": inventory,
            "producer_sha": producer_authority["producer_sha"],
            "mac_head": producer_authority["mac_head"],
            "feature_order_sha256": producer_authority["feature_order_sha256"],
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "three_refit_parity_sha256": sha256_file(parity_path),
            "fixture_hash_validation": fixtures,
            "oof_manifest_sha256": sha256_file(oof_manifest_path),
            "metrics_sha256": sha256_file(metrics_summary_path),
        }
    )
    score_timestamps = pd.to_datetime(list(observed_decisions), utc=True)
    return {
        "family": "lightgbm_rank_xendcg",
        "source_root": _relative(XENDCG_ROOT),
        "oof_manifest": file_evidence(oof_manifest_path),
        "oof_verification": {
            "listed_part_count": len(oof_manifest.get("files", [])),
            "verified_part_count": len(verified_records),
            "part_record_digest": canonical_hash(verified_records, compact=True),
            "row_count": actual_rows,
            "distinct_assets": len(observed_assets),
            "distinct_decision_timestamps": len(observed_decisions),
            "distinct_refit_ordinals": len(set(observed_ordinals)),
            "all_part_hashes_rows_and_keys_match": not oof_failures,
            "failures": oof_failures,
            "strict_training_cutoff_before_score_count": strict_cutoff_count,
            "locked_holdout_score_rows": int(
                (score_timestamps >= LOCKED_HOLDOUT_START).sum()
            ),
            "manifest_provisional": bool(oof_manifest.get("provisional")),
            "terminal_completeness_state": oof_manifest.get(
                "terminal_completeness_state"
            ),
        },
        "model_inventory": inventory,
        "fixture_hash_verification": fixtures,
        "producer_authority": {
            "source_manifest": file_evidence(source_manifest_path),
            "parity": file_evidence(parity_path),
            "producer_source": file_evidence(producer_path),
            "prequential_engine_source": file_evidence(engine_path),
            "metrics_evaluator_source": file_evidence(evaluator_path),
            "producer_hash_matches_manifest": sha256_file(producer_path)
            == primary_source["sha256"],
            "producer_source_classification": source_manifest[
                "producer_source_classification"
            ],
            "advertised_git_head": source_manifest["advertised_commit"],
            "feature_order_sha256": source_manifest["feature_authority"][
                "feature_order_sha256"
            ],
            "runtime_arguments": source_manifest["runtime_arguments"],
        },
        "metrics": {
            "summary": file_evidence(metrics_summary_path),
            "per_t_metrics": file_evidence(per_t_path),
            "decision_trace": file_evidence(decision_trace_path),
            "saved": metrics_summary,
            "recomputed": reproduced_metrics,
            "transaction_cost_bps": 0.0,
        },
        "import_authority": {
            "file": file_evidence(import_authority_path),
            "recorded_authority_hash": import_authority["authority_hash"],
            "recomputed_authority_hash": authority_hash,
            "authority_hash_matches": authority_hash
            == import_authority["authority_hash"],
            "validation_file": file_evidence(validation_path),
            "recorded_stable_artifact_hash": validation["stable_artifact_hash"],
            "recomputed_stable_artifact_hash": stable_artifact_hash,
            "stable_artifact_hash_matches": stable_artifact_hash
            == validation["stable_artifact_hash"],
        },
    }


def synthetic_future_feature_dependence() -> dict[str, Any]:
    timestamps = pd.date_range("2024-01-02T09:00:00Z", periods=180, freq="5min")
    base = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "session_date": "2024-01-02",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": np.linspace(100.0, 101.0, len(timestamps)),
            "volume": 1000.0,
            "trade_count": 10,
            "vwap": 100.0,
        }
    )
    changed = base.copy()
    changed.loc[changed.index[-1], ["high", "close", "vwap"]] = 150.0
    base_features = compute_stock_features(base)
    changed_features = compute_stock_features(changed)
    decision_index = int(np.flatnonzero(timestamps == pd.Timestamp("2024-01-02T14:35:00Z"))[0])
    changes: dict[str, Any] = {}
    for predictor in LEAKED_PREDICTORS:
        before = float(base_features.iloc[decision_index][predictor])
        after = float(changed_features.iloc[decision_index][predictor])
        changes[predictor] = {
            "before": before,
            "after": after,
            "changed_when_only_future_final_bar_changed": not _close(before, after),
        }
    causal_control_before = float(base_features.iloc[decision_index]["ret_5m"])
    causal_control_after = float(changed_features.iloc[decision_index]["ret_5m"])
    return {
        "decision_timestamp": timestamps[decision_index].isoformat(),
        "mutated_future_timestamp": timestamps[-1].isoformat(),
        "predictors": changes,
        "causal_control_ret_5m": {
            "before": causal_control_before,
            "after": causal_control_after,
            "unchanged": _close(causal_control_before, causal_control_after),
        },
        "all_leaked_predictors_change": all(
            row["changed_when_only_future_final_bar_changed"]
            for row in changes.values()
        ),
    }


def persisted_feature_leakage_example() -> dict[str, Any]:
    feature_path = FEATURE_ROOT / "stock/asset=AAPL/year=2024/features.parquet"
    raw_path = RAW_BAR_ROOT / "symbol=AAPL/year=2024/bars.parquet"
    feature = pd.read_parquet(
        _openable_path(feature_path),
        columns=[
            "decision_timestamp",
            "session_date",
            "overnight_gap",
            "previous_session_return",
            "two_session_return",
        ],
    )
    bars = pd.read_parquet(
        _openable_path(raw_path),
        columns=["timestamp_utc", "session_date", "open", "close"],
    )
    session = "2024-01-02"
    decision = pd.Timestamp("2024-01-02T14:35:00Z")
    feature_row = feature[
        (feature["session_date"].astype(str) == session)
        & (pd.to_datetime(feature["decision_timestamp"], utc=True) == decision)
    ].iloc[0]
    session_bars = bars[bars["session_date"].astype(str) == session].sort_values(
        "timestamp_utc"
    )
    current_close = float(
        session_bars[
            pd.to_datetime(session_bars["timestamp_utc"], utc=True)
            == pd.Timestamp("2024-01-02T14:30:00Z")
        ].iloc[0]["close"]
    )
    first_open = float(session_bars.iloc[0]["open"])
    future_last_close = float(session_bars.iloc[-1]["close"])
    calculations = {
        "previous_session_return": future_last_close / first_open - 1.0,
        "overnight_gap": first_open / future_last_close - 1.0,
        "two_session_return": current_close / future_last_close - 1.0,
    }
    persisted = {
        predictor: float(feature_row[predictor]) for predictor in LEAKED_PREDICTORS
    }
    return {
        "feature_file": file_evidence(feature_path),
        "raw_bar_file": file_evidence(raw_path),
        "decision_timestamp": decision.isoformat(),
        "future_source_timestamp": pd.Timestamp(
            session_bars.iloc[-1]["timestamp_utc"]
        ).isoformat(),
        "future_last_close": future_last_close,
        "persisted_values": persisted,
        "values_recomputed_with_future_close": calculations,
        "all_values_match_future_close_calculation": all(
            _close(persisted[name], calculations[name], tolerance=1e-7)
            for name in LEAKED_PREDICTORS
        ),
    }


def audit_feature_authority() -> dict[str, Any]:
    predictor_manifest = _json(PREDICTOR_MANIFEST)
    predictors = list(predictor_manifest["predictors"])
    compact_feature_hash = canonical_hash(predictors, compact=True)
    return {
        "predictor_manifest": file_evidence(PREDICTOR_MANIFEST),
        "feature_implementation": file_evidence(FEATURE_IMPLEMENTATION),
        "predictor_count": len(predictors),
        "manifest_hash_recomputed": canonical_hash(predictors),
        "manifest_hash_recorded": predictor_manifest["hash"],
        "feature_order_sha256_recomputed": compact_feature_hash,
        "feature_order_sha256_expected": EXPECTED_FEATURE_ORDER_SHA256,
        "leaked_predictors_in_manifest": [
            predictor for predictor in LEAKED_PREDICTORS if predictor in predictors
        ],
        "synthetic_future_perturbation": synthetic_future_feature_dependence(),
        "persisted_example": persisted_feature_leakage_example(),
    }


def audit_paper_binding() -> dict[str, Any]:
    paper_config = REPOSITORY_ROOT / "configs/paper/alpaca_7day_trial.yaml"
    config_text = paper_config.read_text(encoding="utf-8")
    order_adapter = REPOSITORY_ROOT / "core/paper/ds24_order_adapter.py"
    adapter_text = order_adapter.read_text(encoding="utf-8")
    config_hash = re.search(r'DS24_CONFIG_HASH = "([0-9a-f]+)"', adapter_text)
    policy_hash = re.search(
        r'DS24_TRAINING_POLICY_HASH = "([0-9a-f]+)"', adapter_text
    )
    incumbent = (
        DS24_ROOT
        / "stage_outputs/ds24_p8_r9_20260822T000000Z/01_r8_candidate_lock.json"
    )
    candidate_source = re.search(r"(?m)^\s+source:\s*(\S+)\s*$", config_text)
    audited_names = ("huber", "xendcg", "rankxendcg", "elastic_net")
    return {
        "active_paper_config": file_evidence(paper_config),
        "active_candidate_source": candidate_source.group(1) if candidate_source else "",
        "active_submit_orders": bool(
            re.search(r"(?m)^\s+submit_orders:\s*true\s*$", config_text)
        ),
        "active_paper_broker": bool(
            re.search(r"(?m)^\s+paper:\s*true\s*$", config_text)
        ),
        "active_costs_bps": {"transaction_cost": 10.0, "slippage": 5.0},
        "audited_model_names_present_in_active_config": [
            name for name in audited_names if name in config_text.lower()
        ],
        "ds24_order_adapter": file_evidence(order_adapter),
        "ds24_order_adapter_config_hash": config_hash.group(1) if config_hash else "",
        "ds24_order_adapter_training_policy_hash": policy_hash.group(1)
        if policy_hash
        else "",
        "bound_incumbent_authority": file_evidence(incumbent),
        "bound_incumbent": _json(incumbent),
    }


def _key_authorities() -> dict[str, Any]:
    paths = {
        "huber_r36_containment": STAGE_ROOT
        / "R7_R36_04_huber_fatal_and_namespace_audit.json",
        "huber_r37_lineage": STAGE_ROOT / "R7_R37_03_namespace_lineage.json",
        "policy_authority": STAGE_ROOT / "R7_R14_01_policy_authority.json",
        "extended_model_data_authority": STAGE_ROOT
        / "92_r7_r1_extended_model_data_authority.json",
        "extended_model_data_partition_manifest": STAGE_ROOT
        / "91_r7_r1_extended_model_data_partition_manifest.csv",
        "feature_contract": DS24_ROOT / "r7_master_feature_contract.json",
        "pit_validation": DS24_ROOT / "r6_pit_validation.json",
        "cost_aware_programme_state": DS24_ROOT
        / "cost_aware_policy_experiment"
        / "cost_aware_policy_terminal_r4_20260917T141014Z"
        / "current_programme_state.json",
        "cost_aware_candidate_registry": DS24_ROOT
        / "cost_aware_policy_experiment"
        / "cost_aware_policy_terminal_r4_20260917T141014Z"
        / "candidate_registry.json",
        "certification_architecture": REPOSITORY_ROOT
        / "docs/architecture/certification_and_promotion.md",
        "paper_architecture": REPOSITORY_ROOT
        / "docs/architecture/paper_trading_architecture.md",
        "point_in_time_architecture": REPOSITORY_ROOT
        / "docs/architecture/point_in_time_and_lineage.md",
        "policy_worker_source": REPOSITORY_ROOT
        / "scripts/local/ds24_p8_r14_e3g_c2_r7_r14_policy_worker.py",
        "huber_estimator_source": REPOSITORY_ROOT
        / "scripts/local/ds24_p8_r14_e3g_c2_r7_r10_zero_engineering_disposition.py",
        "elastic_adapter_source": REPOSITORY_ROOT
        / "scripts/local/ds24_elastic_net_adapter.py",
        "elastic_estimator_source": REPOSITORY_ROOT
        / "core/research/ml/stock_level_benchmark_models.py",
        "metrics_evaluator_source": REPOSITORY_ROOT
        / "core/research/ml/ds24_metrics_only_evaluator.py",
        "target_builder_source": REPOSITORY_ROOT
        / "core/research/ml/five_minute_target_dataset.py",
    }
    return {name: file_evidence(path) for name, path in paths.items()}


def build_evidence() -> dict[str, Any]:
    return {
        "audit_id": "DS24_PAPER_PROVENANCE_FORENSIC_20260926",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head": subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "scope": {
            "read_only_source_and_canonical_artifact_access": True,
            "training_started": False,
            "orders_placed": False,
            "canonical_artifacts_modified": False,
        },
        "feature_authority": audit_feature_authority(),
        "models": {
            "huber": audit_v3_family("huber", "metrics_only_v3_r37_huber_replay"),
            "lightgbm_rank_xendcg": audit_xendcg(),
            "elastic_net": audit_v3_family(
                "elastic_net", "metrics_only_v3_r40_elastic_net"
            ),
        },
        "paper_binding": audit_paper_binding(),
        "key_authorities": _key_authorities(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute DS24 PAPER-provenance evidence without fitting models."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New audit evidence JSON path. Existing files are refused.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit evidence: {output}")
    audit_root = (REPOSITORY_ROOT / "docs/audits").resolve()
    if audit_root not in output.parents:
        raise ValueError(f"Output must be under the new audit namespace {audit_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = build_evidence()
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
