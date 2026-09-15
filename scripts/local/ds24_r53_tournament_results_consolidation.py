from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


STAGE_ROOT = Path(
    "docs/dream_system/components/DS-24_independent_five_minute_selector/"
    "stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z"
)
WORKER_ROOT = STAGE_ROOT / "r7_r14_policy_workers"
MAC_AUX_ROOT = Path("mac_aux_runs/queue=DS24_MAC_AUX_NINE_FAMILY_R1")
R53_DIRNAME = "R53_19_family_tournament_results"

REQUESTED_FAMILIES: tuple[tuple[str, str], ...] = (
    ("ridge_policy_v1_control", "ridge_policy_v1_control"),
    ("pca_ridge_policy_v1_control", "pca_ridge_policy_v1_control"),
    ("spline_additive_ridge", "spline_additive_ridge"),
    ("elastic_net", "elastic_net"),
    ("rff_ridge", "rff_ridge"),
    ("huber", "huber"),
    ("mlp", "mlp"),
    ("random_forest", "random_forest"),
    ("extra_trees", "extra_trees"),
    ("gradient_boosting", "gradient_boosting"),
    ("lightgbm_rank_xendcg", "lightgbm_rank_xendcg"),
    ("lightgbm_lambdarank", "lightgbm_lambdarank"),
    ("DLinear", "DLinear"),
    ("PatchTST", "PatchTST"),
    ("Transformer", "Transformer"),
    ("iTransformer", "iTransformer"),
    ("Momentum Transformer", "Momentum Transformer"),
    ("Market Context Encoder", "Market Context Encoder"),
    ("Temporal Fusion Transformer", "Temporal Fusion Transformer"),
)

ALIASES: dict[str, str] = {
    "ridge": "ridge_policy_v1_control",
    "pca_ridge": "pca_ridge_policy_v1_control",
    "dlinear": "DLinear",
    "patchtst": "PatchTST",
    "transformer": "Transformer",
    "itransformer": "iTransformer",
    "momentum_transformer": "Momentum Transformer",
    "market_context_encoder": "Market Context Encoder",
    "temporal_fusion_transformer": "Temporal Fusion Transformer",
}

AUTHORITY_FILENAMES = {
    "resolved_performance_summary_v3.json",
    "resolved_performance_checkpoint_v3.json",
    "rank_ic_v3_manifest.json",
    "daily_portfolio_returns_v3_manifest.json",
    "transaction_costs_v3_manifest.json",
    "r31_performance_summary.json",
    "family_execution_summary.json",
    "terminal_results.json",
    "progress.json",
    "checkpoint.json",
    "ensemble_oof_scores_manifest_v2.json",
}

OUTPUT_FILES = (
    "R53_pre_results_audit_runtime_snapshot.json",
    "R53_post_results_audit_runtime_snapshot.json",
    "R53_19_family_results_matrix.csv",
    "R53_19_family_results_matrix.json",
    "R53_completed_final_results_leaderboard.csv",
    "R53_rank_ic_leaderboard.csv",
    "R53_sharpe_leaderboard.csv",
    "R53_running_provisional_results.csv",
    "R53_common_window_readiness.csv",
    "R53_cross_host_results_matrix.csv",
    "R53_source_evidence_manifest.json",
    "R53_missing_result_authority.json",
    "R53_19_FAMILY_TOURNAMENT_RESULTS_REPORT.md",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_family(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if value in dict(REQUESTED_FAMILIES):
        return value
    lowered = value.strip().lower().replace("-", "_")
    if lowered.startswith("family="):
        lowered = lowered.split("=", 1)[1]
    return ALIASES.get(lowered, value if value in dict(REQUESTED_FAMILIES) else None)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def maybe_read_json(path: Path) -> Any | None:
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def nested_get(payload: Any, paths: Iterable[tuple[str, ...]]) -> Any:
    for path in paths:
        current = payload
        for key in path:
            if not isinstance(current, dict) or key not in current:
                break
            current = current[key]
        else:
            return current
    return None


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    return int(number)


def normalize_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def days_between(start: Any, end: Any) -> int | None:
    if not isinstance(start, str) or not isinstance(end, str):
        return None
    try:
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max((end_dt.date() - start_dt.date()).days, 0)


@dataclass(frozen=True)
class MetricEvidence:
    family: str
    path: Path
    payload: dict[str, Any]
    source_tier: str
    source_host: str
    imported: bool
    retained_oof: bool
    rank: int

    @property
    def namespace(self) -> str:
        if self.path.name == "resolved_performance_summary_v3.json":
            return self.path.parent.name
        return self.path.parent.as_posix()


def latest_existing(paths: Iterable[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def family_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        family = canonical_family(part)
        if family:
            return family
        if part.startswith("family="):
            family = canonical_family(part)
            if family:
                return family
    return None


def extract_family(payload: Any, path: Path) -> str | None:
    if isinstance(payload, dict):
        direct = canonical_family(payload.get("family"))
        if direct:
            return direct
        nested = canonical_family(nested_get(payload, (("metadata", "family"), ("summary", "family"))))
        if nested:
            return nested
    return family_from_path(path)


def is_r53_output_path(path: Path, output_root: Path) -> bool:
    try:
        path.resolve().relative_to(output_root.resolve())
        return True
    except ValueError:
        return False


def write_text(output_root: Path, name: str, text: str) -> None:
    path = output_root / name
    if not is_r53_output_path(path, output_root):
        raise RuntimeError(f"refusing to write outside R53 output root: {path}")
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(output_root: Path, name: str, payload: Any) -> None:
    write_text(output_root, name, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(output_root: Path, name: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = output_root / name
    if not is_r53_output_path(path, output_root):
        raise RuntimeError(f"refusing to write outside R53 output root: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def load_family_authority(path: Path) -> dict[str, dict[str, Any]]:
    payload = maybe_read_json(path)
    if not isinstance(payload, dict):
        return {}
    raw = payload.get("by_family")
    if isinstance(raw, dict):
        return {family: value for key, value in raw.items() if (family := canonical_family(key)) and isinstance(value, dict)}
    families = payload.get("families")
    if isinstance(families, list):
        result: dict[str, dict[str, Any]] = {}
        for item in families:
            if isinstance(item, dict) and (family := canonical_family(item.get("family"))):
                result[family] = item
        return result
    return {}


def load_latest_ownership(stage_root: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    authority_paths = [
        stage_root / "R51_cross_host_ownership_state.json",
        stage_root / "R49_cross_host_ownership_state.json",
        stage_root / "R47A_cross_host_ownership_state.json",
        stage_root / "R44_cross_host_family_ownership.json",
    ]
    used: list[str] = []
    merged: dict[str, dict[str, Any]] = {}
    for path in reversed(authority_paths):
        authority = load_family_authority(path)
        if authority:
            used.append(normalize_rel(path))
            merged.update(authority)
    return merged, used


def repository_population(stage_root: Path, worker_root: Path, mac_aux_root: Path) -> list[str]:
    families: set[str] = set()
    for authority in [
        stage_root / "R42_full_family_readiness_matrix.json",
        stage_root / "R44_cross_host_family_ownership.json",
        stage_root / "R51_cross_host_ownership_state.json",
    ]:
        families.update(load_family_authority(authority))
    if worker_root.exists():
        for child in worker_root.iterdir():
            if child.is_dir() and (family := canonical_family(child.name)):
                families.add(family)
    if mac_aux_root.exists():
        for child in mac_aux_root.iterdir():
            if child.is_dir() and (family := family_from_path(child)):
                families.add(family)
    return [family for family, _ in REQUESTED_FAMILIES if family in families] + sorted(
        families.difference(dict(REQUESTED_FAMILIES))
    )


def discover_candidate_files(stage_root: Path, worker_root: Path, mac_aux_root: Path) -> list[Path]:
    roots = [worker_root, mac_aux_root, stage_root / "R52_patchtst_real_path_smoke"]
    files: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for directory, dirnames, filenames in os.walk(root, onerror=lambda _error: None):
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if dirname != R53_DIRNAME
                and not dirname.endswith("_parts")
                and not dirname.startswith("decision_date=")
                and dirname not in {"model_store", "prediction_partitions"}
            ]
            parent = Path(directory)
            for filename in filenames:
                if filename in AUTHORITY_FILENAMES and not filename.startswith("._"):
                    files.add(parent / filename)
    for pattern in ("R47A_xendcg_*.json", "R49_lambdarank_*.json", "R51_lambdarank_*.json"):
        files.update(stage_root.glob(pattern))
    return sorted(files, key=lambda p: normalize_rel(p).lower())


def classify_source(path: Path, payload: dict[str, Any], family: str) -> tuple[str, int, str, bool, bool]:
    text = normalize_rel(path).lower()
    imported = family in {"lightgbm_rank_xendcg", "lightgbm_lambdarank", "DLinear"} and (
        "mac_aux_runs" in text or "import" in text or family == "lightgbm_rank_xendcg"
    )
    retained_oof = "oof" in text or path.name == "ensemble_oof_scores_manifest_v2.json"
    source_host = "MAC" if "mac_aux_runs" in text or imported else "DELL"
    if path.name == "resolved_performance_summary_v3.json":
        if imported:
            return ("C", 20, source_host, imported, retained_oof)
        return ("A", 10, source_host, False, retained_oof)
    if path.name == "family_execution_summary.json" or "import_result" in path.name:
        return ("C", 30, source_host, imported, retained_oof)
    if path.name == "r31_performance_summary.json":
        accepted = bool(nested_get(payload, (("accepted_performance_results", "accepted"), ("display_performance", "accepted"))))
        return ("B" if accepted else "E", 40 if accepted else 70, source_host, imported, retained_oof)
    if "quarantine" in text or "superseded" in text:
        return ("E", 80, source_host, imported, retained_oof)
    return ("B", 50, source_host, imported, retained_oof)


def discover_metric_evidence(stage_root: Path, worker_root: Path, mac_aux_root: Path) -> dict[str, list[MetricEvidence]]:
    evidence: dict[str, list[MetricEvidence]] = {family: [] for family, _ in REQUESTED_FAMILIES}
    for path in discover_candidate_files(stage_root, worker_root, mac_aux_root):
        payload = maybe_read_json(path)
        if not isinstance(payload, dict):
            continue
        family = extract_family(payload, path)
        if family not in evidence:
            continue
        tier, rank, source_host, imported, retained_oof = classify_source(path, payload, family)
        if path.name not in {"resolved_performance_summary_v3.json", "r31_performance_summary.json", "family_execution_summary.json"}:
            rank += 25
        evidence[family].append(MetricEvidence(family, path, payload, tier, source_host, imported, retained_oof, rank))
    for family in evidence:
        evidence[family].sort(key=lambda item: (item.rank, -item.path.stat().st_mtime))
    return evidence


def accepted_metrics_root(progress: dict[str, Any] | None) -> str | None:
    if not progress:
        return None
    value = progress.get("metrics_root")
    return value if isinstance(value, str) and value else None


def load_progress(stage_root: Path, worker_root: Path, family: str) -> dict[str, Any] | None:
    candidates = [
        worker_root / family / "progress.json",
        worker_root / (family.lower().replace(" ", "_")) / "progress.json",
        worker_root / ALIASES.get(family.lower().replace(" ", "_"), family) / "progress.json",
    ]
    for path in candidates:
        payload = maybe_read_json(path)
        if isinstance(payload, dict):
            return payload
    return None


def choose_primary_evidence(
    family: str,
    items: list[MetricEvidence],
    progress: dict[str, Any] | None,
) -> tuple[MetricEvidence | None, list[dict[str, Any]]]:
    rejected: list[dict[str, Any]] = []
    preferred_root = accepted_metrics_root(progress)
    summaries = [item for item in items if item.path.name in {"resolved_performance_summary_v3.json", "r31_performance_summary.json", "family_execution_summary.json"}]
    if preferred_root:
        preferred = [
            item for item in summaries if item.path.name == "resolved_performance_summary_v3.json"
            and normalize_rel(item.path.parent).replace("\\", "/") == preferred_root.replace("\\", "/")
        ]
        if preferred:
            selected = preferred[0]
            rejected.extend(rejection_records(summaries, selected, "accepted live progress metrics_root has precedence"))
            return selected, rejected
    if summaries:
        selected = summaries[0]
        rejected.extend(rejection_records(summaries, selected, "stronger source tier or newer accepted authority selected"))
        return selected, rejected
    if items:
        selected = items[0]
        rejected.extend(rejection_records(items, selected, "no summary artifact available"))
        return selected, rejected
    return None, []


def rejection_records(items: list[MetricEvidence], selected: MetricEvidence, reason: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in items:
        if item.path == selected.path:
            continue
        records.append(
            {
                "path": normalize_rel(item.path),
                "artifact_type": item.path.name,
                "source_tier": item.source_tier,
                "why_rejected": reason,
            }
        )
    return records


def is_terminal(progress: dict[str, Any] | None, summary: dict[str, Any] | None) -> bool:
    if isinstance(progress, dict):
        if progress.get("terminal_reached") is True:
            return True
        last_completed = str(progress.get("last_completed_T") or "")
        if last_completed.startswith("2026-06-30"):
            return True
        return False
    if isinstance(summary, dict):
        pending = as_int(nested_get(summary, (("coverage", "pending_score_rows"), ("pending_score_rows",))))
        last_resolved = str(summary.get("last_resolved_decision_timestamp") or "")
        if pending == 0 and last_resolved.startswith("2026-06-30"):
            return True
        status = str(summary.get("status") or "").upper()
        return status in {"FINAL", "COMPLETE", "TERMINAL", "ACCEPTED_FINAL"}
    return False


def extract_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    rank_ic = payload.get("rank_ic") if isinstance(payload.get("rank_ic"), dict) else {}
    ci = rank_ic.get("dependence_aware_95_ci") if isinstance(rank_ic.get("dependence_aware_95_ci"), dict) else {}
    returns = payload.get("returns") if isinstance(payload.get("returns"), dict) else {}
    legacy_ic = payload.get("information_coefficient") if isinstance(payload.get("information_coefficient"), dict) else {}
    legacy_perf = nested_get(payload, (("portfolio_performance", "top_n_equal_weight"),))
    legacy_perf = legacy_perf if isinstance(legacy_perf, dict) else {}
    coverage = payload.get("coverage") if isinstance(payload.get("coverage"), dict) else {}
    return {
        "first_decision_timestamp": first_present(coverage.get("first_decision_timestamp"), payload.get("first_decision_timestamp")),
        "last_decision_timestamp": first_present(coverage.get("last_decision_timestamp"), payload.get("last_decision_timestamp")),
        "first_resolved_decision_timestamp": first_present(
            payload.get("first_resolved_decision_timestamp"),
            legacy_perf.get("first_resolved_decision_timestamp"),
        ),
        "last_resolved_decision_timestamp": first_present(
            payload.get("last_resolved_decision_timestamp"),
            legacy_perf.get("last_resolved_decision_timestamp"),
        ),
        "trading_days": first_present(returns.get("daily_return_rows"), legacy_perf.get("resolved_session_count")),
        "valid_rank_ic_timestamps": first_present(rank_ic.get("valid_timestamps"), legacy_ic.get("ic_observation_count")),
        "daily_result_rows": first_present(returns.get("daily_return_rows"), legacy_perf.get("resolved_session_count")),
        "resolved_performance_rows": first_present(
            payload.get("resolved_performance_rows"),
            legacy_perf.get("resolved_portfolio_observations"),
            payload.get("metric_rows"),
            payload.get("rank_ic_rows"),
        ),
        "eligible_coverage": coverage.get("eligible_resolved_fraction"),
        "terminal_censored_rows": first_present(coverage.get("terminal_censored_rows"), coverage.get("censored_terminal_count")),
        "pending_rows": first_present(coverage.get("pending_score_rows"), coverage.get("missing_prediction_count")),
        "mean_spearman_rank_ic": first_present(
            rank_ic.get("mean_spearman_rank_ic"),
            legacy_ic.get("mean_spearman_rank_ic"),
            payload.get("mean_spearman_rank_ic"),
        ),
        "median_spearman_rank_ic": first_present(rank_ic.get("median_spearman_rank_ic"), legacy_ic.get("median_spearman_rank_ic")),
        "daily_mean_spearman_rank_ic": rank_ic.get("daily_mean_spearman_rank_ic"),
        "positive_ic_fraction": first_present(rank_ic.get("positive_fraction"), legacy_ic.get("rank_ic_positive_fraction")),
        "hac_mean": ci.get("mean"),
        "hac_standard_error": ci.get("standard_error"),
        "hac_lower_95": ci.get("lower"),
        "hac_upper_95": ci.get("upper"),
        "hac_lag": ci.get("lag"),
        "inference_method": rank_ic.get("inference_method"),
        "pearson_ic": legacy_ic.get("pearson_ic"),
        "ndcg": first_present(payload.get("ndcg"), nested_get(payload, (("ranking_metrics", "ndcg"),))),
        "ndcg_at_20": first_present(payload.get("ndcg_at_20"), nested_get(payload, (("ranking_metrics", "ndcg_at_20"),))),
        "top_n_hit_rate": first_present(payload.get("top_n_hit_rate"), legacy_perf.get("hit_rate")),
        "directional_accuracy": nested_get(payload, (("ranking_metrics", "directional_accuracy"),)),
        "mean_hit_rate": first_present(nested_get(payload, (("ranking_metrics", "mean_hit_rate"),)), legacy_perf.get("hit_rate")),
        "score_population_rows": nested_get(payload, (("ranking_metrics", "score_population_rows"),)),
        "top_n_rows": first_present(payload.get("topn_rows"), payload.get("decision_rows")),
        "mean_daily_gross_return": returns.get("mean_daily_gross_return"),
        "mean_daily_net_return": returns.get("mean_daily_net_return"),
        "annualized_return": first_present(returns.get("annualized_return_from_daily_returns"), legacy_perf.get("annualized_return")),
        "geometric_annual_return": returns.get("geometric_annual_return"),
        "annualized_volatility": first_present(returns.get("annualized_volatility_from_daily_returns"), legacy_perf.get("annualized_volatility")),
        "sharpe": first_present(returns.get("daily_sharpe"), legacy_perf.get("sharpe_ratio")),
        "cumulative_gross_return": first_present(returns.get("cumulative_gross_return"), legacy_perf.get("cumulative_gross_return")),
        "cumulative_net_return": first_present(returns.get("cumulative_net_return"), legacy_perf.get("cumulative_net_return")),
        "maximum_drawdown": first_present(returns.get("maximum_drawdown"), legacy_perf.get("maximum_drawdown")),
        "win_rate": first_present(returns.get("win_rate"), legacy_perf.get("hit_rate")),
        "best_period": returns.get("best_period"),
        "worst_period": returns.get("worst_period"),
        "turnover": first_present(returns.get("mean_turnover"), legacy_perf.get("turnover")),
        "total_estimated_transaction_costs": first_present(returns.get("total_estimated_costs"), legacy_perf.get("estimated_transaction_costs")),
        "break_even_cost": returns.get("break_even_cost"),
        "sleeve_count": returns.get("sleeve_count"),
        "simultaneous_capital_limit": returns.get("simultaneous_capital_limit"),
        "portfolio_contract": returns.get("portfolio_contract"),
        "target_contract": payload.get("target_contract"),
    }


def cost_classification(metrics: dict[str, Any]) -> str:
    gross = as_float(metrics.get("cumulative_gross_return"))
    net = as_float(metrics.get("cumulative_net_return"))
    costs = as_float(metrics.get("total_estimated_transaction_costs"))
    if gross is None and net is None and costs is None:
        return "COST_INFORMATION_ABSENT"
    if costs == 0.0 or (gross is not None and net is not None and abs(gross - net) < 1e-12):
        return "ZERO_COST_ECONOMICS"
    if costs is not None and costs > 0:
        return "NONZERO_COSTS_APPLIED"
    return "COST_INFORMATION_ABSENT"


def warning_flags(row: dict[str, Any]) -> str:
    flags: list[str] = []
    rank_ic = as_float(row.get("mean_spearman_rank_ic"))
    sharpe = as_float(row.get("sharpe"))
    turnover = as_float(row.get("turnover"))
    if rank_ic is not None and abs(rank_ic) >= 0.4:
        flags.append("EXTREME_RANK_IC_REQUIRES_SCRUTINY")
    if sharpe is not None and abs(sharpe) >= 5:
        flags.append("EXTREME_SHARPE_REQUIRES_SCRUTINY")
    if row.get("economic_cost_classification") == "ZERO_COST_ECONOMICS" and turnover is not None and turnover > 0:
        flags.append("ZERO_COST_WITH_NONZERO_TURNOVER")
    if as_int(row.get("daily_result_rows")) is not None and as_int(row.get("daily_result_rows")) < 60:
        flags.append("SHORT_EVALUATION_WINDOW")
    if row.get("imported_result"):
        flags.append("IMPORTED_RESULT")
    if row.get("performance_status") == "LEGACY_LIMITED":
        flags.append("LEGACY_RESULT")
    if row.get("performance_status") == "PROVISIONAL_RUNNING":
        flags.append("PROVISIONAL_RESULT")
    if row.get("r53_result_state") == "EXTERNAL_MAC_RESULT_NOT_LOCALLY_AVAILABLE":
        flags.append("MAC_AUTHORITY_NOT_LOCAL")
    if row.get("r53_result_state") == "CONFIGURATION_AUTHORITY_REQUIRED":
        flags.append("CONFIGURATION_BLOCKED")
    return ";".join(dict.fromkeys(flags))


def classify_row(
    family: str,
    evidence: MetricEvidence | None,
    ownership: dict[str, Any],
    progress: dict[str, Any] | None,
) -> tuple[str, str, bool, str]:
    owner_state = str(ownership.get("owner_state") or ownership.get("ownership_state") or "")
    readiness = str(ownership.get("readiness_state") or "")
    complete_terminal = is_terminal(progress, evidence.payload if evidence else None)
    if family == "Temporal Fusion Transformer" and "CONFIGURATION_AUTHORITY_REQUIRED" in (owner_state + readiness):
        return "CONFIGURATION_AUTHORITY_REQUIRED", "BLOCKED", False, "NO_RESULT"
    if family == "lightgbm_lambdarank" and "MISSING" in json.dumps(ownership).upper():
        return "EXTERNAL_MAC_RESULT_NOT_LOCALLY_AVAILABLE", "MISSING", False, "NO_RESULT"
    if evidence is None:
        if "RUNNING" in owner_state or "RUNNING" in readiness:
            return "RUNNING_PROVISIONAL_RESULT", "PROVISIONAL_RUNNING", False, "PROVISIONAL_NOT_FINAL"
        if "READY" in readiness:
            return "READY_NOT_STARTED", "NOT_STARTED", False, "NO_RESULT"
        if "BLOCK" in owner_state + readiness:
            return "BLOCKED", "BLOCKED", False, "NO_RESULT"
        return "COMPLETE_RESULT_AUTHORITY_MISSING", "MISSING", False, "NO_RESULT"
    if evidence.source_tier == "E":
        return "COMPLETE_LEGACY_RESULT_LIMITED", "LEGACY_LIMITED", False, "LEGACY_NOT_DIRECTLY_COMPARABLE"
    if evidence.imported or owner_state == "COMPLETE_IMPORTED":
        return "COMPLETE_IMPORTED_RESULT_AVAILABLE", "ACCEPTED_IMPORTED_FINAL", True, "IMPORTED_RETAINED_OOF_DIFFERENT_WINDOW"
    if complete_terminal and evidence.source_tier == "A":
        return "COMPLETE_FINAL_RESULT_AVAILABLE", "ACCEPTED_FINAL", True, "DIRECTLY_COMPARABLE_V3_COMMON_WINDOW"
    if "RUNNING" in owner_state or "RUNNING" in readiness or not complete_terminal:
        return "RUNNING_PROVISIONAL_RESULT", "PROVISIONAL_RUNNING", False, "PROVISIONAL_NOT_FINAL"
    return "COMPLETE_RESULT_AUTHORITY_MISSING", "MISSING", False, "NO_RESULT"


def runtime_snapshot(stage_root: Path, worker_root: Path, ownership: dict[str, dict[str, Any]]) -> dict[str, Any]:
    processes = inspect_processes()
    supervisor_candidates = [p for p in processes if "supervisor" in str(p.get("command_line", "")).lower() and "ds24" in str(p.get("command_line", "")).lower()]
    worker_candidates = [p for p in processes if "ds24" in str(p.get("command_line", "")).lower() and "worker" in str(p.get("command_line", "")).lower()]
    rows: list[dict[str, Any]] = []
    for index, (family, _) in enumerate(REQUESTED_FAMILIES, start=1):
        progress = load_progress(stage_root, worker_root, family)
        current = {
            "family": family,
            "family_index": index,
            "current_cursor": first_present(
                progress.get("last_completed_T") if progress else None,
                progress.get("last_completed_refit_T") if progress else None,
            ),
            "metric_rows": progress.get("metric_rows") if progress else None,
            "resolved_rows": progress.get("resolved_performance_rows") if progress else None,
            "pending_rows": None,
            "terminal_flag": bool(is_terminal(progress, None)),
            "ownership_host": ownership.get(family, {}).get("execution_owner"),
            "dell_eligibility": ownership.get(family, {}).get("dell_eligible"),
            "mac_eligibility": ownership.get(family, {}).get("mac_eligible"),
            "runtime_state": ownership.get(family, {}).get("owner_state") or ownership.get(family, {}).get("readiness_state"),
        }
        rows.append(current)
    disk = shutil.disk_usage(stage_root if stage_root.exists() else Path.cwd())
    return {
        "timestamp_utc": utc_now(),
        "supervisor_pid": supervisor_candidates[0].get("pid") if supervisor_candidates else None,
        "supervisor_alive": bool(supervisor_candidates),
        "lease_owner": maybe_read_json(stage_root / "R7_R27_tournament_supervisor.lease.json"),
        "active_model_process_count": len(worker_candidates),
        "active_worker_names_pids": worker_candidates,
        "per_family_reconstructed_state": rows,
        "resource_state": {
            "disk_total_bytes": disk.total,
            "disk_used_bytes": disk.used,
            "disk_free_bytes": disk.free,
        },
        "holdout_flag": False,
        "paper_orders": 0,
        "live_orders": 0,
        "full_prediction_guard": {"full_prediction_files_generated_by_r53": 0},
        "workers_stopped_by_r53": 0,
        "workers_launched_by_r53": 0,
        "supervisor_restarted_by_r53": 0,
        "model_fits_caused_by_r53": 0,
        "predictions_generated_by_r53": 0,
        "outer_holdout_accessed": False,
    }


def inspect_processes() -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -match 'ds24|DS24|r7_r14|policy_worker|supervisor' } | "
        "Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine | ConvertTo-Json -Depth 3"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0 or not completed.stdout.strip():
        return []
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []
    items = payload if isinstance(payload, list) else [payload]
    return [
        {
            "pid": item.get("ProcessId"),
            "parent_pid": item.get("ParentProcessId"),
            "creation_time": item.get("CreationDate"),
            "executable": item.get("ExecutablePath"),
            "command_line": item.get("CommandLine"),
        }
        for item in items
        if isinstance(item, dict)
    ]


def build_rows(stage_root: Path, worker_root: Path, mac_aux_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ownership, ownership_sources = load_latest_ownership(stage_root)
    evidence_by_family = discover_metric_evidence(stage_root, worker_root, mac_aux_root)
    repo_population = repository_population(stage_root, worker_root, mac_aux_root)
    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "requested_19_family_population": [family for family, _ in REQUESTED_FAMILIES],
        "repository_authoritative_population": repo_population,
        "ownership_sources": ownership_sources,
        "families": {},
    }
    for index, (family, display_name) in enumerate(REQUESTED_FAMILIES, start=1):
        progress = load_progress(stage_root, worker_root, family)
        selected, rejected = choose_primary_evidence(family, evidence_by_family.get(family, []), progress)
        metrics = extract_metrics(selected.payload) if selected else {}
        owner = ownership.get(family, {})
        state, perf_status, final_result, comparability = classify_row(family, selected, owner, progress)
        cost_state = cost_classification(metrics) if selected else "NOT_APPLICABLE"
        family_has_retained_oof = any(item.retained_oof for item in evidence_by_family.get(family, []))
        row: dict[str, Any] = {
            "family": family,
            "display_name": display_name,
            "family_index": index,
            "execution_host": owner.get("execution_owner"),
            "ownership_state": owner.get("ownership_state") or owner.get("owner_state"),
            "current_runtime_state": owner.get("owner_state") or owner.get("readiness_state"),
            "r53_result_state": state,
            "performance_status": perf_status,
            "result_final": final_result,
            "result_provisional": perf_status == "PROVISIONAL_RUNNING",
            "terminal_reached": is_terminal(progress, selected.payload if selected else None),
            "terminal_cursor": progress.get("last_completed_T") if progress else None,
            "current_cursor": first_present(progress.get("last_completed_T") if progress else None, progress.get("last_completed_refit_T") if progress else None),
            "result_source_tier": selected.source_tier if selected else None,
            "result_source_path": normalize_rel(selected.path) if selected else None,
            "result_source_sha256": sha256_file(selected.path) if selected else None,
            "metrics_namespace": selected.namespace if selected else None,
            "evaluation_contract_id": selected.payload.get("evaluation_contract_id") if selected else None,
            "evaluation_contract_version": selected.payload.get("evaluation_contract_version") if selected else None,
            "evaluation_contract_hash": selected.payload.get("evaluation_contract_hash") if selected else None,
            "target_contract": metrics.get("target_contract"),
            "source_host": selected.source_host if selected else owner.get("execution_owner"),
            "imported_result": bool(selected.imported) if selected else owner.get("ownership_state") == "COMPLETE_IMPORTED",
            "retained_oof_result": bool(selected.retained_oof or family_has_retained_oof) if selected else family_has_retained_oof,
            "economic_cost_classification": cost_state,
            "comparability_classification": comparability,
            "same_target_contract": None,
            "same_evaluation_contract": selected.payload.get("evaluation_contract_hash") == "c5d24ca5ba4182d6e9080a7ca98b0c32a054bffe77a56fd09052de516cae84bc" if selected else None,
            "same_portfolio_contract": metrics.get("portfolio_contract") == "twelve_staggered_equal_capital_sleeves" if metrics.get("portfolio_contract") else None,
            "same_cost_assumption": cost_state == "ZERO_COST_ECONOMICS",
            "same_development_window": comparability == "DIRECTLY_COMPARABLE_V3_COMMON_WINDOW",
            "overlapping_window": bool(metrics.get("first_resolved_decision_timestamp") and metrics.get("last_resolved_decision_timestamp")),
            "full_common_window_available": comparability == "DIRECTLY_COMPARABLE_V3_COMMON_WINDOW",
            "imported_vs_native": "IMPORTED" if selected and selected.imported else "NATIVE_OR_NONE",
            "legacy_vs_v3": "V3" if selected and selected.source_tier in {"A", "C"} else ("LEGACY" if selected else "NO_RESULT"),
            "same_predictor_authority": None,
        }
        row.update(metrics)
        row["calendar_duration_days"] = days_between(row.get("first_resolved_decision_timestamp"), row.get("last_resolved_decision_timestamp"))
        row["warnings"] = warning_flags(row)
        rows.append(row)
        manifest["families"][family] = {
            "selected": evidence_record(selected, "selected highest-authority available source") if selected else None,
            "alternatives_rejected": rejected,
            "all_sources_used": [evidence_record(item, "candidate source inspected") for item in evidence_by_family.get(family, [])],
        }
    apply_final_window_comparability(rows)
    return rows, manifest


def apply_final_window_comparability(rows: list[dict[str, Any]]) -> None:
    final_v3 = [
        row
        for row in rows
        if row.get("performance_status") == "ACCEPTED_FINAL"
        and row.get("first_resolved_decision_timestamp")
        and row.get("last_resolved_decision_timestamp")
    ]
    if not final_v3:
        return
    windows: dict[tuple[Any, Any], int] = {}
    for row in final_v3:
        key = (row.get("first_resolved_decision_timestamp"), row.get("last_resolved_decision_timestamp"))
        windows[key] = windows.get(key, 0) + 1
    common_window = max(
        windows,
        key=lambda key: (
            windows[key],
            days_between(key[0], key[1]) or 0,
            str(key),
        ),
    )
    for row in final_v3:
        key = (row.get("first_resolved_decision_timestamp"), row.get("last_resolved_decision_timestamp"))
        if key == common_window:
            row["comparability_classification"] = "DIRECTLY_COMPARABLE_V3_COMMON_WINDOW"
            row["same_development_window"] = True
            row["full_common_window_available"] = True
        else:
            row["comparability_classification"] = "V3_DIFFERENT_EVALUATION_WINDOWS"
            row["same_development_window"] = False
            row["full_common_window_available"] = False
            flags = [flag for flag in str(row.get("warnings") or "").split(";") if flag]
            flags.append("DIFFERENT_WINDOW_FROM_RANKING_FAMILIES")
            row["warnings"] = ";".join(dict.fromkeys(flags))


def evidence_record(evidence: MetricEvidence | None, reason: str) -> dict[str, Any] | None:
    if evidence is None:
        return None
    stat = evidence.path.stat()
    return {
        "path": normalize_rel(evidence.path),
        "file_sha256": sha256_file(evidence.path),
        "artifact_type": evidence.path.name,
        "source_tier": evidence.source_tier,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "why_selected": reason,
    }


MATRIX_FIELDS = [
    "family",
    "display_name",
    "family_index",
    "execution_host",
    "ownership_state",
    "current_runtime_state",
    "r53_result_state",
    "performance_status",
    "result_final",
    "result_provisional",
    "terminal_reached",
    "terminal_cursor",
    "current_cursor",
    "result_source_tier",
    "result_source_path",
    "result_source_sha256",
    "metrics_namespace",
    "evaluation_contract_id",
    "evaluation_contract_version",
    "evaluation_contract_hash",
    "target_contract",
    "source_host",
    "imported_result",
    "retained_oof_result",
    "first_decision_timestamp",
    "last_decision_timestamp",
    "first_resolved_decision_timestamp",
    "last_resolved_decision_timestamp",
    "calendar_duration_days",
    "trading_days",
    "valid_rank_ic_timestamps",
    "daily_result_rows",
    "resolved_performance_rows",
    "eligible_coverage",
    "terminal_censored_rows",
    "pending_rows",
    "mean_spearman_rank_ic",
    "median_spearman_rank_ic",
    "daily_mean_spearman_rank_ic",
    "positive_ic_fraction",
    "hac_mean",
    "hac_standard_error",
    "hac_lower_95",
    "hac_upper_95",
    "hac_lag",
    "inference_method",
    "pearson_ic",
    "ndcg",
    "ndcg_at_20",
    "top_n_hit_rate",
    "directional_accuracy",
    "mean_hit_rate",
    "score_population_rows",
    "top_n_rows",
    "annualized_return",
    "geometric_annual_return",
    "annualized_volatility",
    "sharpe",
    "cumulative_gross_return",
    "cumulative_net_return",
    "maximum_drawdown",
    "win_rate",
    "turnover",
    "total_estimated_transaction_costs",
    "break_even_cost",
    "sleeve_count",
    "simultaneous_capital_limit",
    "portfolio_contract",
    "economic_cost_classification",
    "comparability_classification",
    "warnings",
    "same_target_contract",
    "same_evaluation_contract",
    "same_portfolio_contract",
    "same_cost_assumption",
    "same_development_window",
    "overlapping_window",
    "full_common_window_available",
    "imported_vs_native",
    "legacy_vs_v3",
    "same_predictor_authority",
]


def rank_rows(rows: list[dict[str, Any]], key: str, reverse: bool = True) -> list[dict[str, Any]]:
    eligible = [row for row in rows if as_float(row.get(key)) is not None]
    eligible.sort(key=lambda row: as_float(row.get(key)) or 0.0, reverse=reverse)
    result = []
    for rank, row in enumerate(eligible, start=1):
        copy = dict(row)
        copy["rank"] = rank
        result.append(copy)
    return result


def write_outputs(output_root: Path, rows: list[dict[str, Any]], manifest: dict[str, Any], pre_snapshot: dict[str, Any], post_snapshot: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root, "R53_pre_results_audit_runtime_snapshot.json", pre_snapshot)
    write_json(output_root, "R53_post_results_audit_runtime_snapshot.json", post_snapshot)
    write_csv(output_root, "R53_19_family_results_matrix.csv", rows, MATRIX_FIELDS)
    write_json(output_root, "R53_19_family_results_matrix.json", rows)
    final_rows = [row for row in rows if row.get("performance_status") in {"ACCEPTED_FINAL", "ACCEPTED_IMPORTED_FINAL"}]
    final_ranked = rank_rows(final_rows, "mean_spearman_rank_ic")
    write_csv(output_root, "R53_completed_final_results_leaderboard.csv", final_ranked, ["rank"] + MATRIX_FIELDS)
    rank_fields = [
        "rank",
        "family",
        "mean_spearman_rank_ic",
        "median_spearman_rank_ic",
        "hac_lower_95",
        "hac_upper_95",
        "positive_ic_fraction",
        "valid_rank_ic_timestamps",
        "first_resolved_decision_timestamp",
        "last_resolved_decision_timestamp",
        "calendar_duration_days",
        "comparability_classification",
        "performance_status",
    ]
    write_csv(output_root, "R53_rank_ic_leaderboard.csv", rank_rows(final_rows, "mean_spearman_rank_ic"), rank_fields)
    sharpe_fields = [
        "family",
        "sharpe",
        "annualized_return",
        "annualized_volatility",
        "cumulative_net_return",
        "maximum_drawdown",
        "win_rate",
        "turnover",
        "total_estimated_transaction_costs",
        "economic_cost_classification",
        "first_resolved_decision_timestamp",
        "last_resolved_decision_timestamp",
        "comparability_classification",
    ]
    write_csv(output_root, "R53_sharpe_leaderboard.csv", rank_rows(final_rows, "sharpe"), ["rank"] + sharpe_fields)
    provisional = [dict(row, snapshot_timestamp_utc=pre_snapshot["timestamp_utc"]) for row in rows if row.get("performance_status") == "PROVISIONAL_RUNNING"]
    write_csv(output_root, "R53_running_provisional_results.csv", provisional, ["snapshot_timestamp_utc"] + MATRIX_FIELDS)
    write_csv(output_root, "R53_common_window_readiness.csv", common_window_rows(rows), [
        "family",
        "retained_timestamp_level_metric_evidence_exists",
        "retained_daily_returns_exist",
        "retained_compact_oof_exists",
        "earliest_available_date",
        "latest_available_date",
        "common_window_reconstruction_possible_without_refitting",
        "blocker",
    ])
    write_csv(output_root, "R53_cross_host_results_matrix.csv", cross_host_rows(rows), [
        "family",
        "production_host",
        "current_owner",
        "completion_host",
        "dell_import_state",
        "final_result_locally_accessible",
        "local_authority_path",
        "source_authority_path",
        "duplicate_compute_risk",
        "notes",
    ])
    write_json(output_root, "R53_source_evidence_manifest.json", manifest)
    missing = missing_authority(rows)
    write_json(output_root, "R53_missing_result_authority.json", missing)
    write_text(output_root, "R53_19_FAMILY_TOURNAMENT_RESULTS_REPORT.md", render_report(rows, manifest, missing, pre_snapshot, post_snapshot))


def common_window_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        has_metrics = bool(row.get("result_source_path") and row.get("resolved_performance_rows"))
        has_daily = bool(row.get("daily_result_rows"))
        has_oof = bool(row.get("retained_oof_result") or "oof" in str(row.get("result_source_path")).lower())
        possible = has_metrics and has_daily and row.get("performance_status") in {"ACCEPTED_FINAL", "ACCEPTED_IMPORTED_FINAL", "PROVISIONAL_RUNNING"}
        result.append(
            {
                "family": row["family"],
                "retained_timestamp_level_metric_evidence_exists": has_metrics,
                "retained_daily_returns_exist": has_daily,
                "retained_compact_oof_exists": has_oof,
                "earliest_available_date": row.get("first_resolved_decision_timestamp"),
                "latest_available_date": row.get("last_resolved_decision_timestamp"),
                "common_window_reconstruction_possible_without_refitting": possible,
                "blocker": "" if possible else row.get("r53_result_state"),
            }
        )
    return result


def cross_host_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        imported = bool(row.get("imported_result"))
        if row["family"] == "lightgbm_lambdarank" and row.get("r53_result_state") == "EXTERNAL_MAC_RESULT_NOT_LOCALLY_AVAILABLE":
            dell_import_state = "MISSING_TRANSFER"
        elif imported and row.get("result_final"):
            dell_import_state = "IMPORTED"
        else:
            dell_import_state = "NATIVE_OR_NOT_APPLICABLE"
        result.append(
            {
                "family": row["family"],
                "production_host": row.get("source_host") or row.get("execution_host"),
                "current_owner": row.get("ownership_state"),
                "completion_host": row.get("source_host") if row.get("result_final") else "",
                "dell_import_state": dell_import_state,
                "final_result_locally_accessible": bool(row.get("result_final") and row.get("result_source_path")),
                "local_authority_path": row.get("result_source_path"),
                "source_authority_path": row.get("result_source_path"),
                "duplicate_compute_risk": "FORBIDDEN_FOR_MAC_OWNED_FAMILY" if row.get("execution_host") == "MAC" else "",
                "notes": row.get("r53_result_state"),
            }
        )
    return result


def missing_authority(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "generated_at_utc": utc_now(),
        "completed_family_with_missing_final_result": [row["family"] for row in rows if row["r53_result_state"] == "COMPLETE_RESULT_AUTHORITY_MISSING"],
        "mac_only_result_not_locally_present": [row["family"] for row in rows if row["r53_result_state"] == "EXTERNAL_MAC_RESULT_NOT_LOCALLY_AVAILABLE"],
        "running_result_only": [row["family"] for row in rows if row["performance_status"] == "PROVISIONAL_RUNNING"],
        "legacy_result_with_no_accepted_v3_authority": [row["family"] for row in rows if row["performance_status"] == "LEGACY_LIMITED"],
        "configuration_blocker": [row["family"] for row in rows if row["r53_result_state"] == "CONFIGURATION_AUTHORITY_REQUIRED"],
        "unavailable_cost_information": [row["family"] for row in rows if row["economic_cost_classification"] == "COST_INFORMATION_ABSENT"],
        "unavailable_evaluation_dates": [row["family"] for row in rows if not row.get("first_resolved_decision_timestamp") or not row.get("last_resolved_decision_timestamp")],
        "unavailable_common_window_evidence": [row["family"] for row in rows if not row.get("full_common_window_available")],
    }


def render_report(rows: list[dict[str, Any]], manifest: dict[str, Any], missing: dict[str, Any], pre: dict[str, Any], post: dict[str, Any]) -> str:
    counts = {status: sum(1 for row in rows if row["performance_status"] == status) for status in sorted({row["performance_status"] for row in rows})}
    final_rows = [row for row in rows if row["performance_status"] in {"ACCEPTED_FINAL", "ACCEPTED_IMPORTED_FINAL"}]
    lines = [
        "# DS24 R53 19-Family Tournament Results Report",
        "",
        "## Executive summary",
        "",
        f"- 19 requested families: {len(rows)} accounted for.",
        f"- Accepted final results: {counts.get('ACCEPTED_FINAL', 0)}.",
        f"- Accepted imported final results: {counts.get('ACCEPTED_IMPORTED_FINAL', 0)}.",
        f"- Legacy-limited complete results: {counts.get('LEGACY_LIMITED', 0)}.",
        f"- Running provisional results: {counts.get('PROVISIONAL_RUNNING', 0)}.",
        f"- Ready/not started: {counts.get('NOT_STARTED', 0)}.",
        f"- Blocked: {counts.get('BLOCKED', 0)}.",
        f"- Missing result authority: {counts.get('MISSING', 0)}.",
        "",
        "A high Rank IC does not prove live profitability. A high simulated Sharpe does not prove live tradability. Zero-cost economics are not genuine net-of-cost economics. Different evaluation periods are not directly comparable. Imported Mac results remain valid historical tournament evidence when accepted, but must be labelled as such. Provisional running results are not final. Legacy/quarantined evidence is not promotion authority.",
        "",
        "## Current tournament state",
        "",
    ]
    for row in rows:
        lines.append(f"- {row['family']}: {row['r53_result_state']} ({row['performance_status']})")
    lines.extend(["", "## Completed models", ""])
    for row in final_rows:
        lines.append(f"### {row['family']}")
        lines.append(f"- Source: {row.get('result_source_path')}")
        lines.append(f"- Rank IC: {row.get('mean_spearman_rank_ic')}; Sharpe: {row.get('sharpe')}; annual return: {row.get('annualized_return')}; max drawdown: {row.get('maximum_drawdown')}.")
        lines.append(f"- Evaluation window: {row.get('first_resolved_decision_timestamp')} to {row.get('last_resolved_decision_timestamp')}.")
    lines.extend(["", "## Running models", ""])
    for row in rows:
        if row["performance_status"] == "PROVISIONAL_RUNNING":
            lines.append(f"- {row['family']}: cursor {row.get('current_cursor')}, source {row.get('result_source_path')}")
    lines.extend(["", "## Rank-IC leaderboard", ""])
    for row in rank_rows(final_rows, "mean_spearman_rank_ic"):
        lines.append(f"- {row['rank']}. {row['family']}: {row.get('mean_spearman_rank_ic')} ({row.get('comparability_classification')})")
    lines.extend(["", "## Economic leaderboard", ""])
    for row in rank_rows(final_rows, "sharpe"):
        lines.append(f"- {row['rank']}. {row['family']}: Sharpe {row.get('sharpe')}, annual return {row.get('annualized_return')}, cost {row.get('economic_cost_classification')}")
    lines.extend(["", "## Mac vs Dell results", ""])
    for row in rows:
        if row.get("source_host") == "MAC" or row.get("execution_host") == "MAC":
            lines.append(f"- {row['family']}: owner {row.get('ownership_state')}, state {row.get('r53_result_state')}, source {row.get('result_source_path')}")
    lines.extend(["", "## Evaluation-window differences", ""])
    for row in rows:
        if row.get("result_source_path"):
            lines.append(f"- {row['family']}: {row.get('first_resolved_decision_timestamp')} to {row.get('last_resolved_decision_timestamp')} ({row.get('comparability_classification')})")
    lines.extend(["", "## Cost-model limitations", ""])
    for row in rows:
        if row.get("economic_cost_classification") in {"ZERO_COST_ECONOMICS", "COST_INFORMATION_ABSENT"}:
            lines.append(f"- {row['family']}: {row.get('economic_cost_classification')}; turnover {row.get('turnover')}")
    lines.extend(["", "## Results not directly comparable", ""])
    for row in rows:
        if row.get("comparability_classification") != "DIRECTLY_COMPARABLE_V3_COMMON_WINDOW":
            lines.append(f"- {row['family']}: {row.get('comparability_classification')}")
    lines.extend(["", "## Missing authorities", "", json.dumps(missing, indent=2, sort_keys=True), "", "## Key scientific cautions", ""])
    for row in rows:
        if row.get("warnings"):
            lines.append(f"- {row['family']}: {row.get('warnings')}")
    lines.extend(["", "## Exact source manifest", "", json.dumps(manifest, indent=2, sort_keys=True), "", "## Runtime safety", ""])
    lines.append(f"- Pre worker PIDs: {[p.get('pid') for p in pre.get('active_worker_names_pids', [])]}")
    lines.append(f"- Post worker PIDs: {[p.get('pid') for p in post.get('active_worker_names_pids', [])]}")
    lines.append("- R53 workers stopped: 0; workers launched: 0; supervisor restarted: 0; model fits: 0; predictions generated: 0; paper orders: 0; live orders: 0; outer holdout accessed: false.")
    return "\n".join(lines) + "\n"


def run(stage_root: Path, worker_root: Path, mac_aux_root: Path, output_root: Path) -> dict[str, Any]:
    ownership, _ = load_latest_ownership(stage_root)
    pre_snapshot = runtime_snapshot(stage_root, worker_root, ownership)
    rows, manifest = build_rows(stage_root, worker_root, mac_aux_root)
    post_snapshot = runtime_snapshot(stage_root, worker_root, ownership)
    write_outputs(output_root, rows, manifest, pre_snapshot, post_snapshot)
    counts = {status: sum(1 for row in rows if row["performance_status"] == status) for status in sorted({row["performance_status"] for row in rows})}
    classification = (
        "DS24_R53_TOURNAMENT_POPULATION_AUTHORITY_MISMATCH"
        if manifest["requested_19_family_population"] != manifest["repository_authoritative_population"]
        else (
            "DS24_R53_19_FAMILY_RESULTS_MATRIX_COMPLETE_WITH_MISSING_EXTERNAL_AUTHORITIES"
            if counts.get("MISSING", 0)
            else "DS24_R53_19_FAMILY_TOURNAMENT_RESULTS_AUTHORITY_COMPLETE"
        )
    )
    return {
        "classification": classification,
        "requested_families": len(rows),
        "accepted_final": counts.get("ACCEPTED_FINAL", 0),
        "accepted_imported_final": counts.get("ACCEPTED_IMPORTED_FINAL", 0),
        "legacy_limited": counts.get("LEGACY_LIMITED", 0),
        "running_provisional": counts.get("PROVISIONAL_RUNNING", 0),
        "ready_not_started": counts.get("NOT_STARTED", 0),
        "blocked": counts.get("BLOCKED", 0),
        "missing_result_authority": counts.get("MISSING", 0),
        "output_root": normalize_rel(output_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate DS24 R53 19-family tournament results.")
    parser.add_argument("--stage-root", type=Path, default=STAGE_ROOT)
    parser.add_argument("--worker-root", type=Path, default=WORKER_ROOT)
    parser.add_argument("--mac-aux-root", type=Path, default=MAC_AUX_ROOT)
    parser.add_argument("--output-root", type=Path, default=STAGE_ROOT / R53_DIRNAME)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable run summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args.stage_root, args.worker_root, args.mac_aux_root, args.output_root)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Requested families: {summary['requested_families']}")
        print(f"Accepted final: {summary['accepted_final']}")
        print(f"Accepted imported final: {summary['accepted_imported_final']}")
        print(f"Legacy limited: {summary['legacy_limited']}")
        print(f"Running provisional: {summary['running_provisional']}")
        print(f"Ready/not started: {summary['ready_not_started']}")
        print(f"Blocked: {summary['blocked']}")
        print(f"Missing result authority: {summary['missing_result_authority']}")
        print(f"Classification: {summary['classification']}")
        print(f"Output root: {summary['output_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
