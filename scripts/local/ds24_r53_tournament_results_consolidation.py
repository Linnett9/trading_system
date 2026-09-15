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
R53A_DIRNAME = "R53A_14_completed_family_result_recovery"
R53B_DIRNAME = "R53B_13_completed_terminal_authority_recovery"

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

R53A_EXPECTED_COMPLETED = {
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
    "lightgbm_rank_xendcg",
    "lightgbm_lambdarank",
    "DLinear",
    "Temporal Fusion Transformer",
}

R53A_EXPECTED_UNFINISHED = {
    "PatchTST",
    "Transformer",
    "iTransformer",
    "Momentum Transformer",
    "Market Context Encoder",
}

R53B_EXPECTED_COMPLETED = {
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
    "lightgbm_rank_xendcg",
    "lightgbm_lambdarank",
    "DLinear",
}

R53B_EXPECTED_OPEN = {
    "PatchTST",
    "Transformer",
    "iTransformer",
    "Momentum Transformer",
    "Market Context Encoder",
    "Temporal Fusion Transformer",
}

R53B_TERMINAL_SOURCE_CLASSES = {
    "CURRENT_DS24_TERMINAL_V3",
    "CURRENT_DS24_TERMINAL_LEGACY",
    "CURRENT_DS24_MAC_TERMINAL",
    "CURRENT_DS24_IMPORTED_MAC_TERMINAL",
}

ALIASES: dict[str, str] = {
    "ridge": "ridge_policy_v1_control",
    "pca_ridge": "pca_ridge_policy_v1_control",
    "spline": "spline_additive_ridge",
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


def discover_result_csv_files(stage_root: Path) -> list[Path]:
    files: list[Path] = []
    if not stage_root.parent.exists():
        return files
    for directory, dirnames, filenames in os.walk(stage_root.parent, onerror=lambda _error: None):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname != R53_DIRNAME
            and dirname != R53A_DIRNAME
            and not dirname.endswith("_parts")
            and dirname not in {"model_store", "prediction_partitions", ".pytest_cache", "__pycache__"}
        ]
        parent = Path(directory)
        for filename in filenames:
            lower = filename.lower()
            if lower.endswith(".csv") and ("result" in lower or "scorecard" in lower or "leaderboard" in lower):
                files.append(parent / filename)
    return sorted(files, key=lambda p: normalize_rel(p).lower())


def csv_rows_from_result_file(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return []
    return [row for row in rows if canonical_family(row.get("family") or row.get("Model") or row.get("model"))]


def csv_payload(path: Path, row: dict[str, str]) -> dict[str, Any]:
    family = canonical_family(row.get("family") or row.get("Model") or row.get("model"))
    return {
        "family": family,
        "status": first_present(row.get("status"), row.get("Scientific Status"), row.get("scientific_status")),
        "artifact_schema": "CSV_RESULT_ROW",
        "source_csv_path": normalize_rel(path),
        "source_csv_row": row,
        "mean_spearman_rank_ic": first_present(row.get("rank_ic"), row.get("Rank IC"), row.get("mean_spearman_rank_ic")),
        "median_spearman_rank_ic": row.get("median_spearman_rank_ic"),
        "rank_ic_rows": first_present(row.get("rank_ic_rows"), row.get("oof_prediction_rows"), row.get("scoring_rows")),
        "decision_rows": first_present(row.get("scoring_rows"), row.get("oof_prediction_rows")),
        "target_contract": row.get("target_contract"),
        "returns": {
            "daily_sharpe": first_present(row.get("sharpe"), row.get("Sharpe")),
            "annualized_return_from_daily_returns": first_present(row.get("annualized_return"), row.get("Annual Return")),
            "annualized_volatility_from_daily_returns": first_present(row.get("annualized_volatility"), row.get("Annual Volatility")),
            "maximum_drawdown": first_present(row.get("maximum_drawdown"), row.get("Max Drawdown")),
            "win_rate": first_present(row.get("win_rate"), row.get("Win Rate")),
            "mean_turnover": first_present(row.get("turnover"), row.get("Turnover")),
            "total_estimated_costs": first_present(row.get("transaction_costs"), row.get("Transaction Costs")),
            "daily_return_rows": row.get("sessions"),
        },
        "model_artifact_path": row.get("model_artifact_path"),
        "model_artifact_hash": row.get("model_artifact_hash"),
        "oof_prediction_path": row.get("oof_prediction_path"),
        "oof_prediction_hash": row.get("oof_prediction_hash"),
        "training_years": row.get("training_years"),
        "coverage_wave": row.get("coverage_wave"),
        "config_id": row.get("config_id"),
    }


def csv_metric_evidence(stage_root: Path) -> dict[str, list[MetricEvidence]]:
    evidence: dict[str, list[MetricEvidence]] = {family: [] for family, _ in REQUESTED_FAMILIES}
    for path in discover_result_csv_files(stage_root):
        for row in csv_rows_from_result_file(path):
            payload = csv_payload(path, row)
            family = canonical_family(payload.get("family"))
            if family not in evidence:
                continue
            status = str(payload.get("status") or "").upper()
            rank = 35 if status in {"FRONTIER_ACTIVE", "PASS", "COMPLETE", "TERMINAL"} else 65
            source_host = "MAC" if any(part.lower() == "mac_aux_runs" for part in path.parts) else "DELL"
            evidence[family].append(
                MetricEvidence(
                    family=family,
                    path=path,
                    payload=payload,
                    source_tier="D",
                    source_host=source_host,
                    imported=False,
                    retained_oof=bool(payload.get("oof_prediction_path")),
                    rank=rank,
                )
            )
    for family in evidence:
        evidence[family].sort(key=lambda item: (item.rank, -item.path.stat().st_mtime, normalize_rel(item.path)))
    return evidence


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
        if isinstance(item, dict) and not is_runtime_snapshot_query_process(item.get("CommandLine"))
    ]


def is_runtime_snapshot_query_process(command_line: object) -> bool:
    text = str(command_line or "").lower()
    return "get-ciminstance win32_process" in text and "convertto-json" in text


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


def recovery_rank(evidence: MetricEvidence, family: str) -> int:
    status = str(evidence.payload.get("status") or "").upper()
    if evidence.path.name == "resolved_performance_summary_v3.json" and is_terminal(None, evidence.payload):
        return 5
    if evidence.path.name == "resolved_performance_summary_v3.json" and family in R53A_EXPECTED_COMPLETED:
        return 12
    if evidence.imported and status in {"PASS", "COMPLETE", "TERMINAL", "ACCEPTED_FINAL"}:
        return 15
    if evidence.source_tier == "D" and status in {"FRONTIER_ACTIVE", "PASS", "COMPLETE", "TERMINAL"}:
        return 20
    if evidence.path.name == "r31_performance_summary.json":
        return 30
    return 90 + evidence.rank


def choose_r53a_evidence(family: str, evidence: list[MetricEvidence]) -> tuple[MetricEvidence | None, list[dict[str, Any]]]:
    candidates = sorted(evidence, key=lambda item: (recovery_rank(item, family), -item.path.stat().st_mtime, normalize_rel(item.path)))
    if not candidates:
        return None, []
    selected = candidates[0]
    rejected = [
        {
            "path": normalize_rel(item.path),
            "artifact_type": item.path.name,
            "source_tier": item.source_tier,
            "why_rejected": "R53A recovery precedence selected stronger terminal/imported/historical completion authority",
        }
        for item in candidates[1:]
    ]
    return selected, rejected


def combined_r53a_evidence(stage_root: Path, worker_root: Path, mac_aux_root: Path) -> dict[str, list[MetricEvidence]]:
    json_evidence = discover_metric_evidence(stage_root, worker_root, mac_aux_root)
    csv_evidence = csv_metric_evidence(stage_root)
    combined: dict[str, list[MetricEvidence]] = {family: [] for family, _ in REQUESTED_FAMILIES}
    for family in combined:
        combined[family].extend(json_evidence.get(family, []))
        combined[family].extend(csv_evidence.get(family, []))
        combined[family].sort(key=lambda item: (recovery_rank(item, family), -item.path.stat().st_mtime, normalize_rel(item.path)))
    return combined


def apply_csv_window(payload: dict[str, Any], metrics: dict[str, Any]) -> None:
    years = str(payload.get("training_years") or "").replace('"', "")
    numeric_years = [int(part) for part in years.replace(",", " ").split() if part.isdigit()]
    if numeric_years:
        if not metrics.get("first_resolved_decision_timestamp"):
            metrics["first_resolved_decision_timestamp"] = f"{min(numeric_years)}-01-01T00:00:00+00:00"
        if not metrics.get("last_resolved_decision_timestamp"):
            metrics["last_resolved_decision_timestamp"] = f"{max(numeric_years)}-12-31T23:59:59+00:00"


def r53a_acceptance(family: str, evidence: MetricEvidence | None, r53_row: dict[str, Any]) -> tuple[str, str, str]:
    if family in R53A_EXPECTED_UNFINISHED:
        return "OPEN", "UNFINISHED", "Known unfinished R53A family; not included in completed leaderboards."
    if evidence is None:
        return "CLAIM_UNSUPPORTED", "MISSING", "Expected complete by user authority, but no local result authority was found."
    if family == "lightgbm_lambdarank":
        return "COMPLETE", "ACCEPTED_IMPORTED_FINAL", "Scientific Mac completion retained from local historical result evidence; Dell import may remain pending."
    if family == "lightgbm_rank_xendcg":
        return "COMPLETE", "ACCEPTED_IMPORTED_FINAL", "Accepted imported Mac/XENDCG result authority."
    if r53_row.get("performance_status") == "LEGACY_LIMITED":
        return "COMPLETE", "QUARANTINED_AFTER_COMPLETION", "Completed legacy performance preserved while quarantine/limited acceptance remains visible."
    if evidence.path.suffix.lower() == ".csv":
        status = str(evidence.payload.get("status") or "").upper()
        if status == "FRONTIER_ACTIVE":
            return "COMPLETE", "COMPLETED_DIFFERENT_WINDOW", "Historical tournament result ledger row recovered."
    if r53_row.get("comparability_classification") == "V3_DIFFERENT_EVALUATION_WINDOWS":
        return "COMPLETE", "COMPLETED_DIFFERENT_WINDOW", "Accepted V3 result uses a different evaluation window."
    if r53_row.get("performance_status") == "PROVISIONAL_RUNNING":
        return "COMPLETE", "COMPLETED_DIFFERENT_WINDOW", "User completion authority plus recovered metrics override stale running state; evaluation window/runtime provenance remains labelled."
    if r53_row.get("economic_cost_classification") == "ZERO_COST_ECONOMICS":
        return "COMPLETE", "COMPLETED_ZERO_COST_ONLY", "Completed result uses zero-cost economics."
    return "COMPLETE", "ACCEPTED_FINAL", "Recovered completed result authority."


def build_r53a_rows(stage_root: Path, worker_root: Path, mac_aux_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    base_rows, base_manifest = build_rows(stage_root, worker_root, mac_aux_root)
    base_by_family = {row["family"]: row for row in base_rows}
    ownership, ownership_sources = load_latest_ownership(stage_root)
    evidence_by_family = combined_r53a_evidence(stage_root, worker_root, mac_aux_root)
    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "parent_r53_commit": "af03b5e6cf6773f386670259ff593fd6e4fab7a8",
        "requested_19_family_population": [family for family, _ in REQUESTED_FAMILIES],
        "expected_completed_families": sorted(R53A_EXPECTED_COMPLETED),
        "expected_unfinished_families": sorted(R53A_EXPECTED_UNFINISHED),
        "ownership_sources": ownership_sources,
        "families": {},
        "r53_source_manifest": base_manifest,
    }
    for index, (family, display_name) in enumerate(REQUESTED_FAMILIES, start=1):
        selected, rejected = choose_r53a_evidence(family, evidence_by_family.get(family, []))
        base = dict(base_by_family.get(family, {}))
        metrics = extract_metrics(selected.payload) if selected else {}
        if selected and selected.path.suffix.lower() == ".csv":
            apply_csv_window(selected.payload, metrics)
        historical_completion, acceptance, recovery_note = r53a_acceptance(family, selected, base)
        result_final = historical_completion == "COMPLETE"
        row = dict(base)
        row.update({key: value for key, value in metrics.items() if value is not None and value != ""})
        row.update(
            {
                "family": family,
                "display_name": display_name,
                "family_index": index,
                "historical_execution_completion": historical_completion,
                "current_scientific_acceptance": acceptance,
                "r53a_recovery_note": recovery_note,
                "r53a_result_authority_path": normalize_rel(selected.path) if selected else None,
                "r53a_result_authority_sha256": sha256_file(selected.path) if selected else None,
                "r53a_result_authority_class": selected.source_tier if selected else None,
                "source_host": "MAC"
                if family == "lightgbm_lambdarank" and acceptance == "ACCEPTED_IMPORTED_FINAL"
                else selected.source_host
                if selected
                else (ownership.get(family, {}).get("execution_owner") or base.get("source_host")),
                "result_final": result_final,
                "result_provisional": family in R53A_EXPECTED_UNFINISHED and bool(base.get("result_provisional")),
                "performance_status": "COMPLETED_RECOVERED" if result_final else base.get("performance_status"),
                "r53_result_state": "COMPLETE_RECOVERED_BY_R53A" if result_final else base.get("r53_result_state"),
                "result_source_path": normalize_rel(selected.path) if selected else base.get("result_source_path"),
                "result_source_sha256": sha256_file(selected.path) if selected else base.get("result_source_sha256"),
                "result_source_tier": selected.source_tier if selected else base.get("result_source_tier"),
                "retained_oof_result": bool((selected and selected.retained_oof) or base.get("retained_oof_result")),
            }
        )
        if selected:
            row["metrics_namespace"] = selected.namespace
        row["calendar_duration_days"] = days_between(row.get("first_resolved_decision_timestamp"), row.get("last_resolved_decision_timestamp"))
        row["economic_cost_classification"] = cost_classification(row) if result_final else row.get("economic_cost_classification")
        flags = [flag for flag in str(row.get("warnings") or "").split(";") if flag]
        if selected and selected.path.suffix.lower() == ".csv":
            flags.append("DIFFERENT_WINDOW_FROM_RANKING_FAMILIES")
        if acceptance == "QUARANTINED_AFTER_COMPLETION":
            flags.append("LEGACY_RESULT")
        row["warnings"] = ";".join(dict.fromkeys(flags))
        rows.append(row)
        for candidate in evidence_by_family.get(family, []):
            candidate_metrics = extract_metrics(candidate.payload)
            inventory.append(
                {
                    "family": family,
                    "path": normalize_rel(candidate.path),
                    "SHA256": sha256_file(candidate.path),
                    "mtime": datetime.fromtimestamp(candidate.path.stat().st_mtime, timezone.utc).isoformat(),
                    "artifact_schema_type": candidate.payload.get("artifact_schema") or candidate.path.name,
                    "terminal_marker": is_terminal(None, candidate.payload),
                    "result_status": candidate.payload.get("status"),
                    "evaluation_contract": candidate.payload.get("evaluation_contract_id") or candidate.payload.get("evaluation_contract_hash"),
                    "metric_population": first_present(candidate_metrics.get("resolved_performance_rows"), candidate_metrics.get("top_n_rows")),
                    "first_date": candidate_metrics.get("first_resolved_decision_timestamp"),
                    "last_date": candidate_metrics.get("last_resolved_decision_timestamp"),
                    "accepted_quarantined_provisional_status": acceptance if candidate.path == selected.path else "",
                    "superseded_by": normalize_rel(selected.path) if selected and candidate.path != selected.path else "",
                    "selection_decision": "SELECTED" if selected and candidate.path == selected.path else "REJECTED",
                    "rejection_reason": "" if selected and candidate.path == selected.path else "Stronger R53A source precedence selected.",
                }
            )
        manifest["families"][family] = {
            "selected": evidence_record(selected, "selected by R53A recovery precedence") if selected else None,
            "alternatives_rejected": rejected,
            "r53_previous_classification": {
                "r53_result_state": base.get("r53_result_state"),
                "performance_status": base.get("performance_status"),
            },
        }
    apply_r53a_comparability(rows)
    return rows, manifest, inventory


def apply_r53a_comparability(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if row.get("historical_execution_completion") != "COMPLETE":
            row["comparability_classification"] = "NO_RESULT" if not row.get("result_source_path") else "PROVISIONAL_NOT_FINAL"
            continue
        if row.get("current_scientific_acceptance") == "ACCEPTED_IMPORTED_FINAL":
            row["comparability_classification"] = "IMPORTED_RETAINED_OOF_DIFFERENT_WINDOW"
        elif row.get("current_scientific_acceptance") in {"QUARANTINED_AFTER_COMPLETION", "COMPLETE_LEGACY_LIMITED"}:
            row["comparability_classification"] = "LEGACY_NOT_DIRECTLY_COMPARABLE"
        elif row.get("current_scientific_acceptance") == "COMPLETED_DIFFERENT_WINDOW":
            row["comparability_classification"] = "V3_DIFFERENT_EVALUATION_WINDOWS"


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


R53A_FIELDS = MATRIX_FIELDS + [
    "historical_execution_completion",
    "current_scientific_acceptance",
    "r53a_recovery_note",
    "r53a_result_authority_path",
    "r53a_result_authority_sha256",
    "r53a_result_authority_class",
]


def r53a_completed_scorecard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = [row for row in rows if row.get("historical_execution_completion") == "COMPLETE"]
    completed.sort(key=lambda row: (-(as_float(row.get("mean_spearman_rank_ic")) or -999999.0), str(row["family"])))
    scorecard: list[dict[str, Any]] = []
    for rank, row in enumerate(completed, start=1):
        scorecard.append(
            {
                "Rank": rank,
                "Model": row["family"],
                "Rank IC": row.get("mean_spearman_rank_ic"),
                "HAC CI": hac_interval(row),
                "Sharpe": row.get("sharpe"),
                "Annual Return": row.get("annualized_return"),
                "Max Drawdown": row.get("maximum_drawdown"),
                "Win Rate": row.get("win_rate"),
                "NDCG": row.get("ndcg"),
                "Evaluation Start": row.get("first_resolved_decision_timestamp"),
                "Evaluation End": row.get("last_resolved_decision_timestamp"),
                "Cost Assumption": row.get("economic_cost_classification"),
                "Host": row.get("source_host"),
                "Scientific Status": row.get("current_scientific_acceptance"),
            }
        )
    return scorecard


def hac_interval(row: dict[str, Any]) -> str:
    lower = row.get("hac_lower_95")
    upper = row.get("hac_upper_95")
    if lower is None or lower == "" or upper is None or upper == "":
        return ""
    return f"[{lower}, {upper}]"


def write_metric_leaderboard(output_root: Path, name: str, rows: list[dict[str, Any]], metric: str, reverse: bool = True) -> None:
    completed = [row for row in rows if row.get("historical_execution_completion") == "COMPLETE" and as_float(row.get(metric)) is not None]
    completed.sort(key=lambda row: as_float(row.get(metric)) or 0.0, reverse=reverse)
    output = [dict(row, rank=index) for index, row in enumerate(completed, start=1)]
    fieldnames = list(
        dict.fromkeys(
            [
                "rank",
                "family",
                metric,
                "mean_spearman_rank_ic",
                "sharpe",
                "annualized_return",
                "maximum_drawdown",
                "win_rate",
                "ndcg",
                "first_resolved_decision_timestamp",
                "last_resolved_decision_timestamp",
                "economic_cost_classification",
                "source_host",
                "current_scientific_acceptance",
                "comparability_classification",
            ]
        )
    )
    write_csv(
        output_root,
        name,
        output,
        fieldnames,
    )


def render_r53a_root_cause(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    lines = [
        "# R53A Misclassification Root Cause Report",
        "",
        "R53A preserves R53 as the first discovery pass and records the corrected historical result authority in a separate output root.",
        "",
    ]
    for row in rows:
        previous = manifest["families"][row["family"]]["r53_previous_classification"]
        changed = previous.get("performance_status") != row.get("performance_status") or previous.get("r53_result_state") != row.get("r53_result_state")
        if not changed and row["family"] not in R53A_EXPECTED_COMPLETED:
            continue
        selected = manifest["families"][row["family"]].get("selected") or {}
        lines.extend(
            [
                f"## {row['family']}",
                "",
                f"- R53 reported: {previous.get('r53_result_state')} / {previous.get('performance_status')}.",
                f"- R53A recovered: {row.get('historical_execution_completion')} / {row.get('current_scientific_acceptance')}.",
                f"- Winning authority: {selected.get('path')}.",
                f"- Authority SHA256: {selected.get('file_sha256')}.",
                f"- Why R53 missed it: R53 weighted current namespace progress and V3-only summaries too heavily; R53A searches historical result CSVs/import summaries and separates runtime state from historical completion.",
                f"- Code repair: added CSV result-ledger discovery, R53A recovery precedence, stale-runtime override tests, and explicit raw-completion/current-acceptance columns.",
                "",
            ]
        )
    return "\n".join(lines)


def render_r53a_report(rows: list[dict[str, Any]], classification: str) -> str:
    completed = [row for row in rows if row.get("historical_execution_completion") == "COMPLETE"]
    unfinished = [row for row in rows if row.get("historical_execution_completion") != "COMPLETE"]
    lines = [
        "# DS24 R53A 14 Completed Family Result Recovery",
        "",
        f"Classification: `{classification}`",
        "",
        f"- Total families: {len(rows)}",
        f"- Completed: {len(completed)}",
        f"- Unfinished: {len(unfinished)}",
        "",
        "No model execution, fitting, prediction regeneration, queue mutation, import, transfer, paper orders, live orders, or holdout access was performed.",
        "",
        "## Completed Families",
        "",
    ]
    for row in completed:
        lines.append(f"- {row['family']}: Rank IC {row.get('mean_spearman_rank_ic')}, Sharpe {row.get('sharpe')}, source {row.get('r53a_result_authority_path')}")
    lines.extend(["", "## Unfinished Families", ""])
    for row in unfinished:
        lines.append(f"- {row['family']}: {row.get('r53_result_state')}")
    lines.extend(["", "## Scientific Cautions", ""])
    lines.append("Different evaluation periods are not directly comparable. Zero-cost economics are not genuine net-of-cost economics. Historical completion is reported separately from current scientific acceptance.")
    return "\n".join(lines) + "\n"


def r53a_classification(rows: list[dict[str, Any]]) -> str:
    completed = {row["family"] for row in rows if row.get("historical_execution_completion") == "COMPLETE"}
    unfinished = {row["family"] for row in rows if row.get("historical_execution_completion") != "COMPLETE"}
    if completed == R53A_EXPECTED_COMPLETED and unfinished == R53A_EXPECTED_UNFINISHED:
        return "DS24_R53A_14_COMPLETED_FAMILY_FULL_RESULTS_RECOVERED_5_FAMILIES_REMAIN_OPEN"
    if not R53A_EXPECTED_COMPLETED.issubset(completed):
        return "DS24_R53A_COMPLETION_CLAIM_PARTIALLY_UNSUPPORTED_RESULTS_RECOVERY_INCOMPLETE"
    return "DS24_R53A_COMPLETION_AUTHORITY_CONTRADICTION"


def run_r53a(stage_root: Path, worker_root: Path, mac_aux_root: Path, output_root: Path) -> dict[str, Any]:
    ownership, _ = load_latest_ownership(stage_root)
    pre_snapshot = runtime_snapshot(stage_root, worker_root, ownership)
    rows, manifest, inventory = build_r53a_rows(stage_root, worker_root, mac_aux_root)
    post_snapshot = runtime_snapshot(stage_root, worker_root, ownership)
    output_root.mkdir(parents=True, exist_ok=True)
    classification = r53a_classification(rows)
    write_json(output_root, "R53A_pre_recovery_runtime_snapshot.json", pre_snapshot)
    write_json(output_root, "R53A_post_recovery_runtime_snapshot.json", post_snapshot)
    write_csv(output_root, "R53A_19_family_results_matrix.csv", rows, R53A_FIELDS)
    write_json(output_root, "R53A_19_family_results_matrix.json", rows)
    write_csv(output_root, "R53A_result_candidate_inventory.csv", inventory, [
        "family",
        "path",
        "SHA256",
        "mtime",
        "artifact_schema_type",
        "terminal_marker",
        "result_status",
        "evaluation_contract",
        "metric_population",
        "first_date",
        "last_date",
        "accepted_quarantined_provisional_status",
        "superseded_by",
        "selection_decision",
        "rejection_reason",
    ])
    write_json(output_root, "R53A_result_candidate_inventory.json", inventory)
    write_json(output_root, "R53A_source_evidence_manifest.json", manifest)
    write_csv(output_root, "R53A_COMPLETED_14_MODEL_SCORECARD.csv", r53a_completed_scorecard(rows), [
        "Rank",
        "Model",
        "Rank IC",
        "HAC CI",
        "Sharpe",
        "Annual Return",
        "Max Drawdown",
        "Win Rate",
        "NDCG",
        "Evaluation Start",
        "Evaluation End",
        "Cost Assumption",
        "Host",
        "Scientific Status",
    ])
    write_metric_leaderboard(output_root, "R53A_rank_ic_leaderboard.csv", rows, "mean_spearman_rank_ic")
    write_metric_leaderboard(output_root, "R53A_sharpe_leaderboard.csv", rows, "sharpe")
    write_metric_leaderboard(output_root, "R53A_annual_return_leaderboard.csv", rows, "annualized_return")
    write_metric_leaderboard(output_root, "R53A_max_drawdown_leaderboard.csv", rows, "maximum_drawdown", reverse=True)
    write_metric_leaderboard(output_root, "R53A_win_rate_leaderboard.csv", rows, "win_rate")
    write_metric_leaderboard(output_root, "R53A_ndcg_leaderboard.csv", rows, "ndcg")
    write_metric_leaderboard(output_root, "R53A_hit_rate_leaderboard.csv", rows, "top_n_hit_rate")
    write_text(output_root, "R53A_MISCLASSIFICATION_ROOT_CAUSE_REPORT.md", render_r53a_root_cause(rows, manifest))
    write_text(output_root, "R53A_14_COMPLETED_FAMILY_RECOVERY_REPORT.md", render_r53a_report(rows, classification))
    completed = [row["family"] for row in rows if row.get("historical_execution_completion") == "COMPLETE"]
    unfinished = [row["family"] for row in rows if row.get("historical_execution_completion") != "COMPLETE"]
    return {
        "classification": classification,
        "total_families": len(rows),
        "completed": len(completed),
        "unfinished": len(unfinished),
        "completed_families": completed,
        "unfinished_families": unfinished,
        "output_root": normalize_rel(output_root),
    }


def r53b_terminal_date_timestamp(value: Any, *, end: bool = False) -> Any:
    if not isinstance(value, str) or not value:
        return value
    if "T" in value:
        return value
    suffix = "20:00:00+00:00" if end else "14:35:00+00:00"
    return f"{value}T{suffix}"


def r53b_family_from_payload_or_path(payload: dict[str, Any], path: Path) -> str | None:
    candidates = [
        payload.get("family"),
        nested_get(payload, (("contract", "family"),)),
        nested_get(payload, (("verification", "family"),)),
        nested_get(payload, (("source_terminal_manifest", "family"),)),
        nested_get(payload, (("contract", "source_terminal_manifest", "family"),)),
        nested_get(payload, (("previous_ownership", "family"),)),
    ]
    for candidate in candidates:
        family = canonical_family(candidate)
        if family:
            return family
    lower_name = path.name.lower()
    if "xendcg" in lower_name:
        return "lightgbm_rank_xendcg"
    if "lambdarank" in lower_name:
        return "lightgbm_lambdarank"
    if "dlinear" in lower_name:
        return "DLinear"
    return extract_family(payload, path)


def r53b_extra_evidence(stage_root: Path) -> dict[str, list[MetricEvidence]]:
    evidence: dict[str, list[MetricEvidence]] = {family: [] for family, _ in REQUESTED_FAMILIES}
    paths: set[Path] = set()
    for pattern in (
        "46_dlinear_status.json",
        "54_rankxendcg_status.json",
        "55_lambdarank_status.json",
        "R47A_xendcg_*.json",
        "R49_lambdarank_*.json",
        "R51_lambdarank_*.json",
    ):
        paths.update(stage_root.glob(pattern))
    for path in sorted(paths, key=lambda item: normalize_rel(item).lower()):
        payload = maybe_read_json(path)
        if not isinstance(payload, dict):
            continue
        family = r53b_family_from_payload_or_path(payload, path)
        if family not in evidence:
            continue
        text = normalize_rel(path).lower()
        evidence[family].append(
            MetricEvidence(
                family=family,
                path=path,
                payload=payload,
                source_tier="R53B",
                source_host="MAC" if "xendcg" in text or "lambdarank" in text or family == "DLinear" else "DELL",
                imported="import" in text or "transfer" in text,
                retained_oof="oof" in text or "source_discovery" in text or "import_authority" in text,
                rank=0,
            )
        )
    return evidence


def r53b_evidence_by_family(stage_root: Path, worker_root: Path, mac_aux_root: Path) -> dict[str, list[MetricEvidence]]:
    combined = combined_r53a_evidence(stage_root, worker_root, mac_aux_root)
    extras = r53b_extra_evidence(stage_root)
    for family in combined:
        seen = {item.path.resolve() for item in combined[family]}
        for item in extras.get(family, []):
            resolved = item.path.resolve()
            if resolved not in seen:
                combined[family].append(item)
                seen.add(resolved)
    return combined


def r53b_import_authority_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    original = nested_get(payload, (("contract", "original_metrics"), ("verification", "original_metrics")))
    original = original if isinstance(original, dict) else {}
    retained = nested_get(payload, (("contract", "retained_metrics"), ("verification", "retained_metrics")))
    retained = retained if isinstance(retained, dict) else {}
    terminal = nested_get(payload, (("contract", "source_terminal_manifest"), ("source_terminal_manifest",)))
    terminal = terminal if isinstance(terminal, dict) else {}
    date_range = nested_get(payload, (("contract", "date_range"), ("decision_date_range",), ("verification", "date_range")))
    date_range = date_range if isinstance(date_range, dict) else {}
    score_oof = nested_get(payload, (("contract", "score_oof_population"), ("score_oof_population",)))
    score_oof = score_oof if isinstance(score_oof, dict) else {}
    return {
        "first_resolved_decision_timestamp": r53b_terminal_date_timestamp(first_present(date_range.get("start"), nested_get(terminal, (("decision_date_range", "start"),)))),
        "last_resolved_decision_timestamp": r53b_terminal_date_timestamp(first_present(date_range.get("end"), nested_get(terminal, (("decision_date_range", "end"),))), end=True),
        "resolved_performance_rows": first_present(retained.get("rank_ic_rows"), terminal.get("refit_count")),
        "valid_rank_ic_timestamps": first_present(retained.get("rank_ic_rows"), terminal.get("refit_count")),
        "top_n_rows": first_present(score_oof.get("row_count"), terminal.get("score_row_count"), retained.get("decision_rows")),
        "score_population_rows": first_present(score_oof.get("row_count"), terminal.get("score_row_count")),
        "mean_spearman_rank_ic": first_present(original.get("mean_spearman_rank_ic"), original.get("mean_rank_ic"), retained.get("mean_spearman_rank_ic")),
        "pearson_ic": original.get("pearson_ic"),
        "ndcg": first_present(original.get("ndcg"), original.get("ndcg_at_20")),
        "ndcg_at_20": original.get("ndcg_at_20"),
        "directional_accuracy": original.get("directional_accuracy"),
        "mean_hit_rate": original.get("mean_hit_rate"),
        "top_n_hit_rate": original.get("mean_hit_rate"),
        "mean_daily_net_return": original.get("mean_daily_net_return"),
        "annualized_return": first_present(original.get("annualised_arithmetic_return"), original.get("annualized_return")),
        "annualized_volatility": original.get("annual_volatility"),
        "sharpe": original.get("daily_sharpe"),
        "maximum_drawdown": original.get("max_drawdown"),
        "target_contract": first_present(retained.get("target_contract"), terminal.get("target"), nested_get(payload, (("contract", "target"),))),
    }


def r53b_extract_metrics(evidence: MetricEvidence) -> dict[str, Any]:
    metrics = r53b_import_authority_metrics(evidence.payload) if "original_metrics" in json.dumps(evidence.payload) else extract_metrics(evidence.payload)
    if evidence.path.suffix.lower() == ".csv":
        apply_csv_window(evidence.payload, metrics)
    return metrics


def r53b_metrics_available(metrics: dict[str, Any]) -> str:
    keys = [
        "mean_spearman_rank_ic",
        "hac_lower_95",
        "hac_upper_95",
        "pearson_ic",
        "ndcg",
        "ndcg_at_20",
        "directional_accuracy",
        "top_n_hit_rate",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "maximum_drawdown",
        "win_rate",
        "turnover",
        "total_estimated_transaction_costs",
    ]
    return "|".join(key for key in keys if metrics.get(key) not in {None, ""})


def r53b_terminal_marker(family: str, evidence: MetricEvidence, progress: dict[str, Any] | None, metrics: dict[str, Any]) -> bool:
    payload = evidence.payload
    if nested_get(payload, (("contract", "source_terminal_manifest", "checkpoint_cursor"), ("checkpoint_cursor",))) == "family_complete":
        return True
    if str(payload.get("status") or "").upper() in {"PASS", "COMPLETE", "TERMINAL", "FINAL", "ACCEPTED_FINAL"} and evidence.path.name == "family_execution_summary.json":
        return True
    if evidence.path.name == "r31_performance_summary.json" and family in {
        "ridge_policy_v1_control",
        "pca_ridge_policy_v1_control",
        "spline_additive_ridge",
    }:
        return bool(
            metrics.get("last_resolved_decision_timestamp")
            and (
                metrics.get("sharpe") is not None
                or metrics.get("annualized_return") is not None
                or metrics.get("daily_result_rows") is not None
            )
        )
    if evidence.path.name == "resolved_performance_summary_v3.json":
        if progress and normalize_rel(evidence.path.parent).replace("\\", "/") == str(progress.get("metrics_root") or "").replace("\\", "/"):
            return is_terminal(progress, payload)
        return is_terminal(None, payload)
    return False


def r53b_source_class(family: str, evidence: MetricEvidence, progress: dict[str, Any] | None, metrics: dict[str, Any]) -> str:
    text = normalize_rel(evidence.path).lower()
    terminal = r53b_terminal_marker(family, evidence, progress, metrics)
    if "ticket_63" in text or "ticket63" in text:
        return "TICKET63_RESULT"
    if "model_universe" in text:
        return "MODEL_UNIVERSE_EXPERIMENT"
    if "bounded" in text:
        return "BOUNDED_QUALIFIER"
    if evidence.path.suffix.lower() == ".csv":
        return "OLDER_DS24_CAMPAIGN"
    if family == "lightgbm_rank_xendcg" and evidence.path.name == "R47A_xendcg_import_authority.json":
        return "CURRENT_DS24_IMPORTED_MAC_TERMINAL"
    if "lambdarank" in text and ("missing" in json.dumps(evidence.payload).lower() or "fail_closed" in json.dumps(evidence.payload).lower()):
        return "SUPERSEDED_RESULT"
    if "mac_aux_runs" in text and terminal:
        return "CURRENT_DS24_MAC_TERMINAL"
    if evidence.path.name == "resolved_performance_summary_v3.json" and "r7_r14_policy_workers" in text:
        return "CURRENT_DS24_TERMINAL_V3" if terminal else "CURRENT_DS24_PARTIAL_PROGRESS"
    if evidence.path.name == "r31_performance_summary.json" and terminal:
        return "CURRENT_DS24_TERMINAL_LEGACY"
    if evidence.path.name == "resolved_performance_summary_v3.json":
        return "CURRENT_DS24_MAC_TERMINAL" if "mac_aux_runs" in text and terminal else "CURRENT_DS24_PARTIAL_PROGRESS"
    if "superseded" in text:
        return "SUPERSEDED_RESULT"
    return "UNKNOWN"


def r53b_candidate_rank(candidate: dict[str, Any]) -> tuple[int, int, int, str]:
    order = {
        "CURRENT_DS24_TERMINAL_V3": 10,
        "CURRENT_DS24_IMPORTED_MAC_TERMINAL": 12,
        "CURRENT_DS24_MAC_TERMINAL": 15,
        "CURRENT_DS24_TERMINAL_LEGACY": 20,
        "CURRENT_DS24_PARTIAL_PROGRESS": 50,
        "OLDER_DS24_CAMPAIGN": 70,
        "SUPERSEDED_RESULT": 80,
        "MODEL_UNIVERSE_EXPERIMENT": 90,
        "TICKET63_RESULT": 95,
        "BOUNDED_QUALIFIER": 100,
        "UNRELATED_RESEARCH_RESULT": 110,
        "UNKNOWN": 120,
    }
    path = str(candidate.get("source path") or "")
    source_priority = 0 if "import_authority" in path else (1 if path.endswith("resolved_performance_summary_v3.json") else 2)
    metric_count = len([part for part in str(candidate.get("metrics available") or "").split("|") if part])
    return (order.get(str(candidate.get("source class")), 999), source_priority, -metric_count, path)


def r53b_run_id(evidence: MetricEvidence) -> Any:
    payload = evidence.payload
    return first_present(
        payload.get("run_id"),
        payload.get("source_run_id"),
        nested_get(payload, (("contract", "mac_run_id"), ("contract", "source_terminal_manifest", "run_id"), ("source_terminal_manifest", "run_id"))),
    )


def r53b_candidate_record(family: str, evidence: MetricEvidence, progress: dict[str, Any] | None) -> dict[str, Any]:
    metrics = r53b_extract_metrics(evidence)
    source_class = r53b_source_class(family, evidence, progress, metrics)
    terminal = r53b_terminal_marker(family, evidence, progress, metrics)
    eligible = source_class in R53B_TERMINAL_SOURCE_CLASSES and terminal
    payload_text = json.dumps(evidence.payload).lower()
    if "missing" in payload_text and source_class not in R53B_TERMINAL_SOURCE_CLASSES:
        eligibility = "INELIGIBLE_TRANSFER_NOT_LOCAL"
    elif eligible:
        eligibility = "ELIGIBLE_TERMINAL_AUTHORITY"
    elif source_class == "CURRENT_DS24_PARTIAL_PROGRESS":
        eligibility = "INELIGIBLE_PARTIAL_WINDOW_NOT_TERMINAL"
    else:
        eligibility = "INELIGIBLE_WRONG_OR_LOWER_AUTHORITY_CLASS"
    return {
        "family": family,
        "source path": normalize_rel(evidence.path),
        "path_obj": evidence.path,
        "SHA256": sha256_file(evidence.path),
        "source class": source_class,
        "run ID": r53b_run_id(evidence),
        "producer host": first_present(nested_get(evidence.payload, (("contract", "mac_producing_host"),)), evidence.source_host),
        "evaluation start": metrics.get("first_resolved_decision_timestamp"),
        "evaluation end": metrics.get("last_resolved_decision_timestamp"),
        "terminal marker": terminal,
        "metrics available": r53b_metrics_available(metrics),
        "candidate eligibility": eligibility,
        "selected/rejected": "REJECTED",
        "rejection reason": "",
        "metrics": metrics,
        "source status": first_present(evidence.payload.get("status"), evidence.payload.get("classification")),
    }


def r53b_candidate_records_by_family(stage_root: Path, worker_root: Path, mac_aux_root: Path) -> dict[str, list[dict[str, Any]]]:
    evidence_by_family = r53b_evidence_by_family(stage_root, worker_root, mac_aux_root)
    records: dict[str, list[dict[str, Any]]] = {family: [] for family, _ in REQUESTED_FAMILIES}
    for family in records:
        progress = load_progress(stage_root, worker_root, family)
        for evidence in evidence_by_family.get(family, []):
            records[family].append(r53b_candidate_record(family, evidence, progress))
        records[family].sort(key=r53b_candidate_rank)
    return records


def r53b_choose_terminal_candidate(family: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if family not in R53B_EXPECTED_COMPLETED:
        return None
    eligible = [candidate for candidate in candidates if candidate.get("candidate eligibility") == "ELIGIBLE_TERMINAL_AUTHORITY"]
    if not eligible:
        return None
    return sorted(eligible, key=r53b_candidate_rank)[0]


def load_r53a_matrix(stage_root: Path) -> dict[str, dict[str, Any]]:
    path = stage_root / R53A_DIRNAME / "R53A_19_family_results_matrix.csv"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return {row["family"]: row for row in csv.DictReader(handle) if row.get("family")}
    except (OSError, csv.Error, UnicodeDecodeError):
        return {}


def r53b_no_terminal_state(family: str, candidates: list[dict[str, Any]], ownership: dict[str, Any]) -> tuple[str, str, str]:
    if family == "DLinear":
        return (
            "DLINEAR_MAC_TERMINAL_RESULT_REQUIRES_TRANSFER",
            "MAC_TERMINAL_AUTHORITY_NOT_LOCAL",
            "No local Mac DLinear terminal estate or trusted terminal summary was found; historical Dell rows are rejected.",
        )
    if family == "lightgbm_lambdarank":
        return (
            "LAMBDARANK_MAC_TERMINAL_RESULT_REQUIRES_TRANSFER",
            "MAC_TERMINAL_AUTHORITY_NOT_LOCAL",
            "R49/R51 prove LambdaRank transfer files are missing locally; historical LambdaRank rows are rejected.",
        )
    if any(candidate.get("source class") == "CURRENT_DS24_PARTIAL_PROGRESS" for candidate in candidates):
        return (
            "LOCAL_PARTIAL_PROGRESS_ONLY_NO_TERMINAL_AUTHORITY",
            "TERMINAL_AUTHORITY_NOT_LOCAL",
            "Only current partial-window progress was found; no eligible terminal authority selected.",
        )
    return (
        "TERMINAL_AUTHORITY_NOT_LOCAL",
        "TERMINAL_AUTHORITY_NOT_LOCAL",
        "No eligible terminal tournament authority was found locally.",
    )


def r53b_scientific_acceptance(family: str, selected: dict[str, Any] | None) -> str:
    if selected is None:
        return "TERMINAL_AUTHORITY_NOT_LOCAL"
    source_class = selected.get("source class")
    if family in {"ridge_policy_v1_control", "pca_ridge_policy_v1_control", "spline_additive_ridge"}:
        return "QUARANTINED_AFTER_COMPLETION"
    if source_class == "CURRENT_DS24_IMPORTED_MAC_TERMINAL":
        return "ACCEPTED_IMPORTED_FINAL"
    if source_class == "CURRENT_DS24_MAC_TERMINAL":
        return "MAC_SCIENTIFIC_COMPLETION"
    return "ACCEPTED_FINAL"


def r53b_comparability(family: str, selected: dict[str, Any] | None, state: str) -> str:
    if selected is None:
        if state == "LOCAL_PARTIAL_PROGRESS_ONLY_NO_TERMINAL_AUTHORITY":
            return "PARTIAL_WINDOW_NOT_TERMINAL"
        return "NO_TERMINAL_LOCAL_AUTHORITY"
    if family in {"ridge_policy_v1_control", "pca_ridge_policy_v1_control", "spline_additive_ridge"}:
        return "LEGACY_QUARANTINED_NOT_ACCEPTED_LEADERBOARD"
    if selected.get("source class") in {"CURRENT_DS24_MAC_TERMINAL", "CURRENT_DS24_IMPORTED_MAC_TERMINAL"}:
        return "MAC_IMPORTED_DIFFERENT_EXECUTION_HOST"
    metrics = selected.get("metrics") or {}
    start = str(metrics.get("first_resolved_decision_timestamp") or "")
    end = str(metrics.get("last_resolved_decision_timestamp") or "")
    if start.startswith("2016-02-02") and end.startswith("2026-06-30"):
        return "DIRECTLY_COMPARABLE_V3_COMMON_WINDOW"
    return "V3_DIFFERENT_EVALUATION_WINDOWS"


def r53b_missing_metric_reason(row: dict[str, Any], fields: Iterable[str]) -> str:
    reasons: dict[str, str] = {}
    for field in fields:
        if row.get(field) not in {None, ""}:
            continue
        if not row.get("source path"):
            reasons[field] = "TERMINAL_AUTHORITY_NOT_LOCALLY_AVAILABLE_OR_TRANSFER_REQUIRED"
        else:
            reasons[field] = "NOT_CALCULATED_BY_SELECTED_TERMINAL_AUTHORITY"
    return json.dumps(reasons, sort_keys=True)


def r53b_identity_binding(family: str, row: dict[str, Any], ownership: dict[str, Any], progress: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "tournament_run_id": first_present(row.get("run ID"), progress.get("run_id") if progress else None, "DS24_R7_R14_POLICY_TOURNAMENT"),
        "family_id": family,
        "model_configuration_identity": first_present(row.get("model_config_authority_hash"), ownership.get("authority_hash")),
        "feature_authority": first_present(row.get("feature_order_sha256"), "DS24_CANONICAL_5M_MODEL_DATASET_EXTENDED_2016_2026_V1"),
        "target_authority": first_present(row.get("target_contract"), "forward_return_60m__decision_5m"),
        "decision_cadence": "5m",
        "refit_policy": "daily_session_v1",
        "development_endpoint": "2026-06-30T20:00:00+00:00",
        "evaluation_contract": row.get("evaluation_contract_id"),
        "execution_host": row.get("execution host"),
        "output_namespace": row.get("source path"),
        "producer_source": row.get("producer source"),
        "model_checkpoint_identity": row.get("source SHA"),
        "current_scientific_acceptance": row.get("scientific acceptance"),
    }


R53B_SCORECARD_FIELDS = [
    "model",
    "terminal authority class",
    "execution host",
    "evaluation start",
    "evaluation end",
    "Rank IC",
    "HAC lower",
    "HAC upper",
    "Pearson IC",
    "NDCG",
    "NDCG@20",
    "directional accuracy",
    "hit rate",
    "annual return",
    "annual volatility",
    "Sharpe",
    "cumulative return",
    "maximum drawdown",
    "win rate",
    "turnover",
    "total estimated costs",
    "cost classification",
    "resolved rows",
    "OOF rows",
    "source path",
    "source SHA",
    "scientific acceptance",
    "comparability class",
    "missing_metric_reason",
]


R53B_INVENTORY_FIELDS = [
    "family",
    "source path",
    "SHA256",
    "source class",
    "run ID",
    "producer host",
    "evaluation start",
    "evaluation end",
    "terminal marker",
    "metrics available",
    "candidate eligibility",
    "selected/rejected",
    "rejection reason",
]


def r53b_scorecard_row(row: dict[str, Any]) -> dict[str, Any]:
    scorecard = {
        "model": row["family"],
        "terminal authority class": row.get("terminal authority class"),
        "execution host": row.get("execution host"),
        "evaluation start": row.get("evaluation start"),
        "evaluation end": row.get("evaluation end"),
        "Rank IC": row.get("mean_spearman_rank_ic"),
        "HAC lower": row.get("hac_lower_95"),
        "HAC upper": row.get("hac_upper_95"),
        "Pearson IC": row.get("pearson_ic"),
        "NDCG": row.get("ndcg"),
        "NDCG@20": row.get("ndcg_at_20"),
        "directional accuracy": row.get("directional_accuracy"),
        "hit rate": first_present(row.get("mean_hit_rate"), row.get("top_n_hit_rate"), row.get("win_rate")),
        "annual return": row.get("annualized_return"),
        "annual volatility": row.get("annualized_volatility"),
        "Sharpe": row.get("sharpe"),
        "cumulative return": first_present(row.get("cumulative_net_return"), row.get("cumulative_gross_return")),
        "maximum drawdown": row.get("maximum_drawdown"),
        "win rate": row.get("win_rate"),
        "turnover": row.get("turnover"),
        "total estimated costs": row.get("total_estimated_transaction_costs"),
        "cost classification": row.get("cost classification"),
        "resolved rows": row.get("resolved_performance_rows"),
        "OOF rows": row.get("score_population_rows"),
        "source path": row.get("source path"),
        "source SHA": row.get("source SHA"),
        "scientific acceptance": row.get("scientific acceptance"),
        "comparability class": row.get("comparability class"),
    }
    metric_fields = [
        "Rank IC",
        "HAC lower",
        "HAC upper",
        "Pearson IC",
        "NDCG",
        "NDCG@20",
        "directional accuracy",
        "hit rate",
        "annual return",
        "annual volatility",
        "Sharpe",
        "cumulative return",
        "maximum drawdown",
        "win rate",
        "turnover",
        "total estimated costs",
    ]
    scorecard["missing_metric_reason"] = r53b_missing_metric_reason(scorecard, metric_fields)
    return scorecard


def build_r53b_rows(stage_root: Path, worker_root: Path, mac_aux_root: Path) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any] | None,
]:
    ownership, ownership_sources = load_latest_ownership(stage_root)
    r53a_by_family = load_r53a_matrix(stage_root)
    candidate_records = r53b_candidate_records_by_family(stage_root, worker_root, mac_aux_root)
    all_rows: list[dict[str, Any]] = []
    scorecard_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    decisions: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "parent_r53a_commit": "a0344185ef51145ad0a13e873c306bd7406cc831",
        "parent_r53_commit": "af03b5e6cf6773f386670259ff593fd6e4fab7a8",
        "families": {},
    }
    identity_bindings: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "ownership_sources": ownership_sources,
        "families": {},
    }
    common_window_rows: list[dict[str, Any]] = []
    for index, (family, display_name) in enumerate(REQUESTED_FAMILIES, start=1):
        candidates = candidate_records.get(family, [])
        selected = r53b_choose_terminal_candidate(family, candidates)
        owner = ownership.get(family, {})
        progress = load_progress(stage_root, worker_root, family)
        lifecycle = "COMPLETED" if family in R53B_EXPECTED_COMPLETED else "OPEN"
        if selected:
            metrics = dict(selected["metrics"])
            state = "LOCAL_TERMINAL_AUTHORITY_SELECTED"
            note = "Selected eligible terminal authority by tournament lineage/source class."
        elif lifecycle == "OPEN":
            metrics = {}
            state = "NOT_COMPLETED"
            note = "Open R53B lifecycle family; historical/provisional metrics are excluded from the completed scorecard."
        else:
            state, _acceptance, note = r53b_no_terminal_state(family, candidates, owner)
            metrics = {}
        for candidate in candidates:
            if selected and candidate["source path"] == selected["source path"]:
                candidate["selected/rejected"] = "SELECTED"
                candidate["rejection reason"] = ""
            elif selected and candidate.get("source class") == "CURRENT_DS24_PARTIAL_PROGRESS":
                candidate["rejection reason"] = "PARTIAL_WINDOW_NOT_TERMINAL; selected terminal authority supersedes this source."
            elif lifecycle == "OPEN":
                candidate["rejection reason"] = "R53B lifecycle marks family open; historical metrics cannot imply completion."
            elif candidate.get("candidate eligibility") == "INELIGIBLE_TRANSFER_NOT_LOCAL":
                candidate["rejection reason"] = "Transfer or source estate missing locally."
            elif not selected:
                candidate["rejection reason"] = "No eligible terminal tournament authority; lower-authority source rejected."
            else:
                candidate["rejection reason"] = "Lower-precedence or wrong lineage source rejected."
            inventory_rows.append(candidate)
        acceptance = "OPEN_NOT_COMPLETED" if lifecycle == "OPEN" else (r53b_scientific_acceptance(family, selected) if selected else r53b_no_terminal_state(family, candidates, owner)[1])
        source_class = selected.get("source class") if selected else ("NOT_COMPLETED" if lifecycle == "OPEN" else state)
        row: dict[str, Any] = {
            "family": family,
            "display_name": display_name,
            "family_index": index,
            "r53b_lifecycle": lifecycle,
            "historical_terminal_result": state,
            "terminal authority class": source_class,
            "execution host": first_present(selected.get("producer host") if selected else None, owner.get("execution_owner")),
            "source path": selected.get("source path") if selected else "",
            "source SHA": selected.get("SHA256") if selected else "",
            "run ID": selected.get("run ID") if selected else first_present(progress.get("run_id") if progress else None, owner.get("import_source_run_id")),
            "producer source": selected.get("source path") if selected else owner.get("authority_source"),
            "evaluation start": selected.get("evaluation start") if selected else "",
            "evaluation end": selected.get("evaluation end") if selected else "",
            "terminal marker": bool(selected),
            "current cursor": first_present(progress.get("last_completed_T") if progress else None, progress.get("last_completed_refit_T") if progress else None),
            "scientific acceptance": acceptance,
            "comparability class": r53b_comparability(family, selected, state),
            "r53b_recovery_note": note,
            "r53a_source_path": r53a_by_family.get(family, {}).get("r53a_result_authority_path"),
            "r53a_source_sha256": r53a_by_family.get(family, {}).get("r53a_result_authority_sha256"),
            "r53a_scientific_acceptance": r53a_by_family.get(family, {}).get("current_scientific_acceptance"),
            "r53a_rejected_reason": "",
        }
        row.update(metrics)
        row["cost classification"] = cost_classification(row) if selected else "NOT_APPLICABLE"
        if row["r53a_source_path"] and row["r53a_source_path"] != row["source path"]:
            row["r53a_rejected_reason"] = "R53A source was partial, historical, transfer-missing, or wrong lifecycle for R53B terminal-authority rules."
        all_rows.append(row)
        if lifecycle == "COMPLETED":
            scorecard = r53b_scorecard_row(row)
            scorecard_rows.append(scorecard)
            decisions["families"][family] = {
                "selected_source": row["source path"] or None,
                "source_hash": row["source SHA"] or None,
                "terminal_authority_class": source_class,
                "reason_selected": note,
                "rejected_r53a_source": row["r53a_source_path"] if row["r53a_source_path"] != row["source path"] else None,
                "reason_r53a_source_rejected": row["r53a_rejected_reason"],
                "terminal_completion_evidence": {
                    "terminal_marker": row["terminal marker"],
                    "current_cursor": row["current cursor"],
                    "candidate_count": len(candidates),
                },
                "final_evaluation_window": {"start": row["evaluation start"], "end": row["evaluation end"]},
                "confidence": "HIGH" if selected else "LOW",
            }
            identity_bindings["families"][family] = r53b_identity_binding(family, row, owner, progress)
            common_window_rows.append(
                {
                    "family": family,
                    "retained compact OOF": bool(selected and selected.get("source class") in {"CURRENT_DS24_MAC_TERMINAL", "CURRENT_DS24_IMPORTED_MAC_TERMINAL"}),
                    "retained Rank IC series": bool(selected and row.get("source path") and "resolved_performance_summary_v3.json" in row.get("source path", "")),
                    "retained daily returns": bool(selected and row.get("sharpe") not in {None, ""}),
                    "earliest timestamp": row["evaluation start"],
                    "latest timestamp": row["evaluation end"],
                    "common-window recomputation possible without refitting": bool(selected and row.get("source path")),
                    "blocker": "" if selected and row.get("source path") else "Terminal authority or retained timestamp-level evidence is not locally available.",
                }
            )
    scorecard_rows.sort(key=lambda row: (-(as_float(row.get("Rank IC")) if as_float(row.get("Rank IC")) is not None else -999999.0), str(row["model"])))
    for rank, row in enumerate(scorecard_rows, start=1):
        row["rank"] = rank
    dlinear_transfer = r53b_dlinear_transfer_required(stage_root) if not any(row["family"] == "DLinear" and row["source path"] for row in all_rows) else None
    return all_rows, scorecard_rows, inventory_rows, decisions, identity_bindings, common_window_rows, dlinear_transfer


def r53b_dlinear_transfer_required(stage_root: Path) -> dict[str, Any]:
    destination = Path("mac_aux_runs/queue=DS24_MAC_AUX_NINE_FAMILY_R1/family=dlinear")
    archive = Path("C:/Users/Brandon/Desktop/ds24_dlinear_mac_terminal_transfer_r1.tar")
    return {
        "classification": "DLINEAR_MAC_TERMINAL_RESULT_REQUIRES_TRANSFER",
        "expected_mac_source_roots": [
            "/Users/brandonlinnett/Desktop/trading_system/mac_aux_runs/queue=DS24_MAC_AUX_NINE_FAMILY_R1/family=dlinear",
            "/Users/brandonlinnett/Desktop/trading_system/mac_aux_runs/queue=DS24_MAC_AUX_NINE_FAMILY_R1/family=DLinear",
        ],
        "required_files": [
            "family_execution_summary.json",
            "metrics_only_v3/resolved_performance_summary_v3.json",
            "metrics_only_v3/checkpoint.json",
            "ensemble_oof_scores_manifest_v2.json",
            "ensemble_oof_partition_ledger_v2.csv",
            "model_artifacts/*",
            "ensemble_oof_scores_v2/**/part-refit=*.parquet",
            "checksums or transfer sha256 sidecar",
        ],
        "dell_destination": normalize_rel(destination),
        "expected_hashes": {},
        "recommended_archive_name": archive.name,
        "dell_archive_path": str(archive),
        "transfer_verification_command": f"python scripts/local/ds24_r53_tournament_results_consolidation.py --r53b --verify-transfer {archive}",
        "import_reconciliation_command": "python scripts/local/ds24_r53_tournament_results_consolidation.py --r53b",
        "note": "No transfer is executed by R53B; this is an exact manifest for the next manual Mac-to-Dell copy.",
    }


def r53b_classification(rows: list[dict[str, Any]]) -> str:
    completed_rows = [row for row in rows if row.get("r53b_lifecycle") == "COMPLETED"]
    missing = [row for row in completed_rows if not row.get("source path")]
    if not missing:
        return "DS24_R53B_13_COMPLETED_TERMINAL_AUTHORITIES_RECOVERED_6_FAMILIES_OPEN"
    if {row["family"] for row in missing} == {"DLinear"}:
        return "DS24_R53B_12_LOCAL_TERMINAL_AUTHORITIES_RECOVERED_DLINEAR_MAC_TRANSFER_REQUIRED_6_FAMILIES_OPEN"
    return "DS24_R53B_COMPLETED_TERMINAL_AUTHORITY_RECOVERY_PARTIAL_EXTERNAL_TRANSFER_REQUIRED"


def r53b_accepted_rank_ic_rows(scorecard_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in scorecard_rows
        if row.get("scientific acceptance") not in {"QUARANTINED_AFTER_COMPLETION", "TERMINAL_AUTHORITY_NOT_LOCAL", "MAC_TERMINAL_AUTHORITY_NOT_LOCAL"}
        and as_float(row.get("Rank IC")) is not None
    ]
    rows.sort(key=lambda row: as_float(row.get("Rank IC")) or 0.0, reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["accepted_rank"] = rank
    return rows


def render_r53b_errors(rows: list[dict[str, Any]]) -> str:
    by_family = {row["family"]: row for row in rows}
    focus = [
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "mlp",
        "lightgbm_rank_xendcg",
        "lightgbm_lambdarank",
        "DLinear",
        "Temporal Fusion Transformer",
    ]
    lines = [
        "# R53B R53A Source Selection Errors",
        "",
        "R53A is preserved as an audit artifact. R53B rejects sources that do not bind to terminal tournament identity.",
        "",
    ]
    for family in focus:
        row = by_family.get(family, {})
        lines.extend(
            [
                f"## {family}",
                "",
                f"- R53A source: {row.get('r53a_source_path') or 'none'}",
                f"- R53A status: {row.get('r53a_scientific_acceptance') or 'none'}",
                f"- R53B authority: {row.get('source path') or row.get('historical_terminal_result')}",
                f"- R53B class: {row.get('terminal authority class')}",
                f"- Correction: {row.get('r53b_recovery_note')}",
                "",
            ]
        )
    return "\n".join(lines)


def render_r53b_report(rows: list[dict[str, Any]], classification: str) -> str:
    completed = [row for row in rows if row.get("r53b_lifecycle") == "COMPLETED"]
    open_rows = [row for row in rows if row.get("r53b_lifecycle") == "OPEN"]
    lines = [
        "# DS24 R53B Terminal Tournament Authority Recovery",
        "",
        f"Classification: `{classification}`",
        "",
        f"- Completed lifecycle families: {len(completed)}",
        f"- Open lifecycle families: {len(open_rows)}",
        "- R53/R53A outputs were not overwritten.",
        "- Read-only audit: no model execution, resume, scoring, queue mutation, broker call, or holdout access.",
        "",
        "## Completed Families",
        "",
    ]
    for row in completed:
        lines.append(
            f"- {row['family']}: {row.get('terminal authority class')} | {row.get('scientific acceptance')} | "
            f"{row.get('evaluation start') or 'no local terminal window'} to {row.get('evaluation end') or 'no local terminal window'}"
        )
    lines.extend(["", "## Open Families", ""])
    for row in open_rows:
        lines.append(f"- {row['family']}: {row.get('historical_terminal_result')}")
    return "\n".join(lines) + "\n"


def run_r53b(stage_root: Path, worker_root: Path, mac_aux_root: Path, output_root: Path) -> dict[str, Any]:
    ownership, _ = load_latest_ownership(stage_root)
    pre_snapshot = runtime_snapshot(stage_root, worker_root, ownership)
    rows, scorecard, inventory, decisions, identities, common_window, dlinear_transfer = build_r53b_rows(stage_root, worker_root, mac_aux_root)
    post_snapshot = runtime_snapshot(stage_root, worker_root, ownership)
    output_root.mkdir(parents=True, exist_ok=True)
    classification = r53b_classification(rows)
    write_json(output_root, "R53B_pre_recovery_runtime_snapshot.json", pre_snapshot)
    write_json(output_root, "R53B_post_recovery_runtime_snapshot.json", post_snapshot)
    write_json(output_root, "R53B_family_identity_bindings.json", identities)
    write_csv(output_root, "R53B_candidate_result_inventory.csv", inventory, R53B_INVENTORY_FIELDS)
    write_json(output_root, "R53B_terminal_authority_decisions.json", decisions)
    write_csv(output_root, "R53B_FINAL_13_COMPLETED_MODEL_SCORECARD.csv", scorecard, ["rank"] + R53B_SCORECARD_FIELDS)
    accepted = r53b_accepted_rank_ic_rows([dict(row) for row in scorecard])
    write_csv(output_root, "R53B_accepted_rank_ic_leaderboard.csv", accepted, ["accepted_rank", "rank"] + R53B_SCORECARD_FIELDS)
    quarantined = [row for row in scorecard if row.get("scientific acceptance") == "QUARANTINED_AFTER_COMPLETION"]
    write_csv(output_root, "R53B_quarantined_completed_results.csv", quarantined, ["rank"] + R53B_SCORECARD_FIELDS)
    write_csv(output_root, "R53B_all_completed_descriptive_table.csv", scorecard, ["rank"] + R53B_SCORECARD_FIELDS)
    write_csv(
        output_root,
        "R53B_common_window_readiness.csv",
        common_window,
        [
            "family",
            "retained compact OOF",
            "retained Rank IC series",
            "retained daily returns",
            "earliest timestamp",
            "latest timestamp",
            "common-window recomputation possible without refitting",
            "blocker",
        ],
    )
    if dlinear_transfer:
        write_json(output_root, "R53B_dlinear_transfer_required.json", dlinear_transfer)
    write_text(output_root, "R53B_R53A_SOURCE_SELECTION_ERRORS.md", render_r53b_errors(rows))
    write_text(output_root, "R53B_TERMINAL_AUTHORITY_RECOVERY_REPORT.md", render_r53b_report(rows, classification))
    completed = [row["family"] for row in rows if row.get("r53b_lifecycle") == "COMPLETED"]
    open_families = [row["family"] for row in rows if row.get("r53b_lifecycle") == "OPEN"]
    return {
        "classification": classification,
        "total_families": len(rows),
        "completed": len(completed),
        "open": len(open_families),
        "completed_families": completed,
        "open_families": open_families,
        "local_terminal_authorities": sum(1 for row in rows if row.get("r53b_lifecycle") == "COMPLETED" and row.get("source path")),
        "output_root": normalize_rel(output_root),
    }


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
    parser.add_argument("--r53a", action="store_true", help="Run the R53A historical completion recovery pass.")
    parser.add_argument("--r53b", action="store_true", help="Run the R53B terminal-authority recovery pass.")
    parser.add_argument("--verify-transfer", type=Path, default=None, help="Reserved transfer-verification path recorded by R53B.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable run summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    if args.r53a and output_root == STAGE_ROOT / R53_DIRNAME:
        output_root = args.stage_root / R53A_DIRNAME
    if args.r53b and output_root == STAGE_ROOT / R53_DIRNAME:
        output_root = args.stage_root / R53B_DIRNAME
    summary = (
        run_r53b(args.stage_root, args.worker_root, args.mac_aux_root, output_root)
        if args.r53b
        else (
            run_r53a(args.stage_root, args.worker_root, args.mac_aux_root, output_root)
            if args.r53a
            else run(args.stage_root, args.worker_root, args.mac_aux_root, output_root)
        )
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.r53b:
        print(f"Total families: {summary['total_families']}")
        print(f"Completed: {summary['completed']}")
        print(f"Open: {summary['open']}")
        print(f"Local terminal authorities: {summary['local_terminal_authorities']}")
        print(f"Classification: {summary['classification']}")
        print(f"Output root: {summary['output_root']}")
    elif args.r53a:
        print(f"Total families: {summary['total_families']}")
        print(f"Completed: {summary['completed']}")
        print(f"Unfinished: {summary['unfinished']}")
        print(f"Classification: {summary['classification']}")
        print(f"Output root: {summary['output_root']}")
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
