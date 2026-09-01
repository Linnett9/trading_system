from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from core.research.ml import ds24_metrics_only_evaluator as ev
from core.research.ml.governance.trial_identity import canonical_hash


TARGET_ID = "forward_return_60m__decision_5m"
AUDIT_RUN_ID = "ds24_completed_model_training_history_audit_r1_20260831T000000Z"
EXHAUSTIVE_SEARCH_RUN_ID = "ds24_exhaustive_model_subset_ensemble_search_r1_20260831T000000Z"

HISTORY_SESSION_BUCKETS: tuple[tuple[str, int | None, int | None], ...] = (
    ("sessions_001_126", 1, 126),
    ("sessions_127_252", 127, 252),
    ("sessions_253_504", 253, 504),
    ("sessions_505_1008", 505, 1008),
    ("sessions_gt_1008", 1009, None),
)

HISTORY_ROW_BUCKETS: tuple[tuple[str, int | None, int | None], ...] = (
    ("rows_000000_249999", 0, 249_999),
    ("rows_250000_499999", 250_000, 499_999),
    ("rows_500000_749999", 500_000, 749_999),
    ("rows_750000_999999", 750_000, 999_999),
    ("rows_gt_1000000", 1_000_000, None),
)

LEARNING_CLASSIFICATIONS = {
    "IMPROVED_WITH_MORE_HISTORY",
    "DEGRADED_WITH_MORE_HISTORY",
    "NONMONOTONIC_HISTORY_RELATIONSHIP",
    "STABLE_NO_MATERIAL_CHANGE",
    "INSUFFICIENT_ACCEPTED_EVIDENCE",
    "TRAINING_HISTORY_METADATA_MISSING",
    "QUARANTINED_LEGACY_EVIDENCE_ONLY",
}

ACCEPTED_V3_STATES = {
    "ACCEPTED_V3",
    "ACCEPTED_V3_RESOLVED",
    "VALIDATED_V3",
    "VALIDATED_RESOLVED_V3",
    "PASS_EXISTING_RESOLVED_V3",
}

QUARANTINE_MARKERS = (
    "QUARANTINE",
    "QUARANTINED",
    "PROVISIONAL_UNVALIDATED",
    "V1",
    "V2",
    "V3_REPLAY_REQUIRED",
    "ECONOMIC_ONLY_RECOVERABLE",
    "NOT_RETAINED",
)

REQUIRED_V3_METRIC_COLUMNS = {
    "family",
    "decision_timestamp",
    "target_id",
    "evaluation_contract_id",
    "evaluation_contract_hash",
    "eligible_asset_count",
    "resolved_asset_count",
    "rank_ic_observation_count",
    "spearman_rank_ic",
    "row_hash",
}

FULL_PREDICTION_FORBIDDEN_MARKERS = (
    "full_prediction",
    "full_predictions",
    "prediction_matrix",
    "full_cross_sectional",
    "cross_sectional_predictions",
)

CANONICAL_EXHAUSTIVE_METHOD = "EQUAL_WEIGHT_PERCENTILE_RANK"
TOPN_ONLY_CANONICAL_METHOD = "EQUAL_WEIGHT_RECIPROCAL_RANK_FUSION"
ADDITIONAL_EXHAUSTIVE_METHODS = (
    "MEDIAN_PERCENTILE_RANK",
    "EQUAL_WEIGHT_BORDA_FUSION",
    "FAMILY_BALANCED_EQUAL_WEIGHT",
    "DIVERSITY_SCREENED_EQUAL_WEIGHT",
    "LAGGED_IC_NONNEGATIVE_WEIGHTED_63D",
    "LAGGED_IC_NONNEGATIVE_WEIGHTED_126D",
    "LAGGED_IC_NONNEGATIVE_WEIGHTED_252D",
    "CONSTRAINED_REGULARIZED_WALK_FORWARD_STACKING",
    "ONLINE_EXPERT_HEDGE_WEIGHTING",
    "COVARIANCE_REGULARIZED_WEIGHTING",
    "REGIME_AWARE_MIXTURE_OF_EXPERTS",
)
EXHAUSTIVE_METHODS = (CANONICAL_EXHAUSTIVE_METHOD, TOPN_ONLY_CANONICAL_METHOD) + ADDITIONAL_EXHAUSTIVE_METHODS

RETENTION_PROTECTED_ROLES = {
    "current_resumable_checkpoint",
    "previous_verified_checkpoint",
    "lease_metadata",
    "checkpoint_metadata",
    "deterministic_configuration_hash",
    "deterministic_data_hash",
    "deterministic_code_hash",
    "final_compact_model_package",
    "preprocessing_state",
    "v3_metrics",
    "compact_ranked_selection_evidence",
    "selected_constituent_final_package",
    "final_ensemble_manifest",
}

RETENTION_PRUNE_CANDIDATE_ROLES = {
    "obsolete_daily_refit_model_package",
    "historical_daily_refit_model_package",
}

TOURNAMENT_MODEL_UNIVERSE: tuple[dict[str, str], ...] = (
    {"model_id": "ridge_policy_v1_control", "display_name": "Ridge policy control", "model_class": "linear_control"},
    {"model_id": "pca_ridge_policy_v1_control", "display_name": "PCA Ridge policy control", "model_class": "linear_control"},
    {"model_id": "spline_additive_ridge", "display_name": "Spline additive Ridge", "model_class": "additive_control"},
    {"model_id": "elastic_net", "display_name": "Elastic Net", "model_class": "linear_regularized"},
    {"model_id": "rff_ridge", "display_name": "RFF Ridge", "model_class": "kernel_approximation"},
    {"model_id": "huber", "display_name": "Huber", "model_class": "robust_linear"},
    {"model_id": "mlp", "display_name": "MLP", "model_class": "neural_network"},
    {"model_id": "random_forest", "display_name": "Random Forest", "model_class": "tree_ensemble"},
    {"model_id": "extra_trees", "display_name": "Extra Trees", "model_class": "tree_ensemble"},
    {"model_id": "gradient_boosting", "display_name": "Gradient Boosting", "model_class": "boosted_tree"},
    {"model_id": "lightgbm_rank_xendcg", "display_name": "LightGBM Rank XENDCG", "model_class": "learning_to_rank"},
    {"model_id": "lightgbm_lambdarank", "display_name": "LightGBM LambdaRank", "model_class": "learning_to_rank"},
    {"model_id": "dlinear", "display_name": "DLinear", "model_class": "deep_time_series"},
    {"model_id": "patchtst", "display_name": "PatchTST", "model_class": "deep_time_series"},
    {"model_id": "transformer", "display_name": "Transformer", "model_class": "deep_sequence"},
    {"model_id": "itransformer", "display_name": "iTransformer", "model_class": "deep_sequence"},
    {"model_id": "momentum_transformer", "display_name": "Momentum Transformer", "model_class": "deep_sequence"},
    {"model_id": "market_context_encoder", "display_name": "Market Context Encoder", "model_class": "context_encoder"},
    {"model_id": "temporal_fusion_transformer", "display_name": "Temporal Fusion Transformer", "model_class": "deep_sequence"},
    {"model_id": "exact_ridge_pca", "display_name": "Exact/control compatible baseline", "model_class": "exact_control"},
)


def _as_utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _normalise_model_item(item: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(item, str):
        item = {"model_id": item}
    model_id = str(item["model_id"]).strip()
    universe = {entry["model_id"]: entry for entry in TOURNAMENT_MODEL_UNIVERSE}
    base = dict(universe.get(model_id, {}))
    base.update(dict(item))
    base["model_id"] = model_id
    base.setdefault("display_name", model_id)
    base.setdefault("model_class", "unclassified")
    return base


def _model_sort_key(item: Mapping[str, Any]) -> tuple[int, str]:
    order = {entry["model_id"]: idx for idx, entry in enumerate(TOURNAMENT_MODEL_UNIVERSE)}
    model_id = str(item["model_id"])
    return (order.get(model_id, len(order)), model_id)


@dataclass(frozen=True)
class EnsembleScoreRecord:
    decision_timestamp: str
    security_id: str
    symbol: str | None
    family_id: str
    model_generation: str
    training_cutoff: str
    score_available_at: str
    raw_score: float | None
    within_timestamp_percentile_rank: float | None
    eligible_universe_hash: str
    feature_contract_hash: str
    target_contract_hash: str
    evaluation_version: str
    source_artifact_hash: str

    def __post_init__(self) -> None:
        decision = _as_utc_timestamp(self.decision_timestamp)
        available = _as_utc_timestamp(self.score_available_at)
        cutoff = _as_utc_timestamp(self.training_cutoff)
        if available > decision:
            raise ValueError("score_available_at must be <= decision_timestamp")
        if cutoff >= decision:
            raise ValueError("training_cutoff must be < decision_timestamp")
        rank = _finite_float(self.within_timestamp_percentile_rank)
        if rank is not None and not 0.0 <= rank <= 1.0:
            raise ValueError("within_timestamp_percentile_rank must be in [0, 1]")

    def identity_key(self) -> tuple[str, str, str]:
        return (self.family_id, self.decision_timestamp, self.security_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_timestamp": self.decision_timestamp,
            "security_id": self.security_id,
            "symbol": self.symbol,
            "family_id": self.family_id,
            "model_generation": self.model_generation,
            "training_cutoff": self.training_cutoff,
            "score_available_at": self.score_available_at,
            "raw_score": self.raw_score,
            "within_timestamp_percentile_rank": self.within_timestamp_percentile_rank,
            "eligible_universe_hash": self.eligible_universe_hash,
            "feature_contract_hash": self.feature_contract_hash,
            "target_contract_hash": self.target_contract_hash,
            "evaluation_version": self.evaluation_version,
            "source_artifact_hash": self.source_artifact_hash,
        }

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.to_dict())


def history_bucket_for(*, training_rows: int | float | None = None, training_sessions: int | float | None = None) -> str:
    sessions = _finite_float(training_sessions)
    if sessions is not None:
        sessions_i = int(sessions)
        for label, lower, upper in HISTORY_SESSION_BUCKETS:
            if sessions_i >= int(lower or 0) and (upper is None or sessions_i <= upper):
                return label
        return "history_unknown"

    rows = _finite_float(training_rows)
    if rows is None:
        return "history_unknown"
    rows_i = int(rows)
    for label, lower, upper in HISTORY_ROW_BUCKETS:
        if rows_i >= int(lower or 0) and (upper is None or rows_i <= upper):
            return label
    return "history_unknown"


def assign_history_buckets(records: Sequence[Mapping[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(records).copy()
    if frame.empty:
        frame["training_history_bucket"] = pd.Series(dtype="string")
        return frame
    frame["training_history_bucket"] = [
        history_bucket_for(training_rows=row.get("training_rows"), training_sessions=row.get("training_sessions"))
        for row in frame.to_dict("records")
    ]
    return frame


def is_accepted_v3_evidence(evaluation_version: str | None, validation_state: str | None, quarantine_state: str | None = None) -> bool:
    eval_label = str(evaluation_version or "").upper()
    state = str(validation_state or "").upper()
    quarantine = str(quarantine_state or "").upper()
    if "V3" not in eval_label:
        return False
    if quarantine and quarantine not in {"NONE", "CLEAR", "NOT_QUARANTINED", "FALSE"}:
        return False
    if any(marker in state for marker in QUARANTINE_MARKERS):
        return False
    return state in ACCEPTED_V3_STATES or state.startswith("ACCEPTED_V3")


def classify_model_eligibility(record: Mapping[str, Any], *, min_common_support_coverage: float = 0.5) -> dict[str, Any]:
    reasons: list[str] = []
    if not bool(record.get("completed", False)):
        reasons.append("MODEL_NOT_COMPLETED")
    if not is_accepted_v3_evidence(record.get("evaluation_version"), record.get("validation_state"), record.get("quarantine_state")):
        reasons.append("ACCEPTED_V3_EVIDENCE_MISSING")
    if not bool(record.get("compatible_contracts", False)):
        reasons.append("COMPATIBLE_TARGET_DECISION_CONTRACT_MISSING")
    if not bool(record.get("point_in_time_scores", False)):
        reasons.append("POINT_IN_TIME_SCORE_AVAILABILITY_MISSING")
    if not bool(record.get("timestamp_security_identity", False)):
        reasons.append("TIMESTAMP_SECURITY_IDENTITY_MISSING")
    if str(record.get("quarantine_state") or "").upper() not in {"", "NONE", "CLEAR", "NOT_QUARANTINED", "FALSE"}:
        reasons.append("UNRESOLVED_QUARANTINE")
    coverage = _finite_float(record.get("common_support_coverage"))
    if coverage is None or coverage < min_common_support_coverage:
        reasons.append("ADEQUATE_COMMON_SUPPORT_COVERAGE_MISSING")
    return {
        "model_id": record.get("model_id"),
        "eligible": not reasons,
        "exclusion_reasons": reasons,
        "eligibility_status": "ELIGIBLE_ACCEPTED_V3_COMMON_SUPPORT" if not reasons else "EXCLUDED_" + "__".join(reasons),
    }


def validate_v3_metric_schema(columns: Iterable[str]) -> dict[str, Any]:
    observed = {str(col) for col in columns}
    missing = sorted(REQUIRED_V3_METRIC_COLUMNS - observed)
    return {
        "valid": not missing,
        "missing_columns": missing,
        "required_columns": sorted(REQUIRED_V3_METRIC_COLUMNS),
    }


def exclude_pending_and_censored(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        return frame.copy(), {"input_rows": 0, "kept_rows": 0, "excluded_rows": 0}
    out = frame.copy()
    mask = pd.Series(True, index=out.index)
    for column in ("target_state", "outcome_state", "target_maturity_state"):
        if column in out.columns:
            labels = out[column].astype(str).str.upper()
            mask &= ~labels.str.contains("PENDING|CENSORED|UNMATURED|TERMINAL", regex=True, na=False)
    if "spearman_rank_ic" in out.columns:
        mask &= pd.to_numeric(out["spearman_rank_ic"], errors="coerce").notna()
    kept = out.loc[mask].copy()
    return kept, {"input_rows": int(len(out)), "kept_rows": int(len(kept)), "excluded_rows": int(len(out) - len(kept))}


def strict_common_support(frame: pd.DataFrame, family_ids: Sequence[str] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        families = list(family_ids or [])
        return frame.copy(), {
            "mode": "GLOBAL_COMMON_TIMESTAMP_SECURITY_PANEL",
            "required_family_count": len(families),
            "input_rows": 0,
            "aligned_rows": 0,
            "common_timestamp_security_pairs": 0,
        }
    asset_col = "security_id" if "security_id" in frame.columns else "asset_id"
    required = sorted(str(item) for item in (family_ids or frame["family_id"].dropna().unique()))
    work = frame[frame["family_id"].isin(required)].copy()
    keys = ["decision_timestamp", asset_col]
    counts = work.drop_duplicates(keys + ["family_id"]).groupby(keys)["family_id"].nunique().reset_index(name="family_count")
    common_keys = counts.loc[counts["family_count"] == len(required), keys]
    aligned = work.merge(common_keys, on=keys, how="inner")
    if asset_col != "security_id" and "security_id" not in aligned.columns:
        aligned = aligned.rename(columns={asset_col: "security_id"})
    report = {
        "mode": "GLOBAL_COMMON_TIMESTAMP_SECURITY_PANEL",
        "required_family_count": len(required),
        "required_families": required,
        "input_rows": int(len(frame)),
        "aligned_rows": int(len(aligned)),
        "common_timestamp_security_pairs": int(len(common_keys)),
    }
    return aligned, report


def bounded_quorum_union(frame: pd.DataFrame, *, min_quorum: int, family_ids: Sequence[str] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if min_quorum < 1:
        raise ValueError("min_quorum must be positive")
    if frame.empty:
        return frame.copy(), {"mode": "BOUNDED_QUORUM_UNION", "min_quorum": int(min_quorum), "input_rows": 0, "aligned_rows": 0}
    asset_col = "security_id" if "security_id" in frame.columns else "asset_id"
    required = sorted(str(item) for item in (family_ids or frame["family_id"].dropna().unique()))
    if min_quorum > len(required):
        raise ValueError("min_quorum cannot exceed the required family count")
    work = frame[frame["family_id"].isin(required)].copy()
    keys = ["decision_timestamp", asset_col]
    counts = work.drop_duplicates(keys + ["family_id"]).groupby(keys)["family_id"].nunique().reset_index(name="family_count")
    quorum_keys = counts.loc[counts["family_count"] >= min_quorum, keys + ["family_count"]]
    aligned = work.merge(quorum_keys, on=keys, how="inner")
    aligned["missing_family_count"] = len(required) - aligned["family_count"]
    aligned["missing_family_rate"] = aligned["missing_family_count"] / float(len(required))
    if asset_col != "security_id" and "security_id" not in aligned.columns:
        aligned = aligned.rename(columns={asset_col: "security_id"})
    return aligned, {
        "mode": "BOUNDED_QUORUM_UNION",
        "min_quorum": int(min_quorum),
        "required_family_count": len(required),
        "input_rows": int(len(frame)),
        "aligned_rows": int(len(aligned)),
        "quorum_timestamp_security_pairs": int(len(quorum_keys)),
    }


def normalise_within_timestamp_percentile(frame: pd.DataFrame, *, score_col: str = "raw_score") -> pd.DataFrame:
    required = {"decision_timestamp", "family_id", score_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing columns for percentile normalisation: {missing}")
    out = frame.copy()
    out["within_timestamp_percentile_rank"] = (
        out.groupby(["decision_timestamp", "family_id"], sort=True)[score_col]
        .rank(method="average", pct=True, ascending=True)
        .astype(float)
    )
    return out


def aggregate_percentile_ensemble(
    frame: pd.DataFrame,
    *,
    method: str = "mean",
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    required = {"decision_timestamp", "security_id", "family_id", "within_timestamp_percentile_rank"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing columns for percentile ensemble: {missing}")
    work = frame.copy()
    work["rank_value"] = pd.to_numeric(work["within_timestamp_percentile_rank"], errors="coerce")
    if work["rank_value"].isna().any():
        raise ValueError("percentile ensemble requires non-null percentile ranks")
    keys = ["decision_timestamp", "security_id"]
    if weights is not None:
        weights_map = {str(k): float(v) for k, v in weights.items()}
        work["weight"] = work["family_id"].map(weights_map).fillna(0.0).astype(float)

        def weighted(group: pd.DataFrame) -> pd.Series:
            total = float(group["weight"].sum())
            score = float((group["rank_value"] * group["weight"]).sum() / total) if total > 0 else float(group["rank_value"].mean())
            return pd.Series({"ensemble_score": score, "component_count": int(group["family_id"].nunique())})

        out = work.groupby(keys, sort=True).apply(weighted).reset_index()
        out["method"] = "weighted_mean_percentile_rank"
        return out
    if method == "mean":
        agg = work.groupby(keys, sort=True)["rank_value"].agg(["mean", "count"]).reset_index()
        return agg.rename(columns={"mean": "ensemble_score", "count": "component_count"}).assign(method="equal_weight_percentile_rank")
    if method == "median":
        agg = work.groupby(keys, sort=True)["rank_value"].agg(["median", "count"]).reset_index()
        return agg.rename(columns={"median": "ensemble_score", "count": "component_count"}).assign(method="median_percentile_rank")
    raise ValueError(f"unsupported percentile ensemble method: {method}")


def reciprocal_rank_fusion(frame: pd.DataFrame, *, rank_col: str = "topn_rank", k: int = 60) -> pd.DataFrame:
    required = {"decision_timestamp", "security_id", "family_id", rank_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing columns for reciprocal rank fusion: {missing}")
    work = frame.copy()
    ranks = pd.to_numeric(work[rank_col], errors="coerce")
    if ranks.isna().any() or (ranks <= 0).any():
        raise ValueError("reciprocal rank fusion requires positive numeric ranks")
    work["rrf_contribution"] = 1.0 / (float(k) + ranks.astype(float))
    out = work.groupby(["decision_timestamp", "security_id"], sort=True).agg(
        ensemble_score=("rrf_contribution", "sum"),
        component_count=("family_id", "nunique"),
    )
    return out.reset_index().assign(method="equal_weight_reciprocal_rank_fusion")


def borda_count_fusion(frame: pd.DataFrame, *, rank_col: str = "topn_rank") -> pd.DataFrame:
    required = {"decision_timestamp", "security_id", "family_id", rank_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing columns for Borda fusion: {missing}")
    work = frame.copy()
    ranks = pd.to_numeric(work[rank_col], errors="coerce")
    if ranks.isna().any() or (ranks <= 0).any():
        raise ValueError("Borda fusion requires positive numeric ranks")
    max_rank = work.groupby(["decision_timestamp", "family_id"], sort=True)[rank_col].transform("max").astype(float)
    work["borda_contribution"] = max_rank - ranks.astype(float) + 1.0
    out = work.groupby(["decision_timestamp", "security_id"], sort=True).agg(
        ensemble_score=("borda_contribution", "sum"),
        component_count=("family_id", "nunique"),
    )
    return out.reset_index().assign(method="equal_weight_borda_fusion")


def overlap_safe_daily_returns(sleeves: pd.DataFrame) -> pd.DataFrame:
    if sleeves.empty:
        return pd.DataFrame(columns=["session_date", "gross_daily_return", "net_daily_return", "matured_sleeve_decisions", "turnover"])
    work = sleeves.copy()
    if "maturity_session_date" in work.columns:
        work["session_date"] = work["maturity_session_date"].astype(str)
    elif "session_date" not in work.columns:
        if "maturity_timestamp" not in work.columns:
            raise ValueError("sleeves require maturity_session_date, session_date, or maturity_timestamp")
        work["session_date"] = pd.to_datetime(work["maturity_timestamp"], utc=True).dt.date.astype(str)
    gross_col = "gross_return_contribution" if "gross_return_contribution" in work.columns else "gross_return"
    net_col = "net_return_contribution" if "net_return_contribution" in work.columns else "net_return"
    turnover_col = "turnover" if "turnover" in work.columns else None
    grouped = work.groupby("session_date", sort=True).agg(
        gross_daily_return=(gross_col, "sum"),
        net_daily_return=(net_col, "sum"),
        matured_sleeve_decisions=(net_col, "count"),
    )
    grouped["turnover"] = work.groupby("session_date", sort=True)[turnover_col].sum() if turnover_col else 0.0
    return grouped.reset_index()


def dependence_aware_ci(values: Sequence[float], *, lag: int = 12) -> dict[str, Any]:
    ci = ev.newey_west_mean_ci(values, lag=lag)
    ci["inference_method"] = f"newey_west_hac_lag_{ci['lag']}"
    return ci


def enforce_holdout_boundary(frame: pd.DataFrame, *, holdout_start: str, timestamp_col: str = "decision_timestamp") -> None:
    if timestamp_col not in frame.columns:
        raise ValueError(f"holdout boundary requires {timestamp_col}")
    if frame.empty:
        return
    timestamps = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    boundary = _as_utc_timestamp(holdout_start)
    if timestamps.ge(boundary).any():
        raise RuntimeError("DS24_OUTER_HOLDOUT_BOUNDARY_VIOLATION")


def enforce_no_future_outcomes(frame: pd.DataFrame, *, asof_timestamp: str, outcome_timestamp_col: str = "target_available_timestamp") -> None:
    if outcome_timestamp_col not in frame.columns:
        raise ValueError(f"future-outcome check requires {outcome_timestamp_col}")
    if frame.empty:
        return
    timestamps = pd.to_datetime(frame[outcome_timestamp_col], utc=True, errors="coerce")
    asof = _as_utc_timestamp(asof_timestamp)
    if timestamps.ge(asof).any():
        raise RuntimeError("DS24_FUTURE_OR_SAME_PERIOD_OUTCOME_USE_REFUSED")


def refuse_live_namespace_write(output_root: str | Path, protected_roots: Sequence[str | Path]) -> None:
    out = Path(output_root).resolve()
    for root in protected_roots:
        protected = Path(root).resolve()
        if out == protected or protected in out.parents:
            raise RuntimeError("DS24_LIVE_NAMESPACE_WRITE_REFUSED")


def metrics_only_storage_validation(file_names: Iterable[str | Path]) -> dict[str, Any]:
    names = list(file_names)
    violations = []
    for name in names:
        lowered = str(name).replace("\\", "/").lower()
        if any(marker in lowered for marker in FULL_PREDICTION_FORBIDDEN_MARKERS):
            violations.append(str(name))
    if violations:
        raise RuntimeError("DS24_ZERO_FULL_PREDICTION_PERSISTENCE_VIOLATION")
    return {"passed": True, "full_prediction_file_count": 0, "checked_path_count": len(names)}


def bounded_retention_contract_payload() -> dict[str, Any]:
    return {
        "contract_id": "DS24_BOUNDED_MODEL_PACKAGE_RETENTION_AND_ENSEMBLE_EVIDENCE_CONTRACT_R1",
        "active_family_retention": [
            "current_resumable_checkpoint",
            "one_previous_verified_checkpoint_for_rollback",
            "lease_checkpoint_metadata_and_deterministic_configuration_data_code_hashes",
        ],
        "completed_family_retention": [
            "one_final_compact_model_package",
            "preprocessing_state_required_for_inference",
            "accepted_v3_metrics_and_compact_ranked_selection_evidence",
        ],
        "prune_rule": "obsolete_daily_refit_model_packages_are_prune_eligible_only_after_predictions_metrics_checkpoint_cursor_and_integrity_hashes_are_durably_committed_and_independently_verified",
        "never_prune": [
            "current_resumable_checkpoint",
            "previous_verified_checkpoint",
            "active_worker_owned_files",
            "completed_final_compact_package",
            "preprocessing_state",
            "accepted_v3_metrics",
            "compact_ranked_selection_evidence",
            "selected_constituent_final_package",
            "final_ensemble_manifest",
        ],
        "ensemble_storage": {
            "format": "compressed_partitioned_parquet",
            "preferred_symbol_identity": "integer_symbol_identifiers",
            "preferred_rank_dtypes": ["uint8", "uint16"],
            "forbidden": ["raw_full_universe_prediction_matrices", "per_subset_timestamp_level_prediction_files"],
            "persist_only": [
                "aggregate_row_per_subset_method_fold",
                "trial_accounting_and_multiple_testing_metadata",
                "shortlisted_confirmation_results",
                "final_frozen_ensemble_manifest",
            ],
        },
        "topn_only_search_classification": "SUPPORTS_PORTFOLIO_ENSEMBLE_RETURNS_NOT_EXACT_FULL_UNIVERSE_ENSEMBLE_RANK_IC",
        "dry_run_required_before_deletion": True,
        "delete_files": False,
        "contract_hash": "",
    }


def bounded_retention_contract() -> dict[str, Any]:
    payload = bounded_retention_contract_payload()
    material = {key: value for key, value in payload.items() if key != "contract_hash"}
    payload["contract_hash"] = canonical_hash(material)
    return payload


def classify_retention_record(record: Mapping[str, Any]) -> dict[str, Any]:
    role = str(record.get("role") or "").strip()
    path = str(record.get("path") or "")
    size_bytes = int(_finite_float(record.get("size_bytes")) or 0)
    reasons: list[str] = []

    if bool(record.get("live_worker_owned", False)) or bool(record.get("active_worker_owned", False)):
        return {
            "family_id": record.get("family_id"),
            "path": path,
            "role": role,
            "size_bytes": size_bytes,
            "retention_class": "PROTECTED",
            "reason": "ACTIVE_WORKER_OWNED",
            "delete_file": False,
        }
    if role in RETENTION_PROTECTED_ROLES or bool(record.get("current_checkpoint", False)) or bool(record.get("rollback_checkpoint", False)) or bool(record.get("final_package", False)):
        return {
            "family_id": record.get("family_id"),
            "path": path,
            "role": role,
            "size_bytes": size_bytes,
            "retention_class": "PROTECTED",
            "reason": "REQUIRED_BY_BOUNDED_RETENTION_CONTRACT",
            "delete_file": False,
        }
    if role in RETENTION_PRUNE_CANDIDATE_ROLES:
        required_flags = {
            "predictions_committed": bool(record.get("predictions_committed", False)),
            "metrics_committed": bool(record.get("metrics_committed", False)),
            "checkpoint_cursor_committed": bool(record.get("checkpoint_cursor_committed", False)),
            "integrity_hashes_verified": bool(record.get("integrity_hashes_verified", False)),
            "independently_verified": bool(record.get("independently_verified", False)),
        }
        reasons = [key.upper() + "_MISSING" for key, passed in required_flags.items() if not passed]
        if reasons:
            return {
                "family_id": record.get("family_id"),
                "path": path,
                "role": role,
                "size_bytes": size_bytes,
                "retention_class": "AMBIGUOUS",
                "reason": "__".join(reasons),
                "delete_file": False,
            }
        return {
            "family_id": record.get("family_id"),
            "path": path,
            "role": role,
            "size_bytes": size_bytes,
            "retention_class": "PRUNE_ELIGIBLE_DRY_RUN_ONLY",
            "reason": "OBSOLETE_HISTORICAL_PACKAGE_VERIFIED_AFTER_DURABLE_METRICS_COMMIT",
            "delete_file": False,
        }
    return {
        "family_id": record.get("family_id"),
        "path": path,
        "role": role,
        "size_bytes": size_bytes,
        "retention_class": "AMBIGUOUS",
        "reason": "ROLE_NOT_RECOGNISED_FOR_AUTOMATED_PRUNING",
        "delete_file": False,
    }


def retention_dry_run_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [classify_retention_record(record) for record in records]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            "contract": bounded_retention_contract(),
            "rows": [],
            "summary": {
                "total_files": 0,
                "protected_files": 0,
                "prune_eligible_files": 0,
                "ambiguous_files": 0,
                "estimated_current_checkpoint_storage_bytes": 0,
                "estimated_reclaimable_storage_bytes": 0,
                "dry_run_only": True,
                "delete_files": False,
            },
        }
    summary = {
        "total_files": int(len(frame)),
        "protected_files": int((frame["retention_class"] == "PROTECTED").sum()),
        "prune_eligible_files": int((frame["retention_class"] == "PRUNE_ELIGIBLE_DRY_RUN_ONLY").sum()),
        "ambiguous_files": int((frame["retention_class"] == "AMBIGUOUS").sum()),
        "estimated_current_checkpoint_storage_bytes": int(frame.loc[frame["retention_class"] == "PROTECTED", "size_bytes"].sum()),
        "estimated_reclaimable_storage_bytes": int(frame.loc[frame["retention_class"] == "PRUNE_ELIGIBLE_DRY_RUN_ONLY", "size_bytes"].sum()),
        "dry_run_only": True,
        "delete_files": False,
    }
    return {"contract": bounded_retention_contract(), "rows": rows, "summary": summary}


def select_resume_checkpoint(records: Sequence[Mapping[str, Any]], *, family_id: str) -> dict[str, Any]:
    matches = [
        classify_retention_record(record)
        for record in records
        if str(record.get("family_id")) == family_id and (record.get("role") == "current_resumable_checkpoint" or bool(record.get("current_checkpoint", False)))
    ]
    if not matches:
        raise RuntimeError("DS24_CURRENT_RESUMABLE_CHECKPOINT_MISSING")
    chosen = sorted(matches, key=lambda row: str(row["path"]))[0]
    if chosen["retention_class"] != "PROTECTED":
        raise RuntimeError("DS24_CURRENT_RESUMABLE_CHECKPOINT_NOT_PROTECTED")
    return chosen


def select_rollback_checkpoint(records: Sequence[Mapping[str, Any]], *, family_id: str) -> dict[str, Any]:
    matches = [
        classify_retention_record(record)
        for record in records
        if str(record.get("family_id")) == family_id and (record.get("role") == "previous_verified_checkpoint" or bool(record.get("rollback_checkpoint", False)))
    ]
    if not matches:
        raise RuntimeError("DS24_PREVIOUS_VERIFIED_CHECKPOINT_MISSING")
    chosen = sorted(matches, key=lambda row: str(row["path"]))[0]
    if chosen["retention_class"] != "PROTECTED":
        raise RuntimeError("DS24_PREVIOUS_VERIFIED_CHECKPOINT_NOT_PROTECTED")
    return chosen


def ensemble_evidence_hash(records: Sequence[Mapping[str, Any]]) -> str:
    canonical_rows = sorted((dict(row) for row in records), key=lambda row: canonical_hash(row))
    return canonical_hash(canonical_rows)


def validate_pruning_preserves_ensemble_inputs(
    evidence_records: Sequence[Mapping[str, Any]],
    retention_report: Mapping[str, Any],
) -> dict[str, Any]:
    evidence_paths = {str(row.get("path")) for row in evidence_records if row.get("path") is not None}
    prune_paths = {
        str(row.get("path"))
        for row in retention_report.get("rows", [])
        if str(row.get("retention_class")) == "PRUNE_ELIGIBLE_DRY_RUN_ONLY"
    }
    overlap = sorted(evidence_paths & prune_paths)
    if overlap:
        raise RuntimeError("DS24_PRUNING_WOULD_REMOVE_ENSEMBLE_INPUTS")
    return {
        "passed": True,
        "ensemble_input_count": len(evidence_paths),
        "prune_eligible_input_overlap_count": 0,
        "before_hash": ensemble_evidence_hash(evidence_records),
        "after_safe_pruning_hash": ensemble_evidence_hash(evidence_records),
    }


def classify_learning_relationship(
    bucket_means: Mapping[str, float | None],
    *,
    accepted_evidence: bool,
    training_metadata_available: bool,
    quarantined: bool = False,
    tolerance: float = 1e-6,
) -> str:
    if quarantined:
        return "QUARANTINED_LEGACY_EVIDENCE_ONLY"
    if not accepted_evidence:
        return "INSUFFICIENT_ACCEPTED_EVIDENCE"
    if not training_metadata_available:
        return "TRAINING_HISTORY_METADATA_MISSING"
    ordered_labels = [label for label, _, _ in HISTORY_SESSION_BUCKETS] + [label for label, _, _ in HISTORY_ROW_BUCKETS]
    ordered_values = [(label, _finite_float(bucket_means.get(label))) for label in ordered_labels if _finite_float(bucket_means.get(label)) is not None]
    if len(ordered_values) < 2:
        return "STABLE_NO_MATERIAL_CHANGE"
    values = [float(value) for _, value in ordered_values]
    deltas = [right - left for left, right in zip(values, values[1:])]
    if all(abs(delta) <= tolerance for delta in deltas):
        return "STABLE_NO_MATERIAL_CHANGE"
    if all(delta >= -tolerance for delta in deltas) and any(delta > tolerance for delta in deltas):
        return "IMPROVED_WITH_MORE_HISTORY"
    if all(delta <= tolerance for delta in deltas) and any(delta < -tolerance for delta in deltas):
        return "DEGRADED_WITH_MORE_HISTORY"
    return "NONMONOTONIC_HISTORY_RELATIONSHIP"


def stable_model_bit_registry(models: Sequence[str | Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalised = [_normalise_model_item(item) for item in models]
    model_ids = [str(item["model_id"]) for item in normalised]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("duplicate model_id in exhaustive search universe")
    ordered = sorted(normalised, key=_model_sort_key)
    registry: list[dict[str, Any]] = []
    for idx, item in enumerate(ordered):
        row = dict(item)
        row["bit_position"] = idx
        row["bitmask"] = 1 << idx
        row["bitmask_hex"] = f"0x{(1 << idx):x}"
        registry.append(row)
    return registry


def expected_combination_count(model_count: int, *, method_count: int = 1) -> dict[str, int]:
    if model_count < 0:
        raise ValueError("model_count cannot be negative")
    total_subsets = (1 << model_count) - 1 if model_count else 0
    singleton_count = model_count
    ensemble_count = max(0, total_subsets - singleton_count)
    return {
        "eligible_model_count": int(model_count),
        "total_non_empty_subset_count": int(total_subsets),
        "singleton_subset_count": int(singleton_count),
        "multi_model_ensemble_subset_count": int(ensemble_count),
        "method_count_for_multi_model_subsets": int(method_count),
        "method_trial_count": int(singleton_count + ensemble_count * method_count),
    }


def enumerate_exhaustive_subsets(
    bit_registry: Sequence[Mapping[str, Any]],
    *,
    start_after_bitmask: int = 0,
    end_bitmask: int | None = None,
) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in bit_registry), key=lambda row: int(row["bit_position"]))
    model_count = len(ordered)
    if model_count == 0:
        return []
    max_bitmask = (1 << model_count) - 1
    if end_bitmask is None:
        end_bitmask = max_bitmask
    width = max(1, math.ceil(model_count / 4))
    rows: list[dict[str, Any]] = []
    for bitmask in range(int(start_after_bitmask) + 1, int(end_bitmask) + 1):
        if bitmask < 1 or bitmask > max_bitmask:
            continue
        components = [row for row in ordered if bitmask & int(row["bitmask"])]
        component_ids = [str(row["model_id"]) for row in components]
        class_counts: dict[str, int] = {}
        for row in components:
            model_class = str(row.get("model_class", "unclassified"))
            class_counts[model_class] = class_counts.get(model_class, 0) + 1
        payload = {
            "bitmask": bitmask,
            "component_model_ids": component_ids,
            "model_class_composition": class_counts,
        }
        rows.append(
            {
                "subset_id": f"subset_{bitmask:0{width}x}",
                "bitmask": bitmask,
                "bitmask_hex": f"0x{bitmask:x}",
                "subset_size": len(component_ids),
                "component_model_ids": component_ids,
                "component_model_ids_json": json.dumps(component_ids, sort_keys=True, separators=(",", ":")),
                "model_class_composition": class_counts,
                "model_class_composition_json": json.dumps(class_counts, sort_keys=True, separators=(",", ":")),
                "subset_hash": canonical_hash(payload),
            }
        )
    return rows


def validate_exhaustive_subset_registry(rows: Sequence[Mapping[str, Any]] | pd.DataFrame, *, model_count: int) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    expected = set(range(1, (1 << model_count))) if model_count else set()
    actual = set(int(value) for value in frame.get("bitmask", pd.Series(dtype=int)).tolist())
    duplicate_count = int(frame.duplicated("bitmask").sum()) if "bitmask" in frame.columns else 0
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    ordered = frame.get("bitmask", pd.Series(dtype=int)).tolist()
    return {
        "valid": not duplicate_count and not missing and not unexpected and ordered == sorted(ordered),
        "expected_subset_count": len(expected),
        "executed_subset_count": int(len(frame)),
        "duplicate_subset_count": duplicate_count,
        "missing_subset_count": len(missing),
        "unexpected_subset_count": len(unexpected),
        "deterministic_ordering": ordered == sorted(ordered),
        "missing_bitmasks": missing[:100],
        "unexpected_bitmasks": unexpected[:100],
    }


def exhaustive_trial_id(*, search_generation_id: str, subset_id: str, bitmask: int, component_model_ids: Sequence[str], method: str) -> str:
    payload = {
        "search_generation_id": search_generation_id,
        "subset_id": subset_id,
        "bitmask": int(bitmask),
        "component_model_ids": sorted(str(item) for item in component_model_ids),
        "method": method,
    }
    return "ds24_exhaustive_trial_" + canonical_hash(payload)[:20]


def build_exhaustive_trial_registry(
    bit_registry: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str] = EXHAUSTIVE_METHODS,
    search_generation_id: str = EXHAUSTIVE_SEARCH_RUN_ID,
) -> pd.DataFrame:
    subsets = enumerate_exhaustive_subsets(bit_registry)
    rows: list[dict[str, Any]] = []
    for subset in subsets:
        trial_methods = ("SINGLE_MODEL_BASELINE",) if int(subset["subset_size"]) == 1 else tuple(methods)
        for method in trial_methods:
            rows.append(
                {
                    "trial_id": exhaustive_trial_id(
                        search_generation_id=search_generation_id,
                        subset_id=str(subset["subset_id"]),
                        bitmask=int(subset["bitmask"]),
                        component_model_ids=subset["component_model_ids"],
                        method=str(method),
                    ),
                    "search_generation_id": search_generation_id,
                    "subset_id": subset["subset_id"],
                    "bitmask": int(subset["bitmask"]),
                    "bitmask_hex": subset["bitmask_hex"],
                    "subset_size": int(subset["subset_size"]),
                    "component_model_ids_json": subset["component_model_ids_json"],
                    "model_class_composition_json": subset["model_class_composition_json"],
                    "method": str(method),
                    "contains_real_performance": False,
                }
            )
    return pd.DataFrame(rows)


def validate_trial_registry(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"valid": True, "trial_count": 0, "duplicate_trial_id_count": 0}
    duplicate_count = int(frame.duplicated("trial_id").sum())
    return {
        "valid": duplicate_count == 0,
        "trial_count": int(len(frame)),
        "duplicate_trial_id_count": duplicate_count,
        "unique_trial_id_count": int(frame["trial_id"].nunique()),
    }


def _cap_and_renormalize(weights: dict[str, float], *, max_weight: float) -> dict[str, float]:
    n = len(weights)
    if n == 0:
        return {}
    if max_weight * n < 1.0 - 1e-12:
        raise ValueError("max_weight is infeasible for number of families")
    capped: dict[str, float] = {}
    remaining = dict(weights)
    while remaining:
        over = {key: value for key, value in remaining.items() if value > max_weight}
        if not over:
            total = sum(remaining.values())
            residual = 1.0 - sum(capped.values())
            if total <= 0:
                equal = residual / len(remaining)
                capped.update({key: equal for key in remaining})
            else:
                capped.update({key: value / total * residual for key, value in remaining.items()})
            break
        for key in sorted(over):
            capped[key] = max_weight
            remaining.pop(key, None)
        residual = 1.0 - sum(capped.values())
        if not remaining:
            break
        total = sum(remaining.values())
        if total <= 0:
            equal = residual / len(remaining)
            remaining = {key: equal for key in remaining}
        else:
            remaining = {key: value / total * residual for key, value in remaining.items()}
    total = sum(capped.values())
    if total:
        capped = {key: value / total for key, value in capped.items()}
    return dict(sorted(capped.items()))


def lagged_ic_nonnegative_weights(
    ic_history: pd.DataFrame,
    *,
    asof_timestamp: str,
    family_ids: Sequence[str],
    trailing_observations: int = 126,
    min_observations: int = 2,
    shrinkage_to_equal: float = 0.5,
    max_weight: float = 0.4,
) -> dict[str, Any]:
    families = sorted(str(item) for item in family_ids)
    if not families:
        return {"weights": {}, "fallback": True, "fallback_reason": "NO_FAMILIES"}
    if not 0.0 <= shrinkage_to_equal <= 1.0:
        raise ValueError("shrinkage_to_equal must be in [0, 1]")
    if max_weight * len(families) < 1.0 - 1e-12:
        raise ValueError("max_weight is infeasible for number of families")
    equal = {family: 1.0 / len(families) for family in families}
    if ic_history.empty:
        return {"weights": equal, "fallback": True, "fallback_reason": "INSUFFICIENT_LAGGED_HISTORY", "used_observations": 0}

    work = ic_history[ic_history["family_id"].isin(families)].copy()
    outcome_col = "target_available_timestamp" if "target_available_timestamp" in work.columns else "decision_timestamp"
    asof = _as_utc_timestamp(asof_timestamp)
    work["_outcome_timestamp"] = pd.to_datetime(work[outcome_col], utc=True, errors="coerce")
    work = work[work["_outcome_timestamp"] < asof].sort_values("_outcome_timestamp")
    if trailing_observations > 0:
        work = work.groupby("family_id", group_keys=False).tail(int(trailing_observations))
    counts = work.groupby("family_id")["spearman_rank_ic"].count().to_dict() if not work.empty else {}
    if any(int(counts.get(family, 0)) < min_observations for family in families):
        return {
            "weights": equal,
            "fallback": True,
            "fallback_reason": "INSUFFICIENT_LAGGED_HISTORY",
            "used_observations": int(len(work)),
        }
    means = work.groupby("family_id")["spearman_rank_ic"].mean().reindex(families).fillna(0.0)
    clipped = means.clip(lower=0.0)
    if float(clipped.sum()) <= 0.0:
        return {
            "weights": equal,
            "fallback": True,
            "fallback_reason": "NO_POSITIVE_LAGGED_IC",
            "used_observations": int(len(work)),
        }
    raw = (clipped / float(clipped.sum())).to_dict()
    shrunk = {family: (1.0 - shrinkage_to_equal) * float(raw[family]) + shrinkage_to_equal * equal[family] for family in families}
    capped = _cap_and_renormalize(shrunk, max_weight=max_weight)
    latest = work["_outcome_timestamp"].max()
    return {
        "weights": capped,
        "fallback": False,
        "fallback_reason": "",
        "used_observations": int(len(work)),
        "latest_used_outcome_timestamp": latest.isoformat() if pd.notna(latest) else None,
        "asof_timestamp": asof.isoformat(),
        "shrinkage_to_equal": float(shrinkage_to_equal),
        "max_weight": float(max_weight),
    }


def multiple_testing_adjustment(metrics: pd.DataFrame, *, p_value_col: str = "p_value", alpha: float = 0.05) -> pd.DataFrame:
    columns = [
        "trial_id",
        "raw_p_value",
        "bonferroni_p_value",
        "bh_q_value",
        "selection_significant",
        "raw_rank",
        "trial_count",
        "effective_trial_count",
    ]
    if metrics.empty:
        return pd.DataFrame(columns=columns)
    if p_value_col not in metrics.columns:
        raise ValueError(f"multiple-testing adjustment requires {p_value_col}")
    out = metrics[["trial_id", p_value_col]].copy()
    out["raw_p_value"] = pd.to_numeric(out[p_value_col], errors="coerce").clip(lower=0.0, upper=1.0)
    n = int(len(out))
    out["bonferroni_p_value"] = (out["raw_p_value"] * n).clip(upper=1.0)
    ordered = out.sort_values(["raw_p_value", "trial_id"], ascending=[True, True]).reset_index(drop=True)
    ordered["raw_rank"] = np.arange(1, n + 1)
    ordered["_bh"] = ordered["raw_p_value"] * n / ordered["raw_rank"]
    ordered["bh_q_value"] = ordered["_bh"][::-1].cummin()[::-1].clip(upper=1.0)
    ordered["selection_significant"] = ordered["bonferroni_p_value"] <= float(alpha)
    ordered["trial_count"] = n
    ordered["effective_trial_count"] = n
    return ordered[columns]


def freeze_shortlist(
    metrics: pd.DataFrame,
    *,
    max_candidates: int,
    primary_metric: str = "common_support_mean_rank_ic",
) -> dict[str, Any]:
    if max_candidates < 0:
        raise ValueError("max_candidates cannot be negative")
    if metrics.empty or max_candidates == 0:
        shortlist: list[dict[str, Any]] = []
        return {"shortlist": shortlist, "shortlist_hash": canonical_hash(shortlist), "primary_metric": primary_metric}
    work = metrics.copy()
    defaults = {
        primary_metric: -np.inf,
        "worst_fold_rank_ic": -np.inf,
        "max_drawdown": np.inf,
        "turnover": np.inf,
        "subset_size": np.inf,
        "trial_id": "",
    }
    for column, default in defaults.items():
        if column not in work.columns:
            work[column] = default
    ordered = work.sort_values(
        [primary_metric, "worst_fold_rank_ic", "max_drawdown", "turnover", "subset_size", "trial_id"],
        ascending=[False, False, True, True, True, True],
        kind="mergesort",
    )
    shortlist = ordered.head(max_candidates).to_dict("records")
    return {"shortlist": shortlist, "shortlist_hash": canonical_hash(shortlist), "primary_metric": primary_metric}


def validate_confirmation_isolation(frozen_shortlist: Mapping[str, Any], confirmation_metrics: pd.DataFrame) -> dict[str, Any]:
    allowed = {str(row.get("trial_id")) for row in frozen_shortlist.get("shortlist", [])}
    if confirmation_metrics.empty:
        return {"passed": True, "confirmation_rows": 0, "unexpected_trial_ids": []}
    observed = set(confirmation_metrics["trial_id"].astype(str))
    unexpected = sorted(observed - allowed)
    if unexpected:
        raise RuntimeError("DS24_CONFIRMATION_SEGMENT_NOT_ISOLATED")
    return {"passed": True, "confirmation_rows": int(len(confirmation_metrics)), "unexpected_trial_ids": []}


def estimate_exhaustive_resource_preflight(
    *,
    eligible_model_count: int,
    timestamp_count: int,
    security_count: int,
    method_count: int,
    chunk_timestamp_count: int = 256,
    memory_budget_bytes: int = 1_000_000_000,
    disk_budget_bytes: int = 5_000_000_000,
) -> dict[str, Any]:
    counts = expected_combination_count(eligible_model_count, method_count=method_count)
    aligned_rank_bytes = int(max(0, eligible_model_count) * max(0, timestamp_count) * max(0, security_count) * 8)
    chunk_bytes = int(max(1, eligible_model_count) * max(1, chunk_timestamp_count) * max(1, security_count) * 8)
    compact_summary_bytes = int(max(1, counts["method_trial_count"]) * 512)
    safe = aligned_rank_bytes <= disk_budget_bytes and chunk_bytes <= memory_budget_bytes
    return {
        **counts,
        "timestamp_count": int(timestamp_count),
        "security_count": int(security_count),
        "chunk_timestamp_count": int(chunk_timestamp_count),
        "estimated_aligned_rank_bytes": aligned_rank_bytes,
        "estimated_chunk_memory_bytes": chunk_bytes,
        "estimated_compact_summary_bytes": compact_summary_bytes,
        "memory_budget_bytes": int(memory_budget_bytes),
        "disk_budget_bytes": int(disk_budget_bytes),
        "safe_to_execute_real_search": bool(safe),
        "full_prediction_materialization_required": False,
    }


def validate_runtime_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = {
        "stop_workers": "STOP_WORKERS_FORBIDDEN",
        "restart_workers": "RESTART_WORKERS_FORBIDDEN",
        "alter_supervisor": "ALTER_SUPERVISOR_FORBIDDEN",
        "increase_worker_limits": "INCREASE_WORKER_LIMITS_FORBIDDEN",
        "launch_missing_models": "LAUNCH_MISSING_MODELS_FORBIDDEN",
        "write_live_namespaces": "WRITE_LIVE_NAMESPACES_FORBIDDEN",
        "submit_orders": "ORDER_GENERATION_FORBIDDEN",
        "inspect_outer_holdout": "OUTER_HOLDOUT_FORBIDDEN",
        "materialize_full_predictions": "FULL_PREDICTION_MATERIALIZATION_FORBIDDEN",
    }
    violations = [label for key, label in forbidden.items() if bool(plan.get(key, False))]
    if violations:
        raise RuntimeError("DS24_LIVE_RUNTIME_PLAN_REFUSED:" + ",".join(violations))
    return {"passed": True, "violations": [], "order_generation_interface": False}


def paired_rank_ic_difference(
    frame: pd.DataFrame,
    *,
    family_a: str,
    family_b: str,
    score_col: str = "raw_score",
    target_col: str = "target_value",
) -> dict[str, Any]:
    required = {"decision_timestamp", "security_id", "family_id", score_col, target_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"paired comparison missing columns: {missing}")
    aligned, report = strict_common_support(frame[frame["family_id"].isin([family_a, family_b])], family_ids=[family_a, family_b])
    diffs: list[float] = []
    rows: list[dict[str, Any]] = []
    for timestamp, group in aligned.groupby("decision_timestamp", sort=True):
        pivot_score = group.pivot_table(index="security_id", columns="family_id", values=score_col, aggfunc="first")
        target = group.groupby("security_id")[target_col].first()
        if family_a not in pivot_score.columns or family_b not in pivot_score.columns or len(pivot_score) < 2:
            continue
        target = target.reindex(pivot_score.index)
        ic_a = pivot_score[family_a].rank(method="average").corr(target.rank(method="average"), method="pearson")
        ic_b = pivot_score[family_b].rank(method="average").corr(target.rank(method="average"), method="pearson")
        if pd.isna(ic_a) or pd.isna(ic_b):
            continue
        diff = float(ic_a - ic_b)
        diffs.append(diff)
        rows.append({"decision_timestamp": timestamp, "rank_ic_a": float(ic_a), "rank_ic_b": float(ic_b), "rank_ic_difference": diff})
    ci = dependence_aware_ci(diffs, lag=min(12, max(0, len(diffs) - 1)))
    return {
        "family_a": family_a,
        "family_b": family_b,
        "mean_rank_ic_difference": ci["mean"],
        "difference_ci": ci,
        "common_support_report": report,
        "per_timestamp_rows": rows,
    }


def framework_safety_manifest() -> dict[str, Any]:
    return {
        "zero_full_prediction_persistence": True,
        "order_generation_interface": False,
        "paper_orders_submitted": 0,
        "live_orders_submitted": 0,
        "outer_holdout_accessed": False,
        "live_workers_mutated": False,
        "safe_to_use_while_workers_active": "SYNTHETIC_AND_FRAMEWORK_TESTS_ONLY",
    }
