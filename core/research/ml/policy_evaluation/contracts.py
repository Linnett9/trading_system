from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


EVALUATION_IDENTITY_CONTRACT_VERSION = "canonical_policy_evaluation_identity.v1"
ALIGNMENT_CONTRACT_VERSION = "canonical_policy_alignment_evidence.v1"
CANONICAL_RECORD_CONTRACT_VERSION = "canonical_policy_evaluation_record.v1"
HEADLINE_COMPARISON_CONTRACT_VERSION = "canonical_policy_headline_comparison.v1"
TRIAL_ACCOUNTING_CONTRACT_VERSION = "canonical_policy_trial_accounting.v1"

SUPPORTED_POLICY_FAMILIES = (
    "current_champion",
    "no_trade_current_holdings",
    "random_shuffled_ranking",
    "momentum",
    "risk_adjusted_momentum",
    "linear_model",
    "tree_model",
    "sequence_model",
    "ensemble",
    "news",
    "no_news",
    "optimiser",
    "future_challenger",
)

REQUIRED_ALIGNMENT_FIELDS = (
    "decision_dates",
    "eligible_assets_checksum",
    "oos_rows_checksum",
    "target",
    "maturity",
    "cost_scenario",
    "capacity_scenario",
    "constraints",
    "exposure",
    "execution_timing",
    "dataset_manifest",
    "authority_versions",
)

CANONICAL_METRIC_FIELDS = (
    "return",
    "sharpe",
    "sortino",
    "calmar",
    "drawdown",
    "turnover",
    "trades",
    "holdings",
    "cash",
    "costs",
    "breakeven_costs",
    "capacity",
    "ic",
    "coverage",
    "width",
    "seed_stability",
    "worst_period",
    "worst_group",
    "dsr",
    "pbo",
    "trial_family_count",
)

_METRIC_ALIASES = {
    "return": (
        "return",
        "total_return",
        "net_return",
        "net_cumulative_return",
        "cumulative_return",
        "canonical_continuous_return",
        "wealth_multiple_return",
    ),
    "sharpe": ("sharpe", "net_sharpe"),
    "sortino": ("sortino", "net_sortino"),
    "calmar": ("calmar", "calmar_ratio"),
    "drawdown": ("drawdown", "max_drawdown", "maximum_drawdown"),
    "turnover": ("turnover", "average_turnover", "annualised_turnover"),
    "trades": ("trades", "trade_count", "number_of_trades"),
    "holdings": (
        "holdings",
        "average_holdings",
        "average_number_of_positions",
        "average_positions",
        "holding_count",
    ),
    "cash": ("cash", "average_cash", "cash_weight"),
    "costs": (
        "costs",
        "cost_drag",
        "transaction_costs",
        "estimated_transaction_costs",
        "transaction_cost_drag",
    ),
    "breakeven_costs": (
        "breakeven_costs",
        "breakeven_cost_bps",
        "break_even_cost_bps",
        "break_even_transaction_cost_bps",
    ),
    "capacity": ("capacity", "capacity_status", "adv_capacity_status"),
    "ic": (
        "ic",
        "mean_rank_ic",
        "mean_spearman_ic",
        "spearman_rank_ic",
        "rank_ic",
    ),
    "coverage": (
        "coverage",
        "prediction_coverage",
        "coverage_ratio",
        "positive_rank_ic_fraction",
    ),
    "width": ("width", "prediction_interval_width", "mean_width"),
    "seed_stability": ("seed_stability", "rank_stability_mean_spearman"),
    "worst_period": (
        "worst_period",
        "worst_period_return",
        "worst_20_period_drawdown",
    ),
    "worst_group": ("worst_group", "worst_window", "worst_seed"),
    "dsr": ("dsr", "deflated_sharpe_probability"),
    "pbo": ("pbo", "probability_of_backtest_overfitting"),
    "trial_family_count": (
        "trial_family_count",
        "model_family_count",
        "effective_trial_count",
        "effective_search_count",
    ),
}


class AlignmentError(ValueError):
    """Raised when a caller demands headline comparison for misaligned records."""

    def __init__(self, report: Mapping[str, Any]):
        super().__init__("Headline comparison is blocked by misaligned evaluation evidence")
        self.report = dict(report)


def canonical_json(payload: Any) -> str:
    return json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest().upper()


def normalise_evaluation_identity(
    *,
    policy: str,
    model: str,
    target: str,
    universe: str,
    timeframe: str,
    evaluation_period: Mapping[str, Any] | Sequence[Any],
    cost_scenario: str,
    capacity_scenario: str,
    seed: str | int | None,
    dataset_authority_version: str,
    policy_family: str,
) -> dict[str, Any]:
    if policy_family not in SUPPORTED_POLICY_FAMILIES:
        raise ValueError(f"Unsupported canonical policy family: {policy_family}")
    logical = {
        "contract_version": EVALUATION_IDENTITY_CONTRACT_VERSION,
        "policy": _required_text(policy, "policy"),
        "policy_family": policy_family,
        "model": _required_text(model, "model"),
        "target": _required_text(target, "target"),
        "universe": _required_text(universe, "universe"),
        "timeframe": _required_text(timeframe, "timeframe"),
        "evaluation_period": _normalise_period(evaluation_period),
        "cost_scenario": _required_text(cost_scenario, "cost_scenario"),
        "capacity_scenario": _required_text(capacity_scenario, "capacity_scenario"),
        "seed": "not_applicable" if seed in (None, "") else str(seed),
        "dataset_authority_version": _required_text(
            dataset_authority_version, "dataset_authority_version"
        ),
    }
    logical["identity_hash"] = canonical_hash(logical)
    return logical


def normalise_alignment_evidence(
    *,
    decision_dates: Sequence[Any] = (),
    eligible_assets: Mapping[str, Sequence[Any]] | None = None,
    eligible_assets_checksum: str | None = None,
    oos_rows: Sequence[Any] | None = None,
    oos_rows_checksum: str | None = None,
    target: str | None = None,
    maturity: Mapping[str, Any] | str | None = None,
    cost_scenario: Mapping[str, Any] | str | None = None,
    capacity_scenario: Mapping[str, Any] | str | None = None,
    constraints: Mapping[str, Any] | str | None = None,
    exposure: Mapping[str, Any] | str | None = None,
    execution_timing: Mapping[str, Any] | str | None = None,
    dataset_manifest: Mapping[str, Any] | str | None = None,
    authority_versions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    dates = _normalise_sequence(decision_dates)
    if not dates:
        blockers.append("DECISION_DATES_REQUIRED")

    eligible_checksum = eligible_assets_checksum or (
        canonical_hash(_normalise_assets(eligible_assets)) if eligible_assets else None
    )
    if not eligible_checksum:
        blockers.append("ELIGIBLE_ASSETS_REQUIRED")

    rows_checksum = oos_rows_checksum or (
        canonical_hash(_normalise_sequence(oos_rows or ())) if oos_rows else None
    )
    if not rows_checksum:
        blockers.append("OOS_ROWS_REQUIRED")

    manifest = _normalise_dataset_manifest(dataset_manifest)
    if not manifest.get("manifest_id") or not manifest.get("manifest_checksum"):
        blockers.append("DATASET_LINEAGE_REQUIRED")

    versions = {
        str(key): str(value)
        for key, value in sorted(dict(authority_versions or {}).items())
        if value not in (None, "")
    }
    if not versions:
        blockers.append("AUTHORITY_VERSIONS_REQUIRED")

    maturity_payload = _jsonable(maturity if maturity is not None else {})
    if not maturity_payload:
        blockers.append("TARGET_MATURITY_REQUIRED")

    logical = {
        "contract_version": ALIGNMENT_CONTRACT_VERSION,
        "decision_dates": dates,
        "decision_dates_checksum": canonical_hash(dates),
        "eligible_assets_checksum": eligible_checksum,
        "oos_rows_checksum": rows_checksum,
        "target": _text(target),
        "maturity": maturity_payload,
        "cost_scenario": _jsonable(cost_scenario),
        "capacity_scenario": _jsonable(capacity_scenario),
        "constraints": _jsonable(constraints),
        "exposure": _jsonable(exposure),
        "execution_timing": _jsonable(execution_timing),
        "dataset_manifest": manifest,
        "authority_versions": versions,
        "alignment_blockers": sorted(set(blockers)),
    }
    for key in (
        "target",
        "cost_scenario",
        "capacity_scenario",
        "constraints",
        "exposure",
        "execution_timing",
    ):
        if logical[key] in (None, "", {}, []):
            logical["alignment_blockers"].append(f"{key.upper()}_REQUIRED")
    logical["alignment_blockers"] = sorted(set(logical["alignment_blockers"]))
    logical["alignment_hash"] = canonical_hash(
        {key: logical[key] for key in REQUIRED_ALIGNMENT_FIELDS}
    )
    return logical


def normalise_metrics(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    source = dict(metrics or {})
    canonical: dict[str, Any] = {}
    for field in CANONICAL_METRIC_FIELDS:
        canonical[field] = _metric_value(source, field)
    canonical["metric_contract_version"] = "canonical_policy_metric_bundle.v1"
    canonical["raw_metric_fields"] = sorted(str(key) for key in source)
    canonical["missing_metric_fields"] = [
        field for field in CANONICAL_METRIC_FIELDS if canonical[field] is None
    ]
    canonical["available_metric_fields"] = [
        field for field in CANONICAL_METRIC_FIELDS if canonical[field] is not None
    ]
    return canonical


def normalise_trial_accounting(evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(evidence or {})
    family = payload.get("trial_family") if isinstance(payload.get("trial_family"), Mapping) else {}
    family_counts = family.get("counts", {}) if isinstance(family, Mapping) else {}
    locked_holdout = (
        payload.get("locked_holdout")
        if isinstance(payload.get("locked_holdout"), Mapping)
        else payload.get("outer_holdout")
        if isinstance(payload.get("outer_holdout"), Mapping)
        else {}
    )
    dsr = _safeguard_metric(
        payload.get("dsr_evidence") or payload.get("dsr"),
        "deflated_sharpe_probability",
    )
    pbo = _safeguard_metric(
        payload.get("pbo_evidence") or payload.get("pbo"),
        "probability_of_backtest_overfitting",
    )
    logical = {
        "contract_version": TRIAL_ACCOUNTING_CONTRACT_VERSION,
        "status": "EXPLICIT" if payload else "MISSING",
        "trial_family_id": payload.get("trial_family_id") or family.get("family_id"),
        "trial_family_identity": payload.get("trial_family_identity")
        or family.get("family_identity"),
        "dsr": dsr,
        "pbo": pbo,
        "effective_search_count": _first_present(
            payload,
            "effective_search_count",
            "dsr_effective_search_count",
            "effective_trial_count",
        )
        or _first_present(family_counts, "counted_trial_count", "material_effective_search_count"),
        "raw_trial_count": _first_present(
            payload, "raw_trial_count", "logical_trial_count", "trial_count"
        )
        or _first_present(family_counts, "raw_trial_count", "logical_trial_count"),
        "trial_family_count": _first_present(
            payload, "trial_family_count", "model_family_count"
        )
        or _first_present(family_counts, "counted_trial_count", "material_effective_search_count"),
        "trial_family_ids": sorted(
            str(value)
            for value in (
                payload.get("trial_family_ids")
                or ([payload.get("trial_family_id")] if payload.get("trial_family_id") else [])
            )
        ),
        "accounting_result_checksum": payload.get("accounting_result_checksum")
        or payload.get("logical_result_checksum")
        or family.get("logical_result_checksum"),
        "attempt_status_counts": payload.get("attempt_status_counts")
        or family_counts.get("status_counts")
        or {},
        "failed_trials_accounted": payload.get("failed_trials_accounted"),
        "skipped_trials_accounted": payload.get("skipped_trials_accounted"),
        "included_trial_ids": sorted(
            str(value)
            for value in (
                payload.get("included_trial_ids")
                or family.get("included_trial_ids")
                or ()
            )
        ),
        "trial_family_contract_version": payload.get("trial_family_contract_version")
        or family.get("contract_version"),
        "trial_family_complete": payload.get("trial_family_complete")
        if "trial_family_complete" in payload
        else family.get("valid"),
        "adjusted_evidence_status": payload.get("adjusted_evidence_status"),
        "multiplicity_evidence": payload.get("multiplicity_evidence")
        or payload.get("multiplicity_adjusted_benchmark_tests"),
        "dsr_family_size_input": payload.get("dsr_family_size_input"),
        "pbo_family_definition": payload.get("pbo_family_definition") or {},
        "dsr_evidence_hash": payload.get("dsr_evidence_hash"),
        "pbo_evidence_hash": payload.get("pbo_evidence_hash"),
        "multiplicity_evidence_hash": payload.get("multiplicity_evidence_hash"),
        "holdout_identity": payload.get("holdout_identity")
        or locked_holdout.get("holdout_identity"),
        "holdout_valid": payload.get("holdout_valid")
        if "holdout_valid" in payload
        else (
            locked_holdout.get("invalidation_status") == "VALID"
            if locked_holdout
            else None
        ),
        "holdout_invalidation_status": payload.get("holdout_invalidation_status")
        or locked_holdout.get("invalidation_status"),
        "holdout_reuse_attempt_count": _first_present(
            payload, "holdout_reuse_attempt_count"
        )
        or locked_holdout.get("reuse_attempt_count"),
        "holdout_invalidation_reasons": payload.get("holdout_invalidation_reasons")
        or locked_holdout.get("blocking_reasons")
        or [],
    }
    missing = []
    if dsr is None:
        missing.append("DSR_EVIDENCE_MISSING")
    if pbo is None:
        missing.append("PBO_EVIDENCE_MISSING")
    if logical["trial_family_count"] is None:
        missing.append("TRIAL_FAMILY_COUNT_MISSING")
    if not logical["trial_family_id"]:
        missing.append("TRIAL_FAMILY_MISSING")
    if logical["trial_family_complete"] is False:
        missing.append("TRIAL_FAMILY_ACCOUNTING_INCOMPLETE")
        missing.append("TRIAL_ATTEMPTS_INCOMPLETE")
    if logical["failed_trials_accounted"] is False:
        missing.append("FAILED_TRIALS_NOT_ACCOUNTED")
    if logical["skipped_trials_accounted"] is False:
        missing.append("SKIPPED_TRIALS_NOT_ACCOUNTED")
    if (
        logical["trial_family_contract_version"] == "trial_family_accounting_v1"
        and not logical["multiplicity_evidence"]
    ):
        missing.append("MULTIPLICITY_EVIDENCE_MISSING")
    if logical["holdout_invalidation_status"] == "INVALIDATED":
        missing.append("LOCKED_HOLDOUT_INVALIDATED")
        missing.append("LOCKED_HOLDOUT_REUSED")
    if any("UNAUTHORIZED" in str(reason).upper() for reason in logical["holdout_invalidation_reasons"]):
        missing.append("HOLDOUT_ACCESS_UNAUTHORISED")
    if (
        logical["effective_search_count"] is not None
        and logical["trial_family_count"] is not None
        and logical["effective_search_count"] != logical["trial_family_count"]
    ):
        missing.append("DSR_FAMILY_MISMATCH")
    if (
        logical["dsr_family_size_input"] is not None
        and logical["trial_family_count"] is not None
        and logical["dsr_family_size_input"] != logical["trial_family_count"]
    ):
        missing.append("DSR_FAMILY_MISMATCH")
    pbo_family = logical["pbo_family_definition"]
    if isinstance(pbo_family, Mapping):
        declared = pbo_family.get("declared_family_size")
        pbo_ids = sorted(str(value) for value in pbo_family.get("included_trial_ids", []) or [])
        if declared is not None and logical["trial_family_count"] is not None and declared != logical["trial_family_count"]:
            missing.append("PBO_FAMILY_MISMATCH")
        if pbo_ids and logical["included_trial_ids"] and pbo_ids != logical["included_trial_ids"]:
            missing.append("PBO_FAMILY_MISMATCH")
    logical["evidence_complete"] = not missing
    logical["missing_evidence"] = sorted(set(missing))
    logical["trial_accounting_hash"] = canonical_hash(logical)
    return logical


def build_policy_evaluation_record(
    *,
    identity: Mapping[str, Any],
    alignment: Mapping[str, Any],
    metrics: Mapping[str, Any] | None,
    source_surface: str,
    source_artifact: str | None = None,
    trial_accounting: Mapping[str, Any] | None = None,
    no_trade: bool | None = None,
    source_promotion_eligible: bool = False,
) -> dict[str, Any]:
    if identity.get("contract_version") != EVALUATION_IDENTITY_CONTRACT_VERSION:
        raise ValueError("Canonical evaluation identity is required")
    if alignment.get("contract_version") != ALIGNMENT_CONTRACT_VERSION:
        raise ValueError("Canonical alignment evidence is required")
    family = str(identity.get("policy_family"))
    no_trade_policy = bool(
        no_trade if no_trade is not None else family == "no_trade_current_holdings"
    )
    logical = {
        "contract_version": CANONICAL_RECORD_CONTRACT_VERSION,
        "identity": dict(identity),
        "alignment": dict(alignment),
        "metrics": normalise_metrics(metrics),
        "trial_accounting": normalise_trial_accounting(trial_accounting),
        "source": {
            "surface": _required_text(source_surface, "source_surface"),
            "artifact": source_artifact,
        },
        "no_trade_policy": no_trade_policy,
        "promotion": {
            "automatic_promotion": False,
            "source_promotion_eligible": bool(source_promotion_eligible),
            "promotion_decision": "NO_AUTOMATIC_PROMOTION",
        },
    }
    logical["record_validation"] = {
        "alignment_blockers": list(alignment.get("alignment_blockers", ())),
        "metrics_are_schema_complete": set(logical["metrics"]) >= set(CANONICAL_METRIC_FIELDS),
        "trial_evidence_complete": logical["trial_accounting"]["evidence_complete"],
    }
    logical["record_checksum"] = canonical_hash(logical)
    return logical


def headline_alignment_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(record) for record in records]
    blockers: list[str] = []
    field_values: dict[str, list[dict[str, Any]]] = {}
    if len(rows) < 2:
        blockers.append("HEADLINE_COMPARISON_REQUIRES_AT_LEAST_TWO_RECORDS")
    for record in rows:
        identity = record.get("identity", {})
        alignment = record.get("alignment", {})
        record_id = str(identity.get("identity_hash") or record.get("record_checksum") or "")
        for reason in alignment.get("alignment_blockers", ()):
            blockers.append(f"RECORD_ALIGNMENT_EVIDENCE_INCOMPLETE:{record_id}:{reason}")
        trial = record.get("trial_accounting", {})
        if isinstance(trial, Mapping) and trial.get("evidence_complete") is not True:
            missing = trial.get("missing_evidence", []) or []
            suffix = ",".join(str(reason) for reason in missing) or "UNKNOWN"
            blockers.append(f"TRIAL_EVIDENCE_INCOMPLETE:{record_id}:{suffix}")
        if isinstance(trial, Mapping):
            family_identity = (
                trial.get("trial_family_identity")
                or trial.get("trial_family_id")
                or ",".join(str(value) for value in trial.get("trial_family_ids", []) or [])
            )
            field_values.setdefault("trial_family_identity", []).append(
                {
                    "record_identity": record_id,
                    "value_hash": canonical_hash(family_identity),
                    "value": family_identity,
                }
            )
        for field in REQUIRED_ALIGNMENT_FIELDS:
            value = alignment.get(field)
            field_values.setdefault(field, []).append(
                {
                    "record_identity": record_id,
                    "value_hash": canonical_hash(value),
                    "value": value,
                }
            )
    misaligned = []
    for field, values in sorted(field_values.items()):
        hashes = {row["value_hash"] for row in values}
        if len(hashes) > 1:
            reason = f"MISALIGNED_{field.upper()}"
            blockers.append(reason)
            misaligned.append({"field": field, "reason": reason, "values": values})
    allowed = not blockers
    logical = {
        "contract_version": HEADLINE_COMPARISON_CONTRACT_VERSION,
        "status": "ALIGNED" if allowed else "BLOCKED_HEADLINE_COMPARISON",
        "headline_comparison_allowed": allowed,
        "record_count": len(rows),
        "blocking_reasons": sorted(set(blockers)),
        "misaligned_fields": misaligned,
        "comparison_set_checksum": canonical_hash(
            sorted(str(row.get("record_checksum", "")) for row in rows)
        ),
        "no_automatic_promotion": True,
    }
    logical["alignment_report_checksum"] = canonical_hash(logical)
    return logical


def require_headline_alignment(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    report = headline_alignment_report(records)
    if not report["headline_comparison_allowed"]:
        raise AlignmentError(report)
    return report


def build_headline_comparison(
    records: Sequence[Mapping[str, Any]], *, sort_metric: str = "return"
) -> dict[str, Any]:
    if sort_metric not in CANONICAL_METRIC_FIELDS:
        raise ValueError(f"Unsupported headline metric: {sort_metric}")
    report = headline_alignment_report(records)
    if not report["headline_comparison_allowed"]:
        return {
            **report,
            "leaderboard": [],
            "ranking_basis": [],
            "promotion": {"automatic_promotion": False},
        }
    leaderboard = sorted(
        (
            {
                "policy": record["identity"]["policy"],
                "policy_family": record["identity"]["policy_family"],
                "model": record["identity"]["model"],
                "seed": record["identity"]["seed"],
                "record_checksum": record["record_checksum"],
                "no_trade_policy": record["no_trade_policy"],
                **{field: record["metrics"][field] for field in CANONICAL_METRIC_FIELDS},
            }
            for record in records
        ),
        key=lambda row: (_descending(row.get(sort_metric)), row["policy"], row["model"], row["seed"]),
    )
    return {
        **report,
        "leaderboard": [
            {"rank": rank, **row} for rank, row in enumerate(leaderboard, start=1)
        ],
        "ranking_basis": [f"{sort_metric}_desc", "policy_asc", "model_asc", "seed_asc"],
        "promotion": {"automatic_promotion": False},
    }


def _metric_value(source: Mapping[str, Any], field: str) -> Any:
    for name in _METRIC_ALIASES[field]:
        if name in source:
            value = source[name]
            if isinstance(value, str):
                stripped = value.strip()
                if stripped == "":
                    return None
                try:
                    number = float(stripped)
                except ValueError:
                    return stripped
                return number if math.isfinite(number) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value) if math.isfinite(float(value)) else None
            if value is not None:
                return _jsonable(value)
    return None


def _safeguard_metric(value: Any, metric: str) -> Any:
    if isinstance(value, Mapping):
        if value.get("valid") is False:
            return None
        metrics = value.get("result_metrics")
        if isinstance(metrics, Mapping) and metric in metrics:
            return _finite_metric(metrics[metric])
        if metric in value:
            return _finite_metric(value[metric])
    return _finite_metric(value)


def _finite_metric(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            number = float(stripped)
        except ValueError:
            return stripped
        return number if math.isfinite(number) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(float(value)) else None
    return None


def _first_present(payload: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = payload.get(name)
        if value not in (None, ""):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value) if float(value).is_integer() else float(value)
            return value
    return None


def _normalise_period(value: Mapping[str, Any] | Sequence[Any]) -> dict[str, str]:
    if isinstance(value, Mapping):
        start = value.get("start") or value.get("first") or value.get("from")
        end = value.get("end") or value.get("last") or value.get("to")
    else:
        items = list(value)
        if len(items) != 2:
            raise ValueError("evaluation_period must contain exactly two boundaries")
        start, end = items
    start_text = _required_text(start, "evaluation_period.start")
    end_text = _required_text(end, "evaluation_period.end")
    if start_text > end_text:
        raise ValueError("evaluation_period start cannot exceed end")
    return {"start": start_text, "end": end_text}


def _normalise_dataset_manifest(value: Mapping[str, Any] | str | None) -> dict[str, Any]:
    if isinstance(value, Mapping):
        manifest_id = (
            value.get("manifest_id")
            or value.get("dataset_id")
            or value.get("id")
            or value.get("path")
        )
        checksum = (
            value.get("manifest_checksum")
            or value.get("dataset_checksum")
            or value.get("logical_checksum")
            or value.get("sha256")
        )
        authority = (
            value.get("authority_version")
            or value.get("dataset_authority_version")
            or value.get("contract_version")
        )
        return {
            "manifest_id": _text(manifest_id),
            "manifest_checksum": _text(checksum),
            "authority_version": _text(authority),
        }
    if value not in (None, ""):
        return {
            "manifest_id": str(value),
            "manifest_checksum": None,
            "authority_version": None,
        }
    return {"manifest_id": None, "manifest_checksum": None, "authority_version": None}


def _normalise_assets(value: Mapping[str, Sequence[Any]] | None) -> dict[str, list[str]]:
    return {
        str(date): sorted(str(asset).upper() for asset in assets)
        for date, assets in sorted(dict(value or {}).items())
    }


def _normalise_sequence(values: Sequence[Any]) -> list[str]:
    return [str(value) for value in values]


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _descending(value: Any) -> tuple[int, float]:
    if value is None or isinstance(value, bool):
        return (1, math.inf)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return (1, math.inf)
    return (0, -number) if math.isfinite(number) else (1, math.inf)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite float is not canonical JSON: {value!r}")
        return value
    return value
