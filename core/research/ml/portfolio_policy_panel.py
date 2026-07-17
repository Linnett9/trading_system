"""Versioned Wave 4 portfolio-policy panel and pure deterministic mechanics."""
from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.registries.io import canonical_hash

CONTRACT = "horizon_aligned_portfolio_policy.v1"
PANEL_CONTRACT = "horizon_aligned_portfolio_policy_panel.v1"
PANEL_ID = "wave4-horizon-aligned-policy-panel-v1"
POLICY_IDS = (
    "daily_top20_control",
    "staggered_10_session_top10",
    "staggered_10_session_top20",
    "staggered_10_session_top40",
    "rank_hysteresis_20_30",
    "turnover_penalised_aim_v1",
)
COST_SCENARIOS_BPS = (5, 10, 25, 50)
CAPACITY_SCENARIOS_ADV = (0.01, 0.025, 0.05)
WEIGHTING_METHODS = {"equal_weight", "staggered_equal_weight", "hysteresis_equal_weight", "bounded_aim"}
COST_MODEL = "linear_one_way_turnover_bps_v1"


def build_policy_panel(*, source_commit: str, output_path: Path | None = None) -> dict[str, Any]:
    if not source_commit:
        raise ValueError("source_commit is required")
    bundle = load_registry_bundle()
    resolver = RegistryResolver(bundle)
    policies = []
    for policy_id in POLICY_IDS:
        resolution = resolver.resolve("portfolio_policies", policy_id, role="portfolio_policy")
        policy = dict(resolution.entry.payload["policy_contract"])
        validate_policy(policy)
        policies.append({
            "policy_id": policy_id, "policy_checksum": policy_checksum(policy),
            "registry_entry_hash": resolution.entry.entry_hash, "policy": policy,
        })
    manifest = {
        "panel_contract_version": PANEL_CONTRACT, "policy_panel_id": PANEL_ID,
        "policy_count": 6, "policies": policies,
        "cost_scenarios_bps": list(COST_SCENARIOS_BPS),
        "capacity_scenarios_adv_participation": list(CAPACITY_SCENARIOS_ADV),
        "source_registry_checksum": bundle.documents["portfolio_policies"].registry_hash,
        "source_commit": source_commit,
    }
    manifest["panel_checksum"] = canonical_hash(manifest)
    if output_path is not None:
        _atomic_json(output_path, manifest)
    return manifest


def validate_policy(policy: Mapping[str, Any]) -> None:
    required = {
        "policy_id", "policy_contract_version", "selection_size",
        "holding_horizon_sessions", "cohort_count", "cohort_capital_fraction",
        "entry_rank", "retention_rank", "exit_rank", "weighting_method",
        "maximum_stock_weight", "maximum_sector_weight", "maximum_turnover",
        "cost_bps", "adv_participation_limit", "liquidity_eligibility_rule",
        "source_prediction_contract", "calculation_version", "cost_model",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ValueError(f"Policy fields missing: {missing}")
    if policy["policy_contract_version"] != CONTRACT:
        raise ValueError("Unknown policy contract")
    if policy["cost_model"] != COST_MODEL:
        raise ValueError("Unknown cost model")
    if policy["weighting_method"] not in WEIGHTING_METHODS:
        raise ValueError("Unknown weighting method")
    if float(policy["adv_participation_limit"]) not in CAPACITY_SCENARIOS_ADV:
        raise ValueError("ADV participation limit is outside the allowed panel")
    if int(policy["cohort_count"]) > 0 and not math.isclose(
        int(policy["cohort_count"]) * float(policy["cohort_capital_fraction"]), 1.0
    ):
        raise ValueError("Invalid cohort fraction")
    ranks = [policy.get(name) for name in ("entry_rank", "retention_rank", "exit_rank")]
    if any(value is not None for value in ranks):
        if any(value is None for value in ranks) or not (
            0 < int(ranks[0]) <= int(ranks[1]) == int(ranks[2])
        ):
            raise ValueError("Invalid rank thresholds")


def policy_checksum(policy: Mapping[str, Any]) -> str:
    validate_policy(policy)
    return canonical_hash(dict(policy))


def daily_top_k_target_weights(
    ranked_symbols: Sequence[str], *, k: int, exposure_target: float = 1.0,
) -> dict[str, float]:
    selected = list(dict.fromkeys(map(str, ranked_symbols)))[:k]
    return _equal(selected, exposure_target)


def staggered_cohort_target_weights(
    cohorts: Sequence[Mapping[str, Any]], *, current_session: int,
    holding_horizon_sessions: int = 10, exposure_target: float = 1.0,
) -> dict[str, float]:
    active = [
        row for row in cohorts
        if 0 <= current_session - int(row["entry_session"]) < holding_horizon_sessions
    ]
    result: dict[str, float] = {}
    fraction = exposure_target / holding_horizon_sessions
    for cohort in active:
        symbols = list(dict.fromkeys(map(str, cohort["symbols"])))
        for symbol, weight in _equal(symbols, fraction).items():
            result[symbol] = result.get(symbol, 0.0) + weight
    return result


def hysteresis_target_weights(
    ranks: Mapping[str, int], current_holdings: Mapping[str, float], *,
    entry_rank: int = 20, retention_rank: int = 30, exposure_target: float = 1.0,
) -> dict[str, float]:
    selected = {
        symbol for symbol, rank in ranks.items()
        if int(rank) <= entry_rank or (symbol in current_holdings and int(rank) <= retention_rank)
    }
    return _equal(sorted(selected), exposure_target)


def turnover_limited_aim_weights(
    alpha: Mapping[str, float], covariance: Mapping[str, Mapping[str, float]],
    current: Mapping[str, float], sectors: Mapping[str, str], *,
    transaction_cost_penalty: float, maximum_stock_weight: float,
    maximum_sector_weight: float, maximum_turnover: float,
    liquidity_eligible: Mapping[str, bool], exposure_target: float = 1.0,
) -> dict[str, float]:
    eligible = [symbol for symbol in alpha if liquidity_eligible.get(symbol, False)]
    scores = {
        symbol: max(0.0, float(alpha[symbol])) / (
            max(float(covariance.get(symbol, {}).get(symbol, 0.0)), 0.0) ** 0.5
            + max(float(transaction_cost_penalty), 0.0) + 1e-12
        ) for symbol in eligible
    }
    total = sum(scores.values())
    aim = {symbol: min(maximum_stock_weight, exposure_target * score / total) for symbol, score in scores.items()} if total else {}
    for sector in sorted(set(sectors.get(symbol, "") for symbol in aim)):
        members = [symbol for symbol in aim if sectors.get(symbol, "") == sector]
        sector_total = sum(aim[symbol] for symbol in members)
        if sector_total > maximum_sector_weight:
            scale = maximum_sector_weight / sector_total
            for symbol in members:
                aim[symbol] *= scale
    turnover = sum(abs(aim.get(symbol, 0.0) - current.get(symbol, 0.0)) for symbol in set(aim) | set(current))
    if turnover > maximum_turnover:
        scale = maximum_turnover / turnover
        aim = {
            symbol: current.get(symbol, 0.0) + scale * (aim.get(symbol, 0.0) - current.get(symbol, 0.0))
            for symbol in set(aim) | set(current)
        }
    return {symbol: weight for symbol, weight in sorted(aim.items()) if weight > 1e-15}


def transaction_cost(turnover: float, cost_bps: float, *, portfolio_value: float = 1.0) -> float:
    if turnover < 0 or cost_bps not in COST_SCENARIOS_BPS or portfolio_value < 0:
        raise ValueError("Invalid transaction-cost inputs")
    return portfolio_value * turnover * cost_bps / 10_000.0


def clip_trades_to_adv_capacity(
    intended_trades: Mapping[str, float], adv: Mapping[str, float], *,
    participation_limit: float,
) -> dict[str, float]:
    if participation_limit not in CAPACITY_SCENARIOS_ADV:
        raise ValueError("ADV participation limit is outside the allowed panel")
    return {
        symbol: math.copysign(min(abs(float(trade)), max(0.0, float(adv.get(symbol, 0.0))) * participation_limit), float(trade))
        for symbol, trade in sorted(intended_trades.items())
    }


def validate_replay_lineage(evidence: Mapping[str, Any]) -> None:
    required = (
        "component_plan_checksum", "campaign_id", "experiment_ledger_checksum",
        "dataset_id", "daily_spine_id", "symbol_registry_id", "feature_schema_hash",
        "target_contract_hash", "model_id", "decision_date", "source_commit",
    )
    if evidence.get("wave4_gate_status") != "READY_FOR_PORTFOLIO_REPLAY":
        raise ValueError("Wave 4 gate is not READY_FOR_PORTFOLIO_REPLAY")
    if evidence.get("strict_oos_verification") != "VERIFIED_STRICT_OOS":
        raise ValueError("Strict-OOS evidence is missing")
    if evidence.get("target_provenance_contract_version") != "stock_level_target_provenance_v2":
        raise ValueError("Target provenance v2 is required")
    if evidence.get("experiment_ledger_identity_matches") is not True:
        raise ValueError("Experiment-ledger evidence mismatch")
    if evidence.get("complete_finite_prediction_coverage") is not True:
        raise ValueError("Prediction coverage is incomplete")
    missing = [field for field in required if not evidence.get(field)]
    if missing:
        raise ValueError(f"Replay lineage fields missing: {missing}")


def _equal(symbols: Sequence[str], exposure: float) -> dict[str, float]:
    if exposure < 0:
        raise ValueError("Long-only exposure cannot be negative")
    weight = exposure / len(symbols) if symbols else 0.0
    return {symbol: weight for symbol in symbols}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
