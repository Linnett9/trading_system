from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from math import isfinite
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelTriggeredRebalanceResult:
    as_of_timestamp: str
    policy: str
    model_or_champion_source: str
    evaluate_every_run: bool
    fixed_schedule_gate_bypassed_for_candidate_evaluation: bool
    current_holdings: dict[str, float]
    candidate_holdings: dict[str, float]
    current_score: float
    candidate_score: float
    gross_improvement: float
    estimated_transaction_cost: float
    estimated_slippage: float
    net_improvement: float
    turnover: float
    replacement_count: int
    risk_gate_passed: bool
    confidence_gate_passed: bool
    persistence_gate_passed: bool
    cooldown_gate_passed: bool
    decision: str
    reason_codes: tuple[str, ...]
    paper_submit_requested: bool
    paper_submit_allowed: bool
    orders_submitted: bool = False
    live_trading_invoked: bool = False
    champion_mutation_invoked: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        return payload

    def with_submission_result(self, submitted: bool) -> "ModelTriggeredRebalanceResult":
        return replace(
            self,
            decision=(
                "PAPER_REBALANCE_SUBMITTED"
                if submitted
                else self.decision
            ),
            orders_submitted=submitted,
        )


def validate_rebalance_policy(config: Mapping[str, Any]) -> str:
    ml = config.get("ml", {}) or {}
    policy = str(ml.get("rebalance_policy", "fixed_schedule"))
    if policy not in {"fixed_schedule", "model_triggered"}:
        raise ValueError(f"Unknown ml.rebalance_policy: {policy}")
    if policy == "fixed_schedule":
        return policy
    settings = ml.get("model_triggered_rebalance") or {}
    if settings.get("paper_only") is not True:
        raise ValueError("model_triggered rebalance requires paper_only=true")
    if settings.get("safety", {}).get("allow_live_trading") is not False:
        raise ValueError("model_triggered rebalance requires allow_live_trading=false")
    if settings.get("safety", {}).get("allow_champion_mutation") is not False:
        raise ValueError("model_triggered rebalance requires allow_champion_mutation=false")
    if str(config.get("trading", {}).get("mode", "paper")) == "live":
        raise RuntimeError("Model-triggered rebalance is unavailable in live trading")
    return policy


def evaluate_model_triggered_rebalance(
    config: Mapping[str, Any],
    decision: Any,
    *,
    submit_requested: bool,
) -> ModelTriggeredRebalanceResult | None:
    if validate_rebalance_policy(config) == "fixed_schedule":
        return None
    settings = config["ml"]["model_triggered_rebalance"]
    if settings.get("evaluate_every_run") is not True:
        raise ValueError(
            "model_triggered paper rebalance requires evaluate_every_run=true"
        )
    gates = settings.get("gates", {})
    costs = settings.get("costs", {})
    scores = _finite_scores(getattr(decision, "model_context", {}).get("scores", {}))
    candidate = {
        str(symbol): float(weight) * float(decision.exposure_target)
        for symbol, weight in decision.target_weights.items()
        if float(weight) > 0
    }
    current = _current_weights(decision)
    reasons: list[str] = []
    if not current and getattr(decision, "current_positions", None) is None:
        reasons.append("current_portfolio_unavailable")
    if not candidate or not scores:
        reasons.append("candidate_scores_unavailable")
    current_score = sum(weight * scores.get(symbol, 0.0) for symbol, weight in current.items())
    candidate_score = sum(weight * scores.get(symbol, 0.0) for symbol, weight in candidate.items())
    gross = candidate_score - current_score
    symbols = set(current) | set(candidate)
    turnover = sum(abs(candidate.get(s, 0.0) - current.get(s, 0.0)) for s in symbols)
    transaction_cost = turnover * float(costs.get("transaction_cost_bps", 0)) / 10_000
    slippage = turnover * float(costs.get("slippage_bps", 0)) / 10_000
    net = gross - transaction_cost - slippage
    replacements = len(set(candidate) - set(current))
    confidence = _confidence(scores, candidate)
    risk_ok = turnover <= float(gates.get("maximum_turnover", 1.0)) and replacements <= int(gates.get("maximum_replacements", len(candidate)))
    confidence_ok = confidence >= float(gates.get("minimum_confidence", 0.0))
    persistence_ok = int(gates.get("persistence_observations", 1)) <= 1
    cooldown_ok = int(gates.get("cooldown_trading_days", 0)) <= 0
    if net < float(gates.get("minimum_net_improvement", 0.0)):
        reasons.append("minimum_net_improvement_not_met")
    if not risk_ok:
        reasons.append("turnover_or_replacement_gate_failed")
    if not confidence_ok:
        reasons.append("confidence_gate_failed")
    if not persistence_ok:
        reasons.append("persistence_history_unavailable")
    if not cooldown_ok:
        reasons.append("cooldown_history_unavailable")
    recommended = not reasons
    allow_submit = bool(settings.get("allow_paper_submit", False))
    return ModelTriggeredRebalanceResult(
        as_of_timestamp=_timestamp(decision), policy="model_triggered",
        model_or_champion_source=str(settings.get("candidate", {}).get("source", "dual_momentum")),
        evaluate_every_run=True,
        fixed_schedule_gate_bypassed_for_candidate_evaluation=True,
        current_holdings=current, candidate_holdings=candidate,
        current_score=current_score, candidate_score=candidate_score,
        gross_improvement=gross, estimated_transaction_cost=transaction_cost,
        estimated_slippage=slippage, net_improvement=net, turnover=turnover,
        replacement_count=replacements, risk_gate_passed=risk_ok,
        confidence_gate_passed=confidence_ok, persistence_gate_passed=persistence_ok,
        cooldown_gate_passed=cooldown_ok,
        decision="PAPER_REBALANCE_RECOMMENDED" if recommended else "NO_TRADE",
        reason_codes=tuple(reasons), paper_submit_requested=submit_requested,
        paper_submit_allowed=recommended and allow_submit and submit_requested,
    )


def _finite_scores(values: Mapping[str, Any]) -> dict[str, float]:
    output = {}
    for symbol, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(number):
            output[str(symbol)] = number
    return output


def _current_weights(decision: Any) -> dict[str, float]:
    weights: dict[str, float] = {}
    for order in getattr(decision, "orders", []):
        weight = float(getattr(order, "current_weight", 0.0) or 0.0)
        if weight > 0:
            weights[str(order.symbol)] = weight
    return weights


def _confidence(scores: Mapping[str, float], candidate: Mapping[str, float]) -> float:
    selected = [scores[s] for s in candidate if s in scores]
    all_values = list(scores.values())
    if not selected or not all_values:
        return 0.0
    floor, ceiling = min(all_values), max(all_values)
    if ceiling == floor:
        return 0.0
    return (sum(selected) / len(selected) - floor) / (ceiling - floor)


def _timestamp(decision: Any) -> str:
    value = getattr(decision, "timestamp", datetime.utcnow())
    return value.isoformat() if hasattr(value, "isoformat") else str(value)
