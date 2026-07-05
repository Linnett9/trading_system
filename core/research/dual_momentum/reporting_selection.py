from __future__ import annotations

from core.research.dual_momentum.scoring import (
    classify_dual_momentum_result,
    paper_safe_dual_momentum_score,
    production_gap_score,
    risk_regime_score,
)
from core.research.dual_momentum.reporting_types import (
    RISK_REGIME_REPORT_LIMIT,
)


def select_champion(results):
    if not results:
        return None

    production_candidates = _candidates_with_tag(
        results,
        {"production candidate"},
    )
    if production_candidates:
        return max(production_candidates, key=_champion_sort_key)

    near_production_candidates = _candidates_with_tag(
        results,
        {"near-production candidate"},
    )
    if near_production_candidates:
        return max(near_production_candidates, key=_champion_sort_key)

    paper_candidates = _candidates_with_tag(results, {"paper candidate"})
    if paper_candidates:
        return max(paper_candidates, key=_champion_sort_key)

    candidates = [
        item
        for item in results
        if not classify_dual_momentum_result(item["result"]).startswith("rejected")
    ]

    if not candidates:
        candidates = results

    return max(candidates, key=_champion_sort_key)


def select_raw_score_leader(results):
    if not results:
        return None

    return max(results, key=lambda item: risk_regime_score(item["result"]))


def champion_label(item):
    if item is None:
        return "Champion"

    tag = classify_dual_momentum_result(item["result"])
    if tag == "production candidate":
        return "Production champion"

    if tag == "near-production candidate":
        return "Near-production champion"

    if tag == "paper candidate":
        return "Best paper candidate"

    return "Best research candidate"


def _candidates_with_tag(results, tags):
    return [
        item
        for item in results
        if classify_dual_momentum_result(item["result"]) in tags
    ]


def _champion_sort_key(item):
    result = item["result"]
    return (
        -production_gap_score(result),
        paper_safe_dual_momentum_score(result),
        risk_regime_score(result),
        result.result.sharpe,
        result.calmar,
        -result.result.max_drawdown,
        -result.annualized_turnover_percent,
        -result.cost_drag_percent,
    )


def risk_regime_report_items(results, champion_item, raw_leader_item):
    must_show = {
        id(item)
        for item in (champion_item, raw_leader_item)
        if item is not None
    }

    ranked = sorted(
        results,
        key=lambda item: (
            classify_dual_momentum_result(item["result"])
            == "production candidate",
            classify_dual_momentum_result(item["result"])
            == "near-production candidate",
            -production_gap_score(item["result"]),
            paper_safe_dual_momentum_score(item["result"]),
            risk_regime_score(item["result"]),
            item["result"].result.sharpe,
            -item["result"].result.max_drawdown,
            -item["result"].annualized_turnover_percent,
        ),
        reverse=True,
    )

    selected = []
    selected_ids = set()
    for item in ranked:
        if len(selected) < RISK_REGIME_REPORT_LIMIT or id(item) in must_show:
            selected.append(item)
            selected_ids.add(id(item))

    for item in (champion_item, raw_leader_item):
        if item is not None and id(item) not in selected_ids:
            selected.append(item)

    return selected
