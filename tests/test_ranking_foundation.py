from __future__ import annotations

import math

import pytest

from core.research.ml.ranking import (
    DECILE_RELEVANCE_ID, QUINTILE_RELEVANCE_ID, OrderedLogitRanker,
    build_ranking_groups, ranking_metrics, relevance_labels,
)
from core.research.ml.registries import RegistryResolver, load_registry_bundle


def _rows(dates=("2026-01-01","2026-01-02"), count=10):
    return [{"row_id":f"{date}-{index:02d}","decision_timestamp":date,"label_available_timestamp":"2026-01-03","actual_forward_return_10d":float(index),"x":float(index)} for date in dates for index in range(count)]


def test_groups_are_deterministic_complete_and_mature():
    rows=list(reversed(_rows()))
    groups=build_ranking_groups(rows,scoring_cutoff="2026-01-04")
    assert groups.group_keys == ("2026-01-01","2026-01-02")
    assert groups.group_sizes == (10,10)
    assert sum(groups.group_sizes)==len(groups.rows)==20
    assert groups.rows[0]["row_id"]=="2026-01-01-00"
    assert groups.identity_hash==build_ranking_groups(rows,scoring_cutoff="2026-01-04").identity_hash
    with pytest.raises(ValueError,match="not mature"):
        build_ranking_groups(rows,scoring_cutoff="2026-01-02")


def test_groups_reject_duplicate_and_oos_ownership():
    rows=_rows(("2026-01-01",),5)
    with pytest.raises(ValueError,match="unique"):
        build_ranking_groups([*rows,rows[0]],scoring_cutoff="2026-01-03")
    legal=[{**row,"label_available_timestamp":"2026-01-01"} for row in rows]
    with pytest.raises(ValueError,match="OOS"):
        build_ranking_groups(legal,scoring_cutoff="2026-01-01")


def test_quintile_decile_and_ties_are_deterministic():
    rows=_rows(("2026-01-01",),10)
    quintile=relevance_labels(rows,bins=5); decile=relevance_labels(rows,bins=10)
    assert quintile["contract_id"]==QUINTILE_RELEVANCE_ID
    assert sorted(quintile["labels_by_row_id"].values())==[0,0,1,1,2,2,3,3,4,4]
    assert decile["contract_id"]==DECILE_RELEVANCE_ID
    tied=[{**row,"actual_forward_return_10d":float(index//2)} for index,row in enumerate(rows)]
    assert relevance_labels(tied,bins=5)==relevance_labels(list(reversed(tied)),bins=5)
    assert relevance_labels(tied,bins=5)["target_tie_count"]==5
    with pytest.raises(ValueError,match="too small"): relevance_labels(rows[:4],bins=5)
    with pytest.raises(ValueError,match="Collapsed"): relevance_labels([{**r,"actual_forward_return_10d":1.0} for r in rows],bins=5)


def test_ordered_logit_probabilities_and_expected_score():
    rows=_rows(count=10); x=[[row["x"],row["x"]**2/100] for row in rows]; y=[row["actual_forward_return_10d"] for row in rows]
    model=OrderedLogitRanker(max_iter=100).fit(x,y,groups=[row["decision_timestamp"] for row in rows],row_ids=[row["row_id"] for row in rows])
    probabilities=model.predict_proba(x); scores=model.predict(x)
    assert probabilities.shape==(20,5)
    assert all(sum(row)==pytest.approx(1.0) for row in probabilities)
    assert all(math.isfinite(value) for value in scores)
    assert scores[-1]>scores[0]
    assert len(model.diagnostics["coefficient_values"])==2


def test_ranking_metrics_are_per_date_and_ndcg_orders_correctly():
    rows=[]
    for index in range(40): rows.append({"row_id":str(index),"decision_timestamp":"2026-01-01","score":float(index),"reverse":float(-index),"actual_forward_return_10d":float(index),"relevance":index//8})
    perfect=ranking_metrics(rows,score_field="score"); reversed_result=ranking_metrics(rows,score_field="reverse")
    metric=perfect["per_date"][0]
    assert metric["spearman_rank_ic"]==pytest.approx(1.0)
    assert metric["pearson_ic"]==pytest.approx(1.0)
    assert metric["ndcg_at_10"]>reversed_result["per_date"][0]["ndcg_at_10"]
    assert metric["top_minus_bottom_10"]==pytest.approx(30.0)
    assert perfect["ordinary_standard_errors_valid"] is False


def test_registry_contains_distinct_ranking_contracts_and_model():
    bundle=load_registry_bundle(); resolver=RegistryResolver(bundle)
    model=resolver.resolve("selector_models","ordered_logit_ranker",role="selector")
    assert model.entry.payload["feature_schema"].endswith("canonical_v2_daily_tree_cross_sectional_v1.json")
    assert resolver.resolve("ranking_contracts",QUINTILE_RELEVANCE_ID).entry.entry_hash != resolver.resolve("ranking_contracts",DECILE_RELEVANCE_ID).entry.entry_hash
