from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.research.ml.registries.io import canonical_hash


RANKING_PROBLEM_ID = "daily_cross_sectional_ranking_problem_v1"
QUINTILE_RELEVANCE_ID = "within_date_quintile_relevance_v1"
DECILE_RELEVANCE_ID = "within_date_decile_relevance_v1"
METRICS_CONTRACT = "daily_cross_sectional_ranking_metrics_v1"


class OrderedRelevanceScore(float):
    def __new__(cls, value: float, probabilities: Sequence[float]):
        instance = float.__new__(cls, value); instance.probabilities = tuple(float(item) for item in probabilities); return instance


@dataclass(frozen=True)
class RankingGroups:
    rows: tuple[Mapping[str, Any], ...]
    group_keys: tuple[str, ...]
    group_sizes: tuple[int, ...]
    row_to_group: tuple[int, ...]
    group_population_checksums: Mapping[str, str]
    identity_hash: str


def build_ranking_groups(rows: Sequence[Mapping[str, Any]], *, scoring_cutoff: str | None = None, minimum_group_size: int = 5) -> RankingGroups:
    owned: set[str] = set(); grouped: dict[str, list[Mapping[str, Any]]] = {}; dropped = []
    for row in rows:
        row_id = str(row.get("row_id") or ""); decision = str(row.get("decision_timestamp") or row.get("rebalance_date") or "")
        target = row.get("actual_forward_return_10d")
        available = str(row.get("label_available_timestamp") or "")
        if not row_id or row_id in owned: raise ValueError("Ranking rows require unique stable row_id")
        if not decision or target in (None, ""): raise ValueError("Ranking rows require decision timestamp and continuous target")
        if scoring_cutoff and available > scoring_cutoff: raise ValueError("Ranking training label is not mature at scoring cutoff")
        if scoring_cutoff and decision >= scoring_cutoff: raise ValueError("OOS scoring group cannot enter ranking training")
        owned.add(row_id); grouped.setdefault(decision, []).append(row)
    ordered_rows=[]; keys=[]; sizes=[]; mapping=[]; checksums={}
    for decision in sorted(grouped):
        group = sorted(grouped[decision], key=lambda row: str(row["row_id"]))
        if len(group) < minimum_group_size:
            dropped.append({"decision_timestamp": decision, "row_count": len(group), "reason": "GROUP_TOO_SMALL"}); continue
        index=len(keys); keys.append(decision); sizes.append(len(group)); ordered_rows.extend(group); mapping.extend([index]*len(group))
        checksums[decision]=canonical_hash([str(row["row_id"]) for row in group])
    if sum(sizes) != len(ordered_rows): raise RuntimeError("Ranking group-size sum mismatch")
    identity={"contract":RANKING_PROBLEM_ID,"group_keys":keys,"group_sizes":sizes,"population_checksums":checksums,"dropped_groups":dropped}
    return RankingGroups(tuple(ordered_rows),tuple(keys),tuple(sizes),tuple(mapping),checksums,canonical_hash(identity))


def relevance_labels(rows: Sequence[Mapping[str, Any]], *, bins: int = 5, target_field: str = "actual_forward_return_10d", minimum_group_size: int | None = None) -> dict[str, Any]:
    if bins not in {5,10}: raise ValueError("Only quintile or decile relevance contracts are supported")
    minimum = minimum_group_size or bins
    groups: dict[str,list[Mapping[str,Any]]]={}
    for row in rows:
        decision=str(row.get("decision_timestamp") or row.get("rebalance_date") or "")
        value=row.get(target_field)
        if value in (None, ""): raise ValueError("Ranking relevance target is missing")
        groups.setdefault(decision,[]).append(row)
    labels={}; tie_count=0; distributions={}
    for decision, group in sorted(groups.items()):
        if len(group)<minimum: raise ValueError(f"Ranking group too small for {bins} bins: {decision}")
        ordered=sorted(group,key=lambda row:(float(row[target_field]),str(row["row_id"])))
        unique=len({float(row[target_field]) for row in ordered}); tie_count += len(ordered)-unique
        if unique < 2: raise ValueError(f"Collapsed relevance target boundaries: {decision}")
        distribution={str(value):0 for value in range(bins)}
        for index,row in enumerate(ordered):
            label=min(bins-1,(index*bins)//len(ordered)); labels[str(row["row_id"])]=label; distribution[str(label)]+=1
        distributions[decision]=distribution
    contract=QUINTILE_RELEVANCE_ID if bins==5 else DECILE_RELEVANCE_ID
    return {"contract_id":contract,"labels_by_row_id":labels,"class_distributions":distributions,"target_tie_count":tie_count,"tie_policy":"economic ties reported; stable row_id breaks only deterministic bin boundaries","identity_hash":canonical_hash({"contract_id":contract,"labels":labels})}


class OrderedLogitRanker:
    def __init__(self, *, bins: int = 5, max_iter: int = 200, tolerance: float = 1e-7):
        self.bins=bins; self.max_iter=max_iter; self.tolerance=tolerance; self.diagnostics: dict[str,Any]={}

    def fit(self, x, y, *, groups: Sequence[str] | None = None, row_ids: Sequence[str] | None = None):
        try:
            import numpy as np
            from scipy.optimize import minimize
        except ImportError as exc: raise ImportError("ordered_logit_ranker requires scipy; install the project's scientific dependencies") from exc
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        if groups is None or row_ids is None: raise ValueError("ordered_logit_ranker requires date groups and stable row IDs")
        rows=[{"row_id":rid,"decision_timestamp":group,"actual_forward_return_10d":target} for rid,group,target in zip(row_ids,groups,y)]
        relevance=relevance_labels(rows,bins=self.bins); labels=np.asarray([relevance["labels_by_row_id"][str(rid)] for rid in row_ids],dtype=int)
        self.imputer_=SimpleImputer(strategy="median"); self.scaler_=StandardScaler(); matrix=self.scaler_.fit_transform(self.imputer_.fit_transform(x)); n_features=matrix.shape[1]
        def unpack(params):
            beta=params[:n_features]; raw=params[n_features:]; thresholds=np.cumsum(np.exp(raw)); thresholds-=np.mean(thresholds); return beta,thresholds
        def objective(params):
            beta,theta=unpack(params); eta=matrix@beta; cumulative=1/(1+np.exp(-np.clip(theta[:,None]-eta[None,:],-40,40)))
            probabilities=np.vstack((cumulative[0], *(cumulative[i]-cumulative[i-1] for i in range(1,self.bins-1)), 1-cumulative[-1])).T
            return float(-np.log(np.clip(probabilities[np.arange(len(labels)),labels],1e-12,1)).sum())
        started=time.perf_counter(); initial=np.r_[np.zeros(n_features),np.zeros(self.bins-1)]; result=minimize(objective,initial,method="BFGS",options={"maxiter":self.max_iter,"gtol":self.tolerance})
        if not result.success and not math.isfinite(float(result.fun)): raise RuntimeError(f"Ordered-logit optimization failed: {result.message}")
        if not result.success and int(result.nit) >= self.max_iter: raise RuntimeError(f"Ordered-logit optimization did not converge within {self.max_iter} iterations: {result.message}")
        self.beta_,self.thresholds_=unpack(result.x); self.classes_=np.arange(self.bins)
        group_sizes={group:sum(1 for value in groups if value==group) for group in sorted(set(groups))}
        try: standard_errors=np.sqrt(np.diag(np.asarray(result.hess_inv)))[:n_features].tolist()
        except (TypeError,ValueError): standard_errors=None
        self.diagnostics={"training_row_count":len(labels),"group_count":len(group_sizes),"group_size_distribution":group_sizes,"converged":bool(result.success),"optimizer_status":str(result.message),"iteration_count":int(result.nit),"coefficient_values":self.beta_.tolist(),"coefficient_standard_errors":standard_errors,"threshold_values":self.thresholds_.tolist(),"fit_seconds":time.perf_counter()-started,"relevance":relevance,"imputation_statistics":self.imputer_.statistics_.tolist(),"scaling_mean":self.scaler_.mean_.tolist(),"scaling_scale":self.scaler_.scale_.tolist()}
        return self

    def predict_proba(self,x):
        import numpy as np
        matrix=self.scaler_.transform(self.imputer_.transform(x)); eta=matrix@self.beta_; cumulative=1/(1+np.exp(-np.clip(self.thresholds_[:,None]-eta[None,:],-40,40)))
        return np.vstack((cumulative[0],*(cumulative[i]-cumulative[i-1] for i in range(1,self.bins-1)),1-cumulative[-1])).T

    def predict(self,x):
        probabilities=self.predict_proba(x); values=probabilities@self.classes_
        return [OrderedRelevanceScore(value, row) for value,row in zip(values,probabilities)]

    def get_params(self, deep=True): return {"bins":self.bins,"max_iter":self.max_iter,"tolerance":self.tolerance}


def ranking_metrics(rows: Sequence[Mapping[str, Any]], *, score_field: str, target_field: str="actual_forward_return_10d", relevance_field: str="relevance", baseline_score_fields: Sequence[str] = ()) -> dict[str,Any]:
    import numpy as np
    groups: dict[str,list[Mapping[str,Any]]]={}
    for row in rows:
        if row.get(score_field) is None or row.get(target_field) is None: continue
        groups.setdefault(str(row.get("decision_timestamp") or row.get("rebalance_date")),[]).append(row)
    per_date=[]
    for decision,group in sorted(groups.items()):
        scores=np.asarray([float(row[score_field]) for row in group]); targets=np.asarray([float(row[target_field]) for row in group]);
        pearson=_correlation(scores,targets); spearman=_correlation(_average_ranks(scores),_average_ranks(targets)); ordered=sorted(group,key=lambda row:(-float(row[score_field]),str(row.get("row_id"))))
        metrics={"decision_timestamp":decision,"row_count":len(group),"pearson_ic":pearson,"spearman_rank_ic":spearman}
        for k in (10,20,40):
            if len(group)<k: metrics.update({f"ndcg_at_{k}":None,f"top_{k}_mean_return":None,f"bottom_{k}_mean_return":None,f"top_minus_bottom_{k}":None}); continue
            top=ordered[:k]; bottom=ordered[-k:]; top_mean=sum(float(r[target_field]) for r in top)/k; bottom_mean=sum(float(r[target_field]) for r in bottom)/k
            metrics.update({f"ndcg_at_{k}":_ndcg(ordered,k,relevance_field),f"top_{k}_mean_return":top_mean,f"bottom_{k}_mean_return":bottom_mean,f"top_minus_bottom_{k}":top_mean-bottom_mean})
            selected={str(row.get("row_id")) for row in top}; metrics[f"top_{k}_baseline_overlap"]={}
            for baseline in baseline_score_fields:
                baseline_top=sorted(group,key=lambda row:(-float(row[baseline]),str(row.get("row_id"))))[:k]
                metrics[f"top_{k}_baseline_overlap"][baseline]=len(selected & {str(row.get("row_id")) for row in baseline_top})/k
        counts={value:int(np.sum(scores==value)) for value in np.unique(scores)}; largest=max(counts.values(),default=0)
        metrics["ranking_structure"]={"distinct_rank_count":len(counts),"largest_tied_group":largest,"largest_tied_group_fraction":largest/len(group) if group else None,"rank_coverage":float(np.isfinite(scores).mean()),"score_dispersion":float(np.std(scores))}
        per_date.append(metrics)
    return {"contract_version":METRICS_CONTRACT,"per_date":per_date,"target_horizon_sessions":10,"overlapping_targets":True,"adjacent_dates_aggregated":len(per_date)>1,"ordinary_standard_errors_valid":False,"warning":"Adjacent daily ten-session targets are serially dependent; averages are not independent evidence."}


def _average_ranks(values):
    import numpy as np
    order=np.argsort(values,kind="mergesort"); ranks=np.empty(len(values),float); start=0
    while start<len(values):
        end=start+1
        while end<len(values) and values[order[end]]==values[order[start]]: end+=1
        ranks[order[start:end]]=(start+end-1)/2; start=end
    return ranks


def _correlation(left,right):
    import numpy as np
    if len(left)<2 or np.std(left)==0 or np.std(right)==0:return None
    return float(np.corrcoef(left,right)[0,1])


def _ndcg(ordered,k,relevance_field):
    gains=[2**float(row[relevance_field])-1 for row in ordered[:k]]; ideal=sorted((2**float(row[relevance_field])-1 for row in ordered),reverse=True)[:k]
    dcg=sum(g/math.log2(i+2) for i,g in enumerate(gains)); idcg=sum(g/math.log2(i+2) for i,g in enumerate(ideal)); return dcg/idcg if idcg else None
