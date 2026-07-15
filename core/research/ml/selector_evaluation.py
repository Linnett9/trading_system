from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Mapping, Sequence

from core.research.ml.artifact_lineage import VERIFIED_STRICT_OOS, read_artifact_link
from core.research.ml.artifact_lineage import VerificationResult, build_artifact_link, promotion_eligibility
from core.research.ml.experiment_ledger import append_ledger_event, experiment_spec_hash, new_experiment_run_id
from core.research.ml.provenance import source_provenance
from core.research.ml.registries.io import canonical_hash


EVALUATION_CONTRACT = "selector_multi_regime_evaluation_v1"
COST_CONTRACT = "selector_traded_notional_costs_v1"
PORTFOLIO_METRICS_CONTRACT = "selector_portfolio_metrics_v1"
DATE_PANEL_CONTRACT = "selector_evaluation_date_panel_v1"
ORCHESTRATION_CONTRACT = "selector_date_subprocess_orchestration_v1"
COST_SCENARIOS_BPS = (5, 10, 25, 50)
CAPACITY_SCENARIOS_ADV = (0.01, 0.025, 0.05)


def build_date_panel(requested_dates: Sequence[str], available_dates: Sequence[str], *, panel_id: str, resolution: str = "forward_then_backward", target_horizon: int = 10) -> dict[str, Any]:
    available=sorted(set(available_dates)); resolved=[]; resolutions=[]
    if not available: raise ValueError("Date panel requires eligibility dates")
    for requested in requested_dates:
        if requested in available: chosen=requested; method="exact"
        else:
            forward=next((value for value in available if value>requested),None); backward=next((value for value in reversed(available) if value<requested),None)
            chosen=forward or backward if resolution=="forward_then_backward" else backward or forward
            if chosen is None: raise ValueError(f"Cannot resolve panel date: {requested}")
            method="forward" if forward==chosen else "backward"
        if chosen not in resolved: resolved.append(chosen)
        resolutions.append({"requested":requested,"resolved":chosen,"method":method})
    overlap_pairs=sum(1 for left,right in zip(resolved,resolved[1:]) if _day_distance(left,right)<target_horizon)
    payload={"contract_version":DATE_PANEL_CONTRACT,"panel_id":panel_id,"selection_inputs":"calendar_and_dataset_eligibility_only","resolution_rule":resolution,"dates":resolved,"resolutions":resolutions,"target_horizon_sessions":target_horizon,"adjacent_pair_count":max(len(resolved)-1,0),"overlapping_pair_count":overlap_pairs,"overlap_rate":overlap_pairs/max(len(resolved)-1,1),"inferential_independence":False}
    payload["panel_checksum"]=canonical_hash({key:value for key,value in payload.items() if key!="panel_checksum"}); return payload


def daily_top_k(rows: Sequence[Mapping[str, Any]], *, score_field: str, k: int) -> dict[str,float]:
    ranked=sorted(rows,key=lambda row:(-float(row[score_field]),str(row["row_id"])))[:k]
    return {str(row["symbol"]):1/len(ranked) for row in ranked} if ranked else {}


def percentile_weighted_top_k(rows: Sequence[Mapping[str, Any]], *, score_field: str, k: int) -> dict[str,float]:
    ranked=sorted(rows,key=lambda row:(-float(row[score_field]),str(row["row_id"])))[:k]
    raw=list(range(len(ranked),0,-1)); total=sum(raw)
    return {str(row["symbol"]):weight/total for row,weight in zip(ranked,raw)} if total else {}


def staggered_cohorts(decision_rows: Sequence[Mapping[str, Any]], *, score_field: str, k: int, horizon: int = 10) -> list[dict[str,Any]]:
    by_date={date:[row for row in decision_rows if str(row["decision_date"])==date] for date in sorted({str(row["decision_date"]) for row in decision_rows})}; active=[]; output=[]
    for session_index,(date,rows) in enumerate(by_date.items()):
        expired=[cohort for cohort in active if session_index-cohort["formation_index"]>=horizon]; active=[cohort for cohort in active if cohort not in expired]
        new={"cohort_id":f"cohort:{date}","formation_date":date,"formation_index":session_index,"weights":daily_top_k(rows,score_field=score_field,k=k),"selector_artifact_link_hash":_unique_link(rows)}; active.append(new)
        aggregate={}; cohort_rows=[]
        for cohort in active:
            age=session_index-cohort["formation_index"]
            for symbol,weight in cohort["weights"].items(): aggregate[symbol]=aggregate.get(symbol,0)+weight/horizon
            cohort_rows.append({"cohort_id":cohort["cohort_id"],"age":age,"weight_fraction":1/horizon,"selector_artifact_link_hash":cohort["selector_artifact_link_hash"]})
        output.append({"decision_date":date,"new_cohort":new["cohort_id"],"expired_cohorts":[row["cohort_id"] for row in expired],"active_cohorts":cohort_rows,"aggregate_holdings":aggregate,"active_cohort_count":len(active)})
    return output


def rank_hysteresis(rows_by_date: Mapping[str,Sequence[Mapping[str,Any]]], *, score_field: str, enter_rank: int = 20, exit_rank: int = 30) -> list[dict[str,Any]]:
    if enter_rank>exit_rank: raise ValueError("Hysteresis enter rank must not exceed exit rank")
    held:set[str]=set(); output=[]
    for date,rows in sorted(rows_by_date.items()):
        ranked=sorted(rows,key=lambda row:(-float(row[score_field]),str(row["row_id"]))); rank={str(row["symbol"]):index+1 for index,row in enumerate(ranked)}
        previous=set(held); retained={symbol for symbol in held if rank.get(symbol,exit_rank+1)<=exit_rank}; vacancies=max(enter_rank-len(retained),0)
        candidates=[str(row["symbol"]) for row in ranked if rank[str(row["symbol"])]<=enter_rank and str(row["symbol"]) not in retained]; held=retained|set(candidates[:vacancies])
        daily={str(row["symbol"]) for row in ranked[:enter_rank]}; entries=held-previous; exits=previous-held
        output.append({"decision_date":date,"held":sorted(held),"retained":sorted(retained),"entries":sorted(entries),"exits":sorted(exits),"avoided_trades":len(retained-daily),"top_k_overlap":len(held&daily)/enter_rank,"opportunity_cost_symbols":sorted(daily-held)})
    return output


def traded_notional_cost(previous: Mapping[str,float], current: Mapping[str,float], *, cost_bps: float, adv_by_symbol: Mapping[str,float] | None = None, portfolio_value: float = 1.0) -> dict[str,Any]:
    gross_turnover=sum(abs(current.get(symbol,0)-previous.get(symbol,0)) for symbol in set(previous)|set(current)); one_way=gross_turnover/2; traded_notional=gross_turnover*portfolio_value; cost=traded_notional*cost_bps/10000
    capacity={f"max_{value*100:g}_pct_adv":{"status":"VERIFIED" if adv_by_symbol else "UNVERIFIED","limit_fraction":value} for value in CAPACITY_SCENARIOS_ADV}
    return {"cost_contract_version":COST_CONTRACT,"cost_bps":cost_bps,"gross_turnover":gross_turnover,"one_way_turnover":one_way,"traded_notional":traded_notional,"cost":cost,"capacity":capacity}


def portfolio_metrics(periods: Sequence[Mapping[str,Any]], *, annualization: int = 252, risk_free_rate: float = 0.0) -> dict[str,Any]:
    returns=[float(row["net_return"]) for row in periods]; gross=[float(row.get("gross_return",row["net_return"])) for row in periods]; equity=1.0; peak=1.0; max_dd=0.0
    for value in returns: equity*=1+value; peak=max(peak,equity); max_dd=min(max_dd,equity/peak-1)
    avg=mean(returns) if returns else 0; vol=pstdev(returns) if len(returns)>1 else 0; downside=math.sqrt(mean([min(value-risk_free_rate/annualization,0)**2 for value in returns])) if returns else 0
    annual_return=equity**(annualization/len(returns))-1 if returns else None; weights=[abs(float(value)) for row in periods for value in row.get("holdings",{}).values()]
    return {"contract_version":PORTFOLIO_METRICS_CONTRACT,"annualization_periods":annualization,"risk_free_rate":risk_free_rate,"gross_return":sum(gross),"net_return":sum(returns),"annualized_return":annual_return,"annualized_volatility":vol*math.sqrt(annualization),"sharpe":((avg-risk_free_rate/annualization)/vol*math.sqrt(annualization) if vol else None),"sortino":((avg-risk_free_rate/annualization)/downside*math.sqrt(annualization) if downside else None),"maximum_drawdown":max_dd,"calmar":annual_return/abs(max_dd) if annual_return is not None and max_dd<0 else None,"hit_rate":mean([value>0 for value in returns]) if returns else None,"average_turnover":mean([float(row.get("turnover",0)) for row in periods]) if periods else None,"total_cost":sum(float(row.get("cost",0)) for row in periods),"average_holding_age":mean([float(row.get("average_holding_age",0)) for row in periods]) if periods else None,"top_k_continuity":mean([float(row.get("top_k_continuity",0)) for row in periods]) if periods else None,"position_concentration":sum(value*value for value in weights)/max(len(periods),1),"sector_concentration":None,"benchmark_relative_return":sum(float(row.get("net_return",0))-float(row.get("benchmark_return",0)) for row in periods)}


def matched_comparison(left: Sequence[Mapping[str,Any]], right: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    left_map={(str(row["decision_date"]),str(row["row_id"])):row for row in left}; right_map={(str(row["decision_date"]),str(row["row_id"])):row for row in right}; shared=sorted(set(left_map)&set(right_map)); dates=sorted({key[0] for key in shared})
    fields=("rank_ic","ndcg","top_k_return","turnover","cost","net_return","drawdown")
    differences={field:mean([float(left_map[key].get(field,0))-float(right_map[key].get(field,0)) for key in shared]) if shared else None for field in fields}
    return {"contract_version":"selector_matched_comparison_v1","shared_decision_dates":dates,"shared_asset_row_count":len(shared),"left_unmatched_count":len(left_map)-len(shared),"right_unmatched_count":len(right_map)-len(shared),"paired_mean_differences":differences,"matched_results_separate":True,"ordinary_independent_significance_reported":False,"population_mismatch":bool(set(left_map)^set(right_map))}


def regime_summaries(rows: Sequence[Mapping[str,Any]], *, value_field: str="net_return") -> dict[str,Any]:
    dimensions=("market_trend_state","market_volatility_bucket","market_drawdown_bucket","historical_period"); result={}
    for dimension in dimensions:
        buckets={}
        for row in rows: buckets.setdefault(str(row.get(dimension,"UNKNOWN")),[]).append(float(row[value_field]))
        result[dimension]={key:{"count":len(values),"mean":mean(values),"distribution":sorted(values)} for key,values in sorted(buckets.items())}
    return {"contract_version":"descriptive_selector_regimes_v1","fitted_regime_model":False,"descriptive_only":True,"dimensions":result}


def evaluation_identity(*, selector_links, panel, policy_hash, cost_bps, benchmark_identity) -> dict[str,Any]:
    identity={"evaluation_contract":EVALUATION_CONTRACT,"selector_artifact_link_hashes":sorted(link["artifact_link_hash"] for link in selector_links),"date_panel_checksum":panel["panel_checksum"],"policy_entry_hash":policy_hash,"cost_contract_hash":canonical_hash({"contract":COST_CONTRACT,"cost_bps":cost_bps}),"benchmark_identity":benchmark_identity,"target_horizon_sessions":10}
    return {**identity,"evaluation_identity_hash":canonical_hash(identity)}


def verify_evaluation_inputs(manifest_paths: Sequence[Path], *, promotion_mode: bool) -> list[dict[str,Any]]:
    links=[read_artifact_link(path) for path in manifest_paths]
    invalid=[link for link in links if link.get("verification_status")!=VERIFIED_STRICT_OOS]
    if promotion_mode and invalid: raise ValueError("Promotion evaluation requires VERIFIED_STRICT_OOS selector partitions")
    return links


def write_evaluation_partition(*, output_root: Path, model_id: str, decision_date: str, policy_id: str, policy_entry_hash: str, cost_bps: int, panel: Mapping[str,Any], selector_manifest_paths: Sequence[Path], benchmark_identity: str, payload: Mapping[str,Any], promotion_mode: bool, ledger_path: Path) -> dict[str,Any]:
    links=verify_evaluation_inputs(selector_manifest_paths,promotion_mode=promotion_mode); identity=evaluation_identity(selector_links=links,panel=panel,policy_hash=policy_entry_hash,cost_bps=cost_bps,benchmark_identity=benchmark_identity)
    owner=output_root/f"model={model_id}"/f"date={decision_date}"/f"policy={policy_id}"/f"cost_bps={cost_bps}"; manifest_path=owner/"manifest.json"; output_path=owner/"evaluation.json"
    existing=_read_json(manifest_path)
    if existing and existing.get("status")=="complete" and existing.get("identity")==identity and output_path.exists() and _sha256(output_path)==existing.get("output_checksum") and existing.get("manifest_checksum")==canonical_hash({key:value for key,value in existing.items() if key!="manifest_checksum"}):
        return {"status":"skipped_complete","manifest_path":str(manifest_path),"evaluation_identity_hash":identity["evaluation_identity_hash"]}
    source=source_provenance(); spec_hash=experiment_spec_hash(identity); run_id=new_experiment_run_id(spec_hash)
    append_ledger_event(ledger_path,experiment_spec_hash_value=spec_hash,experiment_run_id=run_id,event_status="STARTED",artifact_kind="SELECTOR_EVALUATION",canonical_model_id=model_id,requested_model_id=model_id,registry_hashes={"policy_entry_hash":policy_entry_hash,"cost_contract_hash":identity["cost_contract_hash"]},source_commit=source["git_commit"],metadata={"evaluation_identity_hash":identity["evaluation_identity_hash"]})
    temp=owner.with_name(f".{owner.name}.{uuid.uuid4().hex}.tmp"); temp.mkdir(parents=True,exist_ok=False); _write_json_atomic(temp/"evaluation.json",dict(payload)); checksum=_sha256(temp/"evaluation.json")
    verified=all(link.get("verification_status")==VERIFIED_STRICT_OOS for link in links); result=VerificationResult(VERIFIED_STRICT_OOS if verified else "INSUFFICIENT_EVIDENCE",() if verified else ("UPSTREAM_NOT_VERIFIED_STRICT_OOS",))
    link=build_artifact_link(artifact_kind="SELECTOR_EVALUATION",artifact_id=f"selector-evaluation:{identity['evaluation_identity_hash']}",artifact_manifest_path=manifest_path,artifact_path=output_path,artifact_checksum=checksum,experiment_spec_hash=spec_hash,experiment_run_id=run_id,source_commit=source["git_commit"],registry_identity_version=EVALUATION_CONTRACT,canonical_model_or_policy_id=model_id,model_or_policy_entry_hash=policy_entry_hash,decision_start=decision_date,decision_end=decision_date,strict_oos_claim=verified,strict_oos_evidence={"selector_inputs_verified":verified},upstream_links=links,verification_status=result.status,verification_reasons=list(result.reason_codes),completion_status="complete")
    manifest={"evaluation_contract_version":EVALUATION_CONTRACT,"status":"complete","identity":identity,"output_checksum":checksum,"artifact_link":link,"promotion":promotion_eligibility(link,result),"source_provenance":source}; manifest["manifest_checksum"]=canonical_hash(manifest); _write_json_atomic(temp/"manifest.json",manifest); owner.parent.mkdir(parents=True,exist_ok=True)
    if owner.exists():
        raise FileExistsError(f"Evaluation partition already exists with different identity or invalid checksums: {owner}")
    os.replace(temp,owner); append_ledger_event(ledger_path,experiment_spec_hash_value=spec_hash,experiment_run_id=run_id,event_status="COMPLETED",artifact_kind="SELECTOR_EVALUATION",canonical_model_id=model_id,requested_model_id=model_id,registry_hashes={"policy_entry_hash":policy_entry_hash,"cost_contract_hash":identity["cost_contract_hash"]},source_commit=source["git_commit"],artifact_paths=(str(manifest_path),str(output_path)),metadata={"evaluation_identity_hash":identity["evaluation_identity_hash"]}); return {"status":"complete","manifest_path":str(manifest_path),"evaluation_identity_hash":identity["evaluation_identity_hash"]}


def orchestrate_date_panel(*, dates: Sequence[str], models: Sequence[str], output_root: Path, command_builder: Callable[[str,str,Path],Sequence[str]], concurrency: int = 1, failure_threshold: int = 1, runner: Callable[...,Any] = subprocess.run) -> dict[str,Any]:
    if concurrency<1: raise ValueError("Concurrency must be at least one")
    if failure_threshold<1: raise ValueError("Failure threshold must be at least one")
    ownership=set(); jobs=[]
    for date in dates:
        for model in models:
            owner=output_root/f"date={date}"/f"model={model}"
            if owner in ownership: raise ValueError("Duplicate model/date ownership")
            ownership.add(owner); jobs.append((date,model,owner))
    results=[]; failures=0
    def execute(job):
        date,model,owner=job; owner.mkdir(parents=True,exist_ok=True); log=owner/"subprocess.log"; result_path=owner/"subprocess_result.json"; command=list(command_builder(date,model,owner)); command_hash=canonical_hash(command)
        existing=_read_json(result_path)
        if existing and existing.get("exit_code")==0 and existing.get("command_hash")==command_hash:
            return {**existing,"status":"skipped_complete"}
        completed=runner(command,capture_output=True,text=True,check=False)
        log.write_text((completed.stdout or "")+(completed.stderr or ""),encoding="utf-8")
        row={"date":date,"model":model,"owner":str(owner),"command":command,"command_hash":command_hash,"exit_code":completed.returncode,"log_path":str(log),"status":"complete" if completed.returncode==0 else "failed"}; _write_json_atomic(result_path,row); return row
    cursor=0
    while cursor<len(jobs) and failures<failure_threshold:
        batch=jobs[cursor:cursor+concurrency]
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            pending=[executor.submit(execute,job) for job in batch]
            for future in as_completed(pending):
                row=future.result(); results.append(row); failures+=int(row["exit_code"]!=0)
        cursor+=len(batch)
    manifest={"contract_version":ORCHESTRATION_CONTRACT,"concurrency":concurrency,"failure_threshold":failure_threshold,"results":sorted(results,key=lambda row:(row["date"],row["model"])),"failure_count":failures,"status":"complete" if failures==0 else "failed_threshold"}; _write_json_atomic(output_root/"orchestration_manifest.json",manifest); return manifest


def _unique_link(rows):
    values={str(row.get("selector_artifact_link_hash") or "") for row in rows}
    if len(values)!=1 or "" in values: raise ValueError("Cohort formation requires exact selector ancestry")
    return next(iter(values))


def _day_distance(left,right):
    from datetime import date
    return (date.fromisoformat(right)-date.fromisoformat(left)).days


def _write_json_atomic(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp"); temp.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8"); os.replace(temp,path)


def _sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest().upper()


def _read_json(path):
    try:return json.loads(path.read_text(encoding="utf-8"))
    except (OSError,ValueError,TypeError):return None
