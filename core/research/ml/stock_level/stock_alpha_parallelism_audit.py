from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.runtime_parallelism import apply_stock_alpha_worker_caps
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_report_metadata
from core.research.ml.stock_level_benchmark_types import MODEL_NAMES


@dataclass(frozen=True)
class StockAlphaParallelismAuditPaths:
    json_path: Path
    markdown_path: Path


def write_stock_alpha_parallelism_audit(config: dict[str, Any]) -> StockAlphaParallelismAuditPaths:
    settings = StockLevelResearchConfig.from_mapping(config); ml = config.get("ml", {}); cpu = os.cpu_count() or 1
    caps = apply_stock_alpha_worker_caps(config)
    requested = {"feature_generation": settings.alpha_feature_n_jobs, "model_ranking": settings.model_n_jobs, "target_comparison": settings.target_comparison_n_jobs, "portfolio_policy_sweep": int(ml.get("stock_portfolio_policy_sweep_n_jobs", 4)), "overnight_stages": settings.overnight_stage_n_jobs}
    stages = {
        "feature_generation": {"parallel_capable": True, "requested_workers": requested["feature_generation"], "effective_workers": min(cpu, requested["feature_generation"]), "nested_worker_caps": caps, "configured_workers_ignored": False},
        "model_ranking": {"parallel_capable": True, "requested_workers": requested["model_ranking"], "effective_workers": min(cpu, requested["model_ranking"], len(MODEL_NAMES)), "nested_worker_caps": {"sklearn_n_jobs": 1 if requested["model_ranking"] > 1 else caps["sklearn_n_jobs"], "torch_num_threads": 1 if requested["model_ranking"] > 1 else caps["torch_num_threads"], "numpy_num_threads": 1}, "configured_workers_ignored": False},
        "target_comparison": {"parallel_capable": True, "requested_workers": requested["target_comparison"], "effective_workers": min(cpu, requested["target_comparison"], len(settings.target_columns)), "nested_worker_caps": {"stock_ranker_model_n_jobs": 1 if requested["target_comparison"] > 1 else settings.model_n_jobs, **caps}, "configured_workers_ignored": False},
        "portfolio_policy_sweep": {"parallel_capable": True, "requested_workers": requested["portfolio_policy_sweep"], "effective_workers": min(cpu, requested["portfolio_policy_sweep"]), "nested_worker_caps": {**caps, "policy_nested_workers": 1}, "configured_workers_ignored": False},
        "overnight_stages": {"parallel_capable": False, "requested_workers": requested["overnight_stages"], "effective_workers": 1, "nested_worker_caps": caps, "configured_workers_ignored": requested["overnight_stages"] != 1},
    }
    warnings = []
    if sum(row["effective_workers"] for name, row in stages.items() if name != "overnight_stages") > cpu * 2: warnings.append("Configured stage workers exceed CPU count if stages are run concurrently; overnight stages remain sequential.")
    if any(value > 1 for value in caps.values()): warnings.append("Nested worker caps exceed one and may oversubscribe outer workers.")
    payload = {"mode": "stock_alpha_parallelism_audit_research_only", "cpu_count": cpu, "stages": stages, "oversubscription_warnings": warnings, "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}, **stock_alpha_report_metadata(config, settings.output_dir), "research_only": True, "trading_impact": "none", "production_validated": False, "promotion_thresholds_changed": False}
    paths = StockAlphaParallelismAuditPaths(settings.output_dir / "stock_alpha_parallelism_audit.json", settings.output_dir / "stock_alpha_parallelism_audit.md")
    writer = ResearchArtifactWriter(); writer.write_json(paths.json_path, payload); writer.write_markdown(paths.markdown_path, _markdown(payload)); return paths


def _markdown(payload: dict[str, Any]) -> str:
    lines = ["# Stock Alpha Parallelism Audit", "", "Research only. Trading impact: none. Production validated: false.", "", f"- CPU count: {payload['cpu_count']}", f"- Run size: `{payload['run_size']}`", "", "| Stage | Parallel capable | Requested | Effective | Ignored |", "|---|---|---:|---:|---|"]
    for name, row in payload["stages"].items(): lines.append(f"| {name} | {row['parallel_capable']} | {row['requested_workers']} | {row['effective_workers']} | {row['configured_workers_ignored']} |")
    lines.extend(["", "## Oversubscription warnings"]); lines.extend(f"- {warning}" for warning in payload["oversubscription_warnings"] or ["None"]); lines.extend(["", "Promotion thresholds changed: false.", ""]); return "\n".join(lines)
