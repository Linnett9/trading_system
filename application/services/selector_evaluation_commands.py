from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.research.ml.registries import load_registry_bundle
from core.research.ml.selector_panel_preflight import (
    CHALLENGERS, PRIMARY_MODEL, discover_selector_components, freeze_panel,
    resolve_authoritative_panel, run_preflight,
)


def run_selector_evaluation_preflight(config: Mapping[str, Any], args: Any) -> dict[str, Any]:
    if not args.frozen_panel:
        raise ValueError("--frozen-panel is required")
    if not args.evaluation_output_root:
        raise ValueError("--evaluation-output-root is required")
    report = run_preflight(
        frozen_panel=Path(args.frozen_panel), output_root=Path(args.evaluation_output_root),
        mutate_reports=bool(args.verification_output),
        report_root=Path(args.verification_output) if args.verification_output else None,
    )
    print(json.dumps({key: report[key] for key in ("status", "exit_code", "panel_checksum", "would_run", "would_resume", "would_skip", "blocking_reasons")}, sort_keys=True))
    if report["exit_code"]:
        raise SystemExit(report["exit_code"])
    return report


def run_selector_panel_resolve(config: Mapping[str, Any], args: Any) -> dict[str, Any]:
    if not args.selector_manifest_root: raise ValueError("--selector-manifest-root is required")
    if not args.panel_config: raise ValueError("--panel-config is required")
    if not args.panel_output_root: raise ValueError("--panel-output-root is required")
    panel_config = json.loads(Path(args.panel_config).read_text(encoding="utf-8"))
    bundle = load_registry_bundle()
    components = discover_selector_components([Path(value) for value in args.selector_manifest_root], models=(PRIMARY_MODEL, *CHALLENGERS))
    outcomes = []
    for key in ("operational_panel", "multi_regime_panel"):
        source = panel_config[key]
        panel = resolve_authoritative_panel(
            panel_name=source["panel_id"], panel_version="v1", requested_dates=source["requested_dates"],
            components=components, primary_model=PRIMARY_MODEL, challengers=CHALLENGERS,
            registry_set_hash=bundle.registry_set_hash,
            policy_registry_hash=bundle.documents["portfolio_policies"].registry_hash,
        )
        output = Path(args.panel_output_root) / f"{source['panel_id']}.v1.json"
        publication = freeze_panel(output, panel)
        outcomes.append({"panel_name": source["panel_id"], "status": panel["status"], "panel_checksum": panel["panel_checksum"], "resolved_dates": panel["resolved_dates"], "exclusions": panel["exclusions"], "output": str(output), "publication": publication})
    result = {"panel_resolution_contract": "authoritative_selector_panel_resolution_v1", "component_count": len(components), "panels": outcomes, "selector_fitting_performed": False, "historical_evaluation_performed": False}
    print(json.dumps(result, sort_keys=True))
    if any(row["status"] != "READY" for row in outcomes): raise SystemExit(2)
    return result
