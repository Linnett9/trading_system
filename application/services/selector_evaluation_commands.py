from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.research.ml.registries import load_registry_bundle
from core.research.ml.selector_panel_preflight import (
    CHALLENGERS, PRIMARY_MODEL, discover_selector_components, freeze_panel,
    resolve_authoritative_panel, run_preflight,
)
from core.research.ml.selector_component_readiness import assess_selector_component_readiness
from core.research.ml.selector_dataset_lineage import assess_lineage_repair
from core.research.ml.selector_publication_gates import evaluate_selector_parent_publication_gate
from core.research.ml.stock_level.ordinary_selector_publication import publish_planned_ordinary_component


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


def run_selector_artifact_audit(config: Mapping[str, Any], args: Any) -> dict[str, Any]:
    raise ValueError("ml-selector-artifact-audit was superseded by ml-selector-component-readiness")


def run_selector_component_preflight(config: Mapping[str, Any], args: Any) -> dict[str, Any]:
    required = {"--parent-gate": args.parent_gate, "--selector-dataset-root": args.selector_dataset_root, "--component-output-root": args.component_output_root}
    missing = [name for name, value in required.items() if not value]
    if missing: raise ValueError(f"Required arguments missing: {','.join(missing)}")
    approved = tuple(Path(value) for value in (args.approved_component_root or [args.component_output_root]))
    result = assess_selector_component_readiness(
        parent_gate_path=Path(args.parent_gate),
        authoritative_root=Path(args.component_output_root),
        selector_dataset_root=Path(args.selector_dataset_root),
        config_path=Path(getattr(args, "config", None) or "config/config.ticket_7b3_daily_large_history_regeneration_canonical_v2.yaml"),
        approved_component_roots=approved,
    )
    if args.verification_output: _write_report_pair(Path(args.verification_output), result, "Selector component production preflight")
    print(json.dumps({"status": result["overall_status"], "expected": result["expected_component_count"], "ready": result["ready_component_count"], "missing": result["missing_component_count"], "blockers": result["blockers"]}, sort_keys=True))
    if result["overall_status"] == "BLOCKED": raise SystemExit(2)
    return result


def run_selector_dataset_lineage_audit(config: Mapping[str, Any], args: Any) -> dict[str, Any]:
    required = {"--selector-dataset-manifest": args.selector_dataset_manifest, "--daily-spine-manifest": args.daily_spine_manifest, "--symbol-registry-manifest": args.symbol_registry_manifest}
    missing = [name for name, value in required.items() if not value]
    if missing: raise ValueError(f"Required arguments missing: {','.join(missing)}")
    result = assess_lineage_repair(dataset_root=Path(args.selector_dataset_manifest).parent, daily_spine_manifest=Path(args.daily_spine_manifest), symbol_registry_manifest=Path(args.symbol_registry_manifest))
    if args.verification_output: _write_report_pair(Path(args.verification_output), result, "Frozen selector dataset lineage audit")
    print(json.dumps({"classification": result["classification"], "status": result["status"], "blocking_reasons": result["blocking_reasons"]}, sort_keys=True))
    if result["status"] != "READY": raise SystemExit(2)
    return result


def run_selector_publication_validate(config: Mapping[str, Any], args: Any) -> dict[str, Any]:
    required = (args.symbol_registry_manifest, args.daily_spine_manifest, args.daily_feature_manifest, args.selector_dataset_manifest, args.operational_dates_manifest, args.approved_root)
    if not all(required): raise ValueError("Parent-gate manifest and approved-root arguments are required")
    result = evaluate_selector_parent_publication_gate(
        registry_manifest=Path(args.symbol_registry_manifest),
        spine_manifest=Path(args.daily_spine_manifest),
        feature_manifest=Path(args.daily_feature_manifest),
        dataset_manifest=Path(args.selector_dataset_manifest),
        operational_dates_manifest=Path(args.operational_dates_manifest),
        required_operational_dates=args.required_operational_date or [],
        approved_root=Path(args.approved_root),
    )
    if args.verification_output: _write_report_pair(Path(args.verification_output), result, "Selector parent publication validation")
    print(json.dumps({"status": result["status"], "blockers": result["blockers"], "logical_checksum": result["logical_checksum"]}, sort_keys=True))
    if result["status"] != "READY": raise SystemExit(2)
    return result


def run_selector_component_publish(config: Mapping[str, Any], args: Any) -> dict[str, Any]:
    required = (args.production_plan_job, args.parent_gate, args.training_rows_json, args.prediction_rows_json, args.experiment_ledger)
    if not all(required): raise ValueError("One plan job, parent gate, synthetic row inputs, and experiment ledger are required")
    job = json.loads(Path(args.production_plan_job).read_text(encoding="utf-8"))
    training = json.loads(Path(args.training_rows_json).read_text(encoding="utf-8"))
    prediction = json.loads(Path(args.prediction_rows_json).read_text(encoding="utf-8"))
    result = publish_planned_ordinary_component(
        job=job, parent_gate_path=Path(args.parent_gate),
        training_rows=training, prediction_rows=prediction,
        ledger_path=Path(args.experiment_ledger),
    )
    if args.verification_output: _write_report_pair(Path(args.verification_output), result, "Ordinary selector component publication")
    print(json.dumps(result, sort_keys=True))
    return result


def _counts(rows, field):
    result = {}
    for row in rows: result[row.get(field)] = result.get(row.get(field), 0) + 1
    return result


def _write_report_pair(path: Path, result: Mapping[str, Any], title: str) -> None:
    json_path = path if path.suffix == ".json" else path.with_suffix(".json"); md_path = json_path.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True); json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    lines = [f"# {title}", "", f"- Status: `{result.get('status', 'AUDIT_COMPLETE')}`", f"- Candidates: `{result.get('candidate_count', 'n/a')}`", f"- Components: `{result.get('component_count', 'n/a')}`", "- Fitting performed: `false`", "- Prediction performed: `false`"]
    lines.extend(f"- Blocker: `{reason}`" for reason in result.get("blocking_reasons", [])); md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
