from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from core.research.ml.registries import load_registry_bundle
from core.research.ml.registries.adapters import selector_model_adapter
from core.research.ml.registries.io import canonical_hash
from core.research.ml.selector_research_campaign import (
    BASELINE_CAMPAIGN_ID,
    validate_selector_campaign,
)
from core.research.ml.selector_wave4_input_packages import (
    PACKAGE_CONTRACT as V2_PACKAGE_CONTRACT,
    validate_v2_package,
)
from core.research.ml.selector_panel_preflight import (
    CHALLENGERS, PRIMARY_MODEL, discover_selector_components, freeze_panel,
    resolve_authoritative_panel, run_preflight,
)
from core.research.ml.selector_component_readiness import assess_selector_component_readiness
from core.research.ml.selector_dataset_lineage import assess_lineage_repair
from core.research.ml.selector_publication_gates import evaluate_selector_parent_publication_gate
from core.research.ml.stock_level.ordinary_selector_publication import publish_planned_ordinary_component
from core.research.ml.stock_level.wave4_selector_integration import publish_wave4_component


ORDINARY_DECLARED_RUNNER = (
    "core.research.ml.stock_level_benchmark_models:"
    "TabularModelFactory/SequenceModelFactory"
)
WAVE4_DECLARED_RUNNER = (
    "core.research.ml.stock_level.wave4_selector_integration:"
    "publish_wave4_component"
)
ORDINARY_RUNTIME_OWNER = (
    "core.research.ml.stock_level.ordinary_selector_publication:"
    "publish_planned_ordinary_component"
)
WAVE4_RUNTIME_OWNER = WAVE4_DECLARED_RUNNER


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
    required = (
        args.production_plan_job, args.campaign_manifest,
        args.campaign_identity, args.plan_job_identity,
        args.operational_input_package, args.parent_gate,
        args.training_rows_json, args.prediction_rows_json,
        args.experiment_ledger,
    )
    if not all(required):
        raise ValueError(
            "Campaign, plan job, operational package, row inputs, parent gate, "
            "and experiment ledger are required"
        )
    job = json.loads(Path(args.production_plan_job).read_text(encoding="utf-8"))
    campaign = json.loads(Path(args.campaign_manifest).read_text(encoding="utf-8"))
    package = json.loads(
        Path(args.operational_input_package).read_text(encoding="utf-8")
    )
    training = json.loads(Path(args.training_rows_json).read_text(encoding="utf-8"))
    prediction = json.loads(Path(args.prediction_rows_json).read_text(encoding="utf-8"))
    parent_gate = json.loads(Path(args.parent_gate).read_text(encoding="utf-8"))
    result = dispatch_selector_component_publication(
        campaign=campaign, job=job, package=package,
        training_rows=training, prediction_rows=prediction,
        parent_gate=parent_gate, parent_gate_path=Path(args.parent_gate),
        ledger_path=Path(args.experiment_ledger),
        supplied_campaign_identity=str(args.campaign_identity),
        supplied_plan_job_identity=str(args.plan_job_identity),
        supplied_component_runner=str(args.component_runner or ""),
    )
    if args.verification_output: _write_report_pair(Path(args.verification_output), result, "Selector component publication dispatch")
    print(json.dumps(result, sort_keys=True))
    return result


def dispatch_selector_component_publication(
    *,
    campaign: Mapping[str, Any],
    job: Mapping[str, Any],
    package: Mapping[str, Any],
    training_rows: list[Mapping[str, Any]],
    prediction_rows: list[Mapping[str, Any]],
    parent_gate: Mapping[str, Any],
    parent_gate_path: Path,
    ledger_path: Path,
    supplied_campaign_identity: str,
    supplied_plan_job_identity: str,
    supplied_component_runner: str,
    ordinary_publisher: Callable[..., dict[str, Any]] = publish_planned_ordinary_component,
    wave4_publisher: Callable[..., dict[str, Any]] = publish_wave4_component,
    wave4_adapter: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve one frozen campaign row to exactly one publication owner."""
    validate_selector_campaign(campaign)
    if campaign.get("campaign_identity") != supplied_campaign_identity:
        raise ValueError("Campaign identity mismatch")
    if not supplied_plan_job_identity:
        raise ValueError("Plan-job identity is required")
    matches = [
        row for row in campaign["fitted_component_matrix"]
        if row.get("job_id") == supplied_plan_job_identity
    ]
    if len(matches) != 1:
        raise ValueError("Plan-job identity is absent or ambiguous")
    planned = matches[0]
    for field in ("job_id", "model_id", "prediction_date", "horizon_id"):
        if (job.get(field) or None) != (planned.get(field) or None):
            raise ValueError(f"Runtime plan-job {field} mismatch")
    model_id = str(job["model_id"])
    registered_runner = selector_model_adapter(
        model_id, runner="ordinary"
    ).constructor_owner
    declared_runner = str(planned.get("component_runner") or "")
    if (
        campaign.get("campaign_id") == BASELINE_CAMPAIGN_ID
        and campaign.get("campaign_version") == "v1"
        and model_id in {"ridge", "elastic_net", "ordered_logit_ranker"}
        and not declared_runner
    ):
        declared_runner = ORDINARY_DECLARED_RUNNER
    if campaign.get("campaign_version") == "v2" and not supplied_component_runner:
        raise ValueError("Supplied component runner is required")
    if supplied_component_runner and supplied_component_runner != declared_runner:
        raise ValueError("Supplied component runner disagrees with campaign")
    if registered_runner != declared_runner:
        raise ValueError("Campaign runner disagrees with registry adapter")
    _validate_operational_package(package, job, parent_gate)
    operational_identity = str(package["package_id"])
    evidence = {
        "campaign_identity": supplied_campaign_identity,
        "plan_job_identity": str(
            package.get("plan_job_identity")
            or supplied_plan_job_identity
        ),
        "declared_component_runner": declared_runner,
        "operational_input_identity": operational_identity,
    }
    if declared_runner == ORDINARY_DECLARED_RUNNER:
        return ordinary_publisher(
            job=job, parent_gate_path=parent_gate_path,
            training_rows=training_rows, prediction_rows=prediction_rows,
            ledger_path=ledger_path, **evidence,
            resolved_runtime_owner=ORDINARY_RUNTIME_OWNER,
        )
    if declared_runner != WAVE4_DECLARED_RUNNER:
        raise ValueError(f"Unknown component runner: {declared_runner}")
    adapter = wave4_adapter or adapt_operational_package_to_wave4
    adapted = adapter(
        model_id=model_id, job=job, package=package,
        training_rows=training_rows, prediction_rows=prediction_rows,
    )
    return wave4_publisher(
        model_id=model_id, prediction_date=str(job["prediction_date"]),
        horizon_id=job.get("horizon_id"), fit_input=adapted["fit_input"],
        interaction_contract=adapted.get("interaction_contract"),
        fit_options=adapted.get("fit_options"),
        output_root=Path(str(job["authoritative_output_root"])),
        component_owner=Path(str(job["authoritative_output_root"])),
        parent_gate=parent_gate, ledger_path=ledger_path,
        production_plan_job_checksum=str(job["logical_checksum"]),
        **evidence, resolved_runtime_owner=WAVE4_RUNTIME_OWNER,
    )


def adapt_operational_package_to_wave4(
    *,
    model_id: str,
    job: Mapping[str, Any],
    package: Mapping[str, Any],
    training_rows: list[Mapping[str, Any]],
    prediction_rows: list[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Load a package-owned Wave-4 input; never reconstruct temporal boundaries."""
    fit_input_path = package.get("wave4_fit_input_path")
    if not fit_input_path:
        raise ValueError(
            f"Operational package is not Wave-4 compatible: {model_id}"
        )
    fit_input = json.loads(Path(str(fit_input_path)).read_text(encoding="utf-8"))
    if fit_input.get("operational_input_identity") != package.get("package_id"):
        raise ValueError("Wave-4 input operational identity mismatch")
    if fit_input.get("fold_identity") not in {
        package.get("fold_identity"), None
    } and fit_input.get("split_identity") != package.get("fold_identity"):
        raise ValueError("Wave-4 input fold identity mismatch")
    if fit_input.get("source_rows_embedded") is not True:
        raise ValueError("Wave-4 input must be package-owned and self-contained")
    options = {
        "operational_input_identity": package["package_id"],
        "operational_input_checksum": (
            fit_input.get("dataset_checksum")
            or fit_input.get("logical_input_checksum")
        ),
        "training_boundary_identity": package["fold_identity"],
        "training_cutoff": package["training_cutoff"],
        "purge_sessions": package["purge_sessions"],
        "embargo_sessions": package["embargo_sessions"],
        "source_commit": package["source_git_commit"],
    }
    return {
        "fit_input": fit_input,
        "interaction_contract": fit_input.get("interaction_contract"),
        "fit_options": options,
    }


def _validate_operational_package(
    package: Mapping[str, Any],
    job: Mapping[str, Any],
    parent_gate: Mapping[str, Any],
) -> None:
    if package.get("package_contract_version") == V2_PACKAGE_CONTRACT:
        manifest_path = package.get("package_manifest_path")
        if not manifest_path:
            raise ValueError("V2 operational package manifest path is missing")
        validated = validate_v2_package(Path(str(manifest_path)))
        if validated != dict(package):
            raise ValueError("V2 operational package differs from manifest")
        checks = {
            "production_plan_job_id": job.get("job_id"),
            "model_id": job.get("model_id"),
            "prediction_date": job.get("prediction_date"),
            "selector_dataset_checksum": parent_gate.get(
                "selector_dataset_artifact_checksum"
            ),
        }
        mismatches = [
            field for field, expected_value in checks.items()
            if package.get(field) != expected_value
        ]
        if mismatches:
            raise ValueError(
                "Operational-input package identity mismatch: "
                + ",".join(mismatches)
            )
        if (
            package.get("component_runner")
            != package.get("declared_component_runner")
            or not package.get("resolved_runtime_owner")
        ):
            raise ValueError("V2 package runner ownership mismatch")
        return
    expected = canonical_hash({
        key: value for key, value in package.items()
        if key != "logical_checksum"
    })
    if package.get("logical_checksum") != expected:
        raise ValueError("Operational-input package checksum mismatch")
    checks = {
        "production_plan_job_id": job.get("job_id"),
        "production_plan_job_checksum": job.get("logical_checksum"),
        "model_id": job.get("model_id"),
        "prediction_date": job.get("prediction_date"),
        "selector_dataset_checksum": parent_gate.get(
            "selector_dataset_artifact_checksum"
        ),
    }
    mismatches = [
        field for field, expected_value in checks.items()
        if package.get(field) != expected_value
    ]
    if mismatches:
        raise ValueError(
            "Operational-input package identity mismatch: "
            + ",".join(mismatches)
        )
    if not package.get("package_id") or package.get("publication_status") != "complete":
        raise ValueError("Operational-input package is incomplete")


def _counts(rows, field):
    result = {}
    for row in rows: result[row.get(field)] = result.get(row.get(field), 0) + 1
    return result


def _write_report_pair(path: Path, result: Mapping[str, Any], title: str) -> None:
    json_path = path if path.suffix == ".json" else path.with_suffix(".json"); md_path = json_path.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True); json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    lines = [f"# {title}", "", f"- Status: `{result.get('status', 'AUDIT_COMPLETE')}`", f"- Candidates: `{result.get('candidate_count', 'n/a')}`", f"- Components: `{result.get('component_count', 'n/a')}`", "- Fitting performed: `false`", "- Prediction performed: `false`"]
    lines.extend(f"- Blocker: `{reason}`" for reason in result.get("blocking_reasons", [])); md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
