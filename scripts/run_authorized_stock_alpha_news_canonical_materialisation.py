from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.research.ml.stock_level.stock_alpha_news_canonical_authorization import (
    configuration_checksum,
    load_authorization,
    validate_authorization,
    validate_v2_request,
)
from core.research.ml.stock_level.stock_alpha_news_data_compute import (
    CORPUS,
    NewsDataMaterialisationPlan,
    deterministic_news_data_run_id,
)


def _json(path):
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return payload


def _plan(request, runtime_checksum):
    plan = NewsDataMaterialisationPlan(
        selected_stages=(CORPUS,),
        source_inventory_identity=request["source_assembly_identity"],
        source_inventory_checksum=request["source_assembly_checksum"],
        canonical_corpus_contract_identity=request["canonical_contract_version"],
        canonical_output_compatibility_identity=request[
            "logical_request_identity"
        ],
        canonical_parent_identity=request["source_assembly_identity"],
        canonical_parent_checksum=request["source_assembly_checksum"],
        date_boundary_identity=(
            f"{request['expected_published_min']}_"
            f"{request['expected_published_max']}_"
            f"{request['ingested_at_utc']}"
        ),
        universe_identity=str(request["expected_symbol_count"]),
        availability_policy_identity=request[
            "availability_time_policy_identity"
        ],
        pit_feature_contract_identity="",
        feature_output_compatibility_identity="",
        configuration_checksum=runtime_checksum,
        source_git_commit=request["source_git_commit_evidence"],
    )
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Authorization-aware canonical news materialisation owner."
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--resource-ledger", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        request = _json(args.request)
        validate_v2_request(request)
        runtime = _json(args.runtime_config)
        runtime_checksum = runtime["configuration_checksum"]
        material = runtime["materialisation"]
        if configuration_checksum(material) != runtime_checksum:
            raise ValueError("RUNTIME_CONFIGURATION_CHECKSUM_MISMATCH")
        if material["execution_authorized"] is not False:
            raise ValueError("RUNTIME_CONFIG_MUST_NOT_AUTHORIZE_EXECUTION")
        request_runtime_pairs = (
            ("source_assembly_path", "source_assembly_path"),
            ("source_metadata_path", "source_metadata_path"),
            ("source_assembly_checksum", "source_assembly_checksum"),
            ("canonical_output_root", "canonical_output_root"),
            ("ingested_at_utc", "ingested_at_utc"),
            (
                "ingested_at_utc_provenance_identity",
                "ingested_at_utc_provenance_identity",
            ),
            ("resource_profile", "resource_profile"),
            ("shared_compute_run_root", "shared_compute_run_root"),
            ("resource_ledger_path", "resource_ledger_path"),
            ("run_registry_path", "run_registry_path"),
        )
        for runtime_key, request_key in request_runtime_pairs:
            if material.get(runtime_key) != request.get(request_key):
                raise ValueError("RUNTIME_CONFIGURATION_REQUEST_MISMATCH")
        if (
            material.get("write_enabled") is not True
            or material.get("production_validated") is not False
            or material.get("execution_policy") != "CPU_ONLY_NON_MODEL"
            or material.get("network_access_allowed") is not False
        ):
            raise ValueError("RUNTIME_CONFIGURATION_POLICY_MISMATCH")
        plan = _plan(request, runtime_checksum)
        run_id = deterministic_news_data_run_id(plan)
        authorization = load_authorization(args.authorization)
        validation = validate_authorization(
            authorization, request, runtime_checksum, run_id,
            execution_required=args.execute,
        )
        for supplied, expected in (
            (args.run_root, Path(request["shared_compute_run_root"])),
            (args.resource_ledger, Path(request["resource_ledger_path"])),
            (args.registry, Path(request["run_registry_path"])),
        ):
            if supplied.resolve() != expected.resolve():
                raise ValueError("EXPLICIT_OPERATOR_PATH_MISMATCH")
        output = Path(request["canonical_output_root"]).resolve()
        source_parent = Path(request["source_assembly_path"]).resolve().parent
        if output.exists():
            raise ValueError("CANONICAL_OUTPUT_ALREADY_EXISTS")
        if output == source_parent or source_parent in output.parents:
            raise ValueError("CANONICAL_OUTPUT_NESTED_IN_SOURCE")
        if output in {
            args.run_root.resolve(), args.resource_ledger.resolve(),
            args.registry.resolve(), args.request.parent.resolve(),
        }:
            raise ValueError("CANONICAL_OUTPUT_ALIASES_OPERATOR_PATH")
        if args.plan_only:
            result = {
                "status": (
                    "PLAN_VALIDATED_NOT_AUTHORIZED_FOR_EXECUTION"
                    if not validation["execution_authorized"]
                    else "PLAN_VALIDATED_AUTHORIZED_NOT_EXECUTED"
                ),
                "run_id": run_id,
                "request_identity": request["logical_request_identity"],
                "authorization_identity":
                    validation["authorization_identity"],
                "execution_performed": False,
                "lease_acquired": False,
            }
            print(json.dumps(result, sort_keys=True))
            return 0
        execution_input = args.request.parent / "authorized_execution_input.json"
        payload = {
            "plan": {
                **plan.__dict__,
                "selected_stages": list(plan.selected_stages),
                "corpus_work_units": [],
                "feature_work_units": [],
            },
            "canonical": {
                "source_csv": request["source_assembly_path"],
                "source_metadata_json": request["source_metadata_path"],
                "output_dir": request["canonical_output_root"],
                "expected_source_checksum":
                    request["source_assembly_checksum"],
                "ingested_at_utc": request["ingested_at_utc"],
            },
        }
        execution_input.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        completed = subprocess.run([
            sys.executable,
            str(REPO_ROOT / "scripts/run_stock_alpha_news_data_compute.py"),
            "--request", str(execution_input),
            "--run-root", str(args.run_root),
            "--resource-ledger", str(args.resource_ledger),
            "--registry", str(args.registry),
        ], cwd=REPO_ROOT, check=False)
        return completed.returncode
    except Exception as exc:
        failure = {
            "status": "AUTHORIZED_MATERIALISATION_REJECTED",
            "error_type": type(exc).__name__,
        }
        if isinstance(exc, ValueError):
            error_code = str(exc)
            if error_code and all(
                character.isupper()
                or character.isdigit()
                or character == "_"
                for character in error_code
            ):
                failure["error_code"] = error_code
        print(json.dumps(failure), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
