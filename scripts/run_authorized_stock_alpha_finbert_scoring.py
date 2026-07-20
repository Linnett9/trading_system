from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.research.ml.stock_level.stock_alpha_finbert_scoring_foundation import (
    authorization_template,
    deterministic_run_id,
    execute_authorized_boundary,
    logical_identity,
    validate_model_package,
)
from core.research.ml.stock_level.stock_alpha_finbert_authorized_execution import (
    ProductionExecutionFactory,
    execute_authorized_scoring,
)


def main(argv=None, *, factory_builder=ProductionExecutionFactory) -> int:
    parser = argparse.ArgumentParser(
        description="Validate or execute an exactly authorized FinBERT scoring request."
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--chunk-plan", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--resource-ledger", type=Path)
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args(argv)
    try:
        request = json.loads(args.request.read_text(encoding="utf-8-sig"))
        runtime = json.loads(args.runtime_config.read_text(encoding="utf-8-sig"))
        authorization = json.loads(
            args.authorization.read_text(encoding="utf-8-sig")
        )
        runtime_checksum = logical_identity(runtime)
        if args.plan_only == args.execute:
            raise ValueError("EXACTLY_ONE_OF_PLAN_ONLY_OR_EXECUTE_REQUIRED")
        if args.execute:
            required = {
                "--chunk-plan": args.chunk_plan, "--run-root": args.run_root,
                "--resource-ledger": args.resource_ledger,
                "--registry": args.registry,
            }
            missing = [key for key, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    "EXECUTION_SHARED_PATHS_REQUIRED:" + ",".join(missing)
                )
            chunk_plan = json.loads(
                args.chunk_plan.read_text(encoding="utf-8-sig")
            )
            factory = factory_builder(
                request=request, runtime=runtime, chunk_plan=chunk_plan,
                runs_root=args.run_root, ledger_path=args.resource_ledger,
                registry_path=args.registry,
            )
            result = execute_authorized_scoring(
                request=request, runtime=runtime,
                runtime_checksum=runtime_checksum,
                authorization=authorization, chunk_plan=chunk_plan,
                callbacks=factory.callbacks(), plan_only=False,
            )
            print(json.dumps(result, sort_keys=True))
            return 0
        result = execute_authorized_boundary(
            request=request, runtime_checksum=runtime_checksum,
            authorization=authorization, plan_only=True,
            validate_package=lambda: validate_model_package(
                Path(request["model_package_root"]),
                expected_model_name=request["model_name"],
                expected_revision=request["model_revision"],
            ),
            persist_resource_request=_forbidden_real_execution,
            acquire_lease=_forbidden_real_execution,
            activate_source=_forbidden_real_execution,
            activate_model=_forbidden_real_execution,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({
            "status": "FAILED", "error_type": type(exc).__name__,
            "error": str(exc),
        }, sort_keys=True), file=sys.stderr)
        return 2


def _forbidden_real_execution():
    raise RuntimeError(
        "REAL_EXECUTION_NOT_WIRED_UNTIL_OPERATOR_SELECTION_AND_AUTHORIZATION"
    )


if __name__ == "__main__":
    raise SystemExit(main())
