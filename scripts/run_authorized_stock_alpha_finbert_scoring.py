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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate or execute an exactly authorized FinBERT scoring request."
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        request = json.loads(args.request.read_text(encoding="utf-8-sig"))
        runtime = json.loads(args.runtime_config.read_text(encoding="utf-8-sig"))
        authorization = json.loads(
            args.authorization.read_text(encoding="utf-8-sig")
        )
        runtime_checksum = logical_identity(runtime)
        result = execute_authorized_boundary(
            request=request, runtime_checksum=runtime_checksum,
            authorization=authorization, plan_only=args.plan_only,
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
