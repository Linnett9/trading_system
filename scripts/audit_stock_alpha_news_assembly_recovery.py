from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.research.ml.stock_level.stock_alpha_news_assembly_recovery import (
    AssemblyRecoveryRequest,
    audit_assembly_recovery,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit bounded local recovery evidence for a news assembly."
    )
    parser.add_argument("--recovery-request", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.recovery_request.read_text(encoding="utf-8"))
        payload.pop("notice", None)
        payload.pop("request_identity", None)
        request = AssemblyRecoveryRequest(**{
            **payload,
            "search_roots": tuple(payload["search_roots"]),
            "expected_filename_patterns":
                tuple(payload["expected_filename_patterns"]),
            "expected_provider_scope":
                tuple(payload["expected_provider_scope"]),
        })
        result = audit_assembly_recovery(request, strict=args.strict)
        bounded = {
            key: result[key] for key in (
                "status", "request_identity", "candidate_count",
                "plausible_candidate_count", "full_hash_count",
                "target_checksum_found", "exact_match_count",
                "rebuild_readiness",
            )
        }
        bounded["blocker_count"] = len(result["blockers"])
        bounded["output_root"] = request.output_root
        print(json.dumps(bounded, sort_keys=True) if args.json else (
            f"{bounded['status']}: {bounded['candidate_count']} candidates, "
            f"{bounded['full_hash_count']} hashes"
        ))
        if result["status"] == "BLOCKED":
            return 2
        if args.strict and result["status"] != "READY":
            return 1
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({
            "status": "MALFORMED_OR_INCOMPATIBLE_REQUEST",
            "error_type": type(exc).__name__,
        }), file=sys.stderr)
        return 2
    except Exception as exc:
        print(json.dumps({
            "status": "UNEXPECTED_RECOVERY_FAILURE",
            "error_type": type(exc).__name__,
        }), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
