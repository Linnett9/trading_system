from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.research.ml.stock_level.stock_alpha_news_compute_readiness_request import (
    NewsReadinessDiscoveryRequest,
    build_news_readiness_request,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a reviewed read-only news compute readiness request."
    )
    parser.add_argument("--discovery-request", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--draft-only", action="store_true")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--approve-selection", action="store_true")
    parser.add_argument("--run-audit", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--max-file-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.discovery_request.read_text(encoding="utf-8"))
        discovery = NewsReadinessDiscoveryRequest(
            **{
                **payload,
                "discovery_roots": tuple(payload["discovery_roots"]),
                "selected_stages": tuple(payload["selected_stages"]),
            }
        )
        selection = (
            json.loads(args.selection.read_text(encoding="utf-8"))
            if args.selection else None
        )
        if selection and "selection" in selection:
            selection = selection["selection"]
        result = build_news_readiness_request(
            discovery, output_root=args.output_root, selection=selection,
            approve_selection=args.approve_selection,
            run_audit=args.run_audit, strict=args.strict,
            max_depth=args.max_depth, max_candidates=args.max_candidates,
            max_file_bytes=args.max_file_bytes,
            repository_root=REPO_ROOT,
        )
        bounded = {
            "status": result["status"],
            "discovery_identity": result["discovery_identity"],
            "candidate_count": result["candidate_count"],
            "eligible_candidate_count": result["eligible_candidate_count"],
            "blocker_count": len(result["blockers"]),
            "warning_count": len(result["warnings"]),
            "approved_request_emitted": result["approved_request_emitted"],
            "audit_invoked": result["audit_invoked"],
            "audit_result": result["audit_result"],
            "output_root": str(args.output_root),
        }
        print(json.dumps(bounded, sort_keys=True) if args.json else (
            f"{bounded['status']}: {bounded['candidate_count']} candidates, "
            f"{bounded['blocker_count']} blockers; {args.output_root}"
        ))
        if result["status"] == "BLOCKED":
            return 2
        if result["status"] == "READY_WITH_CONDITIONS":
            return 1 if args.strict else 0
        return 0
    except Exception as exc:
        print(json.dumps({
            "status": "BUILDER_FAILED", "error_type": type(exc).__name__
        }), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
