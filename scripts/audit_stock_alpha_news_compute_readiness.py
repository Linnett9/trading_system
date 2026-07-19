from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.research.ml.stock_level.stock_alpha_news_compute_readiness import (
    NewsComputeReadinessRequest,
    audit_news_compute_readiness,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only shared news compute production-readiness audit."
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--selected-stage", action="append")
    parser.add_argument("--max-blocker-examples", type=int, default=10)
    parser.add_argument("--max-chunk-examples", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.request.read_text(encoding="utf-8"))
        payload["audit_output_path"] = str(args.output_root)
        if args.selected_stage:
            payload["selected_stages"] = args.selected_stage
        request = NewsComputeReadinessRequest(
            **{**payload, "selected_stages": tuple(payload["selected_stages"])}
        )
        report = audit_news_compute_readiness(
            request, max_blocker_examples=args.max_blocker_examples,
            max_chunk_examples=args.max_chunk_examples,
            repository_root=REPO_ROOT,
        )
        result = {
            "audit_mode": report["audit_mode"],
            "overall_readiness": report["overall_readiness"],
            "request_identity": report["request_identity"],
            "blocker_count": len(report["blockers"]),
            "warning_count": len(report["warnings"]),
            "report_path": str(args.output_root / "readiness_report.json"),
        }
        if args.json:
            print(json.dumps(result, sort_keys=True))
        else:
            print(
                f"{result['overall_readiness']}: "
                f"{result['blocker_count']} blockers, "
                f"{result['warning_count']} warnings; "
                f"{result['report_path']}"
            )
        if not args.strict:
            return 0
        return {"READY": 0, "READY_WITH_CONDITIONS": 1, "BLOCKED": 2}[
            report["overall_readiness"]
        ]
    except Exception as exc:
        print(json.dumps({
            "status": "AUDIT_FAILED", "error_type": type(exc).__name__
        }), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
