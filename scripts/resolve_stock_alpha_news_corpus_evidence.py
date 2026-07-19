from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.research.ml.stock_level.stock_alpha_news_corpus_evidence import (
    CorpusEvidenceRequest,
    resolve_corpus_evidence,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve bounded, read-only canonical corpus evidence."
    )
    parser.add_argument("--evidence-request", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--draft-only", action="store_true")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--approve-selection", action="store_true")
    parser.add_argument("--emit-materialisation-request", action="store_true")
    parser.add_argument("--run-plan-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--max-metadata-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.evidence_request.read_text(encoding="utf-8"))
        payload.pop("notice", None)
        payload.pop("evidence_identity", None)
        request = CorpusEvidenceRequest(**{
            **payload,
            "external_roots": tuple(payload["external_roots"]),
            "expected_provider_scope": tuple(payload["expected_provider_scope"]),
        })
        selection = (
            json.loads(args.selection.read_text(encoding="utf-8"))
            if args.selection else None
        )
        if selection and "selection" in selection:
            selection = {
                **selection["selection"],
                "__assembly_checksum": selection.get("assembly_checksum"),
            }
        result = resolve_corpus_evidence(
            request, output_root=args.output_root, selection=selection,
            approve_selection=args.approve_selection,
            emit_materialisation_request=args.emit_materialisation_request,
            run_plan_only=args.run_plan_only, strict=args.strict,
            max_depth=args.max_depth, max_candidates=args.max_candidates,
            max_metadata_bytes=args.max_metadata_bytes,
            repository_root=REPO_ROOT,
        )
        bounded = {
            key: result[key] for key in (
                "status", "evidence_identity", "candidate_count",
                "canonical_status", "assembly_status",
                "approved_request_emitted", "plan_only_invoked",
            )
        }
        bounded["blocker_count"] = len(result["blockers"])
        bounded["output_root"] = str(args.output_root)
        print(json.dumps(bounded, sort_keys=True) if args.json else (
            f"{bounded['status']}: {bounded['candidate_count']} candidates, "
            f"{bounded['blocker_count']} blockers"
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
            "status": "UNEXPECTED_RESOLVER_FAILURE",
            "error_type": type(exc).__name__,
        }), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
