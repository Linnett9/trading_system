from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local import ds24_lightgbm_ranking_readiness as readiness


SUPPORTED_FAMILIES = readiness.SCOPED_FAMILIES
CLI_OPTIONS = (
    "--family",
    "--resume",
    "--resume-generation",
    "--threads",
    "--cache-budget-gb",
    "--prediction-batch-decisions",
    "--evaluation-version",
    "--metrics-root-name",
    "--refit-policy",
    "--synthetic-certify",
    "--publish-readiness",
    "--output-root",
)


class LightGbmRankingWorkerError(RuntimeError):
    pass


def assert_supported_family(family: str) -> str:
    normalized = str(family).strip().lower().replace("-", "_")
    if normalized not in SUPPORTED_FAMILIES:
        raise LightGbmRankingWorkerError(f"DS24_V3_LIGHTGBM_RANKING_UNSUPPORTED_FAMILY:{normalized}")
    return normalized


def run_bounded_certification(family: str, root: Path) -> dict[str, object]:
    family = assert_supported_family(family)
    record = readiness.family_record(family, root)
    return {
        "family": family,
        "state": record["state"],
        "worker_path": str((ROOT / "scripts/local/ds24_v3_lightgbm_ranking_policy_worker.py").resolve()),
        "supported_cli": list(CLI_OPTIONS),
        "query_group_authority": record.get("query_group_authority", {}),
        "objective_smoke": record.get("objective_smoke", {}),
        "resume_proof": record.get("resume_proof", {}),
        "runtime_guards": record.get("runtime_guards", {}),
        "launch_mode": "bounded_certification_only",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DS24 V3 LightGBM ranking policy worker route")
    parser.add_argument("--family", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-generation", default="1")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cache-budget-gb", type=float, default=1.0)
    parser.add_argument("--prediction-batch-decisions", type=int, default=2)
    parser.add_argument("--evaluation-version", default="v3")
    parser.add_argument("--metrics-root-name", default="metrics_only_v3_lightgbm_ranking")
    parser.add_argument("--refit-policy", default="daily_session_v1")
    parser.add_argument("--synthetic-certify", action="store_true")
    parser.add_argument("--publish-readiness", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    family = assert_supported_family(args.family)
    if args.evaluation_version != "v3":
        raise LightGbmRankingWorkerError(f"DS24_V3_LIGHTGBM_RANKING_REQUIRES_V3:{args.evaluation_version}")
    if args.refit_policy != "daily_session_v1":
        raise LightGbmRankingWorkerError(f"DS24_V3_LIGHTGBM_RANKING_REQUIRES_DAILY_SESSION_REFIT:{args.refit_policy}")
    if args.threads != 1:
        raise LightGbmRankingWorkerError(f"DS24_V3_LIGHTGBM_RANKING_REQUIRES_SINGLE_THREAD:{args.threads}")
    if args.publish_readiness:
        manifest = readiness.publish_manifest()
        states = {row["family"]: row["state"] for row in manifest["families"]}
        print(json.dumps({"manifest_path": str(readiness.MANIFEST_PATH), "states": states}, sort_keys=True))
        return 0
    if not args.synthetic_certify:
        print(
            json.dumps(
                {
                    "family": family,
                    "state": "READY_FOR_BOUNDED_CERTIFICATION_ONLY",
                    "reason": "historical launch remains disabled until an explicit supervisor queue update admits this R42 route",
                },
                sort_keys=True,
            )
        )
        return 0
    with tempfile.TemporaryDirectory(prefix="ds24_v3_lightgbm_ranking_worker_cli_") as temp_dir:
        root = args.output_root if args.output_root is not None else Path(temp_dir)
        result = run_bounded_certification(family, root)
        print(json.dumps({"family": family, "state": result["state"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
