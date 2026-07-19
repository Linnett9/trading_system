from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.research.compute.machine_profile import dell_i5_10500_profile
from core.research.compute.resource_lease_ledger import ResourceLeaseLedger
from core.research.ml.stock_level.stock_alpha_finbert_compute import (
    FinBertExecutionPolicy,
    deterministic_run_id,
    execute_finbert_compute_run,
)
from core.research.ml.stock_level.stock_alpha_finbert_news import (
    HuggingFaceFinBertAdapter,
)
from core.research.ml.stock_level.stock_alpha_news_compute_adapters import (
    AuthoritativeCertificationAdapter,
    AuthoritativeFinBertChunkAdapter,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run shared-compute FinBERT scoring.")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--resource-ledger", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args(argv)
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        plan = json.loads(Path(request["scoring_plan_path"]).read_text(encoding="utf-8"))
        policy = FinBertExecutionPolicy(args.device, args.allow_gpu)
        if args.plan_only:
            print(json.dumps({"status": "PLAN_VALID", "run_id":
                              deterministic_run_id(plan, policy)}, sort_keys=True))
            return 0
        with Path(request["source_rows_path"]).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        config = json.loads(Path(request["scoring_config_path"]).read_text(
            encoding="utf-8"))
        profile = dell_i5_10500_profile(source_git_commit=request["source_commit"])
        ledger = ResourceLeaseLedger(
            profile=profile, path=args.resource_ledger,
            available_memory=lambda: profile.total_ram_bytes,
            available_gpus=lambda: profile.gpu_inventory,
        )
        ledger.initialise_ledger()

        def factory(reference, execution_policy):
            return HuggingFaceFinBertAdapter(
                model_id=reference["repository"],
                model_revision=reference["revision"],
                tokenizer_id=reference["tokenizer_repository"],
                tokenizer_revision=reference["tokenizer_revision"],
                device=execution_policy.device, local_files_only=True,
                max_token_length=int(plan["maximum_token_length"]),
            )

        output = Path(request["scoring_output_dir"])
        adapter = AuthoritativeFinBertChunkAdapter(
            scoring_plan=plan, source_rows=rows, output_dir=output,
            scoring_config=config, model_factory=factory,
            scored_at=request["scored_at"],
        )
        certifier = AuthoritativeCertificationAdapter(
            chunk_manifest_path=output / "finbert_chunk_manifest.csv",
            output_path=Path(request["certificate_output_path"]),
            source_commit=request["source_commit"],
        )
        result = execute_finbert_compute_run(
            scoring_plan=plan, adapter=adapter, certify=certifier,
            machine_profile=profile, lease_ledger=ledger,
            runs_root=args.run_root, registry_path=args.registry,
            policy=policy,
        )
        print(json.dumps({"run_id": result["run_id"],
                          "status": result["summary"]["final_run_status"],
                          "summary_path": str(Path(result["run_root"]) / "summary.md")},
                         sort_keys=True))
        successful = (
            result["summary"]["final_run_status"] == "COMPONENTS_COMPLETE"
            and result["registry"].get("health") == "HEALTHY"
        )
        return 0 if successful else 2
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__}),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
