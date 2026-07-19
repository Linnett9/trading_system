from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.research.compute.machine_profile import dell_i5_10500_profile
from core.research.compute.resource_lease_ledger import ResourceLeaseLedger
from core.research.ml.stock_level.stock_alpha_news_compute_adapters import (
    AuthoritativeNewsDataAdapter,
    CanonicalCorpusBinding,
    PitFeatureStoreBinding,
)
from core.research.ml.stock_level.stock_alpha_news_data_compute import (
    NewsDataMaterialisationPlan,
    deterministic_news_data_run_id,
    execute_news_data_compute_run,
)
from core.research.ml.stock_level.stock_alpha_news_pit_policy import (
    StockAlphaNewsPitPolicy,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run shared-compute news data stages.")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--resource-ledger", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        plan = NewsDataMaterialisationPlan(**request["plan"])
        if args.plan_only:
            print(json.dumps({"status": "PLAN_VALID",
                              "run_id": deterministic_news_data_run_id(plan)},
                             sort_keys=True))
            return 0
        canonical = request.get("canonical")
        feature = request.get("feature_store")
        canonical_binding = CanonicalCorpusBinding(
            **{**canonical, "source_csv": Path(canonical["source_csv"]),
               "source_metadata_json": Path(canonical["source_metadata_json"]),
               "output_dir": Path(canonical["output_dir"])}
        ) if canonical else None
        feature_binding = None
        if feature:
            paths = (
                "score_store_manifest_path", "daily_spine_manifest_path",
                "ticker_mapping_manifest_path", "scored_articles_path",
                "spine_rows_path", "ticker_aliases_path", "output_root",
            )
            values = {**feature, **{name: Path(feature[name]) for name in paths}}
            values["pit_policy"] = StockAlphaNewsPitPolicy(**feature["pit_policy"])
            feature_binding = PitFeatureStoreBinding(**values)
        profile = dell_i5_10500_profile(source_git_commit=plan.source_git_commit)
        ledger = ResourceLeaseLedger(
            profile=profile, path=args.resource_ledger,
            available_memory=lambda: profile.total_ram_bytes,
        )
        ledger.initialise_ledger()
        result = execute_news_data_compute_run(
            plan=plan, adapter=AuthoritativeNewsDataAdapter(
                canonical=canonical_binding, feature_store=feature_binding),
            machine_profile=profile, lease_ledger=ledger,
            runs_root=args.run_root, registry_path=args.registry,
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
