from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.research.ml.reference.historical_reporting_entities import (
    CONTRACT_VERSION, preflight, read_csv, selector_audit, sha256_file, validate_intervals,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Historical reporting-entity authority gate")
    result.add_argument("command", choices=("preflight", "audit", "certify"))
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--selector", type=Path, required=True)
    result.add_argument("--selector-manifest", type=Path, required=True)
    result.add_argument("--historical-evidence", type=Path)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--mapping-artifact", type=Path)
    result.add_argument("--output", type=Path)
    return result


def paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.data_root
    return {
        "asset_registry": root / "data/reference/assets/canonical_asset_registry.csv",
        "aliases": root / "data/reference/assets/provider_symbol_aliases.csv",
        "selector": args.selector, "selector_manifest": args.selector_manifest,
        "current_mapping": root / "reports/ml/development/ticket_5b3_full_sec_current_universe/fundamentals_entity_mapping.csv",
        "fund_policy": root / "config/news_source_registry.stock_alpha_etf_funds.yaml",
        "companyfacts_root": root / "data/raw/fundamentals",
        "submissions_root": root / "data/raw",
        "historical_evidence": args.historical_evidence or Path("__HISTORICAL_EVIDENCE_NOT_CONFIGURED__"),
        "overrides": root / "config/reference/historical_reporting_entity_overrides_v1.yaml",
        "output_root": args.output_root,
    }


def main() -> int:
    args = parser().parse_args()
    if args.command == "preflight":
        if args.output:
            raise SystemExit("preflight is mutation-free; --output is not permitted")
        payload = preflight(paths(args))
    else:
        if not args.mapping_artifact:
            raise SystemExit("--mapping-artifact is required")
        rows = read_csv(args.mapping_artifact)
        validation = validate_intervals(rows)
        if args.command == "audit":
            payload = {"contract_version": CONTRACT_VERSION, "validation": validation,
                       "selector_population_audit": selector_audit(args.selector, rows)}
        else:
            payload = {"contract_version": CONTRACT_VERSION, "artifact_path": str(args.mapping_artifact),
                       "artifact_sha256": sha256_file(args.mapping_artifact),
                       "artifact_size_bytes": args.mapping_artifact.stat().st_size,
                       "row_count": len(rows), "validation": validation,
                       "certified": validation["status"] == "PASS"}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if payload.get("status", payload.get("validation", {}).get("status")) in {"READY", "PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
