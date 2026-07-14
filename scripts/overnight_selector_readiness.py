from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infrastructure.data.overnight_selector_readiness import run_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Run overnight selector readiness reports and canonical daily v2 build.")
    parser.add_argument("--smoke-canonical-only", action="store_true", help="Build only the bounded canonical smoke subset.")
    args = parser.parse_args()
    result = run_all(build_full_canonical=not args.smoke_canonical_only)
    print(json.dumps(_summary(result), indent=2, sort_keys=True, default=str))
    return 0


def _summary(result: dict) -> dict:
    canonical = result.get("canonical", {})
    residual = result.get("residual", {})
    selector = result.get("selector", {})
    return {
        "gate": residual.get("gate"),
        "canonical_mode": canonical.get("mode"),
        "canonical_rows": canonical.get("row_count"),
        "canonical_symbols": canonical.get("symbol_count"),
        "canonical_date_min": canonical.get("date_min"),
        "canonical_date_max": canonical.get("date_max"),
        "labeled_spine_rows": (selector.get("labeled") or {}).get("row_count"),
        "inference_spine_rows": (selector.get("inference") or {}).get("row_count"),
        "full_alpha_run": (result.get("alpha") or {}).get("full_run"),
        "news_smoke_status": ((result.get("news") or {}).get("smoke") or {}).get("status"),
        "exposure_status": ((result.get("exposure") or {}).get("selector_oos") or {}).get("status"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
