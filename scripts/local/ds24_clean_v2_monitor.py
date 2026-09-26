from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "research_runs/ds24_clean_v2/DS24_CLEAN_V2_TOURNAMENT_R1_20260926"


def main() -> int:
    status_path = RUN_ROOT / "supervisor_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    disk = shutil.disk_usage(ROOT.anchor or ROOT)
    rows = []
    for family_path in sorted(RUN_ROOT.glob("family=*/resume_state.json")):
        state = json.loads(family_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "family": state.get("family"),
                "state": state.get("terminal_state"),
                "current_historical_date": state.get("latest_scored_decision"),
                "completed_refits": len(state.get("completed_refits", [])),
                "expected_refits": state.get("expected_refits"),
                "score_timestamps": state.get("metrics_cursor", 0),
                "current_rank_ic": state.get("current_rank_ic"),
                "top_n_spread": state.get("top_n_spread"),
                "net_10bps": state.get("net_10bps"),
                "error": state.get("error"),
                "causal_authority_hash": state.get("feature_authority_hash"),
            }
        )
    report = {
        "run_id": "DS24_CLEAN_V2_TOURNAMENT_R1_20260926",
        "classification": status.get("classification", "NOT_STARTED"),
        "supervisor_pid": status.get("supervisor_pid"),
        "families": rows,
        "disk_free_gib": round(disk.free / 1024**3, 3),
        "ram_available_bytes": status.get("ram_available_bytes"),
        "blocking_reasons": status.get("blocking_reasons", []),
        "feature_authority_hash": status.get("feature_authority_hash"),
        "target_authority_hash": status.get("target_authority_hash"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
