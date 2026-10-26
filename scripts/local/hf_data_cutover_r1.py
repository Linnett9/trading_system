"""CLI composition root for HF-DATA-CUTOVER-R1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from application.services.hf_data_cutover_service import (
    CutoverPaths,
    finalize_completed_audit,
    run_audit,
)
from application.services.hf_data_cutover_retirement import execute_retirement
from application.services.hf_data_cutover_iex_pilot import (
    probe_iex_historical_coverage,
    run_iex_pilot,
)
from application.services.hf_data_cutover_iex_backfill import (
    run_iex_raw_backfill,
    run_iex_transport_connectivity_probe,
)
from application.services.hf_data_cutover_iex_adjusted_daily import run_iex_adjusted_daily_build
from application.services.hf_data_cutover_iex_derivation import run_iex_hour_derivation
from application.services.hf_data_cutover_iex_features import run_iex_feature_build
from application.services.hf_data_cutover_feature_validation import run_iex_feature_validation
from application.services.hf_data_cutover_iex_targets import run_iex_target_build
from application.services.hf_data_cutover_iex_validation import run_iex_dataset_validation
from application.services.hf_data_cutover_perf1 import run_perf1


DEFAULT_RUN_ID = "HF_DATA_CUTOVER_R1_20261025T154506Z"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Governed SIP-to-IEX data cutover")
    parser.add_argument("command", choices=("audit", "finalize-audit", "retire", "pilot", "coverage", "transport-probe", "perf1", "backfill", "adjust-daily", "derive-1h", "features", "feature-validate", "targets", "validate"))
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--r1-root", type=Path, default=Path(r"C:\Users\Brandon\hf_alpha_sip_discovery\hf_alpha_sip_r1_20261024t185833z"))
    parser.add_argument("--r2-root", type=Path, default=Path(r"C:\Users\Brandon\hf_alpha_sip_discovery\hf_alpha_sip_r2_20261025t125948z"))
    parser.add_argument("--audit-symbol-count", type=int, default=48)
    parser.add_argument("--expected-intent-sha256")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    repository_root = arguments.repository_root.resolve()
    output_root = arguments.output_root or repository_root / "docs/audits" / DEFAULT_RUN_ID
    paths = CutoverPaths(
        repository_root=repository_root,
        output_root=output_root.resolve(),
        r1_root=arguments.r1_root.resolve(),
        r2_root=arguments.r2_root.resolve(),
    )
    if arguments.command == "audit":
        result = run_audit(paths, symbol_count=arguments.audit_symbol_count)
    elif arguments.command == "finalize-audit":
        result = finalize_completed_audit(paths)
    elif arguments.command == "retire":
        if not arguments.expected_intent_sha256:
            raise SystemExit("retire requires --expected-intent-sha256")
        result = execute_retirement(paths, expected_intent_sha256=arguments.expected_intent_sha256)
    elif arguments.command == "pilot":
        result = run_iex_pilot(paths)
    elif arguments.command == "coverage":
        result = probe_iex_historical_coverage(paths)
    elif arguments.command == "transport-probe":
        result = run_iex_transport_connectivity_probe(paths)
    elif arguments.command == "perf1":
        result = run_perf1(paths)
    elif arguments.command == "backfill":
        result = run_iex_raw_backfill(paths)
    elif arguments.command == "adjust-daily":
        result = run_iex_adjusted_daily_build(paths)
    elif arguments.command == "derive-1h":
        result = run_iex_hour_derivation(paths)
    elif arguments.command == "features":
        result = run_iex_feature_build(paths)
    elif arguments.command == "feature-validate":
        result = run_iex_feature_validation(paths)
    elif arguments.command == "targets":
        result = run_iex_target_build(paths)
    else:
        result = run_iex_dataset_validation(paths)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
