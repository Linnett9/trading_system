from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.clean_v2_certification import (
    certification_cases,
    source_fixture_summary,
    synthetic_multiyear_raw_frames,
)
from core.research.ml.ds24.clean_v2_contracts import authority_bundle, load_contract, stable_hash
from core.research.ml.ds24.clean_v2_features import certify_future_bar_invariance


def main() -> int:
    parser = argparse.ArgumentParser(description="Certify DS24 clean-V2 value causality.")
    parser.add_argument(
        "--output",
        default="research_runs/ds24_clean_v2/DS24_CLEAN_V2_TOURNAMENT_R1_20260926/causality_certificate.json",
    )
    args = parser.parse_args()
    frames = synthetic_multiyear_raw_frames()
    certificate = certify_future_bar_invariance(frames, certification_cases(frames))
    certificate["fixture"] = source_fixture_summary(frames)
    certificate["feature_authority_id"] = load_contract("feature_authority.json")["authority_id"]
    certificate["static_authority_bundle_sha256"] = authority_bundle()["bundle_sha256"]
    certificate["certificate_sha256"] = stable_hash(
        {key: value for key, value in certificate.items() if key != "certificate_sha256"}
    )
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0 if certificate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
