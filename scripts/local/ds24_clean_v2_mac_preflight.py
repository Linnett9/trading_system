from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.clean_v2_contracts import authority_bundle, file_sha256
from core.research.ml.ds24.clean_v2_runtime import validate_ownership
from core.research.ml.ds24.clean_v2_contracts import load_contract


def old_ds24_processes() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,lstart=,command="],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        lowered = line.lower()
        if "ds24" not in lowered or "ds24_clean_v2" in lowered:
            continue
        pieces = line.strip().split(maxsplit=1)
        if pieces and pieces[0].isdigit():
            rows.append({"pid": int(pieces[0]), "command": pieces[1] if len(pieces) > 1 else ""})
    return rows


def manifest_file_hashes_match(
    root: Path,
    manifest: dict[str, Any],
    *,
    relative_path_key: str,
    sha256_key: str,
) -> bool:
    partitions = manifest.get("partitions", [])
    if not manifest.get("complete") or not isinstance(partitions, list) or not partitions:
        return False
    return partition_file_hashes_match(
        root,
        partitions,
        relative_path_key=relative_path_key,
        sha256_key=sha256_key,
    )


def partition_file_hashes_match(
    root: Path,
    partitions: list[dict[str, Any]],
    *,
    relative_path_key: str,
    sha256_key: str,
) -> bool:
    if not partitions:
        return False
    for partition in partitions:
        if not isinstance(partition, dict):
            return False
        relative = Path(str(partition.get(relative_path_key, "")))
        if relative.is_absolute() or ".." in relative.parts:
            return False
        candidate = root / relative
        expected = partition.get(sha256_key)
        if not candidate.is_file() or not isinstance(expected, str):
            return False
        if file_sha256(candidate) != expected:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Mac admission gate for DS24 clean V2.")
    parser.add_argument("--expected-bundle-hash", required=True)
    parser.add_argument("--expected-feature-hash", required=True)
    parser.add_argument("--expected-git-head", required=True)
    parser.add_argument("--expected-target-authority-hash", required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    bundle = authority_bundle()
    ownership = load_contract("cross_host_ownership.json")
    feature = load_contract("feature_authority.json")
    target = load_contract("target_contract.json")
    validate_ownership(ownership["hosts"])
    old = old_ds24_processes()
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False, timeout=15
    )
    supervisor = subprocess.run(
        [sys.executable, "scripts/local/ds24_clean_v2_supervisor.py", "--host", "mac"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    try:
        supervisor_report = json.loads(supervisor.stdout)
    except json.JSONDecodeError:
        supervisor_report = {"ready": False, "blocking_reasons": ["MAC_SUPERVISOR_PREFLIGHT_UNREADABLE"]}
    reasons = list(supervisor_report.get("blocking_reasons", []))
    if old:
        reasons.append("OLD_CONTAMINATED_DS24_PROCESS_ACTIVE")
    if bundle["bundle_sha256"] != args.expected_bundle_hash:
        reasons.append("STATIC_AUTHORITY_BUNDLE_HASH_MISMATCH")
    if supervisor_report.get("feature_authority_hash") != args.expected_feature_hash:
        reasons.append("FEATURE_AUTHORITY_HASH_MISMATCH")
    if git.returncode != 0 or git.stdout.strip() != args.expected_git_head:
        reasons.append("GIT_HEAD_MISMATCH")
    if supervisor_report.get("target_authority_hash") != args.expected_target_authority_hash:
        reasons.append("TARGET_AUTHORITY_HASH_MISMATCH")
    sidecar_root = ROOT / feature["sidecar"]["path"]
    sidecar_manifest_path = sidecar_root / "authority_manifest.json"
    sidecar_manifest = (
        json.loads(sidecar_manifest_path.read_text(encoding="utf-8"))
        if sidecar_manifest_path.is_file()
        else {}
    )
    if not manifest_file_hashes_match(
        sidecar_root,
        sidecar_manifest,
        relative_path_key="relative_path",
        sha256_key="sha256",
    ):
        reasons.append("FEATURE_SIDECAR_FILE_HASH_MISMATCH")
    if not manifest_file_hashes_match(
        ROOT / feature["base_authority"]["path"],
        sidecar_manifest,
        relative_path_key="base_relative_path",
        sha256_key="base_sha256",
    ):
        reasons.append("FEATURE_BASE_FILE_HASH_MISMATCH")
    target_root = ROOT / target["physical_authority"]["delta_path"]
    target_manifest_path = target_root / "authority_manifest.json"
    target_manifest = (
        json.loads(target_manifest_path.read_text(encoding="utf-8"))
        if target_manifest_path.is_file()
        else {}
    )
    if not manifest_file_hashes_match(
        target_root,
        target_manifest,
        relative_path_key="relative_path",
        sha256_key="sha256",
    ):
        reasons.append("TARGET_FILE_HASH_MISMATCH")
    target_base_partitions = (target_manifest.get("base_snapshot") or {}).get(
        "partitions", []
    )
    if not partition_file_hashes_match(
        ROOT / target["physical_authority"]["base_path"],
        target_base_partitions,
        relative_path_key="base_relative_path",
        sha256_key="base_sha256",
    ):
        reasons.append("TARGET_BASE_FILE_HASH_MISMATCH")
    report = {
        "classification": "DS24_CLEAN_V2_MAC_PREFLIGHT",
        "git_head": git.stdout.strip(),
        "static_authority_bundle_sha256": bundle["bundle_sha256"],
        "old_ds24_processes": old,
        "family_ownership": ownership["hosts"]["mac"],
        "feature_authority_hash": supervisor_report.get("feature_authority_hash"),
        "target_authority_hash": supervisor_report.get("target_authority_hash"),
        "target_contract_hash": supervisor_report.get("target_contract_hash"),
        "blocking_reasons": sorted(set(reasons)),
        "ready": not reasons,
        "paper_orders": 0,
        "live_orders": 0,
    }
    if args.launch and report["ready"]:
        launched = subprocess.run(
            [sys.executable, "scripts/local/ds24_clean_v2_supervisor.py", "--host", "mac", "--launch"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        report["launch_exit_code"] = launched.returncode
        report["launch_report"] = launched.stdout
    elif args.launch:
        report["launch_refused"] = True
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
