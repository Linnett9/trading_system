"""Read-only reporter for normalized DS24 V11X package profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.cross_host_throughput import (  # noqa: E402
    ALL_HOST_FAMILY_LANES,
    PackageProfile,
    ProfileContractError,
    distributed_critical_path,
    summarize_family_profiles,
)


def _load_profiles(path: Path) -> list[PackageProfile]:
    documents: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ProfileContractError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        documents = payload if isinstance(payload, list) else [payload]
    return [PackageProfile.from_payload(document) for document in documents]


def build_report(
    profiles: list[PackageProfile], remaining: dict[str, int]
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[PackageProfile]] = {}
    for profile in profiles:
        grouped.setdefault((profile.host, profile.family), []).append(profile)
    matrix: list[dict[str, Any]] = []
    for host, family in ALL_HOST_FAMILY_LANES:
        rows = grouped.get((host, family))
        if rows:
            matrix.append(
                summarize_family_profiles(
                    rows,
                    remaining_packages=remaining.get(f"{host}:{family}"),
                )
            )
        else:
            matrix.append(
                {
                    "host": host,
                    "family": family,
                    "sample_count": 0,
                    "sec_per_refit_p50": None,
                    "sec_per_refit_p90": None,
                    "sec_per_refit_p99": None,
                    "wait_percent": None,
                    "io_percent": None,
                    "fit_percent": None,
                    "score_evaluation_percent": None,
                    "publish_percent": None,
                    "primary_bottleneck": "UNINSTRUMENTED",
                    "remaining_packages": remaining.get(f"{host}:{family}"),
                    "estimated_remaining_hours": None,
                    "uninstrumented_stages": ["ALL"],
                }
            )
    return {
        "classification": "V11X_CROSS_HOST_PROFILE_COMPLETE"
        if all(
            row["sample_count"] > 0 and not row["uninstrumented_stages"]
            for row in matrix
        )
        else "V11X_CROSS_HOST_PROFILE_PARTIAL",
        "performance_matrix": matrix,
        "distributed_critical_path": distributed_critical_path(matrix),
    }


def _parse_remaining(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        try:
            lane, count_text = value.rsplit("=", 1)
            host, family = lane.split(":", 1)
            count = int(count_text)
        except ValueError as exc:
            raise ProfileContractError(
                "--remaining must use HOST:FAMILY=COUNT"
            ) from exc
        if count < 0:
            raise ProfileContractError("Remaining package count cannot be negative")
        result[f"{host}:{family}"] = count
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="+", type=Path)
    parser.add_argument(
        "--remaining",
        action="append",
        default=[],
        metavar="HOST:FAMILY=COUNT",
    )
    args = parser.parse_args()
    try:
        profiles = [
            profile for path in args.profiles for profile in _load_profiles(path)
        ]
        report = build_report(profiles, _parse_remaining(args.remaining))
    except (OSError, json.JSONDecodeError, ProfileContractError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
