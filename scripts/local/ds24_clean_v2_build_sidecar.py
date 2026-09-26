from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.clean_v2_contracts import load_contract, stable_hash
from core.research.ml.ds24.clean_v2_features import (
    REPAIRED_FEATURES,
    SOURCE_TIMESTAMP_SUFFIX,
    FeatureAuthorityError,
    compute_repaired_session_features,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_paths() -> tuple[Path, Path, Path]:
    contract = load_contract("feature_authority.json")
    base = ROOT / contract["base_authority"]["path"]
    raw = ROOT / contract["canonical_raw_source"]["path"]
    output = ROOT / contract["sidecar"]["path"]
    return base, raw, output


def _partitions(base: Path) -> list[Path]:
    return sorted(base.glob("stock/asset=*/year=*/features.parquet"))


def estimate() -> dict[str, Any]:
    import pyarrow.parquet as pq

    contract = load_contract("feature_authority.json")
    tournament = load_contract("tournament_contract.json")
    base = ROOT / contract["base_authority"]["path"]
    partitions = _partitions(base)
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in partitions)
    # Conservative encoded estimate: keys plus one float32 and one timestamp per repair.
    projected = int(rows * (24 + len(REPAIRED_FEATURES) * 12) * 0.65)
    disk = shutil.disk_usage(ROOT.anchor or ROOT)
    tournament_bytes = int(float(tournament["storage"]["projected_tournament_gib_upper_bound"]) * 1024**3)
    reserve_bytes = int(float(tournament["storage"]["minimum_post_launch_free_gib"]) * 1024**3)
    required = projected + tournament_bytes + reserve_bytes
    return {
        "row_count": rows,
        "partition_count": len(partitions),
        "projected_sidecar_bytes": projected,
        "projected_sidecar_gib": round(projected / 1024**3, 3),
        "disk_free_bytes": disk.free,
        "disk_free_gib": round(disk.free / 1024**3, 3),
        "required_free_bytes_including_tournament_and_reserve": required,
        "resource_gate_passed": disk.free >= required,
    }


def _read_raw_context(raw_root: Path, symbol: str, years: list[int]) -> pd.DataFrame:
    paths = []
    for candidate_year in range(min(years) - 1, max(years) + 1):
        candidate = raw_root / f"symbol={symbol}" / f"year={candidate_year}" / "bars.parquet"
        if candidate.is_file():
            paths.append(candidate)
    if not paths:
        raise FeatureAuthorityError(f"No raw bars for {symbol} years {years}")
    columns = [
        "asset_id",
        "canonical_symbol",
        "timestamp_utc",
        "session_date",
        "session_type",
        "open",
        "high",
        "low",
        "close",
    ]
    return pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True).sort_values(
        "timestamp_utc", kind="mergesort"
    )


def _publish_partition(
    base_path: Path,
    raw_sidecar: pd.DataFrame,
    output_root: Path,
) -> dict[str, Any]:
    asset_dir = base_path.parent.parent.name
    year_dir = base_path.parent.name
    year = int(year_dir.split("=", 1)[1])
    base = pd.read_parquet(
        base_path,
        columns=["asset_id", "timestamp_utc", "decision_timestamp", "session_date"],
    )
    merged = base.merge(raw_sidecar, on="timestamp_utc", how="left", validate="one_to_one", indicator=True)
    if len(merged) != len(base) or not (merged["_merge"] == "both").all():
        missing = int((merged["_merge"] != "both").sum())
        raise FeatureAuthorityError(f"Raw repair coverage mismatch for {base_path}: {missing} rows")
    merged = merged.drop(columns=["_merge", "timestamp_utc", "session_date"])
    if merged.duplicated(["asset_id", "decision_timestamp"]).any():
        raise FeatureAuthorityError(f"Duplicate V2 keys for {base_path}")
    for feature in REPAIRED_FEATURES:
        source = pd.to_datetime(merged[f"{feature}{SOURCE_TIMESTAMP_SUFFIX}"], utc=True)
        decision = pd.to_datetime(merged["decision_timestamp"], utc=True)
        if (source.notna() & (source > decision)).any():
            raise FeatureAuthorityError(f"Future source timestamp for {feature} in {base_path}")
        merged[feature] = pd.to_numeric(merged[feature], errors="coerce").astype("float32")
    destination = output_root / "stock" / asset_dir / year_dir / "pit_repair.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.partial")
    merged.to_parquet(temporary, index=False, compression="zstd")
    check = pd.read_parquet(temporary, columns=["asset_id", "decision_timestamp"])
    if len(check) != len(base) or check.duplicated(["asset_id", "decision_timestamp"]).any():
        temporary.unlink(missing_ok=True)
        raise FeatureAuthorityError(f"Written sidecar validation failed for {destination}")
    os.replace(temporary, destination)
    return {
        "base_relative_path": base_path.relative_to(base_path.parents[3]).as_posix(),
        "base_bytes": base_path.stat().st_size,
        "base_sha256": _sha256(base_path),
        "relative_path": destination.relative_to(output_root).as_posix(),
        "row_count": len(merged),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def _build_asset(
    asset_dir: str,
    partition_values: list[str],
    base_value: str,
    raw_value: str,
    output_value: str,
    resume: bool,
) -> list[dict[str, Any]]:
    base = Path(base_value)
    raw = Path(raw_value)
    output = Path(output_value)
    asset_partitions = [Path(value) for value in partition_values]
    rows: list[dict[str, Any]] = []
    pending = []
    for base_path in asset_partitions:
        relative = base_path.relative_to(base)
        destination = output / relative.parent / "pit_repair.parquet"
        if resume and destination.is_file():
            metadata = pd.read_parquet(destination, columns=["asset_id", "decision_timestamp"])
            rows.append(
                {
                    "base_relative_path": base_path.relative_to(base).as_posix(),
                    "base_bytes": base_path.stat().st_size,
                    "base_sha256": _sha256(base_path),
                    "relative_path": destination.relative_to(output).as_posix(),
                    "row_count": len(metadata),
                    "bytes": destination.stat().st_size,
                    "sha256": _sha256(destination),
                }
            )
        else:
            pending.append(base_path)
    if not pending:
        return rows
    years = [int(path.parent.name.split("=", 1)[1]) for path in asset_partitions]
    symbol = asset_dir.split("=", 1)[1]
    if not (raw / f"symbol={symbol}").is_dir():
        symbols = (
            pd.read_parquet(pending[0], columns=["canonical_symbol"])["canonical_symbol"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if len(symbols) != 1:
            raise FeatureAuthorityError(f"Cannot resolve raw symbol for {asset_dir}: {symbols}")
        symbol = symbols[0]
    raw_frame = _read_raw_context(raw, symbol, years)
    repaired = compute_repaired_session_features(raw_frame)
    repair_columns = [
        *REPAIRED_FEATURES,
        *(f"{name}{SOURCE_TIMESTAMP_SUFFIX}" for name in REPAIRED_FEATURES),
    ]
    raw_sidecar = pd.concat(
        [
            raw_frame[["timestamp_utc"]].reset_index(drop=True),
            repaired[repair_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    for base_path in pending:
        rows.append(_publish_partition(base_path, raw_sidecar, output))
    return rows


def build(
    *, maximum_partitions: int | None = None, resume: bool = True, workers: int = 2
) -> dict[str, Any]:
    base, raw, output = _feature_paths()
    gate = estimate()
    if not gate["resource_gate_passed"]:
        raise FeatureAuthorityError(f"Resource gate failed: {gate}")
    if workers < 1 or workers > 4:
        raise FeatureAuthorityError("workers must be between 1 and 4")
    output.mkdir(parents=True, exist_ok=True)
    partitions = _partitions(base)
    if maximum_partitions is not None:
        partitions = partitions[:maximum_partitions]
    partitions_by_asset: dict[str, list[Path]] = {}
    for path in partitions:
        partitions_by_asset.setdefault(path.parent.parent.name, []).append(path)
    jobs = [
        (
            asset_dir,
            [str(path) for path in asset_partitions],
            str(base),
            str(raw),
            str(output),
            resume,
        )
        for asset_dir, asset_partitions in sorted(partitions_by_asset.items())
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        grouped_rows = executor.map(_build_asset_star, jobs)
        rows = [row for group in grouped_rows for row in group]
    rows.sort(key=lambda row: row["base_relative_path"])
    all_partitions = _partitions(base)
    manifest: dict[str, Any] = {
        "authority_id": "CANONICAL_5M_FEATURE_AUTHORITY_FULL_V2_PIT_REPAIRED",
        "run_id": "DS24_CLEAN_V2_FEATURE_REPAIR_R1_20260926",
        "base_authority_identity": load_contract("feature_authority.json")["base_authority"]["authority_identity"],
        "repair_columns": list(REPAIRED_FEATURES),
        "partition_count": len(rows),
        "row_count": sum(int(row["row_count"]) for row in rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "complete": maximum_partitions is None and len(rows) == len(all_partitions),
        "base_snapshot": {
            "partition_count": len(rows),
            "row_count": sum(int(row["row_count"]) for row in rows),
            "total_bytes": sum(int(row["base_bytes"]) for row in rows),
            "aggregate_partition_sha256": stable_hash(
                [
                    {
                        "path": row["base_relative_path"],
                        "sha256": row["base_sha256"],
                        "rows": row["row_count"],
                    }
                    for row in rows
                ]
            ),
        },
        "partitions": rows,
    }
    if maximum_partitions is None:
        expected_rows = int(
            load_contract("feature_authority.json")["base_authority"]["physical_snapshot_row_count"]
        )
        if manifest["row_count"] != expected_rows:
            raise FeatureAuthorityError(
                f"Physical base population changed during build: expected {expected_rows}, "
                f"observed {manifest['row_count']}"
            )
    manifest["logical_sha256"] = stable_hash(manifest)
    encoded_manifest = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    published_manifest = output / "authority_manifest.json"
    if published_manifest.is_file():
        existing = json.loads(published_manifest.read_text(encoding="utf-8"))
        if existing.get("complete") is True:
            if existing != manifest:
                raise FeatureAuthorityError(
                    "Refusing to mutate the published complete V2 feature authority manifest"
                )
            return manifest
    temporary_manifest = output / "authority_manifest.json.partial"
    temporary_manifest.write_text(encoded_manifest, encoding="utf-8")
    os.replace(temporary_manifest, published_manifest)
    return manifest


def _build_asset_star(args: tuple[str, list[str], str, str, str, bool]) -> list[dict[str, Any]]:
    return _build_asset(*args)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the immutable DS24 clean-V2 repair sidecar.")
    parser.add_argument("--build", action="store_true", help="Materialize sidecar partitions; default is estimate only.")
    parser.add_argument("--maximum-partitions", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    result = (
        build(
            maximum_partitions=args.maximum_partitions,
            resume=not args.no_resume,
            workers=args.workers,
        )
        if args.build
        else estimate()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
