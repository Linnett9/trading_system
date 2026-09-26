from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.clean_v2_contracts import stable_hash
from core.research.ml.five_minute_target_dataset import build_five_minute_target_dataset


SOURCE_ROOT = ROOT / "data/processed/alpaca/symbol_bars/sip/5m"
BASE_ROOT = ROOT / "data/processed/ml_targets/five_minute/version=five_minute_targets_v1/run=ticket_71b_60m_20260730T170751Z"
OUTPUT_ROOT = ROOT / "data/processed/ml_targets/five_minute/version=five_minute_targets_clean_v2/run=DS24_CLEAN_V2_TARGET_DELTA_R1_20260926"
TARGET_ID = "forward_return_60m__decision_5m"
START_DATE = "2026-07-01"
END_DATE = "2026-09-23"
SOURCE_CUTOFF = "2026-09-23T20:00:00Z"


class TargetDeltaError(ValueError):
    """Raised when the clean target extension cannot be published safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _symbols() -> list[str]:
    return sorted(path.name.split("=", 1)[1] for path in SOURCE_ROOT.glob("symbol=*") if path.is_dir())


def _destination(symbol: str) -> Path:
    return OUTPUT_ROOT / f"target_id={TARGET_ID}" / f"symbol={symbol}" / "year=2026"


def _base_snapshot() -> dict[str, Any]:
    partitions = sorted(
        (BASE_ROOT / f"target_id={TARGET_ID}").glob("symbol=*/year=*/target_rows.parquet")
    )
    if not partitions:
        raise TargetDeltaError(f"Target base has no partitions: {BASE_ROOT}")
    rows = [
        {
            "base_relative_path": path.relative_to(BASE_ROOT).as_posix(),
            "base_bytes": path.stat().st_size,
            "base_sha256": _sha256(path),
        }
        for path in partitions
    ]
    return {
        "partition_count": len(rows),
        "total_bytes": sum(int(row["base_bytes"]) for row in rows),
        "aggregate_partition_sha256": stable_hash(rows),
        "partitions": rows,
    }


def _validate_partition(symbol: str, path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(
        path,
        columns=[
            "asset_id",
            "decision_timestamp",
            "target_available_timestamp",
            "target_is_trainable",
            "target_id",
            "target_code_hash",
        ],
    )
    if frame.empty:
        return {
            "symbol": symbol,
            "relative_path": path.relative_to(OUTPUT_ROOT).as_posix(),
            "rows": 0,
            "trainable_rows": 0,
            "minimum_decision_timestamp": None,
            "maximum_decision_timestamp": None,
            "maximum_trainable_target_available_timestamp": None,
            "target_code_hashes": [],
            "empty_source_window": True,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    if frame.duplicated(["asset_id", "decision_timestamp", "target_id"]).any():
        raise TargetDeltaError(f"Duplicate target keys for {symbol}")
    if set(frame["target_id"].astype(str)) != {TARGET_ID}:
        raise TargetDeltaError(f"Wrong target identity for {symbol}")
    decisions = pd.to_datetime(frame["decision_timestamp"], utc=True)
    if decisions.dt.date.min().isoformat() < START_DATE or decisions.dt.date.max().isoformat() > END_DATE:
        raise TargetDeltaError(f"Target chronology outside frozen delta for {symbol}")
    maturities = pd.to_datetime(frame.loc[frame["target_is_trainable"], "target_available_timestamp"], utc=True)
    if len(maturities) and maturities.max() > pd.Timestamp(SOURCE_CUTOFF):
        raise TargetDeltaError(f"Trainable target after source cutoff for {symbol}")
    return {
        "symbol": symbol,
        "relative_path": path.relative_to(OUTPUT_ROOT).as_posix(),
        "rows": len(frame),
        "trainable_rows": int(frame["target_is_trainable"].sum()),
        "minimum_decision_timestamp": decisions.min().isoformat(),
        "maximum_decision_timestamp": decisions.max().isoformat(),
        "maximum_trainable_target_available_timestamp": maturities.max().isoformat() if len(maturities) else None,
        "target_code_hashes": sorted(frame["target_code_hash"].astype(str).unique()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _build_symbol(symbol: str, resume: bool) -> dict[str, Any]:
    destination = _destination(symbol)
    target_path = destination / "target_rows.parquet"
    if not (resume and target_path.is_file()):
        build_five_minute_target_dataset(
            source_root=SOURCE_ROOT,
            symbols=[symbol],
            start_date=START_DATE,
            end_date=END_DATE,
            target_ids=[TARGET_ID],
            output_root=destination,
            source_cutoff=SOURCE_CUTOFF,
            training_cutoff=SOURCE_CUTOFF,
        )
    return _validate_partition(symbol, target_path)


def build(
    *, maximum_symbols: int | None = None, resume: bool = True, workers: int = 2
) -> dict[str, Any]:
    symbols = _symbols()
    selected = symbols if maximum_symbols is None else symbols[:maximum_symbols]
    if workers < 1 or workers > 4:
        raise TargetDeltaError("workers must be between 1 and 4")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(_build_symbol, selected, [resume] * len(selected)))
    rows.sort(key=lambda row: row["symbol"])
    maximum_outcome = max(
        row["maximum_trainable_target_available_timestamp"]
        for row in rows
        if row["maximum_trainable_target_available_timestamp"] is not None
    )
    manifest: dict[str, Any] = {
        "authority_id": "DS24_CLEAN_V2_TARGET_AUTHORITY_V1",
        "run_id": "DS24_CLEAN_V2_TARGET_DELTA_R1_20260926",
        "target_id": TARGET_ID,
        "base_root": BASE_ROOT.relative_to(ROOT).as_posix(),
        "base_maximum_decision_timestamp": "2026-06-30T20:00:00Z",
        "delta_start_date": START_DATE,
        "delta_end_date": END_DATE,
        "source_cutoff": SOURCE_CUTOFF,
        "symbol_count": len(rows),
        "row_count": sum(int(row["rows"]) for row in rows),
        "trainable_row_count": sum(int(row["trainable_rows"]) for row in rows),
        "empty_symbol_count": sum(int(row["rows"] == 0) for row in rows),
        "empty_symbols": [row["symbol"] for row in rows if row["rows"] == 0],
        "maximum_historical_outcome_timestamp_consumed": maximum_outcome,
        "complete": maximum_symbols is None and len(rows) == len(symbols),
        "base_snapshot": _base_snapshot() if maximum_symbols is None else None,
        "partitions": rows,
    }
    manifest["logical_sha256"] = stable_hash(manifest)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / "authority_manifest.json"
    encoded_manifest = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("complete") is True:
            if existing != manifest:
                raise TargetDeltaError(
                    "Refusing to mutate the published complete clean target authority manifest"
                )
            return manifest
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(encoded_manifest, encoding="utf-8")
    os.replace(temporary, path)
    return manifest


def status() -> dict[str, Any]:
    expected = len(_symbols())
    published = list(
        OUTPUT_ROOT.glob(
            f"target_id={TARGET_ID}/symbol=*/year=2026/target_rows.parquet"
        )
    )
    manifest_path = OUTPUT_ROOT / "authority_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    return {
        "classification": (
            "DS24_CLEAN_V2_TARGET_DELTA_COMPLETE"
            if manifest.get("complete") is True and len(published) == expected
            else "DS24_CLEAN_V2_TARGET_DELTA_IN_PROGRESS"
        ),
        "published_symbols": len(published),
        "expected_symbols": expected,
        "manifest_complete": manifest.get("complete") is True,
        "manifest_logical_sha256": manifest.get("logical_sha256"),
        "row_count": manifest.get("row_count"),
        "trainable_row_count": manifest.get("trainable_row_count"),
        "output_root": OUTPUT_ROOT.relative_to(ROOT).as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the DS24 clean-V2 2026 target delta.")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--maximum-symbols", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.status:
        payload = status()
    elif not args.build:
        payload = {
            "symbol_count": len(_symbols()),
            "start_date": START_DATE,
            "end_date": END_DATE,
            "source_cutoff": SOURCE_CUTOFF,
            "base_exists": BASE_ROOT.is_dir(),
        }
    else:
        payload = build(
            maximum_symbols=args.maximum_symbols,
            resume=not args.no_resume,
            workers=args.workers,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
