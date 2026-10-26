"""Filesystem and Parquet adapters for the SIP-to-IEX cutover audit."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import psutil
import pyarrow.parquet as pq


@dataclass(frozen=True)
class DirectoryStats:
    path: Path
    bytes: int
    files: int
    minimum_mtime_ns: int | None
    maximum_mtime_ns: int | None


@dataclass(frozen=True)
class AuditSample:
    frame: pd.DataFrame
    symbols: tuple[str, ...]
    years: tuple[int, ...]
    feature_files: tuple[str, ...]
    target_files: tuple[str, ...]
    context_files: tuple[str, ...]
    sidecar_files: tuple[str, ...]


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

def directory_stats(path: Path) -> DirectoryStats:
    """Scan a directory once without retaining a potentially huge file list."""

    total_bytes = 0
    files = 0
    minimum_mtime_ns: int | None = None
    maximum_mtime_ns: int | None = None
    stack = [Path(path)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            stat = entry.stat(follow_symlinks=False)
                            total_bytes += int(stat.st_size)
                            files += 1
                            minimum_mtime_ns = stat.st_mtime_ns if minimum_mtime_ns is None else min(minimum_mtime_ns, stat.st_mtime_ns)
                            maximum_mtime_ns = stat.st_mtime_ns if maximum_mtime_ns is None else max(maximum_mtime_ns, stat.st_mtime_ns)
                    except FileNotFoundError:
                        # A concurrently published unrelated artifact is excluded
                        # from this point-in-time scan and will be visible on retry.
                        continue
        except FileNotFoundError:
            continue
    return DirectoryStats(Path(path), total_bytes, files, minimum_mtime_ns, maximum_mtime_ns)


def resource_state(*, repository_root: Path) -> dict[str, Any]:
    disk = psutil.disk_usage(str(repository_root.anchor or repository_root))
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    processes: list[dict[str, Any]] = []
    relevant_terms = ("python", "ds24", "ds26", "sip", "alpaca", "paper", "parity", "capture")
    for process in psutil.process_iter(["pid", "ppid", "name", "cmdline", "create_time", "memory_info"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
            name = str(process.info.get("name") or "")
            if not any(term in f"{name} {command}".lower() for term in relevant_terms):
                continue
            memory_info = process.info.get("memory_info")
            processes.append(
                {
                    "pid": int(process.info["pid"]),
                    "parent_pid": int(process.info.get("ppid") or 0),
                    "name": name,
                    "command_line": command,
                    "created_epoch_seconds": process.info.get("create_time"),
                    "resident_bytes": int(memory_info.rss) if memory_info else None,
                    "virtual_bytes": int(memory_info.vms) if memory_info else None,
                }
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    processes.sort(key=lambda item: (item["pid"], item["command_line"]))
    return {
        "schema_version": "hf_data_cutover_r1.starting_resource_state.v1",
        "repository_root": str(repository_root.resolve()),
        "disk": {"volume": repository_root.anchor, "total_bytes": int(disk.total), "free_bytes": int(disk.free), "used_bytes": int(disk.used)},
        "physical_memory": {"total_bytes": int(memory.total), "available_bytes": int(memory.available)},
        "commit_proxy": {
            "physical_plus_pagefile_bytes": int(memory.total + swap.total),
            "available_physical_plus_free_pagefile_bytes": int(memory.available + swap.free),
            "note": "Cross-platform psutil proxy; Windows committed-byte authority is added by the launcher when available.",
        },
        "pagefile": {"total_bytes": int(swap.total), "used_bytes": int(swap.used), "free_bytes": int(swap.free), "percent": float(swap.percent)},
        "relevant_processes": processes,
    }


def scheduled_task_details(task_name: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["schtasks", "/Query", "/TN", task_name, "/FO", "LIST", "/V"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "task_name": task_name,
        "query_return_code": completed.returncode,
        "present": completed.returncode == 0,
        "query_output": completed.stdout.strip(),
        "query_error": completed.stderr.strip(),
    }


def parquet_schema(path: Path) -> dict[str, str]:
    schema = pq.ParquetFile(path).schema_arrow
    return {field.name: str(field.type) for field in schema}


def deterministic_symbols(stock_root: Path, *, count: int) -> tuple[str, ...]:
    symbols = sorted(
        path.name.removeprefix("asset=")
        for path in stock_root.glob("asset=*")
        if path.is_dir()
    )
    ranked = sorted(symbols, key=lambda symbol: (hashlib.sha256(symbol.encode("utf-8")).hexdigest(), symbol))
    return tuple(sorted(ranked[:count]))


def _read_existing_parquet(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=list(columns))
    available = set(pq.ParquetFile(path).schema_arrow.names)
    selected = [column for column in columns if column in available]
    # ParquetFile performs an exact file read.  ``pq.read_table`` promotes the
    # file to a Hive dataset and can collide with same-named in-file fields
    # such as target_id under ``target_id=<value>`` directories.
    return pq.ParquetFile(path).read(columns=selected).to_pandas()


def load_feature_target_sample(
    *,
    feature_root: Path,
    repair_root: Path,
    target_root: Path,
    predictor_names: Sequence[str],
    repaired_names: Sequence[str],
    years: Sequence[int] = (2022, 2023, 2024),
    symbol_count: int = 48,
    decision_minute_modulus: int = 30,
) -> AuditSample:
    """Load a deterministic, aligned development-only diagnostic panel.

    The sample rule is target blind: symbol hashes, fixed calendar years, and
    wall-clock decision minute are frozen before any target values are read.
    """

    stock_root = feature_root / "stock"
    symbols = deterministic_symbols(stock_root, count=symbol_count)
    stock_predictors = list(predictor_names[:72])
    context_predictors = list(predictor_names[72:])
    feature_frames: list[pd.DataFrame] = []
    feature_files: list[str] = []
    sidecar_frames: list[pd.DataFrame] = []
    sidecar_files: list[str] = []
    for symbol in symbols:
        for year in years:
            feature_path = stock_root / f"asset={symbol}" / f"year={year}" / "features.parquet"
            if not feature_path.is_file():
                continue
            columns = ["asset_id", "canonical_symbol", "decision_timestamp", "session_date", *stock_predictors]
            frame = _read_existing_parquet(feature_path, columns)
            timestamp = pd.to_datetime(frame["decision_timestamp"], utc=True, errors="coerce")
            frame = frame[timestamp.dt.minute.mod(decision_minute_modulus).eq(0)].copy()
            frame["decision_timestamp"] = pd.to_datetime(frame["decision_timestamp"], utc=True)
            feature_frames.append(frame)
            feature_files.append(str(feature_path.resolve()))
            repair_path = repair_root / "stock" / f"asset={symbol}" / f"year={year}" / "pit_repair.parquet"
            if repair_path.is_file():
                repair = _read_existing_parquet(repair_path, ["asset_id", "decision_timestamp", *repaired_names])
                repair_timestamp = pd.to_datetime(repair["decision_timestamp"], utc=True, errors="coerce")
                repair = repair[repair_timestamp.dt.minute.mod(decision_minute_modulus).eq(0)].copy()
                repair["decision_timestamp"] = pd.to_datetime(repair["decision_timestamp"], utc=True)
                sidecar_frames.append(repair)
                sidecar_files.append(str(repair_path.resolve()))
    if not feature_frames:
        raise FileNotFoundError("No feature partitions matched the frozen audit sample")
    features = pd.concat(feature_frames, ignore_index=True)
    if sidecar_frames:
        repair = pd.concat(sidecar_frames, ignore_index=True)
        features = features.merge(repair, on=["asset_id", "decision_timestamp"], how="left", suffixes=("", "__repair"), validate="one_to_one")
        for name in repaired_names:
            repaired_column = f"{name}__repair"
            if repaired_column in features:
                features[name] = features[repaired_column]
                features.drop(columns=[repaired_column], inplace=True)

    context_frames: list[pd.DataFrame] = []
    context_files: list[str] = []
    for year in years:
        context_path = feature_root / "shared-context" / f"year={year}" / "context.parquet"
        if not context_path.is_file():
            continue
        context = _read_existing_parquet(context_path, ["decision_timestamp", *context_predictors])
        timestamp = pd.to_datetime(context["decision_timestamp"], utc=True, errors="coerce")
        context = context[timestamp.dt.minute.mod(decision_minute_modulus).eq(0)].copy()
        context["decision_timestamp"] = pd.to_datetime(context["decision_timestamp"], utc=True)
        context_frames.append(context)
        context_files.append(str(context_path.resolve()))
    if context_frames:
        context = pd.concat(context_frames, ignore_index=True).drop_duplicates("decision_timestamp", keep="last")
        features = features.merge(context, on="decision_timestamp", how="left", validate="many_to_one")

    target_files: list[str] = []
    target_frames: list[pd.DataFrame] = []
    target_run = target_root / "run=ticket_71b_60m_20260730T170751Z" / "target_id=forward_return_60m__decision_5m"
    for symbol in symbols:
        for year in years:
            target_path = target_run / f"symbol={symbol}" / f"year={year}" / "target_rows.parquet"
            if not target_path.is_file():
                continue
            target = _read_existing_parquet(
                target_path,
                ["asset_id", "decision_timestamp", "target_value", "target_available_timestamp", "target_is_mature", "target_is_trainable"],
            )
            target["decision_timestamp"] = pd.to_datetime(target["decision_timestamp"], utc=True, errors="coerce")
            target = target[target["decision_timestamp"].dt.minute.mod(decision_minute_modulus).eq(0)].copy()
            target = target[target["target_is_mature"].eq(True) & target["target_is_trainable"].eq(True)]
            target_frames.append(target)
            target_files.append(str(target_path.resolve()))
    if not target_frames:
        raise FileNotFoundError("No target partitions matched the frozen audit sample")
    targets = pd.concat(target_frames, ignore_index=True)
    panel = features.merge(targets, on=["asset_id", "decision_timestamp"], how="inner", validate="one_to_one")
    panel = panel.sort_values(["decision_timestamp", "asset_id"], kind="mergesort").reset_index(drop=True)
    return AuditSample(
        frame=panel,
        symbols=symbols,
        years=tuple(int(year) for year in years),
        feature_files=tuple(feature_files),
        target_files=tuple(target_files),
        context_files=tuple(context_files),
        sidecar_files=tuple(sidecar_files),
    )


def hash_paths(paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(paths):
        path = Path(value)
        digest.update(str(path).encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(str(path.stat().st_mtime_ns).encode("ascii"))
    return digest.hexdigest()
