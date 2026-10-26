"""Build and publish the frozen IEX Feature Bank V1 research authority."""

from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import numpy as np

from application.services.hf_data_cutover_service import CutoverAuditError, CutoverPaths, atomic_json, utc_now
from core.research.iex_feature_builders import (
    CROSS_SECTIONAL_FEATURES,
    CROSS_TIMEFRAME_FEATURES,
    DAILY_FEATURES,
    FIVE_MINUTE_FEATURES,
    HOURLY_FEATURES,
    MARKET_STATE_FEATURES,
    build_cross_sectional_and_market_state,
    build_cross_timeframe_features,
    build_daily_features,
    build_five_minute_features,
    build_hourly_features,
)
from core.research.iex_dataset_builders import regular_session_bars
from infrastructure.data.iex_market_data_store import sha256_file
from infrastructure.data.iex_research_artifact_store import IexResearchArtifactStore


FEATURE_WORKERS = 3
MINIMUM_FINAL_RESERVE_BYTES = 15 * 1024**3


def run_iex_feature_build(paths: CutoverPaths) -> dict[str, Any]:
    run_manifest_path = paths.output_root / "RUN_MANIFEST.json"
    run_manifest = _read_json(run_manifest_path)
    if run_manifest.get("state") not in {
        "IEX_1H_COMPLETE_FEATURE_BUILD_PENDING",
        "IEX_FEATURE_BUILD_IN_PROGRESS",
    }:
        raise CutoverAuditError(f"IEX feature build requires complete 1h data, found {run_manifest.get('state')}")
    specification_path = paths.output_root / "IEX_FEATURE_BANK_V1_SPEC.json"
    specification = _read_json(specification_path)
    _validate_frozen_specification(specification)
    roster = tuple(_read_json(paths.output_root / "IEX_ACQUISITION_ROSTER.json")["symbols"])
    run_manifest.update(
        {
            "updated_at_utc": utc_now(),
            "state": "IEX_FEATURE_BUILD_IN_PROGRESS",
            "next_gate": "Publish and validate all frozen IEX Feature Bank V1 tables.",
        }
    )
    atomic_json(run_manifest_path, run_manifest)
    data_root = paths.repository_root / "data/market/iex"
    store = IexResearchArtifactStore(data_root)
    raw_index = _raw_manifest_index(data_root / "raw/5m")
    daily_index = _single_manifest_index(data_root / "derived/1d_adjusted")
    hourly_index = _raw_manifest_index(data_root / "derived/1h")
    spy_raw = _read_manifest_frames(raw_index["SPY"])
    qqq_raw = _read_manifest_frames(raw_index["QQQ"])
    spy_daily = _read_manifest_frames(daily_index["SPY"])
    spy_hourly = _read_manifest_frames(hourly_index["SPY"])
    spec_sha256 = sha256_file(specification_path)
    completed: list[dict[str, Any]] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=FEATURE_WORKERS, thread_name_prefix="iex-features") as executor:
        futures: dict[Future[list[dict[str, Any]]], str] = {
            executor.submit(
                _build_symbol_base,
                symbol,
                store=store,
                raw_manifests=raw_index[symbol],
                daily_manifests=daily_index[symbol],
                hourly_manifests=hourly_index[symbol],
                spy_raw=spy_raw,
                qqq_raw=qqq_raw,
                spy_daily=spy_daily,
                spy_hourly=spy_hourly,
                spec_sha256=spec_sha256,
            ): symbol
            for symbol in roster
        }
        for index, future in enumerate(as_completed(futures), start=1):
            completed.extend(future.result())
            if index % 25 == 0:
                elapsed = max(time.monotonic() - started, 1.0)
                print(
                    "IEX FEATURE BASE "
                    f"symbols={index}/{len(roster)} symbols_per_hour={index / elapsed * 3600:.1f} "
                    f"free_bytes={shutil.disk_usage(data_root).free}",
                    flush=True,
                )
            if shutil.disk_usage(data_root).free <= MINIMUM_FINAL_RESERVE_BYTES:
                for pending in futures:
                    pending.cancel()
                raise CutoverAuditError("IEX feature build stopped at the 15 GiB reserve gate")
    cross_completed = _build_cross_sectional_partitions(
        store=store,
        data_root=data_root,
        years=range(2021, 2027),
        spec_sha256=spec_sha256,
    )
    completed.extend(cross_completed)
    cross_timeframe_completed = _build_cross_timeframe_partitions(
        store=store,
        data_root=data_root,
        raw_index=raw_index,
        roster=roster,
        years=range(2021, 2027),
        spec_sha256=spec_sha256,
    )
    completed.extend(cross_timeframe_completed)
    manifest = _feature_manifest(
        paths=paths,
        spec_sha256=spec_sha256,
        completed=completed,
        symbol_count=len(roster),
    )
    output_path = paths.output_root / "IEX_FEATURE_V1_MANIFEST.json"
    atomic_json(output_path, manifest)
    run_manifest = _read_json(run_manifest_path)
    run_manifest.update(
        {
            "updated_at_utc": utc_now(),
            "state": "IEX_FEATURE_V1_COMPLETE_TARGET_BUILD_PENDING",
            "iex_feature_v1": {
                "manifest_sha256": sha256_file(output_path),
                "rows_across_tables": manifest["rows_across_tables"],
                "predictor_count": manifest["predictor_count"],
            },
            "next_gate": "Build physically separate H30/H60 targets.",
        }
    )
    atomic_json(run_manifest_path, run_manifest)
    return run_manifest


def _build_symbol_base(
    symbol: str,
    *,
    store: IexResearchArtifactStore,
    raw_manifests: tuple[Path, ...],
    daily_manifests: tuple[Path, ...],
    hourly_manifests: tuple[Path, ...],
    spy_raw: pd.DataFrame,
    qqq_raw: pd.DataFrame,
    spy_daily: pd.DataFrame,
    spy_hourly: pd.DataFrame,
    spec_sha256: str,
) -> list[dict[str, Any]]:
    raw = _read_manifest_frames(raw_manifests)
    daily = _read_manifest_frames(daily_manifests)
    hourly = _read_manifest_frames(hourly_manifests)
    raw_hashes = tuple(_read_json(path)["parquet_sha256"] for path in raw_manifests)
    daily_hashes = tuple(_read_json(path)["parquet_sha256"] for path in daily_manifests)
    hourly_hashes = tuple(_read_json(path)["parquet_sha256"] for path in hourly_manifests)
    five_features = build_five_minute_features(raw, raw_spy=spy_raw, raw_qqq=qqq_raw)
    daily_features = build_daily_features(daily, raw_spy_daily=spy_daily)
    hourly_features = build_hourly_features(hourly, hourly_spy=spy_hourly)
    results: list[dict[str, Any]] = []
    for year in range(2021, 2027):
        five_year = five_features.loc[five_features["timestamp"].dt.year.eq(year)].reset_index(drop=True)
        results.append(
            _publish_features(
                store,
                Path("features/v1/5m") / f"symbol={symbol}" / f"year={year}",
                five_year,
                table="5m",
                symbol=symbol,
                year=year,
                source_hashes=raw_hashes,
                spec_sha256=spec_sha256,
            )
        )
        hourly_year = hourly_features.loc[hourly_features["timestamp"].dt.year.eq(year)].reset_index(drop=True)
        results.append(
            _publish_features(
                store,
                Path("features/v1/1h") / f"symbol={symbol}" / f"year={year}",
                hourly_year,
                table="1h",
                symbol=symbol,
                year=year,
                source_hashes=hourly_hashes,
                spec_sha256=spec_sha256,
            )
        )
    results.append(
        _publish_features(
            store,
            Path("features/v1/1d") / f"symbol={symbol}",
            daily_features,
            table="1d",
            symbol=symbol,
            year=None,
            source_hashes=daily_hashes,
            spec_sha256=spec_sha256,
        )
    )
    return results


def _publish_features(
    store: IexResearchArtifactStore,
    relative: Path,
    frame: pd.DataFrame,
    *,
    table: str,
    symbol: str,
    year: int | None,
    source_hashes: tuple[str, ...],
    spec_sha256: str,
) -> dict[str, Any]:
    identity = {
        "dataset": "iex_feature_bank_v1",
        "table": table,
        "symbol": symbol,
        "year": year,
        "relative_directory": str(relative),
        "feature_spec_sha256": spec_sha256,
        "source_hashes": list(source_hashes),
        "builder_version": "iex_feature_bank_v1",
    }
    existing = store.completed(relative, identity)
    if existing is not None:
        return existing
    key_columns = ["symbol", "timestamp"] if "symbol" in frame else ["timestamp"]
    duplicates = int(frame.duplicated(key_columns).sum()) if len(frame) else 0
    if duplicates:
        raise ValueError(f"Feature partition {relative} has duplicate row identities")
    availability_column = "feature_available_at" if "feature_available_at" in frame else "decision_timestamp"
    availability_violations = (
        int((frame[availability_column] < frame["timestamp"]).sum()) if len(frame) and availability_column in frame else 0
    )
    if availability_violations:
        raise ValueError(f"Feature partition {relative} violates its availability contract")
    predictor_columns = [column for column in frame if column not in {"symbol", "timestamp", "session_date", "decision_timestamp", "feature_available_at"}]
    quality = {
        "row_count": int(len(frame)),
        "predictor_count": len(predictor_columns),
        "duplicate_count": duplicates,
        "availability_violation_count": availability_violations,
        "missing_rate_by_feature": {
            column: float(frame[column].isna().mean()) if len(frame) else None for column in predictor_columns
        },
        "nonfinite_rate_by_feature": {
            column: (
                float((~np.isfinite(pd.to_numeric(frame[column], errors="coerce")) & frame[column].notna()).mean())
                if len(frame)
                else None
            )
            for column in predictor_columns
        },
        "constant_nonnull_features": [
            column for column in predictor_columns if frame[column].dropna().nunique() <= 1 and frame[column].notna().any()
        ],
    }
    return store.publish(
        relative,
        frame,
        identity=identity,
        lineage={
            "feature_spec_sha256": spec_sha256,
            "source_parquet_sha256": list(source_hashes),
            "targets_read": False,
        },
        quality=quality,
    )


def _build_cross_sectional_partitions(
    *,
    store: IexResearchArtifactStore,
    data_root: Path,
    years: Iterable[int],
    spec_sha256: str,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    columns = [
        "symbol",
        "timestamp",
        "decision_timestamp",
        "log_return_6b",
        "log_return_12b",
        "ema_distance_24b",
        "beta_residual_return_60m",
        "realized_vol_24b",
        "relative_volume_same_slot_20d",
        "relative_trade_count_same_slot_20d",
    ]
    for year in years:
        paths = sorted((data_root / "features/v1/5m").glob(f"symbol=*/year={year}/part.parquet"))
        source_hashes = tuple(sha256_file(path) for path in paths)
        frames = [pd.read_parquet(path, columns=columns) for path in paths]
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        cross, state = build_cross_sectional_and_market_state(combined)
        for month in range(1, 13):
            cross_month = cross.loc[cross["timestamp"].dt.month.eq(month)].reset_index(drop=True)
            state_month = state.loc[state["timestamp"].dt.month.eq(month)].reset_index(drop=True)
            outputs.append(
                _publish_features(
                    store,
                    Path("features/v1/cross_sectional") / f"year={year}" / f"month={month:02d}",
                    cross_month,
                    table="cross_sectional",
                    symbol="ALL_AVAILABLE",
                    year=year,
                    source_hashes=source_hashes,
                    spec_sha256=spec_sha256,
                )
            )
            outputs.append(
                _publish_features(
                    store,
                    Path("features/v1/market_state") / f"year={year}" / f"month={month:02d}",
                    state_month,
                    table="market_state",
                    symbol="COMMON_CONTEXT",
                    year=year,
                    source_hashes=source_hashes,
                    spec_sha256=spec_sha256,
                )
            )
    return outputs


def _build_cross_timeframe_partitions(
    *,
    store: IexResearchArtifactStore,
    data_root: Path,
    raw_index: Mapping[str, tuple[Path, ...]],
    roster: tuple[str, ...],
    years: Iterable[int],
    spec_sha256: str,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    five_root = data_root / "features/v1/5m"
    daily_root = data_root / "features/v1/1d"
    hourly_root = data_root / "features/v1/1h"
    for year in years:
        spy_path = five_root / "symbol=SPY" / f"year={year}" / "part.parquet"
        spy = pd.read_parquet(spy_path, columns=["timestamp", "log_return_1b"]).rename(
            columns={"log_return_1b": "spy_log_return_1b"}
        )
        candidate_frames: list[pd.DataFrame] = []
        source_paths: list[Path] = [spy_path]
        for symbol in roster:
            five_path = five_root / f"symbol={symbol}" / f"year={year}" / "part.parquet"
            daily_path = daily_root / f"symbol={symbol}" / "part.parquet"
            hourly_paths = [
                hourly_root / f"symbol={symbol}" / f"year={candidate_year}" / "part.parquet"
                for candidate_year in (year - 1, year)
                if candidate_year >= 2021
            ]
            raw_paths = [
                manifest_path
                for manifest_path in raw_index[symbol]
                if int(manifest_path.parent.name.split("=", 1)[1]) in {year - 1, year}
            ]
            source_paths.extend([five_path, daily_path, *hourly_paths, *raw_paths])
            base = pd.read_parquet(
                five_path,
                columns=[
                    "symbol",
                    "timestamp",
                    "decision_timestamp",
                    "log_return_1b",
                    "log_return_6b",
                    "rolling_beta_spy_20d",
                ],
            )
            if base.empty:
                continue
            raw = _read_manifest_frames(raw_paths)
            rth = regular_session_bars(raw)
            session_returns = rth.groupby("session_date", sort=True, observed=True).apply(
                lambda group: float(np.log(group["close"].iloc[-1] / group["open"].iloc[0])),
                include_groups=False,
            )
            prior_five = session_returns.rolling(5, min_periods=5).sum().shift(1)
            base["session_date"] = base["timestamp"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
            base["sip01_prior_5_session_intraday_trend"] = base["session_date"].map(prior_five)
            rth["sip01_current_30m_pullback"] = rth.groupby(
                "session_date", sort=False, observed=True
            )["close"].transform(lambda values: np.log(values) - np.log(values).shift(6))
            base = base.merge(
                rth[["timestamp", "sip01_current_30m_pullback"]],
                on="timestamp",
                how="left",
                validate="one_to_one",
            )
            base = base.merge(spy, on="timestamp", how="left", validate="many_to_one")
            base["beta_residual_return_5m"] = (
                base["log_return_1b"]
                - base["rolling_beta_spy_20d"] * base["spy_log_return_1b"]
            )
            daily = pd.read_parquet(
                daily_path,
                columns=["feature_available_at", "daily_trend_63d"],
            ).sort_values("feature_available_at")
            base = pd.merge_asof(
                base.sort_values("decision_timestamp"),
                daily,
                left_on="decision_timestamp",
                right_on="feature_available_at",
                direction="backward",
                allow_exact_matches=True,
            )
            hourly_frames = [
                pd.read_parquet(path, columns=["feature_available_at", "hourly_trend_12h"])
                for path in hourly_paths
            ]
            hourly = pd.concat(hourly_frames, ignore_index=True).sort_values("feature_available_at")
            base = pd.merge_asof(
                base.sort_values("decision_timestamp"),
                hourly,
                left_on="decision_timestamp",
                right_on="feature_available_at",
                direction="backward",
                allow_exact_matches=True,
                suffixes=("_daily", "_hourly"),
            )
            candidate_frames.append(
                base[
                    [
                        "symbol",
                        "timestamp",
                        "decision_timestamp",
                        "sip01_prior_5_session_intraday_trend",
                        "sip01_current_30m_pullback",
                        "log_return_6b",
                        "daily_trend_63d",
                        "hourly_trend_12h",
                        "beta_residual_return_5m",
                    ]
                ]
            )
        candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
        cross_timeframe = build_cross_timeframe_features(candidates) if not candidates.empty else pd.DataFrame(
            columns=["symbol", "timestamp", "decision_timestamp", *CROSS_TIMEFRAME_FEATURES]
        )
        source_hashes = tuple(
            sorted(
                sha256_file(path) if path.name == "part.parquet" else sha256_file(Path(_read_json(path)["parquet_path"]))
                for path in set(source_paths)
                if path.exists()
            )
        )
        for month in range(1, 13):
            monthly = cross_timeframe.loc[cross_timeframe["timestamp"].dt.month.eq(month)].reset_index(drop=True)
            outputs.append(
                _publish_features(
                    store,
                    Path("features/v1/cross_timeframe") / f"year={year}" / f"month={month:02d}",
                    monthly,
                    table="cross_timeframe",
                    symbol="ALL_AVAILABLE",
                    year=year,
                    source_hashes=source_hashes,
                    spec_sha256=spec_sha256,
                )
            )
    return outputs


def _feature_manifest(
    *,
    paths: CutoverPaths,
    spec_sha256: str,
    completed: list[dict[str, Any]],
    symbol_count: int,
) -> dict[str, Any]:
    table_counts: dict[str, dict[str, int]] = {}
    for row in completed:
        table = str(row["identity"]["table"])
        aggregate = table_counts.setdefault(table, {"partitions": 0, "rows": 0, "bytes": 0})
        aggregate["partitions"] += 1
        aggregate["rows"] += int(row["row_count"])
        aggregate["bytes"] += int(row["parquet_bytes"])
    return {
        "schema_version": "hf_data_cutover_r1.iex_feature_v1_manifest.v1",
        "created_at_utc": utc_now(),
        "research_authority_only": True,
        "feature_spec_sha256": spec_sha256,
        "feature_spec_status": "FROZEN_BEFORE_IEX_TARGET_OUTCOME_ANALYSIS",
        "predictor_count": 80,
        "symbol_count": symbol_count,
        "tables": table_counts,
        "rows_across_tables": sum(row["rows"] for row in table_counts.values()),
        "parquet_bytes": sum(row["bytes"] for row in table_counts.values()),
        "targets_read_by_feature_pipeline": False,
        "raw_ohlcv_duplicated_in_feature_tables": False,
        "daily_hourly_context_repeated_in_5m_rows": False,
        "missing_tables": [],
        "status": "COMPLETE",
    }


def _validate_frozen_specification(specification: Mapping[str, Any]) -> None:
    if specification.get("status") != "FROZEN_BEFORE_IEX_TARGET_OUTCOME_ANALYSIS":
        raise CutoverAuditError("IEX feature specification is not frozen")
    grouped: dict[str, tuple[str, ...]] = {}
    for table in {str(row["table"]) for row in specification["features"]}:
        grouped[table] = tuple(row["name"] for row in specification["features"] if row["table"] == table)
    expected = {
        "5m": FIVE_MINUTE_FEATURES,
        "1d": DAILY_FEATURES,
        "1h": HOURLY_FEATURES,
        "cross_sectional": CROSS_SECTIONAL_FEATURES,
        "market_state": MARKET_STATE_FEATURES,
        "cross_timeframe": CROSS_TIMEFRAME_FEATURES,
    }
    mismatches = {table: {"spec": grouped.get(table), "implementation": names} for table, names in expected.items() if grouped.get(table) != names}
    if mismatches:
        raise CutoverAuditError(f"Frozen IEX feature specification/implementation mismatch: {mismatches}")


def _raw_manifest_index(root: Path) -> dict[str, tuple[Path, ...]]:
    index: dict[str, list[Path]] = {}
    for path in sorted(root.glob("symbol=*/year=*/manifest.json")):
        symbol = path.parent.parent.name.split("=", 1)[1]
        index.setdefault(symbol, []).append(path)
    return {symbol: tuple(paths) for symbol, paths in index.items()}


def _single_manifest_index(root: Path) -> dict[str, tuple[Path, ...]]:
    return {
        path.parent.name.split("=", 1)[1]: (path,)
        for path in sorted(root.glob("symbol=*/manifest.json"))
    }


def _read_manifest_frames(manifest_paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for manifest_path in manifest_paths:
        manifest = _read_json(manifest_path)
        parquet_path = Path(manifest["parquet_path"])
        if sha256_file(parquet_path) != manifest["parquet_sha256"]:
            raise ValueError(f"Source hash mismatch during feature build: {parquet_path}")
        frames.append(pd.read_parquet(parquet_path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["run_iex_feature_build"]
