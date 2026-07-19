from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from core.research.ml.stock_level.prediction_artifacts.math import (
    _trailing_drawdown,
    _trailing_liquidity_score,
    _trailing_return,
    _trailing_volatility,
)
from core.research.ml.stock_level_benchmark_data import (
    SELECTOR_ROW_ID_CONTRACT_VERSION,
    _stable_selector_row_id,
)
from core.research.ml.selector_dataset_lineage import logical_manifest_checksum

SELECTOR_DATASET_CONTRACT_VERSION = "canonical_v2_selector_dataset_v1"
SELECTOR_DATASET_MANIFEST_VERSION = "authoritative_frozen_selector_dataset_v2"
BASELINE_CONTRACT_VERSION = "stock_selector_trailing_signals_v1"
DETERMINISTIC_SIGNAL_COLUMNS = (
    "predicted_momentum_20d",
    "predicted_momentum_60d",
    "predicted_momentum_120d",
    "predicted_volatility_20d",
    "predicted_drawdown_60d",
    "predicted_liquidity_score",
    "predicted_risk_adjusted_momentum",
)
BASELINE_CANDIDATES = {
    "momentum_120d": "predicted_momentum_120d",
    "risk_adjusted_momentum": "predicted_risk_adjusted_momentum",
}


@dataclass(frozen=True)
class SelectorDatasetPaths:
    root: Path
    rows: Path
    baseline_scores: Path
    manifest: Path
    quality_report: Path


def read_selector_dataset_rows(root: Path) -> list[dict[str, Any]]:
    """Join immutable source rows to deterministic signals by stable row identity."""
    import pyarrow.parquet as pq

    rows = pq.read_table(root / "rows.parquet").to_pylist()
    scores = {
        str(row["row_id"]): row
        for row in pq.read_table(root / "baseline_scores.parquet").to_pylist()
    }
    output = []
    for row in rows:
        row_id = _stable_selector_row_id(str(row["asset_id"]), str(row["decision_timestamp"]))
        score = scores.get(row_id)
        if score is None:
            raise RuntimeError(f"Missing deterministic selector signals for row_id={row_id}")
        output.append({**row, **{name: score[name] for name in DETERMINISTIC_SIGNAL_COLUMNS}})
    if len(output) != len(scores):
        raise RuntimeError("Selector rows and baseline scores do not have identical populations")
    return output


def deterministic_baseline_scores(
    *, asset_id: str, decision_timestamp: str, decision_date: str,
    close_dates: list[str], close_values: list[float],
    dollar_volume_dates: list[str], dollar_volume_values: list[float],
) -> dict[str, Any]:
    m20 = _trailing_return(close_dates, close_values, decision_date, lookback=20)
    m60 = _trailing_return(close_dates, close_values, decision_date, lookback=60)
    m120 = _trailing_return(close_dates, close_values, decision_date, lookback=120)
    vol20 = _trailing_volatility(close_dates, close_values, decision_date, lookback=20)
    dd60 = _trailing_drawdown(close_dates, close_values, decision_date, lookback=60)
    liquidity = _trailing_liquidity_score(
        dollar_volume_dates, dollar_volume_values, decision_date, lookback=63
    )
    risk = max(abs(float(vol20 or 0.0)), abs(float(dd60 or 0.0)), 1e-6)
    return {
        "row_id": _stable_selector_row_id(asset_id, decision_timestamp),
        "asset_id": asset_id,
        "decision_timestamp": decision_timestamp,
        "baseline_contract_version": BASELINE_CONTRACT_VERSION,
        "predicted_momentum_20d": _nullable(m20),
        "predicted_momentum_60d": _nullable(m60),
        "predicted_momentum_120d": _nullable(m120),
        "predicted_volatility_20d": _nullable(vol20),
        "predicted_drawdown_60d": _nullable(dd60),
        "predicted_liquidity_score": _nullable(liquidity),
        "predicted_risk_adjusted_momentum": (
            None if m60 == "" else float(m60) / risk
        ),
    }


def build_frozen_selector_dataset(
    source_path: Path, market_root: Path, output_root: Path,
    *, symbols: Iterable[str] | None = None, decision_dates: Iterable[str] | None = None,
    copy_source_rows: bool = True, source_sha256: str | None = None,
    config_hash: str | None = None, daily_spine_manifest_path: Path | None = None,
    daily_feature_manifest_path: Path | None = None,
    symbol_registry_manifest_path: Path | None = None,
    base_artifact_path: Path | None = None,
    base_manifest_path: Path | None = None,
    enriched_manifest_path: Path | None = None,
) -> SelectorDatasetPaths:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    if daily_spine_manifest_path is None or daily_feature_manifest_path is None or symbol_registry_manifest_path is None:
        raise ValueError("Authoritative daily-spine, daily-feature, and symbol-registry parent manifests are required")
    if (
        base_artifact_path is None
        or base_manifest_path is None
        or enriched_manifest_path is None
    ):
        raise ValueError(
            "Frozen selector publication requires explicit base artifact, "
            "base manifest, and enriched manifest"
        )
    parents = _validate_parent_manifests(source_path, daily_spine_manifest_path, daily_feature_manifest_path, symbol_registry_manifest_path)
    final_root = output_root
    if final_root.exists():
        raise FileExistsError(f"Frozen selector dataset already exists: {final_root}")
    output_root = final_root.with_name(f".{final_root.name}.{uuid.uuid4().hex}.tmp")
    output_root.mkdir(parents=True, exist_ok=True)
    selected_symbols = sorted({str(x).upper() for x in symbols or ()})
    selected_dates = sorted({str(x) for x in decision_dates or ()})
    bounded = bool(selected_symbols or selected_dates)
    rows_path = output_root / "rows.parquet"
    baseline_path = output_root / "baseline_scores.parquet"
    source_dataset = ds.dataset(source_path, format="parquet")
    filt = None
    if selected_symbols:
        filt = ds.field("symbol").isin(selected_symbols)
    if selected_dates:
        date_filter = ds.field("decision_session_date").isin(selected_dates)
        filt = date_filter if filt is None else filt & date_filter
    source_table = source_dataset.to_table(filter=filt) if bounded else None
    finite_counts = {name: 0 for name in DETERMINISTIC_SIGNAL_COLUMNS}
    row_id_digests: set[bytes] = set()
    baseline_count = 0
    if bounded:
        source_table = source_table.append_column(
            "row_id",
            pa.array([
                _stable_selector_row_id(str(asset), str(timestamp))
                for asset, timestamp in zip(
                    source_table["asset_id"].to_pylist(),
                    source_table["decision_timestamp"].to_pylist(),
                )
            ]),
        )
        pq.write_table(source_table, rows_path, compression="zstd")
        identity = source_table.select(
            ["symbol", "asset_id", "decision_timestamp", "decision_session_date"]
        ).to_pylist()
        score_rows = _score_identity_rows(identity, market_root, ds)
        pq.write_table(pa.Table.from_pylist(score_rows), baseline_path, compression="zstd")
        for row in score_rows:
            row_id_digests.add(bytes.fromhex(row["row_id"]))
            for name in DETERMINISTIC_SIGNAL_COLUMNS:
                finite_counts[name] += row[name] is not None
        baseline_count = len(score_rows)
    else:
        identity = None
        source_file = pq.ParquetFile(source_path)
        writer = None
        rows_writer = None
        try:
            for index in range(source_file.num_row_groups):
                source_group = source_file.read_row_group(index)
                row_ids = [
                    _stable_selector_row_id(str(asset), str(timestamp))
                    for asset, timestamp in zip(
                        source_group["asset_id"].to_pylist(),
                        source_group["decision_timestamp"].to_pylist(),
                    )
                ]
                derived_group = source_group.append_column("row_id", pa.array(row_ids))
                if rows_writer is None:
                    rows_writer = pq.ParquetWriter(rows_path, derived_group.schema, compression="zstd")
                rows_writer.write_table(derived_group)
                identity_rows = source_group.select([
                    "symbol", "asset_id", "decision_timestamp", "decision_session_date"
                ]).to_pylist()
                score_rows = _score_identity_rows(identity_rows, market_root, ds)
                table = pa.Table.from_pylist(score_rows)
                if writer is None:
                    writer = pq.ParquetWriter(baseline_path, table.schema, compression="zstd")
                writer.write_table(table)
                baseline_count += len(score_rows)
                for row in score_rows:
                    row_id_digests.add(bytes.fromhex(row["row_id"]))
                    for name in DETERMINISTIC_SIGNAL_COLUMNS:
                        finite_counts[name] += row[name] is not None
        finally:
            if writer is not None:
                writer.close()
            if rows_writer is not None:
                rows_writer.close()
    source_count = source_dataset.count_rows()
    derivative_count = source_table.num_rows if bounded else baseline_count
    source_digest = source_sha256 or _sha256(source_path)
    feature_schema = {
        "contract_version": SELECTOR_DATASET_CONTRACT_VERSION,
        "deterministic_signal_columns": list(DETERMINISTIC_SIGNAL_COLUMNS),
        "availability_rule": "all price/volume observations have session_date < decision_session_date",
        "fitted_meta_features": [],
        "missingness_policy": "fail closed for model input; warmup nulls permitted only before eligibility",
    }
    target_columns = [name for name in source_dataset.schema.names if name.startswith("actual_")]
    target_schema = {
        "target_columns": target_columns,
        "primary_target": "actual_forward_return_10d",
        "economic_target_id": "forward_return_10d",
        "target_provenance_contract_version": "stock_level_target_provenance_v2",
        "target_registry_schema_version": "selector_target_identity.v1",
    }
    candidate_schema = {
        "fitted_models": ["ridge", "elastic_net", "random_forest", "gradient_boosting", "dlinear", "patchtst", "transformer", "itransformer", "momentum_transformer", "multitask_transformer", "market_context_encoder", "news_analysis_transformer", "temporal_fusion_transformer"],
        "non_ml_baselines": BASELINE_CANDIDATES,
    }
    _write_json(output_root / "feature_schema.json", feature_schema)
    _write_json(output_root / "target_schema.json", target_schema)
    _write_json(output_root / "candidate_schema.json", candidate_schema)
    checksums = {
        "rows.parquet": _sha256(rows_path),
        "baseline_scores.parquet": _sha256(baseline_path),
        "feature_schema.json": _sha256(output_root / "feature_schema.json"),
        "target_schema.json": _sha256(output_root / "target_schema.json"),
        "candidate_schema.json": _sha256(output_root / "candidate_schema.json"),
    }
    quality = {
        "source_row_count": source_count, "derivative_row_count": derivative_count,
        "row_count_preserved": derivative_count == source_count if not bounded else True,
        "baseline_row_count": baseline_count, "unique_row_ids": len(row_id_digests),
        "row_id_collisions": baseline_count - len(row_id_digests),
        "baseline_finite_counts": finite_counts,
        "bounded": bounded,
    }
    population = _dataset_population_identity(rows_path)
    _validate_rows_against_parents(rows_path, parents)
    from core.research.ml.registries import RegistryResolver, load_registry_bundle

    target = RegistryResolver(load_registry_bundle()).resolve(
        "target_contracts", "forward_return_10d", role="selector"
    )
    from core.research.ml.stock_level.selector_lineage import (
        preflight_frozen_selector_dataset,
    )

    base_rows = pq.read_table(base_artifact_path).to_pylist()
    enriched_rows = pq.read_table(rows_path).to_pylist()
    frozen_preflight = preflight_frozen_selector_dataset(
        daily_spine_manifest=json.loads(
            daily_spine_manifest_path.read_text(encoding="utf-8")
        ),
        base_manifest=json.loads(base_manifest_path.read_text(encoding="utf-8")),
        enriched_manifest=json.loads(
            enriched_manifest_path.read_text(encoding="utf-8")
        ),
        base_rows=base_rows,
        enriched_rows=enriched_rows,
        feature_columns=DETERMINISTIC_SIGNAL_COLUMNS,
    )
    if frozen_preflight["status"] != "READY":
        raise ValueError(
            f"Frozen selector preflight blocked: {frozen_preflight['blockers']}"
        )
    manifest = {
        "manifest_schema_version": SELECTOR_DATASET_MANIFEST_VERSION,
        "frozen_dataset_version": "v2",
        "dataset_id": SELECTOR_DATASET_CONTRACT_VERSION + ("_bounded" if bounded else ""),
        "dataset_path": str(final_root / "rows.parquet"), "dataset_checksum": checksums["rows.parquet"],
        "row_population_checksum": population["row_population_checksum"], "row_count": population["row_count"],
        "date_coverage": population["date_coverage"], "symbol_count": population["symbol_count"],
        "source_path": str(source_path), "source_sha256": source_digest,
        "source_row_count": source_count, "source_symbol_count": 406,
        "row_id_contract": SELECTOR_ROW_ID_CONTRACT_VERSION,
        "feature_contract": SELECTOR_DATASET_CONTRACT_VERSION,
        "baseline_contract": BASELINE_CONTRACT_VERSION,
        "economic_target_id": target.canonical_id,
        "target_provenance_contract_version": (
            "stock_level_target_provenance_v2"
        ),
        "target_registry_schema_version": "selector_target_identity.v1",
        "target_registry_entry_checksum": target.entry.entry_hash,
        "ranking_contract": "daily_cross_sectional_ranking_problem_v1",
        "daily_stock_spine_identity": parents["daily_spine_identity"],
        "daily_stock_spine_version": parents["daily_spine_version"],
        "daily_stock_spine_checksum": parents["daily_spine_checksum"],
        "daily_feature_store_identity": parents["daily_feature_identity"],
        "daily_feature_store_version": parents["daily_feature_version"],
        "daily_feature_store_checksum": parents["daily_feature_checksum"],
        "symbol_registry_identity": parents["symbol_registry_identity"],
        "symbol_registry_version": parents["symbol_registry_version"],
        "symbol_registry_checksum": parents["symbol_registry_checksum"],
        "parent_manifests": parents["parent_manifests"],
        "source_price_artifact_identities": parents["source_price_artifact_identities"],
        "point_in_time_feature_store_identities": parents["point_in_time_feature_store_identities"],
        "builder_identity": "core.research.ml.stock_level.selector_dataset:build_frozen_selector_dataset",
        "builder_run_identity": canonical_dataset_run_identity(source_digest, config_hash, parents),
        "git_commit": _git_commit(), "config_hash": config_hash, "checksums": checksums,
        "feature_schema_checksum": checksums["feature_schema.json"],
        "target_schema_checksum": checksums["target_schema.json"],
        "frozen_preflight": frozen_preflight,
        "bounded_symbols": selected_symbols, "bounded_decision_dates": selected_dates,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(), "publication_status": "complete", "validation_status": "VERIFIED",
    }
    manifest["logical_checksum"] = logical_manifest_checksum(manifest)
    _write_json(output_root / "quality_report.json", quality)
    _write_json(output_root / "manifest.json", manifest)
    _write_json(output_root / "checksums.json", checksums)
    os.replace(output_root, final_root)
    return SelectorDatasetPaths(final_root, final_root / "rows.parquet", final_root / "baseline_scores.parquet", final_root / "manifest.json", final_root / "quality_report.json")


def canonical_dataset_run_identity(source_digest: str, config_hash: str | None, parents: dict[str, Any]) -> str:
    payload = {
        "source_checksum": source_digest,
        "config_hash": config_hash,
        "daily_spine_identity": parents["daily_spine_identity"],
        "daily_spine_checksum": parents.get("daily_spine_checksum"),
        "daily_feature_identity": parents.get("daily_feature_identity"),
        "daily_feature_checksum": parents.get("daily_feature_checksum"),
        "symbol_registry_identity": parents["symbol_registry_identity"],
        "symbol_registry_checksum": parents.get("symbol_registry_checksum"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()


def _validate_parent_manifests(source_path: Path, spine_path: Path, feature_path: Path, registry_path: Path) -> dict[str, Any]:
    spine = json.loads(spine_path.read_text(encoding="utf-8")); feature = json.loads(feature_path.read_text(encoding="utf-8")); registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if spine.get("status") != "READY" or spine.get("dataset_type") != "canonical_daily_stock_spine": raise ValueError("Unknown or unready authoritative daily-spine identity")
    if feature.get("status") != "READY" or feature.get("dataset_type") != "daily_price_features": raise ValueError("Unknown or unready authoritative daily-feature identity")
    if registry.get("status") != "READY" or registry.get("dataset_type") != "canonical_asset_registry_audit": raise ValueError("Unknown or unready authoritative symbol-registry identity")
    source_checksum = _sha256(source_path)
    source_checksums = {str(key): str(value).upper() for key, value in dict(feature.get("source_checksums", {})).items()}
    matching_feature_source = any(Path(key).resolve() == source_path.resolve() and value == source_checksum for key, value in source_checksums.items())
    if not matching_feature_source: raise ValueError("Daily-feature parent source checksum or path mismatch")
    if spine.get("dataset_id") not in set(feature.get("source_dataset_ids", [])): raise ValueError("Daily-feature parent does not reference the authoritative spine")
    spine_file = Path(str(spine.get("spine_artifact_path", "")))
    if not spine_file.exists() or str(spine.get("spine_artifact_checksum", "")).upper() != _sha256(spine_file): raise ValueError("Daily-spine artifact checksum mismatch")
    registry_file = Path(str(registry.get("registry_path", "")))
    if not registry_file.exists() or str(registry.get("registry_content_checksum", "")).upper() != _sha256(registry_file): raise ValueError("Symbol-registry parent checksum mismatch")
    return {"daily_spine_identity": spine["dataset_id"], "daily_spine_version": spine["schema_version"], "daily_spine_checksum": _sha256(spine_path), "spine_path": spine_file, "daily_feature_identity": feature["dataset_id"], "daily_feature_version": feature["schema_version"], "daily_feature_checksum": _sha256(feature_path), "symbol_registry_identity": registry["dataset_id"], "symbol_registry_version": registry["symbol_registry_version"], "symbol_registry_checksum": _sha256(registry_path), "registry_path": registry_file, "parent_manifests": [{"path": str(spine_path), "checksum": _sha256(spine_path)}, {"path": str(feature_path), "checksum": _sha256(feature_path)}, {"path": str(registry_path), "checksum": _sha256(registry_path)}], "source_price_artifact_identities": spine.get("source_price_artifact_identities", []), "point_in_time_feature_store_identities": [feature["dataset_id"]]}


def _dataset_population_identity(rows_path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq
    table = pq.read_table(rows_path, columns=["row_id", "asset_id", "canonical_symbol", "decision_session_date"])
    rows = table.to_pylist(); keys = [(str(row["asset_id"]), str(row["decision_session_date"])) for row in rows]
    if len(keys) != len(set(keys)): raise ValueError("Duplicate stock-date identity in frozen selector dataset")
    canonical_rows = sorted(
        rows,
        key=lambda row: (
            str(row["decision_session_date"]),
            str(row["asset_id"]),
            str(row["row_id"]),
        ),
    )
    if rows != canonical_rows:
        raise ValueError("Noncanonical frozen selector dataset ordering")
    ordered = [str(row["row_id"]) for row in rows]
    dates = sorted({str(row["decision_session_date"]) for row in rows})
    return {"row_count": len(rows), "symbol_count": len({str(row["asset_id"]) for row in rows}), "date_coverage": {"min": dates[0], "max": dates[-1], "count": len(dates)}, "row_population_checksum": hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode()).hexdigest().upper()}


def _validate_rows_against_parents(rows_path: Path, parents: dict[str, Any]) -> None:
    import csv
    import pyarrow.parquet as pq
    with parents["registry_path"].open("r", encoding="utf-8", newline="") as handle:
        registry = {row["asset_id"]: row["canonical_symbol"] for row in csv.DictReader(handle)}
    table = pq.read_table(rows_path, columns=["asset_id", "canonical_symbol", "decision_session_date"])
    spine = pq.read_table(parents["spine_path"], columns=["asset_id", "session_date"])
    spine_keys = {(str(row["asset_id"]), str(row["session_date"])) for row in spine.to_pylist()}
    for row in table.to_pylist():
        expected = registry.get(str(row["asset_id"]))
        if expected is None: raise ValueError(f"Unresolved selector asset: {row['asset_id']}")
        if expected != str(row["canonical_symbol"]): raise ValueError(f"Ambiguous canonical symbol mapping: {row['asset_id']}")
        if (str(row["asset_id"]), str(row["decision_session_date"])) not in spine_keys: raise ValueError(f"Selector row absent from authoritative daily spine: {row['asset_id']}:{row['decision_session_date']}")


def _score_identity_rows(identity: list[dict[str, Any]], market_root: Path, ds: Any) -> list[dict[str, Any]]:
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in identity:
        by_symbol.setdefault(str(row["symbol"]).upper(), []).append(row)
    output: list[dict[str, Any]] = []
    for symbol in sorted(by_symbol):
        bars = ds.dataset(market_root / f"symbol={symbol}", format="parquet", partitioning="hive").to_table(
            columns=["session_date", "model_close", "raw_volume"]
        ).sort_by("session_date").to_pylist()
        dates = [str(row["session_date"]) for row in bars]
        closes = [float(row["model_close"]) for row in bars]
        dollars = [float(row["model_close"]) * float(row["raw_volume"] or 0.0) for row in bars]
        precomputed = _precomputed_signals(dates, closes, dollars)
        for row in sorted(by_symbol[symbol], key=lambda x: str(x["decision_timestamp"])):
            decision_date = str(row["decision_session_date"])
            values = precomputed.get(decision_date)
            if values is None:
                raise RuntimeError(f"Canonical bar missing for selector decision: {symbol} {decision_date}")
            output.append({
                "row_id": _stable_selector_row_id(str(row["asset_id"]), str(row["decision_timestamp"])),
                "asset_id": str(row["asset_id"]), "decision_timestamp": str(row["decision_timestamp"]),
                "baseline_contract_version": BASELINE_CONTRACT_VERSION, **values,
            })
    return output


def _precomputed_signals(dates: list[str], closes: list[float], dollars: list[float]) -> dict[str, dict[str, float | None]]:
    """Linear/vectorized equivalent of the authoritative strictly-prior formulas."""
    import numpy as np
    import pandas as pd

    close = pd.Series(closes, dtype="float64")
    dollar = pd.Series(dollars, dtype="float64")
    prior = close.shift(1)
    m20 = prior / close.shift(21) - 1.0
    m60 = prior / close.shift(61) - 1.0
    m120 = prior / close.shift(121) - 1.0
    returns = close.pct_change(fill_method=None)
    vol20 = returns.shift(1).rolling(20, min_periods=20).std(ddof=0)
    liquidity = np.log1p(dollar.shift(1).rolling(63, min_periods=1).mean())
    dd60 = np.full(len(close), np.nan)
    if len(close) >= 60:
        windows = np.lib.stride_tricks.sliding_window_view(close.to_numpy(), 60)
        running_peaks = np.maximum.accumulate(windows, axis=1)
        drawdowns = windows / running_peaks - 1.0
        # Window ending i-1 belongs to decision/bar index i.
        dd60[60:] = drawdowns[:-1].min(axis=1) if len(drawdowns) > 1 else np.array([], dtype=float)
    output: dict[str, dict[str, float | None]] = {}
    for index, date in enumerate(dates):
        risk = max(abs(_finite_or_zero(vol20.iloc[index])), abs(_finite_or_zero(dd60[index])), 1e-6)
        output[date] = {
            "predicted_momentum_20d": _finite_or_none(m20.iloc[index]),
            "predicted_momentum_60d": _finite_or_none(m60.iloc[index]),
            "predicted_momentum_120d": _finite_or_none(m120.iloc[index]),
            "predicted_volatility_20d": _finite_or_none(vol20.iloc[index]),
            "predicted_drawdown_60d": _finite_or_none(dd60[index]),
            "predicted_liquidity_score": _finite_or_none(liquidity.iloc[index]),
            "predicted_risk_adjusted_momentum": (
                None if _finite_or_none(m60.iloc[index]) is None else float(m60.iloc[index]) / risk
            ),
        }
    return output


def _finite_or_none(value: Any) -> float | None:
    import math
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _finite_or_zero(value: Any) -> float:
    return _finite_or_none(value) or 0.0


def _nullable(value: float | str) -> float | None:
    return None if value == "" else float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
