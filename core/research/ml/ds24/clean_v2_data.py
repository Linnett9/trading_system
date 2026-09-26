from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

from core.research.ml.ds24.clean_v2_contracts import load_contract, stable_hash


class CleanV2DataError(RuntimeError):
    """Raised when the immutable V2 composite data authority is inconsistent."""


@dataclass(frozen=True)
class CleanV2Partition:
    asset_id: str
    year: int
    base_feature_path: Path
    sidecar_path: Path
    target_paths: tuple[Path, ...]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CleanV2DataError(f"Expected a JSON object: {path}")
    return payload


def _path_identity(relative_path: str, *, asset_key: str) -> tuple[str, int]:
    parts = Path(relative_path).parts
    asset_parts = [part for part in parts if part.startswith(f"{asset_key}=")]
    year_parts = [part for part in parts if part.startswith("year=")]
    if len(asset_parts) != 1 or len(year_parts) != 1:
        raise CleanV2DataError(f"Cannot parse partition identity: {relative_path}")
    return asset_parts[0].split("=", 1)[1], int(year_parts[0].split("=", 1)[1])


def _logical_manifest_valid(payload: Mapping[str, Any]) -> bool:
    expected = payload.get("logical_sha256")
    if not isinstance(expected, str):
        return False
    return expected == stable_hash(
        {key: value for key, value in payload.items() if key != "logical_sha256"}
    )


class CleanV2CompositeData:
    """Read-only composition of immutable V1 features, V2 repairs, and targets."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.feature_contract = load_contract("feature_authority.json")
        self.target_contract = load_contract("target_contract.json")
        self.predictor_contract = load_contract("predictor_manifest.json")
        self.predictors = tuple(str(value) for value in self.predictor_contract["predictors"])
        self.stock_predictors = tuple(
            self.predictors[: int(self.predictor_contract["stock_predictor_count"])]
        )
        self.context_predictors = tuple(
            self.predictors[int(self.predictor_contract["stock_predictor_count"]) :]
        )
        if any("target" in name.lower() or "forward_return" in name.lower() for name in self.predictors):
            raise CleanV2DataError("Target-like column found in predictor authority")

        self.base_feature_root = self.root / self.feature_contract["base_authority"]["path"]
        self.sidecar_root = self.root / self.feature_contract["sidecar"]["path"]
        physical_target = self.target_contract["physical_authority"]
        self.base_target_root = self.root / physical_target["base_path"]
        self.delta_target_root = self.root / physical_target["delta_path"]
        self.sidecar_manifest = _read_json(self.sidecar_root / "authority_manifest.json")
        self.delta_target_manifest = _read_json(self.delta_target_root / "authority_manifest.json")
        self._validate_manifests()
        self.partitions = self._partition_inventory()

    def _validate_manifests(self) -> None:
        feature = self.feature_contract
        sidecar = self.sidecar_manifest
        target = self.delta_target_manifest
        expected_feature_partitions = int(
            feature["base_authority"]["physical_snapshot_stock_partitions"]
        )
        expected_assets = int(feature["base_authority"]["physical_snapshot_asset_count"])
        if not (
            sidecar.get("complete") is True
            and sidecar.get("authority_id") == feature["authority_id"]
            and len(sidecar.get("partitions", [])) == expected_feature_partitions
            and int(sidecar.get("row_count", -1))
            == int(feature["base_authority"]["physical_snapshot_row_count"])
            and _logical_manifest_valid(sidecar)
        ):
            raise CleanV2DataError("V2 feature sidecar manifest is incomplete or invalid")
        if not (
            target.get("complete") is True
            and target.get("authority_id") == self.target_contract["authority_id"]
            and target.get("target_id") == self.target_contract["target_id"]
            and int(target.get("symbol_count", -1)) == expected_assets
            and _logical_manifest_valid(target)
        ):
            raise CleanV2DataError("V2 target-delta manifest is incomplete or invalid")

    def _partition_inventory(self) -> tuple[CleanV2Partition, ...]:
        base_targets: dict[tuple[str, int], Path] = {}
        base_snapshot = self.delta_target_manifest.get("base_snapshot") or {}
        for row in base_snapshot.get("partitions", []):
            relative = str(row["base_relative_path"])
            key = _path_identity(relative, asset_key="symbol")
            base_targets[key] = self.base_target_root / relative

        delta_targets: dict[tuple[str, int], Path] = {}
        for row in self.delta_target_manifest.get("partitions", []):
            relative = str(row["relative_path"])
            key = _path_identity(relative, asset_key="symbol")
            delta_targets[key] = self.delta_target_root / relative

        partitions: list[CleanV2Partition] = []
        seen: set[tuple[str, int]] = set()
        for row in self.sidecar_manifest["partitions"]:
            base_relative = str(row["base_relative_path"])
            sidecar_relative = str(row["relative_path"])
            key = _path_identity(base_relative, asset_key="asset")
            if key in seen:
                raise CleanV2DataError(f"Duplicate V2 feature partition: {key}")
            seen.add(key)
            target_paths = tuple(
                path
                for path in (base_targets.get(key), delta_targets.get(key))
                if path is not None
            )
            if not target_paths:
                raise CleanV2DataError(f"No target authority for feature partition: {key}")
            partitions.append(
                CleanV2Partition(
                    asset_id=key[0],
                    year=key[1],
                    base_feature_path=self.base_feature_root / base_relative,
                    sidecar_path=self.sidecar_root / sidecar_relative,
                    target_paths=target_paths,
                )
            )
        return tuple(sorted(partitions, key=lambda row: (row.year, row.asset_id)))

    @staticmethod
    def _date_bounds(session_dates: Sequence[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
        first = pd.Timestamp(min(session_dates), tz="UTC")
        last = pd.Timestamp(max(session_dates), tz="UTC") + pd.Timedelta(days=1)
        return first, last

    @staticmethod
    def _read_timestamp_slice(
        path: Path,
        *,
        columns: Sequence[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        try:
            return pd.read_parquet(
                path,
                columns=list(columns),
                filters=[
                    ("decision_timestamp", ">=", start.to_pydatetime()),
                    ("decision_timestamp", "<", end.to_pydatetime()),
                ],
            )
        except (
            TypeError,
            ValueError,
            pa.ArrowInvalid,
            pa.ArrowNotImplementedError,
        ):
            frame = pd.read_parquet(path, columns=list(columns))
            decisions = pd.to_datetime(frame["decision_timestamp"], utc=True)
            return frame[(decisions >= start) & (decisions < end)].copy()

    def _read_feature_partition(
        self, partition: CleanV2Partition, session_dates: Sequence[str]
    ) -> pd.DataFrame:
        start, end = self._date_bounds(session_dates)
        base_columns = [
            "asset_id",
            "canonical_symbol",
            "decision_timestamp",
            "session_date",
            *self.stock_predictors,
        ]
        try:
            base = pd.read_parquet(
                partition.base_feature_path,
                columns=base_columns,
                filters=[("session_date", "in", list(session_dates))],
            )
        except (
            TypeError,
            ValueError,
            pa.ArrowInvalid,
            pa.ArrowNotImplementedError,
        ):
            base = pd.read_parquet(partition.base_feature_path, columns=base_columns)
            base = base[base["session_date"].astype(str).isin(session_dates)].copy()
        if base.empty:
            return base
        repaired = tuple(self.feature_contract["sidecar"]["repair_columns"])
        provenance = tuple(
            f"{name}{self.feature_contract['sidecar']['source_timestamp_suffix']}"
            for name in repaired
        )
        sidecar = self._read_timestamp_slice(
            partition.sidecar_path,
            columns=["asset_id", "decision_timestamp", *repaired, *provenance],
            start=start,
            end=end,
        )
        sidecar["decision_timestamp"] = pd.to_datetime(
            sidecar["decision_timestamp"], utc=True
        )
        base["decision_timestamp"] = pd.to_datetime(base["decision_timestamp"], utc=True)
        for source_column in provenance:
            source = pd.to_datetime(sidecar[source_column], utc=True, errors="coerce")
            if bool((source.notna() & (source > sidecar["decision_timestamp"])).any()):
                raise CleanV2DataError(
                    f"Future source timestamp in {partition.sidecar_path}:{source_column}"
                )
        replacement = sidecar[["asset_id", "decision_timestamp", *repaired]].copy()
        if replacement.duplicated(["asset_id", "decision_timestamp"]).any():
            raise CleanV2DataError(f"Duplicate V2 sidecar key: {partition.sidecar_path}")
        base = base.drop(columns=list(repaired))
        joined = base.merge(
            replacement,
            on=["asset_id", "decision_timestamp"],
            how="inner",
            validate="one_to_one",
        )
        if len(joined) != len(base):
            raise CleanV2DataError(
                f"V2 sidecar does not fully cover base rows: {partition.asset_id}/{partition.year}"
            )
        return joined

    def _read_targets(
        self, partition: CleanV2Partition, session_dates: Sequence[str]
    ) -> pd.DataFrame:
        start, end = self._date_bounds(session_dates)
        columns = [
            "asset_id",
            "decision_timestamp",
            "target_id",
            "target_value",
            "target_is_trainable",
            "target_available_timestamp",
        ]
        frames = [
            self._read_timestamp_slice(path, columns=columns, start=start, end=end)
            for path in partition.target_paths
        ]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame(columns=columns)
        target = pd.concat(frames, ignore_index=True)
        target["decision_timestamp"] = pd.to_datetime(
            target["decision_timestamp"], utc=True
        )
        target["target_available_timestamp"] = pd.to_datetime(
            target["target_available_timestamp"], utc=True
        )
        if target.duplicated(["asset_id", "decision_timestamp"]).any():
            raise CleanV2DataError(
                f"Overlapping base/delta target keys: {partition.asset_id}/{partition.year}"
            )
        if set(target["target_id"].astype(str)) != {self.target_contract["target_id"]}:
            raise CleanV2DataError("Target identity mismatch")
        return target

    def assemble_sessions(
        self,
        session_dates: Iterable[str],
        *,
        maximum_assets: int | None = None,
    ) -> pd.DataFrame:
        dates = tuple(sorted({str(value) for value in session_dates}))
        if not dates:
            return pd.DataFrame()
        years = {int(value[:4]) for value in dates}
        assets = sorted({row.asset_id for row in self.partitions if row.year in years})
        if maximum_assets is not None:
            assets = assets[: max(1, int(maximum_assets))]
        selected_assets = set(assets)
        rows = [
            row
            for row in self.partitions
            if row.year in years and row.asset_id in selected_assets
        ]
        frames: list[pd.DataFrame] = []
        for partition in rows:
            features = self._read_feature_partition(partition, dates)
            if features.empty:
                continue
            targets = self._read_targets(partition, dates)
            if targets.empty:
                continue
            joined = features.merge(
                targets,
                on=["asset_id", "decision_timestamp"],
                how="inner",
                validate="one_to_one",
            )
            if not joined.empty:
                frames.append(joined)
        if not frames:
            return pd.DataFrame()

        panel = pd.concat(frames, ignore_index=True)
        panel["decision_timestamp"] = pd.to_datetime(
            panel["decision_timestamp"], utc=True
        )
        for year, positions in panel.groupby(
            panel["decision_timestamp"].dt.year, sort=False
        ).groups.items():
            context_path = (
                self.base_feature_root
                / "shared-context"
                / f"year={int(year)}"
                / "context.parquet"
            )
            subset_dates = sorted(
                set(panel.loc[positions, "session_date"].astype(str))
            )
            start, end = self._date_bounds(subset_dates)
            context = self._read_timestamp_slice(
                context_path,
                columns=["decision_timestamp", *self.context_predictors],
                start=start,
                end=end,
            )
            context["decision_timestamp"] = pd.to_datetime(
                context["decision_timestamp"], utc=True
            )
            context = context.drop_duplicates("decision_timestamp")
            mapped = panel.loc[positions, ["decision_timestamp"]].merge(
                context,
                on="decision_timestamp",
                how="left",
                validate="many_to_one",
            )
            panel.loc[positions, list(self.context_predictors)] = mapped[
                list(self.context_predictors)
            ].to_numpy()

        missing = sorted(set(self.predictors) - set(panel.columns))
        if missing:
            raise CleanV2DataError(f"Composite panel missing predictors: {missing}")
        panel[list(self.predictors)] = panel[list(self.predictors)].replace(
            [np.inf, -np.inf], np.nan
        )
        return panel.sort_values(
            ["decision_timestamp", "asset_id"], kind="mergesort"
        ).reset_index(drop=True)

    def decision_spine(self, reference_asset: str = "SPY") -> list[pd.Timestamp]:
        frames: list[pd.DataFrame] = []
        for partition in self.partitions:
            if partition.asset_id != reference_asset:
                continue
            target_columns = [
                "decision_timestamp",
                "target_id",
                "target_is_trainable",
            ]
            for path in partition.target_paths:
                target = pd.read_parquet(path, columns=target_columns)
                target = target[
                    target["target_is_trainable"].astype(bool)
                    & (
                        target["target_id"].astype(str)
                        == self.target_contract["target_id"]
                    )
                ]
                if not target.empty:
                    frames.append(target)
        if not frames:
            raise CleanV2DataError("Reference-asset decision spine is empty")
        values = pd.to_datetime(
            pd.concat(frames, ignore_index=True)["decision_timestamp"], utc=True
        )
        return list(values.drop_duplicates().sort_values())

    @staticmethod
    def target_loader_from_panel(
        panel: pd.DataFrame,
        request: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        keys = request[["asset_id", "decision_timestamp"]].copy()
        keys["asset_id"] = keys["asset_id"].astype(str)
        keys["decision_timestamp"] = pd.to_datetime(
            keys["decision_timestamp"], utc=True
        )
        targets = panel[
            [
                "asset_id",
                "decision_timestamp",
                "target_id",
                "target_value",
                "target_is_trainable",
                "target_available_timestamp",
            ]
        ].copy()
        targets["asset_id"] = targets["asset_id"].astype(str)
        targets["decision_timestamp"] = pd.to_datetime(
            targets["decision_timestamp"], utc=True
        )
        matched = keys.merge(
            targets,
            on=["asset_id", "decision_timestamp"],
            how="left",
            validate="one_to_one",
        )
        return matched, {
            "target_loader": "DS24_CLEAN_V2_COMPOSITE_PANEL",
            "target_rows_loaded": int(matched["target_value"].notna().sum()),
        }
