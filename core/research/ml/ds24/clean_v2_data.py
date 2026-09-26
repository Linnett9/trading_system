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


IDENTITY_COLUMNS = ("asset_id", "decision_timestamp")
REGULAR_SESSION_TYPES = frozenset(
    {"rth", "regular", "regular_session", "early_close"}
)


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


def _normalize_identity_keys(
    frame: pd.DataFrame,
    *,
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = sorted(set(IDENTITY_COLUMNS) - set(frame.columns))
    if missing:
        raise CleanV2DataError(f"{label} is missing identity columns: {missing}")
    key_dtypes = {column: str(frame[column].dtype) for column in IDENTITY_COLUMNS}
    null_counts = {
        column: int(frame[column].isna().sum()) for column in IDENTITY_COLUMNS
    }
    if any(null_counts.values()):
        raise CleanV2DataError(
            f"{label} has null identity keys: {null_counts}"
        )
    normalized = frame.copy()
    try:
        normalized["decision_timestamp"] = pd.to_datetime(
            normalized["decision_timestamp"], utc=True, errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise CleanV2DataError(
            f"{label} has invalid decision_timestamp values"
        ) from exc
    observed_assets = sorted(set(normalized["asset_id"].astype(str)))
    if len(observed_assets) > 1:
        raise CleanV2DataError(
            f"{label} contains multiple asset identities: {observed_assets[:5]}"
        )
    duplicate_count = int(normalized.duplicated(list(IDENTITY_COLUMNS)).sum())
    if duplicate_count:
        raise CleanV2DataError(
            f"{label} has duplicate identity keys: {duplicate_count}"
        )
    return normalized, {
        "key_dtypes": key_dtypes,
        "null_identity_counts": null_counts,
        "duplicate_identity_count": duplicate_count,
        "observed_asset_ids": observed_assets,
    }


def _repair_missingness_summary(
    joined: pd.DataFrame,
    *,
    repaired: Sequence[str],
    provenance: Sequence[str],
    partition_label: str,
) -> dict[str, Any]:
    if "session_type" not in joined:
        raise CleanV2DataError(
            f"Base feature partition lacks session_type: {partition_label}"
        )
    normalized_session_type = (
        joined["session_type"].astype(str).str.strip().str.lower()
    )
    regular_mask = normalized_session_type.isin(REGULAR_SESSION_TYPES)
    extended_mask = ~regular_mask
    contract_columns = [*repaired, *provenance]
    extended_non_null = {
        column: int(joined.loc[extended_mask, column].notna().sum())
        for column in contract_columns
    }
    if any(extended_non_null.values()):
        raise CleanV2DataError(
            "Extended-hours V2 repair values must remain null: "
            f"{partition_label}:{extended_non_null}"
        )
    required_regular_features = (
        "minutes_since_open",
        "minutes_until_close",
        "session_progress",
        "early_close_session_flag",
        "opening_period_flag",
    )
    regular_required_nulls = {
        feature: int(joined.loc[regular_mask, feature].isna().sum())
        for feature in required_regular_features
    }
    if any(regular_required_nulls.values()):
        raise CleanV2DataError(
            "Regular-session calendar repairs must be populated: "
            f"{partition_label}:{regular_required_nulls}"
        )
    value_without_provenance = {
        feature: int(
            (
                joined[feature].notna()
                & joined[source_column].isna()
            ).sum()
        )
        for feature, source_column in zip(repaired, provenance)
    }
    if any(value_without_provenance.values()):
        raise CleanV2DataError(
            "V2 repair value lacks provenance timestamp: "
            f"{partition_label}:{value_without_provenance}"
        )
    provenance_without_value = {
        feature: int(
            (
                joined[feature].isna()
                & joined[source_column].notna()
            ).sum()
        )
        for feature, source_column in zip(repaired, provenance)
    }
    ordered_regular = joined.loc[regular_mask].sort_values(
        ["session_date", "decision_timestamp"], kind="mergesort"
    )
    opening_position = ordered_regular.groupby(
        "session_date", sort=False
    ).cumcount()
    opening_window = ordered_regular.loc[opening_position < 5]
    opening_features = (
        "opening_range_position",
        "session_return_30m",
        "opening_return_30m",
    )
    opening_window_non_null = {
        feature: int(opening_window[feature].notna().sum())
        for feature in opening_features
    }
    if any(opening_window_non_null.values()):
        raise CleanV2DataError(
            "Opening-window repairs populated before the sixth finalized bar: "
            f"{partition_label}:{opening_window_non_null}"
        )
    first_session = str(joined["session_date"].astype(str).min())
    first_history = joined["session_date"].astype(str).eq(first_session)
    history_features = (
        "overnight_gap",
        "previous_session_return",
        "two_session_return",
    )
    return {
        "regular_session_rows": int(regular_mask.sum()),
        "extended_hours_rows": int(extended_mask.sum()),
        "matched_rows_with_any_null_repair_value": int(
            joined.loc[:, list(repaired)].isna().any(axis=1).sum()
        ),
        "repair_null_counts": {
            feature: int(joined[feature].isna().sum()) for feature in repaired
        },
        "extended_non_null_counts": extended_non_null,
        "regular_required_null_counts": regular_required_nulls,
        "value_without_provenance_counts": value_without_provenance,
        "provenance_without_value_counts": provenance_without_value,
        "opening_window_rows": int(len(opening_window)),
        "opening_window_non_null_counts": opening_window_non_null,
        "earliest_selected_session": first_session,
        "earliest_session_history_null_counts": {
            feature: int(joined.loc[first_history, feature].isna().sum())
            for feature in history_features
        },
    }


def _join_repaired_features(
    base: pd.DataFrame,
    sidecar: pd.DataFrame,
    *,
    repaired: Sequence[str],
    source_timestamp_suffix: str,
    partition_label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    provenance = tuple(f"{name}{source_timestamp_suffix}" for name in repaired)
    required_sidecar = {*IDENTITY_COLUMNS, *repaired, *provenance}
    missing_sidecar = sorted(required_sidecar - set(sidecar.columns))
    if missing_sidecar:
        raise CleanV2DataError(
            f"V2 sidecar is missing required columns: {missing_sidecar}"
        )
    normalized_base, base_identity = _normalize_identity_keys(
        base,
        label=f"base feature partition {partition_label}",
    )
    normalized_sidecar, sidecar_identity = _normalize_identity_keys(
        sidecar,
        label=f"V2 sidecar partition {partition_label}",
    )
    if (
        sidecar_identity["observed_asset_ids"]
        and base_identity["observed_asset_ids"]
        != sidecar_identity["observed_asset_ids"]
    ):
        raise CleanV2DataError(
            f"Base/sidecar asset identity mismatch for {partition_label}: "
            f"base={base_identity['observed_asset_ids']}, "
            f"sidecar={sidecar_identity['observed_asset_ids']}"
        )
    for source_column in provenance:
        raw_source = normalized_sidecar[source_column]
        source = pd.to_datetime(raw_source, utc=True, errors="coerce")
        invalid = raw_source.notna() & source.isna()
        if bool(invalid.any()):
            raise CleanV2DataError(
                f"Invalid source timestamp in {partition_label}:{source_column}"
            )
        future = source.notna() & (
            source > normalized_sidecar["decision_timestamp"]
        )
        if bool(future.any()):
            raise CleanV2DataError(
                f"Future source timestamp in {partition_label}:{source_column}"
            )
        normalized_sidecar[source_column] = source

    replacement = normalized_sidecar[
        [*IDENTITY_COLUMNS, *repaired, *provenance]
    ].copy()
    replacement["__v2_sidecar_present"] = True
    retained = normalized_base.drop(columns=list(repaired))
    joined = retained.merge(
        replacement,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
        indicator="__v2_join_status",
        sort=False,
    )
    unmatched = joined["__v2_sidecar_present"].isna()
    unmatched_count = int(unmatched.sum())
    unmatched_examples = (
        joined.loc[
            unmatched,
            [*IDENTITY_COLUMNS, "session_date", "session_type"],
        ]
        .head(5)
        .assign(
            decision_timestamp=lambda rows: rows["decision_timestamp"].map(
                lambda value: pd.Timestamp(value).isoformat()
            )
        )
        .to_dict("records")
    )
    if len(joined) != len(normalized_base) or unmatched_count:
        raise CleanV2DataError(
            "V2 sidecar does not fully cover base rows: "
            f"{partition_label}; base_rows={len(normalized_base)}; "
            f"unmatched_keys={unmatched_count}; examples={unmatched_examples}"
        )
    missingness = _repair_missingness_summary(
        joined,
        repaired=repaired,
        provenance=provenance,
        partition_label=partition_label,
    )
    report = {
        "base_rows": int(len(normalized_base)),
        "sidecar_rows_selected": int(len(normalized_sidecar)),
        "matched_rows": int(len(joined) - unmatched_count),
        "unmatched_base_key_count": unmatched_count,
        "unmatched_base_key_examples": unmatched_examples,
        "base_identity": base_identity,
        "sidecar_identity": sidecar_identity,
        **missingness,
    }
    returned = joined.drop(
        columns=[
            *provenance,
            "session_type",
            "__v2_sidecar_present",
            "__v2_join_status",
        ]
    )
    return returned, report


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

    def _read_base_feature_slice(
        self,
        partition: CleanV2Partition,
        session_dates: Sequence[str],
    ) -> pd.DataFrame:
        base_columns = [
            "asset_id",
            "canonical_symbol",
            "decision_timestamp",
            "session_date",
            "session_type",
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
        return base

    def _feature_partition_inputs(
        self,
        partition: CleanV2Partition,
        session_dates: Sequence[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
        base = self._read_base_feature_slice(partition, session_dates)
        if base.empty:
            return base, pd.DataFrame(), (), ()
        base["decision_timestamp"] = pd.to_datetime(
            base["decision_timestamp"], utc=True, errors="raise"
        )
        repaired = tuple(self.feature_contract["sidecar"]["repair_columns"])
        provenance = tuple(
            f"{name}{self.feature_contract['sidecar']['source_timestamp_suffix']}"
            for name in repaired
        )
        # Session labels are exchange dates, while extended-hours timestamps can
        # cross a UTC midnight. Bound the read by the exact selected base keys;
        # session-date midnights are not valid timestamp coverage bounds.
        start = pd.Timestamp(base["decision_timestamp"].min()).tz_convert("UTC")
        end = (
            pd.Timestamp(base["decision_timestamp"].max()).tz_convert("UTC")
            + pd.Timedelta(microseconds=1)
        )
        sidecar = self._read_timestamp_slice(
            partition.sidecar_path,
            columns=["asset_id", "decision_timestamp", *repaired, *provenance],
            start=start,
            end=end,
        )
        return base, sidecar, repaired, provenance

    def _read_feature_partition(
        self, partition: CleanV2Partition, session_dates: Sequence[str]
    ) -> pd.DataFrame:
        base, sidecar, repaired, _provenance = self._feature_partition_inputs(
            partition, session_dates
        )
        if base.empty:
            return base
        joined, _report = _join_repaired_features(
            base,
            sidecar,
            repaired=repaired,
            source_timestamp_suffix=self.feature_contract["sidecar"][
                "source_timestamp_suffix"
            ],
            partition_label=f"{partition.asset_id}/{partition.year}",
        )
        return joined

    def diagnose_feature_partition(
        self,
        partition: CleanV2Partition,
        session_dates: Sequence[str],
    ) -> dict[str, Any]:
        """Compare the rejected session-date bounds with exact key coverage."""

        base, exact_sidecar, repaired, provenance = self._feature_partition_inputs(
            partition, session_dates
        )
        if base.empty:
            raise CleanV2DataError(
                f"No base rows selected for {partition.asset_id}/{partition.year}"
            )
        legacy_start, legacy_end = self._date_bounds(session_dates)
        legacy_sidecar = self._read_timestamp_slice(
            partition.sidecar_path,
            columns=[*IDENTITY_COLUMNS, *repaired, *provenance],
            start=legacy_start,
            end=legacy_end,
        )
        base_keys = base[
            [*IDENTITY_COLUMNS, "session_date", "session_type"]
        ].copy()
        base_keys["decision_timestamp"] = pd.to_datetime(
            base_keys["decision_timestamp"], utc=True, errors="raise"
        )
        legacy_keys = legacy_sidecar[[*IDENTITY_COLUMNS]].copy()
        legacy_keys["decision_timestamp"] = pd.to_datetime(
            legacy_keys["decision_timestamp"], utc=True, errors="raise"
        )
        legacy_keys["__legacy_present"] = True
        legacy_join = base_keys.merge(
            legacy_keys,
            on=list(IDENTITY_COLUMNS),
            how="left",
            validate="one_to_one",
            sort=False,
        )
        legacy_unmatched = legacy_join["__legacy_present"].isna()
        offending_keys = legacy_join.loc[
            legacy_unmatched,
            [*IDENTITY_COLUMNS, "session_date", "session_type"],
        ].copy()
        exact_lookup = exact_sidecar[
            [*IDENTITY_COLUMNS, *repaired, *provenance]
        ].copy()
        exact_lookup["decision_timestamp"] = pd.to_datetime(
            exact_lookup["decision_timestamp"], utc=True, errors="raise"
        )
        offending_values = offending_keys.merge(
            exact_lookup,
            on=list(IDENTITY_COLUMNS),
            how="left",
            validate="one_to_one",
            sort=False,
        )
        _joined, exact_report = _join_repaired_features(
            base,
            exact_sidecar,
            repaired=repaired,
            source_timestamp_suffix=self.feature_contract["sidecar"][
                "source_timestamp_suffix"
            ],
            partition_label=f"{partition.asset_id}/{partition.year}",
        )
        regular = (
            base["session_type"].astype(str).str.strip().str.lower()
        ).isin(REGULAR_SESSION_TYPES)
        example_columns = [
            *IDENTITY_COLUMNS,
            "session_date",
            "session_type",
            *repaired,
            *provenance,
        ]
        examples = offending_values.loc[:, example_columns].head(5).copy()
        examples["decision_timestamp"] = examples["decision_timestamp"].map(
            lambda value: pd.Timestamp(value).isoformat()
        )
        return {
            "partition": f"{partition.asset_id}/{partition.year}",
            "session_dates": list(session_dates),
            "base_rows_requested": int(len(base)),
            "legacy_sidecar_rows_selected": int(len(legacy_sidecar)),
            "exact_key_range_sidecar_rows_selected": int(len(exact_sidecar)),
            "join_keys": list(IDENTITY_COLUMNS),
            "base_key_dtypes": {
                column: str(base[column].dtype) for column in IDENTITY_COLUMNS
            },
            "sidecar_key_dtypes": {
                column: str(exact_sidecar[column].dtype)
                for column in IDENTITY_COLUMNS
            },
            "base_null_identity_counts": {
                column: int(base[column].isna().sum())
                for column in IDENTITY_COLUMNS
            },
            "sidecar_null_identity_counts": {
                column: int(exact_sidecar[column].isna().sum())
                for column in IDENTITY_COLUMNS
            },
            "base_duplicate_identity_count": int(
                base.duplicated(list(IDENTITY_COLUMNS)).sum()
            ),
            "sidecar_duplicate_identity_count": int(
                exact_sidecar.duplicated(list(IDENTITY_COLUMNS)).sum()
            ),
            "regular_session_rows": int(regular.sum()),
            "extended_hours_rows": int((~regular).sum()),
            "legacy_unmatched_base_key_count": int(legacy_unmatched.sum()),
            "legacy_unmatched_examples_with_exact_sidecar_values": examples.to_dict(
                "records"
            ),
            "legacy_failing_predicate": (
                "len(inner_join(base_session_dates, "
                "sidecar_timestamp_between_session_midnights)) != len(base)"
            ),
            "legacy_predicate_value": (
                f"{int((~legacy_unmatched).sum())} != {len(base)}"
            ),
            "exact_identity_coverage_passed": (
                exact_report["unmatched_base_key_count"] == 0
            ),
            "production_reader_passed": True,
            **exact_report,
        }

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
        asset_ids: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        dates = tuple(sorted({str(value) for value in session_dates}))
        if not dates:
            return pd.DataFrame()
        years = {int(value[:4]) for value in dates}
        available_assets = {
            row.asset_id for row in self.partitions if row.year in years
        }
        assets = sorted(
            available_assets
            if asset_ids is None
            else available_assets & {str(value) for value in asset_ids}
        )
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
        for predictor in self.context_predictors:
            panel[predictor] = np.nan
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
            numeric_context = mapped[list(self.context_predictors)].apply(
                pd.to_numeric, errors="coerce"
            )
            panel.loc[positions, list(self.context_predictors)] = (
                numeric_context.to_numpy(dtype="float64")
            )

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
