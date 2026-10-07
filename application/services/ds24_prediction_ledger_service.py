from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import pandas as pd
from pandas.api.types import is_bool_dtype

from core.research.ml.ds24.evaluation_burn_in import (
    EVALUATION_BURN_IN,
    EvaluationBurnIn,
)
from core.research.ml.ds24.prediction_ledger_contract import (
    PREDICTION_LEDGER_COLUMNS,
    PredictionProvenance,
    validate_prediction_ledger_rows,
)
from core.research.ml.ds24.research_output import prepare_ranked_prediction_ledger_rows
from infrastructure.data.ds24_prediction_ledger import PredictionLedgerStore


class DS24PredictionPublicationError(ValueError):
    """Raised when a scoring result cannot be safely published to the ledger."""


LEDGER_POPULATION_ABSENT: Final = "ABSENT"
LEDGER_POPULATION_COMPLETE_MATCHING: Final = "COMPLETE_MATCHING"


@dataclass(frozen=True)
class DurablePredictionRecovery:
    """Validated ledger-first disposition for an ordered timestamp batch."""

    recovered_batches: Mapping[str, pd.DataFrame]
    absent_timestamps: tuple[str, ...]
    classifications: Mapping[str, str]

    @property
    def recovered_row_count(self) -> int:
        return sum(len(frame) for frame in self.recovered_batches.values())


@dataclass(frozen=True)
class DS24PredictionLedgerPublisher:
    """Application boundary from DS24 scoring output to immutable ledger rows."""

    store: PredictionLedgerStore
    provenance: PredictionProvenance
    burn_in: EvaluationBurnIn = EVALUATION_BURN_IN

    @classmethod
    def from_authority(
        cls,
        *,
        store: PredictionLedgerStore,
        authority: Mapping[str, str],
        run_id: str,
    ) -> "DS24PredictionLedgerPublisher":
        required = {
            "clean_source_hash",
            "feature_authority_hash",
            "target_authority_hash",
            "static_authority_bundle_sha256",
        }
        missing = sorted(required - set(authority))
        if missing:
            raise DS24PredictionPublicationError(
                f"Prediction authority is missing fields: {missing}"
            )
        return cls(
            store=store,
            provenance=PredictionProvenance(
                source_hash=str(authority["clean_source_hash"]),
                feature_authority_hash=str(authority["feature_authority_hash"]),
                target_authority_hash=str(authority["target_authority_hash"]),
                static_bundle_hash=str(authority["static_authority_bundle_sha256"]),
                run_id=run_id,
            ),
        )

    def publish(
        self,
        predictions: pd.DataFrame,
        *,
        family: str,
        refit_id: str,
        refit_timestamp: str | pd.Timestamp,
    ) -> dict[str, object]:
        """Publish one eligible cross-section with derived evaluation eligibility."""

        return self.publish_batches(
            [predictions],
            family=family,
            refit_id=refit_id,
            refit_timestamp=refit_timestamp,
        )

    def publish_batches(
        self,
        prediction_batches: Sequence[pd.DataFrame],
        *,
        family: str,
        refit_id: str,
        refit_timestamp: str | pd.Timestamp,
    ) -> dict[str, object]:
        """Publish timestamp batches used by the current CLEAN V2 worker."""

        if not prediction_batches:
            raise DS24PredictionPublicationError("Prediction batches cannot be empty")
        ranked_batches = [
            self._prepare_batch(
                predictions,
                family=family,
                refit_id=refit_id,
                refit_timestamp=refit_timestamp,
            )
            for predictions in prediction_batches
        ]
        rows = pd.concat(ranked_batches, ignore_index=True)
        result = self.store.append(rows)
        eligibility = rows.groupby("decision_timestamp", sort=True)[
            "evaluation_eligible"
        ].first()
        result["evaluation_eligible_timestamps"] = [
            pd.Timestamp(timestamp).isoformat()
            for timestamp, eligible in eligibility.items()
            if bool(eligible)
        ]
        result["evaluation_burn_in_timestamps"] = [
            pd.Timestamp(timestamp).isoformat()
            for timestamp, eligible in eligibility.items()
            if not bool(eligible)
        ]
        return result

    def record_history_coverage(
        self,
        *,
        family: str,
        pre_ledger_refit_count: int,
        pre_ledger_score_count: int,
        pre_ledger_latest_decision_timestamp: str | None,
    ) -> dict[str, object]:
        return self.store.record_history_coverage(
            family=family,
            pre_ledger_refit_count=pre_ledger_refit_count,
            pre_ledger_score_count=pre_ledger_score_count,
            pre_ledger_latest_decision_timestamp=pre_ledger_latest_decision_timestamp,
        )

    def persisted_burn_in_timestamps(self, *, family: str) -> set[str]:
        return self.store.persisted_burn_in_timestamps(family=family)

    def resolve_durable_populations(
        self,
        *,
        family: str,
        expected_assets_by_timestamp: Mapping[str, Sequence[str]],
        refit_id: str,
        refit_timestamp: str | pd.Timestamp,
    ) -> DurablePredictionRecovery:
        """Classify existing populations without recomputing predictions.

        Complete populations are hydrated from immutable rows.  Partial,
        extra-key, or scientific-provenance mismatches fail closed before any
        caller invokes a model prediction path.
        """

        if not expected_assets_by_timestamp:
            raise DS24PredictionPublicationError(
                "Ledger-first recovery requires expected timestamp populations"
            )
        ordered_timestamps = tuple(
            pd.Timestamp(value).tz_convert("UTC").isoformat()
            for value in expected_assets_by_timestamp
        )
        durable = self.store.read_timestamp_populations(
            family=family,
            decision_timestamps=ordered_timestamps,
        )
        if not durable.empty:
            durable = durable.loc[:, PREDICTION_LEDGER_COLUMNS].copy()
            durable["decision_timestamp"] = pd.to_datetime(
                durable["decision_timestamp"], utc=True, errors="raise"
            )
        expected_refit = pd.Timestamp(refit_timestamp).tz_convert("UTC")
        recovered: dict[str, pd.DataFrame] = {}
        absent: list[str] = []
        classifications: dict[str, str] = {}
        for supplied_timestamp, expected_assets in expected_assets_by_timestamp.items():
            timestamp = pd.Timestamp(supplied_timestamp).tz_convert("UTC")
            timestamp_key = timestamp.isoformat()
            existing = (
                durable.loc[durable["decision_timestamp"] == timestamp].copy()
                if not durable.empty
                else durable.copy()
            )
            expected_keys = {str(asset) for asset in expected_assets}
            if len(expected_keys) != len(tuple(expected_assets)) or not expected_keys:
                raise DS24PredictionPublicationError(
                    f"Invalid expected prediction population: {timestamp_key}"
                )
            if existing.empty:
                classifications[timestamp_key] = LEDGER_POPULATION_ABSENT
                absent.append(timestamp_key)
                continue
            actual_keys = set(existing["asset_id"].astype(str))
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            if missing:
                raise DS24PredictionPublicationError(
                    "PARTIAL_PREDICTION_PUBLICATION_REQUIRES_EXPLICIT_RECOVERY:"
                    f"timestamp={timestamp_key},missing={len(missing)},extra={len(extra)}"
                )
            if extra:
                raise DS24PredictionPublicationError(
                    "UNEXPECTED_DURABLE_PREDICTION_KEYS:"
                    f"timestamp={timestamp_key},extra={len(extra)}"
                )
            validate_prediction_ledger_rows(existing)
            expected_fields: dict[str, object] = {
                "family": str(family),
                "refit_id": str(refit_id),
                "refit_timestamp": expected_refit,
                "evaluation_eligible": self.burn_in.is_evaluation_eligible(timestamp),
                "feature_authority_hash": self.provenance.feature_authority_hash,
                "target_authority_hash": self.provenance.target_authority_hash,
                "static_bundle_hash": self.provenance.static_bundle_hash,
                "run_id": self.provenance.run_id,
            }
            disagreements: list[str] = []
            for field, expected in expected_fields.items():
                values = existing[field]
                if field == "refit_timestamp":
                    values = pd.to_datetime(values, utc=True, errors="raise")
                if not values.eq(expected).all():
                    disagreements.append(field)
            if disagreements:
                raise DS24PredictionPublicationError(
                    "CONFLICTING_DURABLE_PREDICTION_PROVENANCE:"
                    f"timestamp={timestamp_key},fields={','.join(disagreements)}"
                )
            authoritative = existing.sort_values(
                ["cross_section_rank", "asset_id"], kind="mergesort"
            )
            recovered[timestamp_key] = pd.DataFrame(
                {
                    "family": authoritative["family"].astype(str).to_numpy(),
                    "decision_timestamp": authoritative["decision_timestamp"]
                    .map(lambda value: pd.Timestamp(value).isoformat())
                    .to_numpy(),
                    "asset_id": authoritative["asset_id"].astype(str).to_numpy(),
                    "prediction": authoritative["raw_prediction"].to_numpy(dtype=float),
                    "eligible": pd.Series(True, index=range(len(authoritative)), dtype=bool),
                }
            )
            classifications[timestamp_key] = LEDGER_POPULATION_COMPLETE_MATCHING
        return DurablePredictionRecovery(
            recovered_batches=recovered,
            absent_timestamps=tuple(absent),
            classifications=classifications,
        )

    def _prepare_batch(
        self,
        predictions: pd.DataFrame,
        *,
        family: str,
        refit_id: str,
        refit_timestamp: str | pd.Timestamp,
    ) -> pd.DataFrame:

        required = {"family", "decision_timestamp", "asset_id", "prediction", "eligible"}
        missing = sorted(required - set(predictions.columns))
        if missing:
            raise DS24PredictionPublicationError(
                f"DS24 scoring output is missing fields: {missing}"
            )
        families = set(predictions["family"].astype(str))
        if families != {family}:
            raise DS24PredictionPublicationError(
                f"Scoring output family mismatch: expected={family}, actual={sorted(families)}"
            )
        if not is_bool_dtype(predictions["eligible"].dtype):
            raise DS24PredictionPublicationError(
                "DS24 scoring output eligible must have boolean dtype"
            )
        eligible_mask = predictions["eligible"].fillna(False).astype(bool)
        eligible = predictions.loc[
            eligible_mask, ["decision_timestamp", "asset_id", "prediction"]
        ].rename(columns={"prediction": "raw_prediction"})
        if eligible.empty:
            raise DS24PredictionPublicationError(
                "DS24 scoring output contains no eligible-universe predictions"
            )
        return prepare_ranked_prediction_ledger_rows(
            eligible,
            family=family,
            refit_id=refit_id,
            refit_timestamp=refit_timestamp,
            provenance=self.provenance,
            burn_in=self.burn_in,
        )


def evaluation_batch_indexes(
    prediction_batches: Sequence[pd.DataFrame],
    publication: Mapping[str, object],
) -> tuple[int, ...]:
    """Map the publisher's authoritative eligibility decision back to batches."""

    input_timestamps: list[str] = []
    for predictions in prediction_batches:
        timestamps = pd.to_datetime(
            predictions["decision_timestamp"], utc=True, errors="raise"
        ).drop_duplicates()
        if len(timestamps) != 1:
            raise DS24PredictionPublicationError(
                "Each prediction batch must contain exactly one timestamp"
            )
        input_timestamps.append(timestamps.iloc[0].isoformat())
    if len(set(input_timestamps)) != len(input_timestamps):
        raise DS24PredictionPublicationError("Prediction batch timestamps must be unique")

    eligible = set(publication.get("evaluation_eligible_timestamps", []))
    burn_in = set(publication.get("evaluation_burn_in_timestamps", []))
    if eligible & burn_in or eligible | burn_in != set(input_timestamps):
        raise DS24PredictionPublicationError(
            "Ledger eligibility result does not exactly cover prediction batches"
        )
    return tuple(
        index
        for index, timestamp in enumerate(input_timestamps)
        if timestamp in eligible
    )
