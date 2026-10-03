from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd
from pandas.api.types import is_bool_dtype

from core.research.ml.ds24.evaluation_burn_in import (
    EVALUATION_BURN_IN,
    EvaluationBurnIn,
)
from core.research.ml.ds24.prediction_ledger_contract import PredictionProvenance
from core.research.ml.ds24.research_output import prepare_ranked_prediction_ledger_rows
from infrastructure.data.ds24_prediction_ledger import PredictionLedgerStore


class DS24PredictionPublicationError(ValueError):
    """Raised when a scoring result cannot be safely published to the ledger."""


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
