from __future__ import annotations

from pathlib import Path

import pandas as pd

from application.services.ds24_prediction_ledger_service import (
    DS24PredictionLedgerPublisher,
)
from core.research.ml.ds24.clean_v2_contracts import file_sha256
from core.research.ml.ds24.prediction_ledger_contract import PredictionProvenance
from scripts.local.ds24_clean_v2_family_worker import (
    _recover_or_score_prediction_batches,
)


ROOT = Path(__file__).resolve().parents[1]
FORENSIC_PART = ROOT / (
    "docs/audits/ds24_rf_immutable_ledger_forensic_recovery_20261007/"
    "frozen_evidence/existing_rf_20190408_part.parquet"
)
EXPECTED_PART_SHA256 = (
    "d5d2e304f56b0c1eb8b77b913069134219e4dcec6144f8e65d9919f7c898ebf5"
)
FAILURE_TIMESTAMP = pd.Timestamp("2019-04-08T13:40:00Z")


class _FrozenPopulationStore:
    def __init__(self, rows: pd.DataFrame) -> None:
        self.rows = rows

    def read_timestamp_populations(
        self,
        *,
        family: str,
        decision_timestamps: object,
    ) -> pd.DataFrame:
        del decision_timestamps
        return self.rows.loc[self.rows["family"].astype(str) == family].copy()


def test_real_april_8_population_is_recovered_without_predict() -> None:
    assert file_sha256(FORENSIC_PART) == EXPECTED_PART_SHA256
    part = pd.read_parquet(FORENSIC_PART)
    decisions = pd.to_datetime(part["decision_timestamp"], utc=True)
    existing = part.loc[decisions == FAILURE_TIMESTAMP].copy()
    assert len(existing) == 482
    first = existing.iloc[0]
    publisher = DS24PredictionLedgerPublisher(
        store=_FrozenPopulationStore(existing),  # type: ignore[arg-type]
        provenance=PredictionProvenance(
            source_hash="repair-source-operational-only",
            feature_authority_hash=str(first["feature_authority_hash"]),
            target_authority_hash=str(first["target_authority_hash"]),
            static_bundle_hash=str(first["static_bundle_hash"]),
            run_id=str(first["run_id"]),
        ),
    )
    score_population = pd.DataFrame(
        {
            "decision_timestamp": [FAILURE_TIMESTAMP] * len(existing),
            "asset_id": existing["asset_id"].astype(str).tolist(),
            "unused_predictor": 0.0,
        }
    )

    batches, recovery = _recover_or_score_prediction_batches(
        publisher=publisher,
        family="random_forest",
        model=object(),  # prediction would fail if ledger-first recovery regressed
        score_batch=score_population,
        full_panel=score_population,
        predictors=["unused_predictor"],
        metadata={},
        timestamp_batch=(FAILURE_TIMESTAMP,),
        expected_assets_by_timestamp={
            FAILURE_TIMESTAMP.isoformat(): tuple(
                sorted(existing["asset_id"].astype(str))
            )
        },
        refit_id=str(first["refit_id"]),
        refit_timestamp=pd.Timestamp(first["refit_timestamp"]),
    )

    assert recovery.recovered_row_count == 482
    assert recovery.absent_timestamps == ()
    assert len(batches) == 1
    assert batches[0]["asset_id"].tolist() == (
        existing.sort_values(["cross_section_rank", "asset_id"], kind="mergesort")[
            "asset_id"
        ]
        .astype(str)
        .tolist()
    )
