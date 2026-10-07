from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final, Mapping

import numpy as np
import pandas as pd


PREDICTION_LEDGER_SCHEMA_VERSION: Final = "DS24_RANKED_PREDICTION_LEDGER_V1"
RESEARCH_OUTPUT_POLICY_VERSION: Final = "DS24_RESEARCH_OUTPUT_POLICY_V2"
PREDICTION_KEY: Final = ("family", "decision_timestamp", "asset_id")
PREDICTION_LEDGER_COLUMNS: Final = (
    "decision_timestamp",
    "asset_id",
    "family",
    "raw_prediction",
    "cross_section_rank",
    "rank_percentile",
    "eligible_universe_size",
    "refit_id",
    "refit_timestamp",
    "evaluation_eligible",
    "source_hash",
    "feature_authority_hash",
    "target_authority_hash",
    "static_bundle_hash",
    "run_id",
)
FORBIDDEN_LEDGER_COLUMNS: Final = frozenset(
    {
        "target",
        "target_value",
        "realized_return",
        "realised_return",
        "forward_return",
        "actual_return",
        "label",
    }
)
_SAFE_FAMILY_ID: Final = re.compile(r"^[A-Za-z0-9_.-]+$")


class PredictionLedgerContractError(ValueError):
    """Raised when ranked prediction rows violate the shared ledger contract."""


@dataclass(frozen=True)
class PredictionProvenance:
    source_hash: str
    feature_authority_hash: str
    target_authority_hash: str
    static_bundle_hash: str
    run_id: str

    def __post_init__(self) -> None:
        for field_name, value in self.__dict__.items():
            if not str(value).strip():
                raise PredictionLedgerContractError(
                    f"Prediction provenance {field_name} must be non-empty"
                )

    def as_columns(self) -> Mapping[str, str]:
        return dict(self.__dict__)


def build_ranked_prediction_rows(
    predictions: pd.DataFrame,
    *,
    family: str,
    refit_id: str,
    refit_timestamp: str | pd.Timestamp,
    evaluation_eligible: bool,
    provenance: PredictionProvenance,
) -> pd.DataFrame:
    """Return one validated, deterministically ranked eligible cross-section.

    Rank one is the highest score. Score ties are broken by ascending stable
    asset identity, matching the established Dell metrics writer. Percentile
    is derived from that total rank and has the highest score at 1.
    """

    required = {"decision_timestamp", "asset_id", "raw_prediction"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise PredictionLedgerContractError(f"Missing prediction columns: {missing}")
    forbidden = sorted(FORBIDDEN_LEDGER_COLUMNS & set(predictions.columns))
    if forbidden:
        raise PredictionLedgerContractError(
            f"Target/outcome columns are forbidden in the prediction ledger: {forbidden}"
        )
    if predictions.empty:
        raise PredictionLedgerContractError("An eligible prediction cross-section cannot be empty")
    if not _SAFE_FAMILY_ID.fullmatch(str(family)):
        raise PredictionLedgerContractError(
            "family must be a non-empty filesystem-safe identifier"
        )
    if not str(refit_id).strip():
        raise PredictionLedgerContractError("refit_id must be non-empty")

    frame = predictions.loc[:, list(required)].copy()
    frame["decision_timestamp"] = pd.to_datetime(
        frame["decision_timestamp"], utc=True, errors="raise"
    )
    if frame["decision_timestamp"].nunique() != 1:
        raise PredictionLedgerContractError(
            "Each ranked cross-section must contain exactly one decision timestamp"
        )
    frame["asset_id"] = frame["asset_id"].astype(str)
    if frame["asset_id"].str.strip().eq("").any():
        raise PredictionLedgerContractError("asset_id must be non-empty")
    if frame["asset_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["asset_id"].duplicated(False), "asset_id"].unique())
        raise PredictionLedgerContractError(f"Duplicate eligible assets: {duplicates}")
    frame["raw_prediction"] = pd.to_numeric(frame["raw_prediction"], errors="raise")
    if not np.isfinite(frame["raw_prediction"].to_numpy(dtype=float)).all():
        raise PredictionLedgerContractError("raw_prediction must be finite")

    frame = frame.sort_values(
        ["raw_prediction", "asset_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    frame["cross_section_rank"] = np.arange(1, len(frame) + 1, dtype=np.int32)
    frame["rank_percentile"] = (
        (len(frame) - frame["cross_section_rank"] + 1) / len(frame)
    )
    frame["eligible_universe_size"] = np.int32(len(frame))
    frame["family"] = str(family)
    frame["refit_id"] = str(refit_id)
    frame["refit_timestamp"] = pd.to_datetime(refit_timestamp, utc=True, errors="raise")
    frame["evaluation_eligible"] = bool(evaluation_eligible)
    for column, value in provenance.as_columns().items():
        frame[column] = value

    frame = frame.loc[:, PREDICTION_LEDGER_COLUMNS]
    validate_prediction_ledger_rows(frame)
    return frame


def validate_prediction_ledger_rows(rows: pd.DataFrame) -> None:
    missing = sorted(set(PREDICTION_LEDGER_COLUMNS) - set(rows.columns))
    if missing:
        raise PredictionLedgerContractError(f"Missing ledger columns: {missing}")
    forbidden = sorted(FORBIDDEN_LEDGER_COLUMNS & set(rows.columns))
    if forbidden:
        raise PredictionLedgerContractError(f"Forbidden ledger columns: {forbidden}")
    if rows.empty:
        raise PredictionLedgerContractError("Prediction ledger rows cannot be empty")
    if rows.duplicated(list(PREDICTION_KEY)).any():
        raise PredictionLedgerContractError(f"Duplicate prediction keys: {PREDICTION_KEY}")
    unsafe_families = sorted(
        family
        for family in set(rows["family"].astype(str))
        if not _SAFE_FAMILY_ID.fullmatch(family)
    )
    if unsafe_families:
        raise PredictionLedgerContractError(
            f"Unsafe prediction-ledger family identifiers: {unsafe_families}"
        )

    decisions = pd.to_datetime(rows["decision_timestamp"], utc=True, errors="raise")
    refits = pd.to_datetime(rows["refit_timestamp"], utc=True, errors="raise")
    if (refits > decisions).any():
        raise PredictionLedgerContractError("refit_timestamp cannot follow decision_timestamp")
    scores = pd.to_numeric(rows["raw_prediction"], errors="raise").to_numpy(dtype=float)
    ranks = pd.to_numeric(rows["cross_section_rank"], errors="raise").to_numpy(dtype=float)
    percentiles = pd.to_numeric(rows["rank_percentile"], errors="raise").to_numpy(dtype=float)
    sizes = pd.to_numeric(rows["eligible_universe_size"], errors="raise").to_numpy(dtype=int)
    if not np.isfinite(scores).all() or not np.isfinite(ranks).all():
        raise PredictionLedgerContractError("Predictions and ranks must be finite")
    if ((percentiles <= 0.0) | (percentiles > 1.0)).any():
        raise PredictionLedgerContractError("rank_percentile must be in (0, 1]")

    work = rows.assign(_decision=decisions)
    for (_, _), group in work.groupby(["family", "_decision"], sort=False):
        expected_size = len(group)
        if set(group["eligible_universe_size"].astype(int)) != {expected_size}:
            raise PredictionLedgerContractError(
                "eligible_universe_size must equal the complete cross-section row count"
            )
        expected = group.sort_values(
            ["raw_prediction", "asset_id"], ascending=[False, True], kind="mergesort"
        ).copy()
        expected["_rank"] = np.arange(1, expected_size + 1)
        expected["_percentile"] = (expected_size - expected["_rank"] + 1) / expected_size
        actual = group.set_index("asset_id")
        expected = expected.set_index("asset_id")
        if not np.array_equal(
            actual.loc[expected.index, "cross_section_rank"].to_numpy(dtype=int),
            expected["_rank"].to_numpy(dtype=int),
        ):
            raise PredictionLedgerContractError("cross_section_rank is not deterministic")
        if not np.allclose(
            actual.loc[expected.index, "rank_percentile"],
            expected["_percentile"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise PredictionLedgerContractError("rank_percentile is not deterministic")
    if (sizes <= 0).any():
        raise PredictionLedgerContractError("eligible_universe_size must be positive")
