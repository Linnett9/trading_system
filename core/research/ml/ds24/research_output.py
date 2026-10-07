from __future__ import annotations

import pandas as pd

from core.research.ml.ds24.evaluation_burn_in import (
    EVALUATION_BURN_IN,
    EvaluationBurnIn,
)
from core.research.ml.ds24.prediction_ledger_contract import (
    PredictionProvenance,
    build_ranked_prediction_rows,
)


def prepare_ranked_prediction_ledger_rows(
    predictions: pd.DataFrame,
    *,
    family: str,
    refit_id: str,
    refit_timestamp: str | pd.Timestamp,
    provenance: PredictionProvenance,
    burn_in: EvaluationBurnIn = EVALUATION_BURN_IN,
) -> pd.DataFrame:
    """Prepare DS24 ledger rows using the shared evaluation-boundary authority."""

    if "decision_timestamp" not in predictions or predictions.empty:
        raise ValueError("A non-empty decision_timestamp column is required")
    decisions = pd.to_datetime(predictions["decision_timestamp"], utc=True, errors="raise")
    if decisions.nunique() != 1:
        raise ValueError("One ledger preparation call must contain one decision timestamp")
    return build_ranked_prediction_rows(
        predictions,
        family=family,
        refit_id=refit_id,
        refit_timestamp=refit_timestamp,
        evaluation_eligible=burn_in.is_evaluation_eligible(decisions.iloc[0]),
        provenance=provenance,
    )
