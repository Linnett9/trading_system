from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from application.services.ds24_prediction_ledger_service import (
    DS24PredictionLedgerPublisher,
    DS24PredictionPublicationError,
    evaluation_batch_indexes,
)
from core.research.ml.ds24.evaluation_burn_in import (
    DELL_EVALUATION_START,
    EVALUATION_BURN_IN,
)
from core.research.ml.ds24.prediction_ledger_contract import (
    PredictionProvenance,
    build_ranked_prediction_rows,
)
from core.research.ml.ds24.research_output import prepare_ranked_prediction_ledger_rows
from infrastructure.data.ds24_prediction_ledger import (
    PredictionLedgerReader,
    PredictionLedgerStore,
)


def _scores(timestamp: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_timestamp": [timestamp, timestamp],
            "asset_id": ["A", "B"],
            "raw_prediction": [0.2, 0.1],
        }
    )


def _provenance() -> PredictionProvenance:
    return PredictionProvenance("source", "features", "targets", "bundle", "run")


def test_burn_in_boundary_is_exact_and_timestamp_based() -> None:
    assert DELL_EVALUATION_START == pd.Timestamp("2018-01-02T14:35:00Z")
    assert not EVALUATION_BURN_IN.is_evaluation_eligible("2018-01-02T14:30:00Z")
    assert EVALUATION_BURN_IN.is_evaluation_eligible("2018-01-02T14:35:00Z")


def test_predictions_are_ranked_and_persistable_during_burn_in() -> None:
    rows = prepare_ranked_prediction_ledger_rows(
        _scores("2017-12-29T21:00:00Z"),
        family="ridge",
        refit_id="refit",
        refit_timestamp="2017-12-29T20:00:00Z",
        provenance=_provenance(),
    )

    assert len(rows) == 2
    assert not rows["evaluation_eligible"].any()
    assert rows["cross_section_rank"].tolist() == [1.0, 2.0]


def test_returns_and_rank_ic_exclude_burn_in_but_keep_boundary() -> None:
    before = _scores("2017-12-29T21:00:00Z")
    boundary = _scores("2018-01-02T14:35:00Z")
    rows = pd.concat([before, boundary], ignore_index=True)

    returns = EVALUATION_BURN_IN.metric_population(rows, metric="returns")
    rank_ic = EVALUATION_BURN_IN.metric_population(rows, metric="rank_ic")

    assert len(returns) == 2
    assert len(rank_ic) == 2
    assert set(returns["decision_timestamp"]) == {"2018-01-02T14:35:00Z"}


def test_burn_in_contract_has_no_target_or_holdout_reader() -> None:
    fields = set(EVALUATION_BURN_IN.__dataclass_fields__)
    assert "target" not in fields
    assert "holdout" not in fields
    marked = EVALUATION_BURN_IN.mark_predictions(_scores("2017-01-03T14:35:00Z"))
    assert set(marked.columns) == {
        "decision_timestamp",
        "asset_id",
        "raw_prediction",
        "evaluation_eligible",
    }


def test_application_publisher_derives_eligibility_and_full_universe(tmp_path) -> None:
    publisher = DS24PredictionLedgerPublisher.from_authority(
        store=PredictionLedgerStore(tmp_path),
        authority={
            "clean_source_hash": "source",
            "feature_authority_hash": "features",
            "target_authority_hash": "targets",
            "static_authority_bundle_sha256": "bundle",
        },
        run_id="run",
    )
    scoring_output = _scores("2017-12-29T21:00:00Z").rename(
        columns={"raw_prediction": "prediction"}
    )
    scoring_output["family"] = "ridge"
    scoring_output["eligible"] = True

    result = publisher.publish(
        scoring_output,
        family="ridge",
        refit_id="refit",
        refit_timestamp="2017-12-29T20:00:00Z",
    )
    loaded = PredictionLedgerReader(tmp_path).read()

    assert result["rows"] == 2
    assert len(loaded) == 2
    assert not loaded["evaluation_eligible"].any()
    assert "target_value" not in loaded


def test_application_publisher_rejects_empty_eligible_universe(tmp_path) -> None:
    publisher = DS24PredictionLedgerPublisher(
        PredictionLedgerStore(tmp_path), _provenance()
    )
    scoring_output = _scores("2018-01-02T14:35:00Z").rename(
        columns={"raw_prediction": "prediction"}
    )
    scoring_output["family"] = "ridge"
    scoring_output["eligible"] = False

    with pytest.raises(DS24PredictionPublicationError, match="no eligible"):
        publisher.publish(
            scoring_output,
            family="ridge",
            refit_id="refit",
            refit_timestamp="2018-01-01T20:00:00Z",
        )


def test_application_publisher_accepts_current_worker_timestamp_batches(tmp_path) -> None:
    publisher = DS24PredictionLedgerPublisher(
        PredictionLedgerStore(tmp_path), _provenance()
    )
    batches = []
    for timestamp in ("2017-12-29T20:00:00Z", "2018-01-02T14:35:00Z"):
        batch = _scores(timestamp).rename(columns={"raw_prediction": "prediction"})
        batch["family"] = "ridge"
        batch["eligible"] = True
        batches.append(batch)

    result = publisher.publish_batches(
        batches,
        family="ridge",
        refit_id="refit",
        refit_timestamp="2017-12-29T19:00:00Z",
    )
    loaded = PredictionLedgerReader(tmp_path).read()

    assert result["rows"] == 4
    assert evaluation_batch_indexes(batches, result) == (1,)
    assert loaded.groupby("decision_timestamp")["evaluation_eligible"].first().tolist() == [
        False,
        True,
    ]


def test_application_publisher_fails_closed_on_string_eligibility(tmp_path) -> None:
    publisher = DS24PredictionLedgerPublisher(
        PredictionLedgerStore(tmp_path), _provenance()
    )
    scoring_output = _scores("2018-01-02T14:35:00Z").rename(
        columns={"raw_prediction": "prediction"}
    )
    scoring_output["family"] = "ridge"
    scoring_output["eligible"] = "False"

    with pytest.raises(DS24PredictionPublicationError, match="boolean dtype"):
        publisher.publish(
            scoring_output,
            family="ridge",
            refit_id="refit",
            refit_timestamp="2018-01-01T20:00:00Z",
        )


def test_future_mock_family_automatically_uses_shared_publication(tmp_path) -> None:
    publisher = DS24PredictionLedgerPublisher(
        PredictionLedgerStore(tmp_path), _provenance()
    )
    predictions = _scores("2018-01-02T14:35:00Z").rename(
        columns={"raw_prediction": "prediction"}
    )
    predictions["family"] = "future_registered_mock"
    predictions["eligible"] = True

    result = publisher.publish_batches(
        [predictions],
        family="future_registered_mock",
        refit_id="mock-refit",
        refit_timestamp="2018-01-01T20:00:00Z",
    )
    loaded = PredictionLedgerReader(tmp_path).read()

    assert result["rows"] == 2
    assert set(loaded["family"]) == {"future_registered_mock"}
    assert evaluation_batch_indexes([predictions], result) == (0,)


def test_current_source_multi_family_publication_is_generic_and_idempotent(
    tmp_path: Path,
) -> None:
    publisher = DS24PredictionLedgerPublisher(
        PredictionLedgerStore(tmp_path), _provenance()
    )
    publications = {}
    for family in ("random_forest", "huber", "future_registered_mock"):
        predictions = _scores("2017-10-26T13:35:00Z").rename(
            columns={"raw_prediction": "prediction"}
        )
        predictions["family"] = family
        predictions["eligible"] = True
        publications[family] = publisher.publish_batches(
            [predictions],
            family=family,
            refit_id=f"{family}-refit",
            refit_timestamp="2017-10-23T13:35:00Z",
        )

    retry = publisher.publish_batches(
        [
            _scores("2017-10-26T13:35:00Z")
            .rename(columns={"raw_prediction": "prediction"})
            .assign(family="random_forest", eligible=True)
        ],
        family="random_forest",
        refit_id="random_forest-refit",
        refit_timestamp="2017-10-23T13:35:00Z",
    )
    loaded = PredictionLedgerReader(tmp_path).read()

    assert {family: result["rows"] for family, result in publications.items()} == {
        "random_forest": 2,
        "huber": 2,
        "future_registered_mock": 2,
    }
    assert retry["idempotent_parts"] == 1
    assert set(loaded["family"]) == {
        "random_forest",
        "huber",
        "future_registered_mock",
    }
    assert not loaded["evaluation_eligible"].any()
    assert "target_value" not in loaded and "realized_return" not in loaded
