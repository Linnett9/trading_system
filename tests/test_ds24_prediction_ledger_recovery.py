from pathlib import Path

import pandas as pd
import pytest

from application.services.ds24_prediction_ledger_service import (
    DS24PredictionLedgerPublisher,
    DS24PredictionPublicationError,
    LEDGER_POPULATION_ABSENT,
    LEDGER_POPULATION_COMPLETE_MATCHING,
)
from core.research.ml.ds24.prediction_ledger_contract import (
    PredictionLedgerContractError,
    PredictionProvenance,
    build_ranked_prediction_rows,
)
from infrastructure.data.ds24_prediction_ledger import (
    PredictionLedgerStorageError,
    PredictionLedgerStore,
)
from infrastructure.data import ds24_prediction_ledger as storage


def _rows(
    timestamp: str,
    *,
    source_hash: str = "source-before-restart",
    predictions: tuple[float, float] = (0.2, 0.1),
) -> pd.DataFrame:
    return build_ranked_prediction_rows(
        pd.DataFrame(
            {
                "decision_timestamp": [timestamp, timestamp],
                "asset_id": ["A", "B"],
                "raw_prediction": predictions,
            }
        ),
        family="random_forest",
        refit_id="stable-model-vintage",
        refit_timestamp="2017-12-27T14:35:00Z",
        evaluation_eligible=True,
        provenance=PredictionProvenance(
            source_hash=source_hash,
            feature_authority_hash="feature-authority",
            target_authority_hash="target-authority",
            static_bundle_hash="static-authority",
            run_id="R1",
        ),
    )


def _publisher(tmp_path: Path, *, source_hash: str = "source-after-repair") -> DS24PredictionLedgerPublisher:
    return DS24PredictionLedgerPublisher(
        store=PredictionLedgerStore(tmp_path),
        provenance=PredictionProvenance(
            source_hash=source_hash,
            feature_authority_hash="feature-authority",
            target_authority_hash="target-authority",
            static_bundle_hash="static-authority",
            run_id="R1",
        ),
    )


def test_unique_prediction_batch_publishes_once(tmp_path: Path) -> None:
    store = PredictionLedgerStore(tmp_path)

    result = store.append(_rows("2018-01-02T14:35:00Z"))

    assert result["rows"] == 2
    assert result["appended_parts"] == 1
    assert store._manifest()["row_count"] == 2


def test_malformed_duplicate_input_fails_before_publication(tmp_path: Path) -> None:
    store = PredictionLedgerStore(tmp_path)
    duplicate = pd.concat(
        [_rows("2018-01-02T14:35:00Z")] * 2, ignore_index=True
    )

    with pytest.raises(PredictionLedgerContractError, match="Duplicate prediction keys"):
        store.append(duplicate)

    assert not store.manifest_path.exists()


def test_identical_committed_retry_survives_operational_source_change(
    tmp_path: Path,
) -> None:
    store = PredictionLedgerStore(tmp_path)
    original = _rows("2018-01-02T14:35:00Z")
    store.append(original)

    result = store.append(
        _rows("2018-01-02T14:35:00Z", source_hash="source-after-repair")
    )

    assert result["rows"] == 0
    assert result["idempotent_parts"] == 1
    assert result["publications"][0]["idempotent_rows"] == 2
    assert result["publications"][0]["source_hash_mismatch_rows"] == 2
    assert store._manifest()["row_count"] == 2


def test_same_key_with_different_prediction_still_fails_closed(
    tmp_path: Path,
) -> None:
    store = PredictionLedgerStore(tmp_path)
    store.append(_rows("2018-01-02T14:35:00Z"))

    with pytest.raises(PredictionLedgerStorageError, match="Conflicting duplicate"):
        store.append(
            _rows(
                "2018-01-02T14:35:00Z",
                source_hash="source-after-repair",
                predictions=(0.3, 0.1),
            )
        )

    assert store._manifest()["row_count"] == 2


def test_partial_batch_retry_appends_only_uncommitted_rows(tmp_path: Path) -> None:
    store = PredictionLedgerStore(tmp_path)
    store.append(_rows("2018-01-02T14:35:00Z"))
    replay_and_new = pd.concat(
        [
            _rows("2018-01-02T14:35:00Z", source_hash="source-after-repair"),
            _rows("2018-01-02T14:40:00Z", source_hash="source-after-repair"),
        ],
        ignore_index=True,
    )

    result = store.append(replay_and_new)

    assert result["rows"] == 2
    assert result["publications"][0]["idempotent_rows"] == 2
    assert store._manifest()["row_count"] == 4


def test_complete_population_is_hydrated_before_prediction(tmp_path: Path) -> None:
    timestamp = "2018-01-02T14:35:00Z"
    PredictionLedgerStore(tmp_path).append(_rows(timestamp))

    recovery = _publisher(tmp_path).resolve_durable_populations(
        family="random_forest",
        expected_assets_by_timestamp={timestamp: ("A", "B")},
        refit_id="stable-model-vintage",
        refit_timestamp="2017-12-27T14:35:00Z",
    )

    key = pd.Timestamp(timestamp).isoformat()
    assert recovery.classifications == {
        key: LEDGER_POPULATION_COMPLETE_MATCHING
    }
    assert recovery.absent_timestamps == ()
    assert recovery.recovered_row_count == 2
    assert recovery.recovered_batches[key]["prediction"].tolist() == [0.2, 0.1]


def test_absent_population_is_classified_without_publication(tmp_path: Path) -> None:
    timestamp = "2018-01-02T14:35:00Z"

    recovery = _publisher(tmp_path).resolve_durable_populations(
        family="random_forest",
        expected_assets_by_timestamp={timestamp: ("A", "B")},
        refit_id="stable-model-vintage",
        refit_timestamp="2017-12-27T14:35:00Z",
    )

    key = pd.Timestamp(timestamp).isoformat()
    assert recovery.classifications == {key: LEDGER_POPULATION_ABSENT}
    assert recovery.absent_timestamps == (key,)
    assert recovery.recovered_batches == {}


def test_partial_durable_population_fails_closed(tmp_path: Path) -> None:
    timestamp = "2018-01-02T14:35:00Z"
    PredictionLedgerStore(tmp_path).append(_rows(timestamp))

    with pytest.raises(
        DS24PredictionPublicationError,
        match="PARTIAL_PREDICTION_PUBLICATION_REQUIRES_EXPLICIT_RECOVERY",
    ):
        _publisher(tmp_path).resolve_durable_populations(
            family="random_forest",
            expected_assets_by_timestamp={timestamp: ("A", "B", "C")},
            refit_id="stable-model-vintage",
            refit_timestamp="2017-12-27T14:35:00Z",
        )


def test_extra_durable_keys_fail_closed(tmp_path: Path) -> None:
    timestamp = "2018-01-02T14:35:00Z"
    PredictionLedgerStore(tmp_path).append(_rows(timestamp))

    with pytest.raises(
        DS24PredictionPublicationError,
        match="UNEXPECTED_DURABLE_PREDICTION_KEYS",
    ):
        _publisher(tmp_path).resolve_durable_populations(
            family="random_forest",
            expected_assets_by_timestamp={timestamp: ("A",)},
            refit_id="stable-model-vintage",
            refit_timestamp="2017-12-27T14:35:00Z",
        )


@pytest.mark.parametrize(
    ("publisher_kwargs", "refit_id", "refit_timestamp", "field"),
    [
        ({}, "different-model", "2017-12-27T14:35:00Z", "refit_id"),
        ({}, "stable-model-vintage", "2017-12-20T14:35:00Z", "refit_timestamp"),
        ({"source_hash": "operational-only-change"}, "stable-model-vintage", "2017-12-27T14:35:00Z", ""),
    ],
)
def test_recovery_provenance_contract(
    tmp_path: Path,
    publisher_kwargs: dict[str, str],
    refit_id: str,
    refit_timestamp: str,
    field: str,
) -> None:
    timestamp = "2018-01-02T14:35:00Z"
    PredictionLedgerStore(tmp_path).append(_rows(timestamp))
    publisher = _publisher(tmp_path, **publisher_kwargs)
    if not field:
        result = publisher.resolve_durable_populations(
            family="random_forest",
            expected_assets_by_timestamp={timestamp: ("A", "B")},
            refit_id=refit_id,
            refit_timestamp=refit_timestamp,
        )
        assert result.recovered_row_count == 2
        return
    with pytest.raises(
        DS24PredictionPublicationError,
        match=rf"CONFLICTING_DURABLE_PREDICTION_PROVENANCE:.*{field}",
    ):
        publisher.resolve_durable_populations(
            family="random_forest",
            expected_assets_by_timestamp={timestamp: ("A", "B")},
            refit_id=refit_id,
            refit_timestamp=refit_timestamp,
        )


def test_atomic_orphan_is_adopted_before_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamp = "2018-01-02T14:35:00Z"
    real_atomic_json = storage._atomic_json
    failed = False

    def crash_before_manifest(path: Path, payload: dict[str, object]) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("simulated crash after atomic prediction part")
        real_atomic_json(path, payload)

    monkeypatch.setattr(storage, "_atomic_json", crash_before_manifest)
    with pytest.raises(OSError, match="simulated crash"):
        PredictionLedgerStore(tmp_path).append(_rows(timestamp))
    orphan = next(tmp_path.rglob("part-*.parquet"))
    orphan_bytes = orphan.read_bytes()

    recovery = _publisher(tmp_path).resolve_durable_populations(
        family="random_forest",
        expected_assets_by_timestamp={timestamp: ("A", "B")},
        refit_id="stable-model-vintage",
        refit_timestamp="2017-12-27T14:35:00Z",
    )

    assert recovery.recovered_row_count == 2
    assert orphan.read_bytes() == orphan_bytes
    assert PredictionLedgerStore(tmp_path)._manifest()["row_count"] == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("feature_authority_hash", "changed-feature"),
        ("target_authority_hash", "changed-target"),
        ("static_bundle_hash", "changed-static"),
        ("run_id", "changed-run"),
    ],
)
def test_authority_divergence_fails_closed(
    tmp_path: Path, field: str, value: str
) -> None:
    timestamp = "2018-01-02T14:35:00Z"
    PredictionLedgerStore(tmp_path).append(_rows(timestamp))
    provenance = {
        "source_hash": "repair-source",
        "feature_authority_hash": "feature-authority",
        "target_authority_hash": "target-authority",
        "static_bundle_hash": "static-authority",
        "run_id": "R1",
    }
    provenance[field] = value
    publisher = DS24PredictionLedgerPublisher(
        store=PredictionLedgerStore(tmp_path),
        provenance=PredictionProvenance(**provenance),
    )

    with pytest.raises(
        DS24PredictionPublicationError,
        match=rf"CONFLICTING_DURABLE_PREDICTION_PROVENANCE:.*{field}",
    ):
        publisher.resolve_durable_populations(
            family="random_forest",
            expected_assets_by_timestamp={timestamp: ("A", "B")},
            refit_id="stable-model-vintage",
            refit_timestamp="2017-12-27T14:35:00Z",
        )


def test_repeated_complete_recovery_does_not_rewrite_part_or_manifest(
    tmp_path: Path,
) -> None:
    timestamp = "2018-01-02T14:35:00Z"
    store = PredictionLedgerStore(tmp_path)
    store.append(_rows(timestamp))
    part = next(tmp_path.rglob("part-*.parquet"))
    part_bytes = part.read_bytes()
    manifest_bytes = store.manifest_path.read_bytes()

    for _ in range(10):
        recovery = _publisher(tmp_path).resolve_durable_populations(
            family="random_forest",
            expected_assets_by_timestamp={timestamp: ("A", "B")},
            refit_id="stable-model-vintage",
            refit_timestamp="2017-12-27T14:35:00Z",
        )
        assert recovery.recovered_row_count == 2

    assert part.read_bytes() == part_bytes
    assert store.manifest_path.read_bytes() == manifest_bytes
