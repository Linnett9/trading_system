from __future__ import annotations

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import socket

import pandas as pd
import pytest

from core.research.ml.ds24.prediction_ledger_contract import (
    FORBIDDEN_LEDGER_COLUMNS,
    PREDICTION_KEY,
    PREDICTION_LEDGER_COLUMNS,
    PredictionLedgerContractError,
    PredictionProvenance,
    build_ranked_prediction_rows,
)
from core.research.ml.ds24.prediction_ledger_queries import LedgerQuery
from infrastructure.data import ds24_prediction_ledger as storage
from infrastructure.data.ds24_prediction_ledger import (
    PredictionLedgerReader,
    PredictionLedgerStorageError,
    PredictionLedgerStore,
)


def _provenance() -> PredictionProvenance:
    return PredictionProvenance(
        source_hash="clean-source-hash",
        feature_authority_hash="feature-authority-hash",
        target_authority_hash="target-authority-hash",
        static_bundle_hash="static-bundle-hash",
        run_id="synthetic-run",
    )


def _ranked(
    timestamp: str = "2018-01-02T14:35:00Z",
    *,
    scores: tuple[float, ...] = (0.2, 0.9, 0.2, -0.1),
    assets: tuple[str, ...] = ("B", "A", "C", "D"),
    evaluation_eligible: bool = True,
) -> pd.DataFrame:
    return build_ranked_prediction_rows(
        pd.DataFrame(
            {
                "decision_timestamp": [timestamp] * len(scores),
                "asset_id": assets,
                "raw_prediction": scores,
            }
        ),
        family="ridge",
        refit_id="ridge-2017-12-29",
        refit_timestamp="2017-12-29T21:00:00Z",
        evaluation_eligible=evaluation_eligible,
        provenance=_provenance(),
    )


def test_deterministic_ranks_ties_percentiles_and_full_universe_count() -> None:
    ranked = _ranked()

    assert ranked["asset_id"].tolist() == ["A", "B", "C", "D"]
    assert ranked["cross_section_rank"].tolist() == [1, 2, 3, 4]
    assert ranked["rank_percentile"].tolist() == [1.0, 0.75, 0.5, 0.25]
    assert ranked["eligible_universe_size"].tolist() == [4, 4, 4, 4]


def test_rank_output_is_independent_of_input_order() -> None:
    first = _ranked()
    shuffled_input = pd.DataFrame(
        {
            "decision_timestamp": ["2018-01-02T14:35:00Z"] * 4,
            "asset_id": ["D", "C", "A", "B"],
            "raw_prediction": [-0.1, 0.2, 0.9, 0.2],
        }
    )
    second = build_ranked_prediction_rows(
        shuffled_input,
        family="ridge",
        refit_id="ridge-2017-12-29",
        refit_timestamp="2017-12-29T21:00:00Z",
        evaluation_eligible=True,
        provenance=_provenance(),
    )

    pd.testing.assert_frame_equal(first, second)


def test_target_and_realized_values_are_rejected_and_absent() -> None:
    predictions = pd.DataFrame(
        {
            "decision_timestamp": ["2018-01-02T14:35:00Z"],
            "asset_id": ["A"],
            "raw_prediction": [0.1],
            "target_value": [99.0],
        }
    )
    with pytest.raises(PredictionLedgerContractError, match="forbidden"):
        build_ranked_prediction_rows(
            predictions,
            family="ridge",
            refit_id="refit",
            refit_timestamp="2018-01-01T00:00:00Z",
            evaluation_eligible=True,
            provenance=_provenance(),
        )

    assert not (set(PREDICTION_LEDGER_COLUMNS) & FORBIDDEN_LEDGER_COLUMNS)


def test_family_partition_identity_must_be_filesystem_safe() -> None:
    with pytest.raises(PredictionLedgerContractError, match="filesystem-safe"):
        build_ranked_prediction_rows(
            pd.DataFrame(
                {
                    "decision_timestamp": ["2018-01-02T14:35:00Z"],
                    "asset_id": ["A"],
                    "raw_prediction": [0.1],
                }
            ),
            family="../outside",
            refit_id="refit",
            refit_timestamp="2018-01-01T00:00:00Z",
            evaluation_eligible=True,
            provenance=_provenance(),
        )


def test_provenance_is_present_on_every_row() -> None:
    ranked = _ranked()
    for column, expected in _provenance().as_columns().items():
        assert set(ranked[column]) == {expected}


def test_append_is_atomic_partitioned_and_restart_idempotent(tmp_path: Path) -> None:
    store = PredictionLedgerStore(tmp_path)
    rows = _ranked()

    first = store.append(rows)
    second = PredictionLedgerStore(tmp_path).append(rows)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert first["appended_parts"] == 1
    assert second["appended_parts"] == 0
    assert second["idempotent_parts"] == 1
    assert manifest["row_count"] == 4
    assert manifest["parts"][0]["path"].startswith("family=ridge/period=2018-01/")
    assert not list(tmp_path.rglob("*.partial"))


def test_history_coverage_records_incomplete_boundary_and_first_ledger_timestamp(
    tmp_path: Path,
) -> None:
    store = PredictionLedgerStore(tmp_path)
    coverage = store.record_history_coverage(
        family="ridge",
        pre_ledger_refit_count=7,
        pre_ledger_score_count=2363,
        pre_ledger_latest_decision_timestamp="2016-03-23T19:00:00",
    )
    store.append(_ranked())
    store.append(_ranked("2017-12-29T21:00:00Z", evaluation_eligible=False))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    persisted = manifest["history_coverage"]["ridge"]

    assert not coverage["ledger_history_complete"]
    assert coverage["historical_reconstruction"] == "MODEL_RETRAIN_REQUIRED"
    assert persisted["pre_ledger_score_count"] == 2363
    assert persisted["pre_ledger_latest_decision_timestamp"] == "2016-03-23T19:00:00+00:00"
    assert persisted["first_durable_ledger_timestamp"] == "2017-12-29T21:00:00+00:00"
    assert store.persisted_burn_in_timestamps(family="ridge") == {
        "2017-12-29T21:00:00+00:00"
    }


def test_burn_in_completion_reader_is_family_scoped(tmp_path: Path) -> None:
    store = PredictionLedgerStore(tmp_path)
    ridge = _ranked("2017-12-29T21:00:00Z", evaluation_eligible=False)
    momentum = ridge.copy()
    momentum["family"] = "momentum"
    momentum["refit_id"] = "momentum-control"
    store.append(pd.concat([ridge, momentum], ignore_index=True))

    assert store.persisted_burn_in_timestamps(family="ridge") == {
        "2017-12-29T21:00:00+00:00"
    }


def test_duplicate_keys_with_different_content_are_rejected(tmp_path: Path) -> None:
    store = PredictionLedgerStore(tmp_path)
    store.append(_ranked())
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    original_path = tmp_path / manifest["parts"][0]["path"]
    original_bytes = original_path.read_bytes()
    changed = _ranked(scores=(0.3, 0.9, 0.2, -0.1))

    with pytest.raises(PredictionLedgerStorageError, match="Duplicate"):
        store.append(changed)
    assert original_path.read_bytes() == original_bytes


def test_manifest_lock_serializes_two_family_writers(tmp_path: Path) -> None:
    ridge = _ranked()
    momentum = ridge.copy()
    momentum["family"] = "momentum"
    momentum["refit_id"] = "momentum-control"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda rows: PredictionLedgerStore(tmp_path).append(rows), [ridge, momentum])
        )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert sum(result["appended_parts"] for result in results) == 2
    assert manifest["row_count"] == 8
    assert {part["family"] for part in manifest["parts"]} == {"ridge", "momentum"}
    assert not (tmp_path / ".manifest.lock").exists()


def test_abandoned_local_lock_is_recovered_deterministically(tmp_path: Path) -> None:
    lock_path = tmp_path / ".manifest.lock"
    tmp_path.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "hostname": socket.gethostname(),
                "pid": 2147483647,
                "token": "abandoned",
                "created_at_utc": "2000-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    result = PredictionLedgerStore(tmp_path).append(_ranked())

    assert result["appended_parts"] == 1
    assert not lock_path.exists()


def test_crash_after_part_publication_recovers_without_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_atomic_json = storage._atomic_json
    calls = 0

    def fail_first_manifest(path: Path, payload: dict[str, object]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated crash after immutable part publication")
        real_atomic_json(path, payload)

    monkeypatch.setattr(storage, "_atomic_json", fail_first_manifest)
    with pytest.raises(OSError, match="simulated crash"):
        PredictionLedgerStore(tmp_path).append(_ranked())

    result = PredictionLedgerStore(tmp_path).append(_ranked())
    loaded = PredictionLedgerReader(tmp_path).read()
    assert result["appended_parts"] == 1
    assert len(loaded) == 4
    assert not loaded.duplicated(list(PREDICTION_KEY)).any()


def test_read_only_queries_and_post_load_research_helpers(tmp_path: Path) -> None:
    store = PredictionLedgerStore(tmp_path)
    first = _ranked()
    second = _ranked(
        "2018-01-02T14:40:00Z", scores=(0.95, 0.1, 0.3, -0.2)
    )
    store.append(pd.concat([first, second], ignore_index=True))
    reader = PredictionLedgerReader(tmp_path)

    top = reader.read(LedgerQuery(top_n=1))
    bottom = reader.read(LedgerQuery(bottom_n=1))
    percentile = reader.read(LedgerQuery(minimum_rank_percentile=0.75))
    threshold = reader.read(LedgerQuery(minimum_score=0.9))

    assert len(top) == len(bottom) == 2
    assert len(percentile) == 4
    assert set(threshold["raw_prediction"]) == {0.9, 0.95}
    assert not reader.rank_changes(reader.read())["rank_change"].dropna().empty
    assert (reader.score_spreads(reader.read())["score_spread"] > 0).all()
    assert not reader.rebalance_candidates(reader.read(), minimum_rank_change=1.0).empty


def test_certified_outcomes_are_joined_later_not_stored(tmp_path: Path) -> None:
    rows = _ranked()
    PredictionLedgerStore(tmp_path).append(rows)
    loaded = PredictionLedgerReader(tmp_path).read()
    outcomes = loaded[["decision_timestamp", "asset_id"]].copy()
    outcomes["return_3h"] = range(len(outcomes))

    joined = PredictionLedgerReader.join_certified_outcomes(
        loaded,
        outcomes,
        authority_id="certified-price-authority-hash",
        holding_period="3h",
        outcome_column="return_3h",
    )

    assert "return_3h" not in loaded
    assert set(joined["outcome_authority_id"]) == {"certified-price-authority-hash"}
    assert set(joined["holding_period"]) == {"3h"}
