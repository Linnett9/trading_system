from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.research.ml.ds24.clean_v2_certification import (
    certification_cases,
    synthetic_multiyear_raw_frames,
)
from core.research.ml.ds24.clean_v2_contracts import CONFIG_ROOT, authority_bundle, load_contract
from core.research.ml.ds24.clean_v2_data import CleanV2CompositeData
from core.research.ml.ds24.clean_v2_features import (
    REPAIRED_FEATURES,
    SOURCE_TIMESTAMP_SUFFIX,
    certify_future_bar_invariance,
    compose_v2_stock_features,
    compute_repaired_session_features,
)
from core.research.ml.ds24.clean_v2_runtime import (
    LIVE_ORDERS_ALLOWED,
    PAPER_ORDERS_ALLOWED,
    QUALIFIER_YEARS,
    REFIT_POLICY_ID,
    CleanRuntimeError,
    FamilyResumeState,
    assert_compact_output_path,
    clean_refit_schedule,
    metric_key,
    validate_ownership,
    validate_target_use,
)
from scripts.local.ds24_clean_v2_build_sidecar import _publish_partition, _read_raw_context
from scripts.local.ds24_clean_v2_certify import _synthetic_semantic_checks
from scripts.local.ds24_clean_v2_family_worker import build_clean_refit_schedule
from scripts.local.ds24_clean_v2_mac_preflight import (
    clean_source_hash,
    partition_file_hashes_match,
)
from scripts.local.ds24_clean_v2_supervisor import (
    _launch_families,
    _partition_inventory_matches,
)
from scripts.local import ds24_clean_v2_build_target_delta as target_delta_builder


def test_ordered_101_predictor_contract_is_exact_and_target_free() -> None:
    manifest = load_contract("predictor_manifest.json")
    predictors = manifest["predictors"]
    digest = hashlib.sha256(
        json.dumps(predictors, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert len(predictors) == 101
    assert len(set(predictors)) == 101
    assert digest == "22db0fcfe1219a30e0f6926dd8f0af11b4cd8ead2a6462a0df278549fadeb5a0"
    assert manifest["source_manifest_sha256"] == "d21722c41f0e54edc02519184a10596a2847091f0c04e80e4a0503251ffb60ca"
    assert not any("target" in name.lower() or "forward_return" in name.lower() for name in predictors)


def test_repaired_cross_session_formulas_and_source_timestamps_are_exact() -> None:
    frame = synthetic_multiyear_raw_frames()["AAA"]
    repaired = compute_repaired_session_features(frame)
    sessions = frame["session_date"].drop_duplicates().tolist()
    current_session = sessions[2]
    rth = frame["session_type"].isin(["REGULAR", "EARLY_CLOSE"])
    current_indices = frame.index[(frame["session_date"] == current_session) & rth]
    row_index = current_indices[10]
    prior = frame[(frame["session_date"] == sessions[1]) & rth]
    two_back = frame[(frame["session_date"] == sessions[0]) & rth]
    current = frame.loc[row_index]

    assert repaired.loc[row_index, "overnight_gap"] == pytest.approx(
        frame.loc[current_indices[0], "open"] / prior.iloc[-1]["close"] - 1.0
    )
    assert repaired.loc[row_index, "previous_session_return"] == pytest.approx(
        prior.iloc[-1]["close"] / prior.iloc[0]["open"] - 1.0
    )
    assert repaired.loc[row_index, "two_session_return"] == pytest.approx(
        current["close"] / two_back.iloc[-1]["close"] - 1.0
    )
    decision = pd.Timestamp(current["timestamp_utc"]) + pd.Timedelta(minutes=5)
    for feature in REPAIRED_FEATURES:
        source = repaired.loc[row_index, f"{feature}{SOURCE_TIMESTAMP_SUFFIX}"]
        assert pd.isna(source) or pd.Timestamp(source) <= decision


def test_opening_window_values_are_null_until_sixth_bar_is_finalized() -> None:
    frame = synthetic_multiyear_raw_frames()["AAA"]
    repaired = compute_repaired_session_features(frame)
    indices = frame.index[
        (frame["session_date"] == "2017-11-22")
        & (frame["session_type"].isin(["REGULAR", "EARLY_CLOSE"]))
    ]
    assert repaired.loc[indices[:5], "opening_range_position"].isna().all()
    assert repaired.loc[indices[:5], "session_return_30m"].isna().all()
    assert repaired.loc[indices[:5], "opening_return_30m"].isna().all()
    sixth = indices[5]
    session = frame.loc[indices]
    assert repaired.loc[sixth, "session_return_30m"] == pytest.approx(0.0)
    assert repaired.loc[sixth, "opening_return_30m"] == pytest.approx(
        session.iloc[5]["close"] / session.iloc[0]["open"] - 1.0
    )


def test_extended_hours_do_not_define_session_close_or_calendar_state() -> None:
    frame = synthetic_multiyear_raw_frames()["AAA"]
    repaired = compute_repaired_session_features(frame)
    rth = frame[
        (frame["session_date"] == "2017-11-22")
        & (frame["session_type"] == "REGULAR")
    ]
    first = rth.index[0]
    last = rth.index[-1]

    assert repaired.loc[first, "minutes_since_open"] == pytest.approx(5.0)
    assert repaired.loc[first, "minutes_until_close"] == pytest.approx(385.0)
    assert repaired.loc[last, "minutes_until_close"] == pytest.approx(0.0)
    assert repaired.loc[last, "session_progress"] == pytest.approx(1.0)
    assert repaired.loc[first, "opening_period_flag"] == pytest.approx(1.0)
    previous = frame[
        (frame["session_date"] == "2017-11-21")
        & (frame["session_type"] == "REGULAR")
    ]
    assert repaired.loc[first, "overnight_gap"] == pytest.approx(
        rth.iloc[0]["open"] / previous.iloc[-1]["close"] - 1.0
    )
    extended = frame.index[frame["session_type"].isin(["PRE_MARKET", "AFTER_HOURS"])]
    assert repaired.loc[extended, list(REPAIRED_FEATURES)].isna().all().all()


def test_bulk_sidecar_reader_preserves_session_type(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    partition = raw_root / "symbol=AAA" / "year=2024" / "bars.parquet"
    partition.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "asset_id": ["AAA"],
            "canonical_symbol": ["AAA"],
            "timestamp_utc": pd.to_datetime(["2024-01-02T14:30Z"]),
            "session_date": ["2024-01-02"],
            "session_type": ["REGULAR"],
            "open": [10.0],
            "high": [10.1],
            "low": [9.9],
            "close": [10.0],
        }
    ).to_parquet(partition, index=False)

    result = _read_raw_context(raw_root, "AAA", [2024])

    assert result["session_type"].tolist() == ["REGULAR"]


def test_composite_reader_handles_string_timestamp_parquet_filter_fallback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "targets.parquet"
    pd.DataFrame(
        {
            "decision_timestamp": [
                "2024-01-02T14:35:00+00:00",
                "2024-01-03T14:35:00+00:00",
            ],
            "target_value": [0.1, 0.2],
        }
    ).to_parquet(path, index=False)

    frame = CleanV2CompositeData._read_timestamp_slice(
        path,
        columns=["decision_timestamp", "target_value"],
        start=pd.Timestamp("2024-01-03", tz="UTC"),
        end=pd.Timestamp("2024-01-04", tz="UTC"),
    )

    assert frame["target_value"].tolist() == [0.2]


def test_sidecar_publication_accepts_null_canonical_symbol_partition(tmp_path: Path) -> None:
    base_path = tmp_path / "base" / "stock" / "asset=BBWI" / "year=2016" / "features.parquet"
    base_path.parent.mkdir(parents=True)
    decision = pd.Timestamp("2016-01-04T14:35Z")
    timestamp = decision - pd.Timedelta(minutes=5)
    pd.DataFrame(
        {
            "asset_id": ["BBWI"],
            "canonical_symbol": [None],
            "timestamp_utc": [timestamp],
            "decision_timestamp": [decision],
            "session_date": ["2016-01-04"],
        }
    ).to_parquet(base_path, index=False)
    raw_sidecar = pd.DataFrame({"timestamp_utc": [timestamp]})
    for feature in REPAIRED_FEATURES:
        raw_sidecar[feature] = [np.nan]
        raw_sidecar[f"{feature}{SOURCE_TIMESTAMP_SUFFIX}"] = [pd.NaT]

    result = _publish_partition(base_path, raw_sidecar, tmp_path / "output")

    assert result["row_count"] == 1
    assert result["relative_path"] == "stock/asset=BBWI/year=2016/pit_repair.parquet"


def test_all_101_features_are_invariant_to_future_bar_value_perturbation() -> None:
    frames = synthetic_multiyear_raw_frames()
    certificate = certify_future_bar_invariance(frames, certification_cases(frames))

    assert certificate["predictor_count"] == 101
    assert certificate["case_count"] == 9
    assert certificate["passed"] is True
    assert certificate["terminal_classification"] == (
        "101 / 101 FEATURES CAUSAL UNDER FUTURE-BAR PERTURBATION"
    )
    assert all(not case["changed_features"] for case in certificate["cases"])


def test_all_eleven_repaired_formulas_have_independent_semantic_checks() -> None:
    checks = _synthetic_semantic_checks(synthetic_multiyear_raw_frames())

    assert checks["independent_formula_checks_passed"] is True
    assert set(checks["independent_formula_check_details"]) == set(REPAIRED_FEATURES)
    assert all(checks["independent_formula_check_details"].values())
    assert checks["rth_extended_hours_semantics_passed"] is True
    assert checks["calendar_state_semantics_passed"] is True


def test_composite_sidecar_replaces_only_repaired_columns_and_requires_full_coverage() -> None:
    keys = {
        "asset_id": ["AAA", "AAA"],
        "decision_timestamp": pd.to_datetime(["2024-01-02T14:35Z", "2024-01-02T14:40Z"]),
    }
    base = pd.DataFrame({**keys, "ret_5m": [0.1, 0.2], **{name: [99.0, 99.0] for name in REPAIRED_FEATURES}})
    sidecar = pd.DataFrame({**keys, **{name: [float(index), float(index + 1)] for index, name in enumerate(REPAIRED_FEATURES)}})

    result = compose_v2_stock_features(base, sidecar)

    assert result["ret_5m"].tolist() == [0.1, 0.2]
    assert result["overnight_gap"].tolist() == [0.0, 1.0]
    with pytest.raises(ValueError, match="cover"):
        compose_v2_stock_features(base, sidecar.iloc[:1])


def test_target_contract_and_train_score_chronology_fail_closed() -> None:
    target = load_contract("target_contract.json")
    predictors = load_contract("predictor_manifest.json")["predictors"]
    assert target["target_id"] == "forward_return_60m__decision_5m"
    assert target["resolved_contract_sha256"] == "8e2d5458044c17cf60bc1d8e46b71357599f3837ddb53d64d9b00d322d17f419"
    training = pd.DataFrame(
        {
            "decision_timestamp": pd.to_datetime(["2024-01-02T14:35Z"]),
            "target_available_timestamp": pd.to_datetime(["2024-01-02T15:35Z"]),
            "target_is_trainable": [True],
        }
    )
    scoring = pd.DataFrame({"decision_timestamp": pd.to_datetime(["2024-01-03T14:35Z"])})
    validate_target_use(
        predictor_names=predictors,
        training_rows=training,
        scoring_rows=scoring,
        fit_timestamp="2024-01-03T14:35Z",
    )
    training.loc[0, "target_available_timestamp"] = pd.Timestamp("2024-01-04T14:35Z")
    with pytest.raises(CleanRuntimeError, match="maturity"):
        validate_target_use(
            predictor_names=predictors,
            training_rows=training,
            scoring_rows=scoring,
            fit_timestamp="2024-01-03T14:35Z",
        )


def test_five_session_refit_has_no_daily_fallback() -> None:
    sessions = pd.bdate_range("2024-01-02", periods=35, tz="UTC") + pd.Timedelta(hours=14, minutes=35)
    schedule = clean_refit_schedule(sessions)
    assert REFIT_POLICY_ID == "REFIT_EVERY_5_TRADING_SESSIONS_V1"
    assert schedule
    assert all(len(spec.score_session_dates) == 5 for spec in schedule)
    assert [spec.ordinal for spec in schedule] == list(range(len(schedule)))


def test_ownership_qualifier_and_order_guards_are_frozen() -> None:
    ownership = load_contract("cross_host_ownership.json")
    validate_ownership(ownership["hosts"])
    assert QUALIFIER_YEARS == (2017, 2019, 2020, 2022, 2024)
    assert tuple(load_contract("tournament_contract.json")["qualifier"]["calendar_years"]) == QUALIFIER_YEARS
    assert PAPER_ORDERS_ALLOWED is False
    assert LIVE_ORDERS_ALLOWED is False
    duplicate = {"dell": ["transformer"], "mac": ["transformer"]}
    with pytest.raises(CleanRuntimeError, match="both"):
        validate_ownership(duplicate)


def test_amended_registry_lanes_retire_itransformer_and_primary_elastic_net() -> None:
    models = load_contract("model_registry.json")
    tournament = load_contract("tournament_contract.json")
    ownership = load_contract("cross_host_ownership.json")
    scheduled = {
        family for families in tournament["lanes"].values() for family in families
    }
    owned = {
        family for families in ownership["hosts"].values() for family in families
    }

    assert tournament["lanes"]["FULL_CLEAN"] == [
        "random_forest",
        "transformer",
        "huber",
        "ridge_C5",
        "gradient_boosting_C0",
    ]
    assert {"elastic_net_C5", "elastic_net_C6"} <= set(
        tournament["lanes"]["SHORT_REQUALIFICATION"]
    )
    assert tournament["lanes"]["UNSCORED_DISCOVERY"] == [
        "momentum_transformer",
        "market_context_encoder",
        "temporal_fusion_transformer",
    ]
    assert "itransformer" not in scheduled | owned | set(models["families"])
    assert "itransformer" in models["preserved_but_not_scheduled"]
    assert "elastic_net" not in scheduled | owned | set(models["families"])
    assert "elastic_net" in models["preserved_but_not_scheduled"]
    assert ownership["rules"]["automatic_continuation_allowed"] is False


def test_preregistered_gbdt_hypotheses_do_not_reintroduce_daily_refits() -> None:
    models = load_contract("model_registry.json")
    tournament = load_contract("tournament_contract.json")
    evidence = load_contract("prior_evidence_manifest.json")

    assert evidence["classification"] == "PREREGISTERED_CONFIGURATION_HYPOTHESES_ONLY"
    assert [row["prior_rank_ic"] for row in evidence["gbdt_structural_hypotheses"][:3]] == [
        0.4470758692,
        0.4740403016,
        0.4720688783,
    ]
    assert evidence["gbdt_structural_hypotheses"][3]["prior_rank_ic"] == 0.4554969492
    assert tournament["refit_policy"]["id"] == REFIT_POLICY_ID
    assert tournament["refit_policy"]["daily_refit_fallback_allowed"] is False
    assert "daily_session_v1" in tournament["refit_policy"]["forbidden_policy_ids"]
    assert models["families"]["gradient_boosting_C0"]["training"][
        "lookback_sessions"
    ] == 20
    assert models["families"]["gradient_boosting_C0_W40"]["training"][
        "lookback_sessions"
    ] == 40
    assert models["families"]["gradient_boosting_C0_W80"]["training"][
        "lookback_sessions"
    ] == 80
    assert models["preregistered_unresolved_hypotheses"]["gradient_boosting_C7"][
        "automatic_admission"
    ] is False


def test_clean_worker_schedule_uses_five_score_sessions_and_frozen_years() -> None:
    spine = list(
        pd.bdate_range("2016-11-01", "2017-03-31", tz="UTC")
        + pd.Timedelta(hours=14, minutes=35)
    )
    schedule = build_clean_refit_schedule(
        spine,
        lookback_sessions=40,
        qualifier_years=[2017],
    )

    assert schedule
    assert all(len(row.training_session_dates) == 40 for row in schedule)
    assert all(1 <= len(row.score_session_dates) <= 5 for row in schedule)
    assert all(
        all(date.startswith("2017-") for date in row.score_session_dates)
        for row in schedule
    )
    assert all(row.policy_hash == schedule[0].policy_hash for row in schedule)


def test_supervisor_binds_only_clean_v2_workers() -> None:
    tournament = load_contract("tournament_contract.json")
    for host in ("dell", "mac"):
        commands = tournament["worker_commands"][host]
        assert set(commands) == set(_launch_families(host))
        assert all(
            "ds24_clean_v2_family_worker.py" in " ".join(command)
            for command in commands.values()
        )
    assert isinstance(clean_source_hash(), str)
    assert len(clean_source_hash()) == 64


def test_resume_and_metric_keys_are_deterministic_and_idempotent() -> None:
    state = FamilyResumeState("random_forest", "dell", "f", "t", "m", "r")
    key = metric_key(
        family="random_forest",
        decision_timestamp="2024-01-02T14:35:00Z",
        metric="rank_ic",
        cost_bps=10,
    )
    assert key == metric_key(
        family="random_forest",
        decision_timestamp="2024-01-02T14:35:00+00:00",
        metric="rank_ic",
        cost_bps=10,
    )
    assert state.record_refit("2024-01-02T14:35:00Z") is True
    assert state.record_refit("2024-01-02T14:35:00+00:00") is False
    assert state.record_metric(key, "2024-01-02T14:35:00Z") is True
    assert state.record_metric(key, "2024-01-02T14:35:00Z") is False
    assert state.metrics_cursor == 1
    assert state.payload()["paper_orders"] == 0
    assert state.payload()["live_orders"] == 0


def test_compact_output_guard_blocks_permanent_prediction_matrices(tmp_path: Path) -> None:
    assert_compact_output_path(tmp_path / "metrics" / "rank_ic.parquet")
    with pytest.raises(CleanRuntimeError, match="forbidden"):
        assert_compact_output_path(tmp_path / "full_predictions" / "scores.parquet")


def test_preflight_partition_inventory_is_bounded_and_size_checked(tmp_path: Path) -> None:
    partition = tmp_path / "stock" / "asset=AAA" / "pit_repair.parquet"
    partition.parent.mkdir(parents=True)
    partition.write_bytes(b"authority")
    row = {"relative_path": "stock/asset=AAA/pit_repair.parquet", "bytes": 9}

    assert _partition_inventory_matches(
        tmp_path, [row], relative_path_key="relative_path", bytes_key="bytes"
    )
    row["bytes"] = 8
    assert not _partition_inventory_matches(
        tmp_path, [row], relative_path_key="relative_path", bytes_key="bytes"
    )
    escaped = {"relative_path": "../outside.parquet", "bytes": 9}
    assert not _partition_inventory_matches(
        tmp_path, [escaped], relative_path_key="relative_path", bytes_key="bytes"
    )


def test_cross_host_partition_hash_check_detects_same_size_corruption(tmp_path: Path) -> None:
    partition = tmp_path / "target_rows.parquet"
    partition.write_bytes(b"clean")
    expected = hashlib.sha256(b"clean").hexdigest()
    rows = [{"relative_path": partition.name, "sha256": expected}]
    assert partition_file_hashes_match(
        tmp_path, rows, relative_path_key="relative_path", sha256_key="sha256"
    )

    partition.write_bytes(b"dirty")
    assert not partition_file_hashes_match(
        tmp_path, rows, relative_path_key="relative_path", sha256_key="sha256"
    )


def test_target_delta_preserves_explicit_empty_symbol_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(target_delta_builder, "OUTPUT_ROOT", tmp_path)
    path = tmp_path / "target_id=x" / "symbol=JHG" / "year=2026" / "target_rows.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "asset_id": pd.Series(dtype="string"),
            "decision_timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "target_available_timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "target_is_trainable": pd.Series(dtype="bool"),
            "target_id": pd.Series(dtype="string"),
            "target_code_hash": pd.Series(dtype="string"),
        }
    ).to_parquet(path, index=False)

    result = target_delta_builder._validate_partition("JHG", path)

    assert result["rows"] == 0
    assert result["empty_source_window"] is True
    assert result["target_code_hashes"] == []


def test_new_runtime_has_no_deleted_stage_output_dependency() -> None:
    forbidden = "docs/dream_system/components/DS-24_independent_five_minute_selector"
    paths = [
        *Path("core/research/ml/ds24").glob("clean_v2_*.py"),
        *(
            path
            for path in Path("scripts/local").glob("ds24_clean_v2_*.py")
            if path.name != "ds24_clean_v2_finalize_reset.py"
        ),
    ]
    assert paths
    for path in paths:
        normalized = path.read_text(encoding="utf-8").replace("\\", "/")
        assert forbidden not in normalized, path


def test_static_authority_bundle_is_complete_and_hashable() -> None:
    bundle = authority_bundle()
    assert bundle["authority_id"] == "DS24_CLEAN_V2_STATIC_AUTHORITY_BUNDLE_V1"
    assert set(bundle["files"]) == {
        "predictor_manifest.json",
        "feature_authority.json",
        "target_contract.json",
        "eligibility_contract.json",
        "model_registry.json",
        "tournament_contract.json",
        "cross_host_ownership.json",
        "prior_evidence_manifest.json",
    }
    assert all((CONFIG_ROOT / name).is_file() for name in bundle["files"])
