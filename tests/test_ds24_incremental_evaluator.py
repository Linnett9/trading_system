from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.research.ml import ds24_metrics_only_evaluator as ev
from core.research.ml.ds24.incremental_evaluator_state import (
    IncrementalV3SummaryState,
    build_summary_state,
    numerical_differences,
)


META = {
    "model_hash": "model-v1",
    "model_vintage_id": "vintage-v1",
    "preprocessing_hash": "preprocessing-v1",
    "policy_hash": "policy-v1",
    "training_cutoff": "2024-01-02T14:30:00Z",
    "prediction_timestamp": "2024-01-02T14:35:01Z",
    "feature_authority_hash": "feature-authority",
    "target_authority_hash": "target-authority",
    "target_contract_hash": "target-contract",
    "static_authority_bundle_sha256": "authority-bundle",
    "clean_source_hash": "source-v1",
    "attempt_generation": "7",
}


def _predictions(timestamp: str, count: int = 24) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "family": ["random_forest"] * count,
            "decision_timestamp": [timestamp] * count,
            "asset_id": [f"A{index:03d}" for index in range(count)],
            "prediction": np.linspace(1.0, 0.0, count),
        }
    )


def _targets_for_request(keys: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    targets = keys.copy()
    targets["decision_timestamp"] = pd.to_datetime(
        targets["decision_timestamp"], utc=True
    )
    targets["target_available_timestamp"] = (
        targets["decision_timestamp"] + pd.Timedelta(minutes=60)
    )
    targets["target_is_trainable"] = True
    targets["target_value"] = np.arange(len(targets), dtype=float)
    targets["target_id"] = ev.TARGET_ID
    return targets, {"target_rows_loaded": len(targets)}


def test_incremental_summary_matches_full_history_contract() -> None:
    timestamps = pd.date_range(
        "2024-01-02T14:35:00Z", periods=90, freq="5min", tz="UTC"
    )
    rank = pd.DataFrame(
        {
            "decision_timestamp": timestamps,
            "session_date": timestamps.date.astype(str),
            "spearman_rank_ic": np.sin(np.arange(len(timestamps))) / 3.0,
            "eligible_asset_count": 100,
            "resolved_asset_count": 91,
        }
    )
    sleeves = pd.DataFrame(
        {
            "decision_timestamp": timestamps,
            "turnover": np.linspace(0.0, 1.0, len(timestamps)),
            "transaction_cost_contribution": np.linspace(
                0.0, 0.001, len(timestamps)
            ),
        }
    )
    daily = pd.DataFrame(
        {
            "session_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "gross_daily_return": [0.01, -0.004, 0.003],
            "net_daily_return": [0.009, -0.005, 0.002],
        }
    )
    reference = ev.resolved_v3_summary(
        rank,
        daily,
        sleeves,
        pending_rows=48,
        terminal_censored_rows=2,
    )
    state = build_summary_state(
        rank, daily, sleeves, pd.DataFrame(index=range(2))
    )
    restored = IncrementalV3SummaryState.from_dict(state.to_dict())
    candidate = restored.to_summary(
        evaluation_contract_id=ev.RESOLVED_PERFORMANCE_CONTRACT_V3_ID,
        evaluation_contract_hash=ev.resolved_performance_contract_v3_hash(),
        evaluation_contract_version=ev.RESOLVED_PERFORMANCE_CONTRACT_V3_VERSION,
        sleeve_count=ev.DEFAULT_V3_SLEEVE_COUNT,
        pending_rows=48,
        created_at_utc=reference["created_at_utc"],
    )
    assert numerical_differences(reference, candidate) == []


def test_writer_bootstraps_once_and_persists_compact_state(tmp_path: Path) -> None:
    root = tmp_path / "metrics"
    writer = ev.ResolvedPerformanceV3Writer(
        root,
        family="random_forest",
        target_loader=_targets_for_request,
    )
    writer.commit_predictions(
        _predictions("2024-01-02T14:35:00Z"), metadata=META
    )
    bootstrap_rows = writer._log_cache.bootstrap_rows_read
    result = writer.commit_predictions(
        _predictions("2024-01-02T15:35:00Z"), metadata=META
    )
    assert writer._log_cache.bootstrap_rows_read == bootstrap_rows
    assert result["checkpoint"]["incremental_hot_path_historical_rows_read"] == 0
    assert len(writer._log_cache.key_hashes("decision_trace_v3")) == 1
    assert len(writer._log_cache.key_hashes("rank_ic_audit_sample_v3")) == 1
    state = json.loads(writer.incremental_state_path.read_text(encoding="utf-8"))
    assert state["schema_version"] == ev.INCREMENTAL_EVALUATOR_STATE_VERSION
    assert state["authority_identity"]["feature_authority_hash"] == "feature-authority"
    assert state["summary_state_hash"]
    manifest = json.loads(
        ev.parquet_log_manifest_path(root, "rank_ic_v3").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["schema_version"] == "DS24_APPEND_ONLY_PARQUET_LOG_V2"
    assert "parts" not in manifest
    assert manifest["part_count"] == 1
    verifier = ev.ResolvedPerformanceV3Writer(root, family="random_forest")
    verification = verifier.verify_or_rebuild_incremental_state()
    assert verification["status"] == "VALID"


def test_restart_rebuilds_state_after_durable_commit_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "metrics"
    writer = ev.ResolvedPerformanceV3Writer(
        root,
        family="random_forest",
        target_loader=_targets_for_request,
    )
    writer.commit_predictions(
        _predictions("2024-01-02T14:35:00Z"), metadata=META
    )
    state_before = writer.incremental_state_path.read_bytes()
    original_write = ev.write_json_atomic

    def crash_before_state(path: Path, payload: object, **kwargs: object) -> None:
        if Path(path).name == ev.INCREMENTAL_EVALUATOR_STATE_NAME:
            raise RuntimeError("SYNTHETIC_CRASH_BEFORE_INCREMENTAL_STATE")
        original_write(path, payload, **kwargs)

    monkeypatch.setattr(ev, "write_json_atomic", crash_before_state)
    with pytest.raises(RuntimeError, match="SYNTHETIC_CRASH"):
        writer.commit_predictions(
            _predictions("2024-01-02T15:35:00Z"), metadata=META
        )
    assert writer.incremental_state_path.read_bytes() == state_before
    monkeypatch.setattr(ev, "write_json_atomic", original_write)

    resumed = ev.ResolvedPerformanceV3Writer(
        root,
        family="random_forest",
        target_loader=_targets_for_request,
    )
    resumed.commit_predictions(
        _predictions("2024-01-02T15:40:00Z"), metadata=META
    )
    rebuilt = json.loads(
        resumed.incremental_state_path.read_text(encoding="utf-8")
    )
    assert rebuilt["bootstrap_disposition"] == "REBUILT_AFTER_INTERRUPTED_COMMIT"
    rank = ev.read_parquet_log(root, "rank_ic_v3")
    assert not rank.duplicated(["family", "decision_timestamp"]).any()


def test_reference_full_history_and_incremental_writer_are_equivalent_across_boundaries(
    tmp_path: Path,
) -> None:
    root = tmp_path / "metrics"
    writer = ev.ResolvedPerformanceV3Writer(
        root,
        family="random_forest",
        target_loader=_targets_for_request,
    )
    sequence = (
        ("2024-01-02T15:00:00Z", "2024-01-02T14:30:00Z", "model-v1"),
        ("2024-01-02T16:00:00Z", "2024-01-02T14:30:00Z", "model-v1"),
        ("2024-01-03T14:35:00Z", "2024-01-03T14:30:00Z", "model-v2"),
        ("2024-01-03T15:35:00Z", "2024-01-03T14:30:00Z", "model-v2"),
    )
    for timestamp, cutoff, model_hash in sequence:
        writer.commit_predictions(
            _predictions(timestamp),
            metadata={
                **META,
                "training_cutoff": cutoff,
                "model_hash": model_hash,
                "model_vintage_id": f"vintage-{model_hash}",
                "prediction_timestamp": pd.Timestamp(timestamp).isoformat(),
            },
        )

    rank = ev.read_parquet_log(root, "rank_ic_v3")
    sleeves = ev.read_parquet_log(root, "sleeve_maturity_ledger_v3")
    daily = ev.read_parquet_log(root, "daily_portfolio_returns_v3")
    terminal = ev.read_parquet_log(root, "terminal_censored_v3")
    pending = pd.read_parquet(root / "pending_scores_v3.parquet")
    published = json.loads(
        (root / "resolved_performance_summary_v3.json").read_text(
            encoding="utf-8"
        )
    )
    reference = ev.resolved_v3_summary(
        rank,
        daily,
        sleeves,
        pending_rows=len(pending),
        terminal_censored_rows=len(terminal),
    )
    reference["created_at_utc"] = published["created_at_utc"]
    assert numerical_differences(reference, published) == []
    assert not rank.duplicated(["family", "decision_timestamp"]).any()
    assert ev.parquet_log_rows(root, "refit_events_v3") == 2


@pytest.mark.parametrize("history_rows", [32, 64, 128])
def test_n_2n_4n_history_is_scanned_once_not_per_commit(
    tmp_path: Path, history_rows: int
) -> None:
    root = tmp_path / f"metrics-{history_rows}"
    timestamps = pd.date_range(
        "2023-01-02T14:35:00Z", periods=history_rows, freq="5min", tz="UTC"
    )
    rank = pd.DataFrame(
        {
            "family": "random_forest",
            "decision_timestamp": timestamps,
            "session_date": timestamps.date.astype(str),
            "spearman_rank_ic": np.linspace(-0.2, 0.2, history_rows),
            "eligible_asset_count": 100,
            "resolved_asset_count": 90,
        }
    )
    rank["row_hash"] = rank.apply(lambda row: ev.stable_hash(row.to_dict()), axis=1)
    ev.append_parquet_log(root, "rank_ic_v3", rank)
    writer = ev.ResolvedPerformanceV3Writer(
        root,
        family="random_forest",
        target_loader=lambda _keys: (pd.DataFrame(), {"target_rows_loaded": 0}),
    )
    writer.commit_predictions(
        _predictions("2024-01-02T14:35:00Z", count=3), metadata=META
    )
    first_scan_rows = writer._log_cache.bootstrap_rows_read
    result = writer.commit_predictions(
        _predictions("2024-01-02T14:40:00Z", count=3), metadata=META
    )
    assert first_scan_rows == history_rows
    assert writer._log_cache.bootstrap_rows_read == history_rows
    assert writer._summary_state.rank_rows == history_rows
    assert result["checkpoint"]["incremental_hot_path_historical_rows_read"] == 0
    assert result["checkpoint"]["incremental_hot_path_new_rows_considered"] <= 2
    assert writer.incremental_state_path.stat().st_size < 50_000

