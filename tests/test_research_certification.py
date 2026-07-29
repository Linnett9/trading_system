from __future__ import annotations

import csv
import json
from pathlib import Path

from core.research.ml.artifact_lineage import (
    VERIFIED_STRICT_OOS,
    build_artifact_link,
    verify_selector_artifact,
)
from core.research.ml.dataset_build_manifest import (
    build_dataset_build_manifest,
    dataset_manifest_path,
    file_sha256,
    manifest_hash,
    write_manifest,
)
from core.research.ml.registries.io import canonical_hash
from core.research.ml.research_certification import (
    BLOCKED,
    CERTIFIABLE,
    DIAGNOSTIC_ONLY,
    RESEARCH_ONLY,
    build_research_certification_envelope,
    verify_research_certification_envelope,
    write_research_certification_envelope,
)


def test_clean_certifiable_run_spans_dataset_prediction_replay_and_accounting(tmp_path: Path) -> None:
    context = _certification_context(tmp_path)

    envelope = _build_envelope(context)

    assert envelope["certification_status"] == CERTIFIABLE
    assert envelope["dataset_manifests"]
    assert envelope["prediction_artifacts"]
    assert envelope["portfolio_replay_artifacts"]
    assert envelope["promotion"]["automatic_promotion"] is False


def test_dirty_run_without_patch_is_research_only(tmp_path: Path) -> None:
    context = _certification_context(
        tmp_path,
        source_control={"git_commit": "abc", "dirty_worktree": True},
    )

    envelope = _build_envelope(context)

    assert envelope["certification_status"] == RESEARCH_ONLY
    assert "DIRTY_TREE_WITHOUT_CAPTURED_PATCH" in envelope["certification_gates"]["research_only_reasons"]


def test_dirty_run_with_captured_patch_can_be_certifiable(tmp_path: Path) -> None:
    context = _certification_context(
        tmp_path,
        source_control={"git_commit": "abc", "dirty_worktree": True},
    )

    envelope = _build_envelope(context, dirty_patch_hash="A" * 64)

    assert envelope["certification_status"] == CERTIFIABLE
    assert envelope["dirty_patch_hash"] == "A" * 64


def test_stale_dataset_blocks_certification(tmp_path: Path) -> None:
    context = _certification_context(tmp_path)
    context["source"].write_text("parent_id,value\nsource,changed\n", encoding="utf-8")

    envelope = _build_envelope(context)

    assert envelope["certification_status"] == BLOCKED
    assert any(
        reason.startswith("STALE_OR_MISSING_DATASET_PARENT")
        for reason in envelope["certification_gates"]["hard_blocking_reasons"]
    )


def test_missing_authority_version_fails_to_research_only(tmp_path: Path) -> None:
    context = _certification_context(
        tmp_path,
        authority_overrides={"identity_authority_version": None},
    )

    envelope = _build_envelope(context)

    assert envelope["certification_status"] == RESEARCH_ONLY
    assert "AUTHORITY_VERSION_MISSING:identity_authority_version" in envelope["certification_gates"]["research_only_reasons"]


def test_missing_trial_family_is_research_only(tmp_path: Path) -> None:
    context = _certification_context(tmp_path)

    envelope = _build_envelope(context, trial_family_id="")

    assert envelope["certification_status"] == RESEARCH_ONLY
    assert "TRIAL_FAMILY_RECORD_MISSING" in envelope["certification_gates"]["research_only_reasons"]


def test_incomplete_ticket66_trial_evidence_reports_specific_blockers(tmp_path: Path) -> None:
    context = _certification_context(tmp_path)

    envelope = _build_envelope(
        context,
        trial_accounting={
            "trial_family_count": 2,
            "effective_search_count": 1,
            "raw_trial_count": 2,
            "trial_family_complete": False,
            "holdout_invalidation_status": "INVALIDATED",
        },
    )
    reasons = set(envelope["certification_gates"]["research_only_reasons"])

    assert envelope["certification_status"] == RESEARCH_ONLY
    assert "TRIAL_FAMILY_RECORD_MISSING" in reasons
    assert "TRIAL_ATTEMPTS_INCOMPLETE" in reasons
    assert "DSR_FULL_FAMILY_EVIDENCE_MISSING" in reasons
    assert "PBO_DECLARED_FAMILY_EVIDENCE_MISSING" in reasons
    assert "LOCKED_HOLDOUT_REUSED" in reasons
    assert "DSR_FAMILY_SIZE_MISMATCH" in reasons


def test_sequence_run_without_strict_context_blocks(tmp_path: Path) -> None:
    context = _certification_context(tmp_path, sequence_model=True)

    envelope = _build_envelope(
        context,
        sequence_context={
            "authority_version": "sequence_window_authority_v1",
            "strict_context_recorded": False,
        },
    )

    assert envelope["certification_status"] == BLOCKED
    assert "STRICT_SEQUENCE_CONTEXT_MISSING" in envelope["certification_gates"]["hard_blocking_reasons"]


def test_missing_replay_is_research_only(tmp_path: Path) -> None:
    context = _certification_context(tmp_path, include_replay=False)

    envelope = _build_envelope(context)

    assert envelope["certification_status"] == RESEARCH_ONLY
    assert "PORTFOLIO_REPLAY_MISSING" in envelope["certification_gates"]["research_only_reasons"]


def test_prediction_hash_mismatch_blocks(tmp_path: Path) -> None:
    context = _certification_context(tmp_path, prediction_checksum="0" * 64)

    envelope = _build_envelope(context)

    assert envelope["certification_status"] == BLOCKED
    assert "PREDICTION_HASH_MISMATCH" in envelope["certification_gates"]["hard_blocking_reasons"]


def test_exact_artifact_replay_verifier_consumes_envelope(tmp_path: Path) -> None:
    context = _certification_context(tmp_path)
    envelope_path = tmp_path / "envelope.json"
    envelope = write_research_certification_envelope(
        envelope_path,
        **_build_kwargs(context),
    )

    verification = verify_research_certification_envelope(envelope_path)

    assert envelope["certification_status"] == CERTIFIABLE
    assert verification["reproduction_status"] == "EXACT_ARTIFACT_REPLAY"
    assert verification["training_rerun_performed"] is False


def test_diagnostic_legacy_run_is_diagnostic_only(tmp_path: Path) -> None:
    context = _certification_context(tmp_path)

    envelope = _build_envelope(context, diagnostic_legacy=True)

    assert envelope["certification_status"] == DIAGNOSTIC_ONLY


def test_envelope_identity_is_deterministic(tmp_path: Path) -> None:
    context = _certification_context(tmp_path)

    first = _build_envelope(context)
    second = _build_envelope(context)

    assert first["run_id"] == second["run_id"]
    assert first["deterministic_envelope_identity"] == second["deterministic_envelope_identity"]
    assert first["envelope_hash"] == second["envelope_hash"]


def test_no_promotion_when_blocked(tmp_path: Path) -> None:
    context = _certification_context(tmp_path, prediction_checksum="0" * 64)

    envelope = _build_envelope(context)

    assert envelope["certification_status"] == BLOCKED
    assert envelope["promotion"]["automatic_promotion"] is False
    assert envelope["promotion"]["promotion_prohibited"] is True


def _build_envelope(context: dict[str, object], **overrides: object) -> dict[str, object]:
    kwargs = _build_kwargs(context)
    kwargs.update(overrides)
    return build_research_certification_envelope(**kwargs)


def _build_kwargs(context: dict[str, object]) -> dict[str, object]:
    replay_paths = (context["replay_manifest"],) if context.get("replay_manifest") else ()
    return {
        "config": {"ml": {"random_seed": 7}},
        "source_control": context["source_control"],
        "dataset_manifest_paths": (context["dataset_manifest"],),
        "prediction_manifest_paths": (context["prediction_manifest"],),
        "portfolio_replay_paths": replay_paths,
        "execution_scenario": {
            "execution_assumptions": {
                "cost_bps": 0,
                "slippage_bps": 0,
                "trading_impact": "none",
                "promotion_automated": False,
            },
            "no_trade_comparison": {"status": "recorded", "trade_count": 0},
            "null_policy": {"policy_id": "null_no_orders", "trade_allowed": False},
        },
        "trial_family_id": "family-a",
        "trial_accounting": {
            "dsr_evidence": {"deflated_sharpe_probability": 0.8},
            "pbo_evidence": {"probability_of_backtest_overfitting": 0.1},
            "effective_search_count": 1,
            "raw_trial_count": 1,
            "trial_family_count": 1,
            "trial_family_ids": ["family-a"],
        },
        "promotion_evidence": {
            "evidence_complete": True,
            "promotion_triggered": False,
            "promotion_report_checksum": "P" * 64,
        },
        "sequence_context": context.get("sequence_context"),
    }


def _certification_context(
    tmp_path: Path,
    *,
    source_control: dict[str, object] | None = None,
    authority_overrides: dict[str, object] | None = None,
    prediction_checksum: str | None = None,
    include_replay: bool = True,
    sequence_model: bool = False,
) -> dict[str, object]:
    dataset, source, dataset_manifest, dataset_payload = _write_dataset_bundle(
        tmp_path,
        source_control=source_control,
        authority_overrides=authority_overrides,
    )
    prediction_manifest, selector_link = _write_prediction_manifest(
        tmp_path,
        dataset_payload=dataset_payload,
        prediction_checksum=prediction_checksum,
        sequence_model=sequence_model,
    )
    replay_manifest = _write_replay_manifest(tmp_path, selector_link) if include_replay else None
    return {
        "dataset": dataset,
        "source": source,
        "dataset_manifest": dataset_manifest,
        "prediction_manifest": prediction_manifest,
        "replay_manifest": replay_manifest,
        "source_control": source_control or {"git_commit": "abc", "dirty_worktree": False},
        "sequence_context": None,
    }


def _write_dataset_bundle(
    tmp_path: Path,
    *,
    source_control: dict[str, object] | None,
    authority_overrides: dict[str, object] | None,
) -> tuple[Path, Path, Path, dict[str, object]]:
    source = tmp_path / "source.csv"
    dataset = tmp_path / "dataset.csv"
    rows = [
        {"feature_id": "a", "symbol": "AAA", "feature_date": "2024-01-01"},
        {"feature_id": "b", "symbol": "BBB", "feature_date": "2024-01-02"},
    ]
    _write_csv(source, [{"parent_id": "source", "value": "1"}])
    _write_csv(dataset, rows)
    authorities = {
        "canonical_price_authority_version": "canonical_price_pit_v1",
        "universe_authority_version": "pit_universe_authority_v1",
        "identity_authority_version": "historical_identity_authority_v1",
        "corporate_action_authority_version": "corporate_action_pit_v1",
        "market_calendar_authority_version": "market_calendar_v1",
        "target_contract_version": "forward_return_10d",
        "feature_code_version": "feature-code-v1",
        "label_code_version": "label-code-v1",
    }
    authorities.update(authority_overrides or {})
    manifest = build_dataset_build_manifest(
        dataset_id="dataset-a",
        dataset_type="frozen_selector_dataset",
        schema_version="schema-v1",
        producer_command="test-producer",
        producer_module="tests.test_research_certification",
        output_paths=(dataset,),
        source_paths=(source,),
        rows=rows,
        key_fields=("feature_id",),
        configuration_hash_value="config-v1",
        random_seed=7,
        build_timestamp="2026-01-01T00:00:00+00:00",
        source_control=source_control or {"git_commit": "abc", "dirty_worktree": False},
        **authorities,
    )
    manifest_path = dataset_manifest_path(dataset)
    write_manifest(manifest_path, manifest)
    return dataset, source, manifest_path, manifest


def _write_prediction_manifest(
    tmp_path: Path,
    *,
    dataset_payload: dict[str, object],
    prediction_checksum: str | None,
    sequence_model: bool,
) -> tuple[Path, dict[str, object]]:
    prediction_path = tmp_path / "predictions.csv"
    _write_csv(
        prediction_path,
        [
            {"row_id": "a", "score": "0.1"},
            {"row_id": "b", "score": "0.2"},
        ],
    )
    actual_checksum = file_sha256(prediction_path)
    recorded_checksum = prediction_checksum or actual_checksum
    model_id = "dlinear" if sequence_model else "ridge"
    manifest_path = tmp_path / "prediction_manifest.json"
    link = build_artifact_link(
        artifact_kind="BOUNDED_SELECTOR_PREDICTION",
        artifact_id="prediction-a",
        artifact_manifest_path=manifest_path,
        artifact_path=prediction_path,
        artifact_checksum=recorded_checksum,
        experiment_spec_hash="S" * 64,
        experiment_run_id="run-a",
        canonical_model_or_policy_id=model_id,
        model_or_policy_entry_hash="M" * 64,
        dataset_id=dataset_payload["dataset_id"],
        dataset_checksum=dataset_payload["manifest_hash"],
        row_population_hash=canonical_hash(["a", "b"]),
        feature_schema_hash="F" * 64,
        target_contract_hash="T" * 64,
        decision_start="2024-01-02T20:05:00+00:00",
        decision_end="2024-01-02T20:05:00+00:00",
        training_cutoff="2024-01-01T20:05:00+00:00",
        maximum_label_available_timestamp="2024-01-02T19:00:00+00:00",
        strict_oos_claim=True,
        strict_oos_evidence={
            "prediction_quality_passed": True,
            "row_population_verified": True,
        },
        completion_status="complete",
    )
    link.update(verify_selector_artifact(link).to_dict())
    payload = {
        "contract_version": "bounded_daily_selector_v2",
        "validation_status": link["verification_status"],
        "artifact_link": link,
        "prediction_artifact_path": str(prediction_path),
        "prediction_checksum": recorded_checksum,
        "fold_identity": "fold-a",
        "prediction_row_count": 2,
        "oos_row_count": 2,
        "random_seed": 7,
        "experiments": [
            {
                "experiment_spec_hash": "S" * 64,
                "experiment_run_id": "run-a",
            }
        ],
        "target_contract_version": "forward_return_10d",
        "feature_contract_version": "feature-code-v1",
    }
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return manifest_path, link


def _write_replay_manifest(tmp_path: Path, selector_link: dict[str, object]) -> Path:
    path = tmp_path / "stock_level_portfolio_replay_summary.json"
    base = {
        "summary": [{"strategy_id": "ml_signal|null_no_orders", "net_return": 0.0}],
        "policies": ["null_no_orders"],
        "signal_columns": ["ml_signal"],
        "oos_only": True,
        "cost_bps": 0,
        "slippage_bps": 0,
        "top_n": 2,
    }
    link = build_artifact_link(
        artifact_kind="PORTFOLIO_REPLAY",
        artifact_id="replay-a",
        artifact_manifest_path=path,
        artifact_checksum=canonical_hash(base),
        canonical_model_or_policy_id="null_no_orders",
        model_or_policy_entry_hash="N" * 64,
        dataset_id=selector_link["dataset_id"],
        dataset_checksum=selector_link["dataset_checksum"],
        feature_schema_hash=selector_link["feature_schema_hash"],
        target_contract_hash=selector_link["target_contract_hash"],
        decision_start=selector_link["decision_start"],
        decision_end=selector_link["decision_end"],
        strict_oos_claim=True,
        strict_oos_evidence={"selector_input_status": VERIFIED_STRICT_OOS},
        upstream_links=[selector_link],
        verification_status=VERIFIED_STRICT_OOS,
        verification_reasons=[],
        completion_status="complete",
    )
    payload = {
        **base,
        "artifact_link": link,
        "promotion": {
            "promotion_eligible": True,
            "blocking_reasons": [],
        },
        "lineage_mode": "promotion",
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    _write_csv(
        tmp_path / "stock_level_portfolio_replay_summary.csv",
        [{"strategy_id": "ml_signal|null_no_orders", "net_return": "0.0"}],
    )
    _write_csv(
        tmp_path / "stock_level_portfolio_replay_equity_curves.csv",
        [{"rebalance_date": "2024-01-02", "equity": "1.0"}],
    )
    _write_csv(
        tmp_path / "stock_level_portfolio_replay_holdings.csv",
        [{"rebalance_date": "2024-01-02", "symbol": "AAA", "weight": "0.0"}],
    )
    return path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
