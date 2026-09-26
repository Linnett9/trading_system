from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.audits import ds24_paper_provenance_forensic as audit


def test_feature_order_hash_and_future_perturbation_prove_leakage() -> None:
    evidence = audit.audit_feature_authority()

    assert evidence["predictor_count"] == 101
    assert evidence["manifest_hash_recomputed"] == evidence["manifest_hash_recorded"]
    assert (
        evidence["feature_order_sha256_recomputed"]
        == evidence["feature_order_sha256_expected"]
    )
    assert evidence["leaked_predictors_in_manifest"] == list(audit.LEAKED_PREDICTORS)

    perturbation = evidence["synthetic_future_perturbation"]
    assert perturbation["all_leaked_predictors_change"] is True
    assert perturbation["causal_control_ret_5m"]["unchanged"] is True
    assert pd.Timestamp(perturbation["mutated_future_timestamp"]) > pd.Timestamp(
        perturbation["decision_timestamp"]
    )


def test_persisted_aapl_features_use_a_post_decision_close() -> None:
    evidence = audit.persisted_feature_leakage_example()

    assert evidence["all_values_match_future_close_calculation"] is True
    assert pd.Timestamp(evidence["future_source_timestamp"]) > pd.Timestamp(
        evidence["decision_timestamp"]
    )


def test_training_sources_enforce_split_and_maturity_but_persist_equal_refit_boundary() -> None:
    worker_source = (
        audit.REPOSITORY_ROOT
        / "scripts/local/ds24_p8_r14_e3g_c2_r7_r14_policy_worker.py"
    ).read_text(encoding="utf-8")
    assert '(train_panel["decision_timestamp"] < spec.refit_T)' in worker_source
    assert '(train_panel["target_available_timestamp"] <= spec.refit_T)' in worker_source
    assert '"training_cutoff": spec.refit_T.isoformat()' in worker_source

    for family, namespace in (
        ("huber", "metrics_only_v3_r37_huber_replay"),
        ("elastic_net", "metrics_only_v3_r40_elastic_net"),
    ):
        root = audit.WORKER_ROOT / family / namespace
        manifest = json.loads(
            (root / "refit_events_v3_manifest.json").read_text(encoding="utf-8")
        )
        first_part = root / manifest["parts"][0]["path"]
        frame = pd.read_parquet(audit._openable_path(first_part))
        assert (
            pd.to_datetime(frame["training_cutoff"], utc=True)
            == pd.to_datetime(frame["first_scored_decision_timestamp"], utc=True)
        ).all()


def test_xendcg_frozen_producer_constructs_ordered_mature_ranking_fit() -> None:
    producer = (
        audit.XENDCG_AUTHORITY_ROOT
        / "core/research/ml/ds24/mac_aux_queue_r44f2.py"
    )
    engine = (
        audit.XENDCG_AUTHORITY_ROOT
        / "core/research/ml/ds24/canonical_prequential_engine.py"
    )
    source_manifest = json.loads(
        (audit.XENDCG_AUTHORITY_ROOT / "source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    recorded_producer = next(
        row
        for row in source_manifest["source_files"]
        if row["relative_path"] == "core/research/ml/ds24/mac_aux_queue_r44f2.py"
    )

    assert audit.sha256_file(producer) == recorded_producer["sha256"]
    producer_text = producer.read_text(encoding="utf-8")
    engine_text = engine.read_text(encoding="utf-8")
    assert 'train.sort_values(["decision_timestamp", "asset_id"])' in producer_text
    assert 'ordered.groupby("decision_timestamp")["target_value"].rank' in producer_text
    assert 'ordered.groupby("decision_timestamp", sort=True).size().to_list()' in producer_text
    assert "model.fit(ordered[list(predictors)], labels, group=groups)" in producer_text
    assert '(panel["decision_timestamp"] < timestamp)' in engine_text
    assert '(panel["target_available_timestamp"] <= timestamp)' in engine_text


def test_xendcg_model_inventory_and_hash_fixtures_are_complete() -> None:
    model_root = audit.XENDCG_ROOT / "model_artifacts"
    models = sorted(
        path
        for path in model_root.glob("lightgbm_rank_xendcg_refit=*.pkl")
        if not path.name.startswith("._")
    )
    ordinals = audit._model_ordinals(models)
    assert len(models) == audit.EXPECTED_XENDCG_MODEL_COUNT
    assert len(set(ordinals)) == audit.EXPECTED_XENDCG_MODEL_COUNT
    assert min(ordinals) == audit.EXPECTED_XENDCG_FIRST_ORDINAL
    assert max(ordinals) == audit.EXPECTED_XENDCG_LAST_ORDINAL

    source_manifest = json.loads(
        (audit.XENDCG_AUTHORITY_ROOT / "source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    oof_manifest = json.loads(
        (audit.XENDCG_ROOT / "ensemble_oof_scores_manifest_v2.json").read_text(
            encoding="utf-8"
        )
    )
    fixtures = audit._xendcg_fixture_verification(source_manifest, oof_manifest)
    assert set(fixtures) == {"000002", "001132", "002263"}
    assert all(row["model_hash_match"] for row in fixtures.values())
    assert all(row["oof_hash_match"] for row in fixtures.values())


def test_newey_west_lag_twelve_is_deterministic() -> None:
    result = audit.newey_west_mean_ci([float(value) for value in range(1, 31)], lag=12)

    assert result["observations"] == 30
    assert result["lag"] == 12
    assert result["mean"] == pytest.approx(15.5)
    assert result["standard_error"] == pytest.approx(4.358070419093498)
    assert result["lower"] == pytest.approx(6.958181978576743)
    assert result["upper"] == pytest.approx(24.041818021423257)


def test_current_paper_binding_names_none_of_the_audited_models() -> None:
    binding = audit.audit_paper_binding()

    assert binding["active_candidate_source"] == "dual_momentum"
    assert binding["active_submit_orders"] is True
    assert binding["active_paper_broker"] is True
    assert binding["audited_model_names_present_in_active_config"] == []
    assert binding["bound_incumbent"]["model"] == "sklearn GradientBoostingRegressor"
    assert binding["bound_incumbent"]["config_hash"] == binding[
        "ds24_order_adapter_config_hash"
    ]
    assert binding["bound_incumbent"]["training_policy_hash"] == binding[
        "ds24_order_adapter_training_policy_hash"
    ]


def test_complete_sleeve_ledgers_expose_missing_terminal_daily_rows() -> None:
    evidence_path = (
        audit.REPOSITORY_ROOT
        / "docs/audits/ds24_paper_provenance_forensic_20260926/evidence.json"
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    expected = {
        "huber": {"matured": 8, "net_return": 0.007515409754857164},
        "elastic_net": {"matured": 6, "net_return": 0.00469852729502412},
    }
    for family, terminal in expected.items():
        model = evidence["models"][family]
        reconciliation = model["namespace_verification"][
            "daily_recomputed_from_sleeves"
        ]
        assert reconciliation["row_count_matches"] is False
        assert reconciliation["missing_or_extra_rows"] == [
            {
                "ledger_matured_sleeve_decisions": terminal["matured"],
                "ledger_net_daily_return": terminal["net_return"],
                "presence": "right_only",
                "saved_net_daily_return": None,
                "session_date": "2026-06-30",
            }
        ]
        assert model["saved_summary_comparison"]["all_match"] is True
        assert (
            model["complete_sleeve_ledger_summary_comparison"]["all_match"]
            is False
        )
