import json
from types import SimpleNamespace
from datetime import datetime
from pathlib import Path

from core.paper.paper_trading_engine import PaperOrder, PaperTradingEngine
from application.services.paper_trading_service import PaperTradingService


def test_decision_hashes_ignore_target_weight_order_and_tiny_float_noise():
    service = PaperTradingService(config={}, feed=None)
    first = SimpleNamespace(
        exposure_target=0.6750000000001,
        target_weights={
            "TXN": 0.2379696486495513,
            "AMAT": 0.24124989052041312,
        },
        orders=[],
    )
    second = SimpleNamespace(
        exposure_target=0.675,
        target_weights={
            "AMAT": 0.24124989052049999,
            "TXN": 0.23796964864950001,
        },
        orders=[],
    )

    assert service._decision_hashes(first) == service._decision_hashes(second)


def test_append_event_log_writes_structured_jsonl(tmp_path):
    event_log_path = tmp_path / "events.jsonl"
    service = PaperTradingService(
        config={
            "paper_trading": {
                "event_log_path": str(event_log_path),
                "report_dir": str(tmp_path),
            },
        },
        feed=None,
    )
    decision = SimpleNamespace(
        equity=500,
        cash=100,
        orders=[],
    )

    path = service._append_event_log(
        run_id="run_1",
        candidate_id="candidate",
        dry_run=True,
        submit=False,
        decision=decision,
        fill_record=None,
        risk_checks=[],
        post_trade_checks=[],
        blocked_reason=None,
        reproducibility={"config_hash": "abc123"},
        artifact_paths={},
    )

    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["run_id"] == "run_1"
    assert row["candidate_id"] == "candidate"
    assert row["event_type"] == "paper_trading_run"
    assert row["reproducibility"]["config_hash"] == "abc123"


def test_broker_open_order_conflict_blocks_same_symbol(tmp_path):
    service = PaperTradingService(
        config={
            "paper_trading": {
                "execution_adapter": "broker",
                "submit_orders": True,
                "report_dir": str(tmp_path),
            },
            "broker": {
                "adapter": "fake",
                "open_orders": [
                    {
                        "symbol": "AMAT",
                        "status": "open",
                    },
                ],
            },
        },
        feed=None,
    )
    decision = SimpleNamespace(
        cash=500,
        current_positions={},
        orders=[
            SimpleNamespace(symbol="AMAT", price=100),
        ],
    )

    assert service._broker_blocked_reason(decision) == (
        "broker_open_order_conflict:AMAT"
    )


def test_broker_reconciliation_flags_cash_mismatch(tmp_path):
    service = PaperTradingService(
        config={
            "paper_trading": {
                "execution_adapter": "broker",
                "report_dir": str(tmp_path),
                "submit_orders": True,
            },
            "broker": {
                "adapter": "fake",
                "starting_cash": 400,
                "cash_tolerance": 1.0,
            },
        },
        feed=None,
    )
    decision = SimpleNamespace(
        timestamp=datetime(2026, 6, 1),
        cash=500,
        current_positions={},
        orders=[],
    )

    reconciliation = service._broker_reconciliation(decision)

    assert reconciliation["passed"] is False
    assert reconciliation["mismatches"][0]["reason"] == "cash_mismatch"


def test_broker_reconciliation_flags_position_mismatch(tmp_path):
    service = PaperTradingService(
        config={
            "paper_trading": {
                "execution_adapter": "broker",
                "report_dir": str(tmp_path),
            },
            "broker": {
                "adapter": "fake",
                "starting_cash": 500,
                "positions": {"AMAT": 0.25},
                "position_tolerance": 0.000001,
            },
        },
        feed=None,
    )
    decision = SimpleNamespace(
        timestamp=datetime(2026, 6, 1),
        cash=500,
        current_positions={"AMAT": 0.50},
        orders=[],
    )

    reconciliation = service._broker_reconciliation(decision)

    assert reconciliation["passed"] is False
    assert reconciliation["mismatches"][0]["reason"] == "position_mismatch"


def test_broker_reconciliation_report_is_written(tmp_path):
    service = PaperTradingService(
        config={
            "paper_trading": {
                "execution_adapter": "broker",
                "report_dir": str(tmp_path),
            },
            "broker": {"adapter": "fake"},
        },
        feed=None,
    )
    decision = SimpleNamespace(
        timestamp=datetime(2026, 6, 1),
    )
    reconciliation = {
        "passed": True,
        "timestamp": "2026-06-19T00:00:00",
        "broker_adapter": "fake",
        "local_cash": 500,
        "broker_cash": 500,
        "local_positions": {},
        "broker_positions": {},
        "open_orders": [],
        "mismatches": [],
    }

    path = service._save_broker_reconciliation_report(decision, reconciliation)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["passed"] is True
    assert payload["broker_adapter"] == "fake"


def test_reproducibility_metadata_hashes_candidate_config(tmp_path):
    candidate_path = tmp_path / "candidate.yaml"
    candidate_path.write_text("champion_id: test\n", encoding="utf-8")
    service = PaperTradingService(
        config={
            "paper_trading": {
                "execution_adapter": "local_ledger",
                "report_dir": str(tmp_path),
            },
            "broker": {"adapter": "fake"},
            "research": {
                "dual_momentum": {
                    "champion_config_path": str(candidate_path),
                },
            },
        },
        feed=None,
    )

    metadata = service._reproducibility_metadata("candidate_a")

    assert metadata["candidate_id"] == "candidate_a"
    assert metadata["candidate_config_path"] == str(candidate_path)
    assert metadata["candidate_config_hash"]
    assert metadata["config_hash"]


def test_metrics_summary_is_written_with_reproducibility(tmp_path):
    service = PaperTradingService(
        config={
            "paper_trading": {
                "report_dir": str(tmp_path),
                "dashboard_path": str(tmp_path / "dashboard.csv"),
            },
        },
        feed=None,
    )
    decision = SimpleNamespace(
        timestamp=datetime(2026, 6, 1),
        equity=500,
        cash=100,
        orders=[],
    )

    path = service._save_metrics_summary(
        run_id="run_1",
        candidate_id="candidate",
        decision=decision,
        fill_record=None,
        risk_checks=[],
        post_trade_checks=[],
        blocked_reason=None,
        reproducibility={"config_hash": "abc123"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["run_id"] == "run_1"
    assert payload["latest_equity"] == 500
    assert payload["reproducibility"]["config_hash"] == "abc123"


def test_submit_with_fake_broker_updates_paper_state(tmp_path):
    decision_path = tmp_path / "2026-06-01_decision.json"
    decision_path.write_text(
        json.dumps({
            "timestamp": "2026-06-01T04:00:00+00:00",
            "cash": 500,
            "equity": 500,
            "orders": [],
            "model_context": {"strategy": "dual_momentum"},
        }),
        encoding="utf-8",
    )
    service = PaperTradingService(
        config={
                "paper_trading": {
                    "execution_adapter": "broker",
                    "report_dir": str(tmp_path),
                    "fill_log_path": str(tmp_path / "fills.csv"),
                    "min_trade_value": 1,
                    "submit_orders": True,
                },
            "broker": {
                "adapter": "fake",
                "partial_fill_ratio": 1.0,
            },
            "backtest": {
                "starting_equity": 500,
            },
        },
        feed=None,
    )
    decision = SimpleNamespace(
        timestamp=datetime(2026, 6, 1),
        report_path=decision_path,
        cash=500,
        current_positions={},
        orders=[
            PaperOrder(
                symbol="AMAT",
                side="BUY",
                quantity_delta=0.5,
                dollar_delta=50,
                current_weight=0,
                target_weight=0.1,
                drift_weight=0.1,
                price=100,
                reason="test order",
            ),
        ],
    )

    fill_record = service._submit_with_broker(decision)
    state = json.loads((tmp_path / "paper_state.json").read_text(encoding="utf-8"))

    assert fill_record["status"] == "filled"
    assert fill_record["cash_after"] == 450
    assert fill_record["equity_after"] == 500
    assert state["positions"]["AMAT"] == 0.5
    assert Path(tmp_path / "fills.csv").exists()


def test_run_with_broker_execution_adapter_submits_orders(monkeypatch, tmp_path):
    decision_path = tmp_path / "2026-06-01_decision.json"
    decision_path.write_text(
        json.dumps({
            "timestamp": "2026-06-01T04:00:00+00:00",
            "cash": 500,
            "equity": 500,
            "orders": [],
            "model_context": {"strategy": "dual_momentum"},
        }),
        encoding="utf-8",
    )

    decision = SimpleNamespace(
        timestamp=datetime(2026, 6, 1),
        report_path=decision_path,
        equity=500,
        cash=500,
        exposure_target=1.0,
        target_weights={"AMAT": 0.1},
        current_positions={},
        data_freshness={"is_stale": False},
        orders=[
            PaperOrder(
                symbol="AMAT",
                side="BUY",
                quantity_delta=0.5,
                dollar_delta=50,
                current_weight=0,
                target_weight=0.1,
                drift_weight=0.1,
                price=100,
                reason="test order",
            ),
        ],
    )

    monkeypatch.setattr(
        "application.services.paper_trading_service.create_paper_decision",
        lambda config, feed, tester: decision,
    )
    monkeypatch.setattr(
        "application.services.paper_trading_service.PaperTradingService._approval_error",
        lambda self, target_hash, order_hash: None,
    )

    service = PaperTradingService(
        config={
            "paper_trading": {
                "execution_adapter": "broker",
                "report_dir": str(tmp_path),
                "fill_log_path": str(tmp_path / "fills.csv"),
                "min_trade_value": 1,
            },
            "broker": {
                "adapter": "fake",
                "partial_fill_ratio": 1.0,
            },
            "backtest": {
                "starting_equity": 500,
            },
        },
        feed=None,
    )

    result = service.run(dry_run=False, submit=True)

    assert result.blocked_reason is None
    assert result.fill_record is not None
    assert result.fill_record["status"] == "filled"
    assert Path(tmp_path / "fills.csv").exists()
    assert result.reconciliation_report_path is not None
    assert result.reconciliation_report_path.exists()


def test_sync_broker_state_uses_sleeve_cash_and_imports_positions(tmp_path):
    engine = PaperTradingEngine(
        report_dir=tmp_path,
        starting_cash=500,
    )

    sync = engine.sync_broker_state(
        account={
            "cash": 99552.07,
            "equity": 100000.0,
            "buying_power": 399469.3,
        },
        positions={"AAPL": 0.382076, "QQQ": 0.17501},
        prices_by_symbol={"AAPL": 293.32, "QQQ": 710.6},
        sleeve_cash=500,
        source="alpaca",
    )

    state = json.loads((tmp_path / "paper_state.json").read_text(encoding="utf-8"))
    expected_position_value = (0.382076 * 293.32) + (0.17501 * 710.6)

    assert sync["positions"] == {"AAPL": 0.382076, "QQQ": 0.17501}
    assert sync["cash"] == 500 - expected_position_value
    assert state["positions"] == {"AAPL": 0.382076, "QQQ": 0.17501}
    assert state["cash"] == 500 - expected_position_value
    assert state["last_broker_sync"]["cash_source"] == (
        "sleeve_cash_minus_broker_position_value"
    )


def test_broker_reconciliation_skips_account_cash_mismatch_for_sleeve(tmp_path):
    service = PaperTradingService(
        config={
            "paper_trading": {
                "execution_adapter": "broker",
                "report_dir": str(tmp_path),
            },
            "broker": {
                "adapter": "fake",
                "starting_cash": 99552.07,
                "positions": {"AAPL": 0.382076},
                "sleeve_cash": 500,
                "cash_reconciliation": "sleeve",
                "position_tolerance": 0.000001,
            },
        },
        feed=None,
    )
    decision = SimpleNamespace(
        timestamp=datetime(2026, 6, 1),
        cash=387.92025668,
        current_positions={"AAPL": 0.382076},
        orders=[],
    )

    reconciliation = service._broker_reconciliation(decision)

    assert reconciliation["passed"] is True
    assert reconciliation["cash_reconciliation"] == "sleeve"
    assert reconciliation["mismatches"] == []

