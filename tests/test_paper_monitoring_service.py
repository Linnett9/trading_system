import csv
import json

from application.services.paper_monitoring_service import PaperMonitoringService


def test_weekly_summary_and_promotion_checklist_are_written(tmp_path):
    report_dir = tmp_path / "paper"
    report_dir.mkdir()
    dashboard_path = report_dir / "dashboard.csv"
    journal_path = report_dir / "journal.csv"

    with dashboard_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date",
                "run_id",
                "portfolio_value",
                "cash",
                "exposure",
                "orders",
                "risk_status",
                "post_trade_status",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "date": "2026-06-19",
            "run_id": "run_1",
            "portfolio_value": "500",
            "cash": "100",
            "exposure": "0.8",
            "orders": "0",
            "risk_status": "PASS",
            "post_trade_status": "PASS",
        })

    with journal_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "risk_status",
                "post_trade_status",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "timestamp": "2026-06-19T00:00:00",
            "risk_status": "PASS",
            "post_trade_status": "PASS",
            "notes": "",
        })

    service = PaperMonitoringService({
        "paper_trading": {
            "report_dir": str(report_dir),
            "dashboard_path": str(dashboard_path),
            "fill_log_path": str(tmp_path / "fills.csv"),
        },
    })

    summary = service.weekly_summary()
    checklist = service.promotion_checklist()

    assert summary.exists()
    assert checklist.exists()
    assert "Weekly Paper Summary" in summary.read_text(encoding="utf-8")
    assert "Paper Trading Promotion Checklist" in checklist.read_text(
        encoding="utf-8",
    )


def test_weekly_summary_uses_state_when_dashboard_equity_is_cash_only(tmp_path):
    report_dir = tmp_path / "paper"
    report_dir.mkdir()
    dashboard_path = report_dir / "dashboard.csv"
    journal_path = report_dir / "journal.csv"

    with dashboard_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date",
                "run_id",
                "portfolio_value",
                "cash",
                "exposure",
                "orders",
                "risk_status",
                "post_trade_status",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "date": "2026-06-19",
            "run_id": "run_1",
            "portfolio_value": "162.5",
            "cash": "162.5",
            "exposure": "0.0",
            "orders": "0",
            "risk_status": "PASS",
            "post_trade_status": "PASS",
        })

    with journal_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "risk_status",
                "post_trade_status",
                "notes",
            ],
        )
        writer.writeheader()

    (report_dir / "paper_state.json").write_text(
        json.dumps({
            "cash": 162.5,
            "positions": {"AMAT": 0.13},
            "equity_history": [
                {
                    "timestamp": "2026-06-01T04:00:00+00:00",
                    "equity": 500.0,
                    "cash": 162.5,
                },
            ],
        }),
        encoding="utf-8",
    )

    service = PaperMonitoringService({
        "paper_trading": {
            "report_dir": str(report_dir),
            "dashboard_path": str(dashboard_path),
            "fill_log_path": str(tmp_path / "fills.csv"),
        },
    })

    summary = service.weekly_summary()
    text = summary.read_text(encoding="utf-8")

    assert "Latest equity: 500.0000" in text
    assert "Latest cash: 162.5000" in text
    assert "Latest exposure: 0.6750" in text
