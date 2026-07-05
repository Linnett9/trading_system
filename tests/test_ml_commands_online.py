import csv
from datetime import datetime, timedelta

from application.services.ml_commands_online import run_ml_online_intraday_benchmark


def test_online_command_writes_research_only_comparison(tmp_path):
    source = tmp_path / "online.csv"
    output = tmp_path / "result.json"
    start = datetime(2025, 1, 2, 9, 30)
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "observation_id", "observed_at", "label_available_at",
            "label", "momentum", "volatility",
        ])
        writer.writeheader()
        for index in range(20):
            observed = start + timedelta(minutes=5 * index)
            writer.writerow({
                "observation_id": index,
                "observed_at": observed.isoformat(),
                "label_available_at": (observed + timedelta(minutes=10)).isoformat(),
                "label": int(index % 4 >= 2),
                "momentum": index % 7,
                "volatility": index % 5,
            })
    payload = run_ml_online_intraday_benchmark({"ml": {"online_intraday": {
        "dataset_path": str(source),
        "output_path": str(output),
        "minimum_training_samples": 4,
        "periodic_refit_every_bars": 4,
        "include_warm_start_neural": False,
    }}})

    assert output.exists()
    assert payload["observation_count"] == 20
    assert {row["model"] for row in payload["models"]} == {
        "frozen_logistic", "online_logistic", "periodic_refit_logistic"
    }
    assert all(row["temporal_leakage_check_passed"] for row in payload["models"])
