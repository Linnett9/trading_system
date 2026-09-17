from __future__ import annotations

import json

import pandas as pd

from scripts.local import ds24_cost_aware_reconstruct_r2 as reconstruction


def test_producer_spine_excludes_extra_feature_timestamps(tmp_path, monkeypatch) -> None:
    metrics_root = tmp_path / "huber" / "trace"
    metrics_root.mkdir(parents=True)
    path = metrics_root / "part.parquet"
    pd.DataFrame({"decision_timestamp": [
        "2025-04-02T13:35:00+00:00", "2025-04-02T13:40:00+00:00",
    ]}).to_parquet(path, index=False)
    (metrics_root / "decision_trace_v3_manifest.json").write_text(json.dumps({"parts": [{
        "path": "part.parquet",
        "min_decision_timestamp": "2025-04-02T13:35:00+00:00",
        "max_decision_timestamp": "2025-04-02T13:40:00+00:00",
    }]}), encoding="utf-8")
    monkeypatch.setattr(reconstruction, "WORKERS", tmp_path)
    monkeypatch.setattr(reconstruction, "TRACE_ROOTS", {"huber": "trace"})

    accepted, _ = reconstruction._accepted_score_timestamps("huber", "2025-04-02", "2025-04-02")
    causal_features = pd.DataFrame({"decision_timestamp": pd.to_datetime([
        "2025-04-02T13:35:00+00:00", "2025-04-02T13:40:00+00:00",
        "2025-04-02T20:05:00+00:00",
    ], utc=True)})
    admitted = causal_features[causal_features["decision_timestamp"].isin(accepted)]
    assert len(admitted) == 2
    assert pd.Timestamp("2025-04-02T20:05:00Z") not in accepted


def test_model_vintage_catalog_is_bounded_without_eager_unpickling(tmp_path, monkeypatch) -> None:
    models = tmp_path / "huber/models"
    models.mkdir(parents=True)
    (models / "huber_20250402.pkl").write_bytes(b"catalogued, not loaded")
    monkeypatch.setattr(reconstruction, "WORKERS", tmp_path)

    catalog = reconstruction._model_vintages("huber", "2025-04-02", "2025-04-02")

    assert list(catalog) == ["2025-04-02"]
    assert catalog["2025-04-02"][0] == models / "huber_20250402.pkl"
