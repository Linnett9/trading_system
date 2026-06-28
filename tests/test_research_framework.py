from __future__ import annotations

import inspect
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from core.research import framework
from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.parallel import ParallelTaskExecutor
from core.research.framework.registry import FeatureRegistry, ModelRegistry
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.framework.walk_forward import (
    ExpandingWindowSpec,
    ExpandingWindowSplitter,
)


def test_stock_level_typed_config_preserves_existing_keys(tmp_path):
    settings = StockLevelResearchConfig.from_mapping(
        {
            "ml": {
                "output_dir": str(tmp_path / "reports"),
                "stock_level_base_prediction_artifacts_path": "base.csv",
                "stock_level_prediction_artifacts_path": "enriched.csv",
                "stock_ranker_min_train_dates": 12,
                "stock_ranker_test_window_dates": 4,
                "stock_ranker_embargo_dates": 2,
                "stock_ranker_model_n_jobs": 3,
                "stock_ranker_include_engineered_features": True,
            }
        }
    )

    assert str(settings.base_artifact_path) == "base.csv"
    assert str(settings.artifact_path) == "enriched.csv"
    assert settings.min_train_dates == 12
    assert settings.test_window_dates == 4
    assert settings.embargo_dates == 2
    assert settings.model_n_jobs == 3
    assert settings.include_engineered_features is True


def test_expanding_window_splitter_is_chronological_and_embargoed():
    rows = [
        {"rebalance_date": f"2024-01-{index + 1:02d}", "value": index}
        for index in range(8)
    ]
    splitter = ExpandingWindowSplitter(
        ExpandingWindowSpec(
            min_train_dates=3,
            test_window_dates=2,
            embargo_dates=1,
        )
    )

    folds = splitter.split(rows)

    assert len(folds) == 2
    assert folds[0].train_dates == (
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
    )
    assert folds[0].embargoed_dates == ("2024-01-04",)
    assert folds[0].test_dates == ("2024-01-05", "2024-01-06")
    assert all(fold.chronological_guard_passed for fold in folds)


def test_component_registries_preserve_order_and_reject_duplicates():
    features: FeatureRegistry[str] = FeatureRegistry()
    models: ModelRegistry[str] = ModelRegistry()
    features.register("momentum", "feature", metadata={"family": "alpha"})
    models.register("ridge", "model")

    assert features.names() == ("momentum",)
    assert features.get("momentum") == "feature"
    assert features.metadata("momentum") == {"family": "alpha"}
    assert models.names() == ("ridge",)
    with pytest.raises(ValueError, match="already registered"):
        features.register("momentum", "duplicate")


def test_parallel_executor_preserves_order_and_isolates_errors():
    executor = ParallelTaskExecutor[int, int]()

    def worker(value: int) -> int:
        if value == 2:
            raise RuntimeError("synthetic failure")
        return value * value

    result = executor.execute(
        [3, 2, 1],
        worker,
        key=str,
        max_workers=3,
        executor_cls=ThreadPoolExecutor,
    )

    assert list(result.results) == ["3", "1"]
    assert result.results == {"3": 9, "1": 1}
    assert "synthetic failure" in result.errors["2"]


def test_research_artifact_writer_preserves_json_and_csv_shapes(tmp_path):
    writer = ResearchArtifactWriter()
    json_path = tmp_path / "nested" / "report.json"
    csv_path = tmp_path / "nested" / "report.csv"
    markdown_path = tmp_path / "nested" / "report.md"

    writer.write_json(json_path, {"value": 1})
    writer.write_csv(csv_path, [{"a": 1, "b": 2}], fieldnames=["a", "b"])
    writer.write_markdown(markdown_path, "# Report\n")

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"value": 1}
    assert csv_path.read_text(encoding="utf-8") == "a,b\n1,2\n"
    assert markdown_path.read_text(encoding="utf-8") == "# Report\n"


def test_research_framework_has_no_operational_imports():
    source = inspect.getsource(framework)
    for module in (
        "contracts",
        "config",
        "data",
        "logging",
        "parallel",
        "ranking",
        "registry",
        "reporting",
        "walk_forward",
    ):
        imported = __import__(f"core.research.framework.{module}", fromlist=[module])
        source += inspect.getsource(imported)

    assert "infrastructure.broker" not in source
    assert "paper_trading" not in source
    assert "live_trading" not in source
    assert "core.entities.order" not in source
