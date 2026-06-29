import json
import os

from core.research.framework.config import StockLevelResearchConfig
from core.research.ml.runtime_parallelism import THREAD_ENV_VARS, apply_stock_alpha_worker_caps
from core.research.ml.stock_level.stock_alpha_parallelism_audit import write_stock_alpha_parallelism_audit


def test_benchmark_defaults_resolve_four_workers():
    settings = StockLevelResearchConfig.from_mapping({"ml": {"stock_alpha_run_size": "benchmark"}})
    assert settings.alpha_feature_n_jobs == 4
    assert settings.model_n_jobs == 4
    assert settings.target_comparison_n_jobs == 4


def test_parallelism_audit_writes_reports(tmp_path):
    result = write_stock_alpha_parallelism_audit({"ml": {"output_dir": str(tmp_path), "stock_alpha_run_size": "benchmark", "stock_alpha_feature_n_jobs": 4, "stock_ranker_model_n_jobs": 4, "stock_target_comparison_n_jobs": 4, "stock_portfolio_policy_sweep_n_jobs": 4, "sklearn_n_jobs": 1, "torch_num_threads": 1, "numpy_num_threads": 1}})
    assert result.json_path.exists() and result.markdown_path.exists()
    payload = json.loads(result.json_path.read_text())
    assert payload["stages"]["model_ranking"]["requested_workers"] == 4
    assert payload["stages"]["model_ranking"]["nested_worker_caps"]["sklearn_n_jobs"] == 1
    assert payload["promotion_thresholds_changed"] is False


def test_worker_caps_set_native_environment():
    caps = apply_stock_alpha_worker_caps({"ml": {"numpy_num_threads": 1, "torch_num_threads": 1, "sklearn_n_jobs": 1}})
    assert caps == {"numpy_num_threads": 1, "torch_num_threads": 1, "sklearn_n_jobs": 1}
    assert all(os.environ[name] == "1" for name in THREAD_ENV_VARS)
