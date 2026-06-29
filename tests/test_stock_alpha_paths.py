from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir


def test_stock_alpha_roots_are_deterministic_and_separated(tmp_path):
    root = tmp_path / "stock_alpha"
    assert stock_alpha_output_dir({"ml": {"stock_alpha_report_root": str(root), "stock_alpha_run_size": "dev"}}) == root / "dev"
    assert stock_alpha_output_dir({"ml": {"stock_alpha_report_root": str(root), "stock_alpha_run_size": "benchmark"}}) == root / "benchmark"
    assert stock_alpha_output_dir({"ml": {"stock_alpha_report_root": str(root), "stock_alpha_run_size": "full"}}) == root / "full"
