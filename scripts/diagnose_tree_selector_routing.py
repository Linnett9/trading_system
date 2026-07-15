from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.research.ml.stock_level.bounded_selector_runner import BoundedSelectorSettings
from core.research.ml.stock_level.tree_selector_diagnostics import run_tree_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True); parser.add_argument("--feature-schema", type=Path, required=True); parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--decision-date", required=True); parser.add_argument("--model", choices=("random_forest", "gradient_boosting"), required=True)
    group = parser.add_mutually_exclusive_group(); group.add_argument("--training-start-date"); group.add_argument("--trailing-sessions", type=int)
    parser.add_argument("--estimators", type=int, default=3); parser.add_argument("--max-depth", type=int, required=True); parser.add_argument("--min-samples-leaf", type=int, required=True); parser.add_argument("--learning-rate", type=float, default=0.05); parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    prefix = "random_forest" if args.model == "random_forest" else "gradient_boosting"
    smoke = {f"{prefix}_n_estimators": args.estimators, f"{prefix}_max_depth": args.max_depth}
    if args.model == "random_forest": smoke["random_forest_min_samples_leaf"] = args.min_samples_leaf
    else: smoke["gradient_boosting_learning_rate"] = args.learning_rate
    config = {"ml": {"stock_selector_bounded": {"dataset_root": str(args.dataset_root), "output_root": str(args.output_root), "oos_start_date": args.decision_date, "oos_end_date": args.decision_date, "model_allowlist": [args.model], "baseline_allowlist": [], "sklearn_n_jobs": args.workers, "smoke_overrides": smoke}}}
    settings = BoundedSelectorSettings.from_config(config)
    result = run_tree_diagnostic(dataset_root=args.dataset_root, feature_schema_path=args.feature_schema, output_root=args.output_root, decision_date=args.decision_date, model_id=args.model, settings=settings, requested_start_date=args.training_start_date, trailing_sessions=args.trailing_sessions)
    print(json.dumps({key: result[key] for key in ("model_id", "training_window", "training_row_count", "training_date_min", "training_date_max", "oos_row_count", "fit_seconds", "prediction_seconds", "prediction_quality_accepted", "prediction_quality", "output_root") if key in result}, indent=2, default=str))


if __name__ == "__main__": main()
