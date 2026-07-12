from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.ml.artifacts.artifact_writers import MLCoreArtifactWriter
from core.research.ml.immutable_runs import (
    deterministic_run_id,
    file_digest,
    preserve_immutable_run,
    read_run_manifest,
    run_dir_from_latest_completed,
)
from core.research.ml.stock_level_benchmark_data import _available_feature_columns, _prepare_rows
from core.research.ml.stock_level_benchmark_models import _build_tabular_model
from core.research.ml.stock_level_benchmark_types import (
    PREDICTION_PREFIX,
    TABULAR_MODEL_NAMES,
    TARGET_COLUMN,
    TARGET_PROVENANCE_COLUMNS,
    TARGET_PROVENANCE_CONTRACT_VERSION,
)

ARTIFACT_SCHEMA_VERSION = "stock_selector_final_fit_v1"
TEMPORAL_POLICY_VERSION = 1
REQUIRED_METADATA_COLUMNS = (
    "rebalance_date",
    "symbol",
    "label_available_timestamp",
    "label_end_timestamp",
    "target_provenance_contract_version",
)


@dataclass(frozen=True)
class FinalFittedSelectorPaths:
    output_dir: Path
    run_dir: Path
    model_path: Path
    feature_contract_path: Path
    training_manifest_path: Path
    oos_selection_evidence_path: Path
    audit_path: Path
    latest_completed_path: Path


@dataclass(frozen=True)
class LoadedFinalFittedSelector:
    run_dir: Path
    manifest: dict[str, Any]
    feature_contract: dict[str, Any]
    model: Any

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return tuple(self.feature_contract["ordered_feature_columns"])

    @property
    def prediction_column(self) -> str:
        return str(self.feature_contract["selected_signal_column"])

    def predict_rows(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        matrix = self._matrix(rows, feature_columns=self.feature_columns)
        values = [float(value) for value in self.model.predict(matrix)]
        output = []
        for row, value in zip(rows, values):
            output.append(
                {
                    "rebalance_date": row.get("rebalance_date", ""),
                    "symbol": row.get("symbol", ""),
                    self.prediction_column: value,
                }
            )
        return output

    def predict_feature_matrix(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        feature_columns: Sequence[str],
    ) -> list[float]:
        if tuple(feature_columns) != self.feature_columns:
            raise ValueError("Final fitted selector feature order is incompatible")
        return [float(value) for value in self.model.predict(self._matrix(rows, feature_columns=feature_columns))]

    def _matrix(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        feature_columns: Sequence[str],
    ) -> list[list[float]]:
        missing = sorted(
            {
                column
                for row in rows
                for column in feature_columns
                if column not in row
            }
        )
        if missing:
            raise ValueError(f"Missing final fitted selector features: {missing}")
        return [[float(row[column]) for column in feature_columns] for row in rows]


def write_final_fitted_stock_selector(config: dict[str, Any]) -> FinalFittedSelectorPaths:
    ml = dict(config.get("ml", {}) or {})
    selector_output_dir = Path(ml.get("output_dir", "reports/ml/stock_selector"))
    final_output_dir = Path(
        ml.get("stock_selector_final_fit_output_dir", selector_output_dir / "final_fitted_selector")
    )
    source_path = Path(
        ml.get(
            "stock_level_prediction_artifacts_path",
            selector_output_dir / "stock_level_prediction_artifacts.csv",
        )
    )
    benchmark_path = Path(
        ml.get(
            "stock_level_model_ranking_benchmark_path",
            selector_output_dir / "stock_level_model_ranking_benchmark.json",
        )
    )
    leaderboard_path = selector_output_dir / "stock_level_model_ranking_benchmark.csv"
    oos_predictions_path = Path(
        ml.get(
            "stock_level_model_oos_predictions_path",
            selector_output_dir / "stock_level_model_oos_predictions.csv",
        )
    )
    cutoff = _final_fit_cutoff(ml, benchmark_path)
    benchmark = _read_json(benchmark_path)
    leaderboard = _read_csv(leaderboard_path)
    selected = _selected_oos_winner(benchmark, leaderboard)
    feature_columns = tuple(str(column) for column in benchmark.get("feature_columns", []))
    if not feature_columns:
        source_rows_for_features = _read_csv(source_path)
        feature_columns = _available_feature_columns(
            source_rows_for_features,
            include_engineered=bool(ml.get("stock_ranker_include_engineered_features", False)),
        )
    source_rows = _read_csv(source_path)
    prepared_rows, excluded_incomplete = _prepare_rows(source_rows, feature_columns)
    eligible_rows = _eligible_final_fit_rows(prepared_rows, cutoff)
    if not eligible_rows:
        raise RuntimeError(
            "No stock selector rows are label-eligible for final fitting: "
            f"final_fit_decision_timestamp={cutoff}"
        )
    model = _build_tabular_model(
        selected["model_name"],
        int(ml.get("random_seed", benchmark.get("random_seed", 42) or 42)),
        int(ml.get("sklearn_n_jobs", 1)),
    )
    x_train = [[row[column] for column in feature_columns] for row in eligible_rows]
    y_train = [row[TARGET_COLUMN] for row in eligible_rows]
    model.fit(x_train, y_train)
    in_memory_probe = [float(value) for value in model.predict(x_train[: min(5, len(x_train))])]

    final_output_dir.mkdir(parents=True, exist_ok=True)
    model_path = final_output_dir / "final_model.pkl"
    feature_contract_path = final_output_dir / "feature_contract.json"
    training_manifest_path = final_output_dir / "training_manifest.json"
    evidence_path = final_output_dir / "oos_selection_evidence.json"
    audit_path = final_output_dir / "final_fitted_selector_audit.json"

    model_path.write_bytes(pickle.dumps(model))
    source_evidence = _source_oos_evidence(
        selector_output_dir=selector_output_dir,
        benchmark_path=benchmark_path,
        leaderboard_path=leaderboard_path,
        oos_predictions_path=oos_predictions_path,
        selected=selected,
    )
    dataset_hash = file_digest(source_path)
    model_input_hash = _model_input_hash(eligible_rows, feature_columns)
    config_hash = MLCoreArtifactWriter.hash_payload(config)
    feature_contract = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "ordered_feature_columns": list(feature_columns),
        "feature_count": len(feature_columns),
        "target_name": TARGET_COLUMN,
        "target_horizon": _single_value(eligible_rows, "target_horizon"),
        "target_provenance_contract_version": TARGET_PROVENANCE_CONTRACT_VERSION,
        "required_metadata_columns": list(REQUIRED_METADATA_COLUMNS),
        "selected_model_name": selected["model_name"],
        "selected_signal_column": selected["selected_signal_column"],
        "model_configuration": selected["model_configuration"],
        "random_seed": int(ml.get("random_seed", benchmark.get("random_seed", 42) or 42)),
        "final_fit_decision_timestamp": cutoff,
        "training_row_count": len(eligible_rows),
        "symbol_universe": _symbol_universe(eligible_rows),
        "dataset_hash": dataset_hash,
        "model_input_hash": model_input_hash,
        "resolved_config_hash": config_hash,
        "temporal_policy_identity": _temporal_policy_identity(cutoff),
        "source_oos_evidence_identity": source_evidence["identity"],
    }
    training_manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "final_fit_decision_timestamp": cutoff,
        "candidate_input_row_count": len(prepared_rows),
        "eligible_training_row_count": len(eligible_rows),
        "excluded_incomplete_row_count": excluded_incomplete,
        "excluded_immature_label_count": len(prepared_rows) - len(eligible_rows),
        "training_feature_date_min": min(row["rebalance_date"] for row in eligible_rows),
        "training_feature_date_max": max(row["rebalance_date"] for row in eligible_rows),
        "training_label_available_max": max(row["label_available_timestamp"] for row in eligible_rows),
        "symbol_count": len(_symbol_universe(eligible_rows)),
        "symbol_universe": _symbol_universe(eligible_rows),
        "target_horizon": feature_contract["target_horizon"],
        "target_provenance_contract_version": TARGET_PROVENANCE_CONTRACT_VERSION,
        "temporal_policy_version": TEMPORAL_POLICY_VERSION,
        "dataset_hash": dataset_hash,
        "model_input_hash": model_input_hash,
    }
    _write_json(feature_contract_path, feature_contract)
    _write_json(training_manifest_path, training_manifest)
    _write_json(evidence_path, source_evidence)

    reloaded_model = pickle.loads(model_path.read_bytes())
    reloaded_probe = [
        float(value)
        for value in reloaded_model.predict(x_train[: min(5, len(x_train))])
    ]
    equivalent = _equivalent(in_memory_probe, reloaded_probe)
    audit = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "completion_status": "complete" if equivalent else "failed_reload_equivalence",
        "selected_model_name": selected["model_name"],
        "selected_model_configuration": selected["model_configuration"],
        "selected_signal_column": selected["selected_signal_column"],
        "source_oos_evidence": source_evidence,
        "final_fit_decision_timestamp": cutoff,
        "candidate_input_row_count": len(prepared_rows),
        "eligible_training_row_count": len(eligible_rows),
        "excluded_immature_label_count": len(prepared_rows) - len(eligible_rows),
        "training_feature_date_min": training_manifest["training_feature_date_min"],
        "training_feature_date_max": training_manifest["training_feature_date_max"],
        "training_label_available_max": training_manifest["training_label_available_max"],
        "feature_columns": list(feature_columns),
        "feature_count": len(feature_columns),
        "symbol_universe": training_manifest["symbol_universe"],
        "symbol_count": training_manifest["symbol_count"],
        "dataset_hash": dataset_hash,
        "model_input_hash": model_input_hash,
        "reload_prediction_equivalence": {
            "passed": equivalent,
            "in_memory": in_memory_probe,
            "reloaded": reloaded_probe,
        },
        "deep_selector_models_enabled": False,
        "news_enabled": False,
        "champion_pointer_updated": False,
        "research_only": True,
        "trading_impact": "none",
    }
    _write_json(audit_path, audit)
    if not equivalent:
        raise RuntimeError("Reloaded final fitted selector predictions do not match in-memory predictions")

    identity = _final_fit_identity(
        selected=selected,
        feature_contract=feature_contract,
        training_manifest=training_manifest,
        source_evidence=source_evidence,
    )
    run_id = deterministic_run_id("stock_selector_final_fit", identity)
    record = preserve_immutable_run(
        output_dir=final_output_dir,
        run_id=run_id,
        kind="stock_selector_final_fit",
        identity=identity,
        artifact_paths=(
            model_path,
            feature_contract_path,
            training_manifest_path,
            evidence_path,
            audit_path,
        ),
        extra_manifest={
            "selected_model_name": selected["model_name"],
            "selected_signal_column": selected["selected_signal_column"],
            "champion_pointer_updated": False,
        },
    )
    return FinalFittedSelectorPaths(
        output_dir=final_output_dir,
        run_dir=record.run_dir,
        model_path=record.run_dir / model_path.name,
        feature_contract_path=record.run_dir / feature_contract_path.name,
        training_manifest_path=record.run_dir / training_manifest_path.name,
        oos_selection_evidence_path=record.run_dir / evidence_path.name,
        audit_path=record.run_dir / audit_path.name,
        latest_completed_path=record.latest_completed_path,
    )


def load_final_fitted_stock_selector(path: Path) -> LoadedFinalFittedSelector:
    run_dir = _resolve_run_dir(path)
    manifest = read_run_manifest(run_dir)
    if not manifest or manifest.get("run_status") != "complete":
        raise RuntimeError(f"Final fitted selector run is incomplete: {run_dir}")
    feature_contract = _read_json(run_dir / "feature_contract.json")
    if feature_contract.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("Final fitted selector artifact schema is incompatible")
    if feature_contract.get("target_provenance_contract_version") != TARGET_PROVENANCE_CONTRACT_VERSION:
        raise RuntimeError("Final fitted selector target provenance contract is incompatible")
    model_path = run_dir / "final_model.pkl"
    try:
        model = pickle.loads(model_path.read_bytes())
    except Exception as exc:  # pragma: no cover - corrupt pickle branch
        raise RuntimeError(f"Final fitted selector model artifact is corrupt: {model_path}") from exc
    return LoadedFinalFittedSelector(
        run_dir=run_dir,
        manifest=manifest,
        feature_contract=feature_contract,
        model=model,
    )


def assert_final_fit_reuse_compatible(path: Path, expected_identity: Mapping[str, Any]) -> None:
    run_dir = _resolve_run_dir(path)
    manifest = read_run_manifest(run_dir)
    if not manifest or manifest.get("run_status") != "complete":
        raise RuntimeError(f"Final fitted selector run is incomplete: {run_dir}")
    identity = manifest.get("identity", {})
    mismatches = sorted(
        key
        for key, value in expected_identity.items()
        if identity.get(key) != value
    )
    if mismatches:
        raise RuntimeError(f"Final fitted selector reuse identity mismatch: {mismatches}")


def _resolve_run_dir(path: Path) -> Path:
    if (path / "run_manifest.json").exists():
        return path
    latest = run_dir_from_latest_completed(path)
    if latest is None:
        raise RuntimeError(f"No completed final fitted selector run found: {path}")
    return latest


def _selected_oos_winner(
    benchmark: Mapping[str, Any],
    leaderboard: list[dict[str, str]],
) -> dict[str, Any]:
    best = benchmark.get("best_ml_model") or {}
    model_name = str(best.get("name") or "").strip()
    if model_name not in TABULAR_MODEL_NAMES:
        raise RuntimeError(
            "Final fitted stock selector supports only active tabular OOS winners: "
            f"{TABULAR_MODEL_NAMES}; selected={model_name!r}"
        )
    selected_signal = str(best.get("signal_column") or f"{PREDICTION_PREFIX}{model_name}")
    leaderboard_row = next((row for row in leaderboard if row.get("name") == model_name), None)
    if leaderboard_row is None:
        raise RuntimeError(f"Selected OOS winner is missing from leaderboard CSV: {model_name}")
    return {
        "model_name": model_name,
        "selected_signal_column": selected_signal,
        "model_configuration": {
            "model_name": model_name,
            "family": "tabular_regressor",
        },
        "oos_ranking_metric": "mean_spearman_ic",
        "oos_metric_value": best.get("mean_spearman_ic"),
        "leaderboard_row": dict(leaderboard_row),
    }


def _eligible_final_fit_rows(
    rows: list[dict[str, Any]],
    cutoff: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row["rebalance_date"]) <= cutoff
        and str(row["label_available_timestamp"]) <= cutoff
        and str(row.get("target_provenance_contract_version")) == TARGET_PROVENANCE_CONTRACT_VERSION
    ]


def _source_oos_evidence(
    *,
    selector_output_dir: Path,
    benchmark_path: Path,
    leaderboard_path: Path,
    oos_predictions_path: Path,
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    latest_run_dir = run_dir_from_latest_completed(selector_output_dir)
    manifest = read_run_manifest(latest_run_dir) if latest_run_dir else None
    manifest_hash = file_digest(latest_run_dir / "run_manifest.json") if latest_run_dir else None
    identity = {
        "selected_model_name": selected["model_name"],
        "selected_signal_column": selected["selected_signal_column"],
        "oos_ranking_metric": selected["oos_ranking_metric"],
        "oos_metric_value": selected["oos_metric_value"],
        "source_benchmark_run_id": (manifest or {}).get("run_id"),
        "source_benchmark_manifest_sha256": manifest_hash,
        "source_benchmark_artifact_sha256": file_digest(benchmark_path),
        "source_oos_prediction_artifact_sha256": file_digest(oos_predictions_path),
        "source_leaderboard_artifact_sha256": file_digest(leaderboard_path),
    }
    return {
        "identity": identity,
        "leaderboard_row": selected["leaderboard_row"],
        "source_benchmark_run_id": identity["source_benchmark_run_id"],
        "source_benchmark_manifest_hash": manifest_hash,
        "source_oos_prediction_artifact_hash": identity["source_oos_prediction_artifact_sha256"],
        "source_leaderboard_artifact_hash": identity["source_leaderboard_artifact_sha256"],
    }


def _final_fit_identity(
    *,
    selected: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "selected_model_name": selected["model_name"],
        "selected_model_configuration": selected["model_configuration"],
        "ordered_feature_columns": feature_contract["ordered_feature_columns"],
        "target_name": TARGET_COLUMN,
        "target_horizon": feature_contract["target_horizon"],
        "target_provenance_contract_version": TARGET_PROVENANCE_CONTRACT_VERSION,
        "temporal_policy_identity": feature_contract["temporal_policy_identity"],
        "final_fit_decision_timestamp": feature_contract["final_fit_decision_timestamp"],
        "training_row_count": training_manifest["eligible_training_row_count"],
        "symbol_universe": training_manifest["symbol_universe"],
        "dataset_hash": feature_contract["dataset_hash"],
        "model_input_hash": feature_contract["model_input_hash"],
        "resolved_config_hash": feature_contract["resolved_config_hash"],
        "source_oos_evidence_identity": source_evidence["identity"],
    }


def _final_fit_cutoff(ml: Mapping[str, Any], benchmark_path: Path) -> str:
    configured = str(ml.get("stock_selector_final_fit_decision_timestamp", "")).strip()
    if configured:
        return configured
    benchmark = _read_json(benchmark_path)
    folds = (benchmark.get("walk_forward") or {}).get("folds") or []
    candidates = [
        str(fold.get("oos_prediction_date_max") or fold.get("test_end_date") or "")
        for fold in folds
        if str(fold.get("oos_prediction_date_max") or fold.get("test_end_date") or "").strip()
    ]
    if candidates:
        return max(candidates)
    raise RuntimeError(
        "ml.stock_selector_final_fit_decision_timestamp is required when "
        "the benchmark artifact has no walk-forward cutoff"
    )


def _temporal_policy_identity(cutoff: str) -> dict[str, Any]:
    return {
        "version": TEMPORAL_POLICY_VERSION,
        "workflow": "stock_selector_final_fit",
        "training_eligibility_rule": "label_available_timestamp <= final_fit_decision_timestamp",
        "row_cutoff_rule": "rebalance_date <= final_fit_decision_timestamp",
        "final_fit_decision_timestamp": cutoff,
    }


def _model_input_hash(
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
) -> str:
    payload = [
        {
            "rebalance_date": row["rebalance_date"],
            "symbol": row["symbol"],
            "features": {column: row[column] for column in feature_columns},
            "target": row[TARGET_COLUMN],
            "label_available_timestamp": row["label_available_timestamp"],
        }
        for row in rows
    ]
    return MLCoreArtifactWriter.hash_payload(payload)


def _symbol_universe(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(row["symbol"]) for row in rows})


def _single_value(rows: Sequence[Mapping[str, Any]], column: str) -> str | None:
    values = sorted({str(row.get(column, "")) for row in rows if str(row.get(column, "")).strip()})
    return values[0] if len(values) == 1 else None


def _equivalent(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(abs(float(a) - float(b)) <= 1e-12 for a, b in zip(left, right))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
