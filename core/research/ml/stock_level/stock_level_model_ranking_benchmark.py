from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

from core.research.ml.artifacts.artifact_writers import MLCoreArtifactWriter
from core.research.ml.immutable_runs import (
    deterministic_run_id,
    file_digest,
    preserve_immutable_run,
)
from core.research.ml.artifact_lineage import VerificationResult, build_artifact_link, promotion_eligibility
from core.research.ml.registries.io import canonical_hash
from core.research.ml.stock_level_benchmark_types import (
    BASELINE_COLUMNS,
    FEATURE_COLUMNS,
    MODEL_NAMES,
    ModelRunSpec,
    NOTICE,
    PREDICTION_PREFIX,
    RESEARCH_METADATA,
    SEQUENCE_MODEL_NAMES,
    TABULAR_MODEL_NAMES,
    TARGET_COLUMN,
    TARGET_PROVENANCE_COLUMNS,
    TARGET_PROVENANCE_CONTRACT_VERSION,
    StockLevelModelRankingBenchmarkPaths,
)
from core.research.ml.stock_level_benchmark_models import (
    SequenceModelFactory,
    TabularModelFactory,
    _build_tabular_model,
    _model_factories,
    _sequence_feature_columns,
    _sequence_model_factories,
    stock_ranker_model_registry,
)
from core.research.ml.stock_level_benchmark_execution import (
    _MODEL_WORKER_CONTEXT,
    _build_sequences,
    _execute_model_runs,
    _initialize_model_worker,
    _run_initialized_model,
    _run_model_walk_forward,
    _run_model_walk_forward_unlimited,
    _walk_forward_partitions,
)
from core.research.ml.stock_level_benchmark_data import (
    _available_feature_columns,
    _average,
    _base_prediction_row,
    _build_oos_prediction_rows,
    _number,
    _prepare_rows,
    _validate_split_settings,
    _validate_unique_keys,
)
from core.research.ml.stock_level_benchmark_evaluation import (
    _build_leaderboard,
    _compare_to_momentum,
    _evaluate_signal,
)
from core.research.ml.stock_level_benchmark_reporting import (
    _fmt,
    _leaderboard_columns,
    _markdown,
    _output_dir,
    _prediction_columns,
    _read_csv,
    _write_csv,
)
from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.data import CsvRowRepository
from core.research.framework.logging import ResearchStageLogger
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_alpha_run_profile import apply_stock_alpha_run_profile
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_report_metadata
from core.research.ml.stock_level.stock_level_artifact_io import read_stock_level_artifact
from core.research.ml.stock_level.selector_dataset import read_selector_dataset_rows
from core.research.ml.stock_level.stock_alpha_news_contract import validate_news_contract
from core.research.ml.runtime_parallelism import apply_stock_alpha_worker_caps
from core.research.ml.experiment_ledger import append_ledger_event, experiment_spec_hash, new_experiment_run_id
from core.research.ml.provenance import dependency_identity, file_identity, source_provenance
from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.registries.adapters import selector_model_adapter
from core.research.ml.stock_level.stock_alpha_model_sets import FULL_SEQUENCE_MODELS, StockAlphaModelSet, resolve_stock_alpha_model_set
from datetime import datetime, timezone
import time


def _ordered_logit_features() -> tuple[str, ...]:
    from core.research.ml.stock_level.selector_feature_schema import load_feature_schema
    schema = load_feature_schema(Path("config/selector_features/canonical_v2_daily_tree_cross_sectional_v1.json"))
    return tuple(row["name"] for row in schema["features"])


def write_stock_level_model_ranking_benchmark(
    config: dict[str, Any],
) -> StockLevelModelRankingBenchmarkPaths:
    """Train the isolated stock-level benchmark and write research artifacts."""
    settings = StockLevelResearchConfig.from_mapping(config)
    if config.get("ml", {}).get("ordinary_selector_manifest_root"):
        raise ValueError(
            "Ordinary selector publication is owned by the selector component "
            "publication pipeline. Run ml-selector-component-publish with a "
            "validated parent gate and production-plan job; the ranking "
            "benchmark writes research artifacts only."
        )
    thread_caps = apply_stock_alpha_worker_caps(config)
    started_at = datetime.now(timezone.utc).isoformat(); started = time.perf_counter()
    output_dir = settings.output_dir
    selector_dataset_root = Path(str(config.get("ml", {}).get("stock_selector_dataset_root", "")).strip()) if str(config.get("ml", {}).get("stock_selector_dataset_root", "")).strip() else None
    source_path = selector_dataset_root / "rows.parquet" if selector_dataset_root else settings.artifact_path
    if not source_path.exists():
        raise FileNotFoundError(f"Stock-level prediction artifact not found: {source_path}")

    logger = ResearchStageLogger("stock_level_alpha_benchmark")
    with logger.stage("loading"):
        rows = (
            read_selector_dataset_rows(selector_dataset_root)
            if selector_dataset_root
            else read_stock_level_artifact(
                source_path,
                required_columns={"rebalance_date", "symbol"},
                allow_csv_fallback=bool(config.get("ml", {}).get("stock_level_allow_csv_artifact_fallback", False)),
            )
        )
        rows, run_profile = apply_stock_alpha_run_profile(rows, settings)
    feature_columns = _available_feature_columns(
        rows,
        include_engineered=settings.include_engineered_features,
    )
    with logger.stage("training_and_evaluation"):
        model_set = resolve_stock_alpha_model_set(settings.ranker_model_set, include_sequence_models=settings.include_sequence_models)
        news_available = validate_news_contract(config, rows).available
        explicit_model_ids = config.get("ml", {}).get("stock_ranker_model_ids")
        requested_models = tuple(explicit_model_ids if explicit_model_ids is not None else (model for model in model_set.included_models if news_available or model != "news_analysis_transformer"))
        registry_context = _ordinary_selector_registry_context(requested_models, settings.target_column, news_available=news_available)
        selected_canonical = tuple(row["canonical_model_id"] for row in registry_context["models"])
        if "ordered_logit_ranker" in selected_canonical:
            feature_columns = _ordered_logit_features()
        tabular_factories, sequence_factories = _factories_for_model_set(settings, model_set, sklearn_n_jobs=settings.sklearn_n_jobs, torch_num_threads=thread_caps["torch_num_threads"])
        if "ordered_logit_ranker" in selected_canonical:
            tabular_factories["ordered_logit_ranker"] = TabularModelFactory("ordered_logit_ranker", settings.random_seed, settings.sklearn_n_jobs)
        tabular_factories = {name: factory for name, factory in tabular_factories.items() if name in selected_canonical}
        sequence_factories = {name: factory for name, factory in sequence_factories.items() if name in selected_canonical}
        _ordinary_selector_events(config, registry_context, "STARTED", source_path=source_path, feature_columns=feature_columns, settings=settings, thread_caps=thread_caps)
        try:
            predictions, payload = build_stock_level_model_ranking_benchmark(
            rows,
            target_column=settings.target_column,
            feature_columns=feature_columns,
            source_path=str(source_path),
            config_path=str(config.get("config_path", "config/config.yaml")),
            min_train_dates=settings.min_train_dates,
            test_window_dates=settings.test_window_dates,
            walk_forward_mode=settings.walk_forward_mode,
            operating_mode=settings.selector_operating_mode,
            embargo_dates=settings.embargo_dates,
            random_seed=settings.random_seed,
            sklearn_n_jobs=settings.sklearn_n_jobs,
            model_n_jobs=settings.model_n_jobs,
            include_sequence_models=settings.include_sequence_models,
            model_factories=tabular_factories,
            sequence_model_factories=sequence_factories,
            sequence_length=settings.sequence_length,
            sequence_epochs=settings.sequence_epochs,
            sequence_batch_size=settings.sequence_batch_size,
            sequence_device=settings.sequence_device,
            news_contract_available=news_available,
            )
        except BaseException as exc:
            _ordinary_selector_events(config, registry_context, "FAILED", source_path=source_path, feature_columns=feature_columns, settings=settings, thread_caps=thread_caps, error_summary=f"{type(exc).__name__}: {exc}")
            raise
        payload.update(run_profile)
        payload.update(model_set.metadata())
        payload["stock_ranker_model_set"] = settings.ranker_model_set
        payload["requested_models"] = list(requested_models)
        payload["canonical_models"] = list(selected_canonical)
        payload["registry_provenance"] = registry_context
        folds = payload.get("walk_forward", {}).get("folds", [])
        payload["ordinary_selector_provenance"] = {
            "identity_version": registry_context["identity_version"],
            "dataset_identity": file_identity(source_path),
            "feature_schema_paths_and_hashes": [{"canonical_model_id": row["canonical_model_id"], "feature_schema": row["feature_schema"], "model_entry_hash": row["model_entry_hash"]} for row in registry_context["models"]],
            "target_identity": registry_context["target_identity"],
            "hyperparameter_identity": {"sequence_length": settings.sequence_length, "sequence_epochs": settings.sequence_epochs, "sequence_batch_size": settings.sequence_batch_size},
            "random_seed": settings.random_seed,
            "worker_configuration": {"sklearn_n_jobs": settings.sklearn_n_jobs, "model_n_jobs": settings.model_n_jobs, **thread_caps},
            "training_cutoff": max((fold.get("decision_timestamp", "") for fold in folds), default=None),
            "maximum_legal_label_availability_timestamp": max((fold.get("training_label_available_max", "") for fold in folds), default=None),
            "dependency_provenance": registry_context["dependency_provenance"],
            "source_provenance": registry_context["source_provenance"],
            "experiment_runs": registry_context["experiment_runs"],
        }
        payload["dependency_provenance"] = registry_context["dependency_provenance"]
        payload["source_provenance"] = registry_context["source_provenance"]
        payload.update(stock_alpha_report_metadata(config, output_dir, source_artifact_path=source_path))
        payload.update({"started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started, "thread_caps": thread_caps})

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = StockLevelModelRankingBenchmarkPaths(
        csv_path=output_dir / "stock_level_model_ranking_benchmark.csv",
        json_path=output_dir / "stock_level_model_ranking_benchmark.json",
        markdown_path=output_dir / "stock_level_model_ranking_benchmark.md",
        predictions_path=output_dir / "stock_level_model_oos_predictions.csv",
        temporal_audit_path=output_dir / "stock_level_temporal_audit.json",
    )
    decision_dates = sorted({str(row.get("rebalance_date")) for row in predictions if row.get("rebalance_date")})
    ordinary_reasons = (
        "PER_DECISION_TRAINING_CUTOFF_UNVERIFIED", "LABEL_AVAILABILITY_MISSING",
        "ROW_POPULATION_UNVERIFIED", "PREDICTION_QUALITY_MISSING",
    )
    ordinary_link = build_artifact_link(
        artifact_kind="ORDINARY_SELECTOR_PREDICTION",
        artifact_id=f"ordinary-selector:{canonical_hash({'models': list(selected_canonical), 'source': str(source_path)})}",
        artifact_manifest_path=paths.json_path,
        artifact_path=paths.predictions_path,
        artifact_checksum=canonical_hash([str(row.get("row_id") or f"{row.get('rebalance_date')}:{row.get('symbol')}") for row in predictions]),
        source_commit=registry_context["source_provenance"].get("git_commit"),
        registry_identity_version="ordinary_selector_identity_v2_registry",
        requested_model_or_policy_id=",".join(requested_models), canonical_model_or_policy_id=",".join(selected_canonical),
        model_or_policy_entry_hash=canonical_hash(sorted(row["model_entry_hash"] for row in registry_context["models"])),
        dataset_id=registry_context.get("dataset_identity", {}).get("dataset_id") or str(source_path),
        dataset_checksum=file_identity(source_path).get("sha256"),
        feature_schema_hash=canonical_hash([row["feature_schema"] for row in registry_context["models"]]),
        target_contract_hash=registry_context["target_identity"].get("target_entry_hash"),
        decision_start=decision_dates[0] if decision_dates else None, decision_end=decision_dates[-1] if decision_dates else None,
        strict_oos_claim=payload.get("walk_forward", {}).get("out_of_sample_only") is True,
        strict_oos_evidence={"prediction_quality_passed": None, "row_population_verified": False},
        verification_status="DECLARED_STRICT_OOS_UNVERIFIED", verification_reasons=list(ordinary_reasons), completion_status="complete",
    )
    payload["artifact_link"] = ordinary_link
    payload["promotion"] = promotion_eligibility(ordinary_link, VerificationResult("DECLARED_STRICT_OOS_UNVERIFIED", ordinary_reasons))
    with logger.stage("report_generation"):
        writer = ResearchArtifactWriter()
        writer.write_csv(
            paths.csv_path,
            payload["leaderboard"],
            fieldnames=_leaderboard_columns(),
        )
        writer.write_csv(
            paths.predictions_path,
            predictions,
            fieldnames=_prediction_columns(payload["completed_models"]),
        )
        writer.write_json(paths.json_path, payload)
        writer.write_json(
            paths.temporal_audit_path,
            payload.get("temporal_audit", {"version": 1, "folds": [], "summary": {}}),
        )
        writer.write_markdown(paths.markdown_path, _markdown(payload))
    immutable_record = _preserve_stock_selector_benchmark_run(
        output_dir,
        paths,
        payload,
        config=config,
        source_path=source_path,
    )
    _ordinary_selector_events(config, registry_context, "COMPLETED", source_path=source_path, feature_columns=feature_columns, settings=settings, thread_caps=thread_caps, artifact_paths=(str(paths.json_path), str(paths.predictions_path), str(immutable_record.manifest_path)), immutable_run_id=immutable_record.run_id)
    return paths


def _preserve_stock_selector_benchmark_run(
    output_dir: Path,
    paths: StockLevelModelRankingBenchmarkPaths,
    payload: dict[str, Any],
    *,
    config: dict[str, Any],
    source_path: Path,
) -> Any:
    identity = {
        "source_path": str(source_path),
        "source_sha256": file_digest(source_path),
        "target_column": payload.get("target_column"),
        "feature_columns": payload.get("feature_columns"),
        "requested_models": payload.get("requested_models"),
        "completed_models": payload.get("completed_models"),
        "input_row_count": payload.get("input_row_count"),
        "eligible_row_count": payload.get("eligible_row_count"),
        "input_date_count": payload.get("input_date_count"),
        "input_symbol_count": payload.get("input_symbol_count"),
        "oos_row_count": payload.get("oos_row_count"),
        "oos_date_count": payload.get("oos_date_count"),
        "oos_symbol_count": payload.get("oos_symbol_count"),
        "walk_forward": payload.get("walk_forward"),
        "temporal_policy": payload.get("temporal_policy"),
        "target_provenance_contract_version": payload.get(
            "target_provenance_contract_version"
        ),
        "stock_ranker_model_set": payload.get("stock_ranker_model_set"),
        "config_hash": MLCoreArtifactWriter.hash_payload(config),
        "identity_version": (payload.get("ordinary_selector_provenance") or {}).get("identity_version"),
        "registry_models": (payload.get("registry_provenance") or {}).get("models"),
        "target_identity": (payload.get("registry_provenance") or {}).get("target_identity"),
        "dependency_identity": (payload.get("dependency_provenance") or {}).get("hash"),
    }
    run_id = deterministic_run_id("stock_selector_benchmark", identity)
    return preserve_immutable_run(
        output_dir=output_dir,
        run_id=run_id,
        kind="stock_selector_benchmark",
        identity=identity,
        artifact_paths=(
            paths.csv_path,
            paths.json_path,
            paths.markdown_path,
            paths.predictions_path,
            paths.temporal_audit_path,
        ),
        extra_manifest={
            "best_model_name": (payload.get("best_ml_model") or {}).get("name"),
            "champion_pointer_updated": False,
        },
    )


def _ordinary_selector_registry_context(requested_models: tuple[str, ...], target_field: str, *, news_available: bool) -> dict[str, Any]:
    bundle = load_registry_bundle(); resolver = RegistryResolver(bundle)
    models = []
    dependencies: set[str] = {"pyarrow"}
    for requested in requested_models:
        adapter = selector_model_adapter(requested, runner="ordinary", allow_blocked=news_available and requested in {"news_analysis_transformer", "news_transformer"})
        dependencies.update(adapter.dependency_requirements)
        models.append({
            "requested_model_id": requested, "canonical_model_id": adapter.canonical_model_id,
            "model_entry_hash": adapter.entry_hash, "model_family": adapter.model_family,
            "feature_schema": adapter.feature_schema, "target_contract": adapter.target_contract,
            "constructor_owner": adapter.constructor_owner, "checkpoint_support": adapter.checkpoint_support,
            "ranking_problem_contract": adapter.ranking_problem_contract, "relevance_contract": adapter.relevance_contract,
        })
    target = resolver.target_for_field(target_field, role="selector")
    ranking_identities = {entry.canonical_id: entry.entry_hash for entry in bundle.documents["ranking_contracts"].entries}
    return {
        "identity_version": "ordinary_selector_identity_v2_registry", "registry_contract_version": "ml_registry_set_v1",
        "selector_registry_hash": bundle.documents["selector_models"].registry_hash,
        "target_registry_hash": bundle.documents["target_contracts"].registry_hash,
        "registry_set_hash": bundle.registry_set_hash, "models": models,
        "target_identity": {"canonical_target_id": target.canonical_id, "field_name": target_field, "target_entry_hash": target.entry.entry_hash},
        "dependency_provenance": dependency_identity(dependencies), "source_provenance": source_provenance(),
        "experiment_runs": {},
        "ranking_contract_identities": ranking_identities,
    }


def _ordinary_selector_events(config, context, status, *, source_path, feature_columns, settings, thread_caps, artifact_paths=(), error_summary=None, immutable_run_id=None):
    ledger = Path(config.get("ml", {}).get("experiment_ledger_path", "reports/ml/experiments/experiment_ledger.jsonl"))
    for model in context["models"]:
        spec = {
            "identity_version": context["identity_version"], "canonical_model_id": model["canonical_model_id"],
            "requested_model_id": model["requested_model_id"], "model_entry_hash": model["model_entry_hash"],
            "dataset_identity": {"path": str(source_path)}, "feature_columns": list(feature_columns),
            "feature_schema": model["feature_schema"], "target_identity": context["target_identity"],
            "hyperparameters": {"sequence_length": settings.sequence_length, "sequence_epochs": settings.sequence_epochs, "sequence_batch_size": settings.sequence_batch_size},
            "seed": settings.random_seed, "workers": {"sklearn_n_jobs": settings.sklearn_n_jobs, "model_n_jobs": settings.model_n_jobs, **thread_caps},
            "training_window_contract": {"min_train_dates": settings.min_train_dates, "walk_forward_mode": settings.walk_forward_mode, "embargo_dates": settings.embargo_dates},
            "source_commit": context["source_provenance"]["git_commit"], "dependency_hash": context["dependency_provenance"]["hash"],
        }
        spec_hash = experiment_spec_hash(spec); key = model["canonical_model_id"]
        if status == "STARTED" or key not in context["experiment_runs"]:
            context["experiment_runs"][key] = {"experiment_spec_hash": spec_hash, "experiment_run_id": new_experiment_run_id(spec_hash)}
        run = context["experiment_runs"][key]
        append_ledger_event(ledger, experiment_spec_hash_value=run["experiment_spec_hash"], experiment_run_id=run["experiment_run_id"], event_status=status, artifact_kind="MODEL_EXPERIMENT", canonical_model_id=model["canonical_model_id"], requested_model_id=model["requested_model_id"], registry_hashes={"model_entry_hash":model["model_entry_hash"],"selector_registry_hash":context["selector_registry_hash"],"target_entry_hash":context["target_identity"]["target_entry_hash"]}, source_commit=context["source_provenance"]["git_commit"], artifact_paths=artifact_paths, error_summary=error_summary, metadata={"immutable_run_id":immutable_run_id,"identity_version":context["identity_version"]})


def _factories_for_model_set(
    settings: StockLevelResearchConfig,
    model_set: StockAlphaModelSet,
    *,
    sklearn_n_jobs: int,
    torch_num_threads: int,
) -> tuple[dict[str, Callable[[], Any]], dict[str, Callable[[], Any]]]:
    tabular = {name: factory for name, factory in _model_factories(settings.random_seed, sklearn_n_jobs).items() if name in model_set.included_models}
    if not any(name in FULL_SEQUENCE_MODELS for name in model_set.included_models):
        return tabular, {}
    sequence = _sequence_model_factories(
        sequence_length=settings.sequence_length,
        epochs=settings.sequence_epochs,
        batch_size=settings.sequence_batch_size,
        random_seed=settings.random_seed,
        device=settings.sequence_device,
        torch_num_threads=torch_num_threads,
    )
    return tabular, {name: factory for name, factory in sequence.items() if name in model_set.included_models}


def build_stock_level_model_ranking_benchmark(
    rows: list[dict[str, Any]],
    *,
    source_path: str | None = None,
    config_path: str | None = None,
    min_train_dates: int = 52,
    test_window_dates: int = 13,
    walk_forward_mode: str = "block_retrain_research",
    operating_mode: str = "daily_cold_refit_strict",
    embargo_dates: int = 2,
    random_seed: int = 42,
    sklearn_n_jobs: int = 1,
    model_factories: dict[str, Callable[[], Any]] | None = None,
    include_sequence_models: bool = True,
    sequence_length: int = 13,
    sequence_epochs: int = 5,
    sequence_batch_size: int = 256,
    sequence_device: str = "cpu",
    sequence_model_factories: dict[str, Callable[[], Any]] | None = None,
    model_n_jobs: int = 1,
    executor_cls: type | None = None,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
    news_contract_available: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create expanding-window predictions and an OOS ranking leaderboard."""
    _validate_split_settings(min_train_dates, test_window_dates, embargo_dates)
    if walk_forward_mode not in {"daily_retrain_strict", "block_retrain_research"}:
        raise ValueError("Unknown stock selector walk-forward mode")
    if walk_forward_mode == "daily_retrain_strict" and test_window_dates != 1:
        raise ValueError("daily_retrain_strict requires one-date OOS test slices")
    if operating_mode not in {"daily_cold_refit_strict", "daily_checkpoint_update", "daily_score_periodic_refit"}:
        raise ValueError("Unknown stock selector operating mode")
    if operating_mode != "daily_cold_refit_strict":
        raise NotImplementedError(
            f"{operating_mode} is declared but not integrated with selector estimators; "
            "refusing to masquerade as daily_cold_refit_strict"
        )
    if model_n_jobs < 1:
        raise ValueError("stock_ranker_model_n_jobs must be at least one")
    if target_column != TARGET_COLUMN:
        rows = [dict(row, **{TARGET_COLUMN: row.get(target_column, "")}) for row in rows]
    prepared_rows, excluded_row_count = _prepare_rows(rows, feature_columns)
    _validate_unique_keys(prepared_rows)
    dates = sorted({row["rebalance_date"] for row in prepared_rows})
    first_test_index = min_train_dates + embargo_dates
    if len(dates) <= first_test_index:
        raise ValueError(
            "Not enough rebalance dates for the requested walk-forward split: "
            f"found {len(dates)}, need more than {first_test_index}. "
            f"available_rebalance_dates={len(dates)}; "
            f"required_first_test_index={first_test_index}; "
            f"min_train_dates={min_train_dates}; "
            f"test_window_dates={test_window_dates}; "
            f"embargo_dates={embargo_dates}; "
            f"active_config_path={config_path or 'unknown'}; "
            f"source_path={source_path or 'unknown'}. "
            "For dev runs, reduce ml.stock_ranker_min_train_dates, "
            "ml.stock_ranker_test_window_dates, or ml.stock_ranker_embargo_dates, "
            "or use benchmark/full data."
        )
    absent_features = [column for column in feature_columns if not any(row.get(column) not in {None, ""} for row in rows)]
    if absent_features:
        raise ValueError(
            "Stock selector feature contract is missing required columns; "
            f"missing_or_all_null={absent_features}. Build/join the frozen selector derivative."
        )

    effective_sklearn_n_jobs = 1 if model_n_jobs > 1 else sklearn_n_jobs
    effective_torch_num_threads = 1 if model_n_jobs > 1 else None
    tabular_factories = model_factories or _model_factories(
        random_seed, effective_sklearn_n_jobs
    )
    sequence_factories = (
        sequence_model_factories
        if sequence_model_factories is not None
        else _sequence_model_factories(
            sequence_length=sequence_length,
            epochs=sequence_epochs,
            batch_size=sequence_batch_size,
            random_seed=random_seed,
            device=sequence_device,
            torch_num_threads=effective_torch_num_threads,
        )
        if include_sequence_models
        else {}
    )
    if not tabular_factories and not sequence_factories:
        raise ValueError("At least one stock-level model is required")

    news_columns = tuple(
        name
        for name in (rows[0] if rows else {})
        if name.startswith("news_") or "sentiment" in name.lower()
    )
    unavailable_models: list[dict[str, str]] = []
    if "news_analysis_transformer" in sequence_factories and (not news_columns or not news_contract_available):
        sequence_factories = dict(sequence_factories)
        sequence_factories.pop("news_analysis_transformer")
        reason = (
            "news_analysis_transformer unavailable: missing valid point-in-time news contract"
            if news_columns
            else (
                "The stock-level input contains no point-in-time symbol-level "
                "news or sentiment features; synthetic news inputs are forbidden."
            )
        )
        unavailable_models.append(
            {
                "name": "news_analysis_transformer",
                "status": "unavailable",
                "reason": reason,
            }
        )

    specs = [
        ModelRunSpec(name, "tabular", factory, _ordered_logit_features() if name == "ordered_logit_ranker" else feature_columns)
        for name, factory in tabular_factories.items()
    ]
    specs.extend(
        ModelRunSpec(
            name,
            "sequence",
            factory,
            _sequence_feature_columns(name, news_columns, feature_columns),
        )
        for name, factory in sequence_factories.items()
    )
    model_publication_contracts = {
        spec.name: {
            "model_kind": spec.kind,
            "sequence_contract_version": "ordinary_selector_sequence_input_v1" if spec.kind == "sequence" else None,
            "lookback": sequence_length if spec.kind == "sequence" else None,
            "channel_order": list(spec.feature_columns),
            "input_sequence_identity": canonical_hash({"lookback": sequence_length, "channel_order": list(spec.feature_columns)}) if spec.kind == "sequence" else None,
            "checkpoint_capability": "not_integrated",
        }
        for spec in specs
    }
    model_results, model_errors, model_timings = _execute_model_runs(
        specs,
        prepared_rows=prepared_rows,
        dates=dates,
        first_test_index=first_test_index,
        test_window_dates=test_window_dates,
        embargo_dates=embargo_dates,
        sequence_length=sequence_length,
        model_n_jobs=model_n_jobs,
        executor_cls=executor_cls or ProcessPoolExecutor,
    )
    unavailable_models.extend(
        {
            "name": spec.name,
            "status": "error",
            "reason": model_errors[spec.name],
        }
        for spec in specs
        if spec.name in model_errors
    )
    folds, predictions = _build_oos_prediction_rows(
        prepared_rows,
        dates,
        first_test_index=first_test_index,
        test_window_dates=test_window_dates,
        embargo_dates=embargo_dates,
    )
    for model_name, values_by_key in model_results.items():
        column = f"{PREDICTION_PREFIX}{model_name}"
        for row in predictions:
            value = values_by_key[(row["rebalance_date"], row["symbol"])]
            row[column] = float(value)
            if model_name == "ordered_logit_ranker" and hasattr(value, "probabilities"):
                probabilities = tuple(value.probabilities)
                row["ordered_logit_predicted_relevance_class"] = max(range(len(probabilities)), key=lambda index: probabilities[index])
                for index, probability in enumerate(probabilities): row[f"ordered_logit_probability_{index}"] = probability

    ranking_evaluation = None
    if "ordered_logit_ranker" in model_results:
        from core.research.ml.ranking import ranking_metrics, relevance_labels
        relevance = relevance_labels(predictions, bins=5)
        ordered_rows=sorted(predictions,key=lambda row:(float(row[f"{PREDICTION_PREFIX}ordered_logit_ranker"]),str(row["row_id"])))
        ranks={str(row["row_id"]):index+1 for index,row in enumerate(ordered_rows)}
        for row in predictions:
            row["ordered_logit_relevance"] = relevance["labels_by_row_id"][str(row["row_id"])]
            row["ordered_logit_cross_sectional_rank"] = ranks[str(row["row_id"])]
            row["ordered_logit_rank_percentile"] = (ranks[str(row["row_id"])]-1)/max(len(ordered_rows)-1,1)
        ranking_evaluation = {"relevance": relevance, "metrics": ranking_metrics(predictions, score_field=f"{PREDICTION_PREFIX}ordered_logit_ranker", relevance_field="ordered_logit_relevance")}

    _validate_unique_keys(predictions)
    completed_models = tuple(spec.name for spec in specs if spec.name in model_results)
    leaderboard = _build_leaderboard(predictions, tuple(completed_models))
    full_period_baselines = [
        _evaluate_signal(prepared_rows, name, column, kind="baseline")
        for name, column in BASELINE_COLUMNS.items()
    ]
    best_ml = next((row for row in leaderboard if row["kind"] == "ml_model"), None)
    momentum = next(
        row for row in leaderboard if row["name"] == "momentum_120d"
    )
    comparison = _compare_to_momentum(best_ml, momentum)
    target_provenance_summary = _target_provenance_summary(prepared_rows)
    payload = {
        "mode": "stock_level_model_ranking_benchmark_research_only",
        "purpose": (
            "Benchmark simple stock-level return rankers using chronological, "
            "expanding-window out-of-sample predictions."
        ),
        "source_path": source_path,
        "target_column": target_column,
        "feature_columns": list(feature_columns),
        "requested_models": list(MODEL_NAMES),
        "completed_models": list(completed_models),
        "unavailable_models": unavailable_models,
        "baseline_columns": BASELINE_COLUMNS,
        "input_row_count": len(rows),
        "eligible_row_count": len(prepared_rows),
        "excluded_incomplete_row_count": excluded_row_count,
        "input_date_count": len(dates),
        "input_symbol_count": len({row["symbol"] for row in prepared_rows}),
        "oos_row_count": len(predictions),
        "oos_date_count": len({row["rebalance_date"] for row in predictions}),
        "oos_symbol_count": len({row["symbol"] for row in predictions}),
        "prediction_columns": [
            f"{PREDICTION_PREFIX}{name}" for name in completed_models
        ],
        "target_provenance_contract_version": TARGET_PROVENANCE_CONTRACT_VERSION,
        "target_provenance_columns": list(TARGET_PROVENANCE_COLUMNS),
        "target_provenance_summary": target_provenance_summary,
        "parallelism": {
            "requested_workers": model_n_jobs,
            "effective_workers": min(model_n_jobs, len(specs)),
            "nested_sklearn_n_jobs": effective_sklearn_n_jobs,
            "nested_torch_num_threads": effective_torch_num_threads,
            "strategy": "independent_models",
            "stock_ranker_model_n_jobs": model_n_jobs,
            "effective_model_workers": min(model_n_jobs, len(specs)),
            "requested_sklearn_n_jobs": sklearn_n_jobs,
            "effective_per_model_sklearn_n_jobs": effective_sklearn_n_jobs,
            "effective_per_model_torch_num_threads": effective_torch_num_threads,
            "effective_per_model_native_thread_limit": (
                1 if model_n_jobs > 1 else None
            ),
            "folds_parallelized": False,
            "dates_parallelized": False,
        },
        "model_timings": model_timings,
        "model_publication_contracts": model_publication_contracts,
        "ranking_evaluation": ranking_evaluation,
        "walk_forward": {
            "operating_mode": operating_mode,
            "mode": walk_forward_mode,
            "retraining_frequency": "daily" if walk_forward_mode == "daily_retrain_strict" else "per_block",
            "portfolio_target_refresh_frequency": "daily",
            "prediction_horizon_trading_sessions": 10,
            "method": "chronological_expanding_window",
            "min_train_dates": min_train_dates,
            "test_window_dates": test_window_dates,
            "embargo_rebalance_dates": embargo_dates,
            "sequence_length": sequence_length,
            "out_of_sample_only": True,
            "all_chronological_guards_passed": all(
                fold["chronological_guard_passed"] for fold in folds
            ),
            "folds": folds,
        },
        "temporal_policy": {
            "version": 2,
            "workflow": "stock_selector_oos_benchmark",
            "target_provenance_contract_version": TARGET_PROVENANCE_CONTRACT_VERSION,
            "training_window_type": "expanding",
            "decision_timestamp_column": "decision_timestamp",
            "feature_timestamp_column": "feature_timestamp",
            "label_end_timestamp_column": "label_end_timestamp",
            "label_available_timestamp_column": "label_available_timestamp",
            "training_eligibility_rule": "label_available_timestamp <= decision_timestamp",
            "validation_policy": "none_for_stock_selector_models",
            "purge_policy": "exclude_candidate_train_rows_with_unmatured_labels",
            "embargo_rebalance_dates": embargo_dates,
            "minimum_training_dates": min_train_dates,
            "test_window_dates": test_window_dates,
        },
        "temporal_audit": {
            "version": 2,
            "workflow": "stock_selector_oos_benchmark",
            "target_provenance_contract_version": TARGET_PROVENANCE_CONTRACT_VERSION,
            "target_provenance_summary": target_provenance_summary,
            "leakage_checks_passed": all(
                fold["chronological_guard_passed"]
                and fold["label_availability_guard_passed"]
                for fold in folds
            ),
            "folds": folds,
            "summary": {
                "fold_count": len(folds),
                "total_purged_rows": sum(fold["purged_row_count"] for fold in folds),
                "total_oos_rows": len(predictions),
                "duplicate_oos_keys": len(predictions)
                - len({(row["rebalance_date"], row["symbol"]) for row in predictions}),
            },
        },
        "ranking_rule": (
            "Descending mean Spearman IC, then descending top-minus-bottom spread."
        ),
        "ml_beats_momentum_rule": (
            "Best ML model must exceed the OOS-aligned momentum_120d baseline on "
            "both mean Spearman IC and top-minus-bottom spread."
        ),
        "leaderboard": leaderboard,
        "full_period_baselines": full_period_baselines,
        "best_ml_model": best_ml,
        "best_ml_vs_momentum_120d": comparison,
        "ml_beats_momentum_120d": comparison["beats_momentum_120d"],
        **RESEARCH_METADATA,
    }
    return predictions, payload


def _target_provenance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = sum(
        1
        for row in rows
        if all(str(row.get(column, "")).strip() for column in TARGET_PROVENANCE_COLUMNS)
    )
    versions = sorted(
        {
            str(row.get("target_provenance_contract_version"))
            for row in rows
            if str(row.get("target_provenance_contract_version") or "").strip()
        }
    )
    return {
        "total_rows": len(rows),
        "complete_rows": complete,
        "missing_rows": len(rows) - complete,
        "contract_versions": versions,
        "required_columns": list(TARGET_PROVENANCE_COLUMNS),
    }
