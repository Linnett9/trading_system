from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


COMPLETE_V1 = "COMPLETE_V1"
PARTIAL = "PARTIAL"
FAILING = "FAILING"
NOT_TESTED = "NOT_TESTED"


@dataclass(frozen=True)
class ModelContractSpec:
    registry_key: str
    display_name: str
    class_name: str
    module_path: str
    expected_predicted_outputs: tuple[str, ...] = ()
    partial_todos: tuple[str, ...] = ()
    persistence_expected: bool = True
    meta_model: bool = False


@dataclass(frozen=True)
class ModelContractAuditRow:
    registry_key: str
    model_name: str
    class_name: str
    module_path: str
    config_paths: tuple[str, ...]
    train_support: bool
    predict_support: bool
    save_load_support: bool
    artifact_support: bool
    expected_predicted_outputs: tuple[str, ...]
    tests_present: bool
    status: str
    todos: tuple[str, ...]


MODEL_CONTRACT_SPECS = (
    ModelContractSpec(
        "dlinear",
        "DLinear",
        "DLinearSequenceMLModel",
        "core/research/ml/models/dlinear_model.py",
    ),
    ModelContractSpec(
        "patchtst",
        "PatchTST",
        "PatchTSTSequenceMLModel",
        "core/research/ml/models/patchtst_model.py",
    ),
    ModelContractSpec(
        "transformer",
        "Transformer",
        "TransformerSequenceMLModel",
        "core/research/ml/models/transformer_model.py",
    ),
    ModelContractSpec(
        "itransformer",
        "ITransformer",
        "ITransformerSequenceMLModel",
        "core/research/ml/models/itransformer_model.py",
        partial_todos=(
            "Full cross-asset rank-score design is not implemented; do not emit "
            "predicted_rank_score until supported by model internals.",
        ),
    ),
    ModelContractSpec(
        "momentum_transformer",
        "Momentum Transformer",
        "MomentumTransformerSequenceMLModel",
        "core/research/ml/models/momentum_transformer_model.py",
        expected_predicted_outputs=(
            "predicted_trend_score",
            "predicted_regime_score",
            "predicted_size_multiplier",
        ),
    ),
    ModelContractSpec(
        "multitask_transformer",
        "Multi-task Transformer",
        "MultiTaskTransformerSequenceMLModel",
        "core/research/ml/models/multitask_transformer_model.py",
        expected_predicted_outputs=(
            "predicted_forward_return_5d",
            "predicted_forward_return_10d",
            "predicted_future_volatility",
            "predicted_future_drawdown",
        ),
    ),
    ModelContractSpec(
        "market_context_encoder",
        "Market Context Encoder",
        "MarketContextEncoderMLModel",
        "core/research/ml/models/market_context_encoder_model.py",
        expected_predicted_outputs=("predicted_context_risk_multiplier",),
        partial_todos=(
            "Context embeddings and detailed regime diagnostics are intentionally "
            "not part of the v1 artifact contract.",
        ),
    ),
    ModelContractSpec(
        "news_analysis_transformer",
        "News Analysis Transformer",
        "NewsAnalysisTransformerMLModel",
        "core/research/ml/models/news_analysis_transformer_model.py",
        partial_todos=(
            "Full timestamped news ingestion and FinBERT-style sentiment pipeline "
            "remain out of scope for v1.",
        ),
    ),
    ModelContractSpec(
        "temporal_fusion_transformer",
        "Temporal Fusion Transformer",
        "TemporalFusionTransformerMLModel",
        "core/research/ml/models/temporal_fusion_transformer_model.py",
        expected_predicted_outputs=(
            "predicted_forward_return_5d",
            "predicted_forward_return_10d",
            "predicted_future_volatility",
            "predicted_future_drawdown",
        ),
        partial_todos=(
            "Full TFT interpretability artifacts are intentionally out of scope "
            "for v1.",
        ),
    ),
    ModelContractSpec(
        "logistic_regression",
        "Logistic Regression",
        "LogisticRegressionMLModel",
        "core/research/ml/models/registry.py",
    ),
    ModelContractSpec(
        "random_forest",
        "Random Forest",
        "TreeClassifierMLModel",
        "core/research/ml/models/registry.py",
    ),
    ModelContractSpec(
        "gradient_boosting",
        "Gradient Boosting",
        "TreeClassifierMLModel",
        "core/research/ml/models/registry.py",
    ),
    ModelContractSpec(
        "meta_ensemble",
        "Meta Ensemble",
        "run_meta_ensemble",
        "core/research/ml/meta_ensemble.py",
        persistence_expected=False,
        meta_model=True,
        partial_todos=(
            "Meta Ensemble is an orchestration/reporting pipeline, not an IMLModel "
            "with save/load persistence.",
        ),
    ),
)


def build_model_contract_audit(
    *,
    repo_root: Path = Path("."),
) -> list[ModelContractAuditRow]:
    return [
        _audit_spec(spec, repo_root=repo_root)
        for spec in MODEL_CONTRACT_SPECS
    ]


def write_model_contract_audit(
    output_dir: Path | str = Path("reports/ml"),
    *,
    repo_root: Path = Path("."),
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows = build_model_contract_audit(repo_root=repo_root)
    json_path = output_path / "model_contract_audit.json"
    markdown_path = output_path / "model_contract_audit.md"
    json_path.write_text(
        json.dumps([asdict(row) for row in rows], indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown_report(rows), encoding="utf-8")
    return markdown_path, json_path


def _audit_spec(
    spec: ModelContractSpec,
    *,
    repo_root: Path,
) -> ModelContractAuditRow:
    module_path = repo_root / spec.module_path
    module_text = _read_text(module_path)
    config_paths = tuple(_config_paths_for_model(spec.registry_key, repo_root))
    tests_present = _tests_present_for_model(spec, repo_root)
    train_support = spec.meta_model or "def fit(" in module_text
    predict_support = spec.meta_model or (
        "def predict(" in module_text and "def predict_proba(" in module_text
    )
    save_load_support = (
        not spec.persistence_expected
        or ("def save(" in module_text and "def load(" in module_text)
    )
    artifact_support = spec.meta_model or True
    todos = list(spec.partial_todos)

    status = COMPLETE_V1
    if not config_paths:
        status = PARTIAL
        todos.append("Add or document a YAML config for this registry key.")
    if not tests_present:
        status = NOT_TESTED if status == COMPLETE_V1 else status
        todos.append("Add focused contract tests for construction/train/predict/save-load.")
    if not (train_support and predict_support and save_load_support and artifact_support):
        status = FAILING
        todos.append("Missing one or more required v1 model contract capabilities.")

    return ModelContractAuditRow(
        registry_key=spec.registry_key,
        model_name=spec.display_name,
        class_name=spec.class_name,
        module_path=spec.module_path,
        config_paths=config_paths,
        train_support=train_support,
        predict_support=predict_support,
        save_load_support=save_load_support,
        artifact_support=artifact_support,
        expected_predicted_outputs=spec.expected_predicted_outputs,
        tests_present=tests_present,
        status=status,
        todos=tuple(todos),
    )


def _config_paths_for_model(model_type: str, repo_root: Path) -> list[str]:
    config_dir = repo_root / "configs" / "research"
    if not config_dir.exists():
        return []
    matches = []
    pattern = re.compile(rf"model_type:\s*[\"']?{re.escape(model_type)}[\"']?\b")
    for path in sorted(config_dir.glob("*.yaml")):
        if pattern.search(_read_text(path)):
            matches.append(str(path.relative_to(repo_root)))
    return matches


def _tests_present_for_model(spec: ModelContractSpec, repo_root: Path) -> bool:
    tests_dir = repo_root / "tests"
    if not tests_dir.exists():
        return False
    needles = {spec.registry_key, spec.class_name}
    for path in tests_dir.glob("test_*.py"):
        text = _read_text(path)
        if any(needle in text for needle in needles):
            return True
    return False


def _markdown_report(rows: list[ModelContractAuditRow]) -> str:
    lines = [
        "# ML Model Contract Audit",
        "",
        "| Model | Registry Key | Class | Status | Configs | predicted_* Outputs | TODOs |",
        "|---|---|---|---|---:|---|---|",
    ]
    for row in rows:
        outputs = ", ".join(row.expected_predicted_outputs) or "-"
        todos = "<br>".join(row.todos) or "-"
        lines.append(
            "| "
            f"{row.model_name} | "
            f"`{row.registry_key}` | "
            f"`{row.class_name}` | "
            f"{row.status} | "
            f"{len(row.config_paths)} | "
            f"{outputs} | "
            f"{todos} |"
        )
    lines.append("")
    return "\n".join(lines)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")
