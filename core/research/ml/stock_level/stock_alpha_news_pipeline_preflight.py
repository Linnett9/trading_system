from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_alpha_news_contract import (
    write_stock_alpha_news_features_from_config,
)
from core.research.ml.stock_level.stock_alpha_news_contract_ingest import (
    write_stock_alpha_news_contract_ingest,
)
from core.research.ml.stock_level.stock_alpha_news_coverage_audit import (
    write_stock_alpha_news_coverage_audit,
)
from core.research.ml.stock_level.stock_alpha_news_provider_audit import (
    write_stock_alpha_news_provider_audit,
)
from core.research.ml.stock_level.stock_alpha_news_readiness_preflight import (
    write_stock_alpha_news_readiness_preflight,
)


STAGE_ORDER = (
    "provider_audit",
    "contract_ingest",
    "coverage_audit",
    "feature_generation",
    "readiness_preflight",
)


@dataclass(frozen=True)
class StockAlphaNewsPipelinePreflightPaths:
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class _StageSpec:
    name: str
    writer: Callable[[Mapping[str, Any]], Any]
    safe_key: str
    json_attr: str
    markdown_attr: str


def write_stock_alpha_news_pipeline_preflight(
    config: Mapping[str, Any],
) -> StockAlphaNewsPipelinePreflightPaths:
    payload = build_stock_alpha_news_pipeline_preflight(config)
    output_dir = _output_dir(config)
    paths = StockAlphaNewsPipelinePreflightPaths(
        json_path=output_dir / "stock_alpha_news_pipeline_preflight.json",
        markdown_path=output_dir / "stock_alpha_news_pipeline_preflight.md",
    )
    writer = ResearchArtifactWriter()
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))
    return paths


def build_stock_alpha_news_pipeline_preflight(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    enable_flag = bool(ml.get("stock_alpha_news_enable_transformer", False))
    stages: dict[str, dict[str, Any]] = {
        name: _stage_payload(name) for name in STAGE_ORDER
    }
    blocking_issues: list[str] = []
    warning_issues: list[str] = []
    stopped_stage: str | None = None

    for spec in _stage_specs():
        stage = _run_stage(spec, config)
        stages[spec.name] = stage
        blocking_issues.extend(
            f"{spec.name}: {issue}" for issue in stage["blocking_issues"]
        )
        warning_issues.extend(
            f"{spec.name}: {issue}" for issue in stage["warning_issues"]
        )

        if not stage["safe"]:
            disabled_only = (
                spec.name == "readiness_preflight"
                and not enable_flag
                and _only_enable_flag_blockers(stage["blocking_issues"])
            )
            if not disabled_only:
                stopped_stage = spec.name
            break

    pipeline_completed = stages["readiness_preflight"]["completed"]
    pipeline_safe = bool(
        pipeline_completed
        and enable_flag
        and stages["readiness_preflight"]["safe"]
        and not blocking_issues
    )
    return {
        "pipeline_safe_for_news_transformer_training": pipeline_safe,
        "pipeline_completed": pipeline_completed,
        "stopped_stage": stopped_stage,
        "blocking_issues": _dedupe(blocking_issues),
        "warning_issues": _dedupe(warning_issues),
        "stage_order": list(STAGE_ORDER),
        "stages": stages,
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
        "model_training_invoked": False,
        "diagnostics_invoked": False,
        "news_transformer_enable_flag": enable_flag,
        "promotion_thresholds_changed": False,
    }


def _stage_specs() -> tuple[_StageSpec, ...]:
    return (
        _StageSpec(
            "provider_audit",
            write_stock_alpha_news_provider_audit,
            "safe_for_pit_research",
            "json_path",
            "markdown_path",
        ),
        _StageSpec(
            "contract_ingest",
            write_stock_alpha_news_contract_ingest,
            "safe_to_generate_features",
            "audit_json_path",
            "audit_markdown_path",
        ),
        _StageSpec(
            "coverage_audit",
            write_stock_alpha_news_coverage_audit,
            "safe_for_feature_generation",
            "json_path",
            "markdown_path",
        ),
        _StageSpec(
            "feature_generation",
            write_stock_alpha_news_features_from_config,
            "safe_for_readiness_preflight",
            "audit_json_path",
            "audit_markdown_path",
        ),
        _StageSpec(
            "readiness_preflight",
            write_stock_alpha_news_readiness_preflight,
            "safe_to_train_news_transformer",
            "json_path",
            "markdown_path",
        ),
    )


def _run_stage(
    spec: _StageSpec,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    stage = _stage_payload(spec.name, attempted=True)
    try:
        paths = spec.writer(config)
    except (FileNotFoundError, ValueError) as exc:
        return {
            **stage,
            "completed": False,
            "safe": False,
            "blocking_issues": [str(exc)],
        }

    json_path = getattr(paths, spec.json_attr)
    markdown_path = getattr(paths, spec.markdown_attr)
    payload = _read_json(json_path)
    safe = _stage_safe(spec, payload)
    return {
        **stage,
        "completed": True,
        "safe": safe,
        "blocking_issues": list(payload.get("blocking_issues", [])),
        "warning_issues": list(payload.get("warning_issues", [])),
        "output_json_path": str(json_path),
        "output_markdown_path": str(markdown_path),
    }


def _stage_safe(spec: _StageSpec, payload: Mapping[str, Any]) -> bool:
    if spec.name == "feature_generation":
        return not payload.get("blocking_issues")
    return bool(payload.get(spec.safe_key, False))


def _stage_payload(name: str, *, attempted: bool = False) -> dict[str, Any]:
    return {
        "stage_name": name,
        "attempted": attempted,
        "completed": False,
        "safe": False,
        "blocking_issues": [],
        "warning_issues": [],
        "output_json_path": None,
        "output_markdown_path": None,
    }


def _output_dir(config: Mapping[str, Any]) -> Path:
    value = dict(config.get("ml", {}) or {}).get(
        "stock_alpha_news_pipeline_preflight_output_dir"
    )
    if not value:
        raise ValueError("missing ml.stock_alpha_news_pipeline_preflight_output_dir")
    return Path(str(value))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _only_enable_flag_blockers(issues: list[str]) -> bool:
    normalized = set(issues)
    return bool(normalized) and normalized <= {
        "stock_alpha_news_enable_transformer_false"
    }


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _markdown(payload: Mapping[str, Any]) -> str:
    stages = dict(payload.get("stages", {}) or {})
    blocking = payload.get("blocking_issues", []) or ["none"]
    warnings = payload.get("warning_issues", []) or ["none"]
    lines = [
        "# Stock-Alpha News Pipeline Preflight",
        "",
        f"- Pipeline safe for news transformer training: {payload['pipeline_safe_for_news_transformer_training']}",
        f"- Pipeline completed: {payload['pipeline_completed']}",
        f"- Stopped stage: {payload.get('stopped_stage') or 'none'}",
        f"- News transformer enable flag: {payload['news_transformer_enable_flag']}",
        f"- Model training invoked: {payload['model_training_invoked']}",
        f"- Diagnostics invoked: {payload['diagnostics_invoked']}",
        "",
        "## Stages",
    ]
    for stage_name in payload.get("stage_order", STAGE_ORDER):
        stage = dict(stages.get(stage_name, {}) or {})
        lines.extend(
            [
                f"- {stage_name}: attempted={stage.get('attempted', False)}, completed={stage.get('completed', False)}, safe={stage.get('safe', False)}",
                f"  - JSON: {stage.get('output_json_path') or 'none'}",
                f"  - Markdown: {stage.get('output_markdown_path') or 'none'}",
            ]
        )
    lines.extend(
        [
            "",
            "## Blocking Issues",
            *[f"- {issue}" for issue in blocking],
            "",
            "## Warnings",
            *[f"- {issue}" for issue in warnings],
            "",
            "Research-only pipeline preflight. No models were trained, no diagnostics were run, and no trading or execution paths were touched.",
        ]
    )
    return "\n".join(lines)
