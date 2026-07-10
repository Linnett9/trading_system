from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping


def build_news_risk_validation_and_evidence_reports(
    *,
    oos_rows: list[dict[str, Any]],
    replay: Mapping[str, Any],
    price_score_column: str,
    ml: Mapping[str, Any],
    cost_scenarios: Mapping[str, Any],
    news_path: Path,
    news_rows: list[dict[str, str]],
    labeled: list[dict[str, Any]],
    contrarian_validation_stage_reports: Callable[..., dict[str, Any]],
    optional_csv_stage: Callable[..., dict[str, Any]],
    news_evidence_lineage_artifacts: Callable[..., tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]],
) -> dict[str, Any]:
    validation = contrarian_validation_stage_reports(
        rows=oos_rows,
        replay=replay,
        price_score_column=price_score_column,
        config=ml,
        cost_scenarios=cost_scenarios,
        data_audit=replay["replay_data_audit"],
    )
    evidence_stages = {
        "raw_news": optional_csv_stage(ml, "stock_alpha_news_raw_path"),
        "provider_normalized_news": optional_csv_stage(
            ml,
            "stock_alpha_news_collect_output_path",
            "stock_alpha_news_provider_normalized_path",
        ),
        "news_contract": optional_csv_stage(ml, "stock_alpha_news_contract_path"),
        "news_features": {"available": True, "path": str(news_path), "rows": news_rows},
        "joined_candidate_rows": {"available": True, "path": "IN_MEMORY_POINT_IN_TIME_JOIN", "rows": labeled},
        "catastrophic_veto_input_rows": {"available": True, "path": "IN_MEMORY_OOS_ROWS", "rows": oos_rows},
    }
    (
        validation["news_evidence_lineage_report"],
        validation["news_evidence_lineage_by_stage"],
        validation["news_evidence_missing_field_examples"],
        validation["news_evidence_readiness_report"],
    ) = news_evidence_lineage_artifacts(evidence_stages)
    validation["news_evidence_readiness_report"].update(
        {
            "event_taxonomy_status": validation["news_event_taxonomy_report"]["status"],
            "event_taxonomy_research_ready": validation["news_event_taxonomy_report"]["event_taxonomy_research_ready"],
            "duplicate_grouping_status": validation["news_duplicate_grouping_report"]["status"],
            "duplicate_grouping_heuristic_ready": validation["news_duplicate_grouping_report"]["duplicate_grouping_heuristic_ready"],
            "point_in_time_text_safety_status": validation["news_point_in_time_text_safety_report"]["status"],
            "point_in_time_text_safety_ready": validation["news_point_in_time_text_safety_report"]["point_in_time_text_safety_ready"],
            "keyword_baseline_status": validation["news_text_keyword_baseline_report"]["status"],
            "keyword_baseline_ready": validation["news_text_keyword_baseline_report"]["keyword_baseline_ready"],
        }
    )
    return validation
