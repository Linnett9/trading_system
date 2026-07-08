from __future__ import annotations

from typing import Any


def build_news_transformer_training_plan() -> dict[str, Any]:
    return {
        "schema_name": "stock_alpha_news_transformer_training_plan",
        "schema_version": 1,
        "status": "NOT_READY",
        "plan_only": True,
        "enabled": False,
        "training_enabled": False,
        "inference_enabled": False,
        "used_in_strategy": False,
        "used_in_replay": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "validation_passed": False,
        "model_family": "disabled_news_transformer_scaffold",
        "optional_dependencies": ["torch", "transformers"],
        "optional_dependencies_imported": False,
        "model_downloads_enabled": False,
        "training_data_required": [
            "point-in-time availability timestamps",
            "publication timestamps",
            "duplicate/syndication groups",
            "structured event taxonomy",
            "chronological splits",
            "future-only labels",
        ],
        "blockers": [
            "validation spine incomplete",
            "walk-forward not implemented",
            "placebo not implemented",
            "matched controls not implemented",
            "corporate-action/survivorship/missing-news audits incomplete",
            "text model readiness is NOT_READY",
        ],
        "warnings": ["This artifact is a disabled plan only; it does not train, score, or affect strategy results."],
    }
