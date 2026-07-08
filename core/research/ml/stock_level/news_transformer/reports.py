from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

from .readiness import build_news_transformer_readiness_report
from .training_plan import build_news_transformer_training_plan


def build_news_transformer_reports(rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    return {
        "readiness": build_news_transformer_readiness_report(rows),
        "training_plan": build_news_transformer_training_plan(),
    }
