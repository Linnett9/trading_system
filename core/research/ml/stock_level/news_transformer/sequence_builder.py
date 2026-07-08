from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping


def build_disabled_news_sequence_examples(rows: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    del rows
    return []
