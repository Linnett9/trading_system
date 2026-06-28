from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator


class ResearchStageLogger:
    """Structured stage timing without coupling pipelines to a log backend."""

    def __init__(self, pipeline: str, logger: logging.Logger | None = None):
        self.pipeline = pipeline
        self.logger = logger or logging.getLogger("research")

    @contextmanager
    def stage(self, stage: str) -> Iterator[None]:
        started = time.perf_counter()
        self.logger.info(
            "research_stage_started",
            extra={"pipeline": self.pipeline, "stage": stage},
        )
        try:
            yield
        finally:
            self.logger.info(
                "research_stage_completed",
                extra={
                    "pipeline": self.pipeline,
                    "stage": stage,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
