from __future__ import annotations

from typing import Any


def _alignment_config(config: dict[str, Any]) -> dict[str, Any]:
    ml_config = config.get("ml", {})
    return _normalize_audit_config(
        ml_config.get("adjusted_replay_alignment_audit", {}) or {}
    )


def _normalize_audit_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "return_delta_abs_threshold": float(
            config.get("return_delta_abs_threshold", 0.05)
        ),
        "adjustment_ratio_jump_abs_threshold": float(
            config.get("adjustment_ratio_jump_abs_threshold", 0.02)
        ),
        "candidate_net_return_delta_abs_threshold": float(
            config.get("candidate_net_return_delta_abs_threshold", 0.05)
        ),
        "split_ratio_tolerance": float(config.get("split_ratio_tolerance", 0.08)),
        "numeric_tolerance": float(config.get("numeric_tolerance", 1e-10)),
        "top_delta_rows": int(config.get("top_delta_rows", 25)),
    }
