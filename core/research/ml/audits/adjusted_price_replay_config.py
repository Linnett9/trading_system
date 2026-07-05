from __future__ import annotations

from typing import Any


def _adjusted_replay_config(config: dict[str, Any]) -> dict[str, Any]:
    replay = dict(config.get("adjusted_replay", {}) or {})
    policy = str(
        replay.get(
            "missing_symbol_policy",
            config.get("missing_symbol_policy", "fail_closed"),
        )
    )
    if policy not in {"fail_closed", "fallback_raw", "skip_period"}:
        raise ValueError(f"Unsupported adjusted replay missing_symbol_policy: {policy}")
    require_full = bool(
        replay.get(
            "require_full_adjusted_coverage",
            config.get("require_full_adjusted_coverage", True),
        )
    )
    if policy == "fail_closed":
        require_full = True
    return {
        "missing_symbol_policy": policy,
        "require_full_adjusted_coverage": require_full,
        "allow_raw_fallback": policy == "fallback_raw",
        "required_adjusted_coverage_ratio": (
            1.0
            if require_full
            else float(
                replay.get(
                    "required_adjusted_coverage_ratio",
                    config.get("required_adjusted_coverage_ratio", 1.0),
                )
            )
        ),
        "min_independent_periods": int(config.get("min_independent_periods", 36)),
    }
