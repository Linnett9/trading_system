from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from application.services.paper_trading_reporting import report_dir
from core.risk.paper_risk import risk_status


def run_id(candidate_id: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    clean_candidate = (candidate_id or "unknown").replace("/", "_")
    return f"{timestamp}_{clean_candidate}"


def stable_hash(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def file_hash(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproducibility_metadata(
    config: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    dual_config = config.get("research", {}).get("dual_momentum", {})
    candidate_config_path = dual_config.get("champion_config_path")
    return {
        "candidate_id": candidate_id,
        "config_hash": stable_hash(config),
        "candidate_config_path": candidate_config_path,
        "candidate_config_hash": file_hash(candidate_config_path),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "execution_adapter": config.get("paper_trading", {}).get(
            "execution_adapter",
            "local_ledger",
        ),
        "broker_adapter": config.get("broker", {}).get(
            "adapter",
            "fake",
        ),
        "generated_at": datetime.utcnow().isoformat(),
    }


def decision_hashes(decision: Any) -> tuple[str, str]:
    target_payload = json.dumps(
        {
            "exposure_target": round(float(decision.exposure_target), 6),
            "target_weights": {
                symbol: round(float(weight), 6)
                for symbol, weight in sorted(
                    decision.target_weights.items(),
                )
            },
        },
        sort_keys=True,
    )
    order_payload = json.dumps(
        [
            {
                "symbol": order.symbol,
                "side": order.side,
                "quantity_delta": round(order.quantity_delta, 8),
                "dollar_delta": round(order.dollar_delta, 8),
                "target_weight": round(order.target_weight, 8),
                "order_type": order.order_type,
                "limit_price": (
                    round(order.limit_price, 8)
                    if order.limit_price is not None
                    else None
                ),
            }
            for order in decision.orders
        ],
        sort_keys=True,
    )
    return (
        hashlib.sha256(target_payload.encode("utf-8")).hexdigest(),
        hashlib.sha256(order_payload.encode("utf-8")).hexdigest(),
    )


def approval_file(config: dict[str, Any]) -> Path:
    return report_dir(config) / "dry_run_approval.json"


def save_dry_run_approval(
    config: dict[str, Any],
    decision: Any,
    target_hash: str,
    order_hash: str,
    risk_checks: list[Any],
    reproducibility: dict[str, Any],
) -> Path:
    path = approval_file(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "decision_timestamp": decision.timestamp.isoformat(),
        "target_portfolio_hash": target_hash,
        "order_list_hash": order_hash,
        "risk_status": risk_status(risk_checks),
        "orders": len(decision.orders),
        "decision_path": str(decision.report_path),
        "reproducibility": reproducibility,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def approval_error(
    config: dict[str, Any],
    target_hash: str,
    order_hash: str,
) -> str | None:
    path = approval_file(config)
    if not path.exists():
        return "approval_required"

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "approval_invalid"

    if payload.get("target_portfolio_hash") != target_hash:
        return "approval_target_hash_mismatch"
    if payload.get("order_list_hash") != order_hash:
        return "approval_order_hash_mismatch"
    if payload.get("risk_status") in {"ERROR", "CRITICAL"}:
        return "approval_has_blocking_risk"

    return None
