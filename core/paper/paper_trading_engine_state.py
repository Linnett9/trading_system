from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class PaperTradingEngineStateMixin:
    @property
    def state_path(self):
        return self.report_dir / "paper_state.json"

    def status(self, prices_by_symbol=None):
        state = self._load_state()
        cash = float(state.get("cash", self.starting_cash))
        positions = {
            symbol: float(quantity)
            for symbol, quantity in state.get("positions", {}).items()
        }
        prices_by_symbol = prices_by_symbol or {}
        equity = self._equity(cash, positions, prices_by_symbol)
        return {
            "starting_cash": float(state.get("starting_cash", self.starting_cash)),
            "cash": cash,
            "positions": positions,
            "mark_to_market_equity": equity,
            "prices_used": prices_by_symbol,
            "fills": state.get("fills", []),
            "equity_history": state.get("equity_history", []),
            "filled_decision_paths": state.get("filled_decision_paths", []),
            "last_decision_path": state.get("last_decision_path"),
            "last_fill_at": state.get("last_fill_at"),
            "state_path": str(self.state_path),
        }

    def latest_decision_payload(self):
        path = self._latest_decision_path()
        if path is None:
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        payload["report_path"] = str(path)
        return payload

    def repair_state(self, prices_by_symbol=None):
        state = self._load_state()
        prices_by_symbol = prices_by_symbol or {}
        before_fills = len(state.get("fills", []))
        before_equity = len(state.get("equity_history", []))

        # Preserve both real fills and broker-submission audit records.
        fills = [
            fill
            for fill in state.get("fills", [])
            if (
                fill.get("fills")
                or fill.get("submitted_orders")
                or fill.get("status") in {"submitted", "partial", "filled"}
            )
        ]

        positions = {
            symbol: float(quantity)
            for symbol, quantity in state.get("positions", {}).items()
        }
        cash = float(state.get("cash", self.starting_cash))
        equity_history = [
            item
            for item in state.get("equity_history", [])
            if float(item.get("equity", 0) or 0) > 0
        ]
        equity = self._equity(cash, positions, prices_by_symbol)

        if prices_by_symbol:
            equity_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "equity": equity,
                "cash": cash,
                "source": "repair",
            })

        state["starting_cash"] = float(state.get("starting_cash", self.starting_cash))
        state["cash"] = cash
        state["positions"] = positions
        state["fills"] = fills
        state["equity_history"] = equity_history
        state.setdefault("filled_decision_paths", [])
        state["last_repair_at"] = datetime.utcnow().isoformat()
        self._save_state(state)

        return {
            "state_path": str(self.state_path),
            "removed_empty_fills": before_fills - len(fills),
            "removed_bad_equity_snapshots": before_equity - len(equity_history),
            "cash": cash,
            "positions": positions,
            "equity": equity,
        }

    def reset_state(self):
        state = {
            "starting_cash": self.starting_cash,
            "cash": self.starting_cash,
            "positions": {},
            "fills": [],
            "equity_history": [],
            "filled_decision_paths": [],
            "created_at": datetime.utcnow().isoformat(),
            "note": "Paper state reset.",
        }
        self._save_state(state)
        return {
            "state_path": str(self.state_path),
            "cash": self.starting_cash,
            "positions": {},
        }

    def _equity(self, cash, positions, prices_by_symbol):
        return cash + sum(
            quantity * prices_by_symbol.get(symbol, 0)
            for symbol, quantity in positions.items()
        )

    def _float_or_none(self, value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _load_state(self):
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "cash": self.starting_cash,
                "positions": {},
            }

    def _save_state(self, state):
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(state, indent=2),
            encoding="utf-8",
        )

    def _decision_key(self, path):
        try:
            return str(path.resolve())
        except OSError:
            return str(path)

    def _report_path(self, timestamp):
        return self.report_dir / f"{timestamp.date()}_decision.json"

    def _latest_decision_path(self):
        paths = sorted(self.report_dir.glob("*_decision.json"))
        return paths[-1] if paths else None
