from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


class PaperTradingEngineFillMixin:
    def fill_latest_decision(self, decision_path=None):
        self.report_dir.mkdir(parents=True, exist_ok=True)
        path = Path(decision_path) if decision_path else self._latest_decision_path()

        if path is None:
            raise FileNotFoundError("No paper decision file found to fill.")

        decision = json.loads(path.read_text(encoding="utf-8"))
        state = self._load_state()
        decision_key = self._decision_key(path)
        filled_decision_paths = state.setdefault("filled_decision_paths", [])
        cash = float(state.get("cash", self.starting_cash))
        positions = {
            symbol: float(quantity)
            for symbol, quantity in state.get("positions", {}).items()
        }
        prices = {
            order["symbol"]: float(order["price"])
            for order in decision.get("orders", [])
        }

        if decision_key in filled_decision_paths:
            return {
                "status": "already_filled", "already_filled": True, "no_orders": False,
                "filled_at": None, "decision_path": str(path), "decision_timestamp": decision.get("timestamp"),
                "fills": [], "cash_after": cash, "positions_after": positions,
                "equity_after": float(decision.get("equity", self._equity(cash, positions, prices))),
            }

        if not decision.get("orders", []):
            fill_record = {
                "status": "no_orders", "already_filled": False, "no_orders": True,
                "filled_at": None, "decision_path": str(path), "decision_timestamp": decision.get("timestamp"),
                "fills": [], "cash_after": cash, "positions_after": positions,
                "equity_after": float(decision.get("equity", self._equity(cash, positions, prices))),
            }
            filled_decision_paths.append(decision_key)
            state["last_decision_path"] = str(path)
            self._save_state(state)
            return fill_record

        fills = []
        for order in decision.get("orders", []):
            symbol = order["symbol"]
            quantity_delta = float(order["quantity_delta"])
            dollar_delta = float(order["dollar_delta"])
            cash -= dollar_delta
            positions[symbol] = positions.get(symbol, 0) + quantity_delta
            if abs(positions[symbol]) < 1e-10:
                positions.pop(symbol)
            fills.append({"symbol": symbol, "side": order["side"], "quantity_delta": quantity_delta, "dollar_delta": dollar_delta, "price": order["price"], "reason": order["reason"], "fees": 0})

        equity = self._equity(cash, positions, prices)
        fill_record = {"status": "filled", "already_filled": False, "no_orders": False, "filled_at": datetime.utcnow().isoformat(), "decision_path": str(path), "decision_timestamp": decision.get("timestamp"), "fills": fills, "cash_after": cash, "positions_after": positions, "equity_after": equity}
        state.setdefault("fills", []).append(fill_record)
        state.setdefault("equity_history", []).append({"timestamp": decision.get("timestamp"), "equity": equity, "cash": cash})
        filled_decision_paths.append(decision_key)
        state["starting_cash"] = float(state.get("starting_cash", self.starting_cash))
        state["cash"] = cash
        state["positions"] = positions
        state["last_fill_at"] = fill_record["filled_at"]
        state["last_decision_path"] = str(path)
        self._save_state(state)
        self._append_fill_log(fill_record, decision)
        return fill_record

    def apply_external_fill_record(self, decision_path, fill_record):
        path = Path(decision_path)
        decision = json.loads(path.read_text(encoding="utf-8"))
        state = self._load_state()
        decision_key = self._decision_key(path)
        filled_decision_paths = state.setdefault("filled_decision_paths", [])
        if decision_key in filled_decision_paths:
            return {**fill_record, "status": "already_filled", "already_filled": True, "fills": []}
        state.setdefault("fills", []).append(fill_record)
        equity_after = fill_record.get("equity_after")
        cash_after = fill_record.get("cash_after")
        positions_after = fill_record.get("positions_after")
        if equity_after is not None:
            state.setdefault("equity_history", []).append({"timestamp": decision.get("timestamp"), "equity": equity_after, "cash": cash_after})
        status = str(fill_record.get("status", "")).lower()
        if status in {"filled", "no_orders"}:
            filled_decision_paths.append(decision_key)
        state["starting_cash"] = float(state.get("starting_cash", self.starting_cash))
        if cash_after is not None: state["cash"] = cash_after
        else: state.setdefault("cash", self.starting_cash)
        if positions_after is not None: state["positions"] = positions_after
        else: state.setdefault("positions", {})
        if fill_record.get("filled_at"): state["last_fill_at"] = fill_record.get("filled_at")
        state["last_decision_path"] = str(path)
        if status in {"submitted", "partial"}:
            state["last_open_order_decision_path"] = str(path)
            state["last_open_order_status"] = status
        self._save_state(state)
        self._append_fill_log(fill_record, decision)
        return fill_record

    def _append_fill_log(self, fill_record, decision):
        if not fill_record.get("fills"):
            return
        self.fill_log_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["timestamp", "broker_order_id", "symbol", "side", "quantity", "price", "fees", "strategy_id", "candidate_id"]
        exists = self.fill_log_path.exists()
        strategy_id = decision.get("model_context", {}).get("strategy") or "dual_momentum"
        with self.fill_log_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not exists: writer.writeheader()
            for index, fill in enumerate(fill_record["fills"], start=1):
                writer.writerow({"timestamp": fill_record.get("filled_at"), "broker_order_id": fill.get("broker_order_id") or f"paper-{fill_record['decision_timestamp']}-{index}", "symbol": fill["symbol"], "side": fill["side"], "quantity": abs(float(fill["quantity_delta"])), "price": fill["price"], "fees": fill.get("fees", 0), "strategy_id": strategy_id, "candidate_id": self.candidate_id})
