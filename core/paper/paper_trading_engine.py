from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path


@dataclass(frozen=True)
class PaperOrder:
    symbol: str
    side: str
    quantity_delta: float
    dollar_delta: float
    current_weight: float
    target_weight: float
    drift_weight: float
    price: float
    reason: str
    score: float | None = None

    def to_dict(self):
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity_delta": self.quantity_delta,
            "dollar_delta": self.dollar_delta,
            "current_weight": self.current_weight,
            "target_weight": self.target_weight,
            "drift_weight": self.drift_weight,
            "price": self.price,
            "reason": self.reason,
            "score": self.score,
        }


@dataclass(frozen=True)
class PaperDecision:
    timestamp: object
    regime_label: str
    risk_on: bool
    exposure_target: float
    target_weights: dict[str, float]
    current_positions: dict[str, float]
    cash: float
    equity: float
    orders: list[PaperOrder]
    selected_symbols: list[str]
    model_context: dict
    data_freshness: dict
    report_path: Path
    state_path: Path

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "regime_label": self.regime_label,
            "risk_on": self.risk_on,
            "exposure_target": self.exposure_target,
            "target_weights": self.target_weights,
            "current_positions": self.current_positions,
            "cash": self.cash,
            "equity": self.equity,
            "orders": [order.to_dict() for order in self.orders],
            "selected_symbols": self.selected_symbols,
            "model_context": self.model_context,
            "data_freshness": self.data_freshness,
            "state_path": str(self.state_path),
        }


class PaperTradingEngine:
    def __init__(
        self,
        report_dir="reports/paper",
        starting_cash=500,
        min_trade_value=1.0,
        rebalance_threshold=0.0,
    ):
        self.report_dir = Path(report_dir)
        self.starting_cash = starting_cash
        self.min_trade_value = min_trade_value
        self.rebalance_threshold = rebalance_threshold

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
        fills = [
            fill
            for fill in state.get("fills", [])
            if fill.get("fills")
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

    def create_decision(
        self,
        dual_momentum_result,
        prices_by_symbol,
        data_freshness=None,
    ):
        if not dual_momentum_result.selections:
            raise ValueError("Cannot paper trade without a strategy selection.")

        self.report_dir.mkdir(parents=True, exist_ok=True)
        state = self._load_state()
        selection = dual_momentum_result.selections[-1]
        model_context = self._model_context(dual_momentum_result, selection)
        target_weights = selection.target_weights or {}
        positions = {
            symbol: float(quantity)
            for symbol, quantity in state.get("positions", {}).items()
        }
        cash = float(state.get("cash", self.starting_cash))
        equity = self._equity(cash, positions, prices_by_symbol)
        orders = self._orders(
            target_weights=target_weights,
            exposure_target=selection.exposure_target,
            positions=positions,
            prices_by_symbol=prices_by_symbol,
            equity=equity,
            selected_symbols=selection.symbols,
            scores=selection.scores,
            model_context=model_context,
            rebalance_threshold=self.rebalance_threshold,
        )
        report_path = self._report_path(selection.timestamp)
        decision = PaperDecision(
            timestamp=selection.timestamp,
            regime_label=selection.regime_label,
            risk_on=selection.risk_on,
            exposure_target=selection.exposure_target,
            target_weights=target_weights,
            current_positions=positions,
            cash=cash,
            equity=equity,
            orders=orders,
            selected_symbols=selection.symbols,
            model_context=model_context,
            data_freshness=data_freshness or {},
            report_path=report_path,
            state_path=self.state_path,
        )
        report_path.write_text(
            json.dumps(decision.to_dict(), indent=2),
            encoding="utf-8",
        )
        if not self.state_path.exists():
            self._save_state({
                "starting_cash": self.starting_cash,
                "cash": cash,
                "positions": positions,
                "filled_decision_paths": [],
                "created_at": datetime.utcnow().isoformat(),
                "note": "Initial paper state. Orders are not auto-filled.",
            })
        return decision

    def fill_latest_decision(self, decision_path=None):
        self.report_dir.mkdir(parents=True, exist_ok=True)
        path = (
            Path(decision_path)
            if decision_path
            else self._latest_decision_path()
        )
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
                "status": "already_filled",
                "already_filled": True,
                "no_orders": False,
                "filled_at": None,
                "decision_path": str(path),
                "decision_timestamp": decision.get("timestamp"),
                "fills": [],
                "cash_after": cash,
                "positions_after": positions,
                "equity_after": self._equity(cash, positions, prices),
            }

        if not decision.get("orders", []):
            return {
                "status": "no_orders",
                "already_filled": False,
                "no_orders": True,
                "filled_at": None,
                "decision_path": str(path),
                "decision_timestamp": decision.get("timestamp"),
                "fills": [],
                "cash_after": cash,
                "positions_after": positions,
                "equity_after": self._equity(cash, positions, prices),
            }

        fills = []

        for order in decision.get("orders", []):
            symbol = order["symbol"]
            quantity_delta = float(order["quantity_delta"])
            dollar_delta = float(order["dollar_delta"])
            cash -= dollar_delta
            positions[symbol] = positions.get(symbol, 0) + quantity_delta
            if abs(positions[symbol]) < 1e-10:
                positions.pop(symbol)
            fills.append({
                "symbol": symbol,
                "side": order["side"],
                "quantity_delta": quantity_delta,
                "dollar_delta": dollar_delta,
                "price": order["price"],
                "reason": order["reason"],
            })

        equity = self._equity(cash, positions, prices)
        fill_record = {
            "status": "filled",
            "already_filled": False,
            "no_orders": False,
            "filled_at": datetime.utcnow().isoformat(),
            "decision_path": str(path),
            "decision_timestamp": decision.get("timestamp"),
            "fills": fills,
            "cash_after": cash,
            "positions_after": positions,
            "equity_after": equity,
        }
        state.setdefault("fills", []).append(fill_record)
        state.setdefault("equity_history", []).append({
            "timestamp": decision.get("timestamp"),
            "equity": equity,
            "cash": cash,
        })
        filled_decision_paths.append(decision_key)
        state["starting_cash"] = float(state.get("starting_cash", self.starting_cash))
        state["cash"] = cash
        state["positions"] = positions
        state["last_fill_at"] = fill_record["filled_at"]
        state["last_decision_path"] = str(path)
        self._save_state(state)
        return fill_record

    def _orders(
        self,
        target_weights,
        exposure_target,
        positions,
        prices_by_symbol,
        equity,
        selected_symbols,
        scores,
        model_context,
        rebalance_threshold,
    ):
        orders = []
        symbols = sorted(set(positions) | set(target_weights))
        for symbol in symbols:
            price = prices_by_symbol.get(symbol)
            if price is None or price <= 0:
                continue

            current_quantity = positions.get(symbol, 0)
            current_value = current_quantity * price
            current_weight = current_value / equity if equity else 0
            target_weight = target_weights.get(symbol, 0) * exposure_target
            drift_weight = target_weight - current_weight
            if abs(drift_weight) < rebalance_threshold:
                continue

            target_value = equity * target_weight
            dollar_delta = target_value - current_value
            if abs(dollar_delta) < self.min_trade_value:
                continue

            quantity_delta = dollar_delta / price
            side = "BUY" if dollar_delta > 0 else "SELL"
            score = scores.get(symbol)
            reason = self._order_reason(
                symbol=symbol,
                selected_symbols=selected_symbols,
                score=score,
                model_context=model_context,
            )
            orders.append(PaperOrder(
                symbol=symbol,
                side=side,
                quantity_delta=quantity_delta,
                dollar_delta=dollar_delta,
                current_weight=current_weight,
                target_weight=target_weight,
                drift_weight=drift_weight,
                price=price,
                reason=reason,
                score=score,
            ))
        return orders

    def _order_reason(
        self,
        symbol,
        selected_symbols,
        score,
        model_context,
    ):
        if symbol not in selected_symbols:
            return "no longer selected by current model"

        selection_mode = model_context.get("selection_mode")
        ranking_mode = model_context.get("ranking_score_mode")
        if selection_mode == "all_positive":
            reason = "positive momentum asset in all-positive mode"
        else:
            reason = f"ranked inside top {model_context.get('top_n')}"

        if score is not None:
            reason += f"; score={score:.4f}"

        if ranking_mode == "enhanced":
            reason += "; score uses momentum, relative strength, and volatility"
        else:
            reason += "; score uses average momentum"

        return reason

    def _model_context(self, dual_momentum_result, selection):
        config = dual_momentum_result.config
        target_weight_sum = sum((selection.target_weights or {}).values())
        return {
            "strategy": "dual_momentum",
            "selection_mode": config.get("selection_mode"),
            "ranking_score_mode": config.get("ranking_score_mode"),
            "top_n": config.get("top_n"),
            "min_selection_score": config.get("min_selection_score", 0),
            "max_selected_assets": config.get("max_selected_assets"),
            "momentum_periods": config.get("momentum_periods"),
            "weighting": config.get("weighting"),
            "regime_label": selection.regime_label,
            "regime_exposure": getattr(selection, "regime_exposure", 0),
            "exposure_target": selection.exposure_target,
            "risk_on": selection.risk_on,
            "breadth_passes": getattr(selection, "breadth_passes", False),
            "fast_reentry": getattr(selection, "fast_reentry", False),
            "chop_filter_active": getattr(
                selection,
                "chop_filter_active",
                False,
            ),
            "drawdown_guard_active": getattr(
                selection,
                "drawdown_guard_active",
                False,
            ),
            "target_weight_sum": target_weight_sum,
            "selected_count": len(selection.symbols),
            "skipped_assets": self._skipped_assets(config, selection),
            "explanation": self._selection_explanation(config, selection),
        }

    def _selection_explanation(self, config, selection):
        if selection.regime_label == "cash":
            return "The model moved to cash because risk conditions failed."

        mode = config.get("selection_mode")
        if mode == "all_positive":
            universe_reason = (
                "all assets with positive momentum and passing filters"
            )
        else:
            universe_reason = (
                f"the top {config.get('top_n')} ranked assets"
            )

        pieces = [
            f"The model selected {universe_reason}.",
            f"Regime is {selection.regime_label}.",
        ]
        if getattr(selection, "chop_filter_active", False):
            pieces.append(
                "Chop filter reduced exposure because broad momentum is weak."
            )
        if getattr(selection, "fast_reentry", False):
            pieces.append(
                "Fast re-entry allowed partial risk after recovery signals."
            )
        if getattr(selection, "drawdown_guard_active", False):
            pieces.append("Drawdown guard is active.")
        return " ".join(pieces)

    def _skipped_assets(self, config, selection):
        selected = set(selection.symbols)
        min_score = config.get("min_selection_score", 0) or 0
        max_assets = config.get("max_selected_assets")
        skipped = []
        ranked_scores = sorted(
            (selection.scores or {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )
        for rank, (symbol, score) in enumerate(ranked_scores, start=1):
            if symbol in selected:
                continue

            if score < min_score:
                reason = (
                    f"score {score:.4f} below min_selection_score "
                    f"{min_score:.4f}"
                )
            elif max_assets is not None and rank > max_assets:
                reason = f"outside max_selected_assets={max_assets}"
            else:
                reason = "filtered out by selection rules"

            skipped.append({
                "symbol": symbol,
                "score": score,
                "reason": reason,
            })
        return skipped

    def _equity(self, cash, positions, prices_by_symbol):
        return cash + sum(
            quantity * prices_by_symbol.get(symbol, 0)
            for symbol, quantity in positions.items()
        )

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
