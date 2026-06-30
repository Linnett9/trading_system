from datetime import datetime
import json
from pathlib import Path

from core.paper.paper_trading_engine_types import PaperDecision, PaperOrder
from core.paper.paper_trading_engine_state import PaperTradingEngineStateMixin
from core.paper.paper_trading_engine_fills import PaperTradingEngineFillMixin
from core.paper.paper_trading_engine_orders import PaperTradingEngineOrderMixin
from core.paper.paper_trading_engine_context import PaperTradingEngineContextMixin


class PaperTradingEngine(
    PaperTradingEngineContextMixin,
    PaperTradingEngineOrderMixin,
    PaperTradingEngineFillMixin,
    PaperTradingEngineStateMixin,
):
    def __init__(
        self,
        report_dir="reports/paper",
        starting_cash=500,
        min_trade_value=1.0,
        rebalance_threshold=0.0,
        fill_log_path="data/paper/fills.csv",
        candidate_id="",
        supports_fractional=True,
        quantity_precision=6,
        order_type="MARKET",
        limit_offset_bps=0.0,
    ):
        self.report_dir = Path(report_dir)
        self.starting_cash = starting_cash
        self.min_trade_value = min_trade_value
        self.rebalance_threshold = rebalance_threshold
        self.fill_log_path = Path(fill_log_path)
        self.candidate_id = candidate_id
        self.supports_fractional = supports_fractional
        self.quantity_precision = quantity_precision
        self.order_type = order_type.upper()
        self.limit_offset_bps = limit_offset_bps

    def sync_broker_state(
        self,
        account,
        positions,
        prices_by_symbol=None,
        sleeve_cash=None,
        source="broker",
    ):
        """Import broker positions into the local paper ledger.

        This is intentionally a ledger sync, not an order/fill. It is used when
        a real paper broker is the source of truth before the next decision is
        generated. For sleeve-style paper trials, pass sleeve_cash so the local
        ledger keeps the strategy's $500 sleeve separate from the broker's full
        paper-account cash balance.
        """
        state = self._load_state()
        prices_by_symbol = prices_by_symbol or {}

        clean_positions = {
            str(symbol).upper(): float(quantity)
            for symbol, quantity in (positions or {}).items()
            if abs(float(quantity or 0.0)) > 1e-10
        }

        broker_cash = self._float_or_none((account or {}).get("cash"))
        broker_equity = self._float_or_none(
            (account or {}).get("equity")
            if (account or {}).get("equity") is not None
            else (account or {}).get("portfolio_value")
        )
        broker_buying_power = self._float_or_none(
            (account or {}).get("buying_power")
        )

        position_value = sum(
            quantity * float(prices_by_symbol.get(symbol, 0.0) or 0.0)
            for symbol, quantity in clean_positions.items()
        )

        if sleeve_cash is not None:
            local_starting_cash = float(sleeve_cash)
            local_cash = local_starting_cash - position_value
            cash_source = "sleeve_cash_minus_broker_position_value"
        elif broker_cash is not None:
            local_starting_cash = float(
                state.get("starting_cash", self.starting_cash)
            )
            local_cash = float(broker_cash)
            cash_source = "broker_cash"
        else:
            local_starting_cash = float(
                state.get("starting_cash", self.starting_cash)
            )
            local_cash = float(state.get("cash", self.starting_cash))
            cash_source = "existing_local_cash"

        local_equity = self._equity(local_cash, clean_positions, prices_by_symbol)
        unpriced_positions = sorted(
            symbol for symbol in clean_positions
            if float(prices_by_symbol.get(symbol, 0.0) or 0.0) <= 0
        )

        state.setdefault("fills", [])
        state.setdefault("equity_history", [])
        state.setdefault("filled_decision_paths", [])
        state["starting_cash"] = local_starting_cash
        state["cash"] = local_cash
        state["positions"] = clean_positions
        state["last_broker_sync_at"] = datetime.utcnow().isoformat()
        state["last_broker_sync_source"] = source
        state["last_broker_sync"] = {
            "source": source,
            "cash_source": cash_source,
            "broker_cash": broker_cash,
            "broker_equity": broker_equity,
            "broker_buying_power": broker_buying_power,
            "sleeve_cash": (
                float(sleeve_cash) if sleeve_cash is not None else None
            ),
            "position_value": position_value,
            "local_cash": local_cash,
            "local_equity": local_equity,
            "positions": clean_positions,
            "unpriced_positions": unpriced_positions,
        }
        state["equity_history"].append({
            "timestamp": state["last_broker_sync_at"],
            "equity": local_equity,
            "cash": local_cash,
            "source": "broker_sync",
        })

        self._save_state(state)

        return {
            "state_path": str(self.state_path),
            "cash": local_cash,
            "positions": clean_positions,
            "equity": local_equity,
            "position_value": position_value,
            "unpriced_positions": unpriced_positions,
            "source": source,
            "cash_source": cash_source,
        }

    def create_decision(
        self,
        dual_momentum_result,
        prices_by_symbol,
        data_freshness=None,
        data_quality=None,
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
            data_quality=data_quality or {},
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
