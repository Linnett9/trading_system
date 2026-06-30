from __future__ import annotations

from core.paper.paper_trading_engine_types import PaperOrder


class PaperTradingEngineOrderMixin:
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
            quantity_delta = self._normalized_quantity_delta(
                quantity_delta=quantity_delta,
                side=side,
                current_quantity=current_quantity,
            )

            if abs(quantity_delta) <= 0:
                continue

            dollar_delta = quantity_delta * price

            if abs(dollar_delta) < self.min_trade_value:
                continue

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
                order_type=self.order_type,
                limit_price=self._limit_price(price, side),
            ))

        return sorted(
            orders,
            key=lambda order: (
                0 if order.side == "SELL" else 1,
                -abs(order.dollar_delta) if order.side == "SELL" else 0,
                -(order.score if order.score is not None else float("-inf")),
                -abs(order.dollar_delta) if order.side == "BUY" else 0,
                order.symbol,
            ),
        )

    def _normalized_quantity_delta(self, quantity_delta, side, current_quantity):
        if self.supports_fractional:
            return round(quantity_delta, self.quantity_precision)

        whole_quantity = int(abs(quantity_delta))

        if whole_quantity <= 0:
            return 0

        if side == "SELL":
            whole_quantity = min(whole_quantity, int(abs(current_quantity)))
            return -float(whole_quantity)

        return float(whole_quantity)

    def _limit_price(self, price, side):
        if self.order_type != "LIMIT":
            return None

        offset = self.limit_offset_bps / 10000

        if side == "BUY":
            return price * (1 + offset)

        return price * (1 - offset)

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


