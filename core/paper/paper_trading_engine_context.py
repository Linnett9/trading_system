from __future__ import annotations


class PaperTradingEngineContextMixin:
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
            "scores": dict(selection.scores or {}),
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
