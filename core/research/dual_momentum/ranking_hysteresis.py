


class DualMomentumRankingHysteresisMixin:
    def _apply_rank_hysteresis(self, selected, ranked, positions):
        max_replacements = getattr(
            self,
            "max_rebalance_replacements",
            None,
        )
        max_rank_override = getattr(
            self,
            "rank_hysteresis_max_rank",
            None,
        )
        replacement_score_gap = max(
            0,
            getattr(self, "replacement_score_gap", 0) or 0,
        )

        if (
            not getattr(self, "rank_hysteresis_enabled", False)
            and max_replacements is None
            and max_rank_override is None
        ):
            return selected

        if not positions:
            return selected

        margin = max(0, getattr(self, "rank_hysteresis_margin", 0))
        max_rank = (
            max_rank_override
            if max_rank_override is not None
            else self.top_n + margin
        )
        selected_set = set(selected)
        ranked_symbols = [symbol for symbol, _ in ranked]
        ranked_scores = dict(ranked)
        kept = []

        for symbol in positions:
            if symbol not in ranked_symbols:
                continue

            rank = ranked_symbols.index(symbol) + 1
            if (
                symbol in selected_set
                or (
                    getattr(self, "rank_hysteresis_enabled", False)
                    and rank <= max_rank
                )
            ):
                kept.append((rank, symbol))

        kept = [
            symbol
            for _, symbol in sorted(kept, key=lambda item: item[0])
        ]
        target_count = (
            self.max_selected_assets
            if self.max_selected_assets is not None
            else len(selected)
        )
        kept = kept[:target_count]
        replacement_slots = max(0, target_count - len(kept))

        if max_replacements is not None:
            replacement_slots = min(
                replacement_slots,
                max(0, max_replacements),
            )

        displaced = [
            symbol
            for symbol in positions
            if symbol in ranked_scores and symbol not in kept
        ]
        displaced = sorted(
            displaced,
            key=lambda symbol: ranked_symbols.index(symbol),
        )
        replacement_candidates = [
            symbol
            for symbol in selected
            if symbol not in kept
        ]
        replacements = []
        available_displaced = list(displaced)

        for candidate in replacement_candidates:
            if len(replacements) >= replacement_slots:
                break

            incumbent = (
                available_displaced[0]
                if available_displaced
                else None
            )

            if not self._passes_replacement_score_gap(
                candidate,
                [incumbent] if incumbent else [],
                ranked_scores,
                replacement_score_gap,
            ):
                available_displaced.pop(0)
                replacements.append(incumbent)
                continue

            if incumbent:
                available_displaced.pop(0)

            replacements.append(candidate)

        result = kept + replacements

        if not result:
            return selected

        return result[:target_count]
    def _passes_replacement_score_gap(
        self,
        candidate,
        displaced,
        ranked_scores,
        replacement_score_gap,
    ):
        if replacement_score_gap <= 0 or not displaced:
            return True

        candidate_score = ranked_scores.get(candidate)
        displaced_score = ranked_scores.get(displaced[0])

        if candidate_score is None or displaced_score is None:
            return True

        return (candidate_score - displaced_score) >= replacement_score_gap
