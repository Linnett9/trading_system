from core.research.dual_momentum.ranking_filters import DualMomentumRankingFiltersMixin
from core.research.dual_momentum.ranking_hysteresis import DualMomentumRankingHysteresisMixin
from core.research.dual_momentum.ranking_primitives import DualMomentumRankingPrimitivesMixin
from core.research.dual_momentum.ranking_scores import DualMomentumRankingScoresMixin
from core.research.dual_momentum.ranking_selection import DualMomentumRankingSelectionMixin


class DualMomentumRankingMixin(
    DualMomentumRankingSelectionMixin,
    DualMomentumRankingScoresMixin,
    DualMomentumRankingHysteresisMixin,
    DualMomentumRankingFiltersMixin,
    DualMomentumRankingPrimitivesMixin,
):
    pass
