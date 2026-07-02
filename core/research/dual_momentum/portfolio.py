from core.entities.backtest_result import BacktestResult
from core.entities.signal_diagnostics import SignalDiagnostics
from core.research.dual_momentum.analytics import DualMomentumAnalyticsMixin
from core.research.dual_momentum.backtester_init import (
    DualMomentumBacktesterInitMixin,
)
from core.research.dual_momentum.backtester_run import (
    DualMomentumBacktesterRunMixin,
)
from core.research.dual_momentum.models import (
    DualMomentumSelection,
    DualMomentumResult,
)
from core.research.performance_metrics import (
    cagr,
    max_drawdown,
    sharpe_ratio,
    total_return,
)
from core.research.walk_forward import normalize_datetime
from core.services.portfolio_engine import EquityPoint
from core.research.dual_momentum.weighting import DualMomentumWeightingMixin
from core.research.dual_momentum.execution import DualMomentumExecutionMixin
from core.research.dual_momentum.ranking import DualMomentumRankingMixin
from core.research.dual_momentum.regimes import DualMomentumRegimeMixin
from core.research.dual_momentum.data import DualMomentumDataMixin
from core.research.dual_momentum.config_snapshot import (
    DualMomentumConfigSnapshotMixin,
)


class DualMomentumPortfolioBacktester(
    DualMomentumBacktesterInitMixin,
    DualMomentumBacktesterRunMixin,
    DualMomentumAnalyticsMixin,
    DualMomentumWeightingMixin,
    DualMomentumExecutionMixin,
    DualMomentumRankingMixin,
    DualMomentumRegimeMixin,
    DualMomentumDataMixin,
    DualMomentumConfigSnapshotMixin,
):
    pass
