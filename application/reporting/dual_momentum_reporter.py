from core.research.dual_momentum.reporting_console import (
    print_dual_momentum_diagnosis,
    print_dual_momentum_experiments,
    print_dual_momentum_result,
    print_dual_momentum_risk_regime_experiments,
    print_dual_momentum_walk_forward,
)
from core.research.dual_momentum.reporting_explanations import (
    dual_momentum_result_explanation,
    dual_momentum_walk_forward_explanation,
)
from core.research.dual_momentum.reporting_format import (
    format_optional_percent,
    format_percent,
)
from core.research.dual_momentum.reporting_labels import (
    champion_delta_label,
    production_gap_label,
    walk_forward_readiness_label,
)
from core.research.dual_momentum.reporting_selection import (
    _candidates_with_tag,
    _champion_sort_key,
    champion_label,
    risk_regime_report_items,
    select_champion,
    select_raw_score_leader,
)
from core.research.dual_momentum.reporting_types import (
    FROZEN_CHAMPION_CONFIG_NAME,
    FROZEN_CHAMPION_ID,
    RISK_REGIME_REPORT_LIMIT,
)
