from __future__ import annotations

from core.research.evidence.retirement_safety import (
    OperationalDependencyEvidence,
    RetirementDependencyClassification,
    evaluate_operational_retirement,
)


def test_current_dependency_is_not_retirable_when_content_is_preserved() -> None:
    decision = evaluate_operational_retirement(
        OperationalDependencyEvidence(
            source_path=r"C:\Users\Brandon\hf_equity_trial_r1\models.lock.json",
            source_role="RUNTIME_MODEL_LOCK",
            content_preserved=True,
            search_complete=True,
            basename_references=("scripts/hf_r1/native_worker.py:111",),
        )
    )

    assert decision.classification is RetirementDependencyClassification.CURRENT_OPERATIONAL_DEPENDENCY
    assert decision.retirement_eligible is False


def test_incomplete_dependency_search_fails_closed() -> None:
    decision = evaluate_operational_retirement(
        OperationalDependencyEvidence(
            source_path="unknown.json",
            source_role="RESULT_OR_EVIDENCE",
            content_preserved=True,
            search_complete=False,
        )
    )

    assert decision.classification is RetirementDependencyClassification.AMBIGUOUS_NEEDS_REVIEW
    assert decision.retirement_eligible is False


def test_documentary_reference_is_not_a_current_dependency() -> None:
    decision = evaluate_operational_retirement(
        OperationalDependencyEvidence(
            source_path="historical-report.json",
            source_role="HISTORICAL_EVIDENCE",
            content_preserved=True,
            search_complete=True,
            historical_references=("immutable audit report",),
        )
    )

    assert decision.classification is RetirementDependencyClassification.HISTORICAL_REFERENCE_ONLY
    assert decision.retirement_eligible is True
