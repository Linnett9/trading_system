from core.research.ml.reports.overlay_reports import MLOverlayReportWriter


def test_overlay_probabilities_are_averaged_by_feature_date():
    result = MLOverlayReportWriter.probabilities_by_date(
        ["2024-01-01", "2024-01-01", "2024-02-01"],
        [0.2, 0.8, 0.9],
    )

    assert result == {"2024-01-01": 0.5, "2024-02-01": 0.9}


def test_overlay_percentile_threshold_is_fitted_from_training_probabilities():
    threshold = MLOverlayReportWriter.quantile_threshold(
        [0.1, 0.2, 0.3, 0.4, 0.5],
        0.8,
    )

    assert threshold == 0.5
