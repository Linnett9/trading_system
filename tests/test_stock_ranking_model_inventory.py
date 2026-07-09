from core.research.ml.stock_level.model_inventory import (
    FULLY_RUNNABLE,
    PARTIALLY_WIRED,
    score_column_for_model,
    stock_ranking_model_inventory,
    stock_ranking_model_sets,
)


def test_model_inventory_classifies_runnable_and_partial_models():
    rows = {
        row.registry_key: row
        for row in stock_ranking_model_inventory()
    }

    assert rows["ridge"].status == FULLY_RUNNABLE
    assert rows["elastic_net"].oos_benchmark_support is True
    assert rows["dlinear"].status == FULLY_RUNNABLE
    assert rows["patchtst"].status == FULLY_RUNNABLE
    assert rows["news_analysis_transformer"].status == PARTIALLY_WIRED
    assert rows["news_analysis_transformer"].prediction_artifact_support is True


def test_staged_model_sets_use_real_repository_keys():
    model_sets = stock_ranking_model_sets()

    assert model_sets["fast_tabular"] == (
        "ridge",
        "elastic_net",
        "random_forest",
        "gradient_boosting",
    )
    assert "dlinear" in model_sets["sequence_smoke"]
    assert "patchtst" in model_sets["sequence_smoke"]
    assert "transformer" in model_sets["transformer_smoke"]
    assert "market_context_encoder" in model_sets["context_smoke"]
    assert "news_analysis_transformer" not in model_sets["full_research_candidates"]


def test_score_column_resolution_uses_artifact_contract():
    assert (
        score_column_for_model("patchtst")
        == "stock_level_predicted_forward_return_10d_patchtst"
    )
    assert score_column_for_model("momentum_120d") == "predicted_momentum_120d"
