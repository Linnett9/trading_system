# ML Model Contract Audit

| Model | Registry Key | Class | Status | Configs | predicted_* Outputs | TODOs |
|---|---|---|---|---:|---|---|
| DLinear | `dlinear` | `DLinearSequenceMLModel` | COMPLETE_V1 | 5 | - | - |
| PatchTST | `patchtst` | `PatchTSTSequenceMLModel` | COMPLETE_V1 | 6 | - | - |
| Transformer | `transformer` | `TransformerSequenceMLModel` | COMPLETE_V1 | 5 | - | - |
| ITransformer | `itransformer` | `ITransformerSequenceMLModel` | COMPLETE_V1 | 1 | - | Full cross-asset rank-score design is not implemented; do not emit predicted_rank_score until supported by model internals. |
| Momentum Transformer | `momentum_transformer` | `MomentumTransformerSequenceMLModel` | COMPLETE_V1 | 1 | predicted_trend_score, predicted_regime_score, predicted_size_multiplier | - |
| Multi-task Transformer | `multitask_transformer` | `MultiTaskTransformerSequenceMLModel` | COMPLETE_V1 | 2 | predicted_forward_return_5d, predicted_forward_return_10d, predicted_future_volatility, predicted_future_drawdown | - |
| Market Context Encoder | `market_context_encoder` | `MarketContextEncoderMLModel` | COMPLETE_V1 | 1 | predicted_context_risk_multiplier | Context embeddings and detailed regime diagnostics are intentionally not part of the v1 artifact contract. |
| News Analysis Transformer | `news_analysis_transformer` | `NewsAnalysisTransformerMLModel` | COMPLETE_V1 | 1 | - | Full timestamped news ingestion and FinBERT-style sentiment pipeline remain out of scope for v1. |
| Temporal Fusion Transformer | `temporal_fusion_transformer` | `TemporalFusionTransformerMLModel` | COMPLETE_V1 | 1 | predicted_forward_return_5d, predicted_forward_return_10d, predicted_future_volatility, predicted_future_drawdown | Full TFT interpretability artifacts are intentionally out of scope for v1. |
| Logistic Regression | `logistic_regression` | `LogisticRegressionMLModel` | COMPLETE_V1 | 1 | - | - |
| Random Forest | `random_forest` | `TreeClassifierMLModel` | COMPLETE_V1 | 1 | - | - |
| Gradient Boosting | `gradient_boosting` | `TreeClassifierMLModel` | COMPLETE_V1 | 1 | - | - |
| Meta Ensemble | `meta_ensemble` | `run_meta_ensemble` | COMPLETE_V1 | 1 | - | Meta Ensemble is an orchestration/reporting pipeline, not an IMLModel with save/load persistence. |
