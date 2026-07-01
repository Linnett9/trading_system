# Architecture Assets

This directory contains hand-authored SVG architecture diagrams used by
`docs/architecture_diagrams.md`.

Main docs:

- [Architecture diagrams](../../architecture_diagrams.md)
- [Architecture diagram explainer](../../architecture_diagram_explainer.md)

## SVG Index

| SVG | Shows |
| --- | --- |
| `system_architecture_overview.svg` | One-page repository/system overview from CLI and application orchestration through config, domain core, research, infrastructure, outputs, tests, and the blocked research-to-execution path. |
| `stock_alpha_architecture_overview.svg` | One-page stock-alpha architecture from inputs through artifacts, model evaluation, portfolio simulations, validation reports, guardrails, and blocked broker orders. |
| `repository_layers.svg` | Repository dependency layers and generated output areas. |
| `research_execution_boundary.svg` | Research-only modules, simulated outputs, execution-adjacent modules, and the boundary that prevents research reports from placing trades. |
| `stock_alpha_pipeline.svg` | Stock-alpha stage flow from inputs to artifacts, features, model ranking, target comparison, replay, sweep, report, attribution, and overnight summary. |
| `data_lineage.svg` | Data lineage from raw/processed Stooq data and references into features, future labels, artifacts, OOS predictions, and reports. |
| `model_evaluation_loop.svg` | Chronological walk-forward model evaluation loop and OOS prediction flow into portfolio simulations. |
| `gates_and_decisions.svg` | Research correctness gates, quality gates, candidate triage, manual review, deeper validation, and blocked paper/live authorization. |
| `output_report_map.svg` | Canonical stock-alpha output root and the major report/artifact files produced under it. |

These SVGs contain no embedded JavaScript and no remote assets. Long
explanations belong in Markdown docs rather than inside diagram boxes.
