# Reproducibility notes

## Reproduced exactly from the surviving final pipeline

- Final cohort counts for 2012–2021: Hitter 534, SP 216, RP 412
- Temporal split: 2012–2017 / 2018–2019 / 2020–2021
- Table 4 selected 3-class Composite metrics
- Table 5 selected 2-class Composite metrics
- Table 6 Composite vs WAR-only displayed accuracies
- Table 6 paired percentile bootstrap 95% CI (10,000 resamples, seed 42)
- Figure 5–6 model coefficients / RF impurity importance
- AAV correlation reference outputs in `outputs/reference/aav_external_validation/`

## Important model-selection provenance

`run_temporal_final_models.py` calculates all candidate models. The manuscript's representative-model choices are explicitly stored in `config/paper_model_spec.json` and are used by `build_paper_tables.py`.

This is deliberate: the surviving generic automatic `temporal_best_models.csv` selection logic does not reproduce every representative model named in the manuscript. Therefore the GitHub reproduction path separates:

1. model fitting/evaluation of every candidate, and
2. the exact model combination reported in the manuscript.

This discrepancy should be reconciled in the manuscript methods if the repository is intended for external peer review.

## Table 8 provenance issue

The current V2 cohort reproduces PCA variance approximately:

- Hitter ≈ 0.858
- Starting Pitcher ≈ 0.847
- Relief Pitcher ≈ 0.793

However, the exact 0.4:0.6 and 0.6:0.4 label-agreement values in the manuscript are not fully reproduced by the surviving final data/code. The repository therefore recomputes them transparently and reports the mismatch rather than embedding manuscript numbers as if they had been regenerated.

See `outputs/paper/REPRODUCTION_VALIDATION.md` after running the pipeline.
