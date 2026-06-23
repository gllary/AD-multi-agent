# LightGBM Model Artifacts

Directory layout:

```text
lgbm/
  CP1/fold_models/fold*_booster.txt
  CP2/fold_models/fold*_booster.txt
  CP2E/fold_models/fold*_booster.txt
```

This repository includes the frozen five-fold LightGBM model artifacts aligned
with the Cohort D development boundary used in the manuscript.

If models are retrained for a separate exploratory analysis, replace the
corresponding `fold*_booster.txt` files and document that the result is no
longer the frozen manuscript configuration.
