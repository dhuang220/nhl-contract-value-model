# archive/

Off-pipeline code kept for provenance. None of this is imported by the live model
(`src/`), the dashboards, or `generate_rankings`; it documents exploration and
earlier approaches that shaped the final model.

- **`experiment_features.py`** - feature experiments (defensive metrics, Lasso,
  gradient boosting on richer feature sets). Conclusion: a salary-trained model
  can't reward defense the market underpays, so the production model stayed lean.
- **`experiment_gbt.py`** - gradient-boosting comparison against the linear model.
  Conclusion: no meaningful gain over OLS, so OLS stayed in production.
- **`build_historical_dataset.py`** - builds the 2018-23 skater dataset (CapFriendly +
  Hockey-Reference) used for the abandoned pooled-history model and the cap
  regime-shift check.
- **`build_goalie_dataset.py`** - the goalie analog of the above.

To run any of these, invoke from the repo root (e.g. `python -m archive.experiment_gbt`)
so the `src.*` imports resolve.
