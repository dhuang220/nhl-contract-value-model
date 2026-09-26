"""Produce the residual-ranking CSVs the Streamlit app reads.

Ranks EVERY current NHL contract (skaters and goalies) by how far its cap hit
sits above/below what 2025-26 performance predicts - a contemporaneous value
model (all rows the same season, which sidesteps the flat-cap-era vs boom-era
distribution shift). Each player's predicted cap hit comes from 5-fold
cross-validation (fit_and_rank cv=True), so the model never saw that player when
predicting him - a genuinely out-of-sample "was he fairly paid" estimate.

Reads the processed datasets built by src/build_dataset.py (skaters) and
src/build_goalie_2026.py (goalies).
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.model import (
    prepare, prepare_goalies, fit_and_rank, cross_validated_scores,
    FEATURES, GOALIE_FEATURES,
)

# Tuned gradient-boosted-tree config from src/experiment_gbt.py (shallow,
# subsampled - guarded against overfitting on ~650 rows). Predictions come from
# 5-fold CV (fit_and_rank cv=True), which keeps them honestly out-of-sample.
def gbt():
    return GradientBoostingRegressor(
        learning_rate=0.02, max_depth=3, min_samples_leaf=10,
        n_estimators=400, subsample=0.7, random_state=0,
    )

SKATER_IN = "data/processed/skaters_current_dataset.csv"
GOALIE_IN = "data/processed/goalies_current_dataset.csv"
SKATER_OUT = "data/processed/skaters_current_residual_ranking.csv"
GOALIE_OUT = "data/processed/goalies_current_residual_ranking.csv"


def generate_skater_ranking() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = prepare(pd.read_csv(SKATER_IN))
    ranking = fit_and_rank(df, features=FEATURES, cv=True, model=gbt())
    ranking.to_csv(SKATER_OUT, index=False)
    return df, ranking


def generate_goalie_ranking() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = prepare_goalies(pd.read_csv(GOALIE_IN))
    ranking = fit_and_rank(df, features=GOALIE_FEATURES, cv=True)
    ranking.to_csv(GOALIE_OUT, index=False)
    return df, ranking


if __name__ == "__main__":
    sdf, sk = generate_skater_ranking()
    s = cross_validated_scores(sdf, model=gbt(), features=FEATURES)
    print(f"skaters (GBT + xG): {len(sk)} ranked -> {SKATER_OUT}")
    print(f"  CV R2(log)={s['r2_log']:.3f}  MAE=${s['mae_dollars']:,.0f}")

    gdf, g = generate_goalie_ranking()
    gs = cross_validated_scores(gdf, features=GOALIE_FEATURES)
    print(f"goalies: {len(g)} ranked -> {GOALIE_OUT}")
    print(f"  CV R2(log)={gs['r2_log']:.3f}  MAE=${gs['mae_dollars']:,.0f}")

    fmt = lambda d: d.assign(**{c: (d[c] / 1e6).round(1) for c in ["cap_hit", "predicted_cap_hit", "residual"]})
    print("\nMost overpaid skaters ($M):")
    print(fmt(sk.head(6))[["player_name", "position", "cap_hit", "predicted_cap_hit", "residual"]].to_string(index=False))
    print("\nMost underpaid skaters ($M):")
    print(fmt(sk.tail(6).iloc[::-1])[["player_name", "position", "cap_hit", "predicted_cap_hit", "residual"]].to_string(index=False))
