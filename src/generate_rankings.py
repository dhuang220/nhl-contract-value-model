"""Produce the residual-ranking CSVs the Streamlit app reads.

Ranks every current NHL contract by how far its cap hit sits above/below what
2025-26 performance predicts - a contemporaneous value model (all rows the same
season, sidestepping the flat-cap-era vs boom-era shift). Each row gets BOTH a
linear-regression prediction and a gradient-boosted-tree prediction, so the app
can toggle between them and compare side by side. All predictions come from
5-fold cross-validation, so the model never saw that player when predicting him.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.model import prepare, prepare_goalies, cross_validated_scores, FEATURES, GOALIE_FEATURES, TARGET

SKATER_IN = "data/processed/skaters_current_dataset.csv"
GOALIE_IN = "data/processed/goalies_current_dataset.csv"
SKATER_OUT = "data/processed/skaters_current_residual_ranking.csv"
GOALIE_OUT = "data/processed/goalies_current_residual_ranking.csv"

KF = KFold(n_splits=5, shuffle=True, random_state=0)
CEIL = 104_000_000
BASE_COLS = ["player_name", "team", "position", "age", "contract_type", "cap_hit"]


def gbt():
    """Tuned GBT from src/experiment_gbt.py - shallow/subsampled to limit overfit."""
    return GradientBoostingRegressor(
        learning_rate=0.02, max_depth=3, min_samples_leaf=10,
        n_estimators=400, subsample=0.7, random_state=0,
    )


def _cv_dollars(df, features, model):
    pred_log = cross_val_predict(model, df[features], df[TARGET], cv=KF)
    return np.exp(pred_log) * CEIL


def _combined_ranking(df, features):
    """One row per player with both models' predicted cap hit + residual."""
    out = df[[c for c in BASE_COLS if c in df.columns]].copy()
    if "position" not in out:
        out["position"] = "G"
    cap = df["cap_hit"].values
    for label, model in [("linear", make_pipeline(StandardScaler(), LinearRegression())), ("gbt", gbt())]:
        pred = _cv_dollars(df, features, model).round()
        out[f"predicted_{label}"] = pred
        out[f"residual_{label}"] = cap - pred
    return out.sort_values("residual_gbt", ascending=False).reset_index(drop=True)


def generate_skater_ranking():
    df = prepare(pd.read_csv(SKATER_IN))
    r = _combined_ranking(df, FEATURES)
    r.to_csv(SKATER_OUT, index=False)
    return df, r


def generate_goalie_ranking():
    df = prepare_goalies(pd.read_csv(GOALIE_IN))
    r = _combined_ranking(df, GOALIE_FEATURES)
    r.to_csv(GOALIE_OUT, index=False)
    return df, r


if __name__ == "__main__":
    sdf, sk = generate_skater_ranking()
    lin = cross_validated_scores(sdf, features=FEATURES)
    g = cross_validated_scores(sdf, model=gbt(), features=FEATURES)
    print(f"skaters: {len(sk)} ranked (both models) -> {SKATER_OUT}")
    print(f"  Linear CV R2={lin['r2_log']:.3f}  MAE=${lin['mae_dollars']:,.0f}")
    print(f"  GBT    CV R2={g['r2_log']:.3f}  MAE=${g['mae_dollars']:,.0f}")

    gdf, gg = generate_goalie_ranking()
    print(f"goalies: {len(gg)} ranked (both models) -> {GOALIE_OUT}")
