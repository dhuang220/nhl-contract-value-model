"""Produce the residual-ranking CSVs the Streamlit app reads.

Ranks contracts by how far the cap hit sits above/below what 2025-26 performance
predicts - a contemporaneous linear value model. Predictions come from 5-fold
cross-validation, so the model never saw that player when predicting him.
Two skater populations: all current contracts, and 2026 signings only
(fresh-market pricing). Residuals are descriptive, not out-of-sample forecasts.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.model import prepare, prepare_goalies, fit_and_rank, cross_validated_scores, FEATURES, GOALIE_FEATURES, TARGET

SKATER_IN = "data/processed/skaters_current_dataset.csv"
GOALIE_IN = "data/processed/goalies_current_dataset.csv"
SIGNINGS_INS = ["data/processed/skaters_2024signings_dataset.csv",
                "data/processed/skaters_2025signings_dataset.csv",
                "data/processed/skaters_2026signings_dataset.csv"]
SKATER_OUT = "data/processed/skaters_current_residual_ranking.csv"
GOALIE_OUT = "data/processed/goalies_current_residual_ranking.csv"
SIGNINGS_OUT = "data/processed/skaters_signings_residual_ranking.csv"


def _training_signings():
    """Combined 2025 + 2026 fresh-market signings (each on its own walk year)."""
    return pd.concat([prepare(pd.read_csv(p)) for p in SIGNINGS_INS], ignore_index=True)


def _add_segment_flags(r):
    """Flag each ranked player: signed this year (2026), and/or a pending free
    agent (current contract expires after 2026-27). Lets the app show subsets."""
    signed = set(pd.read_csv("data/raw/contracts/signings_2026.csv")["player_name"])
    fa = set(pd.read_csv("data/raw/contracts/pending_fa_2026.csv")["player_name"])
    r["just_signed"] = r["player_name"].isin(signed)
    r["pending_fa"] = r["player_name"].isin(fa)
    return r

KF = KFold(n_splits=5, shuffle=True, random_state=0)
CEIL = 104_000_000
BASE_COLS = ["player_name", "team", "position", "age", "contract_type", "cap_hit"]
# NB: predictions are intentionally NOT capped at the max-contract limit, so an
# elite player's modeled value can run above the cap - the amount over $20.8M
# shows how far his production outstrips what any contract can legally pay.


def _ranking(df, features):
    """One row per player with the linear model's predicted cap hit + residual."""
    out = df[[c for c in BASE_COLS if c in df.columns]].copy()
    if "position" not in out:
        out["position"] = "G"
    model = make_pipeline(StandardScaler(), LinearRegression())
    pred = (np.exp(cross_val_predict(model, df[features], df[TARGET], cv=KF)) * CEIL).round()
    out["predicted_cap_hit"] = pred
    out["residual"] = df["cap_hit"].values - pred
    return out.sort_values("residual", ascending=False).reset_index(drop=True)


def generate_skater_ranking():
    """Fit on 2024-2026 fresh-market signings, then value EVERY current skater.

    Training on fresh deals prices star/upside production properly (the whole-
    league fit under-valued it, diluted by cheap legacy/RFA/ELC contracts).
    Predictions for players not in the signings set are out-of-sample.
    """
    train = _training_signings()
    current = prepare(pd.read_csv(SKATER_IN))
    r = _add_segment_flags(fit_and_rank(train, current, features=FEATURES))
    r.to_csv(SKATER_OUT, index=False)
    return train, current, r


def generate_goalie_ranking():
    df = prepare_goalies(pd.read_csv(GOALIE_IN))
    r = _add_segment_flags(_ranking(df, GOALIE_FEATURES))
    r.to_csv(GOALIE_OUT, index=False)
    return df, r


def generate_signings_ranking():
    """The 2024-2026 signings themselves, ranked in-sample (out-of-sample CV)."""
    df = _training_signings()
    r = _ranking(df, FEATURES)
    r.to_csv(SIGNINGS_OUT, index=False)
    return df, r


if __name__ == "__main__":
    train, current, sk = generate_skater_ranking()
    s = cross_validated_scores(train, features=FEATURES)
    print(f"skaters: fit on {len(train)} 2024-2026 signings (CV R2={s['r2_log']:.3f}), "
          f"valued all {len(sk)} current -> {SKATER_OUT}")

    gdf, gg = generate_goalie_ranking()
    gs = cross_validated_scores(gdf, features=GOALIE_FEATURES)
    print(f"goalies: {len(gg)} -> {GOALIE_OUT}  CV R2={gs['r2_log']:.3f}")

    sdf2, ss = generate_signings_ranking()
    print(f"signings view: {len(ss)} -> {SIGNINGS_OUT}")
