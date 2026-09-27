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

KF = KFold(n_splits=5, shuffle=True, random_state=0)
CEIL = 104_000_000
BASE_COLS = ["player_name", "team", "position", "age", "contract_type", "cap_hit"]
DEF_STRENGTH = 0.15  # how hard the defensive overlay pushes (0 = off)
# NB: predictions are intentionally NOT capped at the max-contract limit.


def _defensive_adjust(ranking, df):
    """Explicit defensive overlay: multiply predicted cap hit by exp(k * z), where
    z is on-ice xGF% standardized WITHIN position. Boosts strong-defense skaters,
    docks weak ones - beyond what the salary market pays. A deliberate, subjective
    adjustment on top of the market model; the input metric is deployment-biased.
    """
    d = df[["player_name", "position", "xgf_pct"]].copy()
    d["is_d"] = (d["position"] == "D").astype(int)
    d["z"] = d.groupby("is_d")["xgf_pct"].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    zmap = dict(zip(d["player_name"], d["z"]))
    r = ranking.copy()
    factor = r["player_name"].map(zmap).fillna(0.0).apply(lambda z: np.exp(DEF_STRENGTH * z))
    r["predicted_cap_hit"] = (r["predicted_cap_hit"] * factor).round()
    r["residual"] = r["cap_hit"] - r["predicted_cap_hit"]
    return r.sort_values("residual", ascending=False).reset_index(drop=True)


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
    r = _defensive_adjust(fit_and_rank(train, current, features=FEATURES), current)
    r.to_csv(SKATER_OUT, index=False)
    return train, current, r


def generate_goalie_ranking():
    df = prepare_goalies(pd.read_csv(GOALIE_IN))
    r = _ranking(df, GOALIE_FEATURES)
    r.to_csv(GOALIE_OUT, index=False)
    return df, r


def generate_signings_ranking():
    """The 2024-2026 signings themselves, ranked in-sample (out-of-sample CV)."""
    df = _training_signings()
    r = _defensive_adjust(_ranking(df, FEATURES), df)
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
