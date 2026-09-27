import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold

FEATURES = [
    "points_per_60",
    "ixg_per_60",
    "toi_per_gp_min",
    "games_played",
    "age",
    "is_ufa",
    "is_defense",
]
# plus_minus dropped: team-dependent noise (it inflated MacKinnon's value on a
# dominant Colorado team past McDavid's); removing it improved MAE and fixed the
# ranking. Still computed in the datasets, just not used as a feature.
GOALIE_FEATURES = [
    "games_started",  # workload / clear #1 status - the dominant driver of goalie pay
    "age",
    "is_ufa",
]
# Quality metrics (save%/gaa/gsaa, and even shot-quality-adjusted GSAx from NST)
# were tested and dropped: goalie pay is driven by workload, not quality. GSAx
# added ~nothing (R2 0.437 -> 0.441) with a backwards coefficient - the market
# pays goalies for being the #1, not for GSAx. GSAx is still computed in the
# dataset (see fetch_nst) and available if the goal ever shifts to value vs pay.
TARGET = "log_cap_hit_pct"


LEAGUE_MIN = 775_000  # cap hits below this are buried/retained figures, not AAVs


def prepare(df: pd.DataFrame, min_gp: int = 20) -> pd.DataFrame:
    """Filter noisy low-GP rows and add encoded features + log target."""
    df = df[(df["games_played"] >= min_gp) & (df["cap_hit"] >= LEAGUE_MIN)].copy()
    df["is_ufa"] = (df["contract_type"] == "UFA").astype(int)
    df["is_defense"] = (df["position"] == "D").astype(int)
    df["log_cap_hit_pct"] = np.log(df["cap_hit_pct"])
    return df.reset_index(drop=True)


def prepare_goalies(df: pd.DataFrame, min_gp: int = 25) -> pd.DataFrame:
    """Goalie analog of prepare(): filter low-GP, impute GSAA, encode + log."""
    df = df[(df["games_played"] >= min_gp) & (df["cap_hit"] >= LEAGUE_MIN)].copy()
    df["is_ufa"] = (df["contract_type"] == "UFA").astype(int)
    df["log_cap_hit_pct"] = np.log(df["cap_hit_pct"])
    return df.reset_index(drop=True)


def cross_validated_scores(df: pd.DataFrame, model=None, features=FEATURES) -> dict:
    """Honest out-of-sample error via 5-fold CV, reported in $ AAV terms."""
    if model is None:
        model = make_pipeline(StandardScaler(), LinearRegression())
    X, y = df[features], df[TARGET]
    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    pred_log = cross_val_predict(model, X, y, cv=cv)
    # back-transform log(cap%) -> cap% -> dollars against the 2026-27 ceiling
    pred_pct, actual_pct = np.exp(pred_log), np.exp(y)
    ceiling = 104_000_000
    err_dollars = np.abs(pred_pct - actual_pct) * ceiling

    ss_res = ((y - pred_log) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {
        "r2_log": 1 - ss_res / ss_tot,
        "mae_dollars": err_dollars.mean(),
        "median_ae_dollars": np.median(err_dollars),
    }


def fit_and_rank(train_df, rank_df=None, model=None, features=FEATURES,
                 ceiling=104_000_000, cv=False):
    """Rank rank_df by residual (actual - predicted) cap hit.

    Positive residual = overpaid, negative = underpaid. `ceiling` converts the
    predicted cap% back to dollars. rank_df defaults to train_df.

    cv=False: in-sample fitted predictions (descriptive - the market the model
    was fit on). cv=True (only when ranking the training set itself): each row's
    prediction comes from 5-fold cross-validation, so the model never saw that
    player - a genuinely out-of-sample "was he fairly paid" estimate (noisier).
    """
    same = rank_df is None
    if rank_df is None:
        rank_df = train_df
    if model is None:
        model = make_pipeline(StandardScaler(), LinearRegression())

    if cv and same:
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        pred_log = cross_val_predict(model, train_df[features], train_df[TARGET], cv=kf)
    else:
        model.fit(train_df[features], train_df[TARGET])
        pred_log = model.predict(rank_df[features])

    pred_pct = np.exp(pred_log)
    cols = [c for c in ["player_name", "player_id", "team", "position", "age", "contract_type", "cap_hit"] if c in rank_df]
    out = rank_df[cols].copy()
    if "position" not in out:
        out["position"] = "G"
    out["predicted_cap_hit"] = (pred_pct * ceiling).round(0)
    out["residual"] = out["cap_hit"] - out["predicted_cap_hit"]
    return out.sort_values("residual", ascending=False).reset_index(drop=True)
