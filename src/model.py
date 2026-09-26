import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold

FEATURES = [
    "points_per_60",
    "toi_per_gp_min",
    "games_played",
    "plus_minus",
    "age",
    "is_ufa",
    "is_defense",
]
TARGET = "log_cap_hit_pct"


def prepare(df: pd.DataFrame, min_gp: int = 20) -> pd.DataFrame:
    """Filter noisy low-GP rows and add encoded features + log target."""
    df = df[df["games_played"] >= min_gp].copy()
    df["is_ufa"] = (df["contract_type"] == "UFA").astype(int)
    df["is_defense"] = (df["position"] == "D").astype(int)
    df["log_cap_hit_pct"] = np.log(df["cap_hit_pct"])
    return df.reset_index(drop=True)


def cross_validated_scores(df: pd.DataFrame, model=None) -> dict:
    """Honest out-of-sample error via 5-fold CV, reported in $ AAV terms."""
    model = model or make_pipeline(StandardScaler(), LinearRegression())
    X, y = df[FEATURES], df[TARGET]
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


def fit_and_rank(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Fit on all rows and rank by residual (actual - predicted) cap-hit %.

    In-sample residuals: descriptive of this class's deals relative to the
    fitted market, not an out-of-sample prediction. Positive residual =
    paid more than the model expects (overpaid); negative = underpaid.
    """
    model = model or make_pipeline(StandardScaler(), LinearRegression())
    X, y = df[FEATURES], df[TARGET]
    model.fit(X, y)

    pred_pct = np.exp(model.predict(X))
    out = df[["player_name", "position", "age", "contract_type", "cap_hit"]].copy()
    out["predicted_cap_hit"] = (pred_pct * 104_000_000).round(0)
    out["residual"] = out["cap_hit"] - out["predicted_cap_hit"]
    return out.sort_values("residual", ascending=False).reset_index(drop=True)
