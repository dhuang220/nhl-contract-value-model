"""Deeper evaluation of gradient-boosted trees for the skater value model.

Exploratory only - does not touch the production linear model. Reuses the
enriched dataset and feature groups from experiment_features.py. MoneyPuck
features are the 2024-25 lagged proxy (same caveat as that module).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, GridSearchCV, cross_val_score

from src.experiment_features import (
    BASE, NHL_EXTRA, MP, KF, CEIL, WALK, _realtime,
)
from src.fetch_nhl_api import fetch_skater_summary, fetch_skater_bios
from src.match_names import match_contracts_to_bios
from src.features import infer_contract_type


def build_with_names() -> pd.DataFrame:
    """Same rows/features as experiment_features.build() plus player_name/team."""
    contracts = pd.read_csv("data/raw/contracts/capwages_current_2026.csv")
    sk = contracts[contracts["position"] != "G"].reset_index(drop=True)
    bios = fetch_skater_bios(WALK)
    matches = match_contracts_to_bios(sk["player_name"].tolist(), bios)
    summ = {r["playerId"]: r for r in fetch_skater_summary(WALK)}
    real = _realtime(WALK)
    mp = pd.read_csv("data/raw/stats/2025.csv")
    mp = mp[mp["situation"] == "all"].set_index("playerId")

    rows = []
    for _, c in sk.iterrows():
        b = matches.get(c["player_name"])
        if not b:
            continue
        pid = b["playerId"]
        s, rt = summ.get(pid), real.get(pid)
        if not s or not rt:
            continue
        gp = s["gamesPlayed"]
        hrs = s["timeOnIcePerGame"] * gp / 3600
        if hrs == 0 or c["cap_hit"] < 775_000 or gp < 20:
            continue
        p60 = lambda v: (v or 0) / hrs
        debut = int(str(b["firstSeasonForGameType"])[:4])
        row = {
            "player_name": c["player_name"], "cap_hit": c["cap_hit"],
            "log_cap_hit_pct": np.log(c["cap_hit"] / CEIL),
            "points_per_60": p60(s["points"]), "toi_per_gp_min": s["timeOnIcePerGame"] / 60,
            "games_played": gp, "plus_minus": s["plusMinus"], "age": int(c["age"]),
            "is_ufa": int(infer_contract_type(int(c["age"]), debut, 2026) == "UFA"),
            "is_defense": int(b["positionCode"] == "D"),
            "pim_per_60": p60(s["penaltyMinutes"]), "shots_per_60": p60(s["shots"]),
            "shooting_pct": s["shootingPct"] or 0, "pp_pts_per_60": p60(s["ppPoints"]),
            "sh_pts_per_60": p60(s["shPoints"]), "gwg": s["gameWinningGoals"],
            "faceoff_pct": s["faceoffWinPct"] if s["faceoffWinPct"] is not None else 0.5,
            "hits_per_60": p60(rt["hits"]), "blocks_per_60": p60(rt["blockedShots"]),
            "takeaways_per_60": p60(rt["takeaways"]), "giveaways_per_60": p60(rt["giveaways"]),
            "corsi_per_60": p60(rt["totalShotAttempts"]),
        }
        if pid in mp.index:
            m = mp.loc[pid]
            row.update({"mp_corsi_pct": m["onIce_corsiPercentage"], "mp_fenwick_pct": m["onIce_fenwickPercentage"],
                        "mp_xgf_pct": m["onIce_xGoalsPercentage"], "mp_ixg_per60": m["I_F_xGoals"] / hrs,
                        "mp_highdanger_per60": m["I_F_highDangerShots"] / hrs})
        rows.append(row)
    return pd.DataFrame(rows)


def _metrics(y, pred_log):
    r2 = 1 - ((y - pred_log) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mae = (np.abs(np.exp(pred_log) - np.exp(y)) * CEIL).mean()
    return r2, mae


def _rank(df, feats, pred_log):
    pred_pct = np.exp(pred_log)
    out = df[["player_name", "cap_hit"]].copy()
    out["pred"] = (pred_pct * CEIL).round(0)
    out["residual"] = out["cap_hit"] - out["pred"]
    return out.sort_values("residual", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    df = build_with_names()
    y = df["log_cap_hit_pct"].values
    print(f"rows: {len(df)}  (MoneyPuck-matched: {df['mp_xgf_pct'].notna().sum()})")

    # ---- 1. tune GBT (guard overfit: shallow, subsampled) on NHL_EXTRA ----
    grid = {
        "learning_rate": [0.02, 0.05, 0.1],
        "n_estimators": [200, 400, 600],
        "max_depth": [2, 3],
        "min_samples_leaf": [10, 20, 30],
        "subsample": [0.7, 0.9],
    }
    gs = GridSearchCV(GradientBoostingRegressor(random_state=0), grid,
                      scoring="r2", cv=KF, n_jobs=-1)
    gs.fit(df[NHL_EXTRA], y)
    best = gs.best_params_
    print("\n[1] best GBT config:", best)
    tuned = GradientBoostingRegressor(random_state=0, **best)
    pred_cv = cross_val_predict(tuned, df[NHL_EXTRA], y, cv=KF)
    r2_cv, mae_cv = _metrics(y, pred_cv)
    print(f"    tuned GBT (NHL_EXTRA): CV R2={r2_cv:.3f}  MAE=${mae_cv:,.0f}")

    # ---- 2. overfitting check: train R2 vs CV R2 ----
    tuned.fit(df[NHL_EXTRA], y)
    r2_train, _ = _metrics(y, tuned.predict(df[NHL_EXTRA]))
    print(f"\n[2] overfit check: train R2={r2_train:.3f}  CV R2={r2_cv:.3f}  gap={r2_train - r2_cv:.3f}")

    # HistGBR (handles NaN -> can use full MP set)
    hist = HistGradientBoostingRegressor(random_state=0, max_depth=3,
                                         learning_rate=0.05, max_iter=400,
                                         min_samples_leaf=20, l2_regularization=1.0)
    r2_h, mae_h = _metrics(y, cross_val_predict(hist, df[NHL_EXTRA + MP], y, cv=KF))
    print(f"    HistGBR (NHL_EXTRA+MP, native-NaN): CV R2={r2_h:.3f}  MAE=${mae_h:,.0f}")

    # ---- 3. head-to-head vs linear on same feature sets ----
    print("\n[3] head-to-head (5-fold CV):")
    lin = lambda: make_pipeline(StandardScaler(), LinearRegression())
    for label, feats in [("NHL_EXTRA", NHL_EXTRA), ("NHL_EXTRA+MP", NHL_EXTRA + MP)]:
        d = df.dropna(subset=feats)
        yy = d["log_cap_hit_pct"].values
        rl, ml = _metrics(yy, cross_val_predict(lin(), d[feats], yy, cv=KF))
        rg, mg = _metrics(yy, cross_val_predict(GradientBoostingRegressor(random_state=0, **best), d[feats], yy, cv=KF))
        print(f"    {label:14} linear R2={rl:.3f} (${ml:,.0f})   GBT R2={rg:.3f} (${mg:,.0f})   n={len(d)}")

    # ---- 4. SHAP interpretability ----
    import shap
    tuned.fit(df[NHL_EXTRA], y)
    expl = shap.TreeExplainer(tuned)
    sv = expl.shap_values(df[NHL_EXTRA])
    imp = sorted(zip(NHL_EXTRA, np.abs(sv).mean(0)), key=lambda t: -t[1])
    print("\n[4] SHAP global importance (mean |SHAP|, top 8):")
    for f, v in imp[:8]:
        print(f"    {f:20} {v:.4f}")
    for name in ["Connor McDavid", "Quinn Hughes", "Leo Carlsson"]:
        idx = df.index[df["player_name"] == name]
        if len(idx):
            i = idx[0]
            contrib = sorted(zip(NHL_EXTRA, sv[i]), key=lambda t: -abs(t[1]))[:3]
            print(f"    {name}: base {float(np.ravel(expl.expected_value)[0]):.2f} + " +
                  ", ".join(f"{f} {c:+.2f}" for f, c in contrib) + f"  (pred {tuned.predict(df[NHL_EXTRA].iloc[[i]])[0]:.2f})")

    # ---- 5. ranking stability: GBT vs linear top-10 ----
    lin_rank = _rank(df, NHL_EXTRA, cross_val_predict(lin(), df[NHL_EXTRA], y, cv=KF))
    gbt_rank = _rank(df, NHL_EXTRA, pred_cv)
    def names(r, tail=False):
        return list((r.tail(10)[::-1] if tail else r.head(10))["player_name"])
    lo, go = names(lin_rank), names(gbt_rank)
    lu, gu = names(lin_rank, True), names(gbt_rank, True)
    print("\n[5] ranking stability (CV-predicted residuals, NHL_EXTRA):")
    print(f"    overpaid  overlap {len(set(lo)&set(go))}/10; GBT-only: {sorted(set(go)-set(lo))}")
    print(f"    underpaid overlap {len(set(lu)&set(gu))}/10; GBT-only: {sorted(set(gu)-set(lu))}")
    from scipy.stats import spearmanr
    m = lin_rank.merge(gbt_rank, on="player_name", suffixes=("_lin", "_gbt"))
    print(f"    Spearman rank corr of residuals (linear vs GBT): {spearmanr(m.residual_lin, m.residual_gbt).correlation:.3f}")
