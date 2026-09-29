"""Feature-and-model comparison for the skater value model.

Builds an enriched 2025-26 feature set from the NHL API (summary + realtime:
PIM, shots, shooting%, special teams, faceoffs, hits, blocks, takeaways,
giveaways, individual Corsi) and optionally MoneyPuck 2024-25 on-ice advanced
stats (Corsi%, Fenwick%, xGF%, danger-adjusted xG), then compares model classes
via 5-fold CV. Exploratory - does not change the production pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold

from src.fetch.nhl_api import fetch_skater_summary, fetch_skater_bios
from src.match_names import match_contracts_to_bios
from src.features import infer_contract_type

WALK, CEIL = "20252026", 104_000_000
KF = KFold(n_splits=5, shuffle=True, random_state=0)


def _realtime(season):
    import requests
    out, start = {}, 0
    while True:
        r = requests.get("https://api.nhle.com/stats/rest/en/skater/realtime", params={
            "limit": 100, "start": start, "sort": "playerId",
            "cayenneExp": f"seasonId={season} and gameTypeId=2"})
        page = r.json()["data"]
        if not page:
            break
        for row in page:
            out[row["playerId"]] = row
        start += 100
    return out


def build():
    contracts = pd.read_csv("data/raw/contracts/capwages_current_2026.csv")
    sk = contracts[contracts["position"] != "G"].reset_index(drop=True)
    bios = fetch_skater_bios(WALK)
    matches = match_contracts_to_bios(sk["player_name"].tolist(), bios)
    summ = {r["playerId"]: r for r in fetch_skater_summary(WALK)}
    real = _realtime(WALK)

    mp = pd.read_csv("data/raw/stats/2025.csv")  # 2024-25 MoneyPuck (season lag)
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
        per60 = lambda v: (v or 0) / hrs
        debut = int(str(b["firstSeasonForGameType"])[:4])
        row = {
            "cap_hit": c["cap_hit"],
            "log_cap_hit_pct": np.log(c["cap_hit"] / CEIL),
            # base 7
            "points_per_60": per60(s["points"]), "toi_per_gp_min": s["timeOnIcePerGame"] / 60,
            "games_played": gp, "plus_minus": s["plusMinus"], "age": int(c["age"]),
            "is_ufa": int(infer_contract_type(int(c["age"]), debut, 2026) == "UFA"),
            "is_defense": int(b["positionCode"] == "D"),
            # NHL extras
            "pim_per_60": per60(s["penaltyMinutes"]), "shots_per_60": per60(s["shots"]),
            "shooting_pct": s["shootingPct"] or 0, "pp_pts_per_60": per60(s["ppPoints"]),
            "sh_pts_per_60": per60(s["shPoints"]), "gwg": s["gameWinningGoals"],
            "faceoff_pct": s["faceoffWinPct"] if s["faceoffWinPct"] is not None else 0.5,
            "hits_per_60": per60(rt["hits"]), "blocks_per_60": per60(rt["blockedShots"]),
            "takeaways_per_60": per60(rt["takeaways"]), "giveaways_per_60": per60(rt["giveaways"]),
            "corsi_per_60": per60(rt["totalShotAttempts"]),
        }
        if pid in mp.index:
            m = mp.loc[pid]
            row.update({"mp_corsi_pct": m["onIce_corsiPercentage"], "mp_fenwick_pct": m["onIce_fenwickPercentage"],
                        "mp_xgf_pct": m["onIce_xGoalsPercentage"], "mp_ixg_per60": (m["I_F_xGoals"]) / hrs,
                        "mp_highdanger_per60": m["I_F_highDangerShots"] / hrs})
        rows.append(row)
    return pd.DataFrame(rows)


BASE = ["points_per_60", "toi_per_gp_min", "games_played", "plus_minus", "age", "is_ufa", "is_defense"]
NHL_EXTRA = BASE + ["pim_per_60", "shots_per_60", "shooting_pct", "pp_pts_per_60", "sh_pts_per_60",
                    "gwg", "faceoff_pct", "hits_per_60", "blocks_per_60", "takeaways_per_60",
                    "giveaways_per_60", "corsi_per_60"]
MP = ["mp_corsi_pct", "mp_fenwick_pct", "mp_xgf_pct", "mp_ixg_per60", "mp_highdanger_per60"]


def cv_r2(df, feats, model):
    d = df.dropna(subset=feats + ["log_cap_hit_pct"])
    pred = cross_val_predict(model, d[feats], d["log_cap_hit_pct"], cv=KF)
    y = d["log_cap_hit_pct"].values
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mae = (np.abs(np.exp(pred) - np.exp(y)) * CEIL).mean()
    return r2, mae, len(d)


if __name__ == "__main__":
    df = build()
    print(f"rows: {len(df)}  (with MoneyPuck 2024-25 match: {df['mp_xgf_pct'].notna().sum()})\n")
    ols = lambda: make_pipeline(StandardScaler(), LinearRegression())
    lasso = lambda: make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=50000, random_state=0))
    gbt = lambda: GradientBoostingRegressor(random_state=0)

    print(f"{'model / features':40} {'CV R2':>7} {'MAE':>12} {'n':>5}")
    for name, feats, m in [
        ("OLS  / base 7", BASE, ols()),
        ("OLS  / + NHL extras", NHL_EXTRA, ols()),
        ("Lasso/ + NHL extras (feat-select)", NHL_EXTRA, lasso()),
        ("GBT  / + NHL extras", NHL_EXTRA, gbt()),
        ("OLS  / base + MoneyPuck", BASE + MP, ols()),
        ("Lasso/ NHL extras + MoneyPuck", NHL_EXTRA + MP, lasso()),
        ("GBT  / NHL extras + MoneyPuck", NHL_EXTRA + MP, gbt()),
    ]:
        r2, mae, n = cv_r2(df, feats, m)
        print(f"{name:40} {r2:>7.3f} {mae:>12,.0f} {n:>5}")

    # which features does Lasso keep / GBT rank?
    d = df.dropna(subset=NHL_EXTRA + MP + ["log_cap_hit_pct"])
    lc = make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=50000, random_state=0)).fit(d[NHL_EXTRA + MP], d["log_cap_hit_pct"])
    coefs = lc.named_steps["lassocv"].coef_
    kept = sorted([(abs(c), f, c) for c, f in zip(coefs, NHL_EXTRA + MP) if abs(c) > 1e-4], reverse=True)
    print("\nLasso-selected features (standardized coef, largest first):")
    for a, f, c in kept:
        print(f"  {f:22} {c:+.3f}")
    dropped = [f for c, f in zip(coefs, NHL_EXTRA + MP) if abs(c) <= 1e-4]
    print("  dropped to zero:", dropped)
