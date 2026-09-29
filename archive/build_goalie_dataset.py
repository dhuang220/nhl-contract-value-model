"""Build a unified historical goalie dataset (2018-19 .. 2022-23).

Goalie parallel of build_historical_dataset.py. Drives from the Hockey-Reference
goalie box scores (SV%, GAA, GSAA, games started) and joins CapFriendly for
contract info (UFA/RFA, cap hit, term) plus GSAx/60 (goals saved above expected),
using the CONTEMPORANEOUS framing. Goalies use a different feature set than
skaters, so this is a separate dataset/model.
"""
import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from src.cap_ceilings import CAP_CEILING

SEASONS = {
    "20182019": "2018-2019",
    "20192020": "2019-2020",
    "20202021": "2020-2021",
    "20212022": "2021-2022",
    "20222023": "2022-2023",
}

OUTPUT_COLS = [
    "player_name", "age", "contract_type", "term_years",
    "games_played", "games_started", "save_pct", "gaa", "gsaa", "gsax60",
    "cap_hit", "cap_hit_pct", "season",
]

_NUM_PREFIX = re.compile(r"^\d+\.\s*")


def _clean_name(raw: str) -> str:
    """'3. Carey Price' -> 'Carey Price'."""
    return _NUM_PREFIX.sub("", str(raw)).strip()


def _strip_accents(s: str) -> str:
    """Normalize accents so CapFriendly (plain) and HR (accented) names match."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)
    )


def _load_capfriendly(season_dir: str) -> pd.DataFrame:
    yy = season_dir.replace("-", "")[2:4] + season_dir.replace("-", "")[6:8]
    path = f"data/raw/capfriendly_historical/{season_dir}/capfriendly{yy}.csv"
    cf = pd.read_csv(path, encoding="latin-1")
    cf = cf[["PLAYER", "SIGNING", "LENGTH", "CAP HIT", "GSAx60"]].copy()
    cf.columns = ["player_name", "contract_type", "term_years", "cap_hit", "gsax60"]
    cf["player_name"] = cf["player_name"].map(_clean_name)
    cf["gsax60"] = pd.to_numeric(cf["gsax60"].replace("-", np.nan), errors="coerce")
    # Restrict to goalies (goalie-only metric GSAx60 populated) so HR goalies
    # can't false-match to a similarly-named skater and inherit their contract.
    cf = cf[cf["gsax60"].notna()].reset_index(drop=True)
    return cf


def _load_hockeyref_goalie(season_dir: str) -> pd.DataFrame:
    yy = season_dir.replace("-", "")[2:4] + season_dir.replace("-", "")[6:8]
    path = f"data/raw/capfriendly_historical/{season_dir}/hockeyref_goalie{yy}.csv"
    hr = pd.read_csv(path, header=1)
    hr = hr[["Player", "Tm", "Age", "GP", "GS", "SV%", "GAA", "GSAA"]].copy()
    hr.columns = ["player_name", "team", "age", "games_played",
                  "games_started", "save_pct", "gaa", "gsaa"]
    # Traded goalies: keep the TOT aggregate (HR goalie files already collapse
    # team-splits into TOT, but guard defensively as the skater builder does).
    has_tot = hr.groupby("player_name")["team"].transform(lambda s: (s == "TOT").any())
    hr = hr[(~has_tot) | (hr["team"] == "TOT")].copy()
    hr = hr.drop_duplicates(subset="player_name", keep="first")
    return hr


def _build_one_season(season_id: str) -> tuple[pd.DataFrame, dict]:
    season_dir = SEASONS[season_id]
    cf = _load_capfriendly(season_dir)
    hr = _load_hockeyref_goalie(season_dir)

    cf_records = cf.to_dict("records")
    cf_norm_names = [_strip_accents(r["player_name"]) for r in cf_records]

    rows, unmatched = [], []
    for _, g in hr.iterrows():
        gp = g["games_played"]
        if not gp or gp <= 0:
            continue
        # drive from HR goalies, match to CapFriendly for the contract
        match = process.extractOne(
            _strip_accents(g["player_name"]), cf_norm_names, scorer=fuzz.WRatio
        )
        _, score, idx = match
        if score < 85:
            unmatched.append(g["player_name"])
            continue
        c = cf_records[idx]
        rows.append({
            "player_name": g["player_name"],
            "age": g["age"],
            "contract_type": c["contract_type"],
            "term_years": c["term_years"],
            "games_played": gp,
            "games_started": g["games_started"],
            "save_pct": g["save_pct"],
            "gaa": g["gaa"],
            "gsaa": g["gsaa"],
            "gsax60": c["gsax60"],
            "cap_hit": c["cap_hit"],
            "cap_hit_pct": c["cap_hit"] / CAP_CEILING[season_id],
            "season": season_id,
        })

    stats = {
        "hr_rows": len(hr),
        "matched": len(rows),
        "unmatched": len(unmatched),
        "unmatched_sample": unmatched[:8],
    }
    return pd.DataFrame(rows, columns=OUTPUT_COLS), stats


def build_goalie_dataset(save_path: str | None = None) -> pd.DataFrame:
    frames, report = [], {}
    for season_id in SEASONS:
        df, stats = _build_one_season(season_id)
        frames.append(df)
        report[season_id] = stats
    full = pd.concat(frames, ignore_index=True)
    # one row per goalie-season (multi-contract years / same-name collisions)
    full = full.drop_duplicates(subset=["player_name", "season"], keep="first").reset_index(drop=True)
    if save_path:
        full.to_csv(save_path, index=False)
    build_goalie_dataset.last_report = report
    return full


if __name__ == "__main__":
    df = build_goalie_dataset("data/processed/goalies_historical_dataset.csv")
    rep = build_goalie_dataset.last_report
    print("=== per-season match rates (HR goalies matched to a CapFriendly contract) ===")
    for s, r in rep.items():
        rate = 100 * r["matched"] / r["hr_rows"]
        print(f"{s}: {r['matched']}/{r['hr_rows']} matched ({rate:.0f}%), "
              f"{r['unmatched']} unmatched e.g. {r['unmatched_sample'][:5]}")
    print(f"\nTOTAL rows: {len(df)}")
    print("contract_type:", df["contract_type"].value_counts().to_dict())
    print("dup player-seasons:", int((df.groupby(['player_name','season']).size() > 1).sum()))
    print("max GP:", df["games_played"].max())
    print("\nunit sanity (min/median/max):")
    print(df[["save_pct", "gaa", "cap_hit_pct", "gsax60"]].describe().loc[["min", "50%", "max"]].round(3))
    print("\ntop cap_hit_pct in 2018-19:")
    top = df[df.season == "20182019"].nlargest(5, "cap_hit_pct")
    print(top[["player_name", "cap_hit", "cap_hit_pct", "save_pct"]].to_string(index=False))
    print("\ngoalie-seasons surviving GP>=25:", int((df["games_played"] >= 25).sum()), "of", len(df))
    print("\n=== head ===")
    print(df.head().to_string(index=False))
