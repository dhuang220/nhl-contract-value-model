"""Build a unified historical skater dataset (2018-19 .. 2022-23).

Joins CapFriendly contract data (UFA/RFA, cap hit, xGF%) to Hockey-Reference
box scores (GP, points, TOI, plus-minus) per season, using the CONTEMPORANEOUS
framing: a season's performance paired with the contract the player is on that
season. Output is schema-compatible with data/processed/skaters_2026_dataset.csv
so the two can be concatenated for modeling.
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

# columns matching the pilot, plus xgf_pct and season
OUTPUT_COLS = [
    "player_name", "position", "age", "contract_type", "term_years",
    "games_played", "points", "points_per_60", "toi_per_gp_min",
    "plus_minus", "cap_hit", "cap_hit_pct", "xgf_pct", "season",
]

_NUM_PREFIX = re.compile(r"^\d+\.\s*")


def _clean_name(raw: str) -> str:
    """'1. Connor McDavid' -> 'Connor McDavid'."""
    return _NUM_PREFIX.sub("", str(raw)).strip()


def _strip_accents(s: str) -> str:
    """'Jakub Voráček' -> 'Jakub Voracek' (CapFriendly stores names plain,
    Hockey-Reference keeps accents; normalize both sides before matching)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)
    )


def _load_capfriendly(season_dir: str) -> pd.DataFrame:
    yy = season_dir.replace("-", "")[2:4] + season_dir.replace("-", "")[6:8]
    path = f"data/raw/capfriendly_historical/{season_dir}/capfriendly{yy}.csv"
    cf = pd.read_csv(path, encoding="latin-1")
    # NB: CapFriendly's "SIGNING AGE" is age at contract signing, not age during
    # this season - we take current-season age from Hockey-Reference instead, so
    # the age feature reflects the aging curve at the time the stats were produced.
    cf = cf[["PLAYER", "SIGNING", "LENGTH", "CAP HIT", "xGF%"]].copy()
    cf.columns = ["player_name", "contract_type", "term_years", "cap_hit", "xgf_pct"]
    cf["player_name"] = cf["player_name"].map(_clean_name)
    cf["xgf_pct"] = pd.to_numeric(cf["xgf_pct"].replace("-", np.nan), errors="coerce")
    return cf


def _load_hockeyref(season_dir: str) -> pd.DataFrame:
    yyyy = season_dir[:4] if season_dir[:2] == "20" else season_dir.replace("-", "")[:4]
    # filenames use the season-end shorthand, e.g. hockeyref1819.csv
    yy = season_dir.replace("-", "")[2:4] + season_dir.replace("-", "")[6:8]
    path = f"data/raw/capfriendly_historical/{season_dir}/hockeyref{yy}.csv"
    hr = pd.read_csv(path, header=1)
    hr = hr[["Player", "Tm", "Age", "Pos", "GP", "PTS", "TOI", "+/-"]].copy()
    hr.columns = ["player_name", "team", "age", "position", "games_played", "points", "toi", "plus_minus"]
    # Drop individual team-split rows for traded players: keep the TOT aggregate.
    # (These HR exports already appear collapsed, but guard defensively.)
    has_tot = hr.groupby("player_name")["team"].transform(lambda s: (s == "TOT").any())
    hr = hr[(~has_tot) | (hr["team"] == "TOT")].copy()
    hr = hr.drop_duplicates(subset="player_name", keep="first")
    return hr


def _build_one_season(season_id: str) -> tuple[pd.DataFrame, dict]:
    season_dir = SEASONS[season_id]
    cf = _load_capfriendly(season_dir)
    hr = _load_hockeyref(season_dir)

    hr_records = hr.to_dict("records")
    # match on accent-stripped names; keep index back to the original record
    hr_norm_names = [_strip_accents(r["player_name"]) for r in hr_records]

    rows, unmatched = [], []
    for _, c in cf.iterrows():
        # WRatio is token-aware so nickname diffs (Mitchell->Mitch, Zachary->Zach)
        # still clear the bar, while goalies (absent from the skater file) land
        # far below it (~55-65) and are correctly dropped.
        match = process.extractOne(
            _strip_accents(c["player_name"]), hr_norm_names, scorer=fuzz.WRatio
        )
        _, score, idx = match
        if score < 85:
            unmatched.append(c["player_name"])
            continue
        h = hr_records[idx]
        toi = h["toi"]
        gp = h["games_played"]
        if not toi or toi <= 0 or not gp or gp <= 0:
            unmatched.append(c["player_name"])
            continue
        rows.append({
            "player_name": c["player_name"],
            "position": h["position"],
            "age": h["age"],
            "contract_type": c["contract_type"],
            "term_years": c["term_years"],
            "games_played": gp,
            "points": h["points"],
            "points_per_60": h["points"] * 60.0 / toi,
            "toi_per_gp_min": toi / gp,
            "plus_minus": h["plus_minus"],
            "cap_hit": c["cap_hit"],
            "cap_hit_pct": c["cap_hit"] / CAP_CEILING[season_id],
            "xgf_pct": c["xgf_pct"],
            "season": season_id,
        })

    stats = {
        "cf_rows": len(cf),
        "matched": len(rows),
        "unmatched": len(unmatched),
        "unmatched_sample": unmatched[:8],
    }
    return pd.DataFrame(rows, columns=OUTPUT_COLS), stats


def build_historical_dataset(save_path: str | None = None) -> pd.DataFrame:
    frames, report = [], {}
    for season_id in SEASONS:
        df, stats = _build_one_season(season_id)
        frames.append(df)
        report[season_id] = stats
    full = pd.concat(frames, ignore_index=True)
    # A few players appear twice in one season in CapFriendly when multiple
    # contracts touch that year (mid-season buyout + new signing, e.g. Evander
    # Kane 2021-22), and rare same-name/different-player collisions (the two
    # Sebastian Ahos) both match the same HR record. Both are tiny (~7 rows);
    # keep one row per player-season. Documented limitation of name-based joins.
    full = full.drop_duplicates(subset=["player_name", "season"], keep="first").reset_index(drop=True)
    if save_path:
        full.to_csv(save_path, index=False)
    build_historical_dataset.last_report = report
    return full


if __name__ == "__main__":
    df = build_historical_dataset("data/processed/skaters_historical_dataset.csv")
    rep = build_historical_dataset.last_report
    print("=== per-season match rates ===")
    for s, r in rep.items():
        rate = 100 * r["matched"] / r["cf_rows"]
        print(f"{s}: {r['matched']}/{r['cf_rows']} matched ({rate:.0f}%), "
              f"{r['unmatched']} unmatched e.g. {r['unmatched_sample'][:5]}")
    print(f"\nTOTAL rows: {len(df)}")
    print("contract_type:", df["contract_type"].value_counts().to_dict())
    print("position:", df["position"].value_counts().to_dict())
    print("\nunit sanity (describe):")
    print(df[["points_per_60", "cap_hit_pct", "toi_per_gp_min"]].describe().loc[["min", "50%", "max"]].round(3))
    print("\ntop cap_hit_pct in 2018-19:")
    top = df[df.season == "20182019"].nlargest(5, "cap_hit_pct")
    print(top[["player_name", "position", "cap_hit", "cap_hit_pct"]].to_string(index=False))
    print("\n=== head ===")
    print(df.head().to_string(index=False))
