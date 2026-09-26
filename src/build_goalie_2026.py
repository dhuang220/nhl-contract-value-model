"""Build the 2026-offseason goalie dataset (walk year = 2025-26).

Joins the 2026 goalie contracts (from the cleaned CapWages paste) to their
2025-26 Hockey-Reference goalie stats, computes UFA/RFA from NHL API goalie
bios, and normalizes cap hit against the 2026-27 ceiling. Schema matches
data/processed/goalies_historical_dataset.csv (gsax60 is NaN - no CapFriendly
after 2024) so the two can be pooled for modeling.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from src.fetch_hockeyref import fetch_goalie_stats_page, parse_goalie_stats
from src.fetch_nhl_api import fetch_goalie_bios
from src.features import infer_contract_type
from src.cap_ceilings import CAP_CEILING

CONTRACTS_CSV = "data/raw/contracts/capwages_2026_cleaned.csv"


def _num(x):
    return pd.to_numeric(str(x).strip().replace("%", "") or np.nan, errors="coerce")


def _best(name: str, choices: list[str], threshold: int = 85):
    matched, score, idx = process.extractOne(name, choices, scorer=fuzz.WRatio)
    return idx if score >= threshold else None


def build_goalie_2026(save_path: str | None = None) -> pd.DataFrame:
    contracts = pd.read_csv(CONTRACTS_CSV)
    goalies = contracts[contracts["position"] == "G"].reset_index(drop=True)

    stats = parse_goalie_stats(fetch_goalie_stats_page(2026))  # 2025-26 walk year
    stat_names = [s["name_display"] for s in stats]

    bios = fetch_goalie_bios("20252026")
    bio_names = [b["goalieFullName"] for b in bios]

    rows = []
    for _, g in goalies.iterrows():
        si = _best(g["player_name"], stat_names)
        if si is None:
            continue  # didn't play in the NHL in 2025-26 -> no walk-year signal
        s = stats[si]
        gp = _num(s["goalie_games"])
        if not gp or gp <= 0:
            continue

        bi = _best(g["player_name"], bio_names)
        if bi is not None:
            debut = int(str(bios[bi]["firstSeasonForGameType"])[:4])
            ctype = infer_contract_type(int(g["age"]), debut, int(g["offseason_year"]))
        elif int(g["age"]) >= 27:
            ctype = "UFA"  # age criterion alone settles it; no debut year needed
        else:
            ctype = np.nan

        rows.append({
            "player_name": g["player_name"],
            "age": _num(s["age"]),
            "contract_type": ctype,
            "term_years": g["term_years"],
            "games_played": gp,
            "games_started": _num(s["goalie_starts"]),
            "save_pct": _num(s["save_pct_goalie"]),
            "gaa": _num(s["goals_against_avg"]),
            "gsaa": _num(s["gs_above_avg"]),
            "gsax60": np.nan,  # CapFriendly-only; gone after 2024
            "cap_hit": g["cap_hit"],
            "cap_hit_pct": g["cap_hit"] / CAP_CEILING["20262027"],
            "season": "20262027",
        })

    df = pd.DataFrame(rows)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


if __name__ == "__main__":
    df = build_goalie_2026("data/processed/goalies_2026_dataset.csv")
    print("2026 goalie rows:", len(df))
    print("contract_type:", df["contract_type"].value_counts(dropna=False).to_dict())
    print(df[["player_name", "age", "contract_type", "games_played", "save_pct", "gaa", "cap_hit", "cap_hit_pct"]].to_string(index=False))
