import pandas as pd

from src.fetch_nhl_api import fetch_skater_summary, fetch_skater_bios
from src.match_names import match_contracts_to_bios
from src.features import fill_in_contract_types, pretty_team
from src.fetch_moneypuck import load_moneypuck_advanced
from src.cap_ceilings import CAP_CEILING

HISTORICAL_CSV = "data/processed/skaters_historical_dataset.csv"
PILOT_CSV = "data/processed/skaters_2026_dataset.csv"


def load_combined_dataset(
    historical_csv: str = HISTORICAL_CSV,
    pilot_csv: str = PILOT_CSV,
    pilot_season: str = "20262027",
) -> pd.DataFrame:
    """Pool historical (2018-23) and pilot (2026) skater rows into one frame.

    Uniform target rule across both: cap_hit_pct = cap_hit / CAP_CEILING[season].
    The pilot lacks xgf_pct (no 2025-26 MoneyPuck file yet) so that column is NaN
    for pilot rows; the pooled model uses plus_minus, which both sources share.
    The pilot's performance is its walk year (2025-26) while its season label is
    the contract season (2026-27) - the accepted contemporaneous-framing offset.
    """
    hist = pd.read_csv(historical_csv)
    hist["season"] = hist["season"].astype(str)
    pilot = pd.read_csv(pilot_csv)
    pilot["season"] = pilot_season
    pilot["xgf_pct"] = pd.NA
    return pd.concat([hist, pilot[hist.columns]], ignore_index=True)


def build_skater_dataset(
    contracts_csv: str,
    walk_year_season: str,
    contract_effective_season: str,
    moneypuck_csv: str | None = None,
) -> pd.DataFrame:
    """Assemble a training-ready skater dataset for one offseason.

    Joins cleaned contract rows to their walk-year (prior season) NHL stats,
    computes the model features, and normalizes cap hit against the ceiling of
    the season the contract takes effect in. Rows whose player can't be matched
    to walk-year stats are dropped (e.g. players who didn't play in the NHL that
    season - they have no performance signal to predict from).
    """
    contracts = pd.read_csv(contracts_csv)
    skaters = contracts[contracts["position"] != "G"].reset_index(drop=True)

    bios = fetch_skater_bios(walk_year_season)
    matches = match_contracts_to_bios(skaters["player_name"].tolist(), bios)
    skaters = fill_in_contract_types(skaters, matches)

    # playerId -> walk-year stat line, so we can attach performance by ID
    summary = {row["playerId"]: row for row in fetch_skater_summary(walk_year_season)}
    mp = load_moneypuck_advanced(moneypuck_csv) if moneypuck_csv else {}

    ceiling = CAP_CEILING[contract_effective_season]
    rows = []
    for _, contract in skaters.iterrows():
        bio = matches.get(contract["player_name"])
        if bio is None:
            continue
        stats = summary.get(bio["playerId"])
        if stats is None:
            continue

        gp = stats["gamesPlayed"]
        toi_per_gp_sec = stats["timeOnIcePerGame"]
        total_toi_hours = (toi_per_gp_sec * gp) / 3600
        if total_toi_hours == 0:
            continue

        rows.append({
            "player_name": contract["player_name"],
            "team": pretty_team(contract.get("team") or contract.get("team_signed")),
            "position": bio["positionCode"],
            "age": contract["age"],
            "contract_type": contract["contract_type"],
            "term_years": contract["term_years"],
            "games_played": gp,
            "points": stats["points"],
            "points_per_60": stats["points"] / total_toi_hours,
            "toi_per_gp_min": toi_per_gp_sec / 60,
            "plus_minus": stats["plusMinus"],
            "ixg_per_60": (m["ixg"] / total_toi_hours) if (m := mp.get(bio["playerId"])) else float("nan"),
            "xga_per_60": (m["xga_on"] / total_toi_hours) if m else float("nan"),
            "xgf_pct": m["xgf_pct"] if m else float("nan"),
            "cap_hit": contract["cap_hit"],
            "cap_hit_pct": contract["cap_hit"] / ceiling,
        })

    df = pd.DataFrame(rows)
    if moneypuck_csv:
        # a few players have NHL stats but no MoneyPuck row - neutral-impute
        for col in ["ixg_per_60", "xga_per_60", "xgf_pct"]:
            df[col] = df[col].fillna(df[col].median())
    return df


if __name__ == "__main__":
    # Build the full-league current skater dataset (walk year 2025-26), with
    # individual xG from the 2025-26 MoneyPuck file.
    df = build_skater_dataset(
        "data/raw/contracts/capwages_current_2026.csv",
        walk_year_season="20252026",
        contract_effective_season="20262027",
        moneypuck_csv="data/raw/stats/2026.csv",
    )
    df.to_csv("data/processed/skaters_current_dataset.csv", index=False)
    print(f"current skaters: {len(df)} rows -> data/processed/skaters_current_dataset.csv")

    # Fresh-market training set: 2025 and 2026 offseason signings (scraped per
    # season from the CapWages tracker), each paired with its own walk year.
    s26 = build_skater_dataset(
        "data/raw/contracts/signings_2026.csv", "20252026", "20262027", "data/raw/stats/2026.csv")
    s26.to_csv("data/processed/skaters_2026signings_dataset.csv", index=False)
    print(f"2026 signings: {len(s26)} rows -> data/processed/skaters_2026signings_dataset.csv")

    s25 = build_skater_dataset(
        "data/raw/contracts/signings_2025.csv", "20242025", "20252026", "data/raw/stats/2025.csv")
    s25.to_csv("data/processed/skaters_2025signings_dataset.csv", index=False)
    print(f"2025 signings: {len(s25)} rows -> data/processed/skaters_2025signings_dataset.csv")

    s24 = build_skater_dataset(
        "data/raw/contracts/signings_2024.csv", "20232024", "20242025", "data/raw/stats/2024.csv")
    s24.to_csv("data/processed/skaters_2024signings_dataset.csv", index=False)
    print(f"2024 signings: {len(s24)} rows -> data/processed/skaters_2024signings_dataset.csv")
