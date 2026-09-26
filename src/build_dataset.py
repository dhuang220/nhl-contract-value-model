import pandas as pd

from src.fetch_nhl_api import fetch_skater_summary, fetch_skater_bios
from src.match_names import match_contracts_to_bios
from src.features import fill_in_contract_types
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
            "position": bio["positionCode"],
            "age": contract["age"],
            "contract_type": contract["contract_type"],
            "term_years": contract["term_years"],
            "games_played": gp,
            "points": stats["points"],
            "points_per_60": stats["points"] / total_toi_hours,
            "toi_per_gp_min": toi_per_gp_sec / 60,
            "plus_minus": stats["plusMinus"],
            "cap_hit": contract["cap_hit"],
            "cap_hit_pct": contract["cap_hit"] / ceiling,
        })

    return pd.DataFrame(rows)
