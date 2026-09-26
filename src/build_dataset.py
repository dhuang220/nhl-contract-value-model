import pandas as pd

from src.fetch_nhl_api import fetch_skater_summary, fetch_skater_bios
from src.match_names import match_contracts_to_bios
from src.features import fill_in_contract_types
from src.cap_ceilings import CAP_CEILING


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
