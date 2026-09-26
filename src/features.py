import pandas as pd


def pretty_team(team) -> str:
    """CapWages slug -> display name: 'edmonton_oilers' -> 'Edmonton Oilers'.

    Passes through abbreviations (e.g. 'TOR') and blanks/NaN unchanged.
    """
    if team is None or (isinstance(team, float) and pd.isna(team)):
        return ""
    team = str(team)
    return team.replace("_", " ").title() if "_" in team else team


def infer_contract_type(age_at_signing: int, debut_year: int, offseason_year: int) -> str:
     """Approximate UFA/RFA status: UFA at 27+ or ~7 seasons since NHL debut, else RFA."""
     years_since_debut = offseason_year - debut_year
     if years_since_debut >= 7 or age_at_signing >= 27:
          return 'UFA'
     return 'RFA'

def fill_in_contract_types(contracts_df: pd.DataFrame, matches: dict[str, dict]) -> pd.DataFrame:
    """Fill contract_type for every contract row with a matched bios record."""
    contracts_df = contracts_df.copy()
    # contract_type round-trips through CSV as an all-empty column, which
    # pandas reads back as float64 (all-NaN) - force it back to string
    # dtype so we can assign 'UFA'/'RFA' strings into it below.
    contracts_df["contract_type"] = contracts_df["contract_type"].astype(object)
    for i, row in contracts_df.iterrows():
        bio = matches.get(row["player_name"])
        if bio is None:
            continue
        debut_year = int(str(bio["firstSeasonForGameType"])[0:4])
        contracts_df.at[i, "contract_type"] = infer_contract_type(
            row["age"], debut_year, row["offseason_year"]
        )
    return contracts_df