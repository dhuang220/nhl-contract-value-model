import pandas as pd

from src.fetch.nhl_api import fetch_skater_summary, fetch_skater_bios
from src.match_names import match_rows_to_bios
from src.features import infer_contract_type, pretty_team
from src.fetch.moneypuck import load_moneypuck_ixg
from src.cap_ceilings import CAP_CEILING


def _age_on(birth_date, offseason_year, fallback=None) -> int:
    """Age as of Sept 15 of the contract's offseason (matches CapWages' convention).

    The CapWages signings tracker stopped populating age (it now returns 0), so age
    is derived from the NHL bio's birth date rather than trusting the contract CSV.
    Falls back to the CSV age only if the birth date is unavailable.
    """
    try:
        by, bm, bd = map(int, str(birth_date)[:10].split("-"))
        return offseason_year - by - ((9, 15) < (bm, bd))
    except (ValueError, TypeError, AttributeError):
        return int(fallback) if fallback else 0


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
    # per-row match, disambiguating same-name players (e.g. the two Sebastian
    # Ahos / Elias Petterssons) by position
    matched = match_rows_to_bios(skaters, bios)

    # playerId -> walk-year stat line, so we can attach performance by ID
    summary = {row["playerId"]: row for row in fetch_skater_summary(walk_year_season)}
    ixg = load_moneypuck_ixg(moneypuck_csv) if moneypuck_csv else {}

    ceiling = CAP_CEILING[contract_effective_season]
    rows = []
    for i, (_, contract) in enumerate(skaters.iterrows()):
        bio = matched[i]
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

        debut_year = int(str(bio["firstSeasonForGameType"])[:4])
        offseason_year = int(contract["offseason_year"])
        age = _age_on(bio.get("birthDate"), offseason_year, contract.get("age"))
        contract_type = infer_contract_type(age, debut_year, offseason_year)

        rows.append({
            "player_name": contract["player_name"],
            "player_id": bio["playerId"],
            "team": pretty_team(contract.get("team") or contract.get("team_signed")),
            "position": bio["positionCode"],
            "age": age,
            "contract_type": contract_type,
            "term_years": contract["term_years"],
            "games_played": gp,
            "points": stats["points"],
            "points_per_60": stats["points"] / total_toi_hours,
            "toi_per_gp_min": toi_per_gp_sec / 60,
            "plus_minus": stats["plusMinus"],
            "ixg_per_60": ixg.get(bio["playerId"], float("nan")) / total_toi_hours,
            "cap_hit": contract["cap_hit"],
            "cap_hit_pct": contract["cap_hit"] / ceiling,
        })

    df = pd.DataFrame(rows)
    if moneypuck_csv and "ixg_per_60" in df:
        # a handful of players have NHL stats but no MoneyPuck row - neutral-impute
        df["ixg_per_60"] = df["ixg_per_60"].fillna(df["ixg_per_60"].median())
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
