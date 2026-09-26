"""Scrape current NHL contracts from CapWages team cap pages.

CapWages' robots.txt permits general crawlers (User-agent: * Allow: /). This
pulls the per-team roster tables for personal, non-commercial analysis only,
rate-limited with an identifying user-agent. The raw output is NOT redistributed
(kept out of the public repo per .gitignore), which keeps clear of the ToS's
concern (bulk redistribution to third parties).
"""
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "nhl-contract-value-model personal non-commercial project (github.com/dhuang220)"
}
CRAWL_DELAY = 3  # be polite

TEAM_SLUGS = [
    "anaheim_ducks", "boston_bruins", "buffalo_sabres", "calgary_flames",
    "carolina_hurricanes", "chicago_blackhawks", "colorado_avalanche",
    "columbus_blue_jackets", "dallas_stars", "detroit_red_wings",
    "edmonton_oilers", "florida_panthers", "los_angeles_kings", "minnesota_wild",
    "montreal_canadiens", "nashville_predators", "new_jersey_devils",
    "new_york_islanders", "new_york_rangers", "ottawa_senators",
    "philadelphia_flyers", "pittsburgh_penguins", "san_jose_sharks",
    "seattle_kraken", "st_louis_blues", "tampa_bay_lightning",
    "toronto_maple_leafs", "utah_mammoth", "vancouver_canucks",
    "vegas_golden_knights", "washington_capitals", "winnipeg_jets",
]

_ROSTER_SECTIONS = ("forward", "defense", "defence", "goalie")
_MONEY = re.compile(r"\$[\d,]+")


def _reformat_name(anchor_text: str) -> str:
    """'McDavid, Connor' -> 'Connor McDavid'."""
    name = anchor_text.split('"')[0].strip()  # drop trailing quoted position
    if ", " in name:
        last, first = name.split(", ", 1)
        return f"{first} {last}"
    return name


def _year_col_index(table) -> int | None:
    for i, th in enumerate(table.find_all("th")):
        if "2026-27" in th.get_text():
            return i
    return None


def parse_team_contracts(soup: BeautifulSoup, team: str) -> list[dict]:
    rows = []
    for table in soup.find_all("table"):
        ths = table.find_all("th")
        if not ths:
            continue
        header = ths[0].get_text(strip=True).lower()
        if not header.startswith(_ROSTER_SECTIONS):
            continue  # skip buyout history, reserve list, AHL, etc.
        year_i = _year_col_index(table)
        if year_i is None:
            continue
        body = table.find("tbody")
        if not body:
            continue
        for tr in body.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if len(cells) <= year_i:
                continue
            anchor = tr.find("a")
            if not anchor:
                continue
            cap_text = cells[year_i].get_text()
            m = _MONEY.search(cap_text)
            if not m:
                continue  # no salary for the current season
            cap_hit = int(m.group(0).replace("$", "").replace(",", ""))
            rows.append({
                "player_name": _reformat_name(anchor.get_text(strip=True)),
                "age": cells[1].get_text(strip=True),
                "position": cells[2].get_text(strip=True),
                "cap_hit": cap_hit,
                "team": team,
            })
    return rows


def fetch_current_contracts(save_path: str | None = None) -> pd.DataFrame:
    all_rows = []
    for slug in TEAM_SLUGS:
        r = requests.get(f"https://capwages.com/teams/{slug}", headers=HEADERS)
        r.raise_for_status()
        all_rows.extend(parse_team_contracts(BeautifulSoup(r.text, "html.parser"), slug))
        time.sleep(CRAWL_DELAY)

    df = pd.DataFrame(all_rows)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df.dropna(subset=["age"]).drop_duplicates(subset=["player_name"])
    df["age"] = df["age"].astype(int)
    df["contract_type"] = ""       # computed downstream (features.infer_contract_type)
    df["term_years"] = pd.NA       # not used as a feature
    df["offseason_year"] = 2026    # current-season framing (walk year 2025-26)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


if __name__ == "__main__":
    df = fetch_current_contracts("data/raw/contracts/capwages_current_2026.csv")
    print(f"scraped {len(df)} current contracts across {df['team'].nunique()} teams")
    print("positions:", df["position"].value_counts().to_dict())
    print(df.nlargest(5, "cap_hit")[["player_name", "position", "age", "cap_hit", "team"]].to_string(index=False))
