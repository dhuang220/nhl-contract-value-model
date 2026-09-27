"""Scrape current NHL contracts from CapWages team cap pages.

CapWages' robots.txt permits general crawlers (User-agent: * Allow: /). This
pulls the per-team roster tables for personal, non-commercial analysis only,
rate-limited with an identifying user-agent. The raw output is NOT redistributed
(kept out of the public repo per .gitignore), which keeps clear of the ToS's
concern (bulk redistribution to third parties).
"""
import json
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

_NEXT = re.compile(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S)

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
    # dedupe on name+team+position (not name alone) so distinct same-name players
    # survive - e.g. Vancouver's two Elias Petterssons (a C and a D)
    df = df.dropna(subset=["age"]).drop_duplicates(subset=["player_name", "team", "position"])
    df["age"] = df["age"].astype(int)
    df["contract_type"] = ""       # computed downstream (features.infer_contract_type)
    df["term_years"] = pd.NA       # not used as a feature
    df["offseason_year"] = 2026    # current-season framing (walk year 2025-26)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


def _iter_players(pp):
    for group in ("roster", "non-roster"):
        for bucket in pp.get("data", {}).get(group, {}).values():
            if isinstance(bucket, list):
                yield from bucket


def fetch_pending_free_agents(save_path: str | None = None) -> pd.DataFrame:
    """Players whose current contract expires after 2026-27 (pending FAs, i.e.
    eligible to sign a new deal this year). From team-page __NEXT_DATA__: a
    player's active contract whose last covered season is 2026-27.
    """
    rows = []
    for slug in TEAM_SLUGS:
        r = requests.get(f"https://capwages.com/teams/{slug}", headers=HEADERS)
        r.raise_for_status()
        m = _NEXT.search(r.text)
        if m:
            pp = json.loads(m.group(1))["props"]["pageProps"]
            for pl in _iter_players(pp):
                cs = pl.get("contracts") or []
                if not cs:
                    continue
                seasons = [x["season"] for x in cs[0].get("details", [])]
                if seasons and max(seasons) == "2026-27" and "2026-27" in seasons:
                    rows.append({
                        "player_name": _reformat_name(pl["name"]),
                        "expiry": cs[0].get("expiryStatus", ""),
                    })
        time.sleep(CRAWL_DELAY)
    df = pd.DataFrame(rows).drop_duplicates(subset="player_name")
    if save_path:
        df.to_csv(save_path, index=False)
    return df


if __name__ == "__main__":
    fa = fetch_pending_free_agents("data/raw/contracts/pending_fa_2026.csv")
    print(f"pending free agents: {len(fa)}")
    print(fa["expiry"].str.extract(r"(UFA|RFA)")[0].value_counts().to_dict())
