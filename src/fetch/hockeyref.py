import time
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "nhl-contract-value-model personal project (github.com/dhuang220)"
}


def fetch_skater_stats_page(season_end_year: int) -> BeautifulSoup:
    """Fetch one season's skater stats page, e.g. 2026 for the 2025-26 season.

    Respects Hockey-Reference's own stated crawl delay (robots.txt: 3s) -
    sleeps before returning so a caller looping over multiple seasons never
    hits them faster than that.
    """
    url = f"https://www.hockey-reference.com/leagues/NHL_{season_end_year}_skaters.html"
    response = requests.get(url, headers=HEADERS)
    time.sleep(3)
    return BeautifulSoup(response.text, "html.parser")


def _parse_stats_table(soup: BeautifulSoup, table_id: str, fields: list[str]) -> list[dict]:
    """Parse an HR season stats table; skip traded-player team-split rows."""
    table = soup.find("table", id=table_id)
    rows = table.find("tbody").find_all("tr")
    out = []
    for row in rows:
        if row.get("class") == ["partial_table"]:
            continue  # individual team-stint rows for traded players
        cells = {f: row.find("td", {"data-stat": f}) for f in fields}
        if any(c is None for c in cells.values()):
            continue  # header-repeat or malformed row
        out.append({f: c.text for f, c in cells.items()})
    return out


def parse_skater_stats(soup: BeautifulSoup) -> list[dict]:
    """Parse the season skater stats table into a list of per-player dicts."""
    return _parse_stats_table(soup, "player_stats", [
        "name_display", "age", "team_name_abbr", "pos", "games", "points", "plus_minus", "time_on_ice",
    ])


def fetch_goalie_stats_page(season_end_year: int) -> BeautifulSoup:
    """Fetch one season's goalie stats page, e.g. 2026 for the 2025-26 season."""
    url = f"https://www.hockey-reference.com/leagues/NHL_{season_end_year}_goalies.html"
    response = requests.get(url, headers=HEADERS)
    time.sleep(3)
    return BeautifulSoup(response.text, "html.parser")


def parse_goalie_stats(soup: BeautifulSoup) -> list[dict]:
    """Parse the season goalie stats table into a list of per-goalie dicts."""
    return _parse_stats_table(soup, "goalie_stats", [
        "name_display", "age", "goalie_games", "goalie_starts",
        "save_pct_goalie", "goals_against_avg", "gs_above_avg",
    ])
