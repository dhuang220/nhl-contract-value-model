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


def parse_skater_stats(soup: BeautifulSoup) -> list[dict]:
    """Parse the season skater stats table into a list of per-player dicts."""
    table = soup.find("table", id="player_stats")
    rows = table.find("tbody").find_all("tr")

    fields = ["name_display", "age", "team_name_abbr", "pos", "games", "points", "plus_minus", "time_on_ice"]
    players = []

    for row in rows:
        if row.get("class") == ["partial_table"]:
            continue  # skip individual team-stint rows for traded players

        player = {}
        for field in fields:
            cell = row.find("td", {"data-stat": field})
            player[field] = cell.text

        players.append(player)

    return players
