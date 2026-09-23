import requests

SKATER_SUMMARY_URL = "https://api.nhle.com/stats/rest/en/skater/summary"


def fetch_skater_season(season_id: str) -> list[dict]:
    """Fetch every skater's season totals for one NHL season (e.g. '20242025')."""
    limit = 100
    start = 0
    all_players = []

    while True:
        response = requests.get(SKATER_SUMMARY_URL, params={
            "limit": limit,
            "start": start,
            "cayenneExp": f"seasonId={season_id}",
        })
        page = response.json()["data"]

        if len(page) == 0:
            break

        all_players.extend(page)
        start += limit

    return all_players
