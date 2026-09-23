import requests

SKATER_SUMMARY_URL = "https://api.nhle.com/stats/rest/en/skater/summary"
SKATER_BIOS_URL = "https://api.nhle.com/stats/rest/en/skater/bios"


def _fetch_all_pages(url: str, season_id: str) -> list[dict]:
    """Page through an NHL stats REST endpoint until a page comes back empty."""
    limit = 100
    start = 0
    all_rows = []

    while True:
        response = requests.get(url, params={
            "limit": limit,
            "start": start,
            "cayenneExp": f"seasonId={season_id} and gameTypeId=2",
        })
        page = response.json()["data"]

        if len(page) == 0:
            break

        all_rows.extend(page)
        start += limit

    # The API occasionally repeats a row across a page boundary; collapse
    # by playerId since duplicates carry identical data.
    deduped = {row["playerId"]: row for row in all_rows}
    return list(deduped.values())


def fetch_skater_summary(season_id: str) -> list[dict]:
    """Fetch every skater's season totals (GP, points, TOI, position, ...)."""
    return _fetch_all_pages(SKATER_SUMMARY_URL, season_id)


def fetch_skater_bios(season_id: str) -> list[dict]:
    """Fetch every skater's bio info (birth date, etc.) for a season."""
    return _fetch_all_pages(SKATER_BIOS_URL, season_id)
