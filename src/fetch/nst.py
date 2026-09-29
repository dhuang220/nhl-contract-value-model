"""Scrape goalie GSAx (goals saved above expected) from Natural Stat Trick.

NST's robots.txt allows /playerteams.php; this makes a single polite request
(identifying user-agent) per season - no rapid crawling - and computes
GSAx = xG Against - Goals Against, the shot-quality-adjusted goalie value metric
that Hockey-Reference (GSAA) and the NHL API don't provide.
"""
import unicodedata

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "nhl-contract-value-model personal non-commercial project (github.com/dhuang220)"
}


def _norm(name: str) -> str:
    """Accent- and case-normalize a name for matching (Meriläinen -> merilainen)."""
    n = "".join(c for c in unicodedata.normalize("NFKD", str(name)) if not unicodedata.combining(c))
    return n.lower().strip()


def fetch_goalie_gsax(season: str = "20252026") -> dict:
    """Return {normalized player name: GSAx} for one season."""
    url = (f"https://www.naturalstattrick.com/playerteams.php?fromseason={season}"
           f"&thruseason={season}&stype=2&sit=all&score=all&stdoi=g&rate=n"
           f"&team=ALL&pos=G&loc=B&toi=0&gpfilt=none")
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    table = BeautifulSoup(r.text, "html.parser").find("table")
    heads = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]
    gi, xi, pi = heads.index("Goals Against"), heads.index("xG Against"), heads.index("Player")

    out = {}
    for tr in table.find("tbody").find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) <= xi:
            continue
        try:
            gsax = float(cells[xi]) - float(cells[gi])
        except ValueError:
            continue
        out[_norm(cells[pi])] = round(gsax, 2)
    return out


if __name__ == "__main__":
    g = fetch_goalie_gsax("20252026")
    print(f"scraped GSAx for {len(g)} goalies")
    top = sorted(g.items(), key=lambda kv: -kv[1])[:5]
    print("top:", top)
