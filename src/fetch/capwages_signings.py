"""Fetch per-season signings from CapWages' signings tracker.

The tracker route is /signings/[[...season]] (a Next.js catch-all), so each
season's data is served statically at
    /_next/data/<buildId>/signings/<season>.json
with clean records: player, age, position, team, date, contractType, length,
capHit. `allSeasons` lists 2026-27 back to 2022-23. This lets us build a
fresh-market training set across multiple recent offseasons.

Emits one contracts CSV per season in the schema build_dataset expects.
"""
import json
import re

import pandas as pd
import requests

from src.fetch.capwages import HEADERS

SEASON_OFFSEASON = {"2026-27": 2026, "2025-26": 2025, "2024-25": 2024,
                    "2023-24": 2023, "2022-23": 2022}


def _build_id() -> str:
    r = requests.get("https://capwages.com/signings", headers=HEADERS)
    r.raise_for_status()
    return re.search(r'"buildId":"([^"]+)"', r.text).group(1)


def _name(last_first: str) -> str:
    if ", " in last_first:
        last, first = last_first.split(", ", 1)
        return f"{first} {last}"
    return last_first


def _cap(s):
    m = re.search(r"[\d,]+", str(s))
    return int(m.group(0).replace(",", "")) if m else None


def fetch_season(season: str, build_id: str | None = None, save_path: str | None = None) -> pd.DataFrame:
    build_id = build_id or _build_id()
    url = f"https://capwages.com/_next/data/{build_id}/signings/{season}.json"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    recs = r.json()["pageProps"]["signings"]
    off = SEASON_OFFSEASON[season]

    rows = []
    for s in recs:
        cap = _cap(s.get("capHit"))
        if not cap:
            continue
        rows.append({
            "player_name": _name(s["player"]),
            "age": s.get("age"),
            "position": s.get("position"),
            "contract_type": "",          # computed downstream
            "term_years": s.get("length"),
            "cap_hit": cap,
            "team": s.get("team"),
            "offseason_year": off,
            "contract_kind": s.get("contractType", ""),
            "unconfirmed": s.get("unconfirmed", False),
        })
    df = pd.DataFrame(rows).dropna(subset=["age"]).drop_duplicates(subset=["player_name"])
    if save_path:
        df.to_csv(save_path, index=False)
    return df


if __name__ == "__main__":
    bid = _build_id()
    for season in ["2026-27", "2025-26", "2024-25"]:
        off = SEASON_OFFSEASON[season]
        df = fetch_season(season, bid, f"data/raw/contracts/signings_{off}.csv")
        print(f"{season}: {len(df)} signings -> data/raw/contracts/signings_{off}.csv")
