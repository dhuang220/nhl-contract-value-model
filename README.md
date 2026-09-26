# NHL Contract Value Model

Estimate what an NHL player *should* be paid from his on-ice performance, then rank
**every current NHL contract** by how far its actual cap hit sits above or below that
estimate - surfacing the most over- and under-paid deals in the league. An interactive
Streamlit dashboard presents the results.

This is a value/inefficiency tool, not a crystal ball: it relates performance to pay and
flags outliers, with an honest error bar. It cannot see prospect upside, intangibles, or
negotiation leverage - which turns out to be exactly what its biggest residuals are made of.

## Headline results

- **Skater model:** predicts `log(cap-hit % of the salary cap)` from seven features
  (scoring rate, ice time, games played, plus-minus, age, UFA/RFA status, position).
  **~0.65 cross-validated R²** (log space), **~$1.36M mean absolute error** across **646
  current skater contracts** (GP-filtered). A goalie model (SV%, GAA, starts, GSAA, age,
  status) covers 54 contracts as a secondary result.
- **The residuals tell a coherent story.** The "most overpaid" are young stars on big
  post-entry deals (Carlsson, Bedard, Fantilli) - their 2025-26 production doesn't yet match
  contracts priced on upside the model can't see. The "most underpaid" are genuine
  below-market veterans, led by **Quinn Hughes** ($7.8M actual vs. ~$15.5M predicted) - an
  elite defenseman on a deal signed years ago. The blind spots are interpretable, which is
  the point.
- **A rising-cap regime shift, measured.** The NHL cap is climbing ~8-9%/yr after a flat
  COVID era. Normalizing pay as a *percent of the cap* handles the changing ceiling (mean
  error on the 2026 class is ~0). But in a leave-one-season-out test, model R² is stable at
  0.53-0.64 across the flat-cap seasons (2018-2023) and **collapses to 0.36 on the 2026 boom
  class** - the performance-to-pay *relationship* itself shifted, so you can't just train on
  history and expect good boom-year predictions. This is why the 2026 class is ranked with a
  2026-calibrated model rather than the pooled historical one.

## Data sources (and why collection looks the way it does)

Cap/contract data and performance stats live in different places, with very different reuse
terms - respecting those terms shaped the whole pipeline:

| Data | Source | How |
|------|--------|-----|
| Skater box scores (GP, points, TOI, +/-), bios | NHL Stats API (`api.nhle.com`) | Programmatic (paginated JSON) |
| Goalie box scores (SV%, GAA, starts, GSAA) | Hockey-Reference | Programmatic scrape - HR permits rate-limited access (verified); respects their 3s crawl-delay |
| Current NHL contracts (cap hit, all 32 teams) | CapWages team pages | Scraped - CapWages' robots.txt permits general crawlers; pulled rate-limited with an identifying UA for personal, non-commercial use, and the raw data is not redistributed |
| Historical contracts + stats, 2018-2023 (incl. xGF%, GSAx) | CapFriendly (via a shared dataset) | Reused with permission; CapFriendly shut down in 2024 |
| On-ice xGF% | MoneyPuck | Manual download (their ToS prohibits scripted access) |

The distinction that shaped collection: **robots.txt** governs what a crawler may *fetch*,
while a site's **Terms of Service** govern what you may *do* with the data. CapWages'
robots.txt allows crawling; its ToS restricts *redistribution to third parties* - so this
project crawls it politely for private analysis and keeps the raw dumps out of the public
repo (`.gitignore`). PuckPedia and Spotrac disallow it outright (ToS and/or bot-blocking) and
are not scraped; MoneyPuck blocks scripted access, so its files are downloaded by hand. See
`NOTES.md` for the full obstacle log.

## Method

- **Target:** cap hit as a percent of that season's cap ceiling (`src/cap_ceilings.py`),
  log-transformed (salary is heavily right-skewed). Percent-of-cap makes contracts from
  different cap eras comparable.
- **Framing:** contemporaneous - a season's performance paired with the contract the player
  is on - so historical and current data pool cleanly.
- **Joins:** performance sources share the NHL `playerId`; contract data (no shared ID) is
  fuzzy-matched on name (`rapidfuzz`, accent- and nickname-aware).
- **Validation:** season-aware. Random k-fold within a season for the headline error;
  leave-one-season-out to expose the regime shift above.
- **UFA/RFA** is computed from age + years since NHL debut (`src/features.py`), since the
  free-agent-status field isn't in the contract source.

## The dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

Toggle skaters/goalies, **search by player name**, and filter by **team**, contract type, and
position. Read the predicted-vs-actual scatter (points above the fair-value line are overpaid)
alongside the most over/under-paid tables.

## Reproduce the data pipeline

```bash
# 1. scrape current contracts from CapWages (rate-limited, ~2 min)
python -m src.fetch_capwages
# 2. build datasets (NHL API + Hockey-Reference fetched live)
python -m src.build_dataset      # skaters: joins contracts to 2025-26 stats
python -m src.build_goalie_2026  # goalies: joins contracts to 2025-26 goalie stats
# 3. generate the ranking CSVs the app reads
python -m src.generate_rankings
```

The historical builders (`build_historical_dataset.py`, `build_goalie_dataset.py`) reproduce
the 2018-2023 season-split analysis from the CapFriendly data, if present.

## Limitations

- **Entry-level contracts look underpaid by construction.** ELC stars (e.g. Celebrini,
  Schaefer on ~$1M rookie deals) are CBA-capped regardless of performance, so the model
  flags them as underpaid - a rule artifact, not a market inefficiency. Filter to UFA
  contracts in the app to see the genuine open-market picture.
- The skater model uses plus-minus as its on-ice feature (a weak proxy); xGF% is available
  historically and is the intended upgrade once a 2025-26 MoneyPuck file is added.
- Goalie results are a smaller, secondary/exploratory model (54 contracts).
- Residuals are descriptive, not causal - "overpaid" means "paid more than performance alone
  predicts," which for young stars often just means the market is paying for upside the model
  can't see.

## Repo layout

```
app.py                     Streamlit dashboard
src/fetch_nhl_api.py       NHL Stats API (skater/goalie summary + bios)
src/fetch_hockeyref.py     Hockey-Reference scraper (skater + goalie season stats)
src/fetch_capwages.py      CapWages team-page scraper (current league-wide contracts)
src/fetch_moneypuck.py     MoneyPuck xGF% loader (manual downloads)
src/clean_capwages_paste.py  Clean a copy-pasted CapWages signings table (legacy path)
src/match_names.py         Fuzzy name matching (contracts <-> stats)
src/features.py            Feature engineering + UFA/RFA inference
src/cap_ceilings.py        Salary cap ceiling by season
src/build_*.py             Dataset assembly (skaters/goalies, historical/2026)
src/model.py               Feature sets, cross-validation, residual ranking
src/generate_rankings.py   Writes the ranking CSVs the app reads
NOTES.md                   Running log of data obstacles and decisions
```
