# NHL Contract Value Model

Estimate what an NHL player *should* be paid from his on-ice performance, then rank the
2026-offseason signings by how far their actual cap hit sits above or below that estimate -
surfacing the most over- and under-paid contracts. An interactive Streamlit dashboard
presents the results.

This is a value/inefficiency tool, not a crystal ball: it relates performance to pay and
flags outliers, with an honest error bar. It cannot see prospect upside, intangibles, or
negotiation leverage - which turns out to be exactly what its biggest residuals are made of.

## Headline results

- **Skater model:** predicts `log(cap-hit % of the salary cap)` from seven features
  (scoring rate, ice time, games played, plus-minus, age, UFA/RFA status, position).
  **~0.72 cross-validated R²** (log space), **~$1.26M mean absolute error** on 137
  GP-filtered 2026 signings.
- **The residuals tell a coherent story.** The "most overpaid" are young RFA stars on
  second contracts (Carlsson, Leonard, Gauthier) - the model only sees their walk-year
  stats, so it can't price the upside teams are paying for. The "most underpaid" are aging
  veterans on cheap deals (Zuccarello, Giroux) - the model overweights recent production and
  underweights age-decline risk. The blind spots are interpretable, which is the point.
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
| Historical contracts + stats, 2018-2023 (incl. xGF%, GSAx) | CapFriendly (via a shared academic dataset) | Reused with permission; CapFriendly shut down in 2024 |
| 2026 contracts (cap hit, term) | CapWages | Manual (their ToS prohibits scraping) |
| On-ice xGF% | MoneyPuck | Manual download (their ToS prohibits scripted access) |

PuckPedia, Spotrac, CapWages, and MoneyPuck all prohibit automated/bulk collection in their
terms, so contract data was collected by hand and the pipeline scrapes only sources that
allow it. See `NOTES.md` for the full obstacle log.

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

Toggle skaters/goalies, filter by contract type and position, and read the predicted-vs-actual
scatter (points above the fair-value line are overpaid) alongside the most over/under-paid tables.

## Reproduce the data pipeline

```bash
# 1. build datasets (NHL API + HR are fetched live; contract CSVs are in data/raw/)
python -m src.build_dataset            # 2026 skaters
python -m src.build_historical_dataset # 2018-2023 skaters
python -m src.build_goalie_dataset     # 2018-2023 goalies
python -m src.build_goalie_2026        # 2026 goalies
# 2. generate the ranking CSVs the app reads
python -m src.generate_rankings
```

## Limitations

- Small, curated samples (137 skaters / 12 goalies in the 2026 class); goalie results are a
  secondary, exploratory result.
- The skater model uses plus-minus as its on-ice feature (a weak proxy); xGF% is available
  historically and is the intended upgrade once a 2025-26 MoneyPuck file is added.
- Residuals are descriptive, not causal - "overpaid" means "paid more than performance alone
  predicts," which for young stars often just means the market is paying for upside the model
  can't see.

## Repo layout

```
app.py                     Streamlit dashboard
src/fetch_nhl_api.py       NHL Stats API (skater/goalie summary + bios)
src/fetch_hockeyref.py     Hockey-Reference scraper (skater + goalie season stats)
src/fetch_moneypuck.py     MoneyPuck xGF% loader (manual downloads)
src/clean_capwages_paste.py  Clean the copy-pasted CapWages signings table
src/match_names.py         Fuzzy name matching (contracts <-> stats)
src/features.py            Feature engineering + UFA/RFA inference
src/cap_ceilings.py        Salary cap ceiling by season
src/build_*.py             Dataset assembly (skaters/goalies, historical/2026)
src/model.py               Feature sets, cross-validation, residual ranking
src/generate_rankings.py   Writes the ranking CSVs the app reads
NOTES.md                   Running log of data obstacles and decisions
```
