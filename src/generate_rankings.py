"""Produce the residual-ranking CSVs the Streamlit app reads.

Skaters: ranked with a 2026-only in-sample model (n=137 is enough, and the
season-split analysis showed pooling flat-cap-era data hurts boom-year fit).
Goalies: only ~12 of the 2026 signings played enough in 2025-26 to stand alone,
so the goalie model is trained on pooled 2018-23 + 2026 goalie-seasons and the
2026 class is ranked against it (heavier small-sample caveat - a secondary result).
Both rankings are in-sample/descriptive, not out-of-sample predictions.

Assumes the processed datasets already exist (see the src/build_*.py modules).
"""
import pandas as pd

from src.model import prepare, prepare_goalies, fit_and_rank, FEATURES, GOALIE_FEATURES

SKATER_OUT = "data/processed/skaters_2026_residual_ranking.csv"
GOALIE_OUT = "data/processed/goalies_2026_residual_ranking.csv"


def generate_skater_ranking() -> pd.DataFrame:
    sk = prepare(pd.read_csv("data/processed/skaters_2026_dataset.csv"))
    ranking = fit_and_rank(sk, features=FEATURES)
    ranking.to_csv(SKATER_OUT, index=False)
    return ranking


def generate_goalie_ranking() -> pd.DataFrame:
    hist = pd.read_csv("data/processed/goalies_historical_dataset.csv")
    hist["season"] = hist["season"].astype(str)
    cur = pd.read_csv("data/processed/goalies_2026_dataset.csv")
    cur["season"] = cur["season"].astype(str)
    pooled = prepare_goalies(pd.concat([hist, cur], ignore_index=True))
    rank_rows = pooled[pooled["season"] == "20262027"]
    ranking = fit_and_rank(pooled, rank_rows, features=GOALIE_FEATURES)
    ranking.to_csv(GOALIE_OUT, index=False)
    return ranking


if __name__ == "__main__":
    sk = generate_skater_ranking()
    print(f"skaters: {len(sk)} ranked -> {SKATER_OUT}")
    g = generate_goalie_ranking()
    print(f"goalies: {len(g)} ranked -> {GOALIE_OUT}")
    print("\nTop 5 overpaid goalies:")
    print(g.head(5)[["player_name", "cap_hit", "predicted_cap_hit", "residual"]].to_string(index=False))
