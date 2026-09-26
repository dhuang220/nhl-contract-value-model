"""Produce the residual-ranking CSVs the Streamlit app reads.

Ranks EVERY current NHL contract (skaters and goalies) by how far its cap hit
sits above/below what 2025-26 performance predicts. Both models are trained
in-sample on the current-market data itself - a contemporaneous value model -
which sidesteps the flat-cap-era vs boom-era distribution shift (all rows are
the same season). Residuals are descriptive (over/under-paid relative to the
current market), not out-of-sample predictions.

Reads the processed datasets built by src/build_dataset.py (skaters) and
src/build_goalie_2026.py (goalies).
"""
import pandas as pd

from src.model import (
    prepare, prepare_goalies, fit_and_rank, cross_validated_scores,
    FEATURES, GOALIE_FEATURES,
)

SKATER_IN = "data/processed/skaters_current_dataset.csv"
GOALIE_IN = "data/processed/goalies_current_dataset.csv"
SKATER_OUT = "data/processed/skaters_current_residual_ranking.csv"
GOALIE_OUT = "data/processed/goalies_current_residual_ranking.csv"


def generate_skater_ranking() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = prepare(pd.read_csv(SKATER_IN))
    ranking = fit_and_rank(df, features=FEATURES)
    ranking.to_csv(SKATER_OUT, index=False)
    return df, ranking


def generate_goalie_ranking() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = prepare_goalies(pd.read_csv(GOALIE_IN))
    ranking = fit_and_rank(df, features=GOALIE_FEATURES)
    ranking.to_csv(GOALIE_OUT, index=False)
    return df, ranking


if __name__ == "__main__":
    sdf, sk = generate_skater_ranking()
    s = cross_validated_scores(sdf, features=FEATURES)
    print(f"skaters: {len(sk)} ranked -> {SKATER_OUT}")
    print(f"  CV R2(log)={s['r2_log']:.3f}  MAE=${s['mae_dollars']:,.0f}")

    gdf, g = generate_goalie_ranking()
    gs = cross_validated_scores(gdf, features=GOALIE_FEATURES)
    print(f"goalies: {len(g)} ranked -> {GOALIE_OUT}")
    print(f"  CV R2(log)={gs['r2_log']:.3f}  MAE=${gs['mae_dollars']:,.0f}")

    fmt = lambda d: d.assign(**{c: (d[c] / 1e6).round(1) for c in ["cap_hit", "predicted_cap_hit", "residual"]})
    print("\nMost overpaid skaters ($M):")
    print(fmt(sk.head(6))[["player_name", "position", "cap_hit", "predicted_cap_hit", "residual"]].to_string(index=False))
    print("\nMost underpaid skaters ($M):")
    print(fmt(sk.tail(6).iloc[::-1])[["player_name", "position", "cap_hit", "predicted_cap_hit", "residual"]].to_string(index=False))
