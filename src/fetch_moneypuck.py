import pandas as pd

def load_moneypuck_xgf(csv_path: str) -> pd.DataFrame:
     """Return columns [playerId, onIce_xGoalsPercentage] for situation == 'all'."""

     df = pd.read_csv(csv_path)
     all_situations = df[df['situation'] == 'all']
     return all_situations[['playerId', 'onIce_xGoalsPercentage']]


def load_moneypuck_ixg(csv_path: str) -> dict:
    """Map playerId -> individual expected goals (I_F_xGoals) for the season."""
    df = pd.read_csv(csv_path)
    allx = df[df["situation"] == "all"]
    return dict(zip(allx["playerId"], allx["I_F_xGoals"]))


def load_moneypuck_advanced(csv_path: str) -> dict:
    """Map playerId -> on-ice metrics (per skater, forwards and defensemen alike):
      ixg    - individual expected goals (offense)
      xga_on - expected goals AGAINST while on ice (defense; lower = better)
      xgf_pct- on-ice xG share, for/(for+against) (two-way impact)
    """
    df = pd.read_csv(csv_path)
    a = df[df["situation"] == "all"]
    return {
        r.playerId: {"ixg": r.I_F_xGoals, "xga_on": r.OnIce_A_xGoals,
                     "xgf_pct": r.onIce_xGoalsPercentage}
        for r in a.itertuples()
    }