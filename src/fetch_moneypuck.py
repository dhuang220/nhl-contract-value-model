import pandas as pd

def load_moneypuck_xgf(csv_path: str) -> pd.DataFrame:
     """Return columns [playerId, onIce_xGoalsPercentage] for situation == 'all'."""

     df = pd.read_csv(csv_path)
     all_situations = df[df['situation'] == 'all']  
     return all_situations[['playerId', 'onIce_xGoalsPercentage']]