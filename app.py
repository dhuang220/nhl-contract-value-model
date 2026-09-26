"""NHL Contract Value dashboard.

Interactive view of the residual ranking: which 2026-offseason signings the
model reads as over- or under-paid, given the player's walk-year performance.
Reads pre-computed ranking CSVs produced by src/generate_rankings.py.
"""
import os

import altair as alt
import pandas as pd
import streamlit as st

RANKINGS = {
    "Skaters": "data/processed/skaters_current_residual_ranking.csv",
    "Goalies": "data/processed/goalies_current_residual_ranking.csv",
}

st.set_page_config(page_title="NHL Contract Value Model", layout="wide")


def millions(x: float) -> str:
    return f"${x/1e6:.1f}M"


@st.cache_data
def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


st.title("NHL Contract Value Model")
st.caption(
    "Predicted cap hit vs. actual, for **every current NHL contract**. "
    "The model estimates a fair AAV from the player's 2025-26 performance "
    "(scoring rate, ice time, durability, age, and free-agency status). "
    "Positive residual = paid more than the model expects (**overpaid**); "
    "negative = **underpaid**. A descriptive value tool, not a crystal ball - "
    "it can't see prospect upside, intangibles, or negotiation leverage."
)

available = {k: v for k, v in RANKINGS.items() if os.path.exists(v)}
if not available:
    st.warning("No ranking data found. Run `python -m src.generate_rankings` first.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    group = st.radio("Player type", list(available.keys()))
    df = load(available[group])

    search = st.text_input("Search player", placeholder="e.g. McDavid")

    teams = sorted(t for t in df["team"].dropna().unique() if t)
    picked_teams = st.multiselect("Team (blank = all)", teams)

    types = sorted(df["contract_type"].dropna().unique())
    picked_types = st.multiselect("Contract type", types, default=types)

    positions = sorted(df["position"].dropna().unique())
    picked_pos = st.multiselect("Position", positions, default=positions)

    top_n = st.slider("Show top N per side", 5, 30, 12)

view = df[df["contract_type"].isin(picked_types) & df["position"].isin(picked_pos)].copy()
if picked_teams:
    view = view[view["team"].isin(picked_teams)]
if search:
    view = view[view["player_name"].str.contains(search.strip(), case=False, na=False)]

if view.empty:
    st.info("No players match the current filters.")
    st.stop()

# summary tiles
c1, c2, c3, c4 = st.columns(4)
c1.metric("Contracts", len(view))
c2.metric("Median cap hit", millions(view["cap_hit"].median()))
most_over = view.loc[view["residual"].idxmax()]
most_under = view.loc[view["residual"].idxmin()]
c3.metric("Most overpaid", most_over["player_name"], millions(most_over["residual"]))
c4.metric("Most underpaid", most_under["player_name"], millions(most_under["residual"]))

# predicted vs actual scatter, diagonal = fair value
lim = float(max(view["cap_hit"].max(), view["predicted_cap_hit"].max())) * 1.05
view["Verdict"] = view["residual"].apply(lambda r: "Overpaid" if r > 0 else "Underpaid")
diag = pd.DataFrame({"x": [0, lim], "y": [0, lim]})

scatter = (
    alt.Chart(view)
    .mark_circle(size=90, opacity=0.7)
    .encode(
        x=alt.X("predicted_cap_hit:Q", title="Predicted cap hit ($)", scale=alt.Scale(domain=[0, lim])),
        y=alt.Y("cap_hit:Q", title="Actual cap hit ($)", scale=alt.Scale(domain=[0, lim])),
        color=alt.Color("Verdict:N", scale=alt.Scale(domain=["Overpaid", "Underpaid"], range=["#d1495b", "#2e86ab"])),
        tooltip=[
            alt.Tooltip("player_name:N", title="Player"),
            alt.Tooltip("team:N", title="Team"),
            alt.Tooltip("position:N", title="Pos"),
            alt.Tooltip("contract_type:N", title="Type"),
            alt.Tooltip("cap_hit:Q", title="Actual", format="$,.0f"),
            alt.Tooltip("predicted_cap_hit:Q", title="Predicted", format="$,.0f"),
            alt.Tooltip("residual:Q", title="Residual", format="$,.0f"),
        ],
    )
)
fair_line = alt.Chart(diag).mark_line(color="#888", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
st.subheader("Predicted vs. actual cap hit")
st.caption("Points above the dashed fair-value line are paid more than predicted; points below, less.")
st.altair_chart((fair_line + scatter).properties(height=460).interactive(), use_container_width=True)


def show_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[["player_name", "team", "position", "age", "contract_type", "cap_hit", "predicted_cap_hit", "residual"]].copy()
    for col in ["cap_hit", "predicted_cap_hit", "residual"]:
        out[col] = out[col].apply(millions)
    return out.rename(columns={
        "player_name": "Player", "team": "Team", "position": "Pos", "age": "Age",
        "contract_type": "Type", "cap_hit": "Actual",
        "predicted_cap_hit": "Predicted", "residual": "Residual",
    })


if search:
    st.subheader(f"Search results ({len(view)})")
    st.dataframe(show_table(view.sort_values("residual", ascending=False)), hide_index=True, use_container_width=True)
else:
    left, right = st.columns(2)
    with left:
        st.subheader("Most overpaid")
        st.dataframe(show_table(view.nlargest(top_n, "residual")), hide_index=True, use_container_width=True)
    with right:
        st.subheader("Most underpaid")
        st.dataframe(show_table(view.nsmallest(top_n, "residual")), hide_index=True, use_container_width=True)
