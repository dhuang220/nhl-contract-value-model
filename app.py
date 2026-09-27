"""NHL Contract Value dashboard.

Interactive view of the residual ranking: which contracts the linear model reads
as over- or under-paid, given 2025-26 performance. Sidebar toggles: player type,
segment (all / just-signed / pending FAs), plus search and filters. Shows NHL
headshots and team logos from the league's public asset CDN. Reads pre-computed
ranking CSVs produced by src/generate_rankings.py.
"""
import os

import altair as alt
import pandas as pd
import streamlit as st

RANKINGS = {
    "Skaters": "data/processed/skaters_current_residual_ranking.csv",
    "Goalies": "data/processed/goalies_current_residual_ranking.csv",
}
SEGMENTS = {
    "All current contracts": None,
    "Just signed this year (2026)": "just_signed",
    "Pending free agents (2027)": "pending_fa",
}
OVER, UNDER = "#cc3311", "#0077bb"  # diverging pair (validated colorblind-safe)

TRICODE = {
    "Anaheim Ducks": "ANA", "Boston Bruins": "BOS", "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY", "Carolina Hurricanes": "CAR", "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL", "Columbus Blue Jackets": "CBJ", "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM", "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN", "Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH", "New Jersey Devils": "NJD", "New York Islanders": "NYI",
    "New York Rangers": "NYR", "Ottawa Senators": "OTT", "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT", "San Jose Sharks": "SJS", "Seattle Kraken": "SEA",
    "St Louis Blues": "STL", "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR",
    "Utah Mammoth": "UTA", "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH", "Winnipeg Jets": "WPG",
}

st.set_page_config(page_title="NHL Contract Value Model", layout="wide")

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
  #hero {{
    background: linear-gradient(100deg, #0b1f3a 0%, #14284a 100%);
    color: #fff; border-radius: 14px; padding: 1.3rem 1.6rem; margin-bottom: 1.1rem;
  }}
  #hero h1 {{ margin: 0; font-size: 1.7rem; letter-spacing: -0.5px; }}
  #hero p  {{ margin: .4rem 0 0; color: #b9c6dc; font-size: .9rem; max-width: 900px; }}
  .card {{
    background: #fff; border: 1px solid #e6e6e6; border-radius: 12px;
    padding: .8rem 1rem; box-shadow: 0 1px 3px rgba(0,0,0,.05); height: 100%;
  }}
  .card .lab {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .5px; color: #8a8a8a; }}
  .card .val {{ font-size: 1.5rem; font-weight: 700; color: #12233d; }}
  .pc {{ display: flex; align-items: center; gap: .7rem; }}
  .pc img.mug {{ width: 52px; height: 52px; border-radius: 50%; background:#eef1f5; object-fit: cover; }}
  .pc .nm {{ font-weight: 700; font-size: 1.05rem; color:#12233d; line-height:1.1; }}
  .pc .rs {{ font-size: 1.15rem; font-weight: 800; }}
  .pc .sub {{ font-size:.78rem; color:#7a7a7a; }}
</style>
""", unsafe_allow_html=True)


def millions(x):
    return f"${x/1e6:.1f}M"


def headshot(pid):
    return f"https://assets.nhle.com/mugs/nhl/latest/{int(pid)}.png" if pd.notna(pid) else ""


def logo(team):
    tri = TRICODE.get(team)
    return f"https://assets.nhle.com/logos/nhl/svg/{tri}_light.svg" if tri else ""


@st.cache_data
def load(path):
    return pd.read_csv(path)


st.markdown(
    "<div id='hero'><h1>NHL Contract Value Model</h1>"
    "<p>A linear model estimates each player's fair cap hit from his 2025-26 performance "
    "(scoring rate, shot quality, ice time, durability, age, free-agency status), fit on "
    "the 2024-2026 free-market signings. <b>Residual = actual − predicted</b>: "
    "positive is <span style='color:#ff8a7a'>overpaid</span>, negative is "
    "<span style='color:#7fc4ff'>underpaid</span>. A value tool, not a crystal ball — "
    "it can't see prospect upside, intangibles, or leverage.</p></div>",
    unsafe_allow_html=True,
)

avail = {pt: p for pt, p in RANKINGS.items() if os.path.exists(p)}
if not avail:
    st.warning("No ranking data found. Run `python -m src.generate_rankings` first.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    group = st.radio("Player type", list(avail.keys()))
    df = load(avail[group]).copy()

    segment = st.radio("Segment", list(SEGMENTS.keys()))
    st.caption("All = every current player. Just signed = this offseason's deals. "
               "Pending FAs = contracts expiring after this season.")
    flag = SEGMENTS[segment]
    if flag and flag in df.columns:
        df = df[df[flag]]

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

view["headshot"] = view["player_id"].apply(headshot)
view["logo"] = view["team"].apply(logo)


def player_card(label, row, color):
    st.markdown(
        f"<div class='card'><div class='lab'>{label}</div><div class='pc'>"
        f"<img class='mug' src='{headshot(row['player_id'])}'>"
        f"<div><div class='nm'>{row['player_name']}</div>"
        f"<div class='sub'>{row['team']} · {row['position']} · {row['contract_type']}</div>"
        f"<div class='rs' style='color:{color}'>{'+' if row['residual']>0 else '−'}"
        f"{millions(abs(row['residual']))}</div></div></div></div>",
        unsafe_allow_html=True,
    )


c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
with c1:
    player_card("Most overpaid", view.loc[view["residual"].idxmax()], OVER)
with c2:
    player_card("Most underpaid", view.loc[view["residual"].idxmin()], UNDER)
c3.markdown(f"<div class='card'><div class='lab'>Contracts</div><div class='val'>{len(view)}</div></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='card'><div class='lab'>Median cap</div><div class='val'>{millions(view['cap_hit'].median())}</div></div>", unsafe_allow_html=True)

# ---- scatter: predicted vs actual ----
st.markdown("### Predicted vs. actual cap hit")
v = view.copy()
v["Actual ($M)"] = v["cap_hit"] / 1e6
v["Predicted ($M)"] = v["predicted_cap_hit"] / 1e6
v["Verdict"] = v["residual"].apply(lambda r: "Overpaid" if r > 0 else "Underpaid")
lim = float(max(v["Actual ($M)"].max(), v["Predicted ($M)"].max())) * 1.05
diag = pd.DataFrame({"x": [0, lim], "y": [0, lim]})
fair = alt.Chart(diag).mark_line(color="#9aa4b2", strokeDash=[5, 5]).encode(x="x:Q", y="y:Q")
pts = (
    alt.Chart(v).mark_circle(size=95, opacity=0.78, stroke="white", strokeWidth=0.6)
    .encode(
        x=alt.X("Predicted ($M):Q", scale=alt.Scale(domain=[0, lim], clamp=True, nice=False), axis=alt.Axis(format="$.0f", grid=False)),
        y=alt.Y("Actual ($M):Q", scale=alt.Scale(domain=[0, lim], clamp=True, nice=False), axis=alt.Axis(format="$.0f", grid=False)),
        color=alt.Color("Verdict:N", scale=alt.Scale(domain=["Overpaid", "Underpaid"], range=[OVER, UNDER]),
                        legend=alt.Legend(orient="top-left", title=None)),
        tooltip=[alt.Tooltip("player_name:N", title="Player"), alt.Tooltip("team:N", title="Team"),
                 alt.Tooltip("position:N", title="Pos"),
                 alt.Tooltip("Actual ($M):Q", format="$.1f"), alt.Tooltip("Predicted ($M):Q", format="$.1f")],
    )
)
st.caption("Above the dashed fair-value line = paid more than predicted (overpaid); below = underpaid.")
# fixed view locked to [0, max] on both axes - no pan/zoom into negative cap hits
st.altair_chart((fair + pts).properties(height=440).configure_view(strokeOpacity=0),
                width='stretch')

# ---- tables with headshots + logos ----
COLCFG = {
    "headshot": st.column_config.ImageColumn(" ", width="small"),
    "player_name": st.column_config.TextColumn("Player"),
    "logo": st.column_config.ImageColumn("Team", width="small"),
    "position": st.column_config.TextColumn("Pos", width="small"),
    "age": st.column_config.NumberColumn("Age", width="small"),
    "contract_type": st.column_config.TextColumn("Type", width="small"),
    "Actual": st.column_config.NumberColumn("Actual", format="$%.1fM"),
    "Predicted": st.column_config.NumberColumn("Predicted", format="$%.1fM"),
    "Residual": st.column_config.NumberColumn("Residual", format="$%.1fM"),
}
ORDER = ["headshot", "player_name", "logo", "position", "age", "contract_type", "Actual", "Predicted", "Residual"]


def table(frame):
    t = frame.copy()
    t["Actual"] = t["cap_hit"] / 1e6
    t["Predicted"] = t["predicted_cap_hit"] / 1e6
    t["Residual"] = t["residual"] / 1e6
    return t[ORDER]


if search:
    st.markdown(f"### Search results ({len(view)})")
    st.dataframe(table(view.sort_values("residual", ascending=False)),
                 column_config=COLCFG, hide_index=True, width='stretch')
else:
    left, right = st.columns(2)
    with left:
        st.markdown(f"### <span style='color:{OVER}'>Most overpaid</span>", unsafe_allow_html=True)
        st.dataframe(table(view.nlargest(top_n, "residual")),
                     column_config=COLCFG, hide_index=True, width='stretch')
    with right:
        st.markdown(f"### <span style='color:{UNDER}'>Most underpaid</span>", unsafe_allow_html=True)
        st.dataframe(table(view.nsmallest(top_n, "residual")),
                     column_config=COLCFG, hide_index=True, width='stretch')
