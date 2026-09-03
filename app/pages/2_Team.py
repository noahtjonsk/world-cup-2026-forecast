import streamlit as st
from src.utils.io import read_parquet
from src.report.team_view import style_profile, strength_summary

st.title("Team profile")
groups = read_parquet("data/processed/groups.parquet")
style = read_parquet("data/processed/team_style.parquet")
ratings = read_parquet("data/processed/team_ratings.parquet")
feats = read_parquet("data/processed/matchup_features.parquet")
team = st.selectbox("Team", sorted(groups["team"].unique()))    # exactly the 48 WC teams
grp = groups.loc[groups["team"] == team, "group"].iloc[0]
st.caption(f"Group {grp}")
s = strength_summary(ratings, feats, team)
c1, c2, c3 = st.columns(3)
c1.metric("Elo", f"{s['elo']:.0f}" if s["elo"] is not None else "n/a")
c2.metric("XI quality", f"{s['xi_quality']:.2f}" if s["xi_quality"] is not None else "n/a")
c3.metric("Role coverage", f"{s['role_coverage']:.2f}" if s["role_coverage"] is not None else "n/a")
st.subheader("Style profile")
profile = style_profile(style, team)
if profile.empty:
    st.info("No event-data style profile for this team (StatsBomb open data covers 59 "
            "national teams; style is a descriptive extra, not a model input here).")
else:
    st.bar_chart(profile, x="metric", y="value")
