import streamlit as st
import pandas as pd
from src.utils.io import read_parquet
from src.report.match_view import match_card, top_drivers

st.title("Match forecast")
st.caption("These are the predictions as published on 11 June 2026, before the tournament "
           "began, and they are deliberately not updated. For what actually happened in "
           "each match, see the Scorecard page.")
preds = read_parquet("data/processed/match_predictions.parquet")
feats = read_parquet("data/processed/matchup_features.parquet")
try:
    imp = pd.read_csv("reports/feature_importance.csv")
except FileNotFoundError:
    imp = pd.DataFrame(columns=["feature", "importance"])
label = preds.apply(lambda r: f"{r['home_team']} vs {r['away_team']} ({pd.Timestamp(r['date']).date()})", axis=1)
pick = st.selectbox("Fixture", list(label))
row = preds.loc[label == pick].iloc[0]
card = match_card(row)
c1, c2, c3 = st.columns(3)
c1.metric(f"{card['home_team']} win", f"{card['p_home']:.0%}")
c2.metric("Draw", f"{card['p_draw']:.0%}")
c3.metric(f"{card['away_team']} win", f"{card['p_away']:.0%}")
st.write(f"Expected goals: {card['exp_goals_home']:.2f} - {card['exp_goals_away']:.2f}")
st.dataframe(card["scoreline_grid"])
fr = feats.loc[feats["match_id"] == row["match_id"]]
if not fr.empty and not imp.empty:
    st.subheader("Top drivers (descriptive)")
    st.dataframe(top_drivers(fr.iloc[0], imp))
