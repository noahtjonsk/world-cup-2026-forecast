import streamlit as st

st.set_page_config(page_title="World Cup 2026 Forecast", layout="wide")
st.title("World Cup 2026 Forecast")
st.caption("Elo + CatBoost W/D/L + Dixon-Coles goals -> 48-team Monte-Carlo. Read-only dashboard.")
st.page_link("pages/1_Match.py", label="Match forecasts")
st.page_link("pages/2_Team.py", label="Team profiles")
st.page_link("pages/3_Tournament.py", label="Tournament odds")
st.page_link("pages/4_Model.py", label="Model & limitations")
st.page_link("pages/5_Bracket.py", label="Predicted bracket")
st.page_link("pages/6_Scorecard.py", label="How the forecast did")
