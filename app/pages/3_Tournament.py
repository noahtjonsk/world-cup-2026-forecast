import streamlit as st
from src.utils.io import read_parquet
from src.report.tournament_view import progression_table, qualification_view
from src.report.model_view import LIMITATIONS

st.title("Tournament progression and title odds")
res = read_parquet("data/processed/simulation_results.parquet")
st.subheader("Reach probabilities (ordered by title odds)")
st.dataframe(progression_table(res, tournament="2026"))
teams = st.multiselect("Highlight teams", sorted(res["team"].unique()))
if teams:
    st.line_chart(qualification_view(res, tournament="2026", teams=teams),
                  x="round", y="prob", color="team")
st.info(LIMITATIONS)        # intervals are in the table; coarse-update caveat travels here
