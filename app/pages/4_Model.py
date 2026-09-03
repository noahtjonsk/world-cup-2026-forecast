import streamlit as st
import pandas as pd
from src.report.model_view import metrics_summary, METHODOLOGY, LIMITATIONS

st.title("Model & limitations")
st.write(METHODOLOGY)
try:
    wf = pd.read_csv("reports/walkforward.csv")
    st.subheader("Walk-forward: CatBoost vs Elo baseline")
    st.dataframe(metrics_summary(wf))
    st.caption("Built from a lineup-free historical feature table: XI-quality / bench / "
               "role-coverage columns are all-NaN, so this comparison measures the "
               "Elo + form + style + context signal only.")
    st.image("reports/calibration.png", caption="Calibration (P(home win))")
    st.subheader("Feature importance")
    st.dataframe(pd.read_csv("reports/feature_importance.csv"))
    st.caption("XI/bench/role-coverage importances are ~0 by construction here (no "
               "historical lineups), not evidence the squad signal is uninformative.")
except FileNotFoundError:
    st.warning("Run src.report.artifacts.build_report_artifacts to generate model artifacts.")
st.subheader("Limitations")
st.warning(LIMITATIONS)
