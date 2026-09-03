import streamlit as st

from src.evaluation import scorecard as sc
from src.report import scorecard_view as view
from src.utils.io import read_parquet

st.set_page_config(layout="wide")
st.title("How the forecast did")

try:
    preds = read_parquet("data/processed/match_predictions.parquet")
    sim = read_parquet("data/processed/simulation_results.parquet")
    results = read_parquet("data/processed/results_2026.parquet")
except FileNotFoundError:
    st.warning("No results yet. Run scripts/ingest/fetch_2026_results.py to fetch them, "
               "then reload this page.")
    st.stop()

joined = sc.join_predictions(preds, results)
metrics = sc.wdl_metrics(joined)
skill = sc.skill_interval(joined)

st.caption(view.WHAT_THIS_SCORES)

cols = st.columns(4)
for col, (label, value) in zip(cols, view.headline(metrics, skill)):
    col.metric(label, value)

st.subheader("Match predictions vs two predictors that know nothing")
st.dataframe(view.metrics_table(metrics), use_container_width=True, hide_index=True)
st.caption(view.HOW_TO_READ)
st.write(
    f"The advantage over the uniform predictor is {skill['mean_advantage']:.3f} in "
    f"log-loss per match, with a bootstrap 95% interval of {skill['ci_low']:.3f} to "
    f"{skill['ci_high']:.3f}. It clears zero, so the edge is real on this tournament, "
    f"and it is modest on {skill['n']} matches."
)

st.subheader("Goals and where the forecast leaned")
left, right = st.columns(2)
left.dataframe(view.goals_summary(sc.goal_metrics(joined)), use_container_width=True,
               hide_index=True)
mix = sc.outcome_mix(joined)
mix["mean_predicted"] = (mix["mean_predicted"] * 100).round(1)
mix["observed"] = (mix["observed"] * 100).round(1)
right.dataframe(mix.rename(columns={"mean_predicted": "mean predicted %",
                                    "observed": "actually happened %"}),
                use_container_width=True, hide_index=True)

st.subheader("Reaching each round")
reliability = sc.round_reliability(sc.round_reach_comparison(sim, results))
st.dataframe(view.reliability_table(reliability), use_container_width=True,
             hide_index=True)
st.caption("Across every team and knockout round: when the model said 25%, did about a "
           "quarter of those happen? This is the part that holds up best.")

st.subheader("The title odds")
st.info(view.champion_sentence(sc.champion_check(sim, results)))

st.subheader("Every match, worst miss first")
st.caption("Ordered by the probability the model gave to the outcome that actually "
           "happened, so the top of this table is where it was most wrong.")
st.dataframe(view.per_match_table(joined, results), use_container_width=True,
             hide_index=True)
