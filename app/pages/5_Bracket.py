import streamlit as st
from src.utils.io import read_parquet
from src.report.bracket_view import slot_candidates, modal_pairings, OFFICIAL_MATCH_NUMBERS
from src.simulation.montecarlo import champion_odds

st.set_page_config(layout="wide")
st.title("Predicted knockout bracket")
st.caption("Slot percentages = share of 10,000 simulated tournaments in which the team "
           "occupies that bracket slot (likely opponents and head-to-head odds are priced "
           "in, because every simulated knockout match is played between the teams that "
           "actually "
           "arrived there). 'Most likely tie' shows that pairing's head-to-head outcome, "
           "extra time and shootouts included.")

bracket = read_parquet("data/processed/bracket_results.parquet")
results = read_parquet("data/processed/simulation_results.parquet")
slots = slot_candidates(bracket)
pairs = modal_pairings(bracket).set_index(["round", "match_idx"])


def _side(rnd, mi, side, k=2):
    s = slots[(slots["round"] == rnd) & (slots["match_idx"] == mi) & (slots["side"] == side)]
    return " / ".join(f"{r['team']} {r['prob']:.0%}" for _, r in s.head(k).iterrows())


def _match_block(col, rnd, mi):
    p = pairs.loc[(rnd, mi)]
    col.markdown(
        f"**M{OFFICIAL_MATCH_NUMBERS[rnd][mi]}**  \n"
        f"{_side(rnd, mi, 'home')}  \n"
        f"{_side(rnd, mi, 'away')}  \n"
        f"<small>Most likely: {p['home_team']} v {p['away_team']} ({p['pair_prob']:.0%}) → "
        f"{p['favorite']} wins {p['p_favorite']:.0%}</small>",
        unsafe_allow_html=True,
    )
    col.divider()


cols = st.columns(5)
for col, rnd in zip(cols, ["R32", "R16", "QF", "SF", "F"]):
    col.subheader(rnd if rnd != "F" else "Final")
    for mi in range(len(OFFICIAL_MATCH_NUMBERS[rnd])):
        _match_block(col, rnd, mi)

champ = champion_odds(results, top=1).iloc[0]
cols[4].metric("Most likely champion", champ["team"], f"{champ['prob']:.1%} title odds")

st.caption("Caveats: third-place qualifiers enter by ranking into the official third-place "
           "slots (an approximation of FIFA's combination-dependent allocation table), and "
           "drawn knockouts are resolved by a goal-model extra time then a 50/50 shootout.")
