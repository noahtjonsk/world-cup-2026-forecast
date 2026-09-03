import pandas as pd

# Engine match_idx (the seeding-list / consecutive-winner order) -> official FIFA
# match number, per the bracket encoded in config/tournaments.yaml. R16 follows from
# pairing consecutive R32 winners: (M74,M77)->89, (M73,M75)->90, (M83,M84)->93,
# (M81,M82)->94, (M76,M78)->91, (M79,M80)->92, (M86,M88)->95, (M85,M87)->96; then
# QF 97..100, SF 101..102, Final 104 (103 is the third-place match, not simulated).
OFFICIAL_MATCH_NUMBERS = {
    "R32": [74, 77, 73, 75, 83, 84, 81, 82, 76, 78, 79, 80, 86, 88, 85, 87],
    "R16": [89, 90, 93, 94, 91, 92, 95, 96],
    "QF": [97, 98, 99, 100],
    "SF": [101, 102],
    "F": [104],
}


def slot_candidates(bracket):
    """How often each team occupies each side of each knockout slot.

    Reads the bracket-counts table and returns tidy rows of [round, match_idx, side,
    team, prob], where prob is the share of simulated runs in which that team held
    that side of that slot. Sorted by probability within each slot and side. No side
    effects."""
    rows = []
    for (rnd, mi), g in bracket.groupby(["round", "match_idx"]):
        total = g["n"].sum()                       # the slot occurs once per run
        for side, col in (("home", "home_team"), ("away", "away_team")):
            probs = g.groupby(col)["n"].sum() / total
            for team, p in probs.sort_values(ascending=False).items():
                rows.append({"round": rnd, "match_idx": mi, "side": side,
                             "team": team, "prob": float(p)})
    return pd.DataFrame(rows, columns=["round", "match_idx", "side", "team", "prob"])


def modal_pairings(bracket):
    """The most likely pairing in each knockout slot, and who wins it.

    Returns [round, match_idx, home_team, away_team, pair_prob, p_home_win, favorite,
    p_favorite]. `p_home_win` counts home wins over runs of that pairing, with extra
    time and shootouts already resolved inside the match engine.

    `favorite` and `p_favorite` restate the same number from the likelier winner's
    side, so it is always at least 0.5. That exists because phrasing the result from
    the nominal home side reads as a prediction that the home side wins, even when it
    is the underdog. No side effects."""
    rows = []
    for (rnd, mi), g in bracket.groupby(["round", "match_idx"]):
        total = g["n"].sum()
        top = g.sort_values("n", ascending=False).iloc[0]
        p_home = float(top["home_wins"] / top["n"])
        fav, p_fav = ((top["home_team"], p_home) if p_home >= 0.5
                      else (top["away_team"], 1.0 - p_home))
        rows.append({"round": rnd, "match_idx": mi,
                     "home_team": top["home_team"], "away_team": top["away_team"],
                     "pair_prob": float(top["n"] / total),
                     "p_home_win": p_home, "favorite": fav, "p_favorite": p_fav})
    return pd.DataFrame(rows, columns=["round", "match_idx", "home_team", "away_team",
                                       "pair_prob", "p_home_win", "favorite", "p_favorite"])
