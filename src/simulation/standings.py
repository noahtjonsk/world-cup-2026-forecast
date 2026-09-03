import pandas as pd

_TABLE_COLS = ["team", "played", "won", "drawn", "lost", "gf", "ga", "gd", "points"]

def group_table(results):
    """Standings (points/GD/GF) for one group from its played results. `results`
    has columns home_team, away_team, home_score, away_score. Win=3, draw=1. Rows
    are alphabetical by team (ranking/tiebreaks are applied separately)."""
    teams = sorted(set(results["home_team"]) | set(results["away_team"]))
    rows = {t: dict(team=t, played=0, won=0, drawn=0, lost=0, gf=0, ga=0, gd=0, points=0)
            for t in teams}
    for _, m in results.iterrows():
        h, a, hs, as_ = m["home_team"], m["away_team"], int(m["home_score"]), int(m["away_score"])
        for t, gf, ga in ((h, hs, as_), (a, as_, hs)):
            rows[t]["played"] += 1
            rows[t]["gf"] += gf
            rows[t]["ga"] += ga
        if hs > as_:
            rows[h]["won"] += 1; rows[h]["points"] += 3; rows[a]["lost"] += 1
        elif hs < as_:
            rows[a]["won"] += 1; rows[a]["points"] += 3; rows[h]["lost"] += 1
        else:
            rows[h]["drawn"] += 1; rows[a]["drawn"] += 1
            rows[h]["points"] += 1; rows[a]["points"] += 1
    for t in teams:
        rows[t]["gd"] = rows[t]["gf"] - rows[t]["ga"]
    return pd.DataFrame([rows[t] for t in teams], columns=_TABLE_COLS)


def _ranked_with_lots(items, key, rng):
    """Sort `items` by `key` (desc); break exact-key ties by random lots (rng)."""
    ordered = sorted(items, key=key, reverse=True)
    out, i = [], 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and key(ordered[j + 1]) == key(ordered[i]):
            j += 1
        block = ordered[i:j + 1]
        if len(block) > 1:
            block = [block[k] for k in rng.permutation(len(block))]
        out.extend(block)
        i = j + 1
    return out


def _overall_key(table, team):
    r = table.loc[table["team"] == team].iloc[0]
    return (int(r["points"]), int(r["gd"]), int(r["gf"]))


def _h2h_block_order(results, tied, rng):
    """Order an overall-tied block by head-to-head (points->GD->goals among the
    tied teams), then drawn lots for any residual tie."""
    if len(tied) == 1:
        return list(tied)
    mask = results["home_team"].isin(tied) & results["away_team"].isin(tied)
    h2h = group_table(results[mask])

    def key(t):
        r = h2h.loc[h2h["team"] == t]
        return _overall_key(h2h, t) if not r.empty else (0, 0, 0)

    return _ranked_with_lots(list(tied), key, rng)


def rank_group(results, rng):
    """Ordered team list (best first) applying the FIFA 2026 tiebreaker chain:
    points -> GD -> goals -> head-to-head -> (fair-play: no data) -> drawn lots."""
    table = group_table(results)
    teams = sorted(list(table["team"]), key=lambda t: _overall_key(table, t), reverse=True)
    out, i = [], 0
    while i < len(teams):
        j = i
        while j + 1 < len(teams) and _overall_key(table, teams[j + 1]) == _overall_key(table, teams[i]):
            j += 1
        out.extend(_h2h_block_order(results, teams[i:j + 1], rng))
        i = j + 1
    return out


def rank_third_places(thirds, k, rng):
    """Rank third-placed teams across groups by points -> GD -> goals (+ lots),
    return the best `k` as a list of (group, team). `thirds` has columns
    group, team, points, gd, gf (one row per group's third-placed team)."""
    items = [(r["group"], r["team"], int(r["points"]), int(r["gd"]), int(r["gf"]))
             for _, r in thirds.iterrows()]
    return rank_third_places_fast(items, k, rng)


def rank_third_places_fast(items, k, rng):
    """rank_third_places on plain tuples [(group, team, points, gd, gf), ...] 
    the Monte-Carlo hot path (no DataFrame per run). Same ordering + lots."""
    ranked = _ranked_with_lots(items, key=lambda it: (it[2], it[3], it[4]), rng=rng)
    return [(it[0], it[1]) for it in ranked[:k]]


def stats_fast(results):
    """{team: (points, gd, gf)} from plain (home, away, home_score, away_score)
    tuples, the tuple twin of group_table for the Monte-Carlo hot path."""
    st = {}
    for h, a, hs, as_ in results:
        ph = st.setdefault(h, [0, 0, 0])
        pa = st.setdefault(a, [0, 0, 0])
        ph[1] += hs - as_; ph[2] += hs
        pa[1] += as_ - hs; pa[2] += as_
        if hs > as_:
            ph[0] += 3
        elif hs < as_:
            pa[0] += 3
        else:
            ph[0] += 1; pa[0] += 1
    return {t: tuple(v) for t, v in st.items()}


def rank_group_fast(results, rng):
    """rank_group on plain result tuples: identical tiebreak chain (points -> GD ->
    goals -> head-to-head -> lots) and identical rng consumption, minus the pandas
    overhead. `results` is [(home, away, home_score, away_score), ...]."""
    st = stats_fast(results)
    key = st.__getitem__
    teams = sorted(sorted(st), key=key, reverse=True)   # alphabetical base = group_table order
    out, i = [], 0
    while i < len(teams):
        j = i
        while j + 1 < len(teams) and key(teams[j + 1]) == key(teams[i]):
            j += 1
        block = teams[i:j + 1]
        if len(block) > 1:
            tied = set(block)
            h2h = stats_fast([r for r in results if r[0] in tied and r[1] in tied])
            out.extend(_ranked_with_lots(block, lambda t: h2h.get(t, (0, 0, 0)), rng))
        else:
            out.append(block[0])
        i = j + 1
    return out
