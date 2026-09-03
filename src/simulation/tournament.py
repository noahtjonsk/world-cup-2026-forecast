from src.simulation.standings import stats_fast, rank_group_fast, rank_third_places_fast
from src.simulation.bracket import seed_knockout, next_round
from src.simulation.match import match_lambdas, simulate_match


def _round_robin(teams):
    """All unordered pairs once; the earlier-listed team is the nominal home side."""
    return [(teams[i], teams[j]) for i in range(len(teams)) for j in range(i + 1, len(teams))]


def simulate_tournament(fmt, groups, params, seeding, rng, hosts=(), rho=0.0, max_goals=10,
                        bracket_log=None):
    """One full-tournament realization. Returns (champion, reached) where reached
    maps each team -> furthest stage index (0=group, 1..len(rounds)=knockout rounds,
    len(rounds)+1=champion).

    fmt keys used: qualifiers_per_group, third_place_slots, rounds.
    groups: {group_label: [team, ...]}; seeding: opening-round token pairings.
    bracket_log: optional list; when given, every knockout match appends
    (round_name, match_idx, home, away, winner), pure recording, consumes no rng."""
    rounds = fmt["rounds"]
    reached = {t: 0 for teams in groups.values() for t in teams}

    # --- group stage --- (tuple fast path: no DataFrames inside the Monte-Carlo loop)
    rankings, thirds_rows = {}, []
    for g, teams in groups.items():
        results = []
        for home, away in _round_robin(teams):
            lam_h, lam_a = match_lambdas(params, home, away, hosts=hosts)
            hg, ag, _ = simulate_match(lam_h, lam_a, rng, rho=rho, max_goals=max_goals, knockout=False)
            results.append((home, away, hg, ag))
        order = rank_group_fast(results, rng)
        rankings[g] = order
        if fmt["third_place_slots"] > 0 and len(order) >= 3:
            pts, gd, gf = stats_fast(results)[order[2]]
            thirds_rows.append((g, order[2], pts, gd, gf))

    thirds_ranked = (rank_third_places_fast(thirds_rows, fmt["third_place_slots"], rng)
                     if fmt["third_place_slots"] > 0 else [])

    # --- knockout ---
    pairings = seed_knockout(rankings, thirds_ranked, seeding)
    champion = None
    for ri, rname in enumerate(rounds):
        stage = ri + 1
        winners = []
        for mi, (home, away) in enumerate(pairings):
            reached[home] = max(reached[home], stage)
            reached[away] = max(reached[away], stage)
            lam_h, lam_a = match_lambdas(params, home, away, hosts=hosts)
            _, _, w = simulate_match(lam_h, lam_a, rng, rho=rho, max_goals=max_goals, knockout=True)
            winner = home if w == "home" else away
            winners.append(winner)
            if bracket_log is not None:
                bracket_log.append((rname, mi, home, away, winner))
        if rname == rounds[-1]:                       # final just played
            champion = winners[0]
            reached[champion] = stage + 1
            break
        pairings = next_round(winners)
    return champion, reached
