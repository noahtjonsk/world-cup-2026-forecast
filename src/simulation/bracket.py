def next_round(winners):
    """Pair consecutive winners into the next knockout round: winners 0&1, 2&3, ...
    Standard single-elimination ordering, so the seeding list order in
    config/tournaments.yaml determines who can meet in later rounds."""
    return [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]


def seed_knockout(rankings, thirds_ranked, seeding):
    """Resolve opening-round token pairings into concrete (home, away) tuples.
    Tokens: '1X'/'2X' = winner/runner-up of group X (X is the group label);
    'T1'..'Tk' = best third-placed teams in ranked order (1-based).
    rankings = {group: [1st, 2nd, 3rd, 4th]}; thirds_ranked = [(group, team), ...]."""
    def resolve(tok):
        if tok[0] in ("1", "2"):
            return rankings[tok[1:]][int(tok[0]) - 1]
        if tok[0] == "T":
            return thirds_ranked[int(tok[1:]) - 1][1]
        raise ValueError(f"bad seeding token {tok!r}")
    return [(resolve(h), resolve(a)) for h, a in seeding]
