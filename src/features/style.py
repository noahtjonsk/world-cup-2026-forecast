def team_style_vector(team_style, team, season):
    """One team-season's style metrics as a dict metric->value (long table).

    `team_style` must be pre-filtered to a single source: duplicate metric rows for the
    same (team, season) raise rather than being averaged, because two sources can
    define the same metric name differently and blending them produces a number that
    means nothing."""
    sub = team_style[(team_style["team"] == team)
                     & (team_style["season"].astype(str) == str(season))]
    if sub["metric"].duplicated().any():
        raise ValueError(
            f"team_style has duplicate metric rows for team={team!r} season={season!r}; "
            "filter to one source (or pre-aggregate) before calling style features."
        )
    return dict(zip(sub["metric"], sub["value"]))


def style_mismatch(team_style, home, away, season):
    """Euclidean distance between two teams' style vectors over shared metrics, plus
    the signed mean_xt gap. Returns {style_mismatch, xt_diff}; NaN distance if the
    teams share no metrics.

    The distance runs over shared metrics only, so two matchups that share different
    metric sets are not strictly on the same scale. Normalising would fix it if that
    ever starts to matter.

    On leakage: team_style carries no date, so keeping it honest is the caller's job.
    Only add a season's style rows once that season has finished, since this matches
    on `season` alone and cannot tell an in-progress season from a completed one."""
    h = team_style_vector(team_style, home, season)
    a = team_style_vector(team_style, away, season)
    shared = set(h) & set(a)
    dist = sum((h[m] - a[m]) ** 2 for m in shared) ** 0.5 if shared else float("nan")
    return {
        "style_mismatch": dist,
        "xt_diff": h.get("mean_xt", float("nan")) - a.get("mean_xt", float("nan")),
    }
