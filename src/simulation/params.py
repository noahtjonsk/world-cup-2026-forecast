def sample_params(fitted, rng, jitter=0.0):
    """Draw a perturbed copy of the fitted Dixon-Coles parameters for one simulated run.

    This is how uncertainty in the fit reaches the title odds: without it every run
    would use the same point estimate and the intervals would be far too narrow. Adds
    independent Gaussian noise of standard deviation `jitter` to each team's attack
    and defence, and passes home advantage and rho through unchanged. A jitter of 0
    returns the point estimate exactly.

    The noise is independent per team, which ignores the correlations in the fit. A
    stricter version would sample from the covariance implied by the likelihood's
    Hessian."""
    if jitter <= 0.0:
        return {"attack": dict(fitted["attack"]), "defence": dict(fitted["defence"]),
                "home_adv": fitted["home_adv"], "rho": fitted.get("rho", 0.0)}
    atk = {t: a + rng.normal(0.0, jitter) for t, a in fitted["attack"].items()}
    dfc = {t: d + rng.normal(0.0, jitter) for t, d in fitted["defence"].items()}
    return {"attack": atk, "defence": dfc,
            "home_adv": fitted["home_adv"], "rho": fitted.get("rho", 0.0)}


def apply_squad_bump(params, squad_strength, coef):
    """Adjust fitted team strengths by current squad talent. Returns a new dict.

    Adds coef * atk_q to attack and coef * dfc_q to defence for each team in
    `squad_strength`. A coefficient of 0, or an empty strength map, returns the
    parameters unchanged. Teams missing from `squad_strength` are left alone.

    Adjusted teams are re-centred afterwards, removing the mean shift, so the overall
    number of goals the model expects stays put and only the relative ordering moves.
    Without that the bump would inflate every scoreline.

    Signs follow match_lambdas: a higher atk_q scores more, a higher dfc_q concedes
    fewer."""
    base_a, base_d = params["attack"], params["defence"]
    atk, dfc = dict(base_a), dict(base_d)
    out = {"attack": atk, "defence": dfc, "home_adv": params["home_adv"],
           "rho": params.get("rho", 0.0)}
    if coef == 0 or not squad_strength:
        return out
    bumped = []
    for t, (aq, dq) in squad_strength.items():
        if t in atk:
            atk[t] += coef * aq
            dfc[t] += coef * dq
            bumped.append(t)
    if bumped:
        # Re-centre: remove the mean shift so the overall goal level is unchanged.
        a_shift = sum(atk[t] - base_a[t] for t in bumped) / len(bumped)
        d_shift = sum(dfc[t] - base_d[t] for t in bumped) / len(bumped)
        for t in bumped:
            atk[t] -= a_shift
            dfc[t] -= d_shift
    return out


def apply_elo_anchor(params, elo_target, weight):
    """Pull each team's net strength toward what its Elo rating implies. Returns a new dict.

    This is the fix for cross-confederation bias. A team that runs up goals against
    weak regional opponents looks stronger in the scoring record than it is, and a
    goal model alone cannot see that. Elo can, because it accounts for who the
    opponent was.

    `elo_target` maps team to a centred strength target. A weight of 0, or an empty
    target, returns the parameters unchanged. A weight of 1 moves each targeted team's
    deviation from the mean all the way onto its target.

    Each team's shift is split evenly across attack and defence, so a team's balance
    between the two survives while its overall level tracks Elo. Teams missing from
    `elo_target` are left alone. No side effects."""
    base_a, base_d = params["attack"], params["defence"]
    atk, dfc = dict(base_a), dict(base_d)
    out = {"attack": atk, "defence": dfc, "home_adv": params["home_adv"],
           "rho": params.get("rho", 0.0)}
    if weight == 0 or not elo_target:
        return out
    teams = [t for t in elo_target if t in atk]
    if not teams:
        return out
    mean_s = sum(base_a[t] + base_d[t] for t in teams) / len(teams)
    for t in teams:
        s_centered = (base_a[t] + base_d[t]) - mean_s
        delta = weight * (elo_target[t] - s_centered)
        atk[t] += delta / 2.0
        dfc[t] += delta / 2.0
    return out
