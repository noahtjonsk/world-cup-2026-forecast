def shootout_winner(rng, p_home=0.5):
    """Resolve a knockout still level after extra time. Returns 'home' or 'away'.

    Deliberately a coin flip by default. Shootout outcomes are close to random at this
    level, and there is no evidence in this project's data to justify anything more
    elaborate. `rng` is a numpy Generator, so a seeded simulation stays reproducible.

    `p_home` exists so a later version could tilt the odds by penalty taker or keeper
    record without changing any caller."""
    return "home" if rng.random() < p_home else "away"
