import numpy as np
from src.simulation.shootout import shootout_winner

def test_shootout_near_coin_flip():
    rng = np.random.default_rng(0)
    outs = [shootout_winner(rng) for _ in range(2000)]
    assert set(outs) == {"home", "away"}
    assert 0.46 < outs.count("home") / 2000 < 0.54        # ~50/50, seeded

def test_shootout_respects_p_home():
    rng = np.random.default_rng(0)
    outs = [shootout_winner(rng, p_home=0.9) for _ in range(2000)]
    assert outs.count("home") / 2000 > 0.86
