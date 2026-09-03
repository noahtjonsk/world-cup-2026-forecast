from src.config import load_params

def test_load_params_reads_recency_window(tmp_path):
    p = tmp_path / "params.yaml"
    p.write_text("recency_months: 24\nsources:\n  results: true\n")
    cfg = load_params(p)
    assert cfg["recency_months"] == 24
    assert cfg["sources"]["results"] is True

def test_params_has_models_block():
    cfg = load_params()
    m = cfg["models"]
    assert m["elo_home_advantage"] == 65.0
    assert m["elo_draw_base"] == 0.30
    assert m["dixon_coles"]["max_goals"] == 10
    assert m["dixon_coles"]["half_life_days"] == 730

def test_dixon_coles_has_weighting_and_prior_keys():
    dc = load_params()["models"]["dixon_coles"]
    assert dc["default_competition_weight"] == 0.2
    assert dc["prior_strength"] == 1.5          # locked per reports/goal_backtest_confirm.md (both: 1.5<2.0<1.0)
    assert dc["prior_scale"] == 0.83            # empirical sd(strength) from Task-4 smoke
    cw = {str(k).lower(): w for k, w in dc["competition_weights"]}  # list of [substr, weight]
    assert cw["friendly"] == 0.2
    assert cw["fifa world cup"] == 1.0
    assert cw["qualification"] == 0.8
    assert cw["cecafa"] == 0.3


def test_prior_strength_locked_to_backtest_choice():
    dc = load_params()["models"]["dixon_coles"]
    assert dc["prior_strength"] > 0.0                  # tuned away from no-op
    assert dc["prior_strength"] == 1.5                 # min held-out RPS for `both`, see reports/goal_backtest_confirm.md

def test_dixon_coles_has_squad_coef():
    dc = load_params()["models"]["dixon_coles"]
    assert "squad_coef" in dc
    assert float(dc["squad_coef"]) >= 0.0


def test_dixon_coles_has_elo_anchor_weight():
    dc = load_params()["models"]["dixon_coles"]
    assert "elo_anchor_weight" in dc and float(dc["elo_anchor_weight"]) >= 0.0
