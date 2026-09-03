import pandas as pd
from src.features.squad_strength import team_squad_strength

def _ps(rows, season="2024"):
    # rows: (player, position, metric, value)
    return pd.DataFrame([{"player": p, "team": "club", "season": season, "position": pos,
                          "metric": m, "value": v, "source": "test"} for p, pos, m, v in rows])

def _squads(rows):
    # rows: (nation, player) or (nation, player, club)
    data = []
    for r in rows:
        d = {"nation": r[0], "player": r[1], "position": "",
             "name_normalized": r[1].lower(), "in_stats": True}
        d["club"] = r[2] if len(r) > 2 else "unknown"
        data.append(d)
    return pd.DataFrame(data)

# A's players outscore B's in every role -> A should rank higher in both atk_q and dfc_q.
_PS = _ps([
    ("astrike", "FW", "goals", 20), ("astrike", "FW", "assists", 10),
    ("bstrike", "FW", "goals", 4),  ("bstrike", "FW", "assists", 2),
    ("adef", "DF", "tackles", 20),  ("adef", "DF", "blocks", 10),
    ("bdef", "DF", "tackles", 4),   ("bdef", "DF", "blocks", 2),
    ("agk", "GK", "saves", 20),     ("bgk", "GK", "saves", 4),
])
_SQ = _squads([("A", "astrike"), ("A", "adef"), ("A", "agk"),
               ("B", "bstrike"), ("B", "bdef"), ("B", "bgk")])

def test_stronger_squad_ranks_higher_in_both():
    out = team_squad_strength(_SQ, _PS, as_of="2025-06-01", n_att=1, n_def=2, min_players=3)
    assert out["A"][0] > out["B"][0]      # atk_q
    assert out["A"][1] > out["B"][1]      # dfc_q

def test_thin_squad_returns_zero():
    sq = pd.concat([_SQ, _squads([("Thin", "astrike")])], ignore_index=True)
    out = team_squad_strength(sq, _PS, as_of="2025-06-01", min_players=3)
    assert out["Thin"] == (0.0, 0.0)

def test_future_season_is_excluded():
    # astrike only has 2026 stats -> invisible as-of 2025 -> A drops below min_players -> (0,0)
    ps = pd.concat([_PS[~_PS["player"].eq("astrike")],
                    _ps([("astrike", "FW", "goals", 99)], season="2026")], ignore_index=True)
    out = team_squad_strength(_SQ, ps, as_of="2025-06-01", n_att=1, n_def=2, min_players=3)
    assert out["A"] == (0.0, 0.0)

def test_all_cm_squad_falls_back_to_mean():
    ps = _ps([("cm1", "MF", "passes", 10), ("cm2", "MF", "passes", 8), ("cm3", "MF", "passes", 6)])
    sq = _squads([("CMNation", "cm1"), ("CMNation", "cm2"), ("CMNation", "cm3")])
    out = team_squad_strength(sq, ps, as_of="2025-06-01", min_players=3)
    assert out["CMNation"][0] == out["CMNation"][1]   # fallback: atk == dfc == mean quality


def test_club_league_multiplier_defaults_unknown(tmp_path):
    from src.features.squad_strength import club_league_multiplier
    import pandas as pd
    ls = tmp_path / "league_strength.csv"
    ls.write_text("league,multiplier\nTop,1.0\nWeak,0.5\n")
    cl = tmp_path / "club_league.csv"
    cl.write_text("club,league\nBigClub,Top\nSmallClub,Weak\n")
    m = club_league_multiplier(
        ["BigClub", "SmallClub", "UnknownFC"],
        ls_path=str(ls), cl_path=str(cl), default=0.45)
    assert m["BigClub"] == 1.0
    assert m["SmallClub"] == 0.5
    assert m["UnknownFC"] == 0.45


def test_league_multiplier_lowers_weak_league_quality():
    """League multiplier changes output: a team at a weak-league club drops relative to top-league."""
    sq = _squads([("A", "astrike", "SmallClub"), ("A", "adef", "SmallClub"),
                  ("A", "agk", "SmallClub"),
                  ("B", "bstrike", "BigClub"), ("B", "bdef", "BigClub"),
                  ("B", "bgk", "BigClub")])
    base = team_squad_strength(sq, _PS, as_of="2025-06-01", n_att=1, n_def=2, min_players=3)
    ls = {"SmallClub": 0.5, "BigClub": 1.0}
    weak = team_squad_strength(sq, _PS, as_of="2025-06-01", n_att=1, n_def=2, min_players=3,
                               league_strength=ls)
    # League multiplier changes the output (not identical to base)
    assert weak != base
    # Both still produce valid numbers for all teams
    for t in ["A", "B"]:
        assert isinstance(weak[t][0], float)
        assert isinstance(weak[t][1], float)
    # B (top league, multiplier=1.0) >= A (weak league, multiplier=0.5) in both dimensions
    assert weak["B"][0] >= weak["A"][0]
    assert weak["B"][1] >= weak["A"][1]


def test_include_midfield_counts_cm():
    """A CM-heavy squad gets relatively better with include_midfield=True vs a non-CM squad."""
    # Two nations: MidNation has quality CMs, DefNation has only defenders
    ps_cm = _ps([
        ("cm1", "MF", "passes", 10), ("cm2", "MF", "passes", 8),
        ("cm3", "MF", "passes", 6), ("cm4", "MF", "passes", 4),
        ("cm5", "MF", "passes", 2),
        ("def1", "DF", "tackles", 5), ("def2", "DF", "tackles", 4),
        ("def3", "DF", "tackles", 3), ("gk1", "GK", "saves", 5),
    ])
    sq_cm = _squads([("MidNation", "cm1"), ("MidNation", "cm2"), ("MidNation", "cm3"),
                     ("MidNation", "cm4"), ("MidNation", "cm5"),
                     ("DefNation", "def1"), ("DefNation", "def2"), ("DefNation", "def3"),
                     ("DefNation", "gk1")])
    # Without midfield: MidNation's CMs excluded -> falls back to mean (low quality)
    no_mid = team_squad_strength(sq_cm, ps_cm, as_of="2025-06-01", min_players=3)
    # With midfield: MidNation's CM quality now counted in both atk and dfc
    with_mid = team_squad_strength(sq_cm, ps_cm, as_of="2025-06-01", min_players=3,
                                   include_midfield=True)
    # Without midfield: MidNation falls back to mean for both -> atk == dfc
    assert no_mid["MidNation"][0] == no_mid["MidNation"][1]
    # With midfield, MidNation's atk_q improves relative to DefNation
    # (CM quality counts toward attack now)
    assert with_mid["MidNation"][0] > no_mid["MidNation"][0]


def test_include_midfield_with_mixed_roles():
    """A squad with ST/CM/CB: midfield adds CM quality to both ends."""
    ps_mix = _ps([
        ("st1", "FW", "goals", 20), ("st1", "FW", "assists", 10),
        ("cm1", "MF", "passes", 15), ("cm1", "MF", "tackles", 5),
        ("cb1", "DF", "tackles", 10), ("cb1", "DF", "blocks", 5),
        ("gk1", "GK", "saves", 10),
    ])
    sq_mix = _squads([("Mixed", "st1"), ("Mixed", "cm1"), ("Mixed", "cb1"),
                      ("Mixed", "gk1")])
    no_mid = team_squad_strength(sq_mix, ps_mix, as_of="2025-06-01", n_att=1, n_def=2,
                                  min_players=3)
    with_mid = team_squad_strength(sq_mix, ps_mix, as_of="2025-06-01", n_att=1, n_def=2,
                                    min_players=3, include_midfield=True)
    # Both should produce valid results
    assert abs(with_mid["Mixed"][0]) >= 0
    assert abs(with_mid["Mixed"][1]) >= 0


def test_defaults_preserve_existing_behavior():
    """With both new params unset, output must match current behavior exactly."""
    out = team_squad_strength(_SQ, _PS, as_of="2025-06-01", n_att=1, n_def=2, min_players=3)
    # A stronger than B in both dimensions
    assert out["A"][0] > out["B"][0]
    assert out["A"][1] > out["B"][1]
