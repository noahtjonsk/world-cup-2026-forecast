from src.simulation.format import load_tournament_format

def test_load_tournament_format_2026():
    fmt = load_tournament_format("2026")
    assert fmt["n_groups"] == 12 and fmt["group_size"] == 4
    assert fmt["qualifiers_per_group"] == 2 and fmt["third_place_slots"] == 8
    assert fmt["rounds"] == ["R32", "R16", "QF", "SF", "F"]
    assert len(fmt["seeding"]) == 16                       # 32 teams -> 16 R32 matches
    # hosts must use the CANONICAL team names from matches/fixtures ("USA" matched nothing)
    assert fmt["hosts"] == ["United States", "Canada", "Mexico"]
    # official R32 bracket: every token used exactly once (1A-1L, 2A-2L, T1-T8)
    toks = [t for pair in fmt["seeding"] for t in pair]
    assert len(toks) == 32 and len(set(toks)) == 32
    expected = {f"{n}{g}" for n in "12" for g in "ABCDEFGHIJKL"} | {f"T{i}" for i in range(1, 9)}
    assert set(toks) == expected

def test_load_tournament_format_2022():
    fmt = load_tournament_format("2022")
    assert fmt["n_groups"] == 8 and fmt["third_place_slots"] == 0
    assert fmt["rounds"] == ["R16", "QF", "SF", "F"]
    assert len(fmt["seeding"]) == 8                        # 16 teams -> 8 R16 matches
