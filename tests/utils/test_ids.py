from src.utils.ids import normalize_name, make_match_id, IDMap

def test_normalize_strips_accents_and_punctuation():
    assert normalize_name("Kylian Mbappé") == "kylian mbappe"
    assert normalize_name("O'Riley, M.") == "o riley m"

def test_match_id_is_deterministic_and_order_sensitive():
    a = make_match_id("2026-06-11", "Brazil", "Serbia")
    b = make_match_id("2026-06-11", "Brazil", "Serbia")
    c = make_match_id("2026-06-11", "Serbia", "Brazil")
    assert a == b and a != c and len(a) == 16

def test_idmap_resolves_by_source_and_by_name():
    m = IDMap()
    m.add("fbref", "abc123", canonical_id="p_mbappe", name="Kylian Mbappé")
    assert m.resolve("fbref", "abc123") == "p_mbappe"
    assert m.resolve_name("kylian mbappe") == "p_mbappe"   # accent-insensitive
    assert m.resolve("understat", "999") is None
