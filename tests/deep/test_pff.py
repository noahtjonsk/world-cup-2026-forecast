def test_pff_exposes_loaders():
    from src.deep import pff
    assert callable(pff.load_pff_tracking)
    assert callable(pff.pff_events_to_spadl)
