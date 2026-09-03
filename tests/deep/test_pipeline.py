def test_pipeline_exposes_expected_callables():
    from src.deep import pipeline
    for fn in ("events_to_spadl", "fit_xt", "rate_xt", "compute_vaep_ratings"):
        assert callable(getattr(pipeline, fn))
