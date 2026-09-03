def test_runner_importable_without_optional_deps():
    # importing must NOT require scipy/catboost (both lazy inside the body)
    from src.states.runner import run_live_update, COARSE_UPDATE_NOTE
    assert callable(run_live_update)
    assert "coarse" in COARSE_UPDATE_NOTE.lower()
