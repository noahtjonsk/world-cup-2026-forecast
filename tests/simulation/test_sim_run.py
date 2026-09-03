def test_run_callables_importable_without_optional_deps():
    # importing must NOT require scipy (fit is lazy inside the body)
    from src.simulation.run import run_simulation_pipeline, write_simulation_report
    assert callable(run_simulation_pipeline) and callable(write_simulation_report)

def test_pipeline_accepts_elo_prior():
    import inspect
    from src.simulation.run import run_simulation_pipeline
    assert "elo_prior" in inspect.signature(run_simulation_pipeline).parameters


def test_post_fit_corrections_shared_and_complete():
    # ONE shared implementation of the Elo anchor + corrected squad bump, used by the
    # pipeline, the live runner, and the dashboard generator (parity guard: the runner
    # previously shipped uncorrected expected goals).
    import inspect
    import src.simulation.run as run
    assert callable(run.apply_post_fit_corrections)
    src = inspect.getsource(run.apply_post_fit_corrections)
    for needle in ("elo_anchor_weight", "squad_coef", "include_midfield", "club_league_multiplier"):
        assert needle in src
    # order matters: anchor first, then squad bump on top
    assert src.index("apply_elo_anchor") < src.index("apply_squad_bump")


def test_pipeline_applies_corrections_exactly_once():
    import inspect
    from src.simulation.run import run_simulation_pipeline
    sig = inspect.signature(run_simulation_pipeline).parameters
    assert "params_corrected" in sig and sig["params_corrected"].default is False
    src = inspect.getsource(run_simulation_pipeline)
    assert "apply_post_fit_corrections" in src


def test_live_runner_uses_shared_corrections():
    # the runner must correct params BEFORE computing per-fixture expected goals and
    # must tell the pipeline not to re-apply (double-anchor/double-bump guard)
    import inspect
    from src.states.runner import run_live_update
    src = inspect.getsource(run_live_update)
    assert "apply_post_fit_corrections" in src
    assert "params_corrected=True" in src
