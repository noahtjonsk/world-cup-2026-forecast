def test_backtest_callable_importable_without_optional_deps():
    from src.simulation.backtest import backtest_tournament
    assert callable(backtest_tournament)
