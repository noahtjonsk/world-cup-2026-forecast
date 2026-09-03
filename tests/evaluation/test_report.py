# tests/evaluation/test_report.py
import src.evaluation.report  # import-smoke: must succeed without catboost/sklearn/matplotlib
from src.evaluation.report import walkforward_compare, plot_calibration


def test_report_callables_importable_without_optional_deps():
    assert callable(walkforward_compare)
    assert callable(plot_calibration)
