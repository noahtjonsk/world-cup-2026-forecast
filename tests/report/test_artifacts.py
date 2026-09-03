def test_artifacts_importable_without_optional_deps():
    # importing must NOT require catboost/scipy/sklearn/matplotlib (all lazy in the body)
    from src.report.artifacts import build_report_artifacts
    assert callable(build_report_artifacts)
