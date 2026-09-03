import pandas as pd
from src.report.model_view import metrics_summary, LIMITATIONS, METHODOLOGY


def test_metrics_summary_means_per_model():
    wf = pd.DataFrame([
        {"model": "elo", "split": 0, "log_loss": 1.00, "rps": 0.20, "brier": 0.60},
        {"model": "elo", "split": 1, "log_loss": 1.20, "rps": 0.22, "brier": 0.62},
        {"model": "catboost", "split": 0, "log_loss": 0.90, "rps": 0.18, "brier": 0.55},
        {"model": "catboost", "split": 1, "log_loss": 0.92, "rps": 0.19, "brier": 0.57},
    ])
    out = metrics_summary(wf)
    assert list(out.columns) == ["model", "log_loss", "rps", "brier"]
    idx = out.set_index("model")
    assert abs(idx.loc["elo", "log_loss"] - 1.10) < 1e-9
    assert abs(idx.loc["catboost", "log_loss"] - 0.91) < 1e-9


def test_limitations_cover_the_section_13_pitfalls():
    text = LIMITATIONS.lower()
    for kw in ["underpowered", "predicted xi", "baseline", "coarse", "domain shift"]:
        assert kw in text
    assert "calibration" in METHODOLOGY.lower()
