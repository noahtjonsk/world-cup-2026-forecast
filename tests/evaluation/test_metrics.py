# tests/evaluation/test_metrics.py
import math
from src.evaluation.metrics import log_loss, brier_score, rps, calibration_table

def test_log_loss_multiclass():
    # classes order ("H","D","A"); pick the true-class prob for each row
    y = ["H", "A"]
    probs = [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]
    expected = -(math.log(0.7) + math.log(0.6)) / 2
    assert abs(log_loss(y, probs) - expected) < 1e-9

def test_brier_score_multiclass():
    assert abs(brier_score(["H"], [[0.7, 0.2, 0.1]]) - 0.14) < 1e-9        # .3^2+.2^2+.1^2
    assert abs(brier_score(["H", "D"], [[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]])
               - ((0.14 + 0.38) / 2)) < 1e-9                               # row2: .2^2+.5^2+.3^2=.38

def test_rps_ordered_cumulative():
    # cum P=[.7,.9,1]; outcome "H" -> cum O=[1,1,1]; RPS=((.7-1)^2+(.9-1)^2)/(3-1)=.05
    assert abs(rps(["H"], [[0.7, 0.2, 0.1]]) - 0.05) < 1e-9

def test_calibration_table_bins_positive_class():
    probs = [[0.1, 0.45, 0.45], [0.2, 0.4, 0.4], [0.8, 0.1, 0.1], [0.9, 0.05, 0.05]]
    y = ["A", "A", "H", "H"]                       # positive_class defaults to "H"
    tbl = calibration_table(y, probs, n_bins=2)    # edges 0,.5,1
    assert len(tbl) == 2
    assert list(tbl.columns) == ["bin_lower", "bin_upper", "n", "mean_pred", "frac_pos"]
    assert abs(tbl.iloc[0]["mean_pred"] - 0.15) < 1e-9 and tbl.iloc[0]["frac_pos"] == 0.0
    assert abs(tbl.iloc[1]["mean_pred"] - 0.85) < 1e-9 and tbl.iloc[1]["frac_pos"] == 1.0
