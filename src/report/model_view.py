import pandas as pd

_METRIC_COLS = ["log_loss", "rps", "brier"]


def metrics_summary(walkforward_df):
    """Mean log-loss, RPS and Brier per model, for the Model page's comparison table.

    Takes walkforward_compare rows of [model, split, log_loss, rps, brier] and
    averages over splits, so CatBoost and the Elo baseline sit side by side."""
    g = walkforward_df.groupby("model")[_METRIC_COLS].mean().reset_index()
    return g.sort_values("model").reset_index(drop=True)[["model"] + _METRIC_COLS]


def calibration_view(calibration_df):
    """Order a calibration_table (evaluation.metrics.calibration_table) by predicted
    probability for plotting [mean_pred vs frac_pos]. Thin pure passthrough."""
    return calibration_df.sort_values("mean_pred").reset_index(drop=True)


METHODOLOGY = (
    "Methodology: Elo prior + CatBoost W/D/L on a small, theory-driven feature set, "
    "with a Dixon-Coles goal model feeding a 48-team Monte-Carlo of the real 2026 "
    "bracket. Player features use a rolling 24-month window; every training row is an "
    "as-of-kickoff snapshot (no leakage). Models are evaluated time-forward (walk-"
    "forward) on log-loss, RPS, Brier and calibration, always against the Elo baseline."
)

LIMITATIONS = (
    "Limitations (read these before trusting any number):\n"
    "- Underpowered data: international history is small (hundreds of matches). Feature "
    "importances are descriptive, not causal; intervals are wide and we do not oversell.\n"
    "- The Elo-only baseline is the bar; we frame success as APPROACHING calibration, "
    "not beating the market. The test set is too small for strong significance claims.\n"
    "- Predicted XI is the biggest uncertainty: it is modelled as a distribution and "
    "propagated through the simulation, not fixed to one lineup.\n"
    "- The 48-team format is novel (out-of-distribution tournament structure); we lean "
    "on team strength rather than format-specific priors.\n"
    "- Club->international domain shift: club-only signal is down-weighted; chemistry "
    "proxies quality and game-state features can leak, so only pre-match versions are used.\n"
    "- Live in-tournament updates are COARSE: Elo + lineups + summary stats only, not "
    "event-level. Forecasts refresh per result; the CatBoost model is reused while the "
    "Dixon-Coles goal model and the Monte-Carlo are rerun each cycle."
)
