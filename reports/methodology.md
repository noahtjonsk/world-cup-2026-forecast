# World Cup 2026 forecast: methodology and limitations

Methodology: Elo prior + CatBoost W/D/L on a small, theory-driven feature set, with a Dixon-Coles goal model feeding a 48-team Monte-Carlo of the real 2026 bracket. Player features use a rolling 24-month window; every training row is an as-of-kickoff snapshot (no leakage). Models are evaluated time-forward (walk-forward) on log-loss, RPS, Brier and calibration, always against the Elo baseline.

## Walk-forward metrics (mean over splits)

| model | log_loss | rps | brier |
|---|---|---|---|
| catboost | 0.9160 | 0.1788 | 0.5392 |
| elo | 1.1355 | 0.1802 | 0.5753 |

## Limitations (read these before trusting any number)

- Underpowered data: international history is small (hundreds of matches). Feature importances are descriptive, not causal; intervals are wide and we do not oversell.
- The Elo-only baseline is the bar; we frame success as APPROACHING calibration, not beating the market. The test set is too small for strong significance claims.
- Predicted XI is the biggest uncertainty: it is modelled as a distribution and propagated through the simulation, not fixed to one lineup.
- The 48-team format is novel (out-of-distribution tournament structure); we lean on team strength rather than format-specific priors.
- Club->international domain shift: club-only signal is down-weighted; chemistry proxies quality and game-state features can leak, so only pre-match versions are used.
- Live in-tournament updates are COARSE: Elo + lineups + summary stats only, not event-level. Forecasts refresh per result; the CatBoost model is reused while the Dixon-Coles goal model and the Monte-Carlo are rerun each cycle.