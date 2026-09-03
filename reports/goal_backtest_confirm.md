# Lambda confirm grid (weighting plus prior)

Window: matches since 2014 | n_splits=3 | beta(prior_scale)=0.83 | lam_grid=(1.0, 1.5, 2.0)
Per-split test sizes: {0: 2836, 1: 2957, 2: 2979} | skipped (unseen team): {0: 146, 1: 24, 2: 2}
Variant: `both` only (competition weighting + Elo prior), the deployed config.

Mean held-out metrics per lambda, sorted by RPS (lower = better):

```
                  rps  log_loss     brier
variant lam                              
both    1.5  0.174688  0.887871  0.523549
        2.0  0.174872  0.888550  0.523994
        1.0  0.175024  0.888872  0.524148
```

## Selection: lambda = 1.5, the lowest mean held-out RPS. Chosen on RPS, never on title odds.

Cross-check: `both` lam=1.5 should reproduce ~0.174688 from reports/goal_backtest.md.