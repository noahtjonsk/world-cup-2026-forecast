# Squad-coefficient calibration (held-out RPS, recent window)

Window: matches since 2023-01-01 | n_splits=3 | coef_grid=(0.0, 0.25, 0.5, 1.0, 2.0)
Per-split test sizes: {0: 863, 1: 886, 2: 881} | skipped (unseen team): {0: 27, 1: 4, 2: 9}
Variant: deployed `both` (competition weighting + Elo prior lam=1.5) + squad bump.
Limitations: 2026-roster proxy for historical squads; signal affects WC teams only.

Mean held-out metrics per coef, sorted by RPS (lower = better):

```
           rps  log_loss     brier
coef                              
0.00  0.173917  0.892304  0.526431
0.25  0.173963  0.892429  0.526601
0.50  0.174169  0.893141  0.527103
1.00  0.174901  0.896330  0.528768
2.00  0.176566  0.910296  0.532514
```

## Selection: coef = 0.0, since no non-zero coefficient beats the 0.173917 baseline. Chosen on held-out RPS, never on title odds.