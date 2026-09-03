# Squad-v2 calibration (held-out RPS, league+midfield corrected)

Window: matches since 2023-01-01 | n_splits=3 | coef_grid=(0.0, 0.25, 0.5, 1.0, 2.0)
Per-split test sizes: {0: 863, 1: 886, 2: 881} | skipped (unseen team): {0: 27, 1: 4, 2: 9}
Variant: deployed both (competition weighting + Elo prior) + squad-v2 bump.
Signal: team_squad_strength with league_strength + include_midfield=True

Mean held-out metrics per coef, sorted by RPS (lower = better):

```
           rps  log_loss     brier
coef                              
0.00  0.173917  0.892304  0.526431
0.25  0.173994  0.892648  0.526704
0.50  0.174191  0.893535  0.527254
1.00  0.174848  0.897143  0.528939
2.00  0.176507  0.912972  0.532907
```

## Result: coef=0.0 does NOT beat coef=0 (0.173917)
Selected best RPS: 0.173917 vs incumbent 0.173917

Note: the corrected signal may stay RPS-neutral, RPS is dominated by
within-confederation matches where talent differences matter less. The signal's
value is as a defensible talent input (Portugal > Colombia, Mexico down), not a
guaranteed accuracy win. Do NOT enable in production solely to move Portugal.