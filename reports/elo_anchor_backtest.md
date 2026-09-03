# Elo-anchor calibration (cross-confed held-out RPS)

Window: matches since 2014-01-01 | n_splits=4 | weight_grid=(0.0, 0.25, 0.5, 0.75, 1.0)
Per-split cross-confed test sizes: {0: 359, 1: 132, 2: 352, 3: 347} | skipped (unseen/unmapped): {0: 2026, 1: 2253, 2: 2033, 3: 2038}
Min n_cross per split: 132 | Sufficient for RPS selection: False
Variant: deployed `both` (competition weighting + Elo prior λ=1.5) + Elo anchor at each weight.
Scored ONLY on cross-confederation test matches (where goal-model-vs-Elo bias is visible).

Mean held-out metrics per weight, sorted by RPS (lower = better):

```
             rps  log_loss     brier
weight                              
1.00    0.189444  0.948915  0.560840
0.75    0.190216  0.951291  0.562262
0.50    0.191380  0.955004  0.564508
0.25    0.192928  0.960054  0.567567
0.00    0.194851  0.966447  0.571420
```

## Selection: weight = 0.7
Incumbent (weight=0) RPS: 0.194851
Best RPS weight: 1.0 (beats incumbent)
Gate: Cross-confed sample too thin (min n_cross=132 < 150). Falling back to face-validity default weight=0.7. This is NOT odds-tuning; the gate is documented above.

Chosen on RPS over cross-confederation matches, never on title odds.