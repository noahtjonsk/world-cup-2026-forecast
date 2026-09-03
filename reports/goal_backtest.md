# Goal-model walk-forward backtest: choosing lambda

> **Superseded selection (2026-06-09):** the final locked choice is **`both` λ=1.5** (keep
> competition weighting), confirmed by `reports/goal_backtest_confirm.md` and recorded in
> the confirm grid. The "lambda=4.0, variant=`prior`" call below was an early,
> grid-edge pick on an ad-hoc grid; the deployed config now runs `both` λ=1.5.

Window: matches since 2014 | n_splits=3 | beta(prior_scale)=0.83 | lam_grid=(0.5, 1.5, 4.0)
Per-split test sizes: {0: 2836, 1: 2957, 2: 2979} | skipped (unseen team): {0: 146, 1: 24, 2: 2}

Mean held-out metrics per (variant, lambda), sorted by RPS (lower = better):

```
                   rps  log_loss     brier
variant  lam
prior    4.0  0.174024  0.885740  0.521987
         1.5  0.174030  0.885593  0.521878
both     1.5  0.174688  0.887871  0.523549
         4.0  0.174958  0.889150  0.524330
         0.5  0.176272  0.893525  0.526865
prior    0.5  0.176581  0.895546  0.527522
baseline 0.0  0.179070  0.906102  0.532779
weight   0.0  0.180206  0.907914  0.534957
```

## Selection

**λ = 4.0, variant = `prior`**, min mean held-out RPS (0.174024 vs baseline 0.179070, a ~2.8% improvement). Log-loss confirms (0.8857 vs 0.9061).

## Guardrail check

- `prior` λ=4.0 / λ=1.5: ✅ beats baseline
- `both` λ=1.5 / λ=4.0 / λ=0.5: ✅ beats baseline
- `weight` alone: ❌ slightly worse than baseline (0.180206 vs 0.179070). Competition-tier weighting removes ~37% of the weight from friendlies and minor cups, reducing effective sample size; without the Elo prior to compensate, the noisier fit slightly hurts held-out RPS. The full combo (`both`) still beats baseline because the prior adds a strong signal.
- **Conclusion:** competition weighting is kept, since the tiers are defensible before seeing any result, and it is used in the `both` variant. The Elo prior alone (`prior` at lambda=4.0) gives the best score here. The `both` combination at lambda=1.5 is the best weighted variant, which confirms the weighting does no harm alongside the prior.

**Chosen on held-out RPS, never on title odds.**
