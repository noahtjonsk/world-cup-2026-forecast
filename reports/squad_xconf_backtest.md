# Cross-confederation calibration of the corrected squad signal

> ⚠️ **VACUOUS RESULT, do NOT read this as "squad doesn't help."** `player_stats` only covers 2024 to 2026, so the squad signal is **zero as-of every historical cutoff below** (the bump did nothing → identical RPS across all coefs). The test could not evaluate the signal. The squad bump was enabled as a documented judgment call instead, at coef=0.4 and later reduced to 0.25; see `docs/design.md` for the reasoning.

Window: matches since 2014-01-01 | n_splits=4 | coef_grid=(0.0, 0.25, 0.5, 1.0, 2.0) | elo_anchor_weights=(0.0, 0.7)
Per-split cross-confed test sizes: {0: 461, 1: 263, 2: 416, 3: 399} | min n_cross/split: 263
Signal: team_squad_strength(league_strength=..., include_midfield=True), as-of test-window start.
Roster-proxy caveat: 2026 rosters proxy historical squads (quality leakage-safe as-of date).

Mean cross-confed held-out metrics per (elo_w, coef):

```
                 rps  log_loss     brier
elo_w coef                              
0.0   0.00  0.192820  0.944134  0.555469
      0.25  0.192820  0.944134  0.555469
      0.50  0.192820  0.944134  0.555469
      1.00  0.192820  0.944134  0.555469
      2.00  0.192820  0.944134  0.555469
0.7   0.00  0.190016  0.935145  0.550415
      0.25  0.190016  0.935145  0.550415
      0.50  0.190016  0.935145  0.550415
      1.00  0.190016  0.935145  0.550415
      2.00  0.190016  0.935145  0.550415
```

## Reading
- Plain DC (elo_w=0.0): squad coef=0.0 best vs coef=0 (0.192820), does talent help in isolation?
- **Production (elo_w=0.7, Elo anchor on): baseline RPS=0.190016; best squad coef on top = 0.0** → squad **does NOT add accuracy** on top of the Elo anchor.

## Verdict: the squad bump on top of the Elo anchor does not beat the deployed baseline (smallest cross-confederation sample per split = 263). Judged on cross-confederation RPS, never on title odds.