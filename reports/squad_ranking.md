# Squad strength: corrected metric

As-of: 2026-06-11 | months=24 | league_strength=450 clubs mapped

## Ranking (league + midfield corrected)

| Rank | Team | atk_q | dfc_q | net |
|------|------|-------|-------|-----|
| 1 | England | +1.463 | +1.262 | +2.726 |
| 2 | Germany | +1.285 | +1.244 | +2.529 |
| 3 | Belgium | +1.080 | +1.345 | +2.425 |
| 4 | Spain | +1.039 | +1.327 | +2.366 |
| 5 | Portugal | +1.132 | +1.079 | +2.210 |
| 6 | Netherlands | +0.849 | +1.343 | +2.192 |
| 7 | France | +1.206 | +0.961 | +2.167 |
| 8 | Argentina | +0.854 | +1.031 | +1.885 |
| 9 | Brazil | +1.158 | +0.481 | +1.639 |
| 10 | Croatia | +0.485 | +0.924 | +1.409 |
| 11 | Japan | +0.609 | +0.453 | +1.063 |
| 12 | Senegal | +0.706 | +0.354 | +1.060 |
| 13 | Morocco | +0.130 | +0.343 | +0.473 |
| 14 | Colombia | -0.098 | +0.265 | +0.167 |
| 15 | Mexico | +0.309 | -0.150 | +0.159 |
| 16 | Uruguay | -0.524 | +0.068 | -0.456 |
| 17 | Ecuador | -0.744 | +0.239 | -0.505 |

## Before vs After (key nations)

| Team | Base net | Corrected net | Delta |
|------|----------|---------------|-------|
| Portugal | +1.658 | +2.210 | +0.552 |
| Colombia | +1.677 | +0.167 | -1.510 |
| Mexico | +1.082 | +0.159 | -0.923 |
| France | +2.166 | +2.167 | +0.001 |
| England | +2.040 | +2.726 | +0.686 |
| Spain | +1.767 | +2.366 | +0.599 |
| Brazil | +1.510 | +1.639 | +0.129 |
| Argentina | +1.737 | +1.885 | +0.148 |
| Morocco | +0.635 | +0.473 | -0.162 |
| Japan | +0.676 | +1.063 | +0.386 |

## Ablation: which fix does what

| Team | Base | +League | +Midfield | Both |
|------|------|---------|-----------|------|
| Portugal | +1.658 | +1.978 | +1.716 | +2.210 |
| Colombia | +1.677 | +1.226 | +1.105 | +0.167 |
| Mexico | +1.082 | -0.274 | +2.285 | +0.159 |
| France | +2.166 | +2.771 | +1.459 | +2.167 |
| England | +2.040 | +2.632 | +2.040 | +2.726 |
| Spain | +1.767 | +2.548 | +1.366 | +2.366 |

## Acceptance checks

- Portugal (+2.210) > Colombia (+0.167): **YES**
- Mexico (+0.159) below Portugal/Colombia: **YES**
- Top-league cores at top (France +2.167, Spain +2.366, England +2.726)