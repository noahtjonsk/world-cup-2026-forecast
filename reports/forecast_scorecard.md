# How the 2026 forecast actually did

What is being scored: the 72 group-stage predictions published on 11 June 2026, before the tournament began, and the tournament odds from the same run. The win/draw/loss probabilities came from the Elo baseline; the expected goals came from the corrected Dixon-Coles model. CatBoost is not scored here, because no CatBoost model was ever published for these fixtures, so the walk-forward figures on the Model page are a separate historical result and do not transfer.

## Match predictions

Over the 72 group-stage matches, against two reference predictors that know nothing about football:

| model | n | log_loss | rps | brier |
|---|---|---|---|---|
| published (Elo baseline) | 72 | 0.901 | 0.1643 | 0.5334 |
| uniform 1/3 | 72 | 1.0986 | 0.2315 | 0.6667 |
| always home | 72 | 1.6308 | 0.3299 | 0.9122 |

Log-loss and RPS are lower-is-better. The two reference rows are there for scale: uniform assigns one third to every outcome, and always-home puts almost everything on the home side. A forecast that cannot beat both has no signal in it.

The forecast beat a predictor that gives every outcome one third by 0.198 in log-loss per match, with a bootstrap 95% interval of 0.070 to 0.321. The interval clears zero, so on this tournament the edge is real rather than a lucky sample. It is a modest edge, and 72 matches is a modest sample, so that is as strong as the claim gets.

The single most likely outcome was correct in 62.5% of matches. That number is easy to read and worth less than the ones above, because it throws away everything the probabilities said about confidence.

## Goals

The expected goals came from the Dixon-Coles model, the same one that drove the simulation.

| measure | value |
|---|---|
| Mean goals predicted per match | 2.5 |
| Mean goals actually scored per match | 2.99 |
| Mean absolute error per team per match | 0.92 |
| Bias per team per match (fewer than actual) | -0.24 |

The model expected fewer goals than the tournament produced, by 0.24 per team per match, which is about half a goal per game.

It also leaned the wrong way on draws, expecting 23.7% against the 27.8% that happened. Those two findings do not obviously share a cause. Fewer goals would normally mean more draws, not fewer, so this is worth noting and watching rather than explaining. One tournament cannot settle it.

How the forecast leaned overall, against what happened:

| outcome | mean predicted % | actually happened % |
|---|---|---|
| home win | 43.2 | 47.2 |
| draw | 23.7 | 27.8 |
| away win | 33.1 | 25.0 |

## Calibration

Predicted probability of a home win against how often the home side actually won, in five buckets.

| bin_lower | bin_upper | n | mean_pred | frac_pos |
|---|---|---|---|---|
| 0.0 | 0.2 | 17.0 | 0.136 | 0.176 |
| 0.2 | 0.4 | 18.0 | 0.307 | 0.389 |
| 0.4 | 0.6 | 17.0 | 0.493 | 0.647 |
| 0.6 | 0.8 | 15.0 | 0.711 | 0.733 |
| 0.8 | 1.0 | 5.0 | 0.843 | 0.4 |

Read the small buckets with care. With 72 matches split five ways, a single result moves a bucket several points.

## Reaching each round

Every team and round from the tournament simulation, bucketed by the probability the model gave, against how often those actually happened.

| predicted probability | cases | mean predicted % | actually happened % |
|---|---|---|---|
| (-0.001, 0.1] | 117 | 2.5 | 0.0 |
| (0.1, 0.25] | 39 | 18.0 | 12.8 |
| (0.25, 0.5] | 31 | 37.1 | 41.9 |
| (0.5, 0.75] | 28 | 63.2 | 75.0 |
| (0.75, 1.0] | 25 | 91.3 | 92.0 |

This is the part that holds up best. Across 240 team-round predictions the observed rates track the predicted ones closely at every level of confidence, which is what calibration is supposed to look like.

## The title odds

Spain won, and was the model's favorite at 25.3%, ranked 1 of 48. Argentina were runners-up. That is one observation and it is worth very little on its own: an outcome given 25% happens about 25% of the time, so a single correct call is not evidence of skill. The 72 match predictions above are.

## What this does not show

The knockout matches are not scored above. No win/draw/loss probabilities were ever published for them, and extra time and shootouts would make the label ambiguous anyway, so they feed the round-reaching comparison instead.

These numbers cannot be compared directly against the walk-forward figures on the Model page. Those measure a different model on a different population of matches, mostly qualifiers and friendlies across many years. A World Cup field is 48 well-covered teams, which Elo rates better than it rates the long tail.

## Worst misses first

Every scored match, ordered by the probability the model gave to the outcome that actually happened. The top of this table is where the model was most wrong.

| date | home_team | away_team | score | predicted | happened | prob given to what happened (%) |
|---|---|---|---|---|---|---|
| 2026-06-15 | Spain | Cape Verde | 0-0 | home win | draw | 8.6 |
| 2026-06-20 | Ecuador | Curaçao | 0-0 | home win | draw | 12.2 |
| 2026-06-23 | England | Ghana | 0-0 | home win | draw | 12.3 |
| 2026-06-24 | South Africa | South Korea | 1-0 | away win | home win | 13.6 |
| 2026-06-17 | Ghana | Panama | 1-0 | away win | home win | 14.1 |
| 2026-06-14 | Ivory Coast | Ecuador | 1-0 | away win | home win | 15.4 |
| 2026-06-12 | Canada | Bosnia and Herzegovina | 1-1 | home win | draw | 18.1 |
| 2026-06-13 | Qatar | Switzerland | 1-1 | away win | draw | 18.3 |
| 2026-06-21 | Uruguay | Cape Verde | 2-2 | home win | draw | 19.8 |
| 2026-06-15 | Saudi Arabia | Uruguay | 1-1 | away win | draw | 20.4 |
| 2026-06-17 | Portugal | DR Congo | 1-1 | home win | draw | 22.8 |
| 2026-06-25 | Japan | Sweden | 1-1 | home win | draw | 23.7 |
| 2026-06-15 | Iran | New Zealand | 2-2 | home win | draw | 25.6 |
| 2026-06-13 | Australia | Turkey | 2-0 | away win | home win | 26.9 |
| 2026-06-15 | Belgium | Egypt | 1-1 | home win | draw | 28.1 |

Full table of all 72 matches is on the Scorecard page of the dashboard.
