# How the forecast works

This is the reference for the modelling decisions: what the pipeline does, which numbers
were chosen and on what evidence, and where the known approximations are. The README is
the tour; this is the part to read before changing a parameter.

## The pipeline end to end

```
matches.parquet (49,555 historical results)
        |
        v
   Elo ratings  ------------------------+
        |                               |
        v                               v
  Dixon-Coles fit                  Elo anchor  (correction 1)
  competition weighting +               |
  Elo-anchored prior                    v
        |                          squad bump  (correction 2)
        +------------------------------>|
                                        v
                       10,000 tournament simulations
                                        |
                                        v
             simulation_results + bracket_results + reports/simulation.md
```

Four models, each doing one job.

**Elo** gives every team one strength number, updated match by match across the full
history. It is the backbone, and it is also the baseline everything else has to beat.
The K factor varies by competition, so a World Cup match moves a rating six times as far
as a friendly does.

**A CatBoost classifier** predicts win, draw or loss for a single match from a small
feature set: the Elo gap, recent form, rest days, style mismatch and home advantage. The
feature set is deliberately small. International football produces only a few hundred
meaningful matches a year, and a larger model would spend its capacity memorising noise.

**A Dixon-Coles goal model** turns team strengths into a distribution over scorelines,
the probability of 2-1, 0-0 and everything else. That is what the simulator needs in
order to play a match rather than just label it. The fit is a maximum-likelihood estimate
over roughly 730 parameters: two per team, plus home advantage and the low-score
correlation term rho.

**A Monte-Carlo simulator** plays the real 2026 bracket 10,000 times. It samples goals
from the Dixon-Coles model, applies the FIFA tie-breaker chain in the groups, and settles
drawn knockouts with a goal-model extra time at a third of the normal scoring rate,
followed by a coin-flip shootout. Counting how often each team reaches each round
produces the odds.

The CatBoost model is deliberately **not** in the bracket. The simulator runs on the goal
model alone, because it needs scorelines rather than outcome labels. CatBoost serves the
per-match view and the backtests. One consequence is visible in the dashboard: on a close
game the Match page's win/draw/loss numbers and its expected goals come from two
different models and can disagree slightly.

## The two corrections

Between the goal-model fit and the simulation sit two adjustments, both in
`src/simulation/params.py` and applied together by `apply_post_fit_corrections` in
`src/simulation/run.py`. Order matters, and it is Elo anchor first.

### 1. The Elo anchor

A goal model learns from scoring records, and scoring records mislead across
confederations. A team that runs up wins against weak regional opponents looks stronger
than it is, and the goal model cannot see that, because it never observes the
counterfactual. Elo does account for opponent quality, so the fix is to pull each team's
goal-scoring strength toward what its Elo implies.

France is the clean example. On raw scoring records France ranked 10th. On Elo, 3rd. The
anchor moves the simulation toward the Elo view, and the resulting top four now sits in
Elo order.

The shift is split evenly across attack and defence, so a team's balance between the two
survives while its overall level tracks Elo.

### 2. The squad bump

Elo is slow. It records what a country has done, not who it can currently field. The
squad bump adds a measure of present talent, built from club-level per-90 statistics,
weighted by the strength of each player's league, over a role-quota best eleven. A
country fielding a deep roster of top-league players is rated above one that is not.

It is applied second, on top of the anchor. The other order would dilute it, because the
anchor would partly undo the adjustment the bump had just made.

## How parameters get chosen

**By held-out ranked probability score, never by how the title odds look.** That rule
matters more than any individual number in the table below. It is very easy to tune a
football model until the favourites look right and to mistake that for accuracy.

Every backtest is walk-forward: each split trains only on matches that precede the ones
it scores, so no fit ever sees its own test set. Test rows whose team never appears in
training are skipped rather than guessed at.

Where a parameter could not be chosen this way, it is recorded below as a judgment call,
with the reason.

| Parameter | Value | Chosen how |
|---|---|---|
| `prior_strength` (lambda) | 1.5 | Lowest held-out RPS on the confirm grid |
| `prior_scale` (beta) | 0.83 | The empirical standard deviation of fitted team strength |
| `competition_weights` | tiered | World Cup, Euros and Copa at 1.0, qualifiers 0.8, friendlies 0.2 |
| `half_life_days` | 730 | A two-year recency half-life on match weights |
| `elo_anchor_weight` | 0.7 | Judgment, see below |
| `squad_coef` | 0.25 | Judgment, see below |
| `sim_runs` | 10,000 | Enough that the interval on the favourite is a few tenths of a point |
| `sim_jitter` | 0.15 | Per-run perturbation of team strength, so fit uncertainty reaches the odds |

### Why `elo_anchor_weight` is 0.7 and not 1.0

Cross-confederation RPS improved monotonically with the weight, and 1.0 scored best. It
was not taken, because the first split contained only 132 cross-confederation matches,
below the threshold of 150 set before the run. A parameter picked on a sample that thin
is fitting noise, so the value fell back to 0.7. The evidence is in
`reports/elo_anchor_backtest.md`.

### Why `squad_coef` is a judgment call

The squad signal cannot be backtested at all. `player_stats` only covers 2024 to 2026,
which means there is no historical cutoff at which the signal exists but the result is
still unknown. An earlier ablation looked like a null result. It was actually vacuous,
because there was nothing there to measure.

So the coefficient states how much to weight current talent over past results. It is not
a measured optimum. It was set by asking what the bump was worth in Elo terms: one unit
of net strength is worth about 229 Elo points, so a coefficient of 0.4 gave England the
equivalent of a 250-point rating boost. That is not a tilt, that is a second rating
system quietly overruling the first. Reduced to 0.25, worth about 156 Elo points for
England, which is a real but proportionate adjustment.

### A limitation of RPS worth knowing

Ranked probability score over all matches is dominated by within-confederation games,
where goal-model strength and Elo largely agree. It is therefore close to blind to
cross-confederation bias, which is exactly what the Elo anchor exists to fix. That is why
the anchor was calibrated on the cross-confederation subset alone rather than on the full
test set. A parameter that improves overall RPS by nothing can still be correcting a real
error.

## Known approximations

Four places where the model knowingly does something simpler than reality.

**Third-place qualification.** Eight of the twelve third-placed teams advance to the round
of 32. FIFA allocates them using a table keyed on *which* groups the qualifiers came from.
The simulator instead enters them by rank, T1 through T8, into the official third-place
slots. The bracket shape is right; a specific third-placed team's path can be wrong.

**Extra time.** Drawn knockouts play 30 minutes sampled from the same goal model at a
third of the normal scoring rate, then go to a coin-flip shootout. Favourites keep their
edge through extra time and lose it at the shootout, which is roughly what happens. There
is no evidence in this data to justify modelling a shootout as anything but a coin flip.

**In-tournament updates are coarse.** After each result the pipeline recomputes Elo,
refits the goal model and reruns the simulation. It does not refit CatBoost, and it does
not ingest event-level data. Ratings, lineups and summary statistics only.

**Two models on one page.** As above, the Match page draws win/draw/loss from the live
outcome model and expected goals from the corrected goal model. On lopsided fixtures they
agree closely. On coin-flip fixtures they can differ by a few points.

## Validating generated data

The most damaging bug in this project was not in a model. The 2026 group draw and fixture
list were generated programmatically and checked for internal consistency, which they
passed, while being wrong in three separate ways:

- Curacao and Panama were in each other's groups, E and L
- four host fixtures in Groups B and D had home and away reversed, so home advantage was
  applied to the wrong side
- the `hosts` list said `USA`, while every table in the project calls that team
  `United States`, so the host never received home advantage at all

The third one is the instructive one. A name that matches nothing does not raise. It
matches nothing silently, and the code falls through to its default. Nothing failed, no
test went red, and the numbers looked plausible enough to publish.

The rule that came out of it: **structural data must be validated against the
authoritative source, not merely checked for internal consistency.**
`scripts/ingest/regenerate_fixtures.py` now rebuilds fixtures and groups from the
post-draw schedule and hard-fails on the match count, on any group not covering all six
pairings, on any unresolvable team name, and on any date outside the tournament window. A
test pins the host names.

Any odds report produced before that fix used the bad draw.

## File formats

**Parquet for the generated tables.** `src/utils/io.py` is the whole of it, two functions
wrapping `df.to_parquet` and `pd.read_parquet`. The reasons to prefer it over CSV are
size and types: `matches` holds 49,555 rows in 1.3 MB against roughly 6 to 8 MB as CSV,
and date columns load back as dates rather than strings needing re-parsing on every
dashboard page view.

None of these files are in version control. `data/` is ignored, because every table is
reproducible from the scripts. A clone is Python, two config files, three reference CSVs
and the screenshots.

**YAML for configuration.** `config/params.yaml` holds every model parameter in one
place, each with a comment recording why it has that value, so changing a setting never
means hunting through code. `config/tournaments.yaml` holds the bracket structure,
including the official 2026 round-of-32 pairings.

**Three CSVs are committed** under `data/reference/`, because they are hand-maintained
rather than derived: the team-to-confederation map, the league strength multipliers, and
the club-to-league map covering the clubs that World Cup squad players play at.

## Layout

```
src/
  ingest/       source adapters and the canonical schema they write into
  features/     leakage-safe feature construction, all as of kickoff
  models/       Elo, CatBoost W/D/L, Dixon-Coles goals, the Elo prior
  simulation/   the match engine, group standings, bracket, Monte-Carlo
  states/       the after-match update loop
  evaluation/   walk-forward backtests and metrics
  report/       pure presenters, numpy and pandas only, fully unit-tested
app/            thin Streamlit pages that call the presenters and render
scripts/
  ingest/       building the input tables
  calibration/  the parameter-selection experiments behind the table above
  pipeline/     the things you actually run
```

The split between `src/report/` and `app/` is deliberate. The presenters hold all the
logic and are unit-tested; the Streamlit pages only read tables and draw them. That is
why the dashboard has test coverage at all, since Streamlit pages themselves can only be
syntax-checked.

`src/schema.py` is the single source of truth for every table's columns.
