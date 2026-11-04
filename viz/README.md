# eval-viz.html

A self-contained dashboard comparing Qwen3.6-35B-A3B quant/prune/distill variants on the
aider-polyglot benchmark. No server, no build step — open the HTML file in a browser.

## Regenerating it

```
python3 generate_viz_data.py
```

Reads `../aider-results-per-test.tsv` + `../harbor-trials.tsv`, writes `eval-viz.html`. Re-run after any change to the
tsv (new eval runs appended, bad rows deleted). `eval-viz.html` is 100% derived output —
never hand-edit it; edit `eval-viz-template.html` (layout/charts) or
`generate_viz_data.py` (data) instead, then regenerate.

No dependencies beyond the Python 3 standard library (`csv`, `json`, `collections`).

## How it's built

```
aider-results-per-test.tsv  ─┐
                             ├─▶  generate_viz_data.py  ─▶  eval-viz.html
eval-viz-template.html     ─┘
```

- **`eval-viz-template.html`** — the real HTML/CSS/JS: every chart, the color system, the
  hover/sort/filter interactivity. Contains one placeholder, `__DATA__`, where the JSON
  payload gets spliced in.
- **`generate_viz_data.py`** — reads the tsv, computes every number the charts need, and
  does `template.replace('__DATA__', json.dumps(data))`. All computation lives here; the
  template only renders.
- **`eval-viz.html`** — the output. Regenerated wholesale each run.

If you're iterating on the data logic, `python3 generate_viz_data.py` and reload the page —
no rebuild step beyond that. If you're iterating on layout/JS, edit
`eval-viz-template.html` directly (or `eval-viz.html` for a one-off visual check), but any
change made only to `eval-viz.html` will be lost the next time the script runs.

## The dataset

`aider-results-per-test.tsv` is one row per aider-polyglot test attempt, across several model
variants and (mostly) 6 language tracks. Key fact about this benchmark: **exercise names
repeat across language tracks** — `bowling` is a distinct task in Python, Go, Java, etc. The
"100 exercise names" in the data actually cover 225 distinct `(language, exercise)` tasks.
Every computation in `generate_viz_data.py` joins on `(language, exercise)`, never on
exercise name alone — grouping by name alone silently blends unrelated tasks together
(this was a real bug caught mid-project; see the `distinct_tasks` vs `total_exercise_names`
split in the stat row).

**The task list is fully dynamic, not a fixed shortlist.** `pins` in `build_dataset()` is
every `(language, exercise)` pair with at least one row, discovered fresh from the tsv on
every regeneration — currently 224 of them. There's no list to keep in sync by hand as the
harness starts running new comparison tasks; a task just needs one row to show up
everywhere (the heatmap, the capability score, the usefulness table), and a task that's
had every one of its rows deleted (e.g. voided by a harness bug) drops back out
automatically, for the same reason — it simply isn't in the tsv to be discovered from
anymore. One model (the "main" one — whichever has run the most distinct tasks, correctly
counted by `(language, exercise)` pair, not exercise name alone, same collision-avoidance
as everywhere else in this file) has been tested against nearly all of them; every other
variant has a partial, uneven slice. `generate_viz_data.py` no longer needs a
"shortlist vs. full suite" split internally at all — that distinction only exists in the
browser now, as the exercise filter's *default selection*.

**`PINS_DEFAULT`** (still the original 15-task pin list, from `shortlist-15.txt`) seeds
which tasks the exercise filter starts with checked — emitted as `data.default_exercises`
— so the page opens to a small, curated, roughly-apples-to-apples comparison instead of
every task the harness has ever touched. It's purely a UI starting point now, not a
computation scope: nothing stops a chart from including a task outside it once the filter
is widened. A default pin that's lost all its data (same voided-task case as above) is
dropped from `default_exercises` the same way it's dropped from `pins`, so it can't seed a
filter selection for a task that no longer exists. `data.summary.total_tasks`
(`len(pins)`) and `data.summary.default_tasks` (`len(default_pins)`) are both emitted for
the stat row and prose; the page's own prose (`<span id="shortlistN1">` and friends near
the top of the JS) is kept live off `activeExercises.size` — re-run every time the filter
changes, not just once at load — since that count can now change at any time, not just
between regenerations.

**Heatmap axes are sorted, not incidental.** Rows (exercises) rank top-to-bottom by a
`quality` composite — the mean of discriminative power, `1 - normalized duration`, and
`1 - flakiness` (flakiness defaults to 0, i.e. "assume consistent," when an exercise has no
repeat runs to judge it by). Columns (models) rank left-to-right by average normalized
token cost, cheapest first. Both scores are computed once in `build_dataset()` and reused —
`quality` is also a sortable column in the usefulness table, and the token ranking is the
same `avg_tokens_norm` the tokens-generated chart uses. A task that looks great on one axis
(e.g. `clock`'s ~0.99 discriminative score) can still rank low overall if it's also
expensive and flaky — the composite is deliberately not just "cost."

**The capability score.** A naive "passed / total attempted" pass rate has three problems
once models are tested unevenly (different coverage of the task list, different numbers of
repeat runs per task) — all three showed up in real data during development, which is why
the score isn't a raw rate:

1. *Repeat runs of the same task shouldn't each count as an independent sample.* A model
   re-run 5x on one easy task would otherwise outweigh a model tested once on five
   different tasks. Fixed by grouping to one k/n per (model, task) pair first, then
   shrinking it (`shrink_toward_prior()` in `generate_viz_data.py`, mirrored as
   `shrinkTowardPrior()` in the template) toward that task's own cross-model average pass
   rate — not toward an agnostic 0.5. **This went through a real, shipped bug**: the first
   version used a [Wilson score interval](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval)
   center, which shrinks small samples toward 0.5. Live data caught why that's wrong — a
   model tested exactly once per task (mostly single confident passes) had those passes
   crushed toward a 50/50 coin flip for "lacking repeat evidence," while a heavily
   repeat-tested model's genuine below-average results on several tasks were trusted at
   full strength, so the repeat-light model ranked *below* one with more, worse-looking
   evidence — backwards from what the score is supposed to measure. Shrinking toward the
   task's own typical pass rate instead means a single pass on a task everyone else also
   passes is trusted close to that task's real difficulty from the start, while a
   repeat-tested model's genuine weak spot still shows up in full once there's enough of
   its own evidence to outweigh the prior (`MODEL_PIN_PRIOR_STRENGTH = 4`, roughly the
   same amount of built-in skepticism a Wilson interval has, just pointed at a sensible
   target instead of 0.5).
2. *An untested task shouldn't just drop out of a model's own denominator.* That silently
   rewards under-testing — a model untested on the two hardest tasks would otherwise look
   better than one that attempted them and failed. Fixed by scoring every model against the
   *same* fixed set of task weights (all 15 pins) and ranking on the **floor**: every
   untested task is assumed to fail. A fully-tested model's floor equals its real weighted
   score exactly, so a coverage gap can never push a model's rank above a more thoroughly-
   tested peer — it can only ever be caught up to once its own real results land, never
   passed on the strength of what's still unknown. `expected_pct` (each untested task
   imputed at that task's own cross-model average pass rate, `pass_rate_across_models` from
   `usefulness_ex` — a neutral "what a typical model does here" guess) and `ceiling_pct`
   (untested assumed to *pass*) are computed alongside and shown in the tooltip/table for
   context, but only the floor decides rank. This was a real bug caught after shipping the
   first version: ranking by the neutral `expected_pct` let a 9/15-tested model with a
   strong-but-small sample float above a 14/15-tested model with more (and slightly worse)
   real evidence, which is exactly the "missing tests get an unfair boost" failure mode this
   whole scoring system exists to prevent. The bar's solid fill is the floor; the faint
   extension past it is the gap up to the ceiling, and collapses to nothing at full
   15/15 coverage.
3. *Not every task is equally informative.* Each task's contribution is weighted by its
   discriminative power (how much it splits models — see `usefulness_ex` below) times its
   reliability (`1 - flakiness`), so a hard, consistently model-splitting task counts for
   more than an easy one every model handles the same way, or a genuinely flaky one whose
   repeat runs disagree with themselves. This weight is `usefulness_ex[].score_weight` —
   shown as its own sortable column ("Score weight") in the exercise usefulness table, so
   the score's weighting is inspectable rather than opaque. It's related to but distinct
   from that same table's `quality` column (which also factors in cost, for ordering the
   heatmap's rows) — a slow-but-discriminating task still counts fully here, since being
   slow doesn't make a task less informative about capability.

   `flakiness` has its own small-sample problem, one level up from the model/exercise
   pairs above: it's a *mean of per-model rates*, and most of those rates come from only
   1-2 repeat runs. An exercise re-run by exactly one model that happened to split 1/2
   reported flakiness 1.0 with the same apparent confidence as an exercise re-run by
   seven models that consistently split ~40/60 — feeding the raw rate through
   `shrink_toward_prior` toward 0.5 wouldn't fix this (it would pull confidently-consistent
   small samples toward 0.5, which the p(1-p) formula would then read as *more* flaky,
   backwards from what's wanted). Instead each exercise's raw flakiness is shrunk toward
   the evidence-weighted average flakiness across every other exercise — in practice, most
   exercises simply aren't flaky, so that average sits low — with `n_flaky_models`
   controlling how much its own reading is trusted over that pooled prior
   (`FLAKINESS_PRIOR_STRENGTH = 3` treats an exercise's own evidence as trustworthy once
   it's backed by about as many models as the prior itself represents). A single-model
   1/2 split now reads around 0.5, not 1.0; a well-evidenced exercise like `clock`
   (7 models' worth of repeat data) barely moves, since its own evidence dominates.

   `discriminative` has the identical shape (`p(1-p)*4`) and the identical risk, just
   measured across *models tested on a pin* (`n_models`) instead of *repeat runs of one
   model*: a pin tested by only 2-3 models landing near a 50/50 split would report
   "maximally discriminative" with the same false confidence a single-model flaky reading
   did. Every current pin has `n_models >= 10`, so this hasn't visibly broken yet the way
   flakiness did, but it isn't structurally protected — a newly added or
   not-yet-caught-up-on pin could hit it directly, and unlike flakiness (which only
   discounts a weight `discriminative` already set), a bad `discriminative` reading would
   corrupt `score_weight` at the source. Shrunk the same way, weighted by `n_models`
   toward the evidence-weighted average `discriminative` across every other pin
   (`DISCRIMINATIVE_PRIOR_STRENGTH = 3`).

Ties are broken by average tries-to-pass, same as before. A mixed-result task counts as an
overall "pass" toward `exercises_passed`/the table view using a `>= 50%` rule (majority of
that model's repeat runs on it succeeded).

**Timeout rate.** `usefulness_ex[].timeout_rate` is the fraction of a task's failures
where `bailed_on_max_duration` was true in the tsv — the harness cut the run off for
running too long, rather than the model erroring out or the run's own logic failing.
Deliberately kept separate from `flakiness`/`quality`/`score_weight` rather than folded
in: a model that reliably times out on a task every single time is *consistent*, not
flaky, so treating "high timeout share" as more inconsistency would be measuring the
wrong thing. It's shown as its own column (with the raw count in parentheses) so a high
value is a visible hint that a task's duration budget may be too tight, worth checking
before trusting its other numbers — not an automatic penalty baked into the score.
Surfaced on the heatmap too: a cell with any timed-out run gets a small corner marker
(hover for exactly which run(s), and how many of that cell's failures were timeouts vs.
genuine), and `detail[].bailed` carries the same flag per individual row for the matrix
tooltip's run-history lines.

**Model labels.** `srv.-m` in the tsv is a full gguf path; `generate_viz_data.py`'s `LABELS`
dict maps the filename to a short display name. Add an entry there when a new model variant
shows up — an unmapped filename raises `KeyError` on purpose, so a new/renamed model gets
noticed instead of silently missing from every chart.

**Quant identity.** Every chart that names a model also shows a small colored dot (bar
labels, heatmap column headers, scatter-point fill, the token strip plot's x-axis markers
and average-dot rings, the per-test table) so quant level reads at a glance without parsing
label text. `quant_of()` in `generate_viz_data.py` regex-matches `IQ3_XXS` / `IQ4_XS` /
`Q4_K_XL` straight out of the gguf filename — no separate table to maintain — and falls back
to a neutral gray "other" for anything that doesn't match. The legend under the page header
(`#quantLegend`) only lists quants actually present in the current data. Colors are the
palette's three CVD-safe categorical slots (`--quant-iq3xxs/-iq4xs/-q4kxl`, defined once at
the top of `eval-viz-template.html` for light/dark); color is always paired with the model
name in text, never the only signal.

### Seeds

A (model, exercise, seed) triple is one independent sample. Only the **latest run per seed**
is scored (rows with an empty seed, i.e. legacy greedy runs, share one seed), so repeating
the same seed can't stack up identical failures. Superseded runs stay in the matrix hover
table, struck through. Per (model, exercise) the pass rate is therefore passing seeds /
seeds, which feeds the capability score continuously. Classification is three-way — **Passed**
(every seed passed), **Mixed** (some did), **Failed** (none did) — instead of a majority vote,
and the cross-model pass rate uses each model's fractional seed pass rate.

**Seed consistency.** The eval stops re-running a test once a seed passes, so a passing cell
with fewer seeds than the maximum has *unknown* seeds, not failed ones. `deriveChart1` fills
those in as passing with probability `q`, the model's seed consistency: among its multi-seed
cells with at least one pass, the pooled share of seeds that passed (all 2/2 -> 1.0, mostly
1/3 -> ~0.33), shrunk toward the same figure across all models (strength 4). Missing seeds
up to the largest seed count seen (3 today) are imputed that way before the usual shrink toward
the exercise's pool rate, so a 1/1 is worth a full 3/3 for a perfectly consistent model and
less for one that usually needs several seeds. All-fail cells are not imputed (failures keep
being retried, so they aren't censored). A model with no multi-seed evidence gets the
all-model figure. The page's tooltip shows each model's `q`. (The static `chart1` in the
JSON is unused by the page and does not include this.)

### Tries score (separate from capability)

The capability score above is pass/fail only. The **Tries** column is a second, independent
0-100 score for how many tries each attempt took *relative to other models on the same
exercise*. Per attempt: a pass at try `k` scores `0.75 + 0.25*tanh((mu - k)/2)` (so 75 =
matched the typical model, ~100 = far fewer tries than others needed, ~50 = far more),
and a failure after `k` tries scores `0.35*exp(-max(0, k - mu)/2)` (at most 35; near 35
when the passes also took about `k` tries, near 0 when most models passed in far fewer).
`mu` is the mean of the *other* models' average passing tries on that exercise (each
model counts once; the attempt's own model is excluded), shrunk toward the global typical
passing tries when few others passed. Per (model, exercise) the attempt scores are
averaged and shrunk toward that exercise's cross-model mean; the model score is the plain
mean over tested exercises, shrunk toward the global mean (strength 3). It's computed
per attempt in `generate_viz_data.py` (`eff` on each `cost_spread` point) and aggregated
live in `deriveChart1`, so it follows the model/exercise/date filters. Because failures
score low it also moves with pass rate; it isn't used for ranking.

## What each chart reads

The JSON payload (embedded in `eval-viz.html` as `<script id="data" type="application/json">`)
has these top-level keys, each consumed by one or more charts:

| Key | Feeds | Notes |
|---|---|---|
| `summary` | stat-tile row, prose spans | overall counts, main model's full-suite pass rate, `total_tasks`/`default_tasks` |
| `default_exercises` | seeds the exercise filter's initial selection | labels for `PINS_DEFAULT`; everything else below is filterable to any subset of `pins`, this is just what starts checked |
| `chart1` | "Capability score by model" bars | filterable to the active task/model selection; ranked by `projected_pct`, not a raw pass rate — see "The capability score" below |
| `perf` | "Generation speed", "Tokens generated", "Quality vs. token cost" | `perf.points` = raw per-run data; `perf.overall`/`perf.by_exercise` = pre-aggregated per model; `tokens_norm` = tokens ÷ that exercise's own mean (only among models that passed it) — the "fair" comparison basis, default view |
| `matrix` / `matrix_models` | "Pass/fail by task × model" heatmap | one row per task with data (currently up to `total_tasks`, filtered down to the active exercise selection for display); a `null` cell = no data for that (model, task) pair, rendered blank, never red; rows ranked by `quality`, columns by the sort option chosen (avg token cost or projected capability); `cell.timeouts` drives the corner marker on cells with a timed-out run |
| `cost_spread` | "Pass vs. fail cost, per exercise" | unlike `perf`, includes *failed* runs too — `cost_spread.points` is every attempt (pass or fail) with both `tokens` and `duration`, so the chart can toggle metric client-side and compute the pass/fail averages and their gap per exercise on the fly |
| `detail` | "Per-test breakdown" table, matrix tooltip's run history | every individual row (respecting the active filter), sortable client-side; `bailed` flags a run that hit the harness's max-duration limit |
| `usefulness` | "Exercise usefulness" scatter + table | `discriminative = p·(1-p)·4` where `p` = fraction of models that passed (peaks at an even split, zero when everyone agrees); `flakiness` = same formula applied within a model's repeated runs, `null` when no task/model pair has more than one run to judge consistency from; `timeout_rate` = fraction of a task's failures that were timeouts rather than genuine failures — see "Timeout rate" below |
| `chart3` | "Tries taken" stacked columns | **not** filterable — every run in the tsv, regardless of the active model/exercise selection |
| `chart4` / `fail_n` | "What failed runs had in common" bars | **not** filterable — every failed run in the tsv, regardless of the active model/exercise selection |

## The floating model/exercise filter

Every filterable chart from "Capability score by model" through "Per-test breakdown"
responds to a global filter panel (the "Filter" button, top-right) instead of reading
`data.*` directly. On every filter change, `applyFilters()` in the template:

1. Filters `data.cost_spread.points` (every attempt, pass or fail, across every task
   `pins` discovered) down to the active models/exercises/date range — this is the one
   raw source with enough fields (`model`, `exercise`, `pass`, `tokens`, `duration`,
   `tries`, `timestamp`) to recompute everything below it. The date range is a
   dual-handle slider (two overlapping `<input type="range">`s) bounded to the actual
   min/max timestamp in the data — it filters on *when the eval ran*, not anything about
   the exercise itself.
2. Re-derives `chart1` (capability score), `perf.points`/`perf.overall`-equivalents
   (tokens/duration/tries panels, speed, quality-vs-tokens), and the heatmap's
   cells/columns — all as small JS functions (`deriveChart1`, `derivePerfPoints`,
   `derivePerfOverall`, `deriveMatrixFiltered`) mirroring the same logic
   `generate_viz_data.py` uses, just scoped to whatever's currently checked.
3. Re-invokes each chart's render function with the freshly derived data.

**Exercise checklist: search, sort, and count**, since the exercise list is the full
dynamic task set (currently 200+), not a fixed 15 — a plain unsorted checkbox list at
that size stops being navigable. `renderExerciseList()` in the template filters by the
search box (substring match against the label) and re-orders the *displayed* list by
`quality` (`data.matrix`'s own default order — no re-sort needed), alphabetically, or by
`n_models` (from `data.usefulness.exercises`, "most tested"); none of that touches the
underlying data or any other chart's ordering, it's purely how the checklist itself is
browsed. `(N/M)` counts sit next to both the "Models" and "Exercises" section headers,
live off `activeModels.size`/`activeExercises.size`. A **Default** button (next to
All/None) resets the exercise selection back to `data.default_exercises` in one click.

**Two deliberate scope limits**, both stated in the filter panel's own copy so they're
never a silent surprise:
- **Tries-taken and failure-mode charts (the last two on the page) ignore the filter.**
  They're explicitly global/unfilterable already (see the table above) — they'd need a
  redesign to mean "filtered by the active model/exercise selection" in the first place.
- **Exercise usefulness (discriminative/flakiness/quality) only responds to the
  exercise filter, not the model filter.** Those numbers are properties of the task
  across the *whole* model field by design (e.g. "does this task split models 50/50"
  only makes sense against the full roster) — filtering rows out from `data.usefulness.
  exercises` is a checkbox click away from being correct if you ever want true
  recomputation, but it wasn't worth porting the discriminative/flakiness formulas to
  JS twice for a metric whose whole point is "how models generally behave," not "how
  the models I've currently got checked behave."

If you add a chart that should respect the filter, give its render function a plain
data parameter (not a `data.*` read) and call it from `applyFilters()` alongside the
others — every existing render function already follows that pattern.

**Persistence.** The selection is saved to `localStorage` (key `eval-viz-filters-v2`)
every time `applyFilters()` runs, and restored once at page load before the panel is
built. Models are stored as a *diff from "everything selected"* (which are deselected,
not which are selected) — unchanged from before, since models still default to
all-on, so a model variant added between sessions defaults to selected rather than
silently vanishing from a stale allow-list. **Exercises are stored as a diff from
`data.default_exercises` instead** (which are added beyond it, which are removed from
it) — since exercises default to a small curated subset, not everything, "deselected
from everything" wouldn't mean anything here; storing both directions means a task the
harness only just started running still shows up correctly unchecked (it was never in
the default, and nobody explicitly added it) without needing to be listed anywhere in
this saved state at all. The date range is stored as absolute timestamps normally, but
as `null`/`null` when it's the full unrestricted range, so "all time" stays "all time"
even as new runs extend the actual max timestamp in later sessions — a literal saved
range would otherwise silently clip out new data. (The storage key moved from `-v1` to
`-v2` when the exercise scheme changed to add/remove; old saved selections are just
dropped once, falling back to the new default — reasonable, since the exercise universe
itself grew from 15 tasks to 200+ in that same change, so old data barely applied
anyway.)

## Extending it

- **New eval data lands / bad rows get deleted:** just re-run the script. New tasks show
  up automatically (unchecked by default, unless they happen to be in `PINS_DEFAULT`);
  deleted tasks disappear automatically. Nothing else needed.
- **New model variant:** add its gguf filename → label to `LABELS` in
  `generate_viz_data.py`.
- **Change what the exercise filter starts with checked:** edit `PINS_DEFAULT`. This no
  longer scopes any computation (`pins` is always every task with data) — it only
  changes `data.default_exercises`, i.e. the filter's initial selection.
- **New chart:** add a card + render function to `eval-viz-template.html` (follow the
  existing pattern — an SVG built from a JS template string, a `showTip`/`hideTip` hover
  handler, CSS custom properties for light/dark), and whatever new data it needs to
  `build_dataset()` in `generate_viz_data.py`.
- **Color/theme:** all colors are CSS custom properties on `.viz-root`, duplicated for
  light (`:root`) and dark (`prefers-color-scheme` + `[data-theme="dark"]`) — see the
  `<style>` block at the top of the template.
