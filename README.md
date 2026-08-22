# eval-viz.html

A self-contained dashboard comparing Qwen3.6-35B-A3B quant/prune/distill variants on the
aider-polyglot benchmark. No server, no build step — open the HTML file in a browser.

## Regenerating it

```
python3 generate_viz_data.py
```

Reads `eval-results-per-test.tsv`, writes `eval-viz.html`. Re-run after any change to the
tsv (new eval runs appended, bad rows deleted). `eval-viz.html` is 100% derived output —
never hand-edit it; edit `eval-viz-template.html` (layout/charts) or
`generate_viz_data.py` (data) instead, then regenerate.

No dependencies beyond the Python 3 standard library (`csv`, `json`, `collections`).

## How it's built

```
eval-results-per-test.tsv  ─┐
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

`eval-results-per-test.tsv` is one row per aider-polyglot test attempt, across several model
variants and (mostly) 6 language tracks. Key fact about this benchmark: **exercise names
repeat across language tracks** — `bowling` is a distinct task in Python, Go, Java, etc. The
"100 exercise names" in the data actually cover 225 distinct `(language, exercise)` tasks.
Every computation in `generate_viz_data.py` joins on `(language, exercise)`, never on
exercise name alone — grouping by name alone silently blends unrelated tasks together
(this was a real bug caught mid-project; see the `distinct_tasks` vs `total_exercise_names`
split in the stat row).

**The shortlist.** Only one model (the "main" one — whichever ran the most distinct
exercises, i.e. the full ~225-task suite) has been tested on everything. The other ten
variants were each run against a fixed 15-task subset (`shortlist-15.txt`, pinned
`(language, exercise)` pairs) for a controlled, apples-to-apples comparison. Every chart
except the two "global" ones (tries-taken histogram, failure-mode breakdown, which use the
full dataset) is scoped to this shortlist — that's the only slice where every model has
directly comparable data.

**Missing exercises are dropped, not faked.** If every row for a shortlist task gets
deleted (e.g. a harness bug voided its results and someone cleaned up the tsv),
`generate_viz_data.py` excludes that task from the shortlist entirely — it won't appear in
the heatmap, the usefulness scatter, or count toward `shortlist_n`. The alternative (letting
it default to 0% pass / 0s duration) would be indistinguishable from "every model tried and
instantly failed," which is a worse lie than just not knowing. `shortlist_n` reflects
*active* tasks; `shortlist_n_intended` (currently 15) is the original pin count, kept around
mostly for sanity-checking how much has been pruned. The page's own prose ("N-task
shortlist", "N failed runs", etc.) is templated from `data.summary` at load time
(`<span id="shortlistN1">` and friends near the top of the JS), not hardcoded — so pruning
more rows won't leave stale numbers in the descriptions.

**Heatmap axes are sorted, not incidental.** Rows (exercises) rank top-to-bottom by a
`quality` composite — the mean of discriminative power, `1 - normalized duration`, and
`1 - flakiness` (flakiness defaults to 0, i.e. "assume consistent," when an exercise has no
repeat runs to judge it by). Columns (models) rank left-to-right by average normalized
token cost, cheapest first. Both scores are computed once in `build_dataset()` and reused —
`quality` is also a sortable column in the usefulness table, and the token ranking is the
same `avg_tokens_norm` the tokens-generated chart uses. A task that looks great on one axis
(e.g. `clock`'s ~0.99 discriminative score) can still rank low overall if it's also
expensive and flaky — the composite is deliberately not just "cost."

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

## What each chart reads

The JSON payload (embedded in `eval-viz.html` as `<script id="data" type="application/json">`)
has these top-level keys, each consumed by one or more charts:

| Key | Feeds | Notes |
|---|---|---|
| `summary` | stat-tile row, prose spans | overall counts, main model's full-suite pass rate |
| `chart1` | "Pass rate by model" bars | shortlist-scoped; ties broken by avg tries-to-pass |
| `perf` | "Generation speed", "Tokens generated", "Quality vs. token cost" | `perf.points` = raw per-run data; `perf.overall`/`perf.by_exercise` = pre-aggregated per model; `tokens_norm` = tokens ÷ that exercise's own mean (only among models that passed it) — the "fair" comparison basis, default view |
| `matrix` / `matrix_models` | "Pass/fail by task × model" heatmap | one row per active shortlist task; a `null` cell = no data for that (model, task) pair, rendered blank, never red; rows ranked by `quality`, columns by avg token cost |
| `cost_spread` | "Pass vs. fail cost, per exercise" | unlike `perf`, includes *failed* runs too — `cost_spread.points` is every shortlist attempt (pass or fail) with both `tokens` and `duration`, so the chart can toggle metric client-side and compute the pass/fail averages and their gap per exercise on the fly |
| `detail` | "Per-test breakdown" table | every individual shortlist row, sortable client-side |
| `usefulness` | "Exercise usefulness" scatter + table | `discriminative = p·(1-p)·4` where `p` = fraction of models that passed (peaks at an even split, zero when everyone agrees); `flakiness` = same formula applied within a model's repeated runs, `null` when no task/model pair has more than one run to judge consistency from |
| `chart3` | "Tries taken" stacked columns | **not** shortlist-scoped — every run in the tsv |
| `chart4` / `fail_n` | "What failed runs had in common" bars | **not** shortlist-scoped — every failed run in the tsv |

## Extending it

- **New eval data lands / bad rows get deleted:** just re-run the script. Nothing else.
- **New model variant:** add its gguf filename → label to `LABELS` in
  `generate_viz_data.py`.
- **Change the shortlist:** edit `PINS_INTENDED`. Everything downstream (chart1, matrix,
  perf, usefulness) recomputes from it automatically.
- **New chart:** add a card + render function to `eval-viz-template.html` (follow the
  existing pattern — an SVG built from a JS template string, a `showTip`/`hideTip` hover
  handler, CSS custom properties for light/dark), and whatever new data it needs to
  `build_dataset()` in `generate_viz_data.py`.
- **Color/theme:** all colors are CSS custom properties on `.viz-root`, duplicated for
  light (`:root`) and dark (`prefers-color-scheme` + `[data-theme="dark"]`) — see the
  `<style>` block at the top of the template.
