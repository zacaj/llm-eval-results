#!/usr/bin/env python3
"""
Regenerates eval-viz.html from eval-results-per-test.tsv.

Usage:
    python3 generate_viz_data.py

Reads eval-results-per-test.tsv + eval-viz-template.html, writes eval-viz.html.
Re-run any time the tsv changes (new runs added, bad rows deleted, etc.) —
there is no other cache; eval-viz.html is fully derived and safe to overwrite.
"""
import csv
import glob
import math
import json
import re
import sys
from collections import defaultdict

TSV_PATH = 'eval-results-per-test.tsv'
TEMPLATE_PATH = 'eval-viz-template.html'
OUTPUT_PATH = 'eval-viz.html'

# srv.-m gguf filename -> display label. Add an entry here when a new model
# variant shows up in the tsv; anything not listed will raise a KeyError so
# stale/renamed models get noticed rather than silently dropped.
LABELS = {
    'Qwen3.6-35B-A3B-Claude-Opus-Distilled-MTP-UD-IQ3_XXS.gguf': 'Qwen3.6-35B Opus-Distilled IQ3_XXS (main)',
    'Qwen3.6-35B-A3B-UD-IQ3_XXS-MTP.gguf': 'Qwen3.6-35B Base IQ3_XXS',
    'Ornith-1.0-35B-unsloth-UD-IQ3_XXS-MTP-zacaj.gguf': 'Ornith-1.0-35B IQ3_XXS',
    'Qwen3.6-35B-A3B-Claude-Opus-Distilled-MTP-UD-IQ3_XXS.pruned-summary.weighted-n48.gguf': 'Qwen3.6-35B Opus-Distilled IQ3_XXS n48-pruned',
    'Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-UD-IQ4_XS-MTP-mudler.gguf': 'Qwen3.6-35B 4.7-Reasoning-Distilled IQ4_XS',
    'GLM-4.7-Flash-UD-IQ3_XXS-unsloth.gguf': 'GLM-4.7-Flash IQ3_XXS',
    'Darwin-35B-A3B-Opus-unsloth-UD-IQ3_XXS-MTP-zacaj.gguf': 'Darwin-35B IQ3_XXS (unsloth)',
    'Darwin-35B-A3B-IQ3_XXS-Opus-bartowski.gguf': 'Darwin-35B IQ3_XXS (bartowski)',
    'Qwen3.6-35B-A3B-UD-IQ4_XS-MTP-unsloth.gguf': 'Qwen3.6-35B Base IQ4_XS',
    'Qwen3.6-35B-A3B-UD-Q4_K_XL-MTP-unsloth.gguf': 'Qwen3.6-35B Base Q4_K_XL',
    'Qwen3.6-35B-A3B-UD-Q4_K_XL-MTP-unsloth.pruned-Qwen3.6-35B-A3B-Claude-Opus-Distilled-MTP-UD-IQ3_XXS.summary.weighted-n68.gguf': 'Qwen3.6-35B Base Q4_K_XL n68-pruned',
    'Agents-A1-unsloth-UD-IQ3_XXS-MTP-zacaj.gguf': 'Agents-A1 IQ3_XXS',
    'gemma-4-26B-A4B-it-qat-UD-Q4_K_XL-unsloth.gguf': 'Gemma-4-26B QAT Q4_K_XL',
    'ox-alpha': 'Ox Alpha',
    'Qwen3.8-27B-UD-IQ3_XXS-unsloth.gguf': 'Qwen3.8-27B Base IQ3_XXS',
    'Qwen3.8-27B-UD-Q4_K_XL-unsloth.gguf': 'Qwen3.8-27B Base Q4_K_XL',
    'Qwen3.6-27B-UD-IQ3_XXS-MTP-unsloth.gguf': 'Qwen3.6-27B Base IQ3_XXS',
    'Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-Q4_K_XL-MTP-mudler-zacaj.gguf': 'Qwen3.6-35B 4.7-Reasoning-Distilled Q4_K_XL',
    'Darwin-35B-A3B-Opus-Q4_K_XL-MTP-zacaj.gguf': 'Darwin-35B Q4_K_XL',
    'Qwen3.6-35B-A3B-Kimi-K2.6-Reasoning-Distilled-IQ3_XXS-MTP-zacaj.gguf': 'Qwen3.6-35B Kimi-2.6-Reasoning-Distilled IQ3_XXS',
    'Darwin-35B-A3B-Opus-Q8_0-plain.gguf': 'Darwin-35B Q8_0 (plain)',
    'Qwen3.8-27B-IQ3_XXS-GSQ-RCO-MTP-ISNTA_DASLab.gguf': 'Qwen3.8-27B GSQ-RCO IQ3_XXS',
    'Ternary-Bonsai-2-27B-PQ2_0-MTP-test.gguf': 'Ternary-Bonsai-2-27B PQ2_0 (test)',
    'Darwin-28B-Opus-unsloth-UD-IQ3_XXS-MTP-zacaj.gguf': 'Darwin-28B IQ3_XXS (unsloth)',
    'Qwen3.8-27B-requant-like-GSQ-RCO-ISTA-DASLab.gguf': 'Qwen3.8-27B GSQ-RCO requant-like',
    'Darwin-28B-Opus-GSQ-RCO-IQ3_XXS-MTP-zacaj.gguf': 'Darwin-28B GSQ-RCO IQ3_XXS',
    'Qwen3.8-27B-UD-IQ3_XXS-unsloth-ssm_beta_bf16.gguf': 'Qwen3.8-27B Base IQ3_XXS (ssm_beta BF16)',
    'Qwen3.8-27B-IQ3_S-GSQ-RCO-MTP-ISNTA_DASLab.gguf': 'Qwen3.8-27B GSQ-RCO IQ3_S',
    'Qwen3.6-27B-IQ4_XS-uniform-imat-zacaj.gguf': 'Qwen3.6-27B Base IQ4_XS (uniform imat)',
    'Qwen3.8-27B-UD-IQ3_XXS-unsloth-ssm_alpha_beta_bf16.gguf': 'Qwen3.8-27B Base IQ3_XXS (ssm_alpha+beta BF16)',
    'Qwen3.8-27B-UD-IQ3_XXS-unsloth-plan-DASLab-imatrix.gguf': 'Qwen3.8-27B Base IQ3_XXS (DASLab plan+imatrix)',
    'Qwen3.8-27B-hybrid-B-gsq32-63.gguf': 'Qwen3.8-27B hybrid-B (GSQ layers 32-63)',
    'Qwen3.8-27B-hybrid-A-gsq0-31.gguf': 'Qwen3.8-27B hybrid-A (GSQ layers 0-31)',
}

# The task list itself is fully dynamic (see `pins` in build_dataset — every
# (language, exercise) with at least one row, discovered fresh from the tsv each run).
# This is just the seed for the exercise filter's initial selection, so the page opens
# to a small curated comparison instead of every task the harness has ever touched.
# From shortlist-15.txt, originally: fixed (language, exercise) pairs, each pinned to
# ONE language track — must match by (language, exercise), not exercise name alone, the
# same collision the dynamic discovery above also has to account for.
PINS_DEFAULT = [
    ('javascript', 'triangle'), ('javascript', 'grade-school'), ('javascript', 'tournament'),
    ('cpp', 'allergies'), ('cpp', 'sublist'), ('go', 'book-store'), ('javascript', 'wordy'),
    ('go', 'two-bucket'), ('javascript', 'alphametics'), ('javascript', 'promises'),
    ('go', 'bowling'), ('cpp', 'spiral-matrix'), ('go', 'paasio'),
    ('javascript', 'resistor-color-trio'), ('cpp', 'clock'),
]


_warned_unmapped = set()


def shrink_toward_prior(raw, n, prior, strength):
    """Evidence-weighted blend of an observed rate toward a prior belief.

    `n` real observations get full-strength say; the prior contributes as if it were
    `strength` additional observations. As n grows past strength, the result converges
    on `raw`; as n shrinks toward 0, it converges on `prior`. This is the one small-
    sample-correction idea used everywhere in this file — the only thing that changes
    per use is what `prior` should be:
      - per (model, exercise): shrink toward that exercise's own cross-model average
        pass rate (NOT toward 0.5 — a model tested once shouldn't be treated as an
        agnostic coin flip when we already know what a typical model does on this
        exercise; see chart1 below, which replaced an earlier, wrong Wilson-toward-0.5
        version of this exact call after real data showed why 0.5 was the wrong prior).
      - per-exercise flakiness / discriminative: shrink toward the average across every
        OTHER exercise, since most exercises simply aren't flaky/maximally-discriminative
        and a thin sample shouldn't be trusted to say otherwise.
    """
    return (n * raw + strength * prior) / (n + strength)


_SHARD_RE = re.compile(r'-\d{5}-of-\d{5}(?=\.gguf$|$|@)')


def short_label(srv_m_path):
    filename = srv_m_path.split('/')[-1]
    # Split GGUFs are served as 'name-00001-of-00002.gguf'; the shard marker isn't part
    # of the model's identity, so it never reaches LABELS lookups or the fallback label.
    filename = _SHARD_RE.sub('', filename)
    # Non-usual effort/sampling settings are encoded as 'file.gguf@<diffs>' by load_rows().
    filename, _, effort = filename.partition('@')
    effort = effort.replace(',', ', ')
    if effort:
        # '(main)' is load-bearing in the JS (highlights the main row), so it stays off
        # a variant that merely shares the main model's file.
        return short_label(filename).replace(' (main)', '') + f' [{effort}]'
    if filename not in LABELS:
        if filename not in _warned_unmapped:
            _warned_unmapped.add(filename)
            print(f'WARNING: no LABELS entry for {filename!r}; falling back to full filename '
                  f'(add a short label to LABELS in generate_viz_data.py)', file=sys.stderr)
        # Unmapped: show the bare model name, not the file extension.
        return filename.removesuffix('.gguf')
    return LABELS[filename]


# Quant bucket, detected straight from the gguf filename rather than hand-mapped —
# every current variant's name carries its own quant tag, so this stays correct as
# models are added without another table to maintain. Charts fall back to "other"
# (rendered neutral gray) for anything that doesn't match, rather than guessing.
QUANT_PATTERN = re.compile(r'IQ3_XXS|IQ3_S|IQ4_XS|Q4_K_XL|IQ1_M|TQ2_0')


def quant_of(srv_m_path):
    m = QUANT_PATTERN.search(srv_m_path)
    return m.group(0) if m else 'other'


def _norm_num(v):
    """'0.90' and '0.9' are the same setting; non-numeric values pass through."""
    try:
        return f'{float(v):g}'
    except (TypeError, ValueError):
        return v


def behavior_settings(row):
    """The per-run knobs that change how a model behaves (and so which 'model' a row is).

    Temperature is the client-sent value when numeric; 'server' (client sends none, so the
    server's own --temp applies) falls back to the captured srv.--temp. Blank (unrecorded)
    values are dropped.
    """
    temp = row['temperature']
    if not temp or temp == 'server':
        temp = row['srv.--temp']
    settings = {
        'effort': row['reasoning_effort'],  # a bundle: also implies the thinking-token budget
        'temp': _norm_num(temp),
        'top_p': _norm_num(row['srv.--top-p']),
        'top_k': _norm_num(row['srv.--top-k']),
        'min_p': _norm_num(row['srv.--min-p']),
        'dry': _norm_num(row['srv.--dry-multiplier']),
    }
    return {k: v for k, v in settings.items() if v}


def load_rows():
    with open(TSV_PATH) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    # The same gguf can be served from different mount dirs (/models, /models3, ...);
    # keying models by full path made those separate models with identical labels
    # (duplicate heatmap column + filter checkbox). The filename is the identity.
    for row in rows:
        row['srv.-m'] = row['srv.-m'].split('/')[-1]
    # The same gguf run at different reasoning efforts or sampling settings is a
    # different model in practice. Each file's most common settings combo keeps the plain
    # label; any other combo becomes its own model, 'file.gguf@<differences>' listing only
    # the settings that differ from the usual (a model run at a single combo is untouched,
    # whatever it is).
    # Per file, each setting's usual value is its own mode; a blank/unknown value (older
    # rows that predate the capture) is never a difference.
    for row in rows:
        row['_settings'] = behavior_settings(row)
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for row in rows:
        for k, v in row['_settings'].items():
            counts[row['srv.-m']][k][v] += 1
    usual = {m: {k: max(c, key=c.get) for k, c in ks.items()} for m, ks in counts.items()}
    for row in rows:
        row['_diffs'] = tuple((k, v) for k, v in row['_settings'].items()
                              if v != usual[row['srv.-m']][k])
    # Coverage (distinct tasks) per file per combo. A combo with similar coverage to the
    # file's biggest one is a peer, not a stray: peers all get labeled, including the usual
    # combo, so 'medium' doesn't sit unlabeled next to '[low]'. Thin side-runs (< half the
    # biggest combo's tasks) stay as before, the usual combo remaining plain next to them.
    SIMILAR = 0.5
    tasks = defaultdict(lambda: defaultdict(set))
    for row in rows:
        tasks[row['srv.-m']][row['_diffs']].add((row['language'], row['exercise']))
    peer_keys = {}
    for m, combos in tasks.items():
        biggest = max(len(t) for t in combos.values())
        peers = [c for c, t in combos.items() if len(t) >= SIMILAR * biggest]
        if len(peers) > 1:
            peer_keys[m] = {k for c in peers for k, _ in c}
    for row in rows:
        m = row['srv.-m']
        diffs = dict(row['_diffs'])
        keys = peer_keys.get(m, set())
        if keys and len(tasks[m][row['_diffs']]) >= SIMILAR * max(len(t) for t in tasks[m].values()):
            # Spell out every setting that differs among the peers, filling in the usual
            # value where this combo doesn't differ (blank/unrecorded ones are skipped).
            diffs = {k: diffs.get(k, usual[m].get(k)) for k in row['_settings'] if k in keys}
        if diffs:
            # Effort stays bare ('@xhigh') for continuity with older labels.
            row['srv.-m'] = m + '@' + ','.join(v if k == 'effort' else f'{k}={v}'
                                               for k, v in diffs.items() if v)
    # A (model, exercise, seed) triple is one independent sample: re-running the same
    # seed just re-samples the same outcome (greedy/legacy rows have an empty seed and
    # count as one shared seed), so repeat runs must not stack up as extra evidence —
    # 5 identical failures on seed 1 are one failure, not five. Only the latest run per
    # seed is scored; earlier ones are flagged `_superseded` and kept for the tooltip.
    latest = {}
    for row in rows:
        key = (row['srv.-m'], row['language'], row['exercise'], row['seed'])
        if key not in latest or row['timestamp'] >= latest[key]['timestamp']:
            latest[key] = row
    for row in rows:
        row['_superseded'] = latest[(row['srv.-m'], row['language'], row['exercise'], row['seed'])] is not row
    return rows


def build_dataset(rows):
    # The task list is fully dynamic: every (language, exercise) pair with at least one
    # row is "in", discovered fresh from the tsv each run — no hardcoded list to keep in
    # sync as the harness adds new comparison tasks over time. A voided/deleted
    # exercise (every one of its rows removed) automatically drops back out, for the
    # same reason: it just no longer appears in `rows`.
    pins = sorted({(row['language'], row['exercise']) for row in rows})
    pinset = set(pins)
    pin_label = {(l, e): f'{e} ({l})' for l, e in pins}

    # PINS_DEFAULT (the original curated shortlist) no longer scopes what's computed —
    # it only seeds which tasks the exercise filter starts with selected, so the initial
    # view stays a small, curated comparison instead of dumping the full (currently
    # 200+) task list on first load. Defensive same as above: a default pin that's lost
    # all its data is dropped rather than seeding a filter selection for a task that no
    # longer exists.
    default_pins = [p for p in PINS_DEFAULT if p in pinset]

    by_model_ex = defaultdict(set)
    for row in rows:
        # (language, exercise), not exercise name alone — the same collision this file
        # already guards against everywhere else (e.g. "bowling" alone spans 5
        # languages across the full task list). Name-only counting would undercount a
        # model's real distinct-task coverage whenever it hit the same-named exercise
        # in more than one language track.
        by_model_ex[row['srv.-m']].add((row['language'], row['exercise']))
    models_sorted = sorted(by_model_ex, key=lambda m: -len(by_model_ex[m]))
    main = models_sorted[0]  # the model with the most distinct tasks = the full-suite run
    matrix_models = [short_label(m) for m in models_sorted]
    model_quant = {short_label(m): quant_of(m) for m in models_sorted}

    # Every row's (language, exercise) is in `pinset` by construction, so this is just
    # `rows` — kept as its own name since the rest of this function was written around
    # a "shortlist" of rows and renaming every reference below isn't worth the diff.
    shortlist_rows = [row for row in rows if not row['_superseded']]

    # ---- per-test breakdown ----
    # Every run, superseded ones included (flagged) — the hover table shows the full
    # history, but only the latest run per seed feeds any statistic.
    detail = []
    for row in rows:
        detail.append({
            'exercise': row['exercise'], 'language': row['language'], 'model': short_label(row['srv.-m']),
            'run_cmd': row['run_cmd'], 'pass': row['pass'] == 'true', 'tries': row['tries_taken'],
            'duration': round(float(row['duration'] or 0), 1), 'timestamp': row['timestamp'],
            'bailed': row['bailed_on_max_duration'] == 'true',
            'ctx_exhausted': int(row['exhausted_context_windows'] or 0),
            'seed': row['seed'], 'superseded': row['_superseded'],
            'tokens': int(row['prompt_tokens'] or 0) + int(row['completion_tokens'] or 0),
        })
    detail.sort(key=lambda d: (d['exercise'], d['model']))

    # ---- chart3: tries-taken histogram, all models/exercises (not shortlist-scoped) ----
    hist = defaultdict(lambda: {'pass': 0, 'fail': 0})
    for row in rows:
        t = row['tries_taken']
        if t == '':
            continue
        hist[t]['pass' if row['pass'] == 'true' else 'fail'] += 1
    chart3 = [{'tries': k, **v} for k, v in sorted(hist.items(), key=lambda kv: int(kv[0]))]

    # ---- chart4: failure-mode breakdown, all failed runs ----
    fail_rows_all = [row for row in rows if row['pass'] == 'false']
    modes = [
        ('error_outputs', 'Produced error output'),
        ('exhausted_context_windows', 'Exhausted context window'),
        ('user_asks', 'Model asked a clarifying question'),
        ('num_malformed_responses', 'Sent a malformed edit'),
        ('test_timeouts', 'Test run timed out'),
    ]
    chart4 = []
    for col, label in modes:
        nz = sum(1 for row in fail_rows_all if int(row[col] or 0) > 0)
        chart4.append({'label': label, 'count': nz, 'pct': round(nz / len(fail_rows_all) * 100, 1)})
    chart4.sort(key=lambda d: -d['count'])

    # ---- summary / stat tiles ----
    main_full = [row for row in rows if row['srv.-m'] == main and row['run_cmd'] == 'full']
    main_full_pass = sum(1 for row in main_full if row['pass'] == 'true')
    summary = {
        'total_runs': len(rows), 'distinct_tasks': len(pins),
        'total_exercise_names': len(set(row['exercise'] for row in rows)),
        'total_models': len(models_sorted),
        'overall_pass': sum(1 for row in rows if row['pass'] == 'true'),
        'overall_fail': sum(1 for row in rows if row['pass'] == 'false'),
        'total_tasks': len(pins), 'default_tasks': len(default_pins),
        'main_label': short_label(main), 'main_full_pass': main_full_pass, 'main_full_n': len(main_full),
    }

    # ---- perf: generation speed + tokens generated, normalized to exercise difficulty ----
    pass_rows = [row for row in rows
                 if (row['language'], row['exercise']) in pinset and row['pass'] == 'true'
                 and row['duration'] and float(row['duration']) > 0 and row['completion_tokens']]
    by_ex_tokens = defaultdict(list)
    for row in pass_rows:
        by_ex_tokens[pin_label[(row['language'], row['exercise'])]].append(int(row['completion_tokens']))
    ex_baseline = {ex: sum(toks) / len(toks) for ex, toks in by_ex_tokens.items()}

    points = []
    for row in pass_rows:
        ex = pin_label[(row['language'], row['exercise'])]
        tokens = int(row['completion_tokens'])
        points.append({
            'model': short_label(row['srv.-m']), 'exercise': ex, 'tokens': tokens,
            'tokens_norm': round(tokens / ex_baseline[ex], 3),
            'tokps': round(tokens / float(row['duration']), 2),
            'duration': round(float(row['duration']), 1),
            'tries': row['tries_taken'], 'tries_num': int(row['tries_taken']) if row['tries_taken'] else None,
        })

    def agg(rows_subset):
        out = []
        by_model = defaultdict(list)
        for p in rows_subset:
            by_model[p['model']].append(p)
        for m in matrix_models:
            ps = by_model.get(m, [])
            if not ps:
                out.append({'model': m, 'n': 0, 'avg_tokps': None, 'avg_tokens': None, 'avg_tokens_norm': None})
                continue
            out.append({'model': m, 'n': len(ps),
                         'avg_tokps': round(sum(p['tokps'] for p in ps) / len(ps), 1),
                         'avg_tokens': round(sum(p['tokens'] for p in ps) / len(ps)),
                         'avg_tokens_norm': round(sum(p['tokens_norm'] for p in ps) / len(ps), 3)})
        return out

    overall = agg(points)
    by_exercise = {pin_label[(l, e)]: agg([p for p in points if p['exercise'] == pin_label[(l, e)]])
                   for l, e in pins}
    perf = {'matrix_models': matrix_models, 'overall': overall, 'by_exercise': by_exercise,
            'exercise_order': [pin_label[p] for p in pins], 'points': points,
            'ex_baseline': {ex: round(v) for ex, v in ex_baseline.items()}}

    # ---- cost_spread: per-exercise pass-vs-fail cost, tokens AND duration, every attempt ----
    # Unlike `perf` above (passing runs only), this includes failures too — the point is
    # to see how much work gets burned on a run that never lands, and how that compares
    # to what a pass costs on the same exercise.
    # ---- tries-efficiency: how many tries each attempt took, RELATIVE to what other
    # models needed on that same exercise. Separate from the capability score (which is
    # pass/fail only) — a first-try pass on an exercise where everyone else needs 4
    # tries is impressive, on one where everyone needs 2 it's merely fine, and a
    # failure after 5 tries is much less damning on an exercise whose passes also took
    # ~5. Per attempt, on a 0-1 scale (0.75 = "matched the typical model"):
    #   pass at try k, reference mean tries mu: 0.75 + 0.25*tanh((mu - k)/2)  -> (0.5, 1)
    #   fail after k tries:                     0.35 * exp(-max(0, k - mu)/2) -> (0, 0.35]
    # mu = mean of OTHER models' per-model average passing tries on that exercise (each
    # model counts once; the attempt's own model is left out so it can't grade itself),
    # shrunk toward the global typical passing tries when few other models passed.
    EFF_SCALE, EFF_FAIL_CEIL, EFF_MU_PRIOR_STRENGTH = 2.0, 0.35, 2
    pass_tries_by_pin_model = defaultdict(lambda: defaultdict(list))
    for row in shortlist_rows:
        if row['pass'] == 'true' and row['tries_taken']:
            pass_tries_by_pin_model[(row['language'], row['exercise'])][short_label(row['srv.-m'])].append(int(row['tries_taken']))
    pin_model_means = {pin: {m: sum(v) / len(v) for m, v in d.items()}
                       for pin, d in pass_tries_by_pin_model.items()}
    all_means = [x for d in pin_model_means.values() for x in d.values()]
    mu_global = sum(all_means) / len(all_means) if all_means else 2.0

    def attempt_efficiency(row):
        if not row['tries_taken']:
            return None
        pin, model, k = (row['language'], row['exercise']), short_label(row['srv.-m']), int(row['tries_taken'])
        ref = [x for m, x in pin_model_means.get(pin, {}).items() if m != model]
        mu = shrink_toward_prior(sum(ref) / len(ref) if ref else mu_global, len(ref), mu_global, EFF_MU_PRIOR_STRENGTH)
        if row['pass'] == 'true':
            return 0.75 + 0.25 * math.tanh((mu - k) / EFF_SCALE)
        return EFF_FAIL_CEIL * math.exp(-max(0.0, max(k, 1) - mu) / EFF_SCALE)

    # All rows, superseded included: the page re-dedupes per date window, so a run
    # superseded globally can still be the latest one inside a narrower window.
    row_eff = {id(row): attempt_efficiency(row) for row in rows}
    eff_vals = [v for v in row_eff.values() if v is not None]
    eff_global = sum(eff_vals) / len(eff_vals) if eff_vals else 0.75
    eff_by_pin_model = defaultdict(lambda: defaultdict(list))
    for row in shortlist_rows:
        if row_eff[id(row)] is not None:
            eff_by_pin_model[(row['language'], row['exercise'])][short_label(row['srv.-m'])].append(row_eff[id(row)])
    eff_pin_prior = {}
    for pin, d in eff_by_pin_model.items():
        means = [sum(v) / len(v) for v in d.values()]
        eff_pin_prior[pin] = sum(means) / len(means)

    cost_rows = [row for row in rows
                 if row['duration'] and float(row['duration']) > 0 and row['completion_tokens']]
    cost_points = [{
        'exercise': pin_label[(row['language'], row['exercise'])],
        'model': short_label(row['srv.-m']), 'pass': row['pass'] == 'true',
        'tokens': int(row['completion_tokens']), 'duration': round(float(row['duration']), 1),
        'tries': row['tries_taken'], 'timestamp': row['timestamp'], 'seed': row['seed'],
        'eff': round(row_eff[id(row)], 4) if row_eff[id(row)] is not None else None,
    } for row in cost_rows]
    cost_spread = {'points': cost_points}

    # ---- usefulness: per-exercise cost / discriminative-power / flakiness / tries-shape ----
    tries_buckets = ['0', '1', '2', '3', '4', '5', '6', '8', '16']
    usefulness_ex = []
    for lang, ex in pins:
        label = pin_label[(lang, ex)]
        ex_rows = [row for row in shortlist_rows if row['language'] == lang and row['exercise'] == ex]
        if not ex_rows:
            continue
        by_model = defaultdict(list)
        for row in ex_rows:
            by_model[short_label(row['srv.-m'])].append(row)
        model_pass_rates, model_mean_durations, flakiness_scores = [], [], []
        for m in matrix_models:
            rs = by_model.get(m, [])
            if not rs:
                continue
            npass = sum(1 for r_ in rs if r_['pass'] == 'true')
            rate = npass / len(rs)
            # Fractional (share of this model's seeds that passed), not a majority-vote
            # 0/1: a model that passes 1 of 3 seeds is partly able, not simply "failed".
            model_pass_rates.append(rate)
            durs = [float(r_['duration']) for r_ in rs if r_['duration']]
            if durs:
                model_mean_durations.append(sum(durs) / len(durs))
            if len(rs) > 1:
                # NOT shrunk toward 0.5 here: this formula (p(1-p)*4) already REWARDS
                # closeness to 0.5, so shrinking the input rate toward 0.5 would inflate
                # the flakiness of every small-sample exercise, even ones that were
                # perfectly consistent (a 2/2 model would still read as "somewhat
                # flaky" once Wilson refuses to trust the 0% end of the interval). The
                # small-sample problem here is fixed below instead, by shrinking the
                # exercise-level flakiness *result* toward a cross-exercise prior.
                flakiness_scores.append(rate * (1 - rate) * 4)
        p = sum(model_pass_rates) / len(model_pass_rates) if model_pass_rates else 0
        discriminative = p * (1 - p) * 4
        avg_duration = sum(model_mean_durations) / len(model_mean_durations) if model_mean_durations else 0
        flakiness = sum(flakiness_scores) / len(flakiness_scores) if flakiness_scores else None
        hist2 = defaultdict(lambda: {'pass': 0, 'fail': 0})
        for row in ex_rows:
            t = row['tries_taken']
            if not t:
                continue
            hist2[t]['pass' if row['pass'] == 'true' else 'fail'] += 1
        tries_hist = [{'tries': b, 'pass': hist2[b]['pass'], 'fail': hist2[b]['fail']}
                      for b in tries_buckets if (hist2[b]['pass'] + hist2[b]['fail']) > 0]
        # What fraction of this task's failures were the harness bailing on a max
        # duration, rather than the model genuinely giving up/erroring out. Surfaced as
        # its own number rather than folded into `flakiness` — a model that reliably
        # ALWAYS times out on a task isn't flaky (it's consistent), so the two aren't
        # the same phenomenon and conflating them would be a judgment call this file
        # shouldn't make silently. High timeout_rate is a hint the duration budget
        # itself may be too tight for this task, not necessarily that the model (or the
        # task's discriminative signal) can't be trusted.
        fail_rows = [row for row in ex_rows if row['pass'] == 'false']
        timeout_fails = sum(1 for row in fail_rows if row['bailed_on_max_duration'] == 'true')
        timeout_rate = round(timeout_fails / len(fail_rows), 3) if fail_rows else None
        # Discriminative power of TRIES (not pass/fail): how much models' average
        # tries-to-pass actually spreads out on this exercise, vs. pass/fail
        # discriminative's "does everyone land the same way". A tight cluster (everyone
        # takes ~2 tries) scores low even if pass/fail is a 50/50 split; a wide spread
        # (some models 1-try it, others need 5+) scores high even if everyone eventually
        # passes. Coefficient of variation (std/mean) of each model's own average
        # passing-tries, clipped to [0, 1] — unlike p(1-p)*4 this isn't bounded by
        # construction, so the clip is what keeps it comparable to the pass/fail number.
        # None (not 0) when fewer than 2 models have a passing run to compare, since
        # there's no spread to measure yet, not "zero spread".
        tries_means_this = list(pin_model_means.get((lang, ex), {}).values())
        n_tries_models = len(tries_means_this)
        if n_tries_models >= 2:
            mean_t = sum(tries_means_this) / n_tries_models
            std_t = (sum((x - mean_t) ** 2 for x in tries_means_this) / n_tries_models) ** 0.5
            tries_discriminative = min(1.0, std_t / mean_t) if mean_t > 0 else 0.0
        else:
            tries_discriminative = None
        usefulness_ex.append({
            'exercise': ex, 'language': lang, 'label': label, 'n_attempts': len(ex_rows),
            'n_models': len(model_pass_rates), 'models_passed': round(sum(model_pass_rates), 1),
            'eff_across_models': round(eff_pin_prior.get((lang, ex), eff_global), 4),
            'pass_rate_across_models': round(p, 3), 'discriminative': round(discriminative, 3),
            'avg_duration': round(avg_duration, 1),
            'flakiness': round(flakiness, 3) if flakiness is not None else None,
            'n_flaky_models': len(flakiness_scores), 'tries_hist': tries_hist,
            'timeout_fails': timeout_fails, 'timeout_rate': timeout_rate,
            'tries_discriminative': round(tries_discriminative, 3) if tries_discriminative is not None else None,
            'n_tries_models': n_tries_models,
        })

    # Flakiness itself is a mean of per-model scores, most of which come from a tiny
    # number of repeats — an exercise re-run by exactly one model that happened to split
    # 1/2 reports flakiness 1.0 with the same confidence as an exercise re-run by seven
    # models that consistently split ~40/60. Same small-sample problem as everywhere
    # else in this file, just one level up (a rate computed FROM rates rather than from
    # raw pass/fail counts). Fixed by shrinking each exercise's flakiness toward the
    # evidence-weighted average flakiness across every OTHER exercise (most of which
    # land at/near 0 — in practice most exercises simply aren't flaky), with the
    # exercise's own n_flaky_models controlling how much its own evidence is trusted
    # over that prior. FLAKINESS_PRIOR_STRENGTH=3 means "trust an exercise's own
    # flakiness reading about as much as 3 models' worth of prior belief" — an exercise
    # with 0-1 models of repeat evidence stays close to the pool average; one with 7 (like
    # `clock`) is barely pulled at all, since its own evidence dominates a 3-strong prior.
    FLAKINESS_PRIOR_STRENGTH = 3
    evidenced = [d for d in usefulness_ex if d['n_flaky_models'] > 0]
    if evidenced:
        total_n = sum(d['n_flaky_models'] for d in evidenced)
        flakiness_prior = sum(d['flakiness'] * d['n_flaky_models'] for d in evidenced) / total_n
        for d in usefulness_ex:
            if d['n_flaky_models'] > 0:
                d['flakiness'] = round(
                    shrink_toward_prior(d['flakiness'], d['n_flaky_models'], flakiness_prior, FLAKINESS_PRIOR_STRENGTH), 3)

    # `discriminative` has the identical shape (p(1-p)x4) and the identical small-sample
    # risk as flakiness above, just measured across MODELS instead of repeat runs of one
    # model: a pin tested by only 2-3 models that happens to land near a 50/50 split
    # would report "maximally discriminative" with the same false confidence paasio's
    # flakiness did. Every current pin has n_models >= 10 so this isn't visibly broken
    # today the way flakiness was, but it's not structurally protected either — a newly
    # added or not-yet-caught-up-on pin could hit exactly this. Worth fixing proactively
    # since `discriminative` feeds `score_weight` directly (sets how much a pin counts
    # toward the capability score), which is higher-stakes than flakiness (which only
    # discounts a weight that discriminative already set). Same empirical-Bayes shrink,
    # this time weighted by n_models and pulled toward the evidence-weighted average
    # discriminative score across every other pin.
    DISCRIMINATIVE_PRIOR_STRENGTH = 3
    total_nm = sum(d['n_models'] for d in usefulness_ex)
    if total_nm:
        discriminative_prior = sum(d['discriminative'] * d['n_models'] for d in usefulness_ex) / total_nm
        for d in usefulness_ex:
            d['discriminative'] = round(
                shrink_toward_prior(d['discriminative'], d['n_models'], discriminative_prior, DISCRIMINATIVE_PRIOR_STRENGTH), 3)

    # tries_discriminative has the same small-sample exposure as discriminative and
    # flakiness above (a pin with only 2 models' worth of tries data shouldn't report a
    # confident spread), fixed the same way: shrink toward the evidence-weighted average
    # across every pin that has at least 2 models' worth of its own evidence.
    TRIES_DISCRIMINATIVE_PRIOR_STRENGTH = 3
    td_evidenced = [d for d in usefulness_ex if d['tries_discriminative'] is not None]
    if td_evidenced:
        total_ntm = sum(d['n_tries_models'] for d in td_evidenced)
        td_prior = sum(d['tries_discriminative'] * d['n_tries_models'] for d in td_evidenced) / total_ntm
        for d in usefulness_ex:
            if d['tries_discriminative'] is not None:
                d['tries_discriminative'] = round(
                    shrink_toward_prior(d['tries_discriminative'], d['n_tries_models'], td_prior, TRIES_DISCRIMINATIVE_PRIOR_STRENGTH), 3)

    # "Quality" composite: how much a task is worth keeping, folded into one number —
    # rewards discriminative power, penalizes cost (duration) and inconsistency
    # (flakiness). Duration is min-max normalized across the active exercises first
    # since it's in seconds while the other two are already 0-1. An exercise with no
    # repeat runs (flakiness == None) is treated as neutral, not penalized — "unproven"
    # isn't the same claim as "inconsistent".
    durations = [d['avg_duration'] for d in usefulness_ex]
    dur_min, dur_max = min(durations), max(durations)
    dur_span = dur_max - dur_min
    for d in usefulness_ex:
        duration_norm = (d['avg_duration'] - dur_min) / dur_span if dur_span else 0
        flakiness_component = d['flakiness'] if d['flakiness'] is not None else 0
        d['quality'] = round((d['discriminative'] + (1 - duration_norm) + (1 - flakiness_component)) / 3, 3)
        # How much this exercise counts toward the chart1 capability score below —
        # discriminative power x reliability, deliberately excluding cost (unlike
        # `quality` above, which also penalizes slow exercises for the heatmap sort;
        # an exercise being slow doesn't make it less informative about capability).
        d['score_weight'] = round(d['discriminative'] * (1 - flakiness_component), 3)
    usefulness_ex.sort(key=lambda d: -d['discriminative'])

    # ---- chart1: capability score per model — replaces a naive pass-rate-over-attempts.
    # Three problems with "passed / total rows" that this fixes:
    #   1. Repeat runs of the same (model, exercise) pair each counted as a separate
    #      sample, so a model re-run 5x on one easy task outweighed a model tested once
    #      on five different tasks. Fixed by grouping to one k/n per (model, exercise)
    #      first, then shrinking it toward that exercise's own cross-model average pass
    #      rate (impute_by_pin below) with strength MODEL_PIN_PRIOR_STRENGTH — NOT toward
    #      an agnostic 0.5 (that was the first version of this fix, and it was wrong: a
    #      model tested once on an easy exercise everyone else also passes shouldn't be
    #      treated as a 50/50 coin flip just because it lacks repeat evidence. Real data
    #      caught this — a model tested exactly once per exercise had its confident single
    #      passes crushed toward 0.5 while a heavily-repeat-tested model's genuine
    #      weaknesses were trusted at full strength, inverting the fairness this whole
    #      system exists for). More repeats now means more trust in the model's OWN rate
    #      over the pin's typical rate, with diminishing returns either way.
    #   2. A model untested on a hard exercise got it dropped from its own denominator
    #      entirely, inflating its rate relative to a model that attempted the same hard
    #      exercise and failed. Fixed by using the SAME fixed denominator (every pin's
    #      weight) for every model; an untested pin is imputed with that exercise's own
    #      cross-model average pass rate (usefulness_ex's `pass_rate_across_models`) —
    #      a neutral "what a typical model does here" guess, not a free pass or a fail.
    #   3. Every exercise counted equally regardless of how much it actually reveals.
    #      Fixed by weighting each pin's contribution by discriminative power (how much
    #      it splits models — see usefulness_ex above) x reliability (1 - flakiness),
    #      so a consistent, model-splitting exercise like a hard algorithmic task counts
    #      for more than one everybody-passes-it-with-mixed-luck exercise.
    # Ranking uses `projected_pct` — see its own comment further down, right before the
    # sort call, for the full reasoning (it replaced a floor-only ranking, which was
    # itself the fix for a plain-midpoint ranking; both prior versions and why they
    # weren't quite right are documented there).
    weight_by_pin = {(d['language'], d['exercise']): d['score_weight'] for d in usefulness_ex}
    impute_by_pin = {(d['language'], d['exercise']): d['pass_rate_across_models'] for d in usefulness_ex}
    full_weight = sum(weight_by_pin.values())
    # Roughly matches the implicit pseudo-observation count (z^2 ~= 3.84) a Wilson
    # interval would use — same amount of built-in skepticism toward small samples,
    # just redirected at a sensible target instead of 0.5.
    MODEL_PIN_PRIOR_STRENGTH = 4
    # Needed to trust a model's own demonstrated bias (how much it over/under-performs
    # the pool on the pins it HAS been tested on) enough to extrapolate that bias onto
    # its untested pins. Higher than MODEL_PIN_PRIOR_STRENGTH because "this model is
    # systematically better/worse than typical" is a bigger claim than "this model's
    # rate on one specific pin" — it needs evidence spread across several distinct pins,
    # not just repeats of one, before it's trusted over "assume no bias, use the pool
    # average" (bias=0, which is exactly what expected_pct already assumes for everyone).
    BIAS_PRIOR_STRENGTH = 6

    chart1 = []
    for m in models_sorted:
        sub_by_pin = defaultdict(list)
        for row in shortlist_rows:
            if row['srv.-m'] == m:
                sub_by_pin[(row['language'], row['exercise'])].append(row)
        if not sub_by_pin:
            continue

        # Pass 1: per-tested-pin estimate, plus this model's own bias vs the pool on the
        # pins it's actually been tested on (positive = does better than typical there).
        tested_est = {}
        tested = exercises_passed = exercises_failed = exercises_mixed = 0
        bias_num = bias_weight = 0.0
        for lang, ex in pins:
            rs = sub_by_pin.get((lang, ex))
            if not rs:
                continue
            tested += 1
            w = weight_by_pin[(lang, ex)]
            k = sum(1 for r in rs if r['pass'] == 'true')
            n = len(rs)
            est = shrink_toward_prior(k / n, n, impute_by_pin[(lang, ex)], MODEL_PIN_PRIOR_STRENGTH)
            tested_est[(lang, ex)] = est
            if k == n:
                exercises_passed += 1
            elif k == 0:
                exercises_failed += 1
            else:
                exercises_mixed += 1
            bias_num += w * (est - impute_by_pin[(lang, ex)])
            bias_weight += w
        raw_bias = bias_num / bias_weight if bias_weight else 0.0
        bias = shrink_toward_prior(raw_bias, tested, 0.0, BIAS_PRIOR_STRENGTH)

        # Pass 2: assemble floor (untested -> 0), ceiling (untested -> 1), expected
        # (untested -> pool average, no bias adjustment — the plain neutral guess), and
        # projected (untested -> pool average adjusted by this model's own shrunk bias,
        # clamped to a valid probability) across every pin.
        num_expected = num_worst = num_best = num_projected = 0.0
        for lang, ex in pins:
            w = weight_by_pin[(lang, ex)]
            if (lang, ex) in tested_est:
                est = tested_est[(lang, ex)]
                num_expected += w * est
                num_worst += w * est
                num_best += w * est
                num_projected += w * est
            else:
                pool_avg = impute_by_pin[(lang, ex)]
                num_expected += w * pool_avg
                num_best += w * 1.0
                num_projected += w * min(1.0, max(0.0, pool_avg + bias))
        expected = num_expected / full_weight if full_weight else 0
        floor = num_worst / full_weight if full_weight else 0
        ceiling = num_best / full_weight if full_weight else 0
        projected = num_projected / full_weight if full_weight else 0
        pass_tries = [int(row['tries_taken']) for row in shortlist_rows
                      if row['srv.-m'] == m and row['pass'] == 'true' and row['tries_taken']]
        avg_tries = round(sum(pass_tries) / len(pass_tries), 2) if pass_tries else None
        chart1.append({
            'label': short_label(m), 'tested': tested, 'total_pins': len(pins),
            'exercises_passed': exercises_passed, 'exercises_failed': exercises_failed,
            'exercises_mixed': exercises_mixed,
            'projected_pct': round(projected * 100, 1),
            # floor_pct: assumes every untested pin fails; ceiling_pct: assumes every
            # untested pin passes. Both are still shown as the honest bracket around
            # projected_pct — a fully-tested model has floor == projected == ceiling.
            'floor_pct': round(floor * 100, 1), 'expected_pct': round(expected * 100, 1),
            'ceiling_pct': round(ceiling * 100, 1), 'avg_tries_on_pass': avg_tries,
        })
    # Ranked by projected_pct, not floor_pct — see BIAS_PRIOR_STRENGTH above for why
    # projected_pct is safe to rank by where plain expected_pct wasn't: expected_pct
    # assumes every untested pin performs at the POOL's typical rate with full
    # confidence regardless of how little of the model has actually been tested, which
    # let an under-tested model with a small favorable sample float above a much more
    # thoroughly-tested peer (real bug, caught in the previous version of this file).
    # projected_pct only extrapolates a model's OWN demonstrated bias onto its untested
    # pins, and shrinks that bias toward zero (i.e. toward the same pool-typical
    # assumption expected_pct uses) when there isn't much tested evidence to support it
    # — so a model with few tested pins can't confidently extrapolate a large bias, and
    # projected_pct degrades gracefully toward expected_pct's neutral guess rather than
    # floor_pct's hard-zero assumption.
    chart1.sort(key=lambda d: (-d['projected_pct'], d['avg_tries_on_pass'] if d['avg_tries_on_pass'] is not None else 999))

    # ---- matrix: heatmap cells. Rows (exercises) ranked by quality, best first;
    # columns (models) ranked by average normalized token cost, cheapest first. ----
    tokens_norm_by_model = {o['model']: o['avg_tokens_norm'] for o in overall}
    matrix_models = sorted(
        matrix_models,
        key=lambda m: (tokens_norm_by_model[m] is None, tokens_norm_by_model[m] or 0),
    )
    label_to_path = {short_label(m): m for m in models_sorted}
    models_for_matrix = [label_to_path[lbl] for lbl in matrix_models]

    quality_by_pin = {(d['language'], d['exercise']): d['quality'] for d in usefulness_ex}
    pins_by_quality = sorted(pins, key=lambda p: -quality_by_pin.get(p, -1))

    matrix = []
    for lang, ex in pins_by_quality:
        label = pin_label[(lang, ex)]
        cells = []
        for m in models_for_matrix:
            sub = [row for row in shortlist_rows
                   if row['srv.-m'] == m and row['exercise'] == ex and row['language'] == lang]
            if not sub:
                cells.append(None)
                continue
            npass = sum(1 for row in sub if row['pass'] == 'true')
            pass_tries = [int(row['tries_taken']) for row in sub if row['pass'] == 'true' and row['tries_taken']]
            avg_tries = round(sum(pass_tries) / len(pass_tries), 2) if pass_tries else None
            timeouts = sum(1 for row in sub if row['bailed_on_max_duration'] == 'true')
            ctx_runs = sum(1 for row in sub if int(row['exhausted_context_windows'] or 0) > 0)
            cells.append({'pass': npass, 'total': len(sub), 'rate': npass / len(sub), 'avg_tries': avg_tries,
                           'timeouts': timeouts, 'ctx_runs': ctx_runs})
        matrix.append({'exercise': ex, 'language': lang, 'label': label,
                        'quality': quality_by_pin[(lang, ex)], 'cells': cells})

    perf['matrix_models'] = matrix_models
    perf['exercise_order'] = [pin_label[p] for p in pins_by_quality]

    return {
        'summary': summary, 'eff_global': round(eff_global, 4), 'chart1': chart1, 'matrix_models': matrix_models, 'matrix': matrix,
        'model_quant': model_quant, 'chart3': chart3, 'chart4': chart4,
        'fail_n': len(fail_rows_all), 'detail': detail,
        'perf': perf, 'usefulness': {'exercises': usefulness_ex}, 'cost_spread': cost_spread,
        # Labels for the exercise filter's initial selection — see PINS_DEFAULT's
        # comment: it's a seed for the UI now, not a computation scope.
        'default_exercises': [pin_label[p] for p in default_pins],
        'test_lists': load_test_lists(pin_label),
    }


def load_test_lists(pin_label):
    # Every *.txt beside the tsv that looks like a test list: '#' comments, then
    # '<language> <exercise> [max_tries] [max_duration]' whitespace-separated. Offered in
    # the filter panel as quick selections; tests no model has run yet are dropped, and
    # a file with none left over is skipped entirely.
    lists = []
    for path in sorted(glob.glob('*.txt')):
        labels, missing = [], 0
        for line in open(path):
            parts = line.split('#')[0].split()
            if len(parts) < 2:
                continue
            key = (parts[0], parts[1])
            if key in pin_label:
                if pin_label[key] not in labels:
                    labels.append(pin_label[key])
            else:
                missing += 1
        if labels:
            lists.append({'name': path[:-4], 'exercises': labels, 'missing': missing})
    return lists


def main():
    rows = load_rows()
    data = build_dataset(rows)

    template = open(TEMPLATE_PATH).read()
    assert template.count('__DATA__') == 1, 'template must contain exactly one __DATA__ placeholder'
    html = template.replace('__DATA__', json.dumps(data))
    open(OUTPUT_PATH, 'w').write(html)
    print(f'wrote {OUTPUT_PATH} ({len(html):,} bytes) from {len(rows)} rows, '
          f'{data["summary"]["total_tasks"]} tasks total ({data["summary"]["default_tasks"]}/{len(PINS_DEFAULT)} default-shown)')


if __name__ == '__main__':
    main()
