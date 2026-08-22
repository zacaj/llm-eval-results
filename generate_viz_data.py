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
import json
import re
from collections import defaultdict

TSV_PATH = 'eval-results-per-test.tsv'
TEMPLATE_PATH = 'eval-viz-template.html'
OUTPUT_PATH = 'eval-viz.html'

# srv.-m gguf filename -> display label. Add an entry here when a new model
# variant shows up in the tsv; anything not listed will raise a KeyError so
# stale/renamed models get noticed rather than silently dropped.
LABELS = {
    'Qwen3.6-35B-A3B-Claude-Opus-Distilled-MTP-UD-IQ3_XXS.gguf': 'Opus-Distilled IQ3_XXS (main)',
    'Qwen3.6-35B-A3B-UD-IQ3_XXS-MTP.gguf': 'Base IQ3_XXS',
    'Ornith-1.0-35B-unsloth-UD-IQ3_XXS-MTP-zacaj.gguf': 'Ornith IQ3_XXS',
    'Qwen3.6-35B-A3B-Claude-Opus-Distilled-MTP-UD-IQ3_XXS.pruned-summary.weighted-n48.gguf': 'Opus-Distilled n48-pruned',
    'Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-UD-IQ4_XS-MTP-mudler.gguf': '4.7-Reasoning-Distilled IQ4_XS',
    'GLM-4.7-Flash-UD-IQ3_XXS-unsloth.gguf': 'GLM-4.7-Flash IQ3_XXS',
    'Darwin-35B-A3B-Opus-unsloth-UD-IQ3_XXS-MTP-zacaj.gguf': 'Darwin IQ3_XXS (unsloth)',
    'Darwin-35B-A3B-IQ3_XXS-Opus-bartowski.gguf': 'Darwin IQ3_XXS (bartowski)',
    'Qwen3.6-35B-A3B-UD-IQ4_XS-MTP-unsloth.gguf': 'Base IQ4_XS',
    'Qwen3.6-35B-A3B-UD-Q4_K_XL-MTP-unsloth.gguf': 'Base Q4_K_XL',
    'Qwen3.6-35B-A3B-UD-Q4_K_XL-MTP-unsloth.pruned-Qwen3.6-35B-A3B-Claude-Opus-Distilled-MTP-UD-IQ3_XXS.summary.weighted-n68.gguf': 'Q4_K_XL n68-pruned',
    'Agents-A1-unsloth-UD-IQ3_XXS-MTP-zacaj.gguf': 'Agents-A1 IQ3_XXS',
}

# The cross-model comparison shortlist, from shortlist-15.txt: fixed (language, exercise)
# pairs, each pinned to ONE language track. This must match by (language, exercise), not
# exercise name alone — the aider-polyglot suite reuses exercise names across language
# tracks (e.g. "bowling" is a distinct task per language), so name-only grouping silently
# blends unrelated tasks together.
PINS_INTENDED = [
    ('javascript', 'triangle'), ('javascript', 'grade-school'), ('javascript', 'tournament'),
    ('cpp', 'allergies'), ('cpp', 'sublist'), ('go', 'book-store'), ('javascript', 'wordy'),
    ('go', 'two-bucket'), ('javascript', 'alphametics'), ('javascript', 'promises'),
    ('go', 'bowling'), ('cpp', 'spiral-matrix'), ('go', 'paasio'),
    ('javascript', 'resistor-color-trio'), ('cpp', 'clock'),
]


def short_label(srv_m_path):
    return LABELS[srv_m_path.split('/')[-1]]


# Quant bucket, detected straight from the gguf filename rather than hand-mapped —
# every current variant's name carries its own quant tag, so this stays correct as
# models are added without another table to maintain. Charts fall back to "other"
# (rendered neutral gray) for anything that doesn't match, rather than guessing.
QUANT_PATTERN = re.compile(r'IQ3_XXS|IQ4_XS|Q4_K_XL')


def quant_of(srv_m_path):
    m = QUANT_PATTERN.search(srv_m_path)
    return m.group(0) if m else 'other'


def load_rows():
    with open(TSV_PATH) as f:
        return list(csv.DictReader(f, delimiter='\t'))


def build_dataset(rows):
    pin_label = {(l, e): f'{e} ({l})' for l, e in PINS_INTENDED}

    # Defensive: only treat a pinned task as part of the shortlist if it still has at
    # least one row. A fully-deleted exercise (e.g. voided by a harness bug and removed
    # entirely) drops out of every chart instead of rendering as fabricated 0/fail data.
    has_data = {(l, e) for l, e in PINS_INTENDED
                if any(r['language'] == l and r['exercise'] == e for r in rows)}
    pins = [(l, e) for l, e in PINS_INTENDED if (l, e) in has_data]
    pinset = set(pins)

    by_model_ex = defaultdict(set)
    for row in rows:
        by_model_ex[row['srv.-m']].add(row['exercise'])
    models_sorted = sorted(by_model_ex, key=lambda m: -len(by_model_ex[m]))
    main = models_sorted[0]  # the model with the most distinct exercises = the full-suite run
    matrix_models = [short_label(m) for m in models_sorted]
    model_quant = {short_label(m): quant_of(m) for m in models_sorted}

    shortlist_rows = [row for row in rows if (row['language'], row['exercise']) in pinset]

    # ---- chart1: pass rate per model, tie-broken by avg tries-to-pass (fewer = better) ----
    chart1 = []
    for m in models_sorted:
        sub = [row for row in shortlist_rows if row['srv.-m'] == m]
        if not sub:
            continue
        npass = sum(1 for row in sub if row['pass'] == 'true')
        pass_tries = [int(row['tries_taken']) for row in sub if row['pass'] == 'true' and row['tries_taken']]
        avg_tries = round(sum(pass_tries) / len(pass_tries), 2) if pass_tries else None
        chart1.append({'label': short_label(m), 'n': len(sub), 'pass': npass,
                        'rate': round(npass / len(sub) * 100, 1), 'avg_tries_on_pass': avg_tries})
    chart1.sort(key=lambda d: (-d['rate'], d['avg_tries_on_pass'] if d['avg_tries_on_pass'] is not None else 999))

    # ---- per-test breakdown ----
    detail = []
    for row in shortlist_rows:
        detail.append({
            'exercise': row['exercise'], 'language': row['language'], 'model': short_label(row['srv.-m']),
            'run_cmd': row['run_cmd'], 'pass': row['pass'] == 'true', 'tries': row['tries_taken'],
            'duration': round(float(row['duration'] or 0), 1), 'timestamp': row['timestamp'],
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
    distinct_tasks = len(set((row['exercise'], row['language']) for row in rows))
    summary = {
        'total_runs': len(rows), 'distinct_tasks': distinct_tasks,
        'total_exercise_names': len(set(row['exercise'] for row in rows)),
        'total_models': len(models_sorted),
        'overall_pass': sum(1 for row in rows if row['pass'] == 'true'),
        'overall_fail': sum(1 for row in rows if row['pass'] == 'false'),
        'shortlist_n': len(pins), 'shortlist_n_intended': len(PINS_INTENDED),
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
    cost_rows = [row for row in shortlist_rows
                 if row['duration'] and float(row['duration']) > 0 and row['completion_tokens']]
    cost_points = [{
        'exercise': pin_label[(row['language'], row['exercise'])],
        'model': short_label(row['srv.-m']), 'pass': row['pass'] == 'true',
        'tokens': int(row['completion_tokens']), 'duration': round(float(row['duration']), 1),
        'tries': row['tries_taken'],
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
            model_pass_rates.append(1 if rate >= 0.5 else 0)
            durs = [float(r_['duration']) for r_ in rs if r_['duration']]
            if durs:
                model_mean_durations.append(sum(durs) / len(durs))
            if len(rs) > 1:
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
        usefulness_ex.append({
            'exercise': ex, 'language': lang, 'label': label, 'n_attempts': len(ex_rows),
            'n_models': len(model_pass_rates), 'models_passed': sum(model_pass_rates),
            'pass_rate_across_models': round(p, 3), 'discriminative': round(discriminative, 3),
            'avg_duration': round(avg_duration, 1),
            'flakiness': round(flakiness, 3) if flakiness is not None else None,
            'n_flaky_models': len(flakiness_scores), 'tries_hist': tries_hist,
        })

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
    usefulness_ex.sort(key=lambda d: -d['discriminative'])

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
            cells.append({'pass': npass, 'total': len(sub), 'rate': npass / len(sub), 'avg_tries': avg_tries})
        matrix.append({'exercise': ex, 'language': lang, 'label': label,
                        'quality': quality_by_pin[(lang, ex)], 'cells': cells})

    perf['matrix_models'] = matrix_models
    perf['exercise_order'] = [pin_label[p] for p in pins_by_quality]

    return {
        'summary': summary, 'chart1': chart1, 'matrix_models': matrix_models, 'matrix': matrix,
        'model_quant': model_quant, 'chart3': chart3, 'chart4': chart4,
        'fail_n': len(fail_rows_all), 'detail': detail,
        'perf': perf, 'usefulness': {'exercises': usefulness_ex}, 'cost_spread': cost_spread,
    }


def main():
    rows = load_rows()
    data = build_dataset(rows)

    template = open(TEMPLATE_PATH).read()
    assert template.count('__DATA__') == 1, 'template must contain exactly one __DATA__ placeholder'
    html = template.replace('__DATA__', json.dumps(data))
    open(OUTPUT_PATH, 'w').write(html)
    print(f'wrote {OUTPUT_PATH} ({len(html):,} bytes) from {len(rows)} rows, '
          f'{data["summary"]["shortlist_n"]}/{len(PINS_INTENDED)} shortlist tasks active')


if __name__ == '__main__':
    main()
