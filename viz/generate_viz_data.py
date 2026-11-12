#!/usr/bin/env python3
"""
Regenerates eval-viz.html from aider-results-per-test.tsv.

Usage:
    python3 generate_viz_data.py

Reads aider-results-per-test.tsv + eval-viz-template.html, writes eval-viz.html.

Each runner's rows go through an adapter (aider_attempt / harbor_attempt) into common ATTEMPT
records (see attempt fields there); build_dataset only reads those, never raw TSV
columns, so another runner (Harbor, viz-harbor-plan.md) only needs its own adapter.
End reason / flags per attempt come from outcomes.py.
Re-run any time the tsv changes (new runs added, bad rows deleted, etc.) —
there is no other cache; eval-viz.html is fully derived and safe to overwrite.
"""
import csv
import glob
import math
import os
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

import outcomes

TSV_PATH = '../aider-results-per-test.tsv'  # run from viz/ (run.sh / collect.py do)
HARBOR_TSV_PATH = '../harbor-trials.tsv'  # harbor/collect.py
HARBOR_REQUESTS_PATH = '../harbor-requests.tsv'  # harbor/collect.py: one row per model request
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

# The task list itself is fully dynamic (see `pins` in build_dataset — every task with
# at least one row, discovered fresh from the data each run). The exercise filter starts
# with the DEFAULT_TOP_N exercises the most distinct models have run, every runner variant
# of each, so the page opens to the best-compared tasks (and their Harbor counterparts)
# and keeps up as testing moves on, instead of a hand-kept list going stale.
DEFAULT_TOP_N = 20

# Short runner code appended to every task label, so no runner is the implicit one.
VARIANT_CODES = {'aider': 'A', 'harbor': 'H', 'harbor-tc': 'HT'}


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
# models are added without another table to maintain. Any llama.cpp-style tag matches
# (IQ3_XXS, Q8_0, TQ3_0, BF16...), each getting its own filter group and color; only
# names with no tag at all fall back to "other" (neutral gray).
QUANT_PATTERN = re.compile(r'(?<![A-Za-z0-9])(?:IQ|TQ|PQ|MXFP|BF|Q|F)\d+(?:_[A-Z0-9]+)*(?![A-Za-z0-9])')


def quant_of(srv_m_path):
    m = QUANT_PATTERN.search(srv_m_path)
    return m.group(0) if m else 'other'


def _norm_num(v):
    """'0.90' and '0.9' are the same setting; non-numeric values pass through."""
    try:
        return f'{float(v):g}'
    except (TypeError, ValueError):
        return v


# run.sh's REASONING_EFFORT bundle: the thinking cap each effort sends per request. Rows
# from before run.sh sent one have srv.--reasoning-budget blank; they ran at the low
# baseline's 4k (run.sh comment), which this table also gives them.
AIDER_EFFORT_THINK = {'low': '4096', 'medium': '8192', 'xhigh': '16384', 'high': '16384'}


def norm_think(v):
    """Thinking cap rounded to thousands ('4k'): aider's low sends 4096, Harbor's proxy cap
    is 4000 -- the same setting in practice, and they must pool."""
    try:
        return f'{round(float(v) / 1000)}k' if float(v) > 0 else ''
    except (TypeError, ValueError):
        return ''


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
        # the cap actually sent; blank (older rows) = the effort's bundle value. Remote
        # (OpenRouter) models never get a local cap.
        'think': '' if row['model'].startswith('openrouter/') else norm_think(
            row.get('srv.--reasoning-budget') or AIDER_EFFORT_THINK.get(row['reasoning_effort'] or 'low')),
        'temp': _norm_num(temp),
        'top_p': _norm_num(row['srv.--top-p']),
        'top_k': _norm_num(row['srv.--top-k']),
        'min_p': _norm_num(row['srv.--min-p']),
        'dry': _norm_num(row['srv.--dry-multiplier']),
    }
    return {k: v for k, v in settings.items() if v}


def harbor_settings(row):
    """behavior_settings for a Harbor trial, on the same keys so both runners' rows are
    labeled together and a Harbor trial joins the aider model it matches (same gguf, same
    samplers, same effort). Effort is Claude Code's (what Strata resolved, from the
    proxy) -- not the same mechanism as aider's bundle, but the same intent. The agent
    only differs between Harbor runs (aider rows have none, and a blank is never a
    difference). The eval proxy's token budget is left out: it only matters when hit
    (budget_hit, rare), and splitting a model by it would cut it off from its other runs."""
    settings = {
        'agent': row['agent'],
        'effort': row.get('px_effort') or row.get('srv.eff.reasoning_effort', ''),
        # thinking cap, rounded like aider's (norm_think) so the two runners pool
        'think': norm_think(row.get('srv.eff.reasoning_budget', '')),
        'temp': _norm_num(row.get('srv.eff.temp') or row.get('srv.--temp', '')),
        'top_p': _norm_num(row.get('srv.--top-p', '')),
        'top_k': _norm_num(row.get('srv.--top-k', '')),
        'min_p': _norm_num(row.get('srv.--min-p', '')),
    }
    return {k: v for k, v in settings.items() if v}


def settings_of(row):
    return harbor_settings(row) if row.get('_runner') == 'harbor' else behavior_settings(row)


def assign_model_keys(rows):
    """Sets each row's srv.-m to its model key: the gguf filename, plus '@<diffs>' when
    it ran at other than the file's usual settings (see short_label)."""
    # The same gguf can be served from different mount dirs (/models, /models3, ...);
    # keying models by full path made those separate models with identical labels
    # (duplicate heatmap column + filter checkbox). The filename is the identity.
    for row in rows:
        # Shard markers too: the same model re-split (-of-00002 vs -of-00003) is the same
        # model, but different keys gave identical labels (duplicate rows).
        row['srv.-m'] = _SHARD_RE.sub('', row['srv.-m'].split('/')[-1])
    # The same gguf run at different reasoning efforts or sampling settings is a
    # different model in practice. Each file's most common settings combo keeps the plain
    # label; any other combo becomes its own model, 'file.gguf@<differences>' listing only
    # the settings that differ from the usual (a model run at a single combo is untouched,
    # whatever it is).
    # Per file, each setting's usual value is its own mode; a blank/unknown value (older
    # rows that predate the capture) is never a difference.
    for row in rows:
        row['_settings'] = settings_of(row)
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
        row['_label'] = {k: v for k, v in diffs.items() if v}
    # The thinking cap mostly follows effort, so a label that already shows an effort
    # difference leaves the cap out -- unless that would merge two different combos
    # (same file and effort, different cap), in which case those keep it.
    shortened = defaultdict(set)
    short = lambda d: tuple((k, v) for k, v in d.items() if not (k == 'think' and 'effort' in d))
    for row in rows:
        shortened[(row['srv.-m'], short(row['_label']))].add(tuple(row['_label'].items()))
    for row in rows:
        label = row['_label']
        s_ = short(label)
        if len(shortened[(row['srv.-m'], s_)]) == 1:
            label = dict(s_)
        if label:
            # Effort stays bare ('@xhigh') for continuity with older labels.
            row['srv.-m'] = row['srv.-m'] + '@' + ','.join(v if k == 'effort' else f'{k}={v}'
                                                           for k, v in label.items())


def _int(v):
    return int(v) if v not in ('', None) else None


def aider_attempt(row):
    """One aider TSV row -> attempt record. Fields every runner's adapter provides:
      runner, variant   'aider', 'aider' (a Harbor suite would be e.g. 'harbor-tc')
      task              (variant, language, exercise): the unit task-level stats use;
                        twin = (language, exercise) links variants of one exercise
      model             model key (gguf filename[@settings diffs]) -> short_label()
      seed, timestamp, run
      pass              bool
      steps             tries (aider) / test calls used, or None
      duration          seconds or None; out_tokens / in_tokens: int or None
      gen_sec           time the model spent on requests (eval proxy), or None: tok/s
                        uses it over duration, which also counts tool/test time
      end_reason, flags outcomes.py
      timed_out, ctx_exhausted   heatmap markers (end reason timeout / context-window count)
      raw               the source row, for runner-specific charts (chart4, summary)"""
    end, flags = outcomes.classify_aider(row)
    lang, ex = row['language'], row['exercise']
    return {
        'runner': 'aider', 'variant': 'aider', 'task': ('aider', lang, ex), 'twin': (lang, ex),
        'language': lang, 'exercise': ex, 'model': row['srv.-m'],
        'seed': row['seed'], 'timestamp': row['timestamp'], 'run': row['run_cmd'],
        'pass': row['pass'] == 'true', 'steps': _int(row['tries_taken']),
        'duration': float(row['duration']) if row['duration'] != '' else None,
        'out_tokens': _int(row['completion_tokens']), 'in_tokens': _int(row['prompt_tokens']),
        'gen_sec': float(row['px_model_sec']) if row.get('px_model_sec') not in ('', '0', None) else None,
        'tokens_at_pass': None,  # an aider attempt stops at its pass: tokens is the figure
        'think_blocks': None, 'think_cuts': None,
        'end_reason': end, 'flags': flags, 'trust': outcomes.trust(flags),
        'timed_out': end == 'timeout',
        'ctx_exhausted': int(row['exhausted_context_windows'] or 0),
        'raw': row,
    }


def mark_superseded(attempts):
    # A (model, exercise, seed) triple is one independent sample: re-running the same
    # seed just re-samples the same outcome (greedy/legacy rows have an empty seed and
    # count as one shared seed), so repeat runs must not stack up as extra evidence —
    # 5 identical failures on seed 1 are one failure, not five. Only the latest run per
    # seed is scored; earlier ones are flagged `_superseded` and kept for the tooltip.
    latest = {}
    for a in attempts:
        key = (a['model'], a['task'], a['seed'])
        if key not in latest or a['timestamp'] >= latest[key]['timestamp']:
            latest[key] = a
    for a in attempts:
        a['superseded'] = latest[(a['model'], a['task'], a['seed'])] is not a


def harbor_attempt(row):
    """One harbor-trials.tsv row -> attempt record (fields: see aider_attempt).
    variant: 'harbor-tc' for test-calls tasks (any N: a task's N is a limiter tuned once,
    not something to compare), else 'harbor'. seed = the trial id: -k attempts are
    independent samples (no seed is sent), so none may collapse into another.
    steps = run-tests calls used (tc tasks only); duration = agent time, minus server
    slowdowns when collect.py measured them."""
    end, flags = outcomes.classify_harbor(row)
    lang, ex = row['language'], row['exercise']
    variant = 'harbor-tc' if re.search(r'__tc\d+$', row['task']) else 'harbor'
    ts = row['started_at'][:19] + 'Z' if row['started_at'] else ''
    dur = row['agent_sec_adj'] or row['agent_sec']
    return {
        'runner': 'harbor', 'variant': variant, 'task': (variant, lang, ex), 'twin': (lang, ex),
        'language': lang, 'exercise': ex, 'model': row['srv.-m'],
        'seed': row['trial'], 'timestamp': ts, 'run': row['job'],
        'pass': row['reward'] == '1',
        'steps': _int(row['test_calls']) if variant == 'harbor-tc' else None,
        'duration': float(dur) if dur else None,
        'out_tokens': _int(row['px_output_tokens'] or row['out_tokens']), 'in_tokens': _int(row['in_tokens']),
        'gen_sec': float(row['px_model_sec']) if row.get('px_model_sec') not in ('', '0', None) else None,
        # output tokens used by the time the code first passed every test (from mid-run
        # grading, harbor-progress.tsv), when known; the agent may keep going after that
        'tokens_at_pass': _int(row.get('first_pass_out_tokens', '')),
        # thinking blocks, and how many the reasoning budget cut off: the per-block rate is
        # the measure (the per-trial reasoning_cut flag hits most multi-turn trials)
        'ctx_max': _int(row.get('ctx_max', '')),
        'think_blocks': _int(row.get('thinking_blocks', '')), 'think_cuts': _int(row.get('thinking_cutoffs', '')),
        'end_reason': end, 'flags': flags, 'trust': outcomes.trust(flags),
        'timed_out': end == 'timeout', 'ctx_exhausted': 0,
        'raw': row,
    }


def read_tsv(path):
    try:
        with open(path) as f:
            return list(csv.DictReader(f, delimiter='\t'))
    except FileNotFoundError:
        return []


def fill_harbor_models(rows, aider_rows):
    """Harbor trials get srv.-m (the gguf served) when the job captured server settings,
    else the agent's model alias mapped through aider's alias -> gguf pairs (jobs from
    before the capture), else the bare alias."""
    alias = defaultdict(set)
    for r in aider_rows:
        if r['srv.-m']:
            alias[r['model'].split('/')[-1].lower()].add(r['srv.-m'].split('/')[-1])
    for r in rows:
        r['_runner'] = 'harbor'
        if not r.get('srv.-m'):
            files = alias.get(r['model'].lower(), set())
            r['srv.-m'] = next(iter(files)) if len(files) == 1 else r['model']


def load_attempts():
    """Scored attempts, plus how many harness/server failures were dropped (end reason
    'harness': the run never got a fair go, e.g. auth errors or no server output, so
    it's no evidence about the model either way)."""
    aider_rows = read_tsv(TSV_PATH)
    harbor_rows = read_tsv(HARBOR_TSV_PATH)
    fill_harbor_models(harbor_rows, aider_rows)
    # one labeling pass over both runners: a model's usual settings are found across all
    # its runs, so Harbor trials land on the aider model with matching settings
    assign_model_keys(aider_rows + harbor_rows)
    attempts = [aider_attempt(r) for r in aider_rows] + [harbor_attempt(r) for r in harbor_rows]
    excluded = sum(1 for a in attempts if a['end_reason'] == 'harness')
    recent = recent_runs(attempts)
    attempts = [a for a in attempts if a['end_reason'] != 'harness']
    mark_superseded(attempts)
    return attempts, excluded, recent


def reasoning_lengths(rows):
    """Per Harbor attempt, the reasoning-token count of each of its model requests (from
    harbor-requests.tsv), for the reasoning-length spread chart. 'x' lists the lengths of
    the requests the thinking cap cut off (a subset of 'r'). Aider has no per-request
    data, so it isn't here. The page groups by model + effort + cap and applies its filters."""
    by_trial = defaultdict(list)
    for q in read_tsv(HARBOR_REQUESTS_PATH):
        by_trial[(q['job'], q['trial'])].append(q)
    out = []
    for a in rows:
        if a['runner'] != 'harbor':
            continue
        reqs = by_trial.get((a['run'], a['seed']))
        if not reqs:
            continue
        raw = a['raw']
        out.append({
            'm': short_label(a['model']), 'l': task_label(a['task']),
            'e': raw.get('px_effort') or raw.get('srv.eff.reasoning_effort', ''),
            'cap': _int(reqs[0]['reasoning_budget']) or _int(raw.get('px_reasoning_budget', '')) or 0,
            'r': [int(q['reasoning_tokens']) for q in reqs],
            'x': [int(q['reasoning_tokens']) for q in reqs if q['budget_cutoff'] == '1'],
        })
    return out


def model_settings(attempts):
    """Per model label: each differentiating setting -> the sorted values its attempts ran
    with (a model key pools runs, so a value can be mixed, e.g. aider + Harbor). Feeds the
    filter panel's settings section. Unrecorded settings show as '(unset)'."""
    vals = defaultdict(lambda: defaultdict(set))
    for a in attempts:
        raw = a['raw']
        st = dict(raw.get('_settings') or settings_of(raw))
        st['runner'] = a['runner']
        if a['runner'] == 'harbor':
            st['budget'] = raw.get('budget', '')
        for k, v in st.items():
            vals[short_label(a['model'])][k].add(v or '(unset)')
    keys = {k for d in vals.values() for k in d}
    return {m: {k: sorted(d.get(k, {'(unset)'})) for k in sorted(keys)} for m, d in vals.items()}


# ---- settings effects / standouts: per-attempt stat values for the page's tables ----
# Each stat: (key, label, kind, better, per-attempt value). 'rate' stats are compared as a
# difference in points; 'ratio' stats as a typical B/A ratio.
def _px(a, col):
    v = a['raw'].get(col, '')
    return float(v) if v not in ('', None) else None


def _per_req(a, col):
    n, v = _px(a, 'px_requests'), _px(a, col)
    return v / n if n and v is not None else None


# Floor on a single turn's output tokens (5th percentile of tokens / turn over all attempts with
# request counts, set by fx_export): a turn can't be arbitrarily short, so net_tokens only
# counts what a run wrote beyond that.
_TOK_FLOOR = 0.0


def _net_tokens(a):
    n, t = _px(a, 'px_requests'), a['out_tokens']
    return max(1.0, t - n * _TOK_FLOOR) if n and t else None


def _capability(a):
    """One 0-1 capability number per attempt: mostly the share of tests passed (aider has no
    per-test counts: its pass/fail stands in), plus credit for a full pass, a little for not
    timing out, minus a penalty for declaring done with tests failing (confidently wrong is
    worse than just not finishing) and a small one for a pass that took more tries / test calls
    than the task's average (a['steps_rel'], set in fx_export)."""
    passed = float(a['pass'])
    tf = _px(a, 'tests_frac_adj') if _px(a, 'tests_frac_adj') is not None else _px(a, 'tests_frac')
    if tf is None:
        tf = passed
    v = 0.65 * tf + 0.30 * passed + 0.05 * (1 - float(a['timed_out']))
    if a['runner'] == 'harbor' and a['end_reason'] == 'done_wrong':
        v -= 0.15
    rel = a.get('steps_rel')
    if passed and rel is not None:
        v -= 0.05 * min(1.0, max(0.0, rel - 1))
    return max(0.0, min(1.0, v))


# better: which way is an improvement ('up' / 'down'), or None for a diagnostic with no
# good direction (more reasoning per request is neither better nor worse by itself)
EFFECT_STATS = [
    ('capability', 'capability score (tests passed, pass, done-wrong, tries)', 'rate', 'up', _capability),
    ('pass', 'pass rate', 'rate', 'up', lambda a: float(a['pass'])),
    ('timeout', 'timeout rate', 'rate', 'down', lambda a: float(a['timed_out'])),
    ('out_tokens', 'output tokens', 'ratio', 'down', lambda a: a['out_tokens'] or None),
    ('duration', 'duration', 'ratio', 'down', lambda a: a['duration'] or None),
    # counts keep a real 0 (an agent that never called run-tests): None means not measured
    ('steps', 'tries / test calls', 'ratio', 'down', lambda a: a['steps']),
    ('net_tokens', 'net output tokens (tokens minus a per-turn floor)', 'ratio', 'down', _net_tokens),
    ('requests', 'turns (model requests)', 'ratio', 'down', lambda a: _px(a, 'px_requests')),
    ('tok_req', 'output tokens / turn', 'ratio', None,
     lambda a: a['out_tokens'] / _px(a, 'px_requests') if a['out_tokens'] and _px(a, 'px_requests') else None),
    ('think_cut', 'thinking cut-off rate', 'rate', 'down',
     lambda a: a['think_cuts'] / a['think_blocks'] if a['think_blocks'] else None),
    # reasoning style / degeneration (Harbor: Claude Code logs keep the thinking text)
    ('think_code', 'share of thinking that is code (Harbor)', 'rate', 'down',
     lambda a: _px(a, 'think_code_chars') / _px(a, 'think_chars') if _px(a, 'think_chars') else None),
    ('copied', 'big writes copied from thinking (Harbor)', 'rate', 'down',
     lambda a: _px(a, 'writes_from_think') / _px(a, 'big_writes') if _px(a, 'big_writes') else None),
    ('spill', 'attempts where thinking spilled into the answer (Harbor)', 'rate', 'down',
     lambda a: float('think_spill' in a['flags']) if a['runner'] == 'harbor' else None),
    ('looped', 'loops per thinking block (Harbor)', 'rate', 'down',
     lambda a: _px(a, 'loop_blocks') / _px(a, 'thinking_blocks') if _px(a, 'thinking_blocks') and _px(a, 'loop_blocks') is not None else None),
    ('loop_stuck', 'share of loops that never recovered (Harbor)', 'rate', 'down',
     lambda a: _px(a, 'loop_unrecovered') / _px(a, 'loop_blocks') if _px(a, 'loop_blocks') else None),
    # glitch / quality counters (None where a runner doesn't record them)
    ('tests_frac', 'tests passed (fraction, Harbor)', 'rate', 'up',
     lambda a: _px(a, 'tests_frac_adj') if _px(a, 'tests_frac_adj') is not None else _px(a, 'tests_frac')),
    ('own_tests', 'attempts that wrote their own tests (Harbor)', 'rate', 'up',
     lambda a: float('wrote_tests' in a['flags']) if a['runner'] == 'harbor' else None),
    ('malformed', 'attempts with malformed calls / edits', 'rate', 'down', lambda a: float('malformed' in a['flags'])),
    # one number for all degeneration signs above, weighted by severity (outcomes.GLITCH_WEIGHTS);
    # test results don't enter it. Aider can't see loops / spills, so compare within a runner.
    ('stability', 'stability: 1 - severity-weighted glitches per attempt (repeats count each time)', 'rate', 'up',
     lambda a: None if a['end_reason'] == 'harness' else 1 - outcomes.instability(a['flags'], a['end_reason'], outcomes.flag_counts(a['raw'], a['runner']))),
    ('tool_err', 'tool calls that errored (Harbor)', 'rate', 'down',
     lambda a: _px(a, 'tool_errors') / _px(a, 'tool_calls') if _px(a, 'tool_calls') else None),
    # behaviour to watch, not good/bad: of the runs of consecutive non-zero shell exits, the share that
    # reach 3+ (pooled over runs, see EFFECT_WEIGHTS)
    ('bash_streak', 'share of non-zero shell-exit streaks that reach 3+ (Harbor)', 'rate', None,
     lambda a: _px(a, 'bash_streaks3') / _px(a, 'bash_streaks') if _px(a, 'bash_streaks') else None),
    ('edit_err', 'edit calls that failed (Harbor)', 'rate', 'down',
     lambda a: _px(a, 'edit_errors') / _px(a, 'edits') if _px(a, 'edits') else None),
    ('glitch_end', 'attempts that ended in a glitch (broken output / raw tool markup)', 'rate', 'down',
     lambda a: float(a['end_reason'] == 'glitch' or 'markup_final' in a['flags'])),
    ('done_wrong', 'declared done with tests failing (Harbor)', 'rate', 'down',
     lambda a: float(a['end_reason'] == 'done_wrong') if a['runner'] == 'harbor' else None),
    ('syntax', 'attempts with syntax / lint errors (aider)', 'rate', 'down',
     lambda a: float(bool(_px(a, 'syntax_errors') or _px(a, 'indentation_errors') or 'lint' in a['flags']))
     if a['runner'] == 'aider' else None),
    # recall / contamination (outcomes.py flags): lower = the result says more about the
    # model's own ability. Harbor detects them from the trajectory; aider only has the
    # inferred flag (the same model recalled this exercise in Harbor).
    ('knows_src', 'named the exercise source itself (Exercism / its specs)', 'rate', 'down',
     lambda a: float('knows_source' in a['flags']) if a['runner'] == 'harbor' else None),
    ('recalled', 'recalled the upstream solution / tests', 'rate', 'down',
     lambda a: float(bool({'recalled_content', 'recalled_inferred'} & a['flags']))),
    ('fetched', 'fetched upstream files / outside URLs', 'rate', 'down',
     lambda a: float(bool({'fetched_source', 'internet'} & a['flags'])) if a['runner'] == 'harbor' else None),
    ('reason_req', 'reasoning tokens / turn', 'ratio', None, lambda a: _per_req(a, 'px_reasoning_tokens') or None),
    ('answer_req', 'answer tokens / turn', 'ratio', None, lambda a: _per_req(a, 'px_answer_tokens') or None),
]
# stats that pool over events, not attempts: the page's per-list mean weights each attempt's value
# by this (default 1), so an attempt with 4 streaks counts 4x one with a single streak
# (stability-style stats: two errors in one attempt count as worse than one, so they pool per event)
EFFECT_WEIGHTS = {
    'bash_streak': lambda a: _px(a, 'bash_streaks'),
    'tool_err': lambda a: _px(a, 'tool_calls'),
    'edit_err': lambda a: _px(a, 'edits'),
    'think_cut': lambda a: a['think_blocks'],
    'think_code': lambda a: _px(a, 'think_chars'),
    'copied': lambda a: _px(a, 'big_writes'),
    'looped': lambda a: _px(a, 'thinking_blocks'),
    'loop_stuck': lambda a: _px(a, 'loop_blocks'),
}
# a rate stat's natural range: the change that counts as a full-size effect (the page's
# thresholds scale with it). Cut-off rates live in 0-10%, so 25 points would make every
# change tiny.
EFFECT_RATE_SPAN = {'think_cut': 0.05, 'looped': 0.05, 'stability': 0.10, 'tool_err': 0.10, 'edit_err': 0.10}
EFFECT_SETTING_KEYS = ('effort', 'think', 'temp', 'top_p', 'top_k', 'min_p', 'dry')
# Name tokens that make poor groups: quant fragments (the quant itself is a group via
# QUANT_PATTERN), and MTP (not named consistently: many MTP files don't say so).
_TOKEN_SKIP = {'gguf', 'of', 'ud', 'q8', 'q4', 'k', 'xl', 'xs', 'xxs', 's', 'm', 'iq3', 'iq4', 'iq1', 'iq2', 'tq2', 'tq1', 'pq2',
               'mtp',
               # uploader / distill tags that mostly overlap other groups
               'claude', 'distilled', 'zacaj', 'isnta', 'daslab'}
# multi-part names that are one thing: matched first and kept whole (so 'Flash-Next' is one
# group, not 'Flash' and 'Next')
_GROUP_PHRASES = ('Flash-Next',)
# tokens that name the same thing (some files say '35B', some '35B-A3B'): one group
# groups naming any of these tokens are left out of the filter panel
_HIDDEN_GROUP_TOKENS = {'RSI'}
_TOKEN_ALIAS = {'35B': '35B-A3B', 'A3B': '35B-A3B'}


def model_groups(ggufs):
    """Groups of ggufs sharing a name token: a family ('Gyro', 'Swift', 'Qwen3.8', '27B'),
    a quant ('IQ1_M') or a trait ('Distilled'). Tokens on 2+ files but not nearly all;
    identical memberships merged under their shortest names. Names split on - _ and
    space, not '.', so versions like 'Qwen3.6' stay whole."""
    tok, quants = defaultdict(set), set()
    for m in ggufs:
        base = _SHARD_RE.sub('', m).removesuffix('.gguf')
        for q in QUANT_PATTERN.findall(base):
            tok[q].add(m)
            quants.add(q)
    # pieces of a quant tag ('IQ3', 'XXS') aren't name tokens of their own
    frag = {f.lower() for q in quants for f in q.split('_')}
    for m in ggufs:
        base = _SHARD_RE.sub('', m).removesuffix('.gguf')
        for ph in _GROUP_PHRASES:
            if ph in base:
                tok[ph].add(m)
                base = base.replace(ph, ' ')
        for t_ in re.split(r'[-_ ]+', base):
            # a leading digit is a version fragment ('4.7'), unless it's a size ('27B')
            numeric = t_[0].isdigit() and not re.fullmatch(r'\d+(\.\d+)?B', t_)
            if len(t_) >= 2 and t_.lower() not in _TOKEN_SKIP and t_.lower() not in frag and not numeric and t_ not in quants:
                tok[_TOKEN_ALIAS.get(t_, t_)].add(m)
    groups = {}
    for t_, ms in tok.items():
        # a quant gets a group even with one file: the mask is how you pick it
        if (1 if t_ in quants else 2) <= len(ms) <= len(ggufs) - 2:
            groups.setdefault(frozenset(ms), []).append(t_)
    out = []
    for ms, names in groups.items():
        names = sorted(names, key=len)
        if _HIDDEN_GROUP_TOKENS & set(names):
            continue
        q = next((n for n in names if n in quants), None)
        out.append({'label': ' / '.join(names[:3]), 'members': sorted(short_label(m) for m in ms),
                    **({'quant': q} if q else {})})
    return out


def fx_export(attempts):
    """Per-attempt records for the page's settings-effects and standouts tables, which it
    computes itself from whatever passes the filter panel (pairing, pooling, intervals:
    see fxCompute in the template). Every stat's per-attempt value is computed here
    (EFFECT_STATS), so the page needs no raw columns. Settings combos are deduplicated."""
    combos, combo_idx, recs = [], {}, []
    global _TOK_FLOOR
    tpr = sorted(a['out_tokens'] / _px(a, 'px_requests') for a in attempts if a['out_tokens'] and _px(a, 'px_requests'))
    _TOK_FLOOR = tpr[int(0.05 * len(tpr))] if tpr else 0.0
    # tries / test calls relative to the task's average (same task + runner), for the capability score
    sums = defaultdict(lambda: [0.0, 0])
    for a in attempts:
        if a['steps']:
            k_ = (a['task'], a['runner'], a['variant'])
            sums[k_][0] += a['steps']; sums[k_][1] += 1
    for a in attempts:
        m_ = sums.get((a['task'], a['runner'], a['variant']))
        a['steps_rel'] = a['steps'] / (m_[0] / m_[1]) if a['steps'] and m_ else None
    for a in attempts:
        st = tuple(sorted((a['raw'].get('_settings') or {}).items()))
        if st not in combo_idx:
            combo_idx[st] = len(combos)
            combos.append(dict(st))
        gguf = a['model'].split('@')[0]
        x = []
        for key, label, kind, better, fn in EFFECT_STATS:
            v = fn(a)
            x.append(None if v is None else round(v, 4))
        xw = {}
        for i, (key, *_) in enumerate(EFFECT_STATS):
            if key in EFFECT_WEIGHTS and x[i] is not None:
                xw[i] = EFFECT_WEIGHTS[key](a)
        recs.append({'m': short_label(a['model']), 'g': short_label(gguf), 'q': quant_of(gguf), 'c': combo_idx[st],
                     't': task_label(a['task']), 'v': a['variant'], 'l': a['language'], 'w': f"{a['language']}/{a['exercise']}",
                     's': a['seed'], 'ts': a['timestamp'], 'x': x, **({'xw': xw} if xw else {})})
    ggufs = sorted({a['model'].split('@')[0] for a in attempts})
    return {
        'stats': [{'key': k, 'label': lb, 'kind': kd, 'better': bt,
                   'span': EFFECT_RATE_SPAN.get(k, 0.25) if kd == 'rate' else None} for k, lb, kd, bt, _ in EFFECT_STATS],
        'combos': combos, 'attempts': recs, 'model_groups': model_groups(ggufs),
        'setting_keys': list(EFFECT_SETTING_KEYS),
    }


RECENT_RUNS = 8
# raw TSV columns summed over a recent run's attempts (label, column), per runner
RECENT_COUNTS = {
    'aider': [('malformed responses', 'num_malformed_responses'), ('edit-format retries', 'hist_edit_format'),
              ('syntax errors', 'syntax_errors'), ('indentation errors', 'indentation_errors'),
              ('lint retries', 'hist_lint'), ('wrong-file edits', 'hist_wrong_file'),
              ('error outputs', 'error_outputs'), ('test timeouts', 'test_timeouts'),
              ('lazy comments', 'lazy_comments'), ('context exhaustions', 'exhausted_context_windows'),
              ('output-limit hits', 'hist_output_limit'), ('server errors', 'hist_server_errors')],
    'harbor': [('tool calls', 'tool_calls'), ('tool errors', 'tool_errors'), ('malformed calls', 'malformed_calls'),
               ('edits', 'edits'), ('edit errors', 'edit_errors'), ('test runs', 'test_runs'),
               ('builds', 'build_runs'), ('failed builds', 'build_fail_runs'),
               ('thinking blocks', 'thinking_blocks'), ('thinking cut off', 'thinking_cutoffs'),
               ('big writes', 'big_writes'), ('writes copied from thinking', 'writes_from_think'),
               ('thinking spills', 'think_spills'), ('looped blocks', 'loop_blocks'),
               ('unrecovered loops', 'loop_unrecovered'), ('max_tokens stops', 'px_max_tokens'),
               ('requests', 'px_requests'), ('rejected requests', 'px_rejected'), ('slow requests', 'px_slow_requests')],
}
RUN_GAP_SEC = 2 * 3600


def recent_runs(attempts, n=RECENT_RUNS):
    """The latest n runs (an aider run_cmd / a Harbor job), harness failures included so
    a broken run still shows up. aider reuses run_cmd when a command is rerun, so one
    run_cmd is split wherever its attempts are more than RUN_GAP_SEC apart."""
    def ts(a):
        try:
            return datetime.fromisoformat(a['timestamp'].replace('Z', '+00:00')).timestamp()
        except ValueError:
            return None
    by_run = defaultdict(list)
    for a in attempts:
        if ts(a) is not None:
            by_run[(a['runner'], a['run'])].append(a)
    groups = []
    for (runner, run), atts in by_run.items():
        atts.sort(key=ts)
        cur = [atts[0]]
        for a in atts[1:]:
            if ts(a) - ts(cur[-1]) > RUN_GAP_SEC:
                groups.append((runner, run, cur))
                cur = []
            cur.append(a)
        groups.append((runner, run, cur))
    groups.sort(key=lambda g: ts(g[2][-1]), reverse=True)
    out = []
    for runner, run, atts in groups[:n]:
        scored = [a for a in atts if a['end_reason'] != 'harness']
        raw = atts[-1]['raw']
        durs = [a['duration'] for a in scored if a['duration'] is not None]
        toks = [a['out_tokens'] for a in scored if a['out_tokens'] is not None]
        gen = [a['gen_sec'] for a in scored if a['gen_sec']]
        ends = defaultdict(int)
        for a in atts:
            ends[a['end_reason']] += 1
        if runner == 'aider':
            details = {'edit format': raw['edit_format'], 'tries': raw['tries'], 'effort': raw['reasoning_effort'],
                       'max tokens': raw['max_tokens'], 'target': raw['target'], 'alias': raw['model']}
        else:
            details = {'agent': f"{raw['agent']} {raw['agent_version']}".strip(), 'effort': raw.get('px_effort', ''),
                       'budget': raw['budget'], 'max dur': raw['max_dur'], 'alias': raw['model']}
        details.update({f'{k}': v for k, v in settings_of(raw).items() if k != 'effort' or runner == 'harbor'})
        out.append({
            'runner': runner, 'run': run, 'model': short_label(atts[-1]['model']),
            'start': atts[0]['timestamp'], 'end': atts[-1]['timestamp'],
            # wall clock: harbor timestamps are trial starts, aider's are exercise ends
            'wall_sec': round(max(ts(a) + (a['duration'] or 0) for a in atts) - ts(atts[0])) if runner == 'harbor'
                        else round(ts(atts[-1]) - min(ts(a) - (a['duration'] or 0) for a in atts)),
            'attempts': len(atts), 'harness': len(atts) - len(scored),
            'passes': sum(a['pass'] for a in scored),
            'tasks': len({a['task'] for a in atts}),
            'seeds': sorted({a['seed'] for a in atts if a['seed']})[:6] if runner == 'aider' else [],
            'avg_dur': round(sum(durs) / len(durs), 1) if durs else None,
            'out_tokens': sum(toks) if toks else None,
            'tok_s': round(sum(toks) / sum(gen), 1) if gen and toks else None,
            'avg_steps': round(sum(a['steps'] for a in scored if a['pass'] and a['steps'] is not None)
                               / max(1, sum(1 for a in scored if a['pass'] and a['steps'] is not None)), 2)
                         if any(a['pass'] and a['steps'] is not None for a in scored) else None,
            'ends': dict(ends),
            # attempts raising each outcomes.py flag, and summed error / event counters
            'flags': dict(sorted(Counter(f for a in atts for f in a['flags'] if f in outcomes.FLAGS).items(),
                                 key=lambda kv: -kv[1])),
            'counts': {lb: tot for lb, col in RECENT_COUNTS.get(runner, [])
                       if (tot := int(sum(_px(a, col) or 0 for a in atts))) or lb in ('tool calls', 'edits')},
            'details': {k: v for k, v in details.items() if v not in ('', None)},
            'results': [{'label': task_label(a['task']), 'pass': a['pass'], 'end': a['end_reason'],
                         'steps': a['steps'], 'dur': a['duration'], 'tok': a['out_tokens'], 'flags': sorted(a['flags'])} for a in atts],
        })
    return out


def task_label(task):
    variant, lang, ex = task
    return f'{ex} ({lang}) ·{VARIANT_CODES.get(variant, variant)}'


def _s(v):
    """steps as the page has always received them: a string, '' when unknown."""
    return '' if v is None else str(v)


# Aider has no trajectory to show recall in, but a model that recalled an exercise's
# solution in Harbor almost certainly knows it in aider too. Each aider attempt gets
# `recalled_inferred` from the Harbor attempts of the same model key (or, weaker, the
# same gguf at any settings) on the same exercise, and a trust discounted by that
# recall rate, gentler than direct evidence (RECALL_INFER_CONFIDENCE).
RECALL_INFER_CONFIDENCE = {'model': 0.7, 'gguf': 0.5}
RECALL_INFER_MIN_RATE = 0.5
# heatmap (computed in the viz, over the date window): a cell where at least half the
# attempts are under LOW_TRUST gets a corner mark; a task where LOW_TRUST_TASK_SHARE of
# all its shown attempts are gets a * on its name
LOW_TRUST = 0.5
LOW_TRUST_TASK_SHARE = 0.5


def infer_recall(rows):
    seen = {'model': defaultdict(list), 'gguf': defaultdict(list)}
    for r in rows:
        if r['runner'] == 'harbor':
            hit = 'recalled_content' in r['flags']
            seen['model'][(r['model'], r['twin'])].append(hit)
            seen['gguf'][(r['model'].split('@')[0], r['twin'])].append(hit)
    full_drop = 1 - outcomes.TRUST['recalled_content']
    for r in rows:
        if r['runner'] != 'aider':
            continue
        for level, key in (('model', r['model']), ('gguf', r['model'].split('@')[0])):
            hits = seen[level].get((key, r['twin']))
            if hits:
                rate = sum(hits) / len(hits)
                if rate >= RECALL_INFER_MIN_RATE:
                    r['flags'] = set(r['flags']) | {'recalled_inferred'}
                r['trust'] = min(r['trust'], round(1 - rate * full_drop * RECALL_INFER_CONFIDENCE[level], 3))
                break


def build_dataset(rows, excluded_harness=0):
    infer_recall(rows)
    # `rows` are attempt records (load_attempts), not TSV rows.
    # The task list is fully dynamic: every (language, exercise) pair with at least one
    # row is "in", discovered fresh from the tsv each run — no hardcoded list to keep in
    # sync as the harness adds new comparison tasks over time. A voided/deleted
    # exercise (every one of its rows removed) automatically drops back out, for the
    # same reason: it just no longer appears in `rows`.
    pins = sorted({row['task'] for row in rows})
    pinset = set(pins)
    pin_label = {t: task_label(t) for t in pins}

    # The default selection only seeds which tasks the exercise filter starts with
    # checked (see DEFAULT_TOP_N); it doesn't scope what's computed.
    twin_models = defaultdict(set)
    for row in rows:
        twin_models[row['twin']].add(row['model'])
    top_twins = set(sorted(twin_models, key=lambda t: (-len(twin_models[t]), t))[:DEFAULT_TOP_N])
    default_pins = [t for t in pins if t[1:] in top_twins]

    by_model_ex = defaultdict(set)
    for row in rows:
        # (language, exercise), not exercise name alone — the same collision this file
        # already guards against everywhere else (e.g. "bowling" alone spans 5
        # languages across the full task list). Name-only counting would undercount a
        # model's real distinct-task coverage whenever it hit the same-named exercise
        # in more than one language track.
        by_model_ex[row['model']].add(row['task'])
    models_sorted = sorted(by_model_ex, key=lambda m: -len(by_model_ex[m]))
    main = models_sorted[0]  # the model with the most distinct tasks = the full-suite run
    matrix_models = [short_label(m) for m in models_sorted]
    model_quant = {short_label(m): quant_of(m) for m in models_sorted}

    # Every row's (language, exercise) is in `pinset` by construction, so this is just
    # `rows` — kept as its own name since the rest of this function was written around
    # a "shortlist" of rows and renaming every reference below isn't worth the diff.
    shortlist_rows = [row for row in rows if not row['superseded']]

    # ---- per-test breakdown ----
    # Every run, superseded ones included (flagged) — the hover table shows the full
    # history, but only the latest run per seed feeds any statistic.
    detail = []
    for row in rows:
        detail.append({
            'exercise': row['exercise'], 'language': row['language'], 'model': short_label(row['model']),
            'run_cmd': row['run'], 'pass': row['pass'], 'tries': _s(row['steps']),
            'duration': round(row['duration'] or 0.0, 1), 'timestamp': row['timestamp'],
            'bailed': row['timed_out'],
            'ctx_exhausted': row['ctx_exhausted'],
            'seed': row['seed'], 'superseded': row['superseded'],
            'end_reason': row['end_reason'], 'flags': sorted(row['flags']), 'variant': row['variant'],
            'trust': row['trust'],
            'label': pin_label[row['task']],
            'think_blocks': row['think_blocks'], 'think_cuts': row['think_cuts'],
            'tokens': row['out_tokens'],
            'requests': _px(row, 'px_requests') or None,  # Harbor only; aider rows carry no proxy counts
        })
    detail.sort(key=lambda d: (d['exercise'], d['model']))

    # ---- chart3: tries-taken histogram, all models/exercises (not shortlist-scoped) ----
    hist = defaultdict(lambda: {'pass': 0, 'fail': 0})
    for row in rows:
        t = _s(row['steps'])
        if t == '':
            continue
        hist[t]['pass' if row['pass'] else 'fail'] += 1
    chart3 = [{'tries': k, **v} for k, v in sorted(hist.items(), key=lambda kv: int(kv[0]))]

    # ---- chart4: failure-mode breakdown, all failed runs: outcomes.py flags (aider's own
    # user_asks / error_outputs counters mislabel or mix these, see outcomes.py) ----
    fail_rows_all = [row for row in rows if not row['pass']]
    flag_counts = defaultdict(int)
    for row in fail_rows_all:
        for f in row['flags']:
            flag_counts[f] += 1
    chart4 = [{'label': outcomes.FLAGS[f], 'count': n, 'pct': round(n / len(fail_rows_all) * 100, 1)}
              for f, n in flag_counts.items()]
    chart4.sort(key=lambda d: -d['count'])

    # ---- summary / stat tiles ----
    main_full = [row for row in rows if row['model'] == main and row['run'] == 'full']
    main_full_pass = sum(1 for row in main_full if row['pass'])
    summary = {
        'total_runs': len(rows), 'distinct_tasks': len(pins),
        'total_exercise_names': len(set(row['exercise'] for row in rows)),
        'total_models': len(models_sorted),
        'overall_pass': sum(1 for row in rows if row['pass']),
        'overall_fail': sum(1 for row in rows if not row['pass']),
        'total_tasks': len(pins), 'default_tasks': len(default_pins),
        'main_label': short_label(main), 'main_full_pass': main_full_pass, 'main_full_n': len(main_full),
        'excluded_harness': excluded_harness,
    }

    # ---- perf: generation speed + tokens generated, normalized to exercise difficulty ----
    pass_rows = [row for row in rows
                 if row['task'] in pinset and row['pass']
                 and row['duration'] and row['duration'] > 0 and row['out_tokens'] is not None]
    by_ex_tokens = defaultdict(list)
    for row in pass_rows:
        by_ex_tokens[pin_label[row['task']]].append(row['out_tokens'])
    ex_baseline = {ex: sum(toks) / len(toks) for ex, toks in by_ex_tokens.items()}

    points = []
    for row in pass_rows:
        ex = pin_label[row['task']]
        tokens = row['out_tokens']
        points.append({
            'model': short_label(row['model']), 'exercise': ex, 'tokens': tokens,
            'tokens_norm': round(tokens / ex_baseline[ex], 3),
            'tokps': round(tokens / (row['gen_sec'] or row['duration']), 2),
            'duration': round(row['duration'], 1),
            'tries': _s(row['steps']), 'tries_num': row['steps'],
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
    by_exercise = {pin_label[t]: agg([p for p in points if p['exercise'] == pin_label[t]])
                   for t in pins}
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
        if row['pass'] and row['steps'] is not None:
            pass_tries_by_pin_model[row['task']][short_label(row['model'])].append(row['steps'])
    pin_model_means = {pin: {m: sum(v) / len(v) for m, v in d.items()}
                       for pin, d in pass_tries_by_pin_model.items()}
    all_means = [x for d in pin_model_means.values() for x in d.values()]
    mu_global = sum(all_means) / len(all_means) if all_means else 2.0

    def attempt_efficiency(row):
        if row['steps'] is None:
            return None
        pin, model, k = row['task'], short_label(row['model']), row['steps']
        ref = [x for m, x in pin_model_means.get(pin, {}).items() if m != model]
        mu = shrink_toward_prior(sum(ref) / len(ref) if ref else mu_global, len(ref), mu_global, EFF_MU_PRIOR_STRENGTH)
        if row['pass']:
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
            eff_by_pin_model[row['task']][short_label(row['model'])].append(row_eff[id(row)])
    eff_pin_prior = {}
    for pin, d in eff_by_pin_model.items():
        means = [sum(v) / len(v) for v in d.values()]
        eff_pin_prior[pin] = sum(means) / len(means)

    cost_rows = [row for row in rows
                 if row['duration'] and row['duration'] > 0 and row['out_tokens'] is not None]
    cost_points = [{
        'exercise': pin_label[row['task']],
        'model': short_label(row['model']), 'pass': row['pass'],
        'tokens': row['out_tokens'], 'duration': round(row['duration'], 1), 'ctx_max': row.get('ctx_max'),
        'tries': _s(row['steps']), 'timestamp': row['timestamp'], 'seed': row['seed'],
        'eff': round(row_eff[id(row)], 4) if row_eff[id(row)] is not None else None,
        'end_reason': row['end_reason'],
        'gen_sec': round(row['gen_sec'], 1) if row['gen_sec'] else None,
        'tokens_at_pass': row['tokens_at_pass'],
        'flags': sorted(row['flags']), 'runner': row['runner'],
        'think_blocks': row['think_blocks'], 'think_cuts': row['think_cuts'],
    } for row in cost_rows]
    cost_spread = {'points': cost_points}

    # ---- usefulness: per-exercise cost / discriminative-power / flakiness / tries-shape ----
    tries_buckets = ['0', '1', '2', '3', '4', '5', '6', '8', '16']
    usefulness_ex = []
    for task in pins:
        variant, lang, ex = task
        label = pin_label[task]
        ex_rows = [row for row in shortlist_rows if row['task'] == task]
        if not ex_rows:
            continue
        by_model = defaultdict(list)
        for row in ex_rows:
            by_model[short_label(row['model'])].append(row)
        model_pass_rates, model_mean_durations, flakiness_scores = [], [], []
        model_mean_tokens = []
        for m in matrix_models:
            rs = by_model.get(m, [])
            if not rs:
                continue
            npass = sum(1 for r_ in rs if r_['pass'])
            rate = npass / len(rs)
            # Fractional (share of this model's seeds that passed), not a majority-vote
            # 0/1: a model that passes 1 of 3 seeds is partly able, not simply "failed".
            model_pass_rates.append(rate)
            durs = [r_['duration'] for r_ in rs if r_['duration'] is not None]
            if durs:
                model_mean_durations.append(sum(durs) / len(durs))
            toks_ = [r_['out_tokens'] for r_ in rs if r_['out_tokens'] is not None]
            if toks_:
                model_mean_tokens.append(sum(toks_) / len(toks_))
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
        avg_out_tokens = sum(model_mean_tokens) / len(model_mean_tokens) if model_mean_tokens else None
        flakiness = sum(flakiness_scores) / len(flakiness_scores) if flakiness_scores else None
        hist2 = defaultdict(lambda: {'pass': 0, 'fail': 0})
        for row in ex_rows:
            t = _s(row['steps'])
            if not t:
                continue
            hist2[t]['pass' if row['pass'] else 'fail'] += 1
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
        fail_rows = [row for row in ex_rows if not row['pass']]
        timeout_fails = sum(1 for row in fail_rows if row['timed_out'])
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
        tries_means_this = list(pin_model_means.get(task, {}).values())
        n_tries_models = len(tries_means_this)
        if n_tries_models >= 2:
            mean_t = sum(tries_means_this) / n_tries_models
            std_t = (sum((x - mean_t) ** 2 for x in tries_means_this) / n_tries_models) ** 0.5
            tries_discriminative = min(1.0, std_t / mean_t) if mean_t > 0 else 0.0
        else:
            tries_discriminative = None
        # Cross-model spread of cost: do models differ a lot in duration / output tokens on
        # this task? (CV of each model's own mean, clipped to [0, 1]; needs 3+ models.)
        def _cv(xs):
            if len(xs) < 3:
                return None
            m_ = sum(xs) / len(xs)
            return min(1.0, (sum((x - m_) ** 2 for x in xs) / len(xs)) ** 0.5 / m_) if m_ > 0 else None
        cost_cvs = []
        for key_ in ('duration', 'out_tokens'):
            per_m = []
            for rs in by_model.values():
                vs = [r_[key_] for r_ in rs if r_[key_]]
                if vs:
                    per_m.append(sum(vs) / len(vs))
            cv_ = _cv(per_m)
            if cv_ is not None:
                cost_cvs.append(cv_)
        cost_discriminative = sum(cost_cvs) / len(cost_cvs) if cost_cvs else None
        usefulness_ex.append({
            '_task': task, 'variant': variant, '_cost_disc': cost_discriminative,
            'exercise': ex, 'language': lang, 'label': label, 'n_attempts': len(ex_rows),
            'n_models': len(model_pass_rates), 'models_passed': round(sum(model_pass_rates), 1),
            'eff_across_models': round(eff_pin_prior.get(task, eff_global), 4),
            'pass_rate_across_models': round(p, 3), 'discriminative': round(discriminative, 3),
            'avg_duration': round(avg_duration, 1),
            'avg_out_tokens': round(avg_out_tokens) if avg_out_tokens is not None else None,
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

    # Fold the secondary spread signals into `discriminative`, somewhat: a task where models
    # all land the same pass/fail but differ a lot in tries, duration or output tokens still
    # separates them. Pass/fail stays dominant (75%); the other 25% is the mean of whichever of
    # tries-discriminative and the cross-model duration / token spread exist (none -> unchanged).
    SECONDARY_DISC_WEIGHT = 0.25
    for d in usefulness_ex:
        d['pass_discriminative'] = d['discriminative']
        sec = [x for x in (d['tries_discriminative'], d.pop('_cost_disc')) if x is not None]
        d['secondary_discriminative'] = round(sum(sec) / len(sec), 3) if sec else None
        if sec:
            d['discriminative'] = round((1 - SECONDARY_DISC_WEIGHT) * d['discriminative']
                                        + SECONDARY_DISC_WEIGHT * sum(sec) / len(sec), 3)

    # Recall: how often models recognised the exercise and recalled the upstream
    # solution or tests (outcomes.py's `recalled_content` flag). A property of the exercise, not the
    # runner, so it's counted per twin over every variant's attempts, but only Harbor
    # trajectories can raise it. Instructions that name the source don't raise it
    # (2026-10-31: 74% recalled with, 81% without), so they count the same. Shrunk toward the cross-exercise rate like the
    # other per-exercise rates; an exercise with no Harbor attempts gets that prior.
    RECALL_PRIOR_STRENGTH = 3
    RECALL_WEIGHT_DISCOUNT = 0.5
    recall_counts = defaultdict(lambda: [0, 0])
    for row in shortlist_rows:
        if row['runner'] != 'harbor':
            continue
        f = row['flags']
        c = recall_counts[row['twin']]
        c[0] += 'recalled_content' in f
        c[1] += 1
    recall_n = sum(n for _, n in recall_counts.values())
    recall_prior = sum(k for k, _ in recall_counts.values()) / recall_n if recall_n else 0.0
    for d in usefulness_ex:
        k, n = recall_counts.get((d['language'], d['exercise']), (0, 0))
        d['recall_attempts'] = n
        d['recall_rate'] = round(shrink_toward_prior(k / n if n else recall_prior, n, recall_prior,
                                                     RECALL_PRIOR_STRENGTH), 3)

    # "Quality" composite: how much a task is worth keeping, folded into one number —
    # rewards discriminative power, penalizes cost (duration), inconsistency
    # (flakiness) and memorisation (recall_rate). Duration is min-max normalized across the active exercises first
    # since it's in seconds while the other two are already 0-1. An exercise with no
    # repeat runs (flakiness == None) is treated as neutral, not penalized — "unproven"
    # isn't the same claim as "inconsistent".
    durations = [d['avg_duration'] for d in usefulness_ex]
    dur_min, dur_max = min(durations), max(durations)
    dur_span = dur_max - dur_min
    for d in usefulness_ex:
        duration_norm = (d['avg_duration'] - dur_min) / dur_span if dur_span else 0
        flakiness_component = d['flakiness'] if d['flakiness'] is not None else 0
        # recall: a task models reliably recall measures memory, not ability
        d['quality'] = round((d['discriminative'] + (1 - duration_norm) + (1 - flakiness_component)
                              + (1 - d['recall_rate'])) / 4, 3)
        # How much this exercise counts toward the chart1 capability score below —
        # discriminative power x reliability, deliberately excluding cost (unlike
        # `quality` above, which also penalizes slow exercises for the heatmap sort;
        # an exercise being slow doesn't make it less informative about capability).
        # Recall discounts it by up to half: a reliably recalled task still says
        # something (recalling correctly is a skill), but less about solving it.
        d['score_weight'] = round(d['discriminative'] * (1 - flakiness_component)
                                  * (1 - RECALL_WEIGHT_DISCOUNT * d['recall_rate']), 3)
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
    weight_by_pin = {d['_task']: d['score_weight'] for d in usefulness_ex}
    impute_by_pin = {d['_task']: d['pass_rate_across_models'] for d in usefulness_ex}
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
            if row['model'] == m:
                sub_by_pin[row['task']].append(row)
        if not sub_by_pin:
            continue

        # Pass 1: per-tested-pin estimate, plus this model's own bias vs the pool on the
        # pins it's actually been tested on (positive = does better than typical there).
        tested_est = {}
        tested = exercises_passed = exercises_failed = exercises_mixed = 0
        bias_num = bias_weight = 0.0
        for task in pins:
            rs = sub_by_pin.get(task)
            if not rs:
                continue
            tested += 1
            w = weight_by_pin[task]
            # trust-weighted (outcomes.TRUST): a suspect attempt (downloaded / recalled
            # the answer) adds little evidence, so the estimate leans on the pin's
            # typical rate instead; the raw pass/fail counts below still count it
            k = sum(r['trust'] for r in rs if r['pass'])
            n = sum(r['trust'] for r in rs)
            est = shrink_toward_prior(k / n, n, impute_by_pin[task], MODEL_PIN_PRIOR_STRENGTH) if n else impute_by_pin[task]
            k_raw = sum(1 for r in rs if r['pass'])
            tested_est[task] = est
            if k_raw == len(rs):
                exercises_passed += 1
            elif k_raw == 0:
                exercises_failed += 1
            else:
                exercises_mixed += 1
            w *= n / len(rs)    # untrusted pins say little about the model's bias either
            bias_num += w * (est - impute_by_pin[task])
            bias_weight += w
        raw_bias = bias_num / bias_weight if bias_weight else 0.0
        bias = shrink_toward_prior(raw_bias, tested, 0.0, BIAS_PRIOR_STRENGTH)

        # Pass 2: assemble floor (untested -> 0), ceiling (untested -> 1), expected
        # (untested -> pool average, no bias adjustment — the plain neutral guess), and
        # projected (untested -> pool average adjusted by this model's own shrunk bias,
        # clamped to a valid probability) across every pin.
        num_expected = num_worst = num_best = num_projected = 0.0
        for task in pins:
            w = weight_by_pin[task]
            if task in tested_est:
                est = tested_est[task]
                num_expected += w * est
                num_worst += w * est
                num_best += w * est
                num_projected += w * est
            else:
                pool_avg = impute_by_pin[task]
                num_expected += w * pool_avg
                num_best += w * 1.0
                num_projected += w * min(1.0, max(0.0, pool_avg + bias))
        expected = num_expected / full_weight if full_weight else 0
        floor = num_worst / full_weight if full_weight else 0
        ceiling = num_best / full_weight if full_weight else 0
        projected = num_projected / full_weight if full_weight else 0
        pass_tries = [row['steps'] for row in shortlist_rows
                      if row['model'] == m and row['pass'] and row['steps'] is not None]
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

    quality_by_pin = {d['_task']: d['quality'] for d in usefulness_ex}
    pins_by_quality = sorted(pins, key=lambda p: -quality_by_pin.get(p, -1))

    matrix = []
    for task in pins_by_quality:
        variant, lang, ex = task
        label = pin_label[task]
        cells = []
        for m in models_for_matrix:
            sub = [row for row in shortlist_rows if row['model'] == m and row['task'] == task]
            if not sub:
                cells.append(None)
                continue
            npass = sum(1 for row in sub if row['pass'])
            pass_tries = [row['steps'] for row in sub if row['pass'] and row['steps'] is not None]
            avg_tries = round(sum(pass_tries) / len(pass_tries), 2) if pass_tries else None
            timeouts = sum(1 for row in sub if row['timed_out'])
            ctx_runs = sum(1 for row in sub if row['ctx_exhausted'] > 0)
            cells.append({'pass': npass, 'total': len(sub), 'rate': npass / len(sub), 'avg_tries': avg_tries,
                           'timeouts': timeouts, 'ctx_runs': ctx_runs})
        matrix.append({'exercise': ex, 'language': lang, 'variant': variant, 'label': label,
                        'quality': quality_by_pin[task], 'cells': cells})

    for d in usefulness_ex:
        del d['_task']

    perf['matrix_models'] = matrix_models
    perf['exercise_order'] = [pin_label[p] for p in pins_by_quality]

    return {
        'summary': summary, 'eff_global': round(eff_global, 4), 'chart1': chart1, 'matrix_models': matrix_models, 'matrix': matrix,
        'model_quant': model_quant, 'chart3': chart3, 'chart4': chart4,
        # end reason -> pass / censored / fail / excluded (outcomes.py), for the budget curves
        'end_reason_groups': {k: g for k, (g, _) in outcomes.END_REASONS.items()},
        'end_reason_labels': {k: label for k, (_, label) in outcomes.END_REASONS.items()},
        'flag_labels': outcomes.FLAGS,
        'good_flags': sorted(outcomes.GOOD_FLAGS),
        'flag_runners': outcomes.FLAG_RUNNERS,
        'low_trust': LOW_TRUST, 'low_trust_task_share': LOW_TRUST_TASK_SHARE,
        'fail_n': len(fail_rows_all), 'detail': detail,
        'reasoning': reasoning_lengths(rows),
        'perf': perf, 'usefulness': {'exercises': usefulness_ex}, 'cost_spread': cost_spread,
        # Labels for the exercise filter's initial selection (DEFAULT_TOP_N).
        'default_exercises': [pin_label[p] for p in default_pins],
        # each variant also offered as a list: the runner/suite filter
        'test_lists': [{'name': f'Runner: {v}', 'exercises': [pin_label[t] for t in pins if t[0] == v], 'missing': 0}
                       for v in sorted({t[0] for t in pins})] + load_test_lists(pin_label),
    }


def load_test_lists(pin_label):
    # Every reusable list in the eval repo's lists/ (not lists/tmp/) that looks like a test list: '#' comments, then
    # '<language> <exercise> [max_tries] [max_duration]' whitespace-separated. Offered in
    # the filter panel as quick selections; tests no model has run yet are dropped, and
    # a file with none left over is skipped entirely.
    lists = []
    # (TB2 rows are just '<task> [calls] [max_dur]')
    seen = set()
    for path in sorted(glob.glob('../../lists/*.txt')):
        if os.path.basename(path) in seen:
            continue
        seen.add(os.path.basename(path))
        labels, missing = [], 0
        for line in open(path):
            parts = line.split('#')[0].split()
            if not parts:
                continue
            # list rows name an exercise, not a runner: every variant of it is selected
            # (TB2 rows are a lone task name: language 'tb2')
            hits = [lbl for t, lbl in pin_label.items()
                    if t[1:] == tuple(parts[:2]) or (t[1] == 'tb2' and t[2] == parts[0])]
            for lbl in hits:
                if lbl not in labels:
                    labels.append(lbl)
            if not hits:
                missing += 1
        if labels:
            lists.append({'name': os.path.basename(path)[:-4], 'exercises': labels, 'missing': missing})
    return lists


def main():
    rows, excluded, recent = load_attempts()
    data = build_dataset(rows, excluded)
    data['recent_runs'] = recent
    data['model_settings'] = model_settings(rows)
    data['fx'] = fx_export(rows)

    template = open(TEMPLATE_PATH).read()
    assert template.count('__DATA__') == 1, 'template must contain exactly one __DATA__ placeholder'
    html = template.replace('__DATA__', json.dumps(data))
    open(OUTPUT_PATH, 'w').write(html)
    print(f'wrote {OUTPUT_PATH} ({len(html):,} bytes) from {len(rows)} attempts ({excluded} harness failures excluded), '
          f'{data["summary"]["total_tasks"]} tasks total ({data["summary"]["default_tasks"]} default-shown)')


if __name__ == '__main__':
    main()
