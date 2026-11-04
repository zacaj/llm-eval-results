#!/usr/bin/env python3
"""Outcome taxonomy shared by the aider and Harbor runners (viz-harbor-plan.md §5).

Every attempt gets exactly one END REASON and any number of FLAGS:

  end reason   why the attempt ended / how it should count
  flag         something else that happened along the way (passes can have them too)

End reasons fall into groups:
  pass       passed
  censored   stopped by a budget (tries / test calls / time / tokens) before passing:
             "didn't pass within X", not "can't pass" -- see §2 of the plan
  fail       ended on its own without passing (the agent declared done, or broke)
  excluded   the harness/server failed, not the model: kept out of every score

    python3 outcomes.py                     # end-reason / flag tables for both runners
    python3 outcomes.py --sample done_wrong # list attempts with that end reason (or flag)

Aider flags partly come from the hist_* columns: counts of patterns in each attempt's
.aider.chat.history.md, written by run.sh track_per_test (backfilled 2026-10-28 for
the attempts whose logs were still on disk; blank = unknown). They exist because the
TSV counters mix things up: `user_asks` counts aider's own auto-answered confirm
prompts (lint fix, create file, open URL), not model questions, and `error_outputs`
mixes server errors (auth, model loading, connection) with model-side problems.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

END_REASONS = {  # key -> (group, label)
    "pass":       ("pass",     "Passed"),
    "exhausted":  ("censored", "Used every try / test call"),
    "timeout":    ("censored", "Hit the time limit"),
    "budget":     ("censored", "Hit the output-token budget"),
    "cut_off":    ("censored", "Cut off mid-reasoning, then stopped"),
    "done_wrong": ("fail",     "Agent declared done, tests fail"),
    "glitch":     ("fail",     "Broken output ended the episode"),
    "harness":    ("excluded", "Harness / server failure"),
}

FLAGS = {
    "context":        "Exhausted the context window",
    "output_limit":   "A reply hit the output-token limit",
    "reasoning_cut":  "Thinking was cut off by the reasoning budget",
    "malformed":      "Malformed edit / tool call",
    "edit_errors":    "Edit tool errors",
    "lint":           "Edit didn't parse (aider lint retry)",
    "wrong_file":     "Edited a file outside the exercise",
    "build_failed":   "Final code didn't build (0 tests ran)",
    "test_timeout":   "Tests timed out (solution loops / too slow)",
    "lazy":           "Left lazy placeholder comments",
    "markup_final":   "Last message was raw tool markup",
    "server_error":   "Server errors during the run",
    "slow_server":    "Server ran unusually slow",
    "proxy_rejected": "Requests rejected by the eval proxy",
    "knows_source":   "Knows the exercise (named Exercism / its specs in its own text)",
    "knows_early":    "Knew the exercise within its first 4 steps",
    "recalled_content": "Recalled the upstream solution or tests (code / test cases after naming it)",
    "fetched_source": "Downloaded upstream exercise files (sandbox has network)",
    "recalled_inferred": "Same model recalled this exercise's solution in Harbor (no aider trajectory to check)",
    "internet":       "Fetched an outside URL itself (curl/wget/WebFetch...; package installs not counted)",
    "source_in_prompt": "The instruction itself names Exercism / its specs",
    "tool_errors":    "Tool calls returned errors",
    "code_in_think":  "Wrote code in its thinking (25%+ of thinking chars in code fences)",
    "copied_from_think": "A big Write/Edit (10+ lines) was copied from earlier thinking (80%+ of its lines)",
    "think_spill":    "Thinking spilled into the answer (a <think> tag in text / tool input)",
    "looped":         "Thinking / text looped (same lines 5+ times)",
    "loop_stuck":     "A loop never recovered: the block ended still repeating (ran into a token cap)",
    "max_tokens":     "A request stopped at max_tokens",
    "never_tested":   "Never ran the tests (no test command, no run-tests call)",
}

# which runners can raise each flag (the viz suffixes single-runner ones ·A / ·H and
# rates them over that runner's attempts only)
AIDER_FLAGS = {"recalled_inferred", "context", "output_limit", "malformed", "lint", "wrong_file", "server_error", "test_timeout", "lazy"}
HARBOR_FLAGS = {"malformed", "edit_errors", "reasoning_cut", "markup_final", "slow_server", "proxy_rejected",
                "build_failed", "knows_source", "knows_early", "recalled_content", "fetched_source", "internet", "source_in_prompt", "tool_errors", "never_tested",
                "think_spill", "looped", "loop_stuck", "max_tokens",
                "code_in_think", "copied_from_think"}
FLAG_RUNNERS = {f: sorted(r for r, fs in (("aider", AIDER_FLAGS), ("harbor", HARBOR_FLAGS)) if f in fs) for f in FLAGS}

# How representative an attempt's result is of the model's own ability, 0..1 (not
# `invalid`, which is a harness fault). It weights the attempt in capability scores;
# flags, end reasons and exercise-quality stats still count it in full. Each value is
# 1 - (how misleading the result is if the flag is right) x (how sure the detection
# is): downloading the answer is ~fully misleading and reliably detected; recalled
# code / test cases are about as misleading but the regex is looser.
TRUST = {
    "fetched_source":   0.05,
    "recalled_content": 0.25,
    "internet":         0.5,    # outside fetch, not (detectably) of the exercise
}


# Model / quant degeneration signs only: how broken the output was, not how well it did
# on the tests (no pass/fail, build failures, never-tested), not budgets (reasoning_cut,
# max_tokens, context), not recall, not server/harness. Weight = how bad one occurrence
# is, 0..1. An attempt's instability combines its signs noisy-OR style
# (1 - prod(1 - w)), so several mild ones add up but never pass 1; a model's stability
# is 1 - its mean instability, so it reflects both severity and frequency.
GLITCH_WEIGHTS = {
    "loop_stuck":   1.0,   # degenerated and never recovered
    "think_spill":  0.7,   # thinking tags leaked into the answer / tool input
    "markup_final": 0.6,   # ended on raw tool markup
    "looped":       0.4,   # repeated itself but recovered
    "malformed":    0.4,   # malformed tool call / edit block
    "lint":         0.3,   # edit didn't parse
    "wrong_file":   0.3,
    "lazy":         0.2,   # placeholder comments
    "edit_errors":  0.15,  # often just stale old_string, but frequent in broken quants
}
GLITCH_END_WEIGHT = 1.0   # end reason "glitch": broken output ended the episode


def instability(flags, end_reason=None) -> float:
    ok = 1.0
    for f in flags:
        ok *= 1 - GLITCH_WEIGHTS.get(f, 0)
    if end_reason == "glitch":
        ok *= 1 - GLITCH_END_WEIGHT
    return 1 - ok


def trust(flags) -> float:
    return min([TRUST[f] for f in flags if f in TRUST], default=1.0)


CODE_IN_THINK_FRAC = 0.25

nz = lambda v: (v or "").strip() not in ("", "0", "false", "0.0", "False")


# ---------------------------------------------------------------- aider

# hist_* column (run.sh track_per_test) -> signal
HIST_COLUMNS = {
    "hist_auth_errors": "auth_error", "hist_server_errors": "server_error",
    "hist_context": "context", "hist_output_limit": "output_limit",
    "hist_edit_format": "malformed", "hist_lint": "lint", "hist_wrong_file": "wrong_file",
}


def classify_aider(row: dict) -> tuple[str, set[str]]:
    h = {sig for col, sig in HIST_COLUMNS.items() if nz(row.get(col))}
    flags = {k for k in ("context", "output_limit", "malformed", "lint", "wrong_file", "server_error") if k in h}
    if nz(row.get("exhausted_context_windows")):
        flags.add("context")
    if nz(row.get("num_malformed_responses")):
        flags.add("malformed")
    if nz(row.get("test_timeouts")):
        flags.add("test_timeout")
    if nz(row.get("lazy_comments")):
        flags.add("lazy")

    # 0 tries = it ended inside its first try (no test result yet): a harness failure only
    # if the model never produced anything (server hung / never started); with tokens it
    # was working and ran out of time
    no_output = row.get("tries_taken") == "0" and not nz(row.get("completion_tokens"))
    if no_output or "auth_error" in h:
        return "harness", flags
    if row["pass"] == "true":
        return "pass", flags
    if row.get("bailed_on_max_duration") == "true":
        return "timeout", flags
    # aider never stops on its own before its last try: every other fail used them all
    return "exhausted", flags


def aider_attempts(path: Path = HERE.parent / "aider-results-per-test.tsv"):
    for r in csv.DictReader(path.open(), delimiter="\t"):
        end, flags = classify_aider(r)
        yield r, end, flags


# ---------------------------------------------------------------- harbor

HARNESS_EXCEPTIONS = {"CancelledError", "AgentSetupTimeoutError"}


def classify_harbor(row: dict) -> tuple[str, set[str]]:
    flags = set()
    if nz(row.get("malformed_calls")):
        flags.add("malformed")
    if nz(row.get("edit_errors")):
        flags.add("edit_errors")
    if nz(row.get("thinking_cutoffs")) or nz(row.get("px_budget_cutoffs")):
        flags.add("reasoning_cut")
    if nz(row.get("markup_final")):
        flags.add("markup_final")
    for col, flag in (("think_spills", "think_spill"), ("loop_blocks", "looped"),
                      ("loop_unrecovered", "loop_stuck"), ("px_max_tokens", "max_tokens")):
        if nz(row.get(col)):
            flags.add(flag)
    if nz(row.get("writes_from_think")):
        flags.add("copied_from_think")
    tc, tcode = row.get("think_chars") or "", row.get("think_code_chars") or ""
    if tc.isdigit() and tcode.isdigit() and int(tc) and int(tcode) / int(tc) >= CODE_IN_THINK_FRAC:
        flags.add("code_in_think")
    if nz(row.get("px_slow_requests")):
        flags.add("slow_server")
    if nz(row.get("px_rejected")):
        flags.add("proxy_rejected")
    if nz(row.get("recall_steps")):
        flags.add("knows_source")
        if (row.get("recall_first_step") or "").isdigit() and int(row["recall_first_step"]) <= 4:
            flags.add("knows_early")
    if nz(row.get("recall_content")):
        flags.add("recalled_content")
    if nz(row.get("fetch_source_ok")):
        flags.add("fetched_source")
    if nz(row.get("net_calls")):
        flags.add("internet")
    if nz(row.get("instr_names_source")):
        flags.add("source_in_prompt")
    if nz(row.get("tool_errors")):
        flags.add("tool_errors")
    if row.get("test_runs") == "0" and not nz(row.get("test_calls")):
        flags.add("never_tested")
    if row.get("reward") == "0" and (row.get("tests_parsed") == "0" or row.get("tests_passed") in ("", "0")
                                     and row.get("tests_total") in ("", "0")):
        flags.add("build_failed")

    exc = row.get("exception", "")
    if row.get("invalid") or row.get("reward") == "" or exc in HARNESS_EXCEPTIONS:
        return "harness", flags
    if row["reward"] == "1":
        return "pass", flags
    if row.get("budget_hit") == "1":
        return "budget", flags
    # the agent's processes are killed after its last run-tests call (exit 137 ->
    # NonZeroAgentExitCodeError): that's the test-call limit, not a crash
    if row.get("test_limit_hit") == "1":
        return "exhausted", flags
    if exc == "AgentTimeoutError":
        return "timeout", flags
    if row.get("final_turn_cutoff") == "1":
        return "cut_off", flags
    if row.get("markup_final") == "1" or exc:
        return "glitch", flags
    # final_no_action is set on every clean finish (passes too): it just means the agent
    # ended its episode itself, so a fail here is the agent declaring done wrongly
    return "done_wrong", flags


def harbor_attempts(path: Path = HERE.parent / "harbor-trials.tsv"):
    for r in csv.DictReader(path.open(), delimiter="\t"):
        end, flags = classify_harbor(r)
        yield r, end, flags


# ---------------------------------------------------------------- report

def report(name: str, attempts: list) -> None:
    ends = Counter(e for _, e, _ in attempts)
    print(f"\n== {name}: {len(attempts)} attempts")
    for k, (g, label) in END_REASONS.items():
        if ends[k]:
            print(f"  {g:9} {k:11} {ends[k]:5}  {label}")
    fl, fl_fail = Counter(), Counter()
    for _, e, f in attempts:
        for x in f:
            fl[x] += 1
            fl_fail[x] += e != "pass"
    print("  flags (attempts / of which not passed):")
    for k, n in fl.most_common():
        print(f"    {k:15} {n:5} {fl_fail[k]:5}  {FLAGS[k]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", help="end reason or flag to list attempts for")
    ap.add_argument("-n", type=int, default=20)
    a = ap.parse_args()
    runs = {"aider": list(aider_attempts()), "harbor": list(harbor_attempts())}
    if not a.sample:
        for name, att in runs.items():
            report(name, att)
        return
    for name, att in runs.items():
        hits = [x for x in att if x[1] == a.sample or a.sample in x[2]]
        print(f"\n== {name}: {len(hits)} with {a.sample}")
        for r, e, f in hits[: a.n]:
            where = (r.get("run_cmd") or r.get("job", ""))[:55]
            who = r.get("trial") or f"{r['language']}/{r['exercise']}"
            print(f"  {e:10} {where:55} {who:45} {','.join(sorted(f))}")


if __name__ == "__main__":
    main()
