"""
Local-model search agents on the private benchmark corpus.

The direct counterpart of the Claude agent run: each question goes to a local
model with read-only search tools (grep, find_files, read_file) and no
retrieval index at all. It searches as many rounds as it likes, up to a step
cap, then submits up to 10 files, which are scored with the same helpers as
every other private-corpus test.

A Claude Sonnet agent doing the same job reached P@5 0.46, R@10 0.96,
MRR 0.92 with about 6 searches per question; the one-pass semantic index
reaches 0.28 / 0.75 / 0.72. This measures where local models land between
and whether the loop, rather than the model, carries the result.

Reported per model: the metrics, average files returned, tool calls, seconds
per question, and how many runs never called submit (those fall back to the
files the agent read, in the order it read them).

Same privacy rules and environment variables as test_04b; unset means skip.

    LOCALSCOPE_CORPUS=/path/to/corpus \\
    LOCALSCOPE_QUESTION_BANK=/path/to/question-bank \\
    python3 -m pytest test_suite/test_04f_private_corpus_local_agent.py -v -s

Optional:
    LCQ_AGENT_MODELS     comma separated Ollama models (default below)
    LCQ_AGENT_QUESTIONS  comma separated question ids to run a subset
    LCQ_AGENT_OUT        path to write every run's raw result as JSON
"""

import json
import os

import pytest

from querying.agent_search import Repo, run_agent
from test_04b_private_corpus_eval import (
    TOP_K,
    _bank_dir,
    _corpus_root,
    load_retrieval_questions,
    score,
)

# The practical laptop size and the strongest local model on this machine.
DEFAULT_MODELS = ["qwen2.5:7b", "qwen3.6:27b"]


def _env_list(name: str) -> list[str]:
    return [v.strip() for v in os.environ.get(name, "").split(",") if v.strip()]


def _save(raw: dict) -> None:
    """Write raw results after every question, so a stopped run keeps its work."""
    out = os.environ.get("LCQ_AGENT_OUT")
    if out:
        with open(os.path.expanduser(out), "w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=1)


def test_local_agent_search():
    corpus, bank = _corpus_root(), _bank_dir()
    if not corpus or not bank:
        pytest.skip("LOCALSCOPE_CORPUS and LOCALSCOPE_QUESTION_BANK must both be set")

    questions = load_retrieval_questions(bank)
    subset = set(_env_list("LCQ_AGENT_QUESTIONS"))
    if subset:
        questions = [q for q in questions if q["id"] in subset]
    models = _env_list("LCQ_AGENT_MODELS") or DEFAULT_MODELS
    repo = Repo(corpus)

    rows, raw = [], {}
    for model in models:
        results, runs = {}, []
        for q in questions:
            run = run_agent(q["question"], repo, model, top_k=TOP_K, log=lambda _: None)
            results[q["id"]] = run.files
            runs.append(run)
            raw.setdefault(model, {})[q["id"]] = {
                "files": run.files, "tool_calls": run.tool_calls,
                "submitted": run.submitted, "fallback": run.fallback,
                "seconds": round(run.seconds, 1), "error": run.error,
            }
            print(f"    {model} {q['id']}: {run.tool_calls} calls, {run.seconds:.0f}s"
                  f"{'' if run.submitted else ', no submit'}", flush=True)
            _save(raw)
        n = len(runs)
        rows.append((model, score(questions, results),
                     sum(r.tool_calls for r in runs) / n,
                     sum(r.seconds for r in runs) / n,
                     sum(1 for r in runs if not r.submitted),
                     sum(1 for r in runs if r.error)))

    print("\n\n=== Private Corpus Local Agent Search ===\n")
    print(f"  n={len(questions)} questions, {len(repo.files)} files searchable, "
          f"up to {TOP_K} files submitted\n")
    print(f"{'Model':<16} {'P@5':>6} {'R@10':>6} {'MRR':>6} {'files':>6} "
          f"{'calls/q':>8} {'s/q':>6} {'no submit':>10} {'errors':>7}")
    print("-" * 80)
    for model, s, calls, secs, no_submit, errors in rows:
        print(f"  {model:<14} {s['p5']:>6.2f} {s['r10']:>6.2f} {s['mrr']:>6.2f} "
              f"{s['files']:>6.1f} {calls:>8.1f} {secs:>6.0f} {no_submit:>10} {errors:>7}")
    print("-" * 80)
    print("  reference: semantic index 0.28 / 0.75 / 0.72, "
          "Claude Sonnet agent 0.46 / 0.96 / 0.92")
    print("=" * 80)
