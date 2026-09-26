"""
Scale test on a large public repository with a PR-harvested question bank.

Everything else in the 04 series runs on a private corpus of a few hundred
files. This asks whether the results hold on a real codebase more than
twenty times larger, written by many people, with ground truth nobody on
this project authored: each question is a merged pull request's
description, and its answer is the non-test source files that PR changed
(see harvest_github_bank.py).

Three arms, same sampled questions, same 10-file budget, same metrics:

    grep, one pass    the question's own words, ranked by rare-term matches
    semantic index    the engine's embedding retrieval
    agent             a local model searching with grep/find/read in a loop

Caveats this test cannot remove: the repository is public, so models may
have seen it in training, and "the files a PR changed" is a proxy for "the
files needed to understand the question". PR descriptions also vary widely
in quality. Compare arms with each other, not with the private corpus.

    LOCALSCOPE_PR_CORPUS=/path/to/checkout \\
    LOCALSCOPE_PR_BANK=/path/to/bank.json \\
    python3 -m pytest test_suite/test_04h_public_repo_scale.py -v -s

Optional:
    LCQ_SCALE_N          questions to sample (default 50, seed 0)
    LCQ_SCALE_INDEX_DIR  persistent index directory, reused if it exists
    LCQ_AGENT_MODELS     agent models (default qwen3.6:35b-a3b)
    LCQ_AGENT_OUT        write every arm's per-question files as JSON
    LCQ_SCALE_HYBRID=1   give the agent the index too: its top files for the
                         question in the first message, plus a
                         semantic_search tool (as in test_04g)
    LCQ_SCALE_SEED=trust with HYBRID, present the index's files as the likely
                         answer to confirm, not a starting point to verify
"""

import json
import math
import os
import random
import time

import pytest

from querying.agent_search import Repo, run_agent
from test_04b_private_corpus_eval import (
    TOP_K,
    build_index,
    per_question,
    retrieve_sources,
    score,
)
from test_04d_private_corpus_grep import grep_rank, load_corpus, query_terms
from test_04g_private_corpus_hybrid_agent import make_semantic

DEFAULT_AGENT_MODELS = ["qwen3.6:35b-a3b"]


def _env_path(name: str) -> str | None:
    value = os.environ.get(name, "")
    return os.path.expanduser(value) if value else None


def _paired_p(deltas: list[float], trials: int = 20000) -> float:
    observed = abs(sum(deltas))
    if observed == 0:
        return 1.0
    rng = random.Random(0)
    extreme = sum(abs(sum(d if rng.random() < 0.5 else -d for d in deltas)) >= observed
                  for _ in range(trials))
    return (extreme + 1) / (trials + 1)


def _save(raw: dict) -> None:
    out = _env_path("LCQ_AGENT_OUT")
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=1)


def test_public_repo_scale(tmp_path_factory):
    corpus, bank = _env_path("LOCALSCOPE_PR_CORPUS"), _env_path("LOCALSCOPE_PR_BANK")
    if not corpus or not bank:
        pytest.skip("LOCALSCOPE_PR_CORPUS and LOCALSCOPE_PR_BANK must both be set")

    import config
    config.RERANK_ENABLED, config.SECOND_ROUND = False, "off"

    with open(bank, encoding="utf-8") as fh:
        questions = json.load(fh)
    n = int(os.environ.get("LCQ_SCALE_N", "50"))
    if n < len(questions):
        questions = random.Random(0).sample(questions, n)
    models = [m.strip() for m in os.environ.get("LCQ_AGENT_MODELS", "").split(",")
              if m.strip()] or DEFAULT_AGENT_MODELS

    raw: dict[str, dict] = {}
    arms = []

    # grep, one pass
    files = load_corpus(corpus)
    all_terms = {t for q in questions for t in query_terms(q["question"])}
    idf = {t: math.log(len(files) / (1 + sum(1 for text in files.values() if t in text)))
           for t in all_terms}
    grep_results = {q["id"]: grep_rank(files, query_terms(q["question"]), idf) for q in questions}
    arms.append(("grep, one pass", grep_results, None, None))
    raw["grep"] = grep_results
    del files

    # semantic index (built once, reused across runs if a directory is given)
    index_dir = _env_path("LCQ_SCALE_INDEX_DIR") or str(tmp_path_factory.mktemp("scale_index"))
    built = False
    if not (os.path.isdir(index_dir) and os.listdir(index_dir)):
        start = time.monotonic()
        build_index(corpus, index_dir)
        built = time.monotonic() - start
        print(f"\n    indexed in {built / 60:.1f} min", flush=True)
    index_results = retrieve_sources(index_dir, questions)
    arms.append(("semantic index", index_results, None, None))
    raw["index"] = index_results
    _save(raw)

    # agents
    repo = Repo(corpus)
    hybrid = os.environ.get("LCQ_SCALE_HYBRID") == "1"
    semantic = make_semantic(index_dir) if hybrid else None
    seed_mode = os.environ.get("LCQ_SCALE_SEED", "verify") if hybrid else False
    for model in models:
        results, runs = {}, []
        for q in questions:
            run = run_agent(q["question"], repo, model, top_k=TOP_K,
                            semantic=semantic, seed=seed_mode, log=lambda _: None)
            results[q["id"]] = run.files
            runs.append(run)
            raw.setdefault(model, {})[q["id"]] = {
                "files": run.files, "tool_calls": run.tool_calls,
                "submitted": run.submitted, "seconds": round(run.seconds, 1),
                "error": run.error,
            }
            print(f"    {model} {q['id']}: {run.tool_calls} calls, {run.seconds:.0f}s"
                  f"{'' if run.submitted else ', no submit'}", flush=True)
            _save(raw)
        arms.append((f"agent{f' + index ({seed_mode})' if hybrid else ''} {model}", results,
                     sum(r.tool_calls for r in runs) / len(runs),
                     sum(r.seconds for r in runs) / len(runs)))

    evidence = [len(q["evidence_files"]) for q in questions]
    print("\n\n=== Public Repository Scale Test ===\n")
    print(f"  n={len(questions)} questions sampled, {len(repo.files)} files, "
          f"{sum(evidence) / len(evidence):.1f} evidence files per question"
          f"{f', index built in {built / 60:.1f} min' if built else ''}\n")
    print(f"{'Arm':<28} {'P@5':>6} {'R@10':>6} {'MRR':>6} {'files':>6} {'calls/q':>8} {'s/q':>6}")
    print("-" * 72)
    for label, results, calls, secs in arms:
        s = score(questions, results)
        extra = f"{calls:>8.1f} {secs:>6.0f}" if calls is not None else f"{'':>8} {'':>6}"
        print(f"  {label:<26} {s['p5']:>6.2f} {s['r10']:>6.2f} {s['mrr']:>6.2f} "
              f"{s['files']:>6.1f} {extra}")
    print("-" * 72)

    base = per_question(questions, index_results)
    for label, results, _, _ in arms:
        if label == "semantic index":
            continue
        rows = per_question(questions, results)
        print(f"\n  {label} vs semantic index, per question:")
        for metric in ("p5", "r10", "mrr"):
            deltas = [rows[q["id"]][metric] - base[q["id"]][metric] for q in questions]
            wins, losses = sum(d > 0 for d in deltas), sum(d < 0 for d in deltas)
            print(f"    {metric:<5} mean {sum(deltas) / len(deltas):+.3f}   better {wins:>2}  "
                  f"worse {losses:>2}  same {len(deltas) - wins - losses:>2}   "
                  f"p={_paired_p(deltas):.3f}")
    print("=" * 72)
