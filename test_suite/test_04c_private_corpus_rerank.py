"""
Cross-encoder reranking A/B on the private benchmark corpus.

Both arms run on the SAME index and the same questions and return the same
number of files (top 10). The only difference is whether a cross-encoder
reorders a wider pool of chunks before the per-file cut. So any difference
in P@5 / R@10 / MRR is ordering quality, not a bigger or smaller net.

Expectations going in: recall can only move within what the wider pool
contains, and precision (0.28 against a ceiling of about 0.52 on this set)
is the metric with the most room.

Alongside the means, each metric gets a paired comparison per question
(wins / losses / ties) and a two-sided sign-flip permutation test on the
mean difference, since 49 questions is small enough that a few points of
difference can easily be noise.

Same privacy rules and environment variables as test_04b; unset means skip.
Reranker models download from Hugging Face on first use.

    LOCALSCOPE_CORPUS=/path/to/corpus \
    LOCALSCOPE_QUESTION_BANK=/path/to/question-bank \
    python3 -m pytest test_suite/test_04c_private_corpus_rerank.py -v -s

LCQ_RERANK_MODELS (comma separated) overrides which rerankers are tried.
"""

import os
import random
import time

import pytest

import config
from test_04b_private_corpus_eval import (
    _bank_dir,
    _corpus_root,
    build_index,
    load_retrieval_questions,
    per_question,
    retrieve_sources,
    score,
)

DEFAULT_RERANKERS = ["BAAI/bge-reranker-v2-m3", "mixedbread-ai/mxbai-rerank-base-v1"]
METRICS = ("p5", "r10", "mrr")


def paired_permutation_p(deltas: list[float], trials: int = 20000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation test for mean(deltas) != 0."""
    observed = abs(sum(deltas))
    if observed == 0:
        return 1.0
    rng = random.Random(seed)
    extreme = 0
    for _ in range(trials):
        if abs(sum(d if rng.random() < 0.5 else -d for d in deltas)) >= observed:
            extreme += 1
    return (extreme + 1) / (trials + 1)


def _arm(index_dir, questions, rerank: bool, model: str | None = None) -> tuple:
    config.RERANK_ENABLED = rerank
    if model:
        config.RERANK_MODEL = model
    if rerank:
        # Load the model outside the timed region so latency is per query.
        from querying.reranker import score_pairs
        score_pairs("warm up", ["warm up"])
    start = time.monotonic()
    results = retrieve_sources(index_dir, questions)
    ms = (time.monotonic() - start) / len(questions) * 1000
    return results, ms


def test_corpus_rerank_ab(tmp_path_factory):
    corpus, bank = _corpus_root(), _bank_dir()
    if not corpus or not bank:
        pytest.skip("LOCALSCOPE_CORPUS and LOCALSCOPE_QUESTION_BANK must both be set")

    questions = load_retrieval_questions(bank)
    rerankers = [
        m.strip() for m in os.environ.get("LCQ_RERANK_MODELS", "").split(",") if m.strip()
    ] or DEFAULT_RERANKERS

    saved = (config.RERANK_ENABLED, config.RERANK_MODEL)
    try:
        index_dir = str(tmp_path_factory.mktemp("corpus_rerank"))
        build_index(corpus, index_dir)

        base_results, base_ms = _arm(index_dir, questions, rerank=False)
        base = per_question(questions, base_results)
        arms = [("vector only (baseline)", score(questions, base_results), base_ms, None)]
        for model in rerankers:
            results, ms = _arm(index_dir, questions, rerank=True, model=model)
            arms.append((f"+ {model}", score(questions, results), ms,
                         per_question(questions, results)))
    finally:
        config.RERANK_ENABLED, config.RERANK_MODEL = saved

    print("\n\n=== Private Corpus Reranking A/B ===\n")
    print(f"  n={len(questions)} questions, same index, top 10 files per arm, "
          f"rerank pool {config.RERANK_POOL} chunks\n")
    print(f"{'Arm':<44} {'P@5':>6} {'R@10':>6} {'MRR':>6} {'files':>6} {'ms/q':>7}")
    print("-" * 80)
    for label, s, ms, _ in arms:
        print(f"  {label:<42} {s['p5']:>6.2f} {s['r10']:>6.2f} {s['mrr']:>6.2f} "
              f"{s['files']:>6.1f} {ms:>7.0f}")
    print("-" * 80)

    for label, _, _, rows in arms[1:]:
        print(f"\n  {label} vs baseline, per question:")
        for metric in METRICS:
            deltas = [rows[q["id"]][metric] - base[q["id"]][metric] for q in questions]
            wins = sum(d > 0 for d in deltas)
            losses = sum(d < 0 for d in deltas)
            ties = len(deltas) - wins - losses
            mean = sum(deltas) / len(deltas)
            p = paired_permutation_p(deltas)
            print(f"    {metric:<5} mean {mean:+.3f}   better {wins:>2}  worse {losses:>2}  "
                  f"same {ties:>2}   p={p:.3f}")
    print("=" * 80)

    for label, s, _, _ in arms:
        assert s["files"] > 0, f"{label}: retrieved nothing"
