"""
Retrieval ablation against an external benchmark corpus + question bank.

This is the experiment the whole graph effort exists to settle: does adding
lexical and graph retrieval actually put the right files in front of the
model more often than vector search alone?

Three arms, same corpus, same questions, same top_k:
    vector    embeddings only (the pre-graph baseline)
    lexical   + identifier full-text index
    hybrid    + symbol routing and one-hop graph expansion

Metric is retrieval hit rate: the fraction of a question's evidence files
that reach the tool's context. Answer quality is deliberately NOT scored
here — that needs the bank's judge-time fields, and retrieval is the thing
this branch changed.

The headline number is the delta on questions the bank predicts graph
should win. The `tie` bucket is the control: those are questions where
graph shouldn't help, so the hybrid arm LOSING there means graph traversal
is injecting noise, which matters more than a win elsewhere.

Corpus and bank are private and live outside this public repo; nothing
from either is committed here. See question_bank.py. Everything skips when
they're absent.

Run:
    python3 -m pytest test_suite/test_18_corpus_eval.py -v -s
"""

import os

import pytest

from question_bank import (
    corpus_dir,
    load_questions,
    question_bank_dir,
    resolvable_at,
    retrieval_eligible,
)

TOP_K = 10

# (arm name, GRAPH_ENABLED, LEXICAL_ENABLED)
ARMS = [
    ("vector", False, False),
    ("lexical", False, True),
    ("hybrid", True, True),
]


# ---------------------------------------------------------------------------
# Skip cleanly unless both private repos are present
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    corpus_dir() is None or question_bank_dir() is None,
    reason=(
        "external corpus and/or question bank unavailable "
        "(set LOCALSCOPE_CORPUS and LOCALSCOPE_QUESTION_BANK)"
    ),
)


@pytest.fixture(scope="session")
def questions():
    qs = load_questions()
    if not qs:
        pytest.skip("question bank empty or PyYAML unavailable")
    eligible = retrieval_eligible(resolvable_at(qs, corpus_dir()))
    if not eligible:
        pytest.skip("no questions resolvable against this corpus checkout")
    return eligible


@pytest.fixture(scope="session")
def corpus_index(tmp_path_factory):
    """Index the external corpus once: vectors, graph, and lexical."""
    from indexing.incremental_indexer import index_repo_incremental
    from indexing.lexical_index import build_lexical_index
    from graph.graph_builder import build_incremental

    corpus = str(corpus_dir())
    index_dir = str(tmp_path_factory.mktemp("corpus_index"))

    index_repo_incremental(
        repo_root=corpus,
        index_dir=index_dir,
        force_full_reindex=True,
        num_workers=2,
        verbose=False,
        log=lambda _: None,
    )
    build_lexical_index(corpus, index_dir, log=lambda _: None)
    build_incremental(corpus, os.path.join(index_dir, "graph.json"), log_fn=lambda _: None)

    return index_dir


@pytest.fixture(scope="session")
def arm_results(questions, corpus_index):
    """Run every question through every arm. Returns {arm: {qid: hit_rate}}."""
    import config
    from querying.query_engine import run_query

    original = (config.GRAPH_ENABLED, config.LEXICAL_ENABLED)
    results: dict[str, dict[str, float]] = {}

    try:
        for arm, graph_on, lexical_on in ARMS:
            config.GRAPH_ENABLED = graph_on
            config.LEXICAL_ENABLED = lexical_on
            per_question: dict[str, float] = {}

            for q in questions:
                try:
                    result = run_query(
                        bug_text=q.question,
                        index_dir=corpus_index,
                        top_k=TOP_K,
                        log=lambda _: None,
                        retrieval_only=True,
                    )
                except Exception:
                    # A query that blows up retrieved nothing; that's a zero,
                    # not a reason to abandon the whole run.
                    per_question[q.id] = 0.0
                    continue

                retrieved = {m.get("source", "") for m in result.get("metas", [])}
                hits = sum(
                    1 for ev in q.evidence_files
                    if any(ev == r or r.endswith(ev) or ev.endswith(r) for r in retrieved)
                )
                per_question[q.id] = hits / len(q.evidence_files)

            results[arm] = per_question
    finally:
        config.GRAPH_ENABLED, config.LEXICAL_ENABLED = original

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _bucket_mean(questions, results, arm, predicate) -> float:
    return _mean(results[arm][q.id] for q in questions if predicate(q))


# ---------------------------------------------------------------------------
# Report — always passes; the numbers are the output
# ---------------------------------------------------------------------------

def test_ablation_report(questions, arm_results):
    arms = [a for a, _, _ in ARMS]

    print("\n\n=== Retrieval Ablation ===\n")
    print(f"Corpus:    {corpus_dir()}")
    print(f"Questions: {len(questions)} retrieval-eligible, top_k={TOP_K}")
    print("Metric:    fraction of a question's evidence files reaching context\n")

    header = f"{'Bucket':<22}" + "".join(f"{a:>10}" for a in arms) + f"{'Δ hybrid':>10}"
    print(header)
    print("-" * len(header))

    def row(label, predicate):
        vals = [_bucket_mean(questions, arm_results, a, predicate) for a in arms]
        n = sum(1 for q in questions if predicate(q))
        if not n:
            return
        delta = vals[-1] - vals[0]
        print(f"  {label + f' (n={n})':<20}" + "".join(f"{v:>10.2f}" for v in vals) + f"{delta:>+10.2f}")

    row("ALL", lambda q: True)
    print()
    for arm_pred in ("graph", "tie", "vector"):
        row(f"predict:{arm_pred}", lambda q, p=arm_pred: q.arm_prediction == p)
    print()
    for cat in sorted({q.category for q in questions}):
        row(f"cat:{cat}", lambda q, c=cat: q.category == c)
    print()
    for lang in sorted({q.language for q in questions}):
        row(f"lang:{lang}", lambda q, l=lang: q.language == l)
    print()
    row("neutral", lambda q: q.is_neutral)
    row("flaw-backed", lambda q: not q.is_neutral)

    print("-" * len(header))
    print("\n  Headline: the predict:graph delta.")
    print("  Control:  predict:tie — hybrid losing there means graph noise.\n")


# ---------------------------------------------------------------------------
# Assertions — the claims worth failing over
# ---------------------------------------------------------------------------

def test_hybrid_does_not_regress_overall(questions, arm_results):
    """Whatever else it does, the hybrid arm must not retrieve worse than
    plain vector search across the bank as a whole."""
    vector = _bucket_mean(questions, arm_results, "vector", lambda q: True)
    hybrid = _bucket_mean(questions, arm_results, "hybrid", lambda q: True)
    assert hybrid >= vector - 0.02, (
        f"hybrid mean hit rate {hybrid:.3f} regressed below vector {vector:.3f}"
    )


def test_control_bucket_not_degraded(questions, arm_results):
    """`tie` questions are the noise check: graph shouldn't help them, but it
    must not hurt them either. Degradation here means graph expansion is
    displacing genuinely relevant files with structurally-adjacent noise."""
    tie = [q for q in questions if q.arm_prediction == "tie"]
    if not tie:
        pytest.skip("no control questions in this bank")

    vector = _bucket_mean(questions, arm_results, "vector", lambda q: q.arm_prediction == "tie")
    hybrid = _bucket_mean(questions, arm_results, "hybrid", lambda q: q.arm_prediction == "tie")
    assert hybrid >= vector - 0.05, (
        f"control bucket degraded: hybrid {hybrid:.3f} vs vector {vector:.3f} "
        f"over {len(tie)} tie questions — graph traversal is adding noise"
    )
