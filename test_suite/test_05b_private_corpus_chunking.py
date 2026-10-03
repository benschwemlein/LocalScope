"""
Chunking strategy comparison on the private benchmark corpus.

The private-corpus counterpart of test_05_chunking_strategies.py. Two
indexes of the same corpus, identical except for how files are split:

    ast    the engine default: code files (Java, TypeScript, JavaScript,
           Python) split along class and method boundaries by astchunk,
           with a file-path context header on each chunk; anything too
           large is split again by characters
    text   every file split by characters with overlap, no structure

Markdown, HTML and JSP files are character-chunked in both arms, since the
AST path only covers code files. Chunk size (800 characters), overlap,
embedding model, questions, retrieval and the 10-file budget are the same,
so any difference is the chunking.

Per-question paired comparison with a two-sided sign-flip permutation test.
Same privacy rules and environment variables as test_04b; unset means skip.

    LOCALSCOPE_CORPUS=/path/to/corpus \\
    LOCALSCOPE_QUESTION_BANK=/path/to/question-bank \\
    python3 -m pytest test_suite/test_05b_private_corpus_chunking.py -v -s
"""

import random
import time

import pytest

from test_04b_private_corpus_eval import (
    _bank_dir,
    _corpus_root,
    corpus_index_exts,
    load_retrieval_questions,
    per_question,
    retrieve_sources,
    score,
)

CHUNK_CHARS = 800
CHUNK_OVERLAP = 200
ARMS = [("ast (default)", True), ("text", False)]


def paired_permutation_p(deltas: list[float], trials: int = 20000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation test for mean(deltas) != 0."""
    observed = abs(sum(deltas))
    if observed == 0:
        return 1.0
    rng = random.Random(seed)
    extreme = sum(
        abs(sum(d if rng.random() < 0.5 else -d for d in deltas)) >= observed
        for _ in range(trials)
    )
    return (extreme + 1) / (trials + 1)


def _index(corpus: str, index_dir: str, use_ast: bool) -> tuple[float, int]:
    """Build one index; return (seconds, chunk count)."""
    import chromadb
    from chromadb.config import Settings

    from indexing.incremental_indexer import index_repo_incremental

    start = time.monotonic()
    index_repo_incremental(
        repo_root=corpus,
        index_dir=index_dir,
        index_exts=corpus_index_exts(),
        chars_per_chunk=CHUNK_CHARS,
        chunk_overlap=CHUNK_OVERLAP,
        use_ast_chunking=use_ast,
        force_full_reindex=True,
        num_workers=2,
        verbose=False,
        log=lambda _: None,
    )
    secs = time.monotonic() - start
    client = chromadb.PersistentClient(path=index_dir, settings=Settings(anonymized_telemetry=False))
    return secs, client.get_collection(client.list_collections()[0].name).count()


def test_chunking_comparison(tmp_path_factory):
    corpus, bank = _corpus_root(), _bank_dir()
    if not corpus or not bank:
        pytest.skip("LOCALSCOPE_CORPUS and LOCALSCOPE_QUESTION_BANK must both be set")

    questions = load_retrieval_questions(bank)
    arms = []
    for label, use_ast in ARMS:
        index_dir = str(tmp_path_factory.mktemp("chunk_ast" if use_ast else "chunk_text"))
        secs, chunks = _index(corpus, index_dir, use_ast)
        results = retrieve_sources(index_dir, questions)
        arms.append((label, score(questions, results), per_question(questions, results), secs, chunks))

    print("\n\n=== Private Corpus Chunking Comparison ===\n")
    print(f"  n={len(questions)} questions, {CHUNK_CHARS} char chunks, "
          f"{CHUNK_OVERLAP} overlap, top 10 files\n")
    print(f"{'Arm':<16} {'P@5':>6} {'R@10':>6} {'MRR':>6} {'files':>6} {'chunks':>7} {'index s':>8}")
    print("-" * 62)
    for label, s, _, secs, chunks in arms:
        print(f"  {label:<14} {s['p5']:>6.2f} {s['r10']:>6.2f} {s['mrr']:>6.2f} "
              f"{s['files']:>6.1f} {chunks:>7} {secs:>8.0f}")
    print("-" * 62)

    (a_label, _, a_rows, _, _), (b_label, _, b_rows, _, _) = arms
    print(f"\n  {b_label} vs {a_label}, per question:")
    for metric in ("p5", "r10", "mrr"):
        deltas = [b_rows[q["id"]][metric] - a_rows[q["id"]][metric] for q in questions]
        wins, losses = sum(d > 0 for d in deltas), sum(d < 0 for d in deltas)
        print(f"    {metric:<5} mean {sum(deltas) / len(deltas):+.3f}   better {wins:>2}  "
              f"worse {losses:>2}  same {len(deltas) - wins - losses:>2}   "
              f"p={paired_permutation_p(deltas):.3f}")

    by_cat: dict[str, list] = {}
    for q in questions:
        by_cat.setdefault(q.get("category", "?"), []).append(q)
    print(f"\n  by category, R@10 / MRR ({a_label} | {b_label}):")
    for cat, qs in sorted(by_cat.items()):
        mean = lambda rows, k: sum(rows[q["id"]][k] for q in qs) / len(qs)
        print(f"    {cat:<7} n={len(qs):>2}   {mean(a_rows, 'r10'):.2f} / {mean(a_rows, 'mrr'):.2f}"
              f"  |  {mean(b_rows, 'r10'):.2f} / {mean(b_rows, 'mrr'):.2f}")
    print("=" * 62)
