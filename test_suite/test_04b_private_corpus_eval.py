"""
Semantic retrieval evaluation against the private benchmark corpus.

Companion to test_04_semantic_eval.py, same three metrics (P@5, R@10, MRR),
different corpus. test_04 runs against the public library-catalog-app, which
anyone can clone and reproduce, but being public it may sit inside model
training data. This file scores the private corpus, whose whole point is that
no model has ever seen it.

PRIVACY: the corpus and its question bank are private repositories. Nothing
from either may appear in this file or in committed output — no question
text, no expected answers, no evidence paths, no symbol names. Both are
located at runtime through environment variables with NO defaults, because a
default value would have to name the private repos:

    LOCALSCOPE_CORPUS         root of the corpus checkout to index
    LOCALSCOPE_QUESTION_BANK  directory of question bank YAML files

Unset means every test here skips. Run with:

    LOCALSCOPE_CORPUS=/path/to/corpus \
    LOCALSCOPE_QUESTION_BANK=/path/to/question-bank \
    python3 -m pytest test_suite/test_04b_private_corpus_eval.py -v -s

This measures retrieval only. run_query() on this branch always finishes by
generating an answer with the chat model, which costs ~10s per question and
contributes nothing to retrieval metrics, so the fixture mirrors run_query's
retrieval stage instead: embed the question, query the collection wide,
deduplicate to the best chunk per source file, cut to top_k.
"""

import os

import pytest
import yaml

from test_04_semantic_eval import precision_at_k, recall_at_k, mrr

TOP_K = 10

# Questions anchored to historical tags or branches; their answer does not
# exist at the corpus tip, so scoring them against a tip index would mark a
# correct tool wrong. The first five are flagged in the bank's README;
# Q-E13-017 is anchored to a historical ref too and scored zero for every arm tried,
# including search agents. IDs only — they reveal nothing.
_NON_TIP_ANCHORED = {
    "Q-E4-007", "Q-E4-008", "Q-E4-010", "Q-E67-007", "Q-E67-012",
    "Q-E13-017",
}


def corpus_index_exts() -> set[str]:
    """The engine's default extensions plus JSP.

    main does not index .jsp, but the corpus has live JSP screens that some
    questions' evidence points at. Leaving them out scores a coverage gap as
    a retrieval failure, and search agents compared against this baseline
    can read them.
    """
    from indexing.indexer import DEFAULT_INDEX_EXTS
    return set(DEFAULT_INDEX_EXTS) | {".jsp"}


def build_index(corpus: str, index_dir: str) -> None:
    from indexing.incremental_indexer import index_repo_incremental

    index_repo_incremental(
        repo_root=corpus,
        index_dir=index_dir,
        index_exts=corpus_index_exts(),
        force_full_reindex=True,
        num_workers=2,  # mxbai-embed-large deadlocks Ollama at 4
        verbose=False,
        log=lambda _: None,
    )


def retrieve_sources(index_dir: str, questions: list[dict],
                     query_prefix: str = "", top_k: int = TOP_K) -> dict:
    """question id -> ordered list of retrieved source files (up to top_k).

    Mirrors run_query's retrieval stage: embed, fetch 3x wide, keep the best
    chunk per source file. query_prefix is prepended to the question only,
    for embedding models that expect a query instruction.
    """
    import chromadb
    from chromadb.config import Settings

    from querying.query_engine import _embed_text

    client = chromadb.PersistentClient(
        path=index_dir, settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(client.list_collections()[0].name)

    results = {}
    for q in questions:
        embedding = _embed_text(query_prefix + q["question"], lambda _: None)
        assert embedding is not None, f"{q['id']}: embedding failed"

        res = collection.query(
            query_embeddings=[embedding],
            n_results=top_k * 3,
            include=["metadatas"],
        )
        sources: list[str] = []
        for meta in res.get("metadatas", [[]])[0]:
            source = meta.get("source", "")
            if source and source not in sources:
                sources.append(source)
            if len(sources) == top_k:
                break
        results[q["id"]] = sources
    return results


def per_question(questions: list[dict], results: dict) -> dict:
    """question id -> {"p5", "r10", "mrr", "files"} for one arm."""
    out = {}
    for q in questions:
        sources = results[q["id"]]
        expected = set(q["evidence_files"])
        out[q["id"]] = {
            "p5": precision_at_k(sources, expected, 5),
            "r10": recall_at_k(sources, expected, 10),
            "mrr": mrr(sources, expected),
            "files": len(sources),
        }
    return out


def score(questions: list[dict], results: dict) -> dict:
    """Mean P@5 / R@10 / MRR and returned-file count over questions."""
    rows = per_question(questions, results).values()
    n = len(questions)
    means = {k: sum(r[k] for r in rows) / n for k in ("p5", "r10", "mrr", "files")}
    return {"n": n, **means}


def _corpus_root() -> str | None:
    path = os.environ.get("LOCALSCOPE_CORPUS", "")
    return os.path.expanduser(path) if path else None


def _bank_dir() -> str | None:
    path = os.environ.get("LOCALSCOPE_QUESTION_BANK", "")
    return os.path.expanduser(path) if path else None


def load_retrieval_questions(bank_dir: str) -> list[dict]:
    """All questions a tip-indexed, no-git-history tool can fairly be scored on.

    Drops `hist` questions (they require git history access, which this tool
    does not have) and the tag/branch-anchored IDs above.
    """
    questions: list[dict] = []
    for name in sorted(os.listdir(bank_dir)):
        if not name.endswith((".yaml", ".yml")):
            continue
        with open(os.path.join(bank_dir, name), encoding="utf-8") as fh:
            entries = yaml.safe_load(fh) or []
        for q in entries:
            if q.get("category") == "hist":
                continue
            if q.get("id") in _NON_TIP_ANCHORED:
                continue
            questions.append(q)
    return questions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def corpus_questions():
    bank = _bank_dir()
    if not bank:
        pytest.skip("LOCALSCOPE_QUESTION_BANK not set")
    if not os.path.isdir(bank):
        pytest.fail(f"LOCALSCOPE_QUESTION_BANK points to a missing directory: {bank}")
    questions = load_retrieval_questions(bank)
    if not questions:
        pytest.fail(f"No retrieval-eligible questions found in {bank}")
    return questions


@pytest.fixture(scope="session")
def corpus_index(tmp_path_factory):
    corpus = _corpus_root()
    if not corpus:
        pytest.skip("LOCALSCOPE_CORPUS not set")
    if not os.path.isdir(corpus):
        pytest.fail(f"LOCALSCOPE_CORPUS points to a missing directory: {corpus}")

    index_dir = str(tmp_path_factory.mktemp("corpus_chroma"))
    build_index(corpus, index_dir)
    return index_dir


@pytest.fixture(scope="session")
def corpus_results(corpus_index, corpus_questions) -> dict:
    return retrieve_sources(corpus_index, corpus_questions)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_every_question_retrieves_something(corpus_questions, corpus_results):
    """Every question must retrieve at least one file.

    A full budget of TOP_K is NOT asserted, because the engine itself can
    under-fill it: run_query fetches top_k*3 chunks and deduplicates to one
    per file, so a couple of chunk-heavy files can leave fewer than top_k
    unique sources (observed on 6 of 50 questions). The report prints the
    per-question count so the shortfall stays visible; any A/B comparison
    against these numbers must hold the returned count constant per question.
    """
    empty = [q["id"] for q in corpus_questions if not corpus_results[q["id"]]]
    assert not empty, f"questions retrieving nothing: {empty}"


def test_corpus_eval_report(corpus_questions, corpus_results):
    """Per-question and mean P@5 / R@10 / MRR, plus returned-file counts.

    Reporting only — the corpus baseline is whatever this prints, and
    pass/fail thresholds belong to comparisons, not to a baseline read.
    """
    print("\n\n=== Private Corpus Retrieval Evaluation ===\n")
    print(f"{'Question':<14} {'cat':<7} {'lang':<6} {'P@5':>6} {'R@10':>6} {'MRR':>6} {'files':>6}")
    print("-" * 56)

    p5s, r10s, mrrs, counts = [], [], [], []
    for q in corpus_questions:
        sources = corpus_results[q["id"]]
        expected = set(q["evidence_files"])
        p5 = precision_at_k(sources, expected, 5)
        r10 = recall_at_k(sources, expected, 10)
        m = mrr(sources, expected)
        p5s.append(p5); r10s.append(r10); mrrs.append(m); counts.append(len(sources))
        print(f"  {q['id']:<12} {q.get('category', '?'):<7} {q.get('language', '?'):<6} "
              f"{p5:>6.2f} {r10:>6.2f} {m:>6.2f} {len(sources):>6d}")

    n = len(corpus_questions)
    print("-" * 56)
    print(f"  {'MEAN (n=' + str(n) + ')':<28} "
          f"{sum(p5s) / n:>6.2f} {sum(r10s) / n:>6.2f} {sum(mrrs) / n:>6.2f} "
          f"{sum(counts) / n:>6.1f}")
    print("=" * 56)
