"""
Grep-only lower bound on the private benchmark corpus.

No Ollama, no embeddings, no language model. The question's own words are
searched for in the corpus the way `grep -i -F` would, and files are ranked by
what they contain. Anything smarter should beat this; how far above it the
semantic index lands is the real measure of what embeddings are worth here.

Kept deliberately naive so it is a floor rather than a competitor:

- Search terms are the question lowercased and split on non-alphanumerics,
  minus a generic English stopword list and words under 3 characters. No
  stemming, no synonyms, no camelCase splitting, nothing tuned to the bank.
- Matching is case-insensitive substring, like grep: "fee" also matches
  "feedback".
- Files searched are exactly the ones the indexer would index (same
  extensions including .jsp, same excluded directories, same size cap).

Two rankings, both returning at most 10 files, scored with the same helpers
as every other private-corpus test:

  plain grep   number of distinct query terms the file contains, ties broken
               by total occurrences
  grep + IDF   sum over matched terms of log(N / files containing the term),
               so rare words count for more than common ones

Same privacy rules and environment variables as test_04b; unset means skip:

    LOCALSCOPE_CORPUS=/path/to/corpus \\
    LOCALSCOPE_QUESTION_BANK=/path/to/question-bank \\
    python3 -m pytest test_suite/test_04d_private_corpus_grep.py -v -s
"""

import math
import os
import re
import time

import pytest

from test_04b_private_corpus_eval import (
    TOP_K,
    _bank_dir,
    _corpus_root,
    corpus_index_exts,
    load_retrieval_questions,
    score,
)

# Mirrors the indexer's exclusions and size cap so both see the same files.
EXCLUDED_DIRS = {
    ".git", ".idea", ".vscode", "node_modules", "build", "dist", "out",
    "target", ".gradle", ".venv", "venv", "__pycache__",
}
MAX_FILE_BYTES = 500_000

# Generic English function words plus question words. Not derived from the
# question bank.
STOPWORDS = set("""
a about above after again against all also am an and any are as at be because
been before being below between both but by can could did do does doing down
during each few for from further had has have having he her here hers him his
how i if in into is it its itself just me more most my no nor not now of off
on once only or other our ours out over own same she should so some such than
that the their theirs them then there these they this those through to too
under until up very was we were what when where which while who whom why will
with would you your yours yet still actually really currently exactly
""".split())


def query_terms(question: str) -> list[str]:
    words = re.split(r"[^a-z0-9]+", question.lower())
    seen, terms = set(), []
    for w in words:
        if len(w) >= 3 and w not in STOPWORDS and w not in seen:
            seen.add(w)
            terms.append(w)
    return terms


def load_corpus(root: str) -> dict[str, str]:
    """relative path -> lowercased file text, for every indexable file."""
    exts = corpus_index_exts()
    files = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for name in filenames:
            if os.path.splitext(name)[1].lower() not in exts:
                continue
            full = os.path.join(dirpath, name)
            try:
                if os.path.getsize(full) > MAX_FILE_BYTES:
                    continue
                with open(full, encoding="utf-8", errors="ignore") as fh:
                    files[os.path.relpath(full, root)] = fh.read().lower()
            except OSError:
                continue
    return files


def grep_rank(corpus: dict[str, str], terms: list[str], idf: dict[str, float] | None,
              top_k: int = TOP_K) -> list[str]:
    """Files containing at least one term, best first, at most top_k."""
    scored = []
    for path, text in corpus.items():
        matched = [t for t in terms if t in text]
        if not matched:
            continue
        occurrences = sum(text.count(t) for t in matched)
        if idf is None:
            key = (len(matched), occurrences)
        else:
            key = (sum(idf[t] for t in matched), occurrences)
        scored.append((key, path))
    scored.sort(key=lambda s: (-s[0][0], -s[0][1], s[1]))
    return [path for _, path in scored[:top_k]]


def test_grep_lower_bound():
    corpus_root, bank = _corpus_root(), _bank_dir()
    if not corpus_root or not bank:
        pytest.skip("LOCALSCOPE_CORPUS and LOCALSCOPE_QUESTION_BANK must both be set")

    questions = load_retrieval_questions(bank)
    start = time.monotonic()
    corpus = load_corpus(corpus_root)
    load_secs = time.monotonic() - start
    assert corpus, f"no indexable files found under {corpus_root}"

    n_files = len(corpus)
    all_terms = {t for q in questions for t in query_terms(q["question"])}
    idf = {
        t: math.log(n_files / (1 + sum(1 for text in corpus.values() if t in text)))
        for t in all_terms
    }

    rows = []
    for label, weights in (("plain grep", None), ("grep + IDF", idf)):
        start = time.monotonic()
        results = {q["id"]: grep_rank(corpus, query_terms(q["question"]), weights)
                   for q in questions}
        ms = (time.monotonic() - start) / len(questions) * 1000
        rows.append((label, score(questions, results), ms))

    avg_terms = sum(len(query_terms(q["question"])) for q in questions) / len(questions)
    print("\n\n=== Private Corpus Grep Lower Bound (no models) ===\n")
    print(f"  {n_files} files loaded in {load_secs:.1f}s, "
          f"{avg_terms:.1f} search terms per question on average\n")
    print(f"{'Arm':<16} {'P@5':>6} {'R@10':>6} {'MRR':>6} {'files':>6} {'ms/q':>7}")
    print("-" * 52)
    for label, s, ms in rows:
        print(f"  {label:<14} {s['p5']:>6.2f} {s['r10']:>6.2f} {s['mrr']:>6.2f} "
              f"{s['files']:>6.1f} {ms:>7.0f}")
    print("-" * 52)
    print(f"  n={len(questions)} questions")
    print("=" * 52)
