"""
Second retrieval round A/B on the private benchmark corpus.

Tests the scripted version of what search agents do between searches: read
the first results, learn the code's own names, search again with them. All
arms share one index and the same questions and return the same number of
files, so differences are what the second round found, not a bigger net.

    one round   embedding search only (the baseline)
    + prf       identifiers picked from the top chunks by rarity, no model
    + llm       identifiers picked by the chat model (config.CHAT_MODEL)

Each arm is compared with the baseline per question (better / worse / same)
with a two-sided sign-flip permutation test, as in test_04c.

Same privacy rules and environment variables as test_04b; unset means skip.
The llm arm needs CHAT_MODEL pulled in Ollama.

    LOCALSCOPE_CORPUS=/path/to/corpus \\
    LOCALSCOPE_QUESTION_BANK=/path/to/question-bank \\
    python3 -m pytest test_suite/test_04e_private_corpus_second_round.py -v -s
"""

import time

import pytest

import config
import querying.second_round as second_round_module
from test_04b_private_corpus_eval import (
    _bank_dir,
    _corpus_root,
    build_index,
    load_retrieval_questions,
    per_question,
    retrieve_sources,
    score,
)
from test_04c_private_corpus_rerank import METRICS, paired_permutation_p

ARMS = [("one round (baseline)", "off"), ("+ second round, prf", "prf"),
        ("+ second round, llm", "llm")]


def _run_arm(index_dir, questions, mode, monkeypatch):
    """Retrieve for every question, recording how many terms round 2 searched."""
    term_counts = []
    real = second_round_module.second_round

    def recording(*args, **kwargs):
        result, terms = real(*args, **kwargs)
        term_counts.append(len(terms))
        return result, terms

    monkeypatch.setattr(second_round_module, "second_round", recording)
    monkeypatch.setattr(config, "SECOND_ROUND", mode)
    start = time.monotonic()
    results = retrieve_sources(index_dir, questions)
    ms = (time.monotonic() - start) / len(questions) * 1000
    monkeypatch.setattr(second_round_module, "second_round", real)
    return results, ms, term_counts


def test_second_round_ab(tmp_path_factory, monkeypatch):
    corpus, bank = _corpus_root(), _bank_dir()
    if not corpus or not bank:
        pytest.skip("LOCALSCOPE_CORPUS and LOCALSCOPE_QUESTION_BANK must both be set")

    questions = load_retrieval_questions(bank)
    monkeypatch.setattr(config, "RERANK_ENABLED", False)
    index_dir = str(tmp_path_factory.mktemp("second_round"))
    build_index(corpus, index_dir)

    arms = []
    for label, mode in ARMS:
        results, ms, term_counts = _run_arm(index_dir, questions, mode, monkeypatch)
        arms.append((label, mode, score(questions, results), ms,
                     per_question(questions, results), term_counts))

    print("\n\n=== Private Corpus Second Retrieval Round A/B ===\n")
    print(f"  n={len(questions)} questions, same index, top 10 files per arm, "
          f"seed chunks {config.SECOND_ROUND_SEED_CHUNKS}, "
          f"up to {config.SECOND_ROUND_TERMS} terms, chat model {config.CHAT_MODEL}\n")
    print(f"{'Arm':<26} {'P@5':>6} {'R@10':>6} {'MRR':>6} {'files':>6} {'ms/q':>7} "
          f"{'terms/q':>8} {'no terms':>9}")
    print("-" * 82)
    for label, mode, s, ms, _, counts in arms:
        terms = f"{sum(counts) / len(counts):.1f}" if counts else "-"
        empty = f"{sum(1 for c in counts if c == 0)}" if counts else "-"
        print(f"  {label:<24} {s['p5']:>6.2f} {s['r10']:>6.2f} {s['mrr']:>6.2f} "
              f"{s['files']:>6.1f} {ms:>7.0f} {terms:>8} {empty:>9}")
    print("-" * 82)

    base = arms[0][4]
    for label, _, _, _, rows, _ in arms[1:]:
        print(f"\n  {label} vs baseline, per question:")
        for metric in METRICS:
            deltas = [rows[q["id"]][metric] - base[q["id"]][metric] for q in questions]
            wins = sum(d > 0 for d in deltas)
            losses = sum(d < 0 for d in deltas)
            p = paired_permutation_p(deltas)
            print(f"    {metric:<5} mean {sum(deltas) / len(deltas):+.3f}   better {wins:>2}  "
                  f"worse {losses:>2}  same {len(deltas) - wins - losses:>2}   p={p:.3f}")
    print("=" * 82)

    for label, _, s, _, _, _ in arms:
        assert s["files"] > 0, f"{label}: retrieved nothing"
