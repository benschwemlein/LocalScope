"""
HyDE A/B on the private benchmark corpus.

Three arms on one index, same questions, same 10-file budget:

    question       embed the question as asked (the baseline)
    snippet        embed only the model's hypothetical code snippet
    both           embed the question followed by the snippet

Each question's snippet is generated once and reused by both HyDE arms, so
they differ only in what is embedded. Per-question paired comparison
against the baseline with a two-sided sign-flip permutation test.

Same privacy rules and environment variables as test_04b; unset means skip.

    LOCALSCOPE_CORPUS=/path/to/corpus \\
    LOCALSCOPE_QUESTION_BANK=/path/to/question-bank \\
    LCQ_HYDE_MODEL=qwen3.6:35b-a3b \\
    python3 -m pytest test_suite/test_04i_private_corpus_hyde.py -v -s

Optional: LCQ_HYDE_HINT describes the codebase to the model;
LCQ_HYDE_OUT writes each question's snippet and retrieved files as JSON.

test_hyde_public_repo runs the same comparison on the large public
repository used by test_04h, with its question bank and saved index:

    LOCALSCOPE_PR_CORPUS=/path/to/checkout \\
    LOCALSCOPE_PR_BANK=/path/to/bank.json \\
    LCQ_SCALE_INDEX_DIR=/path/to/saved/index \\
    LCQ_HYDE_MODEL=qwen3.6:35b-a3b \\
    python3 -m pytest test_suite/test_04i_private_corpus_hyde.py -k public -v -s
"""

import json
import os
import time

import pytest

import config
from test_04b_private_corpus_eval import (
    TOP_K,
    _bank_dir,
    _corpus_root,
    build_index,
    load_retrieval_questions,
    per_question,
    score,
)
from test_04c_private_corpus_rerank import METRICS, paired_permutation_p

ARMS = [("question (baseline)", "off"), ("snippet only", "doc"), ("question + snippet", "both")]


def _retrieve(collection, question: str, text: str) -> list[str]:
    from querying.query_engine import _embed_text, retrieve_chunks

    embedding = _embed_text(text, lambda _: None)
    assert embedding is not None, "embedding failed"
    _, metas, _ = retrieve_chunks(collection, question, embedding, TOP_K, log=lambda _: None)
    return [m.get("source", "") for m in metas]


def test_hyde_ab(tmp_path_factory, monkeypatch):
    corpus, bank = _corpus_root(), _bank_dir()
    if not corpus or not bank:
        pytest.skip("LOCALSCOPE_CORPUS and LOCALSCOPE_QUESTION_BANK must both be set")
    index_dir = str(tmp_path_factory.mktemp("hyde"))
    build_index(corpus, index_dir)
    _run_hyde_ab(index_dir, load_retrieval_questions(bank), "Private Corpus", monkeypatch)


def test_hyde_public_repo(monkeypatch):
    corpus = os.path.expanduser(os.environ.get("LOCALSCOPE_PR_CORPUS", ""))
    bank = os.path.expanduser(os.environ.get("LOCALSCOPE_PR_BANK", ""))
    index_dir = os.path.expanduser(os.environ.get("LCQ_SCALE_INDEX_DIR", ""))
    if not (os.environ.get("LOCALSCOPE_PR_CORPUS") and os.environ.get("LOCALSCOPE_PR_BANK")
            and os.environ.get("LCQ_SCALE_INDEX_DIR")):
        pytest.skip("LOCALSCOPE_PR_CORPUS, LOCALSCOPE_PR_BANK and LCQ_SCALE_INDEX_DIR must be set")
    if not (os.path.isdir(index_dir) and os.listdir(index_dir)):
        build_index(corpus, index_dir)
    with open(bank, encoding="utf-8") as fh:
        questions = json.load(fh)
    _run_hyde_ab(index_dir, questions, "Public Repository", monkeypatch)


def _run_hyde_ab(index_dir, questions, title, monkeypatch):
    import chromadb
    from chromadb.config import Settings

    from querying.hyde import hypothetical_snippet, text_to_embed

    monkeypatch.setattr(config, "RERANK_ENABLED", False)
    monkeypatch.setattr(config, "SECOND_ROUND", "off")
    client = chromadb.PersistentClient(path=index_dir, settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(client.list_collections()[0].name)

    start = time.monotonic()
    snippets = {}
    for q in questions:
        snippets[q["id"]] = hypothetical_snippet(q["question"], log=lambda _: None)
        print(f"    snippet {q['id']}: {len(snippets[q['id']].splitlines())} lines", flush=True)
    gen_secs = (time.monotonic() - start) / len(questions)
    failed = sum(1 for s in snippets.values() if not s)

    arms, raw = [], {"snippets": snippets}
    for label, mode in ARMS:
        results = {
            q["id"]: _retrieve(collection, q["question"],
                               text_to_embed(q["question"], mode, snippets[q["id"]]))
            for q in questions
        }
        raw[mode] = results
        arms.append((label, score(questions, results), per_question(questions, results)))

    out = os.environ.get("LCQ_HYDE_OUT")
    if out:
        with open(os.path.expanduser(out), "w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=1)

    print(f"\n\n=== {title} HyDE A/B ===\n")
    print(f"  n={len(questions)} questions, snippet model {config.HYDE_MODEL}, "
          f"{gen_secs:.1f} s per snippet, {failed} failed"
          f"{', hint: ' + config.HYDE_HINT if config.HYDE_HINT else ''}\n")
    print(f"{'Arm':<24} {'P@5':>6} {'R@10':>6} {'MRR':>6} {'files':>6}")
    print("-" * 52)
    for label, s, _ in arms:
        print(f"  {label:<22} {s['p5']:>6.2f} {s['r10']:>6.2f} {s['mrr']:>6.2f} {s['files']:>6.1f}")
    print("-" * 52)

    base = arms[0][2]
    for label, _, rows in arms[1:]:
        print(f"\n  {label} vs baseline, per question:")
        for metric in METRICS:
            deltas = [rows[q["id"]][metric] - base[q["id"]][metric] for q in questions]
            wins, losses = sum(d > 0 for d in deltas), sum(d < 0 for d in deltas)
            print(f"    {metric:<5} mean {sum(deltas) / len(deltas):+.3f}   better {wins:>2}  "
                  f"worse {losses:>2}  same {len(deltas) - wins - losses:>2}   "
                  f"p={paired_permutation_p(deltas):.3f}")

    by_cat: dict[str, list] = {}
    for q in questions:
        by_cat.setdefault(q.get("category", "?"), []).append(q)
    print("\n  by category, R@10 / MRR (" + " | ".join(label for label, _, _ in arms) + "):")
    for cat, qs in sorted(by_cat.items()):
        cells = "  |  ".join(
            f"{sum(rows[q['id']]['r10'] for q in qs) / len(qs):.2f} / "
            f"{sum(rows[q['id']]['mrr'] for q in qs) / len(qs):.2f}"
            for _, _, rows in arms)
        print(f"    {cat:<7} n={len(qs):>2}   {cells}")
    print("=" * 52)
