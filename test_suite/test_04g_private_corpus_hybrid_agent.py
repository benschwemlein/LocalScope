"""
Hybrid local agents on the private benchmark corpus: search loop + semantic index.

test_04f showed a local 27B with only grep, find and read tools matches a
Claude agent at finding files, but needs about 22 tool calls and 3 minutes per
question, while the one-pass semantic index answers in 26 ms at lower
accuracy. This combines them. The agent gets the same tools plus:

    seed              the index's top 10 files for the question, with a
                      one-line preview each, in its first message
    semantic_search   a tool to query the index with its own wording

The question is whether starting from the index cuts the steps (and time)
without losing the accuracy, and whether it lifts a 7B, which in 04f did worse
than the index on its own.

Same scoring, privacy rules and environment variables as test_04f:

    LOCALSCOPE_CORPUS=/path/to/corpus \\
    LOCALSCOPE_QUESTION_BANK=/path/to/question-bank \\
    python3 -m pytest test_suite/test_04g_private_corpus_hybrid_agent.py -v -s

Optional: LCQ_AGENT_MODELS, LCQ_AGENT_QUESTIONS, LCQ_AGENT_OUT as in 04f.
"""

import json
import os

import pytest

from querying.agent_search import Repo, run_agent
from test_04b_private_corpus_eval import (
    TOP_K,
    _bank_dir,
    _corpus_root,
    build_index,
    load_retrieval_questions,
    score,
)

DEFAULT_MODELS = ["qwen3.6:27b", "qwen2.5:7b"]


def _env_list(name: str) -> list[str]:
    return [v.strip() for v in os.environ.get(name, "").split(",") if v.strip()]


def _preview(chunk: str) -> str:
    """First line of real content, skipping the chunker's file-name header."""
    for line in chunk.splitlines():
        line = line.strip()
        if line and not line.startswith("'''"):
            return line[:120]
    return ""


def make_semantic(index_dir: str):
    """semantic(query) -> [(path, preview)] from the index, as run_query retrieves."""
    import chromadb
    from chromadb.config import Settings

    from querying.query_engine import _embed_text, retrieve_chunks

    client = chromadb.PersistentClient(path=index_dir, settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(client.list_collections()[0].name)

    def semantic(query: str) -> list[tuple[str, str]]:
        embedding = _embed_text(query, lambda _: None)
        if embedding is None:
            return []
        docs, metas, _ = retrieve_chunks(collection, query, embedding, TOP_K, log=lambda _: None)
        return [(m.get("source", ""), _preview(d)) for d, m in zip(docs, metas)]

    return semantic


def _save(raw: dict) -> None:
    out = os.environ.get("LCQ_AGENT_OUT")
    if out:
        with open(os.path.expanduser(out), "w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=1)


def test_hybrid_agent_search(tmp_path_factory):
    corpus, bank = _corpus_root(), _bank_dir()
    if not corpus or not bank:
        pytest.skip("LOCALSCOPE_CORPUS and LOCALSCOPE_QUESTION_BANK must both be set")

    import config
    config.RERANK_ENABLED, config.SECOND_ROUND = False, "off"

    questions = load_retrieval_questions(bank)
    subset = set(_env_list("LCQ_AGENT_QUESTIONS"))
    if subset:
        questions = [q for q in questions if q["id"] in subset]
    models = _env_list("LCQ_AGENT_MODELS") or DEFAULT_MODELS

    index_dir = str(tmp_path_factory.mktemp("hybrid_agent"))
    build_index(corpus, index_dir)
    semantic = make_semantic(index_dir)
    repo = Repo(corpus)

    rows, raw = [], {}
    for model in models:
        results, runs = {}, []
        for q in questions:
            run = run_agent(q["question"], repo, model, top_k=TOP_K,
                            semantic=semantic, seed=True, log=lambda _: None)
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

    print("\n\n=== Private Corpus Hybrid Agent Search (seed + semantic_search) ===\n")
    print(f"  n={len(questions)} questions, {len(repo.files)} files searchable, "
          f"up to {TOP_K} files submitted\n")
    print(f"{'Model':<16} {'P@5':>6} {'R@10':>6} {'MRR':>6} {'files':>6} "
          f"{'calls/q':>8} {'s/q':>6} {'no submit':>10} {'errors':>7}")
    print("-" * 80)
    for model, s, calls, secs, no_submit, errors in rows:
        print(f"  {model:<14} {s['p5']:>6.2f} {s['r10']:>6.2f} {s['mrr']:>6.2f} "
              f"{s['files']:>6.1f} {calls:>8.1f} {secs:>6.0f} {no_submit:>10} {errors:>7}")
    print("-" * 80)
    print("  reference: semantic index 0.28 / 0.75 / 0.72; grep-only agents "
          "27B 0.43 / 0.92 / 0.95 (21.5 calls, 178 s), 7B 0.24 / 0.56 / 0.56")
    print("=" * 80)
