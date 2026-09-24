"""
Embedding model comparison on the private benchmark corpus.

The private-corpus counterpart of test_09_embed_model_comparison.py. test_09 picked
mxbai-embed-large in June 2026 from 10 questions on the public sample app,
and all four models it tried were general-purpose text embedders. This file
re-runs the choice on the harder private corpus with code-aware candidates
added.

Everything except the embedding model is held constant: same corpus, same
questions, same chunking, same 10-file budget, same retrieval stage as
test_04b. Each model builds its own index from scratch in a temp directory.

Some models are trained to expect an instruction on the query side (never on
the documents). Those get their documented prefix; mxbai-embed-large is run
both ways so the engine's current no-prefix behaviour stays the baseline row.

Same privacy rules and environment variables as test_04b; unset means skip:

    LOCALSCOPE_CORPUS=/path/to/corpus \
    LOCALSCOPE_QUESTION_BANK=/path/to/question-bank \
    python3 -m pytest test_suite/test_09b_private_corpus_embed.py -v -s

Models not pulled into Ollama are skipped and reported, not failed.
"""

import time

import pytest
import requests

import config
from test_04b_private_corpus_eval import (
    _bank_dir,
    _corpus_root,
    build_index,
    load_retrieval_questions,
    retrieve_sources,
    score,
)

_CODE_QUERY = "Given a question about a codebase, retrieve the source files that answer it"

# (label, ollama model, query prefix)
CANDIDATES = [
    ("mxbai-embed-large", "mxbai-embed-large", ""),
    ("mxbai-embed-large +prompt", "mxbai-embed-large",
     "Represent this sentence for searching relevant passages: "),
    ("bge-m3", "bge-m3", ""),
    ("qwen3-embedding:0.6b", "qwen3-embedding:0.6b", f"Instruct: {_CODE_QUERY}\nQuery: "),
    ("qwen3-embedding:4b", "qwen3-embedding:4b", f"Instruct: {_CODE_QUERY}\nQuery: "),
    ("embeddinggemma", "embeddinggemma", "task: code retrieval | query: "),
]


def _installed_models() -> set[str]:
    try:
        tags = requests.get(f"{config.OLLAMA_URL.rstrip('/')}/api/tags", timeout=5).json()
    except (requests.RequestException, ValueError):
        return set()
    names = set()
    for m in tags.get("models", []):
        names.add(m["name"])
        if m["name"].endswith(":latest"):
            names.add(m["name"].removesuffix(":latest"))
    return names


def test_corpus_embedding_comparison(tmp_path_factory):
    corpus, bank = _corpus_root(), _bank_dir()
    if not corpus or not bank:
        pytest.skip("LOCALSCOPE_CORPUS and LOCALSCOPE_QUESTION_BANK must both be set")

    questions = load_retrieval_questions(bank)
    installed = _installed_models()
    original_model = config.EMBED_MODEL

    rows, skipped = [], []
    indexes: dict[str, tuple[str, float]] = {}
    try:
        for label, model, prefix in CANDIDATES:
            if model not in installed:
                skipped.append(label)
                continue

            config.EMBED_MODEL = model
            if model not in indexes:
                index_dir = str(tmp_path_factory.mktemp(model.replace(":", "_")))
                start = time.monotonic()
                build_index(corpus, index_dir)
                indexes[model] = (index_dir, time.monotonic() - start)
            index_dir, index_secs = indexes[model]

            start = time.monotonic()
            results = retrieve_sources(index_dir, questions, query_prefix=prefix)
            query_ms = (time.monotonic() - start) / len(questions) * 1000

            rows.append((label, score(questions, results), index_secs, query_ms))
    finally:
        config.EMBED_MODEL = original_model

    print("\n\n=== Private Corpus Embedding Comparison ===\n")
    print(f"{'Model':<28} {'P@5':>6} {'R@10':>6} {'MRR':>6} {'files':>6} "
          f"{'index s':>8} {'query ms':>9}")
    print("-" * 74)
    for label, s, index_secs, query_ms in rows:
        print(f"  {label:<26} {s['p5']:>6.2f} {s['r10']:>6.2f} {s['mrr']:>6.2f} "
              f"{s['files']:>6.1f} {index_secs:>8.0f} {query_ms:>9.0f}")
    print("-" * 74)
    print(f"  n={len(questions)} questions")
    if skipped:
        print(f"  skipped (not pulled in Ollama): {', '.join(skipped)}")
    print("=" * 74)

    assert rows, "no candidate embedding models are installed"
