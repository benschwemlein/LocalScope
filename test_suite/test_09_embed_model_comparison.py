"""
Embedding model comparison — NOT part of the regular test suite.

Run explicitly with:
    python3.13 -m pytest test_suite/test_09_embed_model_comparison.py -m chunking_eval -v -s

Builds one index per embedding model and scores retrieval quality (p@5, r@10, MRR)
against the 10 ground-truth queries from test_04.

Models compared
---------------
  nomic-embed-text   current default (274 MB)
  mxbai-embed-large  larger model, stronger on code retrieval (669 MB)

No pass/fail thresholds — the report is the output.
"""

import pytest
from dataclasses import dataclass

from test_04_semantic_eval import GROUND_TRUTH, precision_at_k, recall_at_k, mrr


EMBED_MODELS = [
    "nomic-embed-text",          # 274 MB — current default, general purpose
    "mxbai-embed-large",         # 669 MB — strong MTEB scores, general purpose
    "snowflake-arctic-embed2",   # 568 MB — trained specifically for retrieval tasks
    "bge-m3",                    # 1.2 GB — top MTEB, strong on technical content
]


def _build_index(sample_app_path: str, index_dir: str, embed_model: str):
    import config
    from indexing.incremental_indexer import index_repo_incremental

    original = config.EMBED_MODEL
    try:
        config.EMBED_MODEL = embed_model
        logs = []
        index_repo_incremental(
            repo_root=sample_app_path,
            index_dir=index_dir,
            force_full_reindex=True,
            num_workers=4,
            verbose=False,
            log=logs.append,
        )
    finally:
        config.EMBED_MODEL = original
    return logs


def _query(index_dir: str, question: str, embed_model: str, top_k: int = 10) -> list[str]:
    import config
    from querying.query_engine import run_query

    original = config.EMBED_MODEL
    try:
        config.EMBED_MODEL = embed_model
        result = run_query(
            bug_text=question,
            index_dir=index_dir,
            top_k=top_k,
            log=lambda _: None,
        )
    finally:
        config.EMBED_MODEL = original

    return [m.get("source", "") for m in result.get("metas", [])]


@pytest.fixture(scope="session")
def embed_indexes(tmp_path_factory, sample_app_path) -> dict:
    """Build one index per embedding model. Returns {model: index_dir}."""
    indexes = {}
    for model in EMBED_MODELS:
        index_dir = str(tmp_path_factory.mktemp(f"chroma_{model.replace(':', '_').replace('-', '_')}"))
        print(f"\n  Building index with {model}...")
        _build_index(sample_app_path, index_dir, model)
        indexes[model] = index_dir
    return indexes


def _score_model(index_dir: str, embed_model: str) -> dict:
    p5s, r10s, mrrs = [], [], []
    per_case = {}

    for case in GROUND_TRUTH:
        sources = _query(index_dir, case.question, embed_model)
        expected = set(case.expected_files)
        p5  = precision_at_k(sources, expected, 5)
        r10 = recall_at_k(sources, expected, 10)
        m   = mrr(sources, expected)
        p5s.append(p5); r10s.append(r10); mrrs.append(m)
        per_case[case.name] = {"p5": p5, "r10": r10, "mrr": m}

    return {
        "mean_p5":  sum(p5s)  / len(p5s),
        "mean_r10": sum(r10s) / len(r10s),
        "mean_mrr": sum(mrrs) / len(mrrs),
        "per_case": per_case,
    }


@pytest.mark.chunking_eval
def test_embed_model_comparison(embed_indexes, sample_app_path):
    """Compare embedding models on retrieval quality. Always passes."""
    print("\n\nScoring all embedding models...")
    results = {}
    for model in EMBED_MODELS:
        print(f"  Scoring {model}...")
        results[model] = _score_model(embed_indexes[model], model)

    sep = "-" * 52
    print("\n\n=== Embedding Model Comparison ===\n")
    print(f"{'Model':<24} {'p@5':>6} {'r@10':>6} {'MRR':>6}")
    print(sep)
    for model in EMBED_MODELS:
        r = results[model]
        print(f"  {model:<22} {r['mean_p5']:>6.2f} {r['mean_r10']:>6.2f} {r['mean_mrr']:>6.2f}")
    print(sep)

    print(f"\n{'Query':<32}", end="")
    for model in EMBED_MODELS:
        print(f" {model:>24}", end="")
    print(f"\n  (values shown are r@10)")
    print("-" * (32 + len(EMBED_MODELS) * 25))
    for case in GROUND_TRUTH:
        print(f"  {case.name:<30}", end="")
        for model in EMBED_MODELS:
            r10 = results[model]["per_case"][case.name]["r10"]
            print(f" {r10:>24.2f}", end="")
        print()

    print("\n" + "=" * 40)


@pytest.mark.chunking_eval
def test_embed_model_best_mrr(embed_indexes):
    """Reports which model has the best MRR. Always passes."""
    results = {m: _score_model(embed_indexes[m], m) for m in EMBED_MODELS}
    best = max(results, key=lambda m: results[m]["mean_mrr"])
    print(f"\n  Best MRR: {best} ({results[best]['mean_mrr']:.2f})")


@pytest.mark.chunking_eval
def test_embed_model_best_recall(embed_indexes):
    """Reports which model has the best r@10. Always passes."""
    results = {m: _score_model(embed_indexes[m], m) for m in EMBED_MODELS}
    best = max(results, key=lambda m: results[m]["mean_r10"])
    print(f"\n  Best r@10: {best} ({results[best]['mean_r10']:.2f})")
