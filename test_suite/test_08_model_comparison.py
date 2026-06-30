"""
Chat model comparison — NOT part of the regular test suite.

Run explicitly with:
    python3.13 -m pytest test_suite/test_08_model_comparison.py -m chunking_eval -v -s

Tests answer quality (faithfulness + reference overlap) across chat models.

Before/after Zack Lowery's June 2026 recommendations:
  Before: qwen2.5:7b  (prior default — Leading Edje benchmark winner)
  After:  gemma4:12b  (Zack's pick for general dev, single consumer GPU)

Uses the same faithfulness and reference overlap metrics as test_07.
No pass/fail thresholds — the report is the output.
"""

import pytest

from test_04_semantic_eval import GROUND_TRUTH
from test_07_answer_quality import (
    REFERENCE_BY_NAME,
    faithfulness_score,
    reference_overlap_score,
)

CHAT_MODELS = [
    "qwen2.5:7b",    # pre-Zack baseline (prior default)
    "gemma4:12b",    # Zack Lowery's pick — general dev, single consumer GPU
]


def _run_query(question: str, index_dir: str, chat_model: str) -> dict:
    import config
    from querying.query_engine import run_query

    original = config.CHAT_MODEL
    try:
        config.CHAT_MODEL = chat_model
        result = run_query(
            bug_text=question,
            index_dir=index_dir,
            top_k=10,
            log=lambda _: None,
        )
    finally:
        config.CHAT_MODEL = original

    return result


@pytest.fixture(scope="session")
def model_results(indexed_app) -> dict:
    """Run all queries with all models. Returns {model: {case_name: result}}."""
    results = {}
    for model in CHAT_MODELS:
        print(f"\n  Running queries with {model}...")
        results[model] = {}
        for case in GROUND_TRUTH:
            result = _run_query(case.question, indexed_app["index_dir"], model)
            results[model][case.name] = {
                "answer": result.get("answer", ""),
                "docs":   result.get("docs", []),
                "metas":  result.get("metas", []),
            }
    return results


@pytest.mark.chunking_eval
def test_model_comparison(model_results):
    """Compare chat models on faithfulness and reference overlap. Always passes."""
    # Compute scores
    scores = {}
    for model in CHAT_MODELS:
        scores[model] = {}
        for case in GROUND_TRUTH:
            r = model_results[model][case.name]
            ref = REFERENCE_BY_NAME.get(case.name)
            faith = faithfulness_score(r["answer"], r["metas"])
            ovlp  = reference_overlap_score(r["answer"], ref.reference) if ref else 0.0
            scores[model][case.name] = {"faith": faith, "ovlp": ovlp}

    # Aggregate
    def mean(vals): return sum(vals) / len(vals) if vals else 0.0

    print("\n\n=== Chat Model Comparison ===\n")
    print(f"{'Model':<24} {'faith':>7} {'ref_ovlp':>9}")
    print("-" * 44)
    for model in CHAT_MODELS:
        faiths = [scores[model][c.name]["faith"] for c in GROUND_TRUTH]
        ovlps  = [scores[model][c.name]["ovlp"]  for c in GROUND_TRUTH]
        print(f"  {model:<22} {mean(faiths):>7.2f} {mean(ovlps):>9.2f}")
    print("-" * 44)

    # Per-query breakdown — faithfulness
    print(f"\n{'Query':<32}", end="")
    for model in CHAT_MODELS:
        print(f" {model:>22}", end="")
    print(f"\n  (faithfulness scores)")
    print("-" * (32 + len(CHAT_MODELS) * 23))
    for case in GROUND_TRUTH:
        print(f"  {case.name:<30}", end="")
        for model in CHAT_MODELS:
            print(f" {scores[model][case.name]['faith']:>22.2f}", end="")
        print()

    # Per-query breakdown — reference overlap
    print(f"\n{'Query':<32}", end="")
    for model in CHAT_MODELS:
        print(f" {model:>22}", end="")
    print(f"\n  (reference overlap scores)")
    print("-" * (32 + len(CHAT_MODELS) * 23))
    for case in GROUND_TRUTH:
        print(f"  {case.name:<30}", end="")
        for model in CHAT_MODELS:
            print(f" {scores[model][case.name]['ovlp']:>22.2f}", end="")
        print()

    print("\n" + "=" * 40)
