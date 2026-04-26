"""
Chunking strategy comparison — NOT part of the regular test suite.

Run explicitly with:
    python3.13 -m pytest test_suite/test_05_chunking_strategies.py -m chunking_eval -v -s

Indexes the sample app once per strategy into separate directories, runs all
10 ground-truth queries against each, and prints a side-by-side comparison table.

Strategies compared
-------------------
  ast_1200   AST/method chunking,  max 1200 chars  (current default)
  ast_800    AST/method chunking,  max  800 chars
  ast_600    AST/method chunking,  max  600 chars
  text_1200  Character chunking,   max 1200 chars
  text_800   Character chunking,   max  800 chars
  text_400   Character chunking,   max  400 chars

No pass/fail thresholds — the report is the output.
"""

import pytest
from dataclasses import dataclass

# Re-use the ground truth and metric helpers from test_04
from test_04_semantic_eval import GROUND_TRUTH, precision_at_k, recall_at_k, mrr


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    name: str
    use_ast: bool
    chunk_size: int
    overlap: int = 200


STRATEGIES = [
    Strategy("ast_1200",  use_ast=True,  chunk_size=1200),
    Strategy("ast_800",   use_ast=True,  chunk_size=800),
    Strategy("ast_600",   use_ast=True,  chunk_size=600),
    Strategy("text_1200", use_ast=False, chunk_size=1200),
    Strategy("text_800",  use_ast=False, chunk_size=800),
    Strategy("text_400",  use_ast=False, chunk_size=400),
]


# ---------------------------------------------------------------------------
# Session fixtures — one index per strategy
# ---------------------------------------------------------------------------

def _build_index(sample_app_path: str, index_dir: str, strategy: Strategy):
    from indexing.incremental_indexer import index_repo_incremental
    logs = []
    index_repo_incremental(
        repo_root=sample_app_path,
        index_dir=index_dir,
        force_full_reindex=True,
        num_workers=4,
        use_ast_chunking=strategy.use_ast,
        chars_per_chunk=strategy.chunk_size,
        chunk_overlap=strategy.overlap,
        verbose=False,
        log=logs.append,
    )
    return logs


@pytest.fixture(scope="session")
def strategy_indexes(tmp_path_factory, sample_app_path):
    """Build one index per strategy. Returns dict of strategy_name -> index_dir."""
    indexes = {}
    for s in STRATEGIES:
        index_dir = str(tmp_path_factory.mktemp(f"chroma_{s.name}"))
        print(f"\n  Building index: {s.name} (ast={s.use_ast}, chunk_size={s.chunk_size})...")
        _build_index(sample_app_path, index_dir, s)
        indexes[s.name] = index_dir
    return indexes


# ---------------------------------------------------------------------------
# Query runner
# ---------------------------------------------------------------------------

def _query(index_dir: str, question: str, top_k: int = 10) -> list[str]:
    from querying.query_engine import run_query
    result = run_query(
        bug_text=question,
        index_dir=index_dir,
        top_k=top_k,
        log=lambda _: None,
    )
    return [m.get("source", "") for m in result.get("metas", [])]


def _score_strategy(index_dir: str) -> dict:
    """Run all ground-truth cases and return aggregate metrics."""
    p5s, r10s, mrrs = [], [], []
    per_case = {}

    for case in GROUND_TRUTH:
        sources = _query(index_dir, case.question)
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.chunking_eval
def test_chunking_strategy_comparison(strategy_indexes, sample_app_path):
    """
    Run all strategies and print a comparison table.
    Always passes — output is the result.
    """
    print("\n\nScoring all strategies...")
    results = {}
    for s in STRATEGIES:
        print(f"  Scoring {s.name}...")
        results[s.name] = _score_strategy(strategy_indexes[s.name])

    # --- Summary table ---
    sep = "-" * 52
    header = f"\n{'Strategy':<14} {'p@5':>6} {'r@10':>6} {'MRR':>6}"
    rows = []
    for s in STRATEGIES:
        r = results[s.name]
        rows.append(
            f"  {s.name:<12} {r['mean_p5']:>6.2f} {r['mean_r10']:>6.2f} {r['mean_mrr']:>6.2f}"
        )

    print("\n\n=== Chunking Strategy Comparison ===")
    print(header)
    print(sep)
    print("\n".join(rows))
    print(sep)

    # --- Per-query breakdown ---
    print(f"\n{'Query':<32}", end="")
    for s in STRATEGIES:
        print(f" {s.name:>10}", end="")
    print()
    print("-" * (32 + len(STRATEGIES) * 11))

    for case in GROUND_TRUTH:
        print(f"  {case.name:<30}", end="")
        for s in STRATEGIES:
            r10 = results[s.name]["per_case"][case.name]["r10"]
            print(f" {r10:>10.2f}", end="")
        print()

    print("\n  (values shown are r@10)\n")
    print("=" * 40)


@pytest.mark.chunking_eval
def test_chunking_strategy_best_mrr(strategy_indexes):
    """Reports which strategy has the best MRR. Always passes."""
    results = {s.name: _score_strategy(strategy_indexes[s.name]) for s in STRATEGIES}
    best = max(results, key=lambda n: results[n]["mean_mrr"])
    print(f"\n  Best MRR: {best} ({results[best]['mean_mrr']:.2f})")


@pytest.mark.chunking_eval
def test_chunking_strategy_best_recall(strategy_indexes):
    """Reports which strategy has the best r@10. Always passes."""
    results = {s.name: _score_strategy(strategy_indexes[s.name]) for s in STRATEGIES}
    best = max(results, key=lambda n: results[n]["mean_r10"])
    print(f"\n  Best r@10: {best} ({results[best]['mean_r10']:.2f})")
