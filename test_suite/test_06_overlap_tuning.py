"""
Text chunking overlap tuning — NOT part of the regular test suite.

Run explicitly with:
    python3.13 -m pytest test_suite/test_06_overlap_tuning.py -m chunking_eval -v -s

Fixes chunk size at 800 (empirically best from test_05) and varies overlap
to find the optimal setting for text-based chunking.

Overlaps compared
-----------------
  text_800_ov0    overlap=0    (no overlap)
  text_800_ov100  overlap=100
  text_800_ov200  overlap=200  (current default)
  text_800_ov400  overlap=400
  text_800_ov600  overlap=600

No pass/fail thresholds — the report is the output.
"""

import pytest
from dataclasses import dataclass

from test_04_semantic_eval import GROUND_TRUTH, precision_at_k, recall_at_k, mrr


@dataclass
class OverlapVariant:
    name: str
    overlap: int
    chunk_size: int = 800


VARIANTS = [
    OverlapVariant("text_800_ov0",   overlap=0),
    OverlapVariant("text_800_ov100", overlap=100),
    OverlapVariant("text_800_ov200", overlap=200),
    OverlapVariant("text_800_ov400", overlap=400),
    OverlapVariant("text_800_ov600", overlap=600),
]


def _build_index(sample_app_path: str, index_dir: str, variant: OverlapVariant):
    from indexing.incremental_indexer import index_repo_incremental
    logs = []
    index_repo_incremental(
        repo_root=sample_app_path,
        index_dir=index_dir,
        force_full_reindex=True,
        num_workers=4,
        use_ast_chunking=False,
        chars_per_chunk=variant.chunk_size,
        chunk_overlap=variant.overlap,
        verbose=False,
        log=logs.append,
    )
    return logs


@pytest.fixture(scope="session")
def overlap_indexes(tmp_path_factory, sample_app_path):
    """Build one index per overlap variant. Returns dict of name -> index_dir."""
    indexes = {}
    for v in VARIANTS:
        index_dir = str(tmp_path_factory.mktemp(f"chroma_{v.name}"))
        print(f"\n  Building index: {v.name} (overlap={v.overlap})...")
        _build_index(sample_app_path, index_dir, v)
        indexes[v.name] = index_dir
    return indexes


def _query(index_dir: str, question: str, top_k: int = 10) -> list[str]:
    from querying.query_engine import run_query
    result = run_query(
        bug_text=question,
        index_dir=index_dir,
        top_k=top_k,
        log=lambda _: None,
    )
    return [m.get("source", "") for m in result.get("metas", [])]


def _score_variant(index_dir: str) -> dict:
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


@pytest.mark.chunking_eval
def test_overlap_comparison(overlap_indexes, sample_app_path):
    """
    Run all overlap variants and print a comparison table.
    Always passes — output is the result.
    """
    print("\n\nScoring all overlap variants...")
    results = {}
    for v in VARIANTS:
        print(f"  Scoring {v.name}...")
        results[v.name] = _score_variant(overlap_indexes[v.name])

    sep = "-" * 52
    header = f"\n{'Variant':<20} {'p@5':>6} {'r@10':>6} {'MRR':>6}"
    rows = []
    for v in VARIANTS:
        r = results[v.name]
        rows.append(
            f"  {v.name:<18} {r['mean_p5']:>6.2f} {r['mean_r10']:>6.2f} {r['mean_mrr']:>6.2f}"
        )

    print("\n\n=== Text Chunking Overlap Comparison (chunk_size=800) ===")
    print(header)
    print(sep)
    print("\n".join(rows))
    print(sep)

    print(f"\n{'Query':<32}", end="")
    for v in VARIANTS:
        print(f" {v.name:>18}", end="")
    print()
    print("-" * (32 + len(VARIANTS) * 19))

    for case in GROUND_TRUTH:
        print(f"  {case.name:<30}", end="")
        for v in VARIANTS:
            r10 = results[v.name]["per_case"][case.name]["r10"]
            print(f" {r10:>18.2f}", end="")
        print()

    print("\n  (values shown are r@10)\n")
    print("=" * 40)


@pytest.mark.chunking_eval
def test_overlap_best_mrr(overlap_indexes):
    """Reports which overlap has the best MRR. Always passes."""
    results = {v.name: _score_variant(overlap_indexes[v.name]) for v in VARIANTS}
    best = max(results, key=lambda n: results[n]["mean_mrr"])
    print(f"\n  Best MRR: {best} ({results[best]['mean_mrr']:.2f})")


@pytest.mark.chunking_eval
def test_overlap_best_recall(overlap_indexes):
    """Reports which overlap has the best r@10. Always passes."""
    results = {v.name: _score_variant(overlap_indexes[v.name]) for v in VARIANTS}
    best = max(results, key=lambda n: results[n]["mean_r10"])
    print(f"\n  Best r@10: {best} ({results[best]['mean_r10']:.2f})")
