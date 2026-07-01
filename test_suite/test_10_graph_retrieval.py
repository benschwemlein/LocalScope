"""
Graph hybrid retrieval benchmark gate — validates SC-001 through SC-010.

Runs all 10 ground-truth queries with LCQ_GRAPH_ENABLED=true and asserts
that graph-enhanced retrieval meets the improvement targets over vector-only
baselines captured in test_04.

Success Criteria
----------------
  SC-001  fine_calculation_strategy R@10 >= 0.80 (baseline: 0.20)
  SC-002  loan_eligibility_chain    R@10 >= 0.80
  SC-003  mean R@10                 >= 0.70
  SC-004  mean P@5                  >= 0.50
  SC-005  mean MRR                  >= 0.88
  SC-006  9 test_04 passing cases   no regression (R@10 >= 0.25 each)
  SC-007  full graph build          <= 60s
  SC-008  incremental update        <= 5s per changed file
  SC-009  graph disabled            results identical to vector-only
  SC-010  corrupt graph             graceful fallback, no crash
"""

import os
import time
import shutil

import pytest

from test_04_semantic_eval import (
    GROUND_TRUTH,
    precision_at_k,
    recall_at_k,
    mrr,
)

# Queries that pass the per-query recall floor in test_04 (all except fine_calculation_strategy)
_TEST04_PASSING = {
    c.name for c in GROUND_TRUTH if c.name != "fine_calculation_strategy"
}


# ---------------------------------------------------------------------------
# Session fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def graph_app(indexed_app):
    """Build the graph index on top of the shared ChromaDB session index."""
    from graph.graph_builder import build_incremental

    sample_path = indexed_app["sample_app_path"]
    index_dir = indexed_app["index_dir"]
    graph_path = os.path.join(index_dir, "graph.json")

    t0 = time.time()
    build_incremental(sample_path, graph_path, log_fn=lambda _: None)
    elapsed = time.time() - t0

    return {
        **indexed_app,
        "graph_path": graph_path,
        "graph_build_time": elapsed,
    }


@pytest.fixture(scope="session")
def graph_query_results(graph_app):
    """Run all ground-truth queries with graph enabled. Results cached for session."""
    import config
    from querying.query_engine import run_query

    original = config.GRAPH_ENABLED
    config.GRAPH_ENABLED = True
    config.GRAPH_ALPHA = 0.3
    config.GRAPH_BETA = 0.6

    try:
        results = {}
        for case in GROUND_TRUTH:
            result = run_query(
                bug_text=case.question,
                index_dir=graph_app["index_dir"],
                top_k=10,
                log=lambda _: None,
            )
            results[case.name] = [m.get("source", "") for m in result.get("metas", [])]
        return results
    finally:
        config.GRAPH_ENABLED = original


# ---------------------------------------------------------------------------
# SC-001  fine_calculation_strategy recall
# ---------------------------------------------------------------------------

def test_sc001_fine_calculation_strategy(graph_query_results):
    """SC-001: fine_calculation_strategy R@10 >= 0.80 (baseline was 0.20)."""
    sources = graph_query_results["fine_calculation_strategy"]
    expected = {c.expected_files for c in GROUND_TRUTH if c.name == "fine_calculation_strategy"}
    expected_flat = set(GROUND_TRUTH[2].expected_files)  # index 2 = fine_calculation_strategy
    r10 = recall_at_k(sources, expected_flat, 10)
    assert r10 >= 0.80, (
        f"fine_calculation_strategy R@10={r10:.2f} < 0.80 — "
        f"strategy files not surfaced via graph; top-10={sources}"
    )


# ---------------------------------------------------------------------------
# SC-002  loan_eligibility_chain recall
# ---------------------------------------------------------------------------

def test_sc002_loan_eligibility_chain(graph_query_results):
    """SC-002: loan_eligibility_chain R@10 >= 0.80."""
    case = next(c for c in GROUND_TRUTH if c.name == "loan_eligibility_chain")
    r10 = recall_at_k(graph_query_results[case.name], set(case.expected_files), 10)
    assert r10 >= 0.80, f"loan_eligibility_chain R@10={r10:.2f} < 0.80"


# ---------------------------------------------------------------------------
# SC-003  mean R@10
# ---------------------------------------------------------------------------

def test_sc003_mean_recall_at_10(graph_query_results):
    """SC-003: mean R@10 across all 10 queries >= 0.70."""
    scores = [
        recall_at_k(graph_query_results[c.name], set(c.expected_files), 10)
        for c in GROUND_TRUTH
    ]
    mean = sum(scores) / len(scores)
    assert mean >= 0.70, f"Mean R@10 {mean:.2f} < 0.70"


# ---------------------------------------------------------------------------
# SC-004  mean P@5
# ---------------------------------------------------------------------------

def test_sc004_mean_precision_at_5(graph_query_results):
    """SC-004: mean P@5 across all 10 queries >= 0.50."""
    scores = [
        precision_at_k(graph_query_results[c.name], set(c.expected_files), 5)
        for c in GROUND_TRUTH
    ]
    mean = sum(scores) / len(scores)
    assert mean >= 0.50, f"Mean P@5 {mean:.2f} < 0.50"


# ---------------------------------------------------------------------------
# SC-005  mean MRR
# ---------------------------------------------------------------------------

def test_sc005_mean_mrr(graph_query_results):
    """SC-005: mean MRR across all 10 queries >= 0.88."""
    scores = [
        mrr(graph_query_results[c.name], set(c.expected_files))
        for c in GROUND_TRUTH
    ]
    mean = sum(scores) / len(scores)
    assert mean >= 0.88, f"Mean MRR {mean:.2f} < 0.88"


# ---------------------------------------------------------------------------
# SC-006  regression guard — no degradation on test_04 passing cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case",
    [c for c in GROUND_TRUTH if c.name in _TEST04_PASSING],
    ids=[c.name for c in GROUND_TRUTH if c.name in _TEST04_PASSING],
)
def test_sc006_no_regression(case, graph_query_results):
    """SC-006: graph must not degrade test_04 passing cases below their recall floor."""
    r10 = recall_at_k(graph_query_results[case.name], set(case.expected_files), 10)
    assert r10 >= 0.25, (
        f"{case.name}: graph degraded R@10 to {r10:.2f} < 0.25 regression floor"
    )


# ---------------------------------------------------------------------------
# SC-007  full graph build timing
# ---------------------------------------------------------------------------

def test_sc007_graph_build_timing(graph_app):
    """SC-007: full graph build for library-catalog-app must complete in <= 60s."""
    elapsed = graph_app["graph_build_time"]
    assert elapsed <= 60, f"Graph build took {elapsed:.1f}s > 60s limit"


# ---------------------------------------------------------------------------
# SC-008  incremental update timing
# ---------------------------------------------------------------------------

def test_sc008_incremental_update_timing(graph_app, tmp_path):
    """SC-008: incremental update for one changed file must complete in <= 5s."""
    from graph.graph_builder import build_incremental

    graph_copy = str(tmp_path / "graph.json")
    shutil.copy2(graph_app["graph_path"], graph_copy)
    hashes_src = os.path.splitext(graph_app["graph_path"])[0] + "_hashes.json"
    if os.path.exists(hashes_src):
        shutil.copy2(hashes_src, str(tmp_path / "graph_hashes.json"))

    # Pick one file to "change"
    sample_path = graph_app["sample_app_path"]
    one_file = "src/main/java/com/example/library/pattern/strategy/OverdueFineContext.java"

    t0 = time.time()
    build_incremental(
        sample_path, graph_copy, changed_files=[one_file], log_fn=lambda _: None
    )
    elapsed = time.time() - t0
    assert elapsed <= 5, f"Incremental update took {elapsed:.2f}s > 5s limit"


# ---------------------------------------------------------------------------
# SC-009  GRAPH_ENABLED=false leaves results unchanged (vector-only fallback)
# ---------------------------------------------------------------------------

def test_sc009_graph_disabled_fallback(indexed_app):
    """SC-009: with graph disabled, results match vector-only retrieval."""
    import config
    from querying.query_engine import run_query

    original = config.GRAPH_ENABLED
    config.GRAPH_ENABLED = False
    try:
        result = run_query(
            bug_text=GROUND_TRUTH[2].question,  # fine_calculation_strategy
            index_dir=indexed_app["index_dir"],
            top_k=10,
            log=lambda _: None,
        )
        sources = [m.get("source", "") for m in result.get("metas", [])]
        expected_flat = set(GROUND_TRUTH[2].expected_files)
        r10 = recall_at_k(sources, expected_flat, 10)
        # Vector-only baseline for this query is 0.20 — should still be the same
        assert r10 <= 0.40, (
            f"Expected vector-only R@10 for fine_calculation_strategy, got {r10:.2f}; "
            "graph may have leaked into the disabled path"
        )
    finally:
        config.GRAPH_ENABLED = original


# ---------------------------------------------------------------------------
# SC-010  corrupt graph → graceful fallback
# ---------------------------------------------------------------------------

def test_sc010_corrupt_graph_fallback(indexed_app, tmp_path):
    """SC-010: corrupt graph.json causes graceful fallback to vector-only, no crash."""
    import config
    from querying.query_engine import run_query

    # Write a corrupt graph.json into a copy of the index dir
    corrupt_index = str(tmp_path / "corrupt_index")
    shutil.copytree(indexed_app["index_dir"], corrupt_index)
    with open(os.path.join(corrupt_index, "graph.json"), "w") as fh:
        fh.write("not valid json {{{{")

    original = config.GRAPH_ENABLED
    config.GRAPH_ENABLED = True
    try:
        result = run_query(
            bug_text=GROUND_TRUTH[0].question,
            index_dir=corrupt_index,
            top_k=5,
            log=lambda _: None,
        )
        assert result.get("docs"), "Expected fallback vector-only results after corrupt graph"
    finally:
        config.GRAPH_ENABLED = original


# ---------------------------------------------------------------------------
# Full report (always passes)
# ---------------------------------------------------------------------------

def test_graph_retrieval_report(graph_query_results, graph_app):
    """Print per-query breakdown and delta vs test_04 baselines. Always passes."""
    # test_04 baselines captured from pre-implementation run
    BASELINE_R10 = {
        "loan_checkout_validation": 0.75,
        "loan_eligibility_chain": 0.60,
        "fine_calculation_strategy": 0.20,
        "hold_state_machine": 0.60,
        "recommendation_engine": 0.75,
        "overdue_batch_processing": 0.67,
        "full_text_search": 0.50,
        "notification_events": 0.75,
        "circulation_rules": 0.67,
        "reading_challenge": 0.75,
    }

    print("\n\n=== Graph Hybrid Retrieval Report ===\n")
    print(f"Graph build time: {graph_app['graph_build_time']:.1f}s")
    print(f"\n{'Query':<30} {'P@5':>6} {'R@10':>6} {'MRR':>6} {'ΔR@10':>7}")
    print("-" * 62)
    p5s, r10s, mrrs = [], [], []
    for case in GROUND_TRUTH:
        sources = graph_query_results[case.name]
        expected = set(case.expected_files)
        p5 = precision_at_k(sources, expected, 5)
        r10 = recall_at_k(sources, expected, 10)
        m = mrr(sources, expected)
        delta = r10 - BASELINE_R10.get(case.name, 0.0)
        p5s.append(p5); r10s.append(r10); mrrs.append(m)
        print(f"  {case.name:<28} {p5:>6.2f} {r10:>6.2f} {m:>6.2f} {delta:>+7.2f}")
    print("-" * 62)
    print(f"  {'MEAN':<28} {sum(p5s)/len(p5s):>6.2f} {sum(r10s)/len(r10s):>6.2f} {sum(mrrs)/len(mrrs):>6.2f}")
    print("\n  Targets: P@5>=0.50  R@10>=0.70  MRR>=0.88")
    print("=" * 50)


# ---------------------------------------------------------------------------
# α/β sweep — reports best weights for this corpus (always passes)
# ---------------------------------------------------------------------------

_ALPHA_VALS = [0.1, 0.3, 0.5, 0.7]
_BETA_VALS  = [0.4, 0.6, 0.8]
_SWEEP_PARAMS = [(a, b) for a in _ALPHA_VALS for b in _BETA_VALS]


@pytest.mark.parametrize("alpha,beta", _SWEEP_PARAMS, ids=[f"a{a}_b{b}" for a, b in _SWEEP_PARAMS])
def test_alphabeta_sweep(alpha, beta, graph_app):
    """Parametrized α/β sweep — reports mean R@10 for each combination. Always passes."""
    import config
    from querying.query_engine import run_query

    original_enabled = config.GRAPH_ENABLED
    original_alpha = config.GRAPH_ALPHA
    original_beta = config.GRAPH_BETA
    config.GRAPH_ENABLED = True
    config.GRAPH_ALPHA = alpha
    config.GRAPH_BETA = beta

    try:
        r10s = []
        for case in GROUND_TRUTH:
            result = run_query(
                bug_text=case.question,
                index_dir=graph_app["index_dir"],
                top_k=10,
                log=lambda _: None,
            )
            sources = [m.get("source", "") for m in result.get("metas", [])]
            r10s.append(recall_at_k(sources, set(case.expected_files), 10))
        mean_r10 = sum(r10s) / len(r10s)
        print(f"\n  α={alpha} β={beta}  mean R@10={mean_r10:.3f}")
    finally:
        config.GRAPH_ENABLED = original_enabled
        config.GRAPH_ALPHA = original_alpha
        config.GRAPH_BETA = original_beta
