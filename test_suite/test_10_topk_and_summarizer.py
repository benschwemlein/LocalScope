"""
Top-K result count and summarizer comparison — NOT part of the regular test suite.

Run explicitly with:
    python3.13 -m pytest test_suite/test_10_topk_and_summarizer.py -m chunking_eval -v -s

Part 1 — Top-K comparison
--------------------------
Tests retrieval quality at top_k = 8, 12, 20, 40.
Metrics: p@5, r@10, MRR (fixed k values, independent of top_k).
Note: top_k=8 caps r@10 at r@8 since fewer results are returned.

Part 2 — Summarizer comparison
--------------------------------
Each ground-truth question is padded to >5000 chars with realistic bug-report
context, then queried with max_chars=5000 (triggers LLM summarization before
embedding) vs max_chars=999999 (raw long text embedded directly).
Tests whether the summarizer improves or hurts retrieval on long inputs.

No pass/fail thresholds — the report is the output.
"""

import pytest

from test_04_semantic_eval import GROUND_TRUTH, precision_at_k, recall_at_k, mrr

TOP_K_VALUES = [8, 12, 20, 40]

# Long-form preamble that pads each question to >5000 chars while staying realistic
LONG_PREAMBLE = """
Investigation notes from on-call rotation — please help locate the relevant source code.

Background:
We are investigating a production issue reported by the trading operations team.
The system processes cryptocurrency market data from multiple exchanges (Bitstamp,
Bitfinex, Coinbase, Paxos/Itbit) and stores quotes in MongoDB. The backend is a
Spring Boot application using reactive programming (Project Reactor / WebFlux).
The frontend is an Angular single-page application that calls REST endpoints to
display charts, orderbooks, and statistics.

The issue was first noticed at 14:32 UTC when the monitoring dashboard showed
elevated error rates on several endpoints. The on-call engineer noticed that
MongoDB connection timeouts were increasing and Kafka consumer lag was growing.
Initial triage suggested the problem was in the data ingestion pipeline but we
have not been able to pinpoint the exact component.

The application uses JWT-based authentication with tokens stored in-memory on
the server side. Revoked tokens are tracked in a thread-safe list that is
periodically refreshed from a Kafka topic. The scheduled tasks run every 60-90
seconds and use distributed locking (@SchedulerLock) to prevent duplicate
execution across multiple instances.

Relevant error from logs:
  [ERROR] ch.xxx.trader - MongoTimeoutException: Timed out after 6000 ms while
  waiting for a server that matches ReadPreferenceServerSelector
  [WARN]  ch.xxx.trader - Kafka consumer lag on USER_LOGOUT_SINK_TOPIC: 1240
  [ERROR] ch.xxx.trader - Failed to refresh revoked token list after 3 attempts

Stack trace fragment:
  at ch.xxx.trader.adapter.cron.ScheduledTask.updateLoggedOutUsers(ScheduledTask.java:87)
  at ch.xxx.trader.usecase.services.JwtTokenService.validateToken(JwtTokenService.java:134)
  at ch.xxx.trader.adapter.config.JwtTokenFilter.doFilter(JwtTokenFilter.java:52)

Additional context:
The deployment uses three application instances behind a load balancer. Each
instance connects to the same MongoDB replica set and Kafka cluster. The issue
appears to affect all three instances simultaneously, suggesting the root cause
is in a shared dependency rather than instance-specific state.

Specifically we want to understand:
""".strip()

# Total preamble is ~2200 chars; each question adds ~50-100 chars — still under 5000.
# Add more padding to guarantee >5000 chars.
EXTRA_PADDING = """

Additional investigation context:
We have checked the following and ruled them out:
1. Network connectivity between app instances and MongoDB — all healthy
2. MongoDB disk space and memory — both within normal ranges
3. Kafka broker health — brokers are up, topic partitions are balanced
4. JVM heap usage — no GC pressure observed

What we have NOT checked yet:
- Whether the scheduled task thread pool is saturated
- Whether the JWT token revocation list is growing unbounded
- Whether the Kafka consumer group rebalancing is causing delays
- Whether the MongoDB connection pool is exhausted under load

Please find the source files responsible for the component described in the
question below, focusing on the Java backend and Angular frontend code.
The codebase follows a hexagonal architecture with adapter, usecase, and domain layers.
""".strip()


def _make_long_query(question: str) -> str:
    """Pad a short question to >5000 chars with realistic bug-report context."""
    combined = f"{LONG_PREAMBLE}\n\n{question}\n\n{EXTRA_PADDING}"
    # Keep padding until well over 5000 chars
    while len(combined) < 5100:
        combined += f"\n\nFurther context: {question}"
    return combined


def _query_topk(index_dir: str, question: str, top_k: int) -> list[str]:
    from querying.query_engine import run_query
    result = run_query(
        bug_text=question,
        index_dir=index_dir,
        top_k=top_k,
        log=lambda _: None,
    )
    return [m.get("source", "") for m in result.get("metas", [])]


def _query_with_max_chars(index_dir: str, question: str, max_chars: int) -> list[str]:
    from querying.query_engine import run_query
    import config
    result = run_query(
        bug_text=question,
        index_dir=index_dir,
        top_k=10,
        max_chars=max_chars,
        summarizer_template=config.SUMMARIZER_TEMPLATE if hasattr(config, 'SUMMARIZER_TEMPLATE') else "",
        log=lambda _: None,
    )
    return [m.get("source", "") for m in result.get("metas", [])]


def _score(sources: list[str], expected: set[str]) -> dict:
    return {
        "p5":  precision_at_k(sources, expected, 5),
        "r10": recall_at_k(sources, expected, min(10, len(sources))),
        "mrr": mrr(sources, expected),
    }


# ---------------------------------------------------------------------------
# Part 1 — Top-K comparison
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test10_index(tmp_path_factory, sample_app_path) -> dict:
    """Own index for test_10 — avoids dimension mismatch with shared indexed_app."""
    from indexing.incremental_indexer import index_repo_incremental
    index_dir = str(tmp_path_factory.mktemp("chroma_test10"))
    logs = []
    index_repo_incremental(
        repo_root=sample_app_path,
        index_dir=index_dir,
        force_full_reindex=True,
        num_workers=1,
        verbose=False,
        log=logs.append,
    )
    return {"index_dir": index_dir}


@pytest.fixture(scope="session")
def topk_results(test10_index) -> dict:
    """Run all queries at each top_k. Returns {top_k: {case_name: sources}}."""
    results = {}
    for k in TOP_K_VALUES:
        print(f"\n  Querying with top_k={k}...")
        results[k] = {}
        for case in GROUND_TRUTH:
            results[k][case.name] = _query_topk(test10_index["index_dir"], case.question, k)
    return results


@pytest.mark.chunking_eval
def test_topk_comparison(topk_results):
    """Compare retrieval quality at different top_k values. Always passes."""
    scores = {}
    for k in TOP_K_VALUES:
        p5s, r10s, mrrs = [], [], []
        per_case = {}
        for case in GROUND_TRUTH:
            s = _score(topk_results[k][case.name], set(case.expected_files))
            p5s.append(s["p5"]); r10s.append(s["r10"]); mrrs.append(s["mrr"])
            per_case[case.name] = s
        scores[k] = {
            "mean_p5":  sum(p5s)  / len(p5s),
            "mean_r10": sum(r10s) / len(r10s),
            "mean_mrr": sum(mrrs) / len(mrrs),
            "per_case": per_case,
        }

    print("\n\n=== Top-K Comparison ===\n")
    print(f"{'top_k':<8} {'p@5':>6} {'r@10':>6} {'MRR':>6}")
    print("-" * 32)
    for k in TOP_K_VALUES:
        r = scores[k]
        marker = "  ← current default" if k == 12 else ""
        print(f"  k={k:<4}  {r['mean_p5']:>6.2f} {r['mean_r10']:>6.2f} {r['mean_mrr']:>6.2f}{marker}")
    print("-" * 32)

    print(f"\n{'Query':<32}", end="")
    for k in TOP_K_VALUES:
        print(f" {'k='+str(k):>8}", end="")
    print(f"\n  (values shown are r@10)")
    print("-" * (32 + len(TOP_K_VALUES) * 9))
    for case in GROUND_TRUTH:
        print(f"  {case.name:<30}", end="")
        for k in TOP_K_VALUES:
            print(f" {scores[k]['per_case'][case.name]['r10']:>8.2f}", end="")
        print()
    print("\n" + "=" * 40)


# ---------------------------------------------------------------------------
# Part 2 — Summarizer comparison
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def summarizer_results(test10_index) -> dict:
    """
    Run each query as a long text (>5000 chars) with and without summarization.
    Returns {mode: {case_name: sources}}
    modes: 'summarized' (max_chars=5000), 'raw_long' (max_chars=999999)
    """
    import config
    results = {"summarized": {}, "raw_long": {}}

    # Load summarizer template from config
    summarizer_tmpl = ""
    try:
        import json, os
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        summarizer_tmpl = cfg.get("prompts_tab", {}).get("summarizer_prompt", "")
    except Exception:
        pass

    print("\n  Running summarizer comparison (long queries >5000 chars)...")
    for case in GROUND_TRUTH:
        long_q = _make_long_query(case.question)
        assert len(long_q) > 5000, f"Query not long enough: {len(long_q)} chars"

        # With summarization (max_chars=5000 triggers summarizer)
        from querying.query_engine import run_query
        result_sum = run_query(
            bug_text=long_q,
            index_dir=test10_index["index_dir"],
            top_k=10,
            max_chars=5000,
            summarizer_template=summarizer_tmpl,
            log=lambda _: None,
        )
        results["summarized"][case.name] = [m.get("source", "") for m in result_sum.get("metas", [])]

        # Without summarization (max_chars very high — raw long text embedded)
        result_raw = run_query(
            bug_text=long_q,
            index_dir=test10_index["index_dir"],
            top_k=10,
            max_chars=999999,
            log=lambda _: None,
        )
        results["raw_long"][case.name] = [m.get("source", "") for m in result_raw.get("metas", [])]

    return results


@pytest.mark.chunking_eval
def test_summarizer_comparison(summarizer_results):
    """Compare summarized vs raw-long embedding for queries >5000 chars. Always passes."""
    modes = ["summarized", "raw_long"]
    scores = {}
    for mode in modes:
        p5s, r10s, mrrs = [], [], []
        per_case = {}
        for case in GROUND_TRUTH:
            s = _score(summarizer_results[mode][case.name], set(case.expected_files))
            p5s.append(s["p5"]); r10s.append(s["r10"]); mrrs.append(s["mrr"])
            per_case[case.name] = s
        scores[mode] = {
            "mean_p5":  sum(p5s)  / len(p5s),
            "mean_r10": sum(r10s) / len(r10s),
            "mean_mrr": sum(mrrs) / len(mrrs),
            "per_case": per_case,
        }

    print("\n\n=== Summarizer Comparison (queries >5000 chars) ===\n")
    print(f"{'Mode':<16} {'p@5':>6} {'r@10':>6} {'MRR':>6}")
    print("-" * 38)
    for mode in modes:
        r = scores[mode]
        print(f"  {mode:<14} {r['mean_p5']:>6.2f} {r['mean_r10']:>6.2f} {r['mean_mrr']:>6.2f}")
    print("-" * 38)

    print(f"\n{'Query':<32} {'summarized':>12} {'raw_long':>10}  (r@10)")
    print("-" * 58)
    for case in GROUND_TRUTH:
        s  = scores["summarized"]["per_case"][case.name]["r10"]
        rl = scores["raw_long"]["per_case"][case.name]["r10"]
        diff = s - rl
        marker = f"  +{diff:.2f}" if diff > 0 else (f"  {diff:.2f}" if diff < 0 else "  =")
        print(f"  {case.name:<30} {s:>12.2f} {rl:>10.2f}{marker}")
    print("\n" + "=" * 40)
