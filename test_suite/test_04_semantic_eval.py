"""
Semantic retrieval evaluation — ground truth queries for the library-catalog-app corpus.

Metrics
-------
  Precision@5  : fraction of top-5 results that are expected files
  Recall@10    : fraction of expected files found in top-10 results
  MRR          : reciprocal rank of the first expected file hit

Thresholds calibrated to mxbai-embed-large + text_800 chunking.
"""

import pytest
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Ground truth cases — queries grounded in com.example.library source
# ---------------------------------------------------------------------------

@dataclass
class GroundTruthCase:
    name: str
    question: str
    expected_files: list[str] = field(default_factory=list)


GROUND_TRUTH: list[GroundTruthCase] = [
    GroundTruthCase(
        name="loan_checkout_validation",
        question=(
            "How does book checkout work? What validations are performed before "
            "a loan is created and what happens if a member has unpaid fines or "
            "too many active loans?"
        ),
        expected_files=[
            "src/main/java/com/example/library/service/LoanService.java",
            "src/main/java/com/example/library/controller/LoanController.java",
            "src/main/java/com/example/library/exception/CheckoutValidationException.java",
            "src/main/java/com/example/library/exception/MembershipExpiredException.java",
        ],
    ),
    GroundTruthCase(
        name="loan_eligibility_chain",
        question=(
            "How is the Chain of Responsibility pattern used to check loan eligibility? "
            "What handlers are in the chain and in what order do they run?"
        ),
        expected_files=[
            "src/main/java/com/example/library/pattern/chain/LoanEligibilityChain.java",
            "src/main/java/com/example/library/pattern/chain/MaxLoansHandler.java",
            "src/main/java/com/example/library/pattern/chain/UnpaidFinesHandler.java",
            "src/main/java/com/example/library/pattern/chain/MembershipActiveHandler.java",
            "src/main/java/com/example/library/pattern/chain/CopyAvailableHandler.java",
        ],
    ),
    GroundTruthCase(
        name="fine_calculation_strategy",
        question=(
            "How are overdue fines calculated? How does the fine amount differ by "
            "membership tier and is there a grace period for any member type?"
        ),
        expected_files=[
            "src/main/java/com/example/library/pattern/strategy/FineCalculationStrategy.java",
            "src/main/java/com/example/library/pattern/strategy/StandardFineStrategy.java",
            "src/main/java/com/example/library/pattern/strategy/PremiumFineStrategy.java",
            "src/main/java/com/example/library/pattern/strategy/StudentFineStrategy.java",
            "src/main/java/com/example/library/pattern/strategy/OverdueFineContext.java",
        ],
    ),
    GroundTruthCase(
        name="hold_state_machine",
        question=(
            "How are holds managed through state transitions? What states can a hold be in "
            "and how does the state machine determine which transitions are valid?"
        ),
        expected_files=[
            "src/main/java/com/example/library/pattern/state/HoldStateMachine.java",
            "src/main/java/com/example/library/pattern/state/HoldContext.java",
            "src/main/java/com/example/library/pattern/state/PendingHoldState.java",
            "src/main/java/com/example/library/pattern/state/ReadyHoldState.java",
            "src/main/java/com/example/library/service/HoldService.java",
        ],
    ),
    GroundTruthCase(
        name="recommendation_engine",
        question=(
            "How does the book recommendation system work? What algorithms are used "
            "and how are recommendations cached to avoid repeated computation?"
        ),
        expected_files=[
            "src/main/java/com/example/library/recommendation/RecommendationEngine.java",
            "src/main/java/com/example/library/recommendation/HybridRecommendationService.java",
            "src/main/java/com/example/library/recommendation/CollaborativeFilteringService.java",
            "src/main/java/com/example/library/recommendation/RecommendationCache.java",
        ],
    ),
    GroundTruthCase(
        name="overdue_batch_processing",
        question=(
            "How are overdue loans detected and processed in batch? How are fines issued "
            "and members notified, and what scheduled job triggers the process?"
        ),
        expected_files=[
            "src/main/java/com/example/library/batch/OverdueBatchProcessor.java",
            "src/main/java/com/example/library/batch/BatchJobService.java",
            "src/main/java/com/example/library/scheduler/OverdueLoanScheduler.java",
        ],
    ),
    GroundTruthCase(
        name="full_text_search",
        question=(
            "How does the library's full-text search work? How is the search index built "
            "and how are results ranked and paginated?"
        ),
        expected_files=[
            "src/main/java/com/example/library/search/FullTextSearchService.java",
            "src/main/java/com/example/library/search/SearchIndexService.java",
            "src/main/java/com/example/library/search/SearchController.java",
            "src/main/java/com/example/library/search/SearchIndexingEventListener.java",
        ],
    ),
    GroundTruthCase(
        name="notification_events",
        question=(
            "How do library domain events trigger member notifications? "
            "Which events send notifications and how are they dispatched asynchronously?"
        ),
        expected_files=[
            "src/main/java/com/example/library/pattern/observer/NotificationEventListener.java",
            "src/main/java/com/example/library/pattern/observer/HoldReadyEvent.java",
            "src/main/java/com/example/library/pattern/observer/BookCheckedOutEvent.java",
            "src/main/java/com/example/library/service/NotificationService.java",
        ],
    ),
    GroundTruthCase(
        name="circulation_rules",
        question=(
            "How are circulation rules applied to determine loan periods, renewal limits, "
            "and fine rates? How does the system pick the most specific rule for a member?"
        ),
        expected_files=[
            "src/main/java/com/example/library/circulation/CirculationRulesEngine.java",
            "src/main/java/com/example/library/circulation/CirculationRuleService.java",
            "src/main/java/com/example/library/circulation/CirculationRuleController.java",
        ],
    ),
    GroundTruthCase(
        name="reading_challenge",
        question=(
            "How do members enroll in and complete reading challenges? "
            "How is reading progress tracked and what determines challenge completion?"
        ),
        expected_files=[
            "src/main/java/com/example/library/readingchallenge/ReadingChallengeService.java",
            "src/main/java/com/example/library/readingchallenge/ReadingChallengeController.java",
            "src/main/java/com/example/library/readingchallenge/ChallengeParticipation.java",
            "src/main/java/com/example/library/readingchallenge/ChallengeProgress.java",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Metric helpers (imported by test_05, test_06, test_09)
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for s in top_k if any(e in s for e in expected))
    return hits / k


def recall_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return 1.0
    top_k = retrieved[:k]
    found = sum(1 for e in expected if any(e in r for r in top_k))
    return found / len(expected)


def mrr(retrieved: list[str], expected: set[str]) -> float:
    for i, source in enumerate(retrieved, start=1):
        if any(e in source for e in expected):
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

PRECISION_AT_5_THRESHOLD = 0.35
RECALL_AT_10_THRESHOLD   = 0.50
MRR_THRESHOLD            = 0.40
PER_QUERY_RECALL_FLOOR   = 0.25


# ---------------------------------------------------------------------------
# Session fixture — index once, reuse for all queries
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def query_results(indexed_app) -> dict:
    """Run all ground-truth queries against the indexed app. Results cached for session."""
    from querying.query_engine import run_query

    results = {}
    for case in GROUND_TRUTH:
        result = run_query(
            bug_text=case.question,
            index_dir=indexed_app["index_dir"],
            top_k=10,
            log=lambda _: None,
        )
        results[case.name] = [m.get("source", "") for m in result.get("metas", [])]
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", GROUND_TRUTH, ids=[c.name for c in GROUND_TRUTH])
def test_recall_floor(case, query_results):
    """Every query must find at least one expected file in the top-10 results."""
    sources  = query_results[case.name]
    expected = set(case.expected_files)
    r10 = recall_at_k(sources, expected, 10)
    assert r10 >= PER_QUERY_RECALL_FLOOR, (
        f"{case.name}: recall@10 {r10:.2f} < floor {PER_QUERY_RECALL_FLOOR} — "
        f"expected one of {list(expected)[:2]} in top-10, got {sources[:5]}"
    )


def test_mean_precision_at_5(query_results):
    """Mean precision@5 across all queries must meet threshold."""
    scores = [
        precision_at_k(query_results[c.name], set(c.expected_files), 5)
        for c in GROUND_TRUTH
    ]
    mean = sum(scores) / len(scores)
    assert mean >= PRECISION_AT_5_THRESHOLD, (
        f"Mean P@5 {mean:.2f} < {PRECISION_AT_5_THRESHOLD}"
    )


def test_mean_recall_at_10(query_results):
    """Mean recall@10 across all queries must meet threshold."""
    scores = [
        recall_at_k(query_results[c.name], set(c.expected_files), 10)
        for c in GROUND_TRUTH
    ]
    mean = sum(scores) / len(scores)
    assert mean >= RECALL_AT_10_THRESHOLD, (
        f"Mean R@10 {mean:.2f} < {RECALL_AT_10_THRESHOLD}"
    )


def test_mean_mrr(query_results):
    """Mean MRR across all queries must meet threshold."""
    scores = [
        mrr(query_results[c.name], set(c.expected_files))
        for c in GROUND_TRUTH
    ]
    mean = sum(scores) / len(scores)
    assert mean >= MRR_THRESHOLD, (
        f"Mean MRR {mean:.2f} < {MRR_THRESHOLD}"
    )


def test_semantic_eval_report(query_results):
    """Print a per-query breakdown table. Always passes."""
    print("\n\n=== Semantic Retrieval Evaluation ===\n")
    print(f"{'Query':<30} {'P@5':>6} {'R@10':>6} {'MRR':>6}")
    print("-" * 52)
    p5s, r10s, mrrs = [], [], []
    for case in GROUND_TRUTH:
        sources  = query_results[case.name]
        expected = set(case.expected_files)
        p5  = precision_at_k(sources, expected, 5)
        r10 = recall_at_k(sources, expected, 10)
        m   = mrr(sources, expected)
        p5s.append(p5); r10s.append(r10); mrrs.append(m)
        print(f"  {case.name:<28} {p5:>6.2f} {r10:>6.2f} {m:>6.2f}")
    print("-" * 52)
    mean_p5  = sum(p5s)  / len(p5s)
    mean_r10 = sum(r10s) / len(r10s)
    mean_mrr = sum(mrrs) / len(mrrs)
    print(f"  {'MEAN':<28} {mean_p5:>6.2f} {mean_r10:>6.2f} {mean_mrr:>6.2f}")
    print(f"\n  Thresholds: P@5>={PRECISION_AT_5_THRESHOLD}  "
          f"R@10>={RECALL_AT_10_THRESHOLD}  MRR>={MRR_THRESHOLD}")
    print("=" * 40)
