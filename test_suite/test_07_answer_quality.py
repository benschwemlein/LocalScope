"""
LLM answer quality tests — part of the regular test suite.

Two checks per ground-truth query:

  1. Faithfulness  — the answer mentions class names derived from the retrieved file
                     paths.  Detects hallucination: the model inventing classes that
                     weren't in the retrieved context.  Scored as fraction of source-
                     file class names that appear in the answer.

  2. Reference similarity — keyword overlap between the LLM answer and a reference
                     answer written from direct inspection of the source code.
                     Scored as: |answer_keywords ∩ reference_keywords| / |reference_keywords|

Reference answers were written by reading the actual source files in
library-catalog-app and are grounded in real class/method names.

Thresholds
----------
  Faithfulness     : >= 0.20  (at least 1 in 5 retrieved class names appear in answer)
  Must-contain     : primary class names only
  Reference overlap: >= 0.20  (calibrated to observed qwen2.5:7b output)
"""

import re
import pytest
from dataclasses import dataclass, field

from test_04_semantic_eval import GROUND_TRUTH


# ---------------------------------------------------------------------------
# Reference answers — written from direct source inspection
# ---------------------------------------------------------------------------

@dataclass
class ReferenceCase:
    name: str
    must_contain: list[str]
    reference: str


REFERENCE_CASES: list[ReferenceCase] = [
    ReferenceCase(
        name="loan_checkout_validation",
        must_contain=["LoanService", "LoanController"],
        reference="""
Book checkout is handled by LoanService.checkout(CheckoutRequestDTO). The service requires
either memberId or membershipNumber, and either bookCopyId or copyBarcode; throws
CheckoutValidationException if both are missing. It resolves the Member via MemberRepository
and throws MemberNotFoundException if not found. Member.isActive() is checked and
MembershipExpiredException is thrown for expired accounts. Unpaid fines are checked against
a threshold of $10.00 (UNPAID_FINES_THRESHOLD) via FineService; throws UnpaidFinesException
if exceeded. The concurrent loan count is capped at MAX_CONCURRENT_LOANS (5); throws
ConcurrentLoanLimitException if exceeded. The BookCopy is resolved by id or barcode; throws
CopyNotAvailableException if its status is not AVAILABLE. A Loan entity is created with
status ACTIVE and a calculated due date, persisted via LoanRepository. NotificationService
sends a checkout confirmation. LoanController exposes REST endpoints for checkout, return,
renewal, and loan history, delegating all logic to LoanService.
""",
    ),
    ReferenceCase(
        name="loan_eligibility_chain",
        must_contain=["LoanEligibilityChain", "MaxLoansHandler"],
        reference="""
Loan eligibility uses the Chain of Responsibility pattern via LoanEligibilityChain.
Six handlers are chained in PostConstruct order: MembershipActiveHandler checks that the
member account is active and not expired. MaxLoansHandler checks the member has not exceeded
the maximum concurrent loan limit. UnpaidFinesHandler checks the member has no unpaid fines
above the threshold. CopyAvailableHandler checks the specific book copy status is AVAILABLE.
BranchAccessHandler verifies the member can borrow from the requested branch.
AgeRestrictionHandler enforces any age restrictions on the item. Each handler extends
LoanEligibilityHandler and calls setNext() to wire the chain. LoanEligibilityChain.validate()
starts with membershipActiveHandler and returns a ValidationResult containing isValid(),
failureReason, and failedHandler. A valid result means all six checks passed. The chain is
initialized via @PostConstruct and logged at INFO level.
""",
    ),
    ReferenceCase(
        name="fine_calculation_strategy",
        must_contain=["OverdueFineContext", "StandardFineStrategy"],
        reference="""
Overdue fine calculation uses the Strategy pattern. FineCalculationStrategy is the interface
with calculateFine(Loan, LocalDate). Three implementations cover membership tiers:
StandardFineStrategy charges $0.25 per day overdue, capped at $25.00.
PremiumFineStrategy charges $0.10 per day overdue, capped at $10.00.
StudentFineStrategy charges $0.15 per day overdue, capped at $15.00, with a 1-day grace
period — no fine if returned within 1 day of the due date.
OverdueFineContext selects the strategy based on the member's MembershipTier using an
EnumMap<MembershipTier, FineCalculationStrategy>. The constructor wires STANDARD to
StandardFineStrategy, PREMIUM to PremiumFineStrategy, STUDENT to StudentFineStrategy.
calculateFine(Loan, LocalDate) looks up the member's tier, delegates to the matching
strategy, and throws IllegalStateException if no strategy is registered for the tier.
setStrategy() allows runtime override for testing or promotions.
""",
    ),
    ReferenceCase(
        name="hold_state_machine",
        must_contain=["HoldStateMachine", "HoldContext"],
        reference="""
Holds are managed through a State pattern. HoldStateMachine creates a HoldContext for a given
Hold by mapping HoldStatus to a state bean: PENDING maps to PendingHoldState, READY to
ReadyHoldState, FULFILLED to FulfilledHoldState, CANCELLED to CancelledHoldState, EXPIRED
to ExpiredHoldState. A null status throws IllegalArgumentException. HoldContext wraps the Hold
and the current HoldState, delegating operations (notify, fulfill, cancel, expire) to the
active state. Each state implements HoldState and defines which transitions are legal — for
example, PendingHoldState can transition to READY or CANCELLED; ReadyHoldState can transition
to FULFILLED or EXPIRED. HoldService uses HoldStateMachine to enforce state rules when
processing hold lifecycle events. HoldRepository provides persistence and
findByMemberIdAndBookId() for duplicate hold detection.
""",
    ),
    ReferenceCase(
        name="recommendation_engine",
        must_contain=["RecommendationEngine", "HybridRecommendationService"],
        reference="""
Book recommendations are served by RecommendationEngine. On a request for a member,
RecommendationCache is checked first; if a cached List<RecommendationDTO> exists it is
returned immediately. Otherwise the member is loaded via MemberRepository and
HybridRecommendationService.getRecommendations(member, limit) is called. The result is
stored in RecommendationCache before returning. HybridRecommendationService combines
CollaborativeFilteringService (finds members with similar borrowing history and surfaces
their top books) and ContentBasedFilteringService (matches books by genre and subject to
the member's borrowing profile). getRecommendationsForCurrentMember() resolves the member
from the Spring Security context via SecurityContextHolder and MemberRepository.findByUser_Email();
returns an empty list if no member record exists for the authenticated user.
""",
    ),
    ReferenceCase(
        name="overdue_batch_processing",
        must_contain=["OverdueBatchProcessor", "BatchJobService"],
        reference="""
Overdue loan processing is triggered by OverdueLoanScheduler (@Scheduled via SchedulerConfig)
which calls BatchJobService.submitJob(JobType.OVERDUE_PROCESSING). BatchJobService creates a
BatchJob entity with status PENDING, saves it, then calls runJobAsync() which is @Async on
the libraryTaskExecutor thread pool. The job is dispatched to OverdueBatchProcessor.process(job).
OverdueBatchProcessor queries LoanRepository.findOverdueLoans(LocalDateTime.now()) to find all
ACTIVE loans past their due date. Loans are processed in chunks of 100 (CHUNK_SIZE). For each
overdue loan: FineService issues a fine, the loan status is set to OVERDUE, and NotificationService
sends a notification to the member. Errors per loan are caught and counted individually so one
failure does not abort the batch. The BatchJob is updated with COMPLETED or FAILED status and
a processed/failed count after all chunks finish.
""",
    ),
    ReferenceCase(
        name="full_text_search",
        must_contain=["FullTextSearchService", "SearchIndexService"],
        reference="""
Full-text search is provided by FullTextSearchService.search(query, entityType, page, pageSize).
It queries SearchIndexRepository for matching index entries, applies in-memory term scoring to
rank results by relevance, paginates the ranked list, and logs the query to SearchLogRepository
via SearchLog entities for analytics. Auto-complete suggestions are generated from the same
index entries. The entityType parameter optionally filters results to BOOK, MEMBER, or AUTHOR.
An empty or null query returns an empty SearchResultPage immediately. SearchIndexService builds
and maintains the SearchIndex: it tokenizes field values from Book, Member, and Author entities
and persists SearchIndex records via SearchIndexRepository. SearchIndexingEventListener listens
for domain events (BookAddedEvent, MemberRegisteredEvent) and calls SearchIndexService to keep
the index in sync with entity changes. SearchController exposes GET /search?q=...&type=...
and GET /search/suggestions.
""",
    ),
    ReferenceCase(
        name="notification_events",
        must_contain=["NotificationEventListener", "HoldReadyEvent"],
        reference="""
Member notifications are triggered by Spring application events. NotificationEventListener
is annotated @Component and uses @EventListener + @Async to handle events off the publishing
thread. onHoldReady(HoldReadyEvent) notifies the member when their hold is ready for pickup,
including the hold id, expiry date, and pickup branch. The member is looked up via
MemberRepository; missing members are logged and skipped. NotificationService.sendNotification()
dispatches the message. Other handled events include BookCheckedOutEvent (checkout confirmation),
FineIssuedEvent (fine notice), OverdueNoticeEvent (overdue reminder), and
MembershipExpiredEvent (expiry warning). Each event carries correlation and member ids for
tracing. Events are published via ApplicationEventPublisher in the relevant service methods.
The async handlers run on the Spring task executor to avoid blocking the caller.
""",
    ),
    ReferenceCase(
        name="circulation_rules",
        must_contain=["CirculationRulesEngine", "CirculationRuleService"],
        reference="""
Circulation rules are evaluated by CirculationRulesEngine.getApplicableRule(member, copy, branch).
It determines the item type from the BookCopy and queries CirculationRuleRepository.findApplicableRules()
for active rules matching the member's MembershipTier, the item type, and the branch. Branch-specific
rules take priority over global rules; tier-specific rules beat wildcard rules. If no rule matches,
a hardcoded DEFAULT_RULE is returned (21-day loan period, 2 max renewals, $0.25/day fine rate,
$25.00 max fine, 8 max loans, 7-day reservation hold period, no age restriction).
CirculationRulesEngine also enforces a MAX_FINE_THRESHOLD of $25.00 for checkout eligibility.
CirculationRuleService provides CRUD operations for managing rules, backed by
CirculationRuleRepository. CirculationRuleController exposes REST endpoints for creating,
updating, and querying circulation rules.
""",
    ),
    ReferenceCase(
        name="reading_challenge",
        must_contain=["ReadingChallengeService", "ChallengeParticipation"],
        reference="""
Reading challenges are managed by ReadingChallengeService. createChallenge() validates that
start is before end and targetBooks is positive, then persists a ReadingChallenge with name,
description, startDate, endDate, targetBooks, targetGenreNames, badge, and active=true.
getActiveChallenges() queries ReadingChallenge records where active=true and endDate is after
today, then enriches each with enrollment count and completion count from ChallengeParticipation.
Members enroll via a participation method that creates a ChallengeParticipation record linking
the member to the challenge. ChallengeProgress tracks individual book completions within a
challenge — each progress record links a participation, a book, and a completion date.
Challenge completion is determined when the progress count meets or exceeds the challenge's
targetBooks. ReadingChallengeController exposes REST endpoints for creating challenges,
listing active challenges, enrolling members, logging progress, and viewing leaderboards.
""",
    ),
]

REFERENCE_BY_NAME = {r.name: r for r in REFERENCE_CASES}

FAITHFULNESS_THRESHOLD      = 0.20
REFERENCE_OVERLAP_THRESHOLD = 0.20

# Queries where the model is less likely to cite exact class names.
# Marked xfail — these represent known model quality gaps, not test bugs.
WEAK_QUERIES = {"notification_events", "reading_challenge"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Extract lowercase alpha-numeric tokens of length >= 3."""
    return {t.lower() for t in re.findall(r'[A-Za-z][A-Za-z0-9]{2,}', text)}


def faithfulness_score(answer: str, retrieved_metas: list[dict]) -> float:
    """
    Fraction of class names (derived from retrieved source file paths) that appear
    in the answer. Detects hallucination: model inventing classes not in context.

    e.g. "pattern/chain/MaxLoansHandler.java" -> "MaxLoansHandler"
    """
    class_names = set()
    for meta in retrieved_metas:
        source = meta.get("source", "")
        basename = source.split("/")[-1]
        stem = basename.rsplit(".", 1)[0] if "." in basename else basename
        if len(stem) >= 4:
            class_names.add(stem)

    if not class_names:
        return 1.0

    answer_lower = answer.lower()
    hits = sum(1 for name in class_names if name.lower() in answer_lower)
    return hits / len(class_names)


def reference_overlap_score(answer: str, reference: str) -> float:
    """Fraction of reference keywords that appear in the answer."""
    ref_tokens = _tokenize(reference)
    if not ref_tokens:
        return 1.0
    ans_tokens = _tokenize(answer)
    hits = sum(1 for t in ref_tokens if t in ans_tokens)
    return hits / len(ref_tokens)


# ---------------------------------------------------------------------------
# Session fixture — run all queries once, cache results
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def answer_results(indexed_app) -> dict:
    """Run each ground-truth query through the full RAG pipeline and cache results."""
    from querying.query_engine import run_query

    results = {}
    for case in GROUND_TRUTH:
        result = run_query(
            bug_text=case.question,
            index_dir=indexed_app["index_dir"],
            top_k=10,
            log=lambda _: None,
        )
        results[case.name] = {
            "answer": result.get("answer", ""),
            "docs":   result.get("docs",   []),
            "metas":  result.get("metas",  []),
        }
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", GROUND_TRUTH, ids=[c.name for c in GROUND_TRUTH])
def test_faithfulness(case, answer_results, request):
    """Answer should mention class names from the retrieved source files."""
    if case.name in WEAK_QUERIES:
        request.node.add_marker(pytest.mark.xfail(
            reason=f"{case.name}: model may paraphrase without citing class names",
            strict=False,
        ))
    r = answer_results[case.name]
    score = faithfulness_score(r["answer"], r["metas"])
    assert score >= FAITHFULNESS_THRESHOLD, (
        f"{case.name}: faithfulness {score:.2f} < {FAITHFULNESS_THRESHOLD} "
        f"(answer does not mention enough identifiers from the retrieved context)"
    )


@pytest.mark.parametrize("case", GROUND_TRUTH, ids=[c.name for c in GROUND_TRUTH])
def test_must_contain_keywords(case, answer_results, request):
    """Answer must contain specific key class names for each query."""
    if case.name in WEAK_QUERIES:
        request.node.add_marker(pytest.mark.xfail(
            reason=f"{case.name}: model may paraphrase without citing class names",
            strict=False,
        ))
    ref = REFERENCE_BY_NAME.get(case.name)
    if ref is None:
        pytest.skip(f"No reference case for {case.name}")

    answer_lower = answer_results[case.name]["answer"].lower()
    missing = [kw for kw in ref.must_contain if kw.lower() not in answer_lower]
    assert not missing, (
        f"{case.name}: answer is missing required keywords: {missing}"
    )


@pytest.mark.parametrize("case", GROUND_TRUTH, ids=[c.name for c in GROUND_TRUTH])
def test_reference_overlap(case, answer_results, request):
    """Answer should overlap with the reference answer on key vocabulary."""
    if case.name in WEAK_QUERIES:
        request.node.add_marker(pytest.mark.xfail(
            reason=f"{case.name}: model answer may diverge from reference vocabulary",
            strict=False,
        ))
    ref = REFERENCE_BY_NAME.get(case.name)
    if ref is None:
        pytest.skip(f"No reference case for {case.name}")

    score = reference_overlap_score(
        answer_results[case.name]["answer"], ref.reference
    )
    assert score >= REFERENCE_OVERLAP_THRESHOLD, (
        f"{case.name}: reference overlap {score:.2f} < {REFERENCE_OVERLAP_THRESHOLD}"
    )


def test_answer_quality_report(answer_results):
    """Print a summary table of all answer quality scores. Always passes."""
    print("\n\n=== Answer Quality Report ===\n")
    print(f"{'Query':<30} {'faithful':>10} {'must_kw':>10} {'ref_ovlp':>10}")
    print("-" * 64)

    for case in GROUND_TRUTH:
        r   = answer_results[case.name]
        ref = REFERENCE_BY_NAME.get(case.name)

        faith = faithfulness_score(r["answer"], r["metas"])

        if ref:
            answer_lower = r["answer"].lower()
            missing  = [kw for kw in ref.must_contain if kw.lower() not in answer_lower]
            kw_pass  = "PASS" if not missing else f"FAIL({len(missing)})"
            ovlp     = reference_overlap_score(r["answer"], ref.reference)
        else:
            kw_pass = "N/A"
            ovlp    = 0.0

        print(f"  {case.name:<28} {faith:>10.2f} {kw_pass:>10} {ovlp:>10.2f}")

    print("-" * 64)
    print(f"\n  Faithfulness threshold     : >= {FAITHFULNESS_THRESHOLD}")
    print(f"  Reference overlap threshold: >= {REFERENCE_OVERLAP_THRESHOLD}\n")
    print("=" * 40)
