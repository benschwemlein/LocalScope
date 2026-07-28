"""
Symbol-anchored query routing — unit tests.

No Ollama or ChromaDB dependency: builds a small synthetic GraphStore
directly and asserts which questions route to which precanned graph query,
and which ones correctly fall through to the conceptual (vector) path.
"""

from graph.edge import Edge, EdgeType
from graph.graph_store import GraphStore
from querying.router import resolve_symbol_to_file, route_query


def build_sample_store() -> GraphStore:
    store = GraphStore()
    store.add_edges([
        Edge(
            "src/main/java/com/example/library/pattern/strategy/OverdueFineContext.java",
            "src/main/java/com/example/library/pattern/strategy/StandardFineStrategy.java",
            EdgeType.INVOKES,
        ),
        Edge(
            "src/main/java/com/example/library/service/LoanController.java",
            "src/main/java/com/example/library/pattern/strategy/OverdueFineContext.java",
            EdgeType.INVOKES,
        ),
        Edge(
            "src/main/java/com/example/library/pattern/strategy/StandardFineStrategy.java",
            "src/main/java/com/example/library/pattern/strategy/FineCalculationStrategy.java",
            EdgeType.INHERITS,
        ),
        Edge(
            "src/main/java/com/example/library/pattern/strategy/PremiumFineStrategy.java",
            "src/main/java/com/example/library/pattern/strategy/FineCalculationStrategy.java",
            EdgeType.INHERITS,
        ),
    ])
    return store


# ---------------------------------------------------------------------------
# Symbol resolution
# ---------------------------------------------------------------------------

def test_resolve_symbol_to_file_exact_match():
    store = build_sample_store()
    resolved = resolve_symbol_to_file(store, "OverdueFineContext")
    assert resolved == "src/main/java/com/example/library/pattern/strategy/OverdueFineContext.java"


def test_resolve_symbol_to_file_case_insensitive():
    store = build_sample_store()
    resolved = resolve_symbol_to_file(store, "overduefinecontext")
    assert resolved == "src/main/java/com/example/library/pattern/strategy/OverdueFineContext.java"


def test_resolve_symbol_to_file_unknown_returns_none():
    store = build_sample_store()
    assert resolve_symbol_to_file(store, "NoSuchClass") is None


# ---------------------------------------------------------------------------
# Routing — symbol-anchored questions
# ---------------------------------------------------------------------------

def test_routes_who_calls_to_callers_of():
    store = build_sample_store()
    result = route_query("Who calls OverdueFineContext?", store)
    assert result is not None
    assert result.operation == "callers_of"
    assert result.files == ["src/main/java/com/example/library/service/LoanController.java"]


def test_routes_what_does_x_call_to_callees_of():
    store = build_sample_store()
    result = route_query("What does OverdueFineContext call?", store)
    assert result is not None
    assert result.operation == "callees_of"
    assert result.files == [
        "src/main/java/com/example/library/pattern/strategy/StandardFineStrategy.java"
    ]


def test_routes_what_implements_to_implementations_of():
    store = build_sample_store()
    result = route_query("What implements FineCalculationStrategy?", store)
    assert result is not None
    assert result.operation == "implementations_of"
    assert set(result.files) == {
        "src/main/java/com/example/library/pattern/strategy/StandardFineStrategy.java",
        "src/main/java/com/example/library/pattern/strategy/PremiumFineStrategy.java",
    }


def test_routes_who_implements_case_insensitive():
    store = build_sample_store()
    result = route_query("who implements finecalculationstrategy", store)
    assert result is not None
    assert result.operation == "implementations_of"


def test_routes_path_between():
    store = build_sample_store()
    result = route_query(
        "What's the path between LoanController and FineCalculationStrategy?", store
    )
    assert result is not None
    assert result.operation == "path_between"
    assert result.files[0] == "src/main/java/com/example/library/service/LoanController.java"
    assert result.files[-1] == (
        "src/main/java/com/example/library/pattern/strategy/FineCalculationStrategy.java"
    )


# ---------------------------------------------------------------------------
# Fall-through — conceptual questions, unknown symbols, empty results
# ---------------------------------------------------------------------------

def test_conceptual_question_does_not_route():
    store = build_sample_store()
    result = route_query(
        "Where is retry logic handled for failed payments?", store
    )
    assert result is None


def test_symbol_pattern_with_unresolvable_symbol_falls_through():
    store = build_sample_store()
    result = route_query("Who calls SomeClassNotInTheGraph?", store)
    assert result is None


def test_symbol_pattern_with_no_results_falls_through():
    """PremiumFineStrategy exists but has no callers — must fall through, not
    return an empty RouteResult that would then produce zero context."""
    store = build_sample_store()
    result = route_query("Who calls PremiumFineStrategy?", store)
    assert result is None
