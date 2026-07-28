"""
End-to-end routing integration tests against the real library-catalog-app
corpus and a real ChromaDB + graph index (Ollama required).

These exist to catch what the pure-unit router/query tests can't: whether
the wiring in query_engine.run_query actually works against a real graph
built by the tree-sitter plugins, and whether the edge types the plugins
extract carry the relationships a Spring-style Java codebase actually has.

Important finding baked into these tests: in this Spring-DI codebase,
INVOKES (object instantiation) essentially never fires for service/component
collaborators — Spring wires them via constructor injection, not `new X()`.
grep confirms the only `new StandardFineStrategy()` / `new
PremiumFineStrategy()` / `new StudentFineStrategy()` / `new
OverdueFineContext()` calls anywhere in the repo are in the JUnit test file,
not production code. REFERENCES (constructor-parameter-type usage) and
INHERITS are what actually carry the Spring dependency graph here — so
callers_of/callees_of should be expected to come back empty for
DI-managed classes, while references_to/implementations_of do the real work.
"""

import os

import pytest


@pytest.fixture(scope="session")
def graph_app(indexed_app):
    """Build the graph index on top of the shared ChromaDB session index."""
    from graph.graph_builder import build_incremental

    sample_path = indexed_app["sample_app_path"]
    index_dir = indexed_app["index_dir"]
    graph_path = os.path.join(index_dir, "graph.json")

    build_incremental(sample_path, graph_path, log_fn=lambda _: None)

    return {**indexed_app, "graph_path": graph_path}


@pytest.fixture()
def graph_enabled():
    """Temporarily flip config.GRAPH_ENABLED on, restoring it afterward."""
    import config

    original = config.GRAPH_ENABLED
    config.GRAPH_ENABLED = True
    yield
    config.GRAPH_ENABLED = original


# ---------------------------------------------------------------------------
# Symbol-anchored routing actually fires end-to-end
# ---------------------------------------------------------------------------

def test_implementations_of_routes_and_returns_all_three_strategies(graph_app, graph_enabled):
    from querying.query_engine import run_query

    logs = []
    result = run_query(
        bug_text="What implements FineCalculationStrategy?",
        index_dir=graph_app["index_dir"],
        top_k=10,
        log=logs.append,
    )

    assert any("Routed to graph query 'implementations_of'" in line for line in logs), logs
    sources = {m.get("source", "") for m in result["metas"]}
    assert any("StandardFineStrategy.java" in s for s in sources)
    assert any("PremiumFineStrategy.java" in s for s in sources)
    assert any("StudentFineStrategy.java" in s for s in sources)


def test_references_to_routes_and_finds_the_injecting_class(graph_app, graph_enabled):
    from querying.query_engine import run_query

    logs = []
    result = run_query(
        bug_text="Who references StandardFineStrategy?",
        index_dir=graph_app["index_dir"],
        top_k=10,
        log=logs.append,
    )

    assert any("Routed to graph query 'references_to'" in line for line in logs), logs
    sources = {m.get("source", "") for m in result["metas"]}
    assert any("OverdueFineContext.java" in s for s in sources)


def test_routed_query_uses_flat_scores_not_similarity_ranking(graph_app, graph_enabled):
    """Graph hits are exact structural matches, not similarity-ranked — every
    result should get the same relative score."""
    from querying.query_engine import run_query

    result = run_query(
        bug_text="What implements FineCalculationStrategy?",
        index_dir=graph_app["index_dir"],
        top_k=10,
        log=lambda _: None,
    )
    assert len(set(result["scores"])) == 1


# ---------------------------------------------------------------------------
# Known limitation: INVOKES is nearly empty for Spring-DI collaborators
# ---------------------------------------------------------------------------

def test_callers_of_di_managed_class_only_finds_the_junit_test(graph_app, graph_enabled):
    """
    OverdueFineContext is Spring-managed (@Component) and every production
    collaborator gets it via constructor injection — grep confirms zero
    `new OverdueFineContext(...)` calls in src/main. The ONE INVOKES edge
    that does exist points at OverdueFineContextTest.java, because the
    graph walks src/test too and the JUnit test constructs it directly for
    setup. This is the real, precise shape of the limitation: callers_of
    isn't "empty" here, it's actively misleading — a caller_of result for
    a DI-managed class in Spring code is likely test scaffolding, not a
    real production dependency, and callers should not assume otherwise.
    """
    from querying.query_engine import run_query

    logs = []
    result = run_query(
        bug_text="Who calls OverdueFineContext?",
        index_dir=graph_app["index_dir"],
        top_k=10,
        log=logs.append,
    )

    assert any("Routed to graph query 'callers_of'" in line for line in logs), logs
    sources = [m.get("source", "") for m in result["metas"]]
    assert sources == ["src/test/java/com/example/library/pattern/OverdueFineContextTest.java"]


# ---------------------------------------------------------------------------
# Conceptual queries: graph expansion doesn't crash, GRAPH_ENABLED=False is inert
# ---------------------------------------------------------------------------

def test_conceptual_query_with_graph_enabled_does_not_crash(graph_app, graph_enabled):
    from querying.query_engine import run_query

    result = run_query(
        bug_text=(
            "How are overdue fines calculated? How does the fine amount differ "
            "by membership tier?"
        ),
        index_dir=graph_app["index_dir"],
        top_k=10,
        log=lambda _: None,
    )
    assert result["docs"]


def test_graph_disabled_never_routes(graph_app):
    """With GRAPH_ENABLED left at its default (off), symbol-anchored
    questions must behave exactly like plain vector search — no routing."""
    import config
    from querying.query_engine import run_query

    assert config.GRAPH_ENABLED is False

    logs = []
    result = run_query(
        bug_text="What implements FineCalculationStrategy?",
        index_dir=graph_app["index_dir"],
        top_k=10,
        log=logs.append,
    )
    assert not any("Routed to graph query" in line for line in logs), logs
    assert not any("Graph expansion added" in line for line in logs), logs
    assert result["docs"]
