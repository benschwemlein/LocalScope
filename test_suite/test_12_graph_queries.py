"""
Precanned graph queries — unit tests.

No Ollama or ChromaDB dependency: builds a small synthetic GraphStore
directly and asserts each query's traversal semantics match what the
plugins actually extract (graph/plugins/*.py), not an idealized call graph.
"""

from graph.edge import Edge, EdgeType
from graph.graph_store import GraphStore
from graph.queries import (
    callers_of,
    callees_of,
    implementations_of,
    references_to,
    importers_of,
    path_between,
    one_hop_neighbors,
)


def build_sample_store() -> GraphStore:
    """
    A small synthetic repo graph:

      OverdueFineContext --INVOKES--> StandardFineStrategy   (constructs it)
      LoanController     --INVOKES--> OverdueFineContext
      StandardFineStrategy --INHERITS--> FineCalculationStrategy   (implements interface)
      PremiumFineStrategy  --INHERITS--> FineCalculationStrategy
      OverdueFineContext --REFERENCES--> FineCalculationStrategy   (field of that type)
      LoanController     --IMPORTS--> OverdueFineContext
      Unrelated          --IMPORTS--> SomethingElse   (disconnected component)
    """
    store = GraphStore()
    store.add_edges([
        Edge("OverdueFineContext.java", "StandardFineStrategy.java", EdgeType.INVOKES),
        Edge("LoanController.java", "OverdueFineContext.java", EdgeType.INVOKES),
        Edge("StandardFineStrategy.java", "FineCalculationStrategy.java", EdgeType.INHERITS),
        Edge("PremiumFineStrategy.java", "FineCalculationStrategy.java", EdgeType.INHERITS),
        Edge("OverdueFineContext.java", "FineCalculationStrategy.java", EdgeType.REFERENCES),
        Edge("LoanController.java", "OverdueFineContext.java", EdgeType.IMPORTS),
        Edge("Unrelated.java", "SomethingElse.java", EdgeType.IMPORTS),
    ])
    return store


def test_callers_of_returns_invokes_predecessors():
    store = build_sample_store()
    assert callers_of(store, "OverdueFineContext.java") == ["LoanController.java"]


def test_callers_of_unknown_file_returns_empty():
    store = build_sample_store()
    assert callers_of(store, "DoesNotExist.java") == []


def test_callees_of_returns_invokes_successors():
    store = build_sample_store()
    assert callees_of(store, "OverdueFineContext.java") == ["StandardFineStrategy.java"]


def test_implementations_of_returns_inherits_predecessors():
    store = build_sample_store()
    impls = set(implementations_of(store, "FineCalculationStrategy.java"))
    assert impls == {"StandardFineStrategy.java", "PremiumFineStrategy.java"}


def test_implementations_of_does_not_include_unrelated_edge_types():
    """A REFERENCES edge into the same target must not be mistaken for INHERITS."""
    store = build_sample_store()
    impls = implementations_of(store, "FineCalculationStrategy.java")
    assert "OverdueFineContext.java" not in impls  # that's a REFERENCES edge, not INHERITS


def test_references_to_returns_references_predecessors():
    store = build_sample_store()
    assert references_to(store, "FineCalculationStrategy.java") == ["OverdueFineContext.java"]


def test_importers_of_returns_imports_predecessors():
    store = build_sample_store()
    assert importers_of(store, "OverdueFineContext.java") == ["LoanController.java"]


def test_path_between_finds_undirected_multi_hop_path():
    store = build_sample_store()
    path = path_between(store, "LoanController.java", "FineCalculationStrategy.java")
    # LoanController -> OverdueFineContext -> FineCalculationStrategy (via REFERENCES)
    # or -> StandardFineStrategy -> FineCalculationStrategy; either is a valid shortest path
    assert path is not None
    assert path[0] == "LoanController.java"
    assert path[-1] == "FineCalculationStrategy.java"
    assert len(path) - 1 <= 3


def test_path_between_disconnected_components_returns_none():
    store = build_sample_store()
    assert path_between(store, "LoanController.java", "Unrelated.java") is None


def test_path_between_unknown_file_returns_none():
    store = build_sample_store()
    assert path_between(store, "LoanController.java", "DoesNotExist.java") is None


def test_path_between_respects_max_hops():
    store = build_sample_store()
    assert path_between(store, "LoanController.java", "FineCalculationStrategy.java", max_hops=1) is None


def test_one_hop_neighbors_is_undirected_and_all_edge_types():
    store = build_sample_store()
    neighbors = set(one_hop_neighbors(store, "OverdueFineContext.java"))
    assert neighbors == {
        "StandardFineStrategy.java",  # outgoing INVOKES
        "LoanController.java",        # incoming INVOKES + IMPORTS
        "FineCalculationStrategy.java",  # outgoing REFERENCES
    }
