"""
GraphStore — unit tests, focused on the MultiDiGraph fix.

Regression coverage for a bug found while writing test_12_graph_queries.py:
GraphStore originally used nx.DiGraph, which allows only one edge per
(source, target) pair. Two files are frequently related by more than one
edge type (e.g. a file that both IMPORTS and INVOKES the same class), and
the second add_edge() call was silently overwriting the first edge's
edge_type instead of adding a parallel edge — corrupting query results
for any file pair with more than one relationship, with no error or
warning. Fixed by switching to nx.MultiDiGraph.

A second bug fixed alongside it: remove_file() only stripped edges where
the given path was the *source*, leaving incoming edges (from files that
still reference the removed/changed file) stale in the graph after every
incremental update.
"""

import json
import os

from graph.edge import Edge, EdgeType
from graph.graph_store import GraphStore, GraphLoadError


def test_parallel_edge_types_between_same_pair_are_not_collapsed():
    """Two distinct relationships between the same file pair (e.g. IMPORTS
    and INVOKES) must both survive — a plain DiGraph would silently drop one."""
    store = GraphStore()
    store.add_edges([
        Edge("LoanController.java", "OverdueFineContext.java", EdgeType.IMPORTS),
        Edge("LoanController.java", "OverdueFineContext.java", EdgeType.INVOKES),
    ])

    edge_data = store._g.get_edge_data("LoanController.java", "OverdueFineContext.java")
    edge_types = {d["edge_type"] for d in edge_data.values()}

    assert edge_types == {"IMPORTS", "INVOKES"}
    assert store.edge_count == 2


def test_remove_file_strips_incoming_edges_too():
    """Removing a file must drop edges where it's the target, not just the source."""
    store = GraphStore()
    store.add_edges([
        Edge("LoanController.java", "OverdueFineContext.java", EdgeType.INVOKES),
        Edge("OverdueFineContext.java", "FineCalculationStrategy.java", EdgeType.REFERENCES),
    ])

    store.remove_file("OverdueFineContext.java")

    assert "OverdueFineContext.java" not in store._g
    assert store.edge_count == 0  # both the incoming and outgoing edge are gone


def test_remove_file_on_absent_file_is_a_no_op():
    store = GraphStore()
    store.add_edges([Edge("A.java", "B.java", EdgeType.IMPORTS)])
    store.remove_file("DoesNotExist.java")  # must not raise
    assert store.edge_count == 1


def test_save_load_roundtrip_preserves_parallel_edges(tmp_path):
    store = GraphStore()
    store.add_edges([
        Edge("A.java", "B.java", EdgeType.IMPORTS),
        Edge("A.java", "B.java", EdgeType.INVOKES),
        Edge("B.java", "C.java", EdgeType.INHERITS),
    ])

    path = str(tmp_path / "graph.json")
    store.save(path)

    loaded = GraphStore.load(path)
    assert loaded.node_count == store.node_count
    assert loaded.edge_count == store.edge_count == 3

    edge_types = {d["edge_type"] for d in loaded._g.get_edge_data("A.java", "B.java").values()}
    assert edge_types == {"IMPORTS", "INVOKES"}


def test_load_missing_file_raises_graph_load_error(tmp_path):
    missing = str(tmp_path / "nope.json")
    try:
        GraphStore.load(missing)
        assert False, "expected GraphLoadError"
    except GraphLoadError:
        pass


def test_load_corrupt_json_raises_graph_load_error(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("not valid json {{{{")
    try:
        GraphStore.load(str(path))
        assert False, "expected GraphLoadError"
    except GraphLoadError:
        pass
