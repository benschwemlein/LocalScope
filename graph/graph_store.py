import json
import math

import networkx as nx

from graph.edge import Edge, EdgeType


class GraphLoadError(Exception):
    pass


class GraphStore:
    def __init__(self) -> None:
        # MultiDiGraph, not DiGraph: two files are frequently related by more
        # than one edge type (e.g. a file both IMPORTS and INVOKES the same
        # class). A plain DiGraph allows only one edge per (source, target)
        # pair, so a second add_edge() silently overwrites the first edge's
        # edge_type instead of adding a parallel edge.
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()

    def add_edges(self, edges: list[Edge]) -> None:
        for e in edges:
            self._g.add_edge(
                e.source,
                e.target,
                edge_type=e.edge_type.value,
                weight=e.weight,
            )

    def remove_file(self, path: str) -> None:
        """Remove path and every edge touching it, incoming or outgoing."""
        if path in self._g:
            self._g.remove_node(path)

    def remove_edges_of_type(self, edge_type: EdgeType) -> int:
        """Drop every edge of one type, leaving nodes and other edges intact.

        Needed by whole-repo passes that must be recomputed rather than
        updated incrementally: clear what the last pass produced, then re-add.
        """
        doomed = [
            (u, v, k) for u, v, k, data in self._g.edges(keys=True, data=True)
            if data.get("edge_type") == edge_type.value
        ]
        self._g.remove_edges_from(doomed)
        return len(doomed)

    def shortest_path_length(self, source: str, target: str) -> float:
        try:
            lengths = nx.single_source_dijkstra_path_length(self._g, source, cutoff=None, weight=None)
            return float(lengths.get(target, math.inf))
        except nx.NodeNotFound:
            return math.inf

    def save(self, path: str) -> None:
        data = nx.node_link_data(self._g, edges="edges")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    @classmethod
    def load(cls, path: str) -> "GraphStore":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            g = nx.node_link_graph(data, edges="edges")
        except Exception as exc:
            raise GraphLoadError(f"Failed to load graph from {path}: {exc}") from exc
        store = cls()
        store._g = g
        return store

    @property
    def node_count(self) -> int:
        return self._g.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._g.number_of_edges()
