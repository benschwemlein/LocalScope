import json
import math

import networkx as nx

from graph.edge import Edge, EdgeType


class GraphLoadError(Exception):
    pass


class GraphStore:
    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()

    def add_edges(self, edges: list[Edge]) -> None:
        for e in edges:
            self._g.add_edge(
                e.source,
                e.target,
                edge_type=e.edge_type.value,
                weight=e.weight,
            )

    def remove_file(self, path: str) -> None:
        edges_to_remove = [(u, v) for u, v in self._g.edges() if u == path]
        self._g.remove_edges_from(edges_to_remove)

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
