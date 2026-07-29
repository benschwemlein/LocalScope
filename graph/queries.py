"""
Precanned graph queries — parameterized structural retrieval operations.

No text2cypher, no LLM in the retrieval path: each operation is a fixed
traversal over the edge types the tree-sitter plugins actually extract
(graph/plugins/*.py). Symbol-anchored questions ("who calls X", "what
implements Y") route here directly instead of through vector search.

Edge semantics (see graph/plugins/*.py for the extraction logic) — read
these carefully before trusting a query name at face value:
  IMPORTS      source imports target's type
  INVOKES      source constructs an instance of target's type (`new Target()`)
               — NOT a general method-call graph. The plugins parse one file
               at a time with no whole-program call resolution, so only
               object instantiation is tracked, not arbitrary method calls.
  INHERITS     source extends/implements target (source is the subclass/
               implementor, target is the superclass/interface)
  REFERENCES   source's code mentions target's type (same-package field/
               param/variable declaration, or an Angular template
               referencing its component)

There is no CONTAINS edge in the current graph — no plugin emits one — so
there is no "contents-of" containment-hierarchy query here, unlike the
original six-query design that assumed one. importers_of (who IMPORTS this
file) fills that slot instead: same "what depends on this file" shape,
backed by data that actually exists rather than an edge type that doesn't.
"""

import networkx as nx

from graph.edge import EdgeType
from graph.graph_store import GraphStore


def _predecessors_by_type(store: GraphStore, target: str, edge_type: EdgeType) -> list[str]:
    g = store._g
    if target not in g:
        return []
    return [
        u for u, _v, data in g.in_edges(target, data=True)
        if data.get("edge_type") == edge_type.value
    ]


def _successors_by_type(store: GraphStore, source: str, edge_type: EdgeType) -> list[str]:
    g = store._g
    if source not in g:
        return []
    return [
        v for _u, v, data in g.out_edges(source, data=True)
        if data.get("edge_type") == edge_type.value
    ]


def callers_of(store: GraphStore, file_path: str) -> list[str]:
    """Files that construct an instance of file_path's class (INVOKES predecessors)."""
    return _predecessors_by_type(store, file_path, EdgeType.INVOKES)


def callees_of(store: GraphStore, file_path: str) -> list[str]:
    """Classes file_path constructs an instance of (INVOKES successors)."""
    return _successors_by_type(store, file_path, EdgeType.INVOKES)


def implementations_of(store: GraphStore, file_path: str) -> list[str]:
    """Files that extend/implement file_path (INHERITS predecessors)."""
    return _predecessors_by_type(store, file_path, EdgeType.INHERITS)


def references_to(store: GraphStore, file_path: str) -> list[str]:
    """Files whose code references file_path's type (REFERENCES predecessors)."""
    return _predecessors_by_type(store, file_path, EdgeType.REFERENCES)


def importers_of(store: GraphStore, file_path: str) -> list[str]:
    """Files that import file_path (IMPORTS predecessors)."""
    return _predecessors_by_type(store, file_path, EdgeType.IMPORTS)


def path_between(
    store: GraphStore, file_a: str, file_b: str, max_hops: int = 5
) -> list[str] | None:
    """
    Shortest path between two files, traversing any edge type undirected.
    Returns the file sequence (inclusive of both endpoints), or None if
    either file is absent from the graph or no path exists within max_hops.
    """
    g = store._g
    if file_a not in g or file_b not in g:
        return None
    g_undir = g.to_undirected()
    try:
        path = nx.shortest_path(g_undir, file_a, file_b)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    if len(path) - 1 > max_hops:
        return None
    return path


def one_hop_neighbors(store: GraphStore, file_path: str) -> list[str]:
    """All files structurally adjacent to file_path via any edge type, either direction."""
    g = store._g
    if file_path not in g:
        return []
    g_undir = g.to_undirected()
    return list(g_undir.neighbors(file_path))


def expansion_neighbors(store: GraphStore, file_path: str, max_depth: int = 3) -> list[str]:
    """Names only; see expansion_neighbors_with_depth for the ranked form."""
    return sorted(expansion_neighbors_with_depth(store, file_path, max_depth))


def expansion_neighbors_with_depth(
    store: GraphStore, file_path: str, max_depth: int = 3
) -> dict[str, int]:
    """
    Neighbors worth pulling into retrieval context: everything one hop away,
    plus anything up to max_depth away whose path crosses a CALLS_ENDPOINT edge.

    Depth is gated on edge type rather than applied uniformly because the two
    kinds of edge have very different precision. IMPORTS and REFERENCES are
    numerous and weak — a Java file imports dozens of things, most irrelevant
    to any given question — so widening past one hop mostly adds noise.
    CALLS_ENDPOINT edges are few and each is a verified route match, and they
    are the only link between the frontend and backend halves of a full-stack
    repo. Without deeper traversal across them the client and server sit in
    disconnected components and no expansion reaches from one to the other:
    evidence typically sits at the ends of a component -> api-service ->
    controller -> domain-service chain, while the REST edge joins the middle.
    """
    g = store._g
    if file_path not in g:
        return {}

    g_undir = g.to_undirected()
    rest = EdgeType.CALLS_ENDPOINT.value

    def crosses_rest(a: str, b: str) -> bool:
        data = g_undir.get_edge_data(a, b) or {}
        # MultiGraph edge data is keyed by parallel-edge index
        return any(d.get("edge_type") == rest for d in data.values())

    found: dict[str, int] = {}
    seen: set[tuple[str, bool]] = {(file_path, False)}
    frontier = [(file_path, 0, False)]

    while frontier:
        node, depth, used_rest = frontier.pop(0)
        if depth >= max_depth:
            continue
        for nb in g_undir.neighbors(node):
            nb_used = used_rest or crosses_rest(node, nb)
            nb_depth = depth + 1
            # Beyond one hop, only paths that have crossed the cross-language
            # bridge earn the extra reach.
            if nb_depth > 1 and not nb_used:
                continue
            if nb != file_path:
                # BFS order means the first arrival is the shortest path
                found.setdefault(nb, nb_depth)
            state = (nb, nb_used)
            if state not in seen:
                seen.add(state)
                frontier.append((nb, nb_depth, nb_used))

    return found
