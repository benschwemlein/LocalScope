import math
from dataclasses import dataclass

import Levenshtein
import networkx as nx

from graph.graph_store import GraphStore


@dataclass
class SeedResult:
    file_path: str
    cos_distance: float
    content_snippet: str = ""


@dataclass
class FusionResult:
    file_path: str
    score: float
    cos_norm: float
    lev: float
    path_distance: float
    source: str  # "vector" or "graph"


def expand_and_rerank(
    seeds: list[SeedResult],
    query_text: str,
    graph_store: GraphStore,
    alpha: float = 0.3,
    beta: float = 0.6,
    max_hops: int = 3,
) -> list[FusionResult]:
    if not seeds:
        return []

    # Build cosine similarity map from seeds
    cos_norm_map: dict[str, float] = {
        s.file_path: max(0.0, min(1.0, 1.0 - s.cos_distance))
        for s in seeds
    }
    snippet_map: dict[str, str] = {s.file_path: s.content_snippet for s in seeds}

    # Undirected BFS expansion from each seed so that both IMPORTS edges
    # (A→B) and INHERITS edges (implementor→interface) are traversable in
    # either direction. This lets us find concrete implementors from a seed
    # that uses the interface (e.g. OverdueFineContext → strategy impls).
    path_dist_map: dict[str, float] = {}

    g = graph_store._g
    g_undir = g.to_undirected()
    for seed in seeds:
        if seed.file_path not in g_undir:
            path_dist_map.setdefault(seed.file_path, 0.0)
            continue
        try:
            lengths = nx.single_source_shortest_path_length(g_undir, seed.file_path, cutoff=max_hops)
        except nx.NodeNotFound:
            path_dist_map.setdefault(seed.file_path, 0.0)
            continue
        for node, dist in lengths.items():
            if node not in path_dist_map or dist < path_dist_map[node]:
                path_dist_map[node] = float(dist)

    # Seed nodes always have distance 0
    for seed in seeds:
        path_dist_map[seed.file_path] = 0.0

    # Score every candidate that has a cosine similarity (i.e., was a seed or the
    # graph-discovered node is also a seed — graph-only nodes without cos_norm are excluded)
    results: list[FusionResult] = []

    all_candidates = set(path_dist_map.keys())

    for fp in all_candidates:
        if fp not in cos_norm_map:
            # Graph-discovered file with no vector embedding — skip per contract
            continue

        cos = cos_norm_map[fp]
        snippet = snippet_map.get(fp, "")
        lev = Levenshtein.ratio(query_text[:500], snippet[:500]) if snippet else 0.0
        dist = path_dist_map.get(fp, math.inf)

        if dist == math.inf:
            continue

        score = (beta ** dist) * (alpha * cos + (1.0 - alpha) * lev)
        is_seed = fp in {s.file_path for s in seeds}
        results.append(FusionResult(
            file_path=fp,
            score=score,
            cos_norm=cos,
            lev=lev,
            path_distance=dist,
            source="vector" if is_seed else "graph",
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results
