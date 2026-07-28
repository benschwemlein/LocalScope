"""
Symbol-anchored query routing.

Detects "who calls X" / "what implements Y" style questions and dispatches
straight to a precanned graph query (graph/queries.py) instead of vector
search — no text2cypher, no LLM in the retrieval path. Falls through
(returns None) for anything that doesn't match a known pattern, or whose
symbol can't be unambiguously resolved to a file in the graph, so the
caller can fall back to the conceptual (vector) retrieval path.
"""

import os
import re

from graph import queries as graph_queries
from graph.graph_store import GraphStore

_SYMBOL = r"([A-Za-z_][A-Za-z0-9_]*)"

# (compiled pattern, operation name) — path_between is handled separately
# since it needs two symbol groups instead of one.
_PATH_BETWEEN = re.compile(
    rf"\bpath\s+(?:between|from)\s+{_SYMBOL}\s+(?:and|to)\s+{_SYMBOL}\b", re.I
)

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\bwho\s+(?:calls|constructs|creates|instantiates)\s+{_SYMBOL}\b", re.I), "callers_of"),
    (re.compile(rf"\bwhat\s+does\s+{_SYMBOL}\s+(?:call|construct|create|instantiate)\b", re.I), "callees_of"),
    (re.compile(rf"\b(?:who|what)\s+(?:implements?|extends?)\s+{_SYMBOL}\b", re.I), "implementations_of"),
    (re.compile(rf"\b(?:who|what)\s+references?\s+{_SYMBOL}\b", re.I), "references_to"),
    (re.compile(rf"\b(?:who|what)\s+imports?\s+{_SYMBOL}\b", re.I), "importers_of"),
]

_OPS = {
    "callers_of": graph_queries.callers_of,
    "callees_of": graph_queries.callees_of,
    "implementations_of": graph_queries.implementations_of,
    "references_to": graph_queries.references_to,
    "importers_of": graph_queries.importers_of,
}


def resolve_symbol_to_file(store: GraphStore, symbol: str) -> str | None:
    """
    Resolve a bare identifier (e.g. "InvoiceProcessor") to a file node in the
    graph (e.g. "src/.../InvoiceProcessor.java"). Case-insensitive match on
    the file's basename without extension. Returns None if zero or more than
    one file matches — an ambiguous symbol isn't safe to route on.
    """
    target = symbol.lower()
    matches = [
        node for node in store._g.nodes
        if os.path.splitext(os.path.basename(node))[0].lower() == target
    ]
    return matches[0] if len(matches) == 1 else None


class RouteResult:
    def __init__(self, operation: str, files: list[str]):
        self.operation = operation
        self.files = files  # result file paths from the graph query, in query order

    def __repr__(self) -> str:
        return f"RouteResult(operation={self.operation!r}, files={self.files!r})"


def route_query(query_text: str, store: GraphStore) -> RouteResult | None:
    """
    Try to match query_text against known symbol-anchored patterns. On a
    match, resolve the symbol(s) to file(s) in the graph and run the
    corresponding precanned query. Returns None if nothing matches, a symbol
    can't be resolved, or the graph query itself comes back empty — any of
    which means the caller should fall back to vector search instead.
    """
    m = _PATH_BETWEEN.search(query_text)
    if m:
        file_a = resolve_symbol_to_file(store, m.group(1))
        file_b = resolve_symbol_to_file(store, m.group(2))
        if file_a and file_b:
            path = graph_queries.path_between(store, file_a, file_b)
            if path:
                return RouteResult("path_between", path)

    for pattern, op_name in _PATTERNS:
        m = pattern.search(query_text)
        if not m:
            continue
        file_path = resolve_symbol_to_file(store, m.group(1))
        if not file_path:
            continue
        result_files = _OPS[op_name](store, file_path)
        if result_files:
            return RouteResult(op_name, result_files)

    return None
