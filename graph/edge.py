from dataclasses import dataclass, field
from enum import Enum


class EdgeType(Enum):
    IMPORTS = "IMPORTS"
    INVOKES = "INVOKES"
    INHERITS = "INHERITS"
    REFERENCES = "REFERENCES"
    CONTAINS = "CONTAINS"
    # Client HTTP call -> server route handler. The only edge type that
    # crosses a language boundary, and the only one derived from matching
    # string routes rather than resolved symbols.
    CALLS_ENDPOINT = "CALLS_ENDPOINT"


@dataclass
class Edge:
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
