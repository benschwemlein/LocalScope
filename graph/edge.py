from dataclasses import dataclass, field
from enum import Enum


class EdgeType(Enum):
    IMPORTS = "IMPORTS"
    INVOKES = "INVOKES"
    INHERITS = "INHERITS"
    REFERENCES = "REFERENCES"
    CONTAINS = "CONTAINS"


@dataclass
class Edge:
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
