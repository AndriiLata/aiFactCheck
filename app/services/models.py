from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str


@dataclass
class Edge:
    subject: str
    predicate: str
    object: str


@dataclass
class Evidence:
    """A ranked path of one or two edges that links claim S and O."""
    path: List[Edge]
    score: float
