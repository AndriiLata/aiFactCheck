from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional


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
    source_kg: str = "dbpedia"


@dataclass
class EntityCandidate:
    surface_form: str
    dbpedia_uri: Optional[str] = None
    wikidata_uri: Optional[str] = None
    score: float = 0.0


@dataclass
class Evidence:
    path: List[Edge]
    score: float
