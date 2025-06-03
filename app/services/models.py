# app/services/models.py
from __future__ import annotations
from dataclasses import dataclass, field # Add field if using default_factory, not needed here yet
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
    source_kg: str = "dbpedia"  # Added to distinguish KG source

@dataclass
class EntityCandidate:
    """Represents a candidate entity with potential URIs from different KGs."""
    surface_form: str # Keep track of what was linked
    dbpedia_uri: Optional[str] = None
    wikidata_uri: Optional[str] = None
    score: float = 0.0

@dataclass
class Evidence:
    """A ranked path of one or two edges that links claim S and O."""
    path: List[Edge]
    score: float