"""
Thin DBpedia helper that pulls **many** 1-hop and 2-hop paths
between *candidate* S and O URIs.
Cached in Redis for 24 h.
"""
from __future__ import annotations

import json, itertools
from typing import List, Tuple

from SPARQLWrapper import SPARQLWrapper, JSON
from .redis_client import rdb
from ..config      import Settings
from .models       import Edge

settings = Settings()


class KGClient:
    def __init__(self, endpoint=None) -> None:
        # REACHES OUR LOCAL DBPEDIA SPARQL ENDPOINT
        # Use the provided endpoint or default to local
        self._sparql = SPARQLWrapper(endpoint or settings.DBPEDIA_ENDPOINT_LOCAL)
        self._sparql.setReturnFormat(JSON)

        # Optional: check endpoint health
        try:
            self._sparql.setQuery("ASK { ?s ?p ?o }")
            self._sparql.query().convert()
        except Exception as e:
            raise RuntimeError(f"DBpedia endpoint not reachable: {e}")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _query(self, q: str) -> list[dict]:
        self._sparql.setQuery(q)
        return self._sparql.query().convert()["results"]["bindings"]

    def _out_edges(self, uri: str, limit: int) -> List[Edge]:
        rows = self._query(f"SELECT ?p ?o WHERE {{ <{uri}> ?p ?o }} LIMIT {limit}")
        return [Edge(uri, row["p"]["value"], row["o"]["value"]) for row in rows]

    def _in_edges(self, uri: str, limit: int) -> List[Edge]:
        rows = self._query(f"SELECT ?s ?p WHERE {{ ?s ?p <{uri}> }} LIMIT {limit}")
        return [Edge(row["s"]["value"], row["p"]["value"], uri) for row in rows]

    def _two_hop(self, s: str, o: str, limit: int) -> List[List[Edge]]:
        q = f"""
        SELECT ?p1 ?x ?p2 WHERE {{
          <{s}> ?p1 ?x .
          ?x    ?p2 <{o}> .
        }} LIMIT {limit}
        """
        rows = self._query(q)
        paths: List[List[Edge]] = []
        for r in rows:
            mid = r["x"]["value"]
            paths.append([
                Edge(s,      r["p1"]["value"], mid),
                Edge(mid,    r["p2"]["value"], o),
            ])
        return paths

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fetch_paths(
        self,
        s_uris: List[str],
        o_uris: List[str],
        *,
        limit_per_edge: int = 500,
        max_hops: int = 2,
        fallback: bool = True,  # Add fallback as a parameter if not present
    ) -> List[List[Edge]]:
        """
        For every candidate (S, O) URI pair return 1-hop and 2-hop paths.
        """
        key = f"paths::{hash(tuple(s_uris))}:{hash(tuple(o_uris))}:{max_hops}"
        if hit := rdb.get(key):
            return [[Edge(**e) for e in path] for path in json.loads(hit)]

        paths: List[List[Edge]] = []

       # Fallback: If no paths and fallback is enabled, try public endpoint
        if not paths and fallback and self._sparql.endpoint != settings.DBPEDIA_ENDPOINT_PUBLIC:
            try:
                public_client = KGClient(endpoint=settings.DBPEDIA_ENDPOINT_PUBLIC)
                return public_client.fetch_paths(
                    s_uris, o_uris, limit_per_edge=limit_per_edge, max_hops=max_hops, fallback=False
                )
            except Exception as e:
                raise RuntimeError(f"Fallback to public DBpedia endpoint failed: {e}")

            
        # All outgoing edges from S + incoming edges to O
        for s in s_uris:
            paths.extend([[e] for e in self._out_edges(s, limit_per_edge)])
        for o in o_uris:
            paths.extend([[e] for e in self._in_edges(o, limit_per_edge)])

        # Two-hop S → X → O
        if max_hops >= 2:
            for s, o in itertools.product(s_uris, o_uris):
                paths.extend(self._two_hop(s, o, limit_per_edge))

        # Cache
        rdb.setex(key, 86_400,
                  json.dumps([[e.__dict__ for e in p] for p in paths]))
        return paths
