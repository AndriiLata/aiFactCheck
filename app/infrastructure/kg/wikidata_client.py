from __future__ import annotations
from typing import List, Dict, Any

from SPARQLWrapper import SPARQLWrapper, JSON

from ...models import Edge


class WikidataQueryClient:
    """
    Thin wrapper for the Wikidata public SPARQL endpoint.
    """

    _EP = "https://query.wikidata.org/sparql"
    _UA = "FactVerificationPipeline/0.2 (Wikidata Client; https://example.com)"

    def __init__(self, *, timeout_s: int = 20) -> None:
        self._sparql = SPARQLWrapper(self._EP)
        self._sparql.setReturnFormat(JSON)
        self._sparql.setTimeout(timeout_s)
        self._sparql.addCustomHttpHeader("User-Agent", self._UA)
        self._sparql.addCustomHttpHeader("Accept", "application/sparql-results+json")

    # ------------------------------------------------------------------ #
    def _q(self, q: str) -> List[Dict[str, Any]]:
        try:
            self._sparql.setQuery(q)
            return self._sparql.queryAndConvert()["results"]["bindings"]
        except Exception as exc:  # pragma: no cover – network
            print(f"[Wikidata] SPARQL failed: {exc}")
            return []

    # ------------------------------------------------------------------ #
    def fetch_outgoing_edges(self, entity: str, limit: int) -> List[Edge]:
        q = f"SELECT ?p ?o WHERE {{ <{entity}> ?p ?o }} LIMIT {limit}"
        return [
            Edge(entity, row["p"]["value"], row["o"]["value"], source_kg="wikidata")
            for row in self._q(q)
        ]

    def fetch_incoming_edges(self, entity: str, limit: int) -> List[Edge]:
        q = f"SELECT ?s ?p WHERE {{ ?s ?p <{entity}> }} LIMIT {limit}"
        return [
            Edge(row["s"]["value"], row["p"]["value"], entity, source_kg="wikidata")
            for row in self._q(q)
        ]

    def fetch_two_hop_paths(self, s_uri: str, o_uri: str, limit: int) -> List[List[Edge]]:
        q = f"""
        SELECT ?p1 ?x ?p2 WHERE {{
            <{s_uri}> ?p1 ?x .
            ?x ?p2 <{o_uri}> .
            FILTER isIRI(?x)
        }} LIMIT {limit}
        """
        paths: List[List[Edge]] = []
        for r in self._q(q):
            mid = r["x"]["value"]
            paths.append(
                [
                    Edge(s_uri, r["p1"]["value"], mid, source_kg="wikidata"),
                    Edge(mid, r["p2"]["value"], o_uri, source_kg="wikidata"),
                ]
            )
        return paths
