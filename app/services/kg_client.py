# app/services/kg_client.py
from __future__ import annotations

import json
import itertools
from typing import List, Tuple, Dict, Any # Added Dict, Any

from SPARQLWrapper import SPARQLWrapper, JSON
from .redis_client import rdb
from ..config import Settings
from .models import Edge

settings = Settings()


class WikidataQueryClient:
    """
    Handles SPARQL queries to Wikidata.
    """
    WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
    USER_AGENT = "FactVerificationPipeline/0.1 (https://example.com/contact; your-email@example.com)"


    def __init__(self, timeout_s: int = 10):
        self._sparql = SPARQLWrapper(self.WIKIDATA_SPARQL_ENDPOINT)
        self._sparql.setReturnFormat(JSON)
        self._sparql.setTimeout(timeout_s)
        self._sparql.addCustomHttpHeader("User-Agent", self.USER_AGENT)
        self._sparql.addCustomHttpHeader("Accept", "application/sparql-results+json")


    def _execute_query(self, q_str: str) -> List[Dict[str, Any]]:
        try:
            self._sparql.setQuery(q_str)
            results = self._sparql.queryAndConvert()
            return results.get("results", {}).get("bindings", [])
        except Exception as e:
            print(f"[WikidataQueryClient] SPARQL query failed: {e}\nQuery:\n{q_str}")
            return []

    def fetch_outgoing_edges(self, entity_uri: str, limit: int) -> List[Edge]:
        # Ensure entity_uri is a full URI, e.g. http://www.wikidata.org/entity/Q123
        query = f"""
        SELECT ?p ?o WHERE {{
          <{entity_uri}> ?p ?o .
          # Best practice: ensure ?p is a property. Wikidata URIs for properties often start with .../prop/direct/
          # This can also be done by checking ?p rdf:type wikibase:Property, but can be slower.
          # FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
        }} LIMIT {limit}
        """
        rows = self._execute_query(query)
        return [
            Edge(
                subject=entity_uri,
                predicate=row["p"]["value"],
                object=row["o"]["value"],
                source_kg="wikidata",
            )
            for row in rows
        ]

    def fetch_incoming_edges(self, entity_uri: str, limit: int) -> List[Edge]:
        query = f"""
        SELECT ?s ?p WHERE {{
          ?s ?p <{entity_uri}> .
          # FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
        }} LIMIT {limit}
        """
        rows = self._execute_query(query)
        return [
            Edge(
                subject=row["s"]["value"],
                predicate=row["p"]["value"],
                object=entity_uri,
                source_kg="wikidata",
            )
            for row in rows
        ]

    def fetch_two_hop_paths(self, s_uri: str, o_uri: str, limit: int) -> List[List[Edge]]:
        # Ensure s_uri and o_uri are full Wikidata entity URIs
        query = f"""
        SELECT ?p1 ?x ?p2 WHERE {{
          <{s_uri}> ?p1 ?x .
          ?x ?p2 <{o_uri}> .
          # Optional: FILTER (isIRI(?x) && STRSTARTS(STR(?x), "http://www.wikidata.org/entity/"))
        }} LIMIT {limit}
        """
        rows = self._execute_query(query)
        paths: List[List[Edge]] = []
        for r in rows:
            mid_node = r["x"]["value"]
            paths.append([
                Edge(s_uri, r["p1"]["value"], mid_node, source_kg="wikidata"),
                Edge(mid_node, r["p2"]["value"], o_uri, source_kg="wikidata"),
            ])
        return paths


class KGClient:
    DBPEDIA_USER_AGENT = "FactVerificationPipeline/0.1 (DBpedia Client; https://example.com/contact; your-email@example.com)"

    def __init__(self, dbp_timeout_s: int = 10, wd_timeout_s: int = 15) -> None:
        # DBpedia SPARQL setup
        self._dbp_sparql = SPARQLWrapper(settings.DBPEDIA_ENDPOINT)
        self._dbp_sparql.setReturnFormat(JSON)
        self._dbp_sparql.setTimeout(dbp_timeout_s)
        self._dbp_sparql.addCustomHttpHeader("User-Agent", self.DBPEDIA_USER_AGENT)
        self._dbp_sparql.addCustomHttpHeader("Accept", "application/sparql-results+json")


        try:
            self._dbp_sparql.setQuery("ASK { ?s ?p ?o }")
            self._dbp_sparql.queryAndConvert()
        except Exception as e:
            raise RuntimeError(f"DBpedia endpoint not reachable at {settings.DBPEDIA_ENDPOINT}: {e}")

        # Wikidata client setup
        self._wd_client = WikidataQueryClient(timeout_s=wd_timeout_s)

    # ------------------------------------------------------------------ #
    # DBpedia Internal helpers
    # ------------------------------------------------------------------ #
    def _dbp_query(self, q: str) -> list[dict]:
        try:
            self._dbp_sparql.setQuery(q)
            results = self._dbp_sparql.queryAndConvert()
            return results.get("results", {}).get("bindings", [])
        except Exception as e:
            print(f"[KGClient-DBpedia] SPARQL query failed: {e}\nQuery:\n{q}")
            return []


    def _dbp_out_edges(self, uri: str, limit: int) -> List[Edge]:
        rows = self._dbp_query(f"SELECT ?p ?o WHERE {{ <{uri}> ?p ?o }} LIMIT {limit}")
        return [Edge(uri, row["p"]["value"], row["o"]["value"], source_kg="dbpedia") for row in rows]

    def _dbp_in_edges(self, uri: str, limit: int) -> List[Edge]:
        rows = self._dbp_query(f"SELECT ?s ?p WHERE {{ ?s ?p <{uri}> }} LIMIT {limit}")
        return [Edge(row["s"]["value"], row["p"]["value"], uri, source_kg="dbpedia") for row in rows]

    def _dbp_two_hop(self, s: str, o: str, limit: int) -> List[List[Edge]]:
        q = f"""
        SELECT ?p1 ?x ?p2 WHERE {{
          <{s}> ?p1 ?x .
          ?x    ?p2 <{o}> .
          FILTER (!isBlank(?x)) # Avoid blank nodes as intermediates if not desired
          FILTER (STRSTARTS(STR(?x), "http://dbpedia.org/")) # Ensure intermediate is a DBpedia entity/resource
        }} LIMIT {limit}
        """
        rows = self._dbp_query(q)
        paths: List[List[Edge]] = []
        for r in rows:
            mid = r["x"]["value"]
            paths.append([
                Edge(s, r["p1"]["value"], mid, source_kg="dbpedia"),
                Edge(mid, r["p2"]["value"], o, source_kg="dbpedia"),
            ])
        return paths

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fetch_paths(
        self,
        s_dbp_uris: List[str],
        s_wd_uris: List[str],
        o_dbp_uris: List[str],
        o_wd_uris: List[str],
        *,
        limit_per_edge_query: int = 100, # Reduced default limit for individual edge queries
        limit_per_two_hop_query: int = 50, # Reduced default limit for two-hop queries
        max_hops: int = 2,
    ) -> List[List[Edge]]:
        """
        For every candidate (S, O) URI pair from DBpedia and Wikidata,
        return 1-hop and 2-hop paths.
        """
        # Enhanced cache key
        s_dbp_uris_tuple = tuple(sorted(s_dbp_uris))
        s_wd_uris_tuple = tuple(sorted(s_wd_uris))
        o_dbp_uris_tuple = tuple(sorted(o_dbp_uris))
        o_wd_uris_tuple = tuple(sorted(o_wd_uris))
        
        key_parts = [
            "pathsV2", # Versioned key
            str(hash(s_dbp_uris_tuple)), str(hash(s_wd_uris_tuple)),
            str(hash(o_dbp_uris_tuple)), str(hash(o_wd_uris_tuple)),
            str(max_hops), str(limit_per_edge_query), str(limit_per_two_hop_query)
        ]
        key = "::".join(key_parts)

        if hit := rdb.get(key):
            try:
                # Deserialization needs to correctly instantiate Edge objects
                loaded_paths = json.loads(hit)
                paths_from_cache: List[List[Edge]] = []
                for p_list in loaded_paths:
                    current_path = []
                    for edge_dict in p_list:
                        current_path.append(Edge(**edge_dict))
                    paths_from_cache.append(current_path)
                print(f"[KGClient] Cache hit for key: {key}, loaded {len(paths_from_cache)} paths.")
                return paths_from_cache
            except json.JSONDecodeError as e:
                print(f"[KGClient] Cache data corruption for key {key}: {e}")
            except TypeError as e:
                 print(f"[KGClient] Cache data structure error for key {key}: {e}")


        paths: List[List[Edge]] = []
        print(f"[KGClient] Fetching paths: DBpedia S:{len(s_dbp_uris)} O:{len(o_dbp_uris)}; Wikidata S:{len(s_wd_uris)} O:{len(o_wd_uris)}")

        # DBpedia Paths
        # 1-hop outgoing from DBpedia subjects
        for s_uri in s_dbp_uris:
            paths.extend([[e] for e in self._dbp_out_edges(s_uri, limit_per_edge_query)])
        # 1-hop incoming to DBpedia objects
        for o_uri in o_dbp_uris:
            paths.extend([[e] for e in self._dbp_in_edges(o_uri, limit_per_edge_query)])
        # 2-hop DBpedia S -> X -> O
        if max_hops >= 2:
            for s, o in itertools.product(s_dbp_uris, o_dbp_uris):
                if s == o: continue # Avoid self-loops at entity level for 2-hop
                paths.extend(self._dbp_two_hop(s, o, limit_per_two_hop_query))
        
        print(f"[KGClient] Found {len(paths)} paths from DBpedia so far.")

        # Wikidata Paths
        # 1-hop outgoing from Wikidata subjects
        for s_uri_wd in s_wd_uris:
            paths.extend([[e] for e in self._wd_client.fetch_outgoing_edges(s_uri_wd, limit_per_edge_query)])
        # 1-hop incoming to Wikidata objects
        for o_uri_wd in o_wd_uris:
            paths.extend([[e] for e in self._wd_client.fetch_incoming_edges(o_uri_wd, limit_per_edge_query)])
        # 2-hop Wikidata S -> X -> O
        if max_hops >= 2:
            for s_wd, o_wd in itertools.product(s_wd_uris, o_wd_uris):
                if s_wd == o_wd: continue
                paths.extend(self._wd_client.fetch_two_hop_paths(s_wd, o_wd, limit_per_two_hop_query))
        
        print(f"[KGClient] Total paths found from all sources: {len(paths)}")

        # Cache the combined results
        try:
            # Serialization needs Edge objects to be dicts
            paths_to_cache = [[e.__dict__ for e in p] for p in paths]
            rdb.setex(key, 86_400, json.dumps(paths_to_cache)) # 24 hours
            print(f"[KGClient] Cached {len(paths)} paths with key: {key}")
        except Exception as e: # Broad exception for caching issues
            print(f"[KGClient] Failed to cache paths for key {key}: {e}")
            
        return paths