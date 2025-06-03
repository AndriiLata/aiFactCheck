from __future__ import annotations

import json
import itertools
from typing import List, Tuple, Dict, Any, Optional # Added Optional

from SPARQLWrapper import SPARQLWrapper, JSON
from .redis_client import rdb  # Assuming rdb is a working Redis client instance
from ..config import Settings
from .models import Edge

settings = Settings()


class WikidataQueryClient:
    """
    Handles SPARQL queries to Wikidata.
    """
    WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
    # IMPORTANT: Replace with your actual contact information for responsible User-Agent string
    USER_AGENT = "FactVerificationPipeline/0.2 (Wikidata Client; https://your-project-website.com; your-contact-email@example.com)"

    def __init__(self, timeout_s: int = 20): # Increased default timeout for Wikidata
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
        query = f"""
        SELECT ?p ?o WHERE {{
          <{entity_uri}> ?p ?o .
          # Consider filtering ?p to be a wikibase:Property if needed, though often not strictly necessary
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
            for row in rows if "p" in row and "o" in row # Basic check for required fields
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
            for row in rows if "s" in row and "p" in row # Basic check
        ]

    def fetch_two_hop_paths(self, s_uri: str, o_uri: str, limit: int) -> List[List[Edge]]:
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
            if "p1" in r and "x" in r and "p2" in r: # Ensure all parts of the path are present
                mid_node = r["x"]["value"]
                paths.append([
                    Edge(s_uri, r["p1"]["value"], mid_node, source_kg="wikidata"),
                    Edge(mid_node, r["p2"]["value"], o_uri, source_kg="wikidata"),
                ])
        return paths


class KGClient:
    # IMPORTANT: Replace with your actual contact information
    DBPEDIA_USER_AGENT = "FactVerificationPipeline/0.2 (DBpedia Client; https://your-project-website.com; your-contact-email@example.com)"

    def __init__(self, dbp_endpoint: Optional[str] = None, dbp_timeout_s: int = 15, wd_timeout_s: int = 20) -> None:
        # Determine the DBpedia endpoint to use
        # Priority: 1. dbp_endpoint param, 2. settings.DBPEDIA_ENDPOINT, 3. settings.DBPEDIA_ENDPOINT_LOCAL (if exists)
        self.initial_dbp_endpoint = dbp_endpoint
        if self.initial_dbp_endpoint is None:
            self.initial_dbp_endpoint = getattr(settings, 'DBPEDIA_ENDPOINT', None)
        if self.initial_dbp_endpoint is None: # Fallback to local if primary default not set
             self.initial_dbp_endpoint = getattr(settings, 'DBPEDIA_ENDPOINT_LOCAL', 'http://localhost:8890/sparql') # Default local

        self._dbp_sparql = SPARQLWrapper(self.initial_dbp_endpoint)
        self._dbp_sparql.setReturnFormat(JSON)
        self._dbp_sparql.setTimeout(dbp_timeout_s)
        self._dbp_sparql.addCustomHttpHeader("User-Agent", self.DBPEDIA_USER_AGENT)
        self._dbp_sparql.addCustomHttpHeader("Accept", "application/sparql-results+json")
        
        self._active_dbp_endpoint = self.initial_dbp_endpoint # Tracks currently used DBP endpoint

        try:
            current_sparql_instance = self._dbp_sparql
            current_sparql_instance.setQuery("ASK { ?s ?p ?o }")
            current_sparql_instance.queryAndConvert()
            print(f"[KGClient] DBpedia endpoint at {self._active_dbp_endpoint} is reachable.")
        except Exception as e:
            print(f"[KGClient] Warning: DBpedia endpoint at {self._active_dbp_endpoint} not initially reachable: {e}")

        self._wd_client = WikidataQueryClient(timeout_s=wd_timeout_s)

    def _get_active_dbp_sparql(self) -> SPARQLWrapper:
        """Returns the currently active DBpedia SPARQLWrapper instance."""
        # This method exists in case _dbp_sparql is temporarily changed during fallback
        return self._dbp_sparql

    def _dbp_query(self, q: str) -> list[dict]:
        sparql_instance = self._get_active_dbp_sparql()
        try:
            sparql_instance.setQuery(q)
            results = sparql_instance.queryAndConvert()
            return results.get("results", {}).get("bindings", [])
        except Exception as e:
            print(f"[KGClient-DBpedia] SPARQL query to {self._active_dbp_endpoint} failed: {e}\nQuery:\n{q}")
            return []

    def _dbp_out_edges(self, uri: str, limit: int) -> List[Edge]:
        rows = self._dbp_query(f"SELECT ?p ?o WHERE {{ <{uri}> ?p ?o }} LIMIT {limit}")
        return [Edge(uri, row["p"]["value"], row["o"]["value"], source_kg="dbpedia") for row in rows if "p" in row and "o" in row]

    def _dbp_in_edges(self, uri: str, limit: int) -> List[Edge]:
        rows = self._dbp_query(f"SELECT ?s ?p WHERE {{ ?s ?p <{uri}> }} LIMIT {limit}")
        return [Edge(row["s"]["value"], row["p"]["value"], uri, source_kg="dbpedia") for row in rows if "s" in row and "p" in row]

    def _dbp_two_hop(self, s: str, o: str, limit: int) -> List[List[Edge]]:
        q = f"""
        SELECT ?p1 ?x ?p2 WHERE {{
          <{s}> ?p1 ?x .
          ?x    ?p2 <{o}> .
          FILTER (!isBlank(?x))
          FILTER (STRSTARTS(STR(?x), "http://dbpedia.org/"))
        }} LIMIT {limit}
        """
        rows = self._dbp_query(q)
        paths: List[List[Edge]] = []
        for r in rows:
            if "p1" in r and "x" in r and "p2" in r:
                mid = r["x"]["value"]
                paths.append([
                    Edge(s, r["p1"]["value"], mid, source_kg="dbpedia"),
                    Edge(mid, r["p2"]["value"], o, source_kg="dbpedia"),
                ])
        return paths

    def _fetch_dbpedia_paths_internal(self, s_dbp_uris: List[str], o_dbp_uris: List[str], 
                                    limit_per_edge_query: int, limit_per_two_hop_query: int, 
                                    max_hops: int) -> List[List[Edge]]:
        """Helper to fetch DBpedia paths using the currently configured _dbp_sparql."""
        internal_dbp_paths = []
        for s_uri in s_dbp_uris:
            internal_dbp_paths.extend([[e] for e in self._dbp_out_edges(s_uri, limit_per_edge_query)])
        for o_uri in o_dbp_uris:
            internal_dbp_paths.extend([[e] for e in self._dbp_in_edges(o_uri, limit_per_edge_query)])
        if max_hops >= 2:
            for s, o in itertools.product(s_dbp_uris, o_dbp_uris):
                if s == o: continue
                internal_dbp_paths.extend(self._dbp_two_hop(s, o, limit_per_two_hop_query))
        return internal_dbp_paths

    def fetch_paths(
        self,
        s_dbp_uris: List[str],
        s_wd_uris: List[str],
        o_dbp_uris: List[str],
        o_wd_uris: List[str],
        *,
        limit_per_edge_query: int = 100,
        limit_per_two_hop_query: int = 50,
        max_hops: int = 2,
        try_public_dbp_fallback: bool = True,
    ) -> List[List[Edge]]:
        
        s_dbp_uris_tuple = tuple(sorted(s_dbp_uris))
        s_wd_uris_tuple = tuple(sorted(s_wd_uris))
        o_dbp_uris_tuple = tuple(sorted(o_dbp_uris))
        o_wd_uris_tuple = tuple(sorted(o_wd_uris))
        
        # Cache key includes the initial DBPedia endpoint to differentiate results if fallback occurs.
        key_parts = [
            "pathsV3.1", # Versioned key
            str(hash(s_dbp_uris_tuple)), str(hash(s_wd_uris_tuple)),
            str(hash(o_dbp_uris_tuple)), str(hash(o_wd_uris_tuple)),
            str(max_hops), str(limit_per_edge_query), str(limit_per_two_hop_query),
            self.initial_dbp_endpoint # Include initial DBP endpoint in key
        ]
        cache_key = "::".join(key_parts)

        if hit := rdb.get(cache_key):
            try:
                loaded_paths_data = json.loads(hit)
                paths_from_cache: List[List[Edge]] = [[Edge(**edge_dict) for edge_dict in p_list] for p_list in loaded_paths_data]
                print(f"[KGClient] Cache hit for key: {cache_key}, loaded {len(paths_from_cache)} paths.")
                return paths_from_cache
            except (json.JSONDecodeError, TypeError) as e:
                print(f"[KGClient] Cache data issue for key {cache_key}: {e}")

        all_paths: List[List[Edge]] = []
        
        # --- DBpedia Path Fetching ---
        self._active_dbp_endpoint = self.initial_dbp_endpoint # Ensure active endpoint is set
        dbp_paths = self._fetch_dbpedia_paths_internal(s_dbp_uris, o_dbp_uris, limit_per_edge_query, limit_per_two_hop_query, max_hops)
        print(f"[KGClient] Found {len(dbp_paths)} DBpedia paths from endpoint: {self._active_dbp_endpoint}.")

        public_dbp_url = getattr(settings, 'DBPEDIA_ENDPOINT_PUBLIC', None)
        if not dbp_paths and try_public_dbp_fallback and public_dbp_url and self.initial_dbp_endpoint != public_dbp_url:
            print(f"[KGClient] No DBpedia paths from {self.initial_dbp_endpoint}. Attempting fallback to public DBpedia: {public_dbp_url}")
            
            original_dbp_sparql_instance = self._dbp_sparql # Save current instance
            original_active_endpoint = self._active_dbp_endpoint
            
            try:
                self._dbp_sparql = SPARQLWrapper(public_dbp_url)
                self._dbp_sparql.setReturnFormat(JSON)
                self._dbp_sparql.setTimeout(original_dbp_sparql_instance.timeout) # Use same timeout
                self._dbp_sparql.addCustomHttpHeader("User-Agent", self.DBPEDIA_USER_AGENT)
                self._dbp_sparql.addCustomHttpHeader("Accept", "application/sparql-results+json")
                self._active_dbp_endpoint = public_dbp_url

                self._dbp_sparql.setQuery("ASK { ?s ?p ?o }") # Health check
                self._dbp_sparql.queryAndConvert()
                print(f"[KGClient] Public DBpedia endpoint {public_dbp_url} is reachable.")

                dbp_paths = self._fetch_dbpedia_paths_internal(s_dbp_uris, o_dbp_uris, limit_per_edge_query, limit_per_two_hop_query, max_hops)
                print(f"[KGClient] Found {len(dbp_paths)} DBpedia paths from public endpoint: {public_dbp_url}.")
            
            except Exception as e:
                print(f"[KGClient] Fallback to public DBpedia endpoint {public_dbp_url} failed: {e}")
            finally:
                # Restore original DBpedia SPARQLWrapper instance and active endpoint
                self._dbp_sparql = original_dbp_sparql_instance
                self._active_dbp_endpoint = original_active_endpoint
        
        all_paths.extend(dbp_paths)

        # --- Wikidata Path Fetching ---
        wikidata_paths: List[List[Edge]] = []
        if s_wd_uris or o_wd_uris: # Only query if there are Wikidata URIs
            print(f"[KGClient] Fetching paths from Wikidata: S URIs:{len(s_wd_uris)}, O URIs:{len(o_wd_uris)}")
            for s_uri_wd in s_wd_uris:
                wikidata_paths.extend([[e] for e in self._wd_client.fetch_outgoing_edges(s_uri_wd, limit_per_edge_query)])
            for o_uri_wd in o_wd_uris:
                wikidata_paths.extend([[e] for e in self._wd_client.fetch_incoming_edges(o_uri_wd, limit_per_edge_query)])
            if max_hops >= 2:
                for s_wd, o_wd in itertools.product(s_wd_uris, o_wd_uris):
                    if s_wd == o_wd: continue
                    wikidata_paths.extend(self._wd_client.fetch_two_hop_paths(s_wd, o_wd, limit_per_two_hop_query))
            print(f"[KGClient] Found {len(wikidata_paths)} paths from Wikidata.")
            all_paths.extend(wikidata_paths)
        
        print(f"[KGClient] Total paths found from all sources: {len(all_paths)}")

        try:
            paths_to_cache = [[e.__dict__ for e in p] for p in all_paths]
            rdb.setex(cache_key, 86_400, json.dumps(paths_to_cache))
            print(f"[KGClient] Cached {len(all_paths)} paths with key: {cache_key}")
        except Exception as e:
            print(f"[KGClient] Failed to cache paths for key {cache_key}: {e}")
            
        return all_paths