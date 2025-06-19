from __future__ import annotations

import itertools
from typing import List

from .dbpedia_client import DBpediaClient
from .wikidata_client import WikidataQueryClient
from ...models import Edge


class KGClient:
    """
    Coordinates DBpedia + Wikidata queries and returns 1- or 2-hop paths.
    """

    def __init__(
        self,
        *,
        dbp_endpoint: str | None = None,
        dbp_timeout_s: int = 15,
        wd_timeout_s: int = 20,
    ) -> None:
        self._dbp = DBpediaClient(endpoint=dbp_endpoint, timeout_s=dbp_timeout_s)
        self._wd = WikidataQueryClient(timeout_s=wd_timeout_s)

    # ------------------------------------------------------------------ #
    def fetch_paths(
        self,
        s_dbp: List[str],
        s_wd: List[str],
        o_dbp: List[str],
        o_wd: List[str],
        *,
        limit_edge: int = 10000,
        limit_two_hop: int = 10000,
        max_hops: int = 2,
    ) -> List[List[Edge]]:
        paths: List[List[Edge]] = []

        # ---- DBpedia ---------------------------------------------------
        #print("fetching subject paths")
        for s in s_dbp:
            paths.extend([[e] for e in self._dbp.fetch_outgoing_edges(s, limit_edge)])
        #print("fetching object paths")
        for o in o_dbp:
            paths.extend([[e] for e in self._dbp.fetch_incoming_edges(o, limit_edge)])
        #print("fetching 2 hop paths")

        if max_hops >= 2:
            for s, o in itertools.product(s_dbp, o_dbp):
                if s != o:
                    paths.extend(self._dbp.fetch_two_hop_paths(s, o, limit_two_hop))
        #print("fetched DBPedia")
        # ---- Wikidata --------------------------------------------------
        for s in s_wd:
            paths.extend([[e] for e in self._wd.fetch_outgoing_edges(s, limit_edge)])
        for o in o_wd:
            paths.extend([[e] for e in self._wd.fetch_incoming_edges(o, limit_edge)])

        if max_hops >= 2:
            for s, o in itertools.product(s_wd, o_wd):
                if s != o:
                    paths.extend(self._wd.fetch_two_hop_paths(s, o, limit_two_hop))

        return paths
