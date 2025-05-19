from __future__ import annotations
import json
from urllib.parse import quote
from typing import List, Dict

from rapidfuzz import fuzz
from SPARQLWrapper import SPARQLWrapper, JSON

from .redis_client import rdb
from ..config import Settings

settings = Settings()
_DEF_BATCH = 500


def _resource_uri(entity: str) -> str:
    if entity.startswith(("http://", "https://")):
        return f"<{entity.strip('<>')}>"
    return f"<http://dbpedia.org/resource/{quote(entity.replace(' ', '_'))}>"


def _fetch_triples_page(entity: str, *, limit: int, offset: int) -> List[Dict]:
    uri = _resource_uri(entity)
    query = f"""
        SELECT ?s ?p ?o WHERE {{
          {{ {uri} ?p ?o . BIND({uri} AS ?s) }}
          UNION
          {{ ?s ?p {uri} . BIND({uri} AS ?o) }}
        }} LIMIT {limit} OFFSET {offset}
    """
    sparql = SPARQLWrapper(settings.DBPEDIA_ENDPOINT)
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)
    data = sparql.query().convert()
    return [
        {k: row[k]["value"] for k in ("s", "p", "o")}
        for row in data["results"]["bindings"]
    ]


def fetch_all_triples(entity: str, batch_size: int = _DEF_BATCH) -> List[Dict]:
    triples: List[Dict] = []
    for offset in range(0, 10_000, batch_size):
        page = _fetch_triples_page(entity, limit=batch_size, offset=offset)
        if not page:
            break
        triples.extend(page)
    return triples


def fetch_top_similar_entities(label: str, *, top_n: int = 10) -> List[Dict]:
    """
    Return the top_n DBpedia URIs with similarity scores for *label*.
    Cached in Redis for 24 h under key 'topn:<lowercase label>'.
    """
    cache_key = f"topn:{label.lower()}"
    if hit := rdb.get(cache_key):
        return json.loads(hit)

    sparql = SPARQLWrapper(settings.DBPEDIA_ENDPOINT)
    safe = label.replace('"', r'\"')
    sparql.setReturnFormat(JSON)
    sparql.setQuery(f'''
        SELECT ?uri ?lbl WHERE {{
          ?uri rdfs:label ?lbl .
          FILTER (lang(?lbl)="en")
          FILTER (CONTAINS(LCASE(?lbl), LCASE("{safe}")))
        }} LIMIT 500
    ''')
    rows = sparql.query().convert()["results"]["bindings"]

    scored = sorted(
        [
            (row["uri"]["value"], fuzz.WRatio(label, row["lbl"]["value"]) / 100)
            for row in rows
        ],
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    result = [{"uri": uri, "score": score} for uri, score in scored]
    rdb.setex(cache_key, 24 * 3600, json.dumps(result))
    return result
