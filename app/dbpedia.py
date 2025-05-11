# app/dbpedia.py
from urllib.parse import quote
from SPARQLWrapper import SPARQLWrapper, JSON

_ENDPOINT = "https://dbpedia.org/sparql"

def _resource_uri(entity: str) -> str:
    encoded = quote(entity.replace(" ", "_"))
    return f"<http://dbpedia.org/resource/{encoded}>"

def _fetch_page(entity: str, limit: int, offset: int) -> list[dict[str, str]]:
    uri = _resource_uri(entity)
    query = f"""
    SELECT ?s ?p ?o WHERE {{
      {{ {uri} ?p ?o .     BIND({uri} AS ?s) }}
      UNION
      {{ ?s ?p {uri} .     BIND({uri} AS ?o) }}
    }}
    LIMIT {limit} OFFSET {offset}
    """
    sparql = SPARQLWrapper(_ENDPOINT)
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)
    results = sparql.query().convert()
    return [
        {
            "subject":   row["s"]["value"],
            "predicate": row["p"]["value"],
            "object":    row["o"]["value"],
        }
        for row in results["results"]["bindings"]
    ]

def fetch_all_triples(entity: str, batch_size: int = 500) -> list[dict[str, str]]:
    """
    Fetch _all_ triples for the given entity by paging through the SPARQL endpoint.
    batch_size controls how many rows per request (tune to avoid timeouts).
    """
    all_triples = []
    offset = 0

    while True:
        page = _fetch_page(entity, limit=batch_size, offset=offset)
        if not page:
            break
        all_triples.extend(page)
        offset += batch_size

    return all_triples

def fetch_hundred_triples(entity: str) -> list[dict[str, str]]:
    """
    Fetch _at most_ 100 triples for the given entity.
    """
    return _fetch_page(entity, limit=100, offset=0)
