from SPARQLWrapper import SPARQLWrapper, JSON
from typing import List, Union
from app.models import Edge

# Allowed predicate namespaces and specific predicates to drop
ALLOWED_PREFIXES = [
    "http://dbpedia.org/ontology/",
    "http://dbpedia.org/property/",
    "http://www.w3.org/2000/01/rdf-schema#",
]
BLACKLIST_PREDICATES = [
    "http://dbpedia.org/ontology/wikiPageWikiLink",
    "http://dbpedia.org/ontology/wikiPageExternalLink",
    "http://dbpedia.org/ontology/wikiPageRevisionID",
    "http://dbpedia.org/ontology/wikiPageLength",
    "http://dbpedia.org/ontology/wikiPageID",
    "http://dbpedia.org/ontology/wikiPageRedirects",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "http://dbpedia.org/property/wikiPageUsesTemplate",
    "http://dbpedia.org/ontology/thumbnail",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://dbpedia.org/property/image",
    "http://dbpedia.org/property/imageCaption",
    "http://dbpedia.org/property/imageWidth",
    "http://dbpedia.org/property/note",
    "http://dbpedia.org/property/caption",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://dbpedia.org/property/reason",
    "http://dbpedia.org/property/date",
    "http://www.w3.org/2000/01/rdf-schema#seeAlso",
    "http://dbpedia.org/property/name",
    "http://dbpedia.org/ontology/wikiPageDisambiguates"
]

# Common PREFIX declarations
PREFIXES = """\
PREFIX dbo:  <http://dbpedia.org/ontology/>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

class KGClient2:
    def __init__(self, endpoint="https://dbpedia.org/sparql", timeout=30, page_size=1000):
        self.sparql = SPARQLWrapper(endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(timeout)
        self.page_size = page_size

        allow = " || ".join(f"STRSTARTS(STR(?p),'{u}')" for u in ALLOWED_PREFIXES)
        deny  = " || ".join(f"STRSTARTS(STR(?p),'{u}')" for u in BLACKLIST_PREDICATES)
        self.predicate_filter = f"FILTER(({allow}) && !({deny}))"
        # Only drop non-English abstracts; keep all other triples
        self.object_filter = "FILTER(!( ?p = dbo:abstract && !langMatches(lang(?o),'en') ))"

    def _page(self, query: str) -> List[dict]:
        rows, offset = [], 0
        while True:
            q = f"{query}\nLIMIT {self.page_size} OFFSET {offset}"
            self.sparql.setQuery(q)
            batch = self.sparql.queryAndConvert()["results"]["bindings"]
            if not batch:
                break
            rows.extend(batch)
            offset += self.page_size
        return rows

    def fetch_paths(
            self,
            uris: Union[str, List[str]],
            *,
            max_hops: int = 1  # we’re only doing single‐hop here
    ) -> List[List[Edge]]:
        """
        Fetch 1‐hop in/out edges from DBpedia for each URI in `uris`.
        You can pass a single URI or a list of URIs; output is always
        a flat List[List[Edge]] where each inner list is a single‐edge path.
        """
        if isinstance(uris, str):
            uris = [uris]

        paths: List[List[Edge]] = []
        for uri in uris:
            # outgoing
            outgoing_q = f"""{PREFIXES}
SELECT ?p ?o WHERE {{
  <{uri}> ?p ?o .
  {self.predicate_filter}
  {self.object_filter}
}}"""
            for row in self._page(outgoing_q):
                e = Edge(
                    subject=uri,
                    predicate=row["p"]["value"],
                    object=row["o"]["value"],
                    source_kg="dbpedia",
                )
                paths.append([e])

            # incoming
            incoming_q = f"""{PREFIXES}
SELECT ?s ?p WHERE {{
  ?s ?p <{uri}> .
  {self.predicate_filter}
}}"""
            for row in self._page(incoming_q):
                e = Edge(
                    subject=row["s"]["value"],
                    predicate=row["p"]["value"],
                    object=uri,
                    source_kg="dbpedia",
                )
                paths.append([e])

        return paths
