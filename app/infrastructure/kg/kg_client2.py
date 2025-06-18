from SPARQLWrapper import SPARQLWrapper, JSON
from typing import List, Union, Tuple
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
    "http://dbpedia.org/property/value",
    "http://dbpedia.org/property/color",
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
    def __init__(self, endpoint="https://dbpedia.org/sparql", timeout=30, page_size=1000, degree_threshold=20000):
        self.sparql = SPARQLWrapper(endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(timeout)
        self.page_size = page_size
        self.degree_threshold = degree_threshold

        allow = " || ".join(f"STRSTARTS(STR(?p),'{u}')" for u in ALLOWED_PREFIXES)
        deny = " || ".join(f"STRSTARTS(STR(?p),'{u}')" for u in BLACKLIST_PREDICATES)
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

    def _count_edges(self, uri: str) -> int:
        q = f"""
        SELECT (COUNT(?p) AS ?count) WHERE {{
          {{ <{uri}> ?p ?o }} UNION {{ ?s ?p <{uri}> }}
        }}
        """
        try:
            result = self._page(q)
            return int(result[0]['count']['value']) if result else 0
        except Exception:
            return float('inf')

    def fetch_paths(self, uris: Union[str, List[str]], *, max_hops: int = 1) -> List[List[Edge]]:
        """
        Fetch 1‐hop in/out edges from DBpedia for each URI in `uris`.
        You can pass a single URI or a list of URIs; output is always
        a flat List[List[Edge]] where each inner list is a single‐edge path.
        """
        if isinstance(uris, str):
            uris = [uris]

        uri_set = set(uris)

        paths: List[List[Edge]] = []
        for uri in uris:
            deg = self._count_edges(uri)
            print("uri", uri)
            print("deg", deg)
            is_high_degree = deg > self.degree_threshold
            if is_high_degree:
                for other_uri in uri_set - {uri}:
                    q_out = f"""{PREFIXES}
                            SELECT ?p WHERE {{ <{uri}> ?p <{other_uri}> . {self.predicate_filter} }}"""
                    for row in self._page(q_out):
                        paths.append([Edge(uri, row['p']['value'], other_uri, "dbpedia")])

                    q_in = f"""{PREFIXES}
                            SELECT ?p WHERE {{ <{other_uri}> ?p <{uri}> . {self.predicate_filter} }}"""
                    for row in self._page(q_in):
                        paths.append([Edge(other_uri, row['p']['value'], uri, "dbpedia")])

                # keep literals and English abstract
                q_literals = f"""{PREFIXES}
                            SELECT ?p ?o WHERE {{
                              <{uri}> ?p ?o .
                              {self.predicate_filter}
                              FILTER(isLiteral(?o))
                              FILTER(!( ?p = dbo:abstract && !langMatches(lang(?o), 'en') ))
                            }}"""
                for row in self._page(q_literals):
                    paths.append([Edge(uri, row['p']['value'], row['o']['value'], "dbpedia")])
            else:

                # outgoing
                outgoing_q = f"""{PREFIXES}
                            SELECT ?p ?o WHERE {{
                              <{uri}> ?p ?o .
                              {self.predicate_filter}
                              {self.object_filter}
                            }}"""
                for row in self._page(outgoing_q):
                    paths.append([Edge(uri, row["p"]["value"], row["o"]["value"], "dbpedia")])

                # incoming
                incoming_q = f"""{PREFIXES}
                            SELECT ?s ?p WHERE {{
                              ?s ?p <{uri}> .
                              {self.predicate_filter}
                            }}"""
                for row in self._page(incoming_q):
                    paths.append([Edge(row["s"]["value"], row["p"]["value"], uri, "dbpedia")])

        return paths


# testing
"""
def pretty_print_ranked_paths(ranked_paths: List[Tuple[List[Edge]]]) -> None:
    def _short(uri: str) -> str:
        return uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]

    print(f"\nTop {len(ranked_paths)} ranked paths:")
    for i, (path) in enumerate(ranked_paths, 1):
        # flatten subject → predicate → (mid → predicate →) object
        elems = []
        for edge in path:
            elems.append(_short(edge.subject))
            elems.append(_short(edge.predicate))
        elems.append(_short(path[-1].object))

        line = " → ".join(elems)
        print(f"[{i:2d}] {line}")

import time
t1=time.time()
kg=KGClient2(degree_threshold=300)
p=kg.fetch_paths(["http://dbpedia.org/resource/Cambridge_Chronicle", "http://dbpedia.org/resource/Bart_Selman", "http://dbpedia.org/resource/United_States"])

print(time.time()-t1)
pretty_print_ranked_paths(p)
"""
