import spacy
from refined.inference.processor import Refined
from SPARQLWrapper import SPARQLWrapper, JSON

# Initialize once
_sparql = SPARQLWrapper("https://dbpedia.org/sparql")
_sparql.setReturnFormat(JSON)
from typing import List, Optional


class EntityLinker2:

    @staticmethod
    def wikidata_to_dbpedia(qid: str) -> Optional[str]:
        """
        Given a Wikidata Q-ID, return the corresponding DBpedia resource URL
        via owl:sameAs, or None if not found.
        """
        query = f"""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    SELECT ?dbp WHERE {{
      ?dbp owl:sameAs <http://www.wikidata.org/entity/{qid}> .
      FILTER(STRSTARTS(STR(?dbp), "http://dbpedia.org/resource/"))
    }}
    LIMIT 1
    """
        _sparql.setQuery(query)
        try:
            results = _sparql.queryAndConvert()["results"]["bindings"]
            return results[0]["dbp"]["value"] if results else None
        except Exception:
            return None

    def link(self, claim: str) -> List[str]:
        refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers',
                                          entity_set="wikipedia")

        spans = refined.process_text(claim)

        # 2) fallback trigger: too few spans
        if len(spans) <= 1:
            nlp = spacy.load("en_core_web_md")
            nlp.add_pipe("entityLinker", last=True)
            doc = nlp(claim)
            urls: List[str] = []
            for ent in doc._.linkedEntities:
                raw_id = ent.get_id()  # e.g. "903257" or "Q903257"
                rid = str(raw_id)
                qid = raw_id if rid.startswith("Q") else f"Q{raw_id}"
                dbp = self.wikidata_to_dbpedia(qid)
                if dbp:
                    urls.append(dbp)
            # dedupe
            seen = set()
            return [u for u in urls if not (u in seen or seen.add(u))]

        # 3) normal branch
        results: List[str] = []
        seen = set()
        for span in spans:
            ent = span.predicted_entity
            if ent and ent.wikidata_entity_id:
                dbp = self.wikidata_to_dbpedia(ent.wikidata_entity_id)
                if dbp and dbp not in seen:
                    seen.add(dbp)
                    results.append(dbp)
                continue

        return results





