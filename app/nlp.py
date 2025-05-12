"""
Entity recognition + KB linking helper
--------------------------------------
1.  Transformer-based spaCy NER (much higher recall/precision than *sm* model).
2.  Normalises each mention to (uri, label, confidence) using DBpedia Spotlight.
3.  Falls back to fuzzy SPARQL label search (RapidFuzz) when Spotlight fails.
Returned list keeps only the *best* candidate per mention.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Iterable, List, Dict

import requests
import spacy
from rapidfuzz import fuzz, process
from SPARQLWrapper import SPARQLWrapper, JSON

# NER label whitelist
_ALLOWED_TYPES: set[str] = {
    "PERSON", "ORG", "GPE", "LOC", "NORP",
    "DATE", "TIME", "QUANTITY", "PERCENT", "MONEY",
    "CARDINAL", "ORDINAL", "EVENT", "WORK_OF_ART",
    "LAW", "PRODUCT", "LANGUAGE",
}

_DBPEDIA_SPOTLIGHT = "https://api.dbpedia-spotlight.org/en/annotate"
_SPARQL_ENDPOINT = "https://dbpedia.org/sparql"


# ---------- spaCy -----------------------------------------------------------------
@lru_cache(maxsize=1)
def _nlp():
    return spacy.load("en_core_web_trf", exclude=["lemmatizer"])


# ---------- public API ------------------------------------------------------------
def extract_linked_entities(sentence: str) -> List[Dict]:
    """
    Args
    ----
    sentence : str
        Raw English sentence.
    Returns
    -------
    list of dicts
        [{'text': 'Barack Obama',
          'label': 'PERSON',
          'uri':   'http://dbpedia.org/resource/Barack_Obama',
          'score': 0.997}, ...]
    """
    doc = _nlp()(sentence)

    linked: List[Dict] = []
    for ent in doc.ents:
        if ent.label_ not in _ALLOWED_TYPES:
            continue
        best = _link_entity(ent.text)
        if best:
            best["text"] = ent.text
            best["label"] = ent.label_
            linked.append(best)

    return linked


# ---------- internal helpers ------------------------------------------------------
def _link_entity(surface: str, *, _top_k: int = 25) -> Dict | None:
    """Try Spotlight first, otherwise fuzzy SPARQL search."""
    cand = _spotlight(surface)
    if cand:
        return cand
    # fallback
    cand = _sparql_fuzzy(surface, top_k=_top_k)
    return cand[0] if cand else None


def _spotlight(surface: str) -> Dict | None:
    """DBpedia Spotlight entity linking."""
    params = {"text": surface, "confidence": 0.4}
    headers = {"Accept": "application/json"}
    try:
        r = requests.get(_DBPEDIA_SPOTLIGHT, params=params,
                         headers=headers, timeout=4)
        if r.status_code == 200:
            data = r.json()
            if "Resources" in data:
                best = max(
                    data["Resources"],
                    key=lambda row: float(row["@similarityScore"]))
                return {
                    "uri": best["@URI"],
                    "score": float(best["@similarityScore"]),
                }
    except requests.RequestException:
        pass
    return None


def _sparql_fuzzy(surface: str, *, top_k: int = 25) -> List[Dict]:
    """Fuzzy-match DBpedia rdfs:label using Levenshtein similarity."""
    sparql = SPARQLWrapper(_SPARQL_ENDPOINT)
    safe = surface.replace('"', r'\"')
    query = f'''
    SELECT ?uri ?label WHERE {{
      ?uri rdfs:label ?label .
      FILTER (lang(?label) = "en")
      FILTER (CONTAINS(LCASE(?label), LCASE("{safe}")))
    }} LIMIT 500
    '''
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)
    try:
        rows = sparql.query().convert()["results"]["bindings"]
        labels = [row["label"]["value"] for row in rows]
        scored = process.extract(
            surface, labels, scorer=fuzz.WRatio, limit=top_k)
        # scored: [(label, score, idx), ...]
        ranked = []
        for lab, score, idx in scored:
            ranked.append({
                "uri": rows[idx]["uri"]["value"],
                "score": score / 100.0,  # 0–1
            })
        return ranked
    except Exception:   # noqa: BLE001
        return []
