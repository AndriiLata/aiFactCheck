"""Extract entities (+ optional DBpedia linking) with LLM first, spaCy backup, Redis-cached SPARQL for labels."""

from __future__ import annotations

import json, os, threading, concurrent.futures
from functools import lru_cache
from typing import List, Dict

from rapidfuzz import fuzz, process
from SPARQLWrapper import SPARQLWrapper, JSON

import spacy

from .openai_client import chat
from .redis_client   import rdb
from ..config        import Settings

settings = Settings()
# ---------------------------------------------------------------------------#
# 1) spaCy – fast local NER fallback
# ---------------------------------------------------------------------------#
@lru_cache(maxsize=1)
def _spacy():
    # en_core_web_sm is tiny (≈12 MB) – download once:  python -m spacy download en_core_web_sm
    return spacy.load("en_core_web_sm", exclude=["lemmatizer"])

_ALLOWED = {
    "PERSON", "ORG", "GPE", "LOC", "NORP",
    "DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY",
    "MONEY", "PERCENT",
}

def _spacy_entities(sentence: str) -> list[dict]:
    doc = _spacy()(sentence)
    return [
        {"text": ent.text, "type": ent.label_}
        for ent in doc.ents
        if ent.label_ in _ALLOWED
    ]

# ---------------------------------------------------------------------------#
# 2) LLM function-calling
# ---------------------------------------------------------------------------#
FUNCTION_SCHEMA = [
    {
        "name": "extract_entities",
        "description": (
            "Identify ALL real-world entities AND numeric facts required to verify a claim. "
            "Include personal names, organisations, places, dates, and explicit numbers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "type": {"type": "string"},
                        },
                        "required": ["text", "type"],
                    },
                }
            },
            "required": ["entities"],
        },
    }
]

def _call_llm(sentence: str) -> list[dict]:
    system = {
        "role": "system",
        "content": (
            "You are an expert named-entity recogniser. "
            "Return ONLY a tool call (no normal assistant message) that calls "
            "`extract_entities` with every entity or numeric fact present."
        ),
    }
    user = {"role": "user", "content": sentence}

    msg = chat([system, user], functions=FUNCTION_SCHEMA)

    if getattr(msg, "tool_calls", None):
        args = json.loads(msg.tool_calls[0].function.arguments)
        return args.get("entities", [])
    return []

# ---------------------------------------------------------------------------#
# 3) DBpedia label search (slow) + Redis cache 24 h
# ---------------------------------------------------------------------------#
def _run_sparql_label_search(label: str, top_k: int = 1) -> list[dict]:
    sparql = SPARQLWrapper(settings.DBPEDIA_ENDPOINT)
    safe   = label.replace('"', r"\"")
    sparql.setReturnFormat(JSON)
    sparql.setQuery(f"""
        SELECT ?uri ?lbl WHERE {{
          ?uri rdfs:label ?lbl .
          FILTER (lang(?lbl) = "en")
          FILTER (CONTAINS(LCASE(?lbl), LCASE("{safe}")))
        }} LIMIT 250
    """)
    rows = sparql.query().convert()["results"]["bindings"]

    cand = [(row["uri"]["value"], row["lbl"]["value"]) for row in rows]
    scored = process.extract(label, [c[1] for c in cand], scorer=fuzz.WRatio, limit=top_k)

    return [{"uri": cand[idx][0], "score": s / 100} for _l, s, idx in scored]

def _link_to_dbpedia(label: str, *, top_k: int = 1) -> list[dict]:
    cache_key = f"lbl:{label.lower()}"
    if hit := rdb.get(cache_key):
        return json.loads(hit)

    # first call → wait up to 2 s, else background
    with concurrent.futures.ThreadPoolExecutor() as pool:
        fut = pool.submit(_run_sparql_label_search, label, top_k)
        try:
            linked = fut.result(timeout=2)       # block max 2 s
        except concurrent.futures.TimeoutError:
            linked = []
            # keep fetching in background
            threading.Thread(
                target=lambda: rdb.setex(cache_key, 86400, json.dumps(_run_sparql_label_search(label, top_k))),
                daemon=True,
            ).start()
    rdb.setex(cache_key, 86400, json.dumps(linked))
    return linked

# ---------------------------------------------------------------------------#
# 4) Public function
# ---------------------------------------------------------------------------#
def extract_linked_entities(sentence: str) -> List[Dict]:
    llm = _call_llm(sentence)
    spa = _spacy_entities(sentence)

    # union by lowercase text
    seen: set[str] = set()
    merged: list[dict] = []

    for ent in llm + spa:
        key = ent["text"].lower()
        if key in seen:
            continue
        seen.add(key)

        # skip DBpedia lookup for pure numbers
        if ent["text"].isdigit():
            merged.append(ent)
            continue

        best = _link_to_dbpedia(ent["text"])
        if best:
            ent.update(best[0])          # add uri & score
        merged.append(ent)

    return merged
