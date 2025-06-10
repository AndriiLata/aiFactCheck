"""
Async evidence collectors

Changes vs. previous version:
* Predicate **black-list** to drop noisy KG edges
* Triple → plain-English sentence template for NLI
"""
from __future__ import annotations

import asyncio
import re
from typing import Dict, List

from ..extraction.triple_extractor import parse_claim_to_triple
from ..linking.entity_linker import EntityLinker
from ...infrastructure.kg.kg_client import KGClient
from ..verification.web_verifier import WebVerifier
from ...models import Triple
from .trust import score_for_url

# ------------------------------------------------------------------ #
# KG filtering helpers
# ------------------------------------------------------------------ #
# predicates we IGNORE entirely (noise)
_EXCLUDE = {
    "wikiPageWikiLink",
    "seeAlso",
    "rdf-schema#seeAlso",
    "type",
    "owl#sameAs",
    "owl#differentFrom",
}

_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")


def _decamel(s: str) -> str:
    return _CAMEL_RE.sub(" ", s).replace("_", " ").strip().lower()


def _edge_to_sentence(edge) -> str:
    sub = edge.subject.split("/")[-1].replace("_", " ")
    pred = edge.predicate.split("/")[-1]
    obj = edge.object.split("/")[-1].replace("_", " ")

    if pred in _EXCLUDE:
        return ""  # will be skipped upstream

    pred_txt = _decamel(pred)
    # quick heuristic: sentences like "Donald Trump position held president of the United States."
    return f"{sub} {pred_txt} {obj}."


# ------------------------------------------------------------------ #
async def _collect_kg(
    triple: Triple, which: str, *, max_edges: int = 50
) -> List[Dict]:
    linker = EntityLinker()
    kg = KGClient()

    # ---- entity linking ---------------------------------------------
    s_cands = linker.link(triple.subject, top_k=3)
    o_cands = linker.link(triple.object, top_k=3)

    s_dbp = [c.dbpedia_uri for c in s_cands if c.dbpedia_uri]
    s_wd = [c.wikidata_uri for c in s_cands if c.wikidata_uri]
    o_dbp = [c.dbpedia_uri for c in o_cands if c.dbpedia_uri]
    o_wd = [c.wikidata_uri for c in o_cands if c.wikidata_uri]

    paths = kg.fetch_paths(
        s_dbp=s_dbp if which == "dbpedia" else [],
        s_wd=s_wd if which == "wikidata" else [],
        o_dbp=o_dbp if which == "dbpedia" else [],
        o_wd=o_wd if which == "wikidata" else [],
        max_hops=1,
        limit_edge=max_edges,
    )

    ev = []
    for p in paths:
        if not p:
            continue
        e = p[0]
        pred_name = e.predicate.split("/")[-1]
        if pred_name in _EXCLUDE:
            continue

        sentence = _edge_to_sentence(e)
        if not sentence:
            continue

        ev.append(
            {
                "snippet": sentence,
                "source": which,  # 'wikidata' | 'dbpedia'
                "trust": 1.0,
            }
        )
    return ev


async def collect_wikidata(triple: Triple) -> List[Dict]:
    return await _collect_kg(triple, "wikidata")


async def collect_dbpedia(triple: Triple) -> List[Dict]:
    return await _collect_kg(triple, "dbpedia")


async def collect_web(triple: Triple, top_k: int = 40) -> List[Dict]:
    wv = WebVerifier(num_results=top_k)
    query = f'"{triple.subject}" "{triple.predicate}" "{triple.object}"'
    results = wv._search(query)

    ev = []
    for r in results:
        snip = r.get("snippet") or ""
        if not snip:
            continue
        link = r.get("link") or ""
        ev.append(
            {
                "snippet": snip,
                "source": link,
                "trust": score_for_url(link),
            }
        )
    return ev


async def gather_evidence(claim: str) -> Dict[str, List[Dict]]:
    triple = parse_claim_to_triple(claim)
    if triple is None:
        return {"triple": None, "wikidata": [], "dbpedia": [], "web": []}

    wd, dbp, web = await asyncio.gather(
        collect_wikidata(triple), collect_dbpedia(triple), collect_web(triple)
    )
    return {"triple": triple, "wikidata": wd, "dbpedia": dbp, "web": web}
