"""
Async evidence collectors

FactGenius-style: Only local DBpedia graph, LLM predicate filtering, label-to-URI linking.
"""
from __future__ import annotations

import asyncio
import re
from typing import Dict, List
import ast, re
import os
import pickle

from ..extraction.triple_extractor import parse_claim_to_triple
from ..linking.entity_linker import EntityLinker
from ..verification.web_verifier import WebVerifier
from ...infrastructure.kg.local_dbpediagraph_client import LocalDBpediaGraphClient
from ...models import Triple
from .trust import score_for_url
from ...infrastructure.llm.llm_client import chat
from difflib import get_close_matches



DBPEDIA_PICKLE_PATH = os.path.join(os.path.dirname(__file__), "../../data/dbpedia_2015_undirected_light.pickle")
with open(DBPEDIA_PICKLE_PATH, "rb") as f:
    dbpedia_kg = pickle.load(f)
kg = LocalDBpediaGraphClient(dbpedia_kg)

label_to_uri = {uri.split("/")[-1].replace("_", " ").lower(): uri for uri in dbpedia_kg}

def link_label_to_uri(label: str) -> str | None:
    return label_to_uri.get(label.lower())

def get_all_predicates(entity_uri: str) -> list[str]:
    return list(dbpedia_kg.get(entity_uri, {}).keys())

def fetch_triples(entity: str, predicate: str) -> list[tuple]:
    return [(entity, predicate, obj) for obj in dbpedia_kg.get(entity, {}).get(predicate, [])]


def call_llm_filter_predicates(claim: str, entity_predicates: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Use an LLM to filter relevant predicates for each entity.
    entity_predicates: {entity_uri: [predicate1, predicate2, ...]}
    Returns: {entity_uri: [filtered_predicate1, ...]}
    """
    entity_string = '\n\n'.join([f'Entity: "{entity}" --> {", ".join(preds)}' for entity, preds in entity_predicates.items()])
    system = (
        "You are an intelligent graph connection finder. "
        "Given a claim and connection options for entities, "
        "filter the predicates that could be relevant to fact-check the claim. "
        "Output only a valid Python dict: {entity_uri: [relevant_predicate1, ...]}"
    )
    user = (
        f"Claim:\n{claim}\n\n"
        "Entities and predicates:\n"
        f"{entity_string}\n\n"
        "For each entity, select the most relevant predicates (from the options) that could help fact-check the claim. "
        "Output a Python dict: {entity_uri: [relevant_predicates]}"
    )
    reply = chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ])
    # Extract the dict from the LLM's reply
    content = reply.content  # Extract the string content
    data = ast.literal_eval(re.findall(r'\{.*?\}', content, re.DOTALL)[0])
    return data

def path_to_str(path):
    # path: [entity, rel1, entity2, rel2, entity3, ...]
    s = path[0].split("/")[-1].replace("_", " ")
    out = s
    for i in range(1, len(path), 2):
        rel = path[i].split("/")[-1]
        obj = path[i+1].split("/")[-1].replace("_", " ")
        out += f" --{rel}--> {obj}"
    return out

# ------------------------------------------------------------------ #
def fuzzy_link_label_to_uri(label: str) -> str | None:
    matches = get_close_matches(label.lower(), label_to_uri.keys(), n=1, cutoff=0.8)
    if matches:
        return label_to_uri[matches[0]]
    return None

def fuzzy_matchEntities(true_entities, predicted_entities, kg):
    """
    Map each true entity to the closest predicted entity (by label).
    Returns a dict: {true_entity: matched_predicted_entity}
    """
    mapping = {}
    pred_labels = {e: e.split("/")[-1].replace("_", " ").lower() for e in predicted_entities}
    for t in true_entities:
        t_label = t.split("/")[-1].replace("_", " ").lower()
        matches = get_close_matches(t_label, pred_labels.values(), n=1, cutoff=0.7)
        if matches:
            # Find the predicted entity with this label
            for pe, plabel in pred_labels.items():
                if plabel == matches[0]:
                    mapping[t] = pe
                    break
        else:
            mapping[t] = None
    return mapping

def validateRelation(resolved_entities, triple, kg):
    """
    For each resolved entity, get all predicates (relations) from the KG.
    Returns: {entity: [[predicate1], [predicate2], ...]}
    """
    rels = {}
    for t, pe in resolved_entities.items():
        if pe and pe in kg:
            preds = list(kg[pe].keys())
            rels[pe] = [[p] for p in preds]
    return rels

async def collect_local_dbpedia(triple: Triple, max_edges: int = 100) -> List[Dict]:
    # 1. Entity linking (FactGenius-style: label, then fuzzy, then fallback)
    predicted_entities = []
    for surface in [triple.subject, triple.object]:
        uri = link_label_to_uri(surface)
        if not uri:
            uri = fuzzy_link_label_to_uri(surface)
        if uri:
            predicted_entities.append(uri)
        else:
            # fallback to your existing linker if needed
            cands = EntityLinker().link(surface, top_k=1)
            if cands and cands[0].dbpedia_uri:
                predicted_entities.append(cands[0].dbpedia_uri)

    # 2. FactGenius-style entity/relation resolution
    true_entities = [triple.subject, triple.object]
    resolved_entities = fuzzy_matchEntities(true_entities, predicted_entities, dbpedia_kg)
    rels = validateRelation(resolved_entities, triple, dbpedia_kg)
    entities = sorted(rels.keys())

    # 3. Get all predicates for each entity (for LLM filtering)
    entity_predicates = {e: list(get_all_predicates(e)) for e in entities}

    # 4. LLM predicate filtering
    filtered_predicates = call_llm_filter_predicates(
        f"{triple.subject} {triple.predicate} {triple.object}", entity_predicates
    )

    # 5. Build rels dict for multi-hop search (FactGenius-style)
    rels = {e: [[p] for p in preds] for e, preds in filtered_predicates.items()}

    # 6. Multi-hop path search
    results = kg.search(entities, rels)
    evidence_strings = [path_to_str(p) for p in results["connected"] + results["walkable"]]

    # 7. Evidence retrieval (FactGenius-style)
    evidence = []
    # Direct evidence
    for entity, predicates in filtered_predicates.items():
        for pred in predicates:
            for s, p, o in fetch_triples(entity, pred):
                pred_name = p.split("/")[-1]
                sentence = f"{s.split('/')[-1].replace('_', ' ')} {pred_name} {o.split('/')[-1].replace('_', ' ')}."
                if not sentence:
                    continue
                evidence.append({
                    "snippet": sentence,
                    "source": "dbpedia",
                    "trust": 0.2,
                    "score": 1.0
                })
                if len(evidence) >= max_edges:
                    return evidence
    # Multi-hop evidence
    for s in evidence_strings:
        evidence.append({
            "snippet": s,
            "source": "dbpedia-multihop",
            "trust": 0.2,
            "score": 1.0
        })
        if len(evidence) >= max_edges:
            return evidence
    return evidence


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
        return {"triple": None, "dbpedia": [], "web": []}

    dbp, web = await asyncio.gather(
        collect_local_dbpedia(triple), collect_web(triple)
    )
    return {"triple": triple, "dbpedia": dbp}
    #return {"triple": triple, "dbpedia": dbp, "web": web}
