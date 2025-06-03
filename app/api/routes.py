from __future__ import annotations

from flask import Blueprint, request, jsonify
from http import HTTPStatus
from typing import List

from app.services.triple_extractor import parse_claim_to_triple
from app.services.kg_client import KGClient
from app.services.evidence_ranker import EvidenceRanker
from app.services.verifier import Verifier
from app.api import api_bp

from app.services.entity_linker import EntityLinker
from app.services.models import EntityCandidate, Edge, Triple

@api_bp.route("/verify", methods=["POST"])
def verify():  # → Tuple[dict, int]
    data = request.get_json(force=True)
    claim = data.get("claim")
    if not claim:
        return jsonify({"error": "JSON body must contain 'claim'"}), HTTPStatus.BAD_REQUEST

    # 1) Triple extraction
    extracted_triple: Triple | None = parse_claim_to_triple(claim)
    if extracted_triple is None:
        return jsonify({"error": "Could not extract a semantic triple from the claim."}), HTTPStatus.UNPROCESSABLE_ENTITY

    # 2) Entity linking -> Get DBpedia and Wikidata URI candidates
    linker = EntityLinker()
    s_candidates: List[EntityCandidate] = linker.link(extracted_triple.subject, top_k=3)
    o_candidates: List[EntityCandidate] = linker.link(extracted_triple.object, top_k=3)

    s_dbp_uris: List[str] = [c.dbpedia_uri for c in s_candidates if c.dbpedia_uri]
    s_wd_uris: List[str] = [c.wikidata_uri for c in s_candidates if c.wikidata_uri]
    
    o_dbp_uris: List[str] = [c.dbpedia_uri for c in o_candidates if c.dbpedia_uri]
    o_wd_uris: List[str] = [c.wikidata_uri for c in o_candidates if c.wikidata_uri]

    if not (s_dbp_uris or s_wd_uris) or not (o_dbp_uris or o_wd_uris):
        error_message = "Could not link entities to knowledge base URIs for the claim."
        if not (s_dbp_uris or s_wd_uris):
            error_message = f"Could not link subject '{extracted_triple.subject}' to any knowledge base URI."
        elif not (o_dbp_uris or o_wd_uris):
            error_message = f"Could not link object '{extracted_triple.object}' to any knowledge base URI."
        
        return jsonify({
            "claim": claim,
            "triple": extracted_triple.__dict__,
            "evidence": [],
            "label": "Not Enough Info", # Default to NEI if linking fails substantially
            "reason": error_message,
            "entity_linking": {
                "subject_candidates": [c.__dict__ for c in s_candidates],
                "object_candidates": [c.__dict__ for c in o_candidates]
            }
        }), HTTPStatus.OK

    # 3) Fetch paths from DBpedia and Wikidata
    # KGClient can be initialized with a specific dbp_endpoint if needed: kg = KGClient(dbp_endpoint="http://my-local-dbpedia:8890/sparql")
    kg = KGClient() 
    paths: List[List[Edge]] = kg.fetch_paths(
        s_dbp_uris, s_wd_uris, 
        o_dbp_uris, o_wd_uris,
        max_hops=2, # Default is 2, can be configured in KGClient
        try_public_dbp_fallback=True # Enable public DBpedia fallback if primary DBP endpoint fails/returns no DBP paths
    )

    # Initialize Verifier once
    verifier = Verifier()

    # LLM fallback if no evidence paths are found from any KG
    if not paths:
        print(f"No paths found from KGs for claim: '{claim}'. Falling back to Verifier's direct classification.")
        # Assuming Verifier.classify can handle empty evidence gracefully or you have a specific llm_fallback_classify
        # For consistency, using classify with empty evidence:
        label, reason = verifier.classify(claim, extracted_triple, [], 0.0)
        return jsonify({
            "claim": claim,
            "triple":   extracted_triple.__dict__,
            "evidence": [],
            "label":    label,
            "reason":   reason,
            "entity_linking": {
                "subject_candidates": [c.__dict__ for c in s_candidates],
                "object_candidates": [c.__dict__ for c in o_candidates]
            }
        }), HTTPStatus.OK

    # 3.5) Rank the paths, considering the claim
    # Ensure EvidenceRanker's __init__ matches this call. Original was EvidenceRanker(claim_text: str)
    ranker = EvidenceRanker(claim_text=claim) 
    # Increased k for more evidence options, can be adjusted (e.g., to 3 as in main)
    evidence_paths: List[List[Edge]] = ranker.top_k(extracted_triple, paths, k=5) 

    best_path_ranked: List[Edge] = []
    score: float = 0.0

    if evidence_paths:
        best_path_ranked = evidence_paths[0]
        # Assuming ranker._score_path is the method to get a single score for the best path
        score = ranker._score_path(extracted_triple, best_path_ranked) 
    else:
        # If top_k returns empty, it means no paths were suitable or paths list was empty initially
        print(f"No suitable evidence paths found after ranking for claim: '{claim}'.")
        # Verification will proceed with empty evidence if best_path_ranked is empty.

    # 4) Verification step (GPT)
    label, reason = verifier.classify(claim, extracted_triple, best_path_ranked, score)

    return jsonify({
        "claim": claim,
        "triple": extracted_triple.__dict__,
        "evidence": [e.__dict__ for e in best_path_ranked],
        "all_top_evidence_paths": [[e.__dict__ for e in p] for p in evidence_paths],
        "label": label,
        "reason": reason,
        "entity_linking": {
            "subject_candidates": [c.__dict__ for c in s_candidates],
            "object_candidates": [c.__dict__ for c in o_candidates]
        }
    }), HTTPStatus.OK