# app/api/routes.py
from __future__ import annotations

from flask import Blueprint, request, jsonify
from http import HTTPStatus
from typing import List # For type hinting

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
    extracted_triple: Triple | None = parse_claim_to_triple(claim) # Type hint
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

        error_message = "Could not link entities to knowledge base URIs."
        if not (s_dbp_uris or s_wd_uris):
            error_message = f"Could not link subject '{extracted_triple.subject}' to any knowledge base URI."
        elif not (o_dbp_uris or o_wd_uris):
             error_message = f"Could not link object '{extracted_triple.object}' to any knowledge base URI."
        
        # Default to NEI if linking fails substantially
        return jsonify({
            "triple": extracted_triple.__dict__,
            "evidence": [],
            "label": "Not Enough Info",
            "reason": error_message,
        }), HTTPStatus.OK # Or UNPROCESSABLE_ENTITY if preferred for this case


    # 3) Fetch paths from DBpedia and Wikidata
    kg = KGClient()

    paths: List[List[Edge]] = kg.fetch_paths(
        s_dbp_uris, s_wd_uris, 
        o_dbp_uris, o_wd_uris,
        max_hops=2 # Default is 2, can be configured
    )

    # 3.5) Rank the paths, considering the claim
    ranker = EvidenceRanker(claim_text=claim) # Pass claim_text as named arg
    # top_k for evidence paths fed to verifier; original code used k=3
    evidence_paths: List[List[Edge]] = ranker.top_k(extracted_triple, paths, k=5) # Increased k for more evidence options

    # Select the best path for current response structure, or adapt verifier for multiple paths
    best_path_ranked: List[Edge] = evidence_paths[0] if evidence_paths else []
    score: float = ranker._score_path(extracted_triple, best_path_ranked) if best_path_ranked else 0.0


    # 4) Verification step (GPT)
    verifier = Verifier()
    # The verifier takes the single 'best_path'. If multiple paths are desired for verification, Verifier needs adjustment.
    label, reason = verifier.classify(claim, extracted_triple, best_path_ranked, score)

    return jsonify({
        "claim": claim,
        "triple": extracted_triple.__dict__,
        "evidence": [e.__dict__ for e in best_path_ranked], # Evidence is the best path
        "all_top_evidence_paths": [[e.__dict__ for e in p] for p in evidence_paths],
        "label": label,
        "reason": reason,
        "entity_linking": {
            "subject_candidates": [c.__dict__ for c in s_candidates],
            "object_candidates": [c.__dict__ for c in o_candidates]
        }
    }), HTTPStatus.OK