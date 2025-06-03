# app/api/routes.py
from __future__ import annotations

from flask import Blueprint, request, jsonify
from http import HTTPStatus
from typing import List, Tuple  # Added Tuple for type hinting

from app.services.triple_extractor import parse_claim_to_triple
from app.services.kg_client import KGClient
from app.services.evidence_ranker import EvidenceRanker  # Ensure this is the updated class
from app.services.verifier import Verifier
from app.api import api_bp  # Assuming api_bp is defined in app.api.__init__.py or similar

from app.services.entity_linker import EntityLinker
from app.services.models import EntityCandidate, Edge, Triple  # Make sure models are imported


@api_bp.route("/verify", methods=["POST"])
def verify():  # → Tuple[dict, int]
    data = request.get_json(force=True)
    claim = data.get("claim")
    if not claim:
        return jsonify({"error": "JSON body must contain 'claim'"}), HTTPStatus.BAD_REQUEST

    # 1) Triple extraction
    extracted_triple: Triple | None = parse_claim_to_triple(claim)
    if extracted_triple is None:
        # If triple extraction fails, we can't proceed with KG-based verification
        # Consider a different Verifier fallback or a more specific error message
        return jsonify({
            "claim": claim,
            "triple": None,
            "evidence": [],
            "label": "Not Enough Info",
            "reason": "Could not extract a semantic triple from the claim.",
            "entity_linking": None
        }), HTTPStatus.OK  # Or UNPROCESSABLE_ENTITY

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
            "label": "Not Enough Info",
            "reason": error_message,
            "entity_linking": {
                "subject_candidates": [c.__dict__ for c in s_candidates],
                "object_candidates": [c.__dict__ for c in o_candidates]
            }
        }), HTTPStatus.OK

    # 3) Fetch paths from DBpedia and Wikidata
    kg = KGClient()
    paths: List[List[Edge]] = kg.fetch_paths(
        s_dbp_uris, s_wd_uris,
        o_dbp_uris, o_wd_uris,
        max_hops=2,
        try_public_dbp_fallback=True
    )

    verifier = Verifier()  # Initialize Verifier once

    if not paths:
        print(f"No paths found from KGs for claim: '{claim}'. Falling back to Verifier's direct classification.")
        label, reason = verifier.classify(claim, extracted_triple, [], 0.0)  # Pass empty evidence
        return jsonify({
            "claim": claim,
            "triple": extracted_triple.__dict__,
            "evidence": [],
            "label": label,
            "reason": reason,
            "entity_linking": {
                "subject_candidates": [c.__dict__ for c in s_candidates],
                "object_candidates": [c.__dict__ for c in o_candidates]
            }
        }), HTTPStatus.OK

    # 3.5) Rank the paths, considering the claim
    # Initialize EvidenceRanker with both claim_text and the extracted_triple
    ranker = EvidenceRanker(claim_text=claim, triple=extracted_triple)

    # Call top_k without the triple argument, as the ranker instance already has it.
    # ranked_paths_with_scores will be List[Tuple[List[Edge], float]]
    ranked_paths_with_scores: List[Tuple[List[Edge], float]] = ranker.top_k(paths, k=5)

    best_path_ranked: List[Edge] = []
    score: float = 0.0
    # This will store just the paths (List[List[Edge]]) for the "all_top_evidence_paths" field
    all_top_paths_for_json: List[List[Edge]] = []

    if ranked_paths_with_scores:
        # Unpack the best path and its score directly from the first element
        best_path_ranked, score = ranked_paths_with_scores[0]
        # Extract just the path component from each tuple for the JSON list
        all_top_paths_for_json = [path_with_score[0] for path_with_score in ranked_paths_with_scores]
    else:
        # If top_k returns empty, it means no paths were suitable or paths list was empty initially
        print(f"No suitable evidence paths found after ranking for claim: '{claim}'.")
        # Verification will proceed with empty best_path_ranked and score 0.0

    # 4) Verification step (GPT)
    # Verifier.classify takes the single best_path_ranked and its score
    label, reason = verifier.classify(claim, extracted_triple, best_path_ranked, score)

    return jsonify({
        "claim": claim,
        "triple": extracted_triple.__dict__,
        "evidence": [e.__dict__ for e in best_path_ranked],  # Evidence is the best ranked path
        # Correctly populate all_top_evidence_paths with lists of edges
        "all_top_evidence_paths": [[e.__dict__ for e in p] for p in all_top_paths_for_json],
        "label": label,
        "reason": reason,
        "entity_linking": {
            "subject_candidates": [c.__dict__ for c in s_candidates],
            "object_candidates": [c.__dict__ for c in o_candidates]
        }
    }), HTTPStatus.OK